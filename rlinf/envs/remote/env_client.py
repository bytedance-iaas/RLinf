# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Drive a simulator that runs on another machine.

Setting ``env.<split>.remote.address`` makes EnvWorker build :class:`RemoteEnv`
where it would build the simulator's own env class. The proxy opens a
HeteroPulse session on an :mod:`~rlinf.envs.remote.env_server`, which builds
the real env with ``get_env_cls`` on one of its GPUs and serves it until the
proxy disconnects.

The proxy implements the calls EnvWorker makes and nothing else, so the
optional features EnvWorker probes for with ``get_env_attr`` read as absent
instead of as remote calls. A chunk step returns its last observation and info
only: EnvWorker reads no others unless an external reward model or online
LeRobot collection is on, and ``validate_cfg`` rejects both with a remote env.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from heteropulse import session
from omegaconf import OmegaConf

#: Name of the request vocabulary between remote envs and env servers.
ENV_APP_NAME = "rlinf-env"
#: Its version.
ENV_PROTOCOL = 1


@functools.cache
def envs_code_hash() -> str:
    """Hash every Python file under ``rlinf/envs``, the code both ends run."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def env_app() -> session.App:
    """What a remote env and an env server agree on: vocabulary and code."""
    return session.App(ENV_APP_NAME, ENV_PROTOCOL, envs_code_hash())


@dataclass(frozen=True)
class RemoteEnvOptions:
    """How a remote env reaches its server, from ``env.<split>.remote``.

    Attributes:
        address: The env server, as ``host:port``.
        streams: Data connections that carry large replies in parallel.
        connect_timeout_s: How long to wait for each connection.
        request_timeout_s: How long one call may take, building the env
            included.
        require_same_code: Refuse a server whose ``rlinf/envs`` differs.
    """

    address: str
    streams: int = 16
    connect_timeout_s: float = 60.0
    request_timeout_s: float = 900.0
    require_same_code: bool = True

    @classmethod
    def from_env_cfg(cls, env_cfg) -> "RemoteEnvOptions":
        """Read the options of one env split."""
        remote = env_cfg.remote
        defaults = cls(address="")
        return cls(
            address=str(remote.address),
            streams=int(remote.get("streams", defaults.streams)),
            connect_timeout_s=float(
                remote.get("connect_timeout_s", defaults.connect_timeout_s)
            ),
            request_timeout_s=float(
                remote.get("request_timeout_s", defaults.request_timeout_s)
            ),
            require_same_code=bool(
                remote.get("require_same_code", defaults.require_same_code)
            ),
        )


class RemoteEnv:
    """Proxy for a simulator env that runs on an env server.

    Built with the arguments EnvWorker passes to an env class; the server
    builds the env of ``cfg.env_type`` with the same ones. ``split`` names the
    env split the proxy serves: the server spreads the sessions of one split
    over its GPUs before it puts two on one GPU.

    Attributes:
        num_envs: Parallel envs the server built.
        seed: The served env's seed, if it has one.
        server: What the server reported when the session opened.
    """

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        split: str = "train",
    ):
        options = RemoteEnvOptions.from_env_cfg(cfg)
        self._session = session.open_session(
            options.address,
            env_app(),
            streams=options.streams,
            group=split,
            require_same_code=options.require_same_code,
            connect_timeout_s=options.connect_timeout_s,
            request_timeout_s=options.request_timeout_s,
        )
        self.server = self._session.server
        self._round_trip_s = 0.0
        self._server_s = 0.0
        # Mirrors the served env's value whenever a reply carries it.
        self._is_start = True
        reply = self._call(
            "make",
            env_type=str(cfg.env_type),
            cfg=OmegaConf.to_container(cfg, resolve=True),
            num_envs=int(num_envs),
            seed_offset=int(seed_offset),
            total_num_processes=int(total_num_processes),
            worker_info=_scalar_fields(worker_info),
            **_process_rng(),
        )
        self.num_envs = reply["num_envs"]
        self.seed = reply["seed"]

    @property
    def is_start(self) -> bool:
        """The served env's ``is_start``, as of the last call that returned it."""
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._call("set_is_start", value=bool(value))
        self._is_start = bool(value)

    def reset(self, **kwargs) -> tuple[Any, Any]:
        """Reset the served env; returns ``(obs, infos)``."""
        reply = self._call("reset", kwargs=kwargs)
        return reply["obs"], reply["infos"]

    def step(self, actions, auto_reset: bool = True) -> tuple:
        """Step the served env once; returns what its ``step`` returns."""
        reply = self._call(
            "step", timed=True, actions=actions, auto_reset=bool(auto_reset)
        )
        return (
            reply["obs"],
            reply["rewards"],
            reply["terminations"],
            reply["truncations"],
            reply["infos"],
        )

    def chunk_step(self, chunk_actions) -> tuple:
        """Run one action chunk on the served env.

        Returns:
            ``(obs_list, rewards, terminations, truncations, infos_list)`` as
            the env's own ``chunk_step`` does, except that the two lists hold
            only the chunk's last observation and info.
        """
        reply = self._call("chunk_step", timed=True, actions=chunk_actions)
        return (
            [reply["obs"]],
            reply["rewards"],
            reply["terminations"],
            reply["truncations"],
            [reply["infos"]],
        )

    def update_reset_state_ids(self) -> None:
        """Advance the served env's reset states."""
        self._call("update_reset_state_ids")

    def transport_times(self) -> dict[str, float]:
        """Seconds spent in step calls since the last call.

        Only ``step`` and ``chunk_step`` count, the calls EnvWorker times as
        ``env_interact_step``. ``remote_transfer`` is the round trip minus
        the server's work: the network and both ends' encoding.
        """
        round_trip, server = self._round_trip_s, self._server_s
        self._round_trip_s = self._server_s = 0.0
        return {
            "remote_round_trip": round_trip,
            "remote_server": server,
            "remote_transfer": round_trip - server,
        }

    def close(self) -> None:
        """End the session, which shuts the served env down. Idempotent."""
        if self._session is None:
            return
        self._session.close()
        self._session = None

    def _call(self, op: str, timed: bool = False, **args) -> dict:
        if self._session is None:
            raise RuntimeError("this remote env is closed")
        reply = self._session.call(op, **args)
        if timed:
            self._round_trip_s += self._session.round_trip_s
            self._server_s += self._session.server_s
        if "is_start" in reply:
            self._is_start = reply.pop("is_start")
        return reply


def _process_rng() -> dict[str, Any]:
    """How the server seeds the process that builds the env.

    A local env draws from EnvWorker's process RNGs, which the worker seeds
    before building it; a served env draws from its own process's. Each session
    gets a seed drawn from this process's torch RNG, so a seeded run repeats
    and every session draws a stream of its own.
    """
    import torch

    return {
        "process_seed": int(torch.randint(0, 2**62, ())),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }


def _scalar_fields(worker_info) -> dict[str, Any]:
    """The plain fields of a ``WorkerInfo``, which simulators may read."""
    fields = getattr(worker_info, "__dict__", None) or {}
    return {
        key: value
        for key, value in fields.items()
        if value is None or isinstance(value, (bool, int, float, str))
    }
