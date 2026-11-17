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

"""Serve simulator envs to EnvWorkers on other machines.

    python -m rlinf.envs.remote.env_server --port 18991 --gpus 0,1

A session is one :class:`~rlinf.envs.remote.env_client.RemoteEnv`.
HeteroPulse (:func:`heteropulse.session.serve`) gives each session a process
of its own on the least busy GPU; :class:`EnvHandler` is what that process
serves: it builds the env as EnvWorker would and answers the calls a remote env
makes. The process exits, freeing the GPU, when its client goes away.

Both ends must run the same ``rlinf/envs`` code, which the session handshake
checks, and the server needs the simulator and its assets installed.

The server has no authentication. Bind it to a network only the training
machines can reach.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import signal
import threading
import time
import types
from typing import Any, Callable, Optional

from heteropulse import devices, server, session

from rlinf.envs.remote.env_client import env_app, envs_code_hash
from rlinf.utils.logging import get_logger

DEFAULT_PORT = 18991


def build_env(env_type, cfg, num_envs, seed_offset, total_num_processes, worker_info):
    """Build the env a session serves, as EnvWorker builds a local one."""
    from rlinf.envs import get_env_cls

    env_cls = get_env_cls(env_type, cfg)
    return env_cls(
        cfg=cfg,
        num_envs=num_envs,
        seed_offset=seed_offset,
        total_num_processes=total_num_processes,
        worker_info=worker_info,
    )


def serve(
    host: str,
    port: int,
    slots: list[devices.DeviceSlot],
    env_factory: str = "rlinf.envs.remote.env_server:build_env",
    max_sessions_per_gpu: Optional[int] = None,
    on_ready: Optional[Callable[[int], None]] = None,
    stop: Optional[threading.Event] = None,
) -> None:
    """Serve remote envs until ``stop`` is set or the process is interrupted.

    Args:
        host: Address to bind.
        port: Port to bind; 0 picks a free one.
        slots: Devices sessions are placed on.
        env_factory: ``module:function`` that builds a session's env; it takes
            :func:`build_env`'s arguments.
        max_sessions_per_gpu: Refuse new sessions once every slot holds this
            many; ``None`` for no limit.
        on_ready: Called with the bound port once the server listens.
        stop: Set it to make the server shut down.
    """
    get_logger().info(f"[env server] rlinf/envs {envs_code_hash()}")
    session.serve(
        host,
        port,
        slots,
        "rlinf.envs.remote.env_server:EnvHandler",
        env_app(),
        handler_kwargs={"env_factory": env_factory},
        max_sessions_per_slot=max_sessions_per_gpu,
        on_ready=on_ready,
        stop=stop,
    )


class EnvHandler:
    """The env one session serves, behind the requests a remote env sends.

    Built in the session's process by HeteroPulse.

    Args:
        open_request: The client's request to open the session.
        env_factory: ``module:function`` that builds the env.
    """

    def __init__(self, open_request: dict, env_factory: str) -> None:
        module_name, _, attr = env_factory.partition(":")
        self._factory = getattr(importlib.import_module(module_name), attr)
        self.env = None

    def handle(self, request: dict) -> dict:
        """Answer one request from a remote env."""
        op = request.get("op")
        if op == "make":
            return self._make(request)
        if self.env is None:
            raise RuntimeError(f"{op!r} before the env was made")
        if op == "reset":
            obs, infos = self.env.reset(**request.get("kwargs", {}))
            return {"obs": _to_cpu(obs), "infos": _to_cpu(infos), **self._is_start()}
        if op == "chunk_step":
            obs, rewards, terminations, truncations, infos = self.env.chunk_step(
                request["actions"]
            )
            return self._step_reply(
                _last(obs), rewards, terminations, truncations, _last(infos)
            )
        if op == "step":
            return self._step_reply(
                *self.env.step(request["actions"], auto_reset=request["auto_reset"])
            )
        if op == "update_reset_state_ids":
            self.env.update_reset_state_ids()
            return {}
        if op == "set_is_start":
            self.env.is_start = bool(request["value"])
            return {}
        raise ValueError(f"unknown request {op!r}")

    def close(self) -> None:
        """Close the served env, if one was made."""
        if self.env is None or not hasattr(self.env, "close"):
            return
        try:
            self.env.close()
        except Exception as exc:
            get_logger().warning(f"[env server] closing the env failed: {exc}")
        self.env = None

    def _make(self, request: dict) -> dict:
        import torch
        from omegaconf import OmegaConf

        from rlinf.utils.utils import seed_everything

        if self.env is not None:
            raise RuntimeError("this session already serves an env")
        # Simulators draw from the process RNGs as they build and reset.
        seed_everything(request["process_seed"])
        if request["deterministic"]:
            torch.use_deterministic_algorithms(True, warn_only=True)
        cfg = OmegaConf.create(request["cfg"])
        # Hydra hands EnvWorker a struct config; envs may depend on that.
        OmegaConf.set_struct(cfg, True)
        started = time.perf_counter()
        self.env = self._factory(
            env_type=request["env_type"],
            cfg=cfg,
            num_envs=request["num_envs"],
            seed_offset=request["seed_offset"],
            total_num_processes=request["total_num_processes"],
            worker_info=types.SimpleNamespace(**request.get("worker_info", {})),
        )
        get_logger().info(
            f"[env server] built {type(self.env).__name__} ({request['env_type']}, "
            f"{request['num_envs']} envs) in {time.perf_counter() - started:.1f} s"
        )
        seed = getattr(self.env, "seed", None)
        return {
            "num_envs": int(getattr(self.env, "num_envs", request["num_envs"])),
            "seed": seed if isinstance(seed, (int, float, str)) else None,
            **self._is_start(),
        }

    def _step_reply(self, obs, rewards, terminations, truncations, infos) -> dict:
        return {
            "obs": _to_cpu(obs),
            "rewards": _to_cpu(rewards),
            "terminations": _to_cpu(terminations),
            "truncations": _to_cpu(truncations),
            "infos": _to_cpu(infos),
            **self._is_start(),
        }

    def _is_start(self) -> dict:
        value = getattr(self.env, "is_start", None)
        return {"is_start": value} if isinstance(value, bool) else {}


def _last(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[-1] if value else None
    return value


def _to_cpu(value: Any) -> Any:
    """Copy every tensor in ``value`` to the CPU; the wire carries only those."""
    if type(value).__module__.startswith("torch"):
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _raise_interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def main(argv: Optional[list[str]] = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Serve RLinf simulator envs to EnvWorkers on other machines."
    )
    parser.add_argument("--host", default="0.0.0.0", help="address to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--gpus",
        help="comma-separated CUDA device ids sessions are spread over; without "
        "it, sessions keep the GPU and MUJOCO_GL this process has",
    )
    parser.add_argument(
        "--max-sessions-per-gpu",
        type=int,
        help="refuse new sessions once every GPU holds this many",
    )
    args = parser.parse_args(argv)
    server.configure_logging(logging.INFO)
    if args.gpus:
        slots = devices.gpu_slots([int(gpu) for gpu in args.gpus.split(",")])
    else:
        slots = devices.inherited_slots(1)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        serve(
            args.host, args.port, slots, max_sessions_per_gpu=args.max_sessions_per_gpu
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
