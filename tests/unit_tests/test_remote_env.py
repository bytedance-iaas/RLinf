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

"""Remote simulation through HeteroPulse: config checks and remote envs.

Remote envs are served from a deterministic stand-in simulator, so those tests
compare the proxy's results with the same simulator run in this process. The
wire layer, the session server and remote rendering are HeteroPulse's and are
tested there.
"""

import functools
import multiprocessing
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("heteropulse")

from heteropulse import transport  # noqa: E402


def _assert_same(actual, expected, path="value"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_same(actual[key], expected[key], f"{path}[{key!r}]")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_same(a, e, f"{path}[{i}]")
    elif isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        assert torch.equal(actual, expected), path
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    else:
        assert actual == expected, path


def test_remote_render_config_is_limited_to_robosuite_simulators():
    from rlinf.config import validate_remote_render_env_cfg

    def cfg(env_type, address):
        return OmegaConf.create(
            {
                "env": {
                    "train": {
                        "env_type": env_type,
                        "remote_render": {"address": address},
                    }
                }
            }
        )

    validate_remote_render_env_cfg(cfg("libero", "gpu-host:18990"))
    validate_remote_render_env_cfg(cfg("maniskill", None))
    with pytest.raises(AssertionError, match="supports env_type"):
        validate_remote_render_env_cfg(cfg("maniskill", "gpu-host:18990"))
    with pytest.raises(ValueError, match="host:port"):
        validate_remote_render_env_cfg(cfg("libero", "gpu-host"))


def test_remote_env_config_names_a_supported_simulator_and_no_per_step_feature():
    from rlinf.config import validate_remote_env_cfg

    def cfg(env_type="libero", **train):
        train = {"env_type": env_type, "remote": {"address": "gpu-host:18991"}, **train}
        return OmegaConf.create({"env": {"train": train}})

    validate_remote_env_cfg(cfg())
    validate_remote_env_cfg(cfg("maniskill"))
    with pytest.raises(AssertionError, match="supports env_type"):
        validate_remote_env_cfg(cfg("behavior"))
    with pytest.raises(
        AssertionError, match=r"cannot run with \['video_cfg.save_video'\]"
    ):
        validate_remote_env_cfg(cfg(video_cfg={"save_video": True}))
    with pytest.raises(AssertionError, match="exclusive"):
        validate_remote_env_cfg(cfg(remote_render={"address": "gpu-host:18990"}))


# --- remote envs --------------------------------------------------------------


class FakeEnv:
    """A deterministic simulator: observations follow the seed and the actions."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.num_envs = num_envs
        self.seed = int(cfg.seed) + seed_offset
        self.is_start = True
        self._rank = getattr(worker_info, "rank", None)
        self._generator = torch.Generator().manual_seed(self.seed)
        self._state = torch.zeros(num_envs, 3)
        self._resets = 0

    def _obs(self):
        level = (self._state.sum(dim=1).abs() * 40 % 256).to(torch.uint8)
        return {
            # 1.5 MiB for 8 envs, which a remote env stripes across its streams.
            "main_images": level[:, None, None, None].expand(-1, 256, 256, 3).clone(),
            "states": self._state.clone(),
            "states_bf16": self._state.to(torch.bfloat16),
            "task_descriptions": [f"task {i}" for i in range(self.num_envs)],
        }

    def reset(self, probe_rng=False, **kwargs):
        self._state = torch.randn(self.num_envs, 3, generator=self._generator)
        self._resets += 1
        was_start, self.is_start = self.is_start, False
        infos = {"resets": self._resets, "was_start": was_start, "rank": self._rank}
        if probe_rng:
            # What a simulator drawing from its process RNGs at reset sees.
            infos["draw"] = torch.rand(()).item()
            infos["deterministic"] = torch.are_deterministic_algorithms_enabled()
        return self._obs(), infos

    def step(self, actions, auto_reset=True):
        actions = torch.as_tensor(actions)
        if (actions == -999).any():
            raise ValueError("poisoned action")
        self._state = self._state + actions
        reward = self._state.sum(dim=1)
        terminated = reward > 3
        return (
            self._obs(),
            reward,
            terminated,
            torch.zeros_like(terminated),
            {"reward": reward.clone()},
        )

    def chunk_step(self, chunk_actions):
        steps = [
            self.step(chunk_actions[:, i], auto_reset=False)
            for i in range(chunk_actions.shape[1])
        ]
        obs, rewards, terminations, truncations, infos = zip(*steps)
        return (
            list(obs),
            torch.stack(rewards, dim=1),
            torch.stack(terminations, dim=1),
            torch.stack(truncations, dim=1),
            list(infos),
        )

    def update_reset_state_ids(self):
        self._resets += 100


def fake_env_factory(
    env_type, cfg, num_envs, seed_offset, total_num_processes, worker_info
):
    return FakeEnv(cfg, num_envs, seed_offset, total_num_processes, worker_info)


@pytest.fixture(scope="module")
def env_server():
    from heteropulse.devices import inherited_slots

    from rlinf.envs.remote import env_server

    ready: queue.Queue = queue.Queue()
    stop = threading.Event()
    thread = threading.Thread(
        target=env_server.serve,
        kwargs={
            "host": "127.0.0.1",
            "port": 0,
            "slots": inherited_slots(1),
            "env_factory": "test_remote_env:fake_env_factory",
            "on_ready": ready.put,
            "stop": stop,
        },
        daemon=True,
    )
    thread.start()
    yield f"127.0.0.1:{ready.get(timeout=60)}"
    stop.set()
    thread.join(timeout=60)


def _remote_env_cfg(address, streams):
    return OmegaConf.create(
        {
            "env_type": "maniskill",
            "seed": 7,
            "remote": {
                "address": address,
                "streams": streams,
                "request_timeout_s": 120,
            },
        }
    )


def _make_pair(address, streams=4):
    from rlinf.envs.remote.env_client import RemoteEnv

    cfg = _remote_env_cfg(address, streams)
    worker_info = SimpleNamespace(rank=3, group_world_size=2, accelerator=object())
    local = FakeEnv(cfg, 8, 1, 2, worker_info)
    remote = RemoteEnv(
        cfg, num_envs=8, seed_offset=1, total_num_processes=2, worker_info=worker_info
    )
    return local, remote


def test_remote_env_returns_what_the_env_returns_with_tails_of_chunks(env_server):
    local, remote = _make_pair(env_server)
    assert (remote.num_envs, remote.seed, remote.is_start) == (8, local.seed, True)
    _assert_same(remote.reset(), local.reset())
    assert remote.is_start is False
    actions = torch.arange(8 * 5 * 3, dtype=torch.float32).reshape(8, 5, 3) / 100
    obs, rewards, terminations, truncations, infos = remote.chunk_step(actions)
    local_obs, *local_tensors, local_infos = local.chunk_step(actions)
    assert len(obs) == len(infos) == 1
    _assert_same(obs[0], local_obs[-1])
    _assert_same(infos[0], local_infos[-1])
    _assert_same([rewards, terminations, truncations], local_tensors)
    remote.is_start = local.is_start = True
    remote.update_reset_state_ids()
    local.update_reset_state_ids()
    _assert_same(remote.reset(), local.reset())
    _assert_same(remote.step(actions[:, 0]), local.step(actions[:, 0]))
    times = remote.transport_times()
    assert times["remote_round_trip"] > times["remote_server"] > 0
    assert remote.transport_times()["remote_round_trip"] == 0.0
    remote.close()


def test_remote_env_raises_the_server_side_exception_and_keeps_serving(env_server):
    _, remote = _make_pair(env_server, streams=0)
    remote.reset()
    with pytest.raises(
        transport.RemoteError, match="ValueError: poisoned action"
    ) as info:
        remote.chunk_step(torch.full((8, 1, 3), -999.0))
    assert "Traceback" in info.value.remote_traceback
    remote.chunk_step(torch.zeros(8, 1, 3))
    remote.close()


def test_remote_env_seeds_each_served_env_from_the_client_rng(env_server):
    from rlinf.envs.remote.env_client import RemoteEnv

    cfg = _remote_env_cfg(env_server, streams=0)

    def probe(seed):
        torch.manual_seed(seed)
        envs = [
            RemoteEnv(
                cfg, num_envs=2, seed_offset=i, total_num_processes=2, worker_info=None
            )
            for i in range(2)
        ]
        infos = [env.reset(probe_rng=True)[1] for env in envs]
        for env in envs:
            env.close()
        return [info["draw"] for info in infos], infos[0]["deterministic"]

    draws, deterministic = probe(5)
    assert probe(5)[0] == draws, "a seeded client must repeat its served envs' draws"
    assert draws[0] != draws[1], "each session must draw a stream of its own"
    assert deterministic is False
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        assert probe(5)[1] is True
    finally:
        torch.use_deterministic_algorithms(False)


def test_env_server_refuses_a_client_running_other_env_code(env_server):
    from heteropulse import session

    from rlinf.envs.remote.env_client import ENV_APP_NAME, ENV_PROTOCOL

    other_code = session.App(ENV_APP_NAME, ENV_PROTOCOL, "0" * 16)
    with pytest.raises(transport.RemoteError, match="rlinf-env code differs"):
        session.open_session(env_server, other_code)


def test_closing_a_remote_env_ends_its_server_process(env_server):
    _, remote = _make_pair(env_server, streams=2)
    pid = remote.server["pid"]
    assert pid in {p.pid for p in multiprocessing.active_children()}
    remote.close()
    deadline = time.monotonic() + 30
    while pid in {p.pid for p in multiprocessing.active_children()}:
        assert time.monotonic() < deadline, "the session process outlived its client"
        time.sleep(0.1)


def test_env_worker_builds_remote_splits_from_the_proxy():
    from rlinf.envs.remote.env_client import RemoteEnv
    from rlinf.workers.env.env_worker import EnvWorker

    remote_cfg = OmegaConf.create(
        {"env_type": "maniskill", "remote": {"address": "h:1"}}
    )
    env_cls = EnvWorker._env_cls(remote_cfg, "eval")
    assert isinstance(env_cls, functools.partial) and env_cls.func is RemoteEnv
    assert env_cls.keywords == {"split": "eval"}


def test_env_worker_logs_evaluation_remote_times_under_their_own_keys():
    from rlinf.workers.env.env_worker import EnvWorker

    class Proxy:
        def transport_times(self):
            return {"remote_round_trip": 1.5, "remote_server": 1.0}

    worker = SimpleNamespace(_timer_metrics={})
    EnvWorker._record_transport_times(worker, Proxy())
    EnvWorker._record_transport_times(worker, Proxy(), prefix="eval_")
    assert worker._timer_metrics == {
        "env/remote_round_trip": 1.5,
        "env/remote_server": 1.0,
        "env/eval_remote_round_trip": 1.5,
        "env/eval_remote_server": 1.0,
    }
