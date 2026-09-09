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

"""Attaching the media index must not assume a training config exists.

A standalone evaluation carries ``env.eval`` and nothing else -- see
``examples/embodiment/config/so101_eval_openpi_pi05.yaml``, whose ``defaults``
merge a task only into ``env.eval``. Every other ``env.train`` read in the
worker is already guarded by ``enable_train``, so the split is deliberate; the
media index read it unconditionally and took the whole run down with
``ConfigAttributeError: Missing key train`` before the first episode.

The mirror case matters too: an eval-only run that *does* keep an ``env.train``
block around (``realworld_eval_dual_franka.yaml`` does) never populates
``env_list``, so a train video flag there describes environments that were
never built.

``EnvWorker`` is driven through ``__new__`` with only the attributes
``_attach_media_index`` touches -- a real one would need a live simulator.
"""

import sys
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

# EnvWorker pulls in the gym wrapper stack at import time; neither is needed to
# exercise the config reads under test.
if "gymnasium" not in sys.modules:
    sys.modules["gymnasium"] = MagicMock()
if "rlinf.envs.wrappers" not in sys.modules:
    sys.modules["rlinf.envs.wrappers"] = MagicMock()

from rlinf.workers.env.env_worker import EnvWorker  # noqa: E402


class _FakeEnv:
    """Only ``set_media_index``, recording what the worker handed it."""

    def __init__(self):
        self.attached = []

    def set_media_index(self, media_index, shard):
        self.attached.append((media_index, shard))


def _worker(cfg, *, enable_train, enable_eval, env_list, eval_env_list):
    worker = object.__new__(EnvWorker)
    worker.cfg = cfg
    worker._rank = 0
    worker.enable_train = enable_train
    worker.enable_eval = enable_eval
    worker.env_list = env_list
    worker.eval_env_list = eval_env_list
    return worker


def _cfg(tmp_path, *, train_video=None, eval_video=None):
    env = {}
    if train_video is not None:
        env["train"] = {"video_cfg": {"save_video": train_video}}
    if eval_video is not None:
        env["eval"] = {"video_cfg": {"save_video": eval_video}}
    return OmegaConf.create(
        {"runner": {"logger": {"log_path": str(tmp_path)}}, "env": env}
    )


def test_eval_only_run_without_a_train_config(tmp_path):
    """The regression: no ``env.train`` key at all, eval recording enabled."""
    eval_env = _FakeEnv()
    worker = _worker(
        _cfg(tmp_path, eval_video=True),
        enable_train=False,
        enable_eval=True,
        env_list=[],
        eval_env_list=[eval_env],
    )

    worker._attach_media_index()

    assert len(eval_env.attached) == 1
    _, shard = eval_env.attached[0]
    assert shard == 0


def test_train_video_flag_is_ignored_when_training_is_disabled(tmp_path):
    """An eval-only run that keeps a dormant ``env.train`` builds no train index."""
    eval_env = _FakeEnv()
    worker = _worker(
        _cfg(tmp_path, train_video=True, eval_video=False),
        enable_train=False,
        enable_eval=True,
        env_list=[],
        eval_env_list=[eval_env],
    )

    worker._attach_media_index()

    assert eval_env.attached == []
    # No shard directory either: an index built here would only ever describe
    # environments that ``init_worker`` skipped.
    assert not (tmp_path / "_rlinf").exists()


def test_a_training_run_still_gets_both_indexes(tmp_path):
    train_env, eval_env = _FakeEnv(), _FakeEnv()
    worker = _worker(
        _cfg(tmp_path, train_video=True, eval_video=True),
        enable_train=True,
        enable_eval=True,
        env_list=[train_env],
        eval_env_list=[eval_env],
    )

    worker._attach_media_index()

    assert len(train_env.attached) == 1
    assert len(eval_env.attached) == 1


def test_recording_disabled_everywhere_attaches_nothing(tmp_path):
    train_env, eval_env = _FakeEnv(), _FakeEnv()
    worker = _worker(
        _cfg(tmp_path, train_video=False, eval_video=False),
        enable_train=True,
        enable_eval=True,
        env_list=[train_env],
        eval_env_list=[eval_env],
    )

    worker._attach_media_index()

    assert train_env.attached == []
    assert eval_env.attached == []


@pytest.mark.parametrize("split", ["train", "eval"])
def test_a_split_without_a_video_cfg_is_not_an_error(tmp_path, split):
    """``video_cfg`` is supplied by the env defaults, not required of every config."""
    cfg = OmegaConf.create(
        {
            "runner": {"logger": {"log_path": str(tmp_path)}},
            "env": {split: {"max_episode_steps": 10}},
        }
    )
    env = _FakeEnv()
    worker = _worker(
        cfg,
        enable_train=split == "train",
        enable_eval=split == "eval",
        env_list=[env] if split == "train" else [],
        eval_env_list=[env] if split == "eval" else [],
    )

    worker._attach_media_index()

    assert env.attached == []
