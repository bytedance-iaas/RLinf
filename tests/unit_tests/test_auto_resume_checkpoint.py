# Copyright 2025 The RLinf Authors.
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

"""Tests for ``ReasoningRunner`` auto-resume checkpoint selection.

``_save_checkpoint`` creates ``global_step_<N>/`` before filling it and writes the
dataloader state last, so an interrupted save leaves a directory that looks valid to
a name-only scan. These tests pin the behaviour that auto resume skips such
directories and falls back to the newest complete checkpoint.
"""

import os

import pytest
from omegaconf import OmegaConf

from rlinf.runners.reasoning_runner import ReasoningRunner


def _write_checkpoint(root: str, step: int, complete: bool, with_critic: bool = False):
    """Create a ``global_step_<step>`` directory, optionally missing its final write."""
    ckpt_dir = os.path.join(root, f"global_step_{step}")
    os.makedirs(os.path.join(ckpt_dir, "actor"), exist_ok=True)
    if with_critic:
        os.makedirs(os.path.join(ckpt_dir, "critic"), exist_ok=True)
    if complete:
        data_dir = os.path.join(ckpt_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "data.pt"), "w") as f:
            f.write("dataloader-state")
    return ckpt_dir


class _StubRunner:
    """Minimal stand-in exposing only what the checkpoint helpers touch."""

    def __init__(self, critic=None):
        self.critic = critic

    _is_complete_checkpoint = ReasoningRunner._is_complete_checkpoint


def test_complete_checkpoint_is_accepted(tmp_path):
    ckpt = _write_checkpoint(str(tmp_path), 40, complete=True)
    assert _StubRunner()._is_complete_checkpoint(ckpt)


def test_checkpoint_without_dataloader_state_is_rejected(tmp_path):
    """A save that died after the actor but before ``data/data.pt``."""
    ckpt = _write_checkpoint(str(tmp_path), 40, complete=False)
    assert not _StubRunner()._is_complete_checkpoint(ckpt)


def test_checkpoint_without_actor_is_rejected(tmp_path):
    """A save that died right after ``global_step_<N>/`` was created."""
    ckpt_dir = os.path.join(str(tmp_path), "global_step_40")
    os.makedirs(ckpt_dir)
    assert not _StubRunner()._is_complete_checkpoint(ckpt_dir)


def test_missing_critic_is_rejected_only_when_critic_is_used(tmp_path):
    ckpt = _write_checkpoint(str(tmp_path), 40, complete=True, with_critic=False)
    assert _StubRunner(critic=None)._is_complete_checkpoint(ckpt)
    assert not _StubRunner(critic=object())._is_complete_checkpoint(ckpt)


def _resolve_auto_resume(log_path: str) -> str | None:
    """Drive the real ``init_workers`` auto-resume branch and return the chosen dir.

    Only the resume-selection half of ``init_workers`` is exercised; the worker
    construction that follows needs a live cluster, so it is stubbed out.
    """
    cfg = OmegaConf.create(
        {"runner": {"resume_dir": "auto", "logger": {"log_path": log_path}}}
    )
    runner = _StubRunner()
    runner.cfg = cfg
    runner.init_rollout_workers = lambda: None
    runner.init_actor_critic_workers = lambda: None

    ReasoningRunner.init_workers(runner)
    return cfg.runner.resume_dir


@pytest.mark.parametrize("newest_complete", [True, False])
def test_auto_resume_picks_newest_complete_checkpoint(tmp_path, newest_complete):
    """The core regression: an interrupted newest save must not be resumed from."""
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    _write_checkpoint(str(checkpoints_dir), 40, complete=True)
    _write_checkpoint(str(checkpoints_dir), 80, complete=newest_complete)

    resume_dir = _resolve_auto_resume(str(tmp_path))

    expected_step = 80 if newest_complete else 40
    assert resume_dir == str(checkpoints_dir / f"global_step_{expected_step}")


def test_auto_resume_falls_back_to_scratch_when_all_incomplete(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    _write_checkpoint(str(checkpoints_dir), 40, complete=False)

    assert _resolve_auto_resume(str(tmp_path)) is None


def test_auto_resume_without_checkpoints_dir_starts_from_scratch(tmp_path):
    assert _resolve_auto_resume(str(tmp_path)) is None
