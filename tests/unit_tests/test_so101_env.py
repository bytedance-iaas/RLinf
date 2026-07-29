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
"""SO101 unit conversion, action-space, and observation-contract tests.

Everything here runs without Isaac Sim: ``so101_utils`` is deliberately
numpy-only, and the observation contract is exercised against a stub instead of a
live simulator. The parts that genuinely need Omniverse (asset loading, reward
firing, throughput) live in ``scripts/smoke_so101_env.py`` and
``scripts/bench_so101_env.py`` in the integration workspace.
"""

import numpy as np
import pytest
import torch

from rlinf.envs.isaaclab.so101_utils import (
    SO101_ACTION_DIM,
    SO101_FOLLOWER_MOTOR_LIMITS,
    SO101_FOLLOWER_USD_JOINT_LIMITS,
    SO101_JOINT_NAMES,
    isaaclab_to_lerobot,
    lerobot_state_scales_torch,
    lerobot_to_isaaclab,
)

RNG = np.random.default_rng(0)
TOL = 1e-5

_USD_LO = np.deg2rad([SO101_FOLLOWER_USD_JOINT_LIMITS[j][0] for j in SO101_JOINT_NAMES])
_USD_HI = np.deg2rad([SO101_FOLLOWER_USD_JOINT_LIMITS[j][1] for j in SO101_JOINT_NAMES])
_MOTOR_LO = np.array(
    [SO101_FOLLOWER_MOTOR_LIMITS[j][0] for j in SO101_JOINT_NAMES], np.float32
)
_MOTOR_HI = np.array(
    [SO101_FOLLOWER_MOTOR_LIMITS[j][1] for j in SO101_JOINT_NAMES], np.float32
)


def _random_radians(n: int) -> np.ndarray:
    """Joint configurations sampled uniformly inside the USD limits."""
    return RNG.uniform(_USD_LO, _USD_HI, size=(n, SO101_ACTION_DIM)).astype(np.float32)


def test_action_dim_is_six():
    assert SO101_ACTION_DIM == 6
    assert len(SO101_JOINT_NAMES) == 6


def test_joint_tables_cover_every_joint():
    assert tuple(SO101_FOLLOWER_USD_JOINT_LIMITS) == SO101_JOINT_NAMES
    assert tuple(SO101_FOLLOWER_MOTOR_LIMITS) == SO101_JOINT_NAMES
    for table in (SO101_FOLLOWER_USD_JOINT_LIMITS, SO101_FOLLOWER_MOTOR_LIMITS):
        for joint, (lo, hi) in table.items():
            assert lo < hi, f"{joint} has an empty range"


def test_action_dim_literal_matches_policy_module():
    """The duplicated literal in ``so101_policy`` must track ``so101_utils``.

    The two cannot import each other (``so101_utils`` lives under the isaaclab env
    package, whose ``__init__`` imports the adapter that imports the policy), so
    the value is written twice on purpose. This is the guard against drift.
    """
    openpi = pytest.importorskip(
        "openpi", reason="so101_policy imports openpi at module level"
    )
    del openpi
    from rlinf.models.embodiment.openpi.policies.so101_policy import (
        SO101_ACTION_DIM as POLICY_DIM,
    )

    assert POLICY_DIM == SO101_ACTION_DIM


@pytest.mark.parametrize("n", [1, 17, 256])
def test_unit_round_trip_radians(n):
    rad = _random_radians(n)
    assert np.allclose(lerobot_to_isaaclab(isaaclab_to_lerobot(rad)), rad, atol=TOL)


def test_unit_round_trip_normalized():
    norm = RNG.uniform(_MOTOR_LO, _MOTOR_HI, size=(64, SO101_ACTION_DIM)).astype(
        np.float32
    )
    assert np.allclose(isaaclab_to_lerobot(lerobot_to_isaaclab(norm)), norm, atol=1e-3)


def test_limits_map_onto_motor_range():
    """USD joint limits must land exactly on the motor range endpoints."""
    assert np.allclose(isaaclab_to_lerobot(_USD_LO[None])[0], _MOTOR_LO, atol=1e-3)
    assert np.allclose(isaaclab_to_lerobot(_USD_HI[None])[0], _MOTOR_HI, atol=1e-3)


def test_gripper_is_not_symmetric():
    """Guards the asymmetry that makes a shared affine map necessary.

    The gripper's USD range is [-10, 100] deg but its motor range is [0, 100]; the
    arm joints are symmetric. A single global scale would therefore be wrong.
    """
    assert SO101_FOLLOWER_USD_JOINT_LIMITS["gripper"] == (-10.0, 100.0)
    assert SO101_FOLLOWER_MOTOR_LIMITS["gripper"] == (0.0, 100.0)
    # Zero motor units is the fully-closed end, i.e. -10 deg, not 0 deg.
    closed = lerobot_to_isaaclab(np.zeros((1, SO101_ACTION_DIM), np.float32))[0, -1]
    assert np.isclose(np.rad2deg(closed), -10.0, atol=1e-3)


@pytest.mark.parametrize("shape", [(SO101_ACTION_DIM,), (4, SO101_ACTION_DIM)])
def test_conversion_preserves_leading_shape(shape):
    x = np.zeros(shape, np.float32)
    assert isaaclab_to_lerobot(x).shape == shape
    assert lerobot_to_isaaclab(x).shape == shape


@pytest.mark.parametrize("bad_dim", [5, 7, 32])
def test_conversion_rejects_wrong_width(bad_dim):
    with pytest.raises(ValueError, match="expected last axis"):
        isaaclab_to_lerobot(np.zeros((3, bad_dim), np.float32))
    with pytest.raises(ValueError, match="expected last axis"):
        lerobot_to_isaaclab(np.zeros((3, bad_dim), np.float32))


def test_torch_state_scales_match_numpy():
    """The on-device state path must agree with the numpy reference."""
    rad = _random_radians(128)
    offset, scale, bias = lerobot_state_scales_torch("cpu")
    got = (torch.rad2deg(torch.from_numpy(rad)) - offset) * scale + bias
    assert np.allclose(got.numpy(), isaaclab_to_lerobot(rad), atol=1e-4)


def test_torch_state_scales_shapes():
    for t in lerobot_state_scales_torch("cpu"):
        assert t.shape == (SO101_ACTION_DIM,)
        assert t.dtype == torch.float32


# --------------------------------------------------------------------------- #
# Action preparation
# --------------------------------------------------------------------------- #


def test_prepare_actions_keeps_so101_gripper_continuous():
    """6-DoF SO101 must never get the Franka binary-gripper treatment.

    SO101's last dim is the gripper *joint angle*; collapsing it to +-1 would
    destroy graded grasping. The 7-DoF Franka path must keep binarizing.
    """
    from rlinf.envs.action_utils import prepare_actions

    a6 = np.zeros((2, 5, SO101_ACTION_DIM), np.float32)
    a6[..., -1] = 0.9
    out = prepare_actions(
        raw_chunk_actions=a6.copy(),
        env_type="isaaclab",
        model_type="openvla",
        num_action_chunks=5,
        action_dim=SO101_ACTION_DIM,
    )
    assert np.allclose(np.asarray(out)[..., -1], 0.9)


def test_prepare_actions_still_binarizes_franka_gripper():
    from rlinf.envs.action_utils import prepare_actions

    a7 = np.zeros((2, 5, 7), np.float32)
    a7[..., -1] = 0.9
    out = np.asarray(
        prepare_actions(
            raw_chunk_actions=a7.copy(),
            env_type="isaaclab",
            model_type="openvla",
            num_action_chunks=5,
            action_dim=7,
        )
    )
    # 0.9 -> 2*0.9-1 = 0.8 -> sign * -1 -> -1.0
    assert np.all(out[..., -1] == -1.0)


def test_prepare_actions_openpi_so101_is_passthrough():
    """OpenPI chunks arrive already in IsaacLab radians via ``SO101Outputs``."""
    from rlinf.envs.action_utils import prepare_actions

    a = RNG.uniform(-1, 1, (3, 4, SO101_ACTION_DIM)).astype(np.float32)
    out = prepare_actions(
        raw_chunk_actions=a.copy(),
        env_type="isaaclab",
        model_type="openpi",
        num_action_chunks=4,
        action_dim=SO101_ACTION_DIM,
    )
    assert np.allclose(np.asarray(out), a)


# --------------------------------------------------------------------------- #
# Registry and observation contract
# --------------------------------------------------------------------------- #

SO101_TASK_ID = "LeIsaac-SO101-LiftCube-Rewarded-v0"


def test_task_is_registered():
    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

    assert SO101_TASK_ID in REGISTER_ISAACLAB_ENVS
    assert REGISTER_ISAACLAB_ENVS[SO101_TASK_ID].__name__ == "IsaaclabSO101Env"


class _BlockOpenPI:
    """Meta-path hook that makes ``import openpi`` fail as if it were absent."""

    PREFIX = "openpi"

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.PREFIX or fullname.startswith(self.PREFIX + "."):
            raise ImportError(f"{fullname} is not installed (simulated)")
        return None


def test_env_registry_does_not_require_openpi():
    """Importing the IsaacLab registry must not drag in openpi.

    Each model gets its own venv (see ``requirements/install.sh``), so a gr00t
    install has no openpi at all. If the SO101 adapter imported the OpenPI
    transforms at module scope, ``isaaclab_franka_stack_cube_ppo_gr00t.yaml`` would
    fail to load its env.

    Simply diffing ``sys.modules`` is not enough: openpi is already imported by the
    time this module is collected, so it would never look "newly imported". The
    only way to prove independence is to make openpi genuinely unimportable and
    re-import from scratch.
    """
    import importlib
    import sys

    purged = ("openpi", "rlinf.envs.isaaclab", "rlinf.models.embodiment.openpi")
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name.split(".")[0] == "openpi" or name.startswith(purged)
    }
    blocker = _BlockOpenPI()
    sys.meta_path.insert(0, blocker)
    try:
        for name in saved:
            del sys.modules[name]
        # Sanity: the blocker must really work, else the assertion below is vacuous.
        with pytest.raises(ImportError):
            importlib.import_module("openpi")
        module = importlib.import_module("rlinf.envs.isaaclab")
        assert "LeIsaac-SO101-LiftCube-Rewarded-v0" in module.REGISTER_ISAACLAB_ENVS
    finally:
        sys.meta_path.remove(blocker)
        for name in [n for n in sys.modules if n.startswith(purged)]:
            del sys.modules[name]
        sys.modules.update(saved)


def test_wrap_obs_contract():
    """``_wrap_obs`` must normalize states and always emit the optional camera keys.

    ``OpenPi0ForRLActionPrediction.obs_processor`` subscripts ``wrist_images`` and
    ``extra_view_images`` directly, so a missing key is a KeyError at rollout time,
    not a graceful fallback. LiftCube has no wrist camera, hence ``None``.
    """
    from rlinf.envs.isaaclab.tasks.so101 import IsaaclabSO101Env

    num_envs = 3
    joint_pos = torch.from_numpy(_random_radians(num_envs))
    front = torch.zeros((num_envs, 8, 8, 3), dtype=torch.uint8)

    # Bypass __init__: constructing the real env needs a live simulator. Only the
    # attributes _wrap_obs touches are stubbed in.
    env = object.__new__(IsaaclabSO101Env)
    env.num_envs = num_envs
    env.task_description = "Lift the red cube up."
    env._state_offset, env._state_scale, env._state_bias = lerobot_state_scales_torch(
        "cpu"
    )

    obs = env._wrap_obs({"policy": {"front": front, "joint_pos": joint_pos}})

    assert set(obs) == {
        "main_images",
        "task_descriptions",
        "states",
        "wrist_images",
        "extra_view_images",
    }
    assert obs["wrist_images"] is None
    assert obs["extra_view_images"] is None
    assert obs["main_images"] is front
    assert obs["task_descriptions"] == ["Lift the red cube up."] * num_envs
    assert obs["states"].shape == (num_envs, SO101_ACTION_DIM)
    # states must be normalized motor units, not raw radians
    assert np.allclose(
        obs["states"].numpy(), isaaclab_to_lerobot(joint_pos.numpy()), atol=1e-4
    )


def test_wrap_obs_passes_through_wrist_when_present():
    """A task that keeps the wrist camera must forward it untouched."""
    from rlinf.envs.isaaclab.tasks.so101 import IsaaclabSO101Env

    wrist = torch.ones((2, 8, 8, 3), dtype=torch.uint8)
    env = object.__new__(IsaaclabSO101Env)
    env.num_envs = 2
    env.task_description = "t"
    env._state_offset, env._state_scale, env._state_bias = lerobot_state_scales_torch(
        "cpu"
    )
    obs = env._wrap_obs(
        {
            "policy": {
                "front": torch.zeros((2, 8, 8, 3), dtype=torch.uint8),
                "wrist": wrist,
                "joint_pos": torch.from_numpy(_random_radians(2)),
            }
        }
    )
    assert obs["wrist_images"] is wrist


def test_wrap_obs_truncates_wider_joint_state():
    """Extra articulation DoFs (if a scene adds them) must be sliced off."""
    from rlinf.envs.isaaclab.tasks.so101 import IsaaclabSO101Env

    env = object.__new__(IsaaclabSO101Env)
    env.num_envs = 1
    env.task_description = "t"
    env._state_offset, env._state_scale, env._state_bias = lerobot_state_scales_torch(
        "cpu"
    )
    obs = env._wrap_obs(
        {
            "policy": {
                "front": torch.zeros((1, 4, 4, 3), dtype=torch.uint8),
                "joint_pos": torch.zeros((1, SO101_ACTION_DIM + 2)),
            }
        }
    )
    assert obs["states"].shape == (1, SO101_ACTION_DIM)
