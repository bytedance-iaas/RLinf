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

"""SO101 arm constants and the LeIsaac <-> LeRobot unit conversion.

Lives under ``rlinf/envs`` rather than next to the OpenPI transforms because the
robot's joint order and limits are properties of the *simulated robot*, and both
sides need them: the env adapter (to normalize observations) and the policy
transforms (to denormalize predicted actions). Mirrors how
``rlinf/envs/robocasa/utils.py`` owns the RoboCasa action-space tables that
``openpi/policies/robocasa_policy.py`` consumes.

Deliberately depends on nothing but numpy at import time (``torch`` is imported
lazily by the one helper that needs it) so that any process -- training, SFT, or
simulator -- can import it.

Unit conversion
---------------

Checkpoints speak LeRobot's normalized motor units; IsaacLab speaks radians on the
USD joint ranges. The two are related by a per-joint affine map, ported here from
LeIsaac's ``convert_leisaac_action_to_lerobot`` /
``convert_lerobot_action_to_leisaac`` (``leisaac/utils/robot_utils.py``).

The limit tables are duplicated from LeIsaac rather than imported because
``leisaac`` transitively imports ``isaaclab.sim`` -> ``carb``, i.e. it only loads
inside a process that has already booted the Omniverse runtime -- which the
training process never does. :func:`assert_limits_match_leisaac` is provided so a
test running inside the simulator process can prove the copies have not drifted
from upstream.
"""

import numpy as np

# Joint order for every SO101 tensor in this module: the concatenation order that
# LeIsaac's ``init_action_cfg`` produces for the ``so101leader`` device, i.e. the
# 5 arm joints followed by the gripper. Mirrors leisaac.utils.constant.
SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

SO101_ACTION_DIM = len(SO101_JOINT_NAMES)

# Ported from leisaac.assets.robots.lerobot. Degrees.
# The USD articulation's joint limits -- the range IsaacLab actually simulates.
SO101_FOLLOWER_USD_JOINT_LIMITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}

# The real Feetech motors' normalized range -- the units LeRobot datasets record,
# and therefore the units the checkpoint was trained on. Note these are *not* the
# same as the USD limits, which is exactly why the affine map is needed.
SO101_FOLLOWER_MOTOR_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}


def assert_limits_match_leisaac() -> None:
    """Fail if the tables above have drifted from LeIsaac's.

    Only callable from a process where ``leisaac`` is importable, i.e. one that has
    already constructed ``AppLauncher``. Intended for the env smoke test.
    """
    from leisaac.assets.robots.lerobot import (
        SO101_FOLLOWER_MOTOR_LIMITS as _upstream_motor,
    )
    from leisaac.assets.robots.lerobot import (
        SO101_FOLLOWER_USD_JOINT_LIMLITS as _upstream_usd,
    )
    from leisaac.utils.constant import SINGLE_ARM_JOINT_NAMES as _upstream_names

    assert tuple(_upstream_names) == SO101_JOINT_NAMES, (
        f"joint order drifted: upstream={tuple(_upstream_names)} local={SO101_JOINT_NAMES}"
    )
    for name, local, upstream in (
        ("usd", SO101_FOLLOWER_USD_JOINT_LIMITS, _upstream_usd),
        ("motor", SO101_FOLLOWER_MOTOR_LIMITS, _upstream_motor),
    ):
        # Upstream dict ordering also defines the column order in its own
        # conversion helpers, so compare ordering too, not just values.
        assert tuple(local.keys()) == tuple(upstream.keys()), (
            f"{name} limit key order drifted: upstream={tuple(upstream.keys())} "
            f"local={tuple(local.keys())}"
        )
        for joint, bounds in upstream.items():
            assert tuple(float(b) for b in bounds) == local[joint], (
                f"{name} limits for {joint} drifted: upstream={bounds} local={local[joint]}"
            )


def _affine_scales(
    src: dict[str, tuple[float, float]], dst: dict[str, tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-joint coefficients mapping ``src`` range onto ``dst`` range."""
    src_lo = np.array([src[j][0] for j in SO101_JOINT_NAMES], dtype=np.float32)
    src_span = np.array(
        [src[j][1] - src[j][0] for j in SO101_JOINT_NAMES], dtype=np.float32
    )
    dst_lo = np.array([dst[j][0] for j in SO101_JOINT_NAMES], dtype=np.float32)
    dst_span = np.array(
        [dst[j][1] - dst[j][0] for j in SO101_JOINT_NAMES], dtype=np.float32
    )
    return src_lo, dst_span / src_span, dst_lo


def isaaclab_to_lerobot(action: np.ndarray) -> np.ndarray:
    """Radians on USD joint limits -> LeRobot normalized motor units.

    Vectorized equivalent of LeIsaac's ``convert_leisaac_action_to_lerobot``
    (which loops over columns). Accepts any shape whose last axis is the 6 joints,
    so a ``(chunk, 6)`` action sequence works as-is.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != SO101_ACTION_DIM:
        raise ValueError(
            f"expected last axis {SO101_ACTION_DIM}, got shape {action.shape}"
        )
    degrees = np.rad2deg(action)
    src_lo, scale, dst_lo = _affine_scales(
        SO101_FOLLOWER_USD_JOINT_LIMITS, SO101_FOLLOWER_MOTOR_LIMITS
    )
    return (degrees - src_lo) * scale + dst_lo


def lerobot_to_isaaclab(action: np.ndarray) -> np.ndarray:
    """LeRobot normalized motor units -> radians on USD joint limits.

    Vectorized equivalent of LeIsaac's ``convert_lerobot_action_to_leisaac``.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != SO101_ACTION_DIM:
        raise ValueError(
            f"expected last axis {SO101_ACTION_DIM}, got shape {action.shape}"
        )
    src_lo, scale, dst_lo = _affine_scales(
        SO101_FOLLOWER_MOTOR_LIMITS, SO101_FOLLOWER_USD_JOINT_LIMITS
    )
    degrees = (action - src_lo) * scale + dst_lo
    return np.deg2rad(degrees)


def lerobot_state_scales_torch(device):
    """Torch coefficients for the radians -> normalized-motor-units map.

    Returns ``(offset, scale, bias)`` such that
    ``(rad2deg(x) - offset) * scale + bias`` equals :func:`isaaclab_to_lerobot`.
    Used by the RLinf env adapter so observation conversion stays on the GPU
    instead of copying joint positions to host every step.
    """
    import torch

    src_lo, scale, dst_lo = _affine_scales(
        SO101_FOLLOWER_USD_JOINT_LIMITS, SO101_FOLLOWER_MOTOR_LIMITS
    )
    as_tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    return as_tensor(src_lo), as_tensor(scale), as_tensor(dst_lo)
