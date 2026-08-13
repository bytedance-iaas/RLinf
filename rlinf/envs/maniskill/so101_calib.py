# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SO101 LeRobot <-> ManiSkill unit conversion.

The LeRobot SO101 dataset stores joint values in LeRobot's NORMALIZED units, not
radians: the 5 arm joints use RANGE_M100_100 (range_min -> -100, range_max ->
+100) and the gripper uses RANGE_0_100 (range_min -> 0, range_max -> 100). See
``lerobot/motors/motors_bus.py::_normalize``. ManiSkill's ``pd_joint_pos``
controller and ``get_qpos`` are in RADIANS.

This module maps between the two using the follower calibration (raw feetech
ticks, 4096 ticks / revolution). Physical joint angle = (raw - homing_offset) *
2*pi/4096. The ManiSkill SO100 URDF joint zero is assumed to coincide with the
real homed zero; per-joint zero/direction offsets that remain after this
conversion are a sim2real calibration detail to tune against the real arm.

NOTE: replace SO101_CALIB with YOUR follower calibration if the robot is
recalibrated.
"""
import numpy as np

TICKS_PER_REV = 4096
RAD_PER_TICK = 2.0 * np.pi / TICKS_PER_REV
# LeRobot writes homing_offset into the servo HW register, so the position it
# reads back is already homed: the homed "zero" sits at max_res/2 = 2048 ticks
# (see feetech `_get_half_turn_homings`: homing = pos - max_res/2). So range_min/
# range_max are in this homed frame and the physical angle = (tick - CENTER)*rad.
CENTER_TICK = TICKS_PER_REV / 2  # 2048

# Joint order matches the dataset action/state layout:
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
SO101_CALIB = [
    {"name": "shoulder_pan", "min": 1110, "max": 3211, "homing": 1919, "mode": "m100"},
    {"name": "shoulder_lift", "min": 1364, "max": 3661, "homing": 1812, "mode": "m100"},
    {"name": "elbow_flex", "min": 506, "max": 2736, "homing": 1306, "mode": "m100"},
    {"name": "wrist_flex", "min": 90, "max": 2393, "homing": -1918, "mode": "m100"},
    {"name": "wrist_roll", "min": 0, "max": 4095, "homing": -1348, "mode": "m100"},
    {"name": "gripper", "min": 2023, "max": 2544, "homing": 963, "mode": "r0_100"},
]

_MIN = np.array([j["min"] for j in SO101_CALIB], dtype=np.float64)
_MAX = np.array([j["max"] for j in SO101_CALIB], dtype=np.float64)
_IS_GRIPPER = np.array([j["mode"] == "r0_100" for j in SO101_CALIB])

# norm -> homed tick:  tick = norm * _RAW_SCALE + _RAW_OFFSET
_RAW_SCALE = np.where(_IS_GRIPPER, (_MAX - _MIN) / 100.0, (_MAX - _MIN) / 200.0)
_RAW_OFFSET = np.where(_IS_GRIPPER, _MIN, (_MAX + _MIN) / 2.0)

# Per-joint sim<->real ALIGNMENT (calibrated against the real front-cam video):
# the ManiSkill SO100 URDF joint axes/zeros differ from the real SO101 servo
# convention, so each joint may need a direction flip (SIGN) and a zero offset
# (OFFSET, radians). Applied as: q_sim = SIGN * (tick-CENTER)*rad + OFFSET.
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# wrist_flex +0.6 rad: tilts the gripper down-forward to grab table objects (the
# generic SO100 wrist zero left it pointing forward = "grabbing air"). Verified
# against the real wrist-cam view (gripper looks down-forward at the red cube).
OFFSET = np.array([0.0, 0.0, 0.0, 0.6, 0.0, 0.0])

# ManiSkill SO100 URDF joint limits (radians), same order as SO101_CALIB. The
# real dataset commands over-travel a few joints at their trajectory extremes
# (shoulder_lift +1.70 > +1.57, elbow_flex -1.67 < -1.57, and especially
# wrist_roll -3.65 < -3.14 because its follower calibration uses the full
# uncalibrated 0..4095 tick range). The articulation would clamp these anyway;
# clipping the *commanded target* here keeps it well-defined. The operative
# grasp-phase values sit comfortably inside the limits, so clipping only trims
# rare extreme frames and does not disturb the validated reaching behaviour.
# Limits match the WIDENED so101 URDF (union of stock SO100 limits and the
# real servo calibration ranges — the stock URDF under-modeled the hardware;
# see rlinf/envs/maniskill/so101_agent.py).
JOINT_LIMITS_LOW = np.array([-2.0, -1.5708, -2.38, -3.01, -3.14159, -1.1])
JOINT_LIMITS_HIGH = np.array([2.0, 2.48, 1.5708, 1.8, 3.14159, 1.1])

# Gripper is a special case: the real feetech parallel gripper and the sim SO100
# revolute jaw have entirely different linkage geometry, so the arm-joint
# tick->rad conversion does NOT apply (it left the jaws stuck at an 8.1 cm gap,
# far wider than the 2.9 cm cube, so the gripper could never close on it). We map
# the LeRobot normalized gripper (0 = fully-closed grasp, 100 = fully open)
# directly onto a sim jaw angle that actually opens/closes around the cube.
# Measured jaw-tip gap vs gripper qpos in sim:
#   qpos -1.0 rad -> ~1.4 cm gap (drives past the 2.9 cm cube -> firm press/grasp)
#   qpos -0.34    -> ~6   cm gap (norm ~44 approach: clears the cube)
#   qpos +0.5 rad -> ~11  cm gap (norm 100 fully open)
GRIPPER_IDX = 5
GRIPPER_RAD_CLOSED = -1.0  # LeRobot gripper norm 0
GRIPPER_RAD_OPEN = 0.5  # LeRobot gripper norm 100
_GRIPPER_SPAN = GRIPPER_RAD_OPEN - GRIPPER_RAD_CLOSED


def norm_to_rad(actions):
    """LeRobot-normalized joint targets (last dim = 6) -> ManiSkill radians."""
    a = np.asarray(actions, dtype=np.float64)
    tick = a * _RAW_SCALE + _RAW_OFFSET
    base = (tick - CENTER_TICK) * RAD_PER_TICK
    rad = SIGN * base + OFFSET
    # Gripper: override the arm-style conversion with the dedicated close<->open
    # map so the jaws actually reach the cube.
    rad[..., GRIPPER_IDX] = (
        GRIPPER_RAD_CLOSED + (a[..., GRIPPER_IDX] / 100.0) * _GRIPPER_SPAN
    )
    rad = np.clip(rad, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
    return rad.astype(np.float32)


def rad_to_norm(qpos):
    """ManiSkill joint radians (last dim = 6) -> LeRobot-normalized units."""
    q = np.asarray(qpos, dtype=np.float64)
    base = (q - OFFSET) / SIGN
    tick = base / RAD_PER_TICK + CENTER_TICK
    norm = (tick - _RAW_OFFSET) / _RAW_SCALE
    # Gripper: inverse of the dedicated close<->open map (used for state feedback).
    norm[..., GRIPPER_IDX] = (
        (q[..., GRIPPER_IDX] - GRIPPER_RAD_CLOSED) / _GRIPPER_SPAN * 100.0
    )
    return norm.astype(np.float32)
