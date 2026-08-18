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
"""OpenPI input/output transforms for the SO101/SO100 6-DoF joint-space arm.

Joint layout (``shoulder_pan``, ``shoulder_lift``, ``elbow_flex``,
``wrist_flex``, ``wrist_roll``, ``gripper``). State and actions are absolute
joint positions, matching both ManiSkill's ``pd_joint_pos`` controller and the
LeRobot-trained PI0.5 checkpoints.

Why the generic ``maniskill_policy`` transforms are not reused
--------------------------------------------------------------

They are written for the 7-dim Franka/WidowX end-effector convention: they slice
to 7 dims and binarize the last one into a ``{-1, +1}`` gripper flag. On SO101
the gripper is a *position-controlled joint* whose continuous value is the
commanded aperture -- sign-collapsing it throws the aperture away and makes
graded grasping impossible, and a 7-wide slice would silently pull in a
neighbouring chunk element.

Where unit conversion happens
-----------------------------

NOT here, deliberately. ``observation/state`` and ``actions`` are in LeRobot
normalized motor units on both sides of this transform. Two reasons:

* This transform instance is shared by the SFT dataloader (which feeds LeRobot
  episodes, already normalized) and RL rollout. Converting here would be right
  for one caller and wrong for the other.
* ``OpenPi0ForRLActionPrediction.output_transform`` applies output transforms in
  a per-sample Python loop, so a conversion here costs one call per env per
  step. ``action_utils.prepare_actions_for_maniskill`` does it once per batch,
  vectorized, on the whole chunk.

The env side converts at its own boundary: actions via
``so101_calib.norm_to_rad`` in ``action_utils``, observations via
``so101_calib.rad_to_norm_torch`` in ``ManiskillEnv._wrap_obs``.
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

# SO101/SO100 active-joint dimension: 5 arm joints + 1 gripper. Duplicated as a
# literal rather than imported from ``rlinf.envs.maniskill.so101_calib`` so that
# importing these transforms never pulls in ``rlinf.envs`` (and with it the
# simulator dependencies) in a training-only process. ``test_so101_policy.py``
# asserts the two definitions agree.
SO101_ACTION_DIM = 6


def make_so101_example() -> dict:
    """Creates a random input example for the SO101 policy."""
    return {
        "observation/state": np.random.rand(SO101_ACTION_DIM),
        "observation/image": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Convert env/dataset inputs to the model format (training and inference)."""

    # Determines which model will be used. Do not change this for your own dataset.
    model_type: _model.ModelType
    # Whether the task/dataset provides a wrist view. When it does not, the slot is
    # zero-filled AND masked out, so the model is told the view is absent rather
    # than being shown a black frame it would treat as real sensor data.
    has_wrist_image: bool = False
    default_prompt: str | None = None

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        wrist_present = (
            self.has_wrist_image and data.get("observation/wrist_image") is not None
        )
        if wrist_present:
            wrist_image = _parse_image(data["observation/wrist_image"])
        else:
            wrist_image = np.zeros_like(base_image)

        # PI0_FAST has no masking path, so every slot must be True for it.
        is_fast = self.model_type == _model.ModelType.PI0_FAST
        wrist_mask = np.True_ if (wrist_present or is_fast) else np.False_
        unused_mask = np.True_ if is_fast else np.False_

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": wrist_mask,
                "right_wrist_0_rgb": unused_mask,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]
        elif self.default_prompt is not None:
            inputs["prompt"] = self.default_prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Convert model outputs back to the 6-dim SO101 joint action (inference only).

    The model's action head is wider than the robot, so this slices out the first
    6 dims. It does *not* binarize the gripper -- see the module docstring.
    """

    output_action_dim: int = SO101_ACTION_DIM
    # Off by default: the ManiSkill path converts units in
    # ``action_utils.prepare_actions_for_maniskill`` (once per batch) rather than
    # here (once per sample). Set a converter for a consumer that wants env-ready
    # units straight out of the transform -- e.g. the IsaacLab SO101 line of work,
    # whose ``prepare_actions_for_isaaclab`` is a pass-through.
    unit_converter: object | None = None

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < self.output_action_dim:
            raise ValueError(
                f"model produced {actions.shape[-1]} action dims, need at least "
                f"{self.output_action_dim} for SO101; check action_dim in the config"
            )
        actions = actions[..., : self.output_action_dim]
        if self.unit_converter is not None:
            actions = self.unit_converter(actions)
        return {"actions": actions}
