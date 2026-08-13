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

Mirrors ``maniskill_policy.py`` but for a 6-dim joint state/action layout
(``shoulder_pan``, ``shoulder_lift``, ``elbow_flex``, ``wrist_flex``,
``wrist_roll``, ``gripper``). State and actions are absolute joint positions,
matching both the ManiSkill ``so100`` ``pd_joint_pos`` controller and the
LeRobot-trained PI0.5 checkpoint.
"""
import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

# SO101/SO100 active-joint dimension: 5 arm joints + 1 gripper.
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
    use_wrist_image: bool = False
    default_prompt: str | None = None

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = (
            _parse_image(data["observation/wrist_image"])
            if self.use_wrist_image and data.get("observation/wrist_image") is not None
            else np.zeros_like(base_image)
        )
        has_wrist_image = (
            self.use_wrist_image and data.get("observation/wrist_image") is not None
        )

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if has_wrist_image else np.False_,
                "right_wrist_0_rgb": np.False_,
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
    """Convert model outputs back to the 6-dim SO101 joint action (inference only)."""

    output_action_dim: int = SO101_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        # The model action tensor is padded to the model action dim; slice out the
        # first ``output_action_dim`` (= 6) as the env-frame SO101 joint targets.
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}
