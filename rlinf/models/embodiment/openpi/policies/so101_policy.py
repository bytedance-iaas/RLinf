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

"""OpenPI transforms for the SO101 arm (LeIsaac tasks).

Why this exists instead of reusing :mod:`isaaclab_policy`
---------------------------------------------------------

:class:`IsaacLabOutputs` is written for the Franka stack-cube task and does two
things that are actively wrong for SO101:

* ``data["actions"][:, :7]`` -- slices to 7 dims. SO101 has 6 (5 arm joints +
  gripper). Taking 7 would either fail or, worse, silently pull in a neighbouring
  chunk element.
* ``np.sign(actions[..., -1])`` -- binarizes the last dim to a Franka-style
  ``{-1, +1}`` gripper command. SO101's gripper is a *position-controlled joint*
  like any other, spanning a continuous range; sign-collapsing it throws away the
  aperture and makes graded grasping impossible.

The joint tables and the LeRobot <-> IsaacLab unit conversion live in
``rlinf.envs.isaaclab.so101_utils`` (they describe the robot, and the env adapter
needs them too). They are imported lazily inside the transforms, matching
:mod:`robocasa_policy`, so that importing this module never pulls in ``rlinf.envs``.
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

# Duplicated as a literal rather than imported from ``so101_utils``: importing that
# module executes ``rlinf/envs/isaaclab/__init__.py``, which imports the SO101 env
# adapter, which imports this module -- a cycle. ``test_so101_env.py`` asserts the
# two definitions agree.
SO101_ACTION_DIM = 6


def make_so101_example() -> dict:
    """Creates a random input example for the SO101 policy."""
    return {
        "observation/state": np.random.rand(SO101_ACTION_DIM),
        "observation/image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "Lift the red cube up.",
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
    """Convert SO101 observations into OpenPI model inputs.

    ``observation/state`` must **already** be in LeRobot normalized motor units.

    No unit conversion happens here, deliberately. This exact transform instance is
    shared by two callers with different input units:

    * the SFT dataloader (``workers/sft/fsdp_cfg_worker.py``), which feeds LeRobot
      episodes -- already normalized;
    * RL rollout (``models/embodiment/openpi_cfg/__init__.py``), which feeds
      observations straight off IsaacLab -- radians.

    Converting here would therefore be right for one caller and wrong for the
    other. The RL side converts at the env boundary instead (see
    ``IsaaclabSO101Env`` / the ``states`` key), which keeps this transform
    single-meaning. Use ``so101_utils.isaaclab_to_lerobot`` if you need the mapping.
    """

    model_type: _model.ModelType

    # LiftCube has no wrist camera (LeIsaac deletes it), so the wrist slot is left
    # masked out rather than filled with a copy of the base view. Set True for
    # tasks that do provide ``observation/wrist_image``.
    has_wrist_image: bool = False

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        state = np.asarray(data["observation/state"], dtype=np.float32)

        if self.has_wrist_image:
            wrist_image = _parse_image(data["observation/wrist_image"])
            wrist_mask = np.True_
        else:
            # Zeros + a False mask: the model is told the view is absent instead of
            # being shown a black frame it would treat as real sensor data. PI0_FAST
            # is the exception -- it has no masking path, so every slot must be True.
            wrist_image = np.zeros_like(base_image)
            wrist_mask = (
                np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
            )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": wrist_mask,
                "right_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }

        if "actions" in data:
            # Same reasoning as ``state``: already in the caller's target units.
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Convert OpenPI outputs into IsaacLab SO101 actions.

    Takes the first 6 dims (the model's action head may be wider than the robot)
    and maps LeRobot normalized units back to radians. Notably it does *not*
    binarize the gripper: on SO101 the gripper is a position-controlled joint and
    its continuous value is the commanded aperture.
    """

    # Set False when the consumer wants raw normalized units (e.g. comparing
    # against a LeRobot dataset rather than stepping IsaacLab).
    convert_units: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < SO101_ACTION_DIM:
            raise ValueError(
                f"model produced {actions.shape[-1]} action dims, need at least "
                f"{SO101_ACTION_DIM} for SO101; check action_dim in the config"
            )
        actions = actions[..., :SO101_ACTION_DIM]
        if self.convert_units:
            # Lazy: keeps ``rlinf.envs`` off this module's import graph, so the
            # transforms load in a training process with no simulator deps.
            from rlinf.envs.isaaclab.so101_utils import lerobot_to_isaaclab

            actions = lerobot_to_isaaclab(actions)
        return {"actions": actions}
