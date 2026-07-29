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
"""OpenPI data config for SO101 (LeIsaac) fine-tuning and RL."""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import so101_policy


@dataclasses.dataclass(frozen=True)
class LeRobotSO101LiftCubeDataConfig(DataConfigFactory):
    """Data config aligned with the SO101 LiftCube fine-tuning recipe.

    Must stay consistent with how the checkpoint was trained. Two SO101-specific
    points:

    * The repack map has **no wrist entry**. LeIsaac's LiftCube deletes the wrist
      camera, so a LeRobot episode recorded from that task only carries
      ``observation.images.front``. Listing a key that does not exist in the
      dataset raises at load time.
    * ``observation.state`` and ``action`` are both the 6 joint values in LeRobot
      normalized motor units -- not an end-effector pose. See
      ``rlinf.envs.isaaclab.so101_utils`` for the unit mapping.
    """

    # Matches LeIsaac's LiftCubeEnvCfg.task_description verbatim, so the prompt the
    # policy sees at RL time is the one it was fine-tuned with.
    default_prompt: str | None = "Lift the red cube up."

    # Whether the underlying dataset/task provides a wrist view.
    has_wrist_image: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_map = {
            "observation/image": "observation.images.front",
            "observation/state": "observation.state",
            "actions": "action",
        }
        if self.has_wrist_image:
            repack_map["observation/wrist_image"] = "observation.images.wrist"

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_map)]
        )

        # Note the asymmetry, which is forced by how RLinf consumes this Group:
        #   * ``inputs``  are used by BOTH the SFT dataloader (LeRobot episodes, already
        #     in normalized motor units) and RL rollout. They therefore must not convert
        #     units -- the RL env normalizes ``states`` at its own boundary instead.
        #   * ``outputs`` are used ONLY by RL rollout
        #     (``openpi_cfg/__init__.py`` puts them in ``output_transforms``; the SFT
        #     worker only ever composes ``data_transforms.inputs``). That makes them the
        #     right and only place to map predicted actions back to IsaacLab radians.
        data_transforms = _transforms.Group(
            inputs=[
                so101_policy.SO101Inputs(
                    model_type=model_config.model_type,
                    has_wrist_image=self.has_wrist_image,
                )
            ],
            outputs=[so101_policy.SO101Outputs(convert_units=True)],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
