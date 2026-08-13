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
"""OpenPI DataConfig for the SO101/SO100 6-DoF joint-space arm."""
import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import so101_policy


@dataclasses.dataclass(frozen=True)
class LeRobotSO101DataConfig(DataConfigFactory):
    """Transform pipeline for a LeRobot SO101 pick-place dataset + PI0.5.

    Actions/state are absolute 6-dim joint positions, so ``extra_delta_transform``
    defaults to ``False`` (no delta conversion). If your LeRobot dataset stores
    *delta* joint actions instead, set it to ``True``.
    """

    # SO101 joint actions are absolute -> no delta conversion by default.
    extra_delta_transform: bool = False
    # LeRobot 0.6.1 datasets name the action column "action" (singular); openpi's
    # default is "actions". This drives delta_timestamps (the action-chunk loader).
    action_sequence_keys: tuple = ("action",)

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        # Repack maps *dataset* keys -> pipeline keys (training only, not inference).
        # henry-guo/so101-pick-place-v2 is a dual-camera dataset: a fixed front view
        # and a wrist view. Match these keys to your dataset ``meta/info.json``
        # ``features`` (the wrist key is commonly ``observation.images.wrist``).
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.front",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                so101_policy.SO101Inputs(
                    model_type=model_config.model_type, use_wrist_image=True
                )
            ],
            outputs=[so101_policy.SO101Outputs()],
        )

        if self.extra_delta_transform:
            # Delta on the 5 arm joints, absolute gripper.
            delta_action_mask = _transforms.make_bool_mask(5, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
