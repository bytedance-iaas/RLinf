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
from collections.abc import Sequence

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import so101_policy


def _reorder_pad_before_tokenize(
    inputs: Sequence[_transforms.DataTransformFn],
) -> tuple[_transforms.DataTransformFn, ...]:
    """Move ``PadStatesAndActions`` ahead of ``TokenizePrompt``.

    Raises rather than silently no-op'ing: the reorder is load-bearing for
    checkpoint compatibility, so if a future openpi renames or drops either
    transform, a quiet fallback would look exactly like a bad checkpoint.
    """
    names = [type(t).__name__ for t in inputs]
    try:
        tok = names.index("TokenizePrompt")
        pad = names.index("PadStatesAndActions")
    except ValueError as exc:
        raise RuntimeError(
            "pad_state_before_tokenize needs both TokenizePrompt and "
            f"PadStatesAndActions in the model transforms, got {names}"
        ) from exc
    if pad < tok:
        return tuple(inputs)  # already in the desired order
    reordered = list(inputs)
    reordered.insert(tok, reordered.pop(pad))
    return tuple(reordered)


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
    # Dataset image keys, overridable for datasets that name their cameras
    # differently. Must match the dataset's ``meta/info.json`` features.
    front_image_key: str = "observation.images.front"
    wrist_image_key: str = "observation.images.wrist"
    has_wrist_image: bool = True

    # Emit the discretized state over all ``max_action_dim`` slots instead of only
    # the 6 real ones.
    #
    # pi0.5 has no ``state_proj`` -- the only path from proprioception into the
    # model is the discretized state embedded in the prompt text -- so the exact
    # prompt layout is part of the observation protocol. Which layout applies
    # depends on the LeRobot VERSION that produced the checkpoint:
    #
    #   lerobot 0.4.4    pads before digitizing -> 32 numerals => True
    #   lerobot >=0.5.0  no pad                 ->  6 numerals => False
    #
    # A checkpoint identifies itself: ``config.json`` fields ``pretrained_revision``
    # / ``use_relative_actions`` / ``relative_exclude_joints`` /
    # ``action_feature_names`` are 0.6.0 additions. The SO101 pick-place checkpoint
    # in use has all four, i.e. 0.6.x, hence the False default matching openpi's
    # own ordering. Flip it only for a 0.4.4-era checkpoint.
    pad_state_before_tokenize: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        # Repack maps *dataset* keys -> pipeline keys (training only, not inference).
        # Listing a key the dataset does not have raises at load time, so the wrist
        # entry only appears when the dataset actually carries that view.
        repack_map = {
            "observation/image": self.front_image_key,
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        if self.has_wrist_image:
            repack_map["observation/wrist_image"] = self.wrist_image_key
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_map)]
        )

        data_transforms = _transforms.Group(
            inputs=[
                so101_policy.SO101Inputs(
                    model_type=model_config.model_type,
                    has_wrist_image=self.has_wrist_image,
                )
            ],
            # No unit_converter: the ManiSkill path converts once per batch in
            # action_utils.prepare_actions_for_maniskill instead of once per
            # sample here. See so101_policy's module docstring.
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
        if self.pad_state_before_tokenize:
            model_transforms = dataclasses.replace(
                model_transforms,
                inputs=_reorder_pad_before_tokenize(model_transforms.inputs),
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
