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
from collections.abc import Sequence

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import so101_policy


def _pad_state_before_tokenize(
    inputs: Sequence[_transforms.DataTransformFn],
) -> tuple[_transforms.DataTransformFn, ...]:
    """Move ``PadStatesAndActions`` ahead of ``TokenizePrompt``.

    ``ModelTransformFactory`` emits them in the opposite order, so the tokenizer
    sees the unpadded 6-D state. Swapping makes the discretized state span all
    ``max_action_dim`` slots, matching how LeRobot's pi0.5 processor built the
    prompt in versions up to 0.4.4 -- 0.5.0 dropped that pad, so this reorder is
    correct only for checkpoints fine-tuned on 0.4.4 or earlier. See
    ``pad_state_before_tokenize`` for how to tell from a checkpoint's own files.

    Raises rather than silently no-op'ing: this reorder is load-bearing for
    checkpoint compatibility, and if a future openpi renames or drops either
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
        # Already in the desired order; nothing to do.
        return tuple(inputs)
    reordered = list(inputs)
    reordered.insert(tok, reordered.pop(pad))
    return tuple(reordered)


@dataclasses.dataclass(frozen=True)
class LeRobotSO101LiftCubeDataConfig(DataConfigFactory):
    """Data config aligned with the SO101 LiftCube fine-tuning recipe.

    Must stay consistent with how the checkpoint was trained. Three SO101-specific
    points:

    * The repack map only gains a wrist entry when ``has_wrist_image`` is set.
      LeIsaac's LiftCube deletes the wrist camera, so a LeRobot episode recorded
      from that task only carries ``observation.images.front``, and listing a key
      that does not exist in the dataset raises at load time.
    * ``observation.state`` and ``action`` are both the 6 joint values in LeRobot
      normalized motor units -- not an end-effector pose. See
      ``rlinf.envs.isaaclab.so101_utils`` for the unit mapping.
    * ``pad_state_before_tokenize`` reproduces the state-token layout of LeRobot
      0.4.4 and earlier. It is version-specific, not a LeRobot-vs-openpi switch;
      see that field's comment.
    """

    # Matches LeIsaac's LiftCubeEnvCfg.task_description verbatim, so the prompt the
    # policy sees at RL time is the one it was fine-tuned with.
    default_prompt: str | None = "Lift the red cube up."

    # Whether the underlying dataset/task provides a wrist view.
    has_wrist_image: bool = False

    # Emit the discretized state over all `max_action_dim` slots instead of only the
    # 6 real ones.
    #
    # pi0.5 has no `state_proj` -- `PI0Pytorch.embed_suffix` reads its `state` argument
    # only under `if not self.pi05:` -- so the *only* path from proprioception into the
    # model is the discretized state embedded in the prompt text. That makes the exact
    # prompt string part of the observation protocol, and it is NOT a free choice: get
    # the layout wrong and the policy reads its joint angles off the wrong tokens.
    #
    # The two candidate layouts, for a 6-DoF arm with max_state_dim=32:
    #
    #   6 numerals   openpi ModelTransformFactory: TokenizePrompt -> PadStatesAndActions
    #                digitizes the unpadded 6-D state
    #                  "Task: ..., State: 137 3 255 117 255 -1;\nAction: "
    #   32 numerals  pad first, then digitize, so the 26 pad slots each contribute a
    #                literal "128" (digitize(0.0) == 128, measured)
    #                  "Task: ..., State: 137 3 255 117 255 -1 128 x26;\nAction: "
    #
    # Which one LeRobot produces depends on its VERSION, which is the subtlety here.
    # `Pi05PrepareStateTokenizerProcessorStep` called `pad_vector(state, max_state_dim)`
    # before digitizing in 0.4.4, and that call was REMOVED in 0.5.0:
    #
    #   lerobot 0.4.4   pads  -> 32 numerals   => this flag must be True
    #   lerobot >=0.5.0 no pad -> 6 numerals   => this flag must be False
    #
    # So set this from the version that produced the checkpoint, not from the fact that
    # LeRobot was involved at all. A checkpoint's own files identify it: a
    # `policy_postprocessor.json` step named `absolute_actions_processor` exists only in
    # 0.6.x, and `config.json` fields `pretrained_revision` / `use_relative_actions` /
    # `relative_exclude_joints` / `action_feature_names` are 0.6.0 additions that make
    # 0.4.4's PI05Config raise DecodingError.
    #
    # Measured cost of getting it wrong, on the SO101 pick-place checkpoint (0.6.x, so
    # False): with True, 108 of 200 token ids differ from LeRobot's reference and the
    # predicted actions diverge by mean 4.275 / max 31.59 motor units; with False the
    # tokens match id-for-id and the divergence drops to mean 0.594, which is below the
    # policy's own 0.793-unit noise-driven spread. See scripts/check_state_token_layout.py
    # and scripts/check_inference_parity.py.
    #
    # Setting this True reuses openpi's own transforms by swapping their order, rather
    # than reimplementing the prompt format.
    pad_state_before_tokenize: bool = False

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
        if self.pad_state_before_tokenize:
            model_transforms = dataclasses.replace(
                model_transforms,
                inputs=_pad_state_before_tokenize(model_transforms.inputs),
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
