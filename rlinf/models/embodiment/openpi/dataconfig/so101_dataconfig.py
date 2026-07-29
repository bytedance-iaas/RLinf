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
    prompt during SFT. See ``pad_state_before_tokenize``.

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
    * ``pad_state_before_tokenize`` reproduces LeRobot's state-token layout; see
      that field's comment.
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
    # prompt string part of the observation protocol, and openpi and LeRobot build it
    # differently:
    #
    #   openpi  ModelTransformFactory: TokenizePrompt -> PadStatesAndActions
    #           => digitizes the 6-D state, prompt carries 6 numerals
    #   LeRobot make_pi05_pre_post_processors: pad_vector(state, 32) -> digitize
    #           => prompt carries 32 numerals, the 26 pad slots all reading 128
    #             (digitize(0.0) == 128, measured)
    #
    # Our checkpoint was SFT'd through LeRobot, so it only ever saw the 32-slot form:
    # 143 tokens vs 39 for the 6-slot form, diverging at token index 3. Setting this
    # True swaps the two transforms so the pad runs first, which reproduces LeRobot's
    # layout exactly -- reusing openpi's own transforms rather than reimplementing the
    # prompt format.
    #
    # Leave it False for checkpoints trained with openpi's native pipeline.
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
