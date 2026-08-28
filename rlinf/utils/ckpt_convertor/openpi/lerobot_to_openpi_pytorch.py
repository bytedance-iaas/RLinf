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

"""Convert a LeRobot ``pi05`` checkpoint to the OpenPI PyTorch layout.

LeRobot fine-tuning is the most common way to produce a pi0.5 policy for a new
robot -- most of the pi0.5 checkpoints published on the Hub are in its format --
but RLinf cannot load one directly. Two things differ, and both fail silently:

1. **Weight key prefix.** LeRobot saves the policy wrapper, so every tensor is
   named ``model.<...>``. ``OpenPi0ForRLActionPrediction`` inherits ``PI0Pytorch``
   and expects bare names. ``rlinf/models/embodiment/openpi/__init__.py`` loads
   with ``strict=False``, so a prefix mismatch drops *every* tensor without
   raising and the policy then runs on its random initialization -- which looks
   like a checkpoint that trains but never succeeds, not like a load error.

2. **Norm-stats format.** LeRobot writes per-feature tensors into a
   ``*_normalizer_processor.safetensors``; openpi wants a ``norm_stats.json``
   holding ``{mean, std, q01, q99}`` under ``state`` / ``actions``, at
   ``<output>/<asset-id>/norm_stats.json``. The other modes in this package copy
   ``norm_stats.json`` verbatim, so none of them can do this.

Stats keep their **native width** (6 for a 6-DoF arm). openpi's ``Normalize``
slices ``stats.mean[..., :x.shape[-1]]`` and runs *before* the pad to
``max_action_dim``, so narrow stats are both correct and self-documenting;
padding them would bury the real action width.

Dtypes are preserved rather than cast. A LeRobot pi0.5 checkpoint is typically
mixed (fp32 norm/layernorm parameters alongside bf16 weights), and casting the
fp32 ones down loses precision the training run kept.

Usage::

    python -m rlinf.utils.ckpt_convertor.openpi.convert lerobot_to_openpi_pytorch \\
        --input-model       /path/to/lerobot_ckpt \\
        --input-norm-stats  /path/to/policy_preprocessor_step_3_normalizer_processor.safetensors \\
        --output-model      /path/to/out_openpi_pytorch \\
        --asset-id          my-dataset
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from safetensors import safe_open

from rlinf.utils.ckpt_convertor.openpi._core import (
    load_safetensors,
    resolve_model_safetensors,
    save_safetensors,
    strip_wrapper_prefix,
)

# LeRobot feature name -> openpi norm-stats key. openpi's transforms emit "state"
# and "actions"; Normalize/Unnormalize look those up.
DEFAULT_FEATURE_MAP = {"observation.state": "state", "action": "actions"}
STAT_KEYS = ("mean", "std", "q01", "q99")

# Tied in the model: one Parameter reached by two state_dict names. Whichever
# loads last wins, which is harmless -- openpi's forward never uses lm_head.
_TIED_KEYS = (
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
    "paligemma_with_expert.paligemma.lm_head.weight",
)


def convert_weights(
    input_model: str | pathlib.Path, output_model: str | pathlib.Path
) -> dict[str, int]:
    """Strip the LeRobot wrapper prefix, preserving dtypes.

    Returns a small report ``{tensors_in, tensors_out, tied_present}`` so callers
    can assert on it. Raises if two source keys collapse onto one bare key, which
    would silently drop a tensor.
    """
    src = resolve_model_safetensors(input_model)
    state_dict = load_safetensors(src)
    # cast_dtype=None: keep the checkpoint's own dtypes (see module docstring).
    bare = strip_wrapper_prefix(state_dict, cast_dtype=None)

    output_model = pathlib.Path(output_model)
    output_model.mkdir(parents=True, exist_ok=True)
    save_safetensors(bare, output_model / "model.safetensors")

    return {
        "tensors_in": len(state_dict),
        "tensors_out": len(bare),
        "tied_present": sum(1 for k in _TIED_KEYS if k in bare),
    }


def convert_norm_stats(
    lerobot_norm: str | pathlib.Path,
    output_path: str | pathlib.Path,
    feature_map: dict[str, str] | None = None,
) -> dict[str, int]:
    """Translate LeRobot's per-feature stat tensors into openpi's norm_stats.json.

    Returns ``{openpi_key: dim}``. Raises on a missing statistic or on a
    degenerate ``q99 - q01`` span, which would make quantile normalization divide
    by ~zero and emit inf.
    """
    feature_map = feature_map or DEFAULT_FEATURE_MAP
    norm_stats: dict[str, dict[str, list[float]]] = {}
    dims: dict[str, int] = {}

    with safe_open(str(lerobot_norm), framework="np") as f:
        available = set(f.keys())
        for lerobot_feature, openpi_key in feature_map.items():
            missing = [
                s for s in STAT_KEYS if f"{lerobot_feature}.{s}" not in available
            ]
            if missing:
                raise ValueError(
                    f"{lerobot_feature}: missing statistics {missing} in "
                    f"{lerobot_norm}. Present keys: {sorted(available)[:8]}..."
                )
            entry = {
                stat: np.asarray(
                    f.get_tensor(f"{lerobot_feature}.{stat}"), dtype=np.float64
                )
                .reshape(-1)
                .tolist()
                for stat in STAT_KEYS
            }
            span = np.asarray(entry["q99"]) - np.asarray(entry["q01"])
            if (np.abs(span) < 1e-6).any():
                raise ValueError(
                    f"{openpi_key}: degenerate q99-q01 span {span.tolist()}; "
                    "quantile normalization would divide by ~zero."
                )
            norm_stats[openpi_key] = entry
            dims[openpi_key] = len(entry["mean"])

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"norm_stats": norm_stats}, indent=2))
    return dims


def convert(
    input_model: str | pathlib.Path,
    input_norm_stats: str | pathlib.Path,
    output_model: str | pathlib.Path,
    asset_id: str,
    feature_map: dict[str, str] | None = None,
    skip_weights: bool = False,
) -> None:
    """Run the full LeRobot -> OpenPI PyTorch conversion."""
    output_model = pathlib.Path(output_model)
    norm_out = output_model / asset_id / "norm_stats.json"

    dims = convert_norm_stats(input_norm_stats, norm_out, feature_map)
    print(f"norm stats -> {norm_out}  dims={dims}")

    if skip_weights:
        print("weights: skipped")
        return
    report = convert_weights(input_model, output_model)
    print(
        f"weights -> {output_model / 'model.safetensors'}  "
        f"{report['tensors_in']} keys in, {report['tensors_out']} out"
    )
    if report["tied_present"] == 2:
        print(
            "  note: embed_tokens and lm_head are tied to one Parameter; "
            "load order decides the winner and openpi never reads lm_head."
        )


def add_arguments(parser) -> None:
    """Register the ``lerobot_to_openpi_pytorch`` mode arguments."""
    parser.add_argument(
        "--input-model",
        required=True,
        help="LeRobot checkpoint directory or its model.safetensors",
    )
    parser.add_argument(
        "--input-norm-stats",
        required=True,
        help="LeRobot *_normalizer_processor.safetensors holding the per-feature stats",
    )
    parser.add_argument(
        "--output-model", required=True, help="output OpenPI PyTorch checkpoint dir"
    )
    parser.add_argument(
        "--asset-id",
        required=True,
        help=(
            "asset directory for the norm stats, written to "
            "<output-model>/<asset-id>/norm_stats.json; must match the "
            "openpi_data.norm_stats_path the training config points at"
        ),
    )
    parser.add_argument(
        "--feature-map",
        default=None,
        help=(
            "JSON object overriding the LeRobot-feature -> openpi-key mapping, "
            f"default {json.dumps(DEFAULT_FEATURE_MAP)}"
        ),
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="only regenerate norm_stats.json",
    )


def run(args) -> None:
    """Execute ``lerobot_to_openpi_pytorch`` from parsed arguments."""
    feature_map = json.loads(args.feature_map) if args.feature_map else None
    convert(
        input_model=args.input_model,
        input_norm_stats=args.input_norm_stats,
        output_model=args.output_model,
        asset_id=args.asset_id,
        feature_map=feature_map,
        skip_weights=args.skip_weights,
    )
