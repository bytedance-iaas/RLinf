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

"""Export an RLinf SFT pi0.5 checkpoint as a LeRobot ``pi05`` checkpoint.

This is the deployment direction, the inverse of ``lerobot_to_openpi_pytorch``.
LeRobot's async inference stack (``lerobot.async_inference.policy_server`` plus
``robot_client``) loads policies through ``policy_class.from_pretrained(path)``,
which only reads the LeRobot layout. Converting lets a robot run the stock
client -- with its action queue, early re-request and overlapping chunk
aggregation -- instead of a hand-written control loop.

The weights themselves need almost nothing done to them: RLinf's openpi backend
*is* the LeRobot pi05 module tree, so the tensor half is pure key renaming and
round-trips bit-identically. What needs care is the metadata travelling alongside
them: the following can be *carried over wrong* rather than *converted wrong*,
and each fails silently rather than loudly.

* **RL-only tensors.** ``add_value_head`` and the flow-noise machinery add
  ``value_head.*`` / ``noise_head.*`` and friends, which the LeRobot model has no
  slot for. They are dropped.
* **The tied embedding.** ``embed_tokens.weight`` is tied to ``lm_head.weight``;
  some LeRobot checkpoints store it and some deduplicate it away (safetensors
  refuses shared storage). The template decides which.
* **Inference steps.** Flow matching integrates an ODE at inference, so the step
  count changes the action for identical weights and inputs. RLinf runs 4; the
  LeRobot PI05 default is 10. Leaving the template's value in place exports a
  *different policy* -- measured on SO101: every joint's error ~26% worse.
* **Normalization.** LeRobot bakes dataset statistics into the processor
  safetensors. The template's stats come from *its* dataset, not from the lineage
  the exported policy was trained under; using them changes the coordinate system
  the policy speaks in. This mode rewrites them from ``--norm-stats``.

The key set is checked against the template and the conversion refuses to write
on any mismatch, so a silently truncated export cannot reach a robot.

**A round trip does not prove the exported policy is correct.** Both directions
are verified lossless, but that only shows the two mappings are mutually inverse;
a quantile convention that differs between openpi and LeRobot would cancel out
across a round trip and still be wrong on a robot. Run an offline check against
the output and confirm it reproduces the source checkpoint's numbers before
putting it on hardware.

Usage::

    python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft_to_lerobot \\
        --ckpt        /path/to/checkpoints/global_step_1000 \\
        --template    /path/to/lerobot_pi05_same_robot \\
        --norm-stats  /path/to/norm_stats.json \\
        --chunk-size  10 \\
        --num-steps   4 \\
        --output      /path/to/out_lerobot
"""

from __future__ import annotations

import json
import pathlib
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from rlinf.utils.ckpt_convertor.openpi._core import save_safetensors

# Heads RLinf adds on top of a pi0.5 that LeRobot has no slot for: RL machinery,
# not the policy.
EXTRA_PREFIXES = (
    "value_head.",
    "noise_head.",
    "q_head.",
    "actor_image_encoder.",
    "actor_state_encoder.",
    "critic_image_encoder.",
    "critic_state_encoder.",
)
# Tied to paligemma.lm_head.weight; the template decides whether it is stored.
TIED_KEY = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
# Wrappers a non-default training strategy may leave behind. LeRobot's own
# "model." wrapper is deliberately absent here: it is re-applied from the
# template later, not stripped as noise.
_TRAINING_WRAPPERS = ("_fsdp_wrapped_module.", "_orig_mod.", "module.")


def _template_prefix(keys) -> str:
    """Read LeRobot's wrapper prefix off the template instead of assuming it."""
    keys = list(keys)
    if keys and all(k.startswith("model.") for k in keys):
        return "model."
    return ""


def load_sft_weights(ckpt: str | pathlib.Path) -> dict[str, torch.Tensor]:
    """Load ``full_weights.pt`` from an RLinf checkpoint directory."""
    ckpt = pathlib.Path(ckpt)
    for rel in (
        "actor/model_state_dict/full_weights.pt",
        "model_state_dict/full_weights.pt",
    ):
        if (ckpt / rel).exists():
            return torch.load(ckpt / rel, map_location="cpu", weights_only=True)
    if ckpt.is_file():
        return torch.load(ckpt, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no full_weights.pt under {ckpt}")


def strip_training_wrappers(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Drop FSDP/compile prefixes, leaving LeRobot's own ``model.`` alone."""
    out = {}
    for key, tensor in state_dict.items():
        for prefix in _TRAINING_WRAPPERS:
            while key.startswith(prefix):
                key = key[len(prefix) :]
        out[key] = tensor
    return out


def convert_weights(
    ckpt: str | pathlib.Path,
    template: str | pathlib.Path,
    output: str | pathlib.Path,
    dtype: str = "keep",
) -> dict[str, int]:
    """Write ``model.safetensors`` in the template's key namespace.

    Raises if the converted key set does not match the template exactly, so a
    partial export cannot reach a robot.
    """
    template = pathlib.Path(template)
    output = pathlib.Path(output)

    state_dict = strip_training_wrappers(load_sft_weights(ckpt))
    dropped_extra = [k for k in state_dict if k.startswith(EXTRA_PREFIXES)]
    for key in dropped_extra:
        del state_dict[key]

    if dtype != "keep":
        target = torch.float32 if dtype == "float32" else torch.bfloat16
        state_dict = {k: v.to(target) for k, v in state_dict.items()}

    with safe_open(str(template / "model.safetensors"), "pt") as f:
        want = set(f.keys())
        want_shapes = {k: tuple(f.get_slice(k).get_shape()) for k in want}
    prefix = _template_prefix(want)

    dropped_tied = False
    if prefix + TIED_KEY not in want and TIED_KEY in state_dict:
        del state_dict[TIED_KEY]
        dropped_tied = True

    state_dict = {prefix + k: v for k, v in state_dict.items()}
    have = set(state_dict)
    missing, unexpected = sorted(want - have), sorted(have - want)
    bad_shape = [
        k for k in sorted(want & have) if tuple(state_dict[k].shape) != want_shapes[k]
    ]
    if missing or unexpected or bad_shape:
        raise ValueError(
            "refusing to write: key sets do not match the template "
            f"(missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatch={bad_shape[:5]})"
        )

    output.mkdir(parents=True, exist_ok=True)
    save_safetensors(state_dict, output / "model.safetensors")
    return {
        "tensors": len(state_dict),
        "dropped_rl_heads": len(dropped_extra),
        "dropped_tied": int(dropped_tied),
    }


def convert_config(
    template: str | pathlib.Path,
    output: str | pathlib.Path,
    chunk_size: int,
    num_steps: int,
) -> dict:
    """Copy the template config, overriding the two fields that change behavior."""
    template, output = pathlib.Path(template), pathlib.Path(output)
    cfg = json.loads((template / "config.json").read_text())
    cfg["chunk_size"] = cfg["n_action_steps"] = chunk_size
    cfg["num_inference_steps"] = num_steps
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    if (template / "train_config.json").exists():
        shutil.copy(template / "train_config.json", output / "train_config.json")
    return cfg


def convert_norm_stats(
    template: str | pathlib.Path,
    output: str | pathlib.Path,
    norm_stats: str | pathlib.Path,
    cfg: dict | None = None,
) -> list[str]:
    """Rewrite the template's processor stats from an openpi ``norm_stats.json``.

    pi0.5 normalizes state and action by QUANTILES, so q01/q99 are the ones that
    act; mean/std are written too to keep the file self-consistent. Returns the
    names of the processor files written.
    """
    template, output = pathlib.Path(template), pathlib.Path(output)
    stats = json.loads(pathlib.Path(norm_stats).read_text())["norm_stats"]
    source = {"observation.state": stats["state"], "action": stats["actions"]}

    mapping = (cfg or {}).get("normalization_mapping", {})
    if mapping and (
        mapping.get("STATE") != "QUANTILES" or mapping.get("ACTION") != "QUANTILES"
    ):
        print(
            "  WARNING: template does not use QUANTILES for state/action; the "
            "stats written below may not be the ones the policy reads"
        )

    output.mkdir(parents=True, exist_ok=True)
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if (template / name).exists():
            shutil.copy(template / name, output / name)

    written = []
    for path in sorted(template.glob("policy_*processor*.safetensors")):
        tensors = load_file(str(path))
        for feature, entry in source.items():
            for stat in ("q01", "q99", "mean", "std"):
                key = f"{feature}.{stat}"
                if key in tensors and stat in entry:
                    new = torch.tensor(entry[stat], dtype=tensors[key].dtype)
                    if new.shape != tensors[key].shape:
                        raise ValueError(
                            f"{path.name}: {key} shape {tuple(new.shape)} != "
                            f"template {tuple(tensors[key].shape)}"
                        )
                    tensors[key] = new
        save_safetensors(tensors, output / path.name)
        written.append(path.name)
    return written


def convert(
    ckpt: str | pathlib.Path,
    template: str | pathlib.Path,
    norm_stats: str | pathlib.Path,
    output: str | pathlib.Path,
    chunk_size: int = 10,
    num_steps: int = 4,
    dtype: str = "keep",
) -> None:
    """Run the full RLinf SFT -> LeRobot export."""
    report = convert_weights(ckpt, template, output, dtype=dtype)
    print(
        f"weights: {report['tensors']} tensors, dropped "
        f"{report['dropped_rl_heads']} RL-only heads, "
        f"tied embed dropped={bool(report['dropped_tied'])}"
    )
    cfg = convert_config(template, output, chunk_size, num_steps)
    print(f"config: chunk_size={chunk_size} num_inference_steps={num_steps}")
    written = convert_norm_stats(template, output, norm_stats, cfg)
    print(f"norm stats rewritten into: {', '.join(written) or '(none found)'}")
    print(
        "NOT YET VERIFIED -- run an offline check against this directory and "
        "confirm it reproduces the source checkpoint's numbers before putting "
        "it on hardware."
    )


def add_arguments(parser) -> None:
    """Register the ``sft_to_lerobot`` mode arguments."""
    parser.add_argument(
        "--ckpt", required=True, help="RLinf checkpoint directory (contains actor/)"
    )
    parser.add_argument(
        "--template",
        required=True,
        help=(
            "LeRobot pi05 checkpoint for the SAME robot; supplies config.json "
            "and the processor JSONs, and defines the target key set"
        ),
    )
    parser.add_argument(
        "--norm-stats",
        required=True,
        help="openpi norm_stats.json of the lineage the policy was trained under",
    )
    parser.add_argument("--output", required=True, help="output LeRobot checkpoint dir")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help=(
            "actions per chunk; must be the horizon the policy was fine-tuned at "
            "-- asking a 10-step policy for 50 returns 40 it never saw a target for"
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=4,
        help=(
            "flow-matching denoising steps at inference; must match the RLinf "
            "model config (4), not LeRobot's default of 10, which yields a "
            "measurably different policy"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["keep", "float32", "bfloat16"],
        default="keep",
        help="'keep' preserves the checkpoint's own mixed bf16/fp32 dtypes",
    )


def run(args) -> None:
    """Execute ``sft_to_lerobot`` from parsed arguments."""
    convert(
        ckpt=args.ckpt,
        template=args.template,
        norm_stats=args.norm_stats,
        output=args.output,
        chunk_size=args.chunk_size,
        num_steps=args.num_steps,
        dtype=args.dtype,
    )
