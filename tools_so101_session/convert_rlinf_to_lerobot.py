# STATUS: ACTIVE — 当前流程在用。 部署：RLinf 检查点 -> LeRobot 格式
"""Export an RLinf openpi PI0.5 checkpoint as a LeRobot-format pi05 checkpoint.

Why: LeRobot's async inference stack (`lerobot.async_inference.policy_server` +
`robot_client`) loads policies with `policy_class.from_pretrained(path)`, which
only reads the LeRobot layout. Converting means the robot laptop can run the
stock `robot_client` -- with its action queue, early re-request and overlapping
chunk aggregation -- instead of a hand-written control loop.

Why it is nearly free: RLinf's openpi backend IS the LeRobot pi05 module tree.
Verified on so101_sft_v15/global_step_1000 against checkpoints/lerobot_pi05_base:
all 812 LeRobot keys present with identical shapes, our file carrying exactly one
extra (`...language_model.embed_tokens.weight`, which LeRobot dedups because it
is tied to lm_head). No tensor is reshaped, split or merged here.

What must NOT be copied blindly is normalization. LeRobot bakes dataset stats
into the processor safetensors; the source checkpoint's stats come from ITS
dataset, while our policy was trained under the frozen v4 sim lineage. Using the
wrong stats changes the coordinate system the policy speaks in and fails
silently. This script writes our norm_stats into the processor files.

    python tools_so101_session/convert_rlinf_to_lerobot.py \
        --ckpt   /data08/henryg/pai/results/so101_sft_v15/so101_sft_openpi_pi05/checkpoints/global_step_1000 \
        --template /data08/henryg/pai/outputs/train/2026-07-14/13-53-42_pi05/checkpoints/last/pretrained_model \
        --norm-stats /data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
        --chunk-size 10 \
        --out /data08/henryg/pai/results/so101_v15_lerobot

The template supplies config.json and the processor JSONs. Use a LeRobot pi05
checkpoint fine-tuned on the SAME robot, so the feature names and dimensions
(observation.images.front / .wrist, observation.state (6), action (6)) already
match what robot_client sends.

THIS SCRIPT DOES NOT PROVE THE CONVERSION IS CORRECT. Run the offline check
against the converted policy and confirm it reproduces the RLinf number
(0.70 held-out for v15/step_1000). Two things it would catch that a key-set
comparison cannot: a quantile-normalisation convention that differs between
openpi and LeRobot, and the fact that RLinf's model carries a `noise_head` the
LeRobot model has no slot for.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# Fields the RLinf model adds on top of a pi0.5 and that LeRobot has no slot for.
# add_value_head / flow_noise create these; they are RL machinery, not the policy.
EXTRA_PREFIXES = ("value_head.", "noise_head.", "q_head.", "actor_image_encoder.",
                  "actor_state_encoder.", "critic_image_encoder.", "critic_state_encoder.")
# Tied to paligemma.lm_head.weight. Some LeRobot checkpoints store it and some
# dedup it away (safetensors refuses shared storage), so follow the template.
TIED_KEY = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"


def common_prefix(keys) -> str:
    """LeRobot wraps the flow model, so its checkpoints carry a uniform prefix
    ('model.'); RLinf saves the inner module bare. Read the prefix off the
    template rather than assuming which side has it."""
    keys = list(keys)
    for pre in ("model.", ""):
        if pre and all(k.startswith(pre) for k in keys):
            return pre
    return ""


def load_rlinf_weights(ckpt: Path) -> dict:
    for rel in ("actor/model_state_dict/full_weights.pt", "model_state_dict/full_weights.pt"):
        p = ckpt / rel
        if p.exists():
            print(f"loading {p}")
            return torch.load(p, map_location="cpu", weights_only=True)
    raise SystemExit(f"no full_weights.pt under {ckpt}")


def strip_wrappers(sd: dict) -> dict:
    """FSDP/compile wrappers can leave prefixes behind. Ours does not, but a
    checkpoint written by a different strategy would. 'model.' is deliberately
    NOT stripped here -- it is LeRobot's own wrapper and is re-applied later to
    match the template."""
    out = {}
    for k, v in sd.items():
        for pre in ("_fsdp_wrapped_module.", "_orig_mod.", "module."):
            while k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="RLinf checkpoint dir (contains actor/)")
    ap.add_argument("--template", required=True,
                    help="LeRobot pi05 checkpoint for the SAME robot; supplies "
                         "config.json and the processor JSONs")
    ap.add_argument("--norm-stats", required=True,
                    help="openpi norm_stats.json of the lineage the policy was trained under")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-size", type=int, default=10,
                    help="actions per chunk. Must be the horizon the policy was "
                         "fine-tuned at -- asking a 10-step policy for 50 steps "
                         "returns 40 steps it never saw a target for")
    ap.add_argument("--num-steps", type=int, default=4,
                    help="flow-matching denoising steps at inference. MUST match "
                         "what produced the numbers you are validating against "
                         "(RLinf model config `num_steps`, 4 here); LeRobot's own "
                         "default is 10 and yields a measurably different policy")
    ap.add_argument("--dtype", choices=["keep", "float32", "bfloat16"], default="keep",
                    help="'keep' preserves the exact dtypes the measured numbers "
                         "were produced with (mixed bf16/fp32)")
    args = ap.parse_args()

    ckpt, template, out = Path(args.ckpt), Path(args.template), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. weights ----------------------------------------------------------
    sd = strip_wrappers(load_rlinf_weights(ckpt))
    dropped_extra = [k for k in sd if k.startswith(EXTRA_PREFIXES)]
    for k in dropped_extra:
        del sd[k]
    if args.dtype != "keep":
        target = torch.float32 if args.dtype == "float32" else torch.bfloat16
        sd = {k: v.to(target) for k, v in sd.items()}

    with safe_open(template / "model.safetensors", "pt") as f:
        want = set(f.keys())
        want_shapes = {k: tuple(f.get_slice(k).get_shape()) for k in want}
    prefix = common_prefix(want)
    print(f"template key prefix: {prefix!r}")

    # Follow the template on the tied embedding: some LeRobot checkpoints keep it,
    # some dedup it away and let the module re-tie on load.
    dropped_tied = False
    if prefix + TIED_KEY not in want and TIED_KEY in sd:
        del sd[TIED_KEY]
        dropped_tied = True

    sd = {prefix + k: v for k, v in sd.items()}
    have = set(sd)
    missing, unexpected = sorted(want - have), sorted(have - want)
    bad_shape = [k for k in sorted(want & have) if tuple(sd[k].shape) != want_shapes[k]]

    print(f"dropped RL-only tensors: {len(dropped_extra)}  dropped tied embed_tokens: {dropped_tied}")
    print(f"keys: template {len(want)}, converted {len(have)}, "
          f"missing {len(missing)}, unexpected {len(unexpected)}, shape mismatch {len(bad_shape)}")
    for label, keys in (("MISSING", missing), ("UNEXPECTED", unexpected), ("SHAPE", bad_shape)):
        for k in keys[:10]:
            print(f"  {label} {k}")
    if missing or unexpected or bad_shape:
        raise SystemExit("refusing to write: key sets do not match the template")

    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f"wrote {out/'model.safetensors'}")

    # ---- 2. config -----------------------------------------------------------
    cfg = json.loads((template / "config.json").read_text())
    before = (cfg.get("chunk_size"), cfg.get("n_action_steps"))
    cfg["chunk_size"] = cfg["n_action_steps"] = args.chunk_size
    # Flow matching integrates an ODE at inference; the number of steps changes
    # the action for identical weights and inputs. RLinf runs 4 (model config
    # `num_steps`), LeRobot's PI05 defaults to 10. Leaving the template's value
    # in place makes the export a DIFFERENT policy -- measured: every joint's
    # error ~26% worse, gate 0.70 -> 0.88.
    before_steps = cfg.get("num_inference_steps")
    cfg["num_inference_steps"] = args.num_steps
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"config: num_inference_steps {before_steps} -> {args.num_steps}")
    print(f"config: chunk_size/n_action_steps {before} -> {args.chunk_size}; "
          f"features {list(cfg.get('input_features', {}))}")
    if (template / "train_config.json").exists():
        shutil.copy(template / "train_config.json", out / "train_config.json")

    # ---- 3. normalization ----------------------------------------------------
    # LeRobot keeps per-feature stats as flat "<feature>.<stat>" tensors. pi05
    # normalises STATE and ACTION by QUANTILES, so q01/q99 are the ones that
    # actually act; mean/std are written too so the file stays self-consistent.
    stats = json.loads(Path(args.norm_stats).read_text())["norm_stats"]
    src = {"observation.state": stats["state"], "action": stats["actions"]}
    mapping = cfg.get("normalization_mapping", {})
    print(f"normalization_mapping = {mapping}")
    if mapping.get("STATE") != "QUANTILES" or mapping.get("ACTION") != "QUANTILES":
        print("  WARNING: template does not use QUANTILES for state/action; the "
              "stats written below may not be the ones the policy reads")

    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if (template / name).exists():
            shutil.copy(template / name, out / name)
    for f in sorted(template.glob("policy_*processor*.safetensors")):
        tensors = load_file(str(f))
        touched = []
        for feat, s in src.items():
            for stat in ("q01", "q99", "mean", "std"):
                key = f"{feat}.{stat}"
                if key in tensors and stat in s:
                    new = torch.tensor(s[stat], dtype=tensors[key].dtype)
                    if new.shape != tensors[key].shape:
                        raise SystemExit(f"{f.name}: {key} shape {tuple(new.shape)} "
                                         f"!= template {tuple(tensors[key].shape)}")
                    tensors[key] = new
                    touched.append(key)
        save_file(tensors, str(out / f.name), metadata={"format": "pt"})
        print(f"wrote {f.name}: replaced {len(touched)} stats "
              f"({', '.join(sorted(set(k.split('.')[-1] for k in touched)))})")

    print(f"\nDONE: {out}")
    print("NOT YET VERIFIED -- run the offline check against this directory and "
          "confirm it reproduces the RLinf ratio before putting it on a robot.")


if __name__ == "__main__":
    main()
