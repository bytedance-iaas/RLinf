"""Equivalence test for convert_rlinf_to_lerobot.py.

A key-set match proves the tensors landed in the right slots. It does not prove
the exported checkpoint BEHAVES like the one we measured, because behaviour also
depends on normalisation stats and on the preprocessing each stack applies. The
only honest check is to re-run the metric that authorised the real-robot trial
and see the same number.

So this runs the offline sim2real check again -- same held-out real episodes,
same frames, same hold-still control -- but through LeRobot's PI05Policy and its
processor pipeline instead of RLinf's. It reuses `load_episode` from
offline_replay_check.py so the data path is identical and only the policy differs.

Run it in the LeRobot environment (the RLinf venv has lerobot 0.1.0, which has
no pi05 policy):

    /root/miniconda3/envs/lerobot/bin/python tools_so101_session/verify_lerobot_export.py \
        --lerobot-ckpt /data08/henryg/pai/results/so101_v15_lerobot \
        --real-root /data08/henryg/pai/data/so101-pick-place-v1-trimmed \
        --ep-start 70 --episodes 5 --frames 10

PASS means the real ratio lands on the RLinf number for the same checkpoint
(0.70 for so101_sft_v15/global_step_1000 on held-out episodes 70-86). A clearly
worse ratio means the export changed the policy -- most likely the normalisation
stats -- and it must not go on the robot.
"""

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/data08/henryg/pai/RLinf/tools_so101_session")

from offline_replay_check import JOINTS, load_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-ckpt", required=True)
    ap.add_argument("--real-root", required=True)
    ap.add_argument("--ep-start", type=int, default=70)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--prompt", default="Grab the red cube")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reference", type=float, default=0.70,
                    help="the RLinf-measured ratio for the same checkpoint")
    ap.add_argument("--tolerance", type=float, default=0.10,
                    help="absolute ratio difference still considered equivalent")
    args = ap.parse_args()

    from lerobot.policies import get_policy_class, make_pre_post_processors

    policy = get_policy_class("pi05").from_pretrained(args.lerobot_ckpt)
    policy.to(args.device).eval()
    override = {"device": args.device}
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=args.lerobot_ckpt,
        preprocessor_overrides={"device_processor": override},
        postprocessor_overrides={"device_processor": override},
    )
    print(f"loaded {args.lerobot_ckpt} | chunk_size={policy.config.chunk_size}", flush=True)

    errs, hold_errs = [], []
    for ep in range(args.ep_start, args.ep_start + args.episodes):
        try:
            states, actions, front, wrist, off_f, off_w = load_episode(args.real_root, ep)
        except Exception as exc:  # noqa: BLE001
            print(f"  episode {ep} unreadable: {exc}", flush=True)
            continue
        n = min(len(states), front.count_frames() - off_f, wrist.count_frames() - off_w)
        for t in np.linspace(5, n - 15, args.frames).astype(int):
            # Same shapes robot_client produces: images (B, C, H, W) float in
            # [0, 1] at native resolution -- PI05 resizes with padding internally
            # (modeling_pi05.py:1191), exactly as openpi does.
            def img(reader, idx):
                a = torch.from_numpy(reader.get_data(idx).copy()).permute(2, 0, 1)
                return (a.float() / 255.0).unsqueeze(0)

            obs = {
                "observation.images.front": img(front, t + off_f),
                "observation.images.wrist": img(wrist, t + off_w),
                "observation.state": torch.from_numpy(states[t][None].astype(np.float32)),
                "task": args.prompt,
            }
            with torch.no_grad():
                chunk = policy.predict_action_chunk(pre(obs))
                if chunk.ndim != 3:
                    chunk = chunk.unsqueeze(0)
                chunk = post(chunk)
            pred = chunk[0].detach().float().cpu().numpy()
            k = min(len(pred), len(actions) - t)
            errs.append(np.abs(pred[:k] - actions[t : t + k]).mean(axis=0))
            hold_errs.append(np.abs(states[t][None] - actions[t : t + k]).mean(axis=0))
        front.close(); wrist.close()

    if not errs:
        raise SystemExit("no usable frames")
    e, h = np.stack(errs).mean(axis=0), np.stack(hold_errs).mean(axis=0)
    print(f"\n=== LeRobot export on held-out real episodes ({len(errs)} frames) ===")
    print(f"  {'joint':14} {'policy MAE':>11} {'hold-still MAE':>15} {'ratio':>7}")
    for j, name in enumerate(JOINTS):
        print(f"  {name:14} {e[j]:11.2f} {h[j]:15.2f} {e[j] / max(h[j], 1e-6):7.2f}")
    ratio = e.mean() / max(h.mean(), 1e-6)
    print(f"  {'MEAN':14} {e.mean():11.2f} {h.mean():15.2f} {ratio:7.2f}")

    delta = abs(ratio - args.reference)
    print(f"\nreal policy/hold-still ratio: {ratio:.2f}  "
          f"(RLinf measured {args.reference:.2f}, delta {delta:.2f})")
    if delta <= args.tolerance:
        print("EXPORT OK: the LeRobot checkpoint reproduces the RLinf policy.")
    else:
        print("EXPORT MISMATCH: do NOT deploy this. Check the normalisation stats "
              "written into the processor safetensors first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
