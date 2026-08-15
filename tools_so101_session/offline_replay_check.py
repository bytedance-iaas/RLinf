"""Offline sim2real check: does the sim-trained policy produce sane actions on
REAL images, without touching the robot?

For each sampled frame of a real LeRobot episode we feed the recorded
(front image, wrist image, joint state) to the policy and compare its predicted
action chunk against what the human teleoperator actually did next.

Two controls make the number interpretable:
  * SIM episodes through the same code path -- the in-distribution reference.
  * a "hold still" predictor (action = current state) -- the scale of motion,
    i.e. the error a policy would get by doing nothing.

A policy that is fine on sim but near the hold-still error on real data is
telling you the real observations are out of distribution.

Usage:
  python tools_so101_session/offline_replay_check.py \
      --ckpt /path/to/checkpoint --episodes 6 --frames 40
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/data08/henryg/pai/RLinf")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def build_policy(ckpt: str, config_name: str, norm_stats: str, chunks: int):
    from omegaconf import OmegaConf

    from rlinf.models import get_model

    cfg = OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": ckpt,
            "precision": None,
            "num_action_chunks": chunks,
            "action_dim": 6,
            "num_steps": 4,
            "add_value_head": True,   # RL checkpoints carry one; SFT ones do not (strict=False)
            "policy_setup": "so100",
            "is_lora": False,
            "load_to_device": True,
            "openpi": {
                "config_name": config_name,
                # The YAML configs define this as `${..num_action_chunks}`, so the
                # sim evals ran at 10. A hand-built config like this one does not
                # inherit that and silently falls back to the dataclass default of
                # 5 -- which is what every ratio reported before 2026-08-15 used.
                # Setting it explicitly puts this tool on the same horizon as the
                # sim numbers and as deploy_policy_server.py.
                "action_chunk": chunks,
                "num_images_in_input": 2,
                "action_horizon": 10,
                "noise_method": "flow_noise",
                "value_after_vlm": True,
                "joint_logprob": True,
            },
            "openpi_data": {"norm_stats_path": norm_stats},
        }
    )
    model = get_model(cfg)
    model.eval()
    return model


def load_episode(root: str, idx: int):
    """Return (states, actions, front_reader, wrist_reader, front_offset, wrist_offset).

    Handles BOTH LeRobot layouts:
      v2.0  data/chunk-000/episode_XXXXXX.parquet + one mp4 per episode
      v3.0  data/chunk-000/file-000.parquet (all episodes, `episode_index` column)
            + videos/<key>/chunk-000/file-00N.mp4 holding many episodes back to back,
            with per-episode timestamps in meta/episodes/*.parquet
    The real SO101 dataset (so101-pick-place-v1-trimmed == henry-guo/so101-pick-place-v2,
    87 episodes) is v3.0; the sim datasets we generate are v2.0.
    """
    import imageio.v2 as imageio
    import pandas as pd

    v2_pq = os.path.join(root, "data", "chunk-000", f"episode_{idx:06d}.parquet")
    if os.path.exists(v2_pq):
        df = pd.read_parquet(v2_pq)
        states = np.stack(df["observation.state"].to_numpy())
        actions = np.stack(df["action"].to_numpy())
        front = imageio.get_reader(os.path.join(root, "videos", "chunk-000",
                                   "observation.images.front", f"episode_{idx:06d}.mp4"))
        wrist = imageio.get_reader(os.path.join(root, "videos", "chunk-000",
                                   "observation.images.wrist", f"episode_{idx:06d}.mp4"))
        return states, actions, front, wrist, 0, 0

    # --- v3.0 ---
    meta = pd.read_parquet(glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet"))[0])
    row = meta[meta["episode_index"] == idx].iloc[0]
    data_f = os.path.join(root, "data", f"chunk-{int(row['data/chunk_index']):03d}",
                          f"file-{int(row['data/file_index']):03d}.parquet")
    df = pd.read_parquet(data_f)
    df = df[df["episode_index"] == idx]
    states = np.stack(df["observation.state"].to_numpy())
    actions = np.stack(df["action"].to_numpy())
    readers, offsets = [], []
    for key in ("observation.images.front", "observation.images.wrist"):
        vf = os.path.join(root, "videos", key,
                          f"chunk-{int(row[f'videos/{key}/chunk_index']):03d}",
                          f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4")
        readers.append(imageio.get_reader(vf))
        # episodes are concatenated inside one mp4: convert the episode's start
        # timestamp into a frame offset (fps comes from meta/info.json).
        # The two cameras are chunked into DIFFERENT numbers of files (front
        # 29147 frames per file, wrist 37375), so their offsets differ for most
        # episodes -- an earlier version asserted they were equal, which only
        # held for the first few episodes and silently blocked every later one.
        fps = json.load(open(os.path.join(root, "meta", "info.json")))["fps"]
        offsets.append(int(round(float(row[f"videos/{key}/from_timestamp"]) * fps)))
    return states, actions, readers[0], readers[1], offsets[0], offsets[1]


def evaluate(model, root: str, episodes: int, frames: int, label: str, prompt: str, no_wrist: bool = False, ep_start: int = 0):
    """Mean absolute error (LeRobot normalized units) of predicted vs recorded actions."""
    errs, hold_errs = [], []
    for ep in range(ep_start, ep_start + episodes):
        try:
            states, actions, front, wrist, off_f, off_w = load_episode(root, ep)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] episode {ep} unreadable: {exc}", flush=True)
            continue
        n = min(len(states), front.count_frames() - off_f, wrist.count_frames() - off_w)
        idxs = np.linspace(5, n - 15, frames).astype(int)
        for t in idxs:
            env_obs = {
                "main_images": torch.from_numpy(front.get_data(t + off_f)[None]),
                "wrist_images": None if no_wrist else torch.from_numpy(wrist.get_data(t + off_w)[None]),
                "states": torch.from_numpy(states[t][None].astype(np.float32)),
                "task_descriptions": [prompt],
                # obs_processor indexes this key unconditionally (see
                # openpi_action_model.py:808), so it must be present even when unused
                "extra_view_images": None,
            }
            with torch.no_grad():
                pred, _ = model.predict_action_batch(env_obs=env_obs, mode="eval")
            pred = pred[0].detach().float().cpu().numpy()          # [chunks, 6]
            k = min(len(pred), len(actions) - t)
            errs.append(np.abs(pred[:k] - actions[t : t + k]).mean(axis=0))
            hold_errs.append(np.abs(states[t][None] - actions[t : t + k]).mean(axis=0))
        front.close(); wrist.close()
    if not errs:
        print(f"  [{label}] no usable frames")
        return None
    e = np.stack(errs).mean(axis=0)
    h = np.stack(hold_errs).mean(axis=0)
    print(f"\n=== {label}  ({len(errs)} frames) ===")
    print(f"  {'joint':14} {'policy MAE':>11} {'hold-still MAE':>15} {'ratio':>7}")
    for j, name in enumerate(JOINTS):
        print(f"  {name:14} {e[j]:11.2f} {h[j]:15.2f} {e[j] / max(h[j], 1e-6):7.2f}")
    print(f"  {'MEAN':14} {e.mean():11.2f} {h.mean():15.2f} {e.mean() / max(h.mean(), 1e-6):7.2f}")
    print("  ratio < 1 means the policy beats doing nothing; >= 1 means it does not.")
    return e.mean(), h.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-name", default="pi05_so101_v10")
    ap.add_argument(
        "--norm-stats",
        default="/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json",
    )
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--ep-start", type=int, default=0,
                    help="first episode index to evaluate. Use this to keep the "
                         "check on episodes the model never trained on: once real "
                         "data goes into co-training, evaluating from episode 0 is "
                         "scoring the training set.")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--prompt", default="Grab the red cube")
    ap.add_argument("--no-wrist", action="store_true",
                    help="feed only the front camera (tests whether dropping the "
                         "known-bad wrist channel makes real observations usable "
                         "without retraining)")
    ap.add_argument(
        "--real-root",
        default=None,
        help="LeRobot root of the REAL dataset (defaults to the HF cache snapshot)",
    )
    ap.add_argument(
        "--sim-root",
        default="/data08/henryg/pai/data/so101-sim-demos-v10",
        help="LeRobot root of a SIM dataset, used as the in-distribution control",
    )
    args = ap.parse_args()

    real_root = args.real_root
    if real_root is None:
        hits = glob.glob(
            "/root/.cache/huggingface/hub/datasets--henry-guo--so101-pick-place-v2/snapshots/*/"
        )
        real_root = hits[0] if hits else None
    if real_root is None:
        raise SystemExit("real dataset not found; pass --real-root")

    print(f"checkpoint : {args.ckpt}")
    print(f"real data  : {real_root}")
    print(f"sim control: {args.sim_root}")
    model = build_policy(args.ckpt, args.config_name, args.norm_stats, args.chunks)

    sim = evaluate(model, args.sim_root, args.episodes, args.frames, "SIM (in-distribution control)", args.prompt, args.no_wrist, 0)
    real = evaluate(model, real_root, args.episodes, args.frames, "REAL (the sim2real question)", args.prompt, args.no_wrist, args.ep_start)

    if sim and real:
        print("\n=== verdict ===")
        print(f"  sim  policy/hold-still ratio: {sim[0] / max(sim[1], 1e-6):.2f}")
        print(f"  real policy/hold-still ratio: {real[0] / max(real[1], 1e-6):.2f}")
        print("  If the real ratio is close to 1 while the sim ratio is well below it,")
        print("  the policy is not reading the real observations -- fix that before")
        print("  putting the arm under its control.")


if __name__ == "__main__":
    main()
