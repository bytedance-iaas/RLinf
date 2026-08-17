# STATUS: ACTIVE — 当前流程在用。 阶段 E3，在副本上追加环 1 数据
"""v10 = ring-1 expansion (expert iteration round 2, wider spawn).

Built by APPENDING to a copy of the v9 dataset instead of re-encoding it:
re-encoding 672 episodes of 2x640x480 AV1 costs ~2.75 h of pure CPU and
produces byte-identical video. Verified on 2026-08-13 that a LeRobotDataset
loaded from disk accepts add_frame/save_episode and updates meta correctly
(episode_000672.parquet + both mp4s + info.json counts).

New material appended:
  1. ring-1 policy rollouts from v9_step_1250 (npz, 429 of them)
  2. planner demos generated in the ring-1 ANNULUS only (h5)

UNIT ASYMMETRY (bit us once): recorder npz `state` is ALREADY normalized while
its `action` is in RADIANS; h5 planner demos are radians for BOTH.
"""
import glob
import json
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0, "/data08/henryg/pai/RLinf")
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from rlinf.envs.maniskill.so101_calib import rad_to_norm  # noqa: E402

SRC_ROOT = "/data08/henryg/pai/data/so101-sim-demos-v9"
OUT_REPO, OUT_ROOT = "so101-sim-demos-v10", "/data08/henryg/pai/data/so101-sim-demos-v10"
TASK = "Grab the red cube"
MIN_LEN, MAX_LEN = 80, 580

shutil.rmtree(OUT_ROOT, ignore_errors=True)
shutil.copytree(SRC_ROOT, OUT_ROOT)
ds = LeRobotDataset(repo_id=OUT_REPO, root=OUT_ROOT)
n_base = ds.meta.total_episodes
print(f"base (v9) episodes: {n_base}", flush=True)


def add(state_n, act_n, front, wrist, n_frames):
    for i in range(n_frames):
        ds.add_frame(
            {
                "observation.state": state_n[i].astype(np.float32),
                "action": act_n[i].astype(np.float32),
                "observation.images.front": front[i],
                "observation.images.wrist": wrist[i],
                "task": TASK,
            }
        )
    ds.save_episode()


n_policy = 0
for npz in sorted(glob.glob("/data08/henryg/pai/data/v10_rollouts/*.npz")):
    d = np.load(npz)
    n_frames = d["action"].shape[0]
    if not (MIN_LEN <= n_frames <= MAX_LEN):
        continue
    add(d["state"], rad_to_norm(d["action"].astype(np.float64)), d["main"], d["wrist"], n_frames)
    n_policy += 1
    if n_policy % 50 == 0:
        print(f"ring1 policy rollouts: {n_policy}", flush=True)

n_planner = 0
for h5p in sorted(glob.glob("/data08/henryg/pai/data/v10_demos_w*/**/*.h5", recursive=True)):
    meta = json.load(open(h5p.replace(".h5", ".json")))
    ok = [e["episode_id"] for e in meta["episodes"] if e["success"]]
    f = h5py.File(h5p, "r")
    for eid in ok:
        t = f[f"traj_{eid}"]
        acts = np.asarray(t["actions"], dtype=np.float64)
        n_frames = acts.shape[0]
        if not (MIN_LEN <= n_frames <= MAX_LEN):
            continue
        add(
            rad_to_norm(np.asarray(t["obs/agent/qpos"], dtype=np.float64)[:n_frames]),
            rad_to_norm(acts),
            np.asarray(t["obs/sensor_data/3rd_view_camera/rgb"]),
            np.asarray(t["obs/sensor_data/wrist_camera/rgb"]),
            n_frames,
        )
        n_planner += 1
    f.close()

total = n_base + n_policy + n_planner
print(
    f"DONE: {total} episodes (v9 base {n_base} + ring1 policy {n_policy} + "
    f"annulus planner {n_planner}) -> {OUT_ROOT}"
)
