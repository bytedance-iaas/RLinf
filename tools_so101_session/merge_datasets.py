"""Merge REAL (henry-guo/so101-pick-place-v2, 87 eps) + SIM v4 (420 eps) into
one LeRobot dataset `so101-mix-v5`.

Legality (skill §5): both are 30 fps, 640x480, same feature schema
(observation.images.front/wrist, observation.state, action) — homogeneity
holds for the FIRST time thanks to the v4 parity work. Plain concatenation
(real appears once; sim dominates ~4:1 by frames — declared choice, no
invented weighting).
"""
import glob
import shutil
import sys

sys.path.insert(0, "/data08/henryg/pai/RLinf")

import cv2
import numpy as np
import pandas as pd

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

OUT_REPO = "so101-mix-v5"
OUT_ROOT = "/data08/henryg/pai/data/so101-mix-v5"
TASK = "Grab the red cube"
FPS = 30

shutil.rmtree(OUT_ROOT, ignore_errors=True)

features = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
    "observation.images.front": {
        "dtype": "video", "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.wrist": {
        "dtype": "video", "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
}

ds = LeRobotDataset.create(
    repo_id=OUT_REPO, fps=FPS, root=OUT_ROOT, features=features, use_videos=True
)

def add_source(root, label):
    kept = 0
    parquets = sorted(glob.glob(f"{root}/data/chunk-*/*.parquet"))
    for pq in parquets:
        df = pd.read_parquet(pq)
        for ep, g in df.groupby("episode_index"):
            g = g.sort_values("frame_index")
            caps = {}
            for cam in ("front", "wrist"):
                vids = glob.glob(f"{root}/videos/chunk-*/observation.images.{cam}/episode_{ep:06d}.mp4")
                if not vids:
                    caps = None
                    break
                caps[cam] = cv2.VideoCapture(vids[0])
            if caps is None:
                continue
            n = len(g)
            ok_all = True
            frames = {"front": [], "wrist": []}
            for cam, cap in caps.items():
                for _ in range(n):
                    ok, fr = cap.read()
                    if not ok:
                        ok_all = False
                        break
                    frames[cam].append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                cap.release()
            if not ok_all or len(frames["front"]) != n or len(frames["wrist"]) != n:
                print(f"skip {label} ep{ep}: frame count mismatch", flush=True)
                continue
            states = np.stack(g["observation.state"].values)
            actions = np.stack(g["action"].values)
            for i in range(n):
                ds.add_frame(
                    {
                        "observation.state": states[i].astype(np.float32),
                        "action": actions[i].astype(np.float32),
                        "observation.images.front": frames["front"][i],
                        "observation.images.wrist": frames["wrist"][i],
                        "task": TASK,
                    }
                )
            ds.save_episode()
            kept += 1
            if kept % 20 == 0:
                print(f"{label}: {kept} episodes merged", flush=True)
    print(f"{label} DONE: {kept} episodes", flush=True)
    return kept

n_real = add_source("/root/.cache/huggingface/lerobot/henry-guo/so101-pick-place-v2", "REAL")

# SIM side: read from the source h5 cells (cv2 cannot decode the AV1 videos in
# the converted dataset; the h5 path is the same proven code as convert_v4).
import h5py, json
from rlinf.envs.maniskill.so101_calib import rad_to_norm
n_sim = 0
for h5path in sorted(glob.glob("/data08/henryg/pai/data/v4_demos_cell_*/**/*.h5", recursive=True)):
    meta = json.load(open(h5path.replace(".h5", ".json")))
    ok_ids = [e["episode_id"] for e in meta["episodes"] if e["success"]]
    f = h5py.File(h5path, "r")
    for eid in ok_ids:
        t = f[f"traj_{eid}"]
        acts = np.asarray(t["actions"], dtype=np.float64)
        T = acts.shape[0]
        if T > 580 or T < 80:
            continue
        qpos = np.asarray(t["obs/agent/qpos"], dtype=np.float64)[:T]
        front = np.asarray(t["obs/sensor_data/3rd_view_camera/rgb"])
        wrist = np.asarray(t["obs/sensor_data/wrist_camera/rgb"])
        state_n = rad_to_norm(qpos)
        act_n = rad_to_norm(acts)
        for i in range(T):
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
        n_sim += 1
        if n_sim % 25 == 0:
            print(f"SIM: {n_sim} episodes merged", flush=True)
    f.close()
print(f"SIM DONE: {n_sim} episodes", flush=True)
print(f"MERGE DONE: real={n_real} sim={n_sim} total={n_real + n_sim} -> {OUT_ROOT}")
