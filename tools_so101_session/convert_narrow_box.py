# STATUS: ACTIVE — 当前流程在用。 阶段 C2，窄框示范 -> 数据集
"""Convert stratified planner demos (h5, TRUE-task env) -> LeRobot so101-sim-demos-v8.

Images are 160x120 (4:3, matches the real 640x480 through resize_with_pad).
Only successful episodes; unit conversion via so101_calib (same module as RL).
"""
import glob
import json
import shutil
import sys

sys.path.insert(0, "/data08/henryg/pai/RLinf")

import h5py
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from rlinf.envs.maniskill.so101_calib import rad_to_norm

SRC_GLOB = "/data08/henryg/pai/data/v8_demos_w*/**/*.h5"
OUT_REPO = "so101-sim-demos-v8"
OUT_ROOT = "/data08/henryg/pai/data/so101-sim-demos-v8"
TASK = "Grab the red cube"
FPS = 30  # = actual generator control_freq = real dataset fps
MAX_LEN = 580  # budget 640 / 1.1

files = sorted(glob.glob(SRC_GLOB, recursive=True))
print(f"h5 files: {len(files)}")

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

kept, lens = 0, []
for h5path in files:
    meta = json.load(open(h5path.replace(".h5", ".json")))
    ok_ids = [e["episode_id"] for e in meta["episodes"] if e["success"]]
    f = h5py.File(h5path, "r")
    for eid in ok_ids:
        t = f[f"traj_{eid}"]
        qpos = np.asarray(t["obs/agent/qpos"], dtype=np.float64)
        acts = np.asarray(t["actions"], dtype=np.float64)
        front = np.asarray(t["obs/sensor_data/3rd_view_camera/rgb"])
        wrist = np.asarray(t["obs/sensor_data/wrist_camera/rgb"])
        T = acts.shape[0]
        if T > MAX_LEN or T < 80:
            continue
        state_n = rad_to_norm(qpos[:T])
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
        kept += 1
        lens.append(T)
    f.close()
    print(f"{h5path.split('/')[-2]}: kept so far {kept}", flush=True)

lens = np.array(lens)
print(f"DONE: {kept} episodes -> {OUT_ROOT}")
if kept:
    print(f"length median={int(np.median(lens))} p90={int(np.percentile(lens, 90))} max={int(lens.max())}")
