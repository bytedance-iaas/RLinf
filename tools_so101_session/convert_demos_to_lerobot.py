# STATUS: SUPERSEDED — 早期任务规格，已被主线取代。别用来复现。 最早的通用转换器，已按阶段拆成四个
"""Convert successful ManiSkill scripted demos (h5) -> LeRobot v2.x dataset.

State/action are converted from ManiSkill radians to LeRobot normalized units
via so101_calib.rad_to_norm so the sim dataset matches the real dataset's unit
convention (arm RANGE_M100_100, gripper RANGE_0_100).
"""
import json
import shutil
import sys

sys.path.insert(0, "/data08/henryg/pai/RLinf")

import h5py
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from rlinf.envs.maniskill.so101_calib import rad_to_norm

H5 = "/data08/henryg/pai/data/so101_sim_demos/20260805_220027.h5"
META = "/data08/henryg/pai/data/so101_sim_demos/20260805_220027.json"
OUT_REPO = "so101-sim-demos"
OUT_ROOT = "/data08/henryg/pai/data/so101_sim_demos_lerobot"
TASK = "Grab the red cube"
FPS = 15  # sim control frequency

meta = json.load(open(META))
ok_ids = [e["episode_id"] for e in meta["episodes"] if e["success"]]
print(f"successful episodes: {len(ok_ids)}/{len(meta['episodes'])}")

shutil.rmtree(OUT_ROOT, ignore_errors=True)

features = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
    "observation.images.front": {
        "dtype": "video", "shape": (128, 128, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.wrist": {
        "dtype": "video", "shape": (128, 128, 3),
        "names": ["height", "width", "channel"],
    },
}

ds = LeRobotDataset.create(
    repo_id=OUT_REPO, fps=FPS, root=OUT_ROOT, features=features, use_videos=True
)

f = h5py.File(H5, "r")
for n, eid in enumerate(ok_ids):
    t = f[f"traj_{eid}"]
    qpos = np.asarray(t["obs/agent/qpos"], dtype=np.float64)      # (T+1, 6) rad
    acts = np.asarray(t["actions"], dtype=np.float64)             # (T, 6) rad targets
    front = np.asarray(t["obs/sensor_data/3rd_view_camera/rgb"])  # (T+1,128,128,3)
    wrist = np.asarray(t["obs/sensor_data/wrist_camera/rgb"])
    T = acts.shape[0]
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
    if (n + 1) % 10 == 0:
        print(f"saved {n + 1}/{len(ok_ids)} episodes", flush=True)

print(f"DONE: {len(ok_ids)} episodes -> {OUT_ROOT}")
