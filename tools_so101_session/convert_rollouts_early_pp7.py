# STATUS: SUPERSEDED — 早期任务规格，已被主线取代。别用来复现。 pp 时代 rollout 转换
"""Convert on-policy SUCCESS rollouts (npz from SO101_COLLECT_DIR) -> LeRobot dataset.

npz layout (per episode, written by ManiskillEnv recorder):
  main   (T,128,128,3) uint8  - policy's front view
  wrist  (T,128,128,3) uint8  - policy's wrist view
  state  (T,6) float32        - ALREADY LeRobot-normalized (so101_state_norm)
  action (T,6) float32        - env units = RADIANS -> convert via rad_to_norm
"""
import glob
import shutil
import sys

sys.path.insert(0, "/data08/henryg/pai/RLinf")

import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from rlinf.envs.maniskill.so101_calib import rad_to_norm

SRC = ["/data08/henryg/pai/data/so101_pp5_rollouts", "/data08/henryg/pai/data/so101_pp6_rollouts", "/data08/henryg/pai/data/so101_pp7_hard"]
OUT_REPO = "so101-sim-demos-pp7"
OUT_ROOT = "/data08/henryg/pai/data/so101-sim-demos-pp7"
TASK = "Grab the red cube"
FPS = 15
MIN_LEN, MAX_LEN = 30, 220  # sanity bounds (success episodes; 240 budget)

files = sorted(f for d in SRC for f in glob.glob(d + "/*.npz"))
print(f"found {len(files)} episodes")

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

kept, lens = 0, []
for path in files:
    d = np.load(path)
    T = d["action"].shape[0]
    if not (MIN_LEN <= T <= MAX_LEN):
        print(f"skip {path}: len {T} outside [{MIN_LEN},{MAX_LEN}]")
        continue
    act_n = rad_to_norm(d["action"].astype(np.float64))
    state_n = d["state"]
    front, wrist = d["main"], d["wrist"]
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
    if kept % 25 == 0:
        print(f"saved {kept} episodes", flush=True)

lens = np.array(lens)
print(f"DONE: {kept} episodes -> {OUT_ROOT}")
print(f"length median={int(np.median(lens))} p90={int(np.percentile(lens, 90))} max={int(lens.max())}")
