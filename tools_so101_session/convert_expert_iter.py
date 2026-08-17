# STATUS: ACTIVE — 当前流程在用。 阶段 D2，规划器 h5 + 策略 npz 混合
"""v9 = expert iteration round 1 in the legacy box.
Pools: (1) the 247 planner demos (h5, v8 source) and (2) SUCCESSFUL policy
rollouts collected from v8_step_2500 (npz). iRe-VLA: mixing the original expert
data with new on-policy successes prevents the mode narrowing that pure
self-distillation causes.
"""
import glob, json, shutil, sys
sys.path.insert(0, "/data08/henryg/pai/RLinf")
import h5py, numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from rlinf.envs.maniskill.so101_calib import rad_to_norm

OUT_REPO, OUT_ROOT = "so101-sim-demos-v9", "/data08/henryg/pai/data/so101-sim-demos-v9"
TASK, FPS = "Grab the red cube", 30
MIN_LEN, MAX_LEN = 80, 580
shutil.rmtree(OUT_ROOT, ignore_errors=True)
features = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
    "observation.images.front": {"dtype": "video", "shape": (480, 640, 3), "names": ["height","width","channel"]},
    "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["height","width","channel"]},
}
ds = LeRobotDataset.create(repo_id=OUT_REPO, fps=FPS, root=OUT_ROOT, features=features, use_videos=True)

def add(state_n, act_n, front, wrist, T):
    for i in range(T):
        ds.add_frame({"observation.state": state_n[i].astype(np.float32),
                      "action": act_n[i].astype(np.float32),
                      "observation.images.front": front[i],
                      "observation.images.wrist": wrist[i],
                      "task": TASK})
    ds.save_episode()

n_planner = 0
for h5p in sorted(glob.glob("/data08/henryg/pai/data/v8_demos_w*/**/*.h5", recursive=True)):
    meta = json.load(open(h5p.replace(".h5", ".json")))
    ok = [e["episode_id"] for e in meta["episodes"] if e["success"]]
    f = h5py.File(h5p, "r")
    for eid in ok:
        t = f[f"traj_{eid}"]
        acts = np.asarray(t["actions"], dtype=np.float64); T = acts.shape[0]
        if not (MIN_LEN <= T <= MAX_LEN):
            continue
        add(rad_to_norm(np.asarray(t["obs/agent/qpos"], dtype=np.float64)[:T]),
            rad_to_norm(acts),
            np.asarray(t["obs/sensor_data/3rd_view_camera/rgb"]),
            np.asarray(t["obs/sensor_data/wrist_camera/rgb"]), T)
        n_planner += 1
    f.close()
print(f"planner demos: {n_planner}", flush=True)

n_policy = 0
for npz in sorted(glob.glob("/data08/henryg/pai/data/v9_rollouts/*.npz")):
    d = np.load(npz); T = d["action"].shape[0]
    if not (MIN_LEN <= T <= MAX_LEN):
        continue
    add(d["state"], rad_to_norm(d["action"].astype(np.float64)), d["main"], d["wrist"], T)
    n_policy += 1
    if n_policy % 50 == 0:
        print(f"policy rollouts: {n_policy}", flush=True)
print(f"DONE: {n_planner + n_policy} episodes (planner {n_planner} + policy {n_policy}) -> {OUT_ROOT}")
