"""With wrist_flex +0.6 baked in, re-sweep the wrist-camera mount pose so the sim
wrist view looks forward-down at the workspace (matching the real wrist cam)."""
import glob
import os
import numpy as np
import pandas as pd
import torch
import imageio.v2 as imageio
import gymnasium as gym
import rlinf.envs.maniskill.tasks.so101_pick_place as T
import rlinf.envs.maniskill.so101_calib as C
from rlinf.envs.maniskill import import_all_tasks

OUT = os.environ.get("SCRATCH", "/tmp/so101_runs")
DS = "/root/.cache/huggingface/lerobot/henry-guo/so101-pick-place-v2"


def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


ep = pd.read_parquet(sorted(glob.glob(f"{DS}/data/**/*.parquet", recursive=True))[0])
ep = ep[ep["episode_index"] == 0]
actions = np.stack(ep["action"].values)[::5]
init_state = np.stack(ep["observation.state"].values)[0]
mid = len(actions) // 2  # a reach-phase frame to render each candidate at

# candidate wrist-cam (eye, target) in Fixed_Jaw frame — aim forward-down variants
cands = [
    ([0.0, 0.0, 0.04], [0.0, -0.05, -0.18]),
    ([0.0, 0.02, 0.04], [0.0, -0.02, -0.20]),
    ([0.0, -0.02, 0.05], [0.0, -0.12, -0.12]),
    ([0.0, 0.0, 0.06], [0.0, 0.05, -0.18]),
    ([0.0, 0.03, 0.05], [0.0, 0.10, -0.15]),
    ([0.0, 0.0, 0.03], [0.0, 0.15, -0.10]),
]

import_all_tasks()
tiles = []
for i, (eye, tgt) in enumerate(cands):
    T.WRIST_CAM_EYE = eye
    T.WRIST_CAM_TARGET = tgt
    T.WRIST_CAM_FOV = 1.5
    env = gym.make("SO101GrabRedCube-v1", num_envs=1, obs_mode="rgb",
                   control_mode="pd_joint_pos", sim_backend="gpu", render_mode="rgb_array")
    dev = env.unwrapped.device
    env.reset(seed=0)
    env.unwrapped.agent.robot.set_qpos(
        torch.tensor(C.norm_to_rad(init_state), dtype=torch.float32, device=dev).reshape(1, -1))
    w = None
    for j, a in enumerate(actions):
        obs, *_ = env.step(torch.tensor(C.norm_to_rad(a), dtype=torch.float32, device=dev).reshape(1, -1))
        if j == mid:
            w = to_np(obs["sensor_data"]["wrist_camera"]["rgb"][0]).astype(np.uint8)
    tiles.append(w)
    env.close()
    print("cand", i, eye, tgt, flush=True)

# 2x3 grid + a real wrist frame at the same timestep for reference
real = [np.asarray(x) for x in imageio.get_reader(f"{DS}/videos/chunk-000/observation.images.wrist/episode_000000.mp4")]
def rs(img, H=180):
    h, w = img.shape[:2]; W = int(w * H / h); return img[(np.arange(H) * h // H)][:, (np.arange(W) * w // W)]
realref = rs(real[min(len(real) - 1, mid * 5)])
k = 2
gt = [np.repeat(np.repeat(t, k, 0), k, 1) for t in tiles]
g = 6
vgap = np.full((gt[0].shape[0], g, 3), 255, np.uint8)
row0 = np.concatenate([gt[0], vgap, gt[1], vgap, gt[2]], axis=1)
row1 = np.concatenate([gt[3], vgap, gt[4], vgap, gt[5]], axis=1)
hgap = np.full((g, row0.shape[1], 3), 255, np.uint8)
imageio.imwrite(f"{OUT}/wrist_remount_grid.png", np.concatenate([row0, hgap, row1], axis=0))
imageio.imwrite(f"{OUT}/wrist_remount_real.png", realref)
print("SAVED grid (cand 0..5) + real ref at same timestep")
