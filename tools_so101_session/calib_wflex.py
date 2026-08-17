# STATUS: TOOL — 通用工具，与具体阶段无关。 标定：wrist_flex 零位偏移扫描
"""Replay episode 0 under several wrist_flex zero-OFFSETs; render the angled view
(gripper tilt visible) + the sim wrist camera (sees what the gripper points at).
Goal: find the offset where the gripper points DOWN at the reach/grab, and the
sim wrist view matches the real wrist video."""
import glob
import os
import numpy as np
import pandas as pd
import torch
import imageio.v2 as imageio
import gymnasium as gym
import rlinf.envs.maniskill.so101_calib as C
from rlinf.envs.maniskill import import_all_tasks

OUT = os.environ.get("SCRATCH", "/tmp/so101_runs")
DS = "/root/.cache/huggingface/lerobot/henry-guo/so101-pick-place-v2"


def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


# real wrist frames (to compare gripper orientation)
rw = imageio.get_reader(f"{DS}/videos/chunk-000/observation.images.wrist/episode_000000.mp4")
real_wrist = [np.asarray(x) for x in rw]

ep = pd.read_parquet(sorted(glob.glob(f"{DS}/data/**/*.parquet", recursive=True))[0])
ep = ep[ep["episode_index"] == 0]
actions = np.stack(ep["action"].values)[::5]
init_state = np.stack(ep["observation.state"].values)[0]

offsets = [0.6, 1.2, 1.8]  # POSITIVE wrist_flex offsets (negative pointed UP per user)

import_all_tasks()
env = gym.make("SO101GrabRedCube-v1", num_envs=1, obs_mode="rgb",
               control_mode="pd_joint_pos", sim_backend="gpu", render_mode="rgb_array")
dev = env.unwrapped.device
key = [len(actions) // 4, len(actions) // 2, 3 * len(actions) // 4]  # reach-phase frames

rows_render, rows_wrist = [], []
for off in offsets:
    C.SIGN = np.array([1, 1, 1, 1, 1, 1], float)
    C.OFFSET = np.array([0, 0, 0, off, 0, 0], float)
    env.reset(seed=0)
    env.unwrapped.agent.robot.set_qpos(
        torch.tensor(C.norm_to_rad(init_state), dtype=torch.float32, device=dev).reshape(1, -1))
    rframes, wframes = [], []
    for i, a in enumerate(actions):
        tgt = torch.tensor(C.norm_to_rad(a), dtype=torch.float32, device=dev).reshape(1, -1)
        obs, *_ = env.step(tgt)
        if i in key:
            rframes.append(to_np(env.render()[0]).astype(np.uint8))
            wframes.append(to_np(obs["sensor_data"]["wrist_camera"]["rgb"][0]).astype(np.uint8))
    lbl = np.full((18, rframes[0].shape[1], 3), 255, np.uint8)  # spacer
    rows_render.append(np.concatenate([lbl] + [f for f in rframes], axis=0) if False else np.concatenate(rframes, axis=1))
    rows_wrist.append(np.concatenate(wframes, axis=1))
    print("offset", off, "done", flush=True)

imageio.imwrite(f"{OUT}/calib_wflex_render.png", np.concatenate(rows_render, axis=0))  # rows=offsets(0,-0.6,-1.2), cols=reach frames
# real wrist at the same key frames (i*5)
rw_key = [real_wrist[min(len(real_wrist) - 1, k * 5)] for k in key]
def rs(img, H):
    h, w = img.shape[:2]; W = int(w * H / h); return img[(np.arange(H) * h // H)][:, (np.arange(W) * w // W)]
Wr = np.concatenate([rs(x, 128) for x in rw_key], axis=1)
sheet = np.concatenate([Wr] + rows_wrist, axis=0)  # row0=real wrist, then sim wrist per offset
imageio.imwrite(f"{OUT}/calib_wflex_wrist.png", sheet)
print("SAVED render (rows=offset 0/-0.6/-1.2) and wrist (row0=REAL, then offsets)")
env.close()
