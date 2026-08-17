# STATUS: TOOL — 通用工具，与具体阶段无关。 标定验证：烘焙后的参数是否对得上真机
"""Verify the baked calibration (wrist_flex +0.6): replay episode 0 and build
real-vs-sim comparisons for BOTH the front (top-down) and the wrist camera."""
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


def rs(img, H):
    h, w = img.shape[:2]; W = int(w * H / h)
    return img[(np.arange(H) * h // H)][:, (np.arange(W) * w // W)]


real_front = [np.asarray(x) for x in imageio.get_reader(f"{DS}/videos/chunk-000/observation.images.front/episode_000000.mp4")]
real_wrist = [np.asarray(x) for x in imageio.get_reader(f"{DS}/videos/chunk-000/observation.images.wrist/episode_000000.mp4")]

ep = pd.read_parquet(sorted(glob.glob(f"{DS}/data/**/*.parquet", recursive=True))[0])
ep = ep[ep["episode_index"] == 0]
actions = np.stack(ep["action"].values)[::5]
init_state = np.stack(ep["observation.state"].values)[0]

import_all_tasks()
env = gym.make("SO101GrabRedCube-v1", num_envs=1, obs_mode="rgb",
               control_mode="pd_joint_pos", sim_backend="gpu", render_mode="rgb_array")
dev = env.unwrapped.device
env.reset(seed=0)
env.unwrapped.agent.robot.set_qpos(
    torch.tensor(C.norm_to_rad(init_state), dtype=torch.float32, device=dev).reshape(1, -1))

sim_front, sim_wrist = [], []
for a in actions:
    obs, *_ = env.step(torch.tensor(C.norm_to_rad(a), dtype=torch.float32, device=dev).reshape(1, -1))
    sim_front.append(to_np(obs["sensor_data"]["3rd_view_camera"]["rgb"][0]).astype(np.uint8))
    sim_wrist.append(to_np(obs["sensor_data"]["wrist_camera"]["rgb"][0]).astype(np.uint8))
env.close()

H = 150
idx = np.linspace(0, len(actions) - 1, 5).astype(int)


def build(realv, simv, name):
    rows = []
    for si in idx:
        ri = min(len(realv) - 1, si * 5)
        r = rs(realv[ri], H); s = rs(simv[si], H)
        gap = np.full((H, 6, 3), 255, np.uint8)
        rows.append(np.concatenate([r, gap, s], axis=1))
    mw = max(x.shape[1] for x in rows)
    rows = [np.pad(x, ((0, 0), (0, mw - x.shape[1]), (0, 0)), constant_values=255) for x in rows]
    sep = np.full((3, mw, 3), 180, np.uint8)
    imageio.imwrite(f"{OUT}/verify_{name}.png", np.concatenate([y for x in rows for y in (x, sep)], axis=0))


build(real_front, sim_front, "front")  # each row: LEFT real | RIGHT sim
build(real_wrist, sim_wrist, "wrist")
print("SAVED verify_front.png and verify_wrist.png (LEFT=real | RIGHT=sim, 5 timesteps)")
