"""Replay a real demo episode's recorded ACTIONS through the SO101 sim via
env.step (pd_joint_pos), rendering the top-down front camera to compare against
the real front-cam. Validates the normalized->radian mapping (units+zero+sign)."""
import glob
import os
import numpy as np
import pandas as pd
import torch
import imageio.v2 as imageio
import gymnasium as gym
from rlinf.envs.maniskill import import_all_tasks
from rlinf.envs.maniskill.so101_calib import norm_to_rad

OUT = os.environ.get("SCRATCH", "/tmp/so101_runs")


def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


f = sorted(glob.glob("/root/.cache/huggingface/lerobot/henry-guo/so101-pick-place-v2/data/**/*.parquet", recursive=True))[0]
ep = pd.read_parquet(f)
ep = ep[ep["episode_index"] == 0]
actions = np.stack(ep["action"].values)[::5]        # subsample the 30fps trajectory
init_state = np.stack(ep["observation.state"].values)[0]
targets = norm_to_rad(actions)                       # [T,6] rad joint targets

import_all_tasks()
env = gym.make("SO101GrabRedCube-v1", num_envs=1, obs_mode="rgb",
               control_mode="pd_joint_pos", sim_backend="gpu", render_mode="rgb_array")
env.reset(seed=0)
# start the arm at the demo's initial joint configuration
robot = env.unwrapped.agent.robot
robot.set_qpos(torch.tensor(norm_to_rad(init_state), dtype=torch.float32,
                            device=env.unwrapped.device).reshape(1, -1))

front, render = [], []
for tgt in targets:
    a = torch.tensor(tgt, dtype=torch.float32, device=env.unwrapped.device).reshape(1, -1)
    obs, *_ = env.step(a)
    front.append(to_np(obs["sensor_data"]["3rd_view_camera"]["rgb"][0]).astype(np.uint8))
    render.append(to_np(env.render()[0]).astype(np.uint8))

imageio.mimwrite(f"{OUT}/replay_front.mp4", front, fps=10)
imageio.mimwrite(f"{OUT}/replay_render.mp4", render, fps=10)
strip = np.concatenate([front[0], front[len(front)//3], front[2*len(front)//3], front[-1]], axis=1)
imageio.imwrite(f"{OUT}/replay_front_strip.png", np.repeat(np.repeat(strip, 2, 0), 2, 1))
print("frames", len(front), "SAVED replay")
env.close()
