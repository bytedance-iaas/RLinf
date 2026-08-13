"""Replay episode 0 under several per-joint SIGN candidates; build real-vs-sim
front-cam comparison sheets to find the sign config that matches the real arm."""
import glob
import numpy as np
import pandas as pd
import torch
import imageio.v2 as imageio
import gymnasium as gym
import rlinf.envs.maniskill.so101_calib as C
from rlinf.envs.maniskill import import_all_tasks

OUT = "/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad"
DS = "/root/.cache/huggingface/lerobot/henry-guo/so101-pick-place-v2"


def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def rs(img, H):
    h, w = img.shape[:2]; W = int(w * H / h)
    return img[(np.arange(H) * h // H)][:, (np.arange(W) * w // W)]


# real front frames
realr = imageio.get_reader(f"{DS}/videos/chunk-000/observation.images.front/episode_000000.mp4")
realf = [np.asarray(x) for x in realr]

# episode 0 actions
ep = pd.read_parquet(sorted(glob.glob(f"{DS}/data/**/*.parquet", recursive=True))[0])
ep = ep[ep["episode_index"] == 0]
actions = np.stack(ep["action"].values)[::5]
init_state = np.stack(ep["observation.state"].values)[0]

candidates = {
    "baseline": [1, 1, 1, 1, 1, 1],
    "flip_pan": [-1, 1, 1, 1, 1, 1],
    "flip_wflex": [1, 1, 1, -1, 1, 1],
    "flip_pan_wflex": [-1, 1, 1, -1, 1, 1],
}

import_all_tasks()
env = gym.make("SO101GrabRedCube-v1", num_envs=1, obs_mode="rgb",
               control_mode="pd_joint_pos", sim_backend="gpu", render_mode="rgb_array")
dev = env.unwrapped.device
H = 200
sim_idx = np.linspace(0, len(actions) - 1, 6).astype(int)

for name, sign in candidates.items():
    C.SIGN = np.array(sign, dtype=float)
    env.reset(seed=0)
    env.unwrapped.agent.robot.set_qpos(
        torch.tensor(C.norm_to_rad(init_state), dtype=torch.float32, device=dev).reshape(1, -1))
    front = []
    for a in actions:
        tgt = torch.tensor(C.norm_to_rad(a), dtype=torch.float32, device=dev).reshape(1, -1)
        obs, *_ = env.step(tgt)
        front.append(to_np(obs["sensor_data"]["3rd_view_camera"]["rgb"][0]).astype(np.uint8))
    rows = []
    for si in sim_idx:
        ri = min(len(realf) - 1, si * 5)
        r = rs(realf[ri], H); s = rs(front[si], H)
        gap = np.full((H, 6, 3), 255, np.uint8)
        rows.append(np.concatenate([r, gap, s], axis=1))
    mw = max(x.shape[1] for x in rows)
    rows = [np.pad(x, ((0, 0), (0, mw - x.shape[1]), (0, 0)), constant_values=255) for x in rows]
    sep = np.full((4, mw, 3), 180, np.uint8)
    sheet = np.concatenate([y for x in rows for y in (x, sep)], axis=0)
    imageio.imwrite(f"{OUT}/calib_{name}.png", sheet)
    print("saved", name, sign, flush=True)
env.close()
print("DONE (each sheet: LEFT=real | RIGHT=sim, 6 timesteps)")
