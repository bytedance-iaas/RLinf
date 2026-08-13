"""Sweep wrist-camera poses to find the forward-down view matching the real one.
Also render the front camera to verify scene fixes. Writes a labeled 3x3 grid."""
import numpy as np
import imageio.v2 as imageio
import gymnasium as gym
import rlinf.envs.maniskill.tasks.so101_pick_place as T
from rlinf.envs.maniskill import import_all_tasks

OUT = "/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad"


def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


import_all_tasks()

# Candidate (eye, target) in the Fixed_Jaw local frame.
cands = [
    ([0, 0, 0.06], [0.15, 0, -0.05]),
    ([0, 0, 0.06], [-0.15, 0, -0.05]),
    ([0, 0, 0.06], [0, 0.15, -0.05]),
    ([0, 0, 0.06], [0, -0.15, -0.05]),
    ([0.05, 0, 0.03], [-0.10, 0, -0.15]),
    ([-0.05, 0, 0.03], [0.10, 0, -0.15]),
    ([0, 0.05, 0.03], [0, -0.10, -0.15]),
    ([0, -0.05, 0.03], [0, 0.10, -0.15]),
    ([0, 0, 0.10], [0, 0, -0.20]),
]

tiles = []
front_saved = False
for i, (eye, tgt) in enumerate(cands):
    T.WRIST_CAM_EYE = eye
    T.WRIST_CAM_TARGET = tgt
    T.WRIST_CAM_FOV = 1.5
    env = gym.make("SO101GrabRedCube-v1", num_envs=1, obs_mode="rgb",
                   control_mode="pd_joint_pos", sim_backend="gpu", render_mode="rgb_array")
    obs, _ = env.reset(seed=1)
    rest = env.unwrapped.agent.robot.get_qpos()[:, :env.action_space.shape[-1]]
    for _ in range(6):
        obs, *_ = env.step(rest)
    w = to_np(obs["sensor_data"]["wrist_camera"]["rgb"][0]).astype(np.uint8)
    tiles.append((i, w))
    if not front_saved:
        f = to_np(obs["sensor_data"]["3rd_view_camera"]["rgb"][0]).astype(np.uint8)
        imageio.imwrite(f"{OUT}/so101_front.png",
                        np.repeat(np.repeat(f, 3, 0), 3, 1))
        front_saved = True
    env.close()
    print(f"cand {i}: eye={eye} target={tgt}", flush=True)

# 3x3 grid, each tile upscaled 2x with a white gutter.
k, g = 2, 6
tile_hw = 128 * k
grid = np.full((3 * tile_hw + 4 * g, 3 * tile_hw + 4 * g, 3), 255, np.uint8)
for idx, img in tiles:
    r, c = idx // 3, idx % 3
    up = np.repeat(np.repeat(img, k, 0), k, 1)
    y = g + r * (tile_hw + g)
    x = g + c * (tile_hw + g)
    grid[y:y + tile_hw, x:x + tile_hw] = up
imageio.imwrite(f"{OUT}/wrist_sweep.png", grid)
print("SAVED grid (index 0..8 left-to-right, top-to-bottom)")
