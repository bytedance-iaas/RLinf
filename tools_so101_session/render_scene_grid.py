"""Render the CURRENT SO101 sim: front camera (what the policy sees) for four
different seeds, plus one wrist view and one free overview camera.

Default spawn mode = full brown zone, so the red and blue cubes land anywhere
in the real task's spawn region -- this is the picture of the true task, not of
the narrowed training box.
"""
import os

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from rlinf.envs.maniskill import import_all_tasks

OUT = os.environ.get("SCRATCH", "/tmp/so101_runs")
SEEDS = [11, 22, 33, 44]


def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def label(img, text):
    """Burn a tiny 5x7 bitmap caption into the top-left corner (no font deps)."""
    glyphs = {
        "0": ["111", "101", "101", "101", "111"], "1": ["010", "110", "010", "010", "111"],
        "2": ["111", "001", "111", "100", "111"], "3": ["111", "001", "111", "001", "111"],
        "4": ["101", "101", "111", "001", "001"], "5": ["111", "100", "111", "001", "111"],
        "S": ["111", "100", "111", "001", "111"], "E": ["111", "100", "110", "100", "111"],
        "D": ["110", "101", "101", "101", "110"], "=": ["000", "111", "000", "111", "000"],
        " ": ["000", "000", "000", "000", "000"],
    }
    x = 6
    for ch in text:
        g = glyphs.get(ch.upper())
        if g:
            for r, row in enumerate(g):
                for c, on in enumerate(row):
                    if on == "1":
                        img[6 + r * 3:9 + r * 3, x + c * 3:x + 3 + c * 3] = [255, 255, 0]
        x += 12
    return img


import_all_tasks()
env = gym.make(
    "SO101GrabRedCube-v1", num_envs=1, obs_mode="rgb",
    control_mode="pd_joint_pos", sim_backend="gpu", render_mode="rgb_array",
)

fronts, wrists = [], []
for sd in SEEDS:
    obs, _ = env.reset(seed=sd)
    rest = env.unwrapped.agent.robot.get_qpos()[:, : env.action_space.shape[-1]]
    for _ in range(6):                      # let the scene settle at the real home pose
        obs, *_ = env.step(rest)
    sensors = obs["sensor_data"]
    fronts.append(label(to_np(sensors["3rd_view_camera"]["rgb"])[0].copy(), f"SEED={sd}"))
    wrists.append(to_np(sensors["wrist_camera"]["rgb"])[0].copy())
    red = env.unwrapped.red_cube.pose.sp.p
    blue = env.unwrapped.blue_cube.pose.sp.p
    print(f"seed {sd}: red=({red[0]:+.3f},{red[1]:+.3f})  blue=({blue[0]:+.3f},{blue[1]:+.3f})", flush=True)

grid = np.concatenate(
    [np.concatenate(fronts[:2], axis=1), np.concatenate(fronts[2:], axis=1)], axis=0
)
imageio.imwrite(os.path.join(OUT, "so101_front_grid.png"), grid)
imageio.imwrite(os.path.join(OUT, "so101_wrist.png"), wrists[0])
imageio.imwrite(os.path.join(OUT, "so101_overview.png"), to_np(env.render())[0])
print("written:", grid.shape, "front grid;", wrists[0].shape, "wrist")
env.close()
