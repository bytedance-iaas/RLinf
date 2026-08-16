import os
"""Render SO101GrabRedCube-v1 policy camera views (front + wrist) + an overview."""
import numpy as np
import imageio.v2 as imageio
import gymnasium as gym
from rlinf.envs.maniskill import import_all_tasks

OUT = os.environ.get("SCRATCH", "/tmp/so101_runs")


def up(img, k=4):
    """Nearest-neighbor upscale HxWxC by k for viewability."""
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


import_all_tasks()
env = gym.make(
    "SO101GrabRedCube-v1", num_envs=1, obs_mode="rgb",
    control_mode="pd_joint_pos", sim_backend="gpu", render_mode="rgb_array",
)
obs, _ = env.reset(seed=0)

# Settle a few steps holding the rest pose so the scene looks natural.
rest = env.unwrapped.agent.robot.get_qpos()[:, :env.action_space.shape[-1]]
for _ in range(8):
    obs, *_ = env.step(rest)

sd = obs["sensor_data"]
front = to_np(sd["3rd_view_camera"]["rgb"][0]).astype(np.uint8)   # [128,128,3]
wrist = to_np(sd["wrist_camera"]["rgb"][0]).astype(np.uint8)
overview = to_np(env.render()[0]).astype(np.uint8)                # human render cam

imageio.imwrite(f"{OUT}/so101_front.png", up(front))
imageio.imwrite(f"{OUT}/so101_wrist.png", up(wrist))
imageio.imwrite(f"{OUT}/so101_overview.png", overview)

# Side-by-side of the two POLICY inputs (what PI0.5 actually sees).
pad = np.full((front.shape[0], 8, 3), 255, np.uint8)
sbs = np.concatenate([front, pad, wrist], axis=1)
imageio.imwrite(f"{OUT}/so101_policy_views.png", up(sbs))

print("qpos_dim=", env.unwrapped.agent.robot.get_qpos().shape,
      "front=", front.shape, "wrist=", wrist.shape, "overview=", overview.shape)
print("SAVED", OUT)
env.close()
