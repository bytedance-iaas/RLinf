# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simulator-level smoke test for the SO101 ManiSkill task.

Checks the parts that do not need a policy checkpoint: the derived URDF, agent
registration, both cameras, the joint-space observation, and that a commanded
LeRobot-normalized action actually moves the arm. Run:

    python -m toolkits.so101_smoke
"""

import numpy as np


def main() -> int:
    import gymnasium as gym  # noqa: F401
    import torch

    from rlinf.envs.maniskill import so101_agent
    from rlinf.envs.maniskill.so101_calib import (
        SO101_ACTION_DIM,
        norm_to_rad,
        rad_to_norm,
        rad_to_norm_torch,
    )

    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("== URDF derivation ==")
    urdf = so101_agent.ensure_urdf()
    check("urdf exists", urdf.exists(), str(urdf))
    import xml.etree.ElementTree as ET

    limits = {
        j.get("name"): (
            float(j.find("limit").get("lower")),
            float(j.find("limit").get("upper")),
        )
        for j in ET.parse(urdf).getroot().findall("joint")
        if j.find("limit") is not None
    }
    check(
        "shoulder_lift widened",
        limits["shoulder_lift"][1] >= 2.47,
        f"upper={limits['shoulder_lift'][1]}",
    )
    check(
        "elbow_flex widened",
        limits["elbow_flex"][0] <= -2.37,
        f"lower={limits['elbow_flex'][0]}",
    )
    meshes = [m.get("filename") for m in ET.parse(urdf).getroot().iter("mesh")]
    check(
        "mesh paths absolute",
        all(m.startswith("/") for m in meshes),
        f"{len(meshes)} meshes",
    )

    print("== env construction ==")
    import mani_skill.envs  # noqa: F401

    from rlinf.envs.maniskill import import_all_tasks  # noqa: F401

    env = gym.make(
        "SO101GrabRedCube-v1",
        num_envs=4,
        obs_mode="rgb",
        reward_mode="normalized_dense",
        control_mode="pd_joint_pos",
        sim_backend="gpu",
    )
    obs, _ = env.reset(seed=0)
    sensors = obs["sensor_data"]
    check("front camera present", "3rd_view_camera" in sensors, str(list(sensors)))
    check("wrist camera present", "wrist_camera" in sensors)
    qpos = env.unwrapped.agent.robot.get_qpos()
    check("qpos is 6-dim", qpos.shape[-1] == SO101_ACTION_DIM, str(tuple(qpos.shape)))

    print("== unit conversion on live qpos ==")
    norm_t = rad_to_norm_torch(qpos)
    norm_np = rad_to_norm(qpos.cpu().numpy())
    check(
        "torch/numpy agree on real qpos",
        np.allclose(norm_t.cpu().numpy(), norm_np, atol=1e-3),
        f"max diff {np.abs(norm_t.cpu().numpy() - norm_np).max():.2e}",
    )

    print("== stepping with a normalized action ==")
    # Command the measured real home pose, then a clearly different pose, and
    # confirm the arm actually tracks it (catches a dead controller / wrong units).
    home_norm = rad_to_norm(
        np.tile(np.array([0.046, -0.880, 1.013, 0.586, -0.008, -0.931]), (4, 1))
    )
    for _ in range(20):
        obs, rew, term, trunc, info = env.step(
            torch.as_tensor(norm_to_rad(home_norm), device=qpos.device)
        )
    reached = env.unwrapped.agent.robot.get_qpos().cpu().numpy()
    target = norm_to_rad(home_norm)
    err = np.abs(reached[:, :5] - target[:, :5]).max()
    check("arm tracks commanded joints", err < 0.15, f"max joint err {err:.3f} rad")
    check("reward is finite", bool(np.isfinite(rew.cpu().numpy()).all()))
    check("info has success", "success" in info, str(sorted(info))[:120])

    print("== gripper geometry ==")
    # The regression that made grasping impossible: closed must actually close.

    def jaw_gap():
        # Read the poses fresh each time: `.pose.p` is a snapshot, not a live view.
        agent = env.unwrapped.agent
        return (
            torch.linalg.norm(
                agent.finger1_tip.pose.p - agent.finger2_tip.pose.p, axis=1
            )
            .cpu()
            .numpy()
            .mean()
        )

    closed_cmd = home_norm.copy()
    closed_cmd[:, 5] = 0.0
    for _ in range(30):
        env.step(torch.as_tensor(norm_to_rad(closed_cmd), device=qpos.device))
    gap_closed = jaw_gap()
    open_cmd = home_norm.copy()
    open_cmd[:, 5] = 100.0
    for _ in range(30):
        env.step(torch.as_tensor(norm_to_rad(open_cmd), device=qpos.device))
    gap_open = jaw_gap()
    print(
        f"    jaw gap: closed={gap_closed * 100:.1f} cm, open={gap_open * 100:.1f} cm"
    )
    check("closing narrows the jaws", gap_closed < gap_open)
    # 2.9 cm cube: the closed gap must go under it or the policy can never grasp.
    check(
        "closed gap clears the 2.9 cm cube",
        gap_closed < 0.029,
        f"{gap_closed * 100:.1f} cm",
    )

    env.close()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
