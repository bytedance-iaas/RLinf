# STATUS: TOOL — 通用工具，与具体阶段无关。 只抓取不放置的简化探针，用来隔离失败环节
"""Generate scripted grasp demos in SO101GrabRedCube-v1 via motion planning.

Adapted from mani_skill.examples.motionplanning.so100.solutions.pick_cube.
Success criterion in our env = red cube grasped & lifted >= 6cm, so after the
grasp we lift straight up instead of moving to PickCube's goal_site.
"""
import argparse
import sys

sys.path.insert(0, "/data08/henryg/pai/RLinf")

import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.examples.motionplanning.so100.motionplanner import (
    SO100ArmMotionPlanningSolver)
from mani_skill.utils.wrappers.record import RecordEpisode

from rlinf.envs.maniskill import import_all_tasks

import_all_tasks()


def solve_grab_red_cube(env, seed=None, vis=False):
    env.reset(seed=seed)
    planner = SO100ArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    FINGER_LENGTH = 0.025
    uenv = env.unwrapped

    obb = get_actor_obb(uenv.red_cube)
    approaching = np.array([0, 0, -1])
    tcp_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * uenv.agent.tcp_pose.sp
    target_closing = tcp_pose.to_transformation_matrix()[:3, 1]
    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing = grasp_info["closing"]
    grasp_pose = uenv.agent.build_grasp_pose(
        approaching, closing, uenv.red_cube.pose.sp.p
    )
    # SO100 gripper frame correction (from the upstream solver)
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(-np.pi / 2, 0, np.pi / 2))

    planner.gripper_state = 0
    # PICK: approach from above, descend, close, lift
    planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.03]) * grasp_pose)
    planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.01]) * grasp_pose)
    planner.close_gripper(gripper_state=-0.8)
    lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)
    res = planner.move_to_pose_with_screw(lift_pose)
    planner.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", type=str, default="/data08/henryg/pai/data/so101_sim_demos")
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()

    env = gym.make(
        "SO101GrabRedCube-v1",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="cpu",
    )
    env = RecordEpisode(
        env,
        output_dir=args.out,
        save_trajectory=True,
        save_video=False,
        source_type="motionplanning",
        source_desc="scripted grasp demos via SO100 motion planner",
    )
    ok = 0
    for i in range(args.num):
        try:
            solve_grab_red_cube(env, seed=args.seed0 + i, vis=args.vis)
            info = env.unwrapped.evaluate()
            succ = bool(info["success"][0])
            grasped = bool(info["is_grasped"][0])
            lifted = bool(info["is_lifted"][0])
            ok += int(succ)
            print(f"demo {i}: success={succ} grasped={grasped} lifted={lifted}", flush=True)
        except Exception as e:
            print(f"demo {i}: FAILED {type(e).__name__}: {e}", flush=True)
    env.close()
    print(f"TOTAL success {ok}/{args.num}")


if __name__ == "__main__":
    main()
