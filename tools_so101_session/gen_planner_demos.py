# STATUS: ACTIVE — 当前流程在用。 阶段 B/C/E 的示范生成器
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


def solve_grab_red_cube(env, seed=None, vis=False, xoff=0.0, yoff=0.0, drop_z=0.08, gjx=0.0, gjy=0.0):
    env.reset(seed=seed)
    _sp = env.unwrapped.red_cube.pose.sp.p
    print(f"  [spawn] {float(_sp[0]):.4f} {float(_sp[1]):.4f}", flush=True)
    # RAISE-ARM prefix (matches real demo step 1): the measured real home pose
    # is folded with the gripper nearly CLOSED and wrist_roll 0 — the planner's
    # grasp routine assumes an open gripper and wrist_roll pi/2. Interpolate to
    # the planner-ready pose first; the episode still STARTS at the real home.
    _q0 = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
    _READY = np.array([0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 0.5], dtype=np.float32)
    for _i in range(1, 25):
        env.step((_q0 + (_READY - _q0) * _i / 24).astype(np.float32))
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
    grasp_pose = sapien.Pose(p=[grasp_pose.p[0] + gjx, grasp_pose.p[1] + gjy, grasp_pose.p[2]], q=grasp_pose.q)

    planner.gripper_state = 0
    # PICK: approach from above, descend, close, lift
    planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.03]) * grasp_pose)
    planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.01]) * grasp_pose)
    planner.close_gripper(gripper_state=-0.8)
    # settle: at 30Hz the fixed-step close spans less wall-time; hold the
    # close command so the jaws physically finish before the micro-lift.
    _qh = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy().copy()
    _qh[5] = -0.8
    for _ in range(10):
        env.step(_qh.astype(np.float32))
    print("  [dbg] after close: grasped=", bool(uenv.agent.is_grasping(uenv.red_cube)[0]), flush=True)
    # micro-lift verification: contact-force flags fire on useless pinches too
    z_before = float(uenv.red_cube.pose.sp.p[2])
    planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.03]) * grasp_pose)
    if float(uenv.red_cube.pose.sp.p[2]) - z_before < 0.015:
        print("  [dbg] BAD GRASP (micro-lift failed), aborting attempt", flush=True)
        planner.close()
        return None
    lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
    r1 = planner.move_to_pose_with_screw(lift_pose)
    print("  [dbg] lift plan:", "FAIL" if r1 == -1 else "ok", " cube z=", float(uenv.red_cube.pose.sp.p[2]), flush=True)
    # PLACE (Phase 2, mirrors the real 12-step demo): transport above the tray,
    # lower slightly, open the gripper so the cube drops in, retreat up.
    # 5-DoF note: the SO100's feasible end-effector yaw is tied to the base-pan
    # bearing, so the grasp orientation is NOT reachable at the box position.
    # Rotate the grasp orientation about world Z by the bearing difference
    # (base->cube vs base->box) to stay inside the feasible family.
    box_p = uenv.box.pose.sp.p
    base_p = uenv.agent.robot.pose.sp.p
    cube_p = uenv.red_cube.pose.sp.p
    a_cube = np.arctan2(cube_p[1] - base_p[1], cube_p[0] - base_p[0])
    a_box = np.arctan2(box_p[1] - base_p[1], box_p[0] - base_p[0])
    dyaw = float(a_box - a_cube)
    place_q = (sapien.Pose(q=euler2quat(0, 0, dyaw)) * sapien.Pose(q=grasp_pose.q)).q
    # TRANSPORT: two-stage FK (vertical lift stage kills drag-mode failures;
    # traverse stage aims at the PAYLOAD) + closed-loop pre-drop refinement
    # (the hang vector rotates with the pan swing, so one-shot compensation
    # leaves 2-3cm error -> measure & correct up to 2x before opening).
    import torch as _t
    robot = uenv.agent.robot

    def fk_goto(txy, tz, steps=33):
        qs = robot.get_qpos()[0].cpu().numpy().copy()
        saved = qs.copy()
        best = None
        for pan in np.linspace(qs[0] - 0.7, qs[0] + 0.7, 15):
            for lift in np.linspace(0.2, 1.4, 9):
                for elb in np.linspace(-1.2, 0.8, 9):
                    for wf in np.linspace(-0.5, 1.2, 7):
                        q = np.array([pan, lift, elb, wf, qs[4], -0.8], dtype=np.float32)
                        robot.set_qpos(_t.tensor(q[None]))
                        tcp = uenv.agent.tcp_pos[0].cpu().numpy()
                        err = np.linalg.norm(tcp[:2] - txy)
                        if (tz - 0.03) < tcp[2] < (tz + 0.04) and (best is None or err < best[0]):
                            best = (err, q.copy())
        robot.set_qpos(_t.tensor(saved[None]))
        if best is None or best[0] > 0.04:
            return False
        q0 = saved
        for i in range(1, steps + 1):
            qi = q0 + (best[1] - q0) * i / steps
            qi[5] = -0.8
            env.step(qi.astype(np.float32))
        return True

    # stage 1: vertical lift at current cube xy (robust even if screw lift failed)
    cube_now = uenv.red_cube.pose.sp.p
    fk_goto(np.array([float(cube_now[0]), float(cube_now[1])]), 0.11, steps=21)
    # stage 2: traverse to payload-compensated tray target
    tcp_now = uenv.agent.tcp_pos[0].cpu().numpy()
    cube_now = uenv.red_cube.pose.sp.p
    hang = np.array([float(cube_now[0] - tcp_now[0]), float(cube_now[1] - tcp_now[1])])
    box_xy = np.array([float(box_p[0]) + xoff, float(box_p[1]) + yoff])
    r2 = 0 if fk_goto(box_xy - hang, drop_z + 0.01) else -1
    # closed-loop refinement: re-measure the CUBE and correct residual error
    for _ in range(2):
        if r2 == -1:
            break
        cube_now = uenv.red_cube.pose.sp.p
        errv = box_xy - np.array([float(cube_now[0]), float(cube_now[1])])
        if np.linalg.norm(errv) < 0.012:
            break
        tcp_now = uenv.agent.tcp_pos[0].cpu().numpy()
        fk_goto(np.array([tcp_now[0], tcp_now[1]]) + errv, drop_z + 0.01, steps=15)
    print("  [dbg] transport:", "FAIL" if r2 == -1 else "ok",
          " grasped=", bool(uenv.agent.is_grasping(uenv.red_cube)[0]),
          " cube=", uenv.red_cube.pose.sp.p.tolist(), flush=True)
    r3 = planner.open_gripper()
    print("  [dbg] after open: cube=", uenv.red_cube.pose.sp.p.tolist(), " box=", uenv.box.pose.sp.p.tolist(), flush=True)
    res = planner.move_to_pose_with_screw(lift_pose)
    # HOMING segment (user-confirmed success semantics: cube in box AND arm
    # back at the initial pose). Joint-space interpolation to the reset pose,
    # gripper returning to its initial near-closed value.
    import torch as _t
    HOME = np.array([0.046, -0.880, 1.013, 0.586, -0.008, -0.931], dtype=np.float32)
    qs = uenv.agent.robot.get_qpos()[0].cpu().numpy()
    steps = 30
    for i in range(1, steps + 1):
        qi = qs + (HOME - qs) * i / steps
        env.step(qi.astype(np.float32))
    for _ in range(12):  # settle at home (PD tracking lag at 30Hz)
        env.step(HOME.astype(np.float32))
    print("  [dbg] homed: home_dist=",
          float(np.abs(uenv.agent.robot.get_qpos()[0].cpu().numpy()[:5] - HOME[:5]).mean()),
          " success=", bool(uenv.evaluate()["success"][0]), flush=True)
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
        sim_config=dict(sim_freq=120, control_freq=30),  # = real dataset fps (user-approved)
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
    variants = [dict(xoff=0.0, yoff=0.0, drop_z=0.08),
                dict(xoff=-0.012, yoff=0.0, drop_z=0.07, gjx=0.004, gjy=-0.004),
                dict(xoff=0.0, yoff=0.012, drop_z=0.09, gjx=-0.004, gjy=0.004)]
    for i in range(args.num):
        succ = False
        for att, var in enumerate(variants):
            try:
                solve_grab_red_cube(env, seed=args.seed0 + i, vis=args.vis, **var)
                info = env.unwrapped.evaluate()
                succ = bool(info["success"][0])
                g = bool(info["is_grasped"][0])
                ib = bool(info["is_in_box"][0])
                stage = "OK" if succ else ("drop-missed" if g is False and not ib else "grasp-fail")
                print(f"demo {i} att{att}: success={succ} stage={stage}", flush=True)
                if succ:
                    break
            except Exception as e:
                print(f"demo {i} att{att}: EXC {type(e).__name__}: {e}", flush=True)
        ok += int(succ)
    env.close()
    print(f"TOTAL success {ok}/{args.num}")


if __name__ == "__main__":
    main()
