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
"""SO101/SO100 "Grab the red cube" task for online RL of an OpenPI PI0.5 policy.

Matches the real henry-guo/so101-pick-place-v2 setup (dataset task label:
"Grab the red cube"): a RED target cube and a BLUE distractor cube (2.9 cm each)
on a light-brown wooden board, plus a box (10 x 7.5 x 4.6 cm) sitting on the
table next to the board, ~20 cm in front of the gripper. The task is to grasp
and lift the red cube; the blue cube and box are scene distractors for visual
sim2real fidelity.

Built on ManiSkill's built-in ``so100`` robot (SO100/SO101 share kinematics).
Two policy cameras (``3rd_view_camera`` front + ``wrist_camera``) feed
ManiskillEnv._wrap_obs. Joint-position control (pd_joint_pos); the 6-dim PI0.5
output maps directly onto [shoulder_pan..wrist_roll, gripper].

NOTE: object sizes are exact; object *positions* / board size / box color are
reasonable placeholders -- calibrate them against a real front/wrist camera
frame before sim2real transfer.
"""

import sapien
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

# Dataset task label (must match the LeRobot prompt the PI0.5 was trained on).
SO101_INSTRUCTION = "Grab the red cube"

# Real object dimensions (meters).
CUBE_HALF = 0.0145  # 2.9 cm cube edge
BOX_HALF = [
    0.0381,
    0.0508,
    0.0222,
]  # USER-MEASURED tray 4x3x1.75in, LONG (10.16cm) edge along y
BOARD_HALF = [
    0.108,
    0.1505,
    0.0024,
]  # FROM DATASET FRAMES (user: 按第一个图片): 21.6cm(x, matches 8.5in) x 30.1cm(y)
BLACK_END_LEN = 0.019  # black band at -y edge, 1.9cm as measured in the dataset frames
BROWN_HALF_Y = (
    2 * BOARD_HALF[1] - BLACK_END_LEN
) / 2  # brown zone half-length (cubes spawn here only)
BOARD_CY_OFF = (
    -BLACK_END_LEN / 2
)  # board center offset so the BROWN zone is centered on the base axis (user: 盒子/棕区/底座中线对齐)
BOARD_COLOR = [0.76, 0.62, 0.42, 1.0]  # kraft cardboard tan
BOX_COLOR = [0.08, 0.08, 0.09, 1.0]  # dark/black-rimmed open tray
STRIP_COLOR = [0.05, 0.05, 0.05, 1.0]  # black strip along the far-right board edge
CREAM_COLOR = [0.96, 0.94, 0.89, 1.0]  # cream/off-white desk surface
GROUND = 0.002  # top of the cream desk cover (objects rest here)
LIFT_HEIGHT = 0.06  # red cube lifted this high above start = success

# Front camera: near-top-down (overhead) to match the real dataset front camera.
# Looks straight down over the workspace; world +x (away from the arm, toward the
# box) maps to image-up so the arm sits at the bottom of the frame. Lowered +
# narrowed so the 12x8" board fills the frame like the real front camera.
# The REAL camera records 640x480 (4:3) with SQUARE pixels (verified: cube
# bbox h/w median 1.000 over 20 frames) and openpi's resize_with_pad
# letterboxes it. Sim therefore renders 160x120 (same 4:3, square pixels) so
# both domains go through the IDENTICAL letterbox path. Focal MEASURED from
# the real board extents at the user-measured camera height (~0.55m above
# board): f=710px at 640w -> 177.5px at 160w.
FRONT_CAM_EYE = [
    -0.520,
    -0.007,
    0.559,
]  # above the brown-zone center + small offsets matching the real framing
FRONT_CAM_TARGET = [-0.520, -0.007, 0.0]
FRONT_CAM_W, FRONT_CAM_H = 640, 480
FRONT_CAM_INTRINSIC = [[755.2, 0.0, 320.0], [0.0, 755.2, 240.0], [0.0, 0.0, 1.0]]
WRIST_CAM_INTRINSIC = [[343.5, 0.0, 320.0], [0.0, 343.5, 240.0], [0.0, 0.0, 1.0]]
FRONT_CAM_UP = [1.0, 0.0, 0.0]
FRONT_CAM_FOV = 0.84  # ~board fills 60% of frame width, as in the real image

# Wrist camera (mounted on the gripper): looks forward-down over the jaw tips so
# the jaws sit at the bottom of the frame and the workspace ahead is visible.
# Workspace is in the Fixed_Jaw -y direction (found by pose sweep).
WRIST_CAMERA_MOUNT_LINK = "Fixed_Jaw"
# Re-pointed after the wrist_flex +0.6 gripper-down offset (so101_calib) so the
# wrist view looks forward-down at the workspace, matching the real wrist cam.
WRIST_CAM_EYE = [0.0, 0.0, 0.04]
WRIST_CAM_TARGET = [0.0, -0.05, -0.18]
WRIST_CAM_FOV = 1.5


@register_env("SO101GrabRedCube-v1", max_episode_steps=640)
class SO101GrabRedCubeEnv(PickCubeEnv):
    """Grasp the red cube on the SO100/SO101 arm; blue cube + box are distractors."""

    SUPPORTED_ROBOTS = ["so100", "so101"]

    def __init__(
        self,
        *args,
        robot_uids="so101",
        spawn_mode: str = "full_board",
        spawn_frac=None,
        **kwargs,
    ):
        """
        Args:
            spawn_mode: ``"full_board"`` (default) spawns both cubes anywhere in
                the brown zone, matching the measured real episodes.
                ``"legacy"`` reproduces the older 6x8 cm sub-box and exists only
                for comparison against historical runs.
            spawn_frac: ``(x0, x1, y0, y1)`` fractions of the active spawn ranges,
                restricting the red cube to a sub-region -- e.g. targeted data
                collection in a weak region. ``None`` uses the full range.
        """
        # so101 = built-in so100 with joint limits widened to the REAL servo
        # calibration ranges (see rlinf/envs/maniskill/so101_agent.py).
        from rlinf.envs.maniskill import so101_agent  # noqa: F401  (registers uid)

        if spawn_mode not in ("full_board", "legacy"):
            raise ValueError(
                f"spawn_mode must be 'full_board' or 'legacy', got {spawn_mode!r}"
            )
        self._spawn_mode = spawn_mode
        if spawn_frac is not None:
            spawn_frac = tuple(float(v) for v in spawn_frac)
            if len(spawn_frac) != 4:
                raise ValueError(
                    f"spawn_frac must be (x0, x1, y0, y1), got {spawn_frac!r}"
                )
        self._spawn_frac = spawn_frac

        # PickCubeEnv keys spawn-center/camera defaults by uid; mirror so100's
        # entry so the unknown uid doesn't silently fall back to the panda
        # layout (which shifts the whole scene by ~0.5m).
        from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS

        if "so101" not in PICK_CUBE_CONFIGS:
            PICK_CUBE_CONFIGS["so101"] = dict(PICK_CUBE_CONFIGS["so100"])
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # --- cameras: front (3rd_view) + wrist ------------------------------------
    @property
    def _default_sensor_configs(self):
        front_pose = sapien_utils.look_at(
            eye=FRONT_CAM_EYE, target=FRONT_CAM_TARGET, up=FRONT_CAM_UP
        )
        wrist_pose = sapien_utils.look_at(eye=WRIST_CAM_EYE, target=WRIST_CAM_TARGET)
        wrist_mount = self.agent.robot.links_map[WRIST_CAMERA_MOUNT_LINK]
        return [
            CameraConfig(
                "3rd_view_camera",
                front_pose,
                FRONT_CAM_W,
                FRONT_CAM_H,
                None,
                0.01,
                100,
                intrinsic=FRONT_CAM_INTRINSIC,
            ),
            CameraConfig(
                "wrist_camera",
                wrist_pose,
                FRONT_CAM_W,
                FRONT_CAM_H,
                None,
                0.01,
                100,
                intrinsic=WRIST_CAM_INTRINSIC,
                mount=wrist_mount,
            ),
        ]

    def _build_open_tray(self, name="box"):
        """Open-top tray: light interior floor + 4 black rim walls. Local origin
        at the tray base center (z=0 = table contact). Outer size = 2*BOX_HALF."""
        outer_half_x, outer_half_y, outer_half_z = BOX_HALF
        wall_half_thickness, floor_half_thickness = 0.004, 0.004
        black = sapien.render.RenderMaterial(base_color=BOX_COLOR)
        light = sapien.render.RenderMaterial(base_color=[0.9, 0.9, 0.9, 1.0])
        builder = self.scene.create_actor_builder()
        # floor (light interior)
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, floor_half_thickness]),
            half_size=[outer_half_x, outer_half_y, floor_half_thickness],
            material=light,
        )
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, floor_half_thickness]),
            half_size=[outer_half_x, outer_half_y, floor_half_thickness],
        )
        # 4 rim walls (black)
        walls = [
            (
                [outer_half_x - wall_half_thickness, 0, outer_half_z],
                [wall_half_thickness, outer_half_y, outer_half_z],
            ),
            (
                [-(outer_half_x - wall_half_thickness), 0, outer_half_z],
                [wall_half_thickness, outer_half_y, outer_half_z],
            ),
            (
                [0, outer_half_y - wall_half_thickness, outer_half_z],
                [outer_half_x, wall_half_thickness, outer_half_z],
            ),
            (
                [0, -(outer_half_y - wall_half_thickness), outer_half_z],
                [outer_half_x, wall_half_thickness, outer_half_z],
            ),
        ]
        for wall_center, wall_half_size in walls:
            builder.add_box_visual(
                pose=sapien.Pose(p=wall_center),
                half_size=wall_half_size,
                material=black,
            )
            builder.add_box_collision(
                pose=sapien.Pose(p=wall_center), half_size=wall_half_size
            )
        builder.set_initial_pose(sapien.Pose(p=[0, 0, 0]))
        return builder.build_kinematic(name=name)

    # --- scene: board + red cube + blue distractor + box ----------------------
    def _load_scene(self, options: dict):
        from mani_skill.utils.scene_builder.table import TableSceneBuilder

        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Cream/off-white desk surface covering the wooden table (matches real).
        self.table_cover = actors.build_box(
            self.scene,
            half_sizes=[0.7, 0.7, 0.001],
            color=CREAM_COLOR,
            name="table_cover",
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[self.cube_spawn_center[0], 0, GROUND - 0.001]),
        )

        # Light-brown wooden board the cubes rest on (kinematic).
        self.board = actors.build_box(
            self.scene,
            half_sizes=BOARD_HALF,
            color=BOARD_COLOR,
            name="board",
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, BOARD_HALF[2]]),
        )
        # Red target cube + blue distractor cube (dynamic). Density set for a
        # USER-BOUNDED mass: real cube <10g -> 8g at 2.9cm => 328 kg/m^3
        # (ManiSkill's default density 1000 gives 24.4g, a 2.4x dynamics error).

        def _build_light_cube(color, name, initial_position):
            cube_builder = self.scene.create_actor_builder()
            cube_builder.add_box_collision(half_size=[CUBE_HALF] * 3, density=328.0)
            cube_builder.add_box_visual(
                half_size=[CUBE_HALF] * 3,
                material=sapien.render.RenderMaterial(base_color=color),
            )
            cube_builder.initial_pose = sapien.Pose(p=initial_position)
            return cube_builder.build(name=name)

        self.red_cube = _build_light_cube(
            [0.85, 0.05, 0.05, 1], "red_cube", [0, 0, CUBE_HALF]
        )
        self.blue_cube = _build_light_cube(
            [0.05, 0.10, 0.85, 1], "blue_cube", [0, 0.08, CUBE_HALF]
        )
        # Open tray (black rim + light interior) beyond the far edge of the board,
        # ~20 cm in front of the gripper -- matches the real black-rimmed open box.
        self.box = self._build_open_tray(name="box")
        # Black END of the board (part of the board, -y side; user-confirmed).
        self.strip = actors.build_box(
            self.scene,
            half_sizes=[BOARD_HALF[0], BLACK_END_LEN / 2, 0.0005],
            color=STRIP_COLOR,
            name="edge_strip",
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 0.005]),
        )
        # PickCubeEnv reward/obs reference self.cube -> alias to the red target.
        self.cube = self.red_cube

    # --- layout (positions are placeholders; calibrate vs real camera) --------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # TableSceneBuilder only special-cases uid "so100"; replicate its
            # placement for so101 (identical geometry, widened-limit URDF).
            if self.robot_uids == "so101":
                import numpy as _np
                from transforms3d.euler import euler2quat as _e2q

                # MEASURED initial pose: median of the 87 real episodes'
                # first-frame observation.state (norm_to_rad), std <=2.5 norm
                # units — the arm starts folded/retracted, gripper near closed.
                qpos0 = _np.array([0.046, -0.880, 1.013, 0.586, -0.008, -0.931])
                qpos0 = (
                    self._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos0))
                    )
                    + qpos0
                )
                self.agent.reset(qpos0)
                self.agent.robot.set_pose(
                    sapien.Pose([-0.725, 0, 0], q=_e2q(0, 0, _np.pi / 2))
                )
            cx, cy = self.cube_spawn_center  # SO100 workspace center (~-0.46, 0)
            board_top = GROUND + 2 * BOARD_HALF[2]
            # Board shifted toward the arm base so its near edge sits at the base
            # (base_x = -0.615); cubes still spawn in the reachable cx region.
            # USER-CONFIRMED (2026-08-09): the real base front edge is flush
            # against the board's near edge (<2cm). Base collision AABB front
            # is at x=-0.654; near edge at ~1.2cm gap with the new 21.6cm depth.
            board_center_x = -0.654 + 0.012 + BOARD_HALF[0]

            # Board center shifted -y so the BROWN zone (not the whole board)
            # is centered on the base/tray axis (user: 中线对齐).
            board_center_y = cy + BOARD_CY_OFF
            self.board.set_pose(
                Pose.create_from_pq(
                    torch.tensor(
                        [[board_center_x, board_center_y, GROUND + BOARD_HALF[2]]]
                    ).repeat(b, 1)
                )
            )
            # Black end of the board at -y (part of the board itself, per user).
            self.strip.set_pose(
                Pose.create_from_pq(
                    torch.tensor(
                        [
                            [
                                board_center_x,
                                board_center_y - BOARD_HALF[1] + BLACK_END_LEN / 2,
                                GROUND + 2 * BOARD_HALF[2] + 0.0005,
                            ]
                        ]
                    ).repeat(b, 1)
                )
            )

            # Red target cube: ALWAYS spawned within the board (with margin) and
            # inside the SO100 reach -- far half (+x toward the box), image-left (+y).
            # ``spawn_frac`` (fractions of the default ranges) restricts the
            # spawn sub-box, e.g. for hard-region targeted data collection.
            # Default = full box.
            # USER-CONFIRMED (2026-08-09): BOTH cubes may appear ANYWHERE on the
            # board (measured: all 87 real first frames span the full board).
            # Full-board spawn is the DEFAULT; ``spawn_mode="legacy"`` reproduces
            # the old 6x8cm sub-box (for historical comparisons only), and
            # ``spawn_frac`` targets a sub-region of the active ranges (for
            # weak-region data collection).
            frac_x0, frac_x1, frac_y0, frac_y1 = self._spawn_frac or (
                0.0,
                1.0,
                0.0,
                1.0,
            )
            red = torch.zeros((b, 3))
            blue = torch.zeros((b, 3))
            if self._spawn_mode == "legacy":
                red[:, 0] = (
                    board_center_x
                    + (frac_x0 + torch.rand((b,)) * (frac_x1 - frac_x0)) * 0.06
                )
                red[:, 1] = (
                    cy
                    + 0.02
                    + (frac_y0 + torch.rand((b,)) * (frac_y1 - frac_y0)) * 0.08
                )
                blue[:, 0] = board_center_x - 0.05 + torch.rand((b,)) * 0.04
                blue[:, 1] = cy - 0.06 + torch.rand((b,)) * 0.04
            else:
                # Spawn zone = BROWN region only (user: cubes never on the black
                # end), centered on the base axis. Margin 2cm (design choice).
                spawn_margin = 0.02
                spawn_x_lo, spawn_x_hi = (
                    -BOARD_HALF[0] + spawn_margin,
                    BOARD_HALF[0] - spawn_margin,
                )
                spawn_y_lo, spawn_y_hi = (
                    -BROWN_HALF_Y + spawn_margin,
                    BROWN_HALF_Y - spawn_margin,
                )
                red[:, 0] = (
                    board_center_x
                    + spawn_x_lo
                    + (frac_x0 + torch.rand((b,)) * (frac_x1 - frac_x0))
                    * (spawn_x_hi - spawn_x_lo)
                )
                red[:, 1] = (
                    cy
                    + spawn_y_lo
                    + (frac_y0 + torch.rand((b,)) * (frac_y1 - frac_y0))
                    * (spawn_y_hi - spawn_y_lo)
                )
                # blue: same brown zone; USER SPEC: no minimum separation, only
                # physical non-overlap (3cm ~= touching cubes)
                blue[:, 0] = (
                    board_center_x
                    + spawn_x_lo
                    + torch.rand((b,)) * (spawn_x_hi - spawn_x_lo)
                )
                blue[:, 1] = (
                    cy + spawn_y_lo + torch.rand((b,)) * (spawn_y_hi - spawn_y_lo)
                )
                for _ in range(10):
                    bad = torch.linalg.norm(blue[:, :2] - red[:, :2], axis=1) < 0.03
                    if not bad.any():
                        break
                    nb = int(bad.sum())
                    blue[bad, 0] = (
                        board_center_x
                        + spawn_x_lo
                        + torch.rand((nb,)) * (spawn_x_hi - spawn_x_lo)
                    )
                    blue[bad, 1] = (
                        cy + spawn_y_lo + torch.rand((nb,)) * (spawn_y_hi - spawn_y_lo)
                    )
            red[:, 2] = board_top + CUBE_HALF
            blue[:, 2] = board_top + CUBE_HALF
            self.red_cube.set_pose(Pose.create_from_pq(red))
            self.blue_cube.set_pose(Pose.create_from_pq(blue))

            # Open tray flush against the board's far edge (no gap), long edge along
            # y (same orientation as the board's long edge).
            box = torch.zeros((b, 3))
            box[:, 0] = board_center_x + BOARD_HALF[0] + BOX_HALF[0]
            box[:, 1] = cy
            box[:, 2] = GROUND
            self.box.set_pose(Pose.create_from_pq(box))

            # Per-env start height for the lift criterion. Keep a FULL-batch
            # buffer and update only the reset envs: _initialize_episode gets a
            # partial env_idx on auto-reset after early termination (= success),
            # and overwriting the whole tensor with the partial batch crashes
            # evaluate() with a size mismatch (first hit the moment v5 scored
            # its first-ever successful grasp).
            if (
                not hasattr(self, "_red_start_z")
                or self._red_start_z.shape[0] != self.num_envs
            ):
                self._red_start_z = torch.zeros(
                    self.num_envs, device=red.device, dtype=red.dtype
                )
            self._red_start_z[env_idx] = red[:, 2]

    # --- obs / success / reward for grab-and-lift -----------------------------
    def _get_obs_extra(self, info: dict):
        obs = {
            "is_grasped": info["is_grasped"],
            "tcp_pose": self.agent.tcp_pose.raw_pose,
        }
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.red_cube.pose.raw_pose,
                tcp_to_obj_pos=self.red_cube.pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def evaluate(self):
        is_grasped = self.agent.is_grasping(self.red_cube)
        lifted = (self.red_cube.pose.p[:, 2] - self._red_start_z) >= LIFT_HEIGHT
        is_robot_static = self.agent.is_static(0.2)
        # Phase 2 success: red cube released INSIDE the tray (matches the real
        # demo semantics: pick the red cube and place it into the box).
        # Tray interior: outer half-dims BOX_HALF minus the 4mm walls; rim top
        # sits at GROUND + 2*BOX_HALF[2].
        rel = self.red_cube.pose.p - self.box.pose.p
        in_box = (
            (rel[:, 0].abs() < BOX_HALF[0] - 0.008)
            & (rel[:, 1].abs() < BOX_HALF[1] - 0.008)
            & (self.red_cube.pose.p[:, 2] < GROUND + 2 * BOX_HALF[2])
        )
        placed = in_box & ~is_grasped
        # USER-CONFIRMED success semantics (2026-08-09): cube in the box AND the
        # arm back at its initial pose ("进盒 + 回到初始位"). Home distance is
        # over the 5 arm joints (gripper free); tolerance 0.08 rad = max real
        # episode deviation 0.076 (measured over all 87) + margin.
        qpos = self.agent.robot.get_qpos()[:, :5]
        if not hasattr(self, "_home_qpos") or self._home_qpos.shape[0] != qpos.shape[0]:
            self._home_qpos = torch.tensor(
                [0.046, -0.880, 1.013, 0.586, -0.008],
                device=qpos.device,
                dtype=qpos.dtype,
            ).expand(qpos.shape[0], -1)
        home_dist = (qpos - self._home_qpos).abs().mean(dim=1)
        is_home = home_dist < 0.08
        return {
            "success": placed & is_home,
            "is_grasped": is_grasped,
            "is_lifted": lifted,
            "is_in_box": in_box,
            "is_placed": placed,
            "home_dist": home_dist,
            "is_robot_static": is_robot_static,
        }

    def compute_dense_reward(self, obs, action, info):
        # Proven grasp-reward recipe (ManiSkill PickCube / lerobot-sim2real
        # SO100GraspCube): reach gradient + binary contact grasp + post-grasp
        # progress gated by is_grasped. NO gripper open/close shaping — every
        # successful reference reward leaves gripper timing to the policy; the
        # binary is_grasped jump (+1 AND unlocking the lift term) is what makes
        # grasping strictly dominate hovering.
        tcp_to_obj = torch.linalg.norm(
            self.red_cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reward = 1 - torch.tanh(5 * tcp_to_obj)
        is_grasped = info["is_grasped"]
        # Gradient bridge for the gripper: the binary is_grasped term gives the
        # gripper dimension NO gradient until the first successful grasp, and
        # this VLA's exploration never stumbles into a close (measured: ~1.3M
        # rollout transitions with zero closes). Reward closing WHEN AT THE CUBE
        # so there is a continuous uphill path hover-open -> closed -> grasped.
        # No open/far terms (the previous far-open term taught hovering).
        from rlinf.envs.maniskill.so101_calib import (
            GRIPPER_RAD_CLOSED,
            GRIPPER_RAD_OPEN,
        )

        grip_q = self.agent.robot.get_qpos()[:, -1]
        closedness = torch.clamp(
            (GRIPPER_RAD_OPEN - grip_q) / (GRIPPER_RAD_OPEN - GRIPPER_RAD_CLOSED),
            0.0,
            1.0,
        )
        reward = reward + 0.5 * (tcp_to_obj < 0.04).float() * closedness
        reward = reward + is_grasped
        lift = torch.clamp(
            (self.red_cube.pose.p[:, 2] - self._red_start_z) / LIFT_HEIGHT, 0.0, 1.0
        )
        reward = reward + lift * is_grasped
        # Phase 2 transport: while holding the cube, pull it toward a waypoint
        # above the tray. Gated by is_grasped so it cannot be farmed empty-handed.
        # Anti-hack arithmetic (checked BEFORE launch): hover-over-tray while
        # holding maxes at ~5.4/step, released-in-box success pays 8/step ->
        # releasing strictly dominates. Episodes do not terminate on success
        # (env.train.ignore_terminations=True), so success pays every step.
        box_target = self.box.pose.p.clone()
        box_target[:, 2] = box_target[:, 2] + 0.10
        cube_to_box = torch.linalg.norm(self.red_cube.pose.p - box_target, axis=1)
        reward = reward + 2.0 * (1 - torch.tanh(5 * cube_to_box)) * is_grasped
        # Homing stage (success = placed AND home, user-confirmed semantics).
        # Reward ladder arithmetic (per-step, ignore_terminations pays states
        # every step): hold-hover-at-box maxes ~5.4 < placed floor 6.0 <=
        # placed+homing <= 7.5 < success 8 -> release strictly dominates
        # holding, homing strictly dominates lingering, success dominates all.
        placed = info["is_placed"]
        home_term = 1 - torch.tanh(3 * info["home_dist"])
        reward = torch.where(placed, 6.0 + 1.5 * home_term, reward)
        reward[info["success"]] = 8
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 8

    def get_language_instruction(self):
        return [SO101_INSTRUCTION] * self.num_envs
