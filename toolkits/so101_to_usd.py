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
"""Export the SO101 sim (robot + task scene) to a single self-contained USD.

Source of truth is what the ManiSkill task actually runs:
  * robot:  rlinf/envs/maniskill/assets/so101/so101.urdf  (widened joint limits,
            matching the real servo calibration) + its STL meshes, baked into the
            stage as UsdGeom.Mesh (USD cannot reference STL)
  * scene:  the constants in rlinf/envs/maniskill/tasks/so101_pick_place.py
            (board + black end, open tray, red/blue cubes, desk cover), placed at
            the same world poses the env uses
  * physics: UsdPhysics articulation/joints with the URDF limits, link masses and
            inertias; cubes get the measured 8 g mass (density 328 kg/m^3)

Usage:
    python -m toolkits.so101_to_usd [--out /path/so101_scene.usda] [--robot-only]
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = f"{REPO}/rlinf/envs/maniskill/assets/so101/so101.urdf"
TASK = f"{REPO}/rlinf/envs/maniskill/tasks/so101_pick_place.py"
DEFAULT_OUT = f"{REPO}/assets/usd/so101_scene.usda"


def task_const(name, default=None):
    src = open(TASK).read()
    m = re.search(rf"^{re.escape(name)}\s*=\s*([^#\n]+)", src, re.M)
    return eval(m.group(1).strip()) if m else default


def rpy_to_quat(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return Gf.Quatf(
        float(cr * cp * cy + sr * sp * sy),
        Gf.Vec3f(
            float(sr * cp * cy - cr * sp * sy),
            float(cr * sp * cy + sr * cp * sy),
            float(cr * cp * sy - sr * sp * cy),
        ),
    )


def parse_xyz_rpy(elem):
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if elem is not None:
        o = elem.find("origin")
        if o is not None:
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
    return xyz, rpy


def add_mesh(stage, path, mesh_file, scale, xyz, rpy, color):
    """Bake an STL into the stage as a UsdGeom.Mesh."""
    tm = trimesh.load(mesh_file, force="mesh")
    verts = np.asarray(tm.vertices, dtype=np.float32) * np.asarray(scale, dtype=np.float32)
    faces = np.asarray(tm.faces, dtype=np.int32)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
    mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateSubdivisionSchemeAttr("none")
    if color:
        mesh.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    x = UsdGeom.Xformable(mesh)
    x.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    x.AddOrientOp().Set(rpy_to_quat(rpy))
    return mesh


def add_box(stage, path, half, pos, color, mass=None, kinematic=False):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(2.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    x = UsdGeom.Xformable(cube)
    x.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    x.AddScaleOp().Set(Gf.Vec3f(float(half[0]), float(half[1]), float(half[2])))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if not kinematic:
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    if mass is not None:
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(float(mass))
    return cube


def main() -> int:
    out = DEFAULT_OUT
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    robot_only = "--robot-only" in sys.argv
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if os.path.exists(out):
        os.remove(out)
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    # ---------------- robot ----------------
    root = ET.parse(URDF).getroot()
    robot = UsdGeom.Xform.Define(stage, "/World/SO101")
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
    # env places the base at (-0.725, 0, 0) rotated +90 deg about Z
    rx = UsdGeom.Xformable(robot)
    rx.AddTranslateOp().Set(Gf.Vec3d(-0.725, 0.0, 0.0))
    rx.AddOrientOp().Set(rpy_to_quat([0, 0, np.pi / 2]))

    urdf_dir = os.path.dirname(URDF)
    n_links = n_joints = n_meshes = 0
    for link in root.findall("link"):
        name = link.get("name")
        lp = f"/World/SO101/{name}"
        lx = UsdGeom.Xform.Define(stage, lp)
        UsdPhysics.RigidBodyAPI.Apply(lx.GetPrim())
        inertial = link.find("inertial")
        if inertial is not None:
            m = inertial.find("mass")
            if m is not None:
                UsdPhysics.MassAPI.Apply(lx.GetPrim()).CreateMassAttr(float(m.get("value")))
        for i, vis in enumerate(link.findall("visual")):
            g = vis.find("geometry/mesh")
            if g is None:
                continue
            f = os.path.join(urdf_dir, g.get("filename"))
            if not os.path.exists(f):
                continue
            scale = [float(v) for v in (g.get("scale") or "1 1 1").split()]
            xyz, rpy = parse_xyz_rpy(vis)
            mat = vis.find("material")
            col = (0.9, 0.9, 0.9)
            if mat is not None and mat.find("color") is not None:
                col = tuple(float(v) for v in mat.find("color").get("rgba").split())
            add_mesh(stage, f"{lp}/visual_{i}", f, scale, xyz, rpy, col)
            n_meshes += 1
        n_links += 1

    for joint in root.findall("joint"):
        jname, jtype = joint.get("name"), joint.get("type")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        xyz, rpy = parse_xyz_rpy(joint)
        jpath = f"/World/SO101/joints/{jname}"
        if jtype == "revolute" or jtype == "continuous":
            j = UsdPhysics.RevoluteJoint.Define(stage, jpath)
            axis = [float(v) for v in (joint.find("axis").get("xyz") if joint.find("axis") is not None else "0 0 1").split()]
            j.CreateAxisAttr("X" if abs(axis[0]) > 0.5 else ("Y" if abs(axis[1]) > 0.5 else "Z"))
            lim = joint.find("limit")
            if lim is not None and lim.get("lower") is not None:
                j.CreateLowerLimitAttr(float(np.degrees(float(lim.get("lower")))))
                j.CreateUpperLimitAttr(float(np.degrees(float(lim.get("upper")))))
        elif jtype == "prismatic":
            j = UsdPhysics.PrismaticJoint.Define(stage, jpath)
        else:
            j = UsdPhysics.FixedJoint.Define(stage, jpath)
        j.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/SO101/{parent}")])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/SO101/{child}")])
        j.CreateLocalPos0Attr(Gf.Vec3f(float(xyz[0]), float(xyz[1]), float(xyz[2])))
        j.CreateLocalRot0Attr(rpy_to_quat(rpy))
        n_joints += 1

    # ---------------- task scene ----------------
    if not robot_only:
        BOARD_HALF = task_const("BOARD_HALF")
        BLACK = task_const("BLACK_END_LEN")
        BOX_HALF = task_const("BOX_HALF")
        CUBE_HALF = task_const("CUBE_HALF")
        GROUND = task_const("GROUND")
        brown_half_y = (2 * BOARD_HALF[1] - BLACK) / 2
        bcx = -0.654 + 0.012 + BOARD_HALF[0]
        board_cy = -BLACK / 2

        add_box(stage, "/World/Scene/desk", [0.7, 0.7, 0.001],
                [bcx, 0, GROUND - 0.001], (0.96, 0.94, 0.89), kinematic=True)
        add_box(stage, "/World/Scene/board", BOARD_HALF,
                [bcx, board_cy, GROUND + BOARD_HALF[2]], (0.76, 0.62, 0.42), kinematic=True)
        add_box(stage, "/World/Scene/board_black_end", [BOARD_HALF[0], BLACK / 2, 0.0005],
                [bcx, board_cy - BOARD_HALF[1] + BLACK / 2, GROUND + 2 * BOARD_HALF[2] + 0.0005],
                (0.05, 0.05, 0.05), kinematic=True)
        # open tray: floor + 4 rim walls (same construction as _build_open_tray)
        ox, oy, oz = BOX_HALF
        wt = ft = 0.004
        tray_x = bcx + BOARD_HALF[0] + ox
        add_box(stage, "/World/Scene/tray/floor", [ox, oy, ft],
                [tray_x, 0, GROUND + ft], (0.9, 0.9, 0.9), kinematic=True)
        for i, (p, hs) in enumerate([
            ([ox - wt, 0, oz], [wt, oy, oz]), ([-(ox - wt), 0, oz], [wt, oy, oz]),
            ([0, oy - wt, oz], [ox, wt, oz]), ([0, -(oy - wt), oz], [ox, wt, oz]),
        ]):
            add_box(stage, f"/World/Scene/tray/wall_{i}", hs,
                    [tray_x + p[0], p[1], GROUND + p[2]], (0.08, 0.08, 0.09), kinematic=True)
        # cubes at the centre of the pp-era (legacy) spawn box; 8 g each
        add_box(stage, "/World/Scene/red_cube", [CUBE_HALF] * 3,
                [bcx + 0.03, 0.06, GROUND + 2 * BOARD_HALF[2] + CUBE_HALF],
                (0.85, 0.05, 0.05), mass=0.008)
        add_box(stage, "/World/Scene/blue_cube", [CUBE_HALF] * 3,
                [bcx - 0.03, -0.06, GROUND + 2 * BOARD_HALF[2] + CUBE_HALF],
                (0.05, 0.10, 0.85), mass=0.008)

        # front camera, exactly as the policy sees it
        eye = task_const("FRONT_CAM_EYE")
        K = task_const("FRONT_CAM_INTRINSIC")
        W, H = task_const("FRONT_CAM_W, FRONT_CAM_H", (640, 480))
        cam = UsdGeom.Camera.Define(stage, "/World/Sensors/front_camera")
        aperture = 20.955
        cam.CreateFocalLengthAttr(float(K[0][0] / W * aperture))
        cam.CreateHorizontalApertureAttr(aperture)
        cam.CreateVerticalApertureAttr(aperture * H / W)
        cx = UsdGeom.Xformable(cam)
        cx.AddTranslateOp().Set(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])))
        cx.AddOrientOp().Set(rpy_to_quat([0, np.pi, 0]))  # looking straight down (-Z)

    stage.GetRootLayer().Save()
    print(f"wrote {out}")
    print(f"  links={n_links} joints={n_joints} baked_meshes={n_meshes} "
          f"scene={'robot-only' if robot_only else 'robot + board/tray/cubes/camera'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
