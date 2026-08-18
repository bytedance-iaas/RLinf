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
"""SO101 agent: ManiSkill's built-in SO100 with joint limits widened to the real
servo calibration ranges (a real2sim fidelity fix).

The stock SO100 URDF under-models the real SO101 hardware: the real follower
calibration reaches shoulder_lift +2.48 rad and elbow_flex -2.38 rad, while the
stock URDF clamps both to +/-1.5708. The real robot places objects into the tray
~23 cm from its base with ease; the sim robot cannot reach that pose at all
until these limits are widened. Both PhysX and the mplib motion planner read
this URDF, so the fix has to live there rather than in the controller.

The widened URDF is *derived* from the installed ManiSkill SO100 one at import
time rather than vendored as a binary asset. That keeps a single source of truth
for the limits -- ``so101_calib.JOINT_LIMITS_{LOW,HIGH}``, which the action
conversion clips against -- and means a checkout needs no out-of-band asset
download to run the task. Meshes are referenced back into the ManiSkill package,
so only a small XML file is written.
"""

import logging
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.so100.so_100 import SO100

from rlinf.envs.maniskill.so101_calib import (
    JOINT_LIMITS_HIGH,
    JOINT_LIMITS_LOW,
    SO101_JOINT_NAMES,
)

logger = logging.getLogger(__name__)

_STOCK_DIR = Path(PACKAGE_ASSET_DIR) / "robots" / "so100"
_STOCK_URDF = _STOCK_DIR / "so100.urdf"
_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "so101"
_URDF_PATH = _ASSET_DIR / "so101.urdf"


def _build_widened_urdf(dst: Path = _URDF_PATH) -> Path:
    """Write an SO100 URDF whose joint limits cover the real servo travel.

    Returns the path written. Mesh ``filename`` attributes are rewritten to
    absolute paths into the ManiSkill package so the output can live anywhere.
    The write is atomic (temp file + replace) because several env workers may
    import this module concurrently.
    """
    tree = ET.parse(_STOCK_URDF)
    root = tree.getroot()

    limits = {
        name: (float(lo), float(hi))
        for name, lo, hi in zip(SO101_JOINT_NAMES, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
    }
    widened = []
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        target = limits.get(joint.get("name"))
        if limit is None or target is None:
            continue
        lo, hi = target
        was = (float(limit.get("lower")), float(limit.get("upper")))
        # Only ever widen. A calibration tighter than the URDF is a calibration
        # question, not a reason to shrink what the simulator can represent.
        new = (min(was[0], lo), max(was[1], hi))
        if new != was:
            widened.append(f"{joint.get('name')} {was} -> {new}")
        limit.set("lower", repr(new[0]))
        limit.set("upper", repr(new[1]))

    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename and not os.path.isabs(filename):
            mesh.set("filename", str((_STOCK_DIR / filename).resolve()))

    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), suffix=".urdf")
    os.close(fd)
    tree.write(tmp, encoding="utf-8", xml_declaration=True)
    os.replace(tmp, dst)

    # mplib reads the SRDF next to the URDF for self-collision pairs; without it
    # the planner treats adjacent links as colliding and refuses most poses.
    stock_srdf = _STOCK_URDF.with_suffix(".srdf")
    if stock_srdf.exists():
        shutil.copyfile(stock_srdf, dst.with_suffix(".srdf"))

    logger.info(
        "Generated SO101 URDF at %s (widened: %s)",
        dst,
        "; ".join(widened) if widened else "none",
    )
    return dst


def ensure_urdf(force: bool = False) -> Path:
    """Return the SO101 URDF path, generating it if needed.

    Regenerates whenever the stock URDF is newer than the derived one, so a
    ManiSkill upgrade cannot leave a stale robot behind.
    """
    if (
        force
        or not _URDF_PATH.exists()
        or _URDF_PATH.stat().st_mtime < _STOCK_URDF.stat().st_mtime
    ):
        return _build_widened_urdf()
    return _URDF_PATH


@register_agent()
class SO101(SO100):
    """SO100 kinematics with the real SO101's joint travel."""

    uid = "so101"
    urdf_path = str(ensure_urdf())
