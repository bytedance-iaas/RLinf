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
"""SO101 agent: ManiSkill's built-in SO100 with joint limits widened to the
REAL servo calibration ranges (real2sim fidelity fix).

The stock SO100 URDF under-models the real SO101 hardware: the real follower
calibration gives shoulder_lift up to +2.47 rad and elbow_flex down to
-2.37 rad, while the stock URDF clamps both to +/-1.5708. The real robot
places objects into the tray ~23cm from its base with ease; the sim robot
could not reach that pose at all until these limits were widened (both PhysX
and the mplib motion planner read this URDF).
"""
from pathlib import Path

from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.so100.so_100 import SO100

_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "so101"


@register_agent()
class SO101(SO100):
    uid = "so101"
    urdf_path = str(_ASSET_DIR / "so101.urdf")
