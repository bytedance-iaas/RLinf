# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .tasks.so101 import IsaaclabSO101Env
from .tasks.stack_cube import IsaaclabStackCubeEnv

REGISTER_ISAACLAB_ENVS = {
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0": IsaaclabStackCubeEnv,
    "LeIsaac-SO101-LiftCube-Rewarded-v0": IsaaclabSO101Env,
    # Same adapter, two-camera task variant: the adapter reads cameras by name from
    # the env config, so restoring the wrist view needs no code change here.
    "LeIsaac-SO101-LiftCube-Rewarded-Wrist-v0": IsaaclabSO101Env,
    # Pick-place: the task the SO101 SFT checkpoint was actually trained on. Same
    # adapter again -- the observation protocol (front/wrist RGB, 6-dim joint
    # action, joint_pos state) is identical, and everything that differs is inside
    # the IsaacLab env config (extra props, a different success predicate).
    #
    # Registering both ids here is separate from registering them with gym: our
    # so101_rl extension does the gym side, but get_env_cls() checks this dict
    # first, so a gym-registered task missing from here fails at env-worker init
    # with "has not been registered" rather than anywhere near gym.make.
    "LeIsaac-SO101-PickPlace-v0": IsaaclabSO101Env,
    "LeIsaac-SO101-PickPlace-Wrist-v0": IsaaclabSO101Env,
    # Same task as -Wrist-v0 with a third-person camera added in a separate `render`
    # observation group, for video capture. The adapter needs no branch for it: it
    # forwards any `render` group to last_render_obs and builds the policy observation
    # from the `policy` group exactly as before.
    "LeIsaac-SO101-PickPlace-Render-v0": IsaaclabSO101Env,
}

__all__ = [list(REGISTER_ISAACLAB_ENVS.keys())]
