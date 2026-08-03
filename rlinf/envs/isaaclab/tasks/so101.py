# Copyright 2026 The RLinf Authors.
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

"""RLinf env adapter for the SO101 arm on LeIsaac tasks.

Differences from the Franka :class:`IsaaclabStackCubeEnv` adapter, all forced by
the robot rather than by choice:

* **State is joint space, not end-effector space.** Franka builds ``states`` from
  ``eef_pos``/``eef_quat``/``gripper_pos`` (a 3+3+1 pose vector). SO101's π₀.₅
  checkpoints are trained on LeRobot data whose ``observation.state`` is the 6
  joint positions, so ``states`` here is ``joint_pos`` -- 6-D, same ordering and
  same units as the action space. Mixing the two conventions is the single most
  likely way to silently destroy policy performance.
* **One camera, not two.** LeIsaac's LiftCube deletes the wrist camera in
  ``__post_init__``, so only ``front`` exists. ``wrist_images`` is reported as
  ``None`` rather than being faked with a copy of the front view or with zeros:
  a duplicated or blank wrist view is indistinguishable from a real one to the
  policy, and would corrupt inference instead of failing loudly. Set
  ``num_images_in_input: 1`` in the model config to match.
* **Camera resolution is configured per camera name.** LeIsaac scenes name their
  sensors ``front``/``wrist``; the Franka scene uses ``table_cam``/``wrist_cam``.

Unit conversion
---------------

``states`` is converted from IsaacLab radians to the checkpoint's LeRobot
normalized motor units *here*, at the env boundary, rather than in
``SO101Inputs``. That transform instance is shared with the SFT dataloader, which
already feeds normalized LeRobot episodes, so converting inside it would be
correct for one caller and wrong for the other. The reverse direction (predicted
action -> radians) lives in ``SO101Outputs``, which only the RL path uses. Both
directions share the joint tables in ``..so101_utils``.
"""

import gymnasium as gym
import torch

from ..isaaclab_env import IsaaclabBaseEnv
from ..so101_utils import SO101_ACTION_DIM, lerobot_state_scales_torch


class IsaaclabSO101Env(IsaaclabBaseEnv):
    """SO101 single-arm tasks (LeIsaac) wrapped for RLinf."""

    # Class-level default because _wrap_obs runs before this subclass's __init__ body:
    # IsaaclabBaseEnv.__init__ resets the env, which calls _wrap_obs. None for every
    # task without a `render` observation group, which is all of them but one.
    last_render_obs = None

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )
        # Cached on the env's device so _wrap_obs stays a couple of fused ops and
        # never round-trips joint positions through the host.
        self._state_offset, self._state_scale, self._state_bias = (
            lerobot_state_scales_torch(self.device)
        )

    def _make_env_function(self):
        """Build the IsaacLab env inside the subprocess.

        ``AppLauncher`` can only run once per process, so this closure is executed
        by :class:`SubProcIsaacLabEnv` in a freshly spawned process.
        """

        def make_env_isaaclab():
            import os

            # Remove DISPLAY variable to force headless mode and avoid GLX errors
            os.environ.pop("DISPLAY", None)
            # Omniverse otherwise blocks on an interactive EULA prompt and dies with
            # "Unable to bootstrap inner kit kernel: EOF when reading a line", since
            # a spawned worker has no tty. Harmless if already set in the parent.
            os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            # Imported only after the app exists: leisaac reaches into
            # isaaclab.envs/isaaclab.sim, which need the Omniverse runtime loaded.
            # so101_rl registers the *-Rewarded-v0 ids that add the sparse success
            # reward LeIsaac itself does not define.
            import leisaac  # noqa: F401
            import so101_rl  # noqa: F401
            from isaaclab_tasks.utils import load_cfg_from_registry

            isaac_env_cfg = load_cfg_from_registry(
                self.isaaclab_env_id, "env_cfg_entry_point"
            )
            # Seed the IsaacLab env config before construction so the simulator's
            # initial reset path is deterministic and doesn't warn about an unset seed.
            isaac_env_cfg.seed = self.seed
            isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs

            # Never let the teleop dataset recorder run under RL rollout. LeIsaac's
            # SingleArmTaskEnvCfg sets recorders=ActionStateRecorderManagerCfg(),
            # whose base defaults are a *fixed* path shared by every process
            # (dataset_export_dir_path="/tmp/isaaclab/logs",
            # dataset_filename="dataset", dataset_export_mode=EXPORT_ALL). HDF5
            # takes an exclusive file lock, so with more than one env rank the
            # second and later ranks die inside RecorderManager.__init__ with
            # "BlockingIOError: [Errno 11] ... unable to lock file" -- before
            # gym.make even returns. Nothing in RLinf reads that dataset.
            #
            # RecorderManagerBaseCfg is IsaacLab's own default for
            # ManagerBasedEnvCfg.recorders and carries no terms, which makes
            # RecorderManager fully inert: __init__ returns before opening any file
            # and every other entry point starts with
            # `if len(self.active_terms) == 0: return`.
            #
            # Also set on the task cfg class itself (so101_rl's
            # LiftCubeRewardedEnvCfg.recorders) so a bare gym.make from the
            # single-process verification scripts is equally clean; this line is
            # what keeps the guarantee for any *other* LeIsaac task id registered
            # against this adapter.
            from isaaclab.managers import RecorderManagerBaseCfg

            isaac_env_cfg.recorders = RecorderManagerBaseCfg()

            # Control rate. LeIsaac ships decimation=1 (60 Hz with sim.dt=1/60),
            # but SFT datasets are recorded at their teleop rate -- 30 fps for
            # henry-guo/so101-pick-place-new -- and an action chunk is a fixed
            # number of *frames*, so replaying it at 2x the recording rate covers
            # half the intended motion in half the intended time. Delegated to the
            # task cfg's own setter because the sparse reward weight and the render
            # interval must move with it; see set_control_decimation.
            decimation = self.cfg.init_params.get("decimation", None)
            if decimation is not None:
                if not hasattr(isaac_env_cfg, "set_control_decimation"):
                    raise ValueError(
                        f"config sets init_params.decimation but task "
                        f"{self.isaaclab_env_id} has no set_control_decimation(); "
                        "setting cfg.decimation directly would leave the sparse "
                        "reward weight stale and silently rescale rewards"
                    )
                isaac_env_cfg.set_control_decimation(int(decimation))

            # LeIsaac names its sensors after the mount point (front/wrist), and a
            # given task may not have all of them -- LiftCube deletes wrist. Only
            # override what the config actually asks for and what the scene has.
            for cam_name in ("front", "wrist"):
                cam_cfg = self.cfg.init_params.get(cam_name, None)
                if cam_cfg is None:
                    continue
                scene_cam = getattr(isaac_env_cfg.scene, cam_name, None)
                if scene_cam is None:
                    raise ValueError(
                        f"config requests camera '{cam_name}' but task "
                        f"{self.isaaclab_env_id} has no such sensor in its scene"
                    )
                scene_cam.height = cam_cfg.height
                scene_cam.width = cam_cfg.width

            env = gym.make(
                self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array"
            ).unwrapped
            return env, sim_app

        return make_env_isaaclab

    def _wrap_obs(self, obs):
        """Map LeIsaac observation keys onto RLinf's contract.

        LeIsaac ``policy`` group -> RLinf:
            ``front``     -> ``main_images``     (num_envs, H, W, 3) uint8
            ``wrist``     -> ``wrist_images``    (None if the task has no wrist cam)
            ``joint_pos`` -> ``states``          (num_envs, 6) float32, LeRobot
                                                 normalized motor units
            cfg.task_description -> ``task_descriptions``

        A task may also declare a ``render`` observation group -- the PickPlace-Render
        variant uses it for a third-person camera. Those frames are kept off the
        returned dict and stashed in :attr:`last_render_obs` instead, so a video-capture
        caller can read them while the policy's input stays exactly what it is for the
        non-render task ids. Putting them in the returned dict would work today (the
        openpi obs_processor reads four keys by name) but would make the observation
        contract depend on which task id is running.
        """
        # Overwritten every step; the caller reads it between steps or not at all.
        self.last_render_obs = obs.get("render")

        policy_obs = obs["policy"]

        joint_pos = policy_obs["joint_pos"][..., :SO101_ACTION_DIM]
        # rad -> deg -> normalized motor units, per joint. Equivalent to LeIsaac's
        # convert_leisaac_action_to_lerobot, kept on-device.
        states = (
            torch.rad2deg(joint_pos) - self._state_offset
        ) * self._state_scale + self._state_bias

        return {
            "main_images": policy_obs["front"],
            "task_descriptions": [self.task_description] * self.num_envs,
            "states": states,
            # Explicit None rather than an absent key: consumers such as
            # OpenPi0ForRLActionPrediction.obs_processor subscript these directly
            # (``if env_obs["wrist_images"] is not None``), so omitting them raises
            # KeyError. LiftCube deletes the wrist camera, hence None -- never a
            # copy of the front view or a zero frame, which the policy could not
            # distinguish from a real observation.
            "wrist_images": policy_obs.get("wrist"),
            "extra_view_images": None,
        }
