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

import os
from typing import Optional, OrderedDict, Union

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.common import torch_clone_dict
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import put_info_on_image, tile_images
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

__all__ = ["ManiskillEnv"]


def extract_termination_from_info(info, num_envs, device):
    if "success" in info:
        if "fail" in info:
            terminated = torch.logical_or(info["success"], info["fail"])
        else:
            terminated = info["success"].clone()
    else:
        if "fail" in info:
            terminated = info["fail"].clone()
        else:
            terminated = torch.zeros(num_envs, dtype=bool, device=device)
    return terminated


class ManiskillEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        env_seed = cfg.seed
        self.seed = env_seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        self.use_full_state = bool(getattr(cfg, "use_full_state", False))
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids

        self.video_cfg = cfg.video_cfg

        self.cfg = cfg

        with open_dict(cfg):
            cfg.init_params.num_envs = num_envs
        env_args = OmegaConf.to_container(cfg.init_params, resolve=True)
        self.env: BaseEnv = gym.make(**env_args)
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )  # [B, ]
        self.record_metrics = record_metrics
        self._is_start = True
        # Optional on-policy trajectory collector, enabled via env var
        # SO101_COLLECT_DIR: successful episodes are flushed there as .npz
        # (obs images/states as seen by the policy, actions in env units).
        self._collect_dir = os.environ.get("SO101_COLLECT_DIR")
        if self._collect_dir:
            os.makedirs(self._collect_dir, exist_ok=True)
            self._rec_bufs = [[] for _ in range(num_envs)]
            self._rec_flushed = [False] * num_envs
            self._rec_last_obs = None
            self._rec_count = 0
        # Optional spawn-vs-outcome diagnostic: CSV rows "spawn_x,spawn_y,success"
        # appended per finished episode. Enable via env var SO101_SPAWN_LOG=<file>.
        self._spawn_log = os.environ.get("SO101_SPAWN_LOG")
        if self._spawn_log:
            self._spawn_xy = np.zeros((num_envs, 2), dtype=np.float64)
        self._init_reset_state_ids()
        self.info_logging_keys = ["is_src_obj_grasped", "consecutive_grasp", "success"]
        self._show_goal_site_visual()
        if self.record_metrics:
            self._init_metrics()

    @property
    def total_num_group_envs(self):
        if hasattr(self.env.unwrapped, "total_num_trials"):
            return self.env.unwrapped.total_num_trials
        if hasattr(self.env, "xyz_configs") and hasattr(self.env, "quat_configs"):
            return len(self.env.xyz_configs) * len(self.env.quat_configs)
        return np.iinfo(np.uint8).max // 2  # TODO

    @property
    def num_envs(self):
        return self.env.unwrapped.num_envs

    @property
    def device(self):
        return self.env.unwrapped.device

    @property
    def elapsed_steps(self):
        return self.env.unwrapped.elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def instruction(self):
        return self.env.unwrapped.get_language_instruction()

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    def _show_goal_site_visual(self):
        """Keep ManiSkill goal-site visualization visible for reward-model RGB input."""
        if not hasattr(self.env.unwrapped, "goal_site"):
            return

        goal_site = self.env.unwrapped.goal_site
        if hasattr(self.env.unwrapped, "_hidden_objects"):
            while goal_site in self.env.unwrapped._hidden_objects:
                self.env.unwrapped._hidden_objects.remove(goal_site)
        if hasattr(goal_site, "show_visual"):
            goal_site.show_visual()

    def _wrap_obs(self, raw_obs, infos=None):
        wrap_obs_mode = getattr(self.cfg, "wrap_obs_mode", "default")
        if wrap_obs_mode == "raw":
            assert infos is not None
            return infos["extracted_obs"]

        if wrap_obs_mode == "simple":
            if self.env.unwrapped.obs_mode == "state":
                return {"states": raw_obs}
            elif self.env.unwrapped.obs_mode == "rgb":
                sensor_data = raw_obs.pop("sensor_data")
                raw_obs.pop("sensor_param")
                if self.use_full_state:
                    state = self._get_full_state_obs()
                else:
                    state = common.flatten_state_dict(
                        raw_obs, use_torch=True, device=self.device
                    )

                main_images = sensor_data["base_camera"]["rgb"]
                sorted_images = OrderedDict(sorted(sensor_data.items()))
                sorted_images.pop("base_camera")
                extra_view_images = (
                    torch.stack([v["rgb"] for v in sorted_images.values()], dim=1)
                    if sorted_images
                    else None
                )
                return {
                    "main_images": main_images,
                    "extra_view_images": extra_view_images,
                    "states": state,
                }

        # Default
        sensor_data = raw_obs["sensor_data"]
        obs_image = sensor_data["3rd_view_camera"]["rgb"].to(
            torch.uint8
        )  # [B, H, W, C]
        # Optional wrist camera: envs that mount a "wrist_camera" sensor (e.g. the
        # SO101 arm) expose a second view. Envs without one fall back to None, which
        # EmbodiedOutput.prepare_observations backfills anyway -- so this preserves
        # prior single-camera behavior for existing envs.
        wrist_image = (
            sensor_data["wrist_camera"]["rgb"].to(torch.uint8)
            if "wrist_camera" in sensor_data
            else None
        )
        proprioception: torch.Tensor = self.env.unwrapped.agent.robot.get_qpos().to(
            obs_image.device, dtype=torch.float32
        )
        # ManiSkill reports joint positions in radians. The SO101 LeRobot dataset
        # records them in LeRobot NORMALIZED units, so convert the proprioceptive
        # state to match the policy's normalization stats. Enable via env cfg
        # ``so101_state_norm: True``.
        if getattr(self.cfg, "so101_state_norm", False):
            from rlinf.envs.maniskill.so101_calib import rad_to_norm

            proprioception = torch.from_numpy(
                rad_to_norm(proprioception.cpu().numpy())
            ).to(proprioception.device)
        return {
            "main_images": obs_image,
            "wrist_images": wrist_image,
            "extra_view_images": None,
            "states": proprioception,
            "task_descriptions": self.instruction,
        }

    def _get_full_state_obs(self):
        base_env = self.env.unwrapped
        mode_attr = "_obs_mode" if hasattr(base_env, "_obs_mode") else "obs_mode"
        original_mode = getattr(base_env, mode_attr)
        setattr(base_env, mode_attr, "state")
        try:
            state_obs = base_env.get_obs()
        finally:
            setattr(base_env, mode_attr, original_mode)

        if isinstance(state_obs, dict):
            return common.flatten_state_dict(
                state_obs, use_torch=True, device=self.device
            )
        return state_obs

    def _calc_step_reward(self, reward, info):
        if getattr(self.cfg, "reward_mode", "default") == "raw":
            pass
        elif getattr(self.cfg, "reward_mode", "default") == "only_success":
            reward = info["success"] * 1.0
        else:
            reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
                self.env.unwrapped.device
            )  # [B, ]
            reward += info["is_src_obj_grasped"] * 0.1
            reward += info["consecutive_grasp"] * 0.1
            reward += (info["success"] & info["is_src_obj_grasped"]) * 1.0
        # diff
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"]
            episode_info["fail_once"] = self.fail_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def _rec_reset(self, extracted_obs, env_idx=None):
        """Clear collector buffers on (partial) reset and cache the new obs."""
        if env_idx is None:
            for i in range(self.num_envs):
                self._rec_bufs[i] = []
                self._rec_flushed[i] = False
        else:
            for i in common.to_numpy(env_idx).tolist():
                self._rec_bufs[i] = []
                self._rec_flushed[i] = False
        if self._rec_last_obs is None or env_idx is None:
            self._rec_last_obs = torch_clone_dict(extracted_obs)
        else:
            for k in ("main_images", "wrist_images", "states"):
                if extracted_obs.get(k) is not None:
                    self._rec_last_obs[k][env_idx] = extracted_obs[k][env_idx]

    def _rec_record_step(self, actions, infos):
        """Append (obs_t, action_t) per env; flush an episode on first success."""
        obs = self._rec_last_obs
        act = common.to_numpy(actions)
        main = common.to_numpy(obs["main_images"])
        wrist = (
            common.to_numpy(obs["wrist_images"])
            if obs.get("wrist_images") is not None
            else None
        )
        states = common.to_numpy(obs["states"])
        success = (
            common.to_numpy(infos["success"]) if "success" in infos else None
        )
        for i in range(self.num_envs):
            if self._rec_flushed[i]:
                continue
            self._rec_bufs[i].append(
                (
                    main[i].copy(),
                    wrist[i].copy() if wrist is not None else None,
                    states[i].copy(),
                    act[i].copy(),
                )
            )
            if success is not None and bool(success[i]) and len(self._rec_bufs[i]) > 5:
                buf = self._rec_bufs[i]
                path = os.path.join(
                    self._collect_dir,
                    f"ep_s{self.seed}_e{i}_{self._rec_count:05d}.npz",
                )
                np.savez_compressed(
                    path,
                    main=np.stack([b[0] for b in buf]),
                    wrist=np.stack([b[1] for b in buf])
                    if buf[0][1] is not None
                    else np.zeros(0),
                    state=np.stack([b[2] for b in buf]),
                    action=np.stack([b[3] for b in buf]),
                )
                self._rec_count += 1
                self._rec_bufs[i] = []
                self._rec_flushed[i] = True

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ):
        if options is None:
            seed = self.seed
            options = (
                {"episode_id": self.reset_state_ids}
                if self.use_fixed_reset_state_ids
                else {}
            )
        raw_obs, infos = self.env.reset(seed=seed, options=options)
        self._show_goal_site_visual()
        extracted_obs = self._wrap_obs(raw_obs, infos=infos)
        if "env_idx" in options:
            env_idx = options["env_idx"]
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        if self._collect_dir:
            self._rec_reset(extracted_obs, options.get("env_idx"))
        if self._spawn_log and hasattr(self.env.unwrapped, "red_cube"):
            xy = common.to_numpy(self.env.unwrapped.red_cube.pose.p[:, :2])
            idx = options.get("env_idx")
            if idx is None:
                self._spawn_xy[:] = xy
            else:
                for i in common.to_numpy(idx).tolist():
                    self._spawn_xy[i] = xy[i]
        return extracted_obs, infos

    def step(
        self, actions: Union[Array, dict] = None, auto_reset=True
    ) -> tuple[Array, Array, Array, Array, dict]:
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        extracted_obs = self._wrap_obs(raw_obs, infos=infos)
        if self._collect_dir:
            self._rec_record_step(actions, infos)
            self._rec_last_obs = torch_clone_dict(
                {
                    k: extracted_obs[k]
                    for k in ("main_images", "wrist_images", "states")
                    if extracted_obs.get(k) is not None
                }
            )
        step_reward = self._calc_step_reward(_reward, infos)

        infos = self._record_metrics(step_reward, infos)
        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)
        if isinstance(truncations, bool):
            truncations = torch.tensor([truncations], device=self.device)
            truncations = truncations.repeat(self.num_envs)
        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    infos["episode"]["fail_at_end"] = infos["fail"].clone()

        dones = torch.logical_or(terminations, truncations)

        if self._spawn_log and dones.any() and self.record_metrics:
            done_idx = torch.nonzero(dones, as_tuple=False).flatten()
            succ = common.to_numpy(self.success_once)
            with open(self._spawn_log, "a") as f:
                for i in common.to_numpy(done_idx).tolist():
                    f.write(
                        f"{self._spawn_xy[i][0]:.4f},{self._spawn_xy[i][1]:.4f},{int(succ[i])}\n"
                    )

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)
        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = torch_clone_dict(extracted_obs)
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = torch_clone_dict(infos)
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset(options=options)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def run(self):
        obs, info = self.reset()
        for step in range(100):
            action = self.env.action_space.sample()
            obs, rew, terminations, truncations, infos = self.step(action)
            print(
                f"Step {step}: obs={obs.keys()}, rew={rew.mean()}, terminations={terminations.float().mean()}, truncations={truncations.float().mean()}"
            )

    # render utils
    def capture_image(self, infos=None):
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) == 3:
            img = img[None]

        if infos is not None:
            for i in range(len(img)):
                info_item = {
                    k: v if np.size(v) == 1 else v[i] for k, v in infos.items()
                }
                img[i] = put_info_on_image(img[i], info_item)
        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                img = tile_images(img, nrows=int(np.sqrt(self.num_envs)))
        return img

    def render(self, info, rew=None):
        if self.video_cfg.info_on_video:
            scalar_info = gym_utils.extract_scalars_from_info(
                common.to_numpy(info), batch_size=self.num_envs
            )
            if rew is not None:
                scalar_info["reward"] = common.to_numpy(rew)
                if np.size(scalar_info["reward"]) > 1:
                    scalar_info["reward"] = [
                        float(rew) for rew in scalar_info["reward"]
                    ]
                else:
                    scalar_info["reward"] = float(scalar_info["reward"])
            image = self.capture_image(scalar_info)
        else:
            image = self.capture_image()
        return image

    def sample_action_space(self):
        return self.env.action_space.sample()
