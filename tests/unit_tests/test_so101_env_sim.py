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
"""SO101 tests that need a live Isaac Sim process.

Skipped unless ``RLINF_RUN_SO101_SIM_TESTS=1``. Booting Omniverse costs minutes
and needs a GPU plus the LeIsaac USD assets, so this cannot run in ordinary CI --
but the two properties it checks are exactly the ones a mocked test cannot reach:

* **CUDA tensors survive the subprocess boundary.** ``SubProcIsaacLabEnv`` ships
  observations over a ``torch.multiprocessing`` queue, which passes CUDA storage by
  *handle* rather than by copy. When that mapping goes wrong the usual symptom is
  garbage or zeros, not an exception, so checking ``.is_cuda`` proves nothing on its
  own -- the values are read back and validated.
* **``reset(env_ids=...)`` rewinds only the named envs.** If a task's ``reset``
  ignored ``env_ids`` and rewound the whole scene, aggregate reward would still look
  reasonable while PPO credit assignment was silently destroyed.

The sim boots once per module (it is the dominant cost); each test asserts on the
shared handle.

Run with:
    RLINF_RUN_SO101_SIM_TESTS=1 LEISAAC_ASSETS_ROOT=... \
        pytest tests/unit_tests/test_so101_env_sim.py
"""

import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RLINF_RUN_SO101_SIM_TESTS") != "1",
    reason="needs a GPU, Isaac Sim, and LeIsaac assets; set RLINF_RUN_SO101_SIM_TESTS=1",
)

TASK_ID = "LeIsaac-SO101-LiftCube-Rewarded-v0"
NUM_ENVS = 4
OBS_KEYS = {
    "main_images",
    "task_descriptions",
    "states",
    "wrist_images",
    "extra_view_images",
}


def _env_cfg():
    """Mirror of ``examples/embodiment/config/env/isaaclab_so101_lift_cube.yaml``.

    Built by hand rather than loaded through hydra so the test does not depend on
    the ``EMBODIED_PATH`` searchpath wiring. Cameras are shrunk to 64px: this test
    cares whether pixels arrive, not what they look like.
    """
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "env_type": "isaaclab",
            "total_num_envs": NUM_ENVS,
            "auto_reset": False,
            "ignore_terminations": False,
            "use_rel_reward": True,
            "seed": 0,
            "group_size": 1,
            "reward_coef": 1.0,
            "use_fixed_reset_state_ids": False,
            "max_steps_per_rollout_epoch": 64,
            "max_episode_steps": 64,
            "video_cfg": {"save_video": False, "info_on_video": False, "fps": 20},
            "init_params": {
                "id": TASK_ID,
                "num_envs": NUM_ENVS,
                "max_episode_steps": 64,
                "task_description": "Lift the red cube up.",
                "front": {"height": 64, "width": 64},
            },
        }
    )


@pytest.fixture(scope="module")
def sim_env():
    """Boot the SO101 env once for the whole module."""
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.pop("DISPLAY", None)

    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

    env = REGISTER_ISAACLAB_ENVS[TASK_ID](
        cfg=_env_cfg(),
        num_envs=NUM_ENVS,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(scope="module")
def reset_obs(sim_env):
    obs, _ = sim_env.reset()
    return obs


# --------------------------------------------------------------------------- #
# CUDA transport across the subprocess boundary
# --------------------------------------------------------------------------- #


def test_env_runs_on_cuda(sim_env):
    assert "cuda" in str(sim_env.device)


def test_states_cross_as_cuda_tensor(reset_obs):
    from rlinf.envs.isaaclab.so101_utils import SO101_ACTION_DIM

    states = reset_obs["states"]
    assert states.is_cuda
    assert states.shape == (NUM_ENVS, SO101_ACTION_DIM)
    # A CUDA handle that failed to map yields garbage rather than raising.
    assert torch.isfinite(states).all()


def test_images_cross_as_cuda_and_are_not_blank(reset_obs):
    images = reset_obs["main_images"]
    assert images.is_cuda
    assert images.shape[0] == NUM_ENVS and images.shape[-1] == 3
    assert images.dtype == torch.uint8
    # Proves the renderer actually produced pixels rather than a zeroed buffer.
    assert int(images.max().item()) > 0


def test_optional_camera_keys_are_none(reset_obs):
    """LiftCube has no wrist camera, but the keys must exist -- see _wrap_obs."""
    assert set(reset_obs) == OBS_KEYS
    assert reset_obs["wrist_images"] is None
    assert reset_obs["extra_view_images"] is None


def test_received_states_decode_to_joint_limits(reset_obs):
    """The values must be real joint positions, not a mis-mapped buffer."""
    from rlinf.envs.isaaclab.so101_utils import lerobot_to_isaaclab

    rad = lerobot_to_isaaclab(reset_obs["states"].cpu().numpy())
    assert np.abs(rad).max() < 3.15


def test_action_crosses_boundary_and_moves_the_arm(sim_env, reset_obs):
    """An action posted from the parent must reach the simulator and take effect."""
    from rlinf.envs.isaaclab.so101_utils import lerobot_to_isaaclab

    start = lerobot_to_isaaclab(reset_obs["states"].cpu().numpy())
    # Actions are absolute joint targets in radians by this point (SO101Outputs has
    # already converted). Nudge shoulder_pan only.
    target = start.copy()
    target[:, 0] += 0.20
    act = torch.as_tensor(target, dtype=torch.float32, device=sim_env.device)
    for _ in range(25):
        obs, reward, terminations, truncations, _ = sim_env.step(act)

    moved = lerobot_to_isaaclab(obs["states"].cpu().numpy())
    assert (moved[:, 0] - start[:, 0] > 0.05).all()

    assert reward.shape == (NUM_ENVS,) and reward.is_cuda
    assert terminations.dtype == torch.bool and terminations.is_cuda
    assert truncations.dtype == torch.bool


# --------------------------------------------------------------------------- #
# Partial reset isolation
# --------------------------------------------------------------------------- #


def test_partial_reset_leaves_other_envs_untouched(sim_env):
    """``reset(env_ids=[1, 3])`` must rewind exactly those envs.

    The arm is first driven well away from its reset pose, so a scene-wide reset
    (the failure being guarded against) is detectable rather than a no-op.
    """
    from rlinf.envs.isaaclab.so101_utils import lerobot_to_isaaclab

    obs, _ = sim_env.reset()
    start = lerobot_to_isaaclab(obs["states"].cpu().numpy())
    far = start.copy()
    far[:, 0] += 0.60
    far_act = torch.as_tensor(far, dtype=torch.float32, device=sim_env.device)
    for _ in range(40):
        obs, *_ = sim_env.step(far_act)
    before = lerobot_to_isaaclab(obs["states"].cpu().numpy())
    steps_before = sim_env.elapsed_steps.clone()

    kept, rewound = [0, 2], [1, 3]
    obs, _ = sim_env.reset(
        env_ids=torch.tensor(rewound, device=sim_env.device),
    )
    after = lerobot_to_isaaclab(obs["states"].cpu().numpy())

    assert np.abs(after[kept, 0] - before[kept, 0]).max() < 1e-3
    assert np.abs(after[rewound, 0] - before[rewound, 0]).min() > 0.05

    # Bookkeeping must be sliced the same way as the simulator state.
    steps_after = sim_env.elapsed_steps
    assert (steps_after[rewound] == 0).all()
    assert (steps_after[kept] == steps_before[kept]).all()
    assert (~sim_env.success_once[rewound]).all()
    assert (sim_env.returns[rewound] == 0).all()


# --------------------------------------------------------------------------- #
# chunk_step -- the path RL rollout actually drives
# --------------------------------------------------------------------------- #


def test_chunk_step_shapes_and_obs_contract(sim_env):
    from rlinf.envs.isaaclab.so101_utils import (
        isaaclab_to_lerobot,
        lerobot_to_isaaclab,
    )

    obs, _ = sim_env.reset()
    rad = lerobot_to_isaaclab(obs["states"].cpu().numpy())
    chunk_len = 5
    chunk = torch.as_tensor(
        np.repeat(rad[:, None, :], chunk_len, axis=1),
        dtype=torch.float32,
        device=sim_env.device,
    )
    obs_list, rewards, terminations, truncations, _ = sim_env.chunk_step(chunk)

    assert len(obs_list) == chunk_len
    for t in (rewards, terminations, truncations):
        assert t.shape == (NUM_ENVS, chunk_len)
    # Every step of the chunk must honour the same key contract, not just the first.
    assert all(set(o) == OBS_KEYS for o in obs_list)

    # States must carry the normalization the checkpoint was trained on.
    states = obs_list[-1]["states"].cpu().numpy()
    round_tripped = isaaclab_to_lerobot(lerobot_to_isaaclab(states))
    assert np.abs(round_tripped - states).max() < 1e-2
