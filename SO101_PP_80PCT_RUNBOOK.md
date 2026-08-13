# SO101 + PI0.5 — Reproducing the 80% Result (pp-era runbook)

**Status of this document.** This is the exact, step-by-step recipe that produced the
project's best policy: `so101_sft_pp6b/.../global_step_1000`, **≈80% deterministic
success** (gate 83.6%, blind fresh seeds 82.8% / 78.1%). Every number below was
measured in this project, not estimated.

---

## 0. What "80%" is — and is not (read this first)

The 80% was achieved on a **subset task**, not the real one:

| | pp-era task (80%) | TRUE task (current repo default) |
|---|---|---|
| Cube spawn | 6 × 8 cm box | whole brown zone, 17.6 × 24.2 cm (**8.9× area**) |
| Real-world coverage | **9%** of the 87 real cube starts | ~100% |
| Success criterion | cube released in tray | cube in tray **AND arm returned home** |
| Cameras | 128 × 128 square | 640 × 480 (real-camera parity) — **keep 640×480; do not restore 128²** |
| Control frequency | 15 Hz nominal (demos actually 20 Hz) | 30 Hz (= real dataset fps) |
| Episode budget | 240 steps | 640 steps |
| Cube mass | 24.4 g (sim default) | 8 g (measured bound) |

On the TRUE task the same pipeline yields **12.5%**. If your goal is a number to
report, this runbook reproduces 80%. If your goal is real-robot transfer, the
80% policy has never seen 91% of the positions the real task presents.

---

## 1. Environment state required (must be restored — the repo has moved on)

The task file `rlinf/envs/maniskill/tasks/so101_pick_place.py` now encodes the TRUE
task. To reproduce pp-era numbers you must restore five constants:

| What | pp-era value | current value | how |
|---|---|---|---|
| Cube spawn | 6 × 8 cm box | full brown zone | `export SO101_SPAWN_MODE=legacy` (already supported, no code edit) |
| Success | in-box only | in-box **and** home | remove the `& is_home` term in `evaluate()` |
| Cameras | `128, 128` | `640, 480` | **DO NOT restore 128² — it was under-resolved** (the policy pipeline keeps 224×168; 128² throws information away). Keep 640×480. |
| Control freq | 15 (nominal; demos actually 20) | 30 | **DO NOT restore 15 — see §1b. It was a defect, not an ingredient.** |
| Budget | 240 | 640 | `max_episode_steps: 240` in the env yaml **and** in `@register_env(...)` |
| Cube mass | sim default | 8 g | drop `density=328.0` in `_build_light_cube` |

Board geometry also changed (30.1 × 21.6 cm with a black end vs the old 12″×8″
kraft board). The pp-era policies were trained on the old board; visual mismatch
degrades them. Recovering that exactly requires reverting `BOARD_HALF`,
`BLACK_END_LEN`, `BOARD_CY_OFF` and the front-camera pose/intrinsics to the
pre-2026-08-09 values (see git history / `SO101_SESSION_LOG.md` Part 2).

### 1a. The recommended way to use this runbook (2026-08-12)

Do **not** rebuild the pp-era environment. Keep every fidelity fix — 640×480
cameras, 30 Hz control, measured board geometry, 8 g cube, homing in the success
criterion — and reproduce the pp *conditions* by narrowing **one** thing: the red
cube's spawn region, back to the pp 6 × 8 cm box:

```bash
export SO101_SPAWN_MODE=legacy     # already supported; no code edit needed
```

Verified to sit inside the current brown zone: spawn x ∈ [−0.534, −0.474],
y ∈ [0.020, 0.100]; brown zone x ∈ [−0.622, −0.446], y ∈ [−0.131, 0.111].

Why this is the right move: the pp-era 80% did not come from low resolution or a
wrong frequency — it came from **demonstration density relative to the grasp
tolerance**. 48 cm² with ~175 demos gives 0.52 cm spacing (pp-era: 0.51 cm), while
the full brown zone with 420 demos gives 1.01 cm — beyond the ±0.7 cm grasp
tolerance, which is where behaviour cloning stops interpolating and starts
memorising. Narrow the spawn, keep everything else correct, then widen the spawn
progressively as the floor allows.

### 1b. Frequency: use 30 Hz, not the pp-era 15 Hz (corrected recipe)

The pp era ran the sim at a nominal 15 Hz while the demo generator actually ran at
ManiSkill's default 20 Hz and the dataset was labelled 15 fps — **three different
numbers in one pipeline**, none of them equal to the real robot's **30 fps**. That
mismatch means the policy's action-chunk timing does not correspond to real time,
so a 15 Hz policy cannot be deployed on the real arm without re-timing.

**Reproducing the historical 80% number bit-exactly requires 15 Hz** (those weights
were trained under it). **Building anything you intend to put on the robot requires
30 Hz.** These are different goals — pick one deliberately:

| goal | frequency | notes |
|---|---|---|
| reproduce the archived 80% checkpoint's score | 15 Hz + old geometry | historical only; `so101_sft_pp6b/global_step_1000` is a 15 Hz policy |
| build a deployable policy | **30 Hz** | must re-generate demos and re-train; existing pp checkpoints are NOT reusable as-is |

Changing the frequency rescales three dependent quantities — change all of them
together (they live in three files; changing one silently forks the pipeline):

| quantity | 15/20 Hz (pp) | **30 Hz (correct)** |
|---|---|---|
| demo median length | ~181 steps | ~272 steps (×1.5) |
| `max_episode_steps` (env yaml **and** `@register_env`) | 240 | **400** (headroom 1.47; must be divisible by `num_action_chunks`) |
| decisions per episode (`budget / num_action_chunks`) | 48 | 80 |
| `global_batch_size` for **1 update/epoch** = `envs × decisions` | 128 × 48 = 6144 | **128 × 80 = 10240** |
| dataset `FPS` in the converter | 15 (wrong) | **30** |
| generator `gym.make(..., sim_config=dict(sim_freq=120, control_freq=30))` | absent (defaulted to 20) | **explicit** |

Also scale the generator's motion-interpolation step counts by 1.5× and add settle
frames after the gripper closes and at the home pose — at 30 Hz a fixed number of
interpolation steps spans less wall-clock time, which broke the planner outright
(probe fell to 1/12 until the step counts were rescaled; 9/12 afterwards).

Verify the whole chain with `python -m toolkits.invariant_audit` — its
`temporal-chain` check compares real dataset fps, sim `control_freq`, and every
converter's `FPS` constant, and fails if they disagree.

**Machine prerequisites** (both cost a full night when missed):
```bash
mount -o remount,size=16G /dev/shm          # container default 64MB breaks NCCL
find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime MUJOCO_GL=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99
```

---

## 2. Stage A — planner demonstrations (CPU, ~3 h)

Script: `scratchpad/gen_so101_demos.py` (in `SO101_SESSION_LOG.md` Part 3).

Settings that mattered:
- gripper CLOSED command `-0.8`; grasp from above via `compute_grasp_info_by_obb`
- **micro-lift verification**: after closing, lift 3 cm; if the cube rises < 1.5 cm,
  abort and retry with a jittered grasp point (contact flags fire on useless pinches)
- **payload-offset compensation**: the held cube hangs 2–4 cm off the TCP; measure
  `payload_xy − tcp_xy` after the grasp and aim the drop at the payload
- **two-stage FK transport**: vertical lift first, then traverse (5-DoF arms cannot
  hit arbitrary 6-DoF poses; do not chase Cartesian IK)
- **closed-loop pre-drop refinement**: re-measure cube error, correct ≤ 2×
- 3 retry variants per episode (grasp jitter ± 4 mm, drop height 0.07/0.08/0.09)

Gates before continuing:
1. probe N ≥ 8, success ≥ 50%
2. **median demo length × 1.2 ≤ episode budget** (150-step demos under an 80-step
   budget once made the task literally impossible)
3. batch run of 240 → keep the successes

pp-era result: **188 successful demos at 78% generator success**.
Law observed 3×: **BC ceiling ≈ generator success rate.**

---

## 3. Stage B — dataset + normalization (CPU, ~30 min)

- Convert only successful episodes; radians → LeRobot normalized units via the
  **same** `so101_calib` module used at RL time.
- fps must equal the generator's real control frequency (a hardcoded 15 while the
  generator ran 20 Hz silently corrupted a whole day).
- `python -m toolkits.lerobot.calculate_norm_stats --config-name pi05_so101_pp --repo-id so101-sim-demos-pp`
- Copy `norm_stats.json` **into every checkpoint dir** you will later load from.
- **Freeze the stats for the lineage.** Recomputing stats mid-lineage and continuing
  SFT degrades monotonically (measured: swapping stats alone cut an unchanged
  checkpoint from 19.5% → 9.4%).

---

## 4. Stage C — SFT (GPU 8×H200, ~1.5 h)

Config: `examples/sft/config/so101_sft_pp2.yaml`

| parameter | value |
|---|---|
| warm start | previous sim-SFT checkpoint (`so101_sft_pp/.../global_step_4000`) |
| lr / schedule | 2.5e-5, cosine, warmup 200, min 2.5e-6 |
| steps / save | 3000 / every 1000 (use **250** if you want the peak — it can appear < 1 epoch) |
| micro / global batch | 16 / 128 |
| `train_expert_only` | False (full fine-tune) |
| `num_action_chunks` | 10 (must equal the SFT action horizon) |

Gate: zero-shot eval on 128 envs — pp-era value **46.9%**. Below ~40% the RL stage
will not work (see §6).

---

## 5. Stage D — RL (GPU, ~3 h to the peak)

Config: `examples/embodiment/config/so101_ppo_openpi_pi05.yaml` plus these overrides:

```
actor.model.model_path=<SFT ckpt>            rollout.model.model_path=<same>
actor.model.openpi.config_name=pi05_so101_pp
env.train.ignore_terminations=True           # hold-style success must pay per step
actor.model.openpi.noise_params=[0.08,0.05,200]   # halved exploration noise
actor.optim.lr=2e-6                          algorithm.update_epoch=2
algorithm.clip_ratio_high=0.1                algorithm.clip_ratio_low=0.1
algorithm.entropy_bonus=0.001
env.train.total_num_envs=128                 env.train.max_episode_steps=240
env.train.max_steps_per_rollout_epoch=240    env.eval.total_num_envs=128
actor.global_batch_size=6144                 runner.save_interval=10
```

**The one invariant that decides success or failure:**
`global_batch_size = num_envs × (budget / num_action_chunks)` → exactly **one
update per epoch**. With 128 envs, 240 steps, chunk 5: 128 × 48 = 6144. When a
budget change silently made this 3 updates/epoch, eval fell 71.9% → 30%.

**Harvest discipline** (this cost 180 wasted epochs once):
- `save_interval ≤ val_check_interval` — the peak checkpoint must be saved
- auto-stop **inside the launcher**, not in your head:
  stop if `eval < best − 0.20`, or `eval < zero-shot baseline` for 3 consecutive evals
- expected shape: climb to a peak around epoch 100, plateau ~40 epochs, then
  monotonic decay. pp-era: 46.9% → **75.0% @ step_100** → 10.9% by step_320.

Take `global_step_100`. Do not warm-start from anything after the peak.

---

## 6. Stage E — expert iteration ×2 (this is what actually reached 80%)

RL gave 46.9 → 75 (fixed-seed; 63.3% on fresh seeds). The last 17 points came from
distillation, which is **zero-risk** and cheap:

1. **Collect**: run deterministic eval of the current best policy on 8 fresh seeds
   with `SO101_COLLECT_DIR=<dir>` set (the recorder in `ManiskillEnv` flushes every
   successful episode as npz). pp-era: 648 successes from 8 × 128 envs.
2. **Convert** those npz to a LeRobot dataset (same unit conversion, same fps).
3. **Gentle SFT** from the *same* policy: lr **1e-5** (not 2.5e-5), 2000 steps,
   save every 500, **reuse the existing norm_stats**.
4. Gate → repeat.

Measured: round 1 **63.3% → 81.6%** (+18); round 2 81.6% → ~80% (saturated).

---

## 7. Evaluation protocol (or your numbers will lie to you)

- **Gate seeds and verification seeds must be disjoint.** Selecting the best
  checkpoint on seeds A,B and reporting its A,B score inflated a 63% policy to 75%.
  Protocol: select on 777/888 → re-measure on ≥2 never-used seeds → report the
  verification number first.
- **Report per-region success**, never only the mean: an "80%" policy had a corner
  at 58%.
- **Gate every saved checkpoint.** Single-checkpoint scores are samples from a noisy
  process (5.5% → 22.7% → 12.5% at 1000-step spacing on the true task).
- 128 envs per eval; more seeds beat more envs.

---

## 8. Pitfall checklist (each one cost ≥ half a day here)

1. `/dev/shm` 64 MB + stale `cuda.shm.*` → `ncclSystemError`, misread as GPU OOM.
2. Editing in-tree env code while a run is live → later evals silently use new code.
3. `micro_batch_size` copied from a 128×128-image recipe → genuine OOM at 640×480.
4. Frequency defined in three places (env yaml, generator's own `gym.make`,
   converter's `FPS`) — change one, grep all.
5. Success that terminates the episode can pay *less* than not succeeding —
   check the per-step arithmetic; use `ignore_terminations=True` for hold-style goals.
6. Contact-based `is_grasping` is true for pinches that cannot lift the cube.
7. Warm-starting from a checkpoint trained under a defective reward poisons every
   descendant.

---

## 9. What still exists on disk (2026-08-12)

| artifact | status |
|---|---|
| `so101_sft_pp6b/.../global_step_1000` (**the 80% policy**) | ✅ kept |
| `so101_sft_pp5/.../global_step_2000` (81.6%) | ✅ kept |
| `so101_sft_openpi_pi05/.../global_step_8000` (real-data SFT) | ✅ kept |
| pp-era datasets (`so101-sim-demos-pp*`) | ❌ deleted — regenerate via §2–3 |
| pp-era RL checkpoints (pp4 step_100 = 75%) | ❌ deleted |
| SFT configs `so101_sft_pp*.yaml`, RL config `so101_ppo_openpi_pi05.yaml` | ✅ kept |
| generator / converter / launcher scripts | ✅ in `SO101_SESSION_LOG.md` Part 3 |

To re-evaluate the 80% policy today you must first restore the §1 environment
constants — under the current TRUE-task env its number will be much lower, because
it has never seen 91% of those spawn positions.

---

## 10. Code changes required (complete list, as of the pp-80% result)

These are all the in-tree modifications the pp-era pipeline depended on. Full
current text of every file is in `SO101_SESSION_LOG.md` Part 2.

### 10.1 New files (the SO101 integration)

| file | purpose |
|---|---|
| `rlinf/envs/maniskill/tasks/so101_pick_place.py` | the task env `SO101GrabRedCube-v1` (auto-registered by dropping it in `tasks/`): scene (board, red+blue cubes, open tray), spawn logic, `evaluate()`, `compute_dense_reward()`, front + wrist `CameraConfig`, `get_language_instruction()` |
| `rlinf/envs/maniskill/so101_agent.py` | robot uid `so101` = ManiSkill's `SO100` subclass pointing at a **widened-limit URDF** (the stock URDF under-models the real arm: real elbow reaches −2.37 rad vs −1.5708 in stock) |
| `rlinf/envs/maniskill/assets/so101/so101.urdf` | joint limits widened to the union of stock + real servo calibration: pan ±2.0, lift [−1.5708, 2.48], elbow [−2.38, 1.5708], wrist_flex [−3.01, 1.8], wrist_roll ±3.14159, gripper ±1.1 |
| `rlinf/envs/maniskill/so101_calib.py` | LeRobot-normalized ↔ radians conversion (see 10.3) |
| `rlinf/models/embodiment/openpi/policies/so101_policy.py` | `SO101Inputs` / `SO101Outputs`: 6-dim joint state + front/wrist images + prompt |
| `rlinf/models/embodiment/openpi/dataconfig/so101_dataconfig.py` | `LeRobotSO101DataConfig`; **`action_sequence_keys=("action",)`** (LeRobot 0.6.1 uses the singular column), `extra_delta_transform=False` (absolute joint targets) |
| `toolkits/preflight_config.py` | CPU pre-launch validator: hydra compose + `validate_cfg` + path existence + batch arithmetic (samples/epoch vs global_batch vs per-rank micro). Catches ~half of all first-launch failures in seconds |

### 10.2 Modified files

**`rlinf/envs/action_utils.py`** — route SO101 actions through the unit conversion
(without this the arm slams into its limits):
```python
def prepare_actions_for_maniskill(raw_chunk_actions, num_action_chunks, action_dim, action_scale, policy):
    if "so100" in policy or "so101" in policy:
        from rlinf.envs.maniskill.so101_calib import norm_to_rad
        return norm_to_rad(raw_chunk_actions)
    if "panda" in policy:
        return raw_chunk_actions
```

**`rlinf/config.py`** (`get_robot_control_mode`) — otherwise "Robot so100 not supported":
```python
elif "so100" in robot or "so101" in robot:
    return "pd_joint_pos"
```

**`rlinf/envs/maniskill/maniskill_env.py`** — three additive changes:
1. `_wrap_obs` emits `wrist_images` when a `wrist_camera` sensor exists (backward
   compatible: envs without one get `None`, which `prepare_observations` backfills);
2. proprioception converted radians → LeRobot-normalized when the env cfg sets
   `so101_state_norm: True`;
3. **rollout recorder** for expert iteration: with `SO101_COLLECT_DIR` set, every
   episode's (front, wrist, state, action) is buffered and flushed to `.npz` on the
   first success — partial-reset safe.

**`rlinf/models/embodiment/openpi/dataconfig/__init__.py`** — register one
`TrainConfig` per dataset generation (`pi05_so101`, `pi05_so101_sim`,
`pi05_so101_pp`, …), each with `Pi0Config(pi05=True, action_horizon=10,
discrete_state_input=True)` and `pytorch_weight_path="checkpoints/torch/pi05_base"`.

**`rlinf/models/embodiment/openpi/openpi_action_model.py`** — honor
`detach_critic_input` on the VLM value path (it previously only covered the suffix
path, so critic gradients could reach shared features).

**Machine-specific in-tree fixes (do not revert on this box):**
- `rlinf/scheduler/collective/collective_group.py`: bracket IPv6 addresses in the
  `tcp://` rendezvous URL, and honor `RLINF_MASTER_ADDR_OVERRIDE` (pin single-node
  collectives to IPv4 loopback).
- `rlinf/scheduler/cluster/cluster.py`: `find_free_port` probes dual-stack
  (`AF_INET6`, `IPV6_V6ONLY=0`). IPv4-only probing published ports already taken on
  IPv6 → flaky `TCPStore recvValue failed` → silent first-rollout hangs. Worth
  upstreaming.

### 10.3 The calibration module (the single most error-prone file)

`so101_calib.py` converts between the dataset's LeRobot-normalized units and sim
radians. Facts that were established by measurement, each of which cost time:

- LeRobot writes `homing_offset` into the servo HW register, so `range_min/max`
  are already in the **homed frame centered at tick 2048**; physical angle =
  `(tick − 2048) · 2π/4096`. Do **not** subtract `homing_offset` in software.
- `SIGN = [1,1,1,1,1,1]` (sign flips made the replay worse — verified against real video).
- `OFFSET = [0, 0, 0, +0.6, 0, 0]` rad: wrist_flex offset makes the gripper point
  down-forward as in the real demos.
- **The gripper needs its own map, not the arm's tick→rad conversion.** Reusing the
  arm conversion left a minimum jaw gap of 8.1 cm against a 2.9 cm cube — the
  gripper could never close, which is why a 12-hour RL run scored exactly zero:
  ```python
  GRIPPER_RAD_CLOSED = -1.0   # LeRobot gripper norm 0   -> ~1.4 cm jaw gap
  GRIPPER_RAD_OPEN   = +0.5   # LeRobot gripper norm 100 -> ~11 cm
  ```
- All joints are clipped to `JOINT_LIMITS_LOW/HIGH` in `norm_to_rad` (real
  calibrations contain uncalibrated full-turn ranges, e.g. wrist_roll min 0 max 4095).
- Round-trip `norm → rad → norm` must be ≈1e-6; validate the **full dataset range**
  of every joint against the sim limits, not just the suspicious one.

### 10.4 Reward and success (task file, pp era)

```python
reward  = 1 - tanh(5 · d_tcp_to_cube)                       # reach
reward += 0.5 · (d < 0.04) · closedness                     # gradient bridge for the gripper
reward += is_grasped                                        # binary contact jump
reward += clamp(lift / LIFT_HEIGHT, 0, 1) · is_grasped      # lift
reward += 2.0 · (1 - tanh(5 · d_cube_to_box_waypoint)) · is_grasped   # transport
reward[success] = 8                                         # success dominates
success = in_box & ~is_grasped                              # released inside the tray
```
Arithmetic checked **before** launching: hover-while-holding maxes at ≈5.4/step,
released-in-box pays 8/step, so releasing strictly dominates holding. With
`ignore_terminations=True` the success state pays every step.

Two shaping mistakes that had to be removed (both cost a day each):
- a `0.3 · openness` term while far: its per-episode time integral (~60 steps) beat
  the `0.5 · closedness` near-term (~5 steps) → permanent hover-open;
- "keep the gripper open while approaching" — the exact opposite of the demos.

### 10.5 Config files

| file | role |
|---|---|
| `examples/embodiment/config/env/maniskill_so101_pick_place.yaml` | env id, `control_mode: pd_joint_pos`, `so101_state_norm: True`, `sim_freq/control_freq`, budget, `reward_mode: normalized_dense` |
| `examples/embodiment/config/so101_ppo_openpi_pi05.yaml` | RL config (actor+rollout `openpi`, `add_value_head: True`, `num_action_chunks: 5`, `action_dim: 6`, `policy_setup: so100`, `openpi.config_name`, `num_images_in_input: 2`) |
| `examples/embodiment/config/so101_eval_openpi_pi05.yaml` | eval-only; note it reads **`rollout.model`** (`model/pi0_5@rollout.model`), not `actor.model` |
| `examples/sft/config/so101_sft_pp*.yaml` | SFT rounds (dataset repo id, warm-start ckpt, norm_stats path, lr/steps/save) |

Hydra notes: use an **absolute** `--config-path`; overrides for keys that do not
exist in the config need `+key=value`, existing keys need `++key=value` to force.

---

## 11. Exact code, step by step

Everything below is extracted verbatim from the running source, so it cannot drift from what actually produced the results. Paths are repo-relative.

### Step 1 — robot: widened-limit SO101 agent

`rlinf/envs/maniskill/so101_agent.py` (whole file). Importing it registers uid `so101`; the task's `__init__` does that import.

```python
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
```

URDF limit changes vs the stock SO100 (radians):

| joint | stock | widened (real servo calibration) |
|---|---|---|
| shoulder_pan | ±1.5708 | **±2.0** |
| shoulder_lift | ±1.5708 | **[-1.5708, 2.48]** |
| elbow_flex | ±1.5708 | **[-2.38, 1.5708]** |
| wrist_flex | ±1.8 | **[-3.01, 1.8]** |
| wrist_roll | ±3.14159 | unchanged |
| gripper | ±1.1 | unchanged |

### Step 2 — units: the calibration module

`rlinf/envs/maniskill/so101_calib.py` (whole file). This is the highest-risk file in the project: an arm-style conversion applied to the gripper left an 8.1 cm minimum jaw gap on a 2.9 cm cube, which is why one 12-hour RL run scored exactly zero.

```python
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
"""SO101 LeRobot <-> ManiSkill unit conversion.

The LeRobot SO101 dataset stores joint values in LeRobot's NORMALIZED units, not
radians: the 5 arm joints use RANGE_M100_100 (range_min -> -100, range_max ->
+100) and the gripper uses RANGE_0_100 (range_min -> 0, range_max -> 100). See
``lerobot/motors/motors_bus.py::_normalize``. ManiSkill's ``pd_joint_pos``
controller and ``get_qpos`` are in RADIANS.

This module maps between the two using the follower calibration (raw feetech
ticks, 4096 ticks / revolution). Physical joint angle = (raw - homing_offset) *
2*pi/4096. The ManiSkill SO100 URDF joint zero is assumed to coincide with the
real homed zero; per-joint zero/direction offsets that remain after this
conversion are a sim2real calibration detail to tune against the real arm.

NOTE: replace SO101_CALIB with YOUR follower calibration if the robot is
recalibrated.
"""
import numpy as np

TICKS_PER_REV = 4096
RAD_PER_TICK = 2.0 * np.pi / TICKS_PER_REV
# LeRobot writes homing_offset into the servo HW register, so the position it
# reads back is already homed: the homed "zero" sits at max_res/2 = 2048 ticks
# (see feetech `_get_half_turn_homings`: homing = pos - max_res/2). So range_min/
# range_max are in this homed frame and the physical angle = (tick - CENTER)*rad.
CENTER_TICK = TICKS_PER_REV / 2  # 2048

# Joint order matches the dataset action/state layout:
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
SO101_CALIB = [
    {"name": "shoulder_pan", "min": 1110, "max": 3211, "homing": 1919, "mode": "m100"},
    {"name": "shoulder_lift", "min": 1364, "max": 3661, "homing": 1812, "mode": "m100"},
    {"name": "elbow_flex", "min": 506, "max": 2736, "homing": 1306, "mode": "m100"},
    {"name": "wrist_flex", "min": 90, "max": 2393, "homing": -1918, "mode": "m100"},
    {"name": "wrist_roll", "min": 0, "max": 4095, "homing": -1348, "mode": "m100"},
    {"name": "gripper", "min": 2023, "max": 2544, "homing": 963, "mode": "r0_100"},
]

_MIN = np.array([j["min"] for j in SO101_CALIB], dtype=np.float64)
_MAX = np.array([j["max"] for j in SO101_CALIB], dtype=np.float64)
_IS_GRIPPER = np.array([j["mode"] == "r0_100" for j in SO101_CALIB])

# norm -> homed tick:  tick = norm * _RAW_SCALE + _RAW_OFFSET
_RAW_SCALE = np.where(_IS_GRIPPER, (_MAX - _MIN) / 100.0, (_MAX - _MIN) / 200.0)
_RAW_OFFSET = np.where(_IS_GRIPPER, _MIN, (_MAX + _MIN) / 2.0)

# Per-joint sim<->real ALIGNMENT (calibrated against the real front-cam video):
# the ManiSkill SO100 URDF joint axes/zeros differ from the real SO101 servo
# convention, so each joint may need a direction flip (SIGN) and a zero offset
# (OFFSET, radians). Applied as: q_sim = SIGN * (tick-CENTER)*rad + OFFSET.
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# wrist_flex +0.6 rad: tilts the gripper down-forward to grab table objects (the
# generic SO100 wrist zero left it pointing forward = "grabbing air"). Verified
# against the real wrist-cam view (gripper looks down-forward at the red cube).
OFFSET = np.array([0.0, 0.0, 0.0, 0.6, 0.0, 0.0])

# ManiSkill SO100 URDF joint limits (radians), same order as SO101_CALIB. The
# real dataset commands over-travel a few joints at their trajectory extremes
# (shoulder_lift +1.70 > +1.57, elbow_flex -1.67 < -1.57, and especially
# wrist_roll -3.65 < -3.14 because its follower calibration uses the full
# uncalibrated 0..4095 tick range). The articulation would clamp these anyway;
# clipping the *commanded target* here keeps it well-defined. The operative
# grasp-phase values sit comfortably inside the limits, so clipping only trims
# rare extreme frames and does not disturb the validated reaching behaviour.
# Limits match the WIDENED so101 URDF (union of stock SO100 limits and the
# real servo calibration ranges — the stock URDF under-modeled the hardware;
# see rlinf/envs/maniskill/so101_agent.py).
JOINT_LIMITS_LOW = np.array([-2.0, -1.5708, -2.38, -3.01, -3.14159, -1.1])
JOINT_LIMITS_HIGH = np.array([2.0, 2.48, 1.5708, 1.8, 3.14159, 1.1])

# Gripper is a special case: the real feetech parallel gripper and the sim SO100
# revolute jaw have entirely different linkage geometry, so the arm-joint
# tick->rad conversion does NOT apply (it left the jaws stuck at an 8.1 cm gap,
# far wider than the 2.9 cm cube, so the gripper could never close on it). We map
# the LeRobot normalized gripper (0 = fully-closed grasp, 100 = fully open)
# directly onto a sim jaw angle that actually opens/closes around the cube.
# Measured jaw-tip gap vs gripper qpos in sim:
#   qpos -1.0 rad -> ~1.4 cm gap (drives past the 2.9 cm cube -> firm press/grasp)
#   qpos -0.34    -> ~6   cm gap (norm ~44 approach: clears the cube)
#   qpos +0.5 rad -> ~11  cm gap (norm 100 fully open)
GRIPPER_IDX = 5
GRIPPER_RAD_CLOSED = -1.0  # LeRobot gripper norm 0
GRIPPER_RAD_OPEN = 0.5  # LeRobot gripper norm 100
_GRIPPER_SPAN = GRIPPER_RAD_OPEN - GRIPPER_RAD_CLOSED


def norm_to_rad(actions):
    """LeRobot-normalized joint targets (last dim = 6) -> ManiSkill radians."""
    a = np.asarray(actions, dtype=np.float64)
    tick = a * _RAW_SCALE + _RAW_OFFSET
    base = (tick - CENTER_TICK) * RAD_PER_TICK
    rad = SIGN * base + OFFSET
    # Gripper: override the arm-style conversion with the dedicated close<->open
    # map so the jaws actually reach the cube.
    rad[..., GRIPPER_IDX] = (
        GRIPPER_RAD_CLOSED + (a[..., GRIPPER_IDX] / 100.0) * _GRIPPER_SPAN
    )
    rad = np.clip(rad, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
    return rad.astype(np.float32)


def rad_to_norm(qpos):
    """ManiSkill joint radians (last dim = 6) -> LeRobot-normalized units."""
    q = np.asarray(qpos, dtype=np.float64)
    base = (q - OFFSET) / SIGN
    tick = base / RAD_PER_TICK + CENTER_TICK
    norm = (tick - _RAW_OFFSET) / _RAW_SCALE
    # Gripper: inverse of the dedicated close<->open map (used for state feedback).
    norm[..., GRIPPER_IDX] = (
        (q[..., GRIPPER_IDX] - GRIPPER_RAD_CLOSED) / _GRIPPER_SPAN * 100.0
    )
    return norm.astype(np.float32)
```

### Step 3 — action path: route SO101 through the conversion

`rlinf/envs/action_utils.py`

```python
def prepare_actions_for_maniskill(
    raw_chunk_actions,
    num_action_chunks,
    action_dim,
    action_scale,
    policy,
) -> torch.Tensor:
    if "so100" in policy or "so101" in policy:
        # The SO101 LeRobot dataset stores joint targets in LeRobot NORMALIZED
        # units (arm: [-100,100], gripper: [0,100]), NOT radians. ManiSkill's
        # pd_joint_pos wants radians. Map via the follower calibration; without
        # this the arm slams to its joint limits / moves the wrong way.
        from rlinf.envs.maniskill.so101_calib import norm_to_rad

        return norm_to_rad(raw_chunk_actions)
    if "panda" in policy:
        # Panda EE-pose policies already emit env-ready actions [num_envs,
        # num_action_chunks, action_dim] for the pd_ee_* controller. Pass through.
        return raw_chunk_actions
```

`rlinf/config.py` — `get_robot_control_mode` (else: "Robot so100 not supported")

```python
                elif "google_robot_static" in robot:
                    return "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
                elif "widowx" in robot:
                    return "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
                elif "so100" in robot or "so101" in robot:
                    # SO100/SO101: absolute joint-position control (the 6-dim
                    # PI0.5 output maps straight onto the arm joints + gripper).
                    return "pd_joint_pos"
                elif "panda" in robot:
```

### Step 4 — observations: wrist camera + normalized state + rollout recorder

`rlinf/envs/maniskill/maniskill_env.py`, default branch of `_wrap_obs` (additive: envs without a wrist camera get `None`, which `prepare_observations` backfills)

```python
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
```

Recorder used by expert iteration (Step 9). Enabled with `SO101_COLLECT_DIR`; flushes each episode as `.npz` on its first success, partial-reset safe.

```python
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
```

### Step 5 — the task: scene, spawn, success, reward

`rlinf/envs/maniskill/tasks/so101_pick_place.py`. Cameras (the pp era used `128, 128` with an fov; the current file renders 640×480 with measured intrinsics — keep 640×480, see §1a):

```python
    def _default_sensor_configs(self):
        front_pose = sapien_utils.look_at(
            eye=FRONT_CAM_EYE, target=FRONT_CAM_TARGET, up=FRONT_CAM_UP
        )
        wrist_pose = sapien_utils.look_at(eye=WRIST_CAM_EYE, target=WRIST_CAM_TARGET)
        wrist_mount = self.agent.robot.links_map[WRIST_CAMERA_MOUNT_LINK]
        return [
            CameraConfig(
                "3rd_view_camera", front_pose, FRONT_CAM_W, FRONT_CAM_H, None,
                0.01, 100, intrinsic=FRONT_CAM_INTRINSIC,
            ),
            CameraConfig(
                "wrist_camera", wrist_pose, FRONT_CAM_W, FRONT_CAM_H, None,
                0.01, 100, intrinsic=WRIST_CAM_INTRINSIC, mount=wrist_mount,
            ),
        ]
```

Spawn. `SO101_SPAWN_MODE=legacy` is the pp-era 6×8 cm box — the one thing you narrow to reproduce pp conditions at full fidelity:

```python
                ]).repeat(b, 1)))

            # Red target cube: ALWAYS spawned within the board (with margin) and
            # inside the SO100 reach -- far half (+x toward the box), image-left (+y).
            # SO101_SPAWN_FRAC="x0,x1,y0,y1" (fractions of the default ranges)
            # restricts the spawn sub-box, e.g. for hard-region targeted data
            # collection. Default = full box.
            # USER-CONFIRMED (2026-08-09): BOTH cubes may appear ANYWHERE on the
            # board (measured: all 87 real first frames span the full board).
            # Full-board spawn is the DEFAULT; SO101_SPAWN_MODE=legacy reproduces
            # the old 6x8cm sub-box (for historical comparisons only), and
            # SO101_SPAWN_FRAC="x0,x1,y0,y1" targets a sub-region of the active
            # ranges (for weak-region data collection).
            frac_x0, frac_x1, frac_y0, frac_y1 = 0.0, 1.0, 0.0, 1.0
            _frac = os.environ.get("SO101_SPAWN_FRAC")
            if _frac:
                frac_x0, frac_x1, frac_y0, frac_y1 = (float(v) for v in _frac.split(","))
            red = torch.zeros((b, 3))
            blue = torch.zeros((b, 3))
            if os.environ.get("SO101_SPAWN_MODE") == "legacy":
                red[:, 0] = board_center_x + (frac_x0 + torch.rand((b,)) * (frac_x1 - frac_x0)) * 0.06
                red[:, 1] = cy + 0.02 + (frac_y0 + torch.rand((b,)) * (frac_y1 - frac_y0)) * 0.08
                blue[:, 0] = board_center_x - 0.05 + torch.rand((b,)) * 0.04
                blue[:, 1] = cy - 0.06 + torch.rand((b,)) * 0.04
            else:
                # Spawn zone = BROWN region only (user: cubes never on the black
                # end), centered on the base axis. Margin 2cm (design choice).
                spawn_margin = 0.02
                spawn_x_lo, spawn_x_hi = -BOARD_HALF[0] + spawn_margin, BOARD_HALF[0] - spawn_margin
                spawn_y_lo, spawn_y_hi = -BROWN_HALF_Y + spawn_margin, BROWN_HALF_Y - spawn_margin
                red[:, 0] = board_center_x + spawn_x_lo + (frac_x0 + torch.rand((b,)) * (frac_x1 - frac_x0)) * (spawn_x_hi - spawn_x_lo)
                red[:, 1] = cy + spawn_y_lo + (frac_y0 + torch.rand((b,)) * (frac_y1 - frac_y0)) * (spawn_y_hi - spawn_y_lo)
                # blue: same brown zone; USER SPEC: no minimum separation, only
                # physical non-overlap (3cm ~= touching cubes)
                blue[:, 0] = board_center_x + spawn_x_lo + torch.rand((b,)) * (spawn_x_hi - spawn_x_lo)
                blue[:, 1] = cy + spawn_y_lo + torch.rand((b,)) * (spawn_y_hi - spawn_y_lo)
                for _ in range(10):
                    bad = (
                        torch.linalg.norm(blue[:, :2] - red[:, :2], axis=1) < 0.03
                    )
                    if not bad.any():
                        break
                    nb = int(bad.sum())
                    blue[bad, 0] = board_center_x + spawn_x_lo + torch.rand((nb,)) * (spawn_x_hi - spawn_x_lo)
                    blue[bad, 1] = cy + spawn_y_lo + torch.rand((nb,)) * (spawn_y_hi - spawn_y_lo)
            red[:, 2] = board_top + CUBE_HALF
```

Success (the pp era had no homing term; the current file requires it):

```python
    def evaluate(self):
        import os

        if os.environ.get("SO101_LOG_DIST"):
            d = torch.linalg.norm(
                self.red_cube.pose.p - self.agent.tcp_pose.p, axis=1
            )
            g = self.agent.is_grasping(self.red_cube)
            grip_q = self.agent.robot.get_qpos()[:, -1]  # gripper joint angle
            gap = torch.linalg.norm(
                self.agent.finger1_tip.pose.p - self.agent.finger2_tip.pose.p,
                axis=1,
            )
            print(
                f"[TCPDIST] min={float(d.min()):.4f} mean={float(d.mean()):.4f} "
                f"grasped={int(g.sum())}/{len(g)} "
                f"grip_q[min={float(grip_q.min()):.2f} max={float(grip_q.max()):.2f}] "
                f"jaw_gap[min={float(gap.min()):.3f} max={float(gap.max()):.3f}]",
                flush=True,
            )
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
```

Reward — the proven manipulation recipe plus ONE gradient bridge for the gripper:

```python
    def compute_dense_reward(self, obs, action, info):
        # Proven grasp-reward recipe (ManiSkill PickCube / lerobot-sim2real
        # SO100GraspCube): reach gradient + binary contact grasp + post-grasp
        # progress gated by is_grasped. NO gripper open/close shaping — every
        # successful reference reward leaves gripper timing to the policy; the
        # binary is_grasped jump (+1 AND unlocking the lift term) is what makes
        # grasping strictly dominate hovering.
        tcp_to_obj = torch.linalg.norm(
            self.red_cube.pose.p - self.agent.tcp_pose.p, axis=1)
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
            (self.red_cube.pose.p[:, 2] - self._red_start_z) / LIFT_HEIGHT, 0.0, 1.0)
        reward = reward + lift * is_grasped
        # Phase 2 transport: while holding the cube, pull it toward a waypoint
        # above the tray. Gated by is_grasped so it cannot be farmed empty-handed.
        # Anti-hack arithmetic (checked BEFORE launch): hover-over-tray while
        # holding maxes at ~5.4/step, released-in-box success pays 8/step ->
        # releasing strictly dominates. Episodes do not terminate on success
        # (env.train.ignore_terminations=True), so success pays every step.
        box_target = self.box.pose.p.clone()
        box_target[:, 2] = box_target[:, 2] + 0.10
        cube_to_box = torch.linalg.norm(
            self.red_cube.pose.p - box_target, axis=1)
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
```

### Step 6 — model wiring: policy transform + data config + TrainConfig

`rlinf/models/embodiment/openpi/policies/so101_policy.py` (whole file)

```python
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
"""OpenPI input/output transforms for the SO101/SO100 6-DoF joint-space arm.

Mirrors ``maniskill_policy.py`` but for a 6-dim joint state/action layout
(``shoulder_pan``, ``shoulder_lift``, ``elbow_flex``, ``wrist_flex``,
``wrist_roll``, ``gripper``). State and actions are absolute joint positions,
matching both the ManiSkill ``so100`` ``pd_joint_pos`` controller and the
LeRobot-trained PI0.5 checkpoint.
"""
import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

# SO101/SO100 active-joint dimension: 5 arm joints + 1 gripper.
SO101_ACTION_DIM = 6


def make_so101_example() -> dict:
    """Creates a random input example for the SO101 policy."""
    return {
        "observation/state": np.random.rand(SO101_ACTION_DIM),
        "observation/image": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Convert env/dataset inputs to the model format (training and inference)."""

    # Determines which model will be used. Do not change this for your own dataset.
    model_type: _model.ModelType
    use_wrist_image: bool = False
    default_prompt: str | None = None

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = (
            _parse_image(data["observation/wrist_image"])
            if self.use_wrist_image and data.get("observation/wrist_image") is not None
            else np.zeros_like(base_image)
        )
        has_wrist_image = (
            self.use_wrist_image and data.get("observation/wrist_image") is not None
        )

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if has_wrist_image else np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]
        elif self.default_prompt is not None:
            inputs["prompt"] = self.default_prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Convert model outputs back to the 6-dim SO101 joint action (inference only)."""

    output_action_dim: int = SO101_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        # The model action tensor is padded to the model action dim; slice out the
        # first ``output_action_dim`` (= 6) as the env-frame SO101 joint targets.
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}
```

`rlinf/models/embodiment/openpi/dataconfig/so101_dataconfig.py` (whole file). `action_sequence_keys=("action",)` is required by LeRobot 0.6.1's singular column.

```python
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
"""OpenPI DataConfig for the SO101/SO100 6-DoF joint-space arm."""
import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import so101_policy


@dataclasses.dataclass(frozen=True)
class LeRobotSO101DataConfig(DataConfigFactory):
    """Transform pipeline for a LeRobot SO101 pick-place dataset + PI0.5.

    Actions/state are absolute 6-dim joint positions, so ``extra_delta_transform``
    defaults to ``False`` (no delta conversion). If your LeRobot dataset stores
    *delta* joint actions instead, set it to ``True``.
    """

    # SO101 joint actions are absolute -> no delta conversion by default.
    extra_delta_transform: bool = False
    # LeRobot 0.6.1 datasets name the action column "action" (singular); openpi's
    # default is "actions". This drives delta_timestamps (the action-chunk loader).
    action_sequence_keys: tuple = ("action",)

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        # Repack maps *dataset* keys -> pipeline keys (training only, not inference).
        # henry-guo/so101-pick-place-v2 is a dual-camera dataset: a fixed front view
        # and a wrist view. Match these keys to your dataset ``meta/info.json``
        # ``features`` (the wrist key is commonly ``observation.images.wrist``).
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.front",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                so101_policy.SO101Inputs(
                    model_type=model_config.model_type, use_wrist_image=True
                )
            ],
            outputs=[so101_policy.SO101Outputs()],
        )

        if self.extra_delta_transform:
            # Delta on the 5 arm joints, absolute gripper.
            delta_action_mask = _transforms.make_bool_mask(5, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
```

One `TrainConfig` per dataset generation, in `dataconfig/__init__.py::_CONFIGS`:

```python
    TrainConfig(
        # Phase-2 (pick-and-place) sim demos: planner grasps the red cube AND
        # places it into the tray. Same conventions as pi05_so101_sim.
        name="pi05_so101_pp",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=True
        ),
        data=LeRobotSO101DataConfig(
            repo_id="so101-sim-demos-pp",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi05_base/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
    ),
```

### Step 7 — demonstration generator (the planner)

`scratchpad/gen_so101_demos.py` — grasp, micro-lift verification, payload-offset compensation, two-stage FK transport, closed-loop pre-drop refinement, homing.

```python
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
```

### Step 8 — dataset conversion

`scratchpad/convert_v4_demos.py` — successful episodes only, units via the SAME calib module used at RL time, `FPS` = the generator's real control frequency.

```python
"""Convert stratified planner demos (h5, TRUE-task env) -> LeRobot so101-sim-demos-v4.

Images are 160x120 (4:3, matches the real 640x480 through resize_with_pad).
Only successful episodes; unit conversion via so101_calib (same module as RL).
"""
import glob
import json
import shutil
import sys

sys.path.insert(0, "/data08/henryg/pai/RLinf")

import h5py
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from rlinf.envs.maniskill.so101_calib import rad_to_norm

SRC_GLOB = "/data08/henryg/pai/data/v4_demos_cell*/**/*.h5"
OUT_REPO = "so101-sim-demos-v4"
OUT_ROOT = "/data08/henryg/pai/data/so101-sim-demos-v4"
TASK = "Grab the red cube"
FPS = 30  # = actual generator control_freq = real dataset fps
MAX_LEN = 580  # budget 640 / 1.1

files = sorted(glob.glob(SRC_GLOB, recursive=True))
print(f"h5 files: {len(files)}")

shutil.rmtree(OUT_ROOT, ignore_errors=True)

features = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
    "observation.images.front": {
        "dtype": "video", "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.wrist": {
        "dtype": "video", "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
}

ds = LeRobotDataset.create(
    repo_id=OUT_REPO, fps=FPS, root=OUT_ROOT, features=features, use_videos=True
)

kept, lens = 0, []
for h5path in files:
    meta = json.load(open(h5path.replace(".h5", ".json")))
    ok_ids = [e["episode_id"] for e in meta["episodes"] if e["success"]]
    f = h5py.File(h5path, "r")
    for eid in ok_ids:
        t = f[f"traj_{eid}"]
        qpos = np.asarray(t["obs/agent/qpos"], dtype=np.float64)
        acts = np.asarray(t["actions"], dtype=np.float64)
        front = np.asarray(t["obs/sensor_data/3rd_view_camera/rgb"])
        wrist = np.asarray(t["obs/sensor_data/wrist_camera/rgb"])
        T = acts.shape[0]
        if T > MAX_LEN or T < 80:
            continue
        state_n = rad_to_norm(qpos[:T])
        act_n = rad_to_norm(acts)
        for i in range(T):
            ds.add_frame(
                {
                    "observation.state": state_n[i].astype(np.float32),
                    "action": act_n[i].astype(np.float32),
                    "observation.images.front": front[i],
                    "observation.images.wrist": wrist[i],
                    "task": TASK,
                }
            )
        ds.save_episode()
        kept += 1
        lens.append(T)
    f.close()
    print(f"{h5path.split('/')[-2]}: kept so far {kept}", flush=True)

lens = np.array(lens)
print(f"DONE: {kept} episodes -> {OUT_ROOT}")
if kept:
    print(f"length median={int(np.median(lens))} p90={int(np.percentile(lens, 90))} max={int(lens.max())}")
```

### Step 9 — expert iteration (collect → convert → gentle SFT)

Collection is just a deterministic eval with the recorder enabled:

```bash
export SO101_COLLECT_DIR=/data08/henryg/pai/data/<name>_rollouts
for SEED in 101 202 303 404 505 606 707 808; do
  .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path <abs>/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
    rollout.model.model_path=$BEST_CKPT \
    rollout.model.openpi.config_name=<train config> \
    rollout.model.openpi_data.norm_stats_path=<stats> \
    env.eval.total_num_envs=128 env.eval.seed=$SEED
done
```

Then convert the `.npz` files (state is already normalized by the env; actions are radians and need `rad_to_norm`) and run SFT from the SAME policy at **lr 1e-5**, 2000 steps, reusing the existing `norm_stats`.

### Step 10 — verification tooling (run these, they are cheap)

```bash
# before every launch: hydra compose + validate_cfg + paths + batch arithmetic
python -m toolkits.preflight_config --config-path <abs cfg dir> --config-name <name> \
    <EXACT launcher overrides, verbatim>

# periodically: catches SILENT wrong-result defects (frequency chain, camera chain,
# spawn coverage vs the real dataset, stats lineage, dataset/env resolution,
# budget headroom, action chunking, eval-seed disjointness)
python -m toolkits.invariant_audit --ckpt <ckpt dir>
```
