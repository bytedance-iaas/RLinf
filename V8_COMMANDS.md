# V8 — every command, every parameter, with rationale

Target configuration: **full real-camera fidelity (640×480 @ 30 Hz, measured board
geometry, 8 g cube, success = in-box AND arm home) with the red-cube spawn narrowed
to the pp-era 6 × 8 cm box.** The narrowing is the only simplification; it exists to
restore pp-era demonstration density (0.44 cm spacing vs the grasp tolerance ±0.7 cm).

All paths are absolute as run on this machine. `$REPO = /data08/henryg/pai/RLinf`,
`$DATA = /data08/henryg/pai/data`, `$RES = /data08/henryg/pai/results`.

---

## Step 0 — session environment (must precede every step)

```bash
cd /data08/henryg/pai/RLinf
export REPO_PATH="$PWD" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# machine prerequisite — skipping this cost one full night
mount -o remount,size=16G /dev/shm
find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete
```

| variable | value | why |
|---|---|---|
| `VK_ICD_FILENAMES`, `LD_LIBRARY_PATH` | private driver libs in `.venv/nvidia_gl` | this box has a compute-only driver; apt's libnvidia-gl is version-mismatched and breaks rendering |
| `MUJOCO_GL` | `egl` | headless rendering |
| `RLINF_MASTER_ADDR_OVERRIDE`, `GLOO/NCCL_SOCKET_IFNAME` | `127.0.0.1`, `lo` | the node's IP is IPv6; pinning single-node collectives to IPv4 loopback removes a class of silent first-rollout hangs |
| `RAY_local_fs_capacity_threshold` | `0.99` | host `/tmp` sits ~96% full; Ray otherwise refuses to create objects |
| `/dev/shm` remount | 16 GB | container default is 64 MB; NCCL allocates ~7 MB shm segments per communicator and fails with `ncclSystemError` |

---

## Step 1 — generate demonstrations (CPU, ~1.8 h, 8 parallel workers)

```bash
export SO101_SPAWN_MODE=legacy          # THE only narrowing: 6x8 cm spawn box

for W in 0 1 2 3 4 5 6 7; do
  SEED=$((90000 + W*1000))
  .venv/bin/python scratchpad/gen_so101_demos.py \
      --num 32 \
      --seed0 $SEED \
      --out /data08/henryg/pai/data/v8_demos_w$W &
done
wait
```

| parameter | value | why |
|---|---|---|
| `SO101_SPAWN_MODE` | `legacy` | restricts the red cube to x ∈ [−0.534, −0.474], y ∈ [0.020, 0.100] (6 × 8 cm = 48 cm²), verified to lie inside the current brown zone. Everything else (cameras, 30 Hz, geometry, mass, homing) stays at true-task values |
| `--num` | 32 per worker (256 total) | targets ~175 successes; actual result **247/256 = 96.5%** |
| `--seed0` | 90000 + 1000·W | disjoint seed ranges so workers never generate identical episodes |
| workers | 8 | CPU-bound; ManiSkill CPU backend is single-threaded per process |

Internals that matter (already in the script): gripper CLOSED `-0.8`; **micro-lift
verification** (+3 cm, cube must rise >1.5 cm, else regrasp with jitter); **payload-offset
compensation** (the held cube hangs 2–4 cm off the TCP — aim the drop at the payload);
**two-stage FK transport** (vertical lift then traverse; 5-DoF arms cannot reach arbitrary
6-DoF poses); **closed-loop pre-drop refinement** (≤2 corrections); **homing segment**
(30 interpolation steps + 12 settle frames, required by the success criterion); 3 retry
variants per episode.

Gate: ≥120 successes, demo median length ≤530 steps. **Actual: 247 successes, median 357.**

---

## Step 2 — convert to a LeRobot dataset (CPU, ~1 h)

```bash
.venv/bin/python scratchpad/convert_v8_demos.py
```

| parameter (inside the script) | value | why |
|---|---|---|
| `SRC_GLOB` | `$DATA/v8_demos_w*/**/*.h5` | all 8 workers |
| `OUT_REPO` / `OUT_ROOT` | `so101-sim-demos-v8` | new dataset id, referenced by the TrainConfig |
| `FPS` | **30** | must equal the generator's real `control_freq`; a hardcoded 15 while the generator ran 20 Hz silently corrupted an earlier day of work |
| image shape | `(480, 640, 3)` | identical to the real dataset, so both go through openpi's `resize_with_pad` identically |
| `MIN_LEN` / `MAX_LEN` | 80 / 580 | 580 = episode budget 640 ÷ 1.1 headroom; shorter than 80 means a degenerate episode |
| unit conversion | `so101_calib.rad_to_norm` | the SAME module used at RL time, so SFT data and RL observations share one convention |

Result: **247 episodes, 88k frames, median 357, demo spacing √(48 cm²/247) = 0.44 cm**
(pp-era 0.51 cm; grasp tolerance ≈ ±0.7 cm).

---

## Step 3 — normalization statistics (no command — deliberately reused)

```bash
# NOT run for v8:
#   python -m toolkits.lerobot.calculate_norm_stats --config-name pi05_so101_v8 --repo-id so101-sim-demos-v8
# instead the v4 stats are reused:
#   /data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
```

Why: v8 continues the **v4 checkpoint lineage** (warm start = `so101_sft_v4/global_step_1000`).
A policy's action decoder is calibrated to the stats it was trained under; recomputing
mid-lineage degrades monotonically — measured: swapping stats alone cut an unchanged
checkpoint from 19.5% → 9.4%. Recompute only when starting from fresh base weights.

---

## Step 4 — baseline: what the warm start scores inside the box (GPU, ~6 min)

```bash
SO101_SPAWN_MODE=legacy \
.venv/bin/python evaluations/eval_embodied_agent.py \
  --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
  --config-name so101_eval_openpi_pi05 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v8 \
  rollout.model.model_path=$RES/so101_sft_v4/so101_sft_openpi_pi05/checkpoints/global_step_1000 \
  rollout.model.openpi.config_name=pi05_so101_v4 \
  rollout.model.openpi_data.norm_stats_path=$REPO/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
  env.eval.total_num_envs=128 \
  env.eval.seed=777
```

Purpose: the number v8 must beat. The same checkpoint scores 12.5% on the full board;
inside the box it should be higher, and the **difference** is what the new demos buy.
(This run failed once on a transient Ray connection error and is queued to be redone.)

---

## Step 5 — pre-flight the SFT config (CPU, seconds — never skip)

```bash
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v8 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v8
```

Runs hydra compose + RLinf `validate_cfg` + path existence + batch arithmetic on CPU.
This class of check caught roughly half of all first-launch failures in this project
(bad override paths, `%num_action_chunks` asserts, updates-per-epoch drift, missing files).
**Overrides must be copied verbatim from the launcher** — a partial override set validates
a different config than the one you will run.

---

## Step 6 — SFT (GPU 8×H200, ~85 min)

```bash
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v8 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v8
```

Config `examples/sft/config/so101_sft_v8.yaml`:

| parameter | value | why |
|---|---|---|
| `actor.model.model_path` | `$RES/so101_sft_v4/.../global_step_1000` | warm start in the SAME visual domain (640×480, 30 Hz). Starting from the real-data-only checkpoint was tried (v5) and was much worse |
| `data.train_data_paths` | `so101-sim-demos-v8` | the 247 in-box demos |
| `actor.model.openpi.config_name` | `pi05_so101_v8` | TrainConfig binding the repo id + `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=True)` |
| `openpi_data.norm_stats_path` | v4 stats | lineage frozen, see Step 3 |
| `optim.lr` / `min_lr` | 2.5e-5 / 2.5e-6, cosine, warmup 200 | the value used for every successful SFT round in this project |
| `runner.max_steps` | 4000 | ≈5.8 epochs over 88k frames |
| `runner.save_interval` | **250** (16 checkpoints) | the peak often appears in under one epoch; a coarse interval loses it. Costs ~446 GB of disk per run — check free space first |
| `actor.micro_batch_size` / `global_batch_size` | 16 / 128 | memory-only knobs; the optimizer step is unchanged (loss is scaled by 1/grad_accum, no BatchNorm in the model) |
| `train_expert_only` | False | full fine-tune, as in every SFT round here |

---

## Step 7 — gate: screen every checkpoint in the box (GPU, ~6 min each × 16)

```bash
export EMBODIED_PATH=$PWD/examples/embodiment
for CK in $RES/so101_sft_v8/so101_sft_openpi_pi05/checkpoints/global_step_*; do
  mkdir -p "$CK/so101-sim-demos-v4"
  cp $REPO/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json "$CK/so101-sim-demos-v4/"

  SO101_SPAWN_MODE=legacy \
  .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
    --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v8 \
    rollout.model.model_path=$CK \
    rollout.model.openpi.config_name=pi05_so101_v8 \
    rollout.model.openpi_data.norm_stats_path=$REPO/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
    env.eval.total_num_envs=128 \
    env.eval.seed=777
done
```

| parameter | value | why |
|---|---|---|
| copy of `norm_stats.json` into each ckpt dir | required | RLinf's rollout workers also look for stats inside the checkpoint directory |
| `env.eval.total_num_envs` | 128 | 128 episodes per eval; more seeds beat more envs for reliability |
| `env.eval.seed` | 777 | the **gate** seed — used for selection only |
| every checkpoint | not just the last | a single checkpoint's score is a sample from a noisy process (observed 5.5% → 22.7% → 12.5% at 1000-step spacing) |

---

## Step 8 — confirm the top 3 on a second gate seed (GPU, ~6 min each)

```bash
# same command as Step 7 with env.eval.seed=888, run only for the best 3 checkpoints
```
Selection = mean of seeds 777 and 888.

---

## Step 9 — honest verification on never-used seeds + full-board reference

```bash
# in-box, seeds that were never used for selection
for SEED in 1313 1414; do
  SO101_SPAWN_MODE=legacy .venv/bin/python evaluations/eval_embodied_agent.py \
    ... rollout.model.model_path=$BEST ... env.eval.seed=$SEED
done

# reference: the same checkpoint on the FULL board (no SO101_SPAWN_MODE)
.venv/bin/python evaluations/eval_embodied_agent.py \
    ... rollout.model.model_path=$BEST ... env.eval.seed=1313
```

| rule | why |
|---|---|
| verification seeds disjoint from gate seeds | selecting on A,B and reporting the A,B score inflated a 63% policy to 75% |
| the reported number is the **verification** number | gate numbers carry selection bias |
| always also report the full-board number | so a narrow-task score is never mistaken for the real task's score |

---

## Step 10 — checkpoint hygiene

```bash
for C in $RES/so101_sft_v8/so101_sft_openpi_pi05/checkpoints/global_step_*; do
  [ "$C" = "$BEST" ] || rm -rf "$C"
done
```
16 checkpoints ≈ 446 GB. Delete non-best immediately after the gate; a full disk once
crashed a run mid-checkpoint-write.

---

## Step 11 — invariant audit (CPU, seconds; run before and after)

```bash
.venv/bin/python -m toolkits.invariant_audit --ckpt $BEST
```
Checks that do not depend on anything crashing: temporal chain (real fps = sim
`control_freq` = converter `FPS`), camera chain (sim aspect and resolution vs the policy
pipeline's 224×168), spawn coverage vs the 87 real cube positions, success semantics
(homing present), norm-stats lineage (checkpoint copy = assets file), dataset/env
resolution match, budget headroom, action chunking, eval-seed disjointness.

---

## Automation

The whole sequence runs unattended as two scripts (both in `SO101_SESSION_LOG.md` Part 3):

```bash
setsid bash scratchpad/gen_v8_legacy.sh    </dev/null >/dev/null 2>&1 &   # Step 1
setsid bash scratchpad/v8_pipeline.sh      </dev/null >/dev/null 2>&1 &   # Steps 2-10
```

Each stage carries its own `timeout` plus one retry, a numeric gate (demo count, demo
length, episode count, preflight), and writes to `scratchpad/v8.status`. Autonomy lives
in the scripts, not in an interactive session: a disconnected session freezes the agent's
reactions, while `setsid` processes keep running.

---

## Results so far (2026-08-12)

| stage | measured |
|---|---|
| planner success in the box | **247/256 = 96.5%** (full board: 58%) |
| demos / median length | 247 / 357 steps |
| **demo spacing** | **0.44 cm** (pp-era 0.51, grasp tolerance ±0.7) |
| SFT | 4000 steps, exit 0, final loss ≈0.0016 |
| gate screen (seed 777), step_250 / 500 / 750 | 7.8% / 0.8% / 3.9% |
| gate screen (seed 777), **step_1000** | **54.7%** |
| gate screen (seed 777), step_1250 / 1500 / 1750 / 2000 | 36.7% / 7.8% / 43.0% / 16.4% |
| gate screen (seed 777), **step_2500** | **61.7%** |
| gate best: **global_step_2500** (777/888) | **61.7% / 56.3% → 59.0%** |
| **HONEST verify, never-used seeds 1313 / 1414** | **57.8% / 55.5%** |
| full-board reference (same ckpt) | 9.4% |

Pre-registered verdict: ≥40% in-box confirms that demonstration density sets the BC floor
(pp-era analogue: 46.9%). **`global_step_1000` reaches 54.7%, above the pp-era floor** —
the hypothesis holds, and the earlier checkpoints were simply under-trained (a reminder
that judging a run from its first two checkpoints is premature: at 15:58 this run was
flagged direction-suspect on the strength of step_250 and step_500 alone).

Numbers above are gate-seed values and carry selection bias; the honest figure is the
verification on never-used seeds 1313/1414, plus the full-board reference (Step 9).

---

# Part 2 — V9: expert iteration (first amplifier)

## Why this design

v8 established the **floor** (56.7% honest in-box). Two amplifiers can turn a floor into
a result:

| amplifier | historical gain | risk |
|---|---|---|
| **expert iteration** (collect successes → gentle re-SFT) | 63.3% → 81.6% (**+18 pts**) | **zero**: worst case is no gain; the starting weights are untouched |
| PPO | 46.9% → 75% (pp era) | high: from a 12% floor it destroyed the policy in 10 iterations |

Expert iteration goes first because (1) it is zero-risk and cheap (~4 h, mostly CPU),
(2) the collection stage doubles as an **unbiased re-measurement** of the current policy
on never-used seeds (measured: 57.0–65.6%, mean 61.3%), and (3) it pushes demonstration
density another notch (0.44 cm → 0.26 cm), and density is the one variable this project
has proven decisive.

**Why mix planner demos in rather than pure self-distillation:** iRe-VLA prescribes
training on the original expert data *together with* the new successful trajectories.
Pure self-distillation narrows the policy — v5 lost 53 points that way. Mix = 247 planner
+ 477 policy = 724 episodes.

**Why lr drops to 1e-5:** this step sharpens existing behaviour rather than teaching new
behaviour; 2.5e-5 would wash out what the policy already knows. The pp era used 1e-5 here too.

## Step 1 — collect the policy's own successes (GPU, ~55 min)

```bash
export SO101_COLLECT_DIR=/data08/henryg/pai/data/v9_rollouts; mkdir -p $SO101_COLLECT_DIR
V8=$RES/so101_sft_v8/so101_sft_openpi_pi05/checkpoints/global_step_2500
for SEED in 2001 2002 2003 2004 2005 2006 2007 2008; do
  SO101_SPAWN_MODE=legacy .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
    --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v9 \
    rollout.model.model_path=$V8 \
    rollout.model.openpi.config_name=pi05_so101_v8 \
    rollout.model.openpi_data.norm_stats_path=$REPO/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
    env.eval.total_num_envs=128 env.eval.seed=$SEED
done
```

| parameter | value | why |
|---|---|---|
| `SO101_COLLECT_DIR` | a directory | switches on the recorder in `ManiskillEnv`: each episode is flushed to `.npz` on its FIRST success (partial-reset safe) |
| seeds | **2001–2008**, all new | disjoint from gate (777/888) and verification (1313/1414), so the 8 numbers are an unbiased estimate |
| envs | 128 | 8 × 128 = 1024 episodes |
| `SO101_SPAWN_MODE` | `legacy` | same domain the policy was trained in |

Measured: 59.4 / 57.0 / 62.5 / 60.9 / 57.0 / 63.3 / 65.6 / 64.8% → **477 successful
trajectories**. Gate: ≥300.

## Step 2 — mixed conversion (CPU, ~1.5 h)

```bash
.venv/bin/python scratchpad/convert_v9_demos.py
```

| source | count | unit handling |
|---|---|---|
| planner demos (v8 h5) | 247 | both `qpos` and `actions` through `rad_to_norm` |
| policy rollouts (npz) | 477 | **state is already normalised** (recorder takes it from `_wrap_obs`); **actions are radians** and need `rad_to_norm` |

That asymmetry is a trap: getting either side wrong silently corrupts the dataset.
Output: `so101-sim-demos-v9`, 724 episodes, spacing ≈ **0.26 cm**.

## Step 3 — gentle SFT (GPU, ~45 min)

```bash
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config --config-path <abs>/examples/sft/config/ \
    --config-name so101_sft_v9 runner.logger.log_path=$RES/so101_sft_v9
.venv/bin/python examples/sft/train_vla_sft.py --config-path <abs>/examples/sft/config/ \
    --config-name so101_sft_v9 runner.logger.log_path=$RES/so101_sft_v9
```

| parameter | value | why |
|---|---|---|
| `model_path` | **v8_step_2500** | continue from the current best; expert iteration is self-sharpening |
| `train_data_paths` | `so101-sim-demos-v9` | the mixed set |
| `norm_stats_path` | **still v4's** | lineage frozen: v4 → v8 → v9 is one lineage |
| `lr` / `min_lr` | **1e-5 / 1e-6** | sharpening, not teaching |
| `max_steps` | **2000** (v8 used 4000) | more data but a gentler pass; guards against overfitting |
| `save_interval` | 250 | the peak can appear anywhere |

## Step 4 — gate and honest verification (GPU, ~1 h)

Same command shape as Part 1 Steps 7–9 with `config_name=pi05_so101_v9`, except:

| rule | why |
|---|---|
| verification seeds are **2323 / 2424** | 1313/1414 were already spent verifying v8; every generation needs fresh seeds or selection bias accumulates |
| always report the full-board number too | v8's full board was 9.4%: a box-trained policy does not transfer outward |

Pre-registered verdict: ≥65% honest → expert iteration works, run another round or move
to (b) widening; 57–65% → diminishing, one round is enough, move to (b); ≤57% → the data
distribution is saturated, switch to the πRL-aligned PPO recipe.

## Ops hardening added this round

Every eval now runs under `timeout 1800` with up to **3 retries** and a full clean
(processes via /proc, Ray, `/dev/shm`) between attempts — a Ray rollout worker died
mid-eval (`SYSTEM_ERROR: Worker unexpectedly exits`) with the driver waiting forever;
neither GPU memory nor shm was exhausted. A long total pipeline timeout is not a substitute.

```bash
setsid bash scratchpad/v9_expert_iter.sh </dev/null >/dev/null 2>&1 &   # fully automated
```
