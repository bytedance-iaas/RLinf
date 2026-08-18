#!/bin/bash
# STATUS: SUPERSEDED — 早期任务规格，已被主线取代。别用来复现。 pp 时代的整段流水线
# NOTE: 仍引用 $SCRATCH/convert_pp_demos.py —— 那个转换器属于早期任务规格，没有随本仓库保留。
#       本脚本已归类 SUPERSEDED，跑不起来是预期的；主线的对应步骤见 SO101_PIPELINE_ZH.md。
# Phase-2 full pipeline: run when GPUs are free. Each stage gates on the
# previous one's verdict; status lines go to STATUS. Fully detached-safe.
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/phase2.status
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

export REPO_PATH="$PWD" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99

log "phase2 pipeline started (pid $$)"

# ---- stage 0: env runtime validation (CPU sim + tiny vulkan ctx) ----
CUDA_VISIBLE_DEVICES=7 timeout 600 .venv/bin/python - <<'PY' >> "$STATUS" 2>&1
import torch, gymnasium as gym, sys; sys.path.insert(0,'.')
from rlinf.envs.maniskill import import_all_tasks; import_all_tasks()
from mani_skill.utils.structs.pose import Pose
env=gym.make("SO101GrabRedCube-v1", num_envs=4, obs_mode="rgb", control_mode="pd_joint_pos", sim_backend="gpu", reward_mode="normalized_dense")
env.reset(seed=0); base=env.unwrapped
for _ in range(5): _,r,_,_,_=env.step(env.action_space.sample())
assert bool(torch.isfinite(torch.as_tensor(r)).all()), "reward not finite"
env.reset(options=dict(env_idx=torch.tensor([1,3])))
ev=base.evaluate(); assert all(tuple(v.shape)==(4,) for v in ev.values()), "partial reset shapes"
bp=base.box.pose.p.clone(); bp[:,2]+=0.015
base.red_cube.set_pose(Pose.create_from_pq(bp))
ev=base.evaluate()
print("STAGE0 OK: in_box:", ev["is_in_box"].tolist(), "keys:", sorted(ev.keys()))
PY
grep -q "STAGE0 OK" "$STATUS" || { log "STAGE0 FAILED - aborting"; exit 1; }
log "stage0 env validation OK"

# ---- stage 1: generate pp demos (probe 2, then 180) ----
export CUDA_VISIBLE_DEVICES=7
.venv/bin/python tools_so101_session/gen_planner_demos.py --num 2 --seed0 5000 --out /data08/henryg/pai/data/so101_pp_demos_probe >> "$STATUS" 2>&1
grep -q "TOTAL success [12]/2" "$STATUS" || { log "STAGE1 PROBE FAILED - aborting"; exit 1; }
log "stage1 probe OK; generating 180 demos"
.venv/bin/python tools_so101_session/gen_planner_demos.py --num 180 --seed0 6000 --out /data08/henryg/pai/data/so101_pp_demos > "$SCRATCH/gen_pp.out" 2>&1
NOK=$(grep -oE 'TOTAL success [0-9]+' "$SCRATCH/gen_pp.out" | grep -oE '[0-9]+$' || echo 0)
log "stage1 done: $NOK/180 successful demos"
[ "${NOK:-0}" -ge 80 ] || { log "STAGE1 too few successes - aborting"; exit 1; }

# ---- stage 2: convert to LeRobot ----
H5=$(ls -t /data08/henryg/pai/data/so101_pp_demos/*.h5 | head -1)
sed -i "s|^H5 = .*|H5 = \"$H5\"|; s|^META = .*|META = \"${H5%.h5}.json\"|" "$SCRATCH/convert_pp_demos.py"
.venv/bin/python "$SCRATCH/convert_pp_demos.py" >> "$STATUS" 2>&1
grep -q "DONE:" "$STATUS" || { log "STAGE2 CONVERT FAILED - aborting"; exit 1; }
log "stage2 conversion done"

# ---- stage 3: norm stats ----
.venv/bin/python -m toolkits.lerobot.calculate_norm_stats --config-name pi05_so101_pp --repo-id so101-sim-demos-pp >> "$SCRATCH/norm_pp.out" 2>&1
[ -f assets/pi05_so101_pp/so101-sim-demos-pp/norm_stats.json ] || { log "STAGE3 NORM STATS FAILED"; exit 1; }
log "stage3 norm stats OK"

# ---- stage 4: SFT (4000 steps) ----
export EMBODIED_PATH=$PWD/examples/sft CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
setsid .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_pp \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_pp \
  > "$SCRATCH/sft_pp.out" 2>&1 </dev/null
log "stage4 SFT exited: $(tail -1 "$SCRATCH/sft_pp.out" | cut -c1-60)"
CK=/data08/henryg/pai/results/so101_sft_pp/so101_sft_openpi_pi05/checkpoints/global_step_4000
[ -d "$CK/actor" ] || CK=$(ls -d /data08/henryg/pai/results/so101_sft_pp/*/checkpoints/global_step_* 2>/dev/null | sort -V | tail -1)
[ -n "$CK" ] && [ -d "$CK/actor" ] || { log "STAGE4 no SFT ckpt - aborting"; exit 1; }
mkdir -p "$CK/so101-sim-demos-pp"; cp assets/pi05_so101_pp/so101-sim-demos-pp/norm_stats.json "$CK/so101-sim-demos-pp/"
log "stage4 SFT ckpt: $CK"

# ---- stage 5: zero-shot eval gate ----
.venv/bin/ray stop --force >/dev/null 2>&1 || true; sleep 5; rm -rf /tmp/ray/session_* 2>/dev/null
export EMBODIED_PATH=$PWD/examples/embodiment
.venv/bin/python evaluations/eval_embodied_agent.py \
  --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
  --config-name so101_eval_openpi_pi05 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_eval_pp \
  rollout.model.model_path="$CK" \
  rollout.model.openpi.config_name=pi05_so101_pp \
  rollout.model.openpi_data.norm_stats_path=/data08/henryg/pai/RLinf/assets/pi05_so101_pp/so101-sim-demos-pp/norm_stats.json \
  env.eval.total_num_envs=128 \
  > "$SCRATCH/eval_pp.out" 2>&1
EV=$(grep -oE "success_once=[0-9.]+" "$SCRATCH/eval_pp.out" | tail -1)
log "stage5 zero-shot eval: ${EV:-unknown}"

# ---- stage 6: conservative RL ----
.venv/bin/ray stop --force >/dev/null 2>&1 || true; sleep 5; rm -rf /tmp/ray/session_* 2>/dev/null
setsid .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
  --config-name so101_ppo_openpi_pi05 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_ppo_pp1 \
  actor.model.model_path="$CK" \
  rollout.model.model_path="$CK" \
  actor.model.openpi.config_name=pi05_so101_pp \
  actor.model.openpi_data.norm_stats_path=/data08/henryg/pai/RLinf/assets/pi05_so101_pp/so101-sim-demos-pp/norm_stats.json \
  env.train.ignore_terminations=True \
  "actor.model.openpi.noise_params=[0.08,0.05,200]" \
  actor.optim.lr=2e-6 \
  algorithm.update_epoch=2 \
  algorithm.clip_ratio_high=0.1 \
  algorithm.clip_ratio_low=0.1 \
  algorithm.entropy_bonus=0.001 \
  > "$SCRATCH/rl_pp1.out" 2>&1 </dev/null &
log "stage6 RL launched (wrapper $!)"
# heartbeat
while true; do
  sleep 1800
  EP=$(grep -c 'success_once=' "$SCRATCH/rl_pp1.out" 2>/dev/null || echo 0)
  LAST=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/rl_pp1.out" | tail -1)
  BEST=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/rl_pp1.out" | sed 's/success_once=//' | sort -g | tail -1)
  grep -qE 'EXIT=' "$SCRATCH/rl_pp1.out" 2>/dev/null && { log "RL ended: $(grep -oE 'EXIT=[0-9]+' "$SCRATCH/rl_pp1.out" | tail -1) at epoch $EP"; break; }
  log "heartbeat: epoch $EP last $LAST best $BEST"
done
