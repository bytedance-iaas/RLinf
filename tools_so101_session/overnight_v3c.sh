#!/bin/bash
# Overnight round 2: S1b more stratified demos (25/cell, extra 25 on weak cells)
# -> S2b rebuild dataset (all v3_demos_cell* incl. round 1) + norm_stats
# -> S3c SFT round 2 (from v3 step_3000) -> S4c gate + bands.
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/v3.status
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
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo

log "v3c (frozen-stats retrain) started"
# S3c: SFT round 2
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v3c \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v3c > "$SCRATCH/preflight_sft_v3c.out" 2>&1
grep -q 'PREFLIGHT OK' "$SCRATCH/preflight_sft_v3c.out" || { log "S3c PREFLIGHT FAIL"; exit 1; }
.venv/bin/ray stop --force >/dev/null 2>&1 || true
for p in $(pgrep -f 'ray::|raylet|gcs_server'); do
  [ "$p" = "$$" ] && continue
  exe=$(readlink /proc/$p/exe 2>/dev/null)
  case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
done
rm -rf /tmp/ray/session_* 2>/dev/null
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
log "S3c SFT launching"
timeout 14400 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v3c \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v3c \
  > "$SCRATCH/sft_v3c.out" 2>&1
log "S3c SFT exit=$?"
CKS=$(ls -d /data08/henryg/pai/results/so101_sft_v3c/*/checkpoints/global_step_* 2>/dev/null | sort -V)
[ -n "$CKS" ] || { log "S3c GATE FAIL: no ckpts"; exit 1; }
for CK in $CKS; do
  mkdir -p "$CK/so101-sim-demos-v3"
  cp assets/pi05_so101_v3/so101-sim-demos-v3/norm_stats.json "$CK/so101-sim-demos-v3/" 2>/dev/null || true
done

# S4c gate
export EMBODIED_PATH=$PWD/examples/embodiment
clean(){
  .venv/bin/ray stop --force >/dev/null 2>&1 || true
  for p in $(pgrep -f 'ray::|raylet|gcs_server|eval_embodied_agent'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
  done
  rm -rf /tmp/ray/session_* 2>/dev/null
}
run_eval(){
  local SF="${4:-}"
  local TRY
  for TRY in 1 2; do
    clean
    SO101_SPAWN_FRAC="$SF" timeout 900 .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
      --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v3 \
      rollout.model.model_path="$1" \
      rollout.model.openpi.config_name=pi05_so101_v3 \
      rollout.model.openpi_data.norm_stats_path=/data08/henryg/pai/RLinf/assets/pi05_so101_v3/so101-sim-demos-v3/norm_stats.json \
      env.eval.total_num_envs=128 \
      env.eval.seed=$2 \
      > "$SCRATCH/eval_v3c_$3_t$TRY.out" 2>&1
    local EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v3c_$3_t$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
    log "eval $3 try$TRY empty, retry"
  done
  return 0
}
BESTCK=""; BESTAVG=0
for CK in $(ls -d /data08/henryg/pai/results/so101_sft_v3c/*/checkpoints/global_step_{1000,2000,3000,4000} 2>/dev/null); do
  E1=$(run_eval "$CK" 777 $(basename $CK)_s777); log "gateC $(basename $CK) s777: ${E1:-FAIL}"
  E2=$(run_eval "$CK" 888 $(basename $CK)_s888); log "gateC $(basename $CK) s888: ${E2:-FAIL}"
  [ -n "${E1:-}" ] && [ -n "${E2:-}" ] || continue
  AVG=$(awk -v a="$E1" -v b="$E2" 'BEGIN{printf "%.4f",(a+b)/2}')
  log "gateC avg $(basename $CK): $AVG"
  if awk -v a="$AVG" -v b="$BESTAVG" 'BEGIN{exit !(a>b)}'; then BESTAVG=$AVG; BESTCK=$CK; fi
done
[ -n "$BESTCK" ] || { log "S4c GATE FAIL"; exit 1; }
echo "$BESTCK" > "$SCRATCH/v3c_best.ck"
V=$(run_eval "$BESTCK" 909 vC_s909); log "VERIFY B s909: ${V:-FAIL}"
B0=$(run_eval "$BESTCK" 606 bandC_right "0,1,0,0.33");   log "B y-band right: ${B0:-FAIL}"
B1=$(run_eval "$BESTCK" 606 bandC_mid   "0,1,0.33,0.66"); log "B y-band middle: ${B1:-FAIL}"
B2=$(run_eval "$BESTCK" 606 bandC_left  "0,1,0.66,1");   log "B y-band left: ${B2:-FAIL}"
log "V3C PIPELINE DONE: best=$BESTCK gate=$BESTAVG verify=${V:-?} bands=[${B0:-?} ${B1:-?} ${B2:-?}]"
