#!/bin/bash
# Overnight round 2: S1b more stratified demos (25/cell, extra 25 on weak cells)
# -> S2b rebuild dataset (all v3_demos_cell* incl. round 1) + norm_stats
# -> S3b SFT round 2 (from v3 step_3000) -> S4b gate + bands.
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
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

log "overnight v3b started (pid $$)"
worker(){
  local W=$1; shift
  for SPEC in "$@"; do
    local FRAC=${SPEC%%;*}; local REST=${SPEC#*;}; local SEED=${REST%%;*}; local NUM=${REST##*;}
    local TAG=B$(echo "$FRAC" | tr ',.' '_-')
    rm -rf /data08/henryg/pai/data/v3_demos_cell$TAG
    SO101_SPAWN_FRAC=$FRAC timeout 7200 .venv/bin/python "$SCRATCH/gen_planner_demos.py" \
      --num $NUM --seed0 $SEED --out /data08/henryg/pai/data/v3_demos_cell$TAG \
      > "$SCRATCH/v3b_gen_${TAG}.out" 2>&1
    local N=$(grep -oE 'TOTAL success [0-9]+' "$SCRATCH/v3b_gen_${TAG}.out" | grep -oE '[0-9]+$' || echo 0)
    log "cellB $FRAC: $N/$NUM (worker $W)"
  done
}
# 16 cells x 25; weak near-base column (x 0-0.25) gets 35
CELLS=()
SEED=70000
for XI in 0 1 2 3; do
  for YI in 0 1 2 3; do
    X0=$(awk -v i=$XI 'BEGIN{printf "%.2f", i*0.25}'); X1=$(awk -v i=$XI 'BEGIN{printf "%.2f", (i+1)*0.25}')
    Y0=$(awk -v i=$YI 'BEGIN{printf "%.2f", i*0.25}'); Y1=$(awk -v i=$YI 'BEGIN{printf "%.2f", (i+1)*0.25}')
    N=25; [ "$XI" = "0" ] && N=35
    CELLS+=("$X0,$X1,$Y0,$Y1;$SEED;$N"); SEED=$((SEED+100))
  done
done
worker 1 "${CELLS[@]:0:4}"  & P1=$!
worker 2 "${CELLS[@]:4:4}"  & P2=$!
worker 3 "${CELLS[@]:8:4}"  & P3=$!
worker 4 "${CELLS[@]:12:4}" & P4=$!
wait $P1 $P2 $P3 $P4
TOTAL=$(grep -hoE 'TOTAL success [0-9]+' "$SCRATCH"/v3b_gen_*.out | grep -oE '[0-9]+' | awk '{s+=$1} END{print s}')
log "S1b DONE: new successes $TOTAL"

# S2b: rebuild dataset from ALL cells (round 1 + B)
timeout 7200 .venv/bin/python "$SCRATCH/convert_v3_demos.py" > "$SCRATCH/convert_v3b.out" 2>&1
grep -q 'DONE:' "$SCRATCH/convert_v3b.out" || { log "S2b GATE FAIL: conversion"; tail -3 "$SCRATCH/convert_v3b.out" >> "$STATUS"; exit 1; }
log "S2b convert: $(grep -E '^DONE|^length' "$SCRATCH/convert_v3b.out" | tr '\n' ' ')"
timeout 3600 .venv/bin/python -m toolkits.lerobot.calculate_norm_stats \
  --config-name pi05_so101_v3 --repo-id so101-sim-demos-v3 > "$SCRATCH/norm_v3b.out" 2>&1
[ -f assets/pi05_so101_v3/so101-sim-demos-v3/norm_stats.json ] || { log "S2b GATE FAIL: norm stats"; exit 1; }
log "S2b norm stats OK"

# S3b: SFT round 2
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v3b \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v3b > "$SCRATCH/preflight_sft_v3b.out" 2>&1
grep -q 'PREFLIGHT OK' "$SCRATCH/preflight_sft_v3b.out" || { log "S3b PREFLIGHT FAIL"; exit 1; }
.venv/bin/ray stop --force >/dev/null 2>&1 || true
for p in $(pgrep -f 'ray::|raylet|gcs_server'); do
  [ "$p" = "$$" ] && continue
  exe=$(readlink /proc/$p/exe 2>/dev/null)
  case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
done
rm -rf /tmp/ray/session_* 2>/dev/null
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
log "S3b SFT launching"
timeout 14400 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v3b \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v3b \
  > "$SCRATCH/sft_v3b.out" 2>&1
log "S3b SFT exit=$?"
CKS=$(ls -d /data08/henryg/pai/results/so101_sft_v3b/*/checkpoints/global_step_* 2>/dev/null | sort -V)
[ -n "$CKS" ] || { log "S3b GATE FAIL: no ckpts"; exit 1; }
for CK in $CKS; do
  mkdir -p "$CK/so101-sim-demos-v3"
  cp assets/pi05_so101_v3/so101-sim-demos-v3/norm_stats.json "$CK/so101-sim-demos-v3/" 2>/dev/null || true
done

# S4b gate
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
      > "$SCRATCH/eval_v3b_$3_t$TRY.out" 2>&1
    local EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v3b_$3_t$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
    log "eval $3 try$TRY empty, retry"
  done
  return 0
}
BESTCK=""; BESTAVG=0
for CK in $(ls -d /data08/henryg/pai/results/so101_sft_v3b/*/checkpoints/global_step_{1000,2000,3000,4000} 2>/dev/null); do
  E1=$(run_eval "$CK" 777 $(basename $CK)_s777); log "gateB $(basename $CK) s777: ${E1:-FAIL}"
  E2=$(run_eval "$CK" 888 $(basename $CK)_s888); log "gateB $(basename $CK) s888: ${E2:-FAIL}"
  [ -n "${E1:-}" ] && [ -n "${E2:-}" ] || continue
  AVG=$(awk -v a="$E1" -v b="$E2" 'BEGIN{printf "%.4f",(a+b)/2}')
  log "gateB avg $(basename $CK): $AVG"
  if awk -v a="$AVG" -v b="$BESTAVG" 'BEGIN{exit !(a>b)}'; then BESTAVG=$AVG; BESTCK=$CK; fi
done
[ -n "$BESTCK" ] || { log "S4b GATE FAIL"; exit 1; }
echo "$BESTCK" > "$SCRATCH/v3b_best.ck"
V=$(run_eval "$BESTCK" 909 vB_s909); log "VERIFY B s909: ${V:-FAIL}"
B0=$(run_eval "$BESTCK" 606 bandB_right "0,1,0,0.33");   log "B y-band right: ${B0:-FAIL}"
B1=$(run_eval "$BESTCK" 606 bandB_mid   "0,1,0.33,0.66"); log "B y-band middle: ${B1:-FAIL}"
B2=$(run_eval "$BESTCK" 606 bandB_left  "0,1,0.66,1");   log "B y-band left: ${B2:-FAIL}"
log "V3B PIPELINE DONE: best=$BESTCK gate=$BESTAVG verify=${V:-?} bands=[${B0:-?} ${B1:-?} ${B2:-?}]"
