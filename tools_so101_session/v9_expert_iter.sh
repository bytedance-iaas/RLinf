#!/bin/bash
# v9 — EXPERT ITERATION round 1 on the v8 policy (user: "a first, then b").
# Zero-risk amplifier: collect the current policy's own successes, mix them with
# the original planner demos (iRe-VLA), and re-SFT gently.
#
#   S1 collect: 8 never-used seeds x 128 envs in the legacy box, recorder on
#   S2 convert: 247 planner demos + new policy rollouts -> so101-sim-demos-v9
#   S3 SFT: from v8_step_2500, lr 1e-5 (gentle), 2000 steps, save 250, v4 stats
#   S4 gate: all ckpts on 777 -> top-3 on 888 -> verify on NEVER-USED 2323/2424
#            + full-board reference
#   S5 hygiene
# Every eval: full clean + timeout 1800 + 3 tries (a Ray worker died mid-eval once).
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/v9.status
V8=/data08/henryg/pai/results/so101_sft_v8/so101_sft_openpi_pi05/checkpoints/global_step_2500
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
CKROOT=/data08/henryg/pai/results/so101_sft_v9/so101_sft_openpi_pi05/checkpoints
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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

clean(){
  .venv/bin/ray stop --force >/dev/null 2>&1 || true
  for p in $(pgrep -f 'ray::|raylet|gcs_server|eval_embodied_agent|train_vla_sft'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
  done
  rm -rf /tmp/ray/session_* 2>/dev/null
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
  SHM=$(df -m /dev/shm | tail -1 | awk '{print $4}')
  [ "${SHM:-0}" -lt 1024 ] && mount -o remount,size=16G /dev/shm 2>/dev/null
  sleep 8
}

run_eval(){  # ckpt seed tag cfg spawnmode collectdir
  local TRY EV
  for TRY in 1 2 3; do
    clean
    SO101_SPAWN_MODE="$5" SO101_COLLECT_DIR="${6:-}" timeout 1800 \
      .venv/bin/python evaluations/eval_embodied_agent.py \
        --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
        --config-name so101_eval_openpi_pi05 \
        runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v9 \
        rollout.model.model_path="$1" \
        rollout.model.openpi.config_name="$4" \
        rollout.model.openpi_data.norm_stats_path=$STATS \
        env.eval.total_num_envs=128 env.eval.seed=$2 \
        > "$SCRATCH/eval_v9_$3_r$TRY.out" 2>&1
    EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v9_$3_r$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
    log "eval $3 try$TRY failed, retrying"
  done
  echo ""
}

FREE=$(df --output=avail -BG /data08 | tail -1 | tr -dc '0-9')
[ "$FREE" -lt 200 ] && { log "ABORT: disk <200G"; exit 3; }
log "v9 expert iteration started (base=v8_step_2500, honest 56.7% in-box)"

# ---- S1: collect the policy's own successes ----
COLLECT=/data08/henryg/pai/data/v9_rollouts
rm -rf $COLLECT; mkdir -p $COLLECT
for SEED in 2001 2002 2003 2004 2005 2006 2007 2008; do
  E=$(run_eval "$V8" $SEED "collect_s$SEED" pi05_so101_v8 legacy "$COLLECT")
  N=$(ls $COLLECT/*.npz 2>/dev/null | wc -l)
  log "collect seed=$SEED success=${E:-FAIL} cumulative_episodes=$N"
done
N=$(ls $COLLECT/*.npz 2>/dev/null | wc -l)
[ "$N" -ge 300 ] || { log "S1 GATE FAIL: only $N rollouts collected"; exit 1; }
log "S1 DONE: $N successful policy rollouts"

# ---- S2: convert (planner + policy mixed) ----
timeout 10800 .venv/bin/python "$SCRATCH/convert_v9_demos.py" > "$SCRATCH/convert_v9.out" 2>&1
grep -q 'DONE:' "$SCRATCH/convert_v9.out" || { log "S2 FAIL: conversion"; tail -3 "$SCRATCH/convert_v9.out" >> "$STATUS"; exit 1; }
log "S2 $(grep '^DONE:' "$SCRATCH/convert_v9.out")"
NEP=$(grep -oE 'DONE: [0-9]+' "$SCRATCH/convert_v9.out" | grep -oE '[0-9]+')
SPACING=$(awk -v n="$NEP" 'BEGIN{printf "%.2f", sqrt(48/n)}')
log "S2 demo spacing now ${SPACING} cm (v8 was 0.44; tolerance ~0.7)"

# ---- S3: gentle SFT ----
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v9 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v9 > "$SCRATCH/preflight_v9.out" 2>&1
grep -q 'PREFLIGHT OK' "$SCRATCH/preflight_v9.out" || { log "S3 PREFLIGHT FAIL"; tail -4 "$SCRATCH/preflight_v9.out" >> "$STATUS"; exit 1; }
clean
log "S3 SFT launching (from v8_step_2500, lr 1e-5, 2000 steps)"
timeout 14400 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v9 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v9 \
  > "$SCRATCH/sft_v9.out" 2>&1
log "S3 SFT exit=$?"
CKS=$(ls -d $CKROOT/global_step_* 2>/dev/null | sort -V)
[ -n "$CKS" ] || { log "S3 GATE FAIL: no ckpts"; exit 1; }
for CK in $CKS; do mkdir -p "$CK/so101-sim-demos-v4"; cp $STATS "$CK/so101-sim-demos-v4/" 2>/dev/null || true; done

# ---- S4: gate ----
export EMBODIED_PATH=$PWD/examples/embodiment
SCREEN=$SCRATCH/v9_screen.txt; : > "$SCREEN"
for CK in $CKS; do
  E=$(run_eval "$CK" 777 "$(basename $CK)_s777" pi05_so101_v9 legacy "")
  log "screen(in-box) $(basename $CK) s777: ${E:-FAIL}"
  [ -n "$E" ] && echo "$E $CK" >> "$SCREEN"
done
BESTCK=""; BESTAVG=0
while read -r E1 CK; do
  E2=$(run_eval "$CK" 888 "$(basename $CK)_s888" pi05_so101_v9 legacy "")
  log "confirm(in-box) $(basename $CK) s888: ${E2:-FAIL}"
  [ -n "$E2" ] || continue
  AVG=$(awk -v a="$E1" -v b="$E2" 'BEGIN{printf "%.4f",(a+b)/2}')
  log "confirm avg $(basename $CK): $AVG"
  if awk -v a="$AVG" -v b="$BESTAVG" 'BEGIN{exit !(a>b)}'; then BESTAVG=$AVG; BESTCK=$CK; fi
done < <(sort -gr "$SCREEN" | head -3)
[ -n "$BESTCK" ] || { log "S4 GATE FAIL"; exit 1; }
echo "$BESTCK" > "$SCRATCH/v9_best.ck"
V1=$(run_eval "$BESTCK" 2323 vfy2323 pi05_so101_v9 legacy ""); log "VERIFY in-box seed2323: ${V1:-FAIL}"
V2=$(run_eval "$BESTCK" 2424 vfy2424 pi05_so101_v9 legacy ""); log "VERIFY in-box seed2424: ${V2:-FAIL}"
FB=$(run_eval "$BESTCK" 2323 fullboard pi05_so101_v9 "" "");   log "FULL-BOARD reference: ${FB:-FAIL}"
log "V9 FINAL: ckpt=$(basename "$BESTCK") gate=$BESTAVG verify=${V1:-?}/${V2:-?} fullboard=${FB:-?} | v8 honest was 0.567, pp-era arc ended at ~0.80"

# ---- S5: hygiene ----
for C in $CKS; do [ "$C" = "$BESTCK" ] || rm -rf "$C"; done
log "v9 done; disk free $(df -h /data08 | tail -1 | awk '{print $4}')"
