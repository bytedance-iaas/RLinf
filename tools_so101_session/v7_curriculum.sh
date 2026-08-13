#!/bin/bash
# v7 CURRICULUM (user-directed 2026-08-12: "先只训中+左带"):
#   restrict the task to the brown zone's middle+left bands (spawn y >= 0.25,
#   12 of 16 cells, 384 demos, planner 65-100% there) and see whether the BC
#   floor recovers toward the pp-era level.
#
#   S1 wait for the curriculum dataset build (CPU, already running)
#   S2 SFT from v4_step_1000, lr 2.5e-5, 4000 steps, save every 250
#      (norm_stats REUSED from v4 — same lineage, subset of the same data)
#   S3 gate every ckpt on seed 777 IN-BAND, top-3 confirmed on 888 in-band
#   S4 honest verification: never-used seeds 1313/1414 in-band + one FULL-BOARD
#      eval of the same ckpt, so the curriculum number is never mistaken for the
#      full-task number
#   S5 hygiene: keep best + neighbours
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/v7.status
BAND="0,1,0.25,1"     # SO101_SPAWN_FRAC: full x, y in [0.25, 1.0]
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
  for p in $(pgrep -f 'ray::|raylet|gcs_server|eval_embodied_agent|train_embodied_agent'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
  done
  rm -rf /tmp/ray/session_* 2>/dev/null
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
  SHM_MB=$(df -m /dev/shm | tail -1 | awk '{print $4}')
  [ "${SHM_MB:-0}" -lt 1024 ] && mount -o remount,size=16G /dev/shm 2>/dev/null
}

log "v7 curriculum started (band y>=0.25) pid $$"

# ---- S1: wait for the dataset ----
D=0
until grep -q 'DONE:' "$SCRATCH/convert_v7.out" 2>/dev/null; do
  sleep 60; D=$((D+60))
  grep -qiE 'Traceback' "$SCRATCH/convert_v7.out" 2>/dev/null && { log "S1 FAIL: conversion crashed"; exit 1; }
  [ $D -ge 7200 ] && { log "S1 TIMEOUT"; exit 1; }
done
log "S1 $(grep -E '^DONE|^length|^curriculum cells' "$SCRATCH/convert_v7.out" | tr '\n' ' ')"
NEP=$(grep -oE 'DONE: [0-9]+' "$SCRATCH/convert_v7.out" | grep -oE '[0-9]+')
[ "${NEP:-0}" -ge 300 ] || { log "S1 GATE FAIL: only $NEP episodes"; exit 1; }

# ---- S2: SFT ----
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v7 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v7 > "$SCRATCH/preflight_v7.out" 2>&1
grep -q 'PREFLIGHT OK' "$SCRATCH/preflight_v7.out" || { log "S2 PREFLIGHT FAIL"; tail -4 "$SCRATCH/preflight_v7.out" >> "$STATUS"; exit 1; }
clean; sleep 5
log "S2 SFT launching (from v4_step_1000, v4 stats)"
timeout 21600 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v7 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v7 \
  > "$SCRATCH/sft_v7.out" 2>&1
log "S2 SFT exit=$?"
CKS=$(ls -d /data08/henryg/pai/results/so101_sft_v7/*/checkpoints/global_step_* 2>/dev/null | sort -V)
[ -n "$CKS" ] || { log "S2 GATE FAIL: no ckpts"; exit 1; }
for CK in $CKS; do
  mkdir -p "$CK/so101-sim-demos-v4"
  cp assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json "$CK/so101-sim-demos-v4/" 2>/dev/null || true
done

# ---- S3/S4: gate in-band, verify in-band on unused seeds, plus full-board ----
export EMBODIED_PATH=$PWD/examples/embodiment
run_eval(){  # ckpt seed tag spawnfrac
  local TRY
  for TRY in 1 2; do
    clean
    SO101_SPAWN_FRAC="$4" timeout 2400 .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
      --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v7 \
      rollout.model.model_path="$1" \
      rollout.model.openpi.config_name=pi05_so101_v7 \
      rollout.model.openpi_data.norm_stats_path=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
      env.eval.total_num_envs=128 env.eval.seed=$2 \
      > "$SCRATCH/eval_v7_$3_t$TRY.out" 2>&1
    local EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v7_$3_t$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
  done
  return 0
}
SCREEN=$SCRATCH/v7_screen.txt; : > "$SCREEN"
for CK in $CKS; do
  E=$(run_eval "$CK" 777 "$(basename $CK)_s777" "$BAND")
  log "screen(in-band) $(basename $CK) s777: ${E:-FAIL}"
  [ -n "${E:-}" ] && echo "$E $CK" >> "$SCREEN"
done
BESTCK=""; BESTAVG=0
while read -r E1 CK; do
  E2=$(run_eval "$CK" 888 "$(basename $CK)_s888" "$BAND")
  log "confirm(in-band) $(basename $CK) s888: ${E2:-FAIL}"
  [ -n "${E2:-}" ] || continue
  AVG=$(awk -v a="$E1" -v b="$E2" 'BEGIN{printf "%.4f",(a+b)/2}')
  log "confirm avg $(basename $CK): $AVG"
  if awk -v a="$AVG" -v b="$BESTAVG" 'BEGIN{exit !(a>b)}'; then BESTAVG=$AVG; BESTCK=$CK; fi
done < <(sort -gr "$SCREEN" | head -3)
[ -n "$BESTCK" ] || { log "S3 GATE FAIL"; exit 1; }
echo "$BESTCK" > "$SCRATCH/v7_best.ck"
V1=$(run_eval "$BESTCK" 1313 verify_1313 "$BAND"); log "VERIFY in-band s1313: ${V1:-FAIL}"
V2=$(run_eval "$BESTCK" 1414 verify_1414 "$BAND"); log "VERIFY in-band s1414: ${V2:-FAIL}"
FB=$(run_eval "$BESTCK" 1313 fullboard_1313 "");   log "FULL-BOARD s1313 (reference): ${FB:-FAIL}"
log "V7 RESULT: ckpt=$(basename "$BESTCK") gate=$BESTAVG verify_in_band=${V1:-?}/${V2:-?} full_board=${FB:-?} | baselines: v4 full-board 0.125, pp-era subset ~0.80"

# ---- S5 hygiene ----
for C in $CKS; do [ "$C" = "$BESTCK" ] || rm -rf "$C"; done
log "v7 done; disk free $(df -h /data08 | tail -1 | awk '{print $4}')"
