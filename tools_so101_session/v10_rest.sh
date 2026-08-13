#!/bin/bash
# v10 S3-S5 — convert (append to a copy of v9), gentle SFT, gate.
#
# Timeouts are sized from MEASURED rates, not round numbers:
#   convert ~3.3 episodes/min x ~600 new episodes = ~3 h  -> 6 h budget
#   SFT     v9's 2000 steps took 44 min                   -> 4 h budget
#   eval    each ~6 min                                   -> 30 min x 3 tries
# (v9's conversion stage was killed by a too-tight timeout; that is why.)
#
# The gate now runs in RING 1 — the region actually trained — with in-box and
# full-board evals kept as continuity references. Verification seeds 3131/3232
# have never been used anywhere.
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/v10.status
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
CKROOT=/data08/henryg/pai/results/so101_sft_v10/so101_sft_openpi_pi05/checkpoints
RING1="0.4294,0.9115,0.5142,0.9817"
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

run_eval(){  # ckpt seed tag frac [spawn_mode]
  # NOTE the 5th arg: legacy mode differs from the equivalent FRAC rectangle in
  # where the BLUE distractor goes, so the historical in-box numbers are only
  # comparable against SO101_SPAWN_MODE=legacy, not against the same rectangle
  # expressed in full-board fractions.
  local TRY EV
  for TRY in 1 2 3; do
    clean
    SO101_SPAWN_FRAC="$4" SO101_SPAWN_MODE="${5:-}" timeout 1800 \
      .venv/bin/python evaluations/eval_embodied_agent.py \
        --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
        --config-name so101_eval_openpi_pi05 \
        runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v10 \
        rollout.model.model_path="$1" \
        rollout.model.openpi.config_name=pi05_so101_v10 \
        rollout.model.openpi_data.norm_stats_path=$STATS \
        env.eval.total_num_envs=128 env.eval.seed=$2 \
        > "$SCRATCH/eval_v10_$3_r$TRY.out" 2>&1
    EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v10_$3_r$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
    log "eval $3 try$TRY failed, retrying"
  done
  echo ""
}

# ---- wait for the annulus generation (S2) ----
gen_done(){ grep -q 'S2 DONE:' "$STATUS"; }
DEADLINE=$(( $(date +%s) + 6*3600 ))
while ! gen_done; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "S3 ABORT: generation never finished"; exit 1; }
  if ! pgrep -f 'bash .*v10_gen.sh' >/dev/null; then
    sleep 5
    gen_done && break
    log "S3 ABORT: generator exited without its completion marker"; exit 1
  fi
  sleep 60
done
log "S3 conversion starting (append-to-v9-copy; no re-encode of the 672 base episodes)"

FREE=$(df --output=avail -BG /data08 | tail -1 | tr -dc '0-9')
[ "$FREE" -lt 200 ] && { log "S3 ABORT: disk <200G"; exit 3; }
timeout 21600 .venv/bin/python "$SCRATCH/convert_v10_demos.py" > "$SCRATCH/convert_v10.out" 2>&1
grep -qa '^DONE:' "$SCRATCH/convert_v10.out" || { log "S3 FAIL: conversion"; tail -3 "$SCRATCH/convert_v10.out" >> "$STATUS"; exit 1; }
log "S3 $(grep -a '^DONE:' "$SCRATCH/convert_v10.out" | tail -1)"
NEP=$(grep -oaE 'DONE: [0-9]+' "$SCRATCH/convert_v10.out" | grep -oE '[0-9]+' | tail -1)
log "S3 ring-1 spacing now $(awk -v n="$NEP" 'BEGIN{printf "%.2f", sqrt(96/n)}') cm over 96 cm^2 (v8 proved 0.44 works; tolerance ~0.7)"

# ---- S4: gentle SFT ----
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v10 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v10 > "$SCRATCH/preflight_v10.out" 2>&1
grep -q 'PREFLIGHT OK' "$SCRATCH/preflight_v10.out" || { log "S4 PREFLIGHT FAIL"; tail -4 "$SCRATCH/preflight_v10.out" >> "$STATUS"; exit 1; }
clean
log "S4 SFT launching (from v9_step_1250, lr 1e-5, 2000 steps)"
timeout 14400 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v10 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v10 \
  > "$SCRATCH/sft_v10.out" 2>&1
log "S4 SFT exit=$?"
CKS=$(ls -d $CKROOT/global_step_* 2>/dev/null | sort -V)
[ -n "$CKS" ] || { log "S4 GATE FAIL: no ckpts"; exit 1; }
for CK in $CKS; do mkdir -p "$CK/so101-sim-demos-v4"; cp $STATS "$CK/so101-sim-demos-v4/" 2>/dev/null || true; done

# ---- S5: gate in ring 1 ----
export EMBODIED_PATH=$PWD/examples/embodiment
SCREEN=$SCRATCH/v10_screen.txt; : > "$SCREEN"
for CK in $CKS; do
  E=$(run_eval "$CK" 777 "$(basename $CK)_s777" "$RING1")
  log "screen(ring1) $(basename $CK) s777: ${E:-FAIL}"
  [ -n "$E" ] && echo "$E $CK" >> "$SCREEN"
done
BESTCK=""; BESTAVG=0
while read -r E1 CK; do
  E2=$(run_eval "$CK" 888 "$(basename $CK)_s888" "$RING1")
  log "confirm(ring1) $(basename $CK) s888: ${E2:-FAIL}"
  [ -n "$E2" ] || continue
  AVG=$(awk -v a="$E1" -v b="$E2" 'BEGIN{printf "%.4f",(a+b)/2}')
  log "confirm avg $(basename $CK): $AVG"
  if awk -v a="$AVG" -v b="$BESTAVG" 'BEGIN{exit !(a>b)}'; then BESTAVG=$AVG; BESTCK=$CK; fi
done < <(sort -gr "$SCREEN" | head -3)
[ -n "$BESTCK" ] || { log "S5 GATE FAIL"; exit 1; }
echo "$BESTCK" > "$SCRATCH/v10_best.ck"
V1=$(run_eval "$BESTCK" 3131 vfy3131 "$RING1"); log "VERIFY ring1 seed3131: ${V1:-FAIL}"
V2=$(run_eval "$BESTCK" 3232 vfy3232 "$RING1"); log "VERIFY ring1 seed3232: ${V2:-FAIL}"
IB=$(run_eval "$BESTCK" 3131 inbox "0,1,0,1" legacy); log "IN-BOX reference (legacy mode, comparable to v9's 0.766): ${IB:-FAIL}"
FB=$(run_eval "$BESTCK" 3131 fullboard "0,1,0,1");                 log "FULL-BOARD reference: ${FB:-FAIL}"
log "V10 FINAL: ckpt=$(basename "$BESTCK") gate=$BESTAVG verify=${V1:-?}/${V2:-?} inbox=${IB:-?} fullboard=${FB:-?} | v9 was ring1 0.516 / inbox 0.766 / fullboard 0.195"

for C in $CKS; do [ "$C" = "$BESTCK" ] || rm -rf "$C"; done
log "v10 done; disk free $(df -h /data08 | tail -1 | awk '{print $4}')"
