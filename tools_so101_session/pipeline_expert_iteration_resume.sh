#!/bin/bash
# STATUS: ACTIVE — 当前流程在用。 阶段 D 的续跑版（原流水线在 S2 超时后接管）
# v9_rest — takes over S3 (gentle SFT) + S4 (gate/verify) + S5 (hygiene) from
# pipeline_expert_iteration.sh, whose S2 `timeout 10800` would have killed the conversion
# ~30 min before it finishes (measured rate 3.3 ep/min, 724 episodes total).
# The timeout wrapper was killed so the converter itself survives; this script
# owns everything after the conversion.
#
# Safety: never runs while the ORIGINAL pipeline is still alive (no double SFT).
# If the converter dies without printing DONE:, restart it ONCE with a 6h budget.
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/v9.status
PARENT_PID=2022079
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
CKROOT=/data08/henryg/pai/results/so101_sft_v9/so101_sft_openpi_pi05/checkpoints
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
# a defunct (zombie) pid still has /proc/<pid> — the original pipeline exits into
# exactly that state under setsid, so test the state field, not the directory.
alive(){ [ -d "/proc/$1" ] || return 1; [ "$(awk '{print $3}' /proc/$1/stat 2>/dev/null)" != "Z" ]; }

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

run_eval(){  # ckpt seed tag cfg spawnmode
  local TRY EV
  for TRY in 1 2 3; do
    clean
    SO101_SPAWN_MODE="$5" timeout 1800 \
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

log "v9_rest supervisor armed (owns S3-S5; conversion timeout defused)"

# ---- wait for S2 ----
RESTARTED=0
DEADLINE=$(( $(date +%s) + 6*3600 ))
while :; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "S2 FAIL: 6h deadline"; exit 1; }
  if grep -qa '^DONE:' "$SCRATCH/convert_v9.out" 2>/dev/null; then break; fi
  if alive $PARENT_PID; then sleep 60; continue; fi          # original pipeline alive -> stay out of its way
  if pgrep -f convert_expert_iter.py >/dev/null; then sleep 60; continue; fi
  if [ "$RESTARTED" = "0" ]; then
    log "S2 converter died without DONE: -> restarting once with a 6h budget"
    RESTARTED=1
    timeout 21600 .venv/bin/python "$SCRATCH/convert_expert_iter.py" > "$SCRATCH/convert_v9.out" 2>&1 &
    sleep 60; continue
  fi
  log "S2 FAIL (restart also died) — aborting"; exit 1
done
# if the ORIGINAL pipeline survived to see DONE: itself, it runs S3 — do not duplicate
sleep 20
if alive $PARENT_PID; then log "v9_rest: original pipeline alive at DONE: — standing down"; exit 0; fi
log "S2 $(grep -a '^DONE:' "$SCRATCH/convert_v9.out" | tail -1)"
NEP=$(grep -oaE 'DONE: [0-9]+' "$SCRATCH/convert_v9.out" | grep -oE '[0-9]+' | tail -1)
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
  E=$(run_eval "$CK" 777 "$(basename $CK)_s777" pi05_so101_v9 legacy)
  log "screen(in-box) $(basename $CK) s777: ${E:-FAIL}"
  [ -n "$E" ] && echo "$E $CK" >> "$SCREEN"
done
BESTCK=""; BESTAVG=0
while read -r E1 CK; do
  E2=$(run_eval "$CK" 888 "$(basename $CK)_s888" pi05_so101_v9 legacy)
  log "confirm(in-box) $(basename $CK) s888: ${E2:-FAIL}"
  [ -n "$E2" ] || continue
  AVG=$(awk -v a="$E1" -v b="$E2" 'BEGIN{printf "%.4f",(a+b)/2}')
  log "confirm avg $(basename $CK): $AVG"
  if awk -v a="$AVG" -v b="$BESTAVG" 'BEGIN{exit !(a>b)}'; then BESTAVG=$AVG; BESTCK=$CK; fi
done < <(sort -gr "$SCREEN" | head -3)
[ -n "$BESTCK" ] || { log "S4 GATE FAIL"; exit 1; }
echo "$BESTCK" > "$SCRATCH/v9_best.ck"
V1=$(run_eval "$BESTCK" 2323 vfy2323 pi05_so101_v9 legacy); log "VERIFY in-box seed2323: ${V1:-FAIL}"
V2=$(run_eval "$BESTCK" 2424 vfy2424 pi05_so101_v9 legacy); log "VERIFY in-box seed2424: ${V2:-FAIL}"
FB=$(run_eval "$BESTCK" 2323 fullboard pi05_so101_v9 "");   log "FULL-BOARD reference: ${FB:-FAIL}"
log "V9 FINAL: ckpt=$(basename "$BESTCK") gate=$BESTAVG verify=${V1:-?}/${V2:-?} fullboard=${FB:-?} | v8 honest was 0.567, pp-era arc ended at ~0.80"

# ---- S5: hygiene ----
for C in $CKS; do [ "$C" = "$BESTCK" ] || rm -rf "$C"; done
log "v9 done; disk free $(df -h /data08 | tail -1 | awk '{print $4}')"
