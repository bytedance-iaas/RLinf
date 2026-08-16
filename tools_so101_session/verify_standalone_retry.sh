#!/bin/bash
# v8 final verification, re-run standalone after a Ray worker death stalled the
# pipeline's verification stage. Best checkpoint = global_step_2500
# (gate: seed777 61.7% / seed888 56.3% -> 59.0%).
#   - two NEVER-USED seeds in the legacy box (the honest number)
#   - one full-board reference with the same checkpoint
# Each eval: full clean (procs + ray + /dev/shm), timeout 1800, up to 3 tries.
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/v8.status
BEST=/data08/henryg/pai/results/so101_sft_v8/so101_sft_openpi_pi05/checkpoints/global_step_2500
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

export REPO_PATH="$PWD" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1
export EMBODIED_PATH=$PWD/examples/embodiment
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
  for p in $(pgrep -f 'ray::|raylet|gcs_server|eval_embodied_agent'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
  done
  rm -rf /tmp/ray/session_* 2>/dev/null
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
  sleep 8
}

run(){  # seed tag spawnmode
  local TRY EV
  for TRY in 1 2 3; do
    clean
    SO101_SPAWN_MODE="$3" timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
      --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v8 \
      rollout.model.model_path="$BEST" \
      rollout.model.openpi.config_name=pi05_so101_v8 \
      rollout.model.openpi_data.norm_stats_path=$STATS \
      env.eval.total_num_envs=128 env.eval.seed=$1 \
      > "$SCRATCH/eval_v8_$2_r$TRY.out" 2>&1
    EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v8_$2_r$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
    log "verify $2 try$TRY failed (worker death / timeout), retrying"
  done
  echo ""
}

log "v8 verification re-run started (best=global_step_2500)"
V1=$(run 1313 vfy1313 legacy); log "VERIFY in-box seed1313: ${V1:-FAIL}"
V2=$(run 1414 vfy1414 legacy); log "VERIFY in-box seed1414: ${V2:-FAIL}"
FB=$(run 1313 fullboard "");   log "FULL-BOARD reference seed1313: ${FB:-FAIL}"
log "V8 FINAL: ckpt=global_step_2500 gate=0.5898 verify=${V1:-?}/${V2:-?} fullboard=${FB:-?} | pp-era floor 0.469 (no homing), v4 full-board 0.125"

# checkpoint hygiene: keep the best plus its two neighbours for a possible RL start
CKROOT=/data08/henryg/pai/results/so101_sft_v8/so101_sft_openpi_pi05/checkpoints
for C in $CKROOT/global_step_*; do
  case "$(basename $C)" in
    global_step_2500|global_step_2250|global_step_2750|global_step_3500) ;;
    *) rm -rf "$C"; log "deleted $(basename $C)";;
  esac
done
log "verification done; disk free $(df -h /data08 | tail -1 | awk '{print $4}')"
