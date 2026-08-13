#!/bin/bash
# v10 S1 — (b) RING-1 EXPANSION, collection stage.
# Ring 1 = the legacy 6x8cm box grown by sqrt(2) per axis about its own centre
# => 8.49 x 11.31 cm = 96 cm^2 (exactly 2x the v8/v9 spawn area), expressed as
# fractions of the FULL-BOARD spawn ranges so the same string works for demo
# generation, collection and eval:
#     x: board_center_x +/- 0.088 m  -> [0.4294, 0.9115]
#     y: cy             +/- 0.121 m  -> [0.5142, 0.9817]
# Both stay inside the true brown zone (checked: 0.9817 < 1.0).
#
# Stage 1 collects v9_step_1250's OWN successes over ring 1 on never-used seeds
# 3001-3008. Two purposes: (a) the first honest measurement of how far the v9
# policy's competence extends past its training box, (b) free in-distribution
# demos exactly where coverage is thin.
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/v10.status
V9=/data08/henryg/pai/results/so101_sft_v9/so101_sft_openpi_pi05/checkpoints/global_step_1250
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
RING1="0.4294,0.9115,0.5142,0.9817"
COLLECT=/data08/henryg/pai/data/v10_rollouts
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
  SHM=$(df -m /dev/shm | tail -1 | awk '{print $4}')
  [ "${SHM:-0}" -lt 1024 ] && mount -o remount,size=16G /dev/shm 2>/dev/null
  sleep 8
}

run_eval(){  # seed tag frac collectdir
  local TRY EV
  for TRY in 1 2 3; do
    clean
    SO101_SPAWN_FRAC="$3" SO101_COLLECT_DIR="${4:-}" timeout 1800 \
      .venv/bin/python evaluations/eval_embodied_agent.py \
        --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
        --config-name so101_eval_openpi_pi05 \
        runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v10 \
        rollout.model.model_path="$V9" \
        rollout.model.openpi.config_name=pi05_so101_v9 \
        rollout.model.openpi_data.norm_stats_path=$STATS \
        env.eval.total_num_envs=128 env.eval.seed=$1 \
        > "$SCRATCH/eval_v10_$2_r$TRY.out" 2>&1
    EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v10_$2_r$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
    log "eval $2 try$TRY failed, retrying"
  done
  echo ""
}

FREE=$(df --output=avail -BG /data08 | tail -1 | tr -dc '0-9')
[ "$FREE" -lt 200 ] && { log "ABORT: disk <200G"; exit 3; }
log "v10 S1 started — ring1 (96 cm^2 = 2x v9 box) collection from v9_step_1250"

# reference point: v9's competence measured on ring 1 (never-used seed, no recording)
R=$(run_eval 3000 ring1_ref "$RING1" "")
log "RING1 baseline (v9_step_1250, seed 3000): ${R:-FAIL}   [v9 in-box 0.766, full-board 0.195]"

rm -rf $COLLECT; mkdir -p $COLLECT
for SEED in 3001 3002 3003 3004 3005 3006 3007 3008; do
  E=$(run_eval $SEED "collect_s$SEED" "$RING1" "$COLLECT")
  N=$(ls $COLLECT/*.npz 2>/dev/null | wc -l)
  log "collect ring1 seed=$SEED success=${E:-FAIL} cumulative_episodes=$N"
done
N=$(ls $COLLECT/*.npz 2>/dev/null | wc -l)
log "v10 S1 DONE: $N successful ring-1 rollouts (disk free $(df -h /data08 | tail -1 | awk '{print $4}'))"
