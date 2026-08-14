#!/bin/bash
# Honest verification of the PPO peak checkpoint (v13 global_step_30, eval 0.7344
# at step 29 under the gate protocol). Seeds 4141/4242 have never been used
# anywhere in this project -- not for gating, not for collection.
# Evaluated with num_action_chunks=10, i.e. exactly how it was trained.
# References: ring 1 is the trained region; in-box (legacy) and full board are
# continuity references against the v9/v10 numbers.
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/v13.status
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
CK=/data08/henryg/pai/results/so101_ppo_v13/so101_ppo_v11/checkpoints/global_step_30
RING1="0.4294,0.9115,0.5142,0.9817"
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
export REPO_PATH="$PWD" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1
export EMBODIED_PATH=$PWD/examples/embodiment
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
run(){ # tag seed frac [spawn_mode]
  local TRY EV
  for TRY in 1 2 3; do
    .venv/bin/ray stop --force >/dev/null 2>&1
    find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
    sleep 6
    SO101_SPAWN_FRAC="$3" SO101_SPAWN_MODE="${4:-}" timeout 1800 \
      .venv/bin/python evaluations/eval_embodied_agent.py \
        --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
        runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v13 \
        rollout.model.model_path="$CK" \
        rollout.model.openpi.config_name=pi05_so101_v10 \
        rollout.model.openpi_data.norm_stats_path=$STATS \
        rollout.model.num_action_chunks=10 \
        env.eval.total_num_envs=128 env.eval.seed=$2 \
        > "$SCRATCH/eval_v13_$1_r$TRY.out" 2>&1
    EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v13_$1_r$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
  done
  echo ""
}
log "HONEST VERIFICATION of the PPO peak (global_step_30), chunks=10, never-used seeds"
V1=$(run ring1_4141 4141 "$RING1");     log "VERIFY ring1 seed4141: ${V1:-FAIL}"
V2=$(run ring1_4242 4242 "$RING1");     log "VERIFY ring1 seed4242: ${V2:-FAIL}"
IB=$(run inbox_4141 4141 "0,1,0,1" legacy); log "IN-BOX (legacy) seed4141: ${IB:-FAIL}   [v9 0.766 / v10 0.750, both at chunks=5]"
FB=$(run full_4141  4141 "0,1,0,1");        log "FULL-BOARD seed4141: ${FB:-FAIL}       [v9 0.195 / v10 0.102, both at chunks=5]"
log "V13 HONEST: ring1=${V1:-?}/${V2:-?} inbox=${IB:-?} fullboard=${FB:-?}"
