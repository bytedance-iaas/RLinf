#!/bin/bash
# Bisect the 57.8% (standalone eval) vs 0.0% (RL harness eval) gap by adding the
# RL-specific model settings to the standalone eval ONE AT A TIME.
#   A: + add_value_head=True, value_after_vlm=True   (a randomly-initialised head,
#      since the SFT checkpoint has none)
#   B: A + noise_params=[0.16,0.12,200]              (builds the ExploreNoiseNet
#      the same way the RL run does)
# Reference: the same checkpoint/seed/region measured 0.578 twenty minutes ago.
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/bisect.status
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
CK=/data08/henryg/pai/results/so101_sft_v10/so101_sft_openpi_pi05/checkpoints/global_step_1000
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
run(){ # tag  extra-overrides...
  local TAG=$1; shift
  .venv/bin/ray stop --force >/dev/null 2>&1
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
  sleep 5
  SO101_SPAWN_FRAC="$RING1" timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
    --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_bisect \
    rollout.model.model_path="$CK" \
    rollout.model.openpi.config_name=pi05_so101_v10 \
    rollout.model.openpi_data.norm_stats_path=$STATS \
    env.eval.total_num_envs=128 env.eval.seed=3131 "$@" > "$SCRATCH/bisect_$TAG.out" 2>&1
  E=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/bisect_$TAG.out" | tail -1 | cut -d= -f2)
  log "BISECT $TAG: ${E:-FAIL}   [baseline 0.578]"
}
run A_valuehead rollout.model.add_value_head=True rollout.model.openpi.value_after_vlm=True
run B_valuehead_noiseparams rollout.model.add_value_head=True rollout.model.openpi.value_after_vlm=True 'rollout.model.openpi.noise_params=[0.16,0.12,200]'
log "BISECT done"
