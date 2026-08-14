#!/bin/bash
# Decisive isolation: does v10_step_1000 still reproduce its afternoon numbers
# (ring 1 = 0.551, full board = 0.102) with the SAME standalone eval path?
#   - reproduces  -> the RL harness is what differs; bisect there next
#   - also ~0     -> the machine/env changed since 19:44 (prime suspect: the
#                    ManiSkill asset pack copied into rlinf/envs/maniskill/assets)
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
run(){ # frac tag seed expect
  .venv/bin/ray stop --force >/dev/null 2>&1
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
  sleep 5
  SO101_SPAWN_FRAC="$1" timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
    --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_bisect \
    rollout.model.model_path="$CK" \
    rollout.model.openpi.config_name=pi05_so101_v10 \
    rollout.model.openpi_data.norm_stats_path=$STATS \
    env.eval.total_num_envs=128 env.eval.seed=$3 > "$SCRATCH/bisect_$2.out" 2>&1
  E=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/bisect_$2.out" | tail -1 | cut -d= -f2)
  log "$2: ${E:-FAIL}   [今天下午实测 $4]"
}
log "isolation started (same eval path as this afternoon, same ckpt)"
run "$RING1" ring1_seed3131 3131 0.5703
run "0,1,0,1" fullboard_seed3131 3131 0.1016
log "isolation done"
