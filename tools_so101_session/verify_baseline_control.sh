#!/bin/bash
# The missing control for the PPO gain claim: the SAME start checkpoint (v10),
# at the SAME chunks=10, on the SAME never-used seeds the PPO peak was verified
# on. Without this, "PPO added N points" only holds on the gate's fixed eval set.
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/v13.status
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
CK=/data08/henryg/pai/results/so101_sft_v10/so101_sft_openpi_pi05/checkpoints/global_step_1000
RING1="0.4294,0.9115,0.5142,0.9817"
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
export REPO_PATH="$PWD" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1 EMBODIED_PATH=$PWD/examples/embodiment
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data RAY_local_fs_capacity_threshold=0.99
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# wait for the offline replay check to release GPU 0
while pgrep -f offline_replay_check.py >/dev/null; do sleep 60; done
sleep 20
run(){ local TRY EV
  for TRY in 1 2 3; do
    .venv/bin/ray stop --force >/dev/null 2>&1
    find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null; sleep 6
    SO101_SPAWN_FRAC="$RING1" timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v13base \
      rollout.model.model_path="$CK" rollout.model.openpi.config_name=pi05_so101_v10 \
      rollout.model.openpi_data.norm_stats_path=$STATS \
      rollout.model.num_action_chunks=10 \
      env.eval.total_num_envs=128 env.eval.seed=$1 > "$SCRATCH/eval_base_$1.out" 2>&1
    EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_base_$1.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
  done; echo ""; }
log "BASELINE control: v10_step_1000 at chunks=10 on the PPO verification seeds"
B1=$(run 4141); log "BASELINE ring1 seed4141: ${B1:-FAIL}   [PPO peak got 0.5859]"
B2=$(run 4242); log "BASELINE ring1 seed4242: ${B2:-FAIL}   [PPO peak got 0.5703]"
log "PPO HONEST GAIN: v13 0.5859/0.5703 vs v10 ${B1:-?}/${B2:-?} (same seeds, same chunks)"
