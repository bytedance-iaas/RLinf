#!/bin/bash
# Decisive: run the v11 RL config in EVAL-ONLY mode (no gradient updates).
# The RL run's first eval landed at step 9 -- i.e. after ~108 updates -- so its
# 0.0 cannot distinguish "the harness builds a broken policy" from "108 updates
# on success-free rollouts destroyed it". This does.
#   ~0.58 -> harness fine; the 0 was training-induced degradation
#   ~0.00 -> the harness itself is the problem
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/bisect.status
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
export SO101_SPAWN_FRAC="0.4294,0.9115,0.5142,0.9817"
.venv/bin/ray stop --force >/dev/null 2>&1
find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
sleep 5
log "ONLY-EVAL of the v11 RL config starting (no updates at all)"
timeout 3600 .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11_eval \
  runner.logger.log_path=/data08/henryg/pai/results/so101_ppo_v11_onlyeval \
  runner.max_epochs=1 \
  > "$SCRATCH/onlyeval_v11.out" 2>&1
E=$(grep -aoE 'success_once=[0-9.]+' "$SCRATCH/onlyeval_v11.out" | tail -1 | cut -d= -f2)
log "ONLY-EVAL result: ${E:-FAIL}   [standalone eval of the same ckpt = 0.578]"
