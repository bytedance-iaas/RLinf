#!/bin/bash
# STATUS: SUPERSEDED — 早期任务规格，已被主线取代。别用来复现。 独立的仿真示范 SFT，被阶段 B 取代
set -uo pipefail
SCRATCH=${SCRATCH:-/tmp/so101_runs}; mkdir -p "$SCRATCH"
cd /data08/henryg/pai/RLinf
export REPO_PATH="$PWD"
export EMBODIED_PATH="$PWD/examples/sft"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Sim-demo SFT: continue from real-data SFT-8000, fine-tune on 100 scripted sim
# grasp demos so the policy has nonzero initial success in sim before RL.
LOG=${SCRATCH:-/tmp/so101_runs}/sft_sim.out

.venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_sim \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_sim \
  > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
