#!/bin/bash
set -uo pipefail
cd /data08/henryg/pai/RLinf
export REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/sft" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
LOG=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad/sft_pp.out
.venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_pp \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_pp \
  > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
