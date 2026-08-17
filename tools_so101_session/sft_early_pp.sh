#!/bin/bash
# STATUS: SUPERSEDED — 早期任务规格，已被主线取代。别用来复现。 pp 时代 SFT，任务规格已废弃
set -uo pipefail
SCRATCH=${SCRATCH:-/tmp/so101_runs}; mkdir -p "$SCRATCH"
cd /data08/henryg/pai/RLinf
export REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/sft" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
LOG=${SCRATCH:-/tmp/so101_runs}/sft_pp.out
.venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_pp \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_pp \
  > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
