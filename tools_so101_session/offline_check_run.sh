#!/bin/bash
cd /data08/henryg/pai/RLinf
export REPO_PATH=$PWD PYTHONPATH=$PWD HF_LEROBOT_HOME=/data08/henryg/pai/data
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export CUDA_VISIBLE_DEVICES=0
timeout 3600 .venv/bin/python tools_so101_session/offline_replay_check.py \
  --ckpt /data08/henryg/pai/results/so101_ppo_v13/so101_ppo_v11/checkpoints/global_step_30 \
  --real-root /data08/henryg/pai/data/so101-pick-place-v1-trimmed \
  --episodes 5 --frames 12
