#!/bin/bash
# STATUS: SUPERSEDED — 早期任务规格，已被主线取代。别用来复现。 上者的评测
set -uo pipefail
SCRATCH=${SCRATCH:-/tmp/so101_runs}; mkdir -p "$SCRATCH"
cd /data08/henryg/pai/RLinf
export REPO_PATH="$PWD"
export EMBODIED_PATH="$PWD/examples/embodiment"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl:${LD_LIBRARY_PATH:-}
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99

# Measure the sim-SFT policy's zero-shot success rate in sim (128 envs).
CKPT=/data08/henryg/pai/results/so101_sft_sim/so101_sft_openpi_pi05/checkpoints/global_step_4000
LOG=${SCRATCH:-/tmp/so101_runs}/eval_simsft.out

.venv/bin/python evaluations/eval_embodied_agent.py \
  --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
  --config-name so101_eval_openpi_pi05 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_eval_simsft \
  rollout.model.model_path="$CKPT" \
  rollout.model.openpi.config_name=pi05_so101_sim \
  rollout.model.openpi_data.norm_stats_path=/data08/henryg/pai/RLinf/assets/pi05_so101_sim/so101-sim-demos/norm_stats.json \
  env.eval.total_num_envs=128 \
  > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
