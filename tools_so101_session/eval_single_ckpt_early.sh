#!/bin/bash
# STATUS: SUPERSEDED — 早期任务规格，已被主线取代。别用来复现。 pp 时代单检查点评测
set -uo pipefail
SCRATCH=${SCRATCH:-/tmp/so101_runs}; mkdir -p "$SCRATCH"
cd /data08/henryg/pai/RLinf
export REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/embodiment" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99
LOG=${SCRATCH:-/tmp/so101_runs}/eval_pp.out
.venv/bin/python evaluations/eval_embodied_agent.py \
  --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
  --config-name so101_eval_openpi_pi05 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_eval_pp \
  rollout.model.model_path=/data08/henryg/pai/results/so101_sft_pp/so101_sft_openpi_pi05/checkpoints/global_step_4000 \
  rollout.model.openpi.config_name=pi05_so101_pp \
  rollout.model.openpi_data.norm_stats_path=/data08/henryg/pai/RLinf/assets/pi05_so101_pp/so101-sim-demos-pp/norm_stats.json \
  env.eval.total_num_envs=128 \
  > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
