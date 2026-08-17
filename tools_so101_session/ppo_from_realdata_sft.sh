#!/bin/bash
# STATUS: SUPERSEDED — 早期任务规格，已被主线取代。别用来复现。 从真机 SFT 直接起 PPO，先决条件不满足
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
export RAY_local_fs_capacity_threshold=0.99

# NOTE: no model_path override -> uses the config default = SFT global_step_8000
OUT=/data08/henryg/pai/results/so101_ppo_sft_restart
LOG=${SCRATCH:-/tmp/so101_runs}/rl_sft_restart.out

.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
  --config-name so101_ppo_openpi_pi05 \
  runner.logger.log_path="$OUT" \
  env.train.total_num_envs=128 \
  env.eval.total_num_envs=128 \
  actor.global_batch_size=2048 \
  > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
