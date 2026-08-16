#!/bin/bash
set -uo pipefail
SCRATCH=${SCRATCH:-/tmp/so101_runs}; mkdir -p "$SCRATCH"
cd /data08/henryg/pai/RLinf
export REPO_PATH="$PWD"
export EMBODIED_PATH="$PWD/examples/embodiment"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
# rendering (RL renders ManiSkill for the rollout obs)
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl:${LD_LIBRARY_PATH:-}
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
# everything (weights, dataset, tokenizer) is local & cached from SFT -> stay offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Ray: /tmp is nearly full on this box -> raise the object-store disk threshold
export RAY_local_fs_capacity_threshold=0.99

CKPT=/data08/henryg/pai/results/so101_ppo_run/so101_ppo_openpi_pi05/checkpoints/global_step_750
OUT=/data08/henryg/pai/results/so101_ppo_run_fixed
LOG=${SCRATCH:-/tmp/so101_runs}/rl_resume750.out

.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
  --config-name so101_ppo_openpi_pi05 \
  runner.logger.log_path="$OUT" \
  actor.model.model_path="$CKPT" \
  rollout.model.model_path="$CKPT" \
  env.train.total_num_envs=128 \
  env.eval.total_num_envs=128 \
  actor.global_batch_size=2048 \
  > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
