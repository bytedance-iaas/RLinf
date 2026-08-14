#!/bin/bash
set -uo pipefail
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
export RAY_local_fs_capacity_threshold=0.99
export SO101_LOG_DIST=1

# default model_path in so101_eval_openpi_pi05.yaml is the SFT global_step_8000
LOG=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad/eval_sft_fixed.out

.venv/bin/python evaluations/eval_embodied_agent.py \
  --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
  --config-name so101_eval_openpi_pi05 \
  runner.logger.log_path=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad/eval_sft_log \
  > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
