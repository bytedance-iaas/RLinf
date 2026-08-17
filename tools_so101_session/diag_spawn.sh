#!/bin/bash
# STATUS: TOOL — 通用工具，与具体阶段无关。 生成位置与成败的相关性诊断
# Spawn-vs-outcome diagnostic on the round-2 best ckpt, FRESH seeds (also an
# unbiased re-verification of the gate number). Writes spawn_pp6.csv rows.
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/pp6.status
CK=$(cat "$SCRATCH/pp6_best.ck")
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

export REPO_PATH="$PWD" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1
export EMBODIED_PATH=$PWD/examples/embodiment
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SO101_SPAWN_LOG=$SCRATCH/spawn_pp6.csv

rm -f "$SO101_SPAWN_LOG"
log "spawn diagnostic starting on $CK"
for SEED in 2020 2121; do
  .venv/bin/ray stop --force >/dev/null 2>&1 || true
  for p in $(pgrep -f 'ray::|raylet|gcs_server'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
  done
  rm -rf /tmp/ray/session_* 2>/dev/null
  .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
    --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_pp6_diag \
    rollout.model.model_path="$CK" \
    rollout.model.openpi.config_name=pi05_so101_pp6 \
    rollout.model.openpi_data.norm_stats_path=/data08/henryg/pai/RLinf/assets/pi05_so101_pp/so101-sim-demos-pp/norm_stats.json \
    env.eval.total_num_envs=128 \
    env.eval.seed=$SEED \
    > "$SCRATCH/diag_spawn_s$SEED.out" 2>&1
  EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/diag_spawn_s$SEED.out" | tail -1)
  log "spawn-diag seed=$SEED $EV rows=$(wc -l < $SO101_SPAWN_LOG 2>/dev/null)"
done
log "spawn diagnostic DONE"
