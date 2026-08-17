#!/bin/bash
# STATUS: EVIDENCE — 一次性对照实验，支撑文档里的某个结论。 动作块长 5 vs 10 的对照。⚠️ 两个臂当时都是 10（手搓 config 不继承 YAML 插值），结论无效
# Does actually emitting 10 actions per inference differ from the 5 every
# reported number was produced with? num_action_chunks (worker-level) sets how
# OFTEN inference is called; openpi.action_chunk (model-level) sets how MANY
# actions come back. Every run so far set only the first.
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/chunk_ab.status
CK=/data08/henryg/pai/results/so101_sft_v15/so101_sft_openpi_pi05/checkpoints/global_step_1000
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
RING1="0.4294,0.9115,0.5142,0.9817"
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
export REPO_PATH=$PWD PYTHONPATH=$PWD HYDRA_FULL_ERROR=1 EMBODIED_PATH=$PWD/examples/embodiment
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p $XDG_RUNTIME_DIR
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data RAY_local_fs_capacity_threshold=0.99
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
run(){ # tag seed extra...
  local TAG=$1 SEED=$2; shift 2
  .venv/bin/ray stop --force >/dev/null 2>&1
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null; sleep 6
  SO101_SPAWN_FRAC="$RING1" timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_chunkab \
    rollout.model.model_path=$CK rollout.model.openpi.config_name=pi05_so101_v15 \
    rollout.model.openpi_data.norm_stats_path=$STATS \
    rollout.model.num_action_chunks=10 \
    env.eval.total_num_envs=128 env.eval.seed=$SEED "$@" > "$SCRATCH/eval_chunkab_$TAG.out" 2>&1
  grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_chunkab_$TAG.out" | tail -1 | cut -d= -f2
}
A=$(run emit5 4141);  log "emit 5 actions/call (what every reported number used): ${A:-FAIL}"
B=$(run emit10 4141 "+rollout.model.openpi.action_chunk=10"); log "emit 10 actions/call: ${B:-FAIL}"
log "reference: v15 step_1000 scored 0.625 on this seed during the night gate"
