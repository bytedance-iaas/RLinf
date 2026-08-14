#!/bin/bash
# Measure the ONE quantity that decides whether PPO can work here: success rate
# in the ROLLOUT distribution (env/success_once), as a function of exploration
# noise. Probe = freeze test (lr=1e-9) so it exercises the real training path
# without changing weights; 1 epoch, rollout_epoch=1 -> ~5 min each.
#
# Reference points measured today: official noise [0.16,0.12,200] -> 0.010,
# while the deterministic eval of the same checkpoint is ~0.54-0.58.
# Target: the highest noise whose rollout success is >= 0.05 (pp4 had 5-9%,
# v10 10-15%; both amplified. v6 0.5% and v11 1.0% never did).
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/noise_sweep.status
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
export REPO_PATH="$PWD" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1
export EMBODIED_PATH=$PWD/examples/embodiment
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SO101_SPAWN_FRAC="0.4294,0.9115,0.5142,0.9817"

clean(){
  .venv/bin/ray stop --force >/dev/null 2>&1
  for p in $(pgrep -f 'ray::|raylet|gcs_server|train_embodied_agent' 2>/dev/null); do
    [ "$p" = "$$" ] && continue; kill -9 $p 2>/dev/null
  done
  rm -rf /tmp/ray/session_* 2>/dev/null
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
  sleep 8
}

probe(){ # tag  "start,end,steps"
  local TAG=$1 NP=$2
  local DIR=/data08/henryg/pai/results/sweep_$TAG
  clean; rm -rf $DIR
  timeout 2400 .venv/bin/python examples/embodiment/train_embodied_agent.py \
    --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11 \
    runner.logger.log_path=$DIR \
    runner.val_check_interval=1 runner.save_interval=1000 runner.max_epochs=1 \
    actor.optim.lr=1e-9 actor.optim.value_lr=1e-9 \
    env.train.rollout_epoch=1 \
    "actor.model.openpi.noise_params=[$NP]" \
    > "$SCRATCH/sweep_$TAG.out" 2>&1
  R=$(.venv/bin/python - "$DIR" <<'PY' 2>/dev/null
import glob,sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
fs=sorted(glob.glob(sys.argv[1]+"/**/events.out.tfevents*", recursive=True))
if not fs: raise SystemExit
ea=EventAccumulator(fs[-1], size_guidance={"scalars":0}); ea.Reload(); t=ea.Tags()["scalars"]
g=lambda k: (f"{ea.Scalars(k)[0].value:.4f}" if k in t and ea.Scalars(k) else "-")
print(f"rollout={g('env/success_once')} eval={g('eval/success_once')} rollout_reward={g('env/reward')}")
PY
)
  log "noise=[$NP]  ${R:-FAILED}"
}

log "noise sweep started (freeze probes; target: rollout success >= 0.05)"
probe official  "0.16,0.12,200"
probe half      "0.08,0.06,200"
probe quarter   "0.04,0.03,200"
probe eighth    "0.02,0.015,200"
log "noise sweep done"
