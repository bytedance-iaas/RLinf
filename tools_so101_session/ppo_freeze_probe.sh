#!/bin/bash
# STATUS: ACTIVE — 当前流程在用。 阶段 F 启动前的先决条件探针（lr=1e-9 冻结测试）
# FREEZE TEST — the correct tool, which I should have used instead of only_eval.
# Runs the REAL training path (same workers, same env creation, same model
# construction, same weight sync) with lr = value_lr = 1e-9, so weights are
# effectively unchanged. val_check_interval=1 puts the first eval after ONE
# epoch, i.e. after ~12 updates of size ~0.
#   eval ~0.58 -> the harness is sound; v11's 0.0 at step 9 was training damage
#   eval ~0.00 -> the harness itself produces a broken policy, independent of
#                 any learning
# (only_eval was the wrong probe: rlinf/config.py:830, env_worker.py:104/108 and
#  huggingface_worker.py:51/70 make that flag change the model-config source and
#  skip training-env creation -- three coupled changes, not one.)
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/bisect.status
LOGDIR=/data08/henryg/pai/results/so101_ppo_v11_freeze
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
.venv/bin/ray stop --force >/dev/null 2>&1
for p in $(pgrep -f 'ray::|raylet|gcs_server' 2>/dev/null); do kill -9 $p 2>/dev/null; done
rm -rf /tmp/ray/session_* 2>/dev/null
find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
sleep 8
rm -rf $LOGDIR
log "FREEZE TEST starting (real training path, lr=value_lr=1e-9, eval every epoch)"
timeout 5400 .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11 \
  runner.logger.log_path=$LOGDIR \
  runner.val_check_interval=1 runner.save_interval=1000 runner.max_epochs=3 \
  actor.optim.lr=1e-9 actor.optim.value_lr=1e-9 \
  > "$SCRATCH/freeze_v11.out" 2>&1
log "FREEZE TEST exit=$?"
E=$(.venv/bin/python - <<'PY' 2>/dev/null
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
fs=sorted(glob.glob("/data08/henryg/pai/results/so101_ppo_v11_freeze/**/events.out.tfevents*", recursive=True))
if fs:
    ea=EventAccumulator(fs[-1], size_guidance={"scalars":0}); ea.Reload()
    t=ea.Tags()["scalars"]
    ev=[f"{x.value:.4f}" for x in ea.Scalars("eval/success_once")] if "eval/success_once" in t else []
    tr=[f"{x.value:.4f}" for x in ea.Scalars("env/success_once")] if "env/success_once" in t else []
    print("eval:"+",".join(ev)+" | train(noisy):"+",".join(tr))
PY
)
log "FREEZE TEST result -> ${E:-no metrics}   [standalone eval of same ckpt = 0.578]"
