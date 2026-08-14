#!/bin/bash
# v13 = v12's fixed rollout distribution + the UPDATE SCHEDULE that actually
# amplified in this project's history.
#
# v12 met the rollout precondition with margin (39.1% vs the 5% line) and still
# lost 55 points in 9 epochs, so the precondition is necessary, not sufficient.
# The remaining structural difference from pp4/v10 (the only two runs that ever
# amplified) is updates per epoch: they did exactly 1, v12 did 12 (official).
# By epoch 4 v12 had performed 48 updates -- more than pp4 had done by epoch 48.
#
# Changed vs v12 (two knobs, both documented in the skill as the calibrated
# quantity of the conservative bundle):
#   rollout_epoch 6 -> 1 and global_batch 2048 -> 4096  => 1 update/epoch
#   lr 5e-6 -> 2e-6
# Unchanged: chunks=10, noise_logvar_range [0.02,0.04], update_epoch 1,
#            clip 0.2, entropy 0, value_lr 1e-4, everything else official.
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/v13.status
LOGDIR=/data08/henryg/pai/results/so101_ppo_v13
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
    [ "$p" = "$$" ] && continue; kill -9 $p 2>/dev/null; done
  rm -rf /tmp/ray/session_* 2>/dev/null
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
  sleep 8
}
clean; rm -rf $LOGDIR
log "v13 launching: 1 update/epoch (global_batch 4096 = samples), lr 2e-6, chunks=10, logvar [0.02,0.04]"
setsid .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11 \
  runner.logger.log_path=$LOGDIR \
  runner.val_check_interval=5 runner.save_interval=5 runner.max_epochs=300 \
  actor.model.num_action_chunks=10 \
  env.train.rollout_epoch=1 \
  actor.global_batch_size=4096 \
  actor.optim.lr=2e-6 \
  "+actor.model.openpi.noise_logvar_range=[0.02,0.04]" \
  > "$SCRATCH/rl_v13.out" 2>&1 &
TRAIN_PID=$!
log "train pid=$TRAIN_PID"
BEST=0; BEST_STEP=-1; FIRST=""; BELOW=0; SEEN=0
while :; do
  sleep 300
  kill -0 $TRAIN_PID 2>/dev/null || { log "training exited on its own"; break; }
  EV=$(.venv/bin/python - <<'PY' 2>/dev/null
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
fs=sorted(glob.glob("/data08/henryg/pai/results/so101_ppo_v13/**/events.out.tfevents*", recursive=True))
if not fs: raise SystemExit
ea=EventAccumulator(fs[-1], size_guidance={"scalars":0}); ea.Reload()
if "eval/success_once" not in ea.Tags()["scalars"]: raise SystemExit
for s in ea.Scalars("eval/success_once"): print(s.step, s.value)
PY
)
  [ -z "$EV" ] && continue
  N=$(echo "$EV" | wc -l); [ "$N" -le "$SEEN" ] && continue
  while read -r STEP VAL; do
    [ -z "$VAL" ] && continue
    [ -z "$FIRST" ] && { FIRST=$VAL; log "first eval (guard baseline): step=$STEP $VAL   [probe measured 0.617 at step 0]"; }
    log "eval step=$STEP success_once=$VAL   [peak $BEST @ $BEST_STEP]"
    awk -v a="$VAL" -v b="$BEST" 'BEGIN{exit !(a>b)}' && { BEST=$VAL; BEST_STEP=$STEP; }
    if awk -v v="$VAL" -v b="$BEST" 'BEGIN{exit !(v < b-0.20)}'; then
      log "AUTO-STOP (collapse): $VAL is >20 points below the peak $BEST @ step $BEST_STEP"
      kill -9 $TRAIN_PID 2>/dev/null; break 2; fi
    if awk -v v="$VAL" -v f="$FIRST" 'BEGIN{exit !(v < f-0.05)}'; then BELOW=$((BELOW+1)); else BELOW=0; fi
    if [ "$BELOW" -ge 3 ]; then
      log "AUTO-STOP (no benefit): 3 consecutive evals below the start $FIRST"
      kill -9 $TRAIN_PID 2>/dev/null; break 2; fi
  done < <(echo "$EV" | tail -n +$((SEEN+1)))
  SEEN=$N
done
log "V13 FINAL: peak eval=$BEST @ step $BEST_STEP (start $FIRST) | v12 peaked 0.359 @4 then died; v10 ring-1 honest 0.551"
log "checkpoints: $(ls -d $LOGDIR/*/checkpoints/global_step_* 2>/dev/null | sort -V | tr '\n' ' ')"
