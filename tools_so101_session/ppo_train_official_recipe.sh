#!/bin/bash
# STATUS: SUPERSEDED — 早期任务规格，已被主线取代。别用来复现。 v11 官方配方首跑：先决条件不满足（带噪 1.0%），失败
# v11 — first reference-aligned PPO run (πRL official recipe, §F of the repro doc).
#
#   start   : v10_step_1000  (ring-1 honest 55.1%)
#   region  : RING 1 via SO101_SPAWN_FRAC (train AND eval)
#   recipe  : update_epoch 1, lr 5e-6 / value_lr 1e-4, clip 0.2, entropy 0,
#             default flow noise [0.16,0.12,200], 24,576 fresh samples/iteration
#             (64 envs x 128 chunks x rollout_epoch 3) -> 12 updates of 2048
#
# The auto-stop guard lives HERE, in the launcher, not in a chat session: a
# session-side watcher dies with the session, and that is exactly how 180
# epochs were once burned degrading a policy past its peak.
# It reads the tensorboard tag `eval/success_once` (distinct from the training
# tag `env/success_once`) and stops on:
#   (a) collapse   : eval < best - 20 points
#   (b) no benefit : 3 consecutive evals below the first eval - 5 points
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/v11.status
LOGDIR=/data08/henryg/pai/results/so101_ppo_v11
RING1="0.4294,0.9115,0.5142,0.9817"
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
export SO101_SPAWN_FRAC="$RING1"          # both env.train and env.eval read this

clean(){
  .venv/bin/ray stop --force >/dev/null 2>&1 || true
  for p in $(pgrep -f 'ray::|raylet|gcs_server|train_embodied_agent|eval_embodied_agent'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
  done
  rm -rf /tmp/ray/session_* 2>/dev/null
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
  SHM=$(df -m /dev/shm | tail -1 | awk '{print $4}')
  [ "${SHM:-0}" -lt 1024 ] && mount -o remount,size=16G /dev/shm 2>/dev/null
  sleep 8
}

FREE=$(df --output=avail -BG /data08 | tail -1 | tr -dc '0-9')
[ "$FREE" -lt 200 ] && { log "ABORT: disk <200G"; exit 3; }

.venv/bin/python -m toolkits.preflight_config \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11 \
  runner.logger.log_path=$LOGDIR > "$SCRATCH/preflight_v11.out" 2>&1
grep -q 'PREFLIGHT OK' "$SCRATCH/preflight_v11.out" || { log "PREFLIGHT FAIL"; tail -6 "$SCRATCH/preflight_v11.out" >> "$STATUS"; exit 1; }
log "preflight OK"

launch(){  # micro_batch_size
  clean
  log "launching PPO (micro_batch_size=$1, ring 1, start = v10_step_1000)"
  setsid .venv/bin/python examples/embodiment/train_embodied_agent.py \
    --config-path $PWD/examples/embodiment/config/ \
    --config-name so101_ppo_v11 \
    runner.logger.log_path=$LOGDIR \
    actor.micro_batch_size=$1 \
    > "$SCRATCH/rl_v11.out" 2>&1 &
  TRAIN_PID=$!
  log "train pid=$TRAIN_PID"
}

rm -rf $LOGDIR; mkdir -p $LOGDIR
launch 32

# --- OOM fallback, pre-declared: micro_batch_size is pure gradient-accumulation
# granularity (loss scaled by 1/grad_accum, no BatchNorm) so halving it does not
# change the math; global_batch_size 2048 must NOT move, it sets updates/epoch.
sleep 600
if grep -qaE 'out of memory|CUDA error: out of memory' "$SCRATCH/rl_v11.out"; then
  log "OOM at micro_batch_size=32 -> retrying at 16 (pure grad-accum change)"
  kill -9 $TRAIN_PID 2>/dev/null
  launch 16
fi

# --- guard loop -------------------------------------------------------------
read_evals(){
  .venv/bin/python - <<'PY' 2>/dev/null
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
fs=sorted(glob.glob("/data08/henryg/pai/results/so101_ppo_v11/**/events.out.tfevents*", recursive=True))
if not fs: raise SystemExit
ea=EventAccumulator(fs[-1], size_guidance={"scalars":0}); ea.Reload()
if "eval/success_once" not in ea.Tags()["scalars"]: raise SystemExit
for s in ea.Scalars("eval/success_once"): print(s.step, s.value)
PY
}

BEST=0; BEST_STEP=-1; FIRST=""; BELOW=0; SEEN=0
while :; do
  sleep 300
  if ! kill -0 $TRAIN_PID 2>/dev/null; then
    log "training process exited on its own"; break
  fi
  EV=$(read_evals); [ -z "$EV" ] && continue
  N=$(echo "$EV" | wc -l)
  [ "$N" -le "$SEEN" ] && continue
  while read -r STEP VAL; do
    [ -z "$VAL" ] && continue
    [ -z "$FIRST" ] && { FIRST=$VAL; log "first eval (treated as the zero-shot baseline for the guard): step=$STEP $VAL"; }
    log "eval step=$STEP success_once=$VAL   [best so far $BEST @ step $BEST_STEP]"
    if awk -v a="$VAL" -v b="$BEST" 'BEGIN{exit !(a>b)}'; then BEST=$VAL; BEST_STEP=$STEP; fi
    if awk -v v="$VAL" -v b="$BEST" 'BEGIN{exit !(v < b-0.20)}'; then
      log "AUTO-STOP (collapse): $VAL is more than 20 points below the peak $BEST @ step $BEST_STEP"
      kill -9 $TRAIN_PID 2>/dev/null; break 2
    fi
    if awk -v v="$VAL" -v f="$FIRST" 'BEGIN{exit !(v < f-0.05)}'; then
      BELOW=$((BELOW+1))
    else
      BELOW=0
    fi
    if [ "$BELOW" -ge 3 ]; then
      log "AUTO-STOP (no benefit): 3 consecutive evals below the start $FIRST"
      kill -9 $TRAIN_PID 2>/dev/null; break 2
    fi
  done < <(echo "$EV" | tail -n +$((SEEN+1)))
  SEEN=$N
done

log "V11 FINAL: peak eval=$BEST at step $BEST_STEP (start $FIRST) | v10 ring-1 honest was 0.551"
CK=$(ls -d $LOGDIR/*/checkpoints/global_step_* 2>/dev/null | sort -V | tr '\n' ' ')
log "checkpoints: ${CK:-none}"
log "NOTE: the peak checkpoint is the deliverable; verify it on never-used seeds before believing this number"
