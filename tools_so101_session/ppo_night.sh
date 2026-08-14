#!/bin/bash
# Overnight PPO exploration (user granted full autonomy 2026-08-13 23:15).
#
# PHASE 1 (already running as noise_sweep.sh): rollout success vs noise level.
# PHASE 2 (here): rollout success vs DECISIONS PER EPISODE. Direct test of the
#   structural explanation: noise is injected per decision, so staying on the BC
#   ridge is a product over decisions. decisions = max_episode_steps/chunks:
#   5->128 (ours), 10->64, 20->32 (official ManiSkill is 16). If the compounding
#   story is right, rollout success must climb steeply as chunks grow, at the
#   SAME per-decision noise. If it does not, the story is wrong.
# PHASE 3 (here): pick a configuration by PRE-REGISTERED rules and run PPO with
#   the in-launcher auto-stop guard.
#
# Batch arithmetic is preserved in every probe (samples % global_batch == 0 and
# per-rank % micro_batch == 0); the final run restores the official invariant of
# 24,576 fresh samples -> 12 updates per iteration by adjusting rollout_epoch.
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/ppo_night.status
SWEEP=$SCRATCH/noise_sweep.status
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
  SHM=$(df -m /dev/shm | tail -1 | awk '{print $4}')
  [ "${SHM:-0}" -lt 1024 ] && mount -o remount,size=16G /dev/shm 2>/dev/null
  sleep 8
}

read_metric(){ # dir tag
  .venv/bin/python - "$1" "$2" <<'PY' 2>/dev/null
import glob,sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
fs=sorted(glob.glob(sys.argv[1]+"/**/events.out.tfevents*", recursive=True))
if not fs: raise SystemExit
ea=EventAccumulator(fs[-1], size_guidance={"scalars":0}); ea.Reload()
t=ea.Tags()["scalars"]
if sys.argv[2] in t and ea.Scalars(sys.argv[2]):
    print(f"{ea.Scalars(sys.argv[2])[0].value:.4f}")
PY
}

# ---------- wait for phase 1 ----------
sweep_done(){ grep -q 'noise sweep done' "$SWEEP" 2>/dev/null; }
DL=$(( $(date +%s) + 3*3600 ))
while ! sweep_done; do
  [ "$(date +%s)" -gt "$DL" ] && { log "ABORT: phase 1 never finished"; exit 1; }
  if ! pgrep -f 'bash .*noise_sweep.sh' >/dev/null; then
    sleep 5; sweep_done && break
    log "ABORT: noise sweep exited without its marker"; exit 1
  fi
  sleep 60
done
log "PHASE 1 results:"; grep -a 'noise=' "$SWEEP" >> "$STATUS"

# ---------- phase 2: decisions per episode ----------
# chunks | decisions | samples(rollout_epoch=1) | global_batch | updates
#    5   |   128     | 64*128 =  8192           | 2048         | 4
#   10   |    64     | 64* 64 =  4096           | 2048         | 2
#   16   |    40     | 64* 40 =  2560           | 1280         | 2
#   20   |    32     | 64* 32 =  2048           | 2048         | 1
probe_chunks(){ # chunks global_batch
  local CH=$1 GB=$2 DIR=/data08/henryg/pai/results/sweep_chunk$1
  clean; rm -rf $DIR
  timeout 2400 .venv/bin/python examples/embodiment/train_embodied_agent.py \
    --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11 \
    runner.logger.log_path=$DIR \
    runner.val_check_interval=1 runner.save_interval=1000 runner.max_epochs=1 \
    actor.optim.lr=1e-9 actor.optim.value_lr=1e-9 \
    env.train.rollout_epoch=1 \
    actor.model.num_action_chunks=$CH actor.global_batch_size=$GB \
    > "$SCRATCH/sweep_chunk$CH.out" 2>&1
  local R=$(read_metric "$DIR" env/success_once)
  local E=$(read_metric "$DIR" eval/success_once)
  log "chunks=$CH ($(( 640 / CH )) decisions/episode)  rollout=${R:-FAILED}  eval=${E:-?}"
  echo "${R:-0} $CH" >> "$SCRATCH/chunk_results.txt"
}
: > "$SCRATCH/chunk_results.txt"
log "PHASE 2: decisions-per-episode sweep at OFFICIAL noise [0.16,0.12,200]"
probe_chunks 10 2048
probe_chunks 20 2048
probe_chunks 16 1280

# ---------- phase 3: choose and launch ----------
# Pre-registered rules, in order:
#  1. Prefer keeping the official recipe intact: if a NOISE level with chunks=5
#     reaches rollout >= 0.05, take the HIGHEST such noise.
#  2. Else, if a CHUNK setting at official noise reaches >= 0.05, take the
#     SMALLEST such chunk (least interface change).
#  3. Else take whatever scored highest overall and run anyway -- an empirical
#     test of the >=5% threshold itself, which is worth one night.
BEST_NOISE=""; BEST_NOISE_VAL=0
while read -r line; do
  np=$(echo "$line" | grep -oE '\[[0-9.,]+\]' | tr -d '[]')
  rv=$(echo "$line" | grep -oE 'rollout=[0-9.]+' | cut -d= -f2)
  [ -z "$np" ] || [ -z "$rv" ] && continue
  if awk -v a="$rv" 'BEGIN{exit !(a>=0.05)}'; then
    # higher noise wins; the sweep is ordered high->low so take the FIRST hit
    [ -z "$BEST_NOISE" ] && { BEST_NOISE="$np"; BEST_NOISE_VAL=$rv; }
  fi
done < <(grep -a 'noise=' "$SWEEP")

CHUNKS=5; ROLLOUT_EPOCH=3; GB=2048; NOISE="0.16,0.12,200"; WHY=""
if [ -n "$BEST_NOISE" ]; then
  NOISE="$BEST_NOISE"; WHY="rule 1: noise [$NOISE] gives rollout $BEST_NOISE_VAL >= 0.05 with the official chunk size"
else
  BESTCH=""; BESTCHV=0
  while read -r rv ch; do
    if awk -v a="$rv" 'BEGIN{exit !(a>=0.05)}'; then
      if [ -z "$BESTCH" ] || [ "$ch" -lt "$BESTCH" ]; then BESTCH=$ch; BESTCHV=$rv; fi
    fi
  done < "$SCRATCH/chunk_results.txt"
  if [ -n "$BESTCH" ]; then
    CHUNKS=$BESTCH; WHY="rule 2: chunks=$BESTCH gives rollout $BESTCHV >= 0.05 at official noise"
    ROLLOUT_EPOCH=$(( 24576 / (64 * (640 / CHUNKS)) ))
    [ "$ROLLOUT_EPOCH" -lt 1 ] && ROLLOUT_EPOCH=1
  else
    BESTALL=$(sort -gr "$SCRATCH/chunk_results.txt" | head -1)
    WHY="rule 3: NOTHING reached 0.05; running the best available anyway as an empirical test of the threshold ($BESTALL)"
    CHUNKS=$(echo "$BESTALL" | awk '{print $2}'); [ -z "$CHUNKS" ] && CHUNKS=5
    ROLLOUT_EPOCH=$(( 24576 / (64 * (640 / CHUNKS)) ))
    [ "$ROLLOUT_EPOCH" -lt 1 ] && ROLLOUT_EPOCH=1
  fi
fi
SAMPLES=$(( 64 * (640 / CHUNKS) * ROLLOUT_EPOCH ))
UPDATES=$(( SAMPLES / GB ))
log "PHASE 3 decision -> $WHY"
log "PHASE 3 config: chunks=$CHUNKS noise=[$NOISE] rollout_epoch=$ROLLOUT_EPOCH -> $SAMPLES samples, $UPDATES updates/epoch (official invariant: 24576 / 12)"

LOGDIR=/data08/henryg/pai/results/so101_ppo_v12
clean; rm -rf $LOGDIR
setsid .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11 \
  runner.logger.log_path=$LOGDIR \
  runner.val_check_interval=5 runner.save_interval=5 runner.max_epochs=200 \
  actor.model.num_action_chunks=$CHUNKS actor.global_batch_size=$GB \
  env.train.rollout_epoch=$ROLLOUT_EPOCH \
  "actor.model.openpi.noise_params=[$NOISE]" \
  > "$SCRATCH/rl_v12.out" 2>&1 &
TRAIN_PID=$!
log "PPO v12 launched (pid $TRAIN_PID); eval every 5 epochs, checkpoint every 5"

BEST=0; BEST_STEP=-1; FIRST=""; BELOW=0; SEEN=0
while :; do
  sleep 300
  kill -0 $TRAIN_PID 2>/dev/null || { log "training exited on its own"; break; }
  EV=$(.venv/bin/python - <<'PY' 2>/dev/null
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
fs=sorted(glob.glob("/data08/henryg/pai/results/so101_ppo_v12/**/events.out.tfevents*", recursive=True))
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
    [ -z "$FIRST" ] && { FIRST=$VAL; log "first eval (guard baseline): step=$STEP $VAL"; }
    log "eval step=$STEP success_once=$VAL   [peak $BEST @ $BEST_STEP]"
    awk -v a="$VAL" -v b="$BEST" 'BEGIN{exit !(a>b)}' && { BEST=$VAL; BEST_STEP=$STEP; }
    if awk -v v="$VAL" -v b="$BEST" 'BEGIN{exit !(v < b-0.20)}'; then
      log "AUTO-STOP (collapse): $VAL is >20 points below the peak $BEST @ step $BEST_STEP"
      kill -9 $TRAIN_PID 2>/dev/null; break 2
    fi
    if awk -v v="$VAL" -v f="$FIRST" 'BEGIN{exit !(v < f-0.05)}'; then BELOW=$((BELOW+1)); else BELOW=0; fi
    if [ "$BELOW" -ge 3 ]; then
      log "AUTO-STOP (no benefit): 3 consecutive evals below the start $FIRST"
      kill -9 $TRAIN_PID 2>/dev/null; break 2
    fi
  done < <(echo "$EV" | tail -n +$((SEEN+1)))
  SEEN=$N
done
log "V12 FINAL: peak eval=$BEST @ step $BEST_STEP (start $FIRST) | v10 ring-1 honest 0.551"
log "checkpoints: $(ls -d $LOGDIR/*/checkpoints/global_step_* 2>/dev/null | sort -V | tr '\n' ' ')"
