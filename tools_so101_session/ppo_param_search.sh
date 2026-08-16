#!/bin/bash
# Overnight PPO exploration, take 2. Full autonomy granted 2026-08-14 00:20.
#
# WHY THIS SHAPE (user's question, which was the right one): our 640-step budget
# was inherited from the DEMO length requirement (30 Hz, demo median ~440 x1.45),
# never audited as an RL parameter. With num_action_chunks=5 that is 128 noisy
# decisions per episode, while the two runs that ever amplified (pp4, v10) had
# 320/5 = 64, and official ManiSkill has 80/5 = 16. The budget cannot be cut
# (the task needs ~440 steps at 30 Hz), so the way back to 64 decisions is
# chunks 5 -> 10 -- which also stops discarding half the model's output, since
# the policy was SFT-trained with action_horizon 10 and we only ever executed 5.
#
# Every probe is a FREEZE test (lr=1e-9): the real training path, weights
# unchanged, reporting BOTH numbers in one epoch --
#   eval/success_once  (deterministic; must not collapse from ~0.55)
#   env/success_once   (rollout distribution; the PPO precondition, needs >=0.05)
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/ppo_night.status
RES=$SCRATCH/axis_results.txt
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
metric(){ .venv/bin/python - "$1" "$2" <<'PY' 2>/dev/null
import glob,sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
fs=sorted(glob.glob(sys.argv[1]+"/**/events.out.tfevents*", recursive=True))
if not fs: raise SystemExit
ea=EventAccumulator(fs[-1], size_guidance={"scalars":0}); ea.Reload()
t=ea.Tags()["scalars"]
if sys.argv[2] in t and ea.Scalars(sys.argv[2]): print(f"{ea.Scalars(sys.argv[2])[0].value:.4f}")
PY
}

# probe TAG CHUNKS GLOBAL_BATCH [extra hydra override]
probe(){
  local TAG=$1 CH=$2 GB=$3 EXTRA=${4:-}
  local DIR=/data08/henryg/pai/results/probe_$TAG
  clean; rm -rf $DIR
  timeout 2400 .venv/bin/python examples/embodiment/train_embodied_agent.py \
    --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11 \
    runner.logger.log_path=$DIR \
    runner.val_check_interval=1 runner.save_interval=1000 runner.max_epochs=1 \
    actor.optim.lr=1e-9 actor.optim.value_lr=1e-9 \
    env.train.rollout_epoch=1 \
    actor.model.num_action_chunks=$CH actor.global_batch_size=$GB \
    ${EXTRA:+"$EXTRA"} > "$SCRATCH/probe_$TAG.out" 2>&1
  R=$(metric "$DIR" env/success_once); E=$(metric "$DIR" eval/success_once)
  log "PROBE $TAG: chunks=$CH ($(( 640 / CH )) decisions/ep) ${EXTRA:+$EXTRA }-> rollout=${R:-FAIL} eval=${E:-FAIL}"
  echo "${R:-0} ${E:-0} $TAG $CH ${EXTRA:-none}" >> "$RES"
}

: > "$RES"
log "=== overnight PPO exploration, take 2 ==="
log "reference: chunks=5 gave rollout 0.010-0.016 / eval 0.51-0.55 (measured tonight)"

# --- the candidate the user pointed at: back to 64 decisions per episode ---
probe chunks10 10 2048

R10=$(awk '$3=="chunks10"{print $1}' "$RES"); E10=$(awk '$3=="chunks10"{print $2}' "$RES")
if awk -v e="${E10:-0}" 'BEGIN{exit !(e<0.45)}'; then
  log "chunks=10 HURTS the policy itself (eval ${E10} < 0.45): 333 ms of open loop is too long."
  log "-> abandoning the chunk axis; sweeping the parameter flow_noise actually reads instead"
  probe lv_half    5 2048 "+actor.model.openpi.noise_logvar_range=[0.04,0.08]"
  probe lv_quarter 5 2048 "+actor.model.openpi.noise_logvar_range=[0.02,0.04]"
  probe lv_tiny    5 2048 "+actor.model.openpi.noise_logvar_range=[0.005,0.01]"
elif awk -v r="${R10:-0}" 'BEGIN{exit !(r<0.05)}'; then
  log "chunks=10 keeps the policy (eval ${E10}) but rollout ${R10} is still <0.05"
  log "-> pushing the same axis further, then trying the noise magnitude at chunks=10"
  probe chunks20 20 2048
  probe chunks10_lv 10 2048 "+actor.model.openpi.noise_logvar_range=[0.02,0.04]"
else
  log "chunks=10 clears both bars (eval ${E10}, rollout ${R10}) -- going straight to PPO"
fi

# --- pick the winner: highest rollout success among probes whose eval survived ---
BEST=$(awk '$2>=0.45' "$RES" | sort -gr | head -1)
[ -z "$BEST" ] && BEST=$(sort -gr "$RES" | head -1)
BR=$(echo "$BEST" | awk '{print $1}'); BE=$(echo "$BEST" | awk '{print $2}')
BTAG=$(echo "$BEST" | awk '{print $3}'); BCH=$(echo "$BEST" | awk '{print $4}')
BEX=$(echo "$BEST" | cut -d' ' -f5-); [ "$BEX" = "none" ] && BEX=""
ROLLOUT_EPOCH=$(( 24576 / (64 * (640 / BCH)) )); [ "$ROLLOUT_EPOCH" -lt 1 ] && ROLLOUT_EPOCH=1
SAMPLES=$(( 64 * (640 / BCH) * ROLLOUT_EPOCH ))
log "WINNER: $BTAG (chunks=$BCH ${BEX:-official noise}) rollout=$BR eval=$BE"
if awk -v a="$BR" 'BEGIN{exit !(a>=0.05)}'; then
  log "precondition MET (>=0.05, the line separating pp4/v10 from v6/v11)"
else
  log "precondition NOT met; running anyway as an empirical test of that threshold"
fi
log "final config: chunks=$BCH rollout_epoch=$ROLLOUT_EPOCH -> $SAMPLES samples -> $(( SAMPLES / 2048 )) updates/epoch (official invariant 24576/12)"

LOGDIR=/data08/henryg/pai/results/so101_ppo_v12
clean; rm -rf $LOGDIR
setsid .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11 \
  runner.logger.log_path=$LOGDIR \
  runner.val_check_interval=5 runner.save_interval=5 runner.max_epochs=200 \
  actor.model.num_action_chunks=$BCH actor.global_batch_size=2048 \
  env.train.rollout_epoch=$ROLLOUT_EPOCH \
  ${BEX:+"$BEX"} > "$SCRATCH/rl_v12.out" 2>&1 &
TRAIN_PID=$!
log "PPO v12 launched (pid $TRAIN_PID), eval+checkpoint every 5 epochs"

BESTV=0; BEST_STEP=-1; FIRST=""; BELOW=0; SEEN=0
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
    log "eval step=$STEP success_once=$VAL   [peak $BESTV @ $BEST_STEP]"
    awk -v a="$VAL" -v b="$BESTV" 'BEGIN{exit !(a>b)}' && { BESTV=$VAL; BEST_STEP=$STEP; }
    if awk -v v="$VAL" -v b="$BESTV" 'BEGIN{exit !(v < b-0.20)}'; then
      log "AUTO-STOP (collapse): $VAL is >20 points below the peak $BESTV @ step $BEST_STEP"
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
log "V12 FINAL: peak eval=$BESTV @ step $BEST_STEP (start $FIRST) | v10 ring-1 honest 0.551"
log "checkpoints: $(ls -d $LOGDIR/*/checkpoints/global_step_* 2>/dev/null | sort -V | tr '\n' ' ')"
