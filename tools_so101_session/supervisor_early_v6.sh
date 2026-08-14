#!/bin/bash
# OVERNIGHT SUPERVISOR (user: "夜间训练，全部同意", 2026-08-11).
# Autonomy lives in THIS script, not in the agent's session (skill §6: a
# disconnected session freezes the agent's reactions).
#
#  A. Watch the v6 run (ppo_early_v6.sh already carries the auto-stop rules).
#  B. FALLBACK (pre-declared): if it dies before the first training step AND the
#     log shows CUDA OOM, relaunch ONCE with a smaller memory footprint
#     (micro_batch 16, 32 train envs, rollout_epoch 6 -> SAME 24,576 samples/iter,
#     same global_batch 2048, same update_epoch 1 => recipe math unchanged).
#  C. When the run ends: pick the best eval from the training log, map it to the
#     saved checkpoint, and produce an HONEST verification (fresh seed 909 +
#     3 y-bands) so the morning report has real numbers.
#  D. Checkpoint hygiene: keep best + its two neighbours, delete the rest.
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/v6.status
LOG=$SCRATCH/rl_v6.out
CKROOT=/data08/henryg/pai/results/so101_ppo_v6/so101_ppo_openpi_pi05/checkpoints
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] [SUP] $*" >> "$STATUS"; }

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

log "supervisor started (pid $$)"

clean(){
  .venv/bin/ray stop --force >/dev/null 2>&1 || true
  for p in $(pgrep -f 'ray::|raylet|gcs_server|eval_embodied_agent|train_embodied_agent'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
  done
  rm -rf /tmp/ray/session_* 2>/dev/null
}

# ---------- A/B: watch for the first training step, fallback once on OOM ----------
D=0
FIRST=0
while [ $D -lt 3000 ]; do
  grep -q 'first step' "$STATUS" 2>/dev/null && { FIRST=1; break; }
  if grep -qiE 'out of memory|ncclSystemError|ncclUnhandledCudaError|Broadcast failed' "$LOG" 2>/dev/null; then
    log "GPU-memory/NCCL failure detected before first step -> applying pre-declared fallback"
    clean; sleep 5
    mv "$LOG" "$LOG.oom_fallback" 2>/dev/null
    setsid .venv/bin/python examples/embodiment/train_embodied_agent.py \
      --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
      --config-name so101_ppo_v6_official \
      runner.logger.log_path=/data08/henryg/pai/results/so101_ppo_v6 \
      actor.micro_batch_size=16 \
      env.train.total_num_envs=16 \
      env.train.rollout_epoch=12 \
      env.eval.total_num_envs=64 \
      > "$LOG" 2>&1 </dev/null &
    log "fallback launched (micro_batch 16, envs 32, rollout_epoch 6; samples/iter unchanged)"
    D=0
    while [ $D -lt 3000 ]; do
      grep -q 'success_once=' "$LOG" 2>/dev/null && { FIRST=1; log "fallback first step: $(grep -oE 'success_once=[0-9.]+' "$LOG" | head -1)"; break; }
      sleep 60; D=$((D+60))
    done
    break
  fi
  sleep 60; D=$((D+60))
done
[ "$FIRST" = "1" ] || { log "ABORT: no first training step and no OOM signature — needs human"; exit 1; }

# ---------- wait for the run to end (ppo_early_v6.sh auto-stop, or trainer death) ----------
while true; do
  sleep 600
  ALIVE=0
  for p in $(pgrep -f 'train_embodied_agent.py'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null); case "$exe" in */python*) ALIVE=1;; esac
  done
  [ "$ALIVE" = "0" ] && { log "training process gone -> proceeding to verification"; break; }
  grep -q 'v6 done' "$STATUS" 2>/dev/null && { log "auto-stop fired -> proceeding to verification"; break; }
done
clean; sleep 5

# ---------- C: honest verification of the best checkpoint ----------
SERIES=$(.venv/bin/python "$SCRATCH/parse_eval_series.py" "$LOG" 2>/dev/null | grep -v epochs)
log "eval series: $(echo "$SERIES" | tr '\n' ' ')"
BESTIDX=$(echo "$SERIES" | awk '{print NR, $1}' | sort -k2 -gr | head -1 | awk '{print $1}')
BESTVAL=$(echo "$SERIES" | awk '{print NR, $1}' | sort -k2 -gr | head -1 | awk '{print $2}')
if [ -z "${BESTIDX:-}" ]; then log "no eval rows — nothing to verify"; exit 0; fi
STEP=$((BESTIDX * 10))
CK=$CKROOT/global_step_$STEP
if [ ! -d "$CK" ]; then
  CK=$(ls -d $CKROOT/global_step_* 2>/dev/null | sort -V | tail -1)
  log "mapped ckpt global_step_$STEP missing; falling back to $(basename "$CK")"
fi
log "best in-training eval #$BESTIDX = $BESTVAL -> verifying $(basename "$CK")"
mkdir -p "$CK/so101-sim-demos-v4"
cp assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json "$CK/so101-sim-demos-v4/" 2>/dev/null || true

run_eval(){ # ckpt seed tag [spawnfrac]
  local SF="${4:-}"
  local TRY
  for TRY in 1 2; do
    clean
    SO101_SPAWN_FRAC="$SF" timeout 2400 .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
      --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v6 \
      rollout.model.model_path="$1" \
      rollout.model.openpi.config_name=pi05_so101_v4 \
      rollout.model.openpi_data.norm_stats_path=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
      env.eval.total_num_envs=128 \
      env.eval.seed=$2 \
      > "$SCRATCH/eval_v6_$3_t$TRY.out" 2>&1
    local EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v6_$3_t$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
    log "eval $3 try$TRY empty, retry"
  done
  return 0
}
V=$(run_eval "$CK" 909 verify_s909);              log "VERIFY (fresh seed 909): ${V:-FAIL}"
B0=$(run_eval "$CK" 606 band_right "0,1,0,0.33");  log "y-band right : ${B0:-FAIL}"
B1=$(run_eval "$CK" 606 band_mid "0,1,0.33,0.66"); log "y-band middle: ${B1:-FAIL}"
B2=$(run_eval "$CK" 606 band_left "0,1,0.66,1");   log "y-band left  : ${B2:-FAIL}"
echo "$CK" > "$SCRATCH/v6_best.ck"
log "V6 FINAL: ckpt=$(basename "$CK") in-train-best=$BESTVAL verify=${V:-?} bands=[${B0:-?} ${B1:-?} ${B2:-?}] (baseline v4_1000 = 0.125)"

# ---------- D: checkpoint hygiene ----------
KEEP="$CK $CKROOT/global_step_$((STEP-10)) $CKROOT/global_step_$((STEP+10))"
for C in $(ls -d $CKROOT/global_step_* 2>/dev/null); do
  case " $KEEP " in *" $C "*) ;; *) rm -rf "$C"; log "deleted $(basename "$C")";; esac
done
log "supervisor done; disk free: $(df -h /data08 | tail -1 | awk '{print $4}')"
