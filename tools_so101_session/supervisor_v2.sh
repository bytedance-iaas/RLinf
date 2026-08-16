#!/bin/bash
# SUPERVISOR v2 — autonomy that DIAGNOSES instead of pattern-matching one guess.
# Replaces supervisor_early_v6.sh, whose single-branch fallback stalled the
# night of 2026-08-11 ("needs human") on a failure class it did not know.
#
# Loop, up to MAX_ATTEMPTS times:
#   1. wait for the first training step (deadline)
#   2. if it fails, CLASSIFY the failure from the log's error text:
#        shm      -> clean /dev/shm + remount 16G                -> retry
#        cudaoom  -> halve micro_batch (min 8)                   -> retry
#        port     -> clean ray/session + fresh ports             -> retry
#        unknown  -> run a MINIMAL-SCALE diagnostic (8 envs, mb 8);
#                    if that also fails, extract the last error text into the
#                    status file (so the morning report has the evidence) and
#                    only then stop.
#   3. on success: watch to completion, then verify best ckpt (fresh seed +
#      3 y-bands) and prune checkpoints.
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/v6.status
LOG=$SCRATCH/rl_v6.out
CKROOT=/data08/henryg/pai/results/so101_ppo_v6/so101_ppo_openpi_pi05/checkpoints
MAX_ATTEMPTS=4
MB=32
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] [SUP2] $*" >> "$STATUS"; }

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

clean(){
  .venv/bin/ray stop --force >/dev/null 2>&1 || true
  for p in $(pgrep -f 'ray::|raylet|gcs_server|eval_embodied_agent|train_embodied_agent'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
  done
  rm -rf /tmp/ray/session_* 2>/dev/null
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
}

classify(){  # -> shm | cudaoom | port | unknown
  grep -qi 'creating shared memory segment' "$LOG" 2>/dev/null && { echo shm; return; }
  grep -qiE "Cuda failure 2 'out of memory'|CUDA out of memory" "$LOG" 2>/dev/null && { echo cudaoom; return; }
  grep -qiE 'EADDRINUSE|address already in use|TCPStore' "$LOG" 2>/dev/null && { echo port; return; }
  echo unknown
}

launch(){  # $1 = extra hydra overrides
  clean; sleep 5
  SHM_MB=$(df -m /dev/shm | tail -1 | awk '{print $4}')
  [ "${SHM_MB:-0}" -lt 1024 ] && { mount -o remount,size=16G /dev/shm 2>/dev/null || true; SHM_MB=$(df -m /dev/shm | tail -1 | awk '{print $4}'); }
  log "launching (shm free ${SHM_MB}MB, micro_batch $MB) extra: ${1:-none}"
  setsid .venv/bin/python examples/embodiment/train_embodied_agent.py \
    --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
    --config-name so101_ppo_v6_official \
    runner.logger.log_path=/data08/henryg/pai/results/so101_ppo_v6 \
    actor.micro_batch_size=$MB $1 \
    > "$LOG" 2>&1 </dev/null &
}

wait_first_step(){  # 0 = started, 1 = failed
  local D=0
  while [ $D -lt 2700 ]; do
    grep -q 'success_once=' "$LOG" 2>/dev/null && return 0
    if grep -qiE 'ncclSystemError|ncclUnhandledCudaError|out of memory|Traceback' "$LOG" 2>/dev/null; then
      sleep 30; return 1
    fi
    sleep 60; D=$((D+60))
  done
  return 1
}

log "supervisor v2 online (watching current run first)"
ATTEMPT=1
while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
  if wait_first_step; then
    log "first step OK on attempt $ATTEMPT: $(grep -oE 'success_once=[0-9.]+' "$LOG" | head -1)"
    break
  fi
  CLASS=$(classify)
  ERRTXT=$(grep -A2 -iE 'ncclSystemError|out of memory|Traceback' "$LOG" 2>/dev/null | grep -iE 'Last error|Error|line' | head -2 | cut -c1-200)
  log "attempt $ATTEMPT FAILED — class=$CLASS | $ERRTXT"
  cp "$LOG" "$LOG.attempt${ATTEMPT}_${CLASS}" 2>/dev/null
  ATTEMPT=$((ATTEMPT+1))
  [ $ATTEMPT -gt $MAX_ATTEMPTS ] && { log "exhausted $MAX_ATTEMPTS attempts — stopping, evidence in $LOG.attempt*"; exit 1; }
  case "$CLASS" in
    shm)     log "remedy: purge+enlarge /dev/shm"; clean; mount -o remount,size=16G /dev/shm 2>/dev/null; launch "" ;;
    cudaoom) MB=$(( MB/2 )); [ $MB -lt 8 ] && MB=8; log "remedy: micro_batch -> $MB"; launch "" ;;
    port)    log "remedy: full ray/session purge"; clean; sleep 20; launch "" ;;
    unknown) log "remedy: MINIMAL-SCALE DIAGNOSTIC (8 envs, mb 8) to separate fixed vs scale cost"
             MB=8; launch "env.train.total_num_envs=8 env.train.rollout_epoch=1 env.eval.total_num_envs=8 actor.global_batch_size=1024" ;;
  esac
done

# ---------- watch to completion ----------
while true; do
  sleep 600
  ALIVE=0
  for p in $(pgrep -f 'train_embodied_agent.py'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null); case "$exe" in */python*) ALIVE=1;; esac
  done
  [ "$ALIVE" = "0" ] && { log "training ended -> verification"; break; }
  grep -q 'v6 done' "$STATUS" 2>/dev/null && { log "auto-stop fired -> verification"; break; }
done
clean; sleep 5

SERIES=$(.venv/bin/python "$SCRATCH/parse_eval_series.py" "$LOG" 2>/dev/null | grep -v epochs)
log "eval series: $(echo "$SERIES" | tr '\n' ' ')"
[ -z "$SERIES" ] && { log "no evals recorded"; exit 0; }
BESTIDX=$(echo "$SERIES" | awk '{print NR, $1}' | sort -k2 -gr | head -1 | awk '{print $1}')
BESTVAL=$(echo "$SERIES" | awk '{print NR, $1}' | sort -k2 -gr | head -1 | awk '{print $2}')
STEP=$((BESTIDX * 10))
CK=$CKROOT/global_step_$STEP
[ -d "$CK" ] || CK=$(ls -d $CKROOT/global_step_* 2>/dev/null | sort -V | tail -1)
log "best in-training eval #$BESTIDX = $BESTVAL -> verifying $(basename "$CK")"
mkdir -p "$CK/so101-sim-demos-v4"
cp assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json "$CK/so101-sim-demos-v4/" 2>/dev/null || true
run_eval(){
  local SF="${4:-}" TRY
  for TRY in 1 2; do
    clean
    SO101_SPAWN_FRAC="$SF" timeout 2400 .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
      --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v6 \
      rollout.model.model_path="$1" \
      rollout.model.openpi.config_name=pi05_so101_v4 \
      rollout.model.openpi_data.norm_stats_path=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
      env.eval.total_num_envs=128 env.eval.seed=$2 \
      > "$SCRATCH/eval_v6_$3_t$TRY.out" 2>&1
    local EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v6_$3_t$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
  done
  return 0
}
V=$(run_eval "$CK" 1313 verify_s1313);             log "VERIFY (never-used seed 1313): ${V:-FAIL}"
V2=$(run_eval "$CK" 1414 verify_s1414);            log "VERIFY (never-used seed 1414): ${V2:-FAIL}"
B0=$(run_eval "$CK" 606 band_right "0,1,0,0.33");   log "y-band right : ${B0:-FAIL}"
B1=$(run_eval "$CK" 606 band_mid "0,1,0.33,0.66");  log "y-band middle: ${B1:-FAIL}"
B2=$(run_eval "$CK" 606 band_left "0,1,0.66,1");    log "y-band left  : ${B2:-FAIL}"
echo "$CK" > "$SCRATCH/v6_best.ck"
log "V6 FINAL: ckpt=$(basename "$CK") in-train-best=$BESTVAL verify=${V:-?}/${V2:-?} bands=[${B0:-?} ${B1:-?} ${B2:-?}] (baseline 0.125)"
KEEP="$CK $CKROOT/global_step_$((STEP-10)) $CKROOT/global_step_$((STEP+10))"
for C in $(ls -d $CKROOT/global_step_* 2>/dev/null); do
  case " $KEEP " in *" $C "*) ;; *) rm -rf "$C";; esac
done
log "supervisor v2 done; disk free $(df -h /data08 | tail -1 | awk '{print $4}')"
