#!/bin/bash
# Autonomous overnight orchestrator: waits for v7 Phase A (critic warmup) to
# finish, then launches Phase B (normal-lr amplification) from the warmed ckpt.
# Runs fully detached — no human interaction needed. All status goes to STATUS.
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/v7_orchestrator.status
ALOG=$SCRATCH/rl_v7a.out
BLOG=$SCRATCH/rl_v7b.out
CKDIR=/data08/henryg/pai/results/so101_ppo_v7/so101_ppo_openpi_pi05/checkpoints
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_sim/so101-sim-demos/norm_stats.json
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

log "orchestrator started (pid $$)"

train_alive(){
  local T=0 p exe
  for p in $(pgrep -f 'train_embodied_agent.py'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in */python*) T=1;; esac
  done
  [ "$T" = "1" ]
}

# ---- 1. wait for Phase A to end (EXIT marker or process gone) ----
while ! grep -qE 'EXIT=' "$ALOG" 2>/dev/null && train_alive; do sleep 120; done
sleep 30
AEXIT=$(grep -oE 'EXIT=[0-9]+' "$ALOG" | tail -1 || echo "EXIT=?")
AEP=$(grep -c 'success_once=' "$ALOG" 2>/dev/null || echo 0)
log "Phase A ended: $AEXIT at epoch $AEP"

# ---- 2. pick the best warmup checkpoint ----
CK=""
for s in global_step_60 global_step_40 global_step_20; do
  [ -d "$CKDIR/$s/actor" ] && { CK="$CKDIR/$s"; break; }
done
if [ -z "$CK" ]; then log "FATAL: no Phase A checkpoint found; aborting (no Phase B)"; exit 1; fi
log "Phase B will start from: $CK"
mkdir -p "$CK/so101-sim-demos"; cp "$STATS" "$CK/so101-sim-demos/" 2>/dev/null || true

# ---- 3. clean ray state (proc-verified, excluding self) ----
cd /data08/henryg/pai/RLinf
.venv/bin/ray stop --force >/dev/null 2>&1 || true
for p in $(pgrep -f 'ray::|raylet|gcs_server|train_embodied_agent'); do
  [ "$p" = "$$" ] && continue
  exe=$(readlink /proc/$p/exe 2>/dev/null)
  case "$exe" in */python*|*raylet*|*gcs_server*)
    st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
done
rm -rf /tmp/ray/session_* 2>/dev/null
sleep 10
log "ray state cleaned"

# ---- 4. disk check ----
FREE_GB=$(df --output=avail -BG /data08 | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 100 ]; then log "FATAL: <100G disk free; aborting"; exit 3; fi

# ---- 5. launch Phase B (normal lr) ----
export REPO_PATH=/data08/henryg/pai/RLinf
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH="$REPO_PATH"
export HYDRA_FULL_ERROR=1
export VK_ICD_FILENAMES=$REPO_PATH/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$REPO_PATH/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99

setsid .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
  --config-name so101_ppo_openpi_pi05 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_ppo_v7b \
  actor.model.model_path="$CK" \
  rollout.model.model_path="$CK" \
  actor.model.openpi.config_name=pi05_so101_sim \
  actor.model.openpi_data.norm_stats_path="$STATS" \
  env.train.total_num_envs=128 \
  env.eval.total_num_envs=128 \
  actor.global_batch_size=2048 \
  > "$BLOG" 2>&1 </dev/null &
BPID=$!
log "Phase B launched (wrapper pid $BPID)"

# ---- 6. startup deadline: full first step within 30 min ----
D=0
until grep -q 'success_once=' "$BLOG" 2>/dev/null || grep -qE 'EXIT=' "$BLOG" 2>/dev/null; do
  sleep 60; D=$((D+60))
  [ $D -ge 1800 ] && { log "PHASE B STARTUP TIMEOUT (no first step in 30min)"; break; }
done
if grep -q 'success_once=' "$BLOG" 2>/dev/null; then
  log "Phase B first step OK: $(grep -oE 'success_once=[0-9.]+' "$BLOG" | head -1)"
fi

# ---- 7. periodic status heartbeat every 30 min ----
while true; do
  if grep -qE 'EXIT=' "$BLOG" 2>/dev/null; then
    log "Phase B ENDED: $(grep -oE 'EXIT=[0-9]+' "$BLOG" | tail -1) at epoch $(grep -c 'success_once=' "$BLOG")"
    break
  fi
  train_alive || { log "Phase B process gone at epoch $(grep -c 'success_once=' "$BLOG")"; break; }
  EP=$(grep -c 'success_once=' "$BLOG" 2>/dev/null || echo 0)
  LAST=$(grep -oE 'success_once=[0-9.]+' "$BLOG" | tail -1)
  BEST=$(grep -oE 'success_once=[0-9.]+' "$BLOG" | sed 's/success_once=//' | sort -g | tail -1)
  log "heartbeat: epoch $EP, last $LAST, best success_once=$BEST"
  sleep 1800
done
log "orchestrator done"
