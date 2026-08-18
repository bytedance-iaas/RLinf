#!/bin/bash
# STATUS: SUPERSEDED — 早期任务规格，已被主线取代。别用来复现。 v6 PPO：起点带噪成功率 0.5%，从未起来
# FINAL RUN: official πRL recipe on the true task, start = v4_step_1000 (12.5%).
# In-launcher auto-stop (harvest law): R1 erosion past peak, R2 dead-below-floor
# with grace, R3 epoch cap.
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/v6.status
LOG=$SCRATCH/rl_v6.out
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

FREE_GB=$(df --output=avail -BG /data08 | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -lt 200 ] && { log "ABORT: disk <200G"; exit 3; }

.venv/bin/ray stop --force >/dev/null 2>&1 || true
for p in $(pgrep -f 'ray::|raylet|gcs_server'); do
  [ "$p" = "$$" ] && continue
  exe=$(readlink /proc/$p/exe 2>/dev/null)
  case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
done
rm -rf /tmp/ray/session_* 2>/dev/null
# /dev/shm hygiene + capacity gate: NCCL builds ~7MB shm segments per comm; the
# container default is 64MB and crashed runs leave cuda.shm.* behind (2882 found
# on 2026-08-12), which ratchets every relaunch into ncclSystemError.
find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
SHM_MB=$(df -m /dev/shm | tail -1 | awk '{print $4}')
if [ "${SHM_MB:-0}" -lt 1024 ]; then
  mount -o remount,size=16G /dev/shm 2>/dev/null || true
  SHM_MB=$(df -m /dev/shm | tail -1 | awk '{print $4}')
fi
log "/dev/shm free: ${SHM_MB}MB"
[ "${SHM_MB:-0}" -lt 512 ] && { log "ABORT: /dev/shm <512MB free — NCCL will fail"; exit 5; }

log "v6 (official recipe) launching from v4_step_1000"
setsid .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
  --config-name so101_ppo_v6_official \
  runner.logger.log_path=/data08/henryg/pai/results/so101_ppo_v6 \
  > "$LOG" 2>&1 </dev/null &
TRAINER=$!

D=0
until grep -q 'success_once=' "$LOG" 2>/dev/null; do
  sleep 60; D=$((D+60))
  [ $D -ge 2400 ] && { log "STARTUP TIMEOUT — check $LOG"; exit 4; }
done
log "first step: $(grep -oE 'success_once=[0-9.]+' "$LOG" | head -1)"

stop_trainer(){
  for p in $(pgrep -f 'train_embodied_agent.py|ray::|raylet|gcs_server'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
  done
  .venv/bin/ray stop --force >/dev/null 2>&1 || true
}

while true; do
  sleep 600
  if ! kill -0 "$TRAINER" 2>/dev/null; then log "trainer exited on its own"; break; fi
  SERIES=$(.venv/bin/python tools_so101_session/parse_eval_series.py "$LOG")
  EPOCHS=$(echo "$SERIES" | grep -oE 'epochs=[0-9]+' | cut -d= -f2)
  EVALS=$(echo "$SERIES" | grep -vE 'epochs=')
  N=$(echo "$EVALS" | grep -c . || echo 0)
  if [ "$N" -ge 1 ]; then
    BEST=$(echo "$EVALS" | sort -g | tail -1)
    LAST=$(echo "$EVALS" | tail -1)
    log "heartbeat: iters=$EPOCHS evals=$N best=$BEST last=$LAST"
    R1=$(.venv/bin/python - "$BEST" "$LAST" "$N" <<'PY'
import sys
best, last, n = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
if n >= 3 and last < best - 0.15:
    print("R1 erosion past peak")
PY
)
    R2=$(echo "$EVALS" | tail -3 | awk -v n="$N" 'BEGIN{bad=0;c=0} {c++; if ($1<0.05) bad++} END{if (n>=6 && c>=3 && bad==3) print "R2 dead below 5% after grace"}')
    [ -n "$R1" ] && { log "AUTO-STOP $R1 (best=$BEST)"; stop_trainer; break; }
    [ -n "$R2" ] && { log "AUTO-STOP $R2"; stop_trainer; break; }
  else
    log "heartbeat: iters=$EPOCHS (no eval yet)"
  fi
  [ "${EPOCHS:-0}" -ge 300 ] && { log "AUTO-STOP R3 iter cap 300"; stop_trainer; break; }
done
log "v6 done. eval series: $(.venv/bin/python tools_so101_session/parse_eval_series.py "$LOG" | grep -v epochs | tr '\n' ' ')"
