#!/bin/bash
# STATUS: SUPERSEDED — 早期任务规格，已被主线取代。别用来复现。 复现论文基准数字，该任务线已停
# Reproduce the published pi0.5 SFT number on the paper's own benchmark.
#   Paper (πRL, arXiv 2510.25889): ManiSkill3 25-task multi-task SFT = 40.1%
#   (RL then takes it to 90.9% Flow-SDE / 89.7% Flow-Noise)
# Runs ONLY after v10's gate is finished -- user chose option B (no queue jump).
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/repro.status
V10=$SCRATCH/v10.status
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

export REPO_PATH="$PWD" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1
export EMBODIED_PATH=$PWD/examples/embodiment
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAY_local_fs_capacity_threshold=0.99
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

clean(){
  .venv/bin/ray stop --force >/dev/null 2>&1 || true
  for p in $(pgrep -f 'ray::|raylet|gcs_server|eval_embodied_agent'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
  done
  rm -rf /tmp/ray/session_* 2>/dev/null
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
  sleep 8
}

# Run immediately: the box is idle (v4b finished 18:53, control 19:00).
# The previous attempt died because its wait-sentinel matched a STALE failure
# marker left by an earlier attempt of the same upstream stage, so it started
# while the v4b gate still owned the GPUs and got killed by that gate's clean().
# Sentinels must be unique per attempt -- a re-queued stage inherits its own
# old failure marker otherwise.
log "reproduction eval starting (official ckpt, official protocol; paper SFT = 0.401)"

EV=""
for TRY in 1 2 3; do
  clean
  timeout 5400 .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path /data08/henryg/pai/RLinf/evaluations/maniskill/ \
    --config-name maniskill_pi05_official_eval \
    runner.logger.log_path=/data08/henryg/pai/results/repro_official \
    > "$SCRATCH/repro_eval_r$TRY.out" 2>&1
  EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/repro_eval_r$TRY.out" | tail -1 | cut -d= -f2)
  [ -n "$EV" ] && break
  log "repro try$TRY produced no number, retrying"
done
if [ -z "$EV" ]; then
  log "REPRO FAIL: no success number after 3 tries"
  tail -15 "$SCRATCH/repro_eval_r3.out" >> "$STATUS"
  exit 1
fi
DELTA=$(awk -v a="$EV" 'BEGIN{printf "%+.1f", (a-0.401)*100}')
log "REPRO RESULT: ours=$EV  paper=0.401  delta=${DELTA} pts"
log "verdict: $(awk -v a="$EV" 'BEGIN{d=(a-0.401); if(d<0)d=-d; print (d<=0.05)?"REPRODUCED (within 5 pts) -- our stack is trustworthy":"MISMATCH -- investigate before trusting any of our own numbers"}')"
