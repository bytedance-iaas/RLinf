#!/bin/bash
# STATUS: ACTIVE — 当前流程在用。 阶段 G 第二轮：CPU 轨，建留出集数据
# TRACK 1 (CPU only) — build the held-out co-training dataset.
#
# Why this run exists: v14 put all 87 real episodes into training and then scored
# the offline sim2real check on episodes 0-3 of that same set. 0.84 is therefore
# a training-set number and says nothing about unseen real observations. Here
# real episodes 0-69 train and 70-86 are never shown to the model, so the gate
# measures generalisation.
#
# Runs on CPU only, so it shares the machine with the GPU track without
# contending for accelerators. Encoding measured at 0.64 episodes/min on the
# previous run, so 210 appended episodes is ~5.5 h -> 8 h budget.
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/night.status
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] [CPU] $*" >> "$STATUS"; }
export REPO_PATH="$PWD" PYTHONPATH="$PWD" HF_LEROBOT_HOME=/data08/henryg/pai/data
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

FREE=$(df --output=avail -BG /data08 | tail -1 | tr -dc '0-9')
[ "$FREE" -lt 150 ] && { log "ABORT: disk <150G"; exit 3; }
log "building held-out co-training dataset (sim 1292 + real[0:70] x3; real[70:87] held out)"
timeout 28800 .venv/bin/python tools_so101_session/convert_cotrain_heldout.py \
  > "$SCRATCH/convert_v15.out" 2>&1
grep -qa '^DONE:' "$SCRATCH/convert_v15.out" || { log "DATASET FAIL"; tail -4 "$SCRATCH/convert_v15.out" >> "$STATUS"; exit 1; }
log "$(grep -a '^DONE:' "$SCRATCH/convert_v15.out" | tail -1)"
log "DATASET READY"
