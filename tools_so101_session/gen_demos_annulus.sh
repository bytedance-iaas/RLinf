#!/bin/bash
# v10 S2 — planner demos in the RING-1 ANNULUS ONLY (ring1 minus the legacy box).
#
# Why only the annulus: the S1 policy-collection is heavily biased toward the
# inner (already-trained) half. Measured ring-1 success 0.516 vs in-box 0.766
# implies the OUTER half succeeds at ~2*0.516-0.766 = 0.27, so 8 seeds x 128
# envs yield ~136 outer successes = 0.59 cm spacing out there, versus 0.44 cm
# (the v8-proven density) inside. This stage buys the outer half back up to
# 0.44 cm: 48 cm^2 / 0.44^2 = 248 demos needed, minus ~136 from the policy
# => ~112 planner demos, split across the four strips in proportion to area.
#
# Strips as SO101_SPAWN_FRAC (full-board fractions; legacy box = x[0.500,0.841]
# y[0.5826,0.9132], ring1 = x[0.4294,0.9115] y[0.5142,0.9817]):
#   left  1.24 x 11.31 cm = 14.0 cm^2  -> 33 demos
#   right 1.24 x 11.31 cm = 14.0 cm^2  -> 33
#   bottom 6.00 x 1.66 cm =  9.9 cm^2  -> 23
#   top    6.00 x 1.66 cm =  9.9 cm^2  -> 23
# 2 workers per strip, 30 attempts each; at the planner's full-board rate (58%)
# that is ~35 successes per strip pair, at its in-box rate (96%) ~58.
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/v10.status
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

export REPO_PATH="$PWD" PYTHONPATH="$PWD"
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl
unset SO101_SPAWN_MODE              # full-board mode; the strips do the narrowing

# wait for S1 (collection) to finish so the two stages never share the GPUs
# NOTE: the completion check must be re-done AFTER observing that the producer
# is gone — the producer writes its sentinel and exits microseconds apart, and
# checking liveness alone turns a normal finish into a false "died" (it did,
# 2026-08-13 07:52:40). Also: never put the sentinel string inside a log
# message, or later greps match our own log line.
DEADLINE=$(( $(date +%s) + 4*3600 ))
done_yet(){ grep -q 'S1 DONE' "$STATUS"; }
while ! done_yet; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "S2 ABORT: stage-1 never finished"; exit 1; }
  if ! pgrep -f 'bash .*collect_policy_successes.sh' >/dev/null; then
    sleep 5
    done_yet && break
    log "S2 ABORT: collector exited without its completion marker"; exit 1
  fi
  sleep 60
done
sleep 30
log "S2 annulus generation started (4 strips x 2 workers x 30 attempts)"

worker(){  # id frac seed
  local W=$1 FRAC=$2 SEED=$3
  rm -rf /data08/henryg/pai/data/v10_demos_w$W
  SO101_SPAWN_FRAC="$FRAC" timeout 14400 .venv/bin/python "$SCRATCH/gen_planner_demos.py" \
    --num 30 --seed0 $SEED --out /data08/henryg/pai/data/v10_demos_w$W \
    > "$SCRATCH/v10_gen_w$W.out" 2>&1
  local N=$(grep -oE 'TOTAL success [0-9]+' "$SCRATCH/v10_gen_w$W.out" | grep -oE '[0-9]+$' || echo 0)
  log "gen worker $W ($FRAC): ${N:-0}/30"
}

LEFT="0.4294,0.5000,0.5142,0.9817"
RIGHT="0.8410,0.9115,0.5142,0.9817"
BOTTOM="0.5000,0.8410,0.5142,0.5826"
TOP="0.5000,0.8410,0.9132,0.9817"
SEED=110000
W=0
for FRAC in "$LEFT" "$LEFT" "$RIGHT" "$RIGHT" "$BOTTOM" "$BOTTOM" "$TOP" "$TOP"; do
  worker $W "$FRAC" $SEED &
  W=$((W+1)); SEED=$((SEED+1000))
done
wait
TOTAL=$(grep -hoE 'TOTAL success [0-9]+' "$SCRATCH"/v10_gen_w*.out | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}')
log "S2 DONE: $TOTAL/240 annulus demos (need >=112 to hold 0.44 cm out there)"
[ "${TOTAL:-0}" -ge 60 ] || { log "S2 GATE FAIL: annulus generation too weak ($TOTAL) — planner may not reach the outer strips"; exit 1; }
log "S2 ready for conversion (v10 = v9's 672 eps + ring-1 policy rollouts + $TOTAL annulus demos)"
