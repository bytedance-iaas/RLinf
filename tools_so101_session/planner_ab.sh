#!/bin/bash
# A/B for the planner drop fix, on the two strips where the current planner is
# worst (LEFT and TOP: 33% hit rate, 5.6-7.0 cm median drop error over 298
# logged attempts). Same strips, same seeds, only the planner differs.
#   A = gen_planner_demos.py     (current, coarse refinement, 2 passes, unchecked)
#   B = gen_planner_demos_finegrid_rejected.py  (fine local grid, 4 passes, return checked)
# Runs AFTER the annulus generation finishes so the two never share the GPUs.
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
unset SO101_SPAWN_MODE

gen_done(){ grep -q 'S2 DONE:' "$STATUS"; }
DEADLINE=$(( $(date +%s) + 3*3600 ))
while ! gen_done; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "AB ABORT: generation never finished"; exit 1; }
  if ! pgrep -f 'bash .*gen_demos_annulus.sh' >/dev/null; then
    sleep 5; gen_done && break
    log "AB ABORT: generator exited without its completion marker"; exit 1
  fi
  sleep 60
done
sleep 20
log "planner A/B started (LEFT+TOP, 12 seeds each, old vs fine-refinement)"

LEFT="0.4294,0.5000,0.5142,0.9817"
TOP="0.5000,0.8410,0.9132,0.9817"
run(){  # tag script frac seed0
  SO101_SPAWN_FRAC="$3" timeout 5400 .venv/bin/python "$SCRATCH/$2" \
    --num 12 --seed0 $4 --out /data08/henryg/pai/data/ab_$1 > "$SCRATCH/ab_$1.out" 2>&1
  local N=$(grep -oE 'TOTAL success [0-9]+' "$SCRATCH/ab_$1.out" | grep -oE '[0-9]+$' || echo 0)
  log "A/B $1: ${N:-0}/12"
}
run oldleft gen_planner_demos.py    "$LEFT" 500000 &
run newleft gen_planner_demos_finegrid_rejected.py "$LEFT" 500000 &
run oldtop  gen_planner_demos.py    "$TOP"  600000 &
run newtop  gen_planner_demos_finegrid_rejected.py "$TOP"  600000 &
wait
log "A/B DONE: $(for t in oldleft newleft oldtop newtop; do printf "%s=%s " $t "$(grep -oE 'TOTAL success [0-9]+' $SCRATCH/ab_$t.out | grep -oE '[0-9]+$')"; done)"
