#!/bin/bash
# STATUS: SUPERSEDED — 早期任务规格，已被主线取代。别用来复现。 阶段 C 生成的旧启动器，现直接调 gen_planner_demos.py
# v8 demo generation — USER-DIRECTED (2026-08-12): keep FULL fidelity
# (640x480 cameras, 30 Hz, true board geometry, 8 g cube, homing success) and
# restrict ONLY the red-cube spawn to the pp-era 6x8 cm box (SO101_SPAWN_MODE=legacy).
# Target: ~175 successes over 48 cm^2 => 0.52 cm demo spacing = pp-era density,
# which is the regime where the BC floor reached 46.9% and the amplifiers worked.
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/v8.status
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

export REPO_PATH="$PWD" PYTHONPATH="$PWD"
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl
export SO101_SPAWN_MODE=legacy      # the ONLY thing narrowed; everything else stays true-task

log "v8 legacy-box generation started (pid $$) — 8 workers x 32 attempts"
worker(){
  local W=$1 SEED=$2
  rm -rf /data08/henryg/pai/data/v8_demos_w$W
  timeout 14400 .venv/bin/python "$SCRATCH/gen_planner_demos.py" \
    --num 32 --seed0 $SEED --out /data08/henryg/pai/data/v8_demos_w$W \
    > "$SCRATCH/v8_gen_w$W.out" 2>&1
  local N=$(grep -oE 'TOTAL success [0-9]+' "$SCRATCH/v8_gen_w$W.out" | grep -oE '[0-9]+$' || echo 0)
  log "worker $W: $N/32"
}
SEED=90000
for W in 0 1 2 3 4 5 6 7; do
  worker $W $SEED &
  SEED=$((SEED+1000))
done
wait
TOTAL=$(grep -hoE 'TOTAL success [0-9]+' "$SCRATCH"/v8_gen_w*.out | grep -oE '[0-9]+' | awk '{s+=$1} END{print s}')
log "v8 generation DONE: $TOTAL/256 successes"
[ "${TOTAL:-0}" -ge 120 ] || { log "GATE FAIL: <120 demos in the legacy box"; exit 1; }

# length gate (30 Hz demos are ~1.5x longer than the 20 Hz pp ones)
LEN=$(.venv/bin/python - <<'PY'
import glob, json, h5py, numpy as np
lens = []
for h5p in glob.glob("/data08/henryg/pai/data/v8_demos_w*/**/*.h5", recursive=True):
    meta = json.load(open(h5p.replace(".h5", ".json")))
    ok = [e["episode_id"] for e in meta["episodes"] if e["success"]]
    f = h5py.File(h5p, "r")
    lens += [f[f"traj_{i}"]["actions"].shape[0] for i in ok]
    f.close()
print(int(np.median(lens)) if lens else 999)
PY
)
log "v8 demo median length: $LEN (budget 640; need <=530)"
log "v8 ready for conversion + SFT"
