#!/bin/bash
# STATUS: ACTIVE — 当前流程在用。 阶段 B 的整段编排（S0 探针 -> S5 检查点清理）
# v4 pipeline (USER-APPROVED parameters, 2026-08-10):
#  S0 planner probe N=12 on the changed env (30Hz/640x480/8g) — gate >=6
#  S1 stratified demos: 16 cells x 45 attempts, 8 CPU workers, seeds 80000+
#  S2 convert -> so101-sim-demos-v4 (480x640) + fresh norm_stats (new lineage)
#  S3 SFT so101_sft_v4 (warm pp6b_1000, lr 2.5e-5, 4000 steps, save 1000), preflighted
#  S4 gate 4 ckpts (777/888) + verify (909) + y-bands (606)
#  S5 checkpoint hygiene: keep best v4 ckpt only IF gate >= 0.15, else keep all
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/v4.status
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

export REPO_PATH="$PWD" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo

FREE_GB=$(df --output=avail -BG /data08 | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -lt 200 ] && { log "ABORT: disk <200G"; exit 3; }
log "v4 pipeline started (pid $$)"

# ---- S0: probe on changed env ----
rm -rf /data08/henryg/pai/data/v4_probe
timeout 7200 .venv/bin/python "$SCRATCH/gen_planner_demos.py" --num 12 --seed0 79000 \
  --out /data08/henryg/pai/data/v4_probe > "$SCRATCH/v4_probe.out" 2>&1
POK=$(grep -oE 'TOTAL success [0-9]+' "$SCRATCH/v4_probe.out" | grep -oE '[0-9]+$' || echo 0)
log "S0 probe: $POK/12"
[ "${POK:-0}" -ge 6 ] || { log "S0 GATE FAIL: probe <6/12 — generator needs rework on 30Hz env"; exit 1; }
# length stats for the budget re-check
PLEN=$(.venv/bin/python - <<'PY'
import h5py, json, glob, numpy as np
h5 = sorted(glob.glob("/data08/henryg/pai/data/v4_probe/*.h5"))[-1]
meta = json.load(open(h5.replace(".h5", ".json")))
ok = [e["episode_id"] for e in meta["episodes"] if e["success"]]
f = h5py.File(h5, "r")
lens = [f[f"traj_{i}"]["actions"].shape[0] for i in ok]
print(int(np.median(lens)))
PY
)
log "S0 demo median length: $PLEN (budget 640; need <=530)"
[ "${PLEN:-999}" -le 530 ] || { log "S0 GATE FAIL: median too long for budget 640"; exit 1; }

# ---- S1: stratified generation, 8 workers x 2 cells ----
worker(){
  local W=$1; shift
  for SPEC in "$@"; do
    local FRAC=${SPEC%%;*}; local SEED=${SPEC##*;}
    local TAG=$(echo "$FRAC" | tr ',.' '_-')
    rm -rf /data08/henryg/pai/data/v4_demos_cell_$TAG
    SO101_SPAWN_FRAC=$FRAC timeout 14400 .venv/bin/python "$SCRATCH/gen_planner_demos.py" \
      --num 45 --seed0 $SEED --out /data08/henryg/pai/data/v4_demos_cell_$TAG \
      > "$SCRATCH/v4_gen_${TAG}.out" 2>&1
    local N=$(grep -oE 'TOTAL success [0-9]+' "$SCRATCH/v4_gen_${TAG}.out" | grep -oE '[0-9]+$' || echo 0)
    log "cell $FRAC: $N/45 (worker $W)"
  done
}
CELLS=(); SEED=80000
for XI in 0 1 2 3; do
  for YI in 0 1 2 3; do
    X0=$(awk -v i=$XI 'BEGIN{printf "%.2f", i*0.25}'); X1=$(awk -v i=$XI 'BEGIN{printf "%.2f", (i+1)*0.25}')
    Y0=$(awk -v i=$YI 'BEGIN{printf "%.2f", i*0.25}'); Y1=$(awk -v i=$YI 'BEGIN{printf "%.2f", (i+1)*0.25}')
    CELLS+=("$X0,$X1,$Y0,$Y1;$SEED"); SEED=$((SEED+100))
  done
done
for W in 0 1 2 3 4 5 6 7; do
  worker $W "${CELLS[@]:$((W*2)):2}" &
  eval "P$W=$!"
done
wait $P0 $P1 $P2 $P3 $P4 $P5 $P6 $P7
TOTAL=$(grep -hoE 'TOTAL success [0-9]+' "$SCRATCH"/v4_gen_*.out | grep -oE '[0-9]+' | awk '{s+=$1} END{print s}')
log "S1 DONE: $TOTAL/720 successes"
[ "${TOTAL:-0}" -ge 250 ] || { log "S1 GATE FAIL: <250 demos"; exit 1; }

# ---- S2: convert + fresh stats ----
timeout 14400 .venv/bin/python "$SCRATCH/convert_fullboard.py" > "$SCRATCH/convert_v4.out" 2>&1
grep -q 'DONE:' "$SCRATCH/convert_v4.out" || { log "S2 GATE FAIL: conversion"; tail -3 "$SCRATCH/convert_v4.out" >> "$STATUS"; exit 1; }
log "S2 convert: $(grep -E '^DONE|^length' "$SCRATCH/convert_v4.out" | tr '\n' ' ')"
timeout 5400 .venv/bin/python -m toolkits.lerobot.calculate_norm_stats \
  --config-name pi05_so101_v4 --repo-id so101-sim-demos-v4 > "$SCRATCH/norm_v4.out" 2>&1
[ -f assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json ] || { log "S2 GATE FAIL: norm stats"; exit 1; }
log "S2 norm stats OK"

# ---- S3: SFT ----
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v4 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v4 > "$SCRATCH/preflight_v4.out" 2>&1
grep -q 'PREFLIGHT OK' "$SCRATCH/preflight_v4.out" || { log "S3 PREFLIGHT FAIL"; tail -5 "$SCRATCH/preflight_v4.out" >> "$STATUS"; exit 1; }
.venv/bin/ray stop --force >/dev/null 2>&1 || true
for p in $(pgrep -f 'ray::|raylet|gcs_server'); do
  [ "$p" = "$$" ] && continue
  exe=$(readlink /proc/$p/exe 2>/dev/null)
  case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
done
rm -rf /tmp/ray/session_* 2>/dev/null
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
log "S3 SFT launching"
timeout 21600 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v4 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v4 \
  > "$SCRATCH/sft_v4.out" 2>&1
log "S3 SFT exit=$?"
CKS=$(ls -d /data08/henryg/pai/results/so101_sft_v4/*/checkpoints/global_step_* 2>/dev/null | sort -V)
[ -n "$CKS" ] || { log "S3 GATE FAIL: no ckpts"; exit 1; }
for CK in $CKS; do
  mkdir -p "$CK/so101-sim-demos-v4"
  cp assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json "$CK/so101-sim-demos-v4/" 2>/dev/null || true
done

# ---- S4: gate + verify + bands ----
export EMBODIED_PATH=$PWD/examples/embodiment
clean(){
  .venv/bin/ray stop --force >/dev/null 2>&1 || true
  for p in $(pgrep -f 'ray::|raylet|gcs_server|eval_embodied_agent'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
  done
  rm -rf /tmp/ray/session_* 2>/dev/null
}
run_eval(){
  local SF="${4:-}"
  local TRY
  for TRY in 1 2; do
    clean
    SO101_SPAWN_FRAC="$SF" timeout 1500 .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
      --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v4 \
      rollout.model.model_path="$1" \
      rollout.model.openpi.config_name=pi05_so101_v4 \
      rollout.model.openpi_data.norm_stats_path=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
      env.eval.total_num_envs=128 \
      env.eval.seed=$2 \
      > "$SCRATCH/eval_v4_$3_t$TRY.out" 2>&1
    local EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v4_$3_t$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
    log "eval $3 try$TRY empty, retry"
  done
  return 0
}
BESTCK=""; BESTAVG=0
for CK in $(ls -d /data08/henryg/pai/results/so101_sft_v4/*/checkpoints/global_step_{1000,2000,3000,4000} 2>/dev/null); do
  E1=$(run_eval "$CK" 777 $(basename $CK)_s777); log "gateV4 $(basename $CK) s777: ${E1:-FAIL}"
  E2=$(run_eval "$CK" 888 $(basename $CK)_s888); log "gateV4 $(basename $CK) s888: ${E2:-FAIL}"
  [ -n "${E1:-}" ] && [ -n "${E2:-}" ] || continue
  AVG=$(awk -v a="$E1" -v b="$E2" 'BEGIN{printf "%.4f",(a+b)/2}')
  log "gateV4 avg $(basename $CK): $AVG"
  if awk -v a="$AVG" -v b="$BESTAVG" 'BEGIN{exit !(a>b)}'; then BESTAVG=$AVG; BESTCK=$CK; fi
done
[ -n "$BESTCK" ] || { log "S4 GATE FAIL: no candidate"; exit 1; }
echo "$BESTCK" > "$SCRATCH/v4_best.ck"
V=$(run_eval "$BESTCK" 909 v4_s909); log "v4 VERIFY s909: ${V:-FAIL}"
B0=$(run_eval "$BESTCK" 606 band_right "0,1,0,0.33");   log "v4 y-band right: ${B0:-FAIL}"
B1=$(run_eval "$BESTCK" 606 band_mid   "0,1,0.33,0.66"); log "v4 y-band middle: ${B1:-FAIL}"
B2=$(run_eval "$BESTCK" 606 band_left  "0,1,0.66,1");   log "v4 y-band left: ${B2:-FAIL}"
log "V4 GATE RESULT: best=$BESTCK gate=$BESTAVG verify=${V:-?} bands=[${B0:-?} ${B1:-?} ${B2:-?}]"

# ---- S5: checkpoint hygiene (user directive: delete obsolete promptly) ----
if awk -v a="$BESTAVG" 'BEGIN{exit !(a>=0.15)}'; then
  for CK in $(ls -d /data08/henryg/pai/results/so101_sft_v4/*/checkpoints/global_step_* 2>/dev/null); do
    [ "$CK" = "$BESTCK" ] || { rm -rf "$CK"; log "S5 deleted non-best $(basename $CK)"; }
  done
else
  log "S5: gate <0.15 — keeping ALL v4 ckpts for diagnosis"
fi
log "V4 PIPELINE DONE"
