#!/bin/bash
# v8 PIPELINE — the target configuration (user, 2026-08-12: "V8 是目标").
# Full fidelity (640x480, 30 Hz, measured geometry, 8 g cube, homing success)
# with ONLY the red-cube spawn narrowed to the pp-era 6x8 cm box.
#
#   S0 wait for demo generation (gen_demos_narrow_box.sh, CPU)
#   S1 convert -> so101-sim-demos-v8 (fps 30, 480x640)
#   S2 stats: REUSE v4 (continuing the v4 lineage — never recompute mid-lineage)
#   S3 BASELINE: eval the warm start (v4_step_1000) IN THE LEGACY BOX — the floor to beat
#   S4 SFT from v4_step_1000 (lr 2.5e-5, 4000 steps, save every 250)
#   S5 gate every ckpt in-box (777) -> top-3 on 888 -> verify on never-used 1313/1414
#      + one FULL-BOARD reference eval (so the narrow number is never mistaken for the wide one)
#   S6 hygiene: keep best only
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/v8.status
WARM=/data08/henryg/pai/results/so101_sft_v4/so101_sft_openpi_pi05/checkpoints/global_step_1000
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

clean(){
  .venv/bin/ray stop --force >/dev/null 2>&1 || true
  for p in $(pgrep -f 'ray::|raylet|gcs_server|eval_embodied_agent|train_vla_sft'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in */python*|*raylet*|*gcs_server*) st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null;; esac
  done
  rm -rf /tmp/ray/session_* 2>/dev/null
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
  SHM=$(df -m /dev/shm | tail -1 | awk '{print $4}')
  [ "${SHM:-0}" -lt 1024 ] && mount -o remount,size=16G /dev/shm 2>/dev/null
}

# ---- S0: wait for generation ----
D=0
until grep -q 'ready for conversion' "$STATUS" 2>/dev/null; do
  sleep 120; D=$((D+120))
  grep -q 'GATE FAIL' "$STATUS" 2>/dev/null && { log "S0: generation gate failed — stopping"; exit 1; }
  [ $D -ge 21600 ] && { log "S0 TIMEOUT (6h)"; exit 1; }
done

# ---- S1: convert ----
timeout 7200 .venv/bin/python "$SCRATCH/convert_narrow_box.py" > "$SCRATCH/convert_v8.out" 2>&1
grep -q 'DONE:' "$SCRATCH/convert_v8.out" || { log "S1 FAIL: conversion"; tail -3 "$SCRATCH/convert_v8.out" >> "$STATUS"; exit 1; }
log "S1 $(grep -E '^DONE|^length' "$SCRATCH/convert_v8.out" | tr '\n' ' ')"
NEP=$(grep -oE 'DONE: [0-9]+' "$SCRATCH/convert_v8.out" | grep -oE '[0-9]+')
[ "${NEP:-0}" -ge 100 ] || { log "S1 GATE FAIL: only $NEP episodes"; exit 1; }
AREA=48   # cm^2 of the legacy box
SPACING=$(awk -v n="$NEP" -v a="$AREA" 'BEGIN{printf "%.2f", sqrt(a/n)}')
log "S1 demo spacing: ${SPACING} cm (pp-era 0.51 cm; grasp tolerance ~0.7 cm)"

run_eval(){  # ckpt seed tag cfgname spawnmode
  local TRY
  for TRY in 1 2; do
    clean
    SO101_SPAWN_MODE="$5" timeout 2400 .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
      --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v8 \
      rollout.model.model_path="$1" \
      rollout.model.openpi.config_name="$4" \
      rollout.model.openpi_data.norm_stats_path=$STATS \
      env.eval.total_num_envs=128 env.eval.seed=$2 \
      > "$SCRATCH/eval_v8_$3_t$TRY.out" 2>&1
    local EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v8_$3_t$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
  done
  return 0
}

# ---- S2/S3: baseline of the warm start inside the legacy box ----
B=$(run_eval "$WARM" 777 baseline_warm_inbox pi05_so101_v4 legacy)
log "S3 BASELINE: warm start (v4_step_1000) in the legacy box = ${B:-FAIL}  [full board it is 0.125]"

# ---- S4: SFT ----
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v8 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v8 > "$SCRATCH/preflight_v8.out" 2>&1
grep -q 'PREFLIGHT OK' "$SCRATCH/preflight_v8.out" || { log "S4 PREFLIGHT FAIL"; tail -4 "$SCRATCH/preflight_v8.out" >> "$STATUS"; exit 1; }
clean; sleep 5
log "S4 SFT launching (warm=v4_step_1000, v4 stats, lr 2.5e-5, save 250)"
timeout 21600 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v8 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v8 \
  > "$SCRATCH/sft_v8.out" 2>&1
log "S4 SFT exit=$?"
CKS=$(ls -d /data08/henryg/pai/results/so101_sft_v8/*/checkpoints/global_step_* 2>/dev/null | sort -V)
[ -n "$CKS" ] || { log "S4 GATE FAIL: no ckpts"; exit 1; }
for CK in $CKS; do mkdir -p "$CK/so101-sim-demos-v4"; cp $STATS "$CK/so101-sim-demos-v4/" 2>/dev/null || true; done

# ---- S5: gate in-box, verify on unused seeds, full-board reference ----
export EMBODIED_PATH=$PWD/examples/embodiment
SCREEN=$SCRATCH/v8_screen.txt; : > "$SCREEN"
for CK in $CKS; do
  E=$(run_eval "$CK" 777 "$(basename $CK)_s777" pi05_so101_v8 legacy)
  log "screen(in-box) $(basename $CK) s777: ${E:-FAIL}"
  [ -n "${E:-}" ] && echo "$E $CK" >> "$SCREEN"
done
BESTCK=""; BESTAVG=0
while read -r E1 CK; do
  E2=$(run_eval "$CK" 888 "$(basename $CK)_s888" pi05_so101_v8 legacy)
  log "confirm(in-box) $(basename $CK) s888: ${E2:-FAIL}"
  [ -n "${E2:-}" ] || continue
  AVG=$(awk -v a="$E1" -v b="$E2" 'BEGIN{printf "%.4f",(a+b)/2}')
  log "confirm avg $(basename $CK): $AVG"
  if awk -v a="$AVG" -v b="$BESTAVG" 'BEGIN{exit !(a>b)}'; then BESTAVG=$AVG; BESTCK=$CK; fi
done < <(sort -gr "$SCREEN" | head -3)
[ -n "$BESTCK" ] || { log "S5 GATE FAIL"; exit 1; }
echo "$BESTCK" > "$SCRATCH/v8_best.ck"
V1=$(run_eval "$BESTCK" 1313 verify_1313 pi05_so101_v8 legacy); log "VERIFY in-box s1313: ${V1:-FAIL}"
V2=$(run_eval "$BESTCK" 1414 verify_1414 pi05_so101_v8 legacy); log "VERIFY in-box s1414: ${V2:-FAIL}"
FB=$(run_eval "$BESTCK" 1313 fullboard_1313 pi05_so101_v8 "");  log "FULL-BOARD reference: ${FB:-FAIL}"
log "V8 RESULT: ckpt=$(basename "$BESTCK") gate=$BESTAVG verify=${V1:-?}/${V2:-?} fullboard=${FB:-?} | baseline warm-in-box=${B:-?} , pp-era floor was 0.469"

for C in $CKS; do [ "$C" = "$BESTCK" ] || rm -rf "$C"; done
log "v8 done; disk free $(df -h /data08 | tail -1 | awk '{print $4}')"
