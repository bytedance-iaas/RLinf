#!/bin/bash
# STATUS: REFUTED — 试过，结论是不行。留着是为了别人不再走一遍。 纯自蒸馏（真机+仿真混合 SFT）——策略越练越窄，掉 53 点
# v5 (USER-APPROVED 2026-08-11): real+sim mixed SFT from SFT-8000.
#  S1 merge real(87)+sim-v4(420) -> so101-mix-v5
#  S2 fresh norm_stats (new lineage from SFT-8000)
#  S3 SFT so101_sft_v5 (lr 2.5e-5, 4000 steps, save EVERY 250), preflighted
#  S4 adaptive gate: screen ALL 16 ckpts on seed 777; top-3 confirm on 888;
#     best -> verify 909 + y-bands 606
#  S5 hygiene: gate>=0.15 -> keep best only
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
STATUS=$SCRATCH/v5.status
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
[ "$FREE_GB" -lt 250 ] && { log "ABORT: disk <250G (16 ckpts x ~32G planned)"; exit 3; }
log "v5 pipeline started (pid $$)"

# ---- S1: merge ----
timeout 14400 .venv/bin/python "$SCRATCH/merge_datasets.py" > "$SCRATCH/merge_v5.out" 2>&1
grep -q 'MERGE DONE' "$SCRATCH/merge_v5.out" || { log "S1 GATE FAIL: merge crashed"; tail -3 "$SCRATCH/merge_v5.out" >> "$STATUS"; exit 1; }
NREAL=$(grep -oE 'real=[0-9]+' "$SCRATCH/merge_v5.out" | cut -d= -f2)
NSIM=$(grep -oE 'sim=[0-9]+' "$SCRATCH/merge_v5.out" | cut -d= -f2)
log "S1 MERGE: real=$NREAL sim=$NSIM"
[ "${NREAL:-0}" -ge 80 ] && [ "${NSIM:-0}" -ge 350 ] || { log "S1 GATE FAIL: counts real=$NREAL sim=$NSIM (need >=80/>=350)"; exit 1; }

# ---- S2: fresh stats ----
timeout 5400 .venv/bin/python -m toolkits.lerobot.calculate_norm_stats \
  --config-name pi05_so101_v5 --repo-id so101-mix-v5 > "$SCRATCH/norm_v5.out" 2>&1
[ -f assets/pi05_so101_v5/so101-mix-v5/norm_stats.json ] || { log "S2 GATE FAIL: norm stats"; exit 1; }
log "S2 norm stats OK"

# ---- S3: SFT ----
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v5 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v5 > "$SCRATCH/preflight_v5.out" 2>&1
grep -q 'PREFLIGHT OK' "$SCRATCH/preflight_v5.out" || { log "S3 PREFLIGHT FAIL"; tail -5 "$SCRATCH/preflight_v5.out" >> "$STATUS"; exit 1; }
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
  --config-name so101_sft_v5 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v5 \
  > "$SCRATCH/sft_v5.out" 2>&1
log "S3 SFT exit=$?"
CKS=$(ls -d /data08/henryg/pai/results/so101_sft_v5/*/checkpoints/global_step_* 2>/dev/null | sort -V)
[ -n "$CKS" ] || { log "S3 GATE FAIL: no ckpts"; exit 1; }
for CK in $CKS; do
  mkdir -p "$CK/so101-mix-v5"
  cp assets/pi05_so101_v5/so101-mix-v5/norm_stats.json "$CK/so101-mix-v5/" 2>/dev/null || true
done

# ---- S4: adaptive gate ----
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
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v5 \
      rollout.model.model_path="$1" \
      rollout.model.openpi.config_name=pi05_so101_v5 \
      rollout.model.openpi_data.norm_stats_path=/data08/henryg/pai/RLinf/assets/pi05_so101_v5/so101-mix-v5/norm_stats.json \
      env.eval.total_num_envs=128 \
      env.eval.seed=$2 \
      > "$SCRATCH/eval_v5_$3_t$TRY.out" 2>&1
    local EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v5_$3_t$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
    log "eval $3 try$TRY empty, retry"
  done
  return 0
}
# screen all ckpts on seed 777
SCREEN=$SCRATCH/v5_screen.txt; : > "$SCREEN"
for CK in $CKS; do
  E=$(run_eval "$CK" 777 $(basename $CK)_s777)
  log "screen $(basename $CK) s777: ${E:-FAIL}"
  [ -n "${E:-}" ] && echo "$E $CK" >> "$SCREEN"
done
# top-3 confirm on 888
BESTCK=""; BESTAVG=0
while read -r E1 CK; do
  E2=$(run_eval "$CK" 888 $(basename $CK)_s888)
  log "confirm $(basename $CK) s888: ${E2:-FAIL}"
  [ -n "${E2:-}" ] || continue
  AVG=$(awk -v a="$E1" -v b="$E2" 'BEGIN{printf "%.4f",(a+b)/2}')
  log "confirm avg $(basename $CK): $AVG"
  if awk -v a="$AVG" -v b="$BESTAVG" 'BEGIN{exit !(a>b)}'; then BESTAVG=$AVG; BESTCK=$CK; fi
done < <(sort -gr "$SCREEN" | head -3)
[ -n "$BESTCK" ] || { log "S4 GATE FAIL: no candidate"; exit 1; }
echo "$BESTCK" > "$SCRATCH/v5_best.ck"
V=$(run_eval "$BESTCK" 909 v5_s909); log "v5 VERIFY s909: ${V:-FAIL}"
B0=$(run_eval "$BESTCK" 606 band_right "0,1,0,0.33");   log "v5 y-band right: ${B0:-FAIL}"
B1=$(run_eval "$BESTCK" 606 band_mid   "0,1,0.33,0.66"); log "v5 y-band middle: ${B1:-FAIL}"
B2=$(run_eval "$BESTCK" 606 band_left  "0,1,0.66,1");   log "v5 y-band left: ${B2:-FAIL}"
log "V5 GATE RESULT: best=$BESTCK gate=$BESTAVG verify=${V:-?} bands=[${B0:-?} ${B1:-?} ${B2:-?}]"

# ---- S5: hygiene ----
if awk -v a="$BESTAVG" 'BEGIN{exit !(a>=0.15)}'; then
  for CK in $CKS; do
    [ "$CK" = "$BESTCK" ] || { rm -rf "$CK"; log "S5 deleted $(basename $CK)"; }
  done
else
  log "S5: gate <0.15 — keeping ALL v5 ckpts for diagnosis"
fi
log "V5 PIPELINE DONE"
