#!/bin/bash
# CONTROLLED TEST of the single unverified link in V10_REPRODUCTION_ZH.md.
#
# Question: can the full-board sim-demo SFT stage warm-start from the REAL-DATA
# SFT checkpoint instead of the pp-era checkpoint that history actually used?
# The pp-era lineage was trained under a superseded task spec (160x120 cameras,
# 20 Hz, no homing) and cannot be reproduced with today's code, so the answer
# decides whether the reproduction doc is correct as written.
#
# Only variable: the warm start. Same dataset (so101-sim-demos-v4, 420 eps),
# same frozen v4 norm_stats, same lr 2.5e-5, same 4000 steps, same full-board
# eval protocol, same verification seed 909 that produced v4's 12.5%.
#
# Pre-registered verdict (binomial sd at p=0.125, n=128, is 2.9 pts):
#   verify >= 9%  -> VALIDATED, the doc stands
#   5-9%          -> DEGRADED, the doc must warn and stage C must be re-checked
#   < 5%          -> INVALID, the doc needs a different bridge stage
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/v4b.status
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
CKROOT=/data08/henryg/pai/results/so101_sft_v4b/so101_sft_openpi_pi05/checkpoints
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
  sleep 8
}

run_eval(){  # ckpt seed tag   (FULL BOARD: no spawn mode, no frac -- v4's protocol)
  local TRY EV
  for TRY in 1 2 3; do
    clean
    timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
      --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v4b \
      rollout.model.model_path="$1" \
      rollout.model.openpi.config_name=pi05_so101_v4 \
      rollout.model.openpi_data.norm_stats_path=$STATS \
      env.eval.total_num_envs=128 env.eval.seed=$2 \
      > "$SCRATCH/eval_v4b_$3_r$TRY.out" 2>&1
    EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v4b_$3_r$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
    log "eval $3 try$TRY produced nothing, retrying"
  done
  echo ""
}

# --- queue directly behind the v10 gate (user reordered: repro eval goes last) ---
prev_done(){ grep -qE 'V10 FINAL|S5 GATE FAIL|S4 GATE FAIL|S3 FAIL|S3 ABORT' "$SCRATCH/v10.status" 2>/dev/null; }
DEADLINE=$(( $(date +%s) + 14*3600 ))
while ! prev_done; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: upstream never finished"; exit 1; }
  if ! pgrep -f 'bash .*pipeline_region_expand.sh' >/dev/null; then
    sleep 5
    prev_done && break
    log "ABORT: upstream exited without a result marker"; exit 1
  fi
  sleep 120
done
sleep 30

FREE=$(df --output=avail -BG /data08 | tail -1 | tr -dc '0-9')
[ "$FREE" -lt 150 ] && { log "ABORT: disk <150G"; exit 3; }

# --- CONTROL: has the env drifted since v4 was measured on 2026-08-11? ---
# The env gained SO101_SPAWN_MODE, SO101_SPAWN_FRAC and the rollout recorder
# since then. Those SHOULD be inert for physics and cameras, but "should be
# inert" is exactly the class of assumption that has cost this project days.
# Re-measure v4's own best checkpoint on TODAY's env, same seed 909. Whatever
# this returns -- not the historical 0.125 -- is the baseline v4b is judged
# against.
V4OLD=/data08/henryg/pai/results/so101_sft_v4/so101_sft_openpi_pi05/checkpoints/global_step_1000
mkdir -p "$V4OLD/so101-sim-demos-v4"; cp $STATS "$V4OLD/so101-sim-demos-v4/" 2>/dev/null || true
BASE=$(run_eval "$V4OLD" 909 control_v4_s909)
log "CONTROL: v4 step_1000 re-measured on today env, seed 909: ${BASE:-FAIL}   [historical 0.125]"
if [ -n "$BASE" ]; then
  log "CONTROL drift: $(awk -v b="$BASE" 'BEGIN{printf "%+.1f pts", (b-0.125)*100}')  $(awk -v b="$BASE" 'BEGIN{d=b-0.125; if(d<0)d=-d; print (d<=0.04)?"(within noise -- historical baseline usable)":"(ENV DRIFTED -- judge against this number, not 0.125)"}')"
else
  BASE=0.125
  log "CONTROL produced no number; falling back to the historical 0.125"
fi

export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ --config-name so101_sft_v4b \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v4b > "$SCRATCH/preflight_v4b.out" 2>&1
grep -q 'PREFLIGHT OK' "$SCRATCH/preflight_v4b.out" || { log "PREFLIGHT FAIL"; tail -4 "$SCRATCH/preflight_v4b.out" >> "$STATUS"; exit 1; }

clean
log "v4b SFT launching (warm start = REAL-DATA SFT step_8000, everything else = v4)"
timeout 14400 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ --config-name so101_sft_v4b \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v4b \
  > "$SCRATCH/sft_v4b.out" 2>&1
log "v4b SFT exit=$?"
CKS=$(ls -d $CKROOT/global_step_* 2>/dev/null | sort -V)
[ -n "$CKS" ] || { log "GATE FAIL: no ckpts"; exit 1; }
for CK in $CKS; do mkdir -p "$CK/so101-sim-demos-v4"; cp $STATS "$CK/so101-sim-demos-v4/" 2>/dev/null || true; done

export EMBODIED_PATH=$PWD/examples/embodiment
SCREEN=$SCRATCH/v4b_screen.txt; : > "$SCREEN"
for CK in $CKS; do
  E=$(run_eval "$CK" 777 "$(basename $CK)_s777")
  log "screen(full board) $(basename $CK) s777: ${E:-FAIL}   [v4 was 0.109 here]"
  [ -n "$E" ] && echo "$E $CK" >> "$SCREEN"
done
BESTCK=""; BESTAVG=0
while read -r E1 CK; do
  E2=$(run_eval "$CK" 888 "$(basename $CK)_s888")
  log "confirm(full board) $(basename $CK) s888: ${E2:-FAIL}"
  [ -n "$E2" ] || continue
  AVG=$(awk -v a="$E1" -v b="$E2" 'BEGIN{printf "%.4f",(a+b)/2}')
  if awk -v a="$AVG" -v b="$BESTAVG" 'BEGIN{exit !(a>b)}'; then BESTAVG=$AVG; BESTCK=$CK; fi
done < <(sort -gr "$SCREEN" | head -3)
[ -n "$BESTCK" ] || { log "GATE FAIL"; exit 1; }

# seed 909 is exactly the seed that produced v4's 12.5% -> directly comparable
V=$(run_eval "$BESTCK" 909 vfy909)
log "VERIFY(full board, seed 909) $(basename $BESTCK): ${V:-FAIL}   [control baseline = $BASE]"
log "V4B FINAL: ckpt=$(basename "$BESTCK") gate=$BESTAVG verify=${V:-?} vs control $BASE (historical 0.125)"
if [ -n "$V" ]; then
  # judged against the CONTROL, so an env shift cannot masquerade as a
  # warm-start effect. 1 sd at p~0.12, n=128 is 2.9 pts.
  log "SUBSTITUTION VERDICT: $(awk -v v="$V" -v b="$BASE" 'BEGIN{
        r=v-b;
        if (r>=-0.03) print "VALIDATED -- within noise of the control; the reproduction doc stands";
        else if (r>=-0.07) print "DEGRADED -- doc must warn; re-check stage C from this checkpoint";
        else print "INVALID -- the doc needs a different bridge stage"}')"
fi
echo "$BESTCK" > "$SCRATCH/v4b_best.ck"
log "v4b done; disk free $(df -h /data08 | tail -1 | awk '{print $4}')"
