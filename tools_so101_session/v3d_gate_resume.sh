#!/bin/bash
# Resume v3d gate after session teardown killed the pipeline mid-gate.
# Done already: 1000 avg 0.0625; 2000 s777 0.0234. Remaining: 2000 s888,
# 3000, 4000, then verify + bands on the best.
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/v3.status
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
    SO101_SPAWN_FRAC="$SF" timeout 900 .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
      --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v3 \
      rollout.model.model_path="$1" \
      rollout.model.openpi.config_name=pi05_so101_v3 \
      rollout.model.openpi_data.norm_stats_path=/data08/henryg/pai/RLinf/assets/pi05_so101_v3/so101-sim-demos-v3/norm_stats.json \
      env.eval.total_num_envs=128 \
      env.eval.seed=$2 \
      > "$SCRATCH/eval_v3d_$3_t$TRY.out" 2>&1
    local EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v3d_$3_t$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
    log "eval $3 try$TRY empty, retry"
  done
  return 0
}

B=/data08/henryg/pai/results/so101_sft_v3d/so101_sft_openpi_pi05/checkpoints
log "v3d gate RESUME"
E=$(run_eval $B/global_step_2000 888 global_step_2000_s888); log "gateD global_step_2000 s888: ${E:-FAIL}"
BESTCK=$B/global_step_1000; BESTAVG=0.0625
AVG2=$(awk -v a="0.0234375" -v b="${E:-0}" 'BEGIN{printf "%.4f",(a+b)/2}'); log "gateD avg global_step_2000: $AVG2"
if awk -v a="$AVG2" -v b="$BESTAVG" 'BEGIN{exit !(a>b)}'; then BESTAVG=$AVG2; BESTCK=$B/global_step_2000; fi
for CK in $B/global_step_3000 $B/global_step_4000; do
  E1=$(run_eval "$CK" 777 $(basename $CK)_s777); log "gateD $(basename $CK) s777: ${E1:-FAIL}"
  E2=$(run_eval "$CK" 888 $(basename $CK)_s888); log "gateD $(basename $CK) s888: ${E2:-FAIL}"
  [ -n "${E1:-}" ] && [ -n "${E2:-}" ] || continue
  AVG=$(awk -v a="$E1" -v b="$E2" 'BEGIN{printf "%.4f",(a+b)/2}')
  log "gateD avg $(basename $CK): $AVG"
  if awk -v a="$AVG" -v b="$BESTAVG" 'BEGIN{exit !(a>b)}'; then BESTAVG=$AVG; BESTCK=$CK; fi
done
echo "$BESTCK" > "$SCRATCH/v3d_best.ck"
V=$(run_eval "$BESTCK" 909 vD_s909); log "v3d VERIFY s909: ${V:-FAIL}"
B0=$(run_eval "$BESTCK" 606 bandD_right "0,1,0,0.33");   log "v3d y-band right: ${B0:-FAIL}"
B1=$(run_eval "$BESTCK" 606 bandD_mid   "0,1,0.33,0.66"); log "v3d y-band middle: ${B1:-FAIL}"
B2=$(run_eval "$BESTCK" 606 bandD_left  "0,1,0.66,1");   log "v3d y-band left: ${B2:-FAIL}"
log "V3D PIPELINE DONE: best=$BESTCK gate=$BESTAVG verify=${V:-?} bands=[${B0:-?} ${B1:-?} ${B2:-?}]"
