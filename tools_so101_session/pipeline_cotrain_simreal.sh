#!/bin/bash
# v14 — sim+real co-training, the first sim2real step.
#
# The offline check established the problem and its target with two symmetric
# measurements on the SAME metric (policy action error / hold-still error):
#     sim-trained policy : sim 0.10  real 4.47   <- ours, unusable on real
#     real-trained policy: sim 3.82  real 0.22   <- the target for `real`
# So the metric is sound and the gap is visual-domain, not a single defect
# (removing the known-bad wrist channel changed real from 4.47 to 4.59).
#
# S1 build the mixed dataset (v10 sim + 87 real x3)      ~80 min CPU
# S2 preflight + gentle SFT from the PPO peak            ~50 min
# S3 gate: EVERY checkpoint gets the offline real ratio (the thing we are
#    trying to fix) AND a sim ring-1 eval (the thing we must not break)
#
# Timeouts are sized from measured rates: video encoding runs ~3.3 episodes/min,
# so 261 appended episodes is ~80 min -> 4 h budget.
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/v14.status
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
CKROOT=/data08/henryg/pai/results/so101_sft_v14/so101_sft_openpi_pi05/checkpoints
REAL=/data08/henryg/pai/data/so101-pick-place-v1-trimmed
RING1="0.4294,0.9115,0.5142,0.9817"
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

export REPO_PATH="$PWD" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

clean(){
  .venv/bin/ray stop --force >/dev/null 2>&1
  for p in $(pgrep -f 'ray::|raylet|gcs_server|eval_embodied_agent|train_vla_sft' 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p" 2>/dev/null
  done
  rm -rf /tmp/ray/session_* 2>/dev/null
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
  SHM=$(df -m /dev/shm | tail -1 | awk '{print $4}')
  [ "${SHM:-0}" -lt 1024 ] && mount -o remount,size=16G /dev/shm 2>/dev/null
  sleep 8
}

FREE=$(df --output=avail -BG /data08 | tail -1 | tr -dc '0-9')
[ "$FREE" -lt 200 ] && { log "ABORT: disk <200G"; exit 3; }

# ---------- S1: mixed dataset ----------
log "S1 building sim+real co-training dataset (v10 sim + 87 real x3)"
timeout 14400 .venv/bin/python tools_so101_session/convert_cotrain_simreal.py > "$SCRATCH/convert_v14.out" 2>&1
grep -qa '^DONE:' "$SCRATCH/convert_v14.out" || { log "S1 FAIL"; tail -4 "$SCRATCH/convert_v14.out" >> "$STATUS"; exit 1; }
log "S1 $(grep -a '^DONE:' "$SCRATCH/convert_v14.out" | tail -1)"
log "S1 $(grep -a 'real share' "$SCRATCH/convert_v14.out" | tail -1)"

# ---------- S2: gentle SFT ----------
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config \
  --config-path $PWD/examples/sft/config/ --config-name so101_sft_v14 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v14 > "$SCRATCH/preflight_v14.out" 2>&1
grep -q 'PREFLIGHT OK' "$SCRATCH/preflight_v14.out" || { log "S2 PREFLIGHT FAIL"; tail -5 "$SCRATCH/preflight_v14.out" >> "$STATUS"; exit 1; }
clean
log "S2 SFT launching (from the PPO peak, lr 1e-5, 2000 steps, save 250)"
timeout 14400 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name so101_sft_v14 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v14 \
  > "$SCRATCH/sft_v14.out" 2>&1
log "S2 SFT exit=$?"
CKS=$(ls -d $CKROOT/global_step_* 2>/dev/null | sort -V)
[ -n "$CKS" ] || { log "S2 GATE FAIL: no ckpts"; exit 1; }
for CK in $CKS; do mkdir -p "$CK/so101-sim-demos-v4"; cp $STATS "$CK/so101-sim-demos-v4/" 2>/dev/null || true; done

# ---------- S3: gate on BOTH axes ----------
export EMBODIED_PATH=$PWD/examples/embodiment
sim_eval(){  # ckpt seed -> success
  local TRY EV
  for TRY in 1 2; do
    clean
    SO101_SPAWN_FRAC="$RING1" timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v14 \
      rollout.model.model_path="$1" rollout.model.openpi.config_name=pi05_so101_v14 \
      rollout.model.openpi_data.norm_stats_path=$STATS \
      rollout.model.num_action_chunks=10 \
      env.eval.total_num_envs=128 env.eval.seed=$2 \
      > "$SCRATCH/eval_v14_$(basename $1)_s$2_r$TRY.out" 2>&1
    EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v14_$(basename $1)_s$2_r$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
  done
  echo ""
}
offline_ratio(){  # ckpt -> "sim_ratio real_ratio"
  CUDA_VISIBLE_DEVICES=0 timeout 3600 .venv/bin/python tools_so101_session/offline_replay_check.py \
    --ckpt "$1" --config-name pi05_so101_v14 --norm-stats $STATS --chunks 10 \
    --real-root $REAL --episodes 4 --frames 10 > "$SCRATCH/offline_v14_$(basename $1).out" 2>&1
  grep -a 'ratio:' "$SCRATCH/offline_v14_$(basename $1).out" | grep -oE '[0-9]+\.[0-9]+' | tr '\n' ' '
}

log "S3 gate: offline real ratio (target <1, real-trained policy gets 0.22) + sim ring-1 (must not collapse from 0.578)"
BEST=""; BESTR=999
for CK in $CKS; do
  R=$(offline_ratio "$CK")
  SIMR=$(echo "$R" | awk '{print $1}'); REALR=$(echo "$R" | awk '{print $2}')
  S1=$(sim_eval "$CK" 4141)
  log "$(basename $CK): offline sim=${SIMR:-?} real=${REALR:-?} | sim ring-1 seed4141=${S1:-FAIL}"
  if [ -n "${REALR:-}" ] && awk -v a="$REALR" -v b="$BESTR" 'BEGIN{exit !(a<b)}'; then BESTR=$REALR; BEST=$CK; fi
done
[ -n "$BEST" ] || { log "S3 FAIL: no checkpoint produced an offline number"; exit 1; }
S2v=$(sim_eval "$BEST" 4242)
log "V14 FINAL: best=$(basename $BEST) offline_real=$BESTR sim_ring1_seed4242=${S2v:-?}"
log "verdict: $(awk -v r="$BESTR" 'BEGIN{
  if (r<1.0) print "co-training WORKS on the offline metric (was 4.47); check the sim number did not collapse";
  else print "co-training insufficient at this mix ratio; next lever is domain randomisation or more real data"}')"
echo "$BEST" > "$SCRATCH/v14_best.ck"
