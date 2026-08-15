#!/bin/bash
# TRACK 2 (GPU) — three measurements, then the held-out co-training run.
#
# A. v14's best scored on real episodes 70-86 instead of 0-3. Still contaminated
#    (all 87 were in its training set) but it is the same slice the honest v15
#    gate will use, so the two numbers become comparable.
# B. the same checkpoint with the wrist channel dropped. Before co-training that
#    changed nothing (4.47 -> 4.59); now that the model has seen REAL wrist
#    images the answer may flip, and it decides whether deployment sends one
#    camera or two.
# C. does more training on the SAME data keep lowering the ratio? v14 flattened
#    at 0.84 over its last three checkpoints, which either means step-limited or
#    converged-on-this-mixture. 2000 more steps separates those.
# D. when track 1 finishes, train on the held-out-split dataset and gate on BOTH
#    axes with the offline check restricted to unseen real episodes.
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/night.status
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
REAL=/data08/henryg/pai/data/so101-pick-place-v1-trimmed
V14=/data08/henryg/pai/results/so101_sft_v14/so101_sft_openpi_pi05/checkpoints/global_step_1750
RING1="0.4294,0.9115,0.5142,0.9817"
HELDOUT_START=70          # real episodes 70..86 were never trained on in v15
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] [GPU] $*" >> "$STATUS"; }

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
  sleep 8
}

offline(){  # ckpt config tag [extra flags] -> "sim_ratio real_ratio"
  local CK=$1 CFG=$2 TAG=$3; shift 3
  CUDA_VISIBLE_DEVICES=0 timeout 3600 .venv/bin/python tools_so101_session/offline_replay_check.py \
    --ckpt "$CK" --config-name "$CFG" --norm-stats $STATS --chunks 10 \
    --real-root $REAL --ep-start $HELDOUT_START --episodes 5 --frames 10 "$@" \
    > "$SCRATCH/offline_$TAG.out" 2>&1
  grep -a 'ratio:' "$SCRATCH/offline_$TAG.out" | grep -oE '[0-9]+\.[0-9]+' | tr '\n' ' '
}
sim_eval(){  # ckpt cfg seed tag
  local TRY EV
  for TRY in 1 2; do
    clean
    SO101_SPAWN_FRAC="$RING1" timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_night \
      rollout.model.model_path="$1" rollout.model.openpi.config_name="$2" \
      rollout.model.openpi_data.norm_stats_path=$STATS \
      rollout.model.num_action_chunks=10 \
      env.eval.total_num_envs=128 env.eval.seed=$3 > "$SCRATCH/eval_$4_r$TRY.out" 2>&1
    EV=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_$4_r$TRY.out" | tail -1 | cut -d= -f2)
    [ -n "$EV" ] && { echo "$EV"; return 0; }
  done
  echo ""
}

# ---- A: v14 best on the slice v15 will be judged on ----
R=$(offline "$V14" pi05_so101_v14 v14_ep70)
log "A. v14 best on real episodes 70-86 (contaminated: it trained on them): $R"

# ---- B: does dropping the wrist channel help now? ----
R=$(offline "$V14" pi05_so101_v14 v14_ep70_nowrist --no-wrist)
log "B. same, front camera only: $R   [before co-training the answer was 'no change']"

# ---- C: is it step-limited? 2000 more steps on the SAME data ----
export EMBODIED_PATH=$PWD/examples/sft
python3 - <<'PY'
s = open("examples/sft/config/so101_sft_v14.yaml").read()
s = s.replace('model_path: "/data08/henryg/pai/results/so101_ppo_v13/so101_ppo_v11/checkpoints/global_step_30"',
              'model_path: "/data08/henryg/pai/results/so101_sft_v14/so101_sft_openpi_pi05/checkpoints/global_step_1750"')
open("examples/sft/config/so101_sft_v14b.yaml", "w").write(
  "# v14b: 2000 MORE steps on the same co-training data, continuing from v14's\n"
  "# best. v14's ratio flattened at 0.84 across its last three checkpoints; this\n"
  "# separates 'step-limited' from 'converged on this mixture'.\n" + s)
PY
clean
log "C. launching +2000 steps on the same data"
timeout 14400 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name so101_sft_v14b \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v14b > "$SCRATCH/sft_v14b.out" 2>&1
CKS=$(ls -d /data08/henryg/pai/results/so101_sft_v14b/so101_sft_openpi_pi05/checkpoints/global_step_* 2>/dev/null | sort -V)
for CK in $CKS; do mkdir -p "$CK/so101-sim-demos-v4"; cp $STATS "$CK/so101-sim-demos-v4/" 2>/dev/null || true; done
export EMBODIED_PATH=$PWD/examples/embodiment
for CK in $(echo "$CKS" | tail -3); do
  R=$(offline "$CK" pi05_so101_v14 "v14b_$(basename $CK)")
  S=$(sim_eval "$CK" pi05_so101_v14 4141 "v14b_$(basename $CK)")
  log "C. $(basename $CK): offline(sim real)=$R | sim ring-1=${S:-FAIL}"
done

# ---- D: the honest run, once track 1 has the dataset ----
data_ready(){ grep -q 'DATASET READY' "$STATUS" 2>/dev/null; }
DL=$(( $(date +%s) + 8*3600 ))
while ! data_ready; do
  [ "$(date +%s)" -gt "$DL" ] && { log "D ABORT: dataset never finished"; exit 1; }
  if ! pgrep -f 'bash .*night_cpu.sh' >/dev/null; then
    sleep 5; data_ready && break
    log "D ABORT: dataset builder exited without its marker"; exit 1
  fi
  sleep 120
done
sleep 20
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config \
  --config-path $PWD/examples/sft/config/ --config-name so101_sft_v15 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v15 > "$SCRATCH/preflight_v15.out" 2>&1
grep -q 'PREFLIGHT OK' "$SCRATCH/preflight_v15.out" || { log "D PREFLIGHT FAIL"; tail -5 "$SCRATCH/preflight_v15.out" >> "$STATUS"; exit 1; }
clean
log "D. launching the held-out co-training run"
timeout 14400 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name so101_sft_v15 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v15 > "$SCRATCH/sft_v15.out" 2>&1
CKS=$(ls -d /data08/henryg/pai/results/so101_sft_v15/so101_sft_openpi_pi05/checkpoints/global_step_* 2>/dev/null | sort -V)
[ -n "$CKS" ] || { log "D FAIL: no ckpts"; exit 1; }
for CK in $CKS; do mkdir -p "$CK/so101-sim-demos-v4"; cp $STATS "$CK/so101-sim-demos-v4/" 2>/dev/null || true; done
export EMBODIED_PATH=$PWD/examples/embodiment
BEST=""; BESTR=999; SIM_FLOOR=0.50
for CK in $CKS; do
  R=$(offline "$CK" pi05_so101_v15 "v15_$(basename $CK)")
  SIMR=$(echo "$R" | awk '{print $1}'); REALR=$(echo "$R" | awk '{print $2}')
  S=$(sim_eval "$CK" pi05_so101_v15 4141 "v15_$(basename $CK)")
  log "D. $(basename $CK): offline sim=${SIMR:-?} real(HELD OUT)=${REALR:-?} | sim ring-1=${S:-FAIL}"
  # both axes decide: the constraint vetoes before the objective is compared
  if [ -n "${REALR:-}" ] && [ -n "${S:-}" ] \
     && awk -v s="$S" -v f="$SIM_FLOOR" 'BEGIN{exit !(s>=f)}' \
     && awk -v a="$REALR" -v b="$BESTR" 'BEGIN{exit !(a<b)}'; then BESTR=$REALR; BEST=$CK; fi
done
[ -n "$BEST" ] || { log "D FAIL: nothing cleared the sim floor"; exit 1; }
S2=$(sim_eval "$BEST" pi05_so101_v15 4242 "v15_best_s4242")
log "NIGHT FINAL: best=$(basename $BEST) held-out real ratio=$BESTR sim ring-1 seed4242=${S2:-?}"
log "reference points: real-trained policy 0.22 (upper bound) | pre-co-training 4.47 | v14 train-set 0.84"
echo "$BEST" > "$SCRATCH/v15_best.ck"
