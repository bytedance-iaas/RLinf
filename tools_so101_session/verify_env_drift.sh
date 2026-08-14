#!/bin/bash
# Re-run of the ENV-DRIFT CONTROL that verify_warmstart_substitution.sh fumbled: it placed the
# control eval BEFORE the line that exports EMBODIED_PATH, so all three tries
# died instantly with KeyError: 'EMBODIED_PATH' and the script fell back to the
# historical 0.125 -- i.e. the very number the control exists to re-check.
#
# Question: does v4's own best checkpoint still score ~12.5% on TODAY's env,
# seed 909? If yes, the historical baseline is usable for judging v4b. If no,
# the baseline itself moved and v4b must be judged against THIS number.
#
# Note: three retries are useless against a config error -- they just repeat it.
# So this script fails loudly on the first try instead of pretending to retry.
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/v4b.status
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
V4OLD=/data08/henryg/pai/results/so101_sft_v4/so101_sft_openpi_pi05/checkpoints/global_step_1000
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

export REPO_PATH="$PWD" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1
export EMBODIED_PATH=$PWD/examples/embodiment       # <-- the line that was missing
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Wait for v4b to finish ENTIRELY -- not just its SFT. First attempt waited only
# for the SFT and collided with v4b's own gate, whose clean() kills every ray
# process on the box: two scripts each assuming exclusive ownership of the GPUs.
# Downstream stages must serialise against each other, not only against the
# stage that produced their input.
v4b_done(){ grep -qE 'V4B FINAL|GATE FAIL|PREFLIGHT FAIL|ABORT:' "$STATUS" 2>/dev/null; }
DEADLINE=$(( $(date +%s) + 8*3600 ))
while ! v4b_done; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "CONTROL-RERUN ABORT: v4b never finished"; exit 1; }
  if ! pgrep -f 'bash .*verify_warmstart_substitution.sh' >/dev/null; then
    sleep 5; v4b_done && break
    log "CONTROL-RERUN ABORT: v4b exited without a completion marker"; exit 1
  fi
  sleep 60
done
sleep 30

.venv/bin/ray stop --force >/dev/null 2>&1 || true
find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete 2>/dev/null
sleep 5
mkdir -p "$V4OLD/so101-sim-demos-v4"; cp $STATS "$V4OLD/so101-sim-demos-v4/" 2>/dev/null || true

timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
  --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
  --config-name so101_eval_openpi_pi05 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v4b \
  rollout.model.model_path="$V4OLD" \
  rollout.model.openpi.config_name=pi05_so101_v4 \
  rollout.model.openpi_data.norm_stats_path=$STATS \
  env.eval.total_num_envs=128 env.eval.seed=909 \
  > "$SCRATCH/eval_v4b_control_rerun.out" 2>&1
E=$(grep -oE 'success_once=[0-9.]+' "$SCRATCH/eval_v4b_control_rerun.out" | tail -1 | cut -d= -f2)
if [ -z "$E" ]; then
  log "CONTROL-RERUN FAILED; last lines:"; tail -5 "$SCRATCH/eval_v4b_control_rerun.out" >> "$STATUS"; exit 1
fi
log "CONTROL-RERUN: v4 step_1000 on today env, seed 909 = $E   [historical 0.125]"
log "CONTROL-RERUN drift: $(awk -v b="$E" 'BEGIN{printf "%+.1f pts", (b-0.125)*100}')  $(awk -v b="$E" 'BEGIN{d=b-0.125; if(d<0)d=-d; print (d<=0.04)?"(within noise -- historical baseline usable)":"(ENV DRIFTED -- judge v4b against this number)"}')"
