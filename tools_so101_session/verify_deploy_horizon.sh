#!/bin/bash
# The YAML defines openpi.action_chunk as ${..num_action_chunks}; hand-built
# configs do not inherit that and fall back to the dataclass default of 5.
# The sim evals went through the YAML (horizon 10); the offline gate and the
# deploy server were hand-built (horizon 5). Two things to settle:
#   1. the deploy server now really returns 10
#   2. what the held-out gate reads at horizon 10 -- the headline 0.79 was
#      measured at 5, and it is the number authorising a real-robot trial
set -uo pipefail
SCRATCH=/tmp/claude-0/-data08-henryg-pai-RLinf/3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad
STATUS=$SCRATCH/horizon.status
CK=/data08/henryg/pai/results/so101_sft_v15/so101_sft_openpi_pi05/checkpoints/global_step_1000
STATS=/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
REAL=/data08/henryg/pai/data/so101-pick-place-v1-trimmed
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
export REPO_PATH=$PWD PYTHONPATH=$PWD EMBODIED_PATH=$PWD/examples/embodiment
export HF_LEROBOT_HOME=/data08/henryg/pai/data HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0

timeout 900 .venv/bin/python tools_so101_session/deploy_policy_server.py \
  --ckpt $CK --config-name pi05_so101_v15 --norm-stats $STATS --port 8099 \
  > $SCRATCH/server_selftest.out 2>&1 &
SRV=$!
for _ in $(seq 60); do grep -qa 'self-test OK' $SCRATCH/server_selftest.out && break; sleep 10; done
log "server: $(grep -a 'self-test OK' $SCRATCH/server_selftest.out || echo 'NO SELF-TEST LINE')"
kill $SRV 2>/dev/null; sleep 5

# held-out gate at the horizon the sim numbers used
timeout 3600 .venv/bin/python tools_so101_session/offline_replay_check.py \
  --ckpt $CK --config-name pi05_so101_v15 --norm-stats $STATS --chunks 10 \
  --real-root $REAL --ep-start 70 --episodes 5 --frames 10 \
  > $SCRATCH/offline_v15_h10.out 2>&1
log "held-out gate at horizon 10: $(grep -a 'ratio:' $SCRATCH/offline_v15_h10.out | tr '\n' ' ')"
log "for comparison, the same checkpoint at horizon 5 read sim 0.10-ish / real 0.79"
log "HORIZON DONE"
