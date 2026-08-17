#!/bin/bash
# STATUS: SUPERSEDED — 早期任务规格，已被主线取代。别用来复现。 下载官方 ManiSkill SFT 检查点，该任务线已停
# Download the published pi0.5 ManiSkill-25Main SFT checkpoint (7.5 GB) so we
# can try to reproduce the paper's 40.1% SFT number on OUR machine. Network
# only -- no GPU. HF_HUB_OFFLINE must be off for this one process.
set -uo pipefail
# Logs and status files. Overridable so the script runs outside the session
# it was written in; without the mkdir every redirect below fails on a
# fresh machine and the script dies before doing anything.
SCRATCH=${SCRATCH:-/tmp/so101_runs}
mkdir -p "$SCRATCH"
cd /data08/henryg/pai/RLinf
log(){ echo "[$(date '+%F %T')] $*" >> "$SCRATCH/repro.status"; }
log "downloading RLinf-Pi05-ManiSkill-25Main-SFT (7.5 GB)"
HF_HUB_OFFLINE=0 timeout 7200 .venv/bin/huggingface-cli download \
  RLinf/RLinf-Pi05-ManiSkill-25Main-SFT \
  --local-dir /data08/henryg/pai/models/RLinf-Pi05-ManiSkill-25Main-SFT \
  > "$SCRATCH/dl_official.out" 2>&1
RC=$?
SZ=$(du -sh /data08/henryg/pai/models/RLinf-Pi05-ManiSkill-25Main-SFT 2>/dev/null | awk '{print $1}')
log "download exit=$RC size=$SZ"
[ -f /data08/henryg/pai/models/RLinf-Pi05-ManiSkill-25Main-SFT/model.safetensors ] \
  && log "REPRO-DL DONE: checkpoint ready, awaiting a GPU slot for the eval" \
  || log "REPRO-DL FAIL"
