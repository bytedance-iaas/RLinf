#!/usr/bin/env bash
# Boot a real uvicorn against a synthetic run tree and check the HTTP surface.
#
# This covers what the pytest suite cannot. `TestClient` runs the app through a
# synchronous portal that waits for the response generator to finish on close, so
# an endpoint that streams forever hangs the test process -- which is why
# test_api.py drives the SSE route handlers directly instead of over the
# transport. Actual chunked delivery, the event-stream content type and the
# proxy-buffering headers therefore only get exercised here.
#
# The run tree is written by hand rather than by importing rlinf. That is the
# same contract the dashboard itself relies on: a directory layout, not a
# library.
#
# Usage: smoke_server.sh [python]     (default: python3)

set -euo pipefail

PYTHON="${1:-python3}"
PORT="${PORT:-8421}"
AUTH_USER="smoke-operator"
AUTH_PASSWORD="smoke-password"
ROOT="$(mktemp -d)"
RUN_ROOT="$ROOT/_rlinf/runs/smoke-run"
SERVER_LOG="$ROOT/server.log"
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ -f "$SERVER_LOG" ]; then
    echo "--- server log ---"
    cat "$SERVER_LOG"
  fi
  rm -rf "$ROOT"
}
trap cleanup EXIT

fail() {
  echo "SMOKE FAIL: $*" >&2
  exit 1
}

mkdir -p "$RUN_ROOT"

cat > "$RUN_ROOT/manifest.json" <<JSON
{
  "schema_version": 2,
  "run_id": "smoke-run",
  "task_type": "embodied",
  "experiment_name": "smoke",
  "project_name": "smoke-proj",
  "step_semantics": "rl_iteration",
  "paths": {"log_path": "$ROOT", "tensorboard": "$ROOT/tensorboard"},
  "metric_aliases": {"actor/training/": "train/actor/"}
}
JSON

# Timestamps are stamped at "now" so the run reads as healthy. A fixture with
# recorded timestamps would be `unreachable` by the time CI runs it, which is
# correct but would not exercise the healthy path.
write_snapshot() {
  local step="$1"
  "$PYTHON" - "$RUN_ROOT" "$step" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

run_root, step = sys.argv[1], int(sys.argv[2])
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
snapshot = {
    "schema_version": 2,
    "run_id": "smoke-run",
    "task_type": "embodied",
    "state": "running",
    "phase": "rollout",
    "phase_since": now,
    "components": {},
    "heartbeat_at": now,
    "heartbeat_seq": step * 10,
    "last_progress_at": now,
    "last_metric_at": now,
    "progress": {"step": step, "max_steps": 100, "step_semantics": "rl_iteration"},
    "timing": {"started_at": now, "elapsed_s": 42.0, "step_time_p50": 3.0},
    "exit": None,
}
# The same atomic publication the training side uses: a reader must never see a
# half-written document, and this script is a writer like any other.
tmp = os.path.join(run_root, f"run.json.tmp.{os.getpid()}")
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(snapshot, handle)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, os.path.join(run_root, "run.json"))
PY
}

write_snapshot 7

printf '%s\n' '{"step": 5, "path": "'"$ROOT"'/ckpt/global_step_5", "saved_at": "2026-08-03T00:00:00Z", "size_bytes": 1024, "resume_dir": "'"$ROOT"'/ckpt/global_step_5", "entry_script": "examples/embodiment/train_embodied_agent.py", "config_name": "smoke"}' \
  > "$RUN_ROOT/checkpoints.jsonl"
printf '%s\n' '{"ts": "2026-08-03T00:00:00Z", "kind": "run_start", "step": 0, "payload": {}}' \
  > "$RUN_ROOT/events.jsonl"

# Two clips whose outcomes differ, plus one with no outcome recorded. The third
# row is the one that matters: it must answer neither the success nor the failure
# query, and only a real request proves the boolean was coerced rather than
# treated as an always-true string.
mkdir -p "$ROOT/videos"
printf '\x00\x00\x00\x18ftypmp42smoke-video-bytes' > "$ROOT/videos/won.mp4"
printf '\x00\x00\x00\x18ftypmp42smoke-video-bytes' > "$ROOT/videos/lost.mp4"
printf '\x00\x00\x00\x18ftypmp42smoke-video-bytes' > "$ROOT/videos/unknown.mp4"
{
  printf '%s\n' '{"path": "'"$ROOT"'/videos/won.mp4", "step": 7, "split": "train", "shard": 0, "num_frames": 60, "fps": 30, "num_success": 3, "num_envs": 8}'
  printf '%s\n' '{"path": "'"$ROOT"'/videos/lost.mp4", "step": 7, "split": "eval", "shard": 0, "num_success": 0, "num_envs": 8}'
  printf '%s\n' '{"path": "'"$ROOT"'/videos/unknown.mp4", "step": 9, "split": "train", "shard": 0}'
} > "$RUN_ROOT/media.rank0.jsonl"

echo "Serving $ROOT on port $PORT"
RLINF_DASHBOARD_SCAN_ROOT="$ROOT" \
RLINF_DASHBOARD_SSE_INTERVAL_S=0.2 \
RLINF_DASHBOARD_DISCOVERY_CACHE_TTL_S=0 \
RLINF_DASHBOARD_AUTH_MODE=basic \
RLINF_DASHBOARD_AUTH_USERNAME="$AUTH_USER" \
RLINF_DASHBOARD_AUTH_PASSWORD="$AUTH_PASSWORD" \
  "$PYTHON" -m rlinf_dashboard --host 127.0.0.1 --port "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/healthz" > /dev/null 2>&1; then
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || fail "server exited during startup"
  sleep 0.5
done
curl -fsS "http://127.0.0.1:$PORT/healthz" > /dev/null || fail "server never became ready"

get() {
  curl -fsS --user "$AUTH_USER:$AUTH_PASSWORD" "http://127.0.0.1:$PORT$1"
}

# Every assertion below is on a JSON body, so it is checked in Python rather
# than by grepping for substrings that could match anywhere in the document.
check() {
  local label="$1"
  local body="$2"
  local expr="$3"
  echo "$body" | "$PYTHON" -c "
import json, sys
body = json.load(sys.stdin)
assert $expr, f'$label: got {body!r}'
print('OK: $label')
" || fail "$label"
}

check "health finds the run" "$(get /api/health)" \
  "body['status'] == 'ok' and body['run_count'] == 1"

check "healthz is public and contains no run metadata" "$(curl -fsS "http://127.0.0.1:$PORT/healthz")" \
  "body == {'status': 'ok'}"

check "the dashboard is closed without credentials" \
  "$(curl -sS -o /dev/null -w '{"code": %{http_code}}' "http://127.0.0.1:$PORT/api/health")" \
  "body['code'] == 401"

check "run is listed and healthy" "$(get /api/runs)" \
  "len(body) == 1 and body[0]['run_id'] == 'smoke-run' and body[0]['health'] == 'healthy' and body[0]['step'] == 7"

check "full status carries the verdict" "$(get /api/runs/smoke-run)" \
  "body['snapshot']['progress']['step'] == 7 and body['health']['health'] == 'healthy'"

check "checkpoints carry resume fields" "$(get /api/runs/smoke-run/checkpoints)" \
  "len(body) == 1 and body[0]['resume_dir'].endswith('global_step_5') and body[0]['config_name'] == 'smoke'"

check "events are readable" "$(get /api/runs/smoke-run/events)" \
  "body[0]['kind'] == 'run_start'"

check "a template is bound for the run" "$(get /api/runs/smoke-run/template)" \
  "body['name'] in {'embodied', 'fallback'} and 'groups' in body"

check "an unknown run is a 404, not a 500" \
  "$(curl -sS --user "$AUTH_USER:$AUTH_PASSWORD" -o /dev/null -w '{"code": %{http_code}}' "http://127.0.0.1:$PORT/api/runs/nope")" \
  "body['code'] == 404"

check "openapi is generated" "$(get /openapi.json)" \
  "'/api/runs/{run_id}/series' in body['paths']"

# Looked up by name rather than by position: entries come back ordered by step,
# and two clips here share a step, so any index-based assertion would be pinning
# tie-break order that the API does not promise.
check "media lists every indexed clip with its counts" "$(get /api/runs/smoke-run/media)" \
  "len(body) == 3 and {p['path'].rsplit('/', 1)[-1]: (p['num_success'], p['num_envs']) for p in body} == {'won.mp4': (3, 8), 'lost.mp4': (0, 8), 'unknown.mp4': (None, None)}"

# Over HTTP because `success` is a query-string boolean: FastAPI has to coerce
# 'true'/'false' from text, and a wrong annotation makes every value truthy --
# which no service-layer test would notice.
check "success=true keeps only clips that worked" \
  "$(get '/api/runs/smoke-run/media?success=true')" \
  "[p['path'].rsplit('/', 1)[-1] for p in body] == ['won.mp4']"

check "success=false keeps only clips that did not" \
  "$(get '/api/runs/smoke-run/media?success=false')" \
  "[p['path'].rsplit('/', 1)[-1] for p in body] == ['lost.mp4']"

check "an eval clip is labelled eval, not train" \
  "$(get '/api/runs/smoke-run/media?split=eval')" \
  "len(body) == 1 and body[0]['path'].endswith('lost.mp4')"

# The bytes themselves: the media route is the only endpoint that opens a file
# from a request parameter, so an allowlist regression here is a file-disclosure
# bug rather than a wrong number.
MEDIA_URL="$(get /api/runs/smoke-run/media | "$PYTHON" -c "
import json, sys
entries = {e['path'].rsplit('/', 1)[-1]: e['url'] for e in json.load(sys.stdin)}
print(entries['won.mp4'])
")"
get "$MEDIA_URL" > "$ROOT/served.mp4" || fail "an indexed clip did not stream"
cmp -s "$ROOT/served.mp4" "$ROOT/videos/won.mp4" \
  || fail "the streamed bytes differ from the file on disk"
echo "OK: an indexed clip streams byte-for-byte"

curl -fsS --user "$AUTH_USER:$AUTH_PASSWORD" --range 0-3 \
  -D "$ROOT/range.headers" "http://127.0.0.1:$PORT$MEDIA_URL" \
  -o "$ROOT/range.bin" || fail "an authenticated media range did not stream"
grep -q '206 Partial Content' "$ROOT/range.headers" \
  || fail "an authenticated range request did not return 206"
head -c 4 "$ROOT/videos/won.mp4" > "$ROOT/range.expected"
cmp -s "$ROOT/range.bin" "$ROOT/range.expected" \
  || fail "the authenticated media range returned the wrong bytes"
echo "OK: authenticated media ranges preserve 206 semantics"

check "an unindexed file is refused" \
  "$(curl -sS --user "$AUTH_USER:$AUTH_PASSWORD" -o /dev/null -w '{"code": %{http_code}}' \
    "http://127.0.0.1:$PORT/api/runs/smoke-run/media/file?path=/etc/passwd")" \
  "body['code'] == 404"

# The part that needs a real server: a stream that stays open, delivers an
# initial event, and delivers a second one after the file underneath changes.
echo "Checking SSE delivery"
check "an unauthenticated stream is refused before it opens" \
  "$(curl -sS -o /dev/null -w '{"code": %{http_code}}' "http://127.0.0.1:$PORT/api/stream/runs/smoke-run")" \
  "body['code'] == 401"
STREAM_OUT="$ROOT/stream.txt"
curl -sS -N --user "$AUTH_USER:$AUTH_PASSWORD" -D "$ROOT/stream.headers" \
  "http://127.0.0.1:$PORT/api/stream/runs/smoke-run" \
  > "$STREAM_OUT" 2>/dev/null &
CURL_PID=$!
sleep 1
write_snapshot 9
sleep 1
kill "$CURL_PID" 2>/dev/null || true
wait "$CURL_PID" 2>/dev/null || true

grep -qi '^content-type: text/event-stream' "$ROOT/stream.headers" \
  || fail "stream did not declare text/event-stream: $(cat "$ROOT/stream.headers")"
grep -qi '^x-accel-buffering: no' "$ROOT/stream.headers" \
  || fail "stream is missing the anti-buffering header a proxy needs"

"$PYTHON" - "$STREAM_OUT" <<'PY'
import json
import sys

raw = open(sys.argv[1], encoding="utf-8").read()
blocks = [b for b in raw.split("\n\n") if b.strip()]
if not blocks:
    sys.exit(f"SMOKE FAIL: no SSE events arrived. Raw: {raw!r}")

# The stream leads with a `retry:` frame telling the browser how long to wait
# before reconnecting. Checked here rather than only in pytest because it has to
# survive real chunked delivery to be worth anything.
if not blocks[0].startswith("retry: "):
    sys.exit(f"SMOKE FAIL: the stream did not lead with a retry hint: {blocks[0]!r}")
print(f"OK: stream declared {blocks[0].strip()}")

steps = []
for block in blocks[1:]:
    lines = dict(
        line.split(": ", 1) for line in block.splitlines() if ": " in line
    )
    if lines.get("event") != "update":
        sys.exit(f"SMOKE FAIL: unexpected SSE event: {block!r}")
    steps.append(json.loads(lines["data"])["snapshot"]["progress"]["step"])

print(f"OK: SSE delivered {len(steps)} events, steps {steps}")
if 9 not in steps:
    sys.exit(
        "SMOKE FAIL: the stream never picked up the updated snapshot; it is "
        f"serving a cached read. Steps seen: {steps}"
    )
PY

echo "SMOKE PASS"
