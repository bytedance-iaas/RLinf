#!/usr/bin/env bash
# Build the standalone image and verify it against a read-only run mount.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD="$(dirname "$HERE")"
REPO_ROOT="$(dirname "$DASHBOARD")"
IMAGE="${IMAGE:-rlinf-dashboard:smoke}"
PORT="${PORT:-8435}"
AUTH_USER="container-operator"
AUTH_PASSWORD="container-password"
WORK="$(mktemp -d)"
CONTAINER="rlinf-dashboard-smoke-$$"

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

RUN_ROOT="$WORK/_rlinf/runs/container-smoke"
mkdir -p "$RUN_ROOT"
cat > "$RUN_ROOT/manifest.json" <<JSON
{
  "schema_version": 2,
  "run_id": "container-smoke",
  "task_type": "embodied",
  "experiment_name": "container-smoke",
  "heartbeat_interval_s": 5.0,
  "paths": {"log_path": "/runs"}
}
JSON
cat > "$RUN_ROOT/run.json" <<JSON
{
  "schema_version": 2,
  "run_id": "container-smoke",
  "task_type": "embodied",
  "state": "finished",
  "phase": null,
  "heartbeat_at": "2026-08-01T00:00:00Z",
  "heartbeat_seq": 1,
  "last_progress_at": "2026-08-01T00:00:00Z",
  "last_metric_at": "2026-08-01T00:00:00Z",
  "progress": {"step": 1, "max_steps": 1, "step_semantics": "rl_iteration"},
  "timing": {"started_at": "2026-08-01T00:00:00Z", "elapsed_s": 1.0},
  "exit": null
}
JSON
chmod -R a+rX "$WORK"

docker build --file "$DASHBOARD/Dockerfile" --tag "$IMAGE" "$REPO_ROOT"
docker run --detach --rm \
  --name "$CONTAINER" \
  --read-only \
  --tmpfs /tmp \
  --publish "127.0.0.1:${PORT}:8420" \
  --volume "$WORK:/runs:ro" \
  --env "RLINF_DASHBOARD_AUTH_MODE=basic" \
  --env "RLINF_DASHBOARD_AUTH_USERNAME=$AUTH_USER" \
  --env "RLINF_DASHBOARD_AUTH_PASSWORD=$AUTH_PASSWORD" \
  "$IMAGE" >/dev/null

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/healthz" > /dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
curl -fsS "http://127.0.0.1:${PORT}/healthz" > /dev/null \
  || { docker logs "$CONTAINER"; exit 1; }

code="$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/api/health")"
[ "$code" = "401" ] || { docker logs "$CONTAINER"; exit 1; }

curl -fsS --user "$AUTH_USER:$AUTH_PASSWORD" \
  "http://127.0.0.1:${PORT}/api/health" > "$WORK/health.json"

python3 - "$WORK/health.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    body = json.load(handle)
assert body["status"] == "ok", body
assert body["run_count"] == 1, body
PY

curl -fsS --user "$AUTH_USER:$AUTH_PASSWORD" "http://127.0.0.1:${PORT}/" \
  | grep -q '<div id="root"' \
  || { docker logs "$CONTAINER"; exit 1; }

echo "CONTAINER SMOKE PASS"
