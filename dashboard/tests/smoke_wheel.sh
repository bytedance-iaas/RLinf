#!/usr/bin/env bash
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Build a wheel, install it into a venv that has never seen this source tree,
# and assert the served page is real: root returns HTML, and every asset that
# HTML references returns 200.
#
# This exists because `pip install rlinf-dashboard` used to produce an API with
# no UI. `frontend/dist` sits beside the package rather than inside it, so it was
# never wheel data, and the only symptom was a debug-level log line. An editable
# install worked, which is why the tests, the smoke server and every manual check
# passed while the shipped artifact was broken.
#
# Usage: bash dashboard/tests/smoke_wheel.sh [python]
set -euo pipefail

PYTHON="${1:-python3}"
PORT="${PORT:-8434}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD="$(dirname "$HERE")"
WORK="$(mktemp -d)"
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill -9 "$SERVER_PID" 2>/dev/null || true
    # Reap it, or the shell prints "Killed: 9" after the pass line and the run
    # looks like it failed at the last moment.
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  echo "WHEEL SMOKE FAIL: $*" >&2
  [ -f "$WORK/server.log" ] && tail -30 "$WORK/server.log" >&2
  exit 1
}

# ---------------------------------------------------------------- build inputs
command -v npm > /dev/null || fail "npm is required to build the frontend"
echo "Building the frontend..."
npm --prefix "$DASHBOARD/frontend" run build > "$WORK/build.log" 2>&1 \
  || { cat "$WORK/build.log" >&2; fail "npm run build failed"; }

"$PYTHON" "$DASHBOARD/scripts/bundle_frontend.py" \
  || fail "bundle_frontend.py failed"
"$PYTHON" "$DASHBOARD/scripts/bundle_frontend.py" --check \
  || fail "bundled frontend is stale immediately after bundling"

# ------------------------------------------------------------------ build wheel
echo "Building the wheel..."
"$PYTHON" -m pip wheel --no-deps -w "$WORK/wheel" "$DASHBOARD" > "$WORK/wheel.log" 2>&1 \
  || { cat "$WORK/wheel.log" >&2; fail "pip wheel failed"; }

WHEEL="$(ls "$WORK"/wheel/rlinf_dashboard-*.whl 2>/dev/null | head -1)"
[ -n "$WHEEL" ] || fail "no wheel produced"

# Assert on the archive before installing: a wheel missing index.html cannot
# serve a UI no matter how it is installed, and this names the cause directly.
"$PYTHON" - "$WHEEL" <<'PY' || exit 1
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
static = [n for n in names if "/static/" in n]
if not any(n.endswith("rlinf_dashboard/static/index.html") for n in static):
    print("WHEEL SMOKE FAIL: wheel contains no rlinf_dashboard/static/index.html", file=sys.stderr)
    print(f"  {len(names)} entries, {len(static)} under static/", file=sys.stderr)
    raise SystemExit(1)
if not any("/static/assets/" in n for n in static):
    print("WHEEL SMOKE FAIL: wheel has index.html but no assets/", file=sys.stderr)
    raise SystemExit(1)
print(f"wheel carries {len(static)} static file(s)")
PY

# -------------------------------------------------------------- install & serve
echo "Installing into a clean venv..."
"$PYTHON" -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install --quiet "$WHEEL" > "$WORK/install.log" 2>&1 \
  || { cat "$WORK/install.log" >&2; fail "pip install failed"; }

# Run from / so the source tree is not importable through sys.path[0]; otherwise
# the installed package could resolve its frontend from the checkout and pass for
# the very reason this test exists to rule out.
mkdir -p "$WORK/root"
cd /
"$WORK/venv/bin/rlinf-dashboard" "$WORK/root" --host 127.0.0.1 --port "$PORT" \
  > "$WORK/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 40); do
  if curl -fsS "http://127.0.0.1:$PORT/api/health" > /dev/null 2>&1; then break; fi
  sleep 0.5
done
curl -fsS "http://127.0.0.1:$PORT/api/health" > /dev/null || fail "server never became ready"

# --------------------------------------------------------------------- assert
curl -fsS "http://127.0.0.1:$PORT/" -o "$WORK/index.html" || fail "GET / failed"
grep -qi "<div id=\"root\"" "$WORK/index.html" \
  || fail "GET / did not return the app shell"

# Every asset the page references must resolve. A wheel can carry index.html and
# miss the assets directory, which renders as a blank page with a 404 in the
# console -- indistinguishable from a broken build unless it is checked here.
ASSETS="$(grep -o '/assets/[A-Za-z0-9_.-]*' "$WORK/index.html" | sort -u)"
[ -n "$ASSETS" ] || fail "index.html references no /assets/ files"
for asset in $ASSETS; do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT$asset")"
  [ "$code" = "200" ] || fail "asset $asset returned $code"
done
echo "verified $(echo "$ASSETS" | wc -w | tr -d ' ') referenced asset(s)"

# A deep link must serve the shell too, or a reload on any route 404s.
curl -fsS "http://127.0.0.1:$PORT/runs/does-not-exist/metrics" -o "$WORK/deep.html" \
  || fail "deep link failed"
grep -qi "<div id=\"root\"" "$WORK/deep.html" || fail "deep link did not return the shell"

# ...but the API must still fail as an API.
code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/nope")"
[ "$code" = "404" ] || fail "unknown API endpoint returned $code, expected 404"

echo "WHEEL SMOKE PASS"
