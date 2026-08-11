# RLinf Dashboard

RLinf Dashboard is a standalone, read-only service for observing active and
completed RLinf runs. It combines a filesystem control plane with TensorBoard
time series so operators can answer two different questions:

- Is the job alive, advancing, and producing metrics?
- Are reward, success, loss, throughput, and stability moving as expected?

The service does not import `rlinf`. Training environments can keep their own
heavy and sometimes incompatible dependencies while the dashboard runs in a
small, independent Python environment.

For a copy-paste source, wheel, container, and live-run workflow, see
[`QUICKSTART.md`](QUICKSTART.md).

## Architecture

The training process writes run metadata under:

```text
<log_path>/_rlinf/runs/<run_id>/
├── manifest.json
├── run.json
├── events.jsonl
├── checkpoints.jsonl
├── heartbeat
└── media.rank<worker>.jsonl
```

`run.json` follows the frozen
[`run.v2.schema.json`](rlinf_dashboard/schemas/run.v2.schema.json) contract.
The dashboard package carries its own immutable copy of that schema and test
fixtures; RLinf's monorepo tests enforce byte-for-byte parity with the producer.

Metric curves are read from the TensorBoard paths recorded in `manifest.json`.
WandB and SwanLab remain supported training backends, but this service does not
query their remote APIs.

## Install

From the dashboard project directory:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

The package intentionally has no dependency on RLinf or PyTorch.

To build the standalone image from the repository root:

```bash
docker build -f dashboard/Dockerfile -t rlinf-dashboard .
docker run --rm -p 8420:8420 -v /path/to/logs:/runs:ro rlinf-dashboard
```

The container runs as a non-root user and supports a read-only log mount and
read-only root filesystem.

## Run

```bash
.venv/bin/rlinf-dashboard /path/to/logs --port 8420
```

One directory, which can be a `runner.logger.log_path` or an ancestor of
several. Discovery searches for `_rlinf/runs/*/manifest.json` within a bounded
depth, so runs that already share an ancestor are covered by naming it:

```
/mnt/runs                        <- the scan root
├── experiment-a/<run>/_rlinf/…
└── experiment-b/<run>/_rlinf/…
```

Gathering runs that do *not* share a filesystem — from several machines — is a
separate problem with its own questions about identity and freshness, and is not
what a second local path would solve.

Configuration can also be supplied with `RLINF_DASHBOARD_` environment
variables:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `RLINF_DASHBOARD_SCAN_ROOT` | `./logs` | One directory. Point it at the common ancestor when runs live in several |
| `RLINF_DASHBOARD_SCAN_MAX_DEPTH` | `6` | Maximum discovery depth below each root |
| `RLINF_DASHBOARD_HEARTBEAT_TIMEOUT_K` | `5.0` | Missed heartbeat intervals before `unreachable` |
| `RLINF_DASHBOARD_HEARTBEAT_INTERVAL_S` | `5.0` | Fallback interval for older manifests |
| `RLINF_DASHBOARD_PROGRESS_TIMEOUT_K` | `10.0` | Step-time budgets before `degraded` |
| `RLINF_DASHBOARD_TIMEOUT_FLOOR_S` | `30.0` | Minimum progress budget during startup |
| `RLINF_DASHBOARD_SSE_INTERVAL_S` | `2.0` | Live update interval |
| `RLINF_DASHBOARD_DISCOVERY_CACHE_TTL_S` | `5.0` | Run discovery cache lifetime |
| `RLINF_DASHBOARD_MAX_SERIES_POINTS` | `4000` | Per-series response budget |
| `RLINF_DASHBOARD_CORS_ORIGINS` | local Vite origins | Allowed development origins |
| `RLINF_DASHBOARD_FRONTEND_DIST` | auto-detected | Optional external frontend bundle |

For example:

```bash
RLINF_DASHBOARD_SCAN_ROOT=/mnt/runs \
  .venv/bin/rlinf-dashboard --host 0.0.0.0 --port 8420
```

## Run and health semantics

`state` is a fact written by the training process: `pending`, `running`,
`finished`, `failed`, or `stopped`. `health` is derived by the reader and is
never persisted:

- `healthy`: heartbeat, progress, and metrics are current, or the run is
  terminal.
- `degraded`: the process is alive but progress or metric publication is stale.
- `unreachable`: the configured heartbeat budget has expired.
- `unknown`: no readable snapshot or heartbeat evidence is available.

Process liveness uses the writer's `heartbeat_interval_s` from the manifest.
Progress and metric staleness use `max(step_time_p50, timeout_floor_s)`, because
valid step duration varies substantially across reasoning and embodied jobs.

## HTTP API

| Path | Returns |
| --- | --- |
| `GET /api/health` | Service status and scan-root diagnostics |
| `GET /api/runs` | Filterable run summaries |
| `GET /api/runs/{run_id}` | Manifest, snapshot, health, and capabilities |
| `GET /api/runs/{run_id}/events` | Paginated lifecycle and phase events |
| `GET /api/runs/{run_id}/checkpoints` | Completed checkpoints, newest first |
| `GET /api/runs/{run_id}/keys` | Available aggregate and per-worker metrics |
| `GET /api/runs/{run_id}/series` | Aggregate or per-worker metric series |
| `GET /api/runs/{run_id}/template` | Metric layout bound to available keys |
| `GET /api/templates` | All loaded metric templates |
| `GET /api/runs/{run_id}/media` | Filterable media index |
| `GET /api/runs/{run_id}/media/file` | Allowlisted media streaming with ranges |
| `GET /api/stream/runs` | Live run summaries over SSE |
| `GET /api/stream/runs/{run_id}` | Live status for one run over SSE |

OpenAPI documentation is available at `/docs`.

Long metric series use an extrema-preserving min/max envelope. First, last,
non-finite, minimum, and maximum samples receive explicit priority so a short
response does not hide a loss spike or divergence signal.

Media paths are treated as untrusted. A file is served only when its exact path
appears in that run's media index and its suffix is allowlisted.

## Browser console

The React console provides run list, overview, metrics, media, events, and
cross-run comparison views. Metric layouts are selected by server-side YAML
templates for embodied RL, reasoning/coding RL, and SFT runs.

For local frontend development:

```bash
cd frontend
npm install
npm run typecheck
npm run check:scales
npm run dev
```

The Vite server proxies `/api` to the Python service on port 8420. Production
builds are served from the same origin by the Python package. See
[`frontend/README.md`](frontend/README.md) for design and verification details.

## Tests

From this directory:

```bash
.venv/bin/python -m pytest tests -q
bash tests/smoke_server.sh .venv/bin/python
bash tests/smoke_wheel.sh .venv/bin/python
bash tests/smoke_container.sh
```

The server smoke test creates a synthetic run tree, launches uvicorn, checks the
core REST endpoints, and verifies live SSE delivery. The wheel smoke test builds
the frontend, installs the wheel in a clean environment, and checks its UI,
schema, license, assets, and deep links. The container smoke test mounts a
synthetic run tree read-only and checks both API and browser shell.

## Current integration constraint

Use a separate `runner.logger.log_path` for concurrent launches. TensorBoard and
per-worker metric paths are not yet run-scoped, so reusing one log path can merge
time series even though control-plane run IDs remain distinct.
