# RLinf Dashboard

A control-plane view of RLinf training runs: what state each run is in, whether it
is still alive, which phase it is in, how far it has got, when it last checkpointed
— plus the metric curves, laid out by a per-task-type template.

TensorBoard, wandb and swanlab are the **data plane**; they answer "what are the
numbers". This is the **control plane**: it answers "should I be worried". The two
are complementary, and this service reads from the former rather than replacing it.

Two halves, documented separately: this file covers the server (a JSON API, useful
on its own — `curl` and `/docs` are a complete interface to it), and
[`frontend/README.md`](frontend/README.md) covers the browser console. The server
mounts the built frontend when `frontend/dist/` exists and works fine when it does
not.

## Why it does not import `rlinf`

RLinf training environments are many and heavy — isaac-sim, omnigibson and sglang
each get their own venv (`requirements/install.sh`,
`cluster.env_configs.python_interpreter_path`). A dashboard that imported `rlinf`
would have to be installed *into* one of those venvs and would inherit its solver
constraints.

So the cross-process contract is only the filesystem layout under
`<log_path>/_rlinf/runs/<run_id>/`, frozen in
[`docs/schemas/run.v2.schema.json`](../docs/schemas/run.v2.schema.json), plus the
TensorBoard event files the run already writes.

A CI job enforces this: it installs this package into a venv where `import rlinf`
fails and runs the whole suite plus a live server there. `dashboard/tests/` also
has a `no_rlinf` fixture that turns a stray import into a test failure on a laptop
rather than only in CI.

## Install

```bash
python3 -m venv /tmp/rlinf-dash          # any venv; deliberately NOT a training one
/tmp/rlinf-dash/bin/pip install -e 'dashboard[test]'
```

Series come from the run's TensorBoard event files, so nothing else needs to be
running for metrics to show up.

## Run

```bash
/tmp/rlinf-dash/bin/rlinf-dashboard /path/to/logs
# or
/tmp/rlinf-dash/bin/python -m rlinf_dashboard /path/to/logs --port 8420
```

Each positional argument is a directory to scan — either a `runner.logger.log_path`
or an ancestor of several. Discovery walks for `_rlinf/runs/*/manifest.json`, so
pointing at a parent of many experiments works and is the normal case.

Configuration is also available through the environment, prefix
`RLINF_DASHBOARD_` (see `rlinf_dashboard/settings.py`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `RLINF_DASHBOARD_SCAN_ROOTS` | `./logs` | Roots to scan: one path, comma-separated paths, or a JSON array |
| `RLINF_DASHBOARD_HEARTBEAT_TIMEOUT_K` | `5.0` | Heartbeat budget, in multiples of the run's own `step_time_p50` |
| `RLINF_DASHBOARD_PROGRESS_TIMEOUT_K` | `10.0` | Same, for step progress and metric writes |
| `RLINF_DASHBOARD_TIMEOUT_FLOOR_S` | `30.0` | Floor on that budget, so a run with no step samples yet is not called dead |
| `RLINF_DASHBOARD_SSE_INTERVAL_S` | `2.0` | Push period for the SSE streams |

The timeouts are multiples of the run's *own* median step time rather than fixed
seconds because verified step times span two orders of magnitude — about 2s for
reasoning, 428s for Pi0.5 on 4×H20. One fixed threshold cannot serve both.

## Endpoints

| Path | Returns |
| --- | --- |
| `GET /api/health` | This server's status, plus each scan root and whether it exists |
| `GET /api/runs` | One summary per discovered run; `?state=`, `?task_type=`, `?refresh=true` |
| `GET /api/runs/{run_id}` | Manifest, snapshot and derived health |
| `GET /api/runs/{run_id}/events` | Lifecycle and phase events, newest last |
| `GET /api/runs/{run_id}/checkpoints` | Completed checkpoints, newest first |
| `GET /api/runs/{run_id}/keys` | Metric keys available, and which sources answered |
| `GET /api/runs/{run_id}/series?keys=a&keys=b` | Point series, decimated with the reduction reported |
| `GET /api/runs/{run_id}/template` | Chart layout, already bound to the keys this run actually has |
| `GET /api/templates` | Every template, for debugging selection |
| `GET /api/runs/{run_id}/media` | Recorded clips; `?split=`, `?step=`, `?min_step=`, `?max_step=` |
| `GET /api/runs/{run_id}/media/file?path=…` | Streams one clip, `Range` supported |
| `GET /api/stream/runs` | SSE: all run summaries, every `sse_interval_s` |
| `GET /api/stream/runs/{run_id}` | SSE: one run's full status |

Interactive docs at `/docs`.

`health` is **derived on read and never persisted** — it is a pure function of the
snapshot's three timestamps and the clock. `state` is what the training process
wrote; `health` is what the reader concludes. That distinction is what lets a
`kill -9`'d run — which leaves `state: running` behind forever, because a dead
process cannot record its own death — still show up as `unreachable`.

## Manual verification

Requires a run tree. Either a real one, or the committed fixtures.

For the browser console's own steps — eleven fixture runs covering every health
verdict and run state, and what each view must show — see
[`frontend/README.md`](frontend/README.md#manual-verification). The checks below
are server-only and need no `npm`.

### 1. One command, end to end (no training needed)

```bash
bash dashboard/tests/smoke_server.sh /tmp/rlinf-dash/bin/python
```

Builds a synthetic run tree, boots a real uvicorn against it, and checks every
endpoint plus the SSE stream — including that the stream picks up a snapshot
rewritten underneath it. Prints `SMOKE PASS`, or the failing check and the server
log. This is what CI runs, and it is the fastest way to confirm an install works.

It covers what pytest cannot: `TestClient` runs the app through a synchronous
portal that waits for the response generator on close, so an endpoint that streams
forever hangs the test process. `test_api.py` therefore drives the SSE handlers
directly, and chunked delivery is verified only here.

### 2. Against the committed fixtures

```bash
mkdir -p /tmp/vrun/_rlinf/runs/fixture-run
cp tests/fixtures/run_state/running.json /tmp/vrun/_rlinf/runs/fixture-run/run.json
cat > /tmp/vrun/_rlinf/runs/fixture-run/manifest.json <<'JSON'
{"schema_version": 2, "run_id": "fixture-run", "task_type": "embodied",
 "experiment_name": "fixture", "step_semantics": "rl_iteration",
 "paths": {"log_path": "/tmp/vrun"}}
JSON

/tmp/rlinf-dash/bin/rlinf-dashboard /tmp/vrun &
curl -s localhost:8420/api/health | python3 -m json.tool
curl -s localhost:8420/api/runs   | python3 -m json.tool
```

Expect `run_count: 1`, and a summary whose `health` is `unreachable` — the fixture's
timestamps are from when it was recorded, so by now its heartbeat has long expired.
That is the correct answer, and it is the quickest way to see that the derivation
is running rather than defaulting to green. (Swap in `hung_training.json` for
`degraded`, `finished.json` for a terminal run that stays `healthy`.)

### 3. Against a live training run

```bash
# In the training venv, on the head node:
bash examples/embodiment/run_embodiment.sh <config_name>   # runner.max_steps=3 is enough

# Anywhere with read access to the log path:
/tmp/rlinf-dash/bin/rlinf-dashboard <log_path>
curl -s localhost:8420/api/runs | python3 -m json.tool
curl -N localhost:8420/api/stream/runs                     # watch it tick
```

Check, in order:

1. `state` walks `pending → running → finished`.
2. `phase` walks the pipeline — for a sync embodied run,
   `sync_weights → rollout → cal_adv_and_returns → train → save_ckpt`.
3. `health` stays `healthy` throughout, and stays `healthy` after the run finishes
   (a terminal run is not unhealthy for having stopped talking).
4. `progress.max_steps` matches the *effective* horizon
   `min(num_steps_per_epoch × max_epochs, runner.max_steps)`, not the config cap.
5. `/api/runs/{id}/checkpoints` gains a row with a real `size_bytes` and
   `duration_s` after the first save.
6. `/api/runs/{id}/series?keys=env/success_once` returns points.

Then the failure paths, which are the ones worth trusting:

```bash
kill -9 <driver_pid>     # health -> unreachable, state stays "running"
kill -STOP <driver_pid>  # health -> unreachable or degraded; see below
```

`SIGSTOP` freezes the heartbeat thread too, so it reads as the process being gone.
The `degraded` verdict is for the case a stopped process cannot produce: heartbeat
fresh, progress stale — a driver blocked in an NCCL collective, whose daemon thread
keeps ticking while no step ever completes. To see that one, use the
`hung_training` fixture.

For an async run (`train_async.py`), also check `components`: `env`, `rollout` and
`actor` should all be `active: true` at once during the loop, and all flip to
`active: false` with `since` preserved at the end. A single scalar `phase` is
semantically wrong for those runners, which is why the field exists.

## Tests

```bash
/tmp/rlinf-dash/bin/python -m pytest dashboard/tests -q
bash dashboard/tests/smoke_server.sh /tmp/rlinf-dash/bin/python
```

`dashboard/tests/conftest.py` reads the JSON Schema and the run fixtures from the
repository root rather than keeping copies. `tests/unit_tests/` on the training
side reads the same bytes, so a change that satisfies one side and breaks the other
cannot pass both suites — which is what keeps the two independent models from
drifting apart without shared code.

CI (`.github/workflows/dashboard-tests.yml`) runs both of the above in a venv that
has never seen `rlinf` or `torch`, on Python 3.10 and 3.13, and asserts that
`import rlinf` fails there. It runs those steps from `/tmp`: at the repository root
the sibling `rlinf/` directory is importable through `sys.path[0]`, so the assertion
would pass for the wrong reason.

The same workflow has two frontend jobs: `design-system` regenerates
`frontend/src/styles/tokens.css` from `frontend/DESIGN.md` and fails if the
committed file differs, and `frontend-build` typechecks, builds, and asserts the
built `index.html` references only assets the build emitted. There is no frontend
test framework — with no server and no logic of its own, the compiler and the
bundler are what catch a real regression there.
