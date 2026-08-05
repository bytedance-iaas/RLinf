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

"""Contract tests for the run control-plane snapshot (``run.json``, v2).

``docs/schemas/run.v2.schema.json`` is the single source of truth shared by the
training side (which writes ``run.json``) and the dashboard (which reads it
without importing ``rlinf``). Neither side can import the other, so these tests
are what keeps the two from drifting.

Covered here:

* every committed fixture validates against the schema;
* the four states a reader must distinguish -- ``running`` / ``finished`` /
  ``failed`` / heartbeat-expired;
* the read-side ``health`` derivation, including the case a single heartbeat
  cannot express: process alive but training thread hung;
* ``components``, which is how an async run reports env/rollout/actor running
  concurrently;
* the invariants the schema itself enforces (no ``stalled`` state, ``failed``
  must carry ``exit``).

The fixtures live in ``tests/fixtures/run_state/`` and are shared verbatim with
``dashboard/tests/`` once that package exists, so both sides parse identical
bytes.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

jsonschema = pytest.importorskip(
    "jsonschema",
    reason="jsonschema ships as an unconditional ray[default] dependency; skip if absent",
)

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SCHEMA_PATH = os.path.join(_REPO_ROOT, "docs", "schemas", "run.v2.schema.json")
FIXTURE_DIR = os.path.join(_REPO_ROOT, "tests", "fixtures", "run_state")

# A reader flags a run once it has been silent for this multiple of its own step
# time. Kept here rather than in the schema: it is read-side policy, not contract.
HEARTBEAT_TIMEOUT_K = 5
PROGRESS_TIMEOUT_K = 10
TIMEOUT_FLOOR_S = 30.0


def _load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def _load_fixture(name):
    with open(os.path.join(FIXTURE_DIR, f"{name}.json")) as f:
        return json.load(f)


def _fixture_names():
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(FIXTURE_DIR) if f.endswith(".json")
    )


def _parse_ts(value):
    """Parse an ISO-8601 timestamp, accepting the ``Z`` suffix the writer emits."""
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def derive_health(run, now):
    """Classify liveness from the three v2 timestamps.

    This mirrors the pure function the dashboard will expose as ``health.py``.
    It is duplicated here on purpose: the dashboard cannot import ``rlinf``, so
    the contract -- not shared code -- is what binds the two implementations.
    A copy that drifts from the fixtures fails these tests.

    Args:
        run: Parsed ``run.json`` mapping.
        now: Reader-side wall clock as an aware ``datetime``.

    Returns:
        One of ``"healthy"``, ``"degraded"``, ``"unreachable"``. Terminal states
        report ``"healthy"``: a finished run is not unhealthy for being silent.
    """
    if run["state"] in ("finished", "failed", "stopped"):
        return "healthy"

    step_time = run.get("timing", {}).get("step_time_p50") or 0.0
    budget = max(step_time, TIMEOUT_FLOOR_S)

    # Process-level liveness first: if the heartbeat thread is gone, nothing else
    # can be trusted.
    if (
        now - _parse_ts(run["heartbeat_at"])
    ).total_seconds() > HEARTBEAT_TIMEOUT_K * budget:
        return "unreachable"

    # Heartbeat fresh but no step advance -- the hung-training-thread signature
    # that a single heartbeat reports as healthy.
    last_progress = _parse_ts(run.get("last_progress_at"))
    if (
        last_progress is not None
        and (now - last_progress).total_seconds() > PROGRESS_TIMEOUT_K * budget
    ):
        return "degraded"

    last_metric = _parse_ts(run.get("last_metric_at"))
    if (
        last_metric is not None
        and (now - last_metric).total_seconds() > PROGRESS_TIMEOUT_K * budget
    ):
        return "degraded"

    return "healthy"


@pytest.mark.parametrize("name", _fixture_names())
def test_fixture_validates_against_schema(name):
    """Every fixture is a legal v2 snapshot."""
    jsonschema.validate(instance=_load_fixture(name), schema=_load_schema())


def test_schema_is_valid_json_schema():
    schema = _load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["properties"]["schema_version"]["const"] == 2


@pytest.mark.parametrize(
    ("name", "expected_state"),
    [
        ("running", "running"),
        ("finished", "finished"),
        ("failed", "failed"),
        ("heartbeat_expired", "running"),
    ],
)
def test_state_parses(name, expected_state):
    """The four reader-visible cases.

    ``heartbeat_expired`` still says ``running`` on disk -- that is the point.
    A killed driver cannot update its own state, so liveness is a read-side
    judgement, never a persisted field.
    """
    assert _load_fixture(name)["state"] == expected_state


def test_state_enum_excludes_stalled():
    """``stalled`` must not be expressible; see architecture constraint 4."""
    states = _load_schema()["properties"]["state"]["enum"]
    assert "stalled" not in states
    assert set(states) == {"pending", "running", "finished", "failed", "stopped"}


def test_failed_run_records_a_reason():
    run = _load_fixture("failed")
    assert run["exit"]["reason"]
    assert run["exit"]["traceback_tail"]


def test_schema_rejects_failed_without_exit():
    """The schema, not just convention, requires a reason for a failure."""
    run = _load_fixture("failed")
    run["exit"] = None
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=run, schema=_load_schema())


def test_schema_rejects_running_with_exit():
    run = _load_fixture("running")
    run["exit"] = {"reason": "boom"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=run, schema=_load_schema())


def test_schema_rejects_unknown_state():
    run = _load_fixture("running")
    run["state"] = "stalled"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=run, schema=_load_schema())


def test_healthy_run_is_healthy():
    run = _load_fixture("running")
    now = _parse_ts(run["heartbeat_at"]) + timedelta(seconds=2)
    assert derive_health(run, now) == "healthy"


def test_expired_heartbeat_is_unreachable():
    """``kill -9`` on the driver: nothing updates, so the reader must decide."""
    run = _load_fixture("heartbeat_expired")
    now = datetime(2026, 8, 3, 15, 0, 0, tzinfo=timezone.utc)
    assert derive_health(run, now) == "unreachable"


def test_fresh_heartbeat_with_stale_progress_is_degraded():
    """A1's payoff: the hung training thread a single heartbeat calls healthy.

    The heartbeat thread is a daemon independent of the training loop, so an
    NCCL hang or stuck env leaves the process happily ticking while no step
    advances.
    """
    run = _load_fixture("hung_training")
    now = _parse_ts(run["heartbeat_at"]) + timedelta(seconds=2)

    assert (now - _parse_ts(run["heartbeat_at"])).total_seconds() < 60, (
        "fixture must have a fresh heartbeat, else it proves nothing"
    )
    assert derive_health(run, now) == "degraded"


def test_stale_metric_path_is_degraded():
    """Steps advance but nothing reaches the metric backend."""
    run = _load_fixture("running")
    run["last_metric_at"] = "2026-08-03T13:15:00Z"
    now = _parse_ts(run["heartbeat_at"]) + timedelta(seconds=2)
    assert derive_health(run, now) == "degraded"


def test_terminal_states_are_never_unhealthy():
    """A finished run is silent by definition; silence must not read as failure."""
    run = _load_fixture("finished")
    now = _parse_ts(run["heartbeat_at"]) + timedelta(days=30)
    assert derive_health(run, now) == "healthy"


def test_async_run_reports_concurrent_components():
    """A2: env/rollout/actor overlap for the whole loop in async runners.

    A single scalar ``phase`` cannot express this, which is why ``components``
    exists alongside it.
    """
    run = _load_fixture("async_components")
    components = run["components"]

    assert {"env", "rollout", "actor"} <= set(components)
    assert all(components[name]["active"] for name in ("env", "rollout", "actor")), (
        "all three components run concurrently in an async runner"
    )
    for name in ("env", "rollout", "actor"):
        assert _parse_ts(components[name]["since"]) is not None


def test_sync_run_leaves_components_empty():
    run = _load_fixture("running")
    assert not run.get("components")


def test_step_semantics_is_declared():
    """A7: label the x-axis honestly rather than silently unify it."""
    assert _load_fixture("running")["progress"]["step_semantics"] == "rl_iteration"
    assert (
        _load_fixture("reasoning_minibatch")["progress"]["step_semantics"]
        == "minibatch"
    )


def test_task_type_enum_matches_config():
    """Drift guard: the schema enum must track ``SUPPORTED_TASK_TYPE``.

    The import is skipped rather than allowed to fail when the heavy stack is
    absent: ``rlinf.config`` pulls in ``ray`` and ``torch``, and every other test
    here reads only fixtures and the schema. Letting this one hard-fail would
    make the command the docs tell people to run -- ``pytest
    tests/unit_tests/test_run_state_contract.py`` -- report a failure that says
    nothing about the contract.
    """
    pytest.importorskip("ray", reason="rlinf.config imports ray at module scope")
    pytest.importorskip("torch", reason="rlinf.config imports torch at module scope")
    from rlinf.config import SUPPORTED_TASK_TYPE

    schema_types = _load_schema()["properties"]["task_type"]["enum"]
    assert sorted(schema_types) == sorted(SUPPORTED_TASK_TYPE)


def test_checkpoint_entry_is_structured_not_a_command_string():
    """A6: store fields, not a pre-baked shell string that can go stale."""
    latest = _load_fixture("running")["latest_checkpoint"]
    assert "resume_cmd" not in latest
    assert latest["resume_dir"]
    assert latest["entry_script"]
    assert latest["config_name"]
