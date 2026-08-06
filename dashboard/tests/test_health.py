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

"""Tests for the liveness derivation.

``derive_health`` is a pure function of a snapshot and a clock, so every case
here is a timestamp matrix -- no filesystem, no server. That is the reason it was
extracted: the interesting failures are all about time, and testing them through
HTTP would mean sleeping.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from conftest import fixture_names, load_fixture

from rlinf_dashboard.health import derive_health
from rlinf_dashboard.models import Health, RunSnapshot

NOW = datetime(2026, 8, 3, 15, 0, 0, tzinfo=timezone.utc)


def snapshot(**overrides) -> RunSnapshot:
    """A healthy running snapshot, minus whatever the test overrides."""
    base = {
        "schema_version": 2,
        "run_id": "20260803-150000-test",
        "task_type": "embodied",
        "state": "running",
        "phase": "rollout",
        "heartbeat_at": NOW - timedelta(seconds=3),
        "heartbeat_seq": 100,
        "last_progress_at": NOW - timedelta(seconds=20),
        "last_metric_at": NOW - timedelta(seconds=20),
        "progress": {"step": 10, "max_steps": 100, "step_semantics": "rl_iteration"},
        "timing": {
            "started_at": NOW - timedelta(seconds=400),
            "elapsed_s": 400.0,
            "step_time_p50": 40.0,
        },
    }
    base.update(overrides)
    return RunSnapshot.model_validate(base)


def test_all_current_is_healthy():
    assert derive_health(snapshot(), NOW).health == Health.HEALTHY


def test_missing_snapshot_is_unknown():
    """A run directory with a manifest but no snapshot is not a dead run.

    It is a run that has not published yet. Calling that unreachable would send
    someone chasing a process that is still starting up.
    """
    verdict = derive_health(None, NOW)
    assert verdict.health == Health.UNKNOWN
    assert "run.json" in verdict.reason


def test_dead_heartbeat_is_unreachable():
    """Enough missed ticks means the process is gone.

    This is the case ``state`` cannot express: a ``kill -9``'d writer leaves
    ``state: running`` behind forever, because a dead process cannot record its
    own death.
    """
    stale = snapshot(heartbeat_at=NOW - timedelta(seconds=40 * 5 + 1))
    verdict = derive_health(stale, NOW)
    assert verdict.health == Health.UNREACHABLE
    assert stale.state.value == "running", "the write-side state is untouched"


def test_fresh_heartbeat_with_stale_progress_is_degraded():
    """The whole reason the model has three timestamps instead of one.

    A driver blocked in an NCCL collective keeps its daemon heartbeat thread
    ticking while no step ever completes. A single heartbeat calls that healthy.
    Comparing heartbeat freshness against progress freshness is what makes the
    hang visible.
    """
    hung = snapshot(
        heartbeat_at=NOW - timedelta(seconds=2),
        last_progress_at=NOW - timedelta(seconds=40 * 10 + 1),
        last_metric_at=NOW - timedelta(seconds=40 * 10 + 1),
    )
    verdict = derive_health(hung, NOW)
    assert verdict.health == Health.DEGRADED
    assert "hung" in verdict.reason


def test_stale_metrics_alone_is_degraded():
    """Steps advancing but nothing reaching a backend: charts are blind."""
    blind = snapshot(
        last_progress_at=NOW - timedelta(seconds=5),
        last_metric_at=NOW - timedelta(seconds=40 * 10 + 1),
    )
    verdict = derive_health(blind, NOW)
    assert verdict.health == Health.DEGRADED
    assert "metric" in verdict.reason


def test_heartbeat_loss_outranks_progress_loss():
    """When both are stale, report the process being gone.

    Both checks would fire; ordering matters because "the process died" and "the
    process is hung" call for different actions, and the first is the one that
    happened.
    """
    both = snapshot(
        heartbeat_at=NOW - timedelta(hours=1),
        last_progress_at=NOW - timedelta(hours=1),
    )
    assert derive_health(both, NOW).health == Health.UNREACHABLE


@pytest.mark.parametrize("state", ["finished", "failed", "stopped"])
def test_terminal_states_are_healthy_however_old(state):
    """A finished run is not unhealthy for having stopped talking.

    Without this, every completed run in the list turns red overnight and the
    health column stops meaning anything.
    """
    done = snapshot(
        state=state,
        heartbeat_at=NOW - timedelta(days=30),
        last_progress_at=NOW - timedelta(days=30),
        exit=({"reason": "boom"} if state != "finished" else None),
    )
    assert derive_health(done, NOW).health == Health.HEALTHY


def test_pending_run_is_healthy():
    assert derive_health(snapshot(state="pending"), NOW).health == Health.HEALTHY


def test_running_without_a_heartbeat_is_unknown():
    """Absent is not the same as fresh.

    Defaulting a missing ``heartbeat_at`` to "now" would report a snapshot that
    never ticked as perfectly healthy.
    """
    verdict = derive_health(snapshot(heartbeat_at=None), NOW)
    assert verdict.health == Health.UNKNOWN


def test_budget_floor_protects_slow_startup():
    """With no step-time samples, the floor -- not zero -- is the *step* budget.

    Startup legitimately takes minutes (sglang warmup, simulator boot), and a
    zero budget would flag every run as hung within a second of launch. This is
    about progress, not liveness: a run this new has a fresh heartbeat.
    """
    starting = snapshot(
        heartbeat_at=NOW - timedelta(seconds=2),
        last_progress_at=None,
        last_metric_at=None,
        timing={"started_at": NOW - timedelta(seconds=100), "elapsed_s": 100.0},
    )
    verdict = derive_health(starting, NOW, timeout_floor_s=30.0)
    assert verdict.budget_s == 30.0
    assert verdict.health == Health.HEALTHY


def test_progress_staleness_scales_with_the_run_s_own_step_time():
    """A fixed progress threshold cannot work across RLinf.

    Verified step times span two orders of magnitude: seconds for reasoning,
    428s for Pi0.5 on 4xH20. The same 300s without a completed step is a hang in
    one and less than one normal step in the other -- which is why *this* budget
    is a multiple of the run's own p50.
    """
    stalled = timedelta(seconds=3000)
    fast = snapshot(
        heartbeat_at=NOW - timedelta(seconds=2),
        last_progress_at=NOW - stalled,
        timing={"started_at": NOW, "elapsed_s": 1.0, "step_time_p50": 2.0},
    )
    slow = snapshot(
        heartbeat_at=NOW - timedelta(seconds=2),
        last_progress_at=NOW - stalled,
        timing={"started_at": NOW, "elapsed_s": 1.0, "step_time_p50": 428.0},
    )
    assert derive_health(fast, NOW).health == Health.DEGRADED
    assert derive_health(slow, NOW).health == Health.HEALTHY


def test_liveness_does_not_scale_with_step_time():
    """A dead driver must be called dead on the same schedule for every run.

    This is the one behaviour in this file that was deliberately reversed. It
    used to assert that a run with a 428s step could go 300s without a heartbeat
    and still read `healthy`, on the theory that a slow run deserves a slow
    timeout. It does not: the heartbeat is a fixed 5s tick from a daemon thread
    that does not wait for a step, so 300s of silence is 60 missed ticks whatever
    the step time is. At the old `5 x max(p50, 30)`, killing the driver of a
    428s/step VLA job left it reading `healthy` for 36 minutes -- and "is the
    process alive" is the single question this page exists to answer.
    """
    silence = timedelta(seconds=300)
    for p50 in (2.0, 428.0):
        wedged = snapshot(
            heartbeat_at=NOW - silence,
            timing={"started_at": NOW, "elapsed_s": 1.0, "step_time_p50": p50},
        )
        verdict = derive_health(wedged, NOW)
        assert verdict.health == Health.UNREACHABLE, (
            f"a driver silent for {silence.seconds}s reads as {verdict.health} "
            f"when p50={p50}s; step time must not buy a dead process more time"
        )


def test_a_killed_driver_is_detected_within_about_half_a_minute():
    """The window a user actually experiences, stated as a number.

    Five missed 5s ticks, floored at 15s: the run flips well inside a minute
    rather than inside an hour.
    """
    slow_steps = {"started_at": NOW, "elapsed_s": 1.0, "step_time_p50": 428.0}
    still_ok = snapshot(heartbeat_at=NOW - timedelta(seconds=20), timing=slow_steps)
    assert derive_health(still_ok, NOW).health == Health.HEALTHY

    gone = snapshot(heartbeat_at=NOW - timedelta(seconds=45), timing=slow_steps)
    assert derive_health(gone, NOW).health == Health.UNREACHABLE


def test_a_slower_configured_heartbeat_widens_only_the_liveness_budget():
    """A deployment that ticks slower must be judged against its own period.

    The v2 snapshot has no field for the writer's interval, so this is the
    reader's setting -- and it has to be honoured, or raising the training-side
    interval would make every run read as unreachable.
    """
    quiet = snapshot(
        heartbeat_at=NOW - timedelta(seconds=100),
        timing={"started_at": NOW, "elapsed_s": 1.0, "step_time_p50": 2.0},
    )
    assert derive_health(quiet, NOW).health == Health.UNREACHABLE
    assert derive_health(quiet, NOW, heartbeat_interval_s=60.0).health == Health.HEALTHY


def test_verdict_carries_its_evidence():
    """The UI must be able to explain a badge, not just show one."""
    verdict = derive_health(snapshot(), NOW)
    assert verdict.heartbeat_age_s == pytest.approx(3.0, abs=0.01)
    assert verdict.progress_age_s == pytest.approx(20.0, abs=0.01)
    assert verdict.budget_s == 40.0
    assert verdict.reason


def test_naive_timestamps_are_read_as_utc():
    """A snapshot hand-edited to drop the ``Z`` must not crash the reader."""
    naive = snapshot(heartbeat_at=datetime(2026, 8, 3, 14, 59, 57))
    assert derive_health(naive, NOW).health == Health.HEALTHY


# ---------------------------------------------------------------- shared fixtures


@pytest.mark.parametrize("name", fixture_names())
def test_every_shared_fixture_classifies(name):
    """Each committed fixture yields a verdict from this implementation.

    The fixtures are shared verbatim with ``tests/unit_tests/`` on the training
    side, which asserts the same classifications through its own copy of this
    logic. Running both over one set of bytes is what keeps the two from drifting
    -- the dashboard cannot import ``rlinf``, so the contract, not shared code,
    is the binding.
    """
    snap = RunSnapshot.model_validate(load_fixture(name))
    verdict = derive_health(snap, _fixture_now(snap))
    assert verdict.health in set(Health)
    assert verdict.reason


def _fixture_now(snap: RunSnapshot) -> datetime:
    """A clock a few seconds after the fixture's newest timestamp."""
    stamps = [
        stamp
        for stamp in (snap.heartbeat_at, snap.last_progress_at, snap.last_metric_at)
        if stamp is not None
    ]
    if not stamps:
        return NOW
    newest = max(stamps)
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=timezone.utc)
    return newest + timedelta(seconds=5)


def test_hung_training_fixture_is_degraded():
    """The fixture exists for exactly this verdict -- pin it."""
    snap = RunSnapshot.model_validate(load_fixture("hung_training"))
    assert derive_health(snap, _fixture_now(snap)).health == Health.DEGRADED


def test_heartbeat_expired_fixture_is_unreachable():
    snap = RunSnapshot.model_validate(load_fixture("heartbeat_expired"))
    now = _fixture_now(snap) + timedelta(hours=1)
    assert derive_health(snap, now).health == Health.UNREACHABLE
