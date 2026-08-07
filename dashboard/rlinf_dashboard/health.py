# Copyright 2026 The RLinf Authors.
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

"""Liveness derivation: the answer to "should I be worried?".

This is a pure function of a snapshot and a clock. That is the point:

* it is why ``state`` has no ``stalled`` value -- a ``kill -9``'d process cannot
  write its own death certificate, so liveness must be decided by the reader;
* it is fully unit-testable by feeding timestamps, with no filesystem;
* the SSE stream, the REST list and any future alerting all consume the same
  verdict, so a run cannot look healthy in one view and dead in another.

*Progress* timeouts are relative to the run's own observed step time. A fixed
threshold cannot cover both short reasoning steps and long embodied steps.

*Liveness* timeouts are not, and deliberately so -- the heartbeat is a
fixed-period tick that keeps running at the same rate however long a step takes.
The two signals therefore use independent budgets.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .models import Health, HealthVerdict, RunSnapshot, RunState

#: States where silence is the expected outcome, not a symptom.
_TERMINAL = (RunState.FINISHED, RunState.FAILED, RunState.STOPPED)

#: Lower bound on the process-liveness budget, in seconds. Independent of the
#: step-time floor, which answers a different question. Its job is to absorb a
#: late tick -- an NFS write that blocked, a moment of clock skew -- rather than
#: to accommodate slow training, which the heartbeat thread does not wait for.
_HEARTBEAT_FLOOR_S = 15.0


def _age_s(then: datetime | None, now: datetime) -> float | None:
    """Seconds since ``then``, or ``None`` if it was never recorded.

    Naive timestamps are read as UTC: the writer emits ``Z``-suffixed UTC, and a
    snapshot hand-edited to drop the suffix should still be readable rather than
    raise on the subtraction.
    """
    if then is None:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (now - then).total_seconds()


def derive_health(
    snapshot: RunSnapshot | None,
    now: datetime | None = None,
    *,
    heartbeat_timeout_k: float = 5.0,
    progress_timeout_k: float = 10.0,
    timeout_floor_s: float = 30.0,
    heartbeat_interval_s: float = 5.0,
) -> HealthVerdict:
    """Classify a run's liveness from the three v2 timestamps.

    A heartbeat proves that the process is alive, while progress and metric
    timestamps show whether useful work and observability are advancing.

    **The two budgets are deliberately different quantities.** Progress staleness
    is measured against step time, because one step is how long a working run may
    legitimately go without advancing. Process liveness is not: the heartbeat is
    a fixed-period tick from a daemon thread that does not slow down with a step.

    Args:
        snapshot: Parsed ``run.json``, or ``None`` if it could not be read.
        now: Reader-side clock. Defaults to ``datetime.now(timezone.utc)``.
        heartbeat_timeout_k: Missed heartbeats tolerated before the process is
            presumed gone. Multiplies ``heartbeat_interval_s``, never step time.
        progress_timeout_k: Multiples of the *step-time* budget before a live
            process with no step advance is presumed hung. Larger than the
            heartbeat multiple because a legitimate step is orders of magnitude
            longer than a heartbeat interval.
        timeout_floor_s: Lower bound on the step-time budget, covering runs with
            no step-time samples yet.
        heartbeat_interval_s: The writer's tick period from ``manifest.json``,
            or the reader's fallback for older manifests.

    Returns:
        A :class:`HealthVerdict` carrying the verdict, a human-readable reason,
        and the ages it was computed from, so the UI can explain itself.
    """
    now = now or datetime.now(timezone.utc)

    if snapshot is None:
        return HealthVerdict(
            health=Health.UNKNOWN,
            reason="No readable run.json for this run.",
        )

    if snapshot.state in _TERMINAL:
        return HealthVerdict(
            health=Health.HEALTHY,
            reason=f"Run is {snapshot.state.value}; silence is expected.",
        )

    # A run that has not started yet has nothing to be late for.
    if snapshot.state == RunState.PENDING:
        return HealthVerdict(
            health=Health.HEALTHY,
            reason="Run has not started yet.",
        )

    step_time = snapshot.timing.step_time_p50 or 0.0
    # What a *step* is allowed to take. Progress and metric staleness only.
    budget = max(step_time, timeout_floor_s)

    # Heartbeat liveness uses the writer's fixed interval, independent of step
    # duration. The floor absorbs slow filesystem writes and modest clock skew.
    heartbeat_budget = max(
        heartbeat_timeout_k * heartbeat_interval_s, _HEARTBEAT_FLOOR_S
    )

    heartbeat_age = _age_s(snapshot.heartbeat_at, now)
    progress_age = _age_s(snapshot.last_progress_at, now)
    metric_age = _age_s(snapshot.last_metric_at, now)

    def verdict(health: Health, reason: str) -> HealthVerdict:
        return HealthVerdict(
            health=health,
            reason=reason,
            heartbeat_age_s=heartbeat_age,
            progress_age_s=progress_age,
            metric_age_s=metric_age,
            budget_s=budget,
            heartbeat_budget_s=heartbeat_budget,
        )

    # A running snapshot with no heartbeat at all means the writer never got far
    # enough to tick. Report it rather than treating a missing field as fresh.
    if heartbeat_age is None:
        return verdict(
            Health.UNKNOWN, "Snapshot claims to be running but has no heartbeat."
        )

    # Process liveness comes first: if the heartbeat thread is gone, nothing else
    # in the snapshot can be trusted to be current. Judged against the tick
    # period, not the step time -- see the note in this function's docstring.
    if heartbeat_age > heartbeat_budget:
        return verdict(
            Health.UNREACHABLE,
            f"No heartbeat for {heartbeat_age:.0f}s "
            f"(budget {heartbeat_budget:.0f}s, {heartbeat_timeout_k:g} ticks of "
            f"{heartbeat_interval_s:g}s); the driver process is probably gone.",
        )

    # Heartbeat fresh, steps not advancing: the hung-training-thread signature.
    if progress_age is not None and progress_age > progress_timeout_k * budget:
        return verdict(
            Health.DEGRADED,
            f"Heartbeat is fresh but no step has completed for "
            f"{progress_age:.0f}s (over {progress_timeout_k:g}x the "
            f"{budget:.0f}s budget); training may be hung.",
        )

    # Steps advancing but metrics not reaching a backend: partial blindness. Not
    # a training failure, but every chart is stale and worth saying so.
    if metric_age is not None and metric_age > progress_timeout_k * budget:
        return verdict(
            Health.DEGRADED,
            f"No metric has been written for {metric_age:.0f}s; "
            f"the metric path may be broken.",
        )

    return verdict(Health.HEALTHY, "Heartbeat, progress and metrics are all current.")
