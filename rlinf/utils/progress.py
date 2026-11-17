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

"""Step-time bookkeeping and ETA estimation for training runs.

A whole-run average is a poor ETA for RL. Evaluation and checkpoint saves are
periodic and far more expensive than a plain step, so an average over all
elapsed time is dragged upward by however many of them happen to have run,
and the estimate swings every time one lands.

This module separates the three costs and recombines them:

    eta = remaining_plain_steps * p50(plain step)
        + remaining_evals       * p50(eval)
        + remaining_saves       * p50(save)

Remaining eval and save counts come from ``val_check_interval`` and
``save_interval`` using the same predicate the runners use, so the estimate
agrees with what will actually happen rather than approximating it.

Everything here is plain stdlib and side-effect free: no IO, no clock reads
except the ones callers pass in, and no import of the rest of ``rlinf``.
That isolation is deliberate -- see :func:`_will_eval_or_save`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

# Step semantics per task type (contract field ``progress.step_semantics``).
# This *labels* existing behaviour and changes nothing: reasoning genuinely
# logs at minibatch granularity (``reasoning_runner.py`` logs training metrics
# at ``logging_steps + i`` for i in range(n_minibatches)), so its x-axis is
# denser than an embodied run's by that factor and the two are not comparable.
STEP_SEMANTICS_BY_TASK_TYPE = {
    "embodied": "rl_iteration",
    "embodied_eval": "rl_iteration",
    "offline": "rl_iteration",
    "sft": "rl_iteration",
    "reasoning": "minibatch",
    "reasoning_eval": "minibatch",
    "coding_online_rl": "minibatch",
}

DEFAULT_STEP_SEMANTICS = "rl_iteration"

# Confidence thresholds. Below MIN_SAMPLES_MEDIUM samples an ETA is barely more
# than a guess; the drift check catches a run whose step time is still changing
# (warmup, growing sequence lengths) where a p50 over the whole history would
# read as confident while being wrong.
MIN_SAMPLES_MEDIUM = 3
MIN_SAMPLES_HIGH = 10
RECENT_WINDOW = 20
DRIFT_RATIO = 0.5


def _will_eval_or_save(
    step: int,
    max_steps: int,
    val_check_interval: int,
    save_interval: int,
) -> tuple[bool, bool]:
    """Whether ``step`` triggers an eval and/or a save.

    Mirrors :func:`rlinf.utils.runner_utils.check_progress` for the arguments a
    runner passes it (``limit_val_batches=1.0``, ``run_time_exceeded=False``).
    ``test_progress_estimator.py`` cross-checks the two over a grid of intervals,
    so this copy cannot silently diverge from the predicate the loop obeys.

    Reimplemented rather than imported for two reasons. ``check_progress``
    asserts that ``save_interval`` divides ``val_check_interval``, and an ETA
    estimator must not raise on a config the runner itself accepted -- this is
    called once per remaining step, which would turn one bad config into a
    crash mid-run. And importing ``runner_utils`` pulls in the whole Ray
    scheduler stack via ``get_logger()``, which would make this module, and
    therefore the run-state contract, unimportable without Ray.

    Args:
        step: The step to test.
        max_steps: Total steps the run intends to execute.
        val_check_interval: ``cfg.runner.val_check_interval``.
        save_interval: ``cfg.runner.save_interval``.

    Returns:
        ``(run_val, save_model)``.
    """
    is_train_end = step == max_steps

    def divisible(interval: int) -> bool:
        return step != 0 and interval != 0 and step % interval == 0

    run_val = (divisible(val_check_interval) or is_train_end) and val_check_interval > 0
    save_model = (divisible(save_interval) or is_train_end) and save_interval > 0
    return run_val, save_model


def step_semantics_for(task_type: str | None) -> str:
    """Return the ``step_semantics`` label for a task type.

    Args:
        task_type: Value of ``cfg.runner.task_type``, or None.

    Returns:
        One of ``rl_iteration``, ``minibatch``, ``optimizer_step``.
    """
    return STEP_SEMANTICS_BY_TASK_TYPE.get(task_type, DEFAULT_STEP_SEMANTICS)


@dataclass
class ProgressEstimator:
    """Track step durations and project a remaining-time estimate.

    Durations are recorded in three buckets because they have different
    frequencies and magnitudes. ``record_step`` takes the *net* step time --
    eval and save time excluded -- so the buckets stay independent and can be
    multiplied by their own remaining counts.

    Args:
        max_steps: Total steps the run intends to execute.
        val_check_interval: ``cfg.runner.val_check_interval``.
        save_interval: ``cfg.runner.save_interval``.
        step_semantics: What one step means; see :data:`STEP_SEMANTICS_BY_TASK_TYPE`.
    """

    max_steps: int
    val_check_interval: int = 0
    save_interval: int = 0
    step_semantics: str = DEFAULT_STEP_SEMANTICS
    _step_times: list[float] = field(default_factory=list)
    _eval_times: list[float] = field(default_factory=list)
    _save_times: list[float] = field(default_factory=list)

    def record_step(self, duration_s: float) -> None:
        """Record one plain step's net duration (excluding eval and save)."""
        if duration_s >= 0:
            self._step_times.append(float(duration_s))

    def record_eval(self, duration_s: float) -> None:
        if duration_s >= 0:
            self._eval_times.append(float(duration_s))

    def record_save(self, duration_s: float) -> None:
        if duration_s >= 0:
            self._save_times.append(float(duration_s))

    @property
    def step_time_p50(self) -> float | None:
        """Median plain-step time, or None before any step completed."""
        if not self._step_times:
            return None
        return statistics.median(self._step_times)

    @property
    def step_time_recent(self) -> list[float]:
        """The tail of recorded step times, for sparkline rendering."""
        return self._step_times[-3:]

    def _remaining_events(self, step: int) -> tuple[int, int]:
        """Count eval and save events still to come after ``step``.

        Uses the runners' own scheduling predicate rather than integer division
        so the count matches reality exactly, including the run-end eval and
        save that both intervals trigger.

        Args:
            step: Steps completed so far.

        Returns:
            ``(remaining_evals, remaining_saves)``.
        """
        evals = saves = 0
        for future_step in range(step + 1, self.max_steps + 1):
            run_val, save_model = _will_eval_or_save(
                future_step,
                self.max_steps,
                self.val_check_interval,
                self.save_interval,
            )
            evals += run_val
            saves += save_model
        return evals, saves

    def eta_s(self, step: int) -> float | None:
        """Estimate seconds remaining after ``step`` completed steps.

        Returns None while no plain step has been timed, because a number
        invented from no samples is worse than an honest absence.
        """
        step_p50 = self.step_time_p50
        if step_p50 is None:
            return None

        remaining_steps = max(0, self.max_steps - step)
        eta = remaining_steps * step_p50

        remaining_evals, remaining_saves = self._remaining_events(step)
        if remaining_evals and self._eval_times:
            eta += remaining_evals * statistics.median(self._eval_times)
        if remaining_saves and self._save_times:
            eta += remaining_saves * statistics.median(self._save_times)
        return eta

    def eta_confidence(self, step: int) -> str:
        """Grade the ETA as ``low`` / ``medium`` / ``high``.

        Degrades to ``low`` when an eval is still due but none has been timed:
        the estimate is then missing a term known to be large, and reporting it
        as trustworthy would be the misleading case.
        """
        samples = len(self._step_times)
        if samples < MIN_SAMPLES_MEDIUM:
            return "low"

        remaining_evals, _ = self._remaining_events(step)
        if remaining_evals and not self._eval_times:
            return "low"

        # Step time still trending means a median over all history is stale.
        recent = self._step_times[-RECENT_WINDOW:]
        overall_p50 = statistics.median(self._step_times)
        recent_p50 = statistics.median(recent)
        if overall_p50 > 0:
            drift = abs(recent_p50 - overall_p50) / overall_p50
            if drift > DRIFT_RATIO:
                return "low" if samples < MIN_SAMPLES_HIGH else "medium"

        return "high" if samples >= MIN_SAMPLES_HIGH else "medium"

    def snapshot(self, step: int) -> dict:
        """Render the ``timing`` fields of the run-state contract.

        ``started_at`` and ``elapsed_s`` are the caller's to fill: this object
        deliberately never reads a clock.
        """
        return {
            "step_time_p50": self.step_time_p50,
            "step_time_recent": self.step_time_recent,
            "eta_s": self.eta_s(step),
            "eta_confidence": self.eta_confidence(step),
        }
