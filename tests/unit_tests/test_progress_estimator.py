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

"""Tests for :mod:`rlinf.utils.progress`.

The two things worth pinning down here are accuracy and honesty.

Accuracy: a whole-run average is a poor RL ETA because eval and save are
periodic and much more expensive than a plain step, so the estimator splits
them into three buckets. :func:`test_eta_error_under_15_percent_with_periodic_eval`
simulates a run with that exact shape and asserts the split beats the average
it replaced.

Honesty: an ETA computed from no samples, or one missing a term known to be
large, must not be reported as trustworthy. Those are the ``eta_confidence``
and ``eta_s() is None`` cases.

:func:`test_local_predicate_matches_check_progress` guards the one duplication
in the module: ``_will_eval_or_save`` reimplements
:func:`rlinf.utils.runner_utils.check_progress` so the estimator stays
importable without Ray and cannot raise on a divisibility mismatch. That copy
is only safe while the two agree, so they are compared over a grid.
"""

import statistics

import pytest

from rlinf.utils.progress import (
    DEFAULT_STEP_SEMANTICS,
    STEP_SEMANTICS_BY_TASK_TYPE,
    ProgressEstimator,
    _will_eval_or_save,
    step_semantics_for,
)


def _estimator(max_steps=100, val_check_interval=0, save_interval=0):
    return ProgressEstimator(
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        save_interval=save_interval,
    )


# --------------------------------------------------------------- bucket basics


def test_no_samples_yields_no_eta():
    """A number invented from zero samples is worse than an honest absence."""
    estimator = _estimator()
    assert estimator.step_time_p50 is None
    assert estimator.eta_s(0) is None
    assert estimator.eta_confidence(0) == "low"


def test_eta_is_plain_steps_times_median_when_no_eval_or_save():
    estimator = _estimator(max_steps=10)
    for _ in range(5):
        estimator.record_step(2.0)

    assert estimator.step_time_p50 == 2.0
    assert estimator.eta_s(5) == pytest.approx(5 * 2.0)


def test_median_not_mean_so_one_outlier_does_not_dominate():
    """A single slow step (a straggler, a hiccup) must not skew the estimate."""
    estimator = _estimator(max_steps=100)
    for _ in range(9):
        estimator.record_step(2.0)
    estimator.record_step(600.0)

    assert estimator.step_time_p50 == 2.0
    assert statistics.mean([2.0] * 9 + [600.0]) > 60


def test_negative_durations_are_ignored():
    """A clock going backwards must not poison the buckets."""
    estimator = _estimator(max_steps=10)
    estimator.record_step(-1.0)
    estimator.record_eval(-1.0)
    estimator.record_save(-1.0)

    assert estimator.step_time_p50 is None


def test_eta_never_negative_past_max_steps():
    """Runners can overshoot ``max_steps``; the ETA must not go negative."""
    estimator = _estimator(max_steps=10)
    for _ in range(3):
        estimator.record_step(1.0)

    assert estimator.eta_s(12) == pytest.approx(0.0)


def test_step_time_recent_is_the_tail():
    estimator = _estimator()
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        estimator.record_step(value)

    assert estimator.step_time_recent == [3.0, 4.0, 5.0]


# ------------------------------------------------------------------ ETA budget


def test_eta_adds_remaining_eval_and_save_cost():
    """The three buckets must be summed, not just the step one."""
    estimator = _estimator(max_steps=10, val_check_interval=5, save_interval=5)
    for _ in range(5):
        estimator.record_step(2.0)
    estimator.record_eval(30.0)
    estimator.record_save(20.0)

    # After step 5: steps 6..10 remain, with eval+save at 10 only.
    remaining_evals, remaining_saves = estimator._remaining_events(5)
    assert (remaining_evals, remaining_saves) == (1, 1)
    assert estimator.eta_s(5) == pytest.approx(5 * 2.0 + 30.0 + 20.0)


def test_eval_cost_omitted_until_a_sample_exists():
    """With no eval timing yet there is nothing to multiply, and confidence says so."""
    estimator = _estimator(max_steps=10, val_check_interval=5, save_interval=5)
    for _ in range(5):
        estimator.record_step(2.0)

    assert estimator.eta_s(5) == pytest.approx(5 * 2.0)
    assert estimator.eta_confidence(5) == "low"


def test_eta_error_under_15_percent_with_periodic_eval():
    """Keep ETA error below 15% on a realistic eval-heavy run.

    Simulates 100 steps at 2s with a 30s eval every 30 steps, replays the first
    60 into the estimator, then compares its projection for the remaining 40
    against ground truth.

    The interval deliberately does not divide ``max_steps`` evenly, because the
    runners force an eval on the final step. That makes eval density higher in
    the remaining window than in the sampled one -- which is precisely where a
    whole-run average goes wrong, and why this is the shape worth testing. The
    naive formula is asserted to be the worse of the two so the tolerance
    cannot be what is passing the test.

    Ground truth is computed from an inline predicate rather than the module's
    own, so the comparison is not circular.
    """
    max_steps, step_s, eval_s, interval = 100, 2.0, 30.0, 30
    estimator = _estimator(
        max_steps=max_steps, val_check_interval=interval, save_interval=0
    )

    def evals_at(step):
        return step % interval == 0 or step == max_steps

    replay_until = 60
    elapsed = 0.0
    for step in range(1, replay_until + 1):
        estimator.record_step(step_s)
        elapsed += step_s
        if evals_at(step):
            estimator.record_eval(eval_s)
            elapsed += eval_s

    remaining_steps = max_steps - replay_until
    remaining_evals = sum(evals_at(s) for s in range(replay_until + 1, max_steps + 1))
    truth = remaining_steps * step_s + remaining_evals * eval_s

    estimated = estimator.eta_s(replay_until)
    error = abs(estimated - truth) / truth
    assert error < 0.15, f"eta {estimated} vs truth {truth} ({error:.1%})"

    # The formula in metric_utils this replaced: elapsed / steps_done * remaining.
    naive = elapsed / replay_until * remaining_steps
    naive_error = abs(naive - truth) / truth
    assert naive_error > error, (
        f"bucketed error {error:.1%} should beat whole-run average "
        f"{naive_error:.1%} (naive {naive} vs truth {truth})"
    )


# ------------------------------------------------------------------ confidence


def test_confidence_low_below_three_samples():
    estimator = _estimator(max_steps=100)
    for _ in range(2):
        estimator.record_step(2.0)

    assert estimator.eta_confidence(2) == "low"


def test_confidence_medium_then_high_as_samples_accumulate():
    estimator = _estimator(max_steps=100)
    for _ in range(3):
        estimator.record_step(2.0)
    assert estimator.eta_confidence(3) == "medium"

    for _ in range(7):
        estimator.record_step(2.0)
    assert estimator.eta_confidence(10) == "high"


def test_confidence_degrades_when_step_time_drifts():
    """Warmup or growing sequence lengths: a whole-history median reads stale."""
    estimator = _estimator(max_steps=100)
    for _ in range(15):
        estimator.record_step(1.0)
    assert estimator.eta_confidence(15) == "high"

    # Recent window moves well away from the overall median.
    for _ in range(15):
        estimator.record_step(10.0)
    assert estimator.eta_confidence(30) == "medium"


def test_confidence_low_when_eval_is_due_but_never_timed():
    """Missing a term known to be large is exactly the misleading case."""
    estimator = _estimator(max_steps=100, val_check_interval=10)
    for _ in range(20):
        estimator.record_step(2.0)

    assert estimator.eta_confidence(20) == "low"

    estimator.record_eval(30.0)
    assert estimator.eta_confidence(20) == "high"


# ------------------------------------------------------------------- semantics


def test_step_semantics_labels_known_task_types():
    assert step_semantics_for("embodied") == "rl_iteration"
    assert step_semantics_for("reasoning") == "minibatch"
    assert step_semantics_for("coding_online_rl") == "minibatch"


def test_step_semantics_falls_back_for_unknown_task_type():
    assert step_semantics_for("something_new") == DEFAULT_STEP_SEMANTICS
    assert step_semantics_for(None) == DEFAULT_STEP_SEMANTICS


def test_every_supported_task_type_has_semantics():
    """A task type added to the config without a label here would silently
    inherit ``rl_iteration`` and be plotted as comparable to embodied runs."""
    from rlinf.config import SUPPORTED_TASK_TYPE

    missing = set(SUPPORTED_TASK_TYPE) - set(STEP_SEMANTICS_BY_TASK_TYPE)
    assert not missing, f"task types missing step_semantics: {sorted(missing)}"


# -------------------------------------------------------- snapshot and parity


def test_snapshot_has_the_contract_timing_fields():
    estimator = _estimator(max_steps=10)
    for _ in range(4):
        estimator.record_step(2.0)

    snapshot = estimator.snapshot(4)
    assert set(snapshot) == {
        "step_time_p50",
        "step_time_recent",
        "eta_s",
        "eta_confidence",
    }
    # started_at/elapsed_s are the caller's: this object never reads a clock.
    assert "started_at" not in snapshot


@pytest.mark.parametrize("max_steps", [1, 7, 20])
@pytest.mark.parametrize("val_check_interval", [0, 1, 3, 5])
@pytest.mark.parametrize("save_interval", [0, 5, 10])
def test_local_predicate_matches_check_progress(
    max_steps, val_check_interval, save_interval
):
    """``_will_eval_or_save`` must agree with the runners' own predicate.

    Skips the grid points where ``check_progress`` asserts (save_interval not
    divisible by val_check_interval) -- the local copy tolerating those is the
    reason it exists, since an ETA must not crash a run whose config the runner
    already accepted.
    """
    from rlinf.utils.runner_utils import check_progress

    if val_check_interval > 0 and save_interval > 0:
        if save_interval % val_check_interval != 0:
            pytest.skip("check_progress asserts on this combination by design")

    for step in range(0, max_steps + 2):
        expected_val, expected_save, _ = check_progress(
            step, max_steps, val_check_interval, save_interval, 1.0
        )
        actual_val, actual_save = _will_eval_or_save(
            step, max_steps, val_check_interval, save_interval
        )
        assert (actual_val, actual_save) == (expected_val, expected_save), (
            f"step={step} max_steps={max_steps} "
            f"val={val_check_interval} save={save_interval}"
        )


def test_predicate_does_not_raise_on_indivisible_intervals():
    """The config ``check_progress`` rejects must still yield an ETA."""
    estimator = _estimator(max_steps=10, val_check_interval=3, save_interval=5)
    for _ in range(3):
        estimator.record_step(1.0)

    assert estimator.eta_s(3) is not None
    assert estimator.eta_confidence(3) in {"low", "medium", "high"}
