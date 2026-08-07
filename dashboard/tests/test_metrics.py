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

"""Reading time series.

The TensorBoard tests write real event files rather than mocking the reader: the
whole question is whether the protobuf path handles what the training side
actually produces, and a mock would only prove the mock matches itself.

The gateway tests use a fake source, because what is being tested there is
fallback ordering, alias resolution and decimation -- policy, not I/O.
"""

from __future__ import annotations

import pytest
from conftest import write_event_file

from rlinf_dashboard.metrics.base import MetricGateway, make_series
from rlinf_dashboard.metrics.tensorboard import TensorboardSource
from rlinf_dashboard.models import RunManifest


def _manifest(tmp_path, **extra) -> RunManifest:
    payload = {
        "schema_version": 2,
        "run_id": "series-run",
        "task_type": "embodied",
        "project_name": "proj",
        "experiment_name": "exp",
        "paths": {
            "log_path": str(tmp_path / "logs"),
            "tensorboard": str(tmp_path / "logs" / "tensorboard"),
        },
        "metric_aliases": {
            "actor/training/": "train/actor/",
            "critic/training/": "train/critic/",
        },
    }
    payload.update(extra)
    return RunManifest.model_validate(payload)


# ------------------------------------------------------------------- tensorboard


def test_reads_scalars_from_a_real_event_file(tmp_path, settings_for):
    log_dir = str(tmp_path / "logs" / "tensorboard")
    write_event_file(
        log_dir,
        {
            "env/success_once": [(0, 0.1), (1, 0.3), (2, 0.55)],
            "train/actor/loss": [(0, 2.0), (1, 1.5), (2, 1.1)],
        },
    )
    manifest = _manifest(tmp_path)
    source = TensorboardSource(settings_for())

    assert source.available(manifest)
    assert "env/success_once" in source.list_keys(manifest)

    series = source.read(manifest, ["env/success_once"])["env/success_once"]
    assert [point.step for point in series.points] == [0, 1, 2]
    assert series.points[-1].value == pytest.approx(0.55)
    assert series.source == "tensorboard"
    assert series.points[0].wall_time is not None


def test_no_event_files_means_the_source_is_unavailable(tmp_path, settings_for):
    """Unavailable, not empty.

    An unavailable source is skipped by the gateway so the next one gets a turn.
    Reporting "available with no data" would end the fallback chain at the first
    source, so a later source holding the data would never be asked.
    """
    (tmp_path / "logs" / "tensorboard").mkdir(parents=True)
    assert not TensorboardSource(settings_for()).available(_manifest(tmp_path))


def test_the_log_dir_falls_back_to_log_path_over_tensorboard(tmp_path, settings_for):
    """An older manifest records only ``log_path``.

    Runners put event files at ``<log_path>/tensorboard`` by convention, so that
    is the one fallback worth having -- and it is a fallback, not the primary,
    because the manifest is what knows where the run actually wrote.
    """
    write_event_file(str(tmp_path / "logs" / "tensorboard"), {"time/step": [(0, 1.0)]})
    manifest = _manifest(tmp_path, paths={"log_path": str(tmp_path / "logs")})
    assert TensorboardSource(settings_for()).available(manifest)


def test_a_per_worker_run_is_read_from_the_all_subdirectory(tmp_path, settings_for):
    """``runner.per_worker_log: true`` moves the aggregate bundle down a level.

    ``MetricLogger`` passes ``log_path_suffix="all"`` for the driver's own bundle
    so the per-rank ones can live beside it, and manifests written before that
    was recorded name the parent. The parent exists and holds only the ``all``
    directory, so without this the run renders with every metric absent and no
    field saying why -- the same failure shape as an unrelocated path.
    """
    log_dir = tmp_path / "logs" / "tensorboard"
    write_event_file(str(log_dir / "all"), {"env/success_once": [(0, 0.4)]})
    manifest = _manifest(tmp_path)  # records the parent, not `all/`
    source = TensorboardSource(settings_for())

    assert source.available(manifest)
    series = source.read(manifest, ["env/success_once"])["env/success_once"]
    assert series.points[-1].value == pytest.approx(0.4)


def test_the_parent_wins_over_all_when_both_hold_event_files(tmp_path, settings_for):
    """The recorded directory is authoritative.

    A resumed run can leave event files in both places (one launch with
    ``per_worker_log`` off, a later one with it on). Preferring ``all/`` would
    silently switch which launch's curve is shown; the negative control for the
    test above is that the fallback stays a fallback.
    """
    log_dir = tmp_path / "logs" / "tensorboard"
    write_event_file(str(log_dir), {"env/return": [(0, 1.0)]})
    write_event_file(str(log_dir / "all"), {"env/return": [(0, 99.0)]})

    series = TensorboardSource(settings_for()).read(_manifest(tmp_path), ["env/return"])
    assert series["env/return"].points[-1].value == pytest.approx(1.0)


def test_a_growing_event_file_is_picked_up(tmp_path, settings_for):
    """A live run appends while the dashboard polls.

    The accumulator is cached to avoid re-parsing on every request, so this is the
    test that the cache reloads instead of pinning the first read forever -- the
    failure mode being a chart that freezes a few seconds after the page opens.
    """
    log_dir = str(tmp_path / "logs" / "tensorboard")
    write_event_file(log_dir, {"env/return": [(0, 1.0)]})
    manifest = _manifest(tmp_path)
    source = TensorboardSource(settings_for())

    first = source.read(manifest, ["env/return"])["env/return"]
    assert len(first.points) == 1

    write_event_file(log_dir, {"env/return": [(1, 2.0), (2, 3.0)]})
    second = source.read(manifest, ["env/return"])["env/return"]
    assert [point.step for point in second.points] == [0, 1, 2]


def test_an_unknown_key_is_absent_rather_than_empty(tmp_path, settings_for):
    """A source reports only what it has; the gateway decides what absence means."""
    write_event_file(str(tmp_path / "logs" / "tensorboard"), {"env/return": [(0, 1.0)]})
    out = TensorboardSource(settings_for()).read(_manifest(tmp_path), ["nope/nope"])
    assert out == {}


# ---------------------------------------------------------------- per-worker


def _write_worker_tree(tmp_path, tree: dict[tuple[str, int], dict]) -> None:
    """Write per-rank bundles the way ``MetricLogger._get_scoped_logger`` does."""
    root = tmp_path / "logs" / "worker_logs"
    for (group, rank), series in tree.items():
        write_event_file(str(root / group / f"rank_{rank}" / "tensorboard"), series)


def _per_worker_manifest(tmp_path, **extra) -> RunManifest:
    return _manifest(
        tmp_path,
        paths={
            "log_path": str(tmp_path / "logs"),
            "tensorboard": str(tmp_path / "logs" / "tensorboard" / "all"),
            "worker_logs": str(tmp_path / "logs" / "worker_logs"),
        },
        **extra,
    )


def test_workers_are_discovered_from_the_recorded_root(tmp_path, settings_for):
    """``(group, rank)`` comes out of the path, so no second index is needed.

    ``MetricLogger`` writes ``<worker_logs>/<Group>/rank_<n>/tensorboard/``, so
    the two dimensions are recoverable by globbing. Sorted, because readdir order
    is not stable and an unstable order would reshuffle chart colours per poll.
    """
    _write_worker_tree(
        tmp_path,
        {
            ("EnvGroup", 1): {"env/success_once": [(0, 0.3)]},
            ("EnvGroup", 0): {"env/success_once": [(0, 0.1)]},
            ("ActorGroup", 0): {"train/actor/loss": [(0, 2.0)]},
        },
    )
    source = TensorboardSource(settings_for())
    assert source.workers(_per_worker_manifest(tmp_path)) == [
        ("ActorGroup", 0),
        ("EnvGroup", 0),
        ("EnvGroup", 1),
    ]


def test_no_worker_root_means_no_drill_down(tmp_path, settings_for):
    """``per_worker_log`` is off by default, so this is the common case.

    The manifest omits ``worker_logs``, and a tree is left at the location a
    reader would guess -- ``<log_path>/worker_logs``, which is the default
    ``MetricLogger`` writes to. That is not a contrived fixture: every embodied
    example config sets ``log_path: "../results"``, so one run with the flag on
    leaves ranks in the directory the *next* run would be guessed to own. Only
    the recorded path may be trusted, or run B advertises run A's cards.
    """
    write_event_file(str(tmp_path / "logs" / "tensorboard"), {"env/return": [(0, 1.0)]})
    _write_worker_tree(tmp_path, {("EnvGroup", 0): {"env/return": [(0, 9.0)]}})

    source = TensorboardSource(settings_for())
    assert source.workers(_manifest(tmp_path)) == []
    assert source.read_workers(_manifest(tmp_path), ["env/return"]) == {}


def test_each_rank_keeps_its_own_values(tmp_path, settings_for):
    """The point of the feature: one line per rank, not their mean.

    The aggregate is an arithmetic mean across ranks
    (``_aggregate_numeric_metrics``), so "one card in four is four times slower"
    shows up there as merely 1.75x. Only the per-rank series can answer which card.
    """
    _write_worker_tree(
        tmp_path,
        {
            ("EnvGroup", 0): {"time/step": [(0, 10.0), (1, 10.0)]},
            ("EnvGroup", 1): {"time/step": [(0, 40.0), (1, 41.0)]},
        },
    )
    out = TensorboardSource(settings_for()).read_workers(
        _per_worker_manifest(tmp_path), ["time/step"]
    )

    by_rank = {series.rank: series for series in out["time/step"]}
    assert by_rank[0].points[-1].value == pytest.approx(10.0)
    assert by_rank[1].points[-1].value == pytest.approx(41.0)
    assert all(series.group == "EnvGroup" for series in out["time/step"])


def test_a_group_that_never_logged_a_key_is_absent_not_zero(tmp_path, settings_for):
    """Only the env group logs ``env/*``; the actor group logs none of it.

    Returning an empty series for the actor group would draw a flat line at the
    axis for a metric it does not measure -- a chart that lies rather than one
    with a missing line.
    """
    _write_worker_tree(
        tmp_path,
        {
            ("EnvGroup", 0): {"env/success_once": [(0, 0.5)]},
            ("ActorGroup", 0): {"train/actor/loss": [(0, 2.0)]},
        },
    )
    out = TensorboardSource(settings_for()).read_workers(
        _per_worker_manifest(tmp_path), ["env/success_once"]
    )
    assert [(s.group, s.rank) for s in out["env/success_once"]] == [("EnvGroup", 0)]


def test_an_empty_rank_directory_is_not_a_worker(tmp_path, settings_for):
    """A bundle's ``tensorboard`` directory is created eagerly on first use.

    So an empty one means that rank logged nothing, and offering it as a line
    would put an entry in the legend that can never draw.
    """
    _write_worker_tree(tmp_path, {("EnvGroup", 0): {"env/return": [(0, 1.0)]}})
    (tmp_path / "logs" / "worker_logs" / "EnvGroup" / "rank_1" / "tensorboard").mkdir(
        parents=True
    )
    assert TensorboardSource(settings_for()).workers(
        _per_worker_manifest(tmp_path)
    ) == [("EnvGroup", 0)]


def test_a_non_rank_directory_is_skipped(tmp_path, settings_for):
    """Only ``rank_<int>`` is a rank. Anything else is not guessed at.

    ``step_1`` is the interesting name here, not ``notes``: it is the case where
    a reader that only tried to parse an integer off the end of the name -- and
    did not first require the ``rank_`` prefix -- would take the digit after the
    fifth character and invent "rank 1" out of a directory that is not a rank at
    all. That would put a line on a chart attributed to a card that never logged
    it, which is worse than omitting it.
    """
    _write_worker_tree(tmp_path, {("EnvGroup", 0): {"env/return": [(0, 1.0)]}})
    for name in ("notes", "step_1"):
        write_event_file(
            str(tmp_path / "logs" / "worker_logs" / "EnvGroup" / name / "tensorboard"),
            {"env/return": [(0, 5.0)]},
        )
    assert TensorboardSource(settings_for()).workers(
        _per_worker_manifest(tmp_path)
    ) == [("EnvGroup", 0)]


def test_per_worker_series_resolve_legacy_key_aliases(tmp_path, settings_for):
    """The drill-down must not lose the alias window the aggregate path has.

    An older run wrote ``actor/training/loss`` into its per-rank bundles too, and
    a template asking for the canonical name has to find it there as well or the
    breakdown is empty for exactly the runs that most need explaining.
    """
    _write_worker_tree(
        tmp_path, {("ActorGroup", 0): {"actor/training/loss": [(0, 2.0), (1, 1.5)]}}
    )
    gateway = MetricGateway(settings_for(), sources=[TensorboardSource(settings_for())])

    out = gateway.read_workers(_per_worker_manifest(tmp_path), ["train/actor/loss"])
    series = out["train/actor/loss"][0]
    assert series.key == "train/actor/loss"
    assert series.points[-1].value == pytest.approx(1.5)


def test_the_gateway_labels_workers_by_their_directory(tmp_path, settings_for):
    """The label is the on-disk spelling, so a legend entry is also a path to grep."""
    _write_worker_tree(tmp_path, {("EnvGroup", 3): {"env/return": [(0, 1.0)]}})
    gateway = MetricGateway(settings_for(), sources=[TensorboardSource(settings_for())])
    assert gateway.workers(_per_worker_manifest(tmp_path)) == ["EnvGroup/rank_3"]


def test_a_source_that_cannot_break_out_ranks_is_skipped(tmp_path, settings_for):
    """Per-worker reading is an optional capability, not part of every source.

    Breaking a series out per rank needs the ranks to be addressable. A source
    that knows a run only by name cannot address one, and forcing it to declare a
    stub returning nothing would put that emptiness behind the same call the real
    answer comes from. The gateway asks whoever can answer instead.
    """
    plain = _FakeSource("plain", {"env/return": [(0, 1.0)]})
    gateway = MetricGateway(settings_for(), sources=[plain])
    assert gateway.workers(_manifest(tmp_path)) == []
    assert gateway.read_workers(_manifest(tmp_path), ["env/return"]) == {}


def test_per_worker_series_are_decimated_like_any_other(tmp_path, settings_for):
    """Four ranks of a long run is four times the points of one.

    The point cap exists because a browser cannot draw an unbounded series; the
    breakdown is the case that most needs it, so skipping decimation here would
    make the feature the one that blows the page up.
    """
    _write_worker_tree(
        tmp_path,
        {
            ("EnvGroup", 0): {
                "env/return": [(step, float(step)) for step in range(3000)]
            }
        },
    )
    gateway = MetricGateway(
        settings_for(max_series_points=100),
        sources=[TensorboardSource(settings_for())],
    )

    series = gateway.read_workers(_per_worker_manifest(tmp_path), ["env/return"])[
        "env/return"
    ][0]
    assert series.decimated is True
    assert series.total_points == 3000
    assert series.points[-1].step == 2999


def test_a_corrupt_event_file_does_not_raise(tmp_path, settings_for):
    """Truncated event files happen on a killed run and must not 500."""
    log_dir = tmp_path / "logs" / "tensorboard"
    log_dir.mkdir(parents=True)
    (log_dir / "events.out.tfevents.1234.host.0").write_bytes(
        b"\x00\x01\x02not-a-proto"
    )

    source = TensorboardSource(settings_for())
    assert source.list_keys(_manifest(tmp_path)) == []
    assert source.read(_manifest(tmp_path), ["env/return"]) == {}


# ----------------------------------------------------------------------- gateway


class _FakeSource:
    """A source with a fixed key -> points map."""

    def __init__(self, name, data, *, available=True, explode=False):
        self.name = name
        self._data = data
        self._available = available
        self._explode = explode
        self.read_calls: list[list[str]] = []

    def available(self, manifest):
        return self._available

    def list_keys(self, manifest):
        if self._explode:
            raise RuntimeError("boom")
        return sorted(self._data)

    def read(self, manifest, keys):
        self.read_calls.append(list(keys))
        if self._explode:
            raise RuntimeError("boom")
        return {
            key: make_series(key, self.name, [(s, v, None) for s, v in self._data[key]])
            for key in keys
            if key in self._data
        }


def test_keys_are_the_union_across_sources(tmp_path, settings_for):
    """A run can log different families to different backends.

    Media only ever reaches wandb, for instance. A key the UI cannot offer is a
    key nobody can chart, so listing takes the union even though reading takes the
    first hit.
    """
    gateway = MetricGateway(
        settings_for(),
        sources=[
            _FakeSource("a", {"env/return": [(0, 1.0)]}),
            _FakeSource("b", {"train/actor/loss": [(0, 2.0)]}),
        ],
    )
    assert gateway.list_keys(_manifest(tmp_path)) == ["env/return", "train/actor/loss"]


def test_one_failing_source_does_not_blank_the_key_list(tmp_path, settings_for):
    gateway = MetricGateway(
        settings_for(),
        sources=[
            _FakeSource("broken", {}, explode=True),
            _FakeSource("ok", {"env/return": [(0, 1.0)]}),
        ],
    )
    assert gateway.list_keys(_manifest(tmp_path)) == ["env/return"]


def test_reads_fall_back_to_the_next_source(tmp_path, settings_for):
    """First source with data wins, per key.

    A fallback chain rather than a merge: the same key in two backends is the same
    numbers written twice, and interleaving them would manufacture duplicate
    steps.
    """
    first = _FakeSource("first", {"env/return": [(0, 1.0)]})
    second = _FakeSource("second", {"env/return": [(0, 9.9)], "time/step": [(0, 40.0)]})
    gateway = MetricGateway(settings_for(), sources=[first, second])

    out = gateway.read(_manifest(tmp_path), ["env/return", "time/step"])
    assert out["env/return"].source == "first"
    assert out["env/return"].points[0].value == pytest.approx(1.0)
    assert out["time/step"].source == "second"


def test_a_key_with_no_data_anywhere_is_reported_as_such(tmp_path, settings_for):
    """No-data-yet and key-does-not-exist look identical without this.

    The first is a run that has not reached its first eval; the second is a typo
    in a template. ``source="none"`` with an empty point list is what lets the UI
    say which.
    """
    gateway = MetricGateway(settings_for(), sources=[_FakeSource("a", {})])
    out = gateway.read(_manifest(tmp_path), ["eval/success_once"])
    assert out["eval/success_once"].source == "none"
    assert out["eval/success_once"].points == []


def test_a_canonical_request_finds_data_under_the_legacy_key(tmp_path, settings_for):
    """The dual-write window's read side.

    During deprecation, older runs have only ``actor/training/loss`` on disk while
    templates ask for ``train/actor/loss``. The manifest carries the alias map so
    this resolution needs no ``import rlinf``.
    """
    source = _FakeSource("tb", {"actor/training/loss": [(0, 2.0), (1, 1.5)]})
    gateway = MetricGateway(settings_for(), sources=[source])

    out = gateway.read(_manifest(tmp_path), ["train/actor/loss"])
    assert out["train/actor/loss"].points[-1].value == pytest.approx(1.5)
    # The response is keyed by what was asked for, not by what was found; the
    # frontend asked for the canonical name and must be able to match it up.
    assert out["train/actor/loss"].key == "train/actor/loss"


def test_a_legacy_request_finds_data_under_the_canonical_key(tmp_path, settings_for):
    """The other direction: a saved dashboard URL outliving the dual write.

    Once the deprecated write is removed, a bookmarked legacy key must still
    resolve, or every saved link breaks on upgrade.
    """
    source = _FakeSource("tb", {"train/critic/loss": [(0, 0.5)]})
    gateway = MetricGateway(settings_for(), sources=[source])
    out = gateway.read(_manifest(tmp_path), ["critic/training/loss"])
    assert out["critic/training/loss"].points[0].value == pytest.approx(0.5)


def test_unavailable_sources_are_never_asked(tmp_path, settings_for):
    unavailable = _FakeSource("off", {"env/return": [(0, 1.0)]}, available=False)
    live = _FakeSource("on", {"env/return": [(0, 2.0)]})
    gateway = MetricGateway(settings_for(), sources=[unavailable, live])

    assert (
        gateway.read(_manifest(tmp_path), ["env/return"])["env/return"].source == "on"
    )
    assert unavailable.read_calls == []


def test_long_series_keep_their_spikes(tmp_path, settings_for):
    """RL curve decimation preserves diagnostically important excursions."""
    spike_step = 5000
    points = [(step, 1.0) for step in range(20_000)]
    points[spike_step] = (spike_step, 999.0)
    source = _FakeSource("tb", {"train/actor/loss": points})
    gateway = MetricGateway(settings_for(max_series_points=1000), sources=[source])

    series = gateway.read(_manifest(tmp_path), ["train/actor/loss"])["train/actor/loss"]
    assert series.decimated is True
    assert series.total_points == 20_000
    assert len(series.points) <= 1000
    # Every surviving value is one that was genuinely recorded...
    assert {point.value for point in series.points} <= {1.0, 999.0}
    # ...and the one that matters is among them.
    assert any(point.value == 999.0 for point in series.points), (
        "the loss blowup was decimated away; the chart would show a clean curve"
    )


@pytest.mark.parametrize("spike_step", [0, 1, 4999, 5000, 7777, 12_345, 19_998, 19_999])
def test_a_spike_survives_wherever_it_falls(tmp_path, spike_step, settings_for):
    """One well-placed index is a weak guarantee; the property is positional.

    A stride-based sampler passes for whichever indices happen to be multiples
    of the stride, so a single fixed spike position tests the sampler's phase
    rather than its behaviour. These positions include both ends and several
    interior points chosen to fall off any plausible stride.
    """
    points = [(step, 1.0) for step in range(20_000)]
    points[spike_step] = (spike_step, 999.0)
    source = _FakeSource("tb", {"train/actor/loss": points})
    gateway = MetricGateway(settings_for(max_series_points=1000), sources=[source])

    series = gateway.read(_manifest(tmp_path), ["train/actor/loss"])["train/actor/loss"]
    spikes = [point for point in series.points if point.value == 999.0]
    assert spikes, f"a spike at step {spike_step} did not survive decimation"
    assert spikes[0].step == spike_step, "the spike kept its step"


def test_a_collapse_survives_too(tmp_path, settings_for):
    """Reward collapse is a downward excursion, and min matters as much as max."""
    points = [(step, 5.0) for step in range(20_000)]
    points[8_888] = (8_888, -100.0)
    source = _FakeSource("tb", {"env/return": points})
    gateway = MetricGateway(settings_for(max_series_points=1000), sources=[source])

    series = gateway.read(_manifest(tmp_path), ["env/return"])["env/return"]
    assert any(point.value == -100.0 for point in series.points), (
        "keeping only per-bucket maxima would hide every collapse"
    )


def test_non_finite_points_are_never_decimated_away(tmp_path, settings_for):
    """A NaN is the event, not a sample -- and an alert depends on it.

    ``nonFiniteSignal`` in the frontend decides whether to report the run as
    broken by scanning the points it receives. A NaN dropped for display size
    does not just leave a gap in a chart; it disarms the warning that the run
    diverged.
    """
    points = [(step, 1.0) for step in range(20_000)]
    points[3_333] = (3_333, float("nan"))
    points[3_334] = (3_334, float("inf"))
    source = _FakeSource("tb", {"train/actor/loss": points})
    gateway = MetricGateway(settings_for(max_series_points=200), sources=[source])

    series = gateway.read(_manifest(tmp_path), ["train/actor/loss"])["train/actor/loss"]
    steps = {point.step for point in series.points}
    assert 3_333 in steps and 3_334 in steps, (
        "the non-finite samples were dropped, so the divergence warning would "
        "never fire for this run"
    )


def test_decimation_respects_its_budget(tmp_path, settings_for):
    """Two points per bucket must not double the payload the budget promised."""
    points = [(step, float(step % 7)) for step in range(50_000)]
    source = _FakeSource("tb", {"env/return": points})
    gateway = MetricGateway(settings_for(max_series_points=1000), sources=[source])

    series = gateway.read(_manifest(tmp_path), ["env/return"])["env/return"]
    assert len(series.points) <= 1000
    assert series.points == sorted(series.points, key=lambda p: p.step), (
        "points must stay in step order; uPlot requires a sorted x axis"
    )


@pytest.mark.parametrize("limit", [2, 3, 5, 8, 17, 100, 4000])
@pytest.mark.parametrize(
    ("shape", "make"),
    [
        ("sawtooth", lambda n: [float(i % 2) for i in range(n)]),
        ("monotonic", lambda n: [float(i) for i in range(n)]),
        ("all_nan", lambda n: [float("nan")] * n),
        ("half_nan", lambda n: [float("nan") if i % 2 else float(i) for i in range(n)]),
    ],
)
def test_the_budget_is_a_bound_not_a_target(tmp_path, limit, shape, make, settings_for):
    """Three claims share one budget, so their sum must still fit inside it.

    First/last, non-finite points and per-bucket extremes each want room. An
    earlier revision floored the bucket count at one, which broke the bound for
    small limits -- and an all-NaN series (a run that diverged completely) would
    otherwise have carried every one of its 50 000 points into the response.
    """
    total = 1_000
    source = _FakeSource("tb", {"env/return": list(enumerate(make(total)))})
    gateway = MetricGateway(settings_for(max_series_points=limit), sources=[source])

    series = gateway.read(_manifest(tmp_path), ["env/return"])["env/return"]
    assert len(series.points) <= limit, f"{shape} at limit {limit} overshot"
    assert series.points[0].step == 0
    assert series.points[-1].step == total - 1


def test_decimation_always_keeps_the_last_point(tmp_path, settings_for):
    """The newest value is the one the header number shows.

    Dropping it to a stride boundary would make the headline metric lag the curve
    by up to a stride, which reads as the run having stalled.
    """
    points = [(step, float(step)) for step in range(10_001)]
    source = _FakeSource("tb", {"env/return": points})
    gateway = MetricGateway(settings_for(max_series_points=100), sources=[source])

    series = gateway.read(_manifest(tmp_path), ["env/return"])["env/return"]
    assert series.points[-1].step == 10_000


def test_a_short_series_is_not_marked_decimated(tmp_path, settings_for):
    source = _FakeSource("tb", {"env/return": [(0, 1.0), (1, 2.0)]})
    gateway = MetricGateway(settings_for(max_series_points=1000), sources=[source])
    series = gateway.read(_manifest(tmp_path), ["env/return"])["env/return"]
    assert series.decimated is False
    assert series.total_points == 2


def test_resumed_steps_keep_the_later_write(tmp_path, settings_for):
    """A resumed run legitimately rewrites steps it already logged.

    Both values are in the event file. The one written after resume is the one
    that is true, and keeping both would draw a curve that jumps backwards.
    """
    series = make_series("env/return", "tb", [(5, 1.0, None), (5, 2.0, None)])
    assert len(series.points) == 1
    assert series.points[0].value == pytest.approx(2.0)


def test_points_come_back_sorted():
    """Event files are append-ordered, which is not step-ordered after a resume."""
    series = make_series("x", "tb", [(3, 3.0, None), (1, 1.0, None), (2, 2.0, None)])
    assert [point.step for point in series.points] == [1, 2, 3]


def test_the_gateway_attributes_the_answer_to_a_source(tmp_path, settings_for):
    """Every series says where it came from, even on the happy path.

    ``source`` is what a support conversation starts from -- "the chart is empty"
    and "the chart is wrong" have different answers depending on whether anything
    answered at all.
    """
    points = [(0, 1.0), (1, 2.0)]
    write_event_file(str(tmp_path / "logs" / "tensorboard"), {"env/return": points})

    series = MetricGateway(settings_for()).read(_manifest(tmp_path), ["env/return"])[
        "env/return"
    ]
    assert series.source == "tensorboard"
    assert [(point.step, point.value) for point in series.points] == points


# --------------------------------------------------------------------- isolation


def test_the_metrics_layer_does_not_import_rlinf(no_rlinf, tmp_path, settings_for):
    """The architectural constraint, exercised on the real read path.

    Enforced in CI by an ``rlinf``-free venv as well; this makes a stray import
    fail on a laptop, where it is cheap to notice.
    """
    write_event_file(str(tmp_path / "logs" / "tensorboard"), {"env/return": [(0, 1.0)]})
    gateway = MetricGateway(settings_for())
    manifest = _manifest(tmp_path)
    assert gateway.list_keys(manifest) == ["env/return"]
    assert gateway.read(manifest, ["env/return"])["env/return"].points


def test_event_files_are_read_without_torch(settings_for, tmp_path):
    """Reading scalars needs the protobuf reader, not a deep learning framework.

    That is what keeps this package installable in a plain venv instead of one of
    the heavy training environments.
    """
    write_event_file(str(tmp_path / "logs" / "tensorboard"), {"env/return": [(0, 1.0)]})
    source = TensorboardSource(settings_for())
    assert source.read(_manifest(tmp_path), ["env/return"])["env/return"].points

    import sys

    assert "torch" not in sys.modules, "the event reader must not pull in torch"
