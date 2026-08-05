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


def test_long_series_are_strided_not_averaged(tmp_path, settings_for):
    """RL curves are read for their spikes; averaging is what hides them.

    A loss blowup or a reward collapse is a single-point event. Strided sampling
    may miss it, but a mean over the window would erase it while looking fine --
    the worse failure, because it is invisible.
    """
    spike_step = 5000
    points = [(step, 1.0) for step in range(20_000)]
    points[spike_step] = (spike_step, 999.0)
    source = _FakeSource("tb", {"train/actor/loss": points})
    gateway = MetricGateway(settings_for(max_series_points=1000), sources=[source])

    series = gateway.read(_manifest(tmp_path), ["train/actor/loss"])["train/actor/loss"]
    assert series.decimated is True
    assert series.total_points == 20_000
    assert len(series.points) <= 1100
    # Every surviving value is one that was genuinely recorded.
    assert {point.value for point in series.points} <= {1.0, 999.0}


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
