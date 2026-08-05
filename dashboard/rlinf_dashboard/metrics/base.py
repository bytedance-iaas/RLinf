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

"""The metric source protocol and the gateway that reads through it.

Series come from TensorBoard event files, which every RLinf run writes. The
protocol keeps event-file parsing behind an interface so that the concerns the
gateway owns -- alias resolution, and telling "no data yet" apart from "no such
key" -- can be tested against a fake instead of a directory of real events.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from ..models import RunManifest, Series, SeriesPoint
from ..settings import Settings

logger = logging.getLogger(__name__)


@runtime_checkable
class MetricSource(Protocol):
    """A place series can be read from."""

    name: str

    def available(self, manifest: RunManifest) -> bool:
        """Whether this source can serve the given run at all."""
        ...

    def list_keys(self, manifest: RunManifest) -> list[str]:
        """Metric keys this source has for the run."""
        ...

    def read(self, manifest: RunManifest, keys: list[str]) -> dict[str, Series]:
        """Read the named series. Missing keys are simply absent from the result."""
        ...


@runtime_checkable
class WorkerAwareSource(Protocol):
    """A source that can also break a series out per ``(worker group, rank)``.

    Kept separate from :class:`MetricSource` because breaking a series out per
    rank needs more than the ability to read one: the ranks have to be
    addressable. Event files are, by path. A source that knows a run only by
    name is not, and would have to answer with a stub returning nothing.
    Declaring the capability optional lets the gateway ask whoever can answer.
    """

    def workers(self, manifest: RunManifest) -> list[tuple[str, int]]:
        """The ``(group, rank)`` pairs with per-worker data, or empty."""
        ...

    def read_workers(
        self, manifest: RunManifest, keys: list[str]
    ) -> dict[str, list[Series]]:
        """Read each key once per worker that logged it."""
        ...


class MetricGateway:
    """Serve series from the first source that has data for a run.

    Sources are consulted in order and the first with data for a key answers for
    it. This is a fallback chain, not a merge: the same key from two sources is
    the same numbers read twice, and interleaving them would only manufacture
    apparent duplicate steps.

    Legacy metric keys are resolved through ``manifest.metric_aliases``, which
    the training side embeds precisely so this package need not import ``rlinf``
    to know that ``actor/training/loss`` and ``train/actor/loss`` are one series
    during the dual-write deprecation window.
    """

    def __init__(self, settings: Settings, sources: list[MetricSource] | None = None):
        self._settings = settings
        if sources is None:
            from .tensorboard import TensorboardSource

            sources = [TensorboardSource(settings)]
        self._sources = sources

    def sources_for(self, manifest: RunManifest) -> list[MetricSource]:
        """Return the sources that can serve this run, in preference order."""
        return [source for source in self._sources if source.available(manifest)]

    def list_keys(self, manifest: RunManifest) -> list[str]:
        """Union of keys across every available source.

        A union rather than first-hit: a run can log different families to
        different backends (media only reaches wandb, for instance), and a key
        the UI cannot offer is a key nobody can chart.
        """
        keys: set[str] = set()
        for source in self.sources_for(manifest):
            try:
                keys.update(source.list_keys(manifest))
            except Exception as exc:  # noqa: BLE001 - one bad source must not blank the list
                logger.warning("Source %s failed to list keys: %s", source.name, exc)
        return sorted(keys)

    def read(self, manifest: RunManifest, keys: list[str]) -> dict[str, Series]:
        """Read series for ``keys``, falling back across sources.

        Args:
            manifest: The run's manifest, which says where its data plane lives.
            keys: Requested metric keys, canonical or legacy.

        Returns:
            Mapping from the *requested* key to its series. A requested key with
            no data anywhere gets an empty series with ``source="none"``, so the
            frontend can distinguish "no data yet" from "key does not exist".
        """
        wanted = {key: _candidate_keys(key, manifest) for key in keys}
        result: dict[str, Series] = {}
        outstanding = dict(wanted)

        for source in self.sources_for(manifest):
            if not outstanding:
                break
            probe = sorted(
                {alias for aliases in outstanding.values() for alias in aliases}
            )
            try:
                found = source.read(manifest, probe)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Source %s failed to read: %s", source.name, exc)
                continue

            for key, aliases in list(outstanding.items()):
                for alias in aliases:
                    series = found.get(alias)
                    if series is not None and series.points:
                        result[key] = self._decimate(
                            series.model_copy(update={"key": key})
                        )
                        del outstanding[key]
                        break

        for key in outstanding:
            result[key] = Series(key=key, source="none")
        return result

    def _worker_sources(self) -> list[WorkerAwareSource]:
        """Sources that can break a series out per rank.

        Not filtered through :meth:`sources_for`: ``available()`` asks whether the
        *aggregate* bundle has data, and the per-worker bundles are a different
        set of directories. A run whose driver bundle is empty while its ranks are
        writing is exactly a run worth drilling into, so gating on the aggregate
        would hide the breakdown in the case it is most wanted.
        """
        return [
            source for source in self._sources if isinstance(source, WorkerAwareSource)
        ]

    def workers(self, manifest: RunManifest) -> list[str]:
        """Worker labels with per-rank data, as ``"<group>/rank_<n>"``.

        A flat list of labels rather than pairs because this is an API response:
        the label is what a chart legend shows and what a caller round-trips, and
        it is one field instead of two that must be kept in step.
        """
        labels: list[str] = []
        for source in self._worker_sources():
            try:
                labels = [
                    worker_label(group, rank)
                    for group, rank in source.workers(manifest)
                ]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Source %s failed to list workers: %s", source.name, exc)
                continue
            if labels:
                break
        return labels

    def read_workers(
        self, manifest: RunManifest, keys: list[str]
    ) -> dict[str, list[Series]]:
        """Read each key broken out per ``(worker group, rank)``.

        Same fallback discipline as :meth:`read`: the first source that has
        per-worker data for a key answers for that key, and a key no source can
        break out is simply absent -- distinct from the aggregate path, which
        returns an empty series for a missing key. Absent here means "no
        breakdown available", and the caller still has the aggregate to show.
        """
        wanted = {key: _candidate_keys(key, manifest) for key in keys}
        result: dict[str, list[Series]] = {}

        for source in self._worker_sources():
            if len(result) == len(wanted):
                break
            probe = sorted({alias for aliases in wanted.values() for alias in aliases})
            try:
                found = source.read_workers(manifest, probe)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Source %s failed to read per-worker series: %s", source.name, exc
                )
                continue

            for key, aliases in wanted.items():
                if key in result:
                    continue
                for alias in aliases:
                    series_list = [s for s in found.get(alias, []) if s.points]
                    if series_list:
                        result[key] = [
                            self._decimate(series.model_copy(update={"key": key}))
                            for series in series_list
                        ]
                        break
        return result

    def _decimate(self, series: Series) -> Series:
        """Strided-sample a long series down to a displayable size.

        Strided rather than averaged: RL curves are read for their spikes (a loss
        blowup, a reward collapse), and averaging is exactly what hides those.
        The last point is always kept so the newest value is never dropped.
        """
        limit = self._settings.max_series_points
        total = len(series.points)
        if total <= limit:
            return series.model_copy(update={"total_points": total})

        stride = total // limit + 1
        sampled = series.points[::stride]
        if sampled[-1] is not series.points[-1]:
            sampled.append(series.points[-1])
        return series.model_copy(
            update={"points": sampled, "decimated": True, "total_points": total}
        )


def worker_label(group: str, rank: int) -> str:
    """``"<group>/rank_<n>"`` -- the on-disk spelling, reused as the UI label.

    Matching the directory layout ``MetricLogger`` writes means a label in the UI
    is also the path to grep, which is the whole point of a drill-down.
    """
    return f"{group}/rank_{rank}"


def _candidate_keys(key: str, manifest: RunManifest) -> list[str]:
    """Expand a key into every spelling a backend might have stored it under.

    Both directions are generated. A canonical request must find data written
    under a legacy prefix (an older run), and a legacy request must find data
    written under the canonical prefix (a newer run whose dual write is gone).
    """
    candidates = [key]
    for legacy, canonical in manifest.metric_aliases.items():
        if key.startswith(canonical):
            candidates.append(legacy + key[len(canonical) :])
        elif key.startswith(legacy):
            candidates.append(canonical + key[len(legacy) :])
    return list(dict.fromkeys(candidates))


def make_series(
    key: str,
    source: str,
    raw: list[tuple[int, float, float | None]],
) -> Series:
    """Build a :class:`Series` from ``(step, value, wall_time)`` triples.

    Points are sorted by step and deduplicated keeping the last write, because a
    resumed run legitimately rewrites steps it already logged, and the value
    after resume is the one that is true.
    """
    by_step: dict[int, tuple[float, float | None]] = {}
    for step, value, wall_time in raw:
        by_step[int(step)] = (float(value), wall_time)
    points = [
        SeriesPoint(step=step, value=value, wall_time=wall_time)
        for step, (value, wall_time) in sorted(by_step.items())
    ]
    return Series(key=key, points=points, source=source, total_points=len(points))
