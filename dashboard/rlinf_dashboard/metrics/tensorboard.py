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

"""Reading series straight out of TensorBoard event files.

This is the source that always works: ``tensorboard`` is an unconditional RLinf
dependency, so every run has event files. Only the protobuf reader is used here,
not the TensorBoard server and not torch -- reading scalars needs neither.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from ..models import RunManifest, Series
from ..settings import Settings
from .base import make_series

logger = logging.getLogger(__name__)

#: Where ``MetricLogger`` puts the driver's aggregate bundle when
#: ``runner.per_worker_log`` is on, so the per-rank bundles get their own room.
_AGGREGATE_SUBDIR = "all"

#: Directory name prefix ``MetricLogger._get_scoped_logger`` gives each rank.
_RANK_PREFIX = "rank_"


class TensorboardSource:
    """Event-file reader with a reload-on-growth cache.

    ``EventAccumulator`` re-reads from its own offset on ``Reload()``, so a live
    run is followed incrementally rather than re-parsed. The accumulator is
    cached per directory and reloaded when the directory's newest event file has
    grown, which is the cheap proxy for "there is something new to read".
    """

    name = "tensorboard"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._lock = threading.Lock()
        #: log_dir -> (accumulator, last_size, last_reload_monotonic)
        self._cache: dict[str, tuple[object, int, float]] = {}

    # ---------------------------------------------------------------- protocol

    def available(self, manifest: RunManifest) -> bool:
        """Whether the run has a log directory with at least one event file.

        An empty directory is not availability: a run configured for TensorBoard
        that has not logged yet must fall through rather than answer with
        nothing, or a source that does have data never gets asked.
        """
        log_dir = self._log_dir(manifest)
        return bool(log_dir) and _has_event_files(log_dir)

    def list_keys(self, manifest: RunManifest) -> list[str]:
        """Scalar tags in the run's event files."""
        accumulator = self._accumulator(self._log_dir(manifest))
        if accumulator is None:
            return []
        try:
            return sorted(accumulator.Tags().get("scalars", []))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cannot list TensorBoard tags: %s", exc)
            return []

    def read(self, manifest: RunManifest, keys: list[str]) -> dict[str, Series]:
        """Read the requested scalar tags; absent tags are simply not returned."""
        return self._read_dir(self._log_dir(manifest), keys)

    # -------------------------------------------------------------- drill-down

    def workers(self, manifest: RunManifest) -> list[tuple[str, int]]:
        """The ``(group, rank)`` pairs this run wrote per-worker metrics for.

        Empty unless the run set ``runner.per_worker_log: true``, which is off by
        default. Sorted by group then rank so a chart's line order is stable
        across polls -- readdir order is not.
        """
        return sorted(self._worker_dirs(manifest).keys())

    def read_workers(
        self, manifest: RunManifest, keys: list[str]
    ) -> dict[str, list[Series]]:
        """Read the requested tags once per ``(group, rank)``.

        Returns:
            Key -> one series per worker that logged it, each tagged with its
            ``group`` and ``rank``. A worker that never logged a key is absent
            rather than present-and-empty: only some groups log any given metric
            (``env/*`` comes from the env group alone), and a flat line at zero
            for the others would be a chart that lies.
        """
        out: dict[str, list[Series]] = {}
        for (group, rank), log_dir in sorted(self._worker_dirs(manifest).items()):
            for key, series in self._read_dir(log_dir, keys).items():
                out.setdefault(key, []).append(
                    series.model_copy(update={"group": group, "rank": rank})
                )
        return out

    def _worker_dirs(self, manifest: RunManifest) -> dict[tuple[str, int], str]:
        """Map ``(group, rank)`` to the event directory holding its bundle.

        ``MetricLogger._get_scoped_logger`` builds
        ``<worker_logs>/<GroupName>/rank_<n>/tensorboard/``, so the two
        dimensions are recoverable from the path and no separate index has to be
        written or kept in sync. Directories with no event files are dropped: the
        bundle's ``tensorboard`` directory is created eagerly on first use, so an
        empty one means that rank logged nothing.
        """
        root = manifest.paths.get("worker_logs")
        if not root or not os.path.isdir(root):
            return {}

        found: dict[tuple[str, int], str] = {}
        try:
            groups = sorted(os.listdir(root))
        except OSError as exc:
            logger.debug("Cannot list per-worker log root %s: %s", root, exc)
            return {}

        for group in groups:
            group_dir = os.path.join(root, group)
            try:
                entries = sorted(os.listdir(group_dir))
            except OSError:
                continue
            for entry in entries:
                if not entry.startswith(_RANK_PREFIX):
                    continue
                try:
                    rank = int(entry[len(_RANK_PREFIX) :])
                except ValueError:
                    # Not a rank directory. Skipped rather than guessed at, so an
                    # unrelated directory under the root cannot become a "rank".
                    continue
                log_dir = os.path.join(group_dir, entry, "tensorboard")
                if _has_event_files(log_dir):
                    found[(group, rank)] = log_dir
        return found

    # ----------------------------------------------------------------- internals

    def _read_dir(self, log_dir: str | None, keys: list[str]) -> dict[str, Series]:
        """Read scalar tags from one event directory."""
        accumulator = self._accumulator(log_dir)
        if accumulator is None:
            return {}

        try:
            available = set(accumulator.Tags().get("scalars", []))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cannot list TensorBoard tags: %s", exc)
            return {}

        out: dict[str, Series] = {}
        for key in keys:
            if key not in available:
                continue
            try:
                events = accumulator.Scalars(key)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Cannot read scalar %s: %s", key, exc)
                continue
            out[key] = make_series(
                key,
                self.name,
                [(event.step, event.value, event.wall_time) for event in events],
            )
        return out

    def _log_dir(self, manifest: RunManifest) -> str | None:
        """Where this run's event files are.

        Read from ``manifest.paths``, which the training side resolved at launch.
        Guessing it here would mean reimplementing each runner's path convention
        -- the reason the manifest records paths in the first place.

        Two candidates per recorded root, in order: the root itself, then its
        ``all/`` subdirectory. A run with ``runner.per_worker_log: true`` writes
        the driver's aggregate bundle into ``tensorboard/all/`` and the per-rank
        bundles elsewhere, and manifests written before that was recorded name
        the parent -- a directory that exists and holds no event files. Trying
        ``all/`` is what keeps those runs readable; the parent is still tried
        first, so nothing changes for the common layout.
        """
        for root in (manifest.paths.get("tensorboard"), _default_root(manifest)):
            if not root:
                continue
            if _has_event_files(root):
                return root
            nested = os.path.join(root, _AGGREGATE_SUBDIR)
            if _has_event_files(nested):
                return nested
        # No event files anywhere. Fall back to the first candidate that is at
        # least a directory, so `available()` reports "configured but silent"
        # rather than "no such run" -- an empty log dir is a run that has not
        # logged yet, and the caller distinguishes the two.
        for root in (manifest.paths.get("tensorboard"), _default_root(manifest)):
            if root and os.path.isdir(root):
                return root
        return None

    def _accumulator(self, log_dir: str | None):
        if not log_dir:
            return None
        try:
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )
        except ImportError:  # pragma: no cover - declared as a hard dependency
            logger.warning("tensorboard is not installed; cannot read event files.")
            return None

        size = _event_bytes(log_dir)
        with self._lock:
            cached = self._cache.get(log_dir)
            if cached is not None:
                accumulator, last_size, last_reload = cached
                fresh = time.monotonic() - last_reload < 1.0
                if size == last_size and fresh:
                    return accumulator
                try:
                    accumulator.Reload()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Reload of %s failed: %s", log_dir, exc)
                self._cache[log_dir] = (accumulator, size, time.monotonic())
                return accumulator

            try:
                # size_guidance 0 means "keep every scalar". The default samples
                # scalars down to 1000 points per tag, which would silently drop
                # the spikes an RL curve is read for -- and decimation is a
                # decision for the gateway, where it is reported to the client.
                accumulator = EventAccumulator(log_dir, size_guidance={"scalars": 0})
                accumulator.Reload()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cannot open event files in %s: %s", log_dir, exc)
                return None
            self._cache[log_dir] = (accumulator, size, time.monotonic())
            return accumulator


def _default_root(manifest: RunManifest) -> str | None:
    """``<log_path>/tensorboard``, the convention every runner follows.

    A fallback for older manifests that recorded only ``log_path``; the manifest
    is what knows where the run actually wrote.
    """
    log_path = manifest.paths.get("log_path")
    return os.path.join(log_path, "tensorboard") if log_path else None


def _event_files(log_dir: str) -> list[str]:
    try:
        return [
            os.path.join(log_dir, name)
            for name in os.listdir(log_dir)
            if name.startswith("events.out.tfevents.")
        ]
    except OSError:
        return []


def _has_event_files(log_dir: str) -> bool:
    return bool(_event_files(log_dir))


def _event_bytes(log_dir: str) -> int:
    """Total size of the event files, as a change detector.

    Total rather than newest-only: a run restarted in the same directory opens a
    second event file, and the newest one starting small would otherwise look
    like the data shrank.
    """
    total = 0
    for path in _event_files(log_dir):
        try:
            total += os.path.getsize(path)
        except OSError:
            continue
    return total
