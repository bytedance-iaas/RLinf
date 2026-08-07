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

"""Finding runs on disk.

Discovery is anchored on ``_rlinf/runs/<run_id>/`` and never guesses paths.
That matters because runners disagree about where everything else lives:
embodied/offline/sft build checkpoint paths from ``runner.logger.log_path`` while
reasoning/coding use ``runner.output_dir``, and metrics sit under
``log_path/tensorboard`` either way. The control-plane directory is the one
location every runner agrees on, and ``manifest.json`` inside it records where
the rest went.

A directory counts as a run only if ``manifest.json`` is present and parseable.
The training side writes the manifest in the reporter's constructor, before any
worker launches, so anything without one is a directory that was created and
then abandoned -- listing it would only add noise.

Those recorded paths are absolute and launch-time, so this is also the one place
that can repair them: it is the only code that knows both where the manifest
*says* the run root is and where it actually found it. See :mod:`.relocate`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

from .models import RunManifest
from .relocate import relocate_paths
from .settings import Settings

logger = logging.getLogger(__name__)

CONTROL_DIR_NAME = "_rlinf"
RUNS_DIR_NAME = "runs"
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class DiscoveredRun:
    """A run directory that has a readable manifest."""

    run_id: str
    run_root: str
    manifest: RunManifest
    #: mtime of ``manifest.json``, used only to order runs when they carry no
    #: ``started_at`` (an older training side, or a truncated write).
    manifest_mtime: float
    #: Set when the manifest's launch-time paths had to be translated into this
    #: machine's namespace, describing the prefix mapping that was applied.
    #: ``None`` when the recorded paths were used as written.
    relocation: dict[str, str] | None = None


class RunDiscovery:
    """Scan the configured roots for runs, with a short-lived cache.

    The cache exists because a scan stats one file per candidate run, and on an
    NFS-mounted scan root that cost shows up directly in ``/runs`` latency. The
    TTL is seconds, so a newly launched run appears essentially immediately.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._cache: list[DiscoveredRun] = []
        self._cache_at: float = 0.0

    def list_runs(self, *, refresh: bool = False) -> list[DiscoveredRun]:
        """Return discovered runs, newest first.

        Args:
            refresh: Bypass the cache. Used by endpoints where staleness would
                be user-visible, such as an explicit reload.

        Returns:
            Runs sorted by start time descending; runs whose start time is
            unknown sort by manifest mtime instead.
        """
        now = time.monotonic()
        if (
            not refresh
            and self._cache
            and now - self._cache_at < self._settings.discovery_cache_ttl_s
        ):
            return self._cache

        runs: dict[str, DiscoveredRun] = {}
        for root in self._settings.scan_roots:
            for run in self._scan_root(os.path.expanduser(root)):
                # The same run can be reachable through two overlapping scan
                # roots (e.g. `/data/logs` and `/data`). Key on the resolved path
                # so it is listed once.
                runs[os.path.realpath(run.run_root)] = run

        ordered = sorted(runs.values(), key=_sort_key, reverse=True)
        self._cache = ordered
        self._cache_at = now
        return ordered

    def find(self, run_id: str) -> DiscoveredRun | None:
        """Look up one run by id, refreshing if it is not already known.

        A miss triggers one refresh so that a run launched moments ago is
        reachable by direct URL without waiting out the cache TTL.
        """
        for run in self.list_runs():
            if run.run_id == run_id:
                return run
        for run in self.list_runs(refresh=True):
            if run.run_id == run_id:
                return run
        return None

    # ------------------------------------------------------------------ scanning

    def _scan_root(self, root: str) -> list[DiscoveredRun]:
        if not os.path.isdir(root):
            logger.debug("Scan root does not exist: %s", root)
            return []

        found: list[DiscoveredRun] = []
        root_depth = root.rstrip(os.sep).count(os.sep)

        for dirpath, dirnames, _filenames in os.walk(root, followlinks=False):
            depth = dirpath.count(os.sep) - root_depth
            if depth >= self._settings.scan_max_depth:
                dirnames[:] = []
                continue

            if CONTROL_DIR_NAME not in dirnames:
                # Skip the heavy, uninteresting subtrees a log path accumulates.
                # Without this a scan root containing checkpoints walks tens of
                # thousands of shard files to find nothing.
                dirnames[:] = [d for d in dirnames if not _is_noise(d)]
                continue

            found.extend(
                self._scan_control_dir(os.path.join(dirpath, CONTROL_DIR_NAME))
            )
            # Nothing below a control directory is another run.
            dirnames[:] = [d for d in dirnames if d != CONTROL_DIR_NAME]

        return found

    def _scan_control_dir(self, control_dir: str) -> list[DiscoveredRun]:
        runs_dir = os.path.join(control_dir, RUNS_DIR_NAME)
        if not os.path.isdir(runs_dir):
            return []

        found: list[DiscoveredRun] = []
        try:
            entries = sorted(os.scandir(runs_dir), key=lambda e: e.name)
        except OSError as exc:
            logger.warning("Cannot list %s: %s", runs_dir, exc)
            return []

        for entry in entries:
            # `latest` is a sibling of `runs/`, so anything symlinked in here is
            # unexpected; resolving it could also double-count a run.
            if not entry.is_dir(follow_symlinks=False):
                continue
            run = self._load(entry.path)
            if run is not None:
                found.append(run)
        return found

    def _load(self, run_root: str) -> DiscoveredRun | None:
        manifest_path = os.path.join(run_root, MANIFEST_NAME)
        try:
            stat = os.stat(manifest_path)
            with open(manifest_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            # A directory with no manifest is not a run. The training side writes
            # the manifest in the reporter constructor, so this is a leftover.
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable manifest %s: %s", manifest_path, exc)
            return None

        run_id = payload.get("run_id") or os.path.basename(run_root)
        try:
            manifest = RunManifest.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 - a bad manifest must not 500
            logger.warning("Ignoring invalid manifest %s: %s", manifest_path, exc)
            return None

        # Repair launch-time absolute paths if this tree is being read from a
        # different namespace than it was written in (container vs host mount,
        # or a tree copied off the cluster). Without this the metric reader finds
        # no event files and the run renders with no charts and no explanation.
        paths, relocation = relocate_paths(
            manifest.paths, manifest.paths.get("run_root"), run_root
        )
        if relocation is not None:
            manifest = manifest.model_copy(update={"paths": paths})

        return DiscoveredRun(
            run_id=run_id,
            run_root=run_root,
            manifest=manifest,
            manifest_mtime=stat.st_mtime,
            relocation=relocation,
        )


def _sort_key(run: DiscoveredRun) -> tuple[float, str]:
    started = run.manifest.started_at
    if started is not None:
        return (started.timestamp(), run.run_id)
    return (run.manifest_mtime, run.run_id)


#: Directory names never worth walking into while looking for `_rlinf`. These are
#: the large data-plane and checkpoint trees that share a log path with it.
_NOISE_DIRS = frozenset(
    {
        "checkpoints",
        "tensorboard",
        "wandb",
        "swanlab",
        "video",
        "worker_logs",
        "__pycache__",
        ".git",
        "node_modules",
    }
)


def _is_noise(name: str) -> bool:
    return name in _NOISE_DIRS or name.startswith("global_step_")
