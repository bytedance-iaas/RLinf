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

"""Reading one run's control plane.

Every read here is a plain ``open`` on a file the training side published
atomically (temp file plus ``os.replace``), so a reader never sees a half-written
snapshot and no locking is involved. That property is why the control plane is
files rather than SQLite: zero dependencies, readable across venvs, survives a
crash, and works on an NFS mount where SQLite locking does not.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from .discovery import DiscoveredRun
from .health import derive_health
from .models import (
    CheckpointEntry,
    Health,
    HealthVerdict,
    MediaEntry,
    RunSnapshot,
    RunStatus,
    RunSummary,
)
from .relocate import relocate_paths
from .settings import Settings

logger = logging.getLogger(__name__)

RUN_JSON = "run.json"
HEARTBEAT_FILE = "heartbeat"
EVENTS_FILE = "events.jsonl"
CHECKPOINTS_FILE = "checkpoints.jsonl"
MEDIA_PREFIX = "media.rank"


@dataclass(frozen=True)
class Event:
    """One line of ``events.jsonl``."""

    ts: str | None
    kind: str
    step: int | None
    payload: dict


#: `read_snapshot`'s marker for "the file is not there", as distinct from "the
#: file is there and wrong". Only the first of the two can be a normal startup.
_SNAPSHOT_MISSING = "run.json has not been written yet."


class StateStore:
    """Read and interpret the control-plane files for discovered runs."""

    def __init__(self, settings: Settings):
        self._settings = settings

    # -------------------------------------------------------------------- status

    def read_snapshot(self, run_root: str) -> tuple[RunSnapshot | None, str | None]:
        """Parse ``run.json``.

        Returns:
            ``(snapshot, error)``. The snapshot is ``None`` with an error message
            when the file is missing or unparseable -- which happens for real: a
            run that dies between the reporter's ``makedirs`` and its first flush
            has a manifest but no snapshot yet.
        """
        path = os.path.join(run_root, RUN_JSON)
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            # Absence is not damage. A run that has a manifest and no snapshot is
            # usually still starting up, and `status` decides which of the two it
            # is; this sentinel is what lets it tell them apart.
            return None, _SNAPSHOT_MISSING
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"run.json is unreadable: {exc}"

        try:
            return RunSnapshot.model_validate(payload), None
        except Exception as exc:  # noqa: BLE001 - never 500 on a bad snapshot
            logger.warning("Invalid snapshot at %s: %s", path, exc)
            return None, f"run.json does not match the v2 contract: {exc}"

    def status(self, run: DiscoveredRun, now: datetime | None = None) -> RunStatus:
        """Build the full status for one run, including the health verdict."""
        now = now or datetime.now(timezone.utc)
        snapshot, error = self.read_snapshot(run.run_root)
        heartbeat_interval_s = (
            run.manifest.heartbeat_interval_s or self._settings.heartbeat_interval_s
        )
        verdict = derive_health(
            snapshot,
            now,
            heartbeat_timeout_k=self._settings.heartbeat_timeout_k,
            progress_timeout_k=self._settings.progress_timeout_k,
            timeout_floor_s=self._settings.timeout_floor_s,
            heartbeat_interval_s=heartbeat_interval_s,
        )
        verdict = self._refine_with_heartbeat_file(
            run.run_root, verdict, now, heartbeat_interval_s
        )

        # The snapshot carries its own copy of the launch-time paths, so it needs
        # the same repair discovery already applied to the manifest -- otherwise
        # the two documents disagree about where this run's files are.
        if snapshot is not None and run.relocation is not None:
            paths, _ = relocate_paths(
                snapshot.paths, snapshot.paths.get("run_root"), run.run_root
            )
            snapshot = snapshot.model_copy(update={"paths": paths})

        initializing, startup_s = self._startup_state(run, error, now)
        if initializing:
            # The startup window is the expected state here, not a fault, so it
            # does not also get reported as one. Past the deadline the error is
            # restored and says what actually failed.
            error = None
            # The verdict stays `unknown` -- nothing has been reported, so
            # nothing can be concluded -- but its reason is what the operator
            # reads on the health bar, and "no readable run.json" describes a
            # normal launch as a broken file.
            verdict = verdict.model_copy(
                update={"reason": "Starting up: no snapshot published yet."}
            )

        return RunStatus(
            run_id=run.run_id,
            manifest=run.manifest,
            snapshot=snapshot,
            health=verdict,
            run_root=run.run_root,
            error=error,
            relocation=run.relocation,
            has_media=self.has_media(run.run_root),
            initializing=initializing,
            startup_elapsed_s=startup_s,
        )

    def _startup_state(
        self, run: DiscoveredRun, error: str | None, now: datetime
    ) -> tuple[bool, float | None]:
        """Whether a snapshot-less run is still within its startup window.

        A run is registered by its manifest and starts reporting when the
        training loop opens its lifecycle. Everything in between is a normal
        startup, however long it looks from outside, so it is reported as such
        until the configured deadline passes.

        Returns:
            ``(initializing, seconds_in_startup)``. The duration is reported even
            once the deadline has passed, because "stuck for 20 minutes" is the
            fact an operator needs and "stuck" alone is not.
        """
        if error != _SNAPSHOT_MISSING:
            return False, None
        started_at = run.manifest.started_at if run.manifest else None
        if started_at is None:
            # No launch time to measure against. Treat it as starting up rather
            # than as broken: a manifest with no timestamp is a producer-side
            # gap, and inventing an alarm from a missing field helps nobody.
            return True, None
        elapsed = (now - started_at).total_seconds()
        return elapsed <= self._settings.startup_grace_s, max(elapsed, 0.0)

    def summary(self, run: DiscoveredRun, now: datetime | None = None) -> RunSummary:
        """Build the flat list-view row for one run."""
        status = self.status(run, now)
        snapshot = status.snapshot
        manifest = run.manifest
        return RunSummary(
            run_id=run.run_id,
            task_type=(snapshot.task_type if snapshot else manifest.task_type),
            experiment_name=manifest.experiment_name,
            state=snapshot.state if snapshot else None,
            health=status.health.health,
            phase=snapshot.phase if snapshot else None,
            step=snapshot.progress.step if snapshot else 0,
            max_steps=snapshot.progress.max_steps if snapshot else None,
            step_semantics=(
                snapshot.progress.step_semantics
                if snapshot
                else manifest.step_semantics
            ),
            started_at=(
                snapshot.timing.started_at if snapshot else manifest.started_at
            ),
            heartbeat_at=snapshot.heartbeat_at if snapshot else None,
            elapsed_s=snapshot.timing.elapsed_s if snapshot else 0.0,
            eta_s=snapshot.timing.eta_s if snapshot else None,
            latest_checkpoint_step=(
                snapshot.latest_checkpoint.step
                if snapshot and snapshot.latest_checkpoint
                else None
            ),
            run_root=run.run_root,
            initializing=status.initializing,
            startup_elapsed_s=status.startup_elapsed_s,
        )

    def _refine_with_heartbeat_file(
        self,
        run_root: str,
        verdict: HealthVerdict,
        now: datetime,
        heartbeat_interval_s: float,
    ) -> HealthVerdict:
        """Distinguish "process gone" from "snapshot writes are failing".

        The training side writes a tiny ``heartbeat`` file on every tick in
        addition to re-rendering ``run.json``, specifically so a reader still has
        an mtime to fall back on when rendering the snapshot is what is broken
        (a full disk, a permissions change mid-run). A fresh mtime alongside a
        stale ``heartbeat_at`` means the process is alive but its snapshot is
        frozen: degraded, not unreachable. Calling that unreachable would send
        someone to restart a run that is still training.
        """
        if verdict.health != Health.UNREACHABLE:
            return verdict

        path = os.path.join(run_root, HEARTBEAT_FILE)
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            return verdict

        age = (now - datetime.fromtimestamp(mtime, tz=timezone.utc)).total_seconds()
        heartbeat_budget = verdict.heartbeat_budget_s or (
            self._settings.heartbeat_timeout_k * heartbeat_interval_s
        )
        if age <= heartbeat_budget:
            return verdict.model_copy(
                update={
                    "health": Health.DEGRADED,
                    "reason": (
                        f"run.json is stale but the heartbeat file was touched "
                        f"{age:.0f}s ago; the process is alive and its snapshot "
                        f"writes are failing."
                    ),
                }
            )
        return verdict

    # -------------------------------------------------------------------- tables

    def read_events(self, run_root: str, limit: int = 200) -> list[Event]:
        """Return the most recent events, oldest first.

        Read whole and sliced rather than seeked backwards: this file gets a
        handful of lines per step, so it stays small, and tail-seeking JSONL
        costs correctness (a partial first line) for no measurable gain.
        """
        rows = _read_jsonl(os.path.join(run_root, EVENTS_FILE))
        if limit > 0:
            rows = rows[-limit:]
        return [
            Event(
                ts=row.get("ts"),
                kind=str(row.get("kind", "unknown")),
                step=row.get("step"),
                payload=row.get("payload") or {},
            )
            for row in rows
        ]

    def read_checkpoints(self, run_root: str) -> list[CheckpointEntry]:
        """Return indexed checkpoints, newest first.

        Only completed saves appear: the training side appends after the save
        returns. That append-after-completion ordering is the whole reason no
        ``WRITING``/``READY`` protocol is needed -- a listed checkpoint is always
        loadable.
        """
        entries: list[CheckpointEntry] = []
        for row in _read_jsonl(os.path.join(run_root, CHECKPOINTS_FILE)):
            try:
                entries.append(CheckpointEntry.model_validate(row))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping malformed checkpoint row: %s", exc)
        entries.sort(key=lambda e: e.step, reverse=True)
        return entries

    def read_media(self, run_root: str) -> list[MediaEntry]:
        """Merge the per-shard media indexes.

        Sharded (``media.rank<k>.jsonl``) because videos are encoded inside env
        worker processes: one file per writer keeps every file single-writer, so
        appends need no cross-process locking.

        Ordering falls back to mtime when a row has no step. The training side
        writes ``step: null`` when it cannot resolve one (an env worker that
        never received ``set_global_step``), and mtime order is still the order
        they were recorded in.
        """
        entries: list[MediaEntry] = []
        try:
            names = sorted(os.listdir(run_root))
        except OSError:
            return []

        for name in names:
            if not name.startswith(MEDIA_PREFIX) or not name.endswith(".jsonl"):
                continue
            for row in _read_jsonl(os.path.join(run_root, name)):
                try:
                    entries.append(MediaEntry.model_validate(row))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Skipping malformed media row: %s", exc)

        def sort_key(entry: MediaEntry) -> tuple[int, int, str]:
            has_step = 0 if entry.step is None else 1
            return (has_step, entry.step or 0, entry.path)

        entries.sort(key=sort_key)
        return entries

    def has_media(self, run_root: str) -> bool:
        """Whether this run recorded any video, without parsing the index.

        Answers the one question the UI needs to decide whether a Media view is
        worth offering, and answers it from a directory listing plus at most one
        ``st_size`` per shard. Parsing every row -- thousands, on a long embodied
        run -- to learn a boolean would put that cost on every status read, and
        this is called once per run page.

        A shard that exists but is empty counts as no media: the writer is created
        when an env worker starts, before any clip is encoded, so the file's
        presence alone would promise a view that renders nothing.
        """
        try:
            names = os.listdir(run_root)
        except OSError:
            return False

        for name in names:
            if not name.startswith(MEDIA_PREFIX) or not name.endswith(".jsonl"):
                continue
            try:
                if os.stat(os.path.join(run_root, name)).st_size > 0:
                    return True
            except OSError:
                continue
        return False


def _read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file, skipping unparseable lines.

    A truncated final line is normal, not corruption: these files are appended
    to by a live process, and a reader can arrive mid-write. Skipping the line is
    correct -- it will be complete on the next read.
    """
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return []
    return rows
