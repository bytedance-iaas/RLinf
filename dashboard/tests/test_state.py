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

"""Reading one run's control-plane files.

Where ``test_health.py`` feeds the derivation timestamps, this feeds it files --
and files written by a live process, which is the harder input. A reader arrives
mid-append, mid-replace, or after the writer died holding a lock it never had. The
tests below are the failure modes that showed up in practice, not hypotheticals.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from rlinf_dashboard.discovery import RunDiscovery
from rlinf_dashboard.models import Health, RunState
from rlinf_dashboard.state import StateStore

NOW = datetime(2026, 8, 3, 15, 0, 0, tzinfo=timezone.utc)


def _snapshot(**overrides) -> dict:
    payload = {
        "schema_version": 2,
        "run_id": "test-run",
        "task_type": "embodied",
        "state": "running",
        "phase": "rollout",
        "heartbeat_at": _iso(NOW - timedelta(seconds=3)),
        "heartbeat_seq": 42,
        "last_progress_at": _iso(NOW - timedelta(seconds=10)),
        "last_metric_at": _iso(NOW - timedelta(seconds=10)),
        "progress": {"step": 7, "max_steps": 100, "step_semantics": "rl_iteration"},
        "timing": {
            "started_at": _iso(NOW - timedelta(seconds=300)),
            "elapsed_s": 300.0,
            "step_time_p50": 40.0,
            "eta_s": 3720.0,
        },
    }
    payload.update(overrides)
    return payload


def _iso(when: datetime) -> str:
    return when.isoformat().replace("+00:00", "Z")


def _store_and_run(run_tree, settings_for, run_id="test-run", **kwargs):
    run_tree(run_id, **kwargs)
    settings = settings_for()
    run = RunDiscovery(settings).find(run_id)
    assert run is not None
    return StateStore(settings), run


# ------------------------------------------------------------------------ snapshots


def test_reads_a_published_snapshot(run_tree, settings_for):
    store, run = _store_and_run(run_tree, settings_for, snapshot=_snapshot())
    snapshot, error = store.read_snapshot(run.run_root)

    assert error is None
    assert snapshot is not None
    assert snapshot.state is RunState.RUNNING
    assert snapshot.phase == "rollout"
    assert snapshot.progress.step == 7


def test_a_run_with_no_snapshot_yet_reports_why(run_tree, settings_for):
    """A manifest without a ``run.json`` is a real, reachable state.

    The manifest is written in the reporter's constructor and the first snapshot
    flush comes later, so a run that dies in between leaves exactly this. It must
    render as "hasn't published yet", not as an error page.
    """
    store, run = _store_and_run(run_tree, settings_for)
    snapshot, error = store.read_snapshot(run.run_root)

    assert snapshot is None
    assert error and "not been written" in error

    status = store.status(run, NOW)
    assert status.health.health is Health.UNKNOWN
    assert status.error == error


def test_a_corrupt_snapshot_does_not_500(run_tree, settings_for):
    """Never let one bad file take down the endpoint.

    Truncation is expected on a crash mid-write, and the run's identity, manifest
    and checkpoint history are all still readable and worth showing.
    """
    store, run = _store_and_run(run_tree, settings_for)
    with open(os.path.join(run.run_root, "run.json"), "w") as handle:
        handle.write('{"schema_version": 2, "run_id": "test-run", "sta')

    snapshot, error = store.read_snapshot(run.run_root)
    assert snapshot is None
    assert error and "unreadable" in error

    status = store.status(run, NOW)
    assert status.run_id == "test-run"
    assert status.manifest is not None


def test_a_schema_violating_snapshot_is_reported_as_such(run_tree, settings_for):
    """Valid JSON that is not a v2 snapshot gets a different message than garbage.

    The two mean different things -- a truncated write versus a version mismatch --
    and the second is only diagnosable if the message says so.
    """
    store, run = _store_and_run(
        run_tree, settings_for, snapshot={"schema_version": 2, "hello": "world"}
    )
    snapshot, error = store.read_snapshot(run.run_root)
    assert snapshot is None
    assert error and "v2 contract" in error


# --------------------------------------------------------------------------- status


def test_status_composes_snapshot_manifest_and_health(run_tree, settings_for):
    store, run = _store_and_run(run_tree, settings_for, snapshot=_snapshot())
    status = store.status(run, NOW)

    assert status.snapshot is not None
    assert status.manifest is not None
    assert status.health.health is Health.HEALTHY
    assert status.health.reason
    assert status.run_root == run.run_root


def test_summary_is_flat_enough_to_sort_on(run_tree, settings_for):
    """The list view sorts and filters on these fields, so they must be populated.

    A summary that leaves ``step`` or ``eta_s`` null forces the frontend back into
    the full status document per row, which is what the flat shape exists to
    avoid.
    """
    store, run = _store_and_run(run_tree, settings_for, snapshot=_snapshot())
    summary = store.summary(run, NOW)

    assert summary.run_id == "test-run"
    assert summary.state is RunState.RUNNING
    assert summary.health is Health.HEALTHY
    assert summary.step == 7
    assert summary.max_steps == 100
    assert summary.step_semantics == "rl_iteration"
    assert summary.eta_s == 3720.0
    assert summary.experiment_name == "test-exp"


def test_summary_falls_back_to_the_manifest_when_there_is_no_snapshot(
    run_tree, settings_for
):
    """A run that never published still belongs in the list.

    Its identity, task type and step semantics all come from the manifest, so the
    row is informative even with no snapshot behind it.
    """
    store, run = _store_and_run(run_tree, settings_for)
    summary = store.summary(run, NOW)

    assert summary.run_id == "test-run"
    assert summary.task_type == "embodied"
    assert summary.state is None
    assert summary.health is Health.UNKNOWN
    assert summary.step_semantics == "rl_iteration"


def test_summary_carries_the_latest_checkpoint_step(run_tree, settings_for):
    store, run = _store_and_run(
        run_tree,
        settings_for,
        snapshot=_snapshot(
            latest_checkpoint={
                "step": 3,
                "path": "/logs/exp/checkpoints/global_step_3",
                "size_bytes": 20350836699,
                "duration_s": 77.4,
            }
        ),
    )
    assert store.summary(run, NOW).latest_checkpoint_step == 3


# ------------------------------------------------------- the heartbeat-file refinement


def test_a_fresh_heartbeat_file_downgrades_unreachable_to_degraded(
    run_tree, settings_for
):
    """Alive-but-not-publishing is not the same as gone.

    The training side touches a tiny ``heartbeat`` file on every tick *as well as*
    re-rendering ``run.json``, so that a reader still has an mtime when rendering
    the snapshot is what broke -- a full disk, a permissions change mid-run.
    Calling that unreachable sends someone to restart a run that is still
    training.
    """
    store, run = _store_and_run(
        run_tree,
        settings_for,
        snapshot=_snapshot(heartbeat_at=_iso(NOW - timedelta(hours=1))),
        heartbeat_seq=99,
    )
    os.utime(os.path.join(run.run_root, "heartbeat"), (NOW.timestamp(),) * 2)

    verdict = store.status(run, NOW).health
    assert verdict.health is Health.DEGRADED
    assert "snapshot writes are failing" in verdict.reason


def test_a_stale_heartbeat_file_leaves_the_run_unreachable(run_tree, settings_for):
    """Both signals stale is the ``kill -9`` case, and must read as gone."""
    store, run = _store_and_run(
        run_tree,
        settings_for,
        snapshot=_snapshot(heartbeat_at=_iso(NOW - timedelta(hours=1))),
        heartbeat_seq=99,
    )
    os.utime(
        os.path.join(run.run_root, "heartbeat"),
        ((NOW - timedelta(hours=1)).timestamp(),) * 2,
    )
    assert store.status(run, NOW).health.health is Health.UNREACHABLE


def test_no_heartbeat_file_leaves_the_verdict_alone(run_tree, settings_for):
    """The refinement only ever softens a verdict, never invents one.

    An older training side writes no heartbeat file at all; the snapshot's own
    timestamp is then the only evidence, and it says the process is gone.
    """
    store, run = _store_and_run(
        run_tree,
        settings_for,
        snapshot=_snapshot(heartbeat_at=_iso(NOW - timedelta(hours=1))),
    )
    assert store.status(run, NOW).health.health is Health.UNREACHABLE


def test_the_refinement_never_touches_a_healthy_verdict(run_tree, settings_for):
    """A stale heartbeat *file* alongside a fresh snapshot proves nothing bad.

    The snapshot is the richer signal and it is current; the file is a fallback
    for when the snapshot is not. Letting the fallback override the primary would
    invert the whole point.
    """
    store, run = _store_and_run(
        run_tree, settings_for, snapshot=_snapshot(), heartbeat_seq=1
    )
    os.utime(
        os.path.join(run.run_root, "heartbeat"),
        ((NOW - timedelta(days=1)).timestamp(),) * 2,
    )
    assert store.status(run, NOW).health.health is Health.HEALTHY


# ---------------------------------------------------------------------------- tables


def test_reads_events_oldest_first_and_honours_the_limit(run_tree, settings_for):
    events = [
        {"ts": _iso(NOW), "kind": "run_start", "step": 0, "payload": {}},
        {
            "ts": _iso(NOW),
            "kind": "phase_enter",
            "step": 1,
            "payload": {"phase": "rollout"},
        },
        {"ts": _iso(NOW), "kind": "ckpt_saved", "step": 1, "payload": {"path": "/x"}},
    ]
    store, run = _store_and_run(run_tree, settings_for, events=events)

    assert [e.kind for e in store.read_events(run.run_root)] == [
        "run_start",
        "phase_enter",
        "ckpt_saved",
    ]
    # The tail is what matters on a long run: recent events, in order.
    assert [e.kind for e in store.read_events(run.run_root, limit=2)] == [
        "phase_enter",
        "ckpt_saved",
    ]


def test_a_truncated_final_jsonl_line_is_skipped_not_fatal(run_tree, settings_for):
    """Arriving mid-append is normal, not corruption.

    These files are written by a live process. The partial line will be complete
    on the next read; dropping it and returning the rest is the only correct
    behaviour, and refusing to parse would blank the event log every few seconds.
    """
    store, run = _store_and_run(
        run_tree,
        settings_for,
        events=[{"ts": _iso(NOW), "kind": "run_start", "step": 0, "payload": {}}],
    )
    with open(os.path.join(run.run_root, "events.jsonl"), "a") as handle:
        handle.write('{"ts": "2026-08-03T15:00:05Z", "kind": "phase_ent')

    events = store.read_events(run.run_root)
    assert [e.kind for e in events] == ["run_start"]


def test_reads_checkpoints_newest_first(run_tree, settings_for):
    """Newest first because the newest is the one someone wants to resume from."""
    rows = [
        {"step": 1, "path": "/ckpt/global_step_1", "size_bytes": 100},
        {"step": 3, "path": "/ckpt/global_step_3", "size_bytes": 300, "is_best": True},
        {"step": 2, "path": "/ckpt/global_step_2", "size_bytes": 200},
    ]
    store, run = _store_and_run(run_tree, settings_for, checkpoints=rows)

    entries = store.read_checkpoints(run.run_root)
    assert [e.step for e in entries] == [3, 2, 1]
    assert entries[0].is_best


def test_only_completed_checkpoints_are_listed(run_tree, settings_for):
    """A4, restated from the read side.

    The index row is appended *after* the save returns, so a listed checkpoint is
    always loadable. That ordering is the entire visibility mechanism -- there is
    no WRITING/READY protocol to consult, and this test pins the assumption the
    absence of one rests on: what is in the file is what finished.
    """
    store, run = _store_and_run(
        run_tree,
        settings_for,
        checkpoints=[{"step": 1, "path": "/ckpt/global_step_1", "duration_s": 77.4}],
    )
    entries = store.read_checkpoints(run.run_root)
    assert len(entries) == 1
    assert entries[0].duration_s == 77.4


def test_a_malformed_checkpoint_row_is_skipped(run_tree, settings_for):
    store, run = _store_and_run(
        run_tree,
        settings_for,
        checkpoints=[{"step": 1, "path": "/ckpt/global_step_1"}],
    )
    with open(os.path.join(run.run_root, "checkpoints.jsonl"), "a") as handle:
        handle.write(json.dumps({"no_step_field": True}) + "\n")

    assert [e.step for e in store.read_checkpoints(run.run_root)] == [1]


def test_media_shards_are_merged(run_tree, settings_for):
    """Sharded because videos are encoded inside env worker processes.

    One file per writer keeps every file single-writer, so appends need no
    cross-process locking -- but the reader has to put them back together.
    """
    media = {
        0: [
            {"path": "/v/r0_s1.mp4", "step": 1, "shard": 0, "split": "train"},
            {"path": "/v/r0_s2.mp4", "step": 2, "shard": 0, "split": "train"},
        ],
        1: [{"path": "/v/r1_s1.mp4", "step": 1, "shard": 1, "split": "train"}],
        3: [{"path": "/v/r3_s2.mp4", "step": 2, "shard": 3, "split": "eval"}],
    }
    store, run = _store_and_run(run_tree, settings_for, media=media)

    entries = store.read_media(run.run_root)
    assert len(entries) == 4
    assert {e.shard for e in entries} == {0, 1, 3}
    assert [e.step for e in entries] == [1, 1, 2, 2]


def test_media_rows_without_a_step_sort_first(run_tree, settings_for):
    """``step: null`` is written when an env worker never got ``set_global_step``.

    Those rows are still real clips. Sorting them ahead of the numbered ones keeps
    them out of the way of the recency slice, which is what the UI shows by
    default.
    """
    media = {
        0: [
            {"path": "/v/known.mp4", "step": 5, "shard": 0},
            {"path": "/v/unknown.mp4", "step": None, "shard": 0},
        ]
    }
    store, run = _store_and_run(run_tree, settings_for, media=media)

    entries = store.read_media(run.run_root)
    assert [os.path.basename(e.path) for e in entries] == ["unknown.mp4", "known.mp4"]


def test_has_media_is_true_when_a_shard_holds_rows(run_tree, settings_for):
    """The cheap answer to "is a Media tab worth offering".

    Asserted alongside ``read_media`` rather than instead of it: the whole point
    of the predicate is that it does *not* parse, so the two could drift apart and
    only the pair of assertions catches it.
    """
    store, run = _store_and_run(
        run_tree,
        settings_for,
        snapshot=_snapshot(),
        media={0: [{"path": "/v/r0_s1.mp4", "step": 1, "shard": 0}]},
    )

    assert store.has_media(run.run_root) is True
    assert len(store.read_media(run.run_root)) == 1
    assert store.status(run, NOW).has_media is True


def test_has_media_is_false_with_no_shards(run_tree, settings_for):
    """The common case, including every SFT and reasoning run.

    ``enable_dump_video`` defaults off, so most embodied runs land here too. A
    Media tab offered to any of them is a tab that always leads to an empty page.
    """
    store, run = _store_and_run(run_tree, settings_for, snapshot=_snapshot())

    assert store.has_media(run.run_root) is False
    assert store.status(run, NOW).has_media is False


def test_an_empty_shard_does_not_count_as_media(run_tree, settings_for):
    """A shard exists before any clip is encoded.

    The writer is created when an env worker starts, so between that moment and
    the first ``flush_video`` there is a zero-byte ``media.rank<k>.jsonl`` on
    disk. Treating the file's presence as the signal would promise a view that
    renders nothing -- which is why the predicate checks ``st_size`` and not
    ``os.path.exists``.
    """
    store, run = _store_and_run(
        run_tree, settings_for, snapshot=_snapshot(), media={0: []}
    )

    assert os.path.exists(os.path.join(run.run_root, "media.rank0.jsonl"))
    assert store.has_media(run.run_root) is False
    assert store.read_media(run.run_root) == []


def test_has_media_ignores_files_that_are_not_shards(run_tree, settings_for):
    """Negative control on the name match.

    The run root is a directory operators poke at, and video lives under it in
    some layouts. Neither an mp4 nor a similarly-named log may satisfy the
    predicate, or the tab appears for a run with no index to render from.
    """
    store, run = _store_and_run(run_tree, settings_for, snapshot=_snapshot())
    for name in ("media.rank0.mp4", "media.jsonl", "mediarank0.jsonl"):
        with open(os.path.join(run.run_root, name), "w") as handle:
            handle.write("not an index\n")

    assert store.has_media(run.run_root) is False


def test_has_media_on_a_missing_run_root_is_false(run_tree, settings_for):
    """An unmounted scan root must not raise on the way to a boolean.

    ``status()`` calls this on every read, so an ``OSError`` here would turn a
    stale mount into a 500 for the whole run rather than a run with no video.
    """
    store, run = _store_and_run(run_tree, settings_for, snapshot=_snapshot())

    assert store.has_media(os.path.join(run.run_root, "does-not-exist")) is False


def test_missing_table_files_read_as_empty(run_tree, settings_for):
    """Absence is the normal early state, not an error.

    A run logs no checkpoints until its first save and no media unless video
    recording is on, so every one of these is empty for most runs most of the
    time.
    """
    store, run = _store_and_run(run_tree, settings_for, snapshot=_snapshot())
    assert store.read_events(run.run_root) == []
    assert store.read_checkpoints(run.run_root) == []
    assert store.read_media(run.run_root) == []


# ------------------------------------------------------------------ read latency


def test_a_single_run_status_read_stays_under_the_budget(run_tree, settings_for):
    """One run's status must read in under 10ms.

    The number matters because the list view and every SSE tick read *per run*: a
    50-run tree at 10ms is already half a second of syscalls per push. The median
    of repeated reads is asserted rather than a single one -- a cold first read
    pays for the page cache and for pydantic building its validators, neither of
    which is what the budget is about, and asserting on a single timing is how a
    perf test becomes a flaky test.

    Generous inputs on purpose: the snapshot carries every optional block, and the
    tables carry more rows than a short run would.
    """
    import statistics
    import time

    store, run = _store_and_run(
        run_tree,
        settings_for,
        snapshot=_snapshot(
            components={
                name: {"active": True, "since": _iso(NOW - timedelta(seconds=60))}
                for name in ("env", "rollout", "actor")
            },
            latest_checkpoint={"step": 5, "path": "/ckpt/global_step_5"},
        ),
        events=[
            {"ts": _iso(NOW), "kind": "phase_enter", "payload": {"phase": "rollout"}}
            for _ in range(200)
        ],
        checkpoints=[{"step": step, "path": f"/ckpt/{step}"} for step in range(50)],
        heartbeat_seq=42,
    )

    # One warm-up read, excluded: it pays the page-cache and validator-build cost
    # that no subsequent read on a polled server ever pays again.
    store.status(run, now=NOW)

    timings = []
    for _ in range(20):
        started = time.perf_counter()
        status = store.status(run, now=NOW)
        timings.append(time.perf_counter() - started)
    assert status.snapshot is not None

    median_ms = statistics.median(timings) * 1000
    assert median_ms < 10.0, (
        f"status() median {median_ms:.2f}ms exceeds the 10ms budget"
    )
