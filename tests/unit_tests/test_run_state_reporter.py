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

"""Tests for the write side of the run control plane.

``test_run_state_contract.py`` covers the shape of ``run.json``; this file
covers the guarantees the writer makes about producing it:

* **Reporting never breaks training.** Every public method swallows its own
  exceptions. An unwritable log directory must cost observability and nothing
  else, so a reporter over a read-only path is exercised across the whole API.
* **Snapshots are atomic.** ``run.json`` is temp-file-plus-rename, so a
  concurrent reader sees the old snapshot or the new one, never a truncated one.
  Tested by hammering a reader thread against a writer.
* **A terminal state is always recorded.** ``run_lifecycle`` is the only place
  every runner passes through, so ``finished`` / ``failed`` / ``stopped`` and
  the metric-logger teardown are asserted there rather than per runner.
* **Three timestamps stay independent.** A heartbeat keeps ticking through an
  NCCL hang, so ``last_progress_at`` and ``last_metric_at`` must advance only
  when their own events happen -- that separation is what lets a reader tell a
  dead process from a hung one.

The reporter writes a real filesystem tree under ``tmp_path``; nothing here
mocks the IO, since atomicity is the property under test.
"""

import json
import os
import stat
import threading
import time

import pytest

pytest.importorskip("omegaconf", reason="omegaconf is a core rlinf dependency")

from omegaconf import OmegaConf  # noqa: E402

from rlinf.utils.run_state import (  # noqa: E402
    MediaIndexWriter,
    RunStateReporter,
    _NullReporter,
    attach_reporter,
    build_media_index,
    build_reporter,
)

RUN_ID = "20260803-120000-test-exp"


def _cfg(log_path, **runner_overrides):
    """A config with the fields the reporter reads, and nothing else."""
    runner = {
        "run_id": RUN_ID,
        "task_type": "embodied",
        "max_steps": 10,
        "val_check_interval": 5,
        "save_interval": 5,
        "logger": {
            "log_path": str(log_path),
            "experiment_name": "test-exp",
            "project_name": "test-project",
        },
    }
    runner.update(runner_overrides)
    return OmegaConf.create(
        {
            "runner": runner,
            "algorithm": {"loss_type": "ppo_actor", "adv_type": "gae"},
            "cluster": {"num_nodes": 1, "component_placement": {"actor": "0-3"}},
        }
    )


@pytest.fixture
def reporter(tmp_path):
    """A started reporter with a long heartbeat, torn down on exit.

    The interval is long so tests drive ticks explicitly instead of racing the
    background thread.
    """
    instance = RunStateReporter(_cfg(tmp_path), heartbeat_interval_s=3600)
    instance.start()
    yield instance
    instance._stop_heartbeat()


def _read(reporter, name="run.json"):
    with open(os.path.join(reporter._run_root, name), encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(reporter, name):
    path = os.path.join(reporter._run_root, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ------------------------------------------------------------------- structure


def test_run_root_is_anchored_on_run_id(tmp_path):
    """Discovery keys on this path, so two runs sharing a log_path must not
    overwrite each other."""
    instance = RunStateReporter(_cfg(tmp_path), heartbeat_interval_s=3600)
    try:
        assert instance._run_root == str(tmp_path / "_rlinf" / "runs" / RUN_ID)
        assert os.path.isdir(instance._run_root)
    finally:
        instance._stop_heartbeat()


def test_manifest_written_at_construction(tmp_path):
    instance = RunStateReporter(_cfg(tmp_path), heartbeat_interval_s=3600)
    try:
        manifest = _read(instance, "manifest.json")
    finally:
        instance._stop_heartbeat()

    assert manifest["run_id"] == RUN_ID
    assert manifest["task_type"] == "embodied"
    assert manifest["schema_version"] == 2
    assert manifest["pid"] == os.getpid()
    assert manifest["step_semantics"] == "rl_iteration"
    # The dashboard cannot import rlinf, so the alias table travels in-band.
    assert isinstance(manifest["metric_aliases"], dict)


def test_manifest_paths_include_checkpoint_and_run_root(tmp_path):
    """A reader must find checkpoints without knowing which runner ran."""
    instance = RunStateReporter(_cfg(tmp_path), heartbeat_interval_s=3600)
    try:
        paths = _read(instance, "manifest.json")["paths"]
    finally:
        instance._stop_heartbeat()

    assert paths["run_root"].endswith(os.path.join("runs", RUN_ID))
    assert paths["checkpoint_root"] == str(tmp_path / "test-exp" / "checkpoints")


def test_tensorboard_path_follows_the_per_worker_layout(tmp_path):
    """``per_worker_log`` moves the driver's own event files into ``all/``.

    ``MetricLogger._create_logger_bundle`` is passed ``log_path_suffix="all"``
    when the flag is on, so recording the parent would point a reader at a
    directory holding only subdirectories. The reader then finds no event files
    and every metric reads as absent -- a blank page with nothing saying why.
    """
    instance = RunStateReporter(
        _cfg(tmp_path, per_worker_log=True), heartbeat_interval_s=3600
    )
    try:
        paths = _read(instance, "manifest.json")["paths"]
    finally:
        instance._stop_heartbeat()

    assert paths["tensorboard"] == str(tmp_path / "tensorboard" / "all")


def test_tensorboard_path_is_the_plain_dir_without_per_worker_log(tmp_path):
    """The negative control for the test above: off by default, so no ``all/``."""
    instance = RunStateReporter(_cfg(tmp_path), heartbeat_interval_s=3600)
    try:
        paths = _read(instance, "manifest.json")["paths"]
    finally:
        instance._stop_heartbeat()

    assert paths["tensorboard"] == str(tmp_path / "tensorboard")


def test_the_per_worker_log_root_is_recorded_when_enabled(tmp_path):
    """The anchor a reader needs to break a metric out per rank.

    ``MetricLogger._get_scoped_logger`` writes each rank's bundle under
    ``<root>/<GroupName>/rank_<n>/tensorboard/``, so recording the root is enough
    for a reader to enumerate groups and ranks by globbing. ``validate_cfg`` sets
    ``per_worker_log_path`` beside the flag; it is honoured rather than rederived
    so an operator who points it elsewhere is not silently overridden.
    """
    cfg = _cfg(
        tmp_path,
        per_worker_log=True,
        per_worker_log_path=str(tmp_path / "elsewhere" / "worker_logs"),
    )
    instance = RunStateReporter(cfg, heartbeat_interval_s=3600)
    try:
        paths = _read(instance, "manifest.json")["paths"]
    finally:
        instance._stop_heartbeat()

    assert paths["worker_logs"] == str(tmp_path / "elsewhere" / "worker_logs")


def test_the_per_worker_log_root_defaults_beside_the_log_path(tmp_path):
    """A config that sets only the flag still gets a usable root.

    ``validate_cfg`` normally fills in the path, but a config assembled in code
    (a test, a notebook) may set just the flag. The default has to match the one
    ``MetricLogger`` uses, or the manifest names a directory nothing writes to.
    """
    instance = RunStateReporter(
        _cfg(tmp_path, per_worker_log=True), heartbeat_interval_s=3600
    )
    try:
        paths = _read(instance, "manifest.json")["paths"]
    finally:
        instance._stop_heartbeat()

    assert paths["worker_logs"] == str(tmp_path / "worker_logs")


def test_no_per_worker_log_root_when_the_flag_is_off(tmp_path):
    """The negative control, and the default for every run.

    Recording the path unconditionally would name a directory nothing ever
    creates, and a reader cannot then tell "enabled but silent" from "never
    enabled" -- so a run with the flag off would advertise a drill-down that has
    no data behind it.
    """
    instance = RunStateReporter(_cfg(tmp_path), heartbeat_interval_s=3600)
    try:
        paths = _read(instance, "manifest.json")["paths"]
    finally:
        instance._stop_heartbeat()

    assert paths["worker_logs"] is None


def test_checkpoint_root_follows_output_dir_when_present(tmp_path):
    """Reasoning-style runners put checkpoints under ``runner.output_dir``."""
    cfg = _cfg(tmp_path, output_dir=str(tmp_path / "out"), experiment_name="reason-exp")
    instance = RunStateReporter(cfg, heartbeat_interval_s=3600)
    try:
        paths = _read(instance, "manifest.json")["paths"]
    finally:
        instance._stop_heartbeat()

    assert paths["checkpoint_root"] == str(
        tmp_path / "out" / "reason-exp" / "checkpoints"
    )


def test_latest_symlink_points_at_this_run(tmp_path):
    """``cat <log_path>/_rlinf/latest/run.json`` is the intended human entry point."""
    instance = RunStateReporter(_cfg(tmp_path), heartbeat_interval_s=3600)
    instance.start()
    try:
        link = tmp_path / "_rlinf" / "latest"
        assert os.path.islink(link)
        # Relative target, so the tree stays valid if the log dir is moved.
        assert os.readlink(link) == os.path.join("runs", RUN_ID)
        with open(link / "run.json", encoding="utf-8") as handle:
            assert json.load(handle)["run_id"] == RUN_ID
    finally:
        instance._stop_heartbeat()


def test_start_moves_state_to_running_and_logs_the_event(reporter):
    assert _read(reporter)["state"] == "running"
    kinds = [event["kind"] for event in _read_jsonl(reporter, "events.jsonl")]
    assert kinds == ["run_start"]


# -------------------------------------------------------------- atomic writing


def test_no_temp_files_left_behind(reporter):
    reporter.set_progress(step=1, step_duration_s=1.0)
    leftovers = [n for n in os.listdir(reporter._run_root) if ".tmp." in n]
    assert leftovers == []


def test_concurrent_reader_never_sees_a_partial_snapshot(reporter):
    """The temp-file-plus-rename guarantee, exercised under contention.

    A reader that catches ``JSONDecodeError`` here would be hiding the bug this
    test exists to catch, so decode errors are collected and asserted empty.
    """
    errors = []
    stop = threading.Event()

    def read_forever():
        while not stop.is_set():
            try:
                _read(reporter)
            except json.JSONDecodeError as exc:
                errors.append(f"partial read: {exc}")
            except FileNotFoundError:
                # os.replace is atomic, so the name is never missing once
                # created; tolerated only for the initial race.
                pass

    thread = threading.Thread(target=read_forever, daemon=True)
    thread.start()
    try:
        for step in range(200):
            reporter.set_progress(step=step, step_duration_s=0.001)
    finally:
        stop.set()
        thread.join(timeout=5)

    assert not errors, errors[:3]


# --------------------------------------------------------- failure containment


@pytest.fixture
def readonly_reporter(tmp_path):
    """A reporter whose run root was made unwritable after construction.

    Construction has to succeed first (it creates the tree), so the directory is
    chmod'ed afterwards. This is the full-disk case: every subsequent write
    fails, and none of it may reach the caller.
    """
    instance = RunStateReporter(_cfg(tmp_path), heartbeat_interval_s=3600)
    instance.start()
    os.chmod(instance._run_root, stat.S_IREAD | stat.S_IEXEC)
    yield instance
    os.chmod(instance._run_root, stat.S_IRWXU)
    instance._stop_heartbeat()


def test_unwritable_run_root_does_not_raise(readonly_reporter):
    """Every public method, against a directory that rejects writes."""
    reporter = readonly_reporter

    reporter.set_progress(step=1, epoch=0, step_duration_s=1.0)
    reporter.notify_metric_written()
    reporter.record_eval_duration(2.0)
    reporter.enter_scope("generate_rollouts")
    reporter.exit_scope("generate_rollouts")
    with reporter.phase("save_ckpt"):
        pass
    reporter.component_enter("env")
    reporter.component_exit("env")
    reporter.record_checkpoint(step=1, path=str(reporter._run_root), duration_s=1.0)
    reporter.record_media({"path": "x.mp4"})
    reporter._tick()
    reporter.mark_finished()


def test_state_still_tracked_in_memory_when_writes_fail(readonly_reporter):
    """Degrade to losing the snapshot, not to losing the bookkeeping."""
    readonly_reporter.set_progress(step=7, step_duration_s=1.5)
    assert readonly_reporter._step == 7
    assert readonly_reporter.progress.step_time_p50 == 1.5


def test_reporter_disabled_when_run_root_cannot_be_created(tmp_path):
    """An unconstructible reporter degrades to a no-op, never to an exception."""
    blocker = tmp_path / "log"
    blocker.write_text("not a directory")

    instance = RunStateReporter(_cfg(blocker), heartbeat_interval_s=3600)
    assert instance._enabled is False
    # Disabled means silent, not broken.
    instance.start()
    instance.set_progress(step=1, step_duration_s=1.0)
    instance.mark_finished()


def test_build_reporter_returns_null_reporter_when_disabled(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.runner.run_state = {"enable": False}
    assert isinstance(build_reporter(cfg), _NullReporter)


def test_null_reporter_absorbs_the_whole_api():
    """Runners call the reporter unconditionally; the no-op must accept all of it."""
    null = _NullReporter()
    null.set_progress(step=1, epoch=0, step_duration_s=1.0)
    null.record_checkpoint(step=1, path="/tmp/x")
    null.component_enter("env")
    with null.phase("save_ckpt"):
        pass
    with null.run_lifecycle():
        pass
    assert null.progress.eta_s(1) is None


def test_null_reporter_progress_is_per_instance():
    """A shared ProgressEstimator would leak max_steps between runners in one
    process, which is how a second run would inherit the first one's horizon."""
    first, second = _NullReporter(), _NullReporter()
    first.progress.max_steps = 500
    assert second.progress.max_steps == 0


# ------------------------------------------------------------ three timestamps


def test_progress_and_metric_timestamps_are_independent(reporter):
    """The A1 separation: a fresh heartbeat alone proves nothing about training.

    Asserted by advancing one clock at a time and checking the others hold
    still, which is what lets a reader distinguish "process died" (heartbeat
    stale) from "training hung" (heartbeat fresh, progress stale).
    """
    assert _read(reporter)["last_progress_at"] is None
    assert _read(reporter)["last_metric_at"] is None

    reporter.set_progress(step=1, step_duration_s=1.0)
    after_progress = _read(reporter)
    assert after_progress["last_progress_at"] is not None
    assert after_progress["last_metric_at"] is None

    reporter.notify_metric_written()
    reporter._tick()
    after_metric = _read(reporter)
    assert after_metric["last_metric_at"] is not None
    assert after_metric["last_progress_at"] == after_progress["last_progress_at"]


def test_heartbeat_advances_without_any_progress(reporter):
    """Exactly the hung-training case: sequence climbs, progress does not."""
    before = _read(reporter)
    reporter._tick()
    reporter._tick()
    after = _read(reporter)

    assert after["heartbeat_seq"] > before["heartbeat_seq"]
    assert after["last_progress_at"] == before["last_progress_at"] is None


def test_heartbeat_file_is_a_fallback_liveness_signal(reporter):
    """A separate tiny file, so a reader still has an mtime if rendering
    run.json is what is failing."""
    reporter._tick()
    path = os.path.join(reporter._run_root, "heartbeat")
    assert os.path.exists(path)
    assert int(open(path, encoding="utf-8").read().strip()) >= 1


def test_heartbeat_thread_is_daemon_and_stops(tmp_path):
    """Daemon so an abrupt exit cannot be held open by the reporter."""
    instance = RunStateReporter(_cfg(tmp_path), heartbeat_interval_s=0.05)
    instance.start()
    try:
        assert instance._heartbeat_thread.daemon is True
        time.sleep(0.2)
        assert _read(instance)["heartbeat_seq"] > 1
    finally:
        instance._stop_heartbeat()
    assert not instance._heartbeat_thread.is_alive()


# -------------------------------------------------------------------- progress


def test_set_progress_records_step_and_epoch(reporter):
    reporter.set_progress(step=4, epoch=2, step_duration_s=1.25)
    progress = _read(reporter)["progress"]

    assert progress["step"] == 4
    assert progress["epoch"] == 2
    assert progress["max_steps"] == 10
    assert progress["step_semantics"] == "rl_iteration"


def test_epoch_stays_absent_when_the_runner_has_none(reporter):
    """``coding_online_rl`` has no epoch; it must read as null, not zero."""
    reporter.set_progress(step=1, step_duration_s=1.0)
    assert _read(reporter)["progress"]["epoch"] is None


def test_rendered_max_steps_follows_the_effective_horizon(reporter):
    """The shown total and the ETA's divisor must be the same number.

    ``runner.max_steps`` in config is a cap, not the plan: runners derive the
    real total from it and ``max_epochs``. A 4xH20 run with ``max_epochs: 1`` and
    ``max_steps: 3`` therefore has an effective total of 1, and rendered
    "step 1 of 3" next to an ``eta_s`` of 0 -- the estimator had the effective
    horizon while this field still had the config cap. Reading both from the
    estimator is what makes that disagreement unrepresentable.
    """
    reporter.progress.max_steps = 1  # what `attach_reporter` does
    reporter.set_progress(step=1, step_duration_s=2.0)

    doc = _read(reporter)
    assert doc["progress"]["max_steps"] == 1
    assert doc["timing"]["eta_s"] == 0.0


def test_timing_carries_eta_from_the_estimator(reporter):
    for step in range(1, 5):
        reporter.set_progress(step=step, step_duration_s=2.0)

    timing = _read(reporter)["timing"]
    assert timing["step_time_p50"] == 2.0
    assert timing["eta_s"] is not None
    assert timing["eta_confidence"] in {"low", "medium", "high"}
    assert timing["elapsed_s"] >= 0


# ----------------------------------------------------------------------- phase


def test_scope_maps_to_a_contract_phase(reporter):
    reporter.enter_scope("generate_rollouts")
    assert _read_after_tick(reporter)["phase"] == "rollout"
    reporter.exit_scope("generate_rollouts")
    assert _read_after_tick(reporter)["phase"] is None


def _read_after_tick(reporter):
    reporter._tick()
    return _read(reporter)


def test_step_scope_is_a_boundary_not_a_phase(reporter):
    """Treating ``step`` as a phase would report every run as permanently in it."""
    reporter.enter_scope("step")
    assert _read_after_tick(reporter)["phase"] is None


def test_innermost_scope_wins(reporter):
    reporter.enter_scope("step")
    reporter.enter_scope("generate_rollouts")
    assert _read_after_tick(reporter)["phase"] == "rollout"
    reporter.exit_scope("generate_rollouts")
    assert _read_after_tick(reporter)["phase"] is None


def test_unmapped_scope_passes_through_verbatim(reporter):
    """A scope added later must show up as itself rather than vanishing."""
    reporter.enter_scope("some_new_scope")
    assert _read_after_tick(reporter)["phase"] == "some_new_scope"


def test_mismatched_exit_does_not_corrupt_the_stack(reporter):
    reporter.enter_scope("generate_rollouts")
    reporter.exit_scope("never_entered")
    assert _read_after_tick(reporter)["phase"] == "rollout"


def test_phase_context_manager_exits_on_exception(reporter):
    """``save_ckpt`` uses this rather than a timer scope; a failing save must
    still clear the phase."""
    with pytest.raises(RuntimeError):
        with reporter.phase("save_ckpt"):
            assert _read_after_tick(reporter)["phase"] == "save_ckpt"
            raise RuntimeError("save blew up")

    assert _read_after_tick(reporter)["phase"] is None


def test_repeated_phase_name_is_allowed(reporter):
    """The reason save_ckpt is a phase and not a ScopedTimer scope: sft can save
    twice in one step, and a duplicate timer scope raises."""
    for _ in range(2):
        with reporter.phase("save_ckpt"):
            pass
    assert _read_after_tick(reporter)["phase"] is None


def test_every_phase_enter_has_a_matching_exit_event(reporter):
    """Both event kinds are in the frozen contract, so both must be written.

    With only ``phase_enter`` lines, a phase still running and one that ended
    just before the next began are indistinguishable in the log -- so the last
    phase of a crashed run would look like it consumed all the time up to the
    crash.
    """
    reporter.enter_scope("step")
    reporter.enter_scope("generate_rollouts")
    reporter.exit_scope("generate_rollouts")
    reporter.enter_scope("actor_training")
    reporter.exit_scope("actor_training")
    reporter.exit_scope("step")

    events = _read_jsonl(reporter, "events.jsonl")
    enters = [e for e in events if e["kind"] == "phase_enter"]
    exits = [e for e in events if e["kind"] == "phase_exit"]
    assert [e["payload"]["scope"] for e in enters] == [
        "step",
        "generate_rollouts",
        "actor_training",
    ]
    assert [e["payload"]["scope"] for e in exits] == [
        "generate_rollouts",
        "actor_training",
        "step",
    ]


def test_phase_exit_reports_the_enclosing_phase(reporter):
    """The payload's `phase` is where the run *is* after the exit, not what left.

    Leaving `generate_rollouts` inside a `step` returns to no phase; that is what
    a reader tailing the log needs in order to track current phase without
    replaying the whole stack.
    """
    reporter.enter_scope("step")
    reporter.enter_scope("generate_rollouts")
    reporter.exit_scope("generate_rollouts")

    exits = [
        e for e in _read_jsonl(reporter, "events.jsonl") if e["kind"] == "phase_exit"
    ]
    assert len(exits) == 1
    assert exits[0]["payload"] == {"scope": "generate_rollouts", "phase": None}


def test_phase_exit_is_written_for_the_context_manager_too(reporter):
    """`save_ckpt` goes through `phase()`, not the timer, and must still pair."""
    with reporter.phase("save_ckpt"):
        pass
    kinds = [e["kind"] for e in _read_jsonl(reporter, "events.jsonl")]
    assert kinds.count("phase_enter") == 1
    assert kinds.count("phase_exit") == 1


# ------------------------------------------------------------------ components


def test_components_track_concurrent_workers(reporter):
    for name in ("env", "rollout", "actor"):
        reporter.component_enter(name)

    components = _read(reporter)["components"]
    assert set(components) == {"env", "rollout", "actor"}
    assert all(entry["active"] for entry in components.values())
    assert all(entry["since"] for entry in components.values())


def test_component_exit_keeps_since(reporter):
    reporter.component_enter("env")
    since = _read(reporter)["components"]["env"]["since"]
    reporter.component_exit("env")

    entry = _read(reporter)["components"]["env"]
    assert entry["active"] is False
    assert entry["since"] == since


def test_synchronous_runner_reports_no_components(reporter):
    assert _read(reporter)["components"] == {}


def test_terminal_state_deactivates_components(reporter):
    """A finished run with an active component would misreport itself."""
    reporter.component_enter("env")
    reporter.mark_finished()

    assert _read(reporter)["components"]["env"]["active"] is False


# ----------------------------------------------------------------- checkpoints


def test_record_checkpoint_appends_after_completion(reporter, tmp_path):
    ckpt = tmp_path / "ckpt" / "global_step_5"
    ckpt.mkdir(parents=True)
    (ckpt / "model.bin").write_bytes(b"x" * 1024)

    reporter.record_checkpoint(
        step=5, path=str(ckpt), duration_s=12.5, metrics={"train/loss": 0.5}
    )

    rows = _read_jsonl(reporter, "checkpoints.jsonl")
    assert len(rows) == 1
    entry = rows[0]
    assert entry["step"] == 5
    assert entry["size_bytes"] == 1024
    assert entry["duration_s"] == 12.5
    assert entry["is_best"] is False
    assert entry["metrics"] == {"train/loss": 0.5}
    # Structured rather than a pre-baked shell command, which would go stale.
    assert entry["resume_dir"] == str(ckpt)
    assert "entry_script" in entry and "config_name" in entry


def test_record_checkpoint_updates_latest_and_emits_an_event(reporter, tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    reporter.record_checkpoint(step=5, path=str(ckpt), duration_s=1.0)

    assert _read(reporter)["latest_checkpoint"]["step"] == 5
    kinds = [event["kind"] for event in _read_jsonl(reporter, "events.jsonl")]
    assert "ckpt_saved" in kinds


def test_checkpoint_duration_feeds_the_eta(reporter, tmp_path):
    """Save cost is one of the three ETA buckets, so it must be recorded here."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    reporter.record_checkpoint(step=5, path=str(ckpt), duration_s=30.0)
    assert reporter.progress._save_times == [30.0]


def test_best_checkpoint_is_flagged(reporter, tmp_path):
    ckpt = tmp_path / "best"
    ckpt.mkdir()
    reporter.record_checkpoint(step=5, path=str(ckpt), is_best=True)
    assert _read_jsonl(reporter, "checkpoints.jsonl")[0]["is_best"] is True


def test_missing_checkpoint_path_yields_null_size(reporter, tmp_path):
    reporter.record_checkpoint(step=5, path=str(tmp_path / "gone"))
    assert _read_jsonl(reporter, "checkpoints.jsonl")[0]["size_bytes"] == 0


# ---------------------------------------------------------------------- media


def test_media_index_is_sharded_per_writer(tmp_path):
    """Videos are encoded in env worker processes, so each writer owns a shard
    and no file needs locking."""
    root = tmp_path / "run"
    first = MediaIndexWriter(str(root), shard=0)
    second = MediaIndexWriter(str(root), shard=3)

    first.append({"path": "a.mp4", "step": 1})
    second.append({"path": "b.mp4", "step": 1})

    assert (root / "media.rank0.jsonl").exists()
    assert (root / "media.rank3.jsonl").exists()
    assert json.loads((root / "media.rank3.jsonl").read_text())["path"] == "b.mp4"


def test_media_index_append_never_raises(tmp_path):
    root = tmp_path / "run"
    writer = MediaIndexWriter(str(root))
    os.chmod(root, stat.S_IREAD | stat.S_IEXEC)
    try:
        writer.append({"path": "a.mp4"})
    finally:
        os.chmod(root, stat.S_IRWXU)


def test_media_index_disabled_when_root_unusable(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    writer = MediaIndexWriter(str(blocker / "run"))

    assert writer._enabled is False
    writer.append({"path": "a.mp4"})


def test_build_media_index_targets_the_run_root(tmp_path):
    writer = build_media_index(_cfg(tmp_path), shard=2)
    assert writer is not None
    writer.append({"path": "a.mp4"})
    assert (tmp_path / "_rlinf" / "runs" / RUN_ID / "media.rank2.jsonl").exists()


def test_build_media_index_returns_none_on_bad_config():
    assert build_media_index(OmegaConf.create({})) is None


# ------------------------------------------------------------------- lifecycle


class _FakeMetricLogger:
    """Stands in for MetricLogger: records how often it was closed."""

    def __init__(self):
        self.finish_calls = 0
        self.callback = None

    def set_log_callback(self, callback):
        self.callback = callback

    def finish(self):
        self.finish_calls += 1


def test_lifecycle_marks_finished_on_clean_exit(reporter):
    with reporter.run_lifecycle():
        reporter.set_progress(step=10, step_duration_s=1.0)

    snapshot = _read(reporter)
    assert snapshot["state"] == "finished"
    assert snapshot["exit"] is None


def test_lifecycle_marks_failed_and_reraises(reporter):
    with pytest.raises(ValueError, match="boom"):
        with reporter.run_lifecycle():
            raise ValueError("boom")

    snapshot = _read(reporter)
    assert snapshot["state"] == "failed"
    assert "ValueError: boom" in snapshot["exit"]["reason"]
    assert "boom" in snapshot["exit"]["traceback_tail"]


def test_lifecycle_marks_stopped_on_interrupt(reporter):
    """Ctrl-C is not a failure; conflating the two makes the state field useless."""
    with pytest.raises(KeyboardInterrupt):
        with reporter.run_lifecycle():
            raise KeyboardInterrupt

    snapshot = _read(reporter)
    assert snapshot["state"] == "stopped"
    assert snapshot["exit"]["reason"] == "KeyboardInterrupt"
    assert snapshot["exit"]["traceback_tail"] is None


def test_lifecycle_marks_stopped_on_a_clean_system_exit(reporter):
    """`sys.exit()` and `sys.exit(0)` are a deliberate, successful shutdown."""
    for code in (None, 0):
        with pytest.raises(SystemExit):
            with reporter.run_lifecycle():
                raise SystemExit(code)

        snapshot = _read(reporter)
        assert snapshot["state"] == "stopped", f"exit code {code!r}"
        assert snapshot["exit"]["traceback_tail"] is None


@pytest.mark.parametrize("code", [1, 2, 255, "boom"])
def test_lifecycle_marks_failed_on_a_non_zero_system_exit(reporter, code):
    """A non-zero exit is a crash on its way out, not a stop.

    This assertion is deliberately the reverse of what it used to be: the test
    required `SystemExit(1)` to record `stopped`. That is what a launcher raises
    after its workers die, and it reached the dashboard as "stopped, reason:
    SystemExit" with no traceback -- a run whose env subprocess had segfaulted,
    presented as one somebody had stopped on purpose. Observed on a real LIBERO
    run; `stopped` and `failed` call for opposite reactions, so the exit code has
    to decide which one it was.

    A string code counts as a failure for the same reason Python does: the
    interpreter prints it to stderr and exits non-zero.
    """
    with pytest.raises(SystemExit):
        with reporter.run_lifecycle():
            raise SystemExit(code)

    snapshot = _read(reporter)
    assert snapshot["state"] == "failed"
    assert "SystemExit" in snapshot["exit"]["reason"]
    assert snapshot["exit"]["traceback_tail"], (
        "a failed run must carry a traceback; without one the page shows a "
        "terminal state and no way to find out why"
    )


def test_lifecycle_records_run_end_event(reporter):
    with reporter.run_lifecycle():
        pass

    events = _read_jsonl(reporter, "events.jsonl")
    assert events[-1]["kind"] == "run_end"
    assert events[-1]["payload"]["state"] == "finished"


def test_lifecycle_stops_the_heartbeat(reporter):
    with reporter.run_lifecycle():
        pass
    assert reporter._heartbeat_stop.is_set()


def test_lifecycle_closes_the_metric_logger(reporter):
    """Teardown is structural here so a runner added later cannot forget it --
    which is exactly how AsyncEmbodiedRunner came to leak its writer."""
    metric_logger = _FakeMetricLogger()
    reporter.attach_metric_logger(metric_logger)

    with reporter.run_lifecycle():
        pass

    assert metric_logger.finish_calls == 1


def test_lifecycle_closes_the_metric_logger_on_failure(reporter):
    metric_logger = _FakeMetricLogger()
    reporter.attach_metric_logger(metric_logger)

    with pytest.raises(ValueError):
        with reporter.run_lifecycle():
            raise ValueError("boom")

    assert metric_logger.finish_calls == 1


def test_metric_logger_held_weakly(reporter):
    """The runner owns the logger; this reference exists only to guarantee
    teardown and must not keep a dead object alive."""
    import gc

    metric_logger = _FakeMetricLogger()
    reporter.attach_metric_logger(metric_logger)
    del metric_logger
    gc.collect()

    assert reporter._metric_logger_ref() is None
    # Nothing to close is not an error.
    with reporter.run_lifecycle():
        pass


def test_failing_metric_logger_does_not_mask_the_run_result(reporter):
    """A broken backend must not turn a finished run into a crash."""

    class Exploding(_FakeMetricLogger):
        def finish(self):
            raise RuntimeError("backend gone")

    reporter.attach_metric_logger(Exploding())
    with reporter.run_lifecycle():
        pass

    assert _read(reporter)["state"] == "finished"


def test_state_never_reports_stalled(reporter):
    """A dead writer cannot record its own death, so liveness is read-side."""
    for drive in (
        reporter.mark_finished,
        lambda: reporter.mark_failed(ValueError("x")),
        reporter.mark_stopped,
    ):
        drive()
        assert _read(reporter)["state"] in {"finished", "failed", "stopped"}


# ------------------------------------------------------------ attach_reporter


class _FakeTimer:
    def __init__(self):
        self.observer = None

    def set_observer(self, observer):
        self.observer = observer


class _FakeRunner:
    def __init__(self, max_steps=42):
        self.max_steps = max_steps
        self.timer = _FakeTimer()
        self.metric_logger = _FakeMetricLogger()


def test_attach_reporter_wires_all_three_connections(tmp_path):
    """Centralized so 7+ runner constructors cannot drift apart."""
    runner = _FakeRunner()
    reporter = attach_reporter(runner, _cfg(tmp_path))
    try:
        assert runner.timer.observer is reporter
        assert runner.metric_logger.callback == reporter.notify_metric_written
        assert reporter.progress.max_steps == 42
    finally:
        reporter._stop_heartbeat()


def test_attach_reporter_tolerates_a_bare_runner(tmp_path):
    """Eval runners have no timer or metric logger; wiring must not require them."""

    class Bare:
        pass

    reporter = attach_reporter(Bare(), _cfg(tmp_path))
    try:
        assert reporter is not None
    finally:
        reporter._stop_heartbeat()
