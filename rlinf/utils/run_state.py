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

"""Run control-plane reporter: what a run is doing, written to disk.

RLinf's loggers (TensorBoard, wandb, SwanLab) are the *data plane* -- scalar
time series. This module is the *control plane*: the one-per-run facts needed
to answer "is my job alive, how far along is it, and where is the last
checkpoint?"

Design constraints, all load-bearing:

* **Files, not a database, and no network.** A reader must work across
  virtualenvs (the training envs are multi-venv and very heavy) and after a
  crash. The driver must not depend on a reader being alive, so nothing is
  pushed anywhere. SQLite is avoided because its locking is unreliable on NFS.
* **Plain stdlib only.** No new dependency for any of this.
* **Reporting must never break training.** Every public method swallows and
  logs its own exceptions. A full disk degrades observability; it must not
  take down a job that is otherwise fine.
* **Atomic replacement.** ``run.json`` is written to a temp file in the same
  directory and then ``os.replace``'d, so a concurrent reader sees either the
  old snapshot or the new one, never a truncated one.

The contract is ``docs/schemas/run.v2.schema.json``, shared verbatim with the
reader side. See ``docs/source-en/rst_source/guides/run_state.rst``.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import threading
import traceback
import weakref
from datetime import datetime, timezone
from typing import Any

from rlinf.utils.logging import get_logger
from rlinf.utils.metric_naming import alias_table
from rlinf.utils.progress import ProgressEstimator, step_semantics_for

SCHEMA_VERSION = 2

#: Default heartbeat period. Well under any plausible reader timeout so a
#: healthy run is never mistaken for a dead one.
DEFAULT_HEARTBEAT_INTERVAL_S = 5.0

#: How much of a failure traceback to persist. Enough to identify the fault
#: without turning the snapshot into a log file.
TRACEBACK_TAIL_CHARS = 4000


def _is_successful_system_exit(exc: SystemExit) -> bool:
    """Whether ``SystemExit`` carries Python's successful exit status."""
    code = exc.code
    return code is None or (isinstance(code, int) and code == 0)


#: Timer scope -> contract phase. Scope names differ across runners for the
#: same activity, so they are normalized here rather than leaked to readers.
#: Unmapped scopes pass through verbatim, so a scope added later shows up as
#: itself instead of vanishing.
SCOPE_TO_PHASE = {
    "generate_rollouts": "rollout",
    "construct_rollout_batch": "rollout",
    "sync_weights": "sync_weights",
    "update_rollout_weights": "sync_weights",
    "actor_training": "train",
    "evaluate": "eval",
    "eval": "eval",
    "prepare_data": "prepare_data",
    "cal_adv_and_returns": "cal_adv_and_returns",
}

#: ``step`` wraps an entire iteration, so it is a step boundary rather than a
#: phase. Treating it as one would report every run as permanently in "step".
STEP_SCOPE = "step"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    """Format as ISO-8601 with a ``Z`` suffix, matching the schema fixtures."""
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str | None:
    """Best-effort commit of the checked-out tree; None outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _dir_size_bytes(path: str) -> int | None:
    """Recursive size of a checkpoint directory, or None if unreadable."""
    try:
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                with contextlib.suppress(OSError):
                    total += os.path.getsize(os.path.join(root, name))
        return total
    except OSError:
        return None


class RunStateReporter:
    """Write the control plane for one run.

    Instantiate once per driver, in the runner. All state lives in memory under
    a lock and is rendered to ``run.json`` on every change and on every
    heartbeat tick, so the file is always a complete current snapshot.

    Args:
        cfg: The full resolved config. Reads ``runner.logger.log_path``,
            ``runner.run_id``, ``runner.task_type``, ``algorithm``, ``cluster``.
        heartbeat_interval_s: Heartbeat period in seconds.
    """

    def __init__(
        self,
        cfg,
        heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    ):
        self._logger = get_logger()
        self._lock = threading.Lock()
        self._enabled = True
        self._cfg = cfg
        self._heartbeat_interval_s = heartbeat_interval_s

        self._state = "pending"
        self._phase: str | None = None
        self._phase_since: datetime | None = None
        self._scope_stack: list[str] = []
        self._components: dict[str, dict[str, Any]] = {}

        self._started_at = _utc_now()
        self._heartbeat_at = self._started_at
        self._heartbeat_seq = 0
        self._last_progress_at: datetime | None = None
        self._last_metric_at: datetime | None = None

        self._step = 0
        self._epoch: int | None = None
        self._latest_checkpoint: dict | None = None
        self._exit: dict | None = None

        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()
        self._metric_logger_ref: weakref.ReferenceType | None = None

        try:
            runner_cfg = cfg.runner
            self._run_id = str(runner_cfg.get("run_id", None) or "unknown-run")
            self._task_type = str(runner_cfg.get("task_type", None) or "embodied")
            # Only a provisional horizon. `runner.max_steps` in config is a cap,
            # not the plan: runners derive the real total from it and
            # `max_epochs` (see `EmbodiedRunner.set_max_steps`), so the config
            # value can be larger than the run will ever reach.
            # `attach_reporter` overwrites this with the runner's effective value
            # once that exists.
            self._max_steps = runner_cfg.get("max_steps", None)
            log_path = runner_cfg.logger.log_path
            self._run_root = os.path.join(log_path, "_rlinf", "runs", self._run_id)
            self._control_root = os.path.join(log_path, "_rlinf")
            self._log_path = log_path
            self.progress = ProgressEstimator(
                max_steps=int(self._max_steps) if self._max_steps else 0,
                val_check_interval=int(runner_cfg.get("val_check_interval", 0) or 0),
                save_interval=int(runner_cfg.get("save_interval", 0) or 0),
                step_semantics=step_semantics_for(self._task_type),
            )
            os.makedirs(self._run_root, exist_ok=True)
        except Exception as exc:  # noqa: BLE001 - never break training
            self._enabled = False
            self._logger.warning(f"Run-state reporting disabled: {exc}")
            return

        self._write_manifest()
        self._update_latest_symlink()

    # ---------------------------------------------------------------- helpers

    def _run_file(self, name: str) -> str:
        return os.path.join(self._run_root, name)

    def _append_jsonl(self, name: str, record: dict) -> None:
        """Append one line and flush. Single writer per file, so no locking."""
        if not self._enabled:
            return
        try:
            with open(self._run_file(name), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
                handle.flush()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(f"Failed to append to {name}: {exc}")

    def _write_manifest(self) -> None:
        """Write the run invariants once. Nothing here changes mid-run."""
        try:
            cfg = self._cfg
            cluster_cfg = cfg.get("cluster", None)
            algorithm_cfg = cfg.get("algorithm", None)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self._run_id,
                "task_type": self._task_type,
                "experiment_name": cfg.runner.logger.get("experiment_name", None),
                "project_name": cfg.runner.logger.get("project_name", None),
                "step_semantics": self.progress.step_semantics,
                "algorithm": {
                    "loss_type": algorithm_cfg.get("loss_type", None)
                    if algorithm_cfg
                    else None,
                    "adv_type": algorithm_cfg.get("adv_type", None)
                    if algorithm_cfg
                    else None,
                },
                "cluster": {
                    "num_nodes": cluster_cfg.get("num_nodes", None)
                    if cluster_cfg
                    else None,
                    "component_placement": _plain(
                        cluster_cfg.get("component_placement", None)
                        if cluster_cfg
                        else None
                    ),
                },
                "git_sha": _git_sha(),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "started_at": _iso(self._started_at),
                "heartbeat_interval_s": self._heartbeat_interval_s,
                "resumed_from": cfg.runner.get("resume_dir", None),
                "paths": self._paths(),
                # Embedded so a reader can resolve legacy metric keys without
                # importing rlinf, which it deliberately cannot do.
                "metric_aliases": alias_table(),
            }
            self._atomic_write("manifest.json", manifest)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(f"Failed to write run manifest: {exc}")

    def _paths(self) -> dict:
        return {
            "log_path": self._log_path,
            "tensorboard": self._tensorboard_dir(),
            "worker_logs": self._worker_log_root(),
            "video_root": os.path.join(self._log_path, "video"),
            "checkpoint_root": self._checkpoint_root(),
            "run_root": self._run_root,
        }

    def _tensorboard_dir(self) -> str:
        """Where the driver's own event files land.

        With ``runner.per_worker_log: true`` the aggregate bundle moves one level
        down, into ``tensorboard/all/``, to leave room for the per-rank bundles
        (``MetricLogger`` passes ``log_path_suffix="all"``). Recording the parent
        for such a run points a reader at a directory that exists and holds no
        event files, which is indistinguishable from a run that has not logged
        yet -- every metric reads as absent with nothing to explain why.
        """
        tensorboard = os.path.join(self._log_path, "tensorboard")
        try:
            if bool(self._cfg.runner.get("per_worker_log", False)):
                return os.path.join(tensorboard, "all")
        except Exception as exc:  # noqa: BLE001 - never break training
            self._logger.debug(f"Could not resolve the TensorBoard dir: {exc}")
        return tensorboard

    def _worker_log_root(self) -> str | None:
        """Where the per-rank metric bundles land, or None when there are none.

        With ``runner.per_worker_log: true`` every ``(worker group, rank)`` that
        logs gets its own backend bundle under
        ``<root>/<GroupName>/rank_<n>/tensorboard/`` (``MetricLogger.
        _get_scoped_logger``). Recording the root is what lets a reader break a
        metric out per rank -- the group and rank are recoverable from the path,
        so discovery is a glob and needs no second index.

        ``None`` when the flag is off, which is the default. An unconditional
        value would name a directory nothing ever creates, and a reader cannot
        tell "configured but empty" from "never enabled" once the path is there.
        """
        try:
            if not bool(self._cfg.runner.get("per_worker_log", False)):
                return None
            configured = self._cfg.runner.get("per_worker_log_path", None)
            # `validate_cfg` sets this alongside the flag, but a config assembled
            # in code (tests, a notebook) may only set the flag; fall back to the
            # same default `MetricLogger` uses so the two never disagree.
            return str(configured or os.path.join(self._log_path, "worker_logs"))
        except Exception as exc:  # noqa: BLE001 - never break training
            self._logger.debug(f"Could not resolve the per-worker log root: {exc}")
            return None

    def _checkpoint_root(self) -> str | None:
        """Where this runner writes checkpoints.

        Runners disagree: embodied/offline/sft derive it from
        ``runner.logger.log_path`` + ``logger.experiment_name``, while
        reasoning/coding use ``runner.output_dir`` + ``runner.experiment_name``.
        Resolving it once here is what lets a reader find checkpoints without
        knowing which runner produced them -- discovery is anchored on
        ``_rlinf/runs/<run_id>/`` and never guesses paths.
        """
        try:
            runner_cfg = self._cfg.runner
            output_dir = runner_cfg.get("output_dir", None)
            if output_dir:
                experiment_name = runner_cfg.get(
                    "experiment_name", None
                ) or runner_cfg.logger.get("experiment_name", "default")
                return os.path.join(output_dir, str(experiment_name), "checkpoints")
            experiment_name = runner_cfg.logger.get("experiment_name", "default")
            return os.path.join(self._log_path, str(experiment_name), "checkpoints")
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(f"Could not resolve checkpoint root: {exc}")
            return None

    def _atomic_write(self, name: str, payload: dict) -> None:
        """Write JSON via temp file + ``os.replace``.

        The temp file is in the same directory so the rename stays within one
        filesystem, which is what makes it atomic.
        """
        target = self._run_file(name)
        tmp = f"{target}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)

    def _update_latest_symlink(self) -> None:
        """Point ``_rlinf/latest`` at this run.

        Convenience for humans (``cat .../latest/run.json``). Best effort: some
        filesystems disallow symlinks, which must not be fatal.
        """
        try:
            link = os.path.join(self._control_root, "latest")
            target = os.path.join("runs", self._run_id)
            tmp_link = f"{link}.tmp.{os.getpid()}"
            with contextlib.suppress(OSError):
                os.remove(tmp_link)
            os.symlink(target, tmp_link)
            os.replace(tmp_link, link)
        except OSError as exc:
            self._logger.debug(f"Could not update 'latest' symlink: {exc}")

    # ------------------------------------------------------------- rendering

    def _render_locked(self) -> dict:
        """Build the snapshot. Caller must hold ``self._lock``."""
        now = _utc_now()
        timing = {
            "started_at": _iso(self._started_at),
            "elapsed_s": max(0.0, (now - self._started_at).total_seconds()),
        }
        timing.update(self.progress.snapshot(self._step))

        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self._run_id,
            "task_type": self._task_type,
            "algorithm": self._algorithm_snapshot(),
            "state": self._state,
            "phase": self._phase,
            "phase_since": _iso(self._phase_since),
            "components": dict(self._components),
            "heartbeat_at": _iso(self._heartbeat_at),
            "heartbeat_seq": self._heartbeat_seq,
            "last_progress_at": _iso(self._last_progress_at),
            "last_metric_at": _iso(self._last_metric_at),
            "progress": {
                "step": self._step,
                # Use the estimator's effective horizon so progress and ETA
                # always describe the same plan.
                "max_steps": self.progress.max_steps or None,
                "epoch": self._epoch,
                "step_semantics": self.progress.step_semantics,
            },
            "timing": timing,
            "latest_checkpoint": self._latest_checkpoint,
            "paths": self._paths(),
            "cluster": self._cluster_snapshot(),
            "exit": self._exit,
        }

    def _algorithm_snapshot(self) -> dict | None:
        algorithm_cfg = self._cfg.get("algorithm", None)
        if algorithm_cfg is None:
            return None
        return {
            "loss_type": algorithm_cfg.get("loss_type", None),
            "adv_type": algorithm_cfg.get("adv_type", None),
        }

    def _cluster_snapshot(self) -> dict:
        cluster_cfg = self._cfg.get("cluster", None)
        if cluster_cfg is None:
            return {}
        return {
            "num_nodes": cluster_cfg.get("num_nodes", None),
            "component_placement": _plain(cluster_cfg.get("component_placement", None)),
        }

    def _flush(self) -> None:
        """Re-render ``run.json``. Safe to call from any thread."""
        if not self._enabled:
            return
        try:
            with self._lock:
                payload = self._render_locked()
            self._atomic_write("run.json", payload)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(f"Failed to write run.json: {exc}")

    # ------------------------------------------------------------- heartbeat

    def _heartbeat_worker(self) -> None:
        """Tick until stopped.

        ``Event.wait`` rather than ``sleep`` so shutdown is immediate instead of
        waiting out a full interval.
        """
        while not self._heartbeat_stop.wait(self._heartbeat_interval_s):
            self._tick()

    def _tick(self) -> None:
        with self._lock:
            self._heartbeat_at = _utc_now()
            self._heartbeat_seq += 1
            seq = self._heartbeat_seq
        # A tiny separate file so a reader still has an mtime to fall back on
        # if rendering run.json is what is failing.
        try:
            with open(self._run_file("heartbeat"), "w", encoding="utf-8") as handle:
                handle.write(f"{seq}\n")
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(f"Heartbeat file write failed: {exc}")
        self._flush()

    def start(self) -> None:
        """Move to ``running`` and start the heartbeat thread."""
        if not self._enabled:
            return
        with self._lock:
            self._state = "running"
        self._append_jsonl(
            "events.jsonl",
            {
                "ts": _iso(_utc_now()),
                "kind": "run_start",
                "step": self._step,
                "payload": {"run_id": self._run_id, "task_type": self._task_type},
            },
        )
        self._tick()
        # Daemon so an abrupt driver exit cannot be held open by this thread.
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_worker, daemon=True, name="rlinf-run-heartbeat"
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    # ---------------------------------------------------------- progress API

    def set_progress(
        self,
        step: int,
        epoch: int | None = None,
        step_duration_s: float | None = None,
    ) -> None:
        """Record that the run advanced to ``step``.

        Updates ``last_progress_at``, which is what distinguishes a hung
        training loop from a dead process: the heartbeat thread keeps ticking
        through an NCCL hang, so a fresh heartbeat alone proves nothing about
        training.

        Args:
            step: Steps completed.
            epoch: Only if the runner already tracks it; never synthesized.
            step_duration_s: Net step time, excluding eval and save.
        """
        if not self._enabled:
            return
        try:
            with self._lock:
                self._step = int(step)
                if epoch is not None:
                    self._epoch = int(epoch)
                self._last_progress_at = _utc_now()
                if step_duration_s is not None:
                    self.progress.record_step(step_duration_s)
            self._flush()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(f"Failed to record progress: {exc}")

    def notify_metric_written(self) -> None:
        """Mark that metrics reached a backend (updates ``last_metric_at``).

        Called on the metric path, so it only touches memory -- no IO. The next
        heartbeat publishes it.
        """
        if not self._enabled:
            return
        try:
            with self._lock:
                self._last_metric_at = _utc_now()
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(f"Failed to record metric timestamp: {exc}")

    def record_eval_duration(self, duration_s: float) -> None:
        if self._enabled:
            self.progress.record_eval(duration_s)

    # ------------------------------------------------------------- phase API

    def enter_scope(self, scope: str) -> None:
        """Push a timer scope; the innermost non-``step`` one becomes the phase."""
        if not self._enabled:
            return
        try:
            with self._lock:
                self._scope_stack.append(scope)
                self._recompute_phase_locked()
            self._append_jsonl(
                "events.jsonl",
                {
                    "ts": _iso(_utc_now()),
                    "kind": "phase_enter",
                    "step": self._step,
                    "payload": {"scope": scope, "phase": self._phase},
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(f"Failed to enter scope {scope}: {exc}")

    def exit_scope(self, scope: str) -> None:
        """Pop a timer scope and record that the phase ended.

        The paired ``phase_exit`` is what makes a phase's duration derivable from
        the event log alone: with only ``phase_enter`` lines, a phase that is
        still running and one that ended just before the next began look the
        same, so the last phase of a crashed run would appear to have taken all
        the time up to the crash.
        """
        if not self._enabled:
            return
        try:
            with self._lock:
                # Remove the innermost matching entry: scopes nest, but a
                # mismatched exit must not corrupt the stack.
                for index in range(len(self._scope_stack) - 1, -1, -1):
                    if self._scope_stack[index] == scope:
                        del self._scope_stack[index]
                        break
                self._recompute_phase_locked()
                # Read under the lock; `phase` is now the enclosing scope, if any.
                phase_after = self._phase
                step = self._step
            self._append_jsonl(
                "events.jsonl",
                {
                    "ts": _iso(_utc_now()),
                    "kind": "phase_exit",
                    "step": step,
                    "payload": {"scope": scope, "phase": phase_after},
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(f"Failed to exit scope {scope}: {exc}")

    def _recompute_phase_locked(self) -> None:
        phase = None
        for scope in reversed(self._scope_stack):
            if scope == STEP_SCOPE:
                continue
            phase = SCOPE_TO_PHASE.get(scope, scope)
            break
        if phase != self._phase:
            self._phase = phase
            self._phase_since = _utc_now()

    @contextlib.contextmanager
    def phase(self, name: str):
        """Report a phase that has no timer scope.

        Checkpoint saving uses this instead of ``ScopedTimer``: recording the
        same scope twice before ``consume_durations()`` raises, and some runners
        legitimately save twice within one step (periodic plus best-so-far).
        """
        self.enter_scope(name)
        try:
            yield
        finally:
            self.exit_scope(name)

    # --------------------------------------------------------- component API

    def component_enter(self, name: str) -> None:
        """Mark an async component active.

        Async runners start env, rollout, and actor before the loop and keep
        all three live throughout, so a single scalar ``phase`` cannot describe
        them. Synchronous runners never call this and report ``{}``.
        """
        if not self._enabled:
            return
        try:
            with self._lock:
                self._components[name] = {
                    "active": True,
                    "since": _iso(_utc_now()),
                }
            self._flush()
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(f"Failed to mark component {name}: {exc}")

    def component_exit(self, name: str) -> None:
        if not self._enabled:
            return
        try:
            with self._lock:
                existing = self._components.get(name, {})
                self._components[name] = {
                    "active": False,
                    "since": existing.get("since"),
                }
            self._flush()
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(f"Failed to clear component {name}: {exc}")

    # -------------------------------------------------------- checkpoint API

    def record_checkpoint(
        self,
        step: int,
        path: str,
        duration_s: float | None = None,
        is_best: bool = False,
        metrics: dict | None = None,
    ) -> None:
        """Append a completed checkpoint to the index.

        Call this *after* the save finishes. That ordering is the whole
        mechanism behind checkpoint visibility: a reader trusting this file
        cannot observe a half-written checkpoint, so no ``WRITING``/``READY``
        protocol is needed.

        ``resume_dir`` / ``entry_script`` / ``config_name`` are stored as
        separate fields rather than a pre-baked shell command, which would go
        stale as soon as anything about the launch changed.
        """
        if not self._enabled:
            return
        try:
            if duration_s is not None:
                self.progress.record_save(duration_s)
            entry = {
                "step": int(step),
                "path": path,
                "saved_at": _iso(_utc_now()),
                "size_bytes": _dir_size_bytes(path),
                "duration_s": duration_s,
                "is_best": bool(is_best),
                "metrics": metrics or {},
                "resume_dir": path,
                "entry_script": _entry_script(),
                "config_name": self._cfg.runner.logger.get("experiment_name", None),
            }
            self._append_jsonl("checkpoints.jsonl", entry)
            self._append_jsonl(
                "events.jsonl",
                {
                    "ts": entry["saved_at"],
                    "kind": "ckpt_saved",
                    "step": int(step),
                    "payload": {"path": path, "is_best": bool(is_best)},
                },
            )
            with self._lock:
                self._latest_checkpoint = entry
            self._flush()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(f"Failed to record checkpoint: {exc}")

    def record_media(self, record: dict, shard: int = 0) -> None:
        """Append to the per-writer media index.

        Sharded by writer because video files are produced inside env worker
        processes; one file per shard keeps every writer a single writer.
        """
        self._append_jsonl(f"media.rank{shard}.jsonl", record)

    # ------------------------------------------------------------ terminal API

    def mark_finished(self) -> None:
        self._set_terminal("finished", None)

    def mark_failed(self, exc: BaseException) -> None:
        reason = f"{type(exc).__name__}: {exc}"
        tail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[
            -TRACEBACK_TAIL_CHARS:
        ]
        self._set_terminal("failed", {"reason": reason, "traceback_tail": tail})

    def mark_stopped(self, reason: str = "interrupted") -> None:
        self._set_terminal("stopped", {"reason": reason, "traceback_tail": None})

    def _set_terminal(self, state: str, exit_info: dict | None) -> None:
        if not self._enabled:
            return
        try:
            self._stop_heartbeat()
            with self._lock:
                self._state = state
                self._exit = exit_info
                self._phase = None
                self._phase_since = None
                # Async components cannot still be running once the driver is
                # done; leaving them active would misreport a finished run.
                for name, info in list(self._components.items()):
                    self._components[name] = {
                        "active": False,
                        "since": info.get("since"),
                    }
            self._append_jsonl(
                "events.jsonl",
                {
                    "ts": _iso(_utc_now()),
                    "kind": "run_end",
                    "step": self._step,
                    "payload": {"state": state, "exit": exit_info},
                },
            )
            self._flush()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(f"Failed to record terminal state {state}: {exc}")

    def attach_metric_logger(self, metric_logger) -> None:
        """Remember the metric logger so the lifecycle can close it.

        Held weakly: the runner owns the logger, and this reference exists only
        to guarantee teardown, not to keep it alive.
        """
        self._metric_logger_ref = weakref.ref(metric_logger)

    def _finish_metrics(self) -> None:
        """Close the metric backends. Idempotent via ``MetricLogger.finish``.

        Lifecycle cleanup makes logger teardown consistent across runners.
        """
        metric_logger = self._metric_logger_ref() if self._metric_logger_ref else None
        if metric_logger is None:
            return
        try:
            metric_logger.finish()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(f"Failed to close metric logger: {exc}")

    @contextlib.contextmanager
    def run_lifecycle(self):
        """Wrap a run so it always records a terminal state.

        Placed around the whole loop, this records ``failed`` on exception and
        ``stopped`` on interrupt.

        ``kill -9`` is deliberately not covered: it cannot be, which is exactly
        why liveness is derived read-side from the timestamps.
        """
        self.start()
        try:
            try:
                yield self
            except KeyboardInterrupt as exc:
                self.mark_stopped(reason=f"{type(exc).__name__}")
                raise
            except SystemExit as exc:
                if _is_successful_system_exit(exc):
                    self.mark_stopped(reason=f"{type(exc).__name__}")
                else:
                    self.mark_failed(exc)
                raise
            except BaseException as exc:
                self.mark_failed(exc)
                raise
            else:
                self.mark_finished()
        finally:
            self._finish_metrics()


class MediaIndexWriter:
    """Append-only media index for one writer process.

    Videos are encoded inside env worker processes, so the index has many
    writers. Each gets its own ``media.rank<k>.jsonl`` shard, which keeps every
    file single-writer and needs no locking.

    This is deliberately *not* a :class:`RunStateReporter`. A reporter in an env
    worker would start a second heartbeat thread and overwrite the driver's
    ``run.json``; this class only ever appends to its own shard.

    Args:
        run_root: ``<log_path>/_rlinf/runs/<run_id>``.
        shard: Writer index, normally the worker rank.
    """

    def __init__(self, run_root: str, shard: int = 0):
        self._path = os.path.join(run_root, f"media.rank{int(shard)}.jsonl")
        self._enabled = True
        try:
            os.makedirs(run_root, exist_ok=True)
        except OSError as exc:
            self._enabled = False
            get_logger().warning(f"Media index disabled: {exc}")

    def append(self, record: dict) -> None:
        """Append one media record. Never raises."""
        if not self._enabled:
            return
        try:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
                handle.flush()
        except Exception as exc:  # noqa: BLE001 - observability is not critical
            get_logger().debug(f"Failed to append media record: {exc}")


def build_media_index(cfg, shard: int = 0) -> MediaIndexWriter | None:
    """Create a media index writer for a worker process, or None if impossible."""
    try:
        runner_cfg = cfg.runner
        run_id = str(runner_cfg.get("run_id", None) or "unknown-run")
        run_root = os.path.join(runner_cfg.logger.log_path, "_rlinf", "runs", run_id)
        return MediaIndexWriter(run_root, shard=shard)
    except Exception as exc:  # noqa: BLE001
        get_logger().debug(f"Media index unavailable: {exc}")
        return None


def _plain(value: Any) -> Any:
    """Convert OmegaConf containers to plain Python for json.dump."""
    if value is None:
        return None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _entry_script() -> str | None:
    """The script that launched this run, for reconstructing a resume command."""
    import sys

    return sys.argv[0] if sys.argv else None


class _NullReporter:
    """No-op stand-in so runners can call the API unconditionally.

    Returned when reporting is disabled, which keeps the call sites free of
    ``if self.reporter is not None`` noise.
    """

    def __init__(self):
        # Per-instance, not a class attribute: callers set
        # ``reporter.progress.max_steps``, which on a shared object would leak
        # between runners in the same process.
        self.progress = ProgressEstimator(max_steps=0)

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop

    @contextlib.contextmanager
    def phase(self, name: str):
        yield

    @contextlib.contextmanager
    def run_lifecycle(self):
        yield self


def attach_reporter(runner, cfg) -> RunStateReporter | _NullReporter:
    """Build a reporter and connect it to a runner's timer and metric logger.

    Every runner needs the same three connections, so they live here rather
    than being restated (and drifting) in each constructor:

    * the timer observer, so phase follows the scopes the loop already declares;
    * the metric-log callback, so ``last_metric_at`` advances;
    * ``max_steps``, so the ETA has a horizon.

    Call after ``self.timer``, ``self.metric_logger`` and ``self.max_steps``
    exist.

    Args:
        runner: The runner instance being wired.
        cfg: Full resolved config.

    Returns:
        The reporter, which the caller should store as ``self.reporter``.
    """
    reporter = build_reporter(cfg)
    max_steps = getattr(runner, "max_steps", None)
    if max_steps:
        reporter.progress.max_steps = int(max_steps)
    timer = getattr(runner, "timer", None)
    if timer is not None and hasattr(timer, "set_observer"):
        timer.set_observer(reporter)
    metric_logger = getattr(runner, "metric_logger", None)
    if metric_logger is not None and hasattr(metric_logger, "set_log_callback"):
        metric_logger.set_log_callback(reporter.notify_metric_written)
        reporter.attach_metric_logger(metric_logger)
    return reporter


def build_reporter(cfg) -> RunStateReporter | _NullReporter:
    """Create a reporter, or a no-op if disabled or unconstructible.

    Controlled by ``cfg.runner.run_state.enable`` (default on). Falling back to
    a no-op rather than raising keeps a config or filesystem problem from
    blocking a training job.
    """
    try:
        runner_cfg = cfg.runner
        run_state_cfg = runner_cfg.get("run_state", None)
        if run_state_cfg is not None and not run_state_cfg.get("enable", True):
            return _NullReporter()
        interval = DEFAULT_HEARTBEAT_INTERVAL_S
        if run_state_cfg is not None:
            interval = float(
                run_state_cfg.get("heartbeat_interval_s", interval) or interval
            )
        reporter = RunStateReporter(cfg, heartbeat_interval_s=interval)
        return reporter if reporter._enabled else _NullReporter()
    except Exception as exc:  # noqa: BLE001
        get_logger().warning(f"Run-state reporting unavailable: {exc}")
        return _NullReporter()
