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

"""A failing metric log must not be able to hold a finished run open.

``EmbodiedRunner`` hands metric logging to a background thread through a
``queue.Queue``, and ``_finish_run`` waits for that queue to drain before writing
the run's terminal state. Two ways that wait could never end:

* the worker raised inside ``log_func`` and skipped ``task_done()``, so the
  queue's unfinished count never reached zero;
* the backend hung *inside* ``log_func`` and the join had no deadline.

Either one leaves a run whose training is over stuck in ``running`` forever --
the dashboard shows it as live, and the driver never exits. The runner's own
comment already warned about the first hole from the ordering side, so this
pins the behaviour rather than the implementation.

The runner imports ray at module load and its ``__init__`` builds a cluster, so
the queue machinery is exercised on a bare object with just those attributes
bound -- the same reason ``test_runner_env_step.py`` reads its runners as source.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import pytest

from rlinf.runners.embodied_runner import EmbodiedRunner


def _runner() -> EmbodiedRunner:
    """An EmbodiedRunner with only the logging-thread machinery started.

    ``object.__new__`` skips ``__init__``, which would need a live Ray cluster.
    """
    runner = object.__new__(EmbodiedRunner)
    runner.logger = logging.getLogger("test-runner")
    runner.stop_logging = False
    runner.log_queue = queue.Queue()
    runner.log_thread = threading.Thread(target=runner._log_worker, daemon=True)
    runner.log_thread.start()
    return runner


def test_a_raising_log_entry_still_marks_itself_done():
    """The queue's unfinished count must reach zero even when the call fails."""
    runner = _runner()

    def boom():
        raise RuntimeError("backend is closed")

    runner.log_queue.put((boom, ()))
    runner.log_queue.put((lambda: None, ()))

    deadline = time.monotonic() + 5.0
    while runner.log_queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.01)

    assert runner.log_queue.unfinished_tasks == 0, (
        "a log entry that raised never called task_done(), so the drain in "
        "_finish_run would wait on it forever"
    )
    runner.stop_logging = True


def test_the_drain_gives_up_on_a_backend_that_hangs():
    """A wedged backend must not keep a finished run from finishing.

    Distinct from the raising case: nothing throws here, so `task_done` in a
    `finally` cannot help. Only a deadline on the wait can.
    """
    runner = _runner()
    release = threading.Event()

    runner.log_queue.put((lambda: release.wait(30.0), ()))
    time.sleep(0.2)  # let the worker pick it up and block inside log_func

    started = time.monotonic()
    runner._drain_log_queue(timeout_s=0.5)
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, (
        f"_drain_log_queue blocked {elapsed:.1f}s on a hung backend; a finished "
        "run would stay 'running' for as long as the backend stays stuck"
    )
    assert runner.log_queue.unfinished_tasks == 1, (
        "the item is genuinely still outstanding -- the point is that the run "
        "finishes anyway, not that the queue drained"
    )
    release.set()
    runner.stop_logging = True


def test_the_drain_waits_for_work_that_does_finish():
    """The deadline must not turn into 'drop the last flush'.

    The final metric write of a long run is real work and is exactly the one a
    reader wants; giving up on it early would trade a hang for missing data.
    """
    runner = _runner()
    runner.log_queue.put((lambda: time.sleep(0.3), ()))

    started = time.monotonic()
    runner._drain_log_queue(timeout_s=10.0)
    elapsed = time.monotonic() - started

    assert runner.log_queue.unfinished_tasks == 0
    assert elapsed >= 0.25, "returned before the queued work could have completed"
    runner.stop_logging = True


def test_the_runner_declares_a_bounded_drain_timeout():
    """A deadline that is not set is a deadline that does not exist."""
    assert isinstance(EmbodiedRunner.LOG_DRAIN_TIMEOUT_S, (int, float))
    assert 0 < EmbodiedRunner.LOG_DRAIN_TIMEOUT_S < 3600


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
