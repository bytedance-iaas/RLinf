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

"""Tests for runner logging-thread teardown.

Metric tables are rendered on a background thread fed by ``log_queue``.
``_log_worker`` re-checks ``stop_logging`` between items, so teardown must drain
the queue *before* raising the flag -- otherwise the loop exits with entries
still pending, those entries never get their ``task_done()``, and the
``log_queue.join()`` inside teardown blocks forever.
"""

import queue
import threading
import time

import pytest

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.runners.offline_runner import OfflineRunner


class _FakeBackend:
    def __init__(self):
        self.finished = False

    def finish(self):
        self.finished = True


class _FakeMetricLogger:
    def __init__(self):
        self.backend = _FakeBackend()
        self._all_loggers = [{"tensorboard": self.backend}]

    def finish(self):
        for logger in self._all_loggers:
            for instance in logger.values():
                instance.finish()


def _make_stub(runner_cls):
    """Build a stub carrying the real ``_finish_run`` and ``_log_worker``."""

    class _Stub:
        _finish_run = runner_cls._finish_run
        _log_worker = runner_cls._log_worker

        def __init__(self):
            self.metric_logger = _FakeMetricLogger()
            self.stop_logging = False
            self.log_queue = queue.Queue()
            self.drained = []
            self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
            self.log_thread.start()

    return _Stub()


def _slow_log(stub, index):
    """Stand in for ``print_metrics_table``, which is not instant."""
    time.sleep(0.05)
    stub.drained.append(index)


def _finish_with_timeout(stub, timeout=10.0):
    """Call the real ``_finish_run`` off-thread so a hang fails instead of stalling."""
    done = threading.Event()
    error = []

    def call():
        try:
            stub._finish_run()
        except BaseException as exc:  # pragma: no cover - surfaced via assert
            error.append(exc)
        finally:
            done.set()

    threading.Thread(target=call, daemon=True).start()
    returned = done.wait(timeout=timeout)
    assert not error, f"_finish_run raised {error[0]!r}"
    return returned


@pytest.mark.parametrize("runner_cls", [EmbodiedRunner, OfflineRunner])
@pytest.mark.parametrize("pending", [0, 1, 5])
def test_finish_run_drains_pending_logs(runner_cls, pending):
    """Teardown must not hang, regardless of how many tables are still queued."""
    stub = _make_stub(runner_cls)
    for index in range(pending):
        stub.log_queue.put((_slow_log, (stub, index)))

    assert _finish_with_timeout(stub), (
        f"{runner_cls.__name__}._finish_run() hung with {pending} queued log(s); "
        "the queue must be drained before stop_logging is set"
    )
    assert stub.drained == list(range(pending)), "queued logs were dropped"
    assert stub.metric_logger.backend.finished, "metric logger was not closed"
    assert stub.stop_logging, "logging thread was not asked to stop"
    assert not stub.log_thread.is_alive(), "logging thread did not exit"
