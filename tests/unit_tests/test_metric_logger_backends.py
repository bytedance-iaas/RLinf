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

"""Tests for :class:`~rlinf.utils.metric_logger.MetricLogger` backend dispatch.

Backends are exercised through fakes rather than real wandb/swanlab/TensorBoard
writers: what is under test is the dispatch contract, not any vendor's client.

The properties that matter:

* **``finish`` is idempotent.** Runners call it explicitly, ``__del__`` calls it
  again at interpreter shutdown, and ``run_lifecycle``'s ``finally`` now calls it
  a third time. Every one of those paths is real, so double-closing must be a
  no-op rather than an error.
* **An unsupported capability warns and skips.** ``log_table`` used to raise,
  which turned "my backend cannot draw tables" into a crashed training run.
* **Keys are normalized once, at this single entry point.** That is what makes
  the dual write cover all runners.
* **The control-plane callback cannot break the data plane.** A failing
  ``last_metric_at`` notification must not lose a metric.
"""

import pytest

pytest.importorskip("omegaconf", reason="omegaconf is a core rlinf dependency")

from omegaconf import OmegaConf  # noqa: E402

from rlinf.utils import metric_naming  # noqa: E402
from rlinf.utils.metric_logger import (  # noqa: E402
    DEFAULT_FLUSH_INTERVAL_S,
    MetricBackend,
    MetricLogger,
    _BaseBackend,
)


class FakeBackend(_BaseBackend):
    """Records every call so dispatch can be asserted."""

    name = "fake"

    def __init__(self):
        self.logged = []
        self.tables = []
        self.media = []
        self.flushes = 0
        self.finishes = 0
        self._finished = False

    def log(self, data, step):
        self.logged.append((dict(data), step))

    def log_table(self, df_data, name, step):
        self.tables.append((df_data, name, step))

    def log_media(self, path, name, step, **kwargs):
        self.media.append((path, name, step, kwargs))

    def flush(self):
        self.flushes += 1

    def finish(self):
        if self._finished:
            return
        self._finished = True
        self.finishes += 1


class Tableless(_BaseBackend):
    """A backend with only ``log``, exercising the warn-and-skip defaults."""

    name = "tableless"

    def __init__(self):
        self.logged = []

    def log(self, data, step):
        self.logged.append((dict(data), step))


@pytest.fixture(autouse=True)
def reset_warned_prefixes():
    metric_naming._warned_prefixes.clear()
    yield
    metric_naming._warned_prefixes.clear()


def _cfg(tmp_path, **logger_overrides):
    logger_cfg = {
        "log_path": str(tmp_path),
        "experiment_name": "test-exp",
        "project_name": "test-project",
        # No backends: each test installs its own fakes, so nothing here needs
        # wandb, swanlab, or a real SummaryWriter.
        "logger_backends": [],
    }
    logger_cfg.update(logger_overrides)
    return OmegaConf.create({"runner": {"logger": logger_cfg}})


@pytest.fixture
def metric_logger(tmp_path):
    """A logger with no real backends; tests attach fakes."""
    return MetricLogger(_cfg(tmp_path))


def _install(metric_logger, **backends):
    metric_logger.logger.update(backends)
    return backends


# ------------------------------------------------------------------- protocol


def test_fakes_satisfy_the_backend_protocol():
    """``MetricBackend`` is runtime-checkable so a fake proves the shape."""
    assert isinstance(FakeBackend(), MetricBackend)


def test_real_backend_adapters_satisfy_the_protocol():
    """wandb and swanlab are wrapped in adapters rather than used as bare
    modules, so that flush/log_media mean the same thing everywhere."""
    from rlinf.utils.metric_logger import _SwanlabBackend, _WandbBackend

    class _Module:
        def __getattr__(self, name):
            return lambda *a, **k: None

    for adapter in (_WandbBackend(_Module()), _SwanlabBackend(_Module())):
        assert isinstance(adapter, MetricBackend), adapter.name


def test_unsupported_backend_name_is_rejected(tmp_path):
    """A typo in ``logger_backends`` should fail at construction, not silently
    log nowhere for a whole run."""
    with pytest.raises(AssertionError, match="Unsupported logger backend"):
        MetricLogger(_cfg(tmp_path, logger_backends=["not_a_backend"]))


def test_del_is_safe_on_a_partially_constructed_logger(tmp_path):
    """A config error must surface as itself.

    ``__init__`` asserts on an unknown backend name before the instance
    attributes exist, and ``__del__`` then runs on the half-built object. Without
    class-level defaults that raised ``AttributeError`` from inside ``__del__``,
    which Python reports as an unraisable exception and which buries the actual
    config mistake.
    """
    import gc

    with pytest.raises(AssertionError):
        MetricLogger(_cfg(tmp_path, logger_backends=["not_a_backend"]))
    gc.collect()

    # Reachable directly too, without relying on gc timing.
    orphan = MetricLogger.__new__(MetricLogger)
    orphan.__del__()


def test_string_backend_config_is_accepted(tmp_path):
    logger = MetricLogger(_cfg(tmp_path, logger_backends="tensorboard"))
    assert logger.logger_backends == ["tensorboard"]


def test_none_backend_config_means_no_backends(tmp_path):
    logger = MetricLogger(_cfg(tmp_path, logger_backends=None))
    assert logger.logger_backends == []


# ------------------------------------------------------------------- dispatch


def test_log_reaches_every_backend(metric_logger):
    fakes = _install(metric_logger, a=FakeBackend(), b=FakeBackend())
    metric_logger.log({"env/success_once": 0.5}, step=3)

    for backend in fakes.values():
        assert backend.logged == [({"env/success_once": 0.5}, 3)]


def test_log_can_target_a_subset_of_backends(metric_logger):
    fakes = _install(metric_logger, a=FakeBackend(), b=FakeBackend())
    metric_logger.log({"x": 1.0}, step=1, backend=["a"])

    assert len(fakes["a"].logged) == 1
    assert fakes["b"].logged == []


def test_log_normalizes_keys_once_at_the_entry_point(metric_logger):
    """Normalizing here is what makes the dual write cover all runners."""
    fake = _install(metric_logger, a=FakeBackend())["a"]
    metric_logger.log({"actor/training/loss": 1.5}, step=1)

    data, _ = fake.logged[0]
    assert data == {"actor/training/loss": 1.5, "train/actor/loss": 1.5}


def test_log_table_dispatches_through_backends(metric_logger):
    fake = _install(metric_logger, a=FakeBackend())["a"]
    metric_logger.log_table("dataframe", "my_table", 4)

    assert fake.tables == [("dataframe", "my_table", 4)]


def test_log_table_warns_and_skips_when_unsupported(metric_logger, capsys):
    """This used to raise, turning a missing capability into a crashed run."""
    fake = _install(metric_logger, a=Tableless())["a"]
    metric_logger.log_table("dataframe", "my_table", 4)

    # No exception, and the backend still works for what it does support.
    metric_logger.log({"x": 1.0}, step=1)
    assert len(fake.logged) == 1


def test_log_media_dispatches_with_kwargs(metric_logger):
    fake = _install(metric_logger, a=FakeBackend())["a"]
    metric_logger.log_media("/tmp/v.mp4", "rollout", 7, fps=30)

    assert fake.media == [("/tmp/v.mp4", "rollout", 7, {"fps": 30})]


def test_log_media_warns_and_skips_when_unsupported(metric_logger):
    _install(metric_logger, a=Tableless())
    metric_logger.log_media("/tmp/v.mp4", "rollout", 7)


# ---------------------------------------------------------------------- flush


def test_flush_interval_defaults(metric_logger):
    assert metric_logger.flush_interval_s == DEFAULT_FLUSH_INTERVAL_S


def test_flush_interval_is_configurable(tmp_path):
    logger = MetricLogger(_cfg(tmp_path, flush_interval_s=1.5))
    assert logger.flush_interval_s == 1.5


def test_log_does_not_flush_on_every_call(metric_logger):
    """Per-call flushing on a per-step path would be needless IO."""
    fake = _install(metric_logger, a=FakeBackend())["a"]
    for step in range(5):
        metric_logger.log({"x": 1.0}, step=step)

    assert fake.flushes == 0


def test_log_flushes_once_the_interval_elapses(tmp_path):
    """TensorBoard's own buffering can otherwise leave a live run looking
    stalled in the UI for minutes."""
    logger = MetricLogger(_cfg(tmp_path, flush_interval_s=0.0))
    fake = _install(logger, a=FakeBackend())["a"]

    logger.log({"x": 1.0}, step=1)
    assert fake.flushes == 1


def test_explicit_flush_reaches_all_backends(metric_logger):
    fakes = _install(metric_logger, a=FakeBackend(), b=FakeBackend())
    metric_logger.flush()

    assert all(backend.flushes == 1 for backend in fakes.values())


def test_a_failing_flush_does_not_stop_the_others(metric_logger):
    class Exploding(FakeBackend):
        def flush(self):
            raise RuntimeError("disk full")

    fakes = _install(metric_logger, bad=Exploding(), good=FakeBackend())
    metric_logger.flush()

    assert fakes["good"].flushes == 1


# ---------------------------------------------------------- finish idempotence


def test_finish_closes_every_backend(metric_logger):
    fakes = _install(metric_logger, a=FakeBackend(), b=FakeBackend())
    metric_logger.finish()

    assert all(backend.finishes == 1 for backend in fakes.values())


def test_finish_is_idempotent(metric_logger):
    """Three real callers now: runner teardown, ``__del__``, and
    ``run_lifecycle``'s finally. A5 depends on this."""
    fake = _install(metric_logger, a=FakeBackend())["a"]

    metric_logger.finish()
    metric_logger.finish()
    metric_logger.finish()

    assert fake.finishes == 1


def test_del_after_explicit_finish_does_not_double_close(tmp_path):
    logger = MetricLogger(_cfg(tmp_path))
    fake = _install(logger, a=FakeBackend())["a"]

    logger.finish()
    logger.__del__()

    assert fake.finishes == 1


def test_tensorboard_backend_close_is_idempotent():
    """The writer is closed by runner teardown and again at shutdown."""
    from rlinf.utils.metric_logger import _TensorboardLogger

    class FakeWriter:
        def __init__(self):
            self.closes = 0
            self.flushes = 0

        def flush(self):
            self.flushes += 1

        def close(self):
            self.closes += 1

    backend = _TensorboardLogger.__new__(_TensorboardLogger)
    backend.writer = FakeWriter()
    backend._closed = False

    backend.finish()
    backend.finish()
    assert backend.writer.closes == 1

    # Flushing a closed writer must not resurrect it either.
    backend.flush()
    assert backend.writer.flushes == 0


@pytest.mark.parametrize("adapter_name", ["_WandbBackend", "_SwanlabBackend"])
def test_module_adapters_finish_once(adapter_name):
    import rlinf.utils.metric_logger as module

    calls = []

    class _Module:
        def finish(self):
            calls.append(1)

        def __getattr__(self, name):
            return lambda *a, **k: None

    adapter = getattr(module, adapter_name)(_Module())
    adapter.finish()
    adapter.finish()

    assert len(calls) == 1


# ------------------------------------------------------------------- callback


def test_log_callback_fires_after_each_log(metric_logger):
    """This is what keeps ``last_metric_at`` current, so a reader can tell a
    broken metric path from a stalled run."""
    calls = []
    metric_logger.set_log_callback(lambda: calls.append(1))
    _install(metric_logger, a=FakeBackend())

    metric_logger.log({"x": 1.0}, step=1)
    metric_logger.log({"x": 2.0}, step=2)

    assert len(calls) == 2


def test_a_failing_callback_does_not_lose_the_metric(metric_logger):
    """The control plane must not be able to break the data plane."""

    def explode():
        raise RuntimeError("reporter died")

    metric_logger.set_log_callback(explode)
    fake = _install(metric_logger, a=FakeBackend())["a"]

    metric_logger.log({"x": 1.0}, step=1)
    assert len(fake.logged) == 1


def test_no_callback_is_fine(metric_logger):
    """MetricLogger never imports run_state; the callback is optional."""
    fake = _install(metric_logger, a=FakeBackend())["a"]
    metric_logger.log({"x": 1.0}, step=1)
    assert len(fake.logged) == 1
