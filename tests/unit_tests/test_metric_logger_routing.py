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

"""Each vendor-logger bundle must write to the run it created.

``wandb.log`` and ``swanlab.log`` are module-level calls that target whichever
run is *currently active*, and ``init`` makes its new run active. With
``runner.per_worker_log: true`` a bundle is created for the aggregate and then
one per (worker group, rank), so with module-level logging every rank's ``init``
silently re-points the bundles created before it -- the aggregate's metrics end
up filed under the last rank to start, and ``finish()`` closes that rank's run.

No exception is raised when this happens. The numbers are real, the run is real,
and only the attribution is wrong, which is why it needs a test rather than a
try/except. The tensorboard backend was never affected: it holds a writer bound
to a path, which is the property the other two now have as well.

Uses fake modules that reproduce exactly the one behaviour that matters -- init
mutates a module-level ``run`` -- so the test needs neither package installed.
"""

from __future__ import annotations

import pytest

from rlinf.utils.metric_logger import _SwanlabBackend, _WandbBackend


class _FakeRun:
    """A vendor run handle. Records what was logged to *it* specifically."""

    def __init__(self, name: str):
        self.name = name
        self.logged: list[tuple[dict, int]] = []
        self.finished = False

    def log(self, data=None, step=None, **kwargs):
        # wandb's module-level log takes the payload positionally in some call
        # sites and as `data=` in others; accept both the way the real one does.
        if data is None and kwargs:
            data = kwargs
        self.logged.append((data, step))

    def finish(self):
        self.finished = True


class _FakeVendorModule:
    """Module-level ``log``/``finish`` that follow the active run, like the real ones."""

    def __init__(self):
        self.run: _FakeRun | None = None

    def init(self, name: str) -> _FakeRun:
        self.run = _FakeRun(name)
        return self.run

    def log(self, data=None, step=None, **kwargs):
        self.run.log(data=data, step=step, **kwargs)

    def finish(self):
        self.run.finish()

    class Table:  # noqa: D106 - stand-in for wandb.Table
        def __init__(self, dataframe=None):
            self.dataframe = dataframe

    class Video:  # noqa: D106 - stand-in for wandb.Video
        def __init__(self, path, **kwargs):
            self.path = path


@pytest.mark.parametrize("backend_cls", [_WandbBackend, _SwanlabBackend])
def test_a_later_init_does_not_steal_an_earlier_bundles_metrics(backend_cls):
    """The aggregate bundle keeps its own run after per-rank bundles start."""
    module = _FakeVendorModule()

    aggregate_run = module.init("aggregate")
    aggregate = backend_cls(module, aggregate_run)

    # Per-worker logging then creates a bundle per rank; each init makes its own
    # run the module-level active one.
    rank_runs = [module.init(f"rank_{n}") for n in range(2)]
    ranks = [backend_cls(module, run) for run in rank_runs]

    aggregate.log({"train/loss": 1.0}, step=1)
    for n, rank in enumerate(ranks):
        rank.log({"train/loss": float(n)}, step=1)

    assert aggregate_run.logged == [({"train/loss": 1.0}, 1)], (
        "the aggregate's metrics were written to another run: a later init made "
        "its own run the module-level target"
    )
    for n, run in enumerate(rank_runs):
        assert run.logged == [({"train/loss": float(n)}, 1)], (
            f"rank_{n} did not receive exactly its own metrics"
        )


@pytest.mark.parametrize("backend_cls", [_WandbBackend, _SwanlabBackend])
def test_finish_closes_the_bundles_own_run(backend_cls):
    """Otherwise the aggregate's `finish` closes whichever rank started last."""
    module = _FakeVendorModule()
    aggregate_run = module.init("aggregate")
    aggregate = backend_cls(module, aggregate_run)
    last_rank_run = module.init("rank_1")

    aggregate.finish()

    assert aggregate_run.finished is True
    assert last_rank_run.finished is False, (
        "finishing the aggregate closed a rank's run, so that rank's later "
        "writes would go to a closed run"
    )


def test_wandb_tables_and_videos_go_to_the_owning_run():
    """The media paths log through the handle too, not just the scalar path."""
    module = _FakeVendorModule()
    aggregate_run = module.init("aggregate")
    aggregate = _WandbBackend(module, aggregate_run)
    rank_run = module.init("rank_0")

    aggregate.log_table(df_data=[[1]], name="table", step=2)
    aggregate.log_media(path="/tmp/clip.mp4", name="video", step=3)

    assert len(aggregate_run.logged) == 2
    assert rank_run.logged == [], "a rank's run received the aggregate's media"


@pytest.mark.parametrize("backend_cls", [_WandbBackend, _SwanlabBackend])
def test_a_backend_built_without_a_handle_still_works(backend_cls):
    """Construction without an explicit run must not break.

    wandb exposes the active run as `wandb.run`, so that is picked up; swanlab
    has no such attribute and falls back to module-level calls, which is the old
    behaviour and correct when there is only one run.
    """
    module = _FakeVendorModule()
    run = module.init("only")
    backend = backend_cls(module)
    backend.log({"a": 1.0}, step=0)
    assert run.logged == [({"a": 1.0}, 0)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
