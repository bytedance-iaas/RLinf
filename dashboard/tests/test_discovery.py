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

Discovery is where a dashboard most often fails in a way that looks like nothing:
it shows zero runs and gives no hint why. So these tests are mostly about the
awkward trees a real log path becomes -- nested experiments, a checkpoint
directory with thousands of shards next door, two scan roots overlapping, a run
that died before writing anything.
"""

from __future__ import annotations

import json
import os

import pytest

from rlinf_dashboard.discovery import RunDiscovery


def _manifest(run_id: str, **extra) -> dict:
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "task_type": "embodied",
        "experiment_name": run_id,
    }
    payload.update(extra)
    return payload


def test_finds_a_run_under_a_scan_root(run_tree, settings_for):
    run_tree("20260803-100000-alpha")
    runs = RunDiscovery(settings_for()).list_runs()
    assert [run.run_id for run in runs] == ["20260803-100000-alpha"]


def test_set_root_scans_somewhere_else(tmp_path, settings_for):
    """An operator repointing the server sees the other root's runs."""
    for name, run_id in (("logs", "run-default"), ("elsewhere", "run-other")):
        root = tmp_path / name / "_rlinf" / "runs" / run_id
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps(_manifest(run_id)))

    discovery = RunDiscovery(settings_for())
    assert discovery.is_default_root is True
    assert [run.run_id for run in discovery.list_runs()] == ["run-default"]

    discovery.set_root(str(tmp_path / "elsewhere"))
    assert discovery.root == str(tmp_path / "elsewhere")
    assert discovery.is_default_root is False
    assert [run.run_id for run in discovery.list_runs()] == ["run-other"]

    # The configured value is untouched, which is what makes reset possible.
    assert discovery.default_root == str(tmp_path / "logs")
    discovery.set_root(None)
    assert discovery.is_default_root is True
    assert [run.run_id for run in discovery.list_runs()] == ["run-default"]


def test_set_root_drops_the_cache(tmp_path, settings_for):
    """Without this, the new root serves the old root's runs for a whole TTL.

    Every caller today happens to refresh right after repointing, so this is
    asserted here rather than through the API: the guarantee belongs to the
    method, and a future caller that does not refresh must not read stale runs
    under the new root's name.
    """
    for name, run_id in (("logs", "run-default"), ("elsewhere", "run-other")):
        root = tmp_path / name / "_rlinf" / "runs" / run_id
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps(_manifest(run_id)))

    discovery = RunDiscovery(settings_for(discovery_cache_ttl_s=3600.0))
    assert [run.run_id for run in discovery.list_runs()] == ["run-default"]

    discovery.set_root(str(tmp_path / "elsewhere"))
    # No `refresh=True`: the cache must already be gone.
    assert [run.run_id for run in discovery.list_runs()] == ["run-other"]


def test_finds_runs_nested_below_the_scan_root(tmp_path, settings_for):
    """The normal case: one root over many experiments.

    Users point the dashboard at the parent of a dozen experiment directories,
    each with its own ``_rlinf``. Requiring the root to *be* a log path would mean
    one server per experiment.
    """
    for name in ("exp-a", "exp-b"):
        root = tmp_path / "logs" / name / "_rlinf" / "runs" / f"run-{name}"
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps(_manifest(f"run-{name}")))

    runs = RunDiscovery(settings_for()).list_runs()
    assert {run.run_id for run in runs} == {"run-exp-a", "run-exp-b"}


def test_a_directory_without_a_manifest_is_not_a_run(tmp_path, settings_for):
    """The manifest is written in the reporter's constructor, before any worker.

    So a run directory with no manifest is not "starting up" -- it is a leftover.
    Listing it would put a permanently blank row in the table.
    """
    (tmp_path / "logs" / "_rlinf" / "runs" / "empty-dir").mkdir(parents=True)
    assert RunDiscovery(settings_for()).list_runs() == []


def test_an_unparseable_manifest_is_skipped_not_fatal(run_tree, settings_for, tmp_path):
    """One truncated manifest must not blank the whole list.

    A manifest can be caught mid-write, and the other twenty runs on the machine
    are still perfectly readable.
    """
    run_tree("good-run")
    broken = tmp_path / "logs" / "_rlinf" / "runs" / "broken-run"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{not json at all")

    runs = RunDiscovery(settings_for()).list_runs()
    assert [run.run_id for run in runs] == ["good-run"]


def test_runs_are_ordered_newest_first(tmp_path, settings_for):
    """The list view's default order; the newest run is the one being watched."""
    for run_id, started in [
        ("older", "2026-08-01T10:00:00Z"),
        ("newest", "2026-08-03T10:00:00Z"),
        ("middle", "2026-08-02T10:00:00Z"),
    ]:
        root = tmp_path / "logs" / "_rlinf" / "runs" / run_id
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps(_manifest(run_id, started_at=started))
        )

    runs = RunDiscovery(settings_for()).list_runs()
    assert [run.run_id for run in runs] == ["newest", "middle", "older"]


def test_runs_without_started_at_fall_back_to_manifest_mtime(tmp_path, settings_for):
    """A v1 manifest has no ``started_at`` and must still sort sensibly."""
    for index, run_id in enumerate(["first", "second"]):
        root = tmp_path / "logs" / "_rlinf" / "runs" / run_id
        root.mkdir(parents=True)
        path = root / "manifest.json"
        path.write_text(json.dumps(_manifest(run_id)))
        os.utime(path, (1_700_000_000 + index * 100,) * 2)

    runs = RunDiscovery(settings_for()).list_runs()
    assert [run.run_id for run in runs] == ["second", "first"]


def test_a_run_reachable_by_two_paths_is_listed_once(tmp_path, settings_for):
    """One root can still reach one run twice, through a symlinked directory.

    Deduping on the resolved path keeps it from appearing twice under different
    spellings, which would look like a duplicated job.
    """
    root = tmp_path / "logs" / "_rlinf" / "runs" / "dup-run"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(_manifest("dup-run")))
    alias = tmp_path / "logs" / "mirror"
    try:
        alias.symlink_to(tmp_path / "logs", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")

    settings = settings_for(scan_root=str(tmp_path))
    runs = RunDiscovery(settings).list_runs()
    assert [run.run_id for run in runs] == ["dup-run"]


def test_the_latest_symlink_is_not_listed_as_a_second_run(run_tree, settings_for):
    """``latest`` points at a run directory; following it would double-count.

    It lives beside ``runs/`` rather than inside it, but symlinks are refused
    while scanning regardless -- a resolved symlink is the same run under another
    name.
    """
    run_tree("linked-run")
    runs = RunDiscovery(settings_for()).list_runs()
    assert [run.run_id for run in runs] == ["linked-run"]


def test_a_missing_scan_root_is_not_an_error(settings_for):
    """A mistyped path must yield an empty list, not a 500.

    ``/api/health`` reports which roots exist, which is where that mistake gets
    diagnosed; crashing the list endpoint would hide it.
    """
    assert RunDiscovery(settings_for(scan_root="/nonexistent/path")).list_runs() == []


def test_checkpoint_trees_are_not_walked(tmp_path, settings_for):
    """A scan must not descend into the data-plane trees sharing the log path.

    Checkpoint directories hold tens of thousands of shard files. Walking them to
    find nothing turns ``/api/runs`` from milliseconds into seconds, and on NFS
    much worse. The canary here is a file planted deep inside one: if the walk
    reached it, the pruning is gone.
    """
    logs = tmp_path / "logs"
    run_root = logs / "_rlinf" / "runs" / "real-run"
    run_root.mkdir(parents=True)
    (run_root / "manifest.json").write_text(json.dumps(_manifest("real-run")))

    buried = logs / "checkpoints" / "global_step_100" / "_rlinf" / "runs" / "decoy"
    buried.mkdir(parents=True)
    (buried / "manifest.json").write_text(json.dumps(_manifest("decoy")))

    runs = RunDiscovery(settings_for()).list_runs()
    assert [run.run_id for run in runs] == ["real-run"]


def test_depth_bound_stops_a_runaway_walk(tmp_path, settings_for):
    """Bound the walk so a deep tree cannot hang the endpoint."""
    deep = tmp_path / "logs"
    for level in range(10):
        deep = deep / f"level{level}"
    run_root = deep / "_rlinf" / "runs" / "too-deep"
    run_root.mkdir(parents=True)
    (run_root / "manifest.json").write_text(json.dumps(_manifest("too-deep")))

    assert RunDiscovery(settings_for(scan_max_depth=3)).list_runs() == []


# ------------------------------------------------------------------------- caching


def test_the_cache_serves_repeated_calls(run_tree, settings_for):
    """A scan stats a file per candidate; on NFS that lands in request latency."""
    run_tree("cached-run")
    discovery = RunDiscovery(settings_for(discovery_cache_ttl_s=300.0))
    assert len(discovery.list_runs()) == 1

    run_tree("added-later")
    assert len(discovery.list_runs()) == 1, "second call should be cached"
    assert len(discovery.list_runs(refresh=True)) == 2


def test_find_refreshes_before_giving_up(run_tree, settings_for):
    """A run launched moments ago must be reachable by direct URL.

    Someone pastes a run URL right after launching. Waiting out the cache TTL
    would 404 a run that plainly exists, so a miss forces one rescan.
    """
    discovery = RunDiscovery(settings_for(discovery_cache_ttl_s=300.0))
    discovery.list_runs()  # warm the cache while empty

    run_tree("just-launched")
    assert discovery.find("just-launched") is not None
    assert discovery.find("never-existed") is None


def test_manifest_metric_aliases_survive_discovery(run_tree, settings_for):
    """The aliases are why the gateway can resolve legacy keys without ``rlinf``.

    They are embedded in the manifest by the training side for exactly that
    reason, so losing them here would silently break every chart on a run from
    before the rename.
    """
    run_tree("aliased-run")
    run = RunDiscovery(settings_for()).find("aliased-run")
    assert run is not None
    assert run.manifest.metric_aliases["actor/training/"] == "train/actor/"
