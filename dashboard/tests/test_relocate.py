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

"""Reading a run tree from a different namespace than it was written in.

This is not a hypothetical. The bug these tests pin was found by exporting a
real container run to a laptop: 56 metric keys became 0, the run rendered with
no charts, and nothing in the API said why. The container-to-host-mount case is
the *standard* Docker deployment, so the failure was on the main path.
"""

from __future__ import annotations

import json
import os

from rlinf_dashboard.discovery import RunDiscovery
from rlinf_dashboard.relocate import derive_prefix, relocate_paths
from rlinf_dashboard.settings import Settings

# --------------------------------------------------------------- prefix derivation


def test_prefix_is_derived_from_the_shared_run_id_tail():
    """The two paths end in the same `_rlinf/runs/<id>`, which pins the mapping."""
    prefix = derive_prefix(
        "/workspace/run1/logs/_rlinf/runs/20260804-abc",
        "/data/mnt/logs/_rlinf/runs/20260804-abc",
    )
    assert prefix == ("/workspace/run1", "/data/mnt")


def test_identical_paths_need_no_translation():
    root = "/logs/_rlinf/runs/r1"
    assert derive_prefix(root, root) is None


def test_trailing_slash_is_not_a_difference():
    assert derive_prefix("/logs/_rlinf/runs/r1/", "/logs/_rlinf/runs/r1") is None


def test_unrelated_paths_yield_no_mapping():
    """Different run ids are not two views of one directory.

    Deriving a prefix here would let a stray manifest rewrite another run's
    paths, which is worse than the blank charts this module exists to fix.
    """
    assert derive_prefix("/a/_rlinf/runs/r1", "/b/_rlinf/runs/DIFFERENT") is None


def test_missing_recorded_root_is_not_an_error():
    """An older training side may not have recorded `run_root` at all."""
    assert derive_prefix(None, "/logs/_rlinf/runs/r1") is None


# ------------------------------------------------------------------- path rewriting


def _tree(tmp_path, recorded_prefix="/workspace/orig"):
    """A run tree on disk whose manifest claims it lives somewhere else."""
    actual = tmp_path / "mounted"
    run_root = actual / "_rlinf" / "runs" / "r1"
    run_root.mkdir(parents=True)
    (actual / "tensorboard").mkdir()
    paths = {
        "log_path": f"{recorded_prefix}",
        "tensorboard": f"{recorded_prefix}/tensorboard",
        "video_root": f"{recorded_prefix}/video",
        "run_root": f"{recorded_prefix}/_rlinf/runs/r1",
    }
    return str(actual), str(run_root), paths


def test_existing_path_is_rewritten_to_where_it_actually_is(tmp_path):
    _actual, run_root, paths = _tree(tmp_path)
    out, relocation = relocate_paths(paths, paths["run_root"], run_root)

    assert out["tensorboard"] == os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(run_root))), "tensorboard"
    )
    assert os.path.isdir(out["tensorboard"])
    assert relocation is not None
    assert relocation["from_prefix"] == "/workspace/orig"
    assert "tensorboard" in relocation["rewritten"]


def test_the_per_worker_log_root_is_relocated_too(tmp_path):
    """A container-written ``worker_logs`` must survive the same translation.

    Left untranslated, the run still charts (the aggregate bundle relocates) but
    the per-rank drill-down silently offers nothing -- the same class of blank
    that relocation exists to prevent, one layer down.
    """
    actual, run_root, paths = _tree(tmp_path)
    worker_logs = os.path.join(actual, "worker_logs", "EnvGroup", "rank_0")
    os.makedirs(worker_logs)
    paths["worker_logs"] = "/workspace/orig/worker_logs"

    out, relocation = relocate_paths(paths, paths["run_root"], run_root)
    assert out["worker_logs"] == os.path.join(actual, "worker_logs")
    assert relocation is not None
    assert "worker_logs" in relocation["rewritten"]


def test_a_path_that_does_not_exist_either_way_keeps_the_recorded_value(tmp_path):
    """Never substitute a guess for a path that is wrong anyway.

    `video_root` has no counterpart in this tree. Reporting what the training
    side actually recorded is more useful than a fabricated local path, which
    would send someone looking in a directory that never held videos.
    """
    _actual, run_root, paths = _tree(tmp_path)
    out, _ = relocate_paths(paths, paths["run_root"], run_root)
    assert out["video_root"] == "/workspace/orig/video"


def test_a_recorded_path_that_resolves_is_left_alone(tmp_path):
    """A reader may share the writer's namespace for some paths and not others.

    A bind-mounted checkpoint volume mounted at its original path must not be
    rewritten just because the log directory moved.
    """
    shared = tmp_path / "shared-ckpt"
    shared.mkdir()
    _actual, run_root, paths = _tree(tmp_path)
    paths["checkpoint_root"] = str(shared)

    out, _ = relocate_paths(paths, paths["run_root"], run_root)
    assert out["checkpoint_root"] == str(shared)


def test_run_root_is_always_the_on_disk_location(tmp_path):
    """Discovery's value is correct by construction; the recorded one may not be.

    Leaving the stale `run_root` in place would show the user a path they cannot
    open on the machine they are looking at.
    """
    _actual, run_root, paths = _tree(tmp_path)
    out, relocation = relocate_paths(paths, paths["run_root"], run_root)
    assert out["run_root"] == run_root
    assert relocation is not None


def test_untranslated_tree_returns_the_input_unchanged(tmp_path):
    """No relocation means no copy and no `relocation` field on the API."""
    actual = tmp_path / "logs"
    run_root = actual / "_rlinf" / "runs" / "r1"
    run_root.mkdir(parents=True)
    paths = {"log_path": str(actual), "run_root": str(run_root)}

    out, relocation = relocate_paths(paths, paths["run_root"], str(run_root))
    assert relocation is None
    assert out is paths


# ------------------------------------------------------------ end to end via discovery


def test_discovery_repairs_a_container_written_tree(tmp_path):
    """The regression this module exists for, end to end.

    A manifest written inside a container, read from a host mount. Before the
    fix the metric reader found no event files here and the run rendered empty.
    """
    log_path = tmp_path / "hostmount"
    run_root = log_path / "_rlinf" / "runs" / "20260804-abc"
    run_root.mkdir(parents=True)
    (log_path / "tensorboard").mkdir()
    (run_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "20260804-abc",
                "task_type": "embodied",
                "paths": {
                    "log_path": "/workspace/f19/run1/logs",
                    "tensorboard": "/workspace/f19/run1/logs/tensorboard",
                    "run_root": "/workspace/f19/run1/logs/_rlinf/runs/20260804-abc",
                },
            }
        )
    )

    runs = RunDiscovery(Settings(scan_roots=[str(log_path)])).list_runs()
    assert len(runs) == 1
    run = runs[0]
    assert run.relocation is not None
    assert run.manifest.paths["tensorboard"] == str(log_path / "tensorboard")
    assert os.path.isdir(run.manifest.paths["tensorboard"])


def test_media_streams_from_a_relocated_tree(tmp_path):
    """A clip listed from a moved tree must also be fetchable.

    Listing every video and then 404-ing all of them is a worse failure than not
    listing them, because it reads as corrupt output rather than a moved tree.
    """
    from fastapi.testclient import TestClient

    from rlinf_dashboard.api import create_app

    log_path = tmp_path / "hostmount"
    run_root = log_path / "_rlinf" / "runs" / "r1"
    run_root.mkdir(parents=True)
    (log_path / "tensorboard").mkdir()
    clip = log_path / "video" / "eval" / "seed_0" / "0.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    recorded = "/workspace/f19/run1/logs"
    (run_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "r1",
                "task_type": "embodied",
                "paths": {
                    "log_path": recorded,
                    "tensorboard": f"{recorded}/tensorboard",
                    "video_root": f"{recorded}/video",
                    "run_root": f"{recorded}/_rlinf/runs/r1",
                },
            }
        )
    )
    (run_root / "media.rank0.jsonl").write_text(
        json.dumps(
            {
                "path": f"{recorded}/video/eval/seed_0/0.mp4",
                "step": 0,
                "split": "eval",
                "num_success": 0,
                "num_envs": 2,
            }
        )
        + "\n"
    )

    client = TestClient(create_app(Settings(scan_roots=[str(log_path)])))
    rows = client.get("/api/runs/r1/media").json()
    assert len(rows) == 1
    assert rows[0]["path"] == str(clip)

    # Fetch exactly the URL the listing handed out, which is what a browser does.
    assert client.get(rows[0]["url"]).status_code == 200


def test_relocation_does_not_rewrite_the_resume_command(tmp_path):
    """`resume_dir` and `entry_script` must stay as the training side wrote them.

    They exist to be pasted into a command on the machine that will run it. A
    path translated into the reader's namespace would produce a resume command
    that works only on the laptop someone happened to be browsing from.
    """
    from fastapi.testclient import TestClient

    from rlinf_dashboard.api import create_app

    log_path = tmp_path / "hostmount"
    run_root = log_path / "_rlinf" / "runs" / "r1"
    run_root.mkdir(parents=True)
    (log_path / "tensorboard").mkdir()
    recorded = "/workspace/f19/run1/logs"
    (run_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "r1",
                "task_type": "embodied",
                "paths": {
                    "log_path": recorded,
                    "tensorboard": f"{recorded}/tensorboard",
                    "run_root": f"{recorded}/_rlinf/runs/r1",
                },
            }
        )
    )
    (run_root / "checkpoints.jsonl").write_text(
        json.dumps(
            {
                "step": 1,
                "path": f"{recorded}/exp/checkpoints/global_step_1",
                "saved_at": "2026-08-04T08:38:55Z",
                "resume_dir": f"{recorded}/exp/checkpoints/global_step_1",
                "entry_script": "/workspace/tree/examples/embodiment/x.py",
                "config_name": "libero_10_ppo_openpi_pi05",
            }
        )
        + "\n"
    )

    client = TestClient(create_app(Settings(scan_roots=[str(log_path)])))
    entry = client.get("/api/runs/r1/checkpoints").json()[0]
    assert entry["resume_dir"] == f"{recorded}/exp/checkpoints/global_step_1"
    assert entry["entry_script"] == "/workspace/tree/examples/embodiment/x.py"


def test_discovery_leaves_a_local_tree_untouched(tmp_path):
    """A run read on the machine that wrote it reports no relocation at all."""
    log_path = tmp_path / "logs"
    run_root = log_path / "_rlinf" / "runs" / "r1"
    run_root.mkdir(parents=True)
    (log_path / "tensorboard").mkdir()
    (run_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "r1",
                "task_type": "embodied",
                "paths": {
                    "log_path": str(log_path),
                    "tensorboard": str(log_path / "tensorboard"),
                    "run_root": str(run_root),
                },
            }
        )
    )

    runs = RunDiscovery(Settings(scan_roots=[str(log_path)])).list_runs()
    assert runs[0].relocation is None
