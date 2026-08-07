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

"""Shared test scaffolding for the standalone dashboard package."""

from __future__ import annotations

import json
import os
import shutil
from importlib.resources import files
from pathlib import Path

import pytest

SCHEMA_RESOURCE = files("rlinf_dashboard").joinpath("schemas", "run.v2.schema.json")
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "run_state"


def fixture_names() -> list[str]:
    if not FIXTURE_DIR.is_dir():
        return []
    return sorted(
        path.stem for path in FIXTURE_DIR.iterdir() if path.suffix == ".json"
    )


def load_fixture(name: str) -> dict:
    with (FIXTURE_DIR / f"{name}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def load_schema() -> dict:
    with SCHEMA_RESOURCE.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def schema() -> dict:
    return load_schema()


@pytest.fixture
def run_tree(tmp_path):
    """Build a control-plane directory tree the way the training side writes one.

    Returns a callable ``make(run_id, snapshot=..., **files)`` so a test can
    describe the run it needs rather than assemble paths. The layout mirrors
    ``RunStateReporter``: ``<log_path>/_rlinf/runs/<run_id>/`` with a ``latest``
    symlink beside ``runs/``.
    """
    log_path = tmp_path / "logs"

    def make(
        run_id: str,
        *,
        manifest: dict | None = None,
        snapshot: dict | None = None,
        events: list[dict] | None = None,
        checkpoints: list[dict] | None = None,
        media: dict[int, list[dict]] | None = None,
        heartbeat_seq: int | None = None,
    ) -> str:
        run_root = log_path / "_rlinf" / "runs" / run_id
        run_root.mkdir(parents=True, exist_ok=True)

        base_manifest = {
            "schema_version": 2,
            "run_id": run_id,
            "task_type": "embodied",
            "experiment_name": "test-exp",
            "project_name": "test-proj",
            "step_semantics": "rl_iteration",
            "heartbeat_interval_s": 5.0,
            "paths": {
                "log_path": str(log_path),
                "tensorboard": str(log_path / "tensorboard"),
                "video_root": str(log_path / "video"),
            },
            "metric_aliases": {
                "actor/training/": "train/actor/",
                "critic/training/": "train/critic/",
            },
        }
        if manifest is not None:
            base_manifest.update(manifest)
        (run_root / "manifest.json").write_text(json.dumps(base_manifest, indent=2))

        if snapshot is not None:
            payload = dict(snapshot)
            payload.setdefault("run_id", run_id)
            (run_root / "run.json").write_text(json.dumps(payload, indent=2))

        if events:
            (run_root / "events.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events)
            )
        if checkpoints:
            (run_root / "checkpoints.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in checkpoints)
            )
        for shard, rows in (media or {}).items():
            (run_root / f"media.rank{shard}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
        if heartbeat_seq is not None:
            (run_root / "heartbeat").write_text(f"{heartbeat_seq}\n")

        link = log_path / "_rlinf" / "latest"
        if link.is_symlink() or link.exists():
            link.unlink()
        try:
            link.symlink_to(os.path.join("runs", run_id))
        except OSError:
            pass
        return str(run_root)

    make.log_path = str(log_path)  # type: ignore[attr-defined]
    return make


@pytest.fixture
def settings_for(tmp_path):
    """Build :class:`Settings` pointed at a temporary tree.

    ``.env`` discovery is disabled: a developer's own ``.env`` next to the repo
    would otherwise leak into the test and change which roots are scanned.
    """
    from rlinf_dashboard.settings import Settings

    def make(**overrides):
        kwargs = {
            "scan_roots": [str(tmp_path / "logs")],
            "discovery_cache_ttl_s": 0.0,
            "cors_origins": [],
            "_env_file": None,
        }
        kwargs.update(overrides)
        return Settings(**kwargs)

    return make


@pytest.fixture
def no_rlinf(monkeypatch):
    """Assert that nothing under test can import ``rlinf``.

    The hard architectural constraint for this package. Enforced here as well as
    by the CI job so a stray ``import rlinf`` fails on a laptop rather than only
    in the isolated venv.
    """
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "rlinf" or name.startswith("rlinf."):
            raise AssertionError(
                f"rlinf_dashboard must never import {name}: the cross-process "
                f"contract is the filesystem and HTTP only."
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    yield


def write_event_file(log_dir: str, series: dict[str, list[tuple[int, float]]]) -> None:
    """Write a real TensorBoard event file, or skip if the writer is unavailable.

    A real event file rather than a mock: the point of these tests is that the
    protobuf reader handles what the training side actually produces, and a mock
    would only prove the mock matches itself.
    """
    summary_writer = pytest.importorskip(
        "tensorboard.summary.writer.event_file_writer",
        reason="tensorboard is a declared dependency; skip if absent",
    )
    from tensorboard.compat.proto import event_pb2, summary_pb2

    os.makedirs(log_dir, exist_ok=True)
    writer = summary_writer.EventFileWriter(log_dir)
    try:
        for key, points in series.items():
            for step, value in points:
                summary = summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag=key, simple_value=value)]
                )
                writer.add_event(
                    event_pb2.Event(step=step, summary=summary, wall_time=1000.0 + step)
                )
    finally:
        writer.close()


def cleanup(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
