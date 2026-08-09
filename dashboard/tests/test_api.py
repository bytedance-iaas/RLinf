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

"""The HTTP surface, end to end through a real ASGI client.

Every layer below has its own tests, so what these add is the wiring: that the
route exists, that it returns the shape the frontend expects, and that a bad
request produces a status code rather than a traceback. A dashboard that 500s on
one malformed run file is worse than one that shows that run as broken.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from conftest import write_event_file
from fastapi.testclient import TestClient

from rlinf_dashboard.api import create_app

NOW = datetime.now(timezone.utc)


def _iso(when: datetime) -> str:
    return when.isoformat().replace("+00:00", "Z")


def _snapshot(**overrides) -> dict:
    payload = {
        "schema_version": 2,
        "run_id": "api-run",
        "task_type": "embodied",
        "state": "running",
        "phase": "rollout",
        "heartbeat_at": _iso(NOW - timedelta(seconds=2)),
        "heartbeat_seq": 12,
        "last_progress_at": _iso(NOW - timedelta(seconds=5)),
        "last_metric_at": _iso(NOW - timedelta(seconds=5)),
        "progress": {"step": 4, "max_steps": 10, "step_semantics": "rl_iteration"},
        "timing": {
            "started_at": _iso(NOW - timedelta(seconds=200)),
            "elapsed_s": 200.0,
            "step_time_p50": 40.0,
            "eta_s": 240.0,
            "eta_confidence": "medium",
        },
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def client(tmp_path, run_tree, settings_for):
    """A live app over a single running embodied run with real event files."""
    write_event_file(
        str(tmp_path / "logs" / "tensorboard"),
        {
            "env/success_once": [(0, 0.1), (1, 0.2), (2, 0.4), (3, 0.55)],
            "train/actor/loss": [(0, 2.0), (1, 1.6), (2, 1.3), (3, 1.1)],
            "my/custom/metric": [(0, 7.0)],
        },
    )
    run_tree(
        "api-run",
        snapshot=_snapshot(),
        events=[
            {"ts": _iso(NOW), "kind": "run_start", "step": 0, "payload": {}},
            {
                "ts": _iso(NOW),
                "kind": "phase_enter",
                "step": 1,
                "payload": {"phase": "rollout"},
            },
        ],
        checkpoints=[
            {
                "step": 3,
                "path": "/logs/exp/checkpoints/global_step_3",
                "saved_at": _iso(NOW),
                "size_bytes": 20350836699,
                "duration_s": 77.4,
                "resume_dir": "/logs/exp/checkpoints/global_step_3",
                "entry_script": "examples/embodiment/train_embodied_agent.py",
                "config_name": "libero_10_ppo_pi05",
            },
        ],
        heartbeat_seq=12,
    )
    with TestClient(create_app(settings_for())) as test_client:
        yield test_client


# ---------------------------------------------------------------------------- health


def test_health_reports_the_scan_root(client):
    """A dashboard showing zero runs is almost always a mistyped path.

    Reporting the root, whether it exists and how many runs are under it turns
    that into a one-look diagnosis instead of a support question.
    """
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["run_count"] == 1
    assert body["scan_root"]["exists"] is True
    assert body["scan_root"]["run_count"] == 1


def test_health_flags_a_nonexistent_scan_root(tmp_path, settings_for):
    settings = settings_for(scan_root=str(tmp_path / "typo"))
    with TestClient(create_app(settings)) as client:
        body = client.get("/api/health").json()
        assert body["scan_root"]["exists"] is False
        assert body["run_count"] == 0


def test_health_separates_a_wrong_root_from_a_missing_one(tmp_path, settings_for):
    """Both cases show an empty table; only the counts tell them apart.

    A root pointed one level off the runs exists and holds nothing, which is a
    different fix from a root that is not there at all.
    """
    (tmp_path / "empty-but-real").mkdir()
    settings = settings_for(scan_root=str(tmp_path / "empty-but-real"))
    with TestClient(create_app(settings)) as client:
        body = client.get("/api/health").json()
        assert body["scan_root"]["exists"] is True
        assert body["scan_root"]["run_count"] == 0


# ------------------------------------------------------------------------- run lists


def test_lists_runs_with_the_fields_the_table_sorts_on(client):
    rows = client.get("/api/runs").json()
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "api-run"
    assert row["state"] == "running"
    assert row["health"] == "healthy"
    assert row["phase"] == "rollout"
    assert row["step"] == 4
    assert row["max_steps"] == 10
    assert row["step_semantics"] == "rl_iteration"
    assert row["eta_s"] == 240.0
    assert row["latest_checkpoint_step"] is None


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("state=running", 1),
        ("state=finished", 0),
        ("task_type=embodied", 1),
        ("task_type=sft", 0),
    ],
)
def test_list_filters(client, query, expected):
    assert len(client.get(f"/api/runs?{query}").json()) == expected


def test_full_status_carries_the_health_verdict_and_its_evidence(client):
    """The UI must be able to explain a badge, not just show one."""
    body = client.get("/api/runs/api-run").json()
    assert body["snapshot"]["phase"] == "rollout"
    assert body["manifest"]["experiment_name"] == "test-exp"
    assert body["health"]["health"] == "healthy"
    assert body["health"]["reason"]
    # The run's own p50, which is above the 30s floor.
    assert body["health"]["budget_s"] == 40.0
    assert body["health"]["heartbeat_budget_s"] == 25.0
    assert body["error"] is None


def test_an_unknown_run_is_a_404_not_a_500(client):
    assert client.get("/api/runs/no-such-run").status_code == 404
    assert client.get("/api/runs/no-such-run/series?keys=x").status_code == 404
    assert client.get("/api/runs/no-such-run/media").status_code == 404


# ----------------------------------------------------------------- events and ckpts


def test_events_come_back_in_order(client):
    events = client.get("/api/runs/api-run/events").json()
    assert [event["kind"] for event in events] == ["run_start", "phase_enter"]
    assert events[1]["payload"]["phase"] == "rollout"


def test_checkpoints_include_the_structured_resume_fields(client):
    """The frontend assembles a display command from structured fields.

    A pre-baked command string goes stale the moment anything about the launch
    changes, and would bake in paths that may not exist on the reader's machine.
    """
    entries = client.get("/api/runs/api-run/checkpoints").json()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["step"] == 3
    assert entry["size_bytes"] == 20350836699
    assert entry["duration_s"] == 77.4
    assert entry["resume_dir"].endswith("global_step_3")
    assert entry["entry_script"].endswith("train_embodied_agent.py")
    assert entry["config_name"] == "libero_10_ppo_pi05"


# ---------------------------------------------------------------------------- series


def test_keys_lists_what_the_run_logged(client):
    body = client.get("/api/runs/api-run/keys").json()
    assert "env/success_once" in body["keys"]
    assert body["sources"] == ["tensorboard"]


def test_series_returns_points_keyed_by_the_requested_name(client):
    body = client.get(
        "/api/runs/api-run/series?keys=env/success_once&keys=train/actor/loss"
    ).json()
    assert set(body) == {"env/success_once", "train/actor/loss"}
    points = body["env/success_once"]["points"]
    assert [point["step"] for point in points] == [0, 1, 2, 3]
    assert body["env/success_once"]["source"] == "tensorboard"


def test_a_series_with_no_data_is_distinguishable_from_a_missing_run(client):
    """A not-logged-yet metric must not look like a wrong URL.

    An eval metric before the first eval is the common case, and it has to render
    as an empty chart rather than an error.
    """
    body = client.get("/api/runs/api-run/series?keys=eval/success_once").json()
    assert body["eval/success_once"]["source"] == "none"
    assert body["eval/success_once"]["points"] == []


def test_series_requires_at_least_one_key(client):
    """A missing required query parameter is a 422, not an empty answer."""
    assert client.get("/api/runs/api-run/series").status_code == 422


# ---------------------------------------------------------------- per-worker


@pytest.fixture
def per_worker_client(tmp_path, run_tree, settings_for):
    """A run with ``per_worker_log`` on: an ``all/`` bundle plus two ranks."""
    logs = tmp_path / "logs"
    write_event_file(
        str(logs / "tensorboard" / "all"),
        {"time/step": [(0, 25.0), (1, 25.5)]},
    )
    for rank, values in ((0, 10.0), (1, 40.0)):
        write_event_file(
            str(logs / "worker_logs" / "EnvGroup" / f"rank_{rank}" / "tensorboard"),
            {"time/step": [(0, values), (1, values + 1)]},
        )
    run_tree(
        "pw-run",
        manifest={
            "paths": {
                "log_path": str(logs),
                "tensorboard": str(logs / "tensorboard" / "all"),
                "worker_logs": str(logs / "worker_logs"),
            }
        },
        snapshot=_snapshot(run_id="pw-run"),
    )
    with TestClient(create_app(settings_for())) as test_client:
        yield test_client


def test_keys_advertises_the_runs_workers(per_worker_client):
    """The UI offers the drill-down only when there is something to drill into."""
    body = per_worker_client.get("/api/runs/pw-run/keys").json()
    assert body["workers"] == ["EnvGroup/rank_0", "EnvGroup/rank_1"]


def test_keys_reports_no_workers_without_per_worker_logging(client):
    """The default. An always-present control that comes back empty is worse."""
    assert client.get("/api/runs/api-run/keys").json()["workers"] == []


def test_expanding_ranks_adds_series_beside_the_aggregate(per_worker_client):
    """The aggregate keeps its own key; each rank gets a suffixed one.

    One flat map rather than a nested document, so a caller that ignores
    ``expand`` sees exactly the response it saw before this feature existed.
    """
    body = per_worker_client.get(
        "/api/runs/pw-run/series?keys=time/step&expand=ranks"
    ).json()
    assert set(body) == {
        "time/step",
        "time/step@EnvGroup/rank_0",
        "time/step@EnvGroup/rank_1",
    }
    # The aggregate is the driver's own value, not recomputed from the ranks.
    assert body["time/step"]["points"][-1]["value"] == pytest.approx(25.5)
    assert body["time/step@EnvGroup/rank_1"]["points"][-1]["value"] == pytest.approx(
        41.0
    )
    assert body["time/step@EnvGroup/rank_1"]["group"] == "EnvGroup"
    assert body["time/step@EnvGroup/rank_1"]["rank"] == 1


def test_series_without_expand_is_unchanged(per_worker_client):
    """The negative control for the test above: opt-in, and off by default.

    A page that suddenly returns 4x the series for every chart would be a silent
    cost increase on every run that happens to have the flag on.
    """
    body = per_worker_client.get("/api/runs/pw-run/series?keys=time/step").json()
    assert set(body) == {"time/step"}
    assert body["time/step"]["group"] is None
    assert body["time/step"]["rank"] is None


def test_expanding_a_run_with_no_worker_logs_is_not_an_error(client):
    """Asking to expand a run that cannot is the aggregate alone, not a 4xx.

    A bookmarked URL with ``expand=ranks`` must keep working when it is opened on
    a run launched without the flag.
    """
    response = client.get("/api/runs/api-run/series?keys=env/success_once&expand=ranks")
    assert response.status_code == 200
    assert set(response.json()) == {"env/success_once"}


def test_an_unknown_expand_value_is_rejected(per_worker_client):
    """A typo must fail loudly rather than silently return the aggregate."""
    response = per_worker_client.get(
        "/api/runs/pw-run/series?keys=time/step&expand=rank"
    )
    assert response.status_code == 422


# -------------------------------------------------------------------------- template


def test_the_template_arrives_already_bound_to_the_run_s_keys(client):
    """Binding is server-side so the frontend never learns the metric taxonomy.

    That is what makes a new ``task_type`` a YAML change rather than a frontend
    change.
    """
    body = client.get("/api/runs/api-run/template").json()
    assert body["name"] == "embodied"
    assert body["step_axis_label"] == "RL iteration"

    charted = {
        key
        for group in body["groups"]
        for chart in group["charts"]
        for key in chart["keys"]
    }
    assert "env/success_once" in charted
    # Absent from the event file, so its chart was dropped rather than left blank.
    assert "env/episode_len" not in charted
    # Logged but claimed by no group -- surfaced, not silently dropped.
    assert body["unmatched"] == ["my/custom/metric"]
    assert body["north_star"]["key"] == "env/success_once"
    assert body["north_star"]["resolved"] is True


def test_all_templates_are_listable_for_debugging_selection(client):
    names = {template["name"] for template in client.get("/api/templates").json()}
    assert {"embodied", "reasoning", "sft", "fallback"} <= names


# ----------------------------------------------------------------------------- media


def test_media_endpoints_serve_only_indexed_files(tmp_path, run_tree, settings_for):
    """The allowlist, exercised through HTTP.

    ``resolve`` has its own unit tests; this checks a refusal becomes a 404 rather
    than an exception escaping the route.
    """
    videos = tmp_path / "videos"
    videos.mkdir()
    clip = videos / "rollout.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    outsider = tmp_path / "secret.mp4"
    outsider.write_bytes(b"secret")

    run_tree(
        "media-api-run",
        snapshot=_snapshot(run_id="media-api-run"),
        media={0: [{"path": str(clip), "step": 2, "split": "train", "shard": 0}]},
    )
    with TestClient(create_app(settings_for())) as client:
        listing = client.get("/api/runs/media-api-run/media").json()
        assert len(listing) == 1
        assert client.get("/api/runs/media-api-run/media/steps").json() == [2]

        served = client.get(listing[0]["url"])
        assert served.status_code == 200
        assert served.headers["content-type"] == "video/mp4"

        refused = client.get(
            "/api/runs/media-api-run/media/file", params={"path": str(outsider)}
        )
        assert refused.status_code == 404


def test_media_filters_by_success_over_http(tmp_path, run_tree, settings_for):
    """``?success=`` reaches the service and the counts survive serialization.

    Checked through HTTP because the query parameter is a boolean: FastAPI has to
    coerce ``true``/``false`` from the string, and a wrong annotation would make
    every value truthy without any test at the service layer noticing.
    """
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("won", "lost"):
        (videos / f"{name}.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42fake")

    run_tree(
        "media-success-run",
        snapshot=_snapshot(run_id="media-success-run"),
        media={
            0: [
                {
                    "path": str(videos / "won.mp4"),
                    "step": 1,
                    "num_success": 1,
                    "num_envs": 4,
                },
                {
                    "path": str(videos / "lost.mp4"),
                    "step": 2,
                    "num_success": 0,
                    "num_envs": 4,
                },
            ]
        },
    )
    with TestClient(create_app(settings_for())) as client:
        base = "/api/runs/media-success-run/media"
        assert len(client.get(base).json()) == 2

        won = client.get(base, params={"success": "true"}).json()
        assert [row["step"] for row in won] == [1]
        assert won[0]["num_success"] == 1
        assert won[0]["num_envs"] == 4

        lost = client.get(base, params={"success": "false"}).json()
        assert [row["step"] for row in lost] == [2]


def test_media_supports_range_requests(tmp_path, run_tree, settings_for):
    """Seeking within a clip must not mean downloading it whole.

    ``FileResponse`` is used precisely for this; a plain byte response would make
    every scrub re-fetch the file.
    """
    videos = tmp_path / "videos"
    videos.mkdir()
    clip = videos / "long.mp4"
    clip.write_bytes(bytes(range(256)))

    run_tree(
        "range-run",
        snapshot=_snapshot(run_id="range-run"),
        media={0: [{"path": str(clip), "step": 1, "shard": 0}]},
    )
    with TestClient(create_app(settings_for())) as client:
        response = client.get(
            "/api/runs/range-run/media/file",
            params={"path": str(clip)},
            headers={"Range": "bytes=10-19"},
        )
        assert response.status_code == 206
        assert response.content == bytes(range(10, 20))


# ------------------------------------------------------------------------------- SSE
#
# These endpoints are exercised without the TestClient transport. An SSE route is
# an infinite generator that ends only when the client disconnects, and
# TestClient's synchronous portal waits for it to finish on close -- so even
# opening and immediately closing one hangs the test run.
#
# So the handlers are called directly for their response metadata, and `_sse_loop`
# is driven directly for its payloads. Driving the loop is also what makes the
# interesting assertions possible: how many pushes to take, and what changes on
# disk in between. Real transport behaviour is covered by the live-uvicorn smoke
# check in CI.


def _call_stream_handler(client, path: str, **path_params):
    """Invoke a stream route's handler and return its response object."""
    import asyncio

    endpoint = next(
        route.endpoint
        for route in client.app.routes
        if getattr(route, "path", None) == path
    )
    return asyncio.run(
        endpoint(request=object(), services=client.app.state.services, **path_params)
    )


def test_stream_endpoints_set_the_anti_buffering_headers(client):
    """SSE rather than WebSocket: a one-way push of a small document.

    A bidirectional protocol buys nothing here and costs reconnect logic, which
    ``EventSource`` provides for free. It does mean an intermediate proxy can
    buffer the stream, batching a 2s push into minute-long bursts -- which is what
    these headers prevent.
    """
    response = _call_stream_handler(client, "/api/stream/runs")
    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-accel-buffering"] == "no"


def test_streaming_an_unknown_run_fails_before_the_stream_opens(client):
    """The client must see a status code, not an error inside an accepted stream.

    An error delivered mid-stream is far harder for a browser to act on than a 404
    on the request itself.
    """
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as raised:
        _call_stream_handler(client, "/api/stream/runs/{run_id}", run_id="nope")
    assert raised.value.status_code == 404


def test_the_stream_declares_its_reconnect_delay(client):
    """Reconnect must be automatic *and* paced to this server.

    ``EventSource`` reconnects unprompted, so the protocol covers half of that --
    but the delay it picks unprompted is a hardcoded 3s that knows nothing about
    the configured interval. The ``retry:`` frame is what makes the two agree, and
    it must lead the stream: sent after the first update it would not apply to a
    connection that drops before then.
    """
    import asyncio

    from rlinf_dashboard.api import _sse_loop

    services = client.app.state.services
    services.settings.sse_interval_s = 7.5

    class _Request:
        async def is_disconnected(self):
            return False

    async def first_chunk():
        generator = _sse_loop(_Request(), services, lambda: {"ok": True})
        try:
            return await generator.__anext__()
        finally:
            await generator.aclose()

    assert asyncio.run(first_chunk()) == "retry: 7500\n\n"


def test_the_loop_pushes_the_rendered_payload(client):
    payloads = _drive_sse(client, lambda services: _render_summaries(services), count=1)
    assert payloads[0][0]["run_id"] == "api-run"
    assert payloads[0][0]["health"] == "healthy"


def test_the_loop_repushes_after_the_snapshot_changes(client, tmp_path):
    """A live run's step advances between pushes; the stream must show it.

    Re-rendering per tick rather than caching is the whole point -- a stream that
    pushed a constant document would be a slower page reload.
    """
    run_json = tmp_path / "logs" / "_rlinf" / "runs" / "api-run" / "run.json"

    def advance():
        run_json.write_text(
            json.dumps(_snapshot(progress={"step": 9, "max_steps": 10}))
        )

    payloads = _drive_sse(
        client,
        lambda services: _render_status(services, "api-run"),
        count=2,
        between=advance,
    )
    assert payloads[0]["snapshot"]["progress"]["step"] == 4
    assert payloads[1]["snapshot"]["progress"]["step"] == 9


def test_the_loop_survives_a_render_failure(client, tmp_path):
    """A transient read failure must not disconnect the UI.

    These files are written by a live process, so a hiccup is expected. Closing
    the stream would make the dashboard look disconnected when it is the data that
    stumbled. The store turns a corrupt snapshot into an ``error`` field rather
    than an exception, so the stream keeps pushing either way.
    """
    run_json = tmp_path / "logs" / "_rlinf" / "runs" / "api-run" / "run.json"

    payloads = _drive_sse(
        client,
        lambda services: _render_status(services, "api-run"),
        count=2,
        between=lambda: run_json.write_text("{truncated"),
    )
    assert payloads[0]["error"] is None
    assert payloads[1]["snapshot"] is None
    assert payloads[1]["error"]


def test_the_loop_emits_an_error_event_instead_of_dying(client):
    """A renderer that raises outright still must not end the stream."""
    boom = 0

    def render():
        nonlocal boom
        boom += 1
        if boom == 1:
            raise RuntimeError("nfs stat failed")
        return {"ok": True}

    events = _drive_sse(client, lambda services: render(), count=2, raw=True)
    assert events[0][0] == "error"
    assert "nfs stat failed" in events[0][1]["detail"]
    assert events[1] == ("update", {"ok": True})


def _render_summaries(services) -> list[dict]:
    return [
        summary.model_dump(mode="json")
        for summary in (
            services.store.summary(run) for run in services.discovery.list_runs()
        )
    ]


def _render_status(services, run_id: str) -> dict:
    run = services.discovery.find(run_id)
    return services.store.status(run).model_dump(mode="json")


def _drive_sse(client, render_with_services, *, count, between=None, raw=False):
    """Run ``_sse_loop`` for ``count`` pushes and return what it yielded.

    ``between`` runs after the first push, which is how a mid-stream change on
    disk is expressed. The sleep interval is zeroed so this costs no wall time.

    The leading ``retry:`` frame is dropped here rather than in each caller: it is
    a one-time reconnect hint carrying no payload, and ``test_the_stream_declares
    _its_reconnect_delay`` is what pins it.
    """
    import asyncio

    from rlinf_dashboard.api import _sse_loop

    services = client.app.state.services
    services.settings.sse_interval_s = 0.0

    class _Request:
        """Just enough of ``Request`` for the loop: it only asks about disconnect."""

        async def is_disconnected(self):
            return False

    async def collect():
        out = []
        generator = _sse_loop(
            _Request(), services, lambda: render_with_services(services)
        )
        try:
            async for chunk in generator:
                if chunk.startswith("retry:"):
                    continue
                out.append(chunk)
                if len(out) == 1 and between is not None:
                    between()
                if len(out) >= count:
                    break
        finally:
            await generator.aclose()
        return out

    chunks = asyncio.run(collect())
    parsed = [_parse_sse_chunk(chunk) for chunk in chunks]
    return parsed if raw else [payload for _, payload in parsed]


def _parse_sse_chunk(chunk: str) -> tuple[str, object]:
    """Split one ``event:``/``data:`` frame into its name and decoded payload."""
    event = "message"
    data = ""
    for line in chunk.strip().splitlines():
        if line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            data = line[len("data: ") :]
    return event, json.loads(data)


# --------------------------------------------------------------------- odd run trees


def test_a_run_with_no_snapshot_still_appears(run_tree, settings_for):
    """A run between its manifest and its first flush.

    It has to be listed -- a launch that never gets past startup is one of the
    things the dashboard exists to make visible, and dropping the row would hide
    it completely. Inside the grace period it is listed as starting up, which is
    a different claim from listing it as broken.
    """
    run_tree("never-published")
    with TestClient(create_app(settings_for())) as client:
        rows = client.get("/api/runs").json()
        assert [row["run_id"] for row in rows] == ["never-published"]
        assert rows[0]["health"] == "unknown"
        assert rows[0]["initializing"] is True

        body = client.get("/api/runs/never-published").json()
        assert body["snapshot"] is None
        assert body["initializing"] is True
        assert body["error"] is None
        # The page still needs a layout to render into.
        assert (
            client.get("/api/runs/never-published/template").json()["name"]
            == "embodied"
        )


def test_a_startup_past_its_deadline_is_reported_over_http(run_tree, settings_for):
    """The other side of the same state: still no snapshot, no longer excusable."""
    started = datetime.now(timezone.utc) - timedelta(seconds=900)
    run_tree(
        "stuck-start",
        manifest={"started_at": started.isoformat().replace("+00:00", "Z")},
    )
    with TestClient(create_app(settings_for(startup_grace_s=600.0))) as client:
        body = client.get("/api/runs/stuck-start").json()

        assert body["initializing"] is False
        assert body["error"]
        assert body["startup_elapsed_s"] > 600.0


def test_a_corrupt_snapshot_does_not_break_the_list(tmp_path, run_tree, settings_for):
    """One bad file must not take down the endpoint for every other run."""
    run_tree("good-run", snapshot=_snapshot(run_id="good-run"))
    run_tree("bad-run")
    bad = tmp_path / "logs" / "_rlinf" / "runs" / "bad-run" / "run.json"
    bad.write_text("}{not json")

    with TestClient(create_app(settings_for())) as client:
        rows = client.get("/api/runs").json()
        assert {row["run_id"] for row in rows} == {"good-run", "bad-run"}
        assert client.get("/api/runs/bad-run").status_code == 200


def test_a_dead_run_is_reported_unreachable_over_http(run_tree, settings_for):
    """The end-to-end version of the case ``state`` cannot express.

    ``kill -9`` leaves ``state: running`` on disk forever, because a dead process
    cannot record its own death. The verdict has to come from the reader, and this
    is the assertion that it survives all the way out to JSON.
    """
    run_tree(
        "dead-run",
        snapshot=_snapshot(
            run_id="dead-run", heartbeat_at=_iso(NOW - timedelta(hours=2))
        ),
    )
    with TestClient(create_app(settings_for())) as client:
        row = client.get("/api/runs?state=running").json()[0]
        assert row["state"] == "running"
        assert row["health"] == "unreachable"


def test_openapi_is_generated(client):
    """The schema is the frontend's contract; a broken route annotation breaks it."""
    body = client.get("/openapi.json").json()
    assert "/api/runs/{run_id}/series" in body["paths"]


def test_the_server_never_imports_rlinf(no_rlinf, run_tree, settings_for):
    """The architectural constraint, over the whole request path.

    Also enforced by an ``rlinf``-free venv in CI; having it here means a stray
    import fails on a laptop, where it is cheap to find.
    """
    run_tree("isolated-run", snapshot=_snapshot(run_id="isolated-run"))
    with TestClient(create_app(settings_for())) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/runs").json()[0]["run_id"] == "isolated-run"
        assert client.get("/api/runs/isolated-run/template").status_code == 200
        assert client.get("/api/runs/isolated-run/keys").status_code == 200


# ----------------------------------------------------------------------- frontend


def test_no_frontend_build_leaves_the_api_alone(tmp_path, settings_for):
    """The API must be fully usable without a Node toolchain.

    An operator reading run status with curl, or the isolation CI job, has no
    ``dist/``. Requiring a build to start the server would put a frontend
    toolchain between someone and their crashed run.
    """
    settings = settings_for(frontend_dist=str(tmp_path / "no-such-dist"))
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").status_code == 200
        # Nothing is mounted, so the root is a plain 404 rather than HTML.
        assert client.get("/").status_code == 404


def _write_dist(root):
    """A minimal build tree: what Vite emits, reduced to what routing needs."""
    dist = root / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>app</title>", "utf-8")
    (dist / "assets" / "index-abc123.js").write_text("console.log(1)", "utf-8")
    return dist


def test_a_built_frontend_is_served_from_the_api_origin(tmp_path, settings_for):
    """Same origin is the point: no CORS grant, and SSE is not cross-origin."""
    dist = _write_dist(tmp_path)
    settings = settings_for(frontend_dist=str(dist))
    with TestClient(create_app(settings)) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert "<title>app</title>" in index.text
        asset = client.get("/assets/index-abc123.js")
        assert asset.status_code == 200
        assert asset.text == "console.log(1)"


def test_a_deep_link_falls_back_to_index(tmp_path, settings_for):
    """``/runs/<id>`` is a real URL a user can paste or reload.

    Routing happens in the browser, so the server has to answer any non-API
    path with the app shell rather than a 404.
    """
    settings = settings_for(frontend_dist=str(_write_dist(tmp_path)))
    with TestClient(create_app(settings)) as client:
        response = client.get("/runs/20260804-083509-libero_10_ppo_openpi_pi05")
        assert response.status_code == 200
        assert "<title>app</title>" in response.text


def test_an_unknown_api_path_stays_a_json_404(tmp_path, settings_for):
    """The SPA catch-all must not swallow the API namespace.

    A registered route matches before the catch-all, but a mistyped or removed
    endpoint falls through to it. Answering that with 200 + HTML turns a clear
    404 into a JSON parse error in the client, which is a much harder bug to
    read from a browser console.
    """
    settings = settings_for(frontend_dist=str(_write_dist(tmp_path)))
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/does-not-exist")
        assert response.status_code == 404
        assert response.json()["detail"].startswith("Unknown endpoint")
        # A real route is unaffected.
        assert client.get("/api/health").status_code == 200


def test_the_frontend_route_cannot_escape_the_dist_directory(tmp_path, settings_for):
    """A path traversal must land on the app shell, never on a host file."""
    dist = _write_dist(tmp_path)
    (tmp_path / "secret.txt").write_text("do not serve", "utf-8")
    settings = settings_for(frontend_dist=str(dist))
    with TestClient(create_app(settings)) as client:
        response = client.get("/../secret.txt")
        assert "do not serve" not in response.text
