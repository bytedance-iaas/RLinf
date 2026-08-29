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

"""HTTP surface: REST for queries, SSE for the live control plane.

SSE rather than WebSocket. The control plane is a one-way push of a small
document every couple of seconds; a bidirectional protocol buys nothing and costs
reconnect logic, since ``EventSource`` reconnects on its own.

FastAPI and uvicorn to match ``rlinf/workers/rollout/sglang_server``, so anyone
who has read that server can read this one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .auth import BasicAuthMiddleware
from .discovery import DiscoveredRun, RunDiscovery
from .media import MediaService, content_type_for
from .metrics import MetricGateway, worker_label
from .models import (
    CheckpointEntry,
    MediaEntry,
    RunStatus,
    RunSummary,
    Series,
)
from .poster import PosterService, PosterUnavailable
from .registry import TemplateRegistry, bind_keys
from .settings import Settings, get_settings
from .state import StateStore

logger = logging.getLogger(__name__)


class Services:
    """The server's long-lived objects.

    Bundled so a test can construct the whole read path against a temporary
    directory in one line, and so the caches inside discovery and the TensorBoard
    reader live as long as the process rather than as long as a request.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.discovery = RunDiscovery(settings)
        self.store = StateStore(settings)
        self.gateway = MetricGateway(settings)
        self.templates = TemplateRegistry()
        self.poster = PosterService(settings)
        self.media = MediaService(self.store, self.poster)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app.

    Args:
        settings: Override settings. Tests pass a temporary ``scan_root``.

    Returns:
        A configured :class:`FastAPI` instance.
    """
    settings = settings or get_settings()
    services = Services(settings)

    app = FastAPI(
        title="RLinf Dashboard",
        version=__version__,
        summary="Control-plane view of RLinf training runs.",
        description=(
            "Reads the run control plane written by rlinf.utils.run_state "
            "(<log_path>/_rlinf/runs/<run_id>/) and time series from "
            "TensorBoard event files. This service never imports rlinf: the "
            "contract is the filesystem layout frozen in "
            "rlinf_dashboard/schemas/run.v2.schema.json."
        ),
    )
    app.state.services = services

    if settings.auth_enabled:
        assert settings.auth_username is not None
        assert settings.auth_password is not None
        app.add_middleware(
            BasicAuthMiddleware,
            username=settings.auth_username,
            password=settings.auth_password,
            public_paths=("/healthz",),
        )
        _declare_basic_auth(app)

    # Added after BasicAuthMiddleware so Starlette places CORS on the outside:
    # browser preflight requests are answered without exposing protected data,
    # while the actual cross-origin request must still authenticate.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET"],
            allow_headers=["*"],
        )

    _register_routes(app)
    _mount_frontend(app, settings)
    return app


def _declare_basic_auth(app: FastAPI) -> None:
    """Describe middleware-enforced Basic Auth in the generated OpenAPI schema."""
    base_openapi = app.openapi

    def authenticated_openapi() -> dict:
        schema = base_openapi()
        security_schemes = schema.setdefault("components", {}).setdefault(
            "securitySchemes", {}
        )
        security_schemes["basicAuth"] = {"type": "http", "scheme": "basic"}
        schema["security"] = [{"basicAuth": []}]
        return schema

    app.openapi = authenticated_openapi


#: Where the built frontend lives, checked in order.
#:
#: A source checkout prefers ``frontend/dist`` so a local rebuild is visible.
#: Installed wheels fall back to the bundled ``static`` directory.
_DIST_CANDIDATES = (
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend",
        "dist",
    ),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"),
)


def _find_frontend_dist() -> str | None:
    """First candidate directory that holds a real build, or ``None``."""
    for candidate in _DIST_CANDIDATES:
        if os.path.isfile(os.path.join(candidate, "index.html")):
            return candidate
    return None


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    """Serve the built frontend from the same origin as the API, if it exists.

    Same origin matters: the browser then needs no CORS grant and the SSE stream
    is not a cross-origin request, so a deployment behaves like the dev server
    with its ``/api`` proxy. Registered after the API routes so nothing here can
    shadow them.

    A missing ``dist/`` is normal -- the server is useful without a UI (curl,
    scripts, the OpenAPI page), and requiring a Node build to start it would put
    a toolchain between an operator and their run status. Mounted only when the
    build is present, which is also what keeps the isolation CI job (no rlinf,
    no npm) passing.
    """
    dist = settings.frontend_dist or _find_frontend_dist()
    index = os.path.join(dist, "index.html") if dist else ""
    if not index or not os.path.isfile(index):
        logger.debug("No frontend build found; serving the API only.")
        return

    from fastapi.staticfiles import StaticFiles

    # Fingerprinted assets: safe to cache hard, and served before the catch-all.
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(dist, "assets")),
        name="assets",
    )

    @app.get("/", include_in_schema=False)
    def _index() -> FileResponse:
        return FileResponse(index)

    @app.get("/{path:path}", include_in_schema=False)
    def _spa(path: str) -> FileResponse:
        """Return ``index.html`` for client-side routes.

        The app routes in the browser, so a deep link like ``/runs/<id>`` is a
        real URL a user can paste or reload.

        ``/api`` is excluded explicitly. A registered API route matches before
        this one, but an *unregistered* one falls through to here, and answering
        a mistyped endpoint with 200 + HTML turns a clear 404 into a client-side
        JSON parse error -- the API must fail as an API.
        """
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail=f"Unknown endpoint: /{path}")
        candidate = os.path.normpath(os.path.join(dist, path))
        if (
            path
            and os.path.isfile(candidate)
            and os.path.commonpath([candidate, dist]) == dist
        ):
            return FileResponse(candidate)
        return FileResponse(index)


def get_services(request: Request) -> Services:
    """FastAPI dependency: the per-process service bundle built by ``create_app``."""
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def _scan_root_state(services: Services, run_count: int) -> dict:
    """The scan root as the browser needs to see it.

    One shape, built once, returned by both ``/api/health`` and the endpoint
    that changes it -- the page renders the answer to a change with the same
    code that renders the state it polled, so the two can never disagree.

    ``run_count`` is a parameter rather than read here because the two callers
    differ on whether a cached count is acceptable.
    """
    discovery = services.discovery
    root = discovery.root
    return {
        "path": root,
        "exists": os.path.isdir(os.path.expanduser(root)),
        "run_count": run_count,
        # What the reset control returns to, and whether there is anything to
        # reset. Sent as data so the page never has to guess which of the two
        # it is looking at.
        "default_path": discovery.default_root,
        "is_default": discovery.is_default_root,
        "editable": services.settings.scan_root_editable,
    }


class ScanRootUpdate(BaseModel):
    """A directory to scan, or ``None`` to go back to the configured one."""

    path: str | None = Field(
        default=None,
        description=(
            "Directory on the server to scan for runs. Null or empty returns to "
            "the root this server was started with."
        ),
    )


def _require_run(services: Services, run_id: str) -> DiscoveredRun:
    run = services.discovery.find(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}")
    return run


def _register_routes(app: FastAPI) -> None:  # noqa: C901 - one function per route
    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        """Report process liveness without scanning or exposing run metadata."""
        return {"status": "ok"}

    @app.get("/api/health", summary="Liveness of this server")
    def health(services: ServicesDep) -> dict:
        """Report the server's own health, not any run's.

        Includes whether the scan root exists and how many runs are under it: a
        dashboard showing zero runs is almost always a mistyped path or a root
        one level off, and those two facts separate the two cases at a glance.
        """
        run_count = len(services.discovery.list_runs())
        return {
            "status": "ok",
            "version": __version__,
            "scan_root": _scan_root_state(services, run_count),
            "run_count": run_count,
            # "Why is the media grid all placeholders" is otherwise a silent
            # condition with no way to tell a disabled feature from a missing
            # binary.
            "posters": {
                "enabled": services.settings.poster_enabled,
                "available": services.poster.available,
                "ffmpeg": services.poster.ffmpeg_path,
            },
        }

    @app.put("/api/scan-root", summary="Scan a different directory")
    def set_scan_root(update: ScanRootUpdate, services: ServicesDep) -> dict:
        """Repoint discovery, or return it to the configured default.

        Process-global and in memory: the change is what every viewer of this
        server sees, and a restart forgets it. See
        :meth:`RunDiscovery.set_root`.

        A path that is not a directory is refused rather than accepted. The
        alternative -- taking it and letting the run list go empty -- answers a
        typo by discarding the root that was working, and the page cannot tell
        the operator which of the two things happened.
        """
        if not services.settings.scan_root_editable:
            raise HTTPException(
                status_code=403,
                detail=(
                    "This server does not allow the scan root to be changed "
                    "from the browser."
                ),
            )

        wanted = (update.path or "").strip()
        if wanted and not os.path.isdir(os.path.expanduser(wanted)):
            raise HTTPException(
                status_code=400,
                detail=f"No such directory on the server: {wanted}",
            )

        services.discovery.set_root(wanted or None)
        # Refreshed rather than cached: this response is the answer to "did it
        # work", and a count carried over from the previous root is not one.
        return _scan_root_state(
            services, len(services.discovery.list_runs(refresh=True))
        )

    @app.get("/api/runs", summary="List discovered runs")
    def list_runs(
        services: ServicesDep,
        refresh: bool = Query(False, description="Bypass the discovery cache"),
        state: str | None = Query(None, description="Filter by lifecycle state"),
        task_type: str | None = Query(None, description="Filter by task type"),
    ) -> list[RunSummary]:
        now = datetime.now(timezone.utc)
        runs = services.discovery.list_runs(refresh=refresh)
        summaries = [services.store.summary(run, now) for run in runs]
        if state:
            summaries = [s for s in summaries if s.state and s.state.value == state]
        if task_type:
            summaries = [s for s in summaries if s.task_type == task_type]
        return summaries

    @app.get("/api/runs/{run_id}", summary="Full status for one run")
    def get_run(run_id: str, services: ServicesDep) -> RunStatus:
        run = _require_run(services, run_id)
        return services.store.status(run)

    @app.get("/api/runs/{run_id}/events", summary="Lifecycle and phase events")
    def get_events(
        run_id: str,
        services: ServicesDep,
        limit: int = Query(200, ge=1, le=5000),
    ) -> list[dict]:
        run = _require_run(services, run_id)
        return [
            {
                "ts": event.ts,
                "kind": event.kind,
                "step": event.step,
                "payload": event.payload,
            }
            for event in services.store.read_events(run.run_root, limit=limit)
        ]

    @app.get("/api/runs/{run_id}/checkpoints", summary="Completed checkpoints")
    def get_checkpoints(run_id: str, services: ServicesDep) -> list[CheckpointEntry]:
        run = _require_run(services, run_id)
        return services.store.read_checkpoints(run.run_root)

    @app.get("/api/runs/{run_id}/keys", summary="Metric keys available for a run")
    def get_keys(run_id: str, services: ServicesDep) -> dict:
        run = _require_run(services, run_id)
        keys = services.gateway.list_keys(run.manifest)
        return {
            "keys": keys,
            "sources": [s.name for s in services.gateway.sources_for(run.manifest)],
            # Empty unless the run set `runner.per_worker_log: true`. The UI uses
            # this to decide whether to offer a per-rank drill-down at all, rather
            # than offering a control that would come back empty.
            "workers": services.gateway.workers(run.manifest),
        }

    @app.get("/api/runs/{run_id}/series", summary="Metric history")
    def get_series(
        run_id: str,
        services: ServicesDep,
        keys: Annotated[list[str], Query(description="Metric keys; repeatable")],
        expand: str | None = Query(
            None,
            pattern="^ranks$",
            description=(
                "`ranks` adds one series per (worker group, rank) beside each "
                "aggregate, for runs with `runner.per_worker_log: true`."
            ),
        ),
    ) -> dict[str, Series]:
        """Read series, optionally broken out per worker rank.

        The aggregate series keeps its own key, and each per-rank series is keyed
        ``"<key>@<group>/rank_<n>"``. One flat map rather than a nested document
        so the unexpanded response shape is unchanged and a caller that ignores
        `expand` needs no migration; the `group`/`rank` fields on each series are
        the machine-readable form of that suffix.
        """
        run = _require_run(services, run_id)
        result = services.gateway.read(run.manifest, keys)
        if expand != "ranks":
            return result

        for key, series_list in services.gateway.read_workers(
            run.manifest, keys
        ).items():
            for series in series_list:
                if series.group is None or series.rank is None:
                    continue
                label = worker_label(series.group, series.rank)
                result[f"{key}@{label}"] = series.model_copy(
                    update={"key": f"{key}@{label}"}
                )
        return result

    @app.get("/api/runs/{run_id}/template", summary="Chart layout for a run")
    def get_run_template(run_id: str, services: ServicesDep) -> dict:
        """Return the run's template with only the charts it has data for.

        Binding happens server-side so the frontend renders whatever it is given
        and never has to know which metrics a task type emits.
        """
        run = _require_run(services, run_id)
        snapshot, _ = services.store.read_snapshot(run.run_root)
        template = services.templates.select_for(run.manifest, snapshot)
        available = services.gateway.list_keys(run.manifest)
        return bind_keys(template, available, run.manifest.metric_aliases)

    @app.get("/api/templates", summary="All templates")
    def list_templates(services: ServicesDep) -> list[dict]:
        return services.templates.all()

    @app.get("/api/runs/{run_id}/media", summary="Recorded videos")
    def get_media(
        run_id: str,
        services: ServicesDep,
        split: str | None = Query(None, pattern="^(train|eval)$"),
        step: int | None = Query(None, ge=0),
        min_step: int | None = Query(None, ge=0),
        max_step: int | None = Query(None, ge=0),
        success: bool | None = Query(
            None,
            description=(
                "Keep only clips where at least one env succeeded (true) or none "
                "did (false). Clips with no recorded outcome are excluded either "
                "way rather than assumed to be failures."
            ),
        ),
        limit: int = Query(200, ge=1, le=5000),
    ) -> list[MediaEntry]:
        run = _require_run(services, run_id)
        return services.media.list_media(
            run,
            split=split,
            step=step,
            min_step=min_step,
            max_step=max_step,
            success=success,
            limit=limit,
        )

    @app.get("/api/runs/{run_id}/media/steps", summary="Steps that have media")
    def get_media_steps(run_id: str, services: ServicesDep) -> list[int]:
        run = _require_run(services, run_id)
        return services.media.steps(run)

    @app.get("/api/runs/{run_id}/media/file", summary="Stream one video")
    def get_media_file(run_id: str, path: str, services: ServicesDep) -> FileResponse:
        """Serve a video, but only one this run's own index lists.

        ``FileResponse`` handles ``Range`` requests, which is what lets a browser
        seek within a clip instead of downloading it whole.
        """
        run = _require_run(services, run_id)
        resolved = services.media.resolve(run, path)
        if resolved is None:
            raise HTTPException(
                status_code=404,
                detail="No such media file in this run's index.",
            )
        return FileResponse(
            resolved,
            media_type=content_type_for(resolved),
            filename=os.path.basename(resolved),
        )

    @app.get("/api/runs/{run_id}/media/poster", summary="One frame from a clip")
    def get_media_poster(
        run_id: str,
        path: str,
        services: ServicesDep,
        v: str | None = Query(
            None,
            description=(
                "Content stamp for cache busting. Ignored by the server -- it "
                "exists so a clip rewritten in place gets a different URL and "
                "cannot be served from a browser cache as a stale frame."
            ),
        ),
    ) -> FileResponse:
        """Serve a cached single-frame poster, rendering it on first request.

        Goes through the same allowlist as the clip itself: a poster can only be
        made from a path this run's own index lists, so this adds no reach beyond
        what ``/media/file`` already grants.

        A clip with no decodable frame is a 404 and stays one -- the client then
        shows its placeholder. A busy render queue is a 503, which is retryable
        and deliberately not cached.
        """
        run = _require_run(services, run_id)
        resolved = services.media.resolve(run, path)
        if resolved is None:
            raise HTTPException(
                status_code=404,
                detail="No such media file in this run's index.",
            )
        try:
            poster = services.poster.poster_for(resolved)
        except PosterUnavailable:
            raise HTTPException(
                status_code=503,
                detail="Poster renderer is busy. Retry shortly.",
                headers={"Retry-After": "2"},
            ) from None
        if poster is None:
            raise HTTPException(
                status_code=404,
                detail="No poster frame could be decoded from this clip.",
            )
        return FileResponse(
            poster,
            media_type="image/jpeg",
            # Safe to cache hard only because the listing stamps each URL with the
            # clip's size and mtime: a clip rewritten in place yields a different
            # `v`, hence a different URL, so a stale frame is unreachable rather
            # than merely unlikely.
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )

    @app.get("/api/stream/runs", summary="SSE stream of all run summaries")
    async def stream_runs(request: Request, services: ServicesDep) -> StreamingResponse:
        async def generate():
            async for chunk in _sse_loop(
                request,
                services,
                lambda: [
                    s.model_dump(mode="json")
                    for s in (
                        services.store.summary(run)
                        for run in services.discovery.list_runs()
                    )
                ],
            ):
                yield chunk

        return StreamingResponse(
            generate(), media_type="text/event-stream", headers=_SSE_HEADERS
        )

    @app.get("/api/stream/runs/{run_id}", summary="SSE stream for one run")
    async def stream_run(
        run_id: str, request: Request, services: ServicesDep
    ) -> StreamingResponse:
        run = _require_run(services, run_id)

        async def generate():
            async for chunk in _sse_loop(
                request,
                services,
                lambda: services.store.status(run).model_dump(mode="json"),
            ):
                yield chunk

        return StreamingResponse(
            generate(), media_type="text/event-stream", headers=_SSE_HEADERS
        )


#: `no-store` and `X-Accel-Buffering: no` keep an intermediate proxy from
#: buffering the stream, which would make a 2s push arrive in minute-long
#: batches and defeat the point of streaming.
_SSE_HEADERS = {
    "Cache-Control": "no-store",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def _sse_loop(request: Request, services: Services, render):
    """Push ``render()`` output until the client goes away.

    Each payload is read from disk in a worker thread: these are blocking
    ``open`` calls, and doing them on the event loop would stall every other
    connection for the duration of an NFS stat.

    A rendering failure emits an ``error`` event and keeps the stream open. The
    files being read are written by a live process, so a transient failure is
    expected; closing the stream would make the UI look disconnected when it is
    the data that hiccuped.
    """
    interval = services.settings.sse_interval_s
    # Reconnection is the browser's job -- `EventSource` does it unprompted -- but
    # the delay it picks is a hardcoded 3s that knows nothing about this server's
    # cadence. Sending `retry:` ties the two together: a dashboard pushing every
    # 10s stops reconnecting three times too eagerly, and one pushing twice a
    # second stops waiting six intervals to notice a restart.
    yield f"retry: {int(interval * 1000)}\n\n"
    while True:
        if await request.is_disconnected():
            return
        try:
            payload = await asyncio.to_thread(render)
            yield _sse_event("update", payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SSE render failed: %s", exc)
            yield _sse_event("error", {"detail": str(exc)})
        await asyncio.sleep(interval)


def _sse_event(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


app = None


def get_app() -> FastAPI:
    """Lazily construct the module-level app for ``uvicorn`` string targets.

    Lazy so that importing this module does not read the environment; a test that
    imports it to call :func:`create_app` with its own settings must not have
    already built an app from whatever was in the ambient env.
    """
    global app
    if app is None:
        app = create_app()
    return app
