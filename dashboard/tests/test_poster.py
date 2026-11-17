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

"""Poster rendering: the cache contract, the degradation path, and the boundary.

Two properties matter more than the picture itself.

*Degradation.* A deployment with no ffmpeg must stay a working dashboard. That
means no poster URLs in the listing (so the client never asks), a 404 rather than
a 500 if something asks anyway, and clips that still play.

*Cache identity.* Posters are served with a year-long ``immutable`` cache
directive, which is only safe because the URL carries the clip's size and mtime.
A clip rewritten in place -- a resumed run redoing a step -- must therefore
produce a different key, or every viewer keeps the previous run's frame.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import threading
import time

import pytest

from rlinf_dashboard.discovery import RunDiscovery
from rlinf_dashboard.media import MediaService
from rlinf_dashboard.poster import PosterService, PosterUnavailable, find_ffmpeg
from rlinf_dashboard.state import StateStore

#: A real encoder is needed to assert anything about actual frames. Everything
#: about degradation and cache identity is testable without one.
ffmpeg_required = pytest.mark.skipif(
    find_ffmpeg() is None, reason="no ffmpeg available to decode a frame"
)


def _make_clip(path: str, seconds: float = 1.0, size: int = 512) -> str:
    """Encode a real MP4 with the same writer shape the training side uses.

    Mirrors ``imageio.get_writer`` in that no ``+faststart`` is requested, so the
    ``moov`` atom lands at the end exactly as it does in a production clip.

    The default size is chosen to be wider than ``poster_width``, matching the
    1024x1024 tiled grids a real run writes. A source *narrower* than the poster
    is a different case with its own test.
    """
    ffmpeg = find_ffmpeg()
    assert ffmpeg is not None
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={size}x{size}:rate=30:duration={seconds}",
            "-pix_fmt",
            "yuv420p",
            "-y",
            path,
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture
def poster_run(tmp_path, run_tree, settings_for):
    """A run with one indexed clip, plus the services to read it."""
    videos = tmp_path / "videos"
    videos.mkdir()
    clip = str(videos / "step_1.mp4")

    run_tree(
        "poster-run",
        media={0: [{"path": clip, "step": 1, "split": "train", "num_envs": 1}]},
    )

    def build(**overrides):
        overrides.setdefault("poster_cache_dir", str(tmp_path / "cache"))
        settings = settings_for(**overrides)
        poster = PosterService(settings)
        media = MediaService(StateStore(settings), poster)
        run = RunDiscovery(settings).find("poster-run")
        return run, media, poster

    return build, clip


# --------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------


def test_listing_omits_poster_url_when_no_ffmpeg(poster_run):
    """No encoder means no URL, so the client never sends forty doomed requests."""
    build, clip = poster_run
    open(clip, "wb").write(b"\x00" * 64)
    run, media, poster = build(ffmpeg_path="/nonexistent/ffmpeg")

    assert poster.available is False
    entries = media.list_media(run)
    assert entries[0].poster_url is None
    # The clip itself is still playable: a missing thumbnail is cosmetic.
    assert entries[0].url is not None


def test_listing_omits_poster_url_when_disabled(poster_run):
    """`poster_enabled: false` is the same contract as a missing binary."""
    build, clip = poster_run
    open(clip, "wb").write(b"\x00" * 64)
    run, media, poster = build(poster_enabled=False)

    assert poster.available is False
    assert media.list_media(run)[0].poster_url is None


def test_undecodable_clip_yields_none_not_an_exception(poster_run):
    """A 64-byte stub is what `make_demo_runs.py` writes; it must not 500."""
    build, clip = poster_run
    open(clip, "wb").write(b"\x00" * 64)
    _, _, poster = build()

    if not poster.available:
        pytest.skip("no ffmpeg available")
    assert poster.poster_for(clip) is None


def test_no_staging_file_survives_a_failed_render(poster_run):
    """A half-written frame must never be left where a cache hit would find it."""
    build, clip = poster_run
    open(clip, "wb").write(b"\x00" * 64)
    _, _, poster = build()
    if not poster.available:
        pytest.skip("no ffmpeg available")

    poster.poster_for(clip)
    cache_root = os.path.dirname(os.path.dirname(poster.cache_path_for(clip)))
    leftovers = [
        name for _, _, files in os.walk(cache_root) for name in files if ".tmp" in name
    ]
    assert leftovers == []


@ffmpeg_required
def test_an_unwritable_cache_degrades_instead_of_failing(poster_run, tmp_path):
    """A read-only container is a documented deployment, not an error.

    `dashboard/Dockerfile` runs as a non-root user and the quickstart recommends
    `--read-only`, so the cache directory can legitimately be unwritable. The
    grid must fall back to placeholders and the clips must still play.
    """
    build, clip = poster_run
    _make_clip(clip)

    readonly = tmp_path / "readonly"
    readonly.mkdir()
    os.chmod(readonly, 0o500)
    _, _, poster = build(poster_cache_dir=str(readonly / "posters"))

    try:
        assert poster.poster_for(clip) is None, "should degrade, not raise"
    finally:
        os.chmod(readonly, 0o700)


def test_image_rows_get_no_poster(tmp_path, run_tree, settings_for):
    """A PNG is already its own preview; rendering one would be pure waste."""
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    run_tree("img-run", media={0: [{"path": str(image), "step": 1}]})

    settings = settings_for(poster_cache_dir=str(tmp_path / "cache"))
    media = MediaService(StateStore(settings), PosterService(settings))
    run = RunDiscovery(settings).find("img-run")

    assert media.list_media(run)[0].poster_url is None


# --------------------------------------------------------------------------
# Cache identity
# --------------------------------------------------------------------------


@ffmpeg_required
def test_rewriting_a_clip_in_place_changes_the_cache_key(poster_run):
    """The property that makes a one-day `immutable` response safe.

    A resumed run that redoes a step overwrites the clip at the same path. If the
    key did not move, every viewer would keep seeing the previous attempt's frame
    until the cache expired.
    """
    build, clip = poster_run
    _make_clip(clip)
    _, _, poster = build()

    before = poster.cache_path_for(clip)
    os.utime(clip, (0, 0))
    assert poster.cache_path_for(clip) != before


@ffmpeg_required
def test_poster_url_carries_the_content_stamp(poster_run):
    """The listing's URL must move with the file, or the cache directive lies."""
    build, clip = poster_run
    _make_clip(clip)
    run, media, _ = build()

    first = media.list_media(run)[0].poster_url
    assert first is not None and "&v=" in first

    os.utime(clip, (0, 0))
    assert media.list_media(run)[0].poster_url != first


@ffmpeg_required
def test_width_change_invalidates_the_cache(poster_run):
    """Rendered geometry is part of the artifact, so it is part of the key."""
    build, clip = poster_run
    _make_clip(clip)
    _, _, narrow = build(poster_width=320)
    _, _, wide = build(poster_width=640)

    assert narrow.cache_path_for(clip) != wide.cache_path_for(clip)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


@ffmpeg_required
def test_renders_a_real_jpeg_and_caches_it(poster_run):
    build, clip = poster_run
    _make_clip(clip)
    _, _, poster = build()

    first = poster.poster_for(clip)
    assert first is not None
    with open(first, "rb") as handle:
        assert handle.read(2) == b"\xff\xd8", "not a JPEG"

    # Second call is a cache hit: same path, and it does not re-render.
    stamp = os.stat(first).st_mtime_ns
    assert poster.poster_for(clip) == first
    assert os.stat(first).st_mtime_ns == stamp


@ffmpeg_required
def test_poster_is_smaller_than_the_clip(poster_run):
    """The entire point: a contact sheet must not cost what the videos cost.

    The assertion is weaker than the effect. On a real 1024x1024 LIBERO clip the
    measured ratio is ~73x (1.1MB -> 15KB), but a synthetic source cannot show
    that: ``testsrc`` is a near-static pattern, so h264's P-frames cost almost
    nothing and the whole clip is about the size of its one I-frame. Asserting a
    real ratio here would only be asserting a property of the fixture. What this
    still catches is the regression that matters -- a poster growing past the
    clip it stands in for.
    """
    build, clip = poster_run
    _make_clip(clip, seconds=2.0)
    _, _, poster = build()

    rendered = poster.poster_for(clip)
    assert os.path.getsize(rendered) < os.path.getsize(clip)


@ffmpeg_required
def test_a_clip_narrower_than_the_poster_is_not_upscaled(poster_run):
    """Never spend bytes adding no detail.

    Upscaling a small clip can make the "preview" larger than the thing it
    previews, which inverts the whole point of the grid.
    """
    build, clip = poster_run
    _make_clip(clip, seconds=2.0, size=64)
    _, _, poster = build(poster_width=320)

    rendered = poster.poster_for(clip)
    assert rendered is not None
    # No image library is a dependency here, so the JPEG's SOF0 frame header is
    # parsed directly for its dimensions.
    with open(rendered, "rb") as handle:
        blob = handle.read()
    marker = blob.index(b"\xff\xc0")
    height, width = struct.unpack(">HH", blob[marker + 5 : marker + 9])
    assert (width, height) == (64, 64), f"got {width}x{height}, expected no upscale"


@ffmpeg_required
def test_a_clip_shorter_than_the_seek_still_yields_a_frame(poster_run):
    """The no-seek retry. A 0.1s clip has nothing at 0.5s, but it has a frame."""
    build, clip = poster_run
    _make_clip(clip, seconds=0.1)
    _, _, poster = build(poster_seek_s=5.0)

    assert poster.poster_for(clip) is not None


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


def test_a_full_queue_raises_rather_than_blocking(poster_run):
    """A burst must fail fast, not pin starlette's threadpool.

    503 is retryable and the client falls back to a placeholder; a blocked thread
    would instead make the whole server stop answering.
    """
    build, clip = poster_run
    open(clip, "wb").write(b"\x00" * 64)
    _, _, poster = build(poster_max_concurrency=1, poster_queue_timeout_s=0.05)
    if not poster.available:
        pytest.skip("no ffmpeg available")

    held = threading.Event()
    release = threading.Event()

    def hog():
        poster._slots.acquire()
        held.set()
        release.wait(timeout=5)
        poster._slots.release()

    worker = threading.Thread(target=hog, daemon=True)
    worker.start()
    held.wait(timeout=5)
    try:
        with pytest.raises(PosterUnavailable):
            poster.poster_for(clip)
    finally:
        release.set()
        worker.join(timeout=5)


# --------------------------------------------------------------------------
# Cache trimming
# --------------------------------------------------------------------------


def _fill_cache(service, count, size, start_mtime=1_000_000):
    """Write ``count`` fake cached frames, oldest first, one second apart.

    Returns the paths in age order, so a test can name which ones an
    oldest-first policy is obliged to remove.
    """
    paths = []
    directory = os.path.join(service._cache_dir, "aa")
    os.makedirs(directory, exist_ok=True)
    for index in range(count):
        path = os.path.join(directory, f"frame{index:03d}.jpg")
        with open(path, "wb") as handle:
            handle.write(b"\x00" * size)
        stamp = start_mtime + index
        os.utime(path, (stamp, stamp))
        paths.append(path)
    return paths


def _cache_size(service):
    return sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, names in os.walk(service._cache_dir)
        for name in names
    )


def test_trim_deletes_oldest_first(poster_run):
    """The policy the user asked for: over budget, the oldest frames go."""
    _, _, poster = poster_run[0]()
    mb = 1024 * 1024
    # 12MB of frames against a 10MB budget.
    paths = _fill_cache(poster, count=12, size=mb)

    poster._trim(limit=10 * mb)

    survivors = [p for p in paths if os.path.exists(p)]
    removed = [p for p in paths if not os.path.exists(p)]
    # Trims to the 90% low-water mark (9MB), so three of twelve go.
    assert removed == paths[:3], "did not remove the three oldest"
    assert survivors == paths[3:], "removed something that was not oldest"


def test_trim_stops_at_a_low_water_mark(poster_run):
    """Trim below the limit, not to it, or the next render sweeps again."""
    _, _, poster = poster_run[0]()
    mb = 1024 * 1024
    _fill_cache(poster, count=20, size=mb)

    poster._trim(limit=10 * mb)

    assert _cache_size(poster) <= int(10 * mb * 0.9)


def test_trim_leaves_a_cache_within_budget_alone(poster_run):
    _, _, poster = poster_run[0]()
    mb = 1024 * 1024
    paths = _fill_cache(poster, count=5, size=mb)

    poster._trim(limit=10 * mb)

    assert all(os.path.exists(p) for p in paths)


def test_zero_budget_disables_trimming(poster_run):
    """An operator who wants an unbounded cache must be able to say so."""
    _, _, poster = poster_run[0](poster_cache_max_mb=0)
    paths = _fill_cache(poster, count=8, size=1024 * 1024)

    poster._account(written=999 * 1024 * 1024)

    assert all(os.path.exists(p) for p in paths), "trimmed despite a zero budget"


def test_sweeps_are_amortised_not_per_render(poster_run):
    """The directory walk must not sit on the render path.

    A sweep per render would turn a cold grid of forty cards into forty walks of
    a cache holding thousands of files.
    """
    _, _, poster = poster_run[0](poster_cache_max_mb=100)
    sweeps = []
    poster._trim = lambda limit: sweeps.append(limit)

    poster._bytes_since_sweep = 0
    for _ in range(50):
        poster._account(written=15 * 1024)  # a measured frame

    assert sweeps == [], "swept before accumulating a sweep interval"

    poster._account(written=poster._sweep_every())
    assert len(sweeps) == 1


@pytest.mark.parametrize("budget_mb", [1, 10, 100, 1000])
def test_the_cache_settles_inside_its_budget_not_above_it(poster_run, budget_mb):
    """Low-water mark plus sweep interval must not exceed the budget.

    Regression: a floor on the sweep interval made these independent, and a 1MB
    budget settled at 1.57MB in an end-to-end run -- trimming to 0.9MB, then
    accumulating a full 1MB floor before the next sweep. The two fractions are
    complements, and this is the property that says so.
    """
    _, _, poster = poster_run[0](poster_cache_max_mb=budget_mb)
    limit = budget_mb * 1024 * 1024

    high_water = int(limit * 0.9) + poster._sweep_every()

    assert high_water <= limit, (
        f"cache can reach {high_water} bytes against a {limit}-byte budget"
    )


def test_first_render_after_startup_sweeps(poster_run):
    """Reclaims a cache left oversized by a previous process or a lowered budget.

    Without this, a dashboard restarted with a smaller budget would keep the old
    cache forever, since nothing else ever looks at the directory.
    """
    _, _, poster = poster_run[0](poster_cache_max_mb=10)
    assert poster._bytes_since_sweep >= poster._sweep_every()


def test_trim_spares_a_render_in_flight(poster_run):
    """A staging file belongs to a live render; deleting it breaks that render."""
    _, _, poster = poster_run[0]()
    os.makedirs(poster._cache_dir, exist_ok=True)
    staging = os.path.join(poster._cache_dir, "abc.12345.678.tmp.jpg")
    with open(staging, "wb") as handle:
        handle.write(b"\x00" * 1024)

    poster._trim(limit=1)

    assert os.path.exists(staging)


def test_trim_removes_stale_staging_debris(poster_run):
    """A process killed mid-write leaves staging files nothing else collects."""
    _, _, poster = poster_run[0]()
    os.makedirs(poster._cache_dir, exist_ok=True)
    debris = os.path.join(poster._cache_dir, "old.1.2.tmp.jpg")
    with open(debris, "wb") as handle:
        handle.write(b"\x00" * 1024)
    old = time.time() - 7200
    os.utime(debris, (old, old))

    poster._trim(limit=100 * 1024 * 1024)

    assert not os.path.exists(debris)


@ffmpeg_required
def test_a_trimmed_poster_is_rerendered_on_demand(poster_run):
    """Eviction is a cost, not a loss: the frame comes back when asked for."""
    build, clip = poster_run
    _make_clip(clip)
    _, _, poster = build()

    first = poster.poster_for(clip)
    os.unlink(first)

    again = poster.poster_for(clip)
    assert again == first
    assert os.path.getsize(again) > 0


# --------------------------------------------------------------------------
# The HTTP surface
# --------------------------------------------------------------------------


@pytest.fixture
def poster_client(tmp_path, run_tree, settings_for):
    """A live app over a run with one real clip."""
    from fastapi.testclient import TestClient

    from rlinf_dashboard.api import create_app

    videos = tmp_path / "videos"
    videos.mkdir()
    clip = str(videos / "step_1.mp4")
    if find_ffmpeg() is not None:
        _make_clip(clip)
    else:
        open(clip, "wb").write(b"\x00" * 64)

    run_tree(
        "http-run",
        media={0: [{"path": clip, "step": 1, "split": "train", "num_envs": 1}]},
    )
    settings = settings_for(poster_cache_dir=str(tmp_path / "cache"))
    with TestClient(create_app(settings)) as test_client:
        yield test_client, clip


def test_poster_refuses_a_path_outside_the_index(poster_client, tmp_path):
    """The same allowlist as the clip endpoint. A poster grants no extra reach.

    Rendering reads a file, so an unindexed path must be refused here exactly as
    it is for streaming -- otherwise the feature would be a way around the
    boundary the media service exists to enforce.
    """
    client, _ = poster_client
    secret = tmp_path / "secret.mp4"
    secret.write_bytes(b"\x00" * 64)

    response = client.get(
        "/api/runs/http-run/media/poster", params={"path": str(secret)}
    )
    assert response.status_code == 404


def test_poster_404s_for_an_unknown_run(poster_client):
    client, clip = poster_client
    response = client.get("/api/runs/no-such-run/media/poster", params={"path": clip})
    assert response.status_code == 404


@ffmpeg_required
def test_poster_endpoint_serves_a_cacheable_jpeg(poster_client):
    client, clip = poster_client
    response = client.get("/api/runs/http-run/media/poster", params={"path": clip})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"
    # Safe only because the listing stamps the URL; see `_poster_url`.
    assert "immutable" in response.headers["cache-control"]


def test_undecodable_clip_is_a_404_not_a_500(poster_client):
    """A stub clip must degrade to a placeholder, never to an error page."""
    client, clip = poster_client
    if find_ffmpeg() is None:
        pytest.skip("no ffmpeg available")
    open(clip, "wb").write(b"\x00" * 64)

    response = client.get("/api/runs/http-run/media/poster", params={"path": clip})
    assert response.status_code == 404


def test_health_reports_poster_capability(poster_client):
    """Make "why is the grid all placeholders" answerable without guessing."""
    client, _ = poster_client
    posters = client.get("/api/health").json()["posters"]

    assert posters["enabled"] is True
    assert posters["available"] == (find_ffmpeg() is not None)


# --------------------------------------------------------------------------
# Binary discovery
# --------------------------------------------------------------------------


def test_explicit_path_is_not_second_guessed(tmp_path):
    """A configured path that does not exist is an error, not a reason to search.

    Silently falling back would make a typo look like it worked, with posters
    coming from a binary the operator did not choose.
    """
    assert find_ffmpeg(str(tmp_path / "nope")) is None


def test_explicit_path_wins(tmp_path):
    fake = tmp_path / "my-ffmpeg"
    fake.write_text("#!/bin/sh\n")
    assert find_ffmpeg(str(fake)) == str(fake)


def test_bundled_binary_is_preferred_over_path(monkeypatch, tmp_path):
    """The bundled static build wins over ``PATH``.

    It is the same binary the training side encodes with, so preferring it makes
    decode behaviour match the encoder rather than match whatever the host
    happens to have installed.
    """
    on_path = tmp_path / "path-ffmpeg"
    on_path.write_text("#!/bin/sh\n")
    bundled = tmp_path / "bundled-ffmpeg"
    bundled.write_text("#!/bin/sh\n")

    monkeypatch.setattr(shutil, "which", lambda _name: str(on_path))
    fake_module = type("M", (), {"get_ffmpeg_exe": staticmethod(lambda: str(bundled))})
    monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", fake_module)

    assert find_ffmpeg() == str(bundled)


def test_falls_back_to_path_when_bundled_is_absent(monkeypatch, tmp_path):
    """A host-installed ffmpeg is still usable when the wheel carries none."""
    on_path = tmp_path / "path-ffmpeg"
    on_path.write_text("#!/bin/sh\n")

    monkeypatch.setattr(shutil, "which", lambda _name: str(on_path))
    broken = type(
        "M",
        (),
        {"get_ffmpeg_exe": staticmethod(lambda: (_ for _ in ()).throw(RuntimeError()))},
    )
    monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", broken)

    assert find_ffmpeg() == str(on_path)
