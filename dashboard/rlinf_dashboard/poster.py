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

"""Single-frame posters for recorded clips, so the media grid is not forty black boxes.

Why a server-side frame at all, when a browser can decode video on its own: the
clips are written by ``imageio.get_writer`` with no ``-movflags +faststart``, so
the ``moov`` atom lands at the *end* of the file -- measured at 98% in on a real
LIBERO clip (``ftyp@0 free@32 mdat@40 moov@713882``). A browser that wants one
frame must therefore probe the head, fetch the tail for the index, then come back
for the frame: three Range round-trips per clip, and some browsers give up and
pull the whole file. At forty cards against a six-connection origin limit, that
is exactly the queue that makes unrelated API calls hang.

One decoded frame is ~15KB against a ~1.1MB clip, so the grid gets 73x cheaper
per card while gaining a picture. The cost is one ffmpeg run per clip, once,
cached on disk thereafter.

The work is bounded on purpose. In the deployment this was measured against the
dashboard is its own container with a 2-core cgroup limit next to a 60-core
trainer, so ffmpeg here cannot reach the trainer's CPU budget at all -- but it
can still starve *this* process's own request handling, which is what the
semaphore and the thread cap are for.

Disk is bounded too. The cache holds a frame per clip forever otherwise, and a
long-lived dashboard sees runs indefinitely. ``poster_cache_max_mb`` caps it,
enforced oldest-first; the sweep is amortised against bytes written rather than
run per render, so the directory walk stays off the request path.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import threading
import time

from .settings import Settings

logger = logging.getLogger(__name__)

#: Extensions worth decoding a frame from. An image is already its own poster.
VIDEO_SUFFIXES = (".mp4", ".webm")

#: Age past which a staging file cannot belong to a live render and is treated
#: as debris from a killed process. Well clear of ``poster_timeout_s``.
_STAGING_DEBRIS_S = 3600.0

#: Fraction of the budget a sweep trims down to. Its complement is the sweep
#: interval, so the cache is bounded by the budget rather than by the budget
#: plus whatever accumulated since the last walk. The two must stay in step.
_LOW_WATER = 0.9


def _default_cache_dir() -> str:
    """Cache location when none is configured.

    Under the XDG cache dir rather than beside the clips: the run's log tree is
    the training job's output, and a reader must not write into it.
    """
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "rlinf-dashboard", "posters")


def find_ffmpeg(configured: str = "") -> str | None:
    """Locate an ffmpeg binary, or ``None`` if this deployment has none.

    Order: an explicit setting, then ``imageio-ffmpeg``'s bundled static build,
    then ``PATH``. The bundled build is checked before ``PATH`` because it is the
    same binary the training side writes clips with (``imageio.get_writer``), so
    decode behaviour matches the encoder by construction.

    Args:
        configured: Explicit path from settings. Empty means "search".

    Returns:
        A path to an executable ffmpeg, or ``None``.
    """
    if configured:
        return configured if os.path.isfile(configured) else None

    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:  # noqa: BLE001 - any import or probe failure means "not here"
        pass

    return shutil.which("ffmpeg")


class PosterUnavailable(Exception):
    """Raised when a poster cannot be produced right now, but might be later.

    Distinct from "this clip has no frame": a busy timeout is a 503 the client
    may retry, while an undecodable file is a 404 that will not change.
    """


class PosterService:
    """Decode one frame per clip, cache it on disk, and bound the concurrency."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._cache_dir = settings.poster_cache_dir or _default_cache_dir()
        self._ffmpeg = (
            find_ffmpeg(settings.ffmpeg_path) if settings.poster_enabled else None
        )
        # Bounds how many decodes run at once. `def` endpoints run in starlette's
        # threadpool, so without this a grid of forty cards would put forty
        # ffmpeg processes on a two-core budget and the API would stop answering.
        self._slots = threading.BoundedSemaphore(
            max(1, settings.poster_max_concurrency)
        )
        # Trimming walks the whole cache, so it is amortised against bytes
        # written rather than run per render. Starting the counter above the
        # threshold makes the first render after startup sweep once, which is
        # what reclaims a cache left oversized by a previous process or by a
        # budget that has since been lowered.
        self._sweep_lock = threading.Lock()
        self._bytes_since_sweep = self._sweep_every()
        # Latches the first "cannot write the cache" so the log carries one line
        # rather than one per card. Not a permanent disable: the condition can
        # clear, and retrying costs a failed `makedirs`.
        self._cache_unwritable = False
        if settings.poster_enabled and self._ffmpeg is None:
            logger.info(
                "No ffmpeg found; media posters are disabled and the grid will "
                "show placeholder cards. Install imageio-ffmpeg, or set "
                "RLINF_DASHBOARD_FFMPEG_PATH."
            )

    @property
    def available(self) -> bool:
        """Whether posters can be produced at all.

        Read by the media listing so a deployment with no ffmpeg reports no
        poster URL, rather than handing the browser forty URLs that all 404.
        """
        return self._ffmpeg is not None

    @property
    def ffmpeg_path(self) -> str | None:
        """The resolved binary, for ``/api/health`` to report."""
        return self._ffmpeg

    def cache_path_for(self, source: str) -> str:
        """Where this clip's poster is cached.

        Keyed by identity *and* content stamp (size, mtime), so a clip that gets
        overwritten in place -- a rerun writing the same step -- invalidates its
        poster instead of serving the previous run's frame forever.
        """
        try:
            stat = os.stat(source)
            stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            stamp = "0:0"
        digest = hashlib.sha256(
            f"{os.path.realpath(source)}|{stamp}|w{self._settings.poster_width}".encode()
        ).hexdigest()
        # Two-level fan-out: 1900 clips in one directory is fine, but a long-lived
        # cache across many runs is not, and a nested layout costs nothing now.
        return os.path.join(self._cache_dir, digest[:2], f"{digest}.jpg")

    def poster_for(self, source: str) -> str | None:
        """Return a cached poster path for a clip, rendering it if needed.

        Args:
            source: Absolute path to a clip, already allowlisted by
                :class:`~rlinf_dashboard.media.MediaService`.

        Returns:
            Path to a JPEG, or ``None`` if this file has no decodable frame.

        Raises:
            PosterUnavailable: The render slots were all busy. Retryable.
        """
        if not self.available:
            return None
        if not source.lower().endswith(VIDEO_SUFFIXES):
            return None

        cached = self.cache_path_for(source)
        if os.path.isfile(cached) and os.path.getsize(cached) > 0:
            return cached

        # A timed acquire rather than a blocking one. Holding a threadpool thread
        # indefinitely behind a full queue turns one slow grid into an
        # unresponsive server; failing fast lets the client fall back to the
        # placeholder card and try again on the next paint.
        if not self._slots.acquire(timeout=self._settings.poster_queue_timeout_s):
            raise PosterUnavailable(source)
        try:
            # Re-check: a request that queued behind the one which rendered this
            # exact clip should not render it a second time.
            if os.path.isfile(cached) and os.path.getsize(cached) > 0:
                return cached
            rendered = self._render(source, cached)
        finally:
            self._slots.release()

        # Outside the render slots on purpose: a directory walk holding one would
        # be time no decode can use, on a budget of two.
        if rendered is not None:
            self._account(os.path.getsize(rendered))
        return rendered

    def _sweep_every(self) -> int:
        """Bytes to write between cache sweeps.

        A tenth of the budget, which is what keeps the cache inside it rather
        than merely near it. Trimming stops at ``_LOW_WATER`` (90%), so the most
        that can accumulate before the next sweep is the remaining 10% -- the two
        fractions are complements on purpose, and changing one without the other
        lets the cache settle above its budget.

        Rare enough to stay off the render path: with a 100MB budget and ~15KB
        frames, one walk per ~680 renders.
        """
        limit = self._cache_limit_bytes()
        return max(1, int(limit * (1.0 - _LOW_WATER))) if limit else 0

    def _cache_limit_bytes(self) -> int:
        """The configured budget in bytes; ``0`` means unbounded."""
        return max(0, self._settings.poster_cache_max_mb) * 1024 * 1024

    def _account(self, written: int) -> None:
        """Record a write and sweep once enough has accumulated."""
        limit = self._cache_limit_bytes()
        if not limit:
            return

        self._bytes_since_sweep += written
        if self._bytes_since_sweep < self._sweep_every():
            return

        # Non-blocking: if another thread is already walking the cache, this one
        # has nothing to add by waiting to walk it again.
        if not self._sweep_lock.acquire(blocking=False):
            return
        try:
            self._bytes_since_sweep = 0
            self._trim(limit)
        except OSError as exc:  # noqa: BLE001 - a full cache must not fail a render
            logger.warning("Poster cache sweep failed: %s", exc)
        finally:
            self._sweep_lock.release()

    def _trim(self, limit: int) -> None:
        """Delete the oldest frames until the cache fits its budget.

        Oldest by mtime, which for these files is when they were rendered. That
        makes this FIFO rather than LRU: a frame is not kept alive by being
        looked at. The distinction costs little here, since the oldest frames
        belong to the least recently opened runs, and a wrong eviction is only
        ~75ms of re-rendering rather than lost data.
        """
        frames: list[tuple[int, int, str]] = []
        total = 0
        now = time.time()

        for root, _, names in os.walk(self._cache_dir):
            for name in names:
                path = os.path.join(root, name)
                try:
                    stat = os.stat(path)
                except OSError:
                    continue

                # Staging files belong to a render in flight; deleting one would
                # break it. Only those far older than the render timeout can be
                # debris from a process that was killed mid-write.
                if name.endswith(".tmp.jpg"):
                    if now - stat.st_mtime > _STAGING_DEBRIS_S:
                        _unlink(path)
                    continue

                frames.append((stat.st_mtime_ns, stat.st_size, path))
                total += stat.st_size

        if total <= limit:
            return

        # Down to a low-water mark rather than exactly to the limit, so the next
        # render does not immediately trip another walk. The headroom this leaves
        # is exactly one sweep interval; see `_sweep_every`.
        target = int(limit * _LOW_WATER)
        frames.sort()
        removed = 0
        for _, size, path in frames:
            if total <= target:
                break
            _unlink(path)
            total -= size
            removed += 1

        logger.info(
            "Trimmed %d poster(s) to keep the cache under %d MB",
            removed,
            self._settings.poster_cache_max_mb,
        )

    def _render(self, source: str, destination: str) -> str | None:
        """Decode one frame to ``destination``. ``None`` if the clip yields none."""
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
        except OSError as exc:
            # A read-only filesystem is a supported way to run this container --
            # the deployment docs recommend it -- and a full disk happens on its
            # own. Neither is a reason to fail a request: the grid falls back to
            # placeholder cards and the clips still play. Logged once, because
            # the alternative is one identical line per card.
            if not self._cache_unwritable:
                self._cache_unwritable = True
                logger.warning(
                    "Poster cache directory %s is not writable (%s); serving "
                    "placeholder cards. Point RLINF_DASHBOARD_POSTER_CACHE_DIR "
                    "at a writable path to enable previews.",
                    self._cache_dir,
                    exc,
                )
            return None

        # Seek past the opening frame: the first frame of a rollout is the
        # simulator's reset pose, which looks identical across every clip in a
        # run and so distinguishes nothing. A short clip may have nothing at that
        # timestamp, hence the no-seek retry below.
        rendered = self._run_ffmpeg(
            source, destination, seek=self._settings.poster_seek_s
        )
        if rendered is None:
            rendered = self._run_ffmpeg(source, destination, seek=None)
        return rendered

    def _run_ffmpeg(
        self, source: str, destination: str, seek: float | None
    ) -> str | None:
        """One ffmpeg invocation. Returns the path, or ``None`` on any failure."""
        assert self._ffmpeg is not None

        # Written to a private temp name and renamed, so a concurrent reader
        # never opens a half-written JPEG and a killed render leaves no partial
        # file to be cached as if it were complete.
        #
        # The name must still end in `.jpg`: ffmpeg picks the output muxer from
        # the extension, and a bare `.tmp` makes it fail with "Unable to find a
        # suitable output format" before decoding anything.
        staging = f"{destination}.{os.getpid()}.{threading.get_ident()}.tmp.jpg"

        command = [self._ffmpeg, "-nostdin", "-loglevel", "error"]
        if seek is not None:
            # Before `-i`, which makes it an input seek: ffmpeg jumps to the
            # nearest keyframe instead of decoding from the start. On the 4s
            # clips measured here it makes no difference, but it is what keeps
            # cost flat if clip length grows.
            command += ["-ss", str(seek)]
        command += [
            "-i",
            source,
            "-frames:v",
            "1",
            # `min(w,iw)` so a clip narrower than the target is never upscaled --
            # that would spend bytes to add no detail, and can make the "preview"
            # larger than the clip it previews.
            #
            # `-2` keeps the height even, which the JPEG encoder requires for
            # subsampled chroma; `-1` can produce an odd height and fail.
            "-vf",
            f"scale='min({self._settings.poster_width},iw)':-2",
            "-q:v",
            str(self._settings.poster_quality),
            # One thread per decode. The point is not speed -- a single frame is
            # ~75ms either way -- but leaving the other core for this process's
            # own request handling on a two-core budget.
            "-threads",
            "1",
            # Stated rather than inferred from the extension, so the staging
            # file's name is never load-bearing again.
            "-f",
            "image2",
            "-y",
            staging,
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=self._settings.poster_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Poster render timed out for %s", source)
            _unlink(staging)
            return None
        except OSError as exc:
            logger.warning("Poster render could not start for %s: %s", source, exc)
            _unlink(staging)
            return None

        if (
            result.returncode != 0
            or not os.path.isfile(staging)
            or os.path.getsize(staging) == 0
        ):
            if result.returncode != 0:
                logger.debug(
                    "ffmpeg failed on %s: %s",
                    source,
                    result.stderr.decode("utf-8", "replace")[-400:],
                )
            _unlink(staging)
            return None

        try:
            os.replace(staging, destination)
        except OSError as exc:
            logger.warning("Could not commit poster for %s: %s", source, exc)
            _unlink(staging)
            return None
        return destination


def _unlink(path: str) -> None:
    """Remove a file, ignoring the case where it was never created."""
    try:
        os.unlink(path)
    except OSError:
        pass
