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

"""Server configuration.

Every value is overridable by an ``RLINF_DASHBOARD_``-prefixed environment
variable, so the same wheel runs against a laptop's ``./logs`` and against an
NFS mount without a config file.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Where to look for runs and how patient to be about silence.

    Timeout multipliers are read-side policy. Progress staleness scales with the
    run's observed step time; process liveness scales with the writer's fixed
    heartbeat interval.
    """

    model_config = SettingsConfigDict(
        env_prefix="RLINF_DASHBOARD_",
        env_file=".env",
        extra="ignore",
    )

    #: The directory to scan: a ``runner.logger.log_path``, or any ancestor of
    #: several, searched for ``_rlinf/runs/*/manifest.json``.
    #:
    #: One root, not a list. Several local directories was never the shape of the
    #: real problem -- runs that belong together already share an ancestor, and
    #: the ones that do not are usually on another machine, which a second local
    #: path cannot reach anyway. Gathering runs across nodes is a separate design
    #: with its own questions about identity and freshness; leaving a list here in
    #: the meantime only invited half of it.
    scan_root: str = "./logs"

    #: Whether the scan root may be repointed at runtime from the browser.
    #:
    #: On by default, because the dashboard is an operator tool and the operator
    #: is the person who chose the root in the first place. Turn it off for a
    #: deployment where the reader is not the owner: with it on, anyone who can
    #: reach this server can point it at any directory the process can read and
    #: enumerate the RLinf run trees under it. That is bounded -- discovery only
    #: reports directories holding ``_rlinf/runs/*/manifest.json``, and media is
    #: still served only for files a run's own index lists -- but it does move
    #: the choice of root from whoever starts the container to whoever opens the
    #: page, so it belongs behind authentication on a shared host.
    scan_root_editable: bool = True

    #: How deep below a scan root to look for the ``_rlinf`` marker. Bounded
    #: because a scan root on NFS can be arbitrarily large and an unbounded walk
    #: would make ``/runs`` latency depend on unrelated data.
    scan_max_depth: int = 6

    #: Missed heartbeats tolerated before a run is called unreachable. Multiplies
    #: ``heartbeat_interval_s``, *not* step time: the heartbeat is a fixed-period
    #: tick that does not slow down when a step does.
    heartbeat_timeout_k: float = 5.0

    #: Multiples of ``max(step_time_p50, timeout_floor_s)`` before a live process
    #: with no step advance is called degraded. This one *should* scale with step
    #: time -- one step is how long a working run may go without advancing.
    #: Kept identical to the values in
    #: ``tests/unit_tests/test_run_state_contract.py``; the shared fixtures fail
    #: if the two drift.
    progress_timeout_k: float = 10.0

    #: Lower bound on the *step-time* budget, for runs with no step-time samples
    #: yet (startup can take minutes: sglang warmup, simulator boot).
    timeout_floor_s: float = 30.0

    #: Fallback heartbeat period for manifests written before the producer began
    #: recording its configured interval.
    heartbeat_interval_s: float = 5.0

    #: How long a run may hold a manifest without a ``run.json`` before that
    #: counts as a stuck startup rather than a normal one.
    #:
    #: The producer writes ``manifest.json`` when the reporter is constructed and
    #: ``run.json`` when the training loop enters ``run_lifecycle``; everything
    #: between -- Ray cluster boot, worker allocation, model load, simulator
    #: warmup -- happens in that window, so on a VLA job it is minutes, not
    #: milliseconds. Generous on purpose: waiting too long only delays an alarm,
    #: while waiting too little reports every normal startup as a broken run.
    startup_grace_s: float = 600.0

    #: SSE push period. The training side heartbeats every 5s by default, so
    #: polling faster than this only costs syscalls -- which is why this matches
    #: it rather than beating it. A two-second push also made the header's "as
    #: of" readout restless for no new information.
    sse_interval_s: float = 5.0

    #: Discovery result cache TTL. A scan stats every candidate manifest, which
    #: is the expensive part of ``/runs`` on a large NFS tree.
    discovery_cache_ttl_s: float = 5.0

    #: Max points returned per series. Larger series use an extrema-preserving
    #: envelope so spikes remain visible.
    max_series_points: int = 4000

    #: CORS origins for the Vite dev server. Empty in production, where the
    #: frontend is served from this same origin.
    #:
    #: ``NoDecode`` because pydantic-settings JSON-decodes list-typed fields
    #: *before* validators run, so without it a plain
    #: ``RLINF_DASHBOARD_CORS_ORIGINS=http://localhost:5273`` raises a
    #: ``SettingsError`` at startup instead of reaching ``_split_csv``.
    #:
    #: The port must match ``server.port`` in
    #: ``frontend/vite.config.ts``; a mismatch shows up as an empty dashboard
    #: with CORS errors only in the browser console.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5273", "http://127.0.0.1:5273"]
    )

    #: Where the built frontend lives. Empty means ``../frontend/dist`` relative
    #: to this package, which is where ``npm run build`` puts it. A missing build
    #: is not an error: the API alone is useful, and requiring a Node toolchain
    #: to read run status would be a worse default.
    frontend_dist: str = ""

    #: Decode one frame per clip to use as a poster in the media grid.
    #:
    #: Off turns the grid back into placeholder cards that load nothing until
    #: clicked. On but with no ffmpeg present is the same thing: the feature
    #: degrades rather than erroring, because a dashboard that cannot draw a
    #: thumbnail is still a working dashboard.
    poster_enabled: bool = True

    #: Explicit ffmpeg path. Empty searches ``imageio-ffmpeg``'s bundled static
    #: build first (the same binary the training side encodes with), then
    #: ``PATH``.
    ffmpeg_path: str = ""

    #: Where rendered posters are cached. Empty means
    #: ``$XDG_CACHE_HOME/rlinf-dashboard/posters``.
    #:
    #: Deliberately not beside the clips: the run's log tree belongs to the
    #: training job, and this service is read-only with respect to it.
    poster_cache_dir: str = ""

    #: Concurrent ffmpeg renders. The measured cost is ~75ms of one core per
    #: clip, so the ceiling exists to protect this process's own responsiveness
    #: on a small CPU budget, not to protect the trainer -- in a sidecar
    #: deployment the cgroup already does that.
    poster_max_concurrency: int = 2

    #: How long a request waits for a render slot before giving up with 503.
    #: Bounded so a burst cannot pin starlette's threadpool.
    poster_queue_timeout_s: float = 5.0

    #: Wall-clock ceiling on one ffmpeg run.
    poster_timeout_s: float = 20.0

    #: Timestamp to grab the frame from. Past zero because frame zero is the
    #: simulator's reset pose, identical across every clip in a run.
    poster_seek_s: float = 0.5

    #: Poster width in pixels; height follows the source aspect ratio. 320 puts a
    #: 1024x1024 tiled grid at ~15KB, against ~1.1MB for the clip itself.
    poster_width: int = 320

    #: Disk budget for the poster cache, in MB. ``0`` disables trimming.
    #:
    #: Enforced by deleting the oldest frames first, by modification time, which
    #: is when each was rendered. Deletion is not loss: a trimmed frame is
    #: re-rendered in ~75ms the next time its card is shown, so the budget trades
    #: a little repeated work for a bounded directory.
    #:
    #: 100MB holds roughly 6800 frames at the measured ~15KB each -- several
    #: times more clips than a single run produces, so a normal working set is
    #: never trimmed at all.
    poster_cache_max_mb: int = 100

    #: JPEG quality scale, ffmpeg's ``-q:v``: 2 is best, 31 is worst.
    poster_quality: int = 6

    #: Authentication is deliberately explicit: Helm sets ``basic`` so missing
    #: or misspelled Secret env vars fail closed, while source development keeps
    #: its backwards-compatible ``disabled`` default.
    auth_mode: Literal["disabled", "basic"] = "disabled"

    #: Static HTTP Basic credentials. Both values are required in ``basic`` mode
    #: and forbidden in ``disabled`` mode, so a partially applied deployment
    #: cannot silently choose its own security posture.
    #: The password is secret-typed so settings reprs and validation diagnostics
    #: cannot accidentally print it.
    auth_username: str | None = None
    auth_password: SecretStr | None = None

    @model_validator(mode="after")
    def _validate_auth_pair(self) -> "Settings":
        """Require one complete, unambiguous Basic Auth credential pair."""
        username = self.auth_username
        password = (
            self.auth_password.get_secret_value()
            if self.auth_password is not None
            else None
        )
        if self.auth_mode == "disabled":
            if username is not None or password is not None:
                raise ValueError(
                    "Set RLINF_DASHBOARD_AUTH_MODE=basic when providing "
                    "dashboard auth credentials."
                )
            return self
        if username is None or password is None:
            raise ValueError(
                "RLINF_DASHBOARD_AUTH_MODE=basic requires both "
                "RLINF_DASHBOARD_AUTH_USERNAME and "
                "RLINF_DASHBOARD_AUTH_PASSWORD."
            )
        if not username.strip() or not password.strip():
            raise ValueError("Dashboard auth username and password must not be blank.")
        if ":" in username:
            raise ValueError("Dashboard auth username must not contain ':'.")
        return self

    @property
    def auth_enabled(self) -> bool:
        """Whether the validated static credential pair is configured."""
        return self.auth_mode == "basic"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value):
        """Accept a comma-separated string, which is all an env var can carry.

        A JSON array is still accepted, since that is what a reader who knows
        pydantic-settings will reach for first.
        """
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                try:
                    decoded = json.loads(text)
                except ValueError:
                    pass
                else:
                    if isinstance(decoded, list):
                        return [str(item) for item in decoded]
            return [item.strip() for item in text.split(",") if item.strip()]
        return value

    @field_validator("scan_root", mode="before")
    @classmethod
    def _reject_a_list(cls, value):
        """Reject multiple roots instead of scanning only part of the request.

        Silently ignoring a root would leave the operator with missing runs that
        look like a discovery failure.
        """
        if isinstance(value, (list, tuple)):
            raise ValueError(
                "scan_root takes a single directory. Point it at the common "
                f"ancestor of {', '.join(str(item) for item in value)} instead."
            )
        if isinstance(value, str) and "," in value:
            raise ValueError(
                "scan_root takes a single directory, not a comma-separated list. "
                "Point it at the common ancestor of those paths instead."
            )
        return value

    @field_validator("scan_root")
    @classmethod
    def _no_empty_scan_root(cls, value: str) -> str:
        """Fall back to the default rather than scanning nothing.

        ``export RLINF_DASHBOARD_SCAN_ROOT=`` is easy to leave in a shell
        profile, and an empty value makes the server report zero runs with
        nothing in ``/api/health`` to explain why.
        """
        return value.strip() or "./logs"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide settings, constructed on first use.

    Not a module-level constant so tests can reset it, and so importing this
    module never reads the environment as a side effect.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Replace the process-wide settings. For tests and for ``__main__``."""
    global _settings
    _settings = settings
