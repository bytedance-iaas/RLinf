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
from typing import Annotated

from pydantic import Field, field_validator
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
    #: polling faster than this only costs syscalls.
    sse_interval_s: float = 2.0

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
        """Turn the old plural spelling into an error rather than a surprise.

        ``RLINF_DASHBOARD_SCAN_ROOTS`` and a second positional path both used to
        work. Silently ignoring either would leave a dashboard scanning one
        directory while its operator believes it is scanning two, and the symptom
        -- some runs missing -- looks like a discovery bug rather than a config
        change.
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
