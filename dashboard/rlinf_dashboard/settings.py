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

    The timeout constants are read-side *policy*, not contract: they are
    deliberately absent from ``run.v2.schema.json`` so that tuning them never
    counts as a schema change. They are expressed as multiples of the run's own
    observed step time, because "silent for 60s" means something entirely
    different for a 2s reasoning step than for a 400s embodied step.
    """

    model_config = SettingsConfigDict(
        env_prefix="RLINF_DASHBOARD_",
        env_file=".env",
        extra="ignore",
    )

    #: Directories to scan. Each is a ``runner.logger.log_path`` (or any ancestor
    #: of several), searched for ``_rlinf/runs/*/manifest.json``.
    #:
    #: ``NoDecode`` because pydantic-settings JSON-decodes list-typed fields
    #: *before* validators run, so without it
    #: ``RLINF_DASHBOARD_SCAN_ROOTS=/logs`` raises a ``SettingsError`` at startup
    #: instead of reaching ``_split_csv``. A path is the natural thing to put in
    #: that variable, and requiring ``["/logs"]`` would be a trap.
    scan_roots: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["./logs"]
    )

    #: How deep below a scan root to look for the ``_rlinf`` marker. Bounded
    #: because a scan root on NFS can be arbitrarily large and an unbounded walk
    #: would make ``/runs`` latency depend on unrelated data.
    scan_max_depth: int = 6

    #: Multiples of ``max(step_time_p50, timeout_floor_s)``. Kept identical to
    #: the values in ``tests/unit_tests/test_run_state_contract.py``; the shared
    #: fixtures fail if the two drift.
    heartbeat_timeout_k: float = 5.0
    progress_timeout_k: float = 10.0

    #: Lower bound on the timeout budget, for runs with no step-time samples yet
    #: (startup can take minutes: sglang warmup, simulator boot).
    timeout_floor_s: float = 30.0

    #: SSE push period. The training side heartbeats every 5s by default, so
    #: polling faster than this only costs syscalls.
    sse_interval_s: float = 2.0

    #: Discovery result cache TTL. A scan stats every candidate manifest, which
    #: is the expensive part of ``/runs`` on a large NFS tree.
    discovery_cache_ttl_s: float = 5.0

    #: Max points returned per series. Above this the response is decimated by
    #: strided sampling, since no display has more than a few thousand pixels.
    max_series_points: int = 4000

    #: CORS origins for the Vite dev server. Empty in production, where the
    #: frontend is served from this same origin. ``NoDecode`` for the same reason
    #: as ``scan_roots``. The port must match ``server.port`` in
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

    @field_validator("scan_roots", "cors_origins", mode="before")
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

    @field_validator("scan_roots")
    @classmethod
    def _no_empty_scan_roots(cls, value: list[str]) -> list[str]:
        """Fall back to the default rather than scanning nothing.

        ``export RLINF_DASHBOARD_SCAN_ROOTS=`` is easy to leave in a shell
        profile, and an empty list makes the server report zero runs with nothing
        in ``/api/health`` to explain why. Unlike ``cors_origins``, where empty is
        a meaningful production setting, an empty scan list has no valid use.
        """
        return value or ["./logs"]


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
