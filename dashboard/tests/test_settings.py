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

"""Environment-variable configuration.

The list-typed settings are the ones worth testing. pydantic-settings
JSON-decodes a list field's raw string *before* any validator runs, so
``RLINF_DASHBOARD_SCAN_ROOTS=/logs`` -- the obvious thing to write, and what the
README documents -- raised ``SettingsError`` at startup until these fields were
annotated ``NoDecode``. The server refusing to boot on its own documented
variable is exactly the kind of failure no unit test elsewhere would catch,
because every other test constructs ``Settings`` from keyword arguments.
"""

from __future__ import annotations

import pytest

from rlinf_dashboard.settings import Settings, get_settings, set_settings


def _settings(monkeypatch, **env) -> Settings:
    """Build settings from environment variables alone.

    ``_env_file=None`` so a developer's own ``.env`` beside the repo cannot
    change the answer.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


# ------------------------------------------------------------------- list fields


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The single most likely value, and the one the README shows.
        ("/logs", ["/logs"]),
        ("/logs,/mnt/other", ["/logs", "/mnt/other"]),
        # Whitespace around a comma is what a human types.
        (" /logs , /mnt/other ", ["/logs", "/mnt/other"]),
        # A JSON array is what someone who knows pydantic-settings reaches for.
        ('["/logs","/mnt/other"]', ["/logs", "/mnt/other"]),
        # Trailing separators must not produce empty roots, which would scan the
        # process's cwd by accident.
        ("/logs,", ["/logs"]),
    ],
    ids=["single", "csv", "csv-spaced", "json", "trailing-comma"],
)
def test_scan_roots_accepts_every_reasonable_spelling(monkeypatch, raw, expected):
    assert _settings(monkeypatch, RLINF_DASHBOARD_SCAN_ROOTS=raw).scan_roots == expected


def test_cors_origins_parses_the_same_way(monkeypatch):
    """Same field type, same trap, so the same annotation is needed."""
    settings = _settings(
        monkeypatch, RLINF_DASHBOARD_CORS_ORIGINS="http://a:5173,http://b:5173"
    )
    assert settings.cors_origins == ["http://a:5173", "http://b:5173"]


def test_an_empty_value_falls_back_to_the_default(monkeypatch):
    """An unset-but-exported variable must not scan nothing.

    ``export RLINF_DASHBOARD_SCAN_ROOTS=`` in a shell profile is easy to do by
    accident, and a server scanning ``[]`` reports zero runs with no explanation.
    """
    assert _settings(monkeypatch, RLINF_DASHBOARD_SCAN_ROOTS="").scan_roots == [
        "./logs"
    ]


# ----------------------------------------------------------------- scalar fields


def test_timeout_policy_is_overridable(monkeypatch):
    """These are read-side policy, deliberately absent from the schema.

    Tuning them must not require a code change, because the right multiple
    depends on the cluster: a 2s reasoning step and a 428s embodied step want
    very different patience.
    """
    settings = _settings(
        monkeypatch,
        RLINF_DASHBOARD_HEARTBEAT_TIMEOUT_K="3",
        RLINF_DASHBOARD_PROGRESS_TIMEOUT_K="8.5",
        RLINF_DASHBOARD_TIMEOUT_FLOOR_S="120",
    )
    assert settings.heartbeat_timeout_k == 3.0
    assert settings.progress_timeout_k == 8.5
    assert settings.timeout_floor_s == 120.0


def test_an_unknown_prefixed_variable_is_ignored(monkeypatch):
    """``extra="ignore"``: a stale variable from an older version must not block boot."""
    settings = _settings(monkeypatch, RLINF_DASHBOARD_SOMETHING_REMOVED="1")
    assert settings.scan_roots == ["./logs"]


# ------------------------------------------------------------------- the accessor


def test_the_process_wide_accessor_is_replaceable(monkeypatch):
    """``set_settings`` is how ``__main__`` injects CLI arguments.

    The app is built from whatever ``get_settings`` returns, so a CLI override
    that did not stick would silently serve the environment's roots instead.
    """
    monkeypatch.setattr("rlinf_dashboard.settings._settings", None)
    injected = Settings(scan_roots=["/injected"], _env_file=None)
    set_settings(injected)
    assert get_settings() is injected
