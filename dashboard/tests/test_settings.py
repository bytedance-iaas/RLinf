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

"""Environment-variable configuration.

Two things are worth testing here. The list-typed settings, because
pydantic-settings JSON-decodes a list field's raw string *before* any validator
runs, so ``RLINF_DASHBOARD_CORS_ORIGINS=http://a:5173`` -- the obvious thing to
write -- raised ``SettingsError`` at startup until the field was annotated
``NoDecode``. And the scan root, because it used to be a list: an invocation
that passed two directories must now fail loudly rather than quietly scan one of
them, or the symptom is missing runs that look like a discovery bug.

Both are failures no other test would catch, because every other test constructs
``Settings`` from keyword arguments.
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


# --------------------------------------------------------------- the scan root


def test_the_scan_root_is_read_from_the_environment(monkeypatch):
    assert _settings(monkeypatch, RLINF_DASHBOARD_SCAN_ROOT="/logs").scan_root == "/logs"


def test_a_comma_separated_scan_root_is_refused(monkeypatch):
    """The old spelling has to fail, not silently name a directory with a comma.

    Someone carrying an ``RLINF_DASHBOARD_SCAN_ROOTS=/a,/b`` line forward would
    otherwise get a server scanning a path that does not exist, reported as an
    empty dashboard.
    """
    with pytest.raises(Exception, match="single directory"):
        _settings(monkeypatch, RLINF_DASHBOARD_SCAN_ROOT="/logs,/mnt/other")


def test_a_list_scan_root_is_refused():
    with pytest.raises(Exception, match="common ancestor"):
        Settings(scan_root=["/logs", "/mnt/other"], _env_file=None)


def test_cors_origins_parses_the_same_way(monkeypatch):
    """Same field type, same trap, so the same annotation is needed."""
    settings = _settings(
        monkeypatch, RLINF_DASHBOARD_CORS_ORIGINS="http://a:5173,http://b:5173"
    )
    assert settings.cors_origins == ["http://a:5173", "http://b:5173"]


def test_an_empty_value_falls_back_to_the_default(monkeypatch):
    """An unset-but-exported variable must not scan nothing.

    ``export RLINF_DASHBOARD_SCAN_ROOT=`` in a shell profile is easy to do by
    accident, and a server scanning "" reports zero runs with no explanation.
    """
    assert _settings(monkeypatch, RLINF_DASHBOARD_SCAN_ROOT="  ").scan_root == "./logs"


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
    assert settings.scan_root == "./logs"


# ------------------------------------------------------------------- the accessor


def test_the_process_wide_accessor_is_replaceable(monkeypatch):
    """``set_settings`` is how ``__main__`` injects CLI arguments.

    The app is built from whatever ``get_settings`` returns, so a CLI override
    that did not stick would silently serve the environment's root instead.
    """
    monkeypatch.setattr("rlinf_dashboard.settings._settings", None)
    injected = Settings(scan_root="/injected", _env_file=None)
    set_settings(injected)
    assert get_settings() is injected
