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

List-typed settings need ``NoDecode`` so comma-separated CORS origins reach
their validator unchanged. The scan root is intentionally singular: lists and
comma-separated paths must fail loudly rather than leave an operator with a
partially scanned filesystem.

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
    monkeypatch.delenv("RLINF_DASHBOARD_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("RLINF_DASHBOARD_AUTH_PASSWORD", raising=False)
    monkeypatch.delenv("RLINF_DASHBOARD_AUTH_MODE", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


# --------------------------------------------------------------- the scan root


def test_the_scan_root_is_read_from_the_environment(monkeypatch):
    assert (
        _settings(monkeypatch, RLINF_DASHBOARD_SCAN_ROOT="/logs").scan_root == "/logs"
    )


def test_a_comma_separated_scan_root_is_refused(monkeypatch):
    """Multiple directories must fail instead of becoming one invalid path."""
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


# ---------------------------------------------------------------- authentication


def test_auth_credentials_are_read_as_a_pair(monkeypatch):
    settings = _settings(
        monkeypatch,
        RLINF_DASHBOARD_AUTH_MODE="basic",
        RLINF_DASHBOARD_AUTH_USERNAME="operator",
        RLINF_DASHBOARD_AUTH_PASSWORD="correct horse battery staple",
    )

    assert settings.auth_enabled
    assert settings.auth_username == "operator"
    assert settings.auth_password.get_secret_value() == "correct horse battery staple"
    assert "correct horse battery staple" not in repr(settings)


def test_auth_is_disabled_only_when_both_values_are_unset(monkeypatch):
    assert not _settings(monkeypatch).auth_enabled


@pytest.mark.parametrize(
    ("env", "message"),
    [
        (
            {
                "RLINF_DASHBOARD_AUTH_MODE": "basic",
                "RLINF_DASHBOARD_AUTH_USERNAME": "operator",
            },
            "requires both",
        ),
        (
            {
                "RLINF_DASHBOARD_AUTH_MODE": "basic",
                "RLINF_DASHBOARD_AUTH_PASSWORD": "secret",
            },
            "requires both",
        ),
        (
            {
                "RLINF_DASHBOARD_AUTH_MODE": "basic",
                "RLINF_DASHBOARD_AUTH_USERNAME": "",
                "RLINF_DASHBOARD_AUTH_PASSWORD": "",
            },
            "must not be blank",
        ),
        (
            {
                "RLINF_DASHBOARD_AUTH_MODE": "basic",
                "RLINF_DASHBOARD_AUTH_USERNAME": "operator",
                "RLINF_DASHBOARD_AUTH_PASSWORD": "   ",
            },
            "must not be blank",
        ),
        (
            {
                "RLINF_DASHBOARD_AUTH_MODE": "basic",
                "RLINF_DASHBOARD_AUTH_USERNAME": "team:operator",
                "RLINF_DASHBOARD_AUTH_PASSWORD": "secret",
            },
            "must not contain ':'",
        ),
    ],
)
def test_incomplete_or_ambiguous_auth_is_refused(monkeypatch, env, message):
    with pytest.raises(Exception, match=message):
        _settings(monkeypatch, **env)


def test_credentials_are_refused_when_auth_mode_is_disabled(monkeypatch):
    with pytest.raises(Exception, match="AUTH_MODE=basic"):
        _settings(
            monkeypatch,
            RLINF_DASHBOARD_AUTH_USERNAME="operator",
            RLINF_DASHBOARD_AUTH_PASSWORD="secret",
        )


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
