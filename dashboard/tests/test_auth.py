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

"""Authentication coverage across API, streams, media, and static frontend."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from rlinf_dashboard.api import create_app
from rlinf_dashboard.settings import Settings


def _authorization(username: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    return {"Authorization": f"Basic {token}"}


@pytest.fixture
def auth_client(tmp_path):
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text('<div id="root"></div>')
    (assets / "app.js").write_text("console.log('rlinf')")
    settings = Settings(
        scan_root=str(tmp_path / "logs"),
        cors_origins=["http://localhost:5273"],
        frontend_dist=str(dist),
        auth_mode="basic",
        auth_username="operator",
        auth_password="p:ssword",
        _env_file=None,
    )
    with TestClient(create_app(settings)) as client:
        yield client


def test_only_minimal_health_is_public(auth_client):
    response = auth_client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "scan_root" not in response.text
    assert auth_client.get("/api/health").status_code == 401
    assert auth_client.get("/healthz/").status_code == 401


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/assets/app.js",
        "/runs/example/metrics",
        "/api/runs",
        "/api/runs/example/media/file?path=/tmp/example.mp4",
        "/api/stream/runs",
        "/docs",
        "/openapi.json",
        "/api/not-a-real-route",
    ],
)
def test_every_dashboard_surface_requires_auth(auth_client, path):
    response = auth_client.get(path)

    assert response.status_code == 401
    assert response.text == "Authentication required\n"
    assert response.headers["www-authenticate"] == (
        'Basic realm="RLinf Dashboard", charset="UTF-8"'
    )
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "authorization",
    [
        "Bearer token",
        "Basic !!!not-base64!!!",
        "Basic " + base64.b64encode(b"missing-colon").decode("ascii"),
        "Basic " + base64.b64encode(b"operator\xff:secret").decode("ascii"),
    ],
)
def test_malformed_credentials_are_indistinguishable(auth_client, authorization):
    response = auth_client.get("/api/health", headers={"Authorization": authorization})

    assert response.status_code == 401
    assert response.text == "Authentication required\n"


@pytest.mark.parametrize(
    ("username", "password"),
    [("other", "p:ssword"), ("operator", "wrong"), ("other", "wrong")],
)
def test_wrong_credentials_are_refused(auth_client, username, password):
    assert (
        auth_client.get(
            "/api/health", headers=_authorization(username, password)
        ).status_code
        == 401
    )


def test_correct_credentials_preserve_route_semantics(auth_client):
    headers = _authorization("operator", "p:ssword")

    assert auth_client.get("/", headers=headers).status_code == 200
    assert auth_client.get("/assets/app.js", headers=headers).status_code == 200
    assert auth_client.get("/runs/example/metrics", headers=headers).status_code == 200
    assert auth_client.get("/api/health", headers=headers).status_code == 200
    assert auth_client.get("/api/not-a-real-route", headers=headers).status_code == 404


def test_basic_scheme_is_case_insensitive_and_accepts_header_whitespace(auth_client):
    authorization = _authorization("operator", "p:ssword")["Authorization"]
    token = authorization.split(" ", maxsplit=1)[1]

    assert (
        auth_client.get(
            "/api/health", headers={"Authorization": f"bAsIc   {token}"}
        ).status_code
        == 200
    )


def test_openapi_declares_the_middleware_security_scheme(auth_client):
    schema = auth_client.get(
        "/openapi.json", headers=_authorization("operator", "p:ssword")
    ).json()

    assert schema["components"]["securitySchemes"]["basicAuth"] == {
        "type": "http",
        "scheme": "basic",
    }
    assert schema["security"] == [{"basicAuth": []}]


def test_middleware_configuration_does_not_reveal_the_password(auth_client):
    middleware_repr = repr(auth_client.app.user_middleware)

    assert "p:ssword" not in middleware_repr
    assert "**********" in middleware_repr


def test_cors_preflight_does_not_require_credentials(auth_client):
    response = auth_client.options(
        "/api/runs",
        headers={
            "Origin": "http://localhost:5273",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5273"

    unauthorized = auth_client.get(
        "/api/runs", headers={"Origin": "http://localhost:5273"}
    )
    assert unauthorized.status_code == 401
    assert (
        unauthorized.headers["access-control-allow-origin"] == "http://localhost:5273"
    )


def test_username_and_password_are_both_compared(auth_client, monkeypatch):
    calls: list[tuple[bytes, bytes]] = []

    def compare(left: bytes, right: bytes) -> bool:
        calls.append((left, right))
        return left == right

    monkeypatch.setattr("rlinf_dashboard.auth.secrets.compare_digest", compare)
    response = auth_client.get(
        "/api/health", headers=_authorization("wrong", "p:ssword")
    )

    assert response.status_code == 401
    assert len(calls) == 2
