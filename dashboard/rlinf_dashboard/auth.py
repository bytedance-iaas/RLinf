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

"""Process-wide HTTP Basic authentication for the dashboard."""

from __future__ import annotations

import base64
import binascii
import secrets
from collections.abc import Iterable

from pydantic import SecretStr
from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class BasicAuthMiddleware:
    """Protect every HTTP surface with one static username and password.

    Implemented as plain ASGI middleware rather than ``BaseHTTPMiddleware`` so
    the dashboard's long-lived SSE responses keep streaming without an extra
    buffering task. Only explicitly named health endpoints bypass auth.

    Args:
        app: The wrapped ASGI application.
        username: Expected Basic Auth username.
        password: Expected Basic Auth password.
        public_paths: Exact paths that remain available without credentials.
    """

    def __init__(
        self,
        app: ASGIApp,
        username: str,
        password: SecretStr,
        public_paths: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.username = username.encode("utf-8")
        self.password = password.get_secret_value().encode("utf-8")
        self.public_paths = frozenset(public_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Authenticate an HTTP request, or pass non-HTTP ASGI events through."""
        if scope["type"] != "http" or scope.get("path") in self.public_paths:
            await self.app(scope, receive, send)
            return

        credentials = self._credentials(Headers(scope=scope).get("authorization"))
        if credentials is not None:
            username, password = credentials
            # Evaluate both comparisons so a wrong username does not skip the
            # password check and create a simple username timing oracle.
            username_matches = secrets.compare_digest(
                username.encode("utf-8"), self.username
            )
            password_matches = secrets.compare_digest(
                password.encode("utf-8"), self.password
            )
            if username_matches and password_matches:
                await self.app(scope, receive, send)
                return

        response = PlainTextResponse(
            "Authentication required\n",
            status_code=401,
            headers={
                "WWW-Authenticate": 'Basic realm="RLinf Dashboard", charset="UTF-8"',
                "Cache-Control": "no-store",
            },
        )
        await response(scope, receive, send)

    @staticmethod
    def _credentials(header: str | None) -> tuple[str, str] | None:
        """Decode a Basic Authorization header without leaking parse details."""
        if not header:
            return None
        parts = header.split(None, maxsplit=1)
        if len(parts) != 2:
            return None
        scheme, token = parts
        if scheme.lower() != "basic" or not token:
            return None
        try:
            decoded = base64.b64decode(token, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return None
        username, separator, password = decoded.partition(":")
        if not separator:
            return None
        return username, password
