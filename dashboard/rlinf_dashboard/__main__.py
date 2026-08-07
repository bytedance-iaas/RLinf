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

"""Command-line entry point: ``rlinf-dashboard`` or ``python -m rlinf_dashboard``."""

from __future__ import annotations

import argparse
import logging
import sys

from .settings import Settings, set_settings


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, then serve.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="rlinf-dashboard",
        description="Serve the RLinf control-plane dashboard.",
    )
    parser.add_argument(
        "log_paths",
        nargs="*",
        help=(
            "Directories to scan for runs -- each a runner.logger.log_path or an "
            "ancestor of several. Defaults to RLINF_DASHBOARD_SCAN_ROOTS or ./logs."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload on code changes (development only).",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    overrides = {}
    if args.log_paths:
        overrides["scan_roots"] = args.log_paths
    settings = Settings(**overrides)
    set_settings(settings)

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Install this package with: pip install rlinf-dashboard",
            file=sys.stderr,
        )
        return 1

    print(f"Scanning: {', '.join(settings.scan_roots)}", file=sys.stderr)
    print(f"Dashboard: http://{args.host}:{args.port}/api/health", file=sys.stderr)

    # `--reload` needs an import string so the reloader can re-import; the direct
    # object path is used otherwise so the app is built once, in this process,
    # with the settings assembled above.
    if args.reload:
        uvicorn.run(
            "rlinf_dashboard.api:get_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level,
        )
    else:
        from .api import create_app

        uvicorn.run(
            create_app(settings),
            host=args.host,
            port=args.port,
            log_level=args.log_level,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
