#!/usr/bin/env python3
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

"""Copy the built frontend into the Python package so a wheel can carry it.

``frontend/dist`` is a sibling of the Python package and is not wheel data. This
copies the build to ``rlinf_dashboard/static/``, which ``package-data`` includes.

Run after ``npm run build`` and before ``pip wheel``. Kept as a separate step
rather than a setuptools build hook on purpose: a hook would make ``pip wheel``
require Node, and the isolation CI job installs this package in a venv that has
neither npm nor rlinf.

Usage:
    python scripts/bundle_frontend.py [--check]

``--check`` verifies the bundled copy is present and current without writing,
which is what CI asserts before building a release.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DASHBOARD = HERE.parent
SOURCE = DASHBOARD / "frontend" / "dist"
TARGET = DASHBOARD / "rlinf_dashboard" / "static"


def _relevant(root: Path) -> set[Path]:
    """Every file under ``root``, as paths relative to it."""
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


def check() -> int:
    """Report whether the bundled copy is present and matches ``dist/``."""
    if not TARGET.is_dir() or not (TARGET / "index.html").is_file():
        print(f"MISSING: no bundled frontend at {TARGET}", file=sys.stderr)
        print(
            "Run: npm --prefix frontend run build && python scripts/bundle_frontend.py"
        )
        return 1
    if not SOURCE.is_dir():
        # Nothing to compare against; the bundled copy is all there is, which is
        # the normal state in a source checkout that has not run npm.
        print(f"ok: bundled frontend present at {TARGET} (no dist/ to compare)")
        return 0

    source_files, target_files = _relevant(SOURCE), _relevant(TARGET)
    if source_files != target_files:
        missing = sorted(str(p) for p in source_files - target_files)
        extra = sorted(str(p) for p in target_files - source_files)
        print(f"STALE: bundled frontend differs from {SOURCE}", file=sys.stderr)
        if missing:
            print(f"  not bundled: {missing}", file=sys.stderr)
        if extra:
            print(f"  stale leftovers: {extra}", file=sys.stderr)
        return 1
    differing = [
        str(rel)
        for rel in sorted(source_files)
        if not filecmp.cmp(SOURCE / rel, TARGET / rel, shallow=False)
    ]
    if differing:
        print(
            f"STALE: {len(differing)} file(s) differ, e.g. {differing[:3]}",
            file=sys.stderr,
        )
        return 1
    print(f"ok: bundled frontend at {TARGET} matches {SOURCE}")
    return 0


def bundle() -> int:
    """Replace the bundled copy with the current ``frontend/dist``."""
    if not (SOURCE / "index.html").is_file():
        print(
            f"No build at {SOURCE}. Run `npm --prefix frontend run build` first.",
            file=sys.stderr,
        )
        return 1
    # Replace wholesale so hashed assets from an older build cannot ship.
    if TARGET.exists():
        shutil.rmtree(TARGET)
    shutil.copytree(SOURCE, TARGET)
    count = len(_relevant(TARGET))
    print(f"bundled {count} file(s) from {SOURCE} -> {TARGET}")
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the bundled copy is present and current; do not write.",
    )
    args = parser.parse_args()
    return check() if args.check else bundle()


if __name__ == "__main__":
    raise SystemExit(main())
