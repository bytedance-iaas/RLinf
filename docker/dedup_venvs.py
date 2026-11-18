#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardlink identical installed files across the venvs under one directory.

The image build keeps uv's cache in a BuildKit cache mount, outside the image,
so uv copies every package into its venv instead of hardlinking it from a cache
in the same layer. Run after each install.sh call of a RUN, this links the
copies again: a file that a dist-info RECORD lists, and whose content still
matches the recorded sha256, becomes a hardlink to an identical file of
another venv, as uv's hardlink mode would have made it. A file changed after
installation no longer matches its RECORD and keeps its own copy, and so does
everything no RECORD lists (git checkouts, assets, generated files).

The venvs must not be modified once linked: a later in-place write to a linked
file shows in every venv that shares it, as it would with uv's hardlinks.

Usage::

    dedup_venvs[VENV_ROOT]  # default: $UV_PATH, else /opt/venv
"""

from __future__ import annotations

import base64
import csv
import hashlib
import os
import stat
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def recorded_files(root: Path) -> dict[tuple[str, int], set[str]]:
    """Maps each RECORD (sha256, size) to the venv paths listed with it."""
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for record in root.glob("*/lib/python*/site-packages/*.dist-info/RECORD"):
        site_packages = record.parent.parent
        venv = str(site_packages.parents[2]) + os.sep
        with open(record, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 3 or not row[1].startswith("sha256="):
                    continue
                if not row[2].isdigit() or int(row[2]) == 0:
                    continue
                path = os.path.normpath(os.path.join(site_packages, row[0]))
                if path.startswith(venv):
                    groups[(row[1][len("sha256=") :], int(row[2]))].add(path)
    return groups


def has_digest(path: str, digest: str) -> bool:
    """Whether the file's sha256, in RECORD's encoding, is ``digest``."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return base64.urlsafe_b64encode(sha.digest()).rstrip(b"=").decode() == digest


def link_group(digest: str, size: int, paths: set[str]) -> tuple[int, int]:
    """Points every verified path of one RECORD entry at a single inode.

    Returns:
        The number of paths replaced by a link, and the bytes those copies took.
    """
    inodes: dict[tuple[int, int], list[tuple[str, os.stat_result]]] = defaultdict(list)
    for path in sorted(paths):
        # Two distributions can list one file (a fork installed over the
        # package it forks); uninstalling either deletes it under the other.
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            continue
        if stat.S_ISREG(st.st_mode) and st.st_size == size:
            inodes[(st.st_dev, st.st_ino)].append((path, st))
    if len(inodes) < 2:
        return 0, 0

    # Prefer the inode an earlier run already shared, so it is not relinked.
    candidates = sorted(inodes.values(), key=lambda names: (-len(names), names[0][0]))
    source = next((n for n in candidates if has_digest(n[0][0], digest)), None)
    if source is None:
        return 0, 0
    source_path, source_st = source[0]
    owner = (stat.S_IMODE(source_st.st_mode), source_st.st_uid, source_st.st_gid)

    replaced = freed = 0
    for names in candidates:
        path, st = names[0]
        if names is source or (stat.S_IMODE(st.st_mode), st.st_uid, st.st_gid) != owner:
            continue
        if not has_digest(path, digest):
            continue
        for path, _ in names:
            os.link(source_path, f"{path}.dedup")
            os.replace(f"{path}.dedup", path)
        replaced += len(names)
        freed += size
    return replaced, freed


def main() -> int:
    """Links the venvs under the directory given on the command line."""
    root = Path(
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("UV_PATH", "/opt/venv")
    )
    groups = [(d, s, p) for (d, s), p in recorded_files(root).items() if len(p) > 1]
    replaced = freed = 0
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as pool:
        for count, size in pool.map(lambda group: link_group(*group), groups):
            replaced += count
            freed += size
    print(
        f"[dedup_venvs] {root}: linked {replaced} files, saving {freed / 2**30:.2f} GiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
