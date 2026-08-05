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

"""Serving recorded rollout videos.

Video playback itself is a later milestone; the index has to exist now because
retrofitting it would mean backfilling every historical run. What is here is the
query and streaming layer on top of that index.

Access is allowlisted rather than filtered: a request may only fetch a path that
appears in one of the run's own media index rows. Video paths are absolute
strings written by env worker processes, so treating them as untrusted input and
requiring an exact match against the index is what keeps a request from reading
arbitrary files -- no amount of ``..`` normalization is as reliable as an
allowlist.
"""

from __future__ import annotations

import logging
import os

from .discovery import DiscoveredRun
from .models import MediaEntry
from .relocate import relocate_file
from .state import StateStore

logger = logging.getLogger(__name__)

#: Extensions the index is allowed to hand out. RecordVideo writes MP4; the rest
#: cover manual additions and image logging.
ALLOWED_SUFFIXES = (".mp4", ".webm", ".gif", ".png", ".jpg", ".jpeg")

_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".gif": "image/gif",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class MediaService:
    """Query the media index and resolve index rows to files on disk."""

    def __init__(self, store: StateStore):
        self._store = store

    def list_media(
        self,
        run: DiscoveredRun,
        *,
        split: str | None = None,
        step: int | None = None,
        min_step: int | None = None,
        max_step: int | None = None,
        success: bool | None = None,
        limit: int = 200,
    ) -> list[MediaEntry]:
        """Return index rows for a run, filtered and URL-annotated.

        Args:
            run: The run whose index to read.
            split: Keep only ``train`` or ``eval`` rows.
            step: Keep only rows at exactly this step.
            min_step: Lower bound, inclusive.
            max_step: Upper bound, inclusive.
            success: ``True`` keeps clips where at least one env succeeded,
                ``False`` keeps clips where none did. Rows with no success
                information are dropped by either value rather than guessed at --
                "show me the failures" must not surface clips whose outcome is
                simply unrecorded.
            limit: Maximum rows returned; the newest are kept, since a long run
                produces thousands of clips and the recent ones are the ones
                worth watching.

        Returns:
            Matching entries, each with ``url`` set to a streaming endpoint.
        """
        entries = self._store.read_media(run.run_root)

        if split is not None:
            entries = [entry for entry in entries if entry.split == split]
        if step is not None:
            entries = [entry for entry in entries if entry.step == step]
        if success is not None:
            entries = [
                entry
                for entry in entries
                if _succeeded(entry) is not None and _succeeded(entry) == success
            ]
        if min_step is not None:
            entries = [
                entry
                for entry in entries
                if entry.step is not None and entry.step >= min_step
            ]
        if max_step is not None:
            entries = [
                entry
                for entry in entries
                if entry.step is not None and entry.step <= max_step
            ]

        if limit > 0:
            entries = entries[-limit:]

        # The index records the path the env worker wrote to, which is only valid
        # in that namespace. Translate before building the URL so a tree read
        # from a host mount still streams, rather than listing every clip and
        # 404-ing all of them.
        return [
            entry.model_copy(
                update={
                    "path": relocate_file(entry.path, run.relocation),
                    "url": self._url(
                        run.run_id, relocate_file(entry.path, run.relocation)
                    ),
                }
            )
            for entry in entries
        ]

    def resolve(self, run: DiscoveredRun, path: str) -> str | None:
        """Map a requested path to a file, or ``None`` if it is not allowed.

        The path must appear verbatim in this run's own media index. Both sides
        are compared as real paths so a symlinked log directory still matches,
        while a path that is merely *reachable* from the run root but never
        indexed is refused.
        """
        target = _realpath(path)
        if target is None or not target.lower().endswith(ALLOWED_SUFFIXES):
            return None

        for entry in self._store.read_media(run.run_root):
            # Compare against the translated path, since that is what the listing
            # handed the client. The allowlist property is unchanged: the path
            # must still correspond to a row in this run's own index.
            indexed = relocate_file(entry.path, run.relocation)
            if _realpath(indexed) == target:
                return target if os.path.isfile(target) else None
        return None

    def steps(self, run: DiscoveredRun) -> list[int]:
        """Distinct steps that have media, ascending.

        Lets the UI mark which points on a curve have a clip behind them, which
        is the whole reason the index carries a step at all.
        """
        steps = {
            entry.step
            for entry in self._store.read_media(run.run_root)
            if entry.step is not None
        }
        return sorted(steps)

    @staticmethod
    def _url(run_id: str, path: str) -> str:
        from urllib.parse import quote

        return f"/api/runs/{quote(run_id, safe='')}/media/file?path={quote(path)}"


def _realpath(path: str) -> str | None:
    """Resolve a path, or ``None`` if the string cannot name a file at all.

    ``os.path.realpath`` raises on an embedded null byte rather than returning
    something unresolvable, and this runs on a request parameter. Returning
    ``None`` turns that into the 404 it should be instead of a 500.
    """
    try:
        return os.path.realpath(path)
    except (OSError, ValueError):
        return None


def _succeeded(entry: MediaEntry) -> bool | None:
    """Whether any env in a clip succeeded, or ``None`` if unrecorded.

    ``num_success`` is authoritative when present; the scalar ``success`` covers
    a single-env clip. Neither present means the row predates the field or the
    env has no success notion, and the caller must not treat that as a failure.
    """
    if entry.num_success is not None:
        return entry.num_success > 0
    return entry.success


def content_type_for(path: str) -> str:
    """Map a file extension to a MIME type.

    A wrong type makes a browser download a clip instead of playing it, so the
    default is the neutral one rather than a guess at video.
    """
    return _CONTENT_TYPES.get(
        os.path.splitext(path)[1].lower(), "application/octet-stream"
    )
