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

"""Making launch-time absolute paths usable from wherever the tree is read.

``manifest.paths`` records absolute paths resolved on the machine that ran the
job. That is deliberate -- runners disagree about where checkpoints and metrics
go, so the manifest records the answer rather than making every reader
reimplement each runner's convention.

But it means the paths are only valid in the namespace that produced them, and
the common deployments do not share one:

* the job runs in a container under ``/workspace/...`` and the dashboard reads
  the same volume mounted at ``/data/logs`` on the host;
* a run tree is copied off a cluster to look at locally;
* the log volume is remounted somewhere else.

In every case the tree is intact and only the prefix moved, and the symptom is
the worst kind: ``list_keys`` finds no event files, so the run renders with zero
charts and nothing anywhere says why. A run that looks metric-less is
indistinguishable from a run that logged nothing.

The fix needs no guessing, because the manifest already contains the answer.
``paths.run_root`` is the launch-time location of the very directory discovery
just found on disk. The difference between the two is the prefix translation,
exactly:

    manifest: paths.run_root = /workspace/run1/logs/_rlinf/runs/<id>
    on disk:  run_root       = /data/logs/_rlinf/runs/<id>
    =>        /workspace/run1/logs  ->  /data/logs

Applied to the other recorded paths, ``tensorboard`` resolves and the charts
come back.

Two rules keep this honest:

* **Only rewrite a path that does not exist, to one that does.** If the recorded
  path resolves, it wins -- no rewrite. If the rewrite does not resolve either,
  the original is kept so the API still reports what the training side actually
  said. A relocation that invents a path would be worse than the blank charts it
  replaces.
* **Report it.** The translation is exposed as ``relocation`` on the run status,
  so a surprising path in the UI can be traced to this module rather than
  looking like the training side recorded something wrong.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: Paths whose value is a directory or file on the machine that ran the job.
#: ``run_root`` is excluded: it is the evidence used to derive the mapping, and
#: discovery's on-disk value for it is already correct by construction.
_RELOCATABLE = (
    "log_path",
    "tensorboard",
    "worker_logs",
    "video_root",
    "checkpoint_root",
)


def derive_prefix(
    recorded_run_root: str | None, actual_run_root: str
) -> tuple[str, str] | None:
    """Find the prefix translation between two paths to the same run directory.

    Args:
        recorded_run_root: ``manifest.paths['run_root']``, as written at launch.
        actual_run_root: Where discovery actually found the directory.

    Returns:
        ``(old_prefix, new_prefix)``, or ``None`` when no translation is needed
        or none can be derived.
    """
    if not recorded_run_root:
        return None

    recorded = recorded_run_root.rstrip("/")
    actual = actual_run_root.rstrip("/")
    if recorded == actual:
        return None

    # Peel identical trailing components. What remains on each side is the pair
    # of prefixes. The shared tail is at minimum `_rlinf/runs/<run_id>`, so a
    # match is structural rather than coincidental.
    old_parts = recorded.split(os.sep)
    new_parts = actual.split(os.sep)
    shared = 0
    while (
        shared < len(old_parts)
        and shared < len(new_parts)
        and old_parts[len(old_parts) - 1 - shared]
        == new_parts[len(new_parts) - 1 - shared]
    ):
        shared += 1

    if shared == 0:
        # Not even the run id matches. These are not two views of one directory,
        # so there is no translation to make.
        return None

    old_prefix = os.sep.join(old_parts[: len(old_parts) - shared])
    new_prefix = os.sep.join(new_parts[: len(new_parts) - shared])
    if old_prefix == new_prefix:
        return None
    return old_prefix, new_prefix


def relocate_paths(
    paths: dict[str, str | None],
    recorded_run_root: str | None,
    actual_run_root: str,
) -> tuple[dict[str, str | None], dict[str, str] | None]:
    """Rewrite recorded paths into the reading machine's namespace.

    Args:
        paths: ``manifest.paths`` or ``snapshot.paths``.
        recorded_run_root: The launch-time ``run_root`` from the same document.
        actual_run_root: Where discovery found the run directory.

    Returns:
        ``(paths, relocation)``. ``paths`` is the input when nothing was
        rewritten, otherwise a new dict. ``relocation`` describes what was
        translated, or is ``None`` when nothing was.
    """
    prefix = derive_prefix(recorded_run_root or paths.get("run_root"), actual_run_root)
    if prefix is None:
        return paths, None
    old_prefix, new_prefix = prefix

    out = dict(paths)
    rewritten: list[str] = []
    for name in _RELOCATABLE:
        value = out.get(name)
        if not value or not value.startswith(old_prefix):
            continue
        # A recorded path that still resolves is authoritative: the reader may
        # genuinely share the writer's namespace for some paths and not others
        # (a bind-mounted checkpoint volume, say).
        if os.path.exists(value):
            continue
        candidate = new_prefix + value[len(old_prefix) :]
        if not os.path.exists(candidate):
            # Keep the original rather than substituting a path that is also
            # wrong -- the API should report what training recorded, not a guess.
            continue
        out[name] = candidate
        rewritten.append(name)

    # `run_root` is set unconditionally: discovery's value is correct by
    # construction, and leaving the stale one in would make the UI show a path
    # the reader cannot open.
    if out.get("run_root") != actual_run_root:
        out["run_root"] = actual_run_root
        rewritten.append("run_root")

    if not rewritten:
        return paths, None

    logger.info(
        "Relocated run paths %s: %s -> %s",
        ",".join(rewritten),
        old_prefix or "(root)",
        new_prefix or "(root)",
    )
    return out, {
        "from_prefix": old_prefix,
        "to_prefix": new_prefix,
        "rewritten": ",".join(rewritten),
    }


def relocate_file(path: str, relocation: dict[str, str] | None) -> str:
    """Apply a known relocation to one recorded file path.

    For per-file paths recorded in an index -- an mp4 in ``media.rank*.jsonl`` --
    where the prefix mapping was already derived for the run as a whole.

    Note what this is deliberately *not* used for: the checkpoint index's
    ``resume_dir`` and ``entry_script``. Those exist to be pasted into a command
    run on the training machine, where the recorded path is the correct one.
    Rewriting them into the reader's namespace would produce a resume command
    that only works on the laptop someone was browsing from.

    Args:
        path: A path as recorded by the training side.
        relocation: The mapping from :func:`relocate_paths`, or ``None``.

    Returns:
        The translated path when translation applies and lands on something that
        exists, otherwise ``path`` unchanged.
    """
    if not relocation or not path:
        return path
    old_prefix = relocation.get("from_prefix", "")
    new_prefix = relocation.get("to_prefix", "")
    if not path.startswith(old_prefix) or os.path.exists(path):
        return path
    candidate = new_prefix + path[len(old_prefix) :]
    return candidate if os.path.exists(candidate) else path
