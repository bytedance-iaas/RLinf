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

"""Canonical metric key naming.

The same quantity is named differently depending on which runner logged it:
embodied runners emit ``train/actor/*`` while reasoning runners emit
``actor/training/*``. A dashboard template keyed on one of those spellings is
wrong for half the runs, so keys are normalized at the single write entry point.

Migration is a **dual write**: both the canonical and the legacy key are
emitted for one release cycle. Existing wandb and TensorBoard dashboards keep
working, and the legacy key is removed in a later minor version. A deprecation
warning fires once per prefix -- once per *prefix*, not per key or per step,
since these are logged every step and a per-call warning would bury the log.
"""

from __future__ import annotations

from rlinf.utils.logging import get_logger

#: Legacy prefix -> canonical prefix. Longest match wins, so a more specific
#: legacy prefix is not shadowed by a shorter one.
LEGACY_TO_CANONICAL = {
    "actor/training/": "train/actor/",
    "critic/training/": "train/critic/",
}

_warned_prefixes: set[str] = set()


def canonical_key(key: str) -> str:
    """Map a metric key to its canonical spelling.

    Args:
        key: Metric key as the producing worker spelled it.

    Returns:
        The canonical key, or ``key`` unchanged if no alias applies.
    """
    for legacy, canonical in sorted(
        LEGACY_TO_CANONICAL.items(), key=lambda item: -len(item[0])
    ):
        if key.startswith(legacy):
            return canonical + key[len(legacy) :]
    return key


def dual_write(data: dict) -> dict:
    """Add canonical keys alongside legacy ones.

    Both spellings are returned so dashboards built on either keep working
    through the deprecation window.

    Args:
        data: Metrics as produced, possibly using legacy keys.

    Returns:
        A new dict containing the original keys plus canonical aliases. Never
        overwrites a canonical key the caller already supplied.
    """
    if not data:
        return data

    result = dict(data)
    for key, value in data.items():
        canonical = canonical_key(key)
        if canonical == key or canonical in result:
            continue
        result[canonical] = value
        prefix = key.rsplit("/", 1)[0] + "/"
        if prefix not in _warned_prefixes:
            _warned_prefixes.add(prefix)
            _warn_deprecated(prefix, canonical)
    return result


def _warn_deprecated(prefix: str, canonical: str) -> None:
    """Announce a deprecated prefix once, without risking the metric path.

    ``get_logger()`` resolves the worker logger singleton, so it is not
    guaranteed to succeed in every process that logs metrics. This runs on the
    hot path for every step, and a deprecation notice must never be what costs a
    run its metrics.
    """
    try:
        get_logger().warning(
            f"Metric prefix '{prefix}' is deprecated; use "
            f"'{canonical.rsplit('/', 1)[0]}/'. Both are logged for now and "
            f"the legacy prefix will be removed in a future minor release."
        )
    except Exception:  # noqa: BLE001 - a warning is never worth a crash
        pass


def alias_table() -> dict:
    """The alias map, for embedding in ``manifest.json``.

    A reader gets the mapping from the manifest instead of importing ``rlinf``,
    which it cannot do.
    """
    return dict(LEGACY_TO_CANONICAL)
