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

"""Declarative chart templates.

A template says which metrics matter for a kind of run and how to lay them out.
Keeping that in YAML on the server -- rather than in the frontend -- is what
makes adding a ``task_type`` a data change instead of a code change, and it
versions the layout alongside the repo that produces the metrics.

Selection key is ``task_type``, optionally refined by ``algorithm.loss_type``. A
three-dimensional key (paradigm x domain x algorithm) was considered and
rejected: with four templates it is pure complexity. ``extends`` covers the real
reuse case (PPO and GRPO sharing a policy-health group) at a fraction of the
cost.

RL's internal variation is larger than the SFT-versus-RL split, which is why
there is no single "RL" template: embodied runs have environments and simulator
video, reasoning runs have no environment but do have FLOPs and sequence-level
metrics, offline runs have ``replay_buffer/*``.
"""

from __future__ import annotations

import copy
import logging
import os

import yaml

from .models import RunManifest, RunSnapshot

logger = logging.getLogger(__name__)

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
FALLBACK_NAME = "fallback"


class TemplateRegistry:
    """Load templates from disk and pick one per run."""

    def __init__(self, template_dir: str | None = None):
        self._dir = template_dir or TEMPLATE_DIR
        self._templates: dict[str, dict] = {}
        self._by_task_type: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        """Re-read every YAML file in the template directory.

        Exposed so an operator can drop in a template and pick it up without a
        restart, and so tests can point at a temporary directory.
        """
        templates: dict[str, dict] = {}
        try:
            names = sorted(os.listdir(self._dir))
        except OSError as exc:
            logger.error("Cannot list template dir %s: %s", self._dir, exc)
            self._templates, self._by_task_type = {}, {}
            return

        for filename in names:
            if not filename.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(self._dir, filename)
            try:
                with open(path, encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
            except (OSError, yaml.YAMLError) as exc:
                logger.error("Ignoring bad template %s: %s", path, exc)
                continue
            name = data.get("name") or os.path.splitext(filename)[0]
            data["name"] = name
            templates[name] = data

        self._templates = {name: _resolve(name, templates) for name in templates}
        self._by_task_type = {}
        for name, data in self._templates.items():
            for task_type in data.get("task_types") or []:
                self._by_task_type[str(task_type)] = name
            # A template may also claim a `task_type:loss_type` pair, which is
            # how a future algorithm-specific layout attaches without changing
            # the selection logic here.
            for pair in data.get("loss_types") or []:
                self._by_task_type[str(pair)] = name

    # ------------------------------------------------------------------ lookup

    def names(self) -> list[str]:
        """Names of every loaded template, sorted."""
        return sorted(self._templates)

    def get(self, name: str) -> dict | None:
        """Return a template by name, or ``None``.

        A deep copy, because ``bind_keys`` rewrites ``groups`` per request and a
        shared dict would make the first run's key set stick for every run after
        it.
        """
        return copy.deepcopy(self._templates.get(name))

    def all(self) -> list[dict]:
        """Return every template, each a copy, for the ``/api/templates`` view."""
        return [copy.deepcopy(self._templates[name]) for name in self.names()]

    def select(
        self,
        task_type: str | None,
        loss_type: str | None = None,
    ) -> dict:
        """Choose the template for a run.

        The refined ``task_type:loss_type`` key is tried first so an
        algorithm-specific template can exist without displacing the general one.

        Args:
            task_type: ``cfg.runner.task_type`` as recorded in the snapshot.
            loss_type: ``cfg.algorithm.loss_type``, when present.

        Returns:
            A template dict, always. Falls back to the generic prefix-grouping
            template, because an unrecognised ``task_type`` must still render.
        """
        if task_type and loss_type:
            name = self._by_task_type.get(f"{task_type}:{loss_type}")
            if name:
                return copy.deepcopy(self._templates[name])
        if task_type:
            name = self._by_task_type.get(task_type)
            if name:
                return copy.deepcopy(self._templates[name])
        return copy.deepcopy(self._templates.get(FALLBACK_NAME, _EMPTY_TEMPLATE))

    def select_for(
        self,
        manifest: RunManifest | None,
        snapshot: RunSnapshot | None,
    ) -> dict:
        """Pick a template from whichever of the two documents is readable.

        The snapshot is preferred but a run that died before its first flush has
        only a manifest, and that run still needs a page to be shown on.
        """
        task_type = None
        loss_type = None
        if snapshot is not None:
            task_type = str(snapshot.task_type)
            if snapshot.algorithm is not None:
                loss_type = snapshot.algorithm.loss_type
        elif manifest is not None:
            task_type = str(manifest.task_type)
            if manifest.algorithm is not None:
                loss_type = manifest.algorithm.loss_type
        return self.select(task_type, loss_type)


def bind_keys(
    template: dict,
    available: list[str],
    aliases: dict[str, str] | None = None,
) -> dict:
    """Fill a template in against the keys a run actually logged.

    Three jobs:

    * drop chart entries whose keys are all absent, so a template listing both
      ``train/actor/loss`` and ``train/actor/pg_loss`` shows one chart rather
      than one empty one;
    * collapse keys that are aliases of each other down to one, so a chart lists
      a series once even when the run wrote it under two names;
    * for ``auto_group`` templates, bin the run's real keys by prefix, which is
      how an unknown ``task_type`` gets a usable page with no template written
      for it.

    Args:
        template: A template dict from :class:`TemplateRegistry`.
        available: Metric keys the gateway reports for this run.
        aliases: ``manifest.metric_aliases``, legacy prefix -> canonical prefix.
            During the dual-write window a run logs *both* spellings of every
            renamed key, so a template that lists both -- which it must, to serve
            runs from either side of the rename -- would otherwise chart one
            series as two identical lines. Passing the aliases lets the canonical
            spelling win and the legacy twin be treated as already charted, which
            also keeps it out of ``unmatched``, where it would read as a metric
            the layout forgot.

    Returns:
        A copy with a ``groups`` list containing only charts that have data, plus
        ``unmatched`` listing logged keys no group claimed -- visible on purpose,
        since a silently dropped metric looks like a missing feature.
    """
    template = copy.deepcopy(template)
    present = set(available)
    claimed: set[str] = set()
    canonical = _canonicaliser(aliases or {}, present)

    groups = []
    for group in template.get("groups") or []:
        charts = []
        for chart in group.get("charts") or []:
            keys = []
            for key in chart.get("keys") or []:
                if key not in present:
                    continue
                # Two declared spellings of one series collapse to whichever the
                # aliases name canonical; `dict.fromkeys` order-preserving dedupe
                # keeps the template's own ordering for everything else.
                keys.append(canonical(key))
            keys = list(dict.fromkeys(keys))
            if not keys:
                continue
            chart = dict(chart, keys=keys)
            # Mark every spelling claimed, not just the one charted, so an
            # uncharted twin does not resurface as an unmatched metric.
            for key in keys:
                claimed.update(_alias_group(key, aliases or {}, present))
            charts.append(chart)
        if charts:
            groups.append(dict(group, charts=charts))

    if template.get("auto_group"):
        groups.extend(_auto_groups(template, sorted(present - claimed), claimed))

    template["groups"] = groups
    template["unmatched"] = sorted(present - claimed)
    template["north_star"] = _resolve_north_star(
        template.get("north_star"), present, canonical
    )
    return template


def _alias_group(key: str, aliases: dict[str, str], present: set[str]) -> set[str]:
    """Every spelling of ``key`` this run actually logged, including itself."""
    spellings = {key}
    for legacy, canon in aliases.items():
        if key.startswith(canon):
            spellings.add(legacy + key[len(canon) :])
        elif key.startswith(legacy):
            spellings.add(canon + key[len(legacy) :])
    return spellings & present | {key}


def _canonicaliser(aliases: dict[str, str], present: set[str]):
    """Build a function mapping a logged key to its preferred spelling.

    Preferred means canonical *and present*: rewriting a legacy key to a
    canonical one the run never wrote would replace a chart that works with one
    that has no data.
    """

    def canonical(key: str) -> str:
        for legacy, canon in aliases.items():
            if key.startswith(legacy):
                candidate = canon + key[len(legacy) :]
                if candidate in present:
                    return candidate
        return key

    return canonical


def _auto_groups(template: dict, keys: list[str], claimed: set[str]) -> list[dict]:
    """Bin keys by declared prefix, longest prefix first."""
    prefixes = sorted(
        template.get("prefix_groups") or [],
        key=lambda entry: -len(str(entry.get("prefix", ""))),
    )
    buckets: dict[str, dict] = {}
    other: list[str] = []

    for key in keys:
        for entry in prefixes:
            prefix = str(entry.get("prefix", ""))
            if prefix and key.startswith(prefix):
                title = entry.get("title") or prefix.rstrip("/")
                bucket = buckets.setdefault(
                    title, {"title": title, "charts": [], "unit": entry.get("unit")}
                )
                bucket["charts"].append({"keys": [key], "title": key})
                claimed.add(key)
                break
        else:
            other.append(key)

    groups = [buckets[title] for title in buckets]
    if other:
        groups.append(
            {
                "title": template.get("other_group_title", "Other"),
                "charts": [{"keys": [key], "title": key} for key in other],
                "collapsed": True,
            }
        )
        claimed.update(other)
    return groups


#: Presentation fields that describe *a metric*, not the north-star slot. These
#: are what must travel with the key when a fallback wins, and what must never be
#: inherited from a candidate that lost.
_METRIC_SEMANTICS = ("label", "format", "goal")


def _north_star_candidates(north_star: dict) -> list[dict]:
    """Normalise either declaration shape into an ordered candidate list.

    Two shapes are accepted:

    ``candidates:``
        The shape to write. Each entry carries its own ``label``/``format``/
        ``goal`` and one or more ``keys`` that share those semantics -- aliases
        for the same quantity, which is the only case where sharing is correct.

    ``key:`` plus optional ``fallback_keys:``
        The original shape. Safe on its own; safe with fallbacks only when every
        fallback happens to mean the same thing as the primary. Because the
        template cannot say whether it does, a fallback from this shape inherits
        *nothing* -- see ``_resolve_north_star`` for why silence beats a guess.
    """
    if north_star.get("candidates"):
        out = []
        for candidate in north_star["candidates"]:
            keys = candidate.get("keys") or (
                [candidate["key"]] if candidate.get("key") else []
            )
            semantics = {
                field: candidate[field]
                for field in _METRIC_SEMANTICS
                if candidate.get(field) is not None
            }
            out.append({"keys": list(keys), **semantics})
        return out

    primary = north_star.get("key")
    declared = {
        field: north_star[field]
        for field in _METRIC_SEMANTICS
        if north_star.get(field) is not None
    }
    out = [{"keys": [primary] if primary else [], **declared}]
    # Each legacy fallback is its own candidate with no inherited semantics.
    out += [{"keys": [key]} for key in (north_star.get("fallback_keys") or [])]
    return out


def _resolve_north_star(
    north_star: dict | None,
    present: set[str],
    canonical=lambda key: key,
) -> dict | None:
    """Pick the first north-star candidate that has data, with *its* semantics.

    A headline metric with no data is worse than none: an empty hero number reads
    as a broken run rather than as a metric this run does not log.

    The winner's ``label``/``format``/``goal`` come from the winning candidate,
    never from the one that was declared first. Carrying the primary's
    presentation onto a fallback is how an SFT run that logs no eval metric ends
    up rendering ``train/loss`` as "Eval accuracy", formatted as a percentage,
    with ``goal: maximize`` -- so a *falling* loss, which is the run working,
    draws a north star that reads as getting worse. Mislabelling is bad; an
    inverted goal inverts every trend verdict derived from it.

    A candidate that declares no ``format``/``goal`` gets none, and the UI falls
    back to its own neutral defaults. That is deliberate: for the legacy
    ``fallback_keys`` shape the template never said what those keys mean, and a
    plain number under an honest name is the correct rendering of "we do not
    know", where a borrowed percentage is a confident lie.

    The winner is canonicalised for the same reason chart keys are: the headline
    number and the chart of the same metric must request the same key, or the two
    can disagree once the legacy spelling stops being written.
    """
    if not north_star:
        return None

    slot = {
        field: value
        for field, value in north_star.items()
        if field
        not in {*_METRIC_SEMANTICS, "key", "keys", "fallback_keys", "candidates"}
    }
    candidates = _north_star_candidates(north_star)

    for candidate in candidates:
        for key in candidate["keys"]:
            if key and key in present:
                semantics = {
                    field: candidate[field]
                    for field in _METRIC_SEMANTICS
                    if candidate.get(field) is not None
                }
                return {**slot, **semantics, "key": canonical(key), "resolved": True}

    # Nothing resolved. Report the *first* candidate's identity, since that is the
    # metric this template would have led with, and the UI says it is missing.
    first = candidates[0] if candidates else {"keys": []}
    unresolved = {
        field: first[field]
        for field in _METRIC_SEMANTICS
        if first.get(field) is not None
    }
    keys = first.get("keys") or []
    return {**slot, **unresolved, "key": keys[0] if keys else None, "resolved": False}


def _resolve(name: str, raw: dict[str, dict], _seen: tuple[str, ...] = ()) -> dict:
    """Apply single-inheritance ``extends``, child overriding parent.

    Groups merge by title so a child can add charts to an inherited group instead
    of restating it. A cycle stops resolution and returns the template as
    written, which keeps a bad template from hanging the server.
    """
    data = copy.deepcopy(raw[name])
    parent_name = data.pop("extends", None)
    if not parent_name:
        return data
    if parent_name in _seen or parent_name not in raw:
        logger.error("Template %s has a bad or cyclic 'extends': %s", name, parent_name)
        return data

    parent = _resolve(parent_name, raw, _seen + (name,))
    merged = {**parent, **data}

    by_title = {group.get("title"): group for group in parent.get("groups") or []}
    for group in data.get("groups") or []:
        title = group.get("title")
        if title in by_title:
            base = by_title[title]
            by_title[title] = {
                **base,
                **group,
                "charts": (base.get("charts") or []) + (group.get("charts") or []),
            }
        else:
            by_title[title] = group
    merged["groups"] = list(by_title.values())
    return merged


_EMPTY_TEMPLATE = {
    "name": "empty",
    "task_types": [],
    "groups": [],
    "auto_group": True,
    "prefix_groups": [],
    "step_axis_label": "Step",
    # Reached only when the template directory itself failed to load, so nothing is
    # known about the run's kind. Left open for the same reason as the fallback:
    # the tab appears only if clips exist.
    "has_media_view": True,
}
