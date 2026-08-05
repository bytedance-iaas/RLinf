#!/usr/bin/env python3
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

"""Validate DESIGN.md against the google-labs-code/design.md specification.

Why a local script rather than ``npx @google/design.md lint``: the upstream CLI
is at ``alpha`` and would be a network dependency of this repo's checks. The
rules it enforces are a fixed, published list, and the ones that can actually
break this file -- a broken ``{token.ref}``, sections out of order, a duplicate
heading, a mistyped component property -- are cheap to check directly. This keeps
the design system verifiable in CI with no npm install.

Severities follow the spec's table: unknown headings and token names are
preserved silently, unknown component properties warn, duplicate headings and
unresolvable references are errors.

Usage:
    python scripts/lint_design_md.py [path]     (default: DESIGN.md)
"""

from __future__ import annotations

import os
import re
import sys

import yaml

#: The eight prose sections, in the order the spec fixes. Any may be omitted;
#: those present must appear in this relative order.
CANONICAL_SECTIONS = [
    "Overview",
    "Colors",
    "Typography",
    "Layout",
    "Elevation & Depth",
    "Shapes",
    "Components",
    "Do's and Don'ts",
]

#: Accepted alternative spellings, normalised before the order check.
SECTION_ALIASES = {
    "Brand & Style": "Overview",
    "Layout & Spacing": "Layout",
    "Elevation": "Elevation & Depth",
}

TYPOGRAPHY_KEYS = {
    "fontFamily",
    "fontSize",
    "fontWeight",
    "lineHeight",
    "letterSpacing",
    "fontFeature",
    "fontVariation",
}

COMPONENT_KEYS = {
    "backgroundColor",
    "textColor",
    "typography",
    "rounded",
    "padding",
    "size",
    "height",
    "width",
}

#: Only these units are legal in a Dimension.
_DIMENSION = re.compile(r"-?[\d.]+(px|em|rem)")


def _split(raw: str) -> tuple[dict, str, str]:
    """Return (front matter, front-matter text, prose body).

    The front matter must open and close with a line of exactly ``---``.
    """
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if match is None:
        raise SystemExit("ERROR   front matter delimiters (--- ... ---) not found")
    return yaml.safe_load(match.group(1)) or {}, match.group(1), match.group(2)


def _check_sections(body: str, errors: list[str]) -> list[str]:
    """Duplicate headings are an error; known sections must keep spec order."""
    headings = re.findall(r"^## (.+)$", body, re.M)
    if len(headings) != len(set(headings)):
        errors.append(f"duplicate '##' heading (spec: reject the file): {headings}")

    normalised = [SECTION_ALIASES.get(h, h) for h in headings]
    known = [h for h in normalised if h in CANONICAL_SECTIONS]
    if known != sorted(known, key=CANONICAL_SECTIONS.index):
        errors.append(f"sections out of spec order: {known}")
    return normalised


def _resolve(front: dict, path: str):
    """Walk a dotted ``{a.b}`` reference path, or return ``None``."""
    node = front
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _check_dimension(where: str, value, warnings: list[str]) -> None:
    """Flag a string dimension carrying a unit the spec does not permit.

    Bare numbers pass: the spec allows unitless values in ``spacing`` (column
    counts, ratios) and for ``lineHeight`` as a font-size multiplier.
    """
    if isinstance(value, (int, float)):
        return
    if isinstance(value, str) and not value.startswith("{"):
        if not _DIMENSION.fullmatch(value):
            warnings.append(f"{where}: {value!r} is not a px/em/rem dimension")


#: Which generated custom-property name a token of each group turns into, so the
#: orphan check can look for actual consumers instead of only prose mentions.
#: Mirrors ``gen_tokens.PREFIXES``; typography expands to several properties per
#: token (``--type-body-md-size``, ``-weight``, ...) and so matches by prefix.
_USED_AS = {
    "colors": lambda key: {f"--color-{key}"},
    "rounded": lambda key: {f"--radius-{key}"},
}

_CUSTOM_PROPERTY = re.compile(r"--[a-z0-9]+(?:-[a-z0-9]+)*")


def _properties_used_near(path: str) -> set[str] | None:
    """Collect every custom property named in the sibling ``src/`` tree.

    ``tokens.css`` is skipped because it is the generated definition site: every
    token appears there by construction, so including it would make the orphan
    check vacuous. Returns ``None`` when there is no ``src/`` to read, which is
    the case when linting a DESIGN.md outside a frontend tree -- the caller then
    falls back to prose mentions alone rather than reporting every token orphaned.
    """
    src = os.path.join(os.path.dirname(os.path.abspath(path)), "src")
    if not os.path.isdir(src):
        return None
    used: set[str] = set()
    for dirpath, _, filenames in os.walk(src):
        for filename in filenames:
            if filename == "tokens.css":
                continue
            if not filename.endswith((".css", ".ts", ".tsx")):
                continue
            with open(os.path.join(dirpath, filename), encoding="utf-8") as handle:
                used.update(_CUSTOM_PROPERTY.findall(handle.read()))
    return used


def main(path: str) -> int:
    """Lint one DESIGN.md; return 1 if any error-severity rule fired."""
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()

    front, front_text, body = _split(raw)
    errors: list[str] = []
    warnings: list[str] = []

    if not front.get("name"):
        errors.append("front matter is missing the required 'name'")
    if "primary" not in (front.get("colors") or {}):
        errors.append("colors.primary is expected")

    normalised = _check_sections(body, errors)
    known = [h for h in normalised if h in CANONICAL_SECTIONS]
    unknown = [h for h in normalised if h not in CANONICAL_SECTIONS]
    print(f"sections     {' -> '.join(known)}")
    if unknown:
        print(f"extra        {unknown} (preserved; not an error)")

    # broken-ref, the one error-severity token rule.
    refs = set(re.findall(r'"\{([^}]+)\}"', front_text))
    for ref in sorted(refs):
        if _resolve(front, ref) is None:
            errors.append(f"broken-ref: {{{ref}}} does not resolve")
    print(f"references   {len(refs)} checked")

    for group in ("rounded", "spacing"):
        for key, value in (front.get(group) or {}).items():
            _check_dimension(f"{group}.{key}", value, warnings)

    for name, spec in (front.get("typography") or {}).items():
        for key in spec:
            if key not in TYPOGRAPHY_KEYS:
                errors.append(f"typography.{name}: unknown key {key!r}")
        for key in ("fontSize", "letterSpacing"):
            if key in spec:
                _check_dimension(f"typography.{name}.{key}", spec[key], warnings)

    for name, spec in (front.get("components") or {}).items():
        for key in spec:
            if key not in COMPONENT_KEYS:
                warnings.append(f"unknown-key: components.{name}.{key}")

    # orphaned-tokens: defined, but reachable from nothing. A token is in use if
    # another token references it, the prose names it, *or* the app consumes the
    # custom property it generates. That last source is the one that matters:
    # checking prose alone flagged eleven live tokens (the eight series colours
    # are read from src/lib/series.ts, three type levels only from app.css), and a
    # check that is eleven-twelfths false alarm is a check nobody reads.
    used = _properties_used_near(path)
    for group in ("colors", "typography", "rounded"):
        for key in front.get(group) or {}:
            if f"{group}.{key}" in refs or key in body:
                continue
            if used is not None:
                if group == "typography":
                    # One token becomes several properties; any of them counts.
                    if any(name.startswith(f"--type-{key}-") for name in used):
                        continue
                elif _USED_AS[group](key) & used:
                    continue
            warnings.append(f"orphaned-token: {group}.{key}")

    print(
        "tokens       "
        f"{len(front.get('colors') or {})} colors, "
        f"{len(front.get('typography') or {})} typography, "
        f"{len(front.get('rounded') or {})} rounded, "
        f"{len(front.get('spacing') or {})} spacing, "
        f"{len(front.get('components') or {})} components"
    )
    print()

    for message in errors:
        print(f"ERROR   {message}")
    for message in warnings:
        print(f"WARN    {message}")
    if not errors and not warnings:
        print("clean: no errors, no warnings")
    elif not errors:
        print(f"\n{len(warnings)} warning(s), 0 errors")

    return 1 if errors else 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "DESIGN.md"
    if not os.path.exists(target):
        raise SystemExit(f"ERROR   no such file: {target}")
    raise SystemExit(main(target))
