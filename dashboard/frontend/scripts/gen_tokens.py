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

"""Generate ``src/styles/tokens.css`` from ``DESIGN.md``.

DESIGN.md is the normative source for every colour, type level and dimension in
this app. Hand-maintaining a parallel CSS file guarantees the two drift, and a
drifted design system is worse than none: the prose says one thing, the pixels
another, and nobody knows which is current.

So the CSS is generated, committed, and checked. ``--check`` re-generates into
memory and diffs, which is what CI runs -- editing tokens.css by hand then fails
with a message pointing back at DESIGN.md.

This is the spec's ``export`` step done locally rather than through
``npx @google/design.md export``, for the same reason the linter is local: no
network dependency in a repo check, for a transform this small.

Usage:
    python scripts/gen_tokens.py              # write src/styles/tokens.css
    python scripts/gen_tokens.py --check      # exit 1 if the file is stale
"""

from __future__ import annotations

import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DESIGN_MD = os.path.join(ROOT, "DESIGN.md")
OUTPUT = os.path.join(ROOT, "src", "styles", "tokens.css")

#: CSS custom-property prefix per token group. ``spacing`` becomes ``--space-``
#: because ``--spacing-md`` reads worse than ``--space-md`` in rules.
PREFIXES = {
    "colors": "color",
    "rounded": "radius",
    "spacing": "space",
}


def _load() -> dict:
    with open(DESIGN_MD, encoding="utf-8") as handle:
        raw = handle.read()
    match = re.match(r"^---\n(.*?)\n---\n", raw, re.S)
    if match is None:
        raise SystemExit("DESIGN.md has no front matter")
    return yaml.safe_load(match.group(1)) or {}


def _deref(value, front: dict):
    """Resolve a ``{group.name}`` reference to a ``var(--...)`` expression.

    Emitting a ``var()`` rather than the literal keeps the cascade intact: a
    component token that points at ``colors.primary`` should follow any override
    of ``--color-primary``, not freeze the value it had at generation time.
    """
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\{([^}]+)\}", value)
    if match is None:
        return value
    path = match.group(1)
    group, _, name = path.partition(".")
    if group == "typography":
        # Typography is a bundle of properties, not one value; components
        # referencing it are handled by emitting a font shorthand group instead.
        return None
    prefix = PREFIXES.get(group)
    if prefix is None:
        return None
    return f"var(--{prefix}-{name})"


def _dimension(value) -> str:
    """Render a dimension, leaving unitless numbers unitless.

    ``spacing.overview-columns: 4`` is a column count, not 4px; emitting ``4px``
    would produce a grid with four pixel-wide columns.
    """
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def render(front: dict) -> str:
    """Render base tokens and any named theme overrides as CSS properties."""
    lines: list[str] = [
        "/* GENERATED FROM DESIGN.md -- do not edit.",
        " *",
        " * Regenerate with:  python scripts/gen_tokens.py",
        " * Verify with:      python scripts/gen_tokens.py --check",
        " *",
        " * Every value here has a rationale in DESIGN.md's prose. Change the",
        " * token there, regenerate, and the reason travels with the value.",
        " */",
        "",
        ":root {",
        "  color-scheme: dark;",
    ]

    def section(title: str) -> None:
        lines.append("")
        lines.append(f"  /* {title} */")

    section("Colors")
    for name, value in (front.get("colors") or {}).items():
        lines.append(f"  --color-{name}: {value};")

    section("Radii")
    for name, value in (front.get("rounded") or {}).items():
        lines.append(f"  --radius-{name}: {_dimension(value)};")

    section("Spacing and sizes")
    for name, value in (front.get("spacing") or {}).items():
        lines.append(f"  --space-{name}: {_dimension(value)};")

    section("Typography")
    families: dict[str, str] = {}
    for name, spec in (front.get("typography") or {}).items():
        family = spec.get("fontFamily")
        if family:
            # One --font-* per distinct family, so a stack change is one edit.
            slug = re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-")
            families.setdefault(slug, family)
        for prop, css in (
            ("fontSize", "size"),
            ("fontWeight", "weight"),
            ("lineHeight", "leading"),
            ("letterSpacing", "tracking"),
            ("fontFeature", "feature"),
        ):
            if prop in spec:
                lines.append(f"  --type-{name}-{css}: {spec[prop]};")
    lines.append("")
    for slug, family in families.items():
        # A generic fallback keeps numbers monospaced even if the woff2 fails to
        # load, which matters more than matching the exact face.
        fallback = (
            "ui-monospace, SFMono-Regular, Menlo, monospace"
            if "mono" in slug
            else "system-ui, -apple-system, Segoe UI, sans-serif"
        )
        lines.append(f"  --font-{slug}: '{family}', {fallback};")

    section("Component tokens")
    for name, spec in (front.get("components") or {}).items():
        for prop, css in (
            ("backgroundColor", "bg"),
            ("textColor", "fg"),
            ("rounded", "radius"),
            ("padding", "pad"),
            ("height", "height"),
            ("width", "width"),
            ("size", "size"),
        ):
            if prop not in spec:
                continue
            resolved = _deref(spec[prop], front)
            if resolved is None:
                resolved = spec[prop]
            lines.append(f"  --{name}-{css}: {resolved};")

    lines.append("}")

    # The base palette is the dark, no-JavaScript fallback. Named theme blocks
    # override only colours, so typography, spacing and component decisions stay
    # identical when the operator switches appearance. Keeping the override in
    # this generated file also makes DESIGN.md the single source of truth for
    # both palettes rather than hiding half the system in handwritten CSS.
    base_color_names = set((front.get("colors") or {}).keys())
    for theme_name, theme in (front.get("themes") or {}).items():
        colors = theme.get("colors") or {}
        if not colors:
            raise SystemExit(f"Theme {theme_name!r} has no color overrides")
        theme_color_names = set(colors.keys())
        if theme_color_names != base_color_names:
            missing = ", ".join(sorted(base_color_names - theme_color_names)) or "none"
            extra = ", ".join(sorted(theme_color_names - base_color_names)) or "none"
            raise SystemExit(
                f"Theme {theme_name!r} must override the complete base palette "
                f"(missing: {missing}; extra: {extra})"
            )
        lines.extend(["", f':root[data-theme="{theme_name}"] {{'])
        lines.append(f"  color-scheme: {theme_name};")
        for name, value in colors.items():
            lines.append(f"  --color-{name}: {value};")
        lines.append("}")

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """Write or verify ``tokens.css``; return a process exit code."""
    css = render(_load())

    if "--check" in argv:
        if not os.path.exists(OUTPUT):
            print(f"MISSING {OUTPUT}\nRun: python scripts/gen_tokens.py")
            return 1
        with open(OUTPUT, encoding="utf-8") as handle:
            current = handle.read()
        if current != css:
            print(
                f"STALE   {os.path.relpath(OUTPUT, ROOT)} does not match DESIGN.md.\n"
                "        Edit the token in DESIGN.md (where its rationale lives), "
                "then run:\n"
                "          python scripts/gen_tokens.py"
            )
            return 1
        print(f"ok: {os.path.relpath(OUTPUT, ROOT)} is current with DESIGN.md")
        return 0

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        handle.write(css)
    count = css.count("\n  --")
    print(f"wrote {os.path.relpath(OUTPUT, ROOT)} ({count} custom properties)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
