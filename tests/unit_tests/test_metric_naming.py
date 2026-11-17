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

"""Tests for canonical metric key naming.

The same quantity is spelled differently by different runners, so keys are
normalized at the single write entry point. The migration is a *dual write*:
both spellings are emitted so existing wandb and TensorBoard dashboards remain
compatible.

The properties that matter:

* the legacy key survives -- dropping it is the breaking change this design
  exists to avoid;
* a caller's own canonical value is never overwritten by an alias;
* the deprecation warning fires once per prefix, not per key or per step, since
  these are logged every step and a per-call warning would bury the log;
* :func:`alias_table` stays serializable, because the dashboard reads the
  mapping out of ``manifest.json`` rather than importing ``rlinf``.
"""

import json

import pytest

from rlinf.utils import metric_naming
from rlinf.utils.metric_naming import (
    LEGACY_TO_CANONICAL,
    alias_table,
    canonical_key,
    dual_write,
)


@pytest.fixture(autouse=True)
def reset_warned_prefixes():
    """The warn-once set is module state; isolate it between tests."""
    metric_naming._warned_prefixes.clear()
    yield
    metric_naming._warned_prefixes.clear()


# ---------------------------------------------------------------- canonical_key


def test_legacy_prefix_is_rewritten():
    assert canonical_key("actor/training/loss") == "train/actor/loss"
    assert canonical_key("critic/training/loss") == "train/critic/loss"


def test_unknown_key_passes_through():
    assert canonical_key("env/success_once") == "env/success_once"
    assert canonical_key("train/actor/loss") == "train/actor/loss"
    assert canonical_key("") == ""


def test_only_a_prefix_match_counts():
    """A legacy prefix appearing mid-key is not an alias."""
    assert canonical_key("x/actor/training/loss") == "x/actor/training/loss"


def test_suffix_is_preserved_including_extra_segments():
    assert (
        canonical_key("actor/training/grad_norm/layer_0")
        == "train/actor/grad_norm/layer_0"
    )


def test_longest_prefix_wins(monkeypatch):
    """A more specific legacy prefix must not be shadowed by a shorter one."""
    monkeypatch.setitem(LEGACY_TO_CANONICAL, "actor/", "generic/")
    try:
        assert canonical_key("actor/training/loss") == "train/actor/loss"
        assert canonical_key("actor/other") == "generic/other"
    finally:
        LEGACY_TO_CANONICAL.pop("actor/", None)


# ------------------------------------------------------------------ dual_write


def test_dual_write_keeps_the_legacy_key():
    """Removing it would break every existing dashboard, which is the whole
    reason this is a dual write rather than a rename."""
    result = dual_write({"actor/training/loss": 1.5})

    assert result == {"actor/training/loss": 1.5, "train/actor/loss": 1.5}


def test_dual_write_does_not_mutate_the_input():
    data = {"actor/training/loss": 1.5}
    dual_write(data)
    assert data == {"actor/training/loss": 1.5}


def test_dual_write_leaves_non_aliased_keys_alone():
    data = {"env/success_once": 0.4, "time/step": 2.0}
    assert dual_write(data) == data


def test_dual_write_never_overwrites_a_caller_supplied_canonical_key():
    """If the producer already emits both, the canonical value it chose wins."""
    result = dual_write({"actor/training/loss": 1.5, "train/actor/loss": 99.0})
    assert result["train/actor/loss"] == 99.0


def test_dual_write_handles_empty_and_falsy_input():
    assert dual_write({}) == {}
    assert dual_write(None) is None


def test_dual_write_handles_mixed_batches():
    result = dual_write(
        {
            "actor/training/loss": 1.0,
            "critic/training/value_loss": 2.0,
            "env/success_once": 0.3,
        }
    )
    assert result["train/actor/loss"] == 1.0
    assert result["train/critic/value_loss"] == 2.0
    assert result["env/success_once"] == 0.3
    assert len(result) == 5


def test_dual_write_preserves_non_numeric_values():
    """Values are passed through untouched; only keys are the concern here."""
    result = dual_write({"actor/training/note": "warmup"})
    assert result["train/actor/note"] == "warmup"


# ------------------------------------------------------------- warn once only


def test_deprecation_warning_fires_once_per_prefix():
    """Logged every step, so a per-call warning would bury the log."""
    warnings = []

    class _Recorder:
        def warning(self, message):
            warnings.append(message)

    import rlinf.utils.metric_naming as module

    original = module.get_logger
    module.get_logger = lambda: _Recorder()
    try:
        for _ in range(50):
            dual_write({"actor/training/loss": 1.0, "actor/training/entropy": 0.1})
    finally:
        module.get_logger = original

    assert len(warnings) == 1
    assert "actor/training/" in warnings[0]
    # The message must name the replacement, not just complain.
    assert "train/actor/" in warnings[0]


def test_distinct_prefixes_each_warn_once():
    warnings = []

    class _Recorder:
        def warning(self, message):
            warnings.append(message)

    import rlinf.utils.metric_naming as module

    original = module.get_logger
    module.get_logger = lambda: _Recorder()
    try:
        for _ in range(10):
            dual_write({"actor/training/loss": 1.0, "critic/training/value_loss": 2.0})
    finally:
        module.get_logger = original

    assert len(warnings) == 2


def test_a_failing_logger_does_not_break_the_metric_path(monkeypatch):
    """Warning about a deprecated name must not cost a run its metrics.

    ``get_logger()`` resolves the worker logger singleton, so it is not
    guaranteed to succeed in every process that logs metrics -- and this runs on
    the hot path once per step.
    """
    import rlinf.utils.metric_naming as module

    def explode():
        raise RuntimeError("no worker logger in this process")

    monkeypatch.setattr(module, "get_logger", explode)

    result = dual_write({"actor/training/loss": 1.0})
    assert result["train/actor/loss"] == 1.0


# ----------------------------------------------------------------- alias table


def test_alias_table_is_a_copy():
    """The manifest writer must not be able to mutate the module's table."""
    table = alias_table()
    table["bogus/"] = "nope/"
    assert "bogus/" not in LEGACY_TO_CANONICAL


def test_alias_table_is_json_serializable():
    """It travels inside ``manifest.json`` so the dashboard needs no rlinf import."""
    assert json.loads(json.dumps(alias_table())) == dict(LEGACY_TO_CANONICAL)


def test_alias_table_entries_are_prefixes():
    """Prefix matching is the contract; a bare key would silently never match."""
    for legacy, canonical in alias_table().items():
        assert legacy.endswith("/"), legacy
        assert canonical.endswith("/"), canonical


def test_canonical_targets_are_not_themselves_legacy():
    """A canonical prefix that is also a legacy key would need two passes to
    resolve, and ``canonical_key`` only makes one."""
    for canonical in LEGACY_TO_CANONICAL.values():
        assert canonical not in LEGACY_TO_CANONICAL
