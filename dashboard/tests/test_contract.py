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

"""The contract between the training side and this one.

``docs/schemas/run.v2.schema.json`` is the only shared artifact. The training side
writes snapshots from a dataclass, this side reads them into pydantic models, and
neither imports the other. Without a test that both sides' behaviour is pinned to
one document, the two models drift silently and the drift only shows up as a blank
dashboard.

So this file asserts three separate things:

1. every committed fixture validates against the schema (the training side's
   ``tests/unit_tests/test_run_state_contract.py`` asserts the same thing over the
   same bytes);
2. every fixture parses into :class:`RunSnapshot` with its values intact;
3. the model's field set covers the schema's property set -- the check that
   actually catches drift, since 1 and 2 both pass when the training side adds a
   field this reader ignores.
"""

from __future__ import annotations

import json

import pytest
from conftest import fixture_names, load_fixture

from rlinf_dashboard.models import (
    Progress,
    RunSnapshot,
    RunState,
    Timing,
)

jsonschema = pytest.importorskip("jsonschema", reason="install dashboard[test]")


def test_fixtures_exist():
    """A vacuously-passing parametrized suite is worse than a failing one.

    Every test below is parametrized over the fixture directory, so if the path
    in ``conftest`` ever stops resolving, all of them would report as passed with
    zero cases. This is the guard against that.
    """
    names = fixture_names()
    assert len(names) >= 5, f"expected the shared run fixtures, found {names}"


@pytest.mark.parametrize("name", fixture_names())
def test_fixture_validates_against_the_shared_schema(schema, name):
    """The same assertion the training-side suite makes, over the same bytes."""
    jsonschema.validate(instance=load_fixture(name), schema=schema)


@pytest.mark.parametrize("name", fixture_names())
def test_fixture_parses_into_the_read_model(name):
    payload = load_fixture(name)
    snapshot = RunSnapshot.model_validate(payload)

    assert snapshot.run_id == payload["run_id"]
    assert snapshot.state.value == payload["state"]
    assert snapshot.schema_version == 2
    assert snapshot.progress.step == payload["progress"]["step"]


def test_read_model_covers_every_schema_property(schema):
    """The real drift guard.

    A field the training side writes and this model lacks is dropped silently --
    ``extra="ignore"`` is deliberate leniency for forward compatibility, and it
    means a missing field cannot fail a parse. Comparing the two field sets is
    the only place that omission becomes visible.
    """
    schema_props = set(schema["properties"])
    model_fields = set(RunSnapshot.model_fields)
    missing = schema_props - model_fields
    assert not missing, (
        f"RunSnapshot is missing schema properties {sorted(missing)}; "
        f"extra='ignore' means these parse cleanly and vanish."
    )


@pytest.mark.parametrize(
    ("field", "model"),
    [("progress", Progress), ("timing", Timing)],
)
def test_nested_models_cover_their_schema_properties(schema, field, model):
    schema_props = set(schema["properties"][field]["properties"])
    missing = schema_props - set(model.model_fields)
    assert not missing, f"{model.__name__} is missing {sorted(missing)}"


def test_state_has_no_stalled_value(schema):
    """``stalled`` is absent by design, on both sides.

    A process that was ``kill -9``'d cannot record its own death, so no write-side
    value could ever mean "stalled" honestly. Liveness is derived on read instead
    (``health.py``). If this ever fails, someone has re-introduced a state the
    writer cannot truthfully set.
    """
    allowed = set(schema["properties"]["state"]["enum"])
    assert "stalled" not in allowed
    assert allowed == {state.value for state in RunState}


def test_health_is_not_a_snapshot_field(schema):
    """Health is computed on read and must never be persisted.

    A persisted verdict is a verdict that goes stale: the moment the writer dies,
    the last thing it wrote was "healthy", and that would be what the dashboard
    shows forever.
    """
    assert "health" not in schema["properties"]
    assert "health" not in RunSnapshot.model_fields


# ------------------------------------------------------------------ specific shapes


def test_async_components_fixture_reports_concurrency():
    """The fixture that justifies ``components`` existing at all.

    Async runners start env, rollout and actor before the ``while`` loop and join
    after it, so all three are active for the entire run. A single scalar ``phase``
    is not merely imprecise there -- it is wrong.
    """
    snapshot = RunSnapshot.model_validate(load_fixture("async_components"))
    active = {name for name, comp in snapshot.components.items() if comp.active}
    assert len(active) >= 2, f"expected concurrent components, got {active}"
    for name in active:
        assert snapshot.components[name].since is not None, (
            f"{name} is active but has no 'since'; the UI cannot show how long"
        )


def test_reasoning_fixture_labels_its_step_axis_honestly():
    """A7: reasoning's step is a minibatch index, and says so.

    reasoning_runner.py advances its counter per minibatch. The label exists so
    the frontend can name the x-axis correctly and refuse to compare a reasoning
    curve against an embodied one, whose step is a whole RL iteration. This is a
    label on existing behaviour, not a change to it.
    """
    snapshot = RunSnapshot.model_validate(load_fixture("reasoning_minibatch"))
    assert snapshot.progress.step_semantics == "minibatch"


def test_failed_fixture_says_why():
    """The schema requires ``exit`` on failure; check the payload is usable."""
    snapshot = RunSnapshot.model_validate(load_fixture("failed"))
    assert snapshot.state is RunState.FAILED
    assert snapshot.exit is not None
    assert snapshot.exit.reason


def test_running_fixture_carries_no_exit_info():
    snapshot = RunSnapshot.model_validate(load_fixture("running"))
    assert snapshot.exit is None


def test_finished_fixture_has_a_checkpoint_with_resume_fields():
    """A6: resume is structured fields, not a pre-baked shell command.

    A command string bakes in paths and flags that go stale the moment anything
    about the launch changes. These three fields let the frontend render a command
    that is correct at display time.
    """
    snapshot = RunSnapshot.model_validate(load_fixture("finished"))
    checkpoint = snapshot.latest_checkpoint
    assert checkpoint is not None
    assert checkpoint.resume_dir
    assert checkpoint.entry_script


# ------------------------------------------------------------------ leniency


def test_unknown_fields_are_ignored_not_fatal():
    """Forward compatibility: a newer writer must not break an older reader.

    The training side and the dashboard are installed into different venvs and
    upgrade independently, so this ordering is normal, not exceptional. Refusing
    to parse would black out the whole page over one unrecognised key.
    """
    payload = load_fixture("running")
    payload["a_field_from_the_future"] = {"nested": [1, 2, 3]}
    payload["progress"]["tokens"] = 12345

    snapshot = RunSnapshot.model_validate(payload)
    assert snapshot.state is RunState.RUNNING
    assert not hasattr(snapshot, "a_field_from_the_future")


def test_a_missing_optional_block_still_parses():
    """An older writer must not break a newer reader either.

    v1 snapshots have no ``components`` and no ``step_semantics``; those runs are
    still worth listing.
    """
    payload = load_fixture("running")
    for key in ("components", "phase", "last_metric_at", "latest_checkpoint"):
        payload.pop(key, None)
    payload["progress"].pop("step_semantics", None)

    snapshot = RunSnapshot.model_validate(payload)
    assert snapshot.components == {}
    assert snapshot.progress.step_semantics is None


def test_progress_fraction_is_bounded_and_none_without_a_horizon():
    """``max_steps`` can legitimately be null or be exceeded.

    Null when the runner has not resolved its horizon yet; exceeded when a resumed
    run passes the cap it was launched with. A progress bar past 100% reads as a
    bug in the dashboard, so it is clamped here.
    """
    assert Progress(step=5, max_steps=None).fraction is None
    assert Progress(step=5, max_steps=0).fraction is None
    assert Progress(step=5, max_steps=10).fraction == 0.5
    assert Progress(step=50, max_steps=10).fraction == 1.0


def test_snapshot_round_trips_through_json():
    """What the API returns must be re-readable.

    The SSE loop serializes these models and the frontend parses them, so a field
    that cannot survive a round trip is a field the UI never sees.
    """
    snapshot = RunSnapshot.model_validate(load_fixture("finished"))
    again = RunSnapshot.model_validate(json.loads(snapshot.model_dump_json()))
    assert again.state is snapshot.state
    assert again.heartbeat_at == snapshot.heartbeat_at
    assert again.progress.step == snapshot.progress.step
