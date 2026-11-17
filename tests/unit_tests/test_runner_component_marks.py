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

"""The pipelined embodied loop must report components, not a training scope.

``EmbodiedRunner.run_pipeline`` overlaps actor training with rollout on purpose
-- that overlap *is* the pipelining. So the two ways of reporting it are not
interchangeable:

* A ``self.timer("actor_training")`` scope around it would enclose the
  ``generate_rollouts`` scope, and both would be reported as durations that sum
  into ``time/step``. The same wall-clock second would be counted twice, and the
  phase-durations chart -- whose whole premise is that its bars add up to the
  step -- would show a step longer than the step.
* ``reporter.component_enter/exit`` describes concurrency without claiming
  disjoint time, which is exactly what this loop needs.

These assertions read the call sites because importing and running the driver
loop requires a live Ray cluster.
"""

from __future__ import annotations

import ast
import os

RUNNER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "rlinf",
    "runners",
    "embodied_runner.py",
)


def _method(name: str) -> ast.FunctionDef:
    """Return the AST of one ``EmbodiedRunner`` method."""
    with open(RUNNER, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "EmbodiedRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"EmbodiedRunner.{name} not found in {RUNNER}")


def _reporter_calls(node: ast.AST, method: str) -> list[str]:
    """Every literal argument passed to ``self.reporter.<method>(...)``."""
    found = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Attribute):
            continue
        if sub.func.attr != method:
            continue
        owner = sub.func.value
        if (
            isinstance(owner, ast.Attribute)
            and owner.attr == "reporter"
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "self"
        ):
            if sub.args and isinstance(sub.args[0], ast.Constant):
                found.append(sub.args[0].value)
    return found


def _timer_scopes(node: ast.AST) -> list[str]:
    """Every literal scope name passed to ``self.timer(...)``."""
    found = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Attribute):
            continue
        if sub.func.attr != "timer":
            continue
        if not (isinstance(sub.func.value, ast.Name) and sub.func.value.id == "self"):
            continue
        if sub.args and isinstance(sub.args[0], ast.Constant):
            found.append(sub.args[0].value)
    return found


def test_pipelined_loop_marks_all_three_concurrent_components():
    """A missing mark makes a live component look idle for the whole run."""
    entered = _reporter_calls(_method("run_pipeline"), "component_enter")
    assert set(entered) == {"env", "rollout", "actor"}


def test_every_component_enter_has_an_exit():
    """An unbalanced mark leaves the component `active` forever.

    The run would then report a component still working after the loop ended --
    which reads as a hung worker rather than a bookkeeping slip.
    """
    node = _method("run_pipeline")
    assert sorted(_reporter_calls(node, "component_enter")) == sorted(
        _reporter_calls(node, "component_exit")
    )


def test_pipelined_loop_does_not_scope_actor_training():
    """The double-count this file exists to prevent.

    In `run_pipeline` actor training is launched before the rollout wait and
    joined after it, so a timer scope around it would contain the
    `generate_rollouts` scope. Both feed `time/step`, so the phase bars would
    exceed the step they are meant to decompose.
    """
    scopes = _timer_scopes(_method("run_pipeline"))
    assert "actor_training" not in scopes
    # The scopes it does keep are genuinely disjoint stretches of the step.
    assert "generate_rollouts" in scopes
    assert "sync_weights" in scopes


def test_the_synchronous_loop_does_scope_actor_training():
    """The contrast that makes the rule above a rule and not an omission.

    `_run_impl` waits for training inside its own scope, so there the duration
    is real and disjoint -- and there it must be a timer scope, since that is
    what feeds `time/actor_training`.
    """
    scopes = _timer_scopes(_method("_run_impl"))
    assert "actor_training" in scopes
    assert "generate_rollouts" in scopes


def test_the_synchronous_loop_reports_no_components():
    """Nothing overlaps in `_run_impl`, so `phase` already describes it fully.

    Marking components there would claim concurrency that does not exist.
    """
    assert _reporter_calls(_method("_run_impl"), "component_enter") == []
