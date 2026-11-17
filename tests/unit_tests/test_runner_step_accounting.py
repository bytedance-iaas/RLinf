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

"""Observability must not distort or slow the thing it observes.

* **The step loop gains no blocking call for diagnostics.** ``_advance_env_step``
  labels videos without synchronizing the whole environment group.

* **``step_duration_s`` covers the whole step.** The async runners reported
  progress after weight sync while excluding eval and checkpoint work.

Read as source, like ``test_runner_env_step.py``: these runners import ray at
module load and a driver loop needs a live cluster.
"""

from __future__ import annotations

import ast
import os

import pytest

RUNNERS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "rlinf",
    "runners",
)

ASYNC_LOOPS = [
    ("async_embodied_runner.py", "AsyncEmbodiedRunner"),
    ("async_ppo_embodied_runner.py", "AsyncPPOEmbodiedRunner"),
]


def _module(name: str) -> ast.Module:
    with open(os.path.join(RUNNERS, name), encoding="utf-8") as handle:
        return ast.parse(handle.read())


def _method(module: str, cls: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(_module(module)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found in {module}")


def _is_wait_on(node: ast.AST, receiver_attr: str) -> bool:
    """Whether ``node`` is ``self.<receiver_attr>.<anything>(...).wait(...)``."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    if node.func.attr != "wait":
        return False
    inner = node.func.value
    if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)):
        return False
    owner = inner.func.value
    return (
        isinstance(owner, ast.Attribute)
        and owner.attr == receiver_attr
        and isinstance(owner.value, ast.Name)
        and owner.value.id == "self"
    )


def test_the_media_label_never_blocks_the_step_loop():
    """``_advance_env_step`` must dispatch and move on.

    It is called once per iteration by both async runners, and the env group is
    the one busy inside ``interact`` for the whole run. Waiting on it makes a
    video label a synchronous dependency of training.
    """
    node = _method("embodied_runner.py", "EmbodiedRunner", "_advance_env_step")
    offenders = [sub for sub in ast.walk(node) if _is_wait_on(sub, "env")]
    assert not offenders, (
        "_advance_env_step waits on the env group; a per-step round-trip for a "
        "video label is a cost the dashboard imposes on training and then hides"
    )


@pytest.mark.parametrize(("module", "cls"), ASYNC_LOOPS)
def test_progress_is_reported_after_the_weight_sync(module: str, cls: str):
    """The reported duration must cover the whole iteration.

    Asserted positionally because that is exactly what went wrong: the call was
    correct, the arithmetic was correct, and it simply sat too early in the
    block. ``update_rollout_weights`` is the tail that was being missed.
    """
    node = _method(module, cls, "_run_impl")
    loops = [sub for sub in ast.walk(node) if isinstance(sub, ast.While)]
    assert loops, f"{cls}._run_impl has no step loop"

    def line_of(predicate) -> int | None:
        found = [
            sub.lineno
            for loop in loops
            for sub in ast.walk(loop)
            if isinstance(sub, ast.Call) and predicate(sub)
        ]
        return max(found) if found else None

    def is_call_named(name):
        return lambda call: (
            isinstance(call.func, ast.Attribute) and call.func.attr == name
        )

    progress_line = line_of(is_call_named("set_progress"))
    sync_line = line_of(is_call_named("update_rollout_weights"))

    assert progress_line is not None, f"{cls} reports no progress in its loop"
    assert sync_line is not None, f"{cls} does not sync weights in its loop"
    assert progress_line > sync_line, (
        f"{cls} calls set_progress at line {progress_line}, before "
        f"update_rollout_weights at line {sync_line}: the reported step duration "
        "would exclude the weight sync, so the dashboard's step time and ETA "
        "would read faster than the run really is and disagree with time/step"
    )


@pytest.mark.parametrize(("module", "cls"), ASYNC_LOOPS)
def test_eval_and_save_stay_outside_the_reported_duration(module: str, cls: str):
    """The other side of the contract: what is excluded is excluded on purpose.

    ``step_duration_s`` is a *net* step time. The synchronous runner says so
    outright -- eval and checkpointing are projected separately in the ETA, so
    folding them into the step would make the estimate lurch on every validation
    round. Async runners must exclude the same two and nothing else.

    So the measurement has to sit after the weight sync and before eval/save.
    Asserting only the first half would let a fix for the missing tail swing
    past that boundary and quietly change what the number means.
    """
    node = _method(module, cls, "_run_impl")
    loops = [sub for sub in ast.walk(node) if isinstance(sub, ast.While)]

    def lines_of(name: str) -> list[int]:
        return [
            sub.lineno
            for loop in loops
            for sub in ast.walk(loop)
            if isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == name
        ]

    progress = lines_of("set_progress")
    assert progress, f"{cls} reports no progress in its loop"

    for excluded in ("_save_checkpoint", "evaluate"):
        for line in lines_of(excluded):
            assert line > max(progress), (
                f"{cls} calls {excluded} at line {line}, before set_progress at "
                f"line {max(progress)}: {excluded} would be counted in the "
                "reported step duration, which is meant to be net of eval and "
                "save so the ETA does not lurch on validation rounds"
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
