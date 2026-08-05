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

"""Every training loop must tell the env workers which step it is on.

``RecordVideo.set_global_step`` is what puts a step on a media index row, and a
row without one cannot be tied back to a point on a curve -- the dashboard's
``/media/steps`` returns nothing and the UI cannot mark which iterations have a
clip. ``EnvWorker.set_global_step`` forwards it to the wrappers, but a forwarder
nobody calls does nothing: the synchronous runner called it, both async runners
did not, and a real four-GPU async PPO run wrote eight clips all carrying
``step: null``.

Nothing failed. The videos played, the index parsed, the API answered. That is
what makes this worth asserting at the call site rather than trusting a unit test
of the wrapper -- ``tests/unit_tests/test_media_index.py`` covers
``set_global_step`` thoroughly and passed the whole time, because it calls the
method the production code was skipping. Same shape as the bug documented in
``test_runner_component_marks.py``: an API-level suite proving a method works
says nothing about whether anyone invokes it.

Read as source rather than run, for the same reason as that file -- these runners
import ray at module load and a driver loop needs a live cluster.
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

# Every embodied runner, and the method that drives its step loop. The async
# pair reaches the env group through the shared ``_advance_env_step`` helper
# because ``interact`` owns those workers for the whole run; the synchronous
# runner calls the group inline, where the env workers are idle at that point.
LOOPS = [
    ("embodied_runner.py", "EmbodiedRunner", "_run_impl"),
    ("embodied_runner.py", "EmbodiedRunner", "run_pipeline"),
    ("async_embodied_runner.py", "AsyncEmbodiedRunner", "_run_impl"),
    ("async_ppo_embodied_runner.py", "AsyncPPOEmbodiedRunner", "_run_impl"),
]


def _method(module: str, cls: str, name: str) -> ast.FunctionDef:
    """Return the AST of one runner method."""
    path = os.path.join(RUNNERS, module)
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found in {path}")


def _sets_env_step(node: ast.AST) -> bool:
    """Whether this method advances the env step, directly or via the helper."""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Attribute):
            continue
        # self._advance_env_step()
        if sub.func.attr == "_advance_env_step":
            return True
        # self.env.set_global_step(...)
        if sub.func.attr == "set_global_step":
            owner = sub.func.value
            if (
                isinstance(owner, ast.Attribute)
                and owner.attr == "env"
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "self"
            ):
                return True
    return False


@pytest.mark.parametrize(("module", "cls", "name"), LOOPS)
def test_loop_tells_env_workers_the_step(module: str, cls: str, name: str):
    """A loop that skips this indexes every clip it records as ``step: null``."""
    assert _sets_env_step(_method(module, cls, name)), (
        f"{cls}.{name} sets the step on the actor and rollout groups but not on "
        "the env group, so recorded videos land in the media index with no step"
    )


def test_async_loops_advance_the_step_inside_the_loop():
    """Once at startup is not enough: the label must move with the run.

    ``interact`` is called once and runs for the whole run, so the async runners
    cannot re-enter it per step the way the synchronous loop does. The startup
    call labels the first step; without a second call inside the ``while``, every
    clip for the rest of the run carries step 0.
    """
    for module, cls in [
        ("async_embodied_runner.py", "AsyncEmbodiedRunner"),
        ("async_ppo_embodied_runner.py", "AsyncPPOEmbodiedRunner"),
    ]:
        node = _method(module, cls, "_run_impl")
        loops = [sub for sub in ast.walk(node) if isinstance(sub, ast.While)]
        assert loops, f"{cls}._run_impl has no step loop"
        assert any(_sets_env_step(loop) for loop in loops), (
            f"{cls}._run_impl advances the env step only before its loop, so "
            "every clip after the first step would be labelled with the first"
        )


def test_the_helper_never_lets_a_video_label_kill_a_run():
    """A media label is diagnostics; training must outlive its failure.

    The async runners reach a worker group that is mid-``interact``. If that call
    can raise -- a group already draining, or a runner whose env workers predate
    the forwarder -- an unguarded call would end a multi-hour run over a field
    the training loop does not read.
    """
    node = _method("embodied_runner.py", "EmbodiedRunner", "_advance_env_step")
    handlers = [sub for sub in ast.walk(node) if isinstance(sub, ast.Try)]
    assert handlers, "_advance_env_step must not propagate a failed step update"
    # Silence is not the goal either: an operator who sees `step: null` in the
    # index needs a line in the log saying the update was refused.
    warned = any(
        isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr in {"warning", "warn", "error"}
        for handler in handlers
        for sub in ast.walk(handler)
    )
    assert warned, "a swallowed step update must still be logged"
