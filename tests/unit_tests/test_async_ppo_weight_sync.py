# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner


class _Handle:
    def __init__(self, done: bool = False) -> None:
        self.is_done = done
        self.wait_calls = 0

    def done(self) -> bool:
        return self.is_done

    def wait(self) -> None:
        self.wait_calls += 1
        self.is_done = True


class _Actor:
    def __init__(self, handles: list[_Handle]) -> None:
        self.handles = handles
        self.sync_calls = 0

    def sync_model_to_rollout(self) -> _Handle:
        handle = self.handles[self.sync_calls]
        self.sync_calls += 1
        return handle


class _Rollout:
    def __init__(
        self,
        blocking_handles: list[_Handle],
        background_handles: list[_Handle],
    ) -> None:
        self.blocking_handles = blocking_handles
        self.background_handles = background_handles
        self.blocking_calls = 0
        self.background_calls = 0

    def sync_model_from_actor(self) -> _Handle:
        handle = self.blocking_handles[self.blocking_calls]
        self.blocking_calls += 1
        return handle

    def request_actor_sync_model(self) -> _Handle:
        handle = self.background_handles[self.background_calls]
        self.background_calls += 1
        return handle


def _make_runner(actor: _Actor, rollout: _Rollout) -> AsyncPPOEmbodiedRunner:
    runner = object.__new__(AsyncPPOEmbodiedRunner)
    runner.actor = actor
    runner.rollout = rollout
    runner._pending_rollout_weight_sync = None
    return runner


def test_blocking_weight_sync_waits_for_both_sides() -> None:
    actor_handle = _Handle()
    rollout_handle = _Handle()
    actor = _Actor([actor_handle])
    rollout = _Rollout([rollout_handle], [])
    runner = _make_runner(actor, rollout)

    runner.update_rollout_weights()

    assert actor.sync_calls == 1
    assert rollout.blocking_calls == 1
    assert actor_handle.wait_calls == 1
    assert rollout_handle.wait_calls == 1
    assert runner._pending_rollout_weight_sync is None


def test_nonblocking_weight_sync_coalesces_until_previous_sync_finishes() -> None:
    first_actor_handle = _Handle()
    first_rollout_handle = _Handle(done=True)
    second_actor_handle = _Handle()
    second_rollout_handle = _Handle(done=True)
    actor = _Actor([first_actor_handle, second_actor_handle])
    rollout = _Rollout([], [first_rollout_handle, second_rollout_handle])
    runner = _make_runner(actor, rollout)

    runner.update_rollout_weights(no_wait=True)
    runner.update_rollout_weights(no_wait=True)

    assert actor.sync_calls == 1
    assert rollout.background_calls == 1
    assert first_actor_handle.wait_calls == 0
    assert first_rollout_handle.wait_calls == 0

    first_actor_handle.is_done = True
    runner.update_rollout_weights(no_wait=True)

    assert first_actor_handle.wait_calls == 1
    assert first_rollout_handle.wait_calls == 1
    assert actor.sync_calls == 2
    assert rollout.background_calls == 2
    assert runner._pending_rollout_weight_sync == (
        second_rollout_handle,
        second_actor_handle,
    )


def test_blocking_weight_sync_drains_an_inflight_background_sync() -> None:
    pending_actor_handle = _Handle()
    pending_rollout_handle = _Handle()
    blocking_actor_handle = _Handle()
    blocking_rollout_handle = _Handle()
    actor = _Actor([blocking_actor_handle])
    rollout = _Rollout([blocking_rollout_handle], [])
    runner = _make_runner(actor, rollout)
    runner._pending_rollout_weight_sync = (
        pending_rollout_handle,
        pending_actor_handle,
    )

    runner.update_rollout_weights(no_wait=False)

    assert pending_actor_handle.wait_calls == 1
    assert pending_rollout_handle.wait_calls == 1
    assert blocking_actor_handle.wait_calls == 1
    assert blocking_rollout_handle.wait_calls == 1
    assert runner._pending_rollout_weight_sync is None
