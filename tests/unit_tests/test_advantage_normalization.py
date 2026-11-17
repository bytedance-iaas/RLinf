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

"""Advantage normalization must survive a batch with one valid entry.

`Tensor.std()` applies Bessel's correction by default, so a single element
divides by zero and yields NaN. The `+ 1e-5` epsilon absorbs a zero std but not
a NaN, and the guards in front of it only tested for an *empty* selection, so
`n == 1` fell straight through. The NaN then reaches the policy loss, the
gradients, and the grad norm; FSDP skips the optimizer step, and the run
continues with nothing learned.

This is reachable whenever the per-rank batch narrows to one valid entry --
e.g. a global batch equal to the world size.

The `n >= 2` cases are pinned as well: this fix must not change the values that
already worked.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from rlinf.algorithms.advantages import compute_raw_advantages  # noqa: E402
from rlinf.algorithms.utils import safe_normalize  # noqa: E402


def test_one_valid_entry_does_not_produce_nan():
    array = torch.tensor([[3.0]])
    mask = torch.ones(1, 1, dtype=torch.bool)

    out = safe_normalize(array.clone(), mask)

    assert not torch.isnan(out).any()


def test_one_valid_entry_normalizes_to_zero():
    """No spread means no signal: the entry sits exactly at its own mean."""
    array = torch.tensor([[3.0]])
    mask = torch.ones(1, 1, dtype=torch.bool)

    out = safe_normalize(array.clone(), mask)

    assert out.item() == pytest.approx(0.0)


def test_one_valid_entry_among_masked_ones():
    """The count that matters is the masked-in one, not the tensor's size."""
    array = torch.tensor([[1.0], [7.0], [2.0]])
    mask = torch.tensor([[False], [True], [False]])

    out = safe_normalize(array.clone(), mask)

    assert not torch.isnan(out).any()
    assert out[1].item() == pytest.approx(0.0)


@pytest.mark.parametrize("n", [2, 3, 8])
def test_multiple_entries_keep_the_previous_values(n):
    """Pins the unchanged path: still the Bessel-corrected std, still epsilon."""
    array = torch.arange(float(n)).reshape(n, 1)
    mask = torch.ones(n, 1, dtype=torch.bool)

    out = safe_normalize(array.clone(), mask)

    valid = array[mask]
    expected = (array - valid.mean()) / (valid.std() + 1e-5)
    assert torch.allclose(out, expected)
    assert not torch.isnan(out).any()


def test_no_mask_normalizes_over_every_element():
    """``loss_mask=None`` makes the selection an unsqueeze, not a filter.

    Its first dimension is 1 regardless of how many elements it holds, so the
    single-entry check has to count elements rather than rows.
    """
    array = torch.arange(12.0).reshape(4, 3)

    out = safe_normalize(array.clone(), None)

    expected = (array - array.mean()) / (array.std() + 1e-5)
    assert torch.allclose(out, expected[None])
    assert not torch.isnan(out).any()


def test_an_empty_selection_is_returned_untouched():
    array = torch.tensor([[1.0], [2.0]])
    mask = torch.zeros(2, 1, dtype=torch.bool)

    out = safe_normalize(array.clone(), mask)

    assert torch.equal(out, array)


def test_a_constant_batch_still_relies_on_the_epsilon():
    """std == 0 was always handled by the epsilon; that must not regress."""
    array = torch.full((4, 1), 5.0)
    mask = torch.ones(4, 1, dtype=torch.bool)

    out = safe_normalize(array.clone(), mask)

    assert not torch.isnan(out).any()
    assert torch.allclose(out, torch.zeros_like(out))


def test_raw_advantages_with_a_single_sample():
    """`compute_raw_advantages` carried its own copy of the same guard."""
    rewards = torch.tensor([2.0])
    loss_mask = torch.ones(1, 1)

    advantages, _ = compute_raw_advantages(
        rewards=rewards, loss_mask=loss_mask, normalize_advantages=True
    )

    assert not torch.isnan(advantages).any()
    assert math.isfinite(advantages.sum().item())


def test_raw_advantages_with_several_samples():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss_mask = torch.ones(1, 4)

    advantages, _ = compute_raw_advantages(
        rewards=rewards, loss_mask=loss_mask, normalize_advantages=True
    )

    assert not torch.isnan(advantages).any()
    assert advantages.mean().item() == pytest.approx(0.0, abs=1e-5)
