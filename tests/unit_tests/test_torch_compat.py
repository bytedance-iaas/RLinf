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

import torch

from rlinf.utils.torch_compat import (
    should_set_torch_nccl_avoid_record_streams,
    silence_accumulate_grad_stream_mismatch_warning,
)


def test_torch_2_6_retains_avoid_record_streams_override():
    assert should_set_torch_nccl_avoid_record_streams("2.6.0+cu124")


def test_torch_2_11_uses_default_avoid_record_streams_behavior():
    assert not should_set_torch_nccl_avoid_record_streams("2.11.0+cu130")


def test_torch_2_8_nightly_uses_default_avoid_record_streams_behavior():
    assert not should_set_torch_nccl_avoid_record_streams("2.8.0a0+gitabcdef")


_STREAM_MISMATCH_SWITCH = "set_warn_on_accumulate_grad_stream_mismatch"


def test_stream_mismatch_warning_is_silenced_when_torch_supports_it():
    if not hasattr(torch.autograd.graph, _STREAM_MISMATCH_SWITCH):
        # Older torch does not emit the warning either, so there is nothing to
        # silence; the degradation path is covered by the test below.
        return
    assert silence_accumulate_grad_stream_mismatch_warning()


def test_stream_mismatch_warning_degrades_on_torch_without_the_switch():
    saved = getattr(torch.autograd.graph, _STREAM_MISMATCH_SWITCH, None)
    if saved is not None:
        delattr(torch.autograd.graph, _STREAM_MISMATCH_SWITCH)
    try:
        assert not silence_accumulate_grad_stream_mismatch_warning()
    finally:
        if saved is not None:
            setattr(torch.autograd.graph, _STREAM_MISMATCH_SWITCH, saved)
