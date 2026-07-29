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

from rlinf.utils.torch_compat import should_set_torch_nccl_avoid_record_streams


def test_torch_2_6_retains_avoid_record_streams_override():
    assert should_set_torch_nccl_avoid_record_streams("2.6.0+cu124")


def test_torch_2_11_uses_default_avoid_record_streams_behavior():
    assert not should_set_torch_nccl_avoid_record_streams("2.11.0+cu130")


def test_torch_2_8_nightly_uses_default_avoid_record_streams_behavior():
    assert not should_set_torch_nccl_avoid_record_streams("2.8.0a0+gitabcdef")
