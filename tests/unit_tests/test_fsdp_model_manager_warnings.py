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

from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    _should_warn_about_actor_precision,
)


def test_openpi_manages_its_own_precision():
    assert not _should_warn_about_actor_precision(
        SupportedModel.OPENPI.value, torch.bfloat16
    )


def test_generic_low_precision_actor_keeps_recommendation():
    assert _should_warn_about_actor_precision(
        SupportedModel.QWEN2_5.value, torch.bfloat16
    )


def test_fp32_actor_does_not_emit_precision_recommendation():
    assert not _should_warn_about_actor_precision(
        SupportedModel.QWEN2_5.value, torch.float32
    )
