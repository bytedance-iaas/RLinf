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

from __future__ import annotations

import torch
import torch.nn.functional as F

# Kernels stay compiled unless a caller turns this off before the first forward.
# Isaac Sim jobs do that: their bundled torch cannot import the venv's triton.
_torch_compile_enabled = True


def set_torch_compile(enabled: bool) -> None:
    """Enable or disable OpenPI fused-kernel compilation for this process."""
    global _torch_compile_enabled
    _torch_compile_enabled = bool(enabled)


def torch_compile_enabled() -> bool:
    return _torch_compile_enabled


# Gemma attention runs through CANN's FlashAttentionScore only when a caller
# turns this on, which requires an Ascend NPU and torch_npu.
_npu_fused_attention_enabled = False


def set_npu_fused_attention(enabled: bool) -> None:
    """Route Gemma attention through ``npu_fusion_attention`` for this process."""
    global _npu_fused_attention_enabled
    _npu_fused_attention_enabled = bool(enabled)


def npu_fused_attention_enabled() -> bool:
    return _npu_fused_attention_enabled


# Non-adaptive RMSNorm runs through ``npu_rms_norm`` only when a caller turns
# this on. The adaptive variant keeps the explicit path: it applies its scale
# and shift in float32 after normalising, and the fused kernel returns the
# input dtype, so routing it through would round before the affine.
_npu_fused_rms_norm_enabled = False


def set_npu_fused_rms_norm(enabled: bool) -> None:
    """Route non-adaptive Gemma RMSNorm through ``npu_rms_norm`` for this process."""
    global _npu_fused_rms_norm_enabled
    _npu_fused_rms_norm_enabled = bool(enabled)


def npu_fused_rms_norm_enabled() -> bool:
    return _npu_fused_rms_norm_enabled


# RoPE's rotation runs through ``npu_rotary_mul`` only when a caller turns this
# on. It covers the rotation, not the cos/sin tables: Pi0.5's positions come
# from a cumulative sum over the validity mask, so the tables depend on the
# padding and are still rebuilt per call.
_npu_fused_rope_enabled = False


def set_npu_fused_rope(enabled: bool) -> None:
    """Route the Gemma RoPE rotation through ``npu_rotary_mul`` for this process."""
    global _npu_fused_rope_enabled
    _npu_fused_rope_enabled = bool(enabled)


def npu_fused_rope_enabled() -> bool:
    return _npu_fused_rope_enabled


def _gelu_glu_eager(
    gate_input: torch.Tensor, value_input: torch.Tensor
) -> torch.Tensor:
    return F.gelu(gate_input) * value_input


_gelu_glu_compiled = torch.compile(_gelu_glu_eager)


def gelu_glu(gate_input: torch.Tensor, value_input: torch.Tensor) -> torch.Tensor:
    """Fused GELU-GLU activation: ``gelu(gate_input) * value_input``."""
    fn = _gelu_glu_compiled if _torch_compile_enabled else _gelu_glu_eager
    return fn(gate_input, value_input)


def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "mp_bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[dtype_str]
