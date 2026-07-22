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

from types import SimpleNamespace

import torch
import torch.nn as nn

from rlinf.models.embodiment.openpi.fused_prefix_layer import (
    FusedGemmaPrefixLayer,
    apply_fused_prefix_layers,
)


class _Norm(nn.Module):
    def __init__(self, cond_dim: int | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))
        self.eps = 1e-6
        self.cond_dim = cond_dim


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class _Layer(nn.Module):
    def __init__(self, cond_dim: int | None = None) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _Mlp()
        self.input_layernorm = _Norm(cond_dim)
        self.post_attention_layernorm = _Norm(cond_dim)


class _LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            hidden_size=4,
        )
        self.layers = nn.ModuleList([_Layer(), _Layer(cond_dim=4)])


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        language_model = _LanguageModel()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Module()
        self.paligemma_with_expert.paligemma.language_model = language_model


def test_fused_prefix_is_disabled_by_default() -> None:
    model = _Policy()
    original_layers = list(model.paligemma_with_expert.paligemma.language_model.layers)

    replaced = apply_fused_prefix_layers(model)

    assert replaced == 0
    assert list(model.paligemma_with_expert.paligemma.language_model.layers) == (
        original_layers
    )


def test_fused_prefix_replaces_only_standard_rmsnorm_layers() -> None:
    model = _Policy()
    language_model = model.paligemma_with_expert.paligemma.language_model
    standard_layer = language_model.layers[0]
    adarms_layer = language_model.layers[1]
    q_proj = standard_layer.self_attn.q_proj

    replaced = apply_fused_prefix_layers(model, enabled=True)

    assert replaced == 1
    assert isinstance(language_model.layers[0], FusedGemmaPrefixLayer)
    assert language_model.layers[0].self_attn.q_proj is q_proj
    assert language_model.layers[0].layer_idx == 0
    assert language_model.layers[1] is adarms_layer


def test_fused_prefix_skips_models_without_paligemma() -> None:
    assert apply_fused_prefix_layers(nn.Linear(2, 2), enabled=True) == 0
