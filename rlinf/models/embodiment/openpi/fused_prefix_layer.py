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

"""Drop-in fused replacement for the prefix-side (PaliGemma VLM) GemmaDecoderLayer.

Wraps the KernelAgent fused layer (`PrefixTrainFn`: fused fwd + hand-written
grad-only bwd, honors an arbitrary additive attention mask) as an nn.Module with
the same forward signature as `GemmaDecoderLayer`, so it drops into
`paligemma.language_model.layers` without touching the model's forward dispatch.

Only for the standard-RMSNorm prefix side (``use_adarms=False``). The
action-expert (adaRMS) layers are left untouched. Set
``actor.model.openpi.enable_fused_prefix`` to enable the replacement.

Set ``RLINF_FUSED_BACKWARD_COUNTER=1`` to confirm whether a workload sends
gradients through the prefix VLM.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from rlinf.utils.logging import get_logger

_logger = get_logger()


class _BackwardCounter:
    """Process-global counter to confirm whether fused backward is ever used."""

    fwd = 0
    bwd = 0
    logged = False


class FusedGemmaPrefixLayer(nn.Module):
    """Same forward API as GemmaDecoderLayer; internals run the fused kernel.

    Holds the ORIGINAL submodules (self_attn.{q,k,v,o}_proj, input_layernorm,
    post_attention_layernorm, mlp.{gate,up,down}_proj) so that (a) weights load
    from the SFT checkpoint unchanged, (b) external code that reads e.g.
    `layer.self_attn.q_proj.weight.dtype` (openpi_action_model.py) still works,
    (c) FSDP wraps the same params. forward reads them live (FSDP-safe).
    """

    def __init__(
        self,
        orig_layer: nn.Module,
        meta: tuple[int, int, int],
        layer_idx: int,
    ) -> None:
        super().__init__()
        # keep the original submodules verbatim (params + attribute access)
        self.self_attn = orig_layer.self_attn
        self.mlp = orig_layer.mlp
        self.input_layernorm = orig_layer.input_layernorm
        self.post_attention_layernorm = orig_layer.post_attention_layernorm
        self.meta = meta  # (num_heads, num_kv_heads, head_dim)
        self.eps = getattr(orig_layer.input_layernorm, "eps", 1e-6)
        # slot in language_model.layers; where K/V are written in the Cache
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings=None,  # fused kernel computes rope itself; unused
        adarms_cond: torch.Tensor | None = None,  # prefix side => None
        **kwargs,
    ) -> tuple[torch.Tensor]:
        if adarms_cond is not None:
            raise RuntimeError(
                "FusedGemmaPrefixLayer got adarms_cond != None; it is only valid "
                "for the standard prefix (use_adarms=False) side."
            )

        from .fused_kernels.layer_train import PrefixTrainFn

        # Prefix-cache build passes a Cache (use_cache=True) and expects this
        # layer's K/V written into it at layer_idx (HF convention); suffix
        # denoise then attends the cached prefix K/V. Mirror v3's reference.
        want_kv = use_cache or past_key_value is not None
        res = PrefixTrainFn.apply(
            hidden_states,
            self.input_layernorm.weight,
            self.self_attn.q_proj.weight,
            self.self_attn.k_proj.weight,
            self.self_attn.v_proj.weight,
            self.self_attn.o_proj.weight,
            self.post_attention_layernorm.weight,
            self.mlp.gate_proj.weight,
            self.mlp.up_proj.weight,
            self.mlp.down_proj.weight,
            float(self.eps),
            self.meta,
            attention_mask,
            position_ids,
            want_kv,
        )

        if want_kv:
            out, k, v = res
            if past_key_value is not None:
                cache_kwargs = None
                if cache_position is not None:
                    cache_kwargs = {"cache_position": cache_position}
                past_key_value.update(k, v, self.layer_idx, cache_kwargs)
        else:
            out = res

        if os.environ.get("RLINF_FUSED_BACKWARD_COUNTER", "0") == "1":
            _BackwardCounter.fwd += 1
            # hook fires only if this output participates in a backward pass
            if out.requires_grad:
                out.register_hook(_fused_bwd_hook)

        return (out,)


def _fused_bwd_hook(grad: torch.Tensor) -> torch.Tensor:
    _BackwardCounter.bwd += 1
    if not _BackwardCounter.logged:
        _logger.info(
            f"[fused-prefix] backward IS used (fwd_calls={_BackwardCounter.fwd}, "
            f"first bwd fired). Prefix VLM is NOT no-grad."
        )
        _BackwardCounter.logged = True
    return grad


def apply_fused_prefix_layers(model: nn.Module, enabled: bool = False) -> int:
    """Swap prefix VLM GemmaDecoderLayers for the fused version.

    Only ``paligemma.language_model`` is modified, so the action-expert adaRMS
    layers remain unchanged.

    Args:
        model: OpenPi policy containing ``paligemma_with_expert``.
        enabled: Whether to replace eligible prefix decoder layers.

    Returns:
        The number of replaced layers.
    """
    if not enabled:
        return 0

    pg = getattr(model, "paligemma_with_expert", None)
    if pg is None:
        _logger.warning("[fused-prefix] no paligemma_with_expert; skip.")
        return 0

    lm = pg.paligemma.language_model
    cfg = lm.config
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    meta = (n_heads, n_kv, head_dim)

    n = 0
    for i, layer in enumerate(lm.layers):
        # only standard (non-adaRMS) layers; prefix VLM has cond_dim=None
        if getattr(layer.input_layernorm, "cond_dim", None) is not None:
            continue
        lm.layers[i] = FusedGemmaPrefixLayer(layer, meta, layer_idx=i)
        n += 1

    _logger.info(
        f"[fused-prefix] replaced {n} prefix VLM GemmaDecoderLayers "
        "with fused versions."
    )
    return n
