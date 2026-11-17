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

"""Numerics for the fused Pi0.5 prefix kernels, against a pure-torch reference.

Needs a CUDA device and Triton, so the whole module skips on CPU-only runners.
The shapes are shrunk from the gemma_2b prefix (2048/16384, 8 q heads, 1 kv
head, head_dim 256) to keep the test quick; head_dim stays 256 because the
attention launch configs branch on it.

The padded case matters more than it looks: openpi's ``make_att_2d_masks``
builds ``pad_2d_masks`` as an outer product, so a padded query is masked along
*both* axes and its whole score row is the sentinel. Attention has to keep the
softmax statistics for such a row accurate enough that backward rebuilds the
same probabilities the forward used.
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused kernels need a CUDA device"
)

HIDDEN = 512
INTERMEDIATE = 1024
N_HEADS = 8
N_KV = 1
HEAD_DIM = 256
EPS = 1e-6
ROPE_THETA = 10000.0
SENTINEL = -2.3819763e38
TOL = 2e-2
DTYPE = torch.bfloat16


def _weights(device: torch.device) -> list[torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(0)

    def rand(*shape: int, std: float = 0.02) -> torch.Tensor:
        t = (torch.randn(*shape, generator=gen) * std).to(device).to(DTYPE)
        return t.requires_grad_(True)

    return [
        rand(HIDDEN, std=0.1),  # input_layernorm
        rand(N_HEADS * HEAD_DIM, HIDDEN),  # q_proj
        rand(N_KV * HEAD_DIM, HIDDEN),  # k_proj
        rand(N_KV * HEAD_DIM, HIDDEN),  # v_proj
        rand(HIDDEN, N_HEADS * HEAD_DIM),  # o_proj
        rand(HIDDEN, std=0.1),  # post_attention_layernorm
        rand(INTERMEDIATE, HIDDEN),  # gate_proj
        rand(INTERMEDIATE, HIDDEN),  # up_proj
        rand(HIDDEN, INTERMEDIATE),  # down_proj
    ]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


def _reference(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """The GemmaDecoderLayer the fused kernels replace, in plain torch."""
    w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd = weights
    batch, seq, _ = x.shape

    def rms(t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        var = t.float().pow(2).mean(-1, keepdim=True)
        return (t.float() * torch.rsqrt(var + EPS) * (1.0 + w.float())).to(t.dtype)

    h = rms(x, w_ln)
    q = F.linear(h, wq).view(batch, seq, N_HEADS, HEAD_DIM).transpose(1, 2)
    k = F.linear(h, wk).view(batch, seq, N_KV, HEAD_DIM).transpose(1, 2)
    v = F.linear(h, wv).view(batch, seq, N_KV, HEAD_DIM).transpose(1, 2)

    half = HEAD_DIM // 2
    inv = 1.0 / (
        ROPE_THETA ** (torch.arange(0, half, device=x.device).float() * 2.0 / HEAD_DIM)
    )
    freqs = torch.einsum("bs,d->bsd", position_ids.float(), inv)
    emb = torch.cat((freqs, freqs), -1)
    cos, sin = emb.cos().to(DTYPE)[:, None], emb.sin().to(DTYPE)[:, None]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin

    groups = N_HEADS // N_KV
    kx, vx = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    scores = torch.matmul(q, kx.transpose(-1, -2)).float() * (HEAD_DIM**-0.5)
    attn = torch.softmax(scores + mask.float(), -1).to(DTYPE)
    o = torch.matmul(attn, vx).transpose(1, 2).reshape(batch, seq, N_HEADS * HEAD_DIM)

    res = x + F.linear(o, wo)
    hn = rms(res, w_pln)
    mlp = F.linear(hn, wg)
    return res + F.linear(F.gelu(mlp, approximate="tanh") * F.linear(hn, wu), wd)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()


def _openpi_mask(
    lengths: list[int], seq: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(pad, position_ids, additive mask), built the way openpi builds them."""
    lens = torch.tensor(lengths, device=device)
    pad = torch.arange(seq, device=device)[None, :] < lens[:, None]
    position_ids = (torch.cumsum(pad.long(), dim=1) - 1).clamp_min(0)
    pad_2d = pad[:, None, :] & pad[:, :, None]
    mask = torch.where(pad_2d[:, None, :, :], 0.0, SENTINEL)
    return pad, position_ids, mask


@pytest.mark.parametrize(
    ("name", "lengths"),
    [("unpadded", [96, 96]), ("ragged", [96, 61])],
)
def test_prefix_train_fn_matches_reference(name: str, lengths: list[int]) -> None:
    from rlinf.models.embodiment.openpi.fused_kernels.layer_train import PrefixTrainFn

    device = torch.device("cuda")
    torch.manual_seed(0)
    batch, seq = len(lengths), 96
    weights = _weights(device)
    x = torch.randn(batch, seq, HIDDEN, device=device, dtype=DTYPE) * 0.5
    x = x.requires_grad_(True)
    pad, position_ids, mask = _openpi_mask(lengths, seq, device)
    grad_out = torch.randn_like(x)
    keep = pad[:, :, None]

    def collect(out: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        out.backward(grad_out)
        grads = [x.grad.clone()] + [w.grad.clone() for w in weights]
        for t in (x, *weights):
            t.grad = None
        return out.detach().clone(), grads

    ref_out, ref_grads = collect(_reference(x, weights, mask, position_ids))
    fused_out, fused_grads = collect(
        PrefixTrainFn.apply(
            x,
            *weights,
            EPS,
            (N_HEADS, N_KV, HEAD_DIM),
            mask,
            position_ids,
            False,
        )
    )

    # padded rows are undefined on both sides; compare only the real tokens
    assert _rel(fused_out * keep, ref_out * keep) < TOL, name
    names = ["grad_x", *("w%d" % i for i in range(len(weights)))]
    for label, got, want in zip(names, fused_grads, ref_grads, strict=True):
        if label == "grad_x":
            got, want = got * keep, want * keep
        assert _rel(got, want) < TOL, f"{name}/{label}"


def test_use_cache_returns_hf_layout_kv() -> None:
    from rlinf.models.embodiment.openpi.fused_kernels.layer_train import PrefixTrainFn

    device = torch.device("cuda")
    torch.manual_seed(0)
    batch, seq = 2, 96
    weights = [w.detach() for w in _weights(device)]
    x = torch.randn(batch, seq, HIDDEN, device=device, dtype=DTYPE) * 0.5
    _, position_ids, mask = _openpi_mask([seq, seq - 21], seq, device)
    meta = (N_HEADS, N_KV, HEAD_DIM)

    with torch.no_grad():
        plain = PrefixTrainFn.apply(x, *weights, EPS, meta, mask, position_ids, False)
        cached, k, v = PrefixTrainFn.apply(
            x, *weights, EPS, meta, mask, position_ids, True
        )

    assert k.shape == (batch, N_KV, seq, HEAD_DIM)
    assert v.shape == (batch, N_KV, seq, HEAD_DIM)
    assert k.is_contiguous() and v.is_contiguous()
    # use_cache must not perturb the hidden states it also returns
    assert torch.equal(plain, cached)


def test_fused_attention_normalises_a_fully_masked_row() -> None:
    """A row whose every key is masked must degrade to a uniform average.

    ``pad_2d_masks`` produces such rows for every padded query, and getting the
    softmax denominator wrong there scales the whole layer's weight gradients.
    """
    from rlinf.models.embodiment.openpi.fused_kernels.attention import (
        _torch_attention,
        fused_attention,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    batch, seq = 2, 96
    scale = HEAD_DIM**-0.5

    def qkv(heads: int) -> torch.Tensor:
        t = torch.randn(batch, heads, seq, HEAD_DIM, device=device, dtype=DTYPE) * 0.3
        return t.requires_grad_(True)

    q, k, v = qkv(N_HEADS), qkv(N_KV), qkv(N_KV)
    _, _, mask = _openpi_mask([seq, seq - 40], seq, device)
    grad_out = torch.randn(batch, seq, N_HEADS, HEAD_DIM, device=device, dtype=DTYPE)

    ref = _torch_attention(q, k, v, mask, scale)
    ref.backward(grad_out)
    ref_grads = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None

    out = fused_attention(q, k, v, mask, scale)
    out.backward(grad_out)

    assert _rel(out, ref) < TOL
    for label, t, want in zip(("q", "k", "v"), (q, k, v), ref_grads, strict=True):
        assert _rel(t.grad, want) < TOL, label


def test_broadcast_position_ids_is_rejected() -> None:
    """A [1, S] position_ids must raise, not read past the end of the buffer.

    ``transformers`` synthesises exactly that shape when a caller omits
    position_ids, and the RoPE kernel indexes the tensor per batch entry.
    """
    from rlinf.models.embodiment.openpi.fused_kernels.layer_train import PrefixTrainFn

    device = torch.device("cuda")
    torch.manual_seed(0)
    batch, seq = 2, 96
    weights = [w.detach() for w in _weights(device)]
    x = torch.randn(batch, seq, HIDDEN, device=device, dtype=DTYPE) * 0.5
    _, position_ids, mask = _openpi_mask([seq, seq], seq, device)

    with torch.no_grad(), pytest.raises(ValueError, match=r"position_ids must be"):
        PrefixTrainFn.apply(
            x,
            *weights,
            EPS,
            (N_HEADS, N_KV, HEAD_DIM),
            mask,
            position_ids[:1],  # the [1, S] transformers default
            False,
        )


def test_query_broadcast_mask_is_rejected() -> None:
    """[B, 1, 1, Sk] is a padding-only mask shape the kernels cannot broadcast."""
    from rlinf.models.embodiment.openpi.fused_kernels.attention import fused_attention

    device = torch.device("cuda")
    torch.manual_seed(0)
    batch, seq = 2, 96

    def qkv(heads: int) -> torch.Tensor:
        return torch.randn(batch, heads, seq, HEAD_DIM, device=device, dtype=DTYPE)

    q, k, v = qkv(N_HEADS), qkv(N_KV), qkv(N_KV)
    mask = torch.zeros(batch, 1, 1, seq, device=device, dtype=torch.float32)

    with torch.no_grad(), pytest.raises(ValueError, match=r"mask query axis"):
        fused_attention(q, k, v, mask, HEAD_DIM**-0.5)
