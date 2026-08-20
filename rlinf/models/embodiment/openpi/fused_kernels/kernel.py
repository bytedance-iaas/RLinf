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

"""Fused Gemma decoder layer -- Triton forward kernels + ``kernel_function``.

The layer this file implements (``GemmaDecoderLayer.forward``, both the standard
and the adaRMS variant)::

    r = x
    h = Norm_in(x)  # RMSNorm(1+w)  |  adaRMS(scale,shift)
    h = SelfAttn(h)  # GQA + RoPE + additive mask
    h = r + h[*gate_in]  # gated residual on the adaRMS path
    r = h
    h = Norm_post(h)
    h = down(gelu_tanh(gate(h)) * up(h))
    out = r + h[*gate_post]

What is fused, and why
----------------------
The projections (q/k/v/o, gate/up/down) are ~90% of the layer's FLOPs and are
issued as cuBLAS GEMMs -- on an H20 (148 TF bf16 dense, 4.0 TB/s) they are
compute-bound and a Triton GEMM does not beat cuBLAS at these shapes.  What
*is* worth fusing is everything between them, and that is what this file
provides:

  * ``_norm_fwd_kernel``      -- (optional gated residual add) + fp32-variance
                                 RMSNorm / adaRMS in a single pass, also
                                 emitting the reciprocal std for backward.
  * ``_rope_transpose_kernel``-- RoPE and the ``[B,S,H*D] -> [B,H,S,D]`` layout
                                 change in one pass, computing cos/sin inline
                                 instead of materialising rotary tables.
  * ``_gelu_mul_kernel``      -- ``gelu_tanh(gate) * up`` over the 16384-wide
                                 intermediate in one pass instead of three.
  * ``_gated_add_kernel``     -- the final (gated) residual.
  * attention -- the flash kernels in :mod:`attention`, so the
    ``[B,Hq,S,S]`` score matrix is never materialised and the GQA key/value
    heads are never expanded with ``repeat_interleave``.

No path in this file falls back to the PyTorch reference layer.

``kernel_function`` is forward-only (it is what an inference / no-grad call
uses).  The autograd-capable training entry points live in :mod:`layer_train`
(whole layer) and :mod:`attention` (attention alone).
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from .attention import _attention_forward

__all__ = ["kernel_function", "HEAD_DIM"]

HEAD_DIM = 256
ROPE_THETA = 10000.0

_GELU_C0 = tl.constexpr(0.7978845608028654)  # sqrt(2/pi)
_GELU_C1 = tl.constexpr(0.044715)
_LOG2E = tl.constexpr(1.4426950408889634)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _warps_for(block):
    if block >= 4096:
        return 16
    if block >= 2048:
        return 8
    if block >= 512:
        return 4
    return 2


@functools.lru_cache(maxsize=None)
def _inv_freq(head_dim, device_str, theta=ROPE_THETA):
    """1 / theta**(2i/head_dim), i < head_dim/2 -- built once per device.

    Computed on the host in fp32 exactly as the reference does, so the rotary
    angles agree bit-for-bit with ``ref_tests._rotary`` before the cos/sin.
    """
    half = head_dim // 2
    dev = torch.device(device_str)
    i = torch.arange(0, half, device=dev, dtype=torch.float32)
    return (1.0 / (theta ** (i * 2.0 / head_dim))).contiguous()


def _rope_config(S):
    """(BLOCK_S, num_warps) for the RoPE/layout kernel -- measured on H20."""
    return (min(64, _next_pow2(max(S, 1))), 8)


def _mm(a2d, w):
    """``a2d @ w.T`` (an F.linear without the autograd bookkeeping)."""
    return torch.mm(a2d, w.t())


# ---------------------------------------------------------------------------
# (gated residual +) RMSNorm / adaRMS
# ---------------------------------------------------------------------------
@triton.jit
def _norm_fwd_kernel(
    X,
    O,
    Gate,
    Res,
    Y,
    Rstd,
    W,
    Mod,
    seq,
    N,
    eps,
    BLOCK: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    HAS_GATE: tl.constexpr,
    ADARMS: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // seq
    offs = tl.arange(0, BLOCK)
    msk = offs < N

    x = tl.load(X + row * N + offs, mask=msk, other=0.0).to(tl.float32)
    if ADD_RESIDUAL:
        o = tl.load(O + row * N + offs, mask=msk, other=0.0).to(tl.float32)
        if HAS_GATE:
            g = tl.load(Gate + b * N + offs, mask=msk, other=0.0).to(tl.float32)
            o = o * g
        # round the residual to the storage dtype *before* the variance so the
        # backward pass, which reloads Res, sees exactly these values.
        xs = (x + o).to(Res.dtype.element_ty)
        tl.store(Res + row * N + offs, xs, mask=msk)
        x = xs.to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + row, r)

    normed = x * r
    if ADARMS:
        sc = tl.load(Mod + b * 3 * N + offs, mask=msk, other=0.0).to(tl.float32)
        sh = tl.load(Mod + b * 3 * N + N + offs, mask=msk, other=0.0).to(tl.float32)
        y = normed * (1.0 + sc) + sh
    else:
        w = tl.load(W + offs, mask=msk, other=0.0).to(tl.float32)
        y = normed * (1.0 + w)
    tl.store(Y + row * N + offs, y.to(Y.dtype.element_ty), mask=msk)


def norm_forward(
    x, weight, eps, *, residual=None, gate=None, mod=None, seq=None, num_warps=None
):
    """Returns (y, rstd, res).

    ``residual`` present -> ``res = residual + x * gate`` is formed first and
    normalised (``x`` is then the attention/MLP branch output).  ``mod`` present
    -> adaRMS: ``(1+scale)*xhat + shift`` with scale/shift sliced out of the
    ``[B, 3N]`` modulation.
    """
    branch = x
    base = residual if residual is not None else x
    B, S, N = base.shape
    seq = S if seq is None else seq
    rows = B * S

    y = torch.empty_like(base)
    rstd = torch.empty(rows, device=base.device, dtype=torch.float32)
    res = torch.empty_like(base) if residual is not None else None

    BLOCK = _next_pow2(N)
    _norm_fwd_kernel[(rows,)](
        residual if residual is not None else x,
        branch,
        gate if gate is not None else y,
        res if res is not None else y,
        y,
        rstd,
        weight if weight is not None else y,
        mod if mod is not None else y,
        seq,
        N,
        eps,
        BLOCK=BLOCK,
        ADD_RESIDUAL=residual is not None,
        HAS_GATE=gate is not None,
        ADARMS=mod is not None,
        num_warps=num_warps or _warps_for(BLOCK),
    )
    return y, rstd, res


# ---------------------------------------------------------------------------
# RoPE + [B,S,H*D] -> [B,H,S,D]
# ---------------------------------------------------------------------------
@triton.jit
def _rope_transpose_kernel(
    X,
    Out,
    InvFreq,
    Pos,
    S,
    sxs,
    sob,
    soh,
    sos,
    HALF: tl.constexpr,
    BLOCK_S: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    HAS_POS: tl.constexpr,
):
    pid_s = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_h = tl.arange(0, HALF)
    valid = offs_s < S

    xp = X + b * S * sxs + h * 2 * HALF + offs_s[:, None] * sxs + offs_h[None, :]
    x1 = tl.load(xp, mask=valid[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(xp + HALF, mask=valid[:, None], other=0.0).to(tl.float32)

    if APPLY_ROPE:
        if HAS_POS:
            pos = tl.load(Pos + b * S + offs_s, mask=valid, other=0).to(tl.float32)
        else:
            pos = offs_s.to(tl.float32)
        inv = tl.load(InvFreq + offs_h)
        ang = pos[:, None] * inv[None, :]
        c = tl.cos(ang)
        s = tl.sin(ang)
        o1 = x1 * c - x2 * s
        o2 = x2 * c + x1 * s
    else:
        o1 = x1
        o2 = x2

    op = Out + b * sob + h * soh + offs_s[:, None] * sos + offs_h[None, :]
    tl.store(op, o1.to(Out.dtype.element_ty), mask=valid[:, None])
    tl.store(op + HALF, o2.to(Out.dtype.element_ty), mask=valid[:, None])


def rope_transpose(
    x_flat, n_heads, head_dim, position_ids, *, apply_rope=True, cfg=None
):
    """[B, S, n_heads*head_dim] -> [B, n_heads, S, head_dim] (+ RoPE)."""
    B, S, _ = x_flat.shape
    out = torch.empty(
        (B, n_heads, S, head_dim), device=x_flat.device, dtype=x_flat.dtype
    )
    inv = _inv_freq(head_dim, str(x_flat.device)) if apply_rope else x_flat
    if position_ids is not None:
        position_ids = position_ids.contiguous()
    BLOCK_S, nw = cfg or _rope_config(S)
    _rope_transpose_kernel[(triton.cdiv(S, BLOCK_S), n_heads, B)](
        x_flat,
        out,
        inv,
        position_ids if position_ids is not None else x_flat,
        S,
        x_flat.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        HALF=head_dim // 2,
        BLOCK_S=BLOCK_S,
        APPLY_ROPE=apply_rope,
        HAS_POS=position_ids is not None,
        num_warps=nw,
    )
    return out


# ---------------------------------------------------------------------------
# gelu_tanh(gate) * up
# ---------------------------------------------------------------------------
@triton.jit
def _tanh(x):
    """tanh without libdevice: (e-1)/(e+1) with e = exp(2x), clamped."""
    e = tl.exp2(tl.minimum(x * (2.0 * _LOG2E), 60.0))
    return (e - 1.0) / (e + 1.0)


@triton.jit
def _gelu_mul_kernel(G, U, Y, n_elem, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    msk = offs < n_elem
    g = tl.load(G + offs, mask=msk, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=msk, other=0.0).to(tl.float32)
    t = _tanh(_GELU_C0 * (g + _GELU_C1 * g * g * g))
    tl.store(Y + offs, (0.5 * g * (1.0 + t) * u).to(Y.dtype.element_ty), mask=msk)


def gelu_mul(g, u):
    y = torch.empty_like(g)
    n = g.numel()
    BLOCK = 1024
    _gelu_mul_kernel[(triton.cdiv(n, BLOCK),)](g, u, y, n, BLOCK=BLOCK, num_warps=4)
    return y


# ---------------------------------------------------------------------------
# (gated) residual add
# ---------------------------------------------------------------------------
@triton.jit
def _gated_add_kernel(
    A, Bt, Gate, Y, seq, N, BLOCK: tl.constexpr, HAS_GATE: tl.constexpr
):
    row = tl.program_id(0)
    b = row // seq
    offs = tl.arange(0, BLOCK)
    msk = offs < N
    a = tl.load(A + row * N + offs, mask=msk, other=0.0).to(tl.float32)
    v = tl.load(Bt + row * N + offs, mask=msk, other=0.0).to(tl.float32)
    if HAS_GATE:
        g = tl.load(Gate + b * N + offs, mask=msk, other=0.0).to(tl.float32)
        v = v * g
    tl.store(Y + row * N + offs, (a + v).to(Y.dtype.element_ty), mask=msk)


def gated_add(a, b_, gate=None):
    B, S, N = a.shape
    y = torch.empty_like(a)
    BLOCK = _next_pow2(N)
    _gated_add_kernel[(B * S,)](
        a,
        b_,
        gate if gate is not None else y,
        y,
        S,
        N,
        BLOCK=BLOCK,
        HAS_GATE=gate is not None,
        num_warps=_warps_for(BLOCK),
    )
    return y


# ---------------------------------------------------------------------------
# the fused layer forward
# ---------------------------------------------------------------------------
def kernel_function(
    hidden_states,
    input_layernorm_weight,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    post_attention_layernorm_weight,
    gate_proj_weight,
    up_proj_weight,
    down_proj_weight,
    eps,
    attention_mask=None,
    position_ids=None,
    adarms_cond=None,
    input_dense=None,
    post_dense=None,
):
    """Fused Gemma decoder layer, forward only.

    Positional arguments follow ``problem.md``'s ``run()``.  The keyword
    arguments extend it to everything the real openpi call sites need: an
    arbitrary additive ``attention_mask`` ``[B, 1|Hq, S, S]``, explicit RoPE
    ``position_ids`` ``[B, S]`` (default: ``arange(S)``), and the action-expert
    adaRMS path (``adarms_cond`` + ``input_dense``/``post_dense`` ``(w, b)``).
    """
    x = hidden_states
    B, S, H = x.shape
    n_heads = q_proj_weight.shape[0] // HEAD_DIM
    n_kv = k_proj_weight.shape[0] // HEAD_DIM
    scale = HEAD_DIM**-0.5
    eps = float(eps)

    x = x if x.is_contiguous() else x.contiguous()

    mod1 = mod2 = gate1 = gate2 = None
    if adarms_cond is not None:
        if input_dense is None or post_dense is None:
            raise ValueError("adarms_cond given without input_dense/post_dense")
        mod1 = torch.addmm(input_dense[1], adarms_cond, input_dense[0].t())
        mod2 = torch.addmm(post_dense[1], adarms_cond, post_dense[0].t())
        gate1 = mod1[:, 2 * H :].contiguous()
        gate2 = mod2[:, 2 * H :].contiguous()
        mod1, mod2 = mod1.contiguous(), mod2.contiguous()

    # ---- input norm ----
    h, _, _ = norm_forward(x, input_layernorm_weight, eps, mod=mod1)
    h2d = h.view(B * S, H)

    # ---- attention ----
    q = rope_transpose(
        _mm(h2d, q_proj_weight).view(B, S, -1), n_heads, HEAD_DIM, position_ids
    )
    k = rope_transpose(
        _mm(h2d, k_proj_weight).view(B, S, -1), n_kv, HEAD_DIM, position_ids
    )
    v = rope_transpose(
        _mm(h2d, v_proj_weight).view(B, S, -1),
        n_kv,
        HEAD_DIM,
        position_ids,
        apply_rope=False,
    )

    attn_out, _, _ = _attention_forward(q, k, v, attention_mask, scale)
    o = _mm(attn_out.reshape(B * S, n_heads * HEAD_DIM), o_proj_weight).view(B, S, H)

    # ---- post-attention norm (fused with the gated residual) ----
    h, _, res = norm_forward(
        o, post_attention_layernorm_weight, eps, residual=x, gate=gate1, mod=mod2
    )
    h2d = h.view(B * S, H)

    # ---- MLP ----
    act = gelu_mul(_mm(h2d, gate_proj_weight), _mm(h2d, up_proj_weight))
    m = _mm(act, down_proj_weight).view(B, S, H)

    return gated_add(res, m, gate2)
