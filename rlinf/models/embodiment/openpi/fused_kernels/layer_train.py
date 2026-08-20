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

"""Autograd-capable training interface for the fused Gemma prefix layer.

``PrefixTrainFn`` is the training entry point: a ``torch.autograd.Function``
whose forward is the fused Triton layer from :mod:`kernel` (keeping the
activations backward needs) and whose backward is a *direct* gradient
computation -- flash-attention backward kernels plus fused Triton kernels for
the RMSNorm, RoPE and gelu-mul stages.  The layer's forward is never recomputed
and the PyTorch reference layer is never called.

Nothing is recomputed: the forward keeps ``x``, both normalised activations and
their reciprocal stds, the RoPE'd q/k/v, the attention output and its
log-sum-exp, the residual, and the MLP's ``gate``/``up``/``act`` -- about 240 MB
at the gemma_2b prefix shape, which buys a backward that is pure gradient
arithmetic.

``use_cache=True``
------------------
Returns ``(hidden_states, k, v)`` where ``k``/``v`` are this layer's RoPE'd keys
and *pre-GQA-expansion* values in HuggingFace's contiguous ``[B, n_kv, S,
head_dim]`` layout, ready to hand to a ``DynamicCache``.  They are the very
tensors attention consumed, so ``hidden_states`` is bit-identical to the
``use_cache=False`` call.  Incoming gradients on those outputs (``grad_k`` /
``grad_v``) are folded into this layer's own ``dk``/``dv`` before the QKV
projections' gradients are formed, so a suffix that trains through a cached
prefix still moves the prefix's weights.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .attention import _attention_backward, _attention_forward
from .kernel import (
    _GELU_C0,
    _GELU_C1,
    _inv_freq,
    _mm,
    _next_pow2,
    _rope_config,
    _tanh,
    _warps_for,
    gated_add,
    gelu_mul,
    norm_forward,
    rope_transpose,
)

__all__ = ["PrefixTrainFn", "prefix_train_forward", "prefix_train_backward"]


# ---------------------------------------------------------------------------
# RMSNorm backward
# ---------------------------------------------------------------------------
@triton.jit
def _rmsnorm_bwd_kernel(
    DY,
    X,
    W,
    Rstd,
    DX,
    DXAcc,
    DWpart,
    rows,
    N,
    BLOCK: tl.constexpr,
    HAS_DXACC: tl.constexpr,
):
    """y = x * r * (1 + w),  r = rsqrt(mean(x^2) + eps)

        dw_i = sum_rows dy_i * x_i * r
        dx_i = r * (g_i*dy_i - (x_i*r) * sum_j(dy_j * x_j * r * g_j) / N)

    Each program walks a strided slice of the rows so the ``dw`` partial stays
    in registers; the [num_programs, N] partials are reduced on the host, which
    keeps the result deterministic (no atomics)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    msk = offs < N

    g = 1.0 + tl.load(W + offs, mask=msk, other=0.0).to(tl.float32)
    dw = tl.zeros([BLOCK], dtype=tl.float32)

    for row in range(pid, rows, nprog):
        dy = tl.load(DY + row * N + offs, mask=msk, other=0.0).to(tl.float32)
        x = tl.load(X + row * N + offs, mask=msk, other=0.0).to(tl.float32)
        r = tl.load(Rstd + row)
        xr = x * r

        dw += dy * xr
        c = tl.sum(dy * xr * g, axis=0)
        dx = r * (g * dy - xr * (c / N))
        if HAS_DXACC:
            dx += tl.load(DXAcc + row * N + offs, mask=msk, other=0.0).to(tl.float32)
        tl.store(DX + row * N + offs, dx.to(DX.dtype.element_ty), mask=msk)

    tl.store(DWpart + pid * N + offs, dw, mask=msk)


def rmsnorm_backward(dy, x, weight, rstd, dx_acc=None, cfg=None):
    """Returns (dx, dw).  ``dx_acc`` is added into dx inside the kernel (the
    residual branch's gradient), saving a separate elementwise pass."""
    B, S, N = x.shape
    rows = B * S
    dx = torch.empty_like(x)
    G, nw = cfg or (min(rows, 512), _warps_for(_next_pow2(N)))
    dw_part = torch.empty((G, N), device=x.device, dtype=torch.float32)
    BLOCK = _next_pow2(N)
    _rmsnorm_bwd_kernel[(G,)](
        dy,
        x,
        weight,
        rstd,
        dx,
        dx_acc if dx_acc is not None else dx,
        dw_part,
        rows,
        N,
        BLOCK=BLOCK,
        HAS_DXACC=dx_acc is not None,
        num_warps=nw,
    )
    return dx, dw_part.sum(0).to(weight.dtype)


# ---------------------------------------------------------------------------
# RoPE backward (+ [B,H,S,D] -> [B,S,H*D])
# ---------------------------------------------------------------------------
@triton.jit
def _rope_bwd_kernel(
    DOut,
    DX,
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

    dp = DOut + b * sob + h * soh + offs_s[:, None] * sos + offs_h[None, :]
    d1 = tl.load(dp, mask=valid[:, None], other=0.0).to(tl.float32)
    d2 = tl.load(dp + HALF, mask=valid[:, None], other=0.0).to(tl.float32)

    if APPLY_ROPE:
        if HAS_POS:
            pos = tl.load(Pos + b * S + offs_s, mask=valid, other=0).to(tl.float32)
        else:
            pos = offs_s.to(tl.float32)
        inv = tl.load(InvFreq + offs_h)
        ang = pos[:, None] * inv[None, :]
        c = tl.cos(ang)
        s = tl.sin(ang)
        # forward: o1 = x1*c - x2*s ; o2 = x2*c + x1*s  (an orthogonal rotation)
        x1 = d1 * c + d2 * s
        x2 = d2 * c - d1 * s
    else:
        x1 = d1
        x2 = d2

    xp = DX + b * S * sxs + h * 2 * HALF + offs_s[:, None] * sxs + offs_h[None, :]
    tl.store(xp, x1.to(DX.dtype.element_ty), mask=valid[:, None])
    tl.store(xp + HALF, x2.to(DX.dtype.element_ty), mask=valid[:, None])


def rope_transpose_backward(d_bhsd, position_ids, *, apply_rope=True):
    """[B, H, S, D] grads -> [B, S, H*D] grads."""
    B, n_heads, S, head_dim = d_bhsd.shape
    out = torch.empty(
        (B, S, n_heads * head_dim), device=d_bhsd.device, dtype=d_bhsd.dtype
    )
    inv = _inv_freq(head_dim, str(d_bhsd.device)) if apply_rope else out
    if position_ids is not None:
        position_ids = position_ids.contiguous()
    BLOCK_S, nw = _rope_config(S)
    _rope_bwd_kernel[(triton.cdiv(S, BLOCK_S), n_heads, B)](
        d_bhsd,
        out,
        inv,
        position_ids if position_ids is not None else out,
        S,
        out.stride(1),
        d_bhsd.stride(0),
        d_bhsd.stride(1),
        d_bhsd.stride(2),
        HALF=head_dim // 2,
        BLOCK_S=BLOCK_S,
        APPLY_ROPE=apply_rope,
        HAS_POS=position_ids is not None,
        num_warps=nw,
    )
    return out


# ---------------------------------------------------------------------------
# gelu_tanh(gate) * up  backward
# ---------------------------------------------------------------------------
@triton.jit
def _gelu_mul_bwd_kernel(DY, G, U, DG, DU, n_elem, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    msk = offs < n_elem
    dy = tl.load(DY + offs, mask=msk, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=msk, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=msk, other=0.0).to(tl.float32)

    t = _tanh(_GELU_C0 * (g + _GELU_C1 * g * g * g))
    gelu = 0.5 * g * (1.0 + t)
    dgelu = 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * _GELU_C0 * (
        1.0 + 3.0 * _GELU_C1 * g * g
    )

    tl.store(DG + offs, (dy * u * dgelu).to(DG.dtype.element_ty), mask=msk)
    tl.store(DU + offs, (dy * gelu).to(DU.dtype.element_ty), mask=msk)


def gelu_mul_backward(dy, g, u):
    dg = torch.empty_like(g)
    du = torch.empty_like(u)
    n = g.numel()
    BLOCK = 1024
    _gelu_mul_bwd_kernel[(triton.cdiv(n, BLOCK),)](
        dy, g, u, dg, du, n, BLOCK=BLOCK, num_warps=4
    )
    return dg, du


# ---------------------------------------------------------------------------
# forward / backward for the whole prefix layer
# ---------------------------------------------------------------------------
def prefix_train_forward(
    x,
    w_ln,
    wq,
    wk,
    wv,
    wo,
    w_pln,
    wg,
    wu,
    wd,
    eps,
    meta,
    attention_mask=None,
    position_ids=None,
    use_cache=False,
):
    """Fused forward that also hands back everything backward needs.

    Returns ``(out, saved)``; ``saved["kv_cache"]`` is ``(k, v)`` when
    ``use_cache``.
    """
    n_heads, n_kv, head_dim = meta
    B, S, H = x.shape
    BS = B * S
    q_dim = n_heads * head_dim
    scale = head_dim**-0.5
    eps = float(eps)

    x = x if x.is_contiguous() else x.contiguous()

    # ---- input RMSNorm ----
    h1, rstd1, _ = norm_forward(x, w_ln, eps)
    h1_2d = h1.view(BS, H)

    # ---- QKV + RoPE (+ layout) ----
    q = rope_transpose(_mm(h1_2d, wq).view(B, S, -1), n_heads, head_dim, position_ids)
    k = rope_transpose(_mm(h1_2d, wk).view(B, S, -1), n_kv, head_dim, position_ids)
    v = rope_transpose(
        _mm(h1_2d, wv).view(B, S, -1), n_kv, head_dim, position_ids, apply_rope=False
    )

    # ---- flash attention ----
    attn, lse, lsum = _attention_forward(q, k, v, attention_mask, scale)
    attn_2d = attn.reshape(BS, q_dim)
    o = _mm(attn_2d, wo).view(B, S, H)

    # ---- residual + post-attention RMSNorm ----
    h2, rstd2, res1 = norm_forward(o, w_pln, eps, residual=x)
    h2_2d = h2.view(BS, H)

    # ---- MLP ----
    gate = _mm(h2_2d, wg)
    up = _mm(h2_2d, wu)
    act = gelu_mul(gate, up)
    m = _mm(act, wd).view(B, S, H)
    out = gated_add(res1, m)

    saved = {
        "x": x,
        "h1": h1,
        "rstd1": rstd1,
        "q": q,
        "k": k,
        "v": v,
        "attn": attn,
        "lse": lse,
        "lsum": lsum,
        "res1": res1,
        "h2": h2,
        "rstd2": rstd2,
        "gate": gate,
        "up": up,
        "act": act,
        "w": (w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd),
        "mask": attention_mask,
        "pos": position_ids,
        "meta": meta,
        "scale": scale,
        "eps": eps,
        "shape": (B, S, H),
    }
    if use_cache:
        saved["kv_cache"] = (k, v)
    return out, saved


def prefix_train_backward(saved, grad_out, dk_cache=None, dv_cache=None):
    """Direct gradients for (x, w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd)."""
    n_heads, n_kv, head_dim = saved["meta"]
    B, S, H = saved["shape"]
    BS = B * S
    q_dim = n_heads * head_dim
    w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd = saved["w"]
    x, pos, mask = saved["x"], saved["pos"], saved["mask"]

    dout = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
    dout_2d = dout.view(BS, H)

    # ---- MLP ----
    dWd = torch.mm(dout_2d.t(), saved["act"])
    dact = torch.mm(dout_2d, wd)
    dgate, dup = gelu_mul_backward(dact, saved["gate"], saved["up"])

    h2_2d = saved["h2"].view(BS, H)
    dWg = torch.mm(dgate.t(), h2_2d)
    dWu = torch.mm(dup.t(), h2_2d)
    dh2 = torch.addmm(torch.mm(dgate, wg), dup, wu).view(B, S, H)

    # ---- post-attention RMSNorm (its dx accumulates the residual branch) ----
    dres1, dw_pln = rmsnorm_backward(
        dh2, saved["res1"], w_pln, saved["rstd2"], dx_acc=dout
    )
    dres1_2d = dres1.view(BS, H)

    # ---- o_proj + attention ----
    attn_2d = saved["attn"].reshape(BS, q_dim)
    dWo = torch.mm(dres1_2d.t(), attn_2d)
    dattn = torch.mm(dres1_2d, wo).view(B, S, n_heads, head_dim)

    dq, dk, dv = _attention_backward(
        dattn,
        saved["q"],
        saved["k"],
        saved["v"],
        mask,
        saved["scale"],
        saved["attn"],
        saved["lse"],
        saved["lsum"],
    )
    # gradients arriving through the exported (cached) K/V
    if dk_cache is not None:
        dk = dk + dk_cache
    if dv_cache is not None:
        dv = dv + dv_cache

    # ---- RoPE backward + QKV projections ----
    dq_f = rope_transpose_backward(dq, pos).view(BS, q_dim)
    dk_f = rope_transpose_backward(dk, pos).view(BS, n_kv * head_dim)
    dv_f = rope_transpose_backward(dv, pos, apply_rope=False).view(BS, n_kv * head_dim)

    h1_2d = saved["h1"].view(BS, H)
    dWq = torch.mm(dq_f.t(), h1_2d)
    dWk = torch.mm(dk_f.t(), h1_2d)
    dWv = torch.mm(dv_f.t(), h1_2d)
    dh1 = torch.addmm(torch.addmm(torch.mm(dq_f, wq), dk_f, wk), dv_f, wv).view(B, S, H)

    # ---- input RMSNorm (its dx accumulates the residual branch) ----
    dx, dw_ln = rmsnorm_backward(dh1, x, w_ln, saved["rstd1"], dx_acc=dres1)

    return dx, dw_ln, dWq, dWk, dWv, dWo, dw_pln, dWg, dWu, dWd


class PrefixTrainFn(torch.autograd.Function):
    """``apply(x, w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd, eps, meta,
    attention_mask=None, position_ids=None, use_cache=False)``."""

    @staticmethod
    def forward(
        ctx,
        x,
        w_ln,
        wq,
        wk,
        wv,
        wo,
        w_pln,
        wg,
        wu,
        wd,
        eps,
        meta,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
    ):
        out, saved = prefix_train_forward(
            x,
            w_ln,
            wq,
            wk,
            wv,
            wo,
            w_pln,
            wg,
            wu,
            wd,
            eps,
            meta,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        ctx.train_ctx = saved
        if use_cache:
            k, v = saved["kv_cache"]
            return out, k, v
        return out

    @staticmethod
    def backward(ctx, grad_out, grad_k=None, grad_v=None):
        g = prefix_train_backward(
            ctx.train_ctx, grad_out, dk_cache=grad_k, dv_cache=grad_v
        )
        # + eps, meta, attention_mask, position_ids, use_cache
        return (*g, None, None, None, None, None)
