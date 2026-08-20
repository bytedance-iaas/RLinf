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

"""Autograd-capable fused (flash) attention in Triton for the Gemma layer.

Public API
----------
``fused_attention(q, k, v, mask, scale)``
    Differentiable attention.  ``q`` is ``[B, Hq, Sq, D]``, ``k``/``v`` are
    ``[B, Hkv, Sk, D]`` (GQA: ``Hq`` must be a multiple of ``Hkv``), ``mask`` is
    an optional additive bias broadcastable as ``[B, 1 | Hq, Sq, Sk]``.  The
    result is ``[B, Sq, Hq, D]`` -- the layout ``transformers``'
    ``eager_attention_forward`` returns, so it drops into the same call sites.

    ``Sq == Sk`` (joint / square self attention) and ``Sq < Sk`` (a suffix of
    queries consuming a prefix KV cache) are both supported; nothing about the
    kernels assumes a square score matrix or a causal structure -- the entire
    structure comes from the additive ``mask``.

``_torch_attention(q, k, v, mask, scale)``
    The pure-PyTorch equivalent.  It exists as the portable reference the test
    suite compares against; it is *not* used as a fallback by
    ``fused_attention`` on any shape.

Numerics
--------
Scores are accumulated in fp32 and the softmax is an online (flash) fp32
softmax, so the ``[B, Hq, Sq, Sk]`` score matrix is never materialised in
either direction.  The statistics saved for backward are kept in base-2 units,
matching the ``exp2`` used in the kernels, and as *two* tensors -- the row max
and ``log2(sum)`` -- rather than one log-sum-exp.  Summing them would be lossy
on a fully masked row, where the max is pinned at ``NEG_CLAMP * LOG2E`` and the
fp32 ulp at that magnitude swallows ``log2(sum)`` whole; backward would then
rebuild ``p`` a factor of ``Sk`` too large.  openpi produces exactly such rows
(``pad_2d_masks`` masks padded queries along both axes), so this is a live
case, not a corner one.  The split is a private convention between the forward
and backward kernels here.

Masked entries are clamped to ``NEG_CLAMP`` before the exponential.  Both
sentinels used by openpi / HuggingFace (``-2.38e38`` and ``finfo(bf16).min``)
saturate an fp32 score to exactly the sentinel, so a fully-masked row degrades
to a uniform distribution -- which is precisely what ``torch.softmax`` does with
the same input, keeping the two paths consistent instead of producing NaN.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["fused_attention", "_torch_attention"]

LOG2E = tl.constexpr(1.4426950408889634)
NEG_CLAMP = tl.constexpr(-1e30)


# ---------------------------------------------------------------------------
# Pure-torch reference (portable; mirrors transformers eager_attention_forward)
# ---------------------------------------------------------------------------
def _torch_attention(q, k, v, mask, scale):
    """[B,Hq,Sq,D] x [B,Hkv,Sk,D] -> [B,Sq,Hq,D], differentiable."""
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    attn = torch.matmul(q, k.transpose(2, 3)) * scale
    if mask is not None:
        attn = attn + mask[:, :, :, : k.shape[-2]]
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v).transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Launch configuration
# ---------------------------------------------------------------------------
def _fwd_config(D, has_mask):
    # measured on H20 (sm90) at the gemma_2b prefix shape
    if D >= 256:
        return (64, 32, 8, 3) if has_mask else (128, 32, 8, 3)
    if D >= 128:
        return 128, 64, 8, 3
    return 128, 64, 4, 3


def _bwd_dq_config(D, has_mask):
    if D >= 256:
        # the 128-row tile needs more registers than the masked variant has
        return (64, 32, 4, 4) if has_mask else (128, 32, 8, 3)
    return 64, 64, 8, 2


def _bwd_dkdv_config(D, has_mask):
    if D >= 256:
        return 32, 32, 4, 3
    return 64, 64, 8, 2


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
@triton.jit
def _attn_fwd_kernel(
    Q,
    K,
    V,
    Msk,
    Out,
    Lse,
    Lsum,
    sqb,
    sqh,
    sqm,
    sqd,
    skb,
    skh,
    skn,
    skd,
    svb,
    svh,
    svn,
    svd,
    smb,
    smh,
    smm,
    smn,
    sob,
    som,
    soh,
    sod,
    slb,
    slh,
    Sq,
    Sk,
    sm_scale,
    n_heads,
    GROUPS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    b = off_bh // n_heads
    h = off_bh % n_heads
    hkv = h // GROUPS

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_valid = offs_m < Sq

    q = tl.load(
        Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        mask=m_valid[:, None],
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    k_base = K + b * skb + hkv * skh
    v_base = V + b * svb + hkv * svh
    m_base = Msk + b * smb + h * smh

    for start_n in range(0, Sk, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < Sk

        k = tl.load(
            k_base + offs_n[:, None] * skn + offs_d[None, :] * skd,
            mask=n_valid[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if HAS_MASK:
            mv = tl.load(
                m_base + offs_m[:, None] * smm + offs_n[None, :] * smn,
                mask=m_valid[:, None] & n_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            qk = qk + mv
        qk = tl.where(n_valid[None, :], qk, NEG_CLAMP)
        qk = tl.maximum(qk, NEG_CLAMP) * LOG2E

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        # zero the out-of-range lanes so a fully masked row normalises by the
        # real ``Sk`` rather than by the padded block count
        p = tl.where(n_valid[None, :], tl.exp2(qk - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(
            v_base + offs_n[:, None] * svn + offs_d[None, :] * svd,
            mask=n_valid[:, None],
            other=0.0,
        )
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_safe[:, None]

    # ``m_i`` and ``log2(sum)`` are stored separately and never summed: a fully
    # masked row pins ``m_i`` at ``NEG_CLAMP * LOG2E``, whose fp32 ulp dwarfs
    # ``log2(sum)``, so the sum would silently drop the second term and the
    # backward's ``exp2(qk - lse)`` would come out ``Sk`` times too large.
    tl.store(Lse + b * slb + h * slh + offs_m, m_i, mask=m_valid)
    tl.store(Lsum + b * slb + h * slh + offs_m, tl.log2(l_safe), mask=m_valid)
    tl.store(
        Out + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :] * sod,
        acc.to(Out.dtype.element_ty),
        mask=m_valid[:, None],
    )


def _attention_forward(q, k, v, mask, scale, cfg=None):
    """Flash forward.

    Returns ``(out [B,Sq,Hq,D], lse [B,Hq,Sq], lsum [B,Hq,Sq])``; ``lse`` is the
    fp32 row max and ``lsum`` the base-2 log of the softmax denominator, kept
    apart so backward can rebuild ``p`` exactly (see the store in the kernel).
    """
    B, Hq, Sq, D = q.shape
    Hkv, Sk = k.shape[1], k.shape[2]
    assert Hq % Hkv == 0, f"Hq={Hq} not a multiple of Hkv={Hkv}"
    assert k.shape[-1] == D and v.shape[-1] == D
    groups = Hq // Hkv

    out = torch.empty((B, Sq, Hq, D), device=q.device, dtype=q.dtype)
    lse = torch.empty((B, Hq, Sq), device=q.device, dtype=torch.float32)
    lsum = torch.empty_like(lse)

    if mask is not None:
        mask = mask if mask.shape[-1] == Sk else mask[:, :, :, :Sk].contiguous()
        smb, smh, smm, smn = mask.stride()
        if mask.shape[1] == 1:
            smh = 0
        m_ptr = mask
    else:
        m_ptr, smb, smh, smm, smn = q, 0, 0, 0, 0

    BLOCK_M, BLOCK_N, num_warps, num_stages = cfg or _fwd_config(D, mask is not None)
    grid = (triton.cdiv(Sq, BLOCK_M), B * Hq)
    _attn_fwd_kernel[grid](
        q,
        k,
        v,
        m_ptr,
        out,
        lse,
        lsum,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        smb,
        smh,
        smm,
        smn,
        *out.stride(),
        lse.stride(0),
        lse.stride(1),
        Sq,
        Sk,
        scale,
        Hq,
        GROUPS=groups,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HAS_MASK=mask is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, lse, lsum


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------
@triton.jit
def _attn_delta_kernel(
    Out,
    DO,
    Delta,
    sob,
    som,
    soh,
    sod,
    sgb,
    sgm,
    sgh,
    sgd,
    sdb,
    sdh,
    Sq,
    n_heads,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Delta[b,h,m] = sum_d out[b,m,h,d] * do[b,m,h,d] (fp32)."""
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    b = off_bh // n_heads
    h = off_bh % n_heads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_valid = offs_m < Sq

    o = tl.load(
        Out + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :] * sod,
        mask=m_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO + b * sgb + h * sgh + offs_m[:, None] * sgm + offs_d[None, :] * sgd,
        mask=m_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(Delta + b * sdb + h * sdh + offs_m, tl.sum(o * do, 1), mask=m_valid)


@triton.jit
def _attn_bwd_dq_kernel(
    Q,
    K,
    V,
    Msk,
    DO,
    DQ,
    Lse,
    Lsum,
    Delta,
    sqb,
    sqh,
    sqm,
    sqd,
    skb,
    skh,
    skn,
    skd,
    svb,
    svh,
    svn,
    svd,
    smb,
    smh,
    smm,
    smn,
    sgb,
    sgm,
    sgh,
    sgd,
    sxb,
    sxh,
    sxm,
    sxd,
    slb,
    slh,
    Sq,
    Sk,
    sm_scale,
    n_heads,
    GROUPS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    b = off_bh // n_heads
    h = off_bh % n_heads
    hkv = h // GROUPS

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_valid = offs_m < Sq

    q = tl.load(
        Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
        mask=m_valid[:, None],
        other=0.0,
    )
    do = tl.load(
        DO + b * sgb + h * sgh + offs_m[:, None] * sgm + offs_d[None, :] * sgd,
        mask=m_valid[:, None],
        other=0.0,
    )
    lse = tl.load(Lse + b * slb + h * slh + offs_m, mask=m_valid, other=0.0)
    lsum = tl.load(Lsum + b * slb + h * slh + offs_m, mask=m_valid, other=0.0)
    delta = tl.load(Delta + b * slb + h * slh + offs_m, mask=m_valid, other=0.0)

    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    k_base = K + b * skb + hkv * skh
    v_base = V + b * svb + hkv * svh
    m_base = Msk + b * smb + h * smh

    for start_n in range(0, Sk, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < Sk

        k = tl.load(
            k_base + offs_n[:, None] * skn + offs_d[None, :] * skd,
            mask=n_valid[:, None],
            other=0.0,
        )
        v = tl.load(
            v_base + offs_n[:, None] * svn + offs_d[None, :] * svd,
            mask=n_valid[:, None],
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if HAS_MASK:
            mv = tl.load(
                m_base + offs_m[:, None] * smm + offs_n[None, :] * smn,
                mask=m_valid[:, None] & n_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            qk = qk + mv
        qk = tl.where(n_valid[None, :], qk, NEG_CLAMP)
        qk = tl.maximum(qk, NEG_CLAMP) * LOG2E

        # (qk - max) first -- both carry the clamp's magnitude and cancel
        # exactly -- then the denominator, which is a small number
        p = tl.where(
            n_valid[None, :], tl.exp2((qk - lse[:, None]) - lsum[:, None]), 0.0
        )
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None]) * sm_scale).to(k.dtype)
        dq = tl.dot(ds, k, dq)

    tl.store(
        DQ + b * sxb + h * sxh + offs_m[:, None] * sxm + offs_d[None, :] * sxd,
        dq,
        mask=m_valid[:, None],
    )


@triton.jit
def _attn_bwd_dkdv_kernel(
    Q,
    K,
    V,
    Msk,
    DO,
    DK,
    DV,
    Lse,
    Lsum,
    Delta,
    sqb,
    sqh,
    sqm,
    sqd,
    skb,
    skh,
    skn,
    skd,
    svb,
    svh,
    svn,
    svd,
    smb,
    smh,
    smm,
    smn,
    sgb,
    sgm,
    sgh,
    sgd,
    spb,
    sph,
    spn,
    spd,
    slb,
    slh,
    Sq,
    Sk,
    sm_scale,
    n_heads,
    GROUPS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """One program per (key block, query head): writes per-q-head fp32 partials
    at DK/DV[b, h, n, :].  Reducing the partials over the GQA group happens on
    the host, which keeps the kernel free of atomics and the result
    deterministic."""
    start_n = tl.program_id(0)
    off_bh = tl.program_id(1)
    b = off_bh // n_heads
    h = off_bh % n_heads
    hkv = h // GROUPS

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    n_valid = offs_n < Sk

    k = tl.load(
        K + b * skb + hkv * skh + offs_n[:, None] * skn + offs_d[None, :] * skd,
        mask=n_valid[:, None],
        other=0.0,
    )
    v = tl.load(
        V + b * svb + hkv * svh + offs_n[:, None] * svn + offs_d[None, :] * svd,
        mask=n_valid[:, None],
        other=0.0,
    )

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    m_base = Msk + b * smb + h * smh

    for start_m in range(0, Sq, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_valid = offs_m < Sq

        q = tl.load(
            Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd,
            mask=m_valid[:, None],
            other=0.0,
        )
        do = tl.load(
            DO + b * sgb + h * sgh + offs_m[:, None] * sgm + offs_d[None, :] * sgd,
            mask=m_valid[:, None],
            other=0.0,
        )
        lse = tl.load(Lse + b * slb + h * slh + offs_m, mask=m_valid, other=0.0)
        lsum = tl.load(Lsum + b * slb + h * slh + offs_m, mask=m_valid, other=0.0)
        delta = tl.load(Delta + b * slb + h * slh + offs_m, mask=m_valid, other=0.0)

        qkT = tl.dot(k, tl.trans(q)) * sm_scale  # [BLOCK_N, BLOCK_M]
        if HAS_MASK:
            mv = tl.load(
                m_base + offs_m[None, :] * smm + offs_n[:, None] * smn,
                mask=m_valid[None, :] & n_valid[:, None],
                other=0.0,
            ).to(tl.float32)
            qkT = qkT + mv
        qkT = tl.where(n_valid[:, None] & m_valid[None, :], qkT, NEG_CLAMP)
        qkT = tl.maximum(qkT, NEG_CLAMP) * LOG2E

        pT = tl.exp2((qkT - lse[None, :]) - lsum[None, :])
        pT = tl.where(n_valid[:, None] & m_valid[None, :], pT, 0.0)
        dv = tl.dot(pT.to(do.dtype), do, dv)

        dpT = tl.dot(v, tl.trans(do))
        dsT = (pT * (dpT - delta[None, :]) * sm_scale).to(q.dtype)
        dk = tl.dot(dsT, q, dk)

    tl.store(
        DK + b * spb + h * sph + offs_n[:, None] * spn + offs_d[None, :] * spd,
        dk,
        mask=n_valid[:, None],
    )
    tl.store(
        DV + b * spb + h * sph + offs_n[:, None] * spn + offs_d[None, :] * spd,
        dv,
        mask=n_valid[:, None],
    )


def _attention_backward(do, q, k, v, mask, scale, out, lse, lsum, cfg=None, cfg2=None):
    """Returns (dq, dk, dv) with the layouts of (q, k, v)."""
    B, Hq, Sq, D = q.shape
    Hkv, Sk = k.shape[1], k.shape[2]
    groups = Hq // Hkv

    do = do.contiguous() if do.stride(-1) != 1 else do
    delta = torch.empty((B, Hq, Sq), device=q.device, dtype=torch.float32)

    if mask is not None:
        mask = mask if mask.shape[-1] == Sk else mask[:, :, :, :Sk].contiguous()
        smb, smh, smm, smn = mask.stride()
        if mask.shape[1] == 1:
            smh = 0
        m_ptr = mask
    else:
        m_ptr, smb, smh, smm, smn = q, 0, 0, 0, 0

    has_mask = mask is not None
    BLOCK_M, BLOCK_N, num_warps, num_stages = cfg or _bwd_dq_config(D, has_mask)
    BLOCK_M2, BLOCK_N2, num_warps2, num_stages2 = cfg2 or _bwd_dkdv_config(D, has_mask)

    _attn_delta_kernel[(triton.cdiv(Sq, 64), B * Hq)](
        out,
        do,
        delta,
        *out.stride(),
        *do.stride(),
        delta.stride(0),
        delta.stride(1),
        Sq,
        Hq,
        D=D,
        BLOCK_M=64,
        num_warps=4,
    )

    dq = torch.empty((B, Hq, Sq, D), device=q.device, dtype=torch.float32)
    _attn_bwd_dq_kernel[(triton.cdiv(Sq, BLOCK_M), B * Hq)](
        q,
        k,
        v,
        m_ptr,
        do,
        dq,
        lse,
        lsum,
        delta,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        smb,
        smh,
        smm,
        smn,
        *do.stride(),
        *dq.stride(),
        lse.stride(0),
        lse.stride(1),
        Sq,
        Sk,
        scale,
        Hq,
        GROUPS=groups,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HAS_MASK=mask is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # per-q-head fp32 partials, reduced over the GQA group afterwards
    dk_p = torch.empty((B, Hq, Sk, D), device=q.device, dtype=torch.float32)
    dv_p = torch.empty((B, Hq, Sk, D), device=q.device, dtype=torch.float32)
    _attn_bwd_dkdv_kernel[(triton.cdiv(Sk, BLOCK_N2), B * Hq)](
        q,
        k,
        v,
        m_ptr,
        do,
        dk_p,
        dv_p,
        lse,
        lsum,
        delta,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        smb,
        smh,
        smm,
        smn,
        *do.stride(),
        *dk_p.stride(),
        lse.stride(0),
        lse.stride(1),
        Sq,
        Sk,
        scale,
        Hq,
        GROUPS=groups,
        D=D,
        BLOCK_M=BLOCK_M2,
        BLOCK_N=BLOCK_N2,
        HAS_MASK=mask is not None,
        num_warps=num_warps2,
        num_stages=num_stages2,
    )
    if groups == 1:
        dk, dv = dk_p.to(k.dtype), dv_p.to(v.dtype)
    else:
        dk = dk_p.view(B, Hkv, groups, Sk, D).sum(2).to(k.dtype)
        dv = dv_p.view(B, Hkv, groups, Sk, D).sum(2).to(v.dtype)
    return dq.to(q.dtype), dk, dv


class _AttnFn(torch.autograd.Function):
    """Flash attention forward + a true flash backward (no score matrix, no
    forward recompute through a PyTorch reference)."""

    @staticmethod
    def forward(ctx, q, k, v, mask, scale):
        # The bwd kernels index q/k/v through explicit strides, so a transposed
        # (non-contiguous) q -- what `[B,S,H,D].transpose(1,2)` hands us -- is
        # safe here; only the innermost dim has to be unit-stride.
        q = q if q.stride(-1) == 1 else q.contiguous()
        k = k if k.stride(-1) == 1 else k.contiguous()
        v = v if v.stride(-1) == 1 else v.contiguous()
        out, lse, lsum = _attention_forward(q, k, v, mask, scale)
        ctx.save_for_backward(
            q, k, v, out, lse, lsum, mask if mask is not None else None
        )
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse, lsum, mask = ctx.saved_tensors
        dq, dk, dv = _attention_backward(do, q, k, v, mask, ctx.scale, out, lse, lsum)
        return dq, dk, dv, None, None


def fused_attention(q, k, v, mask=None, scale=None):
    """Autograd-capable fused attention; see the module docstring."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _AttnFn.apply(q, k, v, mask, float(scale))
