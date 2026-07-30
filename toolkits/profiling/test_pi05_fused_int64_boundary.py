"""Validate Pi0.5 fused-kernel offsets beyond the signed-int32 boundary.

Run this file from a directory containing the preserved ``kernel.py``,
``layer_train.py``, and ``problem.py`` modules.  It launches the production
shape that originally failed (M=154880, N=16384, K=2048), then compares rows
on both sides of the 2**31 flattened-output boundary with an fp32 PyTorch
reference.  The comparison intentionally samples rows so the reference does
not allocate another multi-gigabyte output.
"""

from __future__ import annotations

import gc
import sys

import torch
import torch.nn.functional as F
from kernel import _matmul
from layer_train import _down_bwd_fused, _twoout_mm

M = 154_880
N = 16_384
K = 2_048
ROWS = torch.tensor([0, 1, 131_071, 131_072, M - 2, M - 1], device="cuda")
TOL = 2e-2


def _report(name: str, actual: torch.Tensor, expected: torch.Tensor) -> bool:
    actual = actual.float()
    expected = expected.float()
    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    rel = max_abs / (expected.abs().max().item() + 1e-9)
    finite = torch.isfinite(actual).all().item()
    passed = finite and rel < TOL
    print(
        f"  [{'PASS' if passed else 'FAIL'}] {name:<24} "
        f"max_abs={max_abs:.3e} rel={rel:.3e}"
    )
    return passed


def _clear(*values: torch.Tensor) -> None:
    del values
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return 0

    torch.manual_seed(20260722)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    dtype = torch.bfloat16
    boundary_row = (2**31) // N
    print(
        f"shape M={M}, N={N}, K={K}; M*N={M * N:,}; "
        f"first row at/after 2**31 offset={boundary_row}"
    )
    ok = True

    a = torch.empty((M, K), device=device, dtype=dtype).normal_(0, 0.02)
    w = torch.empty((N, K), device=device, dtype=dtype).normal_(0, 0.02)
    a_rows = a.index_select(0, ROWS)
    ref_linear = F.linear(a_rows.float(), w.float())

    o1, o2 = _twoout_mm(a, w, mode=0)
    torch.cuda.synchronize()
    ok &= _report("twoout gate linear", o1.index_select(0, ROWS), ref_linear)
    ok &= _report(
        "twoout gate gelu",
        o2.index_select(0, ROWS),
        F.gelu(ref_linear, approximate="tanh"),
    )

    u, product = _twoout_mm(a, w, mode=1, other=o2)
    torch.cuda.synchronize()
    ok &= _report("twoout up linear", u.index_select(0, ROWS), ref_linear)
    ok &= _report(
        "twoout up product",
        product.index_select(0, ROWS),
        ref_linear * o2.index_select(0, ROWS).float(),
    )
    del o1, o2, u, product
    gc.collect()
    torch.cuda.empty_cache()

    out = _matmul(a, w)
    torch.cuda.synchronize()
    ok &= _report("generic matmul", out.index_select(0, ROWS), ref_linear)
    del a, w, a_rows, ref_linear, out
    gc.collect()
    torch.cuda.empty_cache()

    dout = torch.empty((M, K), device=device, dtype=dtype).normal_(0, 0.02)
    wd = torch.empty((K, N), device=device, dtype=dtype).normal_(0, 0.02)
    gl = torch.empty((M, N), device=device, dtype=dtype).normal_(0, 0.1)
    u_in = torch.empty((M, N), device=device, dtype=dtype).normal_(0, 0.1)
    gate = F.gelu(gl, approximate="tanh")
    dgl, du = _down_bwd_fused(dout, wd, gl, u_in, gate)
    torch.cuda.synchronize()

    dout_rows = dout.index_select(0, ROWS).float()
    dprod = dout_rows @ wd.float()
    gl_rows = gl.index_select(0, ROWS).float().requires_grad_(True)
    gelu_rows = F.gelu(gl_rows, approximate="tanh")
    (dgelu,) = torch.autograd.grad(gelu_rows, gl_rows, torch.ones_like(gelu_rows))
    expected_dgl = dgelu * (dprod * u_in.index_select(0, ROWS).float())
    expected_du = dprod * gate.index_select(0, ROWS).float()
    ok &= _report("down backward dgl", dgl.index_select(0, ROWS), expected_dgl)
    ok &= _report("down backward du", du.index_select(0, ROWS), expected_du)

    peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"peak allocated memory: {peak_gib:.2f} GiB")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
