# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S7b: characterize the gfx906 ``_C.silu_and_mul`` corruption.

Facts established with ``s7b_activation_probe.py``: on gfx906 / DTK 26.04 the
fp16 ``_C.silu_and_mul`` kernel (``csrc/activation_kernels.cu:173``,
``silu_kernel`` + ``act_and_mul_kernel``) returns garbage for ~1.8% of the
elements while ``F.silu(gate) * up`` is exact.  This script narrows down which
part of the kernel is at fault:

* is the corruption in the activation (``silu_kernel``) or in the gating mul?
  -> compare ``silu_and_mul`` with ``mul_and_silu`` (same kernel, ACT_FIRST
     swapped) and with ``gelu_and_mul`` (no ``expf``, uses ``erf``);
* does it depend on the dtype -> half / bfloat16 / float;
* does it depend on the shape / launch geometry -> vary d and num_tokens;
* where are the bad elements -> index histogram over the flattened output.

    python tools/dcu/s7b_activation_matrix.py
"""

import torch  # noqa: F401

import vllm._C  # noqa: F401

OPS = {
    "silu_and_mul": (torch.ops._C.silu_and_mul, lambda g, u: torch.nn.functional.silu(g) * u),
    "mul_and_silu": (torch.ops._C.mul_and_silu, lambda g, u: g * torch.nn.functional.silu(u)),
    "gelu_and_mul": (torch.ops._C.gelu_and_mul, lambda g, u: torch.nn.functional.gelu(g) * u),
}


def run(name, x):
    op, ref_fn = OPS[name]
    d = x.shape[-1] // 2
    out = torch.full(x.shape[:-1] + (d,), float("nan"), dtype=x.dtype, device=x.device)
    op(out, x)
    torch.cuda.synchronize()
    ref = ref_fn(x[..., :d], x[..., d:])
    o = out.float()
    r = ref.float()
    nbad = int((~torch.isfinite(o)).sum())
    err = (o - r).abs()
    err = torch.where(torch.isfinite(err), err, torch.full_like(err, float("inf")))
    return nbad, o.numel(), err.max().item(), int((err > 1e-2).sum())


def main():
    torch.manual_seed(0)
    print("=== dtype sweep, shape (5, 6144), input in [-5.9, 5.9] ===")
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = ((torch.rand(5, 6144, device="cuda") * 2 - 1) * 5.9).to(dtype)
        for name in OPS:
            nbad, n, maxerr, nerr = run(name, x)
            print(f"[s7b-mat] {name:<14} {str(dtype):<16} nonfinite={nbad}/{n} maxerr={maxerr:.6g} nerr>1e-2={nerr}")

    print("=== constants, fp16, shape (2, 6144) ===")
    for c in (0.0, 1.0, -1.0, 5.9, -5.9, 20.0, -20.0):
        x = torch.full((2, 6144), c, device="cuda", dtype=torch.float16)
        nbad, n, maxerr, nerr = run("silu_and_mul", x)
        print(f"[s7b-mat] silu_and_mul const={c:<6} nonfinite={nbad}/{n} maxerr={maxerr:.6g} nerr>1e-2={nerr}")

    print("=== shape sweep, fp16, silu_and_mul ===")
    for rows, cols in ((1, 6144), (2, 6144), (5, 6144), (5, 2048), (5, 12288), (64, 6144), (512, 6144)):
        x = ((torch.rand(rows, cols, device="cuda") * 2 - 1) * 5.9).half()
        nbad, n, maxerr, nerr = run("silu_and_mul", x)
        print(f"[s7b-mat] silu_and_mul rows={rows:<4} 2d={cols:<6} nonfinite={nbad}/{n} maxerr={maxerr:.6g} nerr>1e-2={nerr}")

    print("=== bad-index pattern, fp16, shape (5, 6144) ===")
    x = ((torch.rand(5, 6144, device="cuda") * 2 - 1) * 5.9).half()
    op, ref_fn = OPS["silu_and_mul"]
    d = x.shape[-1] // 2
    out = torch.empty((5, d), dtype=x.dtype, device=x.device)
    op(out, x)
    torch.cuda.synchronize()
    err = (out.float() - ref_fn(x[:, :d], x[:, d:]).float()).abs()
    bad = (err > 1e-2)
    idx = bad.nonzero()
    print(f"[s7b-mat] bad={int(bad.sum())} tokens={sorted(set(idx[:, 0].tolist()))}")
    counts = bad.sum(dim=0)
    nz = counts.nonzero().flatten()
    print(f"[s7b-mat] bad cols: min={int(nz.min())} max={int(nz.max())} ncols={nz.numel()}")
    # launch geometry is block=min(d,1024): idx, idx+1024, idx+2048 for d=3072
    for k in range(0, d, 1024):
        seg = counts[k : k + 1024]
        print(f"[s7b-mat] iter {k // 1024}: bad={int(seg.sum())} bad_threads={int((seg > 0).sum())}")
    rows = counts[:8].tolist()
    print(f"[s7b-mat] first 8 col counts={rows}")
    print("[s7b-mat] DONE")


main()
