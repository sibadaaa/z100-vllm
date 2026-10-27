# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S7b: scope of the gfx906 packed-activation defect, with a relative metric.

Runs the gating activations registered in
``csrc/libtorch_stable/activation_kernels.cu`` through the ``__half2`` packed
branch (``d % 8 == 0`` -> ``use_vec == true``, ``:250-256``) for fp16 and bf16
and reports, per dtype, the non-finite count and the largest *relative* error
against an fp32 reference.  A pure fp16/bf16 rounding error stays below ~1e-3
relative; the defect shows up as relative errors of order 1 or as inf/NaN.

    python tools/dcu/s7b_activation_scope.py
"""

import torch  # noqa: F401

import vllm._C  # noqa: F401

rows, d = 64, 3072
ones = torch.ones(1, device="cuda")


def metrics(out, ref):
    o = out.float()
    nfin = int((~torch.isfinite(o)).sum())
    diff = (o - ref).abs()
    rel = diff / ref.abs().clamp_min(1e-2)
    rel = torch.where(torch.isfinite(rel), rel, torch.full_like(rel, float("inf")))
    return nfin, int((rel > 0.05).sum()), rel.max().item()


CASES = {
    "silu_and_mul": (
        lambda out, x: torch.ops._C.silu_and_mul(out, x),
        lambda g, u: torch.nn.functional.silu(g.float()) * u.float(),
    ),
    "mul_and_silu": (
        lambda out, x: torch.ops._C.mul_and_silu(out, x),
        lambda g, u: g.float() * torch.nn.functional.silu(u.float()),
    ),
    "silu_and_mul_with_clamp": (
        lambda out, x: torch.ops._C.silu_and_mul_with_clamp(out, x, 20.0, 1.0, 0.0),
        lambda g, u: torch.nn.functional.silu(torch.clamp(g.float(), max=20.0))
        * torch.clamp(u.float(), min=-20.0, max=20.0),
    ),
    "gelu_and_mul": (
        lambda out, x: torch.ops._C.gelu_and_mul(out, x),
        lambda g, u: torch.nn.functional.gelu(g.float()) * u.float(),
    ),
}

for dtype in (torch.float16, torch.bfloat16):
    torch.manual_seed(5)
    x = ((torch.rand(rows, 2 * d, device="cuda") * 2 - 1) * 5.9).to(dtype)
    g, u = x[:, :d], x[:, d:]
    print(f"--- dtype={dtype} shape={tuple(x.shape)} d={d} (vec path) ---")
    for name, (fn, ref_fn) in CASES.items():
        ref = ref_fn(g, u)
        worst_nfin = 0
        worst_rel_n = 0
        worst_rel = 0.0
        runs_bad = 0
        for _ in range(10):
            out = torch.empty((rows, d), dtype=dtype, device=x.device)
            fn(out, x)
            torch.cuda.synchronize()
            nfin, nrel, mrel = metrics(out, ref)
            worst_nfin = max(worst_nfin, nfin)
            worst_rel_n = max(worst_rel_n, nrel)
            worst_rel = max(worst_rel, mrel)
            runs_bad += (nfin > 0) or (mrel > 0.05)
        print(
            f"[s7b-scope] {dtype} {name:<24} runs_bad={runs_bad}/10 "
            f"worst_nonfinite={worst_nfin} worst_rel>0.05={worst_rel_n} worst_rel={worst_rel:.6g}"
        )

# Scalar branch (d not a multiple of 8) for the same data, fp16: must be clean.
torch.manual_seed(5)
d2 = d - 4
x = ((torch.rand(rows, 2 * d2, device="cuda") * 2 - 1) * 5.9).half()
out = torch.empty((rows, d2), dtype=x.dtype, device=x.device)
torch.ops._C.silu_and_mul(out, x)
torch.cuda.synchronize()
ref = torch.nn.functional.silu(x[:, :d2].float()) * x[:, d2:].float()
print(f"[s7b-scope] fp16 scalar-branch d={d2} (d%8={d2 % 8}) metrics={metrics(out, ref)}")
print("[s7b-scope] DONE")
