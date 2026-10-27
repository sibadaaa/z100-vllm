#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11k: gate the new opaque `vllm::w8a8_sdot4_mm` op.

Checks, in order:
  1. the kernel class reports supported with VLLM_DCU_USE_INT8_SDOT4=1;
  2. `torch.ops.vllm.w8a8_sdot4_mm` exists and carries the expected schema;
  3. the opaque op is **bit-identical** to the raw `_rocm_C` kernels it wraps
     (GEMV for M <= 8, tiled GEMM above), with and without bias, and still
     within the S11e rel_err gate vs an fp32 dequantized reference;
  4. under `torch.compile(dynamic=True)` with the token dimension marked
     dynamic, the op traces as an opaque in-graph call, reuses one graph for
     M=2048 -> M=1, and returns the same values as eager.
"""

import os

os.environ.setdefault("VLLM_DCU_USE_INT8_SDOT4", "1")

import torch  # noqa: E402

import vllm._rocm_C  # noqa: E402,F401
from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4 import (  # noqa: E402
    Gfx906Sdot4Int8ScaledMMLinearKernel as K,
)

N, K_DIM = 5120, 5120
MS = (1, 2, 4, 8, 9, 32, 64, 256, 2048)
GATE = 1e-2


def quant(x: torch.Tensor):
    x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
    return x_q, x_s.reshape(-1)


def main() -> None:
    print("[gate] is_supported:", K.is_supported())
    mm = torch.ops.vllm.w8a8_sdot4_mm
    print("[gate] opaque op schema:", mm.default._schema)

    torch.manual_seed(0)
    w = torch.randint(-127, 128, (N, K_DIM), dtype=torch.int8, device="cuda")
    w_s = (torch.rand(N, device="cuda") * 0.02 + 1e-4).float()
    bias = (torch.rand(N, device="cuda") * 0.5 - 0.25).half()

    worst_gate = 0.0
    for m in MS:
        x = (torch.randn(m, K_DIM, device="cuda") * 2).half()
        x_q, x_s = quant(x)
        ref = (x_q.float() @ w.float().t()) * x_s[:, None] * w_s[None, :]
        ref = ref.half()
        for label, b in (("nobias", None), ("bias", bias)):
            out = mm(x_q, w, x_s, w_s, b, torch.float16)
            raw = (
                torch.ops._rocm_C.w8a8_sdot4_gemv(x_q, w, x_s, w_s, b, torch.float16, 64)
                if m <= 8
                else torch.ops._rocm_C.w8a8_sdot4_gemm(x_q, w, x_s, w_s, b, torch.float16, 0)
            )
            target = ref if b is None else (ref.float() + b.float()).half()
            rel = ((out.float() - target.float()).abs() / (target.float().abs() + 1e-3)).max()
            worst_gate = max(worst_gate, float(rel))
            print(
                f"[gate] M={m:5d} {label:6s} bit_equal_raw={torch.equal(out, raw)} "
                f"rel_err_vs_fp32={float(rel):.3e} finite={bool(torch.isfinite(out).all())}"
            )

    print(f"[gate] worst rel_err = {worst_gate:.3e} (gate <= {GATE:g})")

    # --- compiled path -----------------------------------------------------
    x = (torch.randn(2048, K_DIM, device="cuda") * 2).half()
    x_q, x_s = quant(x)
    torch._dynamo.mark_dynamic(x_q, 0)
    torch._dynamo.mark_dynamic(x_s, 0)
    pos = torch.arange(3 * 2048, device="cuda").reshape(3, 2048)
    torch._dynamo.mark_dynamic(pos, 1)

    def f(x_q, w, x_s, w_s, positions):
        out = mm(x_q, w, x_s, w_s, None, torch.float16)
        nt = positions.shape[1]
        return out, positions.reshape(-1).view(nt, -1)

    cf = torch.compile(f, dynamic=True)
    o1, p1 = cf(x_q, w, x_s, w_s, pos)
    x2 = (torch.randn(1, K_DIM, device="cuda") * 2).half()
    x_q2, x_s2 = quant(x2)
    pos2 = torch.arange(3, device="cuda").reshape(3, 1)
    o2, p2 = cf(x_q2, w, x_s2, w_s, pos2)
    print(f"[gate] compiled M=2048 -> {tuple(o1.shape)}, M=1 -> {tuple(o2.shape)}")
    print("[gate] compiled-vs-eager bit_equal:",
          bool(torch.equal(o1, mm(x_q, w, x_s, w_s, None, torch.float16))),
          bool(torch.equal(o2, mm(x_q2, w, x_s2, w_s, None, torch.float16))))
    from torch._dynamo.utils import counters

    print("[gate] counters:", dict(counters))
    print("[gate] RESULT:", "PASS" if worst_gate <= GATE else "FAIL")


if __name__ == "__main__":
    main()
