#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11l: close the *uncovered prefill M* gap left by S11e/S11k.

S11e checked M in {1,2,4,8,16,32,64,256,512} and S11k added {9..16, 2048} for the
seven real 27B (N, K) pairs.  The S11l prompts prefill at M ~= 17..40 (the
prompts are 2-16 tokens long *plus* the chat template, and the 8K arm prefills
at ~7500), so M in {17..31} -- exactly the range that no run has covered -- is
the most likely place for a real sdot4-vs-Triton numeric difference if one
exists.

Prints one line per (N, K, M) with `bit_equal`; exits non-zero if any mismatch.
"""

import sys

import torch

import vllm._rocm_C  # noqa: F401
import vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4  # noqa: F401
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    triton_scaled_mm,
)

SHAPES = (
    (12288, 5120, "self_attn.q_proj"),
    (1024, 5120, "self_attn.k_proj/v_proj"),
    (5120, 6144, "self_attn.o_proj"),
    (10240, 5120, "linear_attn.in_proj_qkv"),
    (6144, 5120, "linear_attn.in_proj_z"),
    (17408, 5120, "mlp.gate_proj/up_proj"),
    (5120, 17408, "mlp.down_proj"),
)
MS = tuple(range(17, 33)) + (40, 48, 64, 96, 128, 129, 256, 512)


def main() -> None:
    torch.manual_seed(0)
    bad = []
    checked = 0
    for n, k, name in SHAPES:
        wq = (torch.randn(n, k, device="cuda") * 40).to(torch.int8)
        ws = torch.rand(n, device="cuda", dtype=torch.float32) * 0.02 + 1e-3
        for m in MS:
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            xq, xs, _ = ops.scaled_int8_quant(x, None, None, symmetric=True)
            xs = xs.reshape(-1)
            gemv = m <= 8
            if gemv:
                out = torch.ops._rocm_C.w8a8_sdot4_gemv(
                    xq, wq, xs, ws, None, x.dtype, 64
                )
            else:
                out = torch.ops._rocm_C.w8a8_sdot4_gemm(xq, wq, xs, ws, None, x.dtype, 0)
            tri = triton_scaled_mm(
                xq, wq.t(), scale_a=xs, scale_b=ws, out_dtype=x.dtype
            )
            eq = torch.equal(out, tri)
            checked += 1
            if not eq:
                bad.append((name, n, k, m))
                d = (out.float() - tri.float()).abs().max().item()
                print(f"[mrange] MISMATCH {name} M={m} max_abs_diff={d:.4e}")
    print(f"[mrange] checked {checked} combos, mismatches={len(bad)}")
    if bad:
        for b in bad:
            print(f"[mrange] bad {b}")
        sys.exit(1)


if __name__ == "__main__":
    main()
