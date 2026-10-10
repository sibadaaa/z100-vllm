#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11k: close the one M gap left by S11e's `--mode check`.

S11e verified sdot4 == triton_scaled_mm bit-for-bit at M in
{1,2,4,8,16,32,64,256,512} for the seven real 27B (N, K) pairs.  The S11k
client prompt prefills at M ~= 9-15, which no run had covered, so this script
re-checks exactly those M for all seven shapes.

A mismatch here would be a *numeric* explanation for the greedy divergence
(the decode step itself is M=1 and was covered), while a full match pushes the
explanation to the rest of the compiled graph.
"""

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
MS = (9, 10, 11, 12, 13, 14, 15, 16, 2048)


def main() -> None:
    torch.manual_seed(0)
    bad = []
    for n, k, name in SHAPES:
        w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        w_s = (torch.rand(n, device="cuda") * 0.02 + 1e-4).float()
        for m in MS:
            x = (torch.randn(m, k, device="cuda") * 2).half()
            x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
            x_s = x_s.reshape(-1)
            mine = torch.ops.vllm.w8a8_sdot4_mm(x_q, w, x_s, w_s, None, torch.float16)
            tr = triton_scaled_mm(
                x_q, w.t(), scale_a=x_s, scale_b=w_s, out_dtype=torch.float16
            )
            same = bool(torch.equal(mine, tr))
            if not same:
                bad.append((name, n, k, m, float((mine.float() - tr.float()).abs().max())))
            print(f"  N={n:6d} K={k:6d} M={m:5d} bit_equal_triton={same}  # {name}")
    print(f"[mcheck] checked {len(SHAPES) * len(MS)} combos, mismatches={len(bad)}")
    for row in bad:
        print("[mcheck] MISMATCH", row)


if __name__ == "__main__":
    main()
