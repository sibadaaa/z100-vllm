#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11l: how much does dynamic per-token int8 quantization amplify a tiny input
perturbation?

Motivation: two `VLLM_DCU_USE_INT8_SDOT4=1` engines (S11k's cold-compiled one and
S11l's arm1) produced *different top-1/top-2 margins at the first decode step*
(up to ~1 nat), even though the int8 GEMM is bitwise identical to Triton.  A
plain fp16 perturbation of ~1e-3 relative cannot do that.  Dynamic symmetric
int8 quantization (`scaled_int8_quant`) is a *quantizer*: a 1-ULP change in the
input can move elements across a rounding boundary and change the whole int8
code vector by +-1 LSB, i.e. ~1/127 ~ 8e-3 relative -- a discrete jump that then
propagates through 96 layers.

This probe measures that amplification on the seven real 27B (N, K) shapes, for
both the sdot4 GEMV and `triton_scaled_mm`, and compares it with an fp16 matmul
that skips quantization.

Usage: python s11l_quant_sensitivity.py
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
M = 1
# Perturbation sizes: one fp16 ULP near 1.0, then 1e-5 .. 1e-2 relative.
EPS = (2**-11, 2**-10, 2**-9)


def main() -> None:
    torch.manual_seed(0)
    print(f"[qsens] M={M} eps={EPS}")
    print(
        f"{'shape':<26}{'eps':>9}{'int8_codes_changed':>20}"
        f"{'sdot4_rel':>12}{'triton_rel':>12}{'fp16_rel':>12}"
    )
    for n, k, name in SHAPES:
        x = torch.randn(M, k, device="cuda", dtype=torch.float16)
        wq = (torch.randn(n, k, device="cuda") * 40).to(torch.int8)
        ws = torch.rand(n, device="cuda", dtype=torch.float32) * 0.02 + 1e-3
        xq0, xs0, _ = ops.scaled_int8_quant(x, None, None, symmetric=True)
        xs0 = xs0.reshape(-1)
        base = torch.ops._rocm_C.w8a8_sdot4_gemv(xq0, wq, xs0, ws, None, x.dtype, 64)
        base_tri = triton_scaled_mm(
            xq0, wq.t(), scale_a=xs0, scale_b=ws, out_dtype=x.dtype
        )
        # fp16 reference without any quantization
        wf = wq.to(torch.float16) * ws.unsqueeze(1).to(torch.float16)
        base_f16 = x @ wf.t()
        for eps in EPS:
            # perturb every element in the same relative direction; scale eps by
            # the per-element magnitude so that "eps" is a relative perturbation.
            xp = (x.float() * (1.0 + eps)).to(torch.float16)
            if torch.equal(xp, x):
                print(f"{name:<26}{eps:>9.1e}{'no fp16 change':>20}")
                continue
            xq1, xs1, _ = ops.scaled_int8_quant(xp, None, None, symmetric=True)
            xs1 = xs1.reshape(-1)
            codes = int((xq1 != xq0).sum().item())
            out = torch.ops._rocm_C.w8a8_sdot4_gemv(
                xq1, wq, xs1, ws, None, x.dtype, 64
            )
            out_tri = triton_scaled_mm(
                xq1, wq.t(), scale_a=xs1, scale_b=ws, out_dtype=x.dtype
            )
            out_f16 = xp @ wf.t()

            def rel(a, b):
                d = (a.float() - b.float()).abs().max().item()
                return d / max(b.float().abs().max().item(), 1e-6)

            print(
                f"{name:<26}{eps:>9.1e}{codes:>20}"
                f"{rel(out, base):>12.3e}{rel(out_tri, base_tri):>12.3e}"
                f"{rel(out_f16, base_f16):>12.3e}"
            )


if __name__ == "__main__":
    main()
