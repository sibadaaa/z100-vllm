#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11l: gate the diagnostic `VLLM_DCU_SDOT4_INNER` switch.

With `VLLM_DCU_SDOT4_INNER=sdot4` the opaque op must be bit-identical to the
raw `_rocm_C` kernels (S11k K7 invariant); with `=triton` it must be
bit-identical to `triton_scaled_mm` (the `=0` baseline math).  Run once per
value and compare the printed checksums.

Usage: VLLM_DCU_SDOT4_INNER=<sdot4|triton> python s11l_inner_probe.py
"""

import os

import torch  # noqa: F401  (must precede `import vllm`, hard rule 8)

from vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4 import (
    _GEMV_MAX_M,
    _SDOT4_INNER,
)
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
    triton_scaled_mm,
)


def ref_raw(x_q, w, x_s, w_s, bias, dtype, m):
    if m <= _GEMV_MAX_M:
        return torch.ops._rocm_C.w8a8_sdot4_gemv(x_q, w, x_s, w_s, bias, dtype, 64)
    return torch.ops._rocm_C.w8a8_sdot4_gemm(x_q, w, x_s, w_s, bias, dtype, 0)


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    print(f"[inner] env={os.environ.get('VLLM_DCU_SDOT4_INNER')} module={_SDOT4_INNER}")
    for m, n, k in ((1, 512, 256), (4, 512, 256), (64, 512, 256), (2048, 512, 256)):
        x = torch.randn(m, k, device=dev, dtype=torch.float16)
        wq = (torch.randn(n, k, device=dev) * 40).to(torch.int8)
        ws = torch.rand(n, device=dev, dtype=torch.float32) * 0.02 + 1e-3
        from vllm import _custom_ops as ops

        x_q, x_s, _ = ops.scaled_int8_quant(x, None, None, symmetric=True)
        x_s = x_s.reshape(-1)
        bias = torch.randn(n, device=dev, dtype=torch.float16) * 0.01
        out = torch.ops.vllm.w8a8_sdot4_mm(x_q, wq, x_s, ws, bias, x.dtype)
        raw = ref_raw(x_q, wq, x_s, ws, bias, x.dtype, m)
        tri = triton_scaled_mm(
            x_q, wq.t(), scale_a=x_s, scale_b=ws, out_dtype=x.dtype, bias=bias
        )
        print(
            f"[inner] M={m:>4} equal_raw={torch.equal(out, raw)} "
            f"equal_triton={torch.equal(out, tri)} "
            f"sum={out.double().sum().item():.6f} "
            f"raw_sum={raw.double().sum().item():.6f} "
            f"tri_sum={tri.double().sum().item():.6f}"
        )


if __name__ == "__main__":
    main()
