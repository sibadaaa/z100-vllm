# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S8 W8A8 INT8 numeric check on gfx906 using the fork's own Triton kernel.

``TritonInt8ScaledMMLinearKernel`` (the only ROCm INT8 kernel left once aiter is
unavailable) calls ``triton_scaled_mm``, which on gfx906 relies on the self-built
triton's ``supportsVDot() == VEGA`` support so that the int8 dot maps to
``v_dot4_i32_i8`` instead of an emulated FMA path.

    python tools/dcu/s8_int8_gfx906_check.py

Prints ``INT8_SCALED_MM_OK`` / ``INT8_SCALED_MM_FAIL`` plus the measured error.
"""

from __future__ import annotations

import sys


def main() -> int:
    import torch
    import importlib.metadata as md
    import triton

    print(f"torch dist  = {md.version('torch')} hip={torch.version.hip}")
    print(f"triton dist = {md.version('triton')} / {triton.__file__}")
    print(f"device      = {torch.cuda.get_device_properties(0).gcnArchName}")

    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )

    torch.manual_seed(0)
    dev = "cuda"
    failures = 0
    for (m, k, n, per_token, per_channel) in [
        (16, 512, 256, True, True),
        (128, 1024, 512, True, True),
        (256, 2048, 1024, True, True),
        (1, 1024, 512, True, True),
        (64, 512, 256, False, True),
        (64, 512, 256, True, False),
    ]:
        # Quantize a fp16 reference GEMM to int8 the same way the W8A8 path does.
        a_f = (torch.randn(m, k, device=dev) * 2.0).half()
        b_f = (torch.randn(k, n, device=dev) * 2.0).half()

        if per_token:
            a_scale = a_f.abs().amax(dim=1, keepdim=True).float() / 127.0
        else:
            a_scale = a_f.abs().amax().float().reshape(1, 1) / 127.0
        if per_channel:
            b_scale = b_f.abs().amax(dim=0, keepdim=True).float().t() / 127.0
        else:
            b_scale = b_f.abs().amax().float().reshape(1, 1) / 127.0

        a_q = torch.clamp((a_f.float() / a_scale).round(), -128, 127).to(torch.int8)
        b_q = torch.clamp((b_f.float() / b_scale.t()).round(), -128, 127).to(torch.int8)

        ref = (a_q.float() @ b_q.float()) * a_scale * b_scale.t()
        out = triton_scaled_mm(
            a_q, b_q, a_scale, b_scale, torch.float16
        ).float()

        rel = ((out - ref).abs().max() / ref.abs().max().clamp(min=1e-6)).item()
        ok = rel < 5e-2
        failures += 0 if ok else 1
        print(
            f"{'OK  ' if ok else 'FAIL'} M={m:4d} K={k:5d} N={n:5d} "
            f"per_token={per_token} per_channel={per_channel} rel_err={rel:.3e}"
        )

    print("INT8_SCALED_MM_OK" if failures == 0 else f"INT8_SCALED_MM_FAIL n={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
