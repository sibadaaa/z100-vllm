#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11f: numeric + performance probe for the gfx906 FP8 emulation kernels.

Validates ``torch.ops._rocm_C.fp8_emu_gemv`` / ``..._gemm`` against an fp32
reference on the *exact* operand tensors the kernel consumes (so the only
expected error is the fp32 accumulation order, not the fp8 quantization), and
measures them against the best available fp16 baseline (rocBLAS) plus the
fp16 roofline (weight bytes / HBM bandwidth).

Run inside the DTK container:

    . /mnt/game/port/setenv-211.sh
    HIP_VISIBLE_DEVICES=2 python tools/dcu/s11f_fp8_emu_probe.py --mode check
    HIP_VISIBLE_DEVICES=2 python tools/dcu/s11f_fp8_emu_probe.py --mode bench
"""

import argparse
import sys

import torch

# 27B-class shapes: (N, K).
SHAPES = [
    (17408, 5120),
    (8192, 5120),
    (7168, 5120),
    (5120, 5120),
    (512, 5120),
    (48, 5120),
]
MS = [1, 2, 4, 8, 16, 64, 128, 512, 1024]


def _lut(fp8_dtype: torch.dtype) -> torch.Tensor:
    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_fp8 import (
        _device_lut,
        fp8_to_fp16_lut,
    )

    lut = fp8_to_fp16_lut(fp8_dtype)
    return _device_lut(fp8_dtype, torch.device("cuda")) if torch.cuda.is_available() else lut


def quantize(x: torch.Tensor, per_token: bool, fp8_dtype: torch.dtype):
    """Symmetric fp8 quantization along the last dim; returns (q, scale)."""
    finfo = torch.finfo(fp8_dtype)
    if per_token:
        amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = (amax / finfo.max).to(torch.float32)
    else:
        amax = x.abs().amax().clamp_min(1e-12)
        scale = (amax / finfo.max).to(torch.float32).reshape(1)
    q = (x / scale).clamp(finfo.min, finfo.max).to(fp8_dtype)
    return q, scale.reshape(-1)


def reference(a_q, a_s, w_q, w_s):
    a = a_q.to(torch.float32) * a_s.reshape(-1, 1).to(torch.float32)
    w = w_q.to(torch.float32) * w_s.reshape(-1, 1).to(torch.float32)
    return a @ w.t()


def check(args) -> int:
    torch.manual_seed(0)
    device = "cuda"
    worst = 0.0
    failures = 0
    total = 0
    for fp8_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        lut = _lut(fp8_dtype)
        for n, k in SHAPES:
            # Weight distribution: per-channel scaled (typical compressed-tensors).
            w = torch.randn(n, k, device=device) * 0.02
            w_q, w_s = quantize(w, per_token=False, fp8_dtype=fp8_dtype)
            for m in [x for x in MS if x <= args.max_m]:
                a = torch.randn(m, k, device=device) * 0.5
                for per_token in (True, False):
                    a_q, a_s = quantize(a, per_token=per_token, fp8_dtype=fp8_dtype)
                    ref = reference(a_q, a_s, w_q, w_s)
                    for out_dtype in (torch.float16, torch.float32):
                        if m <= 8:
                            out = torch.ops._rocm_C.fp8_emu_gemv(
                                a_q, w_q, a_s, w_s, lut, None, out_dtype, 64
                            )
                        else:
                            out = torch.ops._rocm_C.fp8_emu_gemm(
                                a_q, w_q, a_s, w_s, lut, None, out_dtype, 0
                            )
                        err = (out.to(torch.float32) - ref).abs()
                        # Normalize by the tile scale, not element-wise: a
                        # near-zero reference element makes element-wise rel
                        # error meaningless, while max|ref| is the standard
                        # GEMM error metric.
                        scale = ref.abs().max().clamp_min(1e-12)
                        norm = (err.max() / scale).item()
                        abs_err = err.max().item()
                        tol = args.rel_tol if out_dtype == torch.float32 else args.abs_tol
                        ok = norm < tol
                        total += 1
                        if not ok:
                            failures += 1
                            print(
                                f"FAIL {fp8_dtype} M={m} N={n} K={k} "
                                f"per_token={per_token} out={out_dtype} "
                                f"max|err|/max|ref|={norm:.3e} abs={abs_err:.3e}"
                            )
                        worst = max(worst, norm)
                    # Same-tensor reference through the fp16 LUT widening: this
                    # isolates the kernel's widening from the quantization.
        print(f"# checked {fp8_dtype}")

    print(
        f"# check: {total - failures}/{total} cases within tolerance "
        f"(worst max|err|/max|ref|={worst:.3e}, fp32_tol={args.rel_tol:g}, "
        f"abs_tol={args.abs_tol:g})"
    )
    return 1 if failures else 0


def bench(args) -> int:
    torch.manual_seed(0)
    device = "cuda"
    fp8_dtype = torch.float8_e4m3fn
    lut = _lut(fp8_dtype)

    def timeit(fn, warmup=10, iters=30):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    print(f"# device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    for n, k in SHAPES:
        w = torch.randn(n, k, device=device) * 0.02
        w_q, w_s = quantize(w, per_token=False, fp8_dtype=fp8_dtype)
        w_fp16 = (w_q.to(torch.float16) * w_s.reshape(-1, 1).to(torch.float16))
        del w
        for m in args.ms:
            a = torch.randn(m, k, device=device) * 0.5
            a_q, a_s = quantize(a, per_token=True, fp8_dtype=fp8_dtype)
            a_fp16 = a.to(torch.float16)
            wbytes = n * k  # fp8: one byte per weight
            ms_emu = timeit(
                lambda: (
                    torch.ops._rocm_C.fp8_emu_gemv(
                        a_q, w_q, a_s, w_s, lut, None, torch.float16, 64
                    )
                    if m <= 8
                    else torch.ops._rocm_C.fp8_emu_gemm(
                        a_q, w_q, a_s, w_s, lut, None, torch.float16, 0
                    )
                )
            )
            ms_fp16 = timeit(lambda: torch.nn.functional.linear(a_fp16, w_fp16))
            print(
                f"RESULT M={m:<5} N={n:<6} K={k:<6} emu={ms_emu:8.3f} ms "
                f"({wbytes / ms_emu / 1e6:7.1f} GB/s) "
                f"fp16_rocblas={ms_fp16:8.3f} ms  speedup={ms_fp16 / ms_emu:5.2f}x"
            )
            del a, a_q, a_s
        del w_q, w_s, w_fp16
        torch.cuda.empty_cache()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "bench", "both"], default="check")
    ap.add_argument("--rel-tol", type=float, default=1e-3)
    ap.add_argument("--abs-tol", type=float, default=1e-3)
    ap.add_argument("--max-m", type=int, default=128)
    ap.add_argument("--ms", type=int, nargs="*", default=[1, 8, 64, 512])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU visible", file=sys.stderr)
        return 2
    rc = 0
    if args.mode in ("check", "both"):
        rc |= check(args)
    if args.mode in ("bench", "both"):
        rc |= bench(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
