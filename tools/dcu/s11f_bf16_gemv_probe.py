#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11f: dense fp16 / bf16 decode-GEMV probe for the gfx906 emulation kernel.

Compares ``torch.ops._rocm_C.bf16_emu_gemv`` against the current vLLM fallback
(`torch.nn.functional.linear` -> rocBLAS) on the shapes that matter for a
27B-class model at decode batch sizes, and checks the results against an fp32
reference.

Usage (inside the DTK container):

    . /mnt/game/port/setenv-211.sh
    HIP_VISIBLE_DEVICES=2 python tools/dcu/s11f_bf16_gemv_probe.py
"""

import argparse
import sys

import torch
import torch.nn.functional as F

SHAPES = [
    (17408, 5120),
    (8192, 5120),
    (7168, 5120),
    (5120, 5120),
    (512, 5120),
    (48, 5120),
]


def bench(fn, warmup=10, iters=30):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--check", action="store_true", default=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU visible", file=sys.stderr)
        return 2
    import vllm  # noqa: F401  (loads _rocm_C)

    if not hasattr(torch.ops._rocm_C, "bf16_emu_gemv"):
        print("torch.ops._rocm_C.bf16_emu_gemv is not registered", file=sys.stderr)
        return 2

    torch.manual_seed(0)
    print(f"# device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    worst = 0.0
    for dtype in (torch.float16, torch.bfloat16):
        for n, k in SHAPES:
            w = (torch.randn(n, k, device="cuda") * 0.02).to(dtype)
            for m in args.ms:
                x = (torch.randn(m, k, device="cuda") * 0.5).to(dtype)
                ref = F.linear(x.to(torch.float32), w.to(torch.float32))
                out = torch.ops._rocm_C.bf16_emu_gemv(x, w, None, 64, 0, False)
                ulp = (
                    torch.finfo(dtype).eps
                    * ref.abs().max().clamp_min(1e-30).item()
                )
                err = (
                    (out.to(torch.float32) - ref.to(dtype).to(torch.float32))
                    .abs()
                    .max()
                    .item()
                )
                worst = max(worst, err / ulp)

                ms_emu = bench(
                    lambda: torch.ops._rocm_C.bf16_emu_gemv(x, w, None, 64, 0, False)
                )
                ms_ref = bench(lambda: F.linear(x, w))
                bytes_moved = n * k * x.element_size()
                print(
                    f"RESULT {str(dtype):>16} M={m:<3} N={n:<6} K={k:<6} "
                    f"emu={ms_emu:8.4f} ms ({bytes_moved / ms_emu / 1e6:6.1f} GB/s) "
                    f"rocblas={ms_ref:8.4f} ms  speedup={ms_ref / ms_emu:5.2f}x  "
                    f"err={err / ulp:5.2f} ulp"
                )
                del x
            del w
            torch.cuda.empty_cache()
    print(f"# worst error over all cases: {worst:.2f} ulp of the output dtype")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
