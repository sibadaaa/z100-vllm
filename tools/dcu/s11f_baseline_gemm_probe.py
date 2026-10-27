#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11f baseline: dense fp16 / bf16 / fp32 GEMM+GEMV on gfx906 (rocBLAS).

This measures the *current* unquantized-vLLM path (``torch.nn.functional.linear``
-> rocBLAS) for the shapes that matter on a 27B-class model, so that the new
gfx906 BF16/FP8 emulation kernels have a baseline to beat.  It deliberately does
not use vLLM: it is a pure torch-side probe, safe to run on any free card.

Usage (inside the DTK container):

    . /mnt/game/port/setenv-211.sh
    HIP_VISIBLE_DEVICES=2 python tools/dcu/s11f_baseline_gemm_probe.py

Output: one ``RESULT`` line per (dtype, shape, M) with ms / GFLOP/s / GB/s.
"""

import argparse
import sys

import torch
import torch.nn.functional as F

# 27B-class layer shapes: (out_features, in_features).
DEFAULT_SHAPES = [
    (17408, 5120),  # fused gate/up
    (8192, 5120),
    (7168, 5120),
    (5120, 5120),  # qkv / o / down
    (512, 5120),  # small expert / lm-head slice
    (48, 5120),  # tp-sharded tail
]
DEFAULT_MS = [1, 2, 4, 8, 16, 64, 128, 512, 1024, 4096]


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtypes", default="float16,bfloat16,float32")
    ap.add_argument("--ms", default=",".join(str(m) for m in DEFAULT_MS))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU visible", file=sys.stderr)
        return 2

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print(f"# device={name} arch={props.gcnArchName} sm={props.multi_processor_count}")
    print(f"# torch={torch.__version__} hip={torch.version.hip}")
    print(f"# warmup={args.warmup} iters={args.iters}")

    dtypes = [getattr(torch, d.strip()) for d in args.dtypes.split(",")]
    ms = [int(m) for m in args.ms.split(",")]

    for dtype in dtypes:
        for n, k in DEFAULT_SHAPES:
            w = (torch.randn(n, k, dtype=torch.float32, device="cuda") * 0.02).to(dtype)
            for m in ms:
                x = (torch.randn(m, k, dtype=torch.float32, device="cuda") * 0.02).to(
                    dtype
                )
                try:
                    ms_per = bench(lambda: F.linear(x, w), args.warmup, args.iters)
                except Exception as exc:  # noqa: BLE001 - report and keep going
                    print(f"RESULT {dtype} M={m} N={n} K={k} ERROR {exc}")
                    continue
                flops = 2.0 * m * n * k
                bytes_moved = (m * k + n * k + m * n) * x.element_size()
                gflops = flops / (ms_per * 1e6)
                print(
                    f"RESULT {str(dtype):>16} M={m:<5} N={n:<6} K={k:<6} "
                    f"{ms_per:8.3f} ms  {gflops:8.1f} GFLOP/s  "
                    f"{bytes_moved / ms_per / 1e6:7.1f} GB/s"
                )
                del x
            del w
            torch.cuda.empty_cache()

    print(f"# peak_mem_allocated={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
