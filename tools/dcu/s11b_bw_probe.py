#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11b: measure the achievable device memory bandwidth / fp16 throughput.

Sanity anchor for the "the int8 GEMM runs at ~39 GB/s of weight reads"
observation: measure a plain device-to-device copy (read+write) and an fp16
matmul on the same card, via torch only (no self-built binaries).
"""

import time

import torch

N_BYTES = 512 * 1024 * 1024


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print(f"device={name} cu={props.multi_processor_count} mem={props.total_memory / 2**30:.1f} GiB")

    n = N_BYTES // 2
    a = torch.empty(n, dtype=torch.float16, device="cuda")
    b = torch.empty(n, dtype=torch.float16, device="cuda")
    a.fill_(1.0)
    dt = bench(lambda: b.copy_(a))
    moved = N_BYTES * 2  # read + write
    print(f"copy 512 MiB fp16: {dt * 1e3:.3f} ms -> {moved / dt / 1e9:.1f} GB/s (read+write)")

    ai = torch.empty(N_BYTES, dtype=torch.int8, device="cuda")
    bi = torch.empty(N_BYTES, dtype=torch.int8, device="cuda")
    ai.fill_(1)
    dt = bench(lambda: bi.copy_(ai))
    print(f"copy 512 MiB int8: {dt * 1e3:.3f} ms -> {N_BYTES * 2 / dt / 1e9:.1f} GB/s (read+write)")

    # fp16 GEMV-ish shapes taken from the 27B decode path.
    for m, k, nn in ((1, 5120, 8704), (1, 8704, 5120), (1, 5120, 124160)):
        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        w = torch.randn(nn, k, dtype=torch.float16, device="cuda")
        dt = bench(lambda x=x, w=w: torch.nn.functional.linear(x, w), iters=10)
        flops = 2 * m * k * nn
        wbytes = nn * k * 2
        print(
            f"fp16 linear M={m} N={nn} K={k}: {dt * 1e3:.3f} ms "
            f"-> {flops / dt / 1e12:.2f} TFLOP/s, weight read {wbytes / dt / 1e9:.1f} GB/s"
        )


if __name__ == "__main__":
    main()
