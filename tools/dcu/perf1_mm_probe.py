#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Perf1: price the three `triton_matmul_kernel` decode fallbacks + gemv routes.

Trace evidence (rank0, 50 decode steps): `triton_matmul_kernel` = 2658 calls,
742.1 ms, 14.84 ms/step (6.7% of the 219.6 ms step), in three grid groups.
`vllm/model_executor/layers/utils.py::triton_matmul` is reached from
`rocm_unquantized_gemm_impl` for every *unquantized* linear with n <= 16 tokens
(`n <= 16 and not on_gfx9()`); production runs it with BLOCK_SIZE_N=64.

Module identification (three independent lines of evidence, all consistent):

  grid (1,1,1)   48/step  GDN `in_proj_ba` (fused b+a): per-rank N=24, K=5120.
        evidence: 2400 = 48 layers x 50 steps; the previous kernel is always the
        fused `fp8_emu_gemm_block_kernel` for in_proj_qkvz (per-rank N=4096 ->
        grid 64); the checkpoint stores in_proj_b/in_proj_a separately, and one
        fused ba call per layer is the only fit for 48/step.
  grid (20,1,1)  ~2/step  MTP `mtp.fc` (ColumnParallelLinear, per-rank N=1280,
        K=10240): 20*64 = 1280 = 5120/4; the preceding kernel is always a `Cat`
        (the 2*hidden MTP input concat) and the following one an AllGather
        (gather_output=True) then an RMSNorm.
  grid (970,1,1) ~3/step  `lm_head` vocab shard: per-rank N=62080 = 248320/4,
        K=5120; 970*64 = 62080, and the next kernel is an AllGather_Ring (the
        vocab-parallel logit gather) in 155/156 cases.

All three weights are BF16 in the checkpoint and are cast to FP16 by
`--dtype float16` (the trace instantiates c10::Half everywhere).

This probe re-measures the real shapes against every available route:
Triton (current), F.linear/rocBLAS, and the pre-existing gfx906 emulated GEMV
`torch.ops._rocm_C.bf16_emu_gemv` (kMsMax=8, one block per output column), which
`_gfx906_gemv_shape_ok` currently admits only for n_tokens==1.

  HIP_VISIBLE_DEVICES=1 python tools/dcu/perf1_mm_probe.py

Note the two-part fix the numbers feed: the production recipe exports
`S11F_BF16_GEMV=0` -> `VLLM_DCU_USE_BF16_GEMV=0`, so the whole dense GEMV path
is off today (`_gfx906_dense_gemv_enabled`, utils.py:453-464) and the Triton
fallback at utils.py:562 catches all three modules.  Any win therefore needs
(a) the env enabled and (b) `_gfx906_gemv_shape_ok` widened from today's
`n_tokens == 1` / `n_tokens == 4 and bf16 and 128 <= n_out <= 1024` to the
cells measured here.
"""

import argparse

import torch

from vllm.model_executor.layers.utils import (
    _GFX906_GEMV_AVAILABLE,
    triton_matmul,
)


def bench(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def gemv(x, w, threads, route):
    return torch.ops._rocm_C.bf16_emu_gemv(x, w, None, threads, route, False)


def rel_err(out, x, w, max_cols=512):
    # fp64 reference on an output-column subset: the full lm_head reference
    # would need a 1.27 GB fp64 weight copy, and GPU 1 is shared with the
    # TP=4 service (--gpu-memory-utilization 0.85).
    cols = min(w.shape[0], max_cols)
    ref = x.double() @ w[:cols].double().t()
    denom = ref.abs().max().clamp_min(1e-6)
    return ((out[:, :cols].double() - ref).abs().max() / denom).item()


def measure(name, M, K, N, dtype, iters, threads_list):
    torch.manual_seed(3)
    x = (torch.randn(M, K, device="cuda", dtype=dtype) * 0.1).contiguous()
    w = (torch.randn(N, K, device="cuda", dtype=dtype) * 0.05).contiguous()
    wbytes = N * K * x.element_size()
    print(f"\n{name}: M={M} K={K} N={N} dtype={dtype} weights={wbytes / 1e6:.2f} MB")
    try:
        out = triton_matmul(x, w)
        us = bench(lambda: triton_matmul(x, w), iters)
        print(
            f"  {'triton_matmul (current)':28s} {us:9.1f} us "
            f"{wbytes / (us * 1e-6) / 1e9:7.1f} GB/s  rel_err={rel_err(out, x, w):.2e}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  triton_matmul failed: {type(exc).__name__}: {exc}")

    try:
        us = bench(lambda: torch.nn.functional.linear(x, w), iters)
        print(
            f"  {'F.linear (rocBLAS)':28s} {us:9.1f} us "
            f"{wbytes / (us * 1e-6) / 1e9:7.1f} GB/s"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  F.linear failed: {type(exc).__name__}: {exc}")

    if not threads_list or not _GFX906_GEMV_AVAILABLE:
        return
    for threads in threads_list:
        for route in (0, 1, 2):
            label = f"bf16_emu_gemv nt={threads} r={route}"
            try:
                out = gemv(x, w, threads, route)
                us = bench(lambda: gemv(x, w, threads, route), iters)
                print(
                    f"  {label:28s} {us:9.1f} us "
                    f"{wbytes / (us * 1e-6) / 1e9:7.1f} GB/s  "
                    f"rel_err={rel_err(out, x, w):.2e}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {label} failed: {type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--threads", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = ap.parse_args()
    dt = getattr(torch, args.dtype)
    print(
        f"device={torch.cuda.get_device_name(0)} "
        f"bf16_emu_gemv={_GFX906_GEMV_AVAILABLE} dtype={args.dtype}"
    )

    # (A) the in_proj_ba routing question: M in {1,2,3,4}, N in {24,48}, K=5120
    for n_out in (24, 48):
        for m in (1, 2, 3, 4):
            measure(
                f"in_proj_ba N={n_out}", m, 5120, n_out, dt, args.iters, args.threads
            )
    # (B),(C) the other two fallbacks, same routes and M range
    for m in (1, 2, 3, 4):
        measure("mtp.fc", m, 10240, 1280, dt, args.iters, args.threads)
    for m in (1, 2, 3, 4):
        measure("lm_head", m, 5120, 62080, dt, args.iters, args.threads)
    if args.dtype == "float16":
        measure(
            "in_proj_ba N=24 (bf16 spot check)",
            3,
            5120,
            24,
            torch.bfloat16,
            args.iters,
            args.threads,
        )
        measure(
            "mtp.fc (bf16 spot check)",
            3,
            10240,
            1280,
            torch.bfloat16,
            args.iters,
            args.threads,
        )


if __name__ == "__main__":
    raise SystemExit(main())
