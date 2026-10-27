#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""S22: is the DFlash2 draft tower's bf16 matmul on the best available kernel?

The FP8 decode trace (2026-10-16) shows `triton_matmul_kernel` = 29.8 ms/step
(13.2% of a 225 ms step, 81.6 calls/step, 365.6 us median).  The draft tower is
bf16, so `vllm/model_executor/layers/utils.py::rocm_unquantized_gemm_impl`
selects, for M <= 16 on gfx906, `triton_matmul` -- whose autotune space has a
single config (BM=min(16,M), BN=64, BK=64, 2 warps, 3 stages).  The gfx906
emulated bf16 GEMV (`bf16_emu_gemv`) is never reached because
`_gfx906_gemv_shape_ok()` rejects n_tokens >= 3 for wide N, and rocBLAS
(`F.linear`) is only reached for M > 16.

This probe measures all three at the draft tower's real shapes so the dispatch
question can be decided by numbers rather than by the shape-gate comment.

  HIP_VISIBLE_DEVICES=0 python tools/dcu/s22_draft_gemm_probe.py \
      [--ms 1,2,4,8,16,32] [--iters 30] [--outdir /mnt/game/s22]
"""

import argparse
import json
import os

import torch  # noqa: F401  (must precede vllm imports)


def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,2,4,8,16,32")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--outdir", default="/mnt/game/s22")
    ap.add_argument("--threads", default="64,128")
    args = ap.parse_args()

    from vllm.model_executor.layers.utils import (
        _register_gfx906_dense_gemv,
        triton_matmul,
    )

    _register_gfx906_dense_gemv()
    import torch.nn.functional as F

    # DFlash2 draft tower, TP=4 per rank (config.json: 5 layers, hidden 5120,
    # 32 q heads x 128, 8 kv heads, intermediate 17408).
    shapes = [
        (34816, 5120),  # gate_up
        (5120, 17408),  # down_proj
        (6144, 5120),  # qkv
        (4096, 5120),  # o_proj
        (1536, 5120),  # selector / small
    ]
    ms_list = [int(x) for x in args.ms.split(",")]
    threads_list = [int(x) for x in args.threads.split(",")]
    dt = torch.bfloat16
    rows = []

    print(f"device={torch.cuda.get_device_name(0)} dtype={dt}")
    print(
        f"{'N:K':>13}{'M':>5}{'triton ms':>11}{'rocblas ms':>12}"
        f"{'emu ms':>10}{'roc/triton':>11}{'emu/triton':>11}{'GB/s(tri)':>11}"
    )
    for n, k in shapes:
        w = (torch.randn(n, k, device="cuda") * 0.02).to(dt)
        wb = w.t().contiguous()  # [K, N] for F.linear-free triton path check
        del wb
        for m in ms_list:
            x = (torch.randn(m, k, device="cuda") * 0.5).to(dt)
            t_tri = bench(lambda: triton_matmul(x, w), iters=args.iters)
            t_roc = bench(lambda: F.linear(x, w), iters=args.iters)
            # bf16_emu_gemv only accepts M <= 8 (kMsMax in bf16_emu_gemv.cu).
            best_emu, best_thr = float("nan"), 0
            if m <= 8:
                best_emu = float("inf")
                for thr in threads_list:
                    t = bench(
                        lambda: torch.ops._rocm_C.bf16_emu_gemv(
                            x, w, None, thr, 0, False
                        ),
                        iters=args.iters,
                    )
                    if t < best_emu:
                        best_emu, best_thr = t, thr
            gbs = n * k * 2 / t_tri / 1e6
            row = dict(
                shape=f"{n}:{k}",
                n=n,
                k=k,
                m=m,
                triton_ms=t_tri,
                rocblas_ms=t_roc,
                emu_ms=best_emu,
                emu_threads=best_thr,
                roc_over_triton=t_roc / t_tri,
                emu_over_triton=best_emu / t_tri,
                triton_gbs=gbs,
            )
            rows.append(row)
            print(
                f"{row['shape']:>13}{m:>5}{t_tri:>11.4f}{t_roc:>12.4f}"
                f"{best_emu:>10.4f}{row['roc_over_triton']:>11.2f}"
                f"{row['emu_over_triton']:>11.2f}{gbs:>11.1f}"
            )
        del w

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "draft_gemm_probe.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
