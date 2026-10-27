#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure the hardware constants that feed tools/dcu/sol_baseline.py.

Runs the things a SOL baseline needs and nothing else:

  1. HBM copy bandwidth   (dst.copy_(src))            -> GB/s, 2 x bytes
  2. HBM read bandwidth   (x.sum())                   -> GB/s
  3. fp16 GEMM roof       (4096^3 matmul, rocBLAS)    -> TFLOP/s
  4. decode weight stream (W[N,5120]: M=1/8 GEMM, mv, row-sum)
  5. cross-card P2P copy  (128 MiB, all pairs)        -> GB/s

Every case is warmed up and timed with CUDA events; the reported value is the
median over `--samples` event pairs, so one co-tenant burst does not move it.
Requires an idle device set: it allocates ~5 GB per card.

Usage (inside the GPU container)::

    python3 /tmp/sol_measure.py --devices 0,1,2,3 --json /tmp/sol.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from functools import partial

import torch


def timed(op: Callable[[], object], warmup: int, iters: int) -> float:
    """Median seconds per call of `op` over `iters` event-timed calls."""
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        op()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) / 1e3 for s, e in zip(starts, ends))


def gbs(op: Callable[[], object], nbytes: int, samples: int) -> float:
    return nbytes / timed(op, 3, samples) / 1e9


def measure_device(dev: int, size_gb: float, samples: int) -> dict:
    torch.cuda.set_device(dev)
    out: dict = {"device": dev, "name": torch.cuda.get_device_name(dev)}
    n = int(size_gb * 2**30 // 2)  # fp16 elements

    src = torch.ones(n, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    out["copy_gbs"] = gbs(partial(torch.Tensor.copy_, dst, src), 2 * n * 2, samples)

    big = torch.ones(2 * n, dtype=torch.float16, device="cuda")
    out["read_gbs"] = gbs(partial(torch.Tensor.sum, big), 2 * n * 2, samples)
    del big

    a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    t = timed(partial(torch.matmul, a, b), 3, max(3, samples // 2))
    out["gemm4096_tflops"] = 2 * 4096**3 / t / 1e12
    del a, b

    # Decode-shaped weight streaming: how fast can a kernel read W and still
    # reduce it?  The gap between `sum_dim1` and the GEMM/GEMV rows is the
    # small-M kernel inefficiency that dominates decode.
    sweep = []
    for n_out in (1280, 3584, 4096, 5120, 8704):
        w = torch.randn(n_out, 5120, dtype=torch.float16, device="cuda")
        wt = w.t()
        w_bytes = w.numel() * 2
        x1 = torch.randn(1, 5120, dtype=torch.float16, device="cuda")
        x8 = torch.randn(8, 5120, dtype=torch.float16, device="cuda")
        v = torch.randn(5120, dtype=torch.float16, device="cuda")
        sweep.append(
            {
                "N": n_out,
                "gemm_m1_gbs": gbs(partial(torch.matmul, x1, wt), w_bytes, samples),
                "gemm_m8_gbs": gbs(partial(torch.matmul, x8, wt), w_bytes, samples),
                "mv_gbs": gbs(partial(torch.mv, w, v), w_bytes, samples),
                "sum_dim1_gbs": gbs(partial(torch.sum, w, 1), w_bytes, samples),
            }
        )
        del w, wt, x1, x8, v
        torch.cuda.empty_cache()
    out["gemv_sweep"] = sweep
    del src, dst
    torch.cuda.empty_cache()
    return out


def measure_p2p(devices: list[int]) -> list[dict]:
    n = 128 * 2**20 // 2
    rows = []
    for i in devices:
        for j in devices:
            if i == j:
                continue
            src = torch.ones(n, dtype=torch.float16, device=f"cuda:{i}")
            dst = torch.empty(n, dtype=torch.float16, device=f"cuda:{j}")
            copy = partial(torch.Tensor.copy_, dst, src)
            with torch.cuda.device(j):
                for _ in range(3):
                    copy()
                torch.cuda.synchronize(j)
                torch.cuda.synchronize(i)
                best = min(timed(copy, 0, 3) for _ in range(3))
            rows.append({"src": i, "dst": j, "gbs": n * 2 / best / 1e9})
            del src, dst
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None, help="comma list, default: all")
    ap.add_argument("--size-gb", type=float, default=1.5)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--json", help="write raw results here")
    args = ap.parse_args()

    devices = (
        [int(d) for d in args.devices.split(",")]
        if args.devices
        else list(range(torch.cuda.device_count()))
    )
    result = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device_count": torch.cuda.device_count(),
        "peer_access": [
            [int(torch.cuda.can_device_access_peer(i, j)) for j in devices]
            for i in devices
        ],
        "devices": [measure_device(d, args.size_gb, args.samples) for d in devices],
        "p2p": measure_p2p(devices),
    }
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
