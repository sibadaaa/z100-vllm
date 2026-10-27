#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GEMV bandwidth probe: which geometry actually reaches the HBM roof?

Decode is a GEMV (M <= 8) over FP8/FP16 weights.  On gfx906 the same byte
count streams at 784 GB/s in a pure reduction but only 197-292 GB/s through
`torch.mv` / `x @ W.T`.  This probe isolates the knobs that explain the gap:

  ROWS       rows of W per program (how much work one block owns)
  BLOCK_K    K elements per load step (vector width = ROWS*BLOCK_K/threads)
  num_warps  threads per program
  MODE       gemv (dot with x) vs rows-sum (pure read control)

Run on an idle card::

    python3 gemv_probe.py --n 8704 --k 5120
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_rows(
    W,  # [N, K]
    X,  # [K]
    Y,  # [N]
    K,
    stride_wn,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = pid * ROWS + tl.arange(0, ROWS)
    off = tl.arange(0, BLOCK_K)
    acc = tl.zeros((ROWS,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + off
        w = tl.load(W + rn[:, None] * stride_wn + kk[None, :])
        if MODE == 0:
            x = tl.load(X + kk)
            acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
        else:
            acc += tl.sum(w.to(tl.float32), axis=1)
    tl.store(Y + rn, acc)


def bench(fn, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) / 1e3 for s, e in zip(starts, ends))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8704)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--json", help="write raw rows here")
    args = ap.parse_args()
    n, k = args.n, args.k
    torch.cuda.set_device(0)

    w = torch.randn(n, k, dtype=torch.float16, device="cuda")
    x = torch.randn(k, dtype=torch.float16, device="cuda")
    y = torch.empty(n, dtype=torch.float32, device="cuda")
    ref = torch.mv(w, x).float()
    w_bytes = w.numel() * 2

    base = {
        "torch_mv_gbs": w_bytes / bench(lambda: torch.mv(w, x)) / 1e9,
        "torch_sum_dim1_gbs": w_bytes / bench(lambda: w.sum(dim=1)) / 1e9,
        "torch_matmul_gbs": w_bytes / bench(lambda: x[None, :] @ w.t()) / 1e9,
    }

    rows = []
    for r, bk, nw, mode in itertools.product(
        (1, 2, 4, 8), (256, 512, 1024, 2048), (2, 4, 8), (0, 1)
    ):
        if r * bk < 8 * 32 * nw:  # < 8 fp16 per thread: no 16 B vectors
            continue
        if r * bk > 8192:  # keep register pressure sane
            continue
        grid = (n // r,)

        def run(r=r, bk=bk, nw=nw, mode=mode, grid=grid):
            _gemv_rows[grid](
                w,
                x,
                y,
                k,
                w.stride(0),
                BLOCK_K=bk,
                ROWS=r,
                MODE=mode,
                num_warps=nw,
            )

        try:
            t = bench(run, 2, 10)
        except Exception as exc:  # noqa: BLE001 - report, don't abort the sweep
            rows.append(
                {
                    "rows": r,
                    "block_k": bk,
                    "warps": nw,
                    "mode": mode,
                    "gbs": None,
                    "err": str(exc)[:120],
                }
            )
            continue
        rel = ""
        if mode == 0:
            rel = float((y - ref).abs().max() / (ref.abs().max() + 1e-9))
        rows.append(
            {
                "rows": r,
                "block_k": bk,
                "warps": nw,
                "mode": mode,
                "vec_bytes": r * bk * 2 // (32 * nw),
                "gbs": w_bytes / t / 1e9,
                "rel_err": rel,
            }
        )

    rows.sort(key=lambda d: -(d["gbs"] or 0))
    print(json.dumps({"baseline": base, "top": rows[:12]}, indent=1))
    print("\n| rows | BLOCK_K | warps | mode | vec B/thr | GB/s |")
    print("|---:|---:|---:|---:|---:|---:|")
    for d in rows[:20]:
        print(
            f"| {d['rows']} | {d['block_k']} | {d['warps']} | "
            f"{'gemv' if d['mode'] == 0 else 'sum'} | {d['vec_bytes']} | "
            f"{(d['gbs'] or 0):.0f} |"
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"baseline": base, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
