#!/usr/bin/env python3
"""Sum the weight bytes each fp8-emu GEMM call streams, per decode step.

Per call the kernel reads its whole [N, K] weight tile. `grid.x == ceil(N/64)`
for the <32,64,4,4> instantiation, so N is recoverable from the trace; K is not
in the trace args, so it is supplied per call set via --k (default 5120, the
model's hidden size). The output is an upper bound on the unavoidable weight
traffic and the effective bandwidth achieved, to be compared with the measured
hardware ceiling (776 GB/s flat stream, 670 GB/s spec).

Usage:
    python tools/dcu/perf1_weight_bytes.py <trace.json.gz> [--kernel SUBSTR] [--k 5120]
"""

import argparse
import gzip
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--kernel", default="fp8_emu_gemm_block_kernel<32,64")
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--decimal-bytes", action="store_true")
    args = ap.parse_args()

    with gzip.open(args.trace, "rt") as fh:
        ev = json.load(fh)["traceEvents"]

    hist = defaultdict(int)
    total_us = 0.0
    n = 0
    for e in ev:
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        if args.kernel not in e.get("name", ""):
            continue
        grid = e.get("args", {}).get("grid")
        if not grid:
            continue
        gx = grid[0]
        hist[gx] += 1
        n += 1
        total_us += e.get("dur", 0.0)

    if not n:
        raise SystemExit("no matching kernel events")

    n_bytes = sum(gx * 64 * args.k for gx, c in hist.items() for _ in range(c))
    gb = n_bytes / 1e9
    secs = total_us / 1e6
    print(f"kernel filter      : {args.kernel}")
    print(f"calls              : {n}")
    print(f"grid.x histogram   : {dict(sorted(hist.items()))}")
    print(f"assumed K          : {args.k}")
    print(f"bytes streamed     : {gb:.1f} GB")
    print(f"kernel wall time   : {secs:.3f} s")
    print(f"effective BW       : {gb / secs:.1f} GB/s")
    print(f"at 670 GB/s would be: {gb / 670 * 1000:.1f} ms")
    print(f"at 776 GB/s would be: {gb / 776 * 1000:.1f} ms")


if __name__ == "__main__":
    main()
