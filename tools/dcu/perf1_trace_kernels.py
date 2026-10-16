#!/usr/bin/env python3
"""Aggregate a Kineto chrome trace by kernel name (count, total us, grid/block).

Usage:
    python tools/dcu/perf1_trace_kernels.py <trace.json.gz> [--top N]
"""

import argparse
import gzip
import json
from collections import defaultdict


def load(path):
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--match", default="")
    args = ap.parse_args()

    ev = load(args.trace)["traceEvents"]
    agg = defaultdict(lambda: [0, 0.0, set()])
    for e in ev:
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        name = e.get("name", "")
        if args.match and args.match not in name:
            continue
        a = agg[name]
        a[0] += 1
        a[1] += e.get("dur", 0.0)
        args_ = e.get("args", {})
        a[2].add(
            (
                tuple(args_.get("grid", ())) if args_.get("grid") else None,
                tuple(args_.get("block", ())) if args_.get("block") else None,
            )
        )

    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
    print(f"{'kernel':<58}{'n':>7}{'total_ms':>11}{'us/call':>10}  shapes")
    for name, (n, tot, shapes) in rows[: args.top]:
        sh = sorted({s for s in shapes if s[0]})
        sh_s = "; ".join(f"g{g}b{b}" for g, b in sh[:4])
        print(f"{name[:58]:<58}{n:>7}{tot / 1000:>11.1f}{tot / n:>10.1f}  {sh_s}")

    total = sum(v[1] for v in agg.values())
    print(f"\ntotal kernel time: {total / 1000:.1f} ms over {sum(v[0] for v in agg.values())} launches")


if __name__ == "__main__":
    main()
