#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S13: per-prompt greedy divergence point and top-1/top-2 margin.

For two `s11l_flip_stats.py` JSONL arms, reports for every prompt the first
position where the chosen token id differs, the top-1/top-2 margin (`gap`) at
that position on both sides, and the worst margin observed at any divergence.
The project's established口径 is: a flip whose margin is <0.5 nat is a near-tie.

Usage:
    python tools/dcu/s13_flip_gaps.py a.jsonl b.jsonl [--label-a A --label-b B]
"""

import argparse
import json


def load(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["prompt_id"]] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    print(f"{'prompt':10s} {'n':>4s} {'first':>6s} {'gap_a':>8s} {'gap_b':>8s} "
          f"{'min_gap':>8s} {'flips':>6s} {'worst_gap':>9s}")
    for pid in a:
        if pid not in b:
            continue
        ra, rb = a[pid]["rows"], b[pid]["rows"]
        n = min(len(ra), len(rb))
        first, gaps, worst = None, (None, None), 0.0
        flips = 0
        for i in range(n):
            if ra[i]["token"] == rb[i]["token"]:
                continue
            flips += 1
            worst = max(worst, min(ra[i]["gap"], rb[i]["gap"]))
            if first is None:
                first = i
                gaps = (ra[i]["gap"], rb[i]["gap"])
        if first is None:
            print(f"{pid:10s} {n:4d} {'—':>6s} {'—':>8s} {'—':>8s} {'—':>8s} {0:6d} {'—':>9s}")
        else:
            print(f"{pid:10s} {n:4d} {first:6d} {gaps[0]:8.3f} {gaps[1]:8.3f} "
                  f"{min(gaps):8.3f} {flips:6d} {worst:9.3f}")
    print(f"# arms: {args.label_a} vs {args.label_b}; gap = top1 - top2 logprob (nat)")


if __name__ == "__main__":
    main()
