#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14b: compact per-arm first-token / quality table from greedy JSONL arms.

    s14b_firsttok.py greedy-A1.jsonl greedy-A2.jsonl ...

Prints one row per (arm, prompt): the first three greedy tokens with their
top-1/top-2 margins, the number of generated tokens and a crude degeneration
score (the longest repeated token cycle / generated length), so the "graph
degrades into an English loop" claim can be checked numerically per arm.
"""

import argparse
import json
import sys


def load(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["prompt_id"]] = rec
    return out


def loop_score(ids):
    """Fraction of the stream covered by the longest repeated cycle (period<=16)."""
    n = len(ids)
    best = 0
    for p in range(1, min(17, n)):
        reps = 0
        i = n - p
        while i - p >= 0 and ids[i - p:i] == ids[i:i + p]:
            reps += 1
            i -= p
        best = max(best, (reps + 1) * p)
    return best / n if n else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--ntok", type=int, default=3)
    args = ap.parse_args()

    print(f"{'arm':6s} {'prompt':18s} {'n':>4s} {'loop':>6s} "
          f"{'tok0':>8s} {'gap0':>7s} {'tok1':>8s} {'gap1':>7s} {'tok2':>8s} "
          f"{'ids_sha16':>16s}  text_head")
    for path in args.arms:
        arm = path.split("greedy-")[-1].replace(".jsonl", "")
        data = load(path)
        for pid, rec in data.items():
            rows = rec["rows"]
            toks = [r["token"] for r in rows[:args.ntok]]
            gaps = [r["gap"] for r in rows[:args.ntok]]
            while len(toks) < args.ntok:
                toks.append("")
                gaps.append(float("nan"))
            print(f"{arm:6s} {pid:18s} {len(rows):4d} {loop_score(rec['token_ids']):6.2f} "
                  f"{toks[0]!r:>8s} {gaps[0]:7.3f} {toks[1]!r:>8s} {gaps[1]:7.3f} "
                  f"{toks[2]!r:>8s} {rec['token_ids_sha256'][:16]:>16s}  "
                  f"{rec['text'][:46]!r}")


if __name__ == "__main__":
    sys.exit(main())
