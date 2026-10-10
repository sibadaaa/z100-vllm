#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14: TP=2 vs TP=4 per-token logprob comparison (G2).

Reads two JSONL arms produced by ``s11l_flip_stats.py`` (and/or
``s14_greedy.py``) and reports, per prompt:

  * the length of the identical-token prefix,
  * over that common prefix, the max / mean |delta logprob| (nat),
  * at the first divergent position, the chosen token and the top-1/top-2
    margin (``gap``) on both sides, plus the worst margin over all divergences.

The verdict is deliberately split into "numerical near-tie" (every divergence
happened at a top-1/top-2 margin below the project's 0.5 nat threshold) versus
"not a near-tie" so the caller has to explain the mechanism instead of just
reporting "mismatch".

Usage:
    python tools/dcu/s14_logprob_compare.py p4.jsonl p2.jsonl --label-a TP4 --label-b TP2
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
    ap.add_argument("--near-tie", type=float, default=0.5,
                    help="top1-top2 margin (nat) below which a flip is a near-tie")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)

    print(f"# {args.label_a}={args.a}")
    print(f"# {args.label_b}={args.b}")
    print(f"# near-tie threshold: gap < {args.near_tie} nat")
    print(f"{'prompt':14s} {'nA':>4s} {'nB':>4s} {'pre':>4s} {'maxd':>9s} "
          f"{'meand':>9s} {'>0.01':>6s} {'first':>6s} {'tokA':>6s} {'tokB':>6s} "
          f"{'gapA':>8s} {'gapB':>8s} {'minGap':>8s} {'flips':>6s} {'worst':>8s}")

    prompts = 0
    identical = 0
    total_tok = 0
    gmax = 0.0
    gsum = 0.0
    gcnt = 0
    big = 0
    diverged_prompts = []
    all_near_tie = True

    for pid in a:
        if pid not in b:
            continue
        ra, rb = a[pid]["rows"], b[pid]["rows"]
        n = min(len(ra), len(rb))
        prompts += 1
        total_tok += n
        if n == 0:
            print(f"{pid:14s} {len(ra):4d} {len(rb):4d}  (empty)")
            continue

        pre = 0
        dmax = 0.0
        dsum = 0.0
        dbig = 0
        for i in range(n):
            if ra[i]["token"] != rb[i]["token"]:
                break
            pre += 1
            d = abs(ra[i]["logprob"] - rb[i]["logprob"])
            dsum += d
            dmax = max(dmax, d)
            if d > 0.01:
                dbig += 1
        if pre == n:
            identical += 1
        gmax = max(gmax, dmax)
        gsum += dsum
        gcnt += pre
        big += dbig

        first, tok_a, tok_b, gap_a, gap_b, worst = None, "", "", float("nan"), float("nan"), 0.0
        flips = 0
        for i in range(n):
            if ra[i]["token"] == rb[i]["token"]:
                continue
            flips += 1
            mg = min(ra[i]["gap"], rb[i]["gap"])
            worst = max(worst, mg)
            if first is None:
                first, tok_a, tok_b = i, ra[i]["token"], rb[i]["token"]
                gap_a, gap_b = ra[i]["gap"], rb[i]["gap"]
        if first is None:
            print(f"{pid:14s} {len(ra):4d} {len(rb):4d} {pre:4d} {dmax:9.5f} "
                  f"{(dsum / pre if pre else 0.0):9.5f} {dbig:6d} {'—':>6s} {'—':>6s} {'—':>6s} "
                  f"{'—':>8s} {'—':>8s} {'—':>8s} {0:6d} {'—':>8s}")
        else:
            if worst >= args.near_tie:
                all_near_tie = False
            diverged_prompts.append((pid, first, min(gap_a, gap_b), worst))
            print(f"{pid:14s} {len(ra):4d} {len(rb):4d} {pre:4d} {dmax:9.5f} "
                  f"{(dsum / pre if pre else 0.0):9.5f} {dbig:6d} {first:6d} "
                  f"{tok_a[:6]!r:>6s} {tok_b[:6]!r:>6s} {gap_a:8.3f} {gap_b:8.3f} "
                  f"{min(gap_a, gap_b):8.3f} {flips:6d} {worst:8.3f}")

    print("")
    print("# ---- summary ----")
    print(f"# prompts compared        : {prompts}")
    print(f"# prompts token-identical : {identical}/{prompts}")
    print(f"# tokens on common prefix : {gcnt} of {total_tok}")
    print(f"# max  |dlogprob| (nat)   : {gmax:.6f}")
    print(f"# mean |dlogprob| (nat)   : {(gsum / gcnt if gcnt else 0.0):.6f}")
    print(f"# positions |dlogprob|>0.01: {big}")
    if diverged_prompts:
        worst_all = max(d[3] for d in diverged_prompts)
        print(f"# diverged prompts        : {len(diverged_prompts)} "
              f"(pid, first_idx, min_gap_at_first, worst_gap) = {diverged_prompts}")
        print(f"# worst min-gap (nat)     : {worst_all:.3f}")
    if identical == prompts and prompts > 0:
        verdict = "IDENTICAL: every compared prompt produced the same token ids"
    elif all_near_tie:
        verdict = ("NEAR-TIE FLIPS: every divergence happened at a top-1/top-2 margin "
                   f"below {args.near_tie} nat -> numerical, not scheduling")
    else:
        verdict = ("NOT A NEAR-TIE: at least one divergence happened at a margin "
                   f">= {args.near_tie} nat -> mechanism must be explained")
    print(f"# VERDICT: {verdict}")


if __name__ == "__main__":
    main()
