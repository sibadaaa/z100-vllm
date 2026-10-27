#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14b: same-context "prefix profile" between two arms.

Reads two ``prefix-<ARM>.jsonl`` files produced by ``s14b_greedy.py
--prompts-file prefixes.txt --max-tokens 1``.  Every record is a *different*
context (a prefix of one base text) but the SAME context for both arms, so the
top-k logprobs of the first generated token are directly comparable.

The profile answers the localization question that a single first-token sample
cannot:

  * |delta| already large on the shortest prefix and flat  -> the difference is
    introduced in the prompt-independent part (embedding / first layers /
    projection kernels),
  * |delta| growing with prefix length                     -> accumulation
    through the recurrent state (mechanism B: rounding amplified by the model),
  * |delta| small everywhere except a jump at some length  -> a length-dependent
    path switch (a real bug).

Usage:
    s14b_compare_prefix.py prefix-A4.jsonl prefix-A2b.jsonl --label-a A4 --label-b A2b
"""

import argparse
import json


def load(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    args = ap.parse_args()
    A = {r["prompt_id"]: r for r in load(args.a)}
    B = {r["prompt_id"]: r for r in load(args.b)}

    print(f"# {args.label_a}={args.a}")
    print(f"# {args.label_b}={args.b}")
    print(f"{'prefix':10s} {'ntokA':>5s} {'ntokB':>5s} {'sameids':>7s} "
          f"{'maxd':>8s} {'at_tok':>12s} {'tokA':>10s} {'tokB':>10s} {'gapA':>7s} "
          f"{'gapB':>7s} {'lpA':>8s} {'lpB':>8s}")
    rows = []
    for pid, ra in A.items():
        if pid not in B:
            continue
        rb = B[pid]
        ta = {t[0]: t[1] for t in ra["rows"][0]["top"]} if ra["rows"] else {}
        tb = {t[0]: t[1] for t in rb["rows"][0]["top"]} if rb["rows"] else {}
        sh = set(ta) & set(tb)
        ds = {t: abs(ta[t] - tb[t]) for t in sh}
        m = max(ds.values()) if ds else float("nan")
        tok = max(ds, key=ds.get) if ds else ""
        same = ra.get("prompt_token_ids") == rb.get("prompt_token_ids")
        ga = ra["rows"][0]["gap"] if ra["rows"] else float("nan")
        gb = rb["rows"][0]["gap"] if rb["rows"] else float("nan")
        la = ra["rows"][0]["logprob"] if ra["rows"] else float("nan")
        lb = rb["rows"][0]["logprob"] if rb["rows"] else float("nan")
        ca = ra["rows"][0]["token"] if ra["rows"] else ""
        cb = rb["rows"][0]["token"] if rb["rows"] else ""
        rows.append((pid, m, ca, cb, same))
        print(f"{pid:10s} {len(ra.get('prompt_token_ids') or []):5d} "
              f"{len(rb.get('prompt_token_ids') or []):5d} {str(same):>7s} {m:8.4f} "
              f"{tok[:12]!r:>12s} {ca[:10]!r:>10s} {cb[:10]!r:>10s} {ga:7.3f} {gb:7.3f} "
              f"{la:8.4f} {lb:8.4f}")

    print("")
    print("# ---- summary ----")
    if rows:
        mx = max(r[1] for r in rows)
        flips = [r[0] for r in rows if r[2] != r[3]]
        mismatch = [r[0] for r in rows if not r[4]]
        print(f"# prefixes compared        : {len(rows)}")
        print(f"# max |dlogprob| (nat)     : {mx:.6f}")
        print(f"# argmax differs at        : {flips if flips else 'none'}")
        print(f"# tokeniser mismatch at    : {mismatch if mismatch else 'none'}")
        first, last = rows[0], rows[-1]
        print(f"# trend: {first[0]}={first[1]:.4f} -> {last[0]}={last[1]:.4f} nat")


if __name__ == "__main__":
    main()
