#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14g: N-way variant clustering of ``s11l_flip_stats.py`` JSONL files.

Answers "which engine starts landed on the same variant" without depending on a
reference arm:

  * every run gets a *variant signature* -- the sha256 of the concatenated
    greedy token-id stream of the whole battery.  Equal signature == bit-level
    identical answers == same variant;
  * pairwise table: identical prompts, first divergence, max/mean |dlogprob|
    over the positions that were still aligned;
  * an explicit variant -> runs mapping.

Usage:
    s14g_classify.py flip-A.jsonl flip-B.jsonl ...
"""

import argparse
import hashlib
import json
from collections import OrderedDict


def load(paths):
    runs = OrderedDict()
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                runs.setdefault(rec["label"], {})[rec["prompt_id"]] = rec
    return runs


def toks(rec):
    ids = rec.get("token_ids")
    if ids:
        return [int(t) for t in ids]
    return [r["token"] for r in rec.get("rows") or []]


def lps(rec):
    return [r["logprob"] for r in rec.get("rows") or []]


def signature(run, order):
    h = hashlib.sha256()
    for pid in order:
        h.update(",".join(str(t) for t in toks(run[pid])).encode())
        h.update(b"|")
    return h.hexdigest()[:12]


def pair(ra, rb, order):
    ident = 0
    firsts = []
    worst = 0.0
    sumd = 0.0
    n = 0
    for pid in order:
        a, b = toks(ra[pid]), toks(rb[pid])
        la, lb = lps(ra[pid]), lps(rb[pid])
        if a == b:
            ident += 1
        else:
            firsts.append(next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))))
        for i in range(min(len(a), len(b), len(la), len(lb))):
            if a[i] != b[i]:
                break
            d = abs(la[i] - lb[i])
            worst = max(worst, d)
            sumd += d
            n += 1
    return ident, firsts, worst, (sumd / n if n else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    runs = load(args.files)
    names = list(runs)
    order = list(next(iter(runs.values())))

    sigs = {n: signature(runs[n], order) for n in names}
    groups = OrderedDict()
    for n in names:
        groups.setdefault(sigs[n], []).append(n)

    print(f"# {len(names)} runs, {len(order)} prompts: {', '.join(order)}")
    print("# variant signature (sha256[:12] of the concatenated greedy token stream)")
    for n in names:
        print(f"  {n:10s} {sigs[n]}")
    print("# variants")
    for i, (s, ns) in enumerate(groups.items(), 1):
        print(f"  V{i} {s}: {', '.join(ns)}")
    print()
    hdr = f"{'pair':22s} {'ids_identical':>13s} {'max|dlp|':>9s} {'mean|dlp|':>10s} {'first_div':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ident, firsts, worst, mean = pair(runs[a], runs[b], order)
            print(f"{a + ' vs ' + b:22s} {f'{ident}/{len(order)}':>13s} {worst:9.4f} {mean:10.5f} "
                  f"{(min(firsts) if firsts else -1):10d}")


if __name__ == "__main__":
    main()
