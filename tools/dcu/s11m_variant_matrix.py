#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11m: full pairwise "token variant" matrix over every greedy battery file.

Reads all `flip-*.jsonl` files (tools/dcu/s11l_flip_stats.py output), compares
every pair by greedy token ids position by position and prints:

  * a matrix where `=` means "0 positions differ" (same variant),
  * the first divergence index per prompt for a few interesting pairs.

Usage:
    python tools/dcu/s11m_variant_matrix.py --glob '/mnt/game/s11*/flip-*.jsonl' \
        [--order arm1,arm3,...] [--json-out OUT]
"""

import argparse
import glob
import json
import os
import sys


def load(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["prompt_id"]] = [int(t) for t in rec["token_ids"]]
    return out


def pair_stats(a, b):
    tot = diff = 0
    for pid in a:
        if pid not in b:
            continue
        x, y = a[pid], b[pid]
        n = min(len(x), len(y))
        tot += n
        diff += sum(1 for i in range(n) if x[i] != y[i])
        diff += abs(len(x) - len(y))
    return tot, diff


def first_diffs(a, b):
    out = []
    for pid in a:
        if pid not in b:
            continue
        x, y = a[pid], b[pid]
        fd = None
        for i in range(min(len(x), len(y))):
            if x[i] != y[i]:
                fd = i
                break
        if fd is None and len(x) != len(y):
            fd = min(len(x), len(y))
        out.append((pid, fd))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/mnt/game/s11m/flip-*.jsonl")
    ap.add_argument("--extra-glob", default="/mnt/game/s11l/flip-*.jsonl")
    ap.add_argument("--order", default="")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    files = {}
    for pat in (args.glob, args.extra_glob):
        for p in glob.glob(pat):
            files[os.path.basename(p)[len("flip-"):-len(".jsonl")]] = p
    data = {k: load(v) for k, v in sorted(files.items())}

    labels = list(data)
    if args.order:
        want = [x for x in args.order.split(",") if x]
        labels = [x for x in want if x in data] + [x for x in labels if x not in want]

    print("labels: " + ", ".join(labels))
    print()
    print("%-12s" % "" + "".join("%-10s" % l[:9] for l in labels))
    for a in labels:
        row = "%-12s" % a[:11]
        for b in labels:
            if a == b:
                row += "%-10s" % "="
                continue
            tot, diff = pair_stats(data[a], data[b])
            row += "%-10s" % ("=" if diff == 0 else "%d/%d" % (diff, tot))
        print(row)

    if args.pairs:
        print()
        print("# first-diff per prompt")
        for spec in args.pairs.split(";"):
            if ":" not in spec:
                continue
            a, b = spec.split(":", 1)
            if a in data and b in data:
                print("%s vs %s: %s" % (a, b, first_diffs(data[a], data[b])))

    if args.json_out:
        out = {a: {b: pair_stats(data[a], data[b]) for b in labels} for a in labels}
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print("\n# json -> %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
