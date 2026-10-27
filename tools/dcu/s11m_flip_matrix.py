#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11m: N-way comparison of the greedy token battery across engine restarts.

Reads the JSONL files written by ``tools/dcu/s11l_flip_stats.py`` (one row per
prompt, carrying ``token_ids`` plus per-position logprobs) and reports, for
every pair of labels, whether the greedy token ids are *exactly* the same --
which is the S11m reproducibility gate.

Usage:
    python tools/dcu/s11m_flip_matrix.py run1.jsonl run2.jsonl run3.jsonl ...
"""

import argparse
import json
import sys
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


def tokens(rec):
    ids = rec.get("token_ids")
    if ids:
        return [int(t) for t in ids]
    toks = [r["token"] for r in rec.get("rows") or []]
    return toks


def logprobs(rec):
    return [r["logprob"] for r in rec.get("rows") or []]


def compare(ref_lbl, ref, oth_lbl, oth):
    a, b = tokens(ref), tokens(oth)
    la, lb = logprobs(ref), logprobs(oth)
    n = min(len(a), len(b))
    diffs = [i for i in range(n) if a[i] != b[i]]
    matched = [abs(la[i] - lb[i]) for i in range(min(len(la), len(lb), n))
               if a[i] == b[i]]
    return {
        "len_a": len(a),
        "len_b": len(b),
        "n_compared": n,
        "n_diff": len(diffs),
        "first_diff": diffs[0] if diffs else None,
        "max_matched_delta": max(matched) if matched else 0.0,
        "diffs": [
            {
                "i": i,
                "a": a[i],
                "b": b[i],
                "lp_a": la[i] if i < len(la) else None,
                "lp_b": lb[i] if i < len(lb) else None,
            }
            for i in diffs[:8]
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--label-prefix", default=None,
                    help="only compare labels starting with this prefix")
    args = ap.parse_args()

    runs = load(args.jsonl)
    labels = [l for l in runs if not args.label_prefix
              or l.startswith(args.label_prefix)]
    print(f"# labels: {labels}")
    if len(labels) < 2:
        print("# need >=2 labels")
        return 2

    prompts = sorted(set().union(*(runs[l].keys() for l in labels)))
    out = {"labels": labels, "per_prompt": {}, "pairwise": {}}
    all_identical = True
    ref = labels[0]
    # every label against the first one
    for oth in labels[1:]:
        tot = dif = 0
        worst = 0.0
        for pid in prompts:
            if pid not in runs[ref] or pid not in runs[oth]:
                continue
            c = compare(ref, runs[ref][pid], oth, runs[oth][pid])
            tot += c["n_compared"]
            dif += c["n_diff"]
            worst = max(worst, c["max_matched_delta"])
            out["per_prompt"].setdefault(pid, {})[f"{ref}_vs_{oth}"] = c
            if c["n_diff"]:
                all_identical = False
                print(f"[DIFF] {pid}: {ref} vs {oth}: {c['n_diff']}/"
                      f"{c['n_compared']} positions, first={c['first_diff']}"
                      f" max_matched_delta={c['max_matched_delta']:.6f}")
                for d in c["diffs"]:
                    print(f"        i={d['i']} {ref}={d['a']}({d['lp_a']}) "
                          f"{oth}={d['b']}({d['lp_b']})")
        out["pairwise"][f"{ref}_vs_{oth}"] = {
            "positions": tot, "diff": dif, "max_matched_delta": worst
        }
        print(f"[PAIR] {ref} vs {oth}: {dif}/{tot} positions differ, "
              f"max_matched_delta={worst:.6f}")

    # identical-set check across all labels
    sets = {l: [tuple(tokens(runs[l][p])) for p in prompts if p in runs[l]]
            for l in labels}
    base = sets[ref]
    for l in labels[1:]:
        if sets[l] != base:
            all_identical = False
    print(f"\n# ALL_IDENTICAL={all_identical}")
    out["all_identical"] = all_identical
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"# json -> {args.json_out}")
    return 0 if all_identical else 1


if __name__ == "__main__":
    sys.exit(main())
