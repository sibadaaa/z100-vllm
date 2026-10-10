#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14d: bit-level comparison of two ``s14d_client.py`` JSONL files.

Two instruments:

  * token stream  -- greedy token ids, first divergence position, |dlogprob| on
    the common prefix (nat), the top-1/top-2 margin at the divergence,
  * same-context  -- ``prompt_logprobs`` top-k on the byte-identical PROMPT
    positions: max/mean |dlogprob|, argmax flips.  Independent of
    autoregressive divergence, so it also catches sub-token-level drift.

Usage:
    s14d_compare.py A.jsonl B.jsonl --label-a S4 --label-b W4
"""

import argparse
import json


def load(path):
    recs = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["prompt_id"]] = r
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--csv", default=None, help="optional per-prompt CSV output")
    args = ap.parse_args()

    A, B = load(args.file_a), load(args.file_b)
    order = [r["prompt_id"] for r in sorted(A.values(), key=lambda r: r["order"])]
    order += [p for p in B if p not in order]

    print(f"# {args.label_a} = {args.file_a}")
    print(f"# {args.label_b} = {args.file_b}")
    hdr = (f"{'prompt':18s} {'nA':>4s} {'nB':>4s} {'ids=':>5s} {'first':>6s} "
           f"{'maxd':>9s} {'gapA':>8s} {'gapB':>8s} | {'plp_n':>5s} {'plp_max':>8s} "
           f"{'plp_mean':>9s} {'flips':>6s}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    n_ident = n_div = 0
    worst = (0.0, None)
    for pid in order:
        ra, rb = A.get(pid), B.get(pid)
        if ra is None or rb is None:
            print(f"{pid:18s} MISSING (A={ra is not None} B={rb is not None})")
            continue
        ta, tb = ra["token_ids"], rb["token_ids"]
        ident = ta == tb
        n_ident += ident
        n_div += (not ident)
        first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if first is None and len(ta) != len(tb):
            first = min(len(ta), len(tb))
        # |dlogprob| on the common prefix of the generated stream
        maxd = 0.0
        common = min(len(ra["rows"]), len(rb["rows"])) if first is None else first
        for i in range(common):
            xa, xb = ra["rows"][i], rb["rows"][i]
            if xa["token_id"] == xb["token_id"]:
                maxd = max(maxd, abs(xa["logprob"] - xb["logprob"]))
            else:
                break
        gapA = ra["rows"][first]["gap"] if first is not None and first < len(ra["rows"]) else None
        gapB = rb["rows"][first]["gap"] if first is not None and first < len(rb["rows"]) else None

        # same-context prompt_logprobs
        pa, pb = ra.get("prompt_logprobs") or [], rb.get("prompt_logprobs") or []
        npos = min(len(pa), len(pb))
        plp_max = plp_mean = 0.0
        flips = 0
        plp_n = 0
        for i in range(npos):
            ea, eb = pa[i], pb[i]
            if not ea or not eb:
                continue
            da = dict(ea)
            db = dict(eb)
            shared = set(da) & set(db)
            if not shared:
                continue
            plp_n += 1
            d = [abs(da[t] - db[t]) for t in shared]
            plp_max = max(plp_max, max(d))
            plp_mean += sum(d) / len(d)
            if max(da, key=da.get) != max(db, key=db.get):
                flips += 1
        plp_mean = plp_mean / plp_n if plp_n else 0.0
        if first is not None and maxd > worst[0]:
            worst = (maxd, pid)
        print(f"{pid:18s} {len(ta):4d} {len(tb):4d} {str(ident):>5s} "
              f"{str(first):>6s} {maxd:9.6f} "
              f"{(f'{gapA:.3f}' if gapA is not None else '-'):>8s} "
              f"{(f'{gapB:.3f}' if gapB is not None else '-'):>8s} | "
              f"{plp_n:5d} {plp_max:8.5f} {plp_mean:9.5f} {flips:6d}")
        rows.append(dict(prompt=pid, n_a=len(ta), n_b=len(tb), ids_equal=ident,
                         first_div=first, max_dlogprob_common=maxd,
                         gap_a=gapA, gap_b=gapB, plp_positions=plp_n,
                         plp_max=plp_max, plp_mean=plp_mean, plp_argmax_flips=flips))

    total = n_ident + n_div
    print("-" * len(hdr))
    print(f"# prompts compared            : {total}")
    print(f"# prompts token-identical     : {n_ident}/{total}"
          f"   (inconsistency rate = {n_div}/{total} = "
          f"{(n_div / total if total else 0):.4f})")
    print(f"# worst |dlogprob| at first divergence: {worst[0]:.6f} nat"
          f"  ({worst[1]})")
    print(f"# total generated tokens      : "
          f"{sum(len(A[p]['token_ids']) for p in order if p in A)} vs "
          f"{sum(len(B[p]['token_ids']) for p in order if p in B)}")
    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"# csv -> {args.csv}")


if __name__ == "__main__":
    main()
