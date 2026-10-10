#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14d: N-arm pairwise determinism matrix.

For every ordered pair of ``s14d_client.py`` JSONL files, report:
  * prompts whose greedy token stream is bit-identical (out of n),
  * mean over prompts of the mean same-context |dlogprob| (nat),
  * worst over prompts of the max same-context |dlogprob| (nat),
  * first-generated-token argmax agreement (out of n),
  * median first divergence position of the generated stream.

Usage:
    s14d_matrix.py probe-W4.jsonl probe-S4.jsonl probe-A3.jsonl probe-A4.jsonl
"""

import argparse
import json
import statistics


def load(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["prompt_id"]] = r
    return out


def pair(ra, rb):
    ta, tb = ra["token_ids"], rb["token_ids"]
    ident = ta == tb
    first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if first is None and len(ta) != len(tb):
        first = min(len(ta), len(tb))
    pa, pb = ra["prompt_logprobs"], rb["prompt_logprobs"]
    pn = 0
    pmax = 0.0
    psum = 0.0
    for i in range(min(len(pa), len(pb))):
        ea, eb = pa[i], pb[i]
        if not ea or not eb:
            continue
        da, db = dict(ea), dict(eb)
        sh = set(da) & set(db)
        if not sh:
            continue
        pn += 1
        d = [abs(da[t] - db[t]) for t in sh]
        pmax = max(pmax, max(d))
        psum += sum(d) / len(d)
    la = [p for p in pa if p]
    lb = [p for p in pb if p]
    t1a = max(la[-1], key=lambda e: e[1])[0] if la else None
    t1b = max(lb[-1], key=lambda e: e[1])[0] if lb else None
    return dict(ident=ident, first=first, pn=pn,
                pmean=(psum / pn if pn else 0.0), pmax=pmax, tok1_eq=(t1a == t1b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--names", default=None, help="comma separated labels")
    args = ap.parse_args()
    names = args.names.split(",") if args.names else [
        f.split("probe-")[-1].replace(".jsonl", "") for f in args.files]
    arms = {n: load(f) for n, f in zip(names, args.files)}
    ids = [r["prompt_id"] for r in sorted(list(arms.values())[0].values(), key=lambda r: r["order"])]
    n = len(ids)

    hdr = f"{'pair':24s} {'ids_identical':>13s} {'tok1_agree':>10s} {'med_first_div':>13s} {'mean_plp_mean':>13s} {'worst_plp_max':>13s}"
    print(f"# arms: {', '.join(names)}   prompts: {n}")
    print(hdr)
    print("-" * len(hdr))
    for i, a in enumerate(names):
        for b in names:
            if a >= b:
                continue
            res = [pair(arms[a][p], arms[b][p]) for p in ids]
            ni = sum(r["ident"] for r in res)
            nt1 = sum(r["tok1_eq"] for r in res)
            firsts = [r["first"] for r in res if r["first"] is not None]
            print(f"{a + ' vs ' + b:24s} {f'{ni}/{n}':>13s} {f'{nt1}/{n}':>10s} "
                  f"{(statistics.median(firsts) if firsts else -1):13.1f} "
                  f"{statistics.mean(r['pmean'] for r in res):13.4f} "
                  f"{max(r['pmax'] for r in res):13.4f}")


if __name__ == "__main__":
    main()
