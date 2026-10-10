#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14d: deeper reading of the same-context difference.

``s14d_compare.py`` gives the scalar verdict; this script answers *what shape*
the difference has, which is what decides between "rounding" and "materially
different computation":

  * per prompt: how many prompt positions differ (|d| > 1e-6), the smallest
    position at which they differ, and the |d| PROFILE over positions,
  * the |d| of the *argmax* token (the only token that can change the answer)
    versus the |d| of the top-k tail,
  * first generated token argmax agreement (the "does the answer change" test).
"""

import argparse
import json


def load(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["prompt_id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    ids = [r["prompt_id"] for r in sorted(A.values(), key=lambda r: r["order"])]

    hdr = (f"{'prompt':18s} {'plp_n':>5s} {'ndiff':>5s} {'first':>5s} {'maxall':>8s} "
           f"{'maxargmax':>9s} {'maxtail':>8s} {'argmax_flips':>12s} "
           f"{'tok1_A':>8s} {'tok1_B':>8s} {'tok1_eq':>7s}")
    print(f"# A={args.label_a}  B={args.label_b}")
    print(hdr)
    print("-" * len(hdr))
    n_first_eq = 0
    n = 0
    for pid in ids:
        ra, rb = A[pid], B[pid]
        pa, pb = ra["prompt_logprobs"], rb["prompt_logprobs"]
        profile = []
        ndiff = first = 0
        maxall = maxarg = maxtail = 0.0
        flips = 0
        for i in range(min(len(pa), len(pb))):
            ea, eb = pa[i], pb[i]
            if not ea or not eb:
                continue
            da, db = dict(ea), dict(eb)
            sh = set(da) & set(db)
            if not sh:
                continue
            am_a = max(da, key=da.get)
            am_b = max(db, key=db.get)
            d_am = abs(da.get(am_a, 0) - db.get(am_a, 0)) if am_a == am_b else None
            d_tail = max((abs(da[t] - db[t]) for t in sh if t not in (am_a, am_b)),
                         default=0.0)
            d_all = max(abs(da[t] - db[t]) for t in sh)
            maxall = max(maxall, d_all)
            maxtail = max(maxtail, d_tail)
            if d_am is not None:
                maxarg = max(maxarg, d_am)
            if am_a != am_b:
                flips += 1
            if d_all > 1e-6:
                ndiff += 1
                if first == 0:
                    first = i
            profile.append(d_all if d_all > 1e-6 else 0.0)
        # first generated token = argmax at the last prompt position
        la = [p for p in pa if p]
        lb = [p for p in pb if p]
        t1a = max(la[-1], key=lambda e: e[1])[0] if la else None
        t1b = max(lb[-1], key=lambda e: e[1])[0] if lb else None
        eq = t1a == t1b
        n += 1
        n_first_eq += eq
        print(f"{pid:18s} {len(la):5d} {ndiff:5d} {first:5d} {maxall:8.4f} "
              f"{maxarg:9.4f} {maxtail:8.4f} {flips:12d} {str(t1a):>8s} {str(t1b):>8s} "
              f"{str(eq):>7s}")
        if args.profile:
            prof = " ".join(f"{i}:{v:.3f}" for i, v in enumerate(profile) if v > 1e-6)
            print(f"    profile(positions with |d|>1e-6): {prof}")
    print("-" * len(hdr))
    print(f"# first-generated-token argmax agreement: {n_first_eq}/{n}")
    print("# NOTE maxargmax = max |d| over positions where both arms' argmax is the "
          "same token (the 'does the dominant logit move' number);")
    print("#      maxtail   = max |d| over the non-argmax top-k entries (the "
          "'competition structure moves' number).")


if __name__ == "__main__":
    main()
