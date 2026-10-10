#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14b: compare two arms' ``prompt_logprobs`` on IDENTICAL contexts.

Input: two JSONL files produced by ``s14b_prompt_logprobs.py`` (one record per
prompt, ``positions`` = per-prompt-position top-k logprobs).

For every prompt whose ``prompt_token_ids`` match between the two arms, the
tool reports the per-position |delta logprob| over the union of the two top-k
sets.  Because the context is identical, this number is the *pure arithmetic
distance* between the two compilation modes; it is not polluted by the
autoregressive divergence that makes generated-token comparisons useless after
the first flip.

Verdict thresholds (nat, log-softmax margins; softmax normalisation cancels, so
these are logit distances):

  * ``< 0.1``   -> ROUNDING-LEVEL: the two modes compute the same function up to
                   fp16 accumulation order (mechanism B),
  * ``>= 0.5``  -> MATERIAL: at least one mode computes a different function
                   (mechanism A) -- the same-context logit gap of order a nat
                   cannot come from rounding.

Usage:
    s14b_compare_promptlp.py plp-A1.jsonl plp-A2.jsonl --label-a A1 --label-b A2
"""

import argparse
import json


def load(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["prompt_id"]] = rec
    return out


def tops(entry):
    """id -> logprob (align by token id; vLLM may leave the decoded string unset)."""
    return {t[1]: t[2] for t in entry.get("top", [])}


def names(entry):
    return {t[1]: t[0] for t in entry.get("top", [])}


def disp(m, k):
    return str(m.get(k) if m.get(k) is not None else k)[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--profile", action="store_true",
                    help="also print the per-position max|dlogprob| profile (where the "
                         "two arms start to differ: accumulation vs a single bad position)")
    ap.add_argument("--profile-step", type=int, default=8)
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)

    print(f"# {args.label_a}={args.a}")
    print(f"# {args.label_b}={args.b}")
    print(f"{'prompt':18s} {'npos':>5s} {'sameids':>7s} {'maxd':>9s} {'meand':>9s} "
          f"{'@maxd_tok':>12s} {'flips':>5s} {'first':>5s} {'gapA':>7s} {'gapB':>7s} "
          f"{'lasttokA':>12s} {'lasttokB':>12s} {'lastd':>8s}")

    gmax, gsum, gcnt, gflips = 0.0, 0.0, 0, 0
    for pid in a:
        if pid not in b:
            continue
        ra, rb = a[pid], b[pid]
        same_ids = ra.get("prompt_token_ids") == rb.get("prompt_token_ids")
        pa, pb = ra["positions"], rb["positions"]
        n = min(len(pa), len(pb))
        dmax, dsum, dcnt, dmax_tok = 0.0, 0.0, 0, ""
        flips, first, gap_a, gap_b = 0, None, float("nan"), float("nan")
        for i in range(n):
            ta, tb = tops(pa[i]), tops(pb[i])
            if not ta or not tb:
                continue
            for tok in set(ta) | set(tb):
                if tok not in ta or tok not in tb:
                    continue
                d = abs(ta[tok] - tb[tok])
                dsum += d
                dcnt += 1
                if d > dmax:
                    dmax, dmax_tok = d, (names(pa[i]).get(tok) or names(pb[i]).get(tok) or str(tok))
            if pa[i].get("argmax_id") != pb[i].get("argmax_id"):
                flips += 1
                if first is None:
                    first = i
                    gap_a, gap_b = pa[i]["gap"], pb[i]["gap"]
        if dcnt:
            gmax = max(gmax, dmax)
            gsum += dsum
            gcnt += dcnt
        gflips += flips
        # last prompt position == the context that decides the FIRST generated token
        la, lb = pa[-1], pb[-1]
        ta, tb = tops(la), tops(lb)
        na, nb = names(la), names(lb)
        lastd = max((abs(ta[t] - tb[t]) for t in set(ta) & set(tb)), default=float("nan"))
        print(f"{pid:18s} {n:5d} {str(same_ids):>7s} {dmax:9.5f} "
              f"{(dsum / dcnt if dcnt else float('nan')):9.5f} {dmax_tok[:12]:>12s} "
              f"{flips:5d} {(first if first is not None else -1):5d} {gap_a:7.3f} {gap_b:7.3f} "
              f"{disp(na, la.get('argmax_id')):>12s} {disp(nb, lb.get('argmax_id')):>12s} {lastd:8.5f}")

        if args.profile:
            prof = []
            for i in range(0, n, args.profile_step):
                ta, tb = tops(pa[i]), tops(pb[i])  # noqa: F841
                sh = set(ta) & set(tb)
                prof.append(f"{i}:{max((abs(ta[t] - tb[t]) for t in sh), default=float('nan')):.3f}")
            ta, tb = tops(pa[n - 1]), tops(pb[n - 1])
            sh = set(ta) & set(tb)
            prof.append(f"{n - 1}:{max((abs(ta[t] - tb[t]) for t in sh), default=float('nan')):.3f}")
            print(f"{'':18s} profile(i:max|d|) " + " ".join(prof))

    print("")
    print("# ---- summary ----")
    print(f"# max  |dlogprob| on identical contexts (nat) : {gmax:.6f}")
    print(f"# mean |dlogprob| on identical contexts (nat) : {(gsum / gcnt if gcnt else 0.0):.6f}")
    print(f"# argmax flips over prompt positions           : {gflips}")
    if gmax < 0.1:
        v = "ROUNDING-LEVEL (mechanism B): same function up to fp16 accumulation order"
    elif gmax < 0.5:
        v = "SMALL (grey zone): above pure rounding but below the 0.5 nat project threshold"
    else:
        v = "MATERIAL (mechanism A): a >=0.5 nat same-context logit gap is not rounding"
    print(f"# VERDICT: {v}")


if __name__ == "__main__":
    main()
