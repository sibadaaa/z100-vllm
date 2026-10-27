#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S14b: how far is a vLLM arm from the CPU/HF truth at the FIRST generated token?

``plp-HF*.jsonl`` (from ``s14b_hf_ref.py``) carries ``next_token_top``: the
top-k logprobs of the distribution that decides the first generated token.
``greedy-<ARM>.jsonl`` carries the same distribution for a vLLM arm in
``rows[0]['top']`` (context: the same rendered prompt — the two are compared
only when their ``prompt_token_ids`` agree, see ``--ids-check``).

Reported per prompt:
  * truth argmax and its margin,
  * arm argmax and its margin,
  * whether they agree,
  * the truth argmax's logprob *in the arm* and |delta| against the truth value
    (``nan`` when the truth token is not in the arm's top-k),
  * the arm's max |delta| over the shared top-k.

Usage:
    s14b_truth_compare.py greedy-B1.jsonl plp-HF4B-bf16.jsonl --label-arm B1-eager --label-truth HF4B-bf16
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")       # greedy-<ARM>.jsonl
    ap.add_argument("truth")     # plp-HF*.jsonl
    ap.add_argument("--label-arm", default="arm")
    ap.add_argument("--label-truth", default="truth")
    args = ap.parse_args()
    A = load(args.arm)
    T = load(args.truth)

    print(f"# arm   = {args.label_arm}  ({args.arm})")
    print(f"# truth = {args.label_truth}  ({args.truth})")
    print(f"{'prompt':18s} {'truth_top1':>12s} {'t_gap':>7s} {'arm_top1':>12s} {'a_gap':>7s} "
          f"{'match':>5s} {'truth_lp_in_arm':>15s} {'d_truth':>8s} {'maxd_shared':>11s}")
    agree = 0
    n = 0
    ds = []
    for pid, ta in T.items():
        if pid not in A or not A[pid]["rows"]:
            continue
        truth_top = ta.get("next_token_top") or []
        arm_top = A[pid]["rows"][0]["top"]
        if not truth_top or not arm_top:
            continue
        n += 1
        # greedy-<ARM>.jsonl stores [token_str, logprob]; plp-HF stores [token_str, id, logprob]
        tm = {t[0]: t[-1] for t in truth_top}
        am = {t[0]: t[-1] for t in arm_top}
        tid = truth_top[0][0]
        aid = arm_top[0][0]
        ok = tid == aid
        agree += int(ok)
        lp_in_arm = am.get(tid, float("nan"))
        d_truth = abs(lp_in_arm - tm[tid]) if tid in am else float("nan")
        if tid in am:
            ds.append(d_truth)
        sh = set(tm) & set(am)
        maxd = max((abs(tm[t] - am[t]) for t in sh), default=float("nan"))
        print(f"{pid:18s} {truth_top[0][0][:12]!r:>12s} {truth_top[0][-1] - truth_top[1][-1]:7.3f} "
              f"{arm_top[0][0][:12]!r:>12s} {arm_top[0][-1] - arm_top[1][-1]:7.3f} "
              f"{str(ok):>5s} {lp_in_arm:15.4f} {d_truth:8.4f} {maxd:11.4f}")
    print("")
    print(f"# prompts compared       : {n}")
    print(f"# first-token argmax match: {agree}/{n}")
    if ds:
        print(f"# mean |truth_lp - arm_lp| on the truth token : {sum(ds) / len(ds):.4f} nat")
        print(f"# max  |truth_lp - arm_lp| on the truth token : {max(ds):.4f} nat")


if __name__ == "__main__":
    main()
