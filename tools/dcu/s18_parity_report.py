#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# S18: offline parity report for two `s11l_flip_stats.py` JSONL arms.
#
#   python3 tools/dcu/s18_parity_report.py --a flip-ctl.jsonl --b flip-k1.jsonl \
#       [--label-a ctl --label-b k1] [--json /mnt/game/s18/parity-ctl-k1.json]
#
# Reports, per prompt and in aggregate:
#   * greedy flip count/rate (token stream parity)
#   * the FIRST divergence with both sides' top-1 margin at that index
#   * a positional-shift check (Jaccard of top-5 at neighbouring indices) so a
#     stream offset can be told apart from a distribution change
#   * matched-position |logprob| deltas (the protocol's own noise proxy)
#   * a verdict hint: FLIP-AT-CONFIDENT / near-tie / identical
try:
    import torch  # noqa: F401  (hard constraint 8 in DTK images)
except Exception:
    pass

import argparse
import json
import os


def load(path: str) -> dict:
    return {r["prompt_id"]: r for r in (json.loads(l) for l in open(path))}


def top_set(row, n=5):
    return {t for t, _ in row.get("top", [])[:n]}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--confident-nat", type=float, default=1.0,
                    help="|gap| above which a first-divergence flip is 'confident'")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    common = [p for p in A if p in B]
    tot = flips = 0
    rows = []
    matched_deltas = []
    for pid in common:
        ia, ib = A[pid]["token_ids"], B[pid]["token_ids"]
        tot += min(len(ia), len(ib))
        flips += sum(1 for x, y in zip(ia, ib) if x != y)
        for x, y, ra, rb in zip(ia, ib, A[pid]["rows"], B[pid]["rows"]):
            if x == y:
                matched_deltas.append(abs(ra["logprob"] - rb["logprob"]))
        div = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), None)
        if div is None:
            rows.append({"prompt": pid, "div": None})
            continue
        ra, rb = A[pid]["rows"][div], B[pid]["rows"][div]
        # positional-shift check around div
        jac = {}
        for j in range(max(0, div - 3), min(len(A[pid]["rows"]), div + 4)):
            jac[j] = round(jaccard(top_set(A[pid]["rows"][j]), top_set(rb)), 3)
        best_j = max(jac, key=jac.get) if jac else None
        rows.append({
            "prompt": pid, "div": div,
            "a_token": ra["token"], "a_gap": round(ra["gap"], 3),
            "b_token": rb["token"], "b_gap": round(rb["gap"], 3),
            "a_in_b_top5": ra["token"] in top_set(rb),
            "b_in_a_top5": rb["token"] in top_set(ra),
            "same_top5_set": round(jaccard(top_set(ra), top_set(rb)), 3),
            "shift_jaccard": jac, "shift_peak_at": best_j,
        })

    confident = [r for r in rows if r.get("div") is not None
                 and r["a_gap"] >= args.confident_nat]
    samepos = [r for r in rows if r.get("div") is not None
               and r["shift_peak_at"] == r["div"]]
    report = {
        "a": args.label_a, "b": args.label_b,
        "positions": tot, "flips": flips,
        "flip_rate": (flips / tot) if tot else None,
        "prompts": len(common),
        "prompts_identical": sum(1 for r in rows if r.get("div") is None),
        "max_matched_logprob_delta": max(matched_deltas) if matched_deltas else None,
        "mean_matched_logprob_delta": (
            sum(matched_deltas) / len(matched_deltas) if matched_deltas else None),
        "first_divergence": rows,
        "confident_first_flips": len(confident),
        "shift_peak_at_same_index": len(samepos),
    }
    print(f"=== parity {args.label_a} vs {args.label_b}")
    print(f"flips={flips}/{tot} ({report['flip_rate']:.3f})  "
          f"identical prompts={report['prompts_identical']}/{len(common)}")
    print(f"matched-position |dlogprob|: mean={report['mean_matched_logprob_delta']:.4f} "
          f"max={report['max_matched_logprob_delta']:.4f} nat")
    print(f"confident first flips (|gap|>={args.confident_nat} nat): {len(confident)}"
          f"  |  shift-peak-at-same-index: {len(samepos)}/{len(common)}")
    for r in rows:
        if r.get("div") is None:
            print(f"  {r['prompt']:<10} identical")
        else:
            print(f"  {r['prompt']:<10} div@{r['div']:<3} "
                  f"{args.label_a}:{r['a_token']!r}(gap {r['a_gap']}) | "
                  f"{args.label_b}:{r['b_token']!r}(gap {r['b_gap']}) | "
                  f"top5-jac={r['same_top5_set']} shift-peak@{r['shift_peak_at']}")
    if os.environ.get("S18_PARITY_VERDICT", "1") == "1":
        if flips == 0:
            verdict = "IDENTICAL"
        elif confident:
            verdict = "FLIP-AT-CONFIDENT -> material difference, not numeric noise"
        else:
            verdict = "near-tie flips only -> numeric-level difference"
        print(f"VERDICT: {verdict}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(f"[parity] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
