#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11l: turn `s11l_flip_compare.py --json-out` files into the archive tables.

Usage:
    s11l_flip_summary.py cmp-arm1-arm2.json cmp-arm3-arm2.json ...
"""

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--md", action="store_true", help="emit markdown tables")
    args = ap.parse_args()

    for path in args.files:
        with open(path, encoding="utf-8") as fh:
            res = json.load(fh)
        a, b = res["label_a"], res["label_b"]
        print(f"\n===== {a}  vs  {b}   ({path})")
        print(
            f"{'prompt':<10}{'n':>4}{'flips':>6}{'first':>6}"
            f"{'match':>8}{'maxΔsame':>10}"
        )
        for p in res["per_prompt"]:
            print(
                f"{p['prompt_id']:<10}{p['n_compared']:>4}{len(p['flips']):>6}"
                f"{str(p['first_flip']):>6}{p['match_rate']:>8.3f}"
                f"{p['max_matched_delta']:>10.4f}"
            )
        margins = []
        for p in res["per_prompt"]:
            for f in p["flips"]:
                cand = [abs(v) for v in (f["dec_a"], f["dec_b"]) if v is not None]
                m = max(cand) if cand else None
                margins.append((p["prompt_id"], f["i"], f["tok_a"], f["tok_b"],
                                f["gap_a"], f["gap_b"], f["dec_a"], f["dec_b"], m))
        print(f"  flips detail ({len(margins)}):")
        for pid, i, ta, tb, ga, gb, da, db, m in margins:
            fmt = lambda v: "None" if v is None else f"{v:.3f}"  # noqa: E731
            print(
                f"    {pid:<10} i={i:>3} '{ta}' vs '{tb}'  gap_a={ga:.3f} "
                f"gap_b={gb:.3f} cross_a={fmt(da)} cross_b={fmt(db)} "
                f"margin={'None(>top5)' if m is None else f'{m:.3f}'}"
            )
        known = [m for *_, m in margins if m is not None]
        s = res["summary"]
        print(
            "  SUMMARY "
            + json.dumps(
                {
                    "pair": f"{a} vs {b}",
                    "positions": s["positions"],
                    "flips": s["flips"],
                    "flip_rate": round(s["flip_rate"], 5),
                    "prompts_with_flip": f"{s['prompts_with_flip']}/{s['prompts']}",
                    "max_matched_delta": round(s["max_matched_delta"], 5),
                    "worst_min_gap(top1-top2)": round(s["worst_min_gap"], 4)
                    if s["worst_min_gap"] is not None
                    else None,
                    "max_flip_cross_margin": round(max(known), 4) if known else None,
                    "flips_outside_top5": len(margins) - len(known),
                    "all_flips_below_0p5_nat": all(m < 0.5 for m in known)
                    and len(known) == len(margins),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
