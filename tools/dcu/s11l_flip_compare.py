#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11l: compare two `s11l_flip_stats.py` JSONL arms token by token.

Reports, per prompt, every greedy flip with the top-1/top-2 margin on both
sides, the logprob the flip token had in the *other* arm's distribution, the
per-prompt maximum |logprob delta| at matched positions, the token match rate,
and a global summary (flip count, flip rate, worst margin, worst delta).

Usage:
    s11l_flip_compare.py --a a.jsonl --b b.jsonl --label-a sdot4 --label-b ctl0
"""

import argparse
import json


def load(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["prompt_id"]] = rec
    return out


def lp_of(rec: dict, i: int, token: str | None):
    """Logprob that arm `rec` assigned to `token` at position i (top-k only)."""
    rows = rec["rows"]
    if i >= len(rows) or token is None:
        return None
    for tok, lp in rows[i]["top"]:
        if tok == token:
            return lp
    return None


def compare(a: dict, b: dict, la: str, lb: str) -> dict:
    per_prompt = []
    total_pos = total_flip = 0
    worst_margin = None
    worst_margin_at = None
    worst_delta = 0.0
    worst_delta_at = None
    for pid in a:
        if pid not in b:
            continue
        ra, rb = a[pid], b[pid]
        ta, tb = ra["token_ids"] or [], rb["token_ids"] or []
        n = min(len(ta), len(tb))
        rows_a, rows_b = ra["rows"], rb["rows"]
        flips = []
        dmax = 0.0
        dmax_at = None
        for i in range(n):
            if ta[i] != tb[i]:
                tok_a, tok_b = rows_a[i]["token"], rows_b[i]["token"]
                # logprob each arm gave the *other* arm's token (None if it is
                # outside that arm's top-k).
                cross_a = lp_of(ra, i, tok_b)
                cross_b = lp_of(rb, i, tok_a)
                dec_a = cross_a - rows_a[i]["logprob"] if cross_a is not None else None
                dec_b = cross_b - rows_b[i]["logprob"] if cross_b is not None else None
                margin = min(
                    [m for m in (rows_a[i]["gap"], rows_b[i]["gap"]) if m is not None]
                )
                flips.append(
                    {
                        "i": i,
                        "tok_a": tok_a,
                        "tok_b": tok_b,
                        "gap_a": rows_a[i]["gap"],
                        "gap_b": rows_b[i]["gap"],
                        "min_gap": margin,
                        "dec_a": dec_a,
                        "dec_b": dec_b,
                        "lp_a_of_b": cross_a,
                        "lp_b_of_a": cross_b,
                    }
                )
                if worst_margin is None or margin > worst_margin:
                    worst_margin = margin
                    worst_margin_at = (pid, i)
            else:
                # same token: logprob delta of the same decision
                d = abs(rows_a[i]["logprob"] - rows_b[i]["logprob"])
                if d > dmax:
                    dmax, dmax_at = d, i
                if d > worst_delta:
                    worst_delta = d
                    worst_delta_at = (pid, i)
        total_pos += n
        total_flip += len(flips)
        per_prompt.append(
            {
                "prompt_id": pid,
                "prompt": ra["prompt"],
                "n_compared": n,
                "len_a": len(ta),
                "len_b": len(tb),
                "flips": flips,
                "first_flip": flips[0]["i"] if flips else None,
                "max_matched_delta": dmax,
                "max_matched_delta_at": dmax_at,
                "match_rate": (n - len(flips)) / n if n else 0.0,
            }
        )
    return {
        "label_a": la,
        "label_b": lb,
        "per_prompt": per_prompt,
        "summary": {
            "positions": total_pos,
            "flips": total_flip,
            "flip_rate": total_flip / total_pos if total_pos else 0.0,
            "prompts_with_flip": sum(1 for p in per_prompt if p["flips"]),
            "prompts": len(per_prompt),
            "max_matched_delta": worst_delta,
            "max_matched_delta_at": worst_delta_at,
            "worst_min_gap": worst_margin,
            "worst_min_gap_at": worst_margin_at,
            "all_flips_below_0p5_nat": worst_margin is not None and worst_margin < 0.5,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    res = compare(load(args.a), load(args.b), args.label_a, args.label_b)
    print(f"== {res['label_a']} vs {res['label_b']}")
    for p in res["per_prompt"]:
        print(
            f"  {p['prompt_id']:<8} n={p['n_compared']:>3} flips={len(p['flips']):>2} "
            f"first={p['first_flip']} match={p['match_rate']:.3f} "
            f"max_matched_delta={p['max_matched_delta']:.4f}@{p['max_matched_delta_at']}"
        )
        for f in p["flips"]:
            print(
                f"      i={f['i']:>3} '{f['tok_a']}' vs '{f['tok_b']}' "
                f"gap_a={f['gap_a']:.3f} gap_b={f['gap_b']:.3f} "
                f"min_gap={f['min_gap']:.3f} "
                f"dec_a={f['dec_a'] if f['dec_a'] is None else round(f['dec_a'], 3)} "
                f"dec_b={f['dec_b'] if f['dec_b'] is None else round(f['dec_b'], 3)}"
            )
    s = res["summary"]
    print(
        "SUMMARY "
        + json.dumps(
            {
                "positions": s["positions"],
                "flips": s["flips"],
                "flip_rate": round(s["flip_rate"], 5),
                "prompts_with_flip": s["prompts_with_flip"],
                "prompts": s["prompts"],
                "max_matched_delta": round(s["max_matched_delta"], 5),
                "max_matched_delta_at": s["max_matched_delta_at"],
                "worst_min_gap": None
                if s["worst_min_gap"] is None
                else round(s["worst_min_gap"], 4),
                "worst_min_gap_at": s["worst_min_gap_at"],
                "all_flips_below_0p5_nat": s["all_flips_below_0p5_nat"],
            },
            ensure_ascii=False,
        )
    )
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
