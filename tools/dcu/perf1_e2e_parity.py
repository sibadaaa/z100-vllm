#!/usr/bin/env python3
"""Compare two perf1_e2e_bench.py artifacts: greedy token parity + perf delta.

Usage:
    python3 tools/dcu/perf1_e2e_parity.py CONTROL.json ARM.json [--json out.json]

Exit status 0 = token streams identical (and logprobs within tolerance),
1 = divergence or missing parity data, 2 = bad input.

Parity rules (an arm that changes the token stream is a correctness regression
until proven otherwise):
  * prompt token ids must match exactly (same tokenizer / chat path),
  * generated token id streams must match exactly, element by element,
  * top-1 logprobs must match within --tol (default 1e-2 nats); larger drift is
    reported but scored separately from the discrete id stream.
Within-arm repeat determinism is checked too (all repeats of a cfg length must
produce the same stream), because a nondeterministic arm cannot be compared.
"""

import argparse
import json
import sys


def _key(run):
    return (run.get("prompt_cfg"), run.get("rep", 0))


def compare_streams(name, a, b, tol):
    """a, b: lists of parity runs for the same (prompt_cfg, rep)."""
    out = {"case": name, "verdict": "confirmed", "notes": []}
    if a is None or b is None:
        out["verdict"] = "could-not-verify"
        out["notes"].append("missing run on one side")
        return out
    pa, pb = a.get("prompt_token_ids") or [], b.get("prompt_token_ids") or []
    ta, tb = a.get("token_ids") or [], b.get("token_ids") or []
    la, lb = a.get("logprobs") or [], b.get("logprobs") or []
    out.update({
        "prompt_tokens_a": a.get("prompt_tokens"), "prompt_tokens_b": b.get("prompt_tokens"),
        "gen_tokens_a": len(ta), "gen_tokens_b": len(tb),
        "ttft_ms_a": a.get("ttft_ms"), "ttft_ms_b": b.get("ttft_ms"),
    })
    if pa != pb:
        out["prompt_ids_equal"] = False
        first = next((i for i, (x, y) in enumerate(zip(pa, pb)) if x != y), min(len(pa), len(pb)))
        out["prompt_first_divergence"] = first
        out["verdict"] = "refuted"
        out["notes"].append(f"prompt token ids differ at index {first}")
    else:
        out["prompt_ids_equal"] = True

    out["token_ids_equal"] = ta == tb
    n = min(len(ta), len(tb))
    first_div = next((i for i in range(n) if ta[i] != tb[i]), None)
    if first_div is None and len(ta) != len(tb):
        first_div = n
    out["common_prefix"] = first_div if first_div is not None else len(ta)
    if ta != tb:
        out["first_divergence"] = first_div
        out["divergence"] = {
            "index": first_div,
            "a": ta[first_div:first_div + 6],
            "b": tb[first_div:first_div + 6],
            "a_text": (a.get("token_texts") or [])[first_div:first_div + 6],
            "b_text": (b.get("token_texts") or [])[first_div:first_div + 6],
        }
        text_a = "".join(a.get("token_texts") or [])
        text_b = "".join(b.get("token_texts") or [])
        out["text_equal"] = text_a == text_b
        out["n_diff_positions"] = sum(1 for x, y in zip(ta, tb) if x != y) + abs(len(ta) - len(tb))
        if out["text_equal"]:
            out["verdict"] = "partially-confirmed"
            out["notes"].append(
                f"token ids differ ({out['n_diff_positions']} positions, first {first_div}) "
                f"but decoded text is byte-identical"
            )
        else:
            out["verdict"] = "refuted"
            out["notes"].append(
                f"token id streams differ: {len(ta)} vs {len(tb)} tokens, "
                f"common prefix {first_div}, {out['n_diff_positions']} differing positions, "
                f"decoded text differs"
            )

    pairs = [(x, y) for x, y in zip(la, lb) if x is not None and y is not None]
    if pairs:
        diffs = [abs(x - y) for x, y in pairs]
        out["logprob_max_abs_diff"] = max(diffs)
        out["logprob_mean_abs_diff"] = sum(diffs) / len(diffs)
        out["logprob_over_tol"] = sum(1 for d in diffs if d > tol)
        if out["logprob_over_tol"]:
            out["notes"].append(
                f"{out['logprob_over_tol']}/{len(diffs)} top-1 logprobs differ by >{tol} "
                f"(max {out['logprob_max_abs_diff']:.4g})"
            )
    else:
        out["notes"].append("no logprobs captured")
    if out["verdict"] == "confirmed" and not out["notes"]:
        out["notes"].append("token ids and prompt ids identical")
    return out


def compare_perf(a, b):
    rows = []
    for cfg in sorted({r.get("prompt_cfg") for r in a.get("runs", [])} |
                      {r.get("prompt_cfg") for r in b.get("runs", [])}):
        ra = [r for r in a.get("runs", []) if r.get("prompt_cfg") == cfg]
        rb = [r for r in b.get("runs", []) if r.get("prompt_cfg") == cfg]
        if not ra or not rb:
            continue
        row = {"prompt_cfg": cfg}

        def mean(rows_, key):
            vals = [r[key] for r in rows_ if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        for key in ("ttft_ms", "prefill_tps", "decode_tps_net", "step_ms_p50",
                    "tokens_per_step"):
            va, vb = mean(ra, key), mean(rb, key)
            row[f"{key}_a"] = round(va, 3) if va is not None else None
            row[f"{key}_b"] = round(vb, 3) if vb is not None else None
            if va and vb:
                row[f"{key}_delta_pct"] = round((vb - va) / va * 100, 2)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("arm")
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    with open(args.control) as f:
        A = json.load(f)
    with open(args.arm) as f:
        B = json.load(f)

    aruns = {(r.get("prompt_cfg"), r.get("rep", 0)): r for r in A.get("parity_runs", [])}
    bruns = {(r.get("prompt_cfg"), r.get("rep", 0)): r for r in B.get("parity_runs", [])}

    report = {
        "control": args.control, "arm": args.arm,
        "control_arm_name": A.get("config", {}).get("arm"),
        "arm_name": B.get("config", {}).get("arm"),
        "cases": [], "within_arm_determinism": [], "perf": compare_perf(A, B),
    }
    diverged = False
    partial = False
    missing = False

    if not aruns or not bruns:
        missing = True
        report["error"] = "no parity_runs in one or both artifacts"
    for key in sorted(set(aruns) | set(bruns)):
        case = compare_streams(f"p{key[0]}_rep{key[1]}", aruns.get(key), bruns.get(key), args.tol)
        report["cases"].append(case)
        if case["verdict"] == "refuted":
            diverged = True
        elif case["verdict"] == "partially-confirmed":
            partial = True
        elif case["verdict"] == "could-not-verify":
            missing = True

    # within-arm determinism: same cfg length must give the same stream over reps
    for label, data in (("control", A), ("arm", B)):
        by_len = {}
        for r in data.get("parity_runs", []):
            by_len.setdefault(r.get("prompt_cfg"), []).append(r)
        for cfg, runs in sorted(by_len.items()):
            if len(runs) < 2:
                continue
            base = runs[0].get("token_ids")
            same = all(r.get("token_ids") == base for r in runs[1:])
            report["within_arm_determinism"].append(
                {"arm": label, "prompt_cfg": cfg, "repeats": len(runs), "identical": same})
            if not same:
                diverged = True

    verdict = ("refuted" if diverged else
               "partially-confirmed" if partial else
               "could-not-verify" if missing else "confirmed")
    report["verdict"] = verdict
    report["summary"] = {
        "lexical_parity": "fail" if diverged else
                          ("text-only" if partial else
                           ("incomplete" if missing else "pass")),
        "cases": len(report["cases"]),
    }
    print(json.dumps(report, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
    if diverged:
        return 1
    return 2 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
