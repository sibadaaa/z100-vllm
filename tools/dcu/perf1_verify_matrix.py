#!/usr/bin/env python3
"""Reusable parity matrix + perf delta table over perf1 arm artifacts.

    python3 tools/dcu/perf1_verify_matrix.py \
        --out-dir /mnt/game/perf1 --title "task-4 baseline" \
        --pair k0ref:control --pair k0ref:k3 --pair control:k3 \
        k0ref=/mnt/game/perf1/k0ref_e2e.json \
        control=/mnt/game/perf1/control_e2e.json \
        k3=/mnt/game/perf1/k3_e2e.json

Writes <out-dir>/verify_<a>_vs_<b>.json per pair (the full comparator report) and
<out-dir>/VERIFY_MATRIX.md with one row per (pair, case) plus the perf deltas.
Reuses tools/dcu/perf1_e2e_parity.py so the verdict semantics are identical.
"""

import argparse
import importlib.util
import json
import os
import sys


def load_comparator(path):
    spec = importlib.util.spec_from_file_location("perf1_parity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", nargs="+", help="label=path")
    ap.add_argument("--pair", action="append", default=[], help="a:b (repeatable)")
    ap.add_argument("--out-dir", default="/mnt/game/perf1")
    ap.add_argument("--title", default="perf1 verification matrix")
    ap.add_argument("--tol", type=float, default=1e-2)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    comp = load_comparator(os.path.join(here, "perf1_e2e_parity.py"))

    data = {}
    for item in args.artifacts:
        label, _, path = item.partition("=")
        with open(path) as f:
            data[label] = json.load(f)

    pairs = [tuple(p.split(":")) for p in args.pair]
    os.makedirs(args.out_dir, exist_ok=True)
    rows, verdicts = [], []
    for a, b in pairs:
        if a not in data or b not in data:
            print(f"[skip] unknown pair {a}:{b}", file=sys.stderr)
            continue
        A, B = data[a], data[b]
        aruns = {(r.get("prompt_cfg"), r.get("rep", 0)): r for r in A.get("parity_runs", [])}
        bruns = {(r.get("prompt_cfg"), r.get("rep", 0)): r for r in B.get("parity_runs", [])}
        report = {"control": a, "arm": b, "cases": [], "pair": f"{a}:{b}"}
        worst = "confirmed"
        for key in sorted(set(aruns) | set(bruns)):
            case = comp.compare_streams(f"p{key[0]}_rep{key[1]}", aruns.get(key), bruns.get(key), args.tol)
            report["cases"].append(case)
            rows.append((f"{a} vs {b}", case))
            if case["verdict"] == "refuted":
                worst = "refuted"
            elif case["verdict"] == "partially-confirmed" and worst != "refuted":
                worst = "partially-confirmed"
            elif case["verdict"] == "could-not-verify" and worst == "confirmed":
                worst = "could-not-verify"
        report["verdict"] = worst
        report["perf"] = comp.compare_perf(A, B)
        verdicts.append(f"{a} vs {b}: {worst}")
        with open(os.path.join(args.out_dir, f"verify_{a}_vs_{b}.json"), "w") as f:
            json.dump(report, f, indent=2)

    lines = [f"# {args.title}", "", "## parity verdicts", ""]
    for v in verdicts:
        lines.append(f"* {v}")
    lines += ["", "| pair | case | verdict | gen A | gen B | common prefix | diff pos | "
              "text equal | logprob max | over tol |", "|---|---|---|---|---|---|---|---|---|---|"]
    for pair, c in rows:
        lines.append(
            f"| {pair} | {c['case']} | {c['verdict']} | {c.get('gen_tokens_a')} | "
            f"{c.get('gen_tokens_b')} | {c.get('common_prefix')} | {c.get('n_diff_positions')} | "
            f"{c.get('text_equal')} | "
            f"{c.get('logprob_max_abs_diff'):.4g} | {c.get('logprob_over_tol')} |"
            if c.get("logprob_max_abs_diff") is not None else
            f"| {pair} | {c['case']} | {c['verdict']} | - | - | - | - | - | - | - |")
    lines += ["", "## perf deltas (means per configured prompt)", "",
              "| pair | prompt cfg | TTFT A->B | prefill A->B | decode A->B | step p50 A->B | "
              "tok/step A->B |", "|---|---|---|---|---|---|---|"]
    for a, b in pairs:
        rep_path = os.path.join(args.out_dir, f"verify_{a}_vs_{b}.json")
        if not os.path.exists(rep_path):
            continue
        rep = json.load(open(rep_path))
        for p in rep.get("perf", []):
            lines.append(
                f"| {a} vs {b} | {p['prompt_cfg']} | {p.get('ttft_ms_a')} -> {p.get('ttft_ms_b')} "
                f"({p.get('ttft_ms_delta_pct')}%) | {p.get('prefill_tps_a')} -> {p.get('prefill_tps_b')} "
                f"({p.get('prefill_tps_delta_pct')}%) | {p.get('decode_tps_net_a')} -> "
                f"{p.get('decode_tps_net_b')} ({p.get('decode_tps_net_delta_pct')}%) | "
                f"{p.get('step_ms_p50_a')} -> {p.get('step_ms_p50_b')} "
                f"({p.get('step_ms_p50_delta_pct')}%) | {p.get('tokens_per_step_a')} -> "
                f"{p.get('tokens_per_step_b')} ({p.get('tokens_per_step_delta_pct')}%) |")
    out = os.path.join(args.out_dir, "VERIFY_MATRIX.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(verdicts))
    print(f"[saved] {out}")


if __name__ == "__main__":
    sys.exit(main())
