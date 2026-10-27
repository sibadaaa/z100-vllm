#!/usr/bin/env python3
"""Aggregate one S23 window's arm logs into the comparison table.

Never compares across windows: the table only holds arms that ran back-to-back
inside the same window directory, and every delta is computed against the
window's own control arm (plan §3 O6.2, §5 pitfall 4).

  .venv/bin/python tools/dcu/s23_report.py .s23/w1/O6/w1a [--control control]

Writes ``window-summary.json`` + ``window-summary.md`` into the window dir and
prints the markdown table.
"""

import argparse
import glob
import json
import os
import statistics


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def arm_record(arm_dir):
    rec = {"arm": os.path.basename(arm_dir), "probes": []}
    desc = load(os.path.join(arm_dir, "descriptor.json")) or {}
    rec["descriptor"] = desc
    rec["so_md5"] = desc.get("so_md5_promoted", "")
    rec["switches"] = desc.get("switch_effective", {})
    rec["git_rev_short"] = desc.get("git_rev_short", "")

    for path in sorted(glob.glob(os.path.join(arm_dir, "s20_*_*.json"))):
        payload = load(path) or {}
        summary = payload.get("summary") or {}
        runs = payload.get("runs") or []
        if not summary:
            continue
        rec["probes"].append({
            "file": os.path.basename(path),
            "step_p50_ms": summary.get("step_p50_ms"),
            "step_mean_ms": summary.get("srv_step_ms"),
            "out_tps": summary.get("out_tok_per_s"),
            "accept": summary.get("accept_len"),
            "tok_per_step": summary.get("srv_tok_per_step"),
            "ttft_ms": summary.get("ttft_ms"),
            "n_runs": len(runs),
            "out_tps_runs": [r.get("out_tok_per_s") for r in runs],
            "step_runs": [r.get("srv_step_ms") for r in runs],
        })

    sweeps = []
    for path in sorted(glob.glob(os.path.join(arm_dir, "sweep_*_*.json"))):
        payload = load(path) or {}
        for row in payload.get("results") or []:
            row["file"] = os.path.basename(path)
            sweeps.append(row)
    rec["sweeps"] = sweeps
    # Aggregate the repeated sweeps per concurrency: a single c=16 pass moves by
    # several percent on its own, so the claimable number is the median over reps.
    by_c = {}
    for row in sweeps:
        by_c.setdefault(row["concurrency"], []).append(row)
    rec["sweep_agg"] = {}
    for conc, rows in sorted(by_c.items()):
        # Rep 1 of a concurrency group runs with a cold prefix cache and lives in a
        # different regime (e.g. c=4 fwd 460-505 t/s cold vs ~900 warm), so the
        # claimable number is the median over the warm reps; both are reported.
        warm = rows[1:] if len(rows) > 1 else rows
        rec["sweep_agg"][conc] = {
            "reps": len(rows),
            "warm_reps": len(warm),
            "fwd_tps_median": statistics.median([r["fwd_tps"] for r in warm]),
            "fwd_tps_median_all": statistics.median([r["fwd_tps"] for r in rows]),
            "gen_tps_median": statistics.median([r["gen_tps"] for r in warm]),
            "step_ms_median": statistics.median([r["step_ms"] for r in warm]),
            "accept_median": statistics.median([r["accept"] for r in warm]),
            "ttft_ms_median": statistics.median([r["ttft_ms"] for r in warm]),
            "fwd_tps_all": [round(r["fwd_tps"], 1) for r in rows],
        }

    parity = []
    for path in sorted(glob.glob(os.path.join(arm_dir, "parity", "*check*.json"))):
        payload = load(path) or {}
        parity.append({"file": os.path.basename(path),
                       "verdict": payload.get("verdict")})
    rec["parity"] = parity

    smoke = load(os.path.join(arm_dir, "smoke.json")) or {}
    rec["smoke"] = {k: smoke.get(k) for k in
                    ("completion_text", "verdict", "matches_reference")}

    rec["tree_fp"] = str((desc.get("extra") or {}).get("profile", ""))
    fp_path = os.path.join(arm_dir, "tree-fingerprint.txt")
    rec["tree_fingerprint"] = ""
    if os.path.isfile(fp_path):
        with open(fp_path) as f:
            rec["tree_fingerprint"] = f.read().strip()
    rec["descriptor_path"] = os.path.join(arm_dir, "descriptor.json")

    p50 = [p["step_p50_ms"] for p in rec["probes"] if p["step_p50_ms"]]
    mean = [p["step_mean_ms"] for p in rec["probes"] if p["step_mean_ms"]]
    outs = [p["out_tps"] for p in rec["probes"] if p["out_tps"]]
    accs = [p["accept"] for p in rec["probes"] if p["accept"]]
    # The first request of every probe is cold (TTFT 3.8 s vs 2.1 s warm) and it
    # deflates out t/s, so keep a warm-only view next to the plain mean.
    warm_outs = [v for p in rec["probes"] for v in (p["out_tps_runs"] or [])[1:]
                 if v]
    warm_steps = [v for p in rec["probes"] for v in (p["step_runs"] or [])[1:] if v]
    ttfts = [p["ttft_ms"] for p in rec["probes"] if p["ttft_ms"]]
    tokps = [p["tok_per_step"] for p in rec["probes"] if p["tok_per_step"]]
    all_runs = [v for p in rec["probes"] for v in p["step_runs"] if v]
    rec["agg"] = {
        "probe_count": len(rec["probes"]),
        "p50_ms_median": statistics.median(p50) if p50 else None,
        "mean_ms_median": statistics.median(mean) if mean else None,
        "run_step_p50_ms": statistics.median(all_runs) if all_runs else None,
        "out_tps_mean": statistics.mean(outs) if outs else None,
        "out_tps_median": statistics.median(outs) if outs else None,
        "accept_mean": statistics.mean(accs) if accs else None,
        "ttft_ms_median": statistics.median(ttfts) if ttfts else None,
        "tok_per_step_median": statistics.median(tokps) if tokps else None,
        "out_tps_warm_median": statistics.median(warm_outs) if warm_outs else None,
        "step_warm_p50_ms": statistics.median(warm_steps) if warm_steps else None,
        "warm_runs": len(warm_steps),
    }
    return rec


def fmt(value, digits=2, suffix=""):
    if value is None:
        return "-"
    return f"{value:.{digits}f}{suffix}"


def delta(cur, ref):
    if not cur or not ref:
        return "-"
    return f"{(cur / ref - 1.0) * 100:+.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("window_dir")
    ap.add_argument("--control", default="control")
    args = ap.parse_args()

    arms = [arm_record(d) for d in sorted(glob.glob(os.path.join(args.window_dir, "*")))
            if os.path.isdir(d) and load(os.path.join(d, "descriptor.json"))]
    if not arms:
        print(f"[report] no arm descriptors under {args.window_dir}")
        return 1
    control = next((a for a in arms if a["arm"] == args.control), None)

    lines = [
        f"## Window {os.path.basename(args.window_dir)}",
        "",
        "| arm | git | so md5 | tree fp | switches (EMU/QNT/FUSE) | probes "
        "| step p50 | step warm p50 | step mean | out t/s | out t/s warm "
        "| accept | tok/step | TTFT ms | p50 vs ctrl | warm vs ctrl | parity |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        sw = a["switches"]
        is_ref = control is not None and a is control
        tag = "/".join(
            "1" if str(sw.get(k, "")).lower() in ("1", "true") else "0"
            for k in ("VLLM_DCU_FP8_EMU_FASTPATH", "VLLM_DCU_FP8_QUANT_FASTPATH",
                      "VLLM_DCU_FP8_FUSE_QUANT")
        )
        par = ",".join(p["verdict"] or "?" for p in a["parity"]) or "-"
        g = a["agg"]
        d50 = "ref" if is_ref else delta(
            g["p50_ms_median"],
            control["agg"]["p50_ms_median"] if control else None,
        )
        dout = "ref" if is_ref else delta(
            g["out_tps_mean"], control["agg"]["out_tps_mean"] if control else None
        )
        dwarm = "ref" if is_ref else delta(
            g["step_warm_p50_ms"],
            control["agg"]["step_warm_p50_ms"] if control else None,
        )
        lines.append(
            f"| {a['arm']} | {a['git_rev_short']} | {(a['so_md5'] or '')[:8]} "
            f"| {a['tree_fingerprint']} | {tag} | {g['probe_count']} "
            f"| {fmt(g['p50_ms_median'])} ms | {fmt(g['step_warm_p50_ms'])} ms "
            f"| {fmt(g['mean_ms_median'])} ms | {fmt(g['out_tps_mean'])} "
            f"| {fmt(g['out_tps_warm_median'])} | {fmt(g['accept_mean'])} "
            f"| {fmt(g['tok_per_step_median'])} | {fmt(g['ttft_ms_median'], 0)} "
            f"| {d50} | {dwarm} | {par} |"
        )

    lines += ["", "Per-probe detail (each probe is 4 back-to-back requests "
                  "of 512 prompt / 128 output tokens):", ""]
    for a in arms:
        lines.append(f"### {a['arm']}  ({a['descriptor'].get('desc', '')})")
        lines.append("")
        lines.append(f"raw: `{a['descriptor_path']}`  so_md5="
                     f"`{a['so_md5']}`  tree_fp=`{a['tree_fingerprint']}`  "
                     f"git=`{a['descriptor'].get('git_rev', '')}`")
        lines.append("")
        lines.append("| probe | runs | step p50 | step mean | out t/s | accept "
                     "| per-run step ms |")
        lines.append("|---|---|---|---|---|---|---|")
        for p in a["probes"]:
            runs = " ".join(fmt(v, 1) for v in (p["step_runs"] or []))
            lines.append(f"| {p['file']} | {p['n_runs']} | {fmt(p['step_p50_ms'])} "
                         f"| {fmt(p['step_mean_ms'])} | {fmt(p['out_tps'])} "
                         f"| {fmt(p['accept'])} | {runs} |")
        lines.append("")
        if a["sweep_agg"]:
            lines.append(
                "| concurrency | reps (warm) | fwd t/s warm med | fwd t/s all "
                "med | gen t/s warm med | step ms warm med | accept | TTFT ms "
                "| fwd t/s per rep (rep1 = cold) |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for conc, agg in a["sweep_agg"].items():
                per_rep = " ".join(str(v) for v in agg["fwd_tps_all"])
                lines.append(
                    f"| {conc} | {agg['reps']} ({agg['warm_reps']}) "
                    f"| {fmt(agg['fwd_tps_median'], 1)} "
                    f"| {fmt(agg['fwd_tps_median_all'], 1)} "
                    f"| {fmt(agg['gen_tps_median'], 1)} "
                    f"| {fmt(agg['step_ms_median'], 1)} "
                    f"| {fmt(agg['accept_median'])} "
                    f"| {fmt(agg['ttft_ms_median'], 0)} | {per_rep} |")
            lines.append("")
        smoke = a.get("smoke") or {}
        lines.append(f"smoke: `{smoke.get('completion_text')!r}` "
                     f"verdict={smoke.get('verdict')}")
        lines.append("")

    md = "\n".join(lines)
    out_json = os.path.join(args.window_dir, "window-summary.json")
    with open(out_json, "w") as f:
        json.dump({"window": os.path.basename(args.window_dir), "arms": arms,
                   "control": args.control}, f, indent=1)
    out_md = os.path.join(args.window_dir, "window-summary.md")
    with open(out_md, "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"[report] wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
