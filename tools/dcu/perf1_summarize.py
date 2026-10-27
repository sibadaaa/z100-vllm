#!/usr/bin/env python3
"""Render perf1 arm artifacts (*_e2e.json / *_anchor_*.json) as a markdown table.

    python3 tools/dcu/perf1_summarize.py --out /mnt/game/perf1/BASELINE.md \
        /mnt/game/perf1/control_e2e.json /mnt/game/perf1/k3_e2e.json ...

Reads the JSON schema written by perf1_e2e_bench.py.  Also picks up
arm_<name>_meta.txt (overrides + so md5 at start) and arm_<name>_so.txt
(so md5 before/after) when present next to the JSON.
"""

import argparse
import hashlib
import json
import os
import sys


def fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r, dict) and r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def ids_hash(ids):
    if not ids:
        return "-"
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()[:12]


def arm_block(path):
    with open(path) as f:
        d = json.load(f)
    cfg = d.get("config", {})
    arm = cfg.get("arm") or os.path.basename(path)
    meta = os.path.join(os.path.dirname(path), f"arm_{arm}_meta.txt")
    so = os.path.join(os.path.dirname(path), f"arm_{arm}_so.txt")
    out = {
        "file": path, "arm": arm, "so_md5": cfg.get("so_md5"), "config": cfg,
        "meta": open(meta).read().strip().splitlines() if os.path.exists(meta) else [],
        "so_check": open(so).read().strip().splitlines() if os.path.exists(so) else [],
        "json": d,
    }
    return out


def render(blocks, title):
    L = [f"# {title}", ""]
    L.append("| arm | so md5 | so stable | overrides | started |")
    L.append("|---|---|---|---|---|")
    for b in blocks:
        stable = "-"
        for line in b["so_check"]:
            if line.startswith("so_stable="):
                stable = line.split("=", 1)[1]
        ov = "; ".join(x for x in b["meta"] if x.startswith("export ")) or "(control)"
        L.append(f"| {b['arm']} | `{(b['so_md5'] or '-')[:12]}` | {stable} | {ov} | "
                 f"{b['config'].get('started', '-')} |")
    L.append("")

    # per-arm perf table
    for b in blocks:
        d = b["json"]
        cfg = b["config"]
        runs = d.get("runs", [])
        L.append(f"## {b['arm']} -- perf (chat completions, temp={cfg.get('temperature')}, "
                 f"ignore_eos={cfg.get('ignore_eos')}, conc=1, g={cfg.get('max_tokens')})")
        L.append("")
        L.append("| prompt (real tok) | repeat | TTFT ms | prefill t/s | decode t/s | "
                 "step p50 ms | step mean ms | tok/step | total ms |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in runs:
            L.append(f"| {r.get('prompt_tokens')} | {r.get('rep')} | {fmt(r.get('ttft_ms'))} | "
                     f"{fmt(r.get('prefill_tps'))} | {fmt(r.get('decode_tps_net'))} | "
                     f"{fmt(r.get('step_ms_p50'))} | {fmt(r.get('step_ms_mean'))} | "
                     f"{fmt(r.get('tokens_per_step'), 3)} | {fmt(r.get('total_ms'))} |")
        L.append("")
        s = d.get("summary_by_prompt", {})
        if s:
            L.append("means: " + " | ".join(
                f"p{k}: TTFT {fmt(v.get('ttft_ms'))}ms prefill {fmt(v.get('prefill_tps'))} "
                f"decode {fmt(v.get('decode_tps'))} p50 {fmt(v.get('step_ms_p50'))} "
                f"tok/step {fmt(v.get('tokens_per_step'), 3)}" for k, v in s.items()))
            L.append("")
        cs = d.get("cold_start")
        if cs:
            L.append(f"cold start (first request after fresh start): prompt {cs.get('prompt_tokens')} "
                     f"TTFT {fmt(cs.get('ttft_ms'))} ms prefill {fmt(cs.get('prefill_tps'))} t/s "
                     f"decode {fmt(cs.get('decode_tps_net'))} t/s")
            warm = [r for r in runs if r.get("prompt_cfg") == cfg.get("cold_length")]
            if warm:
                L.append(f"steady state same length: TTFT {fmt(mean(warm, 'ttft_ms'))} ms "
                         f"prefill {fmt(mean(warm, 'prefill_tps'))} t/s "
                         f"(cold/warm prefill ratio "
                         f"{fmt((mean(warm, 'prefill_tps') or 0) / (cs.get('prefill_tps') or 1), 2)}x)")
            L.append("")
        if d.get("spec"):
            L.append(f"server spec counters delta: `{json.dumps(d['spec'])}`")
            L.append("")
        pr = d.get("parity_runs", [])
        if pr:
            L.append("| parity case | prompt tok | gen tok | prompt-id sha | token-id sha | "
                     "TTFT ms |")
            L.append("|---|---|---|---|---|---|")
            for r in pr:
                L.append(f"| p{r.get('prompt_cfg')}_rep{r.get('rep')} | {r.get('prompt_tokens')} | "
                         f"{len(r.get('token_ids') or [])} | {ids_hash(r.get('prompt_token_ids'))} | "
                         f"{ids_hash(r.get('token_ids'))} | {fmt(r.get('ttft_ms'))} |")
            L.append("")
        csum = d.get("summary")
        if csum:
            L.append(f"summary: `{json.dumps(csum)}`")
            L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="perf1 e2e arm summary")
    args = ap.parse_args()
    blocks = []
    for p in args.json:
        if not os.path.exists(p):
            print(f"[skip] {p} not found", file=sys.stderr)
            continue
        blocks.append(arm_block(p))
    text = render(blocks, args.title)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"[saved] {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main())
