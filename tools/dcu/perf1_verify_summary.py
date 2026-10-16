#!/usr/bin/env python3
"""Flat summary table of every perf1 arm ever measured, for VERIFICATION.md.

    python3 tools/dcu/perf1_verify_summary.py --dir /mnt/game/perf1 \
        --out /mnt/game/perf1/ARM_SUMMARY.md [--append-to /mnt/game/perf1/VERIFICATION.md]

Per arm it reports: the config actually in effect (read from the live worker
environment snapshot, not from the arm name), the anchor-protocol numbers
(p451/g64) when available, the anchor of the full sweep otherwise, the verdict
and - for void arms - the FATAL reason from the arm log.  Void arms are kept in
the table on purpose: they document what was tested and why it did not count.
"""

import argparse
import glob
import json
import os
import re

DISPATCH_KEYS = [
    "VLLM_DCU_FP8_GEMV_MAX_M",
    "VLLM_DCU_FP8_GEMV_STREAM",
    "VLLM_DCU_USE_BF16_GEMV",
    "VLLM_QWEN35_MTP_GDN_TRITON",
    "S11F_MTP_TOKENS",
    "S11F_MBT",
    "S11F_BF16_GEMV",
]


def env_of(d, arm):
    path = os.path.join(d, f"arm_{arm}_environ.txt")
    out = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.rstrip("\n")
            for k in DISPATCH_KEYS:
                if line.startswith(k + "="):
                    out[k] = line.split("=", 1)[1]
    return out


def void_reason(d, arm):
    """Void only if the LAST outcome in the append-only arm log was a failure.

    arm_<arm>.log is appended across sessions, so an arm that failed once and
    succeeded later (e.g. gemv_route) still contains the old FATAL line.
    """
    path = os.path.join(d, f"arm_{arm}.log")
    if not os.path.exists(path):
        return None
    fail_at = ok_at = -1
    fail_msg = None
    for i, line in enumerate(open(path, errors="ignore")):
        if ("FATAL override did not survive" in line or "FATAL derived knob" in line
                or "SMOKE FAILED" in line or "failed to start" in line):
            fail_at, fail_msg = i, line.strip()
        elif f"arm={arm} done" in line and "rc_main=0" in line:
            ok_at = i
    return fail_msg if fail_at > ok_at else None


def mean_of(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def anchor_numbers(d, arm):
    """(source, dict of means) for the anchor protocol when available."""
    a = load(os.path.join(d, f"{arm}_anchor_p512_g64.json"))
    if a and a.get("runs"):
        rows = a["runs"]
        return ("anchor g64 x%d" % len(rows), {k: mean_of(rows, k) for k in
                ("step_ms_p50", "decode_tps_net", "ttft_ms", "prefill_tps",
                 "tokens_per_step")})
    e = load(os.path.join(d, f"{arm}_e2e.json"))
    if e:
        runs = e.get("runs") or []
        sel = [r for r in runs if r.get("prompt_cfg") == 512] or runs
        if sel:
            return ("sweep p512 g%d x%d" % (e["config"].get("max_tokens", "?"), len(sel)),
                    {k: mean_of(sel, k) for k in
                     ("step_ms_p50", "decode_tps_net", "ttft_ms", "prefill_tps",
                      "tokens_per_step")})
        if e.get("parity_runs"):
            return ("parity only", None)
    return (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/mnt/game/perf1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--append-to", default=None)
    args = ap.parse_args()
    d = args.dir

    arms = sorted({os.path.basename(p)[: -len("_e2e.json")]
                   for p in glob.glob(os.path.join(d, "*_e2e.json"))} |
                  {os.path.basename(p)[: -len("_anchor_p512_g64.json")]
                   for p in glob.glob(os.path.join(d, "*_anchor_p512_g64.json"))})

    lines = ["## Appendix: every arm measured (authoritative flat table)", "",
             "Anchor protocol = chat completions, p451 real tokens, g64, temp 0, "
             "ignore_eos, concurrency 1, 3 repeats unless noted. `dispatch` is "
             "read from the live worker environment snapshot, not from the arm "
             "name. Void arms are kept so the record shows what was tested.",
             "",
             "| arm | dispatch (effective) | source | step p50 ms | decode t/s | "
             "TTFT ms | prefill t/s | tok/step | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        env = env_of(d, arm)
        disp = ", ".join(f"{k.replace('VLLM_DCU_', '').replace('VLLM_QWEN35_', '')}={env[k]}"
                         for k in DISPATCH_KEYS if k in env) or "-"
        reason = void_reason(d, arm)
        src, r = anchor_numbers(d, arm)
        if reason:
            verdict = "VOID: " + reason.split("]", 1)[-1].strip()
        elif r is None:
            verdict = "parity only (no perf row)"
        else:
            verdict = "measured"
        if r is None:
            lines.append(f"| {arm} | {disp} | {src or '-'} | - | - | - | - | - | {verdict} |")
        else:
            lines.append(
                f"| {arm} | {disp} | {src} | {r.get('step_ms_p50')} | "
                f"{r.get('decode_tps_net')} | {r.get('ttft_ms')} | "
                f"{r.get('prefill_tps')} | {r.get('tokens_per_step')} | {verdict} |")
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"[saved] {args.out}")
    if args.append_to:
        with open(args.append_to, "a") as f:
            f.write("\n" + text)
        print(f"[appended] {args.append_to}")
    if not args.out and not args.append_to:
        print(text)
    print(f"[summary] {len(arms)} arms")


if __name__ == "__main__":
    main()
