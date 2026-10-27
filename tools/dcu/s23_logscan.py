#!/usr/bin/env python3
"""Scan a service startup log slice for the failure signals O7 has to explain.

The plan requires, if piecewise CUDA graphs cannot start, **exactly three failure
reasons, each with the log line that shows it**.  This helper pulls the candidate
lines out of an arm's ``startup.log`` (the slice the driver saves for the arm)
and groups them by mechanism, so the three reasons are quoted from evidence
rather than guessed.

  .venv/bin/python tools/dcu/s23_logscan.py --log .s23/w1/O7/o7/startup.log \
      [--out scan.json] [--context 2]

Exit 0 when a startup-success marker is present, 1 when the log shows a failure.
"""

import argparse
import json
import re

SIGNALS = {
    "oom": r"out of memory|OutOfMemoryError|No available memory|"
           r"CUDA out of memory|HIP out of memory|memory pool",
    "cudagraph": r"cudagraph|CUDA graph|CUDAGraph|graph pool|"
                 r"capture|replay",
    "compile": r"torch\.compile|Dynamo|dynamo|inductor|ConstraintViolation|"
               r"backend=|fx graph|BackendCompilerFailed",
    "splitting_ops": r"splitting_ops|piecewise|PIECEWISE",
    "import": r"ImportError|ModuleNotFoundError|undefined symbol|"
              r"NoBinaryForGpu|hipError|undefined reference",
    "assert": r"AssertionError|RuntimeError|ValueError|TypeError|KeyError",
    "traceback": r"Traceback \(most recent call last\)",
    "ready": r"Application startup complete|Starting vLLM server|"
             r"init engine .* took",
}

_RX = {name: re.compile(pat, re.I) for name, pat in SIGNALS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--max-per-signal", type=int, default=6)
    args = ap.parse_args()

    with open(args.log, errors="ignore") as f:
        lines = f.read().splitlines()

    hits = {name: [] for name in SIGNALS}
    for i, line in enumerate(lines):
        for name, rx in _RX.items():
            if rx.search(line) and len(hits[name]) < args.max_per_signal:
                lo = max(0, i - args.context)
                hi = min(len(lines), i + args.context + 1)
                hits[name].append({
                    "line_no": i + 1,
                    "line": line.strip()[:400],
                    "context": [x.strip()[:300] for x in lines[lo:hi]],
                })

    ready = bool(hits["ready"])
    errors = [h for name in ("traceback", "assert", "import", "oom")
              for h in hits[name]]
    verdict = "started" if ready and not errors else "failed"
    payload = {"log": args.log, "lines": len(lines), "verdict": verdict,
               "signals": {k: v for k, v in hits.items() if v}}

    for name in SIGNALS:
        if not hits[name]:
            continue
        print(f"### {name} ({len(hits[name])} shown)")
        for hit in hits[name]:
            print(f"  L{hit['line_no']}: {hit['line']}")
        print()
    print(f"[logscan] {args.log}: {len(lines)} lines, verdict={verdict}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[logscan] saved -> {args.out}")
    return 0 if verdict == "started" else 1


if __name__ == "__main__":
    raise SystemExit(main())
