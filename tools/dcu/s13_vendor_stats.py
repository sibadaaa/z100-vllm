#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S13 phase C: summarise the vendor (production-stack) fair-baseline repeats.

Reads the host-side logs written by tools/dcu/s13_vendor_arm.sh
(/mnt/game/s13/hostlogs/vendor-<NAME>-bench-<mode>.log) and, for each ACS state,
prints the per-run decode Median TPOT plus mean/min/max/spread, and whether the
two states' [min, max] intervals overlap.

Usage:
    python tools/dcu/s13_vendor_stats.py /mnt/game/s13/hostlogs
"""

import glob
import os
import re
import sys

PAT = re.compile(r"^(?P<k>[A-Za-z][^:]*?):\s+(?P<v>[\d.]+)\s*$")
WANT = ["Median TPOT (ms)", "Output token throughput (tok/s)", "Median TTFT (ms)"]
NAME = re.compile(r"vendor-(?P<label>.+)-bench-(?P<mode>decode|ttft|conc8)\.log$")


def parse(path: str) -> dict:
    vals = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = PAT.match(line.strip())
            if m and m.group("k") in WANT:
                vals.setdefault(m.group("k"), m.group("v"))
    return vals


def state_of(label: str) -> str:
    # S13VOFF1 -> off, S13VON1 -> on; the historical S12 names are VOFF/VON.
    return "acs_on" if re.search(r"VON", label) else "acs_off"


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/game/s13/hostlogs"
    runs: dict[str, dict[str, dict]] = {"acs_off": {}, "acs_on": {}}
    for pat in ("vendor-S13VOFF*-bench-*.log", "vendor-S13VON*-bench-*.log",
                "vendor-VOFF*-bench-*.log", "vendor-VON*-bench-*.log"):
        for path in sorted(glob.glob(os.path.join(root, pat))):
            m = NAME.search(os.path.basename(path))
            if not m:
                continue
            runs[state_of(m.group("label"))].setdefault(m.group("label"), {})[
                m.group("mode")
            ] = parse(path)

    for state in ("acs_off", "acs_on"):
        print(f"=== {state} ===")
        tpot = []
        for label, modes in sorted(runs[state].items()):
            v = modes.get("decode", {})
            if not v:
                print(f"  {label}: (no decode log)")
                continue
            tp = float(v.get("Median TPOT (ms)", "nan"))
            tpot.append((label, tp))
            print(f"  {label}: decode Median TPOT={tp:.2f}  throughput="
                  f"{v.get('Output token throughput (tok/s)', '-')}  "
                  f"ttft(in=3800)={modes.get('ttft', {}).get('Median TTFT (ms)', '-')}  "
                  f"conc8={modes.get('conc8', {}).get('Output token throughput (tok/s)', '-')}")
        if tpot:
            xs = [x for _, x in tpot]
            print(f"  n={len(xs)} mean={sum(xs) / len(xs):.2f} min={min(xs):.2f} "
                  f"max={max(xs):.2f} spread={(max(xs) - min(xs)) / min(xs) * 100:.1f}%")

    off = [x for _, x in tpot_of(runs, "acs_off")]
    on = [x for _, x in tpot_of(runs, "acs_on")]
    if off and on:
        overlap = not (max(off) < min(on) or max(on) < min(off))
        print(f"=== verdict ===\n  acs_off n={len(off)} [{min(off):.2f}, {max(off):.2f}] "
              f"mean={sum(off) / len(off):.2f}")
        print(f"  acs_on  n={len(on)} [{min(on):.2f}, {max(on):.2f}] "
              f"mean={sum(on) / len(on):.2f}")
        print(f"  intervals_overlap={overlap}  "
              f"mean_ratio(off/on)={sum(off) / len(off) / (sum(on) / len(on)):.4f}")


def tpot_of(runs: dict, state: str) -> list[tuple[str, float]]:
    out = []
    for label, modes in sorted(runs[state].items()):
        v = modes.get("decode", {})
        if v:
            out.append((label, float(v.get("Median TPOT (ms)", "nan"))))
    return out


if __name__ == "__main__":
    main()
