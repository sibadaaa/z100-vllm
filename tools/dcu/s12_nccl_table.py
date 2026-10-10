#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S12: turn the two ``s12_nccl_sweep.sh`` logs into one side-by-side table.

Usage:
    python tools/dcu/s12_nccl_table.py /mnt/game/s12/nccl-acs_off.log \
                                       /mnt/game/s12/nccl-acs_on.log
"""

import re
import sys
from collections import OrderedDict

LINE = re.compile(
    r"^\[(?P<tag>[^\]]+)\] all_reduce bf16 \[1,(?P<n>\d+)\] \(\d+ B\): "
    r"min (?P<min>[\d.]+) med (?P<med>[\d.]+) max (?P<max>[\d.]+) us/call"
)
VARIANT = re.compile(r"^### variant=(?P<name>\S+)")

# acs_off is the "after" state, acs_on is the "before" state.
ORDER = ["baseline", "baseline_rep2", "MIN_NCHANNELS_1", "TREE_1ch",
         "P2P_off_shm", "RING_1ch", "LL128_1ch"]


def load(path):
    out = {}
    variant = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = VARIANT.match(line)
            if m:
                variant = m.group("name")
                continue
            m = LINE.match(line.strip())
            if m and variant:
                out.setdefault(variant, {})[int(m.group("n"))] = float(m.group("min"))
    return out


def main():
    off, on = (load(p) for p in sys.argv[1:3])
    sizes = sorted({n for d in (off, on) for v in d.values() for n in v}) or [0]
    seen = [v for v in ORDER if v in off or v in on]
    for v in list(off) + list(on):
        if v not in seen:
            seen.append(v)
    head = ["variant"] + [f"off {n}" for n in sizes] + [f"on {n}" for n in sizes] \
        + [f"x{n}" for n in sizes]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for v in seen:
        row = [v]
        for n in sizes:
            row.append(f"{off.get(v, {}).get(n, float('nan')):.1f}")
        for n in sizes:
            row.append(f"{on.get(v, {}).get(n, float('nan')):.1f}")
        for n in sizes:
            a, b = off.get(v, {}).get(n), on.get(v, {}).get(n)
            row.append("n/a" if not a or not b else f"{b / a:.2f}x")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
