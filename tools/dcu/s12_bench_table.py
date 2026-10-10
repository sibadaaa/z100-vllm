#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""S12: summarise the ``vllm bench serve`` logs produced by s12_bench.sh.

Usage:
    python tools/dcu/s12_bench_table.py /mnt/game/s12/bench-*.log

Prints one row per log with the metrics the S9/S11l protocol compared:
median/mean TTFT and TPOT, ITL and output throughput.
"""

import glob
import re
import sys
from collections import OrderedDict

WANT = [
    "Successful requests",
    "Benchmark duration (s)",
    "Total input tokens",
    "Output token throughput (tok/s)",
    "Mean TTFT (ms)",
    "Median TTFT (ms)",
    "Mean TPOT (ms)",
    "Median TPOT (ms)",
    "Mean ITL (ms)",
]
PAT = re.compile(r"^(?P<k>[A-Za-z][^:]*?):\s+(?P<v>[\d.]+)\s*$")
NAME = re.compile(r"bench-(?P<label>.+)-(?P<mode>decode|ttft|ttft8k|conc8)\.log$")


def parse(path):
    vals = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = PAT.match(line.strip())
            if m and m.group("k") in WANT:
                vals[m.group("k")] = m.group("v")
    return vals


def main():
    paths = []
    for pat in sys.argv[1:]:
        paths.extend(sorted(glob.glob(pat)))
    rows = OrderedDict()
    for p in paths:
        m = NAME.search(p)
        if not m:
            continue
        rows[(m.group("label"), m.group("mode"))] = parse(p)
    head = ["label", "mode"] + WANT
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for (label, mode), v in rows.items():
        print("| " + " | ".join([label, mode] + [v.get(k, "—") for k in WANT]) + " |")


if __name__ == "__main__":
    main()
