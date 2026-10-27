#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A7 (S11a): key-by-key comparison of two ``a7_non_dtk_probe.py`` outputs.

    python3 tools/dcu/a7_compare_jsonl.py canonical-sim.rocm-real.jsonl \
                                            head-sim.rocm-real.jsonl

The ``tree`` field (an absolute path, different by construction) is dropped.
Exit status is 0 when the two records agree everywhere else.
"""

from __future__ import annotations

import json
import sys

MISSING = "<missing>"


def load(path: str) -> dict:
    with open(path) as handle:
        record = json.loads(handle.read().strip().splitlines()[-1])
    record.pop("tree", None)
    return record


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    left, right = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(left) | set(right))
    diffs = 0
    for key in keys:
        a, b = left.get(key, MISSING), right.get(key, MISSING)
        if a != b:
            diffs += 1
            print(
                f"DIFF {key}\n  {sys.argv[1]}: {json.dumps(a, sort_keys=True)}\n"
                f"  {sys.argv[2]}: {json.dumps(b, sort_keys=True)}"
            )
    print(f"{len(keys) - diffs}/{len(keys)} keys identical ({len(keys)} compared)")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
