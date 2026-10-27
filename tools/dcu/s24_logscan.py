#!/usr/bin/env python3
"""Extract the S24 memory facts from one vLLM startup-log slice.

Answers, per arm (S24 plan section 1.3):
  * model load GiB, available KV cache memory GiB + token count
  * the static KV pool utilisation math (util x total GiB)
  * the two structural warnings: padding layers (25% KV waste) and the
    missing draft KV-cache group
  * resolved config lines that matter for L5 (MBT, max-num-seqs, block size)

  .venv/bin/python tools/dcu/s24_logscan.py --log <dir>/startup.log \
      --out <dir>/memfacts.json
"""

import argparse
import json
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.log, errors="ignore") as f:
        txt = f.read()

    facts = {}

    def grab(pattern, key, cast=str, default=None):
        m = re.search(pattern, txt)
        facts[key] = cast(m.group(1)) if m else default

    grab(r"Available KV cache memory: ([\d.]+) GiB", "kv_pool_gib", float)
    grab(r"GPU KV cache size: ([\d,]+) tokens",
         "pool_tokens", lambda s: int(s.replace(",", "")))
    grab(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x",
         "max_conc_pair", str)
    grab(r"Model loading took ([\d.]+) GiB", "model_load_gib", float)
    for i, m in enumerate(re.finditer(
            r"Add (\d+) padding layers, may waste at most ([\d.]+)%", txt)):
        facts[f"padding_{i}_layers"] = int(m.group(1))
        facts[f"padding_{i}_waste_pct"] = float(m.group(2))
    grab(r"kv cache group sizes \(([\d, ]+)\)", "kv_group_sizes", str)
    grab(r"max_num_batched_tokens=(\d+)", "mbt", int)
    grab(r"max_num_seqs=(\d+)", "max_num_seqs", int)
    grab(r"gpu_memory_utilization=([\d.]+)", "util", float)
    grab(r"kv_cache_dtype=\"?([a-z0-9]+)\"?", "kv_cache_dtype", str)
    grab(r"block_size=(\d+)", "block_size", int)

    facts["draft_kv_group_warning"] = bool(
        re.search(r"no KV cache group could be identified as the\s*draft model", txt))
    # mamba/align page info if present
    m = re.search(r"(?:mamba[_ ]page|page size[^\n]*align[^\n]*)", txt, re.I)
    facts["mamba_align_line"] = m.group(0)[:200] if m else None

    print("[logscan] " + json.dumps(facts))
    with open(args.out, "w") as f:
        json.dump(facts, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
