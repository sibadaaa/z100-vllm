# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S7b: locate the first non-finite tensor produced by the gfx906 eager path.

    python tools/dcu/s7b_scan.py <model> <tag> [<extra vllm kwargs>]

The extra kwargs are evaluated as Python (same convention as
``tools/dcu/s7_compare.sh``), e.g.

    python tools/dcu/s7b_scan.py /mnt/game/models/Qwen3-0.6B eager \
        "enforce_eager=True"

The scan log goes to ``$S7B_SCAN`` / ``/mnt/game/port/s7b/layer_scan-<tag>.jsonl``.
"""

import json
import os
import sys
import time

import torch  # noqa: F401  (must come before vllm on DTK, hard rule 8)

TAG = sys.argv[2] if len(sys.argv) > 2 else "eager"
os.environ.setdefault("S7B_SCAN", f"/mnt/game/port/s7b/layer_scan-{TAG}.jsonl")

# The hooks module must be importable by reference inside the worker process.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s7b_hooks  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

model = sys.argv[1]
extra = sys.argv[3] if len(sys.argv) > 3 else ""
kwargs = eval("dict(" + extra + ")") if extra else {}

PROMPT = "The capital of France is"

t0 = time.time()
llm = LLM(
    model=model,
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.9,
    **kwargs,
)
print(f"[s7b] engine ready in {time.time() - t0:.1f}s tag={TAG}", flush=True)

probe = llm.collective_rpc(s7b_hooks.worker_probe)
print(f"[s7b] worker probe={probe}", flush=True)

n = probe[0]["modules"] if isinstance(probe, list) else probe["modules"]
print(f"[s7b] hooks installed on {n} modules", flush=True)

out = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))
print(f"[s7b] S7B_IDS[{TAG}]={out[0].outputs[0].token_ids}", flush=True)
print(f"[s7b] S7B_TEXT[{TAG}]={out[0].outputs[0].text!r}", flush=True)

scan = os.environ["S7B_SCAN"]
rows = [json.loads(line) for line in open(scan)]
mods = [r for r in rows if r.get("kind") == "MODULE"]
bad = [r for r in mods if "first_nonfinite" in r]
print(f"[s7b] SCAN={scan} modules={len(mods)} nonfinite_modules={len(bad)}", flush=True)
for r in bad[:12]:
    print(
        f"[s7b] NONFINITE pass={r['pass']} i={r['i']} {r['name']} "
        f"({r['cls']}) {r['first_nonfinite']}",
        flush=True,
    )
print(
    "[s7b] S7B_SCAN_DONE",
    flush=True,
)
