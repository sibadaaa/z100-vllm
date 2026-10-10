#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# S7 eager/graph A/B harness. Runs the same greedy prompt under a matrix of
# compilation/cudagraph settings on a single gfx906 card and prints the token
# ids so the modes can be compared byte for byte.
#
#   bash tools/dcu/s7_compare.sh <model> <tag> <extra vllm kwargs as python>
#   bash tools/dcu/s7_compare.sh /mnt/game/models/Qwen3-0.6B eager-cg-off \
#       "enforce_eager=True, compilation_config={'mode': 'none', 'cudagraph_mode': 'none'}"

set -uo pipefail

MODEL="${1:?usage: s7_compare.sh <model> <tag> [<extra kwargs>]}"
TAG="${2:?usage: s7_compare.sh <model> <tag> [<extra kwargs>]}"
EXTRA="${3:-}"
LOG="${4:-/mnt/game/port/s7/s7-compare-${TAG}.log}"

mkdir -p "$(dirname "${LOG}")"

python -u - "$MODEL" "$TAG" "$EXTRA" <<'PY' 2>&1 | tee "${LOG}"
import json
import sys
import time

import torch  # noqa: F401
from vllm import LLM, SamplingParams

model, tag, extra = sys.argv[1], sys.argv[2], sys.argv[3]
kwargs = eval("dict(" + extra + ")") if extra else {}

PROMPT = "The capital of France is"
greedy = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)

t0 = time.time()
llm = LLM(
    model=model,
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.9,
    **kwargs,
)
print(f"[s7] engine ready in {time.time() - t0:.1f}s tag={tag}", flush=True)

out = llm.generate([PROMPT], greedy)
ids = out[0].outputs[0].token_ids
print(f"[s7] S7_IDS[{tag}]={json.dumps(ids)}", flush=True)
print(f"[s7] S7_TEXT[{tag}]={out[0].outputs[0].text!r}", flush=True)

llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
start = time.perf_counter()
out_t = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True))
elapsed = time.perf_counter() - start
n = len(out_t[0].outputs[0].token_ids)
print(f"[s7] S7_DECODE[{tag}] TOKENS={n} SECONDS={elapsed:.4f} MS_PER_TOKEN={elapsed * 1000 / n:.3f}", flush=True)
PY
rc=$?
echo "S7_COMPARE_RC=${rc}"
exit "${rc}"
