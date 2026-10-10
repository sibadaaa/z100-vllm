#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# S7 diagnostic: greedy-generate several fixed prompts in one engine so an
# eager-mode numerical problem can be told apart from a prompt-specific one.
#
#   bash tools/dcu/s7_prompts.sh <model> <tag> <extra kwargs> [log]

set -uo pipefail

MODEL="${1:?usage: s7_prompts.sh <model> <tag> [<extra kwargs>] [log]}"
TAG="${2:?usage: s7_prompts.sh <model> <tag> [<extra kwargs>] [log]}"
EXTRA="${3:-}"
LOG="${4:-/mnt/game/port/s7/s7-prompts-${TAG}.log}"

mkdir -p "$(dirname "${LOG}")"

python -u - "$MODEL" "$TAG" "$EXTRA" <<'PY' 2>&1 | tee "${LOG}"
import sys
import time

import torch  # noqa: F401
from vllm import LLM, SamplingParams

model, tag, extra = sys.argv[1], sys.argv[2], sys.argv[3]
kwargs = eval("dict(" + extra + ")") if extra else {}

PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time,",
    "def fibonacci(n):",
    "The largest planet in the solar system is",
    "Water boils at",
]

llm = LLM(model=model, dtype="float16", max_model_len=2048,
          gpu_memory_utilization=0.9, **kwargs)
print(f"[s7] engine ready tag={tag}", flush=True)

t0 = time.time()
outs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True))
for p, o in zip(PROMPTS, outs):
    print(f"[s7] PROMPT={p!r} -> {o.outputs[0].text!r}", flush=True)
print(f"[s7] SECONDS={time.time() - t0:.2f}", flush=True)
PY
rc=$?
echo "S7_PROMPTS_RC=${rc}"
exit "${rc}"
