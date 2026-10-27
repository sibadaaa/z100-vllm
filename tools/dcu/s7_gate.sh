#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# S7 gate: run vLLM V1 offline inference on a single gfx906 card, report the KV
# cache sizing, the first decoded tokens and the decode latency.
#
#   bash tools/dcu/s7_gate.sh eager <model> [log]
#   bash tools/dcu/s7_gate.sh graph <model> [log]
#
# The prompt and the sampling parameters are identical in both modes, so the
# generated token ids can be compared byte for byte.

set -uo pipefail

MODE="${1:?usage: s7_gate.sh <eager|graph> <model> [log]}"
MODEL="${2:?usage: s7_gate.sh <eager|graph> <model> [log]}"
LOG="${3:-/mnt/game/port/s7/s7-${MODE}.log}"

mkdir -p "$(dirname "${LOG}")"

python -u - "$MODE" "$MODEL" <<'PY' 2>&1 | tee "${LOG}"
import json
import sys
import time

import torch  # noqa: F401  (must come first on DTK, see todolist hard rule 8)
from vllm import LLM, SamplingParams

mode, model = sys.argv[1], sys.argv[2]
PROMPT = "The capital of France is"
greedy = dict(temperature=0.0, ignore_eos=True)

t0 = time.time()
llm = LLM(
    model=model,
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.9,
    enforce_eager=(mode == "eager"),
)
print(f"[s7] engine ready in {time.time() - t0:.1f}s mode={mode}", flush=True)

out = llm.generate([PROMPT], SamplingParams(max_tokens=16, **greedy))
ids = out[0].outputs[0].token_ids
print(f"[s7] S7_IDS_{mode.upper()}={json.dumps(ids)}", flush=True)
print(f"[s7] S7_TEXT_{mode.upper()}={out[0].outputs[0].text!r}", flush=True)

# Warm up on the decode path (prefill once, then one token) so that the timed
# run below is pure decode.
llm.generate([PROMPT], SamplingParams(max_tokens=1, **greedy))
start = time.perf_counter()
out_t = llm.generate([PROMPT], SamplingParams(max_tokens=128, **greedy))
elapsed = time.perf_counter() - start
n = len(out_t[0].outputs[0].token_ids)
print(
    f"[s7] S7_DECODE_{mode.upper()} TOKENS={n} SECONDS={elapsed:.4f} "
    f"MS_PER_TOKEN={elapsed * 1000 / n:.3f}",
    flush=True,
)
PY
rc=$?
echo "S7_GATE_RC=${rc}"
exit "${rc}"
