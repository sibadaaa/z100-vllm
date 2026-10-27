#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# S7 diagnostic: is the gfx906 eager/graph numerical split caused by
# torch.compile, or by CUDA graph capture?
#
#   bash tools/dcu/s7_compile_matrix.sh <model> <cudagraph-none|compile-none|full>
#
#   cudagraph-none  enforce_eager=False, mode=VLLM_COMPILE, cudagraph_mode=NONE
#   compile-none    enforce_eager=False, mode=NONE,          cudagraph_mode=NONE
#   full            upstream defaults (mode=VLLM_COMPILE, FULL_AND_PIECEWISE)

set -uo pipefail

MODEL="${1:?usage: s7_compile_matrix.sh <model> <cudagraph-none|compile-none|full>}"
CASE="${2:?usage: s7_compile_matrix.sh <model> <cudagraph-none|compile-none|full>}"
LOG="/mnt/game/port/s7/s7-matrix-${CASE}.log"

mkdir -p "$(dirname "${LOG}")"

python -u - "$MODEL" "$CASE" <<'PY' 2>&1 | tee "${LOG}"
import sys
import time

import torch  # noqa: F401
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode

model, case = sys.argv[1], sys.argv[2]
if case == "cudagraph-none":
    cc = CompilationConfig(mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.NONE)
    kwargs = {"enforce_eager": False, "compilation_config": cc}
elif case == "compile-none":
    cc = CompilationConfig(mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE)
    kwargs = {"enforce_eager": False, "compilation_config": cc}
elif case == "full":
    kwargs = {}
else:
    raise SystemExit(f"unknown case {case}")

PROMPT = "The capital of France is"
llm = LLM(model=model, dtype="float16", max_model_len=2048,
          gpu_memory_utilization=0.9, **kwargs)
print(f"[s7] engine ready case={case}", flush=True)

out = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True))
print(f"[s7] S7_IDS[{case}]={out[0].outputs[0].token_ids}", flush=True)
print(f"[s7] S7_TEXT[{case}]={out[0].outputs[0].text!r}", flush=True)

llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
start = time.perf_counter()
out_t = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True))
elapsed = time.perf_counter() - start
n = len(out_t[0].outputs[0].token_ids)
print(f"[s7] S7_DECODE[{case}] TOKENS={n} SECONDS={elapsed:.4f} MS_PER_TOKEN={elapsed * 1000 / n:.3f}", flush=True)
PY
rc=$?
echo "S7_MATRIX_RC=${rc}"
exit "${rc}"
