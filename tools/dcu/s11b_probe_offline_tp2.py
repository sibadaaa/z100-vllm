#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cheap probe: does offline `LLM` + TP=2 init devices with fork vs spawn?

Used to root-cause the S11b 27B failure
``RuntimeError: CUDA driver initialization failed`` in WorkerProc.
"""

import sys

import torch  # noqa: F401 - hard rule 8: import torch before vllm
from vllm import LLM, SamplingParams


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "/mnt/game/models/Qwen3-0.6B"
    multi = len(sys.argv) > 2 and sys.argv[2] == "full"
    print(f"PROBE_START model={model} multi={multi}", flush=True)
    print(f"PROBE_CUDA_INIT_BEFORE={torch.cuda.is_initialized()}", flush=True)
    kwargs = {}
    if multi:
        kwargs = {
            "max_model_len": 4096,
            "max_num_batched_tokens": 2048,
            "gpu_memory_utilization": 0.9,
            "skip_mm_profiling": True,
            "enable_prefix_caching": True,
            "mamba_cache_mode": "align",
            "compilation_config": {
                "cudagraph_capture_sizes": [1, 2],
            },
            "cudagraph_metrics": True,
        }
    else:
        kwargs = {
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.5,
            "enforce_eager": True,
        }
    llm = LLM(
        model=model,
        dtype="float16",
        tensor_parallel_size=2,
        disable_custom_all_reduce=True,
        **kwargs,
    )
    print("PROBE_LLM_OK", flush=True)
    out = llm.generate(["Hello"], SamplingParams(max_tokens=4, temperature=0.0))
    print("PROBE_GEN_OK", out[0].outputs[0].text[:60], flush=True)


if __name__ == "__main__":
    main()
