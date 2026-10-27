#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# S18: verify that VLLM_DCU_FP8_GEMV_MAX_M actually changes the block-scaled FP8
# dispatch, and show the M-scaling at the op level (not the raw kernel).
#
#   HIP_VISIBLE_DEVICES=3 python /src/tools/dcu/s18_dispatch_check.py [--max-m 1]
#
# Expected (8704x5120, M=2): GEMV policy ~1.33 ms, GEMM policy ~0.58 ms.
import os
import sys

os.environ.setdefault(
    "VLLM_DCU_FP8_GEMV_MAX_M",
    sys.argv[sys.argv.index("--max-m") + 1] if "--max-m" in sys.argv else "1",
)

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s18_fp8_block_mprobe import BLOCK_K, BLOCK_N, lut_for, make_case  # noqa: E402

from vllm.model_executor.kernels.linear.scaled_mm import gfx906_fp8 as K  # noqa: E402

print(f"[check] VLLM_DCU_FP8_GEMV_MAX_M={os.environ['VLLM_DCU_FP8_GEMV_MAX_M']} "
      f"-> module _GEMV_MAX_M={K._GEMV_MAX_M}")

n, k = 8704, 5120
lut = lut_for()
a_q, w, a_s, w_s = make_case(n, k, 4)
for m in (1, 2, 3, 4):
    a, s = a_q[:m].contiguous(), a_s[:m].contiguous()

    def call():
        return torch.ops.vllm.fp8_emu_mm_block(
            a, w, s, w_s, lut, None, torch.float16, BLOCK_N, BLOCK_K
        )

    for _ in range(5):
        call()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(30):
        call()
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / 30
    print(f"[check] M={m} op-level {ms:.4f} ms  weight_gbs={n * k / ms / 1e6:.0f}")
