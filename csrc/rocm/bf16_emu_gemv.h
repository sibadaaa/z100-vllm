// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <torch/all.h>

// Dense (unquantized) fp16 / bf16 GEMV for gfx906 (Vega20 / Hygon Z100).
//
// This is the decode-path (M <= 8) replacement for `torch.nn.functional.linear`
// on gfx906, where neither the ROCm skinny-GEMM ops (LLMM1/wvSplitK*: excluded
// from the gfx906 build, they target the MFMA ISA) nor the `n <= 16` Triton
// matmul (`not on_gfx9()`) can be used, so every decode-shaped linear layer
// falls through to rocBLAS.  Measured baseline (`F.linear`, this machine):
// bf16 0.758 ms / fp16 0.830 ms for [1 x 5120] @ [17408 x 5120] (235 GB/s),
// i.e. ~24% of the card's HBM bandwidth, with small-N shapes far worse
// (N=512 bf16: 0.215 ms -> 24 GB/s).
//
// Datapath: for fp16 inputs the kernel uses `v_dot2_f32_f16` directly (two
// fp16 MACs per instruction, fp32 accumulator, exact); for bf16 inputs it
// either widens to fp32 with `v_lshlrev_b32 16` (exact for every bf16 value)
// or, opt-in, rebases to fp16 with packed 16-bit ALU ops and uses the dot2
// path.  See csrc/rocm/bf16_emu_gemv.cu for the tradeoffs and the range
// requirements of the fp16 rebase route.
//
// `route`: 0 = auto (fp16 -> dot2, bf16 -> fp32), 1 = fp32 (exact),
//          2 = fp16 dot2 (bf16 requires `range_safe` or a range check).
torch::Tensor bf16_emu_gemv(torch::Tensor x, torch::Tensor w,
                            std::optional<torch::Tensor> bias, int64_t nthreads,
                            int64_t route, bool range_safe);

torch::Tensor bf16_emu_gemv_meta(torch::Tensor x, torch::Tensor w,
                                 std::optional<torch::Tensor> bias,
                                 int64_t nthreads, int64_t route,
                                 bool range_safe);
