// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <torch/all.h>

// W8A8 INT8 GEMM/GEMV built on the gfx906 `V_DOT4_I32_I8` instruction.
// See csrc/rocm/w8a8_sdot4.cu for the layout contract and provenance.
//
// Scale contract (shared with csrc/rocm/fp8_emu_gemm.cu):
//   * `x_s` may hold M elements (dynamic per-token) or 1 (static per-tensor);
//   * `w_s` may hold N elements (per-channel) or 1 (per-tensor).
// A 1-element scale is applied as a scalar (stride 0 inside the kernel).

// Decode path (M <= 8): one block per output column, LDS tree reduction.
// `nthreads` is the GEMV block size (power of two, <= 512; 64 is the tuned
// value).  Returns [M, N] in `out_dtype`.
torch::Tensor w8a8_sdot4_gemv(torch::Tensor xq, torch::Tensor w,
                              torch::Tensor x_s, torch::Tensor w_s,
                              std::optional<torch::Tensor> bias,
                              at::ScalarType out_dtype, int64_t nthreads);

// Prefill path (any M): BM x BN register tile, BK=32 double-buffered LDS.
// `cfg` selects the tile: 0 = auto (32x64 for M<=32, else 64x64), 1..4 =
// explicit (32x64, 64x64, 64x128, 128x128).  Returns [M, N] in `out_dtype`.
torch::Tensor w8a8_sdot4_gemm(torch::Tensor xq, torch::Tensor w,
                              torch::Tensor x_s, torch::Tensor w_s,
                              std::optional<torch::Tensor> bias,
                              at::ScalarType out_dtype, int64_t cfg);

torch::Tensor w8a8_sdot4_gemv_meta(torch::Tensor xq, torch::Tensor w,
                                   torch::Tensor x_s, torch::Tensor w_s,
                                   std::optional<torch::Tensor> bias,
                                   at::ScalarType out_dtype, int64_t nthreads);

torch::Tensor w8a8_sdot4_gemm_meta(torch::Tensor xq, torch::Tensor w,
                                   torch::Tensor x_s, torch::Tensor w_s,
                                   std::optional<torch::Tensor> bias,
                                   at::ScalarType out_dtype, int64_t cfg);
