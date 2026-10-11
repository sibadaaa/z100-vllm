// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <torch/all.h>

// FP8 (W8A8) GEMM/GEMV for gfx906 (Vega20 / Hygon Z100).
//
// gfx906 has no fp8 conversion instruction and no MFMA/WMMA, so this kernel
// emulates FP8 with fp16 hardware: fp8 -> fp16 through a 256-entry LDS table
// (`ds_read_u16`, exact for e4m3fn and e5m2) followed by `v_dot2_f32_f16`
// with fp32 accumulation.  Weights are consumed **in their checkpoint layout**
// ([N, K], one byte per element); nothing is dequantized, transposed or
// materialized into fp16.
//
// See csrc/rocm/fp8_emu_gemm.cu for the layout contract and provenance.

// Decode path (M <= 8): one block per output column, LDS tree reduction.
// `a_s`/`w_s` may hold 1 element (per-tensor) or M/N elements.
torch::Tensor fp8_emu_gemv(torch::Tensor a, torch::Tensor w, torch::Tensor a_s,
                           torch::Tensor w_s, torch::Tensor lut,
                           std::optional<torch::Tensor> bias,
                           at::ScalarType out_dtype, int64_t nthreads);

// Prefill path (any M): BM x BN register tile, BK = 32 double-buffered LDS.
// `cfg` selects the tile: 0 = auto, 1..4 = explicit.
torch::Tensor fp8_emu_gemm(torch::Tensor a, torch::Tensor w, torch::Tensor a_s,
                           torch::Tensor w_s, torch::Tensor lut,
                           std::optional<torch::Tensor> bias,
                           at::ScalarType out_dtype, int64_t cfg);

torch::Tensor fp8_emu_gemv_meta(torch::Tensor a, torch::Tensor w,
                                torch::Tensor a_s, torch::Tensor w_s,
                                torch::Tensor lut,
                                std::optional<torch::Tensor> bias,
                                at::ScalarType out_dtype, int64_t nthreads);

torch::Tensor fp8_emu_gemm_meta(torch::Tensor a, torch::Tensor w,
                                torch::Tensor a_s, torch::Tensor w_s,
                                torch::Tensor lut,
                                std::optional<torch::Tensor> bias,
                                at::ScalarType out_dtype, int64_t cfg);

// Block-scaled variants (FP8 checkpoint with `weight_scale_inv` shaped
// [cdiv(N, block_n), cdiv(K, block_k)] and dynamic per-token-group activations):
//
//   out[m, n] = sum_kb (sum_{k in kb} a[m,k] * w[n,k]) * a_s[m, kb] * w_s[nb, kb]
//
// `w` is [N, K] with stride(1) == 1 (it may be padded along K by
// process_fp8_weight_block_strategy); `a_s` is [M, cdiv(K, block_k)]; `w_s` is
// [cdiv(N, block_n), cdiv(K, block_k)], both float32.  `block_k` must divide K.
torch::Tensor fp8_emu_gemv_block(torch::Tensor a, torch::Tensor w,
                                 torch::Tensor a_s, torch::Tensor w_s,
                                 torch::Tensor lut,
                                 std::optional<torch::Tensor> bias,
                                 at::ScalarType out_dtype, int64_t nthreads,
                                 int64_t block_n, int64_t block_k);

torch::Tensor fp8_emu_gemm_block(torch::Tensor a, torch::Tensor w,
                                 torch::Tensor a_s, torch::Tensor w_s,
                                 torch::Tensor lut,
                                 std::optional<torch::Tensor> bias,
                                 at::ScalarType out_dtype, int64_t cfg,
                                 int64_t block_n, int64_t block_k);

torch::Tensor fp8_emu_gemv_block_meta(torch::Tensor a, torch::Tensor w,
                                      torch::Tensor a_s, torch::Tensor w_s,
                                      torch::Tensor lut,
                                      std::optional<torch::Tensor> bias,
                                      at::ScalarType out_dtype, int64_t nthreads,
                                      int64_t block_n, int64_t block_k);

torch::Tensor fp8_emu_gemm_block_meta(torch::Tensor a, torch::Tensor w,
                                      torch::Tensor a_s, torch::Tensor w_s,
                                      torch::Tensor lut,
                                      std::optional<torch::Tensor> bias,
                                      at::ScalarType out_dtype, int64_t cfg,
                                      int64_t block_n, int64_t block_k);
