// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <torch/all.h>

#include <optional>
#include <tuple>

// S99: prefill per-channel int8 GEMM with in-op fp8 staging (gfx906).
// `fp8_emu_mm_int8_pc` is the production-shaped entry (fp8 checkpoint weight
// in, fp16 out); `fp8_stage_int8_pc` exposes the staging half so the numerics
// can be unit-tested against a host reference, and `w8a8_sdot4_gemm_pc` is the
// GEMM on already-staged int8 weights.
//
// Scale contract: x_s is per-token [M] (or 1); w_s is the staged per-channel
// [N].  Output dtype is float16 (gfx906 has no bf16 datapath).

std::tuple<torch::Tensor, torch::Tensor> fp8_stage_int8_pc(
    torch::Tensor w, torch::Tensor w_s, torch::Tensor lut, int64_t block_n,
    int64_t block_k);

torch::Tensor w8a8_sdot4_gemm_pc(torch::Tensor xq, torch::Tensor w,
                                 torch::Tensor x_s, torch::Tensor w_s,
                                 std::optional<torch::Tensor> bias,
                                 at::ScalarType out_dtype, int64_t cfg);

torch::Tensor fp8_emu_mm_int8_pc(torch::Tensor xq, torch::Tensor w,
                                 torch::Tensor w_s, torch::Tensor x_s,
                                 torch::Tensor lut,
                                 std::optional<torch::Tensor> bias,
                                 at::ScalarType out_dtype, int64_t block_n,
                                 int64_t block_k);

std::tuple<torch::Tensor, torch::Tensor> fp8_stage_int8_pc_meta(
    torch::Tensor w, torch::Tensor w_s, torch::Tensor lut, int64_t block_n,
    int64_t block_k);

torch::Tensor w8a8_sdot4_gemm_pc_meta(torch::Tensor xq, torch::Tensor w,
                                      torch::Tensor x_s, torch::Tensor w_s,
                                      std::optional<torch::Tensor> bias,
                                      at::ScalarType out_dtype, int64_t cfg);

torch::Tensor fp8_emu_mm_int8_pc_meta(torch::Tensor xq, torch::Tensor w,
                                      torch::Tensor w_s, torch::Tensor x_s,
                                      torch::Tensor lut,
                                      std::optional<torch::Tensor> bias,
                                      at::ScalarType out_dtype,
                                      int64_t block_n, int64_t block_k);
