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

// S23 / O5: activation quantization fused into the block-scaled path.  `a` is
// the *unquantized* activation (fp16/bf16), `w_s` the block-scale matrix.
// block_k must be 128 (the quantizer group).  The per-token-group quantizer
// runs inside the decode tile (one launch, `y_s` in LDS) or, for the
// stream/large-M routes or K/block_k > 64, in one extra launch feeding the
// existing kernel.  Returns `out` only; `_quant_x` also returns the
// `(x_q, x_s)` witness for the bit-exact gate -- on the fused-tile route the
// witness is emitted by the same GEMM launch, on the fallback routes by the
// standalone quantizer the GEMM consumes.
torch::Tensor fp8_emu_gemm_block_quant(
    torch::Tensor a, torch::Tensor w, torch::Tensor w_s, torch::Tensor lut,
    std::optional<torch::Tensor> bias, at::ScalarType out_dtype,
    int64_t nthreads, int64_t cfg, int64_t block_n, int64_t block_k, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fp8_emu_gemm_block_quant_x(torch::Tensor a, torch::Tensor w, torch::Tensor w_s,
                           torch::Tensor lut, std::optional<torch::Tensor> bias,
                           at::ScalarType out_dtype, int64_t nthreads,
                           int64_t cfg, int64_t block_n, int64_t block_k,
                           double eps);

torch::Tensor fp8_emu_gemm_block_quant_meta(
    torch::Tensor a, torch::Tensor w, torch::Tensor w_s, torch::Tensor lut,
    std::optional<torch::Tensor> bias, at::ScalarType out_dtype,
    int64_t nthreads, int64_t cfg, int64_t block_n, int64_t block_k, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fp8_emu_gemm_block_quant_x_meta(torch::Tensor a, torch::Tensor w,
                                torch::Tensor w_s, torch::Tensor lut,
                                std::optional<torch::Tensor> bias,
                                at::ScalarType out_dtype, int64_t nthreads,
                                int64_t cfg, int64_t block_n, int64_t block_k,
                                double eps);

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

// W8A16: `a` is fp16 [M, K] and is **never quantized**; only the weight stays
// fp8 [N, K] with fp32 block scales.  Same block-scaled kernel family as
// `fp8_emu_gemm_block`, minus the A-side LUT widening and minus the per-token
// activation scale (there is no `a_s` at all).  `cfg` selects a tile out of the
// W8A16 set (see the host-side mapping in fp8_emu_gemm.cu; 0 = auto rule),
// `nthreads` sizes the decode GEMV.  K must be a multiple of 8 and of
// `block_k`, and `block_k` a multiple of 32.
torch::Tensor fp8_emu_gemm_block_w8a16(
    torch::Tensor a, torch::Tensor w, torch::Tensor w_s, torch::Tensor lut,
    std::optional<torch::Tensor> bias, at::ScalarType out_dtype,
    int64_t nthreads, int64_t cfg, int64_t block_n, int64_t block_k);

torch::Tensor fp8_emu_gemm_block_w8a16_meta(
    torch::Tensor a, torch::Tensor w, torch::Tensor w_s, torch::Tensor lut,
    std::optional<torch::Tensor> bias, at::ScalarType out_dtype,
    int64_t nthreads, int64_t cfg, int64_t block_n, int64_t block_k);

// Materialize one layer's fp8 weight (times its block scales) into fp16,
// keeping the [N, K] row order so the caller can hand `out.t()` to a row-major
// GEMM.  In-place on `out`; no allocation, so the buffer can be reused across
// layers (the prefill router owns it).
void fp8_dequant_to_fp16_out(torch::Tensor out, torch::Tensor wq,
                             torch::Tensor ws, torch::Tensor lut,
                             int64_t block_n, int64_t block_k);

