#pragma once

#include <torch/all.h>

#include <optional>

#include "rocm/bf16_emu_gemv.h"
#include "rocm/fp8_emu_gemm.h"
#include "rocm/w8a8_sdot4.h"

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount);

torch::Tensor wvSplitK_int4_g(const at::Tensor& in_a, const at::Tensor& in_b,
                              const at::Tensor& in_scale,
                              const std::optional<at::Tensor>& in_zero_points,
                              const std::optional<at::Tensor>& in_bias,
                              const int64_t CuCount, const int64_t group_size);

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount);

void wvSplitKQ(const at::Tensor& in_a, const at::Tensor& in_b,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount);

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              bool use_v2_format);

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales, bool use_v2_format);

void moe_gptq_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type, bool use_interleaved_v_cache);

// S53-ATTN: gfx906 decode split-KV attention inner-loop rebuild (HIP).
// Supported subset only (gates inside + the Python dispatch); everything
// else stays on the Triton path. segm_* buffers and reduce_segments are
// shared with the Triton 3D kernel. B832: any block_size % 16 == 0;
// k/v_descale (optional fp32, [num_seqs, nkv] or broadcast) fold into the
// softmax scale / segment-end accumulator.
void gfx906_attnd_decode_inner(
    torch::Tensor& segm_output, torch::Tensor& segm_max,
    torch::Tensor& segm_expsum, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, torch::Tensor& block_tables,
    torch::Tensor& seq_lens, torch::Tensor& cu_query_lens, double scale,
    int64_t num_segments, bool delayed_merge, int64_t tps_hint,
    std::optional<torch::Tensor> k_descale,
    std::optional<torch::Tensor> v_descale);
