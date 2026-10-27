#pragma once

#include <torch/all.h>

// GGUF quantization kernels (in-tree fork path; upstream moved GGUF to the
// out-of-tree vllm-gguf-plugin). Declarations for gguf_torch_bindings.cpp.

torch::Tensor ggml_dequantize(torch::Tensor W, int64_t type, int64_t m,
                              int64_t n,
                              std::optional<at::ScalarType> const& dtype);

torch::Tensor ggml_repack_iq4_xs_to_q8_0(torch::Tensor W, int64_t row,
                                         int64_t col);

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X,
                                  int64_t type, int64_t row);

torch::Tensor ggml_quantize_row_q8_1(torch::Tensor X);

void ggml_quantize_row_q8_1_out(torch::Tensor X, torch::Tensor quant_X);

torch::Tensor ggml_silu_and_mul_quantize_row_q8_1(torch::Tensor X);

void ggml_silu_and_mul_quantize_row_q8_1_out(torch::Tensor X,
                                             torch::Tensor quant_X);

torch::Tensor ggml_sigmoid_and_mul_quantize_row_q8_1(torch::Tensor X,
                                                     torch::Tensor gate);

void ggml_sigmoid_and_mul_quantize_row_q8_1_out(torch::Tensor X,
                                                torch::Tensor gate,
                                                torch::Tensor quant_X);

torch::Tensor ggml_rms_norm_gated_quantize_row_q8_1(
    torch::Tensor X, torch::Tensor weight, torch::Tensor gate, double epsilon,
    bool norm_before_gate);

void ggml_rms_norm_gated_quantize_row_q8_1_out(
    torch::Tensor X, torch::Tensor weight, torch::Tensor gate,
    torch::Tensor quant_X, double epsilon, bool norm_before_gate);

torch::Tensor ggml_mul_mat_vec_q8(torch::Tensor W, torch::Tensor quant_X,
                                  int64_t type, int64_t row, int64_t col,
                                  at::ScalarType dtype);

void ggml_mul_mat_vec_q8_out(torch::Tensor W, torch::Tensor quant_X,
                             torch::Tensor Y, int64_t type, int64_t row,
                             int64_t col);

torch::Tensor ggml_mul_mat_vec_q8_0_fast(torch::Tensor W,
                                         torch::Tensor quant_X, int64_t row,
                                         int64_t col, at::ScalarType dtype);

torch::Tensor ggml_mul_mat_vec_a8_sharded(std::vector<torch::Tensor> W,
                                          torch::Tensor X,
                                          std::vector<int64_t> types);

torch::Tensor ggml_mul_mat_vec_q8_sharded(std::vector<torch::Tensor> W,
                                          torch::Tensor quant_X,
                                          std::vector<int64_t> types,
                                          int64_t col, at::ScalarType dtype);

void ggml_mul_mat_vec_q8_sharded_out(std::vector<torch::Tensor> W,
                                     torch::Tensor quant_X,
                                     torch::Tensor Y,
                                     std::vector<int64_t> types, int64_t col);

torch::Tensor ggml_mul_mat_vec_q8_grouped_same_type(
    std::vector<torch::Tensor> W, torch::Tensor quant_X, int64_t type,
    int64_t col, at::ScalarType dtype);

void ggml_mul_mat_vec_q8_grouped_same_type_out(
    std::vector<torch::Tensor> W, torch::Tensor quant_X, torch::Tensor Y,
    int64_t type, int64_t col);

torch::Tensor ggml_mul_mat_vec_q8_qkv3(torch::Tensor W0, torch::Tensor W1,
                                       torch::Tensor W2,
                                       torch::Tensor quant_X, int64_t type01,
                                       int64_t col, at::ScalarType dtype);

torch::Tensor ggml_mul_mat_a8(torch::Tensor W, torch::Tensor X, int64_t type,
                              int64_t row);

torch::Tensor ggml_moe_a8(torch::Tensor X, torch::Tensor W,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t type,
                          int64_t row, int64_t top_k, int64_t tokens);

torch::Tensor ggml_moe_a8_vec(torch::Tensor X, torch::Tensor W,
                              torch::Tensor topk_ids, int64_t top_k,
                              int64_t type, int64_t row, int64_t tokens);

void ggml_moe_a8_vec_out(torch::Tensor X, torch::Tensor W,
                         torch::Tensor topk_ids, torch::Tensor output,
                         torch::Tensor quant_X, int64_t top_k, int64_t type,
                         int64_t row, int64_t tokens);

torch::Tensor ggml_moe_q8_vec(torch::Tensor quant_X, torch::Tensor W,
                              torch::Tensor topk_ids, int64_t top_k,
                              int64_t type, int64_t row, int64_t tokens,
                              int64_t col, at::ScalarType dtype);

torch::Tensor ggml_moe_q8_vec_weighted_sum(
    torch::Tensor quant_X, torch::Tensor W, torch::Tensor topk_ids,
    torch::Tensor topk_weights, int64_t top_k, int64_t type, int64_t row,
    int64_t tokens, int64_t col, at::ScalarType dtype);

void ggml_moe_q8_vec_weighted_sum_out(
    torch::Tensor quant_X, torch::Tensor W, torch::Tensor topk_ids,
    torch::Tensor topk_weights, torch::Tensor output, int64_t top_k,
    int64_t type, int64_t row, int64_t tokens, int64_t col);

void ggml_moe_a8_vec_silu_q8_weighted_sum_out(
    torch::Tensor X, torch::Tensor W1, torch::Tensor W2,
    torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor w1_output, torch::Tensor quant_X,
    torch::Tensor quant_w1_output, torch::Tensor output, int64_t top_k,
    int64_t w1_type, int64_t w2_type, int64_t w1_row, int64_t w2_row,
    int64_t tokens);

int64_t ggml_moe_get_block_size(int64_t type);
