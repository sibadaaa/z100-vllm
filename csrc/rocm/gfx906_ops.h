#pragma once

#include <torch/all.h>

// gfx906 (Hygon Z100 / MI50-class) emulation kernels.
// Declarations for the ops registered by rocm/gfx906_torch_bindings.cpp.

torch::Tensor rms_norm_gated_gfx906(torch::Tensor input, torch::Tensor weight,
                                    torch::Tensor gate, double epsilon,
                                    bool norm_before_gate);

torch::Tensor gemma_rms_norm_gfx906(torch::Tensor input, torch::Tensor weight,
                                    double epsilon);

std::vector<torch::Tensor> gemma_fused_add_rms_norm_gfx906(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight,
    double epsilon);

torch::Tensor causal_conv1d_gfx906_decode_update(
    torch::Tensor x, torch::Tensor conv_state, torch::Tensor weight,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> conv_state_indices, int64_t pad_slot_id,
    bool silu_activation);

torch::Tensor causal_conv1d_gfx906_mtp_update(
    torch::Tensor x, torch::Tensor conv_state, torch::Tensor weight,
    std::optional<torch::Tensor> bias, torch::Tensor state_indices,
    torch::Tensor cu_seqlens, torch::Tensor num_accepted_tokens,
    int64_t pad_slot_id, bool silu_activation);

torch::Tensor fused_sigmoid_gating_delta_rule_gfx906_decode(
    torch::Tensor A_log, torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state,
    double beta, double threshold, double scale,
    bool use_qk_l2norm_in_kernel);

torch::Tensor fused_sigmoid_gating_delta_rule_gfx906_prefill(
    torch::Tensor A_log, torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state,
    torch::Tensor cu_seqlens, double beta, double threshold, double scale,
    bool use_qk_l2norm_in_kernel);

torch::Tensor fused_sigmoid_gating_delta_rule_gfx906_indexed_decode(
    torch::Tensor A_log, torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state,
    torch::Tensor state_indices, double beta, double threshold, double scale,
    bool use_qk_l2norm_in_kernel);

torch::Tensor fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state(
    torch::Tensor A_log, torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state,
    torch::Tensor state_indices, double beta, double threshold, double scale,
    bool use_qk_l2norm_in_kernel);

torch::Tensor fused_sigmoid_gating_delta_rule_gfx906_mtp_update(
    torch::Tensor A_log, torch::Tensor a, torch::Tensor b,
    torch::Tensor dt_bias, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor state, torch::Tensor state_indices,
    torch::Tensor cu_seqlens, torch::Tensor num_accepted_tokens, double beta,
    double threshold, double scale, bool use_qk_l2norm_in_kernel);

torch::Tensor fused_recurrent_gated_delta_rule_gfx906_packed_decode(
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b,
    torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state,
    torch::Tensor out, torch::Tensor state_indices, double scale,
    bool use_qk_l2norm_in_kernel, bool use_tiled_qk_head_mapping,
    bool use_transposed_state);

torch::Tensor causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode(
    torch::Tensor mixed_qkv, torch::Tensor conv_state,
    torch::Tensor conv_weight, std::optional<torch::Tensor> conv_bias,
    torch::Tensor a, torch::Tensor b, torch::Tensor A_log,
    torch::Tensor dt_bias, torch::Tensor state, torch::Tensor out,
    torch::Tensor state_indices, int64_t pad_slot_id, double scale,
    bool silu_activation, bool use_qk_l2norm_in_kernel,
    bool use_tiled_qk_head_mapping, bool use_transposed_state);

torch::Tensor
causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode(
    torch::Tensor mixed_qkv, torch::Tensor conv_state,
    torch::Tensor conv_weight, std::optional<torch::Tensor> conv_bias,
    torch::Tensor a, torch::Tensor b, torch::Tensor A_log,
    torch::Tensor dt_bias, torch::Tensor state, torch::Tensor out,
    torch::Tensor state_indices, int64_t pad_slot_id, double scale,
    bool silu_activation, bool use_qk_l2norm_in_kernel,
    bool use_tiled_qk_head_mapping, bool use_transposed_state);
