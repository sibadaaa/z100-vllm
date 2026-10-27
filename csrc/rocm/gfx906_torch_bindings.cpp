#include "core/registration.h"
#include "rocm/gfx906_ops.h"

// gfx906 (Hygon Z100 / MI50-class) emulation ops.
// Registered into the legacy `_C` ROCm extension (non-stable torch::Tensor
// ABI). These used to live in csrc/torch_bindings.cpp + csrc/ops.h; the
// upstream sync moved those files into the STABLE_TORCH_LIBRARY extension,
// so they are kept in their own translation unit that only `_C` compiles.
TORCH_LIBRARY_EXPAND(_C, ops_gfx906) {
    ops_gfx906.def(
        "rms_norm_gated_gfx906(Tensor input, Tensor weight, Tensor gate, "
        "float epsilon, bool norm_before_gate) -> Tensor");

    ops_gfx906.impl("rms_norm_gated_gfx906", torch::kCUDA, &rms_norm_gated_gfx906);

    ops_gfx906.def(
        "gemma_rms_norm_gfx906(Tensor input, Tensor weight, float epsilon) -> "
        "Tensor");

    ops_gfx906.impl("gemma_rms_norm_gfx906", torch::kCUDA, &gemma_rms_norm_gfx906);

    ops_gfx906.def(
        "gemma_fused_add_rms_norm_gfx906(Tensor input, Tensor residual, Tensor "
        "weight, float epsilon) -> Tensor[]");

    ops_gfx906.impl("gemma_fused_add_rms_norm_gfx906", torch::kCUDA,
             &gemma_fused_add_rms_norm_gfx906);

    ops_gfx906.def(
        "causal_conv1d_gfx906_decode_update("
        "Tensor x, Tensor conv_state, Tensor weight, Tensor? bias, "
        "Tensor? conv_state_indices, int pad_slot_id, bool silu_activation) "
        "-> Tensor");

    ops_gfx906.impl("causal_conv1d_gfx906_decode_update", torch::kCUDA,
             &causal_conv1d_gfx906_decode_update);

    ops_gfx906.def(
        "causal_conv1d_gfx906_mtp_update("
        "Tensor x, Tensor conv_state, Tensor weight, Tensor? bias, "
        "Tensor state_indices, Tensor cu_seqlens, Tensor num_accepted_tokens, "
        "int pad_slot_id, bool silu_activation) -> Tensor");

    ops_gfx906.impl("causal_conv1d_gfx906_mtp_update", torch::kCUDA,
             &causal_conv1d_gfx906_mtp_update);

    ops_gfx906.def(
        "fused_sigmoid_gating_delta_rule_gfx906_decode("
        "Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k, "
        "Tensor v, Tensor state, float beta, float threshold, float scale, "
        "bool use_qk_l2norm_in_kernel) -> Tensor");

    ops_gfx906.impl("fused_sigmoid_gating_delta_rule_gfx906_decode", torch::kCUDA,
             &fused_sigmoid_gating_delta_rule_gfx906_decode);

    ops_gfx906.def(
        "fused_sigmoid_gating_delta_rule_gfx906_prefill("
        "Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k, "
        "Tensor v, Tensor state, Tensor cu_seqlens, float beta, "
        "float threshold, float scale, bool use_qk_l2norm_in_kernel) -> Tensor");

    ops_gfx906.impl("fused_sigmoid_gating_delta_rule_gfx906_prefill", torch::kCUDA,
             &fused_sigmoid_gating_delta_rule_gfx906_prefill);

    ops_gfx906.def(
        "fused_sigmoid_gating_delta_rule_gfx906_indexed_decode("
        "Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k, "
        "Tensor v, Tensor state, Tensor state_indices, float beta, "
        "float threshold, float scale, bool use_qk_l2norm_in_kernel) -> Tensor");

    ops_gfx906.impl("fused_sigmoid_gating_delta_rule_gfx906_indexed_decode", torch::kCUDA,
             &fused_sigmoid_gating_delta_rule_gfx906_indexed_decode);

    ops_gfx906.def(
        "fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state("
        "Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k, "
        "Tensor v, Tensor state, Tensor state_indices, float beta, "
        "float threshold, float scale, bool use_qk_l2norm_in_kernel) -> Tensor");

    ops_gfx906.impl("fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state",
             torch::kCUDA,
             &fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state);

    ops_gfx906.def(
        "fused_sigmoid_gating_delta_rule_gfx906_mtp_update("
        "Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, Tensor q, Tensor k, "
        "Tensor v, Tensor state, Tensor state_indices, Tensor cu_seqlens, "
        "Tensor num_accepted_tokens, float beta, float threshold, float scale, "
        "bool use_qk_l2norm_in_kernel) -> Tensor");

    ops_gfx906.impl("fused_sigmoid_gating_delta_rule_gfx906_mtp_update", torch::kCUDA,
             &fused_sigmoid_gating_delta_rule_gfx906_mtp_update);

    ops_gfx906.def(
        "fused_recurrent_gated_delta_rule_gfx906_packed_decode("
        "Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
        "Tensor state, Tensor out, Tensor state_indices, float scale, "
        "bool use_qk_l2norm_in_kernel, bool use_tiled_qk_head_mapping, "
        "bool use_transposed_state) -> Tensor");

    ops_gfx906.impl("fused_recurrent_gated_delta_rule_gfx906_packed_decode", torch::kCUDA,
             &fused_recurrent_gated_delta_rule_gfx906_packed_decode);

    ops_gfx906.def(
        "causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode("
        "Tensor mixed_qkv, Tensor conv_state, Tensor conv_weight, "
        "Tensor? conv_bias, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
        "Tensor state, Tensor out, Tensor state_indices, int pad_slot_id, "
        "float scale, bool silu_activation, bool use_qk_l2norm_in_kernel, "
        "bool use_tiled_qk_head_mapping, bool use_transposed_state) -> Tensor");

    ops_gfx906.impl(
        "causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode",
        torch::kCUDA,
        &causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode);

    ops_gfx906.def(
        "causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode("
        "Tensor mixed_qkv, Tensor conv_state, Tensor conv_weight, "
        "Tensor? conv_bias, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
        "Tensor state, Tensor out, Tensor state_indices, int pad_slot_id, "
        "float scale, bool silu_activation, bool use_qk_l2norm_in_kernel, "
        "bool use_tiled_qk_head_mapping, bool use_transposed_state) -> Tensor");

    ops_gfx906.impl(
        "causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode",
        torch::kCUDA,
        &causal_conv1d_recurrent_gated_delta_rule_gfx906_ratio2_packed_decode);
}
