#include "core/registration.h"
#include "quantization/gguf/gguf_ops.h"

// GGUF quantization ops, registered into the legacy `_C` ROCm extension.
// Upstream removed in-tree GGUF C++ (vllm-gguf-plugin); the fork keeps it.
TORCH_LIBRARY_FRAGMENT(_C, ops_gguf) {
    ops_gguf.def(
        "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
        "dtype) -> Tensor");

    ops_gguf.impl("ggml_dequantize", torch::kCUDA, &ggml_dequantize);

    ops_gguf.def(
        "ggml_repack_iq4_xs_to_q8_0(Tensor W, SymInt row, SymInt col) -> Tensor");

    ops_gguf.impl("ggml_repack_iq4_xs_to_q8_0", torch::kCUDA,
             &ggml_repack_iq4_xs_to_q8_0);

    ops_gguf.def(
        "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) "
        "-> Tensor");

    ops_gguf.impl("ggml_mul_mat_vec_a8", torch::kCUDA, &ggml_mul_mat_vec_a8);

    ops_gguf.def("ggml_quantize_row_q8_1(Tensor X) -> Tensor");

    ops_gguf.impl("ggml_quantize_row_q8_1", torch::kCUDA, &ggml_quantize_row_q8_1);

    ops_gguf.def("ggml_quantize_row_q8_1_out(Tensor X, Tensor! quant_X) -> ()");

    ops_gguf.impl("ggml_quantize_row_q8_1_out", torch::kCUDA,
             &ggml_quantize_row_q8_1_out);

    ops_gguf.def("ggml_silu_and_mul_quantize_row_q8_1(Tensor X) -> Tensor");

    ops_gguf.impl("ggml_silu_and_mul_quantize_row_q8_1", torch::kCUDA,
             &ggml_silu_and_mul_quantize_row_q8_1);

    ops_gguf.def(
        "ggml_silu_and_mul_quantize_row_q8_1_out(Tensor X, Tensor! quant_X) -> ()");

    ops_gguf.impl("ggml_silu_and_mul_quantize_row_q8_1_out", torch::kCUDA,
             &ggml_silu_and_mul_quantize_row_q8_1_out);

    ops_gguf.def(
        "ggml_sigmoid_and_mul_quantize_row_q8_1(Tensor X, Tensor gate) -> Tensor");

    ops_gguf.impl("ggml_sigmoid_and_mul_quantize_row_q8_1", torch::kCUDA,
             &ggml_sigmoid_and_mul_quantize_row_q8_1);

    ops_gguf.def(
        "ggml_sigmoid_and_mul_quantize_row_q8_1_out(Tensor X, Tensor gate, "
        "Tensor! quant_X) -> ()");

    ops_gguf.impl("ggml_sigmoid_and_mul_quantize_row_q8_1_out", torch::kCUDA,
             &ggml_sigmoid_and_mul_quantize_row_q8_1_out);

    ops_gguf.def(
        "ggml_rms_norm_gated_quantize_row_q8_1(Tensor X, Tensor weight, "
        "Tensor gate, float epsilon, bool norm_before_gate) -> Tensor");

    ops_gguf.impl("ggml_rms_norm_gated_quantize_row_q8_1", torch::kCUDA,
             &ggml_rms_norm_gated_quantize_row_q8_1);

    ops_gguf.def(
        "ggml_rms_norm_gated_quantize_row_q8_1_out(Tensor X, Tensor weight, "
        "Tensor gate, Tensor! quant_X, float epsilon, bool norm_before_gate) -> ()");

    ops_gguf.impl("ggml_rms_norm_gated_quantize_row_q8_1_out", torch::kCUDA,
             &ggml_rms_norm_gated_quantize_row_q8_1_out);

    ops_gguf.def(
        "ggml_mul_mat_vec_q8(Tensor W, Tensor quant_X, int type, SymInt row, "
        "SymInt col, ScalarType dtype) -> Tensor");

    ops_gguf.impl("ggml_mul_mat_vec_q8", torch::kCUDA, &ggml_mul_mat_vec_q8);

    ops_gguf.def(
        "ggml_mul_mat_vec_q8_out(Tensor W, Tensor quant_X, Tensor! Y, int type, "
        "SymInt row, SymInt col) -> ()");

    ops_gguf.impl("ggml_mul_mat_vec_q8_out", torch::kCUDA, &ggml_mul_mat_vec_q8_out);

    ops_gguf.def(
        "ggml_mul_mat_vec_q8_0_fast(Tensor W, Tensor quant_X, SymInt row, "
        "SymInt col, ScalarType dtype) -> Tensor");

    ops_gguf.impl("ggml_mul_mat_vec_q8_0_fast", torch::kCUDA,
             &ggml_mul_mat_vec_q8_0_fast);

    ops_gguf.def(
        "ggml_mul_mat_vec_a8_sharded(Tensor[] W, Tensor X, int[] types) "
        "-> Tensor");

    ops_gguf.impl("ggml_mul_mat_vec_a8_sharded", torch::kCUDA,
             &ggml_mul_mat_vec_a8_sharded);

    ops_gguf.def(
        "ggml_mul_mat_vec_q8_sharded(Tensor[] W, Tensor quant_X, int[] types, "
        "SymInt col, ScalarType dtype) -> Tensor");

    ops_gguf.impl("ggml_mul_mat_vec_q8_sharded", torch::kCUDA,
             &ggml_mul_mat_vec_q8_sharded);

    ops_gguf.def(
        "ggml_mul_mat_vec_q8_sharded_out(Tensor[] W, Tensor quant_X, Tensor! Y, "
        "int[] types, SymInt col) -> ()");

    ops_gguf.impl("ggml_mul_mat_vec_q8_sharded_out", torch::kCUDA,
             &ggml_mul_mat_vec_q8_sharded_out);

    ops_gguf.def(
        "ggml_mul_mat_vec_q8_grouped_same_type(Tensor[] W, Tensor quant_X, "
        "int type, SymInt col, ScalarType dtype) -> Tensor");

    ops_gguf.impl("ggml_mul_mat_vec_q8_grouped_same_type", torch::kCUDA,
             &ggml_mul_mat_vec_q8_grouped_same_type);

    ops_gguf.def(
        "ggml_mul_mat_vec_q8_grouped_same_type_out(Tensor[] W, Tensor quant_X, "
        "Tensor! Y, int type, SymInt col) -> ()");

    ops_gguf.impl("ggml_mul_mat_vec_q8_grouped_same_type_out", torch::kCUDA,
             &ggml_mul_mat_vec_q8_grouped_same_type_out);

    ops_gguf.def(
        "ggml_mul_mat_vec_q8_qkv3(Tensor W0, Tensor W1, Tensor W2, "
        "Tensor quant_X, int type01, SymInt col, ScalarType dtype) -> Tensor");

    ops_gguf.impl("ggml_mul_mat_vec_q8_qkv3", torch::kCUDA,
             &ggml_mul_mat_vec_q8_qkv3);

    ops_gguf.def(
        "ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");

    ops_gguf.impl("ggml_mul_mat_a8", torch::kCUDA, &ggml_mul_mat_a8);

    ops_gguf.def(
        "ggml_moe_a8(Tensor X, Tensor W, "
        "Tensor sorted_token_ids, Tensor expert_ids, Tensor "
        "num_tokens_post_padded, "
        "int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor");

    ops_gguf.impl("ggml_moe_a8", torch::kCUDA, &ggml_moe_a8);

    ops_gguf.def(
        "ggml_moe_a8_vec(Tensor X, Tensor W, "
        "Tensor topk_ids, int top_k, "
        "int type, SymInt row, SymInt tokens) -> Tensor");

    ops_gguf.impl("ggml_moe_a8_vec", torch::kCUDA, &ggml_moe_a8_vec);

    ops_gguf.def(
        "ggml_moe_a8_vec_out(Tensor X, Tensor W, Tensor topk_ids, "
        "Tensor! output, Tensor! quant_X, int top_k, int type, SymInt row, "
        "SymInt tokens) -> ()");

    ops_gguf.impl("ggml_moe_a8_vec_out", torch::kCUDA, &ggml_moe_a8_vec_out);

    ops_gguf.def(
        "ggml_moe_q8_vec(Tensor quant_X, Tensor W, "
        "Tensor topk_ids, int top_k, "
        "int type, SymInt row, SymInt tokens, SymInt col, ScalarType dtype) "
        "-> Tensor");

    ops_gguf.impl("ggml_moe_q8_vec", torch::kCUDA, &ggml_moe_q8_vec);

    ops_gguf.def(
        "ggml_moe_q8_vec_weighted_sum(Tensor quant_X, Tensor W, "
        "Tensor topk_ids, Tensor topk_weights, int top_k, "
        "int type, SymInt row, SymInt tokens, SymInt col, ScalarType dtype) "
        "-> Tensor");

    ops_gguf.impl("ggml_moe_q8_vec_weighted_sum", torch::kCUDA,
             &ggml_moe_q8_vec_weighted_sum);

    ops_gguf.def(
        "ggml_moe_q8_vec_weighted_sum_out(Tensor quant_X, Tensor W, "
        "Tensor topk_ids, Tensor topk_weights, Tensor! output, int top_k, "
        "int type, SymInt row, SymInt tokens, SymInt col) -> ()");

    ops_gguf.impl("ggml_moe_q8_vec_weighted_sum_out", torch::kCUDA,
             &ggml_moe_q8_vec_weighted_sum_out);

    ops_gguf.def(
        "ggml_moe_a8_vec_silu_q8_weighted_sum_out("
        "Tensor X, Tensor W1, Tensor W2, Tensor topk_ids, Tensor topk_weights, "
        "Tensor! w1_output, Tensor! quant_X, Tensor! quant_w1_output, "
        "Tensor! output, int top_k, int w1_type, int w2_type, SymInt w1_row, "
        "SymInt w2_row, SymInt tokens) -> ()");

    ops_gguf.impl("ggml_moe_a8_vec_silu_q8_weighted_sum_out", torch::kCUDA,
             &ggml_moe_a8_vec_silu_q8_weighted_sum_out);

    ops_gguf.def("ggml_moe_get_block_size", &ggml_moe_get_block_size);
}
