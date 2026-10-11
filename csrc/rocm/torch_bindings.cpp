#include "core/registration.h"
#include "rocm/ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops) {
  // vLLM custom ops for rocm

// skinny_gemms.cu (LLMM1/wvSplitK/wvSplitKrc/wvSplitKQ) is excluded on gfx1250
// (gfx9/gfx11 ISA, unsupported there); skip these registrations to avoid
// undefined symbols. vLLM uses default/Triton GEMM for these ops on gfx1250.
#ifndef VLLM_SKIP_SKINNY_GEMMS
  // Custom gemm op for matrix-vector multiplication
  rocm_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> "
      "Tensor");
  rocm_ops.impl("LLMM1", torch::kCUDA, &LLMM1);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitK(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitK", torch::kCUDA, &wvSplitK);

  // W4A16 grouped skinny GEMM: packed int4 weights, per-group scales,
  // optional zero points [M/8, K/group_size] int32 for asymmetric
  // quantization
  rocm_ops.def(
      "wvSplitK_int4_g(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "Tensor? in_zero_points, Tensor? in_bias, int CuCount, "
      "int group_size) -> Tensor");
  rocm_ops.impl("wvSplitK_int4_g", torch::kCUDA, &wvSplitK_int4_g);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitKrc(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitKrc", torch::kCUDA, &wvSplitKrc);

  // wvSplitK for fp8
  rocm_ops.def(
      "wvSplitKQ(Tensor in_a, Tensor in_b, Tensor? in_bias, Tensor! out_c, "
      "Tensor scale_a, "
      "          Tensor scale_b, int CuCount) -> ()");
  rocm_ops.impl("wvSplitKQ", torch::kCUDA, &wvSplitKQ);
#endif  // VLLM_SKIP_SKINNY_GEMMS

#ifdef VLLM_ROCM_GFX1100
  // W4A16 GPTQ kernels for AMD RDNA3 (gfx1100).
  rocm_ops.def(
      "gptq_gemm_rdna3(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3", torch::kCUDA, &gptq_gemm_rdna3);

  rocm_ops.def(
      "gptq_gemm_rdna3_wmma(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3_wmma", torch::kCUDA, &gptq_gemm_rdna3_wmma);

  rocm_ops.def(
      "moe_gptq_gemm_rdna3(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor b_scales, Tensor b_qzeros, Tensor topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk) -> ()");
  rocm_ops.impl("moe_gptq_gemm_rdna3", torch::kCUDA, &moe_gptq_gemm_rdna3);
#endif

  // Custom attention op
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,"
      "                Tensor max_logits, Tensor tmp_out,"
      "                Tensor query, Tensor key_cache,"
      "                Tensor value_cache, int num_kv_heads,"
      "                float scale, Tensor block_tables,"
      "                Tensor seq_lens,"
      "                Tensor? query_start_loc,"
      "                int block_size,"
      "                int max_seq_len,"
      "                Tensor? alibi_slopes,"
      "                str kv_cache_dtype,"
      "                Tensor k_scale, Tensor v_scale,"
      "                Tensor? fp8_out_scale,"
      "                str mfma_type) -> ()");
  rocm_ops.impl("paged_attention", torch::kCUDA, &paged_attention);

  // W8A8 INT8 GEMM/GEMV on gfx906's `V_DOT4_I32_I8` (DL-Ops).
  // Weights stay [N, K] int8 (never transposed, never dequantized); the
  // caller supplies per-token x_s and per-channel w_s.  The Python-side
  // Gfx906Sdot4Int8ScaledMMLinearKernel is the only intended caller.
  rocm_ops.def(
      "w8a8_sdot4_gemv(Tensor xq, Tensor w, Tensor x_s, Tensor w_s, "
      "Tensor? bias, ScalarType out_dtype, int nthreads) -> Tensor");
  rocm_ops.impl("w8a8_sdot4_gemv", torch::kCUDA, &w8a8_sdot4_gemv);

  rocm_ops.def(
      "w8a8_sdot4_gemm(Tensor xq, Tensor w, Tensor x_s, Tensor w_s, "
      "Tensor? bias, ScalarType out_dtype, int cfg) -> Tensor");
  rocm_ops.impl("w8a8_sdot4_gemm", torch::kCUDA, &w8a8_sdot4_gemm);

  // W8A8 FP8 GEMM/GEMV on gfx906, emulated as fp8 --(256-entry LDS LUT)-->
  // fp16 with `v_dot2_f32_f16` + fp32 accumulation.  Weights stay fp8 in
  // their [N, K] checkpoint layout; nothing is dequantized or copied.
  // The Python-side Gfx906Fp8ScaledMMLinearKernel is the only intended
  // caller (see vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py).
  rocm_ops.def(
      "fp8_emu_gemv(Tensor a, Tensor w, Tensor a_s, Tensor w_s, Tensor lut, "
      "Tensor? bias, ScalarType out_dtype, int nthreads) -> Tensor");
  rocm_ops.impl("fp8_emu_gemv", torch::kCUDA, &fp8_emu_gemv);

  rocm_ops.def(
      "fp8_emu_gemm(Tensor a, Tensor w, Tensor a_s, Tensor w_s, Tensor lut, "
      "Tensor? bias, ScalarType out_dtype, int cfg) -> Tensor");
  rocm_ops.impl("fp8_emu_gemm", torch::kCUDA, &fp8_emu_gemm);

  // Same kernels for FP8 block-scaled checkpoints: `a_s` is
  // [M, cdiv(K, block_k)] and `w_s` is [cdiv(N, block_n), cdiv(K, block_k)].
  // The Python-side Gfx906Fp8BlockScaledMMLinearKernel is the only intended
  // caller (see vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py).
  rocm_ops.def(
      "fp8_emu_gemv_block(Tensor a, Tensor w, Tensor a_s, Tensor w_s, "
      "Tensor lut, Tensor? bias, ScalarType out_dtype, int nthreads, "
      "int block_n, int block_k) -> Tensor");
  rocm_ops.impl("fp8_emu_gemv_block", torch::kCUDA, &fp8_emu_gemv_block);

  rocm_ops.def(
      "fp8_emu_gemm_block(Tensor a, Tensor w, Tensor a_s, Tensor w_s, "
      "Tensor lut, Tensor? bias, ScalarType out_dtype, int cfg, "
      "int block_n, int block_k) -> Tensor");
  rocm_ops.impl("fp8_emu_gemm_block", torch::kCUDA, &fp8_emu_gemm_block);

  // Dense fp16/bf16 decode GEMV (M <= 8) for gfx906: the replacement for the
  // rocBLAS fallback on unquantized linear layers, used by
  // vllm/model_executor/layers/utils.py (rocm_unquantized_gemm_impl) behind
  // VLLM_DCU_USE_BF16_GEMV=1.
  rocm_ops.def(
      "bf16_emu_gemv(Tensor x, Tensor w, Tensor? bias, int nthreads, "
      "int route, bool range_safe) -> Tensor");
  rocm_ops.impl("bf16_emu_gemv", torch::kCUDA, &bf16_emu_gemv);
}

// Meta implementations: `torch.compile` / fake tensor propagation needs the
// output shape without running the kernel.  Registered from the same TU so
// the ops can be traced instead of forcing a graph break.
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Meta, rocm_meta_ops) {
  rocm_meta_ops.impl("w8a8_sdot4_gemv", &w8a8_sdot4_gemv_meta);
  rocm_meta_ops.impl("w8a8_sdot4_gemm", &w8a8_sdot4_gemm_meta);
  rocm_meta_ops.impl("fp8_emu_gemv", &fp8_emu_gemv_meta);
  rocm_meta_ops.impl("fp8_emu_gemm", &fp8_emu_gemm_meta);
  rocm_meta_ops.impl("fp8_emu_gemv_block", &fp8_emu_gemv_block_meta);
  rocm_meta_ops.impl("fp8_emu_gemm_block", &fp8_emu_gemm_block_meta);
  rocm_meta_ops.impl("bf16_emu_gemv", &bf16_emu_gemv_meta);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
