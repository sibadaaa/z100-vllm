// Probe-only op registration for the stage-timed int8-KV attention kernel.
//
// Kept out of csrc/rocm/torch_bindings.cpp on purpose: the variant .so is a
// measurement artifact, never a production build.  TORCH_LIBRARY_EXPAND with
// the same TORCH_EXTENSION_NAME (_rocm_C) is legal from a second translation
// unit; the registered ops merge into the same library.
#include <torch/all.h>
#include <torch/library.h>

#include "core/registration.h"

void gfx906_attnd_decode_inner_kvint8_stage(
    torch::Tensor& segm_output, torch::Tensor& segm_max,
    torch::Tensor& segm_expsum, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, torch::Tensor& k_scale_cache,
    torch::Tensor& v_scale_cache, torch::Tensor& block_tables,
    torch::Tensor& seq_lens, torch::Tensor& cu_query_lens, double scale,
    int64_t num_segments, bool delayed_merge, int64_t tps_hint,
    int64_t seg_stride, torch::Tensor& stage_cycles);

// FRAGMENT (not TORCH_LIBRARY): the production bindings TU already owns
// the namespace in this .so, and only fragments may be repeated.
TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, attn_stage_ops) {
  attn_stage_ops.def(
      "gfx906_attnd_decode_inner_kvint8_stage(Tensor! segm_output, "
      "Tensor! segm_max, Tensor! segm_expsum, Tensor query, "
      "Tensor key_cache, Tensor value_cache, Tensor k_scale_cache, "
      "Tensor v_scale_cache, Tensor block_tables, Tensor seq_lens, "
      "Tensor cu_query_lens, float scale, int num_segments, "
      "bool delayed_merge, int tps_hint, int seg_stride, "
      "Tensor(a!) stage_cycles) -> ()");
  attn_stage_ops.impl("gfx906_attnd_decode_inner_kvint8_stage",
                      torch::kCUDA,
                      &gfx906_attnd_decode_inner_kvint8_stage);
}
