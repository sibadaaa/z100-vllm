// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// S53-ATTN: gfx906 decode split-KV attention, inner-loop rebuild (HIP).
//
// Design: docs/design/z100/s53/z100-s53-attn-inner-design.md
// Construction: docs/design/z100/s53/z100-s53-attn-inner-construction.md
//
// The in-service Triton kernel (kernel_unified_attention_3d) runs the tuned
// decode shape at ~60 GB/s dedup @109k while the same page walk without the
// dot path reaches 673-686 GB/s online (WP4b twin probe). WP0 attribution
// (.s53/h-attrib.md) convicts Triton's per-tile dot operand conversion
// through LDS. This kernel keeps the identical launch contract (same
// block-table walk, same segm_output/max/expsum buffers; reduce_segments
// reused unchanged) and rebuilds the inner loop around gfx906's
// v_dot2_f32_f16:
//
//   * v7 (dim-split half-tile ping-pong): 16 KB LDS -> 4 CTA/CU, and the
//     fetch stream is CONTINUOUS (half-tile ping-pong: compute buf[h&1]
//     while buf[(h+1)&1] streams in; one barrier per half). All 256 lanes
//     are active every half: the 256-dim QK^T dot is split 128/128 across
//     the dh lane pair and completed with a single xor exchange; PV splits
//     the 8-token loop across the same pair (even/odd), with one 16-float
//     pair merge per half. Measured negatives that shaped this form are
//     listed in .s53/i-inner-final.md (per-lane direct loads, merged-q,
//     divergent half-tile, GLC, full-tile double buffer).
//   * softmax: per-half row max via wavefront XOR butterfly; expsum per
//     lane (kDelayMerge) or mirrored per half; segment-end scale
//     unification (exp(m_run - m*)) + butterfly merge. Accumulation order
//     differs from the Triton kernel by design; the acceptance gate is
//     |d| <= 0.011 vs SDPA fp32 (s53 stage-A precedent).
//   * PV: fp32 FMA against the lane's 16-dim output shard (strictly more
//     precise than the fp16 P dot it replaces).
//   * S53-ATTN-B832 (production page shape, user adjudication B of the fp8
//     close-out, .s53/fp8-final.md): the page walk is generalized from
//     page==tile to any block_size multiple of 16 via an incremental cursor
//     (off += 16, wrap == block_size advances one block-table entry), so the
//     hybrid-cache production page (832 tokens, K|V packed in one allocation,
//     token stride 512) walks the same tile stream without per-tile division.
//     Descale folding: k_descale[kvh] folds into the softmax scale and
//     v_descale[kvh] into the segment-end accumulator - both exact scalars
//     commuting with the merge (identical arithmetic when null or 1.0).
//   * Supported subset enforced here and by the Python dispatch: num_seqs 1,
//     causal, fp16 Q/K/V, head 256, block_size % 16 == 0, BLOCK_M 16 /
//     BLOCK_Q 2 / GQA6, 16-byte-aligned KV strides, no sinks/alibi/softcap/
//     qq-bias/sliding-window/mm-prefix/fp8-KV. Everything else stays on the
//     Triton path.

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
// This file is shipped as .hip and compiled directly (it bypasses the build's
// CUDA->HIP hipify pass), so the HIP-native torch headers are used here; the
// at::cuda::* stream/guard spellings below are the HIP-build compat names the
// rest of this extension's hipified sources use.
#include "core/hip_compat.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "rocm/gfx906_emu_common.cuh"

namespace {

constexpr int kHead = 256;            // head_size (padded == actual)
constexpr int kTile = 16;             // == block_size
constexpr int kBlockM = 16;           // (query token, head) rows per CTA
constexpr int kNpkv = 6;              // query heads per kv head (GQA6)
constexpr int kThreads = 256;         // 4 wavefronts x 64
constexpr int kRowUnits = kHead / 8;  // 32 uint4 units per head row

// One fp16-pair (two packed halves in a dword) dot2 step, fp32 accumulate.
using vllm::gfx906_emu::dot2;

__device__ __forceinline__ uint4 zero_uint4() { return make_uint4(0u, 0u, 0u, 0u); }

// LDS slot permutation: bijective in u within a row r. Keeps the V shard
// read (fixed unit, varying row - the column-ish access) at the 2-way b128
// floor instead of 4-way, without any padding.
__device__ __forceinline__ int lds_f0(int u) {
    return ((u & 1) << 4) + (u >> 1);
}
__device__ __forceinline__ int lds_slot(int r, int u) {
    return r * kRowUnits + ((lds_f0(u) + r) & (kRowUnits - 1));
}

template <bool kDelayMerge, int kTPS>
__global__ void __launch_bounds__(kThreads, 2)
attnd_inner_decode_kernel(
    void* __restrict__ segm_output,         // [rows, NQ, NSEG, kHead] fp32/fp16
    float* __restrict__ segm_max,           // [rows, NQ, NSEG] fp32
    float* __restrict__ segm_expsum,        // [rows, NQ, NSEG] fp32
    const uint32_t* __restrict__ query,     // [qlen, NQ, kHead] fp16 pairs
    const uint32_t* __restrict__ key_cache,   // [blks, 16, NKV, kHead] pairs
    const uint32_t* __restrict__ value_cache,
    const int* __restrict__ block_tables,   // [1, W] page ids
    const int* __restrict__ seq_lens,       // [1]
    const int* __restrict__ cu_query_lens,  // [2] = (0, qlen)
    const float scale,
    const int num_segments,
    const int num_query_heads,
    const int kv_page_stride,               // fp16 elements per page
    const int kv_token_stride,              // fp16 elements per token
    const int kv_head_stride,               // fp16 elements per kv head
    const int kv_block_size,                // tokens per page (multiple of 16)
    const float* __restrict__ k_descale,    // per-kvh via +kds, or null
    const int k_descale_stride,             // float elements per kv head
    const float* __restrict__ v_descale,    // per-kvh via +vds, or null
    const int v_descale_stride,             // float elements per kv head
    const int segm_fp16) {                  // segm_output is fp16 (S56 WP2)
  const int kvh = blockIdx.y;
  const int seg = blockIdx.z;
  const int q_len = cu_query_lens[1] - cu_query_lens[0];
  const int seq_len = seq_lens[0];
  const int ctx = seq_len - q_len;

  // Segment split identical to kernel_unified_attention_3d. kTPS > 0 pins
  // the trip count at compile time; 0 falls back to the runtime value.
  const int tiles_per_seg =
      kTPS ? kTPS
           : (seq_len + num_segments * kTile - 1) / (num_segments * kTile);
  if (seg * tiles_per_seg * kTile >= seq_len) return;

  const int tid = threadIdx.x;
  const int t = tid & 15;          // token slot within the tile
  // S58 follow-up: a CTA owns 16 CONSECUTIVE (token, head) pairs.  The old
  // map (qpos = blockIdx.x*2 + m/6) made the m=12..15 rows land on the next
  // qpos and repeat the next CTA's first 4 rows: 59,900 launched rows vs
  // 44,928 needed (1.333x wasted compute, benign duplicate stores).
  const int pair = blockIdx.x * kBlockM + (tid >> 4);
  const int qpos = pair / kNpkv;
  const bool row_valid = qpos < q_len;
  const int token_abs = qpos < q_len ? qpos : q_len - 1;  // clamp OOB reads
  const int head = kvh * kNpkv + (pair % kNpkv);
  const int bound = ctx + qpos;

  // ---- LDS staging: FULL-TILE DOUBLE BUFFER (32 KB -> 2 CTA/CU) ----
  // Compute tile j from buf[j&1] while tile j+1 streams into buf[(j+1)&1]:
  // the fetch stream is continuous and the write/read sides never touch the
  // same buffer, so ONE barrier per tile suffices. Unit (row r, 16-B unit
  // cu) lives at r*32 + ((cu + r) & 31).
  __shared__ uint4 k_lds[2 * kTile * kRowUnits];
  __shared__ uint4 v_lds[2 * kTile * kRowUnits];
  const uint4 zero4 = zero_uint4();
  const int st_row = tid >> 3;         // 32 tokens x 4 uint4 per lane
  const int st_cu0 = (tid & 7) * 4;
  const int st_half = st_row >> 4;
  const int st_rr = st_row & 15;

  const int j_lo = seg * tiles_per_seg;
  const int j_hi = min((seg + 1) * tiles_per_seg,
                       (seq_len + kTile - 1) / kTile);
  // S77 T1: 32 keys per iteration, one Q read each. See wp1 台账.

  float m_run = -INFINITY;
  float l_run = 0.0f;
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

  const uint4* q4 = reinterpret_cast<const uint4*>(
      query + ((size_t)token_abs * num_query_heads + head) * (kHead / 2));

  const int tpp = kv_block_size >> 4;
  const float scale_eff =
      scale * (k_descale ? k_descale[kvh * k_descale_stride] : 1.0f);

  // Two-half page cursor, kept in tokens (matches the per-tile walk).
  int page_idx = j_lo / tpp;
  int off_a = (j_lo - page_idx * tpp) * kTile;
  int page_a = block_tables[page_idx];
  int off_b = off_a + kTile;
  int page_b = page_a;
  if (off_b >= kv_block_size) {
    off_b -= kv_block_size;
    if (j_lo + 1 < j_hi) page_b = block_tables[page_idx + 1];
  }

  uint4 kreg0, kreg1, kreg2, kreg3, vreg0, vreg1, vreg2, vreg3;

  // stage the first group of 32 tokens
  {
    const int pg = st_half ? page_b : page_a;
    const int of = st_half ? off_b : off_a;
    const bool ok = j_lo * kTile + st_row < seq_len;
    const uint4* ksrc = reinterpret_cast<const uint4*>(
        key_cache + ((size_t)pg * kv_page_stride +
                     (size_t)kvh * kv_head_stride +
                     (size_t)(of + st_rr) * kv_token_stride) / 2);
    const uint4* vsrc = reinterpret_cast<const uint4*>(
        value_cache + ((size_t)pg * kv_page_stride +
                       (size_t)kvh * kv_head_stride +
                       (size_t)(of + st_rr) * kv_token_stride) / 2);
    const int r = st_row * kRowUnits;
    kreg0 = ok ? ksrc[st_cu0 + 0] : zero4;
    kreg1 = ok ? ksrc[st_cu0 + 1] : zero4;
    kreg2 = ok ? ksrc[st_cu0 + 2] : zero4;
    kreg3 = ok ? ksrc[st_cu0 + 3] : zero4;
    vreg0 = ok ? vsrc[st_cu0 + 0] : zero4;
    vreg1 = ok ? vsrc[st_cu0 + 1] : zero4;
    vreg2 = ok ? vsrc[st_cu0 + 2] : zero4;
    vreg3 = ok ? vsrc[st_cu0 + 3] : zero4;
    k_lds[r + ((lds_f0(st_cu0 + 0) + st_row) & (kRowUnits - 1))] = kreg0;
    k_lds[r + ((lds_f0(st_cu0 + 1) + st_row) & (kRowUnits - 1))] = kreg1;
    k_lds[r + ((lds_f0(st_cu0 + 2) + st_row) & (kRowUnits - 1))] = kreg2;
    k_lds[r + ((lds_f0(st_cu0 + 3) + st_row) & (kRowUnits - 1))] = kreg3;
    v_lds[r + ((lds_f0(st_cu0 + 0) + st_row) & (kRowUnits - 1))] = vreg0;
    v_lds[r + ((lds_f0(st_cu0 + 1) + st_row) & (kRowUnits - 1))] = vreg1;
    v_lds[r + ((lds_f0(st_cu0 + 2) + st_row) & (kRowUnits - 1))] = vreg2;
    v_lds[r + ((lds_f0(st_cu0 + 3) + st_row) & (kRowUnits - 1))] = vreg3;
    __syncthreads();
  }

  for (int j = j_lo; j < j_hi; j += 2) {
    const bool has_h1 = (j + 1 < j_hi);
    const int jn = j + 2;
    if (jn < j_hi) {
      off_a += 2 * kTile;
      if (off_a >= kv_block_size) {
        off_a -= kv_block_size;
        ++page_idx;
        page_a = block_tables[page_idx];
      }
      off_b = off_a + kTile;
      page_b = page_a;
      if (off_b >= kv_block_size) {
        off_b -= kv_block_size;
        page_b = block_tables[page_idx + 1];
      }
    }

    // ---- one Q read feeds two independent score chains ----
    const int r0 = t;
    const int r1 = kTile + t;
    const uint4* krow0 = k_lds + r0 * kRowUnits;
    const uint4* krow1 = k_lds + r1 * kRowUnits;
    float sa = 0.0f, sb = 0.0f;
#pragma unroll 4
    for (int c = 0; c < kRowUnits; ++c) {
      const uint4 qv = q4[c];
      {
        const uint4 kv = krow0[(lds_f0(c) + r0) & (kRowUnits - 1)];
        sa = dot2(qv.x, kv.x, sa);
        sa = dot2(qv.y, kv.y, sa);
        sa = dot2(qv.z, kv.z, sa);
        sa = dot2(qv.w, kv.w, sa);
      }
      {
        const uint4 kv = krow1[(lds_f0(c) + r1) & (kRowUnits - 1)];
        sb = dot2(qv.x, kv.x, sb);
        sb = dot2(qv.y, kv.y, sb);
        sb = dot2(qv.z, kv.z, sb);
        sb = dot2(qv.w, kv.w, sb);
      }
    }

    const int gbase = tid & ~(16 - 1);
#define S77_HALF(H, S_IN, RV0)                                                \
  {                                                                           \
    const int token_base = (j + (H)) * kTile;                                  \
    float sh = (S_IN) * scale_eff;                                            \
    if (!row_valid || token_base + t > bound) sh = -INFINITY;                  \
    float mt = sh;                                                            \
    _Pragma("unroll") for (int off = 1; off < 16; off <<= 1) mt =              \
        fmaxf(mt, __shfl_xor(mt, off, 16));                                    \
    const float mn = fmaxf(m_run, mt) > -INFINITY ? fmaxf(m_run, mt) : 0.0f;    \
    const float alpha = __expf(m_run - mn);                                    \
    const float p = __expf(sh - mn);                                           \
    m_run = mn;                                                               \
    if (kDelayMerge) {                                                        \
      l_run = l_run * alpha + p;                                              \
    } else {                                                                  \
      float lj = p;                                                           \
      _Pragma("unroll") for (int off = 1; off < 16; off <<= 1) lj =            \
          lj + __shfl_xor(lj, off, 16);                                        \
      l_run = l_run * alpha + lj;                                             \
    }                                                                         \
    _Pragma("unroll") for (int i = 0; i < 16; ++i) acc[i] *= alpha;            \
    float pr[16];                                                             \
    _Pragma("unroll") for (int j2 = 0; j2 < kTile; ++j2) pr[j2] =              \
        __shfl(p, gbase | j2, 16);                                            \
    _Pragma("unroll 2") for (int j2 = 0; j2 < kTile; ++j2) {                   \
      const int rv = (RV0) + j2;                                              \
      const int bv = rv * kRowUnits;                                          \
      const uint4 va = v_lds[bv + ((lds_f0(t * 2) + rv) & (kRowUnits - 1))];   \
      const uint4 vb =                                                        \
          v_lds[bv + ((lds_f0(t * 2 + 1) + rv) & (kRowUnits - 1))];            \
      const float w = pr[j2];                                                 \
      float2 f;                                                               \
      f = __half22float2(*reinterpret_cast<const __half2*>(&va.x));            \
      acc[0] = fmaf(w, f.x, acc[0]);                                          \
      acc[1] = fmaf(w, f.y, acc[1]);                                          \
      f = __half22float2(*reinterpret_cast<const __half2*>(&va.y));            \
      acc[2] = fmaf(w, f.x, acc[2]);                                          \
      acc[3] = fmaf(w, f.y, acc[3]);                                          \
      f = __half22float2(*reinterpret_cast<const __half2*>(&va.z));            \
      acc[4] = fmaf(w, f.x, acc[4]);                                          \
      acc[5] = fmaf(w, f.y, acc[5]);                                          \
      f = __half22float2(*reinterpret_cast<const __half2*>(&va.w));            \
      acc[6] = fmaf(w, f.x, acc[6]);                                          \
      acc[7] = fmaf(w, f.y, acc[7]);                                          \
      f = __half22float2(*reinterpret_cast<const __half2*>(&vb.x));            \
      acc[8] = fmaf(w, f.x, acc[8]);                                          \
      acc[9] = fmaf(w, f.y, acc[9]);                                          \
      f = __half22float2(*reinterpret_cast<const __half2*>(&vb.y));            \
      acc[10] = fmaf(w, f.x, acc[10]);                                        \
      acc[11] = fmaf(w, f.y, acc[11]);                                        \
      f = __half22float2(*reinterpret_cast<const __half2*>(&vb.z));            \
      acc[12] = fmaf(w, f.x, acc[12]);                                        \
      acc[13] = fmaf(w, f.y, acc[13]);                                        \
      f = __half22float2(*reinterpret_cast<const __half2*>(&vb.w));            \
      acc[14] = fmaf(w, f.x, acc[14]);                                        \
      acc[15] = fmaf(w, f.y, acc[15]);                                        \
    }                                                                         \
  }
    S77_HALF(0, sa, 0)
    if (has_h1) S77_HALF(1, sb, kTile)
#undef S77_HALF

    // ---- stage the next 32-key group into the single buffer ----
    // No register prefetch: it would need 8 live uint4 across the compute and
    // pushes the kernel to vgpr 128 + spill. The exposed latency is covered by
    // the other resident CTA (2 CTA/CU). K lands first, then V, so at most four
    // uint4 are transient.
    __syncthreads();
    if (jn < j_hi) {
      const int pg = st_half ? page_b : page_a;
      const int of = st_half ? off_b : off_a;
      const bool st_ok = jn * kTile + st_row < seq_len;
      const int r = st_row * kRowUnits;
      const uint4* ksrc = reinterpret_cast<const uint4*>(
          key_cache + ((size_t)pg * kv_page_stride +
                       (size_t)kvh * kv_head_stride +
                       (size_t)(of + st_rr) * kv_token_stride) / 2);
      const uint4* vsrc = reinterpret_cast<const uint4*>(
          value_cache + ((size_t)pg * kv_page_stride +
                         (size_t)kvh * kv_head_stride +
                         (size_t)(of + st_rr) * kv_token_stride) / 2);
      kreg0 = st_ok ? ksrc[st_cu0 + 0] : zero4;
      kreg1 = st_ok ? ksrc[st_cu0 + 1] : zero4;
      kreg2 = st_ok ? ksrc[st_cu0 + 2] : zero4;
      kreg3 = st_ok ? ksrc[st_cu0 + 3] : zero4;
      k_lds[r + ((lds_f0(st_cu0 + 0) + st_row) & (kRowUnits - 1))] = kreg0;
      k_lds[r + ((lds_f0(st_cu0 + 1) + st_row) & (kRowUnits - 1))] = kreg1;
      k_lds[r + ((lds_f0(st_cu0 + 2) + st_row) & (kRowUnits - 1))] = kreg2;
      k_lds[r + ((lds_f0(st_cu0 + 3) + st_row) & (kRowUnits - 1))] = kreg3;
      vreg0 = st_ok ? vsrc[st_cu0 + 0] : zero4;
      vreg1 = st_ok ? vsrc[st_cu0 + 1] : zero4;
      vreg2 = st_ok ? vsrc[st_cu0 + 2] : zero4;
      vreg3 = st_ok ? vsrc[st_cu0 + 3] : zero4;
      v_lds[r + ((lds_f0(st_cu0 + 0) + st_row) & (kRowUnits - 1))] = vreg0;
      v_lds[r + ((lds_f0(st_cu0 + 1) + st_row) & (kRowUnits - 1))] = vreg1;
      v_lds[r + ((lds_f0(st_cu0 + 2) + st_row) & (kRowUnits - 1))] = vreg2;
      v_lds[r + ((lds_f0(st_cu0 + 3) + st_row) & (kRowUnits - 1))] = vreg3;
    }
    __syncthreads();
  }

  // ---- segment end: merge the per-lane expsums (delayed form) ----
  if (!row_valid) return;
  if (kDelayMerge) {
    float ls = l_run;
#pragma unroll
    for (int off = 1; off < 16; off <<= 1)
      ls += __shfl_xor(ls, off, 16);
    l_run = ls;
  }

  // v_descale folds into the accumulator (exact scalar; commutes with the
  // segment merge since every segment sees the same value).
  if (v_descale) {
    const float vd = v_descale[kvh * v_descale_stride];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] *= vd;
  }

  // Lane t stores output dims [t*16, t*16+16); rows m partition (token, head).
  const size_t seg_base = ((size_t)token_abs * num_query_heads + head) *
                              (num_segments * kHead) +
                          (size_t)seg * kHead + t * 16;
  if (segm_fp16) {
    // S56 WP2: fp16 segment output halves the workspace round trip. The
    // merge weights in reduce_segments cancel the l_run factor, so the
    // fp16 rounding (2^-11 rel) lands on the same order as the final fp16
    // output cast; acceptance stays with the WP2 ref-check gate.
    __half* dsth = reinterpret_cast<__half*>(segm_output) + seg_base;
    uint4 packed[2];
    __half2* h = reinterpret_cast<__half2*>(packed);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      h[i] = __floats2half2_rn(acc[2 * i], acc[2 * i + 1]);
    uint4* dst4h = reinterpret_cast<uint4*>(dsth);
    dst4h[0] = packed[0];
    dst4h[1] = packed[1];
  } else {
    float4* dst4 = reinterpret_cast<float4*>(
        reinterpret_cast<float*>(segm_output) + seg_base);
    dst4[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    dst4[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    dst4[2] = make_float4(acc[8], acc[9], acc[10], acc[11]);
    dst4[3] = make_float4(acc[12], acc[13], acc[14], acc[15]);
  }
  if (t == 0) {
    const size_t stat = ((size_t)token_abs * num_query_heads + head) *
                            num_segments +
                        seg;
    segm_max[stat] = m_run;
    segm_expsum[stat] = l_run;
  }
}

}  // namespace

void gfx906_attnd_decode_inner(
    torch::Tensor& segm_output, torch::Tensor& segm_max,
    torch::Tensor& segm_expsum, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, torch::Tensor& block_tables,
    torch::Tensor& seq_lens, torch::Tensor& cu_query_lens, double scale,
    int64_t num_segments, bool delayed_merge, int64_t tps_hint,
    std::optional<torch::Tensor> k_descale,
    std::optional<torch::Tensor> v_descale) {
  // Fail-fast gates for the supported subset; the Python dispatch checks the
  // same list and falls back to the Triton kernel when anything differs.
  TORCH_CHECK(query.scalar_type() == at::kHalf &&
                  key_cache.scalar_type() == at::kHalf &&
                  value_cache.scalar_type() == at::kHalf,
              "gfx906_attnd_decode_inner: fp16 Q/K/V required");
  TORCH_CHECK(query.stride(2) == 1 && query.stride(1) == kHead,
              "gfx906_attnd_decode_inner: query must be [qlen, NQ, 256]");
  TORCH_CHECK(key_cache.size(1) % kTile == 0 && key_cache.size(1) >= kTile &&
                  value_cache.size(1) == key_cache.size(1) &&
                  key_cache.size(3) == kHead && value_cache.size(3) == kHead,
              "gfx906_attnd_decode_inner: KV pages must be [B%16==0, NKV, 256]");
  // B832: the production cache packs K|V per token (token stride 512, page
  // stride 832*512) and single-KV-head ranks leave the head stride
  // degenerate; only 16-byte unit alignment and shape consistency are
  // structural. Everything is stride-parameterized below.
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1 &&
                  key_cache.stride(1) % 8 == 0 &&
                  key_cache.stride(1) >= kHead &&
                  value_cache.stride(1) == key_cache.stride(1) &&
                  key_cache.stride(0) % 8 == 0 &&
                  value_cache.stride(0) == key_cache.stride(0) &&
                  (key_cache.size(2) == 1 ||
                   (key_cache.stride(2) == kHead &&
                    value_cache.stride(2) == kHead)),
              "gfx906_attnd_decode_inner: unsupported KV layout");
  TORCH_CHECK(key_cache.size(2) == value_cache.size(2) &&
                  query.size(1) == key_cache.size(2) * kNpkv,
              "gfx906_attnd_decode_inner: GQA6 head mapping required");
  TORCH_CHECK(block_tables.scalar_type() == at::kInt &&
              seq_lens.scalar_type() == at::kInt &&
              cu_query_lens.scalar_type() == at::kInt,
              "gfx906_attnd_decode_inner: int32 metadata required");
  TORCH_CHECK(seq_lens.numel() == 1 && cu_query_lens.numel() == 2,
              "gfx906_attnd_decode_inner: single-sequence decode only");
  TORCH_CHECK(query.size(0) <= segm_output.size(0),
              "gfx906_attnd_decode_inner: segment buffer rows too small");
  TORCH_CHECK(num_segments >= 1 && num_segments <= segm_output.size(2),
              "gfx906_attnd_decode_inner: segment count out of range");
  TORCH_CHECK(key_cache.size(1) <= 1 << 20,
              "gfx906_attnd_decode_inner: block size out of range");

  // Descale folding: k_descale -> softmax scale, v_descale -> segment-end
  // accumulator (exact scalars, per kv head). Null = 1.0. Production per-
  // tensor scales arrive as expanded [num_seqs, nkv] views whose head
  // stride is 0 (broadcast) or 1 (materialized), so the head stride is
  // carried through instead of assuming a layout.
  const auto descale_ref = [&](std::optional<torch::Tensor>& d,
                               const char* name) ->
      std::pair<const float*, int> {
    if (!d.has_value() || !d->defined() || d->numel() == 0) return {nullptr, 0};
    TORCH_CHECK(d->scalar_type() == at::kFloat,
                "gfx906_attnd_decode_inner: ", name, " must be float32");
    TORCH_CHECK(d->numel() == 1 || d->size(-1) == key_cache.size(2),
                "gfx906_attnd_decode_inner: ", name,
                " must broadcast to [num_seqs, nkv]");
    const int hstride =
        d->dim() >= 2 ? (int)d->stride(1) : (d->numel() > 1 ? 1 : 0);
    return {d->data_ptr<float>(), hstride};
  };
  auto kds = descale_ref(k_descale, "k_descale");
  auto vds = descale_ref(v_descale, "v_descale");

  const int q_len = (int)query.size(0);
  const int q_blocks = (q_len * kNpkv + kBlockM - 1) / kBlockM;
  const int nkv = (int)key_cache.size(2);
  const int nq = (int)query.size(1);
  // S56 WP2: the segment workspace may be fp16 (halved round trip); the
  // producer store and the reduce kernel both follow the buffer dtype.
  TORCH_CHECK(segm_output.scalar_type() == at::kHalf ||
                  segm_output.scalar_type() == at::kFloat,
              "gfx906_attnd_decode_inner: segm_output must be fp16 or fp32");
  const int segm_is_fp16 =
      segm_output.scalar_type() == at::kHalf ? 1 : 0;
  const dim3 grid(q_blocks, nkv, (unsigned)num_segments);
  const dim3 block(kThreads);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  const uint32_t* qp =
      reinterpret_cast<const uint32_t*>(query.data_ptr<at::Half>());
  const uint32_t* kp =
      reinterpret_cast<const uint32_t*>(key_cache.data_ptr<at::Half>());
  const uint32_t* vp =
      reinterpret_cast<const uint32_t*>(value_cache.data_ptr<at::Half>());
  void* sop = segm_output.data_ptr();
  float* smx = segm_max.data_ptr<float>();
  float* sex = segm_expsum.data_ptr<float>();
  const int page_stride = (int)key_cache.stride(0);
  const int token_stride = (int)key_cache.stride(1);
  const int head_stride = (int)key_cache.stride(2);
  const int block_size = (int)key_cache.size(1);
  const int* btp = block_tables.data_ptr<int>();
  const int* slp = seq_lens.data_ptr<int>();
  const int* cup = cu_query_lens.data_ptr<int>();

  const at::cuda::OptionalCUDAGuard guard(device_of(query));

// Launch with a compile-time tiles-per-segment trip count (Exp E): a known
// trip count lets the compiler keep the staging pipeline rotated across
// iterations. Buckets cover the microbench/service shapes; anything else
// runs the dynamic-trip-count form.
#define ATTN_INNER_LAUNCH(V)                                                  \
  do {                                                                        \
    if (delayed_merge)                                                        \
      attnd_inner_decode_kernel<true, V>                                      \
          <<<grid, block, 0, stream>>>(                                       \
              sop, smx, sex, qp, kp, vp, btp, slp, cup, (float)scale,         \
              (int)num_segments, nq, page_stride, token_stride, head_stride,  \
              block_size, kds.first, kds.second, vds.first, vds.second,       \
              segm_is_fp16);                                                  \
    else                                                                      \
      attnd_inner_decode_kernel<false, V>                                     \
          <<<grid, block, 0, stream>>>(                                       \
              sop, smx, sex, qp, kp, vp, btp, slp, cup, (float)scale,         \
              (int)num_segments, nq, page_stride, token_stride, head_stride,  \
              block_size, kds.first, kds.second, vds.first, vds.second,       \
              segm_is_fp16);                                                  \
  } while (0)

  if (tps_hint == 4) {
    ATTN_INNER_LAUNCH(4);
  } else if (tps_hint == 16) {
    ATTN_INNER_LAUNCH(16);
  } else if (tps_hint == 27) {
    ATTN_INNER_LAUNCH(27);
  } else if (tps_hint == 32) {
    ATTN_INNER_LAUNCH(32);
  } else if (tps_hint == 2048) {
    // PREFILL-LCTX (S56): 131k ctx / 4 segments prefill final chunk
    // (131072 + 63) / 64 = 2048.
    ATTN_INNER_LAUNCH(2048);
  } else if (tps_hint == 4090) {
    // PREFILL-LCTX (S56): 262k ctx / 4 segments prefill final chunk
    // (262144 + 63) / 64 = 4090.
    ATTN_INNER_LAUNCH(4090);
  } else {
    ATTN_INNER_LAUNCH(0);
  }
#undef ATTN_INNER_LAUNCH
}
