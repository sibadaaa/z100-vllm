// S85: int8 KV-cache attention inner loop with the S77 T1 read-reuse form.
//
// Start point: csrc/rocm/attnd_inner_kvint8.hip (S81 production, md5
// 5a16cb37). The ONLY structural change is the QK^T tile walk:
//
//   before: `j++`, one Q row read per tile (16 x ds_read_b128 from q_lds)
//   after:  `j += 2`, one Q row read per TWO tiles, two independent sdot4
//           chains (a0..a3 for tile j, b0..b3 for tile j+1) fed from one
//           register-resident Q row.
//
// Buffering is the asymmetric form from the S85 design (int8 V is only 8 KB
// since S82-b X1, so a 32-key V single buffer would cost 16 KB and drop the
// kernel back to 2 CTA/CU):
//
//   K: 32-key single buffer (8 KB, unchanged size, rows 0..15 = tile j,
//      rows 16..31 = tile j+1); refilled once per iteration (2 tiles).
//   V: 16-key single buffer (8 KB, S79 double-barrier form). One iteration
//      consumes TWO V tiles, so it also refills mid-iteration: after half 0's
//      PV drains V(j), a barrier pair swaps in V(j+1) for half 1.
//   Q: q_lds 4 KB unchanged, read once per pair instead of once per tile.
//
// LDS total is unchanged (8192 + 8192 + 4096 + 128 = 20608 -> 3 CTA/CU).
// Numerics: the per-tile arithmetic order is untouched (same 16-unit sdot4
// chain, same isum reduction, same softmax/PV sequence), so out_md5 must be
// bit-identical to production -- that is the WP0 hard gate.
//
// Everything else (Q prologue quantization, PV fma_mix + X-REG weight
// recompute, segment buffers, host wrapper) is copied verbatim from the
// production file.

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include "core/hip_compat.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace {

constexpr int kHead = 256;
constexpr int kTile = 16;
constexpr int kBlockM = 16;
constexpr int kNpkv = 6;
constexpr int kThreads = 256;
constexpr int kRowUnits = kHead / 8;    // 32 uint4 per row for fp16 V
constexpr int kRowUnits8 = kHead / 16;  // 16 uint4 per row for int8 K (64 dwords)

__device__ __forceinline__ int lds_f0(int u) {
    return ((u & 1) << 4) + (u >> 1);
}
// fp16 V layout (unchanged from production)
__device__ __forceinline__ int lds_slot(int r, int u) {
    return r * kRowUnits + ((lds_f0(u) + r) & (kRowUnits - 1));
}
// int8 K layout: same permutation idea on a 16-unit row
__device__ __forceinline__ int lds_slot8(int r, int u) {
    return r * kRowUnits8 + ((((u & 1) << 3) + (u >> 1) + r) & (kRowUnits8 - 1));
}
// within-row unit index for the int8 K swizzle
__device__ __forceinline__ int lds_off8(int r, int u) {
    return (((u & 1) << 3) + (u >> 1) + r) & (kRowUnits8 - 1);
}

__device__ __forceinline__ int sdot4(uint32_t a, uint32_t b, int acc) {
    int r;
    asm volatile("v_dot4_i32_i8 %0, %1, %2, %3"
                 : "=v"(r)
                 : "v"((int)a), "v"((int)b), "v"(acc));
    return r;
}

__device__ __forceinline__ uint4 zero_uint4() {
    return make_uint4(0u, 0u, 0u, 0u);
}

template <bool kDelayMerge, int kTPS>
__global__ void __launch_bounds__(kThreads, 3)
attnd_inner_kvsdot4_kernel(
    void* __restrict__ segm_output,
    float* __restrict__ segm_max,
    float* __restrict__ segm_expsum,
    const uint32_t* __restrict__ query,
    const int8_t* __restrict__ key_cache,
    const int8_t* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int* __restrict__ cu_query_lens,
    const float scale,
    const int num_segments,
    const int num_query_heads,
    const int kv_page_stride,
    const int kv_token_stride,
    const int kv_head_stride,
    const int kv_block_size,
    const float* __restrict__ k_scale_cache,
    const int ks_page_stride,
    const int ks_token_stride,
    const int ks_head_stride,
    const float* __restrict__ v_scale_cache,
    const int vs_page_stride,
    const int vs_token_stride,
    const int vs_head_stride,
    const int segm_fp16,
    const int seg_stride_in) {
  const int kvh = blockIdx.y;
  const int seg = blockIdx.z;
  // Segment-buffer row stride (the backend allocates the capacity-sized
  // seg dim, e.g. 256) may exceed the active segment count (grid.z); the
  // address math must use the stride, the loop bounds the active count.
  const int seg_stride = seg_stride_in > 0 ? seg_stride_in : num_segments;
  const int q_len = cu_query_lens[1] - cu_query_lens[0];
  const int seq_len = seq_lens[0];
  const int ctx = seq_len - q_len;

  const int tiles_per_seg =
      kTPS ? kTPS
           : (seq_len + num_segments * kTile - 1) / (num_segments * kTile);
  if (seg * tiles_per_seg * kTile >= seq_len) return;

  const int tid = threadIdx.x;
  const int t = tid & 15;
  const int pair = blockIdx.x * kBlockM + (tid >> 4);
  const int qpos = pair / kNpkv;
  const bool row_valid = qpos < q_len;
  const int token_abs = qpos < q_len ? qpos : q_len - 1;
  const int head = kvh * kNpkv + (pair % kNpkv);
  const int bound = ctx + qpos;
  const int my_pair = tid >> 4;

  // ---- LDS: K int8 (8 KB, 32-key SINGLE buffer) + V fp16 (8 KB, 16-key
  // single buffer) + Q int8 (4 KB, one shot) + K scales for 32 keys (128 B).
  __shared__ uint4 k_lds[2 * kTile * kRowUnits8];
  __shared__ uint4 v_lds[1][kTile * kRowUnits];
  __shared__ int8_t q_lds[kBlockM * kHead];
  __shared__ float ks_lds[2 * kTile];

  const uint4 zero4 = zero_uint4();
  const int st_row = tid >> 4;
  const int st_cu = tid & 15;  // int8: one uint4 unit (16 codes) per lane

  const int j_lo = seg * tiles_per_seg;
  const int j_hi = min((seg + 1) * tiles_per_seg,
                       (seq_len + kTile - 1) / kTile);

  // ---- Q prologue: quantize this CTA's 16 Q rows to int8 ---------------
  // Thread (pair, lane) owns dims [lane*16, lane*16+16) of its pair's row.
  float q_scale = 1.0f;
  {
    const int raw_pair = blockIdx.x * kBlockM + my_pair;
    const int rqpos = raw_pair / kNpkv;
    const int rtoken = rqpos < q_len ? rqpos : q_len - 1;
    const int rhead = kvh * kNpkv + (raw_pair % kNpkv);
    const __half* qsrc = reinterpret_cast<const __half*>(query) +
                         ((size_t)rtoken * num_query_heads + rhead) * kHead +
                         st_cu * 16;
    float v[16];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      v[i] = __half2float(qsrc[i]);
      amax = fmaxf(amax, fabsf(v[i]));
    }
    // reduce amax across the 16 lanes of this pair's phase
#pragma unroll
    for (int off = 1; off < 16; off <<= 1)
      amax = fmaxf(amax, __shfl_xor(amax, off, 16));
    const float s = fmaxf(amax, 1e-6f) / 127.0f;
    q_scale = s;
    int8_t* dst = q_lds + my_pair * kHead + st_cu * 16;
    const float inv = 1.0f / s;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float x = v[i] * inv;
      // round half away from zero, then clamp to the signed range
      float r = x >= 0.0f ? floorf(x + 0.5f) : -floorf(-x + 0.5f);
      r = fminf(fmaxf(r, -127.0f), 127.0f);
      dst[i] = (int8_t)(int)r;
    }
  }

  float m_run = -INFINITY;
  float l_run = 0.0f;
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

  const int tpp = kv_block_size >> 4;
  uint4 kreg0, kreg1, vreg0, vreg1;
  float ksc0, ksc1;

  // Tile-stream cursor: (page_idx, page_off, page_next) always names the
  // NEXT tile to stage. K stages two tiles per iteration, V one tile twice
  // per iteration, so the single cursor advances twice per iteration.
  int page_idx = j_lo / tpp;
  int page_off = (j_lo - page_idx * tpp) * kTile;
  int page_next = block_tables[page_idx];

  auto adv = [&]() {
    page_off += kTile;
    if (page_off >= kv_block_size) {
      page_off -= kv_block_size;
      ++page_idx;
      page_next = block_tables[page_idx];
    }
  };

  // stage one K row: 16 int8 codes + one per-key K scale
  auto stage_k = [&](int page, int off, bool ok, uint4& kout, float& ksc) {
    // A tile past the last full page can carry an in-page offset that has
    // already rolled over; treat it as padding rather than reading OOB.
    ok = ok && off >= 0 && off < kv_block_size;
    if (ok) {
      const int32_t* ksrc = reinterpret_cast<const int32_t*>(
          key_cache + (size_t)page * kv_page_stride +
          (size_t)kvh * kv_head_stride + (size_t)off * kv_token_stride +
          st_cu * 16);
      // 4 x int32: the int8_per_token_head row is 256 B data + 4 B scale,
      // so the slot stride (520 B) is only 4-byte aligned -- a uint4 load
      // would fault. Four dword loads are the widest safe access.
      const int32_t a0 = ksrc[0], a1 = ksrc[1], a2 = ksrc[2], a3 = ksrc[3];
      kout = make_uint4((uint32_t)a0, (uint32_t)a1, (uint32_t)a2, (uint32_t)a3);
      ksc = k_scale_cache[page * ks_page_stride + off * ks_token_stride +
                          kvh * ks_head_stride];
    } else {
      kout = zero4;
      ksc = 1.0f;
    }
  };

  // stage one V row: 16 int8 codes dequantized to two uint4 (32 fp16 dims)
  auto stage_v = [&](int page, int off, bool ok, uint4& v0, uint4& v1) {
    ok = ok && off >= 0 && off < kv_block_size;
    if (ok) {
      const int32_t* vsrc = reinterpret_cast<const int32_t*>(
          value_cache + (size_t)page * kv_page_stride +
          (size_t)kvh * kv_head_stride + (size_t)off * kv_token_stride +
          st_cu * 16);
      const float vs = v_scale_cache[page * vs_page_stride +
                                     off * vs_token_stride +
                                     kvh * vs_head_stride];
      uint32_t d[4], e[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int32_t q4 = vsrc[i];
        const __half2 lo = __halves2half2(
            __float2half_rn((float)(q4 << 24 >> 24) * vs),
            __float2half_rn((float)(q4 << 16 >> 24) * vs));
        const __half2 hi = __halves2half2(
            __float2half_rn((float)(q4 << 8 >> 24) * vs),
            __float2half_rn((float)(q4 >> 24) * vs));
        d[i] = *reinterpret_cast<const uint32_t*>(&lo);
        e[i] = *reinterpret_cast<const uint32_t*>(&hi);
      }
      // dims 0..7 -> units (d0,e0),(d1,e1); dims 8..15 -> (d2,e2),(d3,e3)
      v0 = make_uint4(d[0], e[0], d[1], e[1]);
      v1 = make_uint4(d[2], e[2], d[3], e[3]);
    } else {
      v0 = zero4;
      v1 = zero4;
    }
  };

  {
    // ---- prologue: K pair (tiles j_lo, j_lo+1) + V of tile j_lo ----
    const bool ok0 = j_lo * kTile + st_row < seq_len;
    const bool ok1 = (j_lo + 1) * kTile + st_row < seq_len && j_lo + 1 < j_hi;
    int off1 = page_off + kTile;
    int pg1 = page_next;
    if (off1 >= kv_block_size) {
      off1 -= kv_block_size;
      if (j_lo + 1 < j_hi) pg1 = block_tables[page_idx + 1];
    }
    stage_k(page_next, page_off + st_row, ok0, kreg0, ksc0);
    stage_k(pg1, off1 + st_row, ok1, kreg1, ksc1);
    stage_v(page_next, page_off + st_row, ok0, vreg0, vreg1);
    k_lds[lds_slot8(st_row, st_cu)] = kreg0;
    k_lds[lds_slot8(kTile + st_row, st_cu)] = kreg1;
    v_lds[0][lds_slot(st_row, st_cu * 2)] = vreg0;
    v_lds[0][lds_slot(st_row, st_cu * 2 + 1)] = vreg1;
    ks_lds[st_row] = ksc0;
    ks_lds[kTile + st_row] = ksc1;
    __syncthreads();
    adv();  // cursor -> tile j_lo+1 (consumed by iteration j_lo's V refill)
  }

  // S85_HALF: one tile's score -> softmax -> PV. The arithmetic sequence is
  // the production per-tile sequence verbatim; only the score source (chain
  // A for tile j, chain B for tile j+1) and the ks index differ.
  const int gbase = tid & ~(16 - 1);
#define S85_HALF(H, ISUM)                                                      \
  {                                                                            \
    const int token_base = (j + (H)) * kTile;                                   \
    float s = (float)(ISUM) * (q_scale * ks_lds[(H) * kTile + t]) * scale;      \
    if (!row_valid || token_base + t > bound) s = -INFINITY;                    \
    float mt = s;                                                              \
    _Pragma("unroll") for (int off = 1; off < 16; off <<= 1) mt =               \
        fmaxf(mt, __shfl_xor(mt, off, 16));                                     \
    const float mn = fmaxf(m_run, mt) > -INFINITY ? fmaxf(m_run, mt) : 0.0f;    \
    const float alpha = __expf(m_run - mn);                                     \
    const float p = __expf(s - mn);                                            \
    m_run = mn;                                                                \
    if (kDelayMerge) {                                                         \
      l_run = l_run * alpha + p;                                               \
    } else {                                                                   \
      float lj = p;                                                            \
      _Pragma("unroll") for (int off = 1; off < 16; off <<= 1) lj =             \
          lj + __shfl_xor(lj, off, 16);                                         \
      l_run = l_run * alpha + lj;                                              \
    }                                                                          \
    _Pragma("unroll") for (int i = 0; i < 16; ++i) acc[i] *= alpha;             \
    _Pragma("unroll 4") for (int j2 = 0; j2 < kTile; ++j2) {                    \
      const uint4 va = v_lds[0][lds_slot(j2, t * 2)];                            \
      const uint4 vb = v_lds[0][lds_slot(j2, t * 2 + 1)];                       \
      const float w = __shfl(p, gbase | j2, 16);                                 \
      float2 f;                                                                 \
      f = __half22float2(*reinterpret_cast<const __half2*>(&va.x));              \
      acc[0] = fmaf(w, f.x, acc[0]);                                            \
      acc[1] = fmaf(w, f.y, acc[1]);                                            \
      f = __half22float2(*reinterpret_cast<const __half2*>(&va.y));              \
      acc[2] = fmaf(w, f.x, acc[2]);                                            \
      acc[3] = fmaf(w, f.y, acc[3]);                                            \
      f = __half22float2(*reinterpret_cast<const __half2*>(&va.z));              \
      acc[4] = fmaf(w, f.x, acc[4]);                                            \
      acc[5] = fmaf(w, f.y, acc[5]);                                            \
      f = __half22float2(*reinterpret_cast<const __half2*>(&va.w));              \
      acc[6] = fmaf(w, f.x, acc[6]);                                            \
      acc[7] = fmaf(w, f.y, acc[7]);                                            \
      f = __half22float2(*reinterpret_cast<const __half2*>(&vb.x));              \
      acc[8] = fmaf(w, f.x, acc[8]);                                            \
      acc[9] = fmaf(w, f.y, acc[9]);                                            \
      f = __half22float2(*reinterpret_cast<const __half2*>(&vb.y));              \
      acc[10] = fmaf(w, f.x, acc[10]);                                          \
      acc[11] = fmaf(w, f.y, acc[11]);                                          \
      f = __half22float2(*reinterpret_cast<const __half2*>(&vb.z));              \
      acc[12] = fmaf(w, f.x, acc[12]);                                          \
      acc[13] = fmaf(w, f.y, acc[13]);                                          \
      f = __half22float2(*reinterpret_cast<const __half2*>(&vb.w));              \
      acc[14] = fmaf(w, f.x, acc[14]);                                          \
      acc[15] = fmaf(w, f.y, acc[15]);                                          \
    }                                                                          \
  }

  for (int j = j_lo; j < j_hi; j += 2) {
    const bool has_h1 = (j + 1 < j_hi);
    const int jn = j + 2;

    // ---- QK^T: ONE Q row read feeds two independent sdot4 chains ----
    const uint4* krow0 = k_lds + t * kRowUnits8;
    const uint4* krow1 = k_lds + (kTile + t) * kRowUnits8;
    const uint4* qrow = reinterpret_cast<const uint4*>(q_lds + my_pair * kHead);
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    int b0 = 0, b1 = 0, b2 = 0, b3 = 0;
#pragma unroll 4
    for (int c = 0; c < kRowUnits8; ++c) {
      const uint4 qv = qrow[c];
      const uint4 kv0 = krow0[lds_off8(t, c)];
      a0 = sdot4(qv.x, kv0.x, a0);
      a1 = sdot4(qv.y, kv0.y, a1);
      a2 = sdot4(qv.z, kv0.z, a2);
      a3 = sdot4(qv.w, kv0.w, a3);
      // The second row is read unconditionally: k_lds rows 16..31 are always
      // allocated, so last-iteration stale data is harmless (the half is
      // gated off) and the loop body stays branch-free.
      const uint4 kv1 = krow1[lds_off8(kTile + t, c)];
      b0 = sdot4(qv.x, kv1.x, b0);
      b1 = sdot4(qv.y, kv1.y, b1);
      b2 = sdot4(qv.z, kv1.z, b2);
      b3 = sdot4(qv.w, kv1.w, b3);
    }

    S85_HALF(0, (a0 + a1) + (a2 + a3))

    // ---- V single buffer: half 0 drained tile j, swap in tile j+1 ----
    // The S79 double-barrier form is mandatory (a single barrier races the
    // refill against a slower reader's V drain).
    __syncthreads();
    if (has_h1) {
      const bool vok = (j + 1) * kTile + st_row < seq_len;
      stage_v(page_next, page_off + st_row, vok, vreg0, vreg1);
      v_lds[0][lds_slot(st_row, st_cu * 2)] = vreg0;
      v_lds[0][lds_slot(st_row, st_cu * 2 + 1)] = vreg1;
    }
    __syncthreads();

    if (has_h1) S85_HALF(1, (b0 + b1) + (b2 + b3))

    // ---- refill for the next iteration: K pair + the V tile after that ----
    // Staging lives after the compute (S77 convention); no register prefetch
    // across the compute -- 6 live uint4 would blow the vgpr budget.
    __syncthreads();
    if (jn < j_hi) {
      adv();  // cursor -> tile jn
      const bool k0ok = jn * kTile + st_row < seq_len;
      const bool k1ok = (jn + 1) * kTile + st_row < seq_len && jn + 1 < j_hi;
      int off2 = page_off + kTile;
      int pg2 = page_next;
      if (off2 >= kv_block_size) {
        off2 -= kv_block_size;
        if (jn + 1 < j_hi) pg2 = block_tables[page_idx + 1];
      }
      stage_k(page_next, page_off + st_row, k0ok, kreg0, ksc0);
      stage_k(pg2, off2 + st_row, k1ok, kreg1, ksc1);
      stage_v(page_next, page_off + st_row, k0ok, vreg0, vreg1);
      k_lds[lds_slot8(st_row, st_cu)] = kreg0;
      k_lds[lds_slot8(kTile + st_row, st_cu)] = kreg1;
      v_lds[0][lds_slot(st_row, st_cu * 2)] = vreg0;
      v_lds[0][lds_slot(st_row, st_cu * 2 + 1)] = vreg1;
      ks_lds[st_row] = ksc0;
      ks_lds[kTile + st_row] = ksc1;
      adv();  // cursor -> tile jn+1 (next iteration's V refill)
    }
    __syncthreads();
  }
#undef S85_HALF

  if (!row_valid) return;
  if (kDelayMerge) {
    float ls = l_run;
#pragma unroll
    for (int off = 1; off < 16; off <<= 1)
      ls += __shfl_xor(ls, off, 16);
    l_run = ls;
  }

  const size_t seg_base = ((size_t)token_abs * num_query_heads + head) *
                              (seg_stride * kHead) +
                          (size_t)seg * kHead + t * 16;
  if (segm_fp16) {
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
                            seg_stride +
                        seg;
    segm_max[stat] = m_run;
    segm_expsum[stat] = l_run;
  }
}

}  // namespace

void gfx906_attnd_decode_inner_kvint8(
    torch::Tensor& segm_output, torch::Tensor& segm_max,
    torch::Tensor& segm_expsum, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, torch::Tensor& k_scale_cache,
    torch::Tensor& v_scale_cache, torch::Tensor& block_tables,
    torch::Tensor& seq_lens, torch::Tensor& cu_query_lens, double scale,
    int64_t num_segments, bool delayed_merge, int64_t tps_hint,
    int64_t seg_stride) {
  TORCH_CHECK(query.scalar_type() == at::kHalf, "kvint8: fp16 Q required");
  TORCH_CHECK(key_cache.scalar_type() == at::kChar &&
                  value_cache.scalar_type() == at::kChar,
              "kvint8: int8 K/V required");
  TORCH_CHECK(k_scale_cache.scalar_type() == at::kFloat &&
                  v_scale_cache.scalar_type() == at::kFloat,
              "kvint8: fp32 scales required");

  const int q_len = (int)query.size(0);
  const int q_blocks = (q_len * kNpkv + kBlockM - 1) / kBlockM;
  const int nq = (int)query.size(1);
  const int segm_is_fp16 = segm_output.scalar_type() == at::kHalf ? 1 : 0;
  const dim3 block(kThreads);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  const uint32_t* qp =
      reinterpret_cast<const uint32_t*>(query.data_ptr<at::Half>());
  const int8_t* kp = key_cache.data_ptr<int8_t>();
  const int8_t* vp = value_cache.data_ptr<int8_t>();
  const float* ks = k_scale_cache.data_ptr<float>();
  const float* vs = v_scale_cache.data_ptr<float>();
  void* sop = segm_output.data_ptr();
  float* smx = segm_max.data_ptr<float>();
  float* sex = segm_expsum.data_ptr<float>();
  // int8_per_token_head layout is (num_blocks, nkv, block_size, hs+pad) --
  // slot is dim 2 and the head dim is degenerate when nkv == 1, unlike the
  // fp16 (blk, slot, nkv, hs) form the original host code assumed.
  // Tell the two layouts apart structurally: the fp16 cache the original
  // host code assumed is (blk, slot, nkv, hs) with slot == block_size on
  // dim 1; the int8_per_token_head cache is (blk, nkv, slot, hs+pad) with the
  // small nkv on dim 1. dim 2 is the slot axis iff it is the larger of the
  // two AND dim 1 is at most 8 (nkv for this model family).
  const bool slot_dim2 = key_cache.dim() == 4 &&
                         key_cache.size(1) <= 8 &&
                         key_cache.size(2) > key_cache.size(1);
  // S80 FIX: nkv lives on dim 1 in the runtime layout (blk, nkv, slot, hs+pad)
  // and on dim 2 only in the fp16 form (blk, slot, nkv, hs). Reading size(2)
  // unconditionally launched block_size CTAs in grid.y on the int8 cache and
  // indexed Q heads far outside num_query_heads -- the M>1 "kernel" failure
  // S79 chased was this host-side sniff, not a barrier defect.
  const int nkv = (int)(slot_dim2 ? key_cache.size(1) : key_cache.size(2));
  const dim3 grid(q_blocks, nkv, (unsigned)num_segments);
  const int page_stride = (int)key_cache.stride(0);
  const int token_stride =
      (int)(slot_dim2 ? key_cache.stride(2) : key_cache.stride(1));
  const int head_stride =
      (int)(slot_dim2 ? key_cache.stride(1) : key_cache.stride(2));
  const int block_size =
      (int)(slot_dim2 ? key_cache.size(2) : key_cache.size(1));
  const int ks_p = (int)k_scale_cache.stride(0);
  const int ks_t = (int)k_scale_cache.stride(1);
  const int ks_h = (int)k_scale_cache.stride(2);
  const int vs_p = (int)v_scale_cache.stride(0);
  const int vs_t = (int)v_scale_cache.stride(1);
  const int vs_h = (int)v_scale_cache.stride(2);
  const int* btp = block_tables.data_ptr<int>();
  const int* slp = seq_lens.data_ptr<int>();
  const int* cup = cu_query_lens.data_ptr<int>();

  const at::cuda::OptionalCUDAGuard guard(device_of(query));

#define KVSDOT4_LAUNCH(V)                                                     \
  do {                                                                        \
    if (delayed_merge)                                                        \
      attnd_inner_kvsdot4_kernel<true, V><<<grid, block, 0, stream>>>(         \
          sop, smx, sex, qp, kp, vp, btp, slp, cup, (float)scale,             \
          (int)num_segments, nq, page_stride, token_stride, head_stride,      \
          block_size, ks, ks_p, ks_t, ks_h, vs, vs_p, vs_t, vs_h,            \
          segm_is_fp16, seg_stride);                                        \
    else                                                                      \
      attnd_inner_kvsdot4_kernel<false, V><<<grid, block, 0, stream>>>(        \
          sop, smx, sex, qp, kp, vp, btp, slp, cup, (float)scale,             \
          (int)num_segments, nq, page_stride, token_stride, head_stride,      \
          block_size, ks, ks_p, ks_t, ks_h, vs, vs_p, vs_t, vs_h,            \
          segm_is_fp16, seg_stride);                                        \
  } while (0)

  if (tps_hint == 4) {
    KVSDOT4_LAUNCH(4);
  } else if (tps_hint == 16) {
    KVSDOT4_LAUNCH(16);
  } else if (tps_hint == 27) {
    KVSDOT4_LAUNCH(27);
  } else if (tps_hint == 32) {
    KVSDOT4_LAUNCH(32);
  } else if (tps_hint == 2048) {
    KVSDOT4_LAUNCH(2048);
  } else if (tps_hint == 4090) {
    KVSDOT4_LAUNCH(4090);
  } else {
    KVSDOT4_LAUNCH(0);
  }
#undef KVSDOT4_LAUNCH
}
