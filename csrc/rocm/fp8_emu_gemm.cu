// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W8A8 FP8 GEMM/GEMV for gfx906 (Vega20 / Hygon Z100), emulated on the fp16
// datapath.
//
// Why this kernel exists
// ----------------------
// gfx906 has no fp8 support at all: `v_cvt_f32_fp8`, `v_cvt_pk_fp8_f32`,
// `v_cvt_pk_f32_fp8` and `v_cvt_f32_bf8` do not assemble for `-mcpu=gfx906`,
// there is no fp8 MFMA, and `torch._scaled_mm`'s FP8 path is unavailable.  As
// a result every FP8 kernel candidate in vLLM refuses to run on this target
// (`ROCmFP8ScaledMMLinearKernel`: "wvSplitKQ is not built on gfx906 (no MFMA)",
// the torch FP8 kernels: "requires a platform with torch FP8 scaled-MM
// support"), and the only FP8 path that does run (block-scaled, pure Triton)
// falls back to scalar `v_fmac_f32` emulation, measured ~36x slower than fp16
// (docs/result/z100-08b-gating-and-requirements-result.md).
//
// Emulation strategy
// ------------------
//   * fp8 -> fp16 through a 256-entry table in LDS (`ds_read_u16`).  For both
//     float8_e4m3fn and float8_e5m2 every finite code is exactly representable
//     in fp16 (e4m3 keeps 3 mantissa bits and peaks at 448; e5m2 keeps 2 and
//     peaks at 57344 < 65504, its 2^-16 subnormals landing on fp16
//     subnormals), so the widening is lossless -- unlike the packed bit-trick
//     route, which cannot represent e4m3 subnormals (< 2^-6, very common in
//     per-channel scaled weights) without per-lane fix-ups.
//   * `v_dot2_f32_f16`: two fp16 MACs per instruction with an fp32
//     accumulator, which is exactly the accumulation contract FP8 wants.
//   * Weights stay fp8 in HBM.  They are widened inside the kernel (in
//     registers on the way to LDS); no fp16 copy is ever materialized, so the
//     resident footprint is the checkpoint's (a 27B FP8 model is ~27 GiB;
//     an fp16 copy would be 54 GiB and does not fit this platform).
//
// Layout contract (native compressed-tensors / Fp8LinearMethod):
//   * `w` is [N, K] fp8, contiguous, K-major (the layer weight as loaded; the
//     FP8 scaled-MM path hands over a zero-copy [K, N] *view* of it).
//   * `a` is [M, K] fp8, contiguous, per-token (dynamic) or per-tensor scaled.
//   * `a_s` is [M] (per-token) or [1] (per-tensor); `w_s` is [N]
//     (per-channel) or [1] (per-tensor), both float32.
//   * Epilogue order is `(acc * a_s) * w_s`, then cast to `out_dtype`, then
//     bias in `out_dtype` -- the same order as `w8a8_sdot4_*` and as
//     `triton_scaled_mm`, so results are comparable across backends.
//
// Provenance: the tiling/double-buffering skeleton is adapted from
// `csrc/rocm/w8a8_sdot4.cu` (same project, Apache-2.0), which validated this
// shape on the same hardware; the datapath (fp8 LUT widening + `v_dot2_f32_f16`
// + fp32 accumulation) and the LDS tile element size (fp16 pairs instead of
// int32 words) are new.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <c10/util/Float8_e4m3fn.h>

#include <cstdint>
#include <cstdlib>

#include "rocm/fp8_emu_gemm.h"
#include "rocm/fp8_dequant.cuh"
#include "rocm/gfx906_emu_common.cuh"

namespace {

using vllm::gfx906_emu::dot2;
using vllm::gfx906_emu::fp8_lut_stage;
using vllm::gfx906_emu::fp8_lut_u16;
using vllm::gfx906_emu::fp8x4_to_f16x2x2;

constexpr int kMsMax = 8;    // GEMV covers M <= 8 (decode)
constexpr int kNtMax = 512;  // max GEMV threads per block (power of two)

// ---------------------------------------------------------------------------
// decode path: M <= 8
// ---------------------------------------------------------------------------
// One block per output column n (one row of W).  `nt` threads split K/4 dwords;
// each dword is 4 fp8 codes widened to two fp16 pairs and consumed by two
// `v_dot2_f32_f16`s per row of A.  W is read once per row, so the traffic does
// not depend on M.
//
// The A widening is hoisted out of the per-column loop for M >= 2: every block
// re-reads the same A row, so widening it inside the loop costs M LDS reads per
// A element *per block* (measured: at M = 8 the LUT traffic is 9x the dot2
// traffic and the kernel falls behind rocBLAS).  `fp8_emu_widen_a_kernel`
// converts A once per launch into an fp16 scratch (M*K*2 bytes, ~80 KiB at
// M=8/K=5120) and the GEMV then reads fp16 pairs directly.

__global__ void fp8_emu_widen_a_kernel(const uint8_t* __restrict__ a,
                                       const __half* __restrict__ lut,
                                       uint32_t* __restrict__ out,
                                       const int64_t n_dwords) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_dwords) return;
  uint32_t lo, hi;
  fp8x4_to_f16x2x2(lut, reinterpret_cast<const uint32_t*>(a)[i], lo, hi);
  out[2 * i] = lo;
  out[2 * i + 1] = hi;
}

template <int MS, typename T, bool PRECONV>
__global__ void fp8_emu_gemv_kernel(const uint8_t* __restrict__ a,  // [M, K]
                                    const uint8_t* __restrict__ w,  // [N, K]
                                    const uint32_t* __restrict__ a_h2,  // | null
                                    const float* __restrict__ a_s,
                                    const float* __restrict__ w_s,
                                    const __half* __restrict__ lut,  // [256]
                                    const T* __restrict__ bias,      // [N] | null
                                    T* __restrict__ out,             // [M, N]
                                    const int M, const int N, const int K4,
                                    const int a_s_stride,
                                    const int w_s_stride) {
  extern __shared__ float red[];  // [MS][nt]
  __shared__ __half sl[256];

  const int n = blockIdx.x;
  const int t = threadIdx.x;
  const int nt = blockDim.x;

  fp8_lut_stage(sl, lut, nt);
  __syncthreads();

  const uint32_t* __restrict__ wp =
      reinterpret_cast<const uint32_t*>(w) + (int64_t)n * K4;
  const uint32_t* __restrict__ ap = reinterpret_cast<const uint32_t*>(a);

  float acc[MS];
#pragma unroll
  for (int m = 0; m < MS; ++m) acc[m] = 0.0f;

  // Consecutive dwords across threads -> coalesced loads.  The first
  // __syncthreads() above also orders the LUT store.
  for (int i = t; i < K4; i += nt) {
    const uint32_t wa = wp[i];
    uint32_t w0, w1;
    fp8x4_to_f16x2x2(sl, wa, w0, w1);
#pragma unroll
    for (int m = 0; m < MS; ++m) {
      if (m < M) {
        uint32_t x0, x1;
        if (PRECONV) {
          const uint32_t* __restrict__ arow = a_h2 + (int64_t)m * (K4 * 2);
          x0 = arow[2 * i];
          x1 = arow[2 * i + 1];
        } else {
          fp8x4_to_f16x2x2(sl, ap[(int64_t)m * K4 + i], x0, x1);
        }
        acc[m] = dot2(w0, x0, acc[m]);
        acc[m] = dot2(w1, x1, acc[m]);
      }
    }
  }

#pragma unroll
  for (int m = 0; m < MS; ++m) {
    if (m < M) red[m * nt + t] = acc[m];
  }
  __syncthreads();
  for (int off = nt >> 1; off > 0; off >>= 1) {
    if (t < off) {
#pragma unroll
      for (int m = 0; m < MS; ++m) {
        if (m < M) red[m * nt + t] += red[m * nt + t + off];
      }
    }
    __syncthreads();
  }

  if (t < M) {
    const float v =
        red[t * nt] * a_s[t * a_s_stride] * w_s[(int64_t)n * w_s_stride];
    T o = static_cast<T>(v);
    if (bias != nullptr) o += static_cast<T>(bias[n]);
    out[t * (int64_t)N + n] = o;
  }
}

template <typename T>
void launch_gemv(int M, int N, int K4, int nt, int a_s_stride, int w_s_stride,
                 const uint8_t* a, const uint8_t* w,
                 const uint32_t* a_h2, const float* a_s, const float* w_s,
                 const __half* lut, const T* bias, T* out, hipStream_t stream) {
  const size_t smem = [&]() {
    if (M <= 1) return (size_t)1 * nt * sizeof(float);
    if (M <= 4) return (size_t)4 * nt * sizeof(float);
    return (size_t)kMsMax * nt * sizeof(float);
  }();
  dim3 grid(N), block(nt);
#define VLLM_FP8_EMU_GEMV_CASE(MS)                                          \
  do {                                                                      \
    if (a_h2 != nullptr) {                                                  \
      fp8_emu_gemv_kernel<MS, T, true><<<grid, block, smem, stream>>>(      \
          a, w, a_h2, a_s, w_s, lut, bias, out, M, N, K4, a_s_stride,       \
          w_s_stride);                                                      \
    } else {                                                                \
      fp8_emu_gemv_kernel<MS, T, false><<<grid, block, smem, stream>>>(     \
          a, w, nullptr, a_s, w_s, lut, bias, out, M, N, K4, a_s_stride,    \
          w_s_stride);                                                      \
    }                                                                       \
    return;                                                                 \
  } while (0)
  if (M <= 1) VLLM_FP8_EMU_GEMV_CASE(1);
  if (M <= 4) VLLM_FP8_EMU_GEMV_CASE(4);
  VLLM_FP8_EMU_GEMV_CASE(kMsMax);
#undef VLLM_FP8_EMU_GEMV_CASE
}

// ---------------------------------------------------------------------------
// prefill path: M > 8
// ---------------------------------------------------------------------------
// Each block computes a BM x BN tile of C[m, n] = sum_k A[m, k] * W[n, k] with
// both operands fp8 and K-contiguous.  K is walked in BK = 32 element chunks;
// A/W go global -> registers -> (LUT widen) -> LDS with double buffering, so
// the global load of chunk s+1 overlaps the dot2 work on chunk s.  Each LDS
// dword holds two fp16 lanes, i.e. exactly one `v_dot2_f32_f16` operand.

constexpr int kFBk = 32;             // elements per K chunk
constexpr int kFKw4 = kFBk / 4;      // 8 fp8 dwords per row per chunk
constexpr int kFKw = kFBk / 2;       // 16 fp16 dwords per row per chunk
constexpr int kFKp = kFKw + 1;       // 17 (padded LDS stride, bank spread)
constexpr int kFLogKw4 = 3;          // log2(kFKw4)

template <int BM, int BN, int TM, int TN>
struct FConf {
  static constexpr int MG = BM / TM;  // threads along M
  static constexpr int NG = BN / TN;  // threads along N
  static constexpr int NT = MG * NG;
  // fp8 dwords staged per thread.  Ceil, because a decode tile can have
  // BM * kFKw4 < NT (BM = 8 at 128 threads): the surplus threads stage a
  // zeroed dword behind the `idx < BM * kFKw4` guard instead of moving the
  // tile geometry.
  static constexpr int A4 = (BM * kFKw4 + NT - 1) / NT;
  static constexpr int B4 = (BN * kFKw4 + NT - 1) / NT;
};

// Stage one K slice of A/W from global into registers (zero-filled for
// out-of-range M/N/K).
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void fp8_gemm_load(const uint8_t* __restrict__ A,
                                              const uint8_t* __restrict__ W,
                                              int m0, int n0, int M, int N,
                                              int K, int K4, int k0, int tid,
                                              uint32_t* __restrict__ dst) {
  using C = FConf<BM, BN, TM, TN>;
  const uint32_t* __restrict__ ap =
      reinterpret_cast<const uint32_t*>(A + (int64_t)m0 * K + k0);
  const uint32_t* __restrict__ wp =
      reinterpret_cast<const uint32_t*>(W + (int64_t)n0 * K + k0);
  const bool kfull = (k0 + kFBk) <= K;
#pragma unroll
  for (int u = 0; u < C::A4; ++u) {
    const int idx = tid + u * C::NT;
    const int r = idx >> kFLogKw4;
    const int jd = idx & (kFKw4 - 1);
    uint32_t v = 0;
    if (idx < BM * kFKw4 && m0 + r < M &&
        (kfull || (k0 + (jd + 1) * 4) <= K))
      v = ap[(int64_t)r * K4 + jd];
    dst[u] = v;
  }
#pragma unroll
  for (int u = 0; u < C::B4; ++u) {
    const int idx = tid + u * C::NT;
    const int r = idx >> kFLogKw4;
    const int jd = idx & (kFKw4 - 1);
    uint32_t v = 0;
    if (idx < BN * kFKw4 && n0 + r < N &&
        (kfull || (k0 + (jd + 1) * 4) <= K))
      v = wp[(int64_t)r * K4 + jd];
    dst[C::A4 + u] = v;
  }
}

// Registers -> one LDS stage, widening fp8 to fp16 pairs on the way.  The
// stage is passed as a pointer (not as a compile-time subscript) so the
// multi-stage pipeline below can rotate `chunk % stages` at run time without
// unrolling by the least common multiple of the register and stage counts.
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void fp8_gemm_store(int tid,
                                               const uint32_t* __restrict__ src,
                                               const __half* __restrict__ lut,
                                               uint32_t* __restrict__ As,
                                               uint32_t* __restrict__ Bs) {
  using C = FConf<BM, BN, TM, TN>;
#pragma unroll
  for (int u = 0; u < C::A4; ++u) {
    const int idx = tid + u * C::NT;
    if (idx >= BM * kFKw4) continue;
    const int r = idx >> kFLogKw4;
    const int jd = idx & (kFKw4 - 1);
    uint32_t lo, hi;
    fp8x4_to_f16x2x2(lut, src[u], lo, hi);
    As[r * kFKp + 2 * jd] = lo;
    As[r * kFKp + 2 * jd + 1] = hi;
  }
#pragma unroll
  for (int u = 0; u < C::B4; ++u) {
    const int idx = tid + u * C::NT;
    if (idx >= BN * kFKw4) continue;
    const int r = idx >> kFLogKw4;
    const int jd = idx & (kFKw4 - 1);
    uint32_t lo, hi;
    fp8x4_to_f16x2x2(lut, src[C::A4 + u], lo, hi);
    Bs[r * kFKp + 2 * jd] = lo;
    Bs[r * kFKp + 2 * jd + 1] = hi;
  }
}

// One LDS stage -> TM x TN fp32 accumulators (one dot2 = 2 fp16 MACs).
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void fp8_gemm_compute(
    int tx, int ty, const uint32_t* __restrict__ as,
    const uint32_t* __restrict__ bs, float (&acc)[TM][TN]) {
  using C = FConf<BM, BN, TM, TN>;
#pragma unroll
  for (int u = 0; u < kFKw; ++u) {
    uint32_t a[TM], b[TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) a[i] = as[(ty + C::MG * i) * kFKp + u];
#pragma unroll
    for (int j = 0; j < TN; ++j) b[j] = bs[(tx + C::NG * j) * kFKp + u];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        acc[i][j] = dot2(a[i], b[j], acc[i][j]);
      }
    }
  }
}

template <int BM, int BN, int TM, int TN, typename T>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    fp8_emu_gemm_kernel(const uint8_t* __restrict__ A,
                        const uint8_t* __restrict__ W,
                        const float* __restrict__ a_s,
                        const float* __restrict__ w_s,
                        const __half* __restrict__ lut,
                        const T* __restrict__ bias, T* __restrict__ C, int M,
                        int N, int K, int a_s_stride, int w_s_stride) {
  using C_ = FConf<BM, BN, TM, TN>;
  constexpr int NT = C_::NT;
  constexpr int FRAGS = C_::A4 + C_::B4;
  static_assert(C_::MG * TM == BM && C_::NG * TN == BN, "tile/thread mismatch");

  __shared__ uint32_t As[2][BM * kFKp];
  __shared__ uint32_t Bs[2][BN * kFKp];
  __shared__ __half sl[256];

  const int tid = threadIdx.x;
  const int tx = tid % C_::NG;  // along N
  const int ty = tid / C_::NG;  // along M
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int K4 = K >> 2;
  const int NK = (K + kFBk - 1) / kFBk;

  fp8_lut_stage(sl, lut, NT);
  __syncthreads();

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
  }

  uint32_t f0[FRAGS], f1[FRAGS];
  fp8_gemm_load<BM, BN, TM, TN>(A, W, m0, n0, M, N, K, K4, 0, tid, f0);

  // Per iteration: store LDS(S) -> syncthreads -> issue the next global load ->
  // compute from LDS(S).  One syncthreads suffices: it also guarantees the
  // slice consumed by the previous compute is free to reuse.
  int kb = 0;
  for (; kb + 1 < NK; kb += 2) {
    fp8_gemm_store<BM, BN, TM, TN>(tid, f0, sl, As[0], Bs[0]);
    __syncthreads();
    fp8_gemm_load<BM, BN, TM, TN>(A, W, m0, n0, M, N, K, K4,
                                  (kb + 1) * kFBk, tid, f1);
    fp8_gemm_compute<BM, BN, TM, TN>(tx, ty, As[0], Bs[0], acc);

    fp8_gemm_store<BM, BN, TM, TN>(tid, f1, sl, As[1], Bs[1]);
    __syncthreads();
    fp8_gemm_load<BM, BN, TM, TN>(A, W, m0, n0, M, N, K, K4,
                                  (kb + 2) * kFBk, tid, f0);
    fp8_gemm_compute<BM, BN, TM, TN>(tx, ty, As[1], Bs[1], acc);
  }
  if (kb < NK) {  // odd NK: the leftover slice is number 0 (kb is even here)
    fp8_gemm_store<BM, BN, TM, TN>(tid, f0, sl, As[0], Bs[0]);
    __syncthreads();
    fp8_gemm_compute<BM, BN, TM, TN>(tx, ty, As[0], Bs[0], acc);
  }

  // Epilogue: (acc * a_s) * w_s, cast, then bias -- same order as
  // w8a8_sdot4_* and triton_scaled_mm.
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + C_::MG * i;
    if (row >= M) continue;
    const float sa = a_s[row * a_s_stride];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + tx + C_::NG * j;
      if (col >= N) continue;
      const float v = acc[i][j] * sa * w_s[(int64_t)col * w_s_stride];
      T o = static_cast<T>(v);
      if (bias != nullptr) o += static_cast<T>(bias[col]);
      C[(int64_t)row * N + col] = o;
    }
  }
}

template <int BM, int BN, int TM, int TN, typename T>
void launch_gemm(int M, int N, int K, int a_s_stride, int w_s_stride,
                 const uint8_t* A, const uint8_t* W, const float* a_s,
                 const float* w_s, const __half* lut, const T* bias, T* C,
                 hipStream_t stream) {
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block((BM / TM) * (BN / TN));
  fp8_emu_gemm_kernel<BM, BN, TM, TN, T><<<grid, block, 0, stream>>>(
      A, W, a_s, w_s, lut, bias, C, M, N, K, a_s_stride, w_s_stride);
}

// ---------------------------------------------------------------------------
// block-scaled variants (weight_scale_inv is [cdiv(N, block_n), cdiv(K, block_k)])
// ---------------------------------------------------------------------------
// The quantization contract of an FP8 block-scaled checkpoint is
//
//   W[m, n] = W_q[m, n] * w_s[n / block_n, k / block_k]
//
// so the accumulation becomes
//
//   out[m, n] = sum_kb ( sum_{k in kb} a[m,k] * w[n,k] ) * a_s[m, kb] * w_s[nb, kb]
//
// Scaling is linear, so a per-thread partial sum may be scaled *before* the
// cross-thread reduction:
//
//   sum_t sum_kb p[t,kb] * s[kb]  ==  sum_kb s[kb] * sum_t p[t,kb]
//
// which is what lets both kernels below keep exactly one reduction per launch
// (GEMV) and one register tile (GEMM): only the accumulator that belongs to the
// *current* K block is scaled, at the block boundary.
//
// This differs from the per-tensor / per-channel path only in where the scales
// are indexed from; the widening (LDS LUT) and MAC (`v_dot2_f32_f16`) datapath
// is shared.

// GEMV, block scaled.  Each thread owns whole K blocks (kb = t, t + nt, ...)
// instead of strided dwords, so it touches one cache line per block (block_k
// bytes at one byte per fp8 element) and pays the scale lookup once per block_k
// elements rather than once per dword.
//
// `HAS_A_S` is false on the W8A16 route: A is already fp16 (read through the
// PRECONV path) and there is no activation scale, so the per-block factor
// degenerates to the weight scale.
template <int MS, typename T, bool PRECONV, bool HAS_A_S = true>
__global__ void fp8_emu_gemv_block_kernel(
    const uint8_t* __restrict__ a,      // [M, K]
    const uint8_t* __restrict__ w,      // [N, K], row stride w_stride
    const uint32_t* __restrict__ a_h2,  // | null
    const float* __restrict__ a_s,      // [M, nkb]
    const float* __restrict__ w_s,      // [cdiv(N, block_n), nkb]
    const __half* __restrict__ lut,     // [256]
    const T* __restrict__ bias,         // [N] | null
    T* __restrict__ out,                // [M, N]
    const int M, const int N, const int K4, const int64_t w_stride,
    const int nkb, const int dpb, const int block_n) {
  extern __shared__ float red[];  // [MS][nt]
  __shared__ __half sl[256];

  const int n = blockIdx.x;
  const int t = threadIdx.x;
  const int nt = blockDim.x;

  fp8_lut_stage(sl, lut, nt);
  __syncthreads();

  const uint32_t* __restrict__ wp =
      reinterpret_cast<const uint32_t*>(w + (int64_t)n * w_stride);
  const uint32_t* __restrict__ ap = reinterpret_cast<const uint32_t*>(a);
  const float* __restrict__ wsr = w_s + (int64_t)(n / block_n) * nkb;

  float acc[MS];
#pragma unroll
  for (int m = 0; m < MS; ++m) acc[m] = 0.0f;

  for (int kb = t; kb < nkb; kb += nt) {
    const float wsv = wsr[kb];
    float s[MS];
#pragma unroll
    for (int m = 0; m < MS; ++m) {
      s[m] = (m < M) ? (HAS_A_S ? a_s[m * nkb + kb] * wsv : wsv) : 0.0f;
    }
    float loc[MS];
#pragma unroll
    for (int m = 0; m < MS; ++m) loc[m] = 0.0f;

    const int i0 = kb * dpb;
    const int i1 = min(i0 + dpb, K4);
    for (int i = i0; i < i1; ++i) {
      const uint32_t wa = wp[i];
      uint32_t w0, w1;
      fp8x4_to_f16x2x2(sl, wa, w0, w1);
#pragma unroll
      for (int m = 0; m < MS; ++m) {
        if (m < M) {
          uint32_t x0, x1;
          if (PRECONV) {
            const uint32_t* __restrict__ arow = a_h2 + (int64_t)m * (K4 * 2);
            x0 = arow[2 * i];
            x1 = arow[2 * i + 1];
          } else {
            fp8x4_to_f16x2x2(sl, ap[(int64_t)m * K4 + i], x0, x1);
          }
          loc[m] = dot2(w0, x0, loc[m]);
          loc[m] = dot2(w1, x1, loc[m]);
        }
      }
    }
#pragma unroll
    for (int m = 0; m < MS; ++m) acc[m] += loc[m] * s[m];
  }

#pragma unroll
  for (int m = 0; m < MS; ++m) {
    if (m < M) red[m * nt + t] = acc[m];
  }
  __syncthreads();
  for (int off = nt >> 1; off > 0; off >>= 1) {
    if (t < off) {
#pragma unroll
      for (int m = 0; m < MS; ++m) {
        if (m < M) red[m * nt + t] += red[m * nt + t + off];
      }
    }
    __syncthreads();
  }

  if (t < M) {
    T o = static_cast<T>(red[t * nt]);
    if (bias != nullptr) o += static_cast<T>(bias[n]);
    out[t * (int64_t)N + n] = o;
  }
}

template <typename T>
void launch_gemv_block(int M, int N, int K4, int nt, int64_t w_stride, int nkb,
                       int dpb, int block_n, bool has_a_s, const uint8_t* a,
                       const uint8_t* w, const uint32_t* a_h2,
                       const float* a_s, const float* w_s, const __half* lut,
                       const T* bias, T* out, hipStream_t stream) {
  const size_t smem = [&]() {
    if (M <= 1) return (size_t)1 * nt * sizeof(float);
    if (M <= 4) return (size_t)4 * nt * sizeof(float);
    return (size_t)kMsMax * nt * sizeof(float);
  }();
  dim3 grid(N), block(nt);
  // Three variants: A pre-widened (W8A8, M >= 2), A widened inline (W8A8,
  // M == 1) and A already fp16 (W8A16, HAS_A_S == false).
#define VLLM_FP8_EMU_GEMV_BLOCK_LAUNCH(MS, PRE, AS)                          \
  fp8_emu_gemv_block_kernel<MS, T, PRE, AS><<<grid, block, smem, stream>>>(  \
      a, w, (PRE) ? a_h2 : nullptr, a_s, w_s, lut, bias, out, M, N, K4,       \
      w_stride, nkb, dpb, block_n)
#define VLLM_FP8_EMU_GEMV_BLOCK_CASE(MS)                                     \
  do {                                                                       \
    if (a_h2 != nullptr) {                                                   \
      if (has_a_s) {                                                         \
        VLLM_FP8_EMU_GEMV_BLOCK_LAUNCH(MS, true, true);                      \
      } else {                                                               \
        VLLM_FP8_EMU_GEMV_BLOCK_LAUNCH(MS, true, false);                     \
      }                                                                      \
    } else {                                                                 \
      VLLM_FP8_EMU_GEMV_BLOCK_LAUNCH(MS, false, true);                       \
    }                                                                        \
    return;                                                                  \
  } while (0)
  if (M <= 1) VLLM_FP8_EMU_GEMV_BLOCK_CASE(1);
  if (M <= 4) VLLM_FP8_EMU_GEMV_BLOCK_CASE(4);
  VLLM_FP8_EMU_GEMV_BLOCK_CASE(kMsMax);
#undef VLLM_FP8_EMU_GEMV_BLOCK_CASE
#undef VLLM_FP8_EMU_GEMV_BLOCK_LAUNCH
}

// ---------------------------------------------------------------------------
// streaming decode kernel, block-scaled: M <= 8
// ---------------------------------------------------------------------------
// The tiled GEMM above costs 375 us for N=4352/K=5120 regardless of M, i.e.
// 63 GB/s = 9% of the 670 GB/s this card reaches on a flat stream.  The
// standalone roofline probe (pmc/gemv_probe.cpp) shows why: the double-buffered
// 32-element K loop is an on-chip pipeline with a serial dependency per chunk
// and only 68 blocks x 128 threads live, and deleting *all* of its global
// loads makes it slower (341.6 vs 320.4 us), so the exposed latency is LDS /
// barrier, not DRAM.
//
// This kernel is shaped like the probe's `k_stream_rows`, which did reach
// 621.7 GB/s at exactly this geometry (grid = N/64, 512 threads/block), and
// since GEMV-OPT-20261025.md P0 like the probe's `gemv_lds` (451 GB/s at
// M=8 with A in LDS slices):
//
//   * one block per `rows` rows of W, one warp per row at a time;
//   * 64 lanes stride K in uint4 (16 fp8 byte) steps, so every W load is a
//     single fully-coalesced 1 KiB request per warp;
//   * no K-carried dependency: each row restarts the accumulator chain, and
//     the K walk is sliced so the W chunk a lane loads next is independent of
//     the chunk it is dotting (the old per-row PREF pipeline retired with the
//     slice structure -- the probe's lds_* geometry ran without one);
//   * `rows` divides `block_n`, so every row of the block reads the same
//     `w_s[nb, :]` row and the per-block_k scales can be folded once per block
//     into LDS as `a_s[m, kb] * w_s[nb, kb]` (identical arithmetic to
//     `fp8_emu_gemv_block_kernel`: `acc[m] += loc[m] * s[m]`);
//   * A is widened into LDS in K-slices (`kStreamSlice` fp16 per row, staged
//     once per slice and reused by every row of the block), so the M*K LUT
//     lookups are paid once per block instead of once per row of W while the
//     staging buffer stays a fixed ~16 KiB instead of growing with M*K -- the
//     old whole-A tile (82 KiB at M=8/K=5120) exceeded the 44 KiB gate and
//     pushed M >= 5 onto the tiled GEMM, which is what
//     docs/perf/GEMV-OPT-20261025.md P0 fixes.
constexpr int kStreamChunk = 16;  // fp8 elements owned by one lane per step
// A slice length in fp8 elements: 1024 = 64 uint4 = exactly one W chunk per
// lane per (slice, row), i.e. 1 KiB of W per row per slice -- the geometry the
// `gemv_probe` lds_* sweep measured 451 GB/s with at M=8 (fp16 stand-in
// weights, docs/perf/GEMV-OPT-20261025.md §1.2).
constexpr int kStreamSlice = 1024;
// Upper bound for the streaming path's dynamic LDS: the A slice
// (MS * kStreamSlice fp16) + folded scales.  The real gate is min(this, device
// LDS per block - 1 KiB), queried once from the device, so a smaller-LDS part
// degrades to the tiled GEMM instead of failing the launch.
constexpr size_t kStreamSmemCap = (size_t)44 * 1024;

size_t stream_smem_limit() {
  static const size_t lim = []() -> size_t {
    int dev = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&dev) != hipSuccess ||
        hipGetDeviceProperties(&prop, dev) != hipSuccess ||
        prop.sharedMemPerBlock <= 2 * 1024) {
      return kStreamSmemCap;
    }
    const size_t d = (size_t)prop.sharedMemPerBlock - 1024;  // LUT + slack
    return d < kStreamSmemCap ? d : kStreamSmemCap;
  }();
  return lim;
}

template <int MS, int RW, typename T>
__global__ void fp8_emu_gemv_stream_block_kernel(
    const uint8_t* __restrict__ a,      // [M, K] fp8 (a16 == 0) or fp16 (a16)
    const uint8_t* __restrict__ w,      // [N, K], row stride w_stride
    const float* __restrict__ a_s,      // [M, nkb]
    const float* __restrict__ w_s,      // [cdiv(N, block_n), nkb]
    const __half* __restrict__ lut,     // [256]
    const T* __restrict__ bias,         // [N] | null
    T* __restrict__ out,                // [M, N]
    const int M, const int N, const int K, const int64_t w_stride,
    const int rows, const int nkb, const int block_n, const int cpb,
    const int a16) {
  // Dynamic LDS layout: the [MS][kStreamSlice] fp16 A slice first (on the
  // __align__(16) dynamic base, so both the uint4 staging and the uint4 MAC
  // reads stay 16 B aligned), then the [M][nkb] folded scales.  The scales are
  // staged once per block: M*nkb*4 bytes (1.3 KiB at the production
  // block_k = 128), independent of the slice loop.
  extern __shared__ __align__(16) uint32_t dyn[];
  __half* sA = reinterpret_cast<__half*>(dyn);  // [MS][kStreamSlice]
  float* sscale =
      reinterpret_cast<float*>(sA + (int64_t)MS * kStreamSlice);  // [M][nkb]
  __shared__ __half sl[256];

  constexpr int kSliceChunks = kStreamSlice / kStreamChunk;  // W uint4 / slice
  constexpr int kSliceDw = kStreamSlice / 4;   // fp8 dwords per slice row
  constexpr int kSliceV8 = kStreamSlice / 8;   // fp16 uint4 per slice row (a16)
  static_assert(kSliceChunks == 64,
                "one W uint4 per lane per (row, slice): the prefetch pipeline "
                "below relies on it");

  const int t = threadIdx.x;
  const int nt = blockDim.x;
  const int lane = t & 63;
  const int warp = t >> 6;
  const int nwarp = nt >> 6;
  const int K4 = K >> 2;

  fp8_lut_stage(sl, lut, nt);
  __syncthreads();  // also orders the LUT store above

  const int n0 = blockIdx.x * rows;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);

  // One accumulator bank per owned row: the slice-outer walk keeps all RW
  // rows of the warp in flight simultaneously, so their partial sums cannot
  // share registers (the row-outer old kernel restarted acc[] per row).
  float acc[RW][MS];
#pragma unroll
  for (int rr = 0; rr < RW; ++rr) {
#pragma unroll
    for (int m = 0; m < MS; ++m) acc[rr][m] = 0.0f;
  }

  {  // Folded per-block_k scales: rows divides block_n, so one w_s row.
    const int nb = (int)(((int64_t)n0) / block_n);
    const float* __restrict__ wsr = w_s + (int64_t)nb * nkb;
    const int tot = M * nkb;
    for (int i = t; i < tot; i += nt) {
      const int m = i / nkb;
      const int kb = i - m * nkb;
      sscale[i] = (a16 != 0) ? wsr[kb] : a_s[(int64_t)m * nkb + kb] * wsr[kb];
    }
  }

  const uint32_t* __restrict__ a32 = reinterpret_cast<const uint32_t*>(a);
  // W chunk pipeline: the chunk a lane will dot in slice s is loaded two
  // slices ahead into rotating registers.  The slice loop cannot hoist the
  // global loads across its __syncthreads() on its own, so without this the
  // ~600-cycle DRAM latency sits exposed behind the staging barrier of every
  // slice (M=1 gateup: 441 -> 494 GB/s with the pipeline).  The slice loop is
  // peeled by two below so the buffer subscript is a literal -- a runtime
  // `pf[pre & 1]` spills to scratch and measured ~2x slower than no pipeline
  // at all (M=1: 241 GB/s).  kSliceChunks == 64 makes the (row, chunk) ->
  // lane mapping 1:1, so one uint4 per register slot per row is exact.
  uint4 pf[2][RW];
#pragma unroll
  for (int pre = 0; pre < 2; ++pre) {
    const int sp = pre * kStreamSlice;
#pragma unroll
    for (int rr = 0; rr < RW; ++rr) {
      const int row = n0 + warp + rr * nwarp;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (row < N && sp + lane * kStreamChunk < K) {
        v = wv[(((int64_t)row * w_stride) >> 4) + (sp >> 4) + lane];
      }
      pf[pre][rr] = v;
    }
  }

  // One slice body: stage A[s0..], consume pf[B] (slice s0's chunks), refill
  // pf[B] with slice REFILL's chunks.  Both __syncthreads() sit at slice
  // granularity, outside every warp-divergent branch, so the out-of-range
  // rows of the last block only skip compute.
#define VLLM_FP8_STREAM_SLICE(S0, B, REFILL)                                 \
  do {                                                                       \
    const int s0_ = (S0);                                                    \
    const int sl_el_ = min(kStreamSlice, K - s0_);                           \
    const int chunks_ = min(kSliceChunks, sl_el_ >> 4);                      \
    if (a16 != 0) {  /* W8A16: A is already fp16, only copy it. */           \
      const uint4* __restrict__ av = reinterpret_cast<const uint4*>(a);      \
      uint4* __restrict__ sv = reinterpret_cast<uint4*>(sA);                 \
      const int full_ = sl_el_ >> 3;                                         \
      for (int i = t; i < M * kSliceV8; i += nt) {                           \
        const int m = i / kSliceV8;                                          \
        const int v = i - m * kSliceV8;                                      \
        uint4 val = make_uint4(0u, 0u, 0u, 0u);                              \
        if (v < full_) val = av[m * (K >> 3) + (s0_ >> 3) + v];              \
        sv[m * kSliceV8 + v] = val;                                          \
      }                                                                      \
    } else {  /* each fp8 dword widens once per block, as before. */         \
      const int full_ = sl_el_ >> 2;                                         \
      uint32_t* __restrict__ sv = reinterpret_cast<uint32_t*>(sA);           \
      for (int i = t; i < M * kSliceDw; i += nt) {                           \
        const int m = i / kSliceDw;                                          \
        const int d = i - m * kSliceDw;                                      \
        uint32_t lo = 0, hi = 0;                                             \
        if (d < full_) {                                                     \
          fp8x4_to_f16x2x2(sl, a32[(int64_t)m * K4 + (s0_ >> 2) + d], lo, hi); \
        }                                                                    \
        uint32_t* dst = sv + m * (kStreamSlice >> 1) + 2 * d;                \
        dst[0] = lo;                                                         \
        dst[1] = hi;                                                         \
      }                                                                      \
    }                                                                        \
    __syncthreads();                                                         \
    _Pragma("unroll")                                                        \
    for (int rr = 0; rr < RW; ++rr) {                                        \
      const int row = n0 + warp + rr * nwarp;                                \
      const uint4 vv = pf[B][rr];                                            \
      /* Refill this buffer for slice REFILL (its next use).  The load is */ \
      /* independent of sA, so it issues before this slice's compute. */     \
      uint4 vn = make_uint4(0u, 0u, 0u, 0u);                                 \
      const int s2_ = (REFILL);                                              \
      if (s2_ < K && row < N) {                                              \
        const int chunks2 = min(kSliceChunks, (K - s2_) >> 4);               \
        if (lane < chunks2) {                                                \
          vn = wv[(((int64_t)row * w_stride) >> 4) + (s2_ >> 4) + lane];     \
        }                                                                    \
      }                                                                      \
      pf[B][rr] = vn;                                                        \
      if (row < N && lane < chunks_) {  /* no barrier behind it */           \
        uint32_t p[8];                                                       \
        fp8x4_to_f16x2x2(sl, vv.x, p[0], p[1]);                              \
        fp8x4_to_f16x2x2(sl, vv.y, p[2], p[3]);                              \
        fp8x4_to_f16x2x2(sl, vv.z, p[4], p[5]);                              \
        fp8x4_to_f16x2x2(sl, vv.w, p[6], p[7]);                              \
        const int kb = ((s0_ >> 4) + lane) / cpb;                            \
        /* P1: the widened chunk feeds the rows in groups of four, so one */ \
        /* chunk's 8 W registers cover the whole M range instead of one */   \
        /* accumulator group per pass (the M >= 5 instantiation runs the */  \
        /* two groups back to back on the same p[]). */                      \
        _Pragma("unroll")                                                    \
        for (int g = 0; g < (MS + 3) / 4; ++g) {                             \
          _Pragma("unroll")                                                  \
          for (int m = 4 * g; m < (4 * g + 4 < MS ? 4 * g + 4 : MS); ++m) {  \
            if (m < M) {                                                     \
              const uint4* __restrict__ ap =                                 \
                  reinterpret_cast<const uint4*>(                            \
                      sA + (int64_t)m * kStreamSlice +                       \
                      lane * kStreamChunk);                                  \
              const uint4 xa = ap[0];                                        \
              const uint4 xb = ap[1];                                        \
              float loc = 0.0f;                                              \
              loc = dot2(p[0], xa.x, loc);                                   \
              loc = dot2(p[1], xa.y, loc);                                   \
              loc = dot2(p[2], xa.z, loc);                                   \
              loc = dot2(p[3], xa.w, loc);                                   \
              loc = dot2(p[4], xb.x, loc);                                   \
              loc = dot2(p[5], xb.y, loc);                                   \
              loc = dot2(p[6], xb.z, loc);                                   \
              loc = dot2(p[7], xb.w, loc);                                   \
              acc[rr][m] += loc * sscale[m * nkb + kb];                      \
            }                                                                \
          }                                                                  \
        }                                                                    \
      }                                                                      \
    }                                                                        \
    __syncthreads();                                                         \
  } while (0)

  // K walk, peeled by two slices so pf's buffer index is a literal.  Invariant
  // on entry to a body: pf[B] holds slice (S0) iff (S0 / kStreamSlice) % 2 == B.
  int s0 = 0;
  for (; s0 + 2 * kStreamSlice <= K; s0 += 2 * kStreamSlice) {
    VLLM_FP8_STREAM_SLICE(s0, 0, s0 + 2 * kStreamSlice);
    VLLM_FP8_STREAM_SLICE(s0 + kStreamSlice, 1, s0 + 3 * kStreamSlice);
  }
  if (s0 < K) {
    VLLM_FP8_STREAM_SLICE(s0, 0, s0 + 2 * kStreamSlice);
    s0 += kStreamSlice;
    if (s0 < K) VLLM_FP8_STREAM_SLICE(s0, 1, s0 + 2 * kStreamSlice);
  }
#undef VLLM_FP8_STREAM_SLICE

  // Cross-lane reduction; lane 0 owns the row.  The K walk reconverged above,
  // and `row >= N` is warp-uniform.
#pragma unroll
  for (int rr = 0; rr < RW; ++rr) {
    const int row = n0 + warp + rr * nwarp;
    if (row >= N) continue;
#pragma unroll
    for (int m = 0; m < MS; ++m) {
      if (m < M) {
        float v = acc[rr][m];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off);
        if (lane == 0) {
          T o = static_cast<T>(v);
          if (bias != nullptr) o += static_cast<T>(bias[row]);
          out[(int64_t)m * N + row] = o;
        }
      }
    }
  }
}

// Highest MS instantiation `dispatch_gemv_stream` will use for this M.  The
// slice buffer is sized by the template parameter, so the host-side LDS gate
// has to match it.
int stream_ms_for(int M) {
  if (M <= 1) return 1;
  if (M <= 2) return 2;
  if (M <= 4) return 4;
  return kMsMax;
}

// ---------------------------------------------------------------------------
// S30c WP2-N level 2: M == 1 specialization of the streaming decode kernel.
// ---------------------------------------------------------------------------
// Same slice/pipeline geometry as `fp8_emu_gemv_stream_block_kernel`, with the
// M dimension stripped: A is one row (the W8A16 route's fp16 activation), the
// folded-scale table is [nkb] (no [M][nkb] 2-D staging or a_s multiply), the
// MAC loop carries one accumulator per owned row (no [MS] bank, no `m < M`
// predication, no per-M register groups), and the A-stage loop copies a
// single 2 KiB fp16 slice.  Frees the registers the [RW][MS] accumulator bank
// used and removes the M-dim LDS indexing from the inner dot loop.
// Gate: VLLM_DCU_FP8_GEMV_M1 (default 0; W8A16 dispatch, M == 1 only).
template <int RW, typename T>
__global__ void fp8_emu_gemv_stream_m1_kernel(
    const uint8_t* __restrict__ a,      // [1, K] fp16 (W8A16 route)
    const uint8_t* __restrict__ w,      // [N, K], row stride w_stride
    const float* __restrict__ w_s,      // [cdiv(N, block_n), nkb]
    const __half* __restrict__ lut,     // [256]
    const T* __restrict__ bias,         // [N] | null
    T* __restrict__ out,                // [1, N]
    const int N, const int K, const int64_t w_stride, const int rows,
    const int nkb, const int block_n, const int cpb) {
  extern __shared__ __align__(16) uint32_t dyn[];
  __half* sA = reinterpret_cast<__half*>(dyn);  // [kStreamSlice]
  __shared__ __half sl[256];

  constexpr int kSliceChunks = kStreamSlice / kStreamChunk;  // W uint4 / slice
  constexpr int kSliceV8 = kStreamSlice / 8;                 // fp16 uint4/slice
  static_assert(kSliceChunks == 64,
                "one W uint4 per lane per (row, slice): the prefetch pipeline "
                "below relies on it");

  const int t = threadIdx.x;
  const int nt = blockDim.x;
  const int lane = t & 63;
  const int warp = t >> 6;
  const int nwarp = nt >> 6;

  fp8_lut_stage(sl, lut, nt);
  __syncthreads();

  const int n0 = blockIdx.x * rows;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);

  float acc[RW];
#pragma unroll
  for (int rr = 0; rr < RW; ++rr) acc[rr] = 0.0f;

  {  // Folded per-block_k scales: rows divides block_n, so one w_s row.
    const int nb = (int)(((int64_t)n0) / block_n);
    const float* __restrict__ wsr = w_s + (int64_t)nb * nkb;
    float* sscale = reinterpret_cast<float*>(sA + kStreamSlice);  // [nkb]
    for (int kb = t; kb < nkb; kb += nt) sscale[kb] = wsr[kb];
  }

  const uint4* __restrict__ av = reinterpret_cast<const uint4*>(a);
  uint4* __restrict__ sv = reinterpret_cast<uint4*>(sA);

  // W chunk pipeline, identical to variant 1 (two slices ahead, literal
  // buffer subscript via the peel-by-two slice loop).
  uint4 pf[2][RW];
#pragma unroll
  for (int pre = 0; pre < 2; ++pre) {
    const int sp = pre * kStreamSlice;
#pragma unroll
    for (int rr = 0; rr < RW; ++rr) {
      const int row = n0 + warp + rr * nwarp;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (row < N && sp + lane * kStreamChunk < K) {
        v = wv[(((int64_t)row * w_stride) >> 4) + (sp >> 4) + lane];
      }
      pf[pre][rr] = v;
    }
  }

#define VLLM_FP8_STREAM_M1_SLICE(S0, B, REFILL)                              \
  do {                                                                       \
    const int s0_ = (S0);                                                    \
    const int sl_el_ = min(kStreamSlice, K - s0_);                           \
    const int chunks_ = min(kSliceChunks, sl_el_ >> 4);                      \
    /* A is one fp16 row: copy this slice's uint4s, no M loop. */            \
    const int full_ = sl_el_ >> 3;                                           \
    for (int v = t; v < kSliceV8; v += nt) {                                 \
      uint4 val = make_uint4(0u, 0u, 0u, 0u);                                \
      if (v < full_) val = av[(s0_ >> 3) + v];                               \
      sv[v] = val;                                                           \
    }                                                                        \
    __syncthreads();                                                         \
    _Pragma("unroll")                                                        \
    for (int rr = 0; rr < RW; ++rr) {                                        \
      const int row = n0 + warp + rr * nwarp;                                \
      const uint4 vv = pf[B][rr];                                            \
      uint4 vn = make_uint4(0u, 0u, 0u, 0u);                                 \
      const int s2_ = (REFILL);                                              \
      if (s2_ < K && row < N) {                                              \
        const int chunks2 = min(kSliceChunks, (K - s2_) >> 4);               \
        if (lane < chunks2) {                                                \
          vn = wv[(((int64_t)row * w_stride) >> 4) + (s2_ >> 4) + lane];     \
        }                                                                    \
      }                                                                      \
      pf[B][rr] = vn;                                                        \
      if (row < N && lane < chunks_) {                                       \
        uint32_t p[8];                                                       \
        fp8x4_to_f16x2x2(sl, vv.x, p[0], p[1]);                              \
        fp8x4_to_f16x2x2(sl, vv.y, p[2], p[3]);                              \
        fp8x4_to_f16x2x2(sl, vv.z, p[4], p[5]);                              \
        fp8x4_to_f16x2x2(sl, vv.w, p[6], p[7]);                              \
        const int kb = ((s0_ >> 4) + lane) / cpb;                            \
        const uint4* __restrict__ ap =                                       \
            reinterpret_cast<const uint4*>(sA + lane * kStreamChunk);        \
        const uint4 xa = ap[0];                                              \
        const uint4 xb = ap[1];                                              \
        float loc = 0.0f;                                                    \
        loc = dot2(p[0], xa.x, loc);                                         \
        loc = dot2(p[1], xa.y, loc);                                         \
        loc = dot2(p[2], xa.z, loc);                                         \
        loc = dot2(p[3], xa.w, loc);                                         \
        loc = dot2(p[4], xb.x, loc);                                         \
        loc = dot2(p[5], xb.y, loc);                                         \
        loc = dot2(p[6], xb.z, loc);                                         \
        loc = dot2(p[7], xb.w, loc);                                         \
        acc[rr] += loc * reinterpret_cast<const float*>(                     \
                       sA + kStreamSlice)[kb];                               \
      }                                                                      \
    }                                                                        \
    __syncthreads();                                                         \
  } while (0)

  int s0 = 0;
  for (; s0 + 2 * kStreamSlice <= K; s0 += 2 * kStreamSlice) {
    VLLM_FP8_STREAM_M1_SLICE(s0, 0, s0 + 2 * kStreamSlice);
    VLLM_FP8_STREAM_M1_SLICE(s0 + kStreamSlice, 1, s0 + 3 * kStreamSlice);
  }
  if (s0 < K) {
    VLLM_FP8_STREAM_M1_SLICE(s0, 0, s0 + 2 * kStreamSlice);
    s0 += kStreamSlice;
    if (s0 < K) VLLM_FP8_STREAM_M1_SLICE(s0, 1, s0 + 2 * kStreamSlice);
  }
#undef VLLM_FP8_STREAM_M1_SLICE

#pragma unroll
  for (int rr = 0; rr < RW; ++rr) {
    const int row = n0 + warp + rr * nwarp;
    if (row >= N) continue;
    float v = acc[rr];
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off);
    if (lane == 0) {
      T o = static_cast<T>(v);
      if (bias != nullptr) o += static_cast<T>(bias[row]);
      out[row] = o;
    }
  }
}

template <int RW, typename T>
void launch_gemv_stream_m1(int N, int K, int64_t w_stride, int rows, int nt,
                           int nkb, int block_n, int cpb, const uint8_t* A,
                           const uint8_t* W, const float* w_s,
                           const __half* lut, const T* bias, T* C,
                           hipStream_t stream) {
  // One A slice (fp16) + the [nkb] folded scales.
  const size_t smem = (size_t)kStreamSlice * sizeof(__half) +
                      (size_t)nkb * sizeof(float);
  dim3 grid((N + rows - 1) / rows);
  dim3 block(nt);
  fp8_emu_gemv_stream_m1_kernel<RW, T><<<grid, block, smem, stream>>>(
      A, W, w_s, lut, bias, C, N, K, w_stride, rows, nkb, block_n, cpb);
}

// M == 1 dispatch of the m1 kernel; rows/nt come from `stream_block_select`
// (the P2 rows override keeps working), so RW mirrors dispatch_gemv_stream.
template <typename T>
void dispatch_gemv_stream_m1(int N, int K, int64_t w_stride, int rows, int nt,
                             int nkb, int block_n, int cpb, const uint8_t* A,
                             const uint8_t* W, const float* w_s,
                             const __half* lut, const T* bias, T* C,
                             hipStream_t stream) {
  const int rw = rows / (nt >> 6);
  if (rw <= 1) {
    launch_gemv_stream_m1<1, T>(N, K, w_stride, rows, nt, nkb, block_n, cpb, A,
                                W, w_s, lut, bias, C, stream);
  } else if (rw == 2) {
    launch_gemv_stream_m1<2, T>(N, K, w_stride, rows, nt, nkb, block_n, cpb, A,
                                W, w_s, lut, bias, C, stream);
  } else if (rw <= 4) {
    launch_gemv_stream_m1<4, T>(N, K, w_stride, rows, nt, nkb, block_n, cpb, A,
                                W, w_s, lut, bias, C, stream);
  } else {
    launch_gemv_stream_m1<8, T>(N, K, w_stride, rows, nt, nkb, block_n, cpb, A,
                                W, w_s, lut, bias, C, stream);
  }
}

// S30c WP2-N level-2 gate (default 0; flipped to 1 after parity).
int stream_m1_enabled() {
  static const bool on = []() {
    const char* e = std::getenv("VLLM_DCU_FP8_GEMV_M1");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return on;
}

// ---------------------------------------------------------------------------
// S33 WP1: whole-A M == 1 GEMV (numeric bit-identical to the production path).
// ---------------------------------------------------------------------------
// Stages the ENTIRE fp16 activation row (K * 2 B = 10 KB at the production
// K = 5120) plus the folded [nkb] scales into dynamic LDS once, then walks the
// K slices with zero __syncthreads() inside the loop (the m1 kernel pays two
// barriers per 1024-element slice for its A restaging).  Slice/lane/chunk
// traversal order, the widened-chunk dot2 chain, the kb folding and the
// acc += order are line-for-line those of fp8_emu_gemv_stream_m1_kernel, and
// the A values fed to the MACs are the same fp16 bits (a plain copy either
// way), so the results are bit-identical to the production kernel (WP1 判据 B:
// microbench max|delta| identical to the current path, 0-difference level).
// The W prefetch ring is deepened to 3 buffers -- there is no staging barrier
// left to hide the global loads behind (the m1 kernel prefetches 2 slices
// ahead across its per-slice barriers).
//
// Shape gate: the whole-A row must stay small enough that the grid still
// fills the CUs -- down (K = 17408 -> 34 KB row -> 1 block/CU) is excluded and
// stays on the production path (design D2).  qkv/o/GDN-in/GDN-out/gateup
// (K = 5120 -> 10.4 KB LDS) keep 4-5 blocks per CU.
//
// Gate: VLLM_DCU_FP8_GEMV_WA (default 0; W8A16 dispatch, M == 1 only).
constexpr size_t kWholeaMaxLds = (size_t)12 * 1024;

bool wholea_shape_ok(int K, int nkb) {
  const size_t smem = (size_t)K * sizeof(__half) + (size_t)nkb * sizeof(float);
  return smem <= kWholeaMaxLds && smem <= stream_smem_limit();
}

int wholea_enabled() {
  static const bool on = []() {
    const char* e = std::getenv("VLLM_DCU_FP8_GEMV_WA");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return on;
}

template <int RW, typename T>
__global__ void fp8_emu_gemv_wholea_m1_kernel(
    const uint8_t* __restrict__ a,      // [1, K] fp16 (W8A16 route)
    const uint8_t* __restrict__ w,      // [N, K], row stride w_stride
    const float* __restrict__ w_s,      // [cdiv(N, block_n), nkb]
    const __half* __restrict__ lut,     // [256]
    const T* __restrict__ bias,         // [N] | null
    T* __restrict__ out,                // [1, N]
    const int N, const int K, const int64_t w_stride, const int rows,
    const int nkb, const int block_n, const int cpb) {
  extern __shared__ __align__(16) uint32_t dyn[];
  __half* sA = reinterpret_cast<__half*>(dyn);         // [K] whole A row
  float* sscale = reinterpret_cast<float*>(sA + K);    // [nkb]
  __shared__ __half sl[256];

  constexpr int kSliceChunks = kStreamSlice / kStreamChunk;  // W uint4 / slice
  constexpr int kPf = 3;  // W prefetch ring depth (no staging barrier left)
  static_assert(kSliceChunks == 64,
                "one W uint4 per lane per (row, slice): the traversal order "
                "below relies on it");

  const int t = threadIdx.x;
  const int nt = blockDim.x;
  const int lane = t & 63;
  const int warp = t >> 6;
  const int nwarp = nt >> 6;

  // One-time staging: LUT + whole A row + folded scales, then ONE barrier.
  fp8_lut_stage(sl, lut, nt);
  {
    const uint4* __restrict__ av = reinterpret_cast<const uint4*>(a);
    uint4* __restrict__ sv = reinterpret_cast<uint4*>(sA);
    const int kv = K >> 3;  // fp16 uint4s in the row (stream gate: K % 16 == 0)
    for (int v = t; v < kv; v += nt) sv[v] = av[v];
    const int n0 = blockIdx.x * rows;
    const int nb = (int)(((int64_t)n0) / block_n);
    const float* __restrict__ wsr = w_s + (int64_t)nb * nkb;
    for (int kb = t; kb < nkb; kb += nt) sscale[kb] = wsr[kb];
  }
  __syncthreads();  // the only barrier after staging

  const int n0 = blockIdx.x * rows;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);

  float acc[RW];
#pragma unroll
  for (int rr = 0; rr < RW; ++rr) acc[rr] = 0.0f;

  // W chunk pipeline, kPf slices ahead; literal buffer subscripts come from
  // the peel-by-kPf slice loop below (the m1 kernel's peel-by-two trick).
  uint4 pf[kPf][RW];
#pragma unroll
  for (int pre = 0; pre < kPf; ++pre) {
    const int sp = pre * kStreamSlice;
#pragma unroll
    for (int rr = 0; rr < RW; ++rr) {
      const int row = n0 + warp + rr * nwarp;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (row < N && sp + lane * kStreamChunk < K) {
        v = wv[(((int64_t)row * w_stride) >> 4) + (sp >> 4) + lane];
      }
      pf[pre][rr] = v;
    }
  }

  // One slice body: consume pf[B] (slice S0's chunks), refill pf[B] with slice
  // REFILL.  Identical arithmetic to VLLM_FP8_STREAM_M1_SLICE -- the only
  // difference is where the A fp16 values come from (staged whole row).
#define VLLM_FP8_WHOLEA_M1_SLICE(S0, B, REFILL)                              \
  do {                                                                       \
    const int s0_ = (S0);                                                    \
    const int chunks_ = min(kSliceChunks, (K - s0_) >> 4);                   \
    _Pragma("unroll")                                                        \
    for (int rr = 0; rr < RW; ++rr) {                                        \
      const int row = n0 + warp + rr * nwarp;                                \
      const uint4 vv = pf[B][rr];                                            \
      uint4 vn = make_uint4(0u, 0u, 0u, 0u);                                 \
      const int s2_ = (REFILL);                                              \
      if (s2_ < K && row < N) {                                              \
        const int chunks2 = min(kSliceChunks, (K - s2_) >> 4);               \
        if (lane < chunks2) {                                                \
          vn = wv[(((int64_t)row * w_stride) >> 4) + (s2_ >> 4) + lane];     \
        }                                                                    \
      }                                                                      \
      pf[B][rr] = vn;                                                        \
      if (row < N && lane < chunks_) {                                       \
        uint32_t p[8];                                                       \
        fp8x4_to_f16x2x2(sl, vv.x, p[0], p[1]);                              \
        fp8x4_to_f16x2x2(sl, vv.y, p[2], p[3]);                              \
        fp8x4_to_f16x2x2(sl, vv.z, p[4], p[5]);                              \
        fp8x4_to_f16x2x2(sl, vv.w, p[6], p[7]);                              \
        const int kb = ((s0_ >> 4) + lane) / cpb;                            \
        /* Whole-row LDS is absolute: this slice's A chunk sits at s0_. */   \
        const uint4* __restrict__ ap =                                       \
            reinterpret_cast<const uint4*>(sA + s0_ + lane * kStreamChunk);  \
        const uint4 xa = ap[0];                                              \
        const uint4 xb = ap[1];                                              \
        float loc = 0.0f;                                                    \
        loc = dot2(p[0], xa.x, loc);                                         \
        loc = dot2(p[1], xa.y, loc);                                         \
        loc = dot2(p[2], xa.z, loc);                                         \
        loc = dot2(p[3], xa.w, loc);                                         \
        loc = dot2(p[4], xb.x, loc);                                         \
        loc = dot2(p[5], xb.y, loc);                                         \
        loc = dot2(p[6], xb.z, loc);                                         \
        loc = dot2(p[7], xb.w, loc);                                         \
        acc[rr] += loc * sscale[kb];                                         \
      }                                                                      \
    }                                                                        \
  } while (0)

  // K walk, peeled by kPf slices so pf's buffer index is a literal.  Invariant
  // on entry to a body: pf[B] holds slice S0 iff (S0 / kStreamSlice) % kPf==B.
  int s0 = 0;
  for (; s0 + kPf * kStreamSlice <= K; s0 += kPf * kStreamSlice) {
    VLLM_FP8_WHOLEA_M1_SLICE(s0, 0, s0 + kPf * kStreamSlice);
    VLLM_FP8_WHOLEA_M1_SLICE(s0 + kStreamSlice, 1,
                             s0 + (kPf + 1) * kStreamSlice);
    VLLM_FP8_WHOLEA_M1_SLICE(s0 + 2 * kStreamSlice, 2,
                             s0 + (kPf + 2) * kStreamSlice);
  }
  if (s0 < K) {
    VLLM_FP8_WHOLEA_M1_SLICE(s0, 0, s0 + kPf * kStreamSlice);
    s0 += kStreamSlice;
    if (s0 < K) {
      VLLM_FP8_WHOLEA_M1_SLICE(s0, 1, s0 + kPf * kStreamSlice);
      s0 += kStreamSlice;
      if (s0 < K) VLLM_FP8_WHOLEA_M1_SLICE(s0, 2, s0 + kPf * kStreamSlice);
    }
  }
#undef VLLM_FP8_WHOLEA_M1_SLICE

  // Cross-lane reduction; lane 0 owns the row.  Identical to the m1 kernel.
#pragma unroll
  for (int rr = 0; rr < RW; ++rr) {
    const int row = n0 + warp + rr * nwarp;
    if (row >= N) continue;
    float v = acc[rr];
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off);
    if (lane == 0) {
      T o = static_cast<T>(v);
      if (bias != nullptr) o += static_cast<T>(bias[row]);
      out[row] = o;
    }
  }
}

// S33 WP1 fix attempt: L2-direct variant of the whole-A kernel.  The LDS
// variant's A reads are structurally 4-way bank-conflicted (32 B lane stride,
// 16 B-aligned b128); reading the whole A row straight from GLOBAL (L2-hot
// across the grid, perfectly coalesced 2 KB per 64-lane wave) removes the LDS
// traffic entirely.  Same values, same slice/lane/chunk order, same dot2
// chain -> bit-identical results.  LDS now holds only the folded scales.
// Gate: VLLM_DCU_FP8_GEMV_WA_L2=1 (with VLLM_DCU_FP8_GEMV_WA=1).
static bool wholea_l2_enabled() {
  static const bool on = []() {
    const char* e = std::getenv("VLLM_DCU_FP8_GEMV_WA_L2");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return on;
}

template <int RW, typename T>
__global__ void fp8_emu_gemv_wholea_l2_m1_kernel(
    const uint8_t* __restrict__ a,      // [1, K] fp16 (W8A16 route)
    const uint8_t* __restrict__ w,      // [N, K], row stride w_stride
    const float* __restrict__ w_s,      // [cdiv(N, block_n), nkb]
    const __half* __restrict__ lut,     // [256]
    const T* __restrict__ bias,         // [N] | null
    T* __restrict__ out,                // [1, N]
    const int N, const int K, const int64_t w_stride, const int rows,
    const int nkb, const int block_n, const int cpb) {
  extern __shared__ __align__(16) uint32_t dyn[];
  float* sscale = reinterpret_cast<float*>(dyn);  // [nkb]
  __shared__ __half sl[256];

  constexpr int kSliceChunks = kStreamSlice / kStreamChunk;  // W uint4 / slice
  constexpr int kPf = 3;
  static_assert(kSliceChunks == 64,
                "one W uint4 per lane per (row, slice): the traversal order "
                "below relies on it");

  const int t = threadIdx.x;
  const int nt = blockDim.x;
  const int lane = t & 63;
  const int warp = t >> 6;
  const int nwarp = nt >> 6;

  // One-time staging: LUT + folded scales, then ONE barrier.
  fp8_lut_stage(sl, lut, nt);
  {
    const int n0 = blockIdx.x * rows;
    const int nb = (int)(((int64_t)n0) / block_n);
    const float* __restrict__ wsr = w_s + (int64_t)nb * nkb;
    for (int kb = t; kb < nkb; kb += nt) sscale[kb] = wsr[kb];
  }
  __syncthreads();  // the only barrier after staging

  const int n0 = blockIdx.x * rows;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);

  float acc[RW];
#pragma unroll
  for (int rr = 0; rr < RW; ++rr) acc[rr] = 0.0f;

  // W chunk pipeline, kPf slices ahead (literal subscripts via the peel).
  uint4 pf[kPf][RW];
#pragma unroll
  for (int pre = 0; pre < kPf; ++pre) {
    const int sp = pre * kStreamSlice;
#pragma unroll
    for (int rr = 0; rr < RW; ++rr) {
      const int row = n0 + warp + rr * nwarp;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (row < N && sp + lane * kStreamChunk < K) {
        v = wv[(((int64_t)row * w_stride) >> 4) + (sp >> 4) + lane];
      }
      pf[pre][rr] = v;
    }
  }

  // One slice body: identical arithmetic to VLLM_FP8_WHOLEA_M1_SLICE, except
  // the A fp16 values come straight from global (L2-hot) instead of LDS.
#define VLLM_FP8_WHOLEA_L2_M1_SLICE(S0, B, REFILL)                           \
  do {                                                                       \
    const int s0_ = (S0);                                                    \
    const int chunks_ = min(kSliceChunks, (K - s0_) >> 4);                   \
    _Pragma("unroll")                                                        \
    for (int rr = 0; rr < RW; ++rr) {                                        \
      const int row = n0 + warp + rr * nwarp;                                \
      const uint4 vv = pf[B][rr];                                            \
      uint4 vn = make_uint4(0u, 0u, 0u, 0u);                                 \
      const int s2_ = (REFILL);                                              \
      if (s2_ < K && row < N) {                                              \
        const int chunks2 = min(kSliceChunks, (K - s2_) >> 4);               \
        if (lane < chunks2) {                                                \
          vn = wv[(((int64_t)row * w_stride) >> 4) + (s2_ >> 4) + lane];     \
        }                                                                    \
      }                                                                      \
      pf[B][rr] = vn;                                                        \
      if (row < N && lane < chunks_) {                                       \
        uint32_t p[8];                                                       \
        fp8x4_to_f16x2x2(sl, vv.x, p[0], p[1]);                              \
        fp8x4_to_f16x2x2(sl, vv.y, p[2], p[3]);                              \
        fp8x4_to_f16x2x2(sl, vv.z, p[4], p[5]);                              \
        fp8x4_to_f16x2x2(sl, vv.w, p[6], p[7]);                              \
        const int kb = ((s0_ >> 4) + lane) / cpb;                            \
        /* A chunk read straight from global (L2-hot, coalesced) */          \
        const uint4* __restrict__ ap =                                       \
            reinterpret_cast<const uint4*>(a) + (s0_ >> 3) + lane * 2;       \
        const uint4 xa = ap[0];                                              \
        const uint4 xb = ap[1];                                              \
        float loc = 0.0f;                                                    \
        loc = dot2(p[0], xa.x, loc);                                         \
        loc = dot2(p[1], xa.y, loc);                                         \
        loc = dot2(p[2], xa.z, loc);                                         \
        loc = dot2(p[3], xa.w, loc);                                         \
        loc = dot2(p[4], xb.x, loc);                                         \
        loc = dot2(p[5], xb.y, loc);                                         \
        loc = dot2(p[6], xb.z, loc);                                         \
        loc = dot2(p[7], xb.w, loc);                                         \
        acc[rr] += loc * sscale[kb];                                         \
      }                                                                      \
    }                                                                        \
  } while (0)

  int s0 = 0;
  for (; s0 + kPf * kStreamSlice <= K; s0 += kPf * kStreamSlice) {
    VLLM_FP8_WHOLEA_L2_M1_SLICE(s0, 0, s0 + kPf * kStreamSlice);
    VLLM_FP8_WHOLEA_L2_M1_SLICE(s0 + kStreamSlice, 1,
                                s0 + (kPf + 1) * kStreamSlice);
    VLLM_FP8_WHOLEA_L2_M1_SLICE(s0 + 2 * kStreamSlice, 2,
                                s0 + (kPf + 2) * kStreamSlice);
  }
  if (s0 < K) {
    VLLM_FP8_WHOLEA_L2_M1_SLICE(s0, 0, s0 + kPf * kStreamSlice);
    s0 += kStreamSlice;
    if (s0 < K) {
      VLLM_FP8_WHOLEA_L2_M1_SLICE(s0, 1, s0 + kPf * kStreamSlice);
      s0 += kStreamSlice;
      if (s0 < K) VLLM_FP8_WHOLEA_L2_M1_SLICE(s0, 2, s0 + kPf * kStreamSlice);
    }
  }
#undef VLLM_FP8_WHOLEA_L2_M1_SLICE

#pragma unroll
  for (int rr = 0; rr < RW; ++rr) {
    const int row = n0 + warp + rr * nwarp;
    if (row >= N) continue;
    float v = acc[rr];
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off);
    if (lane == 0) {
      T o = static_cast<T>(v);
      if (bias != nullptr) o += static_cast<T>(bias[row]);
      out[row] = o;
    }
  }
}

template <int RW, typename T>
void launch_gemv_wholea_l2_m1(int N, int K, int64_t w_stride, int rows, int nt,
                              int nkb, int block_n, int cpb, const uint8_t* A,
                              const uint8_t* W, const float* w_s,
                              const __half* lut, const T* bias, T* C,
                              hipStream_t stream) {
  const size_t smem = (size_t)nkb * sizeof(float);
  dim3 grid((N + rows - 1) / rows);
  dim3 block(nt);
  fp8_emu_gemv_wholea_l2_m1_kernel<RW, T><<<grid, block, smem, stream>>>(
      A, W, w_s, lut, bias, C, N, K, w_stride, rows, nkb, block_n, cpb);
}

template <typename T>
void dispatch_gemv_wholea_l2_m1(int N, int K, int64_t w_stride, int rows,
                                int nt, int nkb, int block_n, int cpb,
                                const uint8_t* A, const uint8_t* W,
                                const float* w_s, const __half* lut,
                                const T* bias, T* C, hipStream_t stream) {
  const int rw = rows / (nt >> 6);
  if (rw <= 1) {
    launch_gemv_wholea_l2_m1<1, T>(N, K, w_stride, rows, nt, nkb, block_n,
                                   cpb, A, W, w_s, lut, bias, C, stream);
  } else if (rw == 2) {
    launch_gemv_wholea_l2_m1<2, T>(N, K, w_stride, rows, nt, nkb, block_n,
                                   cpb, A, W, w_s, lut, bias, C, stream);
  } else if (rw <= 4) {
    launch_gemv_wholea_l2_m1<4, T>(N, K, w_stride, rows, nt, nkb, block_n,
                                   cpb, A, W, w_s, lut, bias, C, stream);
  } else {
    launch_gemv_wholea_l2_m1<8, T>(N, K, w_stride, rows, nt, nkb, block_n,
                                   cpb, A, W, w_s, lut, bias, C, stream);
  }
}

template <int RW, typename T>
void launch_gemv_wholea_m1(int N, int K, int64_t w_stride, int rows, int nt,
                           int nkb, int block_n, int cpb, const uint8_t* A,
                           const uint8_t* W, const float* w_s,
                           const __half* lut, const T* bias, T* C,
                           hipStream_t stream) {
  // Whole fp16 A row + the [nkb] folded scales.
  const size_t smem = (size_t)K * sizeof(__half) + (size_t)nkb * sizeof(float);
  dim3 grid((N + rows - 1) / rows);
  dim3 block(nt);
  fp8_emu_gemv_wholea_m1_kernel<RW, T><<<grid, block, smem, stream>>>(
      A, W, w_s, lut, bias, C, N, K, w_stride, rows, nkb, block_n, cpb);
}

// M == 1 dispatch of the whole-A kernel; rows/nt come from
// `stream_block_select` (the P2 rows override keeps working), so RW mirrors
// dispatch_gemv_stream_m1.
template <typename T>
void dispatch_gemv_wholea_m1(int N, int K, int64_t w_stride, int rows, int nt,
                             int nkb, int block_n, int cpb, const uint8_t* A,
                             const uint8_t* W, const float* w_s,
                             const __half* lut, const T* bias, T* C,
                             hipStream_t stream) {
  const int rw = rows / (nt >> 6);
  if (rw <= 1) {
    launch_gemv_wholea_m1<1, T>(N, K, w_stride, rows, nt, nkb, block_n, cpb, A,
                                W, w_s, lut, bias, C, stream);
  } else if (rw == 2) {
    launch_gemv_wholea_m1<2, T>(N, K, w_stride, rows, nt, nkb, block_n, cpb, A,
                                W, w_s, lut, bias, C, stream);
  } else if (rw <= 4) {
    launch_gemv_wholea_m1<4, T>(N, K, w_stride, rows, nt, nkb, block_n, cpb, A,
                                W, w_s, lut, bias, C, stream);
  } else {
    launch_gemv_wholea_m1<8, T>(N, K, w_stride, rows, nt, nkb, block_n, cpb, A,
                                W, w_s, lut, bias, C, stream);
  }
}

template <int MS, int RW, typename T>
void launch_gemv_stream(int M, int N, int K, int64_t w_stride, int rows, int nt,
                        int nkb, int block_n, int cpb, bool a16,
                        const uint8_t* A, const uint8_t* W, const float* a_s,
                        const float* w_s, const __half* lut, const T* bias,
                        T* C, hipStream_t stream) {
  const size_t smem = (size_t)MS * kStreamSlice * sizeof(__half) +
                      (size_t)M * nkb * sizeof(float);
  dim3 grid((N + rows - 1) / rows);
  dim3 block(nt);
  fp8_emu_gemv_stream_block_kernel<MS, RW, T><<<grid, block, smem, stream>>>(
      A, W, a_s, w_s, lut, bias, C, M, N, K, w_stride, rows, nkb, block_n,
      cpb, a16 ? 1 : 0);
}

// S23 / O4: highest M that may still take the streaming decode kernel.
// Unset (or <= 0) means "no limit", i.e. today's dispatch untouched; a value V
// means "skip the stream path when M >= V", so `run_block`'s tile branch takes
// the call (with VLLM_DCU_FP8_DECODE_CFG picking the O3 tile).  Parsed once per
// process, same pattern as the knobs above, so an A/B needs no rebuild.
//
// Why a gate at all: on 5120:1536 (K = 1536) at M = 8 the stream kernel
// measures 92.1 us (86 GB/s) against 63.6 us (124 GB/s) for an O3 wide-BK tile,
// because the A slice -- and its LDS read amplification -- grows with M while
// the tiled GEMM is M-flat.  At M = 4 the same shape is 49.7 us on the stream
// path against ~63 us tiled, so the crossover sits between M = 4 and 8.
int stream_max_m() {
  static const int lim = []() {
    const char* e = std::getenv("VLLM_DCU_FP8_STREAM_MAX_M");
    if (e == nullptr) return 0;
    const int v = std::atoi(e);
    return v > 0 ? v : 0;
  }();
  return lim;
}

// P2 A/B knob: rows per block for the streaming kernel, overriding the auto
// rule in `stream_block_select` (0 = auto).  The slice kernel has no per-row
// software pipeline left, so the old prefetch knob
// (VLLM_DCU_FP8_GEMV_PREF, PREF=1/3) retired with it.
int stream_rows_override() {
  static const int v = []() {
    const char* e = std::getenv("VLLM_DCU_FP8_GEMV_STREAM_ROWS");
    if (e == nullptr) return 0;
    const int x = std::atoi(e);
    return (x == 4 || x == 8 || x == 16 || x == 32 || x == 64) ? x : 0;
  }();
  return v;
}

// Streaming path available *and* its geometry chosen.  Returns 0 = none,
// 1 = variant 1, 2 = variant 2.
//
// Variant 1: `rows` divides `block_n` (so a block never straddles a w_s row)
// and the grid still fills the 60 CUs; 512 threads is the block size the
// probe measured 621 GB/s with.  Its LDS budget is one A slice
// (`stream_ms_for(M)` * kStreamSlice fp16) plus the folded scales, so M = 8
// no longer falls out of the path at K = 5120 (the old whole-A tile was
// M*K*2 = 82 KiB there -- GEMV-OPT-20261025.md P0).
// Variant 2 additionally stages the *whole* A tile in registers-then-LDS, so
// its gate is still the full M*K*2 budget, and needs exactly 8 warps x RW=8
// rows and a block_n that rows=64 divides (its per-warp row ownership is
// fixed at launch).
int stream_block_select(int M, int N, int K, int64_t w_stride, int nkb,
                        int block_n, int block_k, int want, int& rows,
                        int& nt) {
  if (want == 0 || M < 1 || M > kMsMax) return 0;
  // S23 / O4: the M gate (see stream_max_m).  0 = no limit = today's dispatch.
  const int stream_m_lim = stream_max_m();
  if (stream_m_lim != 0 && M >= stream_m_lim) return 0;
  if (K < 64 || (K % kStreamChunk) != 0) return 0;
  if ((w_stride % kStreamChunk) != 0) return 0;
  if (block_k < kStreamChunk || (block_k % kStreamChunk) != 0) return 0;
  if (block_n < 8 || (block_n % 8) != 0) return 0;
  const bool want2 = (want == 2 && M <= 4 && (block_n % 64) == 0);
  const size_t smem =
      (want2 ? (size_t)M * K * sizeof(__half)
             : (size_t)stream_ms_for(M) * kStreamSlice * sizeof(__half)) +
      (size_t)M * nkb * sizeof(float);
  if (smem > stream_smem_limit()) return 0;
  if (want2) {
    rows = 64;
    nt = 512;
    return 2;
  }
  // P2 A/B knob (no rebuild): fixed rows per block; must divide block_n so a
  // block never straddles a w_s row.  Block size follows rows (one warp per
  // 8 rows, capped at 512 threads), so rows/nwarp is exact for the kernel's
  // per-warp W pipeline.
  const int rows_ov = stream_rows_override();
  if (rows_ov != 0 && (block_n % rows_ov) == 0) {
    rows = rows_ov;
    nt = rows_ov >= 8 ? 512 : rows_ov * 64;
    return 1;
  }
  // Load balance beats A-restaging: the perf1 s1 table shows 420-445 GB/s once
  // the grid has ~112+ blocks (N=3584/8704) but only 250-400 GB/s at 64-80
  // blocks (N=1280/4096/5120), because ceil(blocks/60) waves leave CUs idle at
  // the tail.  rows=8 (one row per warp at 512 threads, the geometry the
  // gemv_probe lds_* sweep ran 451 GB/s with at M=8) puts every production N
  // at 160+ blocks (N=1280 -> 160, N=8704 -> 1088); rows=4 only when 8 would
  // still leave fewer than kStreamMinBlocks.  The staging cost this re-tunes
  // is L2-resident: A rows are re-read per block, not re-streamed from DRAM.
  constexpr int kStreamMinBlocks = 120;
  rows = 8;
  if ((N + rows - 1) / rows < kStreamMinBlocks && (block_n % 4) == 0 &&
      (N + 3) / 4 >= kStreamMinBlocks)
    rows = 4;
  nt = rows >= 8 ? 512 : rows * 64;
  return 1;
}

// 0 = old GEMV, 1 = streaming variant 1 (default), 2 = variant 2.  Read once
// per process so an A/B needs no rebuild.
int stream_block_mode() {
  static const int mode = []() {
    const char* e = std::getenv("VLLM_DCU_FP8_GEMV_STREAM");
    if (e == nullptr) return 1;
    const int v = std::atoi(e);
    return v <= 0 ? 0 : (v == 2 ? 2 : 1);
  }();
  return mode;
}

template <typename T>
void dispatch_gemv_stream(int M, int N, int K, int64_t w_stride, int rows,
                          int nt, int nkb, int block_n, int cpb, bool a16,
                          const uint8_t* A, const uint8_t* W,
                          const float* a_s, const float* w_s,
                          const __half* lut, const T* bias, T* C,
                          hipStream_t stream) {
  // Rows per warp of the W pipeline: the selector pairs `rows` with a block
  // size of min(512, rows*64), so RW is exact and >= 1.
  const int rw = rows / (nt >> 6);
#define VLLM_FP8_EMU_GEMV_STREAM_RW(MS)                                     \
  do {                                                                      \
    if (rw <= 1) {                                                          \
      launch_gemv_stream<MS, 1, T>(M, N, K, w_stride, rows, nt, nkb,        \
                                   block_n, cpb, a16, A, W, a_s, w_s, lut,  \
                                   bias, C, stream);                        \
      return;                                                               \
    }                                                                       \
    if (rw == 2) {                                                          \
      launch_gemv_stream<MS, 2, T>(M, N, K, w_stride, rows, nt, nkb,        \
                                   block_n, cpb, a16, A, W, a_s, w_s, lut,  \
                                   bias, C, stream);                        \
      return;                                                               \
    }                                                                       \
    if (rw <= 4) {                                                          \
      launch_gemv_stream<MS, 4, T>(M, N, K, w_stride, rows, nt, nkb,        \
                                   block_n, cpb, a16, A, W, a_s, w_s, lut,  \
                                   bias, C, stream);                        \
      return;                                                               \
    }                                                                       \
    launch_gemv_stream<MS, 8, T>(M, N, K, w_stride, rows, nt, nkb,          \
                                 block_n, cpb, a16, A, W, a_s, w_s, lut,    \
                                 bias, C, stream);                          \
    return;                                                                 \
  } while (0)
  if (M <= 1) VLLM_FP8_EMU_GEMV_STREAM_RW(1);
  if (M <= 2) VLLM_FP8_EMU_GEMV_STREAM_RW(2);
  if (M <= 4) VLLM_FP8_EMU_GEMV_STREAM_RW(4);
  VLLM_FP8_EMU_GEMV_STREAM_RW(kMsMax);
#undef VLLM_FP8_EMU_GEMV_STREAM_RW
}

// ---------------------------------------------------------------------------
// streaming decode kernel, variant 2: A hoisted out of the row loop
// ---------------------------------------------------------------------------
// Same geometry as variant 1 (one block per `rows` rows of W, one warp per row
// at a time, lane-strided uint4 loads), but the loop order is K-outer /
// rows-inner with RW rows per warp held in registers:
//
//   for each K chunk:  load this lane's A chunk once (MS x 2 x 128-bit LDS)
//                      for r in 0..RW-1: load W[r][chunk], widen, dot2
//
// A lane's K position is identical for every row, so variant 1's per-row A
// re-read (MS*2 128-bit LDS reads + 4*64 LUT reads per 1 KiB of W) is paid
// once per chunk instead of once per row, the W loads of RW rows are in flight
// together, and the cross-lane reduction happens once per (row, m) for the
// whole K loop instead of per row.  Costs RW*MS fp32 accumulators, so it is
// only instantiated for M <= 4.
template <int MS, int RW, typename T>
__global__ void fp8_emu_gemv_stream2_block_kernel(
    const uint8_t* __restrict__ a, const uint8_t* __restrict__ w,
    const float* __restrict__ a_s, const float* __restrict__ w_s,
    const __half* __restrict__ lut, const T* __restrict__ bias,
    T* __restrict__ out, const int M, const int N, const int K,
    const int64_t w_stride, const int rows, const int nkb, const int block_n,
    const int cpb, const int a16) {
  // `sA` is accessed as uint4 by the MAC reader and (W8A16) written as
  // uint4, so the dynamic LDS base must be 16 B aligned; the declaration
  // alone only guarantees 4 B (the 512 B static LUT currently puts it at a
  // 16 B boundary by layout accident -- make it explicit).
  extern __shared__ __align__(16) uint32_t dyn[];
  __half* sA = reinterpret_cast<__half*>(dyn);                     // [M][K]
  float* sscale = reinterpret_cast<float*>(sA + (int64_t)M * K);   // [M][nkb]
  __shared__ __half sl[256];

  const int t = threadIdx.x;
  const int nt = blockDim.x;
  const int lane = t & 63;
  const int warp = t >> 6;
  const int K4 = K >> 2;

  fp8_lut_stage(sl, lut, nt);
  __syncthreads();
  {
    uint32_t* sA32 = reinterpret_cast<uint32_t*>(sA);
    if (a16 != 0) {  // W8A16: A is already fp16, only copy it.
      const uint4* __restrict__ av = reinterpret_cast<const uint4*>(a);
      uint4* __restrict__ sv = reinterpret_cast<uint4*>(sA);
      const int64_t tot = ((int64_t)M * K) >> 3;  // 8 fp16 per uint4
      for (int64_t i = t; i < tot; i += nt) sv[i] = av[i];
    } else {
      const uint32_t* __restrict__ a32 = reinterpret_cast<const uint32_t*>(a);
      const int64_t tot = (int64_t)M * K4;
      for (int64_t i = t; i < tot; i += nt) {
        uint32_t lo, hi;
        fp8x4_to_f16x2x2(sl, a32[i], lo, hi);
        sA32[2 * i] = lo;
        sA32[2 * i + 1] = hi;
      }
    }
  }
  {
    const int nb = (int)(((int64_t)blockIdx.x * rows) / block_n);
    const float* __restrict__ wsr = w_s + (int64_t)nb * nkb;
    const int tot = M * nkb;
    for (int i = t; i < tot; i += nt) {
      const int m = i / nkb;
      const int kb = i - m * nkb;
      sscale[i] = (a16 != 0) ? wsr[kb] : a_s[(int64_t)m * nkb + kb] * wsr[kb];
    }
  }
  __syncthreads();

  const int n0 = blockIdx.x * rows;
  const int k4v = K4 >> 2;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);
  const int rbase = n0 + warp * RW;

  float acc[RW][MS];
#pragma unroll
  for (int r = 0; r < RW; ++r) {
#pragma unroll
    for (int m = 0; m < MS; ++m) acc[r][m] = 0.0f;
  }

  const int niter = (k4v + 63) >> 6;
  for (int j = 0; j < niter; ++j) {
    const int i = lane + (j << 6);
    const int ic = min(i, k4v - 1);
    const int kb = ic / cpb;

    uint4 xa[MS], xb[MS];
#pragma unroll
    for (int m = 0; m < MS; ++m) {
      if (m < M) {
        const uint4* __restrict__ ap = reinterpret_cast<const uint4*>(
            sA + (int64_t)m * K + ic * kStreamChunk);
        xa[m] = ap[0];
        xb[m] = ap[1];
      }
    }

#pragma unroll
    for (int r = 0; r < RW; ++r) {
      const int row = rbase + r;
      if (row < N) {
        const uint4* __restrict__ wp =
            wv + (((int64_t)row * w_stride) >> 4);
        // Padded lanes (i >= k4v) must contribute exactly zero, not the
        // clamped chunk again.
        uint4 vv = make_uint4(0u, 0u, 0u, 0u);
        if (i < k4v) vv = wp[i];
        uint32_t p[8];
        fp8x4_to_f16x2x2(sl, vv.x, p[0], p[1]);
        fp8x4_to_f16x2x2(sl, vv.y, p[2], p[3]);
        fp8x4_to_f16x2x2(sl, vv.z, p[4], p[5]);
        fp8x4_to_f16x2x2(sl, vv.w, p[6], p[7]);
#pragma unroll
        for (int m = 0; m < MS; ++m) {
          if (m < M) {
            float loc = 0.0f;
            loc = dot2(p[0], xa[m].x, loc);
            loc = dot2(p[1], xa[m].y, loc);
            loc = dot2(p[2], xa[m].z, loc);
            loc = dot2(p[3], xa[m].w, loc);
            loc = dot2(p[4], xb[m].x, loc);
            loc = dot2(p[5], xb[m].y, loc);
            loc = dot2(p[6], xb[m].z, loc);
            loc = dot2(p[7], xb[m].w, loc);
            acc[r][m] += loc * sscale[m * nkb + kb];
          }
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RW; ++r) {
    const int row = rbase + r;
    if (row >= N) continue;
#pragma unroll
    for (int m = 0; m < MS; ++m) {
      if (m < M) {
        float v = acc[r][m];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off);
        if (lane == 0) {
          T o = static_cast<T>(v);
          if (bias != nullptr) o += static_cast<T>(bias[row]);
          out[(int64_t)m * N + row] = o;
        }
      }
    }
  }
}

template <int MS, int RW, typename T>
void launch_gemv_stream2(int M, int N, int K, int64_t w_stride, int rows, int nt,
                         int nkb, int block_n, int cpb, bool a16,
                         const uint8_t* A, const uint8_t* W, const float* a_s,
                         const float* w_s, const __half* lut, const T* bias,
                         T* C, hipStream_t stream) {
  const size_t smem = (size_t)M * K * sizeof(__half) +
                      (size_t)M * nkb * sizeof(float);
  dim3 grid((N + rows - 1) / rows);
  dim3 block(nt);
  fp8_emu_gemv_stream2_block_kernel<MS, RW, T><<<grid, block, smem, stream>>>(
      A, W, a_s, w_s, lut, bias, C, M, N, K, w_stride, rows, nkb, block_n,
      cpb, a16 ? 1 : 0);
}

template <typename T>
void dispatch_gemv_stream2(int M, int N, int K, int64_t w_stride, int rows,
                           int nt, int nkb, int block_n, int cpb, bool a16,
                           const uint8_t* A, const uint8_t* W,
                           const float* a_s, const float* w_s,
                           const __half* lut, const T* bias, T* C,
                           hipStream_t stream) {
  constexpr int kRW = 8;  // rows per warp (= rows / nwarp at 512 threads)
#define VLLM_FP8_EMU_GEMV_STREAM2_CASE(MS)                                  \
  do {                                                                      \
    launch_gemv_stream2<MS, kRW, T>(M, N, K, w_stride, rows, nt, nkb,       \
                                    block_n, cpb, a16, A, W, a_s, w_s, lut,  \
                                    bias, C, stream);                       \
    return;                                                                 \
  } while (0)
  if (M <= 1) VLLM_FP8_EMU_GEMV_STREAM2_CASE(1);
  if (M <= 2) VLLM_FP8_EMU_GEMV_STREAM2_CASE(2);
  VLLM_FP8_EMU_GEMV_STREAM2_CASE(4);
#undef VLLM_FP8_EMU_GEMV_STREAM2_CASE
}

// Stage a K slice from global into registers, honouring the (possibly padded)
// row stride of W: `process_fp8_weight_block_strategy` pads the last dimension
// by VLLM_ROCM_FP8_PADDING, so `w.stride(0)` is K + pad, not K.
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void fp8_gemm_load_block(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ W,
    int64_t w_stride, int m0, int n0, int M, int N, int K, int K4, int k0,
    int tid, uint32_t* __restrict__ dst) {
  using C = FConf<BM, BN, TM, TN>;
  const uint32_t* __restrict__ ap =
      reinterpret_cast<const uint32_t*>(A + (int64_t)m0 * K + k0);
  const uint32_t* __restrict__ wp =
      reinterpret_cast<const uint32_t*>(W + (int64_t)n0 * w_stride + k0);
  const int w4 = (int)(w_stride >> 2);
  const bool kfull = (k0 + kFBk) <= K;
#pragma unroll
  for (int u = 0; u < C::A4; ++u) {
    const int idx = tid + u * C::NT;
    const int r = idx >> kFLogKw4;
    const int jd = idx & (kFKw4 - 1);
    uint32_t v = 0;
    if (m0 + r < M && (kfull || (k0 + (jd + 1) * 4) <= K))
      v = ap[(int64_t)r * K4 + jd];
    dst[u] = v;
  }
#pragma unroll
  for (int u = 0; u < C::B4; ++u) {
    const int idx = tid + u * C::NT;
    const int r = idx >> kFLogKw4;
    const int jd = idx & (kFKw4 - 1);
    uint32_t v = 0;
    if (n0 + r < N && (kfull || (k0 + (jd + 1) * 4) <= K))
      v = wp[(int64_t)r * w4 + jd];
    dst[C::A4 + u] = v;
  }
}

// Fold one finished block_k window into the (scale-applied) accumulator.
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void fp8_gemm_fold_block_scale(
    int tx, int ty, int m0, int n0, int M, int N, int kb, int nkb, int block_n,
    const float* __restrict__ a_s, const float* __restrict__ w_s,
    float (&blk)[TM][TN], float (&acc)[TM][TN]) {
  using C = FConf<BM, BN, TM, TN>;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + C::MG * i;
    if (row >= M) continue;
    const float sa = a_s[(int64_t)row * nkb + kb];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + tx + C::NG * j;
      if (col >= N) continue;
      acc[i][j] += blk[i][j] * sa * w_s[(int64_t)(col / block_n) * nkb + kb];
      blk[i][j] = 0.0f;
    }
  }
}

template <int BM, int BN, int TM, int TN, typename T, int MINB = 1>
__global__ void __launch_bounds__((BM / TM) * (BN / TN), MINB)
    fp8_emu_gemm_block_kernel(const uint8_t* __restrict__ A,
                              const uint8_t* __restrict__ W,
                              const int64_t w_stride,
                              const float* __restrict__ a_s,
                              const float* __restrict__ w_s, const int nkb,
                              const int cpb, const int block_n,
                              const __half* __restrict__ lut,
                              const T* __restrict__ bias,
                              T* __restrict__ C, int M, int N, int K) {
  using C_ = FConf<BM, BN, TM, TN>;
  constexpr int NT = C_::NT;
  constexpr int FRAGS = C_::A4 + C_::B4;
  static_assert(C_::MG * TM == BM && C_::NG * TN == BN, "tile/thread mismatch");
  static_assert(BM * kFKw4 >= 1 && BN * kFKw4 >= 1, "empty tile");

  __shared__ uint32_t As[2][BM * kFKp];
  __shared__ uint32_t Bs[2][BN * kFKp];
  __shared__ __half sl[256];

  const int tid = threadIdx.x;
  const int tx = tid % C_::NG;  // along N
  const int ty = tid / C_::NG;  // along M
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int K4 = K >> 2;
  const int NK = (K + kFBk - 1) / kFBk;

  fp8_lut_stage(sl, lut, NT);
  __syncthreads();

  float acc[TM][TN];  // finished K blocks, scale already applied
  float blk[TM][TN];  // the block_k window being accumulated
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
      blk[i][j] = 0.0f;
    }
  }

  uint32_t f0[FRAGS], f1[FRAGS];
  fp8_gemm_load_block<BM, BN, TM, TN>(A, W, w_stride, m0, n0, M, N, K, K4, 0,
                                      tid, f0);

  int c = 0;  // K chunk index (kFBk elements each)
  for (; c + 1 < NK; c += 2) {
    fp8_gemm_store<BM, BN, TM, TN>(tid, f0, sl, As[0], Bs[0]);
    __syncthreads();
    fp8_gemm_load_block<BM, BN, TM, TN>(A, W, w_stride, m0, n0, M, N, K, K4,
                                        (c + 1) * kFBk, tid, f1);
    fp8_gemm_compute<BM, BN, TM, TN>(tx, ty, As[0], Bs[0], blk);
    if (((c + 1) % cpb) == 0)
      fp8_gemm_fold_block_scale<BM, BN, TM, TN>(tx, ty, m0, n0, M, N, c / cpb,
                                                nkb, block_n, a_s, w_s, blk,
                                                acc);

    fp8_gemm_store<BM, BN, TM, TN>(tid, f1, sl, As[1], Bs[1]);
    __syncthreads();
    fp8_gemm_load_block<BM, BN, TM, TN>(A, W, w_stride, m0, n0, M, N, K, K4,
                                        (c + 2) * kFBk, tid, f0);
    fp8_gemm_compute<BM, BN, TM, TN>(tx, ty, As[1], Bs[1], blk);
    if (((c + 2) % cpb) == 0)
      fp8_gemm_fold_block_scale<BM, BN, TM, TN>(tx, ty, m0, n0, M, N,
                                                (c + 1) / cpb, nkb, block_n,
                                                a_s, w_s, blk, acc);
  }
  if (c < NK) {  // odd NK: the leftover slice is number 0 (c is even here)
    fp8_gemm_store<BM, BN, TM, TN>(tid, f0, sl, As[0], Bs[0]);
    __syncthreads();
    fp8_gemm_compute<BM, BN, TM, TN>(tx, ty, As[0], Bs[0], blk);
    if (((c + 1) % cpb) == 0)
      fp8_gemm_fold_block_scale<BM, BN, TM, TN>(tx, ty, m0, n0, M, N, c / cpb,
                                                nkb, block_n, a_s, w_s, blk,
                                                acc);
  }

  // Epilogue: the per-block scales are already folded in; cast, then bias.
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + C_::MG * i;
    if (row >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + tx + C_::NG * j;
      if (col >= N) continue;
      T o = static_cast<T>(acc[i][j]);
      if (bias != nullptr) o += static_cast<T>(bias[col]);
      C[(int64_t)row * N + col] = o;
    }
  }
}

template <int BM, int BN, int TM, int TN, typename T, int MINB = 1>
void launch_gemm_block(int M, int N, int K, int64_t w_stride, int nkb, int cpb,
                       int block_n, const uint8_t* A, const uint8_t* W,
                       const float* a_s, const float* w_s, const __half* lut,
                       const T* bias, T* C, hipStream_t stream) {
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block((BM / TM) * (BN / TN));
  fp8_emu_gemm_block_kernel<BM, BN, TM, TN, T, MINB>
      <<<grid, block, 0, stream>>>(A, W, w_stride, a_s, w_s, nkb, cpb,
                                   block_n, lut, bias, C, M, N, K);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// same, but with a wider K chunk (BK elements per stage)
// ---------------------------------------------------------------------------
// S20 left exactly one discriminator open between "per-chunk staging
// serialisation" and everything else: `kFBk = 64`, i.e. half as many staging
// rounds for the same staged bytes (.s20/S20_kernel_NOTES.md §5.6, §6.2-6: the
// cfg 16 kernel issues its instructions at only 38% of peak, and the two
// candidate causes -- LDS bank conflicts and the staging rounds -- cannot be
// separated without this variant).  A wider BK keeps the per-byte staging cost
// identical (same widening, same stores, same LDS reads per element) and halves
// the barrier count, the loop overhead and the number of exposed round trips.
//
// Kept self-contained instead of templating kFBk through the shared helpers:
// the prefill tiles (cfgs 1..10) must keep compiling to byte-identical code, and
// `kFBk` also drives the scale-fold cadence (`cpb = block_k / kFBk`), so sharing
// it would put every prefill cfg on the diff for no gain.
template <int BK, int BM, int BN, int TM, int TN>
struct KConf {
  static_assert(BK == 32 || BK == 64 || BK == 128,
                "KConf: LogKw4 is only defined for BK = 32/64/128");
  static constexpr int Kw4 = BK / 4;   // fp8 dwords per row per chunk
  static constexpr int Kw = BK / 2;    // fp16 dwords per row per chunk
  static constexpr int Kp = Kw + 1;    // padded LDS row stride
  static constexpr int LogKw4 = (BK == 32) ? 3 : ((BK == 64) ? 4 : 5);
  static constexpr int MG = BM / TM;
  static constexpr int NG = BN / TN;
  static constexpr int NT = MG * NG;
  static constexpr int A4 = (BM * Kw4 + NT - 1) / NT;
  static constexpr int B4 = (BN * Kw4 + NT - 1) / NT;
};

template <int BK, int BM, int BN, int TM, int TN>
__device__ __forceinline__ void fp8_gemm_load_wide(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ W,
    int64_t w_stride, int m0, int n0, int M, int N, int K, int K4, int k0,
    int tid, uint32_t* __restrict__ dst) {
  using C = KConf<BK, BM, BN, TM, TN>;
  const uint32_t* __restrict__ ap =
      reinterpret_cast<const uint32_t*>(A + (int64_t)m0 * K + k0);
  const uint32_t* __restrict__ wp =
      reinterpret_cast<const uint32_t*>(W + (int64_t)n0 * w_stride + k0);
  const int w4 = (int)(w_stride >> 2);
  const bool kfull = (k0 + BK) <= K;
#pragma unroll
  for (int u = 0; u < C::A4; ++u) {
    const int idx = tid + u * C::NT;
    const int r = idx >> C::LogKw4;
    const int jd = idx & (C::Kw4 - 1);
    uint32_t v = 0;
    if (idx < BM * C::Kw4 && m0 + r < M && (kfull || (k0 + (jd + 1) * 4) <= K))
      v = ap[(int64_t)r * K4 + jd];
    dst[u] = v;
  }
#pragma unroll
  for (int u = 0; u < C::B4; ++u) {
    const int idx = tid + u * C::NT;
    const int r = idx >> C::LogKw4;
    const int jd = idx & (C::Kw4 - 1);
    uint32_t v = 0;
    if (idx < BN * C::Kw4 && n0 + r < N && (kfull || (k0 + (jd + 1) * 4) <= K))
      v = wp[(int64_t)r * w4 + jd];
    dst[C::A4 + u] = v;
  }
}

template <int BK, int BM, int BN, int TM, int TN>
__device__ __forceinline__ void fp8_gemm_store_wide(
    int tid, const uint32_t* __restrict__ src, const __half* __restrict__ lut,
    uint32_t* __restrict__ As, uint32_t* __restrict__ Bs) {
  using C = KConf<BK, BM, BN, TM, TN>;
#pragma unroll
  for (int u = 0; u < C::A4; ++u) {
    const int idx = tid + u * C::NT;
    if (idx >= BM * C::Kw4) continue;
    const int r = idx >> C::LogKw4;
    const int jd = idx & (C::Kw4 - 1);
    uint32_t lo, hi;
    fp8x4_to_f16x2x2(lut, src[u], lo, hi);
    As[r * C::Kp + 2 * jd] = lo;
    As[r * C::Kp + 2 * jd + 1] = hi;
  }
#pragma unroll
  for (int u = 0; u < C::B4; ++u) {
    const int idx = tid + u * C::NT;
    if (idx >= BN * C::Kw4) continue;
    const int r = idx >> C::LogKw4;
    const int jd = idx & (C::Kw4 - 1);
    uint32_t lo, hi;
    fp8x4_to_f16x2x2(lut, src[C::A4 + u], lo, hi);
    Bs[r * C::Kp + 2 * jd] = lo;
    Bs[r * C::Kp + 2 * jd + 1] = hi;
  }
}

template <int BK, int BM, int BN, int TM, int TN>
__device__ __forceinline__ void fp8_gemm_compute_wide(
    int tx, int ty, const uint32_t* __restrict__ as,
    const uint32_t* __restrict__ bs, float (&acc)[TM][TN]) {
  using C = KConf<BK, BM, BN, TM, TN>;
#pragma unroll
  for (int u = 0; u < C::Kw; ++u) {
    uint32_t a[TM], b[TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) a[i] = as[(ty + C::MG * i) * C::Kp + u];
#pragma unroll
    for (int j = 0; j < TN; ++j) b[j] = bs[(tx + C::NG * j) * C::Kp + u];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        acc[i][j] = dot2(a[i], b[j], acc[i][j]);
      }
    }
  }
}


// Wide K chunk with the ORIGINAL two-stage double buffer (2 x BK elements of
// LDS instead of RST+1 stages).  This is the configuration that keeps cfg 16's
// occupancy while still halving the staging rounds: at BK = 64 the LDS footprint
// is 2 x (BM + BN) * (BK/2 + 1) * 4 = 19 KiB, i.e. still 3 blocks/CU, whereas
// the pipelined BK = 64 variant needs 4 stages = 38 KiB and drops to 1 block/CU
// -- which is exactly what costs it 64% at N = 8704 (measured, see
// .s23/w2/O3/sweep/sweep_wide_dev3.log).
template <int BK, int BM, int BN, int TM, int TN, typename T>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    fp8_emu_gemm_block_wide2_kernel(const uint8_t* __restrict__ A,
                                    const uint8_t* __restrict__ W,
                                    const int64_t w_stride,
                                    const float* __restrict__ a_s,
                                    const float* __restrict__ w_s,
                                    const int nkb, const int cpb,
                                    const int block_n,
                                    const __half* __restrict__ lut,
                                    const T* __restrict__ bias,
                                    T* __restrict__ C, int M, int N, int K) {
  using C_ = KConf<BK, BM, BN, TM, TN>;
  constexpr int NT = C_::NT;
  constexpr int FRAGS = C_::A4 + C_::B4;
  static_assert(C_::MG * TM == BM && C_::NG * TN == BN, "tile/thread mismatch");

  __shared__ uint32_t As[2][BM * C_::Kp];
  __shared__ uint32_t Bs[2][BN * C_::Kp];
  __shared__ __half sl[256];

  const int tid = threadIdx.x;
  const int tx = tid % C_::NG;
  const int ty = tid / C_::NG;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int K4 = K >> 2;
  const int NK = (K + BK - 1) / BK;

  fp8_lut_stage(sl, lut, NT);
  __syncthreads();

  float acc[TM][TN];
  float blk[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
      blk[i][j] = 0.0f;
    }
  }

  uint32_t f0[FRAGS], f1[FRAGS];
  fp8_gemm_load_wide<BK, BM, BN, TM, TN>(A, W, w_stride, m0, n0, M, N, K, K4,
                                         0, tid, f0);
  int c = 0;
  for (; c + 1 < NK; c += 2) {
    fp8_gemm_store_wide<BK, BM, BN, TM, TN>(tid, f0, sl, As[0], Bs[0]);
    __syncthreads();
    fp8_gemm_load_wide<BK, BM, BN, TM, TN>(A, W, w_stride, m0, n0, M, N, K, K4,
                                           (c + 1) * BK, tid, f1);
    fp8_gemm_compute_wide<BK, BM, BN, TM, TN>(tx, ty, As[0], Bs[0], blk);
    if (((c + 1) % cpb) == 0)
      fp8_gemm_fold_block_scale<BM, BN, TM, TN>(tx, ty, m0, n0, M, N, c / cpb,
                                                nkb, block_n, a_s, w_s, blk,
                                                acc);
    fp8_gemm_store_wide<BK, BM, BN, TM, TN>(tid, f1, sl, As[1], Bs[1]);
    __syncthreads();
    fp8_gemm_load_wide<BK, BM, BN, TM, TN>(A, W, w_stride, m0, n0, M, N, K, K4,
                                           (c + 2) * BK, tid, f0);
    fp8_gemm_compute_wide<BK, BM, BN, TM, TN>(tx, ty, As[1], Bs[1], blk);
    if (((c + 2) % cpb) == 0)
      fp8_gemm_fold_block_scale<BM, BN, TM, TN>(tx, ty, m0, n0, M, N,
                                                (c + 1) / cpb, nkb, block_n,
                                                a_s, w_s, blk, acc);
  }
  if (c < NK) {
    fp8_gemm_store_wide<BK, BM, BN, TM, TN>(tid, f0, sl, As[0], Bs[0]);
    __syncthreads();
    fp8_gemm_compute_wide<BK, BM, BN, TM, TN>(tx, ty, As[0], Bs[0], blk);
    if (((c + 1) % cpb) == 0)
      fp8_gemm_fold_block_scale<BM, BN, TM, TN>(tx, ty, m0, n0, M, N, c / cpb,
                                                nkb, block_n, a_s, w_s, blk,
                                                acc);
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + C_::MG * i;
    if (row >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + tx + C_::NG * j;
      if (col >= N) continue;
      T o = static_cast<T>(acc[i][j]);
      if (bias != nullptr) o += static_cast<T>(bias[col]);
      C[(int64_t)row * N + col] = o;
    }
  }
}

template <int BK, int BM, int BN, int TM, int TN, typename T>
void launch_gemm_block_wide2(int M, int N, int K, int64_t w_stride, int nkb,
                             int cpb, int block_n, const uint8_t* A,
                             const uint8_t* W, const float* a_s,
                             const float* w_s, const __half* lut, const T* bias,
                             T* C, hipStream_t stream) {
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block((BM / TM) * (BN / TN));
  fp8_emu_gemm_block_wide2_kernel<BK, BM, BN, TM, TN, T>
      <<<grid, block, 0, stream>>>(A, W, w_stride, a_s, w_s, nkb, cpb, block_n,
                                   lut, bias, C, M, N, K);
}


// ---------------------------------------------------------------------------
// W8A16: fp16 activations against fp8 block-scaled weights
// ---------------------------------------------------------------------------
// Same tiling, LDS layout, `v_dot2_f32_f16` MAC order and block-boundary scale
// fold as `fp8_emu_gemm_block_wide2_kernel`; only the A side differs.  A is
// already fp16, so the staging loads two fp16 lanes per dword straight from HBM
// (`K / 2` dwords per row) and stores them into LDS without the 256-entry LUT,
// and the fold degenerates to `blk * w_s` because no per-token activation scale
// exists.  This is the whole point of the W8A16 route: the activation is never
// quantized, so every layer drops the quantize launch and the A-side LUT
// traffic, while the weight stays fp8 in HBM (decode bandwidth unchanged).
//
// The A fragment count doubles against the fp8 tile of the same geometry
// (`Kw = BK / 2` dwords per row instead of `BK / 4`), which is why the W8A16
// tile set stops at BN = 64/128 and omits the W8A8 wide-N tiles (BN = 128,
// TN = 8): those would spill their register fragments.
template <int BK, int BM, int BN, int TM, int TN>
struct A16Conf {
  static_assert(BK == 32 || BK == 64 || BK == 128,
                "A16Conf: LogKw/LogKw4 are only defined for BK = 32/64/128");
  static constexpr int Kw4 = BK / 4;  // fp8 dwords per row per chunk (W side)
  static constexpr int Kw = BK / 2;   // fp16 pairs per row per chunk (A side)
  static constexpr int Kp = Kw + 1;   // padded LDS row stride
  static constexpr int LogKw4 = (BK == 32) ? 3 : ((BK == 64) ? 4 : 5);
  static constexpr int LogKw = (BK == 32) ? 4 : ((BK == 64) ? 5 : 6);
  static constexpr int MG = BM / TM;
  static constexpr int NG = BN / TN;
  static constexpr int NT = MG * NG;
  static constexpr int A16 = (BM * Kw + NT - 1) / NT;
  static constexpr int B4 = (BN * Kw4 + NT - 1) / NT;
};

// Stage one K slice of A (fp16, no widening) and W (fp8 dwords) into registers.
template <int BK, int BM, int BN, int TM, int TN>
__device__ __forceinline__ void w8a16_load(
    const __half* __restrict__ A, const uint8_t* __restrict__ W,
    int64_t w_stride, int m0, int n0, int M, int N, int K, int k0, int tid,
    uint32_t* __restrict__ dst) {
  using C = A16Conf<BK, BM, BN, TM, TN>;
  const int a2 = K >> 1;  // fp16 pairs (= dwords) per row of the activation
  const uint32_t* __restrict__ ap =
      reinterpret_cast<const uint32_t*>(A) + (int64_t)m0 * a2 + (k0 >> 1);
  const uint32_t* __restrict__ wp =
      reinterpret_cast<const uint32_t*>(W + (int64_t)n0 * w_stride + k0);
  const int w4 = (int)(w_stride >> 2);
  const bool kfull = (k0 + BK) <= K;
#pragma unroll
  for (int u = 0; u < C::A16; ++u) {
    const int idx = tid + u * C::NT;
    const int r = idx >> C::LogKw;
    const int jd = idx & (C::Kw - 1);
    uint32_t v = 0;
    if (idx < BM * C::Kw && m0 + r < M && (kfull || (k0 + (jd + 1) * 2) <= K))
      v = ap[(int64_t)r * a2 + jd];
    dst[u] = v;
  }
#pragma unroll
  for (int u = 0; u < C::B4; ++u) {
    const int idx = tid + u * C::NT;
    const int r = idx >> C::LogKw4;
    const int jd = idx & (C::Kw4 - 1);
    uint32_t v = 0;
    if (idx < BN * C::Kw4 && n0 + r < N && (kfull || (k0 + (jd + 1) * 4) <= K))
      v = wp[(int64_t)r * w4 + jd];
    dst[C::A16 + u] = v;
  }
}

// Registers -> one LDS stage.  A dwords go in as they are (two fp16 lanes per
// dword, which is exactly one `v_dot2_f32_f16` operand); W dwords are widened.
template <int BK, int BM, int BN, int TM, int TN>
__device__ __forceinline__ void w8a16_store(
    int tid, const uint32_t* __restrict__ src, const __half* __restrict__ lut,
    uint32_t* __restrict__ As, uint32_t* __restrict__ Bs) {
  using C = A16Conf<BK, BM, BN, TM, TN>;
#pragma unroll
  for (int u = 0; u < C::A16; ++u) {
    const int idx = tid + u * C::NT;
    if (idx >= BM * C::Kw) continue;
    const int r = idx >> C::LogKw;
    const int jd = idx & (C::Kw - 1);
    As[r * C::Kp + jd] = src[u];
  }
#pragma unroll
  for (int u = 0; u < C::B4; ++u) {
    const int idx = tid + u * C::NT;
    if (idx >= BN * C::Kw4) continue;
    const int r = idx >> C::LogKw4;
    const int jd = idx & (C::Kw4 - 1);
    uint32_t lo, hi;
    fp8x4_to_f16x2x2(lut, src[C::A16 + u], lo, hi);
    Bs[r * C::Kp + 2 * jd] = lo;
    Bs[r * C::Kp + 2 * jd + 1] = hi;
  }
}

// Fold one finished block_k window into the accumulator: weight scale only.
template <int BK, int BM, int BN, int TM, int TN>
__device__ __forceinline__ void w8a16_fold_block_scale(
    int tx, int ty, int m0, int n0, int M, int N, int kb, int nkb, int block_n,
    const float* __restrict__ w_s, float (&blk)[TM][TN], float (&acc)[TM][TN]) {
  using C = A16Conf<BK, BM, BN, TM, TN>;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + C::MG * i;
    if (row >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + tx + C::NG * j;
      if (col >= N) continue;
      acc[i][j] += blk[i][j] * w_s[(int64_t)(col / block_n) * nkb + kb];
      blk[i][j] = 0.0f;
    }
  }
}

template <int BK, int BM, int BN, int TM, int TN, typename T>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    fp8_emu_gemm_block_w8a16_kernel(const __half* __restrict__ A,
                                    const uint8_t* __restrict__ W,
                                    const int64_t w_stride,
                                    const float* __restrict__ w_s,
                                    const int nkb, const int cpb,
                                    const int block_n,
                                    const __half* __restrict__ lut,
                                    const T* __restrict__ bias,
                                    T* __restrict__ C, int M, int N, int K) {
  using C_ = A16Conf<BK, BM, BN, TM, TN>;
  constexpr int NT = C_::NT;
  constexpr int FRAGS = C_::A16 + C_::B4;
  static_assert(C_::MG * TM == BM && C_::NG * TN == BN, "tile/thread mismatch");

  __shared__ uint32_t As[2][BM * C_::Kp];
  __shared__ uint32_t Bs[2][BN * C_::Kp];
  __shared__ __half sl[256];

  const int tid = threadIdx.x;
  const int tx = tid % C_::NG;
  const int ty = tid / C_::NG;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int NK = (K + BK - 1) / BK;

  fp8_lut_stage(sl, lut, NT);
  __syncthreads();

  float acc[TM][TN];  // finished K blocks, scale already applied
  float blk[TM][TN];  // the block_k window being accumulated
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
      blk[i][j] = 0.0f;
    }
  }

  uint32_t f0[FRAGS], f1[FRAGS];
  w8a16_load<BK, BM, BN, TM, TN>(A, W, w_stride, m0, n0, M, N, K, 0, tid, f0);
  int c = 0;
  for (; c + 1 < NK; c += 2) {
    w8a16_store<BK, BM, BN, TM, TN>(tid, f0, sl, As[0], Bs[0]);
    __syncthreads();
    w8a16_load<BK, BM, BN, TM, TN>(A, W, w_stride, m0, n0, M, N, K,
                                   (c + 1) * BK, tid, f1);
    fp8_gemm_compute_wide<BK, BM, BN, TM, TN>(tx, ty, As[0], Bs[0], blk);
    if (((c + 1) % cpb) == 0)
      w8a16_fold_block_scale<BK, BM, BN, TM, TN>(tx, ty, m0, n0, M, N, c / cpb,
                                                 nkb, block_n, w_s, blk, acc);

    w8a16_store<BK, BM, BN, TM, TN>(tid, f1, sl, As[1], Bs[1]);
    __syncthreads();
    w8a16_load<BK, BM, BN, TM, TN>(A, W, w_stride, m0, n0, M, N, K,
                                   (c + 2) * BK, tid, f0);
    fp8_gemm_compute_wide<BK, BM, BN, TM, TN>(tx, ty, As[1], Bs[1], blk);
    if (((c + 2) % cpb) == 0)
      w8a16_fold_block_scale<BK, BM, BN, TM, TN>(tx, ty, m0, n0, M, N,
                                                 (c + 1) / cpb, nkb, block_n,
                                                 w_s, blk, acc);
  }
  if (c < NK) {  // odd NK: the leftover slice is number 0 (c is even here)
    w8a16_store<BK, BM, BN, TM, TN>(tid, f0, sl, As[0], Bs[0]);
    __syncthreads();
    fp8_gemm_compute_wide<BK, BM, BN, TM, TN>(tx, ty, As[0], Bs[0], blk);
    if (((c + 1) % cpb) == 0)
      w8a16_fold_block_scale<BK, BM, BN, TM, TN>(tx, ty, m0, n0, M, N, c / cpb,
                                                 nkb, block_n, w_s, blk, acc);
  }

  // Epilogue: the block scales are already folded in; cast, then bias.
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + C_::MG * i;
    if (row >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + tx + C_::NG * j;
      if (col >= N) continue;
      T o = static_cast<T>(acc[i][j]);
      if (bias != nullptr) o += static_cast<T>(bias[col]);
      C[(int64_t)row * N + col] = o;
    }
  }
}

template <int BK, int BM, int BN, int TM, int TN, typename T>
void launch_gemm_block_w8a16(int M, int N, int K, int64_t w_stride, int nkb,
                             int block_k, int block_n, const __half* A,
                             const uint8_t* W, const float* w_s,
                             const __half* lut, const T* bias, T* C,
                             hipStream_t stream) {
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block((BM / TM) * (BN / TN));
  fp8_emu_gemm_block_w8a16_kernel<BK, BM, BN, TM, TN, T>
      <<<grid, block, 0, stream>>>(A, W, w_stride, w_s, nkb, block_k / BK,
                                   block_n, lut, bias, C, M, N, K);
}

// W8A16 tile set (these are *not* the W8A8 cfgs -- see the A16Conf comment):
//   1: BK=32  32x64  2x4  -- fallback for any block_k % 32 == 0
//   2: BK=64  32x64  2x4  -- S24 cfg 36 geometry, prefill default
//   3: BK=64  64x64  4x4  -- S24 cfg 37
//   4: BK=128 32x32  2x2  -- S24 cfg 38
//   5: BK=64   8x64  1x2  -- exact-M decode tile (M <= 8 with the GEMV lowered)
// An incoming W8A8 cfg number is mapped onto the closest member so the existing
// `VLLM_DCU_FP8_PREFILL_CFG` / `VLLM_DCU_FP8_DECODE_CFG` A/B knobs stay usable.
int w8a16_cfg(int M, int N, int64_t block_k, int cfg) {
  // 20..31 are not W8A16 tiles: treat them as the auto rule instead of silently
  // falling through `default` (the entry accepts cfg 0..40 for A/B).
  if (cfg >= 20 && cfg <= 31) cfg = 0;
  switch (cfg) {
    case 32:
    case 33:
    case 34:
    case 35:
    case 36:
    case 39:
    case 40:
      cfg = 2;
      break;
    case 37:
      cfg = 3;
      break;
    case 38:
      cfg = 4;
      break;
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 19:
      // These W8A8 numbers are decode tiles.  Only map them to the exact-M
      // W8A16 tile when the call really is a decode call; a prefill call
      // (M > kMsMax) must not run an BM = 8 tile with M/8 row blocks.
      cfg = (M <= kMsMax) ? 5 : ((N >= 6144) ? 2 : 1);
      break;
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
      cfg = 1;
      break;
    case 0:
    default:
      // Auto rule: exact-M tile on the decode range, the S24 wide-chunk tile on
      // the prefill range once the grid is wide or M is deep enough to amortise
      // the larger per-thread fragment count.
      cfg = (M <= kMsMax) ? 5 : ((N >= 6144 || (M > 32 && N >= 3072)) ? 2 : 1);
      break;
  }
  // A tile can only fold on a block_k boundary when its BK divides block_k.
  if ((cfg == 2 || cfg == 3 || cfg == 5) && (block_k % 64) != 0) cfg = 1;
  if (cfg == 4 && (block_k % 128) != 0) cfg = 1;
  return cfg;
}

template <typename T>
void dispatch_gemm_block_w8a16(int cfg, int M, int N, int K, int64_t w_stride,
                               int nkb, int block_k, int block_n,
                               const __half* A, const uint8_t* W,
                               const float* w_s, const __half* lut,
                               const T* bias, T* C, hipStream_t stream) {
#define VLLM_FP8_EMU_GEMM_W8A16_CASE(CFG, BK, BM, BN, TM, TN)                \
  case CFG:                                                                  \
    launch_gemm_block_w8a16<BK, BM, BN, TM, TN, T>(                          \
        M, N, K, w_stride, nkb, block_k, block_n, A, W, w_s, lut, bias, C,   \
        stream);                                                             \
    break
  switch (cfg) {
    VLLM_FP8_EMU_GEMM_W8A16_CASE(1, 32, 32, 64, 2, 4);
    VLLM_FP8_EMU_GEMM_W8A16_CASE(2, 64, 32, 64, 2, 4);
    VLLM_FP8_EMU_GEMM_W8A16_CASE(3, 64, 64, 64, 4, 4);
    VLLM_FP8_EMU_GEMM_W8A16_CASE(4, 128, 32, 32, 2, 2);
    VLLM_FP8_EMU_GEMM_W8A16_CASE(5, 64, 8, 64, 1, 2);
    default:
      launch_gemm_block_w8a16<32, 32, 64, 2, 4, T>(
          M, N, K, w_stride, nkb, block_k, block_n, A, W, w_s, lut, bias, C,
          stream);
      break;
  }
#undef VLLM_FP8_EMU_GEMM_W8A16_CASE
}

void check_common(const torch::Tensor& a, const torch::Tensor& w,
                  const torch::Tensor& a_s, const torch::Tensor& w_s,
                  const torch::Tensor& lut,
                  const std::optional<torch::Tensor>& bias,
                  at::ScalarType out_dtype, int& a_s_stride, int& w_s_stride) {
  TORCH_CHECK(a.is_cuda() && w.is_cuda() && a_s.is_cuda() && w_s.is_cuda() &&
                  lut.is_cuda(),
              "fp8_emu: all tensors must be on the GPU");
  TORCH_CHECK(a.element_size() == 1 && w.element_size() == 1,
              "fp8_emu: a/w must be 8-bit (fp8) tensors");
  TORCH_CHECK(a.scalar_type() == w.scalar_type(),
              "fp8_emu: a/w must have the same fp8 dtype");
  TORCH_CHECK(a_s.scalar_type() == at::kFloat && w_s.scalar_type() == at::kFloat,
              "fp8_emu: scales must be float32");
  TORCH_CHECK(lut.scalar_type() == at::kHalf && lut.numel() == 256,
              "fp8_emu: lut must be a 256-element fp16 tensor");
  TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && a_s.is_contiguous() &&
                  w_s.is_contiguous() && lut.is_contiguous(),
              "fp8_emu: tensors must be contiguous");
  TORCH_CHECK(w.size(1) == a.size(1), "fp8_emu: K mismatch: a K=", a.size(1),
              " w K=", w.size(1));
  TORCH_CHECK(a_s.numel() == a.size(0) || a_s.numel() == 1,
              "fp8_emu: a_s must have M or 1 elements, got ", a_s.numel());
  TORCH_CHECK(w_s.numel() == w.size(0) || w_s.numel() == 1,
              "fp8_emu: w_s must have N or 1 elements, got ", w_s.numel());
  TORCH_CHECK(out_dtype == at::kHalf || out_dtype == at::kBFloat16 ||
                  out_dtype == at::kFloat,
              "fp8_emu: unsupported out dtype ", out_dtype);
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() &&
                    bias->numel() == w.size(0),
                "fp8_emu: bias must be a contiguous GPU tensor of N elements");
    TORCH_CHECK(bias->scalar_type() == out_dtype,
                "fp8_emu: bias dtype must match out dtype");
  }
  a_s_stride = a_s.numel() == 1 ? 0 : 1;
  w_s_stride = w_s.numel() == 1 ? 0 : 1;
}

template <typename T>
void dispatch_result(const at::Tensor& out, int M, int N, int K,
                     int a_s_stride, int w_s_stride, const uint8_t* A,
                     const uint8_t* W, const uint32_t* a_h2, const float* a_s,
                     const float* w_s, const __half* lut, const void* bias_ptr,
                     bool gemv, int nthreads, int cfg, hipStream_t stream) {
  const T* bias = reinterpret_cast<const T*>(bias_ptr);
  T* C = reinterpret_cast<T*>(out.data_ptr());
  if (gemv) {
    launch_gemv<T>(M, N, K / 4, nthreads, a_s_stride, w_s_stride, A, W, a_h2,
                   a_s, w_s, lut, bias, C, stream);
    return;
  }
#define VLLM_FP8_EMU_GEMM_CASE(CFG, BM, BN, TM, TN)                     \
  case CFG:                                                             \
    launch_gemm<BM, BN, TM, TN, T>(M, N, K, a_s_stride, w_s_stride, A,  \
                                   W, a_s, w_s, lut, bias, C, stream);  \
    break
  switch (cfg) {
    VLLM_FP8_EMU_GEMM_CASE(1, 32, 64, 4, 4);
    VLLM_FP8_EMU_GEMM_CASE(2, 64, 64, 4, 4);
    VLLM_FP8_EMU_GEMM_CASE(3, 64, 128, 4, 8);
    case 4:
    default:
      launch_gemm<128, 128, 8, 8, T>(M, N, K, a_s_stride, w_s_stride, A, W,
                                     a_s, w_s, lut, bias, C, stream);
      break;
  }
#undef VLLM_FP8_EMU_GEMM_CASE
}

void run(const torch::Tensor& a, const torch::Tensor& w,
         const torch::Tensor& a_s, const torch::Tensor& w_s,
         const torch::Tensor& lut, const std::optional<torch::Tensor>& bias,
         torch::Tensor& out, bool gemv, int nthreads, int cfg) {
  int a_s_stride = 1, w_s_stride = 1;
  check_common(a, w, a_s, w_s, lut, bias, out.scalar_type(), a_s_stride,
               w_s_stride);
  const int M = (int)a.size(0), N = (int)w.size(0), K = (int)a.size(1);
  const void* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined()) bias_ptr = bias->data_ptr();

  const uint8_t* A = reinterpret_cast<const uint8_t*>(a.data_ptr());
  const uint8_t* W = reinterpret_cast<const uint8_t*>(w.data_ptr());
  const float* a_s_p = reinterpret_cast<const float*>(a_s.data_ptr());
  const float* w_s_p = reinterpret_cast<const float*>(w_s.data_ptr());
  const __half* lut_p = reinterpret_cast<const __half*>(lut.data_ptr());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  // Hoist the A widening out of the GEMV's per-column loop when it pays off:
  // every block re-reads A, so widening it once per launch removes M LDS reads
  // per A element per block.  M == 1 skips the extra kernel launch (the inline
  // path is already LDS-light there).
  const uint32_t* a_h2 = nullptr;
  at::Tensor a_scratch;
  if (gemv && M >= 2) {
    a_scratch = at::empty({(int64_t)M * K}, a.options().dtype(at::kHalf));
    a_h2 = reinterpret_cast<const uint32_t*>(a_scratch.data_ptr());
    const int64_t n_dwords = (int64_t)M * K / 4;
    const int threads = 256;
    fp8_emu_widen_a_kernel<<<(unsigned)((n_dwords + threads - 1) / threads),
                             threads, 0, stream>>>(A, lut_p,
                                                   const_cast<uint32_t*>(a_h2),
                                                   n_dwords);
  }

  switch (out.scalar_type()) {
    case at::kBFloat16:
      dispatch_result<at::BFloat16>(out, M, N, K, a_s_stride, w_s_stride, A, W,
                                    a_h2, a_s_p, w_s_p, lut_p, bias_ptr, gemv,
                                    nthreads, cfg, stream);
      break;
    case at::kHalf:
      dispatch_result<at::Half>(out, M, N, K, a_s_stride, w_s_stride, A, W,
                                a_h2, a_s_p, w_s_p, lut_p, bias_ptr, gemv,
                                nthreads, cfg, stream);
      break;
    default:
      dispatch_result<float>(out, M, N, K, a_s_stride, w_s_stride, A, W, a_h2,
                             a_s_p, w_s_p, lut_p, bias_ptr, gemv, nthreads,
                             cfg, stream);
      break;
  }
  (void)K;
}

// ---------------------------------------------------------------------------
// block-scaled host side
// ---------------------------------------------------------------------------

void check_block_common(const torch::Tensor& a, const torch::Tensor& w,
                        const torch::Tensor& a_s, const torch::Tensor& w_s,
                        const torch::Tensor& lut,
                        const std::optional<torch::Tensor>& bias,
                        at::ScalarType out_dtype, int64_t block_n,
                        int64_t block_k, int64_t& w_stride, int& nkb) {
  TORCH_CHECK(a.is_cuda() && w.is_cuda() && a_s.is_cuda() && w_s.is_cuda() &&
                  lut.is_cuda(),
              "fp8_emu_block: all tensors must be on the GPU");
  TORCH_CHECK(a.element_size() == 1 && w.element_size() == 1,
              "fp8_emu_block: a/w must be 8-bit (fp8) tensors");
  TORCH_CHECK(a.scalar_type() == w.scalar_type(),
              "fp8_emu_block: a/w must have the same fp8 dtype");
  TORCH_CHECK(a_s.scalar_type() == at::kFloat && w_s.scalar_type() == at::kFloat,
              "fp8_emu_block: scales must be float32");
  TORCH_CHECK(lut.scalar_type() == at::kHalf && lut.numel() == 256,
              "fp8_emu_block: lut must be a 256-element fp16 tensor");
  TORCH_CHECK(a.is_contiguous() && a_s.is_contiguous() &&
                  w_s.is_contiguous() && lut.is_contiguous(),
              "fp8_emu_block: a/a_s/w_s/lut must be contiguous");
  // W may be padded along K by process_fp8_weight_block_strategy, so only the
  // innermost dimension has to be contiguous.
  TORCH_CHECK(w.stride(1) == 1 && w.stride(0) > 0,
              "fp8_emu_block: w must be K-contiguous (stride(1) == 1)");
  TORCH_CHECK(w.size(1) == a.size(1), "fp8_emu_block: K mismatch: a K=",
              a.size(1), " w K=", w.size(1));

  const int64_t K = a.size(1);
  TORCH_CHECK(block_k > 0 && block_k % 4 == 0,
              "fp8_emu_block: block_k must be a positive multiple of 4, got ",
              block_k);
  TORCH_CHECK(block_n > 0, "fp8_emu_block: block_n must be positive, got ",
              block_n);
  TORCH_CHECK(K % block_k == 0,
              "fp8_emu_block: K must be a multiple of block_k (K=", K,
              " block_k=", block_k, ")");
  nkb = (int)(K / block_k);
  w_stride = w.stride(0);
  TORCH_CHECK((w_stride % 4) == 0,
              "fp8_emu_block: w row stride must be a multiple of 4, got ",
              w_stride);

  const int64_t M = a.size(0), N = w.size(0), nblk = (N + block_n - 1) / block_n;
  TORCH_CHECK(a_s.dim() == 2 && a_s.size(0) == M && a_s.size(1) == nkb,
              "fp8_emu_block: a_s must be [M, cdiv(K, block_k)] = [", M, ", ",
              nkb, "], got ", a_s.sizes());
  TORCH_CHECK(w_s.dim() == 2 && w_s.size(0) == nblk && w_s.size(1) == nkb,
              "fp8_emu_block: w_s must be [cdiv(N, block_n), cdiv(K, block_k)]"
              " = [",
              nblk, ", ", nkb, "], got ", w_s.sizes());
  TORCH_CHECK(out_dtype == at::kHalf || out_dtype == at::kBFloat16 ||
                  out_dtype == at::kFloat,
              "fp8_emu_block: unsupported out dtype ", out_dtype);
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() &&
                    bias->numel() == w.size(0),
                "fp8_emu_block: bias must be a contiguous GPU tensor of N "
                "elements");
    TORCH_CHECK(bias->scalar_type() == out_dtype,
                "fp8_emu_block: bias dtype must match out dtype");
  }
}

template <typename T>
void dispatch_gemv_block(int M, int N, int K, int nt, int64_t w_stride, int nkb,
                         int block_n, bool has_a_s, const uint8_t* A,
                         const uint8_t* W, const uint32_t* a_h2,
                         const float* a_s, const float* w_s, const __half* lut,
                         const T* bias, T* C, hipStream_t stream) {
  launch_gemv_block<T>(M, N, K / 4, nt, w_stride, nkb, (int)(K / 4) / nkb,
                       block_n, has_a_s, A, W, a_h2, a_s, w_s, lut, bias, C,
                       stream);
}

// The GEMM path needs block_k to be a whole number of BK = 32 chunks so that a
// scale fold always lands on a chunk boundary.
template <typename T>
void dispatch_gemm_block_cfg(int cfg, int M, int N, int K, int64_t w_stride,
                             int nkb, int cpb, int block_n, const uint8_t* A,
                             const uint8_t* W, const float* a_s,
                             const float* w_s, const __half* lut,
                             const T* bias, T* C, hipStream_t stream) {
#define VLLM_FP8_EMU_GEMM_BLOCK_CASE(CFG, BM, BN, TM, TN)                  \
  case CFG:                                                                \
    launch_gemm_block<BM, BN, TM, TN, T>(M, N, K, w_stride, nkb, cpb,      \
                                         block_n, A, W, a_s, w_s, lut,     \
                                         bias, C, stream);                 \
    break
// Same tile with an explicit minimum-blocks-per-CU hint, which caps the VGPR
// budget (`__launch_bounds__(NT, MINB)` -> 65536/(NT*MINB) VGPR/thread).  The
// default cfg 1 tile needs 216 VGPR, i.e. <= 2 blocks/CU on 60 CUs, which is
// what the S18 profile blamed for the 9% memory-roofline efficiency.
#define VLLM_FP8_EMU_GEMM_BLOCK_CASE_MB(CFG, BM, BN, TM, TN, MINB)         \
  case CFG:                                                                \
    launch_gemm_block<BM, BN, TM, TN, T, MINB>(M, N, K, w_stride, nkb,     \
                                               cpb, block_n, A, W, a_s,    \
                                               w_s, lut, bias, C, stream); \
    break
// Multi-stage K pipeline (S23 / O3).  Numerically identical to the same tile
// in the 2-stage kernel above -- only the buffering depth differs -- so a cfg
// pair (16, N) is a clean A/B.
#define VLLM_FP8_EMU_GEMM_BLOCK_CASE_X(CFG, BK, BM, BN, TM, TN)            \
  case CFG:                                                                 \
    launch_gemm_block_wide2<BK, BM, BN, TM, TN, T>(                         \
        M, N, K, w_stride, nkb, cpb / (BK / kFBk), block_n, A, W, a_s, w_s, \
        lut, bias, C, stream);                                             \
    break
  // cfgs 32..35 stage BK = 64/128 elements per round; the scale fold at
  // `(c + 1) % cpb` only lands on a block_k boundary when block_k is a multiple
  // of BK, i.e. when the chunk count cpb is.
  if ((cfg == 32 || cfg == 35) && (cpb % (64 / kFBk)) != 0) cfg = 16;
  if ((cfg == 33 || cfg == 34) && (cpb % (128 / kFBk)) != 0) cfg = 16;
  // cfgs 36..39 (S24 L1) carry the same constraint into prefill; there the
  // fallback is the same BM/BN/TM/TN tile on the kFBk = 32 kernel, not the
  // decode tile, so a non-multiple block_k degrades to today's behaviour.
  // cfg 40 is the family's last untested design point: same 128-thread
  // geometry as the wide2 decode tiles with BM doubled to 16 (LDS ~13 KiB ->
  // 4-5 blocks/CU, between the decode tiles' 6 and 36-39's 1-2).
  if ((cfg == 36 || cfg == 37 || cfg == 39 || cfg == 40) &&
      (cpb % (64 / kFBk)) != 0)
    cfg = 5;
  if (cfg == 38 && (cpb % (128 / kFBk)) != 0) cfg = 4;
  switch (cfg) {
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(1, 32, 64, 4, 4);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(2, 64, 64, 4, 4);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(3, 64, 128, 4, 8);
    // 5: 256 threads, TM=2 -> 8 accumulators instead of 16 (register relief).
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(5, 32, 64, 2, 4);
    // 6: 256 threads, 4x4 -> 2 A + 2 B dwords staged per thread.
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(6, 32, 128, 2, 8);
    // 7: cfg 1 at 128 threads but forced to 4 / 6 blocks per CU (VGPR cap
    //    128 / 85).
    VLLM_FP8_EMU_GEMM_BLOCK_CASE_MB(7, 32, 64, 4, 4, 4);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE_MB(8, 32, 64, 4, 4, 6);
    // 9: cfg 2 (256 threads) forced to 3 blocks/CU (VGPR cap 85).
    VLLM_FP8_EMU_GEMM_BLOCK_CASE_MB(9, 64, 64, 4, 4, 3);
    // 10: cfg 5 with 4 blocks/CU (VGPR cap 64).
    VLLM_FP8_EMU_GEMM_BLOCK_CASE_MB(10, 32, 64, 2, 4, 4);
    // 11..15: decode tiles for M = 5..8.  The fallback tiles above are sized
    // for prefill (BM = 32/64/128); at M = 8 a BM = 32 tile computes 4x the
    // MACs that are needed and the epilogue discards 3/4 of them.  These tiles
    // are BM = 8 (exact M for the verify batch) and 128 threads, i.e. the same
    // per-thread work as cfg 5 at 1/4 the instruction count.  Measured: only
    // 1.0-1.2x over cfg 5/10 at M = 8, because these tiles have half the
    // resident threads of the BM=32 tile and the kernel is occupancy-bound --
    // cfgs 16..19 below restore the thread count (tools/dcu/s20_kernel_NOTES.md).
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(11, 8, 64, 1, 4);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE_MB(12, 8, 64, 1, 4, 4);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(13, 8, 128, 1, 8);
    // 14: BM = 16 (2x M waste) but TM = 2 keeps the per-thread A/B ratio.
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(14, 16, 64, 2, 4);
    // 15: 64 threads (one warp per row tile).
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(15, 8, 64, 1, 8);
    // 16..19: same exact-M idea but TN=2/TN=1, i.e. twice/4x the threads.  The
    // M=8 decode kernel is occupancy-bound, not issue-bound: total resident
    // threads are (N/BN)*NT = N*BM/(TM*TN), so TN=4 (cfg 11) leaves only 4.5
    // warps/CU at N=8704 while the BM=32 fallback tile runs 9.07.  Measured
    // (tools/dcu/s20_kernel_NOTES.md S20 6): cfg 11/12 0.36-0.39 ms vs cfg
    // 13/15 0.44-0.52 ms, ordered exactly by warps/CU; cfg 16 (TN=2) then
    // reaches 1.7-1.8x over the legacy dispatch at the production shapes.
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(16, 8, 64, 1, 2);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(17, 8, 32, 1, 2);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(18, 8, 64, 1, 1);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(19, 8, 128, 1, 2);
    // 32..35: cfg 16's tile with a wider K chunk on the original two-stage
    // buffer (see the kernel comment). 
    VLLM_FP8_EMU_GEMM_BLOCK_CASE_X(32, 64, 8, 64, 1, 2);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE_X(33, 128, 8, 64, 1, 2);
    // 34/35: BN = 32 so that a wide K chunk still fits three blocks per CU
    // (LDS 2 x (BM + BN) x (BK/2 + 1) x 4 = 20.8 / 10.6 KiB).  BN = 32 also
    // doubles the grid, which is the only way to put more than 12 blocks on a
    // 60-CU device at N = 768.
    VLLM_FP8_EMU_GEMM_BLOCK_CASE_X(34, 128, 8, 32, 1, 2);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE_X(35, 64, 8, 32, 1, 2);
    // 36..39 (S24 L1): the wide2 kernel family extended to prefill shapes.
    // S24 measured the prefill GEMM at 8.0-8.8 TF/s = 41-43% of the dot2 peak
    // with the cfg 0 default tile (docs/design/z100-longctx-vram-optimization-
    // plan.md §1.2), and the M-scan (mscan-dcu3.log) puts cfg 3 ahead of every
    // legacy tile at M >= 1024.  These stage BK = 64/128 elements per round to
    // halve the staging rounds, which is the same lever the S23 decode tiles
    // cashed (+15-24%).  Accumulation order is unchanged (k ascending, dot2
    // pairs, fold on the 128-element block boundary), so every prefill cfg is
    // bitwise identical to cfg 0.  LDS/CU: 36 = 25.3 KiB, 37 = 33.8, 38 = 33.3,
    // 39 = 50.7 (gfx906 has 64 KiB per CU).
    VLLM_FP8_EMU_GEMM_BLOCK_CASE_X(36, 64, 32, 64, 2, 4);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE_X(37, 64, 64, 64, 4, 4);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE_X(38, 128, 32, 32, 2, 2);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE_X(39, 64, 64, 128, 4, 8);
    // 40 (S24 L1 residual): BM16/BN32 @ BK64.
    VLLM_FP8_EMU_GEMM_BLOCK_CASE_X(40, 64, 16, 32, 2, 2);
    case 4:
    default:
      launch_gemm_block<128, 128, 8, 8, T>(M, N, K, w_stride, nkb, cpb,
                                           block_n, A, W, a_s, w_s, lut, bias,
                                           C, stream);
      break;
  }
#undef VLLM_FP8_EMU_GEMM_BLOCK_CASE
#undef VLLM_FP8_EMU_GEMM_BLOCK_CASE_MB
#undef VLLM_FP8_EMU_GEMM_BLOCK_CASE_X
}

// Decode (M = 5..8) tiled-GEMM configuration.  `VLLM_DCU_FP8_DECODE_CFG` is an
// A/B knob (no rebuild): unset or 0 keeps the pre-S20 policy (cfg 5, or cfg 10
// for wide N); 11..15 force one of the exact-M tiles above; anything else picks
// the auto rule.  The default is deliberately the legacy policy until the
// end-to-end A/B accepts the new tiles.
int decode_gemm_cfg(int N) {
  const int wide = 16;  // auto rule: exact-M tile with TN=2 (occupancy)
  const char* e = std::getenv("VLLM_DCU_FP8_DECODE_CFG");
  if (e == nullptr) return (N >= 6144) ? 10 : 5;
  if (e[0] == 'a' || e[0] == 'A') return wide;  // "auto"
  const int v = std::atoi(e);
  if (v >= 11 && v <= 19) return v;
  if (v >= 32 && v <= 35) return v;
  if (v == 0) return (N >= 6144) ? 10 : 5;
  return wide;
}

void run_block(const torch::Tensor& a, const torch::Tensor& w,
               const torch::Tensor& a_s, const torch::Tensor& w_s,
               const torch::Tensor& lut,
               const std::optional<torch::Tensor>& bias, torch::Tensor& out,
               bool gemv, int nthreads, int cfg, int64_t block_n,
               int64_t block_k) {
  int64_t w_stride = 0;
  int nkb = 0;
  check_block_common(a, w, a_s, w_s, lut, bias, out.scalar_type(), block_n,
                     block_k, w_stride, nkb);
  const int M = (int)a.size(0), N = (int)w.size(0), K = (int)a.size(1);
  const void* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined()) bias_ptr = bias->data_ptr();

  const uint8_t* A = reinterpret_cast<const uint8_t*>(a.data_ptr());
  const uint8_t* W = reinterpret_cast<const uint8_t*>(w.data_ptr());
  const float* a_s_p = reinterpret_cast<const float*>(a_s.data_ptr());
  const float* w_s_p = reinterpret_cast<const float*>(w_s.data_ptr());
  const __half* lut_p = reinterpret_cast<const __half*>(lut.data_ptr());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  // Decode (gemv) picks the streaming kernel when its geometry fits: it is the
  // path the S18 probe measures several times faster than the tiled GEMM on
  // the real N=4352/K=5120 shard.  `VLLM_DCU_FP8_GEMV_STREAM` selects it
  // (0 = old GEMV, 1 = streaming variant 1 (default), 2 = variant 2).
  int s_rows = 0, s_nt = 0;
  const int s_mode =
      gemv ? stream_block_select(M, N, K, w_stride, nkb, (int)block_n,
                                 (int)block_k, stream_block_mode(), s_rows,
                                 s_nt)
           : 0;
  const bool use_stream = (s_mode != 0);

  // Shape-aware decode policy (perf1 s1 table, K=5120): the streaming GEMV
  // covers M = 1..8 whenever its slice-based LDS budget fits (one A slice +
  // folded scales; GEMV-OPT-20261025.md P0 lifted the old whole-A-tile gate
  // that pushed M >= 5 onto the tiled GEMM), otherwise the tiled GEMM, which
  // is 3-10x faster than the old GEMV was (M=8: GEMM 245-419 us vs old GEMV
  // 704-3628 us).  cfg 5 (32x64 TM2TN4, 256 threads) wins for N <= 5120 and
  // cfg 10 (same tile at 4 blocks/CU) for wide N; the crossover sits between
  // the measured 5120 and 8704 points.  Prefill (M > 8) never enters this
  // function.
  const int decode_cfg = decode_gemm_cfg(N);
  const bool gemv_use_gemm = gemv && !use_stream && M > 4 &&
                             (block_k % kFBk) == 0;

  const uint32_t* a_h2 = nullptr;
  at::Tensor a_scratch;
  if (gemv && !use_stream && !gemv_use_gemm && M >= 2) {
    a_scratch = at::empty({(int64_t)M * K}, a.options().dtype(at::kHalf));
    a_h2 = reinterpret_cast<const uint32_t*>(a_scratch.data_ptr());
    const int64_t n_dwords = (int64_t)M * K / 4;
    const int threads = 256;
    fp8_emu_widen_a_kernel<<<(unsigned)((n_dwords + threads - 1) / threads),
                             threads, 0, stream>>>(A, lut_p,
                                                   const_cast<uint32_t*>(a_h2),
                                                   n_dwords);
  }

  const int cpb = (int)(block_k / kFBk);
  if (!gemv) {
    TORCH_CHECK(block_k % kFBk == 0,
                "fp8_emu_block: block_k must be a multiple of ", kFBk,
                " for the GEMM path, got ", block_k);
  }

#define VLLM_FP8_EMU_BLOCK_DISPATCH(T, AT)                                   \
  do {                                                                       \
    if (gemv && !gemv_use_gemm) {                                            \
      if (use_stream) {                                                      \
        if (s_mode == 2) {                                                   \
          dispatch_gemv_stream2<T>(M, N, K, w_stride, s_rows, s_nt, nkb,     \
                                   (int)block_n,                             \
                                   (int)(block_k / kStreamChunk), false, A,  \
                                   W, a_s_p, w_s_p, lut_p,                   \
                                   reinterpret_cast<const AT*>(bias_ptr),    \
                                   reinterpret_cast<AT*>(out.data_ptr()),    \
                                   stream);                                  \
        } else {                                                             \
          dispatch_gemv_stream<T>(M, N, K, w_stride, s_rows, s_nt, nkb,      \
                                  (int)block_n,                              \
                                  (int)(block_k / kStreamChunk), false,      \
                                  A, W, a_s_p, w_s_p, lut_p,                 \
                                  reinterpret_cast<const AT*>(bias_ptr),     \
                                  reinterpret_cast<AT*>(out.data_ptr()),     \
                                  stream);                                   \
        }                                                                    \
      } else {                                                               \
        dispatch_gemv_block<T>(M, N, K, nthreads, w_stride, nkb,             \
                               (int)block_n, true, A, W, a_h2, a_s_p, w_s_p, \
                               lut_p, reinterpret_cast<const AT*>(bias_ptr), \
                               reinterpret_cast<AT*>(out.data_ptr()),        \
                               stream);                                      \
      }                                                                      \
    } else {                                                                 \
      dispatch_gemm_block_cfg<T>(gemv ? decode_cfg : cfg, M, N, K,      \
                                 w_stride, nkb, cpb, (int)block_n, A, W,     \
                                 a_s_p, w_s_p, lut_p,                        \
                                 reinterpret_cast<const AT*>(bias_ptr),      \
                                 reinterpret_cast<AT*>(out.data_ptr()),      \
                                 stream);                                    \
    }                                                                        \
  } while (0)
  switch (out.scalar_type()) {
    case at::kBFloat16:
      VLLM_FP8_EMU_BLOCK_DISPATCH(at::BFloat16, at::BFloat16);
      break;
    case at::kHalf:
      VLLM_FP8_EMU_BLOCK_DISPATCH(at::Half, at::Half);
      break;
    default:
      VLLM_FP8_EMU_BLOCK_DISPATCH(float, float);
      break;
  }
#undef VLLM_FP8_EMU_BLOCK_DISPATCH
}


// ---------------------------------------------------------------------------
// W8A16 host side
// ---------------------------------------------------------------------------
// GEMV threshold for the W8A16 M branch.  The W8A8 route splits gemv/gemm on
// the python side with `_GEMV_MAX_M` (VLLM_DCU_FP8_GEMV_MAX_M); the W8A16
// entry owns the branch (routing matrix in
// docs/design/z100-w8a16-prefill-fp16-construction.md §3), so it reads the
// same knob once per process here (invariant 5: never per call) and clamps it
// to the compiled GEMV range.
int w8a16_gemv_max_m() {
  static const int m = []() {
    const char* e = std::getenv("VLLM_DCU_FP8_GEMV_MAX_M");
    if (e == nullptr) return kMsMax;
    const int v = std::atoi(e);
    if (v < 1) return 1;
    return v > kMsMax ? kMsMax : v;
  }();
  return m;
}

void check_block_w8a16(const torch::Tensor& a, const torch::Tensor& w,
                       const torch::Tensor& w_s, const torch::Tensor& lut,
                       const std::optional<torch::Tensor>& bias,
                       at::ScalarType out_dtype, int64_t block_n,
                       int64_t block_k, int64_t& w_stride, int& nkb) {
  TORCH_CHECK(a.is_cuda() && w.is_cuda() && w_s.is_cuda() && lut.is_cuda(),
              "fp8_emu_block_w8a16: all tensors must be on the GPU");
  // The kernels index every operand on one device; several torch ops on
  // different devices would otherwise race instead of failing here.
  TORCH_CHECK(a.device() == w.device() && a.device() == w_s.device() &&
                  a.device() == lut.device() &&
                  (!bias.has_value() || !bias->defined() ||
                   bias->device() == a.device()),
              "fp8_emu_block_w8a16: all tensors must be on the same device");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2,
              "fp8_emu_block_w8a16: a and w must be 2-D, got ",
              a.dim(), " and ", w.dim());
  TORCH_CHECK(a.scalar_type() == at::kHalf,
              "fp8_emu_block_w8a16: activations must be float16 (gfx906 has no "
              "bf16 datapath), got ",
              a.scalar_type());
  TORCH_CHECK(a.is_contiguous(),
              "fp8_emu_block_w8a16: `a` must be contiguous");
  // Unlike the W8A8 route (whose A is a fresh quantizer output), the W8A16
  // activation is the caller's raw tensor and every kernel loads it through
  // 4-byte dwords (the streaming variant through 16-byte uint4, gated in
  // `run_block_w8a16`), so an odd-element view must fail loudly instead of
  // trapping on a misaligned address.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 4 == 0,
              "fp8_emu_block_w8a16: `a` must be 4-byte aligned (got a "
              "2-byte-aligned view); call .contiguous() on it");
  TORCH_CHECK(w.element_size() == 1,
              "fp8_emu_block_w8a16: `w` must be an 8-bit (fp8) tensor");
  TORCH_CHECK(w.stride(1) == 1 && w.stride(0) > 0,
              "fp8_emu_block_w8a16: w must be K-contiguous (stride(1) == 1)");
  TORCH_CHECK(w_s.scalar_type() == at::kFloat && w_s.is_contiguous(),
              "fp8_emu_block_w8a16: w_s must be a contiguous float32 tensor");
  TORCH_CHECK(lut.scalar_type() == at::kHalf && lut.numel() == 256 &&
                  lut.is_contiguous(),
              "fp8_emu_block_w8a16: lut must be a 256-element fp16 tensor");
  TORCH_CHECK(w.size(1) == a.size(1), "fp8_emu_block_w8a16: K mismatch: a K=",
              a.size(1), " w K=", w.size(1));
  const int64_t K = a.size(1), M = a.size(0), N = w.size(0);
  TORCH_CHECK(K % 8 == 0, "fp8_emu_block_w8a16: K must be a multiple of 8, got ",
              K);
  TORCH_CHECK(block_n > 0,
              "fp8_emu_block_w8a16: block_n must be positive, got ", block_n);
  TORCH_CHECK(block_k > 0 && block_k % 32 == 0,
              "fp8_emu_block_w8a16: block_k must be a positive multiple of 32, "
              "got ",
              block_k);
  TORCH_CHECK(K % block_k == 0,
              "fp8_emu_block_w8a16: K must be a multiple of block_k (K=", K,
              " block_k=", block_k, ")");
  nkb = (int)(K / block_k);
  w_stride = w.stride(0);
  TORCH_CHECK((w_stride % 8) == 0,
              "fp8_emu_block_w8a16: w row stride must be a multiple of 8, got ",
              w_stride);
  TORCH_CHECK(w_stride >= K,
              "fp8_emu_block_w8a16: w row stride must cover K (stride(0)=",
              w_stride, " K=", K, ")");
  // W is read as uint32 dwords by both the GEMV and the tile kernels.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 4 == 0,
              "fp8_emu_block_w8a16: `w` data_ptr must be 4-byte aligned");
  const int64_t nblk = (N + block_n - 1) / block_n;
  TORCH_CHECK(w_s.dim() == 2 && w_s.size(0) == nblk && w_s.size(1) == nkb,
              "fp8_emu_block_w8a16: w_s must be [cdiv(N, block_n), "
              "cdiv(K, block_k)] = [",
              nblk, ", ", nkb, "], got ", w_s.sizes());
  TORCH_CHECK(out_dtype == at::kHalf || out_dtype == at::kBFloat16 ||
                  out_dtype == at::kFloat,
              "fp8_emu_block_w8a16: unsupported out dtype ", out_dtype);
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() &&
                    bias->numel() == N,
                "fp8_emu_block_w8a16: bias must be a contiguous GPU tensor of "
                "N elements");
    TORCH_CHECK(bias->scalar_type() == out_dtype,
                "fp8_emu_block_w8a16: bias dtype must match out dtype");
  }
  (void)M;
}

void run_block_w8a16(const torch::Tensor& a, const torch::Tensor& w,
                     const torch::Tensor& w_s, const torch::Tensor& lut,
                     const std::optional<torch::Tensor>& bias, torch::Tensor& out,
                     bool gemv, int nthreads, int cfg, int64_t block_n,
                     int64_t block_k) {
  int64_t w_stride = 0;
  int nkb = 0;
  check_block_w8a16(a, w, w_s, lut, bias, out.scalar_type(), block_n, block_k,
                    w_stride, nkb);
  const int M = (int)a.size(0), N = (int)w.size(0), K = (int)a.size(1);
  const void* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined()) bias_ptr = bias->data_ptr();

  const uint8_t* A = reinterpret_cast<const uint8_t*>(a.data_ptr());
  const __half* A16 = reinterpret_cast<const __half*>(a.data_ptr());
  // The W8A16 GEMV always reads A as fp16 pairs, i.e. through the PRECONV path
  // that the W8A8 route only enters after a separate widening launch.
  const uint32_t* a_h2 = reinterpret_cast<const uint32_t*>(a.data_ptr());
  const uint8_t* W = reinterpret_cast<const uint8_t*>(w.data_ptr());
  const float* w_s_p = reinterpret_cast<const float*>(w_s.data_ptr());
  const __half* lut_p = reinterpret_cast<const __half*>(lut.data_ptr());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  int s_rows = 0, s_nt = 0;
  const int s_mode =
      gemv ? stream_block_select(M, N, K, w_stride, nkb, (int)block_n,
                                 (int)block_k, stream_block_mode(), s_rows,
                                 s_nt)
           : 0;
  bool use_stream = (s_mode != 0);
  // The streaming kernels copy A as flat uint4 (16 B), one step beyond the
  // 4-byte guarantee checked above.  An activation that only meets the dword
  // alignment still runs -- on the non-streaming GEMV, whose staging reads
  // dwords -- so this stays a silent, correctness-preserving fallback.
  if (use_stream && (reinterpret_cast<uintptr_t>(a.data_ptr()) % 16) != 0)
    use_stream = false;
  // Same decode policy as the W8A8 route: M = 5..8 is M-flat on the tile and
  // cheaper there than on the streaming GEMV.
  const bool gemv_use_gemm = gemv && !use_stream && M > 4;
  const int c = w8a16_cfg(M, N, block_k, cfg);

// The W8A16 arm has no activation scale.  The kernels never read `a_s` when
// `a16` (or the HAS_A_S template argument) is off, but pass a device-valid
// pointer -- the weight scale -- instead of nullptr so that stays independent
// of how the compiler if-converts the uniform branch.
#define VLLM_FP8_EMU_BLOCK_W8A16(T, AT)                                      \
  do {                                                                       \
    if (gemv && !gemv_use_gemm) {                                            \
      if (use_stream) {                                                      \
        if (M == 1 && (stream_m1_enabled() || wholea_enabled())) {           \
          if (wholea_enabled() && wholea_shape_ok(K, nkb)) {                 \
            if (wholea_l2_enabled()) {                                       \
              dispatch_gemv_wholea_l2_m1<T>(                                 \
                  N, K, w_stride, s_rows, s_nt, nkb, (int)block_n,           \
                  (int)(block_k / kStreamChunk), A, W, w_s_p, lut_p,         \
                  reinterpret_cast<const AT*>(bias_ptr),                     \
                  reinterpret_cast<AT*>(out.data_ptr()), stream);            \
            } else {                                                         \
              dispatch_gemv_wholea_m1<T>(N, K, w_stride, s_rows, s_nt, nkb,  \
                                         (int)block_n,                       \
                                         (int)(block_k / kStreamChunk), A,   \
                                         W, w_s_p, lut_p,                    \
                                         reinterpret_cast<const AT*>(        \
                                             bias_ptr),                      \
                                         reinterpret_cast<AT*>(              \
                                             out.data_ptr()),                \
                                         stream);                            \
            }                                                                \
          } else {                                                           \
            dispatch_gemv_stream_m1<T>(N, K, w_stride, s_rows, s_nt, nkb,    \
                                       (int)block_n,                         \
                                       (int)(block_k / kStreamChunk), A, W,  \
                                       w_s_p, lut_p,                         \
                                       reinterpret_cast<const AT*>(bias_ptr),\
                                       reinterpret_cast<AT*>(out.data_ptr()),\
                                       stream);                              \
          }                                                                  \
        } else if (s_mode == 2) {                                            \
          dispatch_gemv_stream2<T>(M, N, K, w_stride, s_rows, s_nt, nkb,     \
                                   (int)block_n,                            \
                                   (int)(block_k / kStreamChunk), true, A,  \
                                   W, w_s_p, w_s_p, lut_p,                  \
                                   reinterpret_cast<const AT*>(bias_ptr),    \
                                   reinterpret_cast<AT*>(out.data_ptr()),    \
                                   stream);                                  \
        } else {                                                             \
          dispatch_gemv_stream<T>(M, N, K, w_stride, s_rows, s_nt, nkb,      \
                                  (int)block_n,                              \
                                  (int)(block_k / kStreamChunk), true,       \
                                  A, W, w_s_p, w_s_p, lut_p,                 \
                                  reinterpret_cast<const AT*>(bias_ptr),     \
                                  reinterpret_cast<AT*>(out.data_ptr()),     \
                                  stream);                                   \
        }                                                                    \
      } else {                                                               \
        dispatch_gemv_block<T>(M, N, K, nthreads, w_stride, nkb,             \
                               (int)block_n, /*has_a_s=*/false, A, W, a_h2,  \
                               w_s_p, w_s_p, lut_p,                          \
                               reinterpret_cast<const AT*>(bias_ptr),        \
                               reinterpret_cast<AT*>(out.data_ptr()),        \
                               stream);                                      \
      }                                                                      \
    } else {                                                                 \
      dispatch_gemm_block_w8a16<T>(c, M, N, K, w_stride, nkb, (int)block_k,  \
                                   (int)block_n, A16, W, w_s_p, lut_p,       \
                                   reinterpret_cast<const AT*>(bias_ptr),    \
                                   reinterpret_cast<AT*>(out.data_ptr()),    \
                                   stream);                                  \
    }                                                                        \
  } while (0)
  switch (out.scalar_type()) {
    case at::kBFloat16:
      VLLM_FP8_EMU_BLOCK_W8A16(at::BFloat16, at::BFloat16);
      break;
    case at::kHalf:
      VLLM_FP8_EMU_BLOCK_W8A16(at::Half, at::Half);
      break;
    default:
      VLLM_FP8_EMU_BLOCK_W8A16(float, float);
      break;
  }
#undef VLLM_FP8_EMU_BLOCK_W8A16
}

// ---------------------------------------------------------------------------
// S23 / O5: activation quantization fused into the block-scaled decode GEMM
// ---------------------------------------------------------------------------
// Motivation: a decode step of the 27B model issues 256 activation-quant
// launches + 256 fp8-emu GEMM launches.  Fusing removes the quant launch and
// the A round trip through HBM (~2.15 ms/step of GPU quant time plus 256
// launches), and it is cheap here because A is only M*K bytes (40 KiB at M=8
// and K=5120).
//
// Bit-exactness contract.  The quantizer below reproduces
// `csrc/quantization/w8a8/fp8/per_token_group_quant.cu:189-300`
// (per_token_group_quant_8bit_register_kernel, GROUP_SIZE 128) step for step:
//   1. absmax seeded with `eps`, `fmaxf(m, fabsf(v))` over the group's 128
//      elements.  The maximum is exact and order independent, so a different
//      element-to-lane partition (32 lanes x 4 elements here, 8 lanes x 16
//      there) yields the same value.
//   2. `y_s = m / 448.0f` (a real division, not a reciprocal multiply).
//   3. `inv_y = 1.0f / y_s` (a second real division); `scaled = v * inv_y`.
//   4. `q = fminf(fmaxf(scaled, -448.0f), 448.0f)` then the e4m3 cast through
//      `c10::Float8_e4m3fn`, i.e. the very same header conversion the reference
//      kernel uses, so rounding/saturation cannot drift.
//   5. `x_s` = `y_s` per 128-element group, fp32.
// The resulting fp8 code is then widened through the same 256-entry LDS LUT as
// the unfused path (lossless), so the dot2 stream and the `(acc*a_s)*w_s`
// epilogue order are untouched.
constexpr float kQFp8Max = 448.0f;   // e4m3 `max_8bit`
constexpr float kQFp8Min = -448.0f;
constexpr int kQGroup = 128;         // production block_k / quant group
constexpr int kQMaxNkb = 64;         // static sA budget: K <= 8192 at block_k 128

// One warp per (row, group): 32 lanes x 4 elements.  Standalone quantizer, used
// by the stream fallback (which must feed the existing stream kernel two real
// buffers) and by the `_quant_x` witness variant.
template <typename TA>
__global__ void fp8_emu_quant_a_kernel(const TA* __restrict__ a,
                                       uint8_t* __restrict__ xq,
                                       float* __restrict__ xs, int64_t ngroups,
                                       float eps) {
  const int64_t g = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (g >= ngroups) return;
  const int lane = threadIdx.x & 31;
  const int64_t base = g * kQGroup + lane * 4;
  float v[4];
  const TA* p = a + base;
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = static_cast<float>(p[i]);
  float m = eps;
#pragma unroll
  for (int i = 0; i < 4; ++i) m = fmaxf(m, fabsf(v[i]));
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_xor_sync((unsigned long long)0xffffffffu, m, off, 32));
  const float y_s = m / kQFp8Max;
  const float inv_y = 1.0f / y_s;
  if (lane == 0) xs[g] = y_s;
  uint8_t* q = xq + base;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float sc = fminf(fmaxf(v[i] * inv_y, kQFp8Min), kQFp8Max);
    c10::Float8_e4m3fn qb(sc);
    q[i] = *reinterpret_cast<const uint8_t*>(&qb);
  }
}

// Fused tile: the two-stage wide-chunk skeleton of
// `fp8_emu_gemm_block_wide2_kernel` with BK == block_k == 128, so one K chunk is
// exactly one quant group and the per-group scale is produced where it is
// consumed.  Geometry is fixed (BM=8, BN=64, TM=1, TN=2, NT=256) because the
// A quantizer needs one contiguous 32-lane group per row: with NT=256 and
// A-fragment count 1, thread `tid` owns row `tid >> 5`, position `tid & 31`.
// `a_s` (y_s) lives in LDS ([BM][nkb] fp32 = 1.28 KiB), so the hot path
// allocates `out` and nothing else.
constexpr int kQbk = kQGroup;
constexpr int kQkw4 = kQbk / 4;    // 32 fp8 dwords per row per chunk
constexpr int kQkp = kQbk / 2 + 1; // 65: padded fp16 LDS row stride

template <int BM, int BN, int TM, int TN, typename T, typename TA>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    fp8_emu_gemm_block_q_kernel(const TA* __restrict__ A,
                                const uint8_t* __restrict__ W,
                                const int64_t w_stride,
                                const float* __restrict__ w_s, const int nkb,
                                const int block_n, const __half* __restrict__ lut,
                                const T* __restrict__ bias, T* __restrict__ C,
                                int M, int N, int K, float eps,
                                uint8_t* __restrict__ xq_w,
                                float* __restrict__ xs_w) {
  using C_ = FConf<BM, BN, TM, TN>;
  constexpr int NT = C_::NT;
  constexpr int B4 = (BN * kQkw4 + NT - 1) / NT;
  static_assert(C_::MG * TM == BM && C_::NG * TN == BN, "tile/thread mismatch");
  static_assert(NT == BM * 32, "A quantizer needs one 32-lane group per row");
  static_assert(BM * kQkw4 == NT, "A fragment count must be 1");

  __shared__ uint32_t As[2][BM * kQkp];
  __shared__ uint32_t Bs[2][BN * kQkp];
  __shared__ float sA[BM][kQMaxNkb];
  __shared__ __half sl[256];

  const int tid = threadIdx.x;
  const int tx = tid % C_::NG;
  const int ty = tid / C_::NG;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int K4 = K >> 2;
  const int w4 = (int)(w_stride >> 2);
  const int NK = (K + kQbk - 1) / kQbk;
  const int ar = tid >> 5;    // A row owned by this thread
  const int ajd = tid & 31;   // 4-element position inside the group

  fp8_lut_stage(sl, lut, NT);
  __syncthreads();

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
  }

  const int row_a = m0 + ar;
  const bool arow = row_a < M;

  // Stage chunk `c`: quantize the A group (one per row), widen through the LUT,
  // and store the B slice exactly as the unfused two-stage kernel does.
  auto stage = [&](int c, int slot, const uint8_t* __restrict__ Wsrc,
                   int k0) {
    // ---- A: 4 elements per lane, 32 lanes per row-group ----
    float v[4];
    const int64_t abase = (int64_t)row_a * K + k0 + ajd * 4;
    if (arow) {
      const TA* p = A + abase;
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = static_cast<float>(p[i]);
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = 0.0f;
    }
    float mx = eps;
#pragma unroll
    for (int i = 0; i < 4; ++i) mx = fmaxf(mx, fabsf(v[i]));
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    mx = fmaxf(mx, __shfl_xor_sync((unsigned long long)0xffffffffu, mx, off, 32));
    const float y_s = mx / kQFp8Max;
    const float inv_y = 1.0f / y_s;
    if (ajd == 0) sA[ar][c] = y_s;
    uint32_t lo, hi;
    {
      uint8_t q[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float sc = fminf(fmaxf(v[i] * inv_y, kQFp8Min), kQFp8Max);
        c10::Float8_e4m3fn qb(sc);
        q[i] = *reinterpret_cast<const uint8_t*>(&qb);
      }
      lo = (uint32_t)fp8_lut_u16(sl, q[0]) |
           ((uint32_t)fp8_lut_u16(sl, q[1]) << 16);
      hi = (uint32_t)fp8_lut_u16(sl, q[2]) |
           ((uint32_t)fp8_lut_u16(sl, q[3]) << 16);
      // The witness is redundant across grid.x (every N block quantizes the
      // same A groups), so only grid column 0 writes it.
      if (xq_w != nullptr && arow && blockIdx.x == 0) {
        uint8_t* dst = xq_w + abase;
#pragma unroll
        for (int i = 0; i < 4; ++i) dst[i] = q[i];
      }
    }
    if (xs_w != nullptr && arow && ajd == 0 && blockIdx.x == 0)
      xs_w[(int64_t)row_a * nkb + c] = y_s;
    As[slot][ar * kQkp + 2 * ajd] = lo;
    As[slot][ar * kQkp + 2 * ajd + 1] = hi;

    // ---- B: same staging as fp8_gemm_load_wide + fp8_gemm_store_wide ----
#pragma unroll
    for (int u = 0; u < B4; ++u) {
      const int idx = tid + u * NT;
      const int r = idx >> 5;
      const int jd = idx & 31;
      const bool ok = idx < BN * kQkw4 && n0 + r < N && (k0 + kQbk) <= K;
      const uint32_t vv =
          ok ? reinterpret_cast<const uint32_t*>(Wsrc + (int64_t)(n0 + r) * w_stride +
                                                k0)[jd]
             : 0u;
      uint32_t wlo, whi;
      fp8x4_to_f16x2x2(sl, vv, wlo, whi);
      if (idx < BN * kQkw4) {
        Bs[slot][r * kQkp + 2 * jd] = wlo;
        Bs[slot][r * kQkp + 2 * jd + 1] = whi;
      }
    }
  };

  // compute_wide ACCUMULATES into blk, so it must start at zero: an
  // uninitialized blk adds register garbage for the first group and makes the
  // whole tile nondeterministic (measured: inf on 768:5120, then a finite
  // value that matched the BK=128 reference on the next run).
  float blk[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) blk[i][j] = 0.0f;
  }
  auto consume = [&](int c, int slot) {
    fp8_gemm_compute_wide<kQbk, BM, BN, TM, TN>(tx, ty, As[slot], Bs[slot], blk);
    // Fold this group's scales; cpb == 1 because BK == block_k, so every chunk
    // is exactly one scale block.  `acc += blk * sa * w_s` -- the same order as
    // fp8_gemm_fold_block_scale, just with the 128-element group instead of a
    // block_k window.
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int row = ty + C_::MG * i;
      if (m0 + row >= M) continue;
      const float sa = sA[row][c];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int col = n0 + tx + C_::NG * j;
        if (col >= N) continue;
        acc[i][j] += blk[i][j] * sa * w_s[(int64_t)(col / block_n) * nkb + c];
        blk[i][j] = 0.0f;
      }
    }
  };

  // Two-stage pipeline, identical to the unfused kernel modulo the fused A
  // staging (barrier after stage, consume then next stage).
  int c = 0;
  for (; c + 1 < NK; c += 2) {
    stage(c, 0, W, c * kQbk);
    __syncthreads();
    consume(c, 0);
    stage(c + 1, 1, W, (c + 1) * kQbk);
    __syncthreads();
    consume(c + 1, 1);
  }
  if (c < NK) {
    stage(c, 0, W, c * kQbk);
    __syncthreads();
    consume(c, 0);
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + C_::MG * i;
    if (row >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + tx + C_::NG * j;
      if (col >= N) continue;
      T o = static_cast<T>(acc[i][j]);
      if (bias != nullptr) o += static_cast<T>(bias[col]);
      C[(int64_t)row * N + col] = o;
    }
  }
}

}  // namespace

torch::Tensor fp8_emu_gemv(torch::Tensor a, torch::Tensor w, torch::Tensor a_s,
                           torch::Tensor w_s, torch::Tensor lut,
                           std::optional<torch::Tensor> bias,
                           at::ScalarType out_dtype, int64_t nthreads) {
  const int64_t M = a.size(0), K = a.size(1), N = w.size(0);
  TORCH_CHECK(K % 4 == 0, "fp8_emu: K must be a multiple of 4, got ", K);
  TORCH_CHECK(K >= 4, "fp8_emu: K must be >= 4, got ", K);
  TORCH_CHECK(M >= 1 && M <= kMsMax, "fp8_emu: M must be in [1,", kMsMax,
              "] for the GEMV path, got ", M);
  TORCH_CHECK(nthreads > 0 && nthreads <= kNtMax &&
                  (nthreads & (nthreads - 1)) == 0,
              "fp8_emu: nthreads must be a power of two <= ", kNtMax);

  const c10::cuda::CUDAGuard guard(a.device());
  auto out = at::empty({M, N}, a_s.options().dtype(out_dtype));
  run(a, w, a_s, w_s, lut, bias, out, /*gemv=*/true, (int)nthreads, 0);
  return out;
}

torch::Tensor fp8_emu_gemm(torch::Tensor a, torch::Tensor w, torch::Tensor a_s,
                           torch::Tensor w_s, torch::Tensor lut,
                           std::optional<torch::Tensor> bias,
                           at::ScalarType out_dtype, int64_t cfg) {
  const int64_t M = a.size(0), K = a.size(1), N = w.size(0);
  TORCH_CHECK(K % 4 == 0 && K >= 4,
              "fp8_emu: K must be a positive multiple of 4, got ", K);
  TORCH_CHECK(M >= 1, "fp8_emu: M must be >= 1");
  TORCH_CHECK(cfg >= 0 && cfg <= 4, "fp8_emu: cfg must be 0..4, got ", cfg);

  const c10::cuda::CUDAGuard guard(a.device());
  auto out = at::empty({M, N}, a_s.options().dtype(out_dtype));
  int c = (int)cfg;
  if (c == 0) c = (M <= 32) ? 1 : 2;  // tuned: 32x64 for shallow M, else 64x64
  run(a, w, a_s, w_s, lut, bias, out, /*gemv=*/false, 0, c);
  return out;
}

torch::Tensor fp8_emu_gemv_meta(torch::Tensor a, torch::Tensor w,
                                torch::Tensor a_s, torch::Tensor w_s,
                                torch::Tensor lut,
                                std::optional<torch::Tensor> bias,
                                at::ScalarType out_dtype, int64_t nthreads) {
  return a.new_empty({a.size(0), w.size(0)},
                     a_s.options().dtype(out_dtype));
}

torch::Tensor fp8_emu_gemm_meta(torch::Tensor a, torch::Tensor w,
                                torch::Tensor a_s, torch::Tensor w_s,
                                torch::Tensor lut,
                                std::optional<torch::Tensor> bias,
                                at::ScalarType out_dtype, int64_t cfg) {
  return a.new_empty({a.size(0), w.size(0)},
                     a_s.options().dtype(out_dtype));
}

torch::Tensor fp8_emu_gemv_block(torch::Tensor a, torch::Tensor w,
                                 torch::Tensor a_s, torch::Tensor w_s,
                                 torch::Tensor lut,
                                 std::optional<torch::Tensor> bias,
                                 at::ScalarType out_dtype, int64_t nthreads,
                                 int64_t block_n, int64_t block_k) {
  const int64_t M = a.size(0), K = a.size(1), N = w.size(0);
  TORCH_CHECK(K % 4 == 0 && K >= 4,
              "fp8_emu_block: K must be a positive multiple of 4, got ", K);
  TORCH_CHECK(M >= 1 && M <= kMsMax, "fp8_emu_block: M must be in [1,",
              kMsMax, "] for the GEMV path, got ", M);
  TORCH_CHECK(nthreads > 0 && nthreads <= kNtMax &&
                  (nthreads & (nthreads - 1)) == 0,
              "fp8_emu_block: nthreads must be a power of two <= ", kNtMax);
  TORCH_CHECK(block_k > 0 && block_k % 4 == 0 && K % block_k == 0,
              "fp8_emu_block: block_k must divide K and be a multiple of 4");

  const c10::cuda::CUDAGuard guard(a.device());
  auto out = at::empty({M, N}, a_s.options().dtype(out_dtype));
  run_block(a, w, a_s, w_s, lut, bias, out, /*gemv=*/true, (int)nthreads, 0,
            block_n, block_k);
  return out;
}

torch::Tensor fp8_emu_gemm_block(torch::Tensor a, torch::Tensor w,
                                 torch::Tensor a_s, torch::Tensor w_s,
                                 torch::Tensor lut,
                                 std::optional<torch::Tensor> bias,
                                 at::ScalarType out_dtype, int64_t cfg,
                                 int64_t block_n, int64_t block_k) {
  const int64_t M = a.size(0), K = a.size(1), N = w.size(0);
  TORCH_CHECK(K % 4 == 0 && K >= 4,
              "fp8_emu_block: K must be a positive multiple of 4, got ", K);
  TORCH_CHECK(M >= 1, "fp8_emu_block: M must be >= 1");
  TORCH_CHECK(cfg >= 0 && cfg <= 40, "fp8_emu_block: cfg must be 0..40, got ",
              cfg);

  const c10::cuda::CUDAGuard guard(a.device());
  auto out = at::empty({M, N}, a_s.options().dtype(out_dtype));
  int c = (int)cfg;
  // cfg 0 auto rule, from the perf1 s1 sweep (K=5120, M=1..129):
  //   * cfg 5, 32x64 TM2TN4 at 256 threads, is the best tile for every M
  //     measured up to N=5120 (e.g. N=1280: 245 us vs 343 us for the old
  //     cfg 1 at M=1, and 248 us vs 418 us at M=16);
  //   * cfg 10, the same tile with a 4-blocks/CU VGPR cap, wins once the grid
  //     is wide (N=8704: 414 vs 532 us at M=1, 1541 vs 1976 us at M=129) or
  //     once deep M makes the narrow-tile grid tall (N=4352: M=64 462 vs
  //     592 us, M=129 828 vs 1014 us).
  // Crossovers are interpolated between the measured 5120 and 8704 points and
  // the measured 3584 all-M point; an explicit cfg (1..10) still overrides.
  if (c == 0) {
    if (M <= kMsMax && (block_k % kFBk) == 0) {
      // Decode/spec-verify batch.  Route it through the same policy the gemv
      // entry uses so `VLLM_DCU_FP8_DECODE_CFG` (and its default = the rule
      // below) is honoured on this entry too: previously the knob was only
      // consulted by `fp8_emu_gemv_block`, so reaching the exact-M tiles from
      // the production `M > VLLM_DCU_FP8_GEMV_MAX_M` dispatch required setting
      // `VLLM_DCU_FP8_GEMV_MAX_M=8` as well, which also re-routed M <= 4
      // (tools/dcu/s20_kernel_NOTES.md S20 6.4).
      c = decode_gemm_cfg(N);
    } else if (N >= 6144) {
      c = 10;
    } else if (M > 32 && N >= 4096) {
      c = 10;
    } else {
      c = 5;
    }
  }
  run_block(a, w, a_s, w_s, lut, bias, out, /*gemv=*/false, 0, c, block_n,
            block_k);
  return out;
}

torch::Tensor fp8_emu_gemv_block_meta(torch::Tensor a, torch::Tensor w,
                                      torch::Tensor a_s, torch::Tensor w_s,
                                      torch::Tensor lut,
                                      std::optional<torch::Tensor> bias,
                                      at::ScalarType out_dtype, int64_t nthreads,
                                      int64_t block_n, int64_t block_k) {
  return a.new_empty({a.size(0), w.size(0)}, a_s.options().dtype(out_dtype));
}

torch::Tensor fp8_emu_gemm_block_meta(torch::Tensor a, torch::Tensor w,
                                      torch::Tensor a_s, torch::Tensor w_s,
                                      torch::Tensor lut,
                                      std::optional<torch::Tensor> bias,
                                      at::ScalarType out_dtype, int64_t cfg,
                                      int64_t block_n, int64_t block_k) {
  return a.new_empty({a.size(0), w.size(0)}, a_s.options().dtype(out_dtype));
}

// ---------------------------------------------------------------------------
// S23 / O5 host side: fused activation quant + block-scaled GEMM
// ---------------------------------------------------------------------------
namespace {

// `cfg` for the fused entry.  Python passes `_prefill_cfg(M, N)`, which is 0
// unless VLLM_DCU_FP8_PREFILL_CFG is set; 0 on the decode range resolves to
// VLLM_DCU_FP8_DECODE_CFG so the fallback routes track the unfused dispatch.
// The primary fused tile has a fixed geometry (BM=8, BN=64) and ignores this
// value -- only the stream/GEMM fallbacks consult it, so
// VLLM_DCU_FP8_DECODE_CFG does not retile the fused hot path.
int fused_decode_cfg(int M, int N, int64_t cfg) {
  if (cfg != 0) return (int)cfg;
  return (M <= kMsMax) ? decode_gemm_cfg(N) : 0;
}

// Standalone quantizer launch (stream fallback and the prefill range).
template <typename TA>
void launch_quant_a(const TA* a, uint8_t* xq, float* xs, int64_t ngroups,
                    float eps, hipStream_t stream) {
  const int threads = 128;  // 4 warps = 4 groups per block
  const unsigned blocks = (unsigned)((ngroups + 3) / 4);
  fp8_emu_quant_a_kernel<TA><<<blocks, threads, 0, stream>>>(a, xq, xs, ngroups,
                                                             eps);
}

template <typename T, typename TA>
void dispatch_block_quant(int M, int N, int K, int64_t w_stride,
                          const torch::Tensor& w_s_t, int nkb, int block_n,
                          int block_k, int cfg, int s_mode, int s_rows,
                          int s_nt, float eps, const TA* A, const uint8_t* W,
                          const float* w_s, const __half* lut, const T* bias,
                          T* C, torch::Tensor* xq_w, torch::Tensor* xs_w,
                          hipStream_t stream) {
  const bool use_stream = s_mode != 0;
  // The fused tile keeps the per-group scales in static LDS
  // (`sA[BM][kQMaxNkb]`), so it only applies when K/block_k fits that budget;
  // larger K (or a stream route, or block_k != 128) takes the standalone
  // quantizer route below.
  const bool fused_tile = !use_stream && M <= kMsMax && block_k == kQGroup &&
                          nkb <= kQMaxNkb;

  if (fused_tile) {
    // Single launch; `out` is the only allocation unless `_quant_x` asked for
    // the witness, which the kernel itself emits from grid column 0 -- the
    // exact quantization the GEMM consumed.
    at::Tensor xq, xs;
    uint8_t* xq_p = nullptr;
    float* xs_p = nullptr;
    if (xq_w != nullptr) {
      *xq_w = at::empty({(int64_t)M, (int64_t)K},
                        w_s_t.options().dtype(at::kByte));
      *xs_w = at::empty({(int64_t)M, nkb}, w_s_t.options());
      xq_p = reinterpret_cast<uint8_t*>(xq_w->data_ptr());
      xs_p = reinterpret_cast<float*>(xs_w->data_ptr());
    }
    constexpr int BM = 8, BN = 64, TM = 1, TN = 2;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block((BM / TM) * (BN / TN));
    fp8_emu_gemm_block_q_kernel<BM, BN, TM, TN, T, TA>
        <<<grid, block, 0, stream>>>(A, W, w_stride, w_s, nkb, block_n, lut,
                                     bias, C, M, N, K, eps, xq_p, xs_p);
    return;
  }

  // Fallback (stream decode, prefill range, or K/block_k > kQMaxNkb): one
  // quantizer launch into freshly allocated buffers (never reused across
  // layers, which would alias under CUDA graphs / concurrent requests), then
  // the existing unfused kernels.
  at::Tensor xq, xs;
  uint8_t* xq_p = nullptr;
  float* xs_p = nullptr;
  const int64_t ngroups = (int64_t)M * nkb;
  if (xq_w != nullptr) {
    *xq_w = at::empty({(int64_t)M, (int64_t)K},
                      w_s_t.options().dtype(at::kByte));
    *xs_w = at::empty({(int64_t)M, nkb}, w_s_t.options());
    xq_p = reinterpret_cast<uint8_t*>(xq_w->data_ptr());
    xs_p = reinterpret_cast<float*>(xs_w->data_ptr());
  } else {
    xq = at::empty({(int64_t)M, (int64_t)K}, w_s_t.options().dtype(at::kByte));
    xs = at::empty({(int64_t)M, nkb}, w_s_t.options());
    xq_p = reinterpret_cast<uint8_t*>(xq.data_ptr());
    xs_p = reinterpret_cast<float*>(xs.data_ptr());
  }
  launch_quant_a<TA>(A, xq_p, xs_p, ngroups, eps, stream);

  const uint8_t* aq = xq_p;
  const float* aq_s = xs_p;
  if (use_stream) {
    if (s_mode == 2) {
      dispatch_gemv_stream2<T>(M, N, K, w_stride, s_rows, s_nt, nkb, block_n,
                               (int)(block_k / kStreamChunk), false, aq, W,
                               aq_s, w_s, lut, bias, C, stream);
    } else {
      dispatch_gemv_stream<T>(M, N, K, w_stride, s_rows, s_nt, nkb, block_n,
                              (int)(block_k / kStreamChunk), false, aq, W,
                              aq_s, w_s, lut, bias, C, stream);
    }
    return;
  }
  // Prefill range (M > kMsMax) or GEMM fallback on the decode range (large
  // K/block_k): quant launch + the existing tiled dispatch.
  dispatch_gemm_block_cfg<T>(cfg, M, N, K, w_stride, nkb, (int)(block_k / kFBk),
                             block_n, aq, W, aq_s, w_s, lut, bias, C, stream);
}

void run_block_quant(const torch::Tensor& a, const torch::Tensor& w,
                     const torch::Tensor& w_s, const torch::Tensor& lut,
                     const std::optional<torch::Tensor>& bias,
                     torch::Tensor& out, int64_t cfg,
                     int64_t block_n, int64_t block_k, double eps,
                     torch::Tensor* xq_w, torch::Tensor* xs_w) {
  TORCH_CHECK(a.is_cuda() && w.is_cuda() && w_s.is_cuda() && lut.is_cuda(),
              "fp8_emu_block_quant: all tensors must be on the GPU");
  TORCH_CHECK(a.element_size() == 2,
              "fp8_emu_block_quant: `a` is the *unquantized* activation and must "
              "be 16-bit (fp16/bf16), got element_size ", a.element_size());
  TORCH_CHECK(w.element_size() == 1 && w.scalar_type() == at::kFloat8_e4m3fn,
              "fp8_emu_block_quant: `w` must be fp8 e4m3");
  TORCH_CHECK(a.is_contiguous() && w.stride(1) == 1 && w.stride(0) > 0 &&
                  lut.is_contiguous() && w_s.is_contiguous(),
              "fp8_emu_block_quant: a/lut/w_s contiguous, w K-contiguous");
  TORCH_CHECK(lut.scalar_type() == at::kHalf && lut.numel() == 256,
              "fp8_emu_block_quant: lut must be a 256-element fp16 tensor");
  TORCH_CHECK(w_s.scalar_type() == at::kFloat && w_s.dim() == 2,
              "fp8_emu_block_quant: w_s must be a 2-D fp32 block-scale matrix");
  const int M = (int)a.size(0), N = (int)w.size(0), K = (int)a.size(1);
  TORCH_CHECK((int)w.size(1) == K, "fp8_emu_block_quant: K mismatch");
  TORCH_CHECK(block_k > 0 && K % block_k == 0,
              "fp8_emu_block_quant: block_k must divide K");
  const int nkb = (int)(K / block_k);
  const int64_t w_stride = w.stride(0);
  TORCH_CHECK((w_stride % 4) == 0,
              "fp8_emu_block_quant: w row stride must be a multiple of 4");
  TORCH_CHECK(w_s.size(0) == (N + block_n - 1) / block_n &&
                  w_s.size(1) == nkb,
              "fp8_emu_block_quant: w_s must be [cdiv(N,block_n), cdiv(K,block_k)]");
  // The whole fused entry quantizes in fixed 128-element groups, so a
  // different block_k cannot be served correctly (mismatched scale layout or,
  // for block_k > 128, an out-of-bounds scale write).
  TORCH_CHECK(block_k == kQGroup,
              "fp8_emu_block_quant: block_k must be ", kQGroup, ", got ",
              block_k);
  const void* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() &&
                    bias->numel() == w.size(0) &&
                    bias->scalar_type() == out.scalar_type(),
                "fp8_emu_block_quant: bad bias");
    bias_ptr = bias->data_ptr();
  }

  const __half* lut_p = reinterpret_cast<const __half*>(lut.data_ptr());
  const uint8_t* W = reinterpret_cast<const uint8_t*>(w.data_ptr());
  const float* w_s_p = reinterpret_cast<const float*>(w_s.data_ptr());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const int c = fused_decode_cfg(M, N, cfg);

  int s_rows = 0, s_nt = 0;
  const int s_mode =
      stream_block_select(M, N, K, w_stride, nkb, (int)block_n, (int)block_k,
                          stream_block_mode(), s_rows, s_nt);

#define VLLM_FP8_EMU_BLOCK_QUANT(T, AT)                                      \
  do {                                                                       \
    if (a.scalar_type() == at::kHalf) {                                       \
      dispatch_block_quant<T, at::Half>(                                     \
          M, N, K, w_stride, w_s, nkb, (int)block_n, (int)block_k, c,         \
          s_mode, s_rows, s_nt, (float)eps,                                  \
          reinterpret_cast<const at::Half*>(a.data_ptr()), W, w_s_p, lut_p,   \
          reinterpret_cast<const AT*>(bias_ptr),                             \
          reinterpret_cast<AT*>(out.data_ptr()), xq_w, xs_w, stream);        \
    } else {                                                                 \
      dispatch_block_quant<T, at::BFloat16>(                                 \
          M, N, K, w_stride, w_s, nkb, (int)block_n, (int)block_k, c,         \
          s_mode, s_rows, s_nt, (float)eps,                                  \
          reinterpret_cast<const at::BFloat16*>(a.data_ptr()), W, w_s_p,      \
          lut_p, reinterpret_cast<const AT*>(bias_ptr),                      \
          reinterpret_cast<AT*>(out.data_ptr()), xq_w, xs_w, stream);        \
    }                                                                        \
  } while (0)
  switch (out.scalar_type()) {
    case at::kBFloat16:
      VLLM_FP8_EMU_BLOCK_QUANT(at::BFloat16, at::BFloat16);
      break;
    case at::kHalf:
      VLLM_FP8_EMU_BLOCK_QUANT(at::Half, at::Half);
      break;
    default:
      VLLM_FP8_EMU_BLOCK_QUANT(float, float);
      break;
  }
#undef VLLM_FP8_EMU_BLOCK_QUANT
}

}  // namespace

torch::Tensor fp8_emu_gemm_block_quant(
    torch::Tensor a, torch::Tensor w, torch::Tensor w_s, torch::Tensor lut,
    std::optional<torch::Tensor> bias, at::ScalarType out_dtype,
    int64_t /*nthreads*/, int64_t cfg, int64_t block_n, int64_t block_k,
    double eps) {
  const c10::cuda::CUDAGuard guard(a.device());
  auto out = at::empty({a.size(0), w.size(0)}, w_s.options().dtype(out_dtype));
  run_block_quant(a, w, w_s, lut, bias, out, cfg, block_n, block_k, eps,
                  nullptr, nullptr);
  return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fp8_emu_gemm_block_quant_x(torch::Tensor a, torch::Tensor w, torch::Tensor w_s,
                           torch::Tensor lut, std::optional<torch::Tensor> bias,
                           at::ScalarType out_dtype, int64_t /*nthreads*/,
                           int64_t cfg, int64_t block_n, int64_t block_k,
                           double eps) {
  const c10::cuda::CUDAGuard guard(a.device());
  auto out = at::empty({a.size(0), w.size(0)}, w_s.options().dtype(out_dtype));
  torch::Tensor xq, xs;
  run_block_quant(a, w, w_s, lut, bias, out, cfg, block_n, block_k, eps,
                  &xq, &xs);
  return std::make_tuple(out, xq, xs);
}

torch::Tensor fp8_emu_gemm_block_quant_meta(
    torch::Tensor a, torch::Tensor w, torch::Tensor w_s, torch::Tensor lut,
    std::optional<torch::Tensor> bias, at::ScalarType out_dtype,
    int64_t /*nthreads*/, int64_t cfg, int64_t block_n, int64_t block_k,
    double eps) {
  return a.new_empty({a.size(0), w.size(0)}, w_s.options().dtype(out_dtype));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fp8_emu_gemm_block_quant_x_meta(torch::Tensor a, torch::Tensor w,
                                torch::Tensor w_s, torch::Tensor lut,
                                std::optional<torch::Tensor> bias,
                                at::ScalarType out_dtype, int64_t /*nthreads*/,
                                int64_t cfg, int64_t block_n, int64_t block_k,
                                double eps) {
  return std::make_tuple(
      a.new_empty({a.size(0), w.size(0)}, w_s.options().dtype(out_dtype)),
      a.new_empty({a.size(0), a.size(1)}, w_s.options().dtype(at::kByte)),
      a.new_empty({a.size(0), a.size(1) / block_k}, w_s.options()));
}

// ---------------------------------------------------------------------------
// W8A16 + prefill-dequant entry points
// ---------------------------------------------------------------------------
torch::Tensor fp8_emu_gemm_block_w8a16(torch::Tensor a, torch::Tensor w,
                                       torch::Tensor w_s, torch::Tensor lut,
                                       std::optional<torch::Tensor> bias,
                                       at::ScalarType out_dtype,
                                       int64_t nthreads, int64_t cfg,
                                       int64_t block_n, int64_t block_k) {
  // Check the rank before indexing: a 1-D `a` would otherwise raise an ATen
  // IndexError from a.size(1) and a 3-D `w` would allocate `out` first.
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2,
              "fp8_emu_block_w8a16: a and w must be 2-D, got ", a.dim(),
              " and ", w.dim());
  const int64_t M = a.size(0), K = a.size(1);
  TORCH_CHECK(K % 8 == 0 && K >= 8,
              "fp8_emu_block_w8a16: K must be a positive multiple of 8, got ", K);
  TORCH_CHECK(M >= 1, "fp8_emu_block_w8a16: M must be >= 1");
  TORCH_CHECK(cfg >= 0 && cfg <= 40,
              "fp8_emu_block_w8a16: cfg must be 0..40, got ", cfg);
  TORCH_CHECK(nthreads > 0 && nthreads <= kNtMax &&
                  (nthreads & (nthreads - 1)) == 0,
              "fp8_emu_block_w8a16: nthreads must be a power of two <= ", kNtMax);

  const c10::cuda::CUDAGuard guard(a.device());
  auto out = at::empty({M, w.size(0)}, a.options().dtype(out_dtype));
  run_block_w8a16(a, w, w_s, lut, bias, out,
                  /*gemv=*/M <= w8a16_gemv_max_m(), (int)nthreads, (int)cfg,
                  block_n, block_k);
  return out;
}

torch::Tensor fp8_emu_gemm_block_w8a16_meta(
    torch::Tensor a, torch::Tensor w, torch::Tensor w_s, torch::Tensor lut,
    std::optional<torch::Tensor> bias, at::ScalarType out_dtype,
    int64_t /*nthreads*/, int64_t /*cfg*/, int64_t /*block_n*/,
    int64_t /*block_k*/) {
  return a.new_empty({a.size(0), w.size(0)}, a.options().dtype(out_dtype));
}

void fp8_dequant_to_fp16_out(torch::Tensor out, torch::Tensor wq,
                             torch::Tensor ws, torch::Tensor lut,
                             int64_t block_n, int64_t block_k) {
  const c10::cuda::CUDAGuard guard(out.device());
  vllm::gfx906_emu::dequant_fp8_block_to_fp16(out, wq, ws, lut, block_n,
                                             block_k);
}
