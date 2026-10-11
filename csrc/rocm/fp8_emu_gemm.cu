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

#include <cstdint>

#include "rocm/fp8_emu_gemm.h"
#include "rocm/gfx906_emu_common.cuh"

namespace {

using vllm::gfx906_emu::dot2;
using vllm::gfx906_emu::fp8_lut_stage;
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
  static constexpr int A4 = BM * kFKw4 / NT;  // fp8 dwords loaded per thread
  static constexpr int B4 = BN * kFKw4 / NT;
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
      v = wp[(int64_t)r * K4 + jd];
    dst[C::A4 + u] = v;
  }
}

// Registers -> LDS (slice S), widening fp8 to fp16 pairs on the way.
template <int BM, int BN, int TM, int TN, int S>
__device__ __forceinline__ void fp8_gemm_store(int tid,
                                               const uint32_t* __restrict__ src,
                                               const __half* __restrict__ lut,
                                               uint32_t (&As)[2][BM * kFKp],
                                               uint32_t (&Bs)[2][BN * kFKp]) {
  using C = FConf<BM, BN, TM, TN>;
#pragma unroll
  for (int u = 0; u < C::A4; ++u) {
    const int idx = tid + u * C::NT;
    const int r = idx >> kFLogKw4;
    const int jd = idx & (kFKw4 - 1);
    uint32_t lo, hi;
    fp8x4_to_f16x2x2(lut, src[u], lo, hi);
    As[S][r * kFKp + 2 * jd] = lo;
    As[S][r * kFKp + 2 * jd + 1] = hi;
  }
#pragma unroll
  for (int u = 0; u < C::B4; ++u) {
    const int idx = tid + u * C::NT;
    const int r = idx >> kFLogKw4;
    const int jd = idx & (kFKw4 - 1);
    uint32_t lo, hi;
    fp8x4_to_f16x2x2(lut, src[C::A4 + u], lo, hi);
    Bs[S][r * kFKp + 2 * jd] = lo;
    Bs[S][r * kFKp + 2 * jd + 1] = hi;
  }
}

// LDS slice S -> TM x TN fp32 accumulators (one dot2 = 2 fp16 MACs).
template <int BM, int BN, int TM, int TN, int S>
__device__ __forceinline__ void fp8_gemm_compute(
    int tx, int ty, const uint32_t (&As)[2][BM * kFKp],
    const uint32_t (&Bs)[2][BN * kFKp], float (&acc)[TM][TN]) {
  using C = FConf<BM, BN, TM, TN>;
  const uint32_t* __restrict__ as = As[S];
  const uint32_t* __restrict__ bs = Bs[S];
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
  static_assert((BM * kFKw4) % NT == 0 && (BN * kFKw4) % NT == 0,
                "tile/thread mismatch");

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
    fp8_gemm_store<BM, BN, TM, TN, 0>(tid, f0, sl, As, Bs);
    __syncthreads();
    fp8_gemm_load<BM, BN, TM, TN>(A, W, m0, n0, M, N, K, K4,
                                  (kb + 1) * kFBk, tid, f1);
    fp8_gemm_compute<BM, BN, TM, TN, 0>(tx, ty, As, Bs, acc);

    fp8_gemm_store<BM, BN, TM, TN, 1>(tid, f1, sl, As, Bs);
    __syncthreads();
    fp8_gemm_load<BM, BN, TM, TN>(A, W, m0, n0, M, N, K, K4,
                                  (kb + 2) * kFBk, tid, f0);
    fp8_gemm_compute<BM, BN, TM, TN, 1>(tx, ty, As, Bs, acc);
  }
  if (kb < NK) {  // odd NK: the leftover slice is number 0 (kb is even here)
    fp8_gemm_store<BM, BN, TM, TN, 0>(tid, f0, sl, As, Bs);
    __syncthreads();
    fp8_gemm_compute<BM, BN, TM, TN, 0>(tx, ty, As, Bs, acc);
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
template <int MS, typename T, bool PRECONV>
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
      s[m] = (m < M) ? a_s[m * nkb + kb] * wsv : 0.0f;
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
                       int dpb, int block_n, const uint8_t* a,
                       const uint8_t* w, const uint32_t* a_h2,
                       const float* a_s, const float* w_s, const __half* lut,
                       const T* bias, T* out, hipStream_t stream) {
  const size_t smem = [&]() {
    if (M <= 1) return (size_t)1 * nt * sizeof(float);
    if (M <= 4) return (size_t)4 * nt * sizeof(float);
    return (size_t)kMsMax * nt * sizeof(float);
  }();
  dim3 grid(N), block(nt);
#define VLLM_FP8_EMU_GEMV_BLOCK_CASE(MS)                                    \
  do {                                                                      \
    if (a_h2 != nullptr) {                                                  \
      fp8_emu_gemv_block_kernel<MS, T, true><<<grid, block, smem, stream>>>( \
          a, w, a_h2, a_s, w_s, lut, bias, out, M, N, K4, w_stride, nkb,    \
          dpb, block_n);                                                    \
    } else {                                                                \
      fp8_emu_gemv_block_kernel<MS, T, false><<<grid, block, smem, stream>>>(\
          a, w, nullptr, a_s, w_s, lut, bias, out, M, N, K4, w_stride, nkb, \
          dpb, block_n);                                                    \
    }                                                                       \
    return;                                                                 \
  } while (0)
  if (M <= 1) VLLM_FP8_EMU_GEMV_BLOCK_CASE(1);
  if (M <= 4) VLLM_FP8_EMU_GEMV_BLOCK_CASE(4);
  VLLM_FP8_EMU_GEMV_BLOCK_CASE(kMsMax);
#undef VLLM_FP8_EMU_GEMV_BLOCK_CASE
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

template <int BM, int BN, int TM, int TN, typename T>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
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
  static_assert((BM * kFKw4) % NT == 0 && (BN * kFKw4) % NT == 0,
                "tile/thread mismatch");

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
    fp8_gemm_store<BM, BN, TM, TN, 0>(tid, f0, sl, As, Bs);
    __syncthreads();
    fp8_gemm_load_block<BM, BN, TM, TN>(A, W, w_stride, m0, n0, M, N, K, K4,
                                        (c + 1) * kFBk, tid, f1);
    fp8_gemm_compute<BM, BN, TM, TN, 0>(tx, ty, As, Bs, blk);
    if (((c + 1) % cpb) == 0)
      fp8_gemm_fold_block_scale<BM, BN, TM, TN>(tx, ty, m0, n0, M, N, c / cpb,
                                                nkb, block_n, a_s, w_s, blk,
                                                acc);

    fp8_gemm_store<BM, BN, TM, TN, 1>(tid, f1, sl, As, Bs);
    __syncthreads();
    fp8_gemm_load_block<BM, BN, TM, TN>(A, W, w_stride, m0, n0, M, N, K, K4,
                                        (c + 2) * kFBk, tid, f0);
    fp8_gemm_compute<BM, BN, TM, TN, 1>(tx, ty, As, Bs, blk);
    if (((c + 2) % cpb) == 0)
      fp8_gemm_fold_block_scale<BM, BN, TM, TN>(tx, ty, m0, n0, M, N,
                                                (c + 1) / cpb, nkb, block_n,
                                                a_s, w_s, blk, acc);
  }
  if (c < NK) {  // odd NK: the leftover slice is number 0 (c is even here)
    fp8_gemm_store<BM, BN, TM, TN, 0>(tid, f0, sl, As, Bs);
    __syncthreads();
    fp8_gemm_compute<BM, BN, TM, TN, 0>(tx, ty, As, Bs, blk);
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

template <int BM, int BN, int TM, int TN, typename T>
void launch_gemm_block(int M, int N, int K, int64_t w_stride, int nkb, int cpb,
                       int block_n, const uint8_t* A, const uint8_t* W,
                       const float* a_s, const float* w_s, const __half* lut,
                       const T* bias, T* C, hipStream_t stream) {
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block((BM / TM) * (BN / TN));
  fp8_emu_gemm_block_kernel<BM, BN, TM, TN, T><<<grid, block, 0, stream>>>(
      A, W, w_stride, a_s, w_s, nkb, cpb, block_n, lut, bias, C, M, N, K);
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
                         int block_n, const uint8_t* A, const uint8_t* W,
                         const uint32_t* a_h2, const float* a_s,
                         const float* w_s, const __half* lut, const T* bias,
                         T* C, hipStream_t stream) {
  launch_gemv_block<T>(M, N, K / 4, nt, w_stride, nkb, (int)(K / 4) / nkb,
                       block_n, A, W, a_h2, a_s, w_s, lut, bias, C, stream);
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
  switch (cfg) {
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(1, 32, 64, 4, 4);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(2, 64, 64, 4, 4);
    VLLM_FP8_EMU_GEMM_BLOCK_CASE(3, 64, 128, 4, 8);
    case 4:
    default:
      launch_gemm_block<128, 128, 8, 8, T>(M, N, K, w_stride, nkb, cpb,
                                           block_n, A, W, a_s, w_s, lut, bias,
                                           C, stream);
      break;
  }
#undef VLLM_FP8_EMU_GEMM_BLOCK_CASE
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

  const int cpb = (int)(block_k / kFBk);
  if (!gemv) {
    TORCH_CHECK(block_k % kFBk == 0,
                "fp8_emu_block: block_k must be a multiple of ", kFBk,
                " for the GEMM path, got ", block_k);
  }

#define VLLM_FP8_EMU_BLOCK_DISPATCH(T, AT)                                   \
  do {                                                                       \
    if (gemv) {                                                              \
      dispatch_gemv_block<T>(M, N, K, nthreads, w_stride, nkb,               \
                             (int)block_n, A, W, a_h2, a_s_p, w_s_p, lut_p,  \
                             reinterpret_cast<const AT*>(bias_ptr),          \
                             reinterpret_cast<AT*>(out.data_ptr()), stream); \
    } else {                                                                 \
      dispatch_gemm_block_cfg<T>(cfg, M, N, K, w_stride, nkb, cpb,           \
                                 (int)block_n, A, W, a_s_p, w_s_p, lut_p,    \
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
  TORCH_CHECK(cfg >= 0 && cfg <= 4, "fp8_emu_block: cfg must be 0..4, got ",
              cfg);

  const c10::cuda::CUDAGuard guard(a.device());
  auto out = at::empty({M, N}, a_s.options().dtype(out_dtype));
  int c = (int)cfg;
  if (c == 0) c = (M <= 32) ? 1 : 2;  // same tuned default as the per-tensor path
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
