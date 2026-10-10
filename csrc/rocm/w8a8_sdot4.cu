// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W8A8 INT8 GEMM/GEMV for gfx906 (Vega20) built on the DL-Ops instruction
// `V_DOT4_I32_I8` (clang builtin `__builtin_amdgcn_sdot4`, clamp=false).
//
// Provenance:
//   Ported on 2026-10-08 from the author's earlier out-of-tree project
//   `/home/lyh/server/int8_sdot4/w8a8_sdot4_kernel.hip` (484 lines, same
//   author, Apache-2.0), which validated this shape in production on the
//   same hardware.  The device code below is algorithmically identical; the
//   changes are limited to:
//     * registration as `_rocm_C` ops instead of a standalone
//       `PYBIND11_MODULE` (`w8a8_gemv` -> `w8a8_sdot4_gemv`,
//       `w8a8_gemm` -> `w8a8_sdot4_gemm`),
//     * dropped the CPU int32 oracle (the Python-side validation uses
//       `torch.matmul(xq.int(), w.int().t())` as the exact oracle instead),
//     * added `*_meta` implementations so the ops can be traced by
//       `torch.compile` / fake tensor propagation.
//
// Why this kernel exists: gfx906 has no MFMA/WMMA, but it does have the
// int8 dot product (`V_DOT4_I32_I8`, 4 int8 MACs per instruction).  The
// Triton `tl.dot` path cannot reach it for W8A8 (int8 `tl.dot` still lowers
// to an FMA expansion in the gfx906 Triton backend), which is why the
// Triton-based `scaled_mm` fallback is ~10x slower here.
//
// Layout assumptions (native compressed-tensors W8A8):
//   * weights stay `[N, K]` int8, contiguous, per-channel `w_s` float32
//     (`[N]`).  They are never transposed and never dequantized, so the
//     resident memory footprint does not change.
//   * activations are per-token dynamic symmetric int8, `xq [M, K]` with
//     `x_s [M]`.
//   * `K % 4 == 0` (int32 packing granularity); the Python layer falls back
//     to Triton when that does not hold.
//   * epilogue order matches `triton_scaled_mm`: `(acc * x_s) * w_s`, then
//     cast to the output dtype, then `+ bias`.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "rocm/w8a8_sdot4.h"

namespace {

constexpr int kMsMax = 8;      // GEMV covers M <= 8 (decode)
constexpr int kNtMax = 512;    // max GEMV threads per block (power of two)

// ---------------------------------------------------------------------------
// decode path: M <= 8
// ---------------------------------------------------------------------------
// One block per output column n (i.e. one row of W).  64 threads split K/4
// int32 words; `sdot4` accumulates in int32 (exact), then an LDS tree
// reduction folds the per-thread partials.  W is read once per row and the
// traffic does not depend on M.

template <int MS, typename T>
__global__ void w8a8_sdot4_gemv_kernel(const int8_t* __restrict__ xq,  // [M, K]
                                       const int8_t* __restrict__ w,   // [N, K]
                                       const float* __restrict__ x_s,  // [M]
                                       const float* __restrict__ w_s,  // [N]
                                       const T* __restrict__ bias,     // [N] | null
                                       T* __restrict__ out,            // [M, N]
                                       const int M, const int N, const int K4) {
  extern __shared__ int red[];  // [MS][nt]
  const int n = blockIdx.x;
  const int t = threadIdx.x;
  const int nt = blockDim.x;

  const int32_t* __restrict__ xp = reinterpret_cast<const int32_t*>(xq);
  const int32_t* __restrict__ wp =
      reinterpret_cast<const int32_t*>(w) + (int64_t)n * K4;

  int acc[MS];
#pragma unroll
  for (int m = 0; m < MS; ++m) acc[m] = 0;

  // Consecutive int32 words across threads -> coalesced loads.
  for (int i = t; i < K4; i += nt) {
    const int32_t ww = wp[i];
#pragma unroll
    for (int m = 0; m < MS; ++m) {
      if (m < M) acc[m] = __builtin_amdgcn_sdot4(xp[m * K4 + i], ww, acc[m], false);
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
    const float v = static_cast<float>(red[t * nt]) * x_s[t] * w_s[n];
    // Same order as triton_scaled_mm (and as the GEMM epilogue below):
    // (acc * x_s) * w_s, cast to the output dtype, then add bias in that
    // dtype.  The out-of-tree original folded the bias in fp32 before the
    // cast here, which made the GEMV path disagree with the GEMM path and
    // with the Triton fallback by 1 ulp of fp16.
    T o = static_cast<T>(v);
    if (bias != nullptr) o += static_cast<T>(bias[n]);
    out[t * (int64_t)N + n] = o;
  }
}

template <typename T>
void launch_gemv(int M, int N, int K4, int nt, const int8_t* xq, const int8_t* w,
                 const float* x_s, const float* w_s, const T* bias, T* out,
                 hipStream_t stream) {
  const size_t smem = [&]() {
    if (M <= 1) return (size_t)1 * nt * sizeof(int);
    if (M <= 4) return (size_t)4 * nt * sizeof(int);
    return (size_t)kMsMax * nt * sizeof(int);
  }();
  dim3 grid(N), block(nt);
  if (M <= 1)
    w8a8_sdot4_gemv_kernel<1, T>
        <<<grid, block, smem, stream>>>(xq, w, x_s, w_s, bias, out, M, N, K4);
  else if (M <= 4)
    w8a8_sdot4_gemv_kernel<4, T>
        <<<grid, block, smem, stream>>>(xq, w, x_s, w_s, bias, out, M, N, K4);
  else
    w8a8_sdot4_gemv_kernel<kMsMax, T>
        <<<grid, block, smem, stream>>>(xq, w, x_s, w_s, bias, out, M, N, K4);
}

// ---------------------------------------------------------------------------
// prefill path: M > 8 (the 32x64 / 64x64 tiles below)
// ---------------------------------------------------------------------------
// Each block computes a BM x BN tile of C[m, n] = sum_k A[m, k] * W[n, k]
// with both operands int8 and K-contiguous ("NN" layout, i.e. the native
// [N, K] compressed-tensors weight).
//
//   * K is walked in BK = 32 chunks (8 int32).  A/W tiles go
//     global -> registers -> LDS with double buffering, so the global load
//     of chunk s^1 overlaps the sdot4 work on chunk s.
//   * Each thread owns TM x TN accumulators and issues TM*TN `sdot4`s per
//     K word.  LDS traffic / compute = 4*TM*TN/(TM+TN) MAC per word; >= 10
//     keeps LDS off the critical path.
//   * The LDS row stride is KW + 1 = 9 words (not 8) so that the 16 reader
//     threads land on 16 distinct banks (9*t mod 32 is injective for
//     t = 0..15) -> no read bank conflicts.  The write side is at most
//     3-way, but writes happen 1/20th as often as reads.
//   * Out-of-range M/N/K are handled by zero-filling at staging time and
//     bounds checks at store time; no separate fallback kernel.

constexpr int kGBk = 32;
constexpr int kGKw = kGBk / 4;  // 8 int32 words per K chunk
constexpr int kGKp = kGKw + 1;  // 9 (padded stride)
constexpr int kGLogKw = 3;

template <int BM, int BN, int TM, int TN>
struct GConf {
  static constexpr int MG = BM / TM;  // threads along M
  static constexpr int NG = BN / TN;  // threads along N
  static constexpr int NT = MG * NG;
  static constexpr int AB = BM * kGKw;
  static constexpr int BB = BN * kGKw;
  static constexpr int AFRAGS = AB / NT;
  static constexpr int BFRAGS = BB / NT;
  static constexpr int FRAGS = AFRAGS + BFRAGS;
};

// Stage one K slice of the A/W tiles from global into registers (zero-fill
// out of range).
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void gemm_load(const int8_t* __restrict__ A,
                                          const int8_t* __restrict__ W, int m0,
                                          int n0, int M, int N, int K, int K4,
                                          int k0, int tid,
                                          int32_t* __restrict__ dst) {
  using C = GConf<BM, BN, TM, TN>;
  const int32_t* __restrict__ ap =
      reinterpret_cast<const int32_t*>(A + (int64_t)m0 * K + k0);
  const int32_t* __restrict__ wp =
      reinterpret_cast<const int32_t*>(W + (int64_t)n0 * K + k0);
  const bool kfull = (k0 + kGBk) <= K;
#pragma unroll
  for (int u = 0; u < C::AFRAGS; ++u) {
    const int idx = tid + u * C::NT;
    const int r = idx >> kGLogKw;
    const int kw = idx & (kGKw - 1);
    int32_t v = 0;
    if (m0 + r < M && (kfull || (k0 + (kw + 1) * 4) <= K)) v = ap[(int64_t)r * K4 + kw];
    dst[u] = v;
  }
#pragma unroll
  for (int u = 0; u < C::BFRAGS; ++u) {
    const int idx = tid + u * C::NT;
    const int r = idx >> kGLogKw;
    const int kw = idx & (kGKw - 1);
    int32_t v = 0;
    if (n0 + r < N && (kfull || (k0 + (kw + 1) * 4) <= K)) v = wp[(int64_t)r * K4 + kw];
    dst[C::AFRAGS + u] = v;
  }
}

// Registers -> LDS (slice S).
template <int BM, int BN, int TM, int TN, int S>
__device__ __forceinline__ void gemm_store(int tid,
                                           const int32_t* __restrict__ src,
                                           int32_t (&As)[2][BM * kGKp],
                                           int32_t (&Bs)[2][BN * kGKp]) {
  using C = GConf<BM, BN, TM, TN>;
#pragma unroll
  for (int u = 0; u < C::AFRAGS; ++u) {
    const int idx = tid + u * C::NT;
    As[S][(idx >> kGLogKw) * kGKp + (idx & (kGKw - 1))] = src[u];
  }
#pragma unroll
  for (int u = 0; u < C::BFRAGS; ++u) {
    const int idx = tid + u * C::NT;
    Bs[S][(idx >> kGLogKw) * kGKp + (idx & (kGKw - 1))] = src[C::AFRAGS + u];
  }
}

// LDS slice S -> TM x TN int32 accumulators (one sdot4 = 4 int8 MACs).
template <int BM, int BN, int TM, int TN, int S>
__device__ __forceinline__ void gemm_compute(int tx, int ty,
                                             const int32_t (&As)[2][BM * kGKp],
                                             const int32_t (&Bs)[2][BN * kGKp],
                                             int (&acc)[TM][TN]) {
  using C = GConf<BM, BN, TM, TN>;
  const int32_t* __restrict__ as = As[S];
  const int32_t* __restrict__ bs = Bs[S];
#pragma unroll
  for (int u = 0; u < kGKw; ++u) {
    int32_t a[TM], b[TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) a[i] = as[(ty + C::MG * i) * kGKp + u];
#pragma unroll
    for (int j = 0; j < TN; ++j) b[j] = bs[(tx + C::NG * j) * kGKp + u];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        acc[i][j] = __builtin_amdgcn_sdot4(a[i], b[j], acc[i][j], false);
      }
    }
  }
}

template <int BM, int BN, int TM, int TN, typename T>
__global__ void __launch_bounds__((BM / TM) * (BN / TN)) w8a8_sdot4_gemm_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ W,
    const float* __restrict__ x_s, const float* __restrict__ w_s,
    const T* __restrict__ bias, T* __restrict__ C, int M, int N, int K) {
  using C_ = GConf<BM, BN, TM, TN>;
  constexpr int NT = C_::NT;
  constexpr int FRAGS = C_::FRAGS;
  static_assert((BM * kGKw) % NT == 0 && (BN * kGKw) % NT == 0,
                "tile/thread mismatch");

  __shared__ int32_t As[2][BM * kGKp];
  __shared__ int32_t Bs[2][BN * kGKp];

  const int tid = threadIdx.x;
  const int tx = tid % C_::NG;  // along N
  const int ty = tid / C_::NG;  // along M
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int K4 = K >> 2;
  const int NK = (K + kGBk - 1) / kGBk;

  int acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0;
  }

  int32_t f0[FRAGS], f1[FRAGS];
  gemm_load<BM, BN, TM, TN>(A, W, m0, n0, M, N, K, K4, 0, tid, f0);

  // Per iteration: store LDS(S) -> syncthreads -> issue the next global
  // load -> compute from LDS(S).  One syncthreads suffices: it also
  // guarantees the slice consumed by the previous compute is free to reuse.
  int kb = 0;
  for (; kb + 1 < NK; kb += 2) {
    gemm_store<BM, BN, TM, TN, 0>(tid, f0, As, Bs);
    __syncthreads();
    // NOTE: k0 is a *byte* offset into the K dimension, not a slice index.
    gemm_load<BM, BN, TM, TN>(A, W, m0, n0, M, N, K, K4, (kb + 1) * kGBk, tid, f1);
    gemm_compute<BM, BN, TM, TN, 0>(tx, ty, As, Bs, acc);

    gemm_store<BM, BN, TM, TN, 1>(tid, f1, As, Bs);
    __syncthreads();
    gemm_load<BM, BN, TM, TN>(A, W, m0, n0, M, N, K, K4, (kb + 2) * kGBk, tid, f0);
    gemm_compute<BM, BN, TM, TN, 1>(tx, ty, As, Bs, acc);
  }
  if (kb < NK) {  // odd NK: the leftover slice is number 0 (kb is even here)
    gemm_store<BM, BN, TM, TN, 0>(tid, f0, As, Bs);
    __syncthreads();
    gemm_compute<BM, BN, TM, TN, 0>(tx, ty, As, Bs, acc);
  }

  // Epilogue: identical order to triton_scaled_mm -- (acc * x_s) * w_s,
  // then cast to the output dtype, then add bias.
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + C_::MG * i;
    if (row >= M) continue;
    const float sa = x_s[row];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + tx + C_::NG * j;
      if (col >= N) continue;
      const float v = static_cast<float>(acc[i][j]) * sa * w_s[col];
      T o = static_cast<T>(v);
      if (bias != nullptr) o += static_cast<T>(bias[col]);
      C[(int64_t)row * N + col] = o;
    }
  }
}

template <int BM, int BN, int TM, int TN, typename T>
void launch_gemm(int M, int N, int K, const int8_t* A, const int8_t* W,
                 const float* x_s, const float* w_s, const T* bias, T* C,
                 hipStream_t stream) {
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block((BM / TM) * (BN / TN));
  w8a8_sdot4_gemm_kernel<BM, BN, TM, TN, T>
      <<<grid, block, 0, stream>>>(A, W, x_s, w_s, bias, C, M, N, K);
}

void check_common(const torch::Tensor& xq, const torch::Tensor& w,
                  const torch::Tensor& x_s, const torch::Tensor& w_s,
                  const std::optional<torch::Tensor>& bias,
                  at::ScalarType out_dtype) {
  TORCH_CHECK(xq.is_cuda() && w.is_cuda() && x_s.is_cuda() && w_s.is_cuda(),
              "w8a8_sdot4: all tensors must be on the GPU");
  TORCH_CHECK(xq.scalar_type() == at::kChar && w.scalar_type() == at::kChar,
              "w8a8_sdot4: xq/w must be int8");
  TORCH_CHECK(x_s.scalar_type() == at::kFloat && w_s.scalar_type() == at::kFloat,
              "w8a8_sdot4: scales must be float32");
  TORCH_CHECK(xq.is_contiguous() && w.is_contiguous() && x_s.is_contiguous() &&
                  w_s.is_contiguous(),
              "w8a8_sdot4: tensors must be contiguous");
  TORCH_CHECK(w.size(1) == xq.size(1), "w8a8_sdot4: K mismatch: xq K=",
              xq.size(1), " w K=", w.size(1));
  TORCH_CHECK(x_s.numel() == xq.size(0), "w8a8_sdot4: x_s must have M elements");
  TORCH_CHECK(w_s.numel() == w.size(0), "w8a8_sdot4: w_s must have N elements");
  TORCH_CHECK(out_dtype == at::kHalf || out_dtype == at::kBFloat16 ||
                  out_dtype == at::kFloat,
              "w8a8_sdot4: unsupported out dtype ", out_dtype);
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() &&
                    bias->numel() == w.size(0),
                "w8a8_sdot4: bias must be a contiguous GPU tensor of N elements");
    TORCH_CHECK(bias->scalar_type() == out_dtype,
                "w8a8_sdot4: bias dtype must match out dtype");
  }
}

}  // namespace

torch::Tensor w8a8_sdot4_gemv(torch::Tensor xq, torch::Tensor w,
                              torch::Tensor x_s, torch::Tensor w_s,
                              std::optional<torch::Tensor> bias,
                              at::ScalarType out_dtype, int64_t nthreads) {
  check_common(xq, w, x_s, w_s, bias, out_dtype);
  const int64_t M = xq.size(0), K = xq.size(1), N = w.size(0);
  TORCH_CHECK(K % 4 == 0, "w8a8_sdot4: K must be a multiple of 4, got ", K);
  TORCH_CHECK(M >= 1 && M <= kMsMax, "w8a8_sdot4: M must be in [1,", kMsMax,
              "] for the GEMV path, got ", M);
  TORCH_CHECK(nthreads > 0 && nthreads <= kNtMax &&
                  (nthreads & (nthreads - 1)) == 0,
              "w8a8_sdot4: nthreads must be a power of two <= ", kNtMax);

  const c10::cuda::CUDAGuard guard(xq.device());
  auto out = at::empty({M, N}, x_s.options().dtype(out_dtype));
  const void* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined()) bias_ptr = bias->data_ptr();

  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const int nt = static_cast<int>(nthreads);
  const int K4 = static_cast<int>(K / 4);

  // Plain (non-template) data_ptr() + reinterpret_cast: the templated
  // const_data_ptr<T>() instantiations are not exported by this libtorch.
  const int8_t* xq_p = reinterpret_cast<const int8_t*>(xq.data_ptr());
  const int8_t* w_p = reinterpret_cast<const int8_t*>(w.data_ptr());
  const float* xs_p = reinterpret_cast<const float*>(x_s.data_ptr());
  const float* ws_p = reinterpret_cast<const float*>(w_s.data_ptr());

  switch (out.scalar_type()) {
    case at::kBFloat16:
      launch_gemv<at::BFloat16>(
          M, N, K4, nt, xq_p, w_p, xs_p, ws_p,
          reinterpret_cast<const at::BFloat16*>(bias_ptr),
          reinterpret_cast<at::BFloat16*>(out.data_ptr()), stream);
      break;
    case at::kHalf:
      launch_gemv<at::Half>(M, N, K4, nt, xq_p, w_p, xs_p, ws_p,
                            reinterpret_cast<const at::Half*>(bias_ptr),
                            reinterpret_cast<at::Half*>(out.data_ptr()), stream);
      break;
    default:
      launch_gemv<float>(M, N, K4, nt, xq_p, w_p, xs_p, ws_p,
                         reinterpret_cast<const float*>(bias_ptr),
                         reinterpret_cast<float*>(out.data_ptr()), stream);
      break;
  }
  return out;
}

torch::Tensor w8a8_sdot4_gemm(torch::Tensor xq, torch::Tensor w,
                              torch::Tensor x_s, torch::Tensor w_s,
                              std::optional<torch::Tensor> bias,
                              at::ScalarType out_dtype, int64_t cfg) {
  check_common(xq, w, x_s, w_s, bias, out_dtype);
  const int64_t M = xq.size(0), K = xq.size(1), N = w.size(0);
  TORCH_CHECK(K % 4 == 0 && K >= 4,
              "w8a8_sdot4: K must be a positive multiple of 4, got ", K);
  TORCH_CHECK(M >= 1, "w8a8_sdot4: M must be >= 1");
  TORCH_CHECK(cfg >= 0 && cfg <= 4, "w8a8_sdot4: cfg must be 0..4, got ", cfg);

  const c10::cuda::CUDAGuard guard(xq.device());
  auto out = at::empty({M, N}, x_s.options().dtype(out_dtype));
  const void* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined()) bias_ptr = bias->data_ptr();

  int c = static_cast<int>(cfg);
  if (c == 0) c = (M <= 32) ? 1 : 2;  // tuned: 32x64 for shallow M, else 64x64

  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const int8_t* A = reinterpret_cast<const int8_t*>(xq.data_ptr());
  const int8_t* W = reinterpret_cast<const int8_t*>(w.data_ptr());
  const float* xs = reinterpret_cast<const float*>(x_s.data_ptr());
  const float* ws = reinterpret_cast<const float*>(w_s.data_ptr());
  const int Mi = static_cast<int>(M), Ni = static_cast<int>(N),
            Ki = static_cast<int>(K);

#define VLLM_SDOT4_GEMM_CASE(CFG, BM, BN, TM, TN, T)                       \
  case CFG:                                                               \
    launch_gemm<BM, BN, TM, TN, T>(                                       \
        Mi, Ni, Ki, A, W, xs, ws, reinterpret_cast<const T*>(bias_ptr),   \
        reinterpret_cast<T*>(out.data_ptr()), stream);                    \
    break

  switch (out.scalar_type()) {
    case at::kBFloat16:
      switch (c) {
        VLLM_SDOT4_GEMM_CASE(1, 32, 64, 4, 4, at::BFloat16);
        VLLM_SDOT4_GEMM_CASE(2, 64, 64, 4, 4, at::BFloat16);
        VLLM_SDOT4_GEMM_CASE(3, 64, 128, 4, 8, at::BFloat16);
        default:
          launch_gemm<128, 128, 8, 8, at::BFloat16>(
              Mi, Ni, Ki, A, W, xs, ws,
              reinterpret_cast<const at::BFloat16*>(bias_ptr),
              reinterpret_cast<at::BFloat16*>(out.data_ptr()), stream);
          break;
      }
      break;
    case at::kHalf:
      switch (c) {
        VLLM_SDOT4_GEMM_CASE(1, 32, 64, 4, 4, at::Half);
        VLLM_SDOT4_GEMM_CASE(2, 64, 64, 4, 4, at::Half);
        VLLM_SDOT4_GEMM_CASE(3, 64, 128, 4, 8, at::Half);
        default:
          launch_gemm<128, 128, 8, 8, at::Half>(
              Mi, Ni, Ki, A, W, xs, ws,
              reinterpret_cast<const at::Half*>(bias_ptr),
              reinterpret_cast<at::Half*>(out.data_ptr()), stream);
          break;
      }
      break;
    default:
      switch (c) {
        VLLM_SDOT4_GEMM_CASE(1, 32, 64, 4, 4, float);
        VLLM_SDOT4_GEMM_CASE(2, 64, 64, 4, 4, float);
        VLLM_SDOT4_GEMM_CASE(3, 64, 128, 4, 8, float);
        default:
          launch_gemm<128, 128, 8, 8, float>(
              Mi, Ni, Ki, A, W, xs, ws,
              reinterpret_cast<const float*>(bias_ptr),
              reinterpret_cast<float*>(out.data_ptr()), stream);
          break;
      }
      break;
  }
#undef VLLM_SDOT4_GEMM_CASE
  return out;
}

// ---------------------------------------------------------------------------
// Meta (fake tensor) implementations, so the ops survive `torch.compile`
// tracing without a graph break.  The shapes are fully determined by the
// arguments, so no computation is needed.
// ---------------------------------------------------------------------------

torch::Tensor w8a8_sdot4_gemv_meta(torch::Tensor xq, torch::Tensor w,
                                   torch::Tensor x_s, torch::Tensor w_s,
                                   std::optional<torch::Tensor> bias,
                                   at::ScalarType out_dtype, int64_t nthreads) {
  return at::empty({xq.size(0), w.size(0)}, xq.options().dtype(out_dtype));
}

torch::Tensor w8a8_sdot4_gemm_meta(torch::Tensor xq, torch::Tensor w,
                                   torch::Tensor x_s, torch::Tensor w_s,
                                   std::optional<torch::Tensor> bias,
                                   at::ScalarType out_dtype, int64_t cfg) {
  return at::empty({xq.size(0), w.size(0)}, xq.options().dtype(out_dtype));
}
