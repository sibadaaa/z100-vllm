// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Dense fp16 / bf16 GEMV for gfx906 (Vega20 / Hygon Z100), decode path.
//
// Why
// ---
// On gfx906 the two ROCm fast paths for decode-shaped unquantized linear
// layers are both unavailable: `ops.LLMM1` (skinny GEMM) is excluded from the
// gfx906 build because it targets the MFMA ISA, and the `n <= 16` Triton
// matmul is gated `not on_gfx9()`.  `rocm_unquantized_gemm_impl` therefore
// reaches `torch.nn.functional.linear` -> rocBLAS, which on this card moves
// ~235 GB/s of weights for a [1 x 5120] @ [17408 x 5120] bf16 GEMV (24% of
// HBM) and only ~24 GB/s for small-N shapes (N=512, bf16: 0.215 ms for a
// 5 MiB weight read).
//
// What
// ----
// One thread block per output column n, `nt` threads streaming the row of W
// with coalesced dword loads and an LDS tree reduction across threads -- the
// same blocking that `w8a8_sdot4_gemv` already validated on this hardware.
// Two accumulation routes:
//
//   * dot2 route (`route = 2`, and the default for fp16): each dword holds
//     two fp16 lanes and feeds one `v_dot2_f32_f16` per row of A, i.e. two
//     MACs per instruction with an fp32 accumulator.  For fp16 inputs this is
//     bit-for-bit the same arithmetic class as rocBLAS's fp16 dot2 kernels.
//   * fp32 route (`route = 1`, and the default for bf16): each bf16 lane is
//     widened with `v_lshlrev_b32 16` (exact for every bf16 value, including
//     subnormals and inf) and accumulated with fp32 FMA.  This is the "bf16
//     weights, fp32 math" contract that a bf16 tensor core with fp32
//     accumulation would provide, so it is the safe default.
//   * opt-in bf16 -> fp16 rebase (`route = 2` + `range_safe`): the packed
//     3-ALU sequence from gfx906_emu_common.cuh.  fp16's exponent window is
//     ~30 binades narrower than bf16's, so this route must NOT be used on
//     arbitrary weights: values below 2^-14 flush to zero, which for a weight
//     distribution with sigma ~1e-3 destroys ~6% of the signal (the flushed
//     mass is an absolute, not relative, threshold).  The Python caller is
//     responsible for either passing `range_safe=true` only after a per-row
//     range check, or leaving the route on fp32.
//
// Weights are read exactly as laid out by the caller ([N, K], K contiguous);
// nothing is transposed or copied.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "rocm/bf16_emu_gemv.h"
#include "rocm/gfx906_emu_common.cuh"

namespace {

using vllm::gfx906_emu::bf16x2_to_f16x2_fast;
using vllm::gfx906_emu::bf16x2_to_f16x2_guarded;
using vllm::gfx906_emu::bf16x2_to_f32x2;
using vllm::gfx906_emu::dot2;

constexpr int kMsMax = 8;
constexpr int kNtMax = 512;

// ---------------------------------------------------------------------------
// dot2 route
// ---------------------------------------------------------------------------

// Source pair -> fp16 pair.  `IS_BF16=false` is the fp16 passthrough.
template <bool IS_BF16, bool RANGE_SAFE>
__device__ __forceinline__ uint32_t pair_to_f16x2(uint32_t p) {
  if (!IS_BF16) return p;
  if (RANGE_SAFE) return bf16x2_to_f16x2_fast(p);
  return bf16x2_to_f16x2_guarded(p);
}

template <int MS, typename T, bool IS_BF16, bool RANGE_SAFE>
__global__ void bf16_emu_gemv_dot2_kernel(const uint32_t* __restrict__ xp,
                                          const uint32_t* __restrict__ wp,
                                          const T* __restrict__ bias,
                                          T* __restrict__ out, const int M,
                                          const int N, const int K2) {
  extern __shared__ float red[];  // [MS][nt]
  const int n = blockIdx.x;
  const int t = threadIdx.x;
  const int nt = blockDim.x;

  const uint32_t* __restrict__ wrow = wp + (int64_t)n * K2;

  float acc[MS];
#pragma unroll
  for (int m = 0; m < MS; ++m) acc[m] = 0.0f;

  for (int i = t; i < K2; i += nt) {
    const uint32_t wh = pair_to_f16x2<IS_BF16, RANGE_SAFE>(wrow[i]);
#pragma unroll
    for (int m = 0; m < MS; ++m) {
      if (m < M) {
        const uint32_t xh = pair_to_f16x2<IS_BF16, RANGE_SAFE>(
            xp[(int64_t)m * K2 + i]);
        acc[m] = dot2(wh, xh, acc[m]);
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
    T o = static_cast<T>(red[t * nt]);
    if (bias != nullptr) o += bias[n];
    out[t * (int64_t)N + n] = o;
  }
}

// ---------------------------------------------------------------------------
// fp32 route (bf16 operands widened with v_lshlrev_b32 16, exact)
// ---------------------------------------------------------------------------

template <int MS, typename T>
__global__ void bf16_emu_gemv_f32_kernel(const uint32_t* __restrict__ xp,
                                         const uint32_t* __restrict__ wp,
                                         const T* __restrict__ bias,
                                         T* __restrict__ out, const int M,
                                         const int N, const int K2) {
  extern __shared__ float red[];  // [MS][nt]
  const int n = blockIdx.x;
  const int t = threadIdx.x;
  const int nt = blockDim.x;

  const uint32_t* __restrict__ wrow = wp + (int64_t)n * K2;

  float acc[MS];
#pragma unroll
  for (int m = 0; m < MS; ++m) acc[m] = 0.0f;

  for (int i = t; i < K2; i += nt) {
    float wlo, whi;
    bf16x2_to_f32x2(wrow[i], wlo, whi);
#pragma unroll
    for (int m = 0; m < MS; ++m) {
      if (m < M) {
        float xlo, xhi;
        bf16x2_to_f32x2(xp[(int64_t)m * K2 + i], xlo, xhi);
        acc[m] = fmaf(wlo, xlo, acc[m]);
        acc[m] = fmaf(whi, xhi, acc[m]);
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
    T o = static_cast<T>(red[t * nt]);
    if (bias != nullptr) o += bias[n];
    out[t * (int64_t)N + n] = o;
  }
}

template <typename T>
void launch_dot2(int M, int N, int K2, int nt, bool is_bf16, bool range_safe,
                 const uint32_t* xp, const uint32_t* wp, const T* bias, T* out,
                 hipStream_t stream) {
  const size_t smem = [&]() {
    if (M <= 1) return (size_t)1 * nt * sizeof(float);
    if (M <= 4) return (size_t)4 * nt * sizeof(float);
    return (size_t)kMsMax * nt * sizeof(float);
  }();
  dim3 grid(N), block(nt);
#define VLLM_BF16_GEMV_CASE(MS)                                              \
  do {                                                                       \
    if (is_bf16) {                                                           \
      if (range_safe)                                                        \
        bf16_emu_gemv_dot2_kernel<MS, T, true, true>                         \
            <<<grid, block, smem, stream>>>(xp, wp, bias, out, M, N, K2);    \
      else                                                                   \
        bf16_emu_gemv_dot2_kernel<MS, T, true, false>                        \
            <<<grid, block, smem, stream>>>(xp, wp, bias, out, M, N, K2);    \
    } else {                                                                 \
      bf16_emu_gemv_dot2_kernel<MS, T, false, false>                         \
          <<<grid, block, smem, stream>>>(xp, wp, bias, out, M, N, K2);      \
    }                                                                        \
    return;                                                                  \
  } while (0)
  if (M <= 1) VLLM_BF16_GEMV_CASE(1);
  if (M <= 4) VLLM_BF16_GEMV_CASE(4);
  VLLM_BF16_GEMV_CASE(kMsMax);
#undef VLLM_BF16_GEMV_CASE
}

template <typename T>
void launch_f32(int M, int N, int K2, int nt, const uint32_t* xp,
                const uint32_t* wp, const T* bias, T* out,
                hipStream_t stream) {
  const size_t smem = [&]() {
    if (M <= 1) return (size_t)1 * nt * sizeof(float);
    if (M <= 4) return (size_t)4 * nt * sizeof(float);
    return (size_t)kMsMax * nt * sizeof(float);
  }();
  dim3 grid(N), block(nt);
  if (M <= 1)
    bf16_emu_gemv_f32_kernel<1, T>
        <<<grid, block, smem, stream>>>(xp, wp, bias, out, M, N, K2);
  else if (M <= 4)
    bf16_emu_gemv_f32_kernel<4, T>
        <<<grid, block, smem, stream>>>(xp, wp, bias, out, M, N, K2);
  else
    bf16_emu_gemv_f32_kernel<kMsMax, T>
        <<<grid, block, smem, stream>>>(xp, wp, bias, out, M, N, K2);
}

}  // namespace

torch::Tensor bf16_emu_gemv(torch::Tensor x, torch::Tensor w,
                            std::optional<torch::Tensor> bias, int64_t nthreads,
                            int64_t route, bool range_safe) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "bf16_emu: x/w must be on the GPU");
  TORCH_CHECK(x.scalar_type() == w.scalar_type(),
              "bf16_emu: x/w must share a dtype");
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "bf16_emu: only fp16 and bf16 inputs are supported");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
              "bf16_emu: x/w must be contiguous");
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K, "bf16_emu: K mismatch");
  TORCH_CHECK(K % 2 == 0, "bf16_emu: K must be a multiple of 2, got ", K);
  TORCH_CHECK(M >= 1 && M <= kMsMax, "bf16_emu: M must be in [1,", kMsMax,
              "], got ", M);
  TORCH_CHECK(nthreads > 0 && nthreads <= kNtMax &&
                  (nthreads & (nthreads - 1)) == 0,
              "bf16_emu: nthreads must be a power of two <= ", kNtMax);
  TORCH_CHECK(route >= 0 && route <= 2, "bf16_emu: route must be 0..2, got ",
              route);
  const bool is_bf16 = x.scalar_type() == at::kBFloat16;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() &&
                    bias->numel() == N && bias->scalar_type() == x.scalar_type(),
                "bf16_emu: bias must be a contiguous GPU tensor of N elements "
                "in the input dtype");
  }

  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({M, N}, x.options());

  // route: 0 = auto (fp16 -> dot2, bf16 -> fp32), 1 = fp32, 2 = dot2.
  int r = (int)route;
  if (r == 0) r = is_bf16 ? 1 : 2;
  if (!is_bf16 && r == 1) r = 2;  // fp16 has no fp32 route here (dot2 is exact)

  const uint32_t* xp = reinterpret_cast<const uint32_t*>(x.data_ptr());
  const uint32_t* wp = reinterpret_cast<const uint32_t*>(w.data_ptr());
  const void* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined()) bias_ptr = bias->data_ptr();
  const int Mi = (int)M, Ni = (int)N, K2 = (int)(K / 2);
  const int nt = (int)nthreads;
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  if (r == 2) {
    if (out.scalar_type() == at::kHalf) {
      launch_dot2<at::Half>(Mi, Ni, K2, nt, is_bf16, range_safe, xp, wp,
                            reinterpret_cast<const at::Half*>(bias_ptr),
                            reinterpret_cast<at::Half*>(out.data_ptr()),
                            stream);
    } else {
      launch_dot2<at::BFloat16>(
          Mi, Ni, K2, nt, is_bf16, range_safe, xp, wp,
          reinterpret_cast<const at::BFloat16*>(bias_ptr),
          reinterpret_cast<at::BFloat16*>(out.data_ptr()), stream);
    }
  } else {
    if (out.scalar_type() == at::kHalf) {
      launch_f32<at::Half>(Mi, Ni, K2, nt, xp, wp,
                           reinterpret_cast<const at::Half*>(bias_ptr),
                           reinterpret_cast<at::Half*>(out.data_ptr()),
                           stream);
    } else {
      launch_f32<at::BFloat16>(
          Mi, Ni, K2, nt, xp, wp,
          reinterpret_cast<const at::BFloat16*>(bias_ptr),
          reinterpret_cast<at::BFloat16*>(out.data_ptr()), stream);
    }
  }
  return out;
}

torch::Tensor bf16_emu_gemv_meta(torch::Tensor x, torch::Tensor w,
                                 std::optional<torch::Tensor> bias,
                                 int64_t nthreads, int64_t route,
                                 bool range_safe) {
  return x.new_empty({x.size(0), w.size(0)});
}
