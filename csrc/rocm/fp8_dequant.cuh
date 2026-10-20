// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// fp8 -> fp16 block dequantization for the W8A16 prefill path on gfx906.
//
// Why this file exists
// --------------------
// gfx906 has no fp8 support, so a large-M prefill GEMM that wants to run on
// rocBLAS has to materialize the current layer's weights as fp16 first
// (docs/design/z100-w8a16-prefill-fp16-construction.md, decision 2 / §4.5).
// This kernel produces that copy:
//
//   out[n, k] = fp16( fp32(wq[n, k] widened through `lut`) * ws[n/block_n, k/block_k] )
//
// The copy is per-layer scratch; weights never stay resident in fp16 (the
// §1.3 invariant), so only the fp8 checkpoint footprint is paid.
//
// Layout
// ------
// `out` keeps the [N, K] row order of the fp8 weight (no transpose); the GEMM
// consumes `out.T()`, which hipBLAS takes as a plain column-major operand --
// a [K, N] materialization would cost an LDS transpose and strided stores.
// `wq` is [N, K] fp8 with `stride(1) == 1`; `stride(0)` (passed as `w_stride`)
// may be padded to K + 256 by `_maybe_pad_fp8_weight`, so it is addressed
// explicitly instead of assuming a contiguous row pitch.
//
// Widening / scaling contract
// ---------------------------
// gfx906 cannot convert fp8 in hardware (`v_cvt_f32_fp8` does not assemble),
// and a bit-trick widening cannot represent e4m3 subnormals without per-lane
// fix-ups, so every fp8 code goes through a 256-entry fp16 table.  For both
// e4m3fn and e5m2 every finite code is exactly representable in fp16, hence
// the table is lossless.  It is staged into LDS once per block and read with
// `ds_read_u16` (see `gfx906_emu_common.cuh`).
//
// The block scale is float32 and must be applied in fp32: `fp16(raw) * s` is
// computed as an fp32 multiply and rounded to fp16 exactly once.  Multiplying
// in the fp16 domain would first round `s` itself to fp16 (a second rounding),
// which is not the §4.5 contract.
//
// Geometry
// --------
// One 8-byte word per thread: one `uint2` load of 8 fp8 codes, eight LDS table
// reads, one `uint4` (16 B) store.  The kernel is bandwidth bound and its grid
// is capped at 4096 blocks / 256 threads, with 512 B of LDS.
//
// Header-only on purpose: this file is `#include`d by `fp8_emu_gemm.cu`
// exactly once.  Adding a new .cu to CMakeLists would re-run hipify and force
// a rebuild of every HIP object in the extension.

#pragma once

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "rocm/gfx906_emu_common.cuh"

// out[N,K] fp16 <- wq[N,K] fp8 * ws[n/block_n, k/block_k].
//
// `nkb == K / block_k`, `K % 8 == 0`, `K % block_k == 0`, `block_k % 8 == 0`
// and `w_stride % 8 == 0` are all enforced on the host side below, so the
// grid-stride loop needs no tail guard and both the uint2 load and the uint4
// store are aligned.
__global__ void fp8_dequant_to_fp16_kernel(
    __half* __restrict__ out, const uint8_t* __restrict__ wq,
    const float* __restrict__ ws, const __half* __restrict__ lut, int N,
    int K, int w_stride, int block_n, int block_k, int nkb) {
  extern __shared__ __half sl[];  // 256 * 2 B
  vllm::gfx906_emu::fp8_lut_stage(sl, lut, (int)blockDim.x);
  __syncthreads();

  const int words_per_row = K >> 3;  // one word == 8 fp8 codes
  const int64_t n_words = (int64_t)N * words_per_row;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t wid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       wid < n_words; wid += stride) {
    const int n = (int)(wid / words_per_row);
    const int k0 = (int)(wid % words_per_row) << 3;
    const uint2 v =
        *reinterpret_cast<const uint2*>(wq + (int64_t)n * w_stride + k0);
    const float s = ws[(int64_t)(n / block_n) * nkb + (k0 / block_k)];

    __half h[8];
    const uint32_t w[2] = {v.x, v.y};
#pragma unroll
    for (int u = 0; u < 2; ++u) {
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const uint32_t code = (w[u] >> (8 * b)) & 0xFFu;
        const __half raw =
            __ushort_as_half(vllm::gfx906_emu::fp8_lut_u16(sl, code));
        h[u * 4 + b] = __float2half(__half2float(raw) * s);
      }
    }

    uint4 pack;
    pack.x =
        (uint32_t)__half_as_ushort(h[0]) | ((uint32_t)__half_as_ushort(h[1]) << 16);
    pack.y =
        (uint32_t)__half_as_ushort(h[2]) | ((uint32_t)__half_as_ushort(h[3]) << 16);
    pack.z =
        (uint32_t)__half_as_ushort(h[4]) | ((uint32_t)__half_as_ushort(h[5]) << 16);
    pack.w =
        (uint32_t)__half_as_ushort(h[6]) | ((uint32_t)__half_as_ushort(h[7]) << 16);
    *reinterpret_cast<uint4*>(out + (int64_t)n * K + k0) = pack;
  }
}

namespace vllm {
namespace gfx906_emu {

// Materialize `wq` (fp8, [N, K], row stride possibly padded) times its block
// scales into `out` (fp16, [N, K], contiguous).  `out` must not alias `wq`.
inline void dequant_fp8_block_to_fp16(at::Tensor& out, const at::Tensor& wq,
                                      const at::Tensor& ws,
                                      const at::Tensor& lut, int64_t block_n,
                                      int64_t block_k) {
  TORCH_CHECK(out.is_cuda() && wq.is_cuda() && ws.is_cuda() && lut.is_cuda(),
              "fp8_dequant: all tensors must be on the GPU");
  TORCH_CHECK(out.device() == wq.device() && out.device() == ws.device() &&
                  out.device() == lut.device(),
              "fp8_dequant: all tensors must be on the same device");
  TORCH_CHECK(out.scalar_type() == at::kHalf,
              "fp8_dequant: out must be float16, got ", out.scalar_type());
  TORCH_CHECK(out.dim() == 2 && out.is_contiguous(),
              "fp8_dequant: out must be a contiguous 2-D tensor");
  TORCH_CHECK(wq.dim() == 2 && wq.element_size() == 1,
              "fp8_dequant: wq must be a 2-D 8-bit (fp8) tensor");
  TORCH_CHECK(wq.stride(1) == 1 && wq.stride(0) > 0,
              "fp8_dequant: wq must be K-contiguous (stride(1) == 1)");
  TORCH_CHECK(lut.scalar_type() == at::kHalf && lut.numel() == 256 &&
                  lut.is_contiguous(),
              "fp8_dequant: lut must be a contiguous 256-element fp16 tensor");
  TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.dim() == 2 &&
                  ws.is_contiguous(),
              "fp8_dequant: ws must be a contiguous 2-D float32 tensor");

  const int64_t N = wq.size(0);
  const int64_t K = wq.size(1);
  const int64_t w_stride = wq.stride(0);
  TORCH_CHECK(out.size(0) == N && out.size(1) == K,
              "fp8_dequant: out must be [N, K] = [", N, ", ", K, "], got ",
              out.sizes());
  TORCH_CHECK(K % 8 == 0, "fp8_dequant: K must be a multiple of 8, got ", K);
  TORCH_CHECK(block_k > 0 && block_k % 8 == 0,
              "fp8_dequant: block_k must be a positive multiple of 8, got ",
              block_k);
  TORCH_CHECK(K % block_k == 0,
              "fp8_dequant: K must be a multiple of block_k (K=", K,
              " block_k=", block_k, ")");
  TORCH_CHECK(block_n > 0, "fp8_dequant: block_n must be positive, got ",
              block_n);
  TORCH_CHECK(w_stride % 8 == 0,
              "fp8_dequant: wq row stride must be a multiple of 8, got ",
              w_stride);

  const int64_t nkb = K / block_k;
  const int64_t nblk = (N + block_n - 1) / block_n;
  TORCH_CHECK(ws.size(0) == nblk && ws.size(1) == nkb,
              "fp8_dequant: ws must be [cdiv(N, block_n), K / block_k] = [",
              nblk, ", ", nkb, "], got ", ws.sizes());

  const int64_t kIntMax = std::numeric_limits<int>::max();
  TORCH_CHECK(N <= kIntMax && K <= kIntMax && w_stride <= kIntMax &&
                  block_n <= kIntMax && block_k <= kIntMax && nkb <= kIntMax,
              "fp8_dequant: dimensions exceed the int range");
  // uint2 / uint4 accesses below assume the base pointers are 8 B / 16 B
  // aligned; rows stay aligned because w_stride and K are multiples of 8.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(wq.data_ptr()) % 8 == 0,
              "fp8_dequant: wq data_ptr must be 8-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "fp8_dequant: out data_ptr must be 16-byte aligned");

  const int words_per_row = (int)(K >> 3);
  const int64_t n_words = N * words_per_row;
  if (n_words == 0) return;  // empty N or K: nothing to materialize

  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 4096;
  constexpr size_t kSmemBytes = 256 * sizeof(__half);
  const int blocks = (int)std::min<int64_t>((n_words + kThreads - 1) / kThreads,
                                            kMaxBlocks);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const c10::cuda::CUDAGuard guard(out.device());

  fp8_dequant_to_fp16_kernel<<<blocks, kThreads, kSmemBytes, stream>>>(
      reinterpret_cast<__half*>(out.data_ptr()),
      reinterpret_cast<const uint8_t*>(wq.data_ptr()),
      reinterpret_cast<const float*>(ws.data_ptr()),
      reinterpret_cast<const __half*>(lut.data_ptr()), (int)N, (int)K,
      (int)w_stride, (int)block_n, (int)block_k, (int)nkb);
}

}  // namespace gfx906_emu
}  // namespace vllm
