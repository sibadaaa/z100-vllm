// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// S7b: minimal stand-alone copy of the *scalar* gfx906 activation kernel.
//
// CAUTION: this mirrors csrc/activation_kernels.cu:14-34 and :123-127, which is
// NOT the translation unit the eager fp16 path actually launches.  The torch
// profiler shows the live kernel is the packed/vectorised one from
// csrc/libtorch_stable/activation_kernels.cu (`silu_and_mul` ->
// `act_and_mul_kernel<half, __half2, silu_kernel, packed_silu_kernel, true,
// true, false, false>`), so the assembly below is NOT evidence about the
// defect; it is kept only as the template of the `hipcc -S` inspection method.
// See docs/result/z100-07b-eager-nan-result.md section 3.6.
//
// Compile-only reproducer (hipcc -x hip --cuda-device-only -S).  Never build it
// into an executable: running self-compiled HIP binaries on this host is
// forbidden (todolist hard rule 1).
//
//   hipcc -x hip --cuda-device-only -S -O3 --offload-arch=gfx906 \
//         -o /mnt/game/port/s7b/repro_act_gfx906.s s7b_repro_act_sfu.cu

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  const int token_idx = blockIdx.x;
  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = __ldg(&input[token_idx * 2 * d + idx]);
    const scalar_t y = __ldg(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}

template __global__ void act_and_mul_kernel<half, silu_kernel<half>, true>(
    half*, const half*, int);
template __global__ void act_and_mul_kernel<half, silu_kernel<half>, false>(
    half*, const half*, int);
