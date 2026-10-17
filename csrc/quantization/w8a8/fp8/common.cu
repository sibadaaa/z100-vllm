#include "common.cuh"
#include "core/cuda_compat.h"
#include "../../../dispatch_utils.h"
#include "../../../cub_helpers.h"
#include "../../vectorization_utils.cuh"
#include "../../../torch_utils.h"
#include <torch/csrc/stable/macros.h>

#include <algorithm>

namespace vllm {

namespace {

// 16 fp8 outputs form one 16-byte store; the matching 16-element input load
// is 32 B for 16-bit types and 64 B for fp32.
constexpr int kFp8QuantVecSize = 16;
constexpr int kFp8QuantBlockSize = 256;

template <typename T>
using fp8_quant_vec_t = vec_n_t<T, kFp8QuantVecSize>;

template <typename T>
bool ptr_vec_aligned(const T* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % alignof(fp8_quant_vec_t<T>) == 0;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, VLLM_SHFL_XOR_SYNC(v, offset));
  }
  return v;
}

// Block-wide max, returned to every thread. Warp partials go through one
// shared-memory exchange, so the whole reduction costs a single
// __syncthreads. blockDim.x must be a multiple of WARP_SIZE and at most
// kFp8QuantBlockSize.
__device__ __forceinline__ float block_reduce_max(float v, float* warp_max) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;
  v = warp_reduce_max(v);
  if (lane == 0) {
    warp_max[warp] = v;
  }
  __syncthreads();
  v = warp_max[0];
  for (int w = 1; w < num_warps; ++w) {
    v = fmaxf(v, warp_max[w]);
  }
  return v;
}

}  // namespace

// STRIDE_I_ZERO: true if scale_stride_i == 0 (per-tensor or per-channel)
// STRIDE_J_ZERO: true if scale_stride_j == 0 (per-tensor or per-token)
template <typename scalar_t, typename fp8_type, bool STRIDE_I_ZERO,
          bool STRIDE_J_ZERO>
__global__ void scaled_fp8_quant_kernel_strided_group_shape(
    fp8_type* __restrict__ out, const scalar_t* __restrict__ input,
    const float* __restrict__ scale, int hidden_size, int64_t in_row_stride,
    int64_t out_row_stride) {
  const int64_t token_idx = blockIdx.x;  // one token per block
  const int tid = threadIdx.x;

  const scalar_t* token_in = input + token_idx * in_row_stride;
  fp8_type* token_out = out + token_idx * out_row_stride;

  const float inv_scale = 1.0f / (*scale);

  vectorize_with_alignment<16>(
      token_in, token_out, hidden_size, tid, blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t& src) {
        dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src),
                                                    inv_scale);
      });
}

// Per-tensor abs-max of a contiguous tensor viewed as a flat array of
// 16-element vectors: grid-stride loop, one atomic per block.
template <typename scalar_t, typename fp8_type>
__global__ void __launch_bounds__(kFp8QuantBlockSize)
    absmax_reduction_flat(float* __restrict__ scale,
                          const scalar_t* __restrict__ input,
                          int64_t num_vecs) {
  using in_vec_t = fp8_quant_vec_t<scalar_t>;
  const auto* v_in = reinterpret_cast<const in_vec_t*>(input);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  float thread_max = 0.0f;
#pragma unroll 4
  for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < num_vecs; v += stride) {
    const in_vec_t vec = v_in[v];
#pragma unroll
    for (int i = 0; i < kFp8QuantVecSize; ++i) {
      thread_max = fmaxf(thread_max, fabsf(static_cast<float>(vec.val[i])));
    }
  }

  __shared__ float warp_max[kFp8QuantBlockSize / 32];
  const float block_max = block_reduce_max(thread_max, warp_max);
  if (threadIdx.x == 0) {
    atomicMaxFloat(scale, block_max / quant_type_max_v<fp8_type>);
  }
}

// Per-tensor abs-max for strided or unaligned rows: one block per token.
template <typename scalar_t, typename fp8_type>
__global__ void __launch_bounds__(kFp8QuantBlockSize)
    segmented_max_reduction_strided(float* __restrict__ scale,
                                    const scalar_t* __restrict__ input,
                                    int hidden_size, int64_t in_row_stride,
                                    int64_t num_tokens) {
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }
  const scalar_t* row_ptr = input + token_idx * in_row_stride;

  float thread_max = 0.0f;
  vectorize_read_with_alignment<kFp8QuantVecSize>(
      row_ptr, hidden_size, threadIdx.x, blockDim.x,
      [&] __device__(const scalar_t& v) {
        thread_max = fmaxf(thread_max, fabsf(static_cast<float>(v)));
      });

  __shared__ float warp_max[kFp8QuantBlockSize / 32];
  const float block_max = block_reduce_max(thread_max, warp_max);
  if (threadIdx.x == 0) {
    atomicMaxFloat(scale, block_max / quant_type_max_v<fp8_type>);
  }
}

template <typename scalar_t, typename fp8_type>
__global__ void scaled_fp8_quant_kernel_strided_dynamic(
    fp8_type* __restrict__ out, const scalar_t* __restrict__ input,
    const float* __restrict__ scale, int hidden_size, int64_t in_row_stride,
    int64_t out_row_stride) {
  const int64_t token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const scalar_t* token_in = input + token_idx * in_row_stride;
  fp8_type* token_out = out + token_idx * out_row_stride;

  const float reciprocal_scale = 1.0f / (*scale);
  vectorize_with_alignment<16>(
      token_in, token_out, hidden_size, tid, blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t& src) {
        dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src),
                                                    reciprocal_scale);
      });
}

template <typename scalar_t, typename fp8_type>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel_strided(
    fp8_type* __restrict__ out, float* __restrict__ scale,
    const scalar_t* __restrict__ input, const float* __restrict__ scale_ub,
    int hidden_size, int64_t in_row_stride, int64_t out_row_stride) {
  const int64_t token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  // Use int64 to avoid overflowing an int32 when calculating this offset
  int64_t in_offset = static_cast<int64_t>(token_idx) * in_row_stride;
  int64_t out_offset = static_cast<int64_t>(token_idx) * out_row_stride;
  const scalar_t* token_in = input + in_offset;
  fp8_type* token_out = out + out_offset;

  // 1) per-token absmax
  float absmax_val = 0.f;
  vectorize_read_with_alignment<16>(
      token_in, hidden_size, tid, blockDim.x, [&] __device__(scalar_t v) {
        absmax_val = fmaxf(absmax_val, fabsf(static_cast<float>(v)));
      });

  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;
  const float block_max =
      BlockReduce(tmp).Reduce(absmax_val, CubMaxOp{}, blockDim.x);

  __shared__ float token_scale;
  if (tid == 0) {
    token_scale = scale_ub ? fminf(block_max, *scale_ub) : block_max;
    token_scale = fmaxf(token_scale / quant_type_max_v<fp8_type>,
                        min_scaling_factor<fp8_type>::val());
    scale[token_idx] = token_scale;
  }
  __syncthreads();

  // 2) quantize
  vectorize_with_alignment<16>(
      token_in, token_out, hidden_size, tid, blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t& src) {
        dst = scaled_fp8_conversion<false, fp8_type>(static_cast<float>(src),
                                                     token_scale);
      });
}

}  // namespace vllm

void static_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                             torch::Tensor const& input,  // [..., d]
                             torch::Tensor const& scale)  // [1]
{
  TORCH_CHECK(input.stride(-1) == 1,
              "last dimension of input must be contiguous");
  TORCH_CHECK(out.stride(-1) == 1,
              "last dimension of output must be contiguous");

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  const int block_size = 256;
  dim3 grid(num_tokens);
  dim3 block(block_size);

  const int64_t in_row_stride = input.stride(-2);
  const int64_t out_row_stride = out.stride(-2);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              vllm::scaled_fp8_quant_kernel_strided<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(
                      out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                      scale.data_ptr<float>(), hidden_size, in_row_stride,
                      out_row_stride);
            });
      });
}

void dynamic_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                              torch::Tensor const& input,  // [..., d]
                              torch::Tensor& scale)        // [1]
{
  TORCH_CHECK(input.stride(-1) == 1,
              "last dimension of input must be contiguous");
  TORCH_CHECK(out.stride(-1) == 1,
              "last dimension of output must be contiguous");

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  const int block_size = vllm::kFp8QuantBlockSize;
  dim3 grid(num_tokens);
  dim3 block(block_size);

  const int64_t in_row_stride = input.stride(-2);
  const int64_t out_row_stride = out.stride(-2);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // scale tensor should be initialised to <=0 before reduction
  AT_CUDA_CHECK(
      cudaMemsetAsync(scale.data_ptr<float>(), 0, sizeof(float), stream));

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              const scalar_t* in_ptr = input.const_data_ptr<scalar_t>();
              const bool flat = in_row_stride == hidden_size &&
                                hidden_size % vllm::kFp8QuantVecSize == 0 &&
                                vllm::ptr_vec_aligned(in_ptr);
              if (flat) {
                const int64_t num_vecs = input.numel() / vllm::kFp8QuantVecSize;
                const int64_t max_blocks =
                    static_cast<int64_t>(
                        get_device_prop()->multiProcessorCount) *
                    8;
                const int64_t num_blocks = std::max<int64_t>(
                    1,
                    std::min<int64_t>(
                        max_blocks, (num_vecs + block_size - 1) / block_size));
                vllm::absmax_reduction_flat<scalar_t, fp8_t>
                    <<<num_blocks, block, 0, stream>>>(
                        scale.mutable_data_ptr<float>(), in_ptr, num_vecs);
              } else {
                vllm::segmented_max_reduction_strided<scalar_t, fp8_t>
                    <<<grid, block, 0, stream>>>(
                        scale.mutable_data_ptr<float>(), in_ptr, hidden_size,
                        in_row_stride, static_cast<int64_t>(num_tokens));
              }

              vllm::scaled_fp8_quant_kernel_strided_dynamic<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(
                      out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                      scale.data_ptr<float>(), hidden_size, in_row_stride,
                      out_row_stride);
            });
      });
}

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out,          // [..., d]
    torch::Tensor const& input,  // [..., d]
    torch::Tensor& scales, std::optional<at::Tensor> const& scale_ub) {
  TORCH_CHECK(input.stride(-1) == 1,
              "last dimension of input must be contiguous");
  TORCH_CHECK(out.stride(-1) == 1,
              "last dimension of output must be contiguous");

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  const int block_size = 256;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, block_size));

  const int64_t in_row_stride = input.stride(-2);
  const int64_t out_row_stride = out.stride(-2);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "dynamic_per_token_scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(),
            "dynamic_per_token_scaled_fp8_quant_kernel_fp8_type", [&] {
              vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<
                  scalar_t, fp8_t><<<grid, block, 0, stream>>>(
                  out.data_ptr<fp8_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  hidden_size, in_row_stride, out_row_stride);
            });
      });
}
