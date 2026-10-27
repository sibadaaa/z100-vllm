#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "core/utils.hpp"
#include "cub_helpers.h"
#include "legacy_dispatch_utils.h"
#include "quantization/vectorization_utils.cuh"
#include "torch_utils.h"

namespace vllm {

template <typename scalar_t>
__global__ void rms_norm_gated_gfx906_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight, const scalar_t* __restrict__ gate,
    const float epsilon, const int hidden_size, const int64_t input_stride,
    const int64_t gate_stride, const bool norm_before_gate) {
  __shared__ float s_variance;
  float variance = 0.0f;

  const int64_t row = blockIdx.x;
  const scalar_t* input_row = input + row * input_stride;
  const scalar_t* gate_row = gate + row * gate_stride;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = static_cast<float>(input_row[idx]);
    if (!norm_before_gate) {
      float z = static_cast<float>(gate_row[idx]);
      x *= z / (1.0f + expf(-z));
    }
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar_t* out_row = out + row * hidden_size;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = static_cast<float>(input_row[idx]);
    float z = static_cast<float>(gate_row[idx]);
    const float gate_val = z / (1.0f + expf(-z));
    float val = norm_before_gate ? x * s_variance * gate_val
                                 : x * gate_val * s_variance;
    val *= static_cast<float>(weight[idx]);
    out_row[idx] = static_cast<scalar_t>(val);
  }
}

template <typename scalar_t>
__global__ void gemma_rms_norm_gfx906_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight, const float epsilon,
    const int hidden_size, const int64_t input_stride) {
  __shared__ float s_variance;
  float variance = 0.0f;

  const int64_t row = blockIdx.x;
  const scalar_t* input_row = input + row * input_stride;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = static_cast<float>(input_row[idx]);
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar_t* out_row = out + row * hidden_size;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = static_cast<float>(input_row[idx]);
    const float w = 1.0f + static_cast<float>(weight[idx]);
    out_row[idx] = static_cast<scalar_t>(x * s_variance * w);
  }
}

template <typename scalar_t, typename residual_t>
__global__ void gemma_fused_add_rms_norm_gfx906_kernel(
    scalar_t* __restrict__ out, float* __restrict__ residual_out,
    const scalar_t* __restrict__ input,
    const residual_t* __restrict__ residual,
    const scalar_t* __restrict__ weight, const float epsilon,
    const int hidden_size, const int64_t input_stride,
    const int64_t residual_stride) {
  __shared__ float s_variance;
  float variance = 0.0f;

  const int64_t row = blockIdx.x;
  const scalar_t* input_row = input + row * input_stride;
  const residual_t* residual_row = residual + row * residual_stride;
  float* residual_out_row = residual_out + row * hidden_size;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = static_cast<float>(input_row[idx]) +
                    static_cast<float>(residual_row[idx]);
    variance += x * x;
    residual_out_row[idx] = x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar_t* out_row = out + row * hidden_size;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = residual_out_row[idx];
    const float w = 1.0f + static_cast<float>(weight[idx]);
    out_row[idx] = static_cast<scalar_t>(x * s_variance * w);
  }
}


}  // namespace vllm

torch::Tensor rms_norm_gated_gfx906(torch::Tensor input,    // [..., hidden_size]
                                    torch::Tensor weight,   // [hidden_size]
                                    torch::Tensor gate,     // [..., hidden_size]
                                    double epsilon,
                                    bool norm_before_gate) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == gate.scalar_type(),
              "input and gate must have the same dtype");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "input and weight must have the same dtype");
  TORCH_CHECK(input.sizes() == gate.sizes(),
              "input and gate must have the same shape");
  TORCH_CHECK(input.dim() == 2, "input must have shape [tokens, hidden_size]");
  TORCH_CHECK(input.stride(-1) == 1, "input last dimension must be contiguous");
  TORCH_CHECK(gate.stride(-1) == 1, "gate last dimension must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int hidden_size = input.size(-1);
  TORCH_CHECK(weight.numel() == hidden_size,
              "weight size must match input hidden size");
  const int num_tokens = input.numel() / hidden_size;
  const int64_t input_stride = input.stride(-2);
  const int64_t gate_stride = gate.stride(-2);
  auto out = torch::empty_like(input, input.options().memory_format(
                                          c10::MemoryFormat::Contiguous));

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(),
                               "rms_norm_gated_gfx906_kernel", [&] {
                                 vllm::rms_norm_gated_gfx906_kernel<scalar_t>
                                     <<<grid, block, 0, stream>>>(
                                         out.data_ptr<scalar_t>(),
                                         input.data_ptr<scalar_t>(),
                                         weight.data_ptr<scalar_t>(),
                                         gate.data_ptr<scalar_t>(), epsilon,
                                         hidden_size, input_stride, gate_stride,
                                         norm_before_gate);
                               });
  return out;
}

torch::Tensor gemma_rms_norm_gfx906(torch::Tensor input,
                                    torch::Tensor weight,
                                    double epsilon) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "input and weight must have the same dtype");
  TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
  if (input.stride(-1) != 1 || !input.is_contiguous()) {
    input = input.contiguous();
  }
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int hidden_size = input.size(-1);
  TORCH_CHECK(weight.numel() == hidden_size,
              "weight size must match input hidden size");
  const int num_tokens = input.numel() / hidden_size;
  const int64_t input_stride = input.stride(-2);
  auto out = torch::empty_like(input, input.options().memory_format(
                                          c10::MemoryFormat::Contiguous));

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(),
                               "gemma_rms_norm_gfx906_kernel", [&] {
                                 vllm::gemma_rms_norm_gfx906_kernel<scalar_t>
                                     <<<grid, block, 0, stream>>>(
                                         out.data_ptr<scalar_t>(),
                                         input.data_ptr<scalar_t>(),
                                         weight.data_ptr<scalar_t>(), epsilon,
                                         hidden_size, input_stride);
                               });
  return out;
}

std::vector<torch::Tensor> gemma_fused_add_rms_norm_gfx906(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight,
    double epsilon) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "input and weight must have the same dtype");
  TORCH_CHECK(residual.scalar_type() == input.scalar_type() ||
                  residual.scalar_type() == at::ScalarType::Float,
              "residual must be float32 or have the same dtype as input");
  TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
  TORCH_CHECK(residual.sizes() == input.sizes(),
              "residual shape must match input shape");
  if (input.stride(-1) != 1 || !input.is_contiguous()) {
    input = input.contiguous();
  }
  if (residual.stride(-1) != 1 || !residual.is_contiguous()) {
    residual = residual.contiguous();
  }
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  const int hidden_size = input.size(-1);
  TORCH_CHECK(weight.numel() == hidden_size,
              "weight size must match input hidden size");
  const int num_tokens = input.numel() / hidden_size;
  const int64_t input_stride = input.stride(-2);
  const int64_t residual_stride = residual.stride(-2);
  auto out = torch::empty_like(input, input.options().memory_format(
                                          c10::MemoryFormat::Contiguous));
  auto residual_out =
      torch::empty(input.sizes(),
                   torch::TensorOptions().dtype(torch::kFloat32).device(
                       input.device()));

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (residual.scalar_type() == at::ScalarType::Float) {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "gemma_fused_add_rms_norm_gfx906_kernel", [&] {
          vllm::gemma_fused_add_rms_norm_gfx906_kernel<scalar_t, float>
              <<<grid, block, 0, stream>>>(
                  out.data_ptr<scalar_t>(), residual_out.data_ptr<float>(),
                  input.data_ptr<scalar_t>(), residual.data_ptr<float>(),
                  weight.data_ptr<scalar_t>(), epsilon, hidden_size,
                  input_stride, residual_stride);
        });
  } else {
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "gemma_fused_add_rms_norm_gfx906_kernel", [&] {
          vllm::gemma_fused_add_rms_norm_gfx906_kernel<scalar_t, scalar_t>
              <<<grid, block, 0, stream>>>(
                  out.data_ptr<scalar_t>(), residual_out.data_ptr<float>(),
                  input.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(),
                  weight.data_ptr<scalar_t>(), epsilon, hidden_size,
                  input_stride, residual_stride);
        });
  }

  return {out, residual_out};
}

