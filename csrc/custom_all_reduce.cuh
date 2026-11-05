#pragma once

#include "custom_collective_common.cuh"

namespace vllm {

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

// S33 WP2: push-mode one-shot all-reduce for gfx906.
//
// On gfx906 a GPU's PCIe-outbound READ of another GPU's memory does not
// observe that GPU's L2 dirty lines (the upstream 1stage kernel PULLs peer
// buffers and reads stale data), while PCIe-inbound WRITES into a GPU are
// coherent at that GPU's L2 -- which is why the signal-flag handshake always
// worked. This kernel therefore PUSHes: each rank's staged input (host copy
// into its own fine-grain buffer region at offset rank*chunk) is volatile-
// stored into every peer's buffer region for this rank; after a barrier each
// rank reduces ITS OWN buffer regions with plain local loads (coherent).
//
// Buffer layout (per rank, fine-grain buffer of reg_buffer_sz_bytes):
//   region r = buffer + r * chunk_p elements, chunk_p >= payload P-elems.
// Accumulation order = rank 0..ngpus-1, identical to the 1stage kernel.
// S33 WP2: push-mode one-shot all-reduce for gfx906 (single kernel).
//
// On gfx906 a GPU's PCIe-outbound READ of another GPU's memory does not
// observe that GPU's L2 dirty lines (the upstream 1stage kernel PULLs peer
// buffers and reads stale data), while PCIe-inbound WRITES into a GPU are
// coherent at that GPU's L2 -- which is why the signal-flag handshake always
// worked. This kernel therefore PUSHes: it reads the rank's OWN activation
// (src, plain local load) and writes it into every peer's fine-grain buffer
// region for this rank; after a barrier each rank reduces ITS OWN buffer
// regions with plain local loads (coherent).
//
// Buffer layout (per rank, fine-grain buffer of reg_buffer_sz_bytes):
//   region r = buffer + r * chunk_p elements, chunk_p >= payload P-elems.
// Accumulation order = rank 0..ngpus-1, identical to the 1stage kernel.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_push(RankData* _dp, RankSignals sg, Signal* self_sg,
                             const T* __restrict__ src, T* __restrict__ result,
                             int rank, int size, int chunk_p) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  const P* own_base = (const P*)dp.ptrs[rank];  // my fine-grain buffer
  const P* src_p = (const P*)src;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  // push: write my activation slice into every peer's buffer region for my
  // rank (PCIe-inbound writes are coherent at the target GPU's L2)
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    const P v = src_p[idx];
    // plain 16 B store: the inbound PCIe write is coherent at the target
    // GPU's L2 on gfx906, so the peer's own local reduce sees it
#pragma unroll
    for (int p = 0; p < ngpus; p++) {
      if (p == rank) continue;
      ((P*)((P*)dp.ptrs[p] + (size_t)rank * chunk_p))[idx] = v;
    }
  }
  barrier_mid_volatile<ngpus>(sg, self_sg, rank);
  // local reduce over MY buffer regions (rank 0..ngpus-1 order, as 1stage):
  // region p of my buffer holds rank p's contribution (pushed to me); the
  // diagonal (p == rank) is my own activation, read from src directly (no
  // rank pushes to itself)
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    A acc = upcast(rank == 0 ? src_p[idx] : own_base[idx]);
#pragma unroll
    for (int p = 1; p < ngpus; p++) {
      packed_assign_add(
          acc, upcast(rank == p ? src_p[idx] : own_base[(size_t)p * chunk_p + idx]));
    }
    ((P*)result)[idx] = downcast<P>(acc);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

// S37 W4: push-mode one-shot all-reduce with a fused add-rmsnorm epilogue
// (design docs/design/z100/s34/z100-s34-wp4-ar-fusion.md §2; S37
// construction §2).  The all-reduce itself is the push kernel unchanged:
// same push writes, same barrier sequence, same rank 0..ngpus-1 fp32
// accumulation order.  The epilogue re-rounds the fp32 sum to T (matching
// the unfused path, where the fp16 AR result is written to memory and the
// norm kernel re-reads it) before adding the residual, so the only NEW
// float order is the in-block sum-of-squares tree -- parity is governed by
// the max|delta|/top5-logprob board, not torch.equal.
//
// Launch contract (host-validated, see CustomAllreduce::allreduce_fused):
//   size (P-units) == rows * row_p, row_p <= blockDim.x, rows <= kMaxBlocks;
//   grid = rows (one block per hidden row), block = 1024 threads.  The
//   barriers are per-blockIdx cross-rank handshakes, so a different grid
//   shape than the unfused kernel is safe as long as every rank launches
//   the same geometry.
template <typename T, int ngpus>
__global__ void __launch_bounds__(1024, 1)
    cross_device_reduce_push_fused(RankData* _dp, RankSignals sg,
                                   Signal* self_sg, const T* __restrict__ src,
                                   T* __restrict__ result,
                                   T* __restrict__ residual,
                                   const T* __restrict__ norm_w, int rank,
                                   int size, int chunk_p, int row_p,
                                   float eps) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int kElems = P::size;  // T elements per packed unit
  auto dp = *_dp;
  const P* own_base = (const P*)dp.ptrs[rank];
  const P* src_p = (const P*)src;
  const int r0 = blockIdx.x * row_p;  // my row's P-unit offset
  __shared__ float red[32];           // warp partials for sum(s^2)
  __shared__ float s_rrms;

  barrier_at_start<ngpus>(sg, self_sg, rank);
  // push: write my row's activation slice into every peer's buffer region
  // for my rank (same traffic pattern as the unfused kernel)
  for (int idx = r0 + threadIdx.x; idx < r0 + row_p; idx += blockDim.x) {
    const P v = src_p[idx];
#pragma unroll
    for (int p = 0; p < ngpus; p++) {
      if (p == rank) continue;
      ((P*)((P*)dp.ptrs[p] + (size_t)rank * chunk_p))[idx] = v;
    }
  }
  barrier_mid_volatile<ngpus>(sg, self_sg, rank);
  // local reduce over MY buffer regions (rank 0..ngpus-1 order, as 1stage /
  // unfused push) + epilogue: s = round_T(sum) + residual; residual = s;
  // result = s * rrms * (1 + w)
  //
  // S38 A1: the epilogue applies the Gemma RMSNorm convention -- out =
  // (s * rrms) * (1 + w) with the +1 in fp32 -- exactly the math of the
  // production norm on this stack (ir.ops.fused_add_rms_norm native impl as
  // traced into the serving graphs: x_f32 = fp16(AR) + residual in fp32,
  // residual stored back as fp16, out = (x * rrms) * (weight+1) in fp32,
  // single downcast at the end).  norm_w is therefore the RAW Gemma weight
  // parameter, not a pre-added one.
  const int idx = r0 + threadIdx.x;
  A sv = {};  // at most one packed unit per thread (row_p <= blockDim.x)
  float ssq = 0.0f;
  if (threadIdx.x < row_p) {
    A acc = upcast(rank == 0 ? src_p[idx] : own_base[idx]);
#pragma unroll
    for (int p = 1; p < ngpus; p++) {
      packed_assign_add(
          acc,
          upcast(rank == p ? src_p[idx] : own_base[(size_t)p * chunk_p + idx]));
    }
    sv = upcast(downcast<P>(acc));  // the AR quant point, kept identical
    packed_assign_add(sv, upcast(((const P*)residual)[idx]));
    ((P*)residual)[idx] = downcast<P>(sv);
#pragma unroll
    for (int j = 0; j < kElems; j++) {
      const float x = sv.data[j];
      ssq += x * x;
    }
  }
  // block-wide sum of squares: warp tree, then the first warp reduces the
  // 32 warp partials; inactive lanes contribute 0 so every warp converges
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ssq += __shfl_down(ssq, off);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ssq;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = red[threadIdx.x];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off);
    if (threadIdx.x == 0) {
      const float n = (float)row_p * kElems;
      s_rrms = rsqrtf(v / n + eps);
    }
  }
  __syncthreads();
  if (threadIdx.x < row_p) {
    // weight index is per-row: within-row P-unit offset == threadIdx.x;
    // multiply order matches the standalone norm kernel: (s * rrms) * (1 + w)
    // (Gemma convention, see the S38 A1 note above; the +1 stays in fp32)
    const A w = upcast(((const P*)norm_w)[threadIdx.x]);
    A outv = sv;
#pragma unroll
    for (int j = 0; j < kElems; j++)
      outv.data[j] = (sv.data[j] * s_rrms) * (w.data[j] + 1.0f);
    ((P*)result)[idx] = downcast<P>(outv);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  barrier_at_start<ngpus>(sg, self_sg, rank);

  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  barrier_at_end<ngpus>(sg, self_sg, rank);

  // stage 2: allgather. Note: it's important to match the tid between
  // the two stages, because visibility across devices is only guaranteed
  // between threads that have the same tid. If thread i computes the sum of
  // start + i in the first stage, then thread i also gathers start + i from
  // all ranks.

  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

class CustomAllreduce {
 public:
  int rank_;
  int world_size_;
  // Full NVLink or xGMI connection between GPUs.
  bool fully_connected_;

  RankSignals sg_;
  // Stores a map from a pointer to its peer pointers from all ranks.
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  // Stores rank data from all ranks. This is mainly for cuda graph purposes.
  // For cuda graph to work, all kernel arguments must be fixed during graph
  // capture time. However, the peer pointers are not known during graph
  // capture time. Therefore, during capture, we increment the rank data
  // pointer and use that as the argument to the kernel. The kernel arguments
  // are stored in graph_unreg_buffers_. The actual peer pointers will be
  // filled in at the memory pointed to by the pointers in
  // graph_unreg_buffers_ when the IPC handles are exchanged between ranks.
  //
  // The overall process looks like this:
  // 1. Graph capture.
  // 2. Each rank obtains the IPC handles for each addresses used during cuda
  // graph capture using get_graph_buffer_ipc_meta.
  // 3. (In Python) all gather the IPC handles.
  // 4. Obtain the peer pointers by opening the IPC handles, and store them in
  // the rank data array at corresponding positions.
  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  // a map from IPC handles to opened IPC pointers
  std::map<IPC_KEY, char*> ipc_handles_;

  /**
   * Signals are an array of ipc-enabled buffers from all ranks.
   * For each of the buffer, the layout is as follows:
   * | -- sizeof(Signal) -- | ------ a few MB ----- |
   * The first section is for allreduce synchronization, and the second
   * section is for storing the intermediate results required by some
   * allreduce algos.
   *
   * Note: this class does not own any device memory. Any required buffers
   * are passed in from the constructor.
   */
  CustomAllreduce(Signal** signals, void* rank_data, size_t rank_data_sz,
                  int rank, int world_size, bool fully_connected = true)
      : rank_(rank),
        world_size_(world_size),
        fully_connected_(fully_connected),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
    }
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] =
        ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_ptr,
                                     *((const cudaIpcMemHandle_t*)ipc_handle),
                                     cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      // note: must share the base address of each allocation, or we get wrong
      // address
      if (cuPointerGetAttribute(&base_ptr, rangeStartAddrAttr,
                                (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CUDACHECK(cudaIpcGetMemHandle(
          (cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error(
          "Rank data buffer is overflowed by " +
          std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
  }

  /**
   * Register already-shared IPC pointers.
   */
  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      data.ptrs[i] = ptrs[i];
    }
    auto d_data = d_rank_data_base_++;
    CUDACHECK(
        cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  // Note: when registering graph buffers, we intentionally choose to not
  // deduplicate the addresses. That means if the allocator reuses some
  // addresses, they will be registered again. This is to account for the
  // remote possibility of different allocation patterns between ranks. For
  // example, rank 1 may get the same input address for the second allreduce,
  // but rank 2 got a different address. IPC handles have internal reference
  // counting mechanism so overhead should be small.
  void register_graph_buffers(
      const std::vector<std::string>& handles,
      const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle =
              open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CUDACHECK(cudaMemcpy(d_rank_data_base_, rank_data.data(),
                         sizeof(RankData) * num_buffers,
                         cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  /**
   * Performs allreduce, assuming input has already been registered.
   *
   * Block and grid default configs are results after careful grid search.
   * Using 36 blocks give the best or close to the best runtime on the devices
   * I tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also
   * only take a small amount of SMs. Not quite sure the underlying reason,
   * but my guess is that too many SMs will cause contention on NVLink bus.
   */
  template <typename T>
  void allreduce(cudaStream_t stream, T* input, const T* src, T* output,
                 int size, int threads = 512,
                 int block_limit = defaultBlockLimit, int chunk_p = 0) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "custom allreduce currently requires input length to be multiple "
          "of " +
          std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " +
                               std::to_string(kMaxBlocks) + ". Got " +
                               std::to_string(block_limit));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input)) +
            " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    auto bytes = size * sizeof(typename packed_t<T>::P);
    int blocks = std::min(block_limit, (size + threads - 1) / threads);

    // Check environment variable once
    const char* env_algo = std::getenv("VLLM_CUSTOM_ALLREDUCE_ALGO");
    bool force_1stage = false;
    bool force_2stage = false;
    if (env_algo != nullptr) {
      if (std::strcmp(env_algo, "1stage") == 0 ||
          std::strcmp(env_algo, "oneshot") == 0) {
        force_1stage = true;
      } else if (std::strcmp(env_algo, "2stage") == 0 ||
                 std::strcmp(env_algo, "twoshot") == 0) {
        force_2stage = true;
      } else {
        throw std::runtime_error(
            "Invalid VLLM_CUSTOM_ALLREDUCE_ALGO: " + std::string(env_algo) +
            ". Valid values: 1stage, oneshot, 2stage, twoshot");
      }
    }

#define KL(ngpus, name)                                                       \
  name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size);
#define KLPUSH(ngpus)                                                        \
  cross_device_reduce_push<T, ngpus><<<blocks, threads, 0, stream>>>(        \
      ptrs, sg_, self_sg_, src, output, rank_, size, chunk_p);
#define REDUCE_CASE(ngpus)                              \
  case ngpus: {                                         \
    if (force_1stage) {                                 \
      if (chunk_p > 0) {                                \
        KLPUSH(ngpus)                                   \
      } else {                                          \
        KL(ngpus, cross_device_reduce_1stage);          \
      }                                                 \
    } else if (force_2stage) {                          \
      KL(ngpus, cross_device_reduce_2stage);            \
    } else {                                            \
      if (world_size_ == 2) {                           \
        KL(ngpus, cross_device_reduce_1stage);          \
      } else if (fully_connected_) {                    \
        if ((world_size_ <= 4 && bytes < 512 * 1024) || \
            (world_size_ <= 8 && bytes < 256 * 1024)) { \
          KL(ngpus, cross_device_reduce_1stage);        \
        } else {                                        \
          KL(ngpus, cross_device_reduce_2stage);        \
        }                                               \
      }                                                 \
    }                                                   \
    break;                                              \
  }

    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error(
            "custom allreduce only supports num gpus in (2,4,6,8). Actual "
            "num "
            "gpus = " +
            std::to_string(world_size_));
    }
#undef REDUCE_CASE
#undef KL
  }

  // S37 W4: all-reduce + fused add-rmsnorm epilogue on the push kernel.
  // Same buffer/capture contract as allreduce(); additionally requires
  // (host-checked): numel % hidden == 0, hidden / pack <= threads,
  // rows <= kMaxBlocks, chunk_p >= size.  residual is updated in place.
  template <typename T>
  void allreduce_fused(cudaStream_t stream, T* input, const T* src,
                       T* output, T* residual, const T* norm_w, int size,
                       int hidden, float eps, int threads = 1024,
                       int block_limit = defaultBlockLimit, int chunk_p = 0) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "fused allreduce requires input length to be multiple of " +
          std::to_string(d));
    size /= d;  // elements -> packed units (same convention as allreduce)
    const int row_p = hidden / d;  // packed units per hidden row
    const int rows = row_p > 0 ? size / row_p : 0;
    if (row_p <= 0 || row_p * rows != size)
      throw std::runtime_error(
          "fused allreduce: numel must be a multiple of hidden");
    if (row_p > threads)
      throw std::runtime_error(
          "fused allreduce: hidden row exceeds one block (threads=" +
          std::to_string(threads) + ")");
    if (rows > kMaxBlocks)
      throw std::runtime_error("fused allreduce: too many rows (" +
                               std::to_string(rows) + ")");
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " +
                               std::to_string(kMaxBlocks) + ". Got " +
                               std::to_string(block_limit));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input)) +
            " is not registered!");
      ptrs = it->second;
    }

#define KLPUSH_FUSED(ngpus)                                                  \
  cross_device_reduce_push_fused<T, ngpus><<<rows, threads, 0, stream>>>(    \
      ptrs, sg_, self_sg_, src, output, residual, norm_w, rank_, size,       \
      chunk_p, row_p, eps);
    switch (world_size_) {
      case 2:
        KLPUSH_FUSED(2)
        break;
      case 4:
        KLPUSH_FUSED(4)
        break;
      case 6:
        KLPUSH_FUSED(6)
        break;
      case 8:
        KLPUSH_FUSED(8)
        break;
      default:
        throw std::runtime_error(
            "fused allreduce only supports num gpus in (2,4,6,8). Actual "
            "num gpus = " +
            std::to_string(world_size_));
    }
#undef KLPUSH_FUSED
  }

  void allgather(cudaStream_t stream, void* input, void* output, int size_bytes,
                 int threads = 512, int block_limit = defaultBlockLimit);
  template <typename T>
  void mnnvl_lamport_allgather(cudaStream_t stream, T* input, T* output,
                               void* local_buffer, void* multicast_buffer,
                               uint32_t* epochs, int size_bytes,
                               int stage_size_bytes);
  template <typename T>
  void reduce_scatter(cudaStream_t stream, T* input, T* output, int size,
                      int threads = 512, int block_limit = defaultBlockLimit);
  template <typename T>
  void mnnvl_lamport_reduce_scatter(cudaStream_t stream, T* input, T* output,
                                    void* local_buffer, uint32_t* epochs,
                                    int size, int stage_size_bytes);
  template <typename T>
  void mnnvl_multimem_reduce_scatter(cudaStream_t stream,
                                     const T* multicast_input, T* output,
                                     Signal* local_signal,
                                     Signal* multicast_signal, int size,
                                     int block_limit);

  ~CustomAllreduce() {
    for (auto [_, ptr] : ipc_handles_) {
      CUDACHECK(cudaIpcCloseMemHandle(ptr));
    }
  }
};

/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and
 * add a template instantiation:
 * template void vllm::CustomAllreduce::allreduce<half>(cudaStream_t, half *,
 *                                                       half *, int, int, int);
 */
}  // namespace vllm
