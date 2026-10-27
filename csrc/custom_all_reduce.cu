#include "torch_utils.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>

#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>

#include "custom_all_reduce.cuh"

// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs,
                      torch::stable::Tensor& rank_data, int64_t rank,
                      bool fully_connected) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > vllm::kMaxCustomCollectiveRanks)
    throw std::invalid_argument("world size > 16 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");

  vllm::Signal* ipc_ptrs[vllm::kMaxCustomCollectiveRanks];
  for (int i = 0; i < world_size; i++) {
    ipc_ptrs[i] = reinterpret_cast<vllm::Signal*>(fake_ipc_ptrs[i]);
  }
  return (fptr_t) new vllm::CustomAllreduce(
      ipc_ptrs, rank_data.mutable_data_ptr(), rank_data.numel(), rank,
      world_size, fully_connected);
}

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice (i.e. slicing the first
 * dimension). Currently, we require this because stride information is not
 * passed into the kernels and we treat input tensors as flat.
 *
 * Examples
 * A = torch.zeros(3, 3, 3)
 * 1. A: OK
 * 2. A[1:]: OK
 * 3. A.permute(2, 0, 1): OK
 * 4. A[1:].permute(2, 0, 1): OK
 * 5. A[None].expand(2, -1, -1, -1): Not OK
 * 6. A[:, 1:, 1:]: Not OK
 */
bool _is_weak_contiguous(torch::stable::Tensor& t) {
  if (t.is_contiguous()) {
    return true;
  }
  int64_t storage_nbytes = 0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_size(t.get(), &storage_nbytes));
  return storage_nbytes - t.storage_offset() * t.element_size() ==
         static_cast<int64_t>(t.numel() * t.element_size());
}

/**
 * Performs an out-of-place allreduce and stores result in out.
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 * copied into _reg_buffer.
 */
#if defined(USE_ROCM)
// S33 WP2 fix: on gfx906 + DTK 26.04, the cudaMemcpyAsync D2D copy into the
// fine-grain (uncached) registered buffer runs on the SDMA engine, and its
// completion does not order against peer-device P2P reads -- service showed
// stale input reads (eager arm nondeterministic, graph-replay arm
// deterministically stale) while the flag handshake itself (kernel stores to
// the same fine-grain allocation) always worked.  Copying with a kernel on
// the compute engine and committing with __threadfence_system() makes the
// input data system-visible before the all-reduce kernel (and its flag
// exchange) can start.  Capturable: replaces cudaMemcpyAsync under CUDA
// graph capture as well.
__global__ void dcu_reg_buffer_copy_kernel(
    const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, int nwords,
    const uint8_t* __restrict__ src_tail, uint8_t* __restrict__ dst_tail,
    int tail) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  // S33 WP2: volatile (glc) stores -- plain stores land in the producer's L2
  // and never reach peer GPUs over PCIe on gfx906 (the flags work for the
  // same reason). __threadfence_system() orders the stores below.
  // plain stores: this write is read back LOCALLY by this rank's push kernel
  for (int w = i; w < nwords; w += stride) dst[w] = src[w];
  if (tail && i < tail) dst_tail[i] = src_tail[i];
  __threadfence_system();
}
#endif

// S33 WP2 debug probe (scaled): per call, after the copy kernel, scan the
// whole registered buffer on-device for "all zeros" (the seq0 failure
// signature) and log it; value dumps are downsampled (first 2 calls, then
// every 256th) to keep logs small.
__global__ void dcu_buf_allzero_scan_kernel(const uint16_t* __restrict__ p,
                                            int n, uint32_t* __restrict__ flag,
                                            uint16_t* __restrict__ first16) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  uint32_t any = 0;
  for (int w = i; w < n; w += stride) {
    if (p[w] != 0) any = 1;
  }
  if (i == 0) {
    *flag = any;
    for (int w = 0; w < 16 && w < n; w++) first16[w] = p[w];
  }
}

static int dcu_ar_dbg_call() {
  static int c = 0;
  return c;  // current call index (bumped by dcu_ar_dbg_call_next)
}

static int dcu_ar_dbg_call_next() {
  static int c = 0;
  return c++;
}

static int dcu_ar_debug_calls() {
  static const int k = []() {
    const char* e = std::getenv("VLLM_DCU_CUSTOM_AR_DEBUG");
    return (e != nullptr) ? std::atoi(e) : 0;
  }();
  return k;
}

namespace {
// host-side fp16 bit-pattern -> float (debug probe only)
float dcu_h2f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t man = h & 0x3ff;
  uint32_t f;
  if (exp == 0) {
    f = sign | (man << 13);
  } else if (exp == 31) {
    f = sign | 0x7f800000u | (man << 13);
  } else {
    f = sign | ((uint32_t)(exp - 15 + 127) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

void dcu_ar_debug_dump(cudaStream_t stream, const void* input,
                       const void* output, int64_t numel, bool is_half,
                       int rank, const char* tag) {
  static int seq_postar = 0;
  static int seq_postcopy = 0;
  static int call_seq = 0;
  const int calls = dcu_ar_debug_calls();
  int& seq = (std::strcmp(tag, "postcopy") == 0) ? seq_postcopy : seq_postar;
  const int my_call = call_seq++;  // one call = one postcopy + one postar
  const bool dump_values = (my_call < 2) || (my_call % 256 == 0);
  if (seq >= calls && !dump_values) return;
  const bool scan = std::strcmp(tag, "postcopy") == 0;
  const int n = (int)std::min<int64_t>(16, numel);
  std::vector<uint16_t> hbuf(
      is_half ? (output ? 2 * (size_t)n : (size_t)n) : 0);
  std::vector<float> fbuf(!is_half ? 2 * (size_t)n : 0);
  std::vector<uint32_t> zflag(1, 0);
  uint32_t* zflag_d = nullptr;
  uint16_t* first16_d = nullptr;
  static uint32_t* zflag_buf = nullptr;
  static uint16_t* first16_buf = nullptr;
  if (scan) {
    // persistent device scratch for the all-zero scan
    if (zflag_buf == nullptr) {
      CUDACHECK(cudaMalloc(&zflag_buf, sizeof(uint32_t)));
      CUDACHECK(cudaMalloc(&first16_buf, 16 * sizeof(uint16_t)));
    }
    zflag_d = zflag_buf;
    first16_d = first16_buf;
    const int ntotal = (int)std::min<int64_t>(numel, 1 << 20);
    dcu_buf_allzero_scan_kernel<<<8, 256, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(input), ntotal, zflag_d, first16_d);
  }
  void* in_d = is_half ? (void*)hbuf.data() : (void*)fbuf.data();
  void* out_d = nullptr;
  if (output) {
    out_d = is_half ? (void*)(hbuf.data() + n) : (void*)(fbuf.data() + n);
  }
  const size_t esz = is_half ? sizeof(uint16_t) : sizeof(float);
  if (dump_values) {
    CUDACHECK(cudaMemcpyAsync(in_d, input, (size_t)n * esz,
                              cudaMemcpyDeviceToHost, stream));
    if (out_d) {
      CUDACHECK(cudaMemcpyAsync(out_d, output, (size_t)n * esz,
                                cudaMemcpyDeviceToHost, stream));
    }
  }
  if (scan) {
    CUDACHECK(cudaMemcpyAsync(zflag.data(), zflag_d, sizeof(uint32_t),
                              cudaMemcpyDeviceToHost, stream));
  }
  CUDACHECK(cudaStreamSynchronize(stream));
  if (scan && zflag[0] == 0) {
    fprintf(stderr,
            "[car-dbg] r%d call%d %s numel=%lld POSTCOPY-ALLZERO\n", rank,
            my_call, tag, (long long)numel);
  }
  if (!dump_values) return;
  std::ostringstream os;
  os << std::setprecision(6);
  os << "[car-dbg] r" << rank << " call" << my_call << " " << tag
     << " numel=" << numel << " in=[";
  for (int i = 0; i < n; i++)
    os << (i ? "," : "") << (is_half ? dcu_h2f(hbuf[i]) : fbuf[i]);
  if (out_d) {
    os << "] out=[";
    for (int i = 0; i < n; i++)
      os << (i ? "," : "")
         << (is_half ? dcu_h2f(hbuf[n + i]) : fbuf[n + i]);
  }
  os << "]";
  fprintf(stderr, "%s\n", os.str().c_str());
  seq++;
}
}  // namespace

void all_reduce(fptr_t _fa, torch::stable::Tensor& inp,
                torch::stable::Tensor& out, fptr_t _reg_buffer,
                int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK((inp.scalar_type()) == (out.scalar_type()));
  STD_TORCH_CHECK((inp.numel()) == (out.numel()));
  STD_TORCH_CHECK(_is_weak_contiguous(out));
  STD_TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  // S33 WP2 push mode: oneshot/1stage on ROCm uses the push kernel, which
  // reads the activation directly (src) and writes each peer's fine-grain
  // buffer region for this rank -- no staging copy at all.
  static const bool dcu_push_mode = []() {
#if defined(USE_ROCM)
    const char* e = std::getenv("VLLM_CUSTOM_ALLREDUCE_ALGO");
    return e != nullptr && (std::strcmp(e, "oneshot") == 0 ||
                            std::strcmp(e, "1stage") == 0);
#else
    return false;
#endif
  }();
  int64_t chunk_bytes = 0;
  if (dcu_push_mode && reg_buffer && fa->world_size_ > 0) {
    chunk_bytes = reg_buffer_sz_bytes / fa->world_size_;
  }
  if (reg_buffer) {
    STD_TORCH_CHECK((input_size) <= (chunk_bytes > 0 ? chunk_bytes
                                                     : reg_buffer_sz_bytes));
  } else {
    reg_buffer = inp.mutable_data_ptr();
  }
  switch (out.scalar_type()) {
    case torch::headeronly::ScalarType::Float: {
      const int c = (int)(chunk_bytes / 16);
      fa->allreduce<float>(stream, reinterpret_cast<float*>(reg_buffer),
                           reinterpret_cast<const float*>(inp.const_data_ptr()),
                           reinterpret_cast<float*>(out.mutable_data_ptr()),
                           out.numel(), 512, vllm::defaultBlockLimit, c);
      break;
    }
    case torch::headeronly::ScalarType::Half: {
      const int c = (int)(chunk_bytes / 16);
      fa->allreduce<half>(stream, reinterpret_cast<half*>(reg_buffer),
                          reinterpret_cast<const half*>(inp.const_data_ptr()),
                          reinterpret_cast<half*>(out.mutable_data_ptr()),
                          out.numel(), 512, vllm::defaultBlockLimit, c);
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case torch::headeronly::ScalarType::BFloat16: {
      const int c = (int)(chunk_bytes / 16);
      fa->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<const nv_bfloat16*>(inp.const_data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.mutable_data_ptr()), out.numel(),
          512, vllm::defaultBlockLimit, c);
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "custom allreduce only supports float32, float16 and bfloat16");
  }
  if (dcu_ar_debug_calls() > 0) {
    cudaStreamCaptureStatus cap_status;
    CUDACHECK(cudaStreamIsCapturing(stream, &cap_status));
    if (cap_status == cudaStreamCaptureStatusNone) {
      dcu_ar_debug_dump(
          stream, inp.const_data_ptr(), out.mutable_data_ptr(), inp.numel(),
          out.scalar_type() == torch::headeronly::ScalarType::Half,
          fa->rank_, "postar");
    }
  }
}

void dispose(fptr_t _fa) {
  delete reinterpret_cast<vllm::CustomAllreduce*>(_fa);
}

int64_t meta_size() { return sizeof(vllm::Signal); }

void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fake_ipc_ptrs.size() == fa->world_size_);
  void* ipc_ptrs[vllm::kMaxCustomCollectiveRanks];
  for (int i = 0; i < fake_ipc_ptrs.size(); i++) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  fa->register_buffer(ipc_ptrs);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (int i = 0; i < handles.size(); i++) {
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  }
  bytes.reserve(handles.size());
  fa->register_graph_buffers(bytes, offsets);
}

namespace {
#if defined(USE_ROCM)
// Allocation flag for the custom all-reduce IPC buffers.
//
// VLLM_DCU_CUSTOM_AR_ALLOC selects the hipExtMallocWithFlags flag; unset (or
// an empty value) keeps the upstream behavior bit for bit. gfx906 on DTK 26.04
// needs "finegrained": hipDeviceMallocUncached (0x3) always returns
// hipErrorOutOfMemory there. Read once per process.
unsigned int dcu_custom_ar_alloc_flags() {
  static const unsigned int cached = []() -> unsigned int {
    const char* val = std::getenv("VLLM_DCU_CUSTOM_AR_ALLOC");
    if (val == nullptr || val[0] == '\0' || std::strcmp(val, "uncached") == 0) {
      return hipDeviceMallocUncached;
    }
    if (std::strcmp(val, "finegrained") == 0) {
      return hipDeviceMallocFinegrained;
    }
    if (std::strcmp(val, "default") == 0) {
      return hipDeviceMallocDefault;
    }
    STD_TORCH_CHECK(false,
                    "Invalid VLLM_DCU_CUSTOM_AR_ALLOC=", val,
                    ". Valid values: uncached (default), finegrained, default");
    return hipDeviceMallocUncached;  // unreachable
  }();
  return cached;
}
#endif
}  // namespace

std::tuple<fptr_t, torch::stable::Tensor> allocate_shared_buffer_and_handle(
    int64_t size) {
  int device_index;
  STD_CUDA_CHECK(cudaGetDevice(&device_index));
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  void* buffer;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  const cudaStream_t stream = get_current_cuda_stream(device_index);
  STD_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Allocate buffer
#if defined(USE_ROCM)
  // data buffers default to "uncached" (upstream MI200 behavior);
  // VLLM_DCU_CUSTOM_AR_ALLOC=finegrained is required on gfx906/DTK 26.04.
  STD_CUDA_CHECK(hipExtMallocWithFlags((void**)&buffer, size,
                                       dcu_custom_ar_alloc_flags()));
#else
  STD_CUDA_CHECK(cudaMalloc((void**)&buffer, size));
#endif
  STD_CUDA_CHECK(cudaMemsetAsync(buffer, 0, size, stream));
  STD_CUDA_CHECK(cudaStreamSynchronize(stream));
  STD_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Create IPC memhandle for the allocated buffer.
  // Will use it in open_mem_handle.
  auto handle = torch::stable::empty(
      {static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))},
      torch::headeronly::ScalarType::Byte, std::nullopt,
      torch::stable::Device(torch::stable::DeviceType::CPU));
  STD_CUDA_CHECK(cudaIpcGetMemHandle(
      (cudaIpcMemHandle_t*)handle.mutable_data_ptr(), buffer));

  return std::make_tuple(reinterpret_cast<fptr_t>(buffer), handle);
}

fptr_t open_mem_handle(torch::stable::Tensor& mem_handle) {
  void* ipc_ptr;
  STD_CUDA_CHECK(cudaIpcOpenMemHandle(
      (void**)&ipc_ptr,
      *((const cudaIpcMemHandle_t*)mem_handle.const_data_ptr()),
      cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<fptr_t>(ipc_ptr);
}

void free_shared_buffer(fptr_t buffer) {
  STD_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buffer)));
}
