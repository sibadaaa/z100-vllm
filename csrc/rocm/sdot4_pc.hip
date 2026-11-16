// S99 WP2: per-channel int8 prefill GEMM for gfx906, with in-op fp8 staging.
//
// Scope (construction §WP2.1): a *new* symbol set, the production
// csrc/rocm/w8a8_sdot4.hip stays untouched.
//
//   out[m,n] = s_a[m] * s_w[n] * sum_k q_a[m,k] * q_w[n,k]
//
//   q_a : per-token int8 activation (produced by ops.scaled_int8_quant)
//   q_w : per-channel int8 weight, produced here from the shipped fp8
//         E4M3 checkpoint + 128x128 block scale:
//             q_w[n,k] = round( f8[n,k] * s_b[n/128, k/128] * 127 / A[n] )
//             s_w[n]   = A[n] / 127 ,  A[n] = max_k |w[n,k]|
//         (HB1 measured every shipped block at amax == 448 exactly, but the
//          block *scale* still varies, so the per-block scale multiply is
//          mandatory -- a scale-free LUT is 15-46% relative error.)
//
// The GEMM inner loop is the S86 champion topology: 128x128 / 8x8 / 2 WG/CU,
// compute unroll 2, spill accepted.  Only the staging and the new symbol are
// new; the staging/reduction/epilogue order is the production one.
//
// Why the unroll/launch-bounds change matters: the production 128x128 arm is
// compiled with the full unroll and no occupancy hint, so it lands at 1 WG/CU
// (22.1 TOPS).  The champion is 2 WG/CU (33.3 TOPS) thanks to
// __launch_bounds__(256, 2) + unroll 2 (vgpr 128 / 7 spill).

#include <torch/all.h>

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <c10/util/Float8_e4m3fn.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <tuple>

// ---------------------------------------------------------------------------
// staging: fp8 E4M3 + 128x128 block scales -> per-channel int8 + s_w
// ---------------------------------------------------------------------------

namespace {

constexpr int kStageThreads = 256;

__device__ __forceinline__ float pc_lut(const __half* lut, uint8_t b) {
  return __half2float(lut[b]);
}

// One workgroup per output channel: row amax, then the quantise pass.
__global__ void stage_pc_kernel(const uint8_t* __restrict__ w,
                                const float* __restrict__ sb,
                                const __half* __restrict__ lut, int64_t N,
                                int64_t K, int64_t KB, int64_t block_n,
                                int64_t block_k, int64_t w_stride,
                                int8_t* __restrict__ q,
                                float* __restrict__ sw) {
  const int64_t row = (int64_t)blockIdx.x;
  if (row >= N) return;
  const int64_t nb = row / block_n;
  // The production fp8 weight is K-contiguous but may carry a padded row
  // stride (the W8A16 tile path supports it); the int8 scratch is dense.
  const uint8_t* wr = w + row * w_stride;
  __shared__ float red[kStageThreads];

  float m = 0.0f;
  for (int64_t k = threadIdx.x; k < K; k += kStageThreads) {
    float v = pc_lut(lut, wr[k]) * sb[nb * KB + k / block_k];
    m = fmaxf(m, fabsf(v));
  }
  red[threadIdx.x] = m;
  __syncthreads();
  for (int s = kStageThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  __shared__ float mult_sh;
  if (threadIdx.x == 0) {
    const float A = red[0] > 0.0f ? red[0] : 1.0f;
    mult_sh = 127.0f / A;
    sw[row] = A / 127.0f;
  }
  __syncthreads();

  const float mult = mult_sh;
  int8_t* qr = q + row * K;
  for (int64_t k = threadIdx.x; k < K; k += kStageThreads) {
    float v = pc_lut(lut, wr[k]) * sb[nb * KB + k / block_k] * mult;
    int iv = (int)lrintf(v);
    iv = iv > 127 ? 127 : (iv < -127 ? -127 : iv);
    qr[k] = (int8_t)iv;
  }
}

// ---------------------------------------------------------------------------
// GEMM: S86 champion topology (128x128 / 8x8 / 2 WG/CU, unroll 2)
// ---------------------------------------------------------------------------

constexpr int kGBk = 32;
constexpr int kGKw = kGBk / 4;  // 8 int32 words per K chunk
constexpr int kGKp = kGKw + 1;  // 9 (padded LDS stride)
constexpr int kGLogKw = 3;

template <int BM, int BN, int TM, int TN>
struct GConf {
  static constexpr int MG = BM / TM;
  static constexpr int NG = BN / TN;
  static constexpr int NT = MG * NG;
  static constexpr int AB = BM * kGKw;
  static constexpr int BB = BN * kGKw;
  static constexpr int AFRAGS = AB / NT;
  static constexpr int BFRAGS = BB / NT;
  static constexpr int FRAGS = AFRAGS + BFRAGS;
};

// Same global -> register staging as production (zero-fill OOB).
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void pc_load(const int8_t* __restrict__ A,
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
    if (m0 + r < M && (kfull || (k0 + (kw + 1) * 4) <= K))
      v = ap[(int64_t)r * K4 + kw];
    dst[u] = v;
  }
#pragma unroll
  for (int u = 0; u < C::BFRAGS; ++u) {
    const int idx = tid + u * C::NT;
    const int r = idx >> kGLogKw;
    const int kw = idx & (kGKw - 1);
    int32_t v = 0;
    if (n0 + r < N && (kfull || (k0 + (kw + 1) * 4) <= K))
      v = wp[(int64_t)r * K4 + kw];
    dst[C::AFRAGS + u] = v;
  }
}

template <int BM, int BN, int TM, int TN, int S>
__device__ __forceinline__ void pc_store(int tid,
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

// Compute with unroll 2: this is what keeps the 64-accumulator tile inside the
// 128-vgpr budget at 2 WG/CU (S86 WP1: 33.3 vs 22.1 TOPS at 1 WG/CU).
template <int BM, int BN, int TM, int TN, int S>
__device__ __forceinline__ void pc_compute(int tx, int ty,
                                           const int32_t (&As)[2][BM * kGKp],
                                           const int32_t (&Bs)[2][BN * kGKp],
                                           int (&acc)[TM][TN]) {
  using C = GConf<BM, BN, TM, TN>;
  const int32_t* __restrict__ as = As[S];
  const int32_t* __restrict__ bs = Bs[S];
  _Pragma("unroll 2")
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

template <int BM, int BN, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN), 2)
    w8a8_sdot4_gemm_pc_kernel(const int8_t* __restrict__ A,
                              const int8_t* __restrict__ W,
                              const float* __restrict__ x_s,
                              const float* __restrict__ w_s,
                              const at::Half* __restrict__ bias,
                              at::Half* __restrict__ C, int M, int N,
                              int K, int x_s_stride, int w_s_stride) {
  using C_ = GConf<BM, BN, TM, TN>;
  constexpr int NT = C_::NT;
  constexpr int FRAGS = C_::FRAGS;

  __shared__ int32_t As[2][BM * kGKp];
  __shared__ int32_t Bs[2][BN * kGKp];

  const int tid = threadIdx.x;
  const int tx = tid % C_::NG;
  const int ty = tid / C_::NG;
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
  pc_load<BM, BN, TM, TN>(A, W, m0, n0, M, N, K, K4, 0, tid, f0);

  int kb = 0;
  for (; kb + 1 < NK; kb += 2) {
    pc_store<BM, BN, TM, TN, 0>(tid, f0, As, Bs);
    __syncthreads();
    pc_load<BM, BN, TM, TN>(A, W, m0, n0, M, N, K, K4, (kb + 1) * kGBk, tid,
                            f1);
    pc_compute<BM, BN, TM, TN, 0>(tx, ty, As, Bs, acc);

    pc_store<BM, BN, TM, TN, 1>(tid, f1, As, Bs);
    __syncthreads();
    pc_load<BM, BN, TM, TN>(A, W, m0, n0, M, N, K, K4, (kb + 2) * kGBk, tid,
                            f0);
    pc_compute<BM, BN, TM, TN, 1>(tx, ty, As, Bs, acc);
  }
  if (kb < NK) {
    pc_store<BM, BN, TM, TN, 0>(tid, f0, As, Bs);
    __syncthreads();
    pc_compute<BM, BN, TM, TN, 0>(tx, ty, As, Bs, acc);
  }

  // Epilogue order is the production one: (acc * x_s) * w_s, cast, then bias.
  using half_t = at::Half;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + C_::MG * i;
    if (row >= M) continue;
    const float sa = x_s[row * x_s_stride];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + tx + C_::NG * j;
      if (col >= N) continue;
      const float v =
          static_cast<float>(acc[i][j]) * sa * w_s[col * w_s_stride];
      half_t o = static_cast<half_t>(v);
      if (bias != nullptr) o += bias[col];
      C[(int64_t)row * N + col] = o;
    }
  }
}

template <int BM, int BN, int TM, int TN>
void pc_launch(int M, int N, int K, int x_s_stride, int w_s_stride,
               const int8_t* A, const int8_t* W, const float* x_s,
               const float* w_s, const void* bias, void* C,
               hipStream_t stream) {
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block((BM / TM) * (BN / TN));
  hipLaunchKernelGGL(
      (w8a8_sdot4_gemm_pc_kernel<BM, BN, TM, TN>), dim3(grid), dim3(block), 0,
      stream, A, W, x_s, w_s,
      reinterpret_cast<const at::Half*>(bias),
      reinterpret_cast<at::Half*>(C), M, N, K, x_s_stride, w_s_stride);
}

void stage_launch(const uint8_t* w, const float* sb, const __half* lut,
                  int64_t N, int64_t K, int64_t KB, int64_t block_n,
                  int64_t block_k, int64_t w_stride, int8_t* q, float* sw,
                  hipStream_t stream) {
  hipLaunchKernelGGL(stage_pc_kernel, dim3((unsigned)N),
                     dim3(kStageThreads), 0, stream, w, sb, lut, N, K, KB,
                     block_n, block_k, w_stride, q, sw);
}

int s_stride(const torch::Tensor& t) { return t.numel() == 1 ? 0 : 1; }

void check_pc_gemm(const torch::Tensor& xq, const torch::Tensor& w,
                   const torch::Tensor& x_s, const torch::Tensor& w_s,
                   const std::optional<torch::Tensor>& bias,
                   at::ScalarType out_dtype, int& x_s_stride,
                   int& w_s_stride) {
  TORCH_CHECK(xq.is_cuda() && w.is_cuda() && x_s.is_cuda() && w_s.is_cuda(),
              "w8a8_sdot4_gemm_pc: all tensors must be on the GPU");
  TORCH_CHECK(xq.scalar_type() == at::kChar && w.scalar_type() == at::kChar,
              "w8a8_sdot4_gemm_pc: xq/w must be int8");
  TORCH_CHECK(x_s.scalar_type() == at::kFloat &&
                  w_s.scalar_type() == at::kFloat,
              "w8a8_sdot4_gemm_pc: scales must be float32");
  TORCH_CHECK(xq.is_contiguous() && w.is_contiguous() &&
                  x_s.is_contiguous() && w_s.is_contiguous(),
              "w8a8_sdot4_gemm_pc: tensors must be contiguous");
  TORCH_CHECK(xq.dim() == 2 && w.dim() == 2,
              "w8a8_sdot4_gemm_pc: xq and w must be 2-D");
  TORCH_CHECK(w.size(1) == xq.size(1), "w8a8_sdot4_gemm_pc: K mismatch");
  TORCH_CHECK(x_s.numel() == xq.size(0) || x_s.numel() == 1,
              "w8a8_sdot4_gemm_pc: x_s must have M or 1 elements");
  TORCH_CHECK(w_s.numel() == w.size(0) || w_s.numel() == 1,
              "w8a8_sdot4_gemm_pc: w_s must have N or 1 elements");
  TORCH_CHECK(out_dtype == at::kHalf,
              "w8a8_sdot4_gemm_pc: only float16 output is wired");
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() &&
                    bias->numel() == w.size(0) &&
                    bias->scalar_type() == at::kHalf,
                "w8a8_sdot4_gemm_pc: bias must be a contiguous fp16 GPU "
                "tensor of N elements");
  }
  x_s_stride = s_stride(x_s);
  w_s_stride = s_stride(w_s);
}

}  // namespace

// ---------------------------------------------------------------------------
// public entries
// ---------------------------------------------------------------------------

// Stage one layer: fp8 weight + block scale -> per-channel int8 + s_w.
std::tuple<torch::Tensor, torch::Tensor> fp8_stage_int8_pc(
    torch::Tensor w, torch::Tensor w_s, torch::Tensor lut,
    int64_t block_n, int64_t block_k) {
  TORCH_CHECK(w.is_cuda() && w_s.is_cuda() && lut.is_cuda(),
              "fp8_stage_int8_pc: all tensors must be on the GPU");
  TORCH_CHECK(w.dim() == 2 && w.element_size() == 1,
              "fp8_stage_int8_pc: w must be a 2-D 8-bit (fp8) tensor");
  TORCH_CHECK(w.stride(1) == 1 && w.stride(0) >= w.size(1),
              "fp8_stage_int8_pc: w must be K-contiguous (row stride >= K)");
  TORCH_CHECK(w_s.scalar_type() == at::kFloat && w_s.is_contiguous(),
              "fp8_stage_int8_pc: w_s must be contiguous float32");
  TORCH_CHECK(lut.scalar_type() == at::kHalf && lut.numel() == 256,
              "fp8_stage_int8_pc: lut must be a 256-element fp16 tensor");
  TORCH_CHECK(block_n > 0 && block_k > 0,
              "fp8_stage_int8_pc: block sizes must be positive");
  const int64_t N = w.size(0), K = w.size(1);
  const int64_t w_stride = w.stride(0);
  TORCH_CHECK(K % 4 == 0, "fp8_stage_int8_pc: K must be a multiple of 4");
  const int64_t NB = (N + block_n - 1) / block_n;
  const int64_t KB = (K + block_k - 1) / block_k;
  TORCH_CHECK(w_s.numel() == NB * KB,
              "fp8_stage_int8_pc: w_s must be [", NB, ",", KB, "], got ",
              w_s.numel());

  const c10::cuda::CUDAGuard guard(w.device());
  auto q = at::empty({N, K}, w.options().dtype(at::kChar));
  auto sw = at::empty({N}, lut.options().dtype(at::kFloat));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  stage_launch(reinterpret_cast<const uint8_t*>(w.data_ptr()),
               reinterpret_cast<const float*>(w_s.data_ptr()),
               reinterpret_cast<const __half*>(lut.data_ptr()), N, K, KB,
               block_n, block_k, w_stride,
               reinterpret_cast<int8_t*>(q.data_ptr()),
               reinterpret_cast<float*>(sw.data_ptr()), stream);
  return std::make_tuple(q, sw);
}

// Per-channel W8A8 GEMM on already-staged int8 weights.
torch::Tensor w8a8_sdot4_gemm_pc(torch::Tensor xq, torch::Tensor w,
                                 torch::Tensor x_s, torch::Tensor w_s,
                                 std::optional<torch::Tensor> bias,
                                 at::ScalarType out_dtype, int64_t cfg) {
  int x_s_stride = 1, w_s_stride = 1;
  check_pc_gemm(xq, w, x_s, w_s, bias, out_dtype, x_s_stride, w_s_stride);
  const int64_t M = xq.size(0), K = xq.size(1), N = w.size(0);
  TORCH_CHECK(K % 4 == 0, "w8a8_sdot4_gemm_pc: K must be a multiple of 4");
  TORCH_CHECK(M >= 1, "w8a8_sdot4_gemm_pc: M must be >= 1");
  TORCH_CHECK(cfg == 0 || cfg == 4, "w8a8_sdot4_gemm_pc: cfg must be 0 or 4");

  const c10::cuda::CUDAGuard guard(xq.device());
  auto out = at::empty({M, N}, xq.options().dtype(out_dtype));
  const void* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined()) bias_ptr = bias->data_ptr();
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  pc_launch<128, 128, 8, 8>(
      (int)M, (int)N, (int)K, x_s_stride, w_s_stride,
      reinterpret_cast<const int8_t*>(xq.data_ptr()),
      reinterpret_cast<const int8_t*>(w.data_ptr()),
      reinterpret_cast<const float*>(x_s.data_ptr()),
      reinterpret_cast<const float*>(w_s.data_ptr()), bias_ptr,
      out.data_ptr(), stream);
  return out;
}

// Production entry: fused staging + GEMM for one layer's prefill call.
torch::Tensor fp8_emu_mm_int8_pc(torch::Tensor xq, torch::Tensor w,
                                 torch::Tensor w_s, torch::Tensor x_s,
                                 torch::Tensor lut,
                                 std::optional<torch::Tensor> bias,
                                 at::ScalarType out_dtype, int64_t block_n,
                                 int64_t block_k) {
  auto staged = fp8_stage_int8_pc(w, w_s, lut, block_n, block_k);
  return w8a8_sdot4_gemm_pc(xq, std::get<0>(staged), x_s,
                            std::get<1>(staged), bias, out_dtype, 0);
}

// torch.compile / fake-tensor propagation.
std::tuple<torch::Tensor, torch::Tensor> fp8_stage_int8_pc_meta(
    torch::Tensor w, torch::Tensor w_s, torch::Tensor lut,
    int64_t /*block_n*/, int64_t /*block_k*/) {
  (void)w_s;
  return std::make_tuple(w.new_empty({w.size(0), w.size(1)},
                                     w.options().dtype(at::kChar)),
                         lut.new_empty({w.size(0)},
                                       lut.options().dtype(at::kFloat)));
}

torch::Tensor w8a8_sdot4_gemm_pc_meta(torch::Tensor xq, torch::Tensor w,
                                      torch::Tensor x_s, torch::Tensor w_s,
                                      std::optional<torch::Tensor> bias,
                                      at::ScalarType out_dtype,
                                      int64_t /*cfg*/) {
  (void)x_s;
  (void)w_s;
  (void)bias;
  return xq.new_empty({xq.size(0), w.size(0)},
                      xq.options().dtype(out_dtype));
}

torch::Tensor fp8_emu_mm_int8_pc_meta(torch::Tensor xq, torch::Tensor w,
                                      torch::Tensor w_s, torch::Tensor x_s,
                                      torch::Tensor lut,
                                      std::optional<torch::Tensor> bias,
                                      at::ScalarType out_dtype,
                                      int64_t /*block_n*/,
                                      int64_t /*block_k*/) {
  (void)w_s;
  (void)x_s;
  (void)lut;
  (void)bias;
  return xq.new_empty({xq.size(0), w.size(0)},
                      xq.options().dtype(out_dtype));
}
