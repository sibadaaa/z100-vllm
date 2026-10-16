// SPDX-License-Identifier: Apache-2.0
// S20 standalone HIP probe: gfx906 FP8-emu decode-path roofline + prototypes.
//
// Why standalone: the _rocm_C rebuild is the expensive step in this project, so
// kernel-structure experiments are done here (hipcc, seconds) and only the
// winner is ported into csrc/rocm/fp8_emu_gemm.cu.
//
// Modes:
//   peak-dot2   pure v_dot2_f32_f16 issue rate (ILP 1/2/4/8)
//   peak-lds    LDS u16 table gather + LDS 128-bit read rate
//   peak-gmem   streaming global read bandwidth
//   k3          K-chunked stream prototype (A staged in LDS per K chunk)
//   k1          unchunked stream (whole A in LDS) at M=8 -- LDS-gated reference
//   check       numerics vs an fp64 reference
//
// Build (inside the container):
//   hipcc -O3 --offload-arch=gfx906 -o /tmp/s20_probe s20_probe.cu
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_fp16.h>

#define HIP_CHECK(x)                                                     \
  do {                                                                   \
    hipError_t e_ = (x);                                                 \
    if (e_ != hipSuccess) {                                              \
      printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, \
             __LINE__);                                                  \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

using u16x2_t = unsigned short __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float dot2(uint32_t w2, uint32_t x2, float acc) {
  const __half2 w = *reinterpret_cast<const __half2*>(&w2);
  const __half2 x = *reinterpret_cast<const __half2*>(&x2);
  return __builtin_amdgcn_fdot2(w, x, acc, /*clamp=*/false);
}

__device__ __forceinline__ uint16_t lut_u16(const __half* __restrict__ lut,
                                            uint32_t code) {
  return *reinterpret_cast<const uint16_t*>(lut + (code & 0xFFu));
}

__device__ __forceinline__ uint32_t lut_pair(const __half* __restrict__ lut,
                                             uint32_t two) {
  return (uint32_t)lut_u16(lut, two) |
         ((uint32_t)lut_u16(lut, two >> 8) << 16);
}

__device__ __forceinline__ void widen4(const __half* __restrict__ lut,
                                       uint32_t four, uint32_t& lo,
                                       uint32_t& hi) {
  lo = lut_pair(lut, four);
  hi = lut_pair(lut, four >> 16);
}

__device__ __forceinline__ void lut_stage(__half* dst, const __half* src,
                                          int nt) {
  for (int i = threadIdx.x; i < 256; i += nt) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// peaks
// ---------------------------------------------------------------------------
template <int ILP>
__global__ void k_peak_dot2(uint32_t* sink, int iters) {
  float acc[ILP];
  uint32_t w[ILP], x[ILP];
#pragma unroll
  for (int i = 0; i < ILP; ++i) {
    acc[i] = 0.f;
    w[i] = threadIdx.x * 2654435761u + 12345u + i * 7919u;
    x[i] = threadIdx.x * 40503u + 7u + i * 104729u;
  }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < ILP; ++i) acc[i] = dot2(w[i], x[i], acc[i]);
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < ILP; ++i) s += acc[i];
  if (s == 12345.678f) sink[0] = 1;
}

// mixed inner-loop: 16 LUT gathers + 16 LDS128 reads + 128 dot2 per lane
// (the shipped stream v1 shape at M=8), independent per iteration.
template <int ILP>
__global__ void k_peak_mix(uint32_t* sink, int iters) {
  __shared__ __half sl[256];
  __shared__ uint32_t pad[4096];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) sl[i] = __float2half(0.5f);
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) pad[i] = i * 3 + 1;
  __syncthreads();
  const int lane = threadIdx.x & 63;
  float acc[8];
#pragma unroll
  for (int m = 0; m < 8; ++m) acc[m] = 0.f;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      uint32_t p[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint32_t c0 = (uint32_t)(lane * 7 + q * 13 + it);
        p[q] = lut_pair(sl, c0);
      }
      const uint4* ap = reinterpret_cast<const uint4*>(pad) + (lane * 2);
      const uint4 xa = ap[0];
      const uint4 xb = ap[1 + (m & 1)];
      float loc = 0.f;
      loc = dot2(p[0], xa.x, loc);
      loc = dot2(p[1], xa.y, loc);
      loc = dot2(p[2], xa.z, loc);
      loc = dot2(p[3], xa.w, loc);
      loc = dot2(p[4], xb.x, loc);
      loc = dot2(p[5], xb.y, loc);
      loc = dot2(p[6], xb.z, loc);
      loc = dot2(p[7], xb.w, loc);
      acc[m] += loc;
    }
  }
  float s = 0.f;
#pragma unroll
  for (int m = 0; m < 8; ++m) s += acc[m];
  if (s == 12345.678f) sink[0] = 1;
}

__global__ void k_peak_lds_u16(const __half* __restrict__ lut_src,
                               uint32_t* sink, int iters) {
  __shared__ __half sl[256];
  lut_stage(sl, lut_src, blockDim.x);
  __syncthreads();
  const int lane = threadIdx.x & 63;
  uint32_t idx[8];
#pragma unroll
  for (int q = 0; q < 8; ++q) idx[q] = (uint32_t)(lane * 37 + q * 29);
  uint32_t s = 0;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < 8; ++u) s += lut_u16(sl, idx[u]);
  }
  if (s == 12345u) sink[0] = s;
}

__global__ void k_peak_lds128(uint32_t* sink, int iters) {
  __shared__ uint4 buf[512];
  for (int i = threadIdx.x; i < 2048; i += blockDim.x)
    reinterpret_cast<uint32_t*>(buf)[i] = i;
  __syncthreads();
  uint32_t s = 0;
  const int base = (threadIdx.x * 4) & 511;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      uint4 v = buf[(base + u * 64 + threadIdx.x) & 511];
      s += v.x + v.y + v.z + v.w;
    }
  }
  if (s == 12345u) sink[0] = s;
}

__global__ void k_peak_gmem(const uint4* __restrict__ src, uint32_t* sink,
                            int64_t n4) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  uint32_t s = 0;
  for (; i < n4; i += stride) {
    uint4 v = src[i];
    s += v.x + v.y + v.z + v.w;
  }
  if (s == 12345u) sink[0] = s;
}

// ---------------------------------------------------------------------------
// prototype: K-chunked stream ("k3")
// ---------------------------------------------------------------------------
// Block owns NT/64 rows of W (one row per warp).  A[:, c*KC:(c+1)*KC] is widened
// once per block into LDS (fp16) and reused by every warp; K is walked in KC
// steps so the staged A slice is bounded (M*KC*2 bytes) instead of M*K*2.
//
// Lane->data map is identical to the shipped stream v1 kernel: lane L owns the
// 16 fp8 elements [16L, 16L+16) of the chunk, i.e. exactly one block_k (128))
// window, so the folded a_s*w_s scale is one FMA per (lane, m, chunk) and the
// per-lane accumulation order matches v1.
//
// PREF W chunks are kept in flight in registers (the chunk loop carries a
// __syncthreads, so the loads must be issued before it).
template <int M, int KC, int PREF>
__global__ void __launch_bounds__(1024) k3_kernel(const uint8_t* __restrict__ a,
                          const uint8_t* __restrict__ w,
                          const float* __restrict__ a_s,
                          const float* __restrict__ w_s,
                          const __half* __restrict__ lut, __half* __restrict__ out,
                          const int N, const int K, const int nkb,
                          const int block_n, const int nchunk) {
  constexpr int NT = 1024;             // threads per block (16 warps = 16 rows)
  constexpr int KCW4 = KC / 4;         // fp8 dwords per chunk
  constexpr int KCH = KC / 2;          // fp16 values per chunk
  constexpr int KKB = KC / 128;        // block_k windows per chunk

  extern __shared__ uint32_t dyn[];
  __half* sA[2] = {reinterpret_cast<__half*>(dyn),
                   reinterpret_cast<__half*>(dyn + (size_t)M * KCH)};
  float* sscale = reinterpret_cast<float*>(dyn + (size_t)2 * M * KCH);
  __shared__ __half sl[256];

  const int t = threadIdx.x;
  const int lane = t & 63;
  const int warp = t >> 6;
  const int row = blockIdx.x * (NT / 64) + warp;
  const bool rowok = row < N;

  lut_stage(sl, lut, NT);
  __syncthreads();

  const int nb = (rowok ? row : 0) / block_n;
  const uint4* __restrict__ wp =
      reinterpret_cast<const uint4*>(w + (int64_t)(rowok ? row : 0) * K);

  // stage chunk c into buffer buf
  auto stage = [&](int c, int buf) {
    uint32_t* sA32 = reinterpret_cast<uint32_t*>(sA[buf]);
    const uint32_t* a32 = reinterpret_cast<const uint32_t*>(a) +
                          (int64_t)c * KCW4;
    for (int i = t; i < M * KCW4; i += NT) {
      const int m = i / KCW4;
      const int j = i - m * KCW4;
      uint32_t lo, hi;
      widen4(sl, a32[(int64_t)m * (K / 4) + j], lo, hi);
      sA32[m * KCH + 2 * j] = lo;
      sA32[m * KCH + 2 * j + 1] = hi;
    }
    for (int i = t; i < M * KKB; i += NT) {
      const int m = i / KKB;
      const int kl = i - m * KKB;
      const int kb = c * KKB + kl;
      sscale[(buf ? M * KKB : 0) + m * KKB + kl] =
          a_s[(int64_t)m * nkb + kb] * w_s[(int64_t)nb * nkb + kb];
    }
  };

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  if (nchunk > 0) stage(0, 0);
  __syncthreads();

  uint4 wb[PREF];
#pragma unroll
  for (int q = 0; q < PREF; ++q) {
    wb[q] = make_uint4(0, 0, 0, 0);
    if (rowok && q < nchunk) wb[q] = wp[q * (KC / 16) + lane];
  }

  const int kb_off = lane >> 3;  // 16 elements per lane -> one 128-block
  for (int c = 0; c < nchunk; ++c) {
    const uint4 wv = wb[0];
#pragma unroll
    for (int q = 0; q + 1 < PREF; ++q) wb[q] = wb[q + 1];
    if (c + PREF < nchunk) wb[PREF - 1] = wp[(c + PREF) * (KC / 16) + lane];

    if (c + 1 < nchunk) stage(c + 1, (c + 1) & 1);
    __syncthreads();

    uint32_t p[8];
    widen4(sl, wv.x, p[0], p[1]);
    widen4(sl, wv.y, p[2], p[3]);
    widen4(sl, wv.z, p[4], p[5]);
    widen4(sl, wv.w, p[6], p[7]);

    const uint32_t* sp = reinterpret_cast<const uint32_t*>(sA[c & 1]);
    const float* sc = sscale + ((c & 1) ? M * KKB : 0);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      {
        const uint4* ap =
            reinterpret_cast<const uint4*>(sp + m * KCH) + lane * 2;
        const uint4 xa = ap[0];
        const uint4 xb = ap[1];
        float loc = 0.f;
        loc = dot2(p[0], xa.x, loc);
        loc = dot2(p[1], xa.y, loc);
        loc = dot2(p[2], xa.z, loc);
        loc = dot2(p[3], xa.w, loc);
        loc = dot2(p[4], xb.x, loc);
        loc = dot2(p[5], xb.y, loc);
        loc = dot2(p[6], xb.z, loc);
        loc = dot2(p[7], xb.w, loc);
        acc[m] += loc * sc[m * KKB + kb_off];
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < M; ++m) {
    float v = acc[m];
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off);
    if (lane == 0 && rowok) out[(int64_t)m * N + row] = __float2half(v);
  }
}


// ---------------------------------------------------------------------------
// ablations: where does the stream kernel lose its time?
// ---------------------------------------------------------------------------
// MODE 0: W stream (global -> widen -> 64 dot2 with a register A operand)
// MODE 1: + 16 LDS128 A reads per chunk from a static staged tile
// MODE 2: + per-chunk A staging from global with barriers (== k3 shape)
template <int MODE, int KC, int PREF, int MR = 8, int RW = 1>
__global__ void __launch_bounds__(512) k_abl(const uint8_t* __restrict__ a,
                      const uint8_t* __restrict__ w,
                      const __half* __restrict__ lut, __half* __restrict__ out,
                      const int N, const int K) {
  constexpr int NT = 512;
  constexpr int KCH = KC / 2;  // fp16 dwords per row per chunk
  extern __shared__ uint32_t dyn[];
  __half* sA = reinterpret_cast<__half*>(dyn);
  __shared__ __half sl[256];

  const int t = threadIdx.x;
  const int lane = t & 63;
  const int warp = t >> 6;
  const int rows_pb = (NT / 64) * RW;
  const int row0 = blockIdx.x * rows_pb + warp * RW;
  const int nchunk = K / KC;

  lut_stage(sl, lut, NT);
  if (MODE >= 2) {
    const uint32_t* a32 = reinterpret_cast<const uint32_t*>(a);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int i = t + q * NT;
      uint32_t lo, hi;
      widen4(sl, a32[i], lo, hi);
      reinterpret_cast<uint32_t*>(sA)[2 * i] = lo;
      reinterpret_cast<uint32_t*>(sA)[2 * i + 1] = hi;
    }
  }
  __syncthreads();

  const uint4* __restrict__ wp =
      reinterpret_cast<const uint4*>(w + (int64_t)(row0 < N ? row0 : 0) * K);
  uint4 wb[RW][PREF];
#pragma unroll
  for (int r = 0; r < RW; ++r)
#pragma unroll
    for (int q = 0; q < PREF; ++q) {
      wb[r][q] = make_uint4(0, 0, 0, 0);
      if (row0 + r < N && q < nchunk)
        wb[r][q] = wp[r * (K / 16) + q * (KC / 16) + lane];
    }

  float acc[RW][MR];
#pragma unroll
  for (int r = 0; r < RW; ++r)
#pragma unroll
    for (int m = 0; m < MR; ++m) acc[r][m] = 0.f;

  for (int c = 0; c < nchunk; ++c) {
    uint32_t p[RW][8];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
      const uint4 wv = wb[r][0];
#pragma unroll
      for (int q = 0; q + 1 < PREF; ++q) wb[r][q] = wb[r][q + 1];
      if (c + PREF < nchunk)
        wb[r][PREF - 1] = wp[r * (K / 16) + (c + PREF) * (KC / 16) + lane];
      widen4(sl, wv.x, p[r][0], p[r][1]);
      widen4(sl, wv.y, p[r][2], p[r][3]);
      widen4(sl, wv.z, p[r][4], p[r][5]);
      widen4(sl, wv.w, p[r][6], p[r][7]);
    }

    if (MODE >= 2) {
      if (c + 1 < nchunk) {
        const uint32_t* a32 =
            reinterpret_cast<const uint32_t*>(a) + (int64_t)(c + 1) * (KC / 4);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int i = t + q * NT;
          const int m = i / (KC / 4);
          const int j = i - m * (KC / 4);
          uint32_t lo, hi;
          widen4(sl, a32[(int64_t)m * (K / 4) + j], lo, hi);
          reinterpret_cast<uint32_t*>(sA)[m * KCH + 2 * j] = lo;
          reinterpret_cast<uint32_t*>(sA)[m * KCH + 2 * j + 1] = hi;
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < MR; ++m) {
      uint4 xa, xb;
      if (MODE == 3) {
        // rotating LDS base: real per-chunk A traffic, no staging, no barriers
        const uint4* ap = reinterpret_cast<const uint4*>(sA + (int64_t)m * KCH * 2) +
                          lane * 2 + ((c & 3) << 4);
        xa = ap[0];
        xb = ap[1];
      } else if (MODE >= 1) {
        const uint4* ap =
            reinterpret_cast<const uint4*>(sA + (int64_t)m * KCH * 2) + lane * 2;
        xa = ap[0];
        xb = ap[1];
      } else {
        xa = make_uint4(0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u);
        xb = xa;
      }
#pragma unroll
      for (int r = 0; r < RW; ++r) {
        float loc = 0.f;
        loc = dot2(p[r][0], xa.x, loc);
        loc = dot2(p[r][1], xa.y, loc);
        loc = dot2(p[r][2], xa.z, loc);
        loc = dot2(p[r][3], xa.w, loc);
        loc = dot2(p[r][4], xb.x, loc);
        loc = dot2(p[r][5], xb.y, loc);
        loc = dot2(p[r][6], xb.z, loc);
        loc = dot2(p[r][7], xb.w, loc);
        acc[r][m] += loc;
      }
    }
    if (MODE >= 2) __syncthreads();
  }
  float s = 0.f;
#pragma unroll
  for (int r = 0; r < RW; ++r)
#pragma unroll
    for (int m = 0; m < MR; ++m) s += acc[r][m];
  if (lane == 0 && row0 < N) out[row0] = __float2half(s);
}


// ---------------------------------------------------------------------------
// k6: K-chunked stream, A chunks held in REGISTERS per chunk and reused across
// RW rows of W.  This is the fix for the 16:1 A-LDS traffic of the naive
// structure: per chunk a lane needs MR*(EL*2) bytes of A for EL bytes of W per
// row, so the LDS A traffic is 16/RW : 1 instead of 16:1.
// ---------------------------------------------------------------------------
template <int MR, int EL, int RW, int KC, int MODE = 2>
__global__ void __launch_bounds__(512) k6_kernel(const uint8_t* __restrict__ a,
                          const uint8_t* __restrict__ w,
                          const float* __restrict__ a_s,
                          const float* __restrict__ w_s,
                          const __half* __restrict__ lut,
                          __half* __restrict__ out, const int N, const int K,
                          const int nkb, const int block_n) {
  constexpr int NT = 512;
  constexpr int NROW = NT / 64;              // warps per block
  constexpr int KD = KC / 4;                 // fp8 dwords per chunk
  constexpr int KCD = KC / 2;                // fp16 dwords per row per chunk
  constexpr int ELD = EL / 4;                // fp8 dwords per lane per chunk
  constexpr int ELH = EL / 2;                // fp16 dwords per lane per chunk
  constexpr int KKB = KC / 128;              // block_k windows per chunk
  constexpr int KBLK = EL / 128;             // >0 only when EL >= 128

  extern __shared__ uint32_t dyn[];
  __half* sA[2] = {reinterpret_cast<__half*>(dyn),
                   reinterpret_cast<__half*>(dyn + (size_t)MR * KCD)};
  float* sscale = reinterpret_cast<float*>(dyn + (size_t)2 * MR * KCD);
  __shared__ __half sl[256];

  const int t = threadIdx.x;
  const int lane = t & 63;
  const int warp = t >> 6;
  const int rows_pb = NROW * RW;
  const int row0 = blockIdx.x * rows_pb + warp * RW;
  const int nchunk = K / KC;
  const int kb_lane = (EL * 64 <= 128) ? 0 : 0;
  (void)kb_lane;

  lut_stage(sl, lut, NT);

  auto stage = [&](int c, int buf) {
    uint32_t* sA32 = reinterpret_cast<uint32_t*>(sA[buf]);
    const uint32_t* a32 =
        reinterpret_cast<const uint32_t*>(a) + (int64_t)c * KD;
    // MR*KD fp8 dwords -> MR*KCD fp16 dwords
    for (int i = t; i < MR * KD; i += NT) {
      const int m = i / KD;
      const int j = i - m * KD;
      uint32_t lo, hi;
      widen4(sl, a32[(int64_t)m * (K / 4) + j], lo, hi);
      sA32[m * KCD + 2 * j] = lo;
      sA32[m * KCD + 2 * j + 1] = hi;
    }
    const int nb = (row0 < N ? row0 : 0) / block_n;
    for (int i = t; i < MR * KKB; i += NT) {
      const int m = i / KKB;
      const int kl = i - m * KKB;
      const int kb = c * KKB + kl;
      sscale[(buf ? MR * KKB : 0) + m * KKB + kl] =
          a_s[(int64_t)m * nkb + kb] * w_s[(int64_t)nb * nkb + kb];
    }
  };

  float acc[RW][MR];
#pragma unroll
  for (int r = 0; r < RW; ++r)
#pragma unroll
    for (int m = 0; m < MR; ++m) acc[r][m] = 0.f;

  if (nchunk > 0) stage(0, 0);
  __syncthreads();

  const int kblk = (EL * 64) >> 7;  // block_k index of this lane's window
  const uint4* __restrict__ wp =
      reinterpret_cast<const uint4*>(w + (int64_t)(row0 < N ? row0 : 0) * K);

  for (int c = 0; c < nchunk; ++c) {
    // A slice for this chunk -> registers (MR x 1 uint4)
    uint4 areg[MR];
    {
      const uint32_t* sp = reinterpret_cast<const uint32_t*>(sA[c & 1]) + lane * ELH;
#pragma unroll
      for (int m = 0; m < MR; ++m)
        areg[m] = reinterpret_cast<const uint4*>(sp + m * KCD)[0];
    }
    const float* sc = sscale + ((c & 1) ? MR * KKB : 0) + kblk;

    if (MODE >= 2 && c + 1 < nchunk) stage(c + 1, (c + 1) & 1);
    if (MODE >= 2) __syncthreads();

    // W for RW rows: EL bytes per lane per row = ELD dwords
    uint32_t p[RW][ELH];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
      const uint32_t* wrow =
          reinterpret_cast<const uint32_t*>(w + (int64_t)(row0 + r < N ? row0 + r : 0) * K) +
          (int64_t)c * KD + lane * ELD;
#pragma unroll
      for (int q = 0; q < ELD; ++q) {
        uint32_t lo, hi;
        widen4(sl, wrow[q], lo, hi);
        p[r][2 * q] = lo;
        p[r][2 * q + 1] = hi;
      }
    }

#pragma unroll
    for (int m = 0; m < MR; ++m) {
      const uint4 xa = areg[m];
#pragma unroll
      for (int r = 0; r < RW; ++r) {
        float loc = 0.f;
        if (ELH >= 1) loc = dot2(p[r][0], xa.x, loc);
        if (ELH >= 2) loc = dot2(p[r][1], xa.y, loc);
        if (ELH >= 3) loc = dot2(p[r][2], xa.z, loc);
        if (ELH >= 4) loc = dot2(p[r][3], xa.w, loc);
        acc[r][m] += loc * sc[m * KKB];
      }
    }
    if (MODE >= 2) __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < RW; ++r) {
#pragma unroll
    for (int m = 0; m < MR; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off);
      if (lane == 0 && row0 + r < N) out[(int64_t)m * N + row0 + r] = __float2half(v);
    }
  }
}


// ---------------------------------------------------------------------------
// k8: clean rewrite.  EL = 16 fp8 elements per lane per chunk (KC = EL*64),
// A slice for the chunk read out of LDS into registers and reused across RW
// rows of W; W prefetched PREF chunks ahead; A staging global reads prefetched
// one chunk ahead.
// ---------------------------------------------------------------------------
template <int MR, int RW, int KC, int PREF, int NT = 512, int MODE = 7,
          int EL = 16>
__global__ void __launch_bounds__(NT) k8_kernel(
    const uint8_t* __restrict__ a, const uint8_t* __restrict__ w,
    const float* __restrict__ a_s, const float* __restrict__ w_s,
    const __half* __restrict__ lut, __half* __restrict__ out, const int N,
    const int K, const int nkb, const int block_n) {
  constexpr int NROW = NT / 64;
  constexpr int KD = KC / 4;    // fp8 dwords per chunk
  constexpr int KCD = KC / 2;   // fp16 dwords per A row per chunk
  constexpr int ELD = EL / 4;   // fp8 dwords per lane per row per chunk
  constexpr int ELH = EL / 2;   // fp16 dwords per lane per chunk
  constexpr int NREG = ELH / 4;  // uint4 A registers per row
  constexpr int KKB = KC / 128;
  constexpr int NST = MR * KD / NT;
  static_assert(NST * NT == MR * KD, "staging split");
  static_assert(EL * 64 == KC, "lane coverage");

  extern __shared__ uint32_t dyn[];
  __half* sA[2] = {reinterpret_cast<__half*>(dyn),
                   reinterpret_cast<__half*>(dyn + (size_t)MR * KCD)};
  float* sscale = reinterpret_cast<float*>(dyn + (size_t)2 * MR * KCD);
  __shared__ __half sl[256];

  const int t = threadIdx.x;
  const int lane = t & 63;
  const int warp = t >> 6;
  const int rows_pb = NROW * RW;
  const int row0 = blockIdx.x * rows_pb + warp * RW;
  const int nchunk = K / KC;
  const int nb = (row0 < N ? row0 : 0) / block_n;
  const uint32_t* __restrict__ a32 = reinterpret_cast<const uint32_t*>(a);
  const uint32_t* __restrict__ wp =
      reinterpret_cast<const uint32_t*>(w + (int64_t)(row0 < N ? row0 : 0) * K);
  const int lane_d = lane * ELD;

  for (int i = t; i < 256; i += NT) sl[i] = lut[i];

  uint32_t ast[NST];
  auto a_load = [&](int c) {
#pragma unroll
    for (int q = 0; q < NST; ++q) {
      const int i = t + q * NT;
      const int m = i / KD;
      const int j = i - m * KD;
      ast[q] = (MODE & 2) ? a32[(int64_t)m * (K / 4) + (int64_t)c * KD + j] : 0u;
    }
  };
  auto a_store = [&](int buf) {
    uint32_t* sA32 = reinterpret_cast<uint32_t*>(sA[buf]);
#pragma unroll
    for (int q = 0; q < NST; ++q) {
      const int i = t + q * NT;
      const int m = i / KD;
      const int j = i - m * KD;
      uint32_t lo, hi;
      widen4(sl, ast[q], lo, hi);
      sA32[m * KCD + 2 * j] = lo;
      sA32[m * KCD + 2 * j + 1] = hi;
    }
  };
  auto s_load = [&](int c, int buf) {
    if (!(MODE & 4)) return;
    float* sc = sscale + (buf ? MR * KKB : 0);
    for (int i = t; i < MR * KKB; i += NT) {
      const int m = i / KKB;
      const int kl = i - m * KKB;
      const int kb = c * KKB + kl;
      sc[m * KKB + kl] = a_s[(int64_t)m * nkb + kb] * w_s[(int64_t)nb * nkb + kb];
    }
  };

  if (nchunk > 0) {
    a_load(0);
    a_store(0);
    s_load(0, 0);
  }
  __syncthreads();

  uint32_t wpre[RW][PREF][ELD];
#pragma unroll
  for (int r = 0; r < RW; ++r)
#pragma unroll
    for (int q = 0; q < PREF; ++q)
#pragma unroll
      for (int d = 0; d < ELD; ++d) {
        wpre[r][q][d] = 0u;
        if (q < nchunk)
          wpre[r][q][d] = wp[(int64_t)r * (K / 4) + (int64_t)q * KD + lane_d + d];
      }

  float acc[RW][MR];
#pragma unroll
  for (int r = 0; r < RW; ++r)
#pragma unroll
    for (int m = 0; m < MR; ++m) acc[r][m] = 0.f;

  const int kblk = (int)(((int64_t)lane * EL) >> 7);

  for (int c = 0; c < nchunk; ++c) {
    uint4 areg[MR][NREG];
    if (MODE & 1) {
      const uint32_t* sp =
          reinterpret_cast<const uint32_t*>(sA[c & 1]) + lane_d;
#pragma unroll
      for (int m = 0; m < MR; ++m)
#pragma unroll
        for (int q = 0; q < NREG; ++q)
          areg[m][q] =
              reinterpret_cast<const uint4*>(sp + m * KCD)[q];
    } else {
#pragma unroll
      for (int m = 0; m < MR; ++m)
#pragma unroll
        for (int q = 0; q < NREG; ++q)
          areg[m][q] = make_uint4(0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u);
    }
    if (MODE & 2) {
      a_store((c + 1) & 1);
      if (c + 2 < nchunk) a_load(c + 2);
      s_load(c + 1, (c + 1) & 1);
    }
    if (c + PREF < nchunk) {
#pragma unroll
      for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int d = 0; d < ELD; ++d)
          wpre[r][PREF - 1][d] =
              wp[(int64_t)r * (K / 4) + (int64_t)(c + PREF) * KD + lane_d + d];
    }
    __syncthreads();

    uint32_t p[RW][ELH];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
      uint32_t wv[ELD];
#pragma unroll
      for (int d = 0; d < ELD; ++d) {
        wv[d] = wpre[r][0][d];
#pragma unroll
        for (int q = 0; q + 1 < PREF; ++q) wpre[r][q][d] = wpre[r][q + 1][d];
      }
#pragma unroll
      for (int d = 0; d < ELD; ++d) {
        uint32_t lo, hi;
        widen4(sl, wv[d], lo, hi);
        p[r][2 * d] = lo;
        p[r][2 * d + 1] = hi;
      }
    }
    const float* sc = sscale + ((c & 1) ? MR * KKB : 0) + kblk;
#pragma unroll
    for (int m = 0; m < MR; ++m) {
      const float sm = (MODE & 4) ? sc[m * KKB] : 0.005f;
#pragma unroll
      for (int r = 0; r < RW; ++r) {
        float loc = 0.f;
#pragma unroll
        for (int q = 0; q < NREG; ++q) {
          const uint4 xa = areg[m][q];
          loc = dot2(p[r][4 * q + 0], xa.x, loc);
          loc = dot2(p[r][4 * q + 1], xa.y, loc);
          loc = dot2(p[r][4 * q + 2], xa.z, loc);
          loc = dot2(p[r][4 * q + 3], xa.w, loc);
        }
        acc[r][m] += loc * sm;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < RW; ++r)
#pragma unroll
    for (int m = 0; m < MR; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off);
      if (lane == 0 && row0 + r < N)
        out[(int64_t)m * N + row0 + r] = __float2half(v);
    }
}


// ---------------------------------------------------------------------------
// kG: standalone replica of the production block GEMM inner loop
// (csrc/rocm/fp8_emu_gemm.cu fp8_emu_gemm_block_kernel), parameterised.
// ---------------------------------------------------------------------------
template <int BM, int BN, int TM, int TN, int MINB = 1>
__global__ void __launch_bounds__((BM / TM) * (BN / TN), MINB)
k_gemm8(const uint8_t* __restrict__ A, const uint8_t* __restrict__ W,
        const float* __restrict__ a_s, const float* __restrict__ w_s,
        const __half* __restrict__ lut, __half* __restrict__ C, const int M,
        const int N, const int K, const int nkb, const int block_n) {
  constexpr int MG = BM / TM, NG = BN / TN;
  constexpr int NT = MG * NG;
  constexpr int kFBk = 32, kFKw4 = 8, kFKw = 16, kFKp = 17, kFLogKw4 = 3;
  constexpr int A4 = (BM * kFKw4 + NT - 1) / NT;
  constexpr int B4 = (BN * kFKw4 + NT - 1) / NT;
  constexpr int FRAGS = A4 + B4;
  constexpr int cpb = 128 / kFBk;

  __shared__ uint32_t As[2][BM * kFKp];
  __shared__ uint32_t Bs[2][BN * kFKp];
  __shared__ __half sl[256];

  const int tid = threadIdx.x;
  const int tx = tid % NG, ty = tid / NG;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int K4 = K >> 2;
  const int NK = (K + kFBk - 1) / kFBk;

  for (int i = tid; i < 256; i += NT) sl[i] = lut[i];
  __syncthreads();

  float acc[TM][TN], blk[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = blk[i][j] = 0.f;

  uint32_t f0[FRAGS], f1[FRAGS];
  auto load = [&](int k0, uint32_t* dst) {
    const uint32_t* ap = reinterpret_cast<const uint32_t*>(A + (int64_t)m0 * K + k0);
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(W + (int64_t)n0 * K + k0);
    const bool kfull = (k0 + kFBk) <= K;
#pragma unroll
    for (int u = 0; u < A4; ++u) {
      const int idx = tid + u * NT, r = idx >> kFLogKw4, jd = idx & (kFKw4 - 1);
      uint32_t v = 0;
      if (idx < BM * kFKw4 && m0 + r < M && (kfull || (k0 + (jd + 1) * 4) <= K))
        v = ap[(int64_t)r * K4 + jd];
      dst[u] = v;
    }
#pragma unroll
    for (int u = 0; u < B4; ++u) {
      const int idx = tid + u * NT, r = idx >> kFLogKw4, jd = idx & (kFKw4 - 1);
      uint32_t v = 0;
      if (idx < BN * kFKw4 && n0 + r < N && (kfull || (k0 + (jd + 1) * 4) <= K))
        v = wp[(int64_t)r * K4 + jd];
      dst[A4 + u] = v;
    }
  };
  auto store = [&](int S, const uint32_t* src) {
#pragma unroll
    for (int u = 0; u < A4; ++u) {
      const int idx = tid + u * NT;
      if (idx >= BM * kFKw4) continue;
      const int r = idx >> kFLogKw4, jd = idx & (kFKw4 - 1);
      uint32_t lo, hi;
      widen4(sl, src[u], lo, hi);
      As[S][r * kFKp + 2 * jd] = lo;
      As[S][r * kFKp + 2 * jd + 1] = hi;
    }
#pragma unroll
    for (int u = 0; u < B4; ++u) {
      const int idx = tid + u * NT;
      if (idx >= BN * kFKw4) continue;
      const int r = idx >> kFLogKw4, jd = idx & (kFKw4 - 1);
      uint32_t lo, hi;
      widen4(sl, src[A4 + u], lo, hi);
      Bs[S][r * kFKp + 2 * jd] = lo;
      Bs[S][r * kFKp + 2 * jd + 1] = hi;
    }
  };
  auto compute = [&](int S, float (&a)[TM][TN]) {
    const uint32_t* as = As[S];
    const uint32_t* bs = Bs[S];
#pragma unroll
    for (int u = 0; u < kFKw; ++u) {
      uint32_t va[TM], vb[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) va[i] = as[(ty + MG * i) * kFKp + u];
#pragma unroll
      for (int j = 0; j < TN; ++j) vb[j] = bs[(tx + NG * j) * kFKp + u];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) a[i][j] = dot2(va[i], vb[j], a[i][j]);
    }
  };
  auto fold = [&](int kb, float (&a)[TM][TN]) {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int row = m0 + ty + MG * i;
      if (row >= M) continue;
      const float sa = a_s[(int64_t)row * nkb + kb];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int col = n0 + tx + NG * j;
        if (col >= N) continue;
        acc[i][j] += a[i][j] * sa * w_s[(int64_t)(col / block_n) * nkb + kb];
        a[i][j] = 0.f;
      }
    }
  };

  load(0, f0);
  int c = 0;
  for (; c + 1 < NK; c += 2) {
    store(0, f0);
    __syncthreads();
    load((c + 1) * kFBk, f1);
    compute(0, blk);
    if (((c + 1) % cpb) == 0) fold(c / cpb, blk);
    store(1, f1);
    __syncthreads();
    load((c + 2) * kFBk, f0);
    compute(1, blk);
    if (((c + 2) % cpb) == 0) fold((c + 1) / cpb, blk);
  }
  if (c < NK) {
    store(0, f0);
    __syncthreads();
    compute(0, blk);
    if (((c + 1) % cpb) == 0) fold(c / cpb, blk);
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + ty + MG * i;
    if (row >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + tx + NG * j;
      if (col >= N) continue;
      C[(int64_t)row * N + col] = __float2half(acc[i][j]);
    }
  }
}

// ---------------------------------------------------------------------------
// prototype: unchunked stream at M=8 (whole A in LDS, rows=64/block)
// ---------------------------------------------------------------------------
template <int M, int ROWS, int PREF>
__global__ void __launch_bounds__(512) k1_kernel(const uint8_t* __restrict__ a,
                          const uint8_t* __restrict__ w,
                          const float* __restrict__ a_s,
                          const float* __restrict__ w_s,
                          const __half* __restrict__ lut, __half* __restrict__ out,
                          const int N, const int K, const int nkb,
                          const int block_n) {
  constexpr int NT = 512;
  extern __shared__ uint32_t dyn[];
  __half* sA = reinterpret_cast<__half*>(dyn);
  float* sscale = reinterpret_cast<float*>(dyn + (size_t)M * (K / 2));
  __shared__ __half sl[256];

  const int t = threadIdx.x;
  const int lane = t & 63;
  const int warp = t >> 6;
  const int nwarp = NT / 64;
  const int K4 = K >> 2;

  lut_stage(sl, lut, NT);
  __syncthreads();
  {
    uint32_t* sA32 = reinterpret_cast<uint32_t*>(sA);
    const uint32_t* a32 = reinterpret_cast<const uint32_t*>(a);
    for (int64_t i = t; i < (int64_t)M * K4; i += NT) {
      uint32_t lo, hi;
      widen4(sl, a32[i], lo, hi);
      sA32[2 * i] = lo;
      sA32[2 * i + 1] = hi;
    }
  }
  {
    const int nb = (int)(((int64_t)blockIdx.x * ROWS) / block_n);
    const float* wsr = w_s + (int64_t)nb * nkb;
    for (int i = t; i < M * nkb; i += NT) {
      const int m = i / nkb;
      sscale[i] = a_s[(int64_t)m * nkb + i - m * nkb] * wsr[i - m * nkb];
    }
  }
  __syncthreads();

  const int n0 = blockIdx.x * ROWS;
  const int k4v = K4 >> 2;  // 16-byte vectors per row
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);
  const int niter = (k4v + 63) >> 6;

  for (int r = warp; r < ROWS; r += nwarp) {
    const int row = n0 + r;
    if (row >= N) break;
    const uint4* __restrict__ wp = wv + (((int64_t)row * K) >> 4);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    int i = lane;
    uint4 buf[PREF];
#pragma unroll
    for (int q = 0; q < PREF; ++q) {
      const int ii = i + q * 64;
      buf[q] = make_uint4(0, 0, 0, 0);
      if (ii < k4v) buf[q] = wp[ii];
    }
    for (int j = 0; j < niter; ++j) {
      const uint4 vv = buf[0];
      const int inext = i + PREF * 64;
      uint4 vn = make_uint4(0, 0, 0, 0);
      if (j + PREF < niter && inext < k4v) vn = wp[inext];
#pragma unroll
      for (int q = 0; q + 1 < PREF; ++q) buf[q] = buf[q + 1];
      buf[PREF - 1] = vn;

      uint32_t p[8];
      widen4(sl, vv.x, p[0], p[1]);
      widen4(sl, vv.y, p[2], p[3]);
      widen4(sl, vv.z, p[4], p[5]);
      widen4(sl, vv.w, p[6], p[7]);
      const int ic = min(i, k4v - 1);
      const int kb = ic / (128 / 16);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const uint4* ap =
            reinterpret_cast<const uint4*>(sA + (int64_t)m * K + ic * 16);
        const uint4 xa = ap[0];
        const uint4 xb = ap[1];
        float loc = 0.f;
        loc = dot2(p[0], xa.x, loc);
        loc = dot2(p[1], xa.y, loc);
        loc = dot2(p[2], xa.z, loc);
        loc = dot2(p[3], xa.w, loc);
        loc = dot2(p[4], xb.x, loc);
        loc = dot2(p[5], xb.y, loc);
        loc = dot2(p[6], xb.z, loc);
        loc = dot2(p[7], xb.w, loc);
        acc[m] += loc * sscale[m * nkb + kb];
      }
      i += 64;
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[m];
#pragma unroll
      for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off);
      if (lane == 0) out[(int64_t)m * N + row] = __float2half(v);
    }
  }
}

// ---------------------------------------------------------------------------
// fp64 reference (one thread per output)
// ---------------------------------------------------------------------------
__global__ void k_ref(const uint8_t* __restrict__ a, const uint8_t* __restrict__ w,
                      const float* __restrict__ a_s,
                      const float* __restrict__ w_s,
                      const __half* __restrict__ lut, __half* __restrict__ out,
                      const int M, const int N, const int K, const int nkb,
                      const int block_n) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int n = (int)(idx % N);
  const int m = (int)(idx / N);
  if (m >= M) return;
  const int nb = n / block_n;
  double tot = 0.0;  // per block_k window, then scale (matches kernel semantics)
  for (int kb = 0; kb < nkb; ++kb) {
    double s = 0.0;
    for (int k = kb * 128; k < (kb + 1) * 128; ++k) {
      const double av = __half2float(lut[a[(int64_t)m * K + k] & 0xFF]);
      const double wv = __half2float(lut[w[(int64_t)n * K + k] & 0xFF]);
      s += av * wv;
    }
    tot += s * ((double)a_s[(int64_t)m * nkb + kb] *
                (double)w_s[(int64_t)nb * nkb + kb]);
  }
  out[(int64_t)m * N + n] = __float2half((float)tot);
}

// ---------------------------------------------------------------------------
// host harness
// ---------------------------------------------------------------------------
static __half* make_lut_e4m3() {
  static __half h[256];
  for (int i = 0; i < 256; ++i) {
    uint32_t b = (uint32_t)i;
    uint32_t s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
    float v;
    if (e == 0) {
      v = (m == 0) ? 0.f : (float)m * (1.0f / 8.0f) * powf(2.f, -6.f);
      if (m == 0) v = 0.f;
      else v = ldexpf((float)m / 8.0f, -6);
    } else if (e == 15) {
      v = (m == 7) ? NAN : ldexpf(1.0f + (float)m / 8.0f, 15 - 7);
    } else {
      v = ldexpf(1.0f + (float)m / 8.0f, (int)e - 7);
    }
    if (s) v = -v;
    h[i] = __float2half(v);
  }
  return h;
}

static float frand(uint32_t& st) {
  st = st * 1664525u + 1013904223u;
  return ((st >> 8) & 0xFFFFFF) / (float)0x1000000 * 2.f - 1.f;
}

struct Case {
  int M, N, K;
  std::vector<uint8_t> a, w;
  std::vector<float> a_s, w_s;
};

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "peak-dot2";
  const int N = argc > 2 ? atoi(argv[2]) : 8704;
  const int K = argc > 3 ? atoi(argv[3]) : 5120;
  const int M = argc > 4 ? atoi(argv[4]) : 8;
  const int iters = argc > 5 ? atoi(argv[5]) : 200;

  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  printf("# device %s CUs=%d clk=%d kHz\n", prop.name, prop.multiProcessorCount,
         prop.clockRate);

  uint32_t* d_sink = nullptr;
  HIP_CHECK(hipMalloc(&d_sink, 64));
  HIP_CHECK(hipMemset(d_sink, 0, 64));
  std::vector<uint32_t> h_sink(16);

  if (mode == "peak-dot2") {
    for (int ilp : {1, 2, 4, 8, 16}) {
      hipEvent_t e0, e1;
      HIP_CHECK(hipEventCreate(&e0));
      HIP_CHECK(hipEventCreate(&e1));
      const int nb = 60 * 8, nt = 256;
      auto run = [&](int il) {
        switch (il) {
          case 1: k_peak_dot2<1><<<nb, nt>>>(d_sink, iters); break;
          case 2: k_peak_dot2<2><<<nb, nt>>>(d_sink, iters); break;
          case 4: k_peak_dot2<4><<<nb, nt>>>(d_sink, iters); break;
          case 8: k_peak_dot2<8><<<nb, nt>>>(d_sink, iters); break;
          default: k_peak_dot2<16><<<nb, nt>>>(d_sink, iters); break;
        }
      };
      run(ilp);
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipEventRecord(e0));
      for (int r = 0; r < 5; ++r) run(ilp);
      HIP_CHECK(hipEventRecord(e1));
      HIP_CHECK(hipEventSynchronize(e1));
      float ms = 0.f;
      HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
      const double dot2s = (double)nb * nt * (double)iters * ilp * 5;
      printf("peak-dot2 ilp=%2d  %8.3f ms  %8.2f Gdot2/s  (%.2f T-MAC/s)\n", ilp,
             ms, dot2s / ms / 1e6, dot2s * 2.0 / (ms * 1e-3) / 1e12);
      hipEventDestroy(e0);
      hipEventDestroy(e1);
    }
  } else if (mode == "peak-mix") {
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    const int nb = 60 * 8, nt = 256;
    k_peak_mix<1><<<nb, nt>>>(d_sink, iters);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(e0));
    for (int r = 0; r < 5; ++r) k_peak_mix<1><<<nb, nt>>>(d_sink, iters);
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    const double dot2s = (double)nb * nt * iters * 64.0 * 5;
    const double luts = (double)nb * nt * iters * 64.0 * 5;
    const double ldss = (double)nb * nt * iters * 16.0 * 5;
    printf("peak-mix(8m)   %8.3f ms  %8.2f Gdot2/s  %7.2f Glut/s  %7.2f Glds128/s\n",
           ms, dot2s / ms / 1e6, luts / ms / 1e6, ldss / ms / 1e6);
    printf("  -> per-lane cycles/dot2 %.2f  (assumes 1 CU instr/cycle, 60CU@1.6GHz)\n",
           (ms * 1e-3) * 60 * 1.6e9 / (dot2s / 1.0));
    return 0;
  } else if (mode == "peak-lds") {
    __half* hlut = make_lut_e4m3();
    __half* dlut = nullptr;
    HIP_CHECK(hipMalloc(&dlut, 256 * sizeof(__half)));
    HIP_CHECK(hipMemcpy(dlut, hlut, 256 * sizeof(__half), hipMemcpyHostToDevice));
    {
      hipEvent_t e0, e1;
      HIP_CHECK(hipEventCreate(&e0));
      HIP_CHECK(hipEventCreate(&e1));
      const int nb = 60 * 8, nt = 256;
      k_peak_lds_u16<<<nb, nt>>>(dlut, d_sink, iters);
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipEventRecord(e0));
      for (int r = 0; r < 5; ++r) k_peak_lds_u16<<<nb, nt>>>(dlut, d_sink, iters);
      HIP_CHECK(hipEventRecord(e1));
      HIP_CHECK(hipEventSynchronize(e1));
      float ms = 0.f;
      HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
      double ops = (double)nb * nt * iters * 8.0 * 5;
      printf("peak-lds-u16   %8.3f ms  %8.3f Gops/s  (%.2f Gops/s/CU)\n", ms,
             ops / ms / 1e6, ops / ms / 1e6 / 60);
      HIP_CHECK(hipEventRecord(e0));
      for (int r = 0; r < 5; ++r) k_peak_lds128<<<nb, nt>>>(d_sink, iters);
      HIP_CHECK(hipEventRecord(e1));
      HIP_CHECK(hipEventSynchronize(e1));
      HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
      ops = (double)nb * nt * iters * 8.0 * 5;
      printf("peak-lds-b128  %8.3f ms  %8.3f Gops/s  (%6.1f GB/s, %.2f Gops/s/CU)\n",
             ms, ops / ms / 1e6, ops * 16.0 / ms / 1e6, ops / ms / 1e6 / 60);
      hipEventDestroy(e0);
      hipEventDestroy(e1);
    }
  } else if (mode == "peak-gmem") {
    const int64_t bytes = (int64_t)512 << 20;
    uint8_t* buf = nullptr;
    HIP_CHECK(hipMalloc(&buf, bytes));
    HIP_CHECK(hipMemset(buf, 1, bytes));
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    k_peak_gmem<<<60 * 8, 256>>>(reinterpret_cast<const uint4*>(buf), d_sink,
                                 bytes / 16);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(e0));
    for (int r = 0; r < 5; ++r)
      k_peak_gmem<<<60 * 8, 256>>>(reinterpret_cast<const uint4*>(buf), d_sink,
                                   bytes / 16);
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    printf("peak-gmem read  %8.3f ms  %8.1f GB/s\n", ms,
           (double)bytes * 5 / (ms * 1e-3) / 1e9);
    return 0;
  } else if (mode == "abl") {
    const int BLOCK = 128;
    const int nkb = K / BLOCK;
    Case c;
    c.M = M; c.N = N; c.K = K;
    c.a.assign((size_t)M * K, 0);
    c.w.assign((size_t)N * K, 0);
    uint32_t st = 12345;
    for (auto& v : c.w) v = (uint8_t)(((uint32_t)(frand(st) * 0.5e9f + 0.5e9f)) & 0x6F);
    for (auto& v : c.a) v = (uint8_t)(((uint32_t)(frand(st) * 0.5e9f + 0.5e9f)) & 0x6F);
    const int nbn = N / BLOCK;
    c.a_s.assign((size_t)M * nkb, 0.005f);
    c.w_s.assign((size_t)nbn * nkb, 0.005f);
    for (auto& v : c.a_s) v = 0.002f + (frand(st) * 0.5f + 0.5f) * 0.01f;
    for (auto& v : c.w_s) v = 0.002f + (frand(st) * 0.5f + 0.5f) * 0.01f;
    __half* hlut = make_lut_e4m3();
    uint8_t *da, *dw; float *da_s, *dw_s; __half *dlut, *dout;
    HIP_CHECK(hipMalloc(&da, c.a.size()));
    HIP_CHECK(hipMalloc(&dw, c.w.size()));
    HIP_CHECK(hipMalloc(&da_s, c.a_s.size() * 4));
    HIP_CHECK(hipMalloc(&dw_s, c.w_s.size() * 4));
    HIP_CHECK(hipMemcpy(da_s, c.a_s.data(), c.a_s.size() * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dw_s, c.w_s.data(), c.w_s.size() * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dlut, 512));
    HIP_CHECK(hipMalloc(&dout, (size_t)M * N * 2));
    HIP_CHECK(hipMemcpy(da, c.a.data(), c.a.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dw, c.w.data(), c.w.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dlut, hlut, 512, hipMemcpyHostToDevice));
    const int KC = 1024;
    auto timeit = [&](const char* tag, auto&& fn) {
      fn();
      HIP_CHECK(hipDeviceSynchronize());
      hipEvent_t e0, e1;
      HIP_CHECK(hipEventCreate(&e0));
      HIP_CHECK(hipEventCreate(&e1));
      float ms = 1e9f;
      for (int round = 0; round < 7; ++round) {
        HIP_CHECK(hipEventRecord(e0));
        for (int r = 0; r < iters; ++r) fn();
        HIP_CHECK(hipEventRecord(e1));
        HIP_CHECK(hipEventSynchronize(e1));
        float t = 0.f;
        HIP_CHECK(hipEventElapsedTime(&t, e0, e1));
        if (t < ms) ms = t;
      }
      ms /= iters;
      const double dot2s = (double)M * N * (K / 2.0);
      printf("%-22s %8.4f ms  %6.0f GB/s  dot2=%6.0f G/s\n", tag, ms,
             (double)N * K / (ms * 1e-3) / 1e9, dot2s / ms / 1e6);
      hipEventDestroy(e0); hipEventDestroy(e1);
    };
    auto run = [&](const char* tag, int mode, int pref, int rw, int MR) {
      const int rows_pb = 8 * rw;
      const size_t sm = (mode >= 2) ? 16384 : 0;
      if (mode == 0)
        timeit(tag, [&] { k_abl<0, 1024, 3, 8, 1><<<(N + 7) / 8, 512, 0>>>(da, dw, dlut, dout, N, K); });
      else if (mode == 1 && pref == 3 && rw == 1 && MR == 8)
        timeit(tag, [&] { k_abl<1, 1024, 3, 8, 1><<<(N + 7) / 8, 512, sm>>>(da, dw, dlut, dout, N, K); });
      else if (mode == 1 && pref == 3 && rw == 2)
        timeit(tag, [&] { k_abl<1, 1024, 3, 8, 2><<<(N + rows_pb - 1) / rows_pb, 512, sm>>>(da, dw, dlut, dout, N, K); });
      else if (mode == 1 && pref == 3 && rw == 4)
        timeit(tag, [&] { k_abl<1, 1024, 3, 8, 4><<<(N + rows_pb - 1) / rows_pb, 512, sm>>>(da, dw, dlut, dout, N, K); });
      else if (mode == 2 && rw == 1)
        timeit(tag, [&] { k_abl<2, 1024, 3, 8, 1><<<(N + 7) / 8, 512, 16384>>>(da, dw, dlut, dout, N, K); });
      else if (mode == 2 && rw == 2)
        timeit(tag, [&] { k_abl<2, 1024, 3, 8, 2><<<(N + rows_pb - 1) / rows_pb, 512, 16384>>>(da, dw, dlut, dout, N, K); });
      else if (mode == 2 && rw == 4)
        timeit(tag, [&] { k_abl<2, 1024, 3, 8, 4><<<(N + rows_pb - 1) / rows_pb, 512, 16384>>>(da, dw, dlut, dout, N, K); });
      else if (mode == 1 && rw == 4 && MR == 4)
        timeit(tag, [&] { k_abl<1, 1024, 3, 4, 4><<<(N + rows_pb - 1) / rows_pb, 512, sm>>>(da, dw, dlut, dout, N, K); });
    };
    auto g8run = [&](const char* tag, auto kern, int BM, int BN, int NT, int MINB) {
      const int grid = ((N + BN - 1) / BN) * ((M + BM - 1) / BM);
      printf("# %s BM=%d BN=%d NT=%d MINB=%d grid=%d\n", tag, BM, BN, NT, MINB, grid);
      timeit(tag, [&] {
        kern<<<dim3((N + BN - 1) / BN, (M + BM - 1) / BM), NT>>>(da, dw, da_s, dw_s,
                                                                 dlut, dout, M, N, K,
                                                                 nkb, BLOCK);
      });
    };
    g8run("kG 8x64 T1N4", k_gemm8<8, 64, 1, 4, 1>, 8, 64, 128, 1);
    g8run("kG 8x64 T1N4 mb4", k_gemm8<8, 64, 1, 4, 4>, 8, 64, 128, 4);
    g8run("kG 8x128 T1N4", k_gemm8<8, 128, 1, 4, 1>, 8, 128, 256, 1);
    g8run("kG 8x128 T1N8", k_gemm8<8, 128, 1, 8, 1>, 8, 128, 128, 1);
    g8run("kG 8x64 T1N8", k_gemm8<8, 64, 1, 8, 1>, 8, 64, 64, 1);
    g8run("kG 16x64 T2N4", k_gemm8<16, 64, 2, 4, 1>, 16, 64, 128, 1);
    g8run("kG 32x64 T2N4 (old)", k_gemm8<32, 64, 2, 4, 4>, 32, 64, 256, 4);
    g8run("kG 8x64 T1N2", k_gemm8<8, 64, 1, 2, 1>, 8, 64, 256, 1);
    g8run("kG 8x64 T1N2 mb2", k_gemm8<8, 64, 1, 2, 2>, 8, 64, 256, 2);
    g8run("kG 8x32 T1N2", k_gemm8<8, 32, 1, 2, 1>, 8, 32, 128, 1);
    g8run("kG 8x64 T1N1", k_gemm8<8, 64, 1, 1, 1>, 8, 64, 512, 1);
    g8run("kG 8x128 T1N2", k_gemm8<8, 128, 1, 2, 1>, 8, 128, 512, 1);
    g8run("kG 16x64 T1N2", k_gemm8<16, 64, 1, 2, 1>, 16, 64, 512, 1);
    auto k8run = [&](const char* tag, auto kern, int RW, int KC, int NT, int PREF,
                     int EL) {
      const int rows_pb = (NT / 64) * RW;
      const size_t sm =
          (size_t)2 * 8 * (KC / 2) * 4 + (size_t)2 * 8 * (KC / 128) * 4;
      const int grid = (N + rows_pb - 1) / rows_pb;
      printf("# %s RW=%d KC=%d NT=%d P%d EL=%d rows_pb=%d grid=%d smem=%zu\n", tag,
             RW, KC, NT, PREF, EL, rows_pb, grid, sm);
      timeit(tag, [&] {
        kern<<<grid, NT, sm>>>(da, dw, da_s, dw_s, dlut, dout, N, K, nkb, BLOCK);
      });
    };
    k8run("k8 E8 RW1 KC512 NT256 P3", k8_kernel<8, 1, 512, 3, 256, 7, 8>, 1, 512, 256, 3, 8);
    k8run("k8 E8 RW1 KC512 NT256 P4", k8_kernel<8, 1, 512, 4, 256, 7, 8>, 1, 512, 256, 4, 8);
    k8run("k8 E8 RW1 KC512 NT128 P3", k8_kernel<8, 1, 512, 3, 128, 7, 8>, 1, 512, 128, 3, 8);
    k8run("k8 E8 RW2 KC512 NT256 P3", k8_kernel<8, 2, 512, 3, 256, 7, 8>, 2, 512, 256, 3, 8);
    k8run("k8 E16 RW1 KC1024 NT512 P3", k8_kernel<8, 1, 1024, 3, 512, 7, 16>, 1, 1024, 512, 3, 16);
    k8run("k8 E16 RW1 KC1024 NT512 P2b", k8_kernel<8, 1, 1024, 2, 512, 7, 16>, 1, 1024, 512, 2, 16);
    k8run("k8 E16 RW1 KC1024 NT512 P2", k8_kernel<8, 1, 1024, 2, 512, 7, 16>, 1, 1024, 512, 2, 16);
    k8run("k8 E8 RW1 KC512 NT256 noA", k8_kernel<8, 1, 512, 3, 256, 5, 8>, 1, 512, 256, 3, 8);
    run("abl0  A0 RW1", 0, 3, 1, 8);
    timeit("abl3  A8 rot", [&] {
      k_abl<3, 1024, 3, 8, 1><<<(N + 7) / 8, 512, 16384>>>(da, dw, dlut, dout, N, K);
    });
    timeit("abl3  A1 rot", [&] {
      k_abl<3, 1024, 3, 1, 1><<<(N + 7) / 8, 512, 16384>>>(da, dw, dlut, dout, N, K);
    });
    timeit("abl3  A4 rot", [&] {
      k_abl<3, 1024, 3, 4, 1><<<(N + 7) / 8, 512, 16384>>>(da, dw, dlut, dout, N, K);
    });
    run("abl1  A8 RW1", 1, 3, 1, 8);
    run("abl1  A8 RW2", 1, 3, 2, 8);
    run("abl1  A8 RW4", 1, 3, 4, 8);
    run("abl1  A4 RW4", 1, 3, 4, 4);
    run("abl2  A8 RW1", 2, 3, 1, 8);
    run("abl2  A8 RW2", 2, 3, 2, 8);
    run("abl2  A8 RW4", 2, 3, 4, 8);
    return 0;
  } else if (mode == "check" || mode == "k3" || mode == "k1") {
    const int BLOCK = 128;
    const int nkb = K / BLOCK, nbn = N / BLOCK;
    if (K % BLOCK || N % BLOCK) {
      printf("N,K must be multiples of 128\n");
      return 1;
    }
    Case c;
    c.M = M;
    c.N = N;
    c.K = K;
    c.a.assign((size_t)M * K, 0);
    c.w.assign((size_t)N * K, 0);
    c.a_s.assign((size_t)M * nkb, 0);
    c.w_s.assign((size_t)nbn * nkb, 0);
    uint32_t st = 12345;
    // e4m3 codes with exponent field <= 13 (no NaN/Inf), magnitudes 0.002-120
    auto code = [&]() {
      const uint32_t r = (uint32_t)(frand(st) * 0.5e9f + 0.5e9f);
      uint32_t e = r % 14;             // 0..13
      uint32_t m = (r >> 4) & 7;
      uint32_t s = (r >> 7) & 1;
      return (uint8_t)((s << 7) | (e << 3) | m);
    };
    for (auto& v : c.w) v = code();
    for (auto& v : c.a) v = code();
    for (auto& v : c.a_s) v = 0.002f + (frand(st) * 0.5f + 0.5f) * 0.01f;
    for (auto& v : c.w_s) v = 0.002f + (frand(st) * 0.5f + 0.5f) * 0.01f;

    __half* hlut = make_lut_e4m3();
    uint8_t *da, *dw;
    float *da_s, *dw_s;
    __half *dlut, *dout, *dref;
    HIP_CHECK(hipMalloc(&da, c.a.size()));
    HIP_CHECK(hipMalloc(&dw, c.w.size()));
    HIP_CHECK(hipMalloc(&da_s, c.a_s.size() * 4));
    HIP_CHECK(hipMalloc(&dw_s, c.w_s.size() * 4));
    HIP_CHECK(hipMalloc(&dlut, 512));
    HIP_CHECK(hipMalloc(&dout, (size_t)M * N * 2));
    HIP_CHECK(hipMalloc(&dref, (size_t)M * N * 2));
    HIP_CHECK(hipMemcpy(da, c.a.data(), c.a.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dw, c.w.data(), c.w.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(da_s, c.a_s.data(), c.a_s.size() * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dw_s, c.w_s.data(), c.w_s.size() * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dlut, hlut, 512, hipMemcpyHostToDevice));

    const size_t wbytes = (size_t)N * K;
    auto bench = [&](const char* tag, auto&& fn) {
      fn();
      HIP_CHECK(hipDeviceSynchronize());
      hipEvent_t e0, e1;
      HIP_CHECK(hipEventCreate(&e0));
      HIP_CHECK(hipEventCreate(&e1));
      HIP_CHECK(hipEventRecord(e0));
      for (int r = 0; r < iters; ++r) fn();
      HIP_CHECK(hipEventRecord(e1));
      HIP_CHECK(hipEventSynchronize(e1));
      float ms = 0.f;
      HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
      ms /= iters;
      printf("%-10s N=%5d K=%5d M=%d  %8.4f ms  %7.0f GB/s\n", tag, N, K, M, ms,
             (double)wbytes / (ms * 1e-3) / 1e9);
      hipEventDestroy(e0);
      hipEventDestroy(e1);
      return ms;
    };

    auto do_check = [&](const char* tag) {
      const int64_t nout = (int64_t)M * N;
      k_ref<<<(int)((nout + 255) / 256), 256>>>(da, dw, da_s, dw_s, dlut, dref, M,
                                               N, K, nkb, BLOCK);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<__half> ho(nout), hr(nout);
      HIP_CHECK(hipMemcpy(ho.data(), dout, nout * 2, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(hr.data(), dref, nout * 2, hipMemcpyDeviceToHost));
      double maxrel = 0;
      int ndiff = 0;
      for (int64_t i = 0; i < nout; ++i) {
        const double o = __half2float(ho[i]), r = __half2float(hr[i]);
        const double d = fabs(o - r) / (fabs(r) > 1e-3 ? fabs(r) : 1e-3);
        if (d > maxrel) maxrel = d;
        if (o != r) ++ndiff;
      }
      printf("  [%s] rel_err=%.2e  differing=%d/%lld  sample o=%.6f ref=%.6f\n",
             tag, maxrel, ndiff, (long long)nout, __half2float(ho[0]),
             __half2float(hr[0]));
    };

    const int KC = 1024;
    if ((mode == "k3" || mode == "check") && K % KC == 0) {
      const int nchunk = K / KC;
      const size_t smem = (size_t)2 * M * (KC / 2) * 4 +
                          (size_t)2 * M * (KC / 128) * 4;
      printf("# k3 KC=%d nchunk=%d smem=%zu bytes\n", KC, nchunk, smem);
      HIP_CHECK(hipMemset(dout, 0, (size_t)M * N * 2));
      bench("k3", [&] {
        k3_kernel<8, KC, 3><<<(N + 15) / 16, 1024, smem>>>(
            da, dw, da_s, dw_s, dlut, dout, N, K, nkb, BLOCK, nchunk);
      });
      if (mode == "check") do_check("k3");
    }
    if (mode == "k1" || mode == "check") {
      const size_t smem = (size_t)M * K * 2 + (size_t)M * nkb * 4;
      printf("# k1 smem=%zu bytes (limit 65536)\n", smem);
      if (smem <= 65000) {
        HIP_CHECK(hipMemset(dout, 0, (size_t)M * N * 2));
        bench("k1", [&] {
          k1_kernel<8, 64, 3><<<(N + 63) / 64, 512, smem>>>(
              da, dw, da_s, dw_s, dlut, dout, N, K, nkb, BLOCK);
        });
        if (mode == "check") do_check("k1");
      } else {
        printf("k1 skipped: A does not fit LDS\n");
      }
    }
  } else {
    printf("unknown mode %s\n", mode.c_str());
    return 1;
  }
  return 0;
}
