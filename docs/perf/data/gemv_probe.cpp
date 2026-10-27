// SPDX-License-Identifier: Apache-2.0
// GEMV geometry probe for gfx906 (Vega20 / Z100L), no torch needed.
//
// Question: the same weights stream at ~784 GB/s through a pure reduction but
// only 197-292 GB/s through rocBLAS GEMV.  Which geometry closes the gap?
//
// Each variant keeps "one warp owns one row of W" and sweeps:
//   ROWS    rows per block (grid = N/ROWS)
//   UNROLL  uint4 (16 B) loads kept in flight per lane
//   NWARP   warps per block
//   MODE    0 = read-only sum (control), 1 = dot with one x, 2 = dot with 8 x
//
// Build/run (inside the DTK container):
//   hipcc -O3 --offload-arch=gfx906 gemv_probe.cpp -o /tmp/gemv_probe
//   /tmp/gemv_probe
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                             \
  do {                                                                    \
    hipError_t e_ = (x);                                                  \
    if (e_ != hipSuccess) {                                               \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));            \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

template <int ROWS, int UNROLL, int NWARP, int MODE>
__global__ void gemv_rows(const __half* __restrict__ w,  // [N, K]
                          const __half* __restrict__ x,  // [M, K]
                          float* __restrict__ out,       // [M, N]
                          int N, int K) {
  const int lane = threadIdx.x & 63;
  const int warp = threadIdx.x >> 6;
  const int kv = K >> 3;  // uint4 = 8 half per lane step
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);
  const int n0 = blockIdx.x * ROWS;

  for (int r = warp; r < ROWS; r += NWARP) {
    const int row = n0 + r;
    const uint4* __restrict__ wr = wv + (long)row * kv;
    float2 acc2[8];
    half2 hacc[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      acc2[m] = make_float2(0.f, 0.f);
      hacc[m] = __float2half2_rn(0.f);
    }

    for (int k0 = lane; k0 < kv; k0 += 64 * UNROLL) {
      uint4 v[UNROLL];
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        int idx = k0 + u * 64;
        if (idx < kv) v[u] = wr[idx];
      }
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        int idx = k0 + u * 64;
        if (idx >= kv) continue;
        const __half2* h = reinterpret_cast<const __half2*>(&v[u]);
        if (MODE == 0) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float2 f = __half22float2(h[j]);
            acc2[0].x += f.x;
            acc2[0].y += f.y;
          }
        } else {
          const int base = idx * 8;  // element offset inside the row
          const int nm = (MODE == 1 ? 1 : 8);
#pragma unroll
          for (int m = 0; m < 8; ++m) {
            if (m >= nm) continue;
            const __half2* xh =
                reinterpret_cast<const __half2*>(x + (long)m * K + base);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              if (MODE == 3) {
                hacc[m] = __hfma2(h[j], xh[j], hacc[m]);
              } else {
                float2 fw = __half22float2(h[j]);
                float2 fx = __half22float2(xh[j]);
                acc2[m].x += fw.x * fx.x;
                acc2[m].y += fw.y * fx.y;
              }
            }
          }
        }
      }
    }
    // warp reduction
    const int nm = (MODE == 1 ? 1 : 8);
    if (MODE == 3) {
#pragma unroll
      for (int m = 0; m < 8; ++m) acc2[m] = __half22float2(hacc[m]);
    }
    for (int off = 32; off > 0; off >>= 1) {
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        acc2[m].x += __shfl_down(acc2[m].x, off);
        acc2[m].y += __shfl_down(acc2[m].y, off);
      }
    }
    if (lane == 0) {
      for (int m = 0; m < nm; ++m) out[(long)m * N + row] = acc2[m].x + acc2[m].y;
    }
  }
}


// M=8 with A staged in LDS as a K-slice (the production kernel stages *all* of
// A, which is 8*5120*2 = 82 KB and therefore gets gated out at M >= 5).
// One warp per row, A read from LDS, W streamed in uint4.
template <int ROWS, int NWARP, int M, int SLICE>
__global__ void gemv_lds(const __half* __restrict__ w,
                         const __half* __restrict__ x, float* __restrict__ out,
                         int N, int K) {
  __shared__ __half sA[8][SLICE];
  const int lane = threadIdx.x & 63;
  const int warp = threadIdx.x >> 6;
  const int row0 = blockIdx.x * ROWS;
  constexpr int SV = SLICE / 8;  // uint4 per slice row

  float acc[8];
#pragma unroll
  for (int m = 0; m < 8; ++m) acc[m] = 0.f;

  for (int s0 = 0; s0 < K; s0 += SLICE) {
    for (int i = threadIdx.x; i < M * SV; i += blockDim.x) {
      const int m = i / SV, v = i - m * SV;
      reinterpret_cast<uint4*>(sA)[i] =
          reinterpret_cast<const uint4*>(x + (long)m * K + s0)[v];
    }
    __syncthreads();
    for (int r = warp; r < ROWS; r += NWARP) {
      const uint4* wr =
          reinterpret_cast<const uint4*>(w + (long)(row0 + r) * K) + s0 / 8;
      for (int v = lane; v < SV; v += 64) {
        const uint4 vv = wr[v];
        const __half2* h = reinterpret_cast<const __half2*>(&vv);
#pragma unroll
        for (int m = 0; m < 8; ++m) {
          if (m >= M) continue;
          const __half2* xm =
              reinterpret_cast<const __half2*>(&sA[m][v * 8]);
          float loc = 0.f;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float2 fw = __half22float2(h[j]);
            const float2 fx = __half22float2(xm[j]);
            loc += fw.x * fx.x + fw.y * fx.y;
          }
          acc[m] += loc;
        }
      }
    }
    __syncthreads();
  }
  for (int m = 0; m < M; ++m) {
    float v = acc[m];
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off);
    if (lane == 0) out[(long)m * N + row0 + warp] = v;
  }
}

template <int ROWS, int NWARP, int M, int SLICE>
static void report_lds(const char* name, const __half* w, const __half* x,
                       float* out, int N, int K) {
  auto launch = [](const __half* a, const __half* b, float* c, int n, int k) {
    gemv_lds<ROWS, NWARP, M, SLICE><<<n / ROWS, NWARP * 64>>>(a, b, c, n, k);
  };
  const float g = bench(launch, w, x, out, N, K, (double)N * K * 2);
  std::printf("%-12s ROWS=%-2d NWARP=%-2d M=%d SLICE=%d  GB/s %6.0f\n", name, ROWS,
              NWARP, M, SLICE, g);
}

static float bench(void (*launch)(const __half*, const __half*, float*, int, int),
                   const __half* w, const __half* x, float* out, int N, int K,
                   double bytes) {
  for (int i = 0; i < 3; ++i) launch(w, x, out, N, K);
  CK(hipDeviceSynchronize());
  hipEvent_t a, b;
  CK(hipEventCreate(&a));
  CK(hipEventCreate(&b));
  const int iters = 20;
  CK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) launch(w, x, out, N, K);
  CK(hipEventRecord(b));
  CK(hipDeviceSynchronize());
  float ms = 0.f;
  CK(hipEventElapsedTime(&ms, a, b));
  CK(hipEventDestroy(a));
  CK(hipEventDestroy(b));
  return (float)(bytes / (ms / iters / 1e3) / 1e9);
}

template <int ROWS, int UNROLL, int NWARP, int MODE>
static void launch(const __half* w, const __half* x, float* out, int N, int K) {
  gemv_rows<ROWS, UNROLL, NWARP, MODE>
      <<<N / ROWS, NWARP * 64>>>(w, x, out, N, K);
}

template <int ROWS, int UNROLL, int NWARP>
static void report(const char* name, const __half* w, const __half* x, float* out,
                   int N, int K, double bytes) {
  const float g_sum = bench(&launch<ROWS, UNROLL, NWARP, 0>, w, x, out, N, K, bytes);
  const float g_d1 = bench(&launch<ROWS, UNROLL, NWARP, 1>, w, x, out, N, K, bytes);
  const float g_d8 = bench(&launch<ROWS, UNROLL, NWARP, 2>, w, x, out, N, K, bytes);
  const float g_d8p = bench(&launch<ROWS, UNROLL, NWARP, 3>, w, x, out, N, K, bytes);
  std::printf(
      "%-12s ROWS=%-2d UNROLL=%d NWARP=%-2d  sum %6.0f  dot1 %6.0f  "
      "dot8 %6.0f  dot8pk %6.0f\n",
      name, ROWS, UNROLL, NWARP, g_sum, g_d1, g_d8, g_d8p);
}

int main() {
  const int N = 8704, K = 5120;
  const size_t wb = (size_t)N * K * 2, xb = (size_t)8 * K * 2;
  __half *w, *x;
  float* out;
  CK(hipMalloc(&w, wb));
  CK(hipMalloc(&x, xb));
  CK(hipMalloc(&out, (size_t)8 * N * 4));
  std::vector<__half> hw((size_t)N * K), hx(8 * K);
  for (auto& v : hw) v = __float2half(0.01f);
  for (auto& v : hx) v = __float2half(0.02f);
  CK(hipMemcpy(w, hw.data(), wb, hipMemcpyHostToDevice));
  CK(hipMemcpy(x, hx.data(), xb, hipMemcpyHostToDevice));

  std::printf("N=%d K=%d, weight bytes %.1f MB, ref=784 GB/s read / 815 copy\n", N, K,
              wb / 1e6);
  report<1, 1, 8>("baseline", w, x, out, N, K, wb);
  report<2, 1, 8>("rows2", w, x, out, N, K, wb);
  report<4, 1, 8>("rows4", w, x, out, N, K, wb);
  report<8, 1, 8>("rows8", w, x, out, N, K, wb);
  report<2, 2, 8>("unroll2", w, x, out, N, K, wb);
  report<4, 2, 8>("unroll2", w, x, out, N, K, wb);
  report<4, 4, 8>("unroll4", w, x, out, N, K, wb);
  report<4, 2, 4>("nwarp4", w, x, out, N, K, wb);
  report<4, 2, 16>("nwarp16", w, x, out, N, K, wb);
  report<8, 2, 16>("rows8w16", w, x, out, N, K, wb);
  std::printf("-- A staged in LDS, sliced over K (the M=8 fix) --\n");
  report_lds<8, 8, 8, 512>("lds_m8", w, x, out, N, K);
  report_lds<8, 8, 8, 1024>("lds_m8", w, x, out, N, K);
  report_lds<16, 16, 8, 512>("lds_m8", w, x, out, N, K);
  report_lds<8, 8, 4, 512>("lds_m4", w, x, out, N, K);
  report_lds<8, 8, 1, 512>("lds_m1", w, x, out, N, K);
  return 0;
}
