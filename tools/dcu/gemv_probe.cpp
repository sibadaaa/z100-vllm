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
#include <cmath>
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
static float bench(void (*launch)(const __half*, const __half*, float*, int, int),
                   const __half* w, const __half* x, float* out, int N, int K,
                   double bytes);

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

// ---------------------------------------------------------------------------
// S29 WP-D: W4 + dot8 GEMV probe (docs/perf/W4-RESEARCH-20261025.md §4 step 1).
//
// ISA facts pinned by tools/dcu/dot8_numeric_probe.cpp: BOTH operands of
// v_dot8_i32_i4 / v_dot8_u32_u4 are 8 x 4-bit lanes (nibble u of a word sits
// in bits [4u, 4u+4), low nibble first).  W nibbles are signed (symmetric
// per-group quant).  int8 activations decompose exactly as
//   a = 16 * (a>>4) + ((a&15) - 8) + 8      (H_s = a>>4, L_s = (a&15)-8)
// so with signed-nibble packed views of A:
//   sum_k w*a = 16*dot8_i(w, H) + dot8_i(w, L) + 8*dot8_i(w, 0x11111111)
// (the last term is sum_k w_k, one extra dot per W word).  Packed-A traffic
// is 1 B per W byte (hi+lo), i.e. 2 B per original-fp8 weight byte, matching
// the W4-RESEARCH traffic model.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int32_t dot8_i(int32_t a, int32_t b) {
  int32_t acc = 0;
  asm volatile("v_dot8_i32_i4 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
  return acc;
}

// One warp owns one row of W (signed nibbles); A (int8) staged per K-slice and
// packed into two signed-nibble views in LDS; per-GROUP i32 accumulators fold
// into fp32 with the group scale at each group boundary.
template <int ROWS, int NWARP, int M, int SLICE, int GROUP>
__global__ void gemv_w4_dot8(const uint4* __restrict__ w,  // [N, K/32]
                             const uint4* __restrict__ x8, // [M, K/16] int8
                             const float* __restrict__ scale,  // [M, K/GROUP]
                             float* __restrict__ out,      // [M, N]
                             int N, int K) {
  __shared__ uint32_t sH[SLICE / 8][M];  // [packed-word][row]
  __shared__ uint32_t sL[SLICE / 8][M];
  const int lane = threadIdx.x & 63;
  const int warp = threadIdx.x >> 6;
  const int row0 = blockIdx.x * ROWS;
  const int32_t ONES = 0x11111111;  // 8 x nibble 1

  for (int s0 = 0; s0 < K; s0 += SLICE) {
    // stage + pack A slice: 16 int8 bytes -> 2 hi words + 2 lo words
    for (int i = threadIdx.x; i < M * (SLICE / 16); i += blockDim.x) {
      const int m = i / (SLICE / 16);
      const int c = i - m * (SLICE / 16);
      const uint4 v = x8[(long)m * (K / 16) + s0 / 16 + c];
      const uint32_t* bw = reinterpret_cast<const uint32_t*>(&v);
      uint32_t h[2] = {0, 0}, l[2] = {0, 0};
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const uint32_t b0 = bw[2 * t], b1 = bw[2 * t + 1];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const uint32_t byte =
              (u < 4) ? ((b0 >> (8 * u)) & 0xFFu) : ((b1 >> (8 * (u - 4))) & 0xFFu);
          const uint32_t hs = ((byte >> 4) & 0xFu);          // a>>4, as 2's comp
          const uint32_t ls = (((byte & 0xFu) + 8u) & 0xFu);  // (a&15)-8
          h[t] |= hs << (4 * u);
          l[t] |= ls << (4 * u);
        }
      }
      uint32_t* dh = &sH[(c * 16) / 8][0];
      uint32_t* dl = &sL[(c * 16) / 8][0];
      dh[m] = h[0]; dh[M + m] = h[1];
      dl[m] = l[0]; dl[M + m] = l[1];
    }
    __syncthreads();

    for (int r = warp; r < ROWS; r += NWARP) {
      const int row = row0 + r;
      const uint4* wr = w + (long)row * (K / 32) + s0 / 32;  // uint4 = 32 k
      float acc_f[8];
      int32_t accH[8], accL[8], accW[8];
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        acc_f[m] = 0.f; accH[m] = 0; accL[m] = 0; accW[m] = 0;
      }
      int g_in_slice = -1;

      // lane covers 16 k (a half-chunk): 2 W words + 2 packed-A words, so all
      // 64 lanes are busy at SLICE=1024 (64 x 16 = SLICE)
      const uint32_t* wu32 = reinterpret_cast<const uint32_t*>(wr);
      for (int hc = lane; hc < SLICE / 16; hc += 64) {
        const int g = (hc * 16) / GROUP;
        if (g != g_in_slice) {
          if (g_in_slice >= 0) {
            const int gi = s0 / GROUP + g_in_slice;
#pragma unroll
            for (int m = 0; m < 8; ++m)
              acc_f[m] += (float)(16 * accH[m] + accL[m] + 8 * accW[0]) *
                          scale[(long)m * (K / GROUP) + gi];
          }
          g_in_slice = g;
#pragma unroll
          for (int m = 0; m < 8; ++m) { accH[m] = 0; accL[m] = 0; }
          accW[0] = 0;
        }
        const uint2 wv = *reinterpret_cast<const uint2*>(wu32 + hc * 2);
        const int ai = hc * 2;
        const uint32_t wws[2] = {wv.x, wv.y};
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          // W word j pairs with packed-A word ai+j ONLY
          const uint32_t w32 = wws[j];
          const uint32_t* hw = &sH[ai + j][0];
          const uint32_t* lw = &sL[ai + j][0];
          accW[0] += dot8_i((int32_t)w32, ONES);
#pragma unroll
          for (int m = 0; m < M; ++m) {
            accH[m] += dot8_i((int32_t)w32, (int32_t)hw[m]);
            accL[m] += dot8_i((int32_t)w32, (int32_t)lw[m]);
          }
        }
      }
      if (g_in_slice >= 0) {
        const int gi = s0 / GROUP + g_in_slice;
#pragma unroll
        for (int m = 0; m < 8; ++m)
          acc_f[m] += (float)(16 * accH[m] + accL[m] + 8 * accW[0]) *
                      scale[(long)m * (K / GROUP) + gi];
      }
      for (int m = 0; m < M; ++m) {
        float v = acc_f[m];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off);
        // one K-stride partial per slice; accumulate (caller zero-fills out)
        if (lane == 0) out[(long)m * N + row] += v;
      }
    }
    __syncthreads();
  }
  (void)N;
}

template <int ROWS, int NWARP, int M, int SLICE, int GROUP>
static void report_w4(const char* name, const uint4* w, const uint4* x8,
                      const float* scale, float* out, int N, int K) {
  auto launch = [&](const uint4* a, const uint4* b, const float* s, float* c,
                    int n, int k) {
    gemv_w4_dot8<ROWS, NWARP, M, SLICE, GROUP>
        <<<n / ROWS, NWARP * 64>>>(a, b, s, c, n, k);
  };
  // L2-flushed timing: per-iteration events around the kernel only; a 64 MB
  // dummy write between iterations evicts W (13-22 MB) from L2.
  float* dummy;
  CK(hipMalloc(&dummy, 64u << 20));
  for (int i = 0; i < 3; ++i) launch(w, x8, scale, out, N, K);
  CK(hipDeviceSynchronize());
  hipEvent_t ev_a, ev_b;
  CK(hipEventCreate(&ev_a));
  CK(hipEventCreate(&ev_b));
  const int iters = 30;
  double total_ms = 0.0;
  for (int i = 0; i < iters; ++i) {
    CK(hipEventRecord(ev_a));
    launch(w, x8, scale, out, N, K);
    CK(hipEventRecord(ev_b));
    CK(hipEventSynchronize(ev_b));
    float ms = 0.f;
    CK(hipEventElapsedTime(&ms, ev_a, ev_b));
    total_ms += ms;
    CK(hipMemsetAsync(dummy, i & 0xFF, 64u << 20));
    CK(hipDeviceSynchronize());
  }
  const double ms = total_ms / iters;
  const double raw = (double)N * K / 2.0;  // W4 nibble bytes streamed
  const double fp8eq = (double)N * K;      // original-fp8-byte equivalent
  std::printf(
      "%-16s ROWS=%d NWARP=%d M=%d SLICE=%d G=%d  %7.3f ms  raw %6.0f GB/s  "
      "fp8-eq %6.0f GB/s\n",
      name, ROWS, NWARP, M, SLICE, GROUP, ms, raw / (ms / 1e3) / 1e9,
      fp8eq / (ms / 1e3) / 1e9);
  CK(hipEventDestroy(ev_a));
  CK(hipEventDestroy(ev_b));
  CK(hipFree(dummy));
}

// numeric check: W signed nibbles + int8 A from a deterministic LCG; scales 1;
// double-precision host reference (independent of the kernel decomposition).

static int w4_impulse_check(int N, int K, int M) {
  const int words = K / 8;
  std::vector<uint32_t> hw((size_t)N * words, 0x11111111u);  // all w=+1
  std::vector<uint8_t> ax((size_t)M * K, 0);
  ax[0] = 1;  // a[k=0] = 1 -> expect out = 1 (w_0=1)
  if (M > 1) ax[(size_t)1 * K + 5] = 0x81;  // a=-127 at k=5 -> out2 = -127
  std::vector<float> sc((size_t)M * (K / 128), 1.0f);
  uint4* dw; uint4* dx; float* ds; float* dout;
  CK(hipMalloc(&dw, hw.size() * 4));
  CK(hipMalloc(&dx, (size_t)M * K));
  CK(hipMalloc(&ds, sc.size() * 4));
  CK(hipMalloc(&dout, (size_t)M * N * 4));
  CK(hipMemcpy(dw, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(dx, ax.data(), ax.size(), hipMemcpyHostToDevice));
  CK(hipMemcpy(ds, sc.data(), sc.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemset(dout, 0, (size_t)M * N * 4));
  gemv_w4_dot8<4, 4, 8, 1024, 128><<<N / 4, 4 * 64>>>(dw, dx, ds, dout, N, K);
  hipError_t le = hipGetLastError();
  if (le != hipSuccess) std::printf("LAUNCH ERROR: %s\n", hipGetErrorString(le));
  CK(hipDeviceSynchronize());
  std::vector<float> got((size_t)M * N);
  CK(hipMemcpy(got.data(), dout, got.size() * 4, hipMemcpyDeviceToHost));
  std::printf("impulse m0n0: got %.3f want 1 | m1n0: got %.3f want -127 | "
              "m0n1(zero if unwritten): %.3f\n",
              got[0], got[(size_t)1 * N], got[1]);
  CK(hipFree(dw)); CK(hipFree(dx)); CK(hipFree(ds)); CK(hipFree(dout));
  return 0;
}

static int w4_numeric_check(int N, int K, int M) {
  const int words = K / 8;
  std::vector<uint32_t> hw((size_t)N * words);
  uint32_t rng = 0x12345678u;
  auto next = [&rng]() {
    rng = rng * 1664525u + 1013904223u;
    return rng;
  };
  for (auto& v : hw) {
    uint32_t x = 0;
    for (int u = 0; u < 8; ++u) x |= ((next() >> 24) % 16u) << (4 * u);
    v = x;
  }
  std::vector<uint8_t> ax((size_t)M * K);
  for (auto& v : ax) v = (uint8_t)((next() >> 16) & 0xFF);
  std::vector<float> sc((size_t)M * (K / 128), 1.0f);

  uint4* dw;
  uint4* dx;
  float* ds;
  float* dout;
  CK(hipMalloc(&dw, hw.size() * 4));
  CK(hipMalloc(&dx, (size_t)M * K));
  CK(hipMalloc(&ds, sc.size() * 4));
  CK(hipMalloc(&dout, (size_t)M * N * 4));
  CK(hipMemcpy(dw, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(dx, ax.data(), ax.size(), hipMemcpyHostToDevice));
  CK(hipMemcpy(ds, sc.data(), sc.size() * 4, hipMemcpyHostToDevice));

gemv_w4_dot8<4, 4, 8, 1024, 128><<<N / 4, 4 * 64>>>(dw, dx, ds, dout, N, K);
  CK(hipDeviceSynchronize());
  std::vector<float> got((size_t)M * N);
  CK(hipMemcpy(got.data(), dout, got.size() * 4, hipMemcpyDeviceToHost));

  auto nib = [](uint32_t word, int u) -> int {
    const uint32_t n = (word >> (4 * u)) & 0xFu;
    return (n >= 8) ? (int)n - 16 : (int)n;  // signed nibble
  };
  double num = 0.0, den = 0.0;
  int bad = 0;
  const int nSample = (N < 64 ? N : 64);
  for (int m = 0; m < M; ++m) {
    for (int si = 0; si < nSample; ++si) {
      const int n = si * (N / nSample);
      double ref = 0.0;
      for (int k = 0; k < K; ++k) {
        const int wn = nib(hw[(size_t)n * words + k / 8], k % 8);
        const int a = (int)(int8_t)ax[(size_t)m * K + k];
        ref += (double)wn * (double)a;
      }
      const double g = got[(size_t)m * N + n];
      const double d = std::fabs(g - ref);
      num += d * d;
      den += ref * ref;
      if (d > 1e-3 * (std::fabs(ref) + 1.0)) ++bad;
    }
  }
  const double rel = (den > 0) ? std::sqrt(num / den) : 0.0;
  std::printf("w4 numeric check: rel_l2=%.3e  outliers(>1e-3 rel)=%d\n", rel,
              bad);
  CK(hipFree(dw));
  CK(hipFree(dx));
  CK(hipFree(ds));
  CK(hipFree(dout));
  return bad == 0 ? 0 : 1;
}

static void run_w4(int N, int K) {
  const size_t wb = (size_t)N * K / 2;   // nibble bytes
  const size_t xb = (size_t)8 * K;       // int8 activations
  const size_t sb = (size_t)8 * (K / 128);
  uint4* w;
  uint4* x8;
  float* scale;
  float* out;
  CK(hipMalloc(&w, wb));
  CK(hipMalloc(&x8, xb));
  CK(hipMalloc(&scale, sb * 4));
  CK(hipMalloc(&out, (size_t)8 * N * 4));
  // deterministic fill (values irrelevant to the bandwidth probe)
  std::vector<uint32_t> hw(wb / 4, 0x21463857u);  // mixed signed nibbles
  std::vector<uint8_t> hx(xb, 0x80);              // int8 -128
  std::vector<float> hs(sb, 0.01f);
  CK(hipMemcpy(w, hw.data(), wb, hipMemcpyHostToDevice));
  CK(hipMemcpy(x8, hx.data(), xb, hipMemcpyHostToDevice));
  CK(hipMemcpy(scale, hs.data(), sb * 4, hipMemcpyHostToDevice));

  std::printf("-- W4+dot8 (N=%d K=%d, W %.1f MB nibbles / %.1f MB fp8-eq) --\n",
              N, K, wb / 1e6, (double)N * K / 1e6);
  report_w4<1, 1, 8, 1024, 128>("w4_dot8", w, x8, scale, out, N, K);
  report_w4<2, 2, 8, 1024, 128>("w4_dot8", w, x8, scale, out, N, K);
  report_w4<4, 4, 8, 1024, 128>("w4_dot8", w, x8, scale, out, N, K);
  report_w4<8, 8, 8, 1024, 128>("w4_dot8", w, x8, scale, out, N, K);
  report_w4<4, 4, 8, 2048, 128>("w4_dot8", w, x8, scale, out, N, K);
  report_w4<4, 4, 8, 1024, 64>("w4_dot8_g64", w, x8, scale, out, N, K);
  CK(hipFree(w));
  CK(hipFree(x8));
  CK(hipFree(scale));
  CK(hipFree(out));
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
  report_lds<8, 8, 8, 512>("lds_m8", w, x, out, N, K);
  report_lds<8, 8, 8, 1024>("lds_m8", w, x, out, N, K);
  report_lds<16, 16, 8, 512>("lds_m8", w, x, out, N, K);
  report_lds<8, 8, 4, 512>("lds_m4", w, x, out, N, K);
  report_lds<8, 8, 1, 512>("lds_m1", w, x, out, N, K);
  CK(hipFree(w));
  CK(hipFree(x));
  CK(hipFree(out));

  w4_impulse_check(5120, 5120, 8);
  if (w4_numeric_check(5120, 5120, 8) != 0) std::printf("NUMERIC CHECK FAILED\n");
  run_w4(5120, 5120);
  run_w4(8704, 5120);
  return 0;
}
