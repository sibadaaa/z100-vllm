// gfx906 dot-instruction throughput probe.
// Goal: does v_dot8_i32_i4 issue at the same rate as v_dot4_i32_i8 (=> 2x MACs),
// or does it split into two cycles (=> same MACs as dot4)?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>

#define NACC 8
#define ITER 2048

#define KERNEL(NAME, BODY)                                                    \
  __global__ void NAME(uint32_t *out, uint32_t seed) {                        \
    uint32_t a = seed + threadIdx.x;                                          \
    uint32_t b = a * 3u + 1u;                                                 \
    uint32_t acc[NACC];                                                       \
    _Pragma("unroll") for (int i = 0; i < NACC; ++i) acc[i] = i;              \
    _Pragma("unroll 1") for (int it = 0; it < ITER; ++it) {                    \
      _Pragma("unroll") for (int i = 0; i < NACC; ++i) { BODY }               \
      a += 1u;                                                                \
    }                                                                         \
    uint32_t s = 0;                                                           \
    _Pragma("unroll") for (int i = 0; i < NACC; ++i) s += acc[i];             \
    if (s == 0xdeadbeefu) out[threadIdx.x] = s;                               \
  }

#define DOT4  asm volatile("v_dot4_i32_i8 %0, %1, %2, %0" : "+v"(acc[i]) : "v"(a), "v"(b));
#define DOT8  asm volatile("v_dot8_i32_i4 %0, %1, %2, %0" : "+v"(acc[i]) : "v"(a), "v"(b));
#define UDOT4 asm volatile("v_dot4_u32_u8 %0, %1, %2, %0" : "+v"(acc[i]) : "v"(a), "v"(b));
#define DOT2I asm volatile("v_dot2_i32_i16 %0, %1, %2, %0" : "+v"(acc[i]) : "v"(a), "v"(b));
#define PFMA  asm volatile("v_pk_fma_f16 %0, %1, %2, %0" : "+v"(acc[i]) : "v"(a), "v"(b));
#define FMAF  asm volatile("v_fma_f32 %0, %1, %2, %0" : "+v"(acc[i]) : "v"(a), "v"(b));
#define DOT2F asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc[i]) : "v"(a), "v"(b));

KERNEL(k_dot4, DOT4)
KERNEL(k_dot8, DOT8)
KERNEL(k_udot4, UDOT4)
KERNEL(k_dot2i, DOT2I)
KERNEL(k_pfma, PFMA)
KERNEL(k_fmaf, FMAF)
KERNEL(k_dot2f, DOT2F)

struct Res { const char *name; double ms; double macs; };

template <class K>
Res run(const char *name, K k, double macs_per_insn, int blocks, int threads,
        uint32_t *d_out, int reps) {
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < reps; ++r) k<<<blocks, threads>>>(d_out, r);
  hipDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double insn = (double)blocks * threads * (double)ITER * NACC * reps;
  return {name, ms, insn * macs_per_insn};
}

int main() {
  int dev = 0;
  hipDeviceProp_t p;
  hipGetDeviceProperties(&p, dev);
  printf("device: %s  gcnArch=%s  CU=%d  clock=%.2f GHz\n", p.name, p.gcnArchName,
         p.multiProcessorCount, p.clockRate / 1e6);

  uint32_t *d_out;
  hipMalloc(&d_out, 4096 * sizeof(uint32_t));
  const int blocks = p.multiProcessorCount * 8;
  const int threads = 256;
  const int reps = 20;

  Res r[7] = {
      run("v_fma_f32        (1 MAC, fp32)", k_fmaf, 1, blocks, threads, d_out, reps),
      run("v_pk_fma_f16     (2 MAC, fp16)", k_pfma, 2, blocks, threads, d_out, reps),
      run("v_dot2_f32_f16   (2 MAC, f32 acc)", k_dot2f, 2, blocks, threads, d_out, reps),
      run("v_dot2_i32_i16   (2 MAC, i32 acc)", k_dot2i, 2, blocks, threads, d_out, reps),
      run("v_dot4_i32_i8    (4 MAC, i32 acc)", k_dot4, 4, blocks, threads, d_out, reps),
      run("v_dot4_u32_u8    (4 MAC, u32 acc)", k_udot4, 4, blocks, threads, d_out, reps),
      run("v_dot8_i32_i4    (8 MAC, i32 acc)", k_dot8, 8, blocks, threads, d_out, reps),
  };

  printf("\nblocks=%d threads=%d reps=%d ITER=%d NACC=%d\n", blocks, threads, reps,
         ITER, NACC);
  printf("%-34s %10s %14s %10s\n", "instruction", "ms", "MAC/s", "x fma_f32");
  double base = r[0].macs / (r[0].ms * 1e-3);
  for (auto &x : r) {
    double macs = x.macs / (x.ms * 1e-3);
    printf("%-34s %10.2f %14.2f %10.2f\n", x.name, x.ms, macs / 1e12, macs / base);
  }
  hipFree(d_out);
  return 0;
}
