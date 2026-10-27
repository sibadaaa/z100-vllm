// gfx906 v_dot8_i32_i4 / v_dot8_u32_u4 numerical semantics probe.
//
// Both instructions compute sum over 8 lanes of (nibble_a * nibble_b), taking one
// nibble per 4-bit field of the 4 source bytes (low nibble first).
// The signed variant (i4) interprets 0x8..0xF as -8..-1; the unsigned variant
// (u4) interprets them as 8..15. Packing weights with the wrong variant is a
// data-dependent correctness bug, so this probe pins the contract down.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void dot8_s(int32_t a, int32_t b, int32_t *out) {
  int32_t acc = 0;
  asm volatile("v_dot8_i32_i4 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
  out[0] = acc;
}

__global__ void dot8_u(uint32_t a, uint32_t b, uint32_t *out) {
  uint32_t acc = 0;
  asm volatile("v_dot8_u32_u4 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
  out[0] = acc;
}

int main() {
  int32_t *ds;
  uint32_t *du;
  hipMalloc(&ds, 4);
  hipMalloc(&du, 4);
  int fails = 0;

  struct S { uint32_t a, b; const char *why; int32_t want; } st[] = {
      {0x11111111u, 0x22222222u, "signed: 8 x (1*2)", 16},
      {0xFFFFFFFFu, 0xFFFFFFFFu, "signed: 8 x (-1*-1)", 8},
      {0x11111111u, 0xFFFFFFFFu, "signed: 8 x (1*-1)", -8},
      // nibbles low->high are 8,7,6,5,4,3,2,1; 0x8 is -8 when signed => 20
      {0x12345678u, 0x11111111u, "signed: -8+7+6+5+4+3+2+1", 20},
      {0x00000000u, 0x22222222u, "signed: zero operand", 0},
  };
  struct U { uint32_t a, b; const char *why; uint32_t want; } ut[] = {
      // same nibbles, unsigned: 8+7+6+5+4+3+2+1 = 36
      {0x12345678u, 0x11111111u, "unsigned: 8+7+6+5+4+3+2+1", 36},
      {0xFFFFFFFFu, 0x11111111u, "unsigned: 8 x (15*1)", 120},
  };

  printf("%-38s %-18s %6s %6s\n", "case", "operands", "got", "want");
  for (auto &x : st) {
    int32_t h;
    dot8_s<<<1, 1>>>(x.a, x.b, ds);
    hipDeviceSynchronize();
    hipMemcpy(&h, ds, 4, hipMemcpyDeviceToHost);
    bool ok = h == x.want;
    fails += !ok;
    printf("%-38s %08x/%08x %6d %6d %s\n", x.why, x.a, x.b, h, x.want,
           ok ? "OK" : "MISMATCH");
  }
  for (auto &x : ut) {
    uint32_t h;
    dot8_u<<<1, 1>>>(x.a, x.b, du);
    hipDeviceSynchronize();
    hipMemcpy(&h, du, 4, hipMemcpyDeviceToHost);
    bool ok = h == x.want;
    fails += !ok;
    printf("%-38s %08x/%08x %6u %6u %s\n", x.why, x.a, x.b, h, x.want,
           ok ? "OK" : "MISMATCH");
  }
  printf("\n%s\n", fails ? "FAILED" : "all dot8 semantics confirmed");
  hipFree(ds);
  hipFree(du);
  return fails != 0;
}
