// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Device-side low-precision emulation primitives for gfx906 (Vega20 / Hygon
// Z100), shared by the FP8 and BF16 emulation kernels.
//
// Why this file exists: gfx906 has **no** fp8 conversion, no bf16 conversion
// and no bf16 dot product (`v_cvt_f32_fp8`, `v_cvt_pk_fp8_f32`,
// `v_cvt_f32_bf16`, `v_cvt_pk_bf16_f32`, `v_dot2_f32_bf16` all fail to
// assemble for `-mcpu=gfx906`), and no MFMA/WMMA at all.  Everything below is
// therefore bit arithmetic that the DTK clang backend lowers into packed
// 16-bit ALU ops (`v_pk_lshlrev_b16`, `v_pk_add_u16`, `v_pk_sub_u16`,
// `v_and_b32`, `v_and_or_b32`), a 256-entry LDS table read (`ds_read_u16`), or
// a plain shift (`v_lshlrev_b32`).
//
// The instruction matrix backing every primitive here is probed by
// `tools/dcu/isa_probe.sh` (rows prefixed `FP8EMU-*`, plus the pre-existing
// `BF16-*`/`FP8-*` negative controls) and archived under `docs/result/`.
//
// Conventions:
//   * A 32-bit "packed pair" holds element 2i in the **low** 16 bits and
//     element 2i+1 in the **high** 16 bits (little-endian byte order), so a
//     dword of 4 fp8 bytes expands to (pair0 = bytes 0,1), (pair1 = bytes 2,3).
//   * `fdot2` accumulates in fp32: `acc + w.x*x.x + w.y*x.y`, exactly the
//     fp32-accumulate contract that FP8/BF16 emulation requires.
#pragma once

#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace vllm {
namespace gfx906_emu {

using u16x2_t = unsigned short __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint32_t pk_bits(u16x2_t v) {
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ u16x2_t bits_pk(uint32_t b) {
  u16x2_t v;
  *reinterpret_cast<uint32_t*>(&v) = b;
  return v;
}

// Two fp16 lanes -> one fp32 accumulate via `v_dot2_f32_f16`.
__device__ __forceinline__ float dot2(uint32_t w2, uint32_t x2, float acc) {
  const __half2 w = *reinterpret_cast<const __half2*>(&w2);
  const __half2 x = *reinterpret_cast<const __half2*>(&x2);
  return __builtin_amdgcn_fdot2(w, x, acc, /*clamp=*/false);
}

// ---------------------------------------------------------------------------
// fp8 -> fp16 through a 256-entry table
// ---------------------------------------------------------------------------
// The table is built on the host (`vllm/model_executor/kernels/linear/
// scaled_mm/gfx906_fp8.py`) and staged into LDS once per block.  For both
// float8_e4m3fn and float8_e5m2 every finite code is representable in fp16
// **exactly** (e4m3: 3 mantissa bits, |x| <= 448; e5m2: 2 mantissa bits,
// |x| <= 57344 < 65504, and its 2^-16..2^-15 subnormals land on fp16
// subnormals), so the table is lossless.  NaN/inf codes map to fp16 NaN/inf.
__device__ __forceinline__ uint16_t fp8_lut_u16(const __half* __restrict__ lut,
                                                uint32_t code) {
  return *reinterpret_cast<const uint16_t*>(lut + (code & 0xFFu));
}

__device__ __forceinline__ uint32_t fp8x2_to_f16x2(
    const __half* __restrict__ lut, uint32_t two_bytes) {
  return (uint32_t)fp8_lut_u16(lut, two_bytes) |
         ((uint32_t)fp8_lut_u16(lut, two_bytes >> 8) << 16);
}

// Load the 256-entry table into LDS (`nthreads` >= 256 or a strided loop).
__device__ __forceinline__ void fp8_lut_stage(__half* __restrict__ dst,
                                              const __half* __restrict__ src,
                                              int nthreads) {
  for (int i = threadIdx.x; i < 256; i += nthreads) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// bf16 <-> fp32 / fp16
// ---------------------------------------------------------------------------
// bf16 -> fp32 is a shift of the 16 stored bits into the fp32 exponent field:
// exact for every bf16 value (1 instruction, proven emitted: see the
// `bf16_to_f32_shift` codegen row).
__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
  return __uint_as_float((uint32_t)b << 16);
}

__device__ __forceinline__ void bf16x2_to_f32x2(uint32_t p, float& lo,
                                               float& hi) {
  lo = __uint_as_float(p << 16);
  hi = __uint_as_float(p & 0xFFFF0000u);
}

// fp32 -> bf16, round-to-nearest (half-up on the dropped bits; the same
// sequence DTK's `__float2bfloat16` uses on gfx906).
__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
  return (uint16_t)((__float_as_uint(f) + 0x8000u) >> 16);
}

__device__ __forceinline__ uint32_t f32x2_to_bf16x2(float lo, float hi) {
  return (uint32_t)f32_to_bf16(lo) | ((uint32_t)f32_to_bf16(hi) << 16);
}

// bf16 -> fp16, scalar and fully guarded: fp16's exponent window is ~30
// binades narrower than bf16's, so |x| < 2^-14 flushes to a signed zero and
// |x| >= 65536 saturates to +/-65504.  Values in [2^-14, 65504] are exact
// (bf16 keeps 8 mantissa bits, fp16 has 11).
__device__ __forceinline__ uint16_t bf16_to_f16_guarded(uint16_t b) {
  const uint32_t e = (b >> 7) & 0xFFu;
  const uint32_t s = b & 0x8000u;
  if (e >= 143u) return (uint16_t)(s | 0x7BFFu);  // saturate to +/-65504
  if (e <= 112u) return (uint16_t)s;             // flush to +/-0
  return (uint16_t)(s | ((e - 112u) << 10) | ((b & 0x7Fu) << 3));
}

__device__ __forceinline__ uint32_t bf16x2_to_f16x2_guarded(uint32_t p) {
  return (uint32_t)bf16_to_f16_guarded((uint16_t)p) |
         ((uint32_t)bf16_to_f16_guarded((uint16_t)(p >> 16)) << 16);
}

// Packed bf16x2 -> fp16x2.
//
// Naively shifting the lane left by 3 loses the two high bits of the 8-bit
// bias-127 exponent (they land at lane bits 16/17), which breaks every |x| >=
// 2 -- *and* every value whose exponent has bit 6 set, i.e. essentially all
// normals.  Subtracting the bias difference *before* the shift keeps the
// exponent in lane bits 7..12, so the shift never overflows the lane:
//
//   s = sign<<15 | e8<<7 | m7
//   (s - 112<<7)<<3 = sign<<18 | (e8-112)<<10 | m7<<3   (bits 15..17 dropped)
//
// The two lanes are converted in **32-bit scalar integer math on purpose**:
// both the `u16x2_t` vector form (clang lowers the shift with an
// `op_sel_hi:[0,1]` lane selection and the low lane then receives the
// un-biased value: 2^-14 -> 4.0) and the inline-asm form (V_PK_LSHLREV_B16
// takes *per-lane* shift amounts from the two halves of src0, so a scalar
// immediate shifts only one lane: 2^-14 -> 64.0) were measured wrong on
// hardware.  The explicit form below has no lane-selection subtlety and is
// verified end to end by tests/kernels/test_gfx906_bf16_gemv.py.
//
// **Requires** every lane exponent to be in [113, 143] (|x| in
// [2^-14, ~131072)).  Out-of-range lanes -- including a plain **zero**, whose
// e8 is 0 -- wrap silently: measured, a zero lane converts to 2.0, so the
// fast path is only usable when the caller has verified that *every* element
// of both operands is inside the window (no zeros, no padding, no masked
// positions).  The Python layer therefore keeps `range_safe=False` (the
// guarded scalar form above) and this variant exists for A/B experiments
// only.
__device__ __forceinline__ uint32_t bf16x2_to_f16x2_fast(uint32_t p) {
  const uint32_t lo = p & 0xFFFFu;
  const uint32_t hi = p >> 16;
  const uint32_t lo_h = (((lo - 0x3800u) << 3) & 0x7FFFu) | (lo & 0x8000u);
  const uint32_t hi_h = (((hi - 0x3800u) << 3) & 0x7FFFu) | (hi & 0x8000u);
  return (lo_h & 0xFFFFu) | (hi_h << 16);
}

// fp16 -> bf16 packed (round-to-nearest per lane, then drop the low 16 bits).
__device__ __forceinline__ uint32_t f16x2_to_bf16x2(uint32_t h) {
  const uint32_t lo = ((h << 16) + 0x8000u) >> 16;
  const uint32_t hi = ((h & 0xFFFF0000u) + 0x8000u) >> 16;
  return (lo & 0xFFFFu) | ((hi & 0xFFFFu) << 16);
}

// ---------------------------------------------------------------------------
// fp8 widening used by the staging path (element pairs, not LUT reads)
// ---------------------------------------------------------------------------
// Expand a dword of 4 fp8 codes into two fp16 pairs: (bytes 0,1), (bytes 2,3).
__device__ __forceinline__ void fp8x4_to_f16x2x2(
    const __half* __restrict__ lut, uint32_t four, uint32_t& lo_pair,
    uint32_t& hi_pair) {
  lo_pair = fp8x2_to_f16x2(lut, four);
  hi_pair = fp8x2_to_f16x2(lut, four >> 16);
}

}  // namespace gfx906_emu
}  // namespace vllm
