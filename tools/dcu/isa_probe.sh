#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Z100 (gfx906) ISA capability probe -- S11d.
#
# Answers three questions per instruction / builtin, without ever running a
# self-compiled HIP executable (only `clang -c`, `clang -S`, `llvm-objdump`):
#
#   1. does the DTK assembler accept it for `-mcpu=gfx906`?
#   2. does DTK clang advertise the `__builtin_amdgcn_*` builtin
#      (`__has_builtin` on the amdgcn target)?
#   3. does clang actually *emit* the instruction from device code, and which
#      instruction does it emit?
#
# Usage:
#   tools/dcu/isa_probe.sh [OUTDIR]
#
# Environment:
#   CLANG    assembler/compiler driver   (default: first of $DTK_ROOT/llvm/bin/clang,
#                                         /opt/dtk/lib/llvm/bin/clang, clang)
#   HIPCC    hipcc driver for phase 3b   (default: first of $DTK_ROOT/bin/hipcc, hipcc);
#                                         set HIPCC=skip to skip phase 3b
#   OBJDUMP  llvm-objdump                (default: next to CLANG)
#   ISA_PROBE_STRICT=1  exit 1 when the measured ISA matrix differs from the
#                       recorded gfx906 baseline (default: report only, exit 0)
#   ISA_PROBE_OUT       default output directory (arg wins)
#
# Exit codes: 0 = all phases ran and (non-strict, or strict + baseline matched);
#             1 = harness failure, or strict mismatch, or strict _has_builtin_ miss.
#
# Baseline recorded on 2026-10: see docs/result/z100-11d-z100-isa-capability-audit-result.md

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
_dtk="${DTK_ROOT:-/opt/dtk}"
if [ -z "${CLANG:-}" ]; then
  for c in "${_dtk}/llvm/bin/clang" "${_dtk}/lib/llvm/bin/clang" "$(command -v clang 2>/dev/null)"; do
    [ -n "$c" ] && [ -x "$c" ] && { CLANG="$c"; break; }
  done
fi
[ -x "${CLANG:-}" ] || { echo "isa_probe: no clang found (set CLANG=...)" >&2; exit 1; }
CLANG_DIR="$(dirname "$CLANG")"

if [ "${HIPCC:-}" != "skip" ]; then
  if [ -z "${HIPCC:-}" ]; then
    for h in "${_dtk}/bin/hipcc" "$(command -v hipcc 2>/dev/null)"; do
      [ -n "$h" ] && [ -x "$h" ] && { HIPCC="$h"; break; }
    done
  fi
fi
OBJDUMP="${OBJDUMP:-${CLANG_DIR}/llvm-objdump}"
[ -x "$OBJDUMP" ] || OBJDUMP="$(command -v llvm-objdump 2>/dev/null || true)"

# DTK's clang looks for a gcc installation matching its own defaults; without
# this, `hipcc -S` dies with "'cmath' file not found". Same fix as
# tools/dcu/env.sh (not sourced here so the probe stays env-agnostic).
if [ -z "${CPLUS_INCLUDE_PATH:-}" ]; then
  for _cxx in /usr/include/c++/*; do
    [ -d "$_cxx" ] || continue
    for _inc in "$_cxx" "/usr/include/x86_64-linux-gnu/c++/$(basename "$_cxx")"; do
      [ -d "$_inc" ] || continue
      case ":${CPLUS_INCLUDE_PATH:-}:" in
        *":${_inc}:"*) ;;
        *) export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:+${CPLUS_INCLUDE_PATH}:}${_inc}" ;;
      esac
    done
  done
  unset _cxx _inc
fi

OUTDIR="${1:-${ISA_PROBE_OUT:-/tmp/isa_probe}}"
mkdir -p "$OUTDIR"
WORK="$OUTDIR/work"
rm -rf "$WORK"; mkdir -p "$WORK"

ENV_TAG="${ISA_PROBE_ENV_TAG:-$(hostname)}"
MC="gfx906"
ASM_RESULTS="$OUTDIR/assembler.tsv"
BLT_RESULTS="$OUTDIR/builtins.tsv"
CG_RESULTS="$OUTDIR/codegen.tsv"

echo "======================================================================"
echo "Z100 / gfx906 ISA probe   (S11d)"
echo "======================================================================"
echo "env tag   : $ENV_TAG"
echo "clang     : $CLANG"
"$CLANG" --version | head -3 | sed 's/^/            /'
if [ -n "${HIPCC:-}" ] && [ "$HIPCC" != "skip" ]; then
  echo "hipcc     : $HIPCC"
  { "$HIPCC" --version 2>/dev/null || true; } | head -2 | sed 's/^/            /'
else
  echo "hipcc     : skipped"
fi
[ -n "$OBJDUMP" ] && echo "objdump   : $OBJDUMP"
echo "outdir    : $OUTDIR"
echo "target    : ${MC}"
echo

rc=0
mis=0     # harness-level failures only (syntax-error test line, codegen ERROR/MISSING)
syn=0     # assembler test lines whose *syntax* this script got wrong
record() { # record <file> <status> <name> <detail>
  printf '%s\t%s\t%s\n' "$2" "$3" "$4" >> "$1"
}

# --------------------------------------------------------------------------
# Phase 1: assembler acceptance
# --------------------------------------------------------------------------
# One line of assembly per candidate; `s_endpgm` makes it a complete shader.
# Instructions whose operand shape this script does not know are marked
# "SYNTAX" so a bad test line is never mistaken for a hardware verdict.
INSN_TABLE=$(cat <<'EOF'
INT8-dot4-signed|v_dot4_i32_i8 v0, v1, v2, v3
INT8-dot4-unsigned|v_dot4_u32_u8 v0, v1, v2, v3
INT8-dot2-signed|v_dot2_i32_i16 v0, v1, v2, v3
INT8-dot2-unsigned|v_dot2_u32_u16 v0, v1, v2, v3
INT8-dot8-signed|v_dot8_i32_i4 v0, v1, v2, v3
INT8-dot8-unsigned|v_dot8_u32_u4 v0, v1, v2, v3
INT8-dot4c-acc|v_dot4c_i32_i8 v0, v1, v2
FP16-dot2c-acc|v_dot2c_f32_f16 v0, v1, v2
FP16-dot2|v_dot2_f32_f16 v0, v1, v2, v3
INT24-mad-s|v_mad_i32_i24 v0, v1, v2, v3
INT24-mad-u|v_mad_u32_u24 v0, v1, v2, v3
INT16-mad-i32|v_mad_i32_i16 v0, v1, v2, v3
INT16-mad-u|v_mad_u16 v0, v1, v2, v3
INT16-mad-s|v_mad_i16 v0, v1, v2, v3
FP16-pk-fma|v_pk_fma_f16 v0, v1, v2, v3
FP16-pk-add|v_pk_add_f16 v0, v1, v2
FP16-pk-mul|v_pk_mul_f16 v0, v1, v2
INT16-pk-lshl|v_pk_lshlrev_b16 v0, v1, v2
FP16-pk-max|v_pk_max_f16 v0, v1, v2
INT16-pk-add|v_pk_add_u16 v0, v1, v2
CVT-pkrtz|v_cvt_pkrtz_f16_f32 v0, v1, v2
CVT-f16-f32|v_cvt_f16_f32 v0, v1
CVT-f32-f16|v_cvt_f32_f16 v0, v1
CVT-pk-f16-f32|v_cvt_pk_f16_f32 v0, v1, v2
CVT-pknorm-i16|v_cvt_pknorm_i16_f32 v0, v1, v2
MIX-mad-mix-f32|v_mad_mix_f32 v0, v1, v2, v3
MIX-fma-mix-f32|v_fma_mix_f32 v0, v1, v2, v3
MIX-fma-mixlo-f16|v_fma_mixlo_f16 v0, v1, v2, v3
MIX-fma-mixhi-f16|v_fma_mixhi_f16 v0, v1, v2, v3
MIX-mad-mixlo-f16|v_mad_mixlo_f16 v0, v1, v2, v3
MIX-mad-mixhi-f16|v_mad_mixhi_f16 v0, v1, v2, v3
FP32-fma|v_fma_f32 v0, v1, v2, v3
FP32-mac|v_mac_f32 v0, v1, v2
FP32-fmac|v_fmac_f32 v0, v1, v2
FP32-mad|v_mad_f32 v0, v1, v2, v3
FP8-cvt-f32-fp8|v_cvt_f32_fp8 v0, v1
FP8-cvt-pk-fp8|v_cvt_pk_fp8_f32 v0, v1, v2
FP8-cvt-f32-bf8|v_cvt_f32_bf8 v0, v1
BF16-cvt-f32-bf16|v_cvt_f32_bf16 v0, v1
BF16-dot2|v_dot2_f32_bf16 v0, v1, v2, v3
MFMA-f16|v_mfma_f32_16x16x16f16 v0, v1, v2, v3
MQSAD-pk|v_mqsad_pk_u16_u8 v[0:1], v[2:3], v4, v[6:7]
QSAD-pk|v_qsad_pk_u16_u8 v[0:1], v[2:3], v4, v[6:7]
FP8EMU-pk-sub-u16|v_pk_sub_u16 v0, v1, v2
FP8EMU-pk-min-f16|v_pk_min_f16 v0, v1, v2
FP8EMU-pk-max-f16|v_pk_max_f16 v0, v1, v2
FP8EMU-pack-b32-f16|v_pack_b32_f16 v0, v1, v2
FP8EMU-perm-b32|v_perm_b32 v0, v1, v2, v3
FP8EMU-lshl-or-b32|v_lshl_or_b32 v0, v1, v2, v3
FP8EMU-or3-b32|v_or3_b32 v0, v1, v2, v3
FP8EMU-alignbit-b32|v_alignbit_b32 v0, v1, v2, v3
FP8EMU-bfe-u32|v_bfe_u32 v0, v1, v2, v3
FP8EMU-bfe-i32|v_bfe_i32 v0, v1, v2, v3
FP8EMU-rndne-f32|v_rndne_f32 v0, v1
FP8EMU-cvt-pk-u8-f32|v_cvt_pk_u8_f32 v0, v1, v2, v3
FP8EMU-sat-pk-u8-i16|v_sat_pk_u8_i16 v0, v1
FP8EMU-ds-read-u16|ds_read_u16 v0, v1
FP8EMU-ds-read-b16|ds_read_b16 v0, v1
FP8EMU-ds-bpermute|ds_bpermute_b32 v0, v1, v2
FP8EMU-ds-permute|ds_permute_b32 v0, v1, v2
FP8EMU-dpp-quad|v_mov_b32_dpp v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
FP8EMU-sdwa-byte0|v_cvt_f32_i32_sdwa v0, v1 src0_sel:BYTE_0
FP8EMU-cvt-pk-f32-fp8|v_cvt_pk_f32_fp8 v0, v1
FP8EMU-cvt-pk-bf16-f32|v_cvt_pk_bf16_f32 v0, v1, v2
EOF
)

echo "----------------------------------------------------------------------"
echo "Phase 1: assembler acceptance (clang -target amdgcn-amd-amdhsa -mcpu=${MC})"
echo "----------------------------------------------------------------------"
: > "$ASM_RESULTS"
while IFS='|' read -r name insn; do
  [ -z "${name:-}" ] && continue
  f="$WORK/a.s"
  printf '\t%s\n\ts_endpgm\n' "$insn" > "$f"
  err="$("$CLANG" -target amdgcn-amd-amdhsa -mcpu="$MC" -c "$f" -o "$WORK/a.o" 2>&1)"
  if [ -z "$err" ]; then
    enc=""
    if [ -n "$OBJDUMP" ]; then
      enc="$("$OBJDUMP" -d --mcpu="$MC" "$WORK/a.o" 2>/dev/null | grep -m1 -oE '// [0-9A-Fa-f]+: [0-9A-Fa-f ]+' | sed 's|// ||')"
    fi
    printf 'OK      %-22s %s\n' "$name" "${enc:0:60}"
    record "$ASM_RESULTS" OK "$name" "$insn"
  else
    short="$(echo "$err" | grep -oE 'error: [^,]*' | head -1)"
    case "$short" in
      *"invalid operand"*|*"too few"*|*"too many"*|*"expected"*) st="SYNTAX" ;;
      *) st="FAIL" ;;
    esac
    printf '%-7s %-22s %s\n' "$st" "$name" "$short"
    record "$ASM_RESULTS" "$st" "$name" "$short"
    # A plain FAIL (the assembler rejects an opcode on this GPU) is a *valid
    # measurement*; only a bad test line is a harness error.
    [ "$st" = SYNTAX ] && syn=$((syn + 1))
  fi
done <<< "$INSN_TABLE"
echo "-> $ASM_RESULTS"
echo

# --------------------------------------------------------------------------
# Phase 2: __has_builtin availability
# --------------------------------------------------------------------------
BUILTIN_LIST="__builtin_amdgcn_sdot4
__builtin_amdgcn_sudot4
__builtin_amdgcn_udot4
__builtin_amdgcn_sdot2
__builtin_amdgcn_udot2
__builtin_amdgcn_sdot8
__builtin_amdgcn_sudot8
__builtin_amdgcn_udot8
__builtin_amdgcn_fdot2
__builtin_amdgcn_fdot2_f16_f16
__builtin_amdgcn_fdot2_f32_bf16
__builtin_amdgcn_mad_mix
__builtin_amdgcn_fma_mix
__builtin_amdgcn_fma_mix_f32
__builtin_amdgcn_cvt_pkrtz
__builtin_amdgcn_cvt_pk_u8_f32
__builtin_amdgcn_cvt_pk_i16
__builtin_amdgcn_cvt_pknorm_i16
__builtin_amdgcn_mqsad_pk_u16_u8
__builtin_amdgcn_qsad_pk_u16_u8
__builtin_amdgcn_ds_bpermute
__builtin_amdgcn_ds_swizzle
__builtin_amdgcn_mfma_f32_16x16x16f16
__builtin_amdgcn_mfma_i32_16x16x16i8
__builtin_amdgcn_wmma_f32_16x16x16_f16_w32
__builtin_amdgcn_cvt_pk_fp8_f32
__builtin_amdgcn_cvt_pk_f32_fp8
__builtin_amdgcn_perm
__builtin_amdgcn_s_sleep
__builtin_amdgcn_s_barrier"

H="$WORK/has_builtin.h"
: > "$H"
while IFS= read -r b; do
  [ -z "$b" ] && continue
  {
    printf '#if __has_builtin(%s)\n' "$b"
    printf '#define HAS_%s 1\n' "$b"
    printf '#endif\n'
  } >> "$H"
done <<< "$BUILTIN_LIST"

echo "----------------------------------------------------------------------"
echo "Phase 2: __has_builtin on the amdgcn target (-mcpu=${MC})"
echo "----------------------------------------------------------------------"
DEFS="$WORK/defs.txt"
"$CLANG" -target amdgcn-amd-amdhsa -mcpu="$MC" -E -dM "$H" > "$DEFS" 2>"$WORK/defs.err" || {
  echo "isa_probe: phase 2 preprocessor failed" >&2
  cat "$WORK/defs.err" >&2
  exit 1
}
: > "$BLT_RESULTS"
while IFS= read -r b; do
  [ -z "$b" ] && continue
  if grep -qx "#define HAS_${b} 1" "$DEFS"; then v=1; else v=0; fi
  printf 'HAS=%-2s %s\n' "$v" "$b"
  record "$BLT_RESULTS" "$v" "$b" "has_builtin=$v"
done <<< "$BUILTIN_LIST"
# control: the same header on the host triple must report nothing
if "$CLANG" -E -dM "$H" 2>/dev/null | grep -q '^#define HAS___builtin_amdgcn_'; then
  echo "WARN: amdgcn builtins reported available on the host triple (control failed)"
fi
echo "-> $BLT_RESULTS"
echo

# --------------------------------------------------------------------------
# Phase 3: codegen -- does clang emit the instruction?
# --------------------------------------------------------------------------
# Each entry: name|expected-instruction-regex|source
# Sources are freestanding (no HIP headers) and built for the amdgcn target,
# which is the exact same LLVM backend + MC used by the HIP device pass.
codegen_src() {
  cat <<'EOF'
typedef _Float16 f16;                       /* scalar/packed fp16 math   */
typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
typedef __fp16 hf16x2 __attribute__((ext_vector_type(2))); /* cvt_pkrtz result */
typedef short i16x2 __attribute__((ext_vector_type(2)));
typedef unsigned short u16x2 __attribute__((ext_vector_type(2)));
typedef int i32;
typedef unsigned u32;
typedef long i64;
typedef unsigned long u64;
#define KERNEL __attribute__((global))
EOF
}

# fields: name | expected instruction regex | mode (c = plain C source, asm =
# inline asm; "asm" entries cover opcodes that DTK clang has no builtin for) | source
#
# Two rows below are pinned to the *canonicalised* form, because DTK clang 18
# rewrites `u16x2 - {0xc000,0xc000}` into `v_pk_add_u16 x, x, 2.0` (0x4000 is
# the two's-complement delta of 0xc000), so `v_pk_sub_u16` is never emitted for
# it: the `pk_u16_sub` row therefore expects `v_pk_add_u16`, and
# `bf16_to_f16_bits` is pinned to the `v_pk_lshlrev_b16` that carries the
# op_sel_hi lane-selection trap this row exists to document.  Neither path is
# used by the kernels in csrc/rocm/gfx906_emu_common.cuh (they use the 32-bit
# scalar per-lane form).  Note the regex field cannot contain `|` -- that is the
# table's column separator.
CG_TABLE=$(cat <<'EOF'
sdot4|v_dot4_i32_i8|c|KERNEL void k_sdot4(i32 *o, const i32 *a, const i32 *b) { o[0] = __builtin_amdgcn_sdot4(a[0], b[0], o[0], 0); }
udot4|v_dot4_u32_u8|c|KERNEL void k_udot4(u32 *o, const u32 *a, const u32 *b) { o[0] = __builtin_amdgcn_udot4(a[0], b[0], o[0], 0); }
sdot2|v_dot2_i32_i16|c|KERNEL void k_sdot2(i32 *o, const i16x2 *a, const i16x2 *b) { o[0] = __builtin_amdgcn_sdot2(a[0], b[0], o[0], 0); }
udot2|v_dot2_u32_u16|c|KERNEL void k_udot2(u32 *o, const u16x2 *a, const u16x2 *b) { o[0] = __builtin_amdgcn_udot2(a[0], b[0], o[0], 0); }
sdot8|v_dot8_i32_i4|c|KERNEL void k_sdot8(i64 *o, const i64 *a, const i64 *b) { o[0] = __builtin_amdgcn_sdot8(a[0], b[0], o[0], 0); }
udot8|v_dot8_u32_u4|c|KERNEL void k_udot8(u64 *o, const u64 *a, const u64 *b) { o[0] = __builtin_amdgcn_udot8(a[0], b[0], o[0], 0); }
fdot2|v_dot2_f32_f16|c|KERNEL void k_fdot2(float *o, const f16x2 *a, const f16x2 *b) { o[0] = __builtin_amdgcn_fdot2(a[0], b[0], o[0], 0); }
mqsad_pk|v_mqsad_pk_u16_u8|c|KERNEL void k_mqsad(u64 *o, const u64 *a, const u32 *b) { o[0] = __builtin_amdgcn_mqsad_pk_u16_u8(a[0], b[0], o[0]); }
cvt_pkrtz|v_cvt_pkrtz_f16_f32|c|KERNEL void k_cvt_pkrtz(hf16x2 *o, float a, float b) { *o = __builtin_amdgcn_cvt_pkrtz(a, b); }
cvt_pknorm_i16|v_cvt_pknorm_i16_f32|c|KERNEL void k_pknorm(i16x2 *o, float a, float b) { *o = __builtin_amdgcn_cvt_pknorm_i16(a, b); }
cvt_pk_u8_f32|v_cvt_pk_u8_f32|c|KERNEL void k_cvt_pku8(u32 *o, float a, u32 b) { o[0] = __builtin_amdgcn_cvt_pk_u8_f32(a, 0, b); }
pk_fma_f16|v_pk_fma_f16|c|KERNEL void k_pkfma(f16x2 *o, const f16x2 *a, const f16x2 *b) { o[0] = a[0] * b[0] + o[0]; }
pk_add_f16|v_pk_add_f16|c|KERNEL void k_pkadd(f16x2 *o, const f16x2 *a) { o[0] = o[0] + a[0]; }
pk_mul_f16|v_pk_mul_f16|c|KERNEL void k_pkmul(f16x2 *o, const f16x2 *a) { o[0] = o[0] * a[0]; }
fma_mix_f32|v_fma_mix_f32|c|KERNEL void k_mix(float *o, f16 a, float b) { o[0] = o[0] + (float)a * b; }
fmac_f32|v_fmac_f32|c|KERNEL void k_fmac(float *o, const float *a, const float *b) { o[0] = a[0] * b[0] + o[0]; }
fmac_f32_builtin|v_fmac_f32|c|KERNEL void k_fmac_b(float *o, const float *a, const float *b) { o[0] = __builtin_fmaf(a[0], b[0], o[0]); }
fma_f32_asm|v_fma_f32|asm|KERNEL void k_fma32(float *o, const float *a, const float *b) { float r; __asm__("v_fma_f32 %0, %1, %2, %3" : "=v"(r) : "v"(a[0]), "v"(b[0]), "v"(o[0])); o[0] = r; }
mad_i32_i24_asm|v_mad_i32_i24|asm|KERNEL void k_mad24(i32 *o, const i32 *a, const i32 *b) { i32 r; __asm__("v_mad_i32_i24 %0, %1, %2, %3" : "=v"(r) : "v"(a[0]), "v"(b[0]), "v"(o[0])); o[0] = r; }
fma_mix_f32_asm|v_fma_mix_f32|asm|KERNEL void k_mixasm(float *o, const i32 *a, const i32 *b) { float r; __asm__("v_fma_mix_f32 %0, %1, %2, %3 op_sel:[0,1,0] op_sel_hi:[1,0,0]" : "=v"(r) : "v"(a[0]), "v"(b[0]), "v"(o[0])); o[0] = r; }
pk_u16_add|v_pk_add_u16|c|KERNEL void k_pkaddu16(u16x2 *o, const u16x2 *a) { o[0] = a[0] + (u16x2){0x2000, 0x2000}; }
pk_u16_sub|v_pk_add_u16|c|KERNEL void k_pksubu16(u16x2 *o, const u16x2 *a) { o[0] = a[0] - (u16x2){0xc000, 0xc000}; }
pk_u16_shl|v_pk_lshlrev_b16|c|KERNEL void k_pkshlu16(u16x2 *o, const u16x2 *a) { o[0] = a[0] << 7; }
e5m2_to_f16_bits|v_pk_lshlrev_b16|c|KERNEL void k_e5m2(u16x2 *o, const u16x2 *a) { o[0] = a[0] << 8; }
e4m3_to_f16_bits|v_pk_add_u16|c|KERNEL void k_e4m3(u16x2 *o, const u16x2 *a) { u16x2 s = a[0] & (u16x2){0x8080, 0x8080}; u16x2 m = a[0] & (u16x2){0x7f7f, 0x7f7f}; o[0] = ((m << 7) + (u16x2){0x2000, 0x2000}) | s; }
bf16_to_f16_bits|v_pk_lshlrev_b16|c|KERNEL void k_bf16s(u16x2 *o, const u16x2 *a) { o[0] = (a[0] - (u16x2){0x3800u, 0x3800u}) << 3; }
bf16_to_f32_shift|v_lshlrev_b32|c|KERNEL void k_bf2f32(float *o, const u32 *a) { o[0] = __builtin_bit_cast(float, a[0] << 16); }
f32_to_bf16_round|v_add_u32|c|KERNEL void k_f2bf16(u32 *o, const float *a) { o[0] = __builtin_bit_cast(u32, a[0]) + 0x8000; }
dot2_after_convert|v_dot2_f32_f16|c|KERNEL void k_dot2(float *o, const u16x2 *a) { f16x2 f = *(const f16x2 *)a; o[0] = __builtin_amdgcn_fdot2(f, f, o[0], 0); }
EOF
)

echo "----------------------------------------------------------------------"
echo "Phase 3: codegen (clang -target amdgcn-amd-amdhsa -mcpu=${MC} -O3 -S)"
echo "----------------------------------------------------------------------"
: > "$CG_RESULTS"
while IFS='|' read -r name want mode src; do
  [ -z "${name:-}" ] && continue
  f="$WORK/c_${name}.c"
  { codegen_src; echo "$src"; } > "$f"
  asm="$WORK/c_${name}.s"
  err="$("$CLANG" -target amdgcn-amd-amdhsa -mcpu="$MC" -O3 -S "$f" -o "$asm" 2>&1)"
  if [ ! -s "$asm" ]; then
    printf 'ERROR   %-18s %s\n' "$name" "$(echo "$err" | head -1 | cut -c1-90)"
    record "$CG_RESULTS" ERROR "$name" "$(echo "$err" | head -1 | cut -c1-120)"
    mis=$((mis + 1))
    continue
  fi
  n=$(grep -cE "$want" "$asm")
  line=$(grep -m1 -E "$want" "$asm" | sed 's/^ *//')
  if [ "$n" -gt 0 ]; then
    printf 'EMITTED %-18s %-22s x%-3s %s\n' "$name" "$want" "$n" "${line:0:52}"
    record "$CG_RESULTS" EMITTED "$name" "$want x$n [$mode] :: $line"
  else
    got=$(grep -oE '^\s+v_[a-z0-9_]+' "$asm" | sed 's/^ *//' | sort | uniq -c | sort -rn | head -3 | tr '\n' ';')
    printf 'MISSING %-18s %-22s (top: %s)\n' "$name" "$want" "$got"
    record "$CG_RESULTS" MISSING "$name" "$want [$mode] :: top=$got"
    mis=$((mis + 1))
  fi
done <<< "$CG_TABLE"
echo "-> $CG_RESULTS"
echo

# --------------------------------------------------------------------------
# Phase 3b: the same through the real `hipcc -c --offload-arch=gfx906 -S`
# --------------------------------------------------------------------------
HIP_SRC="$WORK/hip_probe.hip"
cat > "$HIP_SRC" <<'EOF'
#include <hip/hip_runtime.h>
typedef _Float16 f16;
typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
extern "C" __global__ void k_sdot4(const int* a, const int* b, int* o) {
  o[threadIdx.x] = __builtin_amdgcn_sdot4(a[threadIdx.x], b[threadIdx.x], o[threadIdx.x], false);
}
extern "C" __global__ void k_fdot2(const f16x2* a, const f16x2* b, float* o) {
  o[threadIdx.x] = __builtin_amdgcn_fdot2(a[threadIdx.x], b[threadIdx.x], o[threadIdx.x], false);
}
extern "C" __global__ void k_pkfma(f16x2* o, const f16x2* a, const f16x2* b) {
  o[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x] + o[threadIdx.x];
}
extern "C" __global__ void k_mix(float* o, const f16* a, const float* b) {
  o[threadIdx.x] = fmaf((float)a[threadIdx.x], b[threadIdx.x], o[threadIdx.x]);
}
EOF
if [ -n "${HIPCC:-}" ] && [ "$HIPCC" != "skip" ]; then
  echo "----------------------------------------------------------------------"
  echo "Phase 3b: hipcc -S --cuda-device-only --offload-arch=${MC} -O3"
  echo "----------------------------------------------------------------------"
  HIP_ASM="$OUTDIR/hipcc_device.s"
  hip_err="$("$HIPCC" -S --cuda-device-only --offload-arch="$MC" -O3 "$HIP_SRC" -o "$HIP_ASM" 2>&1)"
  if [ -s "$HIP_ASM" ]; then
    echo "-> $HIP_ASM"
    for pair in "sdot4:v_dot4_i32_i8" "fdot2:v_dot2_f32_f16" "pkfma:v_pk_fma_f16" "mix:v_fma_mix_f32"; do
      nm="${pair%%:*}"; ins="${pair##*:}"
      printf '  %-8s %-18s x%s\n' "$nm" "$ins" "$(grep -cE "$ins" "$HIP_ASM")"
    done
    grep -E 'v_fma_mix_f32' "$HIP_ASM" | head -2 | sed 's/^/    /'
  else
    echo "hipcc device compile unavailable in this environment (skipped):"
    echo "$hip_err" | head -3 | sed 's/^/    /'
  fi
  echo
fi

# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
echo "======================================================================"
echo "summary (${ENV_TAG})"
echo "======================================================================"
printf 'assembler : OK=%s FAIL=%s SYNTAX=%s\n' \
  "$(awk -F'\t' '$1=="OK"' "$ASM_RESULTS" | wc -l)" \
  "$(awk -F'\t' '$1=="FAIL"' "$ASM_RESULTS" | wc -l)" \
  "$(awk -F'\t' '$1=="SYNTAX"' "$ASM_RESULTS" | wc -l)"
printf 'builtins  : HAS=1 %s / HAS=0 %s\n' \
  "$(awk -F'\t' '$1=="1"' "$BLT_RESULTS" | wc -l)" \
  "$(awk -F'\t' '$1=="0"' "$BLT_RESULTS" | wc -l)"
printf 'codegen   : EMITTED=%s MISSING=%s ERROR=%s\n' \
  "$(awk -F'\t' '$1=="EMITTED"' "$CG_RESULTS" | wc -l)" \
  "$(awk -F'\t' '$1=="MISSING"' "$CG_RESULTS" | wc -l)" \
  "$(awk -F'\t' '$1=="ERROR"' "$CG_RESULTS" | wc -l)"
echo

# Baseline expectations for gfx906 (the decision table of the S11d archive).
# Format: <kind> <name> <expected-status>
BASELINE=$(cat <<'EOF'
asm INT8-dot4-signed OK
asm INT8-dot4-unsigned OK
asm INT8-dot8-signed OK
asm FP16-dot2 OK
asm FP16-pk-fma OK
asm CVT-pkrtz OK
asm MIX-mad-mix-f32 FAIL
asm MIX-fma-mix-f32 OK
asm FP8-cvt-f32-fp8 FAIL
asm BF16-cvt-f32-bf16 FAIL
asm MFMA-f16 FAIL
blt __builtin_amdgcn_sdot4 1
blt __builtin_amdgcn_fdot2 1
blt __builtin_amdgcn_sdot8 1
blt __builtin_amdgcn_mad_mix 0
blt __builtin_amdgcn_fma_mix_f32 0
cgo sdot4 EMITTED
cgo fdot2 EMITTED
cgo pk_fma_f16 EMITTED
cgo fma_mix_f32 EMITTED
asm FP8EMU-pk-sub-u16 OK
asm FP8EMU-pk-min-f16 OK
asm FP8EMU-pk-max-f16 OK
asm FP8EMU-pack-b32-f16 OK
asm FP8EMU-perm-b32 OK
asm FP8EMU-lshl-or-b32 OK
asm FP8EMU-or3-b32 OK
asm FP8EMU-alignbit-b32 OK
asm FP8EMU-bfe-u32 OK
asm FP8EMU-bfe-i32 OK
asm FP8EMU-rndne-f32 OK
asm FP8EMU-cvt-pk-u8-f32 OK
asm FP8EMU-sat-pk-u8-i16 OK
asm FP8EMU-ds-read-u16 OK
asm FP8EMU-ds-read-b16 FAIL
asm FP8EMU-ds-bpermute OK
asm FP8EMU-ds-permute OK
asm FP8EMU-dpp-quad OK
asm FP8EMU-sdwa-byte0 OK
asm FP8EMU-cvt-pk-f32-fp8 FAIL
asm FP8EMU-cvt-pk-bf16-f32 FAIL
blt __builtin_amdgcn_perm 1
blt __builtin_amdgcn_cvt_pk_u8_f32 1
cgo pk_u16_add EMITTED
cgo pk_u16_sub EMITTED
cgo pk_u16_shl EMITTED
cgo e5m2_to_f16_bits EMITTED
cgo e4m3_to_f16_bits EMITTED
cgo bf16_to_f16_bits EMITTED
cgo bf16_to_f32_shift EMITTED
cgo f32_to_bf16_round EMITTED
cgo dot2_after_convert EMITTED
EOF
)
bad=0
while read -r kind name want; do
  [ -z "${kind:-}" ] && continue
  case "$kind" in
    asm) got="$(awk -F'\t' -v n="$name" '$2==n{print $1}' "$ASM_RESULTS")" ;;
    blt) got="$(awk -F'\t' -v n="$name" '$2==n{print $1}' "$BLT_RESULTS")" ;;
    cgo) got="$(awk -F'\t' -v n="$name" '$2==n{print $1}' "$CG_RESULTS")" ;;
  esac
  if [ "$got" != "$want" ]; then
    printf 'BASELINE-MISMATCH %-4s %-24s want=%s got=%s\n' "$kind" "$name" "$want" "${got:-<none>}"
    bad=$((bad + 1))
  fi
done <<< "$BASELINE"
if [ "$bad" -eq 0 ]; then
  echo "baseline  : MATCH (all ${ISA_PROBE_BASELINE_N:-53} expectations hold)"
else
  echo "baseline  : $bad MISMATCH(es)"
fi
echo
echo "artifacts : $ASM_RESULTS / $BLT_RESULTS / $CG_RESULTS"

# strict mode: fail if the gfx906 baseline moved, or if a probe line/compile
# itself broke (assembler FAIL verdicts are expected and never counted here).
if [ "${ISA_PROBE_STRICT:-0}" = "1" ] &&
  { [ "$bad" -ne 0 ] || [ "$mis" -ne 0 ] || [ "$syn" -ne 0 ]; }; then
  exit 1
fi
exit 0
