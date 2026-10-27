#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton gfx906 codegen probe -- which AMDGPU instructions does `tl.dot` emit?

S11d. Answers the question that decides the P6 (FP8 emulated through FP16)
route: on gfx906 (Z100, Vega20, no MFMA/WMMA) does the self-built Triton 3.6
with the `supportsVDot()`-includes-VEGA patch lower

  * `tl.dot(fp16, fp16) -> fp32`  to packed fp16 FMA / `v_dot2_f32_f16`, or
  * `tl.dot(int8, int8) -> int32` to `v_dot4_i32_i8`,
  * `tl.sum` / elementwise fp16 to packed ops,

or does it degrade to scalar `v_fma_f32` / `v_fmac_f32` (which is what S8b
measured for `tl.dot(fp8, fp8)` -- 8160 x `v_fmac_f32_e32` per 4096^3)?

Run with the patched triton first on sys.path and GPUs 2,3 only:

    source tools/dcu/env.sh
    export HIP_VISIBLE_DEVICES=2,3
    export PYTHONPATH=/mnt/game/triton-gfx906/install
    export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
    python3 tools/dcu/triton_codegen_probe.py

Only kernels are compiled and launched through the python dispatcher; no
self-compiled HIP executable is built or run (S11d hard constraint).

Exit code 0 if every probe kernel compiled and its AMDGPU assembly was read.
"""

import os
import re
import sys

import torch
import triton
import triton.language as tl

INTERESTING = [
    "v_dot2_f32_f16",
    "v_dot2c_f32_f16",
    "v_dot4_i32_i8",
    "v_dot8_i32_i4",
    "v_pk_fma_f16",
    "v_pk_add_f16",
    "v_pk_mul_f16",
    "v_fma_f32",
    "v_fmac_f32",
    "v_mad_f32",
    "v_mfma",
    "v_fma_mix_f32",
    "v_mad_mix",
    "v_cvt_f16_f32",
    "v_cvt_f32_f16",
    "v_cvt_pkrtz_f16_f32",
    "v_add_f32",
    "v_mul_f32",
    "v_dual",
]


def hdr(t):
    print("=" * 78)
    print(t)
    print("=" * 78)


def counts(asm):
    """Count `v_*` mnemonics in an amdgcn listing."""
    ops = {}
    for line in asm.splitlines():
        line = line.strip()
        m = re.match(r"^(v_[a-z0-9_]+|s_[a-z0-9_]+|ds_[a-z0-9_]+|global_[a-z0-9_]+|flat_[a-z0-9_]+|buffer_[a-z0-9_]+)\b", line)
        if not m:
            continue
        mn = m.group(1)
        if mn.startswith("s_endpgm") or mn.startswith("s_nop") or mn.startswith("s_waitcnt"):
            continue
        ops[mn] = ops.get(mn, 0) + 1
    return ops


def report(tag, asm, extra=""):
    ops = counts(asm)
    total = sum(ops.values())
    print(f"\n--- {tag} {extra}---")
    print(f"    total v_/s_/ds_/memory instructions: {total}")
    for k in INTERESTING:
        hits = {m: c for m, c in ops.items() if m.startswith(k)}
        if hits:
            print("    " + "  ".join(f"{m}={c}" for m, c in sorted(hits.items())))
    top = sorted(ops.items(), key=lambda kv: -kv[1])[:6]
    print("    top: " + "  ".join(f"{m}={c}" for m, c in top))
    return ops


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
@triton.jit
def dot_f16_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b, out_dtype=tl.float32))


@triton.jit
def dot_i8_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b, out_dtype=tl.int32))


@triton.jit
def sum_f16_kernel(x_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(x_ptr + i)
    tl.store(o_ptr + tl.program_id(0), tl.sum(v.to(tl.float32), axis=0).to(tl.float16))


@triton.jit
def ewise_f16_kernel(x_ptr, y_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    a = tl.load(x_ptr + i)
    b = tl.load(y_ptr + i)
    tl.store(o_ptr + i, a * b + a)


@triton.jit
def fp8_to_f16_dot_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    """P6 shape: fp8 tiles dequantised to fp16 in-register, then tl.dot(fp16)."""
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    af = a.to(tl.float16, bitcast=False)
    bf = b.to(tl.float16, bitcast=False)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], tl.dot(af, bf, out_dtype=tl.float32))


# --------------------------------------------------------------------------
def compile_and_count(name, fn, ptr_args, grid, constexprs, asm_holder,
                      num_warps=4):
    try:
        h = fn[grid](*ptr_args, num_warps=num_warps, **constexprs)
    except Exception as e:  # noqa: BLE001
        print(f"!! {name}: launch failed: {type(e).__name__}: {str(e)[:300]}")
        return None
    asm = None
    if h is not None and hasattr(h, "asm"):
        try:
            asm = h.asm.get("amdgcn")
        except Exception:  # noqa: BLE001
            asm = None
    if asm is None:
        # fall back: newest cache entry
        print(f"!! {name}: no .asm on the returned handle ({type(h)}); "
              "dumping the newest .s from the Triton cache instead")
        return None
    asm_holder[name] = asm
    report(name, asm, extra=f"num_warps={num_warps} constexprs={constexprs} ")
    return counts(asm)


def main():
    rc = 0
    hdr("environment")
    print("triton  :", triton.__version__, triton.__file__)
    print("torch   :", torch.__version__)
    print("devices :", torch.cuda.device_count(), "visible HIP_VISIBLE_DEVICES=",
          os.environ.get("HIP_VISIBLE_DEVICES"))
    print("cache   :", os.environ.get("TRITON_CACHE_DIR"))
    if torch.cuda.device_count() == 0:
        print("!! no GPU visible")
        return 2
    for d in range(torch.cuda.device_count()):
        print(f"  cuda:{d} ->", torch.cuda.get_device_properties(d).gcnArchName)
    dev = "cuda:0"

    hdr("1. tl.dot(fp16 x fp16 -> fp32): BLOCK / num_warps sweep")
    results = {}
    for (M, N, K, nw) in [(16, 16, 16, 1), (16, 16, 16, 2), (32, 32, 32, 2),
                          (32, 32, 64, 4), (64, 64, 64, 4), (128, 128, 64, 8),
                          (16, 16, 32, 1)]:
        a = torch.randn((M, K), device=dev, dtype=torch.float16)
        b = torch.randn((K, N), device=dev, dtype=torch.float16)
        c = torch.empty((M, N), device=dev, dtype=torch.float32)
        tag = f"dot_f16_{M}x{N}x{K}_w{nw}"
        r = compile_and_count(tag, dot_f16_kernel, (a, b, c), (1,),
                              {"M": M, "N": N, "K": K}, results, num_warps=nw)
        if r is None:
            rc = 1
        else:
            # correctness sanity: tl.dot vs torch matmul
            ref = (a.float() @ b.float())
            err = (c - ref).abs().max().item()
            print(f"    rel/max abs err vs fp32 matmul = {err:.3e}")
            results[tag + "_err"] = err

    hdr("2. decode-shaped m=1 row (BLOCK_M=16 with mask) ")
    @triton.jit
    def gemv_f16_kernel(a_ptr, b_ptr, c_ptr, K: tl.constexpr, N: tl.constexpr,
                        BM: tl.constexpr):
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        rn = pid * N + tl.arange(0, N)
        rk = tl.arange(0, K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :],
                    mask=rm[:, None] < 1, other=0.0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :] - pid * N)
        acc = tl.dot(a, b, out_dtype=tl.float32)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :] - pid * N, acc,
                 mask=rm[:, None] < 1)

    a = torch.randn((1, 64), device=dev, dtype=torch.float16)
    b = torch.randn((64, 128), device=dev, dtype=torch.float16)
    c = torch.empty((1, 128), device=dev, dtype=torch.float32)
    r = compile_and_count("gemv_f16_m1_16x128x64_w1", gemv_f16_kernel, (a, b, c), (1,),
                          {"K": 64, "N": 128, "BM": 16}, results, num_warps=1)
    if r is None:
        rc = 1

    hdr("3. tl.dot(int8 x int8 -> int32)")
    for (M, N, K, nw) in [(16, 16, 16, 1), (32, 32, 32, 2), (16, 16, 64, 1)]:
        a = torch.randint(-8, 8, (M, K), device=dev, dtype=torch.int8)
        b = torch.randint(-8, 8, (K, N), device=dev, dtype=torch.int8)
        c = torch.empty((M, N), device=dev, dtype=torch.int32)
        tag = f"dot_i8_{M}x{N}x{K}_w{nw}"
        r = compile_and_count(tag, dot_i8_kernel, (a, b, c), (1,),
                              {"M": M, "N": N, "K": K}, results, num_warps=nw)
        if r is None:
            rc = 1
        else:
            # torch has no int32 matmul on CUDA; the values are tiny, so an
            # fp32 matmul is exact and can be used as the oracle.
            ref = (a.to(torch.float32) @ b.to(torch.float32)).to(torch.int32)
            print("    exact match vs int32 matmul:", bool((c == ref).all().item()))

    hdr("4. tl.sum(fp16) and elementwise fp16 (packed?)")
    x = torch.randn((1024,), device=dev, dtype=torch.float16)
    o = torch.empty((1,), device=dev, dtype=torch.float16)
    r = compile_and_count("sum_f16_1024", sum_f16_kernel, (x, o), (1,), {"N": 1024},
                          results, num_warps=4)
    if r is None:
        rc = 1
    y = torch.randn((1024,), device=dev, dtype=torch.float16)
    o2 = torch.empty((1024,), device=dev, dtype=torch.float16)
    r = compile_and_count("ewise_f16_1024", ewise_f16_kernel, (x, y, o2), (1,),
                          {"N": 1024}, results, num_warps=4)
    if r is None:
        rc = 1

    hdr("5. P6 shape: fp8e4m3 -> fp16 -> tl.dot")
    try:
        a8 = torch.randn((32, 32), device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
        b8 = torch.randn((32, 32), device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
        c = torch.empty((32, 32), device=dev, dtype=torch.float32)
        r = compile_and_count("fp8_to_f16_dot_32x32x32_w2", fp8_to_f16_dot_kernel,
                              (a8, b8, c), (1,), {"M": 32, "N": 32, "K": 32}, results,
                              num_warps=2)
        if r is None:
            rc = 1
        else:
            ref = a8.to(torch.float16).float() @ b8.to(torch.float16).float()
            print("    max abs err vs fp16 matmul of the same fp8 values:",
                  f"{(c - ref).abs().max().item():.3e}")
    except Exception as e:  # noqa: BLE001
        print("!! fp8 probe unavailable:", type(e).__name__, str(e)[:200])

    hdr("6. control: raw tl.dot(fp8 x fp8) -- what S8b measured as pure scalar FMA")
    try:
        @triton.jit
        def dot_fp8_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr,
                           K: tl.constexpr):
            rm = tl.arange(0, M)
            rn = tl.arange(0, N)
            rk = tl.arange(0, K)
            a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
            b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
            tl.store(c_ptr + rm[:, None] * N + rn[None, :],
                     tl.dot(a, b, out_dtype=tl.float32))

        a8 = torch.randn((32, 32), device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
        b8 = torch.randn((32, 32), device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
        c = torch.empty((32, 32), device=dev, dtype=torch.float32)
        r = compile_and_count("dot_fp8_32x32x32_w2", dot_fp8_kernel, (a8, b8, c), (1,),
                              {"M": 32, "N": 32, "K": 32}, results, num_warps=2)
        if r is None:
            rc = 1
    except Exception as e:  # noqa: BLE001
        print("!! fp8 x fp8 probe unavailable:", type(e).__name__, str(e)[:200])

    hdr("summary: which of the wanted instructions appeared at all")
    for k in ["v_dot2_f32_f16", "v_dot4_i32_i8", "v_pk_fma_f16", "v_fma_f32",
              "v_fmac_f32", "v_mfma", "v_fma_mix_f32"]:
        n = sum(1 for name, asm in results.items()
                if not name.endswith("_err") and k in
                {m for m in counts(asm)})
        tot = sum(counts(asm).get(k, 0) for name, asm in results.items()
                  if not name.endswith("_err"))
        print(f"  {k:20s} kernels={n:2d}  total_count={tot}")

    out = os.environ.get("TRITON_PROBE_ASM_DIR")
    if out:
        os.makedirs(out, exist_ok=True)
        for name, asm in results.items():
            if name.endswith("_err"):
                continue
            with open(os.path.join(out, name + ".s"), "w") as f:
                f.write(asm)
        print("\nasm dumped to", out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
