#!/usr/bin/env python3
"""Minimal gfx906 Triton capability probe (S6 gate 1).

Run with the self-built triton first on sys.path, e.g.

    PYTHONPATH=/mnt/game/triton-gfx906/install python tools/dcu/probe_triton.py

Checks, in increasing order of ISA coverage:
  1. import triton / target detection (arch must be gfx906, warp_size 64)
  2. elementwise add  -> proves the TTGIR/LLVM lowering accepts gfx906
  3. warp reduce      -> exercises TargetInfo::warpReduce (DPP broadcast path)
  4. int8 dot         -> exercises supportsVDot() (v_dot4 / sdot4 path)

Exit code 0 only if every check passes.
"""

import importlib.metadata as md
import sys

import torch
import triton
import triton.language as tl


def hdr(title):
    print("=" * 72)
    print(title)
    print("=" * 72)


@triton.jit
def addk(x_ptr, y_ptr, o_ptr, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    tl.store(o_ptr + i, tl.load(x_ptr + i, mask=m) + tl.load(y_ptr + i, mask=m), mask=m)


@triton.jit
def reducemaxk(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + i, mask=i < N, other=-float("inf"))
    tl.store(o_ptr + tl.program_id(0), tl.max(v, axis=0))


@triton.jit
def int8dotk(a_ptr, b_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b, out_dtype=tl.int32))


def main():
    rc = 0

    hdr("environment")
    print("triton.__version__       =", triton.__version__)
    print("triton.__file__          =", triton.__file__)
    try:
        print("importlib.metadata       =", md.version("triton"))
    except Exception as e:  # pragma: no cover - only for odd installs
        print("importlib.metadata       = <unavailable>", e)
    print("torch.__version__        =", torch.__version__)
    try:
        print("importlib.metadata torch =", md.version("torch"))
    except Exception as e:
        print("importlib.metadata torch = <unavailable>", e)
    from triton.runtime import driver

    tgt = driver.active.get_current_target()
    print("target                   =", tgt)
    print("device                   =", torch.cuda.get_device_name(0))
    print("gcnArchName              =", torch.cuda.get_device_properties(0).gcnArchName)
    print("triton cache dir         =", __import__("os").environ.get("TRITON_CACHE_DIR"))
    if str(tgt.arch) != "gfx906":
        print("!! FAIL: triton target arch is not gfx906")
        rc = 1

    n = 4096
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    o = torch.empty(n, device="cuda")

    hdr("check 1: elementwise add (lowering accepts gfx906)")
    addk[(4,)](x, y, o, n, BLOCK=1024)
    torch.cuda.synchronize()
    err = (o - (x + y)).abs().max().item()
    print("max_abs_err =", err)
    if err < 1e-5:
        print("TRITON_OK")
    else:
        print("!! FAIL: add mismatch")
        rc = 1

    hdr("check 2: warp-level reduce (TargetInfo::warpReduce / DPP path)")
    r = torch.empty(4, device="cuda")
    reducemaxk[(4,)](x, r, n, BLOCK=1024)
    torch.cuda.synchronize()
    ref = x.view(4, 1024).max(dim=1).values
    rerr = (r - ref).abs().max().item()
    print("max_abs_err =", rerr)
    if rerr == 0.0:
        print("REDUCE_OK")
    else:
        print("!! FAIL: reduce mismatch")
        rc = 1

    hdr("check 3: int8 dot (supportsVDot / v_dot4 path)")
    M = N = K = 32
    a = torch.randint(-4, 4, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-4, 4, (K, N), device="cuda", dtype=torch.int8)
    c = torch.zeros((M, N), device="cuda", dtype=torch.int32)
    int8dotk[(1,)](a, b, c, M=M, N=N, K=K)
    torch.cuda.synchronize()
    # torch has no int32 matmul on ROCm; reduce the outer product instead.
    cref = (a.to(torch.int32)[:, :, None] * b.to(torch.int32)[None, :, :]).sum(dim=1)
    derr = (c - cref).abs().max().item()
    print("max_abs_err =", derr)
    if derr == 0:
        print("INT8DOT_OK")
    else:
        print("!! FAIL: int8 dot mismatch")
        rc = 1
    # supportsVDot() is what lets Triton emit v_dot* (sdot4) on gfx906.
    kernel_cache = int8dotk.device_caches[torch.cuda.current_device()][0]
    handle = next(iter(kernel_cache.values())) if kernel_cache else None
    amdgcn = (handle.asm.get("amdgcn", "") if handle is not None else "")
    ndot = amdgcn.count("v_dot")
    print("v_dot occurrences in amdgcn asm =", ndot)
    print("VDOT_IN_ASM =", ndot > 0)
    if ndot == 0:
        print("!! WARN: no v_dot* instruction emitted (sdot4 path inactive)")

    hdr("result")
    print("PROBE_RC =", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
