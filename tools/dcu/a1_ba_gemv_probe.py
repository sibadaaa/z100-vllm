#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A1 probe: GDN ``in_proj_ba`` (N=24, K=5120) at the k=7 verify width.

Production dispatch (workspace tree, ``_gfx906_gemv_shape_ok``) allows the
bf16/fp16 emu GEMV at ``n_out<=1280`` only up to M=4, so at the DFlash2 k=7
verify width (M=8) the layer falls to ``triton_matmul`` (~217 us/call x 48
layers/step in the graph-A trace = ~10.4 ms/step at ~0.57 GB/s of weight
bytes).  This probe times the candidates on the real shape and checks numerics
against an fp64 reference:

* ``torch.ops._rocm_C.bf16_emu_gemv`` with nt=128 (the workspace tree's
  narrow-N thread choice: N=24 <= 4096) and nt=64 (the runtime tree default)
* an inlined copy of the production single-config ``triton_matmul``
  (fp16 output, the current M=8 route)
* ``F.linear`` (rocBLAS)

The model runs fp16 on gfx906 (bf16 models are coerced at load, commit
0e1475163), so fp16 is the production dtype; bf16 is included for reference.

Self-contained on purpose: it does NOT ``import vllm`` (the container's
torchvision breaks the import chain); the kernel is loaded directly from the
production ``_rocm_C.abi3.so``.

Usage (idle GPU, DTK env active):

    python tools/dcu/a1_ba_gemv_probe.py [--so PATH] [--out JSON]
"""

import argparse
import json
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

N, K = 24, 5120
MS = [1, 2, 4, 5, 6, 7, 8]


def _get_autotune_config():
    return [
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
            num_stages=3,
            num_warps=2,
        ),
    ]


def _get_heuristics():
    return {"BLOCK_SIZE_M": lambda args: min(16, triton.next_power_of_2(args["M"]))}


@triton.autotune(configs=_get_autotune_config(), key=["M", "N", "K"])
@triton.heuristics(values=_get_heuristics())
@triton.jit
def _triton_matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a, b):
    """Verbatim production single-config matmul (utils.py, fp16 output)."""
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    N, _ = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _triton_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(1), b.stride(0), c.stride(0), c.stride(1),
    )
    return c


def bench(fn, warmup=20, iters=100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--so",
        default="/mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so",
        help="production _rocm_C.abi3.so to load",
    )
    ap.add_argument("--out", default="/tmp/a1-ba-gemv-probe.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU visible", file=sys.stderr)
        return 2
    torch.ops.load_library(args.so)
    assert hasattr(torch.ops._rocm_C, "bf16_emu_gemv"), "bf16_emu_gemv missing"

    torch.manual_seed(0)
    dev = torch.cuda.get_device_name(0)
    print(f"# device={dev} N={N} K={K} (weight {N * K * 2 / 1024:.1f} KiB: "
          "L2-warm, matching the 48-calls/step steady state)")
    rows = []
    worst = 0.0
    for dtype in (torch.float16, torch.bfloat16):
        for m in MS:
            w = (torch.randn(N, K, device="cuda") * 0.02).to(dtype)
            x = (torch.randn(m, K, device="cuda") * 0.5).to(dtype)
            ref = F.linear(x.to(torch.float64), w.to(torch.float64))
            scale = ref.abs().max().clamp_min(1e-30).item()

            emu = lambda nt: torch.ops._rocm_C.bf16_emu_gemv(
                x, w, None, nt, 0, False
            )
            out_emu128 = emu(128)
            out_tri = triton_matmul(x, w)
            out_lin = F.linear(x, w)
            errs = {
                "emu128": (out_emu128.to(torch.float64) - ref).abs().max().item(),
                "triton": (out_tri.to(torch.float64) - ref).abs().max().item(),
                "rocblas": (out_lin.to(torch.float64) - ref).abs().max().item(),
            }
            for d in errs.values():
                worst = max(worst, d / scale)

            ms_emu128 = bench(lambda: emu(128))
            ms_emu64 = bench(lambda: emu(64))
            ms_tri = bench(lambda: triton_matmul(x, w))
            ms_lin = bench(lambda: F.linear(x, w))
            gbs = N * K * 2 / ms_emu128 / 1e6
            rows.append({
                "dtype": str(dtype).replace("torch.", ""),
                "m": m,
                "emu128_us": ms_emu128 * 1e3,
                "emu64_us": ms_emu64 * 1e3,
                "triton_us": ms_tri * 1e3,
                "rocblas_us": ms_lin * 1e3,
                "emu128_vs_triton": ms_tri / ms_emu128,
                "emu128_gbs": gbs,
                "rel_err_emu128": errs["emu128"] / scale,
                "rel_err_triton": errs["triton"] / scale,
                "rel_err_rocblas": errs["rocblas"] / scale,
            })
            print(
                f"RESULT {dtype.__name__ if hasattr(dtype,'__name__') else dtype} "
                f"M={m} emu128={ms_emu128*1e3:7.1f}us ({gbs:6.1f} GB/s) "
                f"emu64={ms_emu64*1e3:7.1f}us triton={ms_tri*1e3:7.1f}us "
                f"rocblas={ms_lin*1e3:7.1f}us "
                f"emu128/triton={ms_tri/ms_emu128:5.2f}x "
                f"err/|max|: emu={errs['emu128']/scale:.2e} "
                f"triton={errs['triton']/scale:.2e} "
                f"rocblas={errs['rocblas']/scale:.2e}"
            )
            del x, w
            torch.cuda.empty_cache()

    print(f"# worst rel err over all cases: {worst:.3e}")
    with open(args.out, "w") as f:
        json.dump({"device": dev, "n": N, "k": K, "rows": rows}, f, indent=2)
    print(f"# json -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
