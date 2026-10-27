#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Draft M=7 kernel sweep: the DFlash2 draft phase costs ~33.8 ms/round
(docs/result/z100-five-phase-profile.md), the largest single block after the
fp8 GEMMs, and none of its kernels has ever been microbenched.  This tool
measures every draft matmul class at the real per-rank shapes (TP=4, fp16 --
gfx906 coerces bf16 to fp16 at load) with the three production candidates:

* ``triton_matmul`` (the single-config kernel the dispatch currently takes at
  M=7 for n<=16 tokens, and everywhere above the bf16-GEMV gate cells)
* ``torch.ops._rocm_C.bf16_emu_gemv`` (nt=128)
* ``F.linear`` (rocBLAS)

Draft shapes (``/mnt/game/models/Qwen3.8-27B-DFlash2-ms/config.json``, TP=4):
5 layers x {qkv 1536x5120, o 1024x5120, gateup 8704x5120, down 1280x17408},
lm_head 62080x5120 (full vocab shard), selector hidden_proj 256x5120,
grouped-conv kernel_projection 1280x5120 (Replicated).

Usage (idle GPU):

    python tools/dcu/draft_m7_sweep.py [--out docs/perf/data/draft-m7-sweep.json]
"""

import argparse
import json
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# (name, N, K, calls/round from the graph-A trace attribution)
SHAPES = [
    ("draft.qkv", 1536, 5120, 5),
    ("draft.o", 1024, 5120, 5),
    ("draft.gateup", 8704, 5120, 5),
    ("draft.down", 1280, 17408, 5),
    ("draft.lm_head", 62080, 5120, 1),
    ("selector.hidden_proj", 256, 5120, 1),
    ("conv.kernel_proj", 1280, 5120, 2),
]
M = 7  # DFlash2 block: 1 verified prefix token + 7 masked draft tokens


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
    assert a.shape[1] == b.shape[1] and a.is_contiguous() and b.is_contiguous()
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


def bench(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so",
                    default="/home/lyh/z100-vllm/vllm/_rocm_C.abi3.so")
    ap.add_argument("--out",
                    default="docs/perf/data/draft-m7-sweep.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU visible", file=sys.stderr)
        return 2
    torch.ops.load_library(args.so)
    assert hasattr(torch.ops._rocm_C, "bf16_emu_gemv")

    torch.manual_seed(0)
    dev = torch.cuda.get_device_name(0)
    print(f"# device={dev} dtype=fp16 M={M} (per-rank shapes, TP=4)")
    print(f"# {'op':<22} {'wMB':>7} {'tri_us':>8} {'emu_us':>8} {'roc_us':>8} "
          f"{'best':>7} {'best_GBs':>8}  calls/round -> ms/round(best)")
    rows = []
    for name, n, k, calls in SHAPES:
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.float16)
        x = (torch.randn(M, k, device="cuda") * 0.5).to(torch.float16)
        wmb = n * k * 2 / 1e6
        iters = 100 if wmb < 100 else 30

        ms_tri = bench(lambda: triton_matmul(x, w), iters)
        try:
            ms_emu = bench(
                lambda: torch.ops._rocm_C.bf16_emu_gemv(x, w, None, 128, 0, False),
                iters,
            )
        except RuntimeError:
            ms_emu = float("nan")  # M=7 fits kMsMax=8, so this should not fire
        ms_roc = bench(lambda: F.linear(x, w), iters)

        cand = {"triton": ms_tri, "emu": ms_emu, "rocblas": ms_roc}
        best_name = min(cand, key=lambda s: cand[s])
        best = cand[best_name]
        gbs = wmb * 1e6 / (best * 1e-6) / 1e9
        rows.append({
            "op": name, "n": n, "k": k, "calls_per_round": calls, "w_mb": wmb,
            "triton_us": ms_tri, "emu_us": ms_emu, "rocblas_us": ms_roc,
            "best": best_name, "best_us": best, "best_gbs": gbs,
        })
        print(f"# {name:<22} {wmb:7.1f} {ms_tri:8.1f} {ms_emu:8.1f} "
              f"{ms_roc:8.1f} {best_name:>7} {gbs:8.1f}  "
              f"{calls} -> {best * calls / 1e3:7.3f}")
        del x, w
        torch.cuda.empty_cache()

    total_cur = sum(r["triton_us"] * r["calls_per_round"] for r in rows) / 1e3
    total_best = sum(r["best_us"] * r["calls_per_round"] for r in rows) / 1e3
    print(f"# draft matmul total (current=triton): {total_cur:.2f} ms/round -> "
          f"best-per-op: {total_best:.2f} ms/round  (delta {-total_cur + total_best:+.2f})")
    with open(args.out, "w") as f:
        json.dump({"device": dev, "m": M, "rows": rows,
                   "total_current_ms": total_cur, "total_best_ms": total_best},
                  f, indent=2)
    print(f"# json -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
