#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""S22: sweep Triton matmul configs for the DFlash2 draft tower at M=8 (gfx906).

`s22_draft_gemm_probe.py` showed the draft tower's bf16 matmuls run through
`vllm/model_executor/layers/utils.py::triton_matmul`, whose autotune space holds
exactly ONE config (BLOCK_N=64, BLOCK_K=64, num_warps=2, num_stages=3, with
BLOCK_SIZE_M = min(16, next_pow2(M))).  At M=8 the big draft shapes reach only
~110-145 GB/s of the probe's measured 750 GB/s honest read, while rocBLAS
(`F.linear`) is 1.8-2.9x faster on 4 of the 5 shapes.

This probe re-runs the *same* kernel body with a swept config space to find out
whether the gap is a tuning problem (fixable in Python: `get_autotune_config()`)
or a kernel-design problem.

  HIP_VISIBLE_DEVICES=2 python tools/dcu/s22_draft_triton_probe.py \
      [--ms 8] [--iters 20] [--outdir /mnt/game/s22] [--topk 12]
"""

import argparse
import itertools
import json
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Identical body to utils.triton_matmul_kernel (b is [N, K], stride_bk=1).
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
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def run_cfg(a, b, c, cfg):
    M, K = a.shape
    N = b.shape[0]
    bm, bn, bk, gm, nw, ns = cfg
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=bm,
        BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=bk,
        GROUP_SIZE_M=gm,
        num_warps=nw,
        num_stages=ns,
    )


def bench(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="8")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--outdir", default="/mnt/game/s22")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    shapes = [
        (34816, 5120),
        (5120, 17408),
        (6144, 5120),
        (4096, 5120),
        (1536, 5120),
    ]
    ms_list = [int(x) for x in args.ms.split(",")]
    dt = torch.bfloat16
    # baseline = the production autotune config, BM from get_heuristics()
    cfgs = []
    for bm, bn, bk, nw, ns in itertools.product(
        (8, 16), (64, 128, 256), (64, 128), (2, 4, 8), (3, 4)
    ):
        cfgs.append((bm, bn, bk, 1, nw, ns))
    rows = []
    print(f"device={torch.cuda.get_device_name(0)} candidates={len(cfgs)}")
    for n, k in shapes:
        b = (torch.randn(n, k, device="cuda") * 0.02).to(dt)
        for m in ms_list:
            a = (torch.randn(m, k, device="cuda") * 0.5).to(dt)
            c = torch.empty((m, n), device="cuda", dtype=dt)
            base_cfg = (min(16, triton.next_power_of_2(m)), 64, 64, 1, 2, 3)
            t_base = bench(lambda: run_cfg(a, b, c, base_cfg), iters=args.iters)
            res = []
            for cfg in cfgs:
                try:
                    t = bench(lambda: run_cfg(a, b, c, cfg), iters=args.iters)
                except Exception as ex:  # noqa: BLE001
                    print(f"  skip {cfg}: {type(ex).__name__}: {ex}")
                    continue
                res.append((t, cfg))
            res.sort()
            print(f"\n--- {n}:{k} M={m}  baseline(BM{base_cfg[0]},BN64,BK64,w2,s3)"
                  f" = {t_base:.4f} ms ({n*k*2/t_base/1e6:.1f} GB/s)")
            for t, cfg in res[: args.topk]:
                print(
                    f"    BM{cfg[0]:<3}BN{cfg[1]:<4}BK{cfg[2]:<4}w{cfg[4]}s{cfg[5]}"
                    f"  {t:8.4f} ms  {n*k*2/t/1e6:6.1f} GB/s  x{t_base/t:5.2f}"
                )
                rows.append(
                    dict(shape=f"{n}:{k}", n=n, k=k, m=m, cfg=list(cfg),
                         ms=t, gbs=n * k * 2 / t / 1e6, speedup=t_base / t,
                         baseline_ms=t_base)
                )
            del a, c
        del b
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "draft_triton_cfg.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
