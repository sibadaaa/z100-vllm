#!/usr/bin/env python3
"""kernel_microbench.py — z100 rocm101 内核级微基准（空闲 GPU 上跑）。

目的：在**不影响服务**的前提下，用服务同一套内核打点：
  1. fp8-emu W8A16 decode 路径（torch.ops._rocm_C.fp8_emu_gemm_block_w8a16，
     M=8 走 decode GEMV tile、M=64 走 GEMM tile —— 直接支撑
     VLLM_DCU_FP8_GEMV_MAX_M 的调优决策）
  2. bf16 GEMM M 扫描（prefill 主路径，rocBLAS tensile 内核）
  3. GQA sdpa（decode q=1 vs prefill q=4096）
  4. D2D 带宽（HBM 基线）+ elementwise

每个相位用 time.monotonic_ns 记录窗口（与 rocpd 时间戳同源），并带 flops/bytes
元数据，供 kernel_trace_report / tune_report 换算 TFLOP/s 与 GB/s。

用法（一般由 kernel_microbench.sh 包着 rocprofv3 跑）:
  kernel_microbench.py --model-config CONFIG.json --windows-out OUT.json \
                       [--iters N] [--device 0]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.autograd.profiler import record_function

import vllm._rocm_C  # noqa: F401,E402  注册 torch.ops._rocm_C.*（须在 torch 之后）

# --- 服务投影形状（Qwen3.8-27B: hidden 5120, 24q/4kv, head 256, inter 17408）---
H = 5120
QKV_N = (24 + 4 + 4) * 256   # 8192
O_N = 24 * 256               # 6144 -> 5120
FFN_N = 2 * 17408            # 34816
FFN_K = 17408
VOCAB = 248320
HEADS, KV_HEADS, HDIM = 24, 4, 256
BLOCK_N = BLOCK_K = 128
GEMV_THREADS = 64

PROJS = [  # (name, K, N)
    ("qkv", H, QKV_N),
    ("o", O_N, H),
    ("gateup", H, FFN_N),
    ("down", FFN_K, H),
]


def win(name: str, t0: int, t1: int, flops: int = 0, bytes_: int = 0,
        windows: dict | None = None):
    windows[name] = {"t0_mono_ns": t0, "t1_mono_ns": t1,
                     "flops": flops, "bytes": bytes_}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default="/mnt/game/models/Qwen3.8-27B-FP8/config.json")
    ap.add_argument("--windows-out", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="迭代数缩放")
    args = ap.parse_args()

    dev = torch.device("cuda", args.device)
    torch.cuda.set_device(dev)
    windows: dict[str, dict] = {}
    S = args.scale

    # ---- lut / 权重 ----------------------------------------------------------
    lut = torch.arange(256, dtype=torch.float16, device=dev) / 128.0 - 1.0
    fp8w = {n: (torch.randn(N, K, device=dev) * 0.02).clamp(-1, 1).to(torch.float8_e4m3fn)
            for n, K, N in PROJS}
    w_s = {n: torch.rand(N // BLOCK_N, K // BLOCK_K, device=dev) * 0.01 + 0.005
           for n, K, N in PROJS}
    a_fp8 = torch.full((8192, H), 0.2, device=dev).to(torch.float8_e4m3fn)
    a_s_blk = torch.rand(8192, H // BLOCK_K, device=dev) * 0.01 + 0.005

    def w8a16_call(m: int, iters: int):
        for _ in range(iters):
            for n, K, N in PROJS:
                a = torch.randn(m, K, dtype=torch.float16, device=dev) * 0.1
                torch.ops._rocm_C.fp8_emu_gemm_block_w8a16(
                    a, fp8w[n], w_s[n], lut, None, torch.float16,
                    GEMV_THREADS, 0, BLOCK_N, BLOCK_K)
        torch.cuda.synchronize()

    def gemm_bf16_call(m: int, iters: int):
        for _ in range(iters):
            for n, K, N in PROJS:
                a = torch.randn(m, K, dtype=torch.bfloat16, device=dev)
                w = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
                a @ w.t()
        torch.cuda.synchronize()

    def flops_proj(m: int, iters: int) -> int:
        return sum(2 * m * K * N for _n, K, N in PROJS) * iters

    def bytes_w8a16(m: int, iters: int) -> int:
        # 权重 fp8(N*K) + scale + 激活/输出（近似，重在量级）
        b = 0
        for n, K, N in PROJS:
            b += N * K + (N // 128) * (K // 128) * 4 + m * K * 2 + m * N * 2
        return b * iters

    # ---- warmup（不入窗） ----------------------------------------------------
    gemm_bf16_call(8, 3)
    w8a16_call(8, 2)
    torch.cuda.synchronize()

    def gemm_fp16_call(m: int, iters: int):
        for _ in range(iters):
            for n, K, N in PROJS:
                a = torch.randn(m, K, dtype=torch.float16, device=dev)
                w = torch.randn(N, K, dtype=torch.float16, device=dev)
                torch.mm(a, w.t())
        torch.cuda.synchronize()

    # ---- 1b. W8A16 prefill 形状 M 扫描（WP-P1.3：M=709/1024） ----------------
    for m in (709, 1024):
        it = max(min(int(1200 / m) * int(S) or 2, 20), 2)
        t0 = time.monotonic_ns()
        with record_function(f"mb_w8a16_m{m}"):
            w8a16_call(m, it)
        win(f"mb_w8a16_m{m}", t0, time.monotonic_ns(),
            flops=flops_proj(m, it), bytes_=bytes_w8a16(m, it),
            windows=windows)

    # ---- 1. W8A16 decode GEMV（M=8，当前 GEMV_MAX_M 边界） -------------------
    it = max(int(30 * S), 3)
    t0 = time.monotonic_ns()
    with record_function("mb_w8a16_m8"):
        w8a16_call(8, it)
    win("mb_w8a16_m8", t0, time.monotonic_ns(), flops=flops_proj(8, it),
        bytes_=bytes_w8a16(8, it), windows=windows)

    # ---- 2. W8A16 GEMM tile（M=64，越过 GEMV_MAX_M 的臂） --------------------
    it = max(int(10 * S), 3)
    t0 = time.monotonic_ns()
    with record_function("mb_w8a16_m64"):
        w8a16_call(64, it)
    win("mb_w8a16_m64", t0, time.monotonic_ns(), flops=flops_proj(64, it),
        bytes_=bytes_w8a16(64, it), windows=windows)

    # ---- 3. bf16 GEMM M 扫描（prefill 主内核） -------------------------------
    for m in (8, 512, 2048, 8192):
        it = max(int(3000 / m) * int(S) or 2, 2)
        it = max(min(it, 60), 2)
        t0 = time.monotonic_ns()
        with record_function(f"mb_gemm_bf16_m{m}"):
            gemm_bf16_call(m, it)
        win(f"mb_gemm_bf16_m{m}", t0, time.monotonic_ns(),
            flops=flops_proj(m, it), windows=windows)

    # ---- 3b. fp16 GEMM M 扫描（WP-P1.3：R5 的 aten::mm 是 fp16） -------------
    for m in (512, 2048, 8192):
        it = max(int(3000 / m) * int(S) or 2, 2)
        it = max(min(it, 60), 2)
        t0 = time.monotonic_ns()
        with record_function(f"mb_gemm_fp16_m{m}"):
            gemm_fp16_call(m, it)
        win(f"mb_gemm_fp16_m{m}", t0, time.monotonic_ns(),
            flops=flops_proj(m, it), windows=windows)

    # ---- 4. GQA sdpa：decode（q=1, kv=4096） ---------------------------------
    it = max(int(50 * S), 3)
    q = torch.randn(1, HEADS, 1, HDIM, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, KV_HEADS, 4096, HDIM, device=dev, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    torch.cuda.synchronize()
    t0 = time.monotonic_ns()
    with record_function("mb_attn_decode_kv4096"):
        for _ in range(it):
            torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        torch.cuda.synchronize()
    fl = 2 * 2 * HEADS * 1 * HDIM * 4096 * it
    win("mb_attn_decode_kv4096", t0, time.monotonic_ns(), flops=fl, windows=windows)

    # ---- 5. GQA sdpa：prefill（q=kv=4096） -----------------------------------
    it = max(int(5 * S), 2)
    qp = torch.randn(1, HEADS, 4096, HDIM, device=dev, dtype=torch.bfloat16)
    kp = torch.randn(1, KV_HEADS, 4096, HDIM, device=dev, dtype=torch.bfloat16)
    vp = torch.randn_like(kp)
    torch.nn.functional.scaled_dot_product_attention(qp, kp, vp, enable_gqa=True)
    torch.cuda.synchronize()
    t0 = time.monotonic_ns()
    with record_function("mb_attn_prefill4096"):
        for _ in range(it):
            torch.nn.functional.scaled_dot_product_attention(qp, kp, vp, enable_gqa=True)
        torch.cuda.synchronize()
    fl = 2 * 2 * HEADS * 4096 * 4096 * HDIM * it
    win("mb_attn_prefill4096", t0, time.monotonic_ns(), flops=fl, windows=windows)

    # ---- 6. D2D 带宽基线（1 GiB 拷贝） ---------------------------------------
    it = max(int(20 * S), 3)
    big = torch.empty(512 * 2**20, dtype=torch.uint8, device=dev)  # 512MiB
    dst = torch.empty_like(big)
    big.copy_(dst)
    torch.cuda.synchronize()
    t0 = time.monotonic_ns()
    with record_function("mb_memcpy_d2d_1g"):
        for _ in range(it):
            dst.copy_(big)
        torch.cuda.synchronize()
    win("mb_memcpy_d2d_1g", t0, time.monotonic_ns(),
        bytes_=big.numel() * 2 * it, windows=windows)  # rd+wr

    Path(args.windows_out).write_text(json.dumps(windows, indent=1))
    print("MICROBENCH_WINDOWS", args.windows_out)
    for name, w in windows.items():
        dur_ms = (w["t1_mono_ns"] - w["t0_mono_ns"]) / 1e6
        eff = ""
        if w.get("flops"):
            eff = f"  {w['flops'] / (dur_ms / 1e3) / 1e12:.1f} TFLOP/s"
        elif w.get("bytes"):
            eff = f"  {w['bytes'] / (dur_ms / 1e3) / 2**30:.1f} GiB/s"
        print(f"  {name:24s} {dur_ms:8.1f}ms{eff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
