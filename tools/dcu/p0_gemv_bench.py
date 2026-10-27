#!/usr/bin/env python3
"""P0 GEMV acceptance bench (GEMV-OPT-20261025.md §3 P0 / §5).

Times the production fp8-emu block ops at the real decode/verify shapes and
checks the stream path against the tiled path (VLLM_DCU_FP8_GEMV_STREAM=0,
parsed once per process, so reference and test are separate runs):

  ref:   run every (route, shape, M) on the tiled path and save the outputs.
  bench: load the saved outputs, report max|delta|, then time the current
         dispatch (env selects stream variant / rows override).

GB/s counts the fp8 weight bytes (N*K), the same accounting as the
gemv_probe table in docs/perf/GEMV-OPT-20261025.md.

Usage (inside the DTK container, idle GPU):
  PYTHONPATH=/src HIP_VISIBLE_DEVICES=3 python3 tools/dcu/p0_gemv_bench.py \
      --mode ref  --out /tmp/p0ref
  PYTHONPATH=/src HIP_VISIBLE_DEVICES=3 python3 tools/dcu/p0_gemv_bench.py \
      --mode bench --out /tmp/p0ref [--rows 8] [--iters 50]
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
import zlib
from pathlib import Path

import torch
import vllm._rocm_C  # noqa: F401,E402  registers torch.ops._rocm_C.*

BLOCK_N = BLOCK_K = 128
FP8 = torch.float8_e4m3fn

# (name, K, N) -- Qwen3.8-27B TP=4 projections (kernel_microbench_p13.py).
# S33 WP1: + GDN in/out (5120-sized K, design D2 whole-A shape list).
SHAPES = [
    ("qkv", 5120, 2048),
    ("o", 5120, 1536),
    ("gateup", 5120, 8704),
    ("down", 17408, 1280),
    ("gdn_in", 5120, 2560),
    ("gdn_out", 5120, 1536),
]
MS = [1, 2, 3, 4, 8]


def _quant_group_w(x: torch.Tensor, block_n: int = 128, block_k: int = 128):
    """Weight quantization on the checkpoint's (block_n x block_k) grid.

    Returns (fp8 [N, K], w_s [N/block_n, K/block_k]).
    """
    n, k = x.shape
    g = x.reshape(n // block_n, block_n, k // block_k, block_k)
    s = g.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
    q = (g / s[:, None, :, None]).clamp(-448, 448).to(FP8)
    return q.reshape(n, k), s


def _quant_group_a(x: torch.Tensor, block_k: int = 128):
    """Per-token-group activation quantization: (fp8 [M, K], a_s [M, K/block_k])."""
    m, k = x.shape
    g = x.reshape(m, k // block_k, block_k)
    s = g.abs().amax(dim=2).clamp_min(1e-12) / 448.0
    q = (g / s[:, :, None]).clamp(-448, 448).to(FP8)
    return q.reshape(m, k), s


def build_case(k: int, n: int, m: int, route: str, device: str):
    # Deterministic across processes: str hashing is salted (PYTHONHASHSEED),
    # which once made the ref run and the bench run build different inputs.
    gen = torch.Generator(device="cpu").manual_seed(
        zlib.crc32(f"{k}:{n}:{m}:{route}".encode()))
    a_h = torch.randn(m, k, generator=gen, dtype=torch.float32) * 0.05
    w_h = torch.randn(n, k, generator=gen, dtype=torch.float32) * 0.02

    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_fp8 import (
        fp8_to_fp16_lut,
    )

    lut = fp8_to_fp16_lut(FP8).to(device)
    if route == "w8a16":
        a = a_h.to(torch.float16).to(device)
        wq, w_s = _quant_group_w(w_h)
        w_s = w_s.to(device)
        w = wq.to(device)
        a_s = None
    else:
        aq, a_s = _quant_group_a(a_h)
        a_s = a_s.to(device)
        a = aq.to(device)
        wq, w_s = _quant_group_w(w_h)
        w_s = w_s.to(device)
        w = wq.to(device)
    return a, w, a_s, w_s, lut


def run_op(route, a, w, a_s, w_s, lut, out_dtype=torch.float16):
    if route == "w8a16":
        return torch.ops._rocm_C.fp8_emu_gemm_block_w8a16(
            a, w, w_s, lut, None, out_dtype, 64, 0, BLOCK_N, BLOCK_K)
    return torch.ops._rocm_C.fp8_emu_gemv_block(
        a, w, a_s, w_s, lut, None, out_dtype, 64, BLOCK_N, BLOCK_K)


def bench_one(fn, iters: int) -> float:
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
    ap.add_argument("--mode", choices=["ref", "bench"], required=True)
    ap.add_argument("--out", default="/tmp/p0ref")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--routes", default="w8a16,w8a8")
    ap.add_argument("--ms", default=",".join(str(m) for m in MS))
    ap.add_argument("--shapes", default=",".join(s[0] for s in SHAPES))
    ap.add_argument("--rows", type=int, default=0,
                    help="VLLM_DCU_FP8_GEMV_STREAM_ROWS mirror (label only)")
    args = ap.parse_args()

    device = "cuda"
    torch.cuda.init()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    routes = args.routes.split(",")
    ms = [int(x) for x in args.ms.split(",")]
    shapes = [s for s in SHAPES if s[0] in set(args.shapes.split(","))]

    env = {
        "GEMV_STREAM": os.environ.get("VLLM_DCU_FP8_GEMV_STREAM", "unset"),
        "STREAM_ROWS": os.environ.get("VLLM_DCU_FP8_GEMV_STREAM_ROWS", "unset"),
        "rows_label": args.rows,
    }
    results = []
    for name, k, n in shapes:
        for route in routes:
            for m in ms:
                a, w, a_s, w_s, lut = build_case(k, n, m, route, device)
                key = f"{route}_{name}_{k}x{n}_M{m}"
                if args.mode == "ref":
                    out = run_op(route, a, w, a_s, w_s, lut)
                    torch.save(out.cpu(), out_dir / f"{key}.pt")
                    continue
                ref = torch.load(out_dir / f"{key}.pt", weights_only=True).to(device)
                op = functools.partial(run_op, route, a, w, a_s, w_s, lut)
                t_us = bench_one(op, args.iters)
                out = op()
                dmax = (out.float() - ref.float()).abs().max().item()
                gbps = n * k / (t_us * 1e-6) / 1e9
                row = {
                    "case": key, "route": route, "K": k, "N": n, "M": m,
                    "us": round(t_us, 1), "GBps": round(gbps, 1),
                    "max_abs_delta": dmax,
                }
                results.append(row)
                print(json.dumps(row), flush=True)
                del a, w, a_s, w_s, out, ref
                torch.cuda.empty_cache()
    if args.mode == "bench":
        meta = {"env": env, "torch": torch.__version__,
                "results": results}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tag = f"rows{args.rows}" if args.rows else "auto"
        with open(out_dir / f"bench_{tag}_{stamp}.json", "w") as f:
            json.dump(meta, f, indent=1)
        print(f"saved {out_dir}/bench_{tag}_{stamp}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
