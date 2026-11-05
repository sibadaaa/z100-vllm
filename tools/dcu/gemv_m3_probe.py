#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""WP-A2 probe: fp8_emu W8A16 GEMV at the real single-stream decode shapes.

Construction doc §4 (z100-single-stream-decode): the decode step runs 255
fp8_emu_gemv_stream_block_kernel<2,4,1> launches/step at M=3 (MTP k=2 verify)
and the design's L2 lever claims 286 GB/s effective weight bandwidth vs a
same-stack GEMV peak of 590-625 GB/s.  This probe times the exact production
entry point (``torch.ops._rocm_C.fp8_emu_gemm_block_w8a16`` with the production
env: VLLM_DCU_FP8_GEMV_STREAM_ROWS=4, W8A16 route) on the real (K, N) set
extracted from the Qwen3.8-27B-FP8 checkpoint, at M in {1,2,3,4}, and checks
numerics against an fp64 reference.

Shapes (weight [N, K], from the safetensors headers):
  GDN layer x48 : in_proj_qkv 10240x5120, in_proj_z 6144x5120,
                  in_proj_ba 96x5120 (or b/a 48x5120 each), out 5120x6144,
                  gate_up 34816x5120 (merged at load), down 5120x17408
  FA  layer x16 : q 12288x5120, k/v 1024x5120 (or fused qkv 14336x5120),
                  o 5120x6144, gate_up 34816x5120, down 5120x17408

Weighted bandwidth is reported under both projection compositions
(split / fused) because the graph-captured trace cannot attribute shapes.

Self-contained (no vllm import; kernels loaded straight from _rocm_C.abi3.so),
same pattern as tools/dcu/a1_ba_gemv_probe.py.  Timing: cuda events over
iters=50 after warmup=10 (the a1_ba bench()); kernels are ~50-500 us.

Usage (idle GPU, DTK env active):
    VLLM_DCU_FP8_GEMV_STREAM_ROWS=4 python tools/dcu/gemv_m3_probe.py \
        [--so PATH] [--out JSON] [--ms 1,2,3,4] [--iters 50]
"""

import argparse
import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F

# --- the real shapes: (tag, N, K, calls_per_step_split, calls_per_step_fused)
# calls_per_step_* : how many launches of this shape one decode step issues
# across all 64 layers under that composition (lm_head excluded: fp16 kernel).
SHAPES = [
    ("gdn.in_qkv",   10240,  5120, 48,  0),
    ("gdn.in_z",      6144,  5120, 48,  0),
    ("gdn.in_ba",       96,  5120,  0, 48),
    ("gdn.in_b",        48,  5120, 48,  0),
    ("gdn.in_a",        48,  5120, 48,  0),
    ("gdn.out",       5120,  6144, 48, 48),
    ("mlp.gate_up",  34816,  5120, 64, 64),
    ("mlp.down",      5120, 17408, 64, 64),
    ("fa.q",         12288,  5120, 16,  0),
    ("fa.kv",         1024,  5120, 32,  0),
    ("fa.qkv",       14336,  5120,  0, 16),
]

BLOCK_N = BLOCK_K = 128  # checkpoint weight_scale_inv blocks


def fp8_to_fp16_lut() -> torch.Tensor:
    """The exact production table (gfx906_fp8.fp8_to_fp16_lut, e4m3fn)."""
    bits = torch.arange(256, dtype=torch.int32)
    sign = torch.where((bits >> 7) & 1 == 1, -1.0, 1.0)
    exp = (bits >> 3) & 0xF
    man = bits & 0x7
    nan_codes = (exp == 0xF) & (man == 0x7)
    val = torch.where(
        exp == 0,
        man.float() * (1.0 / 8.0) * 2.0 ** (1 - 7),
        (1.0 + man.float() * (1.0 / 8.0)) * torch.pow(2.0, exp.float() - 7),
    )
    out = sign * val
    out = torch.where(nan_codes, torch.tensor(float("nan")), out)
    return out.to(torch.float16).contiguous()


def bench(fn, warmup=10, iters=50) -> float:
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
    ap.add_argument("--so",
                    default="/home/lyh/z100-vllm/vllm/_rocm_C.abi3.so")
    ap.add_argument("--out",
                    default="/home/lyh/z100-vllm/runs/gemv_m3_probe.json")
    ap.add_argument("--ms", default="1,2,3,4")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    # Production geometry knobs (read once per process inside the .so).
    os.environ.setdefault("VLLM_DCU_FP8_GEMV_STREAM_ROWS", "4")
    os.environ.setdefault("VLLM_DCU_FP8_GEMV_STREAM", "1")

    if not torch.cuda.is_available():
        print("no GPU visible", file=sys.stderr)
        return 2
    torch.ops.load_library(args.so)
    assert hasattr(torch.ops._rocm_C, "fp8_emu_gemm_block_w8a16"), \
        "fp8_emu_gemm_block_w8a16 missing"

    torch.manual_seed(0)
    dev = torch.cuda.get_device_name(0)
    ms = [int(x) for x in args.ms.split(",")]
    lut = fp8_to_fp16_lut().cuda()
    print(f"# device={dev} so={args.so} rows_env="
          f"{os.environ['VLLM_DCU_FP8_GEMV_STREAM_ROWS']} ms={ms}", flush=True)

    rows = []
    for tag, n, k, w_split, w_fused in SHAPES:
        w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.float8_e4m3fn)
        assert w.stride(1) == 1 and (w.stride(0) % 32) == 0
        nb, nk = (n + BLOCK_N - 1) // BLOCK_N, k // BLOCK_K
        w_s = torch.rand(nb, nk, device="cuda",
                         dtype=torch.float32) * 0.5 + 0.75
        w64 = lut[w.view(torch.uint8).long()].double()  # [N, K]
        row_scale = w_s.repeat_interleave(BLOCK_N, dim=0)[:n].double()
        for kb in range(nk):  # fold the per-block weight scales in fp64
            w64[:, kb * BLOCK_K:(kb + 1) * BLOCK_K] *= \
                row_scale[:, kb].unsqueeze(1)
        for m in ms:
            a = (torch.randn(m, k, device="cuda") * 0.5).to(torch.float16)
            ref = F.linear(a.double(), w64)  # [m, n] fp64

            def call():
                return torch.ops._rocm_C.fp8_emu_gemm_block_w8a16(
                    a, w, w_s, lut, None, torch.float16, 64, 0,
                    BLOCK_N, BLOCK_K)

            out = call()
            rel = ((out.double() - ref).abs().max()
                   / ref.abs().max().clamp_min(1e-30)).item()
            digest = hashlib.md5(
                out.detach().cpu().contiguous().view(torch.uint8).numpy()
                .tobytes()).hexdigest()
            us = bench(call, warmup=args.warmup, iters=args.iters) * 1e3
            gbs = n * k / us / 1e3  # weight bytes read once per launch
            rows.append({
                "tag": tag, "N": n, "K": k, "M": m, "us": round(us, 2),
                "gbs": round(gbs, 1), "rel_err": float(f"{rel:.3e}"),
                "out_md5": digest,
                "calls_split": w_split, "calls_fused": w_fused,
            })
            print(f"RESULT {tag:14s} N={n:6d} K={k:6d} M={m} "
                  f"t={us:8.2f}us  {gbs:7.1f} GB/s  rel={rel:.2e}",
                  flush=True)
        del w, w_s, w64
        torch.cuda.empty_cache()

    # Weighted effective bandwidth per M under both compositions.
    summary = {}
    for m in ms:
        sel = [r for r in rows if r["M"] == m]
        tot_bytes = sum(r["N"] * r["K"] for r in sel)
        tot_time = sum(r["us"] for r in sel)
        avg = tot_bytes / tot_time / 1e3
        entry = {"mean_gbs_all_shapes": round(avg, 1)}
        for comp in ("split", "fused"):
            calls = sum(r[f"calls_{comp}"] for r in sel)
            if calls == 0:
                continue
            b = sum(r["N"] * r["K"] * r[f"calls_{comp}"] for r in sel)
            t = sum(r["us"] * r[f"calls_{comp}"] for r in sel)
            entry[f"weighted_gbs_{comp}"] = round(b / t / 1e3, 1)
            entry[f"ms_per_step_{comp}"] = round(t / 1e3, 2)
            entry[f"calls_per_step_{comp}"] = calls
        summary[m] = entry
        print(f"SUMMARY M={m} {json.dumps(entry)}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"device": dev, "so": args.so, "rows": rows,
                   "summary": summary}, f, indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
