#!/usr/bin/env python3
"""FP8 W8A16 GEMV bandwidth microbench — cold (L2-defeated) vs hot.

Methodology (operator framework):
  * M=1 GEMV has AI ~= 2 FLOP/B, far below the ridge point -> judge by
    achieved BANDWIDTH, not latency or TFLOP/s.
  * Hot-loop microbenches re-read the same [N,K] fp8 weight every iteration;
    o/qkv shapes (8-18 MB) fit in L2, so hot numbers measure L2 bandwidth,
    not HBM streaming.  COLD rotates through enough weight copies that every
    iteration's working set exceeds L2 -> per-iteration HBM-cold reads, the
    production pattern (each weight is read exactly once per decode step).
  * Ceilings measured on the same device in the same process:
      - D2D copy (read+write) anchor
      - read-only reduce sum (pure read; the honest GEMV target)

Env: VLLM_DCU_FP8_GEMV_M1=1, VLLM_DCU_FP8_GEMV_STREAM_ROWS={4,8} (read once
per process -> one process per rows value).  Run on an idle device; keep
allocations < ~2.5 GB (service workers hold 85% of VRAM on all 4 cards).
"""
import json
import os
import statistics
import sys
import time

import torch

import vllm._rocm_C  # noqa: F401  -- loads the .so, registers torch.ops._rocm_C

SHAPES = [
    # (name, N, K) per-rank (TP4) fp8 GEMV shapes, Qwen3.8-27B-FP8
    ("attn_qkv", 3584, 5120),
    ("gdn_in_qkv", 2560, 5120),
    ("gate_up", 8704, 5120),
    ("gdn_in_z", 1536, 5120),
    ("attn_o", 5120, 1536),
    ("gdn_out", 5120, 1536),
    ("down", 5120, 4352),
]
BLOCK_N = BLOCK_K = 128
COLD_SET_BYTES = 768 * 1024 * 1024  # working-set floor for the cold rotation


def ceil_div(a, b):
    return (a + b - 1) // b


def make_set(N, K, copies):
    ws = [torch.randint(0, 256, (N, K), dtype=torch.uint8, device="cuda")
          for _ in range(copies)]
    w_s = torch.rand(ceil_div(N, BLOCK_N), ceil_div(K, BLOCK_K),
                     dtype=torch.float32, device="cuda") * 1e-3 + 1e-4
    a = torch.randn(1, K, dtype=torch.float16, device="cuda")
    lut = torch.randn(256, dtype=torch.float16, device="cuda")
    out = torch.empty(1, N, dtype=torch.float16, device="cuda")
    return ws, w_s, a, lut, out


def bench_call(fn, iters):
    torch.cuda.synchronize()
    times = []
    for i in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(i)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)  # us
    return times


def gemv_bytes(N, K):
    return N * K + K * 2 + N * 2 + ceil_div(N, BLOCK_N) * ceil_div(K, BLOCK_K) * 4


def main():
    rows = os.environ.get("VLLM_DCU_FP8_GEMV_STREAM_ROWS", "4")
    dev = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print(f"device={dev} l2={getattr(props, 'L2_cache_size', '?')} "
          f"multiprocessors={props.multi_processor_count} rows={rows}")
    op = torch.ops._rocm_C.fp8_emu_gemm_block_w8a16

    res = {"rows": rows, "device": dev, "shapes": {}}

    # ---- ceilings -------------------------------------------------------
    x = torch.empty(512 * 1024 * 1024, dtype=torch.float16, device="cuda")  # 1 GiB
    x.normal_()
    t = bench_call(lambda i: torch.sum(x), 20)
    read_bw = x.numel() * 2 / (statistics.median(t) / 1e6) / 1e9
    print(f"ceiling read-only sum(1GiB fp16): {statistics.median(t):.0f}us "
          f"-> {read_bw:.1f} GB/s")
    res["read_only_GBps"] = round(read_bw, 1)

    y = torch.empty(256 * 1024 * 1024, dtype=torch.float16, device="cuda")  # 512 MiB
    y.normal_()
    t = bench_call(lambda i: y.copy_(x[: y.numel()]), 20)
    d2d = y.numel() * 2 * 2 / (statistics.median(t) / 1e6) / 1e9
    print(f"ceiling d2d copy(512MiB):         {statistics.median(t):.0f}us "
          f"-> {d2d:.1f} GB/s (read+write)")
    res["d2d_GBps"] = round(d2d, 1)
    del x, y
    torch.cuda.empty_cache()

    # ---- GEMV shapes ----------------------------------------------------
    for name, N, K in SHAPES:
        entry = {}
        for mode in ("cold", "hot"):
            nbytes = N * K
            copies = max(2, int(COLD_SET_BYTES // nbytes)) if mode == "cold" else 1
            ws, w_s, a, lut, out = make_set(N, K, copies)
            iters = 40 * copies if mode == "cold" else 200

            def call(i, _ws=ws):
                return op(a, _ws[i % len(_ws)], w_s, lut, None, torch.float16,
                          64, 0, BLOCK_N, BLOCK_K)

            t = bench_call(call, iters)
            med = statistics.median(t)
            bw = gemv_bytes(N, K) / (med / 1e6) / 1e9
            entry[mode] = {"us": round(med, 2), "GBps": round(bw, 1),
                           "copies": copies, "n": len(t)}
            print(f"{name:<12} [{mode:<4}] copies={copies:<3d} {med:8.2f}us "
                  f"-> {bw:7.1f} GB/s  (p10 {sorted(t)[len(t)//10]:.1f}us)")
            del ws
            torch.cuda.empty_cache()
        entry["cold_hot_ratio"] = round(entry["hot"]["us"] / entry["cold"]["us"], 2)
        res["shapes"][name] = entry

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"gemv_bw_rows{rows}.json")
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
