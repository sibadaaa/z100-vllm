#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# S18: microbenchmark the gfx906 block-scaled FP8 emulated GEMM/GEMV across M=1..8
# (the MTP verify shapes). Also sweeps the two levers that are cheap to test
# before writing any new kernel:
#   * force the tiled GEMM path (the production dispatch uses GEMV for M<=8)
#   * sweep the GEMV thread count (production default 64)
#
#   HIP_VISIBLE_DEVICES=0 python /src/tools/dcu/s18_fp8_block_mprobe.py \
#       [--shapes N:K,...] [--ms 1,2,3,4,6,8] [--modes gemv,gemm] \
#       [--threads-list 64,128,256] [--gemm-cfg 0] [--iters 30] [--check]
#
# Reports ms, effective weight bandwidth (GB/s), ratio vs the M=1 GEMV baseline,
# a row-consistency check (same row 0 across M) and rel_err vs an fp32 reference.
try:  # hard constraint 8: import torch before anything vllm-adjacent
    import torch
except Exception:
    raise SystemExit("this probe needs torch")

import argparse
import json
import os

BLOCK_N = 128
BLOCK_K = 128


def lut_for(fp8_dtype=torch.float8_e4m3fn):
    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_fp8 import _device_lut

    return _device_lut(fp8_dtype, torch.device("cuda"))


def make_case(n: int, k: int, m: int, dtype=torch.float16):
    """Block-scaled fp8 weights in checkpoint layout [N, K] + inputs."""
    fp8 = torch.float8_e4m3fn
    w = (torch.randn(n, k, device="cuda") * 0.02).to(fp8)
    w_f = w.to(torch.float32)
    nb_n, nb_k = n // BLOCK_N, k // BLOCK_K
    w_s = (
        w_f.view(nb_n, BLOCK_N, nb_k, BLOCK_K).abs().amax(dim=(1, 3)) / 448.0
    ).clamp_min(1e-8)
    a = (torch.randn(m, k, device="cuda", dtype=dtype) * 0.5).to(torch.float32)
    a_q = a.to(fp8)
    a_s = (
        a_q.to(torch.float32).view(m, nb_k, BLOCK_K).abs().amax(dim=2) / 448.0
    ).clamp_min(1e-8)
    return a_q, w, a_s, w_s


def ref_out(a_q, a_s, w, w_s, out_dtype):
    k = a_q.shape[1]
    nb_k = k // BLOCK_K
    af = a_q.to(torch.float32).view(a_q.shape[0], nb_k, BLOCK_K) * a_s[:, :, None]
    wf = w.to(torch.float32).view(w.shape[0] // BLOCK_N, BLOCK_N, nb_k, BLOCK_K)
    wf = wf * w_s[:, None, :, None]
    wf = wf.reshape(w.shape[0], k)
    return (af.reshape(a_q.shape[0], k) @ wf.t()).to(out_dtype)


def make_call(mode, a_q, w, a_s, w_s, lut, threads, gemm_cfg):
    if mode == "gemv":
        return lambda: torch.ops._rocm_C.fp8_emu_gemv_block(
            a_q, w, a_s, w_s, lut, None, torch.float16, threads, BLOCK_N, BLOCK_K
        )
    if mode == "gemm":
        return lambda: torch.ops._rocm_C.fp8_emu_gemm_block(
            a_q, w, a_s, w_s, lut, None, torch.float16, gemm_cfg, BLOCK_N, BLOCK_K
        )
    raise ValueError(mode)


def timeit(fn, iters: int, rounds: int = 3) -> float:
    """Best-of-`rounds` over `iters` launches: GPU0 is shared with the live
    service and other probes, so the minimum is the only contention-robust
    statistic (the mean of a busy round is not a kernel property)."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    best = float("inf")
    for _ in range(rounds):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters)
    return best


def run(args) -> int:
    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_fp8 import (
        ops_available_block,
    )

    if not ops_available_block():
        print("fp8_emu_*_block ops are NOT registered -> cannot probe")
        return 2

    lut = lut_for()
    variants = []
    for mode in args.modes:
        if mode == "gemv":
            for nt in args.threads_list:
                variants.append((mode, nt))
        else:
            variants.append((mode, 0))

    results = []
    print(f"{'shape':>14} {'mode':>5} {'nt':>4} {'M':>2} {'ms':>8} {'GB/s':>6} "
          f"{'x(base)':>8}  note")
    for shape in args.shapes:
        n, k = (int(x) for x in shape.split(":"))
        w_bytes = n * k
        a_q_full, w, a_s_full, w_s = make_case(n, k, max(args.ms))
        base_ms = None
        first_row = {}
        for mode, nt in variants:
            for m in args.ms:
                a_q = a_q_full[:m].contiguous()
                a_s = a_s_full[:m].contiguous()
                call = make_call(mode, a_q, w, a_s, w_s, lut, nt, args.gemm_cfg)
                ms = timeit(call, args.iters)
                gbs = w_bytes / (ms * 1e-3) / 1e9
                if base_ms is None:
                    base_ms = ms
                note = ""
                if args.check:
                    out = call()
                    ref = ref_out(a_q, a_s, w, w_s, torch.float16)
                    rel = (
                        (out.float() - ref) / ref.abs().clamp_min(1e-3)
                    ).abs().max().item()
                    note = f"rel_err={rel:.2e}"
                    if mode == "gemv" and nt == args.threads_list[0]:
                        first_row[m] = out[0].clone()
                print(f"{n:>7}x{k:<6} {mode:>5} {nt:>4} {m:>2} {ms:>8.4f} "
                      f"{gbs:>6.0f} {ms / base_ms:>8.2f}  {note}")
                results.append(
                    {"n": n, "k": k, "mode": mode, "threads": nt, "m": m, "ms": ms,
                     "weight_gbs": gbs, "ratio_vs_base": ms / base_ms, "note": note}
                )
        if args.check and 1 in first_row and len(first_row) > 1:
            for m, row in first_row.items():
                same = bool(torch.equal(row, first_row[1]))
                print(f"  row-consistency M={m} vs M=1: "
                      f"{'OK' if same else 'DIFFERS'}")
                results.append({"n": n, "k": k, "row_consistency_m": m,
                                "same": same})

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "s18_fp8_block_mprobe.json")
    with open(path, "w") as f:
        json.dump({"iters": args.iters, "block": [BLOCK_N, BLOCK_K],
                   "results": results}, f, indent=1)
    print(f"[probe] wrote {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shapes",
        default="8704:5120,5120:1536",
        help="comma-separated N:K (default: 27B gate_up shard + o_proj shard)",
    )
    ap.add_argument("--ms", default="1,2,3,4")
    ap.add_argument("--modes", default="gemv,gemm")
    ap.add_argument("--threads-list", default="64,128,256")
    ap.add_argument("--gemm-cfg", type=int, default=0)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--outdir", default="/mnt/game/s18")
    args = ap.parse_args()
    args.ms = [int(x) for x in args.ms.split(",")]
    args.shapes = args.shapes.split(",")
    args.modes = args.modes.split(",")
    args.threads_list = [int(x) for x in args.threads_list.split(",")]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
