#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# S18 follow-up probe (perf1 / L3 kernel work): the decode-shaped weight sweep
# plus the tiled-GEMM cfg sweep that tools/dcu/s18_fp8_block_mprobe.py cannot
# express (it fixes --gemm-cfg for the whole run).
#
#   HIP_VISIBLE_DEVICES=0 python tools/dcu/perf1_kernel_mprobe.py \
#       [--shapes 1280:5120,3584:5120,...] [--ms 1,2,3,4] \
#       [--modes gemv,gemm] [--gemm-cfgs 0,1,2,3,5,6,7,8,9,10] \
#       [--iters 20] [--check] [--outdir /mnt/game/perf1]
#
# Mode "gemv" is the production decode entry point (`fp8_emu_gemv_block`), which
# routes to the streaming kernel when its geometry fits (both the stream kernel
# and the older GEMV are reachable through it; VLLM_DCU_FP8_GEMV_STREAM=0 in
# the environment selects the old one).
#
# Reports ms, effective weight bandwidth (GB/s) and, with --check, the max
# relative error against an fp32 reference.

try:  # import torch before anything vllm-adjacent
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


def make_case(n, k, m, dtype=torch.float16):
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


def timeit(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


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
            variants.append((mode, args.threads))
        else:
            for cfg in args.gemm_cfgs:
                variants.append((mode, cfg))

    results = []
    print(
        f"{'shape':>14} {'mode':>5} {'var':>4} {'M':>2} {'ms':>8} {'GB/s':>6}"
        f"  note"
    )
    for shape in args.shapes:
        n, k = (int(x) for x in shape.split(":"))
        w_bytes = n * k
        a_q_full, w, a_s_full, w_s = make_case(n, k, max(args.ms))
        for mode, var in variants:
            for m in args.ms:
                a_q = a_q_full[:m].contiguous()
                a_s = a_s_full[:m].contiguous()
                call = make_call(mode, a_q, w, a_s, w_s, lut, 64, var)
                ms = timeit(call, args.iters)
                gbs = w_bytes / (ms * 1e-3) / 1e9
                note = ""
                if args.check:
                    out = call()
                    ref = ref_out(a_q, a_s, w, w_s, torch.float16)
                    rel = (
                        (out.float() - ref) / ref.abs().clamp_min(1e-3)
                    ).abs().max().item()
                    note = f"rel={rel:.2e}"
                print(
                    f"{n:>7}x{k:<6} {mode:>5} {var:>4} {m:>2} {ms:>8.4f} "
                    f"{gbs:>6.0f}  {note}"
                )
                results.append(
                    {
                        "n": n,
                        "k": k,
                        "mode": mode,
                        "var": var,
                        "m": m,
                        "ms": ms,
                        "weight_gbs": gbs,
                        "note": note,
                    }
                )
        del w, a_q_full, a_s_full, w_s
        torch.cuda.empty_cache()

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, args.json)
    if args.json == "perf1_kernel_mprobe.json":
        tag = os.environ.get("VLLM_DCU_FP8_GEMV_STREAM", "unset")
        pref = os.environ.get("VLLM_DCU_FP8_GEMV_PREF", "unset")
        path = os.path.join(
            args.outdir, f"perf1_kernel_mprobe_stream{tag}_pref{pref}.json"
        )
    with open(path, "w") as f:
        json.dump(
            {
                "iters": args.iters,
                "block": [BLOCK_N, BLOCK_K],
                "gemv_threads": args.threads,
                "stream_env": os.environ.get("VLLM_DCU_FP8_GEMV_STREAM", "<unset>"),
                "results": results,
            },
            f,
            indent=1,
        )
    print(f"[probe] wrote {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="1280:5120,3584:5120,4352:5120,5120:5120,8704:5120")
    ap.add_argument("--ms", default="1,2,3,4")
    ap.add_argument("--modes", default="gemv,gemm")
    ap.add_argument("--gemm-cfgs", default="0,1,2,3,5,6,7,8,9,10")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--outdir", default="/mnt/game/perf1")
    ap.add_argument("--json", default="perf1_kernel_mprobe.json")
    args = ap.parse_args()
    args.ms = [int(x) for x in args.ms.split(",")]
    args.shapes = args.shapes.split(",")
    args.modes = args.modes.split(",")
    args.gemm_cfgs = [int(x) for x in args.gemm_cfgs.split(",")]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
