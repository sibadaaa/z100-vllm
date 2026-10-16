#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# S18: the INT8 counterpart of s18_fp8_block_mprobe.py — how does the gfx906
# W8A8 sdot4 GEMV scale with M (= 1+k, the MTP verify width)?
#
#   HIP_VISIBLE_DEVICES=0 python /src/tools/dcu/s18_int8_mprobe.py \
#       [--shapes N:K,...] [--ms 1,2,3,4,6,8] [--threads 64] [--iters 30] [--check]
#
# Contract (matches csrc/rocm/w8a8_sdot4.cu:386 w8a8_sdot4_gemv):
#   xq int8 [M,K] with per-token scale x_s [M]; w int8 [N,K] with per-channel
#   scale w_s [N]; out[m,n] = (sum_k xq[m,k]*w[n,k]) * x_s[m] * w_s[n].
try:  # hard constraint 8
    import torch
except Exception:
    raise SystemExit("this probe needs torch")

import argparse
import json
import os

_GEMV_THREADS = 64  # vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py:79


def make_case(n: int, k: int, m: int, dtype=torch.float16):
    xf = torch.randn(m, k, device="cuda") * 0.5
    x_scale = (xf.abs().amax(dim=1) / 127.0).clamp_min(1e-8)
    xq = (xf / x_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    wf = torch.randn(n, k, device="cuda") * 0.02
    w_scale = (wf.abs().amax(dim=1) / 127.0).clamp_min(1e-8)
    wq = (wf / w_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return xq, wq, x_scale, w_scale


def ref_out(xq, x_scale, wq, w_scale, out_dtype):
    acc = xq.to(torch.float32) @ wq.to(torch.float32).t()
    return (acc * x_scale[:, None] * w_scale[None, :]).to(out_dtype)


def run(args) -> int:
    if not hasattr(torch.ops._rocm_C, "w8a8_sdot4_gemv"):
        print("torch.ops._rocm_C.w8a8_sdot4_gemv not registered -> cannot probe")
        return 2

    results = []
    print(f"{'shape':>16} {'M':>2} {'ms':>8} {'GB/s':>7} {'x(M=1)':>7}  note")
    for shape in args.shapes:
        n, k = (int(x) for x in shape.split(":"))
        w_bytes = n * k
        xq_full, w, xs_full, ws = make_case(n, k, max(args.ms))
        base_ms = None
        first_row = {}
        for m in args.ms:
            xq, xs = xq_full[:m].contiguous(), xs_full[:m].contiguous()
            torch.cuda.synchronize()

            def call():
                return torch.ops._rocm_C.w8a8_sdot4_gemv(
                    xq, w, xs, ws, None, torch.float16, args.threads
                )

            for _ in range(5):
                call()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(args.iters):
                call()
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / args.iters
            gbs = w_bytes / (ms * 1e-3) / 1e9
            if base_ms is None:
                base_ms = ms
            note = ""
            if args.check:
                out = call()
                ref = ref_out(xq, xs, w, ws, torch.float16)
                rel = (
                    (out.float() - ref) / ref.abs().clamp_min(1e-3)
                ).abs().max().item()
                first_row[m] = out[0].clone()
                note = f"rel_err={rel:.2e}"
            print(
                f"{n:>8}x{k:<7} {m:>2} {ms:>8.4f} {gbs:>7.0f} "
                f"{ms / base_ms:>7.2f}  {note}"
            )
            results.append(
                {"n": n, "k": k, "m": m, "ms": ms, "weight_gbs": gbs,
                 "ratio_vs_m1": ms / base_ms, "threads": args.threads, "note": note}
            )
        if args.check and 1 in first_row and len(first_row) > 1:
            for m, row in first_row.items():
                same = bool(torch.equal(row, first_row[1]))
                print(f"  row-consistency M={m} vs M=1: {'OK' if same else 'DIFFERS'}")
                results.append({"n": n, "k": k, "row_consistency_m": m, "same": same})

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "s18_int8_mprobe.json")
    with open(path, "w") as f:
        json.dump({"iters": args.iters, "threads": args.threads,
                   "results": results}, f, indent=1)
    print(f"[probe] wrote {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="8704:5120,2048:5120,5120:4352,5120:1536")
    ap.add_argument("--ms", default="1,2,3,4,6,8")
    ap.add_argument("--threads", type=int, default=_GEMV_THREADS)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--outdir", default="/mnt/game/s18")
    args = ap.parse_args()
    args.ms = [int(x) for x in args.ms.split(",")]
    args.shapes = args.shapes.split(",")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
