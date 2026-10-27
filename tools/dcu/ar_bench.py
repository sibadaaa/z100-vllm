#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""4-rank RCCL all_reduce microbenchmark with rank-0 JSON output and a sum check.

Message sizes are given as bf16 ``[1, n]`` payloads (``n = bytes / 2``).  Every
size gets >= 50 timed samples and the report carries mean / p50 / p99.
Correctness: every rank contributes a *different* input and the all_reduce
result is compared element-wise with the reference sum computed on the
gathered inputs:
  * constant check: rank r sends ``r+1`` -> expected ``ws*(ws+1)/2`` exactly
    (integers <= 256 are exact in bf16; for larger world sizes the check is
    reported but cannot be exact);
  * tolerance check: rank r sends ``base*(r+1)`` with a common bf16 ``base``;
    the reference is the fp64 sum of the gathered inputs.  bf16 has an 8-bit
    mantissa (2^-8 = 0.39% per rounding), a sum over ws ranks has at most ws-1
    roundings, so the default tolerance is 2% of max|ref| (``--tol-bf16``).

Run through the venv torchrun (4 ranks, requires production stopped):
  /home/lyh/z100-vllm/.venv/bin/torchrun --nproc_per_node=4 \
      --master_port=29610 tools/dcu/ar_bench.py \
      --sizes 10KB,40KB,160KB,1MB,4MB --iters 20 --warmup 10 --samples 50 \
      --out .r3/WP4/data/baseline_rep1.json
Rank 0 prints the table and writes JSON; other ranks print nothing.
"""

import argparse
import json
import math
import os
import socket
import statistics
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

DEFAULT_SIZES = "10KB,40KB,160KB,1MB,4MB"
MIN_SAMPLES = 50
UNITS = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}
ELEM_BYTES = 2  # bf16


def parse_size(token: str) -> int:
    text = token.strip().replace("_", "")
    upper = text.upper()
    for unit in ("GB", "MB", "KB", "B"):
        if upper.endswith(unit):
            return int(round(float(text[: -len(unit)]) * UNITS[unit]))
    return int(text)


def percentile(values, q):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default=DEFAULT_SIZES,
                        help="comma list of bf16 [1,n] payload sizes "
                             f"(default {DEFAULT_SIZES})")
    parser.add_argument("--iters", type=int, default=20,
                        help="all_reduce calls per timed sample")
    parser.add_argument("--warmup", type=int, default=10, help="untimed calls per size")
    parser.add_argument("--samples", type=int,
                        default=MIN_SAMPLES,
                        help=f"timed samples per size (>= {MIN_SAMPLES})")
    parser.add_argument("--out", default="", help="JSON output path (rank 0 only)")
    parser.add_argument("--tag", default="", help="config label (e.g. NCCL_ALGO=Tree)")
    parser.add_argument("--tol-bf16", type=float, default=0.02,
                        help="relative tolerance for the sum check "
                             "(default 0.02 = 2%%)")
    parser.add_argument("--timeout-s", type=float, default=120.0,
                        help="collective timeout; a hang aborts after this "
                             "(default 120 s)")
    args = parser.parse_args(argv)
    if args.samples < MIN_SAMPLES:
        print(f"[ar_bench] samples raised to {MIN_SAMPLES} "
              f"(was {args.samples})", file=sys.stderr)
        args.samples = MIN_SAMPLES
    if args.iters < 1:
        print("[ar_bench] iters must be >= 1", file=sys.stderr)
        raise SystemExit(2)
    return args


def env_snapshot():
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(("NCCL_", "RCCL_", "HIP_VISIBLE_DEVICES", "MASTER_PORT"))
    }


def check(x_src, rank, world_size, tol_rel):
    """Return (const_result, tolerance_result) for one size."""
    x = x_src.clone()
    dist.all_reduce(x)
    torch.cuda.synchronize()

    constant = float(rank + 1)
    expected_const = world_size * (world_size + 1) / 2
    const_max_abs = (x.float() - expected_const).abs().max().item()
    exact_possible = expected_const <= 256.0 and expected_const == int(expected_const)
    const_result = {
        "input_per_rank": constant,
        "expected_sum": expected_const,
        "max_abs_err": const_max_abs,
        "bf16_exact_possible": exact_possible,
        "pass": (bool(exact_possible and const_max_abs == 0.0)
                 if exact_possible else None),
    }

    torch.manual_seed(20261020)
    base = torch.rand(x_src.shape, device=x_src.device,
                      dtype=torch.float32).to(torch.bfloat16)
    contrib = (base * (rank + 1)).to(torch.bfloat16)
    x = contrib.clone()
    dist.all_reduce(x)
    torch.cuda.synchronize()
    gathered = [torch.empty_like(contrib) for _ in range(world_size)]
    dist.all_gather(gathered, contrib)
    ref = torch.stack([t.double() for t in gathered]).sum(dim=0).to(torch.bfloat16)
    diff = (x.float() - ref.float()).abs()
    max_abs = diff.max().item()
    max_scale = ref.float().abs().max().item()
    max_rel = max_abs / max_scale if max_scale else max_abs
    tol_result = {
        "input": "base*(rank+1), common bf16 base",
        "tol_rel": tol_rel,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "pass": bool(max_rel <= tol_rel),
    }
    return const_result, tol_result


def main(argv=None):
    args = parse_args(argv)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout_s))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    sizes = [parse_size(token) for token in args.sizes.split(",") if token.strip()]

    rows = []
    all_pass = True
    try:
        for size_bytes in sizes:
            n = size_bytes // ELEM_BYTES
            label = f"{size_bytes}B"
            x_src = torch.full((1, n), float(rank + 1),
                               dtype=torch.bfloat16, device=device)
            x = x_src.clone()
            for _ in range(max(0, args.warmup)):
                dist.all_reduce(x)
            torch.cuda.synchronize()
            dist.barrier()

            samples_us = []
            for _ in range(args.samples):
                x.copy_(x_src)
                torch.cuda.synchronize()
                dist.barrier()
                start = time.perf_counter()
                for _ in range(args.iters):
                    dist.all_reduce(x)
                torch.cuda.synchronize()
                samples_us.append((time.perf_counter() - start) / args.iters * 1e6)

            const_result, tol_result = check(x_src, rank, world_size, args.tol_bf16)
            payload_bytes = n * ELEM_BYTES
            mean_us = statistics.mean(samples_us)
            row = {
                "label": label,
                "elements": n,
                "payload_bytes": payload_bytes,
                "iters": args.iters,
                "samples": len(samples_us),
                "mean_us": mean_us,
                "p50_us": percentile(samples_us, 0.50),
                "p99_us": percentile(samples_us, 0.99),
                "min_us": min(samples_us),
                "max_us": max(samples_us),
                "std_us": statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0,
                "cv_pct": (statistics.stdev(samples_us) / mean_us * 100.0
                           if len(samples_us) > 1 and mean_us else 0.0),
                "payload_gbs": payload_bytes / (mean_us * 1e-6) / 1e9,
                "correctness": {"constant": const_result, "tolerance": tol_result},
            }
            rows.append(row)
            ok = (const_result["pass"] is not False) and tol_result["pass"]
            all_pass = all_pass and ok
            if rank == 0:
                print(f"all_reduce bf16 [1,{n}] ({payload_bytes} B): "
                      f"mean {mean_us:.1f} p50 {row['p50_us']:.1f} "
                      f"p99 {row['p99_us']:.1f} us/call  "
                      f"bw {row['payload_gbs']:.1f} GB/s  "
                      f"const_err {const_result['max_abs_err']:.3g} "
                      f"tol_err {tol_result['max_rel_err']:.2e} "
                      f"{'OK' if ok else 'CHECK-FAIL'}", flush=True)

        if rank == 0:
            payload = {
                "meta": {
                    "tag": args.tag,
                    "backend": "nccl",
                    "world_size": world_size,
                    "device_names": [
                        torch.cuda.get_device_name(i)
                        for i in range(world_size)
                    ],
                    "torch": torch.__version__,
                    "torch_version_hip": getattr(torch.version, "hip", None),
                    "hostname": socket.gethostname(),
                    "started_at": started_at,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "iters": args.iters,
                    "warmup": args.warmup,
                    "samples": args.samples,
                    "tol_bf16": args.tol_bf16,
                    "all_pass": all_pass,
                    "env": env_snapshot(),
                },
                "sizes": rows,
            }
            if args.out:
                directory = os.path.dirname(os.path.abspath(args.out))
                os.makedirs(directory, exist_ok=True)
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=1, ensure_ascii=False)
                print(f"[ar_bench] wrote {args.out}", flush=True)
            else:
                print(json.dumps(payload, indent=1, ensure_ascii=False), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
