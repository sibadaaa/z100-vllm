#!/usr/bin/env python3
"""S33 WP2 bisection: reproduce the service-side custom-AR corruption in a
torchrun bench by mimicking the service's call pattern:

  * torch.inference_mode()
  * FRESH input tensor pointer every call (service allocates per step),
  * input produced by a device kernel (GEMV-like op) right before the AR,
  * correctness checked against dist.all_reduce (NCCL) every call.

Usage:
  torchrun --nproc_per_node=4 --standalone tools/dcu/car_bisect.py \
      [--mode fresh|same|inference] [--iters 50]
Exit 0 iff all calls match (tolerance 2e-2 vs NCCL fp16 sum); prints the
first mismatching iteration otherwise.
"""

import argparse
import os
import sys

EXIT_OK = 0
EXIT_MISMATCH = 5
EXIT_DISABLED = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fresh",
                    choices=["fresh", "same", "inference", "fresh_inf"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--size", type=int, default=5120)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    os.environ["VLLM_DCU_CUSTOM_AR"] = "1"
    os.environ["VLLM_DCU_CUSTOM_AR_ALLOC"] = "finegrained"
    os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] = "oneshot"

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl")
    if not dist.is_gloo_available():
        print("no gloo", flush=True)
        return EXIT_DISABLED
    cpu_group = dist.new_group(ranks=list(range(world)), backend="gloo")
    dist.barrier(group=cpu_group)

    from vllm.distributed.device_communicators import custom_all_reduce as car
    ca = car.CustomAllreduce(group=cpu_group, device=device)
    ok = torch.tensor([0 if ca.disabled else 1], dtype=torch.int32)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=cpu_group)
    if not bool(ok.item()):
        print(f"[rank{rank}] custom AR disabled", flush=True)
        return EXIT_DISABLED

    inf = args.mode in ("inference", "fresh_inf")
    fresh = args.mode in ("fresh", "fresh_inf")

    ctx = torch.inference_mode(inf)
    mismatches = 0
    first_bad = None
    with ctx:
        gen = torch.Generator(device="cpu").manual_seed(1000 + rank)
        x0 = (torch.randn(args.size, generator=gen, dtype=torch.float32)
              * 0.05).to(device=device, dtype=torch.float16)
        x = x0.clone()
        for it in range(args.iters):
            if fresh:
                # fresh pointer each call, values written by a device kernel
                # (mul) like the service's GEMV output feeding the AR
                x = (x0 * (1.0 + 0.001 * it)).contiguous()
            ref = x.clone()
            dist.all_reduce(ref)
            out = ca.custom_all_reduce(x)
            torch.cuda.synchronize()
            d = (out.float() - ref.float()).abs().max().item()
            if d > 2e-2:
                mismatches += 1
                if first_bad is None:
                    first_bad = (it, d)
    tag = (f"rank{rank} mode={args.mode} iters={args.iters} "
           f"mismatches={mismatches}")
    if first_bad:
        tag += f" first_bad=(iter {first_bad[0]}, max|d|={first_bad[1]:.3e})"
    print(f"[bisect] {tag}", flush=True)
    worst = torch.tensor([mismatches], dtype=torch.int64)
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    if int(worst.item()) > 0:
        return EXIT_MISMATCH
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
