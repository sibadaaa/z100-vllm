#!/usr/bin/env python3
"""perf1: 4-rank RCCL all-reduce ladder over the REAL Qwen3.8-27B TP=4 payloads.

Sizes are the actual per-rank partial-output payloads of a 5120-hidden model in
fp16 (the dtype the service profile shows: ncclDevKernel_AllReduce_*_Sum_f16):
    2 KiB   =  1024 el   (ladder floor)
   10 KiB   =  5120 el   = [1,5120]  (MTP draft / single-token row)
   30 KiB   = 15360 el   = [3,5120]  (k=2 verify row, the dominant size)
  120 KiB   = 61440 el   = [12,5120] (conc-1 prefill tail / batched rows)
    1 MiB   = 524288 el

Runs alongside the live service on :8000; every rank reports its own timing so
the arrival-wait component is visible even in a quiet run.  Variants are driven
by the RCCL/NCCL env of the parent process (one process per variant).

Usage (through the container):
  env <VARIANT> torchrun --nproc_per_node=4 --master_port=29611 \
      tools/dcu/perf1_comm_ar.py
"""

import os
import time

import torch
import torch.distributed as dist

SIZES = [int(x) for x in os.environ.get(
    "PF1_SIZES", "1024,5120,15360,61440,524288").split(",")]
ITERS = int(os.environ.get("PF1_ITERS", "100"))
WARMUP = 10
REPEATS = 2

dist.init_process_group("nccl")
rank = dist.get_rank()
ws = dist.get_world_size()
torch.cuda.set_device(rank)
dev = f"cuda:{rank}"
tag = os.environ.get("PF1_TAG", "?")
name = torch.cuda.get_device_name(rank)
p2p = os.environ.get("HSA_FORCE_FINE_GRAIN_PCIE", "<unset>")
if rank == 0:
    print(
        f"[{tag}] world={ws} HSA_FORCE_FINE_GRAIN_PCIE={p2p} "
        f"ALGO={os.environ.get('NCCL_ALGO','<def>')} "
        f"PROTO={os.environ.get('NCCL_PROTO','<def>')} "
        f"MIN_CH={os.environ.get('NCCL_MIN_NCHANNELS','<def>')} "
        f"LL_TH={os.environ.get('NCCL_LL_THRESHOLD','<def>')}",
        flush=True,
    )
print(f"[{tag}][rank{rank}] {name}", flush=True)

for nel in SIZES:
    x = torch.ones(nel, dtype=torch.float16, device=dev)
    for _ in range(WARMUP):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(REPEATS):
        t0 = time.time()
        for _ in range(ITERS):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        times.append((time.time() - t0) / ITERS * 1e6)
        dist.barrier()
    mine = min(times)
    gathered = [None] * ws
    dist.all_gather_object(gathered, mine)
    if rank == 0:
        us = [round(v, 1) for v in gathered]
        print(
            f"[{tag}] bytes={nel * 2:8d} min={min(us):7.1f} us/call  per-rank={us}",
            flush=True,
        )
print(f"[{tag}] DONE", flush=True)
