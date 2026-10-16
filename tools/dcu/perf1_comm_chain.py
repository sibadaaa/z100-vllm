#!/usr/bin/env python3
"""perf1: staggered all-reduce chain -- does RCCL protocol latency reach the
decode-step critical path?

The quiet ladder (perf1_comm_ar.py) measures a collective with all four ranks
arriving together.  The service never does that: the ranks free-run between
collectives and arrive up to ~310 us apart (median, from the trace), so the
step pays only the *tail* (the last arriver's kernel), not the full latency.

This script reproduces the real pattern: each rank runs a rank-dependent
elementwise burst (the Z100 pair gets extra iterations -> a controlled ~300 us
stagger) and then all-reduces the real 30 KiB payload, 140 times per step for
50 steps -- the service's actual collective count/step.  Report per-rank wall
time of the chain; the max over ranks is the model of the decode step's comm
contribution.

Variants are driven by the parent env, exactly like perf1_comm_ar.py.
"""

import os
import time

import torch
import torch.distributed as dist

ELEMS = 15360  # 30 KiB fp16 = [3,5120], the dominant decode payload
COLLECTIVES_PER_STEP = 140
STEPS = 50
BURST_ELEMS = 2 * 1024 * 1024  # 4 MiB fp16 buffer -> ~1 pass ~10 us
TARGET_FAST_US = float(os.environ.get("PF1_FAST_US", "1200"))
TARGET_STAGGER_US = float(os.environ.get("PF1_STAGGER_US", "300"))

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
dev = f"cuda:{rank}"
tag = os.environ.get("PF1_TAG", "?")
slow = rank >= 2
if rank == 0:
    print(
        f"[{tag}] chain: {STEPS} steps x {COLLECTIVES_PER_STEP} x 30KiB, "
        f"target fast={TARGET_FAST_US}us stagger={TARGET_STAGGER_US}us "
        f"P2P={os.environ.get('HSA_FORCE_FINE_GRAIN_PCIE','<def>')} "
        f"PROTO={os.environ.get('NCCL_PROTO','<def>')} "
        f"ALGO={os.environ.get('NCCL_ALGO','<def>')}",
        flush=True,
    )

buf = torch.ones(BURST_ELEMS, dtype=torch.float16, device=dev)
x = torch.ones(ELEMS, dtype=torch.float16, device=dev)

# calibrate one burst pass
for _ in range(5):
    buf.mul_(1.0001)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    buf.mul_(1.0001)
torch.cuda.synchronize()
burst_us = (time.time() - t0) / 50 * 1e6

target = TARGET_FAST_US + (TARGET_STAGGER_US if slow else 0.0)
iters = max(1, int(round(target / burst_us)))
d = torch.empty(1, device=dev)
dist.all_reduce(d)
if rank == 0:
    print(f"[{tag}] burst={burst_us:.1f} us/pass -> fast={iters} iters, "
          f"slow={max(1, int(round((TARGET_FAST_US + TARGET_STAGGER_US) / burst_us)))}",
          flush=True)

for _ in range(3):
    for _ in range(iters):
        buf.mul_(1.0001)
    dist.all_reduce(x)
torch.cuda.synchronize()
dist.barrier()

t0 = time.time()
for _ in range(STEPS):
    for _ in range(COLLECTIVES_PER_STEP):
        for _ in range(iters):
            buf.mul_(1.0001)
        dist.all_reduce(x)
torch.cuda.synchronize()
wall = time.time() - t0
dist.barrier()
wall = torch.tensor([wall], device=dev)
gathered = [torch.zeros(1, device=dev) for _ in range(dist.get_world_size())]
dist.all_gather(gathered, wall)
if rank == 0:
    us = [round(v.item() * 1e6 / (STEPS * COLLECTIVES_PER_STEP), 1) for v in gathered]
    crit = max(us)
    print(
        f"[{tag}] chain per-collective us=[{us}] critical(max)={crit:.1f} "
        f"-> step(140 coll) = {crit * COLLECTIVES_PER_STEP / 1000:.1f} ms",
        flush=True,
    )
print(f"[{tag}] DONE", flush=True)
