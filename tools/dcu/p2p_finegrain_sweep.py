#!/usr/bin/env python3
"""Message-size sweep for the RCCL fine-grain-PCIe A/B on Z100.

RCCL refuses its P2P transport unless every peer reports
`hipDeviceAttributeFineGrainSupport == 1` and it can allocate the shareable
buffer with `hipDeviceMallocFinegrained`.  On this platform both depend on
`HSA_FORCE_FINE_GRAIN_PCIE=1`, which RCCL itself asks for at startup
("Missing HSA_FORCE_FINE_GRAIN_PCIE=1 from environment ...").

This probe reports, for each message size, the all-reduce latency and
busbandwidth plus the value of the fine-grain attribute, so the P2P-vs-SHM
crossover can be read off directly.  It also verifies the reduction result so
turning the switch on cannot silently corrupt collectives.

Run inside a container with the repo mounted at /src, once per arm:

    docker exec <ctr> bash -lc 'source /src/tools/dcu/env.sh && \
        HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        /src/tools/dcu/p2p_finegrain_sweep.py shm'
    docker exec <ctr> bash -lc 'source /src/tools/dcu/env.sh && \
        HSA_FORCE_FINE_GRAIN_PCIE=1 HIP_VISIBLE_DEVICES=0,1,2,3 \
        torchrun --standalone --nproc_per_node=4 \
        /src/tools/dcu/p2p_finegrain_sweep.py finegrain'
"""

import ctypes
import hashlib
import os
import re
import sys

import torch
import torch.distributed as dist

TAG = sys.argv[1] if len(sys.argv) > 1 else "?"
# (bytes, iters): small messages dominate decode, large ones prefill.
SIZES = [(5120, 300), (10240, 300), (20480, 300), (65536, 200),
         (262144, 100), (1048576, 60), (4194304, 30)]


def fine_grain_attribute():
    """Resolve hipDeviceAttributeFineGrainSupport from the DTK header."""
    header = "/opt/dtk/include/hip/hip_runtime_api.h"
    value = 20004  # DTK 24.04; re-derived below when the header is readable
    try:
        src = open(header).read()
        body = re.search(r"typedef enum hipDeviceAttribute_t\s*\{(.*?)\}"
                         r"\s*hipDeviceAttribute_t", src, re.S).group(1)
        symbols, cur = {}, -1
        for line in body.splitlines():
            line = line.split("//")[0].strip().rstrip(",")
            if not line:
                continue
            if "=" in line:
                name, expr = line.split("=", 1)
                name = name.strip()
                cur = eval(expr.strip(), {}, symbols)
            else:
                name, cur = line, cur + 1
            symbols[name] = cur
            if "FineGrainSupport" in name:
                value = cur
    except Exception as exc:  # header missing -> keep the known constant
        print("  (attribute enum lookup failed: %s; using %d)" % (exc, value))
    hip = ctypes.CDLL("libamdhip64.so")
    out = ctypes.c_int()
    rc = hip.hipDeviceGetAttribute(ctypes.byref(out), ctypes.c_int(value),
                                   ctypes.c_int(torch.cuda.current_device()))
    return value, rc, out.value


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)

    if rank == 0:
        attr, rc, val = fine_grain_attribute()
        print("[%s] HSA_FORCE_FINE_GRAIN_PCIE=%s world=%d "
              "hipDeviceAttributeFineGrainSupport(%d) rc=%d value=%d"
              % (TAG, os.environ.get("HSA_FORCE_FINE_GRAIN_PCIE", "<unset>"),
                 world, attr, rc, val), flush=True)

    for nbytes, iters in SIZES:
        nelem = nbytes // 2  # bf16
        x = torch.ones(nelem, dtype=torch.bfloat16, device=rank)
        for _ in range(20):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            dist.all_reduce(x)
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) / iters * 1000
        # The timed loop reduces the same buffer iters times, so re-seed before
        # checking the value (ones * world).
        x.fill_(1.0)
        dist.all_reduce(x)
        torch.cuda.synchronize()
        ok = "OK" if (rank != 0 or abs(x[0].item() - world) < 1e-3) else "FAIL"
        if rank == 0:
            busbw = nbytes * 2 * (world - 1) / world
            print("[%s] n=%9d B %10.2f us/call  busbw=%7.3f GB/s  sum=%s"
                  % (TAG, nbytes, us, busbw / (us * 1e-6) / 1e9, ok),
                  flush=True)
        del x

    # Bitwise fingerprint of a deterministic reduction, to compare arms.
    v = torch.arange(1024, dtype=torch.float32, device=rank) + rank * 100.0
    dist.all_reduce(v)
    torch.cuda.synchronize()
    if rank == 0:
        print("[%s] fingerprint sum=%.10f sha256=%s"
              % (TAG, v.sum().item(),
                 hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()[:32]),
              flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
