#!/usr/bin/env python3
"""Side-effect check for HSA_FORCE_FINE_GRAIN_PCIE=1 on Z100.

The variable forces fine-grained (coherent) PCIe mappings.  That is what RCCL
needs to open P2P channels, but coherent host mappings are historically slower
than write-combined ones, so before enabling it in production we measure the
transfers it could plausibly affect:

* pinned host -> device and device -> pinned host bandwidth,
* pageable host -> device bandwidth,
* same-device copy and cross-DCU peer copy bandwidth,
* a small fp16 GEMM checked against a float32 reference.

Run once per arm:

    docker exec <ctr> bash -lc 'source /src/tools/dcu/env.sh && \
        HIP_VISIBLE_DEVICES=0,1 python /src/tools/dcu/finegrain_side_effects.py'
    docker exec <ctr> bash -lc 'source /src/tools/dcu/env.sh && \
        HSA_FORCE_FINE_GRAIN_PCIE=1 HIP_VISIBLE_DEVICES=0,1 \
        python /src/tools/dcu/finegrain_side_effects.py'
"""

import os
import time

import torch

TAG = "HSA_FORCE_FINE_GRAIN_PCIE=%s" % os.environ.get(
    "HSA_FORCE_FINE_GRAIN_PCIE", "<unset>")
N = 64 << 20  # bytes per transfer
REPS = 20


def bw(fn, nbytes, reps=REPS):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return nbytes / best / 1e9, best * 1e3


def main():
    dev0, dev1 = 0, 1
    print("[%s] devices=%d" % (TAG, torch.cuda.device_count()))

    pinned = torch.empty(N, dtype=torch.uint8, pin_memory=True)
    pageable = torch.empty(N, dtype=torch.uint8)
    dst = torch.empty(N, dtype=torch.uint8, device=dev0)
    dst1 = torch.empty(N, dtype=torch.uint8, device=dev1)
    gb, ms = bw(lambda: dst.copy_(pinned, non_blocking=True), N)
    print("  pinned H2D    %6.2f GB/s (%7.3f ms)" % (gb, ms))
    gb, ms = bw(lambda: pinned.copy_(dst, non_blocking=True), N)
    print("  pinned D2H    %6.2f GB/s (%7.3f ms)" % (gb, ms))
    gb, ms = bw(lambda: dst.copy_(pageable, non_blocking=True), N)
    print("  pageable H2D  %6.2f GB/s (%7.3f ms)" % (gb, ms))
    gb, ms = bw(lambda: dst.copy_(dst), N)
    print("  local D2D     %6.2f GB/s (%7.3f ms)" % (gb, ms))
    gb, ms = bw(lambda: dst1.copy_(dst), N)
    print("  peer copy_    %6.2f GB/s (%7.3f ms)" % (gb, ms))

    # Numerics: fp16 GEMM against a float32 reference.
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device=dev0)
    b = torch.randn(1024, 1024, device=dev0)
    ref = (a.double() @ b.double())
    got = (a.half() @ b.half()).double()
    rel = ((got - ref).abs().max() / ref.abs().max()).item()
    print("  fp16 GEMM max rel err vs fp64: %.3e" % rel)

    free_b, total_b = torch.cuda.mem_get_info(dev0)
    print("  dev0 free %.2f GiB / %.2f GiB" % (free_b / 2**30, total_b / 2**30))
    del pinned, pageable, dst, dst1, a, b


if __name__ == "__main__":
    main()
