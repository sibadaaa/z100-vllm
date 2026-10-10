# S12: 4-rank all-reduce benchmark, modelled on the user's
# /home/lyh/server/diag/ar_bench.py.  Same protocol (bf16, 20 warmup + 300
# timed iterations, us/call) but with S12 additions:
#   * sizes 10240/20480/40960 kept, plus 1048576 (2 MiB) to exercise the
#     transport that actually uses the PCIe fabric (small messages go through
#     RCCL's SHM path -- see the NCCL_DEBUG=INFO topology capture);
#   * every size is measured 3 times and the MINIMUM is reported, because the
#     first sweep showed >70% run-to-run spread on the 10 KB shape.
#
# Run through torchrun (4 ranks) -- never as a self-compiled HIP binary.
import os
import time

import torch
import torch.distributed as dist

SIZES = [5120, 10240, 20480, 40960, 1048576]
ITERS = 300
WARMUP = 20
REPEATS = 3

dist.init_process_group("nccl")
rank = dist.get_rank()
ws = dist.get_world_size()
torch.cuda.set_device(rank)
dev = f"cuda:{rank}"
tag = os.environ.get("S12_TAG", "?")
print(f"[{tag}][rank{rank}] world={ws} dev={torch.cuda.get_device_name(rank)}", flush=True)

for n in SIZES:
    x = torch.ones(1, n, dtype=torch.bfloat16, device=dev)
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
        dist.barrier()
        times.append((time.time() - t0) / ITERS * 1e6)
    if rank == 0:
        print(
            f"[{tag}] all_reduce bf16 [1,{n}] ({n * 2} B): "
            f"min {min(times):.1f} med {sorted(times)[1]:.1f} max {max(times):.1f} us/call",
            flush=True,
        )
