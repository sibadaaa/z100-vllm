import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank = dist.get_rank(); ws = dist.get_world_size()
torch.cuda.set_device(rank)
dev = f"cuda:{rank}"
print(f"[rank{rank}] world={ws} dev={torch.cuda.get_device_name(rank)}", flush=True)
for n in (5120, 10240, 40960):
    x = torch.ones(1, n, dtype=torch.bfloat16, device=dev)
    for _ in range(20): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.time()
    for _ in range(300): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    us = (time.time() - t0) / 300 * 1e6
    if rank == 0:
        print(f"all_reduce bf16 [1,{n}] ({n*2} B): {us:.1f} us/call", flush=True)
