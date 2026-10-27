"""S13 probe 2: construct vLLM's CustomAllreduce for real in a 2-rank gloo
group on two DCUs, exactly the way CudaCommunicator does it, and report the
outcome (plus the init path decisions).

Run:
  HIP_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --standalone \
      tools/dcu/s13_car_probe2.py

No compiled artifacts are produced or executed (hard constraint 1); this only
calls the already-built vllm._C_custom_ar ops from Python.
"""

import os
import traceback

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="gloo")
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        print(f"world_size={world_size} gloo group; gcnArch="
              f"{torch.cuda.get_device_properties(rank).gcnArchName}")

    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
        custom_ar,
    )

    if rank == 0:
        print("custom_all_reduce.custom_ar =", custom_ar,
              "(library-level probe at import time)")
        from vllm import _custom_ops as ops
        print("meta_size() =", ops.meta_size(), "bytes")

    dist.barrier()
    try:
        comm = CustomAllreduce(group=dist.group.WORLD, device=device)
        dist.barrier()
        if rank == 0:
            print(f"CONSTRUCT_OK disabled={comm.disabled} ptr={comm._ptr} "
                  f"fully_connected={comm.fully_connected}")
        if not comm.disabled:
            for nbytes in (10240, 1024 * 1024, 2 * 1024 * 1024):
                n = nbytes // 2
                x = torch.full((n,), float(rank + 1), dtype=torch.float16, device=device)
                ref = x.clone()
                out = comm.custom_all_reduce(x)
                dist.barrier()
                dist.all_reduce(ref)
                matches = out is not None and torch.equal(out, ref)
                if rank == 0:
                    print(f"  bytes={nbytes:8d} custom_ar_used={out is not None} "
                          f"bitwise_equal_to_gloo={matches}")
    except Exception as error:  # noqa: BLE001
        if rank == 0:
            print("CONSTRUCT_FAILED:", type(error).__name__)
            print(str(error)[:400])
        tb = traceback.format_exc()
        if rank == 0:
            print("---- traceback ----")
            print("\n".join(tb.strip().splitlines()[-14:]))
    if rank == 0:
        print("PROBE2_DONE")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
