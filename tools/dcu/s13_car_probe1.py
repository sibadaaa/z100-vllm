"""S13 probe 1: is the custom all-reduce plumbing present and usable on gfx906?

Single process, one visible DCU. Checks, in order:
  1. torch.ops._C_custom_ar.meta_size()  -- the import-time probe that sets
     custom_all_reduce.custom_ar.
  2. ops.allocate_shared_buffer_and_handle() -- on ROCm this goes through
     hipExtMallocWithFlags(hipDeviceMallocUncached), documented for MI200+.
Run with an explicit HIP_VISIBLE_DEVICES (hard constraint 25).
"""

import os

import torch  # noqa: F401  (must precede vllm: hard constraint 8)

print("torch", torch.__version__, "| visible", os.environ.get("HIP_VISIBLE_DEVICES"))
print("gcnArchName:", torch.cuda.get_device_properties(0).gcnArchName)

from vllm.distributed.device_communicators import custom_all_reduce as car

print("custom_all_reduce.custom_ar =", car.custom_ar)

from vllm import _custom_ops as ops

try:
    print("_C_custom_ar.meta_size() =", ops.meta_size())
except Exception as error:  # noqa: BLE001
    print("meta_size FAILED:", type(error).__name__, error)

for size in (1024 * 1024, 8 * 1024 * 1024 + 4096):
    try:
        ptr, handle = ops.allocate_shared_buffer_and_handle(size)
        print(f"allocate_shared_buffer_and_handle({size}) OK ptr={ptr} handle={handle.shape}")
        ops.free_shared_buffer(ptr)
    except Exception as error:  # noqa: BLE001
        print(f"allocate_shared_buffer_and_handle({size}) FAILED:",
              type(error).__name__, str(error)[:300])
print("PROBE1_DONE")
