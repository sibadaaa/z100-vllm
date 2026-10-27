"""S13 probe 1b: pin the root cause of the failed custom-AR buffer allocation.

Probe 1 showed ops.allocate_shared_buffer_and_handle() -> AcceleratorError
"out of memory" on an idle card.  That op calls
hipExtMallocWithFlags(size, hipDeviceMallocUncached) on ROCm.  Here we call the
HIP runtime directly through ctypes (no compiled artifacts, hard constraint 1)
and compare the four documented flags plus plain hipMalloc on the same size.

Run with an explicit HIP_VISIBLE_DEVICES (hard constraint 25).
"""

import ctypes
import os

MB = 1024 * 1024


def _load_hip() -> ctypes.CDLL:
    for path in (
        "/opt/dtk/lib/libamdhip64.so",
        "/opt/dtk/lib64/libamdhip64.so",
        "libamdhip64.so",
    ):
        if os.path.exists(path) or not path.startswith("/"):
            return ctypes.CDLL(path)
    raise OSError("libamdhip64.so not found")


import torch  # noqa: E402  (must precede the HIP calls: hard constraint 8)

torch.cuda.init()
torch.zeros(8, device="cuda:0")  # force a primary context
hip = _load_hip()

hip.hipGetErrorName.restype = ctypes.c_char_p
hip.hipGetErrorName.argtypes = [ctypes.c_int]

FLAGS = {
    "hipDeviceMallocDefault (0x0)": 0x0,
    "hipDeviceMallocFinegrained (0x1)": 0x1,
    "hipMallocSignalMemory (0x2)": 0x2,
    "hipDeviceMallocUncached (0x3)": 0x3,
}


def report(name: str, rc: int, ptr: int) -> None:
    print(f"  {name:34s} rc={rc:4d} ({hip.hipGetErrorName(rc).decode()}) ptr=0x{ptr:x}")


print("torch", torch.__version__, "| visible", os.environ.get("HIP_VISIBLE_DEVICES"))
print("gcnArchName:", torch.cuda.get_device_properties(0).gcnArchName)
print("plain hipMalloc(1 MiB):")
ptr = ctypes.c_void_p()
rc = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(MB))
report("hipMalloc", rc, ptr.value or 0)
if rc == 0:
    hip.hipFree(ptr)

print("hipExtMallocWithFlags(1 MiB, flag):")
for name, flag in FLAGS.items():
    ptr = ctypes.c_void_p()
    rc = hip.hipExtMallocWithFlags(ctypes.byref(ptr), ctypes.c_size_t(MB), flag)
    report(name, rc, ptr.value or 0)
    if rc == 0:
        hip.hipFree(ptr)

print("PROBE1B_DONE")
