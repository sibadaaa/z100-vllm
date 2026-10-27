#!/usr/bin/env python3
"""Decide whether direct peer-to-peer is possible at each layer on Z100 (gfx906).

Separates three questions that a single "RCCL via SHM" symptom conflates:

1. **Driver/HIP capability**: what does `hipExtMallocWithFlags` return for every
   documented flag, at 4 KiB and 1 MiB, with the HIP error *name*?  RCCL's P2P
   transport allocates its shareable buffer with `hipDeviceMallocUncached` (the
   symbol is a dynamic import of `librccl.so`), so this is the gate for
   `via P2P` channels.
2. **Hardware/driver P2P**: can two DCUs touch each other's VRAM at all?
   `hipDeviceCanAccessPeer` + `hipMemcpyPeer` bandwidth answers that without
   RCCL in the way.
3. **IPC**: can a plain allocation be exported as an IPC handle (the other
   mechanism RCCL and vLLM's custom all-reduce could use)?

ctypes-only: no compiler, no kernels, no HIP binary is executed.  Every
allocation is freed before exit; the peer test uses 64 MiB.

Run inside a container with the repo mounted at /src:

    docker exec <ctr> bash -lc 'source /src/tools/dcu/env.sh && \
        HIP_VISIBLE_DEVICES=0,1 python /src/tools/dcu/p2p_capability_probe.py'
"""

import ctypes
import time

hip = ctypes.CDLL("libamdhip64.so")
hip.hipGetErrorName.restype = ctypes.c_char_p
hip.hipGetErrorString.restype = ctypes.c_char_p


class DeviceArch(ctypes.Structure):
    # 32 single-bit fields (see hipDeviceArch_t); c_uint32 == 4 bytes.
    _fields_ = [("bits", ctypes.c_uint32)]


class DeviceProp(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int * 3),
        ("maxGridSize", ctypes.c_int * 3),
        ("clockRate", ctypes.c_int),
        ("memoryClockRate", ctypes.c_int),
        ("memoryBusWidth", ctypes.c_int),
        ("totalConstMem", ctypes.c_size_t),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("multiProcessorCount", ctypes.c_int),
        ("l2CacheSize", ctypes.c_int),
        ("maxThreadsPerMultiProcessor", ctypes.c_int),
        ("computeMode", ctypes.c_int),
        ("clockInstructionRate", ctypes.c_int),
        ("arch", DeviceArch),
        ("concurrentKernels", ctypes.c_int),
        ("pciDomainID", ctypes.c_int),
        ("pciBusID", ctypes.c_int),
        ("pciDeviceID", ctypes.c_int),
        ("maxSharedMemoryPerMultiProcessor", ctypes.c_size_t),
        ("isMultiGpuBoard", ctypes.c_int),
        ("canMapHostMemory", ctypes.c_int),
        ("gcnArch", ctypes.c_int),
        ("gcnArchName", ctypes.c_char * 256),
        ("integrated", ctypes.c_int),
        ("cooperativeLaunch", ctypes.c_int),
        ("cooperativeMultiDeviceLaunch", ctypes.c_int),
        ("maxTexture1DLinear", ctypes.c_int),
        ("maxTexture1D", ctypes.c_int),
        ("maxTexture2D", ctypes.c_int * 2),
        ("maxTexture3D", ctypes.c_int * 3),
        ("hdpMemFlushCntl", ctypes.c_void_p),
        ("hdpRegFlushCntl", ctypes.c_void_p),
        ("memPitch", ctypes.c_size_t),
        ("textureAlignment", ctypes.c_size_t),
        ("texturePitchAlignment", ctypes.c_size_t),
        ("kernelExecTimeoutEnabled", ctypes.c_int),
        ("ECCEnabled", ctypes.c_int),
        ("tccDriver", ctypes.c_int),
        ("cooperativeMultiDeviceUnmatchedFunc", ctypes.c_int),
        ("cooperativeMultiDeviceUnmatchedGridDim", ctypes.c_int),
        ("cooperativeMultiDeviceUnmatchedBlockDim", ctypes.c_int),
        ("cooperativeMultiDeviceUnmatchedSharedMem", ctypes.c_int),
        ("isLargeBar", ctypes.c_int),
        ("asicRevision", ctypes.c_int),
        ("managedMemory", ctypes.c_int),
        ("directManagedMemAccessFromHost", ctypes.c_int),
        ("concurrentManagedAccess", ctypes.c_int),
        ("pageableMemoryAccess", ctypes.c_int),
        ("pageableMemoryAccessUsesHostPageTables", ctypes.c_int),
    ]


# Values verbatim from /opt/dtk/include/hip/hip_runtime_api.h.
FLAGS = [
    ("Default", 0x0),
    ("Finegrained", 0x1),
    ("hipMallocSignalMemory", 0x2),
    ("Uncached", 0x3),
    ("Contiguous", 0x4),
    ("UncachedXdp", 0x1000),
]


def err(rc):
    return "%s (%s)" % (hip.hipGetErrorName(rc).decode(),
                        hip.hipGetErrorString(rc).decode())


def main():
    print("sizeof(hipDeviceProp_t) as modelled =", ctypes.sizeof(DeviceProp))
    n = ctypes.c_int()
    rc = hip.hipGetDeviceCount(ctypes.byref(n))
    print("hipGetDeviceCount rc=%d %s -> %d device(s)" % (rc, err(rc), n.value))

    print("\n=== 1. device properties (hipDeviceProp_t) ===")
    for i in range(n.value):
        p = DeviceProp()
        rc = hip.hipGetDeviceProperties(ctypes.byref(p), ctypes.c_int(i))
        if rc != 0:
            print("  dev%d: rc=%d %s" % (i, rc, err(rc)))
            continue
        print("  dev%d name=%s gcnArchName=%s gcnArch=0x%x cap=(%d,%d)"
              % (i, p.name.decode(), p.gcnArchName.decode(), p.gcnArch,
                 p.major, p.minor))
        print("       integrated=%d managedMemory=%d canMapHostMemory=%d "
              "isLargeBar=%d" % (p.integrated, p.managedMemory,
                                 p.canMapHostMemory, p.isLargeBar))
        print("       pageableMemoryAccess=%d "
              "pageableMemoryAccessUsesHostPageTables=%d "
              "directManagedMemAccessFromHost=%d concurrentManagedAccess=%d"
              % (p.pageableMemoryAccess,
                 p.pageableMemoryAccessUsesHostPageTables,
                 p.directManagedMemAccessFromHost, p.concurrentManagedAccess))
        print("       totalGlobalMem=%.2f GiB  pci %04x:%02x:%02x"
              % (p.totalGlobalMem / 2**30, p.pciDomainID, p.pciBusID,
                 p.pciDeviceID))

    print("\n=== 2. hipExtMallocWithFlags, 4 KiB and 1 MiB (free VRAM first) ===")
    for size, tag in ((4096, "4 KiB"), (1 << 20, "1 MiB")):
        for name, flag in FLAGS:
            ptr = ctypes.c_void_p()
            rc = hip.hipExtMallocWithFlags(ctypes.byref(ptr),
                                           ctypes.c_size_t(size),
                                           ctypes.c_int(flag))
            print("  %-8s %-21s flag=0x%04x rc=%d  %s -> ptr=%s"
                  % (tag, name, flag, rc, err(rc),
                     hex(ptr.value) if ptr.value else "NULL"))
            if rc == 0 and ptr.value:
                hip.hipFree(ptr)

    print("\n=== 3. memory left on each device (hipMemGetInfo) ===")
    for i in range(n.value):
        hip.hipSetDevice(ctypes.c_int(i))
        free_b, total_b = ctypes.c_size_t(), ctypes.c_size_t()
        rc = hip.hipMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b))
        print("  dev%d rc=%d free=%.2f GiB total=%.2f GiB"
              % (i, rc, free_b.value / 2**30, total_b.value / 2**30))

    print("\n=== 4. hipDeviceCanAccessPeer ===")
    hip.hipSetDevice(ctypes.c_int(0))
    for i in range(n.value):
        row = []
        for j in range(n.value):
            can = ctypes.c_int()
            rc = hip.hipDeviceCanAccessPeer(ctypes.byref(can), ctypes.c_int(i),
                                            ctypes.c_int(j))
            row.append("dev%d->dev%d rc=%d can=%d" % (i, j, rc, can.value))
        print("  " + " | ".join(row))

    if n.value >= 2:
        print("\n=== 5. hipDeviceEnablePeerAccess(dev1, dev0) ===")
        hip.hipSetDevice(ctypes.c_int(0))
        rc = hip.hipDeviceEnablePeerAccess(ctypes.c_int(1), ctypes.c_uint(0))
        print("  rc=%d %s" % (rc, err(rc)))

        print("\n=== 6. real hipMemcpyPeer, 64 MiB ===")
        sz = 64 << 20
        a, b = ctypes.c_void_p(), ctypes.c_void_p()
        hip.hipSetDevice(ctypes.c_int(0))
        rc0 = hip.hipMalloc(ctypes.byref(a), ctypes.c_size_t(sz))
        hip.hipMemset(a, ctypes.c_int(0xAB), ctypes.c_size_t(sz))
        hip.hipSetDevice(ctypes.c_int(1))
        rc1 = hip.hipMalloc(ctypes.byref(b), ctypes.c_size_t(sz))
        print("  hipMalloc rc dev0=%d dev1=%d" % (rc0, rc1))
        if rc0 == 0 and rc1 == 0:
            hip.hipDeviceSynchronize()
            for dst, dstdev, src, srcdev, direction in (
                    (b, 1, a, 0, "dev0->dev1"),
                    (a, 0, b, 1, "dev1->dev0")):
                hip.hipDeviceSynchronize()
                t0 = time.perf_counter()
                rc = hip.hipMemcpyPeer(dst, ctypes.c_int(dstdev), src,
                                       ctypes.c_int(srcdev),
                                       ctypes.c_size_t(sz))
                hip.hipDeviceSynchronize()
                dt = time.perf_counter() - t0
                print("  %s rc=%d %s  %.2f ms  %.2f GB/s"
                      % (direction, rc, err(rc), dt * 1e3, sz / dt / 1e9))
            hip.hipFree(a)
            hip.hipFree(b)

    print("\n=== 7. hipIpcGetMemHandle on a plain allocation ===")
    hip.hipSetDevice(ctypes.c_int(0))
    p = ctypes.c_void_p()
    rc = hip.hipMalloc(ctypes.byref(p), ctypes.c_size_t(1 << 20))
    handle = (ctypes.c_char * 64)()
    rc2 = hip.hipIpcGetMemHandle(ctypes.byref(handle), p)
    print("  hipMalloc rc=%d, hipIpcGetMemHandle rc=%d %s" % (rc, rc2, err(rc2)))
    if rc == 0:
        hip.hipFree(p)

    print("\nNote: the Uncached flag is what librccl.so imports via "
          "hipExtMallocWithFlags for ncclP2pAllocateShareableBuffer; "
          "hipMemcpyPeer above is the hardware path without RCCL.")


if __name__ == "__main__":
    main()
