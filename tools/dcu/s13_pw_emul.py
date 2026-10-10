"""S13: quantify the *mechanism* that ACS accelerates -- a direct peer write --
at the exact decode all-reduce message size, since the real vLLM
CustomAllreduce cannot be constructed on gfx906 (see probes 1/1b/2).

Pure torch, no compiled artifacts.  This is an emulation, not the vLLM custom
AR kernel: it measures the cost of the peer memory traffic that custom AR would
use, so the S12 NCCL all-reduce number (~88-100 us/call at 10 KB) can be
compared against it.

Run with an explicit HIP_VISIBLE_DEVICES (hard constraint 25): 0,1
"""

import statistics
import sys

import torch

ITERS = 200


def timeit(fn, iters: int = ITERS) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us per op


def main() -> None:
    if torch.cuda.device_count() < 2:
        print("need 2 visible devices")
        sys.exit(1)
    print("torch", torch.__version__,
          "| devices", [torch.cuda.get_device_properties(i).gcnArchName
                        for i in range(torch.cuda.device_count())])
    cases = [
        ("decode AR msg (5120 fp16 = 10240 B)", 5120),
        ("1 MiB (524288 fp16)", 524288),
        ("2 MiB (1048576 fp16)", 1048576),
        ("128 MiB (67108864 fp16)", 67108864),
    ]
    for label, n in cases:
        a = torch.zeros(n, dtype=torch.float16, device="cuda:0")
        b = torch.zeros(n, dtype=torch.float16, device="cuda:1")
        a.fill_(1.0)
        b.fill_(2.0)
        nbytes = n * 2
        # warm-up
        for _ in range(10):
            b.copy_(a)
            a.copy_(b)
        torch.cuda.synchronize()

        one_way = timeit(lambda: b.copy_(a))
        local = timeit(lambda: a.copy_(a))
        round_trip = timeit(lambda: (b.copy_(a), a.copy_(b)))
        torch.cuda.synchronize()
        assert torch.equal(b, torch.ones_like(b)) and torch.equal(a, torch.ones_like(a))
        gbps = nbytes / (one_way * 1e-6) / 1e9
        print(f"{label:38s} one-way peer write {one_way:9.2f} us "
              f"({gbps:7.2f} GB/s effective) | local copy {local:8.2f} us | "
              f"round trip {round_trip:9.2f} us")
    print("PW_EMUL_DONE")


if __name__ == "__main__":
    main()
