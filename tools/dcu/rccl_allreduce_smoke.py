#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""4-rank all_reduce smoke test for the locally rebuilt gfx906 RCCL.

Run under torchrun inside the ROCm 10.1 runtime container after sourcing
``tools/dcu/env-rocm101.sh`` (which puts /mnt/game/rccl-gfx906/lib first on
LD_LIBRARY_PATH)::

    torchrun --standalone --nproc_per_node=4 tools/dcu/rccl_allreduce_smoke.py

The script checks two things the P0-R acceptance in
``docs/z100-torch216-rocm100-adapt-plan.md`` section 6.1-P0.5 requires:

1. the process really loaded the rebuilt ``librccl.so.1`` (not the ROCm SDK
   copy that has no gfx906 device code), and
2. ``all_reduce(SUM)`` over ``torch.ones`` returns ``world_size`` for both a
   5120-element and a 1 MiB fp32 tensor, with no NaN/Inf and no hang.

Exit status is non-zero if any check fails; the caller should keep the
NCCL_DEBUG=INFO log for diagnosis. A SIGALRM watchdog turns a hang into a
non-zero exit instead of blocking the job.

Example::

    NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=4 \\
        tools/dcu/rccl_allreduce_smoke.py --rccl-timeout 300
"""

import argparse
import datetime
import os
import signal
import sys

import torch
import torch.distributed as dist

DEFAULT_RCCL_LIB = "/mnt/game/rccl-gfx906/lib/librccl.so.1"

# (case name, element count, dtype). 5120 fp32 and 262144 fp32 = 1 MiB.
CASES = (
    ("5120-elem-fp32", 5120, torch.float32),
    ("1MiB-fp32", 256 * 1024, torch.float32),
)


def _loaded_librccl():
    """Return the path of librccl.so.1 mapped into this process, if any."""
    try:
        with open("/proc/self/maps", encoding="utf-8") as f:
            for line in f:
                if "librccl.so" in line:
                    return line.split()[-1]
    except OSError:
        return None
    return None


def _same_file(path_a, path_b):
    if not path_a or not path_b:
        return False
    if path_a == path_b:
        return True
    try:
        return os.path.realpath(path_a) == os.path.realpath(path_b)
    except OSError:
        return False


def _on_alarm(signum, frame):  # noqa: ARG001
    print(f"SMOKE FAIL: watchdog fired after {signum} signal (hang)", flush=True)
    # os._exit skips dist.destroy_process_group(), which is what we want for a
    # wedged collective; torchrun propagates the non-zero exit to the caller.
    os._exit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-rccl",
        default=os.environ.get("RCCL_SMOKE_EXPECT_RCCL", DEFAULT_RCCL_LIB),
        help="librccl.so.1 path the process is expected to have loaded",
    )
    parser.add_argument(
        "--skip-rccl-check",
        action="store_true",
        help="do not fail when the expected librccl path is not mapped",
    )
    parser.add_argument(
        "--rccl-timeout",
        type=int,
        default=300,
        help="watchdog seconds for the whole run (default: 300)",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(args.rccl_timeout)

    timeout = datetime.timedelta(seconds=min(args.rccl_timeout, 180))
    dist.init_process_group("nccl", timeout=timeout)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    host = os.environ.get("HOSTNAME", "?")
    if rank == 0:
        try:
            nccl_version = torch.cuda.nccl.version()
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            nccl_version = f"unknown ({exc})"
        print(
            f"[rank 0/{world}] host={host} torch={torch.__version__} "
            f"nccl={nccl_version} device={torch.cuda.get_device_name(local_rank)}",
            flush=True,
        )

    rccl_path = _loaded_librccl()
    rccl_ok = _same_file(rccl_path, args.expect_rccl)
    print(
        f"[rank {rank}] loaded rccl={rccl_path} "
        f"expected={args.expect_rccl} match={rccl_ok}",
        flush=True,
    )

    failures = []
    if not rccl_ok and not args.skip_rccl_check:
        failures.append(f"librccl mismatch: loaded={rccl_path}")
        if rank == 0:
            print(
                "SMOKE FAIL: rebuilt librccl not loaded; check that "
                f"{os.path.dirname(args.expect_rccl)} is first on LD_LIBRARY_PATH",
                flush=True,
            )

    for name, numel, dtype in CASES:
        x = torch.ones(numel, dtype=dtype, device=device)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(x).all().item())
        exact = bool(torch.eq(x, float(world)).all().item())
        nbytes = x.nelement() * x.element_size()
        print(
            f"[rank {rank}] case={name} bytes={nbytes} world={world} "
            f"dtype={dtype} finite={finite} all_eq_world={exact} "
            f"min={x.min().item()} max={x.max().item()} "
            f"status={'PASS' if finite and exact else 'FAIL'}",
            flush=True,
        )
        if not (finite and exact):
            failures.append(f"{name}: finite={finite} all_eq_world={exact}")
        dist.barrier()

    dist.barrier()
    if rank == 0:
        if failures:
            print(
                f"SMOKE RESULT: FAIL ({len(failures)} failing checks): {failures}",
                flush=True,
            )
        else:
            print(f"SMOKE RESULT: PASS ({len(CASES)} cases, world={world})", flush=True)
    dist.destroy_process_group()
    signal.alarm(0)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
