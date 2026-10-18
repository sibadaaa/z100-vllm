#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-test for the Hygon DCU DP4X Triton launch guard (F24 workaround).

Runs a sentinel Triton kernel that compiles to the gfx906 VGPR maximum when
launched with few warps. On a DP4X (Z100) board under an upstream-derived HIP
runtime the unguarded kernel silently commits only a prefix of its first store;
with the guard installed (vllm.platforms.hygon_dp4x) the launch is forced to
>=8 warps, which lowers the VGPR count and produces a correct result.

Run inside the ROCm 10.1 runtime container:

    HIP_VISIBLE_DEVICES=<n> python3 tools/dcu/dp4x_guard_selftest.py

Exit code 0 = correct result (guard works or board does not need it),
1 = partial/wrong result.
"""

from __future__ import annotations

import argparse
import json
import sys

BLOCK = 16384
WARPS = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--warps", type=int, default=WARPS)
    ap.add_argument("--no-install", action="store_true")
    args = ap.parse_args()

    import torch
    import triton
    import triton.language as tl

    import vllm.platforms.hygon_dp4x as dp4x

    status = dp4x.dp4x_guard_status()
    print("guard_status:", json.dumps(status, sort_keys=True), flush=True)
    if not args.no_install:
        installed = dp4x.install_dp4x_triton_guard()
        print(f"install_dp4x_triton_guard() -> {installed}", flush=True)

    @triton.jit
    def k_wide(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        z = x * 2.0 + y
        w = z * 3.0 - x
        tl.store(o_ptr + offs, z + w)

    dev = "cuda"
    torch.manual_seed(0)
    x = torch.randn(args.block, device=dev)
    y = torch.randn(args.block, device=dev)
    o = torch.full((args.block,), -7.0, device=dev)
    k_wide[(1,)](x, y, o, BLOCK=args.block, num_warps=args.warps)
    torch.cuda.synchronize()

    written = int((o != -7.0).sum().item())
    finite = bool(torch.isfinite(o).all())
    ref = 7.0 * x + 4.0 * y
    maxdiff = (
        (o.float() - ref.float()).abs().max().item()
        if written == args.block
        else float("inf")
    )
    props = torch.cuda.get_device_properties(0)
    kernel_info = []
    for device_cache in k_wide.device_caches.values():
        for compiled in device_cache[0].values():
            kernel_info.append(
                {
                    "n_regs": getattr(compiled, "n_regs", None),
                    "n_spills": getattr(compiled, "n_spills", None),
                }
            )
    print(
        f"pci_bus={props.pci_bus_id} requested_warps={args.warps} "
        f"written={written}/{args.block} finite={finite} maxdiff={maxdiff} "
        f"kernel={kernel_info}",
        flush=True,
    )
    ok = written == args.block and finite and maxdiff < 1e-2
    print("PASS" if ok else "FAIL (partial commit / wrong result)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
