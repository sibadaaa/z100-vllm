#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Perf1: populate the shared Triton cache for the GDN MTP kernel.

The A/B arm `VLLM_QWEN35_MTP_GDN_TRITON=1` compiles this kernel on the first
decode step of the first request unless the exact production specialization is
already in TRITON_CACHE_DIR.  This script builds the production decode shape
(1 request, 3 speculative tokens, 12 value heads, 4 q/k heads, head dims 128,
float32 state in the native [slot][head][v][k] layout) and launches the kernel
a few times for the default (BV, num_warps) plus the sweep configs, so the arm
starts warm.

  HIP_VISIBLE_DEVICES=1 python tools/dcu/perf1_gdn_warm.py [--configs 4:1 ...]

GPU-1 only, ~1 launch per config plus JIT compile time.
"""

import argparse
import os

import torch

from vllm.model_executor.layers.fla.ops.gfx906_gdn_mtp_triton import (
    gdn_mtp_gfx906_triton,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        nargs="+",
        default=["4:1", "1:1", "8:1", "16:1", "4:2"],
        help="BV:num_warps pairs to compile",
    )
    args = ap.parse_args()
    torch.manual_seed(3)
    g = torch.Generator(device="cuda").manual_seed(3)
    t, h, hv, kd, vd, slots = 3, 4, 12, 128, 128, 8
    f16 = dict(device="cuda", dtype=torch.float16)
    d = dict(
        A_log=torch.randn(hv, generator=g, **f16) * 0.2,
        a=torch.randn(1, t, hv, generator=g, **f16) * 0.2,
        b=torch.randn(1, t, hv, generator=g, **f16) * 0.2,
        dt_bias=torch.randn(hv, generator=g, **f16) * 0.1,
        q=torch.randn(1, t, h, kd, generator=g, **f16) * 2.5,
        k=torch.randn(1, t, h, kd, generator=g, **f16) * 2.5,
        v=torch.randn(1, t, hv, vd, generator=g, **f16) * 2.5,
        state=torch.randn(slots, hv, vd, kd, generator=g, device="cuda") * 0.2,
        idx=torch.tensor([[1, 2, 3]], device="cuda", dtype=torch.int32),
        cu=torch.tensor([0, t], device="cuda", dtype=torch.int32),
        acc=torch.tensor([t], device="cuda", dtype=torch.int32),
    )
    print(
        f"device={torch.cuda.get_device_name(0)} "
        f"cache={os.getenv('TRITON_CACHE_DIR')} configs={args.configs}"
    )
    for cfg in args.configs:
        bv, nw = (int(x) for x in cfg.split(":"))
        out = gdn_mtp_gfx906_triton(
            A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"],
            q=d["q"], k=d["k"], v=d["v"], state=d["state"],
            state_indices=d["idx"], cu_seqlens=d["cu"],
            num_accepted_tokens=d["acc"], beta=1.0, threshold=20.0,
            scale=128 ** -0.5, use_qk_l2norm_in_kernel=True,
            block_v=bv, num_warps=nw,
        )
        torch.cuda.synchronize()
        print(f"  compiled+launched BV={bv} warps={nw} out={tuple(out.shape)}")
    print("cache warm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
