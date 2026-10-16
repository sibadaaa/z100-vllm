#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Perf1: parity harness for the layout-aware gfx906 GDN MTP kernel.

Compares, over randomized production-shaped cases, three implementations of
the same op contract:

  v1      ops.fused_sigmoid_gating_delta_rule_gfx906_mtp_update  (production)
  triton  vllm...fla.ops.gfx906_gdn_mtp_triton.gdn_mtp_gfx906_triton (new)
  eager   torch reference _fused_sigmoid_gating_delta_rule_update_gfx906_eager

Every case runs all three on byte-identical inputs (independent state clones)
and reports: max|dout|, max|dstate|, the number of fp16 output elements that
differ (the model only consumes fp16), and for the state the number of fp32
elements that differ at all.  Cases include invalid (-1) state slots, which
the eager reference does not model, so those are compared v1-vs-triton only.

  HIP_VISIBLE_DEVICES=1 python tools/dcu/perf1_gdn_parity.py
"""

import argparse
import os

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    _fused_sigmoid_gating_delta_rule_update_gfx906_eager as eager_ref,
)
from vllm.model_executor.layers.fla.ops.gfx906_gdn_mtp_triton import (
    gdn_mtp_gfx906_triton,
)

SCALE = 128 ** -0.5


def build(seed, requests, t, h=4, hv=12, k=128, v=128, bad_slots=False):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    f16 = dict(device=dev, dtype=torch.float16)
    total = requests * t
    slots = requests * t
    cu = torch.arange(0, (requests + 1) * t, t, device=dev, dtype=torch.int32)
    # production gives every request/token its own slot; keep them distinct so
    # the v1 (batch-parallel) and eager (per-request) orderings agree.
    idx = torch.arange(slots, device=dev, dtype=torch.int32).reshape(requests, t)
    if bad_slots:
        idx[seed % requests, (seed // requests) % t] = -1
    accepted = torch.randint(
        1, t + 1, (requests,), device=dev, dtype=torch.int32, generator=g
    )
    return dict(
        A_log=(torch.randn(hv, generator=g, **f16) * 0.2),
        a=(torch.randn(total, hv, generator=g, **f16) * 0.2).contiguous(),
        b=(torch.randn(total, hv, generator=g, **f16) * 0.2).contiguous(),
        dt_bias=(torch.randn(hv, generator=g, **f16) * 0.1),
        q=(torch.randn(1, total, h, k, generator=g, **f16) * 2.5).contiguous(),
        k=(torch.randn(1, total, h, k, generator=g, **f16) * 2.5).contiguous(),
        v=(torch.randn(1, total, hv, v, generator=g, **f16) * 2.5).contiguous(),
        state=torch.randn(
            slots, hv, v, k, generator=g, device=dev, dtype=torch.float32
        )
        * 0.2,
        idx=idx,
        cu=cu,
        acc=accepted,
    )


def run_v1(d, state):
    return ops.fused_sigmoid_gating_delta_rule_gfx906_mtp_update(
        A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"], q=d["q"],
        k=d["k"], v=d["v"], state=state, state_indices=d["idx"],
        cu_seqlens=d["cu"], num_accepted_tokens=d["acc"], beta=1.0,
        threshold=20.0, scale=SCALE, use_qk_l2norm_in_kernel=True,
    ), state


def run_triton(d, state):
    return gdn_mtp_gfx906_triton(
        A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"], q=d["q"],
        k=d["k"], v=d["v"], state=state, state_indices=d["idx"],
        cu_seqlens=d["cu"], num_accepted_tokens=d["acc"], beta=1.0,
        threshold=20.0, scale=SCALE, use_qk_l2norm_in_kernel=True,
    ), state


def run_eager(d, state):
    return eager_ref(
        d["A_log"], d["a"], d["b"], d["dt_bias"], d["q"], d["k"], d["v"],
        1.0, 20.0, SCALE, state, True, d["cu"], d["idx"], d["acc"], True,
    )


def cmp_out(tag, got, ref):
    a = got.unsqueeze(0) if got.dim() == 3 else got
    b = ref.unsqueeze(0) if ref.dim() == 3 else ref
    do = (a.float() - b.float()).abs().max().item()
    n_fp16 = int((a.to(torch.float16) != b.to(torch.float16)).sum().item())
    return do, n_fp16, a.numel()


def cmp_state(got, ref):
    df = (got.float() - ref.float()).abs()
    return df.max().item(), int((df > 0).sum().item()), df.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()
    os.environ.setdefault("VLLM_GFX906_GDN_MTP_V2_DEBUG", "1")
    print(f"device={torch.cuda.get_device_name(0)}")
    worst = {"dout": 0.0, "fp16": 0, "dstate": 0.0, "state_diff": 0}
    cases = 0
    for seed in range(args.seeds):
        requests = 1 + seed % 3
        t = 1 + seed % 4
        bad = seed % 3 == 2
        d = build(seed, requests, t, bad_slots=bad)
        v1_out, v1_state = run_v1(d, d["state"].clone())
        tr_out, tr_state = run_triton(d, d["state"].clone())
        do, n16, nel = cmp_out("", tr_out, v1_out)
        ds, ns, nst = cmp_state(tr_state, v1_state)
        line = (
            f"seed={seed} R={requests} T={t} bad_slots={bad} "
            f"max|dout|={do:.3e} fp16-diff={n16}/{nel} "
            f"max|dstate|={ds:.3e} state-diff={ns}/{nst}"
        )
        if not bad:
            e_out, e_state = run_eager(d, d["state"].clone())
            eo, en16, enel = cmp_out("", v1_out, e_out)
            eds, ens, enst = cmp_state(v1_state, e_state)
            line += (
                f" | v1-vs-eager max|dout|={eo:.3e} fp16-diff={en16}/{enel} "
                f"max|dstate|={eds:.3e}"
            )
        print(line)
        cases += 1
        worst["dout"] = max(worst["dout"], do)
        worst["fp16"] = max(worst["fp16"], n16)
        worst["dstate"] = max(worst["dstate"], ds)
        worst["state_diff"] = max(worst["state_diff"], ns)
    print(
        f"\nworst over {cases} cases: max|dout|={worst['dout']:.3e} "
        f"fp16-output differing<={worst['fp16']} "
        f"max|dstate|={worst['dstate']:.3e} state-diff<={worst['state_diff']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
