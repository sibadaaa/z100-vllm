#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Perf1: CPU semantic validation of the gfx906 GDN MTP Triton kernel.

Runs the kernel under Triton's interpreter on CPU and compares it against a
pure-torch port of the C++ v1 kernel's per-token slot logic (including the
negative-slot skip chain, which the torch *eager* reference in
fused_sigmoid_gating.py does not model: that reference `continue`s the whole
sequence when the initial slot index is negative, while the kernels only skip
the affected tokens and re-read the stale slot afterwards).

  TRITON_INTERPRET=1 python tools/dcu/perf1_gdn_interp.py

No GPU is used.
"""

import os

os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402

from vllm.model_executor.layers.fla.ops.gfx906_gdn_mtp_triton import (  # noqa: E402
    gdn_mtp_gfx906_triton,
)


def v1_semantics(
    A_log, a, b, dt_bias, q, k, v, state, idx, cu, acc, scale=0.125 ** 0.5
):
    """Pure-torch port of fused_sigmoid_gating_delta_rule_gfx906_mtp_update_kernel.

    Mirrors the C++ loop exactly: source column is `acc-1` for t == 0 else
    `t-1`; a token is skipped (out = 0, no state write) when the source or the
    destination index is negative; the source state is read from memory for
    every token.
    """
    requests, max_q = idx.shape
    heads, key_dim = q.shape[2], q.shape[3]
    kv_heads, value_dim = v.shape[2], v.shape[3]
    tokens = q.shape[1]
    out = torch.zeros(tokens, kv_heads, value_dim, dtype=q.dtype)
    st = state.clone().float()
    head_ratio = kv_heads // heads
    for r in range(requests):
        seq_start = int(cu[r])
        seq_end = int(cu[r + 1])
        init_pos = int(acc[r]) - 1
        for t in range(max_q):
            if t >= seq_end - seq_start:
                break
            token = seq_start + t
            src_pos = init_pos if t == 0 else t - 1
            src = int(idx[r, src_pos]) if src_pos >= 0 else -1
            dst = int(idx[r, t])
            if src < 0 or dst < 0:
                out[token] = 0
                continue
            for hv in range(kv_heads):
                qh = hv // head_ratio
                qt = q[0, token, qh].float()
                kt = k[0, token, qh].float()
                qt = qt * torch.rsqrt((qt * qt).sum() + 1e-6) * scale
                kt = kt * torch.rsqrt((kt * kt).sum() + 1e-6)
                x = a[token, hv].float() + dt_bias[hv].float()
                bx = 1.0 * x
                sp = torch.log1p(torch.exp(bx)) if bx <= 20.0 else x
                decay = torch.exp(-torch.exp(A_log[hv].float()) * sp)
                beta = torch.sigmoid(b[token, hv].float())
                h = st[src, hv] * decay
                dot = (h * kt[None, :]).sum(dim=1)
                delta = (v[0, token, hv].float() - dot) * beta
                hn = h + delta[:, None] * kt[None, :]
                st[dst, hv] = hn
                out[token, hv] = (hn * qt[None, :]).sum(dim=1).to(out.dtype)
    return out, st


def case(name, requests, t, heads, kv_heads, kd, vd, bad_slots, slots=8):
    torch.manual_seed(7)
    g = torch.Generator().manual_seed(3)
    total = requests * t
    A_log = torch.randn(kv_heads, generator=g) * 0.2
    a = torch.randn(total, kv_heads, generator=g) * 0.2
    b = torch.randn(total, kv_heads, generator=g) * 0.2
    dt_bias = torch.randn(kv_heads, generator=g) * 0.1
    q = torch.randn(1, total, heads, kd, generator=g) * 2.5
    k = torch.randn(1, total, heads, kd, generator=g) * 2.5
    v = torch.randn(1, total, kv_heads, vd, generator=g) * 2.5
    state = torch.randn(slots, kv_heads, vd, kd, generator=g) * 0.2
    cu = torch.arange(0, (requests + 1) * t, t, dtype=torch.int32)
    idx = torch.arange(requests * t, dtype=torch.int32).reshape(requests, t)
    if bad_slots:
        # break the chain: token 0 of request 0, and a mid-chain token of the
        # last request (exercises the "read the stale slot" path)
        idx[0, 0] = -1
        idx[requests - 1, t // 2] = -1
    acc = torch.tensor([t] * requests, dtype=torch.int32)

    ref_out, ref_state = v1_semantics(
        A_log, a, b, dt_bias, q, k, v, state, idx, cu, acc
    )
    st_in = state.clone()
    got = gdn_mtp_gfx906_triton(
        A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v,
        state=st_in, state_indices=idx, cu_seqlens=cu,
        num_accepted_tokens=acc, beta=1.0, threshold=20.0,
        scale=0.125 ** 0.5, use_qk_l2norm_in_kernel=True, block_v=4,
        num_warps=1,
    )
    got_state = st_in  # the op updates `state` in place
    dout = (got.float() - ref_out.float()).abs().max().item()
    dst = (got_state - ref_state).abs().max().item()
    print(
        f"{name:44s} R={requests} T={t} heads={heads}/{kv_heads} K={kd} V={vd} "
        f"bad_slots={int(bad_slots)}: max|dout|={dout:.3e} max|dstate|={dst:.3e}"
    )
    return dout, dst


def main():
    print("TRITON_INTERPRET=1 (CPU only, no GPU)")
    worst = 0.0
    for bad in (False, True):
        for requests in (1, 2):
            for t in (1, 3):
                d, s = case(
                    "small", requests, t, 2, 4, 32, 16, bad, slots=8
                )
                worst = max(worst, d, s)
    # one full production geometry
    d, s = case("production geometry", 1, 3, 4, 12, 128, 128, False, slots=8)
    worst = max(worst, d, s)
    d, s = case("production geometry + bad slots", 1, 3, 4, 12, 128, 128, True,
                slots=8)
    worst = max(worst, d, s)
    print(f"worst |d| over all cases: {worst:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
