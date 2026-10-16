#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Perf1: standalone cost of the gfx906 GDN MTP decode kernel at the real shape.

Production shape (Qwen3.5-27B-FP8, TP=4 rank): 1 request, T=3 speculative
tokens, 12 value heads (48/4), 4 q/k heads (16/4), key/value head dim 128,
float32 SSM state, native cache layout [slots, heads, head_v_dim, head_k_dim]
(k contiguous -> stride(2)=128, stride(3)=1).

  HIP_VISIBLE_DEVICES=1 python tools/dcu/perf1_gdn_bench.py

Measures, per implementation, the pure GPU time of the launched kernel
(torch profiler) and the end-to-end per-call time (wall/event loop):

  v1      ops.fused_sigmoid_gating_delta_rule_gfx906_mtp_update, native state
          (the production path: grid (6,1,1), 256 threads)
  v2      same op with VLLM_GFX906_GDN_MTP_V2=1 and a v-contiguous state view
          (grid (12,1,1): the warp-cooperative kernel that production misses)
  eager   torch reference on the same layout (correctness anchor)
"""

import argparse
import os
import time

os.environ.setdefault("VLLM_GFX906_GDN_MTP_V2_DEBUG", "1")

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (  # noqa: E402
    _fused_sigmoid_gating_delta_rule_update_gfx906_eager as eager_ref,
)
from vllm.model_executor.layers.fla.ops.gfx906_gdn_mtp_triton import (  # noqa: E402
    gdn_mtp_gfx906_triton,
)

SLOTS = 16


def build(t=3, h=4, hv=12, k=128, v=128, slots=SLOTS, seed=11, layout="native"):
    torch.manual_seed(seed)
    dev = "cuda"
    fp16 = dict(device=dev, dtype=torch.float16)
    A_log = (torch.randn(hv, **fp16) * 0.2).contiguous()
    a = (torch.randn(1, t, hv, **fp16) * 0.2).contiguous()
    b = (torch.randn(1, t, hv, **fp16) * 0.2).contiguous()
    dt_bias = (torch.randn(hv, **fp16) * 0.1).contiguous()
    q = (torch.randn(1, t, h, k, **fp16) * 2.5).contiguous()
    kk = (torch.randn(1, t, h, k, **fp16) * 2.5).contiguous()
    vv = (torch.randn(1, t, hv, v, **fp16) * 2.5).contiguous()
    # logical H[v][k]; native cache keeps H as [v][k] with k contiguous
    h_state = torch.randn(slots, hv, v, k, device=dev, dtype=torch.float32) * 0.2
    if layout == "native":
        state = h_state.contiguous()
    elif layout == "transposed":
        # v-contiguous view of the same physical bytes is impossible without a
        # copy; build the real [k][v] buffer (contiguous) whose transpose view
        # is v-contiguous.
        state = h_state.transpose(-1, -2).contiguous()
    else:
        raise ValueError(layout)
    # One distinct state slot per token (slot 0 stays unused, matching the real
    # pool's NULL_BLOCK_ID convention); the table must have >= t columns, so it
    # is generated rather than hard-coded -- T = 8 is the DFlash2 k=7 shape.
    idx = torch.arange(1, t + 1, device=dev, dtype=torch.int32).unsqueeze(0)
    cu = torch.tensor([0, t], device=dev, dtype=torch.int32)
    acc = torch.tensor([t], device=dev, dtype=torch.int32)
    return dict(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=kk, v=vv,
                state=state, idx=idx, cu=cu, acc=acc, layout=layout)


def call_op(d, state):
    out = ops.fused_sigmoid_gating_delta_rule_gfx906_mtp_update(
        A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"],
        q=d["q"], k=d["k"], v=d["v"], state=state, state_indices=d["idx"],
        cu_seqlens=d["cu"], num_accepted_tokens=d["acc"], beta=1.0,
        threshold=20.0, scale=0.125 ** 0.5, use_qk_l2norm_in_kernel=True,
    )
    return out, state


def call_triton(d, state, bv=None, warps=None):
    out = gdn_mtp_gfx906_triton(
        A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"],
        q=d["q"], k=d["k"], v=d["v"], state=state,
        state_indices=d["idx"], cu_seqlens=d["cu"],
        num_accepted_tokens=d["acc"], beta=1.0, threshold=20.0,
        scale=0.125 ** 0.5, use_qk_l2norm_in_kernel=True,
        block_v=bv, num_warps=warps,
    )
    return out, state


def call_eager(d, state):
    o, st = eager_ref(
        d["A_log"], d["a"], d["b"], d["dt_bias"], d["q"], d["k"], d["v"],
        1.0, 20.0, 0.125 ** 0.5, state, True, d["cu"], d["idx"], d["acc"], True,
    )
    return o, st


def event_us(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def kernel_us(fn, iters=22, warmup=3, match=("fused_sigmoid", "gdn_mtp")):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    rows = [
        x
        for x in prof.key_averages()
        if any(m in x.key for m in match)
    ]
    if not rows:
        return None, None
    r = max(rows, key=lambda x: x.self_device_time_total)
    return r.self_device_time_total / iters, r.count / iters


def enqueue_us(fn, iters):
    """CPU time to enqueue one call while the GPU drains the queue."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / iters * 1e6


def _as4d(o):
    # the C++ op returns [tokens, kv_heads, value_dim]; the torch reference
    # returns [1, tokens, kv_heads, value_dim].
    return o.unsqueeze(0) if o.dim() == 3 else o


def parity(name, got, ref, atol=2e-3):
    a = _as4d(got[0]).float()
    b = ref[0].float()
    do = (a - b).abs().max().item()
    ds = (got[1].float() - ref[1].float()).abs().max().item()
    # how many elements survive the fp16 cast identically: the model only ever
    # consumes the fp16 output, so this is the number that matters downstream.
    fa = _as4d(got[0]).to(torch.float16)
    fb = ref[0].to(torch.float16)
    n_diff = int((fa != fb).sum().item())
    print(
        f"  parity {name}: max|dout|={do:.3e} max|dstate|={ds:.3e} "
        f"fp16-out differing={n_diff}/{fa.numel()}"
    )
    return do, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tokens", type=int, default=3)
    ap.add_argument("--skip-eager", action="store_true")
    ap.add_argument("--bv-list", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--warps-list", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()

    t = args.tokens
    native = build(t=t, layout="native")
    trans = build(t=t, layout="transposed")
    print(f"device={torch.cuda.get_device_name(0)} T={t} heads=4 kv_heads=12 "
          f"k=128 v=128 state_dtype=float32")
    print(f"native state shape={tuple(native['state'].shape)} "
          f"strides={native['state'].stride()}")
    print(f"transposed-buffer view stride={trans['state'].transpose(-1, -2).stride()}")
    print(f"VLLM_GFX906_GDN_MTP_V2={os.getenv('VLLM_GFX906_GDN_MTP_V2')!r}")

    # --- correctness: production path vs the torch reference on the native
    # layout (identical inputs, identical written state slots).
    ref = call_eager(native, native["state"].clone())
    got1 = call_op(native, native["state"].clone())
    print("v1 native  (production path):")
    parity("v1-vs-eager", got1, ref)

    # --- triton kernel on the same native layout, fresh state
    got3 = call_triton(native, native["state"].clone())
    print("triton native (new path):")
    parity("triton-vs-eager", got3, ref)
    parity("triton-vs-v1", got3, got1)
    if not args.skip_eager:
        print("eager reference:")
        eager_us = event_us(
            lambda: call_eager(native, native["state"].clone()), 20
        )
        print(f"  us/call={eager_us:.1f}")

    # --- v1 on the transposed view: same kernel, coalesced value strides.
    got1t = call_op(trans, trans["state"].transpose(-1, -2))
    ref_t = call_eager(trans, trans["state"].transpose(-1, -2).clone())
    print("v1 transposed-view (coalescing probe):")
    parity("v1t-vs-eager", got1t, ref_t)

    os.environ["VLLM_GFX906_GDN_MTP_V2"] = "1"
    # v2 needs the transposed state pool ([v] contiguous, stride(2)==1).
    got2 = call_op(trans, trans["state"].transpose(-1, -2))
    print("v2 transposed-view:")
    parity("v2-vs-eager", got2, ref_t)

    cases = [
        ("v1 native (PRODUCTION)", lambda: call_op(native, native["state"])),
        ("v1 transposed view",
         lambda: call_op(trans, trans["state"].transpose(-1, -2))),
        ("v2 transposed view",
         lambda: call_op(trans, trans["state"].transpose(-1, -2))),
    ]
    for bv in args.bv_list:
        for warps in args.warps_list:
            cases.append(
                (
                    f"triton native BV={bv} warps={warps}",
                    lambda bv=bv, w=warps: call_triton(native, native["state"], bv, w),
                )
            )
    print(f"{'impl':26s} {'event':>9} {'kernel':>9} {'enqueue':>9}")
    for name, fn in cases:
        try:
            ev = event_us(fn, args.iters)
            ku, _ = kernel_us(fn)
            host = enqueue_us(fn, args.iters)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:26s} FAILED {type(exc).__name__}: {exc}")
            continue
        kus = f"{ku:9.1f}" if ku else f"{'n/a':>9}"
        print(f"{name:26s} {ev:9.1f} {kus} {host:9.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
