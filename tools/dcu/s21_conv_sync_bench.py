#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""S21: size the gfx906 causal-conv prefill-fallback host syncs it no longer does.

`_causal_conv1d_gfx906_fallback` used to read, for every sequence and every GDN
layer, four host scalars off device tensors:

    query_start_loc[i], query_start_loc[i+1], cache_indices[i], has_initial_state[i]

i.e. 4 DtoH syncs per sequence per layer (48 GDN layers on Qwen3.8-27B => 192
per step at batch 1).  Each sync also drains the launch queue, and the S20
attribution already calls the step launch/CPU-bound, so that CPU time is on the
critical path.

Two of the four now come from the caller's CPU metadata
(`query_start_loc_cpu`); the other two are resolved with device tensor ops
(`_gfx906_conv_state_rows`).  This probe sizes the change on realistic shapes:

    --arm host    query_start_loc_cpu supplied  => 0 device syncs   (shipped)
    --arm device  query_start_loc_cpu omitted   => 2 device syncs/seq/layer
                                                  (the residual/legacy shape)

The delta between the two arms is the cost of the two boundary syncs; the unit
cost is also measured directly so the pre-S21 total (4/seq/layer) can be
extrapolated.

Usage (inside the container):
    HIP_VISIBLE_DEVICES=0 python tools/dcu/s21_conv_sync_bench.py \
        [--dim 8192] [--tokens 2048] [--seqs 1] [--width 4] [--cache-len 10] \
        [--layers 48] [--reps 20]
"""

import argparse
import time

import torch  # noqa: F401  (must precede vllm imports)


def enqueue_ms(fn, reps):
    """Per-call CPU issue time, plus the residual drain (small = CPU-bound)."""
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return (t1 - t0) / reps * 1000.0, (t2 - t1) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--seqs", type=int, default=1)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--cache-len", type=int, default=10)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        _causal_conv1d_gfx906_fallback,
    )

    torch.manual_seed(5)
    dev = "cuda"
    dim, width, cl = a.dim, a.width, a.cache_len
    per = a.tokens // a.seqs
    lens = [per] * (a.seqs - 1) + [a.tokens - per * (a.seqs - 1)]
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + n)

    x = torch.randn(dim, a.tokens, device=dev, dtype=torch.float16)
    weight = torch.randn(dim, width, device=dev, dtype=torch.float16)
    bias = torch.randn(dim, device=dev, dtype=torch.float16)
    conv_state = torch.randn(a.seqs + 1, dim, cl, device=dev, dtype=torch.float16)
    qsl = torch.tensor(starts, device=dev, dtype=torch.int32)
    qsl_cpu = qsl.cpu()
    slots = torch.arange(1, a.seqs + 1, device=dev, dtype=torch.int32)
    has_init = torch.ones(a.seqs, device=dev, dtype=torch.bool)

    def call(host_meta: bool):
        kw = {"query_start_loc_cpu": qsl_cpu} if host_meta else {}
        return _causal_conv1d_gfx906_fallback(
            x, weight, bias, conv_state, qsl, slots, has_init, "silu", -1, **kw
        )

    # Unit cost of one `.item()` behind queued GPU work: the regime the old code
    # ran in (there was always layer work in flight behind it).  ~16 MB of work
    # per probe call is enough to make the drain dominate a bare DtoH.
    filler = torch.randn(1 << 22, device=dev)
    scalar = qsl[0]

    def one_drained_item():
        (filler * 1.0001).sum()
        return scalar.item()

    item_ms = enqueue_ms(one_drained_item, a.reps)[0]
    host_us, host_drain = enqueue_ms(lambda: call(True), a.reps)
    dev_us, dev_drain = enqueue_ms(lambda: call(False), a.reps)

    # The pre-S21 shape: the same call, plus the two further per-sequence scalar
    # reads it used to do on device (`cache_indices[i]`, `has_initial_state[i]`).
    # Reading a different unrelated scalar is faithful to the sync pattern even
    # though the moved code now resolves those two on device.
    def legacy_shape():
        out = call(False)
        for _ in range(2 * a.seqs):
            scalar.item()
        return out

    old_us, old_drain = enqueue_ms(legacy_shape, a.reps)

    per_sync_us = (dev_us - host_us) / (2 * a.seqs) * 1000.0

    print(
        f"device={torch.cuda.get_device_name(0)} dim={dim} tokens={a.tokens} "
        f"seqs={a.seqs} lens={lens} width={width} cache_len={cl} "
        f"layers={a.layers} reps={a.reps}"
    )
    print(f"one drained .item()                      : {item_ms * 1000:9.1f} us")
    print(
        f"[host]   CPU mirror, 0 device syncs      : {host_us * 1000:9.1f} us/call"
        f"  drain/call={host_drain / a.reps:7.1f} us"
    )
    print(
        f"[device] 2 syncs/seq/layer               : {dev_us * 1000:9.1f} us/call"
        f"  drain/call={dev_drain / a.reps:7.1f} us"
    )
    print(
        f"[4 sync] pre-S21 sync pattern            : {old_us * 1000:9.1f} us/call"
        f"  drain/call={old_drain / a.reps:7.1f} us"
    )
    print(
        f"-> measured cost of 2 syncs/seq/layer    : "
        f"{(dev_us - host_us) * 1000:9.1f} us/sync"
        f" ({per_sync_us:.1f} us each)"
    )
    print(
        f"-> host CPU time is "
        f"{'hidden' if host_drain / a.reps > host_us * 1000 else 'EXPOSED'}"
        f" behind {host_drain / a.reps:7.1f} us of GPU work per layer"
    )
    print(
        f"-> shipped vs pre-S21 per layer          : "
        f"{(old_us - host_us) * 1000:9.1f} us/call "
        f"({100.0 * (old_us - host_us) / old_us:5.1f}% of the pre-S21 layer)"
    )
    print(
        f"-> per {a.layers}-layer prefill pass, batch {a.seqs}      : "
        f"{(old_us - host_us) * a.layers:9.2f} ms removed"
        f"  ({old_us * a.layers:8.2f} -> {host_us * a.layers:8.2f} ms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
