#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Perf1: CPU decomposition of the Python cost added by the GDN Triton path.

Motivation: the kernel is 36x faster standalone (19.5 us vs 705 us) yet the e2e
arm is 13.8 ms/step slower, so per call the live path must be paying far more
CPU than the 120 us the standalone enqueue loop measured.  This script splits
that CPU cost into the three layers between the model and the kernel:

  A  fused_sigmoid_gating_delta_rule_update with the C++ path (stub op)
  B  the same with the Triton path (dispatch only, wrapper stubbed out)
  C  gdn_mtp_gfx906_triton's own Python (launch object replaced by a no-op)

so (B - A) + C is the total Python added per call, excluding Triton's launcher,
and C is directly comparable to what the in-worker probe reports as host_us.

  HIP_VISIBLE_DEVICES= python tools/dcu/perf1_gdn_dispatch_cost.py
"""

import os
import time

os.environ.pop("VLLM_QWEN35_MTP_GDN_TRITON", None)
os.environ.pop("VLLM_QWEN35_MTP_GDN_CUSTOM", None)

import torch  # noqa: E402

import vllm.model_executor.layers.fla.ops.fused_sigmoid_gating as fsg  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402

H, HV, KD, VD, T, SLOTS = 4, 12, 128, 128, 3, 8
ITERS = 2000


def build():
    g = torch.Generator().manual_seed(1)
    f16 = torch.float16
    return dict(
        A_log=torch.randn(HV, generator=g) * 0.2,
        a=torch.randn(T, HV, generator=g) * 0.2,
        b=torch.randn(T, HV, generator=g) * 0.2,
        dt_bias=torch.randn(HV, generator=g) * 0.1,
        q=torch.randn(1, T, H, KD, generator=g) * 2.5,
        k=torch.randn(1, T, H, KD, generator=g) * 2.5,
        v=torch.randn(1, T, HV, VD, generator=g) * 2.5,
        cu=torch.tensor([0, T], dtype=torch.int32),
        idx=torch.tensor([[1, 2, 3]], dtype=torch.int32),
        acc=torch.tensor([T], dtype=torch.int32),
        state=torch.randn(SLOTS, HV, VD, KD, generator=g).to(f16) * 0.2,
    )


def bench(fn, iters=ITERS, warmup=50):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = (time.perf_counter() - t0) / iters * 1e6
    return dt


def main():
    d = build()
    fsg._is_gfx906_rocm = lambda: True

    def fake_op(**kwargs):
        q = kwargs["q"]
        return torch.empty(q.shape[1], kwargs["v"].shape[2], kwargs["v"].shape[3],
                           dtype=q.dtype)

    ops.fused_sigmoid_gating_delta_rule_gfx906_mtp_update = fake_op

    def call(**over):
        kw = dict(
            A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"], q=d["q"],
            k=d["k"], v=d["v"], initial_state=d["state"],
            inplace_final_state=True, cu_seqlens=d["cu"],
            ssm_state_indices=d["idx"], num_accepted_tokens=d["acc"],
            scale=KD ** -0.5, use_qk_l2norm_in_kernel=True,
        )
        kw.update(over)
        return fsg.fused_sigmoid_gating_delta_rule_update(**kw)

    def noop(*a, **k):
        return torch.empty(T, HV, VD, dtype=torch.float16)

    # A: C++ path (logical op stubbed after the dispatch branches)
    us_cpp = bench(lambda: call())

    # B: Triton path, wrapper stubbed out -> pure dispatch delta
    os.environ["VLLM_QWEN35_MTP_GDN_TRITON"] = "1"
    fsg._gdn_mtp_triton = noop
    us_dispatch = bench(lambda: call())

    # C: the real wrapper's Python, launch object replaced by a no-op
    fsg._gdn_mtp_triton = fsg._gdn_mtp_triton  # keep dispatch as-is
    import vllm.model_executor.layers.fla.ops.gfx906_gdn_mtp_triton as gdn

    class _FakeKernel:
        def __getitem__(self, grid):
            return lambda *a, **k: None

    real_kernel = gdn._gdn_mtp_update_kernel
    gdn._gdn_mtp_update_kernel = _FakeKernel()
    us_wrapper = bench(
        lambda: gdn.gdn_mtp_gfx906_triton(
            A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"], q=d["q"],
            k=d["k"], v=d["v"], state=d["state"], state_indices=d["idx"],
            cu_seqlens=d["cu"], num_accepted_tokens=d["acc"], beta=1.0,
            threshold=20.0, scale=KD ** -0.5, use_qk_l2norm_in_kernel=True,
        )
    )
    gdn._gdn_mtp_update_kernel = real_kernel

    print(f"T={T} heads={H} kv_heads={HV} KV/VD={KD}/{VD} iters={ITERS}")
    print(f"A dispatch+C++ stub                : {us_cpp:8.2f} us/call")
    print(f"B dispatch+Triton stub (no wrapper) : {us_dispatch:8.2f} us/call")
    print(f"C wrapper Python (launch no-op)     : {us_wrapper:8.2f} us/call")
    print(f"=> dispatch delta (B-A)            : {us_dispatch - us_cpp:8.2f} us/call")
    print(f"=> Python added per call (B-A)+C   : {us_dispatch - us_cpp + us_wrapper:8.2f} us/call")
    print(f"=> per step over 48 calls          : "
          f"{(us_dispatch - us_cpp + us_wrapper) * 48 / 1000:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
