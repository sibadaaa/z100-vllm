#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Perf1: CPU smoke test for the GDN MTP dispatch path (no GPU, no lock).

Guards the class of bug that took the production service down on 2026-10-13:
`fused_sigmoid_gating_delta_rule_update` read AND assigned the module global
`_GFX906_MTP_GDN_DEBUG_REPORTED` in its ineligible-debug branch without a
`global` declaration, so Python treated it as a local and *every* call raised
UnboundLocalError.  `py_compile` cannot see that; calling the function can.

Four modes, all on CPU:
  A) non-MTP decode call (num_accepted_tokens=None)      -> patched C++ decode
  B) MTP call with the Triton switch OFF                 -> patched C++ MTP op
  C) MTP call with VLLM_QWEN35_MTP_GDN_TRITON=1          -> real Triton kernel
                                                            under TRITON_INTERPRET
  D) MTP call whose shape is ineligible (initial_state=None) -> the branch that
     used to raise, with VLLM_QWEN35_MTP_GDN_DEBUG=1

  HIP_VISIBLE_DEVICES= TRITON_INTERPRET=1 python tools/dcu/perf1_gdn_smoke.py
"""

import os

# The kernel under test is the interpreter-mode Triton kernel; set this before
# anything imports triton.
os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ["VLLM_QWEN35_MTP_GDN_DEBUG"] = "1"
os.environ.pop("VLLM_QWEN35_MTP_GDN_TRITON", None)

import torch  # noqa: E402

import vllm.model_executor.layers.fla.ops.fused_sigmoid_gating as fsg  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update as dispatch,
)

H, HV, KD, VD, T, SLOTS = 2, 4, 32, 16, 3, 6
SCALE = KD ** -0.5
PRINTED = []


def fake_op(**kwargs):
    """Stand-in for the C++ op: right shape/dtype, deterministic values."""
    q = kwargs["q"]
    tokens, kv_heads, value_dim = q.shape[1], kwargs["v"].shape[2], kwargs["v"].shape[3]
    out = torch.arange(tokens * kv_heads * value_dim, dtype=q.dtype).reshape(
        tokens, kv_heads, value_dim
    )
    return out


def build(state=True):
    g = torch.Generator().manual_seed(5)
    a = torch.randn(T, HV, generator=g) * 0.2
    b = torch.randn(T, HV, generator=g) * 0.2
    idx = torch.arange(T, dtype=torch.int32).reshape(1, T)
    state_t = torch.randn(SLOTS, HV, VD, KD, generator=g) * 0.2
    return dict(
        A_log=torch.randn(HV, generator=g) * 0.2,
        a=a,
        b=b,
        dt_bias=torch.randn(HV, generator=g) * 0.1,
        q=torch.randn(1, T, H, KD, generator=g) * 2.5,
        k=torch.randn(1, T, H, KD, generator=g) * 2.5,
        v=torch.randn(1, T, HV, VD, generator=g) * 2.5,
        cu=torch.tensor([0, T], dtype=torch.int32),
        idx=idx,
        acc=torch.tensor([T], dtype=torch.int32),
        state=state_t if state else None,
    )


def check(name, fn, expect_out=True):
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        raise
    if expect_out:
        assert out[0] is not None and out[1] is not None, name
        # the MTP op returns [T, HV, VD] and the decode op [1, T, HV, VD];
        # this smoke test only cares that dispatch did not raise.
        assert out[0].dim() in (3, 4), (name, out[0].shape)
    print(f"ok   {name}: out={None if out[0] is None else tuple(out[0].shape)}")
    return out


def main():
    # The dispatcher only takes the gfx906 branch when the platform reports
    # gfx906; the smoke test runs with no visible GPU, so force it.
    fsg._is_gfx906_rocm = lambda: True
    ops.fused_sigmoid_gating_delta_rule_gfx906_mtp_update = fake_op
    ops.fused_sigmoid_gating_delta_rule_gfx906_decode = fake_op
    ops.fused_sigmoid_gating_delta_rule_gfx906_indexed_decode = fake_op
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")

    # A) non-MTP decode: the branch that used to raise UnboundLocalError
    d = build()
    check(
        "A non-MTP decode (CUDA unavailable -> eager fallback)",
        lambda: dispatch(
            A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"], q=d["q"],
            k=d["k"], v=d["v"], initial_state=d["state"], inplace_final_state=True,
            cu_seqlens=d["cu"], ssm_state_indices=d["idx"],
            use_qk_l2norm_in_kernel=True,
        ),
    )

    # B) MTP with the Triton switch OFF -> patched C++ op
    d = build()
    check(
        "B MTP, Triton OFF (C++ op path)",
        lambda: dispatch(
            A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"], q=d["q"],
            k=d["k"], v=d["v"], initial_state=d["state"], inplace_final_state=True,
            cu_seqlens=d["cu"], ssm_state_indices=d["idx"],
            num_accepted_tokens=d["acc"], scale=SCALE, use_qk_l2norm_in_kernel=True,
        ),
    )

    # C) MTP with the Triton switch ON -> the real kernel, interpreter mode
    os.environ["VLLM_QWEN35_MTP_GDN_TRITON"] = "1"
    d = build()
    out = check(
        "C MTP, Triton ON (kernel under TRITON_INTERPRET)",
        lambda: dispatch(
            A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"], q=d["q"],
            k=d["k"], v=d["v"], initial_state=d["state"], inplace_final_state=True,
            cu_seqlens=d["cu"], ssm_state_indices=d["idx"],
            num_accepted_tokens=d["acc"], scale=SCALE, use_qk_l2norm_in_kernel=True,
        ),
    )
    assert torch.equal(
        out[0].reshape(T, HV, VD),
        torch.arange(T * HV * VD, dtype=out[0].dtype).reshape(T, HV, VD),
    ) is False, "sanity: the Triton path must not return the stubbed values"

    # D) ineligible MTP shape (inplace_final_state=False) with DEBUG=1 -> this
    # is the branch that raised UnboundLocalError; re-arm the one-shot print
    # first so the INELIGIBLE line is really emitted.
    os.environ.pop("VLLM_QWEN35_MTP_GDN_TRITON", None)
    fsg._GFX906_MTP_GDN_DEBUG_REPORTED = False
    d = build()
    check(
        "D MTP shape ineligible (inplace_final_state=False) -> eager",
        lambda: dispatch(
            A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"], q=d["q"],
            k=d["k"], v=d["v"], initial_state=d["state"],
            inplace_final_state=False, cu_seqlens=d["cu"],
            ssm_state_indices=d["idx"], num_accepted_tokens=d["acc"],
            scale=SCALE, use_qk_l2norm_in_kernel=True,
        ),
    )

    # E) same ineligible shape with the debug channel OFF must not raise either
    os.environ["VLLM_QWEN35_MTP_GDN_DEBUG"] = "0"
    fsg._GFX906_MTP_GDN_DEBUG_REPORTED = False
    d = build()
    check(
        "E MTP shape ineligible, DEBUG=0",
        lambda: dispatch(
            A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"], q=d["q"],
            k=d["k"], v=d["v"], initial_state=d["state"],
            inplace_final_state=False, cu_seqlens=d["cu"],
            ssm_state_indices=d["idx"], num_accepted_tokens=d["acc"],
            scale=SCALE, use_qk_l2norm_in_kernel=True,
        ),
    )
    # F) S18 bisect switch: VLLM_QWEN35_MTP_GDN_CUSTOM=0 must bypass both gfx906
    # MTP kernels and take the eager path, whatever the Triton switch says.
    os.environ["VLLM_QWEN35_MTP_GDN_CUSTOM"] = "0"
    os.environ["VLLM_QWEN35_MTP_GDN_TRITON"] = "1"
    calls = []
    real_eager = fsg._fused_sigmoid_gating_delta_rule_update_gfx906_eager

    def spy(*a, **kw):
        calls.append(1)
        return real_eager(*a, **kw)

    fsg._fused_sigmoid_gating_delta_rule_update_gfx906_eager = spy
    d = build()
    check(
        "F CUSTOM=0 with TRITON=1 -> eager bisect path",
        lambda: dispatch(
            A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"], q=d["q"],
            k=d["k"], v=d["v"], initial_state=d["state"], inplace_final_state=True,
            cu_seqlens=d["cu"], ssm_state_indices=d["idx"],
            num_accepted_tokens=d["acc"], scale=SCALE, use_qk_l2norm_in_kernel=True,
        ),
    )
    fsg._fused_sigmoid_gating_delta_rule_update_gfx906_eager = real_eager
    assert calls, "CUSTOM=0 did not reach the eager path"
    os.environ.pop("VLLM_QWEN35_MTP_GDN_CUSTOM")
    os.environ.pop("VLLM_QWEN35_MTP_GDN_TRITON")

    print("all smoke modes passed (incl. CUSTOM=0 bisect switch)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
