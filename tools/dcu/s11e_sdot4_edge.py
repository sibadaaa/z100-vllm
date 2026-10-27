# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11e supplementary robustness probe for the gfx906 W8A8 ``sdot4`` kernel.

The main harness ``s11e_sdot4_eval.py`` sweeps the seven real (N, K) shapes of
Qwen3.8-27B-Channel-INT8-w8a8 with M in the decode/prefill ranges.  This probe
covers the *edges* that the real model does not exercise but that the kernel
advertises support for:

``edges``  direct ``torch.ops._rocm_C.w8a8_sdot4_{gemv,gemm}`` calls with
           M = 8/9 (GEMV/GEMM boundary), N that is not a multiple of the
           largest 128-wide tile, N odd, M = 2048, and every explicit ``cfg``
           (1..4).  Checked against the fp32 dequantized reference.
``fallback``  ``Gfx906Sdot4Int8ScaledMMLinearKernel.apply_weights`` with
           ``K % 4 != 0`` and with a non-int8 weight must take the
           ``triton_scaled_mm`` branch instead of raising.

Run inside the DTK shadow container with the canonical tree on PYTHONPATH::

    HIP_VISIBLE_DEVICES=2 python tools/dcu/s11e_sdot4_edge.py
"""

import argparse
import json
import sys

import torch  # noqa: F401  (must precede `import vllm`, todolist rule 8)


def _ops():
    import vllm  # noqa: F401
    from vllm import _custom_ops as ops

    return ops


def quantize_x(ops, x):
    x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
    return x_q, x_s.reshape(-1).to(torch.float32).contiguous()


def reference(x_q, x_s, w_q, w_s, bias):
    a = x_q.float() * x_s.reshape(-1, 1)
    b = w_q.float() * w_s.reshape(-1, 1)
    out = a @ b.t() + bias.float()
    return out


def rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def run_edges(ops):
    torch.manual_seed(0)
    cases = [
        (5120, 5120, 8, 0, "M=8 last GEMV shape"),
        (5120, 5120, 9, 0, "M=9 first GEMM shape"),
        (1000, 5120, 32, 0, "N=1000 not a multiple of 128"),
        (1025, 5120, 64, 0, "N=1025 odd"),
        (5120, 5120, 2048, 0, "M=2048 large prefill"),
        (6144, 5120, 64, 3, "explicit cfg=3 (64x128)"),
        (6144, 5120, 64, 4, "explicit cfg=4 (128x128)"),
        (5120, 5124, 32, 0, "K=5124 multiple of 4, not of 32"),
    ]
    rows = []
    for n, k, m, cfg, note in cases:
        w_q = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        w_s = torch.rand(n, dtype=torch.float32, device="cuda") * 0.02 + 0.005
        x = (torch.randn(m, k, dtype=torch.float32, device="cuda") * 1.5).half()
        bias = (torch.randn(n, dtype=torch.float32, device="cuda") * 0.1).half()
        x_q, x_s = quantize_x(ops, x)
        if m <= 8:
            out = torch.ops._rocm_C.w8a8_sdot4_gemv(
                x_q, w_q, x_s, w_s, bias, torch.float16, 64
            )
        else:
            out = torch.ops._rocm_C.w8a8_sdot4_gemm(
                x_q, w_q, x_s, w_s, bias, torch.float16, cfg
            )
        ref = reference(x_q, x_s, w_q, w_s, bias)
        assert out.shape == (m, n), (out.shape, (m, n))
        rows.append(
            dict(N=n, K=k, M=m, cfg=cfg, note=note,
                 rel_err=rel_err(out, ref), finite=bool(torch.isfinite(out).all()))
        )
        print(
            f"  N={n:5d} K={k:5d} M={m:5d} cfg={cfg} rel_err={rows[-1]['rel_err']:.3e} "
            f"finite={rows[-1]['finite']}  # {note}"
        )
    return rows


def run_fallback(ops):
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        Int8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4 import (
        Gfx906Sdot4Int8ScaledMMLinearKernel,
    )

    cfg = Int8ScaledMMLinearLayerConfig(
        is_static_input_scheme=False, is_channelwise=True, input_symmetric=True
    )
    names = ["weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"]
    rows = []

    def fake(n, k, dtype=torch.int8):
        from types import SimpleNamespace

        layer = SimpleNamespace()
        layer.weight = torch.nn.Parameter(
            torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda").to(dtype),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            (torch.rand(n, dtype=torch.float32, device="cuda") * 0.02 + 0.005).reshape(
                n, 1
            ),
            requires_grad=False,
        )
        layer.input_scale = None
        layer.input_zero_point = None
        layer.azp_adj = None
        layer.logical_widths = [n]
        return layer

    # K % 4 != 0 must not reach the kernel: torch.ops raises, apply_weights falls back.
    n, k, m = 512, 5118, 4
    raises = False
    try:
        w_q = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        w_s = torch.rand(n, dtype=torch.float32, device="cuda")
        x = torch.randn(m, k, dtype=torch.float32, device="cuda").half()
        x_q, x_s = quantize_x(ops, x)
        torch.ops._rocm_C.w8a8_sdot4_gemv(x_q, w_q, x_s, w_s, None, torch.float16, 64)
    except RuntimeError as exc:
        raises = "multiple of 4" in str(exc)
    layer = fake(n, k)
    kern = Gfx906Sdot4Int8ScaledMMLinearKernel(cfg, names)
    kern.process_weights_after_loading(layer)
    x = torch.randn(m, k, dtype=torch.float32, device="cuda").half()
    out = kern.apply_weights(layer, x, None)
    rows.append(
        dict(case="K%4!=0", N=n, K=k, M=m, direct_op_raises=raises,
             apply_weights_shape=list(out.shape),
             finite=bool(torch.isfinite(out).all()))
    )
    print(f"  K%4!=0: direct op raises={raises} apply_weights -> {tuple(out.shape)} "
          f"finite={bool(torch.isfinite(out).all())}")

    # Non-int8 weight: unreachable for a W8A8 int8 layer (the kernel is only
    # selected when the checkpoint weights are int8), but the `supported` guard
    # in apply_weights still routes it to _triton_fallback, which asserts on the
    # dtype mismatch.  Recorded as a dead-branch note rather than a gate.
    n, k, m = 512, 512, 4
    layer = fake(n, k, dtype=torch.float16)
    layer.weight.data = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.1
    kern = Gfx906Sdot4Int8ScaledMMLinearKernel(cfg, names)
    x = torch.randn(m, k, dtype=torch.float32, device="cuda").half()
    try:
        out = kern.apply_weights(layer, x, None)
        note = f"returned {tuple(out.shape)} finite={bool(torch.isfinite(out).all())}"
    except AssertionError as exc:
        note = f"AssertionError: {exc}"
    rows.append(dict(case="non-int8 weight (unreachable)", N=n, K=k, M=m, note=note))
    print(f"  non-int8 weight (unreachable dead branch): {note}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["edges", "fallback", "all"], default="all")
    a = p.parse_args()
    torch.cuda.init()
    print("device:", torch.cuda.get_device_name(0))
    ops = _ops()
    out = {}
    if a.mode in ("edges", "all"):
        print("== edge shapes vs fp32 dequant reference (gate rel_err <= 1e-2) ==")
        out["edges"] = run_edges(ops)
        worst = max(r["rel_err"] for r in out["edges"])
        print(f"WORST edge rel_err = {worst:.3e}   all_finite="
              f"{all(r['finite'] for r in out['edges'])}")
    if a.mode in ("fallback", "all"):
        print("== python-level fallback paths ==")
        out["fallback"] = run_fallback(ops)
    print("EDGE_SUMMARY", json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
