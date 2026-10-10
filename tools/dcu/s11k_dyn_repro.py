#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11k: minimal reproducer for the S11e `ConstraintViolationError`.

The engine run needs ~150 s to reach the crash, all of it checkpoint loading.
This script reproduces the *same* failure signature in a few seconds:

    Constraints violated (L['x'].size()[0])!
      You marked L['x'].size()[0] as dynamic but your code specialized it to
      be a constant (64).
    User stack:
      File ".../s11k_dyn_repro.py", line N, in f
        out = torch.ops._rocm_C.w8a8_sdot4_gemm(xq, w, xs, ws, None, torch.float16, 0)

Variants (argv[1]) validate each candidate fix without booting the engine:

    raw         plain custom-op call (expected: ConstraintViolationError)
    torchop     same graph with a stock ATen op (negative control)
    triton      same shape semantics through triton_scaled_mm (S11e exp. C)
    disable     custom-op call inside a @torch.compiler.disable function
    fake        torch.library.register_fake() on the existing C++ op
    customop    re-export as torch.library.custom_op + register_fake
    direct      vllm.utils.torch_utils.direct_register_custom_op (no tags)
    directtag   ... with tags=(torch.Tag.pt2_compliant_tag,)
    branch      custom_op whose *body* holds the M<=8 GEMV / GEMM branch,
                called with M=2048 then M=1 (production shape pattern)
"""

import sys

import torch

import vllm._rocm_C  # noqa: F401  (registers torch.ops._rocm_C)

N, K = 512, 512
M_BIG, M_SMALL = 64, 8


def _raw_gemm(xq, w, xs, ws, cfg=0):
    return torch.ops._rocm_C.w8a8_sdot4_gemm(xq, w, xs, ws, None, torch.float16, cfg)


def _raw_gemv(xq, w, xs, ws):
    return torch.ops._rocm_C.w8a8_sdot4_gemv(xq, w, xs, ws, None, torch.float16, 64)


def _register(kind: str):
    """Returns a callable `op(xq, w, xs, ws) -> Tensor` or None for `raw`."""
    if kind == "fake":
        try:
            torch.library.register_fake("_rocm_C::w8a8_sdot4_gemm")(
                lambda xq, w, xs, ws, bias, out_dtype, cfg: xq.new_empty(
                    (xq.shape[0], w.shape[0]), dtype=out_dtype
                )
            )
            print("[repro] register_fake on the C++ op succeeded")
        except Exception as exc:
            print(f"[repro] register_fake on the C++ op FAILED: {type(exc).__name__}: {exc}")
            return None
        return _raw_gemm
    if kind in ("customop", "branch"):
        @torch.library.custom_op("s11k::sdot4_gemm", mutates_args=())
        def sdot4_gemm(
            xq: torch.Tensor,
            w: torch.Tensor,
            x_s: torch.Tensor,
            w_s: torch.Tensor,
            out_dtype: torch.dtype,
        ) -> torch.Tensor:
            if xq.shape[0] <= M_SMALL:
                return torch.ops._rocm_C.w8a8_sdot4_gemv(
                    xq, w, x_s, w_s, None, out_dtype, 64
                )
            return torch.ops._rocm_C.w8a8_sdot4_gemm(
                xq, w, x_s, w_s, None, out_dtype, 0
            )

        @sdot4_gemm.register_fake
        def _(xq, w, x_s, w_s, out_dtype):
            return xq.new_empty((xq.shape[0], w.shape[0]), dtype=out_dtype)

        print(f"[repro] registered s11k::sdot4_gemm via torch.library.custom_op ({kind})")
        return lambda xq, w, xs, ws: sdot4_gemm(xq, w, xs, ws, torch.float16)
    if kind in ("direct", "directtag"):
        from vllm.utils.torch_utils import direct_register_custom_op

        def body(xq, w, x_s, w_s, out_dtype):
            return torch.ops._rocm_C.w8a8_sdot4_gemm(
                xq, w, x_s, w_s, None, out_dtype, 0
            )

        def fake(xq, w, x_s, w_s, out_dtype):
            return xq.new_empty((xq.shape[0], w.shape[0]), dtype=out_dtype)

        tags = (torch.Tag.pt2_compliant_tag,) if kind == "directtag" else ()
        try:
            direct_register_custom_op(
                op_name="s11k_sdot4_gemm",
                op_func=body,
                mutates_args=[],
                fake_impl=fake,
                tags=tags,
            )
            print(f"[repro] direct_register_custom_op ok tags={tags}")
        except Exception as exc:
            print(f"[repro] direct_register_custom_op FAILED: {type(exc).__name__}: {exc}")
            return None
        op = torch.ops.vllm.s11k_sdot4_gemm
        return lambda xq, w, xs, ws: op(xq, w, xs, ws, torch.float16)
    return None


def build(kind: str):
    op = _register(kind)

    def f(x, w, xs, ws, positions):
        if kind == "torchop":
            out = torch.relu(x)
        elif kind == "triton":
            from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
                triton_scaled_mm,
            )

            xq = (x * 127.0).to(torch.int8)
            out = triton_scaled_mm(
                xq, w.t(), scale_a=xs, scale_b=ws, out_dtype=torch.float16
            )
        elif kind == "disable":
            out = _disabled_call(x, w, xs, ws)
        else:
            xq = (x * 127.0).to(torch.int8)
            out = op(xq, w, xs, ws)
        nt = positions.shape[1]
        return out, positions.reshape(-1).view(nt, -1)

    return f


@torch.compiler.disable
def _disabled_call(x, w, xs, ws):
    xq = (x * 127.0).to(torch.int8)
    return torch.ops._rocm_C.w8a8_sdot4_gemm(xq, w, xs, ws, None, torch.float16, 0)


def main() -> None:
    kind = sys.argv[1] if len(sys.argv) > 1 else "raw"
    f = build(kind)
    if f is None:
        print(f"[{kind}] setup unavailable, skipping")
        return
    mark = torch._dynamo.mark_dynamic

    def tensors(m):
        x = torch.randn(m, K, dtype=torch.float16, device="cuda")
        xs = torch.empty(m, dtype=torch.float32, device="cuda")
        return x, xs

    w = torch.empty(N, K, dtype=torch.int8, device="cuda")
    ws = torch.empty(N, dtype=torch.float32, device="cuda")
    x, xs = tensors(M_BIG)
    positions = torch.arange(3 * M_BIG, device="cuda").reshape(3, M_BIG)
    mark(x, 0)
    mark(xs, 0)
    mark(positions, 1)
    compiled = torch.compile(f, dynamic=True)
    try:
        o1, p1 = compiled(x, w, xs, ws, positions)
        print(f"[{kind}] big-M call ok: out={tuple(o1.shape)} pos={tuple(p1.shape)}")
    except Exception as exc:
        print(f"[{kind}] big-M call FAILED {type(exc).__name__}: {str(exc)[:400]}")
        return
    x2, xs2 = tensors(M_SMALL)
    pos2 = torch.arange(3 * M_SMALL, device="cuda").reshape(3, M_SMALL)
    try:
        o2, p2 = compiled(x2, w, xs2, ws, pos2)
        print(
            f"[{kind}] small-M call ok: out={tuple(o2.shape)} pos={tuple(p2.shape)} "
            f"(graph reuse across M=64 -> M=8)"
        )
    except Exception as exc:
        print(f"[{kind}] small-M call FAILED {type(exc).__name__}: {str(exc)[:400]}")
    from torch._dynamo.utils import counters

    print(f"[{kind}] counters={dict(counters)}")


if __name__ == "__main__":
    main()
