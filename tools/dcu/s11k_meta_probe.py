#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11k step 1: does the C++ `Meta` registration for the sdot4 ops actually
dispatch, and does fake-tensor propagation keep shapes symbolic?

If the Meta kernel is missing/ineffective, Dynamo has no shape semantics for
``torch.ops._rocm_C.w8a8_sdot4_*`` and the compilation pipeline has to
concretize the symbolic token dimension -- which is exactly the
``ConstraintViolationError`` seen in S11e.
"""

import torch

import vllm._rocm_C  # noqa: F401  (registers torch.ops._rocm_C)

OPS = ("w8a8_sdot4_gemm", "w8a8_sdot4_gemv")
REF_OPS = ("_C::gptq_shuffle", "_rocm_C::wvSplitK")


def has_keys(qualname: str) -> None:
    keys = ("Meta", "CompositeExplicitAutograd", "CompositeImplicitAutograd", "CPU", "CUDA")
    out = []
    for key in keys:
        try:
            out.append(f"{key}={torch._C._dispatch_has_kernel_for_dispatch_key(qualname, key)}")
        except Exception as exc:  # pragma: no cover - diagnostics only
            out.append(f"{key}=ERR({exc})")
    print(f"[dispatch] {qualname:45s} " + " ".join(out))


def main() -> None:
    for op in OPS:
        has_keys(f"_rocm_C::{op}")
    for ref in REF_OPS:
        has_keys(ref)
    print("[dispatch-dump] _rocm_C::w8a8_sdot4_gemm")
    try:
        print(torch._C._dispatch_dump_table("_rocm_C::w8a8_sdot4_gemm"))
    except Exception as exc:
        print("  dump err:", exc)

    from torch._subclasses.fake_tensor import FakeTensorMode

    m, n, k = 2048, 512, 512
    mode = FakeTensorMode()
    with mode:
        xq = mode.from_tensor(torch.empty(m, k, dtype=torch.int8, device="cuda"))
        w = mode.from_tensor(torch.empty(n, k, dtype=torch.int8, device="cuda"))
        xs = mode.from_tensor(torch.empty(m, dtype=torch.float32, device="cuda"))
        ws = mode.from_tensor(torch.empty(n, dtype=torch.float32, device="cuda"))
        for op in OPS:
            args = (xq, w, xs, ws, None, torch.float16)
            try:
                out = (
                    torch.ops._rocm_C.w8a8_sdot4_gemm(*args, 0)
                    if op.endswith("gemm")
                    else torch.ops._rocm_C.w8a8_sdot4_gemv(*args, 64)
                )
                print(
                    f"[fake] {op}: shape={tuple(out.shape)} dtype={out.dtype} "
                    f"type={type(out).__name__} device={out.device} "
                    f"is_fake={isinstance(out, torch.Tensor) and out.is_meta}"
                )
            except Exception as exc:
                print(f"[fake] {op}: FAILED {type(exc).__name__}: {str(exc)[:500]}")

    # Symbolic (non-concrete) sizes: use xfail-style dynamic mode via
    # torch.fx.experimental.symbolic_shapes with a real ShapeEnv.
    print("[fake-symbolic] using torch._dynamo to build a dynamic graph")
    from torch._dynamo import mark_dynamic

    def f(x, w, xs, ws, positions):
        xq = x.to(torch.int8)
        out = torch.ops._rocm_C.w8a8_sdot4_gemm(xq, w, xs, ws, None, torch.float16, 0)
        nt = positions.shape[1]
        return out, positions.reshape(-1).view(nt, -1)

    x = torch.randn(64, k, dtype=torch.float16, device="cuda")
    w = torch.empty(n, k, dtype=torch.int8, device="cuda")
    xs = torch.empty(64, dtype=torch.float32, device="cuda")
    ws = torch.empty(n, dtype=torch.float32, device="cuda")
    positions = torch.arange(3 * 64, device="cuda").reshape(3, 64)
    for dim, t in ((0, x), (0, xs), (1, positions)):
        mark_dynamic(t, dim)
    compiled = torch.compile(f, dynamic=True)
    try:
        o1, p1 = compiled(x, w, xs, ws, positions)
        print("[dyn] first call ok", tuple(o1.shape), tuple(p1.shape))
    except Exception as exc:
        print(f"[dyn] first call FAILED {type(exc).__name__}: {str(exc)[:500]}")
        return
    x2 = torch.randn(8, k, dtype=torch.float16, device="cuda")
    xs2 = torch.empty(8, dtype=torch.float32, device="cuda")
    pos2 = torch.arange(3 * 8, device="cuda").reshape(3, 8)
    try:
        o2, p2 = compiled(x2, w, xs2, ws, pos2)
        print("[dyn] second call (different M) ok", tuple(o2.shape), tuple(p2.shape))
    except Exception as exc:
        print(f"[dyn] second call FAILED {type(exc).__name__}: {str(exc)[:500]}")
    from torch._dynamo.utils import counters

    print("[dyn] dynamo counters:", dict(counters))


if __name__ == "__main__":
    main()
