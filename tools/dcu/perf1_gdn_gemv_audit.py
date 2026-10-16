#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Perf1: CPU-level audit of the gfx906 dense-GEMV routing in layers/utils.py.

No GPU, no lock.  The three terminals of ``rocm_unquantized_gemm_impl`` are
replaced by recording stubs, so the dispatch *decision* per (n_tokens, n_out,
K, dtype, env) is directly observable:

  * ``utils.triton_matmul``                     -> the pre-existing fallback
  * ``torch.ops._rocm_C.bf16_emu_gemv``         -> the emulated dense GEMV
  * ``torch.nn.functional.linear``              -> rocBLAS

It also records the ``nthreads`` argument that actually reaches the kernel,
which is how the new shape-dependent ``_gfx906_gemv_threads()`` is checked.

  HIP_VISIBLE_DEVICES= python tools/dcu/perf1_gdn_gemv_audit.py
"""

import os

os.environ["VLLM_ROCM_USE_SKINNY_GEMM"] = "0"
os.environ.pop("VLLM_DCU_USE_BF16_GEMV", None)
os.environ.pop("VLLM_DCU_USE_GFX906_EMU", None)

import torch  # noqa: E402

import vllm  # noqa: F401,E402
import vllm.model_executor.layers.utils as U  # noqa: E402

CALLS = []
FINDINGS = []


def install_stubs():
    U.on_gfx906 = lambda: True
    U.on_gfx9 = lambda: False
    U._GFX906_GEMV_AVAILABLE = True
    U.ops.ROCM_SKINNY_GEMM_ENABLED = False

    def spy_triton(a, b):
        CALLS.append(("triton", int(a.shape[0]), int(b.shape[0]), None))
        return torch.empty(a.shape[0], b.shape[0], dtype=torch.float16)

    def spy_linear(x, w, bias=None):
        CALLS.append(("linear", int(x.shape[0]), int(w.shape[0]), None))
        return torch.empty(x.shape[0], w.shape[0], dtype=x.dtype)

    def spy_gemv(x, w, bias, nthreads, route, range_safe):
        CALLS.append(("gemv", int(x.shape[0]), int(w.shape[0]), int(nthreads)))
        return torch.empty(x.shape[0], w.shape[0], dtype=x.dtype)

    U.triton_matmul = spy_triton
    U.torch.nn.functional.linear = spy_linear
    torch.ops._rocm_C.bf16_emu_gemv = spy_gemv


def route(m, n, k=5120, dtype=torch.float16, bias=None, env=None, family=None):
    CALLS.clear()
    if env is None:
        os.environ.pop("VLLM_DCU_USE_BF16_GEMV", None)
    else:
        os.environ["VLLM_DCU_USE_BF16_GEMV"] = env
    if family is None:
        os.environ.pop("VLLM_DCU_USE_GFX906_EMU", None)
        U.envs.VLLM_DCU_USE_GFX906_EMU = False
    else:
        os.environ["VLLM_DCU_USE_GFX906_EMU"] = family
        U.envs.VLLM_DCU_USE_GFX906_EMU = family == "1"
    x = torch.zeros(m, k, dtype=dtype)
    w = torch.zeros(n, k, dtype=dtype)
    U.rocm_unquantized_gemm_impl(x, w, bias)
    assert len(CALLS) == 1, f"expected one terminal call, got {CALLS}"
    return CALLS[0]


def check(label, got, want):
    ok = got[0] == want[0] and got[1] == want[1] and got[2] == want[2]
    if want[3] is not None:
        ok = ok and got[3] == want[3]
    tag = "ok  " if ok else "FAIL"
    print(f"{tag} {label:52s} got={got} want={want}")
    if not ok:
        FINDINGS.append((label, got, want))


def main():
    install_stubs()
    print(f"bf16_emu available={U._GFX906_GEMV_AVAILABLE} "
          f"threads(24)={U._gfx906_gemv_threads(24)} "
          f"threads(1280)={U._gfx906_gemv_threads(1280)} "
          f"threads(62080)={U._gfx906_gemv_threads(62080)}")

    print("\n-- env ON (VLLM_DCU_USE_BF16_GEMV=1): expected winner per shape")
    # in_proj_ba (N=24/48, K=5120): gemv for M<=4, threads 128
    for m in (1, 2, 3, 4):
        check(f"in_proj_ba N=24 fp16 M={m}", route(m, 24, env="1"),
              ("gemv", m, 24, 128))
    check("in_proj_ba N=48 fp16 M=4", route(4, 48, env="1"),
          ("gemv", 4, 48, 128))
    check("in_proj_ba N=24 fp16 M=5 -> triton", route(5, 24, env="1"),
          ("triton", 5, 24, None))
    check("in_proj_ba N=24 fp16 M=8 -> triton", route(8, 24, env="1"),
          ("triton", 8, 24, None))
    # mtp.fc (N=1280, K=10240): gemv for M<=4, threads 128
    for m in (1, 2, 3, 4):
        check(f"mtp.fc N=1280 fp16 M={m}", route(m, 1280, k=10240, env="1"),
              ("gemv", m, 1280, 128))
    check("mtp.fc N=1280 fp16 M=5 -> triton", route(5, 1280, k=10240, env="1"),
          ("triton", 5, 1280, None))
    # lm_head (N=62080, K=5120): gemv only for M<=2, threads 64
    for m, want in ((1, "gemv"), (2, "gemv"), (3, "triton"), (4, "triton")):
        threads = 64 if want == "gemv" else None
        check(f"lm_head N=62080 fp16 M={m}", route(m, 62080, env="1"),
              (want, m, 62080, threads))

    print("\n-- env OFF: the new gate must be inert (pre-existing behaviour)")
    for env in (None, "0"):
        for m, n, k in ((1, 24, 5120), (3, 24, 5120), (4, 1280, 10240),
                        (1, 62080, 5120), (2, 62080, 5120), (16, 62080, 5120),
                        (17, 62080, 5120)):
            want = "triton" if m <= 16 else "linear"
            check(f"env={env} M={m} N={n}", route(m, n, k=k, env=env),
                  (want, m, n, None))

    print("\n-- dtype / K / bias gates")
    check("fp32 M=3 N=24 -> triton (dtype gate)", route(3, 24, dtype=torch.float32,
          env="1"), ("triton", 3, 24, None))
    check("fp16 M=3 N=24 K=5121 odd -> triton", route(3, 24, k=5121, env="1"),
          ("triton", 3, 24, None))
    check("fp16 M=3 N=24 with bias -> gemv (bias is passed through)",
          route(3, 24, bias=torch.zeros(24), env="1"), ("gemv", 3, 24, 128))
    check("bf16 M=2 N=1280 -> gemv", route(2, 1280, k=10240, dtype=torch.bfloat16,
          env="1"), ("gemv", 2, 1280, 128))

    print("\n-- family switch VLLM_DCU_USE_GFX906_EMU=1 with BF16 unset")
    check("family=1, in_proj_ba M=3", route(3, 24, env=None, family="1"),
          ("gemv", 3, 24, 128))
    check("family=1, BF16 explicit 0 wins", route(3, 24, env="0", family="1"),
          ("triton", 3, 24, None))

    print("\n-- existing test contract: tests/kernels/test_gfx906_bf16_gemv.py")
    # that test compares rocm_unquantized_gemm_impl(x, w) against a *direct*
    # call with nthreads hardcoded to 64; record what the dispatch now uses.
    got = route(1, 384, k=512, dtype=torch.bfloat16, env="1")
    print(f"     dispatch for M=1 N=384 K=512 bf16 -> {got}; the test's direct "
          f"call uses nthreads=64")
    if got[3] != 64:
        FINDINGS.append(
            ("test_gfx906_bf16_gemv direct-call nthreads", got, 64)
        )
        print("     => bitwise torch.equal(routed, direct) is no longer a valid "
              "reachability guard for this shape; the test must use "
              "_gfx906_gemv_threads(w.shape[0])")

    print(f"\n{len(FINDINGS)} finding(s)")
    for f in FINDINGS:
        print(f"  - {f}")
    return 1 if FINDINGS else 0


if __name__ == "__main__":
    raise SystemExit(main())
