# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guards for the gfx906 dense fp16/bf16 decode GEMV (csrc/rocm/bf16_emu_gemv.cu).

gfx906 has no bf16 arithmetic instruction, so the kernel offers two routes and
this file pins both contracts:

* ``route=1`` (fp32): every bf16 value is widened with a shift (exact) and
  accumulated in fp32 with FMA -- the result must match an fp32 reference to
  within the accumulation order.
* ``route=2`` (fp16 dot2): one `v_dot2_f32_f16` per pair of lanes.  For fp16
  inputs there is no conversion at all, so it is exact modulo the fp32
  accumulation order as well.  For bf16 inputs the operand is *rebased* into
  fp16, whose exponent window is ~30 binades narrower: values below 2^-14 are
  flushed, so this route is only valid for range-checked weights and must
  never produce inf/NaN.

Skipped everywhere except DTK + gfx906.
"""

import pytest
import torch

from vllm.platforms import current_platform


def _gfx906_stack() -> bool:
    """gfx906 on any ROCm stack (DTK or the self-built ROCm 10.1 stack).

    The fork's gfx906 kernels are compiled for both stacks; the DTK-only gate
    that predated the ROCm 10.1 adaptation silently skipped this coverage.
    """
    if not torch.cuda.is_available() or not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx906

    return on_gfx906()


pytestmark = pytest.mark.skipif(
    not _gfx906_stack(), reason="gfx906 only: the emulated GEMV targets gfx906"
)


@pytest.fixture(scope="module")
def gemv():
    import vllm  # noqa: F401

    if not hasattr(torch.ops._rocm_C, "bf16_emu_gemv"):
        pytest.skip("torch.ops._rocm_C.bf16_emu_gemv is not registered")

    def run(x, w, bias=None, route=0, range_safe=False, nthreads=64):
        return torch.ops._rocm_C.bf16_emu_gemv(
            x, w, bias, nthreads, route, range_safe
        )

    return run


def _ulp_rel(out, ref_fp32, tol_ulps=1.5):
    """Max |out - ref| in units of the output dtype's ulp at max|ref|.

    Comparing against the fp32 reference directly would flag the *output
    rounding* (half an ulp for a correctly rounded result) as an error, so the
    kernel is compared against the fp32 reference cast to the output dtype and
    a small ulp count is allowed for the fp32 accumulation order.
    """
    out_dtype = out.dtype
    ref = ref_fp32.to(out_dtype)
    ulp = torch.finfo(out_dtype).eps * ref_fp32.abs().max().clamp_min(1e-30).item()
    err = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    return err / ulp, out, ref


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_gemv_matches_fp32_reference(gemv, dtype, m):
    """Both routes must reproduce fp32 FMA accumulation of the widened inputs."""
    torch.manual_seed(0)
    k, n = 512, 384
    x = (torch.randn(m, k, device="cuda") * 0.5).to(dtype)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(dtype)
    ref = torch.nn.functional.linear(x.to(torch.float32), w.to(torch.float32))

    for route in (0, 1, 2):
        if route == 2 and dtype == torch.bfloat16:
            continue  # needs the range contract, covered separately below
        out = gemv(x, w, None, route=route)
        ulps, _, _ = _ulp_rel(out, ref)
        assert ulps < 8, f"dtype={dtype} m={m} route={route} err={ulps:.2f} ulp"


def test_bf16_fp32_route_is_exact_for_tiny_weights(gemv):
    """The fp32 route must stay exact for weights below fp16's normal range.

    This is the case the fp16 rebase route cannot represent (2^-14 flush), and
    the reason the fp32 route is the default for bf16.
    """
    torch.manual_seed(0)
    m, k, n = 3, 256, 128
    x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 1e-4).to(torch.bfloat16)  # sigma < 2^-13
    ref = torch.nn.functional.linear(x.to(torch.float32), w.to(torch.float32))

    out = gemv(x, w, None, route=1)
    ulps, _, _ = _ulp_rel(out, ref)
    assert ulps < 8, f"route=1 lost precision on tiny weights: {ulps:.2f} ulp"


def test_bf16_guarded_rebase_is_finite_but_flushes_tiny_weights(gemv):
    """The guarded rebase route must never overflow, but it *is* lossy.

    fp16's exponent window is ~30 binades narrower than bf16's, so a weight
    distribution whose sigma is below 2^-14 loses its small entries.  This test
    documents that contract: finite output, visibly different from the exact
    result.  (The Python layer therefore keeps bf16 on the fp32 route unless a
    per-row range check justifies the fast path.)
    """
    torch.manual_seed(0)
    m, k, n = 2, 256, 128
    x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 1e-4).to(torch.bfloat16)
    ref = torch.nn.functional.linear(x.to(torch.float32), w.to(torch.float32))

    out = gemv(x, w, None, route=2, range_safe=False)
    assert torch.isfinite(out).all()
    ulps, _, _ = _ulp_rel(out, ref)
    assert ulps > 8, "expected the flush to be visible for sigma < 2^-14"


def _range_safe_weights(n, k, lo_exp=-13, hi_exp=3):
    """bf16 weights whose magnitudes all sit inside fp16's normal window.

    Magnitudes are drawn log-uniformly from [2^lo_exp, 2^hi_exp], so every lane
    satisfies the fast rebase path's exponent precondition (e8 in [113, 143]).
    """
    exp = torch.randint(lo_exp, hi_exp + 1, (n, k), device="cuda").float()
    mag = torch.pow(2.0, exp)
    sign = torch.where(torch.rand(n, k, device="cuda") < 0.5, -1.0, 1.0)
    return (sign * mag).to(torch.bfloat16)


def test_bf16_fast_rebase_is_accurate_for_range_checked_weights(gemv):
    """With |w| inside [2^-14, 65504] the packed rebase is exact per lane."""
    torch.manual_seed(0)
    m, k, n = 4, 512, 256
    x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
    w = _range_safe_weights(n, k)
    ref = torch.nn.functional.linear(x.to(torch.float32), w.to(torch.float32))
    # Range precondition of the fast path: every |w| in [2^-14, 65504].
    assert w.abs().min().item() >= 2.0**-14 and w.abs().max().item() < 65504.0

    fast = gemv(x, w, None, route=2, range_safe=True)
    ulps, _, _ = _ulp_rel(fast, ref)
    assert ulps < 8, f"range-checked rebase lost precision: {ulps:.2f} ulp"
    # The guarded (scalar) conversion must agree on in-range data.
    guarded = gemv(x, w, None, route=2, range_safe=False)
    assert torch.equal(fast, guarded)


def test_bias_is_added_in_output_dtype(gemv):
    torch.manual_seed(0)
    m, k, n = 2, 256, 192
    x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    bias = (torch.randn(n, device="cuda") * 0.1).to(torch.bfloat16)
    ref = torch.nn.functional.linear(
        x.to(torch.float32), w.to(torch.float32), bias.to(torch.float32)
    )
    out = gemv(x, w, bias, route=0)
    ulps, _, _ = _ulp_rel(out, ref)
    assert ulps < 8, f"bias path error {ulps:.2f} ulp"


@pytest.mark.parametrize("m,k", [(9, 256), (1, 255)])
def test_unsupported_shapes_raise(gemv, m, k):
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    w = torch.randn(64, k, device="cuda").to(torch.bfloat16)
    with pytest.raises(RuntimeError):
        gemv(x, w)


def test_dense_gemm_dispatch_reaches_the_emu_kernel(monkeypatch):
    """The production dispatch must route a decode shape to the emulated GEMV.

    Regression guard for a reachability bug: ``rocm_unquantized_gemm_impl``
    tested the ``n <= 16`` Triton branch *first*, and because upstream
    ``on_gfx9()`` means the CDNA2/CDNA3 parts (gfx90a/gfx942/gfx950) rather
    than "any gfx9" it is False on gfx906 -- so every shape this kernel
    supports was swallowed by the Triton branch and
    ``torch.ops.vllm.bf16_emu_mm`` never ran, even with
    VLLM_DCU_USE_BF16_GEMV=1.  The comparison has to use the same thread count
    the dispatch picks (``_gfx906_gemv_threads``, shape dependent), otherwise
    the two reductions differ in order and the bitwise guard is meaningless.
    """
    import vllm  # noqa: F401
    from vllm.model_executor.layers.utils import (
        _gfx906_gemv_threads,
        rocm_unquantized_gemm_impl,
    )

    monkeypatch.setenv("VLLM_DCU_USE_BF16_GEMV", "1")
    torch.manual_seed(0)
    x = (torch.randn(1, 512, device="cuda") * 0.5).to(torch.bfloat16)
    w = (torch.randn(384, 512, device="cuda") * 0.02).to(torch.bfloat16)

    routed = rocm_unquantized_gemm_impl(x, w)
    direct = torch.ops._rocm_C.bf16_emu_gemv(
        x, w, None, _gfx906_gemv_threads(w.shape[0]), 0, False
    )
    assert torch.equal(routed, direct), "dispatch did not reach bf16_emu_gemv"


def test_dense_gemm_dispatch_is_off_by_default(monkeypatch):
    """An explicit 0 keeps the emulated GEMV out of the path."""
    import vllm  # noqa: F401
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    monkeypatch.setenv("VLLM_DCU_USE_BF16_GEMV", "0")
    torch.manual_seed(0)
    x = (torch.randn(1, 512, device="cuda") * 0.5).to(torch.bfloat16)
    w = (torch.randn(384, 512, device="cuda") * 0.02).to(torch.bfloat16)

    routed = rocm_unquantized_gemm_impl(x, w)
    direct = torch.ops._rocm_C.bf16_emu_gemv(x, w, None, 64, 0, False)
    assert not torch.equal(routed, direct), "switch off must not use the emu GEMV"
