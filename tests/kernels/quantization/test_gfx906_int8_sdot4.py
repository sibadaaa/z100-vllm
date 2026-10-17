# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guards for the gfx906 W8A8 INT8 `sdot4` kernel (csrc/rocm/w8a8_sdot4.cu).

The kernel is the gfx906 answer to the W8A8 INT8 scaled MM that Triton runs at
~2.2 TFLOPS on this target; it accumulates exactly in int32 with the native
`v_dot4_i32_i8` instruction and applies the scales in the epilogue.  These
tests pin, in the same style as the FP8 emulation tests:

1. the int32 accumulator is **bit-exact** (`x_s = w_s = 1`, fp32 output, vs an
   int matmul oracle) for both the GEMV (M <= 8) and GEMM (M > 8) paths;
2. the dequantized result matches an fp32 reference for every scale form the
   kernel accepts -- per-channel ``[N]`` / per-tensor ``[1]`` weight scales and
   dynamic per-token ``[M]`` / static per-tensor ``[1]`` input scales;
3. the selection gates compose the family master switch
   (``VLLM_DCU_USE_GFX906_EMU``) with the per-kernel override.

Skipped everywhere except DTK + gfx906, where the kernel is selectable.
"""

import importlib

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
    not _gfx906_stack(),
    reason="gfx906 only: the sdot4 kernel targets gfx906",
)


@pytest.fixture(scope="module")
def sdot4():
    import vllm  # noqa: F401  (loads the extensions)

    if not hasattr(torch.ops._rocm_C, "w8a8_sdot4_gemm"):
        pytest.skip("torch.ops._rocm_C.w8a8_sdot4_* is not registered")

    def run(x_q, w, x_s, w_s, bias=None, out_dtype=torch.float16, nthreads=64):
        if x_q.shape[0] <= 8:
            return torch.ops._rocm_C.w8a8_sdot4_gemv(
                x_q, w, x_s, w_s, bias, out_dtype, nthreads
            )
        return torch.ops._rocm_C.w8a8_sdot4_gemm(
            x_q, w, x_s, w_s, bias, out_dtype, 0
        )

    return run


def _int_reference(x_q, w):
    # Integer matmul is done on the CPU: torch has no int8 GEMM on ROCm.
    return torch.matmul(x_q.cpu().int(), w.cpu().int().t()).float().cuda()


def _dequant_reference(x_q, x_s, w, w_s):
    a = x_q.to(torch.float32) * x_s.reshape(-1, 1)
    weight = w.to(torch.float32) * w_s.reshape(-1, 1)
    return a @ weight.t()


@pytest.mark.parametrize("m", [1, 4, 8, 32, 64])
def test_int32_accumulator_is_bit_exact(sdot4, m):
    """With unit scales the kernel must reproduce the integer product exactly.

    This is the strongest statement the W8A8 path can make: any accumulator
    rounding or premature scale application would show up immediately.
    """
    torch.manual_seed(0)
    n, k = 256, 512
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
    x_q = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
    ones_m = torch.ones(m, dtype=torch.float32, device="cuda")
    ones_n = torch.ones(n, dtype=torch.float32, device="cuda")

    out = sdot4(x_q, w, ones_m, ones_n, out_dtype=torch.float32)
    exact = _int_reference(x_q, w)
    assert torch.equal(out, exact), (
        f"M={m}: max|diff|={(out - exact).abs().max().item()}"
    )


@pytest.mark.parametrize("m", [1, 8, 16, 129])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_matches_dequantized_reference(sdot4, m, out_dtype):
    """Per-channel weight scales + dynamic per-token input scales."""
    torch.manual_seed(0)
    n, k = 192, 256
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
    w_s = (torch.rand(n, device="cuda") * 0.01 + 0.001).to(torch.float32)
    x_q = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
    x_s = (torch.rand(m, device="cuda") * 0.02 + 0.002).to(torch.float32)

    ref = _dequant_reference(x_q, x_s, w, w_s)
    out = sdot4(x_q, w, x_s, w_s, out_dtype=out_dtype)
    # Tolerance = half an ulp of the *output* dtype: the int32 accumulator is
    # exact, so the only error is the final cast (plus the fp32 accumulation
    # order of the reference).
    tol = {
        torch.float32: 1e-6,
        torch.float16: 1e-3,
        torch.bfloat16: 5e-3,
    }[out_dtype]
    err = (out.to(torch.float32) - ref).abs().max().item() / ref.abs().max().item()
    assert err < tol, f"M={m} {out_dtype}: max|err|/max|ref| = {err}"


def test_per_tensor_weight_scale(sdot4):
    """A 1-element weight scale must broadcast with stride 0."""
    torch.manual_seed(0)
    m, n, k = 4, 128, 128
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
    w_s = torch.tensor([0.0037], dtype=torch.float32, device="cuda")
    x_q = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
    x_s = (torch.rand(m, device="cuda") * 0.02).to(torch.float32)

    ref = _dequant_reference(x_q, x_s, w, w_s)
    out = sdot4(x_q, w, x_s, w_s, out_dtype=torch.float32)
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-6, f"per-tensor w_s: {err}"


def test_static_per_tensor_input_scale(sdot4):
    """A 1-element input scale (static quantization) must broadcast too."""
    torch.manual_seed(0)
    m, n, k = 6, 128, 192
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
    w_s = (torch.rand(n, device="cuda") * 0.01).to(torch.float32)
    x_q = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
    x_s = torch.tensor([0.011], dtype=torch.float32, device="cuda")

    ref = _dequant_reference(x_q, x_s, w, w_s)
    out = sdot4(x_q, w, x_s, w_s, out_dtype=torch.float32)
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-6, f"static x_s: {err}"


def test_both_scales_per_tensor(sdot4):
    """Both scales as scalars: the fully per-tensor W8A8 form."""
    torch.manual_seed(0)
    m, n, k = 3, 64, 128
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
    w_s = torch.tensor([0.0021], dtype=torch.float32, device="cuda")
    x_q = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
    x_s = torch.tensor([0.0093], dtype=torch.float32, device="cuda")

    ref = _dequant_reference(x_q, x_s, w, w_s)
    out = sdot4(x_q, w, x_s, w_s, out_dtype=torch.float32)
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-6


def test_bias_is_added_after_the_cast(sdot4):
    torch.manual_seed(0)
    m, n, k = 2, 96, 64
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
    w_s = (torch.rand(n, device="cuda") * 0.01).to(torch.float32)
    x_q = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
    x_s = (torch.rand(m, device="cuda") * 0.02).to(torch.float32)
    bias = (torch.randn(n, device="cuda") * 0.1).to(torch.float16)

    ref = _dequant_reference(x_q, x_s, w, w_s) + bias.to(torch.float32)
    out = sdot4(x_q, w, x_s, w_s, bias=bias, out_dtype=torch.float16)
    err = (
        (out.to(torch.float32) - ref).abs().max().item()
        / ref.abs().max().item()
    )
    assert err < 1e-3


@pytest.mark.parametrize("k", [12, 68])
def test_non_tile_multiple_k(sdot4, k):
    """K must only need to be a multiple of 4 (int32 packing)."""
    torch.manual_seed(0)
    m, n = 10, 40
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
    w_s = (torch.rand(n, device="cuda") * 0.01).to(torch.float32)
    x_q = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
    x_s = (torch.rand(m, device="cuda") * 0.02).to(torch.float32)

    ref = _dequant_reference(x_q, x_s, w, w_s)
    out = sdot4(x_q, w, x_s, w_s, out_dtype=torch.float32)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_k_not_multiple_of_4_raises():
    x_q = torch.zeros(2, 6, dtype=torch.int8, device="cuda")
    w = torch.zeros(4, 6, dtype=torch.int8, device="cuda")
    s = torch.ones(1, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="multiple of 4"):
        torch.ops._rocm_C.w8a8_sdot4_gemv(x_q, w, s, s, None, torch.float16, 64)


def test_scale_length_is_validated():
    x_q = torch.zeros(2, 8, dtype=torch.int8, device="cuda")
    w = torch.zeros(4, 8, dtype=torch.int8, device="cuda")
    x_s = torch.ones(7, dtype=torch.float32, device="cuda")  # neither M nor 1
    w_s = torch.ones(1, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="x_s must have M or 1"):
        torch.ops._rocm_C.w8a8_sdot4_gemv(x_q, w, x_s, w_s, None, torch.float16, 64)


def test_custom_op_is_registered_with_fake_impl():
    """`torch.compile` needs the opaque op plus a shape contract (S11k)."""
    import vllm  # noqa: F401
    import vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4  # noqa: F401

    if not hasattr(torch.ops.vllm, "w8a8_sdot4_mm"):
        pytest.skip("vllm::w8a8_sdot4_mm is not registered")
    from torch._subclasses.fake_tensor import FakeTensorMode

    x_q = torch.empty(5, 32, dtype=torch.int8, device="cuda")
    w = torch.empty(64, 32, dtype=torch.int8, device="cuda")
    x_s = torch.empty(1, dtype=torch.float32, device="cuda")
    w_s = torch.empty(1, dtype=torch.float32, device="cuda")
    schema = torch.ops.vllm.w8a8_sdot4_mm.default
    assert schema is not None
    with FakeTensorMode(allow_non_fake_inputs=True):
        out = torch.ops.vllm.w8a8_sdot4_mm(x_q, w, x_s, w_s, None, torch.float16)
    assert tuple(out.shape) == (5, 64) and out.dtype == torch.float16


def test_gate_composes_master_switch(monkeypatch):
    """The family master switch must enable it; an explicit 0 must win."""
    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_common import (
        gfx906_emu_enabled,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4 import (
        Gfx906Sdot4Int8ScaledMMLinearKernel,
    )

    def reload_envs():
        import vllm.envs as envs

        importlib.reload(envs)
        return envs

    # Nothing set: off.
    monkeypatch.delenv("VLLM_DCU_USE_INT8_SDOT4", raising=False)
    monkeypatch.delenv("VLLM_DCU_USE_GFX906_EMU", raising=False)
    reload_envs()
    assert not gfx906_emu_enabled("VLLM_DCU_USE_INT8_SDOT4")
    ok, reason = Gfx906Sdot4Int8ScaledMMLinearKernel.is_supported()
    assert not ok and "VLLM_DCU_USE_INT8_SDOT4" in (reason or "")

    # Master switch on: enabled.
    monkeypatch.setenv("VLLM_DCU_USE_GFX906_EMU", "1")
    reload_envs()
    assert gfx906_emu_enabled("VLLM_DCU_USE_INT8_SDOT4")
    ok, reason = Gfx906Sdot4Int8ScaledMMLinearKernel.is_supported()
    assert ok, reason

    # Explicit per-kernel 0 overrides the master.
    monkeypatch.setenv("VLLM_DCU_USE_INT8_SDOT4", "0")
    reload_envs()
    assert not gfx906_emu_enabled("VLLM_DCU_USE_INT8_SDOT4")

    monkeypatch.delenv("VLLM_DCU_USE_GFX906_EMU", raising=False)
    monkeypatch.delenv("VLLM_DCU_USE_INT8_SDOT4", raising=False)
    reload_envs()


def test_can_implement_accepts_static_and_per_tensor_configs():
    """The widened support matrix: symmetric input, any of the scale forms."""
    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4 import (
        Gfx906Sdot4Int8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        Int8ScaledMMLinearLayerConfig,
    )

    def cfg(static, channelwise, symmetric):
        return Int8ScaledMMLinearLayerConfig(
            is_static_input_scheme=static,
            is_channelwise=channelwise,
            input_symmetric=symmetric,
        )

    for static in (False, True):
        for channelwise in (False, True):
            ok, reason = Gfx906Sdot4Int8ScaledMMLinearKernel.can_implement(
                cfg(static, channelwise, True)
            )
            assert ok, f"static={static} channelwise={channelwise}: {reason}"

    ok, reason = Gfx906Sdot4Int8ScaledMMLinearKernel.can_implement(
        cfg(False, True, False)
    )
    assert not ok and "symmetric" in (reason or "")
