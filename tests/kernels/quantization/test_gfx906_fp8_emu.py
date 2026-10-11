# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric guards for the gfx906 FP8 emulation kernel (csrc/rocm/fp8_emu_gemm.cu).

gfx906 has no fp8 hardware, so the kernel widens fp8 to fp16 through a
256-entry LDS table and accumulates with `v_dot2_f32_f16` in fp32.  These tests
pin the two properties that make that emulation valid:

1. the host-side LUT is lossless for every finite e4m3fn / e5m2 code (so the
   kernel adds no conversion error of its own), and
2. the kernel matches an fp32 reference computed from the *same* fp8 operands
   -- i.e. the only remaining error is the fp32 accumulation order (fp32
   output) or the final cast (fp16 output).

They are skipped everywhere except DTK + gfx906, where the kernel is selected.
"""

import pytest
import torch

from vllm.platforms import current_platform


def _dtk_gfx906() -> bool:
    if not torch.cuda.is_available() or not current_platform.is_rocm():
        return False
    from vllm.platforms.hygon_dtk import is_hygon_dtk
    from vllm.platforms.rocm import on_gfx906

    return is_hygon_dtk() and on_gfx906()


pytestmark = pytest.mark.skipif(
    not _dtk_gfx906(),
    reason="DTK + gfx906 only: the FP8 emulation kernel targets gfx906",
)


@pytest.fixture(scope="module")
def fp8_ops():
    import vllm  # noqa: F401  (loads the extensions)
    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_fp8 import (
        _device_lut,
        fp8_to_fp16_lut,
        ops_available,
    )

    if not ops_available():
        pytest.skip("torch.ops._rocm_C.fp8_emu_* is not registered")

    def lut(dtype):
        fp8_to_fp16_lut.cache_clear()
        return _device_lut(dtype, torch.device("cuda"))

    return lut


def _quantize(x, fp8_dtype, per_token):
    finfo = torch.finfo(fp8_dtype)
    dim = -1 if per_token else None
    amax = x.abs().amax(dim=dim, keepdim=per_token).clamp_min(1e-12)
    scale = (amax / finfo.max).to(torch.float32)
    q = (x / scale).clamp(finfo.min, finfo.max).to(fp8_dtype)
    return q, scale.reshape(-1)


def _reference(a_q, a_s, w_q, w_s):
    a = a_q.to(torch.float32) * a_s.reshape(-1, 1)
    w = w_q.to(torch.float32) * w_s.reshape(-1, 1)
    return a @ w.t()


@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
def test_lut_is_lossless_for_every_code(fp8_ops, fp8_dtype):
    """Every finite code must round-trip exactly through the fp16 LUT."""
    lut = fp8_ops(fp8_dtype).cpu()
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(fp8_dtype)
    decoded = codes.to(torch.float32)  # torch's own decode, on CPU
    got = lut.to(torch.float32)
    finite = torch.isfinite(decoded)
    assert torch.equal(got[finite], decoded[finite]), "LUT is not lossless"
    # e5m2 has infinities, e4m3 does not; NaN/inf codes must map alike.
    assert torch.equal(torch.isnan(got), torch.isnan(decoded))
    assert torch.equal(torch.isinf(got), torch.isinf(decoded))
    assert torch.equal(
        torch.copysign(torch.ones_like(got), got)[torch.isinf(got)],
        torch.copysign(torch.ones_like(got), decoded)[torch.isinf(decoded)],
    )


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("per_token", [True, False], ids=["per_token", "per_tensor"])
def test_gemv_matches_fp32_reference(fp8_ops, out_dtype, per_token):
    """M <= 8 decode path: only the accumulation order may differ."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    m, n, k = 4, 512, 1024
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    a_q, a_s = _quantize(a, fp8_dtype, per_token)
    w_q, w_s = _quantize(w, fp8_dtype, per_token=False)
    ref = _reference(a_q, a_s, w_q, w_s)

    out = torch.ops._rocm_C.fp8_emu_gemv(
        a_q, w_q, a_s, w_s, fp8_ops(fp8_dtype), None, out_dtype, 64
    )
    tol = 1e-5 if out_dtype == torch.float32 else 1e-3
    err = (out.to(torch.float32) - ref).abs().max().item() / ref.abs().max().item()
    assert err < tol, f"max|err|/max|ref| = {err}"


@pytest.mark.parametrize("m", [16, 64, 129])
def test_gemm_matches_fp32_reference(fp8_ops, m):
    """M > 8 prefill path (BM/BN tiled, K-chunked double buffering)."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    n, k = 256, 512
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    a_q, a_s = _quantize(a, fp8_dtype, per_token=True)
    w_q, w_s = _quantize(w, fp8_dtype, per_token=False)
    ref = _reference(a_q, a_s, w_q, w_s)

    lut = fp8_ops(fp8_dtype)
    for cfg in (0, 1, 2):
        out = torch.ops._rocm_C.fp8_emu_gemm(
            a_q, w_q, a_s, w_s, lut, None, torch.float32, cfg
        )
        err = (
            (out - ref).abs().max().item() / ref.abs().max().item()
        )
        assert err < 1e-5, f"cfg={cfg} max|err|/max|ref| = {err}"


def test_gemm_handles_odd_shapes(fp8_ops):
    """Non-multiple-of-tile M/N must not read or write out of bounds."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    m, n, k = 10, 100, 68  # K % 4 == 0 but K % 32 != 0, N < BN, M < BM
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    a_q, a_s = _quantize(a, fp8_dtype, per_token=True)
    w_q, w_s = _quantize(w, fp8_dtype, per_token=False)
    ref = _reference(a_q, a_s, w_q, w_s)
    out = torch.ops._rocm_C.fp8_emu_gemm(
        a_q, w_q, a_s, w_s, fp8_ops(fp8_dtype), None, torch.float32, 0
    )
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-5


def test_scalar_scale_and_bias(fp8_ops):
    """Per-tensor scales and a bias must follow the (acc*sa)*sb + bias order."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    m, n, k = 3, 128, 256
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    bias = torch.randn(n, device="cuda", dtype=torch.float16) * 0.1
    a_q, a_s = _quantize(a, fp8_dtype, per_token=False)
    w_q, w_s = _quantize(w, fp8_dtype, per_token=False)
    ref = _reference(a_q, a_s, w_q, w_s) + bias.to(torch.float32)

    out = torch.ops._rocm_C.fp8_emu_gemv(
        a_q, w_q, a_s, w_s, fp8_ops(fp8_dtype), bias, torch.float16, 64
    )
    err = (out.to(torch.float32) - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-3


def test_kernel_gating_follows_env(monkeypatch):
    """The kernel must refuse the target unless explicitly enabled."""
    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_fp8 import (
        Gfx906Fp8ScaledMMLinearKernel,
    )

    monkeypatch.setenv("VLLM_DCU_USE_FP8_EMU", "0")
    import vllm.envs as envs

    envs.__dict__.pop("VLLM_DCU_USE_FP8_EMU", None)
    ok, reason = Gfx906Fp8ScaledMMLinearKernel.is_supported()
    assert not ok and "VLLM_DCU_USE_FP8_EMU" in (reason or "")


def _fp8_config(weight_group_shape, activation_group_shape):
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        FP8ScaledMMLinearLayerConfig as Cfg,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        QuantKey,
        ScaleDesc,
    )

    fp8 = torch.float8_e4m3fn
    return Cfg(
        weight_quant_key=QuantKey(fp8, ScaleDesc(torch.float32, True, weight_group_shape)),
        activation_quant_key=QuantKey(
            fp8, ScaleDesc(torch.float32, False, activation_group_shape)
        ),
        weight_shape=(256, 512),
        input_dtype=torch.float16,
        out_dtype=torch.float16,
    )


def test_can_implement_accepts_per_tensor_and_per_channel():
    """The kernel's v1 support matrix: per-tensor/per-channel weight scales."""
    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_fp8 import (
        Gfx906Fp8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    ok, reason = Gfx906Fp8ScaledMMLinearKernel.can_implement(
        _fp8_config(GroupShape.PER_TENSOR, GroupShape.PER_TOKEN)
    )
    assert ok, reason
    ok, reason = Gfx906Fp8ScaledMMLinearKernel.can_implement(
        _fp8_config(GroupShape.PER_CHANNEL, GroupShape.PER_TENSOR)
    )
    assert ok, reason


def test_can_implement_rejects_block_scaled_config():
    """Block-scaled FP8 must keep its own kernel (a separate, later variant)."""
    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_fp8 import (
        Gfx906Fp8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    ok, reason = Gfx906Fp8ScaledMMLinearKernel.can_implement(
        _fp8_config(GroupShape(128, 128), GroupShape.PER_TOKEN)
    )
    assert not ok and "per-tensor or per-channel" in (reason or "")


# ---------------------------------------------------------------------------
# block-scaled path (weight_scale_inv is [cdiv(N, 128), cdiv(K, 128)])
# ---------------------------------------------------------------------------

BLOCK_N = BLOCK_K = 128


def _quantize_block(x, fp8_dtype, rows_per_group, block_k=BLOCK_K):
    """Symmetric quantization over (rows_per_group, block_k) groups.

    Groups are laid out as ``[rows // rows_per_group, rows_per_group,
    K // block_k, block_k]``; returns ``(q, scale[rows // rows_per_group,
    K // block_k])``.  A weight uses ``rows_per_group = block_n``; a per-token
    activation uses ``rows_per_group = 1``.  Rows are zero-padded to the group
    multiple first (what a block quantizer does; zeros cannot change the amax),
    so ``N % block_n != 0`` is representable.
    """
    finfo = torch.finfo(fp8_dtype)
    rows, k = x.shape
    assert k % block_k == 0
    pad_rows = (-rows) % rows_per_group
    if pad_rows:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_rows))
    padded = rows + pad_rows
    g = x.reshape(padded // rows_per_group, rows_per_group, k // block_k, block_k)
    amax = g.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    scale = (amax / finfo.max).to(torch.float32)
    q = (g / scale).clamp(finfo.min, finfo.max).to(fp8_dtype)
    return (
        q.reshape(padded, k)[:rows].contiguous(),
        scale.reshape(padded // rows_per_group, k // block_k),
    )


def _reference_block(a_q, a_s, w_q, w_s):
    m, k = a_q.shape
    n = w_q.shape[0]
    a = a_q.to(torch.float32) * a_s.repeat_interleave(BLOCK_K, dim=1)[:m]
    w = w_q.to(torch.float32) * w_s.repeat_interleave(BLOCK_N, dim=0)[:n].repeat_interleave(
        BLOCK_K, dim=1
    )[:, :k]
    return a @ w.t()


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("m", [1, 4])
def test_block_gemv_matches_fp32_reference(fp8_ops, out_dtype, m):
    """M <= 8: the only remaining error is the fp32 accumulation order."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    n, k = 384, 1024
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    a_q, a_s = _quantize_block(a, fp8_dtype, rows_per_group=1)
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    ref = _reference_block(a_q, a_s, w_q, w_s)

    out = torch.ops._rocm_C.fp8_emu_gemv_block(
        a_q, w_q, a_s, w_s, fp8_ops(fp8_dtype), None, out_dtype, 64, BLOCK_N,
        BLOCK_K,
    )
    tol = 1e-5 if out_dtype == torch.float32 else 1e-3
    err = (out.to(torch.float32) - ref).abs().max().item() / ref.abs().max().item()
    assert err < tol, f"max|err|/max|ref| = {err}"


@pytest.mark.parametrize("m", [16, 64, 129])
def test_block_gemm_matches_fp32_reference(fp8_ops, m):
    """M > 8 prefill path, K folded per 128-element block."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    n, k = 256, 512
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    a_q, a_s = _quantize_block(a, fp8_dtype, rows_per_group=1)
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    ref = _reference_block(a_q, a_s, w_q, w_s)

    lut = fp8_ops(fp8_dtype)
    for cfg in (0, 1, 2):
        out = torch.ops._rocm_C.fp8_emu_gemm_block(
            a_q, w_q, a_s, w_s, lut, None, torch.float32, cfg, BLOCK_N, BLOCK_K
        )
        err = (out - ref).abs().max().item() / ref.abs().max().item()
        assert err < 1e-5, f"cfg={cfg} max|err|/max|ref| = {err}"


def test_block_gemm_handles_odd_shapes(fp8_ops):
    """N < block_n and M/K not tile multiples must not read out of bounds."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    m, n, k = 10, 100, 384  # N % 128 != 0, K % (128 * 4) != 0, M < BM
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    a_q, a_s = _quantize_block(a, fp8_dtype, rows_per_group=1)
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    ref = _reference_block(a_q, a_s, w_q, w_s)
    out = torch.ops._rocm_C.fp8_emu_gemm_block(
        a_q, w_q, a_s, w_s, fp8_ops(fp8_dtype), None, torch.float32, 0, BLOCK_N,
        BLOCK_K,
    )
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-5, f"max|err|/max|ref| = {err}"


def test_block_accepts_padded_weight_row_stride(fp8_ops):
    """`process_fp8_weight_block_strategy` pads K, so stride(0) != K.

    The padded view must give bit-identical results to the contiguous weight
    (the kernel indexes rows through stride(0)).
    """
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    m, n, k = 3, 256, 512
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    a_q, a_s = _quantize_block(a, fp8_dtype, rows_per_group=1)
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    lut = fp8_ops(fp8_dtype)

    pad = 256  # what _maybe_pad_fp8_weight uses for 1-byte elements
    buf = torch.zeros(n, k + pad, device="cuda", dtype=w_q.dtype)
    buf[:, :k] = w_q
    w_padded = buf[:, :k]
    assert not w_padded.is_contiguous() and w_padded.stride(1) == 1

    dense = torch.ops._rocm_C.fp8_emu_gemv_block(
        a_q, w_q, a_s, w_s, lut, None, torch.float32, 64, BLOCK_N, BLOCK_K
    )
    padded = torch.ops._rocm_C.fp8_emu_gemv_block(
        a_q, w_padded, a_s, w_s, lut, None, torch.float32, 64, BLOCK_N, BLOCK_K
    )
    assert torch.equal(dense, padded)


def test_block_scalar_bias_and_scales():
    """Bias is added in the output dtype, after the scale folding."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    m, n, k = 2, 256, 256
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    bias = torch.randn(n, device="cuda", dtype=torch.float16) * 0.1
    a_q, a_s = _quantize_block(a, fp8_dtype, rows_per_group=1)
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    ref = _reference_block(a_q, a_s, w_q, w_s) + bias.to(torch.float32)

    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_fp8 import _device_lut

    lut = _device_lut(fp8_dtype, torch.device("cuda"))
    out = torch.ops._rocm_C.fp8_emu_gemv_block(
        a_q, w_q, a_s, w_s, lut, bias, torch.float16, 64, BLOCK_N, BLOCK_K
    )
    err = (out.to(torch.float32) - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-3


def test_block_kernel_gating_and_can_implement(monkeypatch):
    """The block kernel must refuse the target unless explicitly enabled."""
    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_fp8 import (
        Gfx906Fp8BlockScaledMMLinearKernel as K,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    monkeypatch.setenv("VLLM_DCU_USE_FP8_EMU", "0")
    import vllm.envs as envs

    envs.__dict__.pop("VLLM_DCU_USE_FP8_EMU", None)
    ok, reason = K.is_supported()
    assert not ok and "VLLM_DCU_USE_FP8_EMU" in (reason or "")

    monkeypatch.setenv("VLLM_DCU_USE_FP8_EMU", "1")
    ok, reason = K.can_implement(_fp8_config(GroupShape(128, 128), GroupShape(1, 128)))
    assert ok, reason
    # A per-tensor / per-channel weight is the *other* kernel's job.
    ok, reason = K.can_implement(
        _fp8_config(GroupShape.PER_CHANNEL, GroupShape.PER_TENSOR)
    )
    assert not ok and "block-scaled" in (reason or "")
    # block_k must be a multiple of 32 (the GEMM folds scales on chunk bounds).
    ok, reason = K.can_implement(_fp8_config(GroupShape(128, 100), GroupShape(1, 100)))
    assert not ok and "multiple of 32" in (reason or "")
