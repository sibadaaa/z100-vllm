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
    reason="gfx906 only: the FP8 emulation kernel targets gfx906",
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
@pytest.mark.parametrize("cfg", [0, 1, 2, 11, 12, 13, 14, 15, 16, 17, 18, 19])
def test_block_gemm_matches_fp32_reference(fp8_ops, m, cfg):
    """M > 8 prefill path, K folded per 128-element block; cfgs 11..15 are the
    BM=8/16 decode tiles, which must also be exact when M exceeds their tile."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    n, k = 256, 512
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    a_q, a_s = _quantize_block(a, fp8_dtype, rows_per_group=1)
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    ref = _reference_block(a_q, a_s, w_q, w_s)

    lut = fp8_ops(fp8_dtype)
    out = torch.ops._rocm_C.fp8_emu_gemm_block(
        a_q, w_q, a_s, w_s, lut, None, torch.float32, cfg, BLOCK_N, BLOCK_K
    )
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-5, f"cfg={cfg} max|err|/max|ref| = {err}"


@pytest.mark.parametrize("m", [5, 6, 7, 8])
@pytest.mark.parametrize("cfg", ["legacy", "auto", 11, 12, 13, 14, 15, 16, 17, 18, 19])
def test_block_gemv_decode_m8_uses_exact_tiles(fp8_ops, monkeypatch, cfg, m):
    """M=5..8/K=5120 cannot stream (M*K*2 > LDS gate) and used to fall back to
    the BM=32 prefill tile, computing 4x the MACs needed.  VLLM_DCU_FP8_DECODE_CFG
    selects the BM=8/16 tiles without a rebuild; every setting must agree with
    the fp32 reference and with each other."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    n, k = 512, 5120  # production K, block 128x128
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    a_q, a_s = _quantize_block(a, fp8_dtype, rows_per_group=1)
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    ref = _reference_block(a_q, a_s, w_q, w_s)
    lut = fp8_ops(fp8_dtype)

    if cfg == "legacy":
        monkeypatch.delenv("VLLM_DCU_FP8_DECODE_CFG", raising=False)
    else:
        monkeypatch.setenv("VLLM_DCU_FP8_DECODE_CFG", str(cfg))
    out = torch.ops._rocm_C.fp8_emu_gemv_block(
        a_q, w_q, a_s, w_s, lut, None, torch.float32, 64, BLOCK_N, BLOCK_K
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


# ---------------------------------------------------------------------------
# decode range M = 1..8: the streaming GEMV
# (csrc/rocm/fp8_emu_gemm.cu, fp8_emu_gemv_stream_block_kernel)
# ---------------------------------------------------------------------------
# The streaming path is selected inside the GEMV entry point when the geometry
# fits (K and the W row stride multiples of 16, M*K*2 LDS budget), and falls
# back to `fp8_emu_gemv_block_kernel` otherwise.  Both must agree with the fp32
# reference; only the fp32 accumulation order may differ.


def _block_gemv(a_q, w_q, a_s, w_s, lut, bias, out_dtype, block_n=BLOCK_N,
                block_k=BLOCK_K):
    return torch.ops._rocm_C.fp8_emu_gemv_block(
        a_q, w_q, a_s, w_s, lut, bias, out_dtype, 64, block_n, block_k
    )


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 8])
def test_block_gemv_decode_m_range(fp8_ops, out_dtype, m):
    """M = 1..8 (draft M=1, MTP verify M=1+spec) must stay in tolerance."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    n, k = 384, 1024
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    a_q, a_s = _quantize_block(a, fp8_dtype, rows_per_group=1)
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    ref = _reference_block(a_q, a_s, w_q, w_s)

    out = _block_gemv(a_q, w_q, a_s, w_s, fp8_ops(fp8_dtype), None, out_dtype)
    tol = 1e-5 if out_dtype == torch.float32 else 1e-3
    err = (out.to(torch.float32) - ref).abs().max().item() / ref.abs().max().item()
    assert err < tol, f"m={m} max|err|/max|ref| = {err}"


def test_block_gemv_verify_shape_m3(fp8_ops):
    """M=3 on a production-shaped shard (K=5120, block 128x128)."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    m, n, k = 3, 1024, 5120
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    a_q, a_s = _quantize_block(a, fp8_dtype, rows_per_group=1)
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    ref = _reference_block(a_q, a_s, w_q, w_s)

    out = _block_gemv(a_q, w_q, a_s, w_s, fp8_ops(fp8_dtype), None, torch.float16)
    err = (out.to(torch.float32) - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-3, f"max|err|/max|ref| = {err}"


def test_block_gemv_m8_large_k_falls_back(fp8_ops):
    """M=8/K=5120 exceeds the streaming LDS budget: the GEMM fallback must work.

    The decode entry point routes M=5..8 with an unfittable M*K to the tiled
    GEMM (measured 3-10x faster than the old GEMV there); this pins that the
    fallback is numerically equivalent.
    """
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    m, n, k = 8, 256, 5120
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    a_q, a_s = _quantize_block(a, fp8_dtype, rows_per_group=1)
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    ref = _reference_block(a_q, a_s, w_q, w_s)

    out = _block_gemv(a_q, w_q, a_s, w_s, fp8_ops(fp8_dtype), None, torch.float16)
    err = (out.to(torch.float32) - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-3, f"max|err|/max|ref| = {err}"


@pytest.mark.parametrize(
    "n,k",
    [(100, 128), (128, 1152), (37, 384), (512, 256)],
    ids=["n<block_n,k=128", "ragged_k", "n<block_n_odd", "k_small"],
)
def test_block_gemv_odd_decode_shapes(fp8_ops, n, k):
    """N < block_n, N not a multiple of the row group, short/ragged K."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    m = 3
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    a_q, a_s = _quantize_block(a, fp8_dtype, rows_per_group=1)
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    ref = _reference_block(a_q, a_s, w_q, w_s)

    out = _block_gemv(a_q, w_q, a_s, w_s, fp8_ops(fp8_dtype), None, torch.float32)
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-5, f"n={n} k={k} max|err|/max|ref| = {err}"


def test_block_gemv_decode_stream_matches_gemm(fp8_ops):
    """Decode M<=8 and the prefill GEMM must agree on the same operands."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    m, n, k = 4, 512, 1024
    a = torch.randn(m, k, device="cuda") * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    a_q, a_s = _quantize_block(a, fp8_dtype, rows_per_group=1)
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    lut = fp8_ops(fp8_dtype)

    gemv = _block_gemv(a_q, w_q, a_s, w_s, lut, None, torch.float32)
    gemm = torch.ops._rocm_C.fp8_emu_gemm_block(
        a_q, w_q, a_s, w_s, lut, None, torch.float32, 0, BLOCK_N, BLOCK_K
    )
    assert torch.allclose(gemv, gemm, rtol=1e-3, atol=1e-3), (
        (gemv - gemm).abs().max().item()
    )


# ---------------------------------------------------------------------------
# W8A16 route: fp16 activations against fp8 block-scaled weights
# (csrc/rocm/fp8_emu_gemm.cu, fp8_emu_gemm_block_w8a16)
# ---------------------------------------------------------------------------
# docs/design/z100-w8a16-prefill-fp16-construction.md.  The activation is no
# longer quantized, so the reference is the *raw* fp16 activation against the
# same fp8 weight: the only remaining error is the fp32 accumulation order (or
# the final cast).  `cfg` values are the W8A16 tile set (1..5) plus the W8A8
# numbers the Python A/B knobs still pass, which the kernel maps onto it.

W8A16_CFGS = [0, 2, 3, 4, 5, 16, 36, 40]


def _dequantized_weight(w_q, w_s):
    """fp32 dequantized weight, i.e. the W8A16 reference operand."""
    n, k = w_q.shape
    return (
        w_q.to(torch.float32)
        * w_s.repeat_interleave(BLOCK_N, dim=0)[:n].repeat_interleave(
            BLOCK_K, dim=1
        )[:, :k]
    )


def _reference_block_w8a16(a, w_q, w_s):
    return a.to(torch.float32) @ _dequantized_weight(w_q, w_s).t()


def _w8a16(a, w_q, w_s, lut, bias=None, out_dtype=torch.float32, cfg=0):
    return torch.ops._rocm_C.fp8_emu_gemm_block_w8a16(
        a, w_q, w_s, lut, bias, out_dtype, 64, cfg, BLOCK_N, BLOCK_K
    )


def _rel_err(out, ref):
    return (out.to(torch.float32) - ref).abs().max().item() / ref.abs().max().item()


def _cos(x, y):
    x = x.to(torch.float64).flatten()
    y = y.to(torch.float64).flatten()
    return float((x * y).sum() / (x.norm() * y.norm()))


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 8])
def test_block_gemv_w8a16(fp8_ops, m, out_dtype):
    """Decode M = 1..8: the GEMV reads the fp16 activation directly."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    n, k = 384, 1024
    a = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    ref = _reference_block_w8a16(a, w_q, w_s)

    out = _w8a16(a, w_q, w_s, fp8_ops(fp8_dtype), out_dtype=out_dtype)
    tol = 1e-5 if out_dtype == torch.float32 else 1e-3
    err = _rel_err(out, ref)
    assert err < tol, f"m={m} max|err|/max|ref| = {err}"


@pytest.mark.parametrize("m", [16, 64, 129, 512])
@pytest.mark.parametrize("cfg", W8A16_CFGS)
def test_block_gemm_w8a16(fp8_ops, m, cfg):
    """Prefill tile path: every W8A16 cfg must match the fp32 reference."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    n, k = 256, 512
    a = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    ref = _reference_block_w8a16(a, w_q, w_s)

    out = _w8a16(a, w_q, w_s, fp8_ops(fp8_dtype), cfg=cfg)
    err = _rel_err(out, ref)
    assert err < 1e-5, f"m={m} cfg={cfg} max|err|/max|ref| = {err}"


def test_block_w8a16_bias_and_bf16_out(fp8_ops):
    """Bias is added in out_dtype, and the bf16 *output* cast still works."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    m, n, k = 17, 256, 512
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = torch.randn(n, k, device="cuda") * 0.02
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    ref = _reference_block_w8a16(a, w_q, w_s) + bias.to(torch.float32).reshape(1, -1)

    out = _w8a16(
        a, w_q, w_s, fp8_ops(fp8_dtype), bias=bias, out_dtype=torch.float16
    )
    assert _rel_err(out, ref) < 1e-3

    bf16 = _w8a16(a, w_q, w_s, fp8_ops(fp8_dtype), out_dtype=torch.bfloat16)
    assert bf16.dtype == torch.bfloat16
    # The bf16 arm carries no bias, so it compares against the unbiased
    # reference (the tolerance covers the final bf16 cast, ~2^-8 relative).
    assert _rel_err(bf16, _reference_block_w8a16(a, w_q, w_s)) < 1e-2


def test_block_w8a16_accepts_padded_weight_row_stride(fp8_ops):
    """`process_fp8_weight_block_strategy` pads K, so stride(0) != K."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    m, n, k, pad = 32, 256, 512, 256
    a = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    lut = fp8_ops(fp8_dtype)

    padded = torch.zeros(n, k + pad, dtype=torch.uint8, device="cuda")
    padded[:, :k] = w_q.view(torch.uint8)
    w_pad = padded.view(fp8_dtype)[:, :k]
    assert w_pad.stride(0) == k + pad and w_pad.stride(1) == 1

    out = _w8a16(a, w_pad, w_s, lut)
    ref = _w8a16(a, w_q.contiguous(), w_s, lut)
    assert torch.equal(out, ref), "padded row stride changed the result"


def test_block_w8a16_rejects_bad_activations_and_scales(fp8_ops):
    """A must be fp16 (gfx906 has no bf16 datapath) and w_s must be well-shaped."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    n, k = 128, 256
    w = torch.randn(n, k, device="cuda") * 0.02
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    lut = fp8_ops(fp8_dtype)

    a_bf16 = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="float16"):
        _w8a16(a_bf16, w_q, w_s, lut)

    a_q, _ = _quantize_block(torch.randn(16, k, device="cuda"), fp8_dtype, 1)
    with pytest.raises(RuntimeError, match="float16"):
        _w8a16(a_q, w_q, w_s, lut)

    a = torch.randn(16, k, device="cuda", dtype=torch.float16)
    # Truncate the K dimension of w_s: nblk=1 here, so [:1] would be a no-op.
    with pytest.raises(RuntimeError, match="w_s must be"):
        _w8a16(a, w_q, w_s[:, :1].contiguous(), lut)

    with pytest.raises(RuntimeError, match="multiple of 8"):
        _w8a16(a[:, :250].contiguous(), w_q[:, :250].contiguous(), w_s, lut)

    with pytest.raises(RuntimeError, match="multiple of 32"):
        torch.ops._rocm_C.fp8_emu_gemm_block_w8a16(
            a, w_q, w_s, lut, None, torch.float32, 64, 0, BLOCK_N, 48
        )


def test_w8a16_route_is_off_unless_asked(fp8_ops):
    """Default build/route stays W8A8: `apply_input_quant` follows the switch.

    Polarity: the base-class hook True == quantize (W8A8), False == raw fp16
    (W8A16).  Switch off (or the op missing) must keep quantization ON -- the
    inverted polarity crashed production at the first forward (the base handed
    raw fp16 to the missing W8A16 op).
    """
    import vllm.model_executor.kernels.linear.scaled_mm.gfx906_fp8 as m

    engaged = m._W8A16 and m.ops_available_block_w8a16()
    assert (
        m.Gfx906Fp8BlockScaledMMLinearKernel.apply_input_quant
        is not engaged
    )
    if m._W8A16_MODE in ("0", ""):
        assert (
            m.Gfx906Fp8BlockScaledMMLinearKernel.apply_input_quant is True
        ), "switch off must keep the W8A8 quantization"


# ---------------------------------------------------------------------------
# prefill fp16 weight materialization (csrc/rocm/fp8_dequant.cuh)
# ---------------------------------------------------------------------------


def _dequant(w_q, w_s, lut):
    out = torch.empty(
        w_q.shape, dtype=torch.float16, device=w_q.device
    )
    torch.ops._rocm_C.fp8_dequant_to_fp16_out(out, w_q, w_s, lut, BLOCK_N, BLOCK_K)
    return out


def test_dequant_kernel_matches_reference(fp8_ops):
    """fp32 multiply then one fp16 rounding, no fp16-domain product.

    §6.2 gate: bit-equal OR <= 1 ulp.  The residual deviation is exact
    midpoint ties: `__float2half` on this toolchain lowers to a
    magic-multiply conversion that double-rounds ties (torch's cast is
    round-to-nearest-even), so out and ref may sit on opposite sides of an
    exact tie -- at most 1 fp16 ulp apart, each a valid nearest value
    (measured ~0.01% of elements, |out-ref| = exactly 1 ulp every time).
    """
    torch.manual_seed(0)
    for fp8_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        n, k = 384, 1024
        w = torch.randn(n, k, device="cuda") * 0.02
        w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
        out = _dequant(w_q, w_s, fp8_ops(fp8_dtype))
        ref = _dequantized_weight(w_q, w_s).to(torch.float16)
        assert out.dtype == torch.float16 and out.shape == w_q.shape
        dist = (
            out.view(torch.int16).to(torch.int32)
            - ref.view(torch.int16).to(torch.int32)
        ).abs()
        assert (dist <= 1).all(), (
            f"off by >1 ulp at {(dist > 1).sum().item()} elements "
            f"(max {dist.max().item()})"
        )


def test_dequant_accepts_padded_weight_row_stride(fp8_ops):
    """The kernel must address W through `stride(0)`, not assume K."""
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    n, k, pad = 256, 512, 256
    w = torch.randn(n, k, device="cuda") * 0.02
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    lut = fp8_ops(fp8_dtype)
    padded = torch.zeros(n, k + pad, dtype=torch.uint8, device="cuda")
    padded[:, :k] = w_q.view(torch.uint8)
    out = _dequant(padded.view(fp8_dtype)[:, :k], w_s, lut)
    assert torch.equal(out, _dequant(w_q.contiguous(), w_s, lut))


def test_dequant_rejects_bad_layout(fp8_ops):
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    n, k = 128, 256
    w_q, w_s = _quantize_block(
        torch.randn(n, k, device="cuda") * 0.02, fp8_dtype, BLOCK_N
    )
    lut = fp8_ops(fp8_dtype)
    with pytest.raises(RuntimeError, match="out must be float16"):
        torch.ops._rocm_C.fp8_dequant_to_fp16_out(
            torch.empty(n, k, dtype=torch.float32, device="cuda"), w_q, w_s, lut,
            BLOCK_N, BLOCK_K,
        )
    with pytest.raises(RuntimeError, match=r"out must be \[N, K\]"):
        torch.ops._rocm_C.fp8_dequant_to_fp16_out(
            torch.empty(n, k // 2, dtype=torch.float16, device="cuda"), w_q, w_s,
            lut, BLOCK_N, BLOCK_K,
        )
    with pytest.raises(RuntimeError, match="multiple of 8"):
        torch.ops._rocm_C.fp8_dequant_to_fp16_out(
            torch.empty(n, k - 4, dtype=torch.float16, device="cuda"),
            w_q[:, : k - 4].contiguous(), w_s, lut, BLOCK_N, BLOCK_K,
        )


class _StubKernel:
    """Minimal stand-in for the block-scaled kernel instance."""

    def __init__(self, out_dtype=torch.float16):
        self.apply_input_quant = False
        self.weight_group_shape = type("GS", (), {"row": BLOCK_N, "col": BLOCK_K})()
        self.config = type("Cfg", (), {"out_dtype": out_dtype})()


class _StubLayer:
    """Minimal stand-in for a quantized linear layer (`FP8BlockParams`)."""

    weight = None
    weight_scale = None
    weight_scale_inv = None
    input_scale = None
    input_scale_ub = None

    def __init__(self, w, w_s, prefix="model.layers.0.mlp.down_proj"):
        self.weight = w
        self.weight_scale = w_s
        self.prefix = prefix


def _prefill_module():
    import vllm.model_executor.kernels.linear.scaled_mm.gfx906_fp16_prefill as pre

    return pre


def test_prefill_route_matches_w8a16(fp8_ops, monkeypatch):
    """The fp16-materialized route must agree with the W8A16 kernel."""
    pre = _prefill_module()
    monkeypatch.setattr(pre, "_MODE", "1")
    monkeypatch.setattr(pre, "_DEFAULT_MIN_M", 0)
    pre._WS.clear()
    pre._WS_STREAM.clear()

    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    m, n, k = 64, 256, 512
    a = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.5
    w = torch.randn(n, k, device="cuda") * 0.02
    w_q, w_s = _quantize_block(w, fp8_dtype, rows_per_group=BLOCK_N)
    lut = fp8_ops(fp8_dtype)

    routed = pre.try_prefill(_StubKernel(), _StubLayer(w_q, w_s), a, None)
    assert routed is not None, "router declined a shape it should serve"
    assert routed.dtype == torch.float16 and routed.shape == (m, n)

    direct = _w8a16(a, w_q, w_s, lut, out_dtype=torch.float16)
    assert _cos(routed, direct) >= 0.99999
    assert _rel_err(routed, _reference_block_w8a16(a, w_q, w_s)) < 1e-3

    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    routed_b = pre.try_prefill(_StubKernel(), _StubLayer(w_q, w_s), a, bias)
    assert routed_b is not None
    assert _cos(routed_b, direct + bias) >= 0.99999


def test_prefill_route_declines_small_m_and_bad_dtypes(fp8_ops, monkeypatch):
    pre = _prefill_module()
    monkeypatch.setattr(pre, "_MODE", "auto")
    monkeypatch.setattr(pre, "_DEFAULT_MIN_M", 4096)
    pre._WS.clear()
    pre._WS_STREAM.clear()

    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    n, k = 256, 512
    w_q, w_s = _quantize_block(
        torch.randn(n, k, device="cuda") * 0.02, fp8_dtype, BLOCK_N
    )
    kernel = _StubKernel()
    layer = _StubLayer(w_q, w_s)

    # Below the min-M floor (the measured table bottoms out at 16): the W8A16
    # tile kernel handles it.
    small = torch.randn(8, k, device="cuda", dtype=torch.float16)
    assert pre.try_prefill(kernel, layer, small, None) is None

    # The W8A8 route (activation quantized in python) must never be intercepted.
    big = torch.randn(4096, k, device="cuda", dtype=torch.float16)
    kernel_quant = _StubKernel()
    kernel_quant.apply_input_quant = True
    assert pre.try_prefill(kernel_quant, layer, big, None) is None

    # bf16 / non-K-major weights are not materialized either.
    assert pre.try_prefill(kernel, layer, big.to(torch.bfloat16), None) is None
    w_kmajor = w_q.t().contiguous().t()
    assert pre.try_prefill(kernel, _StubLayer(w_kmajor, w_s), big, None) is None


def test_prefill_route_falls_back_on_oom(fp8_ops, monkeypatch):
    """A workspace OOM must return None (fall back), never raise."""
    pre = _prefill_module()
    monkeypatch.setattr(pre, "_MODE", "1")
    monkeypatch.setattr(pre, "_DEFAULT_MIN_M", 0)
    pre._WS.clear()
    pre._WS_STREAM.clear()

    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    n, k = 256, 512
    w_q, w_s = _quantize_block(
        torch.randn(n, k, device="cuda") * 0.02, fp8_dtype, BLOCK_N
    )
    a = torch.randn(64, k, device="cuda", dtype=torch.float16)

    real_empty = torch.empty

    def _oom(*args, **kwargs):
        if kwargs.get("dtype") is torch.float16:
            raise torch.OutOfMemoryError("simulated workspace OOM")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", _oom)
    assert pre.try_prefill(_StubKernel(), _StubLayer(w_q, w_s), a, None) is None


def test_prefill_route_snapshot(monkeypatch):
    """Regression guard: the (role, M) -> route decision must not drift."""
    pre = _prefill_module()
    monkeypatch.setattr(pre, "_MODE", "auto")
    monkeypatch.setattr(pre, "_DEFAULT_MIN_M", 4096)

    roles = ["gate_up_proj", "down_proj", "qkv_proj", "o_proj", "out_proj"]
    table = {role: pre._min_m(role, None) for role in roles}
    # The P1-measured crossover table (w8a16_min_m_sweep.py): wide-N gate_up
    # and untested-geometry qkv cross at 128, the rest at the 16 floor.
    assert table == {
        "gate_up_proj": 128,
        "down_proj": 16,
        "qkv_proj": 128,
        "o_proj": 16,
        "out_proj": 16,
    }
    # Unknown role falls back to the default rather than routing everything.
    assert pre._min_m("lm_head", None) == 4096

    decisions = [
        (m, m >= pre._min_m("down_proj", None) and m >= 16)
        for m in (8, 16, 1024, 4095, 4096, 8192)
    ]
    assert decisions == [
        (8, False),
        (16, True),
        (1024, True),
        (4095, True),
        (4096, True),
        (8192, True),
    ]
