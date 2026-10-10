# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression guard for the gfx906 + DTK 26.04 fp16 packed-SiLU miscompile.

``csrc/libtorch_stable/activation_kernels.cu`` used to send every fp16 ``_C``
activation through the ``__half2`` vectorized branch.  On Hygon Z100 (gfx906)
with DTK 26.04 that branch is miscompiled: a few percent of the elements come
back as inf/NaN (and, for ``silu_and_mul_with_clamp``, as wrong finite values),
non-deterministically, while the scalar branch of the same kernel is exact.

The csrc fix forces the scalar branch for fp16 on that exact combination, so
these tests call the raw ``torch.ops._C`` operators -- not the ``CustomOp``
wrappers with their Python fallback -- and they are skipped everywhere else,
where the vectorized path is correct.  See
``docs/result/z100-07b-eager-nan-result.md`` and
``docs/result/z100-07c-packed-silu-csrc-fix-result.md``.
"""

import pytest
import torch

import vllm._C  # noqa: F401
from vllm.platforms import current_platform


def _dtk_gfx906() -> bool:
    if not torch.cuda.is_available() or not current_platform.is_rocm():
        return False
    from vllm.platforms.hygon_dtk import is_hygon_dtk
    from vllm.platforms.rocm import on_gfx906

    return is_hygon_dtk() and on_gfx906()


pytestmark = pytest.mark.skipif(
    not _dtk_gfx906(),
    reason="DTK + gfx906 only: elsewhere the vectorized fp16 path is correct",
)

# d % 8 == 0 selects use_vec for fp16, |gate| <= 5.9 is the range the DTK
# miscompile corrupts, and more than one launch is needed because the bad
# elements move between runs.
_D = 3072
_ATOL = 5e-2
_RTOL = 2e-2


def _half_gate_up(rows: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    x = ((torch.rand(rows, 2 * _D, device="cuda") * 2 - 1) * 5.9).half()
    return x


@torch.inference_mode()
def test_silu_and_mul_fp16_is_finite_and_exact():
    """The fp16 ``_C.silu_and_mul`` must never take the broken packed branch."""
    x = _half_gate_up(8, 1234)
    gate, up = x[:, :_D], x[:, _D:]
    ref = torch.nn.functional.silu(gate.float()) * up.float()

    for trial in range(8):
        out = torch.empty((8, _D), dtype=torch.float16, device="cuda")
        torch.ops._C.silu_and_mul(out, x)
        torch.cuda.synchronize()
        assert torch.isfinite(out).all(), f"launch {trial} produced inf/NaN"
        torch.testing.assert_close(out.float(), ref, atol=_ATOL, rtol=_RTOL)


@torch.inference_mode()
def test_silu_and_mul_with_clamp_fp16_is_finite_and_exact():
    """The clamp variant shares the broken ``act_first`` packed instantiation."""
    x = _half_gate_up(8, 4321)
    limit = 5.9
    gate = torch.clamp(x[:, :_D], max=limit)
    up = torch.clamp(x[:, _D:], min=-limit, max=limit)
    ref = torch.nn.functional.silu(gate.float()) * up.float()

    out = torch.empty((8, _D), dtype=torch.float16, device="cuda")
    torch.ops._C.silu_and_mul_with_clamp(out, x, limit, 1.0, 0.0)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, atol=_ATOL, rtol=_RTOL)


@torch.inference_mode()
def test_moe_activation_silu_fp16_matches_reference():
    """``apply_moe_activation`` calls ``_C.silu_and_mul`` directly (no CustomOp)."""
    from vllm.model_executor.layers.fused_moe.activation import (
        MoEActivation,
        apply_moe_activation,
    )

    torch.manual_seed(3)
    rows, d = 32, 3072
    x = ((torch.rand(rows, 2 * d, device="cuda") * 2 - 1) * 5.9).half()
    ref = torch.nn.functional.silu(x[:, :d].float()) * x[:, d:].float()

    out = torch.empty((rows, d), dtype=torch.float16, device="cuda")
    apply_moe_activation(MoEActivation.SILU, out, x)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, atol=_ATOL, rtol=_RTOL)
