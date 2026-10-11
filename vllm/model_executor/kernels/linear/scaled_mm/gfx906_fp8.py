# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""W8A8 FP8 scaled-MM kernel for gfx906 (Z100), emulated on the fp16 datapath.

The device code lives in ``csrc/rocm/fp8_emu_gemm.cu`` and is registered as
``torch.ops._rocm_C.fp8_emu_gemv`` / ``..._gemm``.

Why a new kernel is needed: gfx906 has no FP8 hardware at all -- no fp8
conversion instruction, no fp8 MFMA, and `torch._scaled_mm`'s FP8 path is not
supported -- so every existing FP8 candidate refuses the target
(``ROCmFP8ScaledMMLinearKernel``: "wvSplitKQ is not built on gfx906 (no MFMA)",
the torch kernels: "requires a platform with torch FP8 scaled-MM support").
The only FP8 path that used to run (block-scaled) fell back to scalar
``v_fmac_f32`` emulation, measured ~36x slower than fp16.

Design notes:

* Weights stay in the layer's native ``[K, N]`` (view of the ``[N, K]``
  checkpoint) layout -- **never dequantized, never copied**.  The kernel is
  handed a zero-copy ``B.t()`` view, so the resident footprint is exactly the
  FP8 checkpoint's (a 27B FP8 model is ~27 GiB; an fp16 copy would need 54 GiB
  and cannot fit this platform).
* fp8 -> fp16 widening happens inside the kernel through a 256-entry table in
  LDS (``ds_read_u16``).  Both ``float8_e4m3fn`` and ``float8_e5m2`` are
  lossless into fp16 (e4m3 peaks at 448 with 3 mantissa bits; e5m2 peaks at
  57344 < 65504 with 2, its subnormals landing on fp16 subnormals), so the
  widening adds **no** error of its own, unlike the packed bit-trick route
  which cannot represent e4m3 subnormals.
* The MAC is ``v_dot2_f32_f16`` (2 fp16 MACs/instruction, fp32 accumulator),
  which is exactly the accumulation contract FP8 wants.
* GEMV (M <= ``_GEMV_MAX_M``) for decode, tiled GEMM above it; the
  GEMV/GEMM branch lives inside the opaque ``custom_op`` body so the traced
  graph carries no symbolic-M branch (same lesson as ``gfx906_sdot4.py``).
* Rollback / A-B switch: ``VLLM_DCU_USE_FP8_EMU=1`` opts in (default off).
  When disabled, the ROCm kernel table falls through to the previous
  behaviour (per-tensor/per-channel FP8 refused, block-scaled on Triton).
"""

import functools

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx906

from .gfx906_common import gfx906_emu_enabled
from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

logger = init_logger(__name__)

# M <= _GEMV_MAX_M goes to the GEMV kernel (must match kMsMax in the .cu).
_GEMV_MAX_M = 8
# GEMV threads per block: tuned value from the INT8 sdot4 project (power of two).
_GEMV_THREADS = 64

_FP8_EMU_MM_OP = "vllm::fp8_emu_mm"


def ops_available() -> bool:
    return hasattr(torch.ops._rocm_C, "fp8_emu_gemm") and hasattr(
        torch.ops._rocm_C, "fp8_emu_gemv"
    )


def ops_available_block() -> bool:
    return hasattr(torch.ops._rocm_C, "fp8_emu_gemm_block") and hasattr(
        torch.ops._rocm_C, "fp8_emu_gemv_block"
    )


# ---------------------------------------------------------------------------
# fp8 -> fp16 conversion table (host side, exact)
# ---------------------------------------------------------------------------
# Every finite e4m3fn / e5m2 value is exactly representable in fp16, so the
# table introduces no rounding: the only error in the FP8 path is the
# quantization error already present in the checkpoint, plus the fp32
# accumulation order.
@functools.lru_cache(maxsize=8)
def fp8_to_fp16_lut(fp8_dtype: torch.dtype) -> torch.Tensor:
    """Build the 256-entry fp16 lookup table for ``fp8_dtype``."""
    bits = torch.arange(256, dtype=torch.int32)
    sign = torch.where((bits >> 7) & 1 == 1, -1.0, 1.0)
    exp = (bits >> 3) & 0xF
    man = bits & 0x7
    if fp8_dtype == torch.float8_e4m3fn:
        mant_scale = 1.0 / 8.0
        bias = 7
        nan_codes = (exp == 0xF) & (man == 0x7)
        inf_codes = torch.zeros_like(nan_codes)
    elif fp8_dtype == torch.float8_e5m2:
        exp = (bits >> 2) & 0x1F
        man = bits & 0x3
        mant_scale = 1.0 / 4.0
        bias = 15
        nan_codes = (exp == 0x1F) & (man != 0)
        inf_codes = (exp == 0x1F) & (man == 0)
    else:
        raise NotImplementedError(f"no fp8 LUT for {fp8_dtype}")

    man_f = man.to(torch.float32)
    exp_f = exp.to(torch.float32)
    # Normal values: (1 + man/2^k) * 2^(e - bias).
    val = torch.where(
        exp == 0,
        man_f * mant_scale * 2.0 ** (1 - bias),  # subnormal
        (1.0 + man_f * mant_scale) * torch.pow(2.0, exp_f - bias),
    )
    out = sign * val
    if inf_codes.any():
        out = torch.where(inf_codes, sign * float("inf"), out)
    out = torch.where(nan_codes, torch.tensor(float("nan")), out)
    lut = out.to(torch.float16)
    assert lut.numel() == 256
    return lut.contiguous()


@functools.lru_cache(maxsize=16)
def _device_lut(fp8_dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """CUDA-resident copy of the LUT, cached per (dtype, device).

    The kernel stages the table into LDS at block start, so it must live on
    the GPU; the table is 512 bytes, so one cached copy per device is free.
    """
    return fp8_to_fp16_lut(fp8_dtype).to(device)


def _register_fp8_emu_mm() -> None:
    """Expose the raw kernels as a ``torch.library.custom_op`` (once).

    Dynamo treats a raw ``TORCH_LIBRARY`` op as opaque *without* a shape
    contract and pins the symbolic token dimension with an ``Eq(dim, hint)``
    guard, which then aborts vLLM's compile with a ``ConstraintViolationError``
    (S11k).  Declaring the op through ``torch.library`` (schema +
    ``register_fake``) keeps the kernel inside the compiled graph without
    leaking a specialization; see ``gfx906_sdot4.py`` for the full write-up.
    """
    if hasattr(torch.ops.vllm, "fp8_emu_mm"):
        return

    @torch.library.custom_op(_FP8_EMU_MM_OP, mutates_args=())
    def fp8_emu_mm(
        a: torch.Tensor,
        w: torch.Tensor,
        a_s: torch.Tensor,
        w_s: torch.Tensor,
        lut: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        # Runs eagerly: the branch is on the concrete runtime M, never traced.
        if a.shape[0] <= _GEMV_MAX_M:
            return torch.ops._rocm_C.fp8_emu_gemv(
                a, w, a_s, w_s, lut, bias, out_dtype, _GEMV_THREADS
            )
        return torch.ops._rocm_C.fp8_emu_gemm(a, w, a_s, w_s, lut, bias, out_dtype, 0)

    @fp8_emu_mm.register_fake
    def _(
        a: torch.Tensor,
        w: torch.Tensor,
        a_s: torch.Tensor,
        w_s: torch.Tensor,
        lut: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        return a.new_empty((a.shape[0], w.shape[0]), dtype=out_dtype)


def _register_fp8_emu_mm_block() -> None:
    """``torch.library.custom_op`` for the block-scaled kernels (once).

    Same reason as the per-tensor op above: a raw ``TORCH_LIBRARY`` op pins the
    symbolic token dimension with an ``Eq(dim, hint)`` guard and aborts vLLM's
    compile (S11k).
    """
    if hasattr(torch.ops.vllm, "fp8_emu_mm_block"):
        return

    @torch.library.custom_op("vllm::fp8_emu_mm_block", mutates_args=())
    def fp8_emu_mm_block(
        a: torch.Tensor,
        w: torch.Tensor,
        a_s: torch.Tensor,
        w_s: torch.Tensor,
        lut: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        block_n: int,
        block_k: int,
    ) -> torch.Tensor:
        # Runs eagerly: the branch is on the concrete runtime M, never traced.
        if a.shape[0] <= _GEMV_MAX_M:
            return torch.ops._rocm_C.fp8_emu_gemv_block(
                a, w, a_s, w_s, lut, bias, out_dtype, _GEMV_THREADS, block_n,
                block_k,
            )
        return torch.ops._rocm_C.fp8_emu_gemm_block(
            a, w, a_s, w_s, lut, bias, out_dtype, 0, block_n, block_k
        )

    @fp8_emu_mm_block.register_fake
    def _(
        a: torch.Tensor,
        w: torch.Tensor,
        a_s: torch.Tensor,
        w_s: torch.Tensor,
        lut: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        block_n: int,
        block_k: int,
    ) -> torch.Tensor:
        return a.new_empty((a.shape[0], w.shape[0]), dtype=out_dtype)


if ops_available():
    _register_fp8_emu_mm()

if ops_available_block():
    _register_fp8_emu_mm_block()


class Gfx906Fp8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """FP8 W8A8 scaled MM emulated with fp16 dot products (gfx906 only)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm."
        if not on_gfx906():
            return False, "requires gfx906 (fp16 dot2 emulation of fp8)."
        if not gfx906_emu_enabled("VLLM_DCU_USE_FP8_EMU"):
            return False, (
                "disabled by VLLM_DCU_USE_FP8_EMU=0 (or unset while "
                "VLLM_DCU_USE_GFX906_EMU=0)."
            )
        if not ops_available():
            return False, "torch.ops._rocm_C.fp8_emu_* is not registered."
        if not hasattr(torch.ops.vllm, "fp8_emu_mm"):
            return False, "torch.ops.vllm.fp8_emu_mm is not registered."
        return True, None

    @classmethod
    def can_implement(
        cls, c: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        w_shape = c.weight_quant_key.scale.group_shape
        if not (w_shape.is_per_tensor() or w_shape.is_per_channel()):
            return (
                False,
                "weight scales must be per-tensor or per-channel; "
                f"got group_shape={w_shape}.",
            )
        a_shape = c.activation_quant_key.scale.group_shape
        if not (a_shape.is_per_tensor() or a_shape.is_per_token()):
            return (
                False,
                "activation scales must be per-tensor or per-token; "
                f"got group_shape={a_shape}.",
            )
        return True, None

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # B is the layer weight as a [K, N] view of the [N, K] checkpoint, so
        # `.t()` is a zero-copy [N, K] view: no transpose copy, no fp16 copy.
        w = B.t()
        if not w.is_contiguous():
            raise RuntimeError(
                "Gfx906Fp8ScaledMMLinearKernel requires the fp8 weight to be "
                "contiguous as [N, K] (its checkpoint layout); got strides "
                f"{tuple(w.stride())}. Refusing to make a copy."
            )
        lut = _device_lut(A.dtype, A.device)
        # a_s/w_s may arrive as [M]/[N] or [M, 1]/[N, 1]; the kernel indexes
        # them linearly and only distinguishes "M/N elements" from "1 element".
        a_s = As.reshape(-1).contiguous() if As.numel() > 1 else As.reshape(1)
        w_s = Bs.reshape(-1).contiguous() if Bs.numel() > 1 else Bs.reshape(1)
        out = torch.ops.vllm.fp8_emu_mm(A, w, a_s, w_s, lut, bias, out_dtype)
        return out.view(*output_shape)


class Gfx906Fp8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    """FP8 block-scaled (e.g. 128x128) W8A8 emulated with fp16 dot products.

    A block-scaled checkpoint carries ``weight_scale_inv`` shaped
    ``[cdiv(N, block_n), cdiv(K, block_k)]`` and, with a dynamic activation
    scheme, a per-token-per-block activation scale shaped
    ``[M, cdiv(K, block_k)]``.  The kernel therefore accumulates per K block and
    applies the product of the two scales at the block boundary:

        out[m, n] = sum_kb (sum_{k in kb} a[m,k] * w[n,k])
                        * a_s[m, kb] * w_s[n // block_n, kb]

    Weights stay fp8 in HBM and are widened inside the kernel; nothing is
    dequantized or copied (a 27B FP8 checkpoint is ~27 GiB, an fp16 copy would
    be 54 GiB and does not fit this platform).

    ``--enforce-eager`` and the default graph mode both work: the GEMV/GEMM
    branch lives inside the opaque ``custom_op`` body, so no symbolic-M branch
    reaches Dynamo (S11k).
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm."
        if not on_gfx906():
            return False, "requires gfx906 (fp16 dot2 emulation of fp8)."
        if not gfx906_emu_enabled("VLLM_DCU_USE_FP8_EMU"):
            return False, (
                "disabled by VLLM_DCU_USE_FP8_EMU=0 (or unset while "
                "VLLM_DCU_USE_GFX906_EMU=0)."
            )
        if not ops_available_block():
            return False, "torch.ops._rocm_C.fp8_emu_*_block is not registered."
        if not hasattr(torch.ops.vllm, "fp8_emu_mm_block"):
            return False, "torch.ops.vllm.fp8_emu_mm_block is not registered."
        return True, None

    @classmethod
    def can_implement(
        cls, c: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        ok, why = super().can_implement(c)
        if not ok:
            return ok, why
        w_shape = c.weight_quant_key.scale.group_shape
        block_n, block_k = int(w_shape.row), int(w_shape.col)
        if block_n <= 0 or block_k <= 0:
            return (
                False,
                "weight scales must be block-scaled (e.g. 128x128); got "
                f"group_shape={w_shape}.",
            )
        if block_k % 32 != 0:
            return False, f"block_k={block_k} must be a multiple of 32."
        k = int(c.weight_shape[1])
        if k % block_k != 0:
            return False, f"K={k} must be a multiple of block_k={block_k}."
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        K = A.shape[1]
        # This path hands over [N, K] (Fp8LinearMethod keeps the checkpoint
        # layout for block-scaled weights), but accept the [K, N] orientation
        # too: both are zero-copy.
        w = B.t() if (B.shape[1] != K and B.shape[0] == K) else B
        if w.shape[1] != K:
            raise RuntimeError(
                "Gfx906Fp8BlockScaledMMLinearKernel: weight K mismatch "
                f"(A K={K}, weight {tuple(B.shape)})."
            )
        if w.stride(1) != 1:
            raise RuntimeError(
                "Gfx906Fp8BlockScaledMMLinearKernel requires the fp8 weight to "
                f"be K-contiguous; got strides {tuple(w.stride())}. Refusing to "
                "make a copy."
            )
        block_n = int(self.weight_group_shape.row)
        block_k = int(self.weight_group_shape.col)
        lut = _device_lut(A.dtype, A.device)
        out = torch.ops.vllm.fp8_emu_mm_block(
            A,
            w,
            As.contiguous(),
            Bs.contiguous(),
            lut,
            None,
            self.config.out_dtype,
            block_n,
            block_k,
        )
        return out
