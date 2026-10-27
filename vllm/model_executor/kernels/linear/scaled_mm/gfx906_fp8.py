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
* ``VLLM_DCU_FP8_EMU_FASTPATH=1`` skips the ``custom_op`` entirely while
  eager and calls the raw kernel; the same switch is the rollback lever
  (default off, i.e. the wrapper always runs).
"""

import contextlib
import functools
import os

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

# Register the ROCm extension before the import-time capability checks below:
# the `torch.ops._rocm_C` namespace is lazy, so without this the decisions this
# module bakes in (which custom ops to register, whether W8A16 is engaged)
# would depend on whether some earlier import happened to touch the extension.
# Same pattern as vllm/platforms/rocm.py; a build tree without the extension
# just keeps the previous (op-missing) behaviour.
with contextlib.suppress(ImportError):
    import vllm._rocm_C  # noqa: F401

logger = init_logger(__name__)

# M <= _GEMV_MAX_M goes to the GEMV kernel (must match kMsMax in the .cu).
# S18 measurement (tools/dcu/s18_fp8_block_mprobe.py, real decode shapes): the
# GEMV cost grows ~linearly with M (M=2 ~2.1x, M=4 ~3.9x) while the tiled GEMM
# is M-flat (M=1..4 all ~0.94x of the M=1 GEMV) -- so speculative-decoding
# verify (M = 1 + num_spec_tokens) wants the GEMM path.  Default keeps the old
# policy; set VLLM_DCU_FP8_GEMV_MAX_M=1 to route every M >= 2 to the GEMM.
# Clamped to the compiled GEMV range (kMsMax in fp8_emu_gemm.cu): an env value
# above it used to hand M = 9..N to `fp8_emu_gemv_block`, whose entry rejects
# M > kMsMax, and it made this python branch disagree with the W8A16 GEMV
# threshold inside the kernel (which clamps the same variable).
_GEMV_MS_MAX = 8
_GEMV_MAX_M = min(max(envs.VLLM_DCU_FP8_GEMV_MAX_M, 0), _GEMV_MS_MAX)
# GEMV threads per block: tuned value from the INT8 sdot4 project (power of two).
_GEMV_THREADS = 64
# Activation absmax floor passed to the fused quant+GEMM entry.  Must equal
# fp8_utils.per_token_group_quant_fp8's default eps so the fused kernel and the
# unfused quantizer stay bit-identical.
_QUANT_EPS = 1e-10

# Prefill (M > _GEMV_MAX_M) tile selection, without a rebuild.
#
# `csrc/rocm/fp8_emu_gemm.cu`'s cfg-0 auto rule was fitted at M = 1..129
# (perf1 s1): cfg 10 (32x64 TM2TN4 at 4 blocks/CU) for wide grids, cfg 5 (the
# same tile at the default occupancy cap) otherwise.  S22 measured the real
# prefill shapes for the first time (M = 1760/3520/7920, shapes
# 8704:5120 / 5120:4352 / 4096:5120 / 3584:5120 / 5120:1536): cfg 10 is >= cfg 5
# on every shape and 1.40x faster on N=3584:5120 (28.40 vs 39.69 ms at M=7920),
# the one shape the .cu rule still sends to cfg 5 (its `N >= 4096` cutoff).
# Tiles 1..4 and 14 were all slower, so cfg 10 is the prefill optimum here.
#
# "0" (default) passes cfg 0 through, i.e. byte-identical to the previous
# behaviour; "auto" applies the rule above with the N >= 3072 fix; 1..19 forces
# that tile for every prefill GEMM (all of them are already compiled into the
# installed .so, so no rebuild is needed to A/B).
_PREFILL_CFG_ENV = "VLLM_DCU_FP8_PREFILL_CFG"
# Snapshot at import: an A/B arm sets the variable before launch, and the hot
# path must not pay a getenv per call (the W8A16 route used to read it on every
# layer of every step, including decode where the value is unused).
_PREFILL_CFG_RAW = os.environ.get(_PREFILL_CFG_ENV, "0").strip().lower()


def _prefill_cfg(M: int, N: int) -> int:
    """Prefill GEMM tile for an ``[M, K] x [N, K]`` call (``cfg`` 0..19)."""
    raw = _PREFILL_CFG_RAW
    if raw in ("", "0"):
        return 0
    if raw == "auto":
        if N >= 6144 or (M > 32 and N >= 3072):
            return 10
        return 5
    return int(raw)

_FP8_EMU_MM_OP = "vllm::fp8_emu_mm"


def ops_available() -> bool:
    return hasattr(torch.ops._rocm_C, "fp8_emu_gemm") and hasattr(
        torch.ops._rocm_C, "fp8_emu_gemv"
    )


def ops_available_block() -> bool:
    return hasattr(torch.ops._rocm_C, "fp8_emu_gemm_block") and hasattr(
        torch.ops._rocm_C, "fp8_emu_gemv_block"
    )


def ops_available_block_quant() -> bool:
    """Whether the kernel build carries the fused quant+GEMM entry point (O5)."""
    return hasattr(torch.ops._rocm_C, "fp8_emu_gemm_block_quant")


def ops_available_block_w8a16() -> bool:
    """Whether the build carries the W8A16 entry point and its custom_op."""
    return hasattr(torch.ops._rocm_C, "fp8_emu_gemm_block_w8a16") and hasattr(
        torch.ops.vllm, "fp8_emu_mm_block_w8a16"
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
    # Static branch: e4m3fn has no inf encoding (``inf_codes`` is all-False), so
    # the e5m2-only where() is skipped and the table is bitwise unchanged.  A
    # tensor predicate here (``if inf_codes.any():``) is a data-dependent branch
    # that Dynamo cannot trace, which blocks torch.compile on gfx906.
    if fp8_dtype == torch.float8_e5m2:
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


def _register_fp8_emu_lut() -> None:
    """``torch.library.custom_op`` wrapper for the LUT lookup (S26).

    Dynamo inlines ``functools.lru_cache``-wrapped helpers, so a traced call to
    :func:`_device_lut` puts the whole 256-entry table construction (25 tensor
    ops on a 256-element tensor plus the H2D copy) into the compiled graph,
    once per layer and per step.  An opaque op keeps it out: the body runs
    eagerly and hits the cache.
    """
    if hasattr(torch.ops.vllm, "fp8_emu_lut"):
        return

    @torch.library.custom_op("vllm::fp8_emu_lut", mutates_args=())
    def fp8_emu_lut(ref: torch.Tensor, dtype_code: int) -> torch.Tensor:
        # ``ref`` is only the tracing anchor; the table is dtype/device keyed.
        dtype = torch.float8_e5m2 if dtype_code else torch.float8_e4m3fn
        return _device_lut(dtype, ref.device)

    @fp8_emu_lut.register_fake
    def _(ref: torch.Tensor, dtype_code: int) -> torch.Tensor:
        return torch.empty(256, dtype=torch.float16, device=ref.device)


def _lut_for(ref: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    """LUT for a caller that may be traced; eager callers keep the plain path.

    The op needs at least one tensor argument: with none, AOT turns the call
    into a compile-time constant and hands back a Meta tensor.
    """
    if not torch.compiler.is_compiling():
        return _device_lut(fp8_dtype, ref.device)
    return torch.ops.vllm.fp8_emu_lut(
        ref, 1 if fp8_dtype == torch.float8_e5m2 else 0
    )


# W8A16 route (docs/design/z100-w8a16-prefill-fp16-construction.md): the
# activation arrives as fp16 and is never quantized, so the A-side LUT widening
# and the per-token quantization launch disappear while the weight stays fp8 in
# HBM.  "0" (default) keeps the W8A8 path bit-for-bit identical; "1" forces the
# route and raises at load if the built op is missing; "auto" takes it whenever
# the op is present.  Read once at import, never per call (P1).
_W8A16_MODE = str(envs.VLLM_DCU_FP8_W8A16).strip().lower()
if _W8A16_MODE not in ("0", "1", "auto"):
    logger.warning_once(
        "gfx906 W8A16: unknown VLLM_DCU_FP8_W8A16=%r; treating it as 0.",
        _W8A16_MODE,
    )
    _W8A16_MODE = "0"
_W8A16 = _W8A16_MODE in ("1", "auto")

# Cheap pre-filter for the prefill-materialization hook in ``apply_weights``:
# with VLLM_DCU_FP16_PREFILL at its default ("0") the override must not touch
# the router at all -- the hook would otherwise add a call + attribute lookup
# to every layer of every step, the very per-layer host cost this workstream
# removes.  The router itself re-checks the switch and the dequant op (the
# authoritative ``gfx906_fp16_prefill._enabled``); this constant only decides
# whether that check runs.
_PREFILL_ROUTE = str(envs.VLLM_DCU_FP16_PREFILL).strip().lower() in ("1", "auto")
# Forces the auto op's tile arm: the fallback below is reached only when the
# prefill router already declined, so it must never materialize.
_FORCE_TILE_MIN_M = 1 << 30

# Eager fast path (S23 O1).  The custom_ops below exist only so Dynamo gets a
# shape contract; while eager that contract buys nothing, but its schema
# check, dispatcher hop and Python frame cost 31-55 us per call, 256 calls per
# step.  Read once at import: an A/B arm sets the variable before launch, and
# the shipped path then pays a single global load per call, not an env lookup.
_EMU_FASTPATH = envs.VLLM_DCU_FP8_EMU_FASTPATH
# Fused quant+GEMM (S23 O5).  Off by default: it replaces the two-launch
# quant -> GEMM chain with one kernel, and the unfused path must stay reachable
# for A/B and rollback.
_FUSE_QUANT = envs.VLLM_DCU_FP8_FUSE_QUANT


def _fp8_emu_mm_impl(
    a: torch.Tensor,
    w: torch.Tensor,
    a_s: torch.Tensor,
    w_s: torch.Tensor,
    lut: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Pick and issue the GEMV/GEMM kernel for one ``[M, K] x [N, K]`` call.

    Shared by the ``custom_op`` body and the eager fast path so the dispatch
    rule and its launch arguments cannot drift between the two.  Branches on
    the concrete runtime M; never traced.

    Args:
        a: Activations, ``[M, K]`` fp8.
        w: Weights, ``[N, K]`` fp8 (K-contiguous).
        a_s: Activation scales (per-tensor or per-token).
        w_s: Weight scales (per-tensor or per-channel).
        lut: fp8 -> fp16 conversion table, device-resident.
        bias: Optional ``[N]`` bias.
        out_dtype: Output dtype (fp32 or fp16).

    Returns:
        ``[M, N]`` output in ``out_dtype``.

    """
    if a.shape[0] <= _GEMV_MAX_M:
        return torch.ops._rocm_C.fp8_emu_gemv(
            a, w, a_s, w_s, lut, bias, out_dtype, _GEMV_THREADS
        )
    return torch.ops._rocm_C.fp8_emu_gemm(a, w, a_s, w_s, lut, bias, out_dtype, 0)


def _fp8_emu_mm_block_impl(
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
    """Block-scaled counterpart of :func:`_fp8_emu_mm_impl`.

    Args:
        a: Activations, ``[M, K]`` fp8.
        w: Weights, ``[N, K]`` fp8 (K-contiguous).
        a_s: Per-token-per-block activation scales, ``[M, K // block_k]``.
        w_s: Per-block weight scales, ``[N // block_n, K // block_k]``.
        lut: fp8 -> fp16 conversion table, device-resident.
        bias: Optional ``[N]`` bias.
        out_dtype: Output dtype.
        block_n: Weight block rows.
        block_k: Weight block columns.

    Returns:
        ``[M, N]`` output in ``out_dtype``.

    """
    if a.shape[0] <= _GEMV_MAX_M:
        return torch.ops._rocm_C.fp8_emu_gemv_block(
            a, w, a_s, w_s, lut, bias, out_dtype, _GEMV_THREADS, block_n, block_k
        )
    return torch.ops._rocm_C.fp8_emu_gemm_block(
        a, w, a_s, w_s, lut, bias, out_dtype,
        _prefill_cfg(a.shape[0], w.shape[0]), block_n, block_k,
    )


def _fp8_emu_mm_block_w8a16_impl(
    a: torch.Tensor,
    w: torch.Tensor,
    w_s: torch.Tensor,
    lut: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    nthreads: int,
    cfg: int,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    """Issue the W8A16 block-scaled kernel for one ``[M, K] x [N, K]`` call.

    ``a`` is fp16 and is *not* quantized; there is no activation scale.  The
    M branch (decode GEMV/stream vs tiled GEMM) stays inside the opaque body so
    Dynamo never sees a symbolic-M branch (S11k); the decode arm always passes
    cfg 0 (the kernel's auto rule picks the exact-M decode tile), so a prefill
    tile chosen for a large-M layer can never leak into a decode call.

    Args:
        a: Activations, ``[M, K]`` float16, contiguous.
        w: Weights, ``[N, K]`` fp8 (K-contiguous; row stride may be padded).
        w_s: Per-block weight scales, ``[N // block_n, K // block_k]`` fp32.
        lut: fp8 -> fp16 conversion table, device-resident.
        bias: Optional ``[N]`` bias in ``out_dtype``.
        out_dtype: Output dtype.
        nthreads: Decode GEMV launch width.
        cfg: Tile selector for the M > GEMV-max arm; 0 = the kernel's auto rule.
        block_n: Weight block rows.
        block_k: Weight block columns.

    Returns:
        ``[M, N]`` output in ``out_dtype``.

    """
    if a.shape[0] <= _GEMV_MAX_M:
        return torch.ops._rocm_C.fp8_emu_gemm_block_w8a16(
            a, w, w_s, lut, bias, out_dtype, nthreads, 0, block_n, block_k
        )
    return torch.ops._rocm_C.fp8_emu_gemm_block_w8a16(
        a, w, w_s, lut, bias, out_dtype, nthreads, cfg, block_n, block_k
    )


def _fp8_emu_mm_block_quant_impl(
    a: torch.Tensor,
    w: torch.Tensor,
    w_s: torch.Tensor,
    lut: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    nthreads: int,
    cfg: int,
    block_n: int,
    block_k: int,
    eps: float,
) -> torch.Tensor:
    """Issue the fused quant+block-GEMM kernel (S23 O5).

    The kernel quantizes the fp16 activation internally with the same recipe as
    ``fp8_utils.per_token_group_quant_fp8`` (so the result is bit-identical) and
    then runs the block-scaled GEMM in the same launch.  It selects the
    stream/tile route itself, exactly like ``fp8_emu_gemv_block``, so this
    helper is M-agnostic and must not pre-filter on ``_GEMV_MAX_M``.

    Args:
        a: Unquantized activations, ``[M, K]`` in ``out_dtype``.
        w: Weights, ``[N, K]`` fp8 (K-contiguous).
        w_s: Per-block weight scales, ``[N // block_n, K // block_k]``.
        lut: fp8 -> fp16 conversion table, device-resident.
        bias: Optional ``[N]`` bias (None on the block-scaled path).
        out_dtype: Output dtype.
        nthreads: GEMV launch width used when the GEMV route is taken.
        cfg: GEMM tile cfg; 0 resolves to ``VLLM_DCU_FP8_DECODE_CFG`` on the
            decode range inside the kernel.  Only the stream/GEMM fallback
            routes consult it: the primary fused decode tile has a fixed
            geometry (BM=8, BN=64), so ``VLLM_DCU_FP8_DECODE_CFG`` does not
            retile it.
        block_n: Weight block rows.
        block_k: Weight block columns (must be 128, the quantizer group).
        eps: Activation absmax floor (must match the quantizer's ``eps``).

    Returns:
        ``[M, N]`` output in ``out_dtype``.

    """
    return torch.ops._rocm_C.fp8_emu_gemm_block_quant(
        a, w, w_s, lut, bias, out_dtype, nthreads, cfg, block_n, block_k, eps
    )


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
        return _fp8_emu_mm_impl(a, w, a_s, w_s, lut, bias, out_dtype)

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
        return _fp8_emu_mm_block_impl(
            a, w, a_s, w_s, lut, bias, out_dtype, block_n, block_k
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


def _register_fp8_emu_mm_block_quant() -> None:
    """``custom_op`` wrappers for the fused quant+GEMM pair (once, S23 O5).

    Same S11k reason as the other wrappers: a raw ``TORCH_LIBRARY`` op would
    pin the symbolic token dimension under Dynamo.  The production op returns
    only ``out``; the witness twin returns ``(out, x_q, x_s)`` and exists for
    the bit-exact gate, because emitting the witness on the hot path would cost
    two extra allocations per call.
    """
    if hasattr(torch.ops.vllm, "fp8_emu_mm_block_quant"):
        return

    @torch.library.custom_op("vllm::fp8_emu_mm_block_quant", mutates_args=())
    def fp8_emu_mm_block_quant(
        a: torch.Tensor,
        w: torch.Tensor,
        w_s: torch.Tensor,
        lut: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        nthreads: int,
        cfg: int,
        block_n: int,
        block_k: int,
        eps: float,
    ) -> torch.Tensor:
        return _fp8_emu_mm_block_quant_impl(
            a, w, w_s, lut, bias, out_dtype, nthreads, cfg, block_n, block_k, eps
        )

    @fp8_emu_mm_block_quant.register_fake
    def _(
        a: torch.Tensor,
        w: torch.Tensor,
        w_s: torch.Tensor,
        lut: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        nthreads: int,
        cfg: int,
        block_n: int,
        block_k: int,
        eps: float,
    ) -> torch.Tensor:
        return a.new_empty((a.shape[0], w.shape[0]), dtype=out_dtype)

    @torch.library.custom_op("vllm::fp8_emu_mm_block_quant_x", mutates_args=())
    def fp8_emu_mm_block_quant_x(
        a: torch.Tensor,
        w: torch.Tensor,
        w_s: torch.Tensor,
        lut: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        nthreads: int,
        cfg: int,
        block_n: int,
        block_k: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops._rocm_C.fp8_emu_gemm_block_quant_x(
            a, w, w_s, lut, bias, out_dtype, nthreads, cfg, block_n, block_k, eps
        )

    @fp8_emu_mm_block_quant_x.register_fake
    def _(
        a: torch.Tensor,
        w: torch.Tensor,
        w_s: torch.Tensor,
        lut: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        nthreads: int,
        cfg: int,
        block_n: int,
        block_k: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            a.new_empty((a.shape[0], w.shape[0]), dtype=out_dtype),
            # Matches the C++ meta kernel: the witness x_q is the raw fp8 code
            # buffer (kByte), not a typed fp8 tensor.
            a.new_empty(a.shape, dtype=torch.uint8),
            a.new_empty(
                (a.shape[0], a.shape[1] // block_k), dtype=torch.float32
            ),
        )


def _register_fp8_emu_mm_block_w8a16() -> None:
    """``torch.library.custom_op`` wrapper for the W8A16 entry point (once).

    Same S11k reason as the other wrappers: a raw ``TORCH_LIBRARY`` op pins the
    symbolic token dimension with an ``Eq(dim, hint)`` guard and aborts vLLM's
    compile.
    """
    if hasattr(torch.ops.vllm, "fp8_emu_mm_block_w8a16"):
        return

    @torch.library.custom_op("vllm::fp8_emu_mm_block_w8a16", mutates_args=())
    def fp8_emu_mm_block_w8a16(
        a: torch.Tensor,
        w: torch.Tensor,
        w_s: torch.Tensor,
        lut: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        nthreads: int,
        cfg: int,
        block_n: int,
        block_k: int,
    ) -> torch.Tensor:
        # Runs eagerly: the branch is on the concrete runtime M, never traced.
        return _fp8_emu_mm_block_w8a16_impl(
            a, w, w_s, lut, bias, out_dtype, nthreads, cfg, block_n, block_k
        )

    @fp8_emu_mm_block_w8a16.register_fake
    def _(
        a: torch.Tensor,
        w: torch.Tensor,
        w_s: torch.Tensor,
        lut: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        nthreads: int,
        cfg: int,
        block_n: int,
        block_k: int,
    ) -> torch.Tensor:
        return a.new_empty((a.shape[0], w.shape[0]), dtype=out_dtype)


def _register_fp8_emu_mm_block_w8a16_auto() -> None:
    """``custom_op`` that owns the W8A16 *route* decision (S26).

    Compiled callers cannot branch on the symbolic token count, and the fp16
    prefill materialization (dequant into a scratch + rocBLAS) is only reached
    above a per-role threshold.  Hoisting the whole decision into one opaque
    op makes the traced graph M-agnostic while the body keeps the exact eager
    policy, so ``torch.compile`` gets the same dispatch eager does.
    """
    if hasattr(torch.ops.vllm, "fp8_emu_mm_block_w8a16_auto"):
        return

    @torch.library.custom_op(
        "vllm::fp8_emu_mm_block_w8a16_auto", mutates_args=()
    )
    def fp8_emu_mm_block_w8a16_auto(
        x: torch.Tensor,
        w: torch.Tensor,
        w_s: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        block_n: int,
        block_k: int,
        min_m: int,
    ) -> torch.Tensor:
        # Runs eagerly: every branch is on the concrete runtime M.
        from . import gfx906_fp16_prefill

        return gfx906_fp16_prefill.materialize_or_tile(
            x, w, w_s, bias, out_dtype, block_n, block_k, min_m
        )

    @fp8_emu_mm_block_w8a16_auto.register_fake
    def _(
        x: torch.Tensor,
        w: torch.Tensor,
        w_s: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        block_n: int,
        block_k: int,
        min_m: int,
    ) -> torch.Tensor:
        return x.new_empty((*x.shape[:-1], w.shape[0]), dtype=out_dtype)


if ops_available():
    _register_fp8_emu_mm()

if ops_available_block():
    _register_fp8_emu_mm_block()

if ops_available_block_quant():
    _register_fp8_emu_mm_block_quant()

if hasattr(torch.ops._rocm_C, "fp8_emu_gemm_block_w8a16"):
    _register_fp8_emu_mm_block_w8a16()
    _register_fp8_emu_mm_block_w8a16_auto()

_register_fp8_emu_lut()


# Whether this build can serve W8A16.  Computed *after* the custom_op
# registration above (the capability check looks at both for the raw entry
# point and the `vllm::` wrapper), and kept as one constant so the class
# attribute, the per-instance activation-dtype check and is_supported() cannot
# disagree.
_W8A16_ENGAGED = _W8A16 and ops_available_block_w8a16()

# The route engages silently by design ("auto" never warns); without a positive
# line in the log an operator cannot tell the W8A16 and W8A8 paths apart from
# serving output alone (P0 retest, 2026-10-21).
logger.info(
    "gfx906 fp8-emu route: VLLM_DCU_FP8_W8A16=%s, W8A16 engaged=%s "
    "(built op %s), GEMV_MAX_M=%d.",
    _W8A16_MODE, _W8A16_ENGAGED, ops_available_block_w8a16(), _GEMV_MAX_M,
)


def _w8a16_applies(input_dtype: torch.dtype) -> bool:
    """Whether a layer with this activation dtype takes the W8A16 route.

    gfx906 has no bf16 datapath, so only fp16 activations may skip the
    activation quantization; any other dtype keeps the shipped W8A8 route (and
    the explicit ``=1`` mode refuses the layer instead of silently changing
    tokens).
    """
    return _W8A16_ENGAGED and input_dtype == torch.float16


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
        # Compiling must stay on the custom_op: it is what keeps Dynamo from
        # pinning the token dimension (S11k).
        if _EMU_FASTPATH and not torch.compiler.is_compiling():
            out = _fp8_emu_mm_impl(A, w, a_s, w_s, lut, bias, out_dtype)
        else:
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

    # S23 O5: the fused kernel quantizes the activation itself, so ``apply_weights``
    # must not quantize in python on this path.  Gated on the op being present in
    # the loaded extension, so the class attribute also tells the base kernel
    # whether the wiring is safe when the artifact and the tree disagree.
    fuses_input_quant = _FUSE_QUANT and ops_available_block_quant()

    # W8A16 polarity: the base-class hook quantizes when True and passes the
    # raw fp16 activation with a 1-element placeholder ``As`` when False
    # (same mechanism FlashInfer/CPU use; P2 -- this kernel must never read
    # that placeholder).  So the hook is False only when the W8A16 route is
    # actually engaged: switch on AND the built op present.  Switch off (or
    # the artifact lacking the op) must stay True -- the W8A8 path, verbatim.
    # Found by the production restart after P1: the inverted polarity made the
    # switch-off path skip quantization and crash on the missing W8A16 op.
    # The build-level answer is narrowed per layer in ``__init__`` (fp16 only).
    apply_input_quant = not _W8A16_ENGAGED

    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)
        # The activation dtype is per layer, the switch/op presence is per
        # build.  A bf16 (or fp32) layer under =auto keeps the shipped W8A8
        # route instead of dying in the first forward; =1 is a hard error.
        if _W8A16_ENGAGED and not _w8a16_applies(config.input_dtype):
            if _W8A16_MODE == "1":
                raise RuntimeError(
                    "VLLM_DCU_FP8_W8A16=1 requires float16 activations, got "
                    f"{config.input_dtype}: gfx906 has no bf16 W8A16 datapath. "
                    "Use VLLM_DCU_FP8_W8A16=auto to keep the W8A8 route for "
                    "this layer."
                )
            logger.warning_once(
                "gfx906 W8A16: %s activations are not supported by the W8A16 "
                "kernel; keeping the W8A8 route for this layer.",
                config.input_dtype,
            )
            self.apply_input_quant = True

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
        # An explicit VLLM_DCU_FP8_W8A16=1 must not silently degrade: an
        # is_supported()==False only falls through to the next kernel (the
        # software Triton block kernel, ~36x slower on gfx906) and would also
        # lose the gfx906 W8A8 path, so this is a hard error instead.
        if _W8A16_MODE == "1" and not _W8A16_ENGAGED:
            raise RuntimeError(
                "VLLM_DCU_FP8_W8A16=1 but the loaded _rocm_C has no W8A16 "
                "entry point (fp8_emu_gemm_block_w8a16 / "
                "vllm::fp8_emu_mm_block_w8a16).  Rebuild the extension, or set "
                "VLLM_DCU_FP8_W8A16=auto/=0 to allow the fallback."
            )
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

    def apply_block_scaled_mm_quantized(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        """Fused-quant variant of :meth:`apply_block_scaled_mm` (S23 O5).

        ``A`` is the unquantized fp16/bf16 activation; the kernel produces the
        activation scales and quantizes internally, so no ``As`` is passed and
        no python-side quantize happens.

        Args:
            A: Unquantized activations, ``[M, K]`` in ``self.config.out_dtype``.
            B: Weight as the checkpoint keeps it (``[N, K]`` or ``[K, N]``).
            Bs: Per-block weight scales, ``[cdiv(N, block_n), cdiv(K, block_k)]``.

        Returns:
            ``[M, N]`` output in ``self.config.out_dtype``.

        """
        K = A.shape[1]
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
        return _fp8_emu_mm_block_quant_impl(
            A,
            w,
            Bs.contiguous(),
            # The LUT widens the *weight* codes; A is still fp16 on this path,
            # so it must be keyed on the weight dtype, not on A.dtype.
            _device_lut(w.dtype, A.device),
            None,
            self.config.out_dtype,
            _GEMV_THREADS,
            _prefill_cfg(A.shape[0], w.shape[0]),
            int(self.weight_group_shape.row),
            int(self.weight_group_shape.col),
            _QUANT_EPS,
        )

    def _apply_block_scaled_mm_w8a16(
        self,
        A: torch.Tensor,
        w: torch.Tensor,
        w_s: torch.Tensor,
        block_n: int,
        block_k: int,
    ) -> torch.Tensor:
        """Issue the W8A16 kernel for ``A`` fp16 ``[M, K]`` and ``w`` fp8.

        Args:
            A: fp16 activations, ``[M, K]``.
            w: fp8 weight, ``[N, K]`` (K-contiguous, row stride may be padded).
            w_s: Per-block weight scales, ``[N // block_n, K // block_k]``.
            block_n: Weight block rows.
            block_k: Weight block columns.

        Returns:
            ``[M, N]`` output in ``self.config.out_dtype``.

        """
        # Traced callers keep everything inside one opaque op (S26): the
        # LUT lookup and the ``A.shape[0] > _GEMV_MAX_M`` cfg choice are both
        # Python-level on a symbolic M, so tracing them bakes the first hint.
        if torch.compiler.is_compiling():
            return torch.ops.vllm.fp8_emu_mm_block_w8a16_auto(
                A,
                w,
                w_s,
                None,
                self.config.out_dtype,
                block_n,
                block_k,
                _FORCE_TILE_MIN_M,
            )
        lut = _device_lut(w.dtype, A.device)
        # The decode arm always passes cfg 0 (the kernel's auto rule picks the
        # exact-M tile); only the M > GEMV arm looks at the prefill knob, so a
        # decode call must not even evaluate it (no env read on the hot path).
        cfg = (
            _prefill_cfg(A.shape[0], w.shape[0])
            if A.shape[0] > _GEMV_MAX_M
            else 0
        )
        # Eager fast path: the custom_op costs 31-55 us per call (S23 O1).
        if _EMU_FASTPATH:
            return _fp8_emu_mm_block_w8a16_impl(
                A, w, w_s, lut, None, self.config.out_dtype, _GEMV_THREADS,
                cfg, block_n, block_k,
            )
        return torch.ops.vllm.fp8_emu_mm_block_w8a16(
            A, w, w_s, lut, None, self.config.out_dtype, _GEMV_THREADS,
            cfg, block_n, block_k,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Prefill fp16 materialization first, then the base-class path.

        Above the router's per-role token threshold the current layer's weight is
        dequantized into one reused fp16 scratch and handed to rocBLAS, which is
        faster than the emulation tile GEMM at large M.  The router only engages
        when this kernel is on the W8A16 route (``apply_input_quant`` False) and
        returns None on every failure, so the fallback is the W8A16 kernel.

        Args:
            layer: The quantized linear layer.
            x: Activation, ``[*leading, K]``.
            bias: Optional ``[N]`` bias.

        Returns:
            ``[*x.shape[:-1], N]`` in the layer's configured out dtype.

        """
        if _PREFILL_ROUTE:
            from . import gfx906_fp16_prefill

            routed = gfx906_fp16_prefill.try_prefill(self, layer, x, bias)
            if routed is not None:
                return routed
        return super().apply_weights(layer, x, bias, **kwargs)

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
        # W8A16 route: the base class skipped the activation quantization, so
        # ``A`` is the raw fp16 activation and ``As`` is a 1-element placeholder
        # that must not be read.  Everything below this branch is the W8A8 path,
        # kept verbatim.
        if not self.apply_input_quant:
            if A.dtype != torch.float16:
                raise RuntimeError(
                    "gfx906 W8A16 requires float16 activations (gfx906 has no "
                    f"bf16 datapath); got {A.dtype}.  Set "
                    "VLLM_DCU_FP8_W8A16=0 to use the W8A8 path."
                )
            return self._apply_block_scaled_mm_w8a16(
                A, w, Bs.contiguous(), block_n, block_k
            )
        lut = _lut_for(A, A.dtype)
        a_s = As.contiguous()
        w_s = Bs.contiguous()
        # Compiling must stay on the custom_op: it is what keeps Dynamo from
        # pinning the token dimension (S11k).
        if _EMU_FASTPATH and not torch.compiler.is_compiling():
            return _fp8_emu_mm_block_impl(
                A, w, a_s, w_s, lut, None, self.config.out_dtype, block_n, block_k
            )
        return torch.ops.vllm.fp8_emu_mm_block(
            A,
            w,
            a_s,
            w_s,
            lut,
            None,
            self.config.out_dtype,
            block_n,
            block_k,
        )
