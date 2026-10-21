# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prefill fp16 weight materialization for the gfx906 FP8 block-scaled kernel.

Above a per-role token threshold it is cheaper to dequantize the *current*
layer's fp8 weight into one reused fp16 scratch and let rocBLAS run
``torch.mm`` than to run the W8A16 emulation tile GEMM.  The fp16 copy never
persists across layers: the scratch is a single grow-only buffer keyed by
device, so the resident cost is one weight, not one weight per layer
(docs/design/z100-w8a16-prefill-fp16-construction.md §4.6).

Contract:

* Single-stream: the buffer is filled and consumed by one CUDA stream.  A call
  from another stream gets ``None`` (fall back to the W8A16 kernel) instead of
  racing on the buffer.
* Eager-only: a call while ``torch.compile`` is tracing returns ``None`` (the
  raw dequant op has no Meta/fake implementation and the M branch is symbolic)
  and the stream-affine scratch is never allocated inside a CUDA-graph capture,
  so a captured graph never contains the route.  ``VLLM_DCU_FP16_PREFILL``
  therefore stays Dynamo-safe.
* Every failure (dtype/layout, OOM, missing op) returns ``None`` so the caller
  falls back to the W8A16 kernel; this module never raises.
* A layer role that is missing from ``_ROLE_MIN_M`` warns once, so a table miss
  is visible instead of silently using ``_DEFAULT_MIN_M``.

The router is only reachable when W8A16 is wired (``apply_input_quant`` is
False, i.e. the activation arrives as fp16 and is not quantized).  The caller
hooks it in ``Gfx906Fp8BlockScaledMMLinearKernel.apply_weights`` before the
base-class quantize -> GEMM path.
"""

import torch

from vllm import envs
from vllm.logger import init_logger

from .BlockScaledMMLinearKernel import FP8BlockParams
from .gfx906_fp8 import _device_lut

logger = init_logger(__name__)

# Read once at import: an A/B arm sets the variable before launch, and the hot
# path then pays a single global load per call (never a getenv).  "0" keeps the
# shipped behaviour, "1" and "auto" both gate on the min-M table below.
_MODE = str(envs.VLLM_DCU_FP16_PREFILL).strip().lower()
if _MODE not in ("0", "1", "auto"):
    logger.warning(
        "gfx906 fp16 prefill: unknown VLLM_DCU_FP16_PREFILL=%r; disabling.",
        _MODE,
    )
    _MODE = "0"

# Per-role token threshold: the M below which the W8A16 tile kernel serves the
# call instead of the fp16 materialization.  Filled from the P1 crossover sweep
# (.s24/probes/w8a16_min_m_sweep.py, gfx906 @ 2026-10-20): the threshold is
# where dequant + rocBLAS becomes strictly faster than the shipped W8A8 layer
# (the tile sits between the two).  Wide-N shapes pay a larger fixed dequant
# cost, so gate_up crosses later; deep-K down_proj wins from the floor.
#
# ``in_proj_qkvz`` is the fused GDN projection that Qwen3.5-family models
# register at runtime (``...linear_attn.in_proj_qkvz``, the largest FP8 layer;
# the split ``in_proj_qkv`` / ``in_proj_z`` names are aliased for the
# gguf/compressed-tensors variants).  Missing it sent 48/64 layers of the
# served model to the 4096 default, i.e. the W8A16 tile -- measured 1.26-1.30x
# slower than the shipped W8A8 path at M >= 512 on the served 4096:5120
# per-rank shape, while the route/tile ratio there is already 0.685 at M = 16
# (.s23/logs/w8a16-p1-perf.log).
_ROLE_MIN_M: dict[str, int | None] = {
    "gate_up_proj": 128,
    "down_proj": 16,
    "qkv_proj": 128,
    "in_proj_qkvz": 16,
    # Split-projection variants: qkv is the same wide-N family as qkv_proj,
    # z is narrow-N/deep-K like out_proj.  Extrapolated until swept.
    "in_proj_qkv": 128,
    "in_proj_z": 16,
    "o_proj": 16,
    "out_proj": 16,
}
_DEFAULT_MIN_M = 4096
_ENV_MIN_M = envs.VLLM_DCU_FP16_PREFILL_MIN_M
if _ENV_MIN_M > 0:
    _DEFAULT_MIN_M = _ENV_MIN_M

logger.info(
    "gfx906 fp16 prefill materialization: VLLM_DCU_FP16_PREFILL=%s, "
    "role min_m %s, default %d.",
    envs.VLLM_DCU_FP16_PREFILL, _ROLE_MIN_M, _DEFAULT_MIN_M,
)

# Minimum workspace size in bytes (0 = size it from the weight); a bigger layer
# still grows the buffer.  Only useful with eager execution: under CUDA-graph
# capture the stream-affine guard declines the route regardless (see the module
# contract), so this cannot pre-warm a graph.
_WS_BYTES = envs.VLLM_DCU_FP16_WS_BYTES

# (device_index, dtype) -> fp16 scratch, plus the stream that filled it.  The
# two maps are separate so ``_WS`` keeps the plain tensor values the contract
# documents while the single-stream guard can still see the owner.
_WS: dict[tuple, torch.Tensor] = {}
_WS_STREAM: dict[tuple, int | None] = {}


_DEQUANT_OP: bool | None = None


def _enabled() -> bool:
    """Whether the router may engage at all (switch on and dequant op built).

    The op lookup is cached after the first call: the hook runs for every
    linear layer of every step while the switch is on, and the extension is
    loaded by the time ``apply_weights`` calls it.
    """
    global _DEQUANT_OP
    if _DEQUANT_OP is None:
        _DEQUANT_OP = hasattr(torch.ops._rocm_C, "fp8_dequant_to_fp16_out")
    return _MODE != "0" and _DEQUANT_OP


def _role(layer: torch.nn.Module) -> str:
    """Linear-layer role, e.g. ``down_proj`` from ``...mlp.down_proj``."""
    return (getattr(layer, "prefix", "") or "").rsplit(".", 1)[-1]


# "1" means "force": the route engages as soon as the tile path is meaningful
# (M >= 16), regardless of the per-role table.  "auto" follows the table.
_FORCE_MIN_M = 16

# Set once the first call actually goes through dequant + rocBLAS; backs the
# one-time "route engaged" INFO line.
_ROUTE_SEEN = False


def _min_m(role: str, _w: torch.Tensor | None) -> int:
    """Token threshold for ``role``; unknown roles use the default."""
    if _MODE == "1":
        return _FORCE_MIN_M
    value = _ROLE_MIN_M.get(role)
    return int(value) if value is not None else _DEFAULT_MIN_M


_UNKNOWN_ROLES_WARNED: set[str] = set()


def _warn_unknown_role(role: str) -> None:
    """Warn once per unknown role, so a missing table entry cannot hide."""
    if role in _UNKNOWN_ROLES_WARNED:
        return
    _UNKNOWN_ROLES_WARNED.add(role)
    logger.warning(
        "gfx906 fp16 prefill: no measured min_m for layer role %r; using "
        "_DEFAULT_MIN_M=%d.  Add the role to _ROLE_MIN_M (or set "
        "VLLM_DCU_FP16_PREFILL_MIN_M) if it should route earlier.",
        role,
        _DEFAULT_MIN_M,
    )


def _workspace(w: torch.Tensor) -> torch.Tensor | None:
    """Return the grow-only fp16 scratch for ``w``, or None to fall back.

    Args:
        w: The layer's fp8 weight, ``[N, K]``; only its shape/device are used.

    Returns:
        A flat fp16 buffer with at least ``w.numel()`` elements, or None when
        the buffer would have to be allocated during graph capture, was filled
        by a different stream, or does not fit in memory.

    """
    key = (w.device.index, torch.float16)
    need = w.numel()
    if _WS_BYTES > 0:
        need = max(need, _WS_BYTES // 2)
    stream = torch.cuda.current_stream(w.device).cuda_stream
    ws = _WS.get(key)
    if ws is not None:
        if _WS_STREAM.get(key) != stream:
            # Filling from a second stream would race the first stream's next
            # layer; the buffer is single-stream by contract.
            logger.warning_once(
                "gfx906 fp16 prefill: workspace belongs to another CUDA "
                "stream; using the W8A16 kernel"
            )
            return None
        if ws.numel() >= need:
            return ws
    if torch.cuda.is_current_stream_capturing():
        logger.warning_once(
            "gfx906 fp16 prefill: refusing to allocate the workspace during "
            "CUDA graph capture; using the W8A16 kernel"
        )
        return None
    try:
        ws = torch.empty(need, dtype=torch.float16, device=w.device)
    except torch.OutOfMemoryError:
        # Never propagate: the W8A16 kernel is the documented fallback.
        logger.warning_once(
            "gfx906 fp16 prefill: workspace OOM; using the W8A16 kernel"
        )
        return None
    _WS[key] = ws
    _WS_STREAM[key] = stream
    return ws


def try_prefill(
    kernel: torch.nn.Module,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    """Route one linear call to fp16 weight materialization plus rocBLAS.

    Dequantizes the layer weight into the shared scratch (``W16`` stays
    ``[N, K]``, row order unchanged) and computes ``x @ W16.T``.  The bias is
    added in the matmul dtype, before the final cast to the configured output
    dtype, matching the base-class bias order.

    Args:
        kernel: The block-scaled kernel instance.  Must have
            ``apply_input_quant`` False (W8A16 wired) and expose
            ``weight_group_shape`` and ``config.out_dtype``.
        layer: The quantized linear layer holding ``weight`` and the block
            scale (``weight_scale_inv`` or ``weight_scale``).
        x: Activation, ``[*leading, K]`` float16 and contiguous.
        bias: Optional ``[N]`` bias; added after the matmul.

    Returns:
        The ``[*x.shape[:-1], N]`` output in ``kernel.config.out_dtype``, or
        None when the caller must fall back to the W8A16 kernel.

    """
    if torch.compiler.is_compiling():
        # Dynamo-safe fallback: ``fp8_dequant_to_fp16_out`` has no Meta/fake
        # implementation and this function branches on a symbolic M, so the
        # traced route is the W8A16 custom_op, not this router.
        return None
    if not _enabled() or getattr(kernel, "apply_input_quant", True):
        return None
    try:
        return _try_prefill(kernel, layer, x, bias)
    except Exception:  # noqa: BLE001 - the fallback contract is "never raise"
        logger.warning_once("gfx906 fp16 prefill: route failed; using the W8A16 kernel")
        return None


def _try_prefill(
    kernel: torch.nn.Module,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    """Body of :func:`try_prefill`; preconditions already checked."""
    if x.dtype != torch.float16 or not x.is_contiguous():
        return None
    # Cheap gates first: while VLLM_DCU_FP16_PREFILL is on this runs for every
    # linear layer of every step, and decode (M <= 8) never routes.
    K = x.shape[-1]
    m = x.numel() // K
    role = _role(layer)
    if role not in _ROLE_MIN_M:
        _warn_unknown_role(role)
    if m < _min_m(role, None) or m < _FORCE_MIN_M:
        return None
    params = FP8BlockParams.from_layer(layer)
    w = params.weight
    w_s = params.block_scale
    if w is None or w.element_size() != 1 or w_s.dtype != torch.float32:
        return None
    if w.dim() != 2 or w_s.dim() != 2 or not w_s.is_contiguous():
        return None
    N = w.shape[0]
    if w.shape[1] != K or w.stride(1) != 1:
        return None
    if K % 8 or w.stride(0) % 8:
        return None
    block_n = int(kernel.weight_group_shape.row)
    block_k = int(kernel.weight_group_shape.col)
    if block_n <= 0 or block_k <= 0 or K % block_k:
        return None
    if tuple(w_s.shape) != ((N + block_n - 1) // block_n, K // block_k):
        return None
    ws = _workspace(w)
    if ws is None:
        return None
    # W16 is [N, K] so hipBLAS eats `W16.T()` as a zero-copy transpose view and
    # the dequant kernel keeps the checkpoint's row order (P12).
    w16 = ws[: w.numel()].view(w.shape)
    torch.ops._rocm_C.fp8_dequant_to_fp16_out(
        w16,
        w,
        w_s,
        _device_lut(w.dtype, w.device),
        block_n,
        block_k,
    )
    out = torch.mm(x.view(m, K), w16.t())
    global _ROUTE_SEEN
    if not _ROUTE_SEEN:
        _ROUTE_SEEN = True
        logger.info(
            "gfx906 fp16 prefill: first materialization route engaged "
            "(role=%s, M=%d, N=%d, K=%d); further routes are silent.",
            role, m, N, K,
        )
    # Contract §1.2: the bias is added in out_dtype, after the output cast --
    # the same order as the emu kernel's epilogue (matters only when out_dtype
    # is fp32; production fp16 makes the cast a no-op).
    out = out.to(dtype=kernel.config.out_dtype)
    if bias is not None:
        out = out + bias.to(dtype=out.dtype)
    return out.view(*x.shape[:-1], N)
