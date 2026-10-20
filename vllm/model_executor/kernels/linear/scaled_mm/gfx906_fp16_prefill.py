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
* No allocation while a stream is capturing: a capture-time allocation would
  bake a stale pointer into the graph, so the router declines instead.
* Every failure (dtype/layout, OOM, missing op) returns ``None`` so the caller
  falls back to the W8A16 kernel; this module never raises.

The router is only reachable when W8A16 is wired (``apply_input_quant`` is
False, i.e. the activation arrives as fp16 and is not quantized).  The caller
hooks it in ``Gfx906Fp8BlockScaledMMLinearKernel.apply_weights`` before the
base-class quantize -> GEMM path.
"""

import torch

from vllm import envs
from vllm.logger import init_logger

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

# Per-role token threshold: None means "use _DEFAULT_MIN_M".  Placeholder until
# P1 measures the crossover; keeping the table the only tuning knob lets every
# promotion/rollback be a one-line change.
_ROLE_MIN_M: dict[str, int | None] = {
    "gate_up_proj": None,
    "down_proj": None,
    "qkv_proj": None,
    "o_proj": None,
    "out_proj": None,
}
_DEFAULT_MIN_M = 4096
_ENV_MIN_M = envs.VLLM_DCU_FP16_PREFILL_MIN_M
if _ENV_MIN_M > 0:
    _DEFAULT_MIN_M = _ENV_MIN_M

# Fixed workspace size in bytes (0 = size it from the weight).  Setting it lets
# the buffer exist before graph capture warms up.
_WS_BYTES = envs.VLLM_DCU_FP16_WS_BYTES

# (device_index, dtype) -> fp16 scratch, plus the stream that filled it.  The
# two maps are separate so ``_WS`` keeps the plain tensor values the contract
# documents while the single-stream guard can still see the owner.
_WS: dict[tuple, torch.Tensor] = {}
_WS_STREAM: dict[tuple, int | None] = {}


def _enabled() -> bool:
    """Whether the router may engage at all (switch on and dequant op built)."""
    return _MODE != "0" and hasattr(torch.ops._rocm_C, "fp8_dequant_to_fp16_out")


def _role(layer: torch.nn.Module) -> str:
    """Linear-layer role, e.g. ``down_proj`` from ``...mlp.down_proj``."""
    return (getattr(layer, "prefix", "") or "").rsplit(".", 1)[-1]


# "1" means "force": the route engages as soon as the tile path is meaningful
# (M >= 16), regardless of the per-role table.  "auto" follows the table.
_FORCE_MIN_M = 16


def _min_m(role: str, _w: torch.Tensor | None) -> int:
    """Token threshold for ``role``; unknown roles use the default."""
    if _MODE == "1":
        return _FORCE_MIN_M
    value = _ROLE_MIN_M.get(role)
    return int(value) if value is not None else _DEFAULT_MIN_M


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
    from .BlockScaledMMLinearKernel import FP8BlockParams
    from .gfx906_fp8 import _device_lut

    if x.dtype != torch.float16 or not x.is_contiguous():
        return None
    params = FP8BlockParams.from_layer(layer)
    w = params.weight
    w_s = params.block_scale
    if w is None or w.element_size() != 1 or w_s.dtype != torch.float32:
        return None
    if w_s.dim() != 2 or not w_s.is_contiguous():
        return None
    K = x.shape[-1]
    N = w.shape[0]
    if w.dim() != 2 or w.shape[1] != K or w.stride(1) != 1:
        return None
    if K % 8 or w.stride(0) % 8:
        return None
    block_n = int(kernel.weight_group_shape.row)
    block_k = int(kernel.weight_group_shape.col)
    if block_n <= 0 or block_k <= 0 or K % block_k:
        return None
    if tuple(w_s.shape) != ((N + block_n - 1) // block_n, K // block_k):
        return None
    m = x.numel() // K
    min_m = _min_m(_role(layer), w)
    if m < min_m or m < 16:
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
    # Contract §1.2: the bias is added in out_dtype, after the output cast --
    # the same order as the emu kernel's epilogue (matters only when out_dtype
    # is fp32; production fp16 makes the cast a no-op).
    out = out.to(dtype=kernel.config.out_dtype)
    if bias is not None:
        out = out + bias.to(dtype=out.dtype)
    return out.view(*x.shape[:-1], N)
