# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""W8A8 INT8 scaled-MM kernel for gfx906 (Z100) using `v_dot4_i32_i8`.

The device code lives in ``csrc/rocm/w8a8_sdot4.cu`` and is registered as
``torch.ops._rocm_C.w8a8_sdot4_gemv`` / ``..._gemm``.  It was ported from
the author's earlier out-of-tree ``int8_sdot4`` project (Apache-2.0); see the
provenance note in the .cu file.

Design notes:

* Weights stay ``[N, K]`` int8 exactly as loaded -- not transposed, not
  dequantized.  That differs from ``TritonInt8ScaledMMLinearKernel`` (which
  transposes to ``[K, N]``) and keeps the resident weight footprint identical
  to the checkpoint.
* M <= ``_GEMV_MAX_M`` (decode) takes the GEMV path; larger M takes the tiled
  GEMM path.  Both accumulate in int32 and apply ``x_s``/``w_s`` only in the
  epilogue, in the same order as ``triton_scaled_mm``.
* Rollback / A-B switch: ``VLLM_DCU_USE_INT8_SDOT4=1`` opts in (default is off
  until the compile integration is proven end to end).  When disabled, or when
  the shape/quantization form is not covered, the ROCm kernel table falls
  through to ``TritonInt8ScaledMMLinearKernel``.
* ``VLLM_DCU_SDOT4_INNER=triton`` is a *diagnostic* override (S11l): it keeps
  the opaque ``custom_op`` in the graph but runs ``triton_scaled_mm`` inside
  its body.  It exists only to separate "the opaque op changed the compiled
  graph" from "the sdot4 kernel math changed the numbers" when attributing
  greedy-token flips; it is not a deployment switch.

``torch.compile`` integration (S11k): the raw ``_rocm_C::w8a8_sdot4_*`` ops are
opaque to Dynamo, which -- at the call site -- pins the symbolic token
dimension with an ``Eq(dim, hint)`` guard and turns it into a constant
(``range_refined_to_singleton``).  vLLM marks that dimension dynamic, so guard
building then fails with ``ConstraintViolationError`` during the profile run.
``torch.library.custom_op`` gives Dynamo an explicit shape contract, so the op
stays an opaque in-graph extern call instead of leaking a specialization.  The
GEMV/GEMM choice also lives *inside* that opaque body: it is a Python-level
branch on the runtime M, and keeping it out of the traced region avoids baking
a symbolic ``M <= 8`` guard into the compiled graph.
"""

import os

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    triton_scaled_mm,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx906

from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)

logger = init_logger(__name__)

# M <= _GEMV_MAX_M goes to the GEMV kernel (must match kMsMax in the .cu).
_GEMV_MAX_M = 8
# GEMV threads per block: tuned value from the source project (power of two).
_GEMV_THREADS = 64


def ops_available() -> bool:
    return hasattr(torch.ops._rocm_C, "w8a8_sdot4_gemm") and hasattr(
        torch.ops._rocm_C, "w8a8_sdot4_gemv"
    )


_SDOT4_MM_OP = "vllm::w8a8_sdot4_mm"

# Diagnostic only (S11l): keep the opaque `custom_op` / compiled-graph shape but
# run `triton_scaled_mm` inside its body, so that "opaque(triton)" can be told
# apart from the baseline "bare triton" (`=0`, where the call is traced into the
# graph).  Not a product switch and not part of the int8 kernel selection; the
# default value keeps the sdot4 kernels.
_SDOT4_INNER = os.environ.get("VLLM_DCU_SDOT4_INNER", "sdot4").strip().lower()


def _register_sdot4_mm() -> None:
    """Expose the raw kernels as a ``torch.library.custom_op`` (once).

    Dynamo treats a raw ``TORCH_LIBRARY`` op as opaque *without* a shape
    contract: it emits an ``Eq(size, hint)`` guard for every symbolic argument
    dimension at the call site, which specializes the dynamically-marked token
    dimension and aborts compilation (see the module docstring).  Declaring the
    op through ``torch.library`` (schema + ``register_fake``) removes that
    specialization while keeping the kernel inside the compiled graph.
    """
    if hasattr(torch.ops.vllm, "w8a8_sdot4_mm"):
        return

    @torch.library.custom_op(_SDOT4_MM_OP, mutates_args=())
    def w8a8_sdot4_mm(
        x_q: torch.Tensor,
        w: torch.Tensor,
        x_s: torch.Tensor,
        w_s: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        # Runs eagerly: the branch is on the concrete runtime M, never traced.
        if _SDOT4_INNER == "triton":
            return triton_scaled_mm(
                x_q,
                w.t(),
                scale_a=x_s,
                scale_b=w_s,
                out_dtype=out_dtype,
                bias=bias,
            )
        if x_q.shape[0] <= _GEMV_MAX_M:
            return torch.ops._rocm_C.w8a8_sdot4_gemv(
                x_q, w, x_s, w_s, bias, out_dtype, _GEMV_THREADS
            )
        return torch.ops._rocm_C.w8a8_sdot4_gemm(
            x_q, w, x_s, w_s, bias, out_dtype, 0
        )

    @w8a8_sdot4_mm.register_fake
    def _(
        x_q: torch.Tensor,
        w: torch.Tensor,
        x_s: torch.Tensor,
        w_s: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        return x_q.new_empty((x_q.shape[0], w.shape[0]), dtype=out_dtype)


if ops_available():
    _register_sdot4_mm()


class Gfx906Sdot4Int8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm."
        if not on_gfx906():
            return False, "requires gfx906 (V_DOT4_I32_I8)."
        if not envs.VLLM_DCU_USE_INT8_SDOT4:
            return False, "disabled by VLLM_DCU_USE_INT8_SDOT4=0."
        if not ops_available():
            return False, "torch.ops._rocm_C.w8a8_sdot4_* is not registered."
        if not hasattr(torch.ops.vllm, "w8a8_sdot4_mm"):
            return False, "torch.ops.vllm.w8a8_sdot4_mm is not registered."
        return True, None

    @classmethod
    def can_implement(
        cls, c: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return False, "supports symmetric input quantization only."
        if c.is_static_input_scheme:
            return False, "supports dynamic per-token input quantization only."
        if not c.is_channelwise:
            return False, "requires per-channel weight scales."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names

        # Keep the native [N, K] layout; only make it contiguous.
        w = getattr(layer, w_q_name)
        if not w.is_contiguous():
            w = w.contiguous()
            replace_parameter(
                layer, w_q_name, torch.nn.Parameter(w, requires_grad=False)
            )

        # Per-channel weight scale, flattened to [N] float32.
        w_s = getattr(layer, w_s_name)
        logical_widths = getattr(layer, "logical_widths", None)
        if (
            logical_widths is not None
            and len(logical_widths) > 1
            and not self.config.is_channelwise
        ):
            w_s = convert_to_channelwise(w_s, logical_widths)
        w_s = w_s.reshape(-1).to(torch.float32).contiguous()
        if w_s.numel() != w.shape[0]:
            # can_implement() already rules this out for the supported configs.
            raise RuntimeError(
                f"Gfx906Sdot4Int8ScaledMMLinearKernel: weight_scale has "
                f"{w_s.numel()} elements but N={w.shape[0]}"
            )
        replace_parameter(layer, w_s_name, torch.nn.Parameter(w_s, requires_grad=False))

        # Dynamic symmetric quantization: no input scale / zero point / azp
        # term is needed (x_s is computed per token in apply_weights).
        setattr(layer, i_s_name, None)
        setattr(layer, i_zp_name, None)
        setattr(layer, azp_adj_name, None)

    def _triton_fallback(
        self,
        x_q: torch.Tensor,
        x_s: torch.Tensor,
        w: torch.Tensor,
        w_s: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        # Triton wants [K, N]; `.t()` is a view, so this costs nothing.
        return triton_scaled_mm(
            x_q, w.t(), scale_a=x_s, scale_b=w_s, out_dtype=out_dtype, bias=bias
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w, w_s, _i_s, _i_zp, _azp = self._get_layer_params(layer)
        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = (*x.shape[:-1], w.shape[0])

        k = w.shape[1]
        supported = (
            w.dtype == torch.int8
            and w_s.dtype == torch.float32
            and k == x_2d.shape[1]
            and k % 4 == 0
            and x_2d.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and x_2d.shape[0] >= 1
        )
        if not supported:
            x_q, x_s, _ = ops.scaled_int8_quant(
                x_2d.contiguous(), None, None, symmetric=True
            )
            return self._triton_fallback(
                x_q, x_s.reshape(-1), w, w_s, bias, x.dtype
            ).view(out_shape)

        x_q, x_s, _ = ops.scaled_int8_quant(
            x_2d.contiguous(), None, None, symmetric=True
        )
        x_s = x_s.reshape(-1)

        # One opaque op call: GEMV (M <= 8) vs tiled GEMM is decided inside the
        # custom op body, so the traced graph carries no symbolic M branch.
        out = torch.ops.vllm.w8a8_sdot4_mm(x_q, w, x_s, w_s, bias, x.dtype)
        return out.view(out_shape)