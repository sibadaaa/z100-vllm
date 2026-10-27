# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Self

import torch

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_fast_ok,
    per_token_group_quant_fp8_shape_ok,
    process_fp8_weight_block_strategy,
)
from vllm.model_executor.utils import replace_parameter

from ..base import (
    FP8Params,
    MMLinearKernel,
)
from .ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig


@dataclass
class FP8BlockParams(FP8Params):
    weight_scale_inv: torch.Tensor | None
    weight_scale: torch.Tensor | None

    WEIGHT_SCALE_INV: ClassVar[str] = "weight_scale_inv"

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> Self:
        return cls(
            weight=getattr(layer, cls.WEIGHT),
            weight_scale_inv=getattr(layer, cls.WEIGHT_SCALE_INV, None),
            weight_scale=getattr(layer, cls.WEIGHT_SCALE, None),
            input_scale=getattr(layer, cls.INPUT_SCALE, None),
            input_scale_ub=getattr(layer, cls.INPUT_SCALE_UB, None),
        )

    @property
    def block_scale_attr(self) -> str:
        """Fp8LinearMethod registers the block scale as ``weight_scale_inv``,
        compressed-tensors as ``weight_scale``."""
        return (
            self.WEIGHT_SCALE
            if self.weight_scale_inv is None
            else self.WEIGHT_SCALE_INV
        )

    @property
    def block_scale(self) -> torch.Tensor:
        scale = getattr(self, self.block_scale_attr)
        assert scale is not None
        return scale


class Fp8BlockScaledMMLinearKernel(
    MMLinearKernel[FP8ScaledMMLinearLayerConfig, FP8BlockParams], ABC
):
    # Set to False in subclasses that accept BF16 input directly (e.g. FlashInfer)
    # and therefore do not need the input quantization step in apply_weights.
    apply_input_quant: ClassVar[bool] = True
    # Set True only by kernels whose GEMM quantizes the fp16 activation itself
    # (S23 O5).  Kept False on the base class so a subclass that does not
    # override it keeps the quantize-then-GEMM path.
    fuses_input_quant: ClassVar[bool] = False

    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)
        act_scale_descriptor = config.activation_quant_key.scale
        self.weight_group_shape = config.weight_quant_key.scale.group_shape
        self.quant_fp8 = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
            use_ue8m0=False,
        )
        # Decided once: the quantizer configuration cannot change after init, so
        # apply_weights pays only a bool load (S23 O2).  Subclasses that replace
        # self.quant_fp8 (Cutlass/torch use column-major scales) fail the
        # identity check in apply_weights and keep the normal path.
        self._quant_fastpath = (
            type(self.quant_fp8) is QuantFP8
            and not self.quant_fp8.use_aiter
            and self.quant_fp8.is_group_quant
            and not self.quant_fp8.static
            and per_token_group_quant_fp8_fast_ok(
                self.quant_fp8.group_size,
                column_major_scales=self.quant_fp8.column_major_scales,
                tma_aligned_scales=self.quant_fp8.tma_aligned_scales,
                use_ue8m0=self.quant_fp8.use_ue8m0,
            )
        )
        self._quant_fastpath_src = self.quant_fp8
        # S23 O5: decided once, same reason as the O2 flag above.  Requires the
        # kernel to quantize internally AND a quantizer configuration that the
        # fused kernel reproduces bit-for-bit (plain 128-group, row-major,
        # plain-fp8, dynamic).
        self._fuse_quant = (
            self.fuses_input_quant
            and type(self.quant_fp8) is QuantFP8
            and not self.quant_fp8.use_aiter
            and self.quant_fp8.is_group_quant
            and not self.quant_fp8.static
            and per_token_group_quant_fp8_shape_ok(
                self.quant_fp8.group_size,
                column_major_scales=self.quant_fp8.column_major_scales,
                tma_aligned_scales=self.quant_fp8.tma_aligned_scales,
                use_ue8m0=self.quant_fp8.use_ue8m0,
            )
        )
        self._fuse_quant_src = self.quant_fp8
        self.use_triton = False

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        act_quant_key = config.activation_quant_key
        if act_quant_key.scale.static:
            return (
                False,
                "Only dynamic per token group activation quantization is supported.",
            )

        return True, None

    def _get_layer_params(self, layer: torch.nn.Module, **kwargs) -> FP8BlockParams:
        return FP8BlockParams.from_layer(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        params = self._get_layer_params(layer)
        new_weight, new_weight_scale = process_fp8_weight_block_strategy(
            params.weight,
            params.block_scale,
        )

        replace_parameter(layer, params.WEIGHT, new_weight.data)
        replace_parameter(layer, params.block_scale_attr, new_weight_scale.data)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        params = self._get_layer_params(layer)
        weight = params.weight
        weight_scale = params.block_scale
        input_scale = params.input_scale
        scale_up = params.input_scale_ub

        # View input as 2D matrix for fp8 methods
        input_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], weight.shape[0]]

        fused = False
        if self.apply_input_quant:
            # S23 O5: the fused kernel quantizes the activation itself, so the
            # whole python-side quant stage disappears and the step issues one
            # kernel instead of two.  Config-keyed (never M-keyed): the decode
            # step is M=8 and must hit this path, and the kernel selects the
            # stream/tile route internally.
            if (
                self._fuse_quant
                and self._fuse_quant_src is self.quant_fp8
                and input_scale is None
                and scale_up is None
                and not torch.compiler.is_compiling()
            ):
                output = self.apply_block_scaled_mm_quantized(
                    input_2d, weight, weight_scale
                )
                fused = True
            # S23 O2: with the whole 128-group configuration matching, the
            # quantizer kernel is called directly.  The CustomOp chain
            # (__call__ -> forward_hip -> forward_cuda -> fp8_utils) only
            # re-derives this configuration per call: ~8 us of the ~43 us,
            # 256 calls per decode step.  Compiling keeps the normal path.
            elif (
                self._quant_fastpath
                and self._quant_fastpath_src is self.quant_fp8
                and input_scale is None
                and scale_up is None
                and not torch.compiler.is_compiling()
            ):
                q_input, input_scale = per_token_group_quant_fp8(
                    input_2d, self.quant_fp8.group_size
                )
            else:
                q_input, input_scale = self.quant_fp8(
                    input_2d, input_scale, scale_up, use_triton=self.use_triton
                )
        else:
            q_input = input_2d
            # Provide a concrete placeholder so apply_block_scaled_mm args are
            # always Tensors. Subclasses with apply_input_quant=False must not
            # use As in apply_block_scaled_mm.
            input_scale = (
                input_scale if input_scale is not None else input_2d.new_empty(1)
            )

        if not fused:
            output = self.apply_block_scaled_mm(
                A=q_input,
                B=weight,
                As=input_scale,
                Bs=weight_scale,
            )

        if bias is not None:
            output = output + bias
        return output.to(dtype=out_dtype).view(*output_shape)

    @abstractmethod
    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def apply_block_scaled_mm_quantized(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        """Fused-quant variant of :meth:`apply_block_scaled_mm` (S23 O5).

        Only called when ``fuses_input_quant`` is True and the fused op exists;
        a kernel that sets the flag must override this, and the default raises
        so a mis-wired subclass fails loudly instead of silently skipping the
        quantize.
        """
        raise NotImplementedError(
            f"{type(self).__name__} sets fuses_input_quant but does not "
            "implement apply_block_scaled_mm_quantized."
        )


class Fp8BlockScaledDynamicMMLinearKernel(Fp8BlockScaledMMLinearKernel, ABC):
    """Dynamic FP8 block-scaled kernel that dispatches at runtime.

    Extends Fp8BlockScaledMMLinearKernel to inherit apply_weights and overrides
    apply_block_scaled_mm to dispatch between two sub-kernels using torch.cond.

    Subclasses must define:
        base_type:     The primary kernel class.
        fallback_type: The fallback kernel class.
    """

    base_type: ClassVar[type[Fp8BlockScaledMMLinearKernel]]
    fallback_type: ClassVar[type[Fp8BlockScaledMMLinearKernel]]

    def __init__(self, config: "FP8ScaledMMLinearLayerConfig") -> None:
        super().__init__(config)
        self.base = self.base_type(config)
        self.fallback = self.fallback_type(config)

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        is_base_supported, reason_1 = cls.base_type.is_supported(compute_capability)
        is_fallback_supported, reason_2 = cls.fallback_type.is_supported(
            compute_capability
        )
        if is_base_supported and is_fallback_supported:
            return True, None
        if not is_base_supported and not is_fallback_supported:
            return (
                False,
                f"base is not supported due to {reason_1}; "
                f"fallback is not supported due to {reason_2}",
            )
        if not is_base_supported:
            return False, f"base is not supported due to {reason_1}"
        return False, f"fallback is not supported due to {reason_2}"

    @classmethod
    def can_implement(
        cls, config: "FP8ScaledMMLinearLayerConfig"
    ) -> tuple[bool, str | None]:
        can_implement_base, reason_1 = cls.base_type.can_implement(config)
        can_implement_fallback, reason_2 = cls.fallback_type.can_implement(config)
        if can_implement_base and can_implement_fallback:
            return True, None
        if not can_implement_base and not can_implement_fallback:
            return (
                False,
                f"base cannot implement due to {reason_1}; "
                f"fallback cannot implement due to {reason_2}",
            )
        if not can_implement_base:
            return False, f"base cannot implement due to {reason_1}"
        return False, f"fallback cannot implement due to {reason_2}"
