# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S8 quantization-path probe on gfx906 (Hygon Z100 / DTK 26.04).

Answers "which kernel does vLLM actually pick for each quantization scheme on
this arch, and if it cannot, does it fail loudly?" without loading a model.

    python tools/dcu/s8_quant_probe.py

Each section prints the concrete selector output; nothing is monkey-patched, so
the result is the same decision the engine makes at model-load time.
"""

from __future__ import annotations

import sys
import traceback


def _show(label: str, fn) -> None:
    try:
        print(f"{label} -> {fn()}")
    except Exception as exc:
        print(f"{label} -> RAISED {type(exc).__name__}: {exc}")
        tb = traceback.format_exc().strip().splitlines()
        for line in tb[-6:]:
            print(f"    | {line}")


def main() -> int:
    import torch

    from vllm import envs
    from vllm.model_executor.kernels.linear import (
        _POSSIBLE_FP8_BLOCK_KERNELS,
        _POSSIBLE_FP8_KERNELS,
        _POSSIBLE_INT8_KERNELS,
        _POSSIBLE_KERNELS,
        choose_mp_linear_kernel,
        choose_scaled_mm_linear_kernel,
        init_int8_linear_kernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision import (
        MPLinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig as RawMPLinearLayerConfig,
    )
    from vllm.platforms import current_platform
    from vllm.platforms import PlatformEnum
    from vllm.scalar_type import scalar_types

    print("=== 0. context ===")
    print(f"platform          = {current_platform._enum}")
    print(f"device capability = {current_platform.get_device_capability()}")

    print("\n=== 1. W8A8 INT8 (production scheme) ===")
    print(f"_POSSIBLE_INT8_KERNELS[ROCM] = {[k.__name__ for k in _POSSIBLE_INT8_KERNELS[PlatformEnum.ROCM]]}")
    for k in _POSSIBLE_INT8_KERNELS[PlatformEnum.ROCM]:
        _show(f"  is_supported({k.__name__})", k.is_supported)
    for desc, args in (
        ("per-tensor sym (channelwise=False, static=True)", (False, True, True, "probe.pt")),
        ("per-channel sym (channelwise=True, static=True)", (True, True, True, "probe.pc")),
        ("per-tensor sym, dynamic act", (False, False, True, "probe.dyn")),
        ("per-tensor asym input", (False, True, False, "probe.asym")),
    ):
        _show(
            f"init_int8_linear_kernel[{desc}]",
            lambda a=args: type(init_int8_linear_kernel(*a)).__name__,
        )

    print("\n=== 2. W4A16 (AWQ / GPTQ / compressed-tensors wNa16) ===")
    print(
        "_POSSIBLE_KERNELS[ROCM] = "
        f"{[k.__name__ for k in _POSSIBLE_KERNELS[PlatformEnum.ROCM]]}"
    )
    for k in _POSSIBLE_KERNELS[PlatformEnum.ROCM]:
        _show(
            f"  can_implement({k.__name__}, uint4b8 g128)",
            lambda kk=k: kk.can_implement(
                MPLinearLayerConfig(
                    full_weight_shape=(1024, 1024),
                    partition_weight_shape=(1024, 1024),
                    weight_type=scalar_types.uint4b8,
                    act_type=torch.float16,
                    group_size=128,
                    zero_points=False,
                )
            ),
        )
    _show(
        "choose_mp_linear_kernel(uint4b8, g128, zp=False)  [AWQ/GPTQ common]",
        lambda: choose_mp_linear_kernel(
            MPLinearLayerConfig(
                full_weight_shape=(1024, 1024),
                partition_weight_shape=(1024, 1024),
                weight_type=scalar_types.uint4b8,
                act_type=torch.float16,
                group_size=128,
                zero_points=False,
            )
        ).__name__,
    )
    _show(
        "choose_mp_linear_kernel(uint4b8, g128, zp=True)   [AWQ with zero points]",
        lambda: choose_mp_linear_kernel(
            MPLinearLayerConfig(
                full_weight_shape=(1024, 1024),
                partition_weight_shape=(1024, 1024),
                weight_type=scalar_types.uint4b8,
                act_type=torch.float16,
                group_size=128,
                zero_points=True,
            )
        ).__name__,
    )
    _show(
        "choose_mp_linear_kernel(uint8b128, g128)          [W8A16]",
        lambda: choose_mp_linear_kernel(
            MPLinearLayerConfig(
                full_weight_shape=(1024, 1024),
                partition_weight_shape=(1024, 1024),
                weight_type=scalar_types.uint8b128,
                act_type=torch.float16,
                group_size=128,
                zero_points=False,
            )
        ).__name__,
    )
    print(f"MPLinearLayerConfig fields = {sorted(RawMPLinearLayerConfig.__dataclass_fields__)}")
    print(
        "extra attrs passed by callers: "
        "has_g_idx="
        f"{'has_g_idx' in RawMPLinearLayerConfig.__dataclass_fields__}"
    )
    _show(
        "choose_mp_linear_kernel(uint4b8, g128, has_g_idx=True) [desc_act/actorder]",
        lambda: choose_mp_linear_kernel(
            MPLinearLayerConfig(
                full_weight_shape=(1024, 1024),
                partition_weight_shape=(1024, 1024),
                weight_type=scalar_types.uint4b8,
                act_type=torch.float16,
                group_size=128,
                zero_points=False,
                has_g_idx=True,
            )
        ).__name__,
    )

    print("\n=== 2b. kernel plumbing (g_idx accessor) ===")

    class _FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qweight = torch.nn.Parameter(
                torch.zeros(128, 8, dtype=torch.int32), requires_grad=False
            )
            self.scales = torch.nn.Parameter(
                torch.zeros(8, 256, dtype=torch.float16), requires_grad=False
            )
            self.qzeros = torch.nn.Parameter(
                torch.zeros(8, 32, dtype=torch.int32), requires_grad=False
            )
            self.g_idx = torch.nn.Parameter(
                torch.zeros(0, dtype=torch.int32), requires_grad=False
            )

    try:
        from vllm.model_executor.kernels.linear.mixed_precision.gfx906_gptq_wna16 import (
            Gfx906GPTQWNA16LinearKernel,
        )

        k = Gfx906GPTQWNA16LinearKernel(
            MPLinearLayerConfig(
                full_weight_shape=(1024, 256),
                partition_weight_shape=(1024, 256),
                weight_type=scalar_types.uint4b8,
                act_type=torch.float16,
                group_size=128,
                zero_points=False,
            ),
            w_q_param_name="qweight",
            w_s_param_name="scales",
            w_zp_param_name="qzeros",
        )
        fake = _FakeLayer()
        print(f"Gfx906GPTQWNA16LinearKernel.w_gidx_name before packing = {k.w_gidx_name}")
        w_q, w_s, w_zp = k._get_weight_params(fake)
        print(
            "_get_weight_params(fake) shapes = "
            f"{tuple(w_q.shape)}, {tuple(w_s.shape)}, {tuple(w_zp.shape)}"
        )
        k.w_gidx_name = "g_idx"
        print(f"_get_gidx_param(fake) after w_gidx_name='g_idx' = {k._get_gidx_param(fake) is not None}")
        k.w_gidx_name = None
        print(f"_get_gidx_param(fake) with w_gidx_name=None = {k._get_gidx_param(fake)}")
    except Exception as exc:
        print(f"plumbing check -> RAISED {type(exc).__name__}: {exc}")

    print("\n=== 3. FP8 (documented as unsupported on gfx906) ===")
    for name, table in (
        ("per-tensor/per-channel", _POSSIBLE_FP8_KERNELS),
        ("block-scaled", _POSSIBLE_FP8_BLOCK_KERNELS),
    ):
        print(f"_POSSIBLE_FP8*[{name}][ROCM] =")
        for k in table[PlatformEnum.ROCM]:
            _show(f"  is_supported({k.__name__})", k.is_supported)

    print("\n=== 4. gfx906 on/off switches for the quant paths ===")
    print(f"VLLM_ROCM_USE_SKINNY_GEMM   = {envs.VLLM_ROCM_USE_SKINNY_GEMM}")
    print(f"VLLM_ROCM_USE_AITER         = {envs.VLLM_ROCM_USE_AITER}")
    print(f"VLLM_ROCM_USE_AITER_LINEAR  = {envs.VLLM_ROCM_USE_AITER_LINEAR}")
    print(f"VLLM_ROCM_USE_GFX906_MOBYDICK_AWQ = {envs.VLLM_ROCM_USE_GFX906_MOBYDICK_AWQ}")
    try:
        from vllm.model_executor.layers.quantization.awq import (
            _use_gfx906_awq_triton,
            _use_gfx906_mobydick_awq,
        )

        print(f"_use_gfx906_awq_triton()    = {_use_gfx906_awq_triton()}")
        print(f"_use_gfx906_mobydick_awq()  = {_use_gfx906_mobydick_awq()}")
    except Exception as exc:
        print(f"awq gfx906 helpers          = <{type(exc).__name__}: {exc}>")

    print("\n=== 5. GGUF (phase B, static check only) ===")
    try:
        from vllm.model_executor.layers.quantization.gguf import (
            _gguf_gfx906_mmvq_max_batch,
            _gguf_gfx906_q4_mmvq_max_batch,
            _gguf_gfx906_q6_mmvq_max_batch,
            _gguf_gfx906_small_batch_mmvq_enabled,
        )

        print(f"_gguf_gfx906_small_batch_mmvq_enabled() = {_gguf_gfx906_small_batch_mmvq_enabled()}")
        print(f"_gguf_gfx906_mmvq_max_batch()  = {_gguf_gfx906_mmvq_max_batch()}")
        print(f"_gguf_gfx906_q4_mmvq_max_batch() = {_gguf_gfx906_q4_mmvq_max_batch()}")
        print(f"_gguf_gfx906_q6_mmvq_max_batch() = {_gguf_gfx906_q6_mmvq_max_batch()}")
    except Exception as exc:
        print(f"gguf helpers = <{type(exc).__name__}: {exc}>")

    print("\n=== 6. MoE + GPTQ/AWQ (MoE WNA16) ===")
    try:
        from vllm.model_executor.layers.quantization import moe_wna16

        print(f"moe_wna16.on_gfx906 = " f"{moe_wna16.on_gfx906()}")
        print(
            "MoE WNA16 Torch fallback trigger: "
            "on_gfx906() and weight_bits not in (4,8) or group_size not in (32,64,128)"
            " -> vllm/model_executor/layers/quantization/moe_wna16.py:506-521"
        )
    except Exception as exc:
        print(f"moe_wna16 = <{type(exc).__name__}: {exc}>")

    print("\n=== 7. gptq_gemm / awq ops present in _C ===")
    for op in (
        "gptq_gemm",
        "gptq_shuffle",
        "awq_gemm",
        "awq_dequantize",
        "cutlass_scaled_mm",
        "static_scaled_int8_quant",
        "dynamic_scaled_int8_quant",
    ):
        print(f"hasattr(torch.ops._C, {op!r}) = {hasattr(torch.ops._C, op)}")

    print("\nQUANT_PROBE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
