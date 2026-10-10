# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S8 runtime probe: which ROCm paths are selected on gfx906 (Hygon Z100).

Run it *after* importing torch (hard rule 8) and with the S7/S8 environment
sourced. It prints one ``KEY = value`` line per audited fact so the output can
be diffed between runs and pasted into the result archive verbatim.

    python tools/dcu/s8_path_probe.py

It never launches a model and never touches the network.
"""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    print(f"torch dist   = {torch.__version__}")
    try:
        import importlib.metadata as md

        print(f"torch meta   = {md.version('torch')}")
    except Exception as exc:  # pragma: no cover
        print(f"torch meta   = <{exc}>")
    print(f"torch hip    = {torch.version.hip}")
    print(f"devcount     = {torch.cuda.device_count()}")
    if torch.cuda.device_count():
        props = torch.cuda.get_device_properties(0)
        print(f"name0        = {props.name} arch={props.gcnArchName}")

    import triton
    import importlib.metadata as md

    print(f"triton dist  = {md.version('triton')} / {triton.__file__}")

    from vllm.platforms import rocm

    print("--- arch predicates ---")
    for name in (
        "on_gfx906",
        "on_cdna",
        "on_gfx9",
        "on_gfx90a",
        "on_gfx942",
        "on_gfx950",
        "on_gfx1250",
        "on_gfx1x",
        "on_rdna4",
    ):
        fn = getattr(rocm, name, None)
        if fn is None:
            print(f"{name:12s} = <absent>")
            continue
        try:
            print(f"{name:12s} = {fn()}")
        except Exception as exc:  # pragma: no cover
            print(f"{name:12s} = <{type(exc).__name__}: {exc}>")
    print(f"cdna_version = {rocm.get_cdna_version()}")
    print(f"_GCN_ARCH    = {rocm._GCN_ARCH}")
    print(f"_ON_CDNA     = {rocm._ON_CDNA}  _ON_GFX9 = {rocm._ON_GFX9}")

    print("--- aiter ---")
    from vllm._aiter_ops import (
        IS_AITER_FOUND,
        is_aiter_found_and_supported,
        is_aiter_found_and_supported_on_rdna4,
        rocm_aiter_ops,
    )

    print(f"IS_AITER_FOUND                        = {IS_AITER_FOUND}")
    print(f"is_aiter_found_and_supported()        = {is_aiter_found_and_supported()}")
    print(
        "is_aiter_found_and_supported_on_rdna4() = "
        f"{is_aiter_found_and_supported_on_rdna4()}"
    )
    for name in (
        "is_enabled",
        "is_linear_enabled",
        "is_mha_enabled",
        "is_mla_enabled",
        "is_fused_moe_enabled",
        "is_rdna_aiter_enabled",
        "is_triton_unified_attn_enabled",
        "is_gdn_triton_kernels_importable",
    ):
        fn = getattr(rocm_aiter_ops, name, None)
        if fn is None:
            print(f"rocm_aiter_ops.{name:34s} = <absent>")
            continue
        try:
            print(f"rocm_aiter_ops.{name:34s} = {fn()}")
        except Exception as exc:  # pragma: no cover
            print(f"rocm_aiter_ops.{name:34s} = <{type(exc).__name__}: {exc}>")

    print("--- skinny GEMM / _rocm_C ---")
    from vllm import _custom_ops as ops

    print(f"ROCM_SKINNY_GEMM_ENABLED = {ops.ROCM_SKINNY_GEMM_ENABLED}")
    import vllm.envs as envs

    print(f"VLLM_ROCM_USE_SKINNY_GEMM (env read) = {envs.VLLM_ROCM_USE_SKINNY_GEMM}")
    print(
        "VLLM_ROCM_USE_AITER                  = "
        f"{envs.VLLM_ROCM_USE_AITER}"
    )
    try:
        import vllm._rocm_C  # noqa: F401

        names = sorted(
            s.name for s in torch._C._jit_get_all_schemas() if s.name.startswith("_rocm_C::")
        )
        print(f"_rocm_C ops = {names}")
    except Exception as exc:
        print(f"_rocm_C ops = <{type(exc).__name__}: {exc}>")

    print("--- attention backend selection ---")
    for use_mla in (False, True):
        try:
            prios = rocm._get_backend_priorities(use_mla=use_mla, use_sparse=False)
            print(f"_get_backend_priorities(use_mla={use_mla}) = {[b.name for b in prios]}")
        except Exception as exc:  # pragma: no cover
            print(
                f"_get_backend_priorities(use_mla={use_mla}) = "
                f"<{type(exc).__name__}: {exc}>"
            )
    print(
        "use_rocm_custom_paged_attention(fp16,h64,blk16,gqa4,8k) = "
        f"{rocm.use_rocm_custom_paged_attention(torch.float16, 64, 16, 4, 8192, 0, 'auto')}"
    )
    print(
        "use_rocm_custom_paged_attention(fp16,h128,blk16,gqa4,8k) = "
        f"{rocm.use_rocm_custom_paged_attention(torch.float16, 128, 16, 4, 8192, 0, 'auto')}"
    )
    print(f"flash_attn_triton_available() = {rocm.flash_attn_triton_available()}")

    try:
        from vllm.attention.utils.fa_utils import is_flash_attn_varlen_func_available

        print(f"is_flash_attn_varlen_func_available() = {is_flash_attn_varlen_func_available()}")
    except Exception as exc:
        print(f"is_flash_attn_varlen_func_available() = <{type(exc).__name__}: {exc}>")

    try:
        from vllm.v1.attention.backends.fa_utils import (
            is_flash_attn_varlen_func_available as v1_avail,
        )

        print(f"v1 is_flash_attn_varlen_func_available() = {v1_avail()}")
    except Exception as exc:
        print(f"v1 is_flash_attn_varlen_func_available() = <{type(exc).__name__}: {exc}>")

    for head_size in (64, 128, 256):
        for dtype in (torch.float16, torch.bfloat16):
            try:
                b = rocm.RocmPlatform.get_vit_attn_backend(head_size, dtype)
                print(f"vit_attn_backend(h{head_size},{dtype}) = {b.name}")
            except Exception as exc:
                print(
                    f"vit_attn_backend(h{head_size},{dtype}) = "
                    f"<{type(exc).__name__}: {exc}>"
                )

    print("--- gfx906 feature gates ---")
    from vllm.platforms.hygon_dtk import gfx906_bf16_enabled

    print(f"gfx906_bf16_enabled() = {gfx906_bf16_enabled()}")
    print(f"VLLM_ROCM_USE_GFX906_MOBYDICK_AWQ = {envs.VLLM_ROCM_USE_GFX906_MOBYDICK_AWQ}")
    print(
        "VLLM_ROCM_USE_GFX906_TRITON_CAUSAL_CONV1D_UPDATE = "
        f"{envs.VLLM_ROCM_USE_GFX906_TRITON_CAUSAL_CONV1D_UPDATE}"
    )
    print("--- gfx906 custom ops present in _C ---")
    for op in (
        "rms_norm_gated_gfx906",
        "gemma_rms_norm_gfx906",
        "gemma_fused_add_rms_norm_gfx906",
        "causal_conv1d_gfx906_decode_update",
        "causal_conv1d_gfx906_mtp_update",
        "fused_sigmoid_gating_delta_rule_gfx906_decode",
        "fused_sigmoid_gating_delta_rule_gfx906_prefill",
    ):
        print(f"hasattr(torch.ops._C, {op!r}) = {hasattr(torch.ops._C, op)}")

    print("PROBE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
