# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared switch for the gfx906 (Z100) emulation kernels.

Three HIP kernels in this tree replace paths that gfx906 cannot run natively:

* ``Gfx906Sdot4Int8ScaledMMLinearKernel`` (csrc/rocm/w8a8_sdot4.cu) -- W8A8
  INT8 on ``v_dot4_i32_i8``; rocBLAS/Triton int8 reaches ~2.2 TFLOPS here.
* ``Gfx906Fp8ScaledMMLinearKernel`` (csrc/rocm/fp8_emu_gemm.cu) -- W8A8 FP8
  emulated on the fp16 datapath; every other FP8 candidate refuses gfx906.
* the dense fp16/bf16 decode GEMV (csrc/rocm/bf16_emu_gemv.cu) -- the
  replacement for the rocBLAS fallback on decode-shaped linear layers.

Each kernel keeps its own env switch, because their benefit/risk profiles
differ (the INT8 kernel changes greedy tokens, the FP8 kernel is only used
when the model is actually FP8).  ``VLLM_DCU_USE_GFX906_EMU=1`` turns all of
them on at once, and an *explicit* per-kernel setting always wins over it:

    VLLM_DCU_USE_GFX906_EMU=1 VLLM_DCU_USE_INT8_SDOT4=0   # everything but int8

The helper below reads the raw environment for the per-kernel variable so that
"unset" and "explicitly 0" stay distinguishable; ``vllm/envs.py`` keeps the
bool accessors it already had, so nothing that reads them changes behaviour.
"""

import os

from vllm import envs

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _tri_state(name: str) -> bool | None:
    """Read ``name`` from the environment; ``None`` when unset/empty."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip().lower()
    if raw == "":
        return None
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return None


def gfx906_emu_enabled(specific_env: str) -> bool:
    """Whether a gfx906 emulation kernel is enabled.

    Args:
        specific_env: name of the per-kernel switch, e.g.
            ``"VLLM_DCU_USE_INT8_SDOT4"``.

    Returns:
        The explicit per-kernel setting when present, otherwise the
        ``VLLM_DCU_USE_GFX906_EMU`` master switch.
    """
    explicit = _tri_state(specific_env)
    if explicit is not None:
        return explicit
    return bool(envs.VLLM_DCU_USE_GFX906_EMU)
