#!/usr/bin/env python3
"""Z100 DCU serving-policy fingerprint (BUILD WP0.4).

Every A/B arm in ``docs/design/z100-decode-prefill-optimization-construction.md``
must cite a policy fingerprint (contracts C3/C4, gate G3).  The 2026-10
"167 vs 129 ms" incident was an *effective-policy* drift: two arms looked like
the same recipe in their shell headers but the engine computed different
values.  This tool therefore prints what the engine will actually use, not the
raw environment:

* ``VLLM_DCU_*`` / ``VLLM_GFX906_*`` / ``VLLM_QWEN35_*`` / ``VLLM_S24_*`` names
  are resolved through the real getter (``vllm/envs.py`` is executed standalone
  so no ``vllm``/``torch``/HIP import happens) or, for names that are not in
  ``envs.py``, through a replica of the consumer's parser (see the ``src`` of
  each entry).
* Defaults, tri-state fallbacks, master-switch composition and compile-time
  clamps are applied: ``VLLM_DCU_FP8_GEMV_MAX_M=16`` prints
  ``requested=16 effective=8`` and warns; ``VLLM_DCU_USE_INT8_SDOT4`` composes
  with ``VLLM_DCU_USE_GFX906_EMU``; model-config-dependent defaults print
  ``auto``.
* Compile-time policy caps (``_GEMV_MS_MAX``, ``GFX906_VERIFY_MAX_QUERY_LEN``,
  ...) are part of the fingerprint; ``--selftest`` re-reads the sources and
  fails if a cap literal drifted.

Output contract (stdout, two lines; warnings go to stderr):

    FP <sha256[:16]>
    FP_FULL <key=value ...>          # keys sorted, values effective

The hash covers both the effective values and the raw requested environment
(contract C3 pins the env verbatim), so any env change changes the hash even
when it is clamped away; the clamp itself stays visible in ``FP_FULL`` and in
the warnings.  ``--json`` prints the full structured record and ``--selftest``
runs the zero-card acceptance checks.

Usage::

    tools/dcu/dcu_policy_fingerprint.py                     # two-line recipe id
    tools/dcu/dcu_policy_fingerprint.py --json               # report/CI record
    tools/dcu/dcu_policy_fingerprint.py --selftest           # zero-card checks
    tools/dcu/dcu_policy_fingerprint.py --prefix "[host-serve] policy: "

The serve launcher can add this one line (kept out of the tool's own repo
change so ``serve_host_dflash2.sh`` stays single-owner)::

    "${PY}" "${REPO}/tools/dcu/dcu_policy_fingerprint.py" \
        --prefix "[host-serve] policy: " 2>&1 || true
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
FALSE_TOKENS = frozenset({"0", "false", "off", "no"})
SCHEMA_VERSION = 1

# Names are resolved from their consumer's parser unless ``envs=True`` (the
# getter in vllm/envs.py is authoritative then).  ``src`` is "path:line" of the
# primary read; ``note`` records extra provenance and semantics that the
# fingerprint deliberately normalises away.
#
# kind: bool | int | int_e | int_d | int_o | float_e | str | str_lower |
#       str_strip | cxx_int
# post: optional normalisation hook, see POSTS.
KNOBS: list[dict] = [
    # ---- VLLM_DCU_*: gfx906 emulation master/per-kernel switches ----------
    dict(
        name="VLLM_DCU_GCN_ARCH",
        kind="str_strip",
        default="",
        src="vllm/envs.py:1529",
        envs=True,
        post="arch_auto",
        note="empty = auto-detect (KFD/torch); consumes as hygon_dtk at :310",
    ),
    dict(
        name="VLLM_DCU_BF16",
        kind="str_lower",
        default="off",
        src="vllm/envs.py:1533",
        envs=True,
        post="bf16_policy",
        note="hygon_dtk.bf16_policy() at vllm/platforms/hygon_dtk.py:329",
    ),
    dict(
        name="VLLM_DCU_CUSTOM_AR",
        kind="bool",
        default=False,
        src="vllm/envs.py:1534",
        envs=True,
        note="dcu_custom_ar_enabled() -> envs getter",
    ),
    dict(
        name="VLLM_DCU_CUSTOM_AR_ALLOC",
        kind="str_strip",
        default="",
        src="vllm/envs.py:1537",
        envs=True,
        note="custom all-reduce IPC allocation flag, read in hygon_dtk.py:174",
    ),
    dict(
        name="VLLM_CUSTOM_ALLREDUCE_ALGO",
        kind="str_strip",
        default="",
        src="vllm/distributed/device_communicators/custom_all_reduce.py:44",
        envs=False,
        note="S33 WP2: forced custom AR algorithm (oneshot on PCIe-only "
        "gfx906 groups); empty = C++ dispatch default",
    ),
    dict(
        name="VLLM_DCU_USE_GFX906_EMU",
        kind="bool",
        default=False,
        src="vllm/envs.py:1542",
        envs=True,
        note="master switch for the three emulation kernels (gfx906_common.py:50)",
    ),
    dict(
        name="VLLM_DCU_USE_INT8_SDOT4",
        kind="bool",
        default=False,
        src="vllm/envs.py:1550",
        envs=True,
        post="emu_tri",
        note="effective = explicit tri-state, else master switch",
    ),
    dict(
        name="VLLM_DCU_USE_FP8_EMU",
        kind="bool",
        default=False,
        src="vllm/envs.py:1558",
        envs=True,
        post="emu_tri",
        note="effective = explicit tri-state, else master switch",
    ),
    dict(
        name="VLLM_DCU_FP8_CVT",
        kind="str_lower",
        default="lut",
        src="vllm/envs.py:1563",
        envs=True,
        post="dead_knob",
        note="dead knob: no consumer left in vllm/, csrc/ or tools/dcu/",
    ),
    dict(
        name="VLLM_DCU_USE_BF16_GEMV",
        kind="bool",
        default=False,
        src="vllm/envs.py:1566",
        envs=True,
        post="emu_tri",
        note="layers/utils.py:543 calls gfx906_emu_enabled(name)",
    ),
    dict(
        name="VLLM_DCU_BF16_ROUTE",
        kind="str_lower",
        default="auto",
        src="vllm/envs.py:1570",
        envs=True,
        post="bf16_route",
        note="mapped to kernel route 0/1/2 at layers/utils.py:558",
    ),
    dict(
        name="VLLM_DCU_FP8_GEMV_MAX_M",
        kind="int",
        default=8,
        src="vllm/envs.py:1574",
        envs=True,
        post="gemv_max_m",
        note="python clamp [0,8] at scaled_mm/gfx906_fp8.py:83",
    ),
    dict(
        name="VLLM_DCU_GEMV_NARROW_MAX_M",
        kind="int",
        default=8,
        src="vllm/model_executor/layers/utils.py",
        envs=False,
        note="A1 narrow-N bf16-GEMV rule cap (n_out<=64); 4 restores pre-A1",
    ),
    dict(
        name="VLLM_DCU_INT8_GEMV_MAX_M",
        kind="int",
        default=8,
        src="vllm/envs.py:1577",
        envs=True,
        post="int8_gemv_max_m",
        note="unclamped at scaled_mm/gfx906_sdot4.py:82; kMsMax=8 in the .cu",
    ),
    dict(
        name="VLLM_DCU_FP8_EMU_FASTPATH",
        kind="bool",
        default=False,
        src="vllm/envs.py:1583",
        envs=True,
    ),
    dict(
        name="VLLM_DCU_FP8_QUANT_FASTPATH",
        kind="bool",
        default=False,
        src="vllm/envs.py:1586",
        envs=True,
    ),
    dict(
        name="VLLM_DCU_FP8_FUSE_QUANT",
        kind="bool",
        default=False,
        src="vllm/envs.py:1590",
        envs=True,
    ),
    dict(
        name="VLLM_DCU_FP8_W8A16",
        kind="str_lower",
        default="auto",
        src="vllm/envs.py:1599",
        envs=True,
        post="w8a16_tri",
        note="validated at scaled_mm/gfx906_fp8.py:253-260",
    ),
    dict(
        name="VLLM_DCU_FP16_PREFILL",
        kind="str_lower",
        default="0",
        src="vllm/envs.py:1602",
        envs=True,
        post="fp16_prefill_tri",
        note="validated at scaled_mm/gfx906_fp16_prefill.py:50-56",
    ),
    dict(
        name="VLLM_DCU_FP16_PREFILL_MIN_M",
        kind="int_e",
        default=0,
        src="vllm/envs.py:1605",
        envs=True,
        post="fp16_min_m",
        note=">0 overrides _DEFAULT_MIN_M=4096 at gfx906_fp16_prefill.py:86",
    ),
    dict(
        name="VLLM_DCU_FP16_WS_BYTES",
        kind="int_e",
        default=0,
        src="vllm/envs.py:1608",
        envs=True,
        note="lower bound for the reused fp16 prefill scratch",
    ),
    # ---- VLLM_S24_*: dynamic verify truncation ----------------------------
    dict(
        name="VLLM_S24_DYNK_LO",
        kind="int_e",
        default=0,
        src="vllm/envs.py:1611",
        envs=True,
        note="0/0 disables the truncation window",
    ),
    dict(
        name="VLLM_S24_DYNK_HI",
        kind="int_e",
        default=0,
        src="vllm/envs.py:1614",
        envs=True,
    ),
    dict(
        name="VLLM_S24_DYNK_EMA",
        kind="float_e",
        default=0.1,
        src="vllm/envs.py:1613",
        envs=True,
    ),
    # ---- VLLM_DCU_*: knobs parsed outside envs.py ------------------------
    dict(
        name="VLLM_DCU_SDOT4_INNER",
        kind="str_lower",
        default="sdot4",
        src="vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py:100",
        note="diagnostic only; 'triton' keeps the custom_op, runs triton inside",
    ),
    dict(
        name="VLLM_DCU_SMALLM_GEMM",
        kind="str_lower",
        default="",
        src="vllm/model_executor/layers/utils.py:554",
        post="smallm",
        note="rocblas/1/true/yes/on opts small-M GEMMs into rocBLAS",
    ),
    dict(
        name="VLLM_DCU_FP8_PREFILL_CFG",
        kind="str_lower",
        default="0",
        src=("vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py:110"),
        post="prefill_cfg",
        note="0=legacy, auto=cfg rule, 1..19=forced tile",
    ),
    dict(
        name="VLLM_DCU_FP8_DECODE_CFG",
        kind="str_strip",
        default=None,
        src="csrc/rocm/fp8_emu_gemm.cu:2154",
        post="decode_cfg",
        note="unset/0=auto (N>=6144 ? 10 : 5); 11..19/32..35 forced; else 16",
    ),
    dict(
        name="VLLM_DCU_FP8_GEMV_STREAM",
        kind="cxx_int",
        default=1,
        src="csrc/rocm/fp8_emu_gemm.cu:868",
        post="gemv_stream",
        note="0=old GEMV, 1=streaming v1, 2=streaming v2",
    ),
    dict(
        name="VLLM_DCU_FP8_GEMV_STREAM_ROWS",
        kind="cxx_int",
        default=0,
        src="csrc/rocm/fp8_emu_gemm.cu:924",
        post="gemv_stream_rows",
        note="rows/block override for the slice stream kernel; 0=auto",
    ),
    dict(
        name="VLLM_DCU_FP8_GEMV_M1",
        kind="cxx_int",
        default=0,
        src="csrc/rocm/fp8_emu_gemm.cu:1120",
        post="cxx_flag",
        note="S30c WP2-N: M==1 m1 stream kernel gate (W8A16 route)",
    ),
    dict(
        name="VLLM_DCU_FP8_GEMV_WA_L2",
        kind="cxx_int",
        default=0,
        src="csrc/rocm/fp8_emu_gemm.cu",
        post="cxx_flag",
        note="S33 WP1: whole-A L2-direct sub-mode (default 0 = LDS variant)",
    ),
    dict(
        name="VLLM_DCU_FP8_GEMV_WA",
        kind="cxx_int",
        default=0,
        src="csrc/rocm/fp8_emu_gemm.cu",
        post="cxx_flag",
        note="S33 WP1: M==1 whole-A GEMV kernel gate (W8A16 route)",
    ),
    dict(
        name="VLLM_DCU_FP8_STREAM_MAX_M",
        kind="cxx_int",
        default=0,
        src="csrc/rocm/fp8_emu_gemm.cu:809",
        post="stream_max_m",
        note="skip streaming path when M >= value; 0 = no limit",
    ),
    # ---- VLLM_GFX906_*: attention / profile / GDN dispatch ---------------
    dict(
        name="VLLM_GFX906_TRITON_ATTN_TUNING",
        kind="bool",
        default=True,
        src="vllm/v1/attention/ops/triton_unified_attention.py:32",
        note="decode scheduling tuning kill switch",
    ),
    dict(
        name="VLLM_GFX906_TRITON_ATTN_PREFILL_3D",
        kind="bool",
        default=False,
        src="vllm/v1/attention/ops/triton_unified_attention.py:47",
        note="opt-in split-KV prefill route",
    ),
    dict(
        name="VLLM_GFX906_ATTN_VERIFY_3D",
        kind="bool",
        default=True,
        src="vllm/v1/attention/ops/triton_unified_attention.py:71",
        note="decode-shaped routing for 1 < max_seqlen_q <= 8",
    ),
    dict(
        name="VLLM_GFX906_ATTN_VERIFY_3D_DRAFT",
        kind="bool",
        default=True,
        src="vllm/v1/attention/ops/triton_unified_attention.py",
        note="S28: extend the small-q 3D routing to the draft shape (128/GQA4)",
    ),
    dict(
        name="VLLM_GFX906_ATTN_PREFILL_BLOCK_M",
        kind="int_e",
        default=0,
        src="vllm/v1/attention/ops/triton_unified_attention.py:98",
    ),
    dict(
        name="VLLM_GFX906_ATTN_PREFILL_TILE",
        kind="int_e",
        default=0,
        src="vllm/v1/attention/ops/triton_unified_attention.py:101",
    ),
    dict(
        name="VLLM_GFX906_ATTN_PREFILL_WARPS",
        kind="int_e",
        default=4,
        src="vllm/v1/attention/ops/triton_unified_attention.py:102",
    ),
    dict(
        name="VLLM_GFX906_ATTN_PREFILL_STAGES",
        kind="int_e",
        default=1,
        src="vllm/v1/attention/ops/triton_unified_attention.py:103",
    ),
    dict(
        name="VLLM_GFX906_ATTN_PREFILL_SEGMENTS",
        kind="int_e",
        default=4,
        src="vllm/v1/attention/ops/triton_unified_attention.py:104",
    ),
    dict(
        name="VLLM_GFX906_ATTN_PREFILL_3D_WARPS",
        kind="int_e",
        default=4,
        src="vllm/v1/attention/ops/triton_unified_attention.py:105",
    ),
    dict(
        name="VLLM_GFX906_ATTN_PREFILL_3D_STAGES",
        kind="int_e",
        default=1,
        src="vllm/v1/attention/ops/triton_unified_attention.py:106",
    ),
    dict(
        name="VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX",
        kind="int_e",
        default=16384,
        src="vllm/v1/attention/ops/triton_unified_attention.py:115",
    ),
    dict(
        name="VLLM_GFX906_ATTN_DECODE_BLOCK_M",
        kind="int_e",
        default=0,
        src="vllm/v1/attention/ops/triton_unified_attention.py:125",
    ),
    dict(
        name="VLLM_GFX906_ATTN_DECODE_WARPS",
        kind="int_e",
        default=0,
        src="vllm/v1/attention/ops/triton_unified_attention.py:128",
    ),
    dict(
        name="VLLM_GFX906_ATTN_DECODE_SEGMENTS",
        kind="int_e",
        default=0,
        src="vllm/v1/attention/ops/triton_unified_attention.py:129",
    ),
    dict(
        name="VLLM_GFX906_ATTN_EAGER_FALLBACK",
        kind="bool",
        default=False,
        src="vllm/v1/attention/backends/triton_attn.py:680",
        note="validation-only eager attention fallback",
    ),
    dict(
        name="VLLM_GFX906_ATTN_FORCE_CAUSAL",
        kind="bool",
        default=False,
        src="vllm/v1/attention/ops/triton_unified_attention.py:1261",
        params={"mode": "eq_1"},
        note="strict string equality with '1'",
    ),
    dict(
        name="VLLM_GFX906_ATTN_MULTI_QUERY_3D",
        kind="bool",
        default=True,
        src="vllm/attention/ops/triton_unified_attention.py:24",
        note="legacy attention op path",
    ),
    dict(
        name="VLLM_GFX906_TRITON_UNIFIED_ATTN",
        kind="bool",
        default=True,
        src="vllm/attention/ops/triton_unified_attention.py:21",
        note="legacy attention op path",
    ),
    dict(
        name="VLLM_GFX906_TRITON_CAUSAL_CONV1D",
        kind="bool",
        default=False,
        src="vllm/model_executor/layers/mamba/ops/causal_conv1d.py:37",
        note="default off: quality side effect, see module comment",
    ),
    dict(
        name="VLLM_GFX906_TRITON_CAUSAL_CONV1D_SYNC",
        kind="bool",
        default=True,
        src="vllm/model_executor/layers/mamba/ops/causal_conv1d.py:53",
        note="launch-safety stream sync",
    ),
    dict(
        name="VLLM_GFX906_TRITON_MROPE",
        kind="bool",
        default=True,
        src="vllm/model_executor/layers/rotary_embedding/mrope.py:17",
    ),
    dict(
        name="VLLM_GFX906_TRITON_RESHAPE_CACHE",
        kind="bool",
        default=True,
        src="vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:24",
        note="0 routes the cache update through the eager fallback",
    ),
    dict(
        name="VLLM_GFX906_GDN_CHUNKED_PREFILL",
        kind="bool",
        default=False,
        src="vllm/model_executor/layers/mamba/gdn_linear_attn.py:109",
        note="opt-in FLA chunked prefill route",
    ),
    dict(
        name="VLLM_GFX906_GDN_TRITON_PREFILL",
        kind="bool",
        default=True,
        src="vllm/model_executor/layers/fla/ops/utils.py:34",
        note="0 routes chunked GDN prefill through eager fallbacks",
    ),
    dict(
        name="VLLM_GFX906_CORE_PROFILE",
        kind="bool",
        default=False,
        src="vllm/v1/engine/core.py:246",
    ),
    dict(
        name="VLLM_GFX906_CORE_PROFILE_LIMIT",
        kind="int",
        default=64,
        src="vllm/v1/engine/core.py:249",
    ),
    dict(
        name="VLLM_GFX906_RUNNER_PROFILE",
        kind="bool",
        default=False,
        src="vllm/v1/worker/gpu_model_runner.py:988",
    ),
    dict(
        name="VLLM_GFX906_RUNNER_PROFILE_LIMIT",
        kind="int",
        default=32,
        src="vllm/v1/worker/gpu_model_runner.py:991",
    ),
    dict(
        name="VLLM_GFX906_DP4X_GUARD",
        kind="str_lower",
        default="auto",
        src="vllm/platforms/hygon_dp4x.py:184",
        post="dp4x_guard",
        note="auto probes KFD; 0/off disables, 1/force forces all",
    ),
    dict(
        name="VLLM_GFX906_DP4X_MIN_WARPS",
        kind="int_d",
        default=8,
        src="vllm/platforms/hygon_dp4x.py:209",
        post="dp4x_min_warps",
        note="clamped to [1,32], invalid -> 8",
    ),
    dict(
        name="VLLM_GFX906_DP4X_WATCH",
        kind="bool",
        default=True,
        src="vllm/platforms/hygon_dp4x.py:219",
        params={"mode": "not_off"},
        note="warns if a kernel still compiles to VGPR=256",
    ),
    dict(
        name="VLLM_GFX906_DP4X_DEBUG",
        kind="bool",
        default=False,
        src="vllm/platforms/hygon_dp4x.py:314",
        params={"mode": "not_off"},
        note="any value outside {0,false,off,no} enables",
    ),
    # ---- C++-only VLLM_GFX906_* switches (aten kernels) ------------------
    dict(
        name="VLLM_GFX906_GDN_DEC_V2",
        kind="cxx_int",
        default=None,
        src="csrc/mamba/gfx906_decode_kernels.cu:2863",
        post="cxx_enabled",
        note="enabled unless atoi(raw) == 0; unset enabled",
    ),
    dict(
        name="VLLM_GFX906_GDN_DEC_V2_DEBUG",
        kind="cxx_int",
        default=0,
        src="csrc/mamba/gfx906_decode_kernels.cu:2874",
        post="cxx_flag",
    ),
    dict(
        name="VLLM_GFX906_GDN_DEC_V2_TIME",
        kind="cxx_int",
        default=0,
        src="csrc/mamba/gfx906_decode_kernels.cu:2902",
        post="cxx_flag",
    ),
    dict(
        name="VLLM_GFX906_GDN_MTP_V2",
        kind="cxx_int",
        default=None,
        src="csrc/mamba/gfx906_decode_kernels.cu:2604",
        post="cxx_enabled",
        note="enabled unless atoi(raw) == 0; unset enabled",
    ),
    dict(
        name="VLLM_GFX906_GDN_MTP_V2_DEBUG",
        kind="cxx_int",
        default=0,
        src="csrc/mamba/gfx906_decode_kernels.cu:2616",
        post="cxx_flag",
    ),
    dict(
        name="VLLM_GFX906_GDN_MTP_V2_TIME",
        kind="cxx_int",
        default=0,
        src="csrc/mamba/gfx906_decode_kernels.cu:2643",
        post="cxx_flag",
    ),
    # ---- S30c WP1-N: native-layout M=1 decode kernel (m1_native) ---------
    dict(
        name="VLLM_GFX906_GDN_DEC_M1",
        kind="cxx_int",
        default=None,
        src="csrc/mamba/gfx906_decode_kernels.cu:2862",
        post="cxx_enabled",
        note="enabled unless atoi(raw) == 0; unset enabled",
    ),
    dict(
        name="VLLM_GFX906_GDN_DEC_M1_CFG",
        kind="cxx_int",
        default=1,
        src="csrc/mamba/gfx906_decode_kernels.cu:2866",
        post="cxx_int",
    ),
    dict(
        name="VLLM_GFX906_GDN_DEC_M1_DEBUG",
        kind="cxx_int",
        default=0,
        src="csrc/mamba/gfx906_decode_kernels.cu:2896",
        post="cxx_flag",
    ),
    dict(
        name="VLLM_GFX906_GDN_DEC_M1_TIME",
        kind="cxx_int",
        default=0,
        src="csrc/mamba/gfx906_decode_kernels.cu:2906",
        post="cxx_flag",
    ),
    # ---- VLLM_QWEN35_*: model-path switches ------------------------------
    dict(
        name="VLLM_QWEN35_CALL_ORDER",
        kind="str_lower",
        default="ba",
        src="vllm/model_executor/models/qwen3_5.py:313",
        post="call_order",
        note="anything but 'ab' selects the default 'ba' call order",
    ),
    dict(
        name="VLLM_QWEN35_CHUNK_DECODE",
        kind="bool",
        default=None,
        src="vllm/model_executor/models/qwen3_5.py:696",
        post="auto_model",
        note="auto = split_projections && gfx906 rocm",
    ),
    dict(
        name="VLLM_QWEN35_COMBINED_PACKED_DECODE",
        kind="bool",
        default=True,
        src="vllm/model_executor/models/qwen3_5.py:302",
        note="gdn_linear_attn.py:432 derives its own default (gfx906)",
    ),
    dict(
        name="VLLM_QWEN35_EMPTY_CORE_ATTN_OUT",
        kind="bool",
        default=False,
        src="vllm/model_executor/layers/mamba/gdn_linear_attn.py:70",
    ),
    dict(
        name="VLLM_QWEN35_EMPTY_CORE_ATTN_SINGLE_TOKEN",
        kind="bool",
        default=True,
        src="vllm/model_executor/models/qwen3_5.py:305",
    ),
    dict(
        name="VLLM_QWEN35_EXPAND_QK",
        kind="bool",
        default=True,
        src="vllm/model_executor/models/qwen3_5.py:279",
    ),
    dict(
        name="VLLM_QWEN35_FORCE_BA_SPLIT",
        kind="bool",
        default=False,
        src="vllm/model_executor/models/qwen3_5.py:1102",
    ),
    dict(
        name="VLLM_QWEN35_FORCE_Z_ONES",
        kind="bool",
        default=False,
        src="vllm/model_executor/layers/mamba/gdn_linear_attn.py:69",
        params={"mode": "eq_1"},
        note="strict string equality with '1'",
    ),
    dict(
        name="VLLM_QWEN35_FUSED_QK_NORM_ROPE",
        kind="bool",
        default=True,
        src="vllm/model_executor/models/qwen3_next.py:1411",
    ),
    dict(
        name="VLLM_QWEN35_GGUF_ALLOW_FULL_CUDAGRAPH",
        kind="bool",
        default=False,
        src="vllm/config/vllm.py:316",
        note="debug-only bypass of the gfx906 GGUF graph guard",
    ),
    dict(
        name="VLLM_QWEN35_GGUF_ALLOW_PIECEWISE_CUDAGRAPH",
        kind="bool",
        default=False,
        src="vllm/config/vllm.py:340",
    ),
    dict(
        name="VLLM_QWEN35_GGUF_DEBUG",
        kind="bool",
        default=False,
        src="vllm/model_executor/models/qwen3_5.py:341",
    ),
    dict(
        name="VLLM_QWEN35_GGUF_DEBUG_FILE",
        kind="str",
        default=None,
        src="vllm/model_executor/models/qwen3_5.py:347",
    ),
    dict(
        name="VLLM_QWEN35_GGUF_DEBUG_LAYER",
        kind="str",
        default="model.layers.0.linear_attn",
        src="vllm/model_executor/models/qwen3_5.py:344",
    ),
    dict(
        name="VLLM_QWEN35_GROUPED_GGUF_BA_MMVQ",
        kind="bool",
        default=True,
        src="vllm/model_executor/models/qwen3_5.py:295",
    ),
    dict(
        name="VLLM_QWEN35_GROUPED_GGUF_MMVQ",
        kind="bool",
        default=True,
        src="vllm/model_executor/models/qwen3_5.py:294",
    ),
    dict(
        name="VLLM_QWEN35_LAYER_PROFILE",
        kind="bool",
        default=False,
        src="vllm/model_executor/models/qwen3_5.py:825",
    ),
    dict(
        name="VLLM_QWEN35_LAYER_PROFILE_LIMIT",
        kind="int_d",
        default=2,
        src="vllm/model_executor/models/qwen3_5.py:831",
    ),
    dict(
        name="VLLM_QWEN35_LAYER_PROFILE_MAX_TOKENS",
        kind="int_o",
        default=None,
        src="vllm/model_executor/models/qwen3_5.py:828",
    ),
    dict(
        name="VLLM_QWEN35_LINEAR_ATTN_PROFILE",
        kind="bool",
        default=False,
        src="vllm/model_executor/models/qwen3_5.py:332",
    ),
    dict(
        name="VLLM_QWEN35_LINEAR_ATTN_PROFILE_LIMIT",
        kind="int_d",
        default=4,
        src="vllm/model_executor/models/qwen3_5.py:338",
    ),
    dict(
        name="VLLM_QWEN35_LINEAR_ATTN_PROFILE_MAX_TOKENS",
        kind="int_o",
        default=None,
        src="vllm/model_executor/models/qwen3_5.py:335",
    ),
    dict(
        name="VLLM_QWEN35_LOAD_DEBUG_FILE",
        kind="str",
        default=None,
        src="vllm/model_executor/models/qwen3_5.py:162",
    ),
    dict(
        name="VLLM_QWEN35_LOCAL_REC_DECODE",
        kind="bool",
        default=False,
        src="vllm/model_executor/models/qwen3_5.py:288",
    ),
    dict(
        name="VLLM_QWEN35_MTP_CONV_CUSTOM",
        kind="bool",
        default=True,
        src=("vllm/model_executor/layers/mamba/ops/causal_conv1d.py:1445"),
        params={"mode": "ne_zero"},
        note="only the exact string '0' disables",
    ),
    dict(
        name="VLLM_QWEN35_MTP_CONV_DEBUG",
        kind="bool",
        default=False,
        src=("vllm/model_executor/layers/mamba/ops/causal_conv1d.py:1464"),
    ),
    dict(
        name="VLLM_QWEN35_MTP_GDN_CUSTOM",
        kind="bool",
        default=True,
        src=("vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py:702"),
        params={"mode": "ne_zero"},
        note="only the exact string '0' disables",
    ),
    dict(
        name="VLLM_QWEN35_MTP_GDN_DEBUG",
        kind="bool",
        default=False,
        src="vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py:45",
    ),
    dict(
        name="VLLM_QWEN35_MTP_GDN_TRITON",
        kind="bool",
        default=False,
        src=("vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py:26"),
    ),
    dict(
        name="VLLM_QWEN35_MTP_GDN_TRITON_BV",
        kind="int",
        default=4,
        src=("vllm/model_executor/layers/fla/ops/gfx906_gdn_mtp_triton.py:72"),
    ),
    dict(
        name="VLLM_QWEN35_MTP_GDN_TRITON_PROBE",
        kind="bool",
        default=False,
        src=("vllm/model_executor/layers/fla/ops/gfx906_gdn_mtp_triton.py:88"),
    ),
    dict(
        name="VLLM_QWEN35_MTP_GDN_TRITON_WARPS",
        kind="int",
        default=1,
        src=("vllm/model_executor/layers/fla/ops/gfx906_gdn_mtp_triton.py:73"),
    ),
    dict(
        name="VLLM_QWEN35_PACKED_DECODE",
        kind="bool",
        default=None,
        src="vllm/model_executor/models/qwen3_5.py:406",
        post="auto_model",
        note="auto = split_projections ? 1 : 0",
    ),
    dict(
        name="VLLM_QWEN35_PACKED_DECODE_TILED_QK",
        kind="bool",
        default=None,
        src="vllm/model_executor/models/qwen3_5.py:413",
        post="auto_model",
        note="auto = split_projections ? 1 : 0",
    ),
    dict(
        name="VLLM_QWEN35_QK_L2NORM",
        kind="bool",
        default=True,
        src="vllm/model_executor/models/qwen3_5.py:291",
    ),
    dict(
        name="VLLM_QWEN35_RATIO2_FUSED_DECODE",
        kind="bool",
        default=True,
        src="vllm/model_executor/layers/fla/ops/fused_recurrent.py:23",
    ),
    dict(
        name="VLLM_QWEN35_RATIO2_FUSED_DECODE_DEBUG",
        kind="bool",
        default=False,
        src="vllm/model_executor/layers/fla/ops/fused_recurrent.py:515",
    ),
    dict(
        name="VLLM_QWEN35_REC_PREFILL",
        kind="bool",
        default=False,
        src="vllm/model_executor/models/qwen3_5.py:285",
    ),
    dict(
        name="VLLM_QWEN35_REVERSE_BA_SHARDS",
        kind="bool",
        default=False,
        src="vllm/model_executor/models/qwen3_5.py:1118",
    ),
    dict(
        name="VLLM_QWEN35_RUNTIME_DEBUG_FILE",
        kind="str",
        default=None,
        src="vllm/model_executor/models/qwen3_next.py:134",
    ),
    dict(
        name="VLLM_QWEN35_SHARED_GDN_CORE",
        kind="bool",
        default=False,
        src="vllm/model_executor/models/qwen3_5.py:322",
    ),
    dict(
        name="VLLM_QWEN35_TILED_QK_EXPAND",
        kind="bool",
        default=None,
        src="vllm/model_executor/models/qwen3_5.py:310",
        post="auto_model",
        note="auto = split_projections ? 1 : 0",
    ),
]

# Compile-time caps: not env knobs, but they bound the policy and a rebuilt
# .so / edited source can silently change them.  ``--selftest`` re-reads each
# literal so drift fails loudly instead of rotting in a report.
CAPS: list[dict] = [
    dict(
        key="CAP_GEMV_MS_MAX",
        path="vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py",
        rx=r"^_GEMV_MS_MAX = (\d+)$",
        expect=8,
    ),
    dict(
        key="CAP_INT8_GEMV_MS_MAX",
        path="csrc/rocm/w8a8_sdot4.cu",
        rx=r"^constexpr int kMsMax = (\d+);",
        expect=8,
    ),
    dict(
        key="CAP_FP8_EMU_MS_MAX",
        path="csrc/rocm/fp8_emu_gemm.cu",
        rx=r"^constexpr int kMsMax = (\d+);",
        expect=8,
    ),
    dict(
        key="CAP_ATTN_VERIFY_MAX_QUERY_LEN",
        path="vllm/v1/attention/ops/triton_unified_attention.py",
        rx=r"^GFX906_VERIFY_MAX_QUERY_LEN = (\d+)$",
        expect=8,
    ),
    dict(
        key="CAP_ATTN_MAX_DECODE_SEGMENTS",
        path="vllm/v1/attention/ops/triton_unified_attention.py",
        rx=r"^GFX906_MAX_DECODE_SEGMENTS = (\d+)$",
        expect=128,
    ),
    dict(
        key="CAP_FP16_PREFILL_FORCE_MIN_M",
        path=("vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py"),
        rx=r"^_FORCE_MIN_M = (\d+)$",
        expect=16,
    ),
]

SCAN_DIRS = ("vllm", "csrc", "tools/dcu")
SCAN_EXTS = (".py", ".cu", ".cuh", ".cpp", ".h", ".sh")
_NAME_RX = re.compile(r'"((?:VLLM_(?:DCU|GFX906|QWEN35|S24|SPEC)_[A-Z0-9_]+))"')
KNOB_NAMES = tuple(entry["name"] for entry in KNOBS)


class ParseError(RuntimeError):
    """The consumer's parser itself would raise (engine import/startup fails)."""


class Warning_:
    """Collected policy warning; rendered to stderr and included in --json."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


def _warn(warnings: list[str], text: str) -> None:
    warnings.append(text)


def _atoi(raw: str) -> int:
    """C ``std::atoi``: leading integer, 0 when there is none."""
    match = re.match(r"\s*([+-]?\d+)", raw)
    return int(match.group(1)) if match else 0


def _tri_state(raw: str | None) -> bool | None:
    """gfx906_common._tri_state: None when unset/empty/unrecognized."""
    if raw is None:
        return None
    token = raw.strip().lower()
    if not token:
        return None
    if token in TRUE_TOKENS:
        return True
    if token in FALSE_TOKENS:
        return False
    return None


def _parse(kind: str, raw: str | None, default, params: dict | None) -> object:
    """Replicate the raw-read half of a consumer's parser."""
    if kind == "bool":
        if raw is None:
            return default
        mode = (params or {}).get("mode", "in_true")
        if mode == "in_true":
            return raw.lower() in TRUE_TOKENS
        if mode == "not_off":
            return raw.strip().lower() not in FALSE_TOKENS
        if mode == "ne_zero":
            return raw != "0"
        if mode == "eq_1":
            return raw == "1"
        raise KeyError(f"unknown bool mode {mode!r}")
    if kind == "int":
        if raw is None:
            return default
        return int(raw)
    if kind == "int_e":
        if raw is None or not raw.strip():
            return default
        return int(raw.strip())
    if kind == "int_d":
        if raw is None or not raw.strip():
            return default
        try:
            return int(raw)
        except ValueError:
            return default
    if kind == "int_o":
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None
    if kind == "float_e":
        if raw is None or not raw.strip():
            return default
        return float(raw)
    if kind == "str":
        return default if raw is None else raw
    if kind == "str_lower":
        return default if raw is None else raw.strip().lower()
    if kind == "str_strip":
        return default if raw is None else raw.strip()
    if kind == "cxx_int":
        return default if raw is None else _atoi(raw)
    raise KeyError(f"unknown kind {kind!r}")


# ---------------------------------------------------------------------------
# Effective-value post-processing (defaults, caches, clamps, composition)
# ---------------------------------------------------------------------------


def _post_arch_auto(value, raw, ctx):
    return ("auto" if value in (None, "") else str(value)), []


def _post_bf16_policy(value, raw, ctx):
    token = str(value or "").strip().lower()
    if token in ("1", "true", "on", "yes", "force"):
        return "on", []
    if token in ("0", "false", "off", "no", ""):
        return "off", []
    if token == "auto":
        return "auto", []
    return "off", [f"VLLM_DCU_BF16={raw!r} unrecognized -> off (bf16_policy)"]


def _post_bf16_route(value, raw, ctx):
    token = str(value or "").strip().lower()
    if token in ("auto", "f32", "f16"):
        return token, []
    return "auto", [f"VLLM_DCU_BF16_ROUTE={raw!r} unrecognized -> auto"]


def _post_emu_tri(value, raw, ctx):
    """gfx906_common.gfx906_emu_enabled: explicit tri-state wins over master."""
    tri = _tri_state(raw)
    master = bool(ctx["values"].get("VLLM_DCU_USE_GFX906_EMU", False))
    effective = master if tri is None else tri
    warnings: list[str] = []
    if tri is None and raw not in (None, ""):
        _warn(
            warnings,
            f"{ctx['name']}={raw!r} is not a recognized tri-state token; "
            f"falling back to VLLM_DCU_USE_GFX906_EMU={int(master)}",
        )
    return effective, warnings


def _post_w8a16_tri(value, raw, ctx):
    token = str(value or "").strip().lower()
    if token in ("0", "1", "auto"):
        return token, []
    return "0", [
        f"VLLM_DCU_FP8_W8A16={raw!r} unrecognized -> 0 (gfx906_fp8.py:253-260)"
    ]


def _post_fp16_prefill_tri(value, raw, ctx):
    token = str(value or "").strip().lower()
    if token in ("0", "1", "auto"):
        return token, []
    text = (
        f"VLLM_DCU_FP16_PREFILL={raw!r} unrecognized -> 0 "
        "(gfx906_fp16_prefill.py:50-56)"
    )
    return "0", [text]


def _post_gemv_max_m(value, raw, ctx):
    cap = int(ctx["caps"].get("CAP_GEMV_MS_MAX", 8))
    requested = int(value)
    effective = min(max(requested, 0), cap)
    warnings: list[str] = []
    if effective != requested:
        _warn(
            warnings,
            "VLLM_DCU_FP8_GEMV_MAX_M: "
            f"requested={requested} effective={effective} "
            f"(clamped to _GEMV_MS_MAX={cap})",
        )
    return effective, warnings


def _post_int8_gemv_max_m(value, raw, ctx):
    cap = int(ctx["caps"].get("CAP_INT8_GEMV_MS_MAX", 8))
    requested = int(value)
    warnings: list[str] = []
    if requested > cap:
        _warn(
            warnings,
            f"VLLM_DCU_INT8_GEMV_MAX_M: requested={requested} exceeds the "
            f"compiled kMsMax={cap}; M in ({cap}, {requested}] would be sent "
            "to the GEMV kernel, whose TORCH_CHECK rejects M > 8 "
            "(gfx906_sdot4.py:82, csrc/rocm/w8a8_sdot4.cu:394)",
        )
    elif requested < 0:
        _warn(
            warnings,
            f"VLLM_DCU_INT8_GEMV_MAX_M: requested={requested} is negative; "
            "no M ever takes the INT8 GEMV path",
        )
    return requested, warnings


def _post_fp16_min_m(value, raw, ctx):
    requested = int(value)
    warnings: list[str] = []
    if requested < 0:
        _warn(
            warnings,
            "VLLM_DCU_FP16_PREFILL_MIN_M: "
            f"requested={requested} is ignored (only >0 overrides the "
            "per-role table; gfx906_fp16_prefill.py:86-88)",
        )
    return requested, warnings


def _post_prefill_cfg(value, raw, ctx):
    token = str(value or "").strip().lower()
    if token in ("", "0"):
        return 0, []
    if token == "auto":
        return "auto", []
    try:
        return int(token), []
    except ValueError:
        raise ParseError(
            f"VLLM_DCU_FP8_PREFILL_CFG={raw!r}: int() raises at the first "
            "prefill GEMM (gfx906_fp8.py:_prefill_cfg)"
        ) from None


def _post_decode_cfg(value, raw, ctx):
    if value is None:
        return "auto", []
    token = str(value)
    if token[:1].lower() == "a":
        return 16, []
    number = _atoi(token)
    if 11 <= number <= 19 or 32 <= number <= 35:
        return number, []
    if number == 0:
        return "auto", []
    return 16, []


def _post_stream_max_m(value, raw, ctx):
    requested = int(value)
    effective = requested if requested > 0 else 0
    warnings: list[str] = []
    if raw is not None and effective != requested:
        _warn(
            warnings,
            "VLLM_DCU_FP8_STREAM_MAX_M: "
            f"requested={requested} effective=0 (no M limit)",
        )
    return effective, warnings


def _post_gemv_stream(value, raw, ctx):
    requested = int(value)
    if requested <= 0:
        effective = 0
    elif requested == 2:
        effective = 2
    else:
        effective = 1
    warnings: list[str] = []
    if raw is not None and requested not in (0, 1, 2):
        _warn(
            warnings,
            "VLLM_DCU_FP8_GEMV_STREAM: "
            f"requested={requested} effective={effective} "
            "(only 0/1/2 are meaningful)",
        )
    return effective, warnings


def _post_gemv_stream_rows(value, raw, ctx):
    requested = int(value)
    effective = requested if requested in (4, 8, 16, 32, 64) else 0
    warnings: list[str] = []
    if raw is not None and requested not in (0, 4, 8, 16, 32, 64):
        _warn(
            warnings,
            "VLLM_DCU_FP8_GEMV_STREAM_ROWS: "
            f"requested={requested} effective={effective} "
            "(0=auto, or one of 4/8/16/32/64)",
        )
    return effective, warnings


def _post_dp4x_guard(value, raw, ctx):
    token = str(value or "").strip().lower()
    if token in {"0", "false", "off", "no", "disable", "disabled"}:
        return "off", []
    if token in {"1", "true", "on", "yes", "force", "all"}:
        return "all", []
    return "auto", []


def _post_dp4x_min_warps(value, raw, ctx):
    requested = int(value)
    effective = max(1, min(requested, 32))
    warnings: list[str] = []
    if effective != requested:
        _warn(
            warnings,
            "VLLM_GFX906_DP4X_MIN_WARPS: "
            f"requested={requested} effective={effective} (clamped to [1,32])",
        )
    return effective, warnings


def _post_call_order(value, raw, ctx):
    return ("ab" if str(value or "").lower() == "ab" else "ba"), []


def _post_auto_model(value, raw, ctx):
    return ("auto" if raw is None else bool(value)), []


def _post_dead_knob(value, raw, ctx):
    warnings: list[str] = []
    if raw is not None:
        _warn(
            warnings,
            f"{ctx['name']}={raw!r} is set but has NO consumer left in "
            "vllm/, csrc/ or tools/dcu/ (dead knob; setting it cannot change "
            "the policy)",
        )
    return value, warnings


def _post_smallm(value, raw, ctx):
    token = str(value or "").strip().lower()
    if token in ("rocblas", "1", "true", "yes", "on"):
        return "rocblas", []
    warnings: list[str] = []
    if token not in ("", "0", "false", "no", "off"):
        _warn(
            warnings,
            f"VLLM_DCU_SMALLM_GEMM={raw!r} unrecognized -> off",
        )
    return "off", warnings


def _post_cxx_enabled(value, raw, ctx):
    # C++: env == nullptr || std::atoi(env) != 0
    return (True if raw is None else _atoi(raw) != 0), []


def _post_cxx_flag(value, raw, ctx):
    return (False if raw is None else _atoi(raw) != 0), []


POSTS = {
    "arch_auto": _post_arch_auto,
    "bf16_policy": _post_bf16_policy,
    "bf16_route": _post_bf16_route,
    "emu_tri": _post_emu_tri,
    "w8a16_tri": _post_w8a16_tri,
    "fp16_prefill_tri": _post_fp16_prefill_tri,
    "gemv_max_m": _post_gemv_max_m,
    "int8_gemv_max_m": _post_int8_gemv_max_m,
    "fp16_min_m": _post_fp16_min_m,
    "prefill_cfg": _post_prefill_cfg,
    "decode_cfg": _post_decode_cfg,
    "stream_max_m": _post_stream_max_m,
    "gemv_stream": _post_gemv_stream,
    "gemv_stream_rows": _post_gemv_stream_rows,
    "dp4x_guard": _post_dp4x_guard,
    "dp4x_min_warps": _post_dp4x_min_warps,
    "call_order": _post_call_order,
    "auto_model": _post_auto_model,
    "dead_knob": _post_dead_knob,
    "smallm": _post_smallm,
    "cxx_enabled": _post_cxx_enabled,
    "cxx_flag": _post_cxx_flag,
}


# ---------------------------------------------------------------------------
# Policy collection
# ---------------------------------------------------------------------------


def load_envs_module():
    """Execute vllm/envs.py standalone (no vllm/__init__, no torch, no HIP).

    Returns ``(module, error)``: the getters are the authoritative parsers for
    the names listed with ``envs=True``; importing them this way avoids the
    package ``__init__`` that pulls torch (and, on this host, the DTK runtime).
    """
    path = os.path.join(REPO_ROOT, "vllm", "envs.py")
    if not os.path.isfile(path):
        return None, f"missing {path}"
    try:
        spec = importlib.util.spec_from_file_location(
            "_dcu_policy_envs_standalone", path
        )
        if spec is None or spec.loader is None:
            return None, f"cannot build a loader spec for {path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, ""
    except Exception as exc:  # noqa: BLE001 - fall back to replicas
        return None, f"{type(exc).__name__}: {exc}"


def read_caps() -> tuple[dict, list[str]]:
    """Read the compile-time caps from source; report drift."""
    caps: dict[str, object] = {}
    warnings: list[str] = []
    for cap in CAPS:
        path = os.path.join(REPO_ROOT, cap["path"])
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            warnings.append(f"cannot read {cap['path']} for {cap['key']}: {exc}")
            caps[cap["key"]] = "?"
            continue
        match = re.search(cap["rx"], text, re.MULTILINE)
        if match is None:
            warnings.append(
                f"{cap['key']}: literal not found in {cap['path']} "
                f"(expected {cap['expect']}); update CAPS"
            )
            caps[cap["key"]] = "?"
            continue
        value = int(match.group(1))
        if value != cap["expect"]:
            warnings.append(
                f"{cap['key']}: source says {value}, CAPS expects "
                f"{cap['expect']} (drift; update the report and CAPS)"
            )
        caps[cap["key"]] = value
    return caps, warnings


def collect(use_envs: bool = True) -> dict:
    """Resolve every knob to its effective value.

    Returns a record consumed by ``render_text`` / ``render_json``:
    ``values`` (native effective), ``full`` (display strings), ``knobs``
    (per-name detail), ``caps``, ``warnings``, ``envs_error``, ``requested``.
    """
    envs_module, envs_error = (
        (None, "disabled") if not use_envs else (load_envs_module())
    )
    caps, caps_warnings = read_caps()
    warnings: list[str] = list(caps_warnings)
    values: dict[str, object] = {}
    details: dict[str, dict] = {}

    def resolve(entry: dict) -> None:
        name = entry["name"]
        raw = os.environ.get(name)
        kind = entry["kind"]
        params = entry.get("params")
        envs_value = None
        envs_error_for_entry = ""
        used_envs = False
        if envs_module is not None and entry.get("envs"):
            used_envs = True
            try:
                envs_value = getattr(envs_module, name)
            except Exception as exc:  # noqa: BLE001 - mirrors engine failure
                envs_error_for_entry = f"envs.py getter raised {exc!r}"
            else:
                try:
                    replica = _parse(kind, raw, entry["default"], params)
                except (ValueError, TypeError) as exc:
                    replica = f"!ERR({exc})"
                if replica != envs_value:
                    _warn(
                        warnings,
                        f"{name}: vllm/envs.py getter ({envs_value!r}) and the "
                        "fingerprint replica "
                        f"({replica!r}) disagree; getter wins, update KNOBS",
                    )
        if used_envs and not envs_error_for_entry:
            base = envs_value
        elif used_envs and envs_module is None:
            base = _parse(kind, raw, entry["default"], params)
        else:
            try:
                base = _parse(kind, raw, entry["default"], params)
            except (ValueError, TypeError) as exc:
                raise ParseError(str(exc)) from None
        ctx = {"name": name, "values": values, "caps": caps, "raw": raw}
        post = POSTS.get(entry.get("post", ""))
        try:
            effective, post_warnings = post(base, raw, ctx) if post else (base, [])
        except (ValueError, TypeError) as exc:
            raise ParseError(str(exc)) from None
        for text in post_warnings:
            _warn(warnings, text)
        if envs_error_for_entry:
            _warn(warnings, f"{name}: {envs_error_for_entry}")
        values[name] = effective
        details[name] = {
            "name": name,
            "kind": kind,
            "src": entry["src"],
            "note": entry.get("note", ""),
            "raw": raw,
            "parser": ("vllm/envs.py getter" if used_envs else "replica"),
            "effective": _jsonify(effective),
        }

    # The master emulation switch must be resolved before the three per-kernel
    # switches that compose with it.
    ordered = sorted(KNOBS, key=lambda e: e["name"] != "VLLM_DCU_USE_GFX906_EMU")
    for entry in ordered:
        try:
            resolve(entry)
        except ParseError as exc:
            # The engine itself would fail to start (or the getter would
            # raise at import): expose it instead of crashing the fingerprint.
            name = entry["name"]
            raw = os.environ.get(name)
            values[name] = "!ERR"
            details[name] = {
                "name": name,
                "kind": entry["kind"],
                "src": entry["src"],
                "note": entry.get("note", ""),
                "raw": raw,
                "parser": "replica",
                "effective": "!ERR",
            }
            _warn(
                warnings,
                f"{name}={raw!r}: parser raises ({exc}); the engine would "
                "fail at import/startup",
            )

    # Derived values that are not env knobs but change the policy.
    gemv_raw = os.environ.get("VLLM_DCU_FP8_GEMV_MAX_M")
    if gemv_raw is None:
        w8a16_max_m = int(caps.get("CAP_FP8_EMU_MS_MAX", 8))
    else:
        number = _atoi(gemv_raw)
        w8a16_max_m = (
            1 if number < 1 else min(number, int(caps.get("CAP_FP8_EMU_MS_MAX", 8)))
        )
    min_m_env = values.get("VLLM_DCU_FP16_PREFILL_MIN_M")
    default_min_m = (
        int(min_m_env) if isinstance(min_m_env, int) and min_m_env > 0 else 4096
    )
    derived = {
        "FP8_GEMV_MAX_M_W8A16": w8a16_max_m,
        "FP16_PREFILL_DEFAULT_MIN_M": default_min_m,
    }
    values.update(derived)

    # Unrecognized VLLM_DCU_* / VLLM_GFX906_* / ... variables that are set:
    # include them raw so a typo or a new knob still changes the fingerprint,
    # and warn because nothing resolves them.
    known = set(KNOB_NAMES) | set(derived)
    raw_extra: dict[str, str] = {}
    for name in sorted(os.environ):
        if name in known:
            continue
        if not name.startswith(
            ("VLLM_DCU_", "VLLM_GFX906_", "VLLM_QWEN35_", "VLLM_S24_", "VLLM_SPEC_")
        ):
            continue
        raw_extra[name] = os.environ[name]
        _warn(
            warnings,
            f"{name} is set but not modelled by this fingerprint; it is "
            "included raw and may silently change the policy. Add it to "
            "KNOBS (see --selftest coverage check).",
        )
    for name, raw in raw_extra.items():
        values[f"{name}@raw"] = raw

    requested = {name: os.environ.get(name) for name in KNOB_NAMES}

    full = {name: _format_value(value) for name, value in values.items()}
    fingerprint = _fingerprint(values, requested, caps)

    return {
        "fingerprint": fingerprint,
        "values": values,
        "full": full,
        "knobs": [details[name] for name in KNOB_NAMES if name in details],
        "caps": caps,
        "derived": derived,
        "requested": requested,
        "warnings": warnings,
        "envs_error": envs_error,
        "raw_extra": raw_extra,
    }


def _jsonify(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _format_value(value) -> str:
    """Render a key=value token: no whitespace, booleans as 1/0."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return "-"
    if isinstance(value, str):
        return value if value else "-"
    return str(value)


def _fingerprint(values: dict, requested: dict, caps: dict) -> str:
    payload = json.dumps(
        {
            "schema": SCHEMA_VERSION,
            "effective": {
                key: _jsonify(value) for key, value in sorted(values.items())
            },
            "requested": {key: value for key, value in sorted(requested.items())},
            "caps": {key: caps[key] for key in sorted(caps)},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "-C", REPO_ROOT, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        return sha or "unknown", bool(porcelain)
    except (OSError, subprocess.SubprocessError):
        return "unknown", False


def render_text(record: dict) -> list[str]:
    tokens = " ".join(f"{key}={record['full'][key]}" for key in sorted(record["full"]))
    return [f"FP {record['fingerprint']}", f"FP_FULL {tokens}"]


def render_json(record: dict, include_git: bool = True) -> str:
    payload = {
        "tool": "dcu_policy_fingerprint",
        "schema": SCHEMA_VERSION,
        "fingerprint": record["fingerprint"],
        "full": {key: record["full"][key] for key in sorted(record["full"])},
        "knobs": record["knobs"],
        "caps": record["caps"],
        "derived": record["derived"],
        "raw_extra": record["raw_extra"],
        "warnings": record["warnings"],
        "envs_source": (
            "vllm/envs.py (standalone, no torch)"
            if record["envs_error"] == ""
            else f"replicas (envs.py unavailable: {record['envs_error']})"
        ),
        "tree": REPO_ROOT,
    }
    if include_git:
        sha, dirty = _git_state()
        payload["git_sha"] = sha
        payload["git_dirty"] = dirty
    return json.dumps(payload, indent=1, sort_keys=True)


# ---------------------------------------------------------------------------
# Coverage scan + selftest
# ---------------------------------------------------------------------------


def scan_knob_names() -> tuple[set[str], list[str]]:
    """Every quoted policy-prefix name in the production tree."""
    names: set[str] = set()
    problems: list[str] = []
    skip_dirs = {
        ".git",
        ".refs",
        ".r3",
        ".venv",
        ".bench",
        "__pycache__",
        "node_modules",
    }
    for base in SCAN_DIRS:
        root = os.path.join(REPO_ROOT, base)
        if not os.path.isdir(root):
            problems.append(f"scan dir missing: {base}")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in skip_dirs and not name.startswith(".s")
            ]
            for filename in filenames:
                if not filename.endswith(SCAN_EXTS):
                    continue
                path = os.path.join(dirpath, filename)
                try:
                    with open(path, encoding="utf-8", errors="ignore") as handle:
                        text = handle.read()
                except OSError as exc:
                    problems.append(f"cannot read {path}: {exc}")
                    continue
                for match in _NAME_RX.finditer(text):
                    name = match.group(1)
                    if not name.endswith("_"):
                        names.add(name)
    return names, problems


def _selftest() -> int:
    failures: list[str] = []
    checks = 0

    def check(ok: bool, label: str, detail: str = "") -> None:
        nonlocal checks
        checks += 1
        if ok:
            print(f"[selftest] PASS {label}")
        else:
            failures.append(label)
            print(f"[selftest] FAIL {label}" + (f": {detail}" if detail else ""))

    # 1. registry shape, provenance paths.
    names = [entry["name"] for entry in KNOBS]
    check(len(names) == len(set(names)), "registry names are unique")
    bad_kinds = [
        entry["name"]
        for entry in KNOBS
        if entry["kind"]
        not in {
            "bool",
            "int",
            "int_e",
            "int_d",
            "int_o",
            "float_e",
            "str",
            "str_lower",
            "str_strip",
            "cxx_int",
        }
    ]
    check(not bad_kinds, "all entry kinds are known", str(bad_kinds))
    bad_posts = [
        entry["name"]
        for entry in KNOBS
        if entry.get("post") and entry["post"] not in POSTS
    ]
    check(not bad_posts, "all post hooks are known", str(bad_posts))
    missing_src = []
    for entry in KNOBS:
        path = entry["src"].split(":")[0]
        if not os.path.isfile(os.path.join(REPO_ROOT, path)):
            missing_src.append(entry["src"])
    check(not missing_src, "all entry sources exist", str(missing_src))

    # 2. envs.py getters load without torch, and agree with the replicas.
    envs_module, envs_error = load_envs_module()
    check(envs_module is not None, "vllm/envs.py loads standalone", envs_error)
    check("torch" not in sys.modules, "no torch import during collection")
    if envs_module is not None:
        disagree = []
        missing = []
        for entry in KNOBS:
            if not entry.get("envs"):
                continue
            try:
                want = getattr(envs_module, entry["name"])
            except AttributeError:
                missing.append(entry["name"])
                continue
            try:
                got = _parse(
                    entry["kind"],
                    os.environ.get(entry["name"]),
                    entry["default"],
                    entry.get("params"),
                )
            except (ValueError, TypeError) as exc:
                got = f"!ERR({exc})"
            if got != want:
                disagree.append(f"{entry['name']}: getter={want!r} repl={got!r}")
        check(
            not missing, "every envs=True name is a vllm/envs.py getter", str(missing)
        )
        check(not disagree, "replicas match vllm/envs.py getters", "; ".join(disagree))

    # 3. compile-time caps match their sources.
    caps, cap_warnings = read_caps()
    check(not cap_warnings, "compile-time caps match source", str(cap_warnings))
    if caps.get("CAP_GEMV_MS_MAX") != caps.get("CAP_FP8_EMU_MS_MAX"):
        print(
            "[selftest] WARN CAP_GEMV_MS_MAX "
            f"({caps.get('CAP_GEMV_MS_MAX')}) != CAP_FP8_EMU_MS_MAX "
            f"({caps.get('CAP_FP8_EMU_MS_MAX')}); the python dispatch clamp "
            "and the fp8_emu kernel disagree"
        )

    # 4. registry covers every quoted policy name in the tree.
    scanned, scan_problems = scan_knob_names()
    check(not scan_problems, "policy-tree scan is clean", str(scan_problems))
    unmodelled = sorted(scanned - set(names))
    stale = sorted(set(names) - scanned)
    check(
        not unmodelled,
        "every policy name in vllm/, csrc/, tools/dcu/ is modelled",
        str(unmodelled),
    )
    check(not stale, "every modelled name still appears in the tree", str(stale))

    # 5. determinism: two collections of the same environment are identical.
    first = collect()
    second = collect()
    check(
        first["fingerprint"] == second["fingerprint"]
        and first["full"] == second["full"],
        "same environment twice -> identical fingerprint",
    )
    base_fp = first["fingerprint"]

    saved = dict(os.environ)
    try:

        def with_env(**updates):
            for key, value in updates.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            return collect()

        def effective(record, name):
            return record["values"].get(name)

        # 6. a real knob change changes the fingerprint.
        arm1 = with_env(VLLM_DCU_FP8_GEMV_MAX_M="1")
        check(
            arm1["fingerprint"] != base_fp,
            "VLLM_DCU_FP8_GEMV_MAX_M=1 changes the fingerprint",
        )
        check(
            effective(arm1, "VLLM_DCU_FP8_GEMV_MAX_M") == 1,
            "GEMV_MAX_M=1 -> effective 1",
        )
        os.environ["VLLM_DCU_FP8_GEMV_MAX_M"] = "8"
        arm8 = collect()
        check(
            arm8["full"] == first["full"],
            "GEMV_MAX_M=8 resolves to the same effective values as unset",
        )
        os.environ.pop("VLLM_DCU_FP8_GEMV_MAX_M", None)
        reset = collect()
        check(
            reset["fingerprint"] == base_fp,
            "unsetting GEMV_MAX_M restores the fingerprint",
        )

        # 7. clamp visibility: 16 is clamped to the compiled kMsMax and warns.
        arm16 = with_env(VLLM_DCU_FP8_GEMV_MAX_M="16")
        check(
            effective(arm16, "VLLM_DCU_FP8_GEMV_MAX_M") == 8,
            "GEMV_MAX_M=16 -> effective 8 (clamped)",
        )
        warn16 = " ".join(arm16["warnings"])
        check(
            "requested=16 effective=8" in warn16,
            "GEMV_MAX_M=16 warns requested=16 effective=8",
            warn16,
        )
        check(
            arm16["full"] == arm8["full"]
            and arm16["fingerprint"] != arm8["fingerprint"],
            "clamped 16 is effective-equal to 8 but hashes differently "
            "(the hash pins the requested env verbatim)",
        )
        check(
            effective(arm16, "FP8_GEMV_MAX_M_W8A16") == 8,
            "W8A16-side GEMV threshold also clamps to 8",
        )
        os.environ.pop("VLLM_DCU_FP8_GEMV_MAX_M", None)

        # 8. INT8 threshold above the compiled range warns but is not clamped.
        arm_int8 = with_env(VLLM_DCU_INT8_GEMV_MAX_M="16")
        check(
            effective(arm_int8, "VLLM_DCU_INT8_GEMV_MAX_M") == 16,
            "INT8_GEMV_MAX_M=16 stays 16 (python does not clamp)",
        )
        check(
            "TORCH_CHECK" in " ".join(arm_int8["warnings"]),
            "INT8_GEMV_MAX_M=16 warns about the kernel TORCH_CHECK",
        )
        with_env(VLLM_DCU_INT8_GEMV_MAX_M=None)

        # 9. tri-state knobs normalise and warn.
        tri = with_env(VLLM_DCU_FP8_W8A16="banana")
        check(
            effective(tri, "VLLM_DCU_FP8_W8A16") == "0", "W8A16=banana -> effective '0'"
        )
        check("unrecognized" in " ".join(tri["warnings"]), "W8A16=banana warns")
        tri_ok = with_env(VLLM_DCU_FP8_W8A16="auto")
        check(effective(tri_ok, "VLLM_DCU_FP8_W8A16") == "auto", "W8A16=auto -> 'auto'")
        os.environ.pop("VLLM_DCU_FP8_W8A16", None)

        # 10. master-switch composition is effective, not raw.
        composed = with_env(VLLM_DCU_USE_GFX906_EMU="1", VLLM_DCU_USE_INT8_SDOT4="0")
        check(
            effective(composed, "VLLM_DCU_USE_INT8_SDOT4") is False,
            "USE_GFX906_EMU=1 + USE_INT8_SDOT4=0 -> INT8 off",
        )
        composed2 = with_env(VLLM_DCU_USE_INT8_SDOT4="yes")
        check(
            effective(composed2, "VLLM_DCU_USE_INT8_SDOT4") is True,
            "explicit per-kernel 'yes' wins over master",
        )
        with_env(VLLM_DCU_USE_GFX906_EMU=None, VLLM_DCU_USE_INT8_SDOT4=None)

        # 11. dead + unmodelled knobs are visible.  Build the typo name
        # dynamically so this file does not trip its own coverage scan.
        dead = with_env(VLLM_DCU_FP8_CVT="bits")
        check(
            "dead knob" in " ".join(dead["warnings"]),
            "VLLM_DCU_FP8_CVT is flagged as a dead knob",
        )
        os.environ.pop("VLLM_DCU_FP8_CVT", None)
        typo_name = "VLLM_DCU_FP8_GEMV_MAX_" + "M2"
        os.environ[typo_name] = "16"
        typo = collect()
        check(
            f"{typo_name}@raw" in typo["full"],
            "an unmodelled VLLM_DCU_* variable is included raw",
        )
        check(
            typo["fingerprint"] != base_fp,
            "an unmodelled variable changes the fingerprint",
        )
        os.environ.pop(typo_name, None)

        # 12. other clamps/normalisations.
        warps = with_env(VLLM_GFX906_DP4X_MIN_WARPS="99")
        check(
            effective(warps, "VLLM_GFX906_DP4X_MIN_WARPS") == 32,
            "DP4X_MIN_WARPS=99 -> 32",
        )
        warps_bad = with_env(VLLM_GFX906_DP4X_MIN_WARPS="lots")
        check(
            effective(warps_bad, "VLLM_GFX906_DP4X_MIN_WARPS") == 8,
            "DP4X_MIN_WARPS=invalid -> 8",
        )
        with_env(VLLM_GFX906_DP4X_MIN_WARPS=None)
        order = with_env(VLLM_QWEN35_CALL_ORDER="AB")
        check(
            effective(order, "VLLM_QWEN35_CALL_ORDER") == "ab",
            "QWEN35_CALL_ORDER=AB -> 'ab'",
        )
        with_env(VLLM_QWEN35_CALL_ORDER=None)
        rows = with_env(VLLM_DCU_FP8_GEMV_STREAM_ROWS="12")
        check(
            effective(rows, "VLLM_DCU_FP8_GEMV_STREAM_ROWS") == 0,
            "FP8_GEMV_STREAM_ROWS=12 -> 0 (only 4/8/16/32/64)",
        )
        with_env(VLLM_DCU_FP8_GEMV_STREAM_ROWS=None)
        rows_ok = with_env(VLLM_DCU_FP8_GEMV_STREAM_ROWS="16")
        check(
            effective(rows_ok, "VLLM_DCU_FP8_GEMV_STREAM_ROWS") == 16,
            "FP8_GEMV_STREAM_ROWS=16 -> 16",
        )
        with_env(VLLM_DCU_FP8_GEMV_STREAM_ROWS=None)
        stream = with_env(VLLM_DCU_FP8_GEMV_STREAM="-1")
        check(
            effective(stream, "VLLM_DCU_FP8_GEMV_STREAM") == 0,
            "FP8_GEMV_STREAM=-1 -> 0",
        )
        with_env(VLLM_DCU_FP8_GEMV_STREAM=None)

        restore = with_env()
        check(
            restore["fingerprint"] == base_fp,
            "restoring the environment restores the fingerprint",
        )
    finally:
        os.environ.clear()
        os.environ.update(saved)

    # 13. JSON round-trips and carries provenance.
    record = collect()
    payload = json.loads(render_json(record, include_git=False))
    check(
        payload["fingerprint"] == record["fingerprint"]
        and "FP8_GEMV_MAX_M_W8A16" in payload["derived"],
        "--json round-trips with derived values",
    )
    check(
        all(knob["src"] for knob in payload["knobs"]),
        "--json carries a source for every knob",
    )
    lines = render_text(record)
    check(
        len(lines) == 2
        and lines[0].startswith("FP ")
        and lines[1].startswith("FP_FULL "),
        "text output is exactly two lines (FP / FP_FULL)",
    )
    check(
        len(lines[0].split()) == 2 and len(lines[0].split()[1]) == 16,
        "fingerprint is sha256[:16]",
    )

    print()
    if failures:
        print(
            f"[selftest] {checks - len(failures)}/{checks} checks passed; "
            f"{len(failures)} FAILED"
        )
        for label in failures:
            print(f"[selftest]   FAILED: {label}")
        return 1
    print(f"[selftest] all {checks} checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print the effective Z100 DCU serving-policy fingerprint (BUILD WP0.4)."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the structured record instead of the two text lines",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run the zero-card acceptance checks and exit",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="prepend this string to every output line (log integration)",
    )
    parser.add_argument(
        "--no-envs-module",
        action="store_true",
        help="ignore vllm/envs.py and use the built-in replicas",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="omit git SHA/dirty fields from --json",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        code = _selftest()
        return code

    started = time.time()
    try:
        record = collect(use_envs=not args.no_envs_module)
    except Exception as exc:  # noqa: BLE001 - never break the launcher
        print(
            f"{args.prefix}[fingerprint] ERROR {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    for warning in record["warnings"]:
        print(f"{args.prefix}[fingerprint] WARN {warning}", file=sys.stderr)
    if record["envs_error"]:
        print(
            f"{args.prefix}[fingerprint] WARN vllm/envs.py getters "
            f"unavailable ({record['envs_error']}); used replicas",
            file=sys.stderr,
        )
    if args.json:
        print(render_json(record, include_git=not args.no_git))
    else:
        for line in render_text(record):
            print(f"{args.prefix}{line}")
    if os.environ.get("DCU_FP_DEBUG"):
        print(
            f"{args.prefix}[fingerprint] collected {len(record['full'])} "
            f"values in {time.time() - started:.3f}s",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
