#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify whether bfloat16 is usable on a Hygon DCU (Z100 / gfx906).

Run this on the DCU host, inside the DTK environment::

    source tools/dcu/env.sh
    python3 tools/dcu/check_bf16.py

Why a probe instead of an assumption: gfx906 (Vega20) has no bf16 ISA. DTK's
assembler rejects both ``v_cvt_f32_bf16`` ("invalid instruction") and
``v_dot2_f32_bf16`` ("instruction not supported on this GPU") for gfx906 while
accepting ``v_pk_fma_f16`` and ``v_dot2_f32_f16``. Every bf16 op is therefore
emulated in software, which can still be numerically correct -- so vLLM probes
the device instead of hard-coding support.

Notes:
* ``torch.cuda.is_bf16_supported()`` returns True on all ROCm builds (see the
  ROCm/pytorch commit "[ROCm] torch.cuda.is_bf16_supported() returns True"), so
  it is reported for completeness but is *not* used as a decision input.
* ``torch.set_default_dtype(torch.bfloat16)`` is not a capability switch: it
  only changes the dtype of newly created float tensors. The vLLM lever is
  ``--dtype bfloat16`` together with the platform's ``supported_dtypes``.
"""

from __future__ import annotations

import sys
import time

import torch


def _device() -> torch.device:
    index = torch.cuda.current_device()
    return torch.device("cuda", index)


def _fmt_ok(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def environment_report(device: torch.device) -> None:
    print("== environment ==")
    print(f"torch                 : {torch.__version__}")
    print(f"torch.version.hip     : {torch.version.hip}")
    print(f"torch.version.cuda    : {torch.version.cuda}")
    print(f"device count          : {torch.cuda.device_count()}")

    try:
        from vllm.platforms.hygon_dtk import (
            dtk_version,
            gcn_arch_from_kfd,
            is_hygon_dtk,
        )

        print(f"hygon_dtk detected    : {is_hygon_dtk()}")
        print(f"DTK version (sysfs)   : {dtk_version()}")
        print(f"gfx arch (KFD sysfs)  : {gcn_arch_from_kfd(device.index)}")
    except Exception as exc:  # vllm may be unbuilt; the probe still works
        print(f"vllm.platforms import : skipped ({exc})")

    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        arch = getattr(prop, "gcnArchName", "n/a")
        print(f"  device[{index}]: {arch!r} sm_count={prop.multi_processor_count}")

    try:
        print(f"is_bf16_supported()   : {torch.cuda.is_bf16_supported()}")
    except Exception as exc:
        print(f"is_bf16_supported()   : error ({exc})")
    print(
        "note                  : is_bf16_supported() is True on every ROCm "
        "build and does not reflect gfx906's missing bf16 ISA."
    )


def check_storage_round_trip(device: torch.device) -> tuple[bool, float]:
    torch.manual_seed(0)
    a = torch.randn(128, 256, device=device, dtype=torch.float32)
    err = (a.to(torch.bfloat16).to(torch.float32) - a).abs().max().item()
    # bf16 keeps 8 mantissa bits -> ~2^-9 relative rounding.
    return err <= 0.01 * a.abs().max().item(), err


def check_gemm(device: torch.device, dtype: torch.dtype) -> tuple[bool, float, float]:
    torch.manual_seed(0)
    a = torch.randn(512, 512, device=device, dtype=torch.float32)
    b = torch.randn(512, 512, device=device, dtype=torch.float32)
    a_d, b_d = a.to(dtype), b.to(dtype)
    reference = a_d.to(torch.float32) @ b_d.to(torch.float32)
    observed = (a_d @ b_d).to(torch.float32)
    if not bool(torch.isfinite(observed).all()):
        return False, float("inf"), float("nan")
    err = (observed - reference).abs().max().item()
    scale = reference.abs().max().item()
    rel = err / max(scale, 1e-6)
    return rel <= 0.05, err, rel


def check_linear(device: torch.device) -> tuple[bool, float]:
    """The shape vLLM actually uses: token-major linear with bf16 weights."""
    torch.manual_seed(0)
    x = torch.randn(8, 1024, device=device, dtype=torch.float32)
    w = torch.randn(2048, 1024, device=device, dtype=torch.float32)
    x_bf, w_bf = x.to(torch.bfloat16), w.to(torch.bfloat16)
    reference = torch.nn.functional.linear(
        x_bf.to(torch.float32), w_bf.to(torch.float32)
    )
    observed = torch.nn.functional.linear(x_bf, w_bf).to(torch.float32)
    err = (observed - reference).abs().max().item()
    scale = reference.abs().max().item()
    rel = err / max(scale, 1e-6)
    return rel <= 0.05, rel


def check_softmax(device: torch.device) -> tuple[bool, float]:
    torch.manual_seed(0)
    x = torch.randn(64, 4096, device=device, dtype=torch.float32).to(torch.bfloat16)
    reference = torch.softmax(x.to(torch.float32), dim=-1)
    observed = torch.softmax(x, dim=-1).to(torch.float32)
    err = (observed - reference).abs().max().item()
    return err <= 0.05, err


def time_gemm(device: torch.device, dtype: torch.dtype, iters: int = 30) -> float:
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, device=device, dtype=torch.float32).to(dtype)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float32).to(dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000 / iters
    tflops = 2 * 4096**3 / (ms / 1000) / 1e12
    print(f"  matmul 4096^3 {str(dtype):>16}: {ms:8.2f} ms  ({tflops:6.2f} TFLOP/s)")
    return ms


def main() -> int:
    if not torch.cuda.is_available():
        print("No GPU is visible to torch; cannot probe bf16.", file=sys.stderr)
        return 2

    device = _device()
    environment_report(device)

    print("\n== bf16 correctness ==")
    ok_storage, storage_err = check_storage_round_trip(device)
    print(
        f"[{_fmt_ok(ok_storage)}] storage round-trip        "
        f"max_abs_err={storage_err:.6f}"
    )

    ok_gemm, gemm_err, gemm_rel = check_gemm(device, torch.bfloat16)
    print(
        f"[{_fmt_ok(ok_gemm)}] bf16 GEMM vs fp32 ref     max_abs_err={gemm_err:.6f} "
        f"rel={gemm_rel * 100:.3f}%"
    )

    ok_linear, linear_rel = check_linear(device)
    print(
        f"[{_fmt_ok(ok_linear)}] linear(bf16 x, bf16 w)      "
        f"rel={linear_rel * 100:.3f}%"
    )

    ok_softmax, softmax_err = check_softmax(device)
    print(
        f"[{_fmt_ok(ok_softmax)}] softmax(bf16)              "
        f"max_abs_err={softmax_err:.6f}"
    )

    print("\n== performance reference (same shapes) ==")
    fp16_ms = time_gemm(device, torch.float16)
    bf16_ms = time_gemm(device, torch.bfloat16)

    passed = ok_storage and ok_gemm and ok_linear and ok_softmax

    print("\n== verdict ==")
    print(f"bf16 numerics : {'usable' if passed else 'NOT usable'}")
    if passed:
        speed = fp16_ms / bf16_ms if bf16_ms else float("nan")
        print(f"bf16 vs fp16  : {speed:.2f}x (matmul)")
        print(
            "recommendation: keep fp16 for throughput; enable bf16 only when a "
            "model requires it:\n"
            "    export VLLM_DCU_BF16=1   # force-enable on gfx906\n"
            "    vllm serve <model> --dtype bfloat16"
        )
        print(
            "note          : vLLM's own probe is the same check; with "
            "VLLM_DCU_BF16=auto, bf16 is offered automatically when this "
            "script passes."
        )
    else:
        print(
            "recommendation: keep fp16 (--dtype float16). Forcing bf16 via\n"
            "    VLLM_DCU_BF16=1\n"
            "is possible for experiments but numerics are not reliable here."
        )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
