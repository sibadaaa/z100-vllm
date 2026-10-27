#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the ``_rocm_C`` extension built for a Hygon DCU (Z100 / gfx906).

Run inside the DTK container, after the extension is built::

    source tools/dcu/env.sh
    python3 tools/dcu/check_rocm_c_ops.py

What it verifies:

1. ``vllm._rocm_C`` imports and which ops it registered. gfx906 excludes
   ``csrc/rocm/skinny_gemms*.cu`` from the build (no MFMA), so
   ``LLMM1``/``wvSplitK``/``wvSplitKQ``/``wvSplitKrc``/``wvSplitK_int4_g``
   are expected to be absent and reported as SKIP, not FAIL; the check
   therefore also passes on a skinny-enabled build.
2. ``paged_attention`` (always built) is compared against a plain-torch
   attention reference on a small case, but only when
   ``use_rocm_custom_paged_attention()`` selects it: the kernel is MFMA-only
   and its body is ``UNREACHABLE_CODE`` on gfx906 (verified on hardware), so
   running it there aborts the process.

Exit codes follow ``tools/dcu/check_bf16.py``: 0 = all executed checks pass,
1 = a check failed, 2 = no usable GPU / ``_rocm_C`` missing.

``import torch`` must come before ``import vllm`` on DTK; see the S4 archive
(``docs/result/z100-04-build-system-dtk-result.md`` §3.11).
"""

from __future__ import annotations

import math
import sys

import torch  # noqa: F401  (must precede vllm on DTK)

import vllm._custom_ops as ops  # noqa: E402
import vllm._rocm_C  # noqa: F401,E402
from vllm.platforms.rocm import (  # noqa: E402
    use_rocm_custom_paged_attention,
)


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _report(name: str, status: str, detail: str) -> None:
    print(f"{status:<4} {name:<18} {detail}")


def _registered_ops() -> list[str]:
    return sorted(
        s.name
        for s in torch._C._jit_get_all_schemas()
        if s.name.startswith("_rocm_C::")
    )


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    got = got.float()
    ref = ref.float()
    return ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()


def check_llmm1(cu_count: int) -> tuple[bool, str]:
    """LLMM1(weight[M, K], x[1, K], rows_per_block) == x @ weight.T."""
    m, k = 256, 256
    weight = torch.randn(m, k, dtype=torch.float16, device=_device())
    x = torch.randn(1, k, dtype=torch.float16, device=_device())
    out = ops.LLMM1(weight, x, 4)
    ref = x @ weight.t()
    err = _rel_err(out, ref)
    return err < 5e-3, f"shape={tuple(out.shape)} rel_err={err:.3e}"


def check_wvsplitk(cu_count: int) -> tuple[bool, str]:
    """wvSplitK(a[M, K], b[N, K], bias, CuCount) == a @ b.T.

    The kernel dispatches on N and only implements N in 1..5
    (csrc/rocm/skinny_gemms.cu:1310-1327).
    """
    m, k, n = 32, 64, 4
    a = torch.randn(m, k, dtype=torch.float16, device=_device())
    b = torch.randn(n, k, dtype=torch.float16, device=_device())
    out = ops.wvSplitK(a, b, cu_count)
    ref = a @ b.t()
    err = _rel_err(out, ref)
    return err < 5e-3, f"shape={tuple(out.shape)} rel_err={err:.3e}"


def check_wvsplitkq(cu_count: int) -> tuple[bool, str]:
    """wvSplitKQ: fp8 weight/activation GEMM with per-tensor scales.

    N is the activation batch and is limited to 1..4
    (csrc/rocm/skinny_gemms.cu:2525-2538).
    """
    m, k, n = 32, 64, 4
    fp8 = (
        torch.float8_e4m3fn
        if hasattr(torch, "float8_e4m3fn")
        else torch.float8_e4m3fnuz
    )
    a = (torch.randn(n, k, device=_device()) * 0.1).to(fp8)
    b = (torch.randn(m, k, device=_device()) * 0.1).to(fp8)
    scale_a = torch.tensor([0.5], dtype=torch.float32, device=_device())
    scale_b = torch.tensor([0.25], dtype=torch.float32, device=_device())
    out = ops.wvSplitKQ(a, b, torch.float16, scale_a, scale_b, cu_count)
    ref = (b.float() @ a.float().t()) * scale_b * scale_a
    err = _rel_err(out, ref)
    return err < 2e-2, f"shape={tuple(out.shape)} rel_err={err:.3e}"


def check_paged_attention() -> tuple[bool, str]:
    """paged_attention vs. plain-torch attention on 16 tokens, block_size 16."""
    torch.manual_seed(0)
    device = _device()
    num_seqs, num_heads, head_size = 1, 2, 64
    num_kv_heads, block_size, seq_len = 2, 16, 16
    num_blocks, x = 1, 16 // 2  # x = 16 bytes / sizeof(fp16)
    scale = 1.0 / math.sqrt(head_size)

    q = torch.randn(num_seqs, num_heads, head_size, dtype=torch.float16, device=device)
    k = torch.randn(num_kv_heads, seq_len, head_size, dtype=torch.float16, device=device)
    v = torch.randn(num_kv_heads, seq_len, head_size, dtype=torch.float16, device=device)

    # key_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    key_cache = (
        k.unsqueeze(0)
        .reshape(num_blocks, num_kv_heads, head_size // x, x, block_size)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    # value_cache: [num_blocks, num_kv_heads, head_size, block_size]
    value_cache = v.transpose(-2, -1).unsqueeze(0).contiguous()

    out = torch.empty(num_seqs, num_heads, head_size, dtype=torch.float16, device=device)
    exp_sums = torch.empty(num_seqs, num_heads, 1, dtype=torch.float32, device=device)
    max_logits = torch.empty_like(exp_sums)
    tmp_out = torch.empty(
        num_seqs, num_heads, 1, head_size, dtype=torch.float16, device=device
    )
    block_tables = torch.zeros(num_seqs, 1, dtype=torch.int32, device=device)
    seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)
    k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    ops.paged_attention_rocm(
        out,
        exp_sums,
        max_logits,
        tmp_out,
        q,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        None,
        block_size,
        seq_len,
        None,
        "auto",
        k_scale,
        v_scale,
        None,
        "f16",
    )

    ref = torch.empty_like(out)
    for h in range(num_heads):
        kv = h % num_kv_heads
        scores = (q[0, h].float() @ k[kv].float().t()) * scale
        probs = torch.softmax(scores, dim=-1)
        ref[0, h] = (probs @ v[kv].float()).to(torch.float16)

    err = _rel_err(out, ref)
    return err < 5e-3, f"rel_err={err:.3e} max_logit={max_logits.max().item():.3f}"


def main() -> int:
    if not torch.cuda.is_available():
        print("NO_GPU: torch.cuda.is_available() is False")
        return 2

    props = torch.cuda.get_device_properties(0)
    print(f"torch      : {torch.__version__} hip={torch.version.hip}")
    print(f"device     : {torch.cuda.get_device_name(0)} arch={props.gcnArchName}")
    print(f"skinny gemm: ROCM_SKINNY_GEMM_ENABLED={ops.ROCM_SKINNY_GEMM_ENABLED}")
    print(f"registered : {' '.join(_registered_ops())}")
    print()

    cu_count = min(torch.cuda.get_device_properties(0).multi_processor_count, 4)
    failed: list[str] = []

    for name, op in (
        ("LLMM1", ops.LLMM1),
        ("wvSplitK", ops.wvSplitK),
        ("wvSplitKQ", ops.wvSplitKQ),
    ):
        if not hasattr(torch.ops._rocm_C, name):
            _report(name, "SKIP", "not registered in this build (skinny GEMM excluded)")
            continue
        try:
            ok, detail = {
                "LLMM1": check_llmm1,
                "wvSplitK": check_wvsplitk,
                "wvSplitKQ": check_wvsplitkq,
            }[name](cu_count)
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            _report(name, "FAIL", f"{type(exc).__name__}: {exc}")
            failed.append(name)
            continue
        _report(name, "OK" if ok else "FAIL", detail)
        if not ok:
            failed.append(name)

    # `_rocm_C::paged_attention` only implements the MFMA kernels; on gfx906
    # (and any arch outside __HIP__GFX9__) csrc/rocm/attention.cu compiles the
    # kernel bodies to UNREACHABLE_CODE, so vLLM routes decode attention to
    # Triton there and this op must not be called. Exercising it would trip a
    # device-side assertion and take the process down with a VM fault, so it is
    # reported rather than run.
    if use_rocm_custom_paged_attention(torch.float16, 64, 16, 1, 16, 0, "auto"):
        try:
            ok, detail = check_paged_attention()
        except Exception as exc:  # noqa: BLE001
            _report("paged_attention", "FAIL", f"{type(exc).__name__}: {exc}")
            failed.append("paged_attention")
        else:
            _report("paged_attention", "OK" if ok else "FAIL", detail)
            if not ok:
                failed.append("paged_attention")
    else:
        _report(
            "paged_attention",
            "SKIP",
            "use_rocm_custom_paged_attention() is False on this arch "
            "(MFMA-only kernel, see csrc/rocm/attention.cu); registration only",
        )

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
