# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S8b probe: is block-scaled FP8 numerically usable on gfx906?

S8 found that ``TritonFp8BlockScaledMMKernel.is_supported()`` returns True on
gfx906 (it only checks ``is_cuda_alike()``, which is also True for ROCm) while
every per-tensor/per-channel FP8 kernel is rejected. gfx906 has no native FP8
instructions, so the question is whether the *emulated* triton path produces
usable numbers. This script answers it with independent pieces of evidence:

1. selection evidence -- what the real selector picks, without a model;
2. numeric evidence -- the exact triton kernels the selected kernel dispatches
   to, compared against the upstream fp32 dequantized reference
   (``tests.kernels.quant_utils.native_w8a8_block_matmul``), over a shape /
   dtype / dynamic-range sweep;
3. the input quantization kernel (``per_token_group_quant_fp8``) on its own;
4. ``Fp8BlockScaledMMLinearKernel.apply_weights`` end to end;
5. an order-of-magnitude timing, to show what "emulated" costs.

Nothing is monkey-patched, so the printed decisions are the ones the engine
would make at model-load time.

    python tools/dcu/s8b_fp8_block_probe.py

Exit code 0 iff every selection expectation and numeric tolerance in the
probe holds; non-zero means the FP8 verdict in
docs/result/z100-08b-gating-and-requirements-result.md must be revisited.
"""

from __future__ import annotations

import traceback


def _show(label: str, fn) -> None:
    try:
        print(f"{label} -> {fn()}")
    except Exception as exc:
        print(f"{label} -> RAISED {type(exc).__name__}: {exc}")
        for line in traceback.format_exc().strip().splitlines()[-5:]:
            print(f"    | {line}")


def error_metrics(out, ref) -> str:
    import torch

    out64 = out.to(torch.float64)
    ref64 = ref.to(torch.float64)
    finite_out = bool(torch.isfinite(out64).all())
    denom = torch.mean(torch.abs(ref64)).item()
    max_abs = torch.max(torch.abs(out64 - ref64)).item()
    if denom:
        rel = torch.mean(torch.abs(out64 - ref64)).item() / denom
    else:
        rel = 0.0 if max_abs == 0.0 else float("inf")
    return (
        f"finite={finite_out} mean_rel_err={rel:.3e} max_abs_err={max_abs:.3e} "
        f"out_absmax={torch.max(torch.abs(out64)).item():.3e} "
        f"ref_absmax={torch.max(torch.abs(ref64)).item():.3e}"
    )


def _vllm_config_ctx():
    from vllm.config import VllmConfig, set_current_vllm_config

    return set_current_vllm_config(VllmConfig())


def section_selection() -> bool:
    import torch

    from vllm.model_executor.kernels.linear import (
        _POSSIBLE_FP8_BLOCK_KERNELS,
        _POSSIBLE_FP8_KERNELS,
        _POSSIBLE_WFP8A16_KERNELS,
        init_fp8_linear_kernel,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        QuantKey,
        ScaleDesc,
    )
    from vllm.platforms import PlatformEnum, current_platform

    ok = True
    print("=== 1. FP8 kernel selection surface on this platform ===")
    print(
        f"platform = {current_platform._enum}  "
        f"fp8_dtype = {current_platform.fp8_dtype()}"
    )
    for name, table in (
        ("_POSSIBLE_FP8_KERNELS[ROCM] (per-tensor/per-channel)", _POSSIBLE_FP8_KERNELS),
        (
            "_POSSIBLE_FP8_BLOCK_KERNELS[ROCM] (block-scaled)",
            _POSSIBLE_FP8_BLOCK_KERNELS,
        ),
        ("_POSSIBLE_WFP8A16_KERNELS[ROCM]", _POSSIBLE_WFP8A16_KERNELS),
    ):
        kernels = table.get(PlatformEnum.ROCM, [])
        print(f"\n{name} = {[k.__name__ for k in kernels] or '<empty>'}")
        for k in kernels:
            _show(f"  is_supported({k.__name__})", k.is_supported)

    print("\n=== 2. what init_fp8_linear_kernel() actually returns ===")
    print("  (inside set_current_vllm_config(VllmConfig()), as tests do)")
    fp8 = current_platform.fp8_dtype()
    block_key_w = QuantKey(
        dtype=fp8,
        scale=ScaleDesc(torch.float32, False, GroupShape(128, 128)),
    )
    block_key_a = QuantKey(
        dtype=fp8,
        scale=ScaleDesc(torch.float32, False, GroupShape(1, 128)),
    )
    per_tensor_w = QuantKey(
        dtype=fp8,
        scale=ScaleDesc(torch.float32, False, GroupShape.PER_TENSOR),
    )
    per_channel_w = QuantKey(
        dtype=fp8,
        scale=ScaleDesc(torch.float32, False, GroupShape.PER_CHANNEL),
    )
    per_token_a = QuantKey(
        dtype=fp8,
        scale=ScaleDesc(torch.float32, False, GroupShape.PER_TOKEN),
    )
    per_tensor_a_static = QuantKey(
        dtype=fp8,
        scale=ScaleDesc(torch.float32, True, GroupShape.PER_TENSOR),
    )
    # (label, act key, weight key, must_select | must_raise)
    cases = (
        ("block-scaled W8A8 (w=[128,128] blocks)", block_key_a, block_key_w,
         "TritonFp8BlockScaledMMKernel"),
        ("per-tensor W8A8 (dynamic act)", per_token_a, per_tensor_w, None),
        ("per-tensor W8A8 (static act)", per_tensor_a_static, per_tensor_w, None),
        ("per-channel W8A8", per_token_a, per_channel_w, None),
    )
    with _vllm_config_ctx():
        for label, akey, wkey, expected in cases:
            try:
                got = type(
                    init_fp8_linear_kernel(
                        activation_quant_key=akey,
                        weight_quant_key=wkey,
                        input_dtype=torch.float16,
                        out_dtype=torch.float16,
                        weight_shape=(256, 512),
                    )
                ).__name__
            except Exception as exc:
                got = f"RAISED {type(exc).__name__}: {exc}"
            print(f"init_fp8_linear_kernel[{label}] -> {got}")
            if expected is not None and got != expected:
                print(f"  !! expected {expected}")
                ok = False

    if ok:
        print("  selection expectations: OK (only block-scaled is accepted)")
    else:
        print("  selection expectations: FAILED")
    return ok


def _make_operands(M, N, K, bn, bk, dev, fp8, scale_a=2.0, scale_w=2.0, seed=0):
    import torch

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    torch.manual_seed(seed)
    a = torch.randn(M, K, dtype=torch.float16, device=dev) * scale_a
    w = torch.randn(N, K, dtype=torch.float32, device=dev) * scale_w
    fp8_max = torch.finfo(fp8).max
    # Pad to whole blocks so N / K need not be multiples of block_n / block_k
    # (the triton kernel masks the tail; the reference clamps it). The padded
    # region gets zero weights, so its block scale is 1e-12 and contributes 0.
    nbn = (N + bn - 1) // bn
    kbn = (K + bk - 1) // bk
    w_pad = w.new_zeros(nbn * bn, kbn * bk)
    w_pad[:N, :K] = w
    wb = w_pad.view(nbn, bn, kbn, bk)
    w_scale = (wb.abs().amax(dim=(1, 3)) / fp8_max).clamp(min=1e-12)
    w_q = (wb / w_scale[:, None, :, None]).view(nbn * bn, kbn * bk)[:N, :K].to(fp8)
    a_q, a_scale = per_token_group_quant_fp8(a, bk, dtype=fp8)
    return a, w, a_q, a_scale, w_q, w_scale


def section_numeric() -> bool:
    import torch

    from tests.kernels.quant_utils import native_w8a8_block_matmul
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )

    print("\n=== 3. numeric sweep: triton block-scaled FP8 GEMM vs fp32 reference ===")
    print("  (gfx906 has no FP8 unit; this exercises the emulated triton path)")
    dev = "cuda"
    fp8 = torch.float8_e4m3fn

    # (M, N, K, block_n, block_k, act_scale, w_scale, out_dtype, tag)
    cases = [
        (1, 512, 512, 128, 128, 2.0, 2.0, torch.float16, "decode M=1"),
        (7, 128, 256, 64, 128, 2.0, 2.0, torch.float16, "M=7, blk_n=64"),
        (16, 256, 256, 128, 128, 2.0, 2.0, torch.float16, "small square"),
        (64, 512, 512, 128, 128, 2.0, 2.0, torch.float16, "square"),
        (128, 576, 512, 128, 128, 2.0, 2.0, torch.float16, "N=576 (tail mask)"),
        (128, 256, 512, 128, 128, 2.0, 2.0, torch.float16, "rect N<K"),
        (256, 1024, 256, 128, 128, 2.0, 2.0, torch.float16, "rect N>K"),
        (512, 512, 1024, 128, 128, 2.0, 2.0, torch.float16, "K=1024"),
        (2048, 512, 512, 128, 128, 2.0, 2.0, torch.float16, "prefill M=2048"),
        (64, 512, 7168, 128, 128, 2.0, 2.0, torch.float16, "long K=7168"),
        (1, 18432, 512, 128, 128, 2.0, 2.0, torch.float16, "wide N=18432, M=1"),
        (64, 7168, 2048, 128, 128, 2.0, 2.0, torch.float16, "FFN-like"),
        (64, 512, 512, 128, 128, 2.0, 2.0, torch.bfloat16, "out=bf16"),
        (64, 512, 512, 128, 128, 2.0, 2.0, torch.float32, "out=fp32"),
        (64, 512, 512, 128, 128, 1e-3, 1e-3, torch.float16, "tiny dynamic range"),
        (64, 512, 512, 128, 128, 1e2, 1e2, torch.float32, "wide dynamic range"),
        (64, 512, 512, 128, 128, 0.0, 2.0, torch.float16, "all-zero activation"),
    ]
    failures = []
    for M, N, K, bn, bk, sa, sw, odt, tag in cases:
        try:
            a, w, a_q, a_scale, w_q, w_scale = _make_operands(
                M, N, K, bn, bk, dev, fp8, sa, sw
            )
            ref = native_w8a8_block_matmul(a_q, w_q, a_scale, w_scale, [bn, bk], odt)
            ref_exact = (a.to(torch.float32) @ w.t()).to(odt)
            out = w8a8_triton_block_scaled_mm(a_q, w_q, a_scale, w_scale, [bn, bk], odt)
            out64 = out.to(torch.float64)
            ref64 = ref.to(torch.float64)
            denom = torch.mean(torch.abs(ref64)).item()
            rel = (
                torch.mean(torch.abs(out64 - ref64)).item() / denom if denom else 0.0
            )
            if (not bool(torch.isfinite(out64).all())) or rel > 1e-2:
                failures.append((tag, rel))
            print(
                f"  [{tag:22s}] M={M:4d} N={N:4d} K={K:4d} blk=[{bn},{bk}] "
                f"out={str(odt).split('.')[-1]:8s}\n"
                f"       qx/qw are already fp8; kernel vs fp32-dequant: "
                f"{error_metrics(out, ref)}\n"
                f"       dequant-vs-ideal-fp32 (quantization error floor): "
                f"{error_metrics(ref, ref_exact)}"
            )
        except Exception as exc:
            failures.append((tag, f"RAISED {type(exc).__name__}: {exc}"))
            print(f"  [{tag:22s}] M={M} N={N} K={K} RAISED {type(exc).__name__}: {exc}")
            for line in traceback.format_exc().strip().splitlines()[-8:]:
                print(f"    | {line}")

    print(f"\n  shapes with rel_err > 1e-2 or non-finite output: {failures or 'NONE'}")
    return not failures


def section_input_quant() -> bool:
    import torch

    from tests.kernels.quant_utils import native_per_token_group_quant_fp8
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    print("\n=== 4. input quantization kernel (per_token_group_quant_fp8) ===")
    print("  QuantFP8.forward_hip -> forward_cuda -> this triton kernel on ROCm")
    fp8 = torch.float8_e4m3fn
    failures = []
    for M, K, gs in ((7, 256, 128), (64, 512, 128), (1, 1024, 128), (128, 384, 64)):
        try:
            torch.manual_seed(0)
            x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 3.0
            q, s = per_token_group_quant_fp8(x, gs, dtype=fp8)
            nq, ns = native_per_token_group_quant_fp8(x, gs, dtype=fp8)
            dq = q.to(torch.float32) * s.to(torch.float32).repeat_interleave(gs, 1)[:K]
            ndq = (
                nq.to(torch.float32) * ns.to(torch.float32).repeat_interleave(gs, 1)[:K]
            )
            xd = x.to(torch.float32)
            rel = torch.mean(torch.abs(dq - xd)).item() / torch.mean(
                torch.abs(xd)
            ).item()
            rel_vs_native = torch.mean(torch.abs(dq - ndq)).item() / torch.mean(
                torch.abs(ndq)
            ).item()
            if rel_vs_native > 1e-3:
                failures.append((M, K, gs, rel_vs_native))
            print(
                f"  M={M:4d} K={K:4d} gs={gs:4d} quant_err_vs_fp32={rel:.3e} "
                f"triton_vs_native_quant={rel_vs_native:.3e}"
            )
        except Exception as exc:
            failures.append((M, K, gs, f"RAISED {type(exc).__name__}: {exc}"))
            print(f"  M={M} K={K} gs={gs} RAISED {type(exc).__name__}: {exc}")
    print(f"  quant shapes disagreeing with native by > 1e-3: {failures or 'NONE'}")
    return not failures


def section_full_kernel() -> bool:
    import torch

    print("\n=== 5. full Fp8BlockScaledMMLinearKernel.apply_weights() path ===")
    try:
        from tests.kernels.quant_utils import native_w8a8_block_matmul
        from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            GroupShape,
            QuantKey,
            ScaleDesc,
        )
        from vllm.platforms import current_platform

        fp8 = current_platform.fp8_dtype()
        bn = bk = 128
        N, K = 512, 512
        M = 64
        with _vllm_config_ctx():
            kernel = init_fp8_linear_kernel(
                activation_quant_key=QuantKey(
                    dtype=fp8,
                    scale=ScaleDesc(torch.float32, False, GroupShape(1, bk)),
                ),
                weight_quant_key=QuantKey(
                    dtype=fp8,
                    scale=ScaleDesc(torch.float32, False, GroupShape(bn, bk)),
                ),
                input_dtype=torch.float16,
                out_dtype=torch.float16,
                weight_shape=(N, K),
            )
            print(f"  selected kernel = {type(kernel).__name__}")

            # Minimal layer carrying the attribute names the kernel reads
            # (Fp8LinearMethod registers weight_scale_inv; see
            # FP8BlockParams.block_scale_attr).
            torch.manual_seed(0)
            w = torch.randn(N, K, dtype=torch.float32, device="cuda") * 2.0
            wb = w.view(N // bn, bn, K // bk, bk)
            w_scale = (wb.abs().amax(dim=(1, 3)) / 448.0).clamp(min=1e-12)
            w_q = (wb / w_scale[:, None, :, None]).reshape(N, K).to(fp8)

            layer = torch.nn.Module()
            layer.weight = torch.nn.Parameter(w_q, requires_grad=False)
            layer.weight_scale_inv = torch.nn.Parameter(w_scale, requires_grad=False)
            layer.weight_scale = None
            layer.input_scale = None
            layer.input_scale_ub = None
            layer.weight_block_size = [bn, bk]
            layer.orig_dtype = torch.float16

            def dequant(w_q, w_s):
                nbn = (w_q.shape[0] + bn - 1) // bn
                kbn = (w_q.shape[1] + bk - 1) // bk
                pad = w_q.new_zeros(nbn * bn, kbn * bk)
                pad[: w_q.shape[0], : w_q.shape[1]] = w_q
                return (
                    pad.to(torch.float32).view(nbn, bn, kbn, bk)
                    * w_s.to(torch.float32)[:, None, :, None]
                ).view(nbn * bn, kbn * bk)[: w_q.shape[0], : w_q.shape[1]]

            with torch.no_grad():
                dequant_before = dequant(layer.weight.data, layer.weight_scale_inv.data)
            print(
                "  before process: "
                f"weight={tuple(layer.weight.shape)} "
                f"scale={tuple(layer.weight_scale_inv.shape)}"
            )
            kernel.process_weights_after_loading(layer)
            print(
                "  after  process: "
                f"weight={tuple(layer.weight.shape)} "
                f"scale={tuple(layer.weight_scale_inv.shape)}"
            )
            with torch.no_grad():
                dequant_after = dequant(layer.weight.data, layer.weight_scale_inv.data)
            print(
                "  process_weights_after_loading dequant max|delta| = "
                f"{(dequant_after - dequant_before).abs().max().item():.3e} "
                "(0 expected unless is_fp8_fnuz())"
            )
            print(f"  is_fp8_fnuz() = {current_platform.is_fp8_fnuz()}")

            x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 2.0
            out = kernel.apply_weights(layer, x)
            print(f"  forward OK: shape={tuple(out.shape)} dtype={out.dtype}")
            print(f"  output finite = {bool(torch.isfinite(out).all())}")

            a_q, a_scale = per_token_group_quant_fp8(x, bk, dtype=fp8)
            ref = native_w8a8_block_matmul(
                a_q,
                layer.weight,
                a_scale,
                layer.weight_scale_inv,
                [bn, bk],
                torch.float16,
            )
            print(f"  apply_weights vs fp32-dequant: {error_metrics(out, ref)}")
            # Independent check that ignores the processed weight entirely:
            # dequantize the *original* weight/scale and compare against the
            # forward using the same quantized activations (a_scale included).
            a_dq = a_q.to(torch.float32) * a_scale.to(torch.float32).repeat_interleave(
                bk, dim=1
            )[:, :K]
            ref_indep = (a_dq @ dequant_before.to(torch.float32).t()).to(torch.float16)
            print(
                "  apply_weights vs ORIGINAL-weight fp32 matmul: "
                f"{error_metrics(out, ref_indep)}"
            )
            if not bool(torch.isfinite(out).all()):
                print("  !! non-finite output")
                return False
            if (out.to(torch.float64) - ref.to(torch.float64)).abs().mean().item() > (
                1e-2 * ref.to(torch.float64).abs().mean().item()
            ):
                print("  !! apply_weights disagrees with the fp32 reference")
                return False
            return True
    except Exception as exc:
        print(f"  RAISED {type(exc).__name__}: {exc}")
        for line in traceback.format_exc().strip().splitlines()[-12:]:
            print(f"    | {line}")
        return False


def section_timing() -> None:
    import torch

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm as fp8_mm,
    )

    print("\n=== 6. order-of-magnitude timing: emulated fp8 vs fp16 vs int8 ===")
    print("  (not a benchmark; only enough to see whether fp8 is hardware-fast)")
    M = N = K = 4096
    bn = bk = 128
    fp8 = torch.float8_e4m3fn
    torch.manual_seed(0)

    def timeit(fn, iters=8):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    try:
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        w = torch.randn(N, K, dtype=torch.float16, device="cuda")
        ms_fp16 = timeit(lambda: a @ w.t())

        a32 = torch.randn(M, K, dtype=torch.float32, device="cuda")
        w32 = torch.randn(N, K, dtype=torch.float32, device="cuda")
        ms_fp32 = timeit(lambda: a32 @ w32.t())

        aq, as_ = per_token_group_quant_fp8(a, bk, dtype=fp8)
        _, _, _, _, wq, ws = _make_operands(M, N, K, bn, bk, "cuda", fp8)

        ms_fp8 = timeit(lambda: fp8_mm(aq, wq, as_, ws, [bn, bk], torch.float16))

        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
            triton_scaled_mm,
        )

        ai8 = torch.randint(-100, 100, (M, K), dtype=torch.int8, device="cuda")
        wi8 = torch.randint(-100, 100, (K, N), dtype=torch.int8, device="cuda")
        sa = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        sb = torch.ones(N, dtype=torch.float32, device="cuda") * 0.01
        ms_int8 = timeit(
            lambda: triton_scaled_mm(ai8, wi8, sa, sb, out_dtype=torch.float16)
        )

        print(f"  4096^3  fp16 torch matmul  : {ms_fp16:8.1f} ms")
        print(f"  4096^3  fp32 torch matmul  : {ms_fp32:8.1f} ms")
        print(f"  4096^3  triton int8 (prod) : {ms_int8:8.1f} ms")
        print(f"  4096^3  triton block fp8   : {ms_fp8:8.1f} ms")
    except Exception as exc:
        print(f"  RAISED {type(exc).__name__}: {exc}")
        for line in traceback.format_exc().strip().splitlines()[-8:]:
            print(f"    | {line}")


def main() -> int:
    import torch  # noqa: F401  (must precede `import vllm`, hard rule 8)

    checks = {}
    checks["selection"] = section_selection()
    checks["numeric"] = section_numeric()
    checks["input_quant"] = section_input_quant()
    checks["full_kernel"] = section_full_kernel()
    section_timing()
    print("\n=== S8b FP8 block-scaled verdict ===")
    for name, ok in checks.items():
        print(f"  {name:12s} {'PASS' if ok else 'FAIL'}")
    rc = 0 if all(checks.values()) else 1
    print(f"FP8_BLOCK_PROBE_DONE rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
