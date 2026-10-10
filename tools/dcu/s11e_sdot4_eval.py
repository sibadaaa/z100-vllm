# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S11e evaluation harness for the gfx906 W8A8 INT8 ``sdot4`` kernel.

Modes
-----
``int``    exactness of the int32 accumulator: run the kernel with
           ``x_s = w_s = 1`` and ``out_dtype=float32`` and compare bit-for-bit
           against ``torch.matmul(xq.int(), wq.int().t()).float()``.
``check``  rel_err against the fp32 dequantized reference
           ``(xq.float() - zp_x) * s_x @ ((wq.float() - zp_w) * s_w).T + bias``
           for every (M, N, K) in the S11e shape matrix, for the sdot4 kernel
           and (for cross-checking) the Triton fallback.
``bench``  ABBA timing of the sdot4 kernel against
           ``TritonInt8ScaledMMLinearKernel``'s ``triton_scaled_mm``, both
           kernel-only and through the full ``apply_weights`` path.

Must be run inside a DTK container with the canonical tree on PYTHONPATH:

    HIP_VISIBLE_DEVICES=2 python tools/dcu/s11e_sdot4_eval.py --mode check
"""

import argparse
import json
import os
import sys
import time

import torch  # noqa: F401  (must precede `import vllm`, see todolist constraint 8)


def _load_vllm():
    import vllm  # noqa: F401
    from vllm import _custom_ops as ops

    return ops


# Real (N, K) int8 weight shapes of Qwen3.8-27B-Channel-INT8-w8a8, read from
# the safetensors index.  N = output features, K = input features.
SHAPES = [
    (12288, 5120, "self_attn.q_proj"),
    (1024, 5120, "self_attn.k_proj/v_proj"),
    (5120, 6144, "self_attn.o_proj"),
    (10240, 5120, "linear_attn.in_proj_qkv"),
    (6144, 5120, "linear_attn.in_proj_z"),
    (17408, 5120, "mlp.gate_proj/up_proj"),
    (5120, 17408, "mlp.down_proj"),
]

MS_GEMV = [1, 2, 4, 8]
MS_GEMM = [32, 64, 256, 512]
MS_ALL = MS_GEMV + [16] + MS_GEMM


def make_inputs(n, k, m, dtype=torch.float16, device="cuda"):
    torch.manual_seed(0)
    # Weight: int8 with a per-output-channel scale (channel-wise symmetric).
    wq = torch.randint(-127, 128, (n, k), dtype=torch.int8, device=device)
    w_s = (torch.rand(n, dtype=torch.float32, device=device) * 0.02 + 0.005)
    # Activation: fp16, per-token dynamic symmetric int8 quant.
    x = (torch.randn(m, k, dtype=torch.float32, device=device) * 1.5).to(dtype)
    return wq, w_s, x


def quantize_x(ops, x):
    x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
    return x_q, x_s.reshape(-1).to(torch.float32).contiguous()


def reference(x_q, x_s, w_q, w_s, bias=None):
    """fp32 dequantized reference (S8 口径, zero points are 0 for symmetric)."""
    a = x_q.float() * x_s.reshape(-1, 1)
    b = w_q.float() * w_s.reshape(-1, 1)
    out = a @ b.t()
    if bias is not None:
        out = out + bias.float()
    return out


def rel_err(out, ref):
    num = (out.float() - ref).norm()
    den = ref.norm().clamp_min(1e-12)
    return (num / den).item()


def run_int(mode_args):
    ops = _load_vllm()
    print("== int32 exactness: sdot4(x_s=1,w_s=1,out=fp32) vs int64 matmul ==")
    rows = []
    for n, k, name in SHAPES:
        wq = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda")
        for m in (1, 4, 8, 32, 64):
            xq = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda")
            ones_m = torch.ones(m, dtype=torch.float32, device="cuda")
            ones_n = torch.ones(n, dtype=torch.float32, device="cuda")
            if m <= 8:
                out = torch.ops._rocm_C.w8a8_sdot4_gemv(
                    xq, wq, ones_m, ones_n, None, torch.float32, 64
                )
            else:
                out = torch.ops._rocm_C.w8a8_sdot4_gemm(
                    xq, wq, ones_m, ones_n, None, torch.float32, 0
                )
            exact = torch.matmul(xq.cpu().int(), wq.cpu().int().t()).float().cuda()
            bit_eq = bool(torch.equal(out, exact))
            maxdiff = (out - exact).abs().max().item()
            rows.append(
                dict(N=int(n), K=int(k), M=int(m), name=name,
                     bit_equal=bit_eq, max_abs_diff=maxdiff)
            )
            print(f"  N={n:6d} K={k:6d} M={m:4d} bit_equal={bit_eq} "
                  f"max_abs_diff={maxdiff:g}")
    print("INT_SUMMARY", json.dumps(rows))
    return 0


def run_check(mode_args):
    ops = _load_vllm()
    print("== numeric check vs fp32 dequant reference ==")
    rows = []
    worst = 0.0
    worst_triton = 0.0
    for n, k, name in SHAPES:
        wq, w_s, x = make_inputs(n, k, 1)
        for m in MS_ALL:
            x = (torch.randn(m, k, dtype=torch.float32, device="cuda") * 1.5).to(
                torch.float16
            )
            bias = (torch.randn(n, dtype=torch.float32, device="cuda") * 0.1).to(
                torch.float16
            )
            x_q, x_s = quantize_x(ops, x)
            ref = reference(x_q, x_s, wq, w_s, bias)
            r = dict(N=int(n), K=int(k), M=int(m), name=name)

            if m <= 8:
                out = torch.ops._rocm_C.w8a8_sdot4_gemv(
                    x_q, wq, x_s, w_s, bias, torch.float16, 64
                )
            else:
                out = torch.ops._rocm_C.w8a8_sdot4_gemm(
                    x_q, wq, x_s, w_s, bias, torch.float16, 0
                )
            r["sdot4_rel_err"] = rel_err(out, ref)
            r["sdot4_finite"] = bool(torch.isfinite(out).all())

            from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
                triton_scaled_mm,
            )

            tr = triton_scaled_mm(
                x_q,
                wq.t().contiguous(),
                scale_a=x_s,
                scale_b=w_s,
                out_dtype=torch.float16,
                bias=bias,
            )
            r["triton_rel_err"] = rel_err(tr, ref)
            r["sdot4_agrees_with_triton"] = bool(torch.equal(out, tr))
            worst = max(worst, r["sdot4_rel_err"])
            worst_triton = max(worst_triton, r["triton_rel_err"])
            rows.append(r)
            print(
                f"  N={n:6d} K={k:6d} M={m:4d} sdot4={r['sdot4_rel_err']:.3e} "
                f"triton={r['triton_rel_err']:.3e} finite={r['sdot4_finite']} "
                f"identical={r['sdot4_agrees_with_triton']}  # {name}"
            )
    print(f"WORST sdot4 rel_err = {worst:.3e}   (gate <= 1e-2)")
    print(f"WORST triton rel_err = {worst_triton:.3e}")
    print("CHECK_SUMMARY", json.dumps(rows))
    return 0


def fq(a, d):
    return a.to(d) if a is not None else None


def bench_callable(fn, iters, rounds):
    """Return (min_ms_per_call, n_used) measured over `rounds` evented loops."""
    best = float("inf")
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters)
    return best


def _fake_layer(wq, w_s, n, k):
    from types import SimpleNamespace

    layer = SimpleNamespace()
    layer.weight = torch.nn.Parameter(wq, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        w_s.reshape(n, 1).contiguous(), requires_grad=False
    )
    layer.input_scale = None
    layer.input_zero_point = None
    layer.azp_adj = None
    layer.logical_widths = [n]
    return layer


def run_bench(mode_args):
    ops = _load_vllm()
    from vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4 import (
        Gfx906Sdot4Int8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        Int8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonInt8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )

    cfg = Int8ScaledMMLinearLayerConfig(
        is_static_input_scheme=False, is_channelwise=True, input_symmetric=True
    )
    param_names = [
        "weight",
        "weight_scale",
        "input_scale",
        "input_zero_point",
        "azp_adj",
    ]
    supported, why = Gfx906Sdot4Int8ScaledMMLinearKernel.is_supported()
    print(f"Gfx906Sdot4Int8ScaledMMLinearKernel.is_supported() = {supported} {why}")
    tsupported, twhy = TritonInt8ScaledMMLinearKernel.is_supported()
    print(f"TritonInt8ScaledMMLinearKernel.is_supported() = {tsupported} {twhy}")

    iters = int(os.environ.get("S11E_ITERS", "50"))
    rounds = int(os.environ.get("S11E_ROUNDS", "5"))
    rows = []
    for n, k, name in SHAPES:
        wq, w_s, _ = make_inputs(n, k, 8)
        w_t = wq.t().contiguous()  # what the Triton path holds after load
        for m in MS_ALL:
            x = (torch.randn(m, k, dtype=torch.float32, device="cuda") * 1.5).to(
                torch.float16
            )
            bias = (torch.randn(n, dtype=torch.float32, device="cuda") * 0.1).to(
                torch.float16
            )
            x_q, x_s = quantize_x(ops, x)

            # ---- kernel-only ------------------------------------------------
            if m <= 8:
                kern_a = lambda: torch.ops._rocm_C.w8a8_sdot4_gemv(  # noqa: E731
                    x_q, wq, x_s, w_s, bias, torch.float16, 64
                )
            else:
                kern_a = lambda: torch.ops._rocm_C.w8a8_sdot4_gemm(  # noqa: E731
                    x_q, wq, x_s, w_s, bias, torch.float16, 0
                )
            kern_b = lambda: triton_scaled_mm(  # noqa: E731
                x_q,
                w_t,
                scale_a=x_s,
                scale_b=w_s,
                out_dtype=torch.float16,
                bias=bias,
            )
            for _ in range(3):
                kern_a()
                kern_b()
            torch.cuda.synchronize()

            # ABBA: A B B A, take the min per method over all rounds.
            timings = {"sdot4": [], "triton": []}
            for _ in range(rounds):
                timings["sdot4"].append(bench_callable(kern_a, iters, 1))
                timings["triton"].append(bench_callable(kern_b, iters, 1))
                timings["triton"].append(bench_callable(kern_b, iters, 1))
                timings["sdot4"].append(bench_callable(kern_a, iters, 1))
            a_ms = min(timings["sdot4"])
            b_ms = min(timings["triton"])

            # ---- full apply_weights path (quant + gemm) ----------------------
            la = _fake_layer(wq.clone(), w_s, n, k)
            lb = _fake_layer(wq.clone(), w_s, n, k)
            ka = Gfx906Sdot4Int8ScaledMMLinearKernel(cfg, param_names)
            kb = TritonInt8ScaledMMLinearKernel(cfg, param_names)
            ka.process_weights_after_loading(la)
            kb.process_weights_after_loading(lb)
            full_a = bench_callable(lambda: ka.apply_weights(la, x, bias), iters, 3)
            full_b = bench_callable(lambda: kb.apply_weights(lb, x, bias), iters, 3)

            flops = 2.0 * m * n * k
            rows.append(
                dict(
                    N=int(n), K=int(k), M=int(m), name=name,
                    sdot4_ms=a_ms, triton_ms=b_ms,
                    speedup=b_ms / a_ms,
                    sdot4_tflops=flops / (a_ms * 1e-3) / 1e12,
                    triton_tflops=flops / (b_ms * 1e-3) / 1e12,
                    sdot4_full_ms=full_a, triton_full_ms=full_b,
                    full_speedup=full_b / full_a,
                )
            )
            print(
                f"  N={n:6d} K={k:6d} M={m:4d}  kern: sdot4={a_ms*1000:9.2f}us "
                f"triton={b_ms*1000:9.2f}us  x{b_ms/a_ms:6.2f} "
                f"({flops/(a_ms*1e-3)/1e12:6.2f} vs {flops/(b_ms*1e-3)/1e12:6.2f} TF)  "
                f"full: {full_a*1000:9.2f} vs {full_b*1000:9.2f}us x{full_b/full_a:6.2f}"
            )
    print("BENCH_SUMMARY", json.dumps(rows))
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["int", "check", "bench"], required=True)
    p.add_argument("--out", default=None, help="write the JSON summary here")
    a = p.parse_args()
    torch.cuda.init()
    print("device:", torch.cuda.get_device_name(0), "count:", torch.cuda.device_count())
    t0 = time.time()
    rc = {"int": run_int, "check": run_check, "bench": run_bench}[a.mode](a)
    print(f"elapsed {time.time()-t0:.1f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
