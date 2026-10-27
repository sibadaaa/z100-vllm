#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-op parity gate for the gfx906 GDN chunked-prefill A/B arm.

Compares, on byte-identical production-shaped inputs, the two prefill
implementations that ``VLLM_GFX906_GDN_CHUNKED_PREFILL`` selects between:

  fused    fused_sigmoid_gating_delta_rule_update -> gfx906 C++ prefill kernel
           (csrc/mamba/gfx906_decode_kernels.cu, the current production path)
  chunked  fla chunk_gated_delta_rule -> Triton chunk chain
           (fla/ops/{cumsum,chunk_scaled_dot_kkt,solve_tril,wy_fast,
            chunk_delta_h,chunk_o}.py)

Both are fed by the same ``fused_post_conv_prep`` outputs, exactly as
``GatedDeltaNetAttention._forward_core`` feeds them, so the only difference is
the recurrent-update implementation.

The comparison is done in the **cache** layout ``[N, Hv, V, K]`` for the state
(mamba_utils.py temporal_state_shape) because a transposed write-back is silent
when head_k_dim == head_v_dim -- that is the failure this gate exists to catch.

Production shapes for Qwen3.8-27B-FP8 on TP=4: 4 k-heads / 12 v-heads per rank,
head_k_dim = head_v_dim = 128, fp16, fp32 state.

  HIP_VISIBLE_DEVICES=1 python tools/dcu/perf1_gdn_chunked_parity.py
"""

import argparse

import torch

from vllm.model_executor.layers.fla.ops.chunk import (
    chunk_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv import (
    fused_post_conv_prep,
)
from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)

HK = 4
HV = 12
K = 128
V = 128
DEV = "cuda"
F16 = dict(device=DEV, dtype=torch.float16)


def build_case(seed: int, seq_lens: list[int]) -> dict:
    gen = torch.Generator(device=DEV).manual_seed(seed)
    total = sum(seq_lens)
    conv = (
        torch.randn(total, 2 * HK * K + HV * V, generator=gen, **F16) * 0.5
    ).contiguous()
    a = torch.randn(total, HV, generator=gen, **F16) * 0.2
    b = torch.randn(total, HV, generator=gen, **F16) * 0.2
    a_log = torch.randn(HV, generator=gen, **F16) * 0.2
    dt_bias = torch.randn(HV, generator=gen, **F16) * 0.1
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        num_k_heads=HK,
        head_k_dim=K,
        head_v_dim=V,
        apply_l2norm=True,
        output_g_exp=False,
    )
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, device=DEV, dtype=torch.int32)
    n = len(seq_lens)
    # Cache layout: [N, Hv, V, K].
    state = (
        torch.randn(n, HV, V, K, generator=gen, device=DEV, dtype=torch.float32) * 0.05
    )
    return dict(
        a=a,
        b=b,
        a_log=a_log,
        dt_bias=dt_bias,
        q=q.unsqueeze(0).contiguous(),
        k=k.unsqueeze(0).contiguous(),
        v=v.unsqueeze(0).contiguous(),
        g=g.unsqueeze(0).contiguous(),
        beta=beta.unsqueeze(0).contiguous(),
        cu=cu,
        state=state,
    )


def run_fused(case: dict) -> tuple[torch.Tensor, torch.Tensor]:
    state = case["state"].clone()
    out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=case["a_log"],
        a=case["a"],
        b=case["b"],
        dt_bias=case["dt_bias"],
        q=case["q"],
        k=case["k"],
        v=case["v"],
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=case["cu"],
        ssm_state_indices=None,
        use_qk_l2norm_in_kernel=False,
    )
    return out, state


def run_chunked(case: dict, swap_state: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    state = case["state"]
    if swap_state:
        # Negative control: feed the [Hv, V, K] cache straight through instead
        # of converting to the [Hv, K, V] operator layout. With
        # head_k_dim == head_v_dim this raises nothing -- the gate must catch it.
        init = state.contiguous()
    else:
        init = state.transpose(-1, -2).contiguous()
    out, final = chunk_gated_delta_rule(
        q=case["q"],
        k=case["k"],
        v=case["v"],
        g=case["g"],
        beta=case["beta"],
        initial_state=init,
        output_final_state=True,
        cu_seqlens=case["cu"],
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )
    if swap_state:
        return out, final
    return out, final.transpose(-1, -2)


def timeit(fn, case, iters: int = 5) -> float:
    for _ in range(2):
        fn(case)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(case)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol-out", type=float, default=1e-3)
    ap.add_argument("--tol-state", type=float, default=6e-4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument(
        "--negative-control",
        action="store_true",
        help="Skip the real chunked run and instead measure the untransposed "
        "state feed, to prove the gate is sensitive to the layout bug.",
    )
    args = ap.parse_args()

    cases = [
        (1, [512]),
        (2, [1024]),
        (3, [2048]),
        (4, [2048, 1024, 512]),
        (5, [700, 700, 700, 700]),
        (6, [1500, 548]),
    ]

    if args.negative_control:
        print("NEGATIVE CONTROL: chunked fed with the unconverted [N,Hv,V,K] state\n")

    failures = 0
    print(f"{'case':>4} {'seq_lens':>22} {'max|dout|':>11} {'max|dstate|':>12} "
          f"{'nan_o':>6} {'nan_s':>6} {'o !=':>8} {'s !=':>8} {'fused ms':>9} "
          f"{'chunk ms':>9} {'speedup':>8}")
    for seed, seq_lens in cases:
        case = build_case(seed, seq_lens)
        out_f, state_f = run_fused(case)
        out_c, state_c = run_chunked(case, swap_state=args.negative_control)
        torch.cuda.synchronize()

        d_out = (out_f.float() - out_c.float()).abs().max().item()
        d_state = (state_f - state_c).abs().max().item()
        nan_o = int(torch.isnan(out_c).sum().item())
        nan_s = int(torch.isnan(state_c).sum().item())
        neq_o = int((out_f != out_c).sum().item())
        neq_s = int((state_f != state_c).sum().item())

        if args.skip_perf or args.negative_control:
            t_f = t_c = float("nan")
            speed = float("nan")
        else:
            t_f = timeit(run_fused, case, args.iters)
            t_c = timeit(run_chunked, case, args.iters)
            speed = t_f / t_c

        bad = (
            d_out > args.tol_out
            or d_state > args.tol_state
            or nan_o
            or nan_s
        )
        failures += int(bad)
        print(
            f"{seed:>4} {str(seq_lens):>22} {d_out:>11.3e} {d_state:>12.3e} "
            f"{nan_o:>6} {nan_s:>6} {neq_o:>8} {neq_s:>8} {t_f:>9.3f} "
            f"{t_c:>9.3f} {speed:>7.2f}x {'FAIL' if bad else 'ok'}"
        )
        del case, out_f, out_c, state_f, state_c

    print(
        f"\nSUMMARY cases={len(cases)} failures={failures} "
        f"tol_out={args.tol_out:g} tol_state={args.tol_state:g}"
    )


if __name__ == "__main__":
    main()
