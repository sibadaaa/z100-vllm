# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os
import sys
from functools import lru_cache

import torch

from vllm import _custom_ops as ops


_GFX906_MTP_GDN_DEBUG_REPORTED = False
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def _gdn_mtp_triton_enabled() -> bool:
    """Perf1 A/B switch for the layout-aware gfx906 MTP kernel (opt-in)."""
    return os.getenv("VLLM_QWEN35_MTP_GDN_TRITON", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _gdn_mtp_triton(**kwargs):
    # Imported lazily so the default (env unset) path never pays for the
    # module import and cannot be affected by it.
    from vllm.model_executor.layers.fla.ops.gfx906_gdn_mtp_triton import (
        gdn_mtp_gfx906_triton,
    )

    return gdn_mtp_gfx906_triton(**kwargs)


def _mtp_debug_enabled() -> bool:
    return os.getenv("VLLM_QWEN35_MTP_GDN_DEBUG", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _report_mtp_custom_hit(q, initial_state, num_accepted_tokens, engine) -> None:
    """One-shot attribution line for the VLLM_QWEN35_MTP_GDN_DEBUG channel."""
    global _GFX906_MTP_GDN_DEBUG_REPORTED
    if _GFX906_MTP_GDN_DEBUG_REPORTED or not _mtp_debug_enabled():
        return
    _GFX906_MTP_GDN_DEBUG_REPORTED = True
    print(
        f"QWEN35_MTP_GDN_CUSTOM_HIT engine={engine} "
        f"q={tuple(q.shape)}/{q.dtype} "
        f"state={tuple(initial_state.shape)}/{initial_state.dtype} "
        f"accepted={num_accepted_tokens.dtype}",
        file=sys.stderr,
    )


def _report_mtp_ineligible(
    num_accepted_tokens,
    cu_seqlens,
    q,
    initial_state,
    inplace_final_state,
    ssm_state_indices,
) -> None:
    """One-shot attribution line when the MTP kernel is not eligible.

    Kept out of the dispatcher with the other debug helpers so that
    `fused_sigmoid_gating_delta_rule_update` contains no store to a module
    global (a missing `global` there raises UnboundLocalError on every call,
    not just when the debug channel is enabled).
    """
    global _GFX906_MTP_GDN_DEBUG_REPORTED
    if _GFX906_MTP_GDN_DEBUG_REPORTED or not _mtp_debug_enabled():
        return
    _GFX906_MTP_GDN_DEBUG_REPORTED = True
    state_desc = (
        None
        if initial_state is None
        else (tuple(initial_state.shape), initial_state.dtype)
    )
    indices_desc = (
        None
        if ssm_state_indices is None
        else tuple(ssm_state_indices.shape)
    )
    print(
        "QWEN35_MTP_GDN_INELIGIBLE "
        f"cu={cu_seqlens is not None} q={tuple(q.shape)}/{q.dtype} "
        f"state={state_desc} "
        f"inplace={inplace_final_state} "
        f"indices={indices_desc} "
        f"accepted={tuple(num_accepted_tokens.shape)}/"
        f"{num_accepted_tokens.dtype}",
        file=sys.stderr,
    )


@lru_cache(maxsize=1)
def _is_gfx906_rocm() -> bool:
    capability = current_platform.get_device_capability()
    return (
        current_platform.is_rocm()
        and capability is not None
        and capability.major == 9
        and capability.minor == 0
    )


def _lookup_state_index(
    ssm_state_indices: torch.Tensor | None,
    seq_idx: int,
    token_idx: int,
) -> int:
    if ssm_state_indices is None:
        return seq_idx
    if ssm_state_indices.ndim == 1:
        return int(ssm_state_indices[seq_idx].item())
    return int(ssm_state_indices[seq_idx, token_idx].item())


def _fused_sigmoid_gating_delta_rule_update_gfx906_eager(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float,
    threshold: float,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    ssm_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0)
    head_ratio = HV // H
    o = torch.empty((B, T, HV, V), device=q.device, dtype=q.dtype)

    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    if cu_seqlens is None:
        seq_ranges = [(batch_idx, batch_idx, 0, T) for batch_idx in range(B)]
    else:
        seq_ranges = []
        for seq_idx in range(len(cu_seqlens) - 1):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            if end > start:
                seq_ranges.append((0, seq_idx, start, end))

    for batch_idx, seq_idx, seq_start, seq_end in seq_ranges:
        if initial_state is None:
            state = torch.zeros((HV, V, K), device=q.device, dtype=torch.float32)
        else:
            init_token_idx = 0
            if num_accepted_tokens is not None:
                init_token_idx = max(int(num_accepted_tokens[seq_idx].item()) - 1, 0)
            state_index = _lookup_state_index(
                ssm_state_indices, seq_idx, init_token_idx
            )
            if state_index < 0:
                continue
            state = initial_state[state_index].float().clone()

        for token_idx in range(seq_start, seq_end):
            local_token_idx = token_idx - seq_start
            for head_idx in range(HV):
                q_head_idx = head_idx // head_ratio
                q_t = q[batch_idx, token_idx, q_head_idx].float()
                k_t = k[batch_idx, token_idx, q_head_idx].float()
                v_t = v[batch_idx, token_idx, head_idx].float()

                x = (
                    a[batch_idx, token_idx, head_idx].float()
                    + dt_bias[head_idx].float()
                )
                softplus_x = torch.where(
                    beta * x <= threshold,
                    (1.0 / beta) * torch.log1p(torch.exp(beta * x)),
                    x,
                )
                g_t = -torch.exp(A_log[head_idx].float()) * softplus_x
                beta_t = torch.sigmoid(b[batch_idx, token_idx, head_idx].float())

                if use_qk_l2norm_in_kernel:
                    q_t = q_t * torch.rsqrt(torch.sum(q_t * q_t) + 1e-6)
                    k_t = k_t * torch.rsqrt(torch.sum(k_t * k_t) + 1e-6)
                q_t = q_t * scale

                state_head = state[head_idx]
                state_head = state_head * torch.exp(g_t)
                v_residual = v_t - torch.sum(state_head * k_t[None, :], dim=1)
                v_residual = v_residual * beta_t
                state_head = state_head + v_residual[:, None] * k_t[None, :]

                o[batch_idx, token_idx, head_idx] = torch.sum(
                    state_head * q_t[None, :], dim=1
                ).to(o.dtype)
                state[head_idx] = state_head

            if inplace_final_state:
                state_index = _lookup_state_index(
                    ssm_state_indices, seq_idx, local_token_idx
                )
                if state_index >= 0:
                    final_state[state_index] = state.to(final_state.dtype)
            else:
                final_state[token_idx] = state.to(final_state.dtype)

    return o, final_state


def _fused_sigmoid_gating_delta_rule_decode_gfx906(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float,
    threshold: float,
    scale: float,
    initial_state: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, T, H, K = q.shape
    HV = v.shape[2]
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0)

    can_use_custom_op = (
        A_log.dtype == a.dtype == b.dtype == dt_bias.dtype == q.dtype == k.dtype
        == v.dtype
        and initial_state.dtype in (q.dtype, torch.float32)
    )
    if can_use_custom_op:
        try:
            o = ops.fused_sigmoid_gating_delta_rule_gfx906_decode(
                A_log=A_log,
                a=a,
                b=b,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                state=initial_state,
                beta=beta,
                threshold=threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            return o, initial_state
        except (AttributeError, RuntimeError):
            pass

    head_ratio = HV // H
    q_head_idx = torch.arange(HV, device=q.device) // head_ratio
    q_t = q[0].index_select(1, q_head_idx).float()
    k_t = k[0].index_select(1, q_head_idx).float()
    v_t = v[0].float()

    if use_qk_l2norm_in_kernel:
        q_t = q_t * torch.rsqrt(torch.sum(q_t * q_t, dim=-1, keepdim=True) + 1e-6)
        k_t = k_t * torch.rsqrt(torch.sum(k_t * k_t, dim=-1, keepdim=True) + 1e-6)
    q_t = q_t * scale

    x = a[0].float() + dt_bias.float().view(1, HV)
    softplus_x = torch.where(
        beta * x <= threshold,
        (1.0 / beta) * torch.log1p(torch.exp(beta * x)),
        x,
    )
    g_t = -torch.exp(A_log.float()).view(1, HV) * softplus_x
    beta_t = torch.sigmoid(b[0].float())

    state = initial_state.float()[:T]
    state = state * torch.exp(g_t).view(T, HV, 1, 1)
    v_residual = v_t - torch.sum(state * k_t[:, :, None, :], dim=-1)
    v_residual = v_residual * beta_t[:, :, None]
    state = state + v_residual[:, :, :, None] * k_t[:, :, None, :]
    o = torch.sum(state * q_t[:, :, None, :], dim=-1).unsqueeze(0).to(q.dtype)

    initial_state[:T].copy_(state.to(initial_state.dtype))
    return o, initial_state


def _fused_sigmoid_gating_delta_rule_prefill_gfx906(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float,
    threshold: float,
    scale: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0)

    can_use_custom_op = (
        A_log.dtype == a.dtype == b.dtype == dt_bias.dtype == q.dtype == k.dtype
        == v.dtype
        and initial_state.dtype in (q.dtype, torch.float32)
        and cu_seqlens.dtype in (torch.int32, torch.int64)
    )
    if can_use_custom_op:
        try:
            o = ops.fused_sigmoid_gating_delta_rule_gfx906_prefill(
                A_log=A_log,
                a=a,
                b=b,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                state=initial_state,
                cu_seqlens=cu_seqlens,
                beta=beta,
                threshold=threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            return o, initial_state
        except (AttributeError, RuntimeError):
            pass

    return _fused_sigmoid_gating_delta_rule_update_gfx906_eager(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        beta=beta,
        threshold=threshold,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def _fused_sigmoid_gating_delta_rule_indexed_decode_gfx906(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float,
    threshold: float,
    scale: float,
    initial_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0)

    can_use_custom_op = (
        A_log.dtype == a.dtype == b.dtype == dt_bias.dtype == q.dtype == k.dtype
        == v.dtype
        and initial_state.dtype in (q.dtype, torch.float32)
        and ssm_state_indices.dtype == torch.int32
    )
    if can_use_custom_op:
        try:
            o = ops.fused_sigmoid_gating_delta_rule_gfx906_indexed_decode(
                A_log=A_log,
                a=a,
                b=b,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                state=initial_state,
                state_indices=ssm_state_indices,
                beta=beta,
                threshold=threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            return o, initial_state
        except (AttributeError, RuntimeError):
            pass

    gathered_state = initial_state[ssm_state_indices].contiguous()
    o, gathered_state = _fused_sigmoid_gating_delta_rule_decode_gfx906(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        beta=beta,
        threshold=threshold,
        scale=scale,
        initial_state=gathered_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    initial_state[ssm_state_indices] = gathered_state.to(initial_state.dtype)
    return o, initial_state


def fused_sigmoid_gating_delta_rule_update_kv_cache_gfx906(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float,
    threshold: float,
    scale: float,
    initial_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0)

    can_use_custom_op = (
        A_log.dtype == a.dtype == b.dtype == dt_bias.dtype == q.dtype == k.dtype
        == v.dtype
        and initial_state.dtype in (q.dtype, torch.float32)
        and ssm_state_indices.dtype == torch.int32
    )
    if can_use_custom_op:
        try:
            o = ops.fused_sigmoid_gating_delta_rule_gfx906_indexed_decode_kv_state(
                A_log=A_log,
                a=a,
                b=b,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                state=initial_state,
                state_indices=ssm_state_indices,
                beta=beta,
                threshold=threshold,
                scale=scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            return o, initial_state
        except (AttributeError, RuntimeError):
            pass

    gathered_state = initial_state[ssm_state_indices].transpose(-1, -2).contiguous()
    o, gathered_state = _fused_sigmoid_gating_delta_rule_decode_gfx906(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        beta=beta,
        threshold=threshold,
        scale=scale,
        initial_state=gathered_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    initial_state[ssm_state_indices] = gathered_state.transpose(-1, -2).to(
        initial_state.dtype
    )
    return o, initial_state


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    q,
    k,
    v,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v

    p_A_log = A_log + i_hv
    if not IS_KDA:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv
    else:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k

    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                tl.int64
            )
            if state_idx < 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
        softplus_x = tl.where(
            beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
        )
        b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x

        b_beta = tl.sigmoid(b_b.to(tl.float32))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        if not IS_KDA:
            b_h *= tl.exp(b_g)
        else:
            b_h *= tl.exp(b_g[None, :])
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if INPLACE_FINAL_STATE:
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int64)
            if final_state_idx >= 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):
    global _GFX906_MTP_GDN_DEBUG_REPORTED
    if _is_gfx906_rocm() and not is_kda:
        if scale is None:
            scale = k.shape[-1] ** -0.5
        else:
            assert scale > 0, "scale must be positive"
        mtp_gfx906_path = (
            cu_seqlens is not None
            and q.shape[0] == 1
            and initial_state is not None
            and inplace_final_state
            and ssm_state_indices is not None
            and num_accepted_tokens is not None
            # S18 bisect switch: VLLM_QWEN35_MTP_GDN_CUSTOM=0 skips the gfx906
            # MTP kernel (opt-in A/B only; the default keeps current
            # behaviour) so MTP-vs-non-spec token parity can be attributed.
            and os.getenv("VLLM_QWEN35_MTP_GDN_CUSTOM", "1") != "0"
        )
        if mtp_gfx906_path:
            if _gdn_mtp_triton_enabled():
                # Perf1 A/B: same math as the C++ MTP kernel, but lanes map
                # onto the key axis of one value row so the persistent state
                # ([slot][head][v][k], k contiguous) is read/written
                # coalesced, and the grid is (VD/BV, kv_heads, requests)
                # instead of a single 6-block serial kernel.  Opt-in only;
                # VLLM_QWEN35_MTP_GDN_CUSTOM=0 still wins (eager bisect path).
                try:
                    o = _gdn_mtp_triton(
                        A_log=A_log,
                        a=a,
                        b=b,
                        dt_bias=dt_bias,
                        q=q,
                        k=k,
                        v=v,
                        state=initial_state,
                        state_indices=ssm_state_indices,
                        cu_seqlens=cu_seqlens,
                        num_accepted_tokens=num_accepted_tokens,
                        beta=beta,
                        threshold=threshold,
                        scale=scale,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    )
                    _report_mtp_custom_hit(
                        q, initial_state, num_accepted_tokens, "triton"
                    )
                    return o.unsqueeze(0), initial_state
                except (
                    AttributeError,
                    RuntimeError,
                    AssertionError,
                    ImportError,
                ) as exc:
                    if _mtp_debug_enabled():
                        print(
                            f"QWEN35_MTP_GDN_TRITON_FALLBACK "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
            try:
                o = ops.fused_sigmoid_gating_delta_rule_gfx906_mtp_update(
                    A_log=A_log,
                    a=a,
                    b=b,
                    dt_bias=dt_bias,
                    q=q,
                    k=k,
                    v=v,
                    state=initial_state,
                    state_indices=ssm_state_indices,
                    cu_seqlens=cu_seqlens,
                    num_accepted_tokens=num_accepted_tokens,
                    beta=beta,
                    threshold=threshold,
                    scale=scale,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )
                _report_mtp_custom_hit(q, initial_state, num_accepted_tokens, "cpp")
                return o.unsqueeze(0), initial_state
            except (AttributeError, RuntimeError) as exc:
                if _mtp_debug_enabled():
                    print(
                        "QWEN35_MTP_GDN_FALLBACK "
                        f"{type(exc).__name__}: {exc}; "
                        f"q={tuple(q.shape)}/{q.dtype} "
                        f"a={tuple(a.shape)}/{a.dtype} "
                        f"A_log={tuple(A_log.shape)}/{A_log.dtype} "
                        f"state={tuple(initial_state.shape)}/{initial_state.dtype} "
                        f"indices={tuple(ssm_state_indices.shape)}/"
                        f"{ssm_state_indices.dtype} "
                        f"cu={tuple(cu_seqlens.shape)}/{cu_seqlens.dtype} "
                        f"accepted={tuple(num_accepted_tokens.shape)}/"
                        f"{num_accepted_tokens.dtype}",
                        file=sys.stderr,
                    )
        elif num_accepted_tokens is not None:
            _report_mtp_ineligible(
                num_accepted_tokens,
                cu_seqlens,
                q,
                initial_state,
                inplace_final_state,
                ssm_state_indices,
            )
        decode_gfx906_path = (
            cu_seqlens is not None
            and q.shape[0] == 1
            and initial_state is not None
            and inplace_final_state
            and num_accepted_tokens is None
        )
        if decode_gfx906_path:
            if ssm_state_indices is not None:
                return _fused_sigmoid_gating_delta_rule_indexed_decode_gfx906(
                    A_log=A_log,
                    a=a,
                    b=b,
                    dt_bias=dt_bias,
                    q=q,
                    k=k,
                    v=v,
                    beta=beta,
                    threshold=threshold,
                    scale=scale,
                    initial_state=initial_state,
                    ssm_state_indices=ssm_state_indices,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )
            if q.shape[1] != initial_state.shape[0]:
                return _fused_sigmoid_gating_delta_rule_prefill_gfx906(
                    A_log=A_log,
                    a=a,
                    b=b,
                    dt_bias=dt_bias,
                    q=q,
                    k=k,
                    v=v,
                    beta=beta,
                    threshold=threshold,
                    scale=scale,
                    initial_state=initial_state,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )
            return _fused_sigmoid_gating_delta_rule_decode_gfx906(
                A_log=A_log,
                a=a,
                b=b,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                beta=beta,
                threshold=threshold,
                scale=scale,
                initial_state=initial_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        return _fused_sigmoid_gating_delta_rule_update_gfx906_eager(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            beta=beta,
            threshold=threshold,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 4

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]}"
            f" when using `cu_seqlens`. Please flatten variable-length"
            f" inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state
