# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx906 (Z100) GDN MTP speculative-decode update, native SSM-cache layout.

Production shape (Qwen3.5-27B-FP8, TP=4, one rank): 1 request, up to 4
speculative tokens, 12 value heads (48/4), 4 q/k heads (16/4), key and value
head dim 128, float32 recurrent state.

Why a second kernel exists
--------------------------
``csrc/mamba/gfx906_decode_kernels.cu`` launches the MTP update as
``ceil(requests * kv_heads * value_dim / 256)`` blocks of 256 threads, one
thread per (request, head, value) column, walking the whole token chain
serially.  The persistent cache is allocated by
``MambaStateShapeCalculator.gated_delta_net_state_shape`` as
``(num_v_heads/tp, head_v_dim, head_k_dim)``, i.e. ``[slot][head][v][k]`` with
**k contiguous**.  In that layout the per-thread key loop is contiguous but
consecutive threads (value index) are 128 floats apart, so every warp-level
state load touches 32 distinct cache lines.

Measured on Z100L, GPU 1 (tools/dcu/perf1_gdn_bench.py, T=3, heads=4,
kv_heads=12, fp32 state):

    v1 (native, production)             714 us/call  (742 us kernel time)
    v1 (same kernel, v-contiguous view)  48 us/call  ( 27 us kernel time)

The 26x gap is the layout, not the 6-block grid (v2, the warp-cooperative
kernel, is only marginally faster at 12 blocks).  This module keeps the
production layout and instead maps **consecutive lanes onto consecutive key
indices of one value row**, so each state row (512 B) is a fully coalesced
load/store, and splits the value dimension across programs so the grid
becomes ``(VD/BV, kv_heads, requests)`` (384-1536 programs) instead of 6.

Numerics are bit-comparable with the C++ kernel's contract: fp32 accumulate,
same softplus/decay/beta expressions, same per-token source/destination slot
rules, same ``out = 0`` when a slot is invalid.  Verified

* on CPU under ``TRITON_INTERPRET=1`` against a pure-torch port of the C++
  per-token slot logic (tools/dcu/perf1_gdn_interp.py): max|dout| <= 6.0e-08,
  max|dstate| <= 2.4e-07 over 10 cases, and exactly 0.0 in the negative-slot
  cases where a broken slot chain makes the C++ kernel re-read a stale slot;
* on GPU against the C++ op and the torch eager reference
  (tools/dcu/perf1_gdn_bench.py, tools/dcu/perf1_gdn_parity.py).

Enabled with ``VLLM_QWEN35_MTP_GDN_TRITON=1``; the default keeps the C++ path.
``VLLM_QWEN35_MTP_GDN_CUSTOM=0`` still short-circuits to the eager bisect path
before either kernel is considered.
"""

import functools
import os
import sys
import time

import torch
import triton
import triton.language as tl


def gdn_mtp_triton_enabled() -> bool:
    return os.getenv("VLLM_QWEN35_MTP_GDN_TRITON", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@functools.lru_cache(maxsize=1)
def _default_launch_cfg() -> tuple[int, int]:
    return (
        int(os.getenv("VLLM_QWEN35_MTP_GDN_TRITON_BV", "4")),
        int(os.getenv("VLLM_QWEN35_MTP_GDN_TRITON_WARPS", "1")),
    )


@functools.lru_cache(maxsize=1)
def _probe_enabled() -> bool:
    """In-worker per-call timing (VLLM_QWEN35_MTP_GDN_TRITON_PROBE=1).

    Added to explain the live-vs-probe gap of the Perf1 A/B: the standalone
    probe sees ~19 us of kernel and ~120 us of host time per call, yet the e2e
    arm is 13.8 ms/step slower than the C++ path.  This records, inside the
    serving worker, the host time and the GPU kernel time per call, keyed by
    the live launch signature, using a 64-call-old CUDA-event ring so nothing
    synchronises.
    """
    return os.getenv("VLLM_QWEN35_MTP_GDN_TRITON_PROBE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class _Probe:
    """Per-call host/GPU timing with a 64-call-old CUDA-event ring."""

    RING = 64

    def __init__(self) -> None:
        self.ready = False
        self.agg: dict[tuple, list] = {}
        self.calls = 0
        self.samples = 0
        self.idx = 0
        self.keys: list[tuple] = [()] * self.RING
        self.host_ring: list[float] = [0.0] * self.RING
        self.gap_ring: list[float] = [0.0] * self.RING
        self.last_end = 0.0

    def _drain(self, slot: int) -> None:
        """Read the entry RING calls old from this slot (long finished)."""
        self.stop[slot].synchronize()
        gpu_us = self.start[slot].elapsed_time(self.stop[slot]) * 1000.0
        self.agg.setdefault(self.keys[slot], []).append(
            (gpu_us, self.host_ring[slot], self.gap_ring[slot])
        )
        self.samples += 1
        if self.samples % 64 == 0:
            self.report()

    def start_call(self, key: tuple) -> int:
        if not self.ready:
            self.start = [
                torch.cuda.Event(enable_timing=True) for _ in range(self.RING)
            ]
            self.stop = [
                torch.cuda.Event(enable_timing=True) for _ in range(self.RING)
            ]
            self.ready = True
        slot = self.idx % self.RING
        # Drain BEFORE overwriting this slot: it still holds the call from
        # RING calls ago.  Draining here (not in end_call) is what keeps the
        # hot path free of synchronisation -- syncing the *current* call's
        # stop event would stall the whole pipeline every call.
        if self.idx >= self.RING:
            self._drain(slot)
        self.keys[slot] = key
        now = time.perf_counter()
        # wall time since the previous call returned: the full per-call host
        # cost including everything above this wrapper (dispatch, kwargs, ...).
        self.interval_us = (now - self.last_end) * 1e6 if self.idx else 0.0
        self.start[slot].record()
        return slot

    def end_call(self, slot: int, host_us: float) -> None:
        self.last_end = time.perf_counter()
        self.host_ring[slot] = host_us
        self.gap_ring[slot] = self.interval_us
        self.stop[slot].record()
        self.calls += 1
        self.idx += 1

    def report(self) -> None:
        print(
            f"GDN_TRITON_PROBE calls={self.calls} samples={self.samples}",
            file=sys.stderr,
        )
        for key, vals in self.agg.items():
            gpu = sum(v[0] for v in vals) / len(vals)
            host = sum(v[1] for v in vals) / len(vals)
            gap = sum(v[2] for v in vals) / len(vals)
            print(
                f"GDN_TRITON_PROBE_RING {key} n={len(vals)} "
                f"host_us={host:.1f} gpu_us={gpu:.1f} interval_us={gap:.1f}",
                file=sys.stderr,
            )
        self.agg.clear()


_PROBE = _Probe()


@triton.jit
def _gdn_mtp_update_kernel(
    A_log_ptr,
    a_ptr,
    b_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    si_ptr,
    cu_ptr,
    acc_ptr,
    out_ptr,
    heads,
    kv_heads,
    beta,
    threshold,
    scale,
    st_slot,
    st_h,
    st_sv,
    st_sk,
    si_req,
    si_tok,
    USE_L2NORM: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BV: tl.constexpr,
):
    """One program per (value block, value head, request); tokens serial.

    q/k/v/a/b are contiguous ([1, T, H, KD], [1, T, HV, VD], [T, HV]); the
    state strides are passed so the kernel also accepts the transposed pool.
    """
    nv = tl.program_id(0)
    hv = tl.program_id(1)
    req = tl.program_id(2)

    seq_start = tl.load(cu_ptr + req)
    seq_end = tl.load(cu_ptr + req + 1)
    num_tokens = seq_end - seq_start
    if num_tokens <= 0:
        return

    q_head = hv // (kv_heads // heads)
    v_offs = nv * BV + tl.arange(0, BV)
    k_offs = tl.arange(0, KD)
    v_mask = v_offs < VD
    row_mask = v_mask[:, None]

    # Token 0 reads the column selected by num_accepted_tokens[req] - 1; token
    # t > 0 reads column t - 1, which this program usually still holds in
    # registers.  The C++ kernel re-reads it from global memory every token, so
    # when the slot chain is broken by a negative (padding) slot index the two
    # must agree: `chain_ok` tracks whether the registers still hold what the
    # next token's source column contains; if not, the row is re-read, which is
    # exactly the C++ behaviour (it reads the stale value in that case).
    pos0 = tl.load(acc_ptr + req) - 1
    pos0_safe = tl.maximum(pos0, 0)
    src_slot0 = tl.load(si_ptr + req * si_req + pos0_safe * si_tok)
    valid0 = (pos0 >= 0) & (src_slot0 >= 0)

    st_base = state_ptr + hv * st_h
    h = tl.load(
        st_base
        + tl.where(valid0, src_slot0, 0) * st_slot
        + v_offs[:, None] * st_sv
        + k_offs[None, :] * st_sk,
        mask=row_mask & valid0,
        other=0.0,
    )
    chain_ok = valid0

    qk_token = heads * KD
    v_token = kv_heads * VD
    out_base = out_ptr + hv * VD + v_offs

    for t in range(0, num_tokens):
        token = seq_start + t
        qk_off = token * qk_token + q_head * KD
        kv = tl.load(k_ptr + qk_off + k_offs).to(tl.float32)
        qv = tl.load(q_ptr + qk_off + k_offs).to(tl.float32)
        if USE_L2NORM:
            k_scale = tl.rsqrt(tl.sum(kv * kv, axis=0) + 1e-6)
            q_scale = tl.rsqrt(tl.sum(qv * qv, axis=0) + 1e-6) * scale
        else:
            k_scale = 1.0
            q_scale = scale
        kq = kv * k_scale
        qq = qv * q_scale

        # v1 semantics: source_state_idx is the index in column pos0 (t == 0)
        # or column t - 1; the token is skipped when either index is negative.
        src_col = tl.where(t == 0, pos0_safe, t - 1)
        src_idx = tl.load(si_ptr + req * si_req + src_col * si_tok)
        src_valid = tl.where(t == 0, valid0, src_idx >= 0)
        dst_slot = tl.load(si_ptr + req * si_req + t * si_tok)
        row_ok = src_valid & (dst_slot >= 0)

        need_mem = ~chain_ok
        h_mem = tl.load(
            st_base
            + tl.where(src_valid, src_idx, 0) * st_slot
            + v_offs[:, None] * st_sv
            + k_offs[None, :] * st_sk,
            mask=row_mask & src_valid & need_mem,
            other=0.0,
        )
        h_in = tl.where(need_mem, h_mem, h)

        x = tl.load(a_ptr + token * kv_heads + hv).to(tl.float32) + tl.load(
            dt_bias_ptr + hv
        ).to(tl.float32)
        bx = beta * x
        softplus_x = tl.where(
            bx <= threshold, (1.0 / beta) * tl.log(1.0 + tl.exp(bx)), x
        )
        decay = tl.exp(-tl.exp(tl.load(A_log_ptr + hv).to(tl.float32)) * softplus_x)
        beta_t = tl.sigmoid(tl.load(b_ptr + token * kv_heads + hv).to(tl.float32))

        hd = h_in * decay
        dot = tl.sum(hd * kq[None, :], axis=1)
        vin = tl.load(
            v_ptr + token * v_token + hv * VD + v_offs, mask=v_mask, other=0.0
        )
        delta = (vin.to(tl.float32) - dot) * beta_t
        hn = hd + delta[:, None] * kq[None, :]
        o = tl.sum(hn * qq[None, :], axis=1)

        dst_ok = row_ok
        tl.store(
            st_base
            + tl.where(dst_ok, dst_slot, 0) * st_slot
            + v_offs[:, None] * st_sv
            + k_offs[None, :] * st_sk,
            hn,
            mask=row_mask & dst_ok,
        )
        tl.store(
            out_base + token * v_token,
            tl.where(dst_ok, o, 0.0).to(out_ptr.dtype.element_ty),
            mask=v_mask,
        )
        # The registers now hold the state written to dst_slot, i.e. what the
        # next token's source column holds -- but only if this token stored.
        # The reload below must see the same value the store wrote, so the
        # register copy is rounded to the state dtype first (a no-op for the
        # float32 pool the model actually uses).
        h = hn.to(state_ptr.dtype.element_ty).to(tl.float32)
        chain_ok = dst_ok


def gdn_mtp_gfx906_triton(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    beta: float,
    threshold: float,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    block_v: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Drop-in replacement for the C++ MTP update, returning [T, HV, VD]."""
    t0 = time.perf_counter()
    requests, max_query_len = state_indices.shape
    tokens, heads, key_dim = q.shape[1], q.shape[2], q.shape[3]
    kv_heads, value_dim = v.shape[2], v.shape[3]
    a2 = a if a.dim() == 2 else a.squeeze(0)
    b2 = b if b.dim() == 2 else b.squeeze(0)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert a2.is_contiguous() and b2.is_contiguous()
    assert cu_seqlens.is_contiguous() and state_indices.is_contiguous()
    assert num_accepted_tokens.is_contiguous()
    assert kv_heads % heads == 0
    assert state.dim() == 4 and state.size(1) == kv_heads
    assert key_dim == state.size(3) and value_dim == state.size(2)

    out = torch.empty((tokens, kv_heads, value_dim), device=q.device, dtype=q.dtype)
    if tokens == 0 or requests == 0:
        return out

    bv = block_v or _default_launch_cfg()[0]
    nw = num_warps or _default_launch_cfg()[1]
    bv = min(bv, triton.next_power_of_2(value_dim))
    grid = (triton.cdiv(value_dim, bv), kv_heads, requests)
    probe_slot = -1
    if _probe_enabled():
        probe_slot = _PROBE.start_call(
            (
                tokens,
                requests,
                heads,
                kv_heads,
                key_dim,
                value_dim,
                bv,
                nw,
                grid,
                str(state.dtype).replace("torch.", ""),
            )
        )
    _gdn_mtp_update_kernel[grid](
        A_log,
        a2,
        b2,
        dt_bias,
        q,
        k,
        v,
        state,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        out,
        heads,
        kv_heads,
        beta,
        threshold,
        scale,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        state_indices.stride(0),
        state_indices.stride(1),
        USE_L2NORM=use_qk_l2norm_in_kernel,
        KD=key_dim,
        VD=value_dim,
        BV=bv,
        num_warps=nw,
    )
    if probe_slot >= 0:
        _PROBE.end_call(probe_slot, (time.perf_counter() - t0) * 1e6)
    return out
