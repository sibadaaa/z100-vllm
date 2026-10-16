# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx906 decode scheduling tests for the v1 Triton unified attention op.

Covers the scheduling knobs migrated from the legacy attention op path:
decode num_warps policy, GQA6 wide-head BLOCK_M=8, exact single-sequence
query-block counts, context-scaled split-KV segments, and end-to-end decode
output equivalence against a torch eager reference.
"""

import pytest
import torch

import vllm.v1.attention.ops.triton_unified_attention as unified_attn
from vllm.platforms import current_platform


def _is_gfx906() -> bool:
    capability = current_platform.get_device_capability()
    return (
        current_platform.is_rocm()
        and capability is not None
        and capability.major == 9
        and capability.minor == 0
    )


def test_decode_num_warps_scales_with_head_and_context():
    assert unified_attn._decode_num_warps(256, 4096) == 4
    assert unified_attn._decode_num_warps(129, 1536) == 4
    assert unified_attn._decode_num_warps(256, 1535) == 2
    assert unified_attn._decode_num_warps(128, 16384) == 2


def test_decode_block_m_uses_eight_for_wide_head_gqa6():
    assert unified_attn._decode_block_m(256, 6) == 8
    assert unified_attn._decode_block_m(256, 8) is None
    assert unified_attn._decode_block_m(128, 6) is None


def test_num_query_blocks_exact_for_single_sequence():
    # decode with BLOCK_Q=1: exact count is 1, upper bound would be 2
    assert unified_attn._num_query_blocks(1, 1, 1) == 1
    # prefill with BLOCK_Q=2: exact ceil vs upper bound floor+1
    assert unified_attn._num_query_blocks(9263, 1, 2) == 4632
    # batched decode keeps the safe upper bound with mapping gaps
    assert unified_attn._num_query_blocks(4, 4, 1) == 4 // 1 + 4


def test_decode_num_segments_scales_with_context():
    seg = unified_attn._decode_num_segments
    # short context stays at the default
    assert seg(1024, 1, 4, 6) == 16
    assert seg(2047, 1, 4, 6) == 16
    # single-sequence GQA6 decode: 4-CTA base grid grows to the 128 cap
    assert seg(2048, 1, 4, 6) == 128
    assert seg(16384, 1, 4, 6) == 128
    assert seg(24576, 1, 4, 6) == 128
    # batched decode reaches the target grid with fewer segments
    assert seg(4096, 4, 4, 6) == 32
    assert seg(4096, 16, 4, 6) == 16


def _reference_decode(q, k_cache, v_cache, block_table, seqused_k, scale):
    """Paged KV decode reference in torch eager."""
    num_seqs, num_q_heads, head_size = q.shape
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    num_queries_per_kv = num_q_heads // num_kv_heads
    outs = []
    for i in range(num_seqs):
        seq_len = int(seqused_k[i])
        num_blocks = (seq_len + block_size - 1) // block_size
        ids = block_table[i, :num_blocks].to(torch.long)
        keys = k_cache.index_select(0, ids).reshape(-1, num_kv_heads, head_size)
        values = v_cache.index_select(0, ids).reshape(-1, num_kv_heads, head_size)
        keys, values = keys[:seq_len], values[:seq_len]
        qh = q[i].reshape(num_kv_heads, num_queries_per_kv, head_size)
        scores = torch.einsum("hgd,thd->hgt", qh.float(), keys.float()) * scale
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        out = torch.einsum("hgt,thd->hgd", probs.float(), values.float())
        outs.append(out.reshape(num_q_heads, head_size))
    return torch.stack(outs)


@pytest.mark.skipif(not _is_gfx906(), reason="gfx906-specific Triton path")
@pytest.mark.parametrize("kv_len", [1024, 4096, 24576])
@pytest.mark.parametrize("num_seqs", [1, 2])
@torch.inference_mode()
def test_v1_unified_attention_gqa6_decode_matches_reference(kv_len: int, num_seqs: int):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    num_kv_heads, num_queries_per_kv, head_size = 4, 6, 256
    num_q_heads = num_kv_heads * num_queries_per_kv
    block_size = 800
    scale = head_size**-0.5

    num_blocks_per_seq = (kv_len + block_size - 1) // block_size
    num_blocks = num_seqs * num_blocks_per_seq
    q = torch.randn(num_seqs, num_q_heads, head_size, device=device, dtype=dtype)
    k = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    cu_seqlens_q = torch.arange(num_seqs + 1, device=device, dtype=torch.int32)
    seqused_k = torch.full((num_seqs,), kv_len, device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        num_seqs, -1
    )
    num_par_softmax_segments = unified_attn.gfx906_decode_segments_capacity()
    headdim_padded = 256
    softmax_segm_output = torch.empty(
        num_seqs,
        num_q_heads,
        num_par_softmax_segments,
        headdim_padded,
        dtype=torch.float32,
        device=device,
    )
    softmax_segm_max = torch.empty(
        num_seqs,
        num_q_heads,
        num_par_softmax_segments,
        dtype=torch.float32,
        device=device,
    )
    softmax_segm_expsum = torch.empty_like(softmax_segm_max)

    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=32,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    ref = _reference_decode(q, k, v, block_table, seqused_k, scale).to(q.dtype)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _is_gfx906(), reason="gfx906-specific Triton path")
@torch.inference_mode()
def test_v1_unified_attention_prefill_split_kv_matches_reference(monkeypatch):
    """Opt-in prefill split-KV (3D) path vs the eager reference."""
    monkeypatch.setattr(unified_attn, "ENABLE_GFX906_ATTN_PREFILL_3D", True)
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    num_kv_heads, num_queries_per_kv, head_size = 4, 6, 256
    num_q_heads = num_kv_heads * num_queries_per_kv
    block_size = 800
    # q_len spans multiple 32-row query blocks and several KV tiles so the
    # segment partition is exercised on both sides of a tile boundary
    q_len, kv_len = 1023, 2048
    scale = head_size**-0.5

    num_blocks = (kv_len + block_size - 1) // block_size
    q = torch.randn(q_len, num_q_heads, head_size, device=device, dtype=dtype)
    k = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([kv_len], device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        1, -1
    )

    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_len,
        seqused_k=seqused_k,
        max_seqlen_k=kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=32,
        num_par_softmax_segments=16,
        softmax_segm_output=None,
        softmax_segm_max=None,
        softmax_segm_expsum=None,
    )
    # the split-KV prefill route must have provisioned its workspace
    assert unified_attn._prefill_segm_state is not None
    assert unified_attn._prefill_segm_state["segs"] == unified_attn.GFX906_PREFILL_SEGMENTS

    context = kv_len - q_len
    keys = k.reshape(-1, num_kv_heads, head_size)[:kv_len]
    values = v.reshape(-1, num_kv_heads, head_size)[:kv_len]
    ref = torch.empty_like(q)
    for t in range(q_len):
        end = context + t + 1
        q_t = q[t].reshape(num_kv_heads, num_queries_per_kv, head_size)
        s = torch.einsum("hgd,thd->hgt", q_t.float(), keys[:end].float()) * scale
        p = torch.softmax(s, dim=-1)
        o = torch.einsum("hgt,thd->hgd", p.float(), values[:end].float())
        ref[t] = o.reshape(num_q_heads, head_size)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _is_gfx906(), reason="gfx906-specific Triton path")
@torch.inference_mode()
def test_v1_unified_attention_prefill_matches_reference():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    num_kv_heads, num_queries_per_kv, head_size = 4, 6, 256
    num_q_heads = num_kv_heads * num_queries_per_kv
    block_size = 800
    q_len, kv_len = 513, 1024
    scale = head_size**-0.5

    num_blocks = (kv_len + block_size - 1) // block_size
    q = torch.randn(q_len, num_q_heads, head_size, device=device, dtype=dtype)
    k = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([kv_len], device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        1, -1
    )

    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_len,
        seqused_k=seqused_k,
        max_seqlen_k=kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=32,
        num_par_softmax_segments=16,
        softmax_segm_output=None,
        softmax_segm_max=None,
        softmax_segm_expsum=None,
    )

    context = kv_len - q_len
    keys = k.reshape(-1, num_kv_heads, head_size)[:kv_len]
    values = v.reshape(-1, num_kv_heads, head_size)[:kv_len]
    # reference: q[t, h*G+g] attends causally over keys[: context + t + 1]
    ref = torch.empty_like(q)
    for t in range(q_len):
        end = context + t + 1
        q_t = q[t].reshape(num_kv_heads, num_queries_per_kv, head_size)
        s = torch.einsum("hgd,thd->hgt", q_t.float(), keys[:end].float()) * scale
        p = torch.softmax(s, dim=-1)
        o = torch.einsum("hgt,thd->hgd", p.float(), values[:end].float())
        ref[t] = o.reshape(num_q_heads, head_size)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _is_gfx906(), reason="gfx906-specific Triton path")
@torch.inference_mode()
def test_v1_unified_attention_bidirectional_sliding_window_matches_reference():
    """Non-causal sliding windows include nearby future keys across tiles."""
    torch.manual_seed(11)
    device = "cuda"
    dtype = torch.float16
    num_kv_heads, num_queries_per_kv, head_size = 1, 2, 64
    num_q_heads = num_kv_heads * num_queries_per_kv
    q_len, context_len, block_size = 35, 19, 16
    kv_len = context_len + q_len
    sliding_window = 5
    scale = head_size**-0.5

    num_blocks = (kv_len + block_size - 1) // block_size
    q = torch.randn(q_len, num_q_heads, head_size, device=device, dtype=dtype)
    k = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([kv_len], device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        1, -1
    )

    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_len,
        seqused_k=seqused_k,
        max_seqlen_k=kv_len,
        softmax_scale=scale,
        causal=False,
        window_size=(sliding_window - 1, sliding_window - 1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )

    keys = k.reshape(-1, num_kv_heads, head_size)[:kv_len]
    values = v.reshape(-1, num_kv_heads, head_size)[:kv_len]
    ref = torch.empty_like(q)
    for t in range(q_len):
        q_pos = context_len + t
        key_pos = torch.arange(kv_len, device=device)
        visible = (key_pos - q_pos).abs() < sliding_window
        q_t = q[t].reshape(num_kv_heads, num_queries_per_kv, head_size)
        scores = torch.einsum(
            "hgd,thd->hgt", q_t.float(), keys.float()
        ) * scale
        scores = scores.masked_fill(~visible[None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        result = torch.einsum("hgt,thd->hgd", probs, values.float())
        ref[t] = result.reshape(num_q_heads, head_size)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


# ---- speculative-decoding verify steps stay decode-shaped --------------------
# A k+1-token verify pass (MTP k=2 -> 3 query tokens) used to be classified as
# prefill because max_seqlen_q > 1, which sent it to the unsplit 2D kernel: at
# 32k that is 18.9 ms/layer instead of 0.9 ms/layer, on every decode step.


def test_small_q_decode_routing_decision():
    tuned = unified_attn._decode_block_m(256, 6)
    assert tuned == 8
    # a 3-token verify pass on the tuned shape is decode-shaped
    assert unified_attn.gfx906_small_q_decode(3, True, tuned, True)
    # plain decode is handled by its own branch, not by this predicate
    assert not unified_attn.gfx906_small_q_decode(1, True, tuned, True)
    # every verify query length up to the ceiling
    assert unified_attn.gfx906_small_q_decode(2, True, tuned, True)
    assert unified_attn.gfx906_small_q_decode(
        unified_attn.GFX906_VERIFY_MAX_QUERY_LEN, True, tuned, True
    )
    # real prefill, missing segment buffers, untuned shapes and other
    # platforms all keep the historical prefill classification
    assert not unified_attn.gfx906_small_q_decode(
        unified_attn.GFX906_VERIFY_MAX_QUERY_LEN + 1, True, tuned, True
    )
    assert not unified_attn.gfx906_small_q_decode(3, False, tuned, True)
    assert not unified_attn.gfx906_small_q_decode(3, True, None, True)
    assert not unified_attn.gfx906_small_q_decode(3, True, tuned, False)


class _KernelSpy:
    """Records launch grids of a @triton.jit kernel without altering it."""

    def __init__(self, kernel, hits: list):
        self._kernel = kernel
        self._hits = hits

    def __getitem__(self, grid):
        self._hits.append(grid)
        return self._kernel[grid]


def _verify_step_inputs(device, dtype, num_q_tokens, kv_len, block_size=16):
    num_kv_heads, num_queries_per_kv, head_size = 1, 6, 256
    num_q_heads = num_kv_heads * num_queries_per_kv
    num_blocks = (kv_len + block_size - 1) // block_size
    q = torch.randn(num_q_tokens, num_q_heads, head_size, device=device, dtype=dtype)
    k = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, num_q_tokens], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([kv_len], device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        1, -1
    )
    return q, k, v, out, cu_seqlens_q, seqused_k, block_table


@pytest.mark.skipif(not _is_gfx906(), reason="gfx906-specific Triton path")
@pytest.mark.parametrize("num_q_tokens", [2, 3, 5])
@torch.inference_mode()
def test_v1_unified_attention_verify_step_uses_split_kv(monkeypatch, num_q_tokens):
    """A k+1-token verify pass must take the 3D split-KV path, and be exact."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    head_size = 256
    num_kv_heads, num_queries_per_kv = 1, 6
    num_q_heads = num_kv_heads * num_queries_per_kv
    kv_len = 4096
    scale = head_size**-0.5
    q, k, v, out, cu_seqlens_q, seqused_k, block_table = _verify_step_inputs(
        device, dtype, num_q_tokens, kv_len
    )

    hits_3d: list = []
    hits_2d: list = []
    monkeypatch.setattr(
        unified_attn,
        "kernel_unified_attention_3d",
        _KernelSpy(unified_attn.kernel_unified_attention_3d, hits_3d),
    )
    monkeypatch.setattr(
        unified_attn,
        "kernel_unified_attention_2d",
        _KernelSpy(unified_attn.kernel_unified_attention_2d, hits_2d),
    )
    monkeypatch.setattr(unified_attn, "ENABLE_GFX906_ATTN_VERIFY_3D", True)

    num_segments = unified_attn.GFX906_MAX_DECODE_SEGMENTS
    segm_shape = (num_q_tokens, num_q_heads, num_segments)
    softmax_segm_output = torch.empty(
        (*segm_shape, head_size), dtype=torch.float32, device=device
    )
    softmax_segm_max = torch.empty(segm_shape, dtype=torch.float32, device=device)
    softmax_segm_expsum = torch.empty_like(softmax_segm_max)

    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=num_q_tokens,
        seqused_k=seqused_k,
        max_seqlen_k=kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=32,
        num_par_softmax_segments=num_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    assert len(hits_2d) == 0, f"verify step fell back to the 2D kernel: {hits_2d}"
    assert len(hits_3d) == 1
    grid_x, grid_y, grid_z = hits_3d[0]
    # one query block per token on the GQA6/256 tuned shape (BLOCK_Q == 1)
    assert grid_x == num_q_tokens
    assert grid_y == num_kv_heads
    assert 0 < grid_z <= num_segments

    context = kv_len - num_q_tokens
    keys = k.reshape(-1, num_kv_heads, head_size)[:kv_len]
    values = v.reshape(-1, num_kv_heads, head_size)[:kv_len]
    ref = torch.empty_like(q)
    for t in range(num_q_tokens):
        end = context + t + 1
        q_t = q[t].reshape(num_kv_heads, num_queries_per_kv, head_size)
        s = torch.einsum("hgd,thd->hgt", q_t.float(), keys[:end].float()) * scale
        p = torch.softmax(s, dim=-1)
        o = torch.einsum("hgt,thd->hgd", p.float(), values[:end].float())
        ref[t] = o.reshape(num_q_heads, head_size)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _is_gfx906(), reason="gfx906-specific Triton path")
@torch.inference_mode()
def test_v1_unified_attention_verify_step_kill_switch_restores_2d(monkeypatch):
    """VLLM_GFX906_ATTN_VERIFY_3D=0 restores the historical 2D routing."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    kv_len = 4096
    num_q_tokens = 3
    num_q_heads = 6
    q, k, v, out, cu_seqlens_q, seqused_k, block_table = _verify_step_inputs(
        device, dtype, num_q_tokens, kv_len
    )

    hits_3d: list = []
    hits_2d: list = []
    monkeypatch.setattr(
        unified_attn,
        "kernel_unified_attention_3d",
        _KernelSpy(unified_attn.kernel_unified_attention_3d, hits_3d),
    )
    monkeypatch.setattr(
        unified_attn,
        "kernel_unified_attention_2d",
        _KernelSpy(unified_attn.kernel_unified_attention_2d, hits_2d),
    )
    monkeypatch.setattr(unified_attn, "ENABLE_GFX906_ATTN_VERIFY_3D", False)

    num_segments = unified_attn.GFX906_MAX_DECODE_SEGMENTS
    segm_shape = (num_q_tokens, num_q_heads, num_segments)
    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=num_q_tokens,
        seqused_k=seqused_k,
        max_seqlen_k=kv_len,
        softmax_scale=256**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=32,
        num_par_softmax_segments=num_segments,
        softmax_segm_output=torch.empty(
            (*segm_shape, 256), dtype=torch.float32, device=device
        ),
        softmax_segm_max=torch.empty(segm_shape, dtype=torch.float32, device=device),
        softmax_segm_expsum=torch.empty(
            segm_shape, dtype=torch.float32, device=device
        ),
    )

    assert len(hits_3d) == 0
    assert len(hits_2d) == 1
