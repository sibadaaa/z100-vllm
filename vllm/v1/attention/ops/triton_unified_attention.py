# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import os
from functools import lru_cache

import torch

from vllm.logger import init_logger
import vllm.envs as envs
from vllm.v1.kv_cache_interface import KVQuantMode


def vllm_is_batch_invariant() -> bool:
    return bool(envs.VLLM_BATCH_INVARIANT)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
is_batch_invariant = vllm_is_batch_invariant()
float8_info = torch.finfo(current_platform.fp8_dtype())
# Longest per-request query (e.g. 1 + num_speculative_tokens) run by split-K.
MAX_3D_QUERY_LEN = 8

# gfx906 decode scheduling knobs, migrated from the legacy attention op path.
# On gfx906 the tuning applies unconditionally; the env var is only an
# emergency kill switch for debugging.
ENABLE_GFX906_ATTN_SCHED_TUNING = os.getenv(
    "VLLM_GFX906_TRITON_ATTN_TUNING", "1"
).lower() in {"1", "true", "yes", "on"}

# Route tuned prefill shapes through the split-KV (3D) kernel on gfx906.
# Independent switch so prefill can be routed independently of the decode
# policy. Opt-in: kernel-level interleaved A/B shows the split-KV kernel
# ~8-9% faster than the 2D kernel at 4k-16k serving shapes, but attention is
# only a fraction of prefill wall time, so the end-to-end gain measured on a
# 27B head-256/GQA6 model was ~1% at 4k and ~2% at 16k tokens. The path also
# costs up to ~0.4 GiB of lazily allocated fp32 segment workspace at the
# default chunk size. Only applies to plain causal attention (no sliding
# window, sinks, alibi, softcap, qq-bias, mm-prefix or fused fp8 output) on
# the tuned head-256/GQA6 shape, and never during CUDA-graph capture (the
# workspace is allocated lazily on first use).
ENABLE_GFX906_ATTN_PREFILL_3D = os.getenv(
    "VLLM_GFX906_TRITON_ATTN_PREFILL_3D", "0"
).lower() in {"1", "true", "yes", "on"}

# Upper bound for split-KV segments the decode policy may request. The
# backend sizes its softmax segment buffers with this value on gfx906.
# Interleaved sweeps at 8k/32k (tools/dcu/perf1_attn_verify.md) put 128 and
# 256 within ~11% of each other and 512+ clearly worse, so the cap is not a
# lever on its own; the win comes from actually reaching this path (below).
GFX906_MAX_DECODE_SEGMENTS = 128

# Query-length ceiling up to which a forward pass is treated as decode-shaped.
# Speculative decoding verifies k+1 tokens per request in one step, so the
# k=2 MTP configuration used in production arrives here with max_seqlen_q == 3.
# Such a step scans the whole KV cache for a handful of query tokens, exactly
# like decode; classifying it as prefill (the old `max_seqlen_q > 1` test) sent
# it to the unsplit 2D kernel, whose grid collapses to one CTA per
# (sequence, kv head) and serializes the entire context scan: 16 full-attention
# layers * (18.9 ms - 0.9 ms) of pure loss at 32k, on every decode step.
# True prefill batches (chunked prefill) stay well above this ceiling.
GFX906_VERIFY_MAX_QUERY_LEN = 8

# Kill switch for the decode-shaped routing above; 0 restores the historical
# `max_seqlen_q > 1 == prefill` classification.
ENABLE_GFX906_ATTN_VERIFY_3D = os.getenv(
    "VLLM_GFX906_ATTN_VERIFY_3D", "1"
).lower() in {"1", "true", "yes", "on"}

# Split-KV segments for the gfx906 prefill 3D path. Prefill already fills the
# GPU with q-block x kv-head CTAs; a small fixed split balances the causal
# load skew between early and late query blocks. Interleaved sweeps over
# segments {2,4,8,16} at 4k-16k serving shapes put 2 and 4 within ~1% of
# each other (8+ regress); 2 halves the fp32 segment workspace, which keeps
# gpu_memory_utilization=0.95 runs within budget at the default chunk size.


def _gfx906_env_int(name: str, default: int) -> int:
    """Read an integer tuning knob, falling back to `default` (S22)."""
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


# S22 prefill tuning knobs. The defaults reproduce the swept configuration
# (BLOCK_M 32 / TILE 32 / 4 warps / 1 stage / 2 segments) exactly, so the
# engine is unchanged unless one of these is set; they exist so the constants
# can be A/B'd without editing this file between arms.
# See tools/dcu/s22_attn_prefill_probe.py for the measurements.
GFX906_PREFILL_BLOCK_M_OVERRIDE = _gfx906_env_int(
    "VLLM_GFX906_ATTN_PREFILL_BLOCK_M", 0
)
GFX906_PREFILL_TILE_OVERRIDE = _gfx906_env_int("VLLM_GFX906_ATTN_PREFILL_TILE", 0)
GFX906_PREFILL_WARPS = _gfx906_env_int("VLLM_GFX906_ATTN_PREFILL_WARPS", 4)
GFX906_PREFILL_STAGES = _gfx906_env_int("VLLM_GFX906_ATTN_PREFILL_STAGES", 1)
GFX906_PREFILL_SEGMENTS = _gfx906_env_int("VLLM_GFX906_ATTN_PREFILL_SEGMENTS", 2)
GFX906_PREFILL_3D_WARPS = _gfx906_env_int("VLLM_GFX906_ATTN_PREFILL_3D_WARPS", 4)
GFX906_PREFILL_3D_STAGES = _gfx906_env_int("VLLM_GFX906_ATTN_PREFILL_3D_STAGES", 1)
# Minimum KV context length for the prefill split-KV path. The S22 controlled
# arms show the 3D path at the shipped two segments is a large win on long
# prefills (p32k TTFT 178.6 -> 169.1 s, -5.3%) but a large loss on short ones
# (c1 1429-token prefill 2612 -> 3376 ms per iteration), so production keeps
# the win only above this context length. 0 keeps the historical behaviour
# (3D whenever the switch is on).
GFX906_PREFILL_3D_MIN_CTX = _gfx906_env_int(
    "VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX", 0
)

# Lazily grown workspace for the prefill split-KV path. Unlike the decode
# segment buffers (sized for the decode batch threshold and provided by the
# backend), prefill needs one row per scheduled token, so the workspace is
# allocated on first use and grown to the largest batch seen. Allocation
# happens outside CUDA-graph capture only, and never runs before KV-cache
# profiling (the profiling forward passes attention metadata as None).
_prefill_segm_state: dict | None = None


def _prefill_segment_buffers(
    num_tokens: int,
    num_query_heads: int,
    num_segments: int,
    headdim_padded: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _prefill_segm_state
    rows = triton.next_power_of_2(num_tokens)
    st = _prefill_segm_state
    if (
        st is None
        or st["rows"] < rows
        or st["heads"] != num_query_heads
        or st["segs"] != num_segments
        or st["dim"] != headdim_padded
        or st["device"] != device
    ):
        _prefill_segm_state = {
            "rows": rows,
            "heads": num_query_heads,
            "segs": num_segments,
            "dim": headdim_padded,
            "device": device,
            "output": torch.empty(
                (rows, num_query_heads, num_segments, headdim_padded),
                dtype=torch.float32,
                device=device,
            ),
            "max": torch.empty(
                (rows, num_query_heads, num_segments),
                dtype=torch.float32,
                device=device,
            ),
            "expsum": torch.empty(
                (rows, num_query_heads, num_segments),
                dtype=torch.float32,
                device=device,
            ),
        }
        st = _prefill_segm_state
    return st["output"], st["max"], st["expsum"]


@lru_cache(maxsize=1)
def _is_gfx906_rocm() -> bool:
    capability = current_platform.get_device_capability()
    return (
        current_platform.is_rocm()
        and capability is not None
        and capability.major == 9
        and capability.minor == 0
    )


def _gfx906_sched_tuning_enabled() -> bool:
    return ENABLE_GFX906_ATTN_SCHED_TUNING and _is_gfx906_rocm()


def gfx906_decode_segments_capacity() -> int:
    """Segment-buffer capacity the backend should allocate on this platform."""
    return GFX906_MAX_DECODE_SEGMENTS if _gfx906_sched_tuning_enabled() else 0


def _decode_num_warps(head_size: int, max_seqlen_k: int) -> int:
    # Wide-head decodes keep 4 warps busy once the KV scan is long enough;
    # short contexts run faster with 2 warps on gfx906.
    if head_size > 128 and max_seqlen_k >= 1536:
        return 4
    return 2


def _decode_block_m(head_size: int, num_queries_per_kv: int) -> int | None:
    # GQA6 with 256-dim heads: one query token spans exactly 6 rows, so an
    # 8-row block covers a single token without padding a second one.
    if head_size == 256 and num_queries_per_kv == 6:
        return 8
    return None


def _prefill_block_m(head_size: int, num_queries_per_kv: int) -> int | None:
    # GQA6 with 256-dim heads: a 32-row block spans 5 query tokens (30 of 32
    # rows carry work) and reuses each K/V tile across them, versus only 2
    # tokens per 16-row block. Sweeps over BLOCK_M/TILE/num_warps/num_stages
    # at 4k-16k prefill shapes favor 32/32/4/1 uniformly on gfx906.
    if head_size == 256 and num_queries_per_kv == 6:
        return GFX906_PREFILL_BLOCK_M_OVERRIDE or 32
    return None


def _num_query_blocks(num_query_tokens: int, num_seqs: int, block_q: int) -> int:
    # Exact block count for single-sequence batches. For multiple sequences,
    # retain the mapping gaps expected by find_seq_idx and use the safe
    # upper bound without realizing query_lens on the CPU.
    if num_seqs == 1:
        return (num_query_tokens + block_q - 1) // block_q
    return num_query_tokens // block_q + num_seqs


def gfx906_small_q_decode(
    max_seqlen_q: int,
    segm_buffers_ready: bool,
    tuned_decode_shape: int | None,
    sched_tuning: bool,
) -> bool:
    """Whether a >1 query-length forward is still decode-shaped on gfx906.

    A speculative-decoding verify pass carries k+1 query tokens but scans the
    whole KV cache once, so it belongs on the split-KV decode path. Only the
    tuned decode shape is rerouted, which keeps every other model on the
    historical `max_seqlen_q > 1 implies prefill` classification.
    """
    return (
        sched_tuning
        and max_seqlen_q > 1
        and ENABLE_GFX906_ATTN_VERIFY_3D
        and segm_buffers_ready
        and tuned_decode_shape is not None
        and max_seqlen_q <= GFX906_VERIFY_MAX_QUERY_LEN
    )


def _decode_num_segments(
    max_seqlen_k: int,
    num_query_blocks: int,
    num_kv_heads: int,
    num_queries_per_kv: int,
) -> int:
    # Keep short-context decode at 16 segments, but increase split-KV
    # parallelism on gfx906 once the KV scan is large enough to amortize
    # the extra reduction work.
    if max_seqlen_k < 2048:
        return 16

    base_grid_size = max(num_query_blocks * num_kv_heads, 1)
    target_grid_size = 1024 if num_queries_per_kv <= 2 else 512
    if max_seqlen_k >= 8192:
        target_grid_size = max(target_grid_size, 1024)
    if max_seqlen_k >= 16384 and num_queries_per_kv == 1:
        target_grid_size = 2048
    if max_seqlen_k >= 24576:
        target_grid_size = 4096 if num_queries_per_kv == 1 else 2048

    segments = (target_grid_size + base_grid_size - 1) // base_grid_size
    return min(128, max(16, triton.next_power_of_2(segments)))


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_CAUSAL: tl.constexpr,  # bool
    USE_PER_SEQ_CAUSAL: tl.constexpr,  # bool
    per_seq_causal_ptr,  # [num_seqs] int32, or None
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    if CAUSAL:
        max_seq_prefix_len = (
            context_len
            + q_block_local_idx * BLOCK_Q
            + (BLOCK_M - 1) // num_queries_per_kv
            + 1
        )
        if USE_MM_PREFIX:
            # image bidirectional attention ranges require a full range
            # including q_block padding to make sure doc mask is correct
            max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
        else:
            # adjust for potential padding in the last q_block by considering
            # the actual sequence length
            max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    else:
        # Bidirectional attention: every query can reach every key, including
        # the query block's own (later) positions.
        max_seq_prefix_len = seq_len

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs] (causal) or
        # [q_abs - SLIDING_WINDOW + 1, q_abs + SLIDING_WINDOW - 1]
        # (bidirectional), where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        # extended by SLIDING_WINDOW - 1 when attention is bidirectional.
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = (
            context_len + qpos_hi
            if CAUSAL
            else context_len + qpos_hi + SLIDING_WINDOW - 1
        )
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            # A drafter can run its own dtype (e.g. bf16 DFlash behind an fp16
            # target) while its KV cache keeps the run's cache dtype; cast to
            # the query dtype so tl.dot sees matching operands.
            K = K_load if K_load.dtype == Q.dtype else K_load.to(Q.dtype)

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load if V_load.dtype == Q.dtype else V_load.to(Q.dtype)

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos[:, None]
        if CAUSAL:
            seq_mask = seq_offset[None, :] <= query_abs_pos
        else:
            # Bidirectional (DFlash drafter blocks): every valid key is
            # visible, then the sliding window below narrows it. Keys past the
            # end of the sequence must stay masked -- their K/V tiles load as
            # zeros, so they would otherwise collect probability mass.
            seq_mask = seq_offset[None, :] < seq_len

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        # Order must match FlexAttention: (causal AND sliding_window) OR mm_prefix
        if SLIDING_WINDOW > 0:
            if CAUSAL:
                seq_mask = seq_mask & (
                    (query_abs_pos - seq_offset) < SLIDING_WINDOW
                )
            else:
                seq_mask = seq_mask & (
                    tl.abs(query_abs_pos - seq_offset) < SLIDING_WINDOW
                )

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )

                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - (context_len + query_pos[:, None])
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            dist = context_len + qpos_lo - seq_offset[:, None]
            if USE_PER_SEQ_CAUSAL:
                is_causal_seq = tl.load(per_seq_causal_ptr + seq_idx)
                sw_mask_v = tl.where(
                    is_causal_seq,
                    dist < SLIDING_WINDOW,
                    (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW),
                )
            elif USE_CAUSAL:
                sw_mask_v = dist < SLIDING_WINDOW
            else:
                sw_mask_v = (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW)
            V = tl.where(sw_mask_v, V, 0.0)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: apply v_scale to P instead of V.
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            # Keep softmax weights representable when the P @ V dot uses FP8
            # tensor cores. Undo the power-of-two scale after the dot.
            if V.dtype == tl.float8e4nv:
                p_scale = 256.0
            elif V.dtype.is_fp8():
                p_scale = 128.0
            else:
                p_scale = 1.0
            acc += tl.dot((P * p_scale).to(V.dtype), V) / p_scale

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]
    if USE_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


@triton.jit
def kernel_unified_attention_3d(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size_padded]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    CAUSAL: tl.constexpr,  # bool
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    if CAUSAL:
        max_seq_prefix_len = (
            context_len
            + q_block_local_idx * BLOCK_Q
            + (BLOCK_M - 1) // num_queries_per_kv
            + 1
        )
        # adjust for potential padding in the last q_block by considering the
        # actual sequence length
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    else:
        # Bidirectional attention: every query can reach every key, including
        # the query block's own (later) positions.
        max_seq_prefix_len = seq_len

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs] (causal) or
        # [q_abs - SLIDING_WINDOW + 1, q_abs + SLIDING_WINDOW - 1]
        # (bidirectional), where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        # extended by SLIDING_WINDOW - 1 when attention is bidirectional.
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = (
            context_len + qpos_hi
            if CAUSAL
            else context_len + qpos_hi + SLIDING_WINDOW - 1
        )
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(
        max(segm_idx * tiles_per_segment, tile_start),
        min((segm_idx + 1) * tiles_per_segment, tile_end),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            # A drafter can run its own dtype (e.g. bf16 DFlash behind an fp16
            # target) while its KV cache keeps the run's cache dtype; cast to
            # the query dtype so tl.dot sees matching operands.
            K = K_load if K_load.dtype == Q.dtype else K_load.to(Q.dtype)

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load if V_load.dtype == Q.dtype else V_load.to(Q.dtype)

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = context_len + query_pos[:, None]
        if CAUSAL:
            seq_mask = seq_offset[None, :] <= query_abs_pos
        else:
            # Bidirectional (DFlash drafter blocks): every valid key is
            # visible, then the sliding window below narrows it. Keys past the
            # end of the sequence must stay masked -- their K/V tiles load as
            # zeros, so they would otherwise collect probability mass.
            seq_mask = seq_offset[None, :] < seq_len

        # Apply sliding window to base mask BEFORE mm_prefix OR.
        # Order must match FlexAttention: (causal AND sliding_window) OR mm_prefix
        if SLIDING_WINDOW > 0:
            if CAUSAL:
                seq_mask = seq_mask & (
                    (query_abs_pos - seq_offset) < SLIDING_WINDOW
                )
            else:
                seq_mask = seq_mask & (
                    tl.abs(query_abs_pos - seq_offset) < SLIDING_WINDOW
                )

        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )

                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - (context_len + query_pos[:, None])
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    # Graph padding must be rejected before sequence lookup or scratch access.
    if query_token_idx >= tl.load(query_start_len_ptr + num_seqs):
        return

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    if seq_len <= 0:
        return

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    # Empty segments must not set the maximum. Unvisited segments retain
    # the M=-inf,L=1 initial state.
    valid_segment = segm_mask & (segm_expsum > 0) & (segm_max != float("-inf"))
    segm_max = tl.where(valid_segment, segm_max, float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segment_scale = tl.where(valid_segment, tl.exp(segm_max - overall_max), 0.0)
    segm_expsum = segm_expsum * segment_scale
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= segment_scale[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization.

    For Gemma3, use 32 for both prefill and decode to better utilize
    the larger head dimension (128/256). For other models, use
    the default vLLM behavior.
    """
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    return 16 if element_size >= 2 else 32


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    use_alibi_sqrt=False,
    # The upstream triton_attn backend passes these newer knobs by keyword.
    # This gfx906 fork implements none of the features behind them yet;
    # accept them for call compatibility and fail loudly when a run
    # actually requests one (see the guard at the top of the body).
    rswa_prefix_lens=None,
    rswa_window: int | None = None,
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE,
    k_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    v_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    chunk_lookback: int = -1,
    # Tensor-descriptor loads (Intel Xe2/Xe3) are numerics-neutral; this
    # fork never uses tensor descriptors, so the flag is accepted and
    # ignored.
    use_td: bool = False,
    # Gemma4 mm_prefix + sliding-window clamping is not implemented here.
    mm_prefix_clamp_sliding_window: bool = False,
):
    if not causal:
        logger.info_once(
            "gfx906 unified attention: running the bidirectional (non-causal) "
            "path for a drafter."
        )
    if not causal and os.environ.get("VLLM_GFX906_ATTN_FORCE_CAUSAL") == "1":
        # A/B arm: force the DFlash drafter's bidirectional block attention onto
        # the causal path, to quantify how much of the acceptance length comes
        # from real non-causal masking.
        logger.warning_once(
            "gfx906 unified attention: forcing non-causal requests onto the "
            "causal path (VLLM_GFX906_ATTN_FORCE_CAUSAL=1)."
        )
        causal = True
    assert q_descale is None, "Q scales not supported"

    if (
        rswa_prefix_lens is not None
        or rswa_window is not None
        or chunk_lookback != -1
        or mm_prefix_clamp_sliding_window
        or k_scale_cache is not None
        or v_scale_cache is not None
        or kv_quant_mode not in (KVQuantMode.NONE, KVQuantMode.FP8_PER_TENSOR)
    ):
        # Per-tensor FP8 needs no new plumbing: its scales arrive through
        # the k_descale / v_descale args above, as before the refactor.
        raise NotImplementedError(
            "The gfx906 Triton unified-attention fork does not implement "
            "R-SWA, chunked attention, per-token-head/INT4 KV caches, or "
            f"mm-prefix sliding-window clamping (kv_quant_mode={kv_quant_mode}, "
            f"chunk_lookback={chunk_lookback}, rswa_window={rswa_window})."
        )

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    gfx906_sched = _gfx906_sched_tuning_enabled()

    tuned_decode_shape = _decode_block_m(head_size, num_queries_per_kv)
    segm_buffers_ready = (
        seq_threshold_3D is not None
        and num_par_softmax_segments is not None
        and softmax_segm_output is not None
        and softmax_segm_max is not None
        and softmax_segm_expsum is not None
    )
    # Decode and a speculative-decoding verify pass both scan the whole KV
    # cache for a handful of query tokens, so both must stay decode-shaped.
    # Restricted to the tuned decode shape so every other model keeps the
    # pre-existing routing.
    small_q_decode = gfx906_small_q_decode(
        max_seqlen_q, segm_buffers_ready, tuned_decode_shape, gfx906_sched
    )
    decode_block_m = (
        tuned_decode_shape
        if gfx906_sched and (max_seqlen_q == 1 or small_q_decode)
        else None
    )
    prefill_block_m = (
        _prefill_block_m(head_size, num_queries_per_kv)
        if gfx906_sched and max_seqlen_q > 1 and not small_q_decode
        else None
    )
    BLOCK_M = decode_block_m or prefill_block_m or (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    if gfx906_sched:
        total_num_q_blocks = _num_query_blocks(q.shape[0], num_seqs, BLOCK_Q)
    else:
        total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # Tile sizes for prefill and decode. Gemma3 models use optimized values.
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    TILE_SIZE_PREFILL = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=True,
    )
    if GFX906_PREFILL_TILE_OVERRIDE:
        TILE_SIZE_PREFILL = GFX906_PREFILL_TILE_OVERRIDE
    TILE_SIZE_DECODE = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=False,
    )

    def _launch_split_kv(
        segm_output,
        segm_max,
        segm_expsum,
        num_segments,
        tile_size,
        num_warps,
        num_stages=1,
    ):
        kernel_unified_attention_3d[
            (total_num_q_blocks, num_kv_heads, num_segments)
        ](
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            TILE_SIZE=tile_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_ALIBI_SQRT=use_alibi_sqrt,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            USE_MM_PREFIX=use_mm_prefix,
            MAX_MM_RANGES=max_mm_ranges,
            mm_prefix_range_ptr=mm_prefix_range,
            SLIDING_WINDOW=(1 + window_size[0]),
            CAUSAL=causal,
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            NUM_SEGMENTS_PER_SEQ=num_segments,
            # num_warps=None keeps the Triton defaults on non-gfx906 platforms
            **(
                {"num_warps": num_warps, "num_stages": num_stages}
                if num_warps
                else {}
            ),
        )
        assert out.stride(1) == head_size, (
            f"USE_TD_QO requires contiguous output heads "
            f"(out.stride(1) = {out.stride(1)} != head_size = {head_size})."
        )

    # Launch the 2D kernel if
    # 1. No split-K threshold or scratch buffers, or the scratch has fewer
    #    rows than query tokens, or
    # 2. max_seqlen_q > 1 and either it exceeds MAX_3D_QUERY_LEN, masking is
    #    non-causal, per-sequence causal, or mm-prefix, or the tuned
    #    large-head path applies, or
    # 3. The number of Q blocks exceeds seq_threshold_3D, or
    # 4. Batch invariance is enabled
    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or (
            max_seqlen_q > 1
            and (
                max_seqlen_q > MAX_3D_QUERY_LEN
                or not use_causal
                or use_per_seq_causal
                or use_mm_prefix
                or tuned_large_head
            )
        )
        or num_seqs * triton.cdiv(max_seqlen_q, BLOCK_Q) > seq_threshold_3D
        or is_batch_invariant
    )
    if use_3d and softmax_segm_max.shape[0] < q.shape[0]:
        logger.debug_once("Split-K scratch rows < query tokens; using 2D.")
        use_3d = False

    # The kernel signature is the same for 2D and 3D — only the launch
    # grid + a handful of constexpr toggles differ.  Per-token-head scale
    # caches and their strides are passed as ``None`` when the
    # ``USE_PER_TOKEN_HEAD_SCALES`` branch is dead so Triton can skip
    # materialising those arguments and the associated registers.
    if use_per_token_head_scales:
        ks_strides = k_scale_cache.stride()
        vs_strides = v_scale_cache.stride()
        ks_blk, ks_slot, ks_head = ks_strides[0], ks_strides[1], ks_strides[2]
        vs_blk, vs_slot, vs_head = vs_strides[0], vs_strides[1], vs_strides[2]
        k_scale_ptr = k_scale_cache
        v_scale_ptr = v_scale_cache
    else:
        ks_blk = ks_slot = ks_head = None
        vs_blk = vs_slot = vs_head = None
        k_scale_ptr = None
        v_scale_ptr = None
    # 3D needs real segm tensors; 2D never touches them.  Pass ``None`` in
    # 2D mode so Triton can skip materialising these pointer arguments.
    segm_output_ptr = softmax_segm_output if use_3d else None
    segm_max_ptr = softmax_segm_max if use_3d else None
    segm_expsum_ptr = softmax_segm_expsum if use_3d else None
    num_segments = num_par_softmax_segments if use_3d else 1

    grid: tuple[Any, ...]
    if not use_3d:
        grid = (total_num_q_blocks, num_kv_heads)
        tile_size = TILE_SIZE_PREFILL
    else:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        tile_size = TILE_SIZE_DECODE

    launch_kwargs: dict[str, int] = {}
    if launch_num_warps is not None:
        launch_kwargs["num_warps"] = launch_num_warps
    if launch_num_stages is not None:
        launch_kwargs["num_stages"] = launch_num_stages

    kernel_unified_attention[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        k_scale_cache_ptr=k_scale_ptr,
        v_scale_cache_ptr=v_scale_ptr,
        scale=softmax_scale,
        q_scale=q_descale,
        k_scale=k_descale,
        v_scale=v_descale,
        out_scale=1 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=use_qq_bias,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=(1 + window_size[0]),
        USE_CAUSAL=use_causal,
        USE_PER_SEQ_CAUSAL=use_per_seq_causal,
        per_seq_causal_ptr=per_seq_causal_ptr,
        USE_MM_PREFIX=use_mm_prefix,
        MAX_MM_RANGES=max_mm_ranges,
        mm_prefix_range_ptr=mm_prefix_range,
        rswa_prefix_lens_ptr=rswa_prefix_lens if use_rswa else seqused_k,
        R_SWA_WINDOW=rswa_window or 0,
        USE_R_SWA=use_rswa,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        stride_ks_blk=ks_blk,
        stride_ks_slot=ks_slot,
        stride_ks_head=ks_head,
        stride_vs_blk=vs_blk,
        stride_vs_slot=vs_slot,
        stride_vs_head=vs_head,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=output_scale is not None,
        IS_3D=use_3d,
        KV_QUANT_MODE=kv_quant_mode,
        Q_IS_FP8=(q.dtype == current_platform.fp8_dtype()),
        CHUNK_LOOKBACK=chunk_lookback,
        CHUNK_SIZE=chunk_size,
        USE_TD=use_td,
        USE_TD_QO=use_td_qo,
        MM_PREFIX_CLAMP_SW=mm_prefix_clamp_sliding_window,
        **launch_kwargs,
    )

    if use_3d:
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=tile_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_segments,
            USE_FP8=output_scale is not None,
        )

    # gfx906: run tuned-shape prefill through the split-KV (3D) kernel. The
    # path is restricted to plain causal attention so the swept configuration
    # (BLOCK_M 32 / TILE 32 / 4 warps / 1 stage / 2 segments) is the only one
    # exercised, and is disabled under CUDA-graph capture because the segment
    # workspace cannot be allocated inside a capture.
    use_prefill_3d = (
        ENABLE_GFX906_ATTN_PREFILL_3D
        and max_seqlen_k >= GFX906_PREFILL_3D_MIN_CTX
        and gfx906_sched
        and prefill_block_m is not None
        and sliding_window_val == 0
        and not use_mm_prefix
        and not use_qq_bias
        and not use_alibi_slopes
        and sinks is None
        and softcap == 0
        and output_scale is None
        and not is_batch_invariant
        and not torch.cuda.is_current_stream_capturing()
    )

    if use_prefill_3d:
        num_segments = GFX906_PREFILL_SEGMENTS
        segm_output, segm_max, segm_expsum = _prefill_segment_buffers(
            q.shape[0],
            num_query_heads,
            num_segments,
            triton.next_power_of_2(head_size),
            q.device,
        )
        _launch_split_kv(
            segm_output,
            segm_max,
            segm_expsum,
            num_segments,
            TILE_SIZE_PREFILL,
            GFX906_PREFILL_3D_WARPS,
            GFX906_PREFILL_3D_STAGES,
        )
        return

    # Launch the 2D kernel if
    # 1. No intermediate tiled softmax buffers for the 3D kernel have been allocated, or
    # 2. The batch includes at least one prefill request, or
    # 3. The number of sequences exceeds the configured threshold, or
    # 4. Batch invariance is enabled
    # A small-query-length verify pass is decode-shaped (see small_q_decode), so
    # it is deliberately not treated as "a prefill request" here.
    if (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or (max_seqlen_q > 1 and not small_q_decode)
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    ):
        kernel_unified_attention_2d[
            (
                total_num_q_blocks,
                num_kv_heads,
            )
        ](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            out_scale=1 / output_scale if output_scale is not None else 1.0,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            TILE_SIZE=TILE_SIZE_PREFILL,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_ALIBI_SQRT=use_alibi_sqrt,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            USE_MM_PREFIX=use_mm_prefix,
            MAX_MM_RANGES=max_mm_ranges,
            mm_prefix_range_ptr=mm_prefix_range,
            SLIDING_WINDOW=(1 + window_size[0]),
            CAUSAL=causal,
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            USE_FP8=output_scale is not None,
            **(
                # tuned prefill shapes run 4 warps; everything else (incl. the
                # decode fallback through the 2D kernel) keeps 2 warps
                (
                    {
                        "num_warps": GFX906_PREFILL_WARPS,
                        "num_stages": GFX906_PREFILL_STAGES,
                    }
                    if prefill_block_m
                    else {"num_warps": 2, "num_stages": 1}
                )
                if gfx906_sched
                else {}
            ),
        )
    else:
        # gfx906: scale split-KV parallelism with the decode batch shape and
        # context length instead of the fixed default, bounded by the segment
        # buffers allocated by the backend.
        num_segments = num_par_softmax_segments
        if gfx906_sched and softmax_segm_output is not None:
            num_segments = min(
                _decode_num_segments(
                    max_seqlen_k,
                    total_num_q_blocks,
                    num_kv_heads,
                    num_queries_per_kv,
                ),
                softmax_segm_output.shape[2],
            )
        _launch_split_kv(
            softmax_segm_output,
            softmax_segm_max,
            softmax_segm_expsum,
            num_segments,
            TILE_SIZE_DECODE,
            _decode_num_warps(head_size, max_seqlen_k) if gfx906_sched else None,
        )
