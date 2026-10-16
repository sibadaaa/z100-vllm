# perf1: the MTP verify step never reached the split-KV decode path

**Question.** `/home/lyh/vllm-test/results/DEV_HANDOFF.md` (§8, §9.1) attributes the
long-context decode collapse to `GFX906_MAX_DECODE_SEGMENTS = 128` capping the
split-KV segment count, and proposes raising that cap as the "long-context
biggest single lever". Is the cap the actual constraint?

**Answer: no.** The cap is irrelevant because the split-KV (3D) decode kernel is
*never launched* for the production decode step. It is worth ~1%, not 68%.

## Change

`vllm/v1/attention/ops/triton_unified_attention.py` classified a forward pass as
prefill whenever `max_seqlen_q > 1`, and prefill takes the *unsplit* 2D kernel.
With the built-in MTP head (`S11F_MTP_TOKENS=2`) every decode step verifies
`k+1 = 3` query tokens, so every decode step was classified as prefill:

* `BLOCK_M` came from `_prefill_block_m` (32 for head 256 / GQA6),
* the unsplit `kernel_unified_attention_2d` was launched with
  `grid = (num_query_blocks, num_kv_heads)`. On TP=4 this model has
  `num_kv_heads == 1` and `num_query_blocks == 1`, i.e. **one CTA per layer
  serially scans the whole KV cache** (measured grid `(1, 1, 1)`),
* the segment buffers allocated by the backend were unused.

A speculative-decode verify step is decode-shaped: it scans the whole context
for a handful of query tokens. The fix routes a small-query-length forward
(`max_seqlen_q <= GFX906_VERIFY_MAX_QUERY_LEN = 8`) through the same decode
path as `max_seqlen_q == 1`, on the tuned decode shape (head 256 / GQA6) only.
Kill switch: `VLLM_GFX906_ATTN_VERIFY_3D=0` restores the old classification.
True prefill batches (chunked prefill, any query length > 8) are untouched, and
off gfx906 the module behaves exactly as before.

## Correctness

* `tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py`: the routing
  predicate, plus a kernel-spy test asserting a 2/3/5-token verify step launches
  `kernel_unified_attention_3d` (never the 2D kernel) and matches a torch eager
  reference; and a kill-switch test asserting the 2D kernel comes back.
* Microbenchmark accuracy vs an fp32 eager reference (max abs error of the fp16
  output, identical inputs, block 16, k=3 query tokens):

  | context | 2D unsplit | 3D split-KV |
  |---|---|---|
  | 512 | 3.03e-05 | **3.12e-05** |
  | 4096 | 1.06e-05 | **1.04e-05** |

  The split-KV path is as accurate as (at 4k marginally more accurate than) the
  path it replaces.
* Greedy token parity (`tools/dcu/perf1_e2e_parity.py`, fix vs kill-switch-off,
  same build, `temperature=0`, 48 generated tokens, 2 repeats each):
  3633-token context: **token streams identical**, max top-1 logprob delta
  0.026 nats. 439-token context: identical for 14 tokens, then one position
  flips (`7` -> `5` in a synthetic numbered list), max logprob delta 0.80 nats —
  a near-tie that the fp16 attention reduction order tips over. Within-arm
  repeats are bit-identical, so the difference is deterministic, not flaky.
  Disclosed per the perf1 parity rule; the 3D path is also the one plain decode
  and the MTP draft head already use in production.

## Result (microbenchmark, /mnt/game/gfx906_attn_tune/bench_decode_attn.py)

Per full-attention layer, median of 10 iterations, Z100L, fp16, block 16,
`nqh=6 / nkvh=1 / head=256 / GQA6`, TILE_SIZE 16, 4 warps, 128 MiB L2 flush:

| context | shape | 2D unsplit | 3D split-KV (segs) |
|---|---|---|---|
| 8192 | k=3 verify | — | 0.286 ms (BM8, 128) / 0.218 ms (BM16, 128) |
| 32768 | k=1 decode | 13.75 ms | 0.435 ms (128) / 0.385 ms (256) |
| 32768 | k=3 verify | 18.87 ms | 0.894 ms (BM8, 128) / 0.670 ms (BM16, 256) |
| 65536 | k=3 verify | — | 1.66 ms (BM8, 128) |
| 131072 | k=3 verify | — | 2.83 ms (BM32, 256) |

Raising `GFX906_MAX_DECODE_SEGMENTS` 128 -> 256 is worth only 5-11% *within* the
already-correct path (and 512+ regresses), so the cap is left at 128: the extra
~100 MiB/rank of fp32 segment workspace is not justified. `BLOCK_M=16` is ~20%
faster than 8 on the verify shape but only ~1% end to end, so the tuned decode
`BLOCK_M` is left unchanged.

## Result (end to end, production recipe, MTP k=2, TP=4, enforce-eager)

`/home/lyh/vllm-test/scripts/bench_baseline.py`, discarded 32k warmup, 3 runs,
steady-state (last) run:

| prompt | metric | control (2D) | verify-3D | delta |
|---|---|---|---|---|
| 7443 | step p50 | 239.5 ms | **131.0 ms** | **-45%** |
| 7443 | decode | 11.13 t/s | **20.16 t/s** | **+81%** |
| 7443 | TTFT | 5978 ms | 5901 ms | -1.3% |
| 7443 | prefill | 1245 t/s | 1261 t/s | +1.3% |
| 30239 | step p50 | 624.0 ms | **135.7 ms** | **-78%** |
| 30239 | decode | 5.13 t/s | **19.66 t/s** | **3.8x** |
| 30239 | TTFT | 6774 ms | 6342 ms | -6.4% |
| 30239 | prefill | 4464 t/s | 4769 t/s | +6.8% |

Decode throughput is now nearly context-independent (21.2 t/s at 4k, 20.2 at
7.4k, 19.7 at 30k) instead of collapsing linearly with context. The TTFT/prefill
gain is the first decode step, which is itself a 3-token verify step.

## Trace evidence (torch profiler, 8 s decode window @30k, rank0)

Control (`VLLM_GFX906_ATTN_VERIFY_3D=0`), GPU busy 97.7%:

| kernel | % busy | calls | us/call | grid |
|---|---|---|---|---|
| `kernel_unified_attention_2d` | **60.7%** | 144 | 32414 | **(1, 1, 1)** x136 |
| `ncclDevKernel_AllReduce_Ring_LL_Sum_f16` | 12.5% | 1090 | 882 | (1,1,1) |
| `fp8_emu_gemm_block_kernel<32,64,2,4>` | 7.8% | 111 | 5438 | (80,35,1) |
| `fused_sigmoid_gating_delta_rule_gfx906_prefill` | 6.7% | 21 | 24490 | (1536,1,1) |
| `fused_sigmoid_gating_delta_rule_gfx906_mtp` | 3.6% | 384 | 715 | (6,1,1) |
| `fp8_emu_gemv_stream_block_kernel` | 2.5% | 2080 | 93 | (160,1,1) |

144 calls / 16 full-attention layers = 9 decode steps, 136 of them with the
single-CTA grid `(1, 1, 1)`. `gfX906_sched` policy had been asking for 128
segments the whole time; the launch simply never used them.

After the fix, same trace, 2 s decode-only window (~9 steps), rank0:
GPU busy 64.8% (the step under the profiler is 238 ms vs 633 ms for control):

| kernel | % busy | calls | us/call | grid |
|---|---|---|---|---|
| `ncclDevKernel_AllReduce_Ring_Simple_Sum_f16` | 27.6% | 963 | 371 | (1,1,1) |
| `fused_sigmoid_gating_delta_rule_gfx906_mtp` | 23.0% | 393 | 759 | (6,1,1) |
| `fp8_emu_gemv_stream_block_kernel` | 16.1% | 2139 | 97 | (160,1,1) |
| `ncclDevKernel_AllReduce_Ring_LL_Sum_f16` | 11.5% | 151 | 988 | (1,1,1) |
| **`kernel_unified_attention_3d`** | **9.7%** | 150 | **839** | **(3, 1, 128)** x141 |
| `per_token_group_quant_8bit_kernel` | 1.6% | 2175 | 10 | — |
| `elementwise_kernel_manual_unroll` | 1.6% | 3082 | 7 | — |

The decode attention kernel is now `kernel_unified_attention_3d` with
`grid = (3, 1, 128)`: three query blocks (one per verified token) x one KV head
x 128 split-KV segments, at 0.84 ms/call — matching the 0.89 ms/call measured in
isolation. No 2D kernel appears anywhere in the decode window; the 2D kernel now
only shows up during prefill.

Composition of the decode step, before -> after (share of GPU-busy time):

| component | control | fix |
|---|---|---|
| attention (decode) | 60.7% | 9.7% |
| NCCL all-reduce | 15.8% | 39.1% |
| GDN (MTP) | 3.6% | 23.0% |
| fp8 GEMV + GEMM | 10.4% | 16.1% |

## Observation

The DEV_HANDOFF §8 diagnosis was reached from a trace that *did* show
`kernel_unified_attention_2d` at 68.7% of the decode step, but read it as "the
2D kernel is the split-KV kernel with too few segments". The 2D kernel has no
segments at all — its absence from the segment-buffer path is the signal.
Two further handoff numbers that are off and matter for attribution:

* the checkpoint has **64** transformer layers (16 full-attention, 48 linear),
  not "36 = 27 + 9"; all per-step attribution scales by 16/9,
* `TILE_SIZE` for decode on this shape is **16**, not 32
  (`_get_tile_size(256, 0, 2, is_prefill=False)`).

## Next

With attention fixed, an 8 s decode window at 30k is ~40% NCCL, ~15% GDN and
~10% fp8 GEMM/GEMV. The remaining levers, in evidence order:

1. **overlap the all-reduce with compute** — the handoff's §5.4 finding that
   99.88% of kernels run on one stream still holds, and NCCL is now the largest
   single component of the decode step;
2. **GDN decode/MTP** (`fused_sigmoid_gating_delta_rule_gfx906_*`, ~10-15%);
3. `kernel_unified_attention_2d` now only shows up during prefill; the GDN
   chunked prefill question from the handoff §12.4 is still open.

Not worth doing (measured here): raising the segment cap, widening the verify
`BLOCK_M`.
