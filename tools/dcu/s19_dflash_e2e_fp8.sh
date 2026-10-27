#!/usr/bin/env bash
# S19 DFlash2 end-to-end arm: FP8 target + z-lab/Qwen3.8-27B-DFlash2 draft (TP4, :8000).
#
#   docker exec -d z100-vllm bash /src/tools/dcu/s19_dflash_e2e_fp8.sh
#   (add S19_MAXLEN=32768 in front for a short-context arm)
#
# The JSON CLI arguments must stay quoted inside this file: passing them through
# `docker exec -d bash -lc '... {"a":1,"b":2} ...'` makes the container shell
# brace-expand the JSON into two words (S19 first attempt, 18:51 -> argparse saw
# `temperature:0.7`).
set -u
export S19_MAXLEN="${S19_MAXLEN:-262144}"
export S19_LOGLEVEL="${S19_LOGLEVEL:-INFO}"
# The DFlash2 drafter is bf16: in fp16 its MLP activations overflow (inf -> NaN
# in the grouped conv), which makes every draft a garbage token. Serve the FP8
# target in fp16 and the drafter in bf16 (+ bf16 draft KV cache).
export S19_DRAFT_DTYPE="${S19_DRAFT_DTYPE:-bfloat16}"
export S19_DRAFT_KV_DTYPE="${S19_DRAFT_KV_DTYPE:-bfloat16}"
# gfx906 offers bf16 only with the DTK emulation path enabled.
export VLLM_DCU_BF16="${VLLM_DCU_BF16:-1}"
# --- DFLASH2_OPT_PLAN winners (A/B verified on this box, 2026-10-13) ---------
# 1) GDN verify/decode on the layout-aware gfx906 Triton kernel. Replaces the
#    C++ v1 kernel that costs 1.97 ms/call (75 ms of a 257 ms k=7 decode step).
#    Measured: step p50 288.6 -> 190.9 ms (c=1, -34%), 299.2 -> ~185 ms (c=4,
#    -38%), decode 11.08 -> 16.63 t/s (c=1, +50%). Acceptance unchanged.
# 2) FLA chunked GDN prefill instead of the fused sigmoid-gating prefill kernel.
#    Measured per-op 14-24x faster at production shapes (T=2048: 44.7 -> 1.9
#    ms/call/layer), max|dout| 3.1e-5 (tol 1e-3), no NaN. End to end: p512 c=1
#    TTFT 1788 -> 1249 ms, p1792 c=4 TTFT 15233 -> 11300 ms, prefill 277 -> 423
#    t/s, and acceptance *rises* to 3.66 from 3.25 tokens/step.
# Both default ON here; `VLLM_QWEN35_MTP_GDN_TRITON=0` /
# `VLLM_GFX906_GDN_CHUNKED_PREFILL=0` restore the previous path without a
# rebuild (the prefill switch is pure Python; the decode one is a C++ kernel
# selection).
export VLLM_QWEN35_MTP_GDN_TRITON="${VLLM_QWEN35_MTP_GDN_TRITON:-1}"
export VLLM_GFX906_GDN_CHUNKED_PREFILL="${VLLM_GFX906_GDN_CHUNKED_PREFILL:-1}"
for knob in VLLM_QWEN35_MTP_GDN_TRITON VLLM_GFX906_GDN_CHUNKED_PREFILL; do
  value="${!knob,,}"
  case "$value" in
    1|true|yes|on) printf -v "$knob" '%s' 1 ;;
    0|false|no|off) printf -v "$knob" '%s' 0 ;;
    *)
      printf 'Invalid %s=%q; expected 0/1, true/false, yes/no, or on/off.\n' \
        "$knob" "${!knob}" >&2
      exit 2
      ;;
  esac
  export "$knob"
done
# 3) FP8-emulated decode GEMM at the spec-verify batch size (M = k+1 = 8).
#    `VLLM_DCU_FP8_GEMV_MAX_M` defaults to 8 in vllm/envs.py, but this launcher
#    used to pin it to 4, which sent every M>4 call straight to the tiled GEMM
#    and silently bypassed the streaming kernel.  Raising it to 8 lets the
#    shape-aware dispatch choose, and `VLLM_DCU_FP8_DECODE_CFG=16` selects the
#    exact-M (BM=8, TM=1, TN=2) tile.  Measured on 2026-10-14 (A/B/A, c=1,
#    512-token prompts, identical workload in all arms, accept length 7.00 in
#    every arm):
#      kernel level, per call: 8704:5120 0.418 -> 0.242 ms (1.73x),
#        5120:4352 0.287 -> 0.167, 4096:5120 0.336 -> 0.198, 3584:5120 0.269 ->
#        0.180; mix-weighted 74.6 -> 41.6 ms/step.
#      end to end c=1: step p50 213.6 -> 199.1 ms (-6.8%), same tokens/step and
#        accept length.  The kernel saves 33 ms/step but only ~14.5 ms reaches
#        the step: the rest is absorbed by all-reduce barrier wait and launch
#        bubbles, which is why this is not a 2-3x single-request win.
#    `VLLM_DCU_FP8_GEMV_MAX_M=4` restores the previous dispatch without a rebuild.
export VLLM_DCU_FP8_GEMV_MAX_M="${VLLM_DCU_FP8_GEMV_MAX_M:-8}"
export VLLM_DCU_FP8_DECODE_CFG="${VLLM_DCU_FP8_DECODE_CFG:-16}"
# 4) Batch-token budget.  vLLM warns that 2048 is below the optimum; measured on
#    2026-10-14 with an A/B/A at c=1 and c=4 (same .so, same env otherwise):
#      c=4 step p50 770.5 ms (2048, mean of two control arms, spread 4.8%)
#                -> 577.9 ms (8192) = -25.0%; per-request output 0.64 -> 0.69 t/s
#      c=1 step p50 217.3 -> 206.5 ms = -5.0%
#    TTFT at c=4 rises (13.7-15.8 s -> 15.6 s) because a bigger budget batches
#    more prefill per step; that is the trade this buys, and it is the right one
#    for the target regime (concurrency <= 4, maximum speed).
#    `S19_MBT=2048` restores the previous behaviour.
export S19_MBT="${S19_MBT:-8192}"
exec bash /src/tools/dcu/s19_dflash_arm.sh \
  fp8 4 8000 /mnt/game/models/Qwen3.8-27B-DFlash2-ms 7 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --override-generation-config '{"temperature":0.7,"repetition_penalty":1.05}' \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
  --host 0.0.0.0
