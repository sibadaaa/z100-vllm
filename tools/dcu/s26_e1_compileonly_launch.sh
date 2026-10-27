#!/usr/bin/env bash
# S26 E1 arm launcher (compile-only, no graphs): production launcher EXCEPT
#   1. PORT 8000 -> 8002
#   2. VLLM_CACHE_ROOT default /mnt/game/vllm-cache-graph
#   3. --enforce-eager dropped; --compilation-config '{"cudagraph_mode":"none"}'
# Purpose: separate the torch.compile contribution from the CUDA-graph one.
# Never installed as the default configuration.
# Original header follows.
#
# Bare-metal (no Docker) launcher for the 27B-FP8 + DFlash2 k=7 production arm.
#
# Runs directly on the host against the rock-6.4.2 kernel driver and the
# DTK 26.04 userspace that the DTK torch/vLLM wheels were built for.
# (Note: the host may also carry DTK 26.10, but that userspace has
# libhipblas.so.3 while the DTK 26.04 wheels need libhipblas.so.2, so the
# service uses the 26.04 userland tree copied to $DTK_ROOT.)
#
#   bash tools/dcu/serve_host_dflash2.sh              # foreground
#   systemd-run --user ... /bin/bash tools/dcu/serve_host_dflash2.sh   # as a service
#
# Env knobs: DTK_ROOT, REPO, PY, PORT, HOST, MAXLEN, MBT, UTIL, DEVICES, ITER_DETAILS.
set -u

REPO="${REPO:-/home/lyh/z100-vllm}"
DTK_ROOT="${DTK_ROOT:-/mnt/game/host-runtime/dtk2604}"
PY="${PY:-${REPO}/.venv/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8002}"
MAXLEN="${MAXLEN:-262144}"
# S24 L5 promotion (2026-10-20): MBT 8192 -> 4096.  The halved activation
# workspace lets the KV pool grow 15.54 -> 16.21 GiB (748,982 -> 783,993
# tokens) and static occupancy drops 94-95% -> 91-92% (headroom ~2.6-3.0 GiB),
# at 32k TTFT +0.7% and unchanged ITL (M=4096 GEMM efficiency ~= M=8192);
# 64k/128k long-prefill soak: 0 OOM/retry/preempt.  See
# docs/result/z100-longctx-vram-optimization-result.md §4/§4.1 and
# docs/design/z100-longctx-vram-optimization-plan.md §L5.  Set MBT=8192 to
# restore the pre-S24 chunk size without a restart of anything else.
MBT="${MBT:-4096}"
UTIL="${UTIL:-0.85}"
DEVICES="${DEVICES:-0,1,2,3}"
# 1 = 打开 vLLM 逐 iteration 明细日志（--enable-logging-iteration-details）。
# 需要逐 step 观测/用 vllm-docker-probe 解析 step 级时设 1；默认关，避免生产日志量变大。
ITER_DETAILS="${ITER_DETAILS:-0}"

export DTK_ROOT
# env.sh exports its own VLLM_CACHE_ROOT/TRITON_CACHE_DIR defaults, so keep any
# caller override here and re-apply it after sourcing.
_VLLM_CACHE_OVERRIDE="${VLLM_CACHE_ROOT:-}"
_TRITON_CACHE_OVERRIDE="${TRITON_CACHE_DIR:-}"
# DTK's env.sh uses bare ${CMAKE_PREFIX_PATH}/${LIBRARY_PATH}/... so it must not
# run under `set -u` (same pitfall recorded in dtk-rocm10/README.md).
set +u
# shellcheck disable=SC1091
. "${REPO}/tools/dcu/env.sh"
set -u

# Bare-metal boot ordering: the rock driver is loaded by hymgr.service, which
# may not have finished when a user service starts. Wait for all 4 HCUs.
for _i in $(seq 1 60); do
  _n=$(hy-smi 2>/dev/null | grep -c '^[0-9]')
  [ "${_n:-0}" -ge 4 ] && break
  echo "[host-serve] waiting for HCUs (${_n:-0}/4, ${_i}/60)"
  sleep 2
done
unset _i _n

export PYTHONPATH="${REPO}:/mnt/game/triton-gfx906/install"
export VLLM_CACHE_ROOT="${_VLLM_CACHE_OVERRIDE:-/mnt/game/vllm-cache-graph}"
export TRITON_CACHE_DIR="${_TRITON_CACHE_OVERRIDE:-/mnt/game/host-serve/triton-cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES="${DEVICES}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export PYTHONUNBUFFERED=1

# --- DFlash2 production arm ---
export VLLM_DCU_BF16=1
export VLLM_DCU_USE_FP8_EMU=1
# 2026-10-21: VLLM_DCU_FP8_W8A16=1 hard-enables the W8A16 route for the FP8
# emu kernel: the raw fp16 activation skips per-token-group quantization and
# only the weight stays fp8 in HBM.  Requires the fp8_emu_gemm_block_w8a16
# entry point in _rocm_C (verified present) and fp16 activations (the target
# model is --dtype float16; the DFlash draft is bf16 but unquantized, so it
# never touches this kernel).  Set 0 to restore the W8A8 route, auto to
# degrade per-layer.  See docs/design/z100-bf16-to-fp16-entry-coercion.md.
export VLLM_DCU_FP8_W8A16="${VLLM_DCU_FP8_W8A16:-1}"
export VLLM_DCU_FP8_GEMV_MAX_M="${VLLM_DCU_FP8_GEMV_MAX_M:-8}"
# 2026-10-21 P0: prefill fp16 materialization.  Above the per-role M thresholds
# (gfx906_fp16_prefill._ROLE_MIN_M) the layer weight is dequantized into one
# reused fp16 scratch and served by rocBLAS (16.5-17.7 TF/s measured) instead
# of the W8A16 tile GEMM (~6.5 TF/s); engages only on the W8A16 route in eager
# mode.  0 restores the all-tile path.  See docs/design/z100-p0-retest-plan.md.
export VLLM_DCU_FP16_PREFILL="${VLLM_DCU_FP16_PREFILL:-auto}"
# S23 promotion (2026-10-19): cfg 35 = the BK=64/BN=32 "wide2" decode tile.
# Measured in-window c=1 step p50 148.58 -> 137.68 ms (-7.34%), output t/s
# +4.13%, bit-exact greedy parity, no c=4/c=16 regression; see
# docs/result/z100-fp8-emu-optimization-result.md §11/§14.  Set 16 to restore
# the pre-S23 kernel (requires the old .so or the S23 artifact, which keeps it).
export VLLM_DCU_FP8_DECODE_CFG="${VLLM_DCU_FP8_DECODE_CFG:-35}"
# S23 O1/O2: eager call-path shortcuts (-13.9 and -3.3 ms/step of host time,
# c=1 neutral, c=4 +6.6% / c=16 +3.55% fwd throughput for O1).  Set 0 to
# restore the previous python path without a rebuild.
export VLLM_DCU_FP8_EMU_FASTPATH="${VLLM_DCU_FP8_EMU_FASTPATH:-1}"
export VLLM_DCU_FP8_QUANT_FASTPATH="${VLLM_DCU_FP8_QUANT_FASTPATH:-1}"
# S24 L3.2 promotion (2026-10-20): dynamic speculative verify truncation.
# A verify step schedules 1+k target rows; beyond 16k/64k computed tokens a
# weak drafter (per-request acceptance EMA < 5) verifies only 1+3 / 1+1 rows.
# The greedy output is unchanged (the verify result is the target's own argmax
# prefix).  ABAB window w2/L32/dynk5: 32k ITL p50 191.6/192.8 (control) ->
# 147.8/146.0 ms (-23%), output t/s +25-30%, TTFT flat, 512 ctx unchanged,
# greedy parity 3/3 in both dynk arms.  Unset all three to restore
# verify-everything (docs/result/z100-longctx-vram-optimization-result.md §11).
export VLLM_S24_DYNK_LO="${VLLM_S24_DYNK_LO:-16384}"
export VLLM_S24_DYNK_HI="${VLLM_S24_DYNK_HI:-65536}"
export VLLM_S24_DYNK_EMA="${VLLM_S24_DYNK_EMA:-0.1}"
# O4's VLLM_DCU_FP8_STREAM_MAX_M stays unset: it is faster but changes the
# accumulation order for 5120:1536 at M=8 and breaks bit-exact greedy parity.
export VLLM_QWEN35_MTP_GDN_TRITON=1
export VLLM_GFX906_GDN_CHUNKED_PREFILL=1
export VLLM_DCU_USE_BF16_GEMV=1
export VLLM_GFX906_ATTN_VERIFY_3D=1
export VLLM_GFX906_ATTN_FORCE_CAUSAL=0

# T4 (2026-10-19): long-context prefill split-KV (3D), adopted after controlled arms:
# cold 36K TTFT -5.53%, hot 36K -7.55%, 32K -2.89%, 14K/16K no regression.
# Requires the default-inert patch in vllm/v1/attention/ops/triton_unified_attention.py
# (MIN_CTX default 16384, SEGMENTS 4, verify-shape guard, info_once log).
# Rollback: drop these three lines (or set PREFILL_3D=0) and restart the service.
export VLLM_GFX906_TRITON_ATTN_PREFILL_3D="${VLLM_GFX906_TRITON_ATTN_PREFILL_3D:-1}"
export VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX="${VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX:-16384}"
export VLLM_GFX906_ATTN_PREFILL_SEGMENTS="${VLLM_GFX906_ATTN_PREFILL_SEGMENTS:-4}"

MODEL="${MODEL:-/mnt/game/models/Qwen3.8-27B-FP8}"
DRAFT="${DRAFT:-/mnt/game/models/Qwen3.8-27B-DFlash2-ms}"

# API key: explicit APIKEY wins, else the 0600 key file written for sharing.
# Empty means no auth (keep this only on a trusted LAN).
if [ -z "${APIKEY:-}" ] && [ -r /mnt/game/host-serve/api-key ]; then
  APIKEY="$(cat /mnt/game/host-serve/api-key)"
fi
APIKEY_ARG=()
[ -n "${APIKEY:-}" ] && APIKEY_ARG=(--api-key "${APIKEY}")
# 可选：逐 iteration 明细（step 级观测）；默认不传，保持现网日志量不变
ITER_ARGS=()
[ "${ITER_DETAILS}" = "1" ] && ITER_ARGS=(--enable-logging-iteration-details)
# S24 L5: --max-num-seqs window scan (128 = vLLM default, so empty keeps today's
# behaviour).  Only meaningful for the memory-margin arms; never set in production.
# 2026-10-21: default 4 (operator call).  The 2026-10-21 restart incident (see
# docs/design/z100-bf16-to-fp16-entry-coercion.md §7) showed the worker pinned
# budget regressing after SIGKILL: the RequestState UVA buffer is
# (max_num_seqs, max_model_len) int32 pinned, so 4 seqs -> 4 MiB boots
# reliably where 256 seqs -> 256 MiB failed.  Override with MAX_SEQS=n.
MAX_SEQS="${MAX_SEQS:-4}"
MNS_ARGS=(--max-num-seqs "${MAX_SEQS}")
# S24 L4: KV cache dtype experiment knob (empty = today's behaviour).  fp8_e4m3
# enables on-the-fly KV quantization with per-tensor scale 1.0; see
# docs/result/z100-longctx-vram-optimization-result.md §15.
KV_ARGS=()
[ -n "${KV_DTYPE:-}" ] && KV_ARGS=(--kv-cache-dtype "${KV_DTYPE}")
SPEC="{\"model\":\"${DRAFT}\",\"method\":\"dflash\",\"num_speculative_tokens\":7,\"dtype\":\"bfloat16\",\"kv_cache_dtype\":\"bfloat16\"}"

echo "[host-serve] date=$(date -Is) host=$(hostname) user=$(id -un)"
echo "[host-serve] dcu=$(cat "${DTK_ROOT}/.dtk_version" 2>/dev/null | head -1) python=${PY}"
echo "[host-serve] model=${MODEL} draft=${DRAFT} tp=4 maxlen=${MAXLEN} mbt=${MBT} util=${UTIL} port=${PORT} iter_details=${ITER_DETAILS}"

mkdir -p "${VLLM_CACHE_ROOT}" 2>/dev/null || true

exec "${PY}" -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve \
  "${MODEL}" \
  --dtype float16 --tensor-parallel-size 4 \
  --gpu-memory-utilization "${UTIL}" --disable-custom-all-reduce \
  --skip-mm-profiling --enable-prefix-caching --mamba-cache-mode align \
  --max-model-len "${MAXLEN}" --max-num-batched-tokens "${MBT}" \
  --served-model-name qwen3.8-27b --host "${HOST}" --port "${PORT}" \
  "${APIKEY_ARG[@]}" \
  "${ITER_ARGS[@]}" \
  "${MNS_ARGS[@]}" \
  "${KV_ARGS[@]}" \
  --per-request-spec-decode-metrics summary \
  --compilation-config '{"cudagraph_mode":"none"}' \
  --speculative-config "${SPEC}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --override-generation-config '{"temperature":0.7,"repetition_penalty":1.05}' \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}'
