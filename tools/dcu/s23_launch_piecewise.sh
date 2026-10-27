#!/usr/bin/env bash
# O7 arm launcher: piecewise CUDA-graph feasibility re-check (S23 O7).
#
# Byte-for-byte the production launcher tools/dcu/serve_host_dflash2.sh EXCEPT
# for the two graph-mode flags (see the O7 markers below): `--enforce-eager` is
# dropped and `--compilation-config {"cudagraph_mode":"piecewise"}` is added.
# Never installed as the default configuration; the O6 arm driver installs it as
# a systemd user-unit drop-in for one maintenance window and removes it after.
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
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-262144}"
MBT="${MBT:-8192}"
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
export VLLM_CACHE_ROOT="${_VLLM_CACHE_OVERRIDE:-/mnt/game/vllm-cache-host}"
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
export VLLM_DCU_FP8_GEMV_MAX_M="${VLLM_DCU_FP8_GEMV_MAX_M:-8}"
export VLLM_DCU_FP8_DECODE_CFG="${VLLM_DCU_FP8_DECODE_CFG:-16}"
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
SPEC="{\"model\":\"${DRAFT}\",\"method\":\"dflash\",\"num_speculative_tokens\":7,\"dtype\":\"bfloat16\",\"kv_cache_dtype\":\"bfloat16\"}"

echo "[o7-arm] piecewise cudagraph arm (no --enforce-eager)"
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
  --per-request-spec-decode-metrics summary \
  --compilation-config '{"cudagraph_mode":"piecewise"}' \
  --speculative-config "${SPEC}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --override-generation-config '{"temperature":0.7,"repetition_penalty":1.05}' \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}'
