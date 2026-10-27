#!/usr/bin/env bash
# S19: container-side launcher for ONE DFlash speculative-decoding arm on
# Hygon Z100 (gfx906).
#
#   docker exec -d z100-vllm bash /src/tools/dcu/s19_dflash_arm.sh \
#       <SLOT> <TP> <PORT> <DRAFT> <K> [EXTRA...]
#
#   SLOT  fp8 | int8       target checkpoint slot (same table as s18_mtp_arm.sh)
#   TP    tensor parallel size
#   PORT  http port inside the container
#   DRAFT draft checkpoint dir, e.g. /mnt/game/models/Qwen3.8-27B-DFlash2
#   K     num_speculative_tokens (DFlash2 block size 8 -> 7 draft tokens)
#
# Environment knobs (all optional):
#   S19_MODEL     target checkpoint dir       (default per SLOT)
#   S19_MAXLEN    --max-model-len             (default 32768)
#   S19_UTIL      --gpu-memory-utilization    (default 0.85)
#   S19_MBT       --max-num-batched-tokens    (default 2048)
#   S19_EAGER     1|0  --enforce-eager        (default 1: eager is the quality arm
#                      on this target -- S14b/S14d)
#   S19_DEVICES   HIP_VISIBLE_DEVICES         (default 0,1,2,3)
#   S19_P2P       1|0  HSA_FORCE_FINE_GRAIN_PCIE (default 1: RCCL P2P)
#   S19_DTYPE     --dtype                     (default float16, production)
#   S19_SERVED    --served-model-name         (default qwen3.8-27b)
#   S19_CACHE     VLLM_CACHE_ROOT             (default /mnt/game/vllm-cache-s19/<slot>)
#   S19_LOG       serve log                   (default /mnt/game/dflash/serve-<label>.log)
#   S19_OUT       artifact dir                (default /mnt/game/dflash)
#   S19_BF16_GEMV 1|0  VLLM_DCU_USE_BF16_GEMV (default 1: the DFlash draft is
#                      bf16, so the gfx906 bf16 GEMV covers its linears)
#   S19_FP8_EMU   1|0  VLLM_DCU_USE_FP8_EMU   (fp8 slot only, default 1)
#   S19_INT8_SDOT4 1|0 VLLM_DCU_USE_INT8_SDOT4(int8 slot only, default 1)
#   S19_EXTRA     extra CLI flags appended before "$@"
set -u
SLOT=${1:?usage: s19_dflash_arm.sh <fp8|int8> <TP> <PORT> <DRAFT> <K> [EXTRA...]}
TP=${2:?usage: s19_dflash_arm.sh <fp8|int8> <TP> <PORT> <DRAFT> <K> [EXTRA...]}
PORT=${3:?usage: s19_dflash_arm.sh <fp8|int8> <TP> <PORT> <DRAFT> <K> [EXTRA...]}
DRAFT=${4:?usage: s19_dflash_arm.sh <fp8|int8> <TP> <PORT> <DRAFT> <K> [EXTRA...]}
K=${5:?usage: s19_dflash_arm.sh <fp8|int8> <TP> <PORT> <DRAFT> <K> [EXTRA...]}
shift 5

LABEL="${SLOT}-tp${TP}-dflash-k${K}"
OUT="${S19_OUT:-/mnt/game/dflash}"
mkdir -p "$OUT"
echo $$ > "/tmp/s19_${LABEL}.pid"

case "$SLOT" in
  fp8)  MODEL=${S19_MODEL:-/mnt/game/models/Qwen3.8-27B-FP8} ;;
  int8) MODEL=${S19_MODEL:-/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8} ;;
  *) echo "[s19] bad SLOT '$SLOT' (fp8|int8)" >&2; exit 2 ;;
esac

[ -f "$DRAFT/config.json" ] || { echo "[s19] draft not found: $DRAFT" >&2; exit 2; }

. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH="/src:/mnt/game/triton-gfx906/install"
export VLLM_CACHE_ROOT="${S19_CACHE:-/mnt/game/vllm-cache-s19/${SLOT}}"
export TRITON_CACHE_DIR="${S19_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES="${S19_DEVICES:-0,1,2,3}"
export HSA_FORCE_FINE_GRAIN_PCIE="${S19_P2P:-${HSA_FORCE_FINE_GRAIN_PCIE:-1}}"
export VLLM_LOGGING_LEVEL="${S19_LOGLEVEL:-INFO}"
export PYTHONUNBUFFERED=1
# The DFlash draft runs its own bf16 attention/MLP; keep the gfx906 bf16 GEMV
# path on unless an A/B arm says otherwise.
export VLLM_DCU_USE_BF16_GEMV="${VLLM_DCU_USE_BF16_GEMV:-${S19_BF16_GEMV:-1}}"

# Keep the non-spec-decode knobs identical to the production arm these numbers
# are compared against (captured from the live :8000 process, S19 baseline).
export VLLM_GFX906_ATTN_VERIFY_3D="${VLLM_GFX906_ATTN_VERIFY_3D:-${S19_ATTN_VERIFY_3D:-1}}"
# A/B arm: force the drafter's bidirectional block attention onto the causal
# path to quantify how much acceptance comes from real non-causal masking.
export VLLM_GFX906_ATTN_FORCE_CAUSAL="${VLLM_GFX906_ATTN_FORCE_CAUSAL:-${S19_ATTN_FORCE_CAUSAL:-0}}"

case "$SLOT" in
  fp8)
    export VLLM_DCU_USE_FP8_EMU="${S19_FP8_EMU:-1}"
    # M = 1+k verify pass stays on the streaming GEMV path (perf1/S11f).
    export VLLM_DCU_FP8_GEMV_MAX_M="${VLLM_DCU_FP8_GEMV_MAX_M:-4}"
    ;;
  int8)
    export VLLM_DCU_USE_INT8_SDOT4="${S19_INT8_SDOT4:-1}"
    export VLLM_DCU_SDOT4_INNER="${S19_SDOT4_INNER:-sdot4}"
    ;;
esac

EAGER=""
[ "${S19_EAGER:-1}" = "1" ] && EAGER="--enforce-eager"

SPEC="--speculative-config {\"model\":\"${DRAFT}\",\"method\":\"dflash\",\"num_speculative_tokens\":${K}"
# The drafter may need its own dtype: a bf16 DFlash checkpoint overflows in
# fp16 (its MLP activations reach ~1.4e5). S19_DRAFT_DTYPE/ S19_DRAFT_KV_DTYPE
# are passed through to SpeculativeConfig.
if [ -n "${S19_DRAFT_DTYPE:-}" ]; then
  SPEC="${SPEC},\"dtype\":\"${S19_DRAFT_DTYPE}\""
fi
if [ -n "${S19_DRAFT_KV_DTYPE:-}" ]; then
  SPEC="${SPEC},\"kv_cache_dtype\":\"${S19_DRAFT_KV_DTYPE}\""
fi
SPEC="${SPEC}}"

{
  echo "[s19] label=$LABEL slot=$SLOT tp=$TP port=$PORT k=$K"
  echo "[s19] model=$MODEL draft=$DRAFT"
  echo "[s19] dtype=${S19_DTYPE:-float16} eager=${S19_EAGER:-1} maxlen=${S19_MAXLEN:-32768} mbt=${S19_MBT:-2048} util=${S19_UTIL:-0.85}"
  echo "[s19] devices=$HIP_VISIBLE_DEVICES p2p=$HSA_FORCE_FINE_GRAIN_PCIE bf16_gemv=$VLLM_DCU_USE_BF16_GEMV"
  echo "[s19] cache=$VLLM_CACHE_ROOT log=${S19_LOG:-$OUT/serve-${LABEL}.log}"
  echo "[s19] date=$(date -Is) host=$(hostname)"
} | tee "$OUT/args-${LABEL}.txt"

cd /src || exit 1
exec python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve \
  "$MODEL" \
  --dtype "${S19_DTYPE:-float16}" --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "${S19_UTIL:-0.85}" --disable-custom-all-reduce \
  --skip-mm-profiling --enable-prefix-caching --mamba-cache-mode align \
  --max-model-len "${S19_MAXLEN:-32768}" --max-num-batched-tokens "${S19_MBT:-2048}" \
  --served-model-name "${S19_SERVED:-qwen3.8-27b}" --host 127.0.0.1 --port "$PORT" \
  --per-request-spec-decode-metrics summary \
  $EAGER $SPEC ${S19_EXTRA:-} "$@" 2>&1 | tee "${S19_LOG:-$OUT/serve-${LABEL}.log}"
