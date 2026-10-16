#!/usr/bin/env bash
# S18: container-side launcher for ONE MTP arm on Hygon Z100 (gfx906).
#
#   docker exec -d z100-s18 bash /src/tools/dcu/s18_mtp_arm.sh <SLOT> <TP> <PORT> <K> [EXTRA...]
#
#   SLOT  fp8 | int8      which checkpoint (fp8 = production slot, int8 = diagnostic arm)
#   TP    tensor parallel size
#   PORT  http port inside the container (host-published 8001 for z100-s18)
#   K     num_speculative_tokens; 0 = MTP OFF (control arm)
#
# Environment knobs (all optional):
#   S18_MODEL     checkpoint dir              (default per SLOT)
#   S18_SPEC_MODEL draft checkpoint dir       (default = S18_MODEL)
#   S18_MAXLEN    --max-model-len             (default 8192)
#   S18_UTIL      --gpu-memory-utilization    (default 0.85)
#   S18_MBT       --max-num-batched-tokens    (default 2048)
#   S18_EAGER     1|0  --enforce-eager        (default 1: graph changes the token
#                      stream on this target -- S14b/S14d -- so eager is the
#                      quality arm; graph is a latency-ceiling arm only)
#   S18_DEVICES   HIP_VISIBLE_DEVICES         (default 0,1,2,3)
#   S18_P2P       1|0  HSA_FORCE_FINE_GRAIN_PCIE (default 1: RCCL P2P, S16/S17)
#   S18_DTYPE     --dtype                     (default float16, production)
#   S18_SERVED    --served-model-name         (default qwen3.8-27b)
#   S18_DEBUG     1|0  MTP gfx906 debug prints (default 1 for K>0)
#   S18_CACHE     VLLM_CACHE_ROOT             (default /mnt/game/vllm-cache-s18/<slot>)
#   S18_LOG       serve log                   (default /mnt/game/s18/serve-<slot>-k<K>.log)
#   S18_OUT       artifact dir                (default /mnt/game/s18)
#   S18_EXTRA     extra CLI flags appended before "$@" (e.g. for A/B on
#                 --mamba-cache-mode / --no-enable-prefix-caching)
set -u
SLOT=${1:?usage: s18_mtp_arm.sh <fp8|int8> <TP> <PORT> <K> [EXTRA...]}
TP=${2:?usage: s18_mtp_arm.sh <fp8|int8> <TP> <PORT> <K> [EXTRA...]}
PORT=${3:?usage: s18_mtp_arm.sh <fp8|int8> <TP> <PORT> <K> [EXTRA...]}
K=${4:?usage: s18_mtp_arm.sh <fp8|int8> <TP> <PORT> <K> [EXTRA...]}
shift 4

LABEL="${SLOT}-tp${TP}-k${K}"
OUT="${S18_OUT:-/mnt/game/s18}"
mkdir -p "$OUT"
echo $$ > "/tmp/s18_${LABEL}.pid"

case "$SLOT" in
  fp8)  MODEL=${S18_MODEL:-/mnt/game/models/Qwen3.8-27B-FP8} ;;
  int8) MODEL=${S18_MODEL:-/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8} ;;
  *) echo "[s18] bad SLOT '$SLOT' (fp8|int8)" >&2; exit 2 ;;
esac
SPEC_MODEL=${S18_SPEC_MODEL:-$MODEL}

. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH="/src:/mnt/game/triton-gfx906/install"
export VLLM_CACHE_ROOT="${S18_CACHE:-/mnt/game/vllm-cache-s18/${SLOT}}"
export TRITON_CACHE_DIR="${S18_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES="${S18_DEVICES:-0,1,2,3}"
export HSA_FORCE_FINE_GRAIN_PCIE="${S18_P2P:-${HSA_FORCE_FINE_GRAIN_PCIE:-1}}"
export VLLM_LOGGING_LEVEL="${S18_LOGLEVEL:-INFO}"
export PYTHONUNBUFFERED=1

# quantization path per slot (never rely on defaults -- S11k/S11f)
case "$SLOT" in
  fp8)
    export VLLM_DCU_USE_FP8_EMU="${S18_FP8_EMU:-1}"
    export VLLM_DCU_USE_BF16_GEMV="${S18_BF16_GEMV:-0}"
    ;;
  int8)
    export VLLM_DCU_USE_INT8_SDOT4="${S18_INT8_SDOT4:-1}"
    export VLLM_DCU_SDOT4_INNER="${S18_SDOT4_INNER:-sdot4}"
    ;;
esac

# MTP on gfx906: prove the custom ops are hit (CUSTOM_HIT), not the fallback.
DEBUG="${S18_DEBUG:-}"
if [ -z "$DEBUG" ]; then
  [ "$K" = "0" ] && DEBUG=0 || DEBUG=1
fi
if [ "$DEBUG" = "1" ]; then
  export VLLM_QWEN35_MTP_CONV_DEBUG=1
  export VLLM_QWEN35_MTP_GDN_DEBUG=1
  export VLLM_GFX906_GDN_MTP_V2_DEBUG=1
fi

EAGER=""
[ "${S18_EAGER:-1}" = "1" ] && EAGER="--enforce-eager"
SPEC=""
[ "$K" != "0" ] && SPEC="--speculative-config {\"model\":\"${SPEC_MODEL}\",\"method\":\"mtp\",\"num_speculative_tokens\":${K}}"

{
  echo "[s18] label=$LABEL slot=$SLOT tp=$TP port=$PORT k=$K"
  echo "[s18] model=$MODEL spec_model=$SPEC_MODEL"
  echo "[s18] dtype=${S18_DTYPE:-float16} eager=${S18_EAGER:-1} maxlen=${S18_MAXLEN:-8192} mbt=${S18_MBT:-2048} util=${S18_UTIL:-0.85}"
  echo "[s18] devices=$HIP_VISIBLE_DEVICES p2p=$HSA_FORCE_FINE_GRAIN_PCIE debug=$DEBUG"
  echo "[s18] cache=$VLLM_CACHE_ROOT log=${S18_LOG:-$OUT/serve-${LABEL}.log}"
  echo "[s18] date=$(date -Is) host=$(hostname)"
} | tee "$OUT/args-${LABEL}.txt"

cd /src || exit 1
exec python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve \
  "$MODEL" \
  --dtype "${S18_DTYPE:-float16}" --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "${S18_UTIL:-0.85}" --disable-custom-all-reduce \
  --skip-mm-profiling --enable-prefix-caching --mamba-cache-mode align \
  --max-model-len "${S18_MAXLEN:-8192}" --max-num-batched-tokens "${S18_MBT:-2048}" \
  --served-model-name "${S18_SERVED:-qwen3.8-27b}" --host 127.0.0.1 --port "$PORT" \
  $EAGER $SPEC ${S18_EXTRA:-} "$@" 2>&1 | tee "${S18_LOG:-$OUT/serve-${LABEL}.log}"
