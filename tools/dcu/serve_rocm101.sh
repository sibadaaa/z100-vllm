#!/usr/bin/env bash
# Start the validated ROCm 10.1 + torch 2.16 stack (P5 production recipe).
#
# This is the recipe every L2/L3/L4 result in
# docs/result/z100-rocm101-acceptance.md was produced with:
#   * copy tree /mnt/game/port/gfx906-vllm-rocm101 (PYTHONPATH prefix)
#   * validated extension md5s 9302ece8 / ffc176af / 7b4185c6 / a250420b
#   * fp16 (default) or bf16 (Q1; verified) on a Z100L card
#   * Triton eager-attention fallback OFF (F14 fixed by the triton rebuild)
#   * card 0/1 only: cards 2/3 are Z100 and mis-execute kernels on this
#     runtime (F24); see /mnt/game/z100-rocm101-validation/f23-evidence/
#     F24-vendor-report.md
#
# Usage:
#   bash tools/dcu/serve_rocm101.sh --model 0.6b [--device 0] [--port 8020]
#   bash tools/dcu/serve_rocm101.sh --model 4b --tp 1 --dtype bfloat16
#   bash tools/dcu/serve_rocm101.sh --model 4b --tp 2 --device 0,1   # TP2 Z100L pair
set -uo pipefail
CT=${CT:-z100-vllm-rocm101-rt}
REPO=${REPO:-/mnt/game/port/gfx906-vllm-rocm101}
MODEL=4b
DEVICE=0
PORT=8020
DTYPE=float16
TP=1
MAXLEN=
UTIL=
SKIP_WARMUP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --max-model-len) MAXLEN="$2"; shift 2 ;;
    --gpu-memory-utilization) UTIL="$2"; shift 2 ;;
    --skip-kernel-warmup) SKIP_WARMUP=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$MODEL" in
  0.6b)
    MODEL_DIR=/mnt/game/models/Qwen3-0.6B
    SERVED=qwen3-0.6b
    MAXLEN=${MAXLEN:-2048}
    UTIL=${UTIL:-0.30}
    ;;
  4b)
    MODEL_DIR=/mnt/game/models/Qwen3.5-4B
    SERVED=qwen3.5-4b
    MAXLEN=${MAXLEN:-4096}
    UTIL=${UTIL:-0.55}
    ;;
  *)
    echo "--model must be 0.6b or 4b" >&2; exit 2 ;;
esac

SERVE_EXTRA=()
# 4B is a multimodal-capable config; text-only serving skips the mm profiling.
[ "$MODEL" = "4b" ] && SERVE_EXTRA+=(--skip-mm-profiling)

ENVARGS=(-e "HIP_VISIBLE_DEVICES=$DEVICE")
[ "$SKIP_WARMUP" = "1" ] && ENVARGS+=(-e "VLLM_SKIP_KERNEL_WARMUP=1")

echo "[serve_rocm101] model=$MODEL($SERVED) dtype=$DTYPE tp=$TP devices=$DEVICE port=$PORT"
echo "[serve_rocm101] extensions: $(md5sum "$REPO"/vllm/*.so 2>/dev/null | awk '{printf "%s ", substr($1,1,8)}')"
echo "[serve_rocm101] env: ${ENVARGS[*]}"

CMD="exec python3 -m vllm.entrypoints.cli.main serve $MODEL_DIR"
CMD="$CMD --dtype $DTYPE --served-model-name $SERVED"
CMD="$CMD --tensor-parallel-size $TP --max-model-len $MAXLEN"
CMD="$CMD --gpu-memory-utilization $UTIL --enforce-eager"
CMD="$CMD ${SERVE_EXTRA[*]:-}"
CMD="$CMD --host 127.0.0.1 --port $PORT"

docker exec "${ENVARGS[@]}" "$CT" bash -c "
  . /src/tools/dcu/env-rocm101.sh >/dev/null 2>&1
  cd $REPO || exit 92
  $CMD
"
