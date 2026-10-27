#!/usr/bin/env bash
# S11l: S9-protocol perf parity against a local 27B arm.
#
#   docker exec z100-s9 bash /src/tools/dcu/s11l_bench.sh <port> <label> <mode>
#   mode: decode | ttft | ttft8k | conc8
#
# Flags are copied verbatim from the S9 archive (/mnt/game/port/s9/bench.sh) so
# the numbers are directly comparable with `bench-27b.log`:
#   decode  in=128  out=128 np=2  warmups=1 conc=1
#   ttft    in=3800 out=1   np=1  warmups=0 conc=1   (S9 used 3800 because the
#           engine ran --max-model-len 4096; ttft8k is the exact baseline shape)
#   conc8   in=512  out=128 np=16 warmups=2 conc=8
set -uo pipefail
PORT="$1"; LABEL="$2"; MODE="$3"
OUT="${S11L_BENCH_OUT:-/mnt/game/s11l}"
mkdir -p "$OUT"
MODEL=/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8

. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
export VLLM_CACHE_ROOT=/mnt/game/vllm-cache-27b

case "$MODE" in
  decode) EXTRA="--random-input-len 128 --random-output-len 128 --num-prompts 2 --num-warmups 1 --max-concurrency 1 --temperature 0 --ignore-eos" ;;
  ttft)   EXTRA="--random-input-len 3800 --random-output-len 1 --num-prompts 1 --num-warmups 0 --max-concurrency 1 --temperature 0 --ignore-eos" ;;
  # S9 baseline used in=8192; this engine runs --max-model-len 8192, so use 8100
  # to leave room for the chat template (~1.1% shorter; compared per prefill token).
  ttft8k) EXTRA="--random-input-len 8100 --random-output-len 1 --num-prompts 1 --num-warmups 0 --max-concurrency 1 --temperature 0 --ignore-eos" ;;
  conc8)  EXTRA="--random-input-len 512 --random-output-len 128 --num-prompts 16 --num-warmups 2 --max-concurrency 8 --temperature 0 --ignore-eos" ;;
  *) echo "bad mode $MODE"; exit 2 ;;
esac

LOG="$OUT/bench-${LABEL}-${MODE}.log"
cd /src || exit 1
echo "[s11l-bench] port=$PORT label=$LABEL mode=$MODE log=$LOG"
python -c "import torch;from vllm.entrypoints.cli.main import main;main()" bench serve \
  --backend openai-chat --base-url "http://127.0.0.1:${PORT}" \
  --endpoint /v1/chat/completions --model "$MODEL" --served-model-name qwen3.8-27b \
  --dataset-name random --seed 0 --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 $EXTRA 2>&1 | tee "$LOG"
echo "[s11l-bench] rc=${PIPESTATUS[0]} label=$LABEL mode=$MODE"
