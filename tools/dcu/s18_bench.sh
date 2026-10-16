#!/usr/bin/env bash
# S18: perf bench against a live arm, in the S9/S11l/S14 protocol so numbers are
# directly comparable with the archives.
#
#   docker exec z100-s18 bash /src/tools/dcu/s18_bench.sh <port> <label> <mode>
#   mode: decode | ttft | ttft8k | conc8
#
#   decode  in=128  out=128 np=2  warmups=1 conc=1
#   ttft    in=3800 out=1   np=1  warmups=0 conc=1   (needs maxlen >= ~4000)
#   ttft8k  in=8100 out=1   np=1  warmups=0 conc=1   (needs maxlen >= 8192)
#   conc8   in=512  out=128 np=16 warmups=2 conc=8
#
# Env: S18_BENCH_OUT (default /mnt/game/s18), S18_BENCH_MODEL, S18_BENCH_SERVED.
set -uo pipefail
PORT="$1"; LABEL="$2"; MODE="$3"
OUT="${S18_BENCH_OUT:-/mnt/game/s18}"
mkdir -p "$OUT"
MODEL="${S18_BENCH_MODEL:-/mnt/game/models/Qwen3.8-27B-FP8}"
SERVED="${S18_BENCH_SERVED:-qwen3.8-27b}"

. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
export VLLM_CACHE_ROOT="${S18_CACHE:-/mnt/game/vllm-cache-s18}"

case "$MODE" in
  decode) EXTRA="--random-input-len 128 --random-output-len 128 --num-prompts 2 --num-warmups 1 --max-concurrency 1 --temperature 0 --ignore-eos" ;;
  ttft)   EXTRA="--random-input-len 3800 --random-output-len 1 --num-prompts 1 --num-warmups 0 --max-concurrency 1 --temperature 0 --ignore-eos" ;;
  ttft8k) EXTRA="--random-input-len 8100 --random-output-len 1 --num-prompts 1 --num-warmups 0 --max-concurrency 1 --temperature 0 --ignore-eos" ;;
  conc8)  EXTRA="--random-input-len 512 --random-output-len 128 --num-prompts 16 --num-warmups 2 --max-concurrency 8 --temperature 0 --ignore-eos" ;;
  *) echo "bad mode $MODE"; exit 2 ;;
esac

LOG="$OUT/bench-${LABEL}-${MODE}.log"
cd /src || exit 1
echo "[s18-bench] port=$PORT label=$LABEL mode=$MODE model=$MODEL log=$LOG"
python -c "import torch;from vllm.entrypoints.cli.main import main;main()" bench serve \
  --backend openai-chat --base-url "http://127.0.0.1:${PORT}" \
  --endpoint /v1/chat/completions --model "$MODEL" --served-model-name "$SERVED" \
  --dataset-name random --seed 0 --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 $EXTRA 2>&1 | tee "$LOG"
echo "[s18-bench] rc=${PIPESTATUS[0]} label=$LABEL mode=$MODE"
