#!/usr/bin/env bash
# S14: perf client, byte-identical request shapes to tools/dcu/s13_bench.sh but
# with n>=4 repeats on the decode/TTFT arms (S14 G3 requires n>=4 + interval),
# and output root /mnt/game/s14.
#
#   docker exec z100-s14 bash /src/tools/dcu/s14_bench.sh <port> <label> <mode> <served> <model_dir>
#   mode: decode | ttft | ttft8k | conc8
#
#   decode  in=128  out=128 np=5  warmups=1 conc=1  -> n=5 measured requests
#   ttft    in=3800 out=1   np=4  warmups=0 conc=1  -> n=4 measured requests
#   ttft8k  in=8100 out=1   np=1  warmups=0 conc=1  -> S13口径 (ms/prefill-token)
#   conc8   in=512  out=128 np=16 warmups=2 conc=8  -> S13口径
set -uo pipefail
PORT="$1"; LABEL="$2"; MODE="$3"; SERVED="${4:-qwen3.8-27b}"; MODEL="${5:-/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8}"
OUT="${S14_BENCH_OUT:-/mnt/game/s14}"
mkdir -p "$OUT"

. /opt/dtk/env.sh >/dev/null 2>&1
. "${S14_SRC:-/src}/tools/dcu/env.sh" >/dev/null 2>&1
export PYTHONPATH=${S14_SRC:-/src}:/mnt/game/triton-gfx906/install
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache

case "$MODE" in
  decode) EXTRA="--random-input-len 128 --random-output-len 128 --num-prompts 5 --num-warmups 1 --max-concurrency 1 --temperature 0 --ignore-eos" ;;
  ttft)   EXTRA="--random-input-len 3800 --random-output-len 1 --num-prompts 4 --num-warmups 0 --max-concurrency 1 --temperature 0 --ignore-eos" ;;
  ttft8k) EXTRA="--random-input-len 8100 --random-output-len 1 --num-prompts 1 --num-warmups 0 --max-concurrency 1 --temperature 0 --ignore-eos" ;;
  conc8)  EXTRA="--random-input-len 512 --random-output-len 128 --num-prompts 16 --num-warmups 2 --max-concurrency 8 --temperature 0 --ignore-eos" ;;
  *) echo "bad mode $MODE"; exit 2 ;;
esac

LOG="$OUT/bench-${LABEL}-${MODE}.log"
cd "${S14_SRC:-/src}" || exit 1
echo "[s14-bench] port=$PORT label=$LABEL mode=$MODE served=$SERVED log=$LOG"
python -c "import torch;from vllm.entrypoints.cli.main import main;main()" bench serve \
  --backend openai-chat --base-url "http://127.0.0.1:${PORT}" \
  --endpoint /v1/chat/completions --model "$MODEL" --served-model-name "$SERVED" \
  --dataset-name random --seed 0 --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 --save-result --result-dir "$OUT" $EXTRA 2>&1 | tee "$LOG"
echo "[s14-bench] rc=${PIPESTATUS[0]} label=$LABEL mode=$MODE"
