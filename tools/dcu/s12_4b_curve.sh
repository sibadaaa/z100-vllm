#!/usr/bin/env bash
# S12: G5 fallback -- TP scaling curve on Qwen3.5-4B (8.8 GB, fits 4 cards),
# because 27B TP=4 cannot start on this 31 GB host (reproduced oom_kill).
#
#   tools/dcu/s12_4b_curve.sh
#
# Three arms (TP=1 on card 0, TP=2 on 0,1, TP=4 on 0,1,2,3), same cache root so
# each TP's AOT artifacts are written once.  Only decode + ttft are measured --
# the point is the TP scaling shape, not another full parity sweep.
set -u
WS=/home/lyh/z100-vllm
RUN="$WS/tools/dcu/s12_run_one.sh"
MODEL=/mnt/game/models/Qwen3.5-4B
CR=/mnt/game/s12/cache-4b

one() { # tp devices port
  echo "[4b] ==== TP=$1 devices=$2 $(date -Is)"
  S12_SERVED=qwen3.5-4b READY_TIMEOUT=2100 \
    bash "$RUN" "S12B4tp$1" "$MODEL" "$1" 1 "$CR" "$3" 4096 "$2" decode,ttft
  echo "[4b] ==== TP=$1 done rc=$? $(date -Is)"
}
one 1 "0"     8071
one 2 "0,1"   8072
one 4 "0,1,2,3" 8073
echo "[4b] curve complete $(date -Is)"
