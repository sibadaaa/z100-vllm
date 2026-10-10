#!/usr/bin/env bash
# S14d: the concurrency-vs-history control arm (E3 follow-up).
#
# One TP=4 EAGER engine (only the 4 GiB /swap.img attached, vm.swappiness=60).
# The very first two requests the engine ever sees are the SAME prompt, strictly
# sequentially (single request in flight, nothing else queued) -> this separates
# "a concurrent decoding stream" from "request history / not being the first
# request" as the cause of the E5 conc2 difference.
#
# Standalone (does not go through s14d_run_one.sh) so it can run while/after the
# E5 driver without editing a script that is being executed.
set -u
cd /home/lyh/z100-vllm
LOGDIR=/mnt/game/s14d/hostlogs
OUT=/mnt/game/s14d
mkdir -p "$LOGDIR" "$OUT"
NAME=E5SEQ
CT=z100-s14
MDL=/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8
CACHE=/mnt/game/s14d/cache-e5-seq
PORT=8623
LOG="$LOGDIR/$NAME.log"
: > "$LOG"

echo "[seq2] $(date -Is) start; swappiness=$(cat /proc/sys/vm/swappiness)"; cat /proc/swaps; free -m
docker exec "$CT" bash -c "mkdir -p $OUT '$CACHE' && rm -f $OUT/seq2-$NAME.jsonl /tmp/s14c_${NAME}.pid" || exit 1

nohup timeout 3600 docker exec -i \
  -e S14_CACHE="$CACHE" -e S14_DEVICES="0,1,2,3" -e S14_SERVED="qwen3.8-27b" \
  -e S14_LOGLEVEL=INFO -e S14_UTIL=0.85 -e S14_MBT=2048 \
  -e S14_TRITON_CACHE=/mnt/game/triton-gfx906/cache -e S14_SRC=/src -e S14_EAGER=1 \
  -e S14_CG_MODE= -e S14_CG_SIZES= -e S14C_TAG="e3_seq2_tp4_eager_sameprompt_twice_sw60" \
  "$CT" bash -s -- "$NAME" "$MDL" 4 1 "$PORT" 262144 < "$(dirname "$0")/s14c_arm_env.sh" >> "$LOG" 2>&1 &
PID=$!
echo "[seq2] engine_wrapper_pid=$PID"

t0=$(date +%s)
while true; do
  grep -q "Application startup complete" "$LOG" && { echo "[seq2] READY after $(( $(date +%s) - t0 ))s"; break; }
  grep -qE "Traceback|EngineCore failed|AcceleratorError" "$LOG" && { echo "[seq2] FATAL"; break; }
  [ $(( $(date +%s) - t0 )) -gt 1800 ] && { echo "[seq2] TIMEOUT"; break; }
  sleep 10
done

docker exec "$CT" python /src/tools/dcu/s14d_seq2.py --port "$PORT" --model qwen3.8-27b \
  --label "$NAME" --out "$OUT/seq2-$NAME.jsonl" 2>&1 | tee "$LOGDIR/$NAME-seq2.log"

docker exec "$CT" bash -c "kill \$(cat /tmp/s14c_${NAME}.pid) 2>/dev/null" || true
sleep 20
docker exec "$CT" bash -c "kill -9 \$(cat /tmp/s14c_${NAME}.pid) 2>/dev/null" || true
kill "$PID" 2>/dev/null || true
sleep 20
pkill VLLM 2>/dev/null || true
sleep 3
echo "[seq2] $(date -Is) done"; cat /proc/swaps; free -m
