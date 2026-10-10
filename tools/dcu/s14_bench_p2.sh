#!/usr/bin/env bash
# S14: take over the ALREADY-RUNNING P2 engine (TP=2 / graph mode / port 8421,
# started by the previous agent), wait for readiness WITHOUT restarting it, then
# run the full G1+G3 battery for that arm.
#
#   tools/dcu/s14_bench_p2.sh
#
# Readiness is detected in hostlogs/P2.log; the engine is never touched except
# through its HTTP port until the caller stops it.
set -u
CT="${S14_CT:-z100-s14}"
NAME="${S14_NAME:-P2}"
PORT="${S14_PORT:-8421}"
SERVED="${S14_SERVED:-qwen3.8-27b}"
MODEL="${S14_MODEL:-/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8}"
SRC="${S14_SRC:-/src}"
LOGDIR="${S14_LOGDIR:-/mnt/game/s14/hostlogs}"
OUT="${S14_BENCH_OUT:-/mnt/game/s14}"
ELOG="$LOGDIR/$NAME.log"
TASKS="${S14_TASKS:-greedy,flip,ttft8k,decode,ttft,conc8}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"

echo "[p2] waiting for readiness in $ELOG (timeout ${READY_TIMEOUT}s) $(date -Is)"
t0=$(date +%s)
DEADLINE=$(( t0 + READY_TIMEOUT ))
while true; do
  if grep -q "Application startup complete" "$ELOG"; then echo "[p2] READY $(date -Is)"; break; fi
  if grep -qE "ConstraintViolationError|EngineCore failed to start|Engine core initialization failed|AcceleratorError|Traceback" "$ELOG"; then
    echo "[p2] FATAL marker in $ELOG $(date -Is)"; exit 1
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then echo "[p2] TIMEOUT $(date -Is)"; exit 2; fi
  sleep 10
done
echo "[p2] ready after $(( $(date +%s) - t0 ))s"
echo "[p2] free_at_ready: $(free -m | tr '\n' ' ')"

IFS=',' read -ra TS <<< "$TASKS"
for t in "${TS[@]}"; do
  echo "[p2] task=$t start $(date -Is)"
  case "$t" in
    greedy) docker exec "$CT" python "$SRC"/tools/dcu/s14_greedy.py --port "$PORT" --model "$SERVED" \
              --label "$NAME" --out "$OUT/greedy-$NAME.jsonl" --max-tokens "${S14_GREEDY_TOKENS:-256}" 2>&1 \
              | tee "$LOGDIR/$NAME-greedy.log" ;;
    flip)   docker exec "$CT" python "$SRC"/tools/dcu/s11l_flip_stats.py --port "$PORT" --model "$SERVED" \
              --label "$NAME" --out "$OUT/flip-$NAME.jsonl" --max-tokens "${S14_FLIP_TOKENS:-128}" 2>&1 \
              | tee "$LOGDIR/$NAME-flip.log" ;;
    decode|ttft|ttft8k|conc8)
            docker exec -e S14_BENCH_OUT="$OUT" "$CT" bash "$SRC"/tools/dcu/s14_bench.sh \
              "$PORT" "$NAME" "$t" "$SERVED" "$MODEL" 2>&1 | tee "$LOGDIR/$NAME-bench-$t.log" ;;
    *) echo "[p2] unknown task $t" ;;
  esac
  echo "[p2] task=$t done rc=$? $(date -Is)"
done
echo "[p2] ALL DONE $(date -Is)"
