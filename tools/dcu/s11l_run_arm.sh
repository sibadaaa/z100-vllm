#!/usr/bin/env bash
# S11l: host-side orchestrator for one 27B arm: start the engine, wait for
# readiness, run the requested client tasks, stop the engine.
#
#   tools/dcu/s11l_run_arm.sh <NAME> <SW> <INNER> <PORT> <MAXLEN> <tasks>
#
# tasks = comma separated subset of: e2e,flip,8k,decode,ttft,ttft8k,conc8
# Host-side artifacts -> $WS/.tmp-11l/<NAME>-*.log (engine log: <NAME>.log)
# Container-side artifacts -> /mnt/game/s11l/ (flip JSONL, bench logs)
set -u
NAME=$1; SW=$2; INNER=$3; PORT=$4; MAXLEN=$5; TASKS=$6
WS=/home/lyh/z100-vllm
CT=z100-s9
LOGDIR="$WS/.tmp-11l"
mkdir -p "$LOGDIR"
ELOG="$LOGDIR/$NAME.log"
: > "$ELOG"

docker exec $CT bash -c "mkdir -p /mnt/game/s11l && rm -f /mnt/game/s11l/flip-${NAME}.jsonl /mnt/game/s11l/flip-${NAME}b.jsonl /tmp/s11l_${NAME}.pid" || exit 1

nohup timeout 3600 docker exec -i $CT bash -s -- "$NAME" "$SW" "$INNER" "$PORT" "$MAXLEN" \
  < "$WS/tools/dcu/s11l_arm_env.sh" >> "$ELOG" 2>&1 &
HOST_PID=$!
echo "[orch] start name=$NAME sw=$SW inner=$INNER port=$PORT maxlen=$MAXLEN tasks=$TASKS host_pid=$HOST_PID"

t0=$(date +%s)
READY=0
while true; do
  if grep -q "Application startup complete" "$ELOG"; then READY=1; break; fi
  if grep -qE "ConstraintViolationError|EngineCore failed to start|RuntimeError" "$ELOG"; then
    echo "[orch] FATAL marker in $ELOG"; break
  fi
  if [ $(( $(date +%s) - t0 )) -gt "${READY_TIMEOUT:-2700}" ]; then echo "[orch] TIMEOUT waiting for readiness"; break; fi
  sleep 10
done
echo "[orch] ready=$READY after $(( $(date +%s) - t0 ))s"

if [ "$READY" = "1" ]; then
  IFS=',' read -ra TS <<< "$TASKS"
  for t in "${TS[@]}"; do
    echo "[orch] task=$t start $(date -Is)"
    case "$t" in
      e2e)    docker exec $CT python /src/tools/dcu/s11k_e2e_client.py --port "$PORT" 2>&1 | tee "$LOGDIR/$NAME-e2e.log" ;;
      flip)   docker exec $CT python /src/tools/dcu/s11l_flip_stats.py --port "$PORT" --label "$NAME" --out "/mnt/game/s11l/flip-$NAME.jsonl" 2>&1 | tee "$LOGDIR/$NAME-flip.log" ;;
      flipb)  docker exec $CT python /src/tools/dcu/s11l_flip_stats.py --port "$PORT" --label "${NAME}b" --out "/mnt/game/s11l/flip-${NAME}b.jsonl" 2>&1 | tee "$LOGDIR/$NAME-flipb.log" ;;
      8k)     # Needs the DTK env (transformers -> torch); s11l_8k_probe.py is not a
              # stdlib-only client like the flip probe.
              docker exec $CT bash -lc ". /opt/dtk/env.sh >/dev/null 2>&1; . /src/tools/dcu/env.sh >/dev/null 2>&1; export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache; export HIP_VISIBLE_DEVICES=\${S11L_DEVICES:-0,1}; python /src/tools/dcu/s11l_8k_probe.py --port $PORT" 2>&1 | tee "$LOGDIR/$NAME-8k.log" ;;
      decode|ttft|ttft8k|conc8) docker exec $CT bash /src/tools/dcu/s11l_bench.sh "$PORT" "$NAME" "$t" 2>&1 | tee "$LOGDIR/$NAME-bench-$t.log" ;;
      *) echo "[orch] unknown task $t" ;;
    esac
    echo "[orch] task=$t done rc=$? $(date -Is)"
  done
fi

docker exec $CT bash -c "kill \$(cat /tmp/s11l_${NAME}.pid) 2>/dev/null" || true
sleep 20
docker exec $CT bash -c "kill -9 \$(cat /tmp/s11l_${NAME}.pid) 2>/dev/null" || true
echo "[orch] done name=$NAME ready=$READY"
