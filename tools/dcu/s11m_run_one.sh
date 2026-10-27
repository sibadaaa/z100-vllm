#!/usr/bin/env bash
# S11m: host-side orchestrator for one 27B run with an explicit cache root.
#
#   tools/dcu/s11m_run_one.sh <NAME> <SW> <CACHE_ROOT> <PORT> <MAXLEN> <TASKS> [EXTRA...]
#
# TASKS = comma separated subset of: flip, flipb, e2e, startup
#   startup -> kill the engine as soon as the AOT load/recompile decision is
#              visible (cheap probe, ~5 min instead of ~28 min)
# Host artifacts: $WS/.tmp-11m/<NAME>.log (engine), <NAME>-*.log (clients),
#                 $WS/.tmp-11m/summary.tsv (one machine-readable row per run)
set -u
NAME=$1; SW=$2; CACHE=$3; PORT=$4; MAXLEN=$5; TASKS=$6; shift 6
EXTRA="$*"
WS=/home/lyh/z100-vllm
CT=z100-s9
LOGDIR="$WS/.tmp-11m"
mkdir -p "$LOGDIR"
ELOG="$LOGDIR/$NAME.log"
: > "$ELOG"

docker exec $CT bash -c "mkdir -p /mnt/game/s11m '$CACHE' && rm -f /mnt/game/s11m/flip-${NAME}.jsonl /mnt/game/s11m/flip-${NAME}b.jsonl" || exit 1

nohup timeout 3600 docker exec -i \
  -e S11M_CACHE="$CACHE" \
  -e S11M_LOGLEVEL="${S11M_LOGLEVEL:-INFO}" \
  -e S11M_DEVICES="${S11M_DEVICES:-0,1}" \
  -e S11M_TRITON_CACHE="${S11M_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}" \
  $CT bash -s -- "$NAME" "$SW" "$PORT" "$MAXLEN" $EXTRA \
  < "$WS/tools/dcu/s11m_arm_env.sh" >> "$ELOG" 2>&1 &
HOST_PID=$!
echo "[orch] start name=$NAME sw=$SW cache=$CACHE port=$PORT maxlen=$MAXLEN tasks=$TASKS extra='$EXTRA' host_pid=$HOST_PID"

if [ "$TASKS" = "startup" ]; then
  # Cheap probe: wait for the AOT decision, then stop.
  t0=$(date +%s)
  while true; do
    if grep -qE "Directly load AOT compilation from path|Compiling model again due to a load failure" "$ELOG"; then break; fi
    if grep -qE "EngineCore failed to start|Traceback" "$ELOG"; then echo "[orch] FATAL marker"; break; fi
    if [ $(( $(date +%s) - t0 )) -gt "${PROBE_TIMEOUT:-900}" ]; then echo "[orch] probe TIMEOUT"; break; fi
    sleep 5
  done
  echo "[orch] probe hit after $(( $(date +%s) - t0 ))s"
  docker exec $CT bash -c "kill \$(cat /tmp/s11m_${NAME}.pid) 2>/dev/null" || true
  sleep 20
  docker exec $CT bash -c "kill -9 \$(cat /tmp/s11m_${NAME}.pid) 2>/dev/null" || true
  echo "[orch] done name=$NAME probe"
  exit 0
fi

t0=$(date +%s)
READY=0
while true; do
  if grep -q "Application startup complete" "$ELOG"; then READY=1; break; fi
  if grep -qE "ConstraintViolationError|EngineCore failed to start" "$ELOG"; then
    echo "[orch] FATAL marker in $ELOG"; break
  fi
  if [ $(( $(date +%s) - t0 )) -gt "${READY_TIMEOUT:-2700}" ]; then echo "[orch] TIMEOUT waiting for readiness"; break; fi
  sleep 10
done
READY_S=$(( $(date +%s) - t0 ))
echo "[orch] ready=$READY after ${READY_S}s"

if [ "$READY" = "1" ]; then
  IFS=',' read -ra TS <<< "$TASKS"
  for t in "${TS[@]}"; do
    echo "[orch] task=$t start $(date -Is)"
    case "$t" in
      flip)  docker exec $CT python /src/tools/dcu/s11l_flip_stats.py --port "$PORT" --label "$NAME" --out "/mnt/game/s11m/flip-$NAME.jsonl" 2>&1 | tee "$LOGDIR/$NAME-flip.log" ;;
      flipb) docker exec $CT python /src/tools/dcu/s11l_flip_stats.py --port "$PORT" --label "${NAME}b" --out "/mnt/game/s11m/flip-${NAME}b.jsonl" 2>&1 | tee "$LOGDIR/$NAME-flipb.log" ;;
      e2e)   docker exec $CT python /src/tools/dcu/s11k_e2e_client.py --port "$PORT" 2>&1 | tee "$LOGDIR/$NAME-e2e.log" ;;
      *) echo "[orch] unknown task $t" ;;
    esac
    echo "[orch] task=$t done rc=$? $(date -Is)"
  done
fi

docker exec $CT bash -c "kill \$(cat /tmp/s11m_${NAME}.pid) 2>/dev/null" || true
sleep 20
docker exec $CT bash -c "kill -9 \$(cat /tmp/s11m_${NAME}.pid) 2>/dev/null" || true

# ---- machine-readable summary -------------------------------------------
aot_load=$(grep -c "Directly load AOT compilation from path" "$ELOG" || true)
aot_recompile=$(grep -c "Compiling model again due to a load failure" "$ELOG" || true)
ind_hit=$(grep -c "Directly load the compiled graph(s)" "$ELOG" || true)
ind_save=$(grep -c "saved AOT compiled function" "$ELOG" || true)
init_line=$(grep -oE "init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s \(compilation: [0-9.]+ s\)" "$ELOG" | head -1)
wload=$(grep -oE "Loading weights took [0-9.]+ seconds" "$ELOG" | head -1)
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$NAME" "$SW" "$CACHE" "$MAXLEN" "$READY" "$READY_S" \
  "${aot_load}" "${aot_recompile}" "${ind_hit}" "${ind_save}" \
  "${init_line:-NA}" "${wload:-NA}" >> "$LOGDIR/summary.tsv"
echo "[orch] summary: ready=${READY_S}s aot_load=${aot_load} aot_recompile=${aot_recompile} ind_hit=${ind_hit} ind_save=${ind_save} | ${init_line:-NA} | ${wload:-NA}"
echo "[orch] done name=$NAME ready=$READY"
