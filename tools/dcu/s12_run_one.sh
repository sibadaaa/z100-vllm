#!/usr/bin/env bash
# S12: host-side orchestrator -- start one engine, run client tasks, stop it.
#
#   tools/dcu/s12_run_one.sh <NAME> <MODEL_DIR> <TP> <SW> <CACHE> <PORT> <MAXLEN> <DEVICES> <TASKS> [EXTRA...]
#
# TASKS = comma separated subset of: flip,flipb,decode,ttft,ttft8k,conc8,e2e
# Host artifacts -> $WS/.tmp-12/<NAME>.log (engine) and <NAME>-*.log (clients)
# Container artifacts -> /mnt/game/s12/
#
# Rule (hard constraint 25): DEVICES is mandatory and explicit.
set -u
NAME=$1; MODEL=$2; TP=$3; SW=$4; CACHE=$5; PORT=$6; MAXLEN=$7; DEVICES=$8; TASKS=$9
shift 9
EXTRA="$*"
WS=/home/lyh/z100-vllm
CT="${S12_CT:-z100-s9}"
LOGDIR="$WS/.tmp-12"
mkdir -p "$LOGDIR"
ELOG="$LOGDIR/$NAME.log"
: > "$ELOG"
SERVED="${S12_SERVED:-qwen3.8-27b}"

docker exec "$CT" bash -c "mkdir -p /mnt/game/s12 '$CACHE' && rm -f /mnt/game/s12/flip-${NAME}.jsonl /mnt/game/s12/flip-${NAME}b.jsonl /tmp/s12_${NAME}.pid" || exit 1

nohup timeout 5400 docker exec -i \
  -e S12_CACHE="$CACHE" \
  -e S12_DEVICES="$DEVICES" \
  -e S12_SERVED="$SERVED" \
  -e S12_LOGLEVEL="${S12_LOGLEVEL:-INFO}" \
  -e S12_TRITON_CACHE="${S12_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}" \
  -e S12_SRC="${S12_SRC:-/src}" \
  "$CT" bash -s -- "$NAME" "$MODEL" "$TP" "$SW" "$PORT" "$MAXLEN" $EXTRA \
  < "$WS/tools/dcu/s12_arm_env.sh" >> "$ELOG" 2>&1 &
echo "[orch] start name=$NAME model=$MODEL tp=$TP sw=$SW port=$PORT maxlen=$MAXLEN dev=$DEVICES tasks=$TASKS extra='$EXTRA'"

t0=$(date +%s)
READY=0
while true; do
  if grep -q "Application startup complete" "$ELOG"; then READY=1; break; fi
  if grep -qE "ConstraintViolationError|EngineCore failed to start|Engine core initialization failed" "$ELOG"; then
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
      flip)   docker exec "$CT" python "${S12_SRC:-/src}"/tools/dcu/s11l_flip_stats.py --port "$PORT" --model "$SERVED" --label "$NAME" --out "/mnt/game/s12/flip-$NAME.jsonl" 2>&1 | tee "$LOGDIR/$NAME-flip.log" ;;
      flipb)  docker exec "$CT" python "${S12_SRC:-/src}"/tools/dcu/s11l_flip_stats.py --port "$PORT" --model "$SERVED" --label "${NAME}b" --out "/mnt/game/s12/flip-${NAME}b.jsonl" 2>&1 | tee "$LOGDIR/$NAME-flipb.log" ;;
      e2e)    docker exec "$CT" python "${S12_SRC:-/src}"/tools/dcu/s11k_e2e_client.py --port "$PORT" 2>&1 | tee "$LOGDIR/$NAME-e2e.log" ;;
      decode|ttft|ttft8k|conc8)
              docker exec "$CT" bash "${S12_SRC:-/src}"/tools/dcu/s12_bench.sh "$PORT" "$NAME" "$t" "$SERVED" "$MODEL" 2>&1 | tee "$LOGDIR/$NAME-bench-$t.log" ;;
      *) echo "[orch] unknown task $t" ;;
    esac
    echo "[orch] task=$t done rc=$? $(date -Is)"
  done
fi

docker exec "$CT" bash -c "kill \$(cat /tmp/s12_${NAME}.pid) 2>/dev/null" || true
sleep 20
docker exec "$CT" bash -c "kill -9 \$(cat /tmp/s12_${NAME}.pid) 2>/dev/null" || true

# ---- machine-readable summary -------------------------------------------
aot_load=$(grep -c "Directly load AOT compilation from path" "$ELOG" || true)
aot_recompile=$(grep -c "Compiling model again due to a load failure" "$ELOG" || true)
init_line=$(grep -oE "init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s \(compilation: [0-9.]+ s\)" "$ELOG" | head -1)
kv=$(grep -oE "Available KV cache memory: [0-9.]+ GiB" "$ELOG" | head -1)
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$NAME" "$MODEL" "$TP" "$SW" "$MAXLEN" "$DEVICES" "$READY" "$READY_S" \
  "${aot_load}" "${aot_recompile}" >> "$LOGDIR/summary.tsv"
echo "[orch] summary: ready=${READY_S}s aot_load=${aot_load} aot_recompile=${aot_recompile} | ${init_line:-NA} | ${kv:-NA}"
echo "[orch] done name=$NAME ready=$READY"
