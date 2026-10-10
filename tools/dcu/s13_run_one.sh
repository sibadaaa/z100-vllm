#!/usr/bin/env bash
# S13: host-side orchestrator -- start one engine, run client tasks, stop it.
# Derived from tools/dcu/s12_run_one.sh; logs go to /mnt/game/s13 (the S12
# script wrote into the workspace .tmp-12, which is outside this step's write
# scope).
#
#   tools/dcu/s13_run_one.sh <NAME> <MODEL_DIR> <TP> <SW> <CACHE> <PORT> <MAXLEN> <DEVICES> <TASKS> [EXTRA...]
#
# TASKS = comma separated subset of: flip,decode,ttft,ttft8k,conc8
# Host artifacts -> /mnt/game/s13/hostlogs/<NAME>.log (engine) and <NAME>-*.log
# Container artifacts -> /mnt/game/s13/
#
# Rule (hard constraint 25): DEVICES is mandatory and explicit.
set -u
NAME=$1; MODEL=$2; TP=$3; SW=$4; CACHE=$5; PORT=$6; MAXLEN=$7; DEVICES=$8; TASKS=$9
shift 9
EXTRA="$*"
CT="${S13_CT:-z100-s9}"
LOGDIR="${S13_LOGDIR:-/mnt/game/s13/hostlogs}"
OUT="${S13_BENCH_OUT:-/mnt/game/s13}"
mkdir -p "$LOGDIR"
ELOG="$LOGDIR/$NAME.log"
: > "$ELOG"
SERVED="${S13_SERVED:-qwen3.8-27b}"
CAR="${S13_CAR:-1}"

docker exec "$CT" bash -c "mkdir -p $OUT '$CACHE' && rm -f $OUT/flip-${NAME}.jsonl /tmp/s13_${NAME}.pid" || exit 1

CAR_ENV=""
if [ "$CAR" = "0" ]; then CAR_ENV="-e S13_CAR=0"; fi
nohup timeout 5400 docker exec -i \
  -e S13_CACHE="$CACHE" \
  -e S13_DEVICES="$DEVICES" \
  -e S13_SERVED="$SERVED" \
  -e S13_LOGLEVEL="${S13_LOGLEVEL:-INFO}" \
  -e S13_TRITON_CACHE="${S13_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}" \
  -e S13_SRC="${S13_SRC:-/src}" \
  $CAR_ENV \
  "$CT" bash -s -- "$NAME" "$MODEL" "$TP" "$SW" "$PORT" "$MAXLEN" $EXTRA \
  < "$(dirname "$0")/s13_arm_env.sh" >> "$ELOG" 2>&1 &
echo "[orch13] start name=$NAME tp=$TP sw=$SW port=$PORT maxlen=$MAXLEN dev=$DEVICES car=$CAR tasks=$TASKS extra='$EXTRA'"

t0=$(date +%s)
READY=0
while true; do
  if grep -q "Application startup complete" "$ELOG"; then READY=1; break; fi
  if grep -qE "ConstraintViolationError|EngineCore failed to start|Engine core initialization failed|AcceleratorError|Traceback" "$ELOG"; then
    echo "[orch13] FATAL marker in $ELOG"; break
  fi
  if [ $(( $(date +%s) - t0 )) -gt "${READY_TIMEOUT:-2700}" ]; then echo "[orch13] TIMEOUT waiting for readiness"; break; fi
  sleep 10
done
READY_S=$(( $(date +%s) - t0 ))
echo "[orch13] ready=$READY after ${READY_S}s"

if [ "$READY" = "1" ]; then
  IFS=',' read -ra TS <<< "$TASKS"
  for t in "${TS[@]}"; do
    echo "[orch13] task=$t start $(date -Is)"
    case "$t" in
      flip)   docker exec "$CT" python "${S13_SRC:-/src}"/tools/dcu/s11l_flip_stats.py --port "$PORT" --model "$SERVED" --label "$NAME" --out "$OUT/flip-$NAME.jsonl" 2>&1 | tee "$LOGDIR/$NAME-flip.log" ;;
      decode|ttft|ttft8k|conc8)
              docker exec -e S13_BENCH_OUT="$OUT" "$CT" bash "${S13_SRC:-/src}"/tools/dcu/s13_bench.sh "$PORT" "$NAME" "$t" "$SERVED" "$MODEL" 2>&1 | tee "$LOGDIR/$NAME-bench-$t.log" ;;
      *) echo "[orch13] unknown task $t" ;;
    esac
    echo "[orch13] task=$t done rc=$? $(date -Is)"
  done
fi

docker exec "$CT" bash -c "kill \$(cat /tmp/s13_${NAME}.pid) 2>/dev/null" || true
sleep 20
docker exec "$CT" bash -c "kill -9 \$(cat /tmp/s13_${NAME}.pid) 2>/dev/null" || true

# ---- machine-readable summary -------------------------------------------
aot_load=$(grep -c "Directly load AOT compilation from path" "$ELOG" || true)
aot_recompile=$(grep -c "Compiling model again due to a load failure" "$ELOG" || true)
init_line=$(grep -oE "init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s \(compilation: [0-9.]+ s\)" "$ELOG" | head -1)
kv=$(grep -oE "Available KV cache memory: [0-9.]+ GiB" "$ELOG" | head -1)
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$NAME" "$MODEL" "$TP" "$SW" "$MAXLEN" "$DEVICES" "$READY" "$READY_S" \
  "${aot_load}" "${aot_recompile}" >> "$LOGDIR/summary.tsv"
echo "[orch13] summary: ready=${READY_S}s aot_load=${aot_load} aot_recompile=${aot_recompile} | ${init_line:-NA} | ${kv:-NA}"
echo "[orch13] done name=$NAME ready=$READY"
