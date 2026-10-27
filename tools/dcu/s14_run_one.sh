#!/usr/bin/env bash
# S14: host-side orchestrator -- start one engine, run client tasks, stop it.
# Derived from tools/dcu/s13_run_one.sh; S14 differences:
#   * default container is z100-s14 (the no-cgroup-limit clone of z100-s9),
#     overridable with S14_CT,
#   * artifacts go to /mnt/game/s14,
#   * readiness timeout default 3600 s (TP=4 cold compile + graph capture),
#   * extra recording: cgroup memory.events (oom_kill) at stop time and
#     docker inspect memory limit of the container actually used.
#
#   tools/dcu/s14_run_one.sh <NAME> <MODEL_DIR> <TP> <SW> <CACHE> <PORT> <MAXLEN> <DEVICES> <TASKS> [EXTRA...]
#
# TASKS = comma separated subset of: flip,decode,ttft,ttft8k,conc8,greedy
# Host artifacts -> /mnt/game/s14/hostlogs/<NAME>.log (engine) and <NAME>-*.log
# Container artifacts -> /mnt/game/s14/
#
# Rule: DEVICES is mandatory and explicit (never unset HIP_VISIBLE_DEVICES).
set -u
NAME=$1; MODEL=$2; TP=$3; SW=$4; CACHE=$5; PORT=$6; MAXLEN=$7; DEVICES=$8; TASKS=$9
shift 9
EXTRA="$*"
CT="${S14_CT:-z100-s14}"
LOGDIR="${S14_LOGDIR:-/mnt/game/s14/hostlogs}"
OUT="${S14_BENCH_OUT:-/mnt/game/s14}"
mkdir -p "$LOGDIR"
ELOG="$LOGDIR/$NAME.log"
: > "$ELOG"
SERVED="${S14_SERVED:-qwen3.8-27b}"
CAR="${S14_CAR:-1}"

echo "[orch14] container=$CT mem_limit=$(docker inspect "$CT" --format '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}')"
echo "[orch14] free_before: $(free -m | tr '\n' ' ')"
docker exec "$CT" bash -c "mkdir -p $OUT '$CACHE' && rm -f $OUT/flip-${NAME}.jsonl /tmp/s14_${NAME}.pid && cat /sys/fs/cgroup/memory.events > $OUT/cgroup-events-${NAME}-before.txt" || exit 1

CAR_ENV=""
if [ "$CAR" = "0" ]; then CAR_ENV="-e S14_CAR=0"; fi
nohup timeout "${S14_ENGINE_TIMEOUT:-7200}" docker exec -i \
  -e S14_CACHE="$CACHE" \
  -e S14_DEVICES="$DEVICES" \
  -e S14_SERVED="$SERVED" \
  -e S14_LOGLEVEL="${S14_LOGLEVEL:-INFO}" \
  -e S14_UTIL="${S14_UTIL:-0.85}" \
  -e S14_MBT="${S14_MBT:-2048}" \
  -e S14_TRITON_CACHE="${S14_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}" \
  -e S14_SRC="${S14_SRC:-/src}" \
  -e S14_EAGER="${S14_EAGER:-0}" \
  -e S14_CG_MODE="${S14_CG_MODE:-}" \
  -e S14_CG_SIZES="${S14_CG_SIZES:-}" \
  $CAR_ENV \
  "$CT" bash -s -- "$NAME" "$MODEL" "$TP" "$SW" "$PORT" "$MAXLEN" $EXTRA \
  < "$(dirname "$0")/s14_arm_env.sh" >> "$ELOG" 2>&1 &
ENGINE_PID=$!
echo "[orch14] engine_wrapper_pid=$ENGINE_PID"
echo "$ENGINE_PID" > "$LOGDIR/$NAME.wrapper.pid"

t0=$(date +%s)
READY=0
while true; do
  if grep -q "Application startup complete" "$ELOG"; then READY=1; break; fi
  if grep -qE "ConstraintViolationError|EngineCore failed to start|Engine core initialization failed|AcceleratorError|Traceback" "$ELOG"; then
    echo "[orch14] FATAL marker in $ELOG"; break
  fi
  if [ $(( $(date +%s) - t0 )) -gt "${READY_TIMEOUT:-3600}" ]; then echo "[orch14] TIMEOUT waiting for readiness"; break; fi
  sleep 10
done
READY_S=$(( $(date +%s) - t0 ))
echo "[orch14] ready=$READY after ${READY_S}s  $(date -Is)"
echo "[orch14] free_after_start: $(free -m | tr '\n' ' ')"
docker exec "$CT" bash -c "cat /sys/fs/cgroup/memory.events; echo '--- current:'; cat /sys/fs/cgroup/memory.current" 2>&1 | tee "$LOGDIR/$NAME-cgroup-after-start.txt"

if [ "$READY" = "1" ]; then
  IFS=',' read -ra TS <<< "$TASKS"
  for t in "${TS[@]}"; do
    echo "[orch14] task=$t start $(date -Is)"
    case "$t" in
      flip)   docker exec "$CT" python "${S14_SRC:-/src}"/tools/dcu/s11l_flip_stats.py --port "$PORT" --model "$SERVED" --label "$NAME" --out "$OUT/flip-$NAME.jsonl" --max-tokens "${S14_FLIP_TOKENS:-128}" 2>&1 | tee "$LOGDIR/$NAME-flip.log" ;;
      greedy) docker exec "$CT" python "${S14_SRC:-/src}"/tools/dcu/s14_greedy.py --port "$PORT" --model "$SERVED" --label "$NAME" --out "$OUT/greedy-$NAME.jsonl" --max-tokens "${S14_GREEDY_TOKENS:-256}" 2>&1 | tee "$LOGDIR/$NAME-greedy.log" ;;
      decode|ttft|ttft8k|conc8)
              docker exec -e S14_BENCH_OUT="$OUT" "$CT" bash "${S14_SRC:-/src}"/tools/dcu/s14_bench.sh "$PORT" "$NAME" "$t" "$SERVED" "$MODEL" 2>&1 | tee "$LOGDIR/$NAME-bench-$t.log" ;;
      smaps)  bash "$(dirname "$0")/s14_smaps.sh" "$NAME" 2>&1 | tee "$LOGDIR/$NAME-smaps.log" ;;
      *) echo "[orch14] unknown task $t" ;;
    esac
    echo "[orch14] task=$t done rc=$? $(date -Is)"
  done
fi

# ---- stop ----------------------------------------------------------------
docker exec "$CT" bash -c "kill \$(cat /tmp/s14_${NAME}.pid) 2>/dev/null" || true
sleep 20
docker exec "$CT" bash -c "kill -9 \$(cat /tmp/s14_${NAME}.pid) 2>/dev/null" || true
kill "$ENGINE_PID" 2>/dev/null || true
sleep 5
echo "[orch14] leftover procs: $(pgrep -c VLLM 2>/dev/null || echo 0)"
docker exec "$CT" bash -c "cat /sys/fs/cgroup/memory.events" | tee "$OUT/cgroup-events-${NAME}-after.txt"

# ---- machine-readable summary -------------------------------------------
aot_load=$(grep -c "Directly load AOT compilation from path" "$ELOG" || true)
aot_recompile=$(grep -c "Compiling model again due to a load failure" "$ELOG" || true)
init_line=$(grep -oE "init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s \(compilation: [0-9.]+ s\)" "$ELOG" | head -1)
kv=$(grep -oE "Available KV cache memory: [0-9.]+ GiB" "$ELOG" | head -1)
oomk=$(docker exec "$CT" bash -c "cat /sys/fs/cgroup/memory.events" | awk '/^oom_kill/{print $2}')
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$NAME" "$MODEL" "$TP" "$SW" "$MAXLEN" "$DEVICES" "$READY" "$READY_S" \
  "${aot_load}" "${aot_recompile}" "${oomk:-NA}" "$(date -Is)" >> "$LOGDIR/summary.tsv"
echo "[orch14] summary: ready=${READY_S}s aot_load=${aot_load} aot_recompile=${aot_recompile} oom_kill=${oomk:-NA} | ${init_line:-NA} | ${kv:-NA}"
echo "[orch14] done name=$NAME ready=$READY"
