#!/usr/bin/env bash
# S14e: host-side orchestrator for ONE measured engine start (grid cell G1..G5).
# Copy of tools/dcu/s14_run_one.sh (never edited) with:
#   * artifacts under /mnt/game/s14e,
#   * a new task `e14` running tools/dcu/s14e_client.py (the 12-prompt
#     deterministic battery: 4 zh + 4 en short + 3 zh/en medium + 1 long zh),
#   * the S14c memory knobs (swap / glibc) left to the caller,
#   * extra recording of the two graph-capture wall times and of the
#     cudagraph bucket count actually used.
#
#   tools/dcu/s14e_run_one.sh <NAME> <MODEL> <TP> <SW> <CACHE> <PORT> <MAXLEN> <DEVICES> [EXTRA...]
#
# Everything else (env passthrough, readiness detection, teardown) is the S14
# recipe verbatim so the numbers stay comparable with S14c/S14d.
set -u
NAME=$1; MODEL=$2; TP=$3; SW=$4; CACHE=$5; PORT=$6; MAXLEN=$7; DEVICES=$8
shift 8
EXTRA="$*"
CT="${S14_CT:-z100-s14}"
LOGDIR="${S14_LOGDIR:-/mnt/game/s14e/hostlogs}"
OUT="${S14_BENCH_OUT:-/mnt/game/s14e}"
mkdir -p "$LOGDIR"
ELOG="$LOGDIR/$NAME.log"
: > "$ELOG"
SERVED="${S14_SERVED:-qwen3.8-27b}"
CAR="${S14_CAR:-1}"

echo "[orch14e] container=$CT mem_limit=$(docker inspect "$CT" --format '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}')"
echo "[orch14e] free_before: $(free -m | tr '\n' ' ')"
echo "[orch14e] cell=$NAME tp=$TP sw=$SW maxlen=$MAXLEN dev=$DEVICES cache=$CACHE cg_mode=${S14_CG_MODE:-default} cg_sizes=${S14_CG_SIZES:-default} util=${S14_UTIL:-0.85} extras='$EXTRA'"
docker exec "$CT" bash -c "mkdir -p $OUT '$CACHE' && rm -f $OUT/probe-${NAME}.jsonl /tmp/s14_${NAME}.pid && cat /sys/fs/cgroup/memory.events > $OUT/cgroup-events-${NAME}-before.txt" || exit 1

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
echo "[orch14e] engine_wrapper_pid=$ENGINE_PID"
echo "$ENGINE_PID" > "$LOGDIR/$NAME.wrapper.pid"

t0=$(date +%s)
READY=0
while true; do
  if grep -q "Application startup complete" "$ELOG"; then READY=1; break; fi
  if grep -qE "ConstraintViolationError|EngineCore failed to start|Engine core initialization failed|AcceleratorError|Traceback" "$ELOG"; then
    echo "[orch14e] FATAL marker in $ELOG"; break
  fi
  if [ $(( $(date +%s) - t0 )) -gt "${READY_TIMEOUT:-3600}" ]; then echo "[orch14e] TIMEOUT waiting for readiness"; break; fi
  sleep 10
done
READY_S=$(( $(date +%s) - t0 ))
echo "[orch14e] ready=$READY after ${READY_S}s  $(date -Is)"
echo "[orch14e] free_after_start: $(free -m | tr '\n' ' ')"
grep -E "Graph capturing finished|init engine \(profile|torch.compile took|Directly load AOT|cudagraph" "$ELOG" | tail -12

if [ "$READY" = "1" ]; then
  # S14E_TASKS: comma separated subset of {flip,e14}, run in the given order.
  #   flip = tools/dcu/s11l_flip_stats.py, the *identical* instrument S11m used
  #          (8 prompts x 64 tokens, no prompt_logprobs) -> directly comparable
  #          with /mnt/game/s11m/flip-S2/S3/S4sw1hit.jsonl via s11m_flip_matrix.py.
  #   e14  = tools/dcu/s14e_client.py, the S14d instrument (12 prompts x 128
  #          tokens + per-prompt-position prompt_logprobs).
  IFS=',' read -ra TS <<< "${S14E_TASKS:-e14}"
  for t in "${TS[@]}"; do
    echo "[orch14e] task=$t start $(date -Is)"
    case "$t" in
      flip) docker exec "$CT" python "${S14_SRC:-/src}"/tools/dcu/s11l_flip_stats.py \
              --port "$PORT" --model "$SERVED" --label "$NAME" \
              --out "$OUT/flip-$NAME.jsonl" 2>&1 | tee "$LOGDIR/$NAME-flip.log" ;;
      e14)  docker exec "$CT" python "${S14_SRC:-/src}"/tools/dcu/s14e_client.py \
              --port "$PORT" --model "$SERVED" --label "$NAME" \
              --out "$OUT/probe-$NAME.jsonl" --max-tokens "${S14E_TOKENS:-128}" \
              2>&1 | tee "$LOGDIR/$NAME-e14.log" ;;
      *) echo "[orch14e] unknown task $t" ;;
    esac
    echo "[orch14e] task=$t done rc=$? $(date -Is)"
  done
fi

# ---- stop ----------------------------------------------------------------
docker exec "$CT" bash -c "kill \$(cat /tmp/s14_${NAME}.pid) 2>/dev/null" || true
sleep 20
docker exec "$CT" bash -c "kill -9 \$(cat /tmp/s14_${NAME}.pid) 2>/dev/null" || true
kill "$ENGINE_PID" 2>/dev/null || true
sleep 5
echo "[orch14e] leftover procs: $(pgrep -c VLLM 2>/dev/null || echo 0)"
docker exec "$CT" bash -c "cat /sys/fs/cgroup/memory.events" | tee "$OUT/cgroup-events-${NAME}-after.txt"

# ---- machine-readable summary -------------------------------------------
aot_load=$(grep -c "Directly load AOT compilation from path" "$ELOG" || true)
aot_recompile=$(grep -c "Compiling model again due to a load failure" "$ELOG" || true)
ind_hit=$(grep -c "Directly load the compiled graph(s)" "$ELOG" || true)
init_line=$(grep -oE "init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s \(compilation: [0-9.]+ s\)" "$ELOG" | head -1)
kv=$(grep -oE "Available KV cache memory: [0-9.]+ GiB" "$ELOG" | head -1)
caps=$(grep -oE "Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB" "$ELOG" | tr '\n' ';')
oomk=$(docker exec "$CT" bash -c "cat /sys/fs/cgroup/memory.events" | awk '/^oom_kill/{print $2}')
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$NAME" "$TP" "$SW" "$MAXLEN" "$DEVICES" "${S14_CG_MODE:-default}" "${S14_CG_SIZES:-default}" \
  "$READY" "$READY_S" "${aot_load}" "${ind_hit}" "${oomk:-NA}" "$(date -Is)" >> "$LOGDIR/summary.tsv"
echo "[orch14e] summary: ready=${READY_S}s aot_load=${aot_load} ind_hit=${ind_hit} oom_kill=${oomk:-NA} | ${init_line:-NA} | ${kv:-NA}"
echo "[orch14e] captures: ${caps:-NA}"
echo "[orch14e] done name=$NAME ready=$READY"
