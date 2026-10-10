#!/usr/bin/env bash
# S14g: host-side orchestrator for ONE measured engine start.
#
# Structurally a copy of tools/dcu/s14e_run_one.sh (S14e files stay frozen) with
# three S14g additions:
#   * artifacts under /mnt/game/s14g by default,
#   * S14G_PIN_PATH / S14G_TRITON_ENV / S14G_PREFIX knobs (Triton autotune pin,
#     extra container env and an optional launcher prefix such as
#     `setarch x86_64 -R`) plumbed through tools/dcu/s14g_arm_env.sh,
#   * an address-layout fingerprint (tools/dcu/s14g_maps.sh) taken right after
#     readiness, plus the raw `Triton autotuning ... best config selected` lines
#     extracted from the engine log into autotune-<NAME>.txt.
#
#   tools/dcu/s14g_run_one.sh <NAME> <MODEL> <TP> <SW> <CACHE> <PORT> <MAXLEN> <DEVICES> [EXTRA...]
set -u
NAME=$1; MODEL=$2; TP=$3; SW=$4; CACHE=$5; PORT=$6; MAXLEN=$7; DEVICES=$8
shift 8
EXTRA="$*"
CT="${S14_CT:-z100-s14}"
LOGDIR="${S14_LOGDIR:-/mnt/game/s14g/hostlogs}"
OUT="${S14_BENCH_OUT:-/mnt/game/s14g}"
mkdir -p "$LOGDIR"
ELOG="$LOGDIR/$NAME.log"
: > "$ELOG"
SERVED="${S14_SERVED:-qwen3.8-27b}"

echo "[orch14g] container=$CT free_before: $(free -m | tr '\n' ' ')"
echo "[orch14g] cell=$NAME tp=$TP sw=$SW maxlen=$MAXLEN dev=$DEVICES cache=$CACHE prefix='${S14G_PREFIX:-}' pin='${S14G_PIN_PATH:-}' triton_env='${S14G_TRITON_ENV:-}'"
docker exec "$CT" bash -c "mkdir -p $OUT '$CACHE' && rm -f $OUT/probe-${NAME}.jsonl $OUT/flip-${NAME}.jsonl /tmp/s14_${NAME}.pid && cat /sys/fs/cgroup/memory.events > $OUT/cgroup-events-${NAME}-before.txt" || exit 1

# Optional env forwarders: only forwarded when the caller actually set them, so
# that a run without the knob is byte-for-byte the S14e environment.
ENVARGS=(
  -e "S14_CACHE=$CACHE" -e "S14_DEVICES=$DEVICES" -e "S14_SERVED=$SERVED"
  -e "S14_LOGLEVEL=${S14_LOGLEVEL:-INFO}" -e "S14_UTIL=${S14_UTIL:-0.9}"
  -e "S14_MBT=${S14_MBT:-2048}"
  -e "S14_TRITON_CACHE=${S14_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}"
  -e "S14_SRC=${S14_SRC:-/src}" -e "S14_EAGER=${S14_EAGER:-0}"
  -e "S14_CG_MODE=${S14_CG_MODE:-}" -e "S14_CG_SIZES=${S14_CG_SIZES:-}"
  -e "S14G_PIN_PATH=${S14G_PIN_PATH:-}"
  -e "S14G_LAUNCH_PREFIX=${S14G_LAUNCH_PREFIX:-}"
  -e "S14G_PIN_MODE=${S14G_PIN_MODE:-}"
  -e "S14G_PIN_CHOICE=${S14G_PIN_CHOICE:-}"
  -e "S14G_AUTOTUNE_LOG=$OUT/autotune-log-$NAME.jsonl"
)
rm -f "$OUT/autotune-log-$NAME.jsonl"
for v in TRITON_PRINT_AUTOTUNING TRITON_CACHE_AUTOTUNING PYTHONHASHSEED TORCH_LOGS \
         OMP_NUM_THREADS TRITON_CACHE_DIR; do
  [ -n "${!v:-}" ] && ENVARGS+=(-e "$v=${!v}")
done

nohup timeout "${S14_ENGINE_TIMEOUT:-7200}" docker exec -i \
  "${ENVARGS[@]}" \
  "$CT" bash -c "exec ${S14G_EXEC_PREFIX:-} bash -s -- $NAME $MODEL $TP $SW $PORT $MAXLEN $EXTRA" \
  < "$(dirname "$0")/s14g_arm_env.sh" >> "$ELOG" 2>&1 &
ENGINE_PID=$!
echo "[orch14g] engine_wrapper_pid=$ENGINE_PID"
echo "$ENGINE_PID" > "$LOGDIR/$NAME.wrapper.pid"

t0=$(date +%s)
READY=0
while true; do
  if grep -q "Application startup complete" "$ELOG"; then READY=1; break; fi
  if grep -qE "ConstraintViolationError|EngineCore failed to start|Engine core initialization failed|AcceleratorError|Traceback" "$ELOG"; then
    echo "[orch14g] FATAL marker in $ELOG"; break
  fi
  if [ $(( $(date +%s) - t0 )) -gt "${READY_TIMEOUT:-3600}" ]; then echo "[orch14g] TIMEOUT waiting for readiness"; break; fi
  sleep 10
done
READY_S=$(( $(date +%s) - t0 ))
echo "[orch14g] ready=$READY after ${READY_S}s $(date -Is)"

# ---- H1 instrument: address layout of the live engine --------------------
if [ "$READY" = "1" ]; then
  docker exec "$CT" bash /src/tools/dcu/s14g_maps.sh > "$OUT/maps-$NAME.txt" 2>&1 || true
  echo "[orch14g] maps fingerprint: $(grep -c '^== pid=' "$OUT/maps-$NAME.txt" || true) procs"
  grep '^== pid=' "$OUT/maps-$NAME.txt" | sed 's/^/[orch14g] /'
fi

if [ "$READY" = "1" ]; then
  IFS=',' read -ra TS <<< "${S14G_TASKS:-flip}"
  for t in "${TS[@]}"; do
    echo "[orch14g] task=$t start $(date -Is)"
    case "$t" in
      flip) docker exec "$CT" python "${S14_SRC:-/src}"/tools/dcu/s11l_flip_stats.py \
              --port "$PORT" --model "$SERVED" --label "$NAME" \
              --out "$OUT/flip-$NAME.jsonl" 2>&1 | tee "$LOGDIR/$NAME-flip.log" ;;
      e14)  docker exec "$CT" python "${S14_SRC:-/src}"/tools/dcu/s14e_client.py \
              --port "$PORT" --model "$SERVED" --label "$NAME" \
              --out "$OUT/probe-$NAME.jsonl" --max-tokens "${S14E_TOKENS:-128}" \
              2>&1 | tee "$LOGDIR/$NAME-e14.log" ;;
      *) echo "[orch14g] unknown task $t" ;;
    esac
    echo "[orch14g] task=$t done $(date -Is)"
  done
fi

# ---- stop ----------------------------------------------------------------
docker exec "$CT" bash -c "kill \$(cat /tmp/s14_${NAME}.pid) 2>/dev/null" || true
sleep 20
docker exec "$CT" bash -c "kill -9 \$(cat /tmp/s14_${NAME}.pid) 2>/dev/null" || true
kill "$ENGINE_PID" 2>/dev/null || true
sleep 5
echo "[orch14g] leftover procs: $(pgrep -c VLLM 2>/dev/null || echo 0)"

# ---- raw autotune evidence ----------------------------------------------
grep -E "Triton autotuning for function|best config selected|s14g-pin" "$ELOG" > "$OUT/autotune-$NAME.txt" || true
n_at=$(grep -c "best config selected" "$OUT/autotune-$NAME.txt" || true)
pinned=$(grep -c "s14g-pin" "$OUT/autotune-$NAME.txt" || true)
ALOG="$OUT/autotune-log-$NAME.jsonl"
hi=$(grep -c '"event": "import"' "$ALOG" 2>/dev/null || true)
he=$(grep -c '"event": "autotune"' "$ALOG" 2>/dev/null || true)
echo "[orch14g] hook(pid x import)=${hi:-0} hook(autotune events)=${he:-0} procs=$(grep -c '"event": "import"' "$ALOG" 2>/dev/null || echo 0)"
aot_load=$(grep -c "Directly load AOT compilation from path" "$ELOG" || true)
init_line=$(grep -oE "init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s \(compilation: [0-9.]+ s\)" "$ELOG" | head -1)
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$NAME" "$TP" "$SW" "$MAXLEN" "$DEVICES" "${S14_CG_MODE:-default}" "${S14_CG_SIZES:-default}" \
  "$READY" "$READY_S" "$aot_load" "$n_at" "$pinned" >> "$LOGDIR/summary.tsv"
echo "[orch14g] summary: ready=${READY_S}s aot_load=${aot_load} autotune_lines=${n_at} pinned_lines=${pinned} | ${init_line:-NA}"
echo "[orch14g] done name=$NAME ready=$READY"
