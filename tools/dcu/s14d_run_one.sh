#!/usr/bin/env bash
# S14d: host-side orchestrator -- start one engine, run the deterministic
# token-stream probe, stop it, append a machine-readable row.
#
# Derived from tools/dcu/s14c_run_one.sh (md5 4b0ba4... at S14d start) with the
# memory tracer dropped (S14d does not measure memory) and the task set reduced
# to the two tasks S14d needs.  The container launcher is reused verbatim
# (tools/dcu/s14c_arm_env.sh) so the production flags stay byte-identical with
# the S14c recipe.
#
#   tools/dcu/s14d_run_one.sh <NAME> <MODEL_DIR> <TP> <SW> <CACHE> <PORT> <MAXLEN> <DEVICES> <TASKS> [EXTRA...]
#
# TASKS = comma separated subset of: probe,decode,conc2
# Rule: DEVICES is mandatory and explicit (never unset HIP_VISIBLE_DEVICES).
set -u
NAME=$1; MODEL=$2; TP=$3; SW=$4; CACHE=$5; PORT=$6; MAXLEN=$7; DEVICES=$8; TASKS=$9
shift 9
EXTRA="$*"
CT="${S14D_CT:-${S14C_CT:-${S14_CT:-z100-s14}}}"
LOGDIR="${S14D_LOGDIR:-/mnt/game/s14d/hostlogs}"
OUT="${S14D_BENCH_OUT:-/mnt/game/s14d}"
mkdir -p "$LOGDIR" "$OUT"
ELOG="$LOGDIR/$NAME.log"
: > "$ELOG"
SERVED="${S14_SERVED:-qwen3.8-27b}"
CAR="${S14_CAR:-1}"

echo "[orch14d] container=$CT mem_limit=$(docker inspect "$CT" --format '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}')"
echo "[orch14d] free_before: $(free -m | tr '\n' ' ')"
echo "[orch14d] swaps_before: $(cat /proc/swaps | tr '\n' '|')"
echo "[orch14d] swappiness=$(cat /proc/sys/vm/swappiness) acs_check=later"
echo "[orch14d] knobs: eager=${S14_EAGER:-0} cg_mode=${S14_CG_MODE:-default} cg_sizes=${S14_CG_SIZES:-default} mbt=${S14_MBT:-2048} util=${S14_UTIL:-0.85} maxnumseqs=${S14C_MAX_NUM_SEQS:-unset} inductor=${S14C_INDUCTOR_CFG:-unset} tag=${S14C_TAG:-none}"
docker exec "$CT" bash -c "mkdir -p $OUT '$CACHE' && rm -f $OUT/probe-${NAME}.jsonl /tmp/s14c_${NAME}.pid && cat /sys/fs/cgroup/memory.events > $OUT/cgroup-events-${NAME}-before.txt" || exit 1

CAR_ENV=()
if [ "$CAR" = "0" ]; then CAR_ENV=(-e S14_CAR=0); fi

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
  -e S14C_MALLOC_ARENA_MAX="${S14C_MALLOC_ARENA_MAX:-}" \
  -e S14C_MALLOC_TRIM_THRESHOLD="${S14C_MALLOC_TRIM_THRESHOLD:-}" \
  -e S14C_MALLOC_MMAP_THRESHOLD="${S14C_MALLOC_MMAP_THRESHOLD:-}" \
  -e S14C_HIP_ALLOC_CONF="${S14C_HIP_ALLOC_CONF:-}" \
  -e S14C_INDUCTOR_CFG="${S14C_INDUCTOR_CFG:-}" \
  -e S14C_MAX_NUM_SEQS="${S14C_MAX_NUM_SEQS:-}" \
  -e S14C_EXTRA_ARGS="${S14C_EXTRA_ARGS:-}" \
  -e S14C_TAG="${S14C_TAG:-}" \
  "${CAR_ENV[@]}" \
  "$CT" bash -s -- "$NAME" "$MODEL" "$TP" "$SW" "$PORT" "$MAXLEN" $EXTRA \
  < "$(dirname "$0")/s14c_arm_env.sh" >> "$ELOG" 2>&1 &
ENGINE_PID=$!
echo "[orch14d] engine_wrapper_pid=$ENGINE_PID"
echo "$ENGINE_PID" > "$LOGDIR/$NAME.wrapper.pid"

t0=$(date +%s)
READY=0
while true; do
  if grep -q "Application startup complete" "$ELOG"; then READY=1; break; fi
  if grep -qE "ConstraintViolationError|EngineCore failed to start|Engine core initialization failed|AcceleratorError|Traceback|Worker proc .* died" "$ELOG"; then
    echo "[orch14d] FATAL marker in $ELOG"; break
  fi
  if [ $(( $(date +%s) - t0 )) -gt "${READY_TIMEOUT:-3600}" ]; then echo "[orch14d] TIMEOUT waiting for readiness"; break; fi
  sleep 10
done
READY_S=$(( $(date +%s) - t0 ))
echo "[orch14d] ready=$READY after ${READY_S}s  $(date -Is)"
echo "[orch14d] free_after_start: $(free -m | tr '\n' ' ')"
echo "[orch14d] swaps_after_start: $(cat /proc/swaps | tr '\n' '|')"
docker exec "$CT" bash -c "cat /sys/fs/cgroup/memory.events; echo '--- current:'; cat /sys/fs/cgroup/memory.current" 2>&1 | tee "$LOGDIR/$NAME-cgroup-after-start.txt"

if [ "$READY" = "1" ]; then
  IFS=',' read -ra TS <<< "$TASKS"
  for t in "${TS[@]}"; do
    echo "[orch14d] task=$t start $(date -Is)"
    case "$t" in
      probe) docker exec "$CT" python "${S14_SRC:-/src}"/tools/dcu/s14d_client.py --port "$PORT" --model "$SERVED" --label "$NAME" --out "$OUT/probe-$NAME.jsonl" --max-tokens "${S14D_PROBE_TOKENS:-128}" 2>&1 | tee "$LOGDIR/$NAME-probe.log" ;;
      decode) docker exec -e S14_BENCH_OUT="$OUT" "$CT" bash "${S14_SRC:-/src}"/tools/dcu/s14_bench.sh "$PORT" "$NAME" decode "$SERVED" "$MODEL" 2>&1 | tee "$LOGDIR/$NAME-bench-decode.log" ;;
      conc2) docker exec "$CT" python "${S14_SRC:-/src}"/tools/dcu/s14d_conc2.py --port "$PORT" --model "$SERVED" --label "$NAME" --out "$OUT/conc2-$NAME.jsonl" 2>&1 | tee "$LOGDIR/$NAME-conc2.log" ;;
      *) echo "[orch14d] unknown task $t" ;;
    esac
    echo "[orch14d] task=$t done rc=$? $(date -Is)"
  done
fi

if [ "${S14D_HOLD_S:-0}" -gt 0 ]; then
  echo "[orch14d] holding engine for ${S14D_HOLD_S}s at $(date -Is)"
  sleep "${S14D_HOLD_S}"
fi

# ---- stop ----------------------------------------------------------------
docker exec "$CT" bash -c "kill \$(cat /tmp/s14c_${NAME}.pid) 2>/dev/null" || true
sleep 20
docker exec "$CT" bash -c "kill -9 \$(cat /tmp/s14c_${NAME}.pid) 2>/dev/null" || true
kill "$ENGINE_PID" 2>/dev/null || true
sleep 25
echo "[orch14d] free_after_stop: $(free -m | tr '\n' ' ')"
echo "[orch14d] swaps_after_stop: $(cat /proc/swaps | tr '\n' '|')"
pkill VLLM 2>/dev/null || true
sleep 5
echo "[orch14d] leftover live VLLM: $(for p in $(pgrep VLLM 2>/dev/null); do st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" = Z ] || echo $p; done | wc -l)"
docker exec "$CT" bash -c "cat /sys/fs/cgroup/memory.events" | tee "$OUT/cgroup-events-${NAME}-after.txt"

# ---- machine-readable summary -------------------------------------------
aot_load=$(grep -c "Directly load AOT compilation from path" "$ELOG" || true)
aot_recompile=$(grep -c "Compiling model again due to a load failure" "$ELOG" || true)
init_line=$(grep -oE "init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s \(compilation: [0-9.]+ s\)" "$ELOG" | head -1)
kv=$(grep -oE "Available KV cache memory: [0-9.]+ GiB" "$ELOG" | head -1)
oomk=$(docker exec "$CT" bash -c "cat /sys/fs/cgroup/memory.events" | awk '/^oom_kill/{print $2}')
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$NAME" "$MODEL" "$TP" "$MAXLEN" "$DEVICES" "$READY" "$READY_S" \
  "${aot_load}" "${aot_recompile}" "${oomk:-NA}" "$(date -Is)" "${S14C_TAG:-none}" >> "$LOGDIR/summary.tsv"
echo "[orch14d] summary: ready=${READY_S}s aot_load=${aot_load} aot_recompile=${aot_recompile} oom_kill=${oomk:-NA} | ${init_line:-NA} | ${kv:-NA}"
echo "[orch14d] done name=$NAME ready=$READY"
