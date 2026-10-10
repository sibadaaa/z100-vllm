#!/usr/bin/env bash
# S14c: host-side orchestrator -- start one measured engine, trace host memory
# for the whole arm, run client tasks, stop it, append a machine-readable row.
#
# Derived from tools/dcu/s14_run_one.sh (md5 e09d46bb…) so the arm semantics stay
# comparable with S14; S14c differences:
#   * container launcher is s14c_arm_env.sh (S14 knobs + MALLOC_ARENA_MAX /
#     inductor_compile_config / max-num-seqs / HIP alloc conf),
#   * a high-resolution memtrace (tools/dcu/s14c_memtrace.sh) is started before
#     the engine and runs for the whole arm (G1),
#   * artifacts default to /mnt/game/s14c,
#   * S14C_* env knobs are forwarded into the container.
#
#   tools/dcu/s14c_run_one.sh <NAME> <MODEL_DIR> <TP> <SW> <CACHE> <PORT> <MAXLEN> <DEVICES> <TASKS> [EXTRA...]
#
# TASKS = comma separated subset of: flip,decode,ttft,ttft8k,conc8,greedy
# Rule: DEVICES is mandatory and explicit (never unset HIP_VISIBLE_DEVICES).
set -u
NAME=$1; MODEL=$2; TP=$3; SW=$4; CACHE=$5; PORT=$6; MAXLEN=$7; DEVICES=$8; TASKS=$9
shift 9
EXTRA="$*"
CT="${S14C_CT:-${S14_CT:-z100-s14}}"
LOGDIR="${S14C_LOGDIR:-/mnt/game/s14c/hostlogs}"
OUT="${S14C_BENCH_OUT:-/mnt/game/s14c}"
mkdir -p "$LOGDIR" "$OUT"
ELOG="$LOGDIR/$NAME.log"
: > "$ELOG"
SERVED="${S14_SERVED:-qwen3.8-27b}"
CAR="${S14_CAR:-1}"

echo "[orch14c] container=$CT mem_limit=$(docker inspect "$CT" --format '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}')"
echo "[orch14c] free_before: $(free -m | tr '\n' ' ')"
echo "[orch14c] swaps_before: $(cat /proc/swaps | tr '\n' '|')"
echo "[orch14c] knobs: MALLOC_ARENA_MAX=${S14C_MALLOC_ARENA_MAX:-unset} hip_alloc=${S14C_HIP_ALLOC_CONF:-default} inductor=${S14C_INDUCTOR_CFG:-unset} mbt=${S14_MBT:-2048} seqs=${S14C_MAX_NUM_SEQS:-unset} cg_mode=${S14_CG_MODE:-default} cg_sizes=${S14_CG_SIZES:-default} eager=${S14_EAGER:-0} tag=${S14C_TAG:-none}"
docker exec "$CT" bash -c "mkdir -p $OUT '$CACHE' && rm -f $OUT/flip-${NAME}.jsonl /tmp/s14c_${NAME}.pid && cat /sys/fs/cgroup/memory.events > $OUT/cgroup-events-${NAME}-before.txt" || exit 1

# ---- G1: high-resolution host memory tracer for the whole arm ----
MT_IV="${S14C_MEMTRACE_IV:-5}"
MT_DUR="${S14C_MEMTRACE_DUR:-2400}"
nohup bash "$(dirname "$0")/s14c_memtrace.sh" "$NAME" "$MT_DUR" "$MT_IV" "$ELOG" \
  > "$LOGDIR/$NAME-memtrace.out" 2>&1 &
MT_PID=$!
echo "[orch14c] memtrace_pid=$MT_PID interval=${MT_IV}s dur=${MT_DUR}s"

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
echo "[orch14c] engine_wrapper_pid=$ENGINE_PID"
echo "$ENGINE_PID" > "$LOGDIR/$NAME.wrapper.pid"

t0=$(date +%s)
READY=0
while true; do
  if grep -q "Application startup complete" "$ELOG"; then READY=1; break; fi
  if grep -qE "ConstraintViolationError|EngineCore failed to start|Engine core initialization failed|AcceleratorError|Traceback|Worker proc .* died" "$ELOG"; then
    echo "[orch14c] FATAL marker in $ELOG"; break
  fi
  if [ $(( $(date +%s) - t0 )) -gt "${READY_TIMEOUT:-3600}" ]; then echo "[orch14c] TIMEOUT waiting for readiness"; break; fi
  sleep 10
done
READY_S=$(( $(date +%s) - t0 ))
echo "[orch14c] ready=$READY after ${READY_S}s  $(date -Is)"
echo "[orch14c] free_after_start: $(free -m | tr '\n' ' ')"
echo "[orch14c] swaps_after_start: $(cat /proc/swaps | tr '\n' '|')"
docker exec "$CT" bash -c "cat /sys/fs/cgroup/memory.events; echo '--- current:'; cat /sys/fs/cgroup/memory.current" 2>&1 | tee "$LOGDIR/$NAME-cgroup-after-start.txt"

if [ "$READY" = "1" ]; then
  IFS=',' read -ra TS <<< "$TASKS"
  for t in "${TS[@]}"; do
    echo "[orch14c] task=$t start $(date -Is)"
    case "$t" in
      flip)   docker exec "$CT" python "${S14_SRC:-/src}"/tools/dcu/s11l_flip_stats.py --port "$PORT" --model "$SERVED" --label "$NAME" --out "$OUT/flip-$NAME.jsonl" --max-tokens "${S14_FLIP_TOKENS:-128}" 2>&1 | tee "$LOGDIR/$NAME-flip.log" ;;
      greedy) docker exec "$CT" python "${S14_SRC:-/src}"/tools/dcu/s14_greedy.py --port "$PORT" --model "$SERVED" --label "$NAME" --out "$OUT/greedy-$NAME.jsonl" --max-tokens "${S14_GREEDY_TOKENS:-256}" 2>&1 | tee "$LOGDIR/$NAME-greedy.log" ;;
      decode|ttft|ttft8k|conc8)
              docker exec -e S14_BENCH_OUT="$OUT" "$CT" bash "${S14_SRC:-/src}"/tools/dcu/s14_bench.sh "$PORT" "$NAME" "$t" "$SERVED" "$MODEL" 2>&1 | tee "$LOGDIR/$NAME-bench-$t.log" ;;
      smaps)  bash "${S14_SRC:-/src}/tools/dcu/s14_smaps.sh" "$NAME" 2>&1 | tee "$LOGDIR/$NAME-smaps.log" ;;
      *) echo "[orch14c] unknown task $t" ;;
    esac
    echo "[orch14c] task=$t done rc=$? $(date -Is)"
  done
fi

# ---- steady-state window (optional): keep the engine up so the memtrace
# ---- captures the post-bench steady state, not just the startup transient.
if [ "${S14C_HOLD_S:-0}" -gt 0 ]; then
  echo "[orch14c] holding engine for ${S14C_HOLD_S}s at $(date -Is)"
  sleep "${S14C_HOLD_S}"
fi

# ---- stop ----------------------------------------------------------------
docker exec "$CT" bash -c "kill \$(cat /tmp/s14c_${NAME}.pid) 2>/dev/null" || true
sleep 20
docker exec "$CT" bash -c "kill -9 \$(cat /tmp/s14c_${NAME}.pid) 2>/dev/null" || true
kill "$ENGINE_PID" 2>/dev/null || true
sleep 25
# capture the post-shutdown host state before stopping the tracer
echo "[orch14c] free_after_stop: $(free -m | tr '\n' ' ')"
echo "[orch14c] swaps_after_stop: $(cat /proc/swaps | tr '\n' '|')"
kill "$MT_PID" 2>/dev/null || true
sleep 1
kill -9 "$MT_PID" 2>/dev/null || true
pkill VLLM 2>/dev/null || true
sleep 5
echo "[orch14c] leftover live VLLM: $(for p in $(pgrep VLLM 2>/dev/null); do st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" = Z ] || echo $p; done | wc -l)"
docker exec "$CT" bash -c "cat /sys/fs/cgroup/memory.events" | tee "$OUT/cgroup-events-${NAME}-after.txt"

# ---- machine-readable summary -------------------------------------------
aot_load=$(grep -c "Directly load AOT compilation from path" "$ELOG" || true)
aot_recompile=$(grep -c "Compiling model again due to a load failure" "$ELOG" || true)
init_line=$(grep -oE "init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s \(compilation: [0-9.]+ s\)" "$ELOG" | head -1)
kv=$(grep -oE "Available KV cache memory: [0-9.]+ GiB" "$ELOG" | head -1)
oomk=$(docker exec "$CT" bash -c "cat /sys/fs/cgroup/memory.events" | awk '/^oom_kill/{print $2}')
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$NAME" "$MODEL" "$TP" "$SW" "$MAXLEN" "$DEVICES" "$READY" "$READY_S" \
  "${aot_load}" "${aot_recompile}" "${oomk:-NA}" "$(date -Is)" "${S14C_TAG:-none}" >> "$LOGDIR/summary.tsv"
echo "[orch14c] summary: ready=${READY_S}s aot_load=${aot_load} aot_recompile=${aot_recompile} oom_kill=${oomk:-NA} | ${init_line:-NA} | ${kv:-NA}"
echo "[orch14c] done name=$NAME ready=$READY"
