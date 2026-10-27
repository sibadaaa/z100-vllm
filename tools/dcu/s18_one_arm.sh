#!/usr/bin/env bash
# S18: run ONE arm end to end inside the experiment container and leave evidence.
#
#   docker exec -d z100-s18 bash /src/tools/dcu/s18_one_arm.sh <SLOT> <TP> <K> <MAXLEN> [LABEL_SUFFIX]
#
# Steps: kill any live engine -> start arm -> wait /health -> MTP probe ->
#        S9-protocol decode bench -> (optional ttft) -> kill engine -> summary.
#
# Env: S18_TOP (default /mnt/game/s18), S18_PROBE_MODE (greedy|prod),
#      S18_BENCH_MODES (default "decode"), S18_START_TIMEOUT (default 1200s),
#      plus everything s18_mtp_arm.sh understands (S18_UTIL, S18_MBT, ...).
set -u
SLOT=${1:?usage: s18_one_arm.sh <fp8|int8> <TP> <K> <MAXLEN> [suffix]}
TP=${2:?usage: s18_one_arm.sh <fp8|int8> <TP> <K> <MAXLEN> [suffix]}
K=${3:?usage: s18_one_arm.sh <fp8|int8> <TP> <K> <MAXLEN> [suffix]}
MAXLEN=${4:?usage: s18_one_arm.sh <fp8|int8> <TP> <K> <MAXLEN> [suffix]}
SUFFIX=${5:-}
LABEL="${SLOT}-tp${TP}-k${K}-m${MAXLEN}${SUFFIX}"
TOP="${S18_TOP:-/mnt/game/s18}"
PORT="${S18_PORT:-8001}"
mkdir -p "$TOP"
SUMMARY="$TOP/summary-${LABEL}.md"
PIDFILE="/tmp/s18_${LABEL}.pid"
SERVE_LOG="$TOP/serve-${LABEL}.log"

. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache

kill_engine() {
  echo "  [kill] terminating engine for $LABEL"
  [ -f "$PIDFILE" ] && kill -9 "$(cat "$PIDFILE")" 2>/dev/null
  pkill -9 -f 'entrypoints.cli.main' 2>/dev/null
  pkill -9 -f 'VLLM::' 2>/dev/null
  pkill -9 VLLM 2>/dev/null
  sleep 4
  for _ in $(seq 1 30); do
    pgrep -f 'entrypoints.cli.main' >/dev/null || break
    sleep 2
  done
  echo "  [kill] remaining: $(pgrep -c -f 'entrypoints.cli.main' 2>/dev/null || echo 0) api / $(pgrep -c VLLM 2>/dev/null || echo 0) workers"
}

echo "=== S18 arm $LABEL start $(date -Is)"
kill_engine

# The arm writes its own S18_LOG + args file; keep them per-label.
S18_LOG="$SERVE_LOG" S18_OUT="$TOP" S18_MAXLEN="$MAXLEN" \
  nohup bash /src/tools/dcu/s18_mtp_arm.sh "$SLOT" "$TP" "$PORT" "$K" \
  >"$TOP/arm-${LABEL}.out" 2>&1 &
T0=$(date +%s)

echo "  [wait] /health on :$PORT (timeout ${S18_START_TIMEOUT:-1200}s)"
# Grace period: bash + DTK env + python bootstrap take a few seconds; do not
# mistake "not started yet" for "exited early".
for _ in $(seq 1 30); do
  pgrep -f 'entrypoints.cli.main' >/dev/null && break
  sleep 2
done
READY=0
while [ $(( $(date +%s) - T0 )) -lt "${S18_START_TIMEOUT:-1200}" ]; do
  if pgrep -f 'entrypoints.cli.main' >/dev/null; then
    if curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then READY=1; break; fi
  else
    echo "  [wait] engine process exited early"; break
  fi
  sleep 5
done
STARTUP=$(( $(date +%s) - T0 ))
echo "  [wait] ready=$READY startup=${STARTUP}s"

{
  echo "# S18 arm: $LABEL"
  echo
  echo "- date: $(date -Is)"
  echo "- slot=$SLOT tp=$TP k=$K maxlen=$MAXLEN port=$PORT"
  echo "- ready=$READY startup_seconds=$STARTUP"
  echo "- serve log: $SERVE_LOG"
  echo
} > "$SUMMARY"

if [ "$READY" = "1" ]; then
  if [ "${S18_FLIP:-0}" = "1" ]; then
    echo "  [flip] recording greedy token/logprob battery (K2 parity)"
    python /src/tools/dcu/s11l_flip_stats.py --port "$PORT" --label "$LABEL" \
      --out "$TOP/flip-${LABEL}.jsonl" >"$TOP/flip-${LABEL}.out" 2>&1
    echo "  [flip] rc=$? -> $TOP/flip-${LABEL}.jsonl"
    {
      echo "## K2 flip stats"
      tail -3 "$TOP/flip-${LABEL}.out"
    } >> "$SUMMARY"
  fi
  for PMODE in ${S18_PROBE_MODES:-greedy}; do
    echo "  [probe] acceptance (mode=$PMODE)"
    python /src/tools/dcu/s18_mtp_probe.py --port "$PORT" --label "$LABEL" \
      --mode "$PMODE" --max-tokens "${S18_PROBE_TOKENS:-192}" \
      --outdir "$TOP" >"$TOP/probe-${LABEL}-${PMODE}.out" 2>&1
    echo "  [probe] rc=$? mode=$PMODE"
    {
      echo "## probe ($PMODE)"
      tail -8 "$TOP/probe-${LABEL}-${PMODE}.out"
    } >> "$SUMMARY"
  done

  for MODE in ${S18_BENCH_MODES:-decode}; do
    echo "  [bench] $MODE"
    bash /src/tools/dcu/s18_bench.sh "$PORT" "$LABEL" "$MODE" >/dev/null 2>&1
    echo "  [bench] rc=$? mode=$MODE"
    grep -E "Median (TTFT|TPOT|E2EL)|Output token throughput|Mean (TTFT|TPOT)" \
      "$TOP/bench-${LABEL}-${MODE}.log" >> "$SUMMARY" 2>/dev/null || true
  done
else
  echo "- ARM FAILED TO START (see $SERVE_LOG)" >> "$SUMMARY"
  tail -30 "$SERVE_LOG" >> "$SUMMARY" 2>/dev/null || true
fi

kill_engine
echo "  [done] summary: $SUMMARY"
echo "S18_ARM_DONE label=$LABEL ready=$READY startup=$STARTUP"
