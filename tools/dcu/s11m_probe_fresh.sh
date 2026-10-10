#!/usr/bin/env bash
# S11m: cheap "fresh compile" probes (~3-5 min each, no full readiness).
#
#   tools/dcu/s11m_probe_fresh.sh <NAME> <CACHE_ROOT> [EXTRA serve args...]
#
# Starts the 27B W8A8 engine with the given cache root and stops as soon as the
# AOT decision is visible: either it loaded an existing artifact
# ("Directly load AOT compilation from path") or it compiled a fresh one
# ("saved AOT compiled function").  Used to compare artifacts produced by two
# independent fresh compiles of the very same configuration.
set -u
NAME=$1; CACHE=$2; shift 2
EXTRA="$*"
WS=/home/lyh/z100-vllm
CT=z100-s9
LOGDIR="$WS/.tmp-11m"
mkdir -p "$LOGDIR"
ELOG="$LOGDIR/$NAME.log"
: > "$ELOG"

docker exec $CT bash -c "mkdir -p '$CACHE'" || exit 1
nohup timeout 1500 docker exec -i \
  -e S11M_CACHE="$CACHE" \
  -e S11M_LOGLEVEL="${S11M_LOGLEVEL:-INFO}" \
  -e S11M_DEVICES="${S11M_DEVICES:-0,1}" \
  -e S11M_TRITON_CACHE="${S11M_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}" \
  -e VLLM_FORCE_AOT_LOAD="${S11M_FORCE_AOT_LOAD:-0}" \
  $CT bash -s -- "$NAME" 1 8061 4096 $EXTRA \
  < "$WS/tools/dcu/s11m_arm_env.sh" >> "$ELOG" 2>&1 &

t0=$(date +%s)
while true; do
  if grep -qE "saved AOT compiled function|Directly load AOT compilation from path" "$ELOG"; then break; fi
  if grep -qE "EngineCore failed to start|Traceback \(most recent call last\)" "$ELOG"; then echo "[probe] FATAL marker"; break; fi
  if [ $(( $(date +%s) - t0 )) -gt "${PROBE_TIMEOUT:-900}" ]; then echo "[probe] TIMEOUT"; break; fi
  sleep 5
done
echo "[probe] decision after $(( $(date +%s) - t0 ))s"
grep -E "Directly load AOT compilation from path|saved AOT compiled function to|Compiling model again|torch.compile took|benchmark_combo_kernel" "$ELOG" \
  | sed 's/^.*INFO //; s/^.*WARNING //' | head -8
docker exec $CT bash -c "kill \$(cat /tmp/s11m_${NAME}.pid) 2>/dev/null" || true
sleep 20
docker exec $CT bash -c "kill -9 \$(cat /tmp/s11m_${NAME}.pid) 2>/dev/null" || true
echo "[probe] done name=$NAME"
