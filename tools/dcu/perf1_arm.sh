#!/usr/bin/env bash
# perf1 A/B arm driver for the :8000 vLLM service (run INSIDE the container).
#
#   docker exec z100-vllm bash /src/tools/dcu/perf1_arm.sh snapshot <name>
#   docker exec z100-vllm bash /src/tools/dcu/perf1_arm.sh stop
#   docker exec z100-vllm bash /src/tools/dcu/perf1_arm.sh start <name> [KEY=VAL ...]
#   docker exec z100-vllm bash /src/tools/dcu/perf1_arm.sh status
#
# The control arm is /mnt/game/port/serve_fp8_8000.sh (production recipe, NO
# --profiler-config).  Override the recipe for another arm with
# PERF1_RECIPE=<script>.  KEY=VAL pairs are exported before the recipe runs and
# are restricted to the knobs the recipes read (S11F_*, VLLM_DCU_*,
# VLLM_GFX906_*, VLLM_QWEN35_*, HSA_*, HIP_VISIBLE_DEVICES).
#
# Everything is logged to /mnt/game/perf1/serve_<name>.log, and the exact
# command line + environment of each arm is saved next to it.
set -u

PERF1_DIR="${PERF1_DIR:-/mnt/game/perf1}"
RECIPE="${PERF1_RECIPE:-/mnt/game/port/serve_fp8_8000.sh}"
PORT="${PERF1_PORT:-8000}"
SVC_PAT='vllm.entrypoints.cli.main'

log() { echo "[perf1-arm $(date -Is)] $*"; }

so_md5() { md5sum /src/vllm/_rocm_C.abi3.so 2>/dev/null | cut -d' ' -f1; }

svc_pid() {
  # newest live (non-zombie) service process
  local p
  for p in $(pgrep -f "$SVC_PAT" 2>/dev/null | sort -n -r); do
    [ -r "/proc/$p/stat" ] || continue
    local st
    st=$(cut -d' ' -f3 "/proc/$p/stat" 2>/dev/null)
    [ "$st" = "Z" ] && continue
    echo "$p"; return 0
  done
  return 1
}

snapshot() {
  local name="${1:-current}" pid
  mkdir -p "$PERF1_DIR"
  pid=$(svc_pid) || { log "snapshot $name: no live service"; return 1; }
  tr '\0' ' ' < "/proc/$pid/cmdline" > "$PERF1_DIR/arm_${name}_cmdline.txt"
  echo >> "$PERF1_DIR/arm_${name}_cmdline.txt"
  tr '\0' '\n' < "/proc/$pid/environ" | sort > "$PERF1_DIR/arm_${name}_environ.txt"
  echo "$pid" > "$PERF1_DIR/arm_${name}_pid.txt"
  log "snapshot $name: pid=$pid -> $PERF1_DIR/arm_${name}_{cmdline,environ}.txt"
  grep -E "S11F|VLLM_DCU|VLLM_GFX906|VLLM_QWEN35|HSA_|HIP_VISIBLE" \
    "$PERF1_DIR/arm_${name}_environ.txt" || true
}

port_up() { curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; }

wait_ready() {
  local timeout="${1:-180}" t0
  t0=$(date +%s)
  while :; do
    if port_up; then
      log "ready after $(( $(date +%s) - t0 ))s: $(curl -sf -m 5 http://127.0.0.1:${PORT}/v1/models | head -c 160)"
      return 0
    fi
    if [ $(( $(date +%s) - t0 )) -ge "$timeout" ]; then
      log "TIMEOUT after ${timeout}s waiting for /v1/models"
      return 1
    fi
    sleep 3
  done
}

wait_free() {
  local timeout="${1:-90}" t0
  t0=$(date +%s)
  while :; do
    if ! svc_pid >/dev/null && ! port_up; then
      log "port ${PORT} free after $(( $(date +%s) - t0 ))s"
      return 0
    fi
    if [ $(( $(date +%s) - t0 )) -ge "$timeout" ]; then
      log "WARN: still busy after ${timeout}s (pid=$(svc_pid || echo none))"
      return 1
    fi
    sleep 2
  done
}

stop_svc() {
  local pid
  if ! pid=$(svc_pid); then log "stop: no live service"; return 0; fi
  log "stop: pid=$pid pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')"
  local pgid
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  if [ -n "$pgid" ] && [ "$pgid" != "1" ]; then
    kill -TERM "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  else
    kill -TERM "$pid" 2>/dev/null
  fi
  local t0; t0=$(date +%s)
  while svc_pid >/dev/null; do
    [ $(( $(date +%s) - t0 )) -ge 45 ] && break
    sleep 2
  done
  if pid=$(svc_pid); then
    log "stop: still alive, SIGKILL pid=$pid"
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    [ -n "$pgid" ] && [ "$pgid" != "1" ] && kill -KILL "-$pgid" 2>/dev/null
    kill -KILL "$pid" 2>/dev/null
    sleep 3
  fi
  # any orphan worker still holding the port
  pkill -f "multiprocessing.spawn" 2>/dev/null || true
  wait_free 60 || true
}

start_arm() {
  local name="$1"; shift
  local key val
  for kv in "$@"; do
    key="${kv%%=*}"; val="${kv#*=}"
    case "$key" in
      S11F_*|VLLM_DCU_*|VLLM_GFX906_*|VLLM_QWEN35_*|HSA_*|HIP_VISIBLE_DEVICES|PERF1_*) ;;
      *) log "REFUSE env override '$key' (not in the allowed knob set)"; return 2 ;;
    esac
    export "$key=$val"
  done
  mkdir -p "$PERF1_DIR"
  local stamp logfile
  stamp=$(date +%Y%m%d-%H%M%S)
  logfile="$PERF1_DIR/serve_${name}.log"
  export S11F_LOG="$logfile"
  export PERF1_ARM="$name"

  if [ ! -f "$RECIPE" ]; then log "recipe not found: $RECIPE"; return 2; fi
  stop_svc

  log "start arm=$name recipe=$RECIPE log=$logfile overrides: $*"
  {
    echo "=== arm=$name stamp=$stamp recipe=$RECIPE overrides: $* ==="
    echo "so_md5_start=$(so_md5)"
    for kv in "$@"; do echo "export $kv"; done
  } > "$PERF1_DIR/arm_${name}_meta.txt"

  # Close inherited fds 3-9: the caller holds /tmp/perf1_build.lock on fd 3, and
  # a detached service that inherits it would keep the lock forever (observed:
  # an orphaned arm blocked the next session's flock).
  setsid nohup bash "$RECIPE" > "$logfile" 2>&1 < /dev/null 3>&- 4>&- 5>&- 6>&- 7>&- 8>&- 9>&- &
  local launcher=$!
  disown "$launcher" 2>/dev/null || true
  log "launcher pid=$launcher (detached)"

  if wait_ready "${PERF1_READY_TIMEOUT:-420}"; then
    sleep 2
    # Fail-fast smoke gate: one 1-token chat request.  A broken tree (e.g. the
    # 2026-10-13 UnboundLocalError in the GDN op) serves HTTP 500 here, so the
    # arm aborts in ~1 min instead of after a full measurement, and the "live"
    # marker the suite greps for is never printed.
    if [ "${PERF1_SMOKE:-1}" = "1" ]; then
      local code
      code=$(curl -s -o /tmp/perf1_smoke_${name}.json -w '%{http_code}' -m 300 \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${S11F_SERVED:-qwen3.8-27b}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
        "http://127.0.0.1:${PORT}/v1/chat/completions")
      log "smoke arm=$name chat/completions HTTP $code"
      if [ "$code" != "200" ]; then
        log "arm $name SMOKE FAILED (HTTP $code) -- refusing to measure; last 25 log lines:"
        tail -25 "$logfile" || true
        return 1
      fi
    fi
    snapshot "$name"
    log "arm $name live"
    return 0
  fi
  log "arm $name FAILED to become ready; last 40 log lines:"
  tail -40 "$logfile" || true
  return 1
}

case "${1:-}" in
  snapshot) shift; snapshot "${1:-current}" ;;
  stop)     stop_svc ;;
  start)    shift; name="${1:?usage: start <name> [KEY=VAL ...]}"; shift; start_arm "$name" "$@" ;;
  ready)    wait_ready "${2:-180}" ;;
  so)       echo "so_md5=$(so_md5) $(readlink -f /src/vllm/_rocm_C.abi3.so)" ;;
  status)
    if pid=$(svc_pid); then
      log "service pid=$pid up=$(port_up && echo yes || echo no)"
      tr '\0' ' ' < "/proc/$pid/cmdline" | tr -s ' ' | head -c 1200; echo
    else
      log "no service process; port_up=$(port_up && echo yes || echo no)"
    fi ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
