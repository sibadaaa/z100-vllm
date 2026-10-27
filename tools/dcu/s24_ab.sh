#!/usr/bin/env bash
# S24 A/B driver (long-context + VRAM plan, docs/design/z100-longctx-vram-optimization-plan.md).
# Same restart/restore discipline as tools/dcu/s23_ab.sh, adapted for S24:
#
#   * windows live under .s24/w<N>/<ID>/<window-tag>/, arms in tools/dcu/s24_arms/
#   * baseline .so is the S23 delivery (d209af9585aaf0049394ced4a3c3211b)
#   * per arm: smoke + longctx probes (TTFT/prefill/ITL, per-rep distinct
#     prompts so prefix caching cannot eat the A/B) + optional greedy parity
#   * per arm: s24_logscan extracts the KV-pool / memory facts for L5
#
#   bash tools/dcu/s24_ab.sh [options] <wN> <ID> <window-tag> <arm> [<arm> ...]
#
# Options
#   --lens 512,8192,32768   longctx prompt lengths (default)
#   --reps N                longctx repeats per lens (default 4)
#   --gen N                 decode tokens per probe (default 16)
#   --no-parity             skip greedy parity capture/check
#   --no-restore            keep last arm env (default: restore + verify)
#   --soak LENS,LENS        extra soak lens run once per arm (default none)
#   --dry-run               preflight + plan only
#   --force                 reuse an existing window directory
set -u

W=/home/lyh/z100-vllm
LOCK=$W/.s23/gpu.lock          # shared with S23: one lock for all GPU windows
ARMS=$W/tools/dcu/s24_arms
C=z100-regress
UNIT=z100-vllm-host.service
LOGFILE=/mnt/game/host-serve/logs/serve-host.log
KEYFILE=/mnt/game/host-serve/api-key
BASE=${S24_BASE:-http://127.0.0.1:8000}
PY=$W/.venv/bin/python
SO_TGT=/mnt/game/port/build-sync/temp/_rocm_C.abi3.so
SO_BACKUP=/mnt/game/perf1
BASELINE_SO_MD5=${S24_BASELINE_SO_MD5:-d209af9585aaf0049394ced4a3c3211b}
LAUNCHER_DEFAULT=$W/tools/dcu/serve_host_dflash2.sh
UNIT_WORKDIR=$W
UNIT_STDOUT=append:$LOGFILE
UNIT_STDERR=append:$LOGFILE
UNIT_TIMEOUT_STOP=30
UNIT_ENV_HOMEPAGE=HOST=0.0.0.0
UNIT_PROPS_SHOWN="Type Restart RestartUSec TimeoutStopUSec WorkingDirectory
StandardOutput StandardError CollectMode ExecStart Environment"

log() { echo "[s24 $(date +%H:%M:%S)] $*"; }
die() { log "FATAL: $*"; exit 1; }

if [ "${S24_LOCK_HELD:-0}" != "1" ]; then
  [ -f "$LOCK" ] || : > "$LOCK"
  export S24_LOCK_HELD=1
  log "waiting for GPU lock $LOCK (window '$*')"
  exec flock -w "${S24_LOCK_WAIT:-3600}" "$LOCK" bash "$0" "$@"
fi

LENS=${LENS:-"512,8192,32768"}
REPS=4
GEN=16
DO_PARITY=1
DO_RESTORE=1
DRY_RUN=0
FORCE=0
SOAK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --lens) LENS=$2; shift 2 ;;
    --reps) REPS=$2; shift 2 ;;
    --gen) GEN=$2; shift 2 ;;
    --no-parity) DO_PARITY=0; shift ;;
    --no-restore) DO_RESTORE=0; shift ;;
    --soak) SOAK=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) break ;;
  esac
done
WN=${1:?usage: s24_ab.sh [opts] <wN> <ID> <window-tag> <arm>...}; shift
ID=${1:?}; shift
WINDOW_TAG=${1:?}; shift
[ $# -ge 1 ] || die "need at least one arm (see $ARMS)"
ARMS_REQ=("$@")

WINDOW_DIR=$W/.s24/$WN/$ID/$WINDOW_TAG
[ -e "$WINDOW_DIR" ] && [ "$FORCE" != "1" ] \
  && die "window dir $WINDOW_DIR exists (use --force to append)"
mkdir -p "$WINDOW_DIR"
WINDOW_START=$(date -Is)
PARITY_REF=""
RESTORED=0
INVALID=""
WINDOW_FP=""
WINDOW_SO=""

api_get() { curl -s -m "${2:-8}" -H "Authorization: Bearer $(cat "$KEYFILE")" "$1"; }
api_code() {
  curl -s -o /dev/null -m "${2:-5}" -w '%{http_code}' \
    -H "Authorization: Bearer $(cat "$KEYFILE")" "$1"
}
service_pid() { systemctl --user show -p MainPID --value "$UNIT"; }
so_md5_live() { docker exec $C bash -lc "md5sum '$SO_TGT'" 2>/dev/null | cut -d' ' -f1; }

wait_ready() {
  local tries=${1:-200} i code ok=0 dead=0
  for i in $(seq 1 "$tries"); do
    code=$(api_code "$BASE/v1/models" 3)
    if [ "$code" = "200" ]; then
      ok=$((ok + 1))
      [ "$ok" -ge 2 ] && { sleep 3; return 0; }
      sleep 3
      continue
    fi
    ok=0
    if systemctl --user is-active --quiet "$UNIT"; then
      dead=0
    else
      dead=$((dead + 1))
      [ "$dead" -ge 3 ] && return 1
    fi
    sleep 3
  done
  return 1
}

show_new_log() {
  local from=${1:-0}
  [ -f "$LOGFILE" ] || return 0
  tail -n +"$((from + 1))" "$LOGFILE" | grep -iE "error|traceback|fail|oom|abort" \
    | tail -n 15
}

changed_py() {
  git -C "$W" status --porcelain -uall | sed 's/^...//' \
    | grep -E '^(vllm|csrc)/.*\.py$' || true
}

tree_fingerprint() {
  {
    git -C "$W" rev-parse HEAD
    changed_py | while read -r p; do
      [ -f "$W/$p" ] && md5sum "$W/$p"
    done
  } | md5sum | cut -c1-16
}

recent_edit_guard() {
  local hits
  for _ in 1 2; do
    hits=$(find "$W/vllm" "$W/csrc" -name '*.py' -newermt '-90 seconds' 2>/dev/null | head -5)
    [ -z "$hits" ] && return 0
    log "WARN python files edited <90s ago: $(echo "$hits" | tr '\n' ' ')"
    sleep 20
  done
  return 1
}

preflight() {
  local dir=$1 tag=$2 rc=0 f
  log "gate 0: recent-edit guard"
  recent_edit_guard || die "a teammate is editing python the service imports; not restarting"
  log "gate 1: py_compile of changed python files"
  : > "$dir/preflight.log"
  for f in $(changed_py); do
    [ -f "$W/$f" ] || continue
    if "$PY" -m py_compile "$W/$f" >> "$dir/preflight.log" 2>&1; then
      echo "OK   $f" >> "$dir/preflight.log"
    else
      echo "FAIL $f" >> "$dir/preflight.log"
      rc=1
    fi
  done
  echo "tree_fingerprint=$(tree_fingerprint)" >> "$dir/preflight.log"
  echo "git_rev=$(git -C "$W" rev-parse --short HEAD)" >> "$dir/preflight.log"
  [ "$rc" != "0" ] && { tail -5 "$dir/preflight.log"; return 1; }

  log "gate 2: container import smoke ($tag)"
  docker exec $C bash -c "cd /src && env ${ARM_ENV:-} \
    DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
    bash -c 'set +u; source tools/dcu/env.sh >/dev/null 2>&1; \
      export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; \
      export HIP_VISIBLE_DEVICES=3; \
      .venv/bin/python tools/dcu/s23_preflight.py'" \
    >> "$dir/preflight.log" 2>&1 || rc=1
  tail -n 4 "$dir/preflight.log"
  return "$rc"
}

clear_manager_env() {
  systemctl --user unset-environment VLLM_DCU_FP8_EMU_FASTPATH \
    VLLM_DCU_FP8_QUANT_FASTPATH VLLM_DCU_FP8_FUSE_QUANT 2>/dev/null || true
  return 0
}

recreate_unit() {
  local launcher=$1 envpairs=${2:-}
  local args=(
    --property=WorkingDirectory="$UNIT_WORKDIR"
    --property=StandardOutput="$UNIT_STDOUT"
    --property=StandardError="$UNIT_STDERR"
    --property=TimeoutStopSec="$UNIT_TIMEOUT_STOP"
    --setenv="$UNIT_ENV_HOMEPAGE"
  )
  local pair
  for pair in $envpairs; do args+=(--setenv="$pair"); done
  systemd-run --user --unit=$UNIT --collect "${args[@]}" \
    /bin/bash "$launcher" > /dev/null || return 1
  return 0
}

unit_props() {
  # shellcheck disable=SC2086
  systemctl --user show "$UNIT" $(for p in $UNIT_PROPS_SHOWN; do printf -- '-p %s ' "$p"; done) \
    2>/dev/null
}

snapshot_props() {
  local dir=$1 tag=$2
  unit_props > "$dir/unit-props-$tag.txt"
}

unit_env_line() {
  grep -m1 '^Environment=' "$1" 2>/dev/null | cut -d= -f2- || true
}

stop_service() {
  systemctl --user stop "$UNIT" 2>/dev/null || true
  local i
  for i in $(seq 1 45); do
    [ "$(service_pid)" = "0" ] && return 0
    sleep 1
  done
  log "WARN unit did not stop in 45s; SIGKILL"
  systemctl --user kill -s SIGKILL "$UNIT" 2>/dev/null || true
  for _ in $(seq 1 15); do
    [ "$(service_pid)" = "0" ] && return 0
    sleep 1
  done
  return 1
}

wait_unit_collected() {
  local i
  for i in $(seq 1 20); do
    [ "$(systemctl --user show -p LoadState --value "$UNIT" 2>/dev/null)" != \
      "loaded" ] && return 0
    systemctl --user reset-failed "$UNIT" > /dev/null 2>&1 || true
    sleep 1
  done
  log "WARN $UNIT still loaded after stop; reset-failed + continue"
  return 0
}

start_service() {
  local launcher=$1 envpairs=${2:-}
  stop_service
  wait_unit_collected
  recreate_unit "$launcher" "$envpairs" || return 1
  log "unit started with $(basename "$launcher") env=[${envpairs}]"
  return 0
}

enforce_so_baseline() {
  local cur note=$WINDOW_DIR/so-at-window-start.txt
  cur=$(so_md5_live)
  echo "md5_at_lock_acquire=$cur" > "$note"
  echo "baseline=$BASELINE_SO_MD5" >> "$note"
  if [ "$cur" = "$BASELINE_SO_MD5" ]; then
    log ".so already at baseline $cur"
    echo "action=none" >> "$note"
    return 0
  fi
  log "live .so is $cur != baseline; restoring $BASELINE_SO_MD5"
  [ "$DRY_RUN" = "1" ] && { echo "action=would-restore" >> "$note"; return 0; }
  docker exec $C bash -lc "test -f '$SO_BACKUP/_rocm_C.$BASELINE_SO_MD5.so'" \
    || die "no baseline backup $SO_BACKUP/_rocm_C.$BASELINE_SO_MD5.so"
  docker exec $C bash -lc "cp -p '$SO_BACKUP/_rocm_C.$BASELINE_SO_MD5.so' '$SO_TGT'" \
    || die "baseline .so restore failed"
  [ "$(so_md5_live)" = "$BASELINE_SO_MD5" ] || die "restored .so md5 mismatch"
  echo "action=restored-baseline" >> "$note"
  return 0
}

promote_so() {
  local src=$1 cur new
  [ -f "$src" ] || src="$SO_BACKUP/_rocm_C.$src.so"
  docker exec $C bash -lc "test -f '$src'" || die ".so candidate $src missing"
  new=$(docker exec $C bash -lc "md5sum '$src'" | cut -d' ' -f1)
  cur=$(so_md5_live)
  docker exec $C bash -lc "test -f '$SO_BACKUP/_rocm_C.$cur.so' || cp -p '$SO_TGT' '$SO_BACKUP/_rocm_C.$cur.so'"
  docker exec $C bash -lc "cp -p '$src' '$SO_TGT'" || die "so install failed"
  [ "$(so_md5_live)" = "$new" ] || die "so md5 after install != $new"
  log "so promoted: $cur -> $new (backup $SO_BACKUP/_rocm_C.$cur.so)"
  echo "old_md5=$cur" > "$2/so-swap.txt"
  echo "new_md5=$new" >> "$2/so-swap.txt"
  echo "src=$src" >> "$2/so-swap.txt"
}

rollback_so() {
  local want=${1:-$BASELINE_SO_MD5} cur
  cur=$(so_md5_live)
  [ "$cur" = "$want" ] && return 0
  docker exec $C bash -lc "test -f '$SO_BACKUP/_rocm_C.$want.so'" \
    || die "no backup $SO_BACKUP/_rocm_C.$want.so to roll back to"
  docker exec $C bash -lc "cp -p '$SO_BACKUP/_rocm_C.$want.so' '$SO_TGT'"
  [ "$(so_md5_live)" = "$want" ] || die "rollback md5 mismatch"
  log "so rolled back $cur -> $want"
  return 10
}

run_arm() {
  local arm=$1 dir=$WINDOW_DIR/$1
  ARM_DESC=""; ARM_ENV=""; ARM_SO=""; ARM_LAUNCHER=""
  [ -f "$ARMS/$arm.env" ] || die "no arm descriptor $ARMS/$arm.env"
  # shellcheck disable=SC1090
  . "$ARMS/$arm.env"
  mkdir -p "$dir"
  echo "$WINDOW_START" > "$dir/window-start.txt"

  log "=== arm $arm: $ARM_DESC"
  if ! preflight "$dir" "$arm"; then
    log "ABORT arm $arm: pre-flight gate failed (no restart)"
    echo "preflight-failed" > "$dir/INVALID"
    INVALID="$INVALID $arm:preflight"
    return 1
  fi
  local FP
  FP=$(tree_fingerprint)
  if [ -n "$WINDOW_FP" ] && [ "$FP" != "$WINDOW_FP" ]; then
    log "ABORT arm $arm: tree changed inside the window ($WINDOW_FP -> $FP)"
    echo "tree-drift" > "$dir/INVALID"
    INVALID="$INVALID $arm:tree-drift"
    return 1
  fi
  WINDOW_FP=${WINDOW_FP:-$FP}
  echo "$FP" > "$dir/tree-fingerprint.txt"

  cat > "$dir/env-requested.txt" <<EOF
arm=$arm
desc=$ARM_DESC
env=$ARM_ENV
so=${ARM_SO:-<none>}
launcher=${ARM_LAUNCHER:-$LAUNCHER_DEFAULT}
lens=$LENS reps=$REPS gen=$GEN soak=$SOAK
EOF
  [ "$DRY_RUN" = "1" ] && { log "dry-run: would launch arm $arm"; return 0; }

  local from want_so
  from=$(wc -l < "$LOGFILE" 2>/dev/null || echo 0)
  want_so=${ARM_SO:-${WINDOW_SO:-$BASELINE_SO_MD5}}
  if [ "$(so_md5_live)" != "$want_so" ]; then
    log "installing arm .so $want_so (live $(so_md5_live))"
    stop_service
    promote_so "$want_so" "$dir"
  fi
  snapshot_props "$dir" before
  if ! start_service "${ARM_LAUNCHER:-$LAUNCHER_DEFAULT}" "$ARM_ENV"; then
    log "ABORT arm $arm: could not recreate/start the unit"
    echo "start-failed" > "$dir/INVALID"
    INVALID="$INVALID $arm:start-failed"
    return 1
  fi
  snapshot_props "$dir" after
  local env_after
  env_after=$(unit_env_line "$dir/unit-props-after.txt")
  log "unit Environment after start: [${env_after}]"
  case "$env_after" in
    *HOST=0.0.0.0*) : ;;
    *) log "WARN arm $arm: HOST=0.0.0.0 missing from the recreated unit"
       INVALID="$INVALID $arm:host-prop" ;;
  esac

  if ! wait_ready 200; then
    log "ABORT arm $arm: service not ready"
    show_new_log "$from" | tee -a "$dir/service.log"
    echo "not-ready" > "$dir/INVALID"
    INVALID="$INVALID $arm:not-ready"
    return 1
  fi
  log "arm $arm ready in $(date +%H:%M:%S); pid=$(service_pid)"
  tail -n +"$((from + 1))" "$LOGFILE" | tail -n 4000 > "$dir/startup.log" 2>/dev/null
  "$PY" "$W/tools/dcu/s24_logscan.py" --log "$dir/startup.log" \
    --out "$dir/memfacts.json" > "$dir/logscan.log" 2>&1 || true
  grep -E "^\[logscan\]" "$dir/logscan.log" | tail -1

  local so_now
  so_now=$(so_md5_live)
  if [ "$so_now" != "$want_so" ]; then
    log "ABORT arm $arm: .so is $so_now, arm requires $want_so"
    echo "so-mismatch" > "$dir/INVALID"
    INVALID="$INVALID $arm:so-mismatch"
    return 1
  fi
  WINDOW_SO=${WINDOW_SO:-$so_now}

  local pid
  pid=$(service_pid)
  "$PY" "$W/tools/dcu/s23_descriptor.py" --out "$dir/descriptor.json" \
    --window "$WN/$ID/$WINDOW_TAG" --arm "$arm" --desc "$ARM_DESC" \
    --so-md5 "$(so_md5_live)" --pid "$pid" --arm-env "$ARM_ENV" \
    --extra "launcher=${ARM_LAUNCHER:-$LAUNCHER_DEFAULT}" \
    --unit-props "$dir/unit-props-after.txt" \
    || { log "ABORT arm $arm: arm env did not reach the engine";
         echo "env-mismatch" > "$dir/INVALID"; INVALID="$INVALID $arm:env-mismatch"; return 1; }
  grep -q '"env_mismatch_ok": true' "$dir/descriptor.json" \
    || { log "ABORT arm $arm: env mismatch (see descriptor.json)";
         echo "env-mismatch" > "$dir/INVALID"; INVALID="$INVALID $arm:env-mismatch"; return 1; }

  if ! "$PY" "$W/tools/dcu/s23_smoke.py" --out "$dir/smoke.json" \
      >> "$dir/service.log" 2>&1; then
    log "ABORT arm $arm: smoke failed"
    tail -5 "$dir/service.log"
    echo "smoke-failed" > "$dir/INVALID"; INVALID="$INVALID $arm:smoke"; return 1
  fi
  grep -E "^\[smoke\]" "$dir/service.log" | tail -3

  log "longctx lens=$LENS reps=$REPS arm=$arm"
  "$PY" "$W/tools/dcu/s24_longctx.py" --lens "$LENS" --reps "$REPS" --gen "$GEN" \
    --tag "${WINDOW_TAG}_${arm}" --out "$dir" > "$dir/longctx.log" 2>&1
  grep -E "^\[SUMMARY\]" "$dir/longctx.log"

  if [ -n "$SOAK" ]; then
    log "soak lens=$SOAK reps=1 arm=$arm"
    "$PY" "$W/tools/dcu/s24_longctx.py" --lens "$SOAK" --reps 1 --gen 16 \
      --seed0 424243 --tag "${WINDOW_TAG}_${arm}_soak" --out "$dir" \
      > "$dir/soak.log" 2>&1
    grep -E "^\[SUMMARY\]|error" "$dir/soak.log" | tail -4
  fi

  # L5 evidence: static occupancy right after the arm's own traffic.
  hy-smi 2>/dev/null | head -12 > "$dir/hysmi-after.txt" || true
  curl -s -m 5 -H "Authorization: Bearer $(cat "$KEYFILE")" "$BASE/metrics" \
    | grep -E "^vllm:(num_requests_running|kv_cache_usage_perc|gpu_cache_usage_perc)" \
    > "$dir/metrics-after.txt" || true

  if [ "$DO_PARITY" = "1" ]; then
    mkdir -p "$dir/parity"
    if [ -z "$PARITY_REF" ]; then
      log "parity baseline capture (arm $arm)"
      "$PY" "$W/tools/dcu/s23_parity.py" --mode capture \
        --out "$dir/parity/${arm}.json" > "$dir/parity/capture.log" 2>&1
      grep -E "^\[(capture|parity)" "$dir/parity/capture.log" | tail -4
      PARITY_REF="$dir/parity/${arm}.json"
    else
      log "parity check vs $PARITY_REF"
      if ! "$PY" "$W/tools/dcu/s23_parity.py" --mode check \
          --baseline "$PARITY_REF" --retry-on-diverge \
          --out "$dir/parity/${arm}-check.json" > "$dir/parity/check.log" 2>&1; then
        log "PARITY BROKEN in arm $arm"
        echo "parity-broken" > "$dir/INVALID"
        INVALID="$INVALID $arm:parity"
      fi
      grep -E "^\[(check|parity)" "$dir/parity/check.log" | tail -4
    fi
  fi
  return 0
}

restore_default() {
  [ "$RESTORED" = "1" ] && return 0
  if [ "$DRY_RUN" = "1" ]; then RESTORED=1; return 0; fi
  [ "$DO_RESTORE" = "1" ] || { log "WARN --no-restore: leaving last arm env"; RESTORED=1; return 0; }
  RESTORED=1
  local dir=$WINDOW_DIR/restore
  mkdir -p "$dir"
  log "restoring default configuration"
  local t0
  t0=$(date +%s)
  rollback_so "$BASELINE_SO_MD5" || log "so restored to $BASELINE_SO_MD5"
  if ! start_service "$LAUNCHER_DEFAULT" ""; then
    log "FATAL: could not restart the production launcher -- MANUAL ACTION NEEDED"
    echo "restore-start-failed" > "$dir/INVALID"
    INVALID="$INVALID restore:start-failed"
    return 1
  fi
  if wait_ready 200; then
    "$PY" "$W/tools/dcu/s23_smoke.py" --out "$dir/smoke.json" \
      >> "$dir/service.log" 2>&1 && log "default restored: ready + smoke ok"
    log "restore took $(( $(date +%s) - t0 ))s"
  else
    log "FATAL: service did not come back on the default configuration"
    echo "restore-not-ready" > "$dir/INVALID"
    INVALID="$INVALID restore"
  fi
  echo "$(date -Is)" > "$dir/window-end.txt"
}

finish() {
  local rc=$?
  trap - EXIT INT TERM
  restore_default
  log "window $WN/$ID/$WINDOW_TAG done (rc=$rc) invalid='$INVALID'"
  [ -n "$INVALID" ] && exit 1
  exit "$rc"
}
trap finish EXIT INT TERM

log "window $WN/$ID/$WINDOW_TAG start; arms: ${ARMS_REQ[*]}; so_live=$(so_md5_live)"
log "lens=$LENS reps=$REPS gen=$GEN soak=$SOAK"
{
  echo "window=$WN/$ID/$WINDOW_TAG"
  echo "start=$WINDOW_START"
  echo "arms=${ARMS_REQ[*]}"
  echo "lens=$LENS reps=$REPS gen=$GEN soak=$SOAK"
  echo "baseline_so_md5=$BASELINE_SO_MD5"
  echo "git_rev=$(git -C "$W" rev-parse HEAD)"
  echo "lock_holder=$$"
} > "$WINDOW_DIR/window.txt"
git -C "$W" status --short > "$WINDOW_DIR/git-status.txt"
clear_manager_env
enforce_so_baseline

for arm in "${ARMS_REQ[@]}"; do
  run_arm "$arm" || log "arm $arm did not complete; continuing"
done
