#!/usr/bin/env bash
# S23 (O6/O7) A/B driver: restart the production *user* unit with a set of
# VLLM_DCU_* overrides, optionally swap the _rocm_C .so, then measure every arm
# back-to-back inside one GPU-lock window.
#
#   bash tools/dcu/s23_ab.sh [options] <window-tag> <arm> [<arm> ...]
#
# Arms are descriptor files in tools/dcu/s23_arms/<arm>.env (see control.env).
# Windows are append-only directories .s23/w1/O6/<window-tag>/<arm>/.
#
# Options
#   --repeat N          c=1 probes per arm (default 4)
#   --sweep C,C         also run s20_conc_sweep for these concurrencies
#   --sweep-reps N      sweep probes per arm (default 1; report aggregates N)
#   --prompt-tokens N   c=1 prompt length (default 512)
#   --max-tokens N      c=1 decode length (default 128)
#   --no-parity         skip the greedy 8-prompt parity pass
#   --no-restore        keep the last arm's env (default: always restore + verify)
#   --unit-gate         gate 1: with the service stopped, run the fp8 pytest
#                       suites in all four switch states
#   --census            per-arm kernel microbenchmark (mprobe) after the probes
#   --ref-arm NAME      reference arm for the delta columns (default: control)
#   --dry-run           run the pre-flight gates and the plan, never restart
#   --force             reuse an existing window directory
#
# Discipline (plan §5/§6, .s23/S23_EXEC_PROTOCOL.md):
#   * the whole window holds .s23/gpu.lock (same inode the container probes use),
#     so nobody's kernel probe can steal the window;
#   * before EVERY restart: py_compile of every python file changed since the
#     baseline commit + a container import smoke; abort the arm on failure;
#   * the tree fingerprint must not change between arms of one window, otherwise
#     the arm difference is not attributable;
#   * the window always ends with the default configuration running and healthy
#     (env unset, baseline .so md5 restored, ready + fixed-prompt smoke).
set -u

W=/home/lyh/z100-vllm
LOCK=$W/.s23/gpu.lock
ARMS=$W/tools/dcu/s23_arms
C=z100-regress
UNIT=z100-vllm-host.service
LOGFILE=/mnt/game/host-serve/logs/serve-host.log
KEYFILE=/mnt/game/host-serve/api-key
BASE=${S23_BASE:-http://127.0.0.1:8000}
PY=$W/.venv/bin/python
SO_TGT=/mnt/game/port/build-sync/temp/_rocm_C.abi3.so
SO_BACKUP=/mnt/game/perf1
BASELINE_SO_MD5=${S23_BASELINE_SO_MD5:-f205be8f43d3ffb1cfb3c6f093bfab4b}
SWITCHES="VLLM_DCU_FP8_EMU_FASTPATH VLLM_DCU_FP8_QUANT_FASTPATH VLLM_DCU_FP8_FUSE_QUANT"
LAUNCHER_DEFAULT=$W/tools/dcu/serve_host_dflash2.sh
# The production unit is transient; these are the properties the checked-in unit
# file documents and the ones the live unit actually carries (verified against
# `systemctl --user show` on 2026-10-19: Restart=no, CollectMode=inactive-or-
# failed, HOST=0.0.0.0, cwd /home/lyh/z100-vllm, both streams appended to the
# serve log, TimeoutStopSec=30s).  Every recreate goes through this one list so
# an arm cannot silently drop a property; the driver additionally diffs the
# property set of the incumbent unit against the recreated one.
UNIT_WORKDIR=$W
UNIT_STDOUT=append:$LOGFILE
UNIT_STDERR=append:$LOGFILE
UNIT_TIMEOUT_STOP=30
UNIT_ENV_HOMEPAGE=HOST=0.0.0.0
# Property names are the ones `systemctl --user show` actually reports; e.g. the
# stop timeout is TimeoutStopUSec, and StandardOutput only carries the mode
# ("append") with no path, so the log path is verified separately by asserting
# the launcher banner shows up in serve-host.log after every start.
UNIT_PROPS_SHOWN="Type Restart RestartUSec TimeoutStopUSec WorkingDirectory
StandardOutput StandardError CollectMode ExecStart Environment"

log() { echo "[s23 $(date +%H:%M:%S)] $*"; }
die() { log "FATAL: $*"; exit 1; }

# ---------------------------------------------------------------- lock window
if [ "${S23_LOCK_HELD:-0}" != "1" ]; then
  [ -d "$W/.s23" ] || mkdir -p "$W/.s23"
  [ -f "$LOCK" ] || : > "$LOCK"
  export S23_LOCK_HELD=1
  log "waiting for GPU lock $LOCK (window '$*')"
  exec flock -w "${S23_LOCK_WAIT:-3600}" "$LOCK" bash "$0" "$@"
fi

# ---------------------------------------------------------------------- args
REPEAT=4
SWEEP=""
SWEEP_REPS=1
PROMPT_TOKENS=512
MAX_TOKENS=128
DO_PARITY=1
DO_RESTORE=1
DRY_RUN=0
FORCE=0
UNIT_GATE=0
REF_ARM=control
while [ $# -gt 0 ]; do
  case "$1" in
    --repeat) REPEAT=$2; shift 2 ;;
    --sweep) SWEEP=$2; shift 2 ;;
    --sweep-reps) SWEEP_REPS=$2; shift 2 ;;
    --prompt-tokens) PROMPT_TOKENS=$2; shift 2 ;;
    --max-tokens) MAX_TOKENS=$2; shift 2 ;;
    --no-parity) DO_PARITY=0; shift ;;
    --no-restore) DO_RESTORE=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --unit-gate) UNIT_GATE=1; shift ;;
    --census) S23_CENSUS=1; shift ;;
    --ref-arm) REF_ARM=$2; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) break ;;
  esac
done
WINDOW=${1:?usage: s23_ab.sh [options] <window-tag> <arm> [<arm> ...]}
shift
[ $# -ge 1 ] || die "need at least one arm (see $ARMS)"
ARMS_REQ=("$@")

WINDOW_DIR=$W/.s23/w1/O6/$WINDOW
[ -e "$WINDOW_DIR" ] && [ "$FORCE" != "1" ] \
  && die "window dir $WINDOW_DIR exists (use --force to append)"
mkdir -p "$WINDOW_DIR"
WINDOW_START=$(date -Is)
PARITY_REF=""
RESTORED=0
INVALID=""
WINDOW_FP=""
WINDOW_SO=""

# --------------------------------------------------------------- http helpers
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
    # A unit that is not even active for ~10 s is not warming up, it is dead
    # (bad .so, bad flag): fail fast instead of burning the whole budget.
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

# -------------------------------------------------------------- gates / preflight
# Only python the engine can import: vllm/ and csrc/ (plus scripts).  A
# teammate's scratch probe under .s23/ or tools/ is neither compiled nor part of
# the drift fingerprint -- it cannot change what the service runs.
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
  [ "${S23_IGNORE_RECENT_EDITS:-0}" = "1" ] && {
    log "WARN continuing despite recent edits (S23_IGNORE_RECENT_EDITS=1)"
    return 0
  }
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
  # The arm's own switch values are injected so --expect proves the env name
  # resolves in vllm.envs before we spend a restart on it.
  docker exec $C bash -c "cd /src && env ${ARM_ENV:-} \
    DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
    bash -c 'set +u; source tools/dcu/env.sh >/dev/null 2>&1; \
      export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; \
      export HIP_VISIBLE_DEVICES=3; \
      .venv/bin/python tools/dcu/s23_preflight.py $ARM_PREFLIGHT_EXPECT'" \
    >> "$dir/preflight.log" 2>&1 || rc=1
  tail -n 12 "$dir/preflight.log"
  return "$rc"
}

# ------------------------------------------------------------------- service
# Leftover manager-environment switches from an earlier attempt would leak into
# every unit we create, so clear them once per window and keep the manager env out
# of the picture entirely (arm values are per-unit --setenv now).
clear_manager_env() {
  systemctl --user unset-environment $SWITCHES 2>/dev/null || true
  local left
  left=$(systemctl --user show-environment | grep -E '^VLLM_DCU_FP8_(EMU|QUANT|FUSE)' || true)
  [ -n "$left" ] && log "WARN manager env still carries: $left"
  return 0
}

# The production unit is transient (`systemd-run --user --unit=z100-vllm-host
# --collect`, FragmentPath /run/user/1000/systemd/transient/...), so drop-ins are
# not available and ExecStart cannot be swapped in place.  Every arm therefore
# recreates the unit from the launcher it wants, with the properties the checked
# in unit file documents; restoring the default configuration recreates it from
# tools/dcu/serve_host_dflash2.sh again.  Transient units inherit the *manager*
# environment, which is how set-environment reaches the engine.
# Arm switches go through --setenv (NOT `systemctl --user set-environment`): the
# injected value then lands in the unit's own Environment= property, which is the
# only place this sandbox can prove it (reading /proc/<engine pid>/environ of a
# systemd-managed process is denied, which made the first canary version report a
# false "env did not reach the engine" on both w1a arms).
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
  systemd-run --user --unit=z100-vllm-host --collect "${args[@]}" \
    /bin/bash "$launcher" > /dev/null || return 1
  return 0
}

# Property snapshot of the incumbent unit, so every arm records what production
# looked like before it and what it looks like after (lead requirement: diff the
# property set, not just ExecStart).
unit_props() {
  # shellcheck disable=SC2086
  systemctl --user show "$UNIT" $(for p in $UNIT_PROPS_SHOWN; do printf -- '-p %s ' "$p"; done) \
    2>/dev/null
}

snapshot_props() {
  local dir=$1 tag=$2
  unit_props > "$dir/unit-props-$tag.txt"
}

start_service() {
  local launcher=$1 envpairs=${2:-}
  stop_service
  wait_unit_collected
  recreate_unit "$launcher" "$envpairs" || return 1
  log "unit started with $(basename "$launcher") env=[${envpairs}]"
  return 0
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
  # TimeoutStopSec is 30s; if the engine ignored SIGTERM long enough to still be
  # alive here, kill it rather than leave VRAM held for the next arm.
  log "WARN unit did not stop in 45s; SIGKILL"
  systemctl --user kill -s SIGKILL "$UNIT" 2>/dev/null || true
  for _ in $(seq 1 15); do
    [ "$(service_pid)" = "0" ] && return 0
    sleep 1
  done
  return 1
}

# A --collect transient unit is garbage collected after stop; recreating while
# the old object still exists fails with "unit already exists", so wait for it.
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

# ------------------------------------------------------- gate 1 (pytest slot)
# The only gate that needs the engine down and all four GPUs free.  Run twice:
# switches off (must reproduce the shipped path) and EMU_FASTPATH=1 (the O1
# path).  Failures are recorded verbatim -- a harness-only failure must be
# reported to the lead, not silently dropped.
# Priority order: drop test_fp8_min_max_helper.py first if the slot gets long,
# never the other two (lead instruction).
GATE_TESTS=${S23_GATE_TESTS:-"tests/kernels/quantization/test_gfx906_fp8_emu.py \
tests/kernels/quantization/test_fp8_quant.py \
tests/kernels/quantization/test_fp8_quant_group.py \
tests/kernels/quantization/test_fp8_min_max_helper.py"}

pytest_run() {
  local logf=$1 envs=$2 rc=0
  log "pytest gate [$envs]"
  timeout "${S23_PYTEST_TIMEOUT:-600}" docker exec $C bash -c "cd /src && env $envs \
    DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
    bash -c 'set +u; source tools/dcu/env.sh >/dev/null 2>&1; \
      export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; \
      export HIP_VISIBLE_DEVICES=${S23_GATE_DEVICES:-3}; \
      .venv/bin/python -m pytest $GATE_TESTS -q'" \
    > "$logf" 2>&1 || rc=1
  tail -n 3 "$logf"
  return "$rc"
}

# One pass per switch state we ship: the shipped path, each fast path alone, and
# all of them on.  Every pass needs the engine down (the tests' reference GEMM
# needs free VRAM), which is why this runs inside the window's stopped slot.
# Ordered by importance so a budget cut drops the least informative pass first:
# the emu-only state was already covered by w1a/w1c for the emu suite, while the
# quant states are exactly what O2 needs.
GATE_PASSES=(
  "switches-off|VLLM_DCU_FP8_EMU_FASTPATH=0 VLLM_DCU_FP8_QUANT_FASTPATH=0 VLLM_DCU_FP8_FUSE_QUANT=0"
  "quant-fastpath-on|VLLM_DCU_FP8_EMU_FASTPATH=0 VLLM_DCU_FP8_QUANT_FASTPATH=1 VLLM_DCU_FP8_FUSE_QUANT=0"
  "all-switches-on|VLLM_DCU_FP8_EMU_FASTPATH=1 VLLM_DCU_FP8_QUANT_FASTPATH=1 VLLM_DCU_FP8_FUSE_QUANT=1"
  "emu-fastpath-on|VLLM_DCU_FP8_EMU_FASTPATH=1 VLLM_DCU_FP8_QUANT_FASTPATH=0 VLLM_DCU_FP8_FUSE_QUANT=0"
)

run_unit_gate() {
  local dir=$WINDOW_DIR/unit-gate v i rc_off=0 rc_on=0
  mkdir -p "$dir"
  if [ "$DRY_RUN" = "1" ]; then
    log "dry-run: would run the pytest gate with the service stopped"
    return 0
  fi
  log "gate 1: stopping the service to free all 4 GPUs"
  stop_service || log "WARN stop not confirmed"
  wait_unit_collected
  for i in $(seq 1 40); do
    v=$(hy-smi 2>/dev/null | awk '/^[0-9]/ {gsub("%","",$6); s+=$6} END {print s+0}')
    [ "${v:-400}" -lt 20 ] && break
    sleep 3
  done
  log "VRAM sum after stop: ${v:-?}%"
  local pass label envs rc rc_all=0 t0="${GATE_T0:-0}" budget="${S23_GATE_BUDGET:-900}"
  t0=$(date +%s)
  : > "$dir/summary.txt"
  for pass in "${GATE_PASSES[@]}"; do
    label=${pass%%|*}
    envs=${pass#*|}
    if [ $(( $(date +%s) - t0 )) -ge "$budget" ]; then
      echo "$label skipped (budget ${budget}s)" >> "$dir/summary.txt"
      log "gate 1: skipping $label (budget reached)"
      continue
    fi
    rc=0
    pytest_run "$dir/pytest-$label.log" "$envs" || rc=1
    echo "$label rc=$rc env=[$envs]" >> "$dir/summary.txt"
    tail -n 1 "$dir/pytest-$label.log" >> "$dir/summary.txt"
    [ "$rc" = "0" ] || rc_all=1
  done
  tail -n 8 "$dir/summary.txt"
  log "gate 1 done: $(grep -c ' rc=0 ' "$dir/summary.txt")/4 passes green ($dir)"
  [ "$rc_all" = "0" ] || INVALID="$INVALID unit-gate"
  return 0
}

# ------------------------------------------------------ .so ownership invariant
# Whoever holds the GPU lock owns the live .so and must leave the baseline
# behind (lead invariant).  Restore before the first arm so that every arm of
# this window runs the same python-only build.
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
  if [ "$DRY_RUN" = "1" ]; then
    echo "action=would-restore (dry-run)" >> "$note"
    return 0
  fi
  docker exec $C bash -lc "test -f '$SO_BACKUP/_rocm_C.$BASELINE_SO_MD5.so'" \
    || die "no baseline backup $SO_BACKUP/_rocm_C.$BASELINE_SO_MD5.so"
  docker exec $C bash -lc "cp -p '$SO_BACKUP/_rocm_C.$BASELINE_SO_MD5.so' '$SO_TGT'" \
    || die "baseline .so restore failed"
  [ "$(so_md5_live)" = "$BASELINE_SO_MD5" ] || die "restored .so md5 mismatch"
  echo "action=restored-baseline" >> "$note"
  return 0
}

# ------------------------------------------------------------------------ .so
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

# ---------------------------------------------------------------------- census
# Kernel-level census for one arm, run AFTER its probes so a microbenchmark
# cannot perturb the e2e numbers.  It reloads the extension from whatever .so is
# installed, so the census belongs to the arm's build; the arm env (e.g.
# VLLM_DCU_FP8_DECODE_CFG) selects the kernel variant under test.
run_census() {
  local dir=$1 cdir
  cdir=/src/${dir#"$W"/}
  log "census: mprobe with arm env [$ARM_ENV]"
  docker exec $C bash -c "cd /src && env ${ARM_ENV:-} \
    DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
    bash -c 'set +u; source tools/dcu/env.sh >/dev/null 2>&1; \
      export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; \
      export HIP_VISIBLE_DEVICES=${S23_CENSUS_DEV:-3}; \
      export VLLM_DCU_USE_FP8_EMU=1 VLLM_DCU_FP8_GEMV_MAX_M=8; \
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
      echo ROMC_MD5=\$(md5sum vllm/_rocm_C.abi3.so | cut -d\" \" -f1); \
      .venv/bin/python -u .s23/probes/o3_mprobe.py --ms 8 --iters 40 --rounds 3 \
        --arms prod --check --out $cdir/census-out.json'" \
    > "$dir/census.log" 2>&1 || log "WARN census failed (see $dir/census.log)"
  grep -E "ROMC_MD5|canary-pre|^\[probe\] wrote" "$dir/census.log" | head -3
  tail -n 6 "$dir/census.log"

  # Kernel-name census: the O4 gate is host logic, so its only observable proof is
  # which kernel runs at M=8 for 5120:1536.
  log "census: kernel names (expect-gate=${ARM_EXPECT_GATE:-any})"
  docker exec $C bash -c "cd /src && env ${ARM_ENV:-} \
    DTK_ROOT=/mnt/game/host-runtime/dtk2604 \
    bash -c 'set +u; source tools/dcu/env.sh >/dev/null 2>&1; \
      export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; \
      export HIP_VISIBLE_DEVICES=${S23_CENSUS_DEV:-3}; \
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
      .venv/bin/python -u tools/dcu/s23_kernel_census.py \
        --expect-gate ${ARM_EXPECT_GATE:-any} \
        --out $cdir/census-kernels.json'" \
    > "$dir/census-kernels.log" 2>&1 || log "WARN kernel census failed"
  grep -E "^\[census\]" "$dir/census-kernels.log" | head -8
  return 0
}

# ------------------------------------------------------------------- one arm
run_arm() {
  local arm=$1 dir=$WINDOW_DIR/$1
  ARM_DESC=""; ARM_ENV=""; ARM_SO=""; ARM_PREFLIGHT_EXPECT=""; ARM_LAUNCHER=""
  ARM_EXPECT_GATE="any"
  ARM_SWEEP="on"
  ARM_CENSUS="${S23_CENSUS:-0}"
  ARM_LOGSCAN=0
  ARM_READY_TRIES=150
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
  FP=$(tree_fingerprint)
  if [ -n "$WINDOW_FP" ] && [ "$FP" != "$WINDOW_FP" ]; then
    if [ "${S23_ALLOW_TREE_DRIFT:-0}" != "1" ]; then
      log "ABORT arm $arm: tree changed inside the window ($WINDOW_FP -> $FP)"
      echo "tree-drift $WINDOW_FP -> $FP" > "$dir/INVALID"
      INVALID="$INVALID $arm:tree-drift"
      return 1
    fi
    log "WARN tree drift allowed ($WINDOW_FP -> $FP)"
  fi
  WINDOW_FP=${WINDOW_FP:-$FP}
  echo "$FP" > "$dir/tree-fingerprint.txt"

  cat > "$dir/env-requested.txt" <<EOF
arm=$arm
desc=$ARM_DESC
env=$ARM_ENV
so=${ARM_SO:-<none>}
launcher=${ARM_LAUNCHER:-$LAUNCHER_DEFAULT}
repeat=$REPEAT prompt_tokens=$PROMPT_TOKENS max_tokens=$MAX_TOKENS
sweep=$SWEEP sweep_reps=$SWEEP_REPS
ready_tries=$ARM_READY_TRIES
EOF
  [ "$DRY_RUN" = "1" ] && { log "dry-run: would launch arm $arm"; return 0; }

  local from want_so
  from=$(wc -l < "$LOGFILE" 2>/dev/null || echo 0)
  # Every arm declares the build it must run: ARM_SO when it swaps, otherwise the
  # window's first-arm build (baseline for a python-only window).  This also heals
  # state after a swap arm so the next arm cannot silently inherit its .so.
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
  if grep -q '^Type=simple' "$dir/unit-props-before.txt"; then
    diff -u "$dir/unit-props-before.txt" "$dir/unit-props-after.txt" \
      > "$dir/unit-props-diff.txt" 2>&1 || true
  else
    # The gate slot already stopped (and collected) the unit, so "before" is an
    # absent unit; diffing it would only emit empty-vs-real noise.
    echo "incumbent unit absent at arm start; no property diff" \
      > "$dir/unit-props-diff.txt"
  fi
  # ExecStart is expected to differ only for a launcher arm (O7); anything else
  # changing means the recreate did not reproduce the production unit.
  local env_after unexpected
  env_after=$(unit_env_line "$dir/unit-props-after.txt")
  log "unit Environment after start: [${env_after}]"
  case "$env_after" in
    *HOST=0.0.0.0*) : ;;
    *) log "WARN arm $arm: HOST=0.0.0.0 missing from the recreated unit"
       INVALID="$INVALID $arm:host-prop" ;;
  esac
  unexpected=$(grep -E '^[-+][A-Za-z]' "$dir/unit-props-diff.txt" \
    | grep -v 'ExecStart' | grep -v 'Environment' | grep -v 'unit absent' || true)
  if [ -n "$unexpected" ]; then
    log "WARN arm $arm: unit properties changed beyond ExecStart/Environment"
    echo "$unexpected" | head -5
  else
    log "unit properties reproduced (only ExecStart/Environment may differ)"
  fi

  local rc_ready=0
  wait_ready "$ARM_READY_TRIES" || rc_ready=1
  # Slice AFTER the wait: the resolved-config line is printed only once the
  # engine initialises, so a pre-wait slice resolves to None.  Kept for failing
  # arms too -- a compile-mode arm (O7) fails inside that path and the lines are
  # the evidence.
  tail -n +"$((from + 1))" "$LOGFILE" | tail -n 4000 > "$dir/startup.log" 2>/dev/null
  local resolved_line=""
  if [ -s "$dir/startup.log" ]; then
    resolved_line=$("$PY" "$W/tools/dcu/s23_resolved.py" --log "$dir/startup.log" \
      --out "$dir/resolved-config.json" 2>&1 | grep -m1 '\[resolved\] mode=' || true)
    log "${resolved_line:-[resolved] unavailable (no init line in slice)}"
  fi
  if [ "${ARM_LOGSCAN:-0}" = "1" ] && [ -s "$dir/startup.log" ]; then
    "$PY" "$W/tools/dcu/s23_logscan.py" --log "$dir/startup.log" \
      --out "$dir/logscan.json" > "$dir/logscan.log" 2>&1 || true
    grep -E "^\[logscan\]" "$dir/logscan.log" | tail -1
  fi
  if [ "$rc_ready" != "0" ]; then
    log "ABORT arm $arm: service not ready after $ARM_READY_TRIES polls"
    show_new_log "$from" | tee -a "$dir/service.log"
    echo "not-ready" > "$dir/INVALID"
    INVALID="$INVALID $arm:not-ready"
    return 1
  fi
  log "arm $arm ready in $(date +%H:%M:%S); pid=$(service_pid)"
  # Proves the StandardOutput=append property took effect on the recreated unit.
  if tail -n +"$((from + 1))" "$LOGFILE" | grep -qE '^\[(host-serve|o7-arm)\]'; then
    log "launcher banner present in $LOGFILE (stdout append OK)"
  else
    log "WARN arm $arm: no launcher banner in $LOGFILE since start"
  fi

  local so_now
  so_now=$(so_md5_live)
  if [ "$so_now" != "$want_so" ]; then
    log "ABORT arm $arm: .so is $so_now, arm requires $want_so"
    echo "so-mismatch $so_now != $want_so" > "$dir/INVALID"
    INVALID="$INVALID $arm:so-mismatch"
    return 1
  fi
  WINDOW_SO=${WINDOW_SO:-$so_now}

  local pid
  pid=$(service_pid)
  "$PY" "$W/tools/dcu/s23_descriptor.py" --out "$dir/descriptor.json" \
    --window "$WINDOW" --arm "$arm" --desc "$ARM_DESC" \
    --so-md5 "$(so_md5_live)" --pid "$pid" --arm-env "$ARM_ENV" \
    --extra "profile=${ARM_PROFILE:-}" \
    --extra "launcher=${ARM_LAUNCHER:-$LAUNCHER_DEFAULT}" \
    --extra "resolved_config=${resolved_line:-<unavailable>}" \
    --unit-props "$dir/unit-props-after.txt" \
    || { log "ABORT arm $arm: arm env did not reach the engine (see descriptor)";
         echo "env-mismatch" > "$dir/INVALID"; INVALID="$INVALID $arm:env-mismatch"; return 1; }

  if ! "$PY" "$W/tools/dcu/s23_smoke.py" --out "$dir/smoke.json" \
      >> "$dir/service.log" 2>&1; then
    log "ABORT arm $arm: smoke failed"
    tail -5 "$dir/service.log"
    echo "smoke-failed" > "$dir/INVALID"; INVALID="$INVALID $arm:smoke"; return 1
  fi
  grep -E "^\[smoke\]" "$dir/service.log" | tail -3

  log "probe c=1 arm=$arm repeat=$REPEAT"
  "$PY" "$W/tools/dcu/s23_probe.py" --prompt-tokens "$PROMPT_TOKENS" \
    --max-tokens "$MAX_TOKENS" --repeat "$REPEAT" \
    --tag "${WINDOW}_${arm}" --out "$dir" > "$dir/probe.log" 2>&1
  grep -E "^\[(run|SUMMARY|saved)" "$dir/probe.log" | tail -6

  if [ "$DO_PARITY" = "1" ]; then
    mkdir -p "$dir/parity"
    if [ -z "$PARITY_REF" ]; then
      log "parity baseline capture (arm $arm)"
      "$PY" "$W/tools/dcu/s23_parity.py" --mode capture \
        --out "$dir/parity/${arm}.json" > "$dir/parity/capture.log" 2>&1
      grep -E "^\[(capture|parity)" "$dir/parity/capture.log" | tail -9
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
      grep -E "^\[(check|parity)" "$dir/parity/check.log" | tail -9
    fi
  fi

  if [ "$ARM_CENSUS" = "1" ]; then
    run_census "$dir"
  fi

  if [ -n "$SWEEP" ] && [ "$ARM_SWEEP" = "on" ]; then
    local r
    for r in $(seq 1 "$SWEEP_REPS"); do
      log "sweep c=$SWEEP arm=$arm rep=$r/$SWEEP_REPS"
      "$PY" "$W/tools/dcu/s23_sweep.py" --concurrency "$SWEEP" \
        --tag "${WINDOW}_${arm}_r${r}" --out "$dir" \
        > "$dir/sweep-r${r}.log" 2>&1
      grep -E "^\[c=" "$dir/sweep-r${r}.log" | tail -4
    done
  fi
  return 0
}

# ------------------------------------------------------------------- restore
restore_default() {
  [ "$RESTORED" = "1" ] && return 0
  if [ "$DRY_RUN" = "1" ]; then RESTORED=1; return 0; fi
  [ "$DO_RESTORE" = "1" ] || { log "WARN --no-restore: leaving last arm env"; RESTORED=1; return 0; }
  RESTORED=1
  local dir=$WINDOW_DIR/restore
  mkdir -p "$dir"
  log "restoring default configuration"
  local t0 t_so t_ready t_smoke
  t0=$(date +%s%N)
  local so_before
  so_before=$(so_md5_live)
  if rollback_so "$BASELINE_SO_MD5"; then :; else log "so restored to $BASELINE_SO_MD5"; fi
  t_so=$(date +%s%N)
  local from
  from=$(wc -l < "$LOGFILE" 2>/dev/null || echo 0)
  snapshot_props "$dir" before
  if ! start_service "$LAUNCHER_DEFAULT" ""; then
    log "FATAL: could not restart the production launcher -- MANUAL ACTION NEEDED"
    echo "restore-start-failed" > "$dir/INVALID"
    INVALID="$INVALID restore:start-failed"
    return 1
  fi
  snapshot_props "$dir" after
  local env_final
  env_final=$(unit_env_line "$dir/unit-props-after.txt")
  if [ "$env_final" = "HOST=0.0.0.0" ]; then
    log "default unit Environment reproduced exactly: [$env_final]"
  else
    log "FATAL: default unit Environment is [${env_final}], expected [HOST=0.0.0.0]"
    echo "restore-env-drift [$env_final]" > "$dir/INVALID"
    INVALID="$INVALID restore:env-drift"
  fi
  diff -u "$WINDOW_DIR/control/unit-props-before.txt" "$dir/unit-props-after.txt" \
    > "$dir/unit-props-diff-vs-window-start.txt" 2>&1 || true
  if wait_ready 150; then
    t_ready=$(date +%s%N)
    local pid smoke_args=""
    pid=$(service_pid)
    "$PY" "$W/tools/dcu/s23_descriptor.py" --out "$dir/descriptor.json" \
      --window "$WINDOW" --arm restore --desc "default configuration" \
      --so-md5 "$(so_md5_live)" --pid "$pid" --arm-env "" \
      --extra "launcher=$LAUNCHER_DEFAULT" --unit-props "$dir/unit-props-after.txt" \
      >> "$dir/service.log" 2>&1
    local ref_smoke=$WINDOW_DIR/$REF_ARM/smoke.json
    [ -f "$ref_smoke" ] && smoke_args="--expect-json $ref_smoke"
    # shellcheck disable=SC2086
    "$PY" "$W/tools/dcu/s23_smoke.py" --out "$dir/smoke.json" $smoke_args \
      >> "$dir/service.log" 2>&1 && log "default restored: ready + smoke ok"
  else
    log "FATAL: service did not come back on the default configuration"
    show_new_log "$from" | tail -20
    echo "restore-not-ready" > "$dir/INVALID"
    INVALID="$INVALID restore"
  fi
  t_smoke=$(date +%s%N)
  {
    echo "so_before=$so_before"
    echo "so_after=$(so_md5_live)"
    echo "baseline_expected=$BASELINE_SO_MD5"
    echo "so_swap_ms=$(( (t_so - t0) / 1000000 ))"
    echo "restart_to_ready_ms=$(( (${t_ready:-$t_smoke} - t_so) / 1000000 ))"
    echo "ready_to_smoke_ms=$(( (t_smoke - ${t_ready:-$t_smoke}) / 1000000 ))"
    echo "total_rollback_s=$(( (t_smoke - t0) / 1000000000 ))"
  } > "$dir/rollback-timing.txt"
  log "rollback timing: $(tr '\n' ' ' < "$dir/rollback-timing.txt")"
  echo "$(date -Is)" > "$dir/window-end.txt"
}

finish() {
  local rc=$?
  trap - EXIT INT TERM
  restore_default
  "$PY" "$W/tools/dcu/s23_report.py" "$WINDOW_DIR" --control "$REF_ARM" || true
  log "window $WINDOW done (rc=$rc) invalid='$INVALID'"
  [ -n "$INVALID" ] && exit 1
  exit "$rc"
}
trap finish EXIT INT TERM

# ---------------------------------------------------------------------- main
log "window $WINDOW start; arms: ${ARMS_REQ[*]}; so_live=$(so_md5_live)"
log "gpu lock held by this window for its whole duration"
{
  echo "window=$WINDOW"
  echo "start=$WINDOW_START"
  echo "arms=${ARMS_REQ[*]}"
  echo "repeat=$REPEAT sweep=$SWEEP prompt_tokens=$PROMPT_TOKENS max_tokens=$MAX_TOKENS"
  echo "baseline_so_md5=$BASELINE_SO_MD5"
  echo "git_rev=$(git -C "$W" rev-parse HEAD)"
  echo "lock_holder=$$"
} > "$WINDOW_DIR/window.txt"
git -C "$W" status --short > "$WINDOW_DIR/git-status.txt"
clear_manager_env
enforce_so_baseline
[ "$UNIT_GATE" = "1" ] && run_unit_gate

for arm in "${ARMS_REQ[@]}"; do
  run_arm "$arm" || log "arm $arm did not complete; continuing"
done
