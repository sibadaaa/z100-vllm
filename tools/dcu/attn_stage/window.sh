#!/usr/bin/env bash
# GPU window for the attn stage probe: stop the MTP k=2 production unit, run
# one command with the variant .so symlinked, restore the production unit and
# verify it came back with the same policy fingerprint.
#
#   bash tools/dcu/attn_stage/window.sh <variant.so> <log> -- <cmd...>
#
# Why a window: the probe measures one kernel's stage split at real shapes; the
# serving engine owns all four DCUs and any co-tenant kernel pollutes both the
# wall time and the per-CTA clock deltas.
set -u
REPO=/home/lyh/z100-vllm
VARIANT=$(readlink -m "${1:?variant .so}")
LOG=$(readlink -m "${2:?log}")
shift 2
[ "${1:-}" = "--" ] && shift
UNIT=z100-vllm-mtp2
# restore whatever the tree pointed at when the window opened (the live arm)
PROD_SO=$(readlink -f "$REPO/vllm/_rocm_C.abi3.so")
echo "[win] production arm to restore: $PROD_SO"
API_KEY_FILE=/mnt/game/host-serve/api-key
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "[win] ===== start $(date -Is) variant=$VARIANT ====="
health() { curl -s -o /dev/null -w "%{http_code}" --max-time 3 \
  http://127.0.0.1:8000/health || true; }
kill_serving() {
  sudo systemctl stop "$UNIT" 2>/dev/null || true
  pkill -f "entrypoints[.]cli" 2>/dev/null || true
  for _ in $(seq 1 40); do
    pgrep -f "entrypoints[.]cli" >/dev/null || break; sleep 2
  done
  pkill -9 -f 'VLLM[:]:Worker' 2>/dev/null || true
  sleep 3
}
restore() {
  local rc=$?
  echo "[win] restoring $UNIT (rc=$rc)"
  ln -sfn "$PROD_SO" "$REPO/vllm/_rocm_C.abi3.so"
  sudo systemctl start "$UNIT" || echo "[win] ERROR start $UNIT"
  local c=""
  for i in $(seq 1 120); do
    sleep 5
    c=$(health)
    [ "$c" = "200" ] && break
  done
  echo "[win] restore health=$c"
  echo "[win] so=$(readlink -f "$REPO/vllm/_rocm_C.abi3.so")"
  echo "[win] md5=$(md5sum "$REPO/vllm/_rocm_C.abi3.so" | cut -c1-12)"
  echo "[win] ===== end $(date -Is) ====="
  return "$rc"
}
trap restore EXIT

echo "[win] production so md5 before: $(md5sum "$REPO/vllm/_rocm_C.abi3.so" | cut -c1-12)"
kill_serving
pgrep -af "entrypoints[.]cli" || echo "[win] production fully stopped"
ln -sfn "$VARIANT" "$REPO/vllm/_rocm_C.abi3.so"
echo "[win] probe so md5: $(md5sum "$VARIANT" | cut -c1-12)"

cd "$REPO"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export DTK_ROOT=/mnt/game/host-runtime/dtk2604
export PYTHONPATH=$REPO
set +u
# shellcheck disable=SC1091
source tools/dcu/env.sh
set -u
"$@"
rc=$?
echo "[win] command rc=$rc"
exit "$rc"
