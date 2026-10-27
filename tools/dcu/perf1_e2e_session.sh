#!/usr/bin/env bash
# Run several perf1 arms sequentially inside ONE build-lock session (container).
#
#   docker exec z100-vllm bash -c 'flock /tmp/perf1_build.lock \
#       bash /src/tools/dcu/perf1_e2e_session.sh \
#         --arm control --arm k3 S11F_MTP_TOKENS=3 --arm control_restore' \
#       > /mnt/game/perf1/arm_session.log 2>&1
#
# The caller holds /tmp/perf1_build.lock for the whole session, so no rebuild
# can swap the .so under a running arm.  The LAST arm listed is the one left
# running on :8000 -- always make it a known-good (production) arm.
set -u
cd /src || exit 1
[ $# -gt 0 ] || { sed -n '2,12p' "$0"; exit 2; }

declare -a names=() kvs=()
cur=-1
for a in "$@"; do
  if [ "$a" = "--arm" ]; then
    names+=(""); kvs+=(""); cur=$((cur + 1))
  elif [ "$cur" -lt 0 ]; then
    echo "error: expected --arm <name> first (got '$a')"; exit 2
  elif [ -z "${names[$cur]}" ]; then
    names[$cur]="$a"
  else
    kvs[$cur]="${kvs[$cur]} $a"
  fi
done

PERF1_DIR="${PERF1_DIR:-/mnt/game/perf1}"
PREFLIGHT_WAIT="${PERF1_PREFLIGHT_WAIT:-300}"

# GPU-exclusive preflight: other teammates' microbenchmarks (torchrun probes,
# s18/s11 probes) do NOT take the build lock, so the session refuses to start
# while one is running -- concurrent GPU load invalidates every arm.
# Own ancestors are excluded: a launcher whose command line merely *mentions*
# one of these patterns (e.g. a wrapper echo) must not block the session.
gpu_busy() {
  local exclude=" " p=$$
  while [ -n "$p" ] && [ "$p" != "1" ] && [ -r "/proc/$p/stat" ]; do
    exclude="$exclude$p "
    p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null)
  done
  pgrep -af 'torchrun|perf1_comm_ar|s18_.*probe|s11.*probe|gemv_probe|hipprof' 2>/dev/null \
    | while read -r pid rest; do
        case "$exclude" in *" $pid "*) ;; *) echo "$pid $rest" ;; esac
      done
}
t0=$(date +%s)
while [ -n "$(gpu_busy)" ]; do
  if [ $(( $(date +%s) - t0 )) -ge "$PREFLIGHT_WAIT" ]; then
    echo "=== [session] ABORT: GPU still busy after ${PREFLIGHT_WAIT}s ==="
    gpu_busy
    exit 4
  fi
  echo "=== [session] waiting for exclusive GPU (${#names[@]} arms queued) ==="
  gpu_busy
  sleep 10
done
echo "=== [session] GPU exclusive, starting $(date -Is) ==="

echo "=== [session] $(date -Is) arms: ${names[*]} ==="
for i in "${!names[@]}"; do
  arm="${names[$i]}"; kv="${kvs[$i]:-}"
  if [ "${arm#restore:}" != "$arm" ]; then
    # start-only group: leaves this arm live without measuring it
    arm="${arm#restore:}"
    echo "=== [session] arm $((i + 1))/${#names[@]}: restore-only -> $arm $kv ==="
    # shellcheck disable=SC2086
    bash tools/dcu/perf1_arm.sh start "$arm" $kv || echo "[session] WARN restore of $arm returned $?"
    bash tools/dcu/perf1_arm.sh status
    continue
  fi
  echo "=== [session] arm $((i + 1))/${#names[@]}: $arm $kv ==="
  # Bare tokens select the measurement shape for this group:
  #   anchor      -> p512/g64 anchor protocol only (attribution arms)
  #   full        -> cold probe + 4-length sweep + parity + anchor
  #   parity_only -> greedy parity capture only (reference arms)
  #   warmup      -> warm the prefill shapes before measuring
  #   decodewarm  -> also run one discarded 16-token decode request first
  export PERF1_PARITY_ONLY=0 PERF1_WARMUP=0 PERF1_DECODE_WARMUP=0 PERF1_MODE=full
  case " $kv " in *" parity_only "*) export PERF1_PARITY_ONLY=1 ;; esac
  case " $kv " in *" anchor "*) export PERF1_MODE=anchor ;; esac
  case " $kv " in *" warmup "*) export PERF1_WARMUP=1 ;; esac
  case " $kv " in *" decodewarm "*) export PERF1_DECODE_WARMUP=1 ;; esac
  kv="$(echo " $kv " | sed -e 's/ parity_only / /g' -e 's/ anchor / /g' \
      -e 's/ full / /g' -e 's/ warmup / /g' -e 's/ decodewarm / /g')"
  # shellcheck disable=SC2086
  bash tools/dcu/perf1_e2e_suite.sh "$arm" $kv
  rc=$?
  echo "=== [session] arm $arm rc=$rc (parity_only=$PERF1_PARITY_ONLY) ==="
  # Canary rule: if the FIRST (production-flags) arm cannot even serve, the
  # tree/build is broken and every later arm would only burn another ~3.5 min
  # of model load.  Abort the whole set instead.
  if [ "$i" -eq 0 ] && [ "$rc" -ne 0 ]; then
    echo "=== [session] ABORT: canary arm '$arm' failed (rc=$rc); tree/build is broken ==="
    exit 5
  fi
done

echo "=== [session] artifacts (md5 + size) ==="
for f in "$PERF1_DIR"/*_e2e.json "$PERF1_DIR"/*_anchor_p512_g64.json; do
  [ -f "$f" ] || continue
  md5sum "$f"
done
echo "=== [session] so md5 per arm ==="
cat "$PERF1_DIR"/arm_*_so.txt 2>/dev/null
echo "=== [session] live service ==="
bash tools/dcu/perf1_arm.sh status
echo "=== [session] done $(date -Is) ==="
