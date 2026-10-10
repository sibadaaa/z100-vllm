#!/usr/bin/env bash
# S14g H1 cell: ASLR experiment in the seccomp-unconfined sibling container.
#
#   A1, A2 = ADDR_NO_RANDOMIZE (setarch x86_64 -R), A3 = control (ASLR on)
#
# All three use the identical memory-saving graph recipe, the same AOT/triton
# cache as S14e G2, the same prompt battery, and the worker-level autotune hook
# (/mnt/game/s14g/hook, installed as a site-packages .pth in z100-s14g) so the
# raw per-run autotuner decisions are recorded.
set -u
export S14_CT=z100-s14g
export S14_CG_MODE=PIECEWISE
export S14_CG_SIZES="1 2 4 8"
export S14_UTIL=0.9
export S14G_TASKS="${S14G_TASKS:-flip,e14}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${S14_BENCH_OUT:-/mnt/game/s14g}"
MODEL="${S14E_MODEL:-/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8}"
CACHE="${S14G_CACHE:-/mnt/game/s14e/g2}"
PORT="${S14G_PORT:-8641}"

SPECS="${S14G_ASPECS:-A1:off A2:off A3:on}"
NAMES=()
for spec in $SPECS; do
  n="${spec%%:*}"; mode="${spec##*:}"
  NAMES+=("$n")
  if [ "$mode" = "off" ]; then PREFIX="setarch x86_64 -R"; else PREFIX=""; fi
  echo "===== [acell] run=$n aslr=$mode prefix='$PREFIX' $(date -Is) ====="
  S14G_EXEC_PREFIX="$PREFIX" bash "$HERE/s14g_run_one.sh" \
    "$n" "$MODEL" 2 1 "$CACHE" "$PORT" 4096 0,1 || echo "[acell] run $n FAILED"
  tail -5 "$OUT/hostlogs/$n.log"
  sleep 20   # let the previous engine release /dev/shm and VRAM before the next
done

echo "===== [acell] variants $(date -Is) ====="
FLIPS=()
for n in "${NAMES[@]}"; do [ -s "$OUT/flip-$n.jsonl" ] && FLIPS+=("$OUT/flip-$n.jsonl"); done
[ ${#FLIPS[@]} -gt 0 ] && python3 "$HERE/s14g_classify.py" "${FLIPS[@]}" | tee "$OUT/variants-A.txt"

echo "===== [acell] e14 comparisons $(date -Is) ====="
for i in "${!NAMES[@]}"; do
  for j in "${!NAMES[@]}"; do
    [ "$j" -le "$i" ] && continue
    a="${NAMES[$i]}"; b="${NAMES[$j]}"
    [ -s "$OUT/probe-$a.jsonl" ] && [ -s "$OUT/probe-$b.jsonl" ] || continue
    echo "----- $a vs $b -----"
    python3 "$HERE/s14e_compare.py" "$OUT/probe-$a.jsonl" "$OUT/probe-$b.jsonl" \
      --label-a "$a" --label-b "$b" --csv "$OUT/cmp-$a-vs-$b.csv" | tail -18 | tee "$OUT/cmp-$a-vs-$b.txt"
  done
done

echo "===== [acell] address fingerprints ====="
for n in "${NAMES[@]}"; do echo "--- $n"; grep -E "^== pid=" "$OUT/maps-$n.txt" 2>/dev/null | sed 's/vmpeak_kB=[0-9]*/vmpeak/'; done
echo "===== [acell] hook (worker-level autotune decisions) ====="
for n in "${NAMES[@]}"; do
  f="$OUT/autotune-log-$n.jsonl"
  echo "--- $n imports=$(grep -c '"event": "import"' "$f" 2>/dev/null || echo 0) autotune=$(grep -c '"event": "autotune"' "$f" 2>/dev/null || echo 0)"
  python3 - "$f" <<'PY' 2>/dev/null
import json, sys
seen = {}
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    if r.get("event") == "autotune":
        seen.setdefault(r["kernel"], r["config"])
for k in sorted(seen):
    print(f"   {k}: {seen[k]}")
PY
done
echo "===== [acell] done $(date -Is) ====="
