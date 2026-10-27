#!/usr/bin/env bash
# S14e: run one grid cell (N sequential engine starts, same cache root) and
# compare every pair of its recordings bit-for-bit.
#
#   S14_CG_MODE=... S14_CG_SIZES="1 2 4 8" S14_UTIL=0.9 \
#     tools/dcu/s14e_drive.sh <CELL> <TP> <SW> <CACHE> <PORT> <MAXLEN> <DEVICES> <NRUNS>
#
# Sequential by construction: discipline #3 forbids two engines at once, and a
# second engine on cards 2/3 would perturb the cell under test.
set -u
CELL=$1; TP=$2; SW=$3; CACHE=$4; PORT=$5; MAXLEN=$6; DEVICES=$7; NRUNS=${8:-2}
MODEL="${S14E_MODEL:-/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8}"
OUT="${S14_BENCH_OUT:-/mnt/game/s14e}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS=()
for i in $(seq 0 $((NRUNS-1))); do
  n=$(printf "%s%s" "$CELL" "$(printf "\\$(printf '%03o' $((97+i)))")")
  RUNS+=("$n")
  echo "===== [drive] cell=$CELL run=$n ($((i+1))/$NRUNS) $(date -Is) ====="
  bash "$HERE/s14e_run_one.sh" "$n" "$MODEL" "$TP" "$SW" "$CACHE" "$PORT" "$MAXLEN" "$DEVICES" \
    || { echo "[drive] run $n FAILED"; }
  tail -6 "$OUT/hostlogs/$n.log"
done
echo "===== [drive] cell=$CELL comparisons $(date -Is) ====="
for i in $(seq 0 $((NRUNS-1))); do
  for j in $(seq $((i+1)) $((NRUNS-1))); do
    a="${RUNS[$i]}"; b="${RUNS[$j]}"
    [ -s "$OUT/probe-$a.jsonl" ] && [ -s "$OUT/probe-$b.jsonl" ] || { echo "SKIP $a vs $b (missing jsonl)"; continue; }
    echo "----- $a vs $b -----"
    python3 "$HERE/s14e_compare.py" "$OUT/probe-$a.jsonl" "$OUT/probe-$b.jsonl" \
      --label-a "$a" --label-b "$b" --csv "$OUT/cmp-$a-vs-$b.csv" | tee "$OUT/cmp-$a-vs-$b.txt"
  done
done
echo "===== [drive] cell=$CELL done $(date -Is) ====="
