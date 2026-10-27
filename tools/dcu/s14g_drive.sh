#!/usr/bin/env bash
# S14g: run one cell (N sequential engine starts, same cache root) and cluster
# the resulting greedy token streams into variants.
#
#   S14G_PIN_PATH=/mnt/game/s14g/pin TRITON_PRINT_AUTOTUNING=1 \
#   S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
#     tools/dcu/s14g_drive.sh <CELL> <TP> <SW> <CACHE> <PORT> <MAXLEN> <DEVICES> <NRUNS>
set -u
CELL=$1; TP=$2; SW=$3; CACHE=$4; PORT=$5; MAXLEN=$6; DEVICES=$7; NRUNS=${8:-2}
MODEL="${S14E_MODEL:-/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8}"
OUT="${S14_BENCH_OUT:-/mnt/game/s14g}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS=()
for i in $(seq 0 $((NRUNS-1))); do
  n=$(printf "%s%s" "$CELL" "$(printf "\\$(printf '%03o' $((97+i)))")")
  RUNS+=("$n")
  echo "===== [drive] cell=$CELL run=$n ($((i+1))/$NRUNS) $(date -Is) ====="
  bash "$HERE/s14g_run_one.sh" "$n" "$MODEL" "$TP" "$SW" "$CACHE" "$PORT" "$MAXLEN" "$DEVICES" \
    || echo "[drive] run $n FAILED"
  tail -4 "$OUT/hostlogs/$n.log"
done
echo "===== [drive] cell=$CELL flip clustering $(date -Is) ====="
FLIPS=()
for n in "${RUNS[@]}"; do [ -s "$OUT/flip-$n.jsonl" ] && FLIPS+=("$OUT/flip-$n.jsonl"); done
[ ${#FLIPS[@]} -gt 0 ] && python3 "$HERE/s14g_classify.py" "${FLIPS[@]}" | tee "$OUT/variants-$CELL.txt"
echo "===== [drive] cell=$CELL e14 comparisons $(date -Is) ====="
for i in $(seq 0 $((NRUNS-1))); do
  for j in $(seq $((i+1)) $((NRUNS-1))); do
    a="${RUNS[$i]}"; b="${RUNS[$j]}"
    [ -s "$OUT/probe-$a.jsonl" ] && [ -s "$OUT/probe-$b.jsonl" ] || continue
    echo "----- $a vs $b -----"
    python3 "$HERE/s14e_compare.py" "$OUT/probe-$a.jsonl" "$OUT/probe-$b.jsonl" \
      --label-a "$a" --label-b "$b" --csv "$OUT/cmp-$a-vs-$b.csv" | tail -20 | tee "$OUT/cmp-$a-vs-$b.txt"
  done
done
echo "===== [drive] cell=$CELL done $(date -Is) ====="
