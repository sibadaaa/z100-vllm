#!/usr/bin/env bash
# S13 phase C: repeat the vendor (production-stack) fair-baseline arm N times in
# one fabric state, so the "ACS off makes the production stack ~5.5% slower"
# observation from S12 (n=2, in-state spread 6.6%) can be settled.
#
#   tools/dcu/s13_vendor_batch.sh <TAG> <FIRST> <LAST>
#   e.g. tools/dcu/s13_vendor_batch.sh S13VOFF 1 4
#
# Every run uses cards 0,1,2,3 explicitly (hard constraint 25) and the same
# production recipe as S12 (TP=4, eager, MAXLEN 262144, CUSTOM_AR=0).
set -u
TAG=$1; FIRST=$2; LAST=$3
WS=/home/lyh/z100-vllm
for i in $(seq "$FIRST" "$LAST"); do
  NAME="${TAG}${i}"
  echo "=================== $NAME start $(date -Is) ==================="
  bash "$WS/tools/dcu/s13_vendor_arm.sh" "$NAME" 4 8060 262144 0,1,2,3
  echo "=================== $NAME done rc=$? $(date -Is) ==================="
  sleep 15
done
python3 "$WS/tools/dcu/s12_bench_table.py" "/mnt/game/s13/hostlogs/vendor-${TAG}*-bench-*.log" 2>/dev/null | head -40
echo "VENDOR_BATCH_DONE tag=$TAG"
