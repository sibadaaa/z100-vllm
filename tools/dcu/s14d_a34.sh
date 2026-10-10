#!/usr/bin/env bash
# S14d: the pivotal control for E4.
#
# E4 showed W4 (live compile) vs S4 (AOT reload) differ on 12/12 prompts with
# ~1.4 nat mean same-context logit difference.  That alone cannot tell whether
# the factor is "live vs reload" or "any two graph runs".
#
# A3 / A4 = the 2nd and 3rd AOT reload of the *same* cache root under the
# byte-identical production shape (8 GiB swap, swappiness=200) and the same
# request battery.  S4 vs A3 vs A4 therefore isolates run-to-run volatility of
# the reload path -- exactly the path a production restart would take.
set -u
cd /home/lyh/z100-vllm
LOGDIR=/mnt/game/s14d/hostlogs
MDL=/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8
export S14C_LOGDIR=$LOGDIR

echo "[a34] $(date -Is) start; swappiness=$(cat /proc/sys/vm/swappiness)"; cat /proc/swaps

for arm in A3 A4; do
  S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
    S14C_TAG="e2_${arm}_tp4_graph_combokernels_on_AOTreload_swap8g_sw200" S14D_HOLD_S=10 READY_TIMEOUT=3600 \
    S14_ENGINE_TIMEOUT=5400 S14D_LOGDIR=$LOGDIR \
    tools/dcu/s14d_run_one.sh "$arm" "$MDL" 4 1 /mnt/game/s14d/cache-e4 8613 262144 0,1,2,3 probe
  echo "[a34] $arm finished $(date -Is)"; free -m; cat /proc/swaps
done

C="python3 tools/dcu/s14d_compare.py"
{ echo "===== E2/E1-A: reload(S4) vs reload(A3) [same cache, same shape] ====="
  $C /mnt/game/s14d/probe-A3.jsonl /mnt/game/s14d/probe-S4.jsonl --label-a A3-reload2 --label-b S4-reload1 --csv /mnt/game/s14d/e2-S4-vs-A3.csv
  echo; echo "===== E2: reload(A4) vs reload(A3) ====="
  $C /mnt/game/s14d/probe-A4.jsonl /mnt/game/s14d/probe-A3.jsonl --label-a A4-reload3 --label-b A3-reload2 --csv /mnt/game/s14d/e2-A3-vs-A4.csv
  echo; echo "===== E2: reload(A4) vs reload(S4) ====="
  $C /mnt/game/s14d/probe-A4.jsonl /mnt/game/s14d/probe-S4.jsonl --label-a A4-reload3 --label-b S4-reload1 --csv /mnt/game/s14d/e2-S4-vs-A4.csv
} 2>&1 | tee "$LOGDIR/E2-reload-vs-reload.txt"
echo "[a34] $(date -Is) done"; free -m; cat /proc/swaps
