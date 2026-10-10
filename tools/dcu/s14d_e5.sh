#!/usr/bin/env bash
# S14d E5: close the "C2 vs P4e 12%" gap -- TP=4 EAGER decode under two host memory
# shapes, everything else identical.
#   arm SW16 = 16 GiB /mnt/game/swapfile attached (total ~20 GiB)  <- matches S14 P4e
#   arm SW4  = only the 4 GiB /swap.img                            <- matches S14c C2
# vm.swappiness = 60 for both (the vintage of P4e and C2).
set -u
cd /home/lyh/z100-vllm
LOGDIR=/mnt/game/s14d/hostlogs
mkdir -p "$LOGDIR"
MDL=/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8
export S14C_LOGDIR=$LOGDIR

sudo -n sysctl -w vm.swappiness=60
echo "[e5] $(date -Is) start; swappiness=$(cat /proc/sys/vm/swappiness)"; cat /proc/swaps

# ---- arm SW16: eager + 16 GiB swapfile -----------------------------------
tools/dcu/s14c_swap.sh on /mnt/game/swapfile 16384 E5-sw16
cat /proc/swaps; free -m
S14C_CT=z100-s14 S14_EAGER=1 \
  S14C_TAG="e5_sw16_tp4_eager_swap20g_sw60" S14D_HOLD_S=10 READY_TIMEOUT=1800 \
  S14_ENGINE_TIMEOUT=3600 S14D_LOGDIR=$LOGDIR \
  tools/dcu/s14d_run_one.sh E5SW16 "$MDL" 4 1 /mnt/game/s14d/cache-e5-sw16 8621 262144 0,1,2,3 conc2,probe,decode
echo "[e5] SW16 finished $(date -Is)"; free -m; cat /proc/swaps

tools/dcu/s14c_swap.sh off /mnt/game/swapfile E5-sw16-end
cat /proc/swaps

# ---- arm SW4: eager + only /swap.img --------------------------------------
S14C_CT=z100-s14 S14_EAGER=1 \
  S14C_TAG="e5_sw4_tp4_eager_swapimg4g_sw60" S14D_HOLD_S=10 READY_TIMEOUT=1800 \
  S14_ENGINE_TIMEOUT=3600 S14D_LOGDIR=$LOGDIR \
  tools/dcu/s14d_run_one.sh E5SW4 "$MDL" 4 1 /mnt/game/s14d/cache-e5-sw4 8622 262144 0,1,2,3 conc2,probe,decode
echo "[e5] SW4 finished $(date -Is)"; free -m; cat /proc/swaps

echo "===== E5 decode summary ====="
for a in E5SW16 E5SW4; do
  echo "--- $a ---"
  grep -E "Median TPOT|Successful|Mean TPOT|P90|P99" "$LOGDIR/$a-bench-decode.log" 2>/dev/null || echo "(no bench log)"
  grep -E "ready=|min MemAvailable|max swap_used" "$LOGDIR/$a.log" 2>/dev/null
done 2>&1 | tee "$LOGDIR/E5-compare.txt"
echo "[e5] $(date -Is) done"; free -m; cat /proc/swaps; cat /proc/sys/vm/swappiness
