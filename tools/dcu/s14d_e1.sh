#!/usr/bin/env bash
# S14d E1: H-AOT 2x2 -- {live torch.compile, AOT disk reload} x {combo kernels on, off}
#          + the extra starts the task book asks for.
#
# Cell A (combo on, default): W4(live) / S4(reload) from s14d_e4.sh, plus
#   A3 = a second independent RELOAD of the same cache under the byte-identical
#   production shape (8 GiB total swap, swappiness 200) -> E2
#   ("same cache, same request order, run twice") falls out of S4 vs A3.
# Cell B (combo off): fresh cache root (the two switches change the AOT key),
#   B1 = live compile, B2 = AOT reload.  Because S14c measured combo-off as the
#   *hungriest* variant (peak 27430 MiB, minAvail 0.20 GiB on 20.5 GiB swap),
#   cell B temporarily attaches the 16 GiB /mnt/game/swapfile (total ~20 GiB)
#   so the arm cannot die of OOM and the live-vs-reload comparison stays clean.
set -u
cd /home/lyh/z100-vllm
LOGDIR=/mnt/game/s14d/hostlogs
mkdir -p "$LOGDIR"
MDL=/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8
COMBOK='{"combo_kernels": false, "benchmark_combo_kernel": false}'
export S14C_LOGDIR=$LOGDIR

echo "[e1] $(date -Is) start; swaps:"; cat /proc/swaps

# ---------------------------------------------------------------- cell B (combo off)
tools/dcu/s14c_swap.sh on /mnt/game/swapfile 16384 E1-combooff
cat /proc/swaps; free -m

S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
  S14C_INDUCTOR_CFG="$COMBOK" \
  S14C_TAG="e1_b1_tp4_graph_combokernels_off_livecompile_swap20g" S14D_HOLD_S=10 READY_TIMEOUT=3600 \
  S14_ENGINE_TIMEOUT=5400 S14D_LOGDIR=$LOGDIR \
  tools/dcu/s14d_run_one.sh B1 "$MDL" 4 1 /mnt/game/s14d/cache-b 8611 262144 0,1,2,3 probe
echo "[e1] B1 finished $(date -Is)"; free -m; cat /proc/swaps

S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
  S14C_INDUCTOR_CFG="$COMBOK" \
  S14C_TAG="e1_b2_tp4_graph_combokernels_off_AOTreload_swap20g" S14D_HOLD_S=10 READY_TIMEOUT=3600 \
  S14_ENGINE_TIMEOUT=5400 S14D_LOGDIR=$LOGDIR \
  tools/dcu/s14d_run_one.sh B2 "$MDL" 4 1 /mnt/game/s14d/cache-b 8612 262144 0,1,2,3 probe
echo "[e1] B2 finished $(date -Is)"; free -m; cat /proc/swaps

# B3 = second reload of the combo-off cache.  After E4/E2 showed that two reloads
# of the combo-ON cache already differ 12/12, the only remaining hope for graph
# is that turning benchmark_combo_kernel/combo_kernels OFF makes the reload path
# deterministic; B2 vs B3 tests exactly that (two starts of the same cache).
S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
  S14C_INDUCTOR_CFG="$COMBOK" \
  S14C_TAG="e1_b3_tp4_graph_combokernels_off_AOTreload2_swap20g" S14D_HOLD_S=10 READY_TIMEOUT=3600 \
  S14_ENGINE_TIMEOUT=5400 S14D_LOGDIR=$LOGDIR \
  tools/dcu/s14d_run_one.sh B3 "$MDL" 4 1 /mnt/game/s14d/cache-b 8614 262144 0,1,2,3 probe
echo "[e1] B3 finished $(date -Is)"; free -m; cat /proc/swaps

tools/dcu/s14c_swap.sh off /mnt/game/swapfile E1-combooff-end
cat /proc/swaps

# ---------------------------------------------------------------- comparisons
C="python3 tools/dcu/s14d_compare.py"
{ echo "===== E1 cell A: live(W4) vs reload(S4) [combo ON, prod shape] ====="
  $C /mnt/game/s14d/probe-S4.jsonl /mnt/game/s14d/probe-W4.jsonl --label-a S4-reload --label-b W4-live --csv /mnt/game/s14d/e1-A-live-vs-reload.csv
  echo; echo "===== E1 cell A second start: reload(S4) vs reload(A3) [combo ON, prod shape] ====="
  $C /mnt/game/s14d/probe-A3.jsonl /mnt/game/s14d/probe-S4.jsonl --label-a A3-reload2 --label-b S4-reload1 --csv /mnt/game/s14d/e1-A-reload-vs-reload.csv
  echo; echo "===== E1 cell B: live(B1) vs reload(B2) [combo OFF] ====="
  $C /mnt/game/s14d/probe-B2.jsonl /mnt/game/s14d/probe-B1.jsonl --label-a B2-reload --label-b B1-live --csv /mnt/game/s14d/e1-B-live-vs-reload.csv
  echo; echo "===== E1 cell B: reload(B2) vs reload(B3) [combo OFF, two starts of one cache] ====="
  $C /mnt/game/s14d/probe-B3.jsonl /mnt/game/s14d/probe-B2.jsonl --label-a B3-reload2 --label-b B2-reload1 --csv /mnt/game/s14d/e1-B-reload-vs-reload.csv
  echo; echo "===== combo ON vs OFF, live: W4 vs B1 ====="
  $C /mnt/game/s14d/probe-B1.jsonl /mnt/game/s14d/probe-W4.jsonl --label-a B1-combooff-live --label-b W4-comboon-live
  echo; echo "===== combo ON vs OFF, reload: S4 vs B2 ====="
  $C /mnt/game/s14d/probe-B2.jsonl /mnt/game/s14d/probe-S4.jsonl --label-a B2-combooff-reload --label-b S4-comboon-reload
} 2>&1 | tee "$LOGDIR/E1-compare.txt"
echo "[e1] $(date -Is) done"; cat /proc/swaps; free -m
