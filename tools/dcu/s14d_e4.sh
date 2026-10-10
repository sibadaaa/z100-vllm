#!/usr/bin/env bash
# S14d E4: production-recipe warmup effectiveness.
#
# Full production flow on one cache root:
#   W4 = empty cache -> live torch.compile + AOT artefacts written to disk (the
#        "warmup" run) -> probe battery -> stop
#   S4 = same cache root -> AOT reload (the "service" run) -> same probe battery
#        -> stop
# Then: bit-level comparison of the two token streams.
#
# Both arms run under the S14c production recipe: TP=4 graph
# (PIECEWISE + --cudagraph-capture-sizes 1 2 4 8), AOT cache, additional 4 GiB
# swapfile (total 8 GiB), vm.swappiness=200, production flags, no --enforce-eager.
set -u
cd /home/lyh/z100-vllm
LOGDIR=/mnt/game/s14d/hostlogs
mkdir -p "$LOGDIR"
MDL=/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8
CACHE=/mnt/game/s14d/cache-e4

echo "[e4] $(date -Is) start"
cat /proc/swaps; free -m

# ---- host memory shape: additional 4 GiB swapfile (total 8 GiB) + swappiness 200
export S14C_LOGDIR=$LOGDIR
tools/dcu/s14c_swap.sh on /mnt/game/s14d/swap4 4096 E4-setup
sudo -n sysctl -w vm.swappiness=200
cat /proc/sys/vm/swappiness; cat /proc/swaps

# ---- W4: live compile (empty cache) = the "warmup" run --------------------
S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
  S14C_TAG="e4_w4_tp4_graph_livecompile_swap8g_sw200" S14D_HOLD_S=10 READY_TIMEOUT=3600 \
  S14_ENGINE_TIMEOUT=5400 S14D_LOGDIR=$LOGDIR \
  tools/dcu/s14d_run_one.sh W4 "$MDL" 4 1 "$CACHE" 8601 262144 0,1,2,3 probe
echo "[e4] W4 finished $(date -Is)"; free -m

# ---- S4: AOT reload (warm cache) = the "service" run ----------------------
S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
  S14C_TAG="e4_s4_tp4_graph_AOTreload_swap8g_sw200" S14D_HOLD_S=10 READY_TIMEOUT=3600 \
  S14_ENGINE_TIMEOUT=5400 S14D_LOGDIR=$LOGDIR \
  tools/dcu/s14d_run_one.sh S4 "$MDL" 4 1 "$CACHE" 8602 262144 0,1,2,3 probe
echo "[e4] S4 finished $(date -Is)"; free -m

# ---- compare --------------------------------------------------------------
python3 tools/dcu/s14d_compare.py /mnt/game/s14d/probe-S4.jsonl /mnt/game/s14d/probe-W4.jsonl \
  --label-a S4-service-reload --label-b W4-warmup-livecompile \
  --csv /mnt/game/s14d/e4-compare.csv 2>&1 | tee "$LOGDIR/E4-compare.txt"
echo "[e4] $(date -Is) done"
cat /proc/swaps; free -m; cat /proc/sys/vm/swappiness
