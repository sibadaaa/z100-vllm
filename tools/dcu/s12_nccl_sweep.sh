#!/usr/bin/env bash
# S12: one fabric state's NCCL/HCCL all-reduce sweep (container side).
#
#   docker exec -e S12_TAG=<acs_off|acs_on> z100-s9 bash /src/tools/dcu/s12_nccl_sweep.sh
#
# Protocol: 4 ranks (HIP_VISIBLE_DEVICES=0,1,2,3), bf16 all-reduce over
# [1,5120] / [1,10240] / [1,20480] / [1,40960], 20 warmup + 300 timed calls.
# Variants are the same knobs the user's /home/lyh/server/diag/ar_sweep.sh
# swept, so the two fabric states are directly comparable.
#
# Raw logs: /mnt/game/s12/nccl-<TAG>.log (+ nccl-<TAG>-debug.log for the P2P
# topology evidence run).
set -u
TAG="${S12_TAG:?S12_TAG must be acs_off|acs_on}"
OUT=/mnt/game/s12
mkdir -p "$OUT"
LOG="$OUT/nccl-$TAG.log"
DBG="$OUT/nccl-$TAG-debug.log"
: > "$LOG"

. /opt/dtk/env.sh >/dev/null 2>&1
export PYTHONPATH=/src
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH:-}
# Hard constraint 25: always pin the card set explicitly, never `unset`.
export HIP_VISIBLE_DEVICES=0,1,2,3
export S12_TAG="$TAG"
unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_P2P_DISABLE NCCL_DEBUG

run() { # name master_port extra_env...
  local name="$1" port="$2"; shift 2
  echo "### variant=$name env='$*' $(date -Is)" | tee -a "$LOG"
  env "$@" timeout 420 torchrun --nproc_per_node=4 --master_port="$port" \
    /src/tools/dcu/s12_ar_bench.py 2>/dev/null | tee -a "$LOG"
  echo "### rc=${PIPESTATUS[0]} variant=$name" | tee -a "$LOG"
}

run baseline        29801
run baseline_rep2   29811
run MIN_NCHANNELS_1 29802 NCCL_MIN_NCHANNELS=1
run LL128_1ch       29803 NCCL_PROTO=LL128 NCCL_MIN_NCHANNELS=1
run TREE_1ch        29804 NCCL_ALGO=Tree NCCL_MIN_NCHANNELS=1
run P2P_off_shm     29805 NCCL_P2P_DISABLE=1 NCCL_MIN_NCHANNELS=1
run RING_1ch        29806 NCCL_ALGO=Ring NCCL_MIN_NCHANNELS=1

# Topology evidence: NCCL_DEBUG=INFO once, only the [1,5120] shape matters here.
echo "### variant=P2P_TOPOLOGY(NCCL_DEBUG=INFO) $(date -Is)" >> "$LOG"
env NCCL_DEBUG=INFO timeout 420 torchrun --nproc_per_node=4 --master_port=29807 \
  /src/tools/dcu/s12_ar_bench.py > "$DBG" 2>&1
echo "### rc=$? topology-debug" >> "$LOG"
grep -aE "P2P|via P2P|NCCL INFO Channel|Using .*algorithm|nranks" "$DBG" | head -40 >> "$LOG"
echo "[s12] sweep done tag=$TAG log=$LOG"
