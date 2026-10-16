#!/usr/bin/env bash
# perf1: run the 4-rank all-reduce ladder once per NCCL/RCCL variant.
# Log: /mnt/game/perf1/comm_matrix.log  (container-side redirection only)
set -u
. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
cd /src

PORT=29611
run() {  # run <tag> <env assignments...>
  local tag="$1"; shift
  echo "### variant=$tag env=[$*] date=$(date -Is)"
  echo "### concurrent: $(pgrep -c -f 'vllm.entrypoints.cli.main' || true) vllm serve procs; service :8000 = $(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health || echo down)"
  PF1_TAG="$tag" timeout 300 env "$@" torchrun --nproc_per_node=4 \
      --master_port=$PORT tools/dcu/perf1_comm_ar.py 2>&1 | grep -E '^\[|Error|error' | head -20
  PORT=$((PORT + 1))
}

run "base_P2P"                 HSA_FORCE_FINE_GRAIN_PCIE=1
run "shm_P2Poff"               HSA_FORCE_FINE_GRAIN_PCIE=0
run "tree"                     HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_ALGO=Tree
run "proto_LL"                 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_PROTO=LL
run "proto_Simple"             HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_PROTO=Simple
run "minch1"                   HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_MIN_NCHANNELS=1
run "ll_thresh128k"            HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_LL_THRESHOLD=131072
echo "### ALLDONE $(date -Is)"
