#!/usr/bin/env bash
# perf1: staggered-chain variants (critical-path predictor).
# MUST run under /tmp/perf1_build.lock.
# stagger=0  -> all four ranks arrive together (the quiet ladder's regime)
# stagger=300-> Z100 pair (ranks 2,3) carries +300 us of compute, as in service
set -u
. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
cd /src
PORT=29671
run() {
  local tag="$1"; local stagger="$2"; shift 2
  echo "### variant=$tag stagger=${stagger}us env=[$*] date=$(date -Is)"
  PF1_TAG="$tag" PF1_STAGGER_US="$stagger" timeout 300 env "$@" torchrun \
      --nproc_per_node=4 --master_port=$PORT tools/dcu/perf1_comm_chain.py 2>&1 \
      | grep -E '^\[' | head -10
  PORT=$((PORT + 1))
}
run "c_def_s0"   0   HSA_FORCE_FINE_GRAIN_PCIE=1
run "c_LL_s0"    0   HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_PROTO=LL
run "c_def_s300" 300 HSA_FORCE_FINE_GRAIN_PCIE=1
run "c_LL_s300"  300 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_PROTO=LL
echo "### CHAINDONE $(date -Is)"
