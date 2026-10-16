#!/usr/bin/env bash
# perf1 round 2 (MUST run under /tmp/perf1_build.lock): protocol selection at
# decode AND prefill payload scales.
#
# Question 1: is NCCL_PROTO=LL's -32% at 10-30 KiB still there, and what does it
# cost at prefill scale (a 2048-token chunk all-reduce is [2048,5120] fp16 =
# 20 MB, i.e. 16x past the 1 MiB point of round 1)?
# Question 2: does an explicit allowed-protocol list (NCCL_PROTO=LL,Simple) let
# RCCL keep LL for small and Simple for large messages?
# Question 3: what does RCCL actually pick per size (NCCL_DEBUG=INFO)?
set -u
. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
cd /src
PORT=29651
# 10 KiB, 30 KiB, 120 KiB, 1 MiB, 4 MiB, 20 MiB   (elements of fp16)
export PF1_SIZES="5120,15360,61440,524288,2097152,10485760"
export PF1_ITERS=60

run() {
  local tag="$1"; shift
  echo "### variant=$tag env=[$*] date=$(date -Is)"
  echo "### service resident: $(pgrep -c -f 'vllm.entrypoints.cli.main' || true) vllm serve proc(s), /v1/models=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models || echo down); lock holder=$$"
  PF1_TAG="$tag" timeout 300 env "$@" torchrun --nproc_per_node=4 \
      --master_port=$PORT tools/dcu/perf1_comm_ar.py 2>&1 | grep -E '^\[' | head -12
  PORT=$((PORT + 1))
}

run "r2_default"   HSA_FORCE_FINE_GRAIN_PCIE=1
run "r2_LL"        HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_PROTO=LL
run "r2_LLSimple"  HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_PROTO=LL,Simple
run "r2_Simple"    HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_PROTO=Simple

echo "### NCCL_DEBUG=INFO probe (30 KiB only) $(date -Is)"
PF1_TAG="dbg" PF1_SIZES="15360" PF1_ITERS=5 timeout 300 env \
    HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING \
    torchrun --nproc_per_node=4 --master_port=$PORT tools/dcu/perf1_comm_ar.py \
    > /mnt/game/perf1/comm_nccl_debug.log 2>&1
grep -E "AllReduce|Channel [0-9]|via |proto|Algo|algo|Setting" /mnt/game/perf1/comm_nccl_debug.log | sort -u | head -40
echo "### R2DONE $(date -Is)"
