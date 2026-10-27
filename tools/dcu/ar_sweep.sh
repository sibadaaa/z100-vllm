#!/bin/bash
source /opt/dtk/env.sh >/dev/null 2>&1
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:$LD_LIBRARY_PATH
cd /tmp
run() {
  echo -n "$1 -> "
  env $2 timeout 200 torchrun --nproc_per_node=4 --master_port=$3 ar_bench.py 2>/dev/null | grep '\[1,5120\]' | head -1
}
run "baseline"            "" 29601
run "MIN_NCHANNELS=1"     "NCCL_MIN_NCHANNELS=1" 29602
run "LL128"               "NCCL_PROTO=LL128 NCCL_MIN_NCHANNELS=1" 29603
run "TREE"                "NCCL_ALGO=Tree NCCL_MIN_NCHANNELS=1" 29604
run "P2P off(shm)"        "NCCL_P2P_DISABLE=1 NCCL_MIN_NCHANNELS=1" 29605
