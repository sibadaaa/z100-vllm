#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Perf1 staged GPU window for the GDN MTP work (task-3 / gdn-opt).
#
# Runs ONLY on GPU 1 (GPU 0 is `kernel-opt`'s).  Must be called with the
# global GPU-exclusive lock held, e.g.:
#
#   docker exec z100-vllm bash -c \
#     'flock -w 1800 /tmp/perf1_build.lock bash /src/tools/dcu/perf1_gdn_gpu.sh'
#
# Logs land in /mnt/game/perf1/.  TRITON_CACHE_DIR points at the same shared
# cache the service uses, so the arm does not pay a JIT stall on first call.
set -u
source /opt/dtk/env.sh >/dev/null 2>&1
source /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
export HIP_VISIBLE_DEVICES=1
cd /src || exit 1

echo "[gdn_gpu] start $(date -Is) pid=$$"
_cso=$(md5sum /src/vllm/_C.abi3.so | cut -d' ' -f1)
_rso=$(md5sum /src/vllm/_rocm_C.abi3.so | cut -d' ' -f1)
echo "[gdn_gpu] so _C=$_cso _rocm_C=$_rso built=$(stat -c %y /src/vllm/_C.abi3.so)"
echo "[gdn_gpu] HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
python tools/dcu/perf1_gdn_bench.py --iters 200 \
    --bv-list 1 4 8 16 --warps-list 1 2 \
    > /mnt/game/perf1/gdn_sweep.log 2>&1
echo "[gdn_gpu] gdn sweep rc=$?"
python tools/dcu/perf1_gdn_parity.py --seeds 8 \
    > /mnt/game/perf1/gdn_parity.log 2>&1
echo "[gdn_gpu] gdn parity rc=$?"
python tools/dcu/perf1_mm_probe.py --iters 30 \
    > /mnt/game/perf1/mm_probe.log 2>&1
echo "[gdn_gpu] mm probe rc=$?"
echo "[gdn_gpu] done $(date -Is)"
