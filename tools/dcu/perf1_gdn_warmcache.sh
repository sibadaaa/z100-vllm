#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Warm-cache-only variant of the Perf1 GDN GPU window (task-3 / gdn-opt).
#
# Use when the verifier's gdn_triton arm must run before the full sweep: it
# only JIT-compiles the kernel into the shared Triton cache, one launch per
# config, on GPU 1.  ~30-60 s.
#
#   docker exec z100-vllm bash -c \
#     'flock -w 1800 /tmp/perf1_build.lock bash /src/tools/dcu/perf1_gdn_warmcache.sh'
set -u
source /opt/dtk/env.sh >/dev/null 2>&1
source /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
export HIP_VISIBLE_DEVICES=1
cd /src || exit 1

echo "[gdn_warm] start $(date -Is)"
echo "[gdn_warm] so _rocm_C=$(md5sum /src/vllm/_rocm_C.abi3.so | cut -d' ' -f1)"
python tools/dcu/perf1_gdn_warm.py > /mnt/game/perf1/gdn_warm.log 2>&1
echo "[gdn_warm] rc=$? done $(date -Is)"
