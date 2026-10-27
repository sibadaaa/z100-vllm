#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# perf1 / L3 post-build session (kernel-opt): correctness gate + updated tables.
#   bash tools/dcu/perf1_kernel_postbuild.sh <tag>
# Logs are written from inside the container to /mnt/game/perf1/.
set -u
TAG="${1:-post1}"
PRE='source /opt/dtk/env.sh >/dev/null 2>&1; source /src/tools/dcu/env.sh >/dev/null 2>&1; export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; cd /src'
SO=$(docker exec z100-vllm bash -c 'md5sum /mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so' | cut -d' ' -f1)
echo "=== so=${SO} tag=${TAG} $(date -Is)"

# 1) correctness gate (46 tests; includes M=1..8 decode range + fallbacks)
docker exec z100-vllm bash -c "$PRE; HIP_VISIBLE_DEVICES=0 python -m pytest \
  tests/kernels/quantization/test_gfx906_fp8_emu.py -q > /mnt/game/perf1/perf1_${TAG}_pytest.log 2>&1"
echo "--- pytest: $(docker exec z100-vllm bash -c "tail -1 /mnt/game/perf1/perf1_${TAG}_pytest.log")"

# 2) decode: stream shape x M (with the M>4 -> GEMM fallback now in place)
docker exec z100-vllm bash -c "$PRE; { echo \"=== postbuild stream table so=${SO}\"; date -Is; \
  export HIP_VISIBLE_DEVICES=0; python tools/dcu/perf1_kernel_mprobe.py \
    --shapes 1280:5120,3584:5120,4096:5120,5120:5120,8704:5120 --ms 1,2,3,4,5,6,8 \
    --modes gemv --iters 20 --check; } > /mnt/game/perf1/perf1_${TAG}_gemv.log 2>&1"
docker exec z100-vllm bash -c "grep -E '^ +[0-9]+x' /mnt/game/perf1/perf1_${TAG}_gemv.log"

# 3) prefill: cfg 0 now auto-maps to cfg 5 / cfg 10 (M=16..129, includes M=129)
docker exec z100-vllm bash -c "$PRE; { echo \"=== postbuild prefill auto so=${SO}\"; date -Is; \
  export HIP_VISIBLE_DEVICES=0; python tools/dcu/perf1_kernel_mprobe.py \
    --shapes 1280:5120,4352:5120,8704:5120 --ms 16,32,64,129 \
    --modes gemm --gemm-cfgs 0,5,10 --iters 10; } > /mnt/game/perf1/perf1_${TAG}_prefill.log 2>&1"
docker exec z100-vllm bash -c "grep -E '^ +[0-9]+x' /mnt/game/perf1/perf1_${TAG}_prefill.log"
echo "=== done so=${SO}"
