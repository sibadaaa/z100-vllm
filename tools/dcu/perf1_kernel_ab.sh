#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# perf1 / L3 clean A/B: rows>=120 load-balance (7e62d7ec) vs rows>=60 (8e98bd60),
# plus the decode M=5..8 GEMM-fallback and the prefill cfg-auto rule.  Both .so
# runs happen back-to-back in the same machine state; the live .so is restored
# and md5-checked at the end.  Host-side writes to /mnt/game are sandboxed, so
# everything runs inside the container.
set -u
PRE='source /opt/dtk/env.sh >/dev/null 2>&1; source /src/tools/dcu/env.sh >/dev/null 2>&1; export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; cd /src'
SHAPES="1280:5120,3584:5120,4096:5120,5120:5120,8704:5120,4352:5120"

docker exec z100-vllm bash -c "
set -e
install=/mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so
new=/mnt/game/perf1/_rocm_C.stream_8e98.so
old=/mnt/game/perf1/_rocm_C.stream_7e62.so
cp -f \$install \$new
flock /tmp/perf1_build.lock bash -c '
  PRE=\"$PRE\"
  install=/mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so
  echo \"--- ARM NEW rows>=120\"; cp -f /mnt/game/perf1/_rocm_C.stream_8e98.so \$install; md5sum \$install
  eval \$PRE; export HIP_VISIBLE_DEVICES=0
  python tools/dcu/perf1_kernel_mprobe.py --shapes $SHAPES --ms 1,2,3,4,5,8 --modes gemv --iters 20 --check > /mnt/game/perf1/perf1_ab_new_gemv.log 2>&1
  python tools/dcu/perf1_kernel_mprobe.py --shapes 1280:5120,4352:5120,8704:5120 --ms 16,32,64,129 --modes gemm --gemm-cfgs 0 --iters 10 > /mnt/game/perf1/perf1_ab_new_prefill.log 2>&1
  echo \"--- ARM OLD rows>=60\"; cp -f /mnt/game/perf1/_rocm_C.stream_7e62.so \$install; md5sum \$install
  eval \$PRE; export HIP_VISIBLE_DEVICES=0
  python tools/dcu/perf1_kernel_mprobe.py --shapes $SHAPES --ms 1,2,3,4,5,8 --modes gemv --iters 20 --check > /mnt/game/perf1/perf1_ab_old_gemv.log 2>&1
  python tools/dcu/perf1_kernel_mprobe.py --shapes 1280:5120,4352:5120,8704:5120 --ms 16,32,64,129 --modes gemm --gemm-cfgs 0 --iters 10 > /mnt/game/perf1/perf1_ab_old_prefill.log 2>&1
  echo \"--- RESTORE\"; cp -f /mnt/game/perf1/_rocm_C.stream_8e98.so \$install; md5sum \$install
'
"
echo "=== NEW(gemv) ==="; docker exec z100-vllm bash -c "grep -E '^ +[0-9]+x' /mnt/game/perf1/perf1_ab_new_gemv.log"
echo "=== OLD(gemv) ==="; docker exec z100-vllm bash -c "grep -E '^ +[0-9]+x' /mnt/game/perf1/perf1_ab_old_gemv.log"
echo "=== NEW(prefill) ==="; docker exec z100-vllm bash -c "grep -E '^ +[0-9]+x' /mnt/game/perf1/perf1_ab_new_prefill.log"
echo "=== OLD(prefill) ==="; docker exec z100-vllm bash -c "grep -E '^ +[0-9]+x' /mnt/game/perf1/perf1_ab_old_prefill.log"
