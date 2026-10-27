#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# perf1 / L3 kernel measurement session (kernel-opt).  Every arm pins GPU 0 and
# writes its log + JSON under /mnt/game/perf1/ **from inside the container**
# (host-side writes to /mnt/game are sandboxed).
#
#   bash tools/dcu/perf1_kernel_measure.sh <tag> [arms...]
#
# Arms:
#   gemv-old     VLLM_DCU_FP8_GEMV_STREAM=0                     (old GEMV)
#   gemv-s1      VLLM_DCU_FP8_GEMV_STREAM=1                     (stream v1, default)
#   gemv-s2      VLLM_DCU_FP8_GEMV_STREAM=2                     (stream v2)
#   gemv-s1p1    VLLM_DCU_FP8_GEMV_STREAM=1 VLLM_DCU_FP8_GEMV_PREF=1
#   gemm         all tiled-GEMM cfgs
#   prefill      gemm cfg 0 at M=16/32/64/129 (no-regression arm)
set -u
TAG="${1:-run}"; shift || true
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(gemv-old gemv-s1 gemv-s2 gemm prefill)

PRE='source /opt/dtk/env.sh >/dev/null 2>&1; source /src/tools/dcu/env.sh >/dev/null 2>&1; export PYTHONPATH=/src:/mnt/game/triton-gfx906/install; cd /src'
SHAPES="1280:5120,3584:5120,4096:5120,5120:5120,8704:5120"
SO=$(docker exec z100-vllm bash -c 'md5sum /mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so' | cut -d' ' -f1)

run_arm () {
  local envs="$1" modes="$2" name="$3" check="${4:-}"
  local log="/mnt/game/perf1/perf1_${TAG}_${name}.log"
  docker exec z100-vllm bash -c "$PRE; { \
      echo \"=== arm ${name}  env='${envs}'  so=${SO}\"; date -Is; \
      export ${envs} HIP_VISIBLE_DEVICES=0; \
      python tools/dcu/perf1_kernel_mprobe.py --shapes $SHAPES --ms 1,2,3,4,8 \
        --modes $modes --gemm-cfgs 0,1,2,3,5,6,7,8,9,10 --iters 20 $check; \
    } > $log 2>&1"
  echo "--- ${name}: $(docker exec z100-vllm bash -c "tail -n +1 $log | head -8 | tail -3")"
  docker exec z100-vllm bash -c "grep -v '^$' $log | tail -35"
}

for arm in "${ARMS[@]}"; do
  case "$arm" in
    gemv-old)  run_arm "VLLM_DCU_FP8_GEMV_STREAM=0" "gemv" "gemvold" "--check" ;;
    gemv-s1)   run_arm "VLLM_DCU_FP8_GEMV_STREAM=1" "gemv" "gemvs1" "--check" ;;
    gemv-s2)   run_arm "VLLM_DCU_FP8_GEMV_STREAM=2" "gemv" "gemvs2" "--check" ;;
    gemv-s1p1) run_arm "VLLM_DCU_FP8_GEMV_STREAM=1 VLLM_DCU_FP8_GEMV_PREF=1" "gemv" "gemvs1p1" "--check" ;;
    gemm)      run_arm "VLLM_DCU_FP8_GEMV_STREAM=0" "gemm" "gemmcfgs" ;;
    prefill)
      log="/mnt/game/perf1/perf1_${TAG}_prefill.log"
      docker exec z100-vllm bash -c "$PRE; { \
          echo \"=== arm prefill  so=${SO}\"; date -Is; \
          export HIP_VISIBLE_DEVICES=0; \
          python tools/dcu/perf1_kernel_mprobe.py \
            --shapes 1280:5120,4352:5120,8704:5120 --ms 16,32,64,129 \
            --modes gemm --gemm-cfgs 0 --iters 10; \
        } > $log 2>&1"
      docker exec z100-vllm bash -c "grep -v '^$' $log | tail -30"
      ;;
    *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
done
echo "=== done, so=${SO}"
