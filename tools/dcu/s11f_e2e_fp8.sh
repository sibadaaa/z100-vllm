#!/usr/bin/env bash
# S11f end-to-end: serve an FP8 checkpoint on Z100 (gfx906) through the emulated
# W8A8 FP8 kernel (csrc/rocm/fp8_emu_gemm.cu).
#
# Container-side launcher (the repo is mounted read-only at /src), so:
#
#   docker exec -d z100-vllm bash /src/tools/dcu/s11f_e2e_fp8.sh \
#       <MODEL_DIR> <TP> <PORT> [EXTRA...]
#
# Log: ${S11F_LOG:-/tmp/serve_fp8.log}.  The line that proves the new kernel is
# live is `Selected Gfx906Fp8ScaledMMLinearKernel for <module>` (vLLM logs it
# once per module at layer construction).
#
# Environment (all optional):
#   S11F_FP8_EMU    1|0  VLLM_DCU_USE_FP8_EMU    (default 1 = the arm under test;
#                        0 is the control: per-tensor FP8 must be refused)
#   S11F_BF16_GEMV  1|0  VLLM_DCU_USE_BF16_GEMV  (default 0: keep this arm clean)
#   S11F_DEVICES         HIP_VISIBLE_DEVICES     (default 0,1,2,3)
#   S11F_MAXLEN          --max-model-len         (default 32768)
#   S11F_UTIL            --gpu-memory-utilization(default 0.85)
#   S11F_MBT             --max-num-batched-tokens(default 2048)
#   S11F_EAGER      1|0  add --enforce-eager      (default 1; graph mode changes
#                        the token stream on this target -- S14b -- so keep the
#                        first arm eager and treat graph as a separate arm)
#   S11F_DTYPE           --dtype                 (default float16, production)
#   S11F_SERVED          --served-model-name     (default qwen3.8-27b-fp8)
#   S11F_LOG             log path                (default /tmp/serve_fp8.log)
#   S11F_CACHE           VLLM_CACHE_ROOT         (default .../vllm-cache-11f/e2e)
#   S11F_SRC             source tree             (default /src)
set -u
MODEL=${1:?usage: s11f_e2e_fp8.sh <MODEL_DIR> <TP> <PORT> [EXTRA...]}
TP=${2:?usage: s11f_e2e_fp8.sh <MODEL_DIR> <TP> <PORT> [EXTRA...]}
PORT=${3:?usage: s11f_e2e_fp8.sh <MODEL_DIR> <TP> <PORT> [EXTRA...]}
shift 3

. /opt/dtk/env.sh >/dev/null 2>&1
. "${S11F_SRC:-/src}/tools/dcu/env.sh" >/dev/null 2>&1
export PYTHONPATH="${S11F_SRC:-/src}:/mnt/game/triton-gfx906/install"
export VLLM_CACHE_ROOT="${S11F_CACHE:-/mnt/game/vllm-cache-11f/e2e}"
export TRITON_CACHE_DIR="${S11F_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES="${S11F_DEVICES:-0,1,2,3}"
export VLLM_DCU_USE_FP8_EMU="${S11F_FP8_EMU:-1}"
export VLLM_DCU_USE_BF16_GEMV="${S11F_BF16_GEMV:-0}"
export VLLM_LOGGING_LEVEL="${S11F_LOGLEVEL:-INFO}"
export PYTHONUNBUFFERED=1

EAGER=""
[ "${S11F_EAGER:-1}" = "1" ] && EAGER="--enforce-eager"

cd "${S11F_SRC:-/src}" || exit 1
echo "[s11f-e2e] model=$MODEL tp=$TP port=$PORT devices=$HIP_VISIBLE_DEVICES" \
     "fp8_emu=$VLLM_DCU_USE_FP8_EMU bf16_gemv=$VLLM_DCU_USE_BF16_GEMV eager=${S11F_EAGER:-1}" \
     "maxlen=${S11F_MAXLEN:-32768} host=$(hostname) date=$(date -Is)"
echo "[s11f-e2e] cache=$VLLM_CACHE_ROOT log=${S11F_LOG:-/tmp/serve_fp8.log}"

exec python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve \
  "$MODEL" \
  --dtype "${S11F_DTYPE:-float16}" --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "${S11F_UTIL:-0.85}" --disable-custom-all-reduce \
  --skip-mm-profiling --enable-prefix-caching --mamba-cache-mode align \
  --max-model-len "${S11F_MAXLEN:-32768}" --max-num-batched-tokens "${S11F_MBT:-2048}" \
  --served-model-name "${S11F_SERVED:-qwen3.8-27b-fp8}" --host 127.0.0.1 --port "$PORT" \
  $EAGER "$@"
