#!/usr/bin/env bash
# S11l: container-side launcher for one 27B W8A8 arm (runs inside z100-s9,
# based on .tmp-11k/arm_env.sh from S11k).
#
#   docker exec -i z100-s9 bash -s -- <NAME> <SW> <INNER> <PORT> <MAXLEN> [EXTRA...] \
#       < /home/lyh/z100-vllm/tools/dcu/s11l_arm_env.sh
#
# SW=1|0      -> VLLM_DCU_USE_INT8_SDOT4
# INNER=sdot4|triton -> VLLM_DCU_SDOT4_INNER (diagnostic, only meaningful when SW=1)
#
# Runs from /src (the workspace, read-only in the container); the four extension
# .so files there are symlinks to the canonical copy, so the Python under test
# is this workspace tree and the binaries are the installed ones.
set -u
NAME=$1; SW=$2; INNER=$3; PORT=$4; MAXLEN=$5; shift 5
echo $$ > "/tmp/s11l_${NAME}.pid"

. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
export VLLM_CACHE_ROOT=${S11L_CACHE:-/mnt/game/vllm-cache-27b}
export TRITON_CACHE_DIR=${S11L_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES=${S11L_DEVICES:-0,1}
export VLLM_DCU_USE_INT8_SDOT4=$SW
export VLLM_DCU_SDOT4_INNER=$INNER
export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1
cd /src || exit 1
echo "[s11l] name=${NAME} sw=${SW} inner=${INNER} port=${PORT} maxlen=${MAXLEN} extra=$*"
echo "[s11l] cache=${VLLM_CACHE_ROOT} devices=${HIP_VISIBLE_DEVICES}"

exec python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve \
  /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 \
  --dtype float16 --tensor-parallel-size 2 --gpu-memory-utilization 0.9 \
  --disable-custom-all-reduce --skip-mm-profiling --enable-prefix-caching \
  --mamba-cache-mode align --max-model-len "$MAXLEN" --max-num-batched-tokens 2048 \
  --served-model-name qwen3.8-27b --host 127.0.0.1 --port "$PORT" "$@"
