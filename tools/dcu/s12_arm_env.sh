#!/usr/bin/env bash
# S12: container-side launcher for one measured engine (27B W8A8 INT8 or the
# 4B TP-scaling model).  Derived from tools/dcu/s11m_arm_env.sh / s11l_arm_env.sh.
#
#   docker exec -i z100-s9 bash -s -- <NAME> <MODEL_DIR> <TP> <SW> <PORT> <MAXLEN> [EXTRA...] \
#       < /home/lyh/z100-vllm/tools/dcu/s12_arm_env.sh
#
# Required env (passed through `docker exec -e`):
#   S12_CACHE    VLLM_CACHE_ROOT for this run (own subdir -- never reuse an
#                unrelated cache root, hard constraint 17)
# Optional env:
#   S12_DEVICES  HIP_VISIBLE_DEVICES (default 0,1)   <- hard constraint 25
#   S12_SERVED   served model name (default qwen3.8-27b)
#   S12_INNER    VLLM_DCU_SDOT4_INNER (default sdot4)
#   S12_TRITON_CACHE, S12_LOGLEVEL, S12_EXTRA_LOAD
set -u
NAME=$1; MODEL=$2; TP=$3; SW=$4; PORT=$5; MAXLEN=$6; shift 6
echo $$ > "/tmp/s12_${NAME}.pid"
date +%s > "/tmp/s12_${NAME}.start"

. /opt/dtk/env.sh >/dev/null 2>&1
. "${S12_SRC:-/src}/tools/dcu/env.sh" >/dev/null 2>&1
export PYTHONPATH=${S12_SRC:-/src}:/mnt/game/triton-gfx906/install
export VLLM_CACHE_ROOT="${S12_CACHE:?S12_CACHE must be set}"
export TRITON_CACHE_DIR="${S12_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES="${S12_DEVICES:-0,1}"
export VLLM_DCU_USE_INT8_SDOT4=$SW
export VLLM_DCU_SDOT4_INNER=${S12_INNER:-sdot4}
export VLLM_LOGGING_LEVEL="${S12_LOGLEVEL:-INFO}"
export PYTHONUNBUFFERED=1
cd "${S12_SRC:-/src}" || exit 1
echo "[s12] name=${NAME} model=${MODEL} tp=${TP} sw=${SW} port=${PORT} maxlen=${MAXLEN} extra=$*"
echo "[s12] cache=${VLLM_CACHE_ROOT} devices=${HIP_VISIBLE_DEVICES} served=${S12_SERVED:-qwen3.8-27b}"

exec python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve \
  "$MODEL" \
  --dtype float16 --tensor-parallel-size "$TP" --gpu-memory-utilization 0.9 \
  --disable-custom-all-reduce --skip-mm-profiling --enable-prefix-caching \
  --mamba-cache-mode align --max-model-len "$MAXLEN" --max-num-batched-tokens 2048 \
  --served-model-name "${S12_SERVED:-qwen3.8-27b}" --host 127.0.0.1 --port "$PORT" "$@"
