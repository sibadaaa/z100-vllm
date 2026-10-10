#!/usr/bin/env bash
# S11m: container-side launcher for one 27B W8A8 INT8 arm (based on
# tools/dcu/s11l_arm_env.sh).  Differences: the cache root is mandatory and
# per-run (three-phase measurement), the log level is configurable, and extra
# `vllm serve` arguments can be appended.
#
#   docker exec -i z100-s9 bash -s -- <NAME> <SW> <PORT> <MAXLEN> [EXTRA...] \
#       < /home/lyh/z100-vllm/tools/dcu/s11m_arm_env.sh
#
# Required env (host passes them through `docker exec -e`):
#   S11M_CACHE   VLLM_CACHE_ROOT for this run (own subdir per phase!)
# Optional env:
#   S11M_LOGLEVEL (default INFO), S11M_TRITON_CACHE, S11M_DEVICES (default 0,1)
set -u
NAME=$1; SW=$2; PORT=$3; MAXLEN=$4; shift 4
echo $$ > "/tmp/s11m_${NAME}.pid"
: > "/tmp/s11m_${NAME}.start"
date +%s > "/tmp/s11m_${NAME}.start"

. /opt/dtk/env.sh >/dev/null 2>&1
. /src/tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
export VLLM_CACHE_ROOT="${S11M_CACHE:?S11M_CACHE must be set}"
export TRITON_CACHE_DIR="${S11M_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES="${S11M_DEVICES:-0,1}"
export VLLM_DCU_USE_INT8_SDOT4=$SW
export VLLM_DCU_SDOT4_INNER=${S11M_INNER:-sdot4}
export VLLM_LOGGING_LEVEL="${S11M_LOGLEVEL:-INFO}"
export PYTHONUNBUFFERED=1
cd /src || exit 1
echo "[s11m] name=${NAME} sw=${SW} port=${PORT} maxlen=${MAXLEN} extra=$*"
echo "[s11m] cache=${VLLM_CACHE_ROOT} devices=${HIP_VISIBLE_DEVICES} " \
     "loglevel=${VLLM_LOGGING_LEVEL} inner=${VLLM_DCU_SDOT4_INNER}"

exec python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve \
  /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 \
  --dtype float16 --tensor-parallel-size 2 --gpu-memory-utilization 0.9 \
  --disable-custom-all-reduce --skip-mm-profiling --enable-prefix-caching \
  --mamba-cache-mode align --max-model-len "$MAXLEN" --max-num-batched-tokens 2048 \
  --served-model-name qwen3.8-27b --host 127.0.0.1 --port "$PORT" "$@"
