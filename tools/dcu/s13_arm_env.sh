#!/usr/bin/env bash
# S13: container-side launcher for one measured engine (27B W8A8 INT8).
# Derived from tools/dcu/s12_arm_env.sh; the only functional change is that
# --disable-custom-all-reduce is now conditional on S13_CAR (1 = pass it, the
# historical recipe; 0 = omit it, which enables vLLM's own CustomAllreduce).
#
#   docker exec -i z100-s9 bash -s -- <NAME> <MODEL_DIR> <TP> <SW> <PORT> <MAXLEN> [EXTRA...] \
#       < /home/lyh/z100-vllm/tools/dcu/s13_arm_env.sh
#
# Required env (passed through `docker exec -e`):
#   S13_CACHE    VLLM_CACHE_ROOT for this run (own subdir, hard constraint 17)
# Optional env:
#   S13_DEVICES  HIP_VISIBLE_DEVICES (default 0,1)   <- hard constraint 25
#   S13_CAR      1 = add --disable-custom-all-reduce (default 1)
#   S13_SERVED, S13_TRITON_CACHE, S13_LOGLEVEL, S13_SRC
set -u
NAME=$1; MODEL=$2; TP=$3; SW=$4; PORT=$5; MAXLEN=$6; shift 6
echo $$ > "/tmp/s13_${NAME}.pid"
date +%s > "/tmp/s13_${NAME}.start"

. /opt/dtk/env.sh >/dev/null 2>&1
. "${S13_SRC:-/src}/tools/dcu/env.sh" >/dev/null 2>&1
export PYTHONPATH=${S13_SRC:-/src}:/mnt/game/triton-gfx906/install
export VLLM_CACHE_ROOT="${S13_CACHE:?S13_CACHE must be set}"
export TRITON_CACHE_DIR="${S13_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES="${S13_DEVICES:-0,1}"
export VLLM_DCU_USE_INT8_SDOT4=$SW
export VLLM_DCU_SDOT4_INNER=${S13_INNER:-sdot4}
export VLLM_LOGGING_LEVEL="${S13_LOGLEVEL:-INFO}"
export PYTHONUNBUFFERED=1
CAR_ARGS=""
if [ "${S13_CAR:-1}" = "1" ]; then CAR_ARGS="--disable-custom-all-reduce"; fi
cd "${S13_SRC:-/src}" || exit 1
echo "[s13] name=${NAME} model=${MODEL} tp=${TP} sw=${SW} port=${PORT} maxlen=${MAXLEN} car=${S13_CAR:-1} extra=$*"
echo "[s13] cache=${VLLM_CACHE_ROOT} devices=${HIP_VISIBLE_DEVICES} served=${S13_SERVED:-qwen3.8-27b}"

exec python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve \
  "$MODEL" \
  --dtype float16 --tensor-parallel-size "$TP" --gpu-memory-utilization 0.9 \
  $CAR_ARGS --skip-mm-profiling --enable-prefix-caching \
  --mamba-cache-mode align --max-model-len "$MAXLEN" --max-num-batched-tokens 2048 \
  --served-model-name "${S13_SERVED:-qwen3.8-27b}" --host 127.0.0.1 --port "$PORT" "$@"
