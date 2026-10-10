#!/usr/bin/env bash
# S14: container-side launcher for one measured engine (27B W8A8 INT8).
# Derived from tools/dcu/s13_arm_env.sh (kept verbatim in structure so the two
# steps stay comparable); S14 changes are:
#   * S14_* env namespace (the S12/S13 scripts must stay frozen as baseline
#     evidence),
#   * the measured口径 is the PRODUCTION recipe from /home/lyh/server/serve_27b_tp4.sh
#     as given in the S14 task book §2: maxlen 262144 / gpu-mem-util 0.85 /
#     MBT 2048 / prefix caching / mamba align / --disable-custom-all-reduce,
#     no --enforce-eager (graph mode).
#
#   docker exec -i z100-s14 bash -s -- <NAME> <MODEL_DIR> <TP> <SW> <PORT> <MAXLEN> [EXTRA...] \
#       < /home/lyh/z100-vllm/tools/dcu/s14_arm_env.sh
#
# Required env (passed through `docker exec -e`):
#   S14_CACHE    VLLM_CACHE_ROOT for this run (own subdir)
# Optional env:
#   S14_DEVICES  HIP_VISIBLE_DEVICES (default 0,1,2,3 for TP=4)  <- hard constraint
#   S14_UTIL     --gpu-memory-utilization (default 0.85, production value)
#   S14_MBT      --max-num-batched-tokens (default 2048, production value)
#   S14_CAR      1 = add --disable-custom-all-reduce (default 1, production value)
#   S14_SERVED, S14_TRITON_CACHE, S14_LOGLEVEL, S14_SRC
#   S14_EAGER    1 = append --enforce-eager.  *** This is arm P4e and it VIOLATES
#                hard constraint 13 (gfx906 forbids --enforce-eager).  It is used
#                here deliberately, on Lead's explicit ruling, because the factual
#                basis of constraint 13 (all-NaN logits under CompilationMode.NONE)
#                was removed by S7b/S7c (root-caused to the fp16 packed
#                `_C.silu_and_mul` branch, fixed in csrc).  Every run with
#                S14_EAGER=1 must be registered as a constraint-13 violation and
#                must pass the NaN check (see docs/result/z100-14-...).
#   S14_CG_MODE  value for `-cc.cudagraph_mode=` (NONE|PIECEWISE|FULL|
#                FULL_DECODE_ONLY|FULL_AND_PIECEWISE); empty = leave default
#   S14_CG_SIZES space-separated list for `--cudagraph-capture-sizes`
#                (must be unquoted, the flag is argparse nargs='+')
set -u
NAME=$1; MODEL=$2; TP=$3; SW=$4; PORT=$5; MAXLEN=$6; shift 6
echo $$ > "/tmp/s14_${NAME}.pid"
date +%s > "/tmp/s14_${NAME}.start"

. /opt/dtk/env.sh >/dev/null 2>&1
. "${S14_SRC:-/src}/tools/dcu/env.sh" >/dev/null 2>&1
export PYTHONPATH=${S14G_PIN_PATH:+${S14G_PIN_PATH}:}${S14_SRC:-/src}:/mnt/game/triton-gfx906/install
export VLLM_CACHE_ROOT="${S14_CACHE:?S14_CACHE must be set}"
export TRITON_CACHE_DIR="${S14_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES="${S14_DEVICES:-0,1,2,3}"
export VLLM_DCU_USE_INT8_SDOT4=$SW
export VLLM_LOGGING_LEVEL="${S14_LOGLEVEL:-INFO}"
export PYTHONUNBUFFERED=1
CAR_ARGS=""
if [ "${S14_CAR:-1}" = "1" ]; then CAR_ARGS="--disable-custom-all-reduce"; fi
# P4e: --enforce-eager. Explicitly gated behind S14_EAGER=1 so the default
# (production graph recipe) stays untouched.
EAGER_ARGS=""
if [ "${S14_EAGER:-0}" = "1" ]; then EAGER_ARGS="--enforce-eager"; fi
# P4g: restrict cudagraph mode / capture-size buckets to shrink the graph pool.
CG_ARGS=""
if [ -n "${S14_CG_MODE:-}" ]; then CG_ARGS="$CG_ARGS -cc.cudagraph_mode=${S14_CG_MODE}"; fi
if [ -n "${S14_CG_SIZES:-}" ]; then CG_ARGS="$CG_ARGS --cudagraph-capture-sizes ${S14_CG_SIZES}"; fi
cd "${S14_SRC:-/src}" || exit 1
echo "[s14] name=${NAME} model=${MODEL} tp=${TP} sw=${SW} port=${PORT} maxlen=${MAXLEN} car=${S14_CAR:-1} util=${S14_UTIL:-0.85} mbt=${S14_MBT:-2048} eager=${S14_EAGER:-0} cg_mode=${S14_CG_MODE:-default} cg_sizes=${S14_CG_SIZES:-default} extra=$*"
echo "[s14] cache=${VLLM_CACHE_ROOT} devices=${HIP_VISIBLE_DEVICES} served=${S14_SERVED:-qwen3.8-27b}"
echo "[s14] host=$(hostname) date=$(date -Is)"

exec ${S14G_LAUNCH_PREFIX:-} python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve \
  "$MODEL" \
  --dtype float16 --tensor-parallel-size "$TP" --gpu-memory-utilization "${S14_UTIL:-0.85}" \
  $CAR_ARGS --skip-mm-profiling --enable-prefix-caching \
  --mamba-cache-mode align --max-model-len "$MAXLEN" --max-num-batched-tokens "${S14_MBT:-2048}" \
  --served-model-name "${S14_SERVED:-qwen3.8-27b}" --host 127.0.0.1 --port "$PORT" \
  $EAGER_ARGS $CG_ARGS "$@"
