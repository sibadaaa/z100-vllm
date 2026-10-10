#!/usr/bin/env bash
# S14c: container-side launcher for one measured engine (27B W8A8 INT8).
#
# Byte-for-byte the S14 recipe (tools/dcu/s14_arm_env.sh, md5 a529ec55…) plus
# the S14c A/B knobs.  S14/S14b scripts are frozen baseline evidence and are
# never edited; this is a copy.
#
#   docker exec -i <CT> bash -s -- <NAME> <MODEL_DIR> <TP> <SW> <PORT> <MAXLEN> [EXTRA...] \
#       < /home/lyh/z100-vllm/tools/dcu/s14c_arm_env.sh
#
# Required env: S14_CACHE
# S14-compatible env: S14_DEVICES S14_UTIL S14_MBT S14_CAR S14_EAGER S14_CG_MODE
#                     S14_CG_SIZES S14_SERVED S14_TRITON_CACHE S14_LOGLEVEL S14_SRC
# S14c-only env (all default to "unset = S14 behaviour"):
#   S14C_MALLOC_ARENA_MAX     -> export MALLOC_ARENA_MAX (glibc arena cap)
#   S14C_MALLOC_TRIM_THRESHOLD / S14C_MALLOC_MMAP_THRESHOLD / S14C_MALLOC_TOP_PAD
#   S14C_HIP_ALLOC_CONF       -> override PYTORCH_HIP_ALLOC_CONF (default expandable_segments:True)
#   S14C_INDUCTOR_CFG         -> -cc.inductor_compile_config='<json>'  *** changes the AOT key ***
#   S14C_MAX_NUM_SEQS         -> --max-num-seqs N
#   S14C_EXTRA_ARGS           -> appended verbatim to the serve command line
#   S14C_TAG                  -> free-form vintage tag echoed into the log
set -u
NAME=$1; MODEL=$2; TP=$3; SW=$4; PORT=$5; MAXLEN=$6; shift 6
echo $$ > "/tmp/s14c_${NAME}.pid"
date +%s > "/tmp/s14c_${NAME}.start"

. /opt/dtk/env.sh >/dev/null 2>&1
. "${S14_SRC:-/src}/tools/dcu/env.sh" >/dev/null 2>&1
export PYTHONPATH=${S14_SRC:-/src}:/mnt/game/triton-gfx906/install
export VLLM_CACHE_ROOT="${S14_CACHE:?S14_CACHE must be set}"
export TRITON_CACHE_DIR="${S14_TRITON_CACHE:-/mnt/game/triton-gfx906/cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF="${S14C_HIP_ALLOC_CONF:-expandable_segments:True}"
export HIP_VISIBLE_DEVICES="${S14_DEVICES:-0,1,2,3}"
export VLLM_DCU_USE_INT8_SDOT4=$SW
export VLLM_LOGGING_LEVEL="${S14_LOGLEVEL:-INFO}"
export PYTHONUNBUFFERED=1

# ---- S14c: glibc heap knobs (must be in the environment *before* python exec
# so that both the APIServer and every spawned worker inherit them) ----
for kv in MALLOC_ARENA_MAX MALLOC_TRIM_THRESHOLD MALLOC_MMAP_THRESHOLD MALLOC_TOP_PAD; do
  v="S14C_${kv}"; val="${!v:-}"
  if [ -n "$val" ]; then export "$kv=$val"; fi
done

CAR_ARGS=()
if [ "${S14_CAR:-1}" = "1" ]; then CAR_ARGS=(--disable-custom-all-reduce); fi
EAGER_ARGS=()
if [ "${S14_EAGER:-0}" = "1" ]; then EAGER_ARGS=(--enforce-eager); fi
CG_ARGS=()
if [ -n "${S14_CG_MODE:-}" ]; then CG_ARGS+=("-cc.cudagraph_mode=${S14_CG_MODE}"); fi
if [ -n "${S14_CG_SIZES:-}" ]; then CG_ARGS+=(--cudagraph-capture-sizes ${S14_CG_SIZES}); fi
SEQ_ARGS=()
if [ -n "${S14C_MAX_NUM_SEQS:-}" ]; then SEQ_ARGS=(--max-num-seqs "${S14C_MAX_NUM_SEQS}"); fi
IND_ARGS=()
if [ -n "${S14C_INDUCTOR_CFG:-}" ]; then IND_ARGS+=("-cc.inductor_compile_config=${S14C_INDUCTOR_CFG}"); fi
XARGS=()
if [ -n "${S14C_EXTRA_ARGS:-}" ]; then read -r -a XARGS <<< "${S14C_EXTRA_ARGS}"; fi

cd "${S14_SRC:-/src}" || exit 1
echo "[s14c] name=${NAME} model=${MODEL} tp=${TP} sw=${SW} port=${PORT} maxlen=${MAXLEN} car=${S14_CAR:-1} util=${S14_UTIL:-0.85} mbt=${S14_MBT:-2048} eager=${S14_EAGER:-0} cg_mode=${S14_CG_MODE:-default} cg_sizes=${S14_CG_SIZES:-default} extra=$*"
echo "[s14c] vintage_tag=${S14C_TAG:-none}"
echo "[s14c] knobs: MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-unset} MALLOC_TRIM_THRESHOLD=${MALLOC_TRIM_THRESHOLD:-unset} PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF} inductor_cfg=${S14C_INDUCTOR_CFG:-unset} max_num_seqs=${S14C_MAX_NUM_SEQS:-unset} extra_args=${S14C_EXTRA_ARGS:-unset}"
echo "[s14c] cache=${VLLM_CACHE_ROOT} devices=${HIP_VISIBLE_DEVICES} served=${S14_SERVED:-qwen3.8-27b}"
echo "[s14c] host=$(hostname) date=$(date -Is)"

exec python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve \
  "$MODEL" \
  --dtype float16 --tensor-parallel-size "$TP" --gpu-memory-utilization "${S14_UTIL:-0.85}" \
  "${CAR_ARGS[@]}" --skip-mm-profiling --enable-prefix-caching \
  --mamba-cache-mode align --max-model-len "$MAXLEN" --max-num-batched-tokens "${S14_MBT:-2048}" \
  --served-model-name "${S14_SERVED:-qwen3.8-27b}" --host 127.0.0.1 --port "$PORT" \
  "${EAGER_ARGS[@]}" "${CG_ARGS[@]}" "${SEQ_ARGS[@]}" "${IND_ARGS[@]}" "${XARGS[@]}" "$@"
