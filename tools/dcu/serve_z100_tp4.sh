#!/usr/bin/env bash
# Z100(gfx906)27B W8A8 INT8 —— 对外服务启动脚本(agent/DSH 可用口径)。
#
# 与 tools/dcu/s14_arm_env.sh 的区别(那个是 benchmark 口径,不带 agent 参数):
#   * 补上生产脚本 serve_27b_tp4.sh 的 agent 三件套:
#       --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
#     缺了它们,任何 tool_choice:"auto" 的请求都会被 vLLM 以 400 拒绝
#     ("auto tool choice requires --enable-auto-tool-choice and --tool-call-parser")。
#   * 补上 --enable-prompt-tokens-details:否则即使前缀缓存命中,
#     响应 usage 里 prompt_tokens_details 恒为 null,DSH/pi-ai 界面显示"缓存命中 0%"。
#   * 补上模板/采样兜底(生产同款):该模型 chat_template 默认 reasoning_effort=xhigh,
#     容易"只出 thinking 不给正文"甚至重复循环:
#       --default-chat-template-kwargs '{"reasoning_effort":"low"}'
#       --override-generation-config    '{"temperature":0.7,"repetition_penalty":1.05}'
#   * 保留 S14c 实测的生产可用图/内存口径:PIECEWISE + capture-sizes 1 2 4 8。
#
# 可覆盖的环境变量:PORT(8000) TP(4) MAXLEN(262144) MBT(2048) UTIL(0.85)
#                   DEVICES(0,1,2,3) CACHE(/mnt/game/serve/vllm-cache) SERVED(qwen3.8-27b)
#                   SDOT4(1) EAGER(0=graph) EXTRA(附加参数)
set -uo pipefail
PORT=${PORT:-8000}
TP=${TP:-4}
MAXLEN=${MAXLEN:-262144}
MBT=${MBT:-2048}
UTIL=${UTIL:-0.85}
DEVICES=${DEVICES:-0,1,2,3}
CACHE=${CACHE:-/mnt/game/serve/vllm-cache}
SERVED=${SERVED:-qwen3.8-27b}
SDOT4=${SDOT4:-1}
EAGER=${EAGER:-0}
MODEL=${MODEL:-/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8}

. /opt/dtk/env.sh >/dev/null 2>&1
. "${SRC:-/src}/tools/dcu/env.sh" >/dev/null 2>&1
export PYTHONPATH="${SRC:-/src}:/mnt/game/triton-gfx906/install"
export VLLM_CACHE_ROOT="$CACHE"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/game/triton-gfx906/cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES="$DEVICES"
export VLLM_DCU_USE_INT8_SDOT4="$SDOT4"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export PYTHONUNBUFFERED=1
mkdir -p "$CACHE"

EAGER_ARGS=""
[ "$EAGER" = "1" ] && EAGER_ARGS="--enforce-eager"
# 图配置:graph 时用 S14c 的省内存/可用口径(PIECEWISE + 4 桶)
CG_ARGS=""
if [ "$EAGER" != "1" ]; then
  CG_ARGS="-cc.cudagraph_mode=PIECEWISE --cudagraph-capture-sizes 1 2 4 8"
fi

echo "[serve] port=$PORT tp=$TP maxlen=$MAXLEN mbt=$MBT util=$UTIL devices=$DEVICES sdot4=$SDOT4 eager=$EAGER"
echo "[serve] cache=$CACHE py=$PYTHONPATH"
echo "[serve] date=$(date -Is) host=$(hostname)"

cd "${SRC:-/src}" || exit 1
exec python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve "$MODEL" \
  --dtype float16 \
  --tensor-parallel-size "$TP" \
  --pipeline-parallel-size 1 \
  --max-model-len "$MAXLEN" \
  --max-num-batched-tokens "$MBT" \
  --gpu-memory-utilization "$UTIL" \
  --disable-custom-all-reduce \
  --skip-mm-profiling \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --override-generation-config '{"temperature":0.7,"repetition_penalty":1.05}' \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  $EAGER_ARGS $CG_ARGS ${EXTRA:-}
