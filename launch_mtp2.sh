#!/usr/bin/env bash
# z100 27B-FP8 生产 launcher（裸机，无 Docker）：CLI 参数 + 通用运行时。
#
#   生产启动只走 systemd： z100-vllm-mtp2.service -> serve_deploy.sh -> 本文件
#   前台复现：            bash serve_deploy.sh
#   干跑看配置：          PRINT_ONLY=1 bash serve_deploy.sh
#
# 分工：env 旋钮在 serve_deploy.sh；本文件只负责
#   1) DTK/HIP 运行时与编译缓存  2) 通用 env 默认值  3) vllm serve 的 CLI 参数。
#
# 覆盖项（单次 A/B，不改文件）：
#   EAGER=1                  --enforce-eager（跳过 cudagraph）
#   COMPILATION_CONFIG=...   覆盖 --compilation-config
#   SPEC_SET='{...}' / SPEC_OFF=1   覆盖/关闭 MTP
#   MBT / MAX_SEQS / KV_DTYPE / PORT / HOST / MODEL / APIKEY
#   PRINT_ONLY=1             打印解析后的 env + 命令行后退出 0
#
# FORBIDDEN: VLLM_USE_BREAKABLE_CUDAGRAPH=1（数值错误 + VM fault，已判死）。
set -u

REPO="${REPO:-/home/lyh/z100-vllm}"
DTK_ROOT="${DTK_ROOT:-/mnt/game/host-runtime/dtk2604}"
PY="${PY:-${REPO}/.venv/bin/python}"
HOST="${HOST:-0.0.0.0}"          # 默认对 LAN 开放；HOST=127.0.0.1 可锁死
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-262144}"
# S24 L5：MBT 8192（生产）。4096 可省激活 workspace 但 TTFT 略差。
MBT="${MBT:-8192}"
UTIL="${UTIL:-0.85}"
DEVICES="${DEVICES:-0,1,2,3}"
ITER_DETAILS="${ITER_DETAILS:-0}"   # 1 = --enable-logging-iteration-details（日志量变大）

# S55 WP4（2026-11-05 采纳）：FULL decode graph + MTP k=2。capture 尺寸跟着
# verify 行数走（1+2 draft -> 2/3/6/12），prefill 保持 eager。
if [ "${EAGER:-0}" = "1" ]; then
  COMPILE_ARGS=(--enforce-eager)
else
  # 注意：默认值放在独立变量里，不要内联进 ${X:-...}——JSON 的 `}` 会被
  # bash 当成展开结束符，重启时会多出一个 `}` 让 argparse 直接拒绝启动。
  _DEFAULT_COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2,3,6,12]}'
  COMPILATION_CONFIG="${COMPILATION_CONFIG:-${_DEFAULT_COMPILATION_CONFIG}}"
  COMPILE_ARGS=(--compilation-config "${COMPILATION_CONFIG}")
fi

export DTK_ROOT
# env.sh 会写它自己的 VLLM_CACHE_ROOT/TRITON_CACHE_DIR 默认值，先记住调用方
# 的覆盖，source 之后再落回去。
_VLLM_CACHE_OVERRIDE="${VLLM_CACHE_ROOT:-}"
_TRITON_CACHE_OVERRIDE="${TRITON_CACHE_DIR:-}"
# DTK env.sh 使用裸 ${CMAKE_PREFIX_PATH} 等，不能在 set -u 下执行。
set +u
# shellcheck disable=SC1091
. "${REPO}/tools/dcu/env.sh"
set -u

# S89 事故加固（2026-11-12）：抢跑 KFD topology 会让 VLLM_DCU_GCN_ARCH 落空，
# 编译缓存键随之改变并静默加载旧图（decode 掉到慢 36x 的 Triton 路径）。
# 服务启动必须在 arch 解析成功后才能继续；探针脚本可 VLLM_DCU_REQUIRE_ARCH=0 放行。
if [ "${VLLM_DCU_REQUIRE_ARCH:-1}" = "1" ] && [ "${VLLM_DCU_GCN_ARCH:-}" != "gfx906" ]; then
  echo "[host-serve] FATAL: VLLM_DCU_GCN_ARCH='${VLLM_DCU_GCN_ARCH:-}' is not" \
       "gfx906 -- KFD topology unreadable (driver still probing?) or the arch" \
       "changed. Refusing to boot: a wrong arch silently selects a different" \
       "compile cache and can load a stale graph. Fix the driver wait or set" \
       "VLLM_DCU_GCN_ARCH=gfx906 explicitly." >&2
  exit 1
fi

# 裸机启动顺序：rock 驱动由 hymgr.service 加载，可能晚于用户服务。等 4 张卡。
for _i in $(seq 1 60); do
  _n=$(hy-smi 2>/dev/null | grep -c '^[0-9]')
  [ "${_n:-0}" -ge 4 ] && break
  echo "[host-serve] waiting for HCUs (${_n:-0}/4, ${_i}/60)"
  sleep 2
done
unset _i _n

export PYTHONPATH="${REPO}:/mnt/game/triton-gfx906/install"
export VLLM_CACHE_ROOT="${_VLLM_CACHE_OVERRIDE:-/mnt/game/vllm-cache-g2}"
export TRITON_CACHE_DIR="${_TRITON_CACHE_OVERRIDE:-/mnt/game/host-serve/triton-cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export HIP_VISIBLE_DEVICES="${DEVICES}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export PYTHONUNBUFFERED=1

# ── FP8 emulation 主栈（2026-10-21 起在役）──────────────────────────────────
export VLLM_DCU_BF16=1
export VLLM_DCU_USE_FP8_EMU=1
# W8A16：权重留 fp8，激活走原始 fp16，跳过 per-token-group 量化。
export VLLM_DCU_FP8_W8A16="${VLLM_DCU_FP8_W8A16:-1}"
# 大 M prefill 把权重反量化进复用的 fp16 scratch，走 rocBLAS（16.5-17.7 TF/s）。
export VLLM_DCU_FP16_PREFILL="${VLLM_DCU_FP16_PREFILL:-auto}"
# S23：BK=64/BN=32 wide2 decode tile（c=1 step −7.3%，逐位一致）。
export VLLM_DCU_FP8_DECODE_CFG="${VLLM_DCU_FP8_DECODE_CFG:-35}"
# S23 O1/O2：eager 调用路径捷径（host 时间 −13.9 / −3.3 ms/step）。
export VLLM_DCU_FP8_EMU_FASTPATH="${VLLM_DCU_FP8_EMU_FASTPATH:-1}"
export VLLM_DCU_FP8_QUANT_FASTPATH="${VLLM_DCU_FP8_QUANT_FASTPATH:-1}"
# S24 L3.2 dyn-k：长上下文只验证 1+3/1+1 行（32k ITL −23%）。LO=0 关闭。
export VLLM_S24_DYNK_LO="${VLLM_S24_DYNK_LO:-16384}"
export VLLM_S24_DYNK_HI="${VLLM_S24_DYNK_HI:-65536}"
export VLLM_S24_DYNK_EMA="${VLLM_S24_DYNK_EMA:-0.1}"
# O4 的 VLLM_DCU_FP8_STREAM_MAX_M 保持不设：快但改变 5120:1536@M=8 累加序，
# 破坏贪心逐位一致。
export VLLM_QWEN35_MTP_GDN_TRITON=1
export VLLM_GFX906_GDN_CHUNKED_PREFILL=1
export VLLM_DCU_USE_BF16_GEMV=1
export VLLM_GFX906_ATTN_VERIFY_3D=1
export VLLM_GFX906_ATTN_FORCE_CAUSAL=0

# S30c WP1-N：GDN M=1 decode 原生化内核（miss 回旧路径）。CFG 1=64x4。
export VLLM_GFX906_GDN_DEC_M1="${VLLM_GFX906_GDN_DEC_M1:-1}"
export VLLM_GFX906_GDN_DEC_M1_CFG="${VLLM_GFX906_GDN_DEC_M1_CFG:-1}"
export VLLM_GFX906_GDN_DEC_M1_DEBUG="${VLLM_GFX906_GDN_DEC_M1_DEBUG:-0}"
export VLLM_GFX906_GDN_DEC_M1_TIME="${VLLM_GFX906_GDN_DEC_M1_TIME:-0}"

# S29 WP-A accept 彩票钉死：rocBLAS handle workspace 大小决定 aux-split，
# 必须固定，否则 accept 长度跨重启漂移（±6-10%）。勿删。
export ROCBLAS_WORKSPACE_CONFIG="${ROCBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HIPBLAS_WORKSPACE_CONFIG="${HIPBLAS_WORKSPACE_CONFIG:-:4096:8}"

# S29 WP-C：FA verify BLOCK_M 8 -> 16（step p50 −3.1 ms，parity 16/16）。
export VLLM_GFX906_ATTN_DECODE_BLOCK_M="${VLLM_GFX906_ATTN_DECODE_BLOCK_M:-16}"

# T4/S81：长上下文 prefill split-KV(3D)，MIN_CTX 与 SEGMENTS 见 serve_deploy。
export VLLM_GFX906_TRITON_ATTN_PREFILL_3D="${VLLM_GFX906_TRITON_ATTN_PREFILL_3D:-1}"
export VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX="${VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX:-4096}"
export VLLM_GFX906_ATTN_PREFILL_SEGMENTS="${VLLM_GFX906_ATTN_PREFILL_SEGMENTS:-4}"
# S51：conv1d fallback prefill 每个 conv 后 drain，避免 90k prefill 卡环。
export VLLM_GFX906_CONV_PREFILL_DRAIN="${VLLM_GFX906_CONV_PREFILL_DRAIN:-1}"

# WP0.4 策略指纹：打印引擎实际生效的每个旋钮 + sha256[:16]，A/B 报告必须引用。
"${PY}" "${REPO}/tools/dcu/dcu_policy_fingerprint.py" \
  --prefix "[host-serve] policy: " 2>&1 || true

MODEL="${MODEL:-/mnt/game/models/Qwen3.8-27B-FP8}"

# API key：显式 APIKEY 优先，否则读 0600 key 文件；空 = 无鉴权（仅限可信 LAN）。
if [ -z "${APIKEY:-}" ] && [ -r /mnt/game/host-serve/api-key ]; then
  APIKEY="$(cat /mnt/game/host-serve/api-key)"
fi
APIKEY_ARG=()
[ -n "${APIKEY:-}" ] && APIKEY_ARG=(--api-key "${APIKEY}")

ITER_ARGS=()
[ "${ITER_DETAILS}" = "1" ] && ITER_ARGS=(--enable-logging-iteration-details)

# MAX_SEQS=4：RequestState UVA buffer 是 (max_num_seqs, max_model_len) int32
# pinned，4 seqs = 4 MiB，SIGKILL 后重启预算也稳（见 DEPLOY.md §注意）。
MAX_SEQS="${MAX_SEQS:-4}"
MNS_ARGS=(--max-num-seqs "${MAX_SEQS}")

KV_ARGS=()
[ -n "${KV_DTYPE:-}" ] && KV_ARGS=(--kv-cache-dtype "${KV_DTYPE}")

# S34/S46：原生 MTP k=2（目标 checkpoint 自带 mtp.* 权重）。SPEC_SET 优先，
# SPEC_OFF=1 关闭，否则 k=2 默认。
_SPEC_DEFAULT='{"method":"mtp","num_speculative_tokens":2}'
SPEC="${SPEC_SET:-${SPEC:-${_SPEC_DEFAULT}}}"
if [ "${SPEC_OFF:-0}" = "1" ] && [ -z "${SPEC_SET:-}" ]; then
  SPEC=""
  echo "[host-serve] SPEC_OFF=1 -> no --speculative-config (MTP disabled)"
fi
SPEC_ARGS=()
[ -n "${SPEC}" ] && SPEC_ARGS=(--speculative-config "${SPEC}")

# S59：三开关必须同开（kv_cache_manager 用 and 串起来）。当前调度块 lcm=1616，
# 所以 match unit 取能整除它的最小值 16，retention 取 1616。
PREFIX_ARGS=()
if [ "${S59_PREFIX_CACHE:-1}" = "1" ]; then
  PREFIX_ARGS=(--prefix-match-unit 16 --enable-mamba-shared-prefix-checkpoint \
               --prefix-cache-retention-interval 1616)
fi

# 默认 chat template kwargs：可被 host wrapper 覆盖（如关思考）。
_DEFAULT_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"low"}'
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-${_DEFAULT_CHAT_TEMPLATE_KWARGS}}"

echo "[host-serve] date=$(date -Is) host=$(hostname) user=$(id -un)"
echo "[host-serve] dcu=$(cat "${DTK_ROOT}/.dtk_version" 2>/dev/null | head -1) python=${PY}"
echo "[host-serve] model=${MODEL} tp=4 maxlen=${MAXLEN} mbt=${MBT} util=${UTIL} port=${PORT} iter_details=${ITER_DETAILS}"
echo "[host-serve] compile=${COMPILE_ARGS[*]}"
echo "[host-serve] spec=${SPEC}"
echo "[host-serve] chat_template_kwargs=${CHAT_TEMPLATE_KWARGS}"
echo "[host-serve] kv=${KV_DTYPE:-default} max_seqs=${MAX_SEQS} prefix_cache=${S59_PREFIX_CACHE:-1}"

mkdir -p "${VLLM_CACHE_ROOT}" 2>/dev/null || true

CMD=(
  "${PY}" -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve
  "${MODEL}"
  --dtype float16 --tensor-parallel-size 4
  --gpu-memory-utilization "${UTIL}"
  --skip-mm-profiling --enable-prefix-caching --mamba-cache-mode align
  --max-model-len "${MAXLEN}" --max-num-batched-tokens "${MBT}"
  --served-model-name qwen3.8-27b --host "${HOST}" --port "${PORT}"
  "${APIKEY_ARG[@]}"
  "${ITER_ARGS[@]}"
  "${MNS_ARGS[@]}"
  "${KV_ARGS[@]}"
  "${PREFIX_ARGS[@]}"
  "${SPEC_ARGS[@]}"
  "${COMPILE_ARGS[@]}"
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/mnt/game/s30c-prof","torch_profiler_with_stack":false}'
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
  --enable-prompt-tokens-details
  --override-generation-config '{"temperature":0.7,"repetition_penalty":1.05}'
  --default-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}"
)

if [ "${PRINT_ONLY:-0}" = "1" ]; then
  echo "[print] --- resolved env (policy knobs) ---"
  env | grep -E '^(VLLM_[A-Z0-9_]+|S59_[A-Z0-9_]+|KV_DTYPE|MAX_SEQS|MBT|MAXLEN|UTIL|DEVICES|PORT|HOST|COMPILATION_CONFIG|PYTORCH_HIP_ALLOC_CONF|HIP_VISIBLE_DEVICES|HSA_FORCE_FINE_GRAIN_PCIE|ROCBLAS_WORKSPACE_CONFIG|HIPBLAS_WORKSPACE_CONFIG|TOKENIZERS_PARALLELISM|PYTHONPATH|VLLM_CACHE_ROOT|TRITON_CACHE_DIR)=' | sort
  echo "[print] --- command ---"
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

exec "${CMD[@]}"
