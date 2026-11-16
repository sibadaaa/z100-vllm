#!/usr/bin/env bash
# z100 生产服务 —— 环境旋钮的唯一权威来源（single source of truth）。
#
#   启动链： z100-vllm-mtp2.service -> 本文件 -> launch_mtp2.sh
#   运维：   bash start.sh {start|stop|restart|status|logs|config}
#   日志：   journalctl -u z100-vllm-mtp2 -f
#
# 本文件只收录"已采纳"的配置，每项都写成可覆盖默认值：
#     VLLM_DCU_FP8_GEMV_MAX_M=8 bash serve_deploy.sh   # 单次 A/B
# 回滚 = 去掉/改回对应行 + restart，不改代码、不换 .so（除非该行注明）。
#
# 当前采纳栈（2026-11-15）：
#   MTP k=2 + FULL decode graph [2,3,6,12] + int8_per_token_head KV
#   + prefix cache 16/1616 + S99 int8 per-channel prefill GEMM（MIN_M=1024）
#   + S53/S58 HIP attention inner + S34 GEMV/AR 栈
#   生产 .so：vllm/_rocm_C.abi3.so -> .s99/arms/build_pc（md5 3ccebe80）
#   回滚点：  .s97/build_phsr（adba0acb）· .s93/_rocm_C_dot2（7d2ed2a2）
#             · .s91/_rocm_C_kvint8fix（5af8b848）
#
# 注意：systemd drop-in 仍带有同值 Environment=（full-graph/g2-ms16/drain），
# 本文件与之处处同值，因此两种启动路径等价；有 root 后可删 drop-in。
set -u
cd /home/lyh/z100-vllm

# ── 运行时 / 图缓存 ─────────────────────────────────────────────────────────
# S89 G2：MS16 臂的 graph cache 必须与旧根隔离，否则加载到 t1int8 期图。
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/mnt/game/vllm-cache-g2}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-2400}"
# S51：90k prefill 的 conv1d 深在飞窗口会卡死计算环，逐 conv drain。
export VLLM_GFX906_CONV_PREFILL_DRAIN="${VLLM_GFX906_CONV_PREFILL_DRAIN:-1}"
# S55：FULL decode graph 下关 dyn-k（LO=0），否则 verify 行数抖动。
export VLLM_S24_DYNK_LO="${VLLM_S24_DYNK_LO:-0}"

# ── 自定义 all-reduce（S33/S34，PCIe-only 4 rank）────────────────────────────
export VLLM_DCU_CUSTOM_AR="${VLLM_DCU_CUSTOM_AR:-1}"
# gfx906 上 IPC buffer 必须细粒度分配，否则 AR 直接失败。
export VLLM_DCU_CUSTOM_AR_ALLOC="${VLLM_DCU_CUSTOM_AR_ALLOC:-finegrained}"
export VLLM_CUSTOM_ALLREDUCE_ALGO="${VLLM_CUSTOM_ALLREDUCE_ALGO:-oneshot}"
# S34-WP2 AR size 门：128 KiB 是 push 微基准甜点（c4 −4.6 ms/step）。
export VLLM_DCU_CUSTOM_AR_MAX_KIB="${VLLM_DCU_CUSTOM_AR_MAX_KIB:-128}"

# ── GEMV / GEMM 路由（decode 小 batch）──────────────────────────────────────
export VLLM_DCU_FP8_GEMV_STREAM_ROWS="${VLLM_DCU_FP8_GEMV_STREAM_ROWS:-4}"  # S34-WP3
export VLLM_DCU_FP8_GEMV_M1="${VLLM_DCU_FP8_GEMV_M1:-1}"
# S89 G2：M=12（MTP verify 形状）走 EXACT 流式臂；内核上限会 clamp 成 8。
export VLLM_DCU_FP8_GEMV_MAX_M="${VLLM_DCU_FP8_GEMV_MAX_M:-16}"
export VLLM_DCU_FP8_GEMV_WA="${VLLM_DCU_FP8_GEMV_WA:-1}"            # S41-A1 wholea
export VLLM_DCU_FP16_GEMV_STREAM="${VLLM_DCU_FP16_GEMV_STREAM:-1}"  # S40-B1 lm_head
# 禁令（S45 审计）：
#   VLLM_DCU_W8A16_SILU_MUL=1 已在 S49 否决（A 流量翻倍，down GEMV 534->319 GB/s）；
#   VLLM_QWEN35_EMPTY_CORE_ATTN_OUT=1 会使贪心流分叉（core_attn_out 的 zeros 承重）。

# ── AR 融合（S44-A4 混合臂）─────────────────────────────────────────────────
export VLLM_DCU_CUSTOM_AR_FUSION="${VLLM_DCU_CUSTOM_AR_FUSION:-1}"
# 全融合（=1）在 piecewise capture 下输出乱码（S38 损坏复现），硬锁 0。
export VLLM_DCU_CUSTOM_AR_FUSION_MLP=0

# ── attention 内核（S53/S58/S80/S81）────────────────────────────────────────
export VLLM_GFX906_ATTN_DECODE_HIP_INNER="${VLLM_GFX906_ATTN_DECODE_HIP_INNER:-1}"
export VLLM_GFX906_ATTN_PREFILL_HIP_INNER="${VLLM_GFX906_ATTN_PREFILL_HIP_INNER:-1}"
# S80/S81：split-KV(3D) prefill 门 16384 -> 4096（8k −14.7%，16k −7.2%）。
export VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX="${VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX:-4096}"
# S81：int8 KV 存储 + v_dot4 内层（有损格式，公式见 DEPLOY.md）。
export KV_DTYPE="${KV_DTYPE:-int8_per_token_head}"

# ── prefill GEMM（S99/S100a，当前最大收益项）────────────────────────────────
# S99：per-channel int8 权重 + in-op fp8 staging；8k −16.9%，32k −19.4%。
export VLLM_DCU_PREFILL_INT8_PC="${VLLM_DCU_PREFILL_INT8_PC:-1}"
# S100a：MIN_M 2048 -> 1024，使 1616/1712/2032 的 mamba 对齐分块也走 int8。
export VLLM_DCU_PREFILL_INT8_PC_MIN_M="${VLLM_DCU_PREFILL_INT8_PC_MIN_M:-1024}"

# ── 前缀缓存（S59，三开关必须同时生效）──────────────────────────────────────
export S59_PREFIX_CACHE="${S59_PREFIX_CACHE:-1}"

# ── 调度器 ──────────────────────────────────────────────────────────────────
# S100b B2 已否决：1 = 保留 partial-tail 的 n-16 SSM checkpoint（多轮命中率 4.5x）；
# 0 只省单轮冷 prefill，会让 turn2 重算 2.4k token，生产保持 1。
export VLLM_GFX906_MAMBA_PARTIAL_TAIL_SPLIT="${VLLM_GFX906_MAMBA_PARTIAL_TAIL_SPLIT:-1}"

exec bash ./launch_mtp2.sh "$@"
