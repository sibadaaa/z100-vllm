# Z100 P0 复测方案:prefill fp16 物化 A/B + 并发恢复(2026-10-21)

基线:`vllm-docker-probe/runs/20261021-033606-serve/REPORT.md`(27B-FP8 / TP=4 /
DFlash2 k=7 / maxlen 262144 / MBT 4096 / max_num_seqs 4)。该基线的全部 prefill
都走 W8A16 tile GEMM(~6.5 TF/s),冷 128k 级 prefill ~379 s,预填充均值
5647 ms / 1k context token。

## 0. 已落地变更(本次提交)

| 项 | 文件 | 内容 |
|---|---|---|
| P0-1 | `tools/dcu/serve_host_dflash2.sh` | 新增 `VLLM_DCU_FP16_PREFILL=auto`(可被调用方覆盖):按角色阈值把 prefill 路由到 dequant→rocBLAS(16.5–17.7 TF/s) |
| P0-2 | `gfx906_fp8.py`、`gfx906_fp16_prefill.py` | 启动 INFO 日志:W8A16 路由状态、物化路由角色阈值、首次物化路由生效 |
| P0-3 | 本文附录 A | `MAX_NUM_SEQS` 4→64 的恢复规程(需维护窗口,见附录) |

> **警告(基线漂移)**:默认值已经翻转过两次 —— `VLLM_DCU_FP8_W8A16`
> `0→auto`(WP2),当前 8000 端口在跑的实例(04:19 起)已经带
> `VLLM_DCU_FP16_PREFILL=auto`。**复测的每一臂都必须显式钉死全部 DCU env**,
> 不能依赖默认值或环境残留,否则就会出现"和基线没变化"的假对照。

## 1. 前置

```bash
# 1.1 停掉现网 8000 实例(它占着全部 4 卡,且配置与两臂都不同)
ps -o pid,cmd -C python | grep "vllm"          # 找到主进程
kill -TERM <pid>                                # 只用 SIGTERM,等 worker 全部退出
# 若 60 s 未退,才允许 SIGKILL;SIGKILL 后必须先做附录 A 的驱动重载再继续

# 1.2 确认 4 卡空闲
hy-smi | grep -c '^[0-9]'                       # 期望 4,无占用进程
sudo fuser -v /dev/kfd                          # 期望无输出

# 1.3 代码就位:z100-vllm 检出含本次提交;probe 框架 .env 的 RUNTIME=host
grep RUNTIME /home/lyh/vllm-test/vllm-docker-probe/.env
```

## 2. 臂定义(与 033606 基线同口径)

模板栈:`vendor/hygon scenario/tp4-prod spec/dflash`;启动覆盖
`MAX_MODEL_LEN=262144 MAX_NUM_BATCHED_TOKENS=4096 MAX_NUM_SEQS=4 PORT=8100`。
两臂唯一差异是 `VLLM_DCU_FP16_PREFILL`:

- **臂 A(对照)** `VLLM_DCU_FP16_PREFILL=0` —— 全部 prefill 走 W8A16 tile,等价 033606 基线
- **臂 B(处理)** `VLLM_DCU_FP16_PREFILL=auto` —— M≥角色阈值走 dequant→rocBLAS

## 3. 每臂执行序列

```bash
cd /home/lyh/vllm-test/vllm-docker-probe

# 033606 同款 S23/S24 内核旋钮,两臂共用(显式钉死,覆盖 .env 里的 GEMV_MAX_M=4)
COMMON_ENV="VLLM_DCU_BF16=1 VLLM_DCU_USE_FP8_EMU=1 VLLM_DCU_FP8_W8A16=1 \
VLLM_DCU_FP8_GEMV_MAX_M=8 VLLM_DCU_FP8_DECODE_CFG=35 \
VLLM_DCU_FP8_EMU_FASTPATH=1 VLLM_DCU_FP8_QUANT_FASTPATH=1 \
VLLM_S24_DYNK_LO=16384 VLLM_S24_DYNK_HI=65536 VLLM_S24_DYNK_EMA=0.1 \
VLLM_QWEN35_MTP_GDN_TRITON=1 VLLM_GFX906_GDN_CHUNKED_PREFILL=1 \
VLLM_DCU_USE_BF16_GEMV=1 VLLM_GFX906_ATTN_VERIFY_3D=1 \
VLLM_GFX906_ATTN_FORCE_CAUSAL=0 VLLM_GFX906_TRITON_ATTN_PREFILL_3D=1 \
VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX=16384 VLLM_GFX906_ATTN_PREFILL_SEGMENTS=4"

# —— 臂 A:把下面的 FP16_PREFILL 换成 0 再跑一遍,即臂 A ——
RUN_TAG=b-prefill-auto   # 臂 A 用 a-prefill-tile
EXTRA_ENV="$COMMON_ENV VLLM_DCU_FP16_PREFILL=auto" \
MAX_MODEL_LEN=262144 MAX_NUM_BATCHED_TOKENS=4096 MAX_NUM_SEQS=4 PORT=8100 \
  hosts/vllm-host.sh up vendor/hygon scenario/tp4-prod spec/dflash

RUN=$(ls -t runs | head -1); echo "RUN=$RUN"

# 3.1 启动日志三确认(臂 A 第 2 条应为 =0 且无第 3 条)
grep -E "gfx906 fp8-emu route:|fp16 prefill materialization:|first materialization route" \
  runs/$RUN/serve.log
# 期望(臂 B):
#   gfx906 fp8-emu route: VLLM_DCU_FP8_W8A16=1, W8A16 engaged=True (built op True), GEMV_MAX_M=8.
#   gfx906 fp16 prefill materialization: VLLM_DCU_FP16_PREFILL=auto, role min_m {...}, default 4096.
#   gfx906 fp16 prefill: first materialization route engaged (role=..., M=..., ...)

# 3.2 全链路探测(REPORT.md,~18 min)
RUN_DIR=runs/$RUN hosts/vllm-host.sh probe

# 3.3 标准化 8 场景套件(suite_results.json)
RUN_DIR=runs/$RUN hosts/vllm-host.sh suite

# 3.4 冷 128k 长上下文(单请求,无共享前缀;先跑这条再看其它,保证是冷缓存)
bash scripts/probe_http.sh --run-dir runs/$RUN/longctx-128k-cold \
  --port 8100 --prompt-tokens 131072 --max-tokens 16 --concurrency 1 --repeat 1

# 3.5 停实例(等价 down,保持与下一臂隔离)
hosts/vllm-host.sh down
```

对比与汇总:

```bash
make -C /home/lyh/vllm-test/vllm-docker-probe \
  compare A=runs/<A目录>/suite_results.json B=runs/<B目录>/suite_results.json
```

## 4. 判定标准

| 口径 | 来源 | 通过线 |
|---|---|---|
| 冷 128k TTFT | longctx-128k-cold/http_summary.json | 臂 B 相对 A **≥20% 改善**(录档物化=tile 的 0.50–0.70×,GEMM 占 prefill 大头) |
| 预填充效率 | REPORT §3 `预填充耗时 ms/1k context token` | 臂 B 下降,且 58 个纯预填充 step 均值同步下降 |
| decode 不回归 | REPORT §3 解码 step 均值/p50、TPOT | 波动 ≤ ±3%(物化路由不碰 M≤阈值 的调用) |
| 数值正确 | 两臂同 prompt 温度 0 各 3 次,前 64 token 对齐 | 一致率 ≥99%(dequant→rocBLAS 与 tile kernel 求和序不同,不要求逐 bit) |
| 稳定性 | serve.log | 0 次 worker 崩溃 / hipError;KV pool token 数变化 <0.5%(物化 workspace ~90 MB/卡) |
| 路由证据 | serve.log | 臂 B 有 "first materialization route engaged",臂 A 没有 |
| 投机解码 | REPORT §5 平均接受长度 | 两臂一致(物化不影响草稿塔) |

## 5. 回滚

- 单独回退路由:`EXTRA_ENV` 里 `VLLM_DCU_FP16_PREFILL=0`(脚本路径:启动前
  `export VLLM_DCU_FP16_PREFILL=0`),无需回代码。
- 彻底回退本次提交:revert `tools/dcu/serve_host_dflash2.sh`、
  `gfx906_fp8.py`、`gfx906_fp16_prefill.py` 的对应 hunk。

## 附录 A:P0-3 驱动重载与 max_num_seqs 恢复(维护窗口执行)

背景:对长驻实例 SIGTERM(30 s)→SIGKILL 后,rock/hycu 驱动层的 worker 可用
pinned 预算不可见收缩,vLLM worker 的 (256, 262144) int32 pinned 分配确定性
失败(`RequestState::UvaBuffer` OOM),`max_num_seqs` 被迫降到 4(运维记录
[z100-bf16-to-fp16-entry-coercion.md §7](z100-bf16-to-fp16-entry-coercion.md))。
独立进程探针无法验证该状态(坏状态下探针也成功),唯一验证方式是
`MAX_NUM_SEQS=64` 真实起服务。

```bash
# 仅在所有 GPU 进程退出后执行;任何一步失败就停下,走 modprobe hycu 兜底或主机重启
sudo systemctl stop hymgr                     # C-3000 监控守护,避免持有设备
sudo rmmod hycu                               # 必须第一个卸载(依赖图源头)
sudo rmmod hydrm_ttm_helper hycu_sched hy_extra hydrm_buddy
sudo rmmod hyttm
sudo rmmod hykcl
sudo modprobe hycu                            # 自动带回全部依赖
sudo systemctl start hymgr
hy-smi | grep -c '^[0-9]'                     # 期望 4

# 验证:64 并发真实起服(复用 §3 的 EXTRA_ENV 与模板)
MAX_NUM_SEQS=64 hosts/vllm-host.sh up vendor/hygon scenario/tp4-prod spec/dflash
# 通过判据:serve.log 无 "UvaBuffer"/"hipErrorOutOfMemory",/health 200
# 通过后:tools/dcu/serve_host_dflash2.sh 的 MAX_SEQS 默认 4→64(当前保持 4)
```

注意:任何一次对长驻实例的 SIGKILL 之后,都应重做本附录再起 64 并发;
干净 SIGTERM 无此必要。
