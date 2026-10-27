# S18d GEMM 派发的端到端验证：MTP 首次净赢（S18-1/K4 门禁）— 执行结果

> 归档时间：2026-10-12
> 执行者：Lead（真机臂，卡位显式隔离）
> 对应设计文档：docs/design/z100-18-verify-gemm-efficiency.md（G-K2 达成）、docs/design/z100-18-mtp-optimization.md §0.2.1
> 前一档：[z100-18c](z100-18c-gemm-dispatch-m-flat-result.md)（内核/op 级证明 GEMM 对 M 平坦）
> 状态：✅ 端到端门禁通过（K2 逐 token 正确性见后续 S18e 档）

## 1. 目标与验收标准

设计文档 K4：**decode TPOT 相对同 flags 非投机臂 ≥1.3×**（同 eager、同 maxlen、同采样）。
S18c 已在内核/op 级证明"`M≥2` 走 tiled GEMM 后成本对 M 平坦"，本步验证它是否真的转化为端到端收益。

## 2. 配置与开关

- 开关：`_GEMV_MAX_M`（FP8/INT8 各一），现由 [`vllm/envs.py`](../vllm/envs.py) 声明：
  `VLLM_DCU_FP8_GEMV_MAX_M` / `VLLM_DCU_INT8_GEMV_MAX_M`，**默认 8（= 旧行为）**；
  设为 `1` 时 `M≥2` 走 tiled GEMM，`M=1` 仍走 GEMV（`5120×1536` 上 M=1 的 GEMV 快 30%）。
- 两个 kernel 模块改为从 `envs` 读取（`gfx906_fp8.py` / `gfx906_sdot4.py`）；`.so` 未改。
- 容器内实测：默认下 `envs.VLLM_DCU_FP8_GEMV_MAX_M=8` 且模块常量 8；显式设为 1 后两者一致为 1（见 §3.3）。

## 3. 原始证据

### 3.1 FP8 / TP=4 / eager / maxlen 8192（生产同构，`HSA_FORCE_FINE_GRAIN_PCIE=1`）

| 臂 | 派发 | greedy MAL | accept_rate | Median TTFT | **Median TPOT** | 吞吐 | 相对 A0 |
|---|---|---|---|---|---|---|---|
| A0（K=0） | GEMV（全 M=1，两策略等价） | — | — | 968.57 ms | **123.32 ms** | 7.70 tok/s | 1.00× |
| A1（K=1） | GEMV（旧） | 1.751 | 0.751 | 1092.52 ms | 136.53 ms | 6.94 tok/s | **0.90×**（净负） |
| A2（K=2） | GEMV（旧） | 2.294 | 0.647 | 1183.04 ms | 145.46 ms | 6.51 tok/s | **0.85×**（净负） |
| **G1（K=1）** | **GEMM** | 1.781 | 0.781 | — | **99.20 ms** | **9.36 tok/s** | **1.24×** |
| **G2（K=2）** | **GEMM** | 2.246 | 0.623 | 1056.24 ms | **83.80 ms** | **10.94 tok/s** | **1.47×** |

（G1/G2 的 TTFT 是 bench 的 median TTFT；G2 = 1056.24 ms）

- **每步成本**（= TPOT × MAL）：A0 123.3 → G1 176.7（+53.4）→ G2 188.2（+11.5）。
  ⇒ GEMM 派发把"每多一个 verify token"的边际成本压到 **11.5 ms**（此前是 ~105 ms），
    剩下那 +53 ms 是 **MTP 头前向 + 每 token 的 attention/GDN/norm/采样**，与 GEMM 无关；
- 接受率对派发不敏感（同一个 27B FP8 模型：MAL 1.751 → 1.781、2.294 → 2.246，均在采样噪声内）。

### 3.2 FP8 / TP=2 / eager / maxlen 8192（同配置下 GEMV vs GEMM 的成对对照）

| 臂 | 派发 | MAL | Median TPOT | 吞吐 |
|---|---|---|---|---|
| K=0 | — | — | **182.55 ms** | 5.17 tok/s |
| K=1 | GEMV（旧） | 1.748 | 192.50 ms | 4.87 tok/s |
| **K=1** | **GEMM** | 1.759 | **137.61 ms** | **6.73 tok/s** |
| **K=2** | **GEMM** | 2.259 | **110.70 ms** | **8.19 tok/s** |

- 成对（同 TP、同 K=1，仅差派发开关）：**192.50 → 137.61 ms = 1.39×**；
- 相对同 TP 的非投机臂：K=1 **1.33×**、K=2 **1.65×**；
- 说明：TP=2 的 K=1 GEMV 臂来自已停手的子代理（其自述"计时可能受并发影响、接受率指标可信"），
  **结论只依赖同一 TP 内 GEMM/GEMV 的成对差**；本档的 TP=4 臂全部由 Lead 独占测量。

### 3.3 开关生效的直接证据（容器内）

```
envs default FP8: 8 | module: 8
envs default INT8: 8 | module: 8
after setenv (envs caches, expect 8): 1        # 显式设置后 envs 读到 1
```
（另见 [S18c](z100-18c-gemm-dispatch-m-flat-result.md) §3.2 的 op 级 0.6202→0.5839 ms 与 §3.1 的 M=1..8 平坦表）

### 3.4 TP 与 MTP 的相互作用

- 同 TP=2 下 K=0 = 182.55 ms；TP=4 下 K=0 = 123.32 ms ⇒ **FP8 槽上 TP=4 本身就快 1.48×**（与 INT8/eager 的旧结论相反，见 S14）；
- 因此生产配方应保持 **TP=4 + GEMM 派发 + MTP**：G2 = 83.80 ms/token（10.94 tok/s）。

## 4. 门禁判定

| 门禁 | 判定 | 证据 |
|---|---|---|
| K1 功能（起/出词/自定义 op 命中） | ✅ | 全部臂 ready；`QWEN35_MTP_GDN_CUSTOM_HIT` ×4 rank（S18b §3.2） |
| K3 接受率（p0 ≥60%、MAL ≥2.0） | ✅ | K=2 MAL 2.246（G2）；prod 采样 MAL 2.10（S18b） |
| **K4 性能（≥1.3×）** | ✅ | K=1 **1.24×**、K=2 **1.47×**（TP=4）；TP=2 K=2 1.65× |
| K2 逐 token 正确性 | ⏳ 见 S18e | 正在同一链里跑（`S18_FLIP=1` + `s11l_flip_compare.py`） |
| TTFT/conc8 不劣化 ≤5% | ⏳ 见 S18e | 同上（`ttft8k` + `conc8`） |
| 绝对吞吐（G-K1，≥400 GB/s） | ❌ 仍未达 | 独立收益项，转入 S18-4c（不再是 MTP 前置） |

## 5. 结论与下一步

1. **MTP 在本平台可用**：生产同构配置（FP8/TP=4/eager）k=2 取 **83.80 ms/token（1.47×、10.94 tok/s）**，k=1 取 99.20 ms（1.24×）；旧结论"净负"由**分派策略**造成，不是 MTP 本身；
2. 剩余可回收项（按收益排序）：
   - **+53 ms/步的 MTP 固定开销**：MTP 头前向、每 token 的 attention/GDN/norm/采样；其中 48 层 GDN 在 spec 下被关掉 packed 快路（`gdn_linear_attn.py:763-769`）、`GFX906_MTP_V2 MISS`（v1 路径）⇒ **S18-4 的目标从"GEMM 摊销"转为"每 token 的非 GEMM 开销"**；
   - k 的进一步扫描（k=3/5）：MAL 增长是否仍划算（当前 k=2 已 1.47×）；
   - 绝对内核吞吐（72–79 GB/s vs 峰值）⇒ baseline 独立提速（S18-4c）；
3. **上线口径**：保持 `_GEMV_MAX_M` 默认 8（bit-stable 纪律），生产显式 `VLLM_DCU_FP8_GEMV_MAX_M=1` + `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`；回滚 = 去掉环境变量（或设 8）+ 去掉 spec 配置。

## 6. 未完成项

1. K2 逐 token 对照与 TTFT8K/conc8 回归（S18e 档）；
2. k=3/5 扫描未做；
3. INT8 侧的 GEMM 派发（`VLLM_DCU_INT8_GEMV_MAX_M=1`）未测；INT8 是否为更优基底待定；
4. 长上下文（maxlen 262144）与 conc>8 未测；
5. `_GEMV_MAX_M` 的默认值是否改为 1（需全形状 ABBA + bit-stability 评估）。

## 7. 复现方式

```bash
# TP=4 k=1 / k=2（GEMM 派发）
docker exec -d z100-s18 bash -lc 'export VLLM_DCU_FP8_GEMV_MAX_M=1; export S18_PROBE_MODES="greedy"; \
  export S18_BENCH_MODES=decode; bash /src/tools/dcu/s18_one_arm.sh fp8 4 1 8192 gemmdisp'
docker exec -d z100-s18 bash -lc 'export VLLM_DCU_FP8_GEMV_MAX_M=1; export S18_PROBE_MODES="greedy"; \
  export S18_BENCH_MODES=decode; bash /src/tools/dcu/s18_one_arm.sh fp8 4 2 8192 gemmdisp'
# 对照（旧派发）
docker exec -d z100-s18 bash -lc 'export S18_PROBE_MODES="greedy"; export S18_BENCH_MODES=decode; \
  bash /src/tools/dcu/s18_one_arm.sh fp8 4 2 8192'
```
产物：`/mnt/game/s18/summary-fp8-tp{2,4}-k{0,1,2}-*`、`probe-*.json`、`bench-*-decode.log`；
回归前的原始件已备份到 `/mnt/game/s18/backup/pre-regression-*`。

## 8. 参考

- [S18c](z100-18c-gemm-dispatch-m-flat-result.md)（内核/op 级：GEMM 对 M 平坦、M=1 仍用 GEMV 的理由）
- [S18b](z100-18b-mtp-first-baselines-and-cost-scaling-result.md)（旧的 M 线性现象与 GEMV 派发下的净负结果）
- `docs/design/z100-18-verify-gemm-efficiency.md`（G-K1 降级为独立项、G-K2 达成）
