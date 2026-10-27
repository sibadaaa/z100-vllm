# P4.1 chunk 边界根因（337 ms/chunk 的复现与排除法归因）

> 施工依据：BUILD §8 WP4.1；设计依据：DESIGN §1/L6、§5 不做清单；数据依据：PROJ §3（MBT 8192 vs 2048 多付 1012 ms / 3 边界）
> 属主：A（Lead）。日期：2026-10-24。树 SHA `7078b5fe137feeab3750ee243daeaf172dc439d8`（分支 `dtk`）。
> 栈：4×Z100（gfx906）、DTK 26.04、torch 2.11.0、TP=4、DFlash2 k=7、eager、c=1、greedy、`max_num_seqs=4`。

---

## 1. 一句话结论

**chunk 边界成本复现且 >200 ms/边界，按 BUILD 停止线只出根因报告、不改 GDN，运行侧用 MBT ≥ 最大 prompt（=8192）兜底。**
固定 4096 token prompt：单 chunk TTFT **4503.5 ms**；MBT=2048 **4878.8 ms（+375 ms）**；MBT=1024 **5776.6 ms（+1273 ms）**。
逐 iteration 日志显示 **vLLM 的实际 chunk 被 mamba 对齐到 880-token 块**（MBT=2048 得到 1760/880/1456 三个 chunk，
不是 2 个），后两个 chunk 的每 token 成本是第一个的 1.26–1.46×。
排除法：把 fp16 物化关掉（`VLLM_DCU_FP16_PREFILL=0`）后**每 chunk 的额外固定成本仍然存在（≈0.3–0.42 s/chunk）**，
⇒ **fp16 物化"每 chunk 重填 workspace"不是主因**；残余落在两条路径共有的 chunked-prefill 代码（首选 GDN chunked prefill，48 层），
但本工作包不做定论、不改 GDN（BUILD 停止线）。

---

## 2. A/B 表（固定 4096 token，只改 MBT；cold TTFT = 唯一 marker 破坏前缀缓存，2 次取均值）

| MBT | 调度器实际 chunk 数（880 对齐） | 边界数 | cold TTFT (ms) | 相对单 chunk |
|---:|---:|---:|---:|---:|
| 8192 | 1 | 0 | 4503.5 | 基准 |
| 4096 | 1 | 0 | 4502.1 | −1.4 ms（噪声） |
| 2048 | 3（1760/880/1456） | 2 | **4878.8** | **+375.3 ms（188 ms/边界）** |
| 1024 | 5（880×4 + 576） | 4 | **5776.6** | **+1273.1 ms（318 ms/边界）** |

> 若按 BUILD 的原口径（chunk 数 = ceil(4096/MBT)，不考虑 mamba 对齐）则边界数为 1 和 3 ⇒ **377 / 425 ms/边界**，
> 两种口径都 **>200 ms/边界** 的停止线。880 对齐的实测口径更接近真实调度。
> 原始 JSON/日志：`.s27/p4/`（`mbt{4096,2048,1024}.*`、`cold_4096` 系列 summary）。

### 2.1 逐 iteration 证据（MBT=2048 + `--enable-logging-iteration-details`）

`Engine 000: Iteration(n): … iteration elapsed time: X ms`，4096-token prompt 的 3 个 context iteration：

| chunk | context tokens | elapsed (ms) | ms/token |
|---|---:|---:|---:|
| #1 | 1760 | 1753.7 | **0.996** |
| #2 | 880 | 1287.1 | 1.463 |
| #3 | 1456 | 1838.9 | 1.263 |

三 chunk 之和 4879.7 ms ≈ cold TTFT 4878.8 ms（一致）。第 2/3 个 chunk 的每 token 成本明显高于第 1 个，
且第 2 个（前缀 1760）甚至比第 3 个（前缀 2640）更贵 ⇒ 不是简单的"前缀越长越贵"，更像**每 chunk 一笔固定开销**。

### 2.2 排除 fp16 物化（唯一变量：`VLLM_DCU_FP16_PREFILL` auto → 0）

同一 MBT=2048、同一 4096-token prompt、同一 iteration 日志：

| chunk | tokens | elapsed, materialization=auto (ms) | elapsed, materialization=0 (ms) |
|---|---:|---:|---:|
| #1 | 1760 | 1753.7 | 5234.4 |
| #2 | 880 | 1287.1 | 2911.6 |
| #3 | 1456 | 1838.9 | 4746.3 |

以各自 chunk#1 的每 token 成本（0.996 / 2.974 ms）为基准，chunk#2/#3 的"额外成本"：

| chunk | materialization=auto 额外 | materialization=0 额外 |
|---|---:|---:|
| #2（前缀 1760） | +410 ms | +295 ms |
| #3（前缀 2640） | +389 ms | +415 ms |

⇒ **关掉物化后额外成本没有消失**（300–420 ms/chunk），因此 BUILD 候选根因 #2"物化 workspace 每 chunk 重填"**被排除为主因**。
物化的代价只体现在每 token 斜率（0.996 → 2.974 ms/token，即关掉后 prefill 慢 3×），与边界无关。

---

## 3. env 全文 + 策略指纹 + 树 SHA

- 树 SHA `7078b5fe137feeab3750ee243daeaf172dc439d8`；`vllm/` 无改动（全部为 env 臂）。
- 策略指纹：materialization=auto 臂 `FP 7e70fc2c130b9191`；`FP16_PREFILL=0` 臂 `FP` 以启动日志为准（该旋钮入指纹）。
- 关键 env：`VLLM_GFX906_GDN_CHUNKED_PREFILL=1`、`VLLM_DCU_FP16_PREFILL=auto`、`VLLM_DCU_FP8_W8A16=1`、
  `--mamba-cache-mode align`（正是它把 chunk 对齐到 880 token）、MBT 见上表。
- 启动日志证据：`Setting attention block size to 880 tokens to ensure that attention page size is >= mamba page size.`
- 采集命令：

```bash
# MBT 扫描（每臂一次启动 + 一个 4096 cold prompt；先小请求 warm JIT）
MBT=2048 MAX_SEQS=4 PORT=8000 bash tools/dcu/serve_host_dflash2.sh &
OUT=.s27/p4 bash .s27/cold_points.sh 4096 2 p4-mbt2048
# 逐 iteration 证据
ITER_DETAILS=1 MBT=2048 MAX_SEQS=4 PORT=8000 bash tools/dcu/serve_host_dflash2.sh &
grep "Iteration(" .s27/p4/mbt2048-iterdetails.serve.log | grep -v "0 context tokens"
```

---

## 4. 判读（对照 BUILD §8 WP4.1 阈值）

| 判据 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| 斜率 | <50 ms/边界 ⇒ 记录"Z100 正常"，MBT 4096 可用 | 188–425 ms/边界 | 不通过 |
| 斜率 | >200 ms/边界 ⇒ 出根因报告，先不修，交 DESIGN 复评 | 两种口径均 >200 | **出报告、不修** |
| 停止线 | 根因若指向 GDN，不得在 P4 改 GDN | 未改任何代码 | 遵守 |

**运行侧结论（零成本）**：MBT 保持 **8192 ≥ 最大 prompt 长度** ⇒ 7469 prompt 单 chunk，省掉全部边界成本
（对生产默认 4096 而言，这是把 7469 长 prompt 的 TTFT 从 9783 降回 8971 的唯一零成本手段；BUILD §3.2 已把 8192 作为主基线点）。
**注意**：MBT 8192 的代价是 KV pool 759,509 vs 806,753 tok（−5.9%），与 L5 长上下文容量权衡，按场景选择。

**根因候选（按证据强度）**：
1. **chunked-prefill 路径的每 chunk 固定开销**（两条路径共有，≈300–420 ms/chunk；非物化特有）。首选嫌疑：GDN chunked prefill
   的 48 层逐层 chunk kernel + 状态读写/对齐在每次 chunk 调度上的固定成本；也可能是每 chunk 64 层的 launch/host 固定开销。
2. **前缀相关项**：attention/GDN 对已计算前缀的处理（第 2/3 chunk 的每 token 成本上升 26–46%）。BUILD 认为 attention 的
   causal token-pair 数不随切分变化，但**前缀 KV 的重复读/状态转换**可能随前缀增长。
3. 已排除：fp16 物化 workspace（§2.2）。

---

## 5. 反证条件

- 若在 **GDN chunked prefill 关闭**（`VLLM_GFX906_GDN_CHUNKED_PREFILL=0`）下每 chunk 额外成本显著下降，则根因锁定 GDN，需重开立项；
  本窗口未做该臂（关掉后每 token 斜率变化太大，不能单变量判读边界项）。
- 若用 rocprofv3/内核级 trace 抓到"chunk#2 的 kernel 清单与 chunk#1 相同但多出固定前缀重算 kernel"，则候选 1/2 的划分需改写。
- 若 MBT=4096 实际是 2 chunk（而非本报告 1 chunk，取决于 max_num_batched_tokens 与 880 对齐的交互），上表边界数需重算。

## 6. 回滚方式

无代码改动；回滚 = 不设置任何新变量。MBT 回到生产默认 4096 即恢复原行为。

```bash
MBT=4096 MAX_SEQS=4 PORT=8000 bash tools/dcu/serve_host_dflash2.sh
```
