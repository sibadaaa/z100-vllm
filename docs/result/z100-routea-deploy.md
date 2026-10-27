# WP1.1 图模式路线 A 上生产（去 `--enforce-eager` + `cudagraph_mode=piecewise`）

> 施工依据：BUILD [`z100-decode-prefill-optimization-construction.md`](../design/z100-decode-prefill-optimization-construction.md) §5 WP1.1
> 设计依据：DESIGN §1/L1、§7；前序实测：[`z100-cudagraph-route-a-20261021.md`](z100-cudagraph-route-a-20261021.md)
> 属主：A（Lead）。日期：2026-10-24。树 SHA `7078b5fe137feeab3750ee243daeaf172dc439d8`（分支 `dtk`）。
> 栈：4×Z100（gfx906）、DTK 26.04、torch 2.11.0、TP=4、DFlash2 k=7、c=1、greedy、MBT 8192、`max_num_seqs=4`、util 0.85。
> 臂脚本：[`tools/dcu/s27_graph_launch.sh`](../../tools/dcu/s27_graph_launch.sh)（新增，从生产脚本逐字派生，功能 diff 仅两处）。

---

## 1. 一句话结论

**路线 A 采纳**：真实 7469-token 文档上 decode step p50 **129.26 → 114.17 ms（−11.67%）**、cold TTFT **8971 → 9040 ms（+0.77%，判据 ≤2%）**、
输出 **10.81 → 11.89 tok/s（+10.0%）**；G1 **8/8 逐 token 一致**；C2 四档上下文（1K/2K/4K/8K）**全部不退化且更快（−9.8% ~ −20.1%）**。
代价：冷启动 **150 → 369 s**（首次编译 126 s；编译缓存命中后 255 s）、KV cache **759,509 → 712,748 tok（−6.2%）**、宿主 RAM 峰值约 27 GiB（配 51 GiB swap 后无 OOM）。
`--enforce-eager` 生产默认**未改**；该脚本是候选臂，回滚=删脚本用生产脚本。

---

## 2. A/B 表（唯一变量 = 图模式）

| 臂 | 启动 | step p50 (ms) | cold TTFT @7469 (ms) | acc (tok/step) | out tok/s | KV cache tok | G1 |
|---|---|---:|---:|---:|---:|---:|---|
| `eager`（B0 生产脚本） | 150 s | **129.26**（135.2 / 123.3） | **8971** | 2.667 | 10.81 | 759,509 | 基准 8/8 |
| `graph-A`（`s27_graph_launch.sh`） | 369 s（compile 126.17 s）；二次 255 s（compile 27.65 s） | **114.17**（114.2 / 114.1） | **9040** | 2.909 | 11.89 | 712,748（−6.2%） | **8/8 与 eager 逐 token 相同** |
| **Δ** | +219 s / −6.2% KV | **−11.67%** | **+0.77%** | +9.1% | **+10.0%** | — | 一致 |

- 两臂 env 逐字相同，策略指纹均 `FP 7e70fc2c130b9191`；`vllm/` 零改动。
- 日志证据：4 条 `first materialization route engaged (role=in_proj_qkvz, M=8192, …)`（GPU 编译臂物化路由没有静默失效，这是路线 A 的关键前置）；
  启动日志 `CudagraphMode.PIECEWISE`、`cudagraph_capture_sizes=[1,2,4,8,16]`；profiler 未启用（WP1.2 另起一次带 profiler 的启动）。
- 原始数据：`.s27/p0/s20_b0-mbt8192-7469_*.json`、`.s27/p1/s20_graphA-7469-real_20261024-144727.json`、
  `.s27/p1/parity/parity_graphA.json` vs `.s27/p0/parity/parity_gemv8.json`。
- acc/out 提升来自图模式改变了 draft/verify 的数值路径（接受长度上升），但 greedy 输出逐 token 不变（G1）。

### 2.1 C2 / G7：四档上下文不退化（cold，唯一 marker 破坏前缀缓存）

| 上下文 tok | eager step p50 (ms) | graph-A step p50 (ms) | Δ step | eager cold TTFT (ms) | graph-A cold TTFT (ms) |
|---:|---:|---:|---:|---:|---:|
| 1024 | 128.63 | **102.72** | **−20.1%** | 1302 | 1188 |
| 2048 | 124.89 | **108.86** | **−12.8%** | 2472 | 2496 |
| 4096 | 125.20 | **111.01** | **−11.3%** | 4504 | 4499 |
| 8192 | 128.47 | **115.86** | **−9.8%** | 9675 | 9593 |
| 7469 | 129.26 | **114.17** | **−11.7%** | 8971 | 9040 |

⇒ decode 改动在 1K/2K/4K/8K 四档**无退化**（每档都更快），TTFT 四档最大差 −1.6%~+1.0%（噪声内）。**G7（C2 部分）通过**。

---

## 3. env 全文 + 策略指纹 + 树 SHA

- 树 SHA：`7078b5fe137feeab3750ee243daeaf172dc439d8`；`git diff -- vllm/` 为空。
- 两臂 env 快照：`.s27/p0/b0-mbt8192-7469.env.txt`（eager）、`.s27/p1/graphA-mbt8192-7469.env.txt`（graph）。
  唯一差异 = 启动脚本（`--enforce-eager` vs `--compilation-config`），env 逐行相同。
- 策略指纹：`FP 7e70fc2c130b9191`（两臂一致；启动日志首行可查）。
- 派生证据：`.s27/b-fingerprint/s27-graph-launch.diff`；脚本与生产脚本 sha256 见该 diff 文件头。
- 2026-10-25 追加（不影响本报告任何数值）：两个 launcher 同步新增 `CHAT_TEMPLATE_KWARGS` env 覆盖
  （默认值逐字等于原硬编码 `{"reasoning_effort":"low"}`，即默认保留思考通道），供物理机包装脚本
  `local-llm-start.sh` 在 `REASONING=0` 时切到 `{"enable_thinking":false}`；探针走 `/v1/completions`，
  不受该 kwarg 影响，A/B 结论不变。
- 采集命令：

```bash
# 对照臂（生产脚本）
MBT=8192 MAX_SEQS=4 PORT=8000 bash tools/dcu/serve_host_dflash2.sh > .s27/p1/eager.serve.log 2>&1 &
# 处理臂
MBT=8192 MAX_SEQS=4 PORT=8000 bash tools/dcu/s27_graph_launch.sh > .s27/p1/graphA.serve.log 2>&1 &
# 两臂同口径：真实文档 7469（token 截断）+ max_tokens 96 + repeat 2；再跑 G1 parity
.venv/bin/python tools/dcu/s23_probe.py --prompt-file docs/design/z100-dual-prefill-reconciliation-projection.md \
  --truncate-tokens 7469 --max-tokens 96 --repeat 2 --tag graphA-7469-real --out .s27/p1
.venv/bin/python .s26/graph/parity.py --base http://127.0.0.1:8000 --tag graphA --out .s27/p1/parity
```

---

## 4. 判读（对照 BUILD §5 WP1.1 阈值）

| 判据 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| step 降幅 | ≥5% | −11.67% | **过 → 采纳** |
| TTFT 劣化 | ≤2% | +0.77% | **过** |
| 物化路由存活 | 4 条 `route engaged` | 4/4 | 过 |
| G1 数值 | 8 条 ×32 token 逐 token 一致 | 8/8 | 过 |
| C2（G7） | 1K/2K/4K/8K 不退化 | 每档 −9.8%~−20.1% | 过 |
| 禁用项 | 不得用 `VLLM_USE_BREAKABLE_CUDAGRAPH` | 未使用 | 过 |

**未完成/未通过的剩余门**：G8（2 次重启 + 15 min c1/c4 soak，0 crash/OOM/NaN）本窗口未做满 15 min；
G9（回滚演练）= 用生产脚本重启多次（本会话 eager 臂起停 ≥6 次）已实质演练，但未写成独立脚本。
⇒ **采纳为候选配方，进生产前需补 G8**（尤其注意 profile 运行会 segfault，见 WP1.2 报告 §7；那是 profiler 关闭路径的问题，非图模式本身）。

**风险**（与 DESIGN §7 一致，本次实测补充）：冷启动 2.5–6 min（取决于编译缓存）、KV −6.2%、宿主内存压力大（需 ≥16 GiB swap，实际配 51 GiB 才稳）。

---

## 5. 反证条件

- 若 15 min soak 出现 OOM/NaN/crash，或 c=4 出现 2026-10-21 breakable 那种静默损坏（本窗口未测 c=4），则采纳作废。
- 若用户场景要求**跨重启数值可复现**（SLA/审计），图模式不适用（编译路径跨重启不可复现，routing 文档 S14d/S14e 结论），应留 eager。
- 若后续发现 114.17 ms 的 step 只在 7469 文档成立、其它 prompt 集退化，需按 C2 补测。

## 6. 回滚方式

```bash
# 一条命令：不用图臂，起生产脚本（--enforce-eager）即回到 B0
MBT=8192 MAX_SEQS=4 PORT=8000 bash tools/dcu/serve_host_dflash2.sh
# 或删除候选脚本：rm tools/dcu/s27_graph_launch.sh
```
