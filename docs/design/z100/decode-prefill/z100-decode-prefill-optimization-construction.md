# Z100 decode / prefill 优化施工图（v1）

> 设计依据：[`z100-decode-prefill-optimization-design.md`](z100-decode-prefill-optimization-design.md)（**DESIGN**）
> 数据依据：[`z100-dual-prefill-reconciliation-projection.md`](z100-dual-prefill-reconciliation-projection.md)（**PROJ**）§1–§11
> 本文性质：**可直接执行**。每个工作包都给到 {属主, 独占文件, 命令, 判读阈值, 停止线, 回滚}。
> 交给其他 agent 时**整篇交付**，不要只给某一节——§2 的契约和 §11 的坑清单是强制项。

---

## 0. 三十秒看懂 + 任务板

五个阶段，**严格串行**；同一阶段内的工作包可并发，前提是**独占文件不重叠**。

| 阶段 | 门 | 工作包 | 属主 | 独占文件 | 前置 |
|---|---|---|---|---|---|
| **P0 测量基座** | G4 | WP0.1 基线冻结 | A | — | — |
| | | WP0.2 GEMV 派发 A/B | A | — | WP0.1 |
| | | WP0.3 纯 n-gram 上限 | A | `tools/dcu/serve_host_dflash2.sh`(加 `SPEC` 覆盖) | WP0.1 |
| | | WP0.4 策略指纹 | B | `tools/dcu/`(新文件) | — |
| | | WP0.5 探针加 `--prompt-file` | B | `tools/dcu/s20_single_probe.py` | — |
| **P1 部署+测量** | G5 | WP1.1 图路线 A | B | `tools/dcu/s27_graph_launch.sh`(新) | G4 |
| | | WP1.2 五相位 round profile | B | `tools/dcu/`(新) | WP1.1 |
| | | WP1.3 接受长度上下文扫描 | A | — | WP0.5 |
| **P2 回收** | — | WP2.x 由 WP1.2 实测分配决定 | 待定 | 见 §6 | G5 |
| **P3 q16 原子包** | G6 | WP3.1 B1 attention 宽度 | D | `vllm/v1/attention/ops/triton_unified_attention.py` | G5 |
| | | WP3.2 B2 物化阈值 | E | `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py` | G5 |
| | | WP3.3 B3 draft_block 解耦 | C | `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | G5 |
| | | WP3.4 lookup 移植 | C | `vllm/v1/worker/gpu/spec_decode/dflash2/lookup.py`(新) | WP3.3 |
| **P4 排队行为** | G7 | WP4.1 chunk 边界根因 | F | 视根因 | G5 |
| **P5 prefill** | G7 | WP5.1 rocBLAS fp16 形状审计 | G | 无（env/库路径） | G5 |

**P3 只在 WP0.3 的结论为"立项"时执行**（判据见 WP0.3 的停止线）。否则整段跳过，写清原因。

---

## 1. 读者先读（最短路径）

1. **PROJ §1** —— 为什么 decode 的口径是 `step / 接受长度`。不读这一节会把接受长度当成性能。
2. **PROJ §6.4–§6.5** —— 目标数字是怎么算出来的（TTFT 双锚点拟合：`559 ms + 1.1160 ms/token`）。
3. **DESIGN §1、§4** —— 七条杠杆的证据等级，以及为什么先做零代码 A/B。
4. **PROJ §11.3** —— 只在做 P3 时读。

### 术语（全篇统一）

| 词 | 含义 |
|---|---|
| `step` | 一次 decode 前向的墙钟，单位 ms。c=1 时等于 `srv_step_ms`（`iteration_tokens_total` 反算） |
| `acc` | 接受长度（每 step 被接受的输出 token 数），`acc = 输出 tok/s × step_ms / 1000` |
| `tok/s` | decode 吞吐，`= 1000 × acc / step_ms`。**不是**"每秒前向次数" |
| `TTFT` | 首个 token 的时间，含 fixed 开销 + prefill |
| `prefill t/s` | `prompt_tokens / TTFT`。**随长度变化**，短 prompt 不可比 |
| `B0` | 基线臂，见 §3.2 |
| `round` | 1Cat 的完整验证轮 = `draft + draft→target + target + target→draft` |

---

## 2. 契约执行细则（怎么落地 DESIGN 的 C1–C6）

| 契约 | 可执行形式 | 违反判定 |
|---|---|---|
| **C1 数值** | 8 条固定 prompt × 32 token greedy，逐 token id 比对 | 任一 token 不同 ⇒ 该改动作废 |
| **C2 不退化** | decode 改动跑 1K/2K/4K/8K；prefill 改动跑 512/709/2048/7469 | 任一档 p50 劣化 >5%（且区间不重叠）⇒ 不进生产 |
| **C3 单变量** | 一次 A/B 的 `git diff --stat` 要么空（env 臂），要么只碰一个文件；env 全集写进报告 | 一个臂两个变量 ⇒ 该臂作废 |
| **C4 出处** | 报告里每个数跟 `{SHA, env 快照文件, prompt_len, ctx, 命令}`；论断带 `【实测】/【先测】/【推断】` | 无数值出处的结论不得进摘要 |
| **C5 诊断隔离** | 诊断核开关默认关；开启时必须在报告里声明，且该组数据不得与关时混算 | 混算 ⇒ 数据作废 |
| **C6 唯一属主** | 见 §0 的独占文件列 | 两人同改一文件 ⇒ 后提交者重跑 |

**额外**：所有 A/B 报告必须贴 WP0.4 的策略指纹（一行哈希 + 全文）。

---

## 3. 底座

### 3.1 环境

- 树：`$WORKDIR`，分支 `dtk`，开工前记 `git rev-parse HEAD`。
- Python 一律 `.venv/bin/python`（**禁止**系统 `python3` / 裸 `pip`）。
- 运行时：`DTK_ROOT=$DATA_ROOT/host-runtime/dtk2604`（宿主 DTK 26.10 有 `libhipblas.so.3`，与轮子不兼容）。
- 4 卡全部空闲；生产 unit 必须先停。
- 结果目录约定 `.s27/`（**已被 `.gitignore` 覆盖**，属于本地产物）。

### 3.2 B0 基线（WP0.1 冻结）

必须冻结 **6 个点**，全部 c=1、greedy、`max_tokens=96`：

| 标签 | prompt | MBT | 备注 |
|---|---:|---:|---|
| `b0-mbt8192-7469` | 7469 | 8192 | 用户实测点，主基线 |
| `b0-mbt4096-7469` | 7469 | **4096** | **生产默认**（`serve_host_dflash2.sh:30`），必须单列 |
| `b0-mbt2048-7469` | 7469 | 2048 | chunk 边界对照 |
| `b0-512` | 512 | 4096 | 短上下文对照 |
| `b0-2048` | 2048 | 4096 | |
| `b0-1k` | 1024 | 4096 | P1.3 的起点 |

> **口径警告**：用户给的 "MBT 8192 / 2048" 是**覆盖默认值**跑的；生产默认是 **4096**。
> 报告里必须写清哪一个是生产值，否则会拿一个非生产配置当基线（本会话已犯过一次）。

**已知参考值**（用于自检，不是判据）：

| 点 | prefill t/s | step ms | acc | tok/s |
|---|---:|---:|---:|---:|
| 7469 @MBT8192 | 839.8 | 129 | 2.84 | 22.0 |
| 7469 @MBT2048 | 754.0 | 131 | 2.95 | 22.5 |

若 P0.1 复现不出 step 在 **129 ± 6 ms** 区间，先查配方（G3 策略指纹），不要往下走。

### 3.3 命令

```bash
cd $WORKDIR
export DTK_ROOT=$DATA_ROOT/host-runtime/dtk2604
mkdir -p .s27/p0

# 一个臂 = 一次启动 + 一组探测；启动后 stop 再起下一臂
MBT=8192 MAX_SEQS=4 PORT=8000 bash tools/dcu/serve_host_dflash2.sh \
  > .s27/p0/b0-mbt8192-7469.serve.log 2>&1 &

# 等 ready（日志出现 "Application startup complete"）
.venv/bin/python tools/dcu/s23_probe.py \
  --prompt-tokens 7469 --max-tokens 96 --repeat 2 \
  --tag b0-mbt8192-7469 --out .s27/p0
```

**注意**：`s23_probe.py` 的 `--prompt-tokens` 造的是**合成 prompt**，与用户的真实 7469 文档不同，
所以 acc 会不同。**acc 只在同一 prompt 源下横向可比**；跨源只能比 step / TTFT / prefill t/s。
WP0.5 加上 `--prompt-file` 之后，P1.3 才有真实文档的 acc。

### 3.4 回滚底座

```bash
# 二进制备份（任何重编前必做）
mkdir -p .s27/bak && cp -a vllm/*.so .s27/bak/ 2>/dev/null || true
# 环境回滚：所有 A/B 都是 env 臂，停服务即回；代码臂按各自 WP 的"回滚"字段
```

---

## 4. P0：测量基座（门 G4）

### WP0.1 —— 基线冻结（属主 A，~1 小时）

- **做**：按 §3.2 的 6 个点各起一次服务、跑探测、存日志与 summary。
- **产物**：`docs/result/z100-b0-freeze.md`（表 + 每个点的 env 快照 + SHA）。
- **判读**：6 个点全部有 `step_ms / acc / TTFT / prefill_tps`；`b0-mbt8192-7469` 的 step 落在 129 ± 6 ms。
- **停止线**：任一臂起不来，或 step 偏离 >10% ⇒ 先修栈（G0），不进 P0.2。
- **回滚**：无代码改动。

### WP0.2 —— GEMV 派发 A/B（属主 A，~40 分钟）

**这是全表最便宜的单项**（DESIGN §1/L2）：算子级已测 GEMV 在 M=8 是平坦 GEMM 的 **7.38×**，
端到端**从未跑过**。[`docs/archive/result/z100-18c-gemm-dispatch-m-flat-result.md:80`](../../../archive/result/z100-18c-gemm-dispatch-m-flat-result.md)

- **唯一变量**：`VLLM_DCU_FP8_GEMV_MAX_M`
- **两臂**：
  ```bash
  # 臂 1（对照，= B0）：8
  VLLM_DCU_FP8_GEMV_MAX_M=8 MBT=8192 MAX_SEQS=4 PORT=8000 \
    bash tools/dcu/serve_host_dflash2.sh > .s27/p0/gemv8.serve.log 2>&1 &
  # 臂 2（处理）：1
  VLLM_DCU_FP8_GEMV_MAX_M=1 MBT=8192 MAX_SEQS=4 PORT=8001 \
    bash tools/dcu/serve_host_dflash2.sh > .s27/p0/gemv1.serve.log 2>&1 &
  ```
  （两个端口可以同时起，但**串行跑探测**，避免抢卡。更稳的做法是一次只起一个。）
- **探针**：同 §3.3，`--tag gemv8` / `--tag gemv1`。
- **判读**：
  - step 降 **≥15%** ⇒ **采纳**，进 P1 的生产候选；同时把 DESIGN §4/D2 的结论固化为
    "verify 形状走 GEMM 派发"，P2 不再考虑重写 GEMV。
  - step 降 **5–15%** ⇒ 采纳但标记"部分成立"，在报告里写清残余在哪（用 WP1.2 的 profile 解释）。
  - step 降 **<5%** ⇒ **停**：说明 W8A16 路径的 GEMV 已不是瓶颈（S18 是 W8A8 测的），
    p3 计划里的 "GEMV 46.1 ms" 整条要重估，P2 的权重移到 L3/L4。
- **必须同时检查数值**：G1 的 8 条 greedy 逐 token 一致。派发策略变化理论上不改数值，**不一致就是 bug**。
- **回滚**：`VLLM_DCU_FP8_GEMV_MAX_M` 回 8。

### WP0.3 —— 纯 n-gram 接受长度上限（属主 A，~1 小时）

目的：给 LABD 判死或立项。CAP 是**纯 n-gram 单独做 draft** 时的接受长度——LABD 的收益不可能超过它。

- **前置小改动**（1 行）：`tools/dcu/serve_host_dflash2.sh:153` 的 `SPEC` 是硬编码，
  改成 `SPEC="${SPEC:-{\"model\":\"${DRAFT}\",\"method\":\"dflash\",...}}"`，让 env 可覆盖。
- **两臂**：
  ```bash
  # 臂 1（对照，dflash k=7）：不设 SPEC，即 B0
  # 臂 2（处理，纯 n-gram）：
  SPEC='{"method":"ngram_gpu","prompt_lookup_min":3,"prompt_lookup_max":12,"num_speculative_tokens":7}' \
    MBT=8192 MAX_SEQS=4 PORT=8002 bash tools/dcu/serve_host_dflash2.sh \
    > .s27/p0/ngram.serve.log 2>&1 &
  ```
  （`ngram_gpu` 不需要 `model`；本树 `vllm/config/speculative.py:1184-1185` 会自动填。）
- **判读**（只看 `b0-mbt8192-7469` 同 prompt 源）：
  - n-gram 臂的 **acc ≥ 2.0** ⇒ LABD **立项**，进 P3。
  - acc **1.0–2.0** ⇒ 边缘：只在 `num_speculative_tokens=15` 且 B1/B2/B3 都修完后才试，不单独立项。
  - acc **≤1.0** ⇒ **砍掉 P3 整段**，在报告里写明"LABD 对本工作负载无效"，
    并把接受长度这条杠杆降级为"仅能通过换 drafter 解决，不在本计划范围内"。
- **停止线**：`ngram_gpu` 起不来但报错与 gfx906 无关 ⇒ 记录错误、跳过本 WP、P3 按"未判定"处理（**不得**当成立项）。
- **回滚**：env 臂，无。

### WP0.4 —— 策略指纹（属主 B，~半天，零卡）

**唯一非性能项，但它是 C3/C4 的执行手段。**

- **做**：
  1. 在 `tools/dcu/` 新建 `dcu_policy_fingerprint.py`：启动时收集全部 `VLLM_DCU_*` / `VLLM_GFX906_*` /
     `VLLM_QWEN35_*` / `VLLM_S24_*` / `VLLM_SPEC_*` 的**实际生效值**（不是 env 原文——
     要处理默认值与 clamp，例如 `_GEMV_MAX_M = min(max(env,0), _GEMV_MS_MAX)`），
     按 key 排序，算 `sha256[:16]` 作指纹。
  2. 输出两行：`<指纹> key=value ...`（全文）；并在报告模板里引用指纹。
  3. 接进 `serve_host_dflash2.sh` 的启动头部（`echo` 一行），不需要改 vLLM 本体。
- **判读**：同一台机器同一个 `.env` 连起两次，指纹必须一致；改任一旋钮指纹必须变。
- **停止线**：若发现某旋钮"设了但被 clamp/忽略"（如 `GEMV_MAX_M=16` → 8），
  **必须在指纹里可见**并以 `WARN` 打印——这正是本会话"167 vs 129"事故的根因类型。
- **回滚**：新增文件，删即可。

### WP0.5 —— 探针加 `--prompt-file`（属主 B，~2 小时）

- **做**：给 `tools/dcu/s20_single_probe.py` 加 `--prompt-file PATH`（与 `--prompt-tokens` 互斥）：
  读纯文本，用服务端 tokenizer 的离线副本计数，**报告实际 token 数**（不要假设）。
- **判读**：给一个 8 KB 文本，能报出 token 数且与 `/tokenize` 端点（若可用）一致。
- **为什么必须**：P1.3 的接受长度扫描**必须用同一份真实文档截断**，
  合成 prompt 的接受长度不可比（PROJ §1 已经吃过"两条 prompt 比 acc"的亏）。
- **回滚**：加法式改动，去掉参数分支即可。

---

## 5. P1：部署 + 测量（门 G5）

### WP1.1 —— 图路线 A 上生产（属主 B，~半天）

- **唯一变量**：去掉 `--enforce-eager` + 加 `--compilation-config`。
- **做法**：**新建** `tools/dcu/s27_graph_launch.sh`，从 `serve_host_dflash2.sh` 派生，
  只改两处（**不要**直接改生产脚本）：
  ```diff
  -  --per-request-spec-decode-metrics summary --enforce-eager \
  +  --per-request-spec-decode-metrics summary \
  +  --compilation-config '{"cudagraph_mode":"piecewise","cudagraph_capture_sizes":[1,2,4,8,16]}' \
  ```
- **前置检查（G0 之前）**：宿主 swap 必须 ≥16 GiB；否则先扩（实测编译臂会把 16 GiB 吃满）。
- **预期（PROJ §5/A、【实测】**）：step 127.5 → **118.4 ms（−7.1%）**；
  TTFT 709 用例 1.38 → 1.35 s（持平）；c=4 聚合 48.4 → 51.6 tok/s；冷启动 99 s → **7.4 min**；KV −8%。
- **判读**：
  - step 降 **≥5%** 且 TTFT 劣化 **≤2%** ⇒ 采纳。
  - TTFT 劣化 >5% ⇒ 检查日志有没有 4 条 `first materialization route engaged`
    （没有 = 物化路由在编译臂失效，参照 `z100-cudagraph-route-a-20261021.md` §1 的①②修复）。
  - step 降 <5% ⇒ 记录为不可判定，**不采纳，不重跑**。
- **必须同时过**：G1（8 条 greedy 逐 token 一致）。
- **回滚**：删掉 `s27_graph_launch.sh`，用回生产脚本。
- **禁止**：`VLLM_USE_BREAKABLE_CUDAGRAPH=1`（已判死：数值坏 + VM fault）。

### WP1.2 —— 五相位 round profile（属主 B，~半天，**G5 的唯一内容**）

**这是整条流水线的量纲来源。** 没有它，P2 不许开工。

- **做**：在 WP1.1 的候选上，用 `torch.profiler` 或 rocprofv3 抓 **20 个稳定 decode step**，
  按 1Cat 的口径切五段并归因：

  | 相位 | 归属算子 |
  |---|---|
  | `draft` | 草稿塔全部算子 + selector |
  | `draft→target` | 草稿 KV 写 / 同步 / 小 kernel gap |
  | `target` | 16 层全注意力 + 48 层 GDN + 256 次/步的量化 GEMM + allreduce |
  | `target→draft` | 隐藏态搬运 / gap |
  | `complete round` | 五段之和 |

- **必须同时给出**（否则 G5 不通过）：
  1. 五段之和与 `srv_step_ms` 的差 **<5%**（差额单列"gap/未知"）。
  2. 每段的 top-5 算子 + 每算子的调用次数/步。
  3. `ates::mm` 必须**只计 decode 步**——历史上把 prefill 的 `aten::mm` 算进 decode 过
     （PROJ 引用的 E2C 口径说明），这一次要按 NVTX 名切干净。
- **判读**：不需要判"好坏"，只需要**可信**。若五段之和与 step 差 >5%，先修采集，不出结论。
- **产物**：`docs/result/z100-five-phase-profile.md`，含分配表（每项 ms/步 + 占比 + 屋顶）。

### WP1.3 —— 接受长度上下文扫描（属主 A，~半天）

- **唯一变量**：prompt 长度（同一份文档，截断到 1024 / 2048 / 4096 / 7469）。
- **用**：WP0.5 的 `--prompt-file`，`max_tokens=96`，`repeat 2`。
- **判读**：
  - 接受长度**随长度单调下降** ⇒ 是**长度效应**，LABD/P3 的收益预期成立。
  - 在 1024 就已经是 ~2.8 ⇒ 是**这条文档的内容**效应，换文档再测一次；
    仍如此 ⇒ 接受长度不是本工作负载的可优化项，P3 的"接受长度 4.0"目标作废。
  - 7469 点与 `b0-mbt8192-7469` 一致（±0.3）⇒ 探针口径正确。
- **产物**：`docs/result/z100-acceptance-vs-context.md`。

---

## 6. P2：按 profile 回收（条件式，门 G5）

**本节不预设任务清单。** WP1.2 出表后，按"可回收 ms"从大到小取，每项一个工作包 `WP2.k`，
每项必须满足：可回收 ≥5 ms/步，且有对应的【实测】屋顶。

前置候选（按 DESIGN §1 的顺序，**不是执行顺序**）：

| 候选 | 位置 | 判据 |
|---|---|---|
| L3 allreduce | `vllm/envs.py:173` `VLLM_DCU_CUSTOM_AR`（当前生产传 `--disable-custom-all-reduce`） | profile 显示 allreduce ≥20 ms/步 |
| L4 rocBLAS 草稿塔/lm_head | 无代码，先查 dtype/形状 | profile 显示该组 ≥25 ms/步 |
| GEMV 残余 | `gfx906_fp8.py` | 仅当 WP0.2 判"部分成立" |

**每一项都必须**：单变量 → G1 → C2 四档上下文 → 报告。禁止一次上两项。

---

## 7. P3：q16 原子包（门 G6）

> **前置**：WP0.3 判"立项"。否则整段跳过。
> **原子性**：WP3.1/3.2/3.3 三项同属一个验收门 G6，**不得分开合入**（DESIGN §4/D3）。
> 三项各自独占文件，可并发开发，但只在全部完成后才允许起服务测。

### WP3.1 —— B1：attention 支持 16 行验证（属主 D）

- **文件**：`vllm/v1/attention/ops/triton_unified_attention.py`（独占）
- **现状**：`:67` `GFX906_VERIFY_MAX_QUERY_LEN = 8`（硬编码）；
  `gfx906_small_q_decode`(:243) 要求 `max_seqlen_q <= GFX906_VERIFY_MAX_QUERY_LEN`(:262)；
  `:1478-1483` 还有第二处同源判断。
- **做**：
  1. 把常量提升为可配置（env `VLLM_GFX906_VERIFY_MAX_QUERY_LEN`，默认 8 保持不变）。
  2. **两处**判断都改用它（`grep -n GFX906_VERIFY_MAX_QUERY_LEN` 必须只剩定义处）。
  3. 补测试：`tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py` 增加 `max_seqlen_q=16` 的用例，
     断言走的是 split-KV 路径（复用现有 `monkeypatch.setattr(unified_attn, "ENABLE_GFX906_ATTN_VERIFY_3D", True)` 的写法）。
- **验收**：
  - 静态：pytest 该文件全绿（G2）。
  - 数值：`max_seqlen_q=16` 的输出与 `max_seqlen_q=1` 逐行 loop 的参考实现相对误差 ≤2e-3（fp16）。
  - 性能：单独微基准下，16 行 verify 的 attention 时间 ≤ 8 行 × 1.4（**不是** ×2）。
- **停止线**：若 16 行在 split-KV 下的 attention 时间 >8 行 × 2，说明 split 数（`_decode_num_segments`）
  对该行数不适用，**停**，把结果写进报告，不改 split 逻辑（那是另一个工作包）。
- **回滚**：env 默认 8 ⇒ 行为与今天完全一致。

### WP3.2 —— B2：物化阈值不能吃掉 q16 的 decode（属主 E）

- **文件**：`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py`（独占）
- **现状**：`:132` `_FORCE_MIN_M = 16`；`:441` 闸门 `m < _min_m(role, None) or m < _FORCE_MIN_M`；
  `:73` `_ROLE_MIN_M` 里 `down_proj` / `o_proj` / `out_proj` / `in_proj_qkvz` / `in_proj_z` 门槛都是 **16**。
  q16 的 decode `M = 16` ⇒ **两个比较都为 False ⇒ 路由生效** ⇒ 每层当步 dequant。
- **做**（二选一，**优先方案 1**）：
  1. 把"是否 prefill"从 `M` 推断改成**显式传入**：在 `propose()` 侧给 kernel 挂一个
     `is_decode_step` 标志（或复用现有的 decode/prefill 分支），闸门加 `and not is_decode_step`。
  2. 若显式标志改动面太大：把 `_FORCE_MIN_M` 与 `_ROLE_MIN_M` 的**最小值**抬到 **32**，
     并写清"q16 的 M=16 不得路由"的原因注释 + 一个断言测试。
- **验收**：
  - 静态：`tests/kernels/quantization/test_gfx906_fp8_emu.py` 加一个用例，
    断言 `M=16` 在 decode 上下文下 `try_prefill` 返回 `None`。
  - 运行：q16 臂启动日志里**不得**出现 `first materialization route engaged`（那是 prefill 的）。
- **停止线**：若抬阈值导致 prefill 在 M=16..31 的短 chunk 上变慢（用 512 token prefill 验），
  则必须走方案 1，不许用方案 2 挤过去。
- **回滚**：改常量 / 加标志，各自独立可回。

### WP3.3 —— B3：`draft_block` 与验证宽度解耦（属主 C）

- **文件**：`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`（独占）
- **现状**：`:71` `self.num_query_per_req = 1 + self.num_speculative_steps`；**全树没有 `draft_block` 符号**。
- **参考实现**（1Cat，`.refs/1cat-vllm` @ `c4f6245`，`dflash/speculator.py`）：
  ```python
  self.draft_block = get_dflash_model_draft_tokens(self.speculative_config)   # :58
  self.num_query_per_req = 1 + self.draft_block                               # :59
  draft_query_capacity = self.max_num_reqs * self.num_query_per_req           # :60
  if draft_query_capacity > self.max_num_tokens:                              # :61-70
      self.max_num_tokens = draft_query_capacity
      self.input_buffers = InputBuffers(...)          # draft 的 query 批次不得吃 target 的 prefill 预算
  if self.draft_block < self.num_speculative_steps:                           # :79-86
      logger.info("%s emits %d model drafts and lookup may fill %d additional ...")
  ```
- **附带移植的小件**：1Cat 的 `vllm/config/speculative.py:96` `get_dflash_model_draft_tokens()`
  （+ `_get_dflash2_checkpoint_draft_tokens()`、`uses_adaptive_dflash_lookup()`），
  以及 `SpeculativeConfig` 上的 `ngram_assist` 字段。**本树没有这些符号**（`grep -rn` 过）。
  语义：`method != "dflash"` 或不带 `ngram_assist` 时**原样返回 `num_speculative_tokens`**（= 今天的行为），
  只有显式开 `ngram_assist` 才去读 checkpoint 的训练 block 宽度。这个"默认不生效"的写法是 B3 能安全落地的关键。
- **做**：照上面四点移植；`draft_block` 从 draft 模型的 `dflash_config` 读，**不要**从 CLI 读。
  所有原先用 `num_speculative_steps` 表示"模型输出宽度"的地方改成 `draft_block`
  （`grep -n num_speculative_steps` 逐个过，`_generate_draft` 里的 `view(num_reqs, …, -1)` 是重点）。
- **验收**：
  - `num_speculative_tokens=7`（= 今天）时，`max_num_tokens` / KV / step 全部与 B0 一致（C2）。
  - `num_speculative_tokens=15` 时能起服务，且 draft 前向的输出形状是 `draft_block` 而不是 16。
- **停止线**：若 `max_num_tokens` 扩张导致 KV pool 下降 >10%，先算清容量账再继续（可能吃掉长上下文）。
- **回滚**：`draft_block` 回退为 `num_speculative_steps`（即今天的行为）。

### WP3.4 —— lookup 移植（属主 C，依赖 WP3.3）

- **新文件**：`vllm/v1/worker/gpu/spec_decode/dflash2/lookup.py`（独占）
- **参考**：1Cat `.refs/1cat-vllm/vllm/v1/worker/gpu/spec_decode/dflash2/lookup.py`（10.5 KB）
  - `_suffix_lookup_kernel`(:20)、`_fuse_draft_kernel`(:115)、`suffix_lookup`(:236)、`fuse_draft`(:288)
  - 准入逻辑逐字在 `:161` / `:165` / `:184`（`take_head` / `take_tail` / `has_tail`）
  - **不要**照抄它的扫描核：本树 `vllm/v1/worker/gpu/spec_decode/ngram/speculator.py:21`
    已有更并行的 `_ngram_scan_kernel`（二维 grid `(num_reqs, n_blocks)` + finalize 归约），
    接它做 `suffix_lookup` 的等价物。
- **env**（`vllm/envs.py`，独占该文件的追加段）：`VLLM_DFLASH2_LOOKUP_{NSTRONG=6, AGREE=0,
  NMIN_TAIL=4, LONG_MIN=6, SEARCH=1<<30, ENTRY_STREAK=2, STICKY=3, ADAPTIVE=1, CHEAP_CONTEXT=0}`。
  **注意 `AGREE=0` 的语义**：`agreement >= 0` 恒真 ⇒ 头替换退化成 `match_len >= NMIN`，
  这是 1Cat 的默认行为，不要"顺手修"。
- **挂点**：在 `propose()` 末尾（本树 `dflash/speculator.py:441` 起），照 1Cat `:1312-1315` 的顺序：
  ```python
  if self.draft_block < self.num_speculative_steps:
      self.draft_tokens[:num_reqs, self.draft_block:].zero_()
  if self.draft_logits is not None:
      self._cache_draft_logits(candidate_ids, num_sample)
  self._apply_lookup(num_reqs)
  ```
- **验收**：
  - G1：`num_speculative_tokens=7` 时输出与 B0 逐 token 一致（lookup 不应启用：`draft_block < num_speculative_steps` 为假）。
  - `num_speculative_tokens=15` 时，8 条 greedy 仍逐 token **与 7 的臂语义等价**（acc 可以变，token 序列不能变坏；
    用同一 prompt 的 target-only 参考对齐）。
  - `take_flags` 计数在长文档 prompt 上 >0，在纯推理 prompt 上 ≈0（准入确实在判内容）。
- **停止线**：若 q16 臂的 step 相对 q8 涨 >1.8×，说明 M=16 的成本没有如预期被摊薄，
  **停**，先回 WP1.2 的 profile 看 M 敏感项，不要继续调参数。

---

## 8. P4：chunk 边界（属主 F，门 G7）

### WP4.1 —— 337 ms/chunk 边界根因

- **已知**：7469 token 从 1 chunk（MBT 8192）切成 4 chunk（MBT 2048）多付 **1012 ms**，即 **337 ms/边界**（PROJ §3）。
  1Cat 用 MBT 4096 + 平衡切分，没有可见的边界惩罚。
- **实验**（固定 4096 token，只改 MBT）：
  `MBT=1024 / 2048 / 4096` 各跑一次 4096-token prefill，斜率 = 每边界成本。
- **候选根因**（按可能性排序，逐个排他）：
  1. **GDN prefill 每 chunk 重算前缀状态**（48 层 × chunk 数）——最可能；
  2. fp16 物化 workspace 每 chunk 重填（`:158` 的 `_workspace` 是单槽、跨层复用）；
  3. attention 的 chunk 边界效应（**理论上是否**：单 chunk 与多 chunk 的因果 token-pair 数相同）。
- **判读**：斜率 <50 ms/边界 ⇒ 记录为"Z100 正常"，MBT 4096 可用，无进一步动作。
  斜率仍 >200 ms/边界 ⇒ 出根因报告，**先不修**，交 DESIGN 复评是否立项。
- **停止线**：若根因指向 GDN，**不要**在 P4 里改 GDN（它是另一个工作流），只出报告。

---

## 9. P5：prefill rocBLAS fp16 形状审计（属主 G，门 G7）

- **目标**：prefill `839.8 → 940–1030 t/s`（+13~25%）。**没有 2×**（SOL 1452 t/s，当前 57.8%）。
- **做**（只调库与形状，不改算法）：
  1. 打印 `torch.mm` 在 prefill 形状（M=7469/4096，N/K 按角色）下的**实际后端**
     （rocBLAS vs hipBLASLt vs Tensile 路径）。
  2. 检查 `ROCBLAS_TENSILE_LIBPATH` 是否指向 DTK 26.04 的完整库；对照 hipBLASLt 是否可用。
  3. 单旋钮扫 `VLLM_DCU_FP16_PREFILL_MIN_M`（只影响路由阈值，不改算法）。
- **判读**：prefill 提升 ≥10% ⇒ 采纳；<5% ⇒ 记为"已到 rocBLAS 形状极限"，写进报告，
  prefill 目标从 940–1030 下调为"当前 840 为实际可达"。
- **停止线**：任何需要改 rocBLAS 库或换 DTK 版本的动作，**不在本图范围内**，只出建议。

---

## 10. 门禁（G0–G9）与统计口径

| 门 | 类型 | 检查 | 通过条件 | 不通过 |
|---|---|---|---|---|
| **G0** | 全局 | 栈身份 + 4 卡空闲 + swap ≥16 GiB | 日志栈版本匹配当前树 | 停，修栈 |
| **G1** | 全局 | 8 条固定 prompt × 32 token greedy 逐 token 比对 | 全一致 | 该改动作废 |
| **G2** | 全局 | `pre-commit run` + 相关 pytest | 全绿 | 不合入 |
| **G3** | 全局 | 策略指纹可打印，且与报告一致 | 一致 | 补，不算通过 |
| **G4** | 阶段 | WP0.1–0.5 全部有完整 summary | 6 点 + 2 臂 + 1 上限 + 指纹 + 探针 | 不进 P1 |
| **G5** | 阶段 | 五相位 profile，五项和与 step p50 差 <5% | 差 <5% | 不进 P2/P3 |
| **G6** | 阶段 | q16 原子包：B1/B2/B3 各有一份单变量证据 | 三份齐 | 不进 P3 联测 |
| **G7** | 阶段 | C2 四档上下文全测 | 无退化 | 不进生产 |
| **G8** | 阶段 | 2 次重启 + 15 min c1/c4 soak | 0 crash/OOM/NaN | 不进生产 |
| **G9** | 阶段 | 回滚演练 | 能回 B0 且 smoke 通过 | 不进生产 |

**统计口径**：每组 `repeat 2`；两臂 p50 差 <5% 且区间重叠 ⇒ 判**不可判定**；
只有不可判定或与信任基线矛盾时，才**只重跑该臂一次**，不整套重跑。

---

## 11. 坑清单（每条都踩过，别重踩）

| 坑 | 症状 | 规避 |
|---|---|---|
| **把非生产配置当基线** | 生产默认 `MBT=4096`，用户实测用的是 8192 | §3.2 必须同时冻结 4096 与 8192 |
| **用两条 prompt 比 acc** | acc 2.84 vs 5.82 被误读成"路径差异" | acc 只在同一 prompt 源下横向比 |
| **拿旧 step 的分解做回收估算** | 240.9 ms 的分解被套到 129 ms 上 | WP1.2 之前不许写 P2 的任务清单 |
| **静默 clamp** | `GEMV_MAX_M=16` 被 `_GEMV_MS_MAX=8` 截成 8 | WP0.4 的指纹必须显示 clamp |
| **诊断扰动** | 开诊断核后延时不代表生产 | C5：诊断组数据单独出，不混算 |
| **编译臂静默关掉物化路由** | TTFT 1.35 → 3.35 s | WP1.1 判读里查 4 条 `first materialization route engaged` |
| **breakable cudagraph** | 数值坏 + GPU VM fault | 明令禁止 |
| **mixed batch + spec decode** | 静默输出损坏 | 不在本图范围内，不要顺手开 |
| **prefix cache 静默归零** | `hits_total=0` 但看起来"开着" | 报告里必须带 `hits_total` |
| **一次上两项优化** | 之后再也拆不开 | C3 + 一轮一个 WP |

---

## 12. 交付物

### 每阶段必交

- [ ] `docs/result/z100-b0-freeze.md`（WP0.1）
- [ ] `docs/result/z100-gemv-dispatch-ab.md`（WP0.2）
- [ ] `docs/result/z100-ngram-upper-bound.md`（WP0.3）
- [ ] `tools/dcu/dcu_policy_fingerprint.py` + 报告模板（WP0.4）
- [ ] `tools/dcu/s20_single_probe.py --prompt-file`（WP0.5）
- [ ] `tools/dcu/s27_graph_launch.sh` + `docs/result/z100-routea-deploy.md`（WP1.1）
- [ ] `docs/result/z100-five-phase-profile.md`（WP1.2，**G5 的唯一凭据**）
- [ ] `docs/result/z100-acceptance-vs-context.md`（WP1.3）
- [ ] P3：`lookup.py` + 三个阻塞改动 + `docs/result/z100-q16-atomic.md`（G6）
- [ ] P4：`docs/result/z100-chunk-boundary-rootcause.md`
- [ ] P5：`docs/result/z100-prefill-rocblas-audit.md`

### 报告模板（每份都必须有这 6 段）

1. **一句话结论**
2. **A/B 表**（臂名 / 唯一变量 / step p50 / acc / TTFT / prefill t/s / 是否采纳）
3. **env 全文 + 策略指纹 + 树 SHA**
4. **判读**（对照本 WP 的阈值，写清过没过）
5. **反证条件**（"什么数据会推翻这个结论"）
6. **回滚方式**（一条命令）

### 最终验收（Must 全绿才算完成）

| 指标 | Must | 实测 | 判定 |
|---|---:|---:|---|
| decode tok/s @7469 | ≥23.7 | | |
| TTFT @7469 | ≤8.65 s | | |
| prefill t/s @7469 | ≥866 | | |
| C1 8 条 greedy | 全一致 | | |
| C2 四档上下文 | 无退化 | | |
| G7–G9 | 全通过 | | |
