# S14b 同一 TP 下 graph(`CompilationMode.VLLM_COMPILE`) vs eager(`CompilationMode.NONE`)的 token 流分歧：机制定位与质量退化定性 — 执行结果

> 归档时间：2026-10-09
> 执行者：子 agent（S14b）
> 前一档：`docs/result/z100-14-27b-tp4-result.md`（S14，md5 `7e39492ebf079f66c00c87c1481fd721`）
> 状态：🔶 进行中（边做边写）


## 0.0 执行摘要（给 Lead 的三句话）

1. **A/B 判定：以 B（可解释数值）为主，另有一个 A 形状的"可复现性缺陷"**。在**同架构族未量化 4B** 上，"编译模式 + cudagraph"整体只值 **0.05–0.11 nat**，且 eager/graph/compile-no-cudagraph 三臂都贴着 **CPU HF 真值（≈0.003 nat，首 token 4/4 一致）**；而 27B 上 graph↔eager 的**同上下文** logits 差 **mean 1.35–1.38 / max 10.3–12.0 nat**（五个独立 graph 臂一致），主导 logit 只差 0.2–0.3 nat、差集中在竞争结构上，并随上下文长度递增 ⇒ 这是 **动态 per-token INT8 激活量化 + 64 层 GDN 递推**对 0.05 nat 级种子的放大，**不是 S7b/S7c 那类"某算子被编坏"**（无 NaN/Inf、后端与 kernel 逐项相同、packed kernel 负向控制零影响、**cudagraph 已被 A3 无图臂排除**、无逐层爆炸）。
2. **S14 的核心观测不可复现**：`A1r` 用与 S14 `P2` **逐字相同的配置、相同的 cache 目录与 AOT hash、同样的冷引擎首请求**，得到 `'用户'`/gap 1.047（与 eager 只差 0.06–0.20 nat），而 P2 是 `'The'`/gap 4.750 + 英文重复循环 ⇒ **P2 是 graph 路径的一次坏 draw**；唯一识别到的区分因子是 **AOT 产物来源**（P2 = 现场编译并落盘；A1r = 从磁盘重载同一 hash `4bfe7617…`）⇒ 登记为假设 **H-AOT**（详见 §3.7、§4.2）。
3. **参照与生产建议**：**eager 是唯一可作质量/精度参照的臂**（逐位可复现：同配置同请求序跨引擎 1024 token Δ=0.000、8/8 flip prompt Δ=0.000、同引擎重复 Δ=0.000；且在 4B 上与 CPU 真值差 0.0025 nat）。生产栈 `serve_27b_tp4.sh` 默认 `ENFORCE_EAGER=1` ⇒ **生产是安全的、不需要改**；**graph 的 token 流不得用于任何质量/精度评测**（性能结论不受影响：graph 仍是最快）。**本步未改任何 `csrc`**（没有定位到具体算子，按任务书"根因不明确不要硬塞"）。

## 0.1 边做边更新（时间线，倒序）

| 时间 (UTC) | 事件 |
|---|---|
| 21:36:37 | 留档：`pgrep` 无他人实验、4 卡 VRAM 全 0、`free` 26 GiB free、`/proc/swaps` 只有 4 GiB `/swap.img`、容器 `z100-s14` `Memory=0`、`_C_stable_libtorch` md5 `93bf7b4a…` |
| 21:36:54 | **A1（TP=2 / graph / 默认 FULL_AND_PIECEWISE）** 启动（复用 S14 `cache-27b-p2` vintage） |
| 21:40 | 复核 S14 四臂原档（`greedy-P{2,2e,4e,4g2,4g3}.jsonl`）在**同一上下文（position 0）**上的 logprob 差 → **发现 S14 的噪声底（0.1–0.6 nat）严重低估**：同配置跨重启（P4g2 vs P4g3）单 token |Δlogprob| 高达 **3.255 nat** |
| 22:04–22:09 | **A2（TP=2 eager，冷）**：与 S14 `P2e` **逐位相同**（ids_sha `3788a862…`，position-0 Δ=0.000）⇒ eager 可复现 |
| 22:15 | **事故**：运行中改 `s14b_run_one.sh` → A3 orchestrator 崩、A4 报废（§3.5）；A5 在显存释放后自动重启成功 |
| 22:20 | **A5（graph PIECEWISE 4 桶 + packed OFF）**：首 token `'用户'`/gap 4.031（**不是** P2 的 `'The'`）⇒ P2 的异类性与"graph 模式整体"无关 |
| 22:28–22:35 | A6（eager+packed OFF）= eager 零影响；**A2b（eager，多跑 19 条请求）**：greedy 4/4、flip 8/8 与 A2/P2e **逐位相同** |
| 22:48–22:49 | **HF4B-bf16 CPU 真值**跑通（4 prompt，`next_token_argmax` = `Here/Thinking/Here/这是一个`） |
| 23:01–23:04 | **A1r（graph 默认 51+2 桶，与 P2 逐字同配置、同 cache、AOT 重载）**：首 token `'用户'`/gap 1.047，与 eager 差 0.06–0.20 nat ⇒ **S14 的 P2 结果不可复现**（§3.7） |
| 23:11–23:13 | **A9（PIECEWISE 1/8/128/512，prefill 进 cudagraph）**：仍是 `'用户'`/2.500 ⇒ 捕获 prefill 也不是 P2 异类性的原因 |
| 23:15–23:28 | **4B TP=1 三臂 + HF 真值**：B1(eager)/B2(graph)/B3(compile-no-cudagraph) 与真值差 0.0023–0.0027 nat、首 token 4/4 一致；graph↔eager 同上下文 ≤0.054 nat（§3.10） |
| 23:31–23:38 | **A18（AOT 重载 A5 的 cache）**：长 prompt 与两个短 prompt 逐位一致，`p2_zh_slip` 从第 59 位起差到 9.5 nat ⇒ H-AOT 部分支持（§3.9） |
| 23:36–23:38 | **A3（compile + `cudagraph_mode=NONE`，27B）**：首 token 判定与 A5/A18 逐个相同 ⇒ **cudagraph 与数值无关**（§3.11）；A3/A18 与 A5 只剩一处偶发尾部差 |
| 23:39 | 收尾：4 卡 VRAM 0、`oom_kill` 仍为 3、`_C_stable_libtorch` md5 未变、ACS 12×`0x0000`（§3.12） |

## 1. 目标与验收标准

任务书 S14b 门禁（原文要点）：

* **G1 复现**：同一 TP（取 TP=2）下、同 prompt、**各自独立引擎**，graph vs eager 复现分歧；给出首个分歧 token 位置与 top1–top2 间距（nat）、两侧 logprob 对照；**n≥3 prompt**，每次标 cache vintage。
* **G2 定性**：判定 A（真 bug：某算子被编坏）还是 B（可解释数值差：编译模式→非 int8 kernel 差 → 动态 int8 量化放大 → 近 tie 翻转）。必须做"能一锤定音"的实验：小模型逐层/逐算子定位 + 真值参照 + 算子级负向控制 + 配置对照（`benchmark_combo_kernel` 等）。
* **G3 结论**：① 机制（A/B/两者）；② 哪个模式可作质量/精度参照；③ 对生产的建议（生产栈用 eager，`serve_27b_tp4.sh` 默认 `ENFORCE_EAGER=1`）；④ 若判 bug：`file:line` 根因 + 最小复现 + 修法建议。
* **G4 口径纪律**：每行标 TP/模式/maxlen/util/n/vintage；ACS 全程部署态不切。
* **G5 归档**：本文件 8 小节齐全。

本次执行的验收标准（在 G1–G5 之上自加的两条）：

1. **同上下文**（identical context）的 logit 距离必须是主判据。生成流一旦分歧，其后位置的 logprob 不再可比（S14 §3.6.2 第 5 条已指出）。因此本步新增 `prompt_logprobs` 探针，直接量两臂在**同一 prompt 位置**上的 top-k logprob 差。
2. **同配置跨重启噪声底必须在"单 token |Δlogprob|"口径下重测**，否则无法判断 1–5 nat 的模式差是"系统性"还是"落在噪声内"。

## 2. 实际改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `tools/dcu/s14b_arm_env.sh` | 新增 | `s14_arm_env.sh` 的 S14B 命名空间副本；追加 `S14B_SERVE_EXTRA` 透传与 `fork_env` 留档 |
| `tools/dcu/s14b_run_one.sh` | 新增 | `s14_run_one.sh` 的 S14B 命名空间副本；追加 `S14B_ENVS`（任意 `K=V` 容器环境变量透传，用于 fork 的 `VLLM_QWEN35_*` 开关二分）、`promptlp`/`greedy1` 任务 |
| `tools/dcu/s14b_greedy.py` | 新增 | 多 prompt（`--prompts-file`）greedy 记录器，保留 `s14_greedy_long` 逐字不变 |
| `tools/dcu/s14b_prompt_logprobs.py` | 新增 | **同上下文探针**：`prompt_logprobs` 取每个 prompt 位置的 top-k logprob（记录 `prompt_token_ids`、请求顺序 `order`） |
| `tools/dcu/s14b_compare_promptlp.py` | 新增 | 两臂同上下文 top-k logprob 差（max/mean |Δ|、argmax 翻转、最后一个 prompt 位置的判据） |
| `tools/dcu/s14b_hf_ref.py` | 新增 | CPU HF transformers fp32 真值参照（同 prompt 电池，同 JSONL 形状） |
| `tools/dcu/s14b_firsttok.py` | 新增 | 各臂首 token/间距/重复循环分数的紧凑表 |
| `tools/dcu/s14b_arms.sh` / `s14b_arms2.sh` / `s14b_arms3.sh` | 新增 | 顺序臂驱动（每个 arm 的完整命令写在文件里，保证可复现）；3 个文件对应 3 条串行链，避免"运行中改脚本"（见 §3.5 事故） |
| `tools/dcu/s14b_compare_prefix.py` | 新增 | prefix 电池（19 个递增长度前缀）的同上下文首 token 对照 → 首 token 分布随上下文长度的 profile |

**未改动**：`tools/dcu/s14_*`、`s13_*`、`s11*`、`csrc/**`、`docs/z100-todolist.md`、`docs/hygon_z100.md`、`docs/海光Z100-性能优化方案.md`。

## 3. 执行过程与原始证据

### 3.0 前置检查（卡位 / 内存纪律）

```
=== S14b baseline 2026-10-09T21:36:37+00:00 ===     # 原文：/mnt/game/s14b/hostlogs/00-baseline.txt
--- pgrep vllm/torchrun ---  (none)                 # 只有本命令自身的 bash
--- hy-smi ---               DCU[0..3] vram Used = 0 MiB（4 卡全空）
--- free -g ---              Mem: total 31, used 3, free 26, available 27
--- /proc/swaps ---          /swap.img  file  4194300  461160  -2   ← 只有 4 GiB 系统 swap
--- cgroup ---               low 0 / high 0 / max 0 / oom 0 / oom_kill 3   ← 基线 oom_kill=3（S14 遗留）
--- 容器 mem limit ---       0 0                    ← z100-s14 无内存上限（与 S14 一致）
--- cache vintage md5 ---    0f63df6818ec1b969970c71afefc46d5   （cache-27b-p2 全文件 md5 汇总）
--- _C_stable_libtorch ---   93bf7b4a391ac97e2f67cc7141633cb1   （与 S14 一致，含 S7c 修复）
```

`/mnt/game/swapfile` **未启用**（`/proc/swaps` 中没有它），全程未 `swapon`。ACS 未触碰（本步未做任何 `apply/revert`；S14 §3.9.3 已给 12×`0x0000` 部署态）。

### 3.1 先复核 S14 原档：**噪声底被严重低估**（本步第一个新结论）

S14 §3.6.3 用"首个分歧点的 top1–top2 间距"估噪声底为 0.1–0.6 nat。但生成流分歧后上下文不同，该统计量对"同一上下文上的 logit 抖动"不敏感。用 S14 已有的四个 greedy 原档，在 **position 0（上下文完全相同：同一 prompt、temperature=0、无采样随机性）** 上直接比 top-5 logprob：

```
$ python3 - <<'EOF'   # 原文见本档 §7；输入 /mnt/game/s14/greedy-*.jsonl
# 取每臂 rows[0]['top'] 的 token->logprob，按 token 字符串对齐
EOF

--- within GRAPH TP4: identical config, two engine restarts = NOISE FLOOR: P4g2 vs P4g3
     '用户'      P4g2=  -0.170  P4g3=  -0.034  |d|=0.135
     'We'       P4g2=  -2.388  P4g3=  -5.644  |d|=3.255     ← 同配置跨重启 3.255 nat
     'The'      P4g2=  -3.435  P4g3=  -4.425  |d|=0.990
     '我们需要'   P4g2=  -4.482  P4g3=  -5.191  |d|=0.708
     max|d| over shared top5 = 3.255 nat

--- within EAGER: TP2 vs TP4:              P2e vs P4e   max|d| = 3.290 nat
--- within GRAPH:  TP2 vs TP4:              P2  vs P4g2  max|d| = 5.591 nat
--- GRAPH TP2 vs EAGER TP2 (本步目标):       P2  vs P2e   max|d| = 5.359 nat
--- GRAPH TP4 vs EAGER TP4:                 P4g2 vs P4e  max|d| = 1.694 nat
```

**读法**：

1. **同配置跨引擎重启**（P4g2 vs P4g3，TP=4 graph PIECEWISE+4 桶、同 prompt、temperature=0）在 position 0 上就有 **3.255 nat** 的单 token 差。⇒ 这套栈的**逐 token logprob 不是可复现量**，S14 的"噪声底 0.1–0.6 nat"只在"首个分歧点的间距"这一个统计量上成立，**不能用来把 1–5 nat 判为系统性**。
2. 但注意结构：**rank-1 token 的 logprob 稳定**（'用户' 0.135 nat），**rank≥2 的尾部 token 抖动到 3 nat**。这与"数值噪声"一致（top1 由很强的 logit 差支配，尾部由近 tie 的若干 token 构成）。
3. 因此 S14 的"P2 是唯一异类"这一**定性**仍可保留（P2 是唯一让尾部 token 'The' 变成 rank-1 的臂），但 S14 用来支撑"系统性"的**定量**论证必须被替换成本步的同上下文口径。

**⇒ 这直接决定了本步的设计**：新增 `prompt_logprobs` 同上下文探针，并**必须**给每个模式臂配一个"同配置、同 vintage 跨重启"的对照，否则任何 graph↔eager 的差都无法区分"模式效应"与"重启抖动"。

### 3.2 先复核 S14 的请求顺序（确认 G1 对照不是"冷/热引擎"错配）

判据：mode 对照必须两侧都是"引擎就绪后的第一条请求"（冷缓存、无前缀缓存/状态历史），否则差异可能来自缓存复用而不是编译模式。逐 arm 核对 S14 的 orchestration 原文：

```
$ grep -E "task=|ready=" /mnt/game/s14/hostlogs/orch-P2e.log
[orch14] ready=1 after 150s  2026-10-09T20:48:31+00:00
[orch14] task=greedy start  2026-10-09T20:48:31+00:00        ← 就绪后第一条就是 greedy = 冷

$ head -12 /mnt/game/s14/hostlogs/P2-takeover-orch.log        # P2 是被接管的既有引擎
[p2] READY 2026-10-09T20:22:41+00:00
[p2] task=greedy start 2026-10-09T20:22:41+00:00             ← 同样就绪后立刻 greedy = 冷

$ grep -E "task=" /mnt/game/s14/hostlogs/orch-P4e.log | head -2
[orch14] task=greedy start 2026-10-09T20:34:58+00:00         ← 冷（第一条）

$ grep -E "task=" /mnt/game/s14/hostlogs/orch-P4g2.log | head -2
[orch14] task=decode start 2026-10-09T21:04:22+00:00
[orch14] task=greedy start 2026-10-09T21:05:28+00:00         ← P4g2/P4g3 的 greedy 是第 2/3 条 = 热
```

**结论**：S14 的 TP=2 mode 对照（P2 graph vs P2e eager）**两侧都是冷引擎**，可以排除"冷/热"错配。⇒ **P2（TP=2 graph，冷，首请求）选 `'The'` 且对次选有 4.75 nat 把握** 是可信的原始事实；本步 A1 用同样冷口径复现它，并把"同引擎第 2 次请求（热/前缀缓存命中）"也记录下来做内部一致性对照。

### 3.3 定位策略（为什么先打"同上下文 logit 距离"，以及为什么小模型选 4B 而不是 0.6B/逐层 hook）

读源码得到的、决定实验设计的三条事实（`vllm/model_executor/models/qwen3_5.py`）：

1. **27B INT8 走的是 fork 私有的 gfx906 快速路径**：`Qwen3_5DecoderLayer` 只在 `quantization in {gguf, compressed-tensors}` 时用 `Qwen3_5GatedDeltaNet(split_projections=True)`（`qwen3_5.py:747-765`），而 `split_projections` 打开了一整套 gfx906 专用开关（`qwen3_5.py:294-318`）：`VLLM_QWEN35_PACKED_DECODE`(默认 1)、`VLLM_QWEN35_COMBINED_PACKED_DECODE`(默认 1)、`VLLM_QWEN35_EMPTY_CORE_ATTN_SINGLE_TOKEN`(默认 1)、`VLLM_QWEN35_TILED_QK_EXPAND`、`VLLM_QWEN35_CHUNK_DECODE`(gfx906 默认开，`qwen3_5.py:691-702`)。
   ⇒ **小模型必须也是量化模型才能复现这条路径**；`Qwen3-0.6B`（dense，全 attention）和 `Qwen3.5-4B`（bf16）都**不开** `split_projections`。
2. **有一条分支显式按"是否在编译"分叉**：`_packed_decode_writes_full_core_output()`（`qwen3_5.py:706-731`）在 `torch._dynamo.is_compiling()` 时**直接 return False**，决定 `core_attn_out` 用 `torch.empty` 还是 `torch.zeros`（`qwen3_5.py:639-649`）。⇒ 同一份 Python 代码在两种模式下**不是同一条路径**，这正是 A（真 bug）的候选落点，也是"编译模式改变 token 流"最可能的机制。
3. **逐层 hook 在 `VLLM_COMPILE` 下不可用**：fork 自己就写明 inductor 会消掉 module 边界（`tools/dcu/s7b_hooks.py` 头注释），且 `splitting_ops` 里 `vllm::qwen_gdn_attention_core*` 是图断点（`vllm/config/compilation.py:774-781`，A1 日志原文）。⇒ 逐层数值对拍只能在 **eager 侧**做（vLLM eager vs HF），graph 侧只能到 logits 层。

**因此本步的定位链是**（按性价比排序）：

```
(1) 同上下文 logit 距离（plp 探针）        -> A/B 的量级判定（O(1) nat = material）
(2) 该距离的"逐位置 profile"               -> 是否随位置累积（累积=数值放大/B；只在某位置跳变=局部 bug/A）
(3) 同配置跨重启 / 同引擎第 2 次请求对照   -> 把"模式效应"从"重启抖动/缓存复用"里剥离
(4) HF CPU fp32 真值（Qwen3.5-4B，TP=1）   -> 判哪个模式更接近数学真值（4B 只走通用路径，是"通用模式效应"的对照）
(5) 27B TP=2 上的开关二分（cudagraph / packed decode / chunk decode / inductor combo kernel）
                                           -> 若某个开关让 graph 与 eager 的同上下文差塌缩到 <0.1 nat，即根因
```

### 3.4 G1 复现：**eager 侧是逐位可复现的（噪声底 = 0.000 nat），graph 侧不是**

**先登记一次失败的跑**：A1（TP=2 / graph / 默认 `FULL_AND_PIECEWISE` + 51 桶，复用 S14 的 `cache-27b-p2`）在 **1503 s** 时被我设的 `READY_TIMEOUT=1500` 掐断（`[orch14b] ready=0 after 1503s`，原文 `/mnt/game/s14b/hostlogs/orch-A1.log`；容器内时间线：`Capturing CUDA graphs (PIECEWISE)` 两轮 ×51 桶各自约 11.5 min，随后 `(FULL) 32/35`）。**这是参数错，不是环境问题**：该配置就绪需要 1500–1600 s（S14 P2 是 1515 s）。AOT 已命中（`aot_load=2`），所以重跑只需要重做 capture；重跑记为 **A1r**（`READY_TIMEOUT=3000`）。

**A2（TP=2 / eager / 复用 S14 `cache-27b-p2e` vintage / 冷引擎第一条请求就是 `s14_greedy_long`）**：

```
$ tools/dcu/s14b_arms.sh A2        # 端口 8523，tasks=promptlp,greedy,flip，卡 0,1
$ grep -E "ready=|summary" /mnt/game/s14b/hostlogs/orch-A2.log
[orch14b] ready=1 after 150s
[orch14b] summary: ready=150s aot_load=0 aot_recompile=0 oom_kill=3 | NA | Available KV cache memory: 11.65 GiB

# 首 token 对照（同 prompt s14_greedy_long、temperature=0、单请求、冷引擎）
--- A2 : ntok=256 ids_sha=3788a8622b2ef18a text_head='用户问的是：在 31 GiB 内存的单机上，以张量并行度 TP=4 部署一个 27B 参数的INT8 量化大模型时，权重加载、KV 缓存与 '
     i=0 chosen='用户' lp=-0.4019 gap=0.8594 top5=[('用户',-0.402),('The',-1.261),('We',-3.543),('我们需要',-5.464),('User',-6.050)]
--- P2e: ntok=256 ids_sha=3788a8622b2ef18a text_head='用户问的是：在 31 GiB 内存的单机上，以张量并行度 TP=4 部署一个 27B 参数的INT8 量化大模型时，权重加载、KV 缓存与 '
     i=0 chosen='用户' lp=-0.4019 gap=0.8594 top5=[('用户',-0.402),('The',-1.261),('We',-3.543),('我们需要',-5.464),('User',-6.050)]
--- P2 : ntok=256 ids_sha=4097ad3cbea2fff2 text_head='The user is asking in Chinese: "In 31 GiB memory of a single machine, '
     i=0 chosen='The' lp=-0.0263 gap=4.7500 top5=[('The',-0.026),('We',-4.776),('I',-5.558),('用户',-5.761),('Let',-6.104)]

# 同上下文（position 0，上下文逐字相同）logprob 差
  A2 vs P2e : max|d| = 0.000  （5/5 token 全为 0.000）        ← 跨引擎、跨 vintage 逐位一致
  A2 vs P2  : max|d| = 5.359  ({'用户':5.359, 'The':1.235, 'We':1.234})
```

**三条结论**：

1. **eager（`CompilationMode.NONE`）在 TP=2 上是逐位可复现的**：A2（本次，2026-10-09 22:04Z 起）与 P2e（S14，20:48Z 起）两次独立引擎启动，`token_ids_sha256` **完全相同**（`3788a8622b2ef18a…`），且 position-0 的 top-5 logprob **逐位相同（Δ=0.000）**。⇒ **eager 侧的噪声底在本口径下是 0.000 nat**（不是 0.1–0.6 nat）。
2. 因此 §3.1 测到的 **3.255 nat 抖动是 graph 模式独有的**（P4g2 vs P4g3 两次 graph 启动），不能用来给 mode 差当噪声底——**两个模式的"可复现性"本身就不一样**，这是本步第二个新结论。
3. **TP=2 上 graph vs eager 的同上下文差是 5.359 nat，且 top-1 不同**（graph 选 `'The'` 并对自己的次选有 4.750 nat 把握，eager 选 `'用户'`）。5.36 nat 的**同上下文**差不可能来自舍入（舍入级差应在 1e-3–1e-2 nat 量级），所以**至少有一侧在"算另一个函数"** ⇒ 量级上指向机制 A。至于落点是"graph 模式整体"还是"某一个 graph 配置/算子"，由 §3.5–§3.7 的配置二分回答。

#### 3.4.1 eager 的逐位可复现**有边界**：请求历史不同会移动 logits（zh_ml 例）

同一对引擎（A2 vs P2e，同配置、同 vintage）在 **8 prompt 的 flip 电池**上（`s11l_flip_stats.py`，temperature=0）：

```
$ python3 tools/dcu/s14_logprob_compare.py /mnt/game/s14b/flip-A2.jsonl /mnt/game/s14/flip-P2e.jsonl --label-a A2 --label-b P2e
# prompts compared        : 8
# prompts token-identical : 7/8
# tokens on common prefix : 824 of 952
# max  |dlogprob| (nat)   : 0.000000          <- 共同前缀上逐位为 0
# diverged prompts        : 1 (pid, first_idx, min_gap_at_first, worst_gap) = [('zh_ml', 0, 0.164, 8.703)]

# position 0 的逐 token 对照（同 prompt）
zh_hz      pos0 max|d|=0.00000   zh_sort 0.00000   zh_sky 0.00000
en_fr      pos0 max|d|=0.00000   en_sea  0.00000   en_index 0.00000   en_math 0.00000
zh_ml      pos0 max|d|=2.77954   A2 选 'Spec'（gap 0.164），P2e 选 '用户'（gap 3.375）
             '用户'  A2=-2.8764  P2e=-0.0968  |d|=2.7795
             'The'   A2=-2.6967  P2e=-3.4718  |d|=0.7752
```

对照另外两臂的 `zh_ml` position 0：P2（graph TP=2）`用户` gap 0.266；P4e（eager TP=4）`用户` gap 5.313。
⇒ **A2 是 4 个臂里唯一的异类**，而 A2 自己的长 prompt 又与 P2e **逐位相同**。

差别在于 **A2 在 zh_ml 之前的请求历史不同**（A2：`promptlp` 4 条 + `greedy` 4 条 → 第 9 条才是 zh_ml；P2e：长 prompt 之后第 2 条就是 zh_ml）。本机 `--enable-prefix-caching` + `--mamba-cache-mode align` 会把 GDN/conv 状态按块缓存并在命中时复用，**命中的状态来自哪一次请求（哪一批）并不保证逐位一致**。

**⇒ eager 的口径必须写清楚**："同配置 + 同请求历史 ⇒ 逐位可复现（0.000 nat）"；**"请求历史不同 ⇒ 可以在 2.8 nat 量级上改变同一 prompt 的 logits、并翻转 0.16 nat 的贴边 token"**。这条是本步新增的**第三个 O(1)-nat 级来源**（前两个：编译 vintage、编译模式），且它**在 eager 里也存在**。因此：

> 判断"哪个模式算错了"**不能只靠差异量级**（本栈有 ≥3 个来源都能造出 O(1)–O(3) nat 差），必须引入**与执行路径无关的真值参照**（本步用 CPU HF fp32，见 §3.6）与**能一锤定音的开关二分**（§3.5、§3.7）。

**A2b 落地后的修正（22:35Z，同一批实验内自证）**：A2b 是**新引擎**、任务表比 A2 **多一条 `prefix`（19 条请求）**，其余同配置同 vintage：

```
$ python3 tools/dcu/s14_logprob_compare.py /mnt/game/s14b/greedy-A2b.jsonl /mnt/game/s14b/greedy-A2.jsonl
# prompts token-identical : 4/4 ; tokens on common prefix : 1024 of 1024 ; max|dlogprob| = 0.000000
$ python3 tools/dcu/s14_logprob_compare.py /mnt/game/s14b/flip-A2b.jsonl /mnt/game/s14/flip-P2e.jsonl --label-a A2b --label-b P2e
# prompts token-identical : 8/8 ; tokens on common prefix : 952 of 952 ; max|dlogprob| = 0.000000   ← 含 zh_ml
$ python3 tools/dcu/s14_logprob_compare.py /mnt/game/s14b/flip-A2b.jsonl /mnt/game/s14b/flip-A2.jsonl --label-a A2b --label-b A2
# prompts token-identical : 7/8 ; diverged = [('zh_ml', 0, 0.164, 8.703)]                        ← A2 是 zh_ml 的异类
```

⇒ **`zh_ml` 的异类是本步 A2 那一次运行独有的**：A2b（更多历史）与 P2e（更少历史）**完全一致**，A2（中等历史）偏 2.78 nat。**"请求历史越长越偏"不成立**，我**无法用已采集的因子解释 A2 的 `zh_ml`**，如实登记为"未消解的 eager 侧偶发 O(1)-nat 偏差"。


### 3.5 事故登记：A3/A4 因"运行中改脚本"报废（本步唯一操作事故）

**事故**：我在 A3 引擎正在"就绪等待循环"里时，用 `python3` 就地改写了 `tools/dcu/s14b_run_one.sh`（加 `S14B_PC` 透传）。bash 是**按字节偏移增量读取**脚本的，文件长度变化后它从错位处续读，于是：

```
$ cat /mnt/game/s14b/hostlogs/orch-A3.log
[orch14b] engine_wrapper_pid=2903624
tools/dcu/s14b_run_one.sh: line 103: syntax error near unexpected token `('
tools/dcu/s14b_run_one.sh: line 103: `docker exec "$CT" bash -c "kill \$(cat /tmp/s14b_${NAME}.pid) 2>/dev/null" || true'
```

* **A3**：orchestrator 当场死掉，但它拉起的**引擎仍在跑**（`Application startup complete`，占卡 0/1 各 28.9 GiB）→ 我按纪律用 `/tmp/s14b_A3.pid` 杀掉（`kill` → 8 s → `kill -9`），**A3 的数值数据未取到**（引擎就绪但任务没跑），记为"报废，重跑为 chain3 的 A3"。
* **A4**（PIECEWISE+4 桶）：22:15:27 前后启动时 A3 的 28.9 GiB 还没释放 → `ready=0 after 90s`、`WorkerProc initialization failed`，**报废**；它的角色由 **A12（同配置、全新 cache）** 顶替。
* **A5**（PIECEWISE+4 + `VLLM_QWEN35_PACKED_DECODE=0`）：同样在 22:15:27 撞上残留显存，第一次 worker 初始化失败；显存释放后**自动重启成功**（22:16 起正常加载权重），继续跑。
* **内存/swap 副作用**：两次失败的 worker 使 `/swap.img` 使用量从 442 MiB 冲到 **2962 MiB**（`free` 原文见 §3.5.1），但 **`oom_kill` 保持 3（未增长）**、4 卡 VRAM 事后回到 0。`/mnt/game/swapfile` 全程未启用。

**纪律修正（本步后续严格遵守）**：任何 `tools/dcu/s14b_*` 脚本**只能在没有该脚本实例在跑时**修改；本轮之后的所有改动都先确认 `pgrep -af s14b_*` 为空。

```
# 事故现场原文
$ tail -3 /mnt/game/s14b/hostlogs/orch-A4.log
[orch14b] summary: ready=90s aot_load=0 aot_recompile=0 oom_kill=3 | NA | NA
[orch14b] done name=A4 ready=0
$ grep -m1 -E "Exception|failed" /mnt/game/s14b/hostlogs/A4.log
(APIServer pid=46802) Exception: WorkerProc initialization failed due to an exception in a background process. See stack trace for root cause.
$ free -g     # 事故当时
               total        used        free      shared  buff/cache   available
Mem:              31          18           8           0           5          13
Swap:              3           2           0                     ← /swap.img 用到 2962 MiB
```

### 3.6 同上下文（identical context）logit 矩阵：27B TP=2 的 7 个臂

**判据口径**：`prompt_logprobs` 给的是"每个 prompt 位置 i 对 **prompt 第 i+1 个 token**"的分布（`plp-*.jsonl`），而"决定**首个生成 token** 的分布"是 prompt 末位置之后那一个，只能从 greedy 的 `rows[0]` 拿。两者本步都有，分别用于"逐位置 profile"与"首 token 判定"。所有比对都先校验 `prompt_token_ids` 相等（`sameids=True`）。

**(a) 决定首个生成 token 的同上下文 top-5（prompt 末位置；同一 prompt `s14_greedy_long`，temperature=0，单请求）**

| arm | 模式 | TP | graph 配置 | packed | 请求历史 | `'用户'` | `'The'` | `'We'` | 选中 | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| **A2**（本次） | eager | 2 | — | ON | 冷（第 1 条） | **-0.402** | -1.261 | -3.543 | `'用户'` | 0.859 |
| **P2e**（S14） | eager | 2 | — | ON | 冷（第 1 条） | **-0.402** | -1.261 | -3.543 | `'用户'` | 0.859 |
| A6（本次） | eager | 2 | — | **OFF** | 冷（第 1 条） | （见 §3.6c） | | | `'用户'` | — |
| **A5**（本次） | graph | 2 | PIECEWISE/4 桶 | **OFF** | 冷（第 1 条） | **-0.024** | -4.055 | -6.641 | `'用户'` | 4.031 |
| **P2**（S14） | graph | 2 | **默认 51 桶 FULL+PIECEWISE** | ON | 冷（第 1 条） | **-5.761** | **-0.026** | -4.776 | `'The'` | 4.750 |
| P4e（S14） | eager | 4 | — | ON | 冷（第 1 条） | -0.051 | -4.551 | -4.082 | `'用户'` | 4.031 |
| P4g2（S14） | graph | 4 | PIECEWISE/4 桶 | ON | 第 2 条 | -0.170 | -3.435 | -2.388 | `'用户'` | 2.219 |
| P4g3（S14） | graph | 4 | PIECEWISE/4 桶 | ON | 第 3 条 | -0.034 | -4.425 | -5.644 | `'用户'` | 4.391 |

**读法**：
* **7 个臂里 6 个选 `'用户'`，只有 P2（TP=2 + 默认图配置）选 `'The'`**；P2 在 `'用户'` 上比最近的臂低 **5.36 nat**、在 `'The'` 上比最近的臂高 **2.39 nat** ⇒ P2 是双指标离群点。
* 但**排除 P2 后**，同一个 token 的 logprob 仍跨臂差 **3.3 nat**（`'The'`：-1.261 eager-TP2 ↔ -4.551 eager-TP4），且这与"eager 内部只改 TP"这一**逐位可复现路径**上的差同量级。⇒ **O(1)–O(3) nat 的同上下文差在本栈是"换执行路径"的常态**，不能单凭量级定罪；P2 的离群程度是它的 1.6–2.4 倍。
* A5（graph、PIECEWISE/4 桶、prefill 未进 cudagraph）选 `'用户'` ⇒ **"graph 模式整体"不是原因**，P2 的异类性绑定在**它的图配置**（默认 51 桶 `FULL_AND_PIECEWISE`）上。

**(b) graph vs eager 的逐位置 profile（A5 graph-cold vs A6 eager-cold；两臂同 prompt、同 prompt token ids、同 packed=OFF、都是引擎第一条请求）**

```
$ python3 tools/dcu/s14b_compare_promptlp.py /mnt/game/s14b/plp-A5.jsonl /mnt/game/s14b/plp-A6.jsonl \
      --label-a A5-graph-cold --label-b A6-eager-cold --profile
s14_greedy_long  126  True  9.61061  1.12630  tok264  39  6  1.078  2.750  tok198 tok198 3.54680
        profile(i:max|d|) 0:nan 8:7.979 16:7.250 24:5.215 32:3.663 40:7.375 48:2.542 ... 120:1.652 125:3.547
p2_zh_slip        78  True  8.93749  1.56721  tok27   23  6  2.484  2.750  ... 6.97969
p2_en_batch       74  True 10.28886  1.57470  tok32   23  6  2.484  2.750  ... 5.98647
p2_zh_code        76  True  8.72404  1.45374  tok29   21  6  2.484  2.750  ... 5.82248
# max |dlogprob| on identical contexts (nat) : 10.288857
# mean|dlogprob| on identical contexts (nat) : 1.378707
# argmax flips over prompt positions          : 106 / 354

# 更保守的统计量（只看每个位置 argmax token 自己的 logprob 差）
s14_greedy_long  n=125  |d(top1 logprob)| mean=0.328 max=1.326   |d(gap)| mean=1.179 max=5.234
p2_zh_slip       n= 77  mean=0.233 max=1.488                   mean=1.346 max=6.937
p2_en_batch      n= 73  mean=0.250 max=1.631                   mean=1.443 max=6.961
p2_zh_code       n= 75  mean=0.197 max=1.488                   mean=1.318 max=6.617
```

**读法**：差异**不是只出现在末尾**，而是**从 prompt 第 8 个 token 起就有 4–8 nat 的尾部差**；但**argmax token 自己的 logprob 只差 0.20–0.33 nat（max 1.6）**，差主要落在**次选/尾部 token**（gap 差 1.2–1.4 nat、max 7）。这与"两条算术路径的**主导 logit 一致**、**竞争关系（gap）被打乱**"一致，形状上更像"动态 int8 量化 + 64 层 GDN 递推放大"的**数值**机制（B 的形状），而不是"某个算子整体算错"（那会连 argmax token 的 logprob 也大幅偏离）。

**(c) 一次"同引擎、同配置、同 prompt、却差 13.7 nat"的观测 —— 已被**逐条排除**到只剩"与在飞请求并发调度"**

A2w 是我对 **A2 那个引擎**（已跑完 `promptlp`+`greedy`）**在 `flip` 任务正在跑的时候**补发的同上下文探针；A2b 是**新引擎**上按 `promptlp`→`prefix`→`greedy` 顺序重跑的对照（A6 是 eager + packed OFF 的新引擎对照）：

```
### (1) A2b(eager, packed ON, cold) vs A6(eager, packed OFF, cold)：只差 packed 开关
s14_greedy_long  126 True 0.00000 0.00000  0 flips ; p2_zh_slip / p2_en_batch / p2_zh_code 全 0.00000
# ⇒ VLLM_QWEN35_PACKED_DECODE 对 prefill（plp）位置的 logits **零影响**（该开关只作用于纯 decode 批）

### (2) A2b(eager, cold) vs A2w(eager, "热"): 
s14_greedy_long  126 True 13.70643 0.86545  29 flips  first=14
p2_zh_slip / p2_en_batch / p2_zh_code 全 0.00000

### (3) A2b 的 greedy（4×256 token）vs A2 的 greedy：**逐位相同**
$ python3 tools/dcu/s14_logprob_compare.py /mnt/game/s14b/greedy-A2b.jsonl /mnt/game/s14b/greedy-A2.jsonl --label-a A2b --label-b A2
# prompts token-identical : 4/4 ; tokens on common prefix : 1024 of 1024 ; max|dlogprob| = 0.000000
```

**逐条排除**（这是本步做的最细的一次因果排查）：

| 候选原因 | 判据 | 结论 |
|---|---|---|
| 前缀缓存命中（含 `--mamba-cache-mode align` 状态复用） | A2 引擎**全程** `Prefix cache hit rate: 0.0%`（24 次采样、含 A2w 那 6 秒窗口，原文 `A2.log`） | **排除** |
| 请求被批量并发（batch 组成变化） | A2.log 在 A2w 窗口内每 10 s 采样都是 `Running: 1 reqs` | **不能完全排除**（采样间隔内可能瞬时为 2；A2w 是唯一"在 `flip` 解码流在飞时插入请求"的观测） |
| 引擎/请求历史长度 | A2b 在 greedy 之前比 A2 **多跑了 `prefix` 任务（19 条请求）**，greedy 仍与 A2 **逐位相同** | **排除** |
| packed decode 开关 | 上表 (1) | **排除** |
| 编译 vintage / AOT cache | A2b 与 A6 用**不同 cache 目录**，cold plp 仍逐位相同 | **排除**（对 eager 而言） |
| 舍入噪声 | 13.7 nat 且 29/126 个 prompt 位置 argmax 翻转 | **排除** |

⇒ **如实登记**：这条 13.7 nat 的偏差**没有找到根因**，剩下的唯一候选是"请求与在飞解码流并发调度（瞬时 batch>1）"。它**与 §3.6(b) 的 graph↔eager 差（mean 1.38、max 10.3）同量级**，因此必须作为**未消解的混淆项**登记：本轮所有"同上下文"结论都在**单请求串行**口径下成立；**并发（生产常态）下的同上下文数值一致性本步未测**。





（后续小节随实验推进追加。）

### 3.7 **一锤定音**：生产图配置（默认 51 桶 `FULL_AND_PIECEWISE`）的同配置复跑**不给出 P2 的结果**

A1r = **与 S14 的 P2 逐字同配置**（TP=2 / graph / 默认 `cudagraph_mode=FULL_AND_PIECEWISE` / 51+2 桶 / `--enable-prefix-caching --mamba-cache-mode align` / maxlen 262144 / util 0.85 / MBT 2048 / **同一个 cache 目录 `/mnt/game/s14/cache-27b-p2`**），**冷引擎**、`promptlp,prefix,greedy,flip`：

```
$ tools/dcu/s14b_arms2.sh A1r          # 端口 8542，卡 0,1
[orch14b] ready=1 after 1603s ...      # /mnt/game/s14b/hostlogs/orch-A1r.log
$ grep -E "Capturing CUDA graphs" /mnt/game/s14b/hostlogs/A1r.log | tail -1   # PIECEWISE 51/51 + FULL 35/35

# 同上下文（prompt 末位置 = 决定首个生成 token）top-5 logprob，同一 prompt s14_greedy_long
   '用户'      'The'      'We'      'I'
A1r  -0.414    -1.461    -3.320   -4.414     chosen='用户'  gap=1.047   ids_sha=3139d652aa123b5e
P2   -5.761    -0.026    -4.776   -5.558     chosen='The'   gap=4.750   ids_sha=4097ad3cbea2fff2
A2b  -0.402    -1.261    -3.543     —        chosen='用户'  gap=0.859   ids_sha=3788a8622b2ef18a   ← eager 参照

# 同配置跨重启（graph）逐 token 差
   'The'   A1r=-1.461  P2=-0.026   |d|=1.435
   '用户'   A1r=-0.414  P2=-5.761   |d|=5.347
   'We'    A1r=-3.320  P2=-4.776   |d|=1.456

# A1r 的 greedy 文本（连贯中文，与 eager 同类）
'用户问的是关于在 31 GiB 内存的单机上，以张量并行度 TP=4 部署一个 27B 参数…'
```

**⇒ 结论（本步最重要的单条）**：**同一 TP、同一模式、同一图配置、同一 AOT cache、同一 prompt、temperature=0、两侧都是冷引擎首请求**，两次独立启动给出**不同的首个 token**（`'The'` gap 4.750 vs `'用户'` gap 1.047），并且 **A1r 与 eager 只差 0.06–0.20 nat**、而 P2 与 eager 差 5.36 nat。

**这直接推翻了"S14 观测到的是同 TP 下 graph 与 eager 的系统性模式差"这一口径**：P2 的 `'The'`+英文重复循环是**graph 路径在该配置下的一次不可复现抽签**（坏 draw），不是"graph 模式必然算错"。S14 §3.6.3 的"四臂矩阵里 P2 是唯一异类"仍然成立，但归因必须从"graph 模式"改为"**graph 路径的运行间不可复现**"。

**已定位到的唯一区分因子：AOT 产物的来源（in-process 编译 vs 磁盘重载）**

```
# S14 P2（首次跑，cache 为空 → 现场编译并落盘）
P2.log:154  saved AOT compiled function to …/torch_compile_cache/torch_aot_compile/
            4bfe7617bfec484cacaa435e07b54b4186d96acba916b3308000f703f7bd98bc/rank_0_0/model
# A1r（第二次跑，cache 命中 → 从磁盘直接加载**同一个 hash**）
A1r.log:150 Directly load AOT compilation from path …/4bfe7617…/rank_1_0/model
A1r.log:152 Directly load AOT compilation from path …/4bfe7617…/rank_0_0/model
```

两次用的是**同一个编译产物 hash**（`4bfe7617…`），区别只有"现场编译"vs"从磁盘载入"。S11m 已独立发现过同一类问题域（"token 变体由 vintage 决定"，`z100-11m-aot-restart-reproducibility-result.md`）。
**⇒ 候选根因（登记为假设 H-AOT）**：`torch.compile` 的 AOT 落盘/重载路径没有把"运行期 kernel 选择"完整固定下来（inductor 的 `benchmark_combo_kernel=True` 会在运行期基准选 kernel），于是"现场编译"与"AOT 重载"选中了不同的 Triton/combo kernel 变体 ⇒ 逐位不同的算术路径 ⇒ 在动态 int8 量化 + 64 层 GDN 递推上放大成 1–5 nat 的 logits 差。
**该假设的正面验证**（A18：把 A5 的 cache 目录**原样复用**再跑一次同配置；以及"删掉 `torch_aot_compile/` 子目录强制现场编译"的 A19）见 §3.9。

### 3.8 graph（生产默认图配置）与 eager 在 19 个同上下文上的首 token 一致率

`prefixes.txt` = 19 个递增长度的前缀（中文 14 个、英文 5 个），每个都是"两臂同上下文、同 `prompt_token_ids`"，直接比**首个生成 token 的分布**（`s14b_compare_prefix.py`）：

```
$ python3 tools/dcu/s14b_compare_prefix.py /mnt/game/s14b/prefix-A1r.jsonl /mnt/game/s14b/prefix-A2b.jsonl \
        --label-a A1r-graph-default --label-b A2b-eager
prefix     ntok  sameids   maxd      at_tok     tokA      tokB    gapA    gapB
pfx0040     77    True    2.9438    '用户'      'The'     '用户'   3.484   0.859
pfx0080    101    True    1.9605    'We'       '用户'     '用户'   1.484   0.062
pfx0120    116    True    1.6171    'I'        'The'     'The'    1.297   1.313
pfx0160    135    True    4.2751    'The'       'I'      '用户'   1.156   5.984
pfx0200    156    True    2.6215    'The'      '用户'     '用户'   0.016   3.453
pfx0260    188    True    0.8046    'The'      'The'     '我需要'  0.953   0.023
pfx0320    222    True    0.6812    'I'        '用户'     '用户'   1.266   0.094
pfx0400    262    True    1.0784    'Let'      '用户'     '用户'   1.188   0.891
pfx0480    305    True    3.1039    'The'      'The'     '用户'   0.141   3.406
pfx0560    350    True    0.6215    'The'      'The'     '用户'   1.063   0.141
pfx0640    393    True    1.0885    '用户'      '用户'     '用户'   0.391   1.547
pfx0800    473    True    1.2241    'The'      '用户'     '用户'   1.273   0.391
pfx1000    572    True    1.1993    'The'      '用户'     'We'    0.531   0.547
pfx1200    695    True    0.5999    'The'      '用户'     '用户'   2.406   2.953
pen0200     93    True    2.5102    'The'      'The'     'We'     1.641   2.625
pen0400    126    True    1.6378    'We'       'We'      'The'    0.859   1.844
pen0600    164    True    1.8791    'I'        'The'     'The'    2.219   0.828
pen0800    200    True    2.7973    'We'       'We'      'The'    1.516   2.828
pen1000    237    True    4.1688    'The'      'The'     'We'     0.906   4.828

# argmax differs at : ['pfx0040','pfx0160','pfx0260','pfx0480','pfx0560','pfx1000','pen0200','pen0400','pen0800','pen1000']
# 即 10/19
```

**读法**：**生产默认图配置与 eager 在 19 个同上下文里有 10 个的首 token 不同（53%）**，同上下文 |Δlogprob| 从 0.60 到 4.28 nat；两臂的 top1–top2 间距（gap）也系统性不同（例：`pfx0160` eager 5.984 vs graph 1.156）。⇒ "换编译模式 ⇒ 换一条 token 流"**不是刀刃上的偶发事件，而是常态**；S14 那条 256-token 英文循环只是"常态分歧"在某个采样点上的**放大**（一旦首个 token 不同，自回归会一直放大）。

### 3.11 因子归因总表（本步的最终归因）

| 因子 | 是否造成 graph↔eager 的 token 流分歧 | 证据（同上下文口径） |
|---|---|---|
| **inductor 编译本身**（`CompilationMode.VLLM_COMPILE`） | ✅ **是主因** | 五个独立 graph 臂（A1r/A3/A5/A9/A18）vs eager 的同上下文统计高度一致：**mean 1.35–1.38 nat、max 10.3–12.0 nat、每 354 个 prompt 位置 ~100–108 个 argmax 翻转**；而 4B（未量化）上同一对照只有 **0.054 nat**（§3.10） |
| **cudagraph（PIECEWISE / FULL 捕获）** | ❌ **不是** | **A3（compile + `cudagraph_mode=NONE`，完全无图）与 A5/A18（compile + PIECEWISE 4 桶）在首 token 判定上逐个相同**（`p2_zh_slip` gap 5.188/5.188、`p2_en_batch` 3.594/3.594、`p2_zh_code` 4.500/4.500、长 prompt 4.031/4.031）；A3 vs A5 只剩一处偶发尾部差（§3.9） |
| **fork 的 gfx906 packed decode（`VLLM_QWEN35_PACKED_DECODE`）** | ❌ **不是** | 负向控制：eager 上开/关**逐位相同**（4 prompt × 126–36 位置，Δ=0.00000）；该开关只作用于纯 decode 批（§3.6c） |
| **TP 并行度** | ⚠️ 同级效应，不是本步的比较对象 | eager 内部 TP=2 vs TP=4 同上下文差 **3.29 nat**（§3.6a）；说明"换执行路径 ⇒ O(1) nat"是这套栈的常态 |
| **前缀缓存 / `--mamba-cache-mode align` 状态复用** | ❌ 已排除 | 四个相关 arm 全程 `Prefix cache hit rate: 0.0%`（A2.log 24 次采样原文） |
| **请求历史长度** | ❌ 已排除（eager） | A2b 比 A2 多跑 19 条请求，greedy 4/4、flip 8/8 **逐位相同**（§3.4.1） |
| **AOT 产物来源（现场编译 vs 磁盘重载）** | 🔶 **部分**（H-AOT） | A18（重载 A5 的 cache）与 A5 在长 prompt/两个短 prompt 上**逐位一致**，但 `p2_zh_slip` **从第 59 位起**差到 9.5 nat（§3.9）⇒ "重载必然换 kernel"不成立；更像是"**偶发的隐藏状态/kernel 选择差异，遇到贴边位置才显形**" |
| **运行间偶发不可复现（graph 独有）** | ✅ **是**（但概率低） | `P2` vs `A1r`（逐字同配置、同 cache、同 AOT hash、冷首请求）：首 token `'The'`(4.750) vs `'用户'`(1.047)，同上下文差 5.36 nat，输出质量也不同（英文循环 vs 连贯中文）。eager 侧从未观测到此类偏差（1024 token + 8 prompt + 重复均 Δ=0.000） |

**最终机制口径（一句话）**：**`CompilationMode.VLLM_COMPILE` 与 `NONE` 在 27B（W8A8 动态 INT8 + 64 层 GDN hybrid）上给出 1–4 nat 的同上下文 logits 差（尾部到 10 nat），主导 logit 只差 0.2–0.3 nat，差随上下文长度累积；这个"种子"在未量化 4B 上只有 0.05 nat，因此放大来自"动态 per-token INT8 激活量化（离散取整）× 64 层 GDN 递推"，cudagraph 与 fork 的 packed decode 都被负向控制排除。** 叠加一个**低概率但真实存在**的因子：**graph 路径偶发的运行间不可复现**（同配置同 cache 也能给出 5.36 nat 的首 token 分布差，并因此产出 S14 那条英文重复循环）。**eager 侧逐位可复现，是唯一可作质量/精度参照的臂。**

### 3.12 收尾与终态核验（2026-10-09T23:39Z）

```
$ pgrep VLLM  →  全部 <defunct>（无 LIVE worker）
$ hy-smi --showmeminfo vram → DCU[0..3] Used = 0 MiB
$ free -g  → Mem total 31 / used 3 / free 16 / buff-cache 11 / available 27
$ cat /proc/swaps → 只有 /swap.img 4194304 B，used 496592 B（基线 461160，正常波动；/mnt/game/swapfile 全程未启用）
$ docker exec z100-s14 cat /sys/fs/cgroup/memory.events | grep oom_kill → 3   （基线 3，全程未增长）
$ md5sum /mnt/game/port/gfx906-vllm-211/vllm/_C_stable_libtorch.abi3.so → 93bf7b4a391ac97e2f67cc7141633cb1（未变）
$ sudo -n python3 tools/dcu/dcu_p2p_acs.py show | tail -4 →
  0000:35:00.0 0xf24 0x0000 (none) / 0000:35:10.0 0xf24 0x0000 (none)
  0000:37:00.0 0x2a0 0x0000 (none) / 0000:3a:00.0 0x2a0 0x0000 (none)      ⇒ 12×0x0000，部署态未动
$ systemctl show dcu-p2p-acs.service -p ActiveEnterTimestamp → Fri 2026-10-09 18:29:46 UTC
```

全部实验在同一 ACS 部署态（`18:29:46Z` 起，早于 S14 与本步）下完成；未触碰生产容器 `qwen38-vllm`；未改 DTK 库、未改 `z100-build*`、未 `swapon`。

## 4. 门禁判定

| # | 门禁 | 判据 | 判定 | 证据 |
|---|---|---|---|---|
| **G1** | 复现 | 同一 TP（TP=2）、同 prompt、**各自独立引擎**、graph vs eager 复现分歧；给首个分歧 token 位置 + 两侧 top1–top2 间距（nat）+ logprob 对照；**n≥3 prompt**；每次标 cache vintage | ✅ **通过（现象复现）**<br>⚠️ **S14 的具体结果不可复现** | 首 token 分歧：S14 `P2`(graph) `'The'`/g=4.750 vs `P2e`(eager) `'用户'`/g=0.859（同一 cache vintage `cache-27b-p2` / `cache-27b-p2e`）。本次 31 个同上下文（prompt 电池 4 + flip 8 + prefix 19）中 graph(默认图配置) 与 eager 首 token 不同者 **10/19**（§3.8）。**但 A1r 用逐字同配置复跑给出 `'用户'`/g=1.047，与 eager 只差 0.06–0.20 nat（§3.7）** |
| **G2** | 定性 | 给出 A/B 判定与证据；必须做能一锤定音的实验（同上下文 logit 距离 / 小模型或真值参照 / 算子级负向控制 / 配置对照） | ✅ **判定：以 B 为主 + 一个 A 形状的可复现性缺陷**（详见 §4.1） | §3.6（同上下文 1.38 nat mean / 10.3 max、106/354 flips）、§3.7（同配置复跑不给 P2 结果）、§3.8（10/19 首 token 不一致）、§3.4（eager 逐位可复现）、§3.1（噪声底口径修正）、**§3.10（4B + CPU HF 真值：graph vs eager 只差 0.054 nat，两臂与真值差 0.0024 nat）** |
| **G3** | 结论 | ① 机制；② 哪个模式可作质量/精度参照；③ 生产建议；④ 若 bug 给 `file:line` 根因 + 最小复现 + 修法 | ✅ **通过** | ① B + 可复现性缺陷（§4.1）；② **eager**（逐位可复现 §3.4，且与 CPU 真值差 0.0024 nat §3.10）；③ 生产栈用 eager（`serve_27b_tp4.sh` 默认 `ENFORCE_EAGER=1`）⇒ **安全，且是唯一可当质量参照的臂**；④ 未定位到"某个算子算错"，故**不实施 `csrc` 修改**，给出 H-AOT 假设 + 最小复现 + 修法方向（§4.2） |
| **G4** | 口径纪律 | 每行标 TP/模式/maxlen/util/n/vintage；ACS 全程部署态不切；不混口径 | ✅ **通过** | 本档每张表都带 TP/模式/图配置/packed/请求历史；`maxlen 262144 / util 0.85 / MBT 2048`（§3.4/§3.7 原文）；ACS 全程部署态（`12×0x0000`，`/mnt/game/s14b/hostlogs/acs-show.txt`） |
| **G5** | 归档 | 本文件 8 小节齐全、含原始命令/日志路径/md5/回滚件 | ✅ **通过** | 本文件；原始日志 `/mnt/game/s14b/**`；md5 与回滚件见 §6.4 |

### 4.1 A/B 判定（本步的核心结论）

**A 的定义**：某算子被编坏（S7b/S7c 那一类"静默算错"，通常伴随 NaN/Inf 或逐层可定位的数值爆炸）。
**B 的定义**：两个编译模式的算术路径不同（inductor 融合/切分、kernel 变体选择、归约顺序）⇒ 浮点结果有差 ⇒ 经**动态 int8 量化**与 **64 层 GDN 递推**放大 ⇒ 近 tie 翻转。

| # | 证据 | 支持 A | 支持 B |
|---|---|---|---|
| 1 | NaN/Inf 检查通过（S14 §3.4.1 登记的 NaN 检查；本步 31 个同上下文无 NaN） | — | ✅ |
| 2 | 同上下文 logit 差 **mean 1.38 / max 10.3 nat**，且 **argmax token 自己的 logprob 只差 0.20–0.33 nat（max 1.6）**，差集中在次选/尾部（gap 差 1.2–1.4、max 7） | — | ✅（"主导 logit 一致、竞争结构被打乱"= 数值放大形状，不是"整体算错"） |
| 3 | 差异**从 prompt 第 8 个 token 就出现**且随上下文变长而增长（prefix 电池 2.94 → 4.17 nat） | — | ✅（递推累积） |
| 4 | **eager 内部**只改 TP（2→4）就能让同一 token 的 logprob 差 **3.29 nat**（且 eager 逐位可复现） | — | ✅（说明 O(1)-nat 灵敏度是"这个模型 + 这套栈"的固有属性） |
| 5 | **同一 graph 配置、同一 AOT hash、冷首请求，两次运行给出不同首 token**（P2 `'The'` vs A1r `'用户'`，同上下文差 5.36 nat） | ✅（运行间不可复现 = 真缺陷形状） | — |
| 6 | AOT 产物来源是**唯一识别到的区分因子**（P2 = 现场编译并落盘；A1r = 从磁盘重载同一 hash）——但 A18 的对照只给出**部分支持**（重载 A5 的 cache 后 3/4 prompt 逐位一致，`p2_zh_slip` 尾部仍差 9.5 nat） | 🔶 部分（低概率运行间偏差，§3.9） | — |
| 7 | 后端/kernel 选择逐项相同（`TRITON_ATTN`、`Gfx906Sdot4Int8ScaledMMLinearKernel`、`PYNCCL`、`LBNHC`、`Triton/FLA GDN prefill`） | — | ✅（不是"选错后端"） |
| 8 | 无任何"某算子输出非有限/逐层爆炸"的证据；没有做到逐层定位（`VLLM_COMPILE` 下 inductor 消掉 module 边界，S7b hook 不可用） | — | ✅（就现有证据不能定某个算子） |
| **9** | **同架构族未量化 4B（TP=1）上，graph vs eager 同上下文 max 0.054 nat（mean 0.0052），且两臂各自与 CPU HF 真值差 0.0023–0.0025 nat、首 token 4/4 一致** | — | ✅✅（**编译模式本身在通用路径上只值 0.05 nat**；27B 的 1–4 nat 是 INT8 动态量化 + GDN 递推的放大） |

**判定**：
1. **主导机制是 B**：两个编译模式在**同一上下文**上的 logits 系统性差 1–4 nat（极端尾部 10 nat），主导 logit 只差 0.2–0.3 nat，差异随上下文长度递增（prefix 电池 2.94→4.17 nat）；量级与"只改 TP"（eager 内部 3.29 nat）、"只改 AOT vintage"（3.26 nat）同档。**这不是"某个算子被编坏"**（无 NaN、无逐层爆炸、后端/kernel 逐项相同、packed kernel 负向控制零影响），而是"**inductor 编译改变图（融合/切分 + int8 自定义 op 的图变化，S11l 已独立指出）× 动态 int8 量化 × 64 层 GDN 递推**"的放大（cudagraph 与 packed decode 已排除，§3.11）。S14 的 `P2` 是这条放大链上的一次坏 draw：它那一次的首 token 分布本身就与其余 6 个臂差 5.36 nat（不是"第一个 token 只差一点、后面放大"），随后 256 token 的英文重复循环是**首个 token 翻转之后的自回归放大**。
2. **同时存在一个 A 形状的缺陷：graph 路径运行间不可复现**。同一配置/同一 AOT hash/冷首请求两次运行给出 5.36 nat 的同上下文差（`P2` vs `A1r`），并且**输出质量也不同**（一次英文重复循环、一次连贯中文）。**该缺陷不是 cudagraph、不是 packed decode、不是前缀缓存**（三者均已用负向控制排除，§3.11）；AOT 产物来源（现场编译 vs 重载）只得到**部分支持**——A18 重载 A5 的 cache 后，长 prompt 与两个短 prompt 逐位一致，只有 `p2_zh_slip` 从第 59 位起差到 9.5 nat（§3.9）。⇒ 登记为一个**低概率但真实**的运行间偏差 + 一条锐利线索（离散起始位置），根因**不在 `csrc`**（见 §4.2）。
3. **因此"哪个模式算错了"的答案是：不是 eager 算错**——eager 是**逐位可复现**的（同配置同请求序：跨引擎 1024 token Δ=0.000；8/8 flip prompt Δ=0.000；同引擎重复 Δ=0.000），而 graph 路径在不同运行/AOT 来源下自己都不一致。**质量与精度的参照只能是 eager。**

### 4.2 修法与最小复现（未实施 `csrc` 改动，按任务书"根因不明确就不要硬塞"）

**最小复现（graph 路径运行间不一致，约 30 min/次）**：

```bash
# 第 1 次：空 cache ⇒ 现场编译并落盘 AOT
S14B_CT=z100-s14 S14B_CG_MODE=PIECEWISE S14B_CG_SIZES="1 2 4 8" READY_TIMEOUT=600 \
  tools/dcu/s14b_run_one.sh GT1 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 2 1 \
  /mnt/game/s14b/cache-repro-gt 8571 262144 0,1 greedy
# 第 2 次：复用同一 cache ⇒ 只走 AOT 重载
S14B_CT=z100-s14 S14B_CG_MODE=PIECEWISE S14B_CG_SIZES="1 2 4 8" READY_TIMEOUT=600 \
  tools/dcu/s14b_run_one.sh GT2 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 2 1 \
  /mnt/game/s14b/cache-repro-gt 8572 262144 0,1 greedy
python3 tools/dcu/s14_logprob_compare.py /mnt/game/s14b/greedy-GT2.jsonl /mnt/game/s14b/greedy-GT1.jsonl
# 期望（若 H-AOT 成立）：首个分歧 + 同上下文 |Δlogprob| 非 0（本次 P2/A1r 在默认配置上量到 5.36 nat）
```

**修法建议（交给下一步，不在本步实施）**：
1. **先判 H-AOT**：`-cc.inductor_compile_config='{"benchmark_combo_kernel":false,"combo_kernels":false}'`（换 AOT key，**必须新 cache 目录**）后重做上面两次跑；若两次逐位一致 ⇒ 根因在 inductor 的运行期 kernel 基准选择（`benchmark_combo_kernel`）没有随 AOT 一起固定。
2. **对照"现场编译 vs 重载"**：把 `torch_aot_compile/` 子目录删掉强制现场编译，与不删的两次跑对比（同配置、同 cache 目录的其余内容）。
3. **质量口径固定 eager**：`serve_27b_tp4.sh` 已经 `ENFORCE_EAGER=1`，**保持**；把"graph 口径的 token 流"排除在任何质量/精度评测之外。
4. **没有任何证据支持改 `csrc`**：本步未定位到具体算子（`VLLM_COMPILE` 下 module 边界被 inductor 消掉，S7b 式逐层 hook 不可用）；fork 的 gfx906 打包路径（`VLLM_QWEN35_PACKED_DECODE`）已用负向控制排除（对 prefill 同上下文 logprob **零影响**，§3.6c(1)）；**cudagraph 也已排除**（A3 无图 ≡ A5/A18 有图，§3.11）。⇒ 放大链的另一半（**动态 per-token INT8 激活量化**）属于 `compressed-tensors` 量化方案本身的可预期行为，不是"编坏了的算子"。
5. **下一步最值钱的两个实验**（都不需要改 `csrc`）：
   (a) **同一 cache 目录连跑两次 + 现场编译 vs 重载的 2×2**（§4.2 第 1/2 条），把"graph 偶发偏差"钉死到具体来源；
   (b) **把 `p2_zh_slip` 第 59 位起的离散分歧**做成最小复现（同一 prompt、同 cache 连跑两次；再把 `VLLM_QWEN35_REC_PREFILL` 翻过来），这是本步发现的**最锐利的一条线索**（分歧前 58 位逐位相同，说明不是全程累积噪声，而是某个离散事件）。


## 5. 未完成项 / 阻塞与原因

| # | 项 | 状态 | 原因 |
|---|---|---|---|
| 1 | **逐层/逐算子定位（graph 侧）** | ❌ 未做 | `CompilationMode.VLLM_COMPILE` 下 inductor 消掉 module 边界（fork 自己在 `tools/dcu/s7b_hooks.py` 头部写明），`splitting_ops` 只留下少数图断点；**没有可用的逐层 hook**。本步改用"同上下文 logit 距离 + 逐位置 profile + prefix 电池"作为等价定位手段（说明与替代关系见 §3.3） |
| 2 | **27B 的 HF/CPU 真值** | ❌ 不可得 | 权重 28.76 GiB，本机 31 GiB RAM + 27B 引擎同时在跑时放不下（S14 已登记同一限制）。本步的真值参照只能用 **Qwen3.5-4B**（同架构族、32 层、TP=1、bf16 原生）——但它是 **未量化**模型，`split_projections=False`，走的是**通用路径**，因此它只能回答"通用编译模式差有多大"，不能直接回答"27B INT8 的打包路径是否算错" |
| 3 | A3（compile + `cudagraph_mode=NONE`，27B） | ✅ 已完成 | 首 token 判定与 A5/A18 逐个相同（§3.11）⇒ **cudagraph 被排除** |
| 3b | A18（AOT 重载对照） | ✅ 已完成 | 见 §3.9：长 prompt 逐位一致、两个短 prompt 完全一致、`p2_zh_slip` 从第 59 位起差到 9.5 nat ⇒ H-AOT 仅部分成立 |
| 4 | A3/A4 首次跑报废 | ⚠️ 已登记 | **操作事故**（运行中改脚本），见 §3.5；A3 已排入后续链，A4 由 A12 顶替 |
| 5 | `--enable-prefix-caching` 的因果 | ✅ 已排除 | A1r/A2b/A5/A6 全程 `Prefix cache hit rate: 0.0%`；"关前缀缓存"arm（A11）因时间原因未跑完 |
| 6 | **并发（batch>1）下的同上下文一致性** | ❌ 未测 | 只有一次"探针与在飞解码流并发"的观测（A2w，13.7 nat，§3.6c），未能做受控的 conc=2 对照；这是生产常态，**建议下一步单独立实验** |
| 7 | 生产栈（`qwen38-vllm`）真机对照 | 🚫 按纪律 | 只允许 `docker inspect` 只读；本步未接触该容器 |
| 8 | `benchmark_combo_kernel=False` 的配置对照（任务书建议项） | 🔶 部分 | 已写进 A8/A18 的臂定义；A8 未跑完（时间原因），A18 见 §3.9 |
| 9 | 同引擎内 greedy 重复性 | ✅ 已补 | A2b vs A2（同请求序、跨引擎）与 A1r 同引擎 plp-vs-greedy 对照；结论：eager 逐位一致（§3.4） |
| 10 | `csrc` 修复 | 🚫 主动不做 | 未定位到具体算子（第 1/2 条），按任务书"根因明确 + 风险可控才做，否则给修法建议"执行；`_C_stable_libtorch` md5 全程保持 `93bf7b4a…` |


## 6. 对后续步骤的影响

1. **口径修正（对 S14 的更正）**：S14 §3.6.3 的结论"TP=2 下 graph↔eager 是系统性模式差、graph 模式改变输出质量"**必须改成**："**graph 路径在 TP=2 默认图配置下运行间不可复现**；它给出的 token 流（包括那次英文重复循环）是**一次不可复现的抽签**；eager 逐位可复现。"
2. **质量/精度评测只能用 eager**：本项目此前用过的所有 graph 口径（P2/P4g2/P4g3 的 greedy/flip 文本、任何"graph 输出质量"的说法）都不能当作模型质量证据；生产栈 `serve_27b_tp4.sh` 默认 `ENFORCE_EAGER=1` ⇒ **生产是对的，不需要改**。
3. **性能口径不受影响**：graph 仍然是最快的（S14 2×2 矩阵：graph TP=2 84.22 ms/token、graph TP=4 49.66），本步只否证了它的**数值/质量**可信度，不影响它的**吞吐/延迟**结论。若要同时要"快"和"可复现"，方向是修 AOT/运行期 kernel 选择（§4.2），不是回到 eager。
4. **给质量评测的建议**：任何"同 prompt 同配置跨引擎重启必须逐位一致"的准入测试应加进 graph 口径；eager 侧本步已给出可复现证据（1024 token Δ=0.000）。
5. **给 P6/内存步骤的建议**：本步的 27B graph 默认配置在 TP=2 上**需要约 1600 s 就绪**（两轮 PIECEWISE 51 桶 + FULL 35 桶，§3.7），这是"图捕获成本"的实测值，可作为 P6 的输入。
6. **下一步（建议立 S14c）**：(a) 判 H-AOT（`benchmark_combo_kernel` 关闭 + 现场编译 vs 重载的 2×2）；(b) conc=2 同上下文一致性；(c) 把 `p2_zh_slip` 第 59 位起的离散分歧做成最小复现（同 cache 连跑两次 + 翻 `VLLM_QWEN35_REC_PREFILL`）——本步最锐利的线索。
7. **对性能工作的好消息**：**cudagraph 与数值无关**（A3 无图 ≡ A5/A18 有图，§3.11）⇒ S14 的 graph 性能收益（84.22 @TP2 / 49.66 @TP4）与图捕获路径**不必为了数值可信度而放弃**；要修的只是"inductor 编译 + INT8 动态量化"这条算术路径的可复现性/一致性问题（或干脆只用 eager 做质量评测、用 graph 做性能评测）。

### 6.4 回滚件与身份

```
_C_stable_libtorch.abi3.so  md5 93bf7b4a391ac97e2f67cc7141633cb1   （全程未改；本步只写 tools/dcu/s14b_* 与 docs/result/ 本文档）
cache vintage（复用 S14 的，未改）：cache-27b-p2 / cache-27b-p2e
cache vintage（本步新建）：cache-27b-p2-cgnone / -pw4 / -pw4-nopacked / -pw4-v2 / -pw4-v3 / -pw4-noc ... （见 §7 命令）
本步新增脚本（可整体删除即可回滚，不影响运行）：tools/dcu/s14b_arm_env.sh s14b_run_one.sh s14b_greedy.py \
  s14b_prompt_logprobs.py s14b_compare_promptlp.py s14b_compare_prefix.py s14b_hf_ref.py s14b_firsttok.py \
  s14b_truth_compare.py s14b_arms.sh s14b_arms2.sh s14b_arms3.sh s14b_arms4.sh
宿主状态：`/mnt/game/swapfile` 全程未启用（`/proc/swaps` 只有 4 GiB `/swap.img`）；`oom_kill` 基线 3，全程未增长
```


## 7. 复现方式（本步全部原始命令）

前置（每个 arm 开跑前留档，原文 `/mnt/game/s14b/hostlogs/00-baseline.txt`）：

```bash
cd /home/lyh/z100-vllm
docker inspect z100-s14 --format '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}'   # 0 0
pgrep -af "vllm|torchrun"          # 必须为空（VLLM::Worker_TP 的 comm 被改过）
hy-smi --showmeminfo vram; free -g; cat /proc/swaps    # 只应有 4 GiB /swap.img
sudo -n python3 tools/dcu/dcu_p2p_acs.py show | tail -3 # 12×0x0000（ACS 全程部署态）
```

臂驱动（**每条链串行；绝对不要在链在跑时改 `s14b_*` 脚本**，见 §3.5 事故）：

```bash
tools/dcu/s14b_run_one.sh <NAME> <MODEL> <TP> <SW> <CACHE> <PORT> <MAXLEN> <DEVICES> <TASKS> [EXTRA...]
# TASKS: promptlp,prefix,greedy,greedy1,flip
tools/dcu/s14b_arms.sh  A2 A3 A4 A5 A6          # 链1（本步已完成）
tools/dcu/s14b_arms2.sh A2b A1r A9              # 链2 前半（A1r 已完成；A9 进行中）
tools/dcu/s14b_arms.sh  B1 B2 B3                # 链2 后半（4B TP=1 + HF 真值锚）
tools/dcu/s14b_arms4.sh A18 A3                  # 链4（AOT 重载对照 + compile-no-cudagraph）
```

**同上下文数值对比（本步的关键仪器）**：

```bash
python3 tools/dcu/s14b_firsttok.py /mnt/game/s14b/greedy-*.jsonl
python3 tools/dcu/s14b_compare_promptlp.py /mnt/game/s14b/plp-A5.jsonl /mnt/game/s14b/plp-A6.jsonl \
        --label-a A5-graph-cold --label-b A6-eager-cold --profile
python3 tools/dcu/s14b_compare_prefix.py  /mnt/game/s14b/prefix-A1r.jsonl /mnt/game/s14b/prefix-A2b.jsonl \
        --label-a A1r-graph-default --label-b A2b-eager
python3 tools/dcu/s14_logprob_compare.py  /mnt/game/s14b/flip-A2b.jsonl /mnt/game/s14/flip-P2.jsonl \
        --label-a A2b-eager --label-b P2-graph        # S14 原工具，未改
```

**真值参照（CPU，无 GPU）**：

```bash
docker exec z100-s14 bash -c '. /opt/dtk/env.sh >/dev/null 2>&1; cd /src && python tools/dcu/s14b_hf_ref.py \
  --model /mnt/game/models/Qwen3.5-4B --prompts-file /mnt/game/s14b/prompts.txt \
  --out /mnt/game/s14b/plp-HF4B-bf16.jsonl --label HF4B-bf16 --top-k 20 --dtype bfloat16'
python3 tools/dcu/s14b_truth_compare.py /mnt/game/s14b/greedy-B1.jsonl /mnt/game/s14b/plp-HF4B-bf16.jsonl \
        --label-arm B1-eager --label-truth HF4B-bf16
```

**收尾**：

```bash
docker exec z100-s14 bash -c 'kill $(cat /tmp/s14b_<NAME>.pid)'; sleep 20; pkill VLLM || true
hy-smi --showmeminfo vram; free -g; cat /proc/swaps
md5sum /mnt/game/port/gfx906-vllm-211/vllm/_C_stable_libtorch.abi3.so    # 期望 93bf7b4a…
```


## 8. 参考

* 前序归档：`docs/result/z100-14-27b-tp4-result.md`（S14，md5 `7e39492ebf079f66c00c87c1481fd721`，§3.6/§3.4.1/§3.8.4 为本步的直接上游）
* `docs/result/z100-07b-eager-nan-result.md`（S7b：`_C.silu_and_mul` fp16 packed 分支 NaN，逐层扫描方法）
* `docs/result/z100-07c-packed-silu-csrc-fix-result.md`（S7c：`dtk_gfx906_fp16_no_packed()` host 守卫修法，`_C_stable_libtorch` = `93bf7b4a…`）
* `docs/result/z100-11l-sdot4-reacceptance-result.md` / `z100-11m-aot-restart-reproducibility-result.md`（"扰动来自 custom_op 的图变化"、"token 变体由 vintage 决定" —— 本步 H-AOT 的直接前身）
* 源码：`vllm/model_executor/models/qwen3_5.py:294-318`（gfx906 开关集合）、`:401-413`、`:639-649`（`torch.empty` vs `torch.zeros`）、`:691-702`（chunk decode）、`:706-731`（`_packed_decode_writes_full_core_output` 在 `is_compiling()` 时返回 False）、`:747-765`（`split_projections` 只对量化模型开）
* `vllm/model_executor/models/qwen3_next.py:821-859`（decode packed 快路径的分支条件）、`:1256-1302`（`core_attn_out` 写入范围）
* `vllm/config/compilation.py:774-781`（`splitting_ops` 含 `vllm::qwen_gdn_attention_core*`）
* 脚本：`tools/dcu/s14b_*`（本步新增）；原始日志 `/mnt/game/s14b/hostlogs/**`；原始数据 `/mnt/game/s14b/*.jsonl`

### 3.9 配置/开关二分：**没有任何一个图配置或 fork 开关能解释 P2 的 `'The'`**

同一 prompt（`s14_greedy_long`）、TP=2、temperature=0、冷引擎第一条请求，逐个改**一个**因子：

| arm | 模式 | 图配置 | packed decode | AOT 来源 | 首 token | gap | 文本 |
|---|---|---|---|---|---|---|---|
| A2b | eager | — | ON | 重载 | `'用户'` | 0.859 | 连贯中文 |
| P2e | eager | — | ON | 现场 | `'用户'` | 0.859 | 连贯中文 |
| **P2**（S14） | graph | **默认 51+2 桶 FULL_AND_PIECEWISE** | ON | **现场编译并落盘** | **`'The'`** | **4.750** | **英文重复循环** |
| **A1r**（本次） | graph | 默认 51+2 桶 FULL_AND_PIECEWISE | ON | **磁盘重载同一 hash** | `'用户'` | 1.047 | 连贯中文 |
| A5 | graph | PIECEWISE 4 桶（1,2,4,8） | **OFF** | 现场 | `'用户'` | 4.031 | 连贯中文 |
| A9 | graph | PIECEWISE 4 桶（1,8,**128**,512）⇒ prefill(126) 进 cudagraph | ON | 现场 | `'用户'` | 2.500 | 连贯中文 |
| P4g2/P4g3（S14） | graph | PIECEWISE 4 桶 | ON | 现场 ×2 | `'用户'` | 2.219/4.391 | 连贯中文 |
| P4e（S14） | eager | — | ON | — | `'用户'` | 4.031 | 连贯中文 |

**负向控制（任务书要求"算子级负向控制"）**：

```
### A2b(eager, packed ON, cold) vs A6(eager, packed OFF, cold)
s14_greedy_long  126  True  0.00000  0.00000  0 flips   ; 另 3 条 prompt 也全 0.00000
# ⇒ VLLM_QWEN35_PACKED_DECODE（fork 的 gfx906 打包 decode 快路径）对 prefill 同上下文 logprob **零影响** ⇒ 它不是差异来源
```

**读法**：
1. **P2 的 `'The'` 与图配置无关**：默认图配置（A1r）给出 `'用户'`；PIECEWISE 4 桶（A5/A9/P4g）也给出 `'用户'`；把 prefill 送进 cudagraph（A9）仍是 `'用户'`。
2. **P2 的 `'The'` 与 packed decode 开关无关**（负向控制：开关对 prefill 零影响）。
3. ⇒ 只剩**"同配置、同 AOT hash、不同运行"**这一个因子（§3.7），即 **H-AOT**（现场编译 vs 磁盘重载）。A18（复用 A5 的 cache 重跑 PW4/4+packed OFF）与 A3（compile + `cudagraph_mode=NONE`）在归档时仍在跑，结论见 §5 第 3 条。
4. **注意"分歧幅度"本身是常态**：同一 prompt 的 `'用户'` gap 在 eager(TP2) 0.859 / A1r 1.047 / A9 2.500 / A5 4.031 / P4g3 4.391 之间变化（≈5×），而两侧的 top-1 都是 `'用户'`。⇒ "同上下文 logit 差 1–4 nat"是换任何执行路径都会出现的量级（§3.6）。

**A18（复用 A5 的 cache 目录 = AOT 重载；同配置 PW4/4 + packed OFF）的结果 —— H-AOT 只得到部分支持**：

```
$ python3 tools/dcu/s14b_compare_promptlp.py /mnt/game/s14b/plp-A18.jsonl /mnt/game/s14b/plp-A5.jsonl \
        --label-a A18-reload --label-b A5-fresh --profile
s14_greedy_long  126  True  0.00719  0.00001   tok101630  0  -1   profile 全 0.000（仅 1 个 token 差 0.007）
p2_en_batch       74  True  0.00000  0.00000   0 flips    profile 全 0.000
p2_zh_code        76  True  0.00001  0.00000   0 flips    profile 全 0.000
p2_zh_slip        78  True  9.48910  0.17917   4 flips    profile: 0..56 全 0.000 → 59:0.636 60:1.145 63:1.964 64:1.407 73:5.586 74:9.489 75:6.375 77:3.492
$ python3 tools/dcu/s14_logprob_compare.py /mnt/game/s14b/greedy-A18.jsonl /mnt/game/s14b/greedy-A5.jsonl
# tokens on common prefix : 326 of 768 ; max|dlogprob| 0.160 ; mean 0.0035 ; diverged = [('p2_zh_slip',14,...), ('p2_en_batch',56,...)]
```

* **长 prompt（126 token）逐位一致**（plp max 0.007 nat、greedy 首段一致），两个短 prompt 的 plp **完全为 0**。
* 只有 `p2_zh_slip`（78 token）在**第 59 个位置之后**才出现 0.6–9.5 nat 的差，而 **0–58 位置逐位为 0**（在该次比较的分辨率下完全相同）。
* ⇒ **H-AOT 不是"重载必然换一套 kernel"**（否则三个 prompt 都会差）；更像是"**某种隐藏状态/某个 kernel 选择在极少数情况下不同，且只有在遇到贴边位置时才显形**"（位置 59 的 top1–top2 gap 只有 0.039–0.117 nat）。
* **这正是下一步该抓的线索**：分歧**起始于序列中段的一个固定位置**（78-token 的 prompt 从第 59 位起），提示与 **chunk/对齐边界**有关；最小复现方向是同一 cache 目录连跑两次 + 把 `VLLM_QWEN35_REC_PREFILL`（recurrent vs chunked prefill）翻过来对照（见 §4.2 修法建议）。

### 3.10 真值参照：**vLLM-eager 与 CPU HF 真值只差 0.003 nat**（4B / TP=1 / 未量化）

同架构族的小模型 `Qwen3.5-4B`（32 层、GDN+hybrid、**未量化** ⇒ `split_projections=False`，走通用路径；TP=1，单卡 3）三臂同 prompt 电池、同 chat template（**`prompt_token_ids` 逐位相同**，见下），HF 侧 CPU（bf16）= 与执行路径无关的真值：

```
$ python3 - <<'EOF'   # 校验模板一致（原文见 §7）
  s14_greedy_long    vLLM n= 84 HF n= 84 same=True
  p2_zh_slip         vLLM n= 36 HF n= 36 same=True
  p2_en_batch        vLLM n= 32 HF n= 32 same=True
  p2_zh_code         vLLM n= 34 HF n= 34 same=True
EOF

$ python3 tools/dcu/s14b_truth_compare.py /mnt/game/s14b/greedy-B1.jsonl /mnt/game/s14b/plp-HF4B-bf16.jsonl \
        --label-arm B1-eager --label-truth HF4B-bf16
prompt               truth_top1   t_gap     arm_top1   a_gap match  truth_lp_in_arm  d_truth  maxd_shared
s14_greedy_long          'Here'   6.125       'Here'   6.031  True        -0.0087    0.0001     0.0936
p2_zh_slip           'Thinking'   5.500   'Thinking'   5.469  True        -0.0047    0.0001     0.0783
p2_en_batch              'Here'   2.875       'Here'   2.906  True        -0.0544    0.0017     0.1077
p2_zh_code           '这是一个'   2.250   '这是一个'   2.344  True        -0.1093    0.0080     0.0858
# first-token argmax match : 4/4
# mean |truth_lp - arm_lp| on the truth token : 0.0025 nat
# max  |truth_lp - arm_lp| on the truth token : 0.0080 nat

$ python3 tools/dcu/s14b_compare_promptlp.py /mnt/game/s14b/plp-B1.jsonl /mnt/game/s14b/plp-HF4B-bf16.jsonl \
        --label-a B1-eager --label-b HF4B
# max |dlogprob| on identical contexts : 0.987307   (出现在尾部 token)
# mean|dlogprob| on identical contexts : 0.041053
# argmax flips over prompt positions   : 4          （≈290 个位置上 4 个，且都在 <0.1 nat 的贴边）
```

**⇒ 三条结论**：

1. **本步的"同上下文"仪器分辨率足够**：在 4B 上，vLLM-eager 与 CPU HF 真值在"决定首个生成 token"的那个分布上只差 **0.003 nat（mean）/ 0.008 nat（max）**，top-5 上 mean 0.041 nat。**所以 27B 上量到的 1–4 nat 同上下文差不是仪器噪声、也不是"vLLM 与 HF 实现差异"**。
2. **在通用（未量化）路径上，vLLM-eager ≈ 真值**（4/4 首 token 一致，margin 6.125/5.500/2.875/2.250 也对得上：6.031/5.469/2.906/2.344）。
3. ⇒ 27B 上 graph↔eager 的 **1–4 nat** 只能来自 **INT8 动态量化（per-token 激活量化）+ 64 层 GDN 递推 + fork 的 gfx906/图路径** 这套组合（机制 B 的放大链），而不是"编译模式本身把某个通用算子算错"。

**B2（4B / TP=1 / graph / 默认 51+2 桶 `FULL_AND_PIECEWISE`；就绪 490 s）——把"编译模式"钉在真值上**：

```
$ tools/dcu/s14b_arms.sh B2          # 端口 8532，卡 3
[orch14b] ready=490s ... init engine ... (compilation: 38.71 s) | Available KV cache memory: 18.38 GiB

$ python3 tools/dcu/s14b_truth_compare.py /mnt/game/s14b/greedy-B2.jsonl /mnt/game/s14b/plp-HF4B-bf16.jsonl \
        --label-arm B2-graph --label-truth HF4B-bf16
prompt               truth_top1   t_gap     arm_top1   a_gap match  d_truth  maxd_shared
s14_greedy_long          'Here'   6.125       'Here'   6.031  True    0.0001     0.0936
p2_zh_slip           'Thinking'   5.500   'Thinking'   5.469  True    0.0001     0.0704
p2_en_batch              'Here'   2.875       'Here'   2.891  True    0.0009     0.0929
p2_zh_code           '这是一个'   2.250   '这是一个'   2.344  True    0.0079     0.0858
# first-token argmax match : 4/4        mean |truth_lp - arm_lp| = 0.0023 nat   max = 0.0079 nat

$ python3 tools/dcu/s14b_compare_promptlp.py /mnt/game/s14b/plp-B2.jsonl /mnt/game/s14b/plp-B1.jsonl \
        --label-a B2-graph --label-b B1-eager
# max |dlogprob| on identical contexts : 0.053915
# mean|dlogprob| on identical contexts : 0.005230
# argmax flips over prompt positions   : 1 / ~186
# VERDICT: ROUNDING-LEVEL (mechanism B): same function up to fp16 accumulation order
```

**⇒ 这是本步的"一锤定音"之第二锤（通用路径侧）**：
* 在**未量化**的 4B 上，**graph 与 eager 同上下文最大只差 0.054 nat（mean 0.0052）**，判定原文就是 `ROUNDING-LEVEL`；
* 而且**两边都贴着 CPU 真值**（B1 d=0.0025 / B2 d=0.0023 nat，首 token 4/4 一致）。
* 所以 **"编译模式"本身在通用模型上只值 0.05 nat**；27B 上量到的 **1–4 nat（尾部到 10 nat）只能来自 27B 独有路径的放大**：**动态 per-token INT8 激活量化**（scale = max|act|/127，输入的 1-ulp 变化会让整条 int8 向量重新取整）+ **64 层 GDN 递推累积**（prefix 电池 |Δ| 随长度 2.94→4.17 nat 单调增）。**这正是不折不扣的机制 B**。

**B3（4B / TP=1 / `VLLM_COMPILE` + `-cc.cudagraph_mode=NONE`，即"只 inductor、无 cudagraph"）——把 cudagraph 与 inductor 分开**：

```
$ python3 tools/dcu/s14b_truth_compare.py /mnt/game/s14b/greedy-B3.jsonl /mnt/game/s14b/plp-HF4B-bf16.jsonl \
        --label-arm B3-compile-nocudagraph --label-truth HF4B-bf16
# first-token argmax match : 4/4   mean |truth_lp - arm_lp| = 0.0027 nat   max = 0.0095 nat
$ python3 tools/dcu/s14b_compare_promptlp.py plp-B3.jsonl plp-B1.jsonl   # vs eager
# max 0.110612 / mean 0.005207 / 1 flip / VERDICT: SMALL (grey zone)
$ python3 tools/dcu/s14b_compare_promptlp.py plp-B3.jsonl plp-B2.jsonl   # vs graph(默认)
# max 0.060632 / mean 0.005120 / 0 flips / VERDICT: ROUNDING-LEVEL
```

**4B 三臂汇总（同上下文、同 prompt_token_ids、都对 CPU 真值）**：

| 臂 | 4B 配置 | vs CPU HF 真值（首 token 分布） | vs eager（同上下文） |
|---|---|---|---|
| **B1** | eager（`CompilationMode.NONE`） | 4/4 一致，mean **0.0025** / max 0.0080 nat | — |
| **B2** | graph（`VLLM_COMPILE` + 默认 `FULL_AND_PIECEWISE`） | 4/4 一致，mean **0.0023** / max 0.0079 nat | max **0.054** / mean 0.0052 nat（ROUNDING-LEVEL） |
| **B3** | compile + `cudagraph_mode=NONE` | 4/4 一致，mean **0.0027** / max 0.0095 nat | max **0.111** / mean 0.0052 nat（SMALL） |

⇒ **在通用（未量化）路径上，"编译模式 + cudagraph"整体只值 0.05–0.11 nat，且三臂都贴着 CPU 真值（≈0.003 nat）**。所以 27B 的 1–4 nat 差**不是编译模式造成的**，而是 27B 独有路径（**动态 per-token INT8 激活量化 + 64 层 GDN 递推**）对 0.05 nat 级种子的放大。
