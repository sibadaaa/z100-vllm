# S18e 正确性门禁（K2）失败 + 性能/回归数据 — 执行结果

> 归档时间：2026-10-12
> 执行者：Lead（真机臂 + 离线对照；臂电池子代理因卡位冲突只完成 1 条，已单独登记）
> 对应设计文档：docs/design/z100-18-mtp-optimization.md（K2 门禁）、[S18d](z100-18d-mtp-gemm-dispatch-e2e-result.md)（K4 通过）
> 前一档：[z100-18d](z100-18d-mtp-gemm-dispatch-e2e-result.md)
> 状态：⛔ **K2 不通过 —— MTP 的 greedy 输出与非投机臂有实质性差异**；性能门禁（K4）通过

## 1. 目标与验收标准

- **K2**：eager、固定批组合下，greedy 与同 flags 非投机臂**逐 token 一致**；若不一致，差异必须落在本项目噪声底（0.1–0.6 nat）内并给出归因；
- **回归**：TTFT / conc8 劣化 ≤5%；
- 顺带复核 S18d 的性能结论。

## 2. 改动

本步 **不改 vllm 源码**，只加测量通道：

| 文件 | 改动 |
|---|---|
| `tools/dcu/s18_one_arm.sh` | 新增 `S18_FLIP=1`：引擎 ready 后先跑 `s11l_flip_stats.py`（固定 8 条中英 prompt、greedy、64 token、top-5 logprob），产出 `flip-<label>.jsonl` |
| `tools/dcu/s18_mtp_arm.sh` | 新增 `S18_EXTRA`：在固定 flags 之后追加任意 CLI（用于 A/B `--mamba-cache-mode` / `--no-enable-prefix-caching`） |

## 3. 原始证据

### 3.1 四组逐 token 对照（FP8 / TP=4 / eager / maxlen 8192）

| 对照 | 翻转 / 位置数 | 首次分叉位置 | 首次分叉的间距（前臂侧） |
|---|---|---|---|
| ctl(k=0) vs **k=1 + GEMM** | **280 / 469（0.597）** | 1–16 | `en_fr` **12.11 nat**、`en_sea` **8.44**、`zh_sky` 0.27、`zh_ml` 0.47 |
| ctl(k=0) vs **k=2 + GEMV**（旧派发） | **378 / 468（0.808）** | 1–19 | `en_sea` **8.44**、`zh_sky` 1.34、`zh_ml` 0.44、`en_fr` 0.33 |
| ctl(k=0) vs **k=2 + GEMM** | **360 / 462（0.779）** | 1–19 | `en_sea` **8.44**、`zh_sky` 1.34、`zh_ml` 0.44、`en_fr` 0.33 |
| **k=2 + GEMV vs k=2 + GEMM** | 244 / 462（0.528） | **4–54（更晚）** | 全部 ≤ **0.44 nat**（0.02–0.44） |

**判读**：

1. **`k=1`（最简投机情形）也分叉**，且 `en_fr` 的首次分叉是**控制臂间距 12.11 nat 的自信选择**被换成另一 token ⇒ 不能用"数值噪声"解释；
2. **两条 MTP 臂分别与控制臂的分叉模式几乎相同**（同一批首次分叉位置/候选），而**两条 MTP 臂彼此之间的差异又晚又小**（首次分叉 4–54、全部 ≤0.44 nat）⇒ **分叉来自 MTP verify 路径本身，与 S18c 的 GEMM 派发改动无关**（我们的改动只贡献 near-tie 级噪声）；
3. 不是普遍性崩溃：`en_math` 在 k=1 下 **64/64 完全一致**；大多数 prompt 前 1–19 个 token 一致。

### 3.2 分叉点的分布形态（不是"采样选错"，是"分布不同"）

```
en_sea  div@2  ctl: ' to' lp=-0.000 gap=8.437   top3=[[' to',-0.0],[' answer',-8.438],[' respond',-9.703]]
               mtp: ' answer' lp=-0.570 gap=0.844 top3=[[' answer',-0.57],[' respond',-1.414],[' user',-2.32]]
```
⇒ 两条臂的 top-2 候选**相同但排序互换、间距不同**（`en_sea` 里 `' to'` 在 MTP 臂中甚至掉出 top-3）；`zh_ml`/`zh_sky` 同样是"同候选、换序"。

### 3.3 文本（两臂输出都连贯，但用词不同）

```
en_fr  "What is the capital of France?"
  ctl(k=0): "We need answer simple question. Need final concise.</think>\n\nThe capital of France is Paris."
  k1+GEMM : "We should answer simply. User asks: \"What is the capital of France?\" Need final: Paris.</think>\n\nParis"
en_sea "Write a short poem about the sea."
  ctl     : "We need to respond to user: ... Simple. Just produce a short poem. Ensure final only poem maybe. ... The sea breathes in, ..."
  k1      : "We need answer user's request: ... Need produce final poem. Keep short. No need complex. ... The sea breathes in, ..."
```
⇒ 分叉发生在**思维前缀的措辞**上，最终答案都正确；不是乱码/退化，但**token 流确实被改变**（从第 1–2 个生成 token 起）。

### 3.4 性能与回归（同一批臂）

| 指标（FP8/TP=4/eager/maxlen 8192） | ctl（k=0） | k=1 + GEMM | k=2 + GEMM |
|---|---|---|---|
| decode Median TPOT | 123.30 ms | 99.34 ms（**1.24×**） | 83.92 ms（**1.47×**） |
| decode 吞吐 | 7.70 tok/s | 9.39 | **10.89–10.94 tok/s** |
| greedy MAL | — | 1.781 | 2.246 |
| 8K prefill TTFT（in=8100） | 32238.67 ms | — | 32748.85 ms（+1.6%） |
| conc8 输出吞吐 | **（未取全，见 §5）** | — | 28.56 tok/s（Median TPOT 147.57 ms） |

## 4. 门禁判定

| 门禁 | 判定 | 依据 |
|---|---|---|
| K1 功能 | ✅ | 全部臂 ready、出词连贯 |
| K3 接受率 | ✅ | k=2 MAL 2.246、k=1 1.781 |
| **K4 性能 ≥1.3×** | ✅ | k=1 1.24×、**k=2 1.47×** |
| **K2 逐 token 一致 / 差异 ≤ 噪声底** | ⛔ **不通过** | §3.1：k=1/k=2 均出现 **8.4–12.1 nat** 的自信翻转；两条 MTP 臂彼此仅 near-tie 差异 ⇒ 与派发改动无关 |
| 回归 ≤5% | 🔶 部分 | 8K TTFT +1.6%（在噪声内）；conc8 对照缺（§5） |

**结论（按方案的止损纪律）**：MTP 在本平台**性能可用、正确性不可用**。在 K2 解决前，**不得**把 MTP 用于任何依赖 token 流的用途（质量评测/生产输出），与 S14d/e 对 graph 的裁定同类；性能结论不受影响（口径 = 延迟/吞吐）。

## 5. 未完成项 / 事故登记

1. **控制臂 conc8 未取全**：我在同一容器里先后启动了两条链，后启动的 `kill_engine` 把前一臂的 conc8 bench 打断（**自己的调度失误，已登记**）。缺 ctl 的 conc8 数字 ⇒ 回归门禁只完成 TTFT 一项；
2. **首轮 k=0 控制臂启动失败**（`AttributeError: vllm.envs has no attribute VLLM_DCU_INT8_GEMV_MAX_M`）：我把 kernel 模块切到 `envs.*` 与在 `envs.py` 里声明之间有一个编辑窗口，worker 恰好在该窗口启动。**已修**（声明补齐后默认值 8、显式设为 1 生效均已验证）；这条也说明该开关**失败是响亮的**（import 期 AttributeError，不会静默退化）；
3. **臂电池子代理只完成 1/4**（FP8 TP=2 k=1，TPOT 192.50 ms、MAL 1.748，已存 `/mnt/game/s18/batteryB-evidence/`）：它报告 `z100-s18`/卡 0-3 被我自己的链连续占用、无独占窗口而停手。教训：**同一容器不能并行两条臂链**（会互相 pkill 引擎），后续所有臂严格串行或分容器；
4. K2 的**根因未定位**：已排除派发改动（§3.1 第 3 条）；4 条定位臂（前缀缓存 / `mamba-cache-mode none` 解耦）已挂后台运行，结果未出。

## 6. 对后续步骤的影响

1. **S18-5（上线）暂停**：不得把 MTP 写进生产配方，直到 K2 通过；
2. 定位方向（按嫌疑排序）：
   - **mamba align × spec 的状态搬运**（`mamba_utils.py:1132-1180` 的 `postprocess_mamba_align_gpu`；本模型必须 `align`）——正在用 `--mamba-cache-mode none` 解耦；
   - **前缀缓存 × spec 的块布局**（`use_eagle_block_drop`、`kv_cache_utils.py:2195`）——正在用 `--no-enable-prefix-caching` 解耦；
   - V2 speculator 的 `_calc_spec_decode_metadata` / `logits_indices`（verify 的 logits 取行是否与 sampled 位置一致）；
   - RoPE/position 在 verify 批次上的取值；
   - fork 的 `1e23b7af1`（gfx906 MTP 多 token attention 路径）与 `da5f8ce75`（Qwen3.8 MTP 正确性）是否覆盖当前 shape；
3. 若定位失败：MTP 的结论应写成"**性能 1.47× 可达，但输出 token 流与非投机路径不等价（最多 12.1 nat 自信翻转）⇒ 本平台当前不宜用于质量敏感场景**"，并把预算转投 S18-4c 的 baseline 内核提速（与 MTP 正交）。

## 7. 复现方式

```bash
# 一条臂 = 起服务 + flip 录 logprob + 探针 + decode bench + 杀引擎
docker exec -d z100-s18 bash -lc 'export VLLM_DCU_FP8_GEMV_MAX_M=1; export S18_FLIP=1; \
  export S18_PROBE_MODES=greedy; export S18_BENCH_MODES=decode; \
  bash /src/tools/dcu/s18_one_arm.sh fp8 4 1 8192 <suffix>'
# 对照（同容器，必须串行！）
docker exec -d z100-s18 bash -lc 'export S18_FLIP=1; export S18_PROBE_MODES=greedy; \
  export S18_BENCH_MODES=decode; bash /src/tools/dcu/s18_one_arm.sh fp8 4 0 8192 ctl'
# 离线逐 token 对照
python3 tools/dcu/s11l_flip_compare.py --a flip-fp8-tp4-k0-m8192ctl.jsonl \
        --b flip-fp8-tp4-k1-m8192k1gemmdisp.jsonl --label-a ctl --label-b k1gemm
```
产物：`/mnt/game/s18/flip-*.jsonl`、`summary-*.md`、`probe-*.json`、`bench-*.log`；回归前原始件在 `/mnt/game/s18/backup/pre-regression-*`，子代理证据在 `/mnt/game/s18/batteryB-evidence/`。

## 8. 参考

- [S18d](z100-18d-mtp-gemm-dispatch-e2e-result.md)（K4 通过、GEMM 派发的端到端收益）
- [S18c](z100-18c-gemm-dispatch-m-flat-result.md)（GEMM 对 M 平坦的内核证据）
- `vllm/model_executor/layers/mamba/gdn_linear_attn.py:935-960`（spec 分支的转置状态处理）
- `docs/result/z100-14d-determinism-result.md`（本项目"token 流可复现性"的既有裁定口径）

---

## 附录 A（2026-10-12 追加）：分叉形态的进一步刻画与"回滚"假设

### A.1 不是位置错位，也不是采样错误

对每个分叉点，把 MTP 臂的 top-5 与控制臂 **相邻位置** 的 top-5 做 Jaccard 比较：

| prompt | 分叉位置 | Jaccard 峰值位置 | 结论 |
|---|---|---|---|
| `zh_sky` | 15 | **j=15 → 1.00** | 同位置、**top-5 完全相同**（`[简短,简洁,简要,回答,简单]` vs `[简洁,简短,回答,简要,简单]`），只是排序不同 |
| `zh_ml` | 16 | **j=16 → 1.00** | 同上（`[机器,概念,机器学习,定义,清楚]` 重排） |
| `en_sea` | 2 | j=2 → 0.43 | 同位置，候选集部分重叠（`to/answer/respond`） |
| `en_fr` | 1 | j=1 → 0.25 | 同位置，候选集部分重叠（`need/should`） |

⇒ **排除"位置错位（off-by-one）"**（若错位，相似度会在 j=div±1 处见峰）；也排除"采样/argmax 实现错误"（MTP 臂选的确实是它自己分布里的 top-1）。

### A.2 目标分布在 verify 路径上被实质改变（原始 logprob）

`en_fr`（"What is the capital of France?"）：

```
i=1  ctl : ' need'  lp=-0.000 gap=12.109   top5=[need -0.00, answer -12.11, should -12.78, user -13.70, need -13.79]
     k1  : ' should' lp=-0.552 gap= 0.328   top5=[should -0.55, need -0.88, '\n' -5.29, have -6.21, must -7.47]
```

- 控制臂在该位置**几乎是确定性的**（` need` p≈1，领先 12.11 nat），MTP 臂却给出一个**更平、且把 ` need` 排到第二**的分布 ⇒ 用数值噪声无法解释；
- `zh_sky` 在**匹配**位置也有系统性偏移：i=1 `用` ctl lp=-0.240 vs k1 lp=-0.426；i=3 `问` ctl -0.040 vs k1 -0.099 ⇒ 整条分布有 ~0.05–0.2 nat 的平移，叠加在个别位置放大成定性翻转；
- 两臂输出都连贯、最终答案都正确（§3.3）⇒ 不是崩溃，而是**分布级扰动**。

### A.3 假设（供下一步判别）

**H-R（回滚不彻底，当前首选）**：verify 步里被拒绝的 draft token **没有从 mamba/GDN 状态（以及/或 KV）中完整回滚**，于是下一个 verify 步的上下文多/少了 token，logits 被系统性改变。与观察一致：① 扰动出现在**第 1–2 个生成 token**（第一次 verify 之后）；② 分布"变平 + 同候选重排"；③ 与派发无关；④ 本项目历史上已修过同族的 spec-token 状态问题（`7026983a8`、`8cfc349ac`、`e2191306c`）。

**H-P（前缀缓存/块对齐）**：spec token 写入 KV 后，块布局/对齐（`mamba-cache-mode align`）与回滚的交互出错。

正在跑的 4 条臂正是这两条假设的直接判别：`--no-enable-prefix-caching` × {k=0,k=1} 与 `--mamba-cache-mode none --no-enable-prefix-caching` × {k=0,k=1}（各带 flip 录制，两两成对比较）。

---

## 附录 B（2026-10-12 追加）：缓存侧变量全部排除 + 行为完全可复现

### B.1 三对对照给出**逐位相同**的数字

| 对照（k=0 vs k=1+GEMM，FP8/TP=4/eager/maxlen 8192） | flips | 首次分叉（`en_fr` / `en_sea` 的间距） | matched-position `mean/max |Δlogprob|` |
|---|---|---|---|
| 默认（`mamba align` + 前缀缓存） | 280/469（0.597） | **12.109 / 8.437 nat** | 0.0535 / 0.7094 |
| `--no-enable-prefix-caching` | **280/469（0.597）** | **12.109 / 8.437** | **0.0535 / 0.7094** |
| `--mamba-cache-mode none --no-enable-prefix-caching` | **280/469（0.597）** | **12.109 / 8.437** | **0.0535 / 0.7094** |

⇒ **前缀缓存与 `mamba-cache-mode` 均被排除**：整张分叉表（位置、两侧 token、间距、逐位置 logprob）**完全可复现**，跨引擎重启、跨缓存模式都一样。

### B.2 由此得到的两个结论

1. **不是随机抖动、不是缓存/状态搬运**。分叉是 **spec 路径的确定性性质**：同样的输入必得同样的输出（这对后续定位是好消息——任何候选修复都能用本协议一键验证）；
2. 嫌疑集中到 **spec-decode 代码路径本身**：V2 speculator（默认）的 verify/logits 行映射、accept 判定，或 position 处理。正在跑 **`VLLM_USE_V2_MODEL_RUNNER=0`（V1 `llm_base_proposer`）** 的成对对照：若 V1 干净 ⇒ 病因在 V2 speculator；若 V1 同样分叉 ⇒ 病因在两条路径共享的部分（rejection sampler / verify metadata / position）。

### B.3 登记：自伤事故（附教训）

为支持 A/B 新增的 `S18_EXTRA` 在 `set -u` 下写成裸 `$S18_EXTRA`，导致**所有未设置该变量的臂启动即失败**（`line 100: S18_EXTRA: unbound variable`）。V1 对照的首跑因此报废（两条臂都被标为 failed），已改为 `${S18_EXTRA:-}` 并复跑。教训：新增可选环境变量必须用 `${VAR:-}`；同时也说明本工具链的失败是**响亮**的（日志一行即定位）。

---

## 附录 C（2026-10-12 追加）：V1 runner 判别不可用；转入 op 级二分

- **`VLLM_USE_V2_MODEL_RUNNER=0`（V1 `llm_base_proposer`）在本 fork 上起不来**：两条臂均 `ready=0`，`/health` 超时后 flip 探针拿到 `HTTP Error 500: Internal Server Error`（`serve-fp8-tp4-k0-m8192ctl_v1.log` 尾部为正常 shutdown，无 traceback 落地）⇒ **该路径不能作为判别臂**（也登记为"本 fork 的 V1 spec 路径不可用"这一事实）；
- 已据此把定位手段改为**代码级二分**：给两个 gfx906 spec 专用 op 各加一个**默认不变**的 env 开关
  - `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py:616` — `VLLM_QWEN35_MTP_GDN_CUSTOM=0` 跳过 GDN MTP kernel；
  - `vllm/model_executor/layers/mamba/ops/causal_conv1d.py:1442` — `VLLM_QWEN35_MTP_CONV_CUSTOM=0` 跳过 conv MTP kernel；
  两条臂（各 k=1 + GEMM + flip）在跑，与既有 `ctl` 的 flip 文件做 `s18_parity_report.py` 对照。
- 新增离线工具 `tools/dcu/s18_parity_report.py`：一条命令给出 flips/位置、首次分叉两侧间距、**位置平移检查**（相邻 index 的 top-5 Jaccard）、匹配位置 `|Δlogprob|` 噪声代理与 VERDICT，避免每次手写分析。

---

## 附录 D（2026-10-12 追加）：rejection sampler 被排除；病因锁定在"目标 verify 前向的 logits"

- **`tests/v1/sample/test_rejection_sampler.py` 在 gfx906 上 75 passed / 0 failed**（`--noconftest`，48.8 s）⇒ Triton 的 greedy/随机接受判定（V1 实现：`token_id = target_argmax_id`、`rejected = draft != target_argmax`，`vllm/v1/sample/rejection_sampler.py:763-781`）在本平台**正确**；
- 已排除清单（每条都有成对真机证据）：

| 候选 | 判定 | 证据 |
|---|---|---|
| 前缀缓存 × spec | ❌ 排除 | `--no-enable-prefix-caching` 成对臂与默认**逐位相同**（附录 B） |
| `mamba-cache-mode align` × spec | ❌ 排除 | `--mamba-cache-mode none --no-enable-prefix-caching` 同样逐位相同 |
| GDN MTP 自定义 op | ❌ 非主因 | `VLLM_QWEN35_MTP_GDN_CUSTOM=0` 后仍分叉（267/463，`en_fr` 分叉**完全不变**），仅少数 prompt 的分叉点后移 |
| rejection sampler / 接受逻辑 | ❌ 排除 | 75/75 单测通过 |
| 数值噪声 | ❌ 排除 | `en_fr` 12.11 nat vs 0.328，且匹配位置 `mean|Δlogprob|`=0.054 |
| 位置错位 | ❌ 排除 | 相邻 index Jaccard 峰值都在同 index（附录 A） |

⇒ 病因收窄为 **目标模型在 verify 批次上的前向本身**：① attention（`TRITON_ATTN` 的 gfx906 多 token query 分支/`triton_unified_attention.py` 的 gfx906 特化）；② verify 行的 position/RoPE；③ conv MTP 状态更新（`noconv` 臂在跑）；④ spec metadata（`spec_state_indices` / `num_accepted_tokens` 的 Python 侧语义）。

### D.1 一处口径提醒（并发导致的污染，勿误读）

`nogdn` 臂的 **probe 吞吐被污染**：我在它的 probe 期间并发跑了 `test_rejection_sampler.py`（GPU/CPU 争用），该臂 probe 记录到 `tok/s=2.03 / wall=473.7s`，**不可与他臂比较**。其 **flip 数据在争用之前已完成**（且逐 token 结果本身是确定性的），因此 §附录 D 的 parity 结论不受影响；`nogdn` 的吞吐数字作废。

---

## 附录 E（2026-10-12 追加）：两个 gfx906 spec 专用 op **都参与**，且各自之外仍有残余

三条 op 级二分臂（均为 FP8/TP=4/eager/maxlen 8192，与同一 `ctl` 对照）：

| 臂 | flips | `en_fr` 首次分叉 | `en_sea` 首次分叉 | 自信翻转（≥1 nat） | 匹配位置 `mean/max |Δlogprob|` |
|---|---|---|---|---|---|---|
| k=1（默认，两个 op 都开） | 280/469（0.597） | **div@1，12.109 nat** | div@2，8.437 | 2 | 0.0535 / 0.7094 |
| k=1 + `VLLM_QWEN35_MTP_GDN_CUSTOM=0` | 267/463（0.577） | **div@1，12.109（完全不变）** | div@2，8.437 | 2 | 0.0613 / 0.8603 |
| k=1 + `VLLM_QWEN35_MTP_CONV_CUSTOM=0` | 259/437（0.593） | **identical（分叉消失！）** | **div@55**（从第 2 后移） | **1** | 0.0496 / 1.3494 |

**判读**：

1. **conv MTP op（`causal_conv1d_gfx906_mtp_update`）是 `en_fr`/`en_sea` 那两个"自信翻转"的直接来源**：关掉它后 `en_fr` 在 64 token 内**逐位一致**，`en_sea` 的分叉从第 2 个 token 后移到第 55 个；
2. **GDN MTP op 对这两个 case 无影响**（关掉后 12.109 一字不差），但它**能改变其它 prompt 的分叉点**（`zh_sort` 从 div@19 移到 14）⇒ 两个 op 都在贡献，量级不同；
3. **关掉 conv op 后仍有残余**（`zh_sky` 3.406 nat、`zh_ml` 1.063 nat 的自信翻转，`max|Δlogprob|` 反而升到 1.35）⇒ **至少还有一个非这两个 op 的来源**；
4. 组合臂（两个都关，k=1 与 k=2）在跑，用来判定"两个 op 是否就是全部来源"。

**工程含义**：若组合臂干净，则 K2 可以靠"spec 下绕开这两个 gfx906 专用 op"闭合（两者在 decode 步里合计占比很小，S11b 实测 gdn+mamba ≈0.4%），代价需要由组合臂的 TPOT 实测确认；若组合臂仍有自信翻转，则残余来源应在 attention/position/metadata 一侧，需要instrumentation 级定位。

### E.1 组合臂（两个 op 都关）与**代价**

| 臂（k=1 + GEMM） | flips | 自信翻转（≥1 nat） | 首次分叉 margins | 匹配位置 `mean|Δlogprob|` |
|---|---|---|---|---|
| 默认（两 op 开） | 280/469 | **2**（en_fr 12.109、en_sea 8.437） | — | 0.0535 |
| 只关 GDN op | 267/463 | 2（en_fr **不变**） | — | 0.0613 |
| 只关 conv op | 259/437 | 1 | en_fr **identical**、en_sea div@55 | 0.0496 |
| **两个都关** | 259/436 | **1**（zh_sky 3.406） | zh_hz 0.125 / zh_ml 0.469 / zh_sort 0.031 / zh_sky **3.406** / en_sea 0.188 / en_index 0.062 / en_math 0.484；en_fr identical | **0.0356** |

- **收益**：关掉 conv op（GDN op 无此效果）后 `en_fr` 从 12.1 nat 自信翻转**变成 64 token 逐位一致**；`en_sea` 分叉从第 2 后移到第 55；匹配位置的平均 logprob 偏移从 0.054 降到 0.036 ⇒ 两个 op 是**可量化的**分叉来源之一；
- **残余**：仍有 1 个自信翻转（`zh_sky` 3.406 nat；控制臂的 `问` 在对方分布里从 -0.04 掉到 -1.83，即 **1.79 nat 的分布位移**），其余 7 条落在 0.03–0.48 nat（符合项目"贴边翻转"口径）⇒ **至少还有一个非这两个 op 的来源**（attention 多 token query / position / 其余按 M 分支的算子）；
- **代价（决定性否决"关 op"方案）**：两 op 都关时引擎吞吐掉到 **Accepted throughput ≈0.95–1.10 tok/s**（正常 6–7，**≈6× 慢**），该臂 probe 10 分钟未跑完 ⇒ **不能靠关 op 上线**；同时也证明这两个 op 与 fallback 路径**数值不等价**（这正是它们必须被修正、而不是被绕过的原因）。

---

## 附录 F（2026-10-12 追加）：**修正附录 E 的结论** —— conv MTP op 与主线语义逐位一致；fallback 才是错的

新探针 `tools/dcu/s18_conv_mtp_sequence.py` 用**多步状态演化**（5 步，`(query_len, num_accepted)` = (2,1)(1,1)(1,2)(2,2)(3,1)）把三条路径与 torch 参考逐位对照：

```
--- custom      step 0..4 max|out-ref| = 0.0000e+00 ×5   final state max|state-ref| = 0.0000e+00
--- triton      step 0..4 max|out-ref| = 0.0000e+00 ×5   final state max|state-ref| = 0.0000e+00
--- fallback    step 0..4 max|out-ref| = 0.201 / 0.468 / 0.276 / 0.419 / 0.896
                final state max|state-ref| = 1.6416
```

（`triton` = 打开 `VLLM_ROCM_USE_GFX906_TRITON_CAUSAL_CONV1D_UPDATE=1` 绕开 gfx906 fallback，走**上游主线**实现。）

**三条结论**：

1. **gfx906 MTP conv 自定义 op 是主线正确的**：它与上游 triton 实现在 5 步演化、变化 `query_len`/`num_accepted` 下**都逐位一致**，且都精确等于文档语义的参考 ⇒ **附录 E 中"conv op 是那两个自信翻转的直接来源"的表述必须更正为**："关掉 conv op 会把 conv 更新切到一条**错误**的 fallback（见下），因此该臂的 parity 变化不能作为 op 有错的证据"；
2. **fallback 路径存在真实缺陷**（输出最大差 0.9、状态差 1.64）：它在 `num_accepted_tokens` 的滑窗语义上不正确。默认路径不走它（`VLLM_ROCM_USE_GFX906_TRITON_CAUSAL_CONV1D_UPDATE=0` 且 eligible），但一旦 eligibility 失败（异常 dtype/布局）就会**静默走错**——**登记为独立缺陷**（建议加"不 eligible 即报错"的门禁，或让 fallback 走 triton）；
3. **K2 病灶重新收窄**：gfx906 的 spec 专用 kernel（conv / GDN）数值上都没问题，rejection sampler 也没问题，缓存模式也没问题 ⇒ 剩下的只能是**喂给这些算子的元数据/状态**（`spec_token_indx` / `spec_state_indices` / `num_accepted_tokens` 的 Python 侧语义，`gdn_linear_attn.py:777-831`）或 **attention / position** 一侧。

**下一步（已定）**：对首个分叉步做**参数级 instrumentation**——在容器里 monkeypatch `causal_conv1d_update` 与 `fused_sigmoid_gating_delta_rule_update`，把前 N 次调用的 `query_start_loc` / `num_accepted_tokens` / `state_indices` / 张量形状打印出来，分别跑 k=0 与 k=1（同一 prompt），逐字段对照"正确 spec 步应有的参数"。

---

## 附录 G（2026-10-12 追加）：参数级 instrumentation 打通；spec 参数形态正常；GDN 契约测试待修正

### G.1 注入通道（S14g 先例）已打通，并修掉一个坑

- 安装 `/usr/local/lib/python3.10/dist-packages/zz_s18_args.pth`（实验后已删除并核对）；
- **坑**：在 `.pth` 里**急切 import vllm** 会把引擎打死（`transformers` 导入链尚未就绪，`ready=0`，22 s 退出）。修法：`.pth` 只启动一个守护线程，轮询 `sys.modules` 直到目标模块出现再 patch；
- 结果：5 个进程（API + 4 worker）全部 `installed`，日志落地 `/mnt/game/s18/s18_args-*.log`。

### G.2 k=1 首请求的 spec 参数（原始）

```
[conv r0 #1] x=None state=None qsl=[0, 2] acc=[1] idx=[1] mql=2
[gdn  r0 #49] q=(1, 2, 4, 128)/float16 state=(1380, 12, 128, 128)/float32
              cu=[0, 2] acc=[1] idx=[[1, 2]] inplace=True
[conv r0 #2] qsl=[0, 2] acc=[1] idx=[3] mql=2          # 下一步 slot 前进
[gdn  r0 #50] cu=[0, 2] acc=[1] idx=[[3, 4]] inplace=True
```

判读：每步 2 个 query token（1 sampled + 1 draft）、`num_accepted_tokens=[1]`、conv/GDN 各拿到本步的 slot 索引；**从形态上看不出错位**（要判定是否"应当为 2"需要知道该步是否接受了 draft——单请求 6 token 里观测到的两步 `acc` 都是 1，与"draft 被拒绝"一致）。⇒ **metadata 侧暂未发现 smoking gun**。
（`x=None state=None` 是因为调用点是按位置参数传的，patch 只读了 kwargs；后续如需张量级证据要改成读 `a`/`args`。）

### G.3 GDN MTP op 的契约测试：**未定论**（新坑已登记）

新探针 `tools/dcu/s18_gdn_mtp_contract.py` 试图验证"op(T 个 token, 接受 a 个) ⇒ 前 a 位的输出与最终状态 == 非投机路径只跑前 a 个 token"。过程中撞到三件事：

1. op 的 `TORCH_CHECK` 明确要求 **`state` 为 `[slots, kv_heads, value_dim, key_dim]`（VxK 转置视图）**（`csrc/mamba/gfx906_decode_kernels.cu:2509-2510`）——与 `gdn_linear_attn.py:944` 传 `ssm_state.transpose(-1,-2)` 一致；我首版用了 KxV ⇒ 对比无效；
2. 改成传转置视图后，**非投机的参考路径**（`_fused_sigmoid_gating_delta_rule_indexed_decode_gfx906`，`fused_sigmoid_gating.py:219`）在 `v_residual = v_t - sum(state * k_t...)` 上形状不匹配（8 vs 16）⇒ **该参考路径有自己的布局约定**（同样需要转置视图或另一条路径），不能拿"裸 cache"当参考；
3. 结论：**契约测试必须严格镜像每条路径的视图约定**（MTP 传转置视图、非投机参考按其约定传），否则得到的是"布局差异"而不是"数值错误"。首版的 0.1225/0.3884 差异**全部作废**，不作为结论。

**下一步（收敛的）**：把参考换成**纯 torch 的 GDN 递推实现**（不经过 vLLM 的任何 gfx906 路径），用它同时校验 MTP op 与非投机路径；这样得到的差异才可归因。

---

## 附录 H（2026-10-12 追加）：**两个 gfx906 spec 专用 kernel 全部被证明数值正确** ⇒ K2 病因不在 kernel

### H.1 GDN MTP kernel vs torch eager 参考：**逐位一致**

先把 op 的真实语义从源码钉死（`csrc/mamba/gfx906_decode_kernels.cu:544-563`）：

```c
initial_position = num_accepted_tokens[req] - 1;
for (token_position = 0; token_position < max_query_len; ++token_position) {
  source_position = (token_position == 0) ? initial_position : token_position - 1;
  source_state_idx = state_indices[req, source_position];   // token 0 从 acc-1 列取起始状态
  ...                                                       // token t>0 链式读 t-1 写的槽
```
⇒ **op 走完整个 speculative token 链，`acc` 只用于选 token 0 的起始状态槽**（与 torch eager 参考 `fused_sigmoid_gating.py:89-101,140-146` 完全同构：`init_token_idx = acc-1`、逐 token 写目标槽、state 布局 `[slots, HV, V, K]`）。

按此语义做**同输入对照**（同 `acc`、同 per-token 槽表 `[[1,2,3,4]]`、同起始状态、state 传 `[slots,HV,V,K]` 视图）：

```
T=4 heads=2 v_heads=4 K=16 V=16 slots=6 state_indices=[[1, 2, 3, 4]]
acc=1: max|out_op-out_ref|=0.0000e+00  touched-slot |dstate|={1:0.0, 2:0.0, 3:0.0, 4:0.0}
acc=2: max|out_op-out_ref|=0.0000e+00  touched-slot |dstate|={1:0.0, 2:0.0, 3:0.0, 4:0.0}
acc=3: max|out_op-out_ref|=0.0000e+00  touched-slot |dstate|={1:0.0, 2:0.0, 3:0.0, 4:0.0}
```

⇒ **GDN MTP kernel 与仓库内 torch 参考逐位一致**。

### H.2 作废与更正

- 附录 G 的 0.1225/0.3884 与后续 0.116/0.420 **全部作废**：它们来自我错误的契约假设（"只跑前 a 个 token"）与错误的视图（KxV vs VxK），不是 kernel 的错误；
- 探针过程中另发现两条**与本结论无关但值得登记**的脆弱点：① 非投机的 indexed-decode 路径（`fused_sigmoid_gating.py:219-221`）在 **K≠V** 时布局不匹配会崩——真实模型 GDN 是 **K=V=128**，所以从未暴露；② op 的 `TORCH_CHECK` 强制 `state` 为 `[slots, kv_heads, value_dim, key_dim]`，而不同路径约定相反，**任何 K≠V 的改动都会踩**。

### H.3 K2 病因的最终收窄

已排除（全部有真机/逐位证据）：前缀缓存、`mamba-cache-mode`、**conv MTP kernel**（另与上游 triton 交叉验证逐位一致）、**GDN MTP kernel**（与 torch 参考逐位一致）、rejection sampler（75/75）、数值噪声、位置错位；spec 参数（`qsl`/`acc`/`idx`/`cu`）形态正常。

⇒ 剩余只可能在：**① attention（`TRITON_ATTN` 的 gfx906 多 token query 分支）；② position/RoPE 在 verify 行上的取值；③ 其余按 M 分支的算子/元数据**。

**下一步（唯一剩下的定位手段）**：用已打通的 `.pth` 注入按**层**记录 hidden state（同一 prompt 分别跑 k=0 与 k=1，记录每层输出的统计量/指纹），找出第一处实质差异的层——直接指向组件，不再靠排除法。

---

## 附录 I（2026-10-12 追加）：**修掉 V1 model runner 真缺陷**；ngram 判别把 K2 病因钉到"通用 spec verify 路径"

### I.1 真缺陷与修复（独立于 MTP 的收益）

`vllm/v1/worker/gpu_model_runner.py` 的 `sample_tokens` 里有两处 `_update_states_after_model_execute` 调用：**4939 行是上游改签名后遗留的旧单参调用**，4941 行才是正确的双参调用。TypeError 发生在参数绑定阶段 ⇒ **整个 V1 model runner 必然崩溃**（本平台默认走 V2，所以一直没被发现）。

```
TypeError: GPUModelRunner._update_states_after_model_execute() missing 1 required positional arg
  File "/src/vllm/v1/worker/gpu_model_runner.py", line 4939, in sample_tokens
```

- **修复**：删掉 4939 行那份陈旧调用（保留 4941 的正确调用）；`py_compile` 通过；
- **验证**：修复前，`ngram` 投机（V2 不支持 ngram ⇒ 强制 V1）每次请求 **HTTP 500**；修复后 ngram 臂**正常起服务并完成 8 条 prompt 的 parity**；
- **附带能力**：**ngram 投机解码在本平台由"完全不可用"变为可用**（也不必再依赖 `VLLM_USE_V2_MODEL_RUNNER=0` 才能复现 V1 路径）。

### I.2 ngram 判别：分叉是**通用问题**，不是 MTP 专属

`ctl(k=0) vs ngram(k=2, V1 runner)`：

| 指标 | 数值 |
|---|---|
| flips | **337/444（0.759）** |
| 自信首次翻转（≥1 nat） | **5**（`zh_sky` **13.687**、`en_index` **11.742**、`en_sea` 5.344、`zh_ml` 4.25、`zh_sort` 1.328） |
| 匹配位置 `mean/max |Δlogprob|` | 0.0514 / 0.7832 nat |
| 位置平移检查 | 5/8 峰值在同 index（非错位） |

⇒ **不含 MTP 头、不含任何 MTP 专用 kernel、走 V1 runner 的 ngram 投机，同样与投机前输出有实质差异**。

### I.3 病因的最终判定

排除清单（全部有逐位/真机证据）：**MTP 头与 MTP 专用 kernel**（conv op 与上游 triton 逐位一致；GDN op 与 torch 参考逐位一致）、前缀缓存、`mamba-cache-mode`、rejection sampler（75/75）、数值噪声、位置错位、**两条 runner 的 Python 逻辑**（V1-ngram 与 V2-MTP 都分叉）。

⇒ 只剩**通用 verify 路径上的模型执行**：**① attention（多 token query 下的 gfx906 `TRITON_ATTN` / unified-attention 分支）；② verify 行的 position/RoPE；③ KV/slot 映射**。

**下一步**：用已打通的 `.pth` 注入做**逐层 hidden-state 指纹**（同 prompt，k=0 vs ngram-k=2 各记 64 层第一遍的统计量），找出第一处实质差异所在的层——若差异首现于 attention 层即指向 ①/③，若首现于 input/RoPE 即指向 ②。

---

## 附录 J（2026-10-12 追加）：**逐层二分把病因钉到 GDN 层（layer 0）的状态搬运**

### J.1 方法

用已打通的 `.pth` 注入 hook `Qwen3NextDecoderLayer.forward`（`qwen3_next.py:1564`），记录**每层输入**的统计量（mean/std/max + row 0 的统计与头两个元素）与 `positions`；同一 prompt（`en_sea`，4 token）分别跑 **k=0** 与 **k=1 + GEMM**。
关键陷阱（已修）：MTP 头内部也含一个 `Qwen3_5DecoderLayer`，其调用会插进日志 ⇒ 必须**按 `positions` 分组**而不是按 64 层切分。

对齐结果（rank 1，TP 分片一致）：

| 臂 | 分组 | 上下文 | 层数 |
|---|---|---|---|
| ctl (k=0) | group 2 | `pos=[60]`（单 token 解码） | 64 |
| k1 (k=1) | group 4 | `pos=[60, 61]`（verify；row 0 = position 60） | 64 |

两者 layer 0 的输入统计量**逐位相同**（`r0std=0.01574`、`r0_0=0.05615` 完全一致）⇒ 对齐成立、输入 token 相同。

### J.2 逐层差异（row 0 的输入统计，ctl vs k1）

| layer | ctl r0std | k1 r0std | 相对差 |
|---|---|---|---|
| **0（输入=embedding）** | 0.01574 | 0.01574 | **0.00e+00** |
| **1（输入=layer0 的输出）** | 0.02829 | 0.02736 | **3.3e-02** ← 首现差异 |
| 2 | 0.01866 | 0.03487 | **8.7e-01** |
| 3 | 0.02287 | 0.02360 | 3.2e-02 |
| … | ~0.02–0.06 | 略高 | 1e-2 ~ 2.6e-01 |
| 63 | 1.12253 | 1.28888 | 1.5e-01 |

### J.3 结论与下一步

- 模型 layer 0 是 **`linear_attention`（GDN）**（`layer_types[0]='linear_attention'`，full-attn 在 3/7/11…）；
- **输入逐位一致、输出开始分叉** ⇒ **差异源头在 GDN 层的计算**；
- 但我们已经证明 **GDN MTP kernel 与 conv MTP kernel 在"同输入"下逐位正确**（附录 F/H）⇒ **问题只能出在喂给该层的状态/元数据**：`conv_state` 槽内容、`ssm` 状态槽内容，或槽映射（`conv_state_indices` / `ssm_state_indices` / `num_accepted_tokens` 语义）；
- **下一步（精确到可一次验证）**：在同一个 patch 里给 conv/GDN 调用加**状态校验和**（对参与调用的槽做 `abs().sum()`/fingerprint），同一 prompt 跑 k=0 与 k=1，逐调用比对 prefill 与首个 decode/verify 步的状态内容——若 prefill 后写入的状态槽不同，就直接锁定状态写回/槽映射这一环。

---

## 附录 K（2026-10-12 追加）：**定位到 conv1d 状态**——packed 假设被否证，ssm 状态无罪，conv 状态两臂不同

### K.1 否证："非投机走 packed 快路、投机走 another 路径"不是原因

控制臂的解码步不走 `fused_sigmoid_gating_delta_rule_update`（走 packed recurrent decode），而 spec 掩码会让 packed 快路被跳过（`gdn_linear_attn.py:763-769`）。于是做判别：**把控制臂的 packed 快路也关掉**（`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0 VLLM_QWEN35_COMBINED_PACKED_DECODE=0`），再与 MTP k=1 比：

| 对照 | flips | 自信翻转 | 匹配位置 mean\|Δlogprob\| |
|---|---|---|---|
| ctl(packed ON) vs k1 | 280/469 | 2（en_fr 12.1、en_sea 8.4） | 0.0535 |
| ctl(packed OFF) vs k1 | 270/469 | **2（en_fr 12.0、en_sea 6.4）** | 0.0519 |
| ctl(packed ON) vs ctl(packed OFF) | 102/469 | **0**（仅贴边） | 0.0239 |

⇒ **两套 GDN 实现只贡献贴边级差异（0.024 nat），不是 K2 主因**；假设被否证。

### K.2 ssm 状态指纹：**两臂逐位相同**

在 `.pth` patch 里加状态指纹（对参与调用的槽做 `abs().sum()`），同 prompt（`en_sea`，4 token）跑 ctl(packed OFF) 与 k1；按 rank 对齐"首个 decode 步 ↔ 首个 verify 步"（k1 的起始槽 = `state_indices[req, acc-1]`）：

```
=== rank 1: 48 层全部 SAME
  L0  ctl slot 1: 4061.9624 | k1 slot 1: 4061.9624  SAME
  L1  ctl slot 2:  123.1555 | k1 slot 3:  123.1555  SAME   (槽号不同、值相同)
  L2  ctl slot 3:  103.6705 | k1 slot 5:  103.6705  SAME
=== rank 3: 48 层全部 SAME
  compared=48 same=48 diff=0（两个 rank 都是）
```

⇒ **ssm（GDN）状态搬运无罪**：同一逻辑步下，两臂读到的状态内容逐位相同（只有物理槽号不同）。

### K.3 conv1d 状态指纹：**两臂不同** ← 病灶

同一逻辑步（位置 60 的首个 decode / verify），rank 1、layer 0：

```
ctl(packed OFF): [state-conv r1 x=(1, 2560) idx=[1] acc=None qsl=None  fp={1: 23734.7344}]
k1 (spec):       [state-conv r1 x=(2, 2560) idx=[1] acc=[1] qsl=[0, 2] fp={1: 27642.0449}]
```

- 两臂读的都是**槽 1**（k1 的起始槽 = `idx[acc-1] = idx[0] = 1`），但指纹相差 **16%** ⇒ **prefill 之后写入该槽的 conv 状态不同**；
- 而 ssm 状态在同一比较下完全相同 ⇒ **不是"整个 mamba 状态池"的问题，而是 conv1d 状态这一步的写回/槽映射**；
- 与逐层二分（附录 J）自洽：layer 0 是 GDN，conv1d 的输出直接进入 GDN 输入 ⇒ layer 0 输出开始分叉。

### K.4 结论与下一步

**K2 的机制链**已收敛为：`prefill 结束时写入 conv_state 槽的内容在投机模式下与非投机模式不同` → 首个 verify 步的 conv 输出不同 → GDN 输入不同 → layer 0 输出分叉 → 逐层放大 → 贪心 token 分叉（自信翻转）。

**下一步（唯一剩下的一环）**：对比 **prefill 阶段的 conv 状态写回**——prefill 不走 `causal_conv1d_update`（我 hook 的是它），所以要对 chunked/fused prefill 的 conv 路径（`causal_conv1d_fn` 及 align 模式的 `block_idx_last_scheduled_token` / `initial_state_idx` 语义）做同样的指纹记录，判定是"写入内容不同"还是"槽映射不同（写到别的槽/别的 token 位置）"。

---

## 附录 L（2026-10-12 追加）：**K2 根因定位完成** —— conv1d 状态窗口选择在投机布局下错位（单元级复现）

### L.1 引擎侧证据（附录 K 续）

同一逻辑步（位置 60 的首个 decode ↔ 首个 verify）、rank 1、layer 0 的 conv 槽**逐状态位**指纹（对 dim 求和）：

```
ctl(packed OFF): idx=[1] acc=None qsl=None  posfp = 1[-612.3, -52.1, -612.3]              (3 个位置)
k1 (spec)      : idx=[1] acc=[1]   qsl=[0,2] posfp = 1[-78.8, -612.3, -52.1, -612.3]    (4 个位置)
```

⇒ 两臂的 conv 槽**长度不同**（`width-1=3` vs **4**，投机模式多一个位置），且内容关系为"k1 = 一个更老的位置 + ctl 的三个位置"。

### L.2 单元级复现：内核取的是"最老 3 个"，不是"最近 3 个"

`tools/dcu/s18_conv_window_test.py`（`state_len=4`、`width=4`、`acc=1`、已知窗口）：

```
|out - conv(last 3 positions)|  = 6.208e-01   <- 正确窗口（最近 width-1 个）
|out - conv(first 3 positions)| = 0.000e+00   <- 内核实际取值 [acc-1 : acc+width-2] = [0:3]
VERDICT: SHIFTED（丢掉最新历史 token）
```

⇒ **`causal_conv1d_gfx906_mtp_update` 的窗口选择 `[acc-1 : acc+width-2]` 仅在 `state_len == width-1`（非投机布局）时正确**；投机布局下 conv 槽更宽（`state_len = width-1 + 投机位置`），窗口应取**最近 `width-1` 个位置**，而内核取了最老的 `width-1` 个 ⇒ 丢掉最新历史。

### L.3 为什么既有的 48 个单测没抓到

`tests/kernels/mamba/test_causal_conv1d.py::test_gfx906_causal_conv1d_mtp_update` 的期望值**是按实现写的同一个约定**（`history = state[num_accepted-1 : num_accepted+width-2]`，且 `state_len=5` 时断言的是 `[acc-1:...]` 切片），因此"内核自洽"却与**非投机布局的实际语义**不一致 ⇒ 单测全绿但端到端分叉。

### L.4 完整机制链（K2 收口）

```
投机模式下 conv 槽比非投机多若干位置（本机实测 4 vs 3）
  → gfx906 MTP conv 内核按 [acc-1 : acc+width-2] 取窗口 ⇒ 取到"最老的 width-1 个"
  → 丢掉最新的历史 token ⇒ conv 输出与非投机路径不同
  → layer 0（GDN，conv 输出直接进 GDN 输入）输出分叉（附录 J：layer 0 输入一致、输出开始分叉）
  → 逐层放大 ⇒ 贪心 token 分叉（含 8.4–13.7 nat 的自信翻转）
```

与所有既有排除项自洽：ssm 状态逐位相同（附录 K）、GDN/conv 内核在"同输入"下逐位正确（附录 F/H，因为它们实现的就是这个约定）、rejection sampler 正确、缓存模式/前缀缓存无关。

### L.5 修复方向（下一轮）

1. **首选**：修 `causal_conv1d_gfx906_mtp_update` 的窗口选择——按 `state_len` 与 `width-1` 的差值补偿偏移（即取**最近 `width-1` 个位置**，等价于 offset = `state_len - (width-1) - (acc-1)` 这类形式），并核对 `pad_slot_id`/`initial_state_idx` 语义；
2. 同步**用真值语义重写单测**（现在的测试断言的是实现约定，必须先修测试），并补一个"宽状态（state_len > width-1）"的用例锁死；
3. 复验：`s18_conv_window_test.py` 变成 CORRECT → 重跑 parity 协议（附录 B 的一键口令）→ 若 flip 降到仅贴边（≤0.5 nat）即 K2 通过 → 进 S18-5 上线。

---

## 附录 M（2026-10-12 追加）：**修正附录 L 的结论** —— 内核与主线一致；病灶是"prefill 写入"与"update 偏移"约定不匹配

### M.1 上游裁判：`[acc-1 : acc+width-2]` **就是主线约定**

`tools/dcu/s18_conv_window_test.py`（`state_len=4`、`width=4`、`acc=1`、已知窗口）现在同时测三条路径：

```
custom_op(raw _C)   |out-conv(last3)|=6.208e-01  |out-conv(first3)|=0.000e+00  -> FIRST3
public_api(gfx906)  |out-conv(last3)|=6.208e-01  |out-conv(first3)|=0.000e+00  -> FIRST3
public_api(triton)  |out-conv(last3)|=6.208e-01  |out-conv(first3)|=1.221e-04  -> FIRST3
```

⇒ **上游 triton 实现同样取"从 `acc-1` 起的 width-1 个位置"**（仅 1.2e-04 数值差）⇒ 附录 L 的"内核丢最新历史"结论**作废**：内核行为与主线一致。

### M.2 真正的契约（源码）

`csrc/mamba/gfx906_decode_kernels.cu:222-238`：

```c
state_offset = num_accepted_tokens[req] - 1;              // 读取起点
history[p] = conv_state[state_idx, dim, state_offset + p]; // p = 0..width-2
guard: state_offset + width - 2 < state_len
```

即 conv 槽是一个**随接受数推进的窗口**：第 n 步的窗口从 `acc_n - 1` 开始。

- **非投机**：`state_len = width-1 = 3`，prefill 把最后 3 个 token 写在位置 [0..2] ⇒ 首个 decode 的 `acc=1`（offset 0）正好读 [0..2] ✓ **自洽**；
- **投机**：`state_len = 4`（实测非投机 3 / 投机 4），而观测到的槽内容是
  `k1 = [X, ctl0, ctl1, ctl2]`（ctl = `[ctl0, ctl1, ctl2]`）⇒ **prefill 往加宽后的槽里写了 4 个 token**，而 update 仍从 offset 0 读 3 个 ⇒ **窗口整体错位一位**（读到 [X, ctl0, ctl1]，丢掉最新的 ctl2）。

### M.3 修正后的机制链

```
投机模式把 conv 槽加宽到 width-1+k（本机 3 -> 4）
  → prefill（gfx906 自定义 causal_conv1d_fn / align 路径）仍按"填满 state_len 个位置"写状态
  → 首个 verify 步 update 以 acc-1=0 为窗口起点读 width-1 个 ⇒ 窗口错位一位（丢最新、含最老）
  → conv 输出 ≠ 非投机路径
  → layer 0（GDN，conv 输出直入 GDN 输入）首个分叉（附录 J）→ 逐层放大 → 贪心 token 分叉
```

（与全部既有证据自洽：ssm 状态逐位相同、conv/GDN 内核各自与上游/torch 参考逐位一致、rejection sampler 75/75、缓存模式无关。）

### M.4 修复方向（两条，二选一）

1. **让 prefill 只写 `width-1` 个 token 到槽的前 `width-1` 个位置**（把多出的 `k` 个位置留给本步的投机 token）——与 update 的 `acc-1` 约定天然自洽；
2. **或让 update 的偏移补偿槽宽差**：`offset = (state_len - (width-1)) - (acc-1)`（首步 acc=1 ⇒ offset 1 ⇒ 读 [1..3] ✓）。

**下一步**：读 `causal_conv1d_fn`（gfx906 prefill / align 路径）的状态写回代码，确认它到底写了几个 token（是否 `state_len`），再决定走哪条修法；修完先用 `s18_conv_window_test.py` 的"宽槽"用例做门禁，再重跑 parity 一键协议。

---

## 附录 N（2026-10-12 追加）：**K2 根因闭环** —— conv 槽必须"前对齐"，而 gfx906 prefill fallback 写的是"最后 cache_state_len 个"

### N.1 槽宽公式（已坐实）

`vllm/model_executor/layers/mamba/mamba_utils.py:213,264,292,323`：

```python
conv_state_shape = cls._orient_conv_shape(conv_dim, conv_kernel - 1 + num_spec)
```

⇒ 投机开启时 conv 槽宽 = `width - 1 + num_spec`（本机 `width=4`、`k=1` ⇒ **3 → 4**，与附录 K 实测的逐状态位指纹一致）。

### N.2 MTP update 内核要求的约定：**前对齐**

`csrc/mamba/gfx906_decode_kernels.cu:222-284`：

```c
state_offset = num_accepted_tokens[req] - 1;
history[p] = conv_state[state_idx, dim, state_offset + p];        // p = 0..width-2  读窗口
... 逐 token 计算/前移 ...
// 写回：把窗口归一化到槽的前部，新 token 追加在 width-2 起
for (p = 0; p < width-2; ++p)
    conv_state[base + p] = conv_state[base + (state_offset + p + 1)];
for (t = 0; t < query_len; ++t)
    conv_state[base + (width - 2 + t)] = candidate_tokens[t];
```

⇒ **每次调用后，状态被归一化为"最近的 `width-1` 个 token 位于槽的前 `width-1` 个位置"**，槽尾多出的 `num_spec` 个位置只承载本步新 token。下一步的窗口起点自然就是 `acc-1`。上游 triton 实现同约定（附录 M 实测三者一致）。

### N.3 病灶：prefill fallback 写的是"最后 `cache_state_len` 个"（不对齐）

`vllm/model_executor/layers/mamba/ops/causal_conv1d.py:54` `_causal_conv1d_gfx906_fallback`（gfx906 prefill 默认走它，见 commit `6728c1c4a`）：

```python
cache_state_len = conv_states.shape[-1]     # = width-1+num_spec（投机 = 4）
state_len       = width - 1                 # = 3
updated_state = full_x.squeeze(0)[:, -cache_state_len:]   # 取最后 4 个
if updated_state.shape[1] < cache_state_len:              # 末尾补零
    padded_state[:, -updated_state.shape[1]:] = updated_state
conv_states[cache_idx].copy_(updated_state)               # 整体写入 → 窗口在位置 [1..3]，而内核从 [0] 读
```

- **非投机**：`cache_state_len == state_len == 3` ⇒ 前对齐 == 后对齐，**两者重合，无 bug**；
- **投机**：`cache_state_len = 4 > 3` ⇒ prefill 把 `[X, ctl0, ctl1, ctl2]` 放在位置 [0..3]，而 update 内核按 `acc-1 = 0` 读 `[0..2] = [X, ctl0, ctl1]` ⇒ **窗口整体错位 `num_spec` 位，丢掉最新的 `ctl2`** —— 与附录 K 的逐状态位指纹（`k1 = [X, ctl0, ctl1, ctl2]` vs `ctl = [ctl0, ctl1, ctl2]`）完全吻合。

### N.4 修法（两处，非投机下皆为 no-op）

1. `_causal_conv1d_gfx906_fallback`（prefill）：把**最近 `state_len = width-1` 个 token 放在槽的前 `state_len` 个位置**，槽尾 `num_spec` 个位置留空/清零（保持"前对齐"）；
2. `_causal_conv1d_gfx906_decode_update_fallback`（非投机解码）：读取/写回由 `[:, -state_len:]` 改为 `[:, :state_len]`（前对齐）。
   - 非投机时 `cache_state_len == state_len` ⇒ 两者逐位等价，**零风险**；投机时与 MTP 内核统一到"前对齐"。

（不改内核：内核与上游 triton 一致，且其写回已强制前对齐。）

### N.5 验收门禁（下一轮执行）

1. **单元门禁**：扩展 `tools/dcu/s18_conv_window_test.py` 为"宽槽（`state_len = width-1+k`）+ 前对齐 prefill 写入"场景，断言首个 verify 步的输出 == 非投机参考；
2. **端到端门禁**：重跑 parity 一键协议（ctl(k=0) vs k=1），期望 flip 从 270–280/469 降到"仅贴边"（≤0.5 nat，0 个自信翻转）；
3. 通过后：跑 k=2/k=3 与 8K/conc 场景，然后进 **S18-5 上线**（`S11F_MTP_TOKENS`/`S11F_MTP_GEMM` 已就位、默认关）。

---

## 附录 O（2026-10-12 追加）：**K2 修复完成并通过门禁**

### O.1 修复内容（`vllm/model_executor/layers/mamba/ops/causal_conv1d.py`）

按主线 FIFO 语义（`causal_conv1d.py` triton update 内核注释 1118-1133：窗口从 `acc-1` 向前读、更新后左移 `acc` 并在末尾追加）把 gfx906 两处 fallback 统一到**前对齐**：

1. 新增 `_gfx906_conv_window_read()` / `_gfx906_conv_window_write()`：槽宽 `cache_len = width-1+num_spec` 大于 `width-1` 时，窗口取**前** `width-1` 个位置；相等时与原先的 `[-width+1:]` **逐位等价**（非投机零影响）；
2. `_causal_conv1d_gfx906_fallback`（prefill）：2 处读 + 2 处写改走上述 helper，写入时把"最近 `width-1` 个 token"放在槽的前部、槽尾 `num_spec` 位清零；
3. `_causal_conv1d_gfx906_decode_update_fallback`（非投机解码）：读/写同样改走 helper。

`git diff --stat`：`1 file changed, 59 insertions(+), 27 deletions(-)`；`py_compile` 通过。

### O.2 单元门禁（PASS）

`tools/dcu/s18_conv_window_test.py`，`state_len=4`、`width=4`、`acc=1`：

```
custom_op(raw _C)        |out-conv(last3)|=6.208e-01  |out-conv(first3)|=0.000e+00
public_api(gfx906)       |out-conv(last3)|=6.208e-01  |out-conv(first3)|=0.000e+00
public_api(triton)       |out-conv(last3)|=6.208e-01  |out-conv(first3)|=1.221e-04
FIXED front-aligned slot |out - conv(last3)| = 0.000e+00  -> PASS
```

⇒ 修复后的 prefill 布局（最新 `width-1` 个在前）+ MTP 内核 ⇒ **与非投机参考逐位一致**。

### O.3 端到端门禁（PASS）

`ctl(k=0) vs k=1(GEMM 派发, 修复后)`，同 S9 协议、8 条 prompt：

| 指标 | 修复前 | **修复后** |
|---|---|---|
| flips | 280/469（ctl packed ON）/ 270/469（OFF） | **154/447（0.345）** |
| **自信首次翻转（≥1 nat）** | **2**（en_fr 12.1、en_sea 8.4–6.4） | **0** |
| 匹配位置 `mean |Δlogprob|` | 0.052–0.054 nat | **0.0338 nat** |
| 判定 | `FLIP-AT-CONFIDENT -> material difference` | **`near-tie flips only -> numeric-level difference`** |

- 残差为**贴边翻转**（最大匹配位置差 0.738 nat，首次分叉处 gap 0.0–0.72 nat），与项目自身基线同量级：`ctl(packed ON) vs ctl(packed OFF)` = 102/469 贴边翻转、**0** 自信翻转、mean 0.0239 nat ⇒ 属**数值级差异**，非实现错误；
- `en_fr`、`en_math` 完全一致；其余 6 条均 top5-jaccard ≥0.667、峰值同 index（非错位）。

### O.4 结论

**K2（投机 vs 非投机贪心 token 一致）通过**：不再存在实质差异，仅剩 fp16 数值级贴边翻转。

**KPI 汇总**：K4 性能 ✅（k=2 = 83.92 ms/token、1.47×、10.94 tok/s）；K2 一致性 ✅（本轮）；K1/K3 见正文。

**下一步（S18-5）**：在修复后的树上重跑 k/TP 矩阵与 8K/conc8，确认性能未回退，然后按方案开生产（`S11F_MTP_TOKENS`/`S11F_MTP_GEMM`，默认关 → 显式开启），并把 conv-state 前对齐的单元用例补进 `tests/kernels/mamba/test_causal_conv1d.py`（现测试断言的是旧实现约定）。

---

## 附录 P（2026-10-12 追加）：回归用例锁定修复

### P.1 新增用例

`tests/kernels/mamba/test_causal_conv1d.py::test_gfx906_conv_state_front_alignment`（`num_spec = 0/1/2`，gfx906 专属）：

- 用 `_causal_conv1d_gfx906_fallback` 对长度 6 的序列做 prefill，槽宽 = `width-1+num_spec`；
- 断言 ① 最新 `width-1` 个 token 位于槽的**前** `width-1` 个位置；② `num_spec>0` 时槽尾保留位全 0；③ `_gfx906_conv_window_read(slot, width-1)` 读回正是该窗口。

**旧代码必挂**：旧对齐写 `full_x[:, -cache_len:]` ⇒ `slot[:, :3] = [X, ctl0, ctl1] ≠ newest = [ctl0, ctl1, ctl2]`（`num_spec=1`）。

### P.2 既有测试不变且仍全绿

`test_gfx906_causal_conv1d_mtp_update`（48 例）断言的是 **update 内核**的 FIFO 约定（`history = state[acc-1 : acc+width-2]`）——本次修复不动内核，故该约定不变、测试保持有效。

### P.3 运行结果

```
$ python -m pytest tests/kernels/mamba/test_causal_conv1d.py -k "gfx906 and (front_alignment or mtp_update)" -q
51 passed, 165 deselected, 18 warnings in 28.52s
```
（= 3 个新用例 + 48 个既有用例）

---

## 附录 Q（2026-10-12 追加）：修复后的性能复验（无回退）

同一 arm 驱动（`fp8 tp=4 k=2 maxlen=8192`），修复后的树：

| 指标 | 修复前（S18d / k1fix 前） | **修复后（k2fix）** |
|---|---|---|
| TPOT（decode bench） | 83.80 / 83.92 ms/token | **80.44 ms/token** |
| 输出吞吐 | 10.94 tok/s | **11.34 tok/s**（probe 10.72、长跑 960 tok @10.35 tok/s） |
| 接受率 / MAL | 0.62–0.78 / 2.25 | **0.6218 / 2.244** |
| TTFT（8K 前 prologue） | 32238.67 ms(ctl) / 32748.85 ms(k=2) | 1072.95 ms（短上下文 bench）；conc 项 15148.56 ms |
| 启动 | 143–198 s | 184 s |
| 修复后 K2 parity | — | **154/447 贴边、0 自信翻转、mean 0.0338 nat** |

⇒ 修复（仅改 prefill 状态写回对齐）**不影响吞吐与接受率**，且消除了实质 token 分叉。

## S18 结论汇总

| 项 | 结果 | 归档 |
|---|---|---|
| 生产停机与回滚配方 | ✅ | `z100-18a-*` |
| 工具/单测基线（48/48 gfx906 MTP 用例） | ✅ | `z100-18a-*` |
| A0/A1/A2 基线（A0 = 123.32 ms/token） | ✅ | `z100-18b-*` |
| 成本模型 + k/TP 矩阵（`≈123+105×(M−1)` ms/step） | ✅ | `z100-18b/c-*` |
| GEMM 派发使 M 平坦（GEMV 7.38× → GEMM 0.98×） | ✅ | `z100-18c-*` |
| **K4 性能**：k=1 99.34 ms（1.24×）、**k=2 83.80–83.92 → 80.44 ms（1.47×，10.9–11.3 tok/s）** | ✅ | `z100-18d/k2fix` |
| V1 model runner 真缺陷修复（`gpu_model_runner.py:4939`）⇒ **ngram 投机变可用** | ✅ | `z100-18e` 附录 I |
| gfx906 MTP 内核全部证明正确（conv vs 上游 triton、GDN vs torch 参考，逐位一致） | ✅ | 附录 F/H |
| **K2 根因**：投机加宽 conv 槽后，prefill fallback 写"最后 cache_len 个"而非"前 width-1 个" | ✅ | 附录 K/M/N |
| **K2 修复**：两处 fallback 统一前对齐（非投机逐位等价） | ✅ | 附录 O |
| **K2 门禁**：154/447 贴边、**0 自信翻转**、`numeric-level difference` | ✅ | 附录 O |
| 回归用例锁定（`test_gfx906_conv_state_front_alignment`，51 passed） | ✅ | 附录 P |
| 性能复验（无回退） | ✅ | 附录 Q |
| INT8 诊断臂 | ⛔ 未跑 —— 用户在中途把目标槽定为 **FP8 block-scaled**，INT8 仅作诊断，已无必要 | — |
| S18-5 开生产 | ⏸ 待用户决定（生产 `:8000` 目前仍停，一行回滚配方不变） | — |
