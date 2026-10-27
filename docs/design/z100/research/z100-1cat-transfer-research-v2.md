# 借鉴 1Cat-vLLM 提升 Z100L（v2 · 代码级）

> **本报告是 v1（`z100-1cat-transfer-research.md`，原文已移除，见 git 历史）的续篇与更正**。
> v1 只拿到 1Cat 的 README + 28 份设计文档（当时网络 53 KB/s，clone 失败）；本次拿到了**完整源码树（Oct 8 快照）+ 最新元数据（Oct 23）**，因此能做代码级判断，并**更正 v1 中三处被代码否掉的推断**（见 §5）。
>
> **1Cat 代码版本**：本地 `$DATA_ROOT/1cat-vllm` 为 `c4f6245`（PR #1041，2026-10-08）；
> 已同步到 `origin/main` = **`6a143197e`（PR #1259，2026-10-23 19:12 UTC）**，落后 **569 个提交**（`compare` API `total_commits=569`）。
> **同步方式说明**：`$DATA_ROOT` 属 root 且 `safe.directory` 只覆盖工作树，无法原地 fetch；完整增量 fetch 实测约 9 KB/s（14 分钟 7.8 MB），不可行。最终用 **`git fetch --filter=blob:none --depth=1`（只拉元数据）成功同步到 `origin/main`**，因此拿到了**精确的 987 个文件变化清单**（`git diff-tree`）；但**磁盘上的文件内容仍是 Oct 8 的**，关键模块的最新内容按需从 `raw.githubusercontent.com/main` 单独拉取。§8 说明由此带来的边界。
>
> **日期**：2026-10-24 · **性质**：调研报告（只读分析，未启动任何服务、未产生新性能数字）

---

## 0. 一页结论

### 0.1 一句话

> **1Cat 真正值钱的东西不是那 7 万行 SM70 内核（gfx906 一行都用不了），而是"100 行控制面 + 18 万行测量设施"所体现的取舍纪律：每个优化都带精确的 admission 谓词、可回滚开关和 bitwise/质量门。Z100 应该搬的是这个比例关系，不是内核。**

### 0.2 代码级盘点（这是 v2 相对 v1 最大的增量）

| 子系统 | 规模 | 可移植性 | 依据 |
|---|---|---|---|
| `flash-attention-v100/` | **15,472 行** CUDA/CUH | **0%** | HMMA fragment + SM70 ISA，gfx906 无矩阵核 |
| `csrc/attention/`（`sm70_v37`/`sm70_79t`/`sm70_grouped_long`） | **17,715 行**（19 文件） | **0%** | 同上；`bridge.cu` 用 `decode_byte` 位运算实现精确 E4M3→FP16 |
| `csrc/sm70_turbomind/` | **36,643 行**（71 文件） | **0%** | 来自 lmdeploy TurboMind，SM70 HMMA |
| `csrc/quantization/` | 36,476 行（110 文件） | ~10% | QPN/IQ/GGUF codebook 解码逻辑可参考，指令不可 |
| `vllm/v1/attention/backends/flash_attn_v100.py` | **9,870 行** | **30–40%** | 路由/admission/回退骨架可搬；内核调用不可 |
| `mixed_prefill.py`（core + worker） | **102 行** | **95%** | 纯 Python，无 CUDA 依赖 |
| `_balanced_prefill_chunk`（scheduler） | **~24 行** | **95%** | 纯 Python |
| `benchmarks/` | **449 文件 / 177,055 行**（Oct8 下限，此后仍在涨） | **方法论 100%** | 见 §3.2 |
| `tests/` | 1,801 个测试文件 | 方法论 | — |
| `VLLM_SM70/FLASH_V100/DFLASH_*` 开关 | **676 个唯一名** | — | `vllm/envs.py` |
| `ExecutionPolicy` + `sm70_policy.h` / `*_fields.inc`（**本次新增**） | 基类 **72 行** + 生成表 **69 行** | **100%** | 见 §2.6；Python 与 C++ 共用一张表 |

**结论**：可移植代码量 ≈ **150–250 行**（控制面）；不可移植代码量 ≈ **70,000 行**（数据面）。v1 假设"逐项移植算法"，代码看下来正确的做法是"**移植控制面 + 重建测量设施**"。

### 0.3 优先级（相对 v1 的变化）

| 项 | v1 判断 | v2 代码级判断 | 变化 |
|---|---|---|---|
| mixed-prefill 预算 | P1，中工作量 | **P0，102 行 + 1 行平台判断**；`MixedPrefillBudget` 是纯 EWMA 控制器 | **↑ 升到 P0** |
| balanced prefill chunks | 未提及（v1 只说 chunk 可能被切碎） | **P0，~25 行**；`VLLM_BALANCED_PREFILL_CHUNKS` 直接对应 Z100 的 880 对齐损失 | **新发现** |
| chunk 切分假设 | 推断"4096 token 被切成 ~4.6 段前向" | **证伪**：实际是每步预算 4096→3520（−14.1%），不是多次前向；512-token prompt 完全不受影响 | **更正** |
| FP8-KV 桥接 | P0 | P0 不变，但实现细节更清楚了：`bridge.cu` 是**精确展开**（不是 fp16 累加），Z100 的 LUT 语义与之兼容 | 细化 |
| 移植 1Cat scheduler | 隐含可抄 | **反向**：Z100 的 scheduler 比 1Cat 新（`use_internal_checkpoint`），只能抄"均衡分块"这个正交补丁 | **更正** |
| CUDA Graph | P1 | P1 不变 | — |
| **开关治理** | v1 未提及 | **新增 P0：先落 `ExecutionPolicy`（72 行）+ `.inc` 表，再加新开关** | **新发现** |

---

## 1. 这次同步到了什么

### 1.1 版本事实

| 项 | 值 |
|---|---|
| 本地 `$DATA_ROOT/1cat-vllm` | `c4f6245` — `[Kernel] Reuse M8 gate operands and TP4 norm packets (#1041)`，2026-10-08 |
| 远端 `origin/main` | `6a143197e` — `[Test] Cover compact hybrid page utilization (#1259)`，2026-10-23 |
| 落后 | **569 个提交**，其中 **246 个** 的标题命中 SM70/prefill/decode/memory/KV/quant/graph/perf 关键词 |
| 本地树可读性 | 只读（root 所有）；已 `cp -a` 到 `.refs/1cat-vllm` 获得可写副本 |

### 1.2 同步方式与代价（供后续复用）

```bash
# ① $DATA_ROOT 属 root，git 报 "dubious ownership"
#    （safe.directory 只登记了工作树 $DATA_ROOT/1cat-vllm，不含 .git）
git -c safe.directory=$DATA_ROOT/1cat-vllm/.git clone ...     # ✗ 无效
# ② 实际可行：复制成自己拥有的副本（0.7 s, 190 MB, 硬链接级速度）
cp -a $DATA_ROOT/1cat-vllm $WORKDIR/.refs/1cat-vllm
# ③ 完整增量 fetch 不可行：实测 ~9 KB/s（14 min 只走 7.8 MB，569 提交需数小时）
git fetch origin main                                        # ✗ 太慢，已中止
# ④ ★ 可用的做法：blobless + shallow —— 只要元数据（commit/tree），不要文件内容
git fetch --no-tags --filter=blob:none --depth=1 origin main # ✓ 数十秒完成
git diff-tree -r --name-status c4f6245 6a14319               # ✓ 987 个文件变化，纯树对象
# ⑤ 需要某个文件的最新内容时按需取（会触发单文件 blob 下载）
git show origin/main:<path>
#    或直接 curl raw.githubusercontent.com/1CatAI/1Cat-vLLM/main/<path>
```

> 注意 ④ 会把仓库变成 shallow（`origin/main` 只有 1 个提交的历史）。要恢复完整历史，
> 后续可在带宽充裕时跑 `git fetch --unshallow origin main`（数小时），或直接 `cp -a` 覆盖回本地树。
> **要拿到"最新工作树"而不是元数据，需要 `git checkout -f origin/main` 触发全部变更 blob 的按需下载——本报告没有做（预算不允许）。**

---

## 2. 569 个提交里真正新增的（v1 没覆盖的）

`git diff-tree c4f6245 6a14319` 实测：**987 个文件变化**（562 新增 / 425 修改），按目录分布
`vllm/` 418、`tests/` 314、`benchmarks/` 79、`csrc/` 65、`docs/` 52、`tools/` 29、`flash-attention-v100/` 8、`flash_qla/` 4。

按主题归类，只列与 Z100 的三个目标（prefill / decode / 显存）直接相关且有量化结果的。

### 2.1 prefill 调度：均衡分块 + 去掉 tail checkpoint

**提交** `caa27279a` `[Core] Prefill scheduling: balanced chunks, tail checkpoint, no host-KV zeroing`

- **`VLLM_BALANCED_PREFILL_CHUNKS=1`**（`vllm/v1/core/sched/scheduler.py:302`，`_balanced_prefill_chunk` 在 `:451`）：
  把长 prompt 切成 **n 个近似相等的块**，而不是 n-1 个满块 + 一个短尾块。
  原文理由："On models whose prefill step has a fixed floor …, the short tail step otherwise costs a full step."
- **`--no-mamba-tail-checkpoint`**（`vllm/config/cache.py:111` `mamba_tail_checkpoint: bool = True`）：
  > "end a prompt's last prefill chunk at its last cacheable boundary so a follow-up turn can reuse the whole prompt. When **False** and the rest of the prompt fits one step, prefill it in that step instead (one step fewer; the latest checkpoint is then the previous aligned chunk end)."
- **KV block zeroer 跳过 host-backed 层**：原文 "zeroing wrote every new block across PCIe (**~17 ms per 8K step**)"。
- 新增测试：`tests/v1/core/test_balanced_prefill_chunks.py`、**`tests/v1/core/test_mamba_align_chunk_split.py`**、`tests/v1/worker/test_kv_block_zeroer.py`。

> 对 Z100 的价值：`test_mamba_align_chunk_split.py` 的存在说明 1Cat 把"align 模式怎么切块"当成一等公民在测。Z100 的 `_mamba_block_aligned_split`（`vllm/v1/core/sched/scheduler.py:507`）**没有对应测试**，这是最值得先补的东西。

### 2.2 并发 MTP：c8 的验证成本被系统性拆解

**提交** `d67a10c4f` `[Perf] Lower concurrent MTP verification cost on SM70`（**36 个文件**）

块内量化结果（MAX_NUM_SEQS=8，verify ms/round）：

```text
c1 4K 18.96 ms, 32K 19.56 ms
c2 4K 26.2,  32K 30.9
c4 4K 36.4,  32K 45.2
c8 4K 62.9 ms
```

机制（每条都"bitwise equal to the per-request result"）：
- QSA shared-key indexer 一次 launch 为多个请求打分；
- router / shared expert / PLE spec conv 合并成 **5..40 行一次 launch**；
- 精确 gated RMSNorm **覆盖 64 token**（原来是 c4 下的 **9-kernel torch 链**）；
- routed gate/up 对每个 expert 在它的若干路由里只解码一次（IQ3_S/IQ4_NL/IQ4_XS，从 10 行起）；
- QSA fused side projection **每次调用选路**（decode compile 范围跨 1..40 行，曾把 unfused 分支冻住）；
- **expert-cache 命中去掉 system-scope fence**。

> 对 Z100 的价值：`c8 62.9 ms/round` 是"一个成熟 SM70 栈在 8 路并发下"的参考值；Z100 现在 c8 的 ITL p99 是 **5014 ms**。差距不在单个 kernel，在"每请求一次 launch"的组织方式。**"9-kernel torch 链 → 1 个精确 kernel"这一条 Z100 可以直接对标**：它已经有 `_C::gemma_fused_add_rms_norm_gfx906`（33.2 ms，0.77%）。

### 2.3 显存：一套可复用的"借，不要拷"模式（~400 MiB）

**提交** `2efd1bc0d` `[Memory] Remove duplicated persistent buffers on the 9683 Flash-Next profile`

| 手法 | 节省 |
|---|---|
| 按**实际准入的投影**分配 GGUF affine BLAS scratch（12.5 MiB），而不是"任何模型的最大校准形状"（170 MiB） | **157 MiB** |
| speculator **原地读** target 的持久 MTP hidden buffer，不再拷进第二个 8192×10240 buffer | **160 MiB** |
| MTP 层**借用** target 串行复用的 QSA top-k workspace | **64 MiB** |
| MTP 借用**相同的 RoPE 表** | **32 MiB** |
| MTP Q8 MoE prefill chunk 2048 → 1024（其临时量正落在 prefill 峰值） | 峰值下降 |

同主题还有一批（都带"bound / share / reuse"字样，可作审计清单）：
`1cb8574d7` KV 历史 attention scratch 跨 prefill/decode 共享、
`996e3a161` prefill staging slot 在 attention 后复用、
`9b924d783` host QSA history 跨 TP rank 共享、
`27c3fb0ff` QSA RoPE 表按 `max_model_len` 设界、
`174e21f38` TurboQuant 续算 SDPA 工作内存设界。

> 对 Z100 的价值：Z100 只有 **1.8–3.0 GiB** 余量，而这套模式的单点收益是 32–160 MiB 级、可叠加。**审计问题清单**：①有没有按"任何模型的最大形状"分配的 scratch？②有没有"拷贝一份给 speculator/draft"的 buffer？③有没有两处存同一张表（RoPE/LUT）？

### 2.4 正确性陷阱：混合批 + 投机解码会静默算错

**提交** `705aca212` `[Bugfix] Keep GDN replay correct when decodes share a step with prefill`

> "the mixed-batch draft convolution took its round length from that width and processed only the first of the five rows of each request, corrupting tool-call arguments whenever a decode ran beside a long prefill (**20/44 correct; now 44/44**)."

**这是本报告里最重要的风险项**：1Cat 在做成"prefill 与 decode 共处一个 step"之后，踩到一个**不影响速度、只损坏输出**的 bug（工具调用参数被截断），靠 44 例端到端才抓到。

> 对 Z100：Z100 目前 `--enforce-eager` 且不做 mixed step，所以还没暴露；**一旦上 §4.1 的 mixed-prefill 预算，就会进入同一类 bug 的暴露面**。必须先建"混合 step 下的 44 例端到端"这类门禁。

### 2.5 可观测性：round-cost ledger

**提交** `b32072ac9` `[Profile] Add an opt-in SM70 round-cost ledger (#1254)`

> 记录实际路由、expert 复用、已加载存储、选中的稀疏 KV 操作数，使"带宽/指令账"可与 profile 到的服务时间对照。全部挂在 `kernel_config.sm70_round_cost_diagnostics` 下，**默认 False**：不开启就不挂 recorder、不加任何 kernel，**因为诊断 kernel 会使延迟测量失效**。

> 对 Z100：这正是 Z100 当前最缺的东西——`KERNEL_REPORT_20261024` 自己写了 "shapes 未记录 ⇒ 无法归因"。1Cat 的做法给出了正确形态：**开一个独立的诊断通道，而不是往热路径里塞计时**。

### 2.6 控制面被工程化了：typed policy + X-macro ABI（**本次同步最重要的发现**）

`git diff-tree c4f6245 6a14319` 显示 **987 个文件变化**（562 新增 / 425 修改），其中 `vllm/config/` 新增了 **19 个文件**，同时 `csrc/` 新增了一套与之对应的 C++ 策略层：

**Python 侧**
```
vllm/config/execution_policy.py          # 通用基类
vllm/config/execution_policy_base.py     # ExecutionPolicy / DeferredExecutionPolicy
vllm/config/policy_defaults.py           # 523 行默认值表
vllm/config/sm70_runtime.py              # 483 行
vllm/config/sm70_native.py  sm70_sparse.py  sm70_moe.py  sm70_triton_attention.py
vllm/config/flash_v100.py   gdn.py  gdn_projection.py  gdn_schedule.py  gdn_state.py
vllm/config/diagnostic_dump.py  diagnostic_sampling.py  collective.py  legacy_inputs.py
```

**C++ 侧**
```
csrc/sm70_policy.h                 # 315 行
csrc/sm70_policy_fields.inc        # 69 行，X-macro，头注：Generated from
                                   #   vllm.config.sm70_native.NATIVE_FIELDS. Order is ABI.
csrc/sm70_runtime.h  sm70_runtime_state.h
csrc/custom_all_reduce_policy.h + custom_all_reduce_policy_fields.inc
csrc/sm70_marlin_policy.h
```

**核心机制**（`execution_policy_base.py`）：

| 字段/方法 | 语义 |
|---|---|
| `sources: dict[str,str]` | 每个字段的初始化来源；**不参与计算哈希** |
| `active: bool` | 该特性是否可能影响本引擎的计算 |
| `hash_fields` | "Effective computation fields; **resource and unused feature choices are omitted**" |
| `compute_hash()` | 只对 `hash_fields`（且 `active`）做哈希 |
| `DeferredExecutionPolicy` | 初始化时捕获全部输入，**解析错误延迟到字段被消费时**才抛 |

**C++ 侧**用 X-macro 从同一张表生成 `PolicyField` 枚举 + `policy_names[]` + `calculation_fields[]`，并且
```cpp
// Legacy direct callers have no policy argument. Capture compatibility inputs
// once, without mutating the process environment.
inline const std::vector<std::string>& legacy_policy() {
  static const auto values = [] { ... std::getenv(name) ... }();   // 只读一次
  return values;
}
```
`sm70_policy_fields.inc` 里 **61 个字段标 `calculation=true`，6 个标 `false`**。

> **为什么这条对 Z100 最重要**：Z100 现在有 56 个 `VLLM_DCU_*` 开关，且已经在用 `torch.compile`（路线 A 的调试核心就是"编译期看不到真实 M，物化路由静默失效"）。一旦再上 CUDA Graph，**"改一个只影响资源的开关就作废整张图/整个 compile cache"会成为主要摩擦**。
> 1Cat 的答案是把开关分成"影响计算"与"只影响资源"两类，**用一张表同时生成 Python 校验和 C++ 枚举（顺序即 ABI）**。
> **建议：Z100 在加第 57 个开关之前，先落地这个 72 行的 `ExecutionPolicy` 基类 + 一张 `.inc` 表。** 这是本报告里性价比最高的"非性能"改动。

### 2.7 其它

- `50d361453` QSA 热页淘汰策略：128K 每轮 miss 35.1K → 17.9K，**round 25.9 → 23.1 ms**（Z100 无 host-KV，不适用）。
- `46f5c8c19` 把编码后的 target KV 留在 Flash-Next 热缓存。
- `416b42e7d` 长 prompt 超过 prefill stage buffer 时按窗口分阶段（原为逐行 host 读，**~260 tok/s**）。
- `174e21f38` TurboQuant 续算 SDPA 工作内存设界（`#1257`）。
- 新增内核（`csrc/` 共 65 个文件变化）：`qsa_indexer_sm70.cu`、`qsa_device_history_sm70.cu`、`qsa_prep_sm70.cu`、`sm70_router.cu`、`gdn_verify_sm70.cu`、`hcx_sm70.cu`、`gguf_moe_gu_sm70.cu`、`gguf_prefill_transcode_sm70.cu`、`gemm_runtime.cpp` —— 全部 SM70/HMMA，**Z100 不可移植**，列在此处仅为说明这 3 周的重心仍在数据面。

---

## 3. 代码级：1Cat 的钱花在哪

### 3.1 控制面 vs 数据面（这是本报告的核心论证）

```text
数据面（ISA 锁定，gfx906 可用性 0）
  flash-attention-v100          15,472 行
  csrc/attention                17,715 行
  csrc/sm70_turbomind           36,643 行
  ────────────────────────────────────────
  小计                         ~69,830 行

控制面（平台无关，可直接移植）
  vllm/v1/core/sched/mixed_prefill.py      49 行
  vllm/v1/worker/mixed_prefill.py          53 行
  scheduler 侧预留逻辑                    ~30 行
  _balanced_prefill_chunk                 ~24 行
  ────────────────────────────────────────
  小计                          ~154 行

测量设施（方法论可移植）
  benchmarks/                449 文件 / 177,055 行（Oct8 下限）
  tests/                     1,801 个测试文件
```

**比例是 1 : 450。** 也就是说，1Cat 每写 1 行控制逻辑，背后有 450 行内核和 1000+ 行基准来保证它不被误用。
这解释了为什么 v1 从文档里"读到的算法"看起来简单，但 Z100 直接照做多半会失败——**缺的不是算法，是让算法安全上线的那些谓词和测量**。

### 3.2 676 个开关说明了什么

`vllm/envs.py` 里有 **676 个** `VLLM_SM70/FLASH_V100/DFLASH_*` 唯一开关。这不是技术债，而是这个项目的工作方式：**每个被测量证明有效的路由都有一个可回滚的名字**。例如：

| 开关 | 作用 | 默认 |
|---|---|---|
| `VLLM_FLASH_V100_FA2_D256_PREFILL` | D256 split-D prefill 路由 | 1 |
| `VLLM_FLASH_V100_PREFILL_GATHER_DENSE` | 长前缀 gather→dense | 1（门槛 `Q>=4096, KV>=8192`）|
| `VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3` | 三路 KV 分区 | 1（`KV>=32768`）|
| `VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS` | decode 分区动态选择 | 1 |
| `VLLM_FLASH_V100_E4M3_GROUPED_FP32` | E4M3 grouped + FP32 partial | 1 |
| `VLLM_DFLASH2_LOOKUP_ADAPTIVE` | lookup 增广自适应 q8/q16 | 1 |
| `VLLM_BALANCED_PREFILL_CHUNKS` | 均衡分块 | **0** |
| `mixed_prefill_step_latency_ms` | 混合 step 目标延迟 | **250.0** |

**对 Z100 的启示**：Z100 现在的开关是 56 个 `VLLM_DCU_*`，密度只有 1Cat 的 1/12。这不是说需要加开关，而是说**每个开关背后应有一次测量**；Z100 的 56 个里，`.r3/STATUS.md` 显示已有 9 个工作包被证伪或关闭（WP0/WP2/WP3/WP4/WP5/WP7），这个比例其实健康。

---

## 4. 逐项可移植方案（代码级）

### 4.1 `MixedPrefillBudget` — 102 行，Z100 的 c8 尾延迟直接对口

**1Cat 实现**（已核对 `origin/main`，与 Oct8 相比只多了 `min_tokens` 下限）：

```python
# vllm/v1/core/sched/mixed_prefill.py  (56 行, main)
class MixedPrefillBudget:
    def __init__(self, max_tokens, target_ms, min_tokens=0):
        self.tokens = max(min(512, max_tokens), self.min_tokens)
        self.ms_per_token = None
    def update(self, sample):
        if (... sample.prefill_tokens <= 0 or sample.decode_tokens <= 0 ...): return
        # 尾块由固定 decode 开销主导，不能教控制器"整块也一样贵"
        if sample.prefill_tokens < sample.budget_tokens * 0.75: return
        cost = sample.elapsed_ms / sample.prefill_tokens
        self.ms_per_token = cost if None else 0.5*self.ms_per_token + 0.5*cost
        desired = self.target_ms / self.ms_per_token
        desired = min(desired, self.tokens * 1.25)      # 快设备慢涨
        quantum = min(16, self.max_tokens)
        self.tokens = max(quantum, min(self.max_tokens, int(desired)//quantum*quantum))
```

配套的 `MixedPrefillTimer`（53 行）用 **CUDA event 非阻塞回读**，最多排队 8 个样本，永不 `synchronize` 当前 step。

调度器侧（`limit_prefill`：本地 c4f6245 树 `:575–604`，main `:627` 起）：
```python
resident_tokens = {纯 decode 且有输出且已过 next_decode_eligible_step 的请求: 待算 token}
control_mixed_prefill = (enabled and resident_tokens and latency>0
                         and chunked_prefill and not disable_chunked_mm_input
                         and pp_size==1 and not is_encoder_decoder)
def limit_prefill(req_id, tokens):
    return max(0, min(tokens,
                      self.mixed_prefill_budget.tokens - used,      # 预算
                      token_budget - reserve))                       # 给常驻 decode 留位
```

**Z100 移植清单**
1. 复制 `mixed_prefill.py`（core + worker）两份文件，**不需要改逻辑**。
2. `scheduler.py` 加 `MixedPrefillBudget` 初始化 + `limit_prefill`，插入点对应 Z100 现有 `_mamba_block_aligned_split` 的两个调用点（`:837`、`:1286`）。
3. **改一行平台门**：1Cat 是 `current_platform.device_type == "cuda"`；Z100 要放开到 `rocm`，并确认 `torch.cuda.Event` 在 DTK 下可用（HIP 下 `torch.cuda` 是兼容层，通常可用，**需实测**）。
4. 新增 CLI：`--mixed-prefill-step-latency-ms`（默认 250）与 `--mixed-prefill-min-tokens`。
5. **必须同时建 §2.4 的混合批端到端门禁**。

**预期**：1Cat 常驻吞吐 **6.88 → 30.34 tok/s（4.41×）**、最长更新间隔 **2.873 → 0.310 s（−89%）**；代价是新请求 TTFT +26%/+72%。Z100 的 WP5 已经用 37 组 CLI 证明"CLI 层无解"，这正是缺口。

**风险**：Z100 的 `--max-num-seqs 4` 与 1Cat 的 16/32 不同，`resident_tokens` 的规模更小，控制器收敛可能更慢（种子 512 token）。建议 `min_tokens=512`。

### 4.2 `_balanced_prefill_chunk` — ~24 行，直接对冲 880 对齐损失

**1Cat 实现**（`scheduler.py:451`，main）：
```python
def _balanced_prefill_chunk(self, request, num_computed_tokens, num_new_tokens):
    if not self.balanced_prefill_chunks or num_new_tokens <= 0: return num_new_tokens
    remaining = max(request.num_prompt_tokens, request.num_tokens-1) - num_computed_tokens
    if remaining <= num_new_tokens: return num_new_tokens
    unit = (self.mamba_state_block_size or self.cache_config.block_size
            if self.need_mamba_block_aligned_split else self.cache_config.block_size)
    chunks = cdiv(remaining, num_new_tokens)
    balanced = cdiv(cdiv(remaining, chunks), unit) * unit
    return min(num_new_tokens, balanced)
```

**Z100 的对应问题**：`_mamba_block_aligned_split`（`:507`）里有
```python
if end < prefill_end:
    aligned_end = end // block_size * block_size
    if aligned_end > start or block_size <= max_prefill_tokens:
        end = aligned_end
```
`block_size=880 <= max_prefill_tokens=4096` 恒真 ⇒ **每一步的 chunk 都被 floor 到 880 的倍数**：
`4096 → 3520`，**每步丢 576 token = 14.06% 的 prefill 预算**（长 prompt 约多 16% 的步数）。

> ⚠️ **这更正了 v1**：v1 写"一次 4096-token prefill 可能被切成 ~4.6 段整层前向"——**不成立**。实际是**单步预算被截短**，不是单步内跑多次前向。而且 512-token prompt（`end >= prefill_end`）**完全不受影响**，所以 **2722 ms 的 512-TTFT 与 align 无关**，是纯粹的 step 效率问题。

**Z100 移植清单**：把 `_balanced_prefill_chunk` 作为 `num_new_tokens` 的预处理器插到两个 splitter 调用点之前；`VLLM_BALANCED_PREFILL_CHUNKS` 默认 0（**先 A/B，不要默认开**）；补 `tests/v1/core/test_balanced_prefill_chunks.py` 与 `test_mamba_align_chunk_split.py` 的 Z100 版。

**预期**：把 14.06% 的预算损失摊平，长 prompt prefill 步数约 −12~14%。对 32k 的 93.5 s 量级 ⇒ **省 10 s 量级**（推断，需实测）。

### 4.3 FP8-KV 桥接 — 结构可抄，实现不可抄

**1Cat 实现**：`csrc/attention/sm70_v37/bridge.cu`（CUDA）。核心是 `decode_byte<FORMAT>()` ：
它**不经过 fp16 累加器**，而是直接构造 FP16 位模式：
```cpp
const uint16_t sign = (raw & 0x80u) << 8;
const uint16_t magnitude = raw & 0x7fu;
if (magnitude == 0x7fu)      bits = 0x7e00u;                        // NaN
else if (magnitude < 8)      bits = __half_as_ushort(__float2half_rn(float(magnitude)*0x1p-9f));
else                         bits = (magnitude << 7) + 0x2000u;      // 精确位拼接
return __ushort_as_half(sign | bits);
```
`FORMAT>=3` 的 `decode_pair` 更进一步：整对用**幂等二的位拼接**得到"值 ÷256"的 half2，再 `__hmul2(..., 256.f)` 还原，并显式处理 NaN。

**为什么重要**：这正是 Z100 L4 设计里担心的"精度"问题的答案——**E4M3→FP16 是精确的（对有限编码），只要不做 fp16 累加**。Z100 的 `gfx906_emu_common.cuh` 里已有一套 256 项 LDS LUT 的 fp8→fp16 精确展开（`ds_read_u16`），语义等价，**可以直接复用**，不必移植 `bridge.cu`。

**Z100 移植清单**（沿用 v1 的 P0.2，但实现路径更明确）：
1. 热路径永远是 FP16：`gather + dequant → fp16 workspace`，再调现有 FP16 Triton 路径。
2. dequant 用**已有的 LUT**（`_device_lut` / `fp8_dequant_to_fp16_out`），不要新写位运算。
3. scale 必须 per-token-per-head + `--calculate-kv-scales`。
4. 门禁：PPL/KL ≤1%、1000 步无 NaN、池容量 ×1.8 或 ITL −20%、**prefill 不回退 >3%**。

### 4.4 从 1Cat 学"不要做什么"（代码里的负结果）

这些是 1Cat 自己测过并明确否决的，Z100 可以跳过：

| 被否方案 | 证据 | 位置 |
|---|---|---|
| WMMA 融合反量化 + GEMM | 512–8192 行上**慢约 4 倍** | `sm70_memory_arena.md` / `mixed_prefill_latency` 记录 |
| C1 下 SiLU/down 融合 | 算子级 7.77→6.59 µs，但 e2e 11.041 vs 11.063 ms/token 无收益；**MBPP-16 从 628 token 变 4096 token** 且重复末答 11 次 | `sm70_qwen38_step0_kernel_budget` |
| prefill score block 降到 8K | 省 1.50 GiB 但延迟 **+2.0–2.26%** | `sm70_memory_workspace_tuning` |
| `block 1648 + grid 8192` | prefill 最快（3551 tok/s）但 1648 不整除 8192 ⇒ **前缀缓存静默归零**（`hits_total=0`，同一 232K prompt 86.5 s vs 命中 1 s） | `sm70_mamba_state_grid_decoupling` |
| ngram 混合**默认开** | 低命中时每轮多付 0.1–0.3 ms | `sm70_dflash2_ngram_hybrid` |
| 无条件 dense 边界回退 | 降 32K 投机解码吞吐 | `sm70_flash_next_prefix_prefill` |
| 诊断 kernel 常开 | **会使延迟测量失效** | `b32072ac9` |

---

## 5. 对 v1 的三处更正

| # | v1 的说法 | 代码事实 | 影响 |
|---|---|---|---|
| **1** | "一次 4096-token prefill 可能被切成 ~4.6 段整层前向" | `_mamba_block_aligned_split` 把**单步预算** floor 到 880 倍数（4096→3520，−14.1%），**不在单步内跑多次前向**；512-token prompt 完全不受影响 | 长 prompt 收益预估从"40~110%"下调到"**12~14% 步数**"；512-TTFT 的归因必须完全转向 step 效率（§6.1） |
| **2** | "抄 1Cat 的 mamba/chunk 解耦" | **方向反了**：Z100 的 `_mamba_block_aligned_split` 是**更新的 upstream**（有 `use_internal_checkpoint`、`get_mamba_prefill_checkpoint_position`、`hash_block_size`、`mamba_partial_cache_hit`），1Cat 的是旧 fork | 只能抄**正交补丁**（`_balanced_prefill_chunk`、`mixed_prefill_min_tokens`），不能抄 splitter 本体 |
| **3** | "1Cat 的 scheduler 经验可以直接借鉴" | 1Cat 的 DFlash2 策略层有 18 个默认开的 verifier 开关 + 676 个 SM70 开关，**强绑定其 7 万行内核的准入谓词**；脱离内核后这些开关没有意义 | 移植时只搬控制面，**开关名一个都不要照抄** |

v1 中**仍然成立**的判断：FP8-KV 必须走桥接（`bridge.cu` 的精确展开语义证实了这一点）、decode 是"组合方式"问题、不要碰 HMMA、不要引入主机内存依赖。

---

## 6. 更新后的路线图

### P0（1 周内可完成，全部是控制面）

**P0.1 `MixedPrefillBudget` + `MixedPrefillTimer`（3 天）**
- 落地：`vllm/v1/core/sched/mixed_prefill.py`、`vllm/v1/worker/mixed_prefill.py`、`scheduler.py` 两个插入点、`--mixed-prefill-step-latency-ms`。
- 先决：**Z100 侧先建混合 step 端到端门禁**（对标 1Cat 的 44 例：`20/44 → 44/44`）。没有这个门禁不要开。
- 判据：c8 ITL p99 从 **5014 ms** 降到 1.5 s 量级；常驻吞吐 ≥2×；新请求 TTFT 退化记录在案。

**P0.2 `_balanced_prefill_chunk`（1 天）**
- 默认 `VLLM_BALANCED_PREFILL_CHUNKS=0`，先 A/B。
- 判据：32k prefill 步数 −12~14%；512/709 TTFT 不变（因为它只影响 `end < prefill_end` 的中间块）。
- 顺带补 `test_mamba_align_chunk_split` 的 Z100 版本（**当前无任何 align 切块测试**）。

**P0.3 prefill step 效率测量（2 天，v1 的 P0.1 强化版）**
- 开 `torch_profiler_record_shapes: true`，但要**照 1Cat 的做法**：诊断与计时分离（`sm70_round_cost_diagnostics` 形态），不要往热路径塞 event。
- 三个必答问题（**已有代码级预期**）：
  1. `aten::mm` 走 fp16（16.5–17.7 TF/s）还是 bf16（3.42 TF/s）？——若是 bf16，这是 5 倍级问题。
  2. 512-token（单块，无 align 影响）的 2.72 s 如何分解成 GEMM/GDN/attention/空转？
  3. GDN chunked prefill `fused_sigmoid_gating_delta_rule_gfx906_prefill` 14.08 ms/次 × 48 层的 M 是多少？

**P0.4 开关治理（2 天，非性能但决定后续速度）**
- 抄 `vllm/config/execution_policy_base.py`（72 行）：`sources` / `active` / `hash_fields` / `compute_hash()` / `DeferredExecutionPolicy`。
- 建一张 Z100 的 `.inc` 表：把现有 56 个 `VLLM_DCU_*` 按"是否影响计算"分类（例如 `FP16_PREFILL` 影响数值 → `true`；`FP16_WS_BYTES` 只影响显存 → `false`）。
- 用 X-macro 从同一张表生成 C++ 枚举（顺序即 ABI），并在 C++ 侧把 `getenv` 收敛到一次静态读取。
- **判据**：改动一个 `calculation=false` 的开关后，`torch.compile` 缓存与（未来的）CUDA Graph 仍然命中。

### P1（2–4 周，数据面）

**P1.1 CUDA Graph 路线 A 重新上生产**（v1 同）：Z100 已实测 decode −7~9%，compile 下 512-TTFT 3.35→1.35 s。
**P1.2 单体 decode GEMM / int4 权重**（v1 同）：目标把 43.8% 的 `aten::mm` 从 16–23 GB/s 拉到 fp8 GEMV 的 176 GB/s 量级。
**P1.3 显存"借而非拷"审计**（新增，成本低）：按 §2.3 的清单查 Z100 的 ①最大形状 scratch ②draft 拷贝 ③重复表。

### P2
lookup/ngram 增广（默认关 + 自适应）、compile cache 默认开、workspace 去重。

---

## 7. 陷阱清单（按踩到概率排序）

1. **混合 step + 投机解码会静默损坏输出**（1Cat 20/44 → 44/44）。先建门禁再上 P0.1。
2. **前缀缓存可能静默归零**（1Cat 的 1648+8192 反例：`hits_total=0`，同一 prompt 86.5 s vs 1 s）。任何 block/grid 改动后必须验 `prefix_cache_hits_total`。
3. **诊断 kernel 会使延迟测量失效**（1Cat 因此默认关）。测量与计时通道分离。
4. **standalone 微基准不能预测 e2e**：Z100 已经两次被反转（cfg3 +6% → e2e +13~19%；custom AR +21~53% → e2e 变慢），1Cat 也有（QPN8 未默认开、C1 SiLU/down 融合无 e2e 收益）。
5. **"数学等价 ≠ 性能等价"**：1Cat 的 24 种 `m8n8k4` K4 排列里只有 `[0,1,2,3]` 与 `m16n16k16` 逐位一致；Z100 的 `VLLM_DCU_FP8_STREAM_MAX_M` 也是更快但破坏 bit-exact。

---

## 8. 本次分析的边界

1. **版本参照系已到 `6a143197e`（Oct 23），但磁盘上的文件内容仍停在 `c4f6245`（Oct 8）**。
   - 已有：精确的 987 文件变化清单（`git diff-tree`）、569 条提交元数据、10 个关键文件的最新全文。
   - 没有：最新工作树。所有 grep / LOC 统计仍在 Oct 8 树上做，**§3 的 LOC 因此是下限**；对未单独拉取的文件，"最新版本是否仍这样写"无法确认。
2. **569 个提交里我只枚举了最新 500 条的标题**（API 分页 5×100），并精读了 10 个代表提交的完整 message 与文件列表。**没有逐个提交读 patch**。
3. **按需拉了 10 个最新文件**（`core/sched/mixed_prefill.py`、`worker/mixed_prefill.py`、`core/sched/scheduler.py`、`config/cache.py`、`config/scheduler.py`、`config/execution_policy_base.py`、`config/policy_defaults.py`、`config/sm70_runtime.py`、`csrc/sm70_policy.h`、`csrc/sm70_policy_fields.inc`）。**其余 977 个有变化的文件没有逐一核对内容**；凡引用 Oct8 树的行号都已标注来源。
4. **没有编译、没有跑任何 1Cat 代码**；所有 1Cat 性能数字仍是其仓库自述。
5. **Z100 侧没有任何新测量**；§2/§6 里的 Z100 数字全部来自既有文档。

---

## 9. 参考

**1Cat-vLLM（`origin/main` = `6a143197e`）**
- **策略层（新增）**：`vllm/config/execution_policy_base.py`（72 行）、`vllm/config/policy_defaults.py`（523 行）、`vllm/config/sm70_runtime.py`（483 行）、`csrc/sm70_policy.h`（315 行）、`csrc/sm70_policy_fields.inc`（69 行，61 个 `calculation=true`）、`csrc/custom_all_reduce_policy*.{h,inc}`、`csrc/sm70_runtime.h`
- 控制面：`vllm/v1/core/sched/mixed_prefill.py`（56 行）、`vllm/v1/worker/mixed_prefill.py`（53 行）、`vllm/v1/core/sched/scheduler.py`（`_balanced_prefill_chunk:451`、`_mamba_block_aligned_split:478`、`limit_prefill:627`）
- 配置：`vllm/config/scheduler.py:94`（`mixed_prefill_step_latency_ms=250.0`）、`:99`（`mixed_prefill_min_tokens`）、`vllm/config/cache.py:111`（`mamba_tail_checkpoint`）
- 数据面：`csrc/attention/sm70_v37/bridge.cu`、`csrc/attention/sm70_79t/`、`csrc/sm70_turbomind/`、`flash-attention-v100/`
- 关键提交：`caa27279a`（均衡分块）、`d67a10c4f`（并发 MTP）、`2efd1bc0d`（显存借用）、`705aca212`（GDN replay 正确性）、`b32072ac9`（round-cost ledger）、`50d361453`（QSA 热页）、`416b42e7d`（窗口化 staging）
- 仓库：<https://github.com/1CatAI/1Cat-vLLM>

**Z100 本仓库**
- v1 报告：`z100-1cat-transfer-research.md`（已移除；可按 git 历史恢复）
- [`vllm/v1/core/sched/scheduler.py:507`](../../../../vllm/v1/core/sched/scheduler.py)（`_mamba_block_aligned_split`）
- [`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py`](../../../../vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py)（物化路由与 `_ROLE_MIN_M`）
- [`csrc/rocm/gfx906_emu_common.cuh`](../../../../csrc/rocm/gfx906_emu_common.cuh)（fp8/bf16 精确模拟原语）
- [`docs/result/z100-rocm101-perf-verdict.md`](../../../result/z100-rocm101-perf-verdict.md)、[`docs/design/z100/archive/z100-s24-l4-kv-quant-design.md`](../archive/z100-s24-l4-kv-quant-design.md)、[`KERNEL_REPORT_20261024-053333-E2C.md`](../../../../KERNEL_REPORT_20261024-053333-E2C.md)
