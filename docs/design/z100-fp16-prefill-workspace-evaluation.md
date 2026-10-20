# Z100 prefill fp16 权重物化：评估（决定 2）

> **已合并**：与决策 1 的统一施工图见
> [`z100-w8a16-prefill-fp16-construction.md`](z100-w8a16-prefill-fp16-construction.md)（施工以该文件为准）；
> 本文件保留为决策 2 的论证与数据来源。
>
> 状态：**仅评估，不跑卡**（用户指示：卡在别的工程）。
> 结论先行：**采用「按层 scratch」，否决「常驻全量 fp16 副本」**；rocBLAS 路线作为 W8A16
> 之后的 prefill 可选加速，`min_m` 与可用性必须等卡空时实测后再定。
> 相关文档：[`z100-fp8-w8a16-implementation.md`](z100-fp8-w8a16-implementation.md)、
> [`z100-fp8-emu-fp16-conversion-plan.md`](z100-fp8-emu-fp16-conversion-plan.md)。

---

## 1. 问题定义

prefill（M 大）时，本平台的自研 GEMM 是「fp8 权重 → LUT 加宽 → `v_dot2_f32_f16`」，加宽与
LDS staging 占了不等于零的指令预算。另一条路是：**把当前层权重一次性解量化成 fp16**，然后交给
rocBLAS/Tensile 的 fp16 GEMM（fp32 累加）。上游同类实现（V100）就是这么做的
（`1cat-vllm: sm70_fp8.py:364-413`、`awq_sm70_gemm.cu:3055-3145`）。

需要决定的是 fp16 权重副本的**生命周期**：

| 方案 | 描述 |
|---|---|
| **A 常驻** | 模型加载后把全部 fp16 权重物化一次，长期驻显存；prefill 直接读 |
| **B 按层 scratch** | 一块进程级 fp16 buffer（≈ 最大层的 `N·K·2 B`），每层 prefill 前物化，用完即复用 |
| **C 不物化** | 保持 fp8 权重，自研 no-LUT GEMM（内核内 dequant，无额外显存/流量） |
| **D 不引入** | prefill 继续用 W8A16 emu GEMM（R1 落地后的默认路径），本议题只留档 |

---

## 2. 前提变化（W8A16 决定之后）

选择做 W8A16（激活不再量化）之后，这条路线**变简单了**：

- A 侧本来就是 fp16 → 物化后可以直接喂 `torch.mm`，不需要 A 量化 kernel，也不需要 `a_s`；
- 输出本来就是 fp16 → epilogue 无 cast；
- 唯一要处理的只有 W：`W16 = fp16(Wq · w_s)`（每次一层、一次 kernel）。

也就是说：**W8A16 是「物化 + 库 GEMM」的前置条件，而不是替代品**。

---

## 3. 约束（全部来自仓库既有事实，不依赖新测量）

| 项 | 值 | 出处 |
|---|---|---|
| 单卡显存 | 32 GiB | `.s24/z100-constraints.md` |
| 生产静态占用 | 30.9 GiB（94.5%），余量 1.8 GiB | `z100-longctx-vram-optimization-plan.md:38,75` |
| 模型加载（含 DFlash2 draft） | 9.04 GiB/卡 | 同上 `:73` |
| ├ 主模型 fp8 分片（TP=4，27 GB 级） | ≈ 6.29 GiB | 27 GB ÷ 4 |
| └ 其余（draft/元数据等） | ≈ 2.75 GiB | 差值 |
| KV 池 | 15.54 GiB（748,982 token）/ 16.21 GiB（MBT=4096） | 同上 `:38,74,225` |
| KV 开销 | **22.34 KB / token / 卡** | 同上 `:74` |
| 层数 × 投影数 | 64 层 × 4 投影 = 256 次 GEMM/步 | `z100-fp8-emu-optimization-result.md`（256 次/步） |
| 生产形状（N,K，TP=4 每卡） | gate_up 8704×5120、down 5120×4352、qkv 3584×5120、o 1536×5120 | 同上 |
| 最大层权重字节 | 8704×5120 = 44.6 MB（fp8）/ 89.2 MB（fp16） | 由形状推出 |

---

## 4. 定量模型

### 4.1 显存

```
A（常驻）: 2 × 6.29 GiB = 12.58 GiB  + 89 MB scratch      ← 必须从 KV 池里挤
B（按层）: 89 MB（固定）                                    ← 可忽略
C        : 0
```

**A 的代价换算成 KV**：12.6 GiB ÷ 22.34 KB/token ≈ **59 万 token** 的 KV 容量。
即 KV 池 16.21 GiB（约 78 万 token）→ **≈ 3.6 GiB / 约 17 万 token**。

对照需求：`c=16 × 8k = 12.8 万 token` 还能放下，但 **`c=16 × 32k = 51 万 token`、
64k/128k 长上下文 soak 直接放不下**——而长上下文正是本项目的核心场景
（`z100-longctx-vram-optimization-plan.md:97,225` 的 64k/128k 验收）。
TP=2 更不可能：fp8 分片 12.58 GiB + fp16 副本 25.16 GiB > 32 GiB，连加载都不成立。

### 4.2 流量（每 prefill pass，TP=4 每卡，64 层）

```
每层 fp8 权重 44.6 MB，fp16 89.2 MB

A: 每层 89.2（GEMM 读 fp16）                        → 64 层 = 5.71 GB
B: 每层 44.6（读 fp8）+ 89.2（写 fp16）+ 89.2（GEMM 读）
                                                   → 64 层 = 14.3 GB
C: 每层 44.6（GEMM 内读 fp8）                       → 64 层 = 2.86 GB
```

在 ~800 GB/s 量级下折算：A ≈ 7.1 ms / pass，B ≈ 17.9 ms / pass，**B−A ≈ 10.8 ms / pass**。
（chunked prefill 每个 8192 token 的 chunk 都会重跑一遍全部层，所以这是**每个 chunk**的成本。）

### 4.3 与 prefill 计算量对比（决定 B 是否可接受）

每卡 FLOPs ≈ 2 × 27e9 × M ÷ 4（TP=4）：

| M（每 chunk） | 每卡 TFLOP | 若 8–10 TF/s 的耗时 | B 的额外流量占比 |
|---|---|---|---|
| 8192 | ~110 | 11–14 s | ≈ 0.13% |
| 1760 | ~23.8 | 2.4–3.0 s | ≈ 0.6% |
| 1024 | ~13.8 | 1.4–1.7 s | ≈ 1.1% |
| 256 | ~3.5 | 0.35–0.44 s | ≈ 4% |

**结论**：只要 `min_m ≥ 1024`，B 的重复物化成本 ≤ ~1%；A 省下的 10.8 ms 换 12.58 GiB，
在任何 M 下都不划算。

---

## 5. 决策矩阵

| 维度 | A 常驻 | **B 按层 scratch** | C 不物化 | D 不引入 |
|---|---|---|---|---|
| 显存 | −12.58 GiB（KV 池 78 万→17 万 token） | −89 MB | 0 | 0 |
| 每 pass 流量 | 5.71 GB | 14.3 GB（+10.8 ms） | 2.86 GB | 2.86 GB |
| prefill 提速潜力 | 高（取决于 Tensile） | 同 A | 中（要自己写 no-LUT GEMM） | 0（就是现状） |
| 长上下文可行性 | **破坏** | 不破坏 | 不破坏 | 不破坏 |
| 实现复杂度 | 中（加载期一次 + 常驻管理） | 中（workspace 生命周期 + 指针绑定） | 高（新 GEMM） | — |
| 失败模式 | 全模型 OOM / KV 缩水 | 单层 OOM → 回退 | 自己 GEMM 慢 | — |
| 结论 | **否决** | **采用** | 备选（Tensile 不达标时） | 基线 |

---

## 6. 采用方案（B）的设计要点

1. **workspace**：进程级 `torch.empty(max_layer_NK, fp16)`，≈ 89 MB；尺寸按
   `max(N·K)`（含 draft 模型的层）。
2. **生命周期**：每层 prefill 前物化 → GEMM → 下一层覆盖。同一条流内上一层的 GEMM 已排在前，
   覆盖安全；**不允许**跨流并发使用（vLLM 单 batch 前向是层串行的，天然满足）。
3. **指针绑定**：**不要**把 `data_ptr()` 当常量传进图/编译产物（上游踩过：
   `sm70_layer_workspaces.py:1-30`）。用层前缀在 runtime 查表，或在 eager 下把 buffer 作为
   参数传入 op。
4. **分配时机**：模型加载期分配（在 KV 池划分与图捕获之前）；捕获期不得分配。
5. **OOM**：`except torch.OutOfMemoryError → warning + 回退到 W8A16 emu GEMM`，不抛。
6. **物化 kernel**：`fp8_dequant_to_fp16_out(w16, wq, w_s, block_n=128, block_k=128)`：
   K-major `[K,N]` 输出（供列主序 GEMM）、每线程 8 个 fp8、32 列 tile 摊薄 scale 读取、
   对齐 `K%8==0 / N%32==0 / block_k==128`（上游同款：`awq_sm70_gemm.cu:3055-3145`）。
   **scale 乘在 fp32 里做，最后 round 到 fp16**（上游是在 fp16 域 `__hmul`，省一条指令但多一次
   舍入；我们选精度优先，并在结果文档里记录该差异）。
7. **数值**：`W16 = fl16(Wq · w_s)` 引入一次 fp16 舍入（≈2⁻¹¹ 相对）。门禁：单层 `cos ≥ 0.99999`、
   rel_err ≤ 2⁻¹⁰、greedy 逐 token parity、logprobs finite。
8. **`min_m`**：按 role 分档（gate_up / down / qkv / o 各一个），初始值待实测；在上游
   （V100/24 GiB）是 3920（两形状 8000）——**不可照抄**，我们必须在自己的平台上重测。
9. **路由**：`M ≥ min_m[role]` 且 workspace 可用且 `A.dtype==fp16` 才走；否则走 W8A16
   emu GEMM。开关 `VLLM_DCU_FP16_PREFILL=0|1|auto`（默认 `0`，实测后晋级 `auto`）。

---

## 7. 卡空后必须验证的清单（本次不做）

| # | 待测项 | 判据（go/no-go） |
|---|---|---|
| V1 | 服务进程内 `torch.mm` fp16 是否可用（历史探针环境报过 `HIPBLAS_STATUS_ALLOC_FAILED`） | 可用才继续；不可用 → 直接转 C |
| V2 | rocBLAS/Tensile fp16 在 4 个生产形状 × `M∈{512,1024,1760,4096,8192}` 的 TF/s | ≥ 1.3 × 同形状 W8A16 emu GEMM 才值得 |
| V3 | 每个 role 的交叉点 `min_m`（扫描 M） | 定 `min_m` 表 |
| V4 | 物化 kernel 带宽与正确性（与 LUT 参照 `torch.equal` 级） | ≥ 600 GB/s 且精确 |
| V5 | workspace 重叠/复用是否影响 KV 池划分 | 静态占用与现状差 ≤ 0.3 GiB |
| V6 | 数值门禁（§6.7）与 e2e parity | 全绿才晋级默认 |
| V7 | 图捕获（若启用）：buffer 指针与 `torch.mm` 是否可捕获 | 不可捕获则在 eager 下用 |

---

## 8. 触发重评的条件

1. 平台换成显存更大的卡（≥48 GiB/卡）或 TP 升到 8 → A 重新评估；
2. `min_m` 实测低于 512（说明 Tensile 在小 M 也赢）→ B 的重复物化成本升到 ~4%，考虑
   「按 role 缓存最近一次物化结果」；
3. 上层决定放弃长上下文（不再需要 64k/128k）→ A 的显存代价可接受，重新比较；
4. V2 显示 Tensile 只快 <1.15× → 直接放弃本议题，prefill 留在 W8A16 emu GEMM。
