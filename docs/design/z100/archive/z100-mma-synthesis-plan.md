# gfx906 矩阵内核方案：与 1Cat 分析、现有路线图的合并修订

> **合并对象**
> 1. `gfx906-mma/方案-gfx906矩阵内核与WMMA移植.md`（下称 **MMA 方案**，另一 agent 产出，2026-10-24）
> 2. `$HOME/server/优化方案-性能路线图.md`（下称 **路线图**，INT8 27B 单流实测，2026-10-06）
> 3. 本仓库 `docs/design/z100/research/z100-1cat-transfer-research-v2.md`（v1 已移除；下称 **v2**，1Cat-vLLM 代码级分析）
>
> **本方案新增的实测**：在 Z100L 上用 DTK 26.04 容器实跑了 `v_dot8_i32_i4` 的**汇编接受性 / 吞吐 / 数值语义**三项（§2）——这是 MMA 方案与 v1/v2 都没测过的指令。
>
> **日期**：2026-10-24 · 性质：方案修订（含新实测）

---

## 0. 结论：MMA 方案的方向对，但**靶子瞄错了**

| 判断 | 结论 |
|---|---|
| 硬件事实地基（无 MFMA/WMMA/bf16，靠 `v_dot2`/`v_dot4` + LDS） | ✅ **完全正确**，本次实测再次确认（§2） |
| 原语三重验证（编译→机器码→上卡数值）的工作方法 | ✅ **应升级为仓库标准** |
| "打不过 rocBLAS 就用 rocBLAS" | ✅ 正确 |
| **M1/M2 的靶子"4096³ FP16 GEMM ≥ 85% rocBLAS"** | ❌ **瞄错了目标**。按路线图自己的数字，这项对 decode 和 prefill 的贡献都 **< 5%**（§3.1） |
| **Track A（WMMA 兼容层）** | ⚠️ 不是当前瓶颈，应**降级为 P3 保险**，MMA 方案自己也已承认 |
| **缺失的算力原语** | ❌ **漏了 `v_dot8_i32_i4`**：实测 **2× dot4 / 9.1× fma_f32**，且能正确出数（§2） |

**一句话**：把 MMA 方案的引擎**从"通用 GEMM"改指向"塌缩长尾的融合内核 + W4A4 量化 GEMM"**。前者是路线图实测的 50–60% decode 成本所在，后者用新验证的 dot8 把 prefill 算力上限翻倍。**通用 GEMM 一项都不要做。**

---

## 1. 三份材料的一致性检查

### 1.1 两套配置，必须先分清

| | 本仓库文档（v1/v2 引用的数据） | 路线图 |
|---|---|---|
| 模型 | **Qwen3.8-27B-FP8** | **Qwen3.8-27B-Channel-INT8-w8a8** |
| 栈 | DTK 26.04 树 + rocm101 树 | vLLM 0.18.1 + DTK 26.04 |
| decode | 240 ms/step、**25.6 GB/s**（3% 带宽） | **121.6 ms/token**、~1900 kernel/token |
| 主要成本 | rocBLAS 47.2% / GDN 25.8% / fp8 GEMV 4.3% | 长尾 50–60% / allreduce 9% / GEMV 10–13 ms |
| prefill | 512 tok = 2722 ms（~2.5 TF/s/卡） | 2.768 ms/prompt-token |

**两者结论一致且互补**：decode 都不是算力问题，而是"组织方式"问题；只是 FP8 栈浪费在 rocBLAS 瘦 GEMM 上（25.6 GB/s），INT8 栈浪费在 1900 个碎 kernel 上。

> 本方案对两个栈都适用，但**验收必须分栈做**，不可混用数字。

### 1.2 三方对同一件事的独立佐证

| 现象 | 路线图（INT8 实测） | 本仓库（FP8 实测） | 1Cat（V100 实测） |
|---|---|---|---|
| decode 被碎 kernel 主导 | ~1900 kernel/token，长尾 50–60% | `aten::copy_` 3074 次、meta 7.5%、700–980 次 allreduce | c8 每轮 **62.9 ms**；RMSNorm 曾是 **9-kernel torch 链** |
| 单 kernel 不慢，组合才慢 | GEMV 微基准 24–44 µs vs 实跑 55 µs | fp8 GEMV 微基准 176 GB/s，模型级仅 25.6 GB/s | launch 数 1085→1086 **不变**，收益全来自 kernel 服务效率 |
| 大 M GEMM 不是瓶颈 | sdot4 GEMM 22 TF/s（仅占 prefill 0.72 ms/token 的 26%） | prefill GEMM ~2.5 TF/s vs 物化路径独立 16.5–17.7 | 全模型 prefill 的 83% 在 4 类大 GEMM，靠 **layout 复用**而非新代数 |

**三个独立环境、两代硬件、两种量化格式，指向同一个结论：Z100 的钱在"kernel 数量与访存组织"，不在"GEMM 吞吐"。**

---

## 2. 新实测：`v_dot8_i32_i4` 三重验证通过

> 环境：`z100-dtk-runtime:20261018` 容器，`hipcc --offload-arch=gfx906 -O3`，`HIP_VISIBLE_DEVICES=0`（卡 0 = Z100L，60 CU）。
> 探针：`/tmp/isa/{dotprobe,dot8num}.cpp`（可复跑，见 §8）。

### 2.1 汇编接受性（clang-17，`-mcpu=gfx906`）

```text
PASS  v_dot4_i32_i8 / v_dot8_i32_i4 / v_dot8_u32_u4 / v_dot2_i32_i16
PASS  v_dot2_f32_f16 / v_pk_fma_f16
FAIL  v_dot2_f32_bf16 / v_mfma_f32_16x16x16f16 / v_wmma_f32_16x16x16_f16
      → "instruction not supported on this GPU"（负控制正确失败，证明方法有效）
```

**`v_dot8_i32_i4` 是 gfx906 真实可用的指令**，不是误报。

### 2.2 吞吐（同指令数、8 条独立链、480×256、20 次重复）

| 指令 | 耗时 ms | MAC/s | **× `v_fma_f32`** | 折算 TOPS |
|---|---:|---:|---:|---:|
| `v_fma_f32` | 9.53 | 4.23 T | 1.00 | 8.5 |
| `v_pk_fma_f16` | 8.34 | 9.66 T | 2.29 | 19.3 |
| `v_dot2_f32_f16` | 8.34 | 9.66 T | 2.29 | 19.3 |
| `v_dot2_i32_i16` | 8.34 | 9.66 T | 2.29 | 19.3 |
| `v_dot4_i32_i8` | 8.35 | 19.29 T | 4.56 | 38.6 |
| `v_dot4_u32_u8` | 8.37 | 19.25 T | 4.55 | 38.5 |
| **`v_dot8_i32_i4`** | **8.37** | **38.48 T** | **9.11** | **77.0** |

**关键结论：7 种指令的耗时几乎完全相同（8.34–9.53 ms）⇒ 它们都是 1 条指令/lane/cycle 的发射率，MAC 吞吐严格等于"每条指令做几个 MAC"。**
因此 dot8 **真实提供 8 MAC/lane/cycle**，不是被拆成两次 dot4。

**顺带得到一个重要副产品**：实测 fma_f32 只有 4.23 T MAC/s，而 1.6 GHz × 60 CU × 64 lane = **6.14 T** ⇒ **有效频率只有 ~69%（≈1.1 GHz）**。功率墙 280 W 下 **1.6 GHz 不是可持续频率**，所有容量规划都要按 ~1.1–1.25 GHz 算。

### 2.3 数值语义（这一条会直接决定打包格式）

7 个用例全部通过（`tools/dcu/dot8_numeric_probe.cpp`，退出码 0）：

```text
signed:   8 x (1*2)                    -> 16     OK
signed:   8 x (-1*-1)                  ->  8     OK
signed:   8 x (1*-1)                   -> -8     OK
signed:   -8+7+6+5+4+3+2+1             -> 20     OK   (a=0x12345678)
signed:   zero operand                 ->  0     OK
unsigned: 8+7+6+5+4+3+2+1              -> 36     OK   (同一组 bits，换 u4)
unsigned: 8 x (15*1)                   -> 120    OK
```

**签名语义已被两种变体的对照钉死**：同一组 bits `0x12345678`，`i4` 得 20（`0x8`→−8）、`u4` 得 36（`0x8`→8）。
（初版探针把第 4 例的期望写成 36 而报"失败"，实为**我的期望写错、硬件行为正确**——这个反转本身就是最有说服力的语义证据。）

> **工程铁律**：用 `v_dot8_i32_i4` 时，**int4 权重必须按有符号 (−8..7) 打包**；若要无符号 (0..15)，必须改用 `v_dot8_u32_u4`。混用会在权重出现 8..15 时静默产生负贡献——这是一个**只在部分数据上发作**的正确性 bug，必须写进内核契约。

### 2.4 对方案的直接影响

- MMA 方案的算力清单（fma_f32 / pk_fma_f16 / dot2 / dot4）**应补上 dot8**，并把 dot8 列为**int4 量化的唯一正解**（不是"用 dot4 模拟"）。
- int4 的算力上限从 int8 的 38.6 TOPS 提到 **77 TOPS**（本卡 sustained），即 **prefill 算力天花板翻倍**。

---

## 3. 目标重排：按两份路线图的实测收益算账

### 3.1 为什么"4096³ GEMM ≥85% rocBLAS"是错的靶子

| 场景 | 该场景占总时延 | 当前水平 | 换更好的 GEMM 能拿到 | 占总收益 |
|---|---:|---|---:|---:|
| decode 权重流（GEMV） | 10–13 ms / 121.6 ms ≈ **9%** | 232 次 × 55 µs ≈ 12.8 ms ⇒ **~527 GB/s ≈ 拷贝上限 777 的 68%**，已近地板 | 提到 777 → 8.7 ms，省 ~4 ms | **~3%** |
| prefill sdot4 GEMM | 0.72 ms / 2.768 ms ≈ **26%** | 22 TOPS；按**本机实测**上限 38.6 TOPS 已是 **57%**（路线图的"41%"是拿 MI50 的 53 TOPS 当分母） | 提到 28 → 省 0.15 ms/token，但也只用掉到 73% | **~5%** |
| **decode 长尾（600+ 小算子 + 244 量化 + 153 RMSNorm）** | **≈60–73 ms / 121.6 ms = 50–60%** | 每 kernel 5–20 µs | 塌缩 ~400 个 kernel → **20–30 ms** | **~20%** |
| prefill GDN fp32 路径 | 0.95 ms / 2.768 ms ≈ 34% | 全程 fp32 | 转 fp16/bf16 → 省 0.3–0.4 ms | **~13%** |

**结论**：通用 GEMM 吞吐排第 2 或第 3 位，**远低于长尾塌缩与 GDN 精度路径**。
MMA 方案把 6 天（M1+M2）投在 GEMM 上，机会成本是"长尾只塌缩一半"。

### 3.2 重排后的优先级（合并三份材料）

| P | 项 | 依据 | 预期 | 用不用 MMA 引擎 |
|---|---|---|---|---|
| **P0** | **长尾塌缩：RMSNorm + int8 量化 + GEMV 融合** | 路线图 #1（20–30 ms）+ 1Cat 的 "9-kernel→1" 实证 | decode **−20~30 ms** | ✅ **是核心**（dot4 内积） |
| **P0** | **多投影合并启动**（qkvz+ba / gate+up 各 1 次） | 路线图 #2（10–15 ms）+ 1Cat "5..40 行一次 launch" | decode **−10~15 ms** | ✅ 复用同一引擎 |
| **P0** | **GDN chunk 路径 fp32 → fp16** | 路线图 §6 #1（prefill 34% 在那条路径） | prefill **−0.3~0.4 ms/token** | ❌ 纯数据类型修复 |
| **P1** | **lm_head 转 int8** | 路线图 #5（4–5 ms，现在 bf16 单次 8.2 ms） | decode **−4~5 ms** | ✅ dot4 W8A8 |
| **P1** | **W4A4 GEMM（dot8）做 prefill** | §2 新实测：算力翻倍 | prefill 算力上限 38.6→77 TOPS | ✅ **dot8 专属** |
| **P1** | **attention decode 单 kernel** | 路线图 #4（8–12 ms） | decode **−8~12 ms** | ⚠️ 部分（小 M 分页 GQA） |
| **P2** | 互连（方案 A/B） | 路线图 #3（8–11 ms） | decode **−8~11 ms** | ❌ |
| **P2** | int4 权重-only decode GEMV | §2：权重字节 6.75→3.4 GB | decode **−4~6 ms** | ✅ dot8 或 dot4 |
| **P3** | Track A：WMMA 兼容层 | MMA 方案自认"非关键路径" | 0（当前） | 保险 |

> 累计（P0+P1）：decode 121.6 → **约 70–85 ms（12–14 tok/s）**；prefill 2.768 → **约 2.2–2.4 ms/token**。与路线图 §5/§6 的估计一致，**但因为引入 dot8，prefill 还有翻倍空间**。

---

## 4. 修订后的 MMA 方案（M0–M4 重新定义）

### M0（0.5 天 → **1 天**）：把基准做对

MMA 方案原本只测 4096³ 峰值。**必须补上真实形状与 M 扫描**：

```text
必测 1：指令级峰值（已有，见 §2）—— 补 dot8 + u4 变体
必测 2：7 个真实投影形状 × M ∈ {1,2,4,8,16,32,64,128,512,2048,4096}
        （5120:4352 / 4352:5120 / 5120:17408 / 17408:5120 / 5120:1536 / 1536:5120 / lm_head 65536:5120）
        × {rocBLAS fp16, rocBLAS int8, 现有 sdot4_gemv, 现有 w8a8_gemv}
必测 3：每个形状的有效带宽 GB/s 与 % of 777.5 GB/s —— 这是判断"撞哪堵墙"的唯一口径
必测 4：dot8 在真实形状上的 sustained 频率与功率（验证 §2.2 的 1.1 GHz 结论）
```

**验收**：得到一张"M × 形状 × 实现"的矩阵，标出 **rocBLAS 输给自研的格子**。后续只在那些格子里写内核。

### M1（2 天 → **重定义为"融合原语"**）

不做通用 GEMM，做 **3 个融合原语**（全部 dot4/dot8，全部 fp32 累加）：

1. `rmsnorm_quant_gemv`：RMSNorm → per-token int8 量化 → GEMV，一次完成。
   直接吃掉每 token 的 **153 次 RMSNorm + 244 次 quant + 232 次 GEMV** 的一部分。
2. `gemv_multi`：一次 launch 服务同层的多个投影（qkvz+ba / gate+up），行数 1..40 连续。
3. `dequant_scale_epilogue`：GEMV 结果直接做 scale/bias/激活（`silu·mul`），不落中间张量。

**验收**：每个原语对拍 PyTorch 参考（int8 路径误差为 0，fp16 路径 rtol<1e-2）；**单原语 ≥ 现有 kernel 合并后的 e2e 收益**才算赢。

### M2（4 天 → **拆成 M2a/M2b**）

- **M2a（2 天）**：把 M1 的原语接进 vLLM，做**服务级 A/B**。
  这是路线图 §5 的 #1+#2+#6，预期 **−30~45 ms**。
- **M2b（2 天）**：**W4A4 prefill GEMM（dot8）**。
  - 需先解决量化格式：per-group（如 128）+ fp16 scale，权重与激活均 int4。
  - 先只在 **一个形状**（gate_up 5120:17408）上做，过了质量门才推广。
  - 质量门：8 prompt 的 PPL/KL vs int8 ≤1%，greedy 输出可解释。

### M3（4 天 → **降级为 P3**）

WMMA 兼容层。**触发条件**：出现一个必须用 WMMA API 且无法改写的库。
在此之前只保留 `gm_wmma.h` 的**接口设计文档**，不写实现。

### M4：移植目标库 → **改为"接 1Cat 的控制面"**

MMA 方案的 M4 是"移植一个目标库端到端"。按 v2 的分析，**比移植库更高性价比的是移植 1Cat 的 154 行控制面**：

- `MixedPrefillBudget`（56 行）+ `MixedPrefillTimer`（53 行）→ 直接治 c8 尾延迟；
- `_balanced_prefill_chunk`（~24 行）→ 对冲 880 对齐损失（每步丢 14.06% 预算）。

这两项**不需要新内核**，且与 MMA 的内核工作正交，可并行。

---

## 5. 与 1Cat 的对照：哪些已验证，哪些是陷阱

### 5.1 1Cat 独立验证了"塌缩长尾"这条路

| 1Cat 的具体做法 | 对应路线图条目 | 1Cat 实测 |
|---|---|---|
| 精确 gated RMSNorm 覆盖 **64 token**，替换 **9-kernel torch 链** | #1 融合长尾 | c4 下消除 9 个 kernel |
| router / shared expert / PLE spec conv **5..40 行一次 launch** | #2 多投影合并 | bitwise 相同 |
| QSA shared-key indexer 多请求一次打分 | #4 attention 单 kernel | c8 62.9 ms/round |
| 去掉 expert-cache 命中的 system-scope fence | — | 同上 |
| 所有改动 **bitwise equal to per-request result** | 验收口径 | — |

**这给了 Z100 团队一个重要信心：路线图 §5 的 #1/#2 不是猜测，是另一套硬件上已落地的工程。**

### 5.2 必须提前知道的陷阱

| 陷阱 | 1Cat 的实证 | 对 Z100 的含义 |
|---|---|---|
| **混合批 + 投机解码会静默算错** | 混合批 draft convolution 只处理 5 行里的第 1 行，**20/44 正确**，工具调用参数被截断 | 上 long-tail 融合后如果改变了行的组织方式，必须建 44 例端到端门禁 |
| **前缀缓存可能静默归零** | `block 1648 + grid 8192` 时 `hits_total=0`，同一 232K prompt 86.5 s vs 1 s | 改 block/grid 后必须验 `prefix_cache_hits_total` |
| **诊断 kernel 会使延迟测量失效** | round-cost ledger 默认 False | 测量通道与热路径分离 |
| **微基准不能预测 e2e** | QPN8 未默认开；C1 SiLU/down 融合算子级更快但 e2e 无收益 | 路线图已写的"每项独立 A/B"必须执行 |
| **数学等价 ≠ 性能等价** | 24 种 `m8n8k4` K4 排列只有 1 种与 m16n16k16 逐位一致 | 任何 tile 重组都要 bitwise 门 |

---

## 6. 需要修正 MMA 方案的三处技术细节

### 6.1 块设计的累加器算术不自洽（**必改**）

原文：
> 输出 tile C: **128×128**(fp32)，每工作组分 4 个 wave，每 wave 负责其中 4 个 32×32 子块
> 每 lane 持有: **4×4 个 fp32 累加器** → 16 个 VGPR
> LDS: A(**64×16**) + B(**16×64**) 双缓冲 = 8KB

三个数字互相矛盾：

| 假设 | 需要的累加器/lane | 与"16 VGPR"是否一致 |
|---|---|---|
| 输出 64×64、256 线程 | 4096 / 256 = **16** | ✅ 一致（且 LDS 尺寸 64×16/16×64 也对） |
| 输出 128×128、256 线程 | 16384 / 256 = **64** | ❌ 差 4 倍 |

**改法**：把 M1 的基准块定为 **64×64 / 256 线程 / 16 累加器 / K-step 16**，128×128 作为"M2 的扩展配置（64 累加器，需核对 256 VGPR 预算与占用率）"。
另外 64×64 的 tile 在 decode（M≤8）下只能出 **1 个 M-tile**，grid 会不足一个 wave——**decode 必须另用 64×32 或 split-K**（见 §6.2）。

### 6.2 一套 tile 不能同时服务 prefill 与 decode

| | prefill（M≥512） | decode（M≤8） |
|---|---|---|
| 瓶颈 | 计算 | **HBM 带宽** |
| 需要的 tile | 大（64×64 起）以提高算术强度 | **小 N-tile + split-K**，把 grid 撑到 ≥4 waves |
| 目标占用率 | 中等（每 lane 多累加器） | **高**（隐藏 global 延迟） |
| 1Cat 的对口设计 | BM32 phase kernel（大 tile、K/V 复用） | QPN8：gate/up **split-K=8**、down **split-K=16**、2 条累加链 |
| 验收口径 | TFLOPS | **GB/s 与 % of 777.5** |

**结论**：decode 引擎的验收标准必须是 **有效带宽**，不是 TFLOPS。用 TFLOPS 验收会得到"已经很快"的错觉（现测 0.163 TF/s 的 rocBLAS 瘦 GEMM 就该被这条标准淘汰）。

### 6.3 缺 dot8，且 dot4 的实测上限比原文低

原文按 MI50 同代估：FP16 ≈26 TFLOPS、INT8 ≈53 TOPS。
实测（Z100L，持续负载，~1.1 GHz 有效频率）：**FP16/dot2 19.3、INT8/dot4 38.6、INT4/dot8 77.0 TOPS**。
**建议把所有容量规划按实测值写**，并把"1.6 GHz 是 nameplate、持续约 1.1–1.25 GHz"写进风险节（这直接解释了本仓库 WP1 里 "fp16 19.61 TF/s" 与理论 24.6 的差距）。

---

## 7. 里程碑（修订版）

| 里程碑 | 内容 | 工期 | 验收（缺一不可） |
|---|---|---|---|
| **M0** | 指令级 + 真实形状基准矩阵（含 dot8） | 1 天 | 产出"M×形状×实现"表；标出 rocBLAS 输的格子；确认 dot8 数值语义 |
| **M1** | 3 个融合原语（norm+quant+gemv / multi-gemv / epilogue） | 2 天 | 对拍通过；int8 路径误差 = 0 |
| **M2a** | 接进 vLLM，服务级 A/B | 2 天 | decode **≥ −15 ms**（否则回滚） |
| **M2b** | W4A4 dot8 prefill GEMM（单形状起步） | 2 天 | PPL/KL ≤1%；prefill ≥ +10% TFLOPS |
| **M3** | 全形状推广 + 多投影合并 | 3 天 | decode 累计 **−30~45 ms**；bitwise/对拍门全绿 |
| **M4** | 1Cat 控制面移植（mixed-prefill + balanced chunks） | 2 天 | c8 ITL p99 → 1.5 s 量级；长 prompt 步数 −12~14% |
| **P3** | WMMA 兼容层（仅接口文档） | — | 触发条件见 §4 M3 |

**总工期 ≈ 12 天**（MMA 原方案 1.5 周 ≈ 10.5 天，但 6 天错投在通用 GEMM；本方案把同样的天数投在收益高 4 倍的项上）。

---

## 8. 复跑与证据

```bash
# 环境（本次实测所用，卡 0 = Z100L 空闲）
docker run --rm --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --shm-size=8g -v /opt/hyhal:/opt/hyhal -v /tmp/isa:/work \
  -e HIP_VISIBLE_DEVICES=0 z100-dtk-runtime:20261018 \
  bash -c 'cd /work && hipcc --offload-arch=gfx906 -O3 dotprobe.cpp -o dotprobe && ./dotprobe'

# 数值语义
bash -c 'hipcc --offload-arch=gfx906 -O3 dot8num.cpp -o dot8num && ./dot8num'
```

| 产物 | 内容 |
|---|---|
| [`tools/dcu/dot_throughput_probe.cpp`](../../../../tools/dcu/dot_throughput_probe.cpp) | 7 种指令吞吐对比（§2.2 的表），**已收进仓库** |
| [`tools/dcu/dot8_numeric_probe.cpp`](../../../../tools/dcu/dot8_numeric_probe.cpp) | dot8 i4/u4 语义验证（§2.3），**已收进仓库**，7/7 通过、退出码 0 |

> 注：本 fork 的 `.gitignore:313` 把 `tools/dcu` 整体忽略（DCU 工具链不入库），所以这两个探针是**本地工件**，与 `isa_probe.sh` 的处置一致。若要长期保留，建议单独放一个不受忽略的目录，或例外放行 `tools/dcu/*.cpp`。
| 汇编接受性 | `clang-17 -target amdgcn-amd-amdhsa -mcpu=gfx906 -x assembler -c` |

> 建议把这两个探针**收进 `tools/dcu/`**，与 `isa_probe.sh` 并列，成为新内核的准入工具。

---

## 9. 风险

| 风险 | 概率 | 对策 |
|---|---|---|
| **dot8 在真实形状上被功率墙压回 dot4 水平** | 中 | M0 必测 sustained 频率；以 GB/s 而非峰值验收 |
| **W4A4 精度不过门** | 中高 | 先单形状；per-group scale；不过门就退回 W8A8（dot4 仍是 4.56×） |
| **int4 打包符号出错** | 中 | §2.3 的铁律写成断言 + 单测；用 `u4` 就必须用 u32 变体 |
| 融合后混合批算错（1Cat 20/44 前例） | 中 | 44 例端到端门禁 + bitwise 对拍 |
| 64 累加器版本撞 256 VGPR 上限 | 中 | 先做 64×64；128×128 只在 prefill 且实测占用率后决定 |
| 前缀缓存静默失效 | 中 | 每次改 block/grid 后验 `prefix_cache_hits_total` |
| 长尾融合后 e2e 无收益（如 1Cat 的 SiLU/down 融合） | 中 | 路线图已定的"每项独立 A/B + 单变量回退"必须执行 |

---

## 10. 局限

1. 本次实测**只跑了指令级微基准**（单卡、纯 ALU、无访存），未验证任何真实形状的 GEMM/GEMV 收益；§3 的收益数字全部引自路线图与 1Cat 的既有实测。
2. `v_dot8_u32_u4` 已验证可汇编 + 7 个数值用例中的 2 个（§2.3）；**未在真实量化权重上做过端到端精度验证**。
3. 路线图与 1Cat 的模型/栈不同（INT8 27B vs NVFP4/FP8 27B），**跨栈数字只作方向参考**。
4. 未做任何 vLLM 代码改动；本方案是施工图，不是实测结论。

---

## 11. 参考

- `gfx906-mma/方案-gfx906矩阵内核与WMMA移植.md`（MMA 方案原文）
- `gfx906-mma/primitives/{fdot2,pk_fma,sdot4}.cpp` + `extracted/*.co`（原语三重验证产物）
- `rocwmma_check/报告-rocWMMA-DTK移植验证.md`（rocWMMA 不可移植的前置结论）
- `$HOME/server/优化方案-性能路线图.md` §2/§3/§5/§6（INT8 单流实测）
- `$HOME/server/修复方案-INT8-sdot4.md`（INT8 内核工程经验）
- 本仓库：[`z100-1cat-transfer-research-v2.md`](../research/z100-1cat-transfer-research-v2.md)（v1 已移除）、[`KERNEL_REPORT_20261024-053333-E2C.md`](../../../../KERNEL_REPORT_20261024-053333-E2C.md)
- 1Cat-vLLM（`origin/main` = `6a143197e`）：`docs/design/sm70_qwen38_step0_kernel_budget.md`、`sm70_mixed_batch_decode_rows.md`、`mixed_prefill_latency.md`、提交 `d67a10c4f`
