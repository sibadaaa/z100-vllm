# S18-4 附件：gfx906 verify GEMM/GEMV 效率与 `M` 摊销（关键路径）

> 上层文档：[S18 方案](z100-18-mtp-optimization.md) · 归档：[S18b](../result/z100-18b-mtp-first-baselines-and-cost-scaling-result.md)
> 状态：⬜ 待执行（等 S18-1 的 TP=2/INT8 判别臂与微基准复跑定基底）
> 适用对象：可独立交给子 agent 的内核任务书

## 1. 问题陈述（全部为本机实测）

| 事实 | 数值 | 来源 |
|---|---|---|
| FP8 槽 decode（TP=4/eager，maxlen 8192） | **123.32 ms/token** | S18b §3.1 A0 |
| MTP k=1 / k=2 | 136.53 / 145.46 ms/token（**−10.7% / −15.9%**） | S18b §3.1 |
| 每投机步耗时 | 123 → 239 (M=2) → 334 (M=3) ms，`≈123+105(M−1)` | S18b §3.1 |
| block-scaled FP8 GEMV 的 M 标度 | M=2 ≈ **2.1×**、M=3 ≈ 3.1×、M=4 ≈ 3.9×（4 个真实 shape 一致） | S18b §3.3 |
| M=1 的有效权重带宽 | **74–89 GB/s**（≈ 峰值的 6–10%） | S18b §3.3 |
| sdot4 decode 的有效带宽（TP=2，84 ms/step，14.5 GB/卡） | ≈ **172 GB/s**（同为离峰值很远） | S11k 归档 |

**结论**：在这台机器上，`1+k` 的 verify 成本 ≈ `(1+k)×` 单 token 成本，**没有摊销**；且两个仿真内核都远未到达带宽或指令吞吐上限。MTP 的收益上限由这一步的**边际成本**决定：

```
每 token 成本 = max(W, M×D) / (1 + Σp_i)        W = 读一遍权重的时间，D = 单行的 dot 工作量
```

- 若 `D ≪ W`（带宽受限）：`M` 增长时 `max()` 不变 ⇒ **额外 token 近乎免费**（MTP 大赚）；
- 若 `D ≈ W`（当前 FP8 实测）：`M=2` 直接翻倍 ⇒ **MTP 必亏**（与实测 −10.7% 吻合）；
- 若 `D > W`：更糟。

⇒ **本项任务不是"顺手调 MTP"，而是把 verify 内核推进到 `D ≪ W` 区间**；同时它本身就是 baseline decode 的提速项（当前 6–10% 带宽利用率 ⇒ 理论上有 5–9× 空间）。

## 2. 根因（已读源码，非猜测）

| # | 根因 | 证据 |
|---|---|---|
| R1 | GEMV 是"**一 block 一输出列**"（`grid(N)`），激活按 `m` 在每个列 block 里重读 | `csrc/rocm/fp8_emu_gemm.cu:456,485-503,542`（`ap[m*K4+i]` 在 `m` 循环内） |
| R2 | fp8→fp16 全靠 **256 项 LDS 查表**（每 dword 4 次 LDS），权重与激活各来一遍 | `csrc/rocm/fp8_emu_gemm.cu:488,498`（`fp8x4_to_f16x2x2`）、内核注释 |
| R3 | `M≥2` 时 host 先跑一个 **widen 内核**把整个激活转成 fp16（额外一遍读写，且把激活字节数翻倍后再按列重读） | `csrc/rocm/fp8_emu_gemm.cu:978-989` |
| R4 | 线程数固定 `nt=64`（INT8）/ 每 block 的 K 循环只有 `K4/nt` 次（5120/4/64 = 20），**迭代太短、ILP/占用率不足** | `csrc/rocm/fp8_emu_gemm.cu:542`、`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py` 的 `_GEMV_THREADS` |
| R5 | INT8 sdot4 结构同 R1（每元素仅 1 条 `sdot4`，但激活仍按列重读、每线程迭代同样短） | `csrc/rocm/w8a8_sdot4.cu:62-91` |
| R6 | MTP 一旦开启，`gdn_linear_attn.py:763-769` 会跳过 gfx906 packed/combined decode 快路；`GFX906_MTP_V2 MISS` 使 48 层 GDN 走 v1 | S18b §3.2；**占比小**（S11b：gdn+mamba 0.4%），列为次要项 |

**明确的非根因**（已否证，不要再查）：`MS=1/4/8` 桶式分派不会多算行（`if (m < M)`，`:490-502`）。

## 3. 目标（可判定的门禁）

**微基准门禁（单卡，合成权重，`tools/dcu/s18_fp8_block_mprobe.py` / `s18_int8_mprobe.py`）**，以真实 decode 形状（`N:K` = 各 TP 分片）为准：

| 门禁 | 阈值 | 含义 |
|---|---|---|
| G-K1 绝对吞吐 | **M=1 ≥ 400 GB/s** 权重字节带宽（当前 74–89） | 证明不是"转换/占用率受限" |
| G-K2 M 摊销 | **M=2 ≤ 1.2× M=1**；**M=4 ≤ 1.6× M=1** | 进入 `D ≪ W` 区间，MTP 有回本空间 |
| G-K3 数值 | 与 fp32 参考 `rel_err ≤ 1e-2`（当前 block 路径实测 `0.00e+00`，**不得退化**） | 不许为速度牺牲精度 |
| G-K4 无形状回退 | **任一已有形状不得变慢**（1Cat 的 "no shape regresses" 纪律） | 逐 shape ABBA |

**端到端门禁（在微基准通过后）**：同 flags 的 `A0(k=0)` 与 `A1(k=1)` 对比，**decode TPOT 改善 ≥1.3×**，TTFT/conc8 劣化 ≤5%，greedy token 与基线一致或差异落在噪声底并归因。

## 4. 设计草案（v1，按优先级）

1. **列 tile + 激活 LDS 常驻（治 R1/R3）**：一个 block 处理 `TN`（32–64）个输出列；K 方向切成 `BK`（128–512）；每个 K-tile 把 **激活 tile 一次性转成 fp16x2 放进 LDS**（`M×BK×2` 字节，M=4/BK=512 ⇒ 4 KiB），供 `TN` 列共享。激活的 HBM/LDS 读放大从 `×N` 降到 `×(N/TN)`。
2. **权重一次转换、LDS 复用（治 R2）**：每个 K-tile 把 `TN×BK` 的 fp8 权重转成 fp16x2 存入 LDS（`TN=32,BK=256` ⇒ 16 KiB），内层循环变成纯 `v_dot2_f32_f16`（操作数都在 LDS/寄存器），把查表从热路径里拿掉。
3. **双缓冲 K-tile**（`ds_read` 异步/预取），让 HBM 与 dot 重叠 —— 这是达到 G-K1 的关键。
4. **去掉 `M≥2` 的 widen 前置 kernel（治 R3）**：在 kernel 内按 tile 转换激活，转换成本与 `M` 无关（每元素只转一次），避免"先写一遍 fp16 再按列重读"。
5. **占用率/ILP 调优（治 R4）**：`nt ∈ {64,128,256}`、每线程处理多个输出列（`N` 方向 unroll）与多个 K-tile；目标是每 block 的工作量足够大、`grid` 足够多，既不空转也不排队。**必须做 ABBA 扫描**，不允许"取一个数就用"。
6. **INT8 侧**（若判别臂显示 INT8 才是 MTP 的正确基底）：同一套列 tile/双缓冲结构套到 `w8a8_sdot4_gemv_kernel`；sdot4 每指令 4 MAC，更容易进入 `D ≪ W`。

**禁止项**（沿用项目既有纪律）：不许物化 fp16 权重（27B 会把显存翻倍）；权重必须保持 checkpoint 布局、kernel 内边读边转；只对 gfx906 生效（`on_gfx906()`/`is_hygon_dtk()` 可回退）；不许改非 DTK 行为。

**两个已知工具链陷阱**（S11f 实测，必须写进内核注释并被单测锁住）：
- `u16x2` 向量位运算被 DTK clang 15/18 加了 `op_sel_hi` lane 选择（`2^-14` → `4.0`）；
- `V_PK_LSHLREV_B16` 的移位量按 lane 取 src0 的两个半字，标量立即数只移一个 lane（`2^-14` → `64.0`）⇒ 用 32 位标量逐 lane。

## 5. 执行顺序与写作用域

| 步 | 内容 | 写作用域 | 门禁 |
|---|---|---|---|
| S18-4a | 微基准扩展：`--modes gemv,gemm` + `--threads-list`（**已完成，待跑**） | `tools/dcu/s18_*_mprobe.py` | 强制 GEMM 路径与 nt 扫描的结果表 |
| S18-4b | 微基准复跑（含行一致性修正版）+ INT8 对标 | 同上 + `/mnt/game/s18` | G-K1/G-K2/G-K3 现状基线 |
| S18-4c | 新内核原型（列 tile + LDS 双缓冲），先只做 FP8 block；**只编译不端到端**跑微基准 | `csrc/rocm/fp8_emu_gemm.{h,cu}` | G-K1…G-K4 |
| S18-4d | INT8 侧同构改造（若判别臂指向 INT8） | `csrc/rocm/w8a8_sdot4.{h,cu}` | 同上 |
| S18-4e | 端到端 A/B（A0 vs A1，eager，同 flags，n≥5） | 只跑脚本 | 端到端门禁 |
| S18-4f | 收尾：md5/回滚件、单测、文档 | 测试文件 + 归档 | 归档齐 |

**与 S18-1/S18-3 的关系**：若判别臂证明 INT8 的 `M=2` 已 ≤1.2×（即 INT8 内核对 M 天然友好），则 S18-4c 的优先级下降，改走"INT8 + MTP 上线"；FP8 的绝对提速仍作为独立收益（S18-4c/d）保留。

## 6. 参考

- 内核源码：`csrc/rocm/fp8_emu_gemm.cu`（`:442-529` GEMV、`:978-989` widen、`:1084-1130` host 入口）、`csrc/rocm/w8a8_sdot4.cu:62-123,386`
- 既有归档：`docs/result/z100-11f-fp8-bf16-emu-kernels-result.md`（内核族契约与两个 clang 陷阱）、`z100-11e2-int8-family-integration-result.md`、`z100-11b-perf-profiling-result.md`（GEMM 占 97.3%）
- 外部参考：1Cat-vLLM 的 "exact-shape 战术" 与 "no shape regresses"（`docs/result/z100-11g-1cat-vllm-reference-analysis-result.md`）
