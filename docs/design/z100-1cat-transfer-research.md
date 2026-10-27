# 借鉴 1Cat-vLLM 提升 Z100L 的 prefill / decode / 显存 —— 调研报告

> **对象**：4× 海光 DCU Z100 系列（gfx906，DCU0/1 = Z100L @1.60 GHz，DCU2/3 = Z100 @1.32 GHz，60 CU，32 GiB/卡）
> **参照项目**：[`1CatAI/1Cat-vLLM`](https://github.com/1CatAI/1Cat-vLLM)（v1.5.1，main `b379775d`，V100/SM70 专用 vLLM 工程分支）
> **本地基线**：[`z100-rocm101-perf-verdict.md`](../result/z100-rocm101-perf-verdict.md)、[`z100-rocm101-perf-baseline-freeze.md`](../result/z100-rocm101-perf-baseline-freeze.md)、[`z100-longctx-vram-optimization-plan.md`](z100-longctx-vram-optimization-plan.md)、[`KERNEL_REPORT_20261024-053333-E2C.md`](../../KERNEL_REPORT_20261024-053333-E2C.md)
> **日期**：2026-10-24
> **性质**：调研报告（只读分析 + 移植方案）。本文**没有**启动服务、**没有**产生新的性能数字；所有 Z100 数字均引自仓库内既有实测文档，所有 1Cat 数字均引自其 README / `docs/design/*`。

**证据标注约定**：【1Cat实测】= 1Cat 文档记录的数字；【Z100实测】= 本仓库文档记录的数字；【推断】= 由前两者推导、尚未在本机验证。

---

## 0. 一页结论

### 0.1 三句话

1. **两边是同一类"被软件抛弃的硬件"**：V100/SM70 与 Z100/gfx906 都没有 bf16 指令、没有 FP8 数据类型；差别在于 V100 有 HMMA 矩阵核而 gfx906 **完全没有**矩阵核。因此 **1Cat 的 kernel 不能搬，1Cat 的方法论必须搬**。
2. **Z100 现在最大的问题不是"某个 kernel 慢"，而是"整条链路都在等"**：decode 每步只需搬 6.8 GB 权重，却跑了 238 ms（**25.6 GB/s = 777 GB/s 拷贝上限的 3%**）；prefill 512 token 跑了 2722 ms（**≈2.5 TF/s/卡 = fp16 实测峰值 19.61 TF/s 的 13%**），而同一套物化路径在独立微基准里能跑 16.5–17.7 TF/s。**两边的差距都是 10–27 倍量级，属于结构性问题，不是调参问题。**
3. **1Cat 给出的答案恰好就是这三类结构性问题**：把"不该进热路径的数据格式"挡在热路径外（FP8-KV 桥接）、把"不该切的 chunk"合起来（Mamba state grid 解耦 / 稀疏保留边界）、把"不该串行的层"并行掉（grouped verify、混合批 decode 行、mixed-prefill 预算）。这三件事在 Z100 上都还没做，且都不需要矩阵核。

### 0.2 优先级总表

| 优先级 | 移植项 | 目标 | 预期量级 | 依据 | 工作量 |
|---|---|---|---|---|---|
| **P0** | **FP8 E4M3 KV cache + 物化桥接** | 显存 | KV 容量 ≈**×1.9**；长上下文 decode KV 读带宽 **÷2** ⇒ ITL −20~35% | 【1Cat实测】桥接后 prefill 4.5–4.9×、全模型 64K prefill −50.75%；【Z100实测】L4 设计已估算容量 ×2 | 中（内核 + 校准，~300 行） |
| **P0** | **Prefill GEMM 形状路由审计** | prefill | TTFT 2722 → **~1400 ms**（若 GEMM 回到 12 TF/s） | 【Z100实测】aten::mm 占 prefill 1.89–2.04 s / 2.72 s；同一路径独立实测 16.5–17.7 TF/s | 低（先测量，后改路由表） |
| **P0** | **Mamba align chunk 解耦** | prefill + 显存 | prefill 吞吐 **+40~110%**；KV 池 **+23~29%** | 【1Cat实测】align 切 chunk 使 8K prefill 5821→2487 tok/s，稀疏保留修复后回到 5356 | 中（调度器 ~150 行） |
| **P1** | **CUDA Graph（路线 A 已验）** | decode | decode **−7~9%**；compile 下 TTFT 3.35→1.35 s 已实测 | 【Z100实测】`z100-cudagraph-route-a-20261021.md` | 低（重新走门禁） |
| **P1** | **单体 decode GEMM 取代 rocBLAS 瘦 GEMM** | decode | 若把 aten::mm 的 43.8% 设备时间压到 fp8 GEMV 同量级，decode 有 **2–3×** 空间 | 【1Cat实测】QPN8 TPOT −7.4%，DRAM 82–88%，<25% 占用率时长 93.6%→48.6% | 高（新内核） |
| **P1** | **混合批 decode 行分组（≤8 行共享 KV tile）** | decode 尾延迟 | 131K 每层 22.70→4.08 ms（**5.5×**） | 【1Cat实测】`sm70_mixed_batch_decode_rows`；【Z100实测】WP5 c8 p99 = 5014 ms，结论是"唯一路径是调度器 decode 预留" | 中高 |
| **P1** | **mixed-prefill step 预算（250 ms）** | 并发/尾延迟 | 常驻吞吐 **4.41×**，最长停顿 2.873→0.310 s（−89%） | 【1Cat实测】`mixed_prefill_latency` | 中 |
| **P2** | **W4A16 / 低位权重量化** | 显存 + decode | 权重 9.04→~5.5 GiB/卡（−3.5 GiB）；带宽受限部分 decode 可达 2× | 【1Cat实测】NVFP4 是其 32 GB 卡跑 27B 的前提；【Z100实测】`gfx906_gptq_wna16`（uint4 W4A16）已在树上；int4 点积指令需先用 `tools/dcu/isa_probe.sh` 确认 | 高 |
| **P2** | **lookup / ngram 增广起草** | 接受率 | +1.5~3%（常规），重复上下文可到 +87% | 【1Cat实测】q7 168.52→自适应 q8 170.98 tok/s；重复上下文 316 tok/s | 中 |
| **P2** | **workspace / 中间量共享** | 显存 | 0.3–1.0 GiB/卡 | 【1Cat实测】prefill score block 24K→16K 省 768 MiB；graph scratch 共享省 1.97 GiB(TP2) | 低中 |

### 0.3 一句话总结

> **1Cat 把 V100 的 attention 从 17.92 TFLOP/s 拉到 60.8 TFLOP/s，靠的不是"更强的 kernel"，而是"让每一拍都有人干活"。Z100 现在的问题是相反的方向：算力空转率极高（decode 3% 带宽利用率、prefill 13% MFU）。所以对 Z100 最有价值的不是 1Cat 的 HMMA 数据流，而是他们对"什么东西不该进热路径"的取舍纪律。**

---

## 1. 为什么 1Cat 的经验对 Z100L 成立

### 1.1 同构性（这是可以借鉴的根本原因）

| 维度 | 1Cat 的 V100 | Z100L | 结论 |
|---|---|---|---|
| 架构代次 | SM70（Volta, 2017） | gfx906（Vega 20, 2018） | 同代 |
| 矩阵核 | **有** HMMA（m16n16k16 / m8n8k4） | **无**（无 HMMA / 无 MFMA） | **最大的差异** |
| bf16 指令 | 无 | 无 | 同 |
| FP8 数据类型 | 无（需模拟） | 无（需 LUT 模拟） | 同 |
| 计算原语 | HMMA + FFMA | `v_pk_fma_f16` / `v_dot2_f32_f16` / `v_dot4_i32_i8`（已用于 `w8a8_sdot4`）；`v_dot8_i32_i4` 待确认 | 不同但同为"上一代" |
| 显存 | 32 GB HBM2 | 32 GiB HBM2 | 同 |
| 带宽 | 900 GB/s | **777 GB/s（实测 D2D）** | 同 |
| fp16 峰值 | 125 TF/s（datasheet） | **24.6 TF/s（理论）/ 19.61 TF/s（实测 SOL）** | Z100L ≈ 1/6 |
| 模型 | Qwen3.8-27B + DFlash2 k=7 | Qwen3.8-27B-FP8 + DFlash2 k=7 | **几乎同一个模型** |
| 拓扑 | 4 卡（NVLink 配对） | 4 卡（**仅 PCIe**，hops=2） | 不同 |

补充一条硬事实（本仓库自己的注释与探测表）：gfx906 上 `v_cvt_f32_fp8`、`v_cvt_pk_fp8_f32`、`v_cvt_f32_bf16`、`v_cvt_pk_bf16_f32`、`v_dot2_f32_bf16` **全部无法为 `-mcpu=gfx906` 汇编**，且**完全没有 MFMA/WMMA**
（[`gfx906_emu_common.cuh`](../../csrc/rocm/gfx906_emu_common.cuh) §why-this-file-exists；`tools/dcu/isa_probe.sh` 把它们列为 `BF16-*`/`FP8-*` 负控制）。
这一点与 V100 不同：V100 虽然没有 bf16/fp8 的**转换/存储**类型，但有 HMMA 可以吃 fp16 操作数。

**关键推论**：两边跑同一个模型、同样的卡数、同样的显存、几乎同样的带宽，但 **Z100L 的算力只有 1/6、互连只有 PCIe**。这意味着：

- **decode 在理论上不该差那么多**。decode 是权重带宽受限的：27B / TP4 ≈ 6.8 GB/步。V100 用 900 GB/s、Z100 用 777 GB/s，理论步时应该只差 16%。而实际是 1Cat 32K 单请求 201 tok/s（≈5 ms）vs Z100 512 上下文 TPOT 45.7 ms —— **差 9 倍**，且 Z100 的上下文还短得多。
- **prefill 差 5 倍是合理的**（算力差 6 倍），**再多的部分是软件损失**。

这就是本报告的核心判断：**Z100 的 decode 问题和 prefill 问题，主要都不是硅的问题，是数据流的问题。**

### 1.2 三个必须承认的硬差异（决定了什么不能搬）

1. **没有矩阵核**。1Cat 几乎所有 attention 收益都来自 HMMA fragment 的复用与调度（BM32 phase kernel、pair-slab、Q/P swizzle、D64 分片）。gfx906 上这些**全部不可移植**，因为根本没有 `mma.sync`。Z100 的等价物是 `v_dot2_f32_f16`（fp16 点积，2 路）和 `v_dot4_i32_i8`（int8 点积，4 路），已在 `csrc/rocm/w8a8_sdot4.cu`、`gfx906_emu_common.cuh` 中使用。
2. **只有 PCIe**。1Cat 的 `tp4_push_gemma_norm`（把 allreduce + 残差 + RMSNorm 融成一个 kernel）依赖全互联 NVLink；Z100 的 WP2 已经证明**自定义 AR 在 e2e 上反而变慢**（c1 TPOT +1.9%/+6.3%，c8 吞吐 −2.8%）。所以这一类要**换形态**：不是替换 AR，而是**融合 AR 的消费者**。
3. **KV 页大小与 chunk 边界耦合**。Z100 用 `--mamba-cache-mode align`（880 token 对齐块），1Cat 用 1648/3296 页；两者前提相同——"KV 页大小决定了哪个 kernel 能被命中、以及 chunk 在哪里被切断"，这一点同构、可以借鉴（见 §4.1 与 §4.3）。

---

## 2. Z100 现状画像（全部为既有实测）

### 2.1 prefill：算力空转 87%

| 指标 | 数值 | 来源 |
|---|---|---|
| 512c1 TTFT p50 | **2722 ms** | E2 verdict |
| 709c1 TTFT p50 | 3743 ms | E2 verdict |
| 32k 冷 prefill | 93.4–93.6 s（≈350 tok/s） | longctx plan §1.1 |
| 其中 `aten::mm` | **1893–2044 ms（43.8%）**，6009 µs/次 | kernel report §2.3 |
| GDN chunked prefill | 675.8 ms（15.6%） | kernel report §2.3 |
| 有效算力 | **≈2.5 TF/s/卡（≈10–13% MFU）** | 【推断】由 27.6 TFLOP/512 tok 反推 |
| 同一物化路径独立实测 | **16.5–17.7 TF/s** | `serve_host_dflash2.sh` 注释 |
| fp16 峰值 | 19.61 TF/s | WP1 SOL |

**这是最刺眼的一处**：同一套"fp8→fp16 物化 + rocBLAS"路径，独立微基准 16.5–17.7 TF/s，进到真实 prefill 只剩 ≈2.5 TF/s。kernel report 明确写了两件事：
- shapes 没记录（`torch_profiler_record_shapes: false`），所以**无法归因**；
- 每次调用 6.0 ms"只与**小 M** GEMM 相容"。

→ **在动任何内核之前，先补一次带 shape 的 profile。** 这是本报告 P0 里的第一个动作，成本极低、信息量最大。

### 2.2 decode：带宽利用率 3%

| 指标 | 数值 | 来源 |
|---|---|---|
| decode step p50 | **240.4 ms** | E2 verdict |
| TPOT p50 | 45.7 ms | E2 verdict |
| 每步权重搬运 | 6.8 GB（fp8, TP4） | kernel report §5.1 |
| 实测权重流速率 | **25.6 GB/s/卡** | kernel report §5.1 |
| 拷贝上限 | **777.5 GB/s** | kernel report §1.1 |
| 带宽下限 | **≈8.8 ms/步** | 【推断】6.8 GB ÷ 777.5 GB/s |
| 有效算力 | 0.4–0.6 TF/s/卡 | kernel report §5.1 |
| 设备时间构成 | rocBLAS 47.2% / GDN 25.8% / fp8 GEMV 4.3% / comm 4.0% / attn 1.4% | kernel report §2.5 |
| allreduce | 700–980 次/例，179–273 µs/次 | kernel report §4.9 |

**结论**：238 ms ÷ 8.8 ms = **27 倍**。而 fp8 GEMV 内核本身在微基准里能跑到 176.5 GB/s（22% 上限），说明**内核不是瓶颈，组合方式才是**：43.8% 的设备时间交给了 rocBLAS 的瘦 GEMM（Tensile 在 M=8 用 `MT32x128x16`、在 M≥512 用 `MT64x32x8`，后者 **K macro-step 只有 8**，实测 16–23 GB/s），而不是 fp8 GEMV 内核。

### 2.3 显存：静态占用 91–95%，余量 1.8–3.0 GiB

| 项 | 每卡 | 说明 |
|---|---|---|
| 模型加载（含 DFlash2 draft） | **9.04 GiB** | 27B-FP8 / TP4 + draft |
| KV 池 | **15.54 GiB（MBT=8192）/ 16.21 GiB（MBT=4096）** | 748,982 / 783,993 token；22.34 KB/token/卡 |
| 实测总占用 | **30.9 GiB（94.5%）**（MBT=8192 实测） | 余量 1.8 GiB；生产已切 MBT=4096（余量 ≈2.6–3.0 GiB） |
| align 页填充浪费 | **≈2.27 GiB（14.6%）** | 880 token 对齐块，15 组 |
| 主机内存 | 31 GiB 总，MemAvailable **6–8 GiB** | 硬约束（page cache 会挤爆） |

**注意**：主机内存只有 6–8 GiB 可用，这一条会**否决** 1Cat 的若干方案（PLE 磁盘/固定主机内存卸载、需要大 pinned 缓冲的预取），见 §6。

---

## 3. 1Cat 技术清单（按 prefill / decode / 显存归类）

> 全部为【1Cat实测】。只列机制与数字，不复述其营销表述。原文档路径见文末。

### 3.1 prefill 侧

| # | 技术 | 机制 | 实测 |
|---|---|---|---|
| **A1** | **FP8-KV 物化桥接** | 不让 FP8 KV 进 attention 热路径：单趟向量化 `fp8_e5m2_paged_kv_to_fp16` gather/expand 到预分配的 784-token FP16 页 workspace，再调已优化的 FP16 稠密 kernel；CUDA op 内无分配、无 host 同步；workspace 跨层共享（256K/Hkv2 约 512 MiB/rank） | **本来 FP8-KV 的 prefill 比 FP16 慢 1.5–2.7×**（128K：450.29 s vs 169.21 s）；桥接后 16K/64K/128K = 14.67/57.99/115.83 ms（**4.52/4.83/4.88×**），expand 本身只占 0.108/0.386/0.764 ms；全模型 64K prefill **−50.75%** |
| **A2** | **对齐块/chunk 解耦** | align 模式原本在**每个 recurrent-state block** 切 chunk，把 8192 token 预算切成很多次完整前向（含 dense 投影 + TP 集合通信）；改为只在 replay 边界 + 共享前缀边界 + 显式周期边界停 | 开 prefix caching 后 8K prefill **5821→2487 tok/s（−57.3%）**；稀疏保留修复后回到 **5356**（32K 5409→3046→**5068**） |
| **A3** | **Mamba state grid 与 KV block 解耦** | `chunk = min(scheduled, mamba_state_block_size)`，而快 prefill 只在某个 `max_seqlen_q` 窗口派发 ⇒ **grid 而非 token 预算决定能否走快路**；修复为显式 `--mamba-block-size` 保留 checkpoint grid，KV 侧不再乘 `ratio` | `--block-size 2048` 时 prefill 2954 tok/s；解耦后 **3336 tok/s 且 KV 池 +23–29%**（1,058,133 token） |
| **A4** | **长前缀 gather→dense + Split-KV3** | 满足 `Q≥4096, KV≥8192, KV>Q` 时把逻辑页 gather 成连续 workspace，调 exact-dense kernel；`KV≥32768` 再三路分区 | D256 split-D：8K 11.13→5.05 ms（2.20×）、64K 87.60→50.45（1.74×）、256K 349.70→210.74（1.66×） |
| **A5** | **BM32 phase kernel** | 每个 M32 group 一个 512-thread CTA：8 个 QK warp 把同一 K fragment 复用于 top/bottom M16，16 warp 各占 D16、把 V fragment 复用于两个 M16 panel | 16K→128K **−19~−20.4%**；eligible warps/cycle 0.479→0.763，L1 global-load 与 L2 read sectors **≈减半**；全模型 prefill −9.74%，TTFT −9.37% |
| **A6** | **token-shard consumer projection** | 消除 `[8000,5120]` 物化 all-gather：reduce-scatter + RMSNorm 只产出本 rank 的 `[2000,5120]` shard，后续投影先吃本地 shard 再吃 peer shard | 目标 8000 token wall ≤1.4545 s（5500 tok/s），基线 1.562 s（5121 tok/s），**尚未落地**；trace 显示 gate/up 30.3% + QKV 21.0% + collective+norm 17.7% + down 14.1% = **83%**，所以不能只盯 attention |
| **A7** | **per-shape 派发校准** | 用真实形状表决定 tile/路由，而不是单一阈值（"dense prefill 阈值 256 行，更小走 TurboMind"） | 明确否决"WMMA 融合反量化+GEMM"：512–8192 行上**慢约 4 倍** |

### 3.2 decode 侧

| # | 技术 | 机制 | 实测 |
|---|---|---|---|
| **B1** | **单体 decode GEMM（QPN8）** | 把 checkpoint 原生 block-FP8 repack 成单个 `[K,N]` code 张量 + 分组 FP16 scale（替换而非并存，省 ~6 GiB/rank 重复）；gate/up split-K=8 双累加链，down split-K=16；**M 判定放在不透明 C++ 算子内部**，防止 AOTInductor 把 M=1 分支特化到后续 prefill；M>8 落到共享 85 MiB FP16 workspace 调 FP16 GEMM | TPOT **16.985→15.808 ms（−7.45%）**；<25% 占用率的服务时长 **93.56%→48.56%**；NCU DRAM 82.7–87.7%、tensor 9.5%；**launch 数 1085.5→1085.9 不变** ⇒ 收益来自 kernel 服务效率，不是 launch gap |
| **B2** | **M8 精确配对 GEMV** | 每个 warp 同时算 gate/up 两个投影并复用 A 寄存器，split-K=8 单累加链；保留反量化/partial sum/舍入/SiLU 语义 | 159.442→155.986 µs（−3.456 µs）；**kernel 数 606→606 不变**；未达 12 ms 整轮目标 |
| **B3** | **FP16 grouped verify** | 每请求把 ≤8 query 行打包，这些行**共享每个 KV tile**；保留 FP32 softmax 概率/PV 算术/归约顺序；用 device 端行长度保 causal 边界 | 32K：192.5 vs 743.4 µs（**3.9×**）；按真实 trace 16 次调用省 **2.7–3.0 ms**；rel-L2 <2.13e-4 |
| **B4** | **混合批 decode 行分组** | 混合批（prefill chunk + 常驻 verify）里按 request-major 每 8 行成组，走与均匀验证批相同的分组算子；每行保自己的因果长度，补零行算子定义为零输出；一次 launch 最多 16 组；行长度由 device 权威 `seq_lens` 推导 | 每层：32K **6.01→1.10 ms**（均匀批 1.37）、131K **22.70→4.08 ms**（均匀批 6.93）= **5.5×**；chunk=1024 时常驻吞吐均值 **+56%** |
| **B5** | **mixed-prefill step 预算** | 有常驻 decoder 时为其**预留 token**并限制同一 step 的 prefill 总量；默认 step 延迟目标 **250 ms**，用已完成的 GPU 测量自校正；纯 decode 仍走 FULL graph | 常驻吞吐 **6.88→30.34 tok/s（4.41×）**、最长更新间隔 **2.873→0.310 s（−89%）**；代价：新请求 TTFT +26%/+72%；纯 decode C1 +0.29%/C8 −1.33% |
| **B6** | **CI/push + norm 融合** | TP4 allreduce 与 Gemma 残差归一化融成**一个 kernel**（NVLink 全互联前提） | 无独立整轮数字；同文档的 M8 norm packet 化省 3.456 µs/次 × 48 层 ≈ 0.166 ms |
| **B7** | **QA 门：K4 tile 不可用** | 24 种 `m8n8k4` K4 排列中只有 `[0,1,2,3]` 与原生 `m16n16k16` 逐位一致；raw HMMA K4 反而 QK +18.2%/PV +27.5% | 负结果，用于校准"数学等价 ≠ 性能等价" |

### 3.3 显存侧

| # | 技术 | 机制 | 实测（每卡） |
|---|---|---|---|
| **C1** | **共享权重 / code / scale** | 去掉兼容 layout 上的重复 QPN2 code 与持久 TurboMind FP16 scale | 模型分配 **10.07→6.47 GiB**，可用 KV 预算 10.57→**12.68 GiB**，逻辑 KV 922,965→**1,107,285 token** |
| **C2** | **共享 graph scratch / draft 快照** | 按行容量分组共享 decode workspace（避免"长上下文容量 × 短请求批"的叉积）；DFlash aux 快照直接存成 draft 投影消费的 dtype，消除 800 MiB FP32 concat + 400 MiB cast | TP4 非 KV 活跃分配 **10.225→8.794 GiB**，KV 池 15.016→**17.406 GiB**；TP2 对应 17.117→15.146 |
| **C3** | **prefill score block 缩小** | `block_tokens * 8192 * 6 * 2 B` 每 device；24K→16K→8K | 24K→16K：2.25→1.50 GiB（**−768 MiB**）；再降到 8K 省 1.50 GiB 但延迟 **+2.0–2.26%**，故 16K 为默认 |
| **C4** | **repack 替换并存** | 新量化布局**替换**旧张量而非并存 | 避免 ~6 GiB/rank 重复 |
| **C5** | **E4M3 KV 本身** | 容量 ×2、读带宽 ÷2（配合 A1 桥接避免 prefill 回退） | 与其 attention 数字同源；1Cat 发布档默认 E4M3 target KV |

---

## 4. 移植判定矩阵（本报告核心）

> 判据：**Z100 上是否已有等价物 / 是否需要矩阵核 / 是否需要 NVLink / 是否受主机内存限制**。

### 4.1 直接可移植（不需要矩阵核，收益路径清晰）

| 1Cat 项 | Z100 现状 | 移植形态 | 预期 | 风险 |
|---|---|---|---|---|
| **A1 FP8-KV 桥接** | `--kv-cache-dtype fp8` 开关已存在（`serve_host_dflash2.sh` 的 `KV_DTYPE`），Triton unified attention 已有 fp8 读路径 + 每张量 descale；但**生产未启用**，因为按当前做法会掉精度 | 抄"桥接"而不是抄"格式"：prefill/decode 热路径前把 FP8 KV gather→dequant 成 FP16 workspace（Z100 已有 LUT 精确 fp8→fp16），再调现有 FP16 路径 | KV 容量 **≈×1.9**；长上下文 ITL **−20~35%**（Z100 自己的 L4 估算） | **精度**：当前实现 per-tensor scale=1.0，必须换 per-token-per-head + 校准；1Cat 也是靠 per-token-head + FP32 累加才过门 |
| **A2/A3 chunk-grid 解耦** | 生产用 `--mamba-cache-mode align`，attn block 抬到 880，MBT 4096 ⇒ **一次 4096 token 的 prefill 可能被切成 ~4.6 段** | 抄"稀疏保留边界"：只在 replay 边界 + 共享前缀边界切；并让 mamba grid 与 KV block 解耦 | prefill **+40~110%**；KV 池 **+23~29%** | 调度器改动面大；必须验前缀缓存命中率不退化（1Cat 的 1648+8192 反例：前缀缓存**静默归零**） |
| **B3 grouped verify** | 已有 `VLLM_GFX906_ATTN_VERIFY_3D=1`（把 ≤8 query 路由到 3D split-KV），方向一致 | 补"共享 KV tile"的语义：让 8 个 verify 行在同一个 KV 段上被一次读入（Triton 里把 `BLOCK_M` 覆盖整段 verify 行，而不是按行切块） | 32K verify 每层 **3–4×** | 因果边界要保持；Z100 已有 `ATTN_FORCE_CAUSAL=0` 的教训 |
| **B5 mixed-prefill 预算** | **完全缺失**。WP5 结论："37 个 CLI 组合都撞吞吐悬崖 ⇒ 唯一路径是调度器 decode 预留（Patch A/B）" | 直接实现 250 ms step 预算：为常驻 decoder 预留 token，限制同 step prefill 总量，用实测自校正 | c8 尾延迟 p99 **5014 ms → 1.5 s 量级**；常驻吞吐 4.4× | 新请求 TTFT 变差（这是明确接受的 trade-off，要写进验收口径） |
| **C1/C2/C4 内存去重** | 有：`fp8_dequant_to_fp16_out` 是**一次性**（256 次，28.3 ms），说明没有每步全权重量化；materialization scratch 已是"单块 grow-only" | 审计"同形状多份 workspace"：3D prefill 的 `SEGMENTS=4` 段缓冲、verify 3D 缓冲、逐 shape 的 graph 缓冲是否并存 | 0.3–1.0 GiB/卡 | 低 |
| **C3 score block 缩小** | Triton 3D prefill 有分段部分输出 | 把 prefill 的段缓冲按 `block_tokens` 上限分配并跨层复用 | 数百 MiB | 低 |

### 4.2 需要换形态才能移植

| 1Cat 项 | 为什么不能照搬 | Z100 的等价形态 |
|---|---|---|
| **A5 BM32 phase kernel** | 依赖 HMMA fragment 所有权与 `ldmatrix` 式共享内存布局 | Triton 层面：把 `BLOCK_M` 从 32 提到 64/128，让 K/V tile 在更多 query 行间复用；或写 HIP 内核用 `v_dot2_f32_f16` 做 4×4 点积阵列 |
| **B1 QPN8（NVFP4/FP8 解码）** | NVFP4 走的是 4-bit codebook + FP16 反量化，依赖 FP4 张量核路径 | **形态相同、格式不同**：Z100 已确认有 `v_dot4_i32_i8`（int8，4× fp32 率）与 `gfx906_gptq_wna16`；int4 的 `v_dot8_i32_i4` 在 `isa_probe.sh` 探测表内但**仓库无归档 PASS 结果，须先验** ⇒ 做"int4 权重 + fp16 激活"的单体 decode GEMV，M≤8 全部由它服务，rocBLAS 只在大 M 出现 |
| **B6 allreduce+norm 融合** | 依赖 NVLink 全互联；Z100 的 WP2 已证自定义 AR e2e 变慢 | **融合消费者而非替换 AR**：Z100 已有 `_C::gemma_fused_add_rms_norm_gfx906`（33.2 ms，0.77% 设备时间）；把它与 AR 的收尾合并成一次 kernel，或至少让 AR 的 output 直接写进 norm 的输入布局（省 `aten::copy_`：Z100 每例 140–173 ms、3.25–5.66%） |
| **A6 token-shard projection** | 依赖 CUTLASS 与 SM70 专属 mixed-dtype residual 语义；Z100 是 rocBLAS | 等价物是"消除 `[M,5120]` 的 all-gather 物化"：TP4 上 allreduce 已占 4.0%，但 gather 物化在 prefill 里更贵；可用 Z100 已有的 `VLLM_DCU_CUSTOM_AR` 骨架改成 reduce-scatter + 本地 shard 先行 |
| **B7 K4 tile 负结果** | 用不上 HMMA | 但要继承它的**方法论**：任何"数学等价"的 tile 改组都要有 bitwise 门；Z100 已经踩过（`VLLM_DCU_FP8_STREAM_MAX_M` 更快但破坏 bit-exact greedy parity） |

### 4.3 不能移植（或明确不划算）

| 1Cat 项 | 原因 |
|---|---|
| 全部 HMMA/SM70 CUDA 内核（Flash-V100、XQA、split-D、pair-slab） | gfx906 无矩阵核，ISA 层不可达 |
| NVFP4 / MXFP4 张量核路径 | 无 FP4 张量核 |
| 128-bit XQA load 优化 | 收益来自 SM70 的 `half8`→`uint4` 合并与 L1 请求数下降；gfx906 的 load 路径与 Triton 自动向量化不同，需重测而非照搬 |
| PLE 磁盘/固定主机内存卸载 | **Z100 主机只有 6–8 GiB 可用**，page cache 已经会挤爆 `hipErrorOutOfMemory`（`.r3` 记录过 5 次连续失败） |
| GGUF / IQ3 codebook 内核 | 与当前 FP8 checkpoint 无关 |
| CUDA Graph + breakable 路线 | Z100 已实测 breakable 是**数值坏 + VMFault**，判死；但 **piecewise 路线 A 是正收益**（见 P1） |

---

## 5. 推荐路线图

### P0（1–2 周，先测量、再做高杠杆低风险项）

**P0.1 带 shape 的 prefill profile（0.5 天，最高信息量）**
- 动作：重跑一次 `512/709/2k/8k` 探测，开 `torch_profiler_record_shapes: true` 与 `ITER_DETAILS=1`。
- 要回答的三个问题：
  1. prefill 的 `aten::mm` 到底走的是 **fp16（16.5–17.7 TF/s）** 还是 **bf16（3.42 TF/s）**？若是 bf16，为什么（draft 是 bf16；`--dtype float16` 下是否有张量漏成 bf16）？
  2. 一次 4096-token prefill 里到底发生了几次"整层前向"（数 `first materialization route engaged` 的次数 / 数 NVTX 的 forward range）？这直接验证 A2/A3 的 chunk 切分假设。
  3. GDN chunked prefill（`fused_sigmoid_gating_delta_rule_gfx906_prefill`，14.08 ms/次 × 48 层 = 676 ms）的 M 是多少？是否被 chunk 切小了。
- 判据：能给出"prefill 2.72 s = A(GEMM) + B(GDN) + C(attention) + D(空转)"的闭合账，且各项与 kernel report 的 43.8%/15.6% 对得上。

**P0.2 FP8 E4M3 KV（L4 立项，1–2 周）**
- 抄 A1 的**结构**：热路径永远是 FP16 KV；FP8 只作为存储格式，进 attention 前由一次 gather+dequant 变成 FP16 workspace。理由：1Cat 的教训是"通用 FP8 kernel 在每个 query CTA 里重复转换同一段 KV"（occupancy 25%、tensor activity 4%），在 gfx906 上同样会犯。
- 必须项：per-token-per-head scale（amax/448）+ `--calculate-kv-scales` 校准；GDN 状态不量化。
- 门禁（沿用 Z100 的 L4 设计，补一条 1Cat 的口径）：
  1. 8 条固定 prompt 的 PPL / logit KL vs fp16-KV **≤1%**；
  2. 1000 步 decode 无 NaN/Inf（含 64k 驻留）；
  3. 池容量 ×1.8 **或** 长上下文 ITL −20%（至少一项 e2e 成立）；
  4. **新增**：prefill 不得回退 >3%（防 1Cat 那个 1.5–2.7× 的坑）。
- 预期：KV 池 783,993 → **≈1.4–1.5M token**；32k decode 每步 KV 读 0.5 GB → 0.25 GB/卡。

**P0.3 chunk-grid 解耦（1 周）**
- 先做**只读验证**：把 `--mamba-cache-mode align` 换成 `none`/调 `--mamba-block-size`，量 prefill 吞吐与 KV 池的变化（这正是 1Cat A3 的 A/B 形态：2048+8192 → 3336 tok/s、KV +23–29%）。
- 再做稀疏保留边界（A2 的机制）：只在 replay / 共享前缀 / 周期边界切 chunk。
- 硬约束：`grid % block == 0`，否则前缀缓存会**静默失效**（1Cat 记录：同一 232K prompt 86.5 s vs 命中约 1 s）。

### P1（2–4 周，decode 结构性改造）

**P1.1 重新启用 CUDA Graph 路线 A（2–3 天）**
- 依据：Z100 自己已实测"路线 A（去 `--enforce-eager` + `cudagraph_mode=piecewise`）在①②修复后是 decode 快 7–9%、prefill 持平、c=4 不回归；compile 下 512-token TTFT 3.35→1.35 s"。
- 为什么现在做：1Cat 的**每一个**性能合同都带 `FULL_AND_PIECEWISE CUDA Graph`；decode 想从 238 ms 往下走，图模式是前提而不是可选项。
- 门禁：沿用 WP0 的 G-0..G-4，但把 c8 尾延迟纳入硬门（WP0 当年正是 c8 −50% 触发关闭）。

**P1.2 单体 decode GEMM（3–4 周，最高天花板）**
- 目标：把 decode 的 43.8% `aten::mm` 从 rocBLAS 瘦 GEMM（16–23 GB/s）迁到 fp8/int4 GEMV 家族（微基准 176.5 GB/s）。理论步时上限 8.8 ms，现实目标先定 **≤100 ms/步**。
- 抄 B1 的三条工程纪律：
  1. **M 判定放在不透明 C++ op 内部**（Z100 已经踩过这个坑并在路线 A 里修好，直接复用）；
  2. **repack 替换而非并存**（否则 +6 GiB/rank 重复）；
  3. **launch 数不变也要做**——1Cat 的 QPN8 证明收益可以完全不来自 launch。
- 量化格式选择：gfx906 已确认有 `v_dot4_i32_i8`（int8，4× fp32 率）；int4 的 `v_dot8_i32_i4` 须先用 `isa_probe.sh` 确认可汇编并实测吞吐，再决定是否作为主路径（若不可用，退回 int8 + 更大 tile）。`gfx906_gptq_wna16` 已存在。**int4 权重**是唯一能让 decode 权重流量 ÷2 的选项（6.8 → 3.4 GB/步），优先级高于继续优化 fp8 tile。

**P1.3 混合批 decode 行分组 + mixed-prefill 预算（2 周）**
- B3+B4+B5 是一条线：先让 verify 行共享 KV tile，再让混合 step 有延迟预算。
- 这是 Z100 WP5 已经用 37 组 CLI 证明"CLI 层无解"之后**唯一剩下的路径**，1Cat 的 4.41× / −89% 是直接对口证据。

### P2（4 周+，收益明确但成本高）

- **W4A16 / int4 权重**：9.04 → ~5.5 GiB/卡，KV 池 +3.5 GiB；同时是 P1.2 的前提。
- **lookup/ngram 增广（B7 类）**：针对 Z100 最差工况（8k 上下文接受率 14.3%、接受长度 2.0）。1Cat 的口径很克制：常规只 +1.46%，重复上下文才有大收益；且**默认关闭**（低命中时每轮多付 0.1–0.3 ms）。建议照抄这个"默认关 + 自适应"策略，只对高重复场景开。
- **workspace 去重（C1/C2/C3）**：先做一次 allocator 快照审计，再按 1Cat"按行容量分组"的方式共享。
- **compile cache 默认开**：1Cat 首启 271 s → 重启 95 s；Z100 的 `torch.compile` 是 77–81 s/worker，迭代速度直接受益。

---

## 6. 负面清单：不要照搬的 8 件事

1. **不要移植任何 HMMA 内核**。ISA 不可达。
2. **不要把"自定义 allreduce"当成 decode 优化**。Z100 WP2 已证 e2e 变慢（`z100-vllm/.r3/WP2/RESULT.md`）。
3. **不要让 FP8 KV 直接进 attention kernel**。必须走桥接（A1）；否则重蹈 1.5–2.7× 回退。
4. **不要用 per-tensor scale=1.0 开 FP8 KV**。当前 `KV_DTYPE` 开关就是这么做的，只能做功能验证、不能上生产。
5. **不要引入主机内存依赖的方案**（PLE 磁盘、大 pinned 预取）。Z100 主机 MemAvailable 只有 6–8 GiB，且 page cache 已造成过 OOM 事故。
6. **不要在 gfx906 上用 bf16 跑热路径 GEMM**。gfx906 无 bf16 指令，bf16 GEMM 实测只有 3.42 TF/s（fp16 的 1/5.7）。DFlash2 draft 必须 bf16 是**已知的数值约束**（fp16 会溢出 MLP 激活），但要想办法把 draft 的**投影**留在 fp16 或量化路径上。
7. **不要为单个负载重启服务做 A/B**。1Cat 也是"一次启动跑多组"。
8. **不要相信 standalone 微基准能预测 e2e**。Z100 已经两次被反转（cfg3 standalone +6% → e2e TTFT +13~19%；custom AR 微基准 +21~53% → e2e 变慢），1Cat 也有同类记录（QPN8 未默认开启、C1 SiLU/down 融合算子级更快但 e2e 无收益）。

---

## 7. 验收口径（建议直接沿用双方已验证的门禁）

| 维度 | 门禁 | 来源 |
|---|---|---|
| prefill | TTFT p50 512/709 ≤2.5 s（Must）≤1.8 s（Target）；**且 prefill 吞吐不得因 KV 格式回退 >3%** | Z100 verdict + 1Cat A1 |
| decode | step p50 ≤210 ms（Must）≤200 ms（Target）；**c8 ITL p99 ≤1.5 s** | Z100 verdict + WP5 |
| 接受率 | 接受长度 ≥4.0、接受率 ≥45%；**8k 上下文单独记一条基线**（当前 2.0 / 14.3%） | Z100 verdict |
| 精度 | greedy bit-exact（除非显式声明改数值）；KV 量化走 PPL/KL 门 | 双方一致 |
| 显存 | KV 池 token 数、静态占用 %、余量 GiB 三项都必须报 | Z100 口径 |
| 统计 | 每负载 2 次重复 + warmup 2 次；p50 差异 <5% 且区间重叠 → 判"不可判定" | Z100 施工图 §1.3 |

---

## 8. 本次调研的局限（请连同结论一起读）

1. **1Cat 的数字全部是引用，不是复现**。其文档合同很严格（每项都带 GPU 型号/TP/上下文/批大小/KV dtype/量化路径/graph 模式/采样策略/接受长度），但未在本机验证。
2. **1Cat 源码只做到"文档级 + 目录级"分析**。到 GitHub 的带宽约 53 KB/s，完整 clone（RPC 失败）与 tarball（超时）都没拿到；本文的 1Cat 依据是 README（1363 行）+ 28 份 `docs/design/*` 原文。**"机制"层可信，"逐行代码细节"层没有核对**。
3. **Z100 的 prefill 归因尚不闭合**。kernel report 明确写了 shapes 未记录（`torch_profiler_record_shapes: false`），所以 §2.1 的"2.5 TF/s vs 16.5–17.7 TF/s"是**反推**而非测量。P0.1 就是为补这个洞。
4. **A2/A3 的 chunk 切分假设未验证**。本仓库只确认了 `mamba_cache_mode == "align"` 会把 attn block 抬到 880、且注释写明"Internal prefill checkpoints land on block boundaries"，但**没有确认一次 4096-token prefill 实际发生几次整层前向**。若只有 1 次，A2/A3 的收益预估要大幅下调 —— 这正是 P0.1 的第二个问题。
5. **`v_dot8_i32_i4` 未确认**。`tools/dcu/isa_probe.sh` 把它列在探测表内，但仓库里没有归档的 PASS 结果；int4 路线（P2）的前提需要先补这一次探测。

---

## 9. 参考

**1Cat-vLLM（外部）**
- 仓库与 README：<https://github.com/1CatAI/1Cat-vLLM>
- 发布：<https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.1>
- 本文引用的设计文档（`docs/design/`）：`sm70_flash_v100_prefill_operator_optimization.md`、`sm70_flash_v100_fp8_kv_long_context.md`、`sm70_fa2_d256_prefill_pipeline.md`、`sm70_mamba_state_grid_decoupling.md`、`sm70_flash_next_prefix_prefill.md`、`sm70_mixed_batch_decode_rows.md`、`sm70_fp16_grouped_verify.md`、`sm70_qwen38_qpn8_decode.md`、`sm70_m8_exact_decode.md`、`sm70_memory_defaults.md`、`sm70_memory_arena.md`、`sm70_memory_workspace_tuning.md`、`mixed_prefill_latency.md`、`sm70_dflash2_lookup_augmented.md`、`sm70_dflash2_ngram_hybrid.md`、`sm70_quantized_batch_reuse.md`、`sm70_qwen38_fp8_prefill_5500.md`、`sm70_release151_compile_cache.md`

**Z100 本仓库**
- [`docs/result/z100-rocm101-perf-verdict.md`](../result/z100-rocm101-perf-verdict.md)
- [`docs/result/z100-rocm101-perf-baseline-freeze.md`](../result/z100-rocm101-perf-baseline-freeze.md)
- [`docs/design/z100-longctx-vram-optimization-plan.md`](z100-longctx-vram-optimization-plan.md)
- [`docs/design/z100-s24-l4-kv-quant-design.md`](z100-s24-l4-kv-quant-design.md)
- [`docs/design/z100-s24-l3-dynamic-k-design.md`](z100-s24-l3-dynamic-k-design.md)
- [`docs/result/z100-cudagraph-route-a-20261021.md`](../result/z100-cudagraph-route-a-20261021.md)
- [`KERNEL_REPORT_20261024-053333-E2C.md`](../../KERNEL_REPORT_20261024-053333-E2C.md)
- [`tools/dcu/serve_host_dflash2.sh`](../../tools/dcu/serve_host_dflash2.sh)（生产配方）
- [`.r3/STATUS.md`](../../.r3/STATUS.md)（三/四轮工作包状态）
