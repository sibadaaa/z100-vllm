# PERF1 实测结果

> 本文件由 Lead 维护；每个数字都必须有原始日志路径与复现命令。
> 证据目录：`/mnt/game/perf1/`（容器内可读）；测试报告基线：`/home/lyh/vllm-test/results/`。

## 0. 基线复核（Lead，2026-10-13）

### 0.1 逐 kernel 复算（`tools/dcu/perf1_trace_kernels.py`）

对 `rank0.1791856344670783769.pt.trace.json.gz`（50 个稳态 decode step + prefill）复算：

| kernel | 调用 | 总 ms | µs/次 | 占比（每 step，219.6 ms） |
|---|---:|---:|---:|---:|
| `fp8_emu_gemm_block_kernel<32,64,4,4>` | 13000 | 4886.1 | 375.9 | **44.4 %** |
| `fp8_emu_gemm_block_kernel<64,64,4,4>`（prefill） | 520 | 2263.6 | 4353.0 | — |
| `fused_sigmoid_gating_delta_rule_gfx906_mtp` | 2400 | 1719.8 | 716.6 | **15.7 %** |
| `nccl AllReduce Ring_Simple_f16` | 3963 | 1464.8 | 369.6 | **22.3 %**（与 LL 合计） |
| `nccl AllReduce Ring_LL_f16` | 3057 | 1013.7 | 331.6 | |
| `fused_sigmoid_gating_delta_rule`（prefill） | 96 | 943.6 | 9829.2 | — |
| `triton_matmul_kernel` | 2658 | 742.1 | 279.2 | 6.7 % |
| `kernel_unified_attention_2d` | 884 | 413.6 | 467.9 | 3.8 % |
| `fp8_emu_gemv_block_kernel<1>`（MTP draft，M=1） | 208 | 55.4 | 266.3 | 0.5 % |

调用结构：每 step = **一次 M=3 的整模型前向**（256 个 linear，走 tiled GEMM）+ 少量 MTP head 调用。
`fp8_emu_gemm_block_kernel` 的 grid.x 直方图 = {56: 850, 64: 2400, 80: 6500, 136: 3250}
⇒ decode 形状的 N ∈ {3584, 4096, 5120, 8704}。

### 0.2 访存 roofline 复核（`tools/dcu/perf1_weight_bytes.py`）

该 kernel 每次调用读取整块 [N, K] 权重，`grid.x = ceil(N/64)` 可反推 N（K 取 5120 上界）：

| 量 | 值 |
|---|---|
| decode 路径总读取量 | **381.2 GB** |
| kernel 墙钟 | **4.886 s** |
| **有效带宽** | **78.0 GB/s** = 规格 670 GB/s 的 **11.6 %** |
| 按实测上限 776 GB/s 折算 | 491 ms ⇒ **9.8 ms/step** |
| 按探针诚实参考 621.7 GB/s 折算 | 614 ms ⇒ **12.3 ms/step** |
| 现状 | **97.5 ms/step** |

⇒ 主 kernel 有 **~8×** 的访存空间；每 rank 每 step 的权重流量 7.6 GB 是**不可避免**的，
问题只在并行度（报告 §4.2：216 VGPR → ≤2 blocks/CU、68 blocks 对 60 CU、occupancy ~11 %）。

### 0.3 微基准基线（`tools/dcu/s18_fp8_block_mprobe.py`，DCU0，服务在跑）

| 形状 (N×K) | 路径 | M=1 | M=2 | M=3 | M=4 |
|---|---|---:|---:|---:|---:|
| 4352×5120 | gemv(64 thr) | 280 | 563 | 808 | 1022 |
| 4352×5120 | gemm | 355 | 356 | 355 | 355 |
| 1280×5120 | gemv(64 thr) | 83 | 158 | 230 | 297 |
| 1280×5120 | gemm | 343 | 344 | 344 | 343 |

**结论**：GEMM 在 M 与 N 两个方向都近似平坦（证明 H2：多算行几乎免费；也说明单 block 的
K 串行循环是成本主体）；GEMV/GEMM 的优劣**随形状反转**（证明 H3：全局阈值派发是错的）。

---

## 1. L2 通信：22 % 的 NCCL 里只有 ~5 % 是真通信

`tools/dcu/perf1_comm_skew.py`（comm-opt，基于 4 个 rank 的 atom pickle，50.14 个 decode step、
7020 次集合通信 = 140.0 次/step）：

| 量 | 值 |
|---|---|
| 关键路径上的**固有**集合通信（各 rank kernel 时长的 min = 最后到达者的尾巴） | 中位 72 µs / 均值 79.5 µs ⇒ **10.1–11.1 ms/step = 4.6–5.1 %** |
| rank0 的 AllReduce 忙时 | 2474.8 ms = 49.4 ms/step，其中 **38.2 ms/step（77 %）是自旋等待** |
| rank2 的 AllReduce 忙时 | 1522.6 ms = 30.4 ms/step，其中 19.2 ms/step 是等待 |
| 判决性证据 | 单次集合通信里 4 个 rank 的**结束**时间中位相差 **4 µs**，而**开始**时间中位相差 **313 µs** |
| 恒等式校验 | `wait_r(i) == max_s gap_s(i) - gap_r(i)` 误差 0.2 % |

⇒ 报告 §7.2 的"22 % NCCL"应改写为 **~5 % 真通信 + ~17 % 等慢卡**。混插卡（2×Z100L + 2×Z100）
的同步代价按均值口径是 **18.9 ms/step（8.6 %）**，中位口径 7 ms/step；**任何 RCCL 环境变量都无法
触及这一项**，只有硬件同型配对才行 —— 而 TP=2 × DP=2 会让每卡权重流量翻倍，在访存受限的
decode 路径上是净亏（见计划 H5 的算术）。

`NCCL_PROTO=LL` 的上限因此只有 (107−75) µs × 140 = **4.5 ms/step（2 %）**，且它在 ≥1 MiB 反而更差
（172.7 vs 126.4 µs），而 prefill 的 all-reduce 是 [2048,5120] f16 ≈ 20 MiB —— 所以只有在
大消息惩罚可忽略时才可采纳（comm-opt 的 matrix2/chain 实验判定中）。

**已证伪（不要再做）**：custom all-reduce（平台结构性未启用，收益 0）、
`NCCL_LL_THRESHOLD=131072`（861 µs，反而慢）、RCCL ALGO/CHANNELS 矩阵（无 separability）。

## 2. L2 GDN：16 % 的根因是**访存不合并**，不是 grid 太小

`fused_sigmoid_gating_delta_rule_gfx906_mtp` 48 次/step × 716.6 µs。gdn-opt 的独立复现与定位：

| 项 | 值 |
|---|---|
| 生产形状下的复现 | **704–715 µs/call**（与 trace 的 716.6 吻合） |
| 根因 | state cache 布局 `[slot][head][v][k]`（k 连续），同一线程 k 循环连续、但**相邻 lane 相隔 512 B** ⇒ 访存完全不合并 |
| 反证 | 同一 kernel、同一 grid，仅换成 v 连续的视图（数值不对，只测性能）：**24–27 µs** |
| 为什么 `mtp_v2` 没生效 | 它要求 `state.stride(2)==1`，生产实际 `stride(2)=128`（`MTP_V2_DEBUG` 实测 `MISS state_s3=1 state_s2=128`） |
| 新 Triton kernel（保持生产布局，lane 映射到一行的 k 轴，grid = 384–1536 programs） | **704.6 → 108.7 µs/call**（事件计时） |
| 逐算子一致性 | vs C++ op：max\|Δout\|=7.6e-06、max\|Δstate\|=1.8e-07（fp16 舍入量级） |
| 预期收益 | 34.4 → ~5.2 ms/step（**−13 %**），开关 `VLLM_QWEN35_MTP_GDN_TRITON=1`（默认关） |

⚠️ 该收益尚未通过 e2e 贪心 token 一致性验收（本模型 48 层递推 + 历史上有 S18e 的 parity 事故），
验收判据见 §4。

## 3. 附带的算子级发现（Triton 回退 6.7 %）

用 grid 算术 + kernel 邻域 + 模块结构三重确认的三个模块：

| 模块 | grid | 次/step | µs/次 | ms/step | 有效带宽 | 判定 |
|---|---|---|---:|---:|---:|---:|---|
| GDN `in_proj_ba`（per-rank N=24, K=5120） | (1,1,1) | 48 | 186.2 | **8.94** | 1.3 GB/s | **GO**：单个 program 跑 80 个 K 迭代、[64,64] tile 对 N=24 浪费 ~10× |
| MTP `mtp.fc`（N=1280, K=10240） | (20,1,1) | 2 | 364.2 | 0.74 | 72 GB/s | GO（小） |
| `lm_head`（N=62080, K=5120, fp16） | (970,1,1) | 3 | 1654.5 | 5.16 | **384 GB/s = 59 % 峰值** | **NO-GO**：已接近 roofline；除非改成 FP8 权重（需改 quant config / checkpoint） |

`lm_head` 在 `modules_to_not_convert` 里，checkpoint 是 BF16 [248320,5120] → 服务端 FP16 =
每 rank 每次调用 636 MB。把它量化成 FP8 可减半字节，属独立课题。

---

## 4. L1 端到端基线与 MTP 深度（verifier，task-4）

口径：chat completions、temp=0、ignore_eos、并发 1、每点 3 次；`so md5 = fbfd45edb007`；
**生产配方但去掉 `--profiler-config`**。JSON：`/mnt/game/perf1/{control,k3}_e2e.json`。

| 真实 prompt | 控制组(k=2) decode t/s | k=3 decode t/s | 变化 |
|---|---:|---:|---:|
| 119 | 10.60 | 11.38 | +7.4 % |
| 451 | 13.93 | 13.68 | −1.8 %（噪声内） |
| 1806 | 11.85 | 14.38 | **+21.3 %** |
| 7443 | 7.78 | 10.49 | **+34.8 %** |
| 四点均值 | 11.04 | 12.48 | **+13.0 %** |

机制：step 时间只涨 5.6–8.7 %（MM 对 M 平坦），而 tok/step 涨 10–41 %（2.526→3.556 @8192），
上下文越长收益越大 —— 生产 maxlen 262144，长上下文正是目标场景。

**冷启动**：新引擎首请求 p7443 的 TTFT = **29656 ms（251 t/s）** vs 稳态 7089 ms（1050 t/s）
⇒ **4.18×** JIT 惩罚；服务就绪还需 ~172–200 s。⇒ 启动预热应写进生产配方。

**一致性**（`tools/dcu/perf1_e2e_parity.py`，raw `/mnt/game/perf1/verify_*.json`）：

| 对比 | 结论 |
|---|---|
| k=0 vs k=0（两个独立实例） | **逐位一致**（logprob 差 0.0）⇒ 测试夹具与确定性成立 |
| k=2 vs k=0 | p512 第 14 token 起分歧（2/64）；p2048 第 59 token 起（5/64） |
| k=3 vs k=0 | p512 第 14 token 起（7/64）；**p2048 与无投机贪心逐位一致** |
| k=3 vs k=2 | 两个 prompt 都在第 59 token 起分歧，文本不同 |

⇒ 与归档的 S18e 结论一致：MTP 本身就会改变 token 流（非新增回归），k=3 在长 prompt 上比 k=2
**更接近**非投机基准，但在 p512 上更远。交付时按"k=3 长上下文更优、短上下文等价"陈述。

---

## 5. L3 内核：流式 decode GEMV（+ 新的 GEMM tile）—— 本程序最大单项收益

现状：decode 的 260 次/step 调用走 tiled GEMM（`VLLM_DCU_FP8_GEMV_MAX_M=1`），
在每个形状上都是 ~344–709 µs。kernel-opt 新增 `fp8_emu_gemv_stream_block_kernel`
（warp-per-row、512 线程、`uint4` 合并流式读 W、A 一次性展宽进 LDS、按 block_k 折叠 scale、
warp shuffle 归约），并把 GEMM 的 tile 也重新调参。实测（s18/perf1 probe，K=5120，`--check`）：

| N×K | M | 旧 GEMV | GEMM cfg0 | **最佳 GEMM** | **stream v1** | vs 最佳 GEMM |
|---|---:|---:|---:|---:|---:|---:|
| 1280 | 1 | 93.2 | 343 | 245 (cfg5) | **26.2** | **9.3×** |
| 1280 | 3 | 236.5 | 344 | 245 | **37.6** | **6.5×** |
| 3584 | 3 | 699.9 | 345 | 247 | **57.5** | **4.3×** |
| 4096 | 3 | 746.9 | 355 | 319 | **98.7** | **3.2×** |
| 5120 | 3 | 955.7 | 366 | 321 | **99.7** | **3.2×** |
| 8704 | 3 | 1547.8 | 709 | 415 | **149.8** | **2.8×** |
| 4352 | 3 | 806.1 | 356 | ~247 | **~100** | **2.5×** |

（单位 µs；全部 rel_err = 0.00e+00，row-consistency OK；`tests/kernels/quantization/test_gfx906_fp8_emu.py` 46 passed。）

**折算到 step**（用 trace 的形状直方图 {N=3584:17, N=4096:48, N=5120:130, N=8704:65} 次/step）：

| 路径 | ms/step |
|---|---:|
| 现状（tiled GEMM） | **97.5** |
| stream GEMV | **28.4** |
| 节省 | **−69.1 ms/step ≈ −31 %（对 219.6 ms 的步时）** |

**同时修好的 GEMM tile**：新增 cfg 5（32×64 TM2TN4 / 256 线程）与 cfg 10（同 tile、4 blocks/CU），
在 GEMM 路径上 1.1–1.7×（N=8704：709 → 414 µs）。cfg-0 自动规则改成 cfg5（M≤32）/cfg10（N≥6144）
后 **prefill M=16..129 提速 1.3–2.0×**（4352 M=64：901→462 µs；8704 M=129：2979→1541 µs）。
prefill 无回归（同 .so A/B，±1.0 %）。

**派发规则**（已实现，`VLLM_DCU_FP8_GEMV_MAX_M` 语义不变，默认/8 = 形状自适应）：

```
stream GEMV  iff M<=8 且 K%16==0 且 w_stride%16==0 且 block_k%16==0 且 block_n%8==0
             且 M*K*2 + M*nkb*4 <= min(44 KiB, deviceLDS-1 KiB)
否则 M>4 -> tiled GEMM（必须：旧 GEMV 在 M=8 比 GEMM 慢 3-9×）
否则     -> 旧 GEMV
```

⇒ **生产配方必须去掉 `VLLM_DCU_FP8_GEMV_MAX_M=1`（或设为 4）**，否则这条收益完全用不上。

### 5.1 最终 .so（`8e98bd60`，含 rows≥120 负载均衡 + M>4 回退）实测

| N×K | M=1 | M=2 | **M=3** | M=4 | M=5..8（回退 tiled GEMM） |
|---|---:|---:|---:|---:|---:|
| 1280 | 25.5 | 29.8 | **37.3** | 50.2 | 245.5 |
| 3584 | 46.0 | 55.8 | **63.4** | 101.1 | 246.3 |
| 4096 | 58.7 | 72.7 | **80.6** | 122.0 | 320.6 |
| 5120 | 61.3 | 75.8 | **84.0** | 126.2 | 321.3 |
| 8704 | 107.6 | 120.9 | **149.9** | — | — |

（µs，全部 rel=0.00e+00。）按 trace 形状直方图折算 **≈25.6 ms/step**（前一版 28.4，改造前 97.5）。
`M=5..8` 现回退到 tiled GEMM（245–321 µs）而非旧 GEMV（704–3628 µs）—— 这是拔高阈值前必须有的回退。

### 5.2 prefill（cfg-0 自动规则：M≤32 用 cfg5，N≥6144 或深 M 用 cfg10）

| N | 旧 cfg0 | 新 cfg0 | cfg5 | cfg10 | 相对旧基线 |
|---|---:|---:|---:|---:|---:|
| 1280 M=16 | 418 | **282.8** | 248.2 | 262.1 | 1.48× |
| 4352 M=64 | 902 | **459.9** | 591.3 | 459.8 | **1.96×** |
| 4352 M=129 | 1636 | **812.8** | 1012.0 | 814.3 | **2.01×** |
| 8704 M=129 | 2979 | **1540.5** | 1976.1 | 1540.3 | **1.93×** |

⇒ 端到端 TTFT 2046 → 1685 ms（**−17.6 %**），prefill 220.4 → 267.6 t/s。

---

## 6. L2 GDN 内核：根因是**访存不合并**（−28 ms/step）

`tools/dcu/perf1_gdn_gpu.sh` / `gdn_sweep.log`（GPU1，真实形状 1 请求、T=3、kv_heads=12、
heads=4、k=v=128、fp32 state、原生布局 `[S,12,128(v),128(k)]` stride `(196608,16384,128,1)`）：

| 实现 | event µs | **kernel µs** | host µs |
|---|---:|---:|---:|
| v1 C++（生产） | 702.0 | **705.2** | 20.3 |
| 新 Triton（BV=4/warps=1） | 112.9 | **19.5** | 119.9 |
| v1 换成 v 连续视图（仅探针，数值不对） | 40.2 | 26.2 | 39.9 |

⇒ **GPU 时间 36×，event 时间 6.2×**；按 48 次/step 计：GPU 33.8 → 0.94 ms，event 33.7 → 5.4 ms
（**−28 ms/step ≈ −12.9 %**）。代价是 Triton 的 host 启动 120 µs/call（C++ 是 20 µs），
即 +4.8 ms/step CPU；由于本平台 CPU 有 2.6× 余量（报告 §4.4），预期净收益约 −28 ms。

**根因（证伪了"grid 只有 6 个 block"的假设）**：state cache 布局 `[slot][head][v][k]` 中 k 连续，
相邻 lane 相隔 512 B，访存完全不合并；换成 v 连续视图后同一 kernel、同一 grid 只要 26.2 µs。
`mtp_v2` 的 C++ 实现之所以没生效，是它的形状门要求 `state.stride(2)==1`，而生产是 128
（`MTP_V2_DEBUG` 实测 `MISS state_s3=1 state_s2=128`）。

**一致性**：triton vs C++：max\|Δout\|=7.6e-06、max\|Δstate\|=1.8e-07、4608 个 fp16 输出中 1 个不同
（v1 自己 vs eager 也有 2/4608 不同）；8 例 GPU 夹具（R 1–3、T 1–4、含 −1 slot）最坏 14/N；
CPU 解释器 10 例对拍纯 torch 版 v1：≤6e-8/2.4e-7，**−1 slot 用例逐位为 0**（并修掉了一个真实分歧：
v1 在链断裂后会重读旧 slot）。开关 `VLLM_QWEN35_MTP_GDN_TRITON=1`（默认关）。

### ⚠️ 但 e2e 是**负结果**（微基准 ≠ 端到端），机制已定位

**受控 A-B-A**（同一 .so `8e98bd60`、k=2、`MAX_M=4`、`BF16=0`，唯一变量是 GDN 开关）：

| arm | step p50 | decode t/s | tok/step |
|---|---:|---:|---:|
| A（GDN 关） | 139.397 ms | 20.93 | 3.048 |
| B（GDN Triton 开，+PROBE+DEBUG） | 150.213 ms（3 次 148.2/153.7/148.7） | 19.01 | 3.048 |
| C（GDN 关，复测） | 139.963 ms | 20.20 | 3.048 |

⇒ A≈C，**+10.3 ms/step 稳定复现，不是漂移**。

**机制（在 worker 内埋点实测，不是推断）**：

| 量 | 值 |
|---|---|
| 真实调用签名 | T=3、R=1、heads=4、kv_heads=12、KD=VD=128、fp32 state 1353 slots、BV=4/warps=1、grid=(32,12,1) —— 与独立探针**逐字节相同** |
| C++ 路径 | kernel 705.2 µs / event 702.0 / **host 20.3 µs** |
| Triton 路径 | kernel **19.5 µs** / event 112.9 / **host 210.5 µs** |
| host 差 | +190 µs/call × 48 = **+9.1 ms/step**（实测 +9.4 ms/step，吻合） |
| 每层 host 时间 | 2.83 ms × 48 = **136 ms ≈ 对照组的 139.4 ms step** |

⇒ **该 step 已处在 CPU 启动开销的临界点上**：Triton 内核省下的 33 ms/step GPU 时间**不在关键路径上**，
而它多花的 CPU 时间在。派发本身只占 0.12 µs/call，wrapper Python 18.6 µs/call，其余是 Triton launcher。

**结论：不进入配方**；C++ 移植在当下同样中性（host 仍 ~20 µs，GPU 收益同样被隐藏）。
该内核在"启动开销被消除"的场景下**可兑现 −33 ms/step**，这是后续方向（报告 §10）。

（教训：任何"某 kernel 快 N 倍"的结论必须用一个**只改一个变量**的 e2e arm 复核，
且必须做 A-B-A 以排除漂移 —— 本次先误判为 −22 %（比较了错的 arm），后经 A-B-A 才定案。）

## 7. L2 Triton 回退：三个模块 + 按形状改路由（−6.6 ms/step）

模块识别（grid 算术 + kernel 邻域 + checkpoint 结构三重确认）与 `mm_probe.log` 实测（fp16）：

| 模块 | 形状 | 当前 Triton | 模拟 GEMV（最佳） | 倍数 | 判定 |
|---|---|---|---:|---:|---:|---|
| GDN `in_proj_ba`（48 次/step） | M=3, N=24, K=5120 | 155.7 µs | **19.0**（nt=128） | **8.2×** | **GO** |
| MTP `mtp.fc`（~2 次/step） | M=3, N=1280, K=10240 | 415.8 µs | **139.5**（nt=128） | **3.0×** | **GO** |
| `lm_head`（~3 次/step） | M=3, N=62080, K=5120 | 1856.5 µs | 1876.0 | 1.0× | M≤2 GO（M=1: 902.8 vs 1589.8），M≥3 保持 Triton |
| `lm_head`（草稿 M=1） | M=1, N=62080 | 1589.8 µs | **902.8** | 1.76× | GO |

已落地（`vllm/model_executor/layers/utils.py`，**默认惰性**，需 `VLLM_DCU_USE_BF16_GEMV=1` 才生效）：
`_gfx906_gemv_shape_ok` 只开放**实测赢过当前 Triton 回退**的格子（`n_out<=1280 and n_tokens<=4`；
`n_tokens<=2` 不限宽度），并让线程数随输出宽度选择（≤4096 用 128，否则 64）。
预期 **−6.6～−8.6 ms/step（3–3.9 %）**。

`lm_head` 在 `modules_to_not_convert` 里（BF16 [248320,5120] → 服务端 FP16 = 636 MB/rank/次），
要减半字节必须改 checkpoint/quant config，属独立课题。

---

## 8. e2e 验收（verifier，stage 2，最终 .so `8e98bd60`）

锚点协议：p451/g64、3 次、temp=0、并发 1；每个 arm 都校验了生效的派发值
（`arm_*_environ.txt`，防"静默覆盖"）。

| arm | step p50 | decode t/s | TTFT ms | prefill t/s | tok/step | vs 基线 |
|---|---:|---:|---:|---:|---:|---|
| task-4 基线（`fbfd45ed`, MAX_M=1） | 214.9 | 14.18 | 2046 | 220.4 | 3.048 | — |
| `ctl_preperf1`（新 .so，MAX_M=1） | 190.9 | 15.93 | **1685** | 267.6 | 3.048 | step **−11.2 %**，decode **+12.3 %**，TTFT **−17.6 %** |
| `kern_verify`（新 .so，MAX_M=4） | **135.0** | **21.30** | 1658 | 272.2 | 3.048 | step **−37 %**，decode **+50 %** |

要点：
1. **仅换 .so（cfg 自动规则）**就已 −11.2 % step / −17.6 % TTFT —— 这是 prefill 层收益，必须单列。
2. **把 M≥2 的交由流式 GEMV** 再叠加 −29.3 % step / +33.7 % decode。
3. 合并相对基线：**step 214.9 → 135.0 ms（−37 %）**，**decode 14.18 → 21.30 t/s（+50 %）**。

**一致性与放行理由**：`kern_verify` vs `ctl_preperf1` 的贪心 token 流在 64 个 token 中有 5 个不同
（首个分歧在第 59 位），分歧前各位置 |Δlogprob| = 0.012–0.152 nat。判定为**可解释的求和顺序噪声**，
予以放行，理由：(a) 流式 kernel 对 fp32 参考的 rel_err = 0.00e+00，与 tiled GEMM 相同，且新增
stream-vs-GEMM 交叉校验在 46 项 pytest 中通过；(b) 该扰动**小于**生产配方已接受的 MTP 扰动
（k=2 vs k=0：p512 2/64、p2048 5/64；S18e 记录真实 prompt 上 60–80 % 位置翻转）；
(c) 64 层 GDN 递推放大近邻 tie 的机制已由 S14b 归档。回滚：`S11F_MTP_GEMM=1`。

---

## 9. 最终配方与全长度验收

**配方**（已落到 `/mnt/game/port/serve_fp8_8000.sh`，备份 `serve_fp8_8000.sh.pre-perf1`）：
`.so 8e98bd60` + `VLLM_DCU_FP8_GEMV_MAX_M=4` + `VLLM_DCU_USE_BF16_GEMV=1` + MTP **k=2** + 启动预热。

| 真实 prompt | TTFT 基线 → 最终 | prefill t/s | decode t/s | step p50 |
|---|---|---|---|---|
| 119 | 873 → **680**（−22 %） | 137.8 → 174.9（+27 %） | 10.60 → **20.39（+92 %）** | 209.5 → **138.8（−34 %）** |
| 451 | 2046 → **1676**（−18 %） | 220.5 → 269.9（+22 %） | 13.93 → **22.25（+60 %）** | 215.3 → **133.6（−38 %）** |
| 1806 | 5010 → **4124**（−18 %） | 388.2 → 471.9（+22 %） | 11.85 → **18.34（+55 %）** | 238.1 → **147.8（−38 %）** |
| 7443 | 7089 → **5977**（−16 %） | 1049.9 → 1245.2（+19 %） | 7.78 → **11.79（+52 %）** | 333.3 → **239.2（−28 %）** |
| **均值** | 3755 → **3114（−17 %）** | 449 → **540（+20 %）** | 11.04 → **18.19（+65 %）** | 249 → **165（−34 %）** |

部署后（预热态，`final_warm`，`so_md5=8e98bd60`）复核，与验收 arm 逐项吻合：
p8192 TTFT **5978 ms**（arm 5977）、prefill **1245 t/s**（arm 1245.2）、step **238.7 ms**（arm 239.2）；
p2048 TTFT 3237 ms、prefill 558 t/s、decode 18.46 t/s。

**冷启动**：预热前首请求 p8192 TTFT **29.7 s**，预热后 **5.98 s**（4.93×）⇒ 预热是生产必需步骤
（`tools/dcu/perf1_warmup.py --lens 8192,2048,512,128` + 丢弃一次 decode）。

### 被证伪的候选项（进入最终报告，避免后人重做）

| 候选 | 结果 | 证据 |
|---|---|---|
| **MTP k=3** | 全长度均值 **k=2 更优 4.8 %**（18.19 vs 17.36 t/s）；k=3 只在 p1806 胜（+6 %） | `combined_k2` vs `combined_v2` |
| **MBT 8192** | **NULL**（step 190.72 vs 190.94、TTFT 1691 vs 1685） | `mbt8192` arm（短上下文；451 anchor 未跨 2048 chunk 边界） |
| GDN Triton | 内核 36×，e2e **+10.3 %** step（CPU 启动受限） | A-B-A 139.40/150.21/139.96 ms |
| `NCCL_PROTO=LL` | 干净复测无收益，20 MiB 慢 2.35× | comm matrix2 |
| custom all-reduce | 平台结构性未启用 | 代码 gate + S13 ±0.4 % |
| `lm_head` 走模拟 GEMV（M≥3） | 更慢 | mm_probe |
| CUDA graph | 空档仅 16 ms/step 且 token 不可复现 | 报告 §7.1 + S14b |

### 结论：瓶颈已经转移

decode step 从 219.6 ms 降到 ~135–148 ms 后，逐层 host 时间（48 × 2.83 ms ≈ 136 ms）已与 step 同量级
⇒ **下一步的杠杆是启动开销（graph capture / 减少 launch 数），不再是 GPU 内核**。GDN 内核在
"启动开销被消除"的场景里可兑现 −33 ms/step。
