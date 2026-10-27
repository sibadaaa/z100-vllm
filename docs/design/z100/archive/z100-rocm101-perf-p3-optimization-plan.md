# Z100 ROCm 10.1 性能优化计划（P3：基于 kernel 档报告的判读与施工图）

> 输入：`$DATA_ROOT/port/perf-runs/20261024-053333-E2C/TUNE_REPORT.md`（kernel 档，2026-10-24 05:33–06:06 UTC）
> 同 run 产物：`kernel_report.json`、`kernel_op_tables/*`、`microbench-report/kernel_report.json`、
> `microbench_windows.json`、`probe-*/metrics_diff.json`、`gpu_telem.jsonl`、`serve_launch.sh`、`traces/`
> 前序判读：[`../result/z100-rocm101-perf-verdict.md`](../../../result/z100-rocm101-perf-verdict.md)、
> [`../result/z100-rocm101-perf-baseline-freeze.md`](../../../result/z100-rocm101-perf-baseline-freeze.md)
> 本文档性质：**判读 + 候选方案矩阵 + 施工排序**。不含代码改动；每一项都给判定阈值与停止线。
> 标注：**【实测】**=本 run 产物可直接复算；**【先测】**=本仓历史 A/B 实测但非本栈；**【推断】**=由实测推导；
> **【待测】**=本报告无法判定，需补测。

---

## 0. 一页结论

1. **本 run = E2C ≈ E2 生产候选**（512c1 TTFT 2731 ms / step 240.9 ms / TPOT 46.0 / acc 5.36、62.3%，与 E2 逐项一致）。
   它跑的是**刻意削薄的配方**：`serve_launch.sh` 只有 5 个 `VLLM_DCU_*` 旋钮，DTK 生产脚本
   ([`tools/dcu/serve_host_dflash2.sh`](../../../../tools/dcu/serve_host_dflash2.sh))里的
   **GDN 双开关、`FP8_EMU_FASTPATH`、`FP8_QUANT_FASTPATH`、`FP8_DECODE_CFG=35`、`S24_DYNK_*`、`ATTN_PREFILL_3D` 全部不在**。
   ⇒ **E1–E4/E2C 的绝对数字不能当作"栈上限"，它们是"W8A16 移植单项"的成绩单，不是生产配方成绩单。**
   这条更正直接推翻 verdict §1 "merge 已带入 S23 后调优" 的表述：**代码在树里，开关没打开。**

2. **报告给的两条判读有一条是错的。**
   `[微基准] w8a16 decode GEMV …（带宽受限属预期）`——按本 run 自己的
   `microbench_windows.json` 复算：M=8 搬 `10.268 GB` 权重用 `58.5 ms` = **175 GB/s**，
   而同一 run 的 D2D 拷贝基线是 **721.5 GiB/s = 774 GB/s**。
   **口径提醒**：拷贝值按 `numel*2`（读+写）计，是"总线总量"；纯读流的天花板按同一口径可以取
   775 GB/s（乐观），也可取 `.r3` 独立测得的 **407 GB/s**（保守）。⇒ **GEMV 距屋顶 2.2× ~ 4.4×**。
   无论取哪一端，结论一致：它不是"带宽受限"，它是**带宽饥饿**，也是 decode 侧最大的单项可回收量。
   （本计划下文所有屋顶计算给**区间**，两端都列，避免用一个口径假装精确。）

3. **decode 240.9 ms 的时间去向（按 48 个 GDN 层 + 256 次/步 GEMM 交叉校验）**【实测+推断】

   | 项 | ms/步 | 占比 | 屋顶(乐观~保守) | 可回收 |
   |---|---:|---:|---:|---:|
   | GDN 线性注意力 `mtp_update` 48×2.04 | **97.9** | 40.6% | ~0.1 | **~93** |
   | W8A16 decode GEMM（256 次/步，150–175 GB/s） | 46.1 | 19.1% | 8.7 ~ 17.7 | **~28–37** |
   | TP allreduce（≈128–143 次/步 × 260 µs） | 37–43 | 15–18% | 12.7–14.2 | ~25–30 |
   | rocBLAS 非量化 GEMM（草稿塔 + lm_head） | 36.9 | 15.3% | ~2 | **~35** |
   | 其余（copy/elementwise/GDN 小算子/空隙） | 17–23 | 7–10% | — | — |
   | **合计** | 240.9 | 100% | **23 ~ 32** | — |

   > 口径说明：`aten::mm` 的 1.893 s 不能算进 decode（4 步总 GPU 只有 1.12 s），
   > 它属于 prefill；allreduce 的 700 次调用需扣掉 prefill 的 64 层×2=128 次，
   > 余下 572/4≈143 次/步（与 64c1 的 980 次反算出的 142 次/步一致），故给 128–143 区间。

   同一 run 的 `gpu_telem.jsonl`：**采样窗口内 GPU 利用率均值 39.3%，单卡功率 87 W / 280 W (31%)**
   ⇒ 这段时间里 GPU 有约 6 成时间没在算。（`hy-smi` 的 `util_perc` 是粗粒度信号，只作旁证；
   但"功率只有额定值的 31%"是同一个方向的独立读数。）这既印证"空隙大"，也说明**必须先把 GPU 喂饱再谈 kernel**。

4. **prefill 2717 ms 的时间去向**【实测】

   | 项 | ms | 占比 |
   |---|---:|---:|
   | 物化后 `aten::mm`（fp16，256 次/次前向） | 1893 | 69.7% |
   | `fused_sigmoid_gating_delta_rule_gfx906_prefill` 48×14.11 | **677** | 24.9% |
   | 其余 | ~146 | 5.4% |

   **折算口径**：该次前向的 NVTX 名为 `execute_context_1(709)_generation_0(0)`，且它的 2.730 s
   恰等于 `prefill_p50 = 2717 ms` ⇒ 这一趟是 **709 token 的前向**（不是标称 512）。
   于是 `aten::mm` 的有效算力 = `2×709×6.75e9 / 1.893 s` = **5.06 TFLOP/s/rank**，
   即实测 fp16 屋顶(18.2~19.6) 的 **26–28%**；整趟 prefill 折算 3.52 TFLOP/s。GDN prefill 单层 14.11 ms
   与历史实测"fused C++ 11.149 ms @[512]"同量级，而**同一个算子走 FLA chunked 是 0.758 ms（14.7×）**【先测】。
   ⇒ 只把 `VLLM_GFX906_GDN_CHUNKED_PREFILL` 打开，prefill 预期 **2717 → ~2100 ms（−23%）**，
   **TTFT 一项即可从 Must ✗ 翻绿（≤2.5 s）**。

5. **本报告的 kernel 级证据是残缺的**：`traces/*.pt.trace.json.gz` 全部解压为 `[]`（62–77 B 空文件），
   `kernel_report.json` 的 `windows.*.kernel_count = 0`（`sum-approx(>3M raw rows)`），
   逐 case 表的 `top kernel` 列全是 `-`。
   ⇒ **verdict §5 要求 P3 做的"kernel 级 trace 对比"这一 run 并没有产出**，只有 op 级聚合表。
   微基准还缺最关键的两格：**fp16 GEMM（不是 bf16）的 M=512/1024/2048 扫描**，
   和 **fp8-emu GEMM 在 prefill M（709/1024）下的效率**——而这两格正是 prefill 69.7% 时间的所在。

6. **建议的动作顺序**（本计划的核心主张）：
   **先把被削掉的配方按单变量补回去（零代码，1 天）→ 再修测量（1 天）→ 最后才动 kernel（3–5 天）**
   → 再上 CUDA graph 路线 A。**在配方复原前做 kernel 优化，等于用一个被低估 30–40% 的基线去校准收益。**

---

## 1. 报告可信度审计

逐项复核本 run 的产物。这一节决定后面所有数字能用多少。

| 项 | 判定 | 证据 / 复算 |
|---|---|---|
| microbench `w8a16_m8` 效率 | ✅ 可信 | `flops/bytes` 由 `microbench_windows.json` 给出；RNG+乘污染仅 2.6%（`categories_us`: gemm 56.64/58.16 ms） |
| microbench `bf16_m*` 效率 | ✅ 可信（但用错 dtype） | 我把"循环内现造 `torch.randn` 权重"当污染源怀疑过，复核 `categories_us` 后**否掉**：m8 污染 11%、m8192 仅 0.5% ⇒ 数字是干净的。真正的问题见下条 |
| microbench dtype 选错 | ❌ **测量缺陷** | 服务 prefill 走的是**物化后的 fp16** `aten::mm`（`--dtype float16`），但微基准第 3 段测的是 **bf16** `a @ w.t()`。用 bf16 的 3.4 TFLOP/s 去描述 fp16 prefill 会低估约 5× |
| `hints[0]` "带宽受限属预期" | ❌ **判读错误** | 175 GB/s vs 拷贝基线 775 GB/s（总量口径）= 22.7%；vs 纯读 407 GB/s（`.r3` 口径）= 43%。两种口径下都不是"已到带宽"，见 §0.2 |
| 带宽口径本身有歧义 | ⚠️ 需双列 | 本 run 的 `mb_memcpy_d2d_1g` 按 `numel*2`（读+写）计得 721.5 GiB/s；`.r3/REPORT-round3.md:30-33` 的 mem 天花板是 **407 GB/s**。两者口径不同（总线总量 vs 单向流），**引用时必须说明取哪个**，否则屋顶差 1.9× |
| `telemetry.vram_pct_max = 0.0` | ❌ **聚合 bug** | `gpu_telem.jsonl` 每卡 `vram_perc` 恒为 **92–93%**；报告取了错字段。（这条很重要：显存已到 93%，"缓存 fp16 权重一份常驻"类方案没有空间） |
| `telemetry.power_w_max = 312 W`（cap 280） | ⚠️ 需降级表述 | 1941 个样本里只有卡 2 单点 312 W；卡 0/1/3 峰值 291/285/283 W；**温度峰值仅 65 ℃**。不是持续性墙，报告"已贴近上限、可能降频"的措辞偏重 |
| kernel 级数据 | ❌ **缺失** | `traces/*` 全为 `[]`；`kernel_report.json.windows[*].kernel_count = 0`；`top kernel` 列全 `-` |
| 遥测覆盖 | ⚠️ 部分 | 2205 s 采集窗口里只有 **分 6–10（约 290 个样本）** 有负载，其余 31 分钟是空转（161 W 总功率=4×40 W 待机）。"整个 bench" 的说法不成立 |
| 标称 token ≠ 实 token | ⚠️ **口径坑** | `probe-512c1` 实际 **709** tok、`probe-709c1` 实际 **979** tok（历史记录 "标称偏低 ~45%"）。**不要把本表 512c1 与 B3 的"手动 512"直接比** |
| 4 卡是否失衡 | ✅ 未失衡 | 同 case 4 个 rank 的 `aten::mm` self_dev 差 <1%（256c4: 979.4/979.1/981.4/986.3 ms）。⇒ 历史"Z100L@1600 vs Z100@1319 时钟不对称"在本 run **没有表现为可见的卡间失衡**，不作为本轮优化项 |
| 功耗/时钟 | ❌ 无 sclk/mclk | 全部 `metrics_*.json` 里 `clock/sclk/mclk/throttle` 出现 0 次。verdict §2 悬而未决的"192 vs 269"解释，本 run 依然无法回答 |

**净结论**：microbench 的 **w8a16、attn、memcpy 三段可用于屋顶计算**；`bf16` 段只能描述草稿塔，
不能描述 prefill；**服务侧只有 op 级聚合，没有 kernel 级时间线**，所以 decode 内部的
"空隙 vs 计算"只能靠遥测（39.3% 利用率）间接判断。

---

## 2. 根因分解（按可回收时间排序）

### R1 — GDN 线性注意力：decode 40.6% + prefill 24.9%，且根因已在树里定位

- **decode**：`_C::fused_sigmoid_gating_delta_rule_gfx906_mtp_update`，4 步窗口 192 次调用（=48 层×4 步）、
  均 **2.040 ms/次**【实测】，逐 case 几乎恒定（64c1 也是 2.045 ms）。
- **根因（树内已写死）**：`vllm/model_executor/layers/fla/ops/gfx906_gdn_mtp_triton.py:1-40` 的模块注释给出
  实测：v1（生产 C++，grid (6,1,1)）**714 µs/call**，同一 kernel 换成 v-contiguous 视图只要 **48 µs**——
  **26× 的差全是布局**：持久状态按 `[slot][head][v][k]` 分配（k 连续），
  于是"相邻 lane 相差 512 B"，每次 warp 级状态载入触到 32 条 cache line。
  把 lane 映射到 key 轴、grid 改成 `(VD/BV, kv_heads, requests)`（384–1536 programs）即可。
- **prefill**：`..._gfx906_prefill` 均 **14.11 ms/次** × 48 = 677 ms【实测】。
  同一算子在 `VLLM_GFX906_GDN_CHUNKED_PREFILL=1` 下历史实测 **0.758 ms @[512] / 1.062 ms @[1024]**
  （14.7–24.0×，parity 6/6，`max|dout| ≤ 1.5e-5`、负对照 6/6 FAIL）【先测】。
- **可回收**：decode GPU 侧 ~93 ms/步，扣掉 Triton 的 host 成本（48×~0.22 ms ≈ +10.8 ms/步）后
  **净 ~82 ms/步**；prefill ~626 ms/次前向（chunked 路径无对应 host 增量）。

> ⚠️ **必须两边都读的反证**：`tools/dcu/perf1_gdn_report.md` 记录过 A-B-A 实验——Triton GDN kernel
> **kernel 快 36×（705→19.5 µs）但 e2e 反而 +9.4 ms/步**，因为那一轮 decode 是 **CPU-launch 主导**
> （Triton 侧 host 成本 +190~215 µs/call × 48 = +10.8 ms/步，而省下的 33 ms GPU 时间不在关键路径上）。
> 而 `tools/dcu/DFLASH2_OPT_RESULT.md` 在 k=7、step 288.6 ms 的 GPU-bound 窗口里量到
> **step 288.6 → 190.9 ms（−97.7 ms）**。**同一个开关，两种符号的结论，取决于该步是 GPU-bound 还是 CPU-bound。**
> 本 run：48 层的 C++ kernel 占 97.9 ms/步（40.6%），且总 GPU 时间(4.325 s) ≥ 总 CPU 时间(4.137 s) ⇒
> **判为 GPU-bound，预期落在 −82 ~ −93 ms 一侧**。但这是**推断，不是实测**，所以 R1 仍必须是"臂"而非"直接上线"。

### R2 — W8A16 decode GEMM 只吃到 23–43% 带宽（46.1 ms/步）

- 交叉校验：decode 每步 256 次 `fp8_emu_gemm_block_w8a16`（1024 调用 / 4 步 = 64 层 × 4 投影）【实测】；
  GEMM 总 46.1 ms/步 ⇒ 180 µs/次；单次平均权重 6.89 GB/256 = 26.9 MB ⇒ **150 GB/s**。
  与微基准 M=8 的 **175 GB/s** 独立吻合。
- 代码侧：`csrc/rocm/fp8_emu_gemm.cu:1718` 的 `fp8_emu_gemm_block_w8a16_kernel<BK=64, BM=8, BN=64, TM=1, TN=2>`
  （= 注释里的 cfg 5「exact-M decode tile」），grid = `(N/64, M/8)`。
  小 N 投影的**网格严重不足**：`o`/`down`（N=5120）只有 **80 个 block**，`qkv`(8192) 128 个，
  `gateup`(34816) 544 个；60 CU 上 80 block = 1.3 波，最后一波尾巴直接吃掉有效带宽。
  K 方向每 `BK=64` 一次 `__syncthreads()` 且只有 2 级缓冲（`As[2]/Bs[2]`），
  K=17408 时 272 次同步 ⇒ 内存级并行被"每级一条在途请求"卡死。
- **可回收**：46.1 → 屋顶 8.7（乐观）~ 17.7 ms（保守），取中段即 **~26–31 ms/步**。
- **口径更正**：历史 S18 的"M=8 GEMV 成本是最优 tile 的 7.4×"与本文不矛盾——
  那是 kernel 间相对比较，本文是**对屋顶**比较；两者都说明这条路值得修，只是天花板更低。

### R3 — 配方被削（零代码，但是本轮性价比最高）

`serve_launch.sh` vs DTK 生产脚本的差集，且每条都有历史实测增量：

| 缺失旋钮 | 默认值 | 作用 | 历史实测增量 | 出处 |
|---|---|---|---|---|
| `VLLM_GFX906_GDN_CHUNKED_PREFILL` | `"0"` | GDN prefill 走 FLA chunk | per-op 14.7–24×；TTFT −31% | `gdn_linear_attn.py:109`；`DFLASH2_OPT_RESULT.md` |
| `VLLM_QWEN35_MTP_GDN_TRITON` | `"0"` | GDN decode 走 Triton | kernel 26–36×；e2e −34% 或 +9.4%（见 R1 反证） | `gfx906_gdn_mtp_triton.py:61` |
| `VLLM_DCU_FP8_DECODE_CFG` | 未设→auto | **W8A8 时代**的 decode tile 选择；W8A16 路径经 `w8a16_cfg()` 映射到固定成员 | O3 在 **W8A8** 上 kernel −15.3~−23.2%、e2e c=1 **−7.34%**；**在 W8A16 上方向未定，见 A3** | `fp8_emu_gemm.cu:2154`、`:1786` |
| `VLLM_DCU_FP8_EMU_FASTPATH` | `False` | eager 下直调 `_rocm_C` | −57.9 µs/次、**−13.9 ms CPU/步** | `envs.py:227` |
| `VLLM_DCU_FP8_QUANT_FASTPATH` | `False` | 激活量化走快路 | **−3.27 ms CPU/步** | `envs.py:233` |
| `VLLM_DCU_USE_BF16_GEMV` | `False` | 用自研 GEMV 替代 rocBLAS 小 M 回退 | **未定**：port-plan:184 记"用户要求：关闭"；另有记录"草稿塔 rocBLAS 比自研快 2–3×" | `envs.py:207` |
| `VLLM_S24_DYNK_LO/HI/EMA` | `0/0/0.1` | 投机验证长度按累计 token 截断 | T2 未 merge | `envs.py:265`；`scheduler.py:299` |
| `VLLM_GFX906_TRITON_ATTN_PREFILL_3D`(+`MIN_CTX=16384`) | `"0"` | 长上下文 prefill split-KV | p32k TTFT −5.3%（短 ctx 大亏，故有 16K 门） | `triton_unified_attention.py:47` |

> **不是"回归 bug"，是刻意的实验设计**：[`z100-w8a16-rocm101-port-plan.md`](../w8a16/z100-w8a16-rocm101-port-plan.md):181-184
> 的 `EXTRA_ENV` 就是这 5 个，目的是把"W8A16 移植"作为单一变量测干净。
> 但**代价是 E 系列的绝对数字不能当栈上限**——而 verdict 已经把 240 ms 当成"host 栈主因"去追 P3 了。

### R4 — rocBLAS 小 M 路径：36.9 ms/步（草稿塔 + lm_head）

- 微基准：**bf16 GEMM M=8 = 0.163 TFLOP/s**、权重流式 **20.4 GB/s**——是 fp8 路径(175 GB/s)的 **1/8.6**，
  是拷贝基线的 **1/38**。使用的是 `MT32x128x16` 宏瓦片（depth 16），M=8 时 A 瓦片浪费 75%，且深度太浅。
- `VLLM_DCU_USE_BF16_GEMV` 正是"替掉这个 rocBLAS 回退"的开关，当前是关的。
- **但这条路有两个硬约束**（务必先读）：
  1. 草稿塔**必须** bf16 激活：改 fp16 会让 MLP 激活溢出（`config/speculative.py:429-435`），
     且 R1 实验里 4/4 worker 因 kv dtype 冲突初始化失败 ⇒ **"草稿塔改 fp16"是死路**。
  2. "草稿塔 rocBLAS 比自研 triton/emu 快 2–3×" 是既有结论 ⇒ 自研 GEMV 未必赢。
- **可回收**：乐观 ~35 ms，悲观 0；只能靠单变量臂判定。

### R5 — prefill fp16 GEMM 只有 26–28% 屋顶（1893 ms）

- 5.06 TFLOP/s/rank（`aten::mm` 一段，709 token 前向折算）vs 实测 fp16 屋顶
  **18.2 TFLOP/s**（`vllm/platforms/rocm.py:288`）/ **19.61 TFLOP/s**（`.r3/REPORT-round3.md:30-33`）。
- 四个未做的定界动作（verdict §2 的 P3.2 原样未动）：
  `torch.mm` 实际后端 → `ROCBLAS_TENSILE_LIBPATH` / `TORCH_BLAS_PREFER_HIPBLASLT` → 形状选择 →
  `VLLM_DCU_FP16_PREFILL_MIN_M` 单 role 扫描。
- 另有**测量缺口**：没有任何一个微基准测过"fp16 `torch.mm` 在 M=709/1024、K/N=服务形状"的效率。
  用 bf16 的 3.4 TFLOP/s 去推断会把问题放大约 5×。
- **可回收**：`aten::mm` 效率翻倍（5.06→10.1）即 **−947 ms**；做到屋顶的一半（9.8）即 −910 ms。

### R6 — prefill 在 48~149 个点上串行化（**已定位到 gfx906 conv1d，且带已知 wedge 风险**）

- 【实测】`aten::item` / `aten::_local_scalar_dense` 149 次，**self CPU 2.389 s = 57.74%**，均值 16.03 ms。
- **根因已定位**：`vllm/model_executor/layers/mamba/ops/causal_conv1d.py:99-106` 的注释把机理和量级都写死了——
  逐序列回退路径每个序列每层要读 `cache_indices[i].item()` 与 `has_initial_state[i].item()`，
  原文："**~2 per sequence per layer（48 层 GDN 模型上 96+ 每次前向）**，and dominated the gfx906 prefill step"。
  48 层 × 2 序列 × ~1.5 ≈ 144，与实测 **149** 吻合。
  旁证：本 run op 表同时出现 `aten::_conv_depthwise2d`(48 次) 与 `aten::_convolution`(48 次)
  ⇒ 确认走的是 **torch 回退路径**，不是 Triton 路径。
- **这不是 2.39 s 的净损失**：阻塞等待与 GPU 执行并行，`self CPU` 含等待时间。
  它证明的是**CPU 无法跑在 GPU 前面**——prefill 被切成 ~149 段、每段约 18 ms GPU 工作就同步一次。
  这与遥测"39.3% 利用率"互相印证。
- **⚠️ 不要直接"消掉 `.item()`"**：同文件 `:42-52` 记录了快路径在 gfx906 上的硬故障——
  "fully-async Triton conv（无 per-layer `.item()` 同步）在长分块 prefill 下**确定性地把 GPU compute ring 卡死**，
  ring 100% 空转、只有 kill 进程能恢复"；缓解手段是**每次 conv 启动后整条流同步**，
  也就是用"48 次 full stream drain"换掉"149 次 DtoH"。
- ⇒ 真正的杠杆不是二选一，而是**消掉 drain 的必要性**（更小的 in-flight 窗口 / 按 chunk 而非按 layer 同步 /
  用 fence 代替全流同步），或干脆在 prefill 用更大的 chunk 摊薄。**这条带硬件级风险，必须单独立项、单独观测。**
- 与历史结论的冲突就此解释：`.r3/REPORT-round3` WP7 判"GDN prefill `.item()` = no-op"是在 **DTK 树**上做的，
  rocm101 树此路径仍有回退 ⇒ **不是新回归，是这次移植没带上该优化**。

### R7 — 启动气泡（本栈未测）

- DTK 侧实测：每步 **2509 GPU event / 4379 HIP API / 13669 ATen op**，GPU idle **54.9/269.5 ms = 20.4%**，
  且论证了"图模式上限 15–20%"【先测】。
- 本 run 因 `traces/*` 全空，**气泡无法量化**；只能用遥测 39.3% 利用率间接说明"不少"。
- 若 R7 占 20%，且 route A 能吃掉其中的 7–9%（DTK 实测 decode −7.1%、TTFT 持平），
  这是唯一已验证、能动"decode 上限"的杠杆。

---

## 3. 候选方案矩阵（探索多种可能性）

每条给：假设 / 证据强度 / 预期收益 / 成本 / 风险 / 判定探针。
**收益均为"相对本 run 240.9 ms / 2731 ms"的估计区间，不是承诺。**

### A. 配方复原（零代码，单变量 A/B）

| 编号 | 动作 | 预期 | 风险 | 探针 |
|---|---|---|---|---|
| **A1** | `VLLM_GFX906_GDN_CHUNKED_PREFILL=1` | prefill −600 ms、**TTFT 2731→~2100 ms** | Triton 3.6.0(rocm101) 与 DTK 自编 triton 不同；该链历史上"曾被误编译而绕过" | 先跑 parity（`perf1_gdn_chunked_parity.py` 同口径），再 512c1/709c1 |
| **A2** | `VLLM_QWEN35_MTP_GDN_TRITON=1` | step **−35 ~ −93 ms**（区间见 R1 反证） | 同上；且 kernel 文档称"最多 4 个投机 token"，本配方 T=8，**覆盖度需补证** | 24 个样本的 `PROBE=1` host/gpu 计时 + 步进 p50 |
| **A3** | `VLLM_DCU_FP8_DECODE_CFG` **扫 11–19**（不是 35，见下） | 未知，可能 ± | **中**：直填 35 会把 M≤8 从 BM=8 换成 BM=32 | 微基准 + 步进 p50 + parity |
| **A4** | `VLLM_DCU_FP8_EMU_FASTPATH=1` + `VLLM_DCU_FP8_QUANT_FASTPATH=1` | −17.2 ms **CPU**/步；仅在 CPU 紧张时兑现 | 低；历史 c=4 +6.6%/+3.55%（O1），故需 c=1 与 c=4 都看 | CPU 时间占比 + 步进 p50（c=1/c=4 各一） |
| **A5** | `VLLM_S24_DYNK_LO=16384 HI=65536` | 长 ctx 投机截断（T2 未 merge，收益未知） | 中；会改接受长度，须同口径比 | 8k c1 + 512c1 接受长度 |
| **A6** | `VLLM_DCU_USE_BF16_GEMV=1` | 0 ~ −35 ms/步 | **高**；port-plan 记"用户要求关闭"；历史说 rocBLAS 更快 | 只做一次单臂，`<3%` 即停 |
| **A7** | `VLLM_GFX906_TRITON_ATTN_PREFILL_3D=1 MIN_CTX=16384` | 仅长 ctx（p32k −5.3%） | 短 ctx 大亏，已由 MIN_CTX 门挡住 | 仅 8k/32k 档，512/709 档不回退即通过 |
| **A8** | `VLLM_GFX906_TRITON_CAUSAL_CONV1D=1`（+ `VLLM_ROCM_USE_GFX906_TRITON_CAUSAL_CONV1D_UPDATE=1`） | 去掉 149 次 DtoH；换 48 次流同步，净值未知 | **高**：该快路径在长分块 prefill 下已知会 **wedge GPU**（`causal_conv1d.py:42-52`，64K 实测复现） | **只做短 ctx（≤1k）单臂**；必须挂 ring 空转/超时观测 + kill 预案；8k/32k 档一律不试 |

> **A 组的纪律**：一次只改**一个**变量；每组都要 `stack_identity` + `serve.log` 的 route 证据 + 4 组 probe。
> 预期 A1+A2+A3 三项叠加即可让 **TTFT ≤2.5 s 与 step ≤210 ms 双双翻绿**（2100 ms / ~150–205 ms）。

> **A3 为什么不能照抄 35（本轮最容易被照抄错的一条）**
> `csrc/rocm/fp8_emu_gemm.cu:1786` 的 `w8a16_cfg()` 显示，W8A16 路径的 **auto 规则**是
> `cfg = (M <= kMsMax) ? 5 : ((N >= 6144 || (M > 32 && N >= 3072)) ? 2 : 1)`，
> 而 **W8A8 的 cfg 号 32–40 被硬映射成固定成员（35 → cfg 2 = BK64/32x64）**，
> 只有 **cfg 11–19** 保留"`M ≤ kMsMax` 时仍取 cfg 5"的语义。
> 本 run 里 profile 抓到的正是 **cfg 5**：`fp8_emu_gemm_block_w8a16_kernel<64, 8, 64, 1, 2>`。
> ⇒ **直接设 `=35` 等于把 M≤8 的最优 BM=8 tile 换成 BM=32**（A 瓦片 75% 浪费），
> 在 c=1 档很可能回退；而在 M>8 档 auto 本来就已经是 cfg 2，35 几乎是空操作。
> O3 的 −7.34% 是在 **W8A8** 路径上量的，**不可直接外推到 W8A16**。所以 A3 要扫的是 11–19 这一组。

### B. CUDA graph 路线 A 在 rocm101 上复验（结构性，动 decode 天花板）

- 依据：DTK 实测 decode **127.5 → 118.4 ms（−7.1%）**、ITL −5.9%、TTFT 持平、c4 中位 +7%、greedy 8/8 一致；
  **前置修复已在树里**（`vllm::fp8_emu_mm_block_w8a16_auto` opaque op + LUT 独立 op）。
- 本栈状态：**rocm101 全臂 `--enforce-eager`，从未跑过 compile/graph 臂。**
- 风险（全部已实测，不是猜测）：ready **306 s**、宿主 +3 GiB、**KV −2.5%**、**图模式跨重启数值不可复现**
  （S14d 逐位一致率 2.4%）⇒ **只能用于吞吐/延迟口径，SLA 与质量必须留 eager**。
- 路线 B（`VLLM_USE_BREAKABLE_CUDAGRAPH=1`）**判死**：`tok/step=1.00`、logprobs NaN/Inf、c=4 KERNEL VMFault。

### C. W8A16 decode GEMV 带宽修复（kernel，动 decode 46.1 ms 的大头）

三个可独立验证的子假设：

- **C1 网格不足**：`o`/`down`（N=5120）只有 80 block / 60 CU。→ **split-K** 或把 BN 从 64 降到 32，
  使 block 数翻倍；代价是归约。判据：微基准 `N=5120, K=6144/17408, M=8` 的 GB/s **≥300**。
- **C2 流水深度不足**：`BK=64` + 2 级缓冲 + 每级一次 `__syncthreads()`，K=17408 时 272 次同步。
  → 3–4 级缓冲 / `__syncthreads` 减半 / cp.async 式预取。判据同上。
- **C3 流式 GEMV 在 M=8 不可用**：既有实现（445 GB/s @M=1，72% 诚实流参考）在 M≥5 被
  **44 KiB LDS 门**挡在 K=5120。→ 把 K 分块流式化以绕开 LDS 门。判据：M=8 达 **≥350 GB/s**。
- **停止线**：微基准改善 **<10%** 或 e2e **<3%** → 停，保留现状（沿用 construction §6.2 的停止线）。

### D. prefill GEMM 定界（测量为主，收益最大但最不确定）

`D1` 打印 `torch.mm` 实际后端 + `ROCBLAS_TENSILE_LIBPATH` / hipBLASLt 可用性；
`D2` 按 role 单旋钮扫 `VLLM_DCU_FP16_PREFILL_MIN_M`；
`D3` **补一格微基准**：fp16 `torch.mm` 在 `M∈{709,1024}, (K,N)∈{服务四形状}`。
判据：单点 ≥10% 才继续；否则接受现状并转向 A/B/C。

### E. conv1d 串行化专项（R6，**前置条件是安全，不是快**）

`E1` 先用 `py-spy`（host_check 报未安装，先装）+ `torch_profiler_with_stack:true` 把 149 次
`_local_scalar_dense` 的调用栈钉死（预期落在 `causal_conv1d._causal_conv1d_gfx906_fallback`）。
`E2` 量清两条路的净账：回退路 149 次 DtoH vs Triton 路 48 次 full stream drain（A8）；
`E3` 真正的目标：**让 drain 不再必要**——例如把"每层一次全流同步"降级为"每 chunk 一次"、
或用更窄的 in-flight 窗口（减少队列深度）替代全流 drain。
判据：先拿到 **wedge 可复现的最小形状**（`causal_conv1d.py:42-52` 说 64K 分块可复现），
**在没有 wedge 观测手段之前，任何 E 项改动都不许上生产**。收益按"prefill 分段 149→<20"评估。

### F. 投机经济性（本报告已排除一半）

- **spec k 扫描 = 死路**（k=7 已最优，`opt-round3:483`）——不要再提。
- 但 **并发档的步进放大**是新现象：`709c2`（2 请求，M=16）step **384.9 ms** vs `512c1`（1 请求，M=8）
  **240.9 ms**——权重流量相同（6.89 GB/步），步进却 +60%。
  `VLLM_DCU_FP8_GEMV_MAX_M` 在 `gfx906_fp8.py:83` 被 **clamp 到 [0,8]**，
  所以 M>8 必然落到 `MT64x32x8`-类 tile（微基准 M=64 仅 36 GB/s 有效流量）。
  ⇒ **可试：为 M∈[9,32] 补一个"权重只读一次"的 T 行 tile**。判据：512c4/709c2 步进 −20% 以上。
  这是一条**尚无人试过**的方向（历史只调过 GEMV_MAX_M 上限，没做中间档 tile）。

### G. 长上下文投机塌陷（独立问题，已记录）

8k c1（实 11723 tok）接受长度 **2.0**、接受率 **14.3%**，step 242–287 ms。
与 512 档结论不可互推。建议单列，不与本轮混合。

---

## 4. 分阶段施工计划

### 4.0 验收基准（沿用 `z100-rocm101-perf-construction.md:55-62`，不另立标准）

| 指标 | Must | Target | 本 run 现值 | 判定 |
|---|---|---|---|---|
| TTFT p50（512 档 = 实 709 tok） | ≤ 2.5 s | ≤ 1.8 s | **2731 ms** | ✗ |
| TTFT p50（709 档 = 实 979 tok） | ≤ 2.5 s | ≤ 1.8 s | **3742 ms** | ✗ |
| decode step p50 | ≤ 210 ms | ≤ 200 ms | **240.9 ms** | ✗ |
| TPOT p50 | ≤ 45 ms | ≤ 40 ms | **46.0 ms** | ✗ |
| 接受长度 | ≥ 4.0 | ≥ 4.5 | 5.36 | ✓ |
| 接受率 | ≥ 45% | ≥ 50% | 62.3% | ✓ |

> 纪律（`construction.md:104-119`）：**禁一次改两个变量**；每臂留 `stack_identity.json` + `serve.log` +
> 4 组 `http_summary.json`；探测顺序 `512c1 → 709c1 → 8k c1 → 512c4`；A/B 期间关 prefix caching；
> 固定 `MAX_NUM_SEQS=4 / TP=4 / maxlen 32768 / MBT 8192 / --enforce-eager`。

### P0 — 配方复原（0.5–1 天，零代码）

1. 用 `serve_host_dflash2.sh` 的 env 块派生 rocm101 版 launcher（**只改 env，不改命令**）。
2. 按 **A1 → A2 → A4 → A3** 的顺序单变量叠加，每步跑 `512c1 / 709c1` 两组。
   （A1 收益最大且机理最硬，先做；A2 收益第二但反证最强；A4 只动 CPU；A3 方向未定放最后，
   因为它可能把 M≤8 的 BM=8 tile 换掉而回退。）
3. **退出条件**：四项都做完，或出现第一项 e2e <3% 即停止该项。
4. **交付**：每组 4 个 probe 目录 + `serve.log` route 证据 + 一页 delta 表。

**预期**：TTFT 2731 → ~2100 ms（A1），step 240.9 → 150–205 ms（A2，A3 若为正再叠加），
**Must 三项（TTFT ≤2.5 s、step ≤210 ms、TPOT ≤45 ms）全部翻绿**。

### P1 — 测量修复（1 天，必须早于任何 kernel 工作）

1. **修 trace**：`traces/*` 全空，查 `profiler-config` 的 `torch_profiler_dir` 与 rocm101 下
   kineto/rocprof 的落盘路径；至少要能出 kernel 级时间线（否则"空隙 vs 计算"永远靠猜）。
2. **修遥测聚合**：`vram_pct_max` 取错字段（实际 92–93%）；补 `sclk/mclk/throttle` 字段。
3. **补两格微基准**：fp16 `torch.mm` 的 M 扫描；`fp8_emu_gemm_block_w8a16` 在 M=709/1024 的效率。
4. **修微基准 dtype**：把 bf16 段复制一份改成 fp16（服务 prefill 用的是 fp16）。
5. **补 k=8 覆盖**：R1 的 Triton GDN 文档只声明"最多 4 个投机 token"，本配方 T=8。

### P2 — kernel 工作（3–5 天）

按 **C1/C2 → F → C3 → D** 的顺序；每项独立开关 + 独立基线（P0 后的基线），停止线同 §3.C。

### P3 — CUDA graph 路线 A 复验（1–2 天）

前置：P0 完成（否则 decode 基线被低估，图的 7% 会被误读）。
门：decode ≤ −5%、TTFT 回退 ≤2%、c4 不回退、greedy 8/8 一致。

### P4 — 验收（按 construction §6.6 的 G0–G8）

Must 全绿 + 2 次重启 + 15 min soak + 回滚演练 + prefix caching on/off 各一次。

---

## 5. 明确不做（避免重走死路）

| 项 | 结论 |
|---|---|
| custom all-reduce（`VLLM_DCU_CUSTOM_AR`） | **判死**：小消息 +21.7%~+53.2% 变慢，e2e c1 TPOT +1.9~+6.3%，c8 吞吐 −2.8% |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | **判死**：NaN/Inf、accept 0、c=4 GPU VMFault |
| 草稿塔改 fp16 | **判死**：4/4 worker kv dtype 冲突 + fp16 溢出 MLP 激活 |
| 草稿塔 weight-only 量化 | **阻塞**：草稿塔必须 bf16 激活，gfx906 fp8 内核无 bf16 激活 datapath |
| spec k 值扫描 | 已证 k=7 最优 |
| GDN C++ kernel 微调（`__launch_bounds__`/block） | 上限 0.73–1.03% —— 要动就走布局（R1），不要微调 |
| O5 量化融合（`FP8_FUSE_QUANT`） | kernel 逐 bit 正确但 e2e **+54.0%** |
| O4 stream 路由（`VLLM_DCU_FP8_STREAM_MAX_M`） | parity 8 条 5 条分歧，不发布 |
| fp8 prefill tile/kFBk 调优 | cfg10 已近上限，无 ≥5% 候选 |
| 短算子融合 | 无单一 >5% 项 |
| `VLLM_DCU_FP8_CVT` | 死旋钮（全仓无消费者） |
| 4 卡时钟不对称 | 本 run 4 rank 差 <1%，不构成瓶颈（实测否证） |

---

## 6. 风险台账

| 风险 | 影响 | 缓解 |
|---|---|---|
| Triton 3.6.0(rocm101) 与 DTK 自编 triton 行为不同 | A1/A2 可能 miscompile → 数值错 | 先跑 parity 门槛（`perf1_gdn_chunked_parity.py`、`perf1_gdn_parity.py`），**过不了就不启用** |
| A2 的 CPU-launch 反例 | 在 CPU-bound 窗口里可能 **+9.4 ms/步** | 先用 `PROBE=1` 量 host/gpu 计时，再决定是否整轮 |
| 显存已 93% | A 组任何"多缓存一份"的方案都放不下 | 本轮所有候选都不增加常驻显存；A1/A2 的 Triton 链路需查 workspace |
| 图模式跨重启不可复现 | 质量/SLA 口径不可用 | 图只用于延迟/吞吐；SLA 留 eager |
| A6 与 port-plan 记录冲突 | 违背既有决定 | 标注为"用户决策项"，只测不默认开 |
| A8 的 GPU wedge | 长分块 prefill 下 compute ring 卡死，需 kill 进程 | 只做短 ctx；挂超时/ring 观测；优先做"消掉 drain 必要性"而不是切开关 |

---

## 7. 开放问题（本报告无法回答，需补测）

1. decode 每步的真实 GPU 空隙是多少？（`traces/*` 空，遥测只能给到 39.3% 利用率）
2. 149 次 `_local_scalar_dense` 已定位到 `causal_conv1d` 的 torch 回退路径（R6），
   **未定的是净收益**：消掉 149 次 DtoH 与引入 48 次流同步（或消掉 drain 的必要性）哪个更划算。
3. prefill 的 fp16 `aten::mm` 究竟打到 rocBLAS 还是 hipBLASLt？Tensile 库路径对不对？
4. M∈[9,32]（c≥2 并发）为什么步进 +60% 而权重流量不变？
5. 312 W 单点越限是真越限还是读数问题？（温度仅 65 ℃，不像墙）
6. "192 vs 269" 的 docker-vs-host 差距至今**没有任何受控 A/B**（4 个变量同时变：runtime / MAX_NUM_SEQS 128→4 / prefix cache / 树）。

---

## 8. 附录：复算命令

```bash
R=$DATA_ROOT/port/perf-runs/20261024-053333-E2C

# 微基准屋顶（GB/s / TFLOP/s）
python3 -c "
import json;w=json.load(open('$R/microbench_windows.json'))
for k,v in w.items():
    d=(v['t1_mono_ns']-v['t0_mono_ns'])/1e9
    print(f\"{k:22s} flops={v['flops']/d/1e12:6.2f}T  bytes={v['bytes']/d/1e9:6.1f}GB/s\")"

# 微基准 kernel 级拆分（去 RNG 污染）
python3 -c "
import json;d=json.load(open('$R/microbench-report/kernel_report.json'))
for k,v in d['windows'].items(): print(k, v['categories_us'], v['kernel_total_ms'])"

# 服务侧 op 级表（每 case × rank）
head -12 $R/kernel_op_tables/512c1__rank0.profiler_out.txt
tail -3  $R/kernel_op_tables/512c1__rank0.profiler_out.txt

# 遥测：负载窗口利用率/功率
python3 -c "
import json,statistics as st
r=[json.loads(l) for l in open('$R/gpu_telem.jsonl') if l.strip()]; t0=r[0]['ts']
act=[x for x in r if 6<=int((x['ts']-t0)//60)<=10]
print('n',len(act),'util',round(st.mean([x['util_perc_mean'] for x in act]),1),
      'W/card',round(st.mean([x['power_w_sum'] for x in act])/4,1))"

# 本 run 到底开了哪些旋钮
grep -E '^\s*(export )?(VLLM_|HSA_)' $R/serve_launch.sh | sort

# 与 DTK 生产配方做差集
diff <(grep -oE 'VLLM_[A-Z0-9_]+' tools/dcu/serve_host_dflash2.sh | sort -u) \
     <(grep -oE 'VLLM_[A-Z0-9_]+' $R/serve_launch.sh | sort -u)
```
