# S99 prefill GEMM per-channel int8（单累加器拓扑）—— 执行结果

> 状态：**出口 A，已投产（2026-11-15 14:43，用户裁决"先应用到最新生产"）**。
> 8k 主门 −16.9%、131k 红线 −11.8%、decode 逐位、G3 质量电池全绿。
> 生产 = S99 变体 `.so`（`3ccebe80`）+ `VLLM_DCU_PREFILL_INT8_PC=1`，
> 策略指纹 `32f56158444b324e`；生产态复测 8k **6816.7 ms（−17.1%）**。
> 采纳与回滚见 `.s99/deploy-adopt.md`。
> 设计与施工：[design](../design/z100/s99-prefill-int8-pc-gemm/z100-s99-prefill-int8-pc-gemm-design.md) /
> [construction](../design/z100/s99-prefill-int8-pc-gemm/z100-s99-prefill-int8-pc-gemm-construction.md)。
> 台账与原始数据：`.s99/`（`.s99/README.md`、`.s99/logs/`、`.s99/state/anchor.md`）。

## 0. 摘要（截至窗口前）

| 项 | 结果 | 门 |
|---|---|---|
| WP0 HB7：8k 稳态 GEMM 份额 | **67.2%**（TTFT 8.36 s / GEMM 5.41 s） | ≥50% ✅ |
| WP1 HB2：冠军拓扑复现 | **33.3 / 33.8 TOPS**（fp16 对照 20.4/19.9） | ≥33 ✅ |
| WP1 HB4：staging | **0.114 s/卡/chunk** | ≤0.53 ✅ |
| WP2 内核 | 新 op **33.0 / 33.6 TOPS**（生产 sdot4 22.6）；vgpr 128 / spill 10 / dot4 384 | — ✅ |
| WP3.1 G1/G5 | max rel **1.443e-2**（全张量）/ **1.299e-2**（每卡分片）；q_mean ∈[−0.006,+0.028] | ≤1.5e-2 ✅ |
| WP3.1 误差分解 | 权重 9–11e-3 固定；激活 8.8e-3（高斯）～6.8e-2（通道离群） | 诊断 |
| WP4 路由 | 14/14 单测 + 13/13 路由 + 4/4 生产钩子集成；op 面 16→19（无缺失） | decode 结构性不可达 |
| **WP5 实测（8k 主门）** | **8219 → 6834 ms，−16.9%**（B 两轮 6841/6826） | ≥−12% ✅ |
| WP5 131k 红线 | 230.3 → **203.1 s，−11.8%** | ≥−5% ✅ |
| WP5 32k | 36.91 → 29.75 s，−19.4% | 记账 |
| WP4 decode 逐位 | c=1/c=4 md5 两臂**完全相同** | ✅ |
| G3 质量电池 | 6/6、6/6、cjk 0.883（与 A 逐用例一致） | 一票否决 ✅ |
| 8k ITL | 37.98 → 39.1~39.4 ms（+3.3%） | ±3%（略超，待复测） |

## 1. 两个图纸修正（执行中发现，已落账）

### 1.1 HB1 的"纯 LUT"不成立

真实 checkpoint 的每个 128×128 块 fp8 码 amax **恰为 448**（43,520 块全命中，
HB1 判据绿），但块缩放 `weight_scale_inv` 在通道内跨 k 块变化最高 **26×**。
无缩放 LUT（`round(f8·127/448)`）的权重误差 **1.5e-1～4.6e-1**；staging 必须
带 per-block 缩放乘法（设计图 §1.2 的公式本身是对的）。带宽账不变。

### 1.2 生产 prefill 不是 W8A16，而是 fp16 物化 + rocBLAS

8k trace：`Cijk_Alik_Bljk_HBH_MT128x128x16` ≈5.1 s + `fp8_dequant_to_fp16_kernel`
698 次。`VLLM_DCU_FP16_PREFILL=auto` 的角色阈值（gate_up 128 / down 16）远低于
8192 ⇒ `apply_weights` 先走 `gfx906_fp16_prefill.try_prefill`。设计图 §1.3 的
插入点 `_apply_block_scaled_mm_w8a16()` 在生产 prefill 上不会被调用。

派发因此改为**两处钩子**（默认关）：主路径 `_try_prefill()` 内、`_materialize()`
之前；兜底保留 `_apply_block_scaled_mm_w8a16()`。

## 2. 实现

- `csrc/rocm/sdot4_pc.hip`（新文件，442 行）+ `sdot4_pc.h`；生产
  `w8a8_sdot4.hip` 零改动（md5 `bf7a678b` 未变）。三个新符号：
  `fp8_stage_int8_pc` / `w8a8_sdot4_gemm_pc` / `fp8_emu_mm_int8_pc`；
- 绑定追加在 `torch_bindings.cpp`（不动现有符号），`CMakeLists.txt` 登记；
- 构建 `.s99/build.sh` 显式重编 bindings / sdot4 / 新文件 / 生产 attention 源，
  `strings | grep -c int8_pc = 28`；变体 `.so` md5 `3ccebe80`；
- 派发 `gfx906_fp8.py`：`VLLM_DCU_PREFILL_INT8_PC`（默认 0）、
  `..._MIN_M`（默认 2048）、豁免表（空：k/v 实测仍优于 W8A16）；
  不满足原路返回，无第三分支；`torch.compile` 走
  `vllm::fp8_emu_mm_int8_pc_auto` opaque body（fake 已注册）。

## 3. 数值与质量证据

### 3.1 内核单测（14/14）

staging 逐位（含 LUT 全 256 码、amax 边界、零行、block 64/64、行 stride 填充）、
GEMM 奇数形状/bias/零通道、真实权重 vs W8A16（rel 1.275e-2）。

### 3.2 G1（对生产真身，7 形状 × M∈{2048,8192}）

| 形状 | M=2048 | M=8192 |
|---|---:|---:|
| mlp.gate/up | 1.276e-2 | 1.277e-2 |
| mlp.down | 1.442e-2 | 1.443e-2 |
| attn.q_proj | 1.344e-2 | 1.345e-2 |
| attn.k/v | 1.402e-2 | 1.403e-2 |
| attn.o_proj | 1.343e-2 | 1.346e-2 |
| in_proj_qkv | 1.280e-2 | 1.283e-2 |
| in_proj_z | 1.323e-2 | 1.325e-2 |

每卡分片形状（8704×5120 / 5120×4352 / 4096×5120 / 5120×1536 / 3584×5120）：
max **1.299e-2**。

### 3.3 误差归因

权重项恒 9–11e-3（与 S86 §3.3 一致）；激活项在高斯下 8.8–9.4e-3，行内/通道
离群下 2.2e-2～8.2e-2。与厂商 W8A8 checkpoint（同契约：per-channel int8 权重 +
dynamic per-token int8 激活）对比：我们的 int8 与其差 ≈3.0e-2，其中 ≈2.85e-2
来自 fp8 checkpoint 自身相对 bf16 的误差，int8 重编码只再加 ~1%。

⇒ **G3 质量电池是决定性门**；截断、α 搜索均已实测为负结果，per-block 激活
缩放 = S86 Route B（已闭案）⇒ 若 G3 红即出口 C。

## 4. 窗口待办（用户批准后）

| 项 | 方法 | 门 |
|---|---|---|
| G3 | `.s99/quality_battery.py`（同 S15 框架，A0 两次 6/6+6/6+cjk0.883 确定性） | 工具 6/6、推理 ≥4/6、中文连贯、无退化 |
| G4/G2 | `.s99/logprob_score.py`（同文本 echo 打分，638 位置） | max|Δ| ≤0.35 nat；报 KL≈0.5E[Δ²] |
| WP4 | `.s99/decode_md5.py`（c=1/c=4 文本 prompt） | A/B md5 全同 |
| WP5 | `svc_probe` 8k/32k/131k，A/B×3 交替 | 8k ≥−12%；131k ≥−5%；TPOT ±3% |
| R6 | 8k 阶段账 GEMM 段 | 5.41 → ~3.3 s |

预登记：8k B 臂中位 ≤ 7.23 s 为过（预测 6.08 s）；131k > 218.8 s 为红。
作业单与驱动：`.s99/window-plan.md`、`.s99/window_arm.sh`（默认 dry-run）。

## 5. 决定性命中：运行期钩子在 `materialize_or_tile`

第一次 B 臂（只挂 `_try_prefill`）**零收益**（8.214 s = A）。取证：
`torch.compile` 下 `_try_prefill` 只跑在 trace 期；运行期执行的是
`vllm::fp8_emu_mm_block_w8a16_auto` 的 body `materialize_or_tile`
（内部再调 `_materialize`）。钩子移入该函数后立即生效（worker 探针
`eligible=True`，8k 6.84 s）。三处钩子最终形态：
`materialize_or_tile`（主）+ `_try_prefill`（eager）+ `_apply_*`（兜底）。

## 6. 投产记录（2026-11-15 14:43）

| 项 | 值 |
|---|---|
| 生产 `.so` | `.s99/arms/build_pc/_rocm_C.abi3.so`（`3ccebe80`） |
| 开关 | `serve_deploy.sh`：`VLLM_DCU_PREFILL_INT8_PC=1`、`..._MIN_M=2048` |
| 指纹 | `32f56158444b324e`（两个新 env 已进 KNOBS） |
| 生产 8k TTFT | **6816.7 ms**（6804.1/6829.3/6816.7，−17.1%） |
| 生产 8k ITL | 38.09 ms（+0.3%） |
| decode md5（c=4） | 与采纳前完全相同 |
| 质量电池 | 6/6、6/6、cjk 0.883（与采纳前逐用例相同） |
| 回滚 | 注释两行 export + restart（`.s99/deploy-adopt.md`） |

## 7. 生产零改动核对（窗口后）

| 项 | 期望 | 当前 |
|---|---|---|
| `csrc/rocm/w8a8_sdot4.hip` | `bf7a678b` | ✅ |
| `csrc/rocm/fp8_emu_gemm.hip` | `16018d1b` | ✅ |
| `csrc/rocm/attnd_inner.hip` | `b93db390` | ✅ |
| `csrc/rocm/attnd_inner_kvint8.hip` | `5926429c` | ✅ |
| `vllm/_rocm_C.abi3.so` | `.s97/arms/build_phsr`（`adba0acb`） | ✅ |
| 服务 | active，FP `8bf8b075a6469af0` | ✅（10:41 因探测事故恢复性重启一次，argv/env/`.so` 未变） |
