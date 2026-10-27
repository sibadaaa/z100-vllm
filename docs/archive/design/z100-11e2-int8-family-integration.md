# S11e2：INT8 `sdot4` 算子并入 gfx906 仿真内核族

> 上层文档：[S11f FP8/BF16 仿真内核](z100-11f-fp8-bf16-emu-kernels.md) · [性能优化方案 §4](../海光Z100-性能优化方案.md)
> 前置：S11e（W8A8 INT8 内核落地）、S11k（`torch.compile` 集成）、S11f（FP8/BF16 仿真内核）
> 状态：✅ 已完成（归档见 [`docs/result/z100-11e2-int8-family-integration-result.md`](../result/z100-11e2-int8-family-integration-result.md)）

## 目的

S11e/S11k 已经把 W8A8 INT8 的 `v_dot4_i32_i8` 内核做进仓库并接到 `_POSSIBLE_INT8_KERNELS`
首位，但它和本轮的 FP8/BF16 仿真内核是**两套独立的东西**：开关各自为政、只有
`tools/dcu/s11e_sdot4_eval.py` 这一份离线评测、`tests/` 下**没有任何单测**、
`can_implement` 只接受 per-channel 权重 scale + 动态 per-token 输入量化。
本步把它提升到与 FP8/BF16 内核同一档：

1. **同一族开关**：`VLLM_DCU_USE_GFX906_EMU=1` 一次打开三个 gfx906 仿真内核
   （INT8 `sdot4`、FP8 emu、dense fp16/bf16 GEMV），**逐个内核的显式设置优先于总开关**
   （`VLLM_DCU_USE_INT8_SDOT4=0` 仍能单独关掉 INT8）。未设置时行为与之前**逐位一致**。
2. **同一套 scale 契约**：权重 scale 支持 per-channel `[N]` **或 per-tensor `[1]`**，
   输入 scale 支持动态 per-token `[M]` **或静态 per-tensor `[1]`**（1 元素按 stride 0 施加）
   —— 与 `fp8_emu_gemm.cu` 完全一致。
3. **同一档测试**：`tests/kernels/quantization/test_gfx906_int8_sdot4.py`（int32 位精确、
   去量化参考、四种 scale 组合、bias、K 非 tile 整除、门控组合、`custom_op` + fake）。
4. **文档同源**：本设计文档 + 归档，并在 S11f 文档的开关表里登记总开关。

## 事实（先读）

| 事实 | 出处 |
|---|---|
| 内核已存在且与 `triton_scaled_mm` **逐位相同**（126/126 组）、decode 形状内核 ×4.9–22.6 | [S11e 归档](../result/z100-11e-w8a8-sdot4-result.md) |
| `torch.compile` 集成已修（`custom_op` + `register_fake`，0 graph break）、`=1` 端到端 **3.9×** | [S11k 归档](../result/z100-11k-sdot4-compile-integration-result.md) |
| 仓库默认仍为 OFF：`=1` 会**数值性**地改变 greedy token（并列点翻转），默认必须逐位稳定 | [S11l](../result/z100-11l-sdot4-reacceptance-result.md) / [S11m](../result/z100-11m-aot-restart-reproducibility-result.md) |
| 生产配方显式 `VLLM_DCU_USE_INT8_SDOT4=1` | [性能优化方案 §4.1](../海光Z100-性能优化方案.md) |

**因此本步不改默认值**：总开关默认 False，INT8 单开关默认 False；"集成"落在
开关组合、scale 契约、测试与文档上，不落在默认行为上。

## 做法

### 1. 内核（`csrc/rocm/w8a8_sdot4.{h,cu}`）

* GEMV/GEMM 两个 kernel 增加 `x_s_stride` / `w_s_stride` 参数，epilogue 改为
  `x_s[row * x_s_stride]`、`w_s[col * w_s_stride]`；`check_common` 放宽为
  `x_s.numel() ∈ {M, 1}`、`w_s.numel() ∈ {N, 1}` 并据此给出 stride。
* 契约与 `csrc/rocm/fp8_emu_gemm.cu` 的注释互相引用，避免两套语义漂移。
* 计算路径**零改动**：仍是 `v_dot4_i32_i8` int32 精确累加、epilogue 才乘 scale，
  因此原有"与 `triton_scaled_mm` 逐位相同"的性质不变（回归见归档 §4）。

### 2. Python（`gfx906_sdot4.py`）

* `can_implement`：只保留"输入必须对称"这一条硬约束（kernel 不实现 azp 修正项），
  不再拒绝静态输入量化与 per-tensor 权重 scale。
* `process_weights_after_loading`：per-tensor 权重 scale 原样保留（`numel == 1`），
  fused 模块仍按 `convert_to_channelwise` 展开；静态输入 scale 保留为 1 元素
  并校验必须是标量；非对称量化所需的 `input_zero_point` / `azp_adj` 仍然清空。
* `apply_weights`：静态输入走 `ops.scaled_int8_quant(x, i_s, None, symmetric=True)`，
  动态仍走 `scale=None`（base 类注释明确"static 恒为 per-tensor、dynamic 恒为 per-token"）；
  形状/scale 不匹配时继续回落 `triton_scaled_mm`。
* 模块 docstring 更新：删掉"直到编译集成修好才默认关闭"的过期理由（S11k 已修），
  改为"默认关闭是因为它会数值性地改变 greedy token"。

### 3. 开关（`gfx906_common.py`，新）

`gfx906_emu_enabled("<PER_KERNEL_ENV>")`：**显式设置**的单项开关优先，否则跟随
`VLLM_DCU_USE_GFX906_EMU`。实现上用 `os.environ` 读原始值以区分"未设置"与"显式 0"
（`vllm/envs.py` 的 bool 访问器保持不变，读它的旧代码行为不变）。
三个内核（INT8 / FP8 / dense GEMV）都改走该入口。

### 4. 测试（`tests/kernels/quantization/test_gfx906_int8_sdot4.py`，新）

28 项：int32 位精确（M ∈ {1,4,8,32,64}）、去量化参考（M ∈ {1,8,16,129} ×
fp16/bf16/fp32 输出，容差取输出 dtype 的半个 ulp）、per-channel/per-tensor ×
动态/静态四种 scale 组合、bias、K ∈ {12,68}（非 tile 整数倍）、K%4≠0 与
scale 长度错误必须抛错、总开关 × 单开关的四种组合、`can_implement` 矩阵、
`vllm::w8a8_sdot4_mm` 的 schema 与 fake 输出形状。

## 验收

1. `pytest tests/kernels/quantization/test_gfx906_int8_sdot4.py` 全绿；
   `test_gfx906_fp8_emu.py` / `test_gfx906_bf16_gemv.py` 不回归（三套合计 56 passed）。
2. `tools/dcu/s11e_sdot4_eval.py --mode int` 仍然 **bit_equal 全 True**、
   `--mode check` 的 rel_err 与 S11e 归档同量级（证明 stride 改动没有动计算路径）。
3. 开关组合：未设置 → 关闭；`VLLM_DCU_USE_GFX906_EMU=1` → 打开；
   再显式 `VLLM_DCU_USE_INT8_SDOT4=0` → 关闭。
4. 机器码：`v_dot4_i32_i8` 计数与 S11e 同档（内核结构未变）。
5. `isa_probe.sh` 基线（53 条）不受影响。

## 失败处置

| 现象 | 处置 |
|---|---|
| `--mode int` 出现 bit_equal=False | 立即回滚 stride 改动：计算路径不应受影响，先查 epilogue 索引是否越界读 |
| fused 模块 + per-tensor 权重 scale 报 `w_s.numel()` 错误 | `convert_to_channelwise` 未生效（`logical_widths` 缺失）；改回拒绝该配置或补转换 |
| 静态输入量化数值偏差 | 检查 `ops.scaled_int8_quant` 的 `scale` 是否是 1 元素；非对称配置应被 `can_implement` 拦下 |
| 总开关打开后 FP8/bf16 行为变化 | 只应改变选核；用 `VLLM_DCU_USE_FP8_EMU=0` 单独关掉对照 |

## 不做（明确边界）

* **不改默认值**（见"事实"表）。
* 不实现非对称激活量化（zero point / azp 修正）——需要额外的 azp 项，且 gfx906 的
  生产模型（27B W8A8）用的是对称动态量化。
* 不做 INT8 的 MoE/GGUF 变体（GGUF 的 `__dp4a` 路径已独立存在于 csrc/quantization/gguf）。
