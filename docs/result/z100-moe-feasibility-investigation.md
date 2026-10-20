# Z100 适配 MoE 模型可行性调查报告

> 日期：2026-10-19 · 范围：`/home/lyh/z100-vllm`（vLLM gfx906 / Hygon Z100 分支）
> 方法：README / docs / git 历史 / 代码静态追踪（只读，未改动任何代码）
> 关联文档：`docs/result/F24-investigation-20261018.md`、`docs/z100-torch216-rocm101-adapt-plan.md`

## 0. 结论摘要（先看这个）

**结论：MoE 在 Z100 上"可行"，但可行性由三个与稠密模型完全不同的约束决定，而不是算力。**

1. **可行**，且已有实测证据：`unsloth/Qwen3.6-35B-A3B-GGUF` 是本仓库唯一列入"已测试并确认可用"的 MoE 模型；qwen35moe GGUF 的多模态与 MTP 也已在单张 32 GiB gfx906 上实测通过（fork 时代自测，见 §4）。
2. **真正的阻塞点不是 MoE 架构，而是"内核是否在 DP4X 故障包络内"**。Z100（DP4X）卡上，VGPR 编译到 256 的内核会静默错算（F24）。当前 `hygon_dp4x.py` 的运行时 guard **只覆盖 Triton 内核**——而**能跑通的 GGUF MoE 和出问题的 WNA16 MoE 恰恰都走 C++ HIP 内核，不在 guard 覆盖范围内**（§2.3）。
3. **量化选择决定成败**：非量化 MoE 走 Triton（**已核实 guard 确实覆写 `num_warps`，风险可控**，§2.2）；MoE + GPTQ-Int4 / AWQ-4bit 是 README 明确记载的失败组合，根因是**gfx906 上 Triton 的 GPTQ/AWQ fused MoE 内核根本无法编译**（§2.3）；GGUF MoE 走 C++ 内核（能跑但不受 guard 保护）。**aiter 后端不是"不稳定"而是硬性不可用**（`get_cdna_version()==0`，§2.4）。
4. **拓扑上有一条"干净"路径**：本机 4 卡中只有卡 0/1 是 Z100L（DP2X，无此问题），卡 2/3 是 Z100（DP4X）。**只用卡 0/1 做 TP=2 可以完全绕开 DP4X 风险**；TP=4 必须用满 4 卡，必然包含 2 张 DP4X 卡。
5. **显存是第二约束**：30B 级 MoE 在 fp16 下需 TP=2~4（且撞主机 30 GiB 内存）；int4 可单卡，但 int4 正好撞上已知失败组合。**≤16B 的非量化 MoE 是风险最低的切入点**（OLMoE-1B-7B 原生 fp16 单卡仅 12.9 GiB）。

**建议路径（按风险从低到高）**：`OLMoE-1B-7B（原生 fp16 单卡）` → `DeepSeek-V2-Lite（int8 单卡）` → `Qwen3.6-35B-A3B-GGUF（已在 gfx906 实测，待补 Z100 验证）` → `Qwen3-30B-A3B W8A16 TP=2`（唯一有深度调优配置的组合）→ 不建议 `MoE+GPTQ-Int4/AWQ-4bit`、`GPT-OSS-20B（需先重新量化）`、`Mixtral fp16`、`235B`。

---

## 1. 硬件与软件基线

### 1.1 实测硬件拓扑（本机 KFD topology 读取）

| torch 卡号 | KFD node | PCI bus | 型号 | `dp4x_status` | 显存 | 判定 |
|---|---|---|---|---|---|---|
| 0 | 1 | 0x2e | **Z100L** | 0 | 32 GiB | DP2X，**正常** |
| 1 | 2 | 0x33 | **Z100L** | 0 | 32 GiB | DP2X，**正常** |
| 2 | 3 | 0x38 | **Z100** | **1** | 32 GiB | **DP4X，VGPR=256 静默错算** |
| 3 | 4 | 0x3b | **Z100** | **1** | 32 GiB | **DP4X，VGPR=256 静默错算** |

- 4 卡均 `gfx_target_version=90106`（gfx906 / Vega20），`vendor_id=7572`（0x1d94 Hygon）。
- 这是 **混合 SKU 部署**：一半的卡带有已知的 DP4X 缺陷。**任何 TP≥3 的配置都会包含故障卡。**

### 1.2 ISA 能力（决定 MoE 权重格式的可行域）

依据 `docs/archive/result/z100-11d-z100-isa-capability-audit-result.md`（机器码级审计）：

| 能力 | gfx906 状态 | 对 MoE 的影响 |
|---|---|---|
| `v_mfma` / `v_wmma`（矩阵核心） | **无** | 所有 GEMM 只能走 VALU/`v_dot*`；MLP 专家计算是纯 SIMT |
| `v_dot4_i32_i8`（sdot4） | **有**（`_C.abi3.so` 中 71,442 条） | INT8 点积可用；GGUF `__dp4a` 与 sdot4 W8A8 就靠它 |
| `v_dot2_f32_f16` | **有** | fp16 点积累加，MoE fp16 GEMM 主力 |
| `v_pk_fma_f16` / `v_fma_mix_f32` | **有** | 仿真内核基础 |
| bf16 转换/指令 | **无原生**（dense bf16 走 fp32 仿真） | bf16 MoE 权重会退化，优先 fp16 |
| FP8 / FP4 硬件 | **完全没有** | FP8 走 fp16 仿真（`csrc/rocm/fp8_emu_gemm.cu`）；**FP4/MXFP4 无对应路径** |

### 1.3 DP4X 故障（F24，MoE 的直接相关性）

来自 `docs/result/F24-investigation-20261018.md`：

- **判据**：Z100（DP4X）卡上，**任何 VGPR 分配取整到 256 的内核**（kernel descriptor `VGRPBlocks 63`，`n_regs>=253`）会在第一条 global store 处被静默中止，只写出 72 个元素，**dmesg 零故障**；`n_regs<=203` 全部正确。
- **两栈差异**：DTK 26.04 用户态（HIP 6.3）**有** DP4X 支持；自编 ROCm 10.1 + torch 2.16（HIP 7.16）**完全没有** DP4X 代码。
- **已落地 workaround**：`vllm/platforms/hygon_dp4x.py` 在进程内包装 `triton.runtime.jit.JITFunction.run`，DP4X 卡上强制 `num_warps>=8`（默认），把 Triton 内核的 VGPR 压到故障包络以下。
- **残留缺口（关键）**：guard **只覆盖 Triton**。F24 §12 用 rocprofv3 追踪确认，guard 之后最后一个未被覆盖的高 VGPR 内核是 **RCCL allreduce `ncclDevKernel_Generic_4<5120>`（VGPR=248）**，并且**纯 allreduce 循环即可在 Z100 上复现 XID:81 → XID:76 kill**（§12.4）。**C++/库内核不在保护范围内，这是一个尚未关闭的故障面。**

---

## 2. MoE 在 vLLM 中的代码路径与 gfx906 选型

### 2.1 选型总表（本节是最重要的工程结论）

| 权重类型 | gfx906 上实际走的内核 | 实现语言 | DP4X guard 覆盖？ | 风险 |
|---|---|---|---|---|
| **非量化 fp16 / bf16** | `fused_experts` → `invoke_fused_moe_kernel`（`TritonExperts`） | **Triton** | ✅ **覆盖** | 中（guard 生效） |
| **GGUF**（Q4_K / Q6_K / …） | `_fused_moe_gguf` → `ops.ggml_moe_a8`（prefill, MMQ）/ `ggml_moe_a8_vec`（decode, moe_vec） | **C++ HIP** | ❌ **不覆盖** | **高**（无保护） |
| **W4A16 / W8A16（GPTQ/AWQ, group 32/64/128）** | `should_moe_wna16_use_cuda`→True → `ops.moe_wna16_gemm` | **C++ HIP** | ❌ **不覆盖** | **高**（且 README 记载失败） |
| W4A16/W8A16 非标准配置（bits∉{4,8} 或 group∉{32,64,128}） | `_apply_gfx906_torch_fallback`（纯 torch） | Python/torch | 部分（算子为 ATen） | 中（安全但极慢） |
| WNA16 且 `should_moe_wna16_use_cuda=False` | `invoke_fused_moe_wna16_triton_kernel` | Triton | ✅ 覆盖 | 中 |
| FP8 per-tensor | `Fp8MoEMethod`（fp16 仿真） | HIP + Triton | 部分 | 中 |
| MXFP4 / NVFP4 | `GptOssMxfp4MoEMethod` 等 | Triton | ✅ 覆盖 | **高**（gfx906 无 FP4，仿真成本大） |
| **aiter**（ROCm 原生 MoE 后端） | **不可用** | — | — | 已排除（见 §2.4） |

### 2.2 非量化 MoE：Triton 路径（guard 覆盖，推荐）

- `UnquantizedFusedMoEMethod.apply` → `fused_experts`（`vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:283`）。
- 选型落到 `TritonExperts`；配置来自 `get_default_config`（`fused_moe.py:1312`），gfx906 走通用分支：
  - `num_warps = 4 if M <= 128 else 8`（`fused_moe.py:1414`）
  - ROCm 强制 `num_stages = 2`（`:1330-1331`，规避 `OutOfResources`）
  - `BLOCK_SIZE_M` 按 batch 16/32/64/128，`BLOCK_SIZE_N` 64/128，`BLOCK_SIZE_K` 64/128
- **在 Z100（DP4X）卡上，这些 Triton 内核会被 guard 统一提升到 `num_warps>=8`**，从而落在故障包络以下。**这是 MoE 场景中唯一"有保护"的路径。**
- **已核实 guard 确实生效**（这一点曾被列为待查风险）：`invoke_fused_moe_kernel` 在 `fused_moe.py:908` 以 `**config` 展开 launch 参数，`config` 来自 `get_default_config` 且**含 `num_warps` 键**；guard 包装的正是 `JITFunction.run`，从 `kwargs` 读 `num_warps`。因此 `get_default_config` 在 `M<=128` 时给出的 `num_warps=4`（`fused_moe.py:1414`）**会被 guard 覆写为 8**。→ **fp16 MoE 的 Triton 路径受保护，风险可控。**

### 2.3 GGUF / WNA16 MoE：C++ 路径（能跑，但不受 guard 保护）

- **GGUF MoE**：`GGUFMoEMethod.apply` → `fused_moe_gguf` → `_fused_moe_gguf`（`vllm/model_executor/layers/quantization/gguf.py:1690`），prefill 走 MMQ（`ops.ggml_moe_a8`）、decode 走 moe_vec（`ops.ggml_moe_a8_vec`）。两者都实现在 **`csrc/quantization/gguf/gguf_kernel.cu`**（`mmq.cuh` / `moe_vec.cuh` / `moe_vec.cuh`），是编译进 `_C_stable_libtorch` / `_C` 的 **HIP C++ 内核**。
- **WNA16 MoE**：`MoeWNA16Method.apply`（`vllm/model_executor/layers/quantization/moe_wna16.py:478`）在 gfx906 且标准配置下调用 `fused_experts`，而 `should_moe_wna16_use_cuda`（`fused_moe.py:1304-1309`）对 gfx906 **恒返回 True**：
  ```python
  if current_platform.is_rocm() and on_gfx906():
      return bit in (4, 8) and group_size in [32, 64, 128]
  ```
  → 路由到 **`ops.moe_wna16_gemm`（`csrc/moe/moe_wna16.cu`，HIP C++）**，而不是 Triton。
- **推论**：F24 §12 的结论直接适用——**这两条 MoE C++ 内核与 RCCL allreduce 属于同一类"guard 未覆盖的高 VGPR 风险"**。它们的实际 VGPR 数**尚未被审计**。**这是本报告识别出的首要未验证风险（R1）。**

### 2.3.1 R1 的直接证据：既有 VGPR 审计**结构上无法**覆盖 MoE

已核对 `/mnt/game/f24-investigation/06-vgpr-256-kernel-audit.md`（F24 的 VGPR 审计，解析 **11,317 个 hsaco**），结论有三点：

1. **该审计中 MoE 关键词 0 命中**（`moe|expert|ggml|wna16` 全文无匹配）。原因是当时 live serving cache 里**从未服务过任何 MoE 模型**——只跑过稠密 0.6B / 4B GDN / 27B-FP8。
2. 已确认的 VGPR=256 内核集合是 **9 个名字，全部是 Triton**（`kernel_unified_attention_2d/3d`、`chunk_fwd_kernel_o`、`_w8a8_triton_block_scaled_mm` 等）。**没有任何 MoE 内核**，与"MoE 未跑过"一致。
3. **方法学限制（关键）**：该审计是**纯 CPU 静态分析，只解析 Triton 的 `.hsaco` 缓存**（`vgpr_audit.py`，从 descriptor `granulated_workitem_vgpr_count` 读取）。**它对编译进 `.so` 的 C++ HIP 内核（GGUF MoE、`moe_wna16_gemm`）完全无能为力。** 因此 MoE 的 C++ 内核**不能用现成方法审计**，唯一可用路径是 F24 §13.4 的 **hip shim**（`hipFuncGetAttributes()` 运行时审计，已确认落盘于 `/mnt/game/f24-investigation/hipshim/libhiplog.so`，支持 `HIPLOG_VGPR_WARN` / `HIPLOG_VGPR_FAIL`）。

> 附带核对：该审计的 P0 建议是"把 guard 的 watcher 阈值从 `n_regs < 256` 改为 253"（253/254/255 与 256 同属故障包络）。**该修复已经落地**：`vllm/platforms/hygon_dp4x.py:60` 现为 `_DP4X_MAX_VGPR = 253`，并在 `_watch_compiled_kernel` 使用。

> 注：`should_moe_wna16_use_cuda` 的 gfx906 分支与 7 个 `AMD_GFX906` 调优配置均来自 **2026-08-18 commit `395959c82`**（"同步开发仓 gfx906 推理优化"）；而 README（最后更新 2026-09-28）**仍**把 MoE+GPTQ-Int4 列为失败。因此该改路由并**没有**修复 WNA16 MoE 的失败。

**WNA16 失败的直接机制（新证据）**：`moe_wna16.py:418` 的 `_apply_gfx906_torch_fallback` 自带 warning，原文说明了根因：

> `[gfx906] Falling back to Torch MoE WNA16 path because the Triton GPTQ/AWQ fused MoE kernel does not compile for gfx906.`

即 **gfx906 上 Triton 的 GPTQ/AWQ fused MoE 内核根本无法编译**（比 README 措辞"处理特定分块大小时出现内存访问错误"更准确）。当前逻辑因此对**非标准配置**（bits∉{4,8} 或 group∉{32,64,128}）走纯 torch 慢回退，对**标准配置**改走 C++ `moe_wna16_gemm`。**两条路都未经 gfx906 端到端验证**——README 的失败结论在这条改路由之后依然保留。

### 2.3.2 W4A16 GPTQ MoE 的实际执行路径：**已定论**

曾存在一个矛盾：模块化内核的 oracle 在 ROCm 上会选 `TritonWNA16Experts`，但 `MoeWNA16Method.apply` 走的是 legacy `fused_experts`。**经独立追踪，定论如下：**

**GPTQ（AutoGPTQ 配置）W4A16 MoE 在 gfx906 上执行的是 C++ `ops.moe_wna16_gemm`。模块化路径根本不会被实例化。**

决定性链条（每一步均可复核）：

1. `auto_gptq.py:238-250`：对 `RoutedExperts` **先**测 `check_moe_marlin_supports_layer(...)`，不支持则**直接返回 `MoeWNA16Config.get_quant_method(...)`**（:248），从而**完全跳过** `AutoGPTQMoEMethod`（oracle/`make_wna16_moe_kernel` 都在那里）。
2. `marlin_utils.py:369-370`：`if current_platform.is_rocm(): return False` ⇒ **所有 ROCm 设备（含 gfx906）恒返回 False** ⇒ 必然走 MoeWNA16 回退。
3. `moe_wna16.py:225-226` → `MoeWNA16Method`。**该类没有 `moe_kernel` 属性**（grep `moe_kernel moe_wna16.py` 无命中），从不调用 `make_wna16_moe_kernel` ⇒ **模块化机制不参与**。
4. `moe_wna16.py:478-541`：W4A16/gs=128 落到 `return fused_experts(...)`（:527）。
5. `fused_moe.py:947` → `should_moe_wna16_use_cuda`（:1304-1306）gfx906 恒 True → `invoke_fused_moe_wna16_cuda_kernel` → **`ops.moe_wna16_gemm`**（:652；绑定 `csrc/moe/torch_bindings.cpp:83,166`；内核 `csrc/moe/moe_wna16.cu`）。

> **重要限定（不要过度泛化）**：以上结论针对 **GPTQ / AutoGPTQ 配置**。若同一 W4A16 权重经由 **compressed-tensors WNA16** 或 **Quark** 加载，则会构建模块化内核并落到 `TritonWNA16Experts`（`oracle/int_wna16.py:90-91`；设备门禁 `experts/triton_moe.py:627-628` 确实接受 ROCm）——**那条路是 Triton，因此反而受 DP4X guard 覆盖**。两条路径的 DP4X 风险恰好相反，选加载器时要留意。

### 2.4 aiter 后端在 gfx906 不可用（已定位精确门禁）

README（`已知限制` 第 4 条）提到"aiter 后端在复杂 MoE 路由场景下不稳定"，但**实际是根本不可用**：

- `is_aiter_found_and_supported()`（`vllm/_aiter_ops.py:203-223`）要求 `current_platform.is_rocm() and IS_AITER_FOUND and get_cdna_version() > 2`；
- `get_cdna_version()`（`vllm/platforms/rocm.py:453-462`）只对 gfx90a/gfx942/gfx950/gfx1250 返回非 0，**gfx906 返回 0** ⇒ aiter 关闭；
- `rocm_aiter_ops.is_fused_moe_enabled`（`_aiter_ops.py:2272-2274`）被 `@if_aiter_supported` 包裹，返回 `None`，平台在 `vllm/platforms/rocm.py:1218` 消费它 ⇒ `experts/rocm_aiter_moe.py` **永不选中**。
- 附带：RDNA3 WNA16 专家路径同样出局（`experts/rdna3_moe.py` 以 `on_gfx1100()` 为门禁）。

### 2.5 全部"guard 覆盖不到"的 MoE 相关 C++/HIP 内核清单

DP4X guard 只包装 Triton，以下 MoE 热路径内核**全部裸奔**（无 `num_warps` 类旋钮，无法用 guard 降 VGPR）：

| 内核 | 文件 | 作用 | 风险 |
|---|---|---|---|
| `moe_wna16_gemm` | `csrc/moe/moe_wna16.cu` | W4A16/W8A16 MoE GEMM | **最高**（实际执行的 MoE GEMM） |
| `ggml_moe_a8` / `ggml_moe_a8_vec*` | `csrc/quantization/gguf/gguf_kernel.cu`（`moe.cuh`/`moe_vec.cuh`） | GGUF MoE prefill/decode | **高**（唯一实测可跑的 MoE 路径） |
| `topk_softmax` | `csrc/moe/topk_softmax_kernels.cu` | MoE 路由 top-k | 中（非 GEMM，但在热路径） |
| `moe_align_block_size` / `moe_sum` | `csrc/moe/moe_align_sum_kernels.cu` | 专家分桶/归约 | 中 |
| `moe_permute_unpermute` | `csrc/moe/moe_permute_unpermute_op.cu` | 模块化路径 permute | 低（仅模块化内核） |
| `fused_globalize_align_block_size`、`grouped_topk`、`moe_lora_align_sum` | `csrc/moe/*` | 辅助 | 低 |

> 这与 F24 §12 的发现同源：guard 之后**最后一个未覆盖的高 VGPR 内核是 RCCL allreduce（248）**，而 MoE 又引入了 6+ 个同类未覆盖内核。**MoE 扩大了 DP4X 的暴露面。**

---

## 3. 显存 / 并行度可行性矩阵

**口径（已按 GiB 统一，避免与 32 GiB 卡混淆）**：`GiB = 参数量(B) × 字节数 × 0.9313`。字节/参数：fp16=2、int8/fp8=1、int4=0.55（含 group scale/zero-point）。预算：**≤26 GiB 舒适 / ≤30 GiB 极限 / >30 GiB 不可行**。**注意**：量化时 embedding / lm_head / norm 通常保持 fp16，故真实 int4/int8 占用比下表**高约 5–10%**（Qwen3.5 vocab 248k×2048 约多 1–2 GiB）。TP 还需满足 KV head 可整除。

| 模型 | 总参数 | 激活/tok | E / topk / **N** | fp16 | int8/fp8 | int4 | 单卡(32 GiB)？ | 最小 TP（每卡占用） | 调优配置 |
|---|---|---|---|---|---|---|---|---|---|
| **OLMoE-1B-7B** | 6.9B | 1.3B | 64 / 8 / 1024 | 12.9 | 6.4 | 3.5 | ✅ **全部格式** | **TP=1**（12.9 / 6.4 / 3.5） | ❌ |
| **DeepSeek-V2-Lite** | 15.7B | 2.4B | 64+2 / 6 / 1408 | 29.2 ⚠️ | 14.6 | 8.0 | ⚠️ fp16 极限（几乎无 KV 余量）；int8/int4 ✅ | **TP=1** 推荐 int8 或 int4 | ❌ |
| **Qwen3-30B-A3B** | 30.5B | 3.3B | 128 / 8 / **768** | 56.8 | 28.4 ⚠️ | 15.6 | ❌ fp16；⚠️ int8 紧；✅ int4 | fp16→**TP=2**(28.4 紧)；int8→TP=2(14.2)；int4→**TP=1**(15.6) | ✅ **仅 int8_w8a16**（E=128,N=768；14 档深度调优） |
| **Qwen3.5/3.6-35B-A3B** | ~35B | ~3B | **256** / 8 / **512** | 65.2 | **32.6 ❌** | 17.9 | ❌ fp16；❌ int8（32.6 > 32）；✅ int4 | fp16→TP=2 **放不下**(32.6)，需 TP=4；int8→TP=2(16.3)；int4→**TP=1**(17.9) | ✅ **仅 int4_w4a16**（E=256,N=512；仅 4 档） |
| **Mixtral 8x7B** | 46.7B | 12.9B | 8 / 2 / 14336 | 87.0 | 43.5 | 23.9 | ❌ fp16/int8；✅ int4 | fp16→TP=4(21.8，**必用故障卡**)；int4→**TP=1**(23.9) | ❌ |
| **GPT-OSS-20B** | 20.9B | 3.6B | 32 / 4 / 2880 | 38.9 | 19.5 | ~10.3 (**MXFP4**) | ❌ fp16；✅ int8/int4 单卡 | fp16→TP=2(19.5)；int4/int8→TP=1 | ❌ |
| **Qwen3-235B-A22B** | 235B | 22B | 128 / 8 / 1536 | 437.7 | 218.9 | 120.4 | ❌ **任何格式都不行** | int4 即使 TP=4 也是 30.1/卡且**零 KV 余量** ⇒ **4×32 GiB 不可行**（int4 实需 ~8 卡） | ❌ |

> **GPT-OSS-20B 的额外问题**：其原生权重是 **MXFP4**，而 gfx906 **无任何 FP4 数据通路**（§1.2）⇒ 必须先**重新量化成 W4A16/W8A16** 才可能跑，不能直接用原 checkpoint。

**调优配置覆盖情况**（`vllm/model_executor/layers/fused_moe/configs/` 中 AMD_GFX906 的 **7 个** JSON）。

> **两个查表口径**（两点都会影响结论）：
> 1. **`N = moe_intermediate_size`**：查表键为 `E, _, N = w2_shape`（`fused_moe.py:1447`），w2 是 down_proj、形状 `[E, hidden, intermediate]` ⇒ 取的是 intermediate 本身，**不是 2×**。
> 2. **dtype 精确匹配**：文件名形如 `E=<E>,N=<N>,device_name=AMD_GFX906[,dtype=<quant>].json`；fp16/bf16 传 `dtype=None` ⇒ **无对应文件，必然回退**；量化类型不匹配同样回退（`fused_moe.py:1094-1179`）。**即：命中调优配置需要 (E, N, dtype) 三者全对。**

| 配置文件 | 对齐的模型形状 | 调优深度（batch 档位数） |
|---|---|---|
| `E=128,N=768,int8_w8a16` | **Qwen3-30B-A3B**（E=128, I=768） | **14 档**（深度调优） |
| `E=128,N=192,int8_w8a16` | 未知（128 专家、I=192 的小专家模型） | **18 档**（深度调优） |
| **`E=256,N=512,int4_w4a16`** | **Qwen3.5/3.6-35B-A3B**（E=256, I=512）→ 依据本仓库 `vllm/transformers_utils/configs/qwen3_5_moe.py:66-69`（`moe_intermediate_size=512`、`num_experts=256`、`hidden_size=2048`、`num_hidden_layers=40`） | 仅 **4 档**（浅） |
| `E=256,N=128/256/1024,int4_w4a16` | 同族其它 intermediate 宽度 | 各 **4 档**（浅） |
| `E=20,N=1536,int4_w4a16` | 未知（20 专家模型，本报告候选中无匹配） | 4 档（浅） |

**关键推论**：

1. **7 个候选中只有 2 个有调优配置，且各自只在一种量化方案下命中**：
   - Qwen3-30B-A3B → 仅 **int8_w8a16** 命中（用 int4 或 fp16 则回退）；
   - Qwen3.5/3.6-35B-A3B → 仅 **int4_w4a16** 命中（用 int8 或 fp16 则回退）。
   - 其余 5 个**永远走** `Using default MoE config. Performance might be sub-optimal!`（**只影响性能，不影响正确性**）。
2. **Qwen3.5/3.6-35B-A3B 是 256 专家、moe_intermediate=512**（不是 128/768）——由本仓库自己的 config 默认值确证。参数账：每层专家 `256 × 3 × 2048 × 512 ≈ 0.805B`，×40 层 ≈ **32.2B**，加共享专家与 GDN/注意力 ≈ **35B**，与 "35B-A3B" 吻合。
3. **`E=256,N=512,int4_w4a16` 恰好匹配 35B-A3B W4A16** —— 而 W4A16/GPTQ-Int4 正是 README 记载**失败**的组合。**"调优配置存在但模型仍失败" ⇒ 问题在 kernel 而非 tile 配置**，与 §2.3.2 定的 C++ 路径结论一致。
4. **调优深度差异有信息量**：两个 `int8_w8a16` 有 14/18 个 batch 档（**认真调过**），全部 `int4_w4a16` 只有 **4 档**（**浅尝/模板生成**）⇒ 历史上对 **W8A16 MoE 的投入远大于 W4A16**，进一步支持"W8A16 比 GPTQ-Int4 更值得优先复测"。
5. **未被任何候选使用的 gfx906 配置**：`E=128,N=192,int8_w8a16`、`E=20,N=1536,int4_w4a16`、`E=256,N={128,256,1024},int4_w4a16` —— **E=256 的 int4 簇明显是为 Qwen3.5-MoE 家族调的**。

### 3.1 并行度与 DP4X 的耦合（重要）

| 配置 | 使用卡 | 是否含 DP4X 卡 | 评价 |
|---|---|---|---|
| TP=1（卡 0 或 1） | 1× Z100L | 否 | **最干净**，推荐首选 |
| **TP=2（卡 0,1）** | 2× Z100L | **否** | **推荐**：完全绕开 DP4X |
| TP=2（混 SKU，如卡 0,2） | 1× Z100L + 1× Z100 | 是 | 有风险 |
| TP=4（全部 4 卡） | 2× Z100L + 2× Z100 | **是** | 必然含 2 张故障卡；另需注意主机内存 |

> 主机内存约 30 GiB，历史记录显示 TP=4 加载期需要 swap（`docs/result/z100-14c-*.md`、S17）。**MoE 的专家权重分散，加载期 staging 压力与稠密模型不同，但同样需要预留。**

---

## 4. 已实测状态汇总（README + git 证据）

### 4.1 已验证可用

| 模型/能力 | 证据 |
|---|---|
| `unsloth/Qwen3.6-35B-A3B-GGUF` | `README.md` "已测试并确认可用" 列表（由 `911e2025d` 加入）——**本仓库唯一显式列出的 MoE 模型** |
| qwen35moe GGUF 生成 | `f811df4e7`："Enables qwen35moe GGUF models to load and generate correctly" |
| qwen35moe GGUF + **MTP=2** | `8cfc349ac`：draft 接受长度 ~2.5（72–85%），decode 50.5 vs 37.6 tok/s（**+35%**），输出连贯 |
| qwen35moe GGUF + **多模态**（mmproj） | `d456527b6`：图片 QA 3/3、正确识别纯红色 |
| 均"在单张 32 GiB gfx906 GPU 上实测可用" | `README.md:30` |

> **归属提醒**：以上为 **fork 时代（2026-09，gfx906 lineage）** 的自测结论，来自 commit body。Z100/DTK 重新基线化（2026-10）**未独立复测**这些 MoE 结论——`docs/archive/z100-todolist.md` 中 Phase B（"GGUF / MoE / MTP / 多模态"）状态仍为 **⬜ 未执行**。

### 4.2 已验证失败

| 模型 | 失败现象（原文） |
|---|---|
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | "MoE + GPTQ Int4 量化组合存在兼容性问题，导致服务无法正常启动或推理失败"；"**Triton kernel 在处理特定分块大小时会出现内存访问错误**"；"**aiter 后端在复杂 MoE 路由场景下不稳定**" |
| `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` | "在 ROCm 6.4 + gfx906 上实测仍无法正常运行，现象包括**返回整串 `!`** 或在引擎初始化/推理阶段崩溃" |
| 泛化声明 | `docs/archive/design/z100-10-phase-b-features.md:18`："已知不支持 \| **MoE + GPTQ-Int4、MoE + AWQ-4bit** \| 直接标注不支持，不排验证" |

### 4.3 未测试

- `README.md:58`：🔄 其他 MoE 模型（`exaone_moe`、`glm4_moe_lite` 等）——同上游同步进来但**未在 gfx906 上验证**。
- 候选矩阵中的 OLMoE / DeepSeek-V2-Lite / Mixtral / Qwen3-30B-A3B / GPT-OSS-20B **均无任何本机 MoE 实测记录**。

### 4.4 性能口径

- `README.md:255`："⚠️ **MoE 量化模型** - **速度显著较慢，不推荐**"。
- **重要限定**：该结论标注为"基于原项目测试结果"，指向上游 `nlzy/vllm-gfx906` 的 Issue #29；**本仓库没有任何本地 MoE 基准数据支撑这个数字**。不应把它当作 Z100 上的实测结论。
- 相邻证据（S22，`docs/archive/result/z100-22-*.md:124`）："长上下文步长才是吞吐天花板（32k→763 ms/步），要动它属于 **attention/MoE 内核工作**"——即 MoE 内核优化被列为**尚未开展的未来杠杆**。

---

## 5. 风险清单（按严重度）

| ID | 风险 | 严重度 | 依据 / 缓解 |
|---|---|---|---|
| **R1** | **GGUF MoE 与 WNA16 MoE 走 C++ HIP 内核（共 6+ 个 MoE 内核，见 §2.5），不在 DP4X guard 覆盖内；其 VGPR 数未审计，且现成的静态审计方法结构上无法审计它们** | **高** | F24 §12 已证明同类未覆盖内核（RCCL 248 VGPR）可在 Z100 上致 worker 死亡；`06-vgpr-256-kernel-audit.md` 11,317 hsaco 中 MoE 0 命中（§2.3.1）。缓解：用 hip shim（`HIPLOG_VGPR_WARN=248`）做运行时审计；或在 Z100 上只用卡 0/1 |
| **R2** | Z100（DP4X）卡占总卡数一半；TP≥3 必然包含故障卡 | 高 | 缓解：TP=2 on 卡 0,1；或在 DP4X 上验证目标 MoE 形状后再上量 |
| **R3** | MoE + GPTQ-Int4 / AWQ-4bit 已知失败（已实测） | 高 | 规避：改用 GGUF 或 W8A16 或非量化 |
| **R4** | **无 gfx906 专用 MoE 测试套件**；`tests/rocm/test_moe_*.py` 因缺 AITER 被整体 skip | 中高 | 缓解：新建最小 MoE 数值门禁（见 §7） |
| **R5** | 调优配置仅 7 个 JSON，候选模型多不覆盖 → 回退通用 tile | 中 | 正确性无影响，性能不可预期；必要时自建 tuned config |
| **R6** | MTP 正确性在本平台不可用（S18e：greedy 输出与非投机臂 8.4–12.1 nat 自信翻转） | 中 | 若 MoE 也要开 MTP，须先解决 K2 token-parity；否则 MTP 仅作性能探索 |
| **R7** | "MoE 显著较慢"来自上游 Issue #29，**无本地基准** | 中 | 需实测；不要把上游数字当 Z100 结论 |
| **R8** | 主机 RAM ~30 GiB，TP=4 加载期需 swap，且历史出现 `/health` 假健康的 OOM | 中 | 沿用 `docs/result/z100-14c-*.md` 的配方（AOT 重载 + swap + swappiness） |
| **R9** | bf16 MoE 无原生支持（无 bf16 指令） | 低中 | 优先 fp16 权重/激活 |
| **R10** | **Qwen3.5/3.6-35B-A3B 是混合架构（Gated-DeltaNet + full-attention）**，与 27B GDN 同源——GDN 的 FLA Triton 内核曾实测 VGPR=256（F24 §1.4），是 MoE 之外的额外风险面 | 中高 | 若选该模型，须同时确认 GDN 内核在 guard 下正确（参照 4B GDN 的验证方法） |
| **R11** | **KV head 数限制可整除的 TP 度**：如 35B-A3B 的 `n_kv_heads=2`，使"干净的 TP=4"不可行（需 KV 复制） | 中 | 选型时把 KV head 可整除性纳入约束，别只看参数量 |

---

## 6. 可行性判定：分档结论

| 档 | 组合 | 判定 | 理由 |
|---|---|---|---|
| **A｜立即可行、风险最低** | **非量化 fp16 MoE，≤16B，TP=1/2 on 卡 0,1**（如 OLMoE-1B-7B、DeepSeek-V2-Lite） | ✅ **推荐起步** | Triton 路径 + guard 覆盖；显存宽裕；绕开 DP4X |
| **B｜已实测可行** | **Qwen3.6-35B-A3B-GGUF，单卡 / Z100L TP=2** | ✅ **可上线（需补 Z100 验证）** | 唯一有实测证据的 MoE；但走 C++ 内核，**DP4X 卡上未验证（R1）** |
| **C｜需工程投入** | Qwen3-30B-A3B **fp16 TP=4** 或 **W8A16 TP=2** | 🔶 **值得投入** | 有 E=128/N=768 的 int8_w8a16 **深度调优**（14 档）配置；TP=4 撞 DP4X + 主机内存 |
| **D｜不建议** | MoE + GPTQ-Int4 / AWQ-4bit（AutoGPTQ）；GPT-OSS-20B（MXFP4 原生）；Qwen3-235B；Mixtral fp16 | ❌ **不建议** | 已实测失败 / 无 FP4 硬件 / 显存不可行 |

### 6.1 七个候选的可行性排序（最可行 → 最不可行）

| 排序 | 模型 | 一句话理由 |
|---|---|---|
| 1 | **OLMoE-1B-7B** | 唯一能**原生 fp16 单卡**跑起来的（12.9 GiB）——不需量化、不需 TP，**风险最低的起点** |
| 2 | **DeepSeek-V2-Lite** | fp16 勉强单卡（29.2 GiB），int8 舒适（14.6 GiB）；MLA 使 KV 便宜 → 卡 0/1 上 TP=1 |
| 3 | **Qwen3-30B-A3B** | int4 单卡（15.6 GiB），且是**唯一有 gfx906 深度调优 W8A16 内核**的模型（E=128/N=768，14 档） |
| 4 | **GPT-OSS-20B** | 显存上很宽松（int4 10.7 GiB）但**必须先把 MXFP4 重新量化成 W4A16/W8A16**——多一道工程步骤 |
| 5 | **Qwen3.5/3.6-35B-A3B** | int4 单卡（17.9 GiB）且唯一命中 `E=256,N=512` int4 配置；但 fp16 **TP=2 放不下**（32.6 > 32 GiB）、`n_kv_heads=2` 挡住干净 TP=4、且带 GDN 混合架构（R10） |
| 6 | **Mixtral 8x7B** | int4 单卡（23.9 GiB）但无调优配置，且**仅 8 专家**导致内核效率差；fp16 被迫 TP=4 用上故障卡 |
| 7 | **Qwen3-235B-A22B** | **不可行**：fp16 437.7 GiB；即使 int4 + TP=4 也是 30.1 GiB/卡且**零 KV 余量**；int4 实际需要 ~8 张卡 |

> **一个可用的"选择器杠杆"**：同样是 W4A16 MoE 权重，**换加载器会换到不同的内核与不同的 DP4X 风险**——AutoGPTQ 配置 → C++ `moe_wna16_gemm`（**不受 guard 保护**）；compressed-tensors WNA16 / Quark → 模块化 `TritonWNA16Experts`（**受 guard 保护**，见 §2.3.2）。若必须用 W4A16 MoE，**优先选 compressed-tensors/Quark 加载路径**，把风险从"裸奔"转为"guard 内"。

### 一句话回答"z100 适配 MoE 的可能性"

**可能性是"分层的可行性"而非"能/不能"**：MoE 的**架构层**（FusedMoE + 路由 + GGUF 专家加载）在本仓库已经打通并有一次实测成功（Qwen3.6-35B-A3B-GGUF）；真正的适配工作是**把 DP4X 故障面从 Triton 扩展到 MoE 所用的 C++ 内核**，以及**补 MoE 的显存/量化组合矩阵**。在卡 0/1 上做 Z100L TP=2 可以立刻获得一个无 DP4X 风险的 MoE 环境；在全部 4 卡上跑 30B 级 MoE 则同时受 DP4X 与主机内存双重约束。

---

## 7. 建议的下一步（最小验证矩阵）

> **前置约束（已核实，2026-10-19 15:2x）**：
> 1. **GPU 全占用**：4 张卡当前被**在线的生产服务**占满——`Qwen3.8-27B-FP8 --tensor-parallel-size 4`（PID 2622752，:8000，`--enforce-eager` + dflash spec-decode，VRAM 100%）。**任何 MoE 验证都需要先协商维护窗口**，或等该槽位让出卡；不存在"空闲卡可直接试"。
> 2. **本机没有任何 MoE 权重**：`/mnt/game/models/` 只有 `Qwen3-0.6B`、`Qwen3.5-4B`、`Qwen3.8-27B-{FP8,Channel-INT8-w8a8,DFlash2}`、`tinyllama-w4a16-group128`（最后一个为**稠密** W4A16，可用于单独验证 W4A16 线性层，但**不能**验证 MoE）。因此第 2/4/5 步都带**权重下载**前置（Qwen3.6-35B-A3B-GGUF ≈20 GB；OLMoE-1B-7B ≈14 GB）。磁盘尚可：`/mnt/game` 余 **328 GB**。
> 3. 第 1 步（hip shim VGPR 审计）**不需要 MoE 权重**，但**需要 GPU**，因此仍受约束 1 限制。

按成本从低到高，每一步都有明确判据：

1. **【0.5 天】C++ MoE 内核的 VGPR 审计（关闭 R1）——唯一可行方法**
   **不能用** `vgpr_audit.py`（那是纯静态 Triton `.hsaco` 解析，见 §2.3.1）。必须用 F24 §13.4 已交付的 **hip shim**（`LD_PRELOAD=/mnt/game/f24-investigation/hipshim/libhiplog.so`，`HIPLOG_VGPR_WARN=248`，内部走 `hipFuncGetAttributes()`）在**卡 2（Z100）**上跑 `Qwen3.6-35B-A3B-GGUF` 与一个 W8A16 MoE，记录 `ggml_moe_a8` / `ggml_moe_a8_vec` / `moe_wna16_gemm` 的 `numRegs`。
   **判据**：三者 `numRegs <= 203` → R1 关闭；任一 ≥248 → MoE 在 DP4X 卡上不可用，必须限定卡 0/1。

2. **【0.5 天】GGUF MoE 在 Z100 卡上的端到端正确性**
   `HIP_VISIBLE_DEVICES=2`（单张 Z100）跑 `Qwen3.6-35B-A3B-GGUF`，与 `HIP_VISIBLE_DEVICES=0`（Z100L）逐位对照。
   **判据**：token 流逐位一致 + 0 dmesg 故障。

3. **【1 天】最小 MoE 数值门禁（关闭 R4）**
   新建 `tests/kernels/test_gfx906_moe.py`：构造小 `RoutedExperts`（E=8/16, N=256, K=512），对 fp16 的 Triton `fused_experts` 与参考实现（`torch` 手写 expert loop）对照；再对 GGUF / WNA16 各一个形状做同样对照。
   **判据**：rel_err < 1e-2 且不依赖 AITER（在无 aiter 环境可跑）。

4. **【1–2 天】A 档可行性验证**
   下载 `OLMoE-1B-7B`（fp16, ~14 GiB），`--tensor-parallel-size 1` on 卡 0，eager decode。
   **判据**：ready 成功、文本合理、`vllm bench serve` 得到 decode/TTFT 基线数字替代 R7 的上游传闻。

5. **【2–3 天】C 档投资决策**
   `Qwen3-30B-A3B`：先试 **W8A16 TP=2 on 卡 0,1**（有调优配置），再试 fp16 TP=4（全卡，需 swap）。
   **判据**：与稠密 27B 的 decode 吞吐对比，判断 A3B 的激活参数优势是否被路由/内核开销吃掉。

6. **【文档】同步 README**
   把"MoE + GPTQ-Int4 / AWQ-4bit 不支持"从"已知无法运行"升级为带机制的说明（路由到 C++ `moe_wna16_gemm`，且该内核不受 DP4X guard 保护），并标注"Z100L 可用 / Z100 待验证"的 SKU 差异。

---

## 8. 证据路径

| 内容 | 路径 |
|---|---|
| DP4X 根因与 VGPR=256 判据 | `docs/result/F24-investigation-20261018.md` §1、§12 |
| DP4X guard 实现 | `vllm/platforms/hygon_dp4x.py` |
| 非量化 MoE 内核选型 | `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:283`、`fused_moe.py:1312-1432` |
| WNA16 MoE 路由 | `vllm/model_executor/layers/quantization/moe_wna16.py:478-540`、`fused_moe.py:1304-1309` |
| WNA16 MoE C++ 内核 | `csrc/moe/moe_wna16.cu` |
| GGUF MoE 路径 | `vllm/model_executor/layers/quantization/gguf.py:1690`、`csrc/quantization/gguf/gguf_kernel.cu:2343` |
| GFX906 调优配置 | `vllm/model_executor/layers/fused_moe/configs/*AMD_GFX906*.json`（7 个） |
| ISA 能力审计 | `docs/archive/result/z100-11d-z100-isa-capability-audit-result.md` |
| qwen35moe GGUF 支持 | commits `f811df4e7`、`8cfc349ac`、`d456527b6`、`911e2025d`、`182cb318a` |
| MoE 失败声明 | `README.md` "已知无法运行" / "已知限制" 第 4 条 |
| Z100 执行记录（MoE 未覆盖） | `docs/archive/z100-todolist.md`（S10 = ⬜） |
| VGPR 审计工具 | F24 §13.4 hip shim（`hipFuncGetAttributes` + `HIPLOG_VGPR_WARN`） |
