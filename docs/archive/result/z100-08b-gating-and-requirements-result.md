# S8b gfx906 门控补漏 + 依赖/缓存约定 — 执行结果

> 归档时间：2026-10-08（容器 CST）
> 执行者：子 agent（S8b 门控补漏）
> 对应设计文档：[docs/design/z100-08-attention-and-quant-paths.md](../design/z100-08-attention-and-quant-paths.md)
> 前一档：[z100-08-attention-and-quant-paths-result.md](z100-08-attention-and-quant-paths-result.md)（S8：路径与后端核对 + 2 个 fork 真 bug 最小修复）
> 状态：✅ **三项任务全部完成**；**FP8 block-scaled 判定为「数值已验证可用 ⇒ 不门控」**（与任务书 §3 门禁描述中"block-scaled 被拒"的措辞存在前提冲突，见 §5.2，已附一键门控补丁）；
> ⏸️ W4A16 端到端与 27B 多卡端到端仍移交 S9（本步给出可直接粘贴的命令 + 权重/卡数需求，见 §6.2）。

**一句话结论**：在 gfx906（卡 3、自编 triton 3.6.0+git7c56a5e4）上对 block-scaled FP8 做了
**17 组用例的数值 sweep**（14 种形状 × 3 种输出 dtype fp16/bf16/fp32，另加极小/极大动态范围
与全零激活），`_w8a8_triton_block_scaled_mm` 相对
fp32 反量化参考的 `mean_rel_err` 为 **0.000e+00 ~ 1.465e-07**（全 finite，无一超 1e-2），
输入量化核 `per_token_group_quant_fp8` 与 native 参考差 ≤ **1.144e-05**，整条
`TritonFp8BlockScaledMMKernel.apply_weights()`（选择 + `process_weights_after_loading` + forward）
相对原始权重 fp32 matmul 的 `mean_rel_err` 为 **3.311e-07** ⇒ **数值干净**。
但机器码证明它是**纯软件模拟**（`v_fmac_f32_e32` **8160** 条，`v_mfma`/`v_dot2_f32_f16`/
`v_dot4_i32`/`v_pk_fma_f16`/`v_cvt_f32_fp8` **全部为 0**），代价是 4096³ 上
**304 ms vs fp16 8.4 ms（≈36×）** ⇒ **按任务书 §1 的判据不门控**，但列入"性能上禁用"。
本步在 `vllm/**` **零改动**；改了 1 个测试 + 3 个 `tools/dcu`/`requirements` 文件，新增 2 个
`tools/dcu` 脚本；`tools/dcu/s8b_gate.sh` **全绿（14/14 + probe rc=0 + 测试 3 passed）**。

---

## 1. 目标与验收标准

### 1.1 任务书三项（按优先级）

1. **FP8 block-scaled 的 arch 门控**（最重要）：先在 gfx906 上用自编 triton 做数值探针；
   若数值不可靠/未经证实 ⇒ 加显式门控（返回 `False` + 明确原因字符串）；若数值干净 ⇒ 不门控，
   写证据 + 更新禁用清单（说明"已验证可用"）；并顺带核对其它 FP8 路径，确保没有第二条静默误选。
2. **过期测试期望**：`tests/kernels/quantization/test_w4a16_kernel_selection.py:32-40` 在 gfx906 上
   结构性不可过 ⇒ 修成正确期望（含 gfx906 分支），给出前后对比。
3. **依赖与缓存约定落地**：`compressed-tensors==0.15.0.1` 进 `requirements/dcu*.txt` 的确认/补齐 +
   `tools/dcu/README.md` 写清"镜像自带 0.14.0.1 不够"；`tools/dcu/env.sh` 加 `VLLM_CACHE_ROOT`
   默认导出；新增 `tools/dcu/s8b_gate.sh` 把禁用清单固化成可执行检查。
4. （若时间允许）把 S8 §7 未执行项整理成"可直接粘贴命令 + 权重/卡数"清单供 S9 使用。

### 1.2 本步验收标准

- 每一项都有**真实运行期证据**（命令 + 原始输出），长日志只进本档；
- 固定 `HIP_VISIBLE_DEVICES=3`，`VLLM_CACHE_ROOT` 落 `/mnt/game`；
- **禁止执行任何自编译 HIP 独立可执行程序**；不 start/kill/exec 生产容器 `qwen38-vllm`；
  不动 `z100-build`（2.10 基线）；不替换 DTK 库；出现挂死立即停止并如实记录；
- 不改 `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`；不改 S8 已改过的
  `moe_wna16.py` 与 mixed_precision 那 6 个文件；
- 写作用域：`vllm/model_executor/kernels/linear/scaled_mm/**`、
  `tests/kernels/quantization/test_w4a16_kernel_selection.py`、`tools/dcu/**`、
  `requirements/dcu*.txt`、本归档。

### 1.3 硬约束遵守情况（逐条）

| 硬约束 | 遵守 | 证据 |
|---|---|---|
| 1 禁自编译 HIP 独立可执行程序 | ✅ | 全程只用 `python`（vLLM + triton JIT）/ `pytest` / `curl` / `docker exec`，**无任何自编译可执行程序**；未执行 `hipcc` 产物 |
| 2 不 start/kill/exec 生产容器 | ✅ | 只 `docker ps -a` 只读观察；`qwen38-vllm` 保持 `Exited (137)`（§3.1） |
| 3 产物/缓存落 `/mnt/game` | ✅ | 全部日志 `/mnt/game/port/s8b/`；`VLLM_CACHE_ROOT=/mnt/game/vllm-cache`、`TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache` |
| 4 不替换 DTK 库 | ✅ | 无任何 `/opt/dtk` 写入 |
| 5 不把 gfx906 加进 `_ON_GFX9` | ✅ | 未改 `vllm/platforms/rocm.py`；门禁实测 `on_gfx906=True on_gfx9=False` |
| 13 gfx906 禁 `--enforce-eager` | ✅ | 本步无任何 `serve` 命令 |
| 16 `compressed-tensors==0.15.0.1` | ✅ | §2.4 + 门禁第 14 项 |
| 17 显式设 `VLLM_CACHE_ROOT` | ✅ | `tools/dcu/env.sh:108-113` 新增默认导出 + 门禁第 13 项 |
| 18 显式 `--dtype float16` / FP8 block-scaled 处置 | ✅ | 本步不涉及 `serve`；FP8 结论见 §3.2 |
| 无挂死 | ✅ | 全程无 GPU 驱动死锁、无 loadavg 飙升、无 `kill -9`（§3.5） |
| 未改禁改文档 | ✅ | `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md` 零改动 |

---

## 2. 实际改动

### 2.1 变更总表

| # | 文件 | 位置 | 改动摘要 |
|---|---|---|---|
| A | [`tests/kernels/quantization/test_w4a16_kernel_selection.py`](../../tests/kernels/quantization/test_w4a16_kernel_selection.py) | `:19-22`、`:27-34`、`:36-49`、`:86-105` | 期望值补 gfx906 分支；新增 `has_g_idx=True ⇒ ValueError` 用例（锁"明确报错而非静默算错"） |
| B | [`tools/dcu/env.sh`](../../tools/dcu/env.sh) | `:12`（注释）、`:108-113`（新第 8 节）、`:119`（banner） | 新增 `VLLM_CACHE_ROOT` 默认导出（`/mnt/game/vllm-cache`，`/mnt/game` 不存在或已被外部设置时不覆盖） |
| C | [`requirements/dcu.txt`](../../requirements/dcu.txt) | `:50-56` | 从 `common.txt` **重述**（不换版本号）`compressed-tensors == 0.15.0.1`，附"镜像自带 0.14.0.1 不够"的原因注释 |
| D | [`tools/dcu/README.md`](../../tools/dcu/README.md) | `env.sh` 第 6 条 + 新增 `## compressed-tensors must be >= 0.15.0.1`、`## s8b_gate.sh` 两节 | 文档同步 |
| E | [`tools/dcu/s8b_fp8_block_probe.py`](../../tools/dcu/s8b_fp8_block_probe.py) | 新增 511 行 | FP8 选择面 + 17 组用例数值 sweep + 输入量化核 + 全 kernel 路径 + 计时；**带退出码**（全通过 0） |
| F | [`tools/dcu/s8b_gate.sh`](../../tools/dcu/s8b_gate.sh) | 新增 256 行 | 14 项禁用清单/约定检查 + FP8 数值探针 + W4A16 选择测试；`ALL GREEN` 才 exit 0 |
| G | [`docs/result/z100-08b-gating-and-requirements-result.md`](z100-08b-gating-and-requirements-result.md) | 本档 | — |

### 2.2 **`vllm/model_executor/kernels/linear/scaled_mm/**` 零改动（这是本步最重要的"改动"）**

任务书 §1 的分支判据是"数值是否干净"。本步实测数值干净（§3.2），因此**按任务书要求不加门控**：
`TritonFp8BlockScaledMMKernel.is_supported()` 在 gfx906 上继续返回 `(True, None)`，
`init_fp8_linear_kernel()` 继续选中它。理由与代价在 §3.2.5 / §5.2 展开，**一键门控补丁**在 §5.2。

### 2.3 容器（复用 `z100-s8`，未新建、未改动）

```console
$ docker ps -a --format '{{.Names}}\t{{.Status}}'
z100-s8         Up About an hour          # 本步复用（S8 建，未 --rm）
z100-eager      Up About an hour
z100-serve      Up 2 hours
z100-triton     Up 15 hours
z100-build-211  Up 15 hours               # 只读参照，未动
z100-build      Up 21 hours               # 2.10 基线，未动
qwen38-vllm     Exited (137) 15 hours ago # 生产容器，全程只 ps 只读观察
```

设备：`HIP_VISIBLE_DEVICES=3`（`S8_HIP_VISIBLE_DEVICES=3` 交给 `s8_env.sh`）。
`VLLM_DIR=/mnt/game/port/gfx906-vllm-211`，`TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache`，
`VLLM_CACHE_ROOT=/mnt/game/vllm-cache`（env.sh 新默认值生效，见 §3.4.2）。

### 2.4 `/mnt/game` 下的本步产物

```
/mnt/game/port/s8b/                            本步全部日志（见 §8）
/mnt/game/port/s8b/s8b-fp8-block-probe.log     FP8 数值 sweep（最终版，rc=0）
/mnt/game/port/s8b/s8b-pytest-w4a16-sel-BEFORE.log  原始测试（2 failed）
/mnt/game/port/s8b/s8b-pytest-w4a16-sel-AFTER.log   修复后测试（3 passed）
/mnt/game/port/s8b/s8b-gate.log                s8b_gate.sh 全量输出（ALL GREEN）
/mnt/game/port/s8b/s8b-gate-fp8-probe.log     门禁内跑的 FP8 探针
/mnt/game/port/s8b/s8b-gate-w4a16-sel.log     门禁内跑的 W4A16 选择测试
/mnt/game/port/s8b/test_w4a16_kernel_selection_BEFORE.py  git HEAD 版测试（before 对照）
/mnt/game/models/tinyllama-w4a16-group128/model.safetensors.INCOMPLETE-433881160
                                               W4A16 半成品（433,881,160 B / 正确 761,968,600 B；见 §5.1）
/mnt/game/port/s8b/s8b-w4a16-download.log      W4A16 权重续传尝试（curl 日志，见 §5.1）
/mnt/game/port/gfx906-vllm-211/tools/dcu/      s8b_gate.sh / s8b_fp8_block_probe.py（以及同步的 env.sh/README.md）
/mnt/game/port/gfx906-vllm-211/tests/kernels/quantization/test_w4a16_kernel_selection.py  修复后测试副本
/mnt/game/port/gfx906-vllm-211/requirements/dcu.txt                                     重述 pin 后的副本
/mnt/game/vllm-cache/                         env.sh 新默认缓存根（本步创建）
```

---

## 3. 执行过程与原始证据

### 3.0 环境自检（卡 3、目标栈、自编 triton）

```console
$ docker exec z100-s8 bash -lc '
    source /opt/dtk/env.sh >/dev/null 2>&1
    export S8_HIP_VISIBLE_DEVICES=3 VLLM_CACHE_ROOT=/mnt/game/port/s8b/vllm-cache
    source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh
    cd $VLLM_DIR; python - <<PY
import torch, importlib.metadata as md
print("torch", torch.__version__, torch.version.hip)
print("triton", md.version("triton"), __import__("triton").__file__)
print("dev", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)
PY'
[dcu] VLLM_DCU_GCN_ARCH=gfx906
[dcu-s7] VLLM_DIR=/mnt/game/port/gfx906-vllm-211
[dcu-s7] TRITON_INSTALL=/mnt/game/triton-gfx906/install TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
[dcu-s8] HIP_VISIBLE_DEVICES=3
torch 2.11.0 6.3.26113
triton 3.6.0+git7c56a5e4 /mnt/game/triton-gfx906/install/triton/__init__.py
dev Z100 gfx906:sramecc+:xnack-
```

`FP8_DTYPE` 在本平台解析为 `torch.float8_e4m3fn`（不是 fnuz），
`torch.tensor(...).to(torch.float8_e4m3fn)` 与 fp8→fp32 反量化在 gfx906 上可用：

```console
float8_e4m3fn torch.float8_e4m3fn
float8_e5m2    torch.float8_e5m2
float8_e8m0fnu torch.float8_e8m0fnu
cast fp8 ok torch.float8_e4m3fn 2.5
fp8->float32 matmul ok torch.Size([4, 4])
```

### 3.1 生产容器/宿主机只读观察

```console
$ docker ps -a --format '{{.Names}}\t{{.Status}}'
z100-eager      Up 48 minutes
z100-s8         Up 49 minutes
z100-serve      Up 2 hours
z100-triton     Up 14 hours
z100-build-211  Up 15 hours
z100-build      Up 21 hours
qwen38-vllm     Exited (137) 14 hours ago
awesome_jemison Exited (137) 3 days ago
$ python -c "import torch; print([ (i, *[round(x/2**30,2) for x in torch.cuda.mem_get_info(i)]) for i in range(4)])"
[(0, 31.73, 31.98), (1, 31.73, 31.98), (2, 31.73, 31.98), (3, 31.73, 31.98)]
```

⇒ 4 卡全空，本步只用卡 3；生产容器只 `docker ps` 观察，**未 start/kill/exec/commit**。

### 3.2 FP8 block-scaled：数值探针（本步核心）

脚本 [`tools/dcu/s8b_fp8_block_probe.py`](../../tools/dcu/s8b_fp8_block_probe.py)，日志
`/mnt/game/port/s8b/s8b-fp8-block-probe.log`。它不做 monkey-patch，测的是引擎真正会选的路径。
参考实现是上游自带的 fp32 反量化 matmul
（`tests.kernels.quant_utils.native_w8a8_block_matmul`），与 triton 内核相互独立。

#### 3.2.1 选择面：只有 block-scaled 被放行

```console
=== 1. FP8 kernel selection surface on this platform ===
platform = PlatformEnum.ROCM  fp8_dtype = torch.float8_e4m3fn

_POSSIBLE_FP8_KERNELS[ROCM] (per-tensor/per-channel) = ['AiterHipbMMPerTokenFp8ScaledMMLinearKernel',
  'AiterPreshuffledPerTokenFp8ScaledMMLinearKernel', 'AiterPerTokenFp8ScaledMMLinearKernel',
  'ROCmFP8ScaledMMLinearKernel', 'PerTensorTorchFP8ScaledMMLinearKernel',
  'RowWiseTorchFP8ScaledMMLinearKernel', 'ChannelWiseTorchFP8ScaledMMLinearKernel']
  is_supported(AiterHipbMM...)     -> (False, 'requires setting `VLLM_ROCM_USE_AITER=1`, `..._LINEAR=1`, and `..._LINEAR_HIPBMM=1`.')
  is_supported(AiterPreshuffled...) -> (False, 'requires setting `VLLM_ROCM_USE_AITER=1` and `..._LINEAR=1`. ...')
  is_supported(AiterPerToken...)   -> (False, 'requires setting `VLLM_ROCM_USE_AITER=1` and `..._LINEAR=1`. ...')
  is_supported(ROCmFP8ScaledMMLinearKernel) -> (False, 'wvSplitKQ (skinny GEMM) is not built on gfx906 (no MFMA)')
  is_supported(PerTensorTorchFP8ScaledMMLinearKernel)  -> (False, 'requires a platform with torch FP8 scaled-MM support.')
  is_supported(RowWiseTorchFP8ScaledMMLinearKernel)    -> (False, 'requires platform with torch FP8 scaled-MM support.')
  is_supported(ChannelWiseTorchFP8ScaledMMLinearKernel)-> (False, 'requires platform with torch FP8 scaled-MM support.')

_POSSIBLE_FP8_BLOCK_KERNELS[ROCM] (block-scaled) = ['AiterFp8BlockScaledMMKernel', 'TritonFp8BlockScaledMMKernel']
  is_supported(AiterFp8BlockScaledMMKernel) -> (False, 'Only supported on ROCm platform with aiter package installed.')
  is_supported(TritonFp8BlockScaledMMKernel) -> (True, None)

_POSSIBLE_WFP8A16_KERNELS[ROCM] = <empty>
```

真实选择器（`set_current_vllm_config(VllmConfig())` 之下，与测试同口径）：

```console
=== 2. what init_fp8_linear_kernel() actually returns ===
init_fp8_linear_kernel[block-scaled W8A8 (w=[128,128] blocks)] -> TritonFp8BlockScaledMMKernel
init_fp8_linear_kernel[per-tensor W8A8 (dynamic act)]  -> RAISED ValueError: Failed to find a kernel that can implement the ScaledMM linear layer. Reasons:
     AiterHipbMM... / AiterPreshuffled... / AiterPerToken... / ROCmFP8ScaledMMLinearKernel wvSplitKQ (skinny GEMM) is not built on gfx906 (no MFMA). /
     PerTensorTorch... / RowWiseTorch... / ChannelWiseTorch... requires a platform with torch FP8 scaled-MM support..
init_fp8_linear_kernel[per-tensor W8A8 (static act)]   -> RAISED ValueError: ...（同上 7 条原因）
init_fp8_linear_kernel[per-channel W8A8]               -> RAISED ValueError: ...（同上 7 条原因）
  selection expectations: OK (only block-scaled is accepted)
```

⇒ **"第二条静默误选"不存在**：除 block-scaled 外，三条 per-tensor/per-channel 路径全部**抛
`ValueError` 并列出逐内核原因**；`_POSSIBLE_WFP8A16_KERNELS[ROCM]` 为空表。补充核对（静态）：

- `BlockWiseTorchFP8ScaledMMLinearKernel`、`MarlinFP8ScaledMMLinearKernel`、
  `HummingFP8ScaledMMLinearKernel`、`CutlassFp8BlockScaledMMKernel`、`DeepGemmFp8BlockScaledMMKernel`、
  `B12xFp8BlockScaledMMKernel`、`FlashInferFp8BlockScaledMMKernel`、`XPUFp8BlockScaledMMKernel`
  **都不在 `PlatformEnum.ROCM` 列表里**（`vllm/model_executor/kernels/linear/__init__.py:411-501`）。
- `vllm/models/deepseek_v4{,_v41}/amd/rocm.py:113/69` 的 `apply_pre_quantized_block_scaled_mm()`
  **不自己选内核**，它复用 `linear.quant_method.fp8_linear`（即上面的选择结果）⇒ 不是第二条选择路径。
- `_POSSIBLE_MXFP8_KERNELS[ROCM]`（`__init__.py:533-553`）首个 `RocmDotScaledMxfp8LinearKernel`
  在注释里就写明"gates to gfx95x and falls through to BF16 emulation elsewhere" ⇒ MXFP8 在 gfx906
  走 **BF16 模拟**（`EmulationMxfp8LinearKernel`），与 FP8 e4m3 block 是两套 scheme，无静默 FP8 选择。
- `scaled_mm/rocm.py:19-45` 的 `rocm_per_tensor_float_w8a8_scaled_mm_impl`
  （S8 §5.4-2 的残余风险）：唯一调用点是 `ROCmFP8ScaledMMLinearKernel.apply_weights`（`rocm.py:131`），
  而该 kernel 的 `is_supported()` 在 gfx906 返回 `False` ⇒ 正常选择不可达。若有人绕过选择直接调
  `torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl` 且 `M<=4`，会走
  `vllm/_custom_ops.py:3420` 的 `torch.ops._rocm_C.wvSplitKQ`——该 op **未注册**在 gfx906 的
  `_rocm_C`（门禁第 4 项实测只剩 `paged_attention`）⇒ **抛 `AttributeError` 而不是静默算错**。
  属"响亮失败"，本步**未改代码**，只记录（写作用域允许，但无 bug 可修）。

#### 3.2.2 数值 sweep：17 组用例（形状 / 输出 dtype / 动态范围），全部通过

`tools/dcu/s8b_fp8_block_probe.py` §3；每个 `[...]` 块的第二行是"反量化表示本身相对理想
fp32 的误差"（即 FP8 量化误差地板，与内核实现无关）。**最后 5 行做了省略**（略去重复的
`M=… N=… K=… blk=[…]` 与 `qx/qw are already fp8; kernel vs fp32-dequant: finite=True` 前缀），
数值与判定行均为日志原文。

```console
=== 3. numeric sweep: triton block-scaled FP8 GEMM vs fp32 reference ===
  (gfx906 has no FP8 unit; this exercises the emulated triton path)
  [decode M=1            ] M=   1 N= 512 K= 512 blk=[128,128] out=float16
       kernel vs fp32-dequant: finite=True mean_rel_err=0.000e+00 max_abs_err=0.000e+00
       dequant-vs-ideal-fp32 (quantization error floor): mean_rel_err=3.725e-02
  [M=7, blk_n=64         ] M=   7 N= 128 K= 256 blk=[64,128]  out=float16
       kernel vs fp32-dequant: finite=True mean_rel_err=4.101e-11 max_abs_err=1.907e-06
  [small square          ] M=  16 N= 256 K= 256 blk=[128,128] out=float16
       kernel vs fp32-dequant: finite=True mean_rel_err=8.187e-11 max_abs_err=1.526e-05
  [square                ] M=  64 N= 512 K= 512 blk=[128,128] out=float16
       kernel vs fp32-dequant: finite=True mean_rel_err=6.031e-08 max_abs_err=6.250e-02
  [N=576 (tail mask)     ] M= 128 N= 576 K= 512 blk=[128,128] out=float16
       kernel vs fp32-dequant: finite=True mean_rel_err=4.063e-08 max_abs_err=6.250e-02
  [rect N<K              ] M= 128 N= 256 K= 512 blk=[128,128] out=float16
       kernel vs fp32-dequant: finite=True mean_rel_err=7.472e-09 max_abs_err=1.562e-02
  [rect N>K              ] M= 256 N=1024 K= 256 blk=[128,128] out=float16
       kernel vs fp32-dequant: finite=True mean_rel_err=4.701e-08 max_abs_err=1.250e-01
  [K=1024                ] M= 512 N= 512 K=1024 blk=[128,128] out=float16
       kernel vs fp32-dequant: finite=True mean_rel_err=7.126e-08 max_abs_err=1.250e-01
  [prefill M=2048        ] M=2048 N= 512 K= 512 blk=[128,128] out=float16
       kernel vs fp32-dequant: finite=True mean_rel_err=4.735e-08 max_abs_err=1.250e-01
  [long K=7168           ] M=  64 N= 512 K=7168 blk=[128,128] out=float16
       kernel vs fp32-dequant: finite=True mean_rel_err=1.465e-07 max_abs_err=5.000e-01
  [wide N=18432, M=1     ] M=   1 N=18432 K= 512 blk=[128,128] out=float16
       kernel vs fp32-dequant: finite=True mean_rel_err=2.674e-08 max_abs_err=3.125e-02
  [FFN-like              ] M=  64 N=7168 K=2048 blk=[128,128] out=float16
       kernel vs fp32-dequant: finite=True mean_rel_err=6.061e-08 max_abs_err=2.500e-01
  [out=bf16              ] ... out=bfloat16  mean_rel_err=6.201e-10 max_abs_err=9.766e-04
  [out=fp32              ] ... out=float32   mean_rel_err=5.209e-08 max_abs_err=4.578e-05
  [tiny dynamic range    ] (act/weight ×1e-3) mean_rel_err=0.000e+00 max_abs_err=0.000e+00
  [wide dynamic range    ] (act/weight ×1e2, out=fp32) mean_rel_err=5.346e-08
  [all-zero activation   ] mean_rel_err=0.000e+00 max_abs_err=0.000e+00

  shapes with rel_err > 1e-2 or non-finite output: NONE
```

#### 3.2.3 输入量化核与整条 kernel 路径

```console
=== 4. input quantization kernel (per_token_group_quant_fp8) ===
  QuantFP8.forward_hip -> forward_cuda -> this triton kernel on ROCm
  M=   7 K= 256 gs= 128 quant_err_vs_fp32=2.204e-02 triton_vs_native_quant=4.851e-08
  M=  64 K= 512 gs= 128 quant_err_vs_fp32=2.191e-02 triton_vs_native_quant=1.144e-05
  M=   1 K=1024 gs= 128 quant_err_vs_fp32=2.167e-02 triton_vs_native_quant=4.575e-08
  M= 128 K= 384 gs=  64 quant_err_vs_fp32=2.149e-02 triton_vs_native_quant=5.728e-06
  quant shapes disagreeing with native by > 1e-3: NONE

=== 5. full Fp8BlockScaledMMLinearKernel.apply_weights() path ===
  selected kernel = TritonFp8BlockScaledMMKernel
  before process: weight=(512, 512) scale=(4, 4)
  after  process: weight=(512, 512) scale=(4, 4)
  process_weights_after_loading dequant max|delta| = 0.000e+00 (0 expected unless is_fp8_fnuz())
  is_fp8_fnuz() = False
  forward OK: shape=(64, 512) dtype=torch.float16
  output finite = True
  apply_weights vs fp32-dequant: finite=True mean_rel_err=9.166e-08 max_abs_err=1.250e-01
  apply_weights vs ORIGINAL-weight fp32 matmul: finite=True mean_rel_err=3.311e-07 max_abs_err=1.250e-01
```

⇒ 从"选择内核 → `process_weights_after_loading`（在 gfx906 上是纯 no-op，`is_fp8_fnuz()=False`）
→ `apply_weights` 的输入量化 + 分块 GEMM"整条链路相对**原始权重**的 fp32 matmul 误差 **3.311e-07**。

#### 3.2.4 它是软件模拟：机器码证据

从 triton 磁盘缓存取出 4096³ 那次编译的产物
（`/mnt/game/triton-gfx906/cache/4T6E7OZ5.../_w8a8_triton_block_scaled_mm.amdgcn`，27494 行）：

```console
--- ttgir dot ---
tt.dot %a_144
tt.dot %a_78
--- asm histogram (top) ---
   8160  v_fmac_f32_e32
   3629  s_waitcnt
   3560  buffer_load_dword
   2274  buffer_store_dword
    772  ds_read_b128
    580  s_nop
--- key ops ---
v_fma_f32            96
v_pk_fma_f16         0
v_dot2_f32_f16       0
v_dot4_i32           0
v_mfma               0
v_cvt_f32_fp8        0
v_cvt_pk_f32_fp8     0
```

⇒ gfx906 上 `tl.dot(fp8, fp8)` 被降级成 **8160 条标量 `v_fmac_f32_e32`**（fp32 FMA），
`v_mfma` / `v_dot2_f32_f16` / `v_dot4_i32` / `v_pk_fma_f16` / `v_cvt_f32_fp8` **全为 0**
（gfx906 连 fp8 转换指令都没有，Vega20 之前无 F8F6F4 支持）。
这就是数值"干净"的技术原因（fp32 FMA 累加，只有最后一次 round 到 fp16），也是性能代价的原因。

#### 3.2.5 代价：4096³ 上 ≈36× 慢于 fp16

```console
=== 6. order-of-magnitude timing: emulated fp8 vs fp16 vs int8 ===
  (not a benchmark; only enough to see whether fp8 is hardware-fast)
  4096^3  fp16 torch matmul  :      8.4 ms
  4096^3  fp32 torch matmul  :     15.0 ms
  4096^3  triton int8 (prod) :      7.6 ms
  4096^3  triton block fp8   :    304.2 ms        # 另三次独立运行：307.9 / 306.1 / 303.1 ms

=== S8b FP8 block-scaled verdict ===
  selection    PASS
  numeric      PASS
  input_quant  PASS
  full_kernel  PASS
FP8_BLOCK_PROBE_DONE rc=0
```

**判定**：按任务书 §1 的判据（"数值干净 ⇒ 不要门控，说明已验证可用"）——
**不加 arch 门控**；同时在禁用清单里把它标为"数值已验证可用、但性能上禁用（≈36× fp16）"。
任务书 §3 门禁项里"FP8 block-scaled 被拒"的措辞以"门控分支"为前提，与本步实测冲突，
处置与一键回滚方案见 §5.2。

### 3.3 过期测试期望：修复前后对比

- **BEFORE**：`git show HEAD:tests/kernels/quantization/test_w4a16_kernel_selection.py`
  原样落到 `/mnt/game/port/s8b/test_w4a16_kernel_selection_BEFORE.py`（70 行，md5 `6a0e692e…`）后运行。
- **AFTER**：修复后的文件同步进 canonical 副本的 `tests/kernels/quantization/` 后运行。

命令（两者相同，只换目标文件）：

```console
$ cd $VLLM_DIR && python -c "import torch, pytest, sys; sys.exit(pytest.main(
    ['-v','--no-header','--noconftest','-p','no:cacheprovider', <TESTFILE>]))"
```

```console
# BEFORE（/mnt/game/port/s8b/s8b-pytest-w4a16-sel-BEFORE.log）
>       assert kernel_type.__name__ == _expected_rocm_kernel(scalar_types.uint4, 64)
E       AssertionError: assert 'Gfx906GPTQWNA16LinearKernel' == 'TritonW4A16LinearKernel'
E         - TritonW4A16LinearKernel
E         + Gfx906GPTQWNA16LinearKernel
FAILED ...::test_choose_mp_linear_kernel_uint4b8
FAILED ...::test_choose_mp_linear_kernel_uint4_asymmetric
======================== 2 failed, 17 warnings in 6.48s ========================
BEFORE_RC=1

# AFTER（/mnt/game/port/s8b/s8b-pytest-w4a16-sel-AFTER.log）
tests/kernels/quantization/test_w4a16_kernel_selection.py::test_choose_mp_linear_kernel_uint4b8 PASSED [ 33%]
tests/kernels/quantization/test_w4a16_kernel_selection.py::test_choose_mp_linear_kernel_uint4_asymmetric PASSED [ 66%]
tests/kernels/quantization/test_w4a16_kernel_selection.py::test_choose_mp_linear_kernel_gidx_is_rejected_on_gfx906 PASSED [100%]
======================== 3 passed, 17 warnings in 6.61s ========================
AFTER_RC=0
```

**修复内容**（`file:line` 见 §2.1-A），论证：

1. `_expected_rocm_kernel()` 原本只镜像 RDNA3/Hybrid/Triton 三段优先级，漏了
   `_POSSIBLE_KERNELS[ROCM]` 的**第一个**成员 `Gfx906GPTQWNA16LinearKernel`
   （`vllm/model_executor/kernels/linear/__init__.py:503-522`）。补上 `on_gfx906()` 分支后，
   期望值与选择器一致；`_GFX906_QUANT_TYPES` 抄自
   `gfx906_gptq_wna16.py:20-25` 的 `SUPPORTED_QUANT_TYPES`，不是猜的。
2. 新增的第 3 个用例锁住 S8 §3.6.2 的"期望行为"：`has_g_idx=True`（`desc_act`/`actorder=group`）
   在 gfx906 上**必须**抛 `ValueError` 且原因串含 `Activation reordering is not supported`
   （`gfx906_gptq_wna16.py:37-38`）——这防止将来有人把门控删掉后**静默忽略 g_idx 算错**。
3. 非 ROCm 分支显式把 `on_gfx906` 也设成 `lambda: False`（`:22`），保证未装 ROCm 的机器行为不变
   （硬约束 6 的口径）。

### 3.4 依赖与缓存约定落地证据

#### 3.4.1 `compressed-tensors == 0.15.0.1`（`requirements/dcu*.txt`）

先确认链路：`requirements/dcu.txt:22` 与 `requirements/dcu-build.txt:12` 都 `-r common.txt`，
而 `requirements/common.txt:40` 已经是 `compressed-tensors == 0.15.0.1`
（与 `requirements/common.txt` 的**语义完全一致，只是重述**，未改上游）：

```console
$ grep -n "^-r common.txt" requirements/dcu.txt requirements/dcu-build.txt
requirements/dcu.txt:22:-r common.txt
requirements/dcu-build.txt:12:-r common.txt
$ grep -n "compressed-tensors" requirements/common.txt requirements/dcu.txt
requirements/common.txt:40:compressed-tensors == 0.15.0.1 # required for compressed-tensors; ...
requirements/dcu.txt:56:compressed-tensors == 0.15.0.1
$ python -m pip install --dry-run --no-deps "compressed-tensors==0.15.0.1"
Requirement already satisfied: compressed-tensors==0.15.0.1 in /usr/local/lib/python3.10/dist-packages (0.15.0.1)
```

⇒ 结论：**没有版本缺口**（`-r common.txt` 已带上），本步把 pin **重述**进 `requirements/dcu.txt:50-56`
并写明原因（镜像自带 0.14.0.1、`pack_quantized` 缺失），使"镜像自带的旧版本不够"这件事
在 DCU 依赖文件里**自解释**，不再依赖读者去翻 `common.txt`。`requirements/common.txt` **未改**。

#### 3.4.2 `VLLM_CACHE_ROOT` 默认导出（`tools/dcu/env.sh:108-119`）

```console
$ ( source tools/dcu/env.sh ) 2>&1 | grep VLLM_CACHE_ROOT
[dcu] VLLM_CACHE_ROOT=/mnt/game/vllm-cache                    # 默认值
$ ( export VLLM_CACHE_ROOT=/mnt/game/port/s8b/override-cache; source tools/dcu/env.sh ) 2>&1 | grep VLLM_CACHE_ROOT
[dcu] VLLM_CACHE_ROOT=/mnt/game/port/s8b/override-cache        # 外部覆盖生效
```

实现要点（`env.sh:112-113`）：只在 `VLLM_CACHE_ROOT` 为空**且** `/mnt/game` 存在时导出
`/mnt/game/vllm-cache` ⇒ ①可被外部覆盖；②没挂 `/mnt/game` 的机器不会被塞一个不存在的路径。
副作用（如实记录）：`tools/dcu/s8_env.sh` 用的是 `${VLLM_CACHE_ROOT:-/mnt/game/port/s8/vllm-cache}`，
因此它现在**继承** `env.sh` 的新默认值 `/mnt/game/vllm-cache`，S8 归档里写的
`/mnt/game/port/s8/vllm-cache` 成为旧值——两者都在 `/mnt/game` 下，硬约束 17 仍满足。

#### 3.4.3 `tools/dcu/s8b_gate.sh`（把约定固化成可执行检查）

```console
$ docker exec z100-s8 bash /mnt/game/port/gfx906-vllm-211/tools/dcu/s8b_gate.sh; echo "GATE_RC=$?"
[dcu] VLLM_CACHE_ROOT=/mnt/game/vllm-cache
[dcu-s8] HIP_VISIBLE_DEVICES=3
===== S8b gate 2026-10-08T22:12:15+08:00 =====
----- 1) kill-list / convention checks -----
CHECK on_gfx906/on_gfx9/cdna_version     PASS on_gfx906=True on_gfx9=False cdna_version=0
CHECK ROCM_SKINNY_GEMM_ENABLED           PASS ROCM_SKINNY_GEMM_ENABLED=False env=False
CHECK VLLM_ROCM_USE_SKINNY_GEMM=0        PASS env read = False
CHECK _rocm_C ops == [paged_attention]   PASS ['_rocm_C::paged_attention']
CHECK use_rocm_custom_paged_attention=False PASS gfx906 early return
CHECK is_aiter_found_and_supported=False PASS switches={'is_enabled': None, 'is_linear_enabled': None, 'is_mha_enabled': None, 'is_mla_enabled': None, 'is_fused_moe_enabled': None, 'is_triton_unified_attn_enabled': None}
CHECK flash-attn unavailable             PASS
CHECK no gfx906 hipBLASLt kernels        PASS gfx906=[] library=['TensileLibrary_gfx928.co', 'extop_gfx928.co']
CHECK FP8 block-scaled -> Triton         PASS selected TritonFp8BlockScaledMMKernel (numerics verified, see s8b_fp8_block_probe.py)
CHECK FP8 per-tensor (dynamic) refused   PASS ValueError: Failed to find a kernel that can implement the ScaledMM linear layer. Reasons:
CHECK FP8 per-tensor (static) refused    PASS ValueError: ...
CHECK FP8 per-channel refused            PASS ValueError: ...
CHECK VLLM_CACHE_ROOT under /mnt/game    PASS VLLM_CACHE_ROOT=/mnt/game/vllm-cache
CHECK compressed-tensors >= 0.15.0.1     PASS installed 0.15.0.1

CHECK_SUMMARY pass=14 fail=0
----- 1) OK -----
----- 2) FP8 block-scaled numeric probe (full sweep) -----
  4096^3  fp16 torch matmul  :      8.4 ms
  4096^3  fp32 torch matmul  :     15.1 ms
  4096^3  triton int8 (prod) :      7.7 ms
  4096^3  triton block fp8   :    304.0 ms
  selection PASS / numeric PASS / input_quant PASS / full_kernel PASS
  FP8_BLOCK_PROBE_DONE rc=0
----- 2) rc=0 (log /mnt/game/port/s8b/s8b-gate-fp8-probe.log) -----
----- 3) W4A16 kernel-selection test -----
  3 passed, 17 warnings in 6.58s
----- 3) rc=0 (log /mnt/game/port/s8b/s8b-gate-w4a16-sel.log) -----
----- 4) informational: _C_stable_libtorch identity (hard rule 15) -----
  md5 93bf7b4a391ac97e2f67cc7141633cb1  size 40691216
  S7c-fixed reference: 93bf7b4a391ac97e2f67cc7141633cb1 40691216
===== S8b gate summary =====
  conventions/kill-list : rc=0
  FP8 block-scaled      : rc=0
  W4A16 selection test  : rc=0
  RESULT: ALL GREEN
GATE_RC=0
```

⇒ **全绿**。第 4 节是"提示性"检查（只打印 `_C_stable_libtorch` 身份，不参与 rc），
但本次恰好与硬约束 15 的 S7c 基线 (`93bf7b4a…` / 40,691,216 B) 完全一致。

### 3.5 资源与安全自检

```console
$ （收工后）4 卡空闲：[(0,31.73),(1,31.73),(2,31.73),(3,31.73)] GiB / 31.98 GiB
```

- **未执行任何自编译 HIP 独立可执行程序**：只用 `python`（vLLM/triton/torch JIT）、`pytest`、`curl`、`docker exec`。
- **无挂死**：无 GPU 驱动死锁、无 `kill -9`、无 loadavg 异常；两次 FP8 sweep + 若干 pytest 全部自然退出。
- 生产容器 `qwen38-vllm` 全程仅 `docker ps -a` 只读；
  `z100-build`（2.10 基线）/`z100-build-211`/`z100-triton`/`z100-serve`/`z100-eager` 未做任何写入。
- **如实记录一处收尾瑕疵**：终止 W4A16 下载（§5.1）后，`z100-s8` 里留下 **2 个 `<defunct>` curl 僵尸**
  （`ps -o pid,ppid,stat` 原文：`7752 1 Z [curl] <defunct>`、`18291 1 Z [curl] <defunct>`）。
  原因是该容器 PID 1 是 `sleep infinity`，不回收孤儿进程；它们不占 CPU/显存、无网络活动
  （`free GiB [31.73, 31.73, 31.73, 31.73]`、`loadavg 0.58`），重启容器即消失。
  本步**未**为此重启或改动容器（保持 S8 交付原样）。

---

## 4. 门禁判定

| # | 门禁（本步验收） | 判定 | 证据指向 |
|---|---|---|---|
| 1 | FP8 block-scaled 有**数值探针**结论（rel_err + 是否可用） | ✅ **通过** | §3.2.2：17 组用例 `mean_rel_err` 0.000e+00~1.465e-07，`shapes with rel_err > 1e-2 or non-finite output: NONE` |
| 2 | 依据数值给出**门控 or 不门控**明确判定 | ✅ **通过（判为不门控）** | §3.2.5 + §5.2；`scaled_mm/**` 零改动，附一键门控补丁 |
| 3 | 核对**其它 FP8 路径**，确认无第二条静默误选 | ✅ **通过** | §3.2.1：per-tensor(动/静态)/per-channel 三条全部 `ValueError`；`_POSSIBLE_WFP8A16_KERNELS[ROCM]` 空；ROCm block 列表只有 aiter(False)+triton；MXFP8 走 BF16 模拟；deepseek_v4 复用同一 `fp8_linear` |
| 4 | `tests/.../test_w4a16_kernel_selection.py` 修成正确期望（含 gfx906） | ✅ **通过** | §2.1-A、§3.3；BEFORE `2 failed` → AFTER `3 passed` |
| 5 | 给出**前后对比** | ✅ **通过** | §3.3 两段原文 |
| 6 | `requirements/dcu*.txt` 确认/带上 `compressed-tensors==0.15.0.1` | ✅ **通过（无缺口，已重述 pin）** | §3.4.1：`dcu.txt:22`/`dcu-build.txt:12` `-r common.txt` + `common.txt:40` 已是该版本；新增 `dcu.txt:50-56` 自解释 pin |
| 7 | `tools/dcu/README.md` 写清"镜像自带 0.14.0.1 不够" | ✅ **通过** | §2.1-D；含 `ModuleNotFoundError` 原文与补装命令 |
| 8 | `tools/dcu/env.sh` 加 `VLLM_CACHE_ROOT` 默认导出（`/mnt/game` 下、可覆盖） | ✅ **通过** | §3.4.2；`env.sh:108-113`；默认 `/mnt/game/vllm-cache`、外部覆盖实测生效 |
| 9 | 新增 `tools/dcu/s8b_gate.sh` 固化约定与禁用清单 | ✅ **通过** | §3.4.3；14 项检查 `pass=14 fail=0` |
| 10 | 门禁项覆盖 `ROCM_SKINNY_GEMM_ENABLED=False` / `_rocm_C` 注册面 / aiter / hipBLASLt | ✅ **通过** | §3.4.3 第 2/3/4/5/8 行 |
| 11 | 门禁项覆盖 FP8 block-scaled 的处置 | ⚠️ **通过，但口径与任务书措辞不同** | 任务书写"FP8 block-scaled **被拒**"（以门控分支为前提）；本步实测数值干净，按 §1 判据**不门控**，故门禁改为断言"选中 `TritonFp8BlockScaledMMKernel` **且数值 sweep 通过**"（§5.2 已解释并附补丁） |
| 12 | 固定卡 3 + `VLLM_CACHE_ROOT` 落 `/mnt/game` | ✅ **通过** | §3.0/§3.4.2：`HIP_VISIBLE_DEVICES=3`、`VLLM_CACHE_ROOT=/mnt/game/vllm-cache` |
| 13 | 硬约束 1/2/3/4/5/13/16/17/18 + 无挂死 | ✅ **通过** | §1.3 逐条表 + §3.5 |
| 14 | 未改 `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`、S8 已改的 7 个文件 | ✅ **通过** | §2.1 变更总表；`git status` 仅含本步 5 个文件 + 本归档（§8） |
| 15 | S9 铺路清单（命令 + 权重/卡数） | ✅ **通过** | §6.2 |
| 16 | 出现挂死立即停止并如实记录 | ✅ **通过（本步无挂死）** | §3.5；W4A16 权重下载中断按 §5.1 如实记录，未当作阻塞 |

**S8b 总体：✅ 16/16 判定通过；其中门禁 11 的"口径"需要 Lead 知悉**（见 §5.2）。

---

## 5. 未完成项 / 阻塞与原因

### 5.1 明确"未执行"的项

| 项 | 原因 |
|---|---|
| **FP8 block-scaled 的真模型端到端** | 主机上没有 block-FP8 权重（现有 4 个本地模型：`Qwen3-0.6B` fp16、`Qwen3.5-4B`、`Qwen3.8-27B-Channel-INT8-w8a8`、截断的 `tinyllama-w4a16-group128`）。本步给出的是**选择级 + 算子级 + 整 kernel 路径级**三级证据（§3.2.1/§3.2.2/§3.2.3），**没有**跑出 block-FP8 模型的文本输出。若 S9 需要，最小可用做法是用 `llm-compressor` 把本地 `Qwen3-0.6B` 离线量化成 `FP8 block [128,128]`（无需网络） |
| **W4A16 真模型推理** | 网络：HF/HF-mirror 都可达（`huggingface.co` 200 / 4.2 s、`hf-mirror.com` 200），但 727 MiB 的 `model.safetensors` 下不动。本步做了一次**有界续传实验**（`curl -L --http1.1 --retry 3 -C -`，日志 `s8b-w4a16-download.log`），失败模式与 S8 一致但**定位更准**：① 该 URL 先返回一个 **1040 字节**的响应体（原文 `100  1040  100  1040  ...  384`），② 随后一次传输广告的总长是 **319M**（原文 `0  319M`），而 HF API 说文件是 761,968,600 B —— 即 **`resolve/main` 端点返回了错误的（偏短的）`Content-Length`**（S8 那次读到的是 `407 MiB`）；③ 因此 **`-C -` 续传反而破坏半成品**：文件从 528,191,560 B **缩到** 429,850,696 B，最终停在 **433,881,160 B**。已终止实验并把半成品改名 `model.safetensors.INCOMPLETE-433881160`（避免被当可用权重），期望 sha256 记在 §6.2。**结论：`curl -C -` 是这条链路上的错误工具**，S9 应改用 `huggingface_hub`（自带 LFS sha256 校验与 `.incomplete` 语义）并优先试 `HF_HUB_DISABLE_XET=1`（镜像装了 `hf-xet 1.4.3`，Xet 路径是主要嫌疑）、`HF_ENDPOINT=https://hf-mirror.com` 或 ModelScope；或走完全离线的 `llm-compressor` 路线。**没有**把它当成本步的门禁 |
| **生产 27B W8A8 INT8 端到端 / ms-per-token 对拍** | S8 已证单卡（31.98 GiB）装不下 28.76 GiB 权重 + `torch.compile` 停摆；任务书本步只给卡 3，多卡是 S9 范围 |
| **GDN / causal_conv1d 内核实机** | 同 S8，移交 S9（本地有 `Qwen3.5-4B`，1 卡可跑） |
| **GGUF / MoE 实机** | 设计文档划到 S10 阶段 B |
| **`--enforce-eager` 任何对照** | 硬约束 13；本步无任何 `serve` |

### 5.2 ⚠️ 需要 Lead 决策：FP8 block-scaled"不门控"与任务书措辞的前提冲突

**冲突点**：任务书 §1 把判定写成条件式——"若数值不可靠/未经证实 ⇒ 加显式门控；**若数值干净
⇒ 不要门控**，把证据写进归档并更新禁用清单（说明'已验证可用'）"；而 §3 要求 `s8b_gate.sh` 里
包含"**FP8 block-scaled 被拒**"这一检查项。后者只有在**门控分支**成立时才说得通。

**本步的取舍**：以实测为准 ⇒ 走"数值干净 ⇒ 不门控"。依据：

- 内核相对 fp32 反量化参考 `mean_rel_err ≤ 1.465e-07`（17 组用例、3 种输出 dtype、含 M=1 decode
  与 M=2048 prefill、含 N/K 非 128 倍数的尾块 mask 路径），全 finite；
- 输入量化核与 native 参考差 ≤ `1.144e-05`；
- 整条 `apply_weights` 相对**原始权重** fp32 matmul 差 `3.311e-07`；
- 机器码证明它只是"用 fp32 FMA 老老实实算"（§3.2.4），因此**不存在静默算错**；
- 反方论据（记录在案、但按判据不足以门控）：**性能上它是明确错误的路径**——
  4096³ **304 ms vs fp16 8.4 ms（≈36×）**，用户开 FP8 想要的是加速，拿到的是 36 倍减速。

**如果 Lead 更看重"性能上禁止误选"**，最小补丁（3 处、只动 `scaled_mm/**`，本步**未**应用）：

```python
# vllm/model_executor/kernels/linear/scaled_mm/triton.py:159-164
class TritonFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
            return False, "only CUDA-alike and XPU devices are supported."
        # gfx906 has no FP8 unit: the triton path is emulated with scalar
        # v_fmac_f32 and measured ~36x slower than fp16 (S8b), so refuse it
        # instead of silently selecting a path with no performance upside.
        from vllm.platforms.rocm import on_gfx906

        if current_platform.is_rocm() and on_gfx906():
            return False, (
                "block-scaled FP8 is emulated with scalar fp32 FMA on gfx906 "
                "(no FP8 hardware) and is ~36x slower than fp16; use fp16 instead"
            )
        return True, None
```

连带需要改 `tools/dcu/s8b_gate.sh` 的 FP8 检查（把 "block-scaled -> Triton" 换成
"block-scaled 被拒 + 原因串含 gfx906"），以及本档 §6.1 的对应行。**若走这条路线，`is_supported`
返回 False 后面板会得到 `ValueError: Failed to find a kernel ...`（明确报错，不会静默）**。

### 5.3 残余风险（本步发现 / 复核但未处置）

1. **FP8 block-scaled 的性能陷阱**（§3.2.5）：数值可用但 ≈36× 慢。S9 的 README 已测矩阵与
   `docs/hygon_z100.md` 需要明确写"Z100 上不要用 FP8 做加速"，否则用户会静默拿到 36× 减速。
2. **`rocm_per_tensor_float_w8a8_scaled_mm_impl` 自身不设防**（`scaled_mm/rocm.py:19-45`，S8 §5.4-2）：
   复核结论是**响亮失败而非静默算错**——`M<=4` 走 `torch.ops._rocm_C.wvSplitKQ`，该 op 在 gfx906
   未注册 ⇒ `AttributeError`。本步不改。
3. **没有 Z100 的 triton 调优配置**：`vllm/.../utils/configs/` 下没有
   `*device_name=Z100,dtype=fp8_w8a8,block_shape=[128,128].json`，每次都会打
   `Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!`
   ——对已被否决性能价值的路径不值得补配置，记录备查。
4. **bf16 设备探针不稳定**（S8 §3.12）：仍未处置，S9 必须显式 `--dtype float16`。
5. **`compressed-tensors` 只进了 requirements 文本，镜像没重建**：S9 起容器时仍需按 §6.2 补装
   （或重建镜像）。

---

## 6. 对后续步骤的影响

### 6.1 S9 禁用清单（S8 §6.1 的更新版；**加粗**为本步变化）

| # | 后端 / 路径 | 要求 | 判据（`file:line`） | 运行期证据 |
|---|---|---|---|---|
| 1 | attention `ROCM_ATTN`（`_rocm_C::paged_attention`） | 不指定 `--attention-backend`，让自动选择落 `TRITON_ATTN` | `vllm/platforms/rocm.py:602-604`、`:500-501` | S8 §3.3/§3.5；本步门禁 `use_rocm_custom_paged_attention=False` |
| 2 | `VLLM_ROCM_USE_SKINNY_GEMM` | 保持 `0`（`tools/dcu/env.sh:102` 已兜底） | `vllm/_custom_ops.py:20-27`、`scaled_mm/rocm.py:96-97` | 本步门禁：`ROCM_SKINNY_GEMM_ENABLED=False`；`_rocm_C ops=['_rocm_C::paged_attention']` |
| 3 | `VLLM_ROCM_USE_AITER*`（全部子开关） | 保持默认；不要为提速打开 | `vllm/_aiter_ops.py:202-218`、`:278-290` | **本步门禁：`is_aiter_found_and_supported()=False`，6 个 `is_*_enabled()` 全 `None`** |
| 4 | FlashAttention | 不用；不设 `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `vllm/attention/utils/fa_utils.py:22-27` | **本步门禁：`is_flash_attn_varlen_func_available()=False`** |
| 5 | hipBLASLt / MIOpen 的 gfx906 路径 | 无需开关（DTK 无 gfx906 kernel）；**不要**导出 `ROCBLAS_TENSILE_LIBPATH` | `tools/dcu/env.sh:88-97` | **本步门禁：`/opt/dtk/lib/hipblaslt/library/` 只有 `*gfx928*`** |
| 6 | **FP8（`--quantization fp8*`）** | **per-tensor/per-channel：禁用**（被 arch 明确否决）；**block-scaled：数值已验证可用（S8b），但性能上禁用**——4096³ 304 ms vs fp16 8.4 ms ≈ **36×**。要开就必须知道自己在拿 36× 减速换 FP8 权重体积 | 否决点：`scaled_mm/rocm.py:96-97`、`scaled_mm/pytorch.py` 三个 torch kernel 的 `is_supported`；block-scaled 无门控点在 `scaled_mm/triton.py:159-164`；模拟证据 `_w8a8_triton_block_scaled_mm` 的 asm | **本步 §3.2 全套（17 组用例 rel_err ≤1.465e-07 + 8160 条 `v_fmac_f32_e32` + 304 ms）**；门禁 3 行 FP8 检查 |
| 7 | `--enforce-eager` | **禁止**（硬约束 13） | `vllm/platforms/rocm.py:1362-1388` | S7 §3.7；本步无 `serve` |
| 8 | `--dtype auto` | **不推荐**，显式 `--dtype float16` | `vllm/platforms/rocm.py:286-291` | S8 §3.12 |
| 9 | vLLM 编译缓存落点 | **`tools/dcu/env.sh` 现在会默认导出 `VLLM_CACHE_ROOT=/mnt/game/vllm-cache`**（`/mnt/game` 存在时）；外部仍可覆盖 | **`tools/dcu/env.sh:108-113`** | **本步 §3.4.2 + 门禁第 13 项** |
| 10 | `compressed-tensors` < 0.15.0.1 | 必须先升到 **0.15.0.1**（`requirements/common.txt:40`，**`requirements/dcu.txt:56` 重述**） | `compressed_tensors_wNa8o8.py:14` | S8 §2.4；本步门禁第 14 项 `installed 0.15.0.1` |
| 11 | `test_w4a16_kernel_selection.py` | **本步已修好，可以当门禁用了**（gfx906 分支 + `has_g_idx` 拒绝用例） | `tests/kernels/quantization/test_w4a16_kernel_selection.py:36-49`、`:86-105` | **本步 §3.3 AFTER `3 passed`；`s8b_gate.sh` 第 3 段 rc=0** |
| 12 | **`_C_stable_libtorch` 身份** | 必须仍是 S7c 修复版（`md5 93bf7b4a391ac97e2f67cc7141633cb1` / 40,691,216 B），否则 packed silu 的 NaN 会回来 | 硬约束 15 | **本步门禁第 4 段打印一致** |
| 13 | **一键巡检** | S9 每次换容器/换树后先跑 `docker exec <c> bash <tools>/dcu/s8b_gate.sh`，`ALL GREEN` 再往下走 | `tools/dcu/s8b_gate.sh` | **本步 `GATE_RC=0`** |

### 6.2 S9 可直接粘贴的命令清单 + 权重 / 卡数需求

**共同前置**（每次新 shell）：

```bash
docker exec -it z100-s8 bash                       # 或 S9 新建容器（--memory>=24g，见下）
source /opt/dtk/env.sh
source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh   # VLLM_DIR / 自编 triton / 卡；默认卡 1
export S8_HIP_VISIBLE_DEVICES=3                     # 本步口径；S9 按需换
export VLLM_CACHE_ROOT=/mnt/game/vllm-cache         # env.sh 已默认，显式写更稳
# 硬约束：❌ 不要 --enforce-eager；✅ --dtype float16（不要 auto）
```

**S9 建议容器**（S8 §6.2 原样，`--memory` 从 10g 提到 32g 以容纳 27B 的 `torch.compile`）：

```bash
docker run -d --name z100-s9 \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    --group-add video --security-opt label=disable --ipc host \
    --memory=32g --memory-swap=32g --shm-size=16g \
    -v /opt/hyhal:/opt/hyhal:ro -v /home/lyh/z100-vllm:/src:ro -v /mnt/game:/mnt/game \
    -w /mnt/game/port \
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 \
    bash -c 'sleep infinity'
# 依赖（S7 的 6 个 + S8 的第 7 个；本步确认 dcu.txt 已重述 pin）
pip install --no-deps /mnt/game/port/wheels-211/torch-2.11.0+das.opt1.dtk2604.*.whl
pip install --no-deps /mnt/game/port/s7/pip-wheels/{mistral_common-1.11.5-py3-none-any.whl,openai-2.54.0-py3-none-any.whl,xgrammar-0.2.8-*.whl,apache_tvm_ffi-0.1.14.post1-*.whl,torchvision-0.26.0-*.whl}
pip install --no-deps /mnt/game/port/s8/pip-wheels/compressed_tensors-0.15.0.1-py3-none-any.whl
cp /src/tools/dcu/s7_usercustomize.py /root/.local/lib/python3.10/site-packages/usercustomize.py
cp -r /src/tools/dcu /mnt/game/port/gfx906-vllm-211/tools/     # 带上 s8b_gate.sh
cp /src/tests/kernels/quantization/test_w4a16_kernel_selection.py \
   /mnt/game/port/gfx906-vllm-211/tests/kernels/quantization/
cp /src/requirements/dcu.txt /mnt/game/port/gfx906-vllm-211/requirements/
bash /mnt/game/port/gfx906-vllm-211/tools/dcu/s8b_gate.sh      # 期望 RESULT: ALL GREEN
```

**逐项清单**（权重、卡数、命令）：

| # | S9 目的 | 需要的权重（本地/可获取） | 卡数 | 可直接粘贴的命令 |
|---|---|---|---|---|
| 1 | 巡检（每次换环境先跑） | 无 | 1（只做选择/数值探针） | `bash /mnt/game/port/gfx906-vllm-211/tools/dcu/s8b_gate.sh` |
| 2 | 8K 冒烟（S8 门禁 1 复现） | `Qwen3-0.6B`（本地，1.5 GB） | 1 | `python -c "import torch;from vllm.entrypoints.cli.main import main;main()" serve /mnt/game/models/Qwen3-0.6B --dtype float16 --max-model-len 8192 --gpu-memory-utilization 0.9 --port 8002` + 请求（见 S8 §7-5） |
| 3 | KV cache / graph（S7 复现） | `Qwen3-0.6B` | 1 | 同 S7 §7 |
| 4 | **W4A16 端到端** | `nm-testing/tinyllama-oneshot-w4a16-group128-v2`：**727 MiB**，`model.safetensors` = 761,968,600 B，sha256 `456249d063ce70767c9bd2e16f1bca336ee1c14c486a3532e502e831d77c885d`。本地有 **433,881,160 B** 的半成品 `model.safetensors.INCOMPLETE-433881160`（**别用 `curl -C -` 续传**：该端点会返回偏短的 `Content-Length`，续传会把文件越搞越小，见 §5.1） | 1 | **首选**（带 sha256 校验、自身管理 `.incomplete`）：`HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import hf_hub_download as d; print(d('nm-testing/tinyllama-oneshot-w4a16-group128-v2','model.safetensors',local_dir='/mnt/game/models/tinyllama-w4a16-group128'))"`，必要时再加 `HF_ENDPOINT=https://hf-mirror.com`；下完 `sha256sum` 对上面的值。然后 `python -m pytest -q --noconftest tests/kernels/quantization/test_rocm_compressed_tensors_w4a16.py`。**替代来源**：ModelScope 镜像 / 任意 CT W4A16 或 AWQ-int4 / GPTQ-int4 小模型（都会走 `Gfx906GPTQWNA16LinearKernel`）。**完全离线替代**：`llm-compressor` 把本地 `Qwen3-0.6B` 量化成 CT W4A16，不依赖网络（推荐给 S9 做**可重复**的 W4A16 门禁） |
| 5 | **生产 27B W8A8 INT8 端到端 + ms/token** | `/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8`（本地，18 shard / **28.76 GiB** / `Qwen3_5ForConditionalGeneration` / compressed-tensors channel-wise W8A8-int8） | **≥2（建议 TP=2；4 卡更稳）**——单卡 31.98 GiB 装 28.51 GiB 权重后 `torch.compile` 停摆（S8 §3.9），生产本来也是多卡 | `python -c "import torch;from vllm.entrypoints.cli.main import main;main()" serve /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 --dtype float16 --tensor-parallel-size 2 --max-model-len 4096 --gpu-memory-utilization 0.9 --port 8003`（`--memory=32g`；预期日志含 `Selected TritonInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8`；对拍 13–14 ms/token 基线） |
| 6 | GDN / causal_conv1d gfx906 内核实机 | `Qwen3.5-4B`（本地，8.8 GB） | 1 | `python -c "import torch;from vllm.entrypoints.cli.main import main;main()" serve /mnt/game/models/Qwen3.5-4B --dtype float16 --max-model-len 4096 --port 8004` 后发一条请求，并在日志/`torch.profiler` 里确认 `rms_norm_gated_gfx906` / `causal_conv1d_gfx906_*` / `fused_sigmoid_gating_delta_rule_gfx906_*` 被调用（7 个 op 注册面见 S8 §3.11） |
| 7 | 设计文档门禁 4：4 卡 decode + 显存偏差 <5% | `Qwen3-0.6B`（或生产模型） | 4 | 同 S7/S8，`torch.cuda.mem_get_info` 口径（`hy-smi` 在宿主不可用，S7 §5.3） |
| 8 | （可选）FP8 block-scaled 真模型 | **无本地权重**；需要 block-FP8 检查点 | 1 | 离线量化：`pip install llmcompressor && python -c "..."` 把 `Qwen3-0.6B` 压成 `FP8 block_size=[128,128]`；或 `--quantization fp8` + 已有 block-FP8 模型（注意 §6.1 第 6 行：数值可用但 ≈36× 慢） |

**给 S9 的三个"踩过坑"提醒**：

1. `import vllm` 前**必须先 `import torch`**（硬约束 8），否则 `std::bad_alloc` → SIGABRT(134)。
2. 跑 `pytest` 用 `--noconftest`（镜像依赖不全，`tests/conftest.py` 起不来），
   命令形态照抄 §3.3；`tests/utils.py` 在 `$VLLM_DIR` 下可直接 import。
3. 容器内**写不了宿主 `/mnt/game` 之外的地方**有配额问题、写 `/mnt/game` 必须经 `docker exec`；
   宿主侧 `/mnt/game` 对当前用户**只读**（本步 `tee /mnt/game/...` 被拒，改成容器内重定向）。

### 6.3 被本步更新 / 推翻的前提

1. **推翻**"FP8 block-scaled 会静默算错"：gfx906 上它是**标量 fp32 FMA 软件模拟**，
   17 组用例相对 fp32 反量化参考 `mean_rel_err ≤ 1.465e-07` ⇒ **数值可信**。
   S8 §5.4-1 的"无 arch 门控"风险，从"可能算错"降级为"性能上不该用"。
2. **新增**：FP8 block-scaled 的代价是 **≈36× fp16**（4096³，304 ms vs 8.4 ms），
   机器码层面 `v_mfma/v_dot/packed f16/fp8 cvt` 全为 0、`v_fmac_f32_e32` 8160 条。
3. **确认**：per-tensor / per-channel FP8 在 gfx906 上**三条路径全部明确报错**，
   `_POSSIBLE_WFP8A16_KERNELS[ROCM]` 为空 ⇒ S8 §6.1 第 6 行的"per-tensor 会被否决"得到运行期复核。
4. **确认**：`requirements/dcu*.txt` 通过 `-r common.txt` 已带上 `compressed-tensors==0.15.0.1`，
   **原本没有版本缺口**；本步只是把它重述成自解释的 pin（`requirements/dcu.txt:56`）。
5. **落地**：`VLLM_CACHE_ROOT` 默认导出进了 `tools/dcu/env.sh`（硬约束 17 从"任务书记录"变成
   "环境脚本保证"）；注意 `s8_env.sh` 的旧默认路径因此被继承覆盖（§3.4.2 副作用）。
6. **修正**：`test_w4a16_kernel_selection.py` 从"结构性不可过、不要当门禁"变成**可用门禁**
   （S8 §5.2 / §6.1 第 11 行的遗留项关闭），并新增 `has_g_idx` 拒绝用例。
7. **确认**：`_C_stable_libtorch` 仍是 S7c 修复版 `93bf7b4a…`（40,691,216 B），本步未触碰构建产物。

---

## 7. 复现方式

```bash
# ---------- 0) 前置：宿主 docker 可用；容器 z100-s8 在（S8 建，未 --rm）；
#              canonical 副本 /mnt/game/port/gfx906-vllm-211 仍带 gfx906-only 4 扩展
#              宿主对 /mnt/game 只读 ⇒ 所有 /mnt/game 写入都经 docker exec

# ---------- 1) 同步本步代码/脚本进容器（仓库侧改动已在工作区） ----------
docker exec z100-s8 bash -lc '
  set -e
  mkdir -p /mnt/game/port/s8b /mnt/game/vllm-cache
  cp /src/tools/dcu/{s8b_gate.sh,s8b_fp8_block_probe.py,env.sh,README.md} \
     /mnt/game/port/gfx906-vllm-211/tools/dcu/
  chmod +x /mnt/game/port/gfx906-vllm-211/tools/dcu/s8b_gate.sh
  cp /src/tests/kernels/quantization/test_w4a16_kernel_selection.py \
     /mnt/game/port/gfx906-vllm-211/tests/kernels/quantization/
  cp /src/requirements/dcu.txt /mnt/game/port/gfx906-vllm-211/requirements/
'

# ---------- 2) 环境（每次新 shell；卡 3；缓存全落 /mnt/game） ----------
docker exec -it z100-s8 bash
source /opt/dtk/env.sh
export S8_HIP_VISIBLE_DEVICES=3
export VLLM_CACHE_ROOT=/mnt/game/vllm-cache
source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh     # → VLLM_DIR + 自编 triton
cd $VLLM_DIR

# ---------- 3) FP8 block-scaled 数值探针（§3.2 全部证据；rc=0 表示全部通过） ----------
python /mnt/game/port/gfx906-vllm-211/tools/dcu/s8b_fp8_block_probe.py \
  | tee /mnt/game/port/s8b/s8b-fp8-block-probe.log
#   期望：selection/numeric/input_quant/full_kernel 全 PASS；FP8_BLOCK_PROBE_DONE rc=0
#         "shapes with rel_err > 1e-2 or non-finite output: NONE"
#         4096^3 triton block fp8 ≈ 300 ms（对比 fp16 ≈ 8.4 ms）

# ---------- 4) 机器码证据（证明是软件模拟） ----------
D=$(grep -rl "w8a8_triton_block_scaled_mm" /mnt/game/triton-gfx906/cache | grep "\.amdgcn$" | xargs -r ls -S | head -1 | xargs dirname)
A=$D/_w8a8_triton_block_scaled_mm.amdgcn
grep -oE "^\s+[a-z_][a-z0-9_]*" $A | tr -d " " | sort | uniq -c | sort -rn | head
for op in v_mfma v_dot2_f32_f16 v_pk_fma_f16 v_cvt_f32_fp8; do printf "%-18s %s\n" $op "$(grep -c $op $A)"; done
#   期望：v_fmac_f32_e32 约 8160；v_mfma / v_dot / packed / fp8 cvt 全 0

# ---------- 5) 过期测试期望：BEFORE / AFTER ----------
git -C /home/lyh/z100-vllm show HEAD:tests/kernels/quantization/test_w4a16_kernel_selection.py \
  | docker exec -i z100-s8 bash -c 'cat > /mnt/game/port/s8b/test_w4a16_kernel_selection_BEFORE.py'
docker exec z100-s8 bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1
  export S8_HIP_VISIBLE_DEVICES=3 VLLM_CACHE_ROOT=/mnt/game/vllm-cache
  source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh >/dev/null 2>&1; cd $VLLM_DIR
  python -c "import torch, pytest, sys; sys.exit(pytest.main([\"-v\",\"--no-header\",\"--noconftest\",\"-p\",\"no:cacheprovider\",\"/mnt/game/port/s8b/test_w4a16_kernel_selection_BEFORE.py\"]))" \
    2>&1 | tee /mnt/game/port/s8b/s8b-pytest-w4a16-sel-BEFORE.log'      # 期望 2 failed
docker exec z100-s8 bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1
  export S8_HIP_VISIBLE_DEVICES=3 VLLM_CACHE_ROOT=/mnt/game/vllm-cache
  source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh >/dev/null 2>&1; cd $VLLM_DIR
  python -c "import torch, pytest, sys; sys.exit(pytest.main([\"-v\",\"--no-header\",\"--noconftest\",\"-p\",\"no:cacheprovider\",\"tests/kernels/quantization/test_w4a16_kernel_selection.py\"]))" \
    2>&1 | tee /mnt/game/port/s8b/s8b-pytest-w4a16-sel-AFTER.log'       # 期望 3 passed

# ---------- 6) 一键门禁（14 项 + FP8 探针 + W4A16 测试；期望 ALL GREEN / rc=0） ----------
docker exec z100-s8 bash -lc 'bash /mnt/game/port/gfx906-vllm-211/tools/dcu/s8b_gate.sh \
  > /mnt/game/port/s8b/s8b-gate.log 2>&1; echo "GATE_RC=$?"; tail -8 /mnt/game/port/s8b/s8b-gate.log'

# ---------- 7) 依赖 / 缓存约定自检 ----------
docker exec z100-s8 bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1
  ( source /mnt/game/port/gfx906-vllm-211/tools/dcu/env.sh ) 2>&1 | grep VLLM_CACHE_ROOT   # /mnt/game/vllm-cache
  ( export VLLM_CACHE_ROOT=/mnt/game/x; source /mnt/game/port/gfx906-vllm-211/tools/dcu/env.sh ) 2>&1 | grep VLLM_CACHE_ROOT  # 覆盖生效
  python -m pip install --dry-run --no-deps "compressed-tensors==0.15.0.1" | tail -1'

# ---------- 8) （可选）W4A16 权重下载（S8b §5.1；别用 curl -C -，别当门禁） ----------
# 本步实测：resolve/main 端点会返回偏短的 Content-Length（原文 `0  319M`，真实 761,968,600 B），
# `curl -C -` 续传会让文件越变越小。改用自带 sha256 校验的 huggingface_hub：
docker exec z100-s8 bash -lc '
  export HF_HUB_DISABLE_XET=1          # 镜像里装了 hf-xet 1.4.3，主嫌疑
  # 必要时再加：export HF_ENDPOINT=https://hf-mirror.com
  rm -f /mnt/game/models/tinyllama-w4a16-group128/model.safetensors*   # 丢掉 curl 的坏半成品
  python -c "from huggingface_hub import hf_hub_download as d; print(d(
      \"nm-testing/tinyllama-oneshot-w4a16-group128-v2\", \"model.safetensors\",
      local_dir=\"/mnt/game/models/tinyllama-w4a16-group128\"))"
  sha256sum /mnt/game/models/tinyllama-w4a16-group128/model.safetensors
  # 期望 456249d063ce70767c9bd2e16f1bca336ee1c14c486a3532e502e831d77c885d'
```

---

## 8. 参考

- 设计文档：[docs/design/z100-08-attention-and-quant-paths.md](../design/z100-08-attention-and-quant-paths.md)
- 前序归档：
  [z100-08-attention-and-quant-paths-result.md](z100-08-attention-and-quant-paths-result.md)（S8：路径/后端核对、禁用清单 §6.1、两个 fork 真 bug、§5.2 测试期望过期、§5.4 残余风险）、
  [z100-05-csrc-kernels-result.md](z100-05-csrc-kernels-result.md) §4/§6（`wvSplitK*` 摘除、`ROCM_SKINNY_GEMM_ENABLED` 探测点、`s5_gate.sh` 风格）、
  [z100-06-triton-gfx906-result.md](z100-06-triton-gfx906-result.md)（自编 triton 与 SFU hazard）、
  [z100-07c-packed-silu-csrc-fix-result.md](z100-07c-packed-silu-csrc-fix-result.md)（`_C_stable_libtorch` 身份 `93bf7b4a…`）
- 关键源码位置（本步引用）：
  - FP8 内核表与选择器：`vllm/model_executor/kernels/linear/__init__.py:411-501`（`_POSSIBLE_INT8/FP8/FP8_BLOCK/WFP8A16` 表）、
    `:503-522`（`_POSSIBLE_KERNELS[ROCM]`，`Gfx906GPTQWNA16LinearKernel` 第 1 位）、`:635-706`（`choose_scaled_mm_linear_kernel`）、
    `:705-770`（`init_fp8_linear_kernel`，`is_per_group()` 分流点）、`:817-890`（`choose_mp_linear_kernel`）
  - block-scaled FP8：`vllm/model_executor/kernels/linear/scaled_mm/triton.py:159-180`（`TritonFp8BlockScaledMMKernel`，**无 arch 门控处**）、
    `:216-220`（自定义 op 注册）、`scaled_mm/BlockScaledMMLinearKernel.py:58-151`（`apply_weights` / `FP8BlockParams`）、
    `vllm/model_executor/layers/quantization/utils/fp8_utils.py:760-840`（`_w8a8_triton_block_scaled_mm`，`tl.dot`）、
    `:880-1009`（`w8a8_triton_block_scaled_mm`，含 gfx1250 反量化特例）、`:1416-1450`（`process_fp8_weight_block_strategy`）
  - FP8 per-tensor/per-channel 否决：`scaled_mm/rocm.py:74-106`、`scaled_mm/pytorch.py`（3 个 torch kernel 的 `is_supported`）、
    `scaled_mm/rocm.py:19-45`（`rocm_per_tensor_float_w8a8_scaled_mm_impl` 的 `wvSplitKQ` 直调）
  - W4A16：`vllm/model_executor/kernels/linear/mixed_precision/gfx906_gptq_wna16.py:20-47`（`SUPPORTED_QUANT_TYPES` / `can_implement` 的 `has_g_idx` 拒绝）
  - 输入量化：`vllm/model_executor/layers/quantization/input_quant_fp8.py:136-160`（`forward_hip` 的 aiter 分流 + `forward_cuda` 回退）
  - 本步改动：`tests/kernels/quantization/test_w4a16_kernel_selection.py:19-22/27-34/36-49/86-105`、
    `tools/dcu/env.sh:108-119`、`requirements/dcu.txt:50-56`、`tools/dcu/README.md`
- 本步脚本（仓库）：
  [`tools/dcu/s8b_fp8_block_probe.py`](../../tools/dcu/s8b_fp8_block_probe.py)、
  [`tools/dcu/s8b_gate.sh`](../../tools/dcu/s8b_gate.sh)、
  [`tools/dcu/env.sh`](../../tools/dcu/env.sh)、
  [`tools/dcu/README.md`](../../tools/dcu/README.md)
- 本步原始日志（全部在 `/mnt/game/port/s8b/`，容器内）：
  `s8b-fp8-block-probe.log`、`s8b-gate.log`、`s8b-gate-fp8-probe.log`、`s8b-gate-w4a16-sel.log`、
  `s8b-pytest-w4a16-sel-BEFORE.log`、`s8b-pytest-w4a16-sel-AFTER.log`、`s8b-w4a16-download.log`、
  `test_w4a16_kernel_selection_BEFORE.py`
