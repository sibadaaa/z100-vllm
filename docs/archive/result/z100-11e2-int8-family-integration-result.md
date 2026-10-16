# S11e2 INT8 `sdot4` 并入 gfx906 仿真内核族 — 执行结果

> 归档时间：2026-10-11
> 执行者：Lead
> 对应设计文档：[docs/design/z100-11e2-int8-family-integration.md](../design/z100-11e2-int8-family-integration.md)
> 前一档：[z100-11e-w8a8-sdot4-result.md](z100-11e-w8a8-sdot4-result.md)（内核落地）、[z100-11k-sdot4-compile-integration-result.md](z100-11k-sdot4-compile-integration-result.md)（编译集成）
> 状态：✅ 完成（内核扩展 + 单测 + 开关组合 + 回归对照全过）

## 1. 目标与验收

设计文档 §验收 原文：

1. 新增 `tests/kernels/quantization/test_gfx906_int8_sdot4.py` 全绿，FP8/BF16 两套不回归；
2. `s11e_sdot4_eval.py --mode int` 仍 bit_equal 全 True、`--mode check` rel_err 与 S11e 归档同量级；
3. 开关组合：未设置 → 关；`VLLM_DCU_USE_GFX906_EMU=1` → 开；再显式 `VLLM_DCU_USE_INT8_SDOT4=0` → 关；
4. `v_dot4_i32_i8` 机器码计数与 S11e 同档；
5. `isa_probe.sh` 53 条基线不受影响。

## 2. 实际改动

| 文件 | 改动 |
|---|---|
| `csrc/rocm/w8a8_sdot4.cu` | GEMV/GEMM kernel 与 launcher 增加 `x_s_stride`/`w_s_stride`；`check_common` 放宽为 `x_s.numel() ∈ {M,1}`、`w_s.numel() ∈ {N,1}` 并推导 stride；epilogue 改为按 stride 取 scale。**计算路径（`v_dot4_i32_i8` int32 精确累加）零改动** |
| `csrc/rocm/w8a8_sdot4.h` | 补 scale 契约注释（与 `fp8_emu_gemm.h` 对齐） |
| `vllm/model_executor/kernels/linear/scaled_mm/gfx906_common.py`（新） | `gfx906_emu_enabled(per_kernel_env)`：显式单项开关优先，否则跟随 `VLLM_DCU_USE_GFX906_EMU`；用 `os.environ` 区分"未设置"与"显式 0" |
| `vllm/envs.py` | 新增 `VLLM_DCU_USE_GFX906_EMU`(默认 False)；`VLLM_DCU_USE_INT8_SDOT4` 注释改为"默认关是因为会数值性地改变 greedy token（S11l/S11m）" |
| `.../scaled_mm/gfx906_sdot4.py` | `can_implement` 只保留"输入对称"；权重 scale 接受 per-channel/per-tensor；静态 per-tensor 输入 scale 保留并校验为标量；`apply_weights` 支持静态/动态两条量化路径；docstring 去掉过期的"编译集成未修好"理由；门控改走 `gfx906_emu_enabled` |
| `.../scaled_mm/gfx906_fp8.py` | 门控改走 `gfx906_emu_enabled`（可被总开关打开） |
| `vllm/model_executor/layers/utils.py` | dense GEMV 门控改走 `_gfx906_dense_gemv_enabled()`（同一入口） |
| `tests/kernels/quantization/test_gfx906_int8_sdot4.py`（新） | 28 项：int32 位精确、去量化参考（3 种输出 dtype）、4 种 scale 组合、bias、K 边界、错误路径、开关组合、`can_implement` 矩阵、`custom_op`+fake |

产物：`vllm/_rocm_C.abi3.so` md5 **`56fefa7c1e9fe36291e99942d941f813`**
（上一版 `c7b825b8ded1a1f8aa179a951d9dd4c4`；工作区与容器构建树硬链接同步）。

## 3. 原始证据

### 3.1 构建

```bash
$ docker exec z100-vllm bash /mnt/game/port/build-211.sh _rocm_C
[2/3] Building HIP object CMakeFiles/_rocm_C.dir/csrc/rocm/w8a8_sdot4.hip.o
[3/3] Linking HIP shared module _rocm_C.abi3.so
----- ninja rc=0 2026-10-11T07:57:10+08:00 -----
```

### 3.2 单测（门禁 1）

```bash
$ cd /mnt/game/port/s11f/t && python -m pytest test_gfx906_int8_sdot4.py -q
27 passed, 1 skipped        # 修掉 custom_op 测试的导入后：
$ python -m pytest test_gfx906_int8_sdot4.py test_gfx906_fp8_emu.py test_gfx906_bf16_gemv.py -q
56 passed, 17 warnings in 21.25s
```

新增用例覆盖的 scale 组合（全部对 fp32 去量化参考，容差 = 输出 dtype 半个 ulp）：
per-channel `[N]` × 动态 per-token `[M]`、per-tensor `[1]` × 动态、per-channel × 静态
per-tensor、**per-tensor × per-tensor**；另有 K ∈ {12,68}（非 tile 整数倍）与
`K % 4 != 0`、scale 长度非 M/1 的报错路径。

### 3.3 回归对照（门禁 2）

```bash
$ HIP_VISIBLE_DEVICES=2 python tools/dcu/s11e_sdot4_eval.py --mode int
  ... 35 个 (N,K) × M ∈ {1,4,8,32,64} 全部 bit_equal=True max_abs_diff=0
elapsed 11.1s

$ export PYTHONPATH=/mnt/game/triton-gfx906/install:$PYTHONPATH   # 自编 triton（S11e 口径）
$ HIP_VISIBLE_DEVICES=2 python tools/dcu/s11e_sdot4_eval.py --mode check
WORST sdot4 rel_err = 2.988e-04   (gate <= 1e-2)
WORST triton rel_err = 2.988e-04
  ... 每个形状 sdot4_agrees_with_triton=true
```

`2.988e-04` 与 [S11e 归档](z100-11e-w8a8-sdot4-result.md) 的记录**逐位一致**，
逐形状 `sdot4_agrees_with_triton=true` ⇒ stride 扩展没有触碰计算路径。

> 环境事实：`--mode check` 必须带上自编 triton 的 `PYTHONPATH`
> （`/mnt/game/triton-gfx906/install`），否则 DTK 自带 triton 在编译
> `triton_scaled_mm` 时报 `PassManager::run failed`（与本步改动无关，属 S11e 记录的口径）。

### 3.4 机器码（门禁 4）

同 S11d 的拆包方法（`.hip_fatbin` → `clang-offload-bundler` → `llvm-objdump`），
最终件 76 个 kernel：

```
s_endpgm             76
v_dot4_i32_i8      9255     <- W8A8 INT8 内核（S11e 落地的那条指令）
v_dot2_f32_f16    18666     <- FP8 / BF16 仿真内核
ds_read_u16        1092     <- FP8 的 256 项 LDS LUT
v_fmac_f32            0     <- 无标量模拟
```

### 3.5 开关组合（门禁 3）

由 `test_gate_composes_master_switch` 断言三种状态：

```
未设置                                    -> gfx906_emu_enabled = False, is_supported = (False, "disabled by VLLM_DCU_USE_INT8_SDOT4=0 ...")
VLLM_DCU_USE_GFX906_EMU=1                 -> True,  is_supported = (True, None)
+ VLLM_DCU_USE_INT8_SDOT4=0（显式覆盖）    -> False
```

### 3.6 ISA 探针（门禁 5）

`ISA_PROBE_STRICT=1 bash tools/dcu/isa_probe.sh` 未受影响（本步未动探针表）：
`baseline: MATCH (all 53 expectations hold)`、退出码 0。

## 4. 判定

| 门禁 | 结果 | 证据 |
|---|---|---|
| 1 单测 | ✅ 三套合计 56 passed | §3.2 |
| 2 回归对照 | ✅ int 35×5 全 bit_equal；check WORST 2.988e-04 与 S11e 一致、与 Triton 逐位一致 | §3.3 |
| 3 开关组合 | ✅ 未设置关 / 总开关开 / 显式 0 覆盖 | §3.5 |
| 4 机器码 | ✅ `v_dot4_i32_i8` 9,255 条仍在，`v_fmac_f32`=0 | §3.4 |
| 5 ISA 基线 | ✅ 53/53 | §3.6 |

## 5. 说明与遗留

1. **默认值未改**：仓库默认仍是总开关 OFF + 三个单项 OFF；生产照旧显式
   `VLLM_DCU_USE_INT8_SDOT4=1`（3.9× decode）。若把总开关默认打开，会同时改变
   27B W8A8 的 greedy token（数值性，S11l/S11m），故不做。
2. **非对称激活量化未实现**：kernel 没有 azp 修正项，`can_implement` 仍要求
   `input_symmetric=True`；命中的配置会回落 `TritonInt8ScaledMMLinearKernel`。
3. **静态输入 scale 的实际覆盖率未端到端验证**：单测覆盖了 kernel 与 Python 契约，
   但没有一个带静态输入量化的真实 checkpoint 在本机跑过（本机只有对称动态量化的 27B
   W8A8）。若后续拿到该类权重，按设计文档 §失败处置 检查。
4. **未做**：INT8 的 MoE/GGUF 变体（GGUF 的 `__dp4a` 路径在 csrc/quantization/gguf 中独立存在）。
