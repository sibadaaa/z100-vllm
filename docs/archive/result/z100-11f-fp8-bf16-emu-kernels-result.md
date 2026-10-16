# S11f / P6 FP8 与 BF16 仿真 HIP 内核 — 执行结果

> 归档时间：2026-10-11
> 执行者：Lead（本任务全程在同一会话内完成）
> 对应设计文档：[docs/design/z100-11f-fp8-bf16-emu-kernels.md](../design/z100-11f-fp8-bf16-emu-kernels.md)
> 前一档：[z100-11e-w8a8-sdot4-result.md](z100-11e-w8a8-sdot4-result.md)（骨架来源）、[z100-11d-z100-isa-capability-audit-result.md](z100-11d-z100-isa-capability-audit-result.md)（指令矩阵）
> 状态：✅ 内核级 + **block-scaled（128×128）** + **端到端**（Qwen3.8-27B-FP8，TP=4）全部完成；
> 2026-10-11 复核修掉 4 处问题（§8），端到端结果见 §9.4

## 1. 目标与验收标准

设计文档 §5 的门禁原文：

1. `ISA_PROBE_STRICT=1 bash tools/dcu/isa_probe.sh` → `baseline: MATCH`、退出码 0、`SYNTAX=0`；
2. `tests/kernels/quantization/test_gfx906_fp8_emu.py` + `tests/kernels/test_gfx906_bf16_gemv.py` 全绿；
3. 机器码门禁：MAC 循环必须是 `v_dot2_f32_f16`（+ fp8 用 `ds_read_u16` LUT），**不得**出现 S8b 那种纯 `v_fmac_f32` 标量模拟；
4. 默认 OFF、开关关闭即回到原行为；非 DTK/非 gfx906 行为不变。

## 2. 实际改动

| 文件 | 行数 | 改动摘要 |
|---|---|---|
| `csrc/rocm/gfx906_emu_common.cuh` | +180（新） | 共享转换层：`dot2`、`bf16x2_to_f32x2`（`v_lshlrev_b32 16`）、`f32_to_bf16`（RN）、`bf16_to_f16_guarded`（逐 lane 饱和/冲零）、`bf16x2_to_f16x2_fast`（窄契约、32 位标量逐 lane）、fp8 LUT 加宽 |
| `csrc/rocm/fp8_emu_gemm.cu` | +593（新） | W8A8 FP8 GEMV（M ≤ 8：一 block 一列 + LDS 归约 + A 加宽外提）与 GEMM（BM×BN、BK=32 双缓冲 LDS、cfg 1..4），fp8→fp16 走 256 项 LDS LUT |
| `csrc/rocm/fp8_emu_gemm.h` | +42（新） | 4 个入口（gemv/gemm + meta） |
| `csrc/rocm/bf16_emu_gemv.cu` | +317（新） | dense fp16/bf16 decode GEMV：route 0 自动 / 1 = fp32（bf16 默认，精确）/ 2 = fp16 dot2 |
| `csrc/rocm/bf16_emu_gemv.h` | +34（新） | 入口 + meta |
| `csrc/rocm/ops.h` / `torch_bindings.cpp` | +2 / +27 | 注册 `fp8_emu_gemv`、`fp8_emu_gemm`、`bf16_emu_gemv`（CUDA + Meta 两处） |
| `CMakeLists.txt` | +2 | `VLLM_ROCM_EXT_SRC` 加 `fp8_emu_gemm.cu`、`bf16_emu_gemv.cu` |
| `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py` | +238（新） | `Gfx906Fp8ScaledMMLinearKernel`（`is_supported`/`can_implement`/`apply_scaled_mm`）、主机侧无损 LUT（按 dtype 缓存 + 每设备缓存）、`torch.library.custom_op("vllm::fp8_emu_mm")` + `register_fake` |
| `vllm/model_executor/kernels/linear/__init__.py` | +4 | ROCM FP8 选核表首位插入 `Gfx906Fp8ScaledMMLinearKernel`（关闭时 `is_supported=False`，行为不变） |
| `vllm/model_executor/layers/utils.py` | +91 | `rocm_unquantized_gemm_impl` 的 gfx906 分支（`vllm::bf16_emu_mm`）+ 实测收益形状表 `_gfx906_gemv_shape_ok` |
| `vllm/envs.py` | +37 | `VLLM_DCU_USE_FP8_EMU`(False)、`VLLM_DCU_FP8_CVT`、`VLLM_DCU_USE_BF16_GEMV`(False)、`VLLM_DCU_BF16_ROUTE` |
| `tools/dcu/isa_probe.sh` | +65 | 21 条 `FP8EMU-*` 汇编行（含 3 条负控制）+ 1 条 builtin + 9 条 codegen 行；基线 21 → **53** |
| `tests/kernels/quantization/test_gfx906_fp8_emu.py` | +235（新） | LUT 无损性（e4m3/e5m2 全 256 码）、GEMV/GEMM 对 fp32 参考、奇数形状、per-tensor/per-token、bias、门控与 `can_implement` 矩阵 |
| `tests/kernels/test_gfx906_bf16_gemv.py` | +176（新） | 两条路线 vs fp32 参考（ulp 口径）、bf16 小权重精确性、窄契约反例、范围检查数据、bias、非法形状 |
| `tools/dcu/s11f_baseline_gemm_probe.py` | +101（新） | rocBLAS fp16/bf16/fp32 基线（27B 形状） |
| `tools/dcu/s11f_fp8_emu_probe.py` | +196（新） | FP8 数值 sweep + 性能对照 |
| `tools/dcu/s11f_bf16_gemv_probe.py` | +103（新） | dense GEMV 对 rocBLAS 的逐形状加速比 + ulp 误差 |

产物：`vllm/_rocm_C.abi3.so` md5 **`c7b825b8ded1a1f8aa179a951d9dd4c4`**
（旧 `ebd6fb8dac572514cf94ca7ec7d828a7`；单测/性能/机器码统计全部针对新件）。
工作区与容器构建树 `/mnt/game/port/gfx906-vllm-211` 的同名文件 md5 一致（两者为硬链接）。

> 注：本步中途一版为 `7415fbb89bfeb4a2aa87a8805644092b`（与最终件仅差注释行号）；
> 最后一次全量重建得到 `c7b825b8…`，其机器码直方图与前一版逐项相同
> （76 kernel / dot2 18,666 / `ds_read_u16` 1,092 / `v_fmac_f32` 0），
> 28 项单测与 288/288 数值 sweep 在该件上重跑通过。

## 3. 执行过程与原始证据

### 3.1 ISA 探针（门禁 1）

```bash
$ bash tools/dcu/isa_probe.sh /tmp/isa_probe_s11f          # 首次
assembler : OK=49 FAIL=15 SYNTAX=0
builtins  : HAS=1 18 / HAS=0 12
codegen   : EMITTED=29 MISSING=0 ERROR=0
baseline  : MATCH (all 53 expectations hold)

$ ISA_PROBE_STRICT=1 bash tools/dcu/isa_probe.sh /tmp/isa_probe_s11f_strict
strict exit=0
```

新增行实测（`assembler.tsv` 原文摘录）：

```
OK    FP8EMU-pk-sub-u16        v_pk_sub_u16 v0, v1, v2
OK    FP8EMU-pk-min-f16        v_pk_min_f16 v0, v1, v2
OK    FP8EMU-pk-max-f16        v_pk_max_f16 v0, v1, v2
OK    FP8EMU-pack-b32-f16      v_pack_b32_f16 v0, v1, v2
OK    FP8EMU-perm-b32          v_perm_b32 v0, v1, v2, v3
OK    FP8EMU-lshl-or-b32       v_lshl_or_b32 v0, v1, v2, v3
OK    FP8EMU-or3-b32           v_or3_b32 v0, v1, v2, v3
OK    FP8EMU-alignbit-b32      v_alignbit_b32 v0, v1, v2, v3
OK    FP8EMU-bfe-u32           v_bfe_u32 v0, v1, v2, v3
OK    FP8EMU-rndne-f32         v_rndne_f32 v0, v1
OK    FP8EMU-cvt-pk-u8-f32     v_cvt_pk_u8_f32 v0, v1, v2, v3
OK    FP8EMU-sat-pk-u8-i16     v_sat_pk_u8_i16 v0, v1
OK    FP8EMU-ds-read-u16       ds_read_u16 v0, v1
FAIL  FP8EMU-ds-read-b16       error: invalid instruction
OK    FP8EMU-ds-bpermute       ds_bpermute_b32 v0, v1, v2
OK    FP8EMU-ds-permute        ds_permute_b32 v0, v1, v2
OK    FP8EMU-dpp-quad          v_mov_b32_dpp v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
OK    FP8EMU-sdwa-byte0        v_cvt_f32_i32_sdwa v0, v1 src0_sel:BYTE_0
FAIL  FP8EMU-cvt-pk-f32-fp8    error: instruction not supported on this GPU
FAIL  FP8EMU-cvt-pk-bf16-f32   error: invalid instruction
```

codegen（`codegen.tsv` 原文摘录）：

```
EMITTED pk_u16_add         v_pk_add_u16        x1    v_pk_add_u16 v2, v2, s4 op_sel_hi:[1,0]
EMITTED pk_u16_sub         v_pk_sub_u16        x1    v_pk_sub_u16 v2, v2, s4
EMITTED pk_u16_shl         v_pk_lshlrev_b16    x1    v_pk_lshlrev_b16 v2, 7, v2 op_sel_hi:[0,1]
EMITTED e5m2_to_f16_bits   v_pk_lshlrev_b16    x1    v_pk_lshlrev_b16 v2, 8, v2 op_sel_hi:[0,1]
EMITTED e4m3_to_f16_bits   v_pk_add_u16        x1    v_pk_add_u16 v3, v3, s5 op_sel_hi:[1,0]
EMITTED bf16_to_f16_bits   v_pk_sub_u16        x1    v_pk_sub_u16 v2, v2, s4
EMITTED bf16_to_f32_shift  v_lshlrev_b32       x1    v_lshlrev_b32_e32 v2, 16, v2
EMITTED f32_to_bf16_round  v_add_u32           x1    v_add_u32_e32 v2, 0x8000, v2
EMITTED dot2_after_convert v_dot2_f32_f16      x1    v_dot2_f32_f16 v2, v4, v4, v5
```

> ⚠️ **本步最重要的教训**：`codegen.tsv` 只证明"算子会被发射"，**不证明 C 源里的位运算语义正确**。
> 见 §5.1：`u16x2` 的 packed 移位被 DTK clang 编出了 lane 选择，端到端数值测试才是判据。

### 3.2 构建（DTK 26.04 / torch 2.11.0+das）

```bash
$ docker exec z100-vllm bash /mnt/game/port/build-211.sh _rocm_C
[2/4] Building HIP object CMakeFiles/_rocm_C.dir/csrc/rocm/bf16_emu_gemv.hip.o
[3/4] Building HIP object CMakeFiles/_rocm_C.dir/csrc/rocm/fp8_emu_gemm.hip.o
[4/4] Linking HIP shared module _rocm_C.abi3.so
----- ninja rc=0 2026-10-11T01:47:27+08:00 -----

$ md5sum /mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so
c7b825b8ded1a1f8aa179a951d9dd4c4
$ strings -a <so> | grep -oE "amdgcn-amd-amdhsa--[a-z0-9:+]+" | sort -u
amdgcn-amd-amdhsa--gfx906
```

**构建系统的已知坑（本步实测）**：改共享头文件 `csrc/rocm/gfx906_emu_common.cuh` 不会触发重编
（ninja 报 `no work to do`，且 `hipify_stage` 里的头文件副本不刷新）；必须 `touch` 相关 `.cu`
再构建。这是 S4d 增量性修复留下的依赖缺口，已在本步实测记录。

### 3.3 数值门禁（门禁 2）

```bash
$ HIP_VISIBLE_DEVICES=2 python tools/dcu/s11f_fp8_emu_probe.py --mode check --max-m 512
# checked torch.float8_e4m3fn
# checked torch.float8_e5m2
# check: 384/384 cases within tolerance (worst max|err|/max|ref|=4.175e-04, fp32_tol=0.001, abs_tol=0.001)
```

> ⚠️ **2026-10-11 复核更正**：探针的 `--rel-tol` 默认是 `1e-3`，归档命令没有覆盖它，
> 所以上面这行**并没有**验证"fp32 输出 < 1e-5"这个说法（原文这里把日志引成了
> `fp32_tol=1e-05`，与归档日志不符）。显式收紧后重跑（当前产物 md5 `56fefa7c…`）：
>
> ```bash
> $ python tools/dcu/s11f_fp8_emu_probe.py --mode check --max-m 512 --rel-tol 1e-5 --abs-tol 1e-3
> # check: 384/384 cases within tolerance (worst max|err|/max|ref|=4.173e-04, fp32_tol=1e-05, abs_tol=0.001)
> ```
>
> 384/384 仍通过，最坏误差全部来自 fp16 输出的末位舍入，结论成立 —— 但必须显式传
> `--rel-tol 1e-5` 才算测过。

覆盖：e4m3fn + e5m2 × 6 个 27B 形状 × M ∈ {1,2,4,8,16,64,128,512} ×
per-token/per-tensor 激活 scale × fp16/fp32 输出。

```bash
$ cd /mnt/game/port/s11f/t && python -m pytest test_gfx906_bf16_gemv.py test_gfx906_fp8_emu.py -q
28 passed, 17 warnings in 20.42s
```

**回归对照**：`tests/kernels/core/test_rocm_misc_ops.py` 有 **4 项在本机失败**，经核对是
**本步之前就存在**的问题：`ROCmFP8ScaledMMLinearKernel.is_supported()` 在 gfx906 上先命中
"wvSplitKQ (skinny GEMM) is not built on gfx906 (no MFMA)" 早退（该早退由适配提交
`3f1304f0d` 引入，本步未改动 `scaled_mm/rocm.py`，`git diff --name-only HEAD | grep rocm.py` = 0），
而上游测试期望的是"VLLM_ROCM_USE_SKINNY_GEMM 未开"的理由串。其余 25 项通过。

### 3.4 性能（卡 2，`HIP_VISIBLE_DEVICES=2`）

FP8（K=5120；对照 fp16 rocBLAS；FP8 本身在 gfx906 上没有可用基线）：

```
RESULT M=1  N=17408 K=5120  emu=0.220 ms (404.8 GB/s) fp16_rocblas=0.826 ms speedup=3.75x
RESULT M=1  N=8192  K=5120  emu=0.105 ms (399.6 GB/s) fp16_rocblas=0.573 ms speedup=5.46x
RESULT M=1  N=7168  K=5120  emu=0.117 ms (312.7 GB/s) fp16_rocblas=0.496 ms speedup=4.22x
RESULT M=1  N=5120  K=5120  emu=0.134 ms (196.2 GB/s) fp16_rocblas=0.272 ms speedup=2.03x
RESULT M=1  N=512   K=5120  emu=0.032 ms              fp16_rocblas=0.080 ms speedup=2.54x
RESULT M=1  N=48    K=5120  emu=0.016 ms              fp16_rocblas=0.025 ms speedup=1.59x
RESULT M=8  N=7168  K=5120  emu=0.491 ms  speedup=1.39x
RESULT M=512 N=5120 K=5120  emu=3.062 ms  speedup=0.84x
RESULT M=64  N=48   K=5120  emu=0.370 ms  speedup=0.07x   <- 未覆盖：小 N 大 M 的 tile 利用率
```

dense fp16/bf16 GEMV（K=5120，对照 rocBLAS；误差列 = 输出 dtype 的 ulp）：

```
RESULT    fp16 M=1 N=17408 emu=0.3008 ms (592.6 GB/s) rocblas=0.8253 speedup=2.74x err=0.35 ulp
RESULT    fp16 M=1 N=8192  emu=0.1444 ms (580.8 GB/s) rocblas=0.5703 speedup=3.95x err=0.39 ulp
RESULT    fp16 M=1 N=5120  emu=0.1292 ms (405.8 GB/s) rocblas=0.2732 speedup=2.11x err=0.19 ulp
RESULT    fp16 M=1 N=512   emu=0.0310 ms               rocblas=0.0463 speedup=1.49x err=0.00 ulp
RESULT    fp16 M=1 N=48    emu=0.0179 ms               rocblas=0.0231 speedup=1.29x err=0.00 ulp
RESULT  bf16   M=1 N=17408 emu=0.2853 ms (624.7 GB/s) rocblas=0.7443 speedup=2.61x err=0.02 ulp
RESULT  bf16   M=1 N=8192  emu=0.1516 ms (553.5 GB/s) rocblas=0.3788 speedup=2.50x err=0.00 ulp
RESULT  bf16   M=1 N=5120  emu=0.1408 ms (372.2 GB/s) rocblas=0.2825 speedup=2.01x err=0.18 ulp
RESULT  bf16   M=1 N=512   emu=0.0368 ms               rocblas=0.2099 speedup=5.71x err=0.00 ulp
RESULT  bf16   M=1 N=48    emu=0.0181 ms               rocblas=0.0446 speedup=2.46x err=0.00 ulp
RESULT  bf16   M=4 N=512   emu=0.0982 ms               rocblas=0.2102 speedup=2.14x err=0.00 ulp
RESULT  bf16   M=8 N=17408 emu=1.5476 ms               rocblas=0.7847 speedup=0.51x  <- 未启用（见下）
# worst error over all cases: 0.39 ulp of the output dtype
```

基线（`s11f_baseline_gemm_probe.py`，同一台机的 rocBLAS）：fp16/bf16 decode GEMV 只有
215–235 GB/s（fp32 因字节多反而到 384 GB/s）；bf16 小 N 更差（N=512：24 GB/s）。

**Python 侧只启用实测获胜的形状**（`_gfx906_gemv_shape_ok`）：单 token（全部 N，
M=1 同时覆盖 fp16/bf16），以及 `bf16 + M=4 + 128 ≤ N ≤ 1024`。
M ≥ 8 与 fp16 小 N 一律回落 rocBLAS —— **没有引入任何已知回退形状**。

### 3.5 机器码门禁（门禁 3）

`.hip_fatbin` 拆包（4 个 bundle → `clang-offload-bundler` → `llvm-objdump`，
方法同 S11d §3.12），最终件 76 个 kernel：

```
s_endpgm             76
v_dot2_f32_f16       18666     <- 两条内核的 MAC 核心
ds_read_u16          1092      <- fp8 → fp16 的 256 项 LDS LUT
v_fma_f32            24        <- bf16 fp32 路线的 FMA
v_fmac_f32           28        <- 全部在 bf16_emu_gemv_f32_kernel（见下）
v_mfma               0
v_cvt_f32_bf16       0
v_dot2_f32_bf16      0
v_cvt_f32_fp8        0
```

> ⚠️ **2026-10-11 复核更正**：归档原文把 `v_fmac_f32` 记成 **0**，那是 **grep 口径错误** ——
> 反汇编里的助记符是 `v_fmac_f32_e32`，用 `\bv_fmac_f32\b` 匹配不到（`_` 是 word 字符）。
> 用归档留下的 `emu_gfx906_final2.dis` 复算：plain grep = **28**、word-boundary grep = 0，
> 与当前产物（md5 `56fefa7c…`）逐项相同。
>
> 按 kernel 归因后的正确口径：**28 条全部位于 `bf16_emu_gemv_f32_kernel`**，也就是设计
> 文档 §3.3 写明要走 `v_fma_f32`/`v_fmac_f32` 的 bf16 **f32 路线**，**设计上就该有**；
> `fp8_emu_*` 内核（LUT + dot2 路线）的 `v_fmac_f32`/`v_fma_f32` 均为 **0**。
> ⇒ 机器码门禁**必须按 kernel 归因**，全局计数会被 `_e32` 后缀坑。

## 4. 判定

| 门禁 | 判据 | 结果 | 证据 |
|---|---|---|---|
| 1 ISA 基线 | STRICT=1 退出 0、53/53 MATCH、SYNTAX=0 | ✅ | §3.1 |
| 2 数值/单测 | 新增 28 passed；FP8 sweep 384/384 | ✅ | §3.2/§3.3 |
| 3 机器码 | dot2 18,666 / LUT 1,092 / `v_fmac_f32` 0 | ✅ | §3.5 |
| 4 默认行为 | 两个 env 默认 False；`is_supported` 首查 rocm+gfx906 | ✅ | 源码 + 单测 `test_kernel_gating_follows_env` |
| 5 FP8 可用性 | 从"全体拒绝"变为"可选内核"（per-tensor / per-channel / per-token） | ✅ | `can_implement` 单测 + 选核表 |
| 6 性能 | FP8 decode 1.6–5.5×（对照 fp16 rocBLAS）；bf16/fp16 decode 1.3–5.7×（590–625 GB/s vs 215–235 GB/s） | ✅（内核级） | §3.4 |
| 7 端到端 | 起服务 + FP8 模型 | ❌ **未做** | §7 |

## 5. 过程中的发现（对后续步骤有价值）

### 5.1 DTK clang 的 packed 16 位路径有两个真实陷阱

1. **`u16x2` 向量位运算被加 `op_sel_hi` lane 选择**：`(s - c) << 3` 形式的 bf16→fp16 重定标，
   clang 15 与 18 都生成 `v_pk_lshlrev_b16 … op_sel_hi:[0,1]`，低 lane 实际拿到未减偏置的值
   （实测 2^-14 → **4.0**）。同一段 C 在 `conv.c` 里编译出的指令序列看起来"正确"，
   只有端到端数值测试能发现。
2. **`V_PK_LSHLREV_B16` 的移位量是按 lane 从 src0 两个半字取的**：改用 inline asm 并给标量
   立即数 `3` 时，只有一个 lane 被移动（实测 2^-14 → **64.0**）。

处置：最终用 **32 位标量逐 lane** 实现快路径（无 lane 选择歧义），并用
`test_bf16_fast_rebase_is_accurate_for_range_checked_weights`（含 `fast == guarded` 的逐位断言）锁住。
FP8 侧不受影响（走 LUT，不做 packed 位运算）。

### 5.2 FP8 的 LUT 路线是"精确"的，位运算路线不是

e4m3 的次正规（|x| < 2^-6）与 fp16 的次正规指数不同，无法用移位+偏置一步映射；
而 256 项 fp16 表对 **e4m3fn 与 e5m2 的全部有限码都无损**（已由
`test_lut_is_lossless_for_every_code` 对全部 256 码逐条断言，含 inf/NaN 与符号）。
因此 fp8 侧的默认路线固定为 LUT（`VLLM_DCU_FP8_CVT=lut`），位运算只留作 A/B。

### 5.3 M=8 的 GEMV 是 LDS 受限，不是带宽受限

M=8 时每个 block 对同一个 A 行要取 2 dword/次，A 的 LDS 访存与 dot2 条数接近 1:1，
内核反而落后 rocBLAS（0.5–0.85×）。M=1（decode 主战场）则稳定在 590–625 GB/s。
把 A 加宽提出循环（M ≥ 2 时多一个 widen kernel）只带来轻微改善 —— 说明瓶颈在
**A 的 LDS 取数**，下一步应改 `uint4` 宽 LDS 读 + 提高 TM/TN。

## 6. 未做（明确登记）

1. **端到端**：未起服务跑 FP8 模型（需要 FP8 权重与足够显存）；未用
   `VLLM_DCU_USE_BF16_GEMV=1` 跑一次真实模型 decode 对比。
2. **block-scaled FP8**：未做（仍走 36× 慢的 Triton）。
3. **小 N 大 M 的 GEMM**：未做 split-K（M=64/N=48 只有 0.07×，因 CTA 数 = 1–2）。
4. **bf16 快路径（`f16` 重定标）**：未在 Python 侧开放（契约要求逐 lane 在窗口内，零值会回绕）。
5. **MTP/GDN 等 fp16-only 派发**（`VLLM_DISPATCH_HALF_TYPES`）未扩展 bf16。
6. **共享头文件的构建依赖**：未修（需要 `touch .cu`；属构建系统遗留）。

## 7. 结论

- Z100 上 **FP8 从"完全不可用"变为"有可选 HIP 内核"**：e4m3fn/e5m2、per-tensor/per-channel/
  per-token、fp16/bf16/fp32 输出，误差仅来自 fp32 累加序与输出舍入（fp32 输出 < 1e-5）。
- **decode（M=1）FP8 比 fp16 rocBLAS 快 1.6–5.5×**，读的是 fp8 字节（17408×5120：
  0.220 ms @ 405 GB/s，即等效 fp16 流量 ~810 GB/s），接近本卡 HBM 上限。
- **dense bf16/fp16 decode GEMV 比 rocBLAS 快 1.3–5.7×**（590–625 GB/s vs 215–235 GB/s），
  误差 ≤ 0.39 ulp；bf16 小 N 收益最大（N=512：5.71×）。
- 两条内核的机器码里 **per-tensor `fp8_emu_*` 内核 `v_fmac_f32` = 0**（`bf16_emu_gemv_f32`
  的 f32 路线有 28 条、设计如此；口径更正见 §8.3）。
- 全部开关默认 OFF；关闭即回到原行为（Triton / rocBLAS）。

---

## 8. 复核与修复（2026-10-11）

本节记录在**换到当前产物**（`56fefa7c…`，S11e2 重建件）上重跑全部门禁时发现的 4 处问题及修复。
所有结论都带原始命令与实测数字；产物 md5 逐次登记。

### 8.1 bf16/fp16 dense GEMV 派发不可达（真 bug，已修）

**症状**：`VLLM_DCU_USE_BF16_GEMV=1` 时 `torch.ops.vllm.bf16_emu_mm` 从未被调用。

**根因**：`vllm/platforms/rocm.py` 里
`_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])`
—— 这是上游语义（CDNA2/CDNA3），**gfx906 不在其中**，所以 `on_gfx9()` 在 Z100 上恒为
`False`。而 `rocm_unquantized_gemm_impl` 里 `n <= 16 and not on_gfx9()` 的 Triton 分支
排在 GEMV 分支**之前**，于是把该内核支持的**全部**形状（`_gfx906_gemv_shape_ok` 只在
`n ∈ {1, 4}` 为 True）都截走了 ⇒ 原注释"这个目标是 gfx9，所以 Triton 快路被跳过"是**反的**。

**证据**（monkeypatch 计数，卡 1）：

```
on_gfx9() = False
  M=1 -> triton_matmul calls=1, bf16_emu_mm calls=0
  M=4 -> triton_matmul calls=1, bf16_emu_mm calls=0
  [on_gfx9 forced True] M=1 -> triton=0 emu=1
```

**修复**：把 GEMV 分支提到 Triton 分支之前，并加 `on_gfx906()` 守卫
（`vllm/model_executor/layers/utils.py`）；`M=8` 等未覆盖形状的行为**不变**（仍走 Triton）。
回归测试：`tests/kernels/test_gfx906_bf16_gemv.py` 新增
`test_dense_gemm_dispatch_reaches_the_emu_kernel` 与 `test_dense_gemm_dispatch_is_off_by_default`。

**收益**（独占卡，warmup=20 / iters=100，K=5120；**基线换成引擎真正在用的 `triton_matmul`**，
归档里的 `F.linear`=rocBLAS 不是生产口径）：

```
M=1 N=17408 bf16 emu=0.2712ms(657GB/s) triton=0.9081ms(196GB/s) rocBLAS=0.6985ms(255GB/s) | emu/triton=3.35x
M=1 N=8192  bf16 emu=0.1445ms(580GB/s) triton=0.5828ms(144GB/s) rocBLAS=0.3306ms(254GB/s) | emu/triton=4.03x
M=1 N=5120  bf16 emu=0.0905ms(579GB/s) triton=0.2953ms(178GB/s) rocBLAS=0.2474ms(212GB/s) | emu/triton=3.26x
M=1 N=512   bf16 emu=0.0252ms(208GB/s) triton=0.2154ms( 24GB/s) rocBLAS=0.1732ms( 30GB/s) | emu/triton=8.54x
M=4 N=512   bf16 emu=0.0902ms( 58GB/s) triton=0.2357ms( 22GB/s) rocBLAS=0.1735ms( 30GB/s) | emu/triton=2.61x
M=1 N=17408 fp16 emu=0.2663ms(669GB/s) triton=0.6082ms(293GB/s) rocBLAS=0.6871ms(259GB/s) | emu/triton=2.28x
M=1 N=5120  fp16 emu=0.0905ms(579GB/s) triton=0.2312ms(227GB/s) rocBLAS=0.2311ms(227GB/s) | emu/triton=2.55x
```

### 8.2 ISA 探针 2 行基线过期（已修）

`ISA_PROBE_STRICT=1 bash tools/dcu/isa_probe.sh` 报：

```
BASELINE-MISMATCH cgo  pk_u16_sub          want=EMITTED got=MISSING
BASELINE-MISMATCH cgo  bf16_to_f16_bits    want=EMITTED got=MISSING
```

**根因**：DTK clang 18 把 `u16x2 - {0xc000,0xc000}` 规范化成
`v_pk_add_u16 x, x, 2.0`（0x4000 是 0xc000 的补码增量），因此 `v_pk_sub_u16` **不再被发射**；
单 TU 复现一致（`one.c` → `v_pk_add_u16 v2, v2, 2.0`）。这两行本来就是设计 §3.2 里被放弃的
packed 路径（真实内核用 32 位标量逐 lane），且当前产物里 `v_pk_sub_u16` = 0 条。

**修复**：`pk_u16_sub` 改钉 `v_pk_add_u16`、`bf16_to_f16_bits` 改钉 `v_pk_lshlrev_b16`
（即带 `op_sel_hi` lane 陷阱的那条指令，正是该行要记录的东西），并在表头注释里写明原因。
注意**regex 字段不能含 `|`**（那是表的列分隔符，第一次修复就是这样踩成 ERROR 的）。

修复后：`ISA_PROBE_STRICT=1` → **REAL_EXIT=0, baseline MATCH (all 53 expectations hold)**，
`codegen EMITTED=29 MISSING=0 ERROR=0`（在 block-scaled 内核落地后复跑同样 53/53）。

### 8.3 归档 `v_fmac_f32 = 0` 是 grep 口径假象（已更正）

反汇编里的助记符是 `v_fmac_f32_e32`，`\bv_fmac_f32\b` 匹配不到（`_` 是 word 字符）。
拿**归档自己的**反汇编文件复算：

```
emu_gfx906_final2.dis:  plain grep v_fmac_f32 = 28   word-boundary grep = 0
                        s_endpgm=76 dot2=18666 lut=1092 fma=24
```

28 条**全部**位于 `bf16_emu_gemv_f32_kernel` —— 即设计 §3.3 写明要走
`v_fma_f32`/`v_fmac_f32` 的 bf16 **f32 路线**，设计上就该有；`fp8_emu_*`（per-tensor）
内核 `v_fmac_f32`/`v_fma_f32` 均为 0。⇒ 机器码门禁**必须按 kernel 归因**。

### 8.4 数值 sweep 的默认门限比文档松 100 倍（已更正）

`tools/dcu/s11f_fp8_emu_probe.py` 的 `--rel-tol` 默认 `1e-3`；归档命令没覆盖它
（归档日志里 `fp32_tol=0.001`），所以原文"fp32 输出用 1e-5 门限"**当时并未被验证**。
显式收紧后重跑（产物 `56fefa7c…`）：

```
$ python tools/dcu/s11f_fp8_emu_probe.py --mode check --max-m 512 --rel-tol 1e-5 --abs-tol 1e-3
# check: 384/384 cases within tolerance (worst max|err|/max|ref|=4.173e-04, fp32_tol=1e-05, abs_tol=1.001e-03)
```

### 8.5 构建系统的硬链接坑（新发现，已规避）

工作区 `/src`（宿主 `/home/lyh/z100-vllm`）与构建树 `/mnt/game/port/gfx906-vllm-211`
是**硬链接**关系。用"写新文件 + rename"的编辑器改源码会**断开硬链接**，构建树仍是旧内容 ⇒
`cmake --build` 报 rc=0 但产物里没有新符号（本次实测：`nm -D | grep fp8_emu_gemv_block` = 0）。
规避：改完 `cp -f` 同步被改的 `csrc/**` 到 `${REPO_DIR}`，再 `build-211.sh _rocm_C` +
`install-211.sh _rocm_C`。注意 **install 会做 RPATH 后处理，install 件 md5 与
`build-211/temp/` 里的中间件不同**（本次 `54e75087…` → `9818d8a2…`）。

---

## 9. block-scaled（128×128）FP8 支持（新增，2026-10-11）

### 9.1 动机

端到端选用的 `Qwen/Qwen3.8-27B-FP8`（ModelScope）是 **block-scaled** 的：
`quantization_config = {activation_scheme: dynamic, fmt: e4m3, quant_method: fp8,
weight_block_size: [128, 128]}`，407 个 scale 全部是 `weight_scale_inv`
（`[cdiv(N,128), cdiv(K,128)]`），没有任何 per-tensor `weight_scale`。
per-tensor 内核 `can_implement` 按设计拒绝它 ⇒ 回落到 `TritonFp8BlockScaledMMKernel`
（标量 `v_fmac_f32` 模拟）。实测该回退路：模型能起、输出连贯，但
**TTFT 4969 ms / decode 1984.8 ms/token**，服务不可用。

### 9.2 实现

契约（与 vLLM 的 `Fp8LinearMethod` block 路径一致）：

```
out[m, n] = sum_kb ( sum_{k in kb} a[m,k] * w[n,k] ) * a_s[m, kb] * w_s[n // block_n, kb]
```

关键点：**标量乘法可以分配到线程局部部分和上再归约**
（`Σ_t Σ_kb p[t,kb]·s[kb] == Σ_kb s[kb]·Σ_t p[t,kb]`），所以只在**当前 K 块的累加器**上乘
scale，就保住了原有的"一次归约树 / 一个寄存器 tile"结构：

| 位置 | 改动 |
|---|---|
| `csrc/rocm/fp8_emu_gemm.cu` | `fp8_emu_gemv_block_kernel`（每线程独占整个 K 块：`kb = t, t+nt, ...`，一个 cache line/块，scale 每 128 元素只取一次）；`fp8_emu_gemm_block_kernel`（在 128-K 边界把 `blk` 折叠进 `acc`）+ `fp8_gemm_fold_block_scale`；`fp8_gemm_load_block`（**支持 W 行 stride**，见下）；`check_block_common` / `run_block` / 4 个入口 |
| `csrc/rocm/fp8_emu_gemm.h`、`torch_bindings.cpp` | `fp8_emu_gemv_block` / `fp8_emu_gemm_block`（+ Meta）注册 |
| `vllm/.../scaled_mm/gfx906_fp8.py` | `Gfx906Fp8BlockScaledMMLinearKernel`（`is_supported`/`can_implement`/`apply_block_scaled_mm`）+ `torch.library.custom_op("vllm::fp8_emu_mm_block")`（GEMV/GEMM 分支仍在 opaque body 内，S11k 教训） |
| `vllm/model_executor/kernels/linear/__init__.py` | 插到 `_POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.ROCM]` **首位** |

两个实现细节值得记：

1. **W 的行 stride 必须支持**：block 路径会走
   `process_fp8_weight_block_strategy` → `_maybe_pad_fp8_weight`（`VLLM_ROCM_FP8_PADDING`
   默认开、K=5120 满足 `stride*1 % 512 == 0`），权重变成 `stride(0)=K+256` 的**非连续**
   视图。内核按 `stride(0)` 索引行（单测 `test_block_accepts_padded_weight_row_stride`
   断言填充视图与连续权重**逐位一致**）。
2. **权重方向自适应**：`apply_block_scaled_mm` 按 `B.shape[1] == K` 判定 `[N,K]` / `[K,N]`，
   两种都零拷贝，`stride(1) != 1` 时明确报错而不是偷偷 copy。

### 9.3 门禁

| 门禁 | 命令 | 结果 |
|---|---|---|
| 单测 | `pytest -q --noconftest tests/kernels/quantization/test_gfx906_fp8_emu.py tests/kernels/test_gfx906_bf16_gemv.py` | **41 passed**（新增 11 项 block：GEMV/GEMM vs fp32 反量化参考、奇数形状（N=100<block_n、M<BM）、填充 stride 逐位一致、bias、门控/`can_implement` 矩阵） |
| ISA | `ISA_PROBE_STRICT=1 bash tools/dcu/isa_probe.sh` | **53/53 MATCH**，退出 0，`SYNTAX=0` |
| 选核 | `choose_scaled_mm_linear_kernel(block_cfg, _POSSIBLE_FP8_BLOCK_KERNELS, quantization="fp8_block_w8a8")` | ON → `Gfx906Fp8BlockScaledMMLinearKernel`；OFF → `TritonFp8BlockScaledMMKernel`（原行为） |
| 机器码（拆 `.hip_fatbin` + `llvm-objdump`，按 kernel 归因） | 见 §8.3 脚本 | 106 kernel；`v_dot2_f32_f16` **37254**、`ds_read_u16` **2184**、`v_mfma` **0**、`v_cvt_f32_fp8` **0**；`v_fmac_f32` **1258** |

`v_fmac_f32` 1258 条**全部**在新增的 block 内核里，且是 **scale 折叠的 FMA**，不是标量 MAC 模拟：
`fp8_emu_gemv_block`（18 kernel）静态 dot2:fmac = 2:1，但那条 fmac 在 K 循环**之外**
（动态约 1:64）；`fp8_emu_gemm_block`（12 kernel）静态 16:1、动态约 1:32；
per-tensor 内核 `v_fmac_f32` 仍为 **0**（逐项未变）。

### 9.4 端到端（Qwen3.8-27B-FP8，TP=4）

模型：ModelScope `Qwen/Qwen3.8-27B-FP8`，**82 文件 / 30.89 GB 与 API 清单逐项尺寸一致**
（`_maybe_pad_fp8_weight` 之外无改动），架构 `Qwen3_5ForConditionalGeneration`
（与生产 INT8 版同架构，64 层 / hidden 5120）。

配方：`tools/dcu/s11f_e2e_fp8.sh <MODEL> 4 <PORT>`（`--dtype float16`、TP=4、
util 0.85、`--disable-custom-all-reduce --skip-mm-profiling --enable-prefix-caching
--mamba-cache-mode align`、maxlen 8192、`--enforce-eager`、`VLLM_DCU_USE_FP8_EMU=1`）。

**选核证据**（两条臂同配方，只差开关）：

```
臂 B（新内核）: INFO [__init__.py:743] Selected Gfx906Fp8BlockScaledMMLinearKernel for Fp8LinearMethod
臂 A（对照）  : INFO [__init__.py:743] Selected TritonFp8BlockScaledMMKernel for Fp8LinearMethod
```

**性能**（同一台机、同一模型、同一配方，4 卡独占，流式计时）：

| 臂 | 内核 | TTFT | decode | 相对 |
|---|---|---|---|---|
| A（对照） | `TritonFp8BlockScaledMMKernel` | 4969 ms | **1984.8 ms/token**（0.50 tok/s） | 1× |
| B（本步） | `Gfx906Fp8BlockScaledMMLinearKernel` | 600 ms | **123.0 ms/token**（8.1 tok/s） | **16.1×** |

同一请求口径（60 prompt / 32 completion，非流式）：85.0 s → 5.38 s。
参考：生产 27B **INT8** 在 eager / TP=4 下是 126.83 ms/token（S14）⇒ FP8 引擎已与生产 INT8
**同档**，剩余成本在 eager 模式与 GDN/attention，不在 FP8 GEMM。

**质量**：同 4 条提示词 × 32 greedy token，两臂 token 流比较

```
zh-fact   32 tokens  逐位一致        max|Δlogprob| 0.317
en-reason 32 tokens  逐位一致        max|Δlogprob| 0.066
zh-math   32 tokens  第 29 位分叉    max|Δlogprob| 0.189
en-code   32 tokens  第 31 位分叉    max|Δlogprob| 0.142
```

2/4 逐位一致；另 2 条只在**最后 1–3 个 token**、且 `Δlogprob ≤ 0.32 nat` 处不同，
与该机型自身噪声底（S14b 实测同配置重复/模式差 0.1–0.6 nat）同量级；4 条输出全部连贯
（含 60%×45 的算术与 120 km/1.5 h 的单位换算推理）。算子级误差另有更硬的门禁：对 fp32
反量化参考 < 1e-5（fp32 输出）。

### 9.5 本步未做

1. **graph 模式**：端到端只跑了 `--enforce-eager`（生产配方），未跑 CUDA graph/compile。
2. **UE8M0 / MXFP8 / 非 128 的 block 形状**：内核通用（`block_n`/`block_k` 运行期参数，
   `block_k % 32 == 0`），但只测了 128×128。
3. **MTP / 视觉塔 / 长上下文 / TP=2**：未测；`--skip-mm-profiling` + 纯文本请求下未触发视觉塔。
4. **block GEMM 的 tile 调优**：沿用 per-tensor 的四组 cfg，未针对 block 路径重新 tune。

---

## 10. 上线为生产槽位（2026-10-11）

把 FP8 模型作为生产槽位的 drop-in 起在 **:8000**（`api-gateway` 的 upstream），
服务名沿用 `qwen3.8-27b`，配方与 `/home/lyh/server/serve_27b_tp4.sh` 一致
（TP=4 / util 0.85 / maxlen 262144 / MBT 2048 / prefix caching / mamba align / eager），
启动脚本 [`/mnt/game/port/serve_fp8_8000.sh`]。回滚 INT8：
`C=z100-vllm bash /home/lyh/server/serve_27b_tp4.sh`。

实测链路（宿主侧发起）：

```
POST http://127.0.0.1:8000/v1/chat/completions   -> 307 tok / 39.07 s (~127 ms/token)
POST http://127.0.0.1:8080/v1/chat/completions   -> gateway 鉴权通过，答 "49"（7²）3.91 s
启动日志: Selected Gfx906Fp8BlockScaledMMLinearKernel for Fp8LinearMethod
          Available KV cache memory: 18.55 GiB / 1,192,990 tokens / 262,144 上下文可并发 4.55x
```

### 10.1 两个上线坑（都踩过）

1. **必须带 `--host 0.0.0.0`**：容器是 bridge 网络，宿主靠 docker 端口映射
   （`8000/tcp -> 0.0.0.0:8000`）访问。只绑 `127.0.0.1` 时容器内自测正常、宿主却
   得到 `connection reset by peer`（docker-proxy 连不上回环）。生产原命令行正是
   `--host 127.0.0.1 --port 8000 --host 0.0.0.0`（后者生效）。
2. **pinned 内存是启动瓶颈**：`vllm/v1/worker/gpu/states.py` 的
   `all_token_ids = StagedWriteTensor((max_num_reqs, max_model_len), int32,
   uva_instead_of_gpu=True)` 在**每个 worker** 上分配 `max_num_reqs × max_model_len × 4 B`
   的 **pinned 主机内存**。本机宿主只有 31 GB RAM、容器 `ulimit -l`（memlock）只有 64 MB，
   maxlen 262144 时该缓冲是 GB 级 ⇒ 启动处在临界点：实测有一次
   `load_model()` 里 `UvaBuffer.__init__` 抛 `hipErrorOutOfMemory`（同参数重试即成功，
   宿主 `MemAvailable` 峰值只剩 ~12.8 GB）。**排障口诀**：日志里 OOM 落在
   `buffer_utils.py` / `pin_memory=True` 就不是显存不足，重试或降 `--max-model-len`。
