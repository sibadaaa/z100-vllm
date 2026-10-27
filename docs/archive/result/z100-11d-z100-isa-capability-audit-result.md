# S11d Z100 原生算力审计 + 算子复用清点 + Triton gfx906 codegen 检查 — 执行结果

> 归档时间：2026-10-08
> 执行者：子 agent（s11d-isa-audit）
> 对应设计文档：`docs/海光Z100-性能优化方案.md` §4A（本步清单）、§4（P2）、§4B（P6）
> 前一档：无（本步为首档；相关背景档 `docs/hygon_z100.md` §2.1、`docs/result/z100-06-triton-gfx906-result.md`、`docs/result/z100-08b-gating-and-requirements-result.md`）
> 状态：✅ 完成（4 项交付全达成；1 项受限，见 §5）
>
> **本步不改 `csrc/**`、不改 `docs/design/*`、不改 `docs/z100-todolist.md`、不改 `docs/result/README.md`。**
> 新增文件：`tools/dcu/isa_probe.sh`、`tools/dcu/triton_codegen_probe.py`、本归档。

---

## 1. 目标与验收标准

任务书门禁（原文要点）：

1. 对候选指令/内建给出四列表：**汇编器接受性 / `__has_builtin` 可用性 / 实测 codegen（真的发出来了）/ 树内现有复用点（`file:line`）**；
2. 给出**两个明确建议**：(a) W8A8 INT8 GEMM 用哪条指令与什么形态；(b) FP8→FP16 模拟 GEMM 用哪条指令（或 Triton 路线）；
3. 交付 `tools/dcu/isa_probe.sh`（汇编器/内建/codegen 三合一、带退出码）与 `tools/dcu/triton_codegen_probe.py`；
4. 硬约束：禁止运行独立自编译 HIP 可执行程序（只允许 `hipcc -c` / `clang -S` / 反汇编 / python 派发）；不动生产容器与 `z100-build` 构建树；GPU 只用 2、3 号卡。

本次执行的验收标准（自设，全部达成）：

| # | 验收项 | 判据 |
|---|---|---|
| A1 | 汇编器接受性 | 43 条候选指令在 **DTK 24.04.3/clang 15（宿主）** 与 **DTK 26.04/clang 18（容器 `z100-build-211`）** 两套工具链上逐条实测，差异注明 |
| A2 | 内建可用性 | 29 个 `__builtin_amdgcn_*` 以 `__has_builtin` 在 amdgcn/gfx906 目标下探测；并用 `#warning` 在 **HIP 设备 pass** 中复核关键项 |
| A3 | 实测 codegen | 20 个 device 片段，`clang -target amdgcn-amd-amdhsa -mcpu=gfx906 -O3 -S` 后 grep 目标指令；另用 `hipcc -S --cuda-device-only --offload-arch=gfx906` 复核 4 条 |
| A4 | Triton codegen | 对 `tl.dot` fp16 / int8 / fp8-to-fp16 / fp8×fp8、`tl.sum`、逐元素 fp16，在 gfx906（GPU 2,3）编译并读 `asm["amdgcn"]`，统计指令计数 |
| A5 | 复用清点 | 树内所有相关点位列成表并判定 gfx906 是否启用；对已构建的 canonical `_C.abi3.so` 做**机器码级**复核；回答"W8A8 GEMM 能否直接复用" |
| A6 | 两个明确建议 | §6 给出结论 + 对 P2/P6 的影响 |

---

## 2. 实际改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `tools/dcu/isa_probe.sh` | 新增（427 行） | 三合一探针：汇编器接受性 + `__has_builtin` + codegen（含 inline-asm 通道与 hipcc 通道），输出 3 个 TSV，附 gfx906 基线断言（`ISA_PROBE_STRICT=1` 时不一致即 `exit 1`） |
| `tools/dcu/triton_codegen_probe.py` | 新增（300 行） | Triton gfx906 codegen 探针：6 组 kernel（fp16 `tl.dot` 形状/num_warps 扫描、m=1 decode 形状、int8 `tl.dot`、`tl.sum`、逐元素 fp16、fp8→fp16→`tl.dot`、fp8×fp8 对照），统计 `v_*` 指令计数 |
| `docs/result/z100-11d-z100-isa-capability-audit-result.md` | 新增（本文件） | 归档 |
| `csrc/**` | — | **无改动**（本步只审计） |

> 说明：探针把候选表内嵌在脚本里（`INSN_TABLE` / `BUILTIN_LIST` / `CG_TABLE` / `BASELINE`），后续要加指令只改表。

---

## 3. 执行过程与原始证据

### 3.0 环境（两套工具链，都测）

```bash
# 宿主（DTK 24.04.3）
/opt/dtk/llvm/bin/clang --version        # clang version 15.0.0
/opt/dtk/bin/hipcc --version             # HIP version: 24.04.0-0 / clang 15.0.0
# 容器 z100-build-211（DTK 26.04，任务书指定用它测 clang 18）
docker exec z100-build-211 bash -lc '/opt/dtk/llvm/bin/clang --version'
#   clang version 18.0.0 (/buildspace/das-build/opt-dev-6.3.3/llvm-project/clang c52cd56477d80cb648f8d8dbcfd2210f04610de8)
docker exec z100-build-211 bash -lc 'hipcc --version'   # HIP version: 6.2.0-0 / clang 18.0.0
docker exec z100-build-211 bash -lc 'python3 -c "import torch,triton;print(torch.__version__,triton.__version__)"'
#   2.11.0 3.6.0          （PYTHONPATH=/mnt/game/triton-gfx906/install；自编 triton 已打 gfx906 补丁）
```

**结论：除 2 条失败指令的错误措辞外，两套工具链的结论逐字节相同**（见 §3.5 的 diff）。

### 3.1 汇编器接受性（门禁一）

命令（探针内每条指令逐条执行，等价手写）：

```bash
CLANG=/opt/dtk/llvm/bin/clang
printf '\tv_dot4_i32_i8 v0, v1, v2, v3\n\ts_endpgm\n' > /tmp/a.s
$CLANG -target amdgcn-amd-amdhsa -mcpu=gfx906 -c /tmp/a.s -o /tmp/a.o   # 空输出 = 接受
$CLANG_DIR/llvm-objdump -d --mcpu=gfx906 /tmp/a.o                        # 取编码
```

两套工具链的完整结果（`OK` = 接受；`FAIL` = 汇编器拒绝；编码列取 `llvm-objdump`）：

| 类别 | 测试指令 | gfx906 汇编器 | 编码（clang15 / clang18 一致） |
|---|---|---|---|
| INT8 dot4 | `v_dot4_i32_i8 v0, v1, v2, v3` | ✅ OK | `D3A84000 1C0E0501` |
| INT8 dot4 | `v_dot4_u32_u8 v0, v1, v2, v3` | ✅ OK | `D3A94000 1C0E0501` |
| INT16 dot2 | `v_dot2_i32_i16 v0, v1, v2, v3` | ✅ OK | `D3A64000 1C0E0501` |
| INT16 dot2 | `v_dot2_u32_u16 v0, v1, v2, v3` | ✅ OK | `D3A74000 1C0E0501` |
| INT4 dot8 | `v_dot8_i32_i4 v0, v1, v2, v3` | ✅ OK | `D3AA4000 1C0E0501` |
| INT4 dot8 | `v_dot8_u32_u4 v0, v1, v2, v3` | ✅ OK | `D3AB4000 1C0E0501` |
| INT8 dot4c | `v_dot4c_i32_i8 v0, v1, v2` | ❌ FAIL `instruction not supported on this GPU` | — |
| FP16 dot2c | `v_dot2c_f32_f16 v0, v1, v2` | ❌ FAIL `instruction not supported on this GPU` | — |
| FP16 dot2 | `v_dot2_f32_f16 v0, v1, v2, v3` | ✅ OK | `D3A34000 1C0E0501` |
| INT24 | `v_mad_i32_i24 v0, v1, v2, v3` | ✅ OK | `D1C20000 040E0501` |
| INT24 | `v_mad_u32_u24 v0, v1, v2, v3` | ✅ OK | `D1C30000 040E0501` |
| INT16 | `v_mad_i32_i16 v0, v1, v2, v3` | ✅ OK | `D1F20000 040E0501` |
| INT16 | `v_mad_u16` / `v_mad_i16` | ✅ OK / ✅ OK | `D2040000…` / `D2050000…` |
| FP16 packed | `v_pk_fma_f16` | ✅ OK | `D38E4000 1C0E0501` |
| FP16 packed | `v_pk_add_f16` / `v_pk_mul_f16` / `v_pk_max_f16` | ✅ OK ×3 | `D38F4000…` / `D3904000…` / `D3924000…` |
| INT16 packed | `v_pk_lshlrev_b16` / `v_pk_add_u16` | ✅ OK ×2 | `D3844000…` / `D38A4000…` |
| CVT | `v_cvt_pkrtz_f16_f32 v0, v1, v2` | ✅ OK | `D2960000 00020501` |
| CVT | `v_cvt_f16_f32` / `v_cvt_f32_f16` | ✅ OK ×2 | `7E001501` / `7E001701` |
| CVT | `v_cvt_pk_f16_f32` | ❌ FAIL（clang15: `invalid instruction`；clang18: `not supported on this GPU`） | — |
| CVT | `v_cvt_pknorm_i16_f32` | ✅ OK | `D2940000 00020501` |
| **MIX** | **`v_mad_mix_f32 v0, v1, v2, v3`** | ❌ **FAIL** `instruction not supported on this GPU` | — |
| **MIX** | **`v_fma_mix_f32 v0, v1, v2, v3`** | ✅ **OK** | `D3A00000 040E0501` |
| MIX | `v_fma_mixlo_f16` / `v_fma_mixhi_f16` | ✅ OK ×2 | `D3A10000…` / `D3A20000…` |
| MIX | `v_mad_mixlo_f16` / `v_mad_mixhi_f16` | ❌ FAIL ×2 `instruction not supported on this GPU` | — |
| FP32 | `v_fma_f32` / `v_mac_f32` / `v_fmac_f32` / `v_mad_f32` | ✅ OK ×4 | `D1CB0000…` / `2C000501` / `76000501` / `D1C10000…` |
| FP8 | `v_cvt_f32_fp8` / `v_cvt_pk_fp8_f32` / `v_cvt_f32_bf8` | ❌ FAIL ×3 `instruction not supported on this GPU` | — |
| BF16 | `v_cvt_f32_bf16` | ❌ FAIL（15: `invalid instruction, did you mean: v_cvt_f32_f16?`；18: `not supported on this GPU`） | — |
| BF16 | `v_dot2_f32_bf16` | ❌ FAIL `not supported on this GPU`（与 `docs/hygon_z100.md` §2.1 逐字一致） | — |
| MFMA | `v_mfma_f32_16x16x16f16` | ❌ FAIL `instruction not supported on this GPU` | — |
| MQSAD | `v_mqsad_pk_u16_u8 v[0:1], v[2:3], v4, v[6:7]` | ✅ OK | `D1E60000 041A0902` |
| QSAD | `v_qsad_pk_u16_u8 v[0:1], v[2:3], v4, v[6:7]` | ⚠️ OK（**疑似 LLVM 宽松**，见下） | `D1E50000 041A0902` |

汇总（脚本自报）：`assembler : OK=31 FAIL=12 SYNTAX=0`（两套环境同）。

**⚠️ 关键判读：汇编器接受 ≠ 硬件支持。** 交叉验证 4 条代表指令在 6 个目标上的接受性
（`/opt/dtk/llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=<cpu> -c`）：

```
gfx900   dot8:--  qsad:OK  dot4:--  fma_mix:--      # Vega10 确实没有 DL ops → 门控正确
gfx906   dot8:OK  qsad:OK  dot4:OK  fma_mix:OK
gfx908   dot8:OK  qsad:OK  dot4:OK  fma_mix:OK
gfx90a   dot8:OK  qsad:OK  dot4:OK  fma_mix:OK
gfx1010  dot8:--  qsad:OK  dot4:--  fma_mix:OK      # RDNA1 移除了 DL ops → 门控正确
gfx1030  dot8:OK  qsad:OK  dot4:OK  fma_mix:OK
```

即 `v_dot4/dot8/fma_mix` 的 gfx906 接受性是**有区分度**的（gfx900/gfx1010 被拒），而
`v_qsad_pk_u16_u8`（gfx10+ 指令）在 gfx900/gfx906 上也被接受 ⇒ **LLVM 对 QSAD 的谓词宽松**，
所以**不能只凭"汇编器接受"就认为硬件有**。外部权威佐证：[LLVM《Syntax of gfx906 Instructions》](https://llvm.github.io/www-releases/22.1.8/docs/AMDGPU/AMDGPUAsmGFX906.html) 的 VOP3P 小节明确列出 gfx906 专属指令
`v_dot2_f32_f16 / v_dot2_i32_i16 / v_dot2_u32_u16 / v_dot4_i32_i8 / v_dot4_u32_u8 /
v_dot8_i32_i4 / v_dot8_u32_u4 / v_fma_mix_f32 / v_fma_mixlo_f16 / v_fma_mixhi_f16`，
**不含** `v_mad_mix_f32`、`v_dot4c_i32_i8`、`v_dot2c_f32_f16`、`v_qsad_pk_u16_u8` —— 与本机汇编器实测完全吻合（QSAD 除外，属 LLVM 宽松）。

### 3.2 `__has_builtin` 可用性（门禁二）

探针做法（`-dM -E` 比 `#if` 更好解析，且必须用 **amdgcn 目标**，宿主 x86 目标下这些内建一律不可见 —— 已做对照，控制组通过）：

```bash
CLANG=/opt/dtk/llvm/bin/clang
$CLANG -target amdgcn-amd-amdhsa -mcpu=gfx906 -E -dM has_builtin.h | grep HAS_
$CLANG -E -dM has_builtin.h | grep HAS_     # 对照：host triple 下 0 条（控制组通过）
```

两套工具链**逐字节相同**的完整结果（29 项）：

```
HAS=1  __builtin_amdgcn_sdot4          HAS=0  __builtin_amdgcn_sudot4
HAS=1  __builtin_amdgcn_udot4          HAS=1  __builtin_amdgcn_sdot2
HAS=1  __builtin_amdgcn_udot2          HAS=1  __builtin_amdgcn_sdot8
HAS=0  __builtin_amdgcn_sudot8         HAS=1  __builtin_amdgcn_udot8
HAS=1  __builtin_amdgcn_fdot2          HAS=0  __builtin_amdgcn_fdot2_f16_f16
HAS=0  __builtin_amdgcn_fdot2_f32_bf16 HAS=0  __builtin_amdgcn_mad_mix
HAS=0  __builtin_amdgcn_fma_mix        HAS=0  __builtin_amdgcn_fma_mix_f32
HAS=1  __builtin_amdgcn_cvt_pkrtz      HAS=1  __builtin_amdgcn_cvt_pk_u8_f32
HAS=1  __builtin_amdgcn_cvt_pk_i16     HAS=1  __builtin_amdgcn_cvt_pknorm_i16
HAS=1  __builtin_amdgcn_mqsad_pk_u16_u8 HAS=1 __builtin_amdgcn_qsad_pk_u16_u8
HAS=1  __builtin_amdgcn_ds_bpermute    HAS=1  __builtin_amdgcn_ds_swizzle
HAS=0  __builtin_amdgcn_mfma_f32_16x16x16f16
HAS=0  __builtin_amdgcn_mfma_i32_16x16x16i8
HAS=0  __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
HAS=0  __builtin_amdgcn_cvt_pk_fp8_f32 HAS=0  __builtin_amdgcn_cvt_pk_f32_fp8
HAS=1  __builtin_amdgcn_s_sleep        HAS=1  __builtin_amdgcn_s_barrier
```

`__has_builtin` 只回答"内建存在与否"，**不回答"gfx906 上能不能用"**：`sdot8`/`udot8` 报 1，`v_dot8_i32_i4` 也真的被汇编器接受（§3.1 正确），但 `sudot4`/`fdot2_f16_f16` 报 0 —— 与 DTK clang 自带的
`/opt/dtk/llvm/lib/clang/15.0.0/../../../include/clang/Basic/BuiltinsAMDGPU.def` 的 feature 门槛完全一致（原文）：

```
286:TARGET_BUILTIN(__builtin_amdgcn_fdot2,      "fV2hV2hfIb", "nc", "dot7-insts")
287:TARGET_BUILTIN(__builtin_amdgcn_fdot2_f16_f16,"hV2hV2hh",  "nc", "dot8-insts")
289:TARGET_BUILTIN(__builtin_amdgcn_fdot2_f32_bf16,"fV2sV2sfIb","nc", "dot8-insts")
290:TARGET_BUILTIN(__builtin_amdgcn_sdot2,      "SiV2SsV2SsSiIb","nc","dot2-insts")
292:TARGET_BUILTIN(__builtin_amdgcn_sdot4,      "SiSiSiSiIb","nc","dot1-insts")
293:TARGET_BUILTIN(__builtin_amdgcn_udot4,      "UiUiUiUiIb","nc","dot7-insts")
294:TARGET_BUILTIN(__builtin_amdgcn_sudot4,     "iIbiIbiiIb","nc","dot8-insts")
295:TARGET_BUILTIN(__builtin_amdgcn_sdot8,      "SiSiSiSiIb","nc","dot1-insts")
296:TARGET_BUILTIN(__builtin_amdgcn_udot8,      "UiUiUiUiIb","nc","dot7-insts")
```

⇒ **gfx906 具备 dot1 + dot2 + dot7 特性，不具备 dot8**，这一个事实解释了全部观测：
`v_dot4_i32_i8 / v_dot8_i32_i4`（dot1）、`v_dot2_i32_i16`（dot2）、`v_dot4_u32_u8 / v_dot2_f32_f16 / v_mqsad_pk_u16_u8`（dot7）可用；
`v_dot4c_i32_i8 / v_dot2c_f32_f16 / v_dot2_f32_bf16 / fdot2_f16_f16`（dot8）不可用。

**HIP 设备 pass 复核**（`#warning` 法，两套工具链都做，因为 `__has_builtin` 在 host/device pass 可能不同）：

```bash
cat > h.hip <<'EOF'
#include <hip/hip_runtime.h>
#if __has_builtin(__builtin_amdgcn_sdot4)
#warning SDOT4_VISIBLE_IN_DEVICE_PASS
#endif
EOF
hipcc -S --cuda-device-only --offload-arch=gfx906 -O3 h.hip -o h.s
```

```
# 宿主 DTK 24.04.3：
has_hip.hip:3:2: warning: SDOT4_BUILTIN_VISIBLE_IN_HIP_DEVICE_PASS [-W#warnings]
has_hip.hip:6:2: warning: FDOT2_BUILTIN_VISIBLE_IN_HIP_DEVICE_PASS [-W#warnings]
# 容器 DTK 26.04：
h.hip:3:2: warning: SDOT4_VISIBLE_IN_DEVICE_PASS [-W#warnings]
dotcount=1                       # 同一个文件里 v_dot4_i32_i8 计数 = 1
```

⇒ `csrc/quantization/gguf/ggml-common.h:1129` 的 `__has_builtin(__builtin_amdgcn_sdot4)` 守卫在
**gfx906 的 HIP 设备 pass 里为真**，`__dp4a` 走 sdot4 分支（不是 4 次乘加的软件展开）。

### 3.3 实测 codegen（门禁三）

做法：freestanding device 片段（`-target amdgcn-amd-amdhsa -mcpu=gfx906 -O3 -S`，与 HIP 设备 pass 同一 LLVM 后端/MC），
外加 `hipcc -S --cuda-device-only --offload-arch=gfx906` 复核。完整 20 项（两套工具链逐字节相同）：

| 来源 | 预期指令 | 结果 | 实测汇编行（原文） |
|---|---|---|---|
| `__builtin_amdgcn_sdot4` | `v_dot4_i32_i8` | ✅ EMITTED ×1 | `v_dot4_i32_i8 v2, v6, v7, v8` |
| `__builtin_amdgcn_udot4` | `v_dot4_u32_u8` | ✅ EMITTED ×1 | `v_dot4_u32_u8 v2, v6, v7, v8` |
| `__builtin_amdgcn_sdot2` | `v_dot2_i32_i16` | ✅ EMITTED ×1 | `v_dot2_i32_i16 v2, v6, v7, v8` |
| `__builtin_amdgcn_udot2` | `v_dot2_u32_u16` | ✅ EMITTED ×1 | `v_dot2_u32_u16 v2, v6, v7, v8` |
| `__builtin_amdgcn_sdot8` | `v_dot8_i32_i4` | ✅ EMITTED ×1 | `v_dot8_i32_i4 v2, v6, v7, v8` |
| `__builtin_amdgcn_udot8` | `v_dot8_u32_u4` | ✅ EMITTED ×1 | `v_dot8_u32_u4 v2, v6, v7, v8` |
| `__builtin_amdgcn_fdot2` | `v_dot2_f32_f16` | ✅ EMITTED ×1 | `v_dot2_f32_f16 v2, v6, v7, v8` |
| `__builtin_amdgcn_mqsad_pk_u16_u8` | `v_mqsad_pk_u16_u8` | ✅ EMITTED ×1 | `v_mqsad_pk_u16_u8 v[2:3], v[6:7], v10, v[8:9]` |
| `__builtin_amdgcn_cvt_pkrtz` | `v_cvt_pkrtz_f16_f32` | ✅ EMITTED ×1 | `v_cvt_pkrtz_f16_f32 v2, v2, v3` |
| `__builtin_amdgcn_cvt_pknorm_i16` | `v_cvt_pknorm_i16_f32` | ✅ EMITTED ×1 | `v_cvt_pknorm_i16_f32 v2, v2, v3` |
| `__builtin_amdgcn_cvt_pk_u8_f32` | `v_cvt_pk_u8_f32` | ✅ EMITTED ×1 | `v_cvt_pk_u8_f32 v2, v2, 0, v3` |
| `__fp16x2 a*b+c`（`_Float16` 向量） | `v_pk_fma_f16` | ✅ EMITTED ×1 | `v_pk_fma_f16 v2, v6, v7, v8` |
| `__fp16x2 +` / `*` | `v_pk_add_f16` / `v_pk_mul_f16` | ✅ ×1 / ×1 | `v_pk_add_f16 v2, v4, v5` / `v_pk_mul_f16 v2, v4, v5` |
| **混合精度** `f32 + (f16)a*b`（无内建，源码直出） | `v_fma_mix_f32` | ✅ **EMITTED** ×1 | `v_fma_mix_f32 v2, v2, v3, v4 op_sel_hi:[1,0,0]` |
| 混合精度（inline asm，带 op_sel） | `v_fma_mix_f32` | ✅ EMITTED ×1 | `v_fma_mix_f32 v2, v6, v7, v8 op_sel:[0,1,0] op_sel_hi:[1,0,0]` |
| `a*b+c`（fp32） | `v_fmac_f32` | ✅ EMITTED ×1 | `v_fmac_f32_e32 v8, v6, v7` |
| `__builtin_fmaf(a,b,c)`（fp32） | `v_fmac_f32` | ✅ EMITTED ×1 | `v_fmac_f32_e32 v8, v6, v7` |
| inline asm `v_fma_f32` | `v_fma_f32` | ✅ EMITTED ×1 | `v_fma_f32 v2, v6, v7, v8` |
| inline asm `v_mad_i32_i24` | `v_mad_i32_i24` | ✅ EMITTED ×1 | `v_mad_i32_i24 v2, v6, v7, v8` |

汇总：`codegen : EMITTED=20 MISSING=0 ERROR=0`。

`hipcc` 通道（`tools/dcu/isa_probe.sh` Phase 3b，同一 `.hip` 文件里 4 个 kernel）：

```
-> hipcc_device.s
  sdot4    v_dot4_i32_i8      x1
  fdot2    v_dot2_f32_f16     x1
  pkfma    v_pk_fma_f16       x1
  mix      v_fma_mix_f32      x1
    	v_fma_mix_f32 v1, v2, v3, v4 op_sel_hi:[1,0,0]
```

**关于 `v_mad_mix_f32` / `v_fma_mix_f32` 的重点确认（任务书点名项）**：

1. `v_mad_mix_f32` **在 gfx906 上不存在** —— 汇编器报 `error: instruction not supported on this GPU`；
   LLVM 的 gfx906 专属指令表里也没有它（Vega10/gfx900 的 `V_MAD_MIX_*` 在 Vega20 被 `V_FMA_MIX_*` 取代）。
2. `v_fma_mix_f32` / `v_fma_mixlo_f16` / `v_fma_mixhi_f16` **存在且可用**（汇编器接受 + 有编码）。
3. **clang 没有对应的 `__builtin_amdgcn_fma_mix*` 内建**（`__has_builtin` 全 0，`BuiltinsAMDGPU.def` 里也没有）→
   要用它必须走 **inline asm**（已验证 `v_fma_mix_f32 %0, %1, %2, %3 op_sel:[...] op_sel_hi:[...]` 经 `"v"` 约束可编译可落码），
   或者**写混合精度的普通表达式**让后端自己发（`o + (float)a * b` 在 `-O3` 下就发 `v_fma_mix_f32`）。
4. 该指令**在本树已构建的 `_C` 里已被真实发射 12,017 次**（见 §3.5），不是纸面能力。

### 3.4 已有资产：前序工程的 W8A8 sdot4 内核（只读复核）

任务书要求"先找现成的"。**找到了，而且是完整可复用资产**（前序工程，只读）：

```
/home/lyh/server/int8_sdot4/w8a8_sdot4_kernel.hip     484 行  ← W8A8 sdot4 GEMV + 分块 GEMM（torch 扩展）
/home/lyh/server/int8_sdot4/int8_sdot4_patch.py       337 行  ← 注入层（把 kernel 插进 Int8ScaledMMLinearKernel 选择）
/home/lyh/server/int8_sdot4/w8a8_sdot4_build.py        95 行  ← 预编译（torch.utils.cpp_extension.load）
/home/lyh/server/int8_sdot4/{check,test,bench,probe}_*.py/sh
/home/lyh/server/修复方案-INT8-sdot4.md                       ← 实测报告（decode 950–1130 → 203–217 ms/token，≈4.5×）
```

其内核形态（源码复核）：
- **decode（M≤8）**：`w8a8_gemv_kernel` —— 每个 block 负责一个输出列 n（即 W 的一行），block 内 64 线程切 K，
  `__builtin_amdgcn_sdot4(xp[m*K4+i], ww, acc[m], false)` 做 int32 精确累加，LDS 树形归约，
  epilogue `red*x_s[t]*w_s[n]` 之后才转 out dtype 再 `+bias`；权重**保持 `[N,K]` int8、不转置不反量化**，W 每行只读一次、与 M 无关。
- **prefill（M>32）**：`w8a8_gemm_kernel<BM,BN,TM,TN>` —— `BM×BN` register tile，K 方向 `BK=32`（8 个 int32）双缓冲
  global→寄存器→LDS，每条 sdot4 做 4 组 int8 MAC，实测 22 TFLOPS（Triton FMA 退化路径只有 2.2 TFLOPS）。
- K 必须 4 的倍数；不满足时上层回落原 Triton 内核。

**本步独立复核（只编译，不运行，符合硬约束）**：把该文件的 device 部分抽出（`sed -n '33,107p'`+`'218,398p'`，
加 `#include <hip/hip_runtime.h>` 与强制实例化），在 **DTK 26.04 / clang 18** 下编译：

```bash
docker exec z100-build-211 bash -lc \
  'hipcc --cuda-device-only -S --offload-arch=gfx906 -O3 -std=c++17 /tmp/dev_only.hip -o /tmp/dev_only.s'
```

```
dev_only.s = 145661 bytes；4 个实例全部生成：
_ZN12_GLOBAL__N_116w8a8_gemv_kernelILi8EfEEv…    (GEMV M≤8, out=fp32)
_ZN12_GLOBAL__N_116w8a8_gemm_kernelILi64ELi64ELi4ELi4EfEEv… (GEMM 64×64 tiled, out=fp32)
_ZN12_GLOBAL__N_116w8a8_gemv_kernelILi1EDF16_EEv… (GEMV M=1, out=fp16)
_ZN12_GLOBAL__N_116w8a8_gemm_kernelILi32ELi64ELi4ELi4EDF16_EEv… (GEMM 32×64, out=fp16)

指令计数：v_dot4_i32_i8 = 777    v_dot8 = 0   v_pk_fma_f16 = 0
          v_fma_mix_f32 = 0      v_mfma = 0   v_fmac_f32 = 0
样例：	v_dot4_i32_i8 v1, v15, v14, v1
```

⇒ 该资产在**本步的工具链（DTK 26.04/clang 18）上编译通过且真的落 `v_dot4_i32_i8`**，
没有任何标量 FMA 退化。

**注入锚点在本树是否齐全**（前序工程的 patch 依赖 4 个符号）：

```bash
grep -rn "_POSSIBLE_INT8_KERNELS" vllm/model_executor/kernels/linear/__init__.py
# 411:_POSSIBLE_INT8_KERNELS: dict[PlatformEnum, list[type[Int8ScaledMMLinearKernel]]] = {
# 418:    PlatformEnum.ROCM: [AiterInt8ScaledMMLinearKernel, TritonInt8ScaledMMLinearKernel],
# 794:        _POSSIBLE_INT8_KERNELS,
grep -rn "^class Int8ScaledMMLinearKernel" vllm/model_executor/kernels/linear/
# vllm/model_executor/kernels/linear/scaled_mm/ScaledMMLinearKernel.py:190:class Int8ScaledMMLinearKernel(
```

| patch 需要的符号 | 本树位置 | 齐备 |
|---|---|---|
| `Int8ScaledMMLinearKernel` | `vllm/model_executor/kernels/linear/scaled_mm/ScaledMMLinearKernel.py:190`（由 `.../linear/__init__.py:172` 再导出） | ✅ |
| `_POSSIBLE_INT8_KERNELS` | `vllm/model_executor/kernels/linear/__init__.py:411`，ROCM 表在 `:418` | ✅ |
| `PlatformEnum` | 同文件（`_POSSIBLE_INT8_KERNELS` 的 key 类型，`:418` 使用 `PlatformEnum.ROCM`） | ✅ |
| `current_platform` | 同文件 `module.current_platform` | ✅ |
| 分派入口（`kp.insert(0, …)`） | `:418` 的 ROCM 列表，第 0 位可插 | ✅ |

树分支为 `gfx906/main`（`git rev-parse --abbrev-ref HEAD`），`csrc/quantization/gguf/` 布局与
`vllm/model_executor/kernels/linear/` 结构都与前序工程一致 ⇒ 注入层**结构上可直接迁移**。

### 3.5 复用清点（门禁四）：树内机器码级复核

除源码清点外，本步把 canonical 扩展的**设备码反汇编**做了统计。方法（DTK 的 `llvm-objdump` 不支持
`--offload-arch`，故手工拆包）：

```bash
SO=/mnt/game/port/gfx906-vllm-211/vllm/_C.abi3.so          # canonical 副本，78 MB（S5 记录同一份）
llvm-objcopy --dump-section .hip_fatbin=/tmp/fat.bin $SO /tmp/null.so
python3 - <<'PY'   # .hip_fatbin 里是 21 个 __CLANG_OFFLOAD_BUNDLE__ 包（.hipFatBinSegment 0x1f8 字节）
# 按魔数 __CLANG_OFFLOAD_BUNDLE__ 切分 → /tmp/bun/b00..b20.bin
PY
for f in /tmp/bun/b*.bin; do
  clang-offload-bundler -type=o -unbundle -targets=hipv4-amdgcn-amd-amdhsa--gfx906 -input=$f -output=${f%.bin}.o
done
for o in /tmp/bun/*.o; do llvm-objdump -d $o; done > /tmp/all_gfx906.dis   # 3,249,543 行 / s_endpgm 2,127
```

**canonical `_C.abi3.so` 的 gfx906 设备码指令直方图**（21 个 code object，1308 个 ggml/vec_dot 类符号）：

| 指令 | 计数 | 判读 |
|---|---|---|
| **`v_dot4_i32_i8`** | **71,442** | **GGUF `__dp4a`→sdot4 路径已在出货二进制里真实发射** |
| `v_dot2_f32_f16` | 4,608 | GPTQ `q_gemm.cu` 的 `__ockl_fdot2` |
| `v_pk_fma_f16` | 4,426 | packed fp16 FMA |
| `v_pk_add_f16` | 6,168 | packed fp16 add |
| **`v_fma_mix_f32`** | **12,017** | **混合精度 FMA（f16 输入 + f32 累加）已在用** |
| `v_fmac_f32` | 65,584 | 标量 fp32 FMA（S8b 的"慢路径"就是它的清一色形态） |
| `v_fma_f32` | 25,742 | 标量 fp32 FMA |
| `v_mad_i32_i24` | 261 | INT24 乘加 |
| `v_mad_mix` / `v_mfma` / `v_dot8` / `v_dot4_u32_u8` / `v_mqsad` | 0 / 0 / 0 / 0 / 0 | 与 ISA 判定一致（`mad_mix` 不存在；MFMA 不存在） |

按 code object 定位（每包一个 TU，含 `llvm-readelf -sW` 抽出的符号名）：

```
b13.o  dot4=0      mix=0      pkfma=4416  dot2=4608   :: vllm::gptq::gemm_half_q_half_gptq_*（GPTQ q_gemm.cu）
b17.o  dot4=71442  mix=9670   pkfma=10    dot2=0      :: dequantize_block_iq*/mul_mat_vec_q5_K_q8_1*/vec_dot_q3_K_q8_1（GGUF）
其余包 dot4=0；mix 零散分布（b00:1408 / b09:220 / b16:243 / b11:150 / b10:87 / b18:40 / b01:184 / b08:15）
```

**⇒ 树内已有、且已在 gfx906 上真实生效的指令复用点**（`file:line` + 是否启用）：

| 复用点（`file:line`） | 用途 | gfx906 是否启用 / 证据 |
|---|---|---|
| `csrc/quantization/gguf/ggml-common.h:1129-1131` | `__dp4a` → `__builtin_amdgcn_sdot4(a,b,c,false)`（`__has_builtin` 守卫） | ✅ 启用：设备 pass `HAS=1`；`_C` 中 `v_dot4_i32_i8` × 71,442 |
| `csrc/quantization/gguf/mmvq.cuh:159` `mul_mat_vec_q8_0_q8_1_cuda`（`:168` `..._row_tile` 快路径） | **int8 权重 × int8 激活 GEMV**，per-32-block scale | ✅ 启用：`gguf_kernel.cu:847` `case 8:`（`GGML_TYPE_Q8_0`）分派到这里；含 ROCm 专用 `BLOCK_SIZE 64`、`ggml_mmvq_grid_vecs`、`row_tile`/`pipe` 调优（`mmvq.cuh:1-20`） |
| `csrc/quantization/gguf/vecdotq.cuh:165` `vec_dot_q8_0_q8_1_impl`、`:179` `vec_dot_q8_1_q8_1_impl` | `__dp4a` 点积 + block scale 折叠 | ✅ 启用（同上路径） |
| `csrc/quantization/gguf/vecdotq.cuh` 50 处 + `mmvq.cuh` 11 处 `__dp4a` | 量化点积 | ✅ 启用 |
| `csrc/quantization/gptq/q_gemm.cu:60` `__ockl_fdot2(dq[i], *a2_ptr++, result, true)` | fp16 点积 → `v_dot2_f32_f16` | ✅ 启用（`_C` 中 4,608 条） |
| `csrc/quantization/gptq/q_gemm.cu:72,83` `__hfma(result_h, qs_h, g_result)` | 混合精度 FMA → `v_fma_mix_f32` | ✅ 启用（`_C` 中 12,017 条） |
| `csrc/quantization/gptq/q_gemm.cu:64,74` `__hfma2` → `v_pk_fma_f16` | packed fp16 FMA | ✅ 启用（4,426 条） |
| `csrc/libtorch_stable/moe/moe_wna16.cu`（`__hfma2`）、`quantization/marlin/*`、`quantization/gptq/qdq_*.cuh` 的 `__hfma2` | W4A16/MoE 反量化+乘加 | ⚪ 已编译进 `_moe_C_stable_libtorch`（`CMakeLists.txt:1349` HIP 分支明确为 gfx906 的 GGUF MoE 路径保留）/`_C_stable_libtorch`；仅相应量化模型才走。`csrc/moe/moe_wna16.cu` 不在任何源列表（本树死文件） |
| `csrc/attention/dtype_float16.cuh:338` inline asm `v_pk_fma_f16` | attention dtype 的 packed fp16 FMA | ⚪ 已编译（经 `attention_dtypes.h:4` 被 `csrc/attention/paged_attention_v1/v2.cu` 收录进 `_C`），但 paged-attention/skinny-GEMM 在 gfx906 是 MFMA-only 桩（S5/S8），**不在活跃路径**。注：`_C` 里 4,426 条 `v_pk_fma_f16` 实际来自 b13 = GPTQ `q_gemm.cu`，不是这条 |
| `csrc/rocm/q_gemm_rdna3.cu:113`、`csrc/rocm/moe_q_gemm_rdna3.cu:65`、`csrc/rocm/skinny_gemms_int4.cu:79` 的 `__builtin_amdgcn_fdot2` | RDNA3 fp16/bf16 点积 | ❌ **未启用**：`q_gemm_rdna3.cu:47 #if defined(__HIPCC__) && defined(__gfx1100__)`；`skinny_gemms_int4.cu` 走 RDNA(gfx11/12) 宏，且 `CMakeLists.txt:1562-1564` 在 gfx906 上把 `skinny_gemms.cu`/`skinny_gemms_int4.cu` 从 `_rocm_C` **REMOVE_ITEM 掉**（注释原文：gfx906 无 MFMA，注册只会暴露没有可用路径的 op） |
| `csrc/rocm/skinny_gemms.cu:2114,2309`（注释提到 gfx12 的 `v_dot4_f32_fp8_fp8`） | fp8 点积 | ❌ 未启用（gfx12 专属） |
| `csrc/quantization/w8a8/int8/scaled_quant.cu:101/144/266/298` | int8 激活量化（静态/动态 per-token） | ✅ 启用，且是 `_C` 的一部分；**纯 fp32 数学，不含任何 dot 指令**；前序 patch 复用了它的 `_C.dynamic_scaled_int8_quant(out, x2c, scales, None)` 以避免每次调用两次分配 |
| `csrc/mamba/gfx906_decode_kernels.cu`（13 个 gfx906 专用 kernel） | GDN/MTP decode | ✅ 启用；只用 `__shfl_xor_sync` 等标准内建，**无 sdot4/packed f16 → 无 ISA 资产可复用** |
| `vllm/model_executor/kernels/linear/mixed_precision/gfx906_gptq_wna16.py:33` | gfx906 的 W4A16 GPTQ 路线 | ✅ 启用（`on_gfx906()` 正向门控） |
| `vllm/model_executor/kernels/linear/__init__.py:411,418` | `_POSSIBLE_INT8_KERNELS[ROCM] = [Aiter, Triton]` | ⚠️ **W8A8 侧没有任何 sdot4**：Aiter 要求 `compute_capability>=90`（gfx906=9,0 出局），Triton 在 gfx906 上退化成 FMA 展开 |
| `vllm/model_executor/kernels/linear/scaled_mm/triton.py:163` `TritonFp8BlockScaledMMKernel` | FP8 block-scaled（P6 目标位） | ⚠️ 可用但 36× 慢（S8b），内核注释里已写明 |

### 3.6 Triton gfx906 codegen 实测（门禁四之二）

命令（**GPU 只用 2,3 号卡**；python 派发，无独立可执行程序）：

```bash
docker exec z100-build-211 bash -lc 'cd /src && source tools/dcu/env.sh >/dev/null 2>&1
  export HIP_VISIBLE_DEVICES=2,3 PYTHONPATH=/mnt/game/triton-gfx906/install \
         TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
  python3 tools/dcu/triton_codegen_probe.py'
```

环境自报：

```
triton  : 3.6.0 /mnt/game/triton-gfx906/install/triton/__init__.py
torch   : 2.11.0
devices : 2 visible HIP_VISIBLE_DEVICES= 2,3
  cuda:0 -> gfx906:sramecc+:xnack-
  cuda:1 -> gfx906:sramecc+:xnack-
```

**(1) `tl.dot(fp16 × fp16 → fp32)`：BLOCK / num_warps 扫描（全部落到 `v_dot2_f32_f16`）**

| kernel | num_warps | 总指令 | `v_dot2_f32_f16` | `v_fma_f32` | `v_fmac_f32` | `v_mfma` | 与 fp32 matmul 的 max abs err |
|---|---|---|---|---|---|---|---|
| 16×16×16 | 1 | 108 | **32** | 0 | 0 | 0 | 9.537e-07 |
| 16×16×16 | 2 | 88 | **16** | 0 | 0 | 0 | 9.537e-07 |
| 32×32×32 | 2 | 258 | **128** | 0 | 0 | 0 | 2.861e-06 |
| 32×32×64 | 4 | 297 | **128** | 0 | 0 | 0 | 7.629e-06 |
| 64×64×64 | 4 | 758 | **512** | 0 | 0 | 0 | 5.722e-06 |
| 128×128×64 | 8 | 2320 | **1024** | 0 | 0 | 0 | 1.144e-05 |
| 16×16×32 | 1 | 167 | **64** | 0 | 0 | 0 | 1.907e-06 |

计数是**满密度**的：`v_dot2_f32_f16` 每条 64 lane × 2 MAC = 128 MAC，
`M*N*K/128`（× num_warps）与计数逐项吻合（16³/128=32、32³/128=256=128×2 warps、64³/128=2048=512×4 warps…）。
**`v_fma_f32`/`v_fmac_f32`/`v_mfma` 全 0** —— 没有 FMA 退化。

**(2) m=1 decode 形状（BLOCK_M=16，mask 掉 15 行）**

```
gemv_f16_m1_16x128x64_w1 : total=1557  v_dot2_f32_f16=1024  v_perm_b32=128
                            buffer_load_dword=77  buffer_store_dword=77  ds_read_b128=64
```

⇒ 发的是 dot，但 `M=1` 也要按 `tl.dot` 的最小 tile `M=16` 计算 ⇒ **16× 的算力浪费**（1024 条 dot 里有 15/16 是掩码行）。
这正是"decode 用自写 GEMV、不用 Triton"的量化依据。

**(3) `tl.dot(int8 × int8 → int32)`：`v_dot4_i32_i8`**

| kernel | num_warps | 总指令 | `v_dot4_i32_i8` | 与 fp32 oracle 的 int32 精确比对 |
|---|---|---|---|---|
| 16×16×16 | 1 | 119 | **16** | exact match ✅ |
| 32×32×32 | 2 | 247 | **64** | exact match ✅ |
| 16×16×64 | 1 | 147 | **64** | exact match ✅ |

（计数同样是满密度：64 lane × 4 MAC = 256 MAC/条，`M*N*K/256`×warps。）
⇒ **打补丁后的 triton（`supportsVDot()` 含 VEGA）在 gfx906 上确实把 int8 `tl.dot` 落到 `v_dot4_i32_i8`**，
与 `docs/hygon_z100.md` §3.3 的"32 条 `v_dot`"一致，本步进一步给出了密度与形状覆盖。

**(4) `tl.sum` / 逐元素 fp16**

```
sum_f16_1024   : v_add_f32_dpp=5  v_add_f32_e32=6  v_cvt_f32_f16_e32=2  v_cvt_f32_f16_sdwa=2  v_cvt_f16_f32_e32=1
                 → 归约**不是** packed：先升 fp32、标量 DPP 归约、再降 fp16
ewise_f16_1024 : v_pk_fma_f16=2   （a*b+a 发了 packed f16 FMA ✅）
```

⇒ 逐元素 fp16 算术**是 packed 的**；`tl.sum` 走标量 fp32 归约（可接受，归约不在 GEMM 热路径）。

**(5) P6 目标形态：fp8 → fp16 → `tl.dot`**

```
fp8_to_f16_dot_32x32x32_w2 : total=683
    v_dot2_f32_f16=128   v_cndmask_b32_e32=160   v_cmp_ne_u16_e32=144
    v_and_b32_e32=48     v_perm_b32=40           v_lshlrev_b16_e32=20
与"同样 fp8 值先转 fp16 再 fp32 matmul"的 max abs err = 0.000e+00
```

⇒ **数值完全正确、且落到了 `v_dot2_f32_f16`**；但 **fp8→fp16 的软转换代价压过了点积**：
128 条 dot 对应约 400 条转换/选择类指令（≈2.4 ALU/点积指令，实测 total 683）。
P6 要拿到收益，**必须把转换摊薄**（见 §6.2）。

**(6) 对照组：直接 `tl.dot(fp8 × fp8)`**

```
dot_fp8_32x32x32_w2 : total=802
    v_fmac_f32_e32=248   v_cndmask_b32_e32=160   v_cmp_ne_u16_e32=144
    v_and_b32_e32=48     v_cvt_f32_f16_e32=16    v_fma_f32=8
    v_dot2_f32_f16=0     v_dot4_i32_i8=0         v_mfma=0
```

⇒ 在更小的 tile 上**复现**了 S8b 的结论（4096³ 时是 8160 条 `v_fmac_f32_e32`，`v_dot*`/`v_mfma`/packed 全 0）：
gfx906 上 fp8 的 `tl.dot` 只能退化成标量 FMA。**FP8 必须先转 fp16 再 dot。**

**Triton 侧总结论（一句话）**：自编 triton 3.6 在 gfx906 上 **`tl.dot` 满密度地吃到 dot 指令** ——
fp16 输入发 `v_dot2_f32_f16`、int8 输入发 `v_dot4_i32_i8`、逐元素 fp16 发 `v_pk_fma_f16`；
只有 fp8（及 fp8→fp16 转换）走软路径，`v_fma_f32`/`v_fmac_f32`/`v_mfma` 在 fp16/int8 dot 里为 0。

### 3.7 两套工具链差异（任务书要求"注明"）

`diff` 三份 TSV（宿主 clang 15 / 容器 clang 18）：

```diff
--- host/assembler.tsv	+++ cont_assembler.tsv
24c24
< FAIL	CVT-pk-f16-f32	error: invalid instruction
---
> FAIL	CVT-pk-f16-f32	error: instruction not supported on this GPU
39c39
< FAIL	BF16-cvt-f32-bf16	error: invalid instruction
---
> FAIL	BF16-cvt-f32-bf16	error: instruction not supported on this GPU
# builtins.tsv : 无差异（0 行 diff）
# codegen.tsv  : 无差异（0 行 diff）
```

⇒ **只有 2 条失败指令的错误措辞不同（都是 FAIL），接受/拒绝结论、内建可用性、codegen 结果三张表在两套工具链上完全一致。**

---

## 4. 门禁判定

| # | 门禁 | 判定 | 证据 |
|---|---|---|---|
| 1 | 四列表（汇编器 / 内建 / codegen / 复用点） | ✅ 通过 | §3.1（43 条，两套工具链）、§3.2（29 项 + HIP 设备 pass 复核）、§3.3（20 项 EMITTED / 0 MISSING）、§3.5（复用表 + 机器码直方图） |
| 2 | `v_mad_mix_f32` 是否存在 | ✅ 已判定 | **不存在**（汇编器 `not supported on this GPU`；LLVM gfx906 指令表未列）；`v_fma_mix_f32` **存在可用**且源码/内联汇编都能落码 |
| 3 | Triton `tl.dot` 是否吃到 packed/dot | ✅ 已判定 | §3.6：fp16 `tl.dot` → `v_dot2_f32_f16`（7/7 形状满密度，`v_fmac_f32`=0）；int8 → `v_dot4_i32_i8`（精确）；fp8 → 248 条标量 `v_fmac_f32_e32` |
| 4 | `isa_probe.sh`（三合一 + 退出码） | ✅ 通过 | 两套环境 `rc=0`，`baseline : MATCH (all 21 expectations hold)`；`ISA_PROBE_STRICT=1` 时 `rc=0`；**负向测试**（把基线里 `MIX-fma-mix-f32 OK` 改成 `FAIL` 后运行）得到 `BASELINE-MISMATCH asm MIX-fma-mix-f32 want=FAIL got=OK` + `NEG_RC=1` ⇒ 门禁真的会咬 |
| 5 | `triton_codegen_probe.py` | ✅ 通过 | `rc=0`，6 组 kernel 全部编译并读出 `asm["amdgcn"]` |
| 6 | "W8A8 的 GEMM 能否直接复用现有代码" | ✅ 已明确回答 | §3.4/§3.5 + §6.1：**能**，前序工程的 `w8a8_sdot4_kernel.hip` + `int8_sdot4_patch.py` 是现成资产，4 个注入锚点在本树全部存在，且本步验证其 device 码在 DTK 26.04/clang 18 上落 777 条 `v_dot4_i32_i8` |
| 7 | 硬约束（只编译不运行 / 不动生产容器 / GPU 只用 2,3） | ✅ 遵守 | 无任何独立 HIP 可执行程序；只用 `hipcc -c/-S`、`clang -S`、`llvm-objdump`、`llvm-readelf`、`clang-offload-bundler`、`python3`（triton 经 python 派发）；仅在 `z100-build-211`（任务书指定）内 `docker exec`；triton 探针 `HIP_VISIBLE_DEVICES=2,3`（自报 `devices : 2`）；未 start/kill/commit 任何容器；未动 `z100-build` |
| 8 | 写作用域 | ✅ 遵守 | 只新增 `tools/dcu/isa_probe.sh`、`tools/dcu/triton_codegen_probe.py`、本归档；`csrc/**`、`docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md` 均未改 |

---

## 5. 未完成项 / 阻塞与原因

1. **`/mnt/game` 写不进去（沙箱限制）**：本会话文件沙箱为 workspace-write，
   `touch /mnt/game/.s11d_write_test` → `Permission denied`，所以**探针原始产物落在 `/tmp/s11d/{host,container}`**，
   容器内产物在 `z100-build-211:/tmp/s11d_cont`。归档里已把三张表与全部关键输出**原文内嵌**，不依赖 `/tmp` 存活。
   若后续要求"产物落 `/mnt/game`"，需在主 agent 侧（或用一个有写权限的容器）代落。
   *（`/mnt/game/triton-gfx906/cache` 由容器内 root 写入 → triton 探针不受影响。）*
2. **`v_dot8_i32_i4` / `v_mqsad_pk_u16_u8` 只有"汇编器 + codegen"证据，无硬件执行证据**：
   本步硬约束禁止运行自编译 HIP 可执行程序，故未做真机数值验证。已有反例警示：`v_qsad_pk_u16_u8`
   在 gfx900/gfx906 上都被汇编器接受（gfx10+ 指令）⇒ 接受性不足为凭。
   **结论里只把 `v_dot4_i32_i8` 当作可依赖项**（前序工程已在生产硬件上验证 13–14 ms/token），
   `v_dot8`（INT4）留作"待真机验证"的候选，不影响 P2/P6 决策。
3. **前序 `w8a8_sdot4_kernel.hip` 的完整 TU 未能在本容器整体编译**：
   它 `#include <ATen/cuda/CUDAContext.h>`，而 `z100-build-211` 的 torch 2.11 头无条件包含
   `cuda_runtime_api.h` / `cusparse.h`，DTK 里没有这两个 shim（`find / -name cuda_runtime_api.h` 只命中 triton 的 nvidia backend）
   ⇒ `hipcc` 无法编译该 TU 的 host 部分。**已用"抽出 device 部分 + 强制实例化"绕开并得到 §3.4 的 777 条 `v_dot4_i32_i8`**，
   即设备码已独立验证；host 侧编译问题属集成阶段（S11e）事项，且其自己的 `w8a8_sdot4_build.py`
   用 `torch.utils.cpp_extension.load` 已在生产容器成功编译过。
4. **未做性能实测**：本步是审计步（硬约束只允许编译），没有跑 kernel 计时；
   P2/P6 的性能结论全部引用前序工程的实测数据（`/home/lyh/server/修复方案-INT8-sdot4.md`）与 S8b 的 304 ms/36×。

---

## 6. 对后续步骤的影响

### 6.1 建议 (a)：W8A8 INT8 GEMM 用哪条指令与什么形态

**指令：`v_dot4_i32_i8`（`__builtin_amdgcn_sdot4(a, b, c, /*clamp=*/false)`）。**
理由（全部可复现）：汇编器接受（`D3A84000 1C0E0501`）；`__has_builtin` = 1（含 HIP 设备 pass 复核）；
clang 两端 codegen 落码；**且是全流程唯一有生产硬件实测先例的 INT8 指令**。
明确**不要**用 `v_dot4c_i32_i8`（gfx906 缺 dot8 特性，汇编器直接拒绝），也不要用 `v_dot4_u32_u8`（W8A8 需要有符号；
且 `_C` 里 0 次使用，无先例）。

**形态（对齐 compressed-tensors W8A8 的原生布局：权重 `[N,K]` int8 + per-channel `w_s` float32，激活 per-token int8 + `x_s`）：**

| 形状 | 结构 | 关键点 |
|---|---|---|
| decode `M ≤ 8` | **一 block 一输出列**：block 内 64 线程切 `K/4` 个 int32，`sdot4` 做 int32 精确累加，LDS 树形归约，epilogue `acc*x_s[m]*w_s[n]` → out dtype → `+bias` | 权重每行只读一次、访存与 M 无关（前序实测 432 GB/s，7 投影 0.224 ms）；**不要**用 `tl.dot`（最小 tile M=16 ⇒ 1/16 利用率，§3.6(2)） |
| prefill `M > 32` | `BM×BN` register tile（实测最优 64×64，M≤32 用 32×64），`BK=32`（8 int32）双缓冲 global→寄存器→LDS，每线程 `TM×TN` 个累加器 | K 必须是 4 的倍数；前序实测 22 TFLOPS vs Triton 2.2 TFLOPS |
| 输出 | fp16（本树 `--dtype float16`） | epilogue 顺序必须与 `triton_scaled_mm` 同序：先 `(acc*x_s)*w_s`，再转 out dtype，再加 bias |

**复用结论：能直接复用，且应当复用**（"改一行胜过写一千行"的极致版本）：

- 内核与注入层都在 `/home/lyh/server/int8_sdot4/`（只读资产），4 个注入锚点在本树全部存在（§3.4 表）；
- 本步独立验证 device 码在 DTK 26.04/clang 18 上编译通过并落 **777 条 `v_dot4_i32_i8`**；
- 生产实测：decode 950–1130 → **203–217 ms/token（≈4.5×）**，int8 线性层 decode 每层 6 投影 13.48 → **0.206 ms**，
  prefill 2.1–2.25 → **22.0–22.4 TFLOPS**，文本与 logprob 逐位一致（前序工程记录）。
- **S11e 的落地方式**：把该 kernel 走现有 CMake/ninja 编入 `_C`/`_rocm_C`（而不是 `cpp_extension.load` 的独立 `.so`），
  保留 `int8_sdot4_patch.py` 的 `Int8ScaledMMLinearKernel` 子类（`is_supported` 判 `on_gfx906()`），
  插到 `_POSSIBLE_INT8_KERNELS[ROCM]` 第 0 位，Triton 作回退；编完必须 `strings <so> | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+"` 仍只有 gfx906。
- **备选（不推荐但可选）**：改造树内 GGUF `Q8_0×Q8_1` MMVQ（`mmvq.cuh:159` / `gguf_kernel.cu:847 case 8`）——
  它已经在 gfx906 上跑 `v_dot4_i32_i8`（`_C` 里 71,442 条）并有 row_tile/pipe 调优，但权重布局是 GGUF per-32-block，
  只覆盖 `vecs 1..4`，且不能直接挂进 `Int8ScaledMMLinearKernel`（需新 op + 适配层），
  适配量大于直接迁移前序内核。**把它当参考实现与回归对照，不当主路线。**

### 6.2 建议 (b)：FP8→FP16 模拟 GEMM 用哪条指令 / 什么路线

**主路线（推荐）：Triton，`tl.dot(fp16, fp16, out_dtype=tl.float32)` ⇒ gfx906 落 `v_dot2_f32_f16`。**
本步已在 gfx906 上实测：7/7 形状（16³ … 128×128×64）满密度发 `v_dot2_f32_f16`，
`v_fma_f32`/`v_fmac_f32`/`v_mfma` **全 0**；P6 目标形态（fp8 tile → `to(tl.float16)` → `tl.dot`）数值与
"同一批 fp8 值先转 fp16 再 fp32 matmul" **逐位相同（max abs err = 0.0）**。
实现落点：`vllm/model_executor/kernels/linear/scaled_mm/triton.py:163` 的 `TritonFp8BlockScaledMMKernel`（S8b 已定位的模拟点）。

**必须同时满足的两个前提（否则拿不到收益）：**

1. **绝不直接 `tl.dot(fp8, fp8)`** —— 实测 32³ tile 只发 248 条 `v_fmac_f32_e32` + 0 条 dot，
   正是 S8b 在 4096³ 上看到的 8160 条标量 FMA 的成因。
2. **把 fp8→fp16 转换摊薄/换实现** —— Triton 目前的软转换约 400 条 ALU/选通指令对应 128 条 dot（≈2.4 ALU/点积）。
   建议二选一（S11f 先做 A/B）：
   - **摊薄（零新代码）**：只对权重 tile 转一次并复用（外层 N/K 循环里 hoist），
     或权重侧预转换到 fp16 而不改变常驻显存（P6 §4B 的设计约束）；
   - **换实现（packed 位运算 / LUT）**：e4m3→fp16 用 packed 位操作
     （`v_pk_lshlrev_b16` / `v_pk_add_u16` / `v_pk_cmp_*`，本步已确认 gfx906 可用）
     或 256 项 LDS 查表（512 B）把转换压到 ~1 ALU/元素级别。
   > 注意：**不能用硬件 fp8 转换** —— `v_cvt_f32_fp8` / `v_cvt_pk_fp8_f32` / `v_cvt_f32_bf8` 在 gfx906 上
   > 全部 `not supported on this GPU`（§3.1），所以 fp8→fp16 必然是软转换。

**HIP 路线（备选，若 Triton 的转换开销摊不薄）**：自写 kernel，边读边转（权重常驻 FP8 不展开），
用 **`v_dot2_f32_f16`（`__builtin_amdgcn_fdot2`）** 做 fp32 累加的点积，配 `v_pk_fma_f16`（编译器从 `_Float16` 向量算术自动发）做逐元素/短归约；
树内先例：`csrc/quantization/gptq/q_gemm.cu:60` 的 `__ockl_fdot2` 已经在 gfx906 上落 `v_dot2_f32_f16`（`_C` 里 4,608 条），
可以直接照抄调用方式。

**`v_fma_mix_f32` 的定位（任务书点名项）**：存在、可用、且**树内已发射 12,017 次**
（`csrc/quantization/gptq/q_gemm.cu:72,83` 的 `__hfma(half, half, half)` 这类混合精度表达式就够）。
它可以把"f16 操作数 + f32 累加"融进一条指令、省掉显式 cvt，适合做 epilogue/短点积的补充；
但**没有 clang 内建**，要用得走 inline asm（`op_sel`/`op_sel_hi` 已实测可编译可落码）。
`v_mad_mix_f32` **在 gfx906 上不存在**（Vega10 → Vega20 改名），任何文档/示例里沿用 `v_mad_mix_f32` 的做法在 Z100 上会编译失败。

### 6.3 对 P2 / P6 的直接影响（决策级）

| 步骤 | 本步结论带来的变化 |
|---|---|
| **S11e（P2）** | 从"自写 `int8_sdot4`"降级为**迁移 + CMake 集成**：内核、注入层、调参方案、实测基线全部现成；风险从"数值正确性未知"降到"集成与编译纪律"。指令确定为 `v_dot4_i32_i8`（`__builtin_amdgcn_sdot4`，clamp=false），形态确定为"M≤8 每列一 block GEMV + M>32 的 64×64/BK32 双缓冲 GEMM"。**不要**碰 `v_dot4c_i32_i8`/`v_dot8`(INT4，待真机验证) |
| **S11f（P6）** | Triton 路线**已验证可行且落到 `v_dot2_f32_f16`**，因此 §4B 的"Triton 路线（推荐先试）"成立；但**瓶颈转移到 fp8→fp16 软转换**（不是点积），S11f 的第一件事应该是量测/优化这个转换（hoist 或 packed 位运算/LUT），而不是继续调 `tl.dot` 参数。HIP 路线保留为备选，其点积指令是 `v_dot2_f32_f16`（`__ockl_fdot2`/`__builtin_amdgcn_fdot2`） |
| P4（KV/attention 调优） | 附带结论：attention dtype 的 `v_pk_fma_f16`（`csrc/attention/dtype_float16.cuh:338`）在 gfx906 上是**已编译但不在活跃路径**（paged-attention 走 MFMA-only 桩），所以 P4 的 packed fp16 收益要靠 Triton attention 侧的 `tl.dot`→`v_dot2_f32_f16`（S6 已打通），不是这条 csrc 路径 |
| 全局 | `v_mfma` 恒 0、bf16 无原生转换、fp8 无硬件转换 —— 这三条**再次确认，不要再投入**（与 S8b、`docs/hygon_z100.md` §2.1 一致） |

---

## 7. 复现方式

```bash
# 0) 前提：仓库根、DTK 已装
cd /home/lyh/z100-vllm

# 1) 探针脚本语法检查
bash -n tools/dcu/isa_probe.sh

# 2) 宿主（DTK 24.04.3 / clang 15）三合一探针 → /tmp/s11d/host
ISA_PROBE_ENV_TAG=host-dtk24.04.3-clang15 ISA_PROBE_OUT=/tmp/s11d/host \
  bash tools/dcu/isa_probe.sh

# 3) 容器（DTK 26.04 / clang 18）三合一探针（任务书指定用 z100-build-211）
docker exec z100-build-211 bash -lc \
  'cd /src && ISA_PROBE_ENV_TAG=container-dtk26.04-clang18 ISA_PROBE_OUT=/tmp/s11d_cont \
   bash tools/dcu/isa_probe.sh'

# 4) 严格模式（基线不符即 exit 1，可作回归门禁）
ISA_PROBE_STRICT=1 bash tools/dcu/isa_probe.sh            # 期望 rc=0
#    负向自检：sed 's/^asm MIX-fma-mix-f32 OK$/asm MIX-fma-mix-f32 FAIL/' \
#      tools/dcu/isa_probe.sh > /tmp/probe_neg.sh && ISA_PROBE_STRICT=1 bash /tmp/probe_neg.sh
#    期望：BASELINE-MISMATCH ... got=OK + rc=1

# 5) 两套工具链结果 diff（预期：只有 2 行错误措辞差异）
docker exec z100-build-211 bash -lc 'cat /tmp/s11d_cont/assembler.tsv' > /tmp/cont_assembler.tsv
diff /tmp/s11d/host/assembler.tsv /tmp/cont_assembler.tsv

# 6) Triton gfx906 codegen 探针（GPU 只用 2,3 号卡）
docker exec z100-build-211 bash -lc 'cd /src && source tools/dcu/env.sh >/dev/null 2>&1
  export HIP_VISIBLE_DEVICES=2,3 \
         PYTHONPATH=/mnt/game/triton-gfx906/install \
         TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache \
         TRITON_PROBE_ASM_DIR=/tmp/s11d_asm
  python3 tools/dcu/triton_codegen_probe.py'

# 7) 前序 sdot4 内核的 device 码复核（只编译，不运行）
docker exec -i z100-build-211 bash -c 'cat > /tmp/orig.hip' \
  < /home/lyh/server/int8_sdot4/w8a8_sdot4_kernel.hip
docker exec z100-build-211 bash -lc 'cd /tmp && {
    echo "#include <hip/hip_runtime.h>"; echo "#include <cstdint>";
    sed -n "33,107p" /tmp/orig.hip; sed -n "218,398p" /tmp/orig.hip;
    echo "typedef _Float16 h16;";
    echo "extern \"C\" __attribute__((used)) void s11d_force() { (void)&w8a8_gemv_kernel<8, float>; (void)&w8a8_gemm_kernel<64, 64, 4, 4, float>; (void)&w8a8_gemv_kernel<1, h16>; (void)&w8a8_gemm_kernel<32, 64, 4, 4, h16>; }";
  } > /tmp/dev_only.hip
  hipcc --cuda-device-only -S --offload-arch=gfx906 -O3 -std=c++17 /tmp/dev_only.hip -o /tmp/dev_only.s
  grep -c v_dot4_i32_i8 /tmp/dev_only.s'      # 期望 777

# 8) canonical `_C.abi3.so` 的设备码直方图（DTK 的 llvm-objdump 不支持 --offload-arch，故手工拆 21 个包）
docker exec z100-build-211 bash -lc '
  SO=/mnt/game/port/gfx906-vllm-211/vllm/_C.abi3.so
  llvm-objcopy --dump-section .hip_fatbin=/tmp/fat.bin $SO /tmp/null.so
  python3 - <<PY
d=open("/tmp/fat.bin","rb").read(); pat=b"__CLANG_OFFLOAD_BUNDLE__"
import os; os.makedirs("/tmp/bun",exist_ok=True)
offs=[]; i=d.find(pat)
while i!=-1: offs.append(i); i=d.find(pat,i+1)
for n,o in enumerate(offs):
    end=offs[n+1] if n+1<len(offs) else len(d)
    open(f"/tmp/bun/b{n:02d}.bin","wb").write(d[o:end])
print("bundles:",len(offs))
PY
  for f in /tmp/bun/b*.bin; do
    clang-offload-bundler -type=o -unbundle -targets=hipv4-amdgcn-amd-amdhsa--gfx906 \
      -input=$f -output=${f%.bin}.o
  done
  for o in /tmp/bun/*.o; do llvm-objdump -d $o; done > /tmp/all_gfx906.dis
  for op in v_dot4_i32_i8 v_dot2_f32_f16 v_pk_fma_f16 v_fma_mix_f32 v_mfma v_dot8; do
    printf "%-18s %s\n" "$op" "$(grep -c $op /tmp/all_gfx906.dis)"; done'
  # 期望：71442 / 4608 / 4426 / 12017 / 0 / 0

# 9) 外部权威佐证（gfx906 专属 VOP3P 指令表）
#    https://llvm.github.io/www-releases/22.1.8/docs/AMDGPU/AMDGPUAsmGFX906.html
```

---

## 8. 参考

- `docs/海光Z100-性能优化方案.md` §4（P2）、§4A（本步清单）、§4B（P6）
- [`docs/hygon_z100.md`](../hygon_z100.md) §2.1（早先 ISA 证据）、§3.3（triton 探针 32 条 `v_dot`）
- [`docs/result/z100-08b-gating-and-requirements-result.md`](z100-08b-gating-and-requirements-result.md)（FP8 block-scaled 的 8160 条 `v_fmac_f32_e32` 机器码证据）
- [`docs/result/z100-06-triton-gfx906-result.md`](z100-06-triton-gfx906-result.md)（自编 triton 的 gfx906 补丁与 `supportsVDot()`）
- `tools/dcu/probe_triton.py`（S6 的 triton 探针框架）
- 前序工程（只读）：`/home/lyh/server/int8_sdot4/w8a8_sdot4_kernel.hip`、`int8_sdot4_patch.py`、`w8a8_sdot4_build.py`、`/home/lyh/server/修复方案-INT8-sdot4.md`、`/home/lyh/server/gfx906_port/适配计划-gfx906-vLLM-DTK.md` §…（"int8_sdot4 patch ACTIVE（sdot4 GEMV+GEMM）"）
- DTK 自带 clang 内建定义表：`/opt/dtk/llvm/lib/clang/15.0.0` 对应的 `/opt/dtk-24.04.3/llvm/include/clang/Basic/BuiltinsAMDGPU.def:286-297`
- 外部：[LLVM《Syntax of gfx906 Instructions》](https://llvm.github.io/www-releases/22.1.8/docs/AMDGPU/AMDGPUAsmGFX906.html)（gfx906 专属 VOP3P 指令表：含 `v_dot4_i32_i8`/`v_dot8_i32_i4`/`v_fma_mix_f32`，不含 `v_mad_mix_f32`/`v_qsad_pk_u16_u8`）
- 本步脚本：[`tools/dcu/isa_probe.sh`](../../tools/dcu/isa_probe.sh)、[`tools/dcu/triton_codegen_probe.py`](../../tools/dcu/triton_codegen_probe.py)
