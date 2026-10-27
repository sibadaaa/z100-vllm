# S11f / P6：Z100（gfx906）FP8 与 BF16 仿真 HIP 内核

> 上层文档：[性能优化方案 §4A/§4B](../海光Z100-性能优化方案.md) · [Todolist](../z100-todolist.md)
> 前置：S11d（算力审计）、S11e/S11k（W8A8 INT8 `sdot4` 骨架与 `custom_op` 集成）
> 状态：✅ 已落地（归档见 [`docs/result/z100-11f-fp8-bf16-emu-kernels-result.md`](../result/z100-11f-fp8-bf16-emu-kernels-result.md)）

## 目的

让 Z100 上**本来完全不可用**的两类低精度路径可用，并且不把权重解量化到 fp16：

1. **FP8（W8A8，e4m3fn / e5m2）**：gfx906 没有任何 fp8 指令、没有 fp8 MFMA，
   `torch._scaled_mm` 的 FP8 路径也不支持，因此 vLLM 里所有 FP8 选核在 gfx906 上
   全部拒绝（`ROCmFP8ScaledMMLinearKernel`：`wvSplitKQ is not built on gfx906 (no MFMA)`；
   torch FP8 系列：`requires a platform with torch FP8 scaled-MM support`）；
   唯一能跑的 block-scaled 路径是纯 Triton 的标量 `v_fmac_f32` 模拟，
   实测慢 36×（[S8b](../result/z100-08b-gating-and-requirements-result.md)）。
2. **BF16**：gfx906 没有任何 bf16 运算指令（转换/点积都不存在），
   且 gfx906 上 decode 形状的未量化线性层两条快路都不可用
   （`LLMM1` 因 MFMA 被排除出 gfx906 构建；`n <= 16` 的 Triton matmul 带 `not on_gfx9()`），
   只能落到 rocBLAS：实测 `[1, 5120] @ [17408, 5120]` bf16 GEMV 仅 235 GB/s
   （HBM 的 24%），小 N 形状更差（N=512：24 GB/s）。

**非目标**：不改量化/加载器语义；不物化 fp16 权重副本（27B FP8 ≈ 27 GB，
fp16 副本 54 GB 装不下）；不做 KV-cache FP8；不做 block-scaled FP8（见 §6）。

## 判定依据（先读）

所有算子选择都以 `tools/dcu/isa_probe.sh` 的实测为准（本步把它扩到 **53 条基线**，
新增行前缀 `FP8EMU-`）：汇编器接受 + `__has_builtin` + **codegen 实际发射**三列表，
外加既有 S11d 审计。要点：

| 能力 | gfx906 | 用途 |
|---|---|---|
| `v_dot2_f32_f16`（`__builtin_amdgcn_fdot2`） | ✅ EMITTED | 两条内核的 MAC 核心：2×fp16 MAC/指令 + **fp32 累加** |
| `ds_read_u16` | ✅ | fp8→fp16 的 256 项 LDS 表（`ds_read_b16` ❌ 不存在） |
| `v_lshlrev_b32 16`、`v_add_u32 0x8000` | ✅（实测即 bf16↔fp32 整条序列） | bf16→fp32 精确展开、fp32→bf16 舍入 |
| `v_cvt_f32_fp8`/`v_cvt_pk_fp8_f32`/`v_cvt_pk_f32_fp8`/`v_cvt_f32_bf8` | ❌ | 必须软件替代 |
| `v_cvt_f32_bf16`/`v_cvt_pk_bf16_f32`/`v_dot2_f32_bf16` | ❌ | 必须软件替代 |
| `v_mfma*` | ❌ | 无张量核 |
| `v_mad_mix_f32` | ❌（Vega20 上已不存在） | 用 `v_fma_mix_f32` / `v_mad_mixlo/hi_f16` |

**关键结论：fp8 与 bf16 的仿真都落在同一条数据通路上**
——「寄存器内把操作数变成 fp16」+「`v_dot2_f32_f16` 做 fp32 累加」+「epilogue 里只乘一次 scale」。

## 方案映射（用户给的 A/B/C 方案 → 内核路线）

| 方案 | 内核实现 | 采纳情况 |
|---|---|---|
| BF16 **A：转 FP16** | 路线 `f16`：权重常驻 bf16，内核内重定标（见 §3.2）后走 `v_dot2_f32_f16` | ⚠️ 采纳为**实验路线**：fp16 的指数窗口比 bf16 窄约 30 个 binade，"逐 lane 精确"的前提是**每个 lane** 的 e8 ∈ [113,143]；零值/次正规会静默回绕（实测：零 lane 变成 2.0），因此默认不启用 |
| BF16 **B：转 FP32** | 路线 `f32`：`v_lshlrev_b32 16` 精确展开 + `v_fma_f32`/`v_fmac_f32` | ✅ **bf16 默认路线**（对全部 bf16 取值精确，含次正规/零/inf） |
| BF16 **C：软件模拟 bf16** | 与 B 等价：载入即 bf16 精度、fp32 累加 = 硬件 bf16 MMA 的数值口径 | ✅ 即路线 `f32`，无需额外内核；Python 侧 `>>16`/`<<16` 模拟不需要 |
| FP8 **A：反量化到 FP16** | 路线 `fp8`：**内核内** LUT 加宽（`ds_read_u16`）→ `v_dot2_f32_f16` → `(acc*a_s)*w_s` | ✅ 采纳；Python/权重副本式反量化**禁止**（内存装不下） |

## 内核设计

### 3.1 `csrc/rocm/gfx906_emu_common.cuh`（共享转换层）

`bf16x2_to_f32x2`（移位精确）、`f32_to_bf16`（RN 舍入）、`bf16_to_f16_guarded`
（逐 lane 保护：e8 ≥ 143 饱和到 ±65504、e8 ≤ 112 冲零）、`bf16x2_to_f16x2_fast`
（窄契约快路径，见 §5）、`fp8_lut_u16`/`fp8x2_to_f16x2`/`fp8_lut_stage`（LUT 加宽）、`dot2`。

**为什么 FP8 走 LUT 而不是位运算**：e4m3 的次正规（|x| < 2^-6 ≈ 0.0156，在 per-channel
scaled 权重里非常常见）无法用「移位 + 加偏置」直接映射到 fp16（两者的次正规指数不同），
位运算路线必须再补逐 lane 修正；而 256 项 fp16 LUT 对 e4m3fn 与 e5m2 的**全部有限码都是
无损的**（e4m3 峰值 448、3 位尾数；e5m2 峰值 57344 < 65504、2 位尾数，其 2^-16 次正规正好
落到 fp16 次正规），一次 `ds_read_u16` 就得到精确值。

### 3.2 BF16 重定标（路线 `f16`）与一个真实坑

设 `s = sign<<15 | e8<<7 | m7`，则 `(s - 112<<7) << 3` 给出 fp16 的
`sign<<15 | (e8-112)<<10 | m7<<3`（前提 e8 ∈ [113,143]）。**注意：直接 `s << 3` 会丢掉
e8 的第 6/7 位（它们落到 lane 的第 16/17 位），对所有 |x| ≥ 2、甚至所有 e8 含 bit6 的
正规数都是错的。**

本步实测到 DTK clang 的两个陷阱（都已写进代码注释，并被测试锁住）：

1. 用 `u16x2_t`（`ext_vector_type(2)`）写这段位运算时，clang 15/18 都会给 packed 移位
   加上 `op_sel_hi:[0,1]` 的 lane 选择，低 lane 实际拿到的是**没减偏置**的值
   （实测 2^-14 → 4.0）。
2. 改用 inline asm 时又踩到 `V_PK_LSHLREV_B16` 的语义：移位量是**按 lane 从 src0 的两个
   半字**取的，给一个标量立即数只会移动一个 lane（实测 2^-14 → 64.0）。

最终实现改成 **32 位标量逐 lane 计算**（无 lane 选择歧义），由
`tests/kernels/test_gfx906_bf16_gemv.py` 端到端锁住。

### 3.3 FP8 W8A8（`csrc/rocm/fp8_emu_gemm.{h,cu}`）

- **契约**：`a`/`w` 为 8 位 fp8、`w` 是 `[N, K]` K 连续（就是 checkpoint 布局，FP8 选核
  传进来的是它的 `[K, N]` 视图，内核用零拷贝 `.t()` 读）；`a_s` 为 `[M]`（per-token）或
  `[1]`（per-tensor），`w_s` 为 `[N]`（per-channel）或 `[1]`；输出 `out_dtype`。
  epilogue 顺序 `(acc*a_s)*w_s` → cast → bias，与 `w8a8_sdot4_*` / `triton_scaled_mm` 对齐。
- **GEMV（M ≤ 8，decode）**：一个 block 一个输出列，`nt=64` 线程跨 K/4 dword 连续读，
  LDS 树归约；**M ≥ 2 时把 A 的加宽提到循环外**（每次 launch 多一个极小的 widen kernel +
  M×K×2 字节 scratch），因为每个 block 都要重读同一份 A，不提出去的话 M=8 时 LUT 访存是
  dot2 的 9 倍。
- **GEMM（M > 8，prefill）**：BM×BN 寄存器 tile + BK=32 双缓冲 LDS，结构照搬
  `w8a8_sdot4.cu`（同一项目、同一硬件已验证），只是 LDS tile 单元从 int32 字换成 fp16 对
  （1 dword = 2 lane = 1 条 dot2）；staging 时用 LUT 把 fp8 加宽成 fp16 对。cfg 0 自动
  （M ≤ 32 → 32×64，否则 64×64），cfg 1..4 显式。

### 3.4 BF16/FP16 decode GEMV（`csrc/rocm/bf16_emu_gemv.{h,cu}`）

一个 block 一个输出列 + LDS 树归约（复用 sdot4 的阻塞方式）。fp16 输入直接走 dot2
（无转换，精确）；bf16 输入 route 0/auto → `f32` 路线，可选 `f16` 路线（需 `range_safe`）。
权重按 `[N, K]` 原样读，不转置不拷贝。

### 3.5 集成点

| 位置 | 内容 |
|---|---|
| `csrc/rocm/{ops.h,torch_bindings.cpp}` | 注册 `fp8_emu_gemv/gemm`、`bf16_emu_gemv`（含 Meta 实现） |
| `CMakeLists.txt` | `VLLM_ROCM_EXT_SRC` 加两个 `.cu` |
| `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py`（新） | `Gfx906Fp8ScaledMMLinearKernel` + 主机侧 LUT + `torch.library.custom_op("vllm::fp8_emu_mm")`（GEMV/GEMM 分支放在 opaque body 内，避免 Dynamo 给 token 维加特化 guard —— S11k 教训） |
| `vllm/model_executor/kernels/linear/__init__.py` | 该内核插到 `_POSSIBLE_FP8_KERNELS[PlatformEnum.ROCM]` **首位**（关闭时 `is_supported` 直接拒绝，行为不变） |
| `vllm/model_executor/layers/utils.py` | `rocm_unquantized_gemm_impl` 的 gfx906 分支：`torch.ops.vllm.bf16_emu_mm`，按实测收益表限定形状（见 §4） |
| `vllm/envs.py` | `VLLM_DCU_USE_FP8_EMU`(默认 False)、`VLLM_DCU_FP8_CVT`、`VLLM_DCU_USE_BF16_GEMV`(默认 False)、`VLLM_DCU_BF16_ROUTE`；**族总开关** `VLLM_DCU_USE_GFX906_EMU`(默认 False，一次打开 INT8/FP8/dense 三个内核，单项显式值优先——见 [S11e2](z100-11e2-int8-family-integration.md)) |

## 4. 实测（详见归档）

FP8（K=5120，卡 2，对照 fp16 rocBLAS —— FP8 本身没有可用基线）：

| 形状 | M=1 模拟 | 带宽 | fp16 rocBLAS | 加速 |
|---|---|---|---|---|
| N=17408 | 0.213 ms | 419 GB/s | 0.827 ms | **3.87×** |
| N=8192 | 0.105 ms | 398 GB/s | 0.570 ms | **5.42×** |
| N=7168 | 0.118 ms | 311 GB/s | 0.495 ms | 4.19× |
| N=5120 | 0.142 ms | 185 GB/s | 0.272 ms | 1.91× |
| N=512 | 0.019 ms | — | 0.080 ms | 4.27× |
| N=48 | 0.022 ms | — | 0.030 ms | 1.39× |

M ≥ 8 时 GEMV 变成 LDS/计算受限（0.5–0.85×），GEMM 在 M=64/512 约为 fp16 rocBLAS 的
0.6–0.9×（**注意基线是 fp16：FP8 的字节数是它的一半**；FP8 的真实对照是"没有可用内核"）。

BF16/FP16 dense GEMV（同样 K=5120，对照 rocBLAS）：

| 形状 | fp16 M=1 | bf16 M=1 | bf16 N=512 M=1 | 备注 |
|---|---|---|---|---|
| N=17408 | 2.74×（593 GB/s） | 2.61×（625 GB/s） | — | rocBLAS 仅 215–235 GB/s |
| N=8192 | 3.95× | 2.50× | — | |
| N=5120 | 2.11× | 2.01× | — | |
| N=512 | 1.49× | **5.71×** | | rocBLAS bf16 小 N 极差 |
| N=48 | 1.29× | **2.46×** | | |

误差：全部 ≤ 0.39 ulp（输出 dtype）。M=8 时内核落后 rocBLAS（0.2–0.9×），
所以 Python 侧只启用实测获胜的形状（单 token；bf16 且 128 ≤ N ≤ 1024、M=4）。

**机器码门禁**（`.hip_fatbin` 拆包 + `llvm-objdump`，76 个 kernel）：
`v_dot2_f32_f16` **18,666**、`ds_read_u16` **1,092**、`v_fma_f32` 24、
**`v_fmac_f32` 0**、`v_mfma` 0、`v_cvt_f32_bf16`/`v_dot2_f32_bf16`/`v_cvt_f32_fp8` 0
—— 即"走的是 dot2 + LUT"，不是 S8b 那种标量模拟。

## 5. 验收（门禁）

1. `ISA_PROBE_STRICT=1 bash tools/dcu/isa_probe.sh` → `baseline: MATCH (all 53 …)`、退出码 0、`SYNTAX=0`。
2. `tests/kernels/quantization/test_gfx906_fp8_emu.py` + `tests/kernels/test_gfx906_bf16_gemv.py`
   全绿（本步 28 passed）：LUT 对全部 256 码无损、GEMV/GEMM 对 fp32 参考的误差仅为
   fp32 累加序（fp32 输出 < 1e-5，fp16 输出 < 1e-3）、奇数形状（M/N/K 非 tile 整数倍）不越界、
   per-tensor/per-channel/per-token、bias、bf16 两条路线的契约（含窄契约的反例）。
3. 机器码门禁：见 §4（`v_fmac_f32` 必须为 0，dot2/LUT 必须存在）。
4. 端到端：`VLLM_DCU_USE_FP8_EMU=1` 时 FP8 per-tensor/per-channel 模型可选到新内核；
   `VLLM_DCU_USE_BF16_GEMV=1` 时 decode 线性层走新 GEMV；两者默认 OFF，关闭即回到
   `TritonFp8BlockScaledMMKernel` / rocBLAS 原行为。
5. 无 DTK/非 gfx906 行为不变：`is_supported` 首查 `current_platform.is_rocm()` + `on_gfx906()`。
6. `csrc` 只新增文件与注册行；`vllm/**` 改动全部带 env 开关。

## 6. 已知限制 / 未做

1. **block-scaled FP8 未做**：仍走 `TritonFp8BlockScaledMMKernel`（慢 36×）。
   新内核的 GEMM 已按 K-chunk 累加，接 block scale 需要在 K 方向按 128 边界切换
   `acc_b` 并在 epilogue 折叠 `As_b*Bs_b`，属下一步。
2. **M ≥ 8 的 GEMV 落后 rocBLAS**：LDS 访存与 dot2 的比值是瓶颈（每次 dot2 取 1 个
   A dword）；下一步方向是用 `uint4` 宽 LDS 读 + 提高 TM/TN。
3. **小 N（≤ 128）大 M 的 GEMM 未被 tile 覆盖**：CTA 数只有 1–2 个，利用率低
   （M=64/N=48：0.37 ms）；需要 split-K（注意会引入非确定求和序，需按本仓库
   "可复现优先"的纪律另做设计）。
4. **bf16 快路径（`f16` 重定标）默认关闭**：契约要求每个 lane 都在窗口内，
   零值会回绕；只有在对**两个操作数**都做完范围检查后才可用。
5. **KV-cache FP8、`--quantization fp8` 的 block 变体**：不在本步。
6. 真机端到端（起服务跑一个 FP8 模型）**本步未做**：内核级数值/性能与门禁已完成，
   端到端留给下一步（需要 FP8 权重与足够显存）。

## 7. 参考

- [性能优化方案 §4A/§4B](../海光Z100-性能优化方案.md)（算力清单与 P6 目标）
- [S11d 算力审计](../result/z100-11d-z100-isa-capability-audit-result.md)（指令矩阵、两套工具链一致）
- [S8b FP8 block-scaled 判定](../result/z100-08b-gating-and-requirements-result.md)（36× 慢路径的机器码证据）
- [S11e/S11k W8A8 `sdot4`](../result/z100-11e-w8a8-sdot4-result.md)（骨架与 `custom_op` 集成）
- [1Cat-vLLM 参考分析](../result/z100-11g-1cat-vllm-reference-analysis-result.md)（"权重压缩常驻 + 边读边转 + 只乘一次 scale"）
