# gfx906（Vega 20 / Z100L）指令与算子能力清单

> **用途**：写 kernel 前先查“这条指令这张卡到底有没有”，避免为一个不存在的
> 指令立项。**判据三级**：`ISA`（LLVM gfx906 指令表里存在，本机汇编器接受）、
> `CODEGEN`（本机 DTK 编译后真的发出这条指令）、`LIB`（本机 DTK 里有对应的
> 预编译 kernel）。
>
> **实测环境**（2026-11-07，4×Z100L 空闲窗口 13:12–13:20 +08）：
> DTK 26.04（clang 17）/ HIP 6.3.26113 / torch 2.11.0；`rocminfo` 报
> `gfx906:sramecc+:xnack-`，60 CU，SIMDs/CU=4，wave64，LDS 64 KB/CU，
> L2 8 MB，clock 1600 MHz。编译探针：`hipcc --offload-arch=gfx906 -S` +
> `rocm-smi` 前后采样（全程 0% 占用）。
>
> **原始证据**（已入库）：探针源码 `tools/dcu/{peak,f64sweep,rate,mix,dot_throughput_probe,gemv_probe}.*`；
> 测量 JSON `docs/perf/data/sol-20261107{,-run2}.json`、`docs/perf/data/ar-20261107-run{1,2}.json`。
> 历史横比见
> [`docs/archive/result/z100-11d-z100-isa-capability-audit-result.md`](../archive/result/z100-11d-z100-isa-capability-audit-result.md)。
> **配套读**：[decode/prefill 极限](SOL-DECODE-PREFILL-20261107.md) —— 本文管“有什么指令”，
> 那篇管“这些指令能跑多快、离极限多远”。

---

## 0. 一页速查

| 想要 | 这张卡有没有 | 用什么 |
|---|---|---|
| fp16 乘加，2 倍速 | ✅ | `v_pk_fma_f16` / `v_dot2_f32_f16` |
| fp32 乘加 | ✅ 基准 1× | `v_fma_f32` / `v_fmac_f32` |
| fp64 乘加（**1:2 还是 1:8？**） | ✅ 但**实测 ≈1:8，不是 1:2** | `v_fma_f64` —— 见 §2.1 |
| int8 dot（4 MAC/指令） | ✅ | `v_dot4_i32_i8` / `v_dot4_u32_u8` |
| int4 dot（8 MAC/指令） | ✅ 且**满速率** | `v_dot8_i32_i4` / `v_dot8_u32_u4` |
| bf16 算术 | ❌ **完全没有** | 只能 fp32 中转，实际比 fp16 慢 |
| fp8 转换/算术 | ❌ | 只能 LUT + `v_dot2_f32_f16` 模拟 |
| 矩阵核 MFMA/WMMA | ❌ | 不存在 |
| fp32 打包（`v_pk_*_f32`） | ❌ | CDNA2 才有 |
| fp64 原子加 | ❌ | 退化成 `global_atomic_cmpswap_x2` 循环 |
| 跨 lane 交换 | ✅ | `ds_bpermute_b32` / `ds_swizzle_b32` / DPP |
| 大页/细粒度内存 | ✅ | `HSA_FORCE_FINE_GRAIN_PCIE=1`（本机生产在用） |

---

## 1. 指令级清单（三级判据）

### 1.1 浮点向量运算

| 指令 | 类别 | ISA | CODEGEN | 本机实测速率 | 说明 |
|---|---|:--:|:--:|---|---|
| `v_fma_f32` / `v_fmac_f32` | fp32 FMA | ✅ | ✅ | **11.97 TF/s**（97% 峰值）@1920 CTA | 基准 1× |
| `v_pk_fma_f16` | fp16×2 FMA | ✅ | ✅ | **23.87 TF/s**（97%）@1920 CTA | 打包 2×，需 ILP |
| `v_pk_add/mul/max/min_f16` | fp16×2 | ✅ | ✅ | — | |
| `v_dot2_f32_f16` | fp16×2，fp32 累加 | ✅ | ✅ | **19.26 TMAC/s = 38.5 TF/s** 折算 @480 CTA | 与 pk_fma 同速率；累加在 fp32 |
| `v_dot2_i32_i16` / `v_dot2_u32_u16` | int16×2 | ✅ | ✅ | 同上 | |
| `v_fma_mix_f32` / `v_fma_mixlo_f16` / `v_fma_mixhi_f16` | 混合精度 FMA | ✅ | ✅（源码直出） | — | **无 builtin**，走普通表达式或 inline asm |
| `v_fma_f64` / `v_add_f64` / `v_mul_f64` | fp64 | ✅ | ✅ | **1.51 TF/s = 12.2% of fp32**（见 §2.1） | 实测 **1:8**，不是规格的 1:2 |
| `v_rcp_f64` / `v_sqrt_f64` / `v_rsq_f64` | fp64 超越 | ✅ | ✅ | — | 原生指令 + IEEE 修正序列 |
| `v_div_scale_f64` / `v_div_fmas_f64` / `v_div_fixup_f64` | fp64 除法序列 | ✅ | ✅ | — | 软件展开，代价高 |
| `v_cmp_*_f64` | fp64 比较 | ✅ | ✅ | — | |

### 1.2 整数 / 量化

| 指令 | MAC/指令 | ISA | CODEGEN | 实测 | 说明 |
|---|:--:|:--:|:--:|---|---|
| `v_dot4_i32_i8` / `v_dot4_u32_u8` | 4 | ✅ | ✅ | **19.26 TMAC/s (77 TF/s 折算)** | = 4× fp32 速率 |
| `v_dot8_i32_i4` / `v_dot8_u32_u4` | 8 | ✅ | ✅ | **38.47 TMAC/s (154 TF/s 折算)** | **满速率 2×**，不是 1× |
| `v_mad_i32_i24` / `v_mad_u32_u24` | 1 | ✅ | ✅ | — | |
| `v_mad_i32_i16` / `v_mad_u16` / `v_mad_i16` | 1 | ✅ | — | | |
| `v_mul_hi_i32_i24` / `v_mul_hi_u32_u24` | 1 | ✅ | — | | |
| `v_perm_b32` / `v_alignbit_b32` / `v_bfe_*` / `v_bfm_*` | — | ✅ | ✅ | — | 反量化/打包主力 |
| `v_cvt_pkrtz_f16_f32` | — | ✅ | ✅ | — | fp32→fp16 打包 |
| `v_cvt_pknorm_i16_f32` / `v_cvt_pk_u8_f32` | — | ✅ | ✅ | — | 量化 |
| `v_sad_u8/u16/u32` / `v_msad_u8` / `v_mqsad_pk_u16_u8` | — | ✅ | ✅ | — | 图像类，科学计算无感 |

### 1.3 明确**不存在**的（本机汇编器逐条拒绝）

| 指令 | 报错 | 影响 |
|---|---|---|
| `v_mfma_f32_16x16x16f16` | `instruction not supported on this GPU` | **无矩阵核** |
| `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` | 同上 | 无 WMMA |
| `v_mad_mix_f32` | 同上 | 只有 `v_fma_mix_*`（Vega20 替换了 Vega10 的 MAD_MIX） |
| `v_dot4c_i32_i8` / `v_dot2c_f32_f16` | 同上 | 无 `c` 变体（累加器复用版本）—— 需 **dot8-insts** |
| `v_dot2_f32_bf16` / `__builtin_amdgcn_fdot2_f32_bf16` | `needs target feature dot9-insts` | **bf16 零支持** |
| `v_cvt_f32_bf16` / `v_cvt_f32_fp8` / `v_cvt_pk_fp8_f32` | 同上 | 无 bf16/fp8 转换指令 |
| `v_pk_*_f32`（packed fp32） | gfx9 无此编码 | fp32 不能指令级双发射 |
| `global_atomic_add_f64` | gfx9 无此编码 | fp64 归约走 CAS 循环 |
| `v_qsad_pk_u16_u8` | ⚠️ 汇编器**接受**但 LLVM 谓词宽松 | **不能凭汇编器接受就认为硬件有** |

> **关键教训**：`v_qsad_pk_u16_u8` 是 gfx10+ 指令，gfx900/gfx906 的汇编器却放行。
> 判据必须是「LLVM gfx906 专属指令表 + 真机 codegen」，不是「汇编器不报错」。
> 本机另一条同类判据：`__has_builtin` 在 host triple 下全 0，必须在
> `-target amdgcn-amd-amdhsa -mcpu=gfx906` 下探测。

### 1.4 同步 / 数据搬运 / 其他

| 能力 | 指令 | 备注 |
|---|---|---|
| Wave64 内交换 | `ds_bpermute_b32` `ds_permute_b32` `ds_swizzle_b32`、DPP、`v_cndmask_b32` | FlashAttention 类 reduce 的基础 |
| LDS | 64 KB/CU（`rocminfo` GROUP segment） | 每 CU 只有 64 KB，tile 上限很紧 |
| 原子 | `global_atomic_add`（i32/i64/fp32）、`*_x2`、`cmpswap` | fp32 有，fp64 无 |
| barrier | `s_barrier` / `__builtin_amdgcn_s_barrier` | |
| 异步拷贝 | ❌ 无 `buffer_load_dword lds` 直通（gfx90a+） | 只能 global→VGPR→LDS |
| 预取 | `buffer_load_dword` + `s_waitcnt` 手工流水 | |

---

## 2. 单卡峰值（实测 vs 规格）

> 口径：60 CU × 64 lane，clk **1600 MHz**（`rocm-smi --showclocks` 全程 level 7）。
> 规格值按 1.6 GHz 复算；厂商 SKU 数（MI50 60 CU@1.725）仅供对照。

| 精度 | 理论（本机 1.6 GHz） | **本机实测** | 达峰率 | 数据来源 |
|---|---:|---:|---:|---|
| fp32 FMA | 12.29 TF/s | **11.97 TF/s** | 97% | §2.1 微基准（1920 CTA） |
| fp16（pk/dot2） | 24.58 TF/s | **23.87 TF/s** | 97% | 同上 |
| int8 dot4 | 49.2 TOPS | **38.5 TOPS**（19.26 TMAC/s×2） | 78% | `dot_throughput_probe`（480 CTA，未扫满） |
| int4 dot8 | 98.3 TOPS | **76.9 TOPS**（38.47 TMAC/s×2） | 78% | 同上 |
| **fp64 FMA** | 6.14 TF/s（按 1:2） | **1.51 TF/s** | **25% of 1:2** | `f64sweep.hip`，见 §2.1 |
| fp64 add | — | 0.755 Tadd/s | — | 同上 |
| fp32 GEMM（4096³，rocBLAS） | 12.29 | **8.25 TF/s** | 67% | `dgemm.py` |
| fp16 GEMM（4096³，rocBLAS） | 24.58 | **19.64 TF/s** | 80% | 4/4 卡 19.63–19.68 |
| fp64 GEMM（4096³，rocBLAS） | 6.14 | **0.957 TF/s** | **15.6%** | `dgemm.py` |
| HBM copy（D2D） | 1024 GB/s | **815–828 GB/s** | 80–81% | 4 卡一致 |
| HBM read（`sum`） | — | **783–784 GB/s** | — | 4 卡一致 |
| P2P copy | 28.4 GB/s | **28.4**（链路 0-1-2 内部）/ **14.3**（经 3 号卡） | — | 见 §3.3 |

### 2.1 最重要的一条修正：**gfx906 的 fp64 是 1:8，不是 1:2**

维基/厂商资料按 Vega 20 “1/2 rate FP64” 宣传（MI50 6.6 TF/s），但**本机实测**：

```
NACC=16, 1920 CTA, 1.6 GHz:
  v_fma_f64  1.510 TF/s   （理论 1:2 = 6.14 → 24.6%）
  v_add_f64  0.755 Tadd/s （FMA 的一半 → 说明 FMA 是 add 的双倍周期）
  v_fma_f32 11.97 TF/s
  ratio fp64/fp32 = 0.126 ≈ 1:8
```

- 内层循环是 **16 条独立 `v_fma_f64`**（已 dump ISA 确认，无额外搬运），
  5120 个 wave 打满 60 CU —— **不是 ILP 或占用不足**。
- rocBLAS DGEMM 4096³ 只有 0.957 TF/s（更低），说明连库也拿不到 1:2。
- 同一 SKU 的 **Z100（非 L）** 对外报 10.8 TF/s FP64；本机 4 张全是 **Z100L**，
  `rocminfo` 明确 `Marketing Name: Z100L`。

**结论：把这台机器当“高 FP64 机器”立项前，先按 1.5 TF/s 算账。** 如果某个
科学计算负载的关键路径是 fp64 GEMM，它的实际算力是 fp16 的 1/16，不是 1/2。

---

## 3. 软件栈里真能调到的算子（本机 DTK 26.04 实测）

判据：`strings -a <lib> | grep gfx906` 计数 + 预编译产物文件名。

| 库 | gfx906 | 能调到什么 |
|---|:--:|---|
| **rocBLAS / hipBLAS** | ✅ 178 个 Tensile 产物 | GEMM：`HH` `HH_HPA` `BB` `BB_HPA` `CC` `DD` `SS` `HS` `HS_HPA` `BS` `BS_HPA` `I8I` `4xi8I` `4xi8I_HPA` `2xi16I` `2xi16I_HPA` `8xi4I` `8xi4I_HPA`；`l_Alik/Bljk/Bjlk/Cijk` 全布局；GEMV/GER/SYRK/TRSM/TRMM/AXPY/DOT/NRM2 |
| **rocFFT / hipFFT** | ✅ | 1D/2D/3D complex+real、R2C/C2R、batched、plan 复用 |
| **rocSPARSE / hipSPARSE** | ✅（653 处符号） | SpMV/SpMM（CSR/COO/BSR/ELL）、SpGEMM、稀疏三角求解、ILU0/IC0、格式转换 |
| **rocSOLVER / hipSOLVER** | ✅（474 处符号） | LU/Cholesky/QR/SVD/特征值/三对角/最小二乘 |
| **MIOpen / hipDNN** | ✅ | Conv(fwd/bwd/wrw)、Pooling、BatchNorm、Softmax、激活、RNN |
| **rocRAND / hipRAND** | ✅ | Philox/MRG/XORWOW，均匀/正态/gamma/poisson |
| **rocPRIM / hipCUB** | ✅ | reduce/scan/sort/select/radix |
| **RCCL** | ✅ | allreduce/allgather/reduce_scatter |
| **rocprofiler / roctracer / hipprof** | ✅ | PC 采样、kernel trace |
| **hipBLASLt** | ❌ **只有 gfx926/928/92a/936/938** | 用不了；torch 启动会打 `No paths matched ...*gfx906*co` |
| **rocALUTION** | ❌ 本 build 无产物 | 无 CG/BiCGStab/GMRES/AMG，迭代解法器要自己拼 |

### 3.1 bf16 在软件栈里是“假支持”

- 硬件：0 条 bf16 指令（§1.3）。
- rocBLAS：有 `BB` / `BB_HPA` 的 gfx906 产物 → **能跑，但内部是 fp32 展开的模拟**。
- 本机 codegen 实测：`__hfma2` on `__hip_bfloat162` → 发出的是
  `v_fmac_f32` ×2 + `v_perm_b32`（转 fp32 算再转回），**比 fp16 慢约 2×**。
- vLLM 侧已有对应处理：`vllm/platforms/rocm.py:285` 注释
  “gfx906 has no bf16 instructions at all”，`gfx906_coerce_model_dtype()` 强制转 fp16。

### 3.2 fp8 在软件栈里也是模拟

- 无 `v_cvt_f32_fp8`：fp8→fp16 走 **256-entry LDS LUT**，MMA 走 `v_dot2_f32_f16`。
- 所以 **fp8 权重只值“字节减半”**，不涨算力：峰值仍按 fp16 的 24.6 TF/s 算。

---

## 4. 对写 kernel 的直接影响（逐条可执行）

1. **别找矩阵核**。所有 GEMM/attention 的峰值得靠 `v_dot2`/`v_pk_fma` 的 2×
   和 ILP 堆出来；`tl.dot` 在 Triton 里也是降级成 VALU。
2. **fp16 想达峰必须 ≥8 条独立累加链**。本机实测：1 条链 6.05 TF/s，
   8 条链 22.7 TF/s（S58 WP0 与本次 `peak.hip` 独立互证）。**这是“为什么
   attention 只有 3.9 TF”的主因之一**（见 `docs/result/z100-s58-result.md`）。
3. **占用要 ≥8 CTA/CU**。本机 `peak.hip` 扫描：2 CTA/CU → 82%，
   8 CTA/CU → 94%，32 CTA/CU → 97%。**≤2 CTA/CU 直接损失 15–18%**，
   这正是 PERF1 里 `216 VGPR → ≤2 blocks/CU` 那条的量化依据。
4. **int4 是免费午餐**。`v_dot8_i32_i4` 满速率（2× dot4），W4 量化在
   compute-bound 下真能拿 2×；但本机 `gemv_probe` 的 w4_dot8 只有
   24 GB/s raw（48 GB/s fp8-eq）—— **W4 的瓶颈在解包，不在 dot8**。
5. **fp64 只能当“能算”，不能当“算得快”**（§2.1）。任何 fp64 归约不要用
   `atomicAdd(double*)`（CAS 循环）；用 warp/block 内规约 + 单点原子。
6. **fp8/bf16 转换不要进内循环**：LUT 是 LDS 操作，会跟 tile 抢那 64 KB。

---

## 5. 复跑方式

```bash
# 判据 1：指令存在性（汇编器）
printf '\tv_dot8_i32_i4 v0, v1, v2, v3\n\ts_endpgm\n' > /tmp/a.s
$DTK/dcc/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx906 -c /tmp/a.s -o /tmp/a.o

# 判据 2：builtin 可见性（必须 amdgcn 目标）
$DTK/dcc/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx906 -E -dM has.h | grep HAS_

# 判据 3：真机 codegen + 速率（本次用的探针，都在 tools/dcu/）
# 注意用 /opt/dtk-26.10/bin/hipcc 显式路径：PATH 上的 dtk2604 hipcc 会去找
# .../dtk2604/hip/lib/llvm/bin/clang++（该路径不存在）而直接失败。
export PATH=/opt/dtk-26.10/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/hyhal/lib:/opt/hyhal/lib:/opt/dtk-26.10/hsa/lib:/opt/dtk-26.10/lib:$LD_LIBRARY_PATH
HCC="hipcc -O3 --offload-arch=gfx906 --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11"
$HCC tools/dcu/dot_throughput_probe.cpp -o /tmp/dot_probe  # 指令速率对比
$HCC tools/dcu/peak.hip        -o /tmp/peak       # fp32/fp16 达峰率 vs CTA
$HCC tools/dcu/f64sweep.hip    -o /tmp/f64sweep   # fp64 FMA/add vs 占用
$HCC tools/dcu/rate.hip        -o /tmp/rate       # 持续 FMA 速率
$HCC tools/dcu/gemv_probe.cpp  -o /tmp/gemv_probe # GEMV 几何 × M 带宽

# 判据 4：库支持
for l in librocblas.so librocfft.so.0 librocsparse.so librocsolver.so libMIOpen.so.1; do
  echo "$l: $(strings -a $DTK/lib/$l | grep -c gfx906)"
done
ls $DTK/lib/rocblas/library | grep -c gfx906      # 178
ls $DTK/hipblaslt/                             # 只有 926/928/92a/936/938 → 无 gfx906
```

> **编译期与运行期用的不是同一套 DTK，这一点必须分开处理**：
>
> | 场景 | 用什么 | 为什么 |
> |---|---|---|
> | 编译探针 / 自写 kernel | `/opt/dtk-26.10/bin/hipcc` + `/opt/dtk-26.10/lib` | dtk2604 的 `hipcc` 指向不存在的 `hip/lib/llvm/bin/clang++`，直接失败 |
> | 编译期头文件 | 加 `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11` | 否则 `#include <cmath>` 找不到 |
> | 运行 torch / 跑服务 | `.s60/serve_state_environ.txt` 的 `LD_LIBRARY_PATH`（生产栈 dtk2604） | 宿主 `/opt/dtk` 缺 `libhipblas.so.2` / `libomp.so`，`import torch` 直接失败 |
>
> **实测确认**：`/opt/dtk-26.10/bin/hipcc` 编 `tools/dcu/peak.hip` 一次通过并产出可执行
> （2026-11-07 13:56）；而 PATH 上的 dtk2604 `hipcc` 同一命令报
> `clang++: No such file or directory`。
