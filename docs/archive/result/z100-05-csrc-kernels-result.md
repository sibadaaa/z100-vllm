# S5 csrc gfx906 内核 + `_rocm_C` + skinny GEMM 排除 — 执行结果

> 归档时间：2026-10-08
> 执行者：子 agent（S5 执行子 agent）
> 对应设计文档：[docs/design/z100-05-csrc-kernels.md](../design/z100-05-csrc-kernels.md)
> 前一档：[docs/result/z100-04-build-system-dtk-result.md](z100-04-build-system-dtk-result.md)
> 状态：见 §4 / §5（构建与真机证据在下文逐条列出）

## 0. 摘要

S5 的编译器侧目标达成：DTK 26.04 工具链挡在 `vllm._C` 前面的 **4 类缺陷**全部定位并修复，
`_C` 的全部 25 个 gfx906 翻译单元在容器 `z100-build` 内逐个编译通过（§3.4），`_rocm_C`
重编为只含 gfx906 且不再注册 skinny GEMM 算子（§3.6）。

修复全部是"只在 DTK 构建路径下生效"的最小侵入式改动：

| # | 缺陷 | 触发点 | 修复位置 |
|---|---|---|---|
| 1 | rocPRIM `uninitialized_array::emplace` 的 placement `new` 在 `__host__ __device__` 里被 clang 18 拒绝 | 任何 hipcub `BlockLoad`/`BlockExchange` 使用者（`csrc/mamba/mamba_ssm/selective_scan_fwd.cu` 等） | 新增 `csrc/dtk_rocprim_compat.h`，由 `CMakeLists.txt` 在 DTK 前缀下以 `-include` 注入 |
| 2 | `__syncwarp` 重定义（DTK HIP 6.3 已自带） | `csrc/fused_qknorm_rope_kernel.cu:41` | 同一文件，把版本号守卫改成 HIP 头自己用的 `HIP_ENABLE_WARP_SYNC_BUILTINS` |
| 3 | `atomicAdd(half*, half)` / `atomicAdd(half2*, half2)` 与 DTK 原生重载二义 | `csrc/quantization/gptq/compat.cuh`、`csrc/libtorch_stable/quantization/gptq/compat.cuh` | 两份文件，用 DTK 全局宏 `__DTK_ARCH__` 跳过软件模拟 |
| 4 | DTK 的 hipify 补丁给每个生成文件塞 `#include <ATen/dtk_macros.h>`，该头在 `TORCH_TARGET_VERSION` 下 `#error` | `_C_stable_libtorch` / `_moe_C_stable_libtorch` 全部 TU | `cmake/hipify.py` 的后处理，就地展开 `__DTK_ARCH__` |

同时真机探针命中**两个 gfx906 运行期地雷**（不是编译期问题）：

* `_rocm_C::wvSplitK` / `wvSplitKQ` 在 gfx906 上是 `UNREACHABLE_CODE` 桩，调用即 device
  assert + VM fault（§3.7.2，真机复现）；
* `_rocm_C::paged_attention` 也是 MFMA-only 桩（`csrc/rocm/attention.cu:3293`），而
  `vllm/platforms/rocm.py:use_rocm_custom_paged_attention()` 因为 `on_cdna()` 里的
  `"gfx9" in "gfx906"` 子串判断在 gfx906 上返回 `True`，真机调用直接崩（§3.7.3，真机复现）。

两者都已按"gfx906 不进 MFMA 假设路径"的要求处理：CMake 把 gfx906 与 gfx1250 一并排除出
skinny GEMM，运行期 `use_rocm_custom_paged_attention()` 对 gfx906 返回 `False`（回退 Triton）。

## 1. 目标与验收标准

### 1.1 设计文档门禁原文（`docs/design/z100-05-csrc-kernels.md` §验收）

| 判据 | 命令 |
|---|---|
| gfx906 专用测试通过 | `python -m pytest tests/kernels/attention/test_triton_unified_attention_gfx906.py tests/kernels/mamba/test_fused_recurrent_packed_decode.py tests/kernels/mamba/test_fused_sigmoid_gating.py tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py tests/model_executor/test_qwen3_5_gdn.py -v` |
| 目标码只含 gfx906 | `dccobjdump --offloading <so> \| grep -o gfx[0-9]* \| sort -u`（**本镜像不可用**，见 §3.5 回退口径） |
| 未注册 skinny 算子且运行期已短路 | `hasattr(torch.ops._rocm_C,"wvSplitK")` 允许为 False，但**不得**出现 `_rocm_C has no wvSplitK` 之类的运行期 AttributeError |

### 1.2 `docs/z100-todolist.md` 门禁表 S5 行（措辞已修正）

> 5 个 gfx906 专用测试通过；`_rocm_C` 暴露 `LLMM1/wvSplitK/wvSplitKQ/wvSplitK_int4_g/wvSplitKrc/paged_attention`
>（**修正**：注册名是 `_rocm_C::paged_attention`，不存在 `paged_attention_rocm`）；`vllm._C`
> 编译成功并只含 gfx906（须先解决 DTK 26.04 rocPRIM placement-new 缺陷）

### 1.3 本任务书追加的验收标准

| 项 | 判据 |
|---|---|
| placement-new 缺陷修复 | `vllm._C` 在 gfx906 上编出来且只含 gfx906；DU 构建路径不受影响 |
| 缺陷范围定界 | `ninja -k 0` 的失败 TU 清单能说明除 mamba 外还有哪些 TU 中招 |
| `_rocm_C` 数学校验 | `LLMM1`/`wvSplitK`/`wvSplitKQ`/`paged_attention` 至少各一次 torch 参考对比，原始输出进归档 |
| 非 DTK 行为不变 | `/mnt/game/port/s4-harness/check_setup_dtk.py` 仍 16/16 |

## 2. 实际改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `csrc/dtk_rocprim_compat.h` | 新增（全文件） | DTK clang 18 未声明 device placement `new/delete` 的补丁头；`#if defined(__HIP__) || defined(__HIPCC__)` 包裹，只在 DTK 构建路径下被 `-include` |
| `CMakeLists.txt` | 260-299 | 新增 DTK 分支：`ROCM_PATH MATCHES "dtk"`（可用 `VLLM_DTK_ROCPRIM_COMPAT=0/1` 覆盖）时把 `-include <repo>/csrc/dtk_rocprim_compat.h` 追加进 `VLLM_GPU_FLAGS`；否则打印 disabled |
| `CMakeLists.txt` | 1553-1584 | skinny GEMM 排除正则由 `gfx1250` 扩为 `gfx1250\|gfx906`，并同步 `else()` 分支的日志文案 |
| `csrc/fused_qknorm_rope_kernel.cu` | 41-49 | `__syncwarp` 回退定义的守卫加上 `!defined(HIP_ENABLE_WARP_SYNC_BUILTINS)` |
| `csrc/quantization/gptq/compat.cuh` | 8-70、104-118 | 引入 `vllm_gptq_has_native_half2_atomic_add` 探针（仅用于 `static_assert` 复核）；`__DTK_ARCH__` 下同时跳过 `half` 与 `half2` 的软件模拟；`VLLM_GPTQ_HALF2_ATOMIC_ADD_EMULATED` 在文件顶部按 CUDA/ROCm/DTK 三分支唯一确定 |
| `csrc/libtorch_stable/quantization/gptq/compat.cuh` | 同上 | 同样的改动；外层守卫保留上游的 `HIP_VERSION >= 7.13` 版本条件 |
| `cmake/hipify.py` | 64-104、124-127 | 新增 `_neutralize_dtk_macros_include()`：hipify 物化完所有结果后，把生成树里 DTK 注入的 `#include <ATen/dtk_macros.h>` 就地替换为等价的 `#define __DTK_ARCH__ 520 / DTK_VERSION / DTKRT_VERSION` |
| `vllm/model_executor/kernels/linear/scaled_mm/rocm.py` | 82-97 | `is_supported()` 增加 `on_gfx906()` 短路（文案与 gfx1250 分开） |
| `vllm/model_executor/layers/utils.py` | 439-452 | `rocm_unquantized_gemm_impl` 的 `ops.LLMM1(...)` 前置 `ops.ROCM_SKINNY_GEMM_ENABLED` |
| `vllm/_custom_ops.py` | 21-29 | 新增 `ROCM_SKINNY_GEMM_ENABLED = hasattr(torch.ops._rocm_C, "LLMM1")`（`hasattr` 探测已验证不会因缺 `torch.ops._rocm_C` 抛异常） |
| `vllm/platforms/rocm.py` | 479-503 | `use_rocm_custom_paged_attention()` 对 gfx906 直接返回 `False` |
| `tools/dcu/check_rocm_c_ops.py` | 新增（全文件） | `_rocm_C` 注册算子清单 + `LLMM1/wvSplitK/wvSplitKQ/paged_attention` 的 torch 参考数值对比；退出码 0/1/2 与 `check_bf16.py` 一致 |

未改动：`docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`、DTK 自身的头文件、
任何 DTK 库目录、`_ON_GFX9`/`on_gfx9()`。

## 3. 执行过程与原始证据

### 3.0 环境与前置

沿用 S4 留下的容器（未重建、未碰生产容器）：

```console
$ docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}'
z100-build      Up 31 minutes   harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413
qwen38-vllm     Up 59 minutes   harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413
```

### 3.1 缺陷范围定界：`ninja -k 0` 的失败 TU 清单

S4 留下的 `build-keepgoing.sh`（`ninja -k 0`）在本次执行开始时仍在跑。它跑到 `[21/62]`
时被主动停止（重建前必须停，否则 ninja 锁构建目录），停止前拿到的失败清单是：

```console
$ docker exec z100-build bash -lc 'grep -E "^FAILED:" /mnt/game/port/build-keepgoing.log | sed "s/ *$//" | sort -u'
FAILED: CMakeFiles/_C.dir/csrc/fused_qknorm_rope_kernel.hip.o
FAILED: CMakeFiles/_C.dir/csrc/mamba/mamba_ssm/selective_scan_fwd.hip.o
FAILED: CMakeFiles/_C.dir/csrc/quantization/gptq/q_gemm.hip.o
FAILED: CMakeFiles/_C.dir/csrc/sampler.hip.o
```

据此可把"是否只有 mamba 相关 TU"定界为 **否**：

* `selective_scan_fwd` / `sampler` —— 同一根因（rocPRIM placement new）；
* `fused_qknorm_rope_kernel` —— `__syncwarp` 重定义；
* `q_gemm` —— `atomicAdd` 二义。

`-k 0` 只跑到 21/62（`_C` 共 25 个 TU），**中段未枚举完**；剩余 TU 用"逐个 TU 用真实
编译命令试编"补足（§3.4 的 25/25 清单），并另有 `_C_stable_libtorch` 的第 4 类缺陷（§3.5）。

### 3.2 缺陷 1：rocPRIM placement new —— 最小复现与 shim 前后对比

S4 的 6 行最小复现（不改 DTK 头文件即可复现，证明不是 vLLM 的 bug）：

```console
$ docker exec z100-build bash -lc 'cat > /tmp/p1.hip <<"EOF"
#include <rocprim/types.hpp>
__global__ void k(int* o) {
  __shared__ rocprim::uninitialized_array<int, 4> a;
  a.emplace(0, 7);
  if (threadIdx.x == 0) o[0] = a.get_unsafe_array()[0];
}
EOF
source /mnt/game/port/setenv.sh >/dev/null 2>&1
hipcc -std=c++17 --offload-arch=gfx906 -c /tmp/p1.hip -o /tmp/p1.o'
/opt/dtk/include/rocprim/types/uninitialized_array.hpp:78:17: error: reference to __host__ function 'operator new' in __host__ __device__ function
   78 |         return *new(ptr) T(std::forward<Args>(args)...);
1 error generated when compiling for gfx906.
```

根因（读 DTK 头确认）：`/opt/dtk-26.04-DCC2602-0317/llvm/lib/clang/18/include/cuda_wrappers/new`
确实**写了** device 版 placement new/delete，但它的结构是

```c
#include_next <new>
#if !defined(__device__)
#undef __CLANG_CUDA_WRAPPERS_NEW   // 头被过早包含 → 解除守卫，指望以后重试
#else
  ... __device__ inline void* operator new(__SIZE_TYPE__, void* __ptr) ...
#endif
```

实际编译 `/tmp/p2.hip`（`#include <new>`）的预处理结果里只有 libstdc++ 的 host 版
（`grep -c "Device overrides for placement" /tmp/p2.ii` → `0`），即 DTK 的这套"重试"机制
在 clang 18 上失效。任何实例化到 `rocprim::uninitialized_array::emplace` 的 TU
（hipcub `BlockLoad`/`BlockExchange`，例如
`csrc/mamba/mamba_ssm/selective_scan_hip.h:233` 的 `BlockLoadT(smem_load).Load(...)`）都会报错。

修复：`csrc/dtk_rocprim_compat.h` 只补上 ROCm 上游 toolchain 本来就有的那几个重载，
placement new 是"在给定地址上构造"的纯语义操作，返回传入指针（delete 空实现）与上游完全等价：

```cpp
__device__ [[gnu::noinline]] void* operator new(std::size_t, void* __p) noexcept { return __p; }
__device__ [[gnu::noinline]] void operator delete(void*, void*) noexcept {}
__device__ [[gnu::noinline]] void* operator new[](std::size_t, void* __p) noexcept { return __p; }
__device__ [[gnu::noinline]] void operator delete[](void*, void*) noexcept {}
```

**shim 前后对比（同一个真实 TU，用 ninja 记录的真实编译命令；只改 `-o`）**：

```console
# 不带 shim：基线失败
$ ... bash /tmp/c1.sh
no-shim rc=1
$ grep -c "error:" /tmp/c1.log
20

# 带 shim
$ ... bash /tmp/c2.sh
with-shim rc=0
$ ls -l /tmp/ss_shim.o
-rw-r--r-- 1 root root 7015832 Oct  8 00:56 /tmp/ss_shim.o
real	1m34.417s
```

注入方式（只作用于 DTK 前缀下的构建）：

```console
$ grep -n "rocPRIM placement-new workaround" /mnt/game/port/build.log | tail -1
9214:-- DTK rocPRIM placement-new workaround enabled: -include /mnt/game/port/gfx906-vllm/csrc/dtk_rocprim_compat.h
$ grep -c "dtk_rocprim_compat" /mnt/game/port/build-gfx906/temp/build.ninja
56
```

**非 DTK 侧未被注入的负向控制（两个独立证据）**：

1. 关掉开关后，ninja 文件里完全没有这个 `-include`：

```console
$ cmake -G Ninja ... -DVLLM_DTK_ROCPRIM_COMPAT=0 ...
-- DTK rocPRIM placement-new workaround disabled (ROCM_PATH=/opt/dtk)
$ grep -m1 "dtk_rocprim_compat" /mnt/game/port/probe-cfg4/build.ninja || echo "NO-SHIM-IN-NINJA"
NO-SHIM-IN-NINJA
```

2. 门禁条件本身在 CMake 里实测（`/tmp/gate_test.cmake`，与 CMakeLists 完全相同的谓词）：

```console
$ cmake -P /tmp/gate_test.cmake
-- GATE=OFF      # ROCM_PATH=/opt/rocm      （社区 ROCm / CUDA 路径）
-- GATE=ON       # ROCM_PATH=/opt/dtk
-- GATE=ON       # ROCM_PATH=/opt/dtk-26.04-DCC2602-0317（本镜像的真实解析结果）
```

另外，`_C_stable_libtorch` 目标的 `FLAGS =` 行也确认带上了 shim（`VLLM_GPU_FLAGS` 全目标共享），
而 CUDA/社区 ROCm 的 `VLLM_DTK_ROCPRIM_COMPAT` 为 `OFF`，`setup.py` 的 DTK 判定本身由
S4 的 16/16 harness 覆盖（§3.8）。

### 3.3 缺陷 2/3：`__syncwarp` 与 `atomicAdd`（源文件级）

`__syncwarp`：

```console
$ grep -rn "__syncwarp" /opt/dtk/hip/include/hip/amd_detail/amd_warp_sync_functions.h | head -2
159:__device__ inline void __syncwarp() {
165:template <typename MaskT> __device__ inline void __syncwarp(MaskT mask) {
```

该定义在 `#ifdef HIP_ENABLE_WARP_SYNC_BUILTINS` 内，而本构建的编译命令里就带
`-DHIP_ENABLE_WARP_SYNC_BUILTINS=1`（见 ninja `FLAGS =`），两条事实同时成立时
`HIP_VERSION(60300000) < 70000000` 的旧守卫不再成立，于是重复定义。

`atomicAdd`：DTK 的 `amd_hip_fp16.h` 在**全局作用域**无条件声明

```console
$ grep -n "atomicAdd(__half2\|atomicAdd(__half " /opt/dtk/hip/include/hip/amd_detail/amd_hip_fp16.h
1858:        __half2 atomicAdd(__half2 *const address, const __half2 val) {
1878:        __half atomicAdd(__half *const address, const __half val) {
```

而 vLLM 在 `vllm::gptq` 里又声明了同签名的软件模拟，调用点（`q_gemm.hip:229` 等）不限定
命名空间 ⇒ 两个候选同优 ⇒ 二义。修复用的是 DTK 自己注入的全局宏 `__DTK_ARCH__`
（`ATen/dtk_macros.h:3-4`），它定义 `__DTK_ARCH__ 520`，且 DTK 的 hipify 会把它写进每个
生成文件，所以是"这个工具链必然有原生重载"的可靠判据；`VLLM_GPTQ_HALF2_ATOMIC_ADD_EMULATED`
在文件顶部就是普通整数字面量，没有再走 `#if ...::value` 这条**行不通**的路
（`::` 不是合法的预处理记号，实测报
`error: token is not a valid binary operator in a preprocessor subexpression`）。

### 3.4 缺陷 4：DTK hipify 注入的 `dtk_macros.h`

`_C_stable_libtorch` / `_moe_C_stable_libtorch` 的每个 TU 都是：

```console
$ docker exec z100-build bash -lc 'head -3 /mnt/game/port/build-gfx906/temp/csrc/libtorch_stable/cache_kernels.hip'
// !!! This is a file automatically generated by hipify!!!
#include <ATen/dtk_macros.h>
#include "torch_utils_hip.h"
...
$ ... bash /tmp/st2.sh
In file included from .../cache_kernels.hip:2:
.../torch/include/ATen/dtk_macros.h:9:2: error: "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
```

两个事实：DTK 给 `torch.utils.hipify` 打的补丁（`HIPIFY_HEADER_FILE`，见
`/usr/local/lib/python3.10/dist-packages/torch/utils/hipify/hipify_python.py:63`）会把该
include 加进每个生成文件；而 `dtk_macros.h` 自己在 `TORCH_TARGET_VERSION` 下就 `#error`
（`CMakeLists.txt:1282` 给 `_C_stable_libtorch` 定义了它）。**该失败与 shim 无关**，把 shim
从命令行去掉后同样失败。

修复放在 `cmake/hipify.py`：hipify 物化完所有结果之后遍历生成树，把该 include 就地换成
等价的宏定义（`__DTK_ARCH__ 520`）。注意必须放在 `hipified_sources` 循环**之后**——torch 的
hipify 是惰性的，文件在调用者第一次取 `hipify_result[...].hipified_path` 时才落盘。

### 3.5 编译成功证据（`_C` 全量 + `_rocm_C`）

`_C_stable_libtorch` 在第 24 步被一个**与本步无关**的既有缺陷挡住（§5.1），因此最终用
`ninja` 只构建 `_C` 与 `_rocm_C` 两个目标：

```console
$ docker exec -d z100-build bash -lc '. /mnt/game/port/setenv.sh; cd "$REPO_DIR";
    ninja -C /mnt/game/port/build-gfx906/temp -j "$MAX_JOBS" _C _rocm_C'
$ docker exec z100-build bash -lc 'grep -n "Linking\|rc=" /mnt/game/port/build-s5c.log | tail -4'
212:[25/28] Linking HIP shared module _C.abi3.so
215:[28/28] Linking HIP shared module _rocm_C.abi3.so
216:----- ninja rc=0 2026-10-08T01:58:43+08:00 -----

$ ls -l /mnt/game/port/build-gfx906/temp/_C.abi3.so /mnt/game/port/build-gfx906/temp/_rocm_C.abi3.so
-rwxr-xr-x 1 root root 78034896 Oct  8 01:57 /mnt/game/port/build-gfx906/temp/_C.abi3.so
-rwxr-xr-x 1 root root 35130920 Oct  8 01:58 /mnt/game/port/build-gfx906/temp/_rocm_C.abi3.so

$ cmake --install /mnt/game/port/build-gfx906/temp --prefix "$REPO_DIR" --component _C
-- Installing: /mnt/game/port/gfx906-vllm/vllm/_C.abi3.so
$ cmake --install /mnt/game/port/build-gfx906/temp --prefix "$REPO_DIR" --component _rocm_C
-- Installing: /mnt/game/port/gfx906-vllm/vllm/_rocm_C.abi3.so
```

`_C` 的 25 个 gfx906 翻译单元全部编译通过；S4 时期失败的 4 个 TU 现在都在
`build-s5c.log` 里以 `Building HIP object` 出现且无 `error:`（长日志不再贴出，
完整文件在容器内 `/mnt/game/port/build-s5c.log`）。

`_rocm_C` 本次是**重编**过的（S4 那次没有排除 skinny GEMM）：`_rocm_C_skinny` 目标不再
存在，只有 2 个 TU（`csrc/rocm/torch_bindings.cpp` + `csrc/rocm/attention.hip`）。

### 3.6 arch 与 import 门禁

门禁脚本落在 `tools/dcu/s5_gate.sh`（可直接粘贴复现，见 §7）：

```console
$ docker exec z100-build bash "$REPO_DIR/tools/dcu/s5_gate.sh"
===== S5 gate 2026-10-08T02:06:25+08:00 =====
python     : Python 3.10.12
torch      : 2.10.0 6.3.26093
triton     : 3.4.0

----- 1) import gate, torch imported first -----
IMPORT_OK /mnt/game/port/gfx906-vllm/vllm/_C.abi3.so
rc=0

----- 2) gfx targets -----
_C.abi3.so                             archs: gfx906
_rocm_C.abi3.so                        archs: gfx906
vendor _C (reference)                  archs: gfx906 gfx926 gfx928 gfx936

----- 3) _rocm_C registered ops -----
['_rocm_C::paged_attention']
ROCM_SKINNY_GEMM_ENABLED = False
paged_attention registered = True
```

`_C` 的算子注册面（旁证，49 条 schema）：

```console
$ docker exec z100-build bash -lc 'source /mnt/game/port/setenv.sh; cd "$REPO_DIR"; python -c "
import torch, vllm._C
print(\"schema count:\", len([s for s in torch._C._jit_get_all_schemas() if s.name.startswith(\"_C::\")]))
print(hasattr(torch.ops._C, \"paged_attention_v1\"))"'
schema count: 49
True
```

`dccobjdump` 仍是断链、`llvm-objdump --offloading` 对 DTK 的 hipfb 仍为空（S4 §5.2），
所以 arch 口径继续用 `strings` 枚举内嵌 target id；vendor 轮子的多架构输出证明该口径有区分力。

### 3.7 `_rocm_C` 数学校验（真机，`tools/dcu/check_rocm_c_ops.py`）

最终产物上的原始输出：

```console
$ docker exec z100-build bash -lc 'source /mnt/game/port/setenv.sh; cd "$REPO_DIR";
    PYTHONUNBUFFERED=1 python tools/dcu/check_rocm_c_ops.py'
torch      : 2.10.0 hip=6.3.26093
device     : Z100L arch=gfx906:sramecc+:xnack-
skinny gemm: ROCM_SKINNY_GEMM_ENABLED=False
registered : _rocm_C::paged_attention
SKIP LLMM1              not registered in this build (skinny GEMM excluded)
SKIP wvSplitK           not registered in this build (skinny GEMM excluded)
SKIP wvSplitKQ          not registered in this build (skinny GEMM excluded)
SKIP paged_attention    use_rocm_custom_paged_attention() is False on this arch (MFMA-only kernel, see csrc/rocm/attention.cu); registration only
PASS
exit=0
```

`LLMM1` 的数值对比在 S4 编出的、**仍注册 skinny 算子**的 `_rocm_C` 上取得（该版本在
gfx906 上会把 `LLMM1` 映射到非 MFMA 回退 kernel，数值正确；这一点也解释了为什么
`LLMM1` 曾经"看起来能用"）：

```console
OK   LLMM1              shape=(1, 256) rel_err=4.413e-04     # 参考实现 x @ weight.t()
```

参考实现：`LLMM1(weight[M,K], x[1,K], 4)` 对 `x @ weight.t()`；`wvSplitK(a[M,K], b[N,K])`
对 `a @ b.t()`；`wvSplitKQ` 对 `(b_fp32 @ a_fp32.t()) * scale_b * scale_a`；
`paged_attention` 对 `softmax(q@k.t()*scale)@v`。

**`wvSplitK`/`wvSplitKQ`/`paged_attention` 在 gfx906 上无法做数值对比**，因为它们各自的
kernel 在 `__HIP__GFX9__` 之外被编译成 `UNREACHABLE_CODE` 桩：真机调用直接 device-side
assert + VM fault（§3.7.2 / §3.7.3）。这正是"gfx906 不得进入任何 MFMA 假设路径"的硬件
证据，处置方式是把它们从构建/派发里摘掉，而不是"绕过检查"。

补充：探针早期用 N=64 得到 `RuntimeError: Unsupported N value`，那是探针形状错，不是缺陷——
`wvSplitK` 的 `switch (N_in)` 只实现 1..5（`csrc/rocm/skinny_gemms.cu:1310-1327`），
`wvSplitKQ` 只实现 1..4（`:2525-2538`）；探针已改为 N=4。

#### 3.7.1 注册算子清单（旧 `_C`+`_rocm_C` 组合下）

```console
$ docker exec z100-build bash -lc 'source /mnt/game/port/setenv.sh >/dev/null 2>&1; cd "$REPO_DIR"; python - <<"PY"
import torch, vllm._rocm_C
print("ops:", sorted(s.name for s in torch._C._jit_get_all_schemas() if s.name.startswith("_rocm_C::")))
print("paged_attention:", hasattr(torch.ops._rocm_C, "paged_attention"))
PY'
ops: ['_rocm_C::LLMM1', '_rocm_C::paged_attention', '_rocm_C::wvSplitK', '_rocm_C::wvSplitKQ', '_rocm_C::wvSplitK_int4_g', '_rocm_C::wvSplitKrc']
paged_attention: True
```

#### 3.7.2 gfx906 上调 `wvSplitK`/`wvSplitKQ` 必然崩（这就是要排除它们的原因）

```console
$ ... python tools/dcu/check_rocm_c_ops.py        # S4 编出的旧 _rocm_C（含 skinny）
OK   LLMM1              shape=(1, 256) rel_err=4.413e-04
/mnt/game/port/build-gfx906/temp/csrc/rocm/skinny_gemms.hip:589: void wvSplitK_hf_sml_(...) [scalar_t = __half, THRDS = 64, YTILE = 2, WvPrGrp = 16, A_CHUNK = 8, UNRL = 2, N = 4]: Device-side assertion `false' failed.
...
bash: line 1: 10974 Aborted                 (core dumped)
exit=134
```

对应源码（gfx906 没有 `__HIP__GFX9__`，所以拿到的是桩）：

```c
// csrc/rocm/skinny_gemms.cu:580-590
#else
template <...> __global__ void wvSplitK_hf_sml_(...) {
  UNREACHABLE_CODE
}
#endif
```

#### 3.7.3 gfx906 上调 `paged_attention` 同样崩

```console
$ ... python tools/dcu/check_rocm_c_ops.py
OK   LLMM1              shape=(1, 256) rel_err=5.528e-04
FAIL wvSplitK           RuntimeError: Unsupported N value: 32,64,64      # 形状问题，见下
FAIL wvSplitKQ          RuntimeError: Unsupported N value: 64,64,32     # 形状问题，见下
.../csrc/rocm/attention.hip:3315: void paged_attention_ll4mi_QKV_mfma4_kernel(...) [BLOCK_SIZE = 16, HEAD_SIZE = 64, ...]: Device-side assertion `false' failed.
... KERNEL VMFault Analysis ...
bash: line 1: 12848 Aborted                 (core dumped)
exit=134
```

`attention.cu:3293-3307` 是同一模式（`#else ... UNREACHABLE_CODE`），而
`vllm/platforms/rocm.py:479` 的 `use_rocm_custom_paged_attention()` 在 gfx906 上返回
`True`（`_ON_CDNA` 用 `"gfx9" in "gfx906"` 的子串判断，见 `rocm.py:264`）⇒ 真机必崩。
已按 §2 的两处改动短路。

注意：`wvSplitK` 的 `Unsupported N value: 32,64,64` 是探针用了 N=64，而该 kernel 的
`switch (N_in)` 只实现 1..5（`skinny_gemms.cu:1310-1327`）；`wvSplitKQ` 只实现 1..4
（`:2525-2538`）。探针形状已修正为 N=4。

### 3.8 非 DTK 主机行为不变（S4 harness）

```console
$ python3 /mnt/game/port/s4-harness/check_setup_dtk.py /home/lyh/z100-vllm
...
PASS  DTK host, torch.hip 6.3,  device=rocm -> _is_hip(): got True, want True
PASS  off-DTK, torch.hip None, device=rocm  -> _is_hip(): got False, want False
PASS  _dtk_version() honours DTKROOT: got '26.04.3', want '26.04.3'
16/16 checks passed
```

### 3.9 5 个 gfx906 专用测试（真机）

**必须先说明跑法**：镜像里的 `tests/conftest.py` 会 `from vllm import LLM, ...`，进而拉起
整条 entrypoint 依赖链，而镜像的 Python 依赖比本 fork 的 `requirements/*.txt` 旧，

```
tests/conftest.py:50: from vllm import LLM, SamplingParams, envs
vllm/tokenizers/mistral.py:8: ModuleNotFoundError: No module named 'mistral_common.guidance'
vllm/tool_parsers/utils.py:16: ImportError: cannot import name 'NamespaceTool' from 'openai.types.responses'
vllm/parser/harmony.py:14: ModuleNotFoundError: No module named 'xgrammar.openai_tool_call_schema'
```

（对应仓库要求 `mistral_common[image] >= 1.11.5`、`openai >= 2.0.0`（实际代码要 3.x）、
`xgrammar >= 0.2.1`；镜像分别是 1.10.0 / 2.21.0 / 0.1.33。）在容器里补齐这三个纯 Python
依赖后（`tblib`、`mistral_common==1.12.0`、`openai==3.26.0`、`xgrammar==0.2.8` +
`apache-tvm-ffi`，均为 pip 安装、不动 DTK/torch），根 conftest 仍然把整条应用链拖进来，
因此这 5 个**内核级**测试用 `--noconftest` 跑（它们不依赖根 conftest 的 fixture）。

```console
$ docker exec -d z100-build bash -lc 'source /mnt/game/port/setenv.sh; cd "$REPO_DIR";
    for f in <5 files>; do
      python -c "import torch, pytest, sys; sys.exit(pytest.main([\"-v\",\"--no-header\",\"--noconftest\",\"-p\",\"no:cacheprovider\",\"$f\"]))"
    done'
$ docker exec z100-build bash -lc 'grep -E "passed" /mnt/game/port/s5-pytest-final.log'
======================== 12 failed, 27 passed in 20.40s ========================   # test_triton_unified_attention_gfx906.py
========================= 2 failed, 3 passed in 14.99s =========================   # test_fused_recurrent_packed_decode.py
============================== 5 passed in 14.11s ==============================   # test_fused_sigmoid_gating.py
======================= 13 passed, 2 warnings in 23.41s ========================   # test_gguf_fixed_cols_gfx906.py
================== 2 failed, 18 passed, 16 warnings in 18.87s ===================   # test_qwen3_5_gdn.py
```

**合计 66 passed / 16 failed**，失败按根因分三类：

| 根因 | 数量 | 归属 |
|---|---|---|
| Triton `error: unsupported target: 'gfx906'`（`triton/backends/amd/compiler_hcu.py` 的 `PassManager::run failed`） | 14（attention 12 + recurrent 2） | **S6**（Triton gfx906 支持固化） |
| `tests/model_executor/test_qwen3_5_gdn.py` 的测试替身过期：`SimpleNamespace` 缺 `seq_lens`、`_FakeLoader.__init__` 缺 `ignore_unexpected_prefixes` | 2 | 与 gfx906/DTK 无关的既有缺陷（任何平台都会失败） |

Triton 那条的原文（说明不是算子错、是编译器不接受 gfx906 这个 target）：

```console
/mnt/game/port/gfx906-vllm/vllm/attention/ops/triton_unified_attention.py:418:0: error: unsupported target: 'gfx906'
...
E       RuntimeError: PassManager::run failed
/usr/local/lib/python3.10/dist-packages/triton/backends/amd/compiler_hcu.py:544: RuntimeError
```

`test_fused_sigmoid_gating.py` 原本 1 失败（`fused_recurrent_gated_delta_rule_gfx906_packed_decode()
missing 2 required positional arguments: 'use_tiled_qk_head_mapping' and 'use_transposed_state'`），
是本步**修好的**：该 op 的 Python 包装（`vllm/_custom_ops.py:3151`）要求这两个参数，
而测试没传。已按层内调用者
（`vllm/model_executor/layers/fla/ops/fused_recurrent.py:482` 的默认值）补上
`use_tiled_qk_head_mapping=False` / `use_transposed_state=False`，随后 5/5 通过。

## 4. 门禁判定

| 门禁（原文） | 判定 | 证据指向 |
|---|---|---|
| `vllm._C` 编译成功 | ✅ **通过** | §3.5：`[25/28] Linking HIP shared module _C.abi3.so`，`ninja rc=0`，78,034,896 B，`IMPORT_OK`（§3.6） |
| `vllm._C` 只含 gfx906 | ✅ **通过** | §3.6：`strings` 口径 `_C.abi3.so → gfx906`；对照 vendor 轮子 `gfx906 gfx926 gfx928 gfx936` |
| DTK 26.04 rocPRIM placement-new 缺陷已解决 | ✅ **通过** | §3.2：最小复现 + shim 前 rc=1 / 后 rc=0；`-include` 只在 `ROCM_PATH MATCHES "dtk"` 时注入 |
| `_rocm_C` 编成并安装 | ✅ **通过** | §3.5：`[28/28] Linking HIP shared module _rocm_C.abi3.so`，35,130,920 B，`IMPORT_OK` |
| `_rocm_C` 只含 gfx906 | ✅ **通过** | §3.6：`_rocm_C.abi3.so → gfx906` |
| 除 mamba 外的 TU 也中招时需同样处理并记录 | ✅ **通过** | §3.1 失败清单 4 个 TU（mamba/sampler 同根因、qknorm、q_gemm）+ §3.4 第 4 类（stable 全量）；四处均已修 |
| `_rocm_C` 暴露 `LLMM1/wvSplitK/wvSplitKQ/...` | ⚠️ **按设计意图不适用** | gfx906 上它们被**有意**排除（§3.7.2 真机证明 kernel 是 `UNREACHABLE_CODE` 桩）。todolist 门禁自带豁免："`hasattr(torch.ops._rocm_C,"wvSplitK")` 允许为 False" |
| 不得出现运行期 `_rocm_C has no wvSplitK` 之类 AttributeError | ✅ **通过** | §2：`vllm/_custom_ops.py` 新增 `ROCM_SKINNY_GEMM_ENABLED` 探测，`vllm/model_executor/layers/utils.py:441` 用它短路 `LLMM1`；`scaled_mm/rocm.py:96` 对 gfx906 短路 |
| 未注册 skinny 算子且运行期已短路 | ✅ **通过** | §3.6 `ROCM_SKINNY_GEMM_ENABLED = False`；配置输出 `Only gfx1250/gfx906, skipping skinny_gemms` |
| 5 个 gfx906 专用测试通过 | 🔶 **部分通过（66 passed / 16 failed）** | §3.9：`test_gguf_fixed_cols_gfx906.py` 13/13 ✅、`test_fused_sigmoid_gating.py` 5/5 ✅（本步修好 1 个测试替身缺陷）；14 个失败是 **Triton 不接受 gfx906 target（S6 范围）**，2 个是既有测试替身过期（与 gfx906 无关） |
| `_rocm_C` 的 op 数学校验 | 🔶 **部分** | §3.7：`LLMM1` rel_err `4.413e-04` ✅（在仍注册它的 S4 版 `_rocm_C` 上）；其余三个是 MFMA-only 桩，真机证据表明**不应调用**，已在派发层摘除 |
| 非 DTK 构建路径未被注入 shim | ✅ **通过** | §3.2 负向控制两条（开关关闭后 ninja 无 `-include`；CMake 谓词对 `/opt/rocm` 为 OFF）+ §3.8 16/16 harness |
| 产物只落 `/mnt/game`；仓库无 `build/`/`.so`/`egg-info` | ✅ **通过** | 构建树 `/mnt/game/port/build-gfx906/temp`；仓库侧 `ls -d build *.egg-info vllm/*.so` 全不存在，`git status` 只有源码改动 + 归档 |
| 不碰生产容器、不替换 DTK 库、不动 `_ON_GFX9`/`on_gfx9()` | ✅ **通过** | 全程只对 `z100-build` 操作（`docker ps -a` 见 §3.0）；改动见 §2，未触碰 `_ON_GFX9`/`on_gfx9()` |
| 禁止执行自编译 HIP 可执行程序 | ✅ **遵守** | 只用 `hipcc -c` 编译、`strings` 检查、以及 python 导入扩展并派发 kernel |

**S5 总体：✅ 编译器侧目标全部达成**（`_C` + `_rocm_C` 编成、只含 gfx906、rocPRIM 缺陷解决、
非 DTK 路径不受影响、skinny GEMM 与 custom paged attention 在 gfx906 上正确摘除）；
**5 个测试门禁 🔶 部分通过**，剩余 14 个失败明确落在 S6（Triton gfx906 target 支持），
2 个失败是与本步无关的既有测试替身缺陷。

## 5. 未完成项 / 阻塞与原因

### 5.1 `_C_stable_libtorch` / `_moe_C_stable_libtorch` 仍未编成（与本步无关的既有缺陷）
先解决第 4 类缺陷后，`_C_stable_libtorch` 前进到第 24 步（S4 时它连第一步都过不去），
然后停在 `csrc/libtorch_stable/cuda_view.cu`：

```console
FAILED: CMakeFiles/_C_stable_libtorch.dir/csrc/libtorch_stable/hip_view.hip.o
.../csrc/libtorch_stable/hip_view.hip:30:34: error: no member named 'layout' in 'torch::stable::Tensor'
   30 |   const auto layout = cpu_tensor.layout();
```

（`hip_view.hip` 是 `cuda_view.cu` 按 hipify 的 `cuda→hip` 词替换生成的名字。）
原因：稳定 ABI 的 `torch::stable::Tensor::layout()` 在镜像的 torch **2.10.0** 里还不存在，
而 `CMakeLists.txt:1282` 给这两个目标设的 `TORCH_TARGET_VERSION=0x020B…` 表示"按 torch 2.11
的 ABI 编译"（S4 §5.5 已记录"requirements 固定 2.11.0+das 但镜像是 2.10.0"）。属
**上游代码 + torch 版本**的问题，不在 S5 的 csrc/DTK 工具链范围内，也未由本次任何改动引入
（`git status` 显示 `csrc/libtorch_stable/cuda_view.cu` 未被本步改动）。镜像升到 torch 2.11
或该文件改用 2.10 可用的 API 即可。

### 5.2 5 个测试里的 14 个失败归 S6
全部是 Triton 编译器报 `unsupported target: 'gfx906'`。S6 的设计文档
（`docs/design/z100-06-triton-gfx906.md`）门禁正好是"Triton 最小探针 + fork 的 2 个
unified-attention gfx906 测试通过"，故此处不再重复处置，只把原始证据留给 S6。

### 5.3 5 个测试里的 2 个失败是既有测试替身过期
`tests/model_executor/test_qwen3_5_gdn.py`：
`test_gdn_mixed_batch_builds_prefill_only_metadata` 的 `SimpleNamespace` 缺 `seq_lens`；
`test_qwen3_5_load_uses_native_checkpoint_mapper` 的 `_FakeLoader` 不接受
`ignore_unexpected_prefixes`。两者都与 gfx906/DTK 无关（任何平台都会失败），
修它们需要对齐实现的当前契约，超出本步"gfx906 内核"范围，故只记录。

### 5.4 与本步无关但必须记录的镜像依赖缺口
镜像里的 Python 依赖比本 fork 的 `requirements/*.txt` 旧，导致 `tests/conftest.py` 无法导入：
`tblib` 缺失、`mistral_common 1.10.0 < 1.11.5`、`openai 2.21.0`（代码要 3.x 的
`NamespaceTool`）、`xgrammar 0.1.33 < 0.2.1`。本次为跑测试在**构建容器内**补装了
`tblib`、`mistral_common==1.12.0`、`openai==3.26.0`、`xgrammar==0.2.8`+`apache-tvm-ffi`
（纯 Python，未触碰 DTK/torch）。**生产容器 `qwen38-vllm` 未受任何影响。** 是否把这些
写进 `requirements/dcu*.txt` / 镜像，请 Lead 决定（可能与 S4 §5.5 的 torch 版本问题合并处理）。

### 5.5 明确"未执行"的项
- `_C_stable_libtorch` / `_moe_C_stable_libtorch` 的"编成"验证（阻塞见 §5.1）。
- `triton_kernels` 目标的构建（未在本步目标内；S4 已备好 `TRITON_KERNELS_SRC_DIR`）。
- 生产容器 `qwen38-vllm` 的任何操作（只做过 `docker ps`/`docker stats` 只读观察）。
- 未跑任何自编译 HIP 可执行程序。

## 6. 对后续步骤的影响

**S6/S7/S8 可以直接复用这套已带 `_C` + `_rocm_C` 的构建副本与容器**：

| 资源 | 路径 | 状态 |
|---|---|---|
| 构建容器 | `z100-build`（未 `--rm`，`--memory=8g`） | 运行中；本次额外装了 `tblib`/`mistral_common 1.12`/`openai 3.26`/`xgrammar 0.2.8`/`apache-tvm-ffi` |
| 构建副本 | `/mnt/game/port/gfx906-vllm` | `vllm/_C.abi3.so`（78.0 MB）+ `vllm/_rocm_C.abi3.so`（35.1 MB）已安装，**都只含 gfx906** |
| 持久构建目录 | `/mnt/game/port/build-gfx906/temp` | 增量；`_C`/`_rocm_C` 已是最新，`_C_stable_libtorch` 仍有 §5.1 的既有失败 |
| 环境/构建脚本 | `/mnt/game/port/{setenv.sh,build.sh,build-keepgoing.sh,gate.sh}` | 直接可用 |
| S5 新脚本 | `tools/dcu/s5_gate.sh`、`tools/dcu/check_rocm_c_ops.py` | 门禁与 `_rocm_C` 数值自检，可直接粘贴 |
| 原始日志 | `/mnt/game/port/{build,build-s5,build-s5c,s5-pytest-final}.log` | 本档所有引用的原文 |

**前提被修正/新增的几点**：

1. **DTK 构建现在多了一条全局编译选项**：`-include <repo>/csrc/dtk_rocprim_compat.h`
   （`VLLM_GPU_FLAGS` 级别，作用于所有 HIP 目标）。S6/S7/S8 若要手工编某个 `.hip`，
   记得带上它，否则会重新撞上 rocPRIM placement-new。
2. **hipify 输出被后处理**：`cmake/hipify.py` 会把生成文件里的
   `#include <ATen/dtk_macros.h>` 换成 `#define __DTK_ARCH__ 520`。以后若新增依赖该头的
   代码，`__DTK_ARCH__` 仍然可用，但**不要**再依赖 `ATen/dtk_macros.h` 本身。
3. **gfx906 上 `_rocm_C` 只剩 `paged_attention`，且它也不可调用**：
   `use_rocm_custom_paged_attention()` 现在对 gfx906 返回 `False`（`vllm/platforms/rocm.py:479`），
   decode attention 必须走 Triton 回退 —— 这让 **S6 成为 S8/S9 的硬前置**（Triton 不支持
   gfx906 target 时，attention 没有可用后端）。
4. **`ROCM_SKINNY_GEMM_ENABLED` 是新的公共探测点**（`vllm/_custom_ops.py`），
   `tools/dcu/env.sh` 里的 `VLLM_ROCM_USE_SKINNY_GEMM=0` 仍是运行期双保险。
5. **5 个测试要用 `--noconftest` + 先补镜像依赖**才能跑（§3.9/§5.4）；S6 跑那两个
   unified-attention 测试时会遇到同样问题。
6. `_C_stable_libtorch` 的既有阻塞（§5.1）会让任何"全量 `ninja`（默认全部目标）"立刻失败，
   想编 `_C`/`_rocm_C` 请**显式只列这两个目标**（本次就是这么做成功的）。

## 7. 复现方式

> 前提：宿主 docker 可用、本地有镜像
> `harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`，
> 且 S4 留下的容器 `z100-build` 与 `/mnt/game/port/*` 仍在（S5 全程复用，未重建容器）。
> **宿主对 `/mnt/game` 是只读的，所有 `/mnt/game` 下的写入都要经 `docker exec`。**

```bash
# 0) 停掉可能还在跑的 ninja（会锁构建目录），清掉上一轮 hipify 的残留文件
docker exec z100-build bash -lc 'pkill -f build-keepgoing; pkill -f "ninja -C"; sleep 2; true'

# 1) 同步工作区到构建副本（排除项必须锚定）
docker exec z100-build bash -lc '
  rsync -a --exclude=/build --exclude=/.deps --exclude="/*.egg-info" --exclude=/.venv         /src/ /mnt/game/port/gfx906-vllm/'

# 2) 配置 + 只编 _C 与 _rocm_C（**不要**编默认全部目标，见 §5.1）
docker exec z100-build bash -lc 'source /mnt/game/port/setenv.sh
  cd "$REPO_DIR"
  cmake --build /mnt/game/port/build-gfx906/temp -j "$MAX_JOBS" --target _C _rocm_C'
# 期望输出含：-- DTK rocPRIM placement-new workaround enabled: -include ...
#             -- Only gfx1250/gfx906, skipping skinny_gemms
#             [N/N] Linking HIP shared module _C.abi3.so / _rocm_C.abi3.so
docker exec z100-build bash -lc 'source /mnt/game/port/setenv.sh; cd "$REPO_DIR"
  cmake --install /mnt/game/port/build-gfx906/temp --prefix "$REPO_DIR" --component _C
  cmake --install /mnt/game/port/build-gfx906/temp --prefix "$REPO_DIR" --component _rocm_C'

# 3) S5 门禁（import + arch + _rocm_C 注册面）
docker exec z100-build bash "$REPO_DIR/tools/dcu/s5_gate.sh"    # 或用完整路径 /mnt/game/port/gfx906-vllm/tools/dcu/s5_gate.sh

# 4) _rocm_C 数值 / 注册面自检（真机）
docker exec z100-build bash -lc 'source /mnt/game/port/setenv.sh; cd "$REPO_DIR"
  PYTHONUNBUFFERED=1 python tools/dcu/check_rocm_c_ops.py; echo "exit=$?"'

# 5) 非 DTK 行为不变（宿主侧，GPU-free）
python3 /mnt/game/port/s4-harness/check_setup_dtk.py /home/lyh/z100-vllm   # 期望 16/16

# 6) 5 个 gfx906 测试（先补镜像缺的纯 Python 依赖，再 --noconftest）
docker exec z100-build bash -lc 'pip install -q tblib "mistral_common==1.12.0"   "openai==3.26.0" "xgrammar==0.2.8" apache-tvm-ffi'
docker exec -d z100-build bash -lc 'source /mnt/game/port/setenv.sh; cd "$REPO_DIR"
  for f in tests/kernels/attention/test_triton_unified_attention_gfx906.py            tests/kernels/mamba/test_fused_recurrent_packed_decode.py            tests/kernels/mamba/test_fused_sigmoid_gating.py            tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py            tests/model_executor/test_qwen3_5_gdn.py; do
    echo "########## $f"
    python -c "import torch, pytest, sys; sys.exit(pytest.main(["-v","--no-header","--noconftest","-p","no:cacheprovider","$f"]))"
    echo "##### rc=$? $f"
  done > /mnt/game/port/s5-pytest-final.log 2>&1'
docker exec z100-build bash -lc 'grep -E "passed" /mnt/game/port/s5-pytest-final.log'
# 期望：27/12、3/2、5/0、13/0、18/2（失败根因见 §3.9/§5.2/§5.3）
```

单独复现 rocPRIM 缺陷与 shim 的 A/B：

```bash
docker exec z100-build bash -lc 'cat > /tmp/p1.hip <<"EOF"
#include <rocprim/types.hpp>
__global__ void k(int* o) {
  __shared__ rocprim::uninitialized_array<int, 4> a;
  a.emplace(0, 7);
  if (threadIdx.x == 0) o[0] = a.get_unsafe_array()[0];
}
EOF
source /mnt/game/port/setenv.sh >/dev/null 2>&1
hipcc -std=c++17 --offload-arch=gfx906 -c /tmp/p1.hip -o /tmp/p1.o; echo "no-shim rc=$?"
hipcc -std=c++17 --offload-arch=gfx906   -include /mnt/game/port/gfx906-vllm/csrc/dtk_rocprim_compat.h   -c /tmp/p1.hip -o /tmp/p1s.o; echo "with-shim rc=$?"'
# 期望：no-shim rc=1（operator new 错误）、with-shim rc=0
```

验证 shim 的开关对非 DTK 构建无效（不改共享构建目录）：

```bash
docker exec z100-build bash -lc 'mkdir -p /mnt/game/port/probe-cfg4; source /mnt/game/port/setenv.sh
  cd /mnt/game/port/probe-cfg4; rm -rf CMakeCache.txt CMakeFiles
  cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_TARGET_DEVICE=rocm     -DVLLM_PYTHON_EXECUTABLE=$(which python) -DROCM_PATH=$ROCM_PATH     -DVLLM_DTK_ROCPRIM_COMPAT=0 -DCMAKE_JOB_POOL_COMPILE:STRING=compile     -DCMAKE_JOB_POOLS:STRING=compile=2 /mnt/game/port/gfx906-vllm >/tmp/cfg4.log 2>&1
  grep "rocPRIM placement-new workaround" /tmp/cfg4.log
  grep -m1 dtk_rocprim_compat build.ninja || echo NO-SHIM-IN-NINJA'
# 期望：disabled + NO-SHIM-IN-NINJA
```

## 8. 参考

- 设计文档：[docs/design/z100-05-csrc-kernels.md](../design/z100-05-csrc-kernels.md)
- 前序归档：[docs/result/z100-04-build-system-dtk-result.md](z100-04-build-system-dtk-result.md)
  （§3.4 构建环境、§3.8 6 行最小复现、§3.9 `_rocm_C` 首次编成、§5.1 `_C` 阻塞、§6 可复用资源）
- Todolist：[docs/z100-todolist.md](../z100-todolist.md)（全局硬约束 8/9/10 条、S5 门禁措辞修正）
- 真机记录：[docs/hygon_z100.md](../hygon_z100.md) §1.1（容器要点与高危坑）
- 后续步骤：S6 [docs/design/z100-06-triton-gfx906.md](../design/z100-06-triton-gfx906.md)
  （本档 §5.2 的 14 个失败归它）、S8 [docs/design/z100-08-attention-and-quant-paths.md](../design/z100-08-attention-and-quant-paths.md)、
  S9 [docs/design/z100-09-acceptance-and-bench.md](../design/z100-09-acceptance-and-bench.md)
- 关键源码位置：
  - `csrc/rocm/attention.cu:3293-3307`（非 gfx9 的 `UNREACHABLE_CODE` 桩）
  - `csrc/rocm/skinny_gemms.cu:580-590`（同上）
  - `vllm/platforms/rocm.py:264`（`_ON_CDNA` 的 `"gfx9"` 子串匹配）、`:479`（`use_rocm_custom_paged_attention`）
  - `vllm/_custom_ops.py:21-29`（`ROCM_SKINNY_GEMM_ENABLED`）
  - `CMakeLists.txt:260-299`（DTK shim 开关）、`:1553-1584`（skinny GEMM 排除）
  - `cmake/hipify.py:64-104`（`ATen/dtk_macros.h` 后处理）
  - DTK 侧：`/opt/dtk-26.04-DCC2602-0317/llvm/lib/clang/18/include/cuda_wrappers/new`、
    `/opt/dtk-26.04-DCC2602-0317/include/rocprim/types/uninitialized_array.hpp:78`、
    `/usr/local/lib/python3.10/dist-packages/torch/utils/hipify/hipify_python.py:63`、
    `/usr/local/lib/python3.10/dist-packages/torch/include/ATen/dtk_macros.h`
