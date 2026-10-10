# S4c 落地 S4b 两项修复并在新栈用全量 ninja 复验 — 执行结果

> 归档时间：2026-10-08
> 执行者：子 agent（S4c）
> 对应设计文档：docs/design/z100-04b-stack-upgrade.md
> 前一档：[docs/result/z100-04b-stack-upgrade-result.md](z100-04b-stack-upgrade-result.md)
> 状态：✅ 完成
>
> **本档是 S4b 的第二档，承接第一档 §5.1**（"两项修复只在构建副本/构建树里，仓库零改动，需 Lead 决策落地"）。
> 第一档的全部结论（新栈上 4 扩展编成、只含 `gfx906`、S5 的 `cuda_view` `layout()` 阻塞消失、
> triton 3.6 仍不支持 gfx906 ⇒ S6 走 VEGA 补丁 + 源码重编）**保持不变**；本档只做两件事：
> **(a)** 把 §5.1 的 ①`cmake/hipify.py`、②`csrc/libtorch_stable/fused_qknorm_rope_kernel.cu` 落进仓库；
> **(b)** 同步进 `z100-build-211` 构建副本后，用**裸 `ninja`（默认 `all`）**在新栈重建全部 4 个扩展并复跑门禁
> （实际跑了两次：§3.4 主证据 + §3.10 可复现性/收尾），从而解除 S5 §6.6 遗留的"不要裸跑全量 ninja"限制。
>
> 硬约束遵守情况：未运行任何自编译 HIP 独立可执行程序（只用 `hipcc` 经 ninja、`strings`、python import/探针）；
> 未触碰生产容器 `qwen38-vllm`（见 §3.9 观察）；未动 `z100-build`（2.10 基线）与 `/mnt/game/port/gfx906-vllm`；
> 未升级/重建 triton（S6 路线，另有 `z100-triton` 容器在跑）；产物只落 `/mnt/game`；`MAX_JOBS=4`、容器 `--memory=8g` 未触限。

## 1. 目标与验收标准

### 1.1 设计文档门禁原文（`docs/design/z100-04b-stack-upgrade.md` §验收）

- `torch`/`triton` 版本判据（用 `importlib.metadata`，不用 `torch.__version__`）；
- `python -c "import torch, vllm, vllm._C, vllm._rocm_C; print('IMPORT_OK')"` 成功（**必须先 `import torch`**）；
- 新 `.so` 只含 `gfx906`：`strings <so> | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sort -u`；
- `ROCM_SKINNY_GEMM_ENABLED` 可导入且为 `False`；
- 回归：gguf fixed-cols **13/13**、fused sigmoid gating **5/5**（`python -c "import torch, pytest; ..."` 起跑）。

### 1.2 本步（S4c）任务书追加的验收标准

1. `cmake/hipify.py` 的 `CUDA_MACROS_MAP` 元组化修复落进仓库，且**对 2.10 的元组形式同样安全**（两种形式都能正确解析）；
2. `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu` 补上与孪生文件 `csrc/fused_qknorm_rope_kernel.cu` 逐字一致的
   `&& !defined(HIP_ENABLE_WARP_SYNC_BUILTINS)` 守卫；
3. 两项同步进 `z100-build-211` 的构建副本后重建，**允许并执行一次全量 `ninja`（默认所有目标）**，四个扩展全部走通；
4. `import torch, vllm, vllm._C, vllm._rocm_C, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch` 全通过
   （`IMPORT_OK_ALL4`）；
5. 四个 `.so` 的 `strings | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sort -u` → 只有 `gfx906`；
6. 记录全量 `ninja` 的 rc 与耗时（证明"全量 ninja 可走通"）。

## 2. 实际改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `cmake/hipify.py` | **新增 17-28 行**（原 `from torch.utils.hipify.hipify_python import ...` 由 17 行下移到 30 行） | 在导入 `hipify_python` **之前**把 DTK torch 2.11 的裸字符串 `CUDA_MACROS_MAP` 值就地补成 1 元组；`isinstance(value, str)` 守卫使 2.10 的元组原样通过 |
| `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu` | **42-50 行**（原 42 行那条 `#if` 被替换；`#endif` 由 50 行变 58 行） | 补上 `&& !defined(HIP_ENABLE_WARP_SYNC_BUILTINS)` 守卫（49-50 行）与 S5 在孪生文件写的 7 行英文注释（42-48 行）；与 `csrc/fused_qknorm_rope_kernel.cu:41-49` **逐字一致** |

改动 A 的确切 diff（`cmake/hipify.py`；**本步新增的就是这 13 行**：

```diff
--- a/cmake/hipify.py
+++ b/cmake/hipify.py
@@ -14,6 +14,19 @@ import argparse
 import os
 import shutil
 
+from torch.utils.hipify import cuda_to_hip_mappings as _cuda_macros
+
+# DTK's torch 2.11 wheel stores CUDA_MACROS_MAP values as bare strings where
+# upstream torch (and DTK's torch 2.10) stores 1-tuples. hipify_python.py does
+# `dst = value[0]`, so a bare string degrades to its first character
+# (CUDA_VERSION -> "D") and every hipified use of that macro breaks. Normalize
+# to the upstream tuple shape before hipify_python is imported, because it
+# snapshots the map while it is being imported. The isinstance guard makes this
+# a no-op for the already-correct 2.10 tuples.
+for _macro, _value in list(_cuda_macros.CUDA_MACROS_MAP.items()):
+    if isinstance(_value, str):
+        _cuda_macros.CUDA_MACROS_MAP[_macro] = (_value,)
+
 from torch.utils.hipify.hipify_python import get_hip_file_path, hipify
```

> **注意**：`cmake/hipify.py` 在本步之前**已含 S5 的未提交改动**（`_neutralize_dtk_macros_include`，即 `git diff` 里的另一段 hunk）。
> 本步只加了上面 13 行；`git diff --stat` 显示的 63 行新增里其余部分是 S5 遗留，不是本步引入。

改动 B 的确切 diff（`csrc/libtorch_stable/fused_qknorm_rope_kernel.cu`）：

```diff
--- a/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu
+++ b/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu
@@ -39,7 +39,15 @@
 #ifdef USE_ROCM
   #define FINAL_MASK 0xffffffffffffffffULL
 
-  #if defined(HIP_VERSION) && HIP_VERSION < 70000000
+  // HIP before 7.0 does not document __syncwarp, but the definition lives in
+  // <hip/amd_detail/amd_warp_sync_functions.h> behind
+  // HIP_ENABLE_WARP_SYNC_BUILTINS. DTK 26.04 reports HIP 6.3 and defines that
+  // macro, so it already declares __syncwarp and the fallback below becomes a
+  // hard redefinition error. Key off the guard macro the HIP header itself
+  // uses; when HIP_ENABLE_WARP_SYNC_BUILTINS is unset (community ROCm 6.x) the
+  // fallback is emitted exactly as before.
+  #if defined(HIP_VERSION) && HIP_VERSION < 70000000 && \
+      !defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
 // On ROCm versions before 7.0, __syncwarp isn't defined. The below
 // implementation is copy/pasted from the implementation in ROCm 7.0
 __device__ inline void __syncwarp() {
```

对齐核对（两个孪生文件的 `#ifdef USE_ROCM` … `#else` 段逐字节相同）：

```console
$ diff <(awk '/#ifdef USE_ROCM/,/^#else/' csrc/fused_qknorm_rope_kernel.cu) \
       <(awk '/#ifdef USE_ROCM/,/^#else/' csrc/libtorch_stable/fused_qknorm_rope_kernel.cu) && echo IDENTICAL
IDENTICAL
```

改动后的仓库哈希：

```console
$ md5sum cmake/hipify.py csrc/libtorch_stable/fused_qknorm_rope_kernel.cu
00245a1752fa8d801d0104784a04604e  cmake/hipify.py
9c694f6a35a705b7f0c3b2abf58a317c  csrc/libtorch_stable/fused_qknorm_rope_kernel.cu
$ git diff --stat -- cmake/hipify.py csrc/libtorch_stable/fused_qknorm_rope_kernel.cu
 cmake/hipify.py                                  | 63 ++++++++++++++++++++++++
 csrc/libtorch_stable/fused_qknorm_rope_kernel.cu | 10 +++-
 2 files changed, 72 insertions(+), 1 deletion(-)
$ git status --porcelain | wc -l
36      # 全部是 S1–S5 的既有改动/未跟踪文档；本步只改上面两个文件，没有新增其他仓库文件
```

## 3. 执行过程与原始证据

### 3.1 根因再确认（改动 A）：DTK 的 torch 2.11 打包不一致 + 消费方在导入期快照

```console
# torch 2.11（z100-build-211）
$ python -c "
from torch.utils.hipify import cuda_to_hip_mappings as cm
import torch.utils.hipify.hipify_python as hp
print('CUDA_MACROS_MAP   :', repr(dict(cm.CUDA_MACROS_MAP)))
print('hp.CUDAMACROS_MAP :', repr(dict(hp.CUDAMACROS_MAP)))
print('hp.PYTORCH_MAP get:', repr(hp.PYTORCH_MAP.get('CUDA_VERSION')), repr(hp.PYTORCH_MAP.get('__CUDA_ARCH__')))"
CUDA_MACROS_MAP   : {'CUDA_VERSION': 'DTK_VERSION', '__CUDA_ARCH__': '__DTK_ARCH__'}
hp.CUDAMACROS_MAP : {'CUDA_VERSION': 'D', '__CUDA_ARCH__': '_'}      <- value[0] 取到首字符
hp.PYTORCH_MAP get: 'TORCH_HIP_VERSION' None

# torch 2.10（z100-build，只读对照）
$ python -c "import torch, importlib.metadata as m; print(m.version('torch'))"
2.10.0+das.opt1.dtk2604.20260325.g6b060a
$ python -c "
from torch.utils.hipify import cuda_to_hip_mappings as cm
import torch.utils.hipify.hipify_python as hp
print(repr(dict(cm.CUDA_MACROS_MAP))); print(dict(hp.CUDAMACROS_MAP))"
{'CUDA_VERSION': ('DTK_VERSION', 1337), '__CUDA_ARCH__': ('__DTK_ARCH__', 1337)}   <- 元组，正常
{'CUDA_VERSION': 'DTK_VERSION', '__CUDA_ARCH__': '__DTK_ARCH__'}
```

补全 S4b §3.7.2 只差最后一块的机制（本步新查明，对后续同类缺陷有用）：

- `hipify_python.py:805-817` 的 `for src, value in CUDA_MACROS_MAP.items(): dst = value[0]; ... CUDAMACROS_MAP[src] = dst`
  是**模块级**代码，`hipify_python` 一被导入就把错误值固定进 `CUDAMACROS_MAP`；
  因此修复必须在**导入它之前**完成 —— 这正是 S4b 的 wrapper（先改 map 再 `runpy` 跑 `hipify.py`）能生效、
  而"先 import hipify_python 再改 `cuda_to_hip_mappings.CUDA_MACROS_MAP`"无效的原因。
- `torch.utils.hipify.cuda_to_hip_mappings` 只 import `collections/os`，`torch.utils.hipify.__init__` 只 import `version`，
  所以"先 import 映射表、再 import `hipify_python`"的写法的顺序是有保证的（本步把它放在 `hipify.py:17`）。
- 为什么只有 `CUDA_VERSION` 炸而 `__CUDA_ARCH__` 正常：`CUDA_VERSION` 同时在 `CUDA_TO_HIP_MAPPINGS`
  （`cuda_to_hip_mappings.py:3400/3429` → `'TORCH_HIP_VERSION'`）里，所以它在 `PYTORCH_TRIE` 里、会被宏替换 pass 命中，
  随后被 `{**PYTORCH_MAP, **CUDAMACROS_MAP}` 用 `'D'` 覆盖；`__CUDA_ARCH__` 不在 trie 里，由 DTK 的
  `CUDATODTK_DICT` 另外替换，所以两种情况下都正确。

### 3.2 改动 A 的端到端探针（同环境、同输入，只换脚本）

探针源（新建，`/mnt/game/port/s4c-probe/proj/probe_macro.cu`）：

```cuda
// S4c probe: does hipify map CUDA_VERSION to DTK_VERSION (correct) or to "D" (bug)?
#include <cuda_runtime.h>

#if CUDA_VERSION >= 11000
int probe_cuda_version_ge_11000;
#endif
#if defined(__CUDA_ARCH__)
int probe_cuda_arch;
#endif
```

```console
# 对照组：未修版（= S4b rsync 进来的构建副本脚本，md5 ed6a6d2afd4f2e9982b3320ca28aaa70）
$ cd /mnt/game/port/s4c-probe/out_control && python /mnt/game/port/gfx906-vllm-211/cmake/hipify.py \
    -p /mnt/game/port/s4c-probe/proj -o /mnt/game/port/s4c-probe/out_control probe_macro.cu
$ grep -nE "^(#if|#ifdef|#include|int )" out_control/probe_macro.hip
6:#if defined(USE_ROCM)
12:#include <hip/hip_runtime.h>
14:#if D >= 11000                      <-- 缺陷复现
15:int probe_cuda_version_ge_11000;
17:#if defined(__DTK_ARCH__)
18:int probe_cuda_arch;

# 修复组：仓库版（/src/cmake/hipify.py = 00245a1752fa8d801d0104784a04604e）
$ cd /mnt/game/port/s4c-probe/out_fixed && python /src/cmake/hipify.py \
    -p /mnt/game/port/s4c-probe/proj -o /mnt/game/port/s4c-probe/out_fixed probe_macro.cu
$ grep -nE "^(#if|#ifdef|#include|int )" out_fixed/probe_macro.hip
6:#if defined(USE_ROCM)
12:#include <hip/hip_runtime.h>
14:#if DTK_VERSION >= 11000            <-- 修复
15:int probe_cuda_version_ge_11000;
17:#if defined(__DTK_ARCH__)
18:int probe_cuda_arch;

$ diff out_control/probe_macro.hip out_fixed/probe_macro.hip
14c14
< #if D >= 11000
---
> #if DTK_VERSION >= 11000
```

"对 2.10 元组形式同样安全"的证据（同环境用 2.10 的元组形状喂给修复后的脚本，输出仍正确、没有被二次包装）：

```console
$ python -c "
from torch.utils.hipify import cuda_to_hip_mappings as cm
for k, v in list(cm.CUDA_MACROS_MAP.items()):
    cm.CUDA_MACROS_MAP[k] = (v, 1337)          # 模拟 torch 2.10 的 ('DTK_VERSION', 1337)
import importlib.util
spec = importlib.util.spec_from_file_location('repo_hipify', '/src/cmake/hipify.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(repr(dict(cm.CUDA_MACROS_MAP)))"
{'CUDA_VERSION': ('DTK_VERSION', 1337), '__CUDA_ARCH__': ('__DTK_ARCH__', 1337)}   <- 守卫放行，未被改写成 (('DTK_VERSION',1337),)
```

> 说明：2.10 容器（`z100-build`）本步**未重编**（硬约束），所以这里的 2.10 安全性是
> "元组形状实测 + `isinstance` 守卫"的组合论证，而不是 2.10 上的整轮重建验证。

### 3.3 同步进构建副本 + 撤掉 S4b 的 hipify wrapper

```console
$ docker exec z100-build-211 bash -lc '
    cp -p /src/cmake/hipify.py /mnt/game/port/gfx906-vllm-211/cmake/hipify.py
    cp -p /src/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu \
          /mnt/game/port/gfx906-vllm-211/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu
    md5sum /src/cmake/hipify.py /mnt/game/port/gfx906-vllm-211/cmake/hipify.py \
           /src/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu \
           /mnt/game/port/gfx906-vllm-211/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu'
00245a1752fa8d801d0104784a04604e  /src/cmake/hipify.py
00245a1752fa8d801d0104784a04604e  /mnt/game/port/gfx906-vllm-211/cmake/hipify.py
9c694f6a35a705b7f0c3b2abf58a317c  /src/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu
9c694f6a35a705b7f0c3b2abf58a317c  /mnt/game/port/gfx906-vllm-211/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu

# 孪生文件守卫段逐字节核对
$ diff <(awk '/#ifdef USE_ROCM/,/^#else/' /src/csrc/fused_qknorm_rope_kernel.cu) \
       <(awk '/#ifdef USE_ROCM/,/^#else/' /src/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu) && echo IDENTICAL
IDENTICAL

# 恢复 hipify 直调（wrapper 不再需要；把 S4b 的 wrapper 版 build.ninja 留档为 .s4c-wrapper-patched）
$ cd /mnt/game/port/build-211/temp
$ cp -p build.ninja build.ninja.s4c-wrapper-patched
$ cp -p build.ninja.s4b-orig build.ninja
$ grep -c "hipify-fix-211" build.ninja
0
$ ls -l build.ninja*
-rw-r--r-- 1 root root 220234 Oct  8 06:53 build.ninja                     <- = build.ninja.s4b-orig（直调 /mnt/game/port/gfx906-vllm-211/cmake/hipify.py）
-rw-r--r-- 1 root root 220234 Oct  8 06:53 build.ninja.s4b-orig
-rw-r--r-- 1 root root 220267 Oct  8 07:15 build.ninja.s4c-wrapper-patched   <- S4b 的 hipify-fix-211.py wrapper 版（证据留存）
```

`cmake/hipify.py` 是 `hipify_all` 自定义命令的输入（`build.ninja:950` 第一项），但**不在** `RERUN_CMAKE` 的输入列表里，
所以改它只会让 hipify 重跑，不会触发重新 configure —— 本次全程没有重跑 configure，构建树仍是 S4b 那一份。

### 3.4 全量 `ninja`（裸 ninja，默认 `all`）：rc=0，17 分 32 秒

构建前的 plan（`ninja -n`）证明这次是真的"从零重编"，且**不含任何 triton 步骤**：

```console
$ cd /mnt/game/port/build-211/temp && ninja -n | sed -E "s/^\[([0-9]+)\/([0-9]+)\] //" | nl -ba | sed -n "1,6p;60,63p"
     1  Building CXX object CMakeFiles/cumem_allocator.dir/csrc/cumem_allocator.cpp.o
     2  Running hipify on all extension source files.
     3  Linking CXX shared module cumem_allocator.abi3.so
     4  Building HIP object CMakeFiles/_C.dir/csrc/mamba/mamba_ssm/selective_scan_fwd.hip.o
     5  Building HIP object CMakeFiles/_C.dir/csrc/mamba/gfx906_decode_kernels.hip.o
     6  Building HIP object CMakeFiles/_C.dir/csrc/cache_kernels.hip.o
    60  Building HIP object CMakeFiles/_moe_C_stable_libtorch.dir/csrc/libtorch_stable/moe/moe_wna16.hip.o
    61  Linking HIP shared module _moe_C_stable_libtorch.abi3.so
    62  Linking HIP shared module _rocm_C.abi3.so
    63  Linking HIP shared module _C_stable_libtorch.abi3.so
$ ninja -n | grep -i triton | wc -l
0
```

> 之所以是 63 步全量：`CMakeFiles/hipify_all` 这个 stamp **永远不会被创建**（见 §3.10 的预存在特性），
> 所以 ninja 每次都把 `hipify_all` 及全部 53 个 `.hip` 判为脏、连带四个扩展的所有 TU 重编 ——
> 这正好让"裸 ninja 一次走通 4 个扩展"成为强证据（不是空跑）。
> （日志里的分母从 `/63` 变 `/62` 只是 ninja 按已完成边重算总数：`cumem_allocator` 的两步先完成、hipify 之后重新编号。）

命令与结果（`build-211.sh` 不传 target ⇒ `cmake --build <dir> -j 4` = 默认 `all`）：

```console
$ MAX_JOBS=4 LOG=/mnt/game/port/s4c-full-ninja.log bash /mnt/game/port/build-211.sh
# 日志头（"S4b build" 是 build-211.sh 里写死的 echo 文本，本步未改脚本）：
===== S4b build 2026-10-08T07:31:37+08:00 targets:  MAX_JOBS=4 =====
[1/63] Building CXX object CMakeFiles/cumem_allocator.dir/csrc/cumem_allocator.cpp.o
[2/63] Linking CXX shared module cumem_allocator.abi3.so
[3/63] Running hipify on all extension source files.
...
Successfully preprocessed all matching files.
Total number of unsupported CUDA function calls: 0
Total number of replaced kernel launches: 309
...
[33/62] Linking HIP shared module _C.abi3.so
...
[55/62] Linking HIP shared module _C_stable_libtorch.abi3.so
...
[59/62] Building HIP object CMakeFiles/_rocm_C.dir/csrc/rocm/attention.hip.o
[60/62] Linking HIP shared module _rocm_C.abi3.so
[61/62] Building HIP object CMakeFiles/_moe_C_stable_libtorch.dir/csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.hip.o
[62/62] Linking HIP shared module _moe_C_stable_libtorch.abi3.so
----- ninja rc=0 2026-10-08T07:49:09+08:00 -----
```

> 说明：上面最后那行 `----- ninja rc=… -----` 来自 `build-211.sh` 的 stdout（`tee` 只捕获 `cmake --build` 的输出，
> 所以 `.log` 文件里没有这行）；`rc=0` 同时被驱动脚本独立记录成下一段的 `S4C_END … rc=0`。

```console
# 计时（宿主侧 wrapper 记录，容器内时间戳）：
S4C_START 2026-10-08T07:31:37+08:00 epoch=1791415897
S4C_END   2026-10-08T07:49:09+08:00 epoch=1791416949 rc=0 wall_s=1052      # = 17 min 32 s

# 计数（长输出见 /mnt/game/port/s4c-full-ninja.log，247 行）：
error:           0
FAILED:          0
build stopped:   0
ninja: warning:  0
[62/62] 收尾：Linking HIP shared module _moe_C_stable_libtorch.abi3.so
```

两次修复在生成树里的效果（同一构建树内直接看产物，不再靠 wrapper 打印）：

```console
$ grep -n "DTK_VERSION >= 12090" .../build-211/temp/csrc/libtorch_stable/activation_kernels.hip | head -2
265:      (DTK_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128)            \
275:    if (DTK_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128) {         \
$ grep -rn " D >= 12090\|defined(D)" .../build-211/temp/csrc/ | head -3
（空：无裸 'D' 残留）
$ sed -n "52,54p" .../build-211/temp/csrc/libtorch_stable/fused_qknorm_rope_kernel.hip
  #if defined(HIP_VERSION) && HIP_VERSION < 70000000 && \
      !defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
// On ROCm versions before 7.0, __syncwarp isn't defined. The below
```

### 3.5 安装四个扩展 + 门禁原文

```console
$ docker exec z100-build-211 bash /mnt/game/port/install-211.sh _C _rocm_C _C_stable_libtorch _moe_C_stable_libtorch
--- install component _C / _rocm_C / _C_stable_libtorch / _moe_C_stable_libtorch
-- Installing: /mnt/game/port/gfx906-vllm-211/vllm/_C_stable_libtorch.abi3.so
-- Set non-toolchain portion of runtime path of ".../_C_stable_libtorch.abi3.so" to ""
-- Installing: /mnt/game/port/gfx906-vllm-211/vllm/_moe_C_stable_libtorch.abi3.so
-- Set non-toolchain portion of runtime path of ".../_moe_C_stable_libtorch.abi3.so" to ""
-rw-r--r-- 1 root root 78102728 .../vllm/_C.abi3.so
-rw-r--r-- 1 root root 40688656 .../vllm/_C_stable_libtorch.abi3.so
-rw-r--r-- 1 root root 29748928 .../vllm/_moe_C_stable_libtorch.abi3.so
-rw-r--r-- 1 root root 35156368 .../vllm/_rocm_C.abi3.so
```

`gate-211.sh` 原始输出（存 `/mnt/game/port/gate-211-s4c.log`，逐字复制）：

```console
===== S4b gate 2026-10-08T07:49:23+08:00 =====
python  : Python 3.10.12
torch   : 2.11.0 6.3.26113
dist    : 2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7 | 3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1
triton  : 3.5.0

----- 1) version gate (distribution metadata = exact pinned versions) -----
torch dist : 2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7
triton dist: 3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1
TORCH_PIN_OK  True
TRITON_PIN_OK True
torch.version.__hcu_version__ = 2.11.0+das.opt1.dtk2604
rc=0

----- 2) import gate, torch imported first -----
/mnt/game/port/gfx906-vllm-211/vllm/__init__.py:7: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from .version import __version__, __version_tuple__  # isort:skip
IMPORT_OK /mnt/game/port/gfx906-vllm-211/vllm/_C.abi3.so
rc=0

----- 2b) literal design-doc wording (no torch preload) -----
terminate called after throwing an instance of 'std::bad_alloc'
  what():  std::bad_alloc
rc=134

----- 3) gfx targets -----
_C.abi3.so                             archs: gfx906
_rocm_C.abi3.so                        archs: gfx906
vendor _C (reference)                  archs: gfx906 gfx926 gfx928 gfx936

----- 4) ROCM_SKINNY_GEMM_ENABLED -----
ROCM_SKINNY_GEMM_ENABLED = False
hasattr LLMM1 = False
rc=0

----- 5) _rocm_C registered ops -----
['_rocm_C::paged_attention']
rc=0
===== gate done 2026-10-08T07:50:16+08:00 =====
```

`2b)` 与 S4b §3.5/§3.7.6 **完全一样**（`std::bad_alloc` / rc=134），是"不先 `import torch` 就 `import vllm._C`"的
**已知、非回归**现象：todolist 硬约束 8 要求先 `import torch`，`gate-211.sh:2b` 只是照抄设计文档字面写法做对照。

### 3.6 四扩展 import + 四个 `.so` 的架构检查（本步主要门禁）

```console
$ source /mnt/game/port/setenv-211.sh >/dev/null 2>&1; cd "$REPO_DIR"
$ python -c "import torch, vllm, vllm._C, vllm._rocm_C, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch; print('IMPORT_OK_ALL4')"
IMPORT_OK_ALL4
rc=0

$ for so in "$REPO_DIR"/vllm/_C.abi3.so "$REPO_DIR"/vllm/_rocm_C.abi3.so \
            "$REPO_DIR"/vllm/_C_stable_libtorch.abi3.so "$REPO_DIR"/vllm/_moe_C_stable_libtorch.abi3.so; do
    printf "%-34s gfx: " "$(basename $so)"
    strings -a $so | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sed "s/.*--//" | sort -u | tr "\n" " "
    printf "| count_distinct="; strings -a $so | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sed "s/.*--//" | sort -u | wc -l
  done
_C.abi3.so                         gfx: gfx906 | count_distinct=1
_rocm_C.abi3.so                    gfx: gfx906 | count_distinct=1
_C_stable_libtorch.abi3.so         gfx: gfx906 | count_distinct=1
_moe_C_stable_libtorch.abi3.so     gfx: gfx906 | count_distinct=1
```

### 3.7 回归（S4b §3.6 的两个 gfx906 内核套件，在最终产物上复跑）

```console
$ for f in tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py tests/kernels/mamba/test_fused_sigmoid_gating.py; do
    python -c "import torch, pytest, sys; sys.exit(pytest.main([\"-q\",\"--no-header\",\"--noconftest\",\"-p\",\"no:cacheprovider\",\"$f\"]))"
  done
########## tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py
13 passed, 3 warnings in 27.43s
##### rc=0
########## tests/kernels/mamba/test_fused_sigmoid_gating.py
5 passed, 1 warning in 16.05s
##### rc=0
```

### 3.8 产物 md5：三个与 S4b 完全一致，`_C_stable_libtorch` 的差异已彻底归因

```console
$ md5sum /mnt/game/port/gfx906-vllm-211/vllm/*.so
6fba05966e153b3d04771d6960c782bb  .../vllm/_C.abi3.so                       # 与 S4b 记录一致
5d1e50e1a77de74f2d308519f8120745  .../vllm/_rocm_C.abi3.so                  # 与 S4b 记录一致
cc2aca69ca9a9afcb67b7428ede181b1  .../vllm/_C_stable_libtorch.abi3.so        # S4b 记录为 773f6882303556a29027f438bbadbe85
996b282c4c6a3d3b46b5e8ba4922af39  .../vllm/_moe_C_stable_libtorch.abi3.so   # 与 S4b 记录一致
```

`_C`/`_rocm_C` 的 md5 与 S4b **逐字节相同**这一点本身是有意义的证据：虽然本次 `hipify` 把
`type_convert_hip.cuh`、`custom_all_gather_reduce_scatter_hip.cuh` 里的 `defined(D) && (D >= 12000)` 改回了
`defined(DTK_VERSION) && (DTK_VERSION >= 12000)`，但这些 token 都落在必然短路的守卫上，行号不变 ⇒ 目标文件与
安装产物逐字节不变，**验证了 S4b §3.7.3"`_C`/`_rocm_C` 语义等价"的论断**。

**重要口径**：安装产物与构建树里的原始产物**不是一份字节**（`cmake --install` 会在原地改 RPATH，体积不变、内容变），
所以四个"安装后 md5"都不等于"原始产物 md5"：

```console
# 构建树里的原始产物（裸全量 ninja 之后直接从 build-211/temp/ 拷贝留存）
048736b122bbf5e4932fe6c4c89266bd  _C.abi3.so                       78102728 B
019b327b01cfa76e389e427f1849433a  _rocm_C.abi3.so                  35156368 B
7068229e9c37ca3d8dd2976b8bc9fb4c  _C_stable_libtorch.abi3.so       40688656 B
af01ed9bb3fc5ac913bff845f502e27a  _moe_C_stable_libtorch.abi3.so   29748928 B
# 对照上一段的四个"安装后" md5：四个全不相同（体积相同）。

`_C_stable_libtorch` 的差异做了受控 A/B（结论：完全良性，不是回归）：

```console
# A) 只把本步新增的 7 行注释删掉（守卫保留，即 S4b 构建副本里的形状），其余一切不变，重编：
$ MAX_JOBS=4 LOG=/mnt/game/port/s4c-ab-nocomment.log bash /mnt/game/port/build-211.sh _C_stable_libtorch
[29/29] Linking HIP shared module _C_stable_libtorch.abi3.so
----- ninja rc=0 -----
-rwxr-xr-x 1 root root 40688664 .../build-211/temp/_C_stable_libtorch.abi3.so
#                                            ^^^^^^^^ 与 S4b §6.1 记录的 40,688,664 B 完全一致
# B) 两次独立构建（裸全量 ninja 的原始产物 vs 之后单目标重编的原始产物）逐字节相同：
$ cmp -l .../_C_stable_libtorch.commented.so .../build-211/temp/_C_stable_libtorch.abi3.so | wc -l
0
# 上句左边那个副本是在 A) 之前从构建树里拷出来的"裸全量 ninja 原始产物"。⇒ 本环境下 HIP 编译/链接可复现。
```

⇒ **8 字节差异来自注释把后续源码行号整体下移 7 行**（RelWithDebInfo 的 DWARF 行表），与代码语义无关；
S4b 记的 `40,688,664 B` 是**构建树里的原始产物**大小，而这个原始大小恰好被上面 A) 复现。

第二处 md5 差异来自 **`cmake --install` 会就地改 RPATH**（这也是 S4b 记的 md5 取自安装后文件的原因）：

```console
$ RAW=.../build-211/temp/_C_stable_libtorch.abi3.so      # 原始产物
$ INST=.../gfx906-vllm-211/vllm/_C_stable_libtorch.abi3.so   # 安装产物
RAW  7068229e9c37ca3d8dd2976b8bc9fb4c  40688656 B
INST cc2aca69ca9a9afcb67b7428ede181b1  40688656 B
$ cmp -l $RAW $INST | wc -l
258                                    # 体积不变，258 字节不同
$ readelf -d $RAW  | grep RUNPATH
 0x000000000000001d (RUNPATH)  Library runpath: [/lib/intel64:/lib/intel64_win:/lib/win-x64:...:/opt/dtk/hip/lib:]
$ readelf -d $INST | grep RUNPATH
（空：安装时 "Set non-toolchain portion of runtime path ... to """
$ bash /mnt/game/port/install-211.sh _C_stable_libtorch && md5sum $INST
cc2aca69ca9a9afcb67b7428ede181b1       # 同一原始产物重复安装 ⇒ 稳定重现
```

> 因此后续步骤**不要**把"安装后 `.so` 的 md5"当作构建身份指纹：安装会重写 RPATH。
> 要判断构建是否可复现，请对**构建树里的原始产物**做 `cmp`/md5。

### 3.9 资源、内存与"未破坏性"自检

```console
$ docker exec z100-build-211 bash -lc 'cat /sys/fs/cgroup/memory.events; cat /sys/fs/cgroup/memory.current'
low 0
high 0
max 0            <- 从未触碰 --memory=8g 上限
oom 0
oom_kill 0
memory.current(构建结束后) = 3349323776   # 3.12 GiB
# 该 cgroup 没有 memory.peak，构建期间未做采样，所以这里只有"构建结束后"的瞬时值；
# 宿主侧观测：构建前 free 的 available = 12 GB、构建后 = 20 GB（未出现整体内存压力）。
```

未破坏性自检（只读）：

```console
# 1) 2.10 基线副本 /mnt/game/port/gfx906-vllm：mtime 仍是 S5 的 01:57/01:58，md5 未变
-rw-r--r-- 1 root root 78034896 2026-10-08_01:57:28 .../gfx906-vllm/vllm/_C.abi3.so
-rw-r--r-- 1 root root 35130920 2026-10-08_01:58:43 .../gfx906-vllm/vllm/_rocm_C.abi3.so
4a721d65e712e15d7018e04ecaa99f44  .../gfx906-vllm/vllm/_C.abi3.so
3616084a0c2cfd52ca667af6ea91620c  .../gfx906-vllm/vllm/_rocm_C.abi3.so
# 2) z100-build 的构建树：.so mtime 仍是 01:57/01:58；容器里 torch 仍是 2.10.0+das.opt1.dtk2604.20260325.g6b060a
-rwxr-xr-x 1 root root 78034896 2026-10-08_01:57:28 /mnt/game/port/build-gfx906/temp/_C.abi3.so
-rwxr-xr-x 1 root root 35130920 2026-10-08_01:58:43 /mnt/game/port/build-gfx906/temp/_rocm_C.abi3.so
# 3) 容器列表
z100-triton      Up 29 minutes       <- S6 路线（不是本步创建，本步未触碰）
z100-build-211   Up About an hour    <- 本步使用的构建容器
z100-build       Up 8 hours          <- 2.10 基线，未触碰
qwen38-vllm      Exited (137) 20 minutes ago
```

> **观察（非本步所为，供 Lead 知悉）**：生产容器 `qwen38-vllm` 在 2026-10-07T23:37:51Z 收到
> `container kill signal=15`、23:38:51Z 收到 `signal=9`（即一次显式的 `docker stop`，`docker events` 原文），
> 23:38:58Z `die`。`docker inspect` 显示 `OOMKilled=false`、`Memory=0`（无容器内存上限），且我的构建容器
> `memory.events` 的 `oom/oom_kill` 全 0、宿主 `dmesg` 未见 oom-kill 记录 —— 该停止**不是** OOM，
> 也不是 S4c 发起的（本步对 `qwen38-vllm` 只做过 `docker ps`/`docker inspect`/`docker events` 只读查询）。

### 3.10 第二次裸 `ninja`（可复现性）与一个**预存在**的构建特性：`hipify_all` 每次都会重跑

§3.8 的 A/B 实验（以及两次**未 source DTK 环境**就调用 `ninja` 的误操作：报
`ImportError: librocm_smi64.so.2`，即 S4b §3.2 记录的"不 source 环境不能 import torch"，
它们只删掉了 `CMakeFiles/hipify_all` 这个 stamp、没有改任何源码或 `.hip`/`.o`）
把构建树留在了"部分目标陈旧"的状态（`ninja -n` 显示 61 步待做），为交付一个内容自洽的构建树，
本步又跑了**第二次裸 `ninja`**：

```console
$ MAX_JOBS=4 LOG=/mnt/game/port/s4c-full-ninja-2.log bash /mnt/game/port/build-211.sh
===== S4b build 2026-10-08T08:02:23+08:00 targets:  MAX_JOBS=4 =====
[1/61] Running hipify on all extension source files.
...
[29/60] Linking HIP shared module _C.abi3.so
[53/60] Linking HIP shared module _C_stable_libtorch.abi3.so
[58/60] Linking HIP shared module _rocm_C.abi3.so
[60/60] Linking HIP shared module _moe_C_stable_libtorch.abi3.so
----- ninja rc=0 2026-10-08T08:18:48+08:00 -----
# wall_s=986（16 min 26 s）；error: 0、FAILED: 0
```

**两次独立的全量构建，四个扩展的原始产物逐字节相同**（`cmp` 全 0）——这是本档对"构建可复现"的正面证据：

```console
$ for s in _C _rocm_C _C_stable_libtorch _moe_C_stable_libtorch; do
    cmp -s raw-before/$s.abi3.so build-211/temp/$s.abi3.so && echo "$s IDENTICAL"; done
_C                             IDENTICAL   048736b122bbf5e4932fe6c4c89266bd
_rocm_C                        IDENTICAL   019b327b01cfa76e389e427f1849433a
_C_stable_libtorch             IDENTICAL   7068229e9c37ca3d8dd2976b8bc9fb4c
_moe_C_stable_libtorch         IDENTICAL   af01ed9bb3fc5ac913bff845f502e27a
# 安装产物 md5 亦不变：6fba0596 / 5d1e50e1 / cc2aca69 / 996b282c；IMPORT_OK_ALL4 复测通过。
```

**预存在特性（不是本步引入，S7/S9 排期要注意）**：`hipify_all` 的 `COMMAND` 里**没有** `cmake -E touch`
去创建它声明的 stamp 输出，因此该 stamp 永远不存在 ⇒ **每次 `ninja` 都会重跑 hipify 并连带重编全部扩展**，
即这两个构建树实际上是"每次都全量重编"，而不是增量：

```console
# 第二轮构建完成之后的构建树（z100-build-211）
$ ls -l /mnt/game/port/build-211/temp/CMakeFiles/hipify_all
ls: cannot access '.../CMakeFiles/hipify_all': No such file or directory
$ cd /mnt/game/port/build-211/temp && ninja -d explain -n 2>&1 | head -3
ninja explain: output CMakeFiles/hipify_all doesn't exist
ninja explain: /mnt/game/port/build-211/temp/csrc/mamba/mamba_ssm/selective_scan_fwd.hip is dirty
ninja explain: CMakeFiles/hipify_all is dirty
$ ninja -n | grep -cE "^\["
61                                  # 刚跑完全量构建，仍然 61 步

# 同一个现象在 2.10 基线构建树里**完全一样**（只读核对）：
$ cd /mnt/game/port/build-gfx906/temp && ls CMakeFiles/hipify_all
ls: cannot access 'CMakeFiles/hipify_all': No such file or directory
$ ninja -n | grep -cE "^\["
64
$ grep -A1 "^build CMakeFiles/hipify_all" build.ninja | grep -c "cmake -E touch"
0
```

> 结论：`z100-build-211` / `z100-build` 的构建树都"永远会全量重编"（约 16–18 min / 次，`-j4`）。
> 本步**未修改** `CMakeLists.txt`（不在写作用域内）；建议 S9 在固化阶段补一个 stamp touch 或 `restat`，
> 或至少在排期上按"每次构建都是全量"来估算。

## 4. 门禁判定

| # | 门禁 | 结果 | 证据 |
|---|---|---|---|
| 1 | `cmake/hipify.py` 元组化修复落仓库 | ✅ | §2 的 diff；md5 `00245a1752…`；`cmake/hipify.py:17-28` |
| 2 | 该修复对 2.11 裸字符串与 2.10 元组都正确 | ✅ | §3.2 端到端探针（`D >= 11000` → `DTK_VERSION >= 11000`）+ 元组形状实测未被二次包装 |
| 3 | stable 孪生文件守卫落仓库、与孪生逐字一致 | ✅ | §2/§3.3（`IDENTICAL`）；`csrc/libtorch_stable/fused_qknorm_rope_kernel.cu:49-50` |
| 4 | 两项同步进 `z100-build-211` 构建副本 | ✅ | §3.3 md5 两两相同 |
| 5 | **全量 `ninja`（默认 `all`）走通：四个扩展全部编成** | ✅ | §3.4：`[62/62]`、`rc=0`、`error: 0`、`FAILED: 0`；四个 `Linking HIP shared module` 齐备 |
| 6 | 全量 ninja 的 rc 与耗时已记录 | ✅ | §3.4：`rc=0`、`wall_s=1052`（07:31:37→07:49:09，17 min 32 s） |
| 7 | 四扩展 import 全通过（`IMPORT_OK_ALL4`） | ✅ | §3.6；`gate-211.sh` 第 2 项 `IMPORT_OK` rc=0 同向 |
| 8 | 四个 `.so` 只含 `gfx906` | ✅ | §3.6，四个文件 `count_distinct=1` 且均为 `gfx906` |
| 9 | 版本判据（`importlib.metadata`）通过 | ✅ | §3.5 第 1 项 `TORCH_PIN_OK/TRITON_PIN_OK True` |
| 10 | `ROCM_SKINNY_GEMM_ENABLED == False`、`_rocm_C::paged_attention` 注册面 | ✅ | §3.5 第 4/5 项 |
| 11 | 回归 gguf 13/13、fused sigmoid gating 5/5 | ✅ | §3.7（13 passed / 5 passed，rc=0） |
| 12 | 硬约束：无自编译 HIP 可执行程序、不碰 `qwen38-vllm`、不动 `z100-build`、不重建 triton、产物只落 `/mnt/game`、`MAX_JOBS≤4`、`--memory=8g` 未触限 | ✅ | §3.9；本步所有 `docker exec` 只针对 `z100-build-211`（只读查询除外） |
| 13 | 构建可复现（附加值）：第二次裸全量 ninja 的四个原始产物与第一次逐字节相同 | ✅ | §3.10：四次 `cmp` 全 0；第二次 `rc=0`、`wall_s=986` |
| — | `gate-211.sh` 第 2b 项（不先 `import torch`） | ⚠️ 已知非回归 | §3.5：`std::bad_alloc` rc=134，与 S4b §3.5/§3.7.6 完全一致 |

**总判定：✅ 通过。** S4b §5.1 的两项修复已落地仓库并在新栈用一次裸 `ninja` 复验；
S5 §6.6 的"不要裸跑全量 ninja"限制在 torch 2.11 新栈上**解除**。

## 5. 未完成项 / 阻塞与原因

- **无阻塞项**。以下为明确"未执行"或留档说明：
  1. **未在 2.10 容器（`z100-build`）重跑构建**：硬约束要求不删/不升级 `z100-build`。改动 A 对 2.10 的安全性
     由"2.10 的 `CUDA_MACROS_MAP` 实测为元组 + `isinstance` 守卫"论证（§3.1/§3.2），未做 2.10 整轮重建。
  2. **未升级/重建 triton**：S6 路线；期间观察到 `z100-triton` 容器在运行（本步未触碰）。
  3. **`triton_kernels` CMake 目标未构建**：裸 ninja 的 `all` 里该目标在本构建树中是空 phony（`ninja -n | grep -i triton` = 0 步），
     与 S4b §5.3 一致，不在本步范围。
  4. **未安装 `cumem_allocator.abi3.so`**：裸 ninja 的 `all` 顺带把它编出来了（`build-211/temp/cumem_allocator.abi3.so`，
     117,816 B），但 S4b 的安装面只有 4 个扩展，本步保持一致、未安装。若 S7+ 需要，`install-211.sh cumem_allocator` 即可。
  5. **`gate-211.sh:2b` 仍是 rc=134**：照抄设计文档字面写法（不先 `import torch`）的已知现象，非本步引入，未改脚本
     （脚本不在写作用域内）。
  6. **未运行任何自编译 HIP 独立可执行程序**（硬约束，历史事故），也未跑真机 e2e 服务冒烟（S9 范围）。
  7. 生产容器 `qwen38-vllm` 现处于 `Exited (137)`，系外部显式 `docker stop`（见 §3.9 观察）；本步未对其做任何写操作。
  8. **未修"每次 ninja 都全量重编"**：`hipify_all` 的 stamp 从不创建（2.10/2.11 两个构建树都有，见 §3.10），
     修它需要改 `CMakeLists.txt`（不在本步写作用域），留给 S9 决定。
- 本步遗留的**非仓库**文件（新增，均在 `/mnt/game/port` 下，不影响 `z100-build`/`gfx906-vllm`）：
  `s4c-full-ninja.log`（第一轮全量）、`s4c-full-ninja-2.log`（第二轮全量）、`gate-211-s4c.log`、
  `s4c-rebuild-final.log`、`s4c-ab-nocomment.log`、`build-211/temp/build.ninja.s4c-wrapper-patched`、
  `s4c-probe/`（hipify 端到端探针、A/B 证据副本、`raw-before/` 第一轮四个原始产物）。

## 6. 对后续步骤的影响

- **S6（triton gfx906）**：入口判断不变（S4b §3.8/§6.2）。以下与 S6 相关的边界情况无变化：
  triton 仍未被本步触碰；`AMDGCN_USE_BUFFER_OPS=0` 与 hcu 后端的注意事项照旧。
- **S7/S8/S9**：可以完全在 `z100-build-211` + `/mnt/game/port/gfx906-vllm-211` 上做，且**现在可以直接用裸 `ninja`**
  （不必再逐个列举 target）；四个扩展齐备且只含 gfx906。**注意**：`_C_stable_libtorch` 的安装产物 md5 已从 S4b 记的
  `773f6882…` 变为 `cc2aca69…`（原因见 §3.8，良性）；判定构建身份请用构建树内原始产物，不要用安装后 md5。
- **构建耗时预算（新信息）**：两个构建树的 `hipify_all` stamp 都不会被创建 ⇒ **任何一次 `ninja` 都是全量重编**
  （§3.10；2.10 与 2.11 树均如此），`-j4` 约 16–18 min。别把 `build-211/temp` 当"增量"目录来安排迭代；
  小改动想快速验证，可只对单个 target 跑 `build-211.sh <target>`（仍会重跑 hipify，但只重编该 target 的 TU）。
  当前树已处于"四个扩展全部由最终源码构建完成"的自洽状态（原始产物 md5 见 §3.10）。
- **"DTK 工具链缺陷清单"**（S4b §6.3 建议在 S9 统一成文）：本档把第 5 条（hipify `CUDA_MACROS_MAP`）
  从"候选修复点"升级为"仓库已修"，并补齐了它的精确触发条件（`hipify_python` 导入期快照 + `CUDA_VERSION`
  同时在 trie 里），第 5 条现在可以写明"修复位置 `cmake/hipify.py:17-28`"。
  `__syncwarp` 孪生漏修项也已在仓库侧闭合（`csrc/libtorch_stable/fused_qknorm_rope_kernel.cu:49-50`）。
- **回滚语义**：若因故回退到 2.10 基线，改动 A 是 no-op（元组守卫），改动 B 仍然必要（与 torch 版本无关）——
  与 S4b §6.3 的判断一致。

## 7. 复现方式

```bash
# 0) 前置：容器 z100-build-211 在跑（S4b §7 的创建命令），/mnt/game/port/gfx906-vllm-211 与 build-211 可用。
#    仓库侧先应用本档 §2 的两处改动（cmake/hipify.py:17-28、csrc/libtorch_stable/fused_qknorm_rope_kernel.cu:42-50）。

# 1) 同步进构建副本（/src 是工作区的只读挂载）
docker exec z100-build-211 bash -lc '
  cp -p /src/cmake/hipify.py /mnt/game/port/gfx906-vllm-211/cmake/hipify.py
  cp -p /src/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu \
        /mnt/game/port/gfx906-vllm-211/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu
  md5sum /src/cmake/hipify.py /mnt/game/port/gfx906-vllm-211/cmake/hipify.py'

# 2) 撤掉 S4b 的 hipify wrapper（修复已进 cmake/hipify.py，不再需要）
docker exec z100-build-211 bash -lc '
  cd /mnt/game/port/build-211/temp
  cp -p build.ninja build.ninja.s4c-wrapper-patched
  cp -p build.ninja.s4b-orig build.ninja
  grep -c hipify-fix-211 build.ninja'          # 期望：0

# 3) 看全量计划（可选）：约 60 步、无 triton
#    注意：该树总是把 hipify_all 与全部 .hip 判为脏（§3.10），所以"刚编完"也照样是这个步数。
docker exec z100-build-211 bash -lc 'cd /mnt/game/port/build-211/temp && ninja -n | grep -cE "^\["'

# 4) 全量 ninja（不传 target = 默认 all）
docker exec z100-build-211 bash -lc 'MAX_JOBS=4 LOG=/mnt/game/port/s4c-full-ninja.log bash /mnt/game/port/build-211.sh'
#    期望：... [62/62] Linking HIP shared module _moe_C_stable_libtorch.abi3.so
#          ----- ninja rc=0 ... -----        （本次两轮耗时 1052 s / 986 s，约 16–18 分钟）
#    想验证可复现：再跑一次（LOG 换 s4c-full-ninja-2.log），然后对 build-211/temp/*.so 做 cmp ⇒ 全 0（§3.10）

# 5) 安装四个扩展
docker exec z100-build-211 bash /mnt/game/port/install-211.sh _C _rocm_C _C_stable_libtorch _moe_C_stable_libtorch

# 6) 门禁
docker exec z100-build-211 bash /mnt/game/port/gate-211.sh
docker exec z100-build-211 bash -lc 'source /mnt/game/port/setenv-211.sh >/dev/null 2>&1; cd "$REPO_DIR"; \
  python -c "import torch, vllm, vllm._C, vllm._rocm_C, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch; print(\"IMPORT_OK_ALL4\")"'
docker exec z100-build-211 bash -lc 'for so in /mnt/game/port/gfx906-vllm-211/vllm/*.so; do \
  printf "%-34s " "$(basename $so)"; \
  strings -a $so | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sed "s/.*--//" | sort -u | tr "\n" " "; echo; done'
#    期望：IMPORT_OK_ALL4；四个文件各只列 gfx906

# 7) 回归（必须先 import torch，且 --noconftest）
docker exec z100-build-211 bash -lc 'source /mnt/game/port/setenv-211.sh >/dev/null 2>&1; cd "$REPO_DIR"
  for f in tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py tests/kernels/mamba/test_fused_sigmoid_gating.py; do
    echo "########## $f"
    python -c "import torch, pytest, sys; sys.exit(pytest.main([\"-q\",\"--no-header\",\"--noconftest\",\"-p\",\"no:cacheprovider\",\"$f\"]))"
  done'
#    期望：13 passed / 5 passed

# 8) hipify 修复的单点复现（可选）：控制组用改动前的脚本（本档 §3.2 的 /mnt/game/port/s4c-probe/ 布局）
docker exec z100-build-211 bash -lc '
  P=/mnt/game/port/s4c-probe; mkdir -p $P/proj $P/out_fixed
  printf "#include <cuda_runtime.h>\n\n#if CUDA_VERSION >= 11000\nint probe_cuda_version_ge_11000;\n#endif\n" > $P/proj/probe_macro.cu
  cd $P/out_fixed && python /src/cmake/hipify.py -p $P/proj -o $P/out_fixed probe_macro.cu >/dev/null
  grep -n "DTK_VERSION >= 11000" $P/out_fixed/probe_macro.hip'   # 期望命中；改动前会是 "#if D >= 11000"
```

## 8. 参考

- 本步设计文档：[docs/design/z100-04b-stack-upgrade.md](../design/z100-04b-stack-upgrade.md)（§验收 / §失败处置）
- 前序归档：
  - **第一档（本档直接承接）**：[z100-04b-stack-upgrade-result.md](z100-04b-stack-upgrade-result.md)
    §3.7.2（根因）、§3.7.3（wrapper 验证）、§3.7.4/§3.7.5（两项修复与失败序列）、
    **§5.1（两项修复的确切内容）**、§6（容器/脚本/资源表）、§7（复现配方）、附（补丁原文）
  - [z100-05-csrc-kernels-result.md](z100-05-csrc-kernels-result.md)：`__syncwarp` 缺陷 #2 与非 stable 孪生文件的原始修复
  - [z100-04-build-system-dtk-result.md](z100-04-build-system-dtk-result.md)：构建环境与 `_DTK_PREFIX`/`PYTORCH_ROCM_ARCH`
- 归档规则与模板：[docs/result/README.md](README.md)
- Todolist 与硬约束：[docs/z100-todolist.md](../z100-todolist.md)（S4c 行、硬约束 8/9/10）
- 关键源码位置（本步改动）：
  - `cmake/hipify.py:17-28`（`CUDA_MACROS_MAP` 元组化；下移的 `hipify_python` 导入在 30 行）
  - `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu:42-50`（守卫 + 注释）
  - 对照：`csrc/fused_qknorm_rope_kernel.cu:41-49`
- torch 侧真相源（z100-build-211 容器内）：
  - `/usr/local/lib/python3.10/dist-packages/torch/utils/hipify/cuda_to_hip_mappings.py:3496-3501`（2.11 裸字符串）
  - `/usr/local/lib/python3.10/dist-packages/torch/utils/hipify/hipify_python.py:803-819`（导入期快照）、`:900-901`（`cumacro_repl` 覆盖）
  - 2.10 对照（z100-build 容器内）：`cuda_to_hip_mappings.py` 的 `CUDA_MACROS_MAP` 值为 `('DTK_VERSION', 1337)`
- 本步产物日志：`/mnt/game/port/s4c-full-ninja.log`（第一轮全量 ninja，rc=0 / 1052 s）、
  `/mnt/game/port/s4c-full-ninja-2.log`（第二轮，rc=0 / 986 s）、`/mnt/game/port/gate-211-s4c.log`、
  `/mnt/game/port/s4c-ab-nocomment.log`、`/mnt/game/port/s4c-rebuild-final.log`、
  `/mnt/game/port/s4c-probe/`（探针 + A/B 证据 + `raw-before/` 第一轮四个原始产物与其 md5.txt）
