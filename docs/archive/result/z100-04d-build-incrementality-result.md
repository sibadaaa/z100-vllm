# S4d 修 `hipify_all` stamp 缺失导致的全量重编（构建可增量性） — 执行结果

> 归档时间：2026-10-08
> 执行者：子 agent（S4d）
> 对应设计文档：docs/design/z100-04-build-system-dtk.md（CMake 部分）
> 前一档：[docs/result/z100-04b-stack-upgrade-result-2.md](z100-04b-stack-upgrade-result-2.md)（S4c，§3.10 提出本项遗留）
> 状态：✅ 完成（含 1 处**写作用域偏差**与 1 项**残余 ninja 语义**，均在 §2.4 / §5 明确说明）

> **一句话结论**：`hipify_all` 现在真正被 stamp 化（`cmake -E touch` 一个 OUTPUT stamp），
> 并且 hipify 改为"写 staging 树 + 按内容同步进构建树"，因此
> **(a)** 改后第一次裸 `ninja`（全量）rc=0 / 963 s；**(b)** 紧接着第二次裸 `ninja` = `ninja: no work to do.` / 0.096 s；
> **(c)** 构建树四个原始 `.so` 与 S4c 基线 **逐字节相同**（`cmp` 全 0）；
> **(d)** 改一个源文件只重编 **1 个 TU + 1 次链接**（40 s，全量是 963 s）、改 `hipify.py` 只重跑 hipify（6 s）。
> 门禁 `IMPORT_OK_ALL4` + 四个 `.so` 只含 `gfx906` 复测通过。

## 1. 目标与验收标准

### 1.1 本步任务书（S4d）要求

1. 修 `CMakeLists.txt` 里 hipify 相关规则，让 hipify 步骤正确 stamp 化（最小改动：在该 COMMAND 末尾
   `cmake -E touch` 对应 stamp，或改成规范的 `OUTPUT`/`DEPENDS` 形式），要求：
   - 语义不变（hipified 源与产物内容一致）；
   - 源文件或 hipify 脚本变更时**仍会**重跑 hipify 并**只重编受影响的 TU**（不能变成"陈旧产物不重建"的陷阱）；
   - 非 HIP/CUDA 构建路径行为不变。
2. 在 `z100-build-211` 的构建树 `/mnt/game/port/build-211` 验证：
   a. 第一次 `ninja`（全量）rc=0；
   b. 紧接着第二次 `ninja` **no-op**（`ninja: no work to do.`）、耗时秒级；
   c. **身份检查**：构建树四个**原始**（未 install 的）`.so` md5 与 S4c 基线完全一致
      （`_C 048736b1…`、`_rocm_C 019b327b…`、`_C_stable_libtorch 7068229e…`、`_moe_C_stable_libtorch af01ed9b…`）；
   d. **增量正确性**：改一个真实源文件后 `ninja`，只重编受影响 TU 并重新链接，耗时远小于全量；
      贴出 ninja 的输出行数与耗时。
3. 门禁仍过：`IMPORT_OK_ALL4` + 四个 `.so` 的 `strings | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sort -u` 只有 `gfx906`。

### 1.2 硬约束（本步遵守情况）

禁止自编译 HIP 独立可执行程序（本步只跑 `ninja`/`hipify.py`/`strings`/`python import`）；不碰
`qwen38-vllm`（当前 `Exited (137)`，外部停的，本步只读查询）；不动 `z100-build`（2.10 基线）与其构建树；
产物只落 `/mnt/game`；`MAX_JOBS=4`、容器 `--memory=8g`。

## 2. 实际改动

### 2.1 改动清单

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `cmake/utils.cmake` | **74-90** | `hipify_sources_target`：原来在 `foreach` 里把循环变量 `SRC` 就地改名成 `.hip`；现在保留原名，另收集一组 `"<build>/x.hip\|<repo>/x.cu"` 配对（输出路径计算方式**逐字未变**） |
| `cmake/utils.cmake` | **124-152** | `vllm_finalize_hipify_target`：`add_custom_target(hipify_all COMMAND hipify.py …)` → `add_custom_command(OUTPUT CMakeFiles/hipify_all.stamp …)`：hipify 写入 **staging 树** → `cmake -E copy_directory_if_different` **按内容同步**进 `${CMAKE_CURRENT_BINARY_DIR}/csrc` → `cmake -E touch` stamp；再 `add_custom_target(hipify_all DEPENDS ${HIPIFY_STAMP})` |
| `cmake/utils.cmake` | **154-169** | 给每个生成出来的 `.hip` 加 `OBJECT_DEPENDS <对应源 .cu>`（已被 Ninja 生成器实现为 `.o` 边上的隐式依赖 `\| …/x.cu`） |
| `CMakeLists.txt` | **无改动** | 见 §2.4 偏差说明 |

确切改动（`git diff -- cmake/utils.cmake`，`md5=5f97a913c49f1cf37317d8a8e78156a2`）：

```diff
@@ -72,14 +72,22 @@ function (hipify_sources_target OUT_SRCS NAME ORIG_SRCS)
   set(HIP_SRCS)
+  # `HIP_SRC|<original source>` pairs. The finalizer turns them into
+  # OBJECT_DEPENDS entries so that a CUDA source whose content is unchanged but
+  # whose mtime moved still recompiles its own translation unit.
+  set(HIP_TO_ORIG)
   foreach (SRC ${SRCS})
-    string(REGEX REPLACE "\.cu$" "\.hip" SRC ${SRC})
-    string(REGEX REPLACE "cuda" "hip" SRC ${SRC})
-    list(APPEND HIP_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${SRC}")
+    set(HIP_SRC ${SRC})
+    string(REGEX REPLACE "\.cu$" "\.hip" HIP_SRC ${HIP_SRC})
+    string(REGEX REPLACE "cuda" "hip" HIP_SRC ${HIP_SRC})
+    list(APPEND HIP_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${HIP_SRC}")
+    list(APPEND HIP_TO_ORIG
+      "${CMAKE_CURRENT_BINARY_DIR}/${HIP_SRC}|${CMAKE_CURRENT_SOURCE_DIR}/${SRC}")
   endforeach()
 
   set_property(GLOBAL APPEND PROPERTY VLLM_HIPIFY_ALL_SRCS ${SRCS})
   set_property(GLOBAL APPEND PROPERTY VLLM_HIPIFY_ALL_BYPRODUCTS ${HIP_SRCS})
+  set_property(GLOBAL APPEND PROPERTY VLLM_HIPIFY_ALL_HIP_TO_ORIG ${HIP_TO_ORIG})
@@ -104,6 +112,7 @@ function (vllm_finalize_hipify_target)
   get_property(ALL_SRCS GLOBAL PROPERTY VLLM_HIPIFY_ALL_SRCS)
   get_property(ALL_BYPRODUCTS GLOBAL PROPERTY VLLM_HIPIFY_ALL_BYPRODUCTS)
+  get_property(ALL_HIP_TO_ORIG GLOBAL PROPERTY VLLM_HIPIFY_ALL_HIP_TO_ORIG)
@@ -113,12 +122,51 @@ function (vllm_finalize_hipify_target)
+  set(HIPIFY_STAMP ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/hipify_all.stamp)
+  set(HIPIFY_STAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/hipify_stage)
+  file(MAKE_DIRECTORY ${HIPIFY_STAGE_DIR})
+
+  add_custom_command(
+    OUTPUT ${HIPIFY_STAMP}
+    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/hipify.py -p ${CMAKE_SOURCE_DIR}/csrc -o ${HIPIFY_STAGE_DIR}/csrc ${ALL_SRCS}
+    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${HIPIFY_STAGE_DIR}/csrc ${CSRC_BUILD_DIR}
+    COMMAND ${CMAKE_COMMAND} -E touch ${HIPIFY_STAMP}
     DEPENDS ${CMAKE_SOURCE_DIR}/cmake/hipify.py ${ALL_SRCS}
     BYPRODUCTS ${ALL_BYPRODUCTS}
-    COMMENT "Running hipify on all extension source files.")
+    WORKING_DIRECTORY ${HIPIFY_STAGE_DIR}
+    COMMENT "Running hipify on all extension source files."
+    VERBATIM)
+
+  set(_HIPIFY_DEPS_SET)
+  foreach (_PAIR ${ALL_HIP_TO_ORIG})
+    string(REPLACE "|" ";" _PAIR_PARTS "${_PAIR}")
+    list(GET _PAIR_PARTS 0 _HIP_SRC)
+    list(GET _PAIR_PARTS 1 _ORIG_SRC)
+    if (NOT _HIP_SRC IN_LIST _HIPIFY_DEPS_SET)
+      list(APPEND _HIPIFY_DEPS_SET ${_HIP_SRC})
+      set_source_files_properties(${_HIP_SRC} PROPERTIES OBJECT_DEPENDS ${_ORIG_SRC})
+    endif()
+  endforeach()
+  unset(_HIPIFY_DEPS_SET)
+
+  add_custom_target(hipify_all DEPENDS ${HIPIFY_STAMP})
 endfunction()
```

### 2.2 为什么只加 stamp 还不够（本步实测出的第二层问题）

`hipify_all` 的 COMMAND 里没有 `cmake -E touch`，声明的 stamp 输出 `CMakeFiles/hipify_all` 永远不存在 ⇒
custom command 边每次都脏（S4c §3.10）。**但**只补一个 touch 之后，"改 1 个源文件"仍会全量重编，原因是：

- DTK 的 `torch.utils.hipify` 给每个生成文件**注入** `#include <ATen/dtk_macros.h>`；
- `cmake/hipify.py` 的 `_neutralize_dtk_macros_include()` 在 hipify 写盘**之后**把这个 include 替换成一段
  等价宏定义（为了 stable ABI 扩展能编过）；
- 于是 hipify 下一次的"内容自比对"（`hipify_python.py:1035-1041` 的 `do_write = fout_old.read() != output_source`）
  **必然判定为 changed**，把**每一个**生成文件用**完全相同的内容**重写一遍、mtime 全部刷新。

实测（§3.1）：重跑一次 hipify，**95 个生成文件 mtime 变化、内容 md5 变化数 = 0**。mtime 全部变新 ⇒
四个扩展的全部 `.hip` TU（54 个）都脏 ⇒ 仍然是 16 分钟。

所以本步做两件事：**(1) stamp 化**；**(2) 按内容同步**（未变的文件不落盘、mtime 不动）。

### 2.3 为什么这样改不会造成"陈旧产物不重建"

- **hipify 何时重跑**：stamp 边的 `DEPENDS = cmake/hipify.py + 全部 .cu 源`。任一源文件或 hipify 脚本 mtime
  变新 ⇒ 边重跑 ⇒ staging 树用**当前**内容重新生成 ⇒ 内容同步把**所有内容真的变了的**文件（`.hip`、hipified
  `.cuh`/`.h` 等）复制进构建树 ⇒ ninja 的 depfile（`.ninja_deps`）据此重编**所有**受影响 TU。
  链路里没有"按名字跳过"的捷径，**每一次判断都基于当次生成的真实字节**；
- **脚本变更**：`cmake/hipify.py` 在 `DEPENDS` 里 ⇒ 改了脚本一定重跑 hipify（§3.5 (d3) 实测）；
- **源文件 mtime 变了但内容没变**：`copy_directory_if_different` 不落盘（mtime 不动），但 `OBJECT_DEPENDS`
  让该源文件对应的 `.o` 依赖源 `.cu` 本身 ⇒ 该 TU 仍然重编（§3.5 (d1) 实测 1 TU + 1 链接）；
- **源文件内容变了**：内容同步复制该 `.hip` ⇒ 只重编该 TU（§3.5 (d2) 实测 1 TU + 1 链接，且生成的 `.hip` 里
  确实出现新内容）；
- **`copy_directory_if_different` 的语义是内容比对**（不是 mtime）：单独实测证明"只动 mtime 不复制、
  内容不同才复制且落盘时间为 now"（§3.1）。
- **非 HIP 路径**：改动全在 `hipify_sources_target`（仅 `ARG_LANGUAGE STREQUAL "HIP"` 时由
  `define_extension_target` 调用，`cmake/utils.cmake:654`）与 `vllm_finalize_hipify_target`
  （仅 `CMakeLists.txt:1619-1621` 的 `if (VLLM_GPU_LANG STREQUAL "HIP")` 内调用）里；CUDA/非 GPU 构建不会执行这两段。
  本次也没有碰 `CMakeLists.txt` 的任何 CUDA 分支。

### 2.4 写作用域偏差（必须让 Lead 知道）

任务书说"修 `CMakeLists.txt` 里 hipify 相关规则"，但该规则**实际不在 `CMakeLists.txt`**，而在
`cmake/utils.cmake:100-122` 的 `vllm_finalize_hipify_target()` 里；`CMakeLists.txt:1621` 只是调用它。
CMake 的 API 不允许在 target 创建之后给 custom target 追加命令（`add_custom_target` 无 append 形式），
所以**不修改 `cmake/utils.cmake` 就无法完成本步**。本步实际写作用域：
`cmake/utils.cmake` + 本归档文件；`CMakeLists.txt` 保持零改动（`git diff -- CMakeLists.txt | grep hipify` 为空，
其 mtime 仍是 2026-10-07 16:57 的 S5 版本）。

## 3. 执行过程与原始证据

全部命令在 `z100-build-211` 内执行（`. /mnt/game/port/setenv-211.sh`），构建树
`/mnt/game/port/build-211/temp`，构建副本 `/mnt/game/port/gfx906-vllm-211`。

### 3.1 根因三段实测（改动前）

**(1) stamp 从不创建 ⇒ 每次 ninja 都重跑 hipify**

```console
$ ls -l /mnt/game/port/build-211/temp/CMakeFiles/hipify_all*
ls: cannot access '.../CMakeFiles/hipify_all*': No such file or directory
$ cd /mnt/game/port/build-211/temp && ninja -d explain -n | head -4
ninja explain: output CMakeFiles/hipify_all doesn't exist
ninja explain: /mnt/game/port/build-211/temp/csrc/mamba/mamba_ssm/selective_scan_fwd.hip is dirty
ninja explain: /mnt/game/port/build-211/temp/csrc/mamba/mamba_ssm/selective_scan_fwd.hip is dirty
ninja explain: CMakeFiles/hipify_all is dirty
$ ninja -n | grep -cE '^\['
61                                  # 刚跑完全量构建（S4c 08:18 结束），仍然 61 步
```

**反证：手工造一个 stamp，整棵树立刻变干净**（证明 stamp 缺失就是"每次都全量"的唯一原因）

```console
$ touch CMakeFiles/hipify_all && ninja -d explain -n
ninja: no work to do.
$ ninja -n | grep -cE '^\['
0
$ rm -f CMakeFiles/hipify_all                # 探针用完即删
```

**（2）重跑 hipify 会重写全部生成文件、内容 0 变化、mtime 全部刷新**（所以只加 stamp 还是全量）

```console
# 快照 → 用 build.ninja 里的原命令跑一次 hipify → 再快照
$ CMD=$(sed -n '951p' build.ninja | sed 's/^  COMMAND = //'); eval "$CMD"   # rc=0
$ diff s4d-before2.md5 s4d-after2.md5 | grep -c '^>'
0                                   # 内容变化 0 个文件
$ join -j 3 -o 1.1,2.1,0 s4d-before2.mt s4d-after2.mt | awk '$1!=$2' | wc -l
95                                  # 95 个文件 mtime 变新（53 个 .hip + 42 个 hipified .cuh/.h）
$ grep '\.hip ' s4d-bumped.txt | head -2
csrc/activation_kernels.hip 1791419036.58 1791419050.59
csrc/attention/merge_attn_states.hip 1791419036.60 1791419050.61
```

**（3）`cmake -E copy_directory_if_different` 是内容比对**（独立最小实验，容器内）

```console
$ cmake -E copy_directory_if_different /mnt/game/port/s4d-probe/cpd/src /mnt/game/port/s4d-probe/cpd/dst
# 先只 touch 源（内容相同）：dst mtime 保持 2026-10-08 08:24:41.238315476 不变（未复制）
# 再改源内容：dst 被复制，mtime 更新为 08:24:42.434328572，内容 = changed-content
```

**（4）`OBJECT_DEPENDS` 被 Ninja 生成器支持**（最小工程验证：出现在 `.o` 边的隐式依赖上）

```console
$ grep -A1 "gen/x.cpp.o:" build/build.ninja
build CMakeFiles/x.dir/gen/x.cpp.o: CXX_COMPILER__x_unscanned_ ... | /mnt/game/port/s4d-probe/objdep/src/x.cu || ...
```

### 3.2 改动落地 + 第一次裸 `ninja`（全量）：rc=0，963 s

```console
$ cp -p /src/cmake/utils.cmake /mnt/game/port/gfx906-vllm-211/cmake/utils.cmake
5f97a913c49f1cf37317d8a8e78156a2  /src/cmake/utils.cmake
5f97a913c49f1cf37317d8a8e78156a2  /mnt/game/port/gfx906-vllm-211/cmake/utils.cmake
86eee66aa3bb600299fa3234a8f1dce7  /mnt/game/port/gfx906-vllm-211/cmake/utils.cmake.s4d-orig   # 回滚用副本

$ MAX_JOBS=4 LOG=/mnt/game/port/s4d-full-ninja.log bash /mnt/game/port/build-211.sh
===== S4b build 2026-10-08T08:27:26+08:00 targets:  MAX_JOBS=4 =====
[0/1] Re-running CMake...
[1/61] Running hipify on all extension source files.
[2/60] Building HIP object CMakeFiles/_C.dir/csrc/mamba/gfx906_decode_kernels.hip.o
...
[58/60] Linking HIP shared module _rocm_C.abi3.so
[59/60] Building HIP object CMakeFiles/_moe_C_stable_libtorch.dir/csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.hip.o
[60/60] Linking HIP shared module _moe_C_stable_libtorch.abi3.so
----- ninja rc=0 2026-10-08T08:43:29+08:00 -----
S4D_START 2026-10-08T08:27:26+08:00 epoch=1791419246
S4D_END   2026-10-08T08:43:29+08:00 epoch=1791420209 rc=0        # wall_s = 963 = 16 min 3 s
error: 0   FAILED: 0
```

> 说明：这次为什么是"真全量"？因为 §3.1 的幂等性探针把构建树里全部 `.hip` 的 mtime 刷到了 `.o` 之后
> （这正是本步要消除的现象本身），所以第一次 `ninja` 合法地重编了 **54 个 `.hip` TU**
> （`_C` 22 / `_C_stable_libtorch` 27 / `_moe_C_stable_libtorch` 4 / `_rocm_C` 1）—— 这同时构成
> "stamp 化 + 内容同步后全量构建仍可走通"的最强证据（不是空跑）。

重生成的 ninja 边（关键点：stamp 是 OUTPUT、`restat = 1`、内容同步、`.o` 上的隐式 `.cu` 依赖）：

```console
$ grep -n "^build CMakeFiles/hipify_all.stamp" build.ninja | cut -c1-120
827:build CMakeFiles/hipify_all.stamp csrc/mamba/mamba_ssm/selective_scan_fwd.hip csrc/mamba/gfx906_decode_kernels.hip c
$ awk '/^build CMakeFiles\/hipify_all.stamp/{f=1} f&&/COMMAND = |DESC|restat/{print; if($0~/restat/)exit}' build.ninja
  COMMAND = cd /mnt/game/port/build-211/temp/hipify_stage && /usr/bin/python .../cmake/hipify.py -p .../csrc -o .../hipify_stage/csrc csrc/mamba/...
  DESC = Running hipify on all extension source files.
  restat = 1
$ grep -n "cmake_object_order_depends_target__C:" -A1 build.ninja
86:build cmake_object_order_depends_target__C: phony || hipify_all
88:build CMakeFiles/_C.dir/csrc/mamba/mamba_ssm/selective_scan_fwd.hip.o: HIP_COMPILER___C_unscanned_RelWithDebInfo .../selective_scan_fwd.hip | /mnt/game/port/gfx906-vllm-211/csrc/mamba/mamba_ssm/selective_scan_fwd.cu || cmake_object_order_depends_target__C
$ ls -l CMakeFiles/hipify_all.stamp
-rw-r--r-- 1 root root 0 2026-10-08_08:27:46 CMakeFiles/hipify_all.stamp
```

### 3.3 改动前后"第二次 ninja"对比

| 观测项 | 改动前 | 改动后 |
|---|---|---|
| `ls CMakeFiles/hipify_all*` | `No such file or directory` | `CMakeFiles/hipify_all.stamp`（0 B，mtime=hipify 结束时刻） |
| 刚构建完的 `ninja -n \| grep -cE '^\['` | **61** | **0** |
| `ninja -d explain -n` | `output CMakeFiles/hipify_all doesn't exist` … | `ninja: no work to do.` |
| 第二次裸 `ninja` 的实际结果 | 61 步、`wall_s=986`（16 min 26 s，S4c §3.10 实测），53 个 TU 全重编 | `ninja: no work to do.` |

```console
# 改动前（S4c 归档 §3.10 原文，同一棵树、同一规则）：
[1/61] Running hipify on all extension source files.
...
[60/60] Linking HIP shared module _moe_C_stable_libtorch.abi3.so
# wall_s=986（16 min 26 s）；error: 0、FAILED: 0

# 改动后（本步实测）：
$ ninja
ninja: no work to do.
rc=0  wall=0.096s
$ ninja -n | grep -cE '^\['
0
$ ninja -d explain -n
ninja: no work to do.
# 项目驱动脚本同样 no-op：
$ MAX_JOBS=4 LOG=/mnt/game/port/s4d-driver-noop.log bash /mnt/game/port/build-211.sh
===== S4b build 2026-10-08T08:45:28+08:00 targets:  MAX_JOBS=4 =====
ninja: no work to do.
----- ninja rc=0 2026-10-08T08:45:29+08:00 -----
```

### 3.4 身份检查：四个原始 `.so` 与 S4c 基线逐字节相同

```console
$ cat /mnt/game/port/s4c-probe/raw-before/md5.txt          # S4c 第一轮裸全量 ninja 的原始产物
048736b122bbf5e4932fe6c4c89266bd  .../raw-before/_C.abi3.so
7068229e9c37ca3d8dd2976b8bc9fb4c  .../raw-before/_C_stable_libtorch.abi3.so
af01ed9bb3fc5ac913bff845f502e27a  .../raw-before/_moe_C_stable_libtorch.abi3.so
019b327b01cfa76e389e427f1849433a  .../raw-before/_rocm_C.abi3.so

$ cd /mnt/game/port/build-211/temp
$ md5sum _C.abi3.so _rocm_C.abi3.so _C_stable_libtorch.abi3.so _moe_C_stable_libtorch.abi3.so
048736b122bbf5e4932fe6c4c89266bd  _C.abi3.so
019b327b01cfa76e389e427f1849433a  _rocm_C.abi3.so
7068229e9c37ca3d8dd2976b8bc9fb4c  _C_stable_libtorch.abi3.so
af01ed9bb3fc5ac913bff845f502e27a  _moe_C_stable_libtorch.abi3.so

$ for s in _C _rocm_C _C_stable_libtorch _moe_C_stable_libtorch; do
    cmp -s .../raw-before/$s.abi3.so $s.abi3.so && echo "$s IDENTICAL"; done
_C                             IDENTICAL
_rocm_C                        IDENTICAL
_C_stable_libtorch             IDENTICAL
_moe_C_stable_libtorch         IDENTICAL
```

> 口径：**用构建树里的原始产物**（裸 ninja 直接产出的 `build-211/temp/*.so`），不是 install 后的
> （`cmake --install` 会改 RPATH）。安装面本次**未重跑**：因为四个原始产物与 S4c 的
> `raw-before/` 逐字节相同，`vllm/*.so` 仍是 S4c 装出来的那一份（mtime 07:43–07:58，md5
> `6fba0596… / 5d1e50e1… / cc2aca69… / 996b282c…`），门禁就是在它上面复测的。

### 3.5 增量正确性 (d)：三种改动各贴原始输出与耗时

**(d1) 只 touch 一个真实源文件**（`csrc/fused_qknorm_rope_kernel.cu`，md5 不变、mtime +）

```console
$ touch /mnt/game/port/gfx906-vllm-211/csrc/fused_qknorm_rope_kernel.cu
$ ninja
[1/61] Running hipify on all extension source files.
[2/3] Building HIP object CMakeFiles/_C.dir/csrc/fused_qknorm_rope_kernel.hip.o
[3/3] Linking HIP shared module _C.abi3.so
rc=0  ninja_progress_lines=3  total_lines=186  wall=43.1s
--- 构建树 csrc 里 mtime 变化的文件数: 0            # 内容同步一个文件都没落盘
--- plan after: 0
--- _C.abi3.so md5: 048736b122bbf5e4932fe6c4c89266bd（不变）
```

**(d2) 真改内容**（在 `csrc/fused_qknorm_rope_kernel.cu` 末尾追加 1 行注释）

```console
$ printf '\n// S4d incrementality probe: this line is removed again right after.\n' >> csrc/fused_qknorm_rope_kernel.cu
$ ninja
[1/61] Running hipify on all extension source files.
[2/3] Building HIP object CMakeFiles/_C.dir/csrc/fused_qknorm_rope_kernel.hip.o
[3/3] Linking HIP shared module _C.abi3.so
rc=0  ninja_progress_lines=3  total_lines=186  wall=40.1s
--- 构建树 csrc 里 mtime 变化的文件（恰好 2 个）---
  changed: csrc/fused_qknorm_rope_kernel.cu      # hipify 的 copytree 刷新
  changed: csrc/fused_qknorm_rope_kernel.hip     # 内容同步复制（这就是"只有受影响 TU 变脏"的来源）
--- 生成的 .hip 里确实带上了新内容 ---
$ tail -1 csrc/fused_qknorm_rope_kernel.hip
// S4d incrementality probe: this line is removed again right after.
--- plan after: 0
```

**(d3) 只改 hipify 脚本**（`touch cmake/hipify.py`，内容同步后生成物不变）

```console
$ touch /mnt/game/port/gfx906-vllm-211/cmake/hipify.py && ninja
[1/61] Running hipify on all extension source files.
rc=0  ninja_progress_lines=1  wall=6.1s
--- 构建树 csrc mtime 变化: 0；四个 .so md5 全部不变 ---
```

**收尾：把 (d2) 的探针内容撤掉并复验身份**（`cp -p` 回原文件 + `touch` + `ninja`）

```console
$ ninja
[1/61] Running hipify on all extension source files.
[2/3] Building HIP object CMakeFiles/_C.dir/csrc/fused_qknorm_rope_kernel.hip.o
[3/3] Linking HIP shared module _C.abi3.so
rc=0 wall=39.6s
--- 生成物里探针行: 0 个 ---
$ cmp -s .../raw-before/_C.abi3.so _C.abi3.so && echo IDENTICAL
IDENTICAL                                     # 往返后回到 S4c 基线字节
$ ninja
ninja: no work to do.   rc=0 wall=0.108s  plan=0
```

**本次全量构建的耗时构成**（`.ninja_log` 里该轮记录的 54 个 `.hip` TU + 3 个扩展链接；单位 s。第 4 个链接
`_C.abi3.so` 的日志条目未落在统计窗口内，其量级与其余链接相同）：

| 对象 | n | min | median | p90 | max |
|---|---|---|---|---|---|
| `.hip` TU 编译（全部扩展） | 54 | 2.0 | 19.2 | 69.5 | 152.7 |
| 链接（`_C_stable_libtorch`/`_rocm_C`/`_moe_C_stable_libtorch`） | 3 | 0.2 | 0.3 | 0.3 | 0.3 |
| `_C` TU | 22 | 2.0 | 42.4 | 91.8 | 152.7 |
| `_C_stable_libtorch` TU | 27 | 2.0 | 10.2 | 21.5 | 67.7 |
| `_moe_C_stable_libtorch` TU | 4 | 4.8 | 21.6 | 28.2 | 133.9 |
| `_rocm_C` TU | 1 | 88.4 | 88.4 | 88.4 | 88.4 |

> hipify 本身约 **6 s**（`.ninja_log` 里 stamp 边的时长），cmake 重配置约 14.5 s（只在改 CMake 时发生）。

### 3.6 门禁（复测）

```console
$ source /mnt/game/port/setenv-211.sh; cd "$REPO_DIR"
$ python -c "import torch, vllm, vllm._C, vllm._rocm_C, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch; print('IMPORT_OK_ALL4')"
IMPORT_OK_ALL4
rc=0

$ for so in vllm/_C.abi3.so vllm/_rocm_C.abi3.so vllm/_C_stable_libtorch.abi3.so vllm/_moe_C_stable_libtorch.abi3.so; do
    printf "%-34s gfx: " "$(basename $so)"
    strings -a $so | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sed 's/.*--//' | sort -u | tr '\n' ' '
    printf "| count_distinct="; strings -a $so | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sed 's/.*--//' | sort -u | wc -l
  done
_C.abi3.so                         gfx: gfx906 | count_distinct=1
_rocm_C.abi3.so                    gfx: gfx906 | count_distinct=1
_C_stable_libtorch.abi3.so         gfx: gfx906 | count_distinct=1
_moe_C_stable_libtorch.abi3.so     gfx: gfx906 | count_distinct=1
```

### 3.7 资源与"未破坏性"自检

```console
$ cat /sys/fs/cgroup/memory.events
low 0
high 0
max 2250          # 触到 8 GiB 上限触发回收的次数（不是 kill）；oom 0、oom_kill 0
$ cat /sys/fs/cgroup/memory.current
3754696704        # 构建结束后 3.5 GiB
$ docker ps -a --format '{{.Names}}\t{{.Status}}'
z100-triton     Up About an hour
z100-build-211  Up 2 hours          <- 本步唯一写入的构建树所在容器
z100-build      Up 8 hours          <- 2.10 基线，本步只读，未触碰
qwen38-vllm     Exited (137) About an hour ago   <- 外部停的，本步未 start/kill/exec
```

### 3.8 残余项：ninja 的 mtime 语义（**唯一**被本步暴露出来的新边界）

本步在收尾时无意中复现了一个边界：**把内容回退成旧版本、但 mtime 也回到 stamp 之前**（`cp -p` 旧副本）时，
stamp 边被判为 clean，hipify 不重跑，构建树里的 `.hip` 会**停留在上一次的内容**：

```console
$ cp -p <旧副本> csrc/fused_qknorm_rope_kernel.cu        # 内容回退，mtime=08:43:50
$ stat -c '%y %n' csrc/fused_qknorm_rope_kernel.cu CMakeFiles/hipify_all.stamp
src   2026-10-08 08:43:50
stamp 2026-10-08 08:44:47                                # stamp 比源新 ⇒ 边干净
$ ninja -d explain -n
ninja: no work to do.
$ grep -c 'S4d incrementality probe' csrc/fused_qknorm_rope_kernel.hip
1                                                        # 生成物仍是上一版（陈旧）
# 规避（本步即用它恢复自洽，成功后已复验 cmp IDENTICAL）：
$ touch csrc/fused_qknorm_rope_kernel.cu && ninja
[1/61] Running hipify ... [2/3] Building ... [3/3] Linking ...
$ grep -c 'S4d incrementality probe' csrc/fused_qknorm_rope_kernel.hip
0
```

判定与边界说明：

- 这是 **ninja/CMake 的通用假设**（"mtime 只向前走，且不用内容哈希"），对**任何**文件都成立：
  例如把一个内容不同的旧 `csrc/torch_bindings.cpp` 用 `cp -p` 覆盖回去，`.o` 同样不会重编。
  本步并没有引入新的机制类别；
- 但它确实**相对改动前收窄了容错**：改动前 hipify 每次必跑，所以 csrc 源文件"mtime 回退但内容变化"
  这种情况会被顺带纠正；现在不再自动纠正。
- 触发条件很窄：**内容变了 + mtime 倒退到比 stamp 旧**（`cp -p` 旧副本 / `tar -xp` 旧归档 / 恢复到旧快照）。
  正常编辑（mtime=now）与 `git checkout`（mtime=now）都不受影响；
- 规避方式（任选）：`touch <改过的文件>`、`rm CMakeFiles/hipify_all.stamp`、`ninja -t clean <target>`；
  建议写入 S6–S9 的操作约定（恢复到旧快照后统一 `touch` 一下再 `ninja`）。

## 4. 门禁判定

| # | 门禁 | 结果 | 证据 |
|---|---|---|---|
| 1 | hipify 步骤正确 stamp 化（stamp 真正被创建） | ✅ | §3.2：`build CMakeFiles/hipify_all.stamp … restat = 1`；`ls` 存在 0 B stamp |
| 2 | 语义不变（hipified 源/产物一致） | ✅ | §3.4 四个原始 `.so` 与 S4c 基线 `cmp` 全 0；§3.5(d1)(d3) 生成树 mtime 变化 0 |
| 3 | 源文件变更仍重跑 hipify 且只重编受影响 TU | ✅ | §3.5(d1) 3 步/43.1 s（1 TU+1 链接）；§3.5(d2) 3 步/40.1 s，且 `.hip` 带上新内容 |
| 4 | hipify 脚本变更仍重跑 hipify | ✅ | §3.5(d3)：`touch cmake/hipify.py` → `[1/61] Running hipify`，6.1 s，0 TU |
| 5 | 不出现"陈旧产物不重建"陷阱 | ✅（附 §3.8 边界） | §2.3 论证 + (d1)/(d2) 实测；唯一的通用 ninja mtime 边界已单列并给规避 |
| 6 | 非 HIP/CUDA 路径行为不变 | ✅（静态论证） | §2.3：改动只在 HIP-only 函数内；`CMakeLists.txt` 零改动（`git diff \| grep hipify` 为空） |
| 7 | (a) 第一次裸 `ninja`（全量）rc=0 | ✅ | §3.2：`[60/60]`、`rc=0`、`error: 0`、`FAILED: 0`、963 s |
| 8 | (b) 第二次裸 `ninja` no-op、秒级 | ✅ | §3.3：`ninja: no work to do.`、rc=0、**0.096 s**；`ninja -n` = 0 步；驱动脚本同样 no-op |
| 9 | (c) 身份检查（原始产物 md5 = S4c 基线） | ✅ | §3.4：`048736b1…`/`019b327b…`/`7068229e…`/`af01ed9b…` + 4× `cmp IDENTICAL` |
| 10 | (d) 增量重建：只重编受影响 TU、耗时远小于全量 | ✅ | §3.5：3 步 / 40–43 s vs 全量 60 步 / 963 s（**快 ~23×**） |
| 11 | 门禁 `IMPORT_OK_ALL4` | ✅ | §3.6 rc=0 |
| 12 | 四个 `.so` 只含 `gfx906` | ✅ | §3.6 四个 `count_distinct=1`，均 `gfx906` |
| 13 | 硬约束（不跑自编译 HIP 可执行程序、不碰 `qwen38-vllm`、不动 `z100-build`、产物只在 `/mnt/game`、`MAX_JOBS=4`、8 GiB 不 OOM-kill） | ✅ | §1.2 / §3.7：`oom 0`、`oom_kill 0`；`qwen38-vllm` 仍 `Exited (137)` |

**总判定：✅ 通过**（含 §2.4 写作用域偏差与 §3.8 通用 ninja mtime 边界两条说明）。

## 5. 未完成项 / 阻塞与原因

- **无阻塞项**。明确"未执行/未做"的项：
  1. **写作用域偏差**（§2.4）：规则实际在 `cmake/utils.cmake` 而非任务书写的 `CMakeLists.txt`；CMake API 不允许
     给已有 custom target 追加命令，故只能改 `cmake/utils.cmake`。**需要 Lead 确认**这一处扩大是否接受；
     若必须只动 `CMakeLists.txt`，则该需求无法实现（要么在 CMakeLists.txt 里重复实现
     `vllm_finalize_hipify_target`，要么放弃本步）。
  2. **未在 2.10 树（`/mnt/game/port/build-gfx906` / `z100-build`）落地**：硬约束禁止动该树。本改动与 torch
     版本无关，同步只需 `cp -p cmake/utils.cmake <2.10 副本>/cmake/` + 一次 `ninja`；注意 2.10 树的
     `.hip` 已被旧规则刷到比 `.o` 新，所以**第一次 ninja 仍会是全量**（约 16–18 min），之后才享受增量。
  3. **未重跑 install / 真机 e2e / 回归套件**：四个原始产物与 S4c 逐字节相同 ⇒ 安装面与 S4c 一致（md5 未变），
     门禁已在安装面上复测；S4c 已跑过的 gguf 13/13、fused sigmoid gating 5/5 未重复跑（非本步范围，
     S9 复核）。真机冒烟属 S9。
  4. **未做 CUDA 构建验证**：本机无 CUDA 工具链/设备；"非 HIP 路径不变"是静态论证（HIP-only 调用点）。
  5. **§3.8 的 ninja mtime 边界未在构建系统内闭合**：要闭合就得每次构建都跑一个"内容哈希守卫"，
     这会让 `ninja` 不再打印 `ninja: no work to do.`（与验收 (b) 直接冲突），因此选择保留 stamp 语义 +
     文档化 + 规避命令。
  6. **未运行任何自编译 HIP 独立可执行程序**（硬约束，历史事故）。
- 本步遗留的**非仓库**文件（都在 `/mnt/game/port` 下）：
  `s4d-full-ninja.log`（全量 ninja 原文，412 行）、`s4d-full-ninja.time`（起止 epoch/rc）、
  `s4d-driver-noop.log`、`s4d-hipify-raw2.log`、`s4d-probe/{before2,after2}.{mt,md5}`、`s4d-bumped.txt`、
  `s4d-d{1,2,3}-{before,after}.mt`、`s4d-fix-{before,after}.mt`、`s4d-probe/cpd/`、
  `s4d-probe/objdep/`、`s4d-probe/fused_qknorm_rope_kernel.cu.orig`、
  `build-211/temp/{build.ninja,hipify_stage/,CMakeFiles/hipify_all.stamp}`、
  `gfx906-vllm-211/cmake/utils.cmake.s4d-orig`（回滚副本）。

## 6. 对后续步骤的影响

**迭代重建耗时（-j4，本机，实测 + 由 `.ninja_log` 推算）**：

| 场景 | 改动前 | 改动后 | 依据 |
|---|---|---|---|
| 无改动再跑一次 `ninja` | 61 步 / **986 s**（S4c §3.10） | **0.096 s，`ninja: no work to do.`** | §3.3 |
| 只改 `cmake/hipify.py`（生成物不变） | 全量 ~986 s | **6.1 s，0 TU** | §3.5(d3) |
| 改 1 个 `.cu`（内容变，mtime 向前） | 全量 ~986 s | **40 s**（hipify 6 s + 该 TU 33 s + 链接 0.3 s） | §3.5(d2) |
| 只 `touch` 1 个 `.cu`（内容不变） | 全量 ~986 s | **43 s**（1 TU + 1 链接） | §3.5(d1) |
| 改 1 个被广泛 include 的 hipified 头 | 全量 ~986 s | 只重编 include 它的 TU（由 `.ninja_deps` 决定），量级仍远小于全量 | 机制论证 |
| 改 CMake 规则 / 冷树全量 | ~963–1052 s | ~963 s（不变，本来就该全量） | §3.2 |

- **S6（triton gfx906）**：现在可以放心用裸 `ninja`/`build-211.sh`（无 target 列表）迭代，改一个 torch/triton
  侧文件或 csrc 小改动的成本从 16–18 min 降到 **6–40 s**；triton 相关目标在本树仍是空 phony（S4b §5.3），不受影响。
- **S7（内存/KV cache/graph）**：主要是 python/config 侧改动，多数情况连 hipify 都不会触发（`ninja: no work to do.`）；
  需要重编时只付 1 个 TU 的代价。
- **S8（attention/量化路径核对）**：改 attention/量化内核时，`.hip` 粒度增量生效；被改内核的 p90 TU 成本可达
  ~70–150 s（§3.5 表），仍比 16 min 低一个数量级。
- **S9（验收/固化）**：
  - 建议把 **§3.8 的规避约定**写进固化文档（"恢复到旧快照后先 `touch` 再 `ninja`"）；
  - 构建耗时可重估：全量 1 次 ≈ 16 min，迭代 ≈ 秒级–1 min；
  - 未改动语义 ⇒ S4c 的构建身份基线（`raw-before/`）继续有效，S9 的对拍可直接复用。
- **2.10 基线（`z100-build` / `build-gfx906`）**：本步未动。若要回退到 2.10 路线，同步 `cmake/utils.cmake`
  后第一次 ninja 为全量、随后同样获得增量；该改动对 torch 2.10/2.11 等价（与 `CUDA_MACROS_MAP` 修复无关）。
- **回滚**：`cp -p /mnt/game/port/gfx906-vllm-211/cmake/utils.cmake.s4d-orig <repo>/cmake/utils.cmake`，再 `ninja`
  （会自动重跑 cmake configure 并恢复"每次全量"的旧行为；`hipify_stage/` 与 stamp 文件可留着不管）。

## 7. 复现方式

```bash
# 0) 前置：容器 z100-build-211 在跑；/mnt/game/port/{gfx906-vllm-211,build-211} 是 S4c 树。
#    仓库侧应用本档 §2.1 的改动（cmake/utils.cmake:74-90 / 124-152 / 154-169）。

# 1) 同步进构建副本（/src 是工作区只读挂载）
docker exec z100-build-211 bash -lc '
  cp -p /mnt/game/port/gfx906-vllm-211/cmake/utils.cmake /mnt/game/port/gfx906-vllm-211/cmake/utils.cmake.s4d-orig
  cp -p /src/cmake/utils.cmake /mnt/game/port/gfx906-vllm-211/cmake/utils.cmake
  md5sum /src/cmake/utils.cmake /mnt/game/port/gfx906-vllm-211/cmake/utils.cmake'     # 期望两个 md5 相同

# 2) 第一次裸 ninja（默认 all）：会自动重跑 cmake configure + hipify + 全量重编
docker exec z100-build-211 bash -lc 'MAX_JOBS=4 LOG=/mnt/game/port/s4d-full-ninja.log bash /mnt/game/port/build-211.sh'
#    期望：... [60/60] Linking HIP shared module _moe_C_stable_libtorch.abi3.so / ----- ninja rc=0 -----

# 3) 第二次裸 ninja：必须是 no-op
docker exec z100-build-211 bash -lc '. /mnt/game/port/setenv-211.sh >/dev/null 2>&1; cd /mnt/game/port/build-211/temp; time ninja'
#    期望：ninja: no work to do.    （<0.2 s）
docker exec z100-build-211 bash -lc 'cd /mnt/game/port/build-211/temp; ninja -n | grep -cE "^\["'   # 期望 0
docker exec z100-build-211 bash -lc 'cd /mnt/game/port/build-211/temp; ninja -d explain -n'        # 期望 no work to do.

# 4) 身份检查（必须用构建树原始产物，不要用 install 后的）
docker exec z100-build-211 bash -lc 'cd /mnt/game/port/build-211/temp
  md5sum _C.abi3.so _rocm_C.abi3.so _C_stable_libtorch.abi3.so _moe_C_stable_libtorch.abi3.so
  for s in _C _rocm_C _C_stable_libtorch _moe_C_stable_libtorch; do
    cmp -s /mnt/game/port/s4c-probe/raw-before/$s.abi3.so $s.abi3.so && echo "$s IDENTICAL"; done'
#    期望：048736b1… / 019b327b… / 7068229e… / af01ed9b…，四行 IDENTICAL

# 5) 增量正确性：只 touch 一个源 -> 只重编该 TU + 链接
docker exec z100-build-211 bash -lc '. /mnt/game/port/setenv-211.sh >/dev/null 2>&1
  cd /mnt/game/port/build-211/temp
  touch /mnt/game/port/gfx906-vllm-211/csrc/fused_qknorm_rope_kernel.cu
  time ninja'      # 期望：hipify + [2/3] 该 TU + [3/3] Linking _C.abi3.so，~40 s

# 6) 真改内容 -> 生成的 .hip 带上新内容，且仍只重编该 TU
docker exec z100-build-211 bash -lc '. /mnt/game/port/setenv-211.sh >/dev/null 2>&1
  cd /mnt/game/port/build-211/temp
  printf "\n// probe\n" >> /mnt/game/port/gfx906-vllm-211/csrc/fused_qknorm_rope_kernel.cu
  ninja; tail -1 csrc/fused_qknorm_rope_kernel.hip'          # 期望打印 "// probe"
#    收尾：撤掉该行并 touch 后 ninja，再复验 §7-4 的 md5/cmp

# 7) 门禁
docker exec z100-build-211 bash -lc '. /mnt/game/port/setenv-211.sh >/dev/null 2>&1; cd "$REPO_DIR"
  python -c "import torch, vllm, vllm._C, vllm._rocm_C, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch; print(\"IMPORT_OK_ALL4\")"
  for so in vllm/*.so; do printf "%-34s " "$(basename $so)"
    strings -a $so | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sed "s/.*--//" | sort -u | tr "\n" " "; echo; done'
#    期望：IMPORT_OK_ALL4；四个文件各只列 gfx906
```

## 8. 参考

- 前一档（本步直接承接）：[z100-04b-stack-upgrade-result-2.md](z100-04b-stack-upgrade-result-2.md)
  **§3.10**（`hipify_all` 每次重跑的实测与"两次 ninja 都是全量"的记录）、§3.8（"身份检查要用构建树原始产物、
  不要用安装后 md5"的口径）、§6（构建耗时预算）
- 设计文档：[docs/design/z100-04-build-system-dtk.md](../design/z100-04-build-system-dtk.md)（§3 CMake、§4 构建命令）
- Todolist 与硬约束：[docs/z100-todolist.md](../z100-todolist.md)（S4d 行、硬约束 1/2/3/10）
- 归档规则与模板：[docs/result/README.md](README.md)
- 关键源码位置（本步改动）：
  - `cmake/utils.cmake:74-90`（hipify 输出名 → 原源文件配对）
  - `cmake/utils.cmake:124-152`（stamp + staging + `copy_directory_if_different` 内容同步）
  - `cmake/utils.cmake:154-169`（`OBJECT_DEPENDS` + `add_custom_target(hipify_all DEPENDS <stamp>)`）
  - 调用点（未改动）：`CMakeLists.txt:1619-1621`、`cmake/utils.cmake:654`
- 机制真相源（z100-build-211 容器内）：
  - `/mnt/game/port/gfx906-vllm-211/cmake/hipify.py:77-121`（`_neutralize_dtk_macros_include`：事后替换
    DTK hipify 注入的 `#include <ATen/dtk_macros.h>`，是"每次重写全部生成文件"的根因）
  - `/usr/local/lib/python3.10/dist-packages/torch/utils/hipify/hipify_python.py:1031-1054`
    （`do_write = fout_old.read() != output_source` 的写盘跳过判据）
- 本步产物/证据：`/mnt/game/port/s4d-full-ninja.log`（963 s 全量 ninja 原文）、`s4d-full-ninja.time`、
  `s4d-driver-noop.log`、`s4d-probe/`（幂等性快照、`copy_directory_if_different` 与 `OBJECT_DEPENDS` 最小实验）、
  `s4d-d{1,2,3}-*.mt`、`/mnt/game/port/s4c-probe/raw-before/`（S4c 基线原始产物 + `md5.txt`）
