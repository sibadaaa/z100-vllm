# S11e P2：gfx906 W8A8 INT8 `sdot4` GEMM 落到本树 — 执行结果

> 归档时间：2026-10-09
> 执行者：子 agent（s11e-sdot4，**续跑第 2 任**；第 1 任静默失败、无遗言）
> 对应设计文档：`docs/海光Z100-性能优化方案.md` §4（P2，B 路线"直接迁移前序已验证的 `int8_sdot4` 算子"）、§4A（算力清单）、§4C.1（exact-shape 战术表 / no shape regresses）
> 前一档：无（本步为首档）。相关背景档：`docs/result/z100-11d-z100-isa-capability-audit-result.md`（指令审计）、`docs/result/z100-11b-perf-profiling-result.md`（P0 剖析）、`docs/result/z100-07c-packed-silu-csrc-fix-result.md` + `docs/result/z100-04d-build-incrementality-result.md`（csrc 增量重建流程）
> 状态：🔶 **部分完成** —— 内核/接入/构建/数值/性能**全过**（63/63 数值、63/63 无 shape 变慢、只含 gfx906）；**端到端不通过**：`VLLM_DCU_USE_INT8_SDOT4=1` 会让 27B 的 `torch.compile` 抛 `ConstraintViolationError` 起不来（同一命令 `=0` 正常，378.84 ms/token）。崩因已用三次对照实验定位到**新自定义 op 本体**（§3.6.3），**生产当前必须保持该开关为 0**（§6）

## 0. 承接说明（第 1 任遗留物清点，**原始核对输出**）

第 1 任（2026-10-08 23:5x – 2026-10-09 00:08）在同一任务上无遗言退出。续跑前先核对工作区，确认**已有 X / 缺 Y**，不重写已有成果。

### 0.1 已有产物（核对为真）

| 类别 | 产物 | 核对结果 |
|---|---|---|
| 新内核 | `csrc/rocm/w8a8_sdot4.cu` | 存在，**517 行**，22124 B，mtime `2026-10-09 00:02`；文件头含 `Provenance:` 注释，注明来自 `/home/lyh/server/int8_sdot4/w8a8_sdot4_kernel.hip`（484 行，同作者，Apache-2.0） |
| 新内核 | `csrc/rocm/w8a8_sdot4.h` | 存在，**34 行**，1835 B，mtime `2026-10-08 23:49` |
| 构建注册 | `CMakeLists.txt:1548` | `set(VLLM_ROCM_EXT_SRC ... "csrc/rocm/w8a8_sdot4.cu" ...)` 已列出 |
| C++ 绑定 | `csrc/rocm/ops.h:5` | `#include "rocm/w8a8_sdot4.h"` 已加 |
| C++ 绑定 | `csrc/rocm/torch_bindings.cpp:97-116` | `w8a8_sdot4_gemv` / `w8a8_sdot4_gemm` 的 `def`+`impl`（`torch::kCUDA`）+ 同 TU 的 `TORCH_LIBRARY_IMPL_EXPAND(..., Meta, ...)` 两个 `*_meta` 实现 |
| Python 绑定 | `vllm/_custom_ops.py:856-883` | 两个 thin wrapper，**+36 行** |
| Python 内核类 | `vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py` | 存在，**174 行**，6555 B，mtime `2026-10-08 23:50`（新增文件） |
| 内核表 | `vllm/model_executor/kernels/linear/__init__.py:421-425` | ROCM INT8 表**第 0 位**插入 `Gfx906Sdot4Int8ScaledMMLinearKernel`（Triton 作回退）；`__init__.py:213` 补 import，**+9 行** |
| 内核表 | `vllm/model_executor/kernels/linear/scaled_mm/__init__.py:23,66` | 补 import + `__all__`，**+4 行** |
| 环境开关 | `vllm/envs.py:168` / `vllm/envs.py:1412-1414` | `VLLM_DCU_USE_INT8_SDOT4: bool = True`，`os.getenv(...,"True").lower() in ("true","1")`，**默认开** |
| 构建产物 | `build-s11e.log` | 2 行：`[1/2] Building CXX ... torch_bindings.cpp.o` / `[2/2] Linking HIP shared module _rocm_C.abi3.so` |
| 构建产物 | `build-s11e-2.log`（28 KB） | 末尾 `[2/3] Building HIP object CMakeFiles/_rocm_C.dir/csrc/rocm/w8a8_sdot4.hip.o` + `[3/3] Linking HIP shared module _rocm_C.abi3.so` → **本 TU 编译 + 链接成功** |
| canonical 安装 | `/mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so` | **已安装新版**：`ebd6fb8dac572514cf94ca7ec7d828a7`，**36,182,960 B**，mtime `2026-10-09 00:03`（与 `build-211/temp/_rocm_C.abi3.so` 同尺寸同时间戳） |
| 回滚件 | `/mnt/game/port/s11e/_rocm_C.abi3.so.before` | 旧版：`5d1e50e1a77de74f2d308519f8120745`，**35,156,368 B**，mtime `2026-10-07 23:48` |
| perf 日志 | `/mnt/game/port/s11e/bench.log`（31 KB，mtime 00:08） | **完整跑完** 7 shape × 9 M = 63 组 ABBA（`BENCH_SUMMARY` 63 行 JSON），`elapsed 92.6s`；最小加速比 **1.02×**（N=5120 K=6144 M=256） |

### 0.2 第 1 任**未做**（本次须补）

- **归档文件缺失**：`docs/result/z100-11e-w8a8-sdot4-result.md` 不存在（本文件即补齐）。
- **数值验收未落盘**：`/mnt/game/port/s11e/` 下**没有** `check.log` / `int.log`；`tools/dcu/s11e_sdot4_eval.py --mode check`（rel_err vs fp32 反量化参考）与 `--mode int`（int32 精确性）**无任何输出证据**。
- **端到端未执行**：`/tmp/s11e/run_27b_tp2.sh` + `/tmp/s11e/req_27b.py` 在 00:06 写好，但 `/mnt/game/port/s11e/27b-*.log` **不存在** → 27B TP=2 开关开/关对比未跑。
- 第 1 任以 root 身份在容器 `z100-build-211` 内运行（`/mnt/game/port/s11e/` 为 `root:root`），本任以 `lyh` 起，见 §5 环境限制。

## 1. 目标与验收标准

### 1.1 设计文档门禁（`docs/海光Z100-性能优化方案.md` §4 P2 原文要点）

- **B 主路线**：直接迁移前序已验证的 `int8_sdot4` 算子（`/home/lyh/server/int8_sdot4/w8a8_sdot4_kernel.hip`，484 行，GEMV + tiled GEMM）；
  - 形态照抄：`__builtin_amdgcn_sdot4(..., false)`（**不要** `v_dot4c_i32_i8`）；**M≤8** = 每输出列一个 block、64 线程切 K/4 个 int32、int32 精确累加 + LDS 归约 + epilogue `acc * x_s * w_s (+bias)`；**M>32** = 64×64 tile + BK=32 双缓冲 LDS；
  - 接入：编进 `_C`/`_rocm_C`（CMake），`is_supported = on_gfx906()` 并插到 ROCm 表**第 0 位**（优先于 Triton），Triton 作回退；
  - **编译纪律**：`strings <so> | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+"` 必须仍只含 gfx906；装进 canonical 副本并记录新旧 md5；**不落地独立可执行程序**；
  - **验收**：同 shape vs fp32 反量化参考 `rel_err ≤ 1e-2`（对齐 S8 口径）；profiler 里该 kernel 与 Triton 版本的时间对比；27B decode 端到端**必须更快**且文本一致；**逐 shape ABBA，"no shape regresses"**。

### 1.2 任务书补充门禁

| # | 门禁 | 判据 |
|---|---|---|
| G1 | 承接而非重写 | 先盘点第 1 任遗留物，已有产物不重写（本档 §0） |
| G2 | 环境开关 | 新增 `VLLM_DCU_USE_INT8_SDOT4`（默认开）用于 A/B 与回滚；**不得**用 monkey-patch |
| G3 | 出处 | 新文件头含 `Provenance:` 注释，注明来自 `/home/lyh/server/int8_sdot4/` |
| G4 | 架构纯净 | 新 `_rocm_C.abi3.so` 的 `amdgcn-amd-amdhsa--gfx*` 串**只含 gfx906** |
| G5 | 增量构建 | 增量重建（新 TU ≈1 个 + 链接）成功，canonical 已换新件并留旧件回滚 |
| G6 | 数值 | M=1/2/4/8（GEMV）+ M=32/64/256（GEMM），K/N 取 27B 真实维度，`rel_err ≤ 1e-2` |
| G7 | 性能 | 同 shape 新内核 vs `TritonInt8ScaledMMLinearKernel`，ABBA，"no shape regresses" |
| G8 | 端到端 | 27B W8A8 TP=2 开关开/关 decode ms/token + 输出文本一致性；跑不动则写"未执行 + 原因 + 续跑命令" |
| G9 | 边做边写 | 归档先建骨架，随后追加（避免第 1 任静默死亡导致证据丢失） |

## 2. 实际改动

> 本步新增/修改的源码**全部由第 1 任写下**（本次续跑核对为真，未重写、未改动一行）；本次续跑只**新增归档 + 运行验证**。下表行号为本档核对时的实际行号。

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `csrc/rocm/w8a8_sdot4.cu` | 新增（517 行） | gfx906 `V_DOT4_I32_I8` W8A8 kernel：`w8a8_sdot4_gemv`（M≤8，每输出列一个 block、`nthreads` 默认 64、LDS 树归约）+ `w8a8_sdot4_gemm`（BM×BN 寄存器 tile、BK=32 双缓冲 LDS、`cfg` 选 tile）+ 两个 `*_meta`（fake tensor）；文件头 `Provenance:` 指向 `/home/lyh/server/int8_sdot4/w8a8_sdot4_kernel.hip`（见 §3.7） |
| `csrc/rocm/w8a8_sdot4.h` | 新增（34 行） | 4+2 个函数声明（2 个 kernel + 2 个 meta），并写明布局契约（权重 `[N,K]` int8 不转置、`x_s [M]`、`w_s [N]`、`K % 4 == 0`、epilogue 顺序同 `triton_scaled_mm`） |
| `CMakeLists.txt` | `:1548` | `VLLM_ROCM_EXT_SRC` 加入 `"csrc/rocm/w8a8_sdot4.cu"`（在 `torch_bindings.cpp` 之后、`skinny_gemms.cu` 之前） |
| `csrc/rocm/ops.h` | `:5` | `#include "rocm/w8a8_sdot4.h"` |
| `csrc/rocm/torch_bindings.cpp` | `:101-118` | 在 `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops)` 内 `def`+`impl`（`torch::kCUDA`）`w8a8_sdot4_gemv` / `w8a8_sdot4_gemm`；紧随其后新增 `TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Meta, rocm_meta_ops)` 注册两个 `*_meta` ⇒ op 落在 **`torch.ops._rocm_C`**（不是 `_C` / `_C_stable_libtorch`，符合硬约束 15 的判定：本轮改的是 `csrc/rocm/**`，对应扩展是 `_rocm_C`） |
| `vllm/_custom_ops.py` | `:856-881` | 两个 thin wrapper `w8a8_sdot4_gemv(...)` / `w8a8_sdot4_gemm(...)` |
| `vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py` | 新增（174 行） | `Gfx906Sdot4Int8ScaledMMLinearKernel`：`is_supported()` = ROCm ∧ `on_gfx906()` ∧ `envs.VLLM_DCU_USE_INT8_SDOT4` ∧ op 已注册（`:61-73`）；`can_implement()` 只支持对称/per-token 动态输入 + per-channel 权重（`:75-85`）；`process_weights_after_loading()` 保留 `[N,K]` 原生布局（不转置、不反量化）（`:86-120`）；`apply_weights()` M≤8 走 GEMV、否则 GEMM，不支持时回退 `triton_scaled_mm`（`:135-174`） |
| `vllm/model_executor/kernels/linear/__init__.py` | `:214`、`:421-425` | import 内核类；`_POSSIBLE_INT8_KERNELS[PlatformEnum.ROCM]` 列表**第 0 位**插入 `Gfx906Sdot4Int8ScaledMMLinearKernel`（原 `[Aiter, Triton]` → `[Gfx906Sdot4, Aiter, Triton]`） |
| `vllm/model_executor/kernels/linear/scaled_mm/__init__.py` | `:23-26`、`:66` | import + 写入 `__all__` |
| `vllm/envs.py` | `:168`、`:1412-1414` | `VLLM_DCU_USE_INT8_SDOT4: bool = True` + `os.getenv("VLLM_DCU_USE_INT8_SDOT4","True").lower() in ("true","1")` |
| `tools/dcu/s11e_sdot4_eval.py` | 新增（333 行，12,567 B，md5 `1d431bd9b6caa18e6152cd757d3d3117`） | 第 1 任写的评测台：`--mode int`（int32 精确性）/ `--mode check`（逐 shape rel_err vs fp32 反量化参考）/ `--mode bench`（逐 shape ABBA）；`SHAPES` 取 27B 真实 (N,K)（`:41-50`） |
| `tools/dcu/s11e_sdot4_edge.py` | 新增（本次续跑第 2 任加，205 行） | **补充**鲁棒性探针：`edges`（M=8/9 边界、N=1000/1025 非整 tile、M=2048、`cfg=3/4`、K=5124）与 `fallback`（`K%4!=0`、非 int8 权重必须走 `triton_scaled_mm` 而非抛错）。不改动第 1 任的评测台，独立文件 |
| `docs/result/z100-11e-w8a8-sdot4-result.md` | 本文件 | 归档（第 1 任未创建，本次续跑补齐） |

**未使用 monkey-patch**：接入走的是内核表 `_POSSIBLE_INT8_KERNELS` + `Int8ScaledMMLinearKernel` 抽象，没有运行时打补丁。
## 3. 执行过程与原始证据

### 3.0 本次运行环境

本步（S11e）**所有真机命令都在 DTK 影子容器 `z100-s9`**（`--memory=32g`、`ShmSize=16g`、bridge 网络、挂载 `/mnt/game` 与 `/home/lyh/z100-vllm→/src`）内以 root 执行；GPU 只用 **2 号卡**（数值/性能）与 **2,3 号卡**（端到端）。

- 原因（本任发现的环境事实，第 1 任未记录）：构建容器 `z100-build-211` 的 `torchvision 0.25.0+das...torch2100` 与 `torch 2.11.0+das` **ABI 不匹配**，`import torchvision` 直接抛
  `RuntimeError: operator torchvision::nms does not exist`；而这条链路（`vllm.model_executor.kernels.linear` → `mixed_precision/cutlass` → `input_quant_fp8` → `vllm._aiter_ops` → … → `transformers` → `torchvision`）**只在 `--mode bench` 走到**，`int`/`check` 两种模式不走。
  `z100-s9` 的 `torchvision 0.26.0+cu130` 可正常 import，故 bench 在 `z100-s9` 跑。构建仍在 `z100-build-211`（未变）。

```bash
# 构建/静态检查容器
docker exec z100-build-211 ...
# 真机运行容器（数值 / 性能 / 端到端）
docker exec z100-s9 bash -lc 'cd /mnt/game/port/gfx906-vllm-211; . /mnt/game/port/setenv-211.sh; \
  export HIP_VISIBLE_DEVICES=2; export PYTHONPATH=/mnt/game/triton-gfx906/install:$PYTHONPATH; \
  export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache/s11e; python tools/dcu/s11e_sdot4_eval.py --mode <int|check|bench>'
```

### 3.1 承接盘点（原始输出）

**git 状态**（只列 S11e 相关；工作区另有大量前序步骤的改动，不属本步）：

```
 M CMakeLists.txt                       # 本步新增 csrc/rocm/w8a8_sdot4.cu 到 VLLM_ROCM_EXT_SRC（:1548）
 M csrc/rocm/ops.h                      # :5 include
 M csrc/rocm/torch_bindings.cpp         # +22: def/impl + Meta
 M vllm/_custom_ops.py                  # +36: 两个 wrapper
 M vllm/envs.py                         # +22: VLLM_DCU_USE_INT8_SDOT4（含本步；envs.py 另有前序改动）
 M vllm/model_executor/kernels/linear/__init__.py            # +9
 M vllm/model_executor/kernels/linear/scaled_mm/__init__.py  # +4
?? csrc/rocm/w8a8_sdot4.cu              # 新增 517 行
?? csrc/rocm/w8a8_sdot4.h               # 新增 34 行
?? vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py  # 新增 174 行
```

**构建日志**（第 1 任，`/mnt/game/port/s11e/build-s11e-2.log` 末尾 3 行 = 本步新 TU 编译 + 链接；`build-s11e.log` 是紧随其后的一次 no-op 增量）：

`/mnt/game/port/s11e/build-s11e.log`（**先**跑，23:54：只有 C++ 绑定，2 步）
```
[1/2] Building CXX object CMakeFiles/_rocm_C.dir/csrc/rocm/torch_bindings.cpp.o
[2/2] Linking HIP shared module _rocm_C.abi3.so
```
`/mnt/game/port/s11e/build-s11e-2.log`（**后**跑，00:03：新 TU 落地。证据只取这 3 行，中间 189 行是 hipify 的逐文件列表）
```
[1/4] Running hipify on all extension source files.
[2/3] Building HIP object CMakeFiles/_rocm_C.dir/csrc/rocm/w8a8_sdot4.hip.o
[3/3] Linking HIP shared module _rocm_C.abi3.so
```
⇒ **增量性成立**：整次重建只编了 **1 个 HIP TU**（`w8a8_sdot4.hip`）+ 一次链接，其余目标文件全部复用（与 S4d 的"单 TU ~40 s"口径一致）。

**canonical 副本 vs 回滚件**：

| 件 | 大小 (B) | md5 | mtime |
|---|---|---|---|
| `/mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so`（新，已安装） | 36,182,960 | `ebd6fb8dac572514cf94ca7ec7d828a7` | 2026-10-09 00:03:04 |
| `/mnt/game/port/build-211/temp/_rocm_C.abi3.so`（构建产物） | 36,182,960 | 同尺寸同时间戳 | 2026-10-09 00:03 |
| `/mnt/game/port/s11e/_rocm_C.abi3.so.before`（回滚件） | 35,156,368 | `5d1e50e1a77de74f2d308519f8120745` | 2026-10-07 23:48 |

### 3.2 op 落在哪个扩展 + 架构纯净性（硬约束 15 / G4）

**判定：`w8a8_sdot4_gemv` / `w8a8_sdot4_gemm` 注册在 `torch.ops._rocm_C`**，`_C` 与 `_C_stable_libtorch` 里都**没有**。本步改的是 `csrc/rocm/**` + `CMakeLists.txt` 的 `VLLM_ROCM_EXT_SRC`，对应扩展就是 `_rocm_C.abi3.so`（`_C_stable_libtorch` 只在改 `csrc/libtorch_stable/**` 时才需要重编，硬约束 15）。

```
_rocm_C.w8a8_sdot4_gemv    hasattr=True
_rocm_C.w8a8_sdot4_gemm    hasattr=True
_C imported
_C_stable_libtorch imported
torch.ops._C.w8a8_sdot4_gemm hasattr=False
torch.ops._C_stable_libtorch.w8a8_sdot4_gemm hasattr=False
```

**导出符号**（`nm -D --defined-only _rocm_C.abi3.so | grep sdot4`，反混淆后 = gemm / gemv / gemm_meta / gemv_meta）：

```
00000000001b37b0 T _Z15w8a8_sdot4_gemmN2at6TensorES0_S0_S0_St8optionalIS0_EN3c1010ScalarTypeEl
00000000001b1b40 T _Z15w8a8_sdot4_gemvN2at6TensorES0_S0_S0_St8optionalIS0_EN3c1010ScalarTypeEl
00000000001b5090 T _Z20w8a8_sdot4_gemm_metaN2at6TensorES0_S0_S0_St8optionalIS0_EN3c1010ScalarTypeEl
00000000001b4f40 T _Z20w8a8_sdot4_gemv_metaN2at6TensorES0_S0_S0_St8optionalIS0_EN3c1010ScalarTypeEl
```

**`strings | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sort -u`（canonical 三件全部实测）**：

| .so | 大小 (B) | mtime | md5 | 架构串 |
|---|---|---|---|---|
| `_C.abi3.so` | 78,102,728 | 2026-10-07 23:43:19 | `6fba05966e153b3d04771d6960c782bb` | `amdgcn-amd-amdhsa--gfx906` |
| `_C_stable_libtorch.abi3.so` | 40,691,216 | 2026-10-08 13:25:02 | `93bf7b4a391ac97e2f67cc7141633cb1` | `amdgcn-amd-amdhsa--gfx906` |
| `_rocm_C.abi3.so`（新） | 36,182,960 | 2026-10-09 00:03:04 | `ebd6fb8dac572514cf94ca7ec7d828a7` | `amdgcn-amd-amdhsa--gfx906` |

⇒ **只含 gfx906**，无 gfx9xx 其它目标、无 gfx10/11/12。前两件的 md5 与 S9b 记录一致（未被动过）；`_C_stable_libtorch` 仍为 `93bf7b4a…`（= 硬约束 15 里 S7c 记录的 canonical 值）。

### 3.3 数值验收 A：int32 累加器精确性（`--mode int`）

`sdot4(x_s=w_s=1, out=fp32)` 与 `torch.matmul(xq.cpu().int(), wq.cpu().int().t())` 比：**35/35 组 `bit_equal=True`、`max_abs_diff=0`**（N/K 取 27B 真实维度，M=1/4/8 走 GEMV，M=32/64 走 GEMM）。RC=0，elapsed 10.3s。

```
  N= 12288 K=  5120 M=   1 bit_equal=True max_abs_diff=0
  N= 12288 K=  5120 M=   4 bit_equal=True max_abs_diff=0
  N= 12288 K=  5120 M=   8 bit_equal=True max_abs_diff=0
  N= 12288 K=  5120 M=  32 bit_equal=True max_abs_diff=0
  N= 12288 K=  5120 M=  64 bit_equal=True max_abs_diff=0
  ...（7 个 (N,K) × 5 个 M 共 35 行，全部同上）
elapsed 10.3s
INT_RC=0
```
逐行摘要（完整 `INT_SUMMARY` JSON 见 `/mnt/game/port/s11e/int.log`）：

| N | K | M | bit_equal | max_abs_diff |
|---|---|---|---|---|
| 12288 | 5120 | 1 | True | 0 |
| 12288 | 5120 | 4 | True | 0 |
| 12288 | 5120 | 8 | True | 0 |
| 12288 | 5120 | 32 | True | 0 |
| 12288 | 5120 | 64 | True | 0 |
| 1024 | 5120 | 1 | True | 0 |
| 1024 | 5120 | 4 | True | 0 |
| 1024 | 5120 | 8 | True | 0 |
| 1024 | 5120 | 32 | True | 0 |
| 1024 | 5120 | 64 | True | 0 |
| 5120 | 6144 | 1 | True | 0 |
| 5120 | 6144 | 4 | True | 0 |
| 5120 | 6144 | 8 | True | 0 |
| 5120 | 6144 | 32 | True | 0 |
| 5120 | 6144 | 64 | True | 0 |
| 10240 | 5120 | 1 | True | 0 |
| 10240 | 5120 | 4 | True | 0 |
| 10240 | 5120 | 8 | True | 0 |
| 10240 | 5120 | 32 | True | 0 |
| 10240 | 5120 | 64 | True | 0 |
| 6144 | 5120 | 1 | True | 0 |
| 6144 | 5120 | 4 | True | 0 |
| 6144 | 5120 | 8 | True | 0 |
| 6144 | 5120 | 32 | True | 0 |
| 6144 | 5120 | 64 | True | 0 |
| 17408 | 5120 | 1 | True | 0 |
| 17408 | 5120 | 4 | True | 0 |
| 17408 | 5120 | 8 | True | 0 |
| 17408 | 5120 | 32 | True | 0 |
| 17408 | 5120 | 64 | True | 0 |
| 5120 | 17408 | 1 | True | 0 |
| 5120 | 17408 | 4 | True | 0 |
| 5120 | 17408 | 8 | True | 0 |
| 5120 | 17408 | 32 | True | 0 |
| 5120 | 17408 | 64 | True | 0 |

### 3.4 数值验收 B：逐 shape vs fp32 反量化参考（`--mode check`，门禁 G6）

口径 = S8：`ref = (xq.float()*x_s) @ (wq.float()*w_s)^T + bias`（对称量化 zp=0），`rel_err = ‖out-ref‖/‖ref‖`。输出 dtype fp16（生产口径）。
**63/63 组通过**：worst sdot4 `rel_err = 2.988e-04`（门禁 ≤ 1e-2，**余量 33×**）；`sdot4_finite` 全 True；**`sdot4_agrees_with_triton` 全 True —— 新内核与 `triton_scaled_mm` 的输出逐位相同**（因此 worst triton rel_err 同为 2.988e-04）。RC=0，elapsed 5.6s。

| N | K | M | 层 | sdot4 rel_err | triton rel_err | sdot4 有限 | 与 Triton 逐位相同 |
|---|---|---|---|---|---|---|---|
| 12288 | 5120 | 1 | self_attn.q_proj | 2.920e-04 | 2.920e-04 | True | True |
| 12288 | 5120 | 2 | self_attn.q_proj | 2.906e-04 | 2.906e-04 | True | True |
| 12288 | 5120 | 4 | self_attn.q_proj | 2.925e-04 | 2.925e-04 | True | True |
| 12288 | 5120 | 8 | self_attn.q_proj | 2.906e-04 | 2.906e-04 | True | True |
| 12288 | 5120 | 16 | self_attn.q_proj | 2.875e-04 | 2.875e-04 | True | True |
| 12288 | 5120 | 32 | self_attn.q_proj | 2.898e-04 | 2.898e-04 | True | True |
| 12288 | 5120 | 64 | self_attn.q_proj | 2.911e-04 | 2.911e-04 | True | True |
| 12288 | 5120 | 256 | self_attn.q_proj | 2.903e-04 | 2.903e-04 | True | True |
| 12288 | 5120 | 512 | self_attn.q_proj | 2.898e-04 | 2.898e-04 | True | True |
| 1024 | 5120 | 1 | self_attn.k_proj/v_proj | 2.975e-04 | 2.975e-04 | True | True |
| 1024 | 5120 | 2 | self_attn.k_proj/v_proj | 2.943e-04 | 2.943e-04 | True | True |
| 1024 | 5120 | 4 | self_attn.k_proj/v_proj | 2.988e-04 | 2.988e-04 | True | True |
| 1024 | 5120 | 8 | self_attn.k_proj/v_proj | 2.901e-04 | 2.901e-04 | True | True |
| 1024 | 5120 | 16 | self_attn.k_proj/v_proj | 2.920e-04 | 2.920e-04 | True | True |
| 1024 | 5120 | 32 | self_attn.k_proj/v_proj | 2.875e-04 | 2.875e-04 | True | True |
| 1024 | 5120 | 64 | self_attn.k_proj/v_proj | 2.920e-04 | 2.920e-04 | True | True |
| 1024 | 5120 | 256 | self_attn.k_proj/v_proj | 2.886e-04 | 2.886e-04 | True | True |
| 1024 | 5120 | 512 | self_attn.k_proj/v_proj | 2.921e-04 | 2.921e-04 | True | True |
| 5120 | 6144 | 1 | self_attn.o_proj | 2.883e-04 | 2.883e-04 | True | True |
| 5120 | 6144 | 2 | self_attn.o_proj | 2.868e-04 | 2.868e-04 | True | True |
| 5120 | 6144 | 4 | self_attn.o_proj | 2.892e-04 | 2.892e-04 | True | True |
| 5120 | 6144 | 8 | self_attn.o_proj | 2.887e-04 | 2.887e-04 | True | True |
| 5120 | 6144 | 16 | self_attn.o_proj | 2.868e-04 | 2.868e-04 | True | True |
| 5120 | 6144 | 32 | self_attn.o_proj | 2.889e-04 | 2.889e-04 | True | True |
| 5120 | 6144 | 64 | self_attn.o_proj | 2.899e-04 | 2.899e-04 | True | True |
| 5120 | 6144 | 256 | self_attn.o_proj | 2.892e-04 | 2.892e-04 | True | True |
| 5120 | 6144 | 512 | self_attn.o_proj | 2.886e-04 | 2.886e-04 | True | True |
| 10240 | 5120 | 1 | linear_attn.in_proj_qkv | 2.871e-04 | 2.871e-04 | True | True |
| 10240 | 5120 | 2 | linear_attn.in_proj_qkv | 2.949e-04 | 2.949e-04 | True | True |
| 10240 | 5120 | 4 | linear_attn.in_proj_qkv | 2.918e-04 | 2.918e-04 | True | True |
| 10240 | 5120 | 8 | linear_attn.in_proj_qkv | 2.898e-04 | 2.898e-04 | True | True |
| 10240 | 5120 | 16 | linear_attn.in_proj_qkv | 2.919e-04 | 2.919e-04 | True | True |
| 10240 | 5120 | 32 | linear_attn.in_proj_qkv | 2.914e-04 | 2.914e-04 | True | True |
| 10240 | 5120 | 64 | linear_attn.in_proj_qkv | 2.902e-04 | 2.902e-04 | True | True |
| 10240 | 5120 | 256 | linear_attn.in_proj_qkv | 2.892e-04 | 2.892e-04 | True | True |
| 10240 | 5120 | 512 | linear_attn.in_proj_qkv | 2.906e-04 | 2.906e-04 | True | True |
| 6144 | 5120 | 1 | linear_attn.in_proj_z | 2.868e-04 | 2.868e-04 | True | True |
| 6144 | 5120 | 2 | linear_attn.in_proj_z | 2.886e-04 | 2.886e-04 | True | True |
| 6144 | 5120 | 4 | linear_attn.in_proj_z | 2.907e-04 | 2.907e-04 | True | True |
| 6144 | 5120 | 8 | linear_attn.in_proj_z | 2.912e-04 | 2.912e-04 | True | True |
| 6144 | 5120 | 16 | linear_attn.in_proj_z | 2.890e-04 | 2.890e-04 | True | True |
| 6144 | 5120 | 32 | linear_attn.in_proj_z | 2.904e-04 | 2.904e-04 | True | True |
| 6144 | 5120 | 64 | linear_attn.in_proj_z | 2.909e-04 | 2.909e-04 | True | True |
| 6144 | 5120 | 256 | linear_attn.in_proj_z | 2.900e-04 | 2.900e-04 | True | True |
| 6144 | 5120 | 512 | linear_attn.in_proj_z | 2.896e-04 | 2.896e-04 | True | True |
| 17408 | 5120 | 1 | mlp.gate_proj/up_proj | 2.890e-04 | 2.890e-04 | True | True |
| 17408 | 5120 | 2 | mlp.gate_proj/up_proj | 2.891e-04 | 2.891e-04 | True | True |
| 17408 | 5120 | 4 | mlp.gate_proj/up_proj | 2.901e-04 | 2.901e-04 | True | True |
| 17408 | 5120 | 8 | mlp.gate_proj/up_proj | 2.906e-04 | 2.906e-04 | True | True |
| 17408 | 5120 | 16 | mlp.gate_proj/up_proj | 2.891e-04 | 2.891e-04 | True | True |
| 17408 | 5120 | 32 | mlp.gate_proj/up_proj | 2.901e-04 | 2.901e-04 | True | True |
| 17408 | 5120 | 64 | mlp.gate_proj/up_proj | 2.902e-04 | 2.902e-04 | True | True |
| 17408 | 5120 | 256 | mlp.gate_proj/up_proj | 2.901e-04 | 2.901e-04 | True | True |
| 17408 | 5120 | 512 | mlp.gate_proj/up_proj | 2.905e-04 | 2.905e-04 | True | True |
| 5120 | 17408 | 1 | mlp.down_proj | 2.715e-04 | 2.715e-04 | True | True |
| 5120 | 17408 | 2 | mlp.down_proj | 2.834e-04 | 2.834e-04 | True | True |
| 5120 | 17408 | 4 | mlp.down_proj | 2.807e-04 | 2.807e-04 | True | True |
| 5120 | 17408 | 8 | mlp.down_proj | 2.751e-04 | 2.751e-04 | True | True |
| 5120 | 17408 | 16 | mlp.down_proj | 2.740e-04 | 2.740e-04 | True | True |
| 5120 | 17408 | 32 | mlp.down_proj | 2.746e-04 | 2.746e-04 | True | True |
| 5120 | 17408 | 64 | mlp.down_proj | 2.744e-04 | 2.744e-04 | True | True |
| 5120 | 17408 | 256 | mlp.down_proj | 2.754e-04 | 2.754e-04 | True | True |
| 5120 | 17408 | 512 | mlp.down_proj | 2.748e-04 | 2.748e-04 | True | True |

> 逐位相同这件事有两层意义：(1) 数值门禁不仅"过"，而是**与现役实现等价**，端到端文本一致性有了强预期；(2) 差异不可能来自 epilogue 顺序（`(acc*x_s)*w_s` 后 cast，再 `+bias`，与 `triton_scaled_mm` 一致）。

### 3.5 性能验收：逐 shape ABBA（`--mode bench`，门禁 G7）

**本步复跑**（第 2 任，独立一次）：`63` 组 = 7 个 27B 真实 (N,K) × 9 个 M（1/2/4/8 = GEMV；16/32/64/256/512 = GEMM）。ABBA = `A B B A`，每点 5 轮 × 50 iter 取 min；`full` = 走完整 `apply_weights`（含 `scaled_int8_quant`）。elapsed 92.0s，RC=0。日志 `/mnt/game/port/s11e/bench-rerun.log`。

| N | K | M | 层 | sdot4 kernel µs | triton kernel µs | kernel × | sdot4 full µs | triton full µs | full × | sdot4 TF | triton TF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12288 | 5120 | 1 | self_attn.q_proj | 107.94 | 1030.03 | 9.54 | 115.44 | 930.02 | 8.06 | 1.17 | 0.12 |
| 12288 | 5120 | 2 | self_attn.q_proj | 163.01 | 1043.40 | 6.40 | 174.93 | 945.88 | 5.41 | 1.54 | 0.24 |
| 12288 | 5120 | 4 | self_attn.q_proj | 278.43 | 1045.96 | 3.76 | 284.70 | 945.30 | 3.32 | 1.81 | 0.48 |
| 12288 | 5120 | 8 | self_attn.q_proj | 515.80 | 1046.45 | 2.03 | 512.91 | 945.71 | 1.84 | 1.95 | 0.96 |
| 12288 | 5120 | 16 | self_attn.q_proj | 357.14 | 1035.88 | 2.90 | 404.21 | 942.83 | 2.33 | 5.64 | 1.94 |
| 12288 | 5120 | 32 | self_attn.q_proj | 337.95 | 1037.40 | 3.07 | 446.75 | 945.42 | 2.12 | 11.91 | 3.88 |
| 12288 | 5120 | 64 | self_attn.q_proj | 503.51 | 951.97 | 1.89 | 583.11 | 1429.62 | 2.45 | 15.99 | 8.46 |
| 12288 | 5120 | 256 | self_attn.q_proj | 1614.87 | 2119.27 | 1.31 | 1629.99 | 2369.56 | 1.45 | 19.95 | 15.20 |
| 12288 | 5120 | 512 | self_attn.q_proj | 2960.33 | 3730.60 | 1.26 | 3022.54 | 4223.76 | 1.40 | 21.76 | 17.27 |
| 1024 | 5120 | 1 | self_attn.k_proj/v_proj | 17.44 | 394.35 | 22.61 | 59.73 | 413.51 | 6.92 | 0.60 | 0.03 |
| 1024 | 5120 | 2 | self_attn.k_proj/v_proj | 27.68 | 385.94 | 13.94 | 59.95 | 423.36 | 7.06 | 0.76 | 0.05 |
| 1024 | 5120 | 4 | self_attn.k_proj/v_proj | 47.11 | 386.38 | 8.20 | 60.08 | 423.59 | 7.05 | 0.89 | 0.11 |
| 1024 | 5120 | 8 | self_attn.k_proj/v_proj | 84.19 | 387.17 | 4.60 | 75.02 | 424.45 | 5.66 | 1.00 | 0.22 |
| 1024 | 5120 | 16 | self_attn.k_proj/v_proj | 190.22 | 393.32 | 2.07 | 197.65 | 423.79 | 2.14 | 0.88 | 0.43 |
| 1024 | 5120 | 32 | self_attn.k_proj/v_proj | 198.04 | 395.59 | 2.00 | 204.38 | 424.52 | 2.08 | 1.69 | 0.85 |
| 1024 | 5120 | 64 | self_attn.k_proj/v_proj | 204.74 | 400.92 | 1.96 | 212.88 | 431.77 | 2.03 | 3.28 | 1.67 |
| 1024 | 5120 | 256 | self_attn.k_proj/v_proj | 307.05 | 883.83 | 2.88 | 320.96 | 911.56 | 2.84 | 8.74 | 3.04 |
| 1024 | 5120 | 512 | self_attn.k_proj/v_proj | 437.82 | 891.73 | 2.04 | 501.02 | 962.75 | 1.92 | 12.26 | 6.02 |
| 5120 | 6144 | 1 | self_attn.o_proj | 62.36 | 559.87 | 8.98 | 68.17 | 901.74 | 13.23 | 1.01 | 0.11 |
| 5120 | 6144 | 2 | self_attn.o_proj | 93.73 | 566.94 | 6.05 | 108.26 | 905.23 | 8.36 | 1.34 | 0.22 |
| 5120 | 6144 | 4 | self_attn.o_proj | 176.66 | 567.62 | 3.21 | 169.51 | 904.85 | 5.34 | 1.42 | 0.44 |
| 5120 | 6144 | 8 | self_attn.o_proj | 344.93 | 567.84 | 1.65 | 327.61 | 905.02 | 2.76 | 1.46 | 0.89 |
| 5120 | 6144 | 16 | self_attn.o_proj | 292.53 | 574.80 | 1.96 | 358.70 | 909.86 | 2.54 | 3.44 | 1.75 |
| 5120 | 6144 | 32 | self_attn.o_proj | 294.96 | 571.12 | 1.94 | 382.12 | 909.23 | 2.38 | 6.83 | 3.53 |
| 5120 | 6144 | 64 | self_attn.o_proj | 377.58 | 578.77 | 1.53 | 462.95 | 921.32 | 1.99 | 10.66 | 6.96 |
| 5120 | 6144 | 256 | self_attn.o_proj | 1260.94 | 1292.39 | 1.02 | 1277.04 | 1465.59 | 1.15 | 12.77 | 12.46 |
| 5120 | 6144 | 512 | self_attn.o_proj | 1882.43 | 2132.65 | 1.13 | 1941.14 | 2502.19 | 1.29 | 17.11 | 15.10 |
| 10240 | 5120 | 1 | linear_attn.in_proj_qkv | 126.74 | 967.60 | 7.63 | 136.30 | 915.97 | 6.72 | 0.83 | 0.11 |
| 10240 | 5120 | 2 | linear_attn.in_proj_qkv | 168.54 | 987.57 | 5.86 | 176.69 | 929.51 | 5.26 | 1.24 | 0.21 |
| 10240 | 5120 | 4 | linear_attn.in_proj_qkv | 250.36 | 989.07 | 3.95 | 258.04 | 931.14 | 3.61 | 1.68 | 0.42 |
| 10240 | 5120 | 8 | linear_attn.in_proj_qkv | 458.12 | 988.90 | 2.16 | 451.24 | 930.61 | 2.06 | 1.83 | 0.85 |
| 10240 | 5120 | 16 | linear_attn.in_proj_qkv | 310.13 | 979.77 | 3.16 | 323.41 | 931.21 | 2.88 | 5.41 | 1.71 |
| 10240 | 5120 | 32 | linear_attn.in_proj_qkv | 357.39 | 984.24 | 2.75 | 379.47 | 933.77 | 2.46 | 9.39 | 3.41 |
| 10240 | 5120 | 64 | linear_attn.in_proj_qkv | 452.18 | 819.92 | 1.81 | 475.23 | 1107.27 | 2.33 | 14.84 | 8.18 |
| 10240 | 5120 | 256 | linear_attn.in_proj_qkv | 1385.58 | 1800.49 | 1.30 | 1401.76 | 2002.12 | 1.43 | 19.37 | 14.91 |
| 10240 | 5120 | 512 | linear_attn.in_proj_qkv | 2506.13 | 3124.43 | 1.25 | 2567.76 | 3562.61 | 1.39 | 21.42 | 17.18 |
| 6144 | 5120 | 1 | linear_attn.in_proj_z | 96.11 | 473.80 | 4.93 | 107.37 | 746.90 | 6.96 | 0.65 | 0.13 |
| 6144 | 5120 | 2 | linear_attn.in_proj_z | 119.80 | 484.40 | 4.04 | 102.46 | 753.11 | 7.35 | 1.05 | 0.26 |
| 6144 | 5120 | 4 | linear_attn.in_proj_z | 164.88 | 484.87 | 2.94 | 163.86 | 752.30 | 4.59 | 1.53 | 0.52 |
| 6144 | 5120 | 8 | linear_attn.in_proj_z | 298.75 | 485.41 | 1.62 | 304.46 | 752.56 | 2.47 | 1.68 | 1.04 |
| 6144 | 5120 | 16 | linear_attn.in_proj_z | 234.25 | 489.90 | 2.09 | 280.91 | 754.19 | 2.68 | 4.30 | 2.05 |
| 6144 | 5120 | 32 | linear_attn.in_proj_z | 268.42 | 493.21 | 1.84 | 294.96 | 753.27 | 2.55 | 7.50 | 4.08 |
| 6144 | 5120 | 64 | linear_attn.in_proj_z | 331.49 | 489.26 | 1.48 | 331.97 | 755.63 | 2.28 | 12.15 | 8.23 |
| 6144 | 5120 | 256 | linear_attn.in_proj_z | 939.75 | 1084.34 | 1.15 | 958.63 | 1210.91 | 1.26 | 17.14 | 14.85 |
| 6144 | 5120 | 512 | linear_attn.in_proj_z | 1609.09 | 2113.22 | 1.31 | 1670.05 | 2407.77 | 1.44 | 20.02 | 15.24 |
| 17408 | 5120 | 1 | mlp.gate_proj/up_proj | 141.12 | 1484.43 | 10.52 | 150.61 | 1357.71 | 9.01 | 1.26 | 0.12 |
| 17408 | 5120 | 2 | mlp.gate_proj/up_proj | 221.12 | 1509.60 | 6.83 | 225.48 | 1379.51 | 6.12 | 1.61 | 0.24 |
| 17408 | 5120 | 4 | mlp.gate_proj/up_proj | 387.20 | 1511.83 | 3.90 | 388.05 | 1383.22 | 3.56 | 1.84 | 0.47 |
| 17408 | 5120 | 8 | mlp.gate_proj/up_proj | 723.21 | 1513.61 | 2.09 | 710.21 | 1382.90 | 1.95 | 1.97 | 0.94 |
| 17408 | 5120 | 16 | mlp.gate_proj/up_proj | 422.46 | 1499.46 | 3.55 | 448.19 | 1387.08 | 3.09 | 6.75 | 1.90 |
| 17408 | 5120 | 32 | mlp.gate_proj/up_proj | 508.10 | 1502.75 | 2.96 | 509.35 | 1391.19 | 2.73 | 11.23 | 3.80 |
| 17408 | 5120 | 64 | mlp.gate_proj/up_proj | 793.43 | 1244.27 | 1.57 | 637.72 | 1772.65 | 2.78 | 14.38 | 9.17 |
| 17408 | 5120 | 256 | mlp.gate_proj/up_proj | 2171.95 | 2728.21 | 1.26 | 2187.66 | 3081.87 | 1.41 | 21.01 | 16.73 |
| 17408 | 5120 | 512 | mlp.gate_proj/up_proj | 4111.44 | 5074.66 | 1.23 | 4180.10 | 5811.48 | 1.39 | 22.20 | 17.99 |
| 5120 | 17408 | 1 | mlp.down_proj | 142.97 | 1560.59 | 10.92 | 156.07 | 2460.03 | 15.76 | 1.25 | 0.11 |
| 5120 | 17408 | 2 | mlp.down_proj | 223.64 | 1557.23 | 6.96 | 234.09 | 2472.76 | 10.56 | 1.59 | 0.23 |
| 5120 | 17408 | 4 | mlp.down_proj | 422.49 | 1557.00 | 3.69 | 424.30 | 2475.16 | 5.83 | 1.69 | 0.46 |
| 5120 | 17408 | 8 | mlp.down_proj | 809.96 | 1557.74 | 1.92 | 809.55 | 2478.82 | 3.06 | 1.76 | 0.92 |
| 5120 | 17408 | 16 | mlp.down_proj | 677.74 | 1559.39 | 2.30 | 905.06 | 2484.97 | 2.75 | 4.21 | 1.83 |
| 5120 | 17408 | 32 | mlp.down_proj | 688.19 | 1561.38 | 2.27 | 917.50 | 2488.84 | 2.71 | 8.29 | 3.65 |
| 5120 | 17408 | 64 | mlp.down_proj | 1014.37 | 1567.48 | 1.55 | 1028.76 | 2493.54 | 2.42 | 11.25 | 7.28 |
| 5120 | 17408 | 256 | mlp.down_proj | 2510.11 | 3486.87 | 1.39 | 2544.83 | 3991.64 | 1.57 | 18.18 | 13.09 |
| 5120 | 17408 | 512 | mlp.down_proj | 4318.32 | 5777.89 | 1.34 | 4387.53 | 6617.30 | 1.51 | 21.14 | 15.80 |


**"no shape regresses" 判定：63/63 组 `kernel ×` ≥ 1 且 `full ×` ≥ 1；`n(kern<1)=0`、`n(full<1)=0`。**
最差两个（仍为正收益，无"变慢的 shape"）：

| 指标 | 最差 shape | 值 |
|---|---|---|
| kernel ×（最差） | N=5120 K=6144 M=256（self_attn.o_proj） | **×1.025**（1260.9 vs 1292.4 µs） |
| full ×（最差） | N=5120 K=6144 M=256（self_attn.o_proj） | **×1.148**（1277.0 vs 1465.6 µs） |

按 M 汇总（最小加速比）：M=1 `×4.93`、M=2 `×4.04`、M=4 `×2.94`、M=8 `×1.62`、M=32 `×1.84`、M=64 `×1.48`、M=256 `×1.02`；最大 kernel × = **22.61**（N=1024 K=5120 M=1，decode 的 k/v_proj）。峰值算力 **22.2 TFLOPS**（M=512，N=17408 K=5120）vs Triton **18.0**。

**与第 1 任 `bench.log` 的复现性对照**（两次独立运行，同一 canonical `.so`）：

| 运行 | min kernel × | min full × | n(kern<1) | n(full<1) | max kernel × | 中位 kernel × |
|---|---|---|---|---|---|---|
| 第 1 任 `bench.log`（00:08） | 1.022 | 1.150 | 0 | 0 | 22.55 | 2.16 |
| 本步 `bench-rerun.log` | 1.025 | 1.148 | 0 | 0 | 22.61 | 2.16 |

逐组比对：**加速比偏差中位 0.2%、最大 2.6%**（ABBA 取 min 的抖动）⇒ 两次运行在同一结论上，`no shape regresses` 由两次独立测量共同支撑。

### 3.6 端到端（27B W8A8 INT8，TP=2，卡 2,3）

命令骨架（容器 `z100-s9`，卡 2,3，`VLLM_WORKER_MULTIPROC_METHOD=spawn`，`VLLM_CACHE_ROOT=/mnt/game/vllm-cache-s11e`）：

```bash
VLLM_DCU_USE_INT8_SDOT4=<1|0> python -c "import torch;from vllm.entrypoints.cli.main import main;main()" serve \
  /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 \
  --dtype float16 --tensor-parallel-size 2 --gpu-memory-utilization 0.9 \
  --disable-custom-all-reduce --skip-mm-profiling --enable-prefix-caching --mamba-cache-mode align \
  --max-model-len <8192|4096> --max-num-batched-tokens 2048 [ -cc.cudagraph_mode=PIECEWISE ] \
  --served-model-name qwen3.8-27b --host 127.0.0.1 --port <8011|8012>
```

#### 3.6.1 尝试 1（本任自行加的 `-cc.cudagraph_mode=PIECEWISE` + `--max-model-len 8192`）：**torch.compile 期崩**，崩点与本内核无关

`/mnt/game/port/s11e/27b-sdot4.log`（08:19:52 起，08:22:44 崩，日志 `/mnt/game/port/s11e/e2e.log`）：

```
--- sdot4 selected kernels ---
      1 Selected Gfx906Sdot4Int8ScaledMMLinearKernel     <-- 真实模型里确实选中了本内核
RuntimeError: Worker failed with error 'Constraints violated (L['inputs_embeds'].size()[0], L['positions'].size()[1])!
  You marked L['positions'].size()[1] as dynamic but your code specialized it to be a constant (2048).
  ... run with TORCH_LOGS="+dynamic"'
User stack:
  vllm/model_executor/models/qwen3_next.py:1685 forward -> qwen3_5.py:875 -> qwen3_next.py:1584
  -> qwen3_next.py:1462 forward -> qwen3_next.py:1452 _project_qkv_gate -> self.rotary_emb(positions, q, k)
  -> vllm/model_executor/layers/rotary_embedding/mrope.py:426 forward_native -> query.view(num_tokens, -1, self.head_size)
```

**贡献**：真实模型日志里出现 `Selected Gfx906Sdot4Int8ScaledMMLinearKernel` ⇒ 内核表第 0 位在生产路径上生效（这条证据独立成立，与端到端成败无关）。
**判定（待 §3.6.3 归因实验补全）**：崩点在 `mrope` 的动态 shape 守卫（`query.view(num_tokens, ...)`），不在 int8 linear 上；已归档的可用配方是 **`--max-model-len 4096` + `PIECEWISE`**（S9）与 **`--max-model-len 8192` + 默认 `FULL_AND_PIECEWISE`**（S9b），本任多加的 `-cc.cudagraph_mode=PIECEWISE` 与 8192 组合是**工程上未验证过**的组合。是否与"本步新增 op"有关，用 §3.6.3 的对照（同 flag、`VLLM_DCU_USE_INT8_SDOT4=0`）判定。

#### 3.6.2 尝试 2：控制组先跑（S9b 配方，默认 cudagraph 模式 @8192），通过后再跑处理组

#### 3.6.2 尝试 2：同一命令下做开关 A/B（控制组先跑）

两组命令**完全相同**（容器 `z100-s9`；卡 2,3；`--max-model-len 8192 --max-num-batched-tokens 2048`；cudagraph 模式**默认**（= S9b 已验证配方）；`VLLM_CACHE_ROOT=/mnt/game/vllm-cache-s11e`），唯一差异是 `VLLM_DCU_USE_INT8_SDOT4`：

| 组 | SW | 结果 | ready 用时 | 真实模型选中的 int8 内核 | 可用 KV | decode ms/token（zh / en，129 token greedy） |
|---|---|---|---|---|---|---|
| 控制组 | **0** | ✅ 起服务并完成 2×129 token | **2460 s（41 min）** | `1 Selected TritonInt8ScaledMMLinearKernel` | 9.9 GiB | **378.84 / 378.95** |
| 处理组 | **1** | ⛔ 启动失败（t=150 s） | — | `1 Selected Gfx906Sdot4Int8ScaledMMLinearKernel` | — | 无输出 |

原始输出（`/mnt/game/port/s11e/{e2e.log,27b-ctl-triton-8192.log,27b-sdot4-8192.log}`）：

```
ARM ctl-triton-8192 READY after 2460s (2026-10-09T09:04:42+08:00)
--- ctl-triton-8192 selected int8 kernels ---
      1 Selected TritonInt8ScaledMMLinearKernel
INFO [monitor.py:53] torch.compile took 196.95 s in total
INFO [model_runner.py:1060] Graph capturing finished in 733 secs, took 1.57 GiB
INFO [gpu_worker.py:719] Available KV cache memory: 9.9 GiB
{"label": "chat_zh", "tokens": 129, "decode_ms_per_token": 378.84145576572337,
 "text_sha256": "15676928b7a104dfc9f8dae4e7e98761663d2505636603a2b67acf99a1d30d2d"}
{"label": "chat_en", "tokens": 129, "decode_ms_per_token": 378.9464360079364,
 "text_sha256": "5c555d4400dc5c56a717e3cf8332ac0267420514da6d7f0864be1020a79626e5"}
===== ARM ctl-triton-8192 stop 2026-10-09T09:06:32+08:00 =====
===== ARM sdot4-8192 start 2026-10-09T09:07:12+08:00 SW=1 port=8012 =====
!! ARM sdot4-8192 exited early (t=150s)
--- sdot4-8192 selected int8 kernels ---
      1 Selected Gfx906Sdot4Int8ScaledMMLinearKernel
Worker_TP0 ERROR torch.fx.experimental.symbolic_shapes.ConstraintViolationError:
  Constraints violated (L['inputs_embeds'].size()[0], L['positions'].size()[1])!
APIServer RuntimeError: Worker failed with error 'Constraints violated (...)'
RESULT ladder1 ctl=1 sdot4=0
```

- **基线复现性**：控制组 378.84 / 378.95 ms/token 与 S9b 归档的 PIECEWISE 380.62、S9 的 380.70 相差 **≤0.5%** ⇒ 本步口径与历史基线一致，数字可比。
- **开关对内核选择的实际作用（两条独立证据）**：`=0` → `Selected TritonInt8ScaledMMLinearKernel`；`=1` → `Selected Gfx906Sdot4Int8ScaledMMLinearKernel`（§3.6.1 也是这条）。即"内核表第 0 位 + 环境开关"在选择层面完全按设计工作。
- **A/B 归因**：同一容器/命令/缓存/卡，唯一变量是开关；`=0` 完整跑通、`=1` 在 torch.compile 阶段 150 s 崩；且**尝试 1**（`-cc.cudagraph_mode=PIECEWISE` + 8192）也是 `=1` 崩、崩点字面相同 ⇒ **崩因是"打开 sdot4 后 torch.compile 路径"，与 cudagraph 模式无关**。
- ⚠️ 处理组**没有**产出 decode ms/token 与文本 ⇒ 端到端门禁不通过（§4）。

#### 3.6.3 崩因定位（本步新增；三个对照实验，含两个负结果）

崩点固定出现在 `mrope.py:426 forward_native` 的 `query.view(num_tokens, -1, self.head_size)`，报"`L['positions'].size()[1]` 被标记为 dynamic 却特化成常量 **2048**"（2048 = `--max-num-batched-tokens`，即 profile 轮的 token 数）。

为定位，保持**同一条命令/同一缓存/同一对卡**，只替换 `Gfx906Sdot4Int8ScaledMMLinearKernel.apply_weights`（`/tmp/s11e_patch_*.py`，诊断用、未进仓库），每轮 150–300 s：

| 实验 | 变量 | `Constraints violated` | `torch.compile took` | 是否进入 CUDA graph capture | 结论 |
|---|---|---|---|---|---|
| 对照 A | 原样（SW=1，默认 cudagraph 模式） | **5** | — | ✗ | 崩（基线） |
| 实验 B | 去掉 Python 的 M 分支（`if m <= 8` 去掉，**始终 GEMM op**） | **5** | — | ✗ | **负结果：Python 的符号维分支不是原因** |
| 实验 C | 同一 kernel 类、同一 `process_weights_after_loading`（[N,K] 布局）、**把自定义 op 换成 `self._triton_fallback`** | **0** | **173.05 s ✅** | **✅ 1（已进入 capture）** | **触发者 = 自定义 op 本体** |

每一轮都在日志里确认补丁生效于**全部 3 个进程**（API server + 2 个 spawn worker 各打一行 `[diag] PATCHED ...`，如 `27b-sdot4-gemmonly2.log:5,36,37`），所以实验 B 的负结果是可信的，不是"补丁没生效"。

- **已排除**：cudagraph 模式（§3.6.1 vs §3.6.2 都崩）、Python 侧按 M 分派（实验 B）、kernel 类插件的其余部分（实验 C 里 `process_weights_after_loading` / `scaled_int8_quant` / `reshape` / `view` 全部保留，编译通过）。
- **定位到**：`torch.ops._rocm_C.w8a8_sdot4_gemm`（`w8a8_sdot4_gemv` 同理，decode 也会走到）与 Dynamo/Inductor 动态 shape 守卫的交互。两个 `*_meta` 实现本身是干净的（`csrc/rocm/w8a8_sdot4.cu:505-517`，只有 `at::empty({xq.size(0), w.size(0)}, xq.options().dtype(out_dtype))`，没有对符号维做比较/分支），所以问题在**用 C++ `TORCH_LIBRARY_IMPL_EXPAND(..., Meta, ...)` 注册的自定义 op 在 vLLM 编译管线里缺少形状语义**这一层，而不在 kernel 数学、不在 Python 包装。
- **已验证不可用的绕法（负结果）**：`-cc.dynamic_shapes_config.type=unbacked` 失败**更早**，且崩在与本内核无关的既有问题上：
  `torch._dynamo.exc.UserError: Could not guard on data-dependent expression Eq(u0, 0)`，
  `Caused by: if x.shape[0] == 0:  # vllm/model_executor/layers/quantization/gguf.py:1363`（`/mnt/game/port/s11e/27b-sdot4-unbacked.log`）。
- **给后续的建议修法**（按代价排序，均未在本步实施）：
  1. 把 op 改成 Python 侧 `torch.library.custom_op` + `register_fake`（或 `torch.library.impl` 走 `register_fake` 而非 C++ Meta），并声明输出形状语义；这是最小改动、最可能直接解决。
  2. 把 `_rocm_C::w8a8_sdot4_gemm/gemv` 加进 `compilation_config.splitting_ops`（用一次 graph break 换取动态 shape 安全），实测其性能代价。
  3. 用 `TORCH_LOGS="+dynamic"` 跑一次短复现（崩点仅 150 s），看 op 额外引入了哪条特化。

#### 3.6.4 未执行项

- 处理组的 decode ms/token 与文本一致性：**未取得**（起服务即失败，见上）。
- `--max-model-len 4096 + PIECEWISE`（S9 配方）下的 sdot4 A/B：**未执行**（成本 ~40 min/组；已由实验 C 给出更根本的结论，故不再重复）。

### 3.7.1 补充鲁棒性探针（本步新增 `tools/dcu/s11e_sdot4_edge.py`）

真实模型只用到 7 个 (N,K)（N/K 都是 128 的整数倍、K 都是 4 的倍数）。探针覆盖内核**声称支持但模型不覆盖**的边界（`HIP_VISIBLE_DEVICES=2`，日志 `/mnt/game/port/s11e/edge.log`，RC=0）：

| 用例 | (N,K,M) | rel_err vs fp32 参考 | 有限 |
|---|---|---|---|
| M=8（GEMV 最后一档） | 5120,5120,8 | 2.887e-04 | ✔ |
| M=9（GEMM 第一档） | 5120,5120,9 | 2.922e-04 | ✔ |
| N=1000（非 128 整 tile） | 1000,5120,32 | 2.863e-04 | ✔ |
| N=1025（奇数 N） | 1025,5120,64 | 2.892e-04 | ✔ |
| M=2048（大 prefill） | 5120,5120,2048 | 2.884e-04 | ✔ |
| `cfg=3`（64×128 tile） | 6144,5120,64 | 2.888e-04 | ✔ |
| `cfg=4`（128×128 tile） | 6144,5120,64 | 2.896e-04 | ✔ |
| K=5124（4 的倍数、非 32 的倍数） | 5120,5124,32 | 2.898e-04 | ✔ |

`WORST edge rel_err = 2.922e-04`，全部有限 ⇒ 源码里的零填充 + store 边界检查（`w8a8_sdot4.cu:190-208,312-326`）在非整 tile / 奇 N / 大 M 下都正确。

回退路径：
- **`K % 4 != 0`**：直接调 op 抛 `RuntimeError: ... K must be a multiple of 4`（`w8a8_sdot4.cu:376`），而 `apply_weights` 走 `triton_scaled_mm` 正常返回 `[4,512]` 且有限 ⇒ **安全网有效**（模型侧用不到，但不会崩）。
- **非 int8 权重**：`apply_weights` 的 `supported` 分支会转 `_triton_fallback`，而后者在 `triton_scaled_mm` 里断言 `input.dtype == weight.dtype` → `AssertionError`。该状态对 W8A8 层**不可达**（内核只在 int8 权重层被选中），记为**死分支/整洁性问题**，不是门禁项。


### 3.7 Provenance 注释（G3）

`csrc/rocm/w8a8_sdot4.cu:1-31` 文件头原文（节选）：

```c
// W8A8 INT8 GEMM/GEMV for gfx906 (Vega20) built on the DL-Ops instruction
// `V_DOT4_I32_I8` (clang builtin `__builtin_amdgcn_sdot4`, clamp=false).
//
// Provenance:
//   Ported on 2026-10-08 from the author's earlier out-of-tree project
//   `/home/lyh/server/int8_sdot4/w8a8_sdot4_kernel.hip` (484 lines, same
//   author, Apache-2.0), which validated this shape in production on the
//   same hardware.  The device code below is algorithmically identical; the
//   changes are limited to:
//     * registration as `_rocm_C` ops instead of a standalone
//       `PYBIND11_MODULE` (`w8a8_gemv` -> `w8a8_sdot4_gemv`,
//       `w8a8_gemm` -> `w8a8_sdot4_gemm`),
//     * dropped the CPU int32 oracle (the Python-side validation uses
//       `torch.matmul(xq.int(), wq.int().t())` as the exact oracle instead),
//     * added `*_meta` implementations so the ops can be traced by
//       `torch.compile` / fake tensor propagation.
```

## 4. 门禁判定

| # | 门禁 | 判定 | 证据 |
|---|---|---|---|
| 1.1-B | 迁移形态照抄（`__builtin_amdgcn_sdot4(...,false)`；M≤8 每列一 block/64 线程/LDS 归约；M>32 tile+BK=32 双缓冲） | ✅ | `csrc/rocm/w8a8_sdot4.cu:1-31`（形态/出处说明）、`:34-` device 码；`int`/`check`/`bench` 三模式真机通过（§3.3–3.5） |
| 1.1-接入 | 编进 `_rocm_C`（CMake）+ `is_supported=on_gfx906()` + ROCm 表第 0 位 + Triton 回退 | ✅ | `CMakeLists.txt:1548`；`gfx906_sdot4.py:61-73`；`kernels/linear/__init__.py:421-425`；`gfx906_sdot4.py:121-133,146-160`（回退分支） |
| 1.1-纪律 | `strings` 只含 gfx906；canonical 换新件并留旧 md5；无独立可执行程序 | ✅ | §3.2 三件架构串均 `amdgcn-amd-amdhsa--gfx906`；新 `ebd6fb8d…` / 旧 `5d1e50e1…`；全过程只跑 python/ninja，无自编译 `.out` |
| 1.1-数值 | 同 shape vs fp32 反量化参考 `rel_err ≤ 1e-2` | ✅ | §3.4：63/63 通过，worst `2.988e-04`；另有 §3.3 int32 精确性 35/35 逐位相同 |
| 1.1-性能 | 同 shape vs Triton，ABBA，"no shape regresses" | ✅ | §3.5：63/63 `kernel × ≥1.022`、`full × ≥1.148`；两次独立运行结论一致 |
| 1.1-端到端 | 27B decode 端到端更快 + 文本一致 | ❌ **不通过**（内核被选中后 27B `torch.compile` 抛 `ConstraintViolationError`，起不来；`=0` 正常） | §3.6.2/§3.6.3（三次对照实验定位到自定义 op 本体） |
| G1 承接 | 先盘点、不重写已有产物 | ✅ | §0；本步源码零改动（`git diff` 与第 1 任一致），只新增归档 + 运行证据 |
| G2 开关 | `VLLM_DCU_USE_INT8_SDOT4`（默认开）可 A/B 与回滚；无 monkey-patch | ✅ | `vllm/envs.py:168,1410-1414`；`gfx906_sdot4.py:69-70`；接入走内核表，无 patch |
| G3 出处 | 新文件头 `Provenance:` | ✅ | §3.7 |
| G4 架构 | 只含 gfx906 | ✅ | §3.2 |
| G5 增量构建 | 新 TU ≈1 个 + 链接，装 canonical | ✅ | §3.1：`[1/4] Running hipify` → `[2/3] Building HIP object …/w8a8_sdot4.hip.o` → `[3/3] Linking HIP shared module _rocm_C.abi3.so`（整次只编 1 个 HIP TU） |
| G6 数值 | M=1/2/4/8 GEMV + M=32/64/256 GEMM，27B 真实 K/N | ✅ | §3.4（M 覆盖 1/2/4/8/16/32/64/256/512，(N,K) = 7 个真实层形状） |
| G7 性能 | 同上 | ✅ | §3.5 |
| G8 端到端 | 27B TP=2 开关开/关对比 decode + 文本一致 | ❌ **不通过**（内核被选中后 27B `torch.compile` 抛 `ConstraintViolationError`，起不来；`=0` 正常） | §3.6.2（控制组 378.84/378.95 ms/token；处理组起不来） |
| G9 边做边写 | 归档先建骨架再追加 | ✅ | 本档 §0 为首批落盘内容（2026-10-09 上午），随后追加 §1–§3 |

**门禁汇总**：编译器侧（G4/G5）+ 数值侧（G6）+ 性能侧（G7）+ 承接/开关/出处（G1–G3、G9）**全过**；**端到端（1.1-端到端 / G8）不通过** —— 内核本身没问题，是它与 `torch.compile` 的交互导致 27B 起不来，须先修（§3.6.3 给出触发者与三个建议修法）。

## 5. 未完成项 / 阻塞与原因

1. **⛔ 端到端被阻塞（本步最严重）**：`VLLM_DCU_USE_INT8_SDOT4=1` 时 27B 在 `torch.compile` 阶段抛
   `torch.fx.experimental.symbolic_shapes.ConstraintViolationError`（`L['positions'].size()[1]` dynamic 却被特化成 2048），150 s 内退出；同一命令 `=0` 正常起服务并给出 378.84/378.95 ms/token。
   - 已定位触发者 = **新自定义 op 本体**（实验 C：同一 kernel 类换成 `_triton_fallback` 后 compile 173 s 通过并进入 capture）；
   - 已排除：cudagraph 模式、Python 的 M 分支、kernel 类其余部分（§3.6.3）；
   - 因此**处理组的 decode 数字与文本一致性缺失**，端到端门禁判不通过。
2. **未取得 4096+PIECEWISE 配方下的 sdot4 端到端数字**：已由实验 C 给出更根本结论，未再花 ~40 min/组重复。
3. **环境限制（新发现，非本步缺陷）**：构建容器 `z100-build-211` 的 `torchvision 0.25.0+das(torch2100)` 与容器内 `torch 2.11.0+das` ABI 不匹配，`import torchvision` 直接抛
   `RuntimeError: operator torchvision::nms does not exist`。凡会 import 到 `vllm.model_executor.kernels.linear`（→ `mixed_precision/cutlass` → `input_quant_fp8` → `vllm._aiter_ops` → `transformers` → `torchvision`）的探针/门禁都必须换 `z100-s9`/`z100-eager` 跑（两容器为 `torchvision 0.26.0`）。`--mode int`/`--mode check` 不受影响。
4. **本任无法直接写 `/mnt/game`**（宿主 sandbox 为 workspace-write，`/mnt/game` 只读；canonical 中 15 个文件为 `root:600`）。所有真机命令都经 `docker exec z100-s9/-build-211`（容器内 root）执行，因此证据文件是 root 所有；这不影响结论。
5. **未跑的补充项**：`w8a8_sdot4_gemv` 的 `nthreads` 扫描（只用 tuned 值 64）、`cfg=1/2` 的显式性能对比（`cfg=0` 自动选择已覆盖 M≤32 与 M>32 两档）、profiler（`rocprof`/`hipprof`）级别的 kernel 计时（本步用 CUDA event 的 kernel-only ABBA 替代，见 §3.5）。

## 6. 对后续步骤的影响

1. **⚠️ S9 / 生产当前必须先关（本步否决"默认自动生效"）**：虽然内核表默认把 `Gfx906Sdot4Int8ScaledMMLinearKernel` 放在 `PlatformEnum.ROCM` INT8 表第 0 位、`is_supported()` 返回 True，但 §3.6.2/§3.6.3 证明**打开它会直接让 27B 起不来**（`torch.compile` 崩）。因此在上游修好之前，生产/压测**必须显式** `VLLM_DCU_USE_INT8_SDOT4=0`，或按本节第 2 条硬回滚 `_rocm_C.abi3.so`。**修好后**的生产行为是：27B W8A8（channel-wise 权重 + per-token 动态对称激活）自动走 sdot4、decode（M≤8）走 GEMV、prefill/大 batch（M>8）走 tiled GEMM，无需 flag。
   - 若要在修好之前验证性能收益，只能走实验 C 那种"同 kernel 类、op 换成 Triton"的对照（不能用于生产，因为那等于没启用新 op）。
2. **怎么关 / 怎么回滚**：
   - 软开关：`VLLM_DCU_USE_INT8_SDOT4=0`（等价于 `false`/`False`）→ `is_supported()` 返回 False，内核表落到 `TritonInt8ScaledMMLinearKernel`，行为与 S9b 基线逐位一致；
   - 硬回滚：`cp /mnt/game/port/s11e/_rocm_C.abi3.so.before /mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so`（旧件 `5d1e50e1a77de74f2d308519f8120745`，35,156,368 B）。
3. **对 S11f（P6，FP16 模拟 FP8）**：可复用的是"gfx906 上 `v_dot4_i32_i8` + LDS 归约 + tile 化的 HIP 模板"与"`csrc/rocm/**` → `_rocm_C` → Python 内核类 → 内核表第 0 位"的接法；但**"自定义 op + C++ `Meta` 注册"这一段在 27B + torch.compile 下不成立**（§3.6.3），照抄前须先修（建议改用 Python 侧 `torch.library.custom_op` + `register_fake`，或把 op 加进 `splitting_ops`）。（FP8 在 gfx906 无转换指令，见 S11d。）
4. **对 P1/P3**：decode 的 GEMM 部分已被本步显著改善（M=1 时 7 个 shape 上 ×6.8–15.8 全算子口径），实测峰值 22.2 TFLOPS（vs Triton 18.0）。若后续 P0 剖析要重新分配预算，应把"GEMM 仍是主瓶颈"这个前提按本步结果下调。
5. **注意事项（给 Lead）**：`_rocm_C.abi3.so` 变大 1.03 MB（35.16 → 36.18 MB），只增不减；**新 op 的 Meta 实现"足够让 torch.compile 不 graph break"这个假设被 §3.6 证伪** —— 恰恰是这条 op 让 27B 的编译抛 `ConstraintViolationError`。S11f 若要照抄"自定义 op + C++ Meta 注册"的接入路径，**必须先解决本步 §3.6.3 的编译问题**，否则新算子同样会在 27B 上起不来。
6. **环境坑（新发现，建议写入 todolist 硬约束）**：构建容器 `z100-build-211` 里 `import torchvision` 失败（torchvision 0.25.0+das 对 torch 2.10 编译、容器内 torch 已是 2.11.0+das），凡是会 import 到 `vllm.model_executor.kernels.linear`（→ `mixed_precision/cutlass` → `input_quant_fp8` → `vllm._aiter_ops` → `transformers` → `torchvision`）的探针/门禁，**必须换到 `z100-s9` 或 `z100-eager` 跑**（两容器 torchvision 0.26.0 正常）。

## 7. 复现方式

```bash
# 0. 前置：canonical 副本已含新件（md5 ebd6fb8dac572514cf94ca7ec7d828a7）

# 1. int32 精确性（35 组）
docker exec z100-s9 bash -lc 'cd /mnt/game/port/gfx906-vllm-211; . /mnt/game/port/setenv-211.sh >/dev/null 2>&1;   export HIP_VISIBLE_DEVICES=2; export PYTHONPATH=/mnt/game/triton-gfx906/install:$PYTHONPATH;   export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache/s11e;   python tools/dcu/s11e_sdot4_eval.py --mode int 2>&1 | tee /mnt/game/port/s11e/int.log'

# 2. 逐 shape 数值门禁（63 组，gate rel_err<=1e-2）
docker exec z100-s9 bash -lc '... python tools/dcu/s11e_sdot4_eval.py --mode check 2>&1 | tee /mnt/game/port/s11e/check.log'

# 3. 逐 shape ABBA 性能（63 组）
docker exec z100-s9 bash -lc '... python tools/dcu/s11e_sdot4_eval.py --mode bench 2>&1 | tee /mnt/game/port/s11e/bench-rerun.log'

# 4. 架构纯净性（任一台宿主）
for f in _C _C_stable_libtorch _rocm_C; do echo "== $f";   strings /mnt/game/port/gfx906-vllm-211/vllm/$f.abi3.so | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sort -u; done

# 5. 补充鲁棒性探针（边界 shape + 回退路径）
docker exec z100-s9 bash -lc 'cd /mnt/game/port/gfx906-vllm-211; . /mnt/game/port/setenv-211.sh >/dev/null 2>&1; \
  export HIP_VISIBLE_DEVICES=2; export PYTHONPATH=/mnt/game/triton-gfx906/install:$PYTHONPATH; \
  export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache/s11e; \
  timeout 600 python /src/tools/dcu/s11e_sdot4_edge.py 2>&1 | tee /mnt/game/port/s11e/edge.log'

# 6. 端到端 A/B（27B W8A8 TP=2，卡 2,3；控制组先跑）
docker exec -i z100-s9 bash -s < /tmp/s11e/e2e_run2.sh     # 见 §3.6.2；日志 /mnt/game/port/s11e/{e2e.log,27b-ctl-triton-8192*.log,27b-sdot4-8192*.log}
# 处理组必崩（150 s）的判定命令：
#   docker exec z100-s9 grep -c "Constraints violated" /mnt/game/port/s11e/27b-sdot4-8192.log   # -> 5

# 7. 崩因定位实验（诊断用，均已执行；补丁只在容器 /tmp 内，未进仓库）
#   C（决定性）：同一 kernel 类、op 换成 _triton_fallback
docker exec -i z100-s9 bash -c 'cat > /tmp/s11e_patch_triton_only.py' < /tmp/s11e/patch_triton_only.py
#   -> markers=3  constraint_violations=0  torch.compile took 173.05 s  capture_bars=1
#   B（负结果）：去掉 Python 的 M 分支、始终 GEMM op
#   -> markers=3  constraint_violations=5  capture_bars=0
#   实验 C 补丁全文（自包含，直接粘到容器 /tmp/s11e_patch_triton_only.py）：
#     import torch
#     from vllm import _custom_ops as ops
#     import vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4 as M
#     K = M.Gfx906Sdot4Int8ScaledMMLinearKernel
#     def apply_weights_triton_only(self, layer, x, bias=None):
#         w, w_s, _i_s, _i_zp, _azp = self._get_layer_params(layer)
#         x_2d = x.reshape(-1, x.shape[-1])
#         out_shape = (*x.shape[:-1], w.shape[0])
#         x_q, x_s, _ = ops.scaled_int8_quant(x_2d.contiguous(), None, None, symmetric=True)
#         out = self._triton_fallback(x_q, x_s.reshape(-1), w, w_s, bias, x.dtype)
#         return out.view(out_shape)
#     K.apply_weights = apply_weights_triton_only
#     print("[diag] PATCHED apply_weights -> Triton-only (no custom op)", flush=True)
#     if __name__ == "__main__":
#         from vllm.entrypoints.cli.main import main
#         main()
#     然后：
#       VLLM_DCU_USE_INT8_SDOT4=1 python /tmp/s11e_patch_triton_only.py serve <模型> \
#         --dtype float16 --tensor-parallel-size 2 --max-model-len 8192 \
#         --max-num-batched-tokens 2048 --gpu-memory-utilization 0.9 --disable-custom-all-reduce \
#         --skip-mm-profiling --enable-prefix-caching --mamba-cache-mode align --port 8016
#   实验 B 同形，只把函数体换成"无条件 torch.ops._rocm_C.w8a8_sdot4_gemm(...)"。
#   （这两个补丁只用于**定位**，不是产品实现；产品实现里不要用 monkey-patch。）

# 6. 软回滚验证（应回落到 Triton）
docker exec z100-s9 bash -lc 'VLLM_DCU_USE_INT8_SDOT4=0 python -c "
import torch;from vllm.model_executor.kernels.linear.scaled_mm.gfx906_sdot4 import Gfx906Sdot4Int8ScaledMMLinearKernel as K
print(K.is_supported())"'
```

## 8. 参考

1. `docs/海光Z100-性能优化方案.md` §4（P2）、§4A（算力清单）、§4C.1（exact-shape 战术表 / no shape regresses）
2. `docs/result/z100-11d-z100-isa-capability-audit-result.md`（`v_dot4_i32_i8` 可用性；"汇编器接受 ≠ 硬件支持"）
3. `docs/result/z100-11b-perf-profiling-result.md`（P0 剖析；本档未从中取 shape 分布 —— 该档无逐 shape 表，27B 真实 (N,K) 由 `/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8/{config.json,model.safetensors.index.json}` 与 `tools/dcu/s11e_sdot4_eval.py:41-50` 给出）
4. `docs/result/z100-07c-packed-silu-csrc-fix-result.md`、`docs/result/z100-04d-build-incrementality-result.md`（csrc → 增量重建 → 装 canonical → md5/回滚）
5. `docs/z100-todolist.md` 硬约束 1–21（15：`_C` 由 `_C_stable_libtorch` 提供；21：用户已授权新增/重编 HIP 算子）
6. 前序工程（只读）：`/home/lyh/server/int8_sdot4/{w8a8_sdot4_kernel.hip,int8_sdot4_patch.py,check_int8_sdot4.py,ab_measure.py}`
