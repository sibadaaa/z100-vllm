# S6 Triton gfx906 支持固化 — 执行结果

> 归档时间：2026-10-08（宿主 UTC 2026-10-07；容器时区 CST，**本档命令时间戳均为容器 CST**）
> 执行者：子 agent（S6 / triton-gfx906）
> 对应设计文档：[docs/design/z100-06-triton-gfx906.md](../design/z100-06-triton-gfx906.md)
> 前一档：S4b [z100-04b-stack-upgrade-result.md](z100-04b-stack-upgrade-result.md)（§8 入口结论）· S5 [z100-05-csrc-kernels-result.md](z100-05-csrc-kernels-result.md)（14 个失败移交本步）
> 状态：✅ **完成**（门禁 1、门禁 2 全过；14/14 `unsupported target` 失败全部消除；门禁 3 属 S8 联测）

---

## 1. 目标与验收标准

### 1.1 设计文档门禁原文（`docs/design/z100-06-triton-gfx906.md` §验收）

1. `python tools/dcu/probe_triton.py` → `TRITON_OK`（误差 < 1e-5）；
2. fork 的两个 gfx906 Triton attention 测试通过：
   `tests/kernels/attention/test_triton_unified_attention_gfx906.py`、
   `tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py`；
3. 8K 上下文服务日志出现 `TRITON_ATTN`（与 S8 联测）；
4. 若使用自编译 triton，记录 `triton.__version__`/源码 commit 与构建日志路径。

### 1.2 本任务书追加的验收标准

- 产出**能在 gfx906 上编出并跑通 triton 内核**的 triton：源码、构建树、安装前缀全部落 `/mnt/game`，
  经 `PYTHONPATH` 前置使用，**不覆盖/不替换 DTK 自带 triton**；
- 在**新建容器 `z100-triton`**（同镜像、GPU/group 参数照抄 `z100-build-211`、`--memory=10g`、不加 `--rm`）内完成，
  不在 `z100-build`/`z100-build-211` 内做 triton 源码构建；
- 统计 S5 的 14 个 `unsupported target: 'gfx906'` 失败消除多少；
- 固化补丁 / 探针 / 构建脚本。

### 1.3 本步额外发现并达成的验收

- 门禁 2 的两个文件在只加 VEGA 补丁后**仍失败 10 项**（编译通过、数值错），
  根因定位为 **gfx906 SFU（超越函数）与前一 VALU 指令的 hazard**（见 §3.5），
  补丁修正后两文件 **39/39 + 12/12 全过**。该根因**不是本步引入**：生产验证过的
  `triton-3.4.0+gitc817b9b6` 轮子同样复现（§3.5.3）。

---

## 2. 实际改动

### 2.1 仓库源码改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `tools/dcu/triton-gfx906-3.6.0.patch` | 新增（133 行 / 5 文件） | gfx906 支持完整补丁，见下表 |
| `tools/dcu/probe_triton.py` | 新增 | 最小 gfx906 探针（add / warp-reduce / int8 dot + `v_dot` 计数） |
| `tools/dcu/build_triton_gfx906.sh` | 新增 | 可复现构建脚本（建容器 → clone v3.6.0 → 打补丁 → 构建到 `/mnt/game` → 门禁命令） |

补丁 5 个文件（全部落在 triton 源码，不触碰 DTK 安装）：

| # | 文件 | 行 | 改动 | 归类 |
|---|---|---|---|---|
| 1 | `third_party/amd/include/TritonAMDGPUToLLVM/TargetUtils.h` | +1 | `ISAFamily` 增加 `VEGA`（保持 `isCDNA()==false`） | VEGA 补丁（移植自 3.4） |
| 2 | `third_party/amd/lib/TritonAMDGPUToLLVM/TargetUtils.cpp` | +3 | `GK_GFX906 → VEGA`；`supportsVDot()` 含 `VEGA`（sdot4/fdot2 根基） | 同上 |
| 3 | `third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp` | +2 | `getWarpSize()` VEGA=64；`warpReduce()` VEGA 走 DPP 广播（gfx906 无 `v_permlane16_b32`） | 同上 |
| 4 | `third_party/amd/backend/driver.py` | +20 | `hip_utils` 模块改用 **HIP runtime 自带头文件**（`$ROCM_PATH/hip/include`）编译；launcher 仍用 vendored 头（需要 `HIP_LAUNCH_CONFIG`） | DTK 兼容（本步新增） |
| 5 | `third_party/amd/backend/compiler.py` | +22 | gfx906：为 SFU 指令插入 `s_nop 2`（`insert_sfu_waits()`，仅 `"gfx906" in options.arch` 时生效） | gfx906 hazard（本步新增） |

**`vllm/**` 无任何改动**（本步不需要；`PYTHONPATH` 前置即可让 fork 用上自编译 triton）。

### 2.2 在 `/mnt/game` 下新建的资源（宿主对 `/mnt/game` 只读，全部经 `docker exec` 写入）

```
/mnt/game/triton-gfx906/src/                 triton v3.6.0 源码（7c56a5e）+ 已打补丁
/mnt/game/triton-gfx906/install/             978 MB 安装前缀（libtriton.so 748,298,568 B）★ 门禁所用
/mnt/game/triton-gfx906/install-prepatch-copy/ 仅手工同步过 2 个 .py 的旧前缀（备份，内容与 install 一致，仅 .pyc/RECORD 不同）
/mnt/game/triton-gfx906/cache/                TRITON_CACHE_DIR（JIT 产物）
/mnt/game/triton-gfx906/logs/build-3.6.0.log  首次构建日志（37 分钟）
/mnt/game/triton-gfx906/logs/rebuild-final.log 补丁后增量重建日志（exit=0）
/mnt/game/triton-gfx906/logs/probe-triton-3.6.0.log  探针原始输出
/mnt/game/triton-gfx906/logs/s6-attn-tests-sfufix.log 6 个测试文件（修复后）原始输出
/mnt/game/triton-gfx906/logs/s6-gate2-final.log       canonical 产物上的门禁 2 复跑
/mnt/game/triton-gfx906/triton-gfx906-3.6.0.patch     补丁副本
/mnt/game/triton-gfx906/{repro*,diag_attn*}.py        SFU hazard 最小复现与 attention 诊断脚本
/mnt/game/triton-gfx906/ref/{nlzy-3.5.x,das-wheel,triton340}/  对照物（nlzy 三文件 / das 轮子 amd 后端 / 生产 3.4 轮子）
/mnt/game/port/gfx906-vllm-triton/           2.11 fork 副本（441 MB，供测试，避免与 S4c 争用 -211）
```

### 2.3 容器实际配置（`docker inspect z100-triton` 原文）

```
Image=harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413
Memory=10737418240 ShmSize=67108864 GroupAdd=[video] SecurityOpt=[label=disable]
Devices=/dev/kfd:/dev/kfd /dev/mkfd:/dev/mkfd /dev/dri:/dev/dri
Mounts=/home/lyh/z100-vllm->/src(rw=false) /mnt/game->/mnt/game(rw=true) /opt/hyhal->/opt/hyhal(rw=false)
```

> 与 `z100-build-211` 的设备/group/memory 参数一致（`docker inspect` 逐项照抄），未加 `--rm`，
> 当前仍在运行（`docker ps` 可见 `z100-triton  Up`）。

---

## 3. 执行过程与原始证据

### 3.0 源码基线与版本判定

```console
$ git clone --depth 1 -b v3.6.0 https://github.com/triton-lang/triton /mnt/game/triton-gfx906/src
$ cd /mnt/game/triton-gfx906/src && git log --oneline -1
7c56a5e [release 3.6] Release Triton to pypi (#9254)          # == tag v3.6.0
$ cat cmake/llvm-hash.txt
f6ded0be897e2878612dd903f7e8bb85448269e5
```

- 上游 **v3.6.0 源码可得**（设计文档担心的"3.6 只有轮子"不成立），故按优先策略直接用 3.6.0 作为基座，
  不退化到 3.5.x，**不存在版本偏差**（目标栈 `triton 3.6.0+das185`）。
- 与目标栈的差异只在 **构建方**：das 轮子是海光用自己的 triton fork + DTK 工具链出的；
  本步用 **上游 v3.6.0 源码 + 上游预编译 LLVM（`llvm-f6ded0be-ubuntu-x64`，1,211,953,633 B，setup.py 自动下载）**
  + DTK 运行时（HIP/ld.lld 来自 `/opt/dtk`）。
  Python 与 `libtriton.so` **同源自洽**，因此 S4b 记录的"`add_convert_to_buffer_ops` 5 vs 4 参"ABI 不一致**不存在**，
  **不需要** `AMDGCN_USE_BUFFER_OPS=0`（本步实测该开关既不解决也不影响 gfx906 出码，见 §3.5.4）。
- 版本判定仍按 S4b 口径使用 `importlib.metadata`（DTK 轮子的 `torch.__version__` 只报基础版号）。

### 3.1 打补丁（可直接 `git apply`）

```console
$ cd /mnt/game/triton-gfx906/src && git apply /src/tools/dcu/triton-gfx906-3.6.0.patch
$ git diff --stat
 third_party/amd/backend/compiler.py                     | 22 ++++++++++++++++++++++
 third_party/amd/backend/driver.py                       | 21 ++++++++++++++++++++-
 third_party/amd/include/TritonAMDGPUToLLVM/TargetUtils.h |  1 +
 third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp    |  3 ++-
 third_party/amd/lib/TritonAMDGPUToLLVM/TargetUtils.cpp   |  3 +++
 5 files changed, 48 insertions(+), 2 deletions(-)
```

未打补丁时 `deduceISAFamily("gfx906")` 返回 `Unknown`，`ConvertTritonAMDGPUToLLVM` 直接
`emitError("unsupported target: 'gfx906'")`（`third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp:114`）——
这正是 S5 那 14 个失败的来源，也是本补丁的落点。

**补丁对 pristine v3.6.0 可用性验证**（另开一份干净 clone）：

```console
$ git clone -q --depth 1 -b v3.6.0 https://github.com/triton-lang/triton /mnt/game/triton-gfx906/verify-src
$ cd /mnt/game/triton-gfx906/verify-src && git log --oneline -1
7c56a5e [release 3.6] Release Triton to pypi (#9254)
$ git apply --check /src/tools/dcu/triton-gfx906-3.6.0.patch && echo GIT_APPLY_CHECK_OK
GIT_APPLY_CHECK_OK (pristine v3.6.0)
```

### 3.2 构建（37 分钟，MAX_JOBS=4）

```console
$ docker exec z100-triton bash -lc '... /mnt/game/triton-gfx906/build.sh ...'   # 见 tools/dcu/build_triton_gfx906.sh
=== build start Thu Oct  8 07:31:13 AM CST 2026 MAX_JOBS=4 gpu_arch=12 ===
Processing ./.
  Preparing metadata (pyproject.toml): finished with 'done'
Building wheels for collected packages: triton
  Building wheel for triton (pyproject.toml): started
  ...（省略 37 分钟编译/链接输出）
  Created wheel for triton: filename=triton-3.6.0+git7c56a5e4-cp310-cp310-linux_x86_64.whl
          size=342307619 sha256=4838a4f4bce66ff07bbf7c546376a28de28ab29946b6b2305646b8ea3d024319
Successfully built triton
Successfully installed triton-3.6.0+git7c56a5e4
=== build exit=0 Thu Oct  8 08:08:19 AM CST 2026 ===
```

- 构建期间 cgroup 峰值 `memory.current` 多次触及 10 GiB 上限（10240 MB），**无 OOM**（page cache 可回收），
  `docker stats` 未见异常；`MAX_JOBS=4` 是这台机器（整机 31 GB、可用 ~12 GB）的安全值。
- 增量重建（补丁 4/5 落地后）：`=== rebuild exit=0 Thu Oct  8 08:42:54 AM CST 2026 ===`，
  重建前缀与门禁所用前缀**除 `.pyc`/`RECORD` 外逐字节一致**：

```console
$ diff -rq -x __pycache__ .../install-prepatch-copy .../install | wc -l
1                      # 仅 dist-info/RECORD
$ md5sum .../install-prepatch-copy/triton/_C/libtriton.so .../install/triton/_C/libtriton.so
9ded5d23a68068c33a5e5bd3f73d0233  .../install-prepatch-copy/triton/_C/libtriton.so
9ded5d23a68068c33a5e5bd3f73d0233  .../install/triton/_C/libtriton.so
```

### 3.3 **DTK 自带 triton 未被触碰**（硬约束）

```console
$ ls -la /usr/local/lib/python3.10/dist-packages/triton/_C/libtriton.so
-rwxr-xr-x 1 root root 870759392 Apr  2  2026 /usr/local/lib/python3.10/dist-packages/triton/_C/libtriton.so
$ python -c "import importlib.metadata as m; print(m.version('triton'))"
3.4.0+git1ef59765                      # 镜像原装，mtime 仍是镜像构建日
$ ls -la /mnt/game/triton-gfx906/install/triton/_C/libtriton.so
-rwxr-xr-x 1 root root 748298568 Oct  8 08:42 .../install/triton/_C/libtriton.so
```

安装方式为 `pip install . --target /mnt/game/triton-gfx906/install`（**不是** `pip install .`），
所以 `/usr/local/lib/python3.10/dist-packages` 全程未被写入；使用时靠 `PYTHONPATH` 前置。

### 3.4 门禁 1：最小探针（`tools/dcu/probe_triton.py`）

```console
$ docker exec z100-triton bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; . /src/tools/dcu/env.sh >/dev/null 2>&1;
    export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache; export PYTHONPATH=/mnt/game/triton-gfx906/install;
    python /src/tools/dcu/probe_triton.py'
triton.__version__       = 3.6.0
triton.__file__          = /mnt/game/triton-gfx906/install/triton/__init__.py
importlib.metadata       = 3.6.0+git7c56a5e4
torch.__version__        = 2.10.0
importlib.metadata torch = 2.10.0+das.opt1.dtk2604.20260325.g6b060a
target                   = GPUTarget(backend='hip', arch='gfx906', warp_size=64)
device                   = Z100L
gcnArchName              = gfx906:sramecc+:xnack-
triton cache dir         = /mnt/game/triton-gfx906/cache
======== check 1: elementwise add (lowering accepts gfx906) ========
max_abs_err = 0.0
TRITON_OK
======== check 2: warp-level reduce (TargetInfo::warpReduce / DPP path) ========
max_abs_err = 0.0
REDUCE_OK
======== check 3: int8 dot (supportsVDot / v_dot4 path) ========
max_abs_err = 0
INT8DOT_OK
v_dot occurrences in amdgcn asm = 32
VDOT_IN_ASM = True
PROBE_RC = 0
```

> 同一探针在 torch **2.11** 栈（`z100-build-211`，`PYTHONPATH=/mnt/game/triton-gfx906/install:<fork>`）下
> `target = GPUTarget(backend='hip', arch='gfx906', warp_size=64)`、`triton 3.6.0+git7c56a5e4`、`torch 2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7`，
> fork `import vllm` 正常——即新栈同样可用（门禁 2 就是在这个栈上跑的）。

### 3.5 修复过程：只加 VEGA 补丁不够，定位到 gfx906 SFU hazard

#### 3.5.1 VEGA 补丁落地后：编译全过，但 10 项数值错

```console
（VEGA 补丁 + driver.py 修复后，尚未加 SFU wait）
======================== 10 failed, 29 passed, 1 warning in 23.87s ========================
 tests/kernels/attention/test_triton_unified_attention_gfx906.py
=========================== 2 failed, 10 passed, 1 warning in 22.31s ========================
 tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py
```

`unsupported target` **已经全部消失**，失败变成数值不匹配：

```console
E       AssertionError: Tensor-likes are not close!
E       Mismatched elements: 264 / 320 (82.5%)
E       Greatest absolute difference: 0.05078125 at index (0, 1, 98) (up to 0.001 allowed)
tests/kernels/attention/test_triton_unified_attention_gfx906.py:276: AssertionError
```

诊断脚本 `/mnt/game/triton-gfx906/diag_attn.py`（fp32 eager 参考 + 两次重复运行）：

```console
head=160 gqa=1 seqs=1  |triton-eagerfp32| = 0.050783   |eagerfp16-eagerfp32| = 0.000064   |triton1-triton2| = 0.000000e+00
head=192 gqa=4 seqs=1  |triton-eagerfp32| = 0.103671   |eagerfp16-eagerfp32| = 0.000075   |triton1-triton2| = 0.000000e+00
head=256 gqa=6 seqs=2  |triton-eagerfp32| = 0.288994   |eagerfp16-eagerfp32| = 0.000114   |triton1-triton2| = 2.890015e-01  ← 非确定
head=128 gqa=4 seqs=2  |triton-eagerfp32| = 0.203134   |eagerfp16-eagerfp32| = 0.000093   |triton1-triton2| = 0.000000e+00
```

- eager fp16 与 fp32 一致到 1e-4 ⇒ 参考没错；triton 输出错到 0.05~0.29，且**同一输入两次运行结果不同** ⇒ 不是精度问题。
- `segm_output/segm_max/segm_expsum` 两次运行**逐位相同**（`differing entries: 0 / 98304`），
  但最终 `out` 差 0.289 ⇒ 问题发生在 `reduce_segments` 及其调用的超越函数上。

#### 3.5.2 最小复现（与 attention 无关的 6 行 kernel）

```console
$ python /mnt/game/triton-gfx906/repro9.py          # S=16, num_warps=1
exp2(m)        bad=OK                 vals=[..., 1.4723, 3.7821, 1.2628, 1.1896, ...]
exp2(m*2)      bad=[8, 9, 10, 11]     vals=[..., 1.4723, 3.7821, 1.2628, 1.1896, ...]   # == exp2(m)：SFU 读到 VALU 写入前的旧值
exp2(m+0.5)    bad=[8, 9, 10, 11]     vals=[..., 1.0, 1.0, 1.0, 1.0, ...]               # == exp2(0)
sqrt(m*4)      bad=[8, 9, 10, 11]     vals=[..., 0.0, 0.0, 0.0, 0.0, ...]               # == sqrt(0)
```

规律（`repro4.py` 全尺寸矩阵，S=8..256 × num_warps=1/2/4）：

```console
S=  16 nw=1 exp:[8, 9, 10, 11]  exp2:OK  log:OK  sqrt:OK  mul:OK
S=  32 nw=1 exp:[8, 9, 10, 11, 24, 25, 26, 27]  exp2:OK  log:[8, 9, 10, 11]  sqrt:OK  mul:OK
S=  64 nw=1 exp:[8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59]  ...
S= 128 nw=1 exp:OK  ...
```

- 出错位置固定为**每个 SIMD16 的第 8–11 道**（`i % 16 ∈ {8,9,10,11}`），4 张卡（`cuda:0..3`）**完全一致**；
- **只在 SFU 指令的操作数由「紧邻的前一条 VALU 指令」写入时发生**：`tl.exp2(m)`（load→SFU）正常，
  `tl.exp2(m*2.0)`（VALU→SFU）出错；`tl.sqrt(m*4)` 同理，而纯 VALU 的 `m*2.0` 存储正常；
- 位置与 `num_warps` 无关、与 buffer ops 开关无关（`AMDGCN_USE_BUFFER_OPS=0` 同样出错）⇒ 非 MUBUF 路径问题；
- 反汇编确认 `.hsaco` 与 `.s` 一致（`v_add_f32_e32 v2, v2, v2` → `v_exp_f32_e32 v2, v2`，`s_waitcnt vmcnt(0)` 也齐）。

#### 3.5.3 该缺陷**不是本步引入**：生产验证过的 3.4 轮子同样复现

```console
$ pip install --no-deps --target /mnt/game/triton-gfx906/ref/triton340 \
      /mnt/game/port/wheels/triton-3.4.0+gitc817b9b6-cp310-cp310-linux_x86_64.whl
$ PYTHONPATH=/mnt/game/triton-gfx906/ref/triton340 python /tmp/repro4.py
/mnt/game/triton-gfx906/ref/triton340/triton/__init__.py 3.4.0+gitc817b9b6
S=  16 nw=1 exp:[8, 9, 10, 11]  exp2:OK  log:OK  sqrt:OK  mul:OK
S=  64 nw=1 exp:[8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59]  ...
```

⇒ VEGA 补丁路线（3.4 与 3.6）在 gfx906 上**一直带着这个数值地雷**，S5/S4b 只是被
`unsupported target` 挡在前面没暴露出来。**这一条对 S7/S8/S9 与生产都有直接影响**（§6）。

#### 3.5.4 修复：SFU 前插一个等待槽（`s_nop 2`）

在 `third_party/amd/backend/compiler.py::HIPBackend.make_hsaco` 中，仅当 `"gfx906" in options.arch` 时
对 AMDGCN 文本插入等待槽（`insert_sfu_waits()`，正则 `^\s*v_(?:exp|log|sqrt|rcp|rsq|sin|cos|ldexp)\w*`）。
实验证据（`s_nop` 确实进了二进制，用 `/opt/dtk/llvm/bin/llvm-objdump -d` 计数）：

```console
nops=0 bad=[8, 9, 10, 11]   s_nop_count=0
nops=2 bad=OK               s_nop_count=1
nops=8 bad=OK               s_nop_count=1
```

修复后全矩阵回归（`repro4.py`）：

```console
S=   8..256 nw=1/2/4: exp:OK  exp2:OK  log:OK  sqrt:OK  mul:OK      （18/18 行全 OK）
repro9.py: exp2(m*2) bad=OK / exp2(m+0.5) bad=OK / exp2(m*2+0.5) bad=OK / sqrt(m*4) bad=OK
```

> ⚠️ 缓存口径：`compile_module_from_src` 与 JIT 磁盘缓存**不包含 include_dirs / 后端 Python 代码的哈希**，
> 打这类补丁后必须清 `TRITON_CACHE_DIR`（本次每次验证都清），否则会命中旧 `.so`/`.hsaco`，
> 出现"补丁没生效"的假象（本步踩过两次，见 §5.2）。

### 3.6 门禁 2：两个 unified-attention gfx906 测试 + S5 五文件全量统计

跑法（先 `import torch`，`--noconftest`，避免根 conftest 拉起整条应用链）：

```console
$ docker exec z100-build-211 bash -lc '
    source /opt/dtk/env.sh >/dev/null 2>&1
    . /mnt/game/port/gfx906-vllm-triton/tools/dcu/env.sh >/dev/null 2>&1
    cd /mnt/game/port/gfx906-vllm-triton
    export PYTHONPATH=/mnt/game/triton-gfx906/install:/mnt/game/port/gfx906-vllm-triton
    export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
    python -c "import torch, pytest, sys; sys.exit(pytest.main([\"-v\",\"--no-header\",\"--noconftest\",\"-p\",\"no:cacheprovider\",\"<file>\"]))"'
```

修复后（`logs/s6-attn-tests-sfufix.log`；`grep FAILED` 计数 = **0**）：

```console
########## FILE: tests/kernels/attention/test_triton_unified_attention_gfx906.py 08:40:42 ##########
======================== 39 passed, 1 warning in 25.18s ========================
########## RC=0 ... 08:41:15 ##########
########## FILE: tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py 08:41:15 ##########
======================== 12 passed, 1 warning in 22.16s ========================
########## RC=0 ... 08:41:44 ##########
########## FILE: tests/kernels/mamba/test_fused_recurrent_packed_decode.py 08:41:44 ##########
======================== 5 passed, 1 warning in 16.29s ========================
########## RC=0 ... 08:42:09 ##########
########## FILE: tests/kernels/mamba/test_fused_sigmoid_gating.py 08:42:09 ##########
======================== 5 passed, 1 warning in 15.35s ========================
########## RC=0 ... 08:42:32 ##########
########## FILE: tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py 08:42:32 ##########
======================= 13 passed, 3 warnings in 27.09s =======================
########## RC=0 ... 08:43:07 ##########
########## FILE: tests/model_executor/test_qwen3_5_gdn.py 08:43:07 ##########
ERROR tests/model_executor/test_qwen3_5_gdn.py - RuntimeError: operator torch...   # 采集期报错，见 §5.3
########## RC=2 ... 08:43:32 ##########
```

canonical 产物（`install/` 换成本次重建的树）上复跑门禁 2（`logs/s6-gate2-final.log`）：

```console
########## FILE: tests/kernels/attention/test_triton_unified_attention_gfx906.py 08:43:53 ##########
======================== 39 passed, 1 warning in 16.50s ========================
########## RC=0 ... 08:44:16 ##########
########## FILE: tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py 08:44:16 ##########
======================== 12 passed, 1 warning in 16.70s ========================
########## RC=0 ... 08:44:40 ##########
```

**S5 14 个 `unsupported target` 失败的去向（逐项）**

| S5 失败项 | S5 结果 | 本步（修复后） |
|---|---|---|
| `test_unified_attention_wide_head_decode_matches_eager[…]` 8 项 | 12 个失败（attention 文件） | **全过**（文件 39 passed / 0 failed） |
| `test_unified_attention_multi_query_3d_matches_eager[…]` 4 项 | 同上 | **全过** |
| `test_fused_recurrent_packed_decode.py::test_packed_decode_batch_head_grid[unsplit/split-grid]` 2 项 | 2 failed | **全过**（文件 5 passed） |
| **合计** | **14 个 `unsupported target`** | **14/14 消除** |

五文件同口径对比（S5 → 本步）：`66 passed / 16 failed` → **`74 passed / 0 failed`**
（39+12+5+5+13；`qwen3_5_gdn` 因 §5.3 环境缺口未计入）。S5 另有的 2 个
`test_qwen3_5_gdn.py` 替身失败（`seq_lens` / `ignore_unexpected_prefixes`）本步**未复评**（采集期即失败）。

### 3.7 顺带澄清的两个入口结论（对应 S4b §8 的遗留）

1. **`AMDGCN_USE_BUFFER_OPS=0` 在本步不再需要**：ABI 不一致是 das 轮子"Python 5 参 / 原生 4 参"造成的，
   自编译 3.6.0 两侧同源；实测开关对 gfx906 出码与数值无影响（§3.5.2）。
2. **不需要 hcu 后端 / `aillvm`**：上游 v3.6.0 只有 `amd`（HIP）后端，`driver.active.get_current_target()`
   直接报 `arch='gfx906', warp_size=64`，`amd` 后端出码可用；das 轮子的
   `backends/hcu` + `/opt/dtk/aillvm` 缺失警告与之无关。

### 3.8 与 `nlzy/triton-gfx906` 的对照（设计文档 §4）

（对照物取自 `gfx906/v3.5.x`：`ref/nlzy-3.5.x/TargetUtils.{h,cpp}`、`TargetInfo.cpp`）

| 点 | nlzy（3.5.x） | 本步（3.6.0） |
|---|---|---|
| `ISAFamily` | 新增 `VEGA20`，并**折进 `isCDNA()==true`** | 新增 `VEGA`，**`isCDNA()` 保持 false**（沿用生产验证结构） |
| `deduceISAFamily` | `GK_GFX906 → VEGA20` | `GK_GFX906 → VEGA` |
| `supportsVDot()` | 含 `VEGA20` | 含 `VEGA`（含 `RDNA4`，3.6 新增枚举） |
| warp 64 | 靠 `isCDNA()` 间接得到 | `getWarpSize()` 显式 `case VEGA: return 64` |
| `warpReduce` DPP | 走 CDNA 分支（同效） | 显式 `isCDNA() \|\| VEGA` 走 DPP 广播 |
| `supportsDirectToLdsLoadBitWidth` | `VEGA20 → 32` 可用 | **未采用**（走 default `false`，与生产验证的 3.4 补丁一致，保守） |
| SFU/`s_nop` hazard | 未见相关改动 | **本步新增**（nlzy 未覆盖，3.5 是否同样踩雷未验证） |

结论：两个最关键成败点（warp 64 + `supportsVDot`）一致，佐证路线正确；本步保留"VEGA 独立"的保守结构，
并额外修掉了两边都没有的 SFU hazard。

### 3.9 未触碰项自检

- 生产容器 `qwen38-vllm`：全程仅 `docker ps` / `docker stats` 观察，未 exec、未改动；
- DTK 的 triton/RCCL/rocBLAS/hipBLASLt/MIOpen/`/opt/hyhal`：未替换（§3.3 有 mtime 证据）；
- 未执行任何自编译 HIP 独立可执行程序：只做了 `clang -x hip --cuda-device-only -S`（仅出汇编）、
  `llvm-objdump`、`pip install --target`、triton JIT 经 python 派发；GPU 侧全程无挂死、无异常退出；
- `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`、`vllm/**` 均未修改。

---

## 4. 门禁判定

| 门禁（原文） | 判定 | 证据指向 |
|---|---|---|
| 1. `python tools/dcu/probe_triton.py` → `TRITON_OK`（误差 < 1e-5） | ✅ **通过** | §3.4：`max_abs_err = 0.0` / `TRITON_OK` / `PROBE_RC = 0`；附加 `REDUCE_OK`(0.0)、`INT8DOT_OK`、`v_dot occurrences = 32` |
| 2. `test_triton_unified_attention_gfx906.py` 通过 | ✅ **通过** | §3.6：`39 passed`，RC=0（canonical 产物复跑 `39 passed`） |
| 2. `test_triton_unified_attn_gfx906_v1.py` 通过 | ✅ **通过** | §3.6：`12 passed`，RC=0 |
| 3. 8K 上下文服务日志出现 `TRITON_ATTN` | ⏸️ **未执行（属 S8 联测）** | 本步只到内核级门禁；S8 用本 triton 起服务时复核 |
| 4. 记录 `triton.__version__`/源码 commit 与构建日志路径 | ✅ **通过** | §3.0/§3.2：`3.6.0+git7c56a5e4`、源码 `7c56a5e (tag v3.6.0)`、`/mnt/game/triton-gfx906/logs/build-3.6.0.log`、`logs/rebuild-final.log` |
| 追加：绝不覆盖/替换 DTK triton；产物只落 `/mnt/game` | ✅ **通过** | §3.3：`--target` 安装，镜像 triton mtime/尺寸未变；源码/构建树/前缀/缓存全在 `/mnt/game/triton-gfx906/` |
| 追加：用新建容器 `z100-triton`，不在基线容器里做 triton 构建 | ✅ **通过** | §2.3；`z100-build`/`z100-build-211` 仅用于跑测试（未做源码构建） |
| 追加：统计 S5 的 14 个 `unsupported target` 消除数 | ✅ **通过** | §3.6：**14/14 消除**；五文件同口径 `66 passed/16 failed` → `74 passed/0 failed` |
| 追加：补丁/探针/构建脚本固化 | ✅ **通过** | §2.1 三个文件已入库（`tools/dcu/`）；补丁对 pristine v3.6.0 `git apply --check` 通过（§3.1） |
| 硬约束：不执行自编译 HIP 可执行程序 | ✅ **遵守** | §3.9；仅编译/反汇编/python 派发 |
| 硬约束：不碰生产容器、不替换 DTK 库、不改禁改文档 | ✅ **遵守** | §3.9 |

**S6 总体：✅ 门禁 1、门禁 2 全过；14/14 `unsupported target` 失败消除；gfx906 triton 已固化可复用。**
门禁 3 属 S8 联测范围，本步不适用。

---

## 5. 未完成项 / 阻塞与原因

### 5.1 已定界的"完成"边界

- 本步交付的是**内核级可用性**（编译 + 数值正确）与固化产物；**端到端服务**（`TRITON_ATTN` 生效、
  8K 上下文、吞吐/精度）未做，属 S8/S9。探针与 attention 测试只用 4 卡中的 0 号卡（默认）与生产容器共享 GPU，
  未观测到相互影响。

### 5.2 已排除的伪阻塞（记录下来避免后续重复踩坑）

- **"补丁没生效"的假象**：`triton.runtime.build.compile_module_from_src` 的缓存键是
  `sha256(src + platform_key)`（`python/triton/runtime/build.py:80`），**不含 `include_dirs`**；
  JIT 磁盘缓存也不含后端 `compiler.py` 的哈希。改这两处后必须清 `TRITON_CACHE_DIR`，
  否则会复用旧的 `hip_utils.so` / `.hsaco`（本次因此白跑了两轮实验）。
- **`.s` 与 `.hsaco` 的 `v_and_b32_e32 v0, 0xf0/48, v0` 差异**：经 prebuilt LLVM 21 与 DTK LLVM 18 两套
  `llvm-objdump` 交叉核对，二者反汇编一致，且两种掩码对 64 条 lane 行为等价（`lane & 0xF0 == 0` 与
  `lane & 0x30 == 0` 都只放行 lane 0–15），**不是**本缺陷来源。

### 5.3 环境缺口（与本步无关，但影响测试口径）

- `tests/model_executor/test_qwen3_5_gdn.py` 在 `z100-build-211` **采集期即失败**：

```console
E   RuntimeError: operator torchvision::nms does not exist
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

  原因是该容器 `torchvision 0.25.0+das.opt1.dtk2604.torch2100.20260325`（**为 torch 2.10 构建**）
  与 torch **2.11.0+das.opt1.dtk2604** 不匹配；用镜像自带 das triton（不挂本步产物）复跑同样报错，
  **与 S6/triton 无关**。S5 当年的 2 个替身失败（`seq_lens`、`ignore_unexpected_prefixes`）
  因此在本步未复评，仍挂在 S5 §5.3 名下。

### 5.4 SFU 修复的性能代价与残余风险

- 代价：每条 SFU 指令前多一个 `s_nop 2`（4 字节、2 周期）。SFU 在 attention softmax 里不是瓶颈，
  未做吞吐对比（本步只验正确性）；若 S8 发现性能回退，可把 `nops=2` 降到能通过的最小值
  （本步实测 2 可过，1 未单独验证）。
- 残余风险：该 hazard 的**硬件/驱动根因未定性**（是 Z100 的 SIMD16 流水特性还是 DTK 运行时行为），
  本步只给出"插等待槽即消除"的经验证据。若后续 DTK 升级改变了 SFU 时序，该补丁应重新评估
  （判定脚本：`/mnt/game/triton-gfx906/repro4.py`、`repro9.py`，一眼可判）。
- `supportsDirectToLdsLoadBitWidth` 未给 VEGA 开 32 位（nlzy 有）；本步保守未采用，
  若 S8 出现 direct-to-lds 相关失败可从此处入手。

---

## 6. 对后续步骤的影响

### 6.1 S7/S8/S9 怎么用这个 triton

```bash
# 任何需要 gfx906 triton 的步骤：把安装前缀前置到 PYTHONPATH（不要 pip install 到系统）
export PYTHONPATH=/mnt/game/triton-gfx906/install:$PYTHONPATH
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache     # 必须在 /mnt/game，别写 /（根分区 ~44 GB）
# 版本核对（不要用 triton.__version__ 的裸版号）
python -c "import importlib.metadata as m, triton; print(m.version('triton'), triton.__file__)"
# 期望：3.6.0+git7c56a5e4 /mnt/game/triton-gfx906/install/triton/__init__.py
```

- **S8（attention 路径）**：现在可以真正走 `TRITON_ATTN`。S5 摘除的 `_rocm_C::paged_attention` /
  `wvSplitK*`（MFMA-only 桩）**不需要**回退，attention 全部交给本 triton；gfx906 分支的
  `_decode_num_warps/_decode_block_m/_decode_num_segments` 三个调度旋钮已在门禁 2 覆盖
  （wide-head、GQA6 BLOCK_M=8、split-KV 段数）。
- **S9（固化/复现）**：把 `tools/dcu/triton-gfx906-3.6.0.patch` + `build_triton_gfx906.sh` 纳入固化清单；
  容器 `z100-triton` 是可复现的构建环境（`--memory=10g`、`MAX_JOBS=4`、37 分钟）。
  **回滚方案**：删 `PYTHONPATH` 前置即回到镜像 das triton 3.4.0（但会失去 gfx906 出码能力，
  等价于回滚 S6 全部成果）。
- **S7（其他 Triton 资产，如 GDN/量化）**：直接受益于同一次修复；凡是含
  `exp/log/sqrt/rsqrt/ldexp` 的 triton 内核，**必须**在带 `s_nop` 的产物上验证，否则可能悄悄数值错。

### 6.2 被本步推翻/更新的前提

1. **推翻**"VEGA 补丁 + 源码重编就够了"（S4b §6.2 的入口判断）：还需要 ①DTK HIP 头结构体（driver.py）
   ②SFU 等待槽（compiler.py）。两条都写进补丁。
2. **推翻**"用 DAS 轮子绕开"的残余可能性：即便解决 ABI，das 轮子也没有 VEGA ISAFamily，
   仍会在 `ConvertTritonAMDGPUToLLVM` 报 `unsupported target`（S4b 已证，本步未改变该结论）。
3. **新增全局风险**：VEGA 路线的 triton 在 gfx906 上对"VALU→SFU"序列系统性出错
   （3.4/3.6 均复现）。**任何用 3.4 老补丁的既有产线/镜像都需要重新评估**——
   包括 `/home/lyh/server/gfx906_port` 描述的"生产已验证"环境；建议把 `compiler.py` 的
   `insert_sfu_waits` 视为 gfx906 triton 的必需项。

---

## 7. 复现方式

```bash
# 0) 前置：宿主 docker 可用，镜像已存在
#    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

# 1) 一键构建（建容器 → clone v3.6.0 → 打补丁 → 装到 /mnt/game；MAX_JOBS 可调）
cd /home/lyh/z100-vllm
bash tools/dcu/build_triton_gfx906.sh            # MAX_JOBS=4 时约 37 分钟（z100-triton）

# 2) 门禁 1：最小探针（容器内）
docker exec z100-triton bash -lc '
  source /opt/dtk/env.sh >/dev/null 2>&1
  . /src/tools/dcu/env.sh >/dev/null 2>&1
  export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
  export PYTHONPATH=/mnt/game/triton-gfx906/install
  python /src/tools/dcu/probe_triton.py'
# 期望：target=GPUTarget(backend='hip', arch='gfx906', warp_size=64) / TRITON_OK / PROBE_RC = 0

# 3) 门禁 2：两个 gfx906 attention 测试（用 S4b 的 2.11 fork 副本；本步另存了一份 gfx906-vllm-triton）
docker exec z100-build-211 bash -lc '
  source /opt/dtk/env.sh >/dev/null 2>&1
  . /mnt/game/port/gfx906-vllm-triton/tools/dcu/env.sh >/dev/null 2>&1
  cd /mnt/game/port/gfx906-vllm-triton
  export PYTHONPATH=/mnt/game/triton-gfx906/install:/mnt/game/port/gfx906-vllm-triton
  export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
  for f in tests/kernels/attention/test_triton_unified_attention_gfx906.py \
           tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py \
           tests/kernels/mamba/test_fused_recurrent_packed_decode.py; do
    python -c "import torch, pytest, sys; sys.exit(pytest.main([\"-v\",\"--no-header\",\"--noconftest\",\"-p\",\"no:cacheprovider\",\"$f\"]))"
  done'
# 期望：39 passed / 12 passed / 5 passed，RC 全 0

# 4) 若要复核 SFU hazard（改回旧行为或换 triton 时用它一眼判定；OK=正常，出现 [8,9,10,11] 即中招）
docker exec z100-build-211 bash -lc '
  source /opt/dtk/env.sh >/dev/null 2>&1
  . /mnt/game/port/gfx906-vllm-triton/tools/dcu/env.sh >/dev/null 2>&1
  export PYTHONPATH=/mnt/game/triton-gfx906/install
  export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache-check
  python /mnt/game/triton-gfx906/repro9.py'

# 5) 版本/产物核对
docker exec z100-triton bash -lc 'ls -la /mnt/game/triton-gfx906/install/triton/_C/libtriton.so;
  ls -la /usr/local/lib/python3.10/dist-packages/triton/_C/libtriton.so'   # 后者必须保持 870759392 B / Apr 2
```

---

## 8. 参考

- 设计文档：[docs/design/z100-06-triton-gfx906.md](../design/z100-06-triton-gfx906.md)
- 前序归档：[z100-04b-stack-upgrade-result.md](z100-04b-stack-upgrade-result.md) §8（入口判断）、
  [z100-05-csrc-kernels-result.md](z100-05-csrc-kernels-result.md) §4/§5（14 个失败）
- 上一轮工程师的 gfx906 资产（只读）：
  `/home/lyh/server/gfx906_port/artifacts/triton-gfx906.patch`（3.4 的 3 文件 VEGA 补丁，本步移植到 3.6.0）、
  `artifacts/triton_build.log`（3.4 构建配方：`pip install . --no-build-isolation --no-deps`、MAX_JOBS=6、14 分钟、277 MB 轮子）
- 上游源码：`https://github.com/triton-lang/triton` tag `v3.6.0`（commit `7c56a5e`，LLVM hash `f6ded0be…`）
- 参考实现（只作对照）：`https://github.com/nlzy/triton-gfx906` 分支 `gfx906/v3.5.x`（`VEGA20` + 折进 `isCDNA()`）
- 关键源码位置（3.6.0）：
  - `third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp:114`（`unsupported target` 抛错点）
  - `third_party/amd/lib/TritonAMDGPUToLLVM/TargetUtils.cpp`（`deduceISAFamily` / `supportsVDot`）
  - `third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp`（`getWarpSize` / `warpReduce`）
  - `third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp:37`（`supportsVDot` → `fdot2`/`sdot4`）
  - `third_party/amd/backend/driver.py`（`hip_utils` 用 `driver.c` 读 `hipDeviceProp_t`）
  - `python/triton/runtime/build.py:80`（`compile_module_from_src` 缓存键不含 include_dirs）
- 原始日志/脚本（全部在 `/mnt/game/triton-gfx906/`）：`logs/build-3.6.0.log`、`logs/rebuild-final.log`、
  `logs/probe-triton-3.6.0.log`、`logs/s6-attn-tests-sfufix.log`、`logs/s6-gate2-final.log`、
  `repro4.py`、`repro9.py`、`diag_attn.py`、`diag_attn_knobs.py`
