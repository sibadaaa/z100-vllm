# S4b 目标栈升级（torch 2.11.0+das / triton 3.6.0+das）与在新栈上重建 — 执行结果

> 归档时间：2026-10-08
> 执行者：子 agent（S4b 目标栈升级 + 新栈重建）
> 对应设计文档：docs/design/z100-04b-stack-upgrade.md
> 前一档：[z100-04-build-system-dtk-result.md](z100-04-build-system-dtk-result.md)（S4）、[z100-05-csrc-kernels-result.md](z100-05-csrc-kernels-result.md)（S5）
> 状态：✅ 完成（S4b 门禁全过；升级的主要动因——`cuda_view.cu` 的 `torch::stable::Tensor::layout()`——已解决；
> 且 `_C_stable_libtorch` / `_moe_C_stable_libtorch` **在本步内也编成了**，代价是两项**构建树本地**修复，仓库侧待落地）

**一句话结论**：把构建基线从镜像自带的 `torch 2.10.0+das / triton 3.4.0` 换成目标栈
`torch 2.11.0+das.opt1.dtk2604 / triton 3.6.0+das185.dtk2604` 后，**四个扩展全部在新栈上从零编译成功且只含 `gfx906`**
（`_C` 78.1 MB、`_rocm_C` 35.2 MB、`_C_stable_libtorch` 40.7 MB、`_moe_C_stable_libtorch` 29.7 MB），
门禁全过、S5 的 2 项 gfx906 内核回归 **13/13 + 5/5 无退化**。
S5 卡住 `_C_stable_libtorch` 的 `cuda_view.cu` 的 `layout()` 问题**已消失**（升级动因达成）；
随后撞上的两个新阻塞都已定位并**在构建树里验证修好**：
① **仅 torch 2.11 才有的 DTK 打包缺陷**——hipify 的 `CUDA_MACROS_MAP` 值写成裸字符串，导致 `CUDA_VERSION` 被改写成 `D`；
② S5 的 `__syncwarp` 修复**漏做了 stable 孪生文件**。两项修复合计约 5 行，**都还没进仓库**（本步约定零源码改动），
需由后续步骤落地后重跑才能复现这两个 `.so`。
另外：`triton 3.6.0+das185` 仍**不能**为 `gfx906` 出码——S5 的 `unsupported target: 'gfx906'` 没有消失，
只是从 3.4 的 TTGIR 阶段挪到了 3.6 的 LLVM lowering 阶段；S6 仍必须走 VEGA 补丁重编路线。

---

## 1. 目标与验收标准

### 1.1 设计文档门禁原文（`docs/design/z100-04b-stack-upgrade.md` §验收）

- `python -c "import torch;print(torch.__version__)"` = `2.11.0+das.opt1.dtk2604`；`triton.__version__` = `3.6.0...`。
- `python -c "import torch, vllm, vllm._C, vllm._rocm_C; print('IMPORT_OK')"` 成功（**必须先 `import torch`**）。
- 新 `.so` 只含 gfx906：`strings <so> | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sort -u`。
- `from vllm._custom_ops import ROCM_SKINNY_GEMM_ENABLED` 可导入且为 `False`。
- 回归（新栈上仍全过）：`tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py` **13/13**、
  `tests/kernels/mamba/test_fused_sigmoid_gating.py` **5/5**（用 `python -c "import torch, pytest; ..."` 起跑，别直接 `pytest`）。
- 顺带判定：新 triton 上 `unsupported target: 'gfx906'` 是否消失。

### 1.2 本次执行追加的验收标准（来自任务书）

- 只装两枚目标轮子（`pip install --no-deps`），**不装** `requirements/dcu-build.txt` 的 common 依赖。
- 全量 rsync（排除项锚定）；新构建目录 `/mnt/game/port/build-211`；`PYTORCH_ROCM_ARCH=gfx906`、`MAX_JOBS<=4`。
- **先只编 `_C _rocm_C`**（不裸跑全量 ninja），再单独试 `_C_stable_libtorch _moe_C_stable_libtorch`，给出 `cuda_view` 结论。
- 给 S6 的入口判断：新 triton 上最小 gfx906 triton 内核能否编译并跑通。
- 硬约束：不执行自编译 HIP 可执行程序；不碰 `qwen38-vllm`；**不删除/不升级 `z100-build`**；
  构建产物只落 `/mnt/game`；仓库 `git status` 不得出现 `build/`/`.so`/`egg-info`。

---

## 2. 实际改动

### 2.1 仓库源码改动

**无。** 本步对 `/home/lyh/z100-vllm` 的源码**零改动**（`git status --short` 与 S5 结束时一致，
见 §3.10 自检）；本次唯一的仓库写入是本归档文件本身。

### 2.2 在 `/mnt/game` 下新建的资源（宿主机对 `/mnt/game` 只读，全部经 `docker exec` 写入）

| 资源 | 路径 | 说明 |
|---|---|---|
| 构建容器 | `z100-build-211` | 同镜像；设备/group/`--ipc host`/`--shm-size` 照抄 `z100-build`；`--memory=8g --memory-swap=8g`；**未加 `--rm`** |
| 构建副本 | `/mnt/game/port/gfx906-vllm-211` | 由 `/src`（工作区，只读挂载）全量 `rsync` 而来 |
| 持久 CMake 目录 | `/mnt/game/port/build-211/temp` | 增量、可续跑 |
| 目标轮子 | `/mnt/game/port/wheels-211/` | torch 2.11.0+das.opt1.dtk2604（新下载）、triton 3.6.0+das185（复用 `wheels/` 下同一文件，字节数一致） |
| 环境脚本 | `/mnt/game/port/setenv-211.sh` | `REPO_DIR`/`BUILD_DIR` 指向 211 副本与 `build-211`，其余同 S4 |
| 配置脚本 | `/mnt/game/port/configure-211.sh` | 与 `setup.py:313-379` 同参数的手工 configure（见 §3.3 说明） |
| 构建 / 安装脚本 | `/mnt/game/port/build-211.sh`、`install-211.sh` | `cmake --build <dir> -j N --target=<显式目标>`，永不裸跑 ninja |
| 门禁脚本 | `/mnt/game/port/gate-211.sh` | 版本 + import + arch + `ROCM_SKINNY_GEMM_ENABLED` + `_rocm_C` 注册面 |
| triton 探针 | `/mnt/game/port/probe-triton-211-add.py`、`probe-triton-211-hcu.py` | S6 入口判断（§3.8） |
| hipify 修复验证（**只在构建树内，不入仓库**） | `/mnt/game/port/hipify-fix-211.py` + `build-211/temp/build.ninja` 的 hipify COMMAND 改写 | 见 §3.7.3 |
| 日志 | `/mnt/game/port/{pip-install-211,configure-211,build-211,build-211-stable,build-211-stable-fixed,build-211-stable-fixed2,gate-211,probe-triton-211-*,hipify-fix-211}.log` | 本档引用的原文 |

### 2.3 容器实际配置（`docker inspect z100-build-211` 原文）

```console
$ docker run -d --name z100-build-211 \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    --group-add video --security-opt label=disable --ipc host \
    --memory=8g --memory-swap=8g \
    -v /opt/hyhal:/opt/hyhal:ro \
    -v /home/lyh/z100-vllm:/src:ro \
    -v /mnt/game:/mnt/game \
    -w /mnt/game/port \
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 \
    bash -c 'sleep infinity'
03531ea02896a95b1782f8deaa4411b08e07b8c8652e37e33c2f8e8aa4ed28eb
$ docker inspect z100-build-211 --format '...'
ImageID=sha256:d1c08bf30382f6d8f49e0ad7130ceb4e3a31484ce4bdd3b1aeb7787b5d2fb14e
Devices=[{"PathOnHost":"/dev/kfd",...},{"PathOnHost":"/dev/mkfd",...},{"PathOnHost":"/dev/dri","CgroupPermissions":"rwm"}]
GroupAdd=["video"]  SecurityOpt=["label=disable"]  IpcMode=host  ShmSize=67108864
Memory=8589934592  MemorySwap=8589934592  AutoRemove=false  WorkingDir=/mnt/game/port
Mounts=/home/lyh/z100-vllm:/src:ro /mnt/game:/mnt/game:rw /opt/hyhal:/opt/hyhal:ro
```

> `z100-build`（2.10 基线容器）**未动**：其两份 `.so` 的大小/时间戳/校验和见 §3.10。

---

## 3. 执行过程与原始证据

### 3.1 装轮子：只装两枚目标栈轮子（`--no-deps`）

先记录新容器里的**升级前**基线：

```console
$ docker exec z100-build-211 bash -lc 'pip show torch triton | grep -E "^(Name|Version)"'
Name: torch   Version: 2.10.0+das.opt1.dtk2604.20260325.g6b060a
Name: triton  Version: 3.4.0+git1ef59765
# 注意：不 source DTK env 时 `import torch` 会 ImportError: librocm_smi64.so.2（已知，见 §3.2）
```

torch 轮子本机没有，按 `requirements/dcu.txt` 的直链下载（`Content-Length: 1215468925`，落盘字节数完全一致）：

```console
$ docker exec z100-build-211 bash -lc 'curl -sL -o /mnt/game/port/wheels-211/torch-2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7-cp310-cp310-manylinux_2_28_x86_64.whl \
    "https://download.sourcefind.cn:65024/directlink/4/pytorch/DAS1.8.5/torch-2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7-cp310-cp310-manylinux_2_28_x86_64.whl"'
$ ls -l /mnt/game/port/wheels-211/
-rw-r--r-- 1 root root 1215468925 torch-2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7-cp310-cp310-manylinux_2_28_x86_64.whl
-rw-rw-r-- 1 1000 1000  120053194 triton-3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
$ md5sum /mnt/game/port/wheels-211/*.whl
21de29c8788359f3fe4f0d89f0f0047c  .../torch-2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7-...whl
a929395abc11c0554cc372f4a485aa61  .../triton-3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1-...whl
```

> triton 那枚与 `https://download.sourcefind.cn:65024/directlink/4/triton/DAS1.8.5/` 的 `Content-Length: 120053194`
> **逐字节一致**，故直接复用 `/mnt/game/port/wheels/` 里的既有副本（S4/S5 已下好）。

安装（**只这两枚，`--no-deps`**；输出全文见 `/mnt/game/port/pip-install-211.log`，下面是尾部原文）：

```console
$ docker exec z100-build-211 bash -lc 'cd /mnt/game/port/wheels-211 && pip install --no-deps ./torch-2.11.0+das.opt1.dtk2604.*.whl ./triton-3.6.0+das185.dtk2604.torch2100.*.whl'
Processing ./torch-2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7-cp310-cp310-manylinux_2_28_x86_64.whl
Processing ./triton-3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
Installing collected packages: triton, torch
  Attempting uninstall: triton
    Found existing installation: triton 3.4.0+git1ef59765
    Successfully uninstalled triton-3.4.0+git1ef59765
  Attempting uninstall: torch
    Found existing installation: torch 2.10.0+das.opt1.dtk2604.20260325.g6b060a
    Successfully uninstalled torch-2.10.0+das.opt1.dtk2604.20260325.g6b060a
Successfully installed torch-2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7 triton-3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1
pip_rc=0
```

**除这两枚轮子外，本步没有安装任何其它 pip 包**（包括没有装 `requirements/dcu-build.txt` 的 common 依赖）。
两枚回归测试全程只依赖镜像已有包 + torch/triton，**没有触发任何补装**（S5 在 2.10 容器里补装的
`tblib/mistral_common/openai/xgrammar` 在新容器里**未安装也不需要**，因为两个测试都用 `--noconftest` 跑，见 §3.6）。

### 3.2 升级后立刻验证（`import torch` 必须带 DTK env）

```console
$ docker exec z100-build-211 bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; python -c "
import torch, triton
print(\"torch.__version__      =\", torch.__version__)
print(\"torch.version.hip      =\", torch.version.hip)
print(\"triton.__version__     =\", triton.__version__)
print(\"cuda available         =\", torch.cuda.is_available())
print(\"device_count           =\", torch.cuda.device_count())
print(\"device name            =\", torch.cuda.get_device_name(0))
print(\"gcnArchName            =\", torch.cuda.get_device_properties(0).gcnArchName)
"'
torch.__version__      = 2.11.0
torch.version.hip      = 6.3.26113
triton.__version__     = 3.5.0
cuda available         = True
device_count           = 4
device name            = Z100L
gcnArchName            = gfx906:sramecc+:xnack-
rc=0
```

> **门禁口径修正（与 S4 §5.5 的同一类问题）**：DTK 的 torch 构建把**发行版本号**的 local tag
> 从 `torch/version.py` 里摘掉了，`torch.__version__`/`triton.__version__` 只报**基础版本**
> （2.11.0 / 3.5.0），这不是本步引入的偏差——**2.10 基线容器行为完全相同**
> （`torch.__version__ = 2.10.0`，而 `pip show torch` = `2.10.0+das.opt1.dtk2604.20260325.g6b060a`）。
> 精确判据有三条，本次全部取到原文：

```console
$ python -c "import importlib.metadata as m; print(m.version('torch'), m.version('triton'))"
2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7 3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1
$ python -c "import torch; print(torch.version.__hcu_version__, torch.version.hip, torch.version.rocm)"
2.11.0+das.opt1.dtk2604 6.3.26113 6.3.3
# 2.10 基线对照（z100-build，只读）
baseline z100-build: 2.10.0 6.3.26093 2.10.0+das.opt1.dtk2604.20260325.g6b060a
baseline triton:     3.4.0              3.4.0+git1ef59765
```

`torch/version.py` 原文（`git_version` 与轮子文件名里的 `gfb4aa7` 对得上）：

```python
__version__ = '2.11.0'
git_version = 'fb4aa77f70ff922133bad8bbb8649f3d953190b6'
hip: Optional[str] = '6.3.26113'
__hcu_version__ = '2.11.0+das.opt1.dtk2604'
rocm: Optional[str] = '6.3.3'
```

⇒ **torch 2.11 与 DTK 26.04 兼容**：HIP 从 `6.3.26093` 到 `6.3.26113` 属同一 HIP 6.3 系列，
4 卡全部可枚举（`Z100L`, `gfx906:sramecc+:xnack-`）。**未触发回退条件。**

### 3.3 全量 rsync + configure（架构固定 gfx906）

```console
$ docker exec z100-build-211 bash -lc 'rsync -a --delete --exclude=/build --exclude=/.deps \
    --exclude=/.tmp-bf16 --exclude="/*.egg-info" --exclude=/.venv /src/ /mnt/game/port/gfx906-vllm-211/'
rsync rc=0
$ du -sh /mnt/game/port/gfx906-vllm-211
265M	/mnt/game/port/gfx906-vllm-211
$ ls /mnt/game/port/gfx906-vllm-211/requirements/build/
cpu.txt cuda.txt dcu.txt rock.txt rocm.txt rust.txt tpu.txt      # 锚定排除生效（--exclude=/build）
```

configure 用与 `setup.py:313-379` 完全同参数的手工 `cmake`（写成 `/mnt/game/port/configure-211.sh`，
理由：`setup.py build_ext` 一次性把**所有** extension 目标都加进 `cmake --build --target=...`，
无法只编 `_C _rocm_C`；参数逐条对齐 S4 §3.7 记录的 setup.py 实际命令行）：

```console
$ docker exec z100-build-211 bash /mnt/game/port/configure-211.sh
cmake /mnt/game/port/gfx906-vllm-211 -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_TARGET_DEVICE=rocm -DVLLM_PYTHON_EXECUTABLE=/usr/bin/python \
  -DVLLM_PYTHON_PATH=/mnt/game/port/gfx906-vllm-211:/usr/local:... \
  -DFETCHCONTENT_BASE_DIR=/mnt/game/port/gfx906-vllm-211/.deps \
  -DCMAKE_JOB_POOL_COMPILE:STRING=compile -DCMAKE_JOB_POOLS:STRING=compile=3 -DROCM_PATH=/opt/dtk
CONFIGURE_RC=0
```

configure 关键行（`/mnt/game/port/configure-211.log` 原文）：

```
13:Building PyTorch for GPU arch: gfx906
88:  Pytorch version >= 2.13.0 expected for ROCm build, saw 2.11.0 instead.      <- 上游版本告警，不阻塞
91:-- HIP supported arches: gfx906;gfx908;gfx90a;gfx942;...;gfx1201
92:-- DTK rocPRIM placement-new workaround enabled: -include /mnt/game/port/gfx906-vllm-211/csrc/dtk_rocprim_compat.h
99:-- Only gfx1250/gfx906, skipping skinny_gemms
100:-- [triton_kernels] Fetch from /mnt/game/port/triton-kernels/triton_kernels
101:-- [triton_kernels] triton_kernels is available at /mnt/game/port/triton-kernels/triton_kernels/
```

⇒ S5 的 DTK shim 在 2.11 上同样被注入（`ROCM_PATH=/opt/dtk` 命中 `MATCHES "dtk"`）；
架构被固定为 **gfx906**；`triton_kernels` 走离线源码，configure 无网络 clone。

### 3.4 只编 `_C _rocm_C`（不裸跑全量 ninja）

```console
$ docker exec -d z100-build-211 bash -lc 'MAX_JOBS=3 bash /mnt/game/port/build-211.sh _C _rocm_C'
$ grep -E "^\[2[0-9]/29\]|^FAILED" /mnt/game/port/build-211.log
[25/29] Building CXX object CMakeFiles/_C.dir/csrc/torch_bindings.cpp.o
[26/29] Linking HIP shared module _C.abi3.so
[27/29] Building CXX object CMakeFiles/_rocm_C.dir/csrc/rocm/torch_bindings.cpp.o
[28/29] Building HIP object CMakeFiles/_rocm_C.dir/csrc/rocm/attention.hip.o
[29/29] Linking HIP shared module _rocm_C.abi3.so
$ grep -c "error:" /mnt/game/port/build-211.log
0
$ stat -c "%n %s %y" /mnt/game/port/build-211/temp/*.so
/mnt/game/port/build-211/temp/_C.abi3.so        78102728  2026-10-08 07:04:52
/mnt/game/port/build-211/temp/_rocm_C.abi3.so   35156368  2026-10-08 07:06:08
```

- configure 结束于 `06:53:48`，`_C` 链接完成 `07:04:52`，`_rocm_C` 链接完成 `07:06:08`
  ⇒ 从零编译 **26 个 TU 约 12 分钟**（`-j3`，容器峰值内存 `2.9 GiB / 8 GiB`，`memory.events` 无 OOM）。
- **全程 0 error**：S5 的 4 类 DTK 工具链修复（rocPRIM placement-new shim、`__syncwarp` 守卫、
  `atomicAdd(half)`、hipify 注入 `dtk_macros.h`）在 torch 2.11 上同样有效，**无需新增修复**。
- 与 2.10 基线的体积对照：`_C` 78,102,728 B（2.10 为 78,034,896 B）、
  `_rocm_C` 35,156,368 B（2.10 为 35,130,920 B）——差异来自新 torch 头文件，属预期。

安装到副本（`--inplace` 等价操作）：

```console
$ docker exec z100-build-211 bash /mnt/game/port/install-211.sh _C _rocm_C
-- Installing: /mnt/game/port/gfx906-vllm-211/vllm/_C.abi3.so
-- Installing: /mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so
-rw-r--r-- 1 root root 78102728 .../vllm/_C.abi3.so
-rw-r--r-- 1 root root 35156368 .../vllm/_rocm_C.abi3.so
$ md5sum .../vllm/_C.abi3.so .../vllm/_rocm_C.abi3.so
6fba05966e153b3d04771d6960c782bb  .../vllm/_C.abi3.so
5d1e50e1a77de74f2d308519f8120745  .../vllm/_rocm_C.abi3.so
```

### 3.5 S4b 门禁原始输出（`docker exec z100-build-211 bash /mnt/game/port/gate-211.sh`）

```console
===== S4b gate 2026-10-08T07:09:53+08:00 =====
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
WARNING 10-08 07:10:08 [rocm.py:70] Failed to import from vllm._C_stable_libtorch with ModuleNotFoundError("No module named 'vllm._C_stable_libtorch'")
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
===== gate done 2026-10-08T07:10:34+08:00 =====
GATE_RC=0
```

逐条原文命令（可直接粘贴）：

```console
$ docker exec z100-build-211 bash -lc 'source /mnt/game/port/setenv-211.sh >/dev/null 2>&1; cd "$REPO_DIR"; \
    python -c "import torch, vllm, vllm._C, vllm._rocm_C; print(\"IMPORT_OK\")"'
IMPORT_OK
$ docker exec z100-build-211 bash -lc 'for so in /mnt/game/port/gfx906-vllm-211/vllm/_C.abi3.so /mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so; do \
    echo -n "$(basename $so): "; strings -a $so | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sed "s/.*--//" | sort -u | tr "\n" " "; echo; done'
_C.abi3.so: gfx906
_rocm_C.abi3.so: gfx906
$ docker exec z100-build-211 bash -lc 'source /mnt/game/port/setenv-211.sh >/dev/null 2>&1; cd "$REPO_DIR"; \
    python -c "import torch, vllm._rocm_C; from vllm._custom_ops import ROCM_SKINNY_GEMM_ENABLED; print(ROCM_SKINNY_GEMM_ENABLED)"'
False
```

- `_rocm_C` 注册面与 S5 一致：`['_rocm_C::paged_attention']`（gfx906 上 `LLMM1/wvSplitK*` 被**有意**排除）。
- 「先 `import vllm`」那条（§2b）在新栈上**仍然是 `std::bad_alloc` → SIGABRT(134)**，
  与 todolist 硬约束 8 及 S4 §5.3 完全一致：**不是升级引入的回归**，所有门禁按"先 import torch"口径执行。

### 3.6 回归：2 个 S5 基线 gfx906 内核测试（新栈）

跑法与 S5 完全一致（`python -c "import torch, pytest, sys; ..."` + `--noconftest`，避免根 conftest 拖入应用链）：

```console
$ docker exec z100-build-211 bash -lc 'source /mnt/game/port/setenv-211.sh >/dev/null 2>&1; cd "$REPO_DIR"; \
    python -c "import torch, pytest, sys; sys.exit(pytest.main([\"-v\",\"--no-header\",\"--noconftest\",\"-p\",\"no:cacheprovider\",\"tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py\"]))"'
...
tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py::test_q6_k_fixed_cols_matches_dequantized_reference PASSED [100%]
======================= 13 passed, 3 warnings in 29.42s ========================
gguf_rc=0

$ docker exec z100-build-211 bash -lc 'source /mnt/game/port/setenv-211.sh >/dev/null 2>&1; cd "$REPO_DIR"; \
    python -c "import torch, pytest, sys; sys.exit(pytest.main([\"-v\",\"--no-header\",\"--noconftest\",\"-p\",\"no:cacheprovider\",\"tests/kernels/mamba/test_fused_sigmoid_gating.py\"]))"'
tests/kernels/mamba/test_fused_sigmoid_gating.py::test_gfx906_packed_decode_matches_indexed_decode PASSED [100%]
======================== 5 passed, 1 warning in 14.74s =========================
mamba_rc=0
```

⇒ **13/13 + 5/5，与 S5 在 2.10 上的基线完全一致**（无回归）。测试用的 HF 样本
`Isotr0py/test-gguf-sample`（284 MB）从 2.10 容器缓存拷到 `/mnt/game/port/hf-cache/` 再放进新容器，
避免联网不确定性；日志里没有出现任何"补装依赖"。

**四扩展齐备后复跑（防止 `_C_stable_libtorch` 一旦可导入会切换代码路径）**：

```console
$ （四个 .so 全部就位后，同样的命令再跑一遍）
########## tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py
13 passed, 3 warnings in 25.39s        ##### rc=0
########## tests/kernels/mamba/test_fused_sigmoid_gating.py
5 passed, 1 warning in 14.44s          ##### rc=0
```

⇒ 引入 `_C_stable_libtorch`/`_moe_C_stable_libtorch` 后两个回归**仍然全过**。

### 3.7 `_C_stable_libtorch` / `_moe_C_stable_libtorch`（升级的主要动因）

#### 3.7.1 结论：`cuda_view` 的 `layout()` 阻塞**已消失**

```console
$ docker exec -d z100-build-211 bash -lc 'MAX_JOBS=3 LOG=/mnt/game/port/build-211-stable.log bash /mnt/game/port/build-211.sh _C_stable_libtorch _moe_C_stable_libtorch'
$ grep -E "^\[|^FAILED" /mnt/game/port/build-211-stable.log
[1/38] Running hipify on all extension source files.
[2/38] Building HIP object CMakeFiles/_C_stable_libtorch.dir/csrc/libtorch_stable/hip_utils_kernels.hip.o
[3/38] Building HIP object CMakeFiles/_C_stable_libtorch.dir/csrc/libtorch_stable/hip_view.hip.o      <- S5 的阻塞点，现在编译通过
[4/38] Building HIP object .../csrc/libtorch_stable/activation_kernels.hip.o
FAILED: CMakeFiles/_C_stable_libtorch.dir/csrc/libtorch_stable/activation_kernels.hip.o
```

`hip_view.hip` 就是 S5 §5.1 里失败的 `cuda_view.cu` 经 hipify 后的名字（`cuda→hip` 词替换），
S5 的报错是 `hip_view.hip:30:34: error: no member named 'layout' in 'torch::stable::Tensor'`；
本次它**干净编译通过**，S5 记录的 `torch::stable::Tensor::layout()` 问题不再出现。

#### 3.7.2 但出现了一个**新的、仅 torch 2.11 才有的 DTK 工具链缺陷**：hipify 把 `CUDA_VERSION` 改写成 `D`

```console
$ grep -nE "error:" /mnt/game/port/build-211-stable.log | head -3
192:/mnt/game/port/build-211/temp/csrc/libtorch_stable/activation_kernels.hip:312:3: error: use of undeclared identifier 'D'
198:...（同一错误重复到 -ferror-limit，20 errors generated when compiling for gfx906）
# 展开链：
/mnt/game/port/build-211/temp/csrc/libtorch_stable/activation_kernels.hip:265:8: note: expanded from macro 'LAUNCH_ACTIVATION_GATE_KERNEL'
  265 |       (D >= 12090 && cc_major >= 10 && num_tokens > 128)            \
```

源码 `csrc/libtorch_stable/activation_kernels.cu:255` 写的是 `CUDA_VERSION >= 12090`。

**根因（已定位到行）**：DTK 给两个 torch 版本打的 hipify 补丁**打包不一致**。

- `torch 2.10`（基线容器 `/usr/local/lib/python3.10/dist-packages/torch/utils/hipify/cuda_to_hip_mappings.py:9555`）：
  ```python
  CUDA_MACROS_MAP = collections.OrderedDict([
      ("CUDA_VERSION", ("DTK_VERSION", API_PYTORCH)),      # 元组 ✓
      ("__CUDA_ARCH__", ("__DTK_ARCH__", API_PYTORCH)),
  ])
  ```
- `torch 2.11`（新栈，同一文件）：
  ```python
  CUDA_MACROS_MAP = collections.OrderedDict([
      ("CUDA_VERSION", "DTK_VERSION"),                     # 裸字符串 ✗
      ("__CUDA_ARCH__", "__DTK_ARCH__"),
  ])
  ```
- 消费方 `torch/utils/hipify/hipify_python.py:831-834`（两版本**完全相同**）：
  ```python
  for src, value in CUDA_MACROS_MAP.items():
      dst = value[0]        # 字符串取下标 -> 'D'
      meta_data = value[1:] # -> 'TK_VERSION'
      CUDAMACROS_MAP[src] = dst
  ```

于是在 torch 2.11 上：`CUDA_VERSION → 'D'`、`__CUDA_ARCH__ → '_'`。运行时验证：

```console
$ python -c "
import torch.utils.hipify.cuda_to_hip_mappings as m
print(dict(m.CUDA_MACROS_MAP))
for k,v in m.CUDA_MACROS_MAP.items(): print(k, '->', v[0])"
{'CUDA_VERSION': 'DTK_VERSION', '__CUDA_ARCH__': '__DTK_ARCH__'}
CUDA_VERSION -> D
__CUDA_ARCH__ -> _
```

**同一份源码在 2.10 构建树里生成的结果是对的**（对照证据，只读旧构建目录）：

```console
$ grep -n ">= 12090" /mnt/game/port/build-gfx906/temp/csrc/libtorch_stable/activation_kernels.hip
265:      (DTK_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128)            \
275:    if (DTK_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128) {         \
```

⇒ 这条**不是** vLLM 源码缺陷、也**不是** S5 遗留，而是 **DTK 26.04 的 torch 2.11 打包缺陷（第 5 类工具链缺陷）**，
只在 torch ≥ 2.11 的 DTK 轮子上出现。

#### 3.7.3 修复验证（**只在构建树里做，未改仓库任何文件**）

在 hipify 进程内把 `CUDA_MACROS_MAP` 的值补成元组，再按 ninja 的原命令跑
`cmake/hipify.py`（脚本：`/mnt/game/port/hipify-fix-211.py`；`build-211/temp/build.ninja`
里 `hipify_all` 的 `COMMAND` 加了这层 wrapper，原文件备份为 `build.ninja.s4b-orig`）：

```console
$ bash /mnt/game/port/hipify-fix-cmd-211.sh
BEFORE: {'CUDA_VERSION': 'DTK_VERSION', '__CUDA_ARCH__': '__DTK_ARCH__'}
AFTER : {'CUDA_VERSION': ('DTK_VERSION',), '__CUDA_ARCH__': ('__DTK_ARCH__',)}
  mapped 'CUDA_VERSION' -> 'DTK_VERSION'
  mapped '__CUDA_ARCH__' -> '__DTK_ARCH__'
hipify_rc=0
$ grep -rn "D >= 12090" /mnt/game/port/build-211/temp/csrc/ | head -3
（空）
$ grep -n "DTK_VERSION >= 12090" /mnt/game/port/build-211/temp/csrc/libtorch_stable/activation_kernels.hip
265:      (DTK_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128)            \
```

修复前后生成的 235 个 hipify 产物里**只有 6 个文件变化**（其余逐字节相同）：

```
csrc/type_convert_hip.cuh                                  (cuda_view 无关；_C 用到)
csrc/custom_all_gather_reduce_scatter_hip.cuh              (_C 用到)
csrc/libtorch_stable/activation_kernels.hip
csrc/libtorch_stable/cache_kernels.hip
csrc/libtorch_stable/custom_all_gather_reduce_scatter.hip
csrc/libtorch_stable/type_convert_hip.cuh
```

**已装好的 `_C`/`_rocm_C` 不受影响**（语义等价，逐处核对）：`_C` 用到的两处 header 里，
被改写的 token 都落在**必然短路**的守卫上——

- `csrc/type_convert_hip.cuh:58`：`#if defined(USE_ROCM) || (defined(DTK_VERSION) && (DTK_VERSION >= 12000))`
  （修复前是 `defined(D) && (D >= 12000)`）→ 两个版本都因 `defined(USE_ROCM)` 为真而进入同一分支；
- `csrc/custom_all_gather_reduce_scatter_hip.cuh:130/236/...`：全部是 `#if !defined(USE_ROCM) && ...`
  → 两个版本都为假。

即：`_C`/`_rocm_C` 的编译结果在修复前后**完全一致**，§3.5/§3.6 的门禁与回归证据对该 `.so` 仍然有效。
**仓库侧的正确修法**（供 S5 收尾/后续步骤决定，本步未实施）通常是在 `cmake/hipify.py` 里
调用 `hipify()` 之前加 3 行：

```python
from torch.utils.hipify import cuda_to_hip_mappings as _m
for _k, _v in list(_m.CUDA_MACROS_MAP.items()):
    if isinstance(_v, str):
        _m.CUDA_MACROS_MAP[_k] = (_v,)
```

（是否把这条落进仓库由 Lead 决定：本步任务书写明"预期对仓库源码零改动"，故未改。）

#### 3.7.4 构建树内应用两项修复后：两个 stable 目标**全部编成**

在 `build-211/temp/build.ninja` 的 hipify `COMMAND` 上挂 wrapper（§3.7.3），
并对 **构建副本**（不是仓库）的 `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu` 补上 S5 在非 stable 孪生文件里
已经加过的守卫（原文件备份为 `…/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu.s4b-orig`）：

```diff
--- /src/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu      (仓库版，未改)
+++ /mnt/game/port/gfx906-vllm-211/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu  (构建副本，已改)
@@ -43,1 +43,2 @@
-  #if defined(HIP_VERSION) && HIP_VERSION < 70000000
+  #if defined(HIP_VERSION) && HIP_VERSION < 70000000 && \
+      !defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
```
（新增行与仓库里 S5 已修好的 `csrc/fused_qknorm_rope_kernel.cu:45-46` **逐字一致**。）

续跑（`build-211-stable-fixed3.log`）：

```console
$ docker exec -d z100-build-211 bash -lc 'MAX_JOBS=3 LOG=/mnt/game/port/build-211-stable-fixed3.log \
    bash /mnt/game/port/build-211.sh _C_stable_libtorch _moe_C_stable_libtorch'
$ grep -E "^\[3[0-9]/37\]|^FAILED" /mnt/game/port/build-211-stable-fixed3.log
[36/37] Building HIP object CMakeFiles/_moe_C_stable_libtorch.dir/csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.hip.o
[37/37] Linking HIP shared module _moe_C_stable_libtorch.abi3.so
$ grep -c "error:" /mnt/game/port/build-211-stable-fixed3.log ; grep -c "build stopped" /mnt/game/port/build-211-stable-fixed3.log
0
0
$ ls -l /mnt/game/port/build-211/temp/{_C_stable_libtorch,_moe_C_stable_libtorch}.abi3.so
-rwxr-xr-x 1 root root 40688664 2026-10-08 07:20 _C_stable_libtorch.abi3.so
-rwxr-xr-x 1 root root 29748928 2026-10-08 07:23 _moe_C_stable_libtorch.abi3.so
```

安装 + 验证：

```console
$ docker exec z100-build-211 bash /mnt/game/port/install-211.sh _C_stable_libtorch _moe_C_stable_libtorch
-- Installing: /mnt/game/port/gfx906-vllm-211/vllm/_C_stable_libtorch.abi3.so
-- Installing: /mnt/game/port/gfx906-vllm-211/vllm/_moe_C_stable_libtorch.abi3.so
-rw-r--r-- 1 root root 78102728 .../vllm/_C.abi3.so
-rw-r--r-- 1 root root 40688664 .../vllm/_C_stable_libtorch.abi3.so
-rw-r--r-- 1 root root 29748928 .../vllm/_moe_C_stable_libtorch.abi3.so
-rw-r--r-- 1 root root 35156368 .../vllm/_rocm_C.abi3.so
$ md5sum .../vllm/*stable_libtorch*.so
773f6882303556a29027f438bbadbe85  .../_C_stable_libtorch.abi3.so
996b282c4c6a3d3b46b5e8ba4922af39  .../_moe_C_stable_libtorch.abi3.so
$ for so in .../_C_stable_libtorch.abi3.so .../_moe_C_stable_libtorch.abi3.so; do \
    echo -n "$(basename $so): "; strings -a $so | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sed "s/.*--//" | sort -u | tr "\n" " "; echo; done
_C_stable_libtorch.abi3.so: gfx906
_moe_C_stable_libtorch.abi3.so: gfx906
$ python -c "import torch, vllm, vllm._C, vllm._rocm_C, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch; print('IMPORT_OK_ALL4')"
IMPORT_OK_ALL4
```

复跑门禁时，之前那条 `WARNING ... Failed to import from vllm._C_stable_libtorch with ModuleNotFoundError`
**已消失**（§3.5 的最终版输出见 §3.7.6）。

> **重要边界说明**：这两个 `.so` 是用**构建副本里的两处本地改动**编出来的
> （`build-211/temp/build.ninja` 的 hipify wrapper、`gfx906-vllm-211/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu` 的守卫），
> **仓库 `/home/lyh/z100-vllm` 未做任何改动**。因此「重新 rsync + 原样 build」**不会**自动复现这两个 `.so`，
> 必须先按 §5.1 把两项修复落到仓库（或在下一次 rsync 后重新应用）。

#### 3.7.5 失败序列小结（新栈上 `_C_stable_libtorch` 的真实路径）

| 阶段 | 现象 | 归属 | 本步处置 |
|---|---|---|---|
| 修前（`build-211-stable.log`） | `hip_view.hip` **通过**（S5 的 `layout()` 已解），但 `activation_kernels.hip:312: error: use of undeclared identifier 'D'` | DTK torch 2.11 hipify 打包缺陷 | 定位到 `CUDA_MACROS_MAP`，构建树内修好（§3.7.2/§3.7.3） |
| 修 hipify 后（`build-211-stable-fixed2.log`） | `[15/38] … fused_qknorm_rope_kernel.hip:55:24: error: redefinition of '__syncwarp'` | S5 缺陷 #2 的 **stable 孪生文件漏修**（与 torch 版本无关） | 构建树内补齐同一守卫（§3.7.4） |
| 两项都修后（`build-211-stable-fixed3.log`） | `[37/37] Linking HIP shared module _moe_C_stable_libtorch.abi3.so`，`error: 0` | — | **两个目标全部编成** |

#### 3.7.6 最终门禁原文（四扩展齐备后复跑，`/mnt/game/port/gate-211-final.log`）

```console
===== S4b gate 2026-10-08T07:24:41+08:00 =====
python  : Python 3.10.12
torch   : 2.11.0 6.3.26113
dist    : 2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7 | 3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1
triton  : 3.5.0
----- 1) version gate -----
TORCH_PIN_OK  True ; TRITON_PIN_OK True ; torch.version.__hcu_version__ = 2.11.0+das.opt1.dtk2604
----- 2) import gate, torch imported first -----
IMPORT_OK /mnt/game/port/gfx906-vllm-211/vllm/_C.abi3.so        <- 不再有 _C_stable_libtorch 缺失告警
rc=0
----- 2b) literal design-doc wording (no torch preload) -----
terminate called after throwing an instance of 'std::bad_alloc'  rc=134   （已知、非回归，见 §3.5）
----- 3) gfx targets -----
_C.abi3.so              archs: gfx906
_rocm_C.abi3.so         archs: gfx906
vendor _C (reference)   archs: gfx906 gfx926 gfx928 gfx936
----- 4) ROCM_SKINNY_GEMM_ENABLED = False ; hasattr LLMM1 = False -----
----- 5) _rocm_C registered ops: ['_rocm_C::paged_attention'] -----
===== gate done 2026-10-08T07:25:22+08:00 =====
```

### 3.8 给 S6 的入口判断：新 triton（3.6.0+das185）对 gfx906 的支持

**探针 0（静态 + 目标探测，不编译）**

```console
$ python -c "import torch, triton; from triton.runtime import driver
d = driver.active
print(type(d)); print(d.get_current_target())"
<class 'triton.backends.hcu.driver.HCUDriver'>
GPUTarget(backend='hip', arch='gfx906', warp_size=64)
```

- 目标探测**正确**：`arch='gfx906'`、`warp_size=64`（VEGA 语义）。
- 轮子里的后端布局从 3.4 的 `backends/amd/compiler_hcu.py` 换成了独立的
  `backends/hcu/{compiler,driver}.py`，`backends/amd/compiler.py` 仍是上游 stock；
  `importlib.metadata.version('triton') = 3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1`。

**探针 A（默认路径 = amd 后端，elementwise add）——新的 Python/原生 ABI 不匹配**

```console
$ python /mnt/game/port/probe-triton-211-add.py
  File "/usr/local/lib/python3.10/dist-packages/triton/backends/amd/compiler.py", line 265, in make_ttgir
    amd.passes.ttgpuir.add_convert_to_buffer_ops(
TypeError: add_convert_to_buffer_ops(): incompatible function arguments. The following argument types are supported:
    1. (arg0: ...pass_manager, arg1: str, arg2: bool, arg3: bool) -> None
Invoked with: <pass_manager object>, 'gfx906', True, False, False
probe_rc=1
```

原因：`triton/compiler/compiler.py:make_backend()` 有显式分流，`_HCU_ARCHS = {gfx926,gfx928,gfx936,gfx938,gfx92a,gfx946}`
**不含 gfx906**，所以 gfx906 走 stock `amd` 后端；而轮子的 `libtriton.so` 是 HCU 补丁版，
`amd.passes.ttgpuir.add_convert_to_buffer_ops` 的 Python 侧（5 参）与原生侧（4 参）对不上。
绕开这一处（`AMDGCN_USE_BUFFER_OPS=0`）后：

**探针 A'（`AMDGCN_USE_BUFFER_OPS=0`）——S5 的 `unsupported target` 仍在，只是挪到了 LLVM 阶段**

```console
$ AMDGCN_USE_BUFFER_OPS=0 python /mnt/game/port/probe-triton-211-add.py
/mnt/game/port/probe-triton-211-add.py:5:0: error: unsupported target: 'gfx906'
...（附带 MLIR reproducer，pipeline 末段：convert-triton-amdgpu-to-llvm{arch=gfx906 ftz=true}）
/mnt/game/port/probe-triton-211-add.py:5:0: note: Pipeline failed while executing [`ConvertTritonAMDGPUToLLVM` ...]
  File ".../triton/backends/amd/compiler.py", line 349, in make_llir
    pm.run(mod, 'make_llir')
RuntimeError: PassManager::run failed
probeA_rc=1
```

**探针 B（强制走轮子自带的 hcu 后端）——同样失败**

```console
$ python /mnt/game/port/probe-triton-211-hcu.py
compilers claiming gfx906 before patch: ['amd', 'hcu']
========================================================================
HCU/DTK TOOLCHAIN ERROR: AI LLVM (aillvm) not found
  DTK root : /opt/dtk
  Expected : /opt/dtk/aillvm
========================================================================
/mnt/game/port/probe-triton-211-hcu.py:17:0: error: unsupported target: 'gfx906'
  File ".../triton/backends/hcu/compiler.py", line 778, in make_llir
    pm.run(mod, 'make_llir')
RuntimeError: PassManager::run failed
probeB_rc=1
```

（`aillvm` 那段是 hcu 后端发现 `ROCM_PATH` basename 含 "dtk" 时打印的**非致命**告警：
`triton/backends/hcu/compiler.py:354-370` 找不到 `/opt/dtk/aillvm` 就回退 `/opt/dtk/llvm`；
本镜像确实没有 `aillvm`。）

**S6 入口结论**：

1. `unsupported target: 'gfx906'` **没有消失**——它从 3.4 的 TTGIR 阶段（hcu 编译器家族判定）
   挪到了 3.6 的 **LLVM lowering** 阶段（amd 路径的 `ConvertTritonAMDGPUToLLVM` / hcu 路径的
   `convert-triton-hcu-to-llvm`）。也就是说 **S6 的 VEGA ISAFamily 补丁仍然必需**，
   das 3.6.0 轮子同样不是为 gfx906 出的码。
2. 3.6 相比 3.4 的**进步**：目标已能被正确识别（`gfx906`, warp 64），不再在 `make_llir` 之前
   因"未知 target"直接失败。
3. 3.6 的**新增障碍**：a) 默认分流会把 gfx906 送去 stock `amd` 后端，而该后端的 Python 与其
   `libtriton.so` ABI 不匹配（`add_convert_to_buffer_ops` 5 vs 4 参），必须先
   `AMDGCN_USE_BUFFER_OPS=0`（或修 mapping）才能走到 LLVM 阶段；b) 轮子的 hcu 后端也过不了
   gfx906 的 lowering，且镜像没有它期望的 `/opt/dtk/aillvm`。
4. 因此 S6 的路线仍按设计文档 §3：**把 3 文件 VEGA 补丁移植到 3.6.0（或退到已验证的 3.5 基座）
   自行重编 triton**；"直接用 DAS 轮子"这条路已被两个探针排除。

所有 triton 探针都**没有出现挂死**：三条命令分别以 rc=1（Python 异常/编译失败）返回，
GPU 无异常、无 hang，未做任何重试放大。

### 3.9 资源与内存

```console
$ docker stats --no-stream --format '{{.Name}} MEM={{.MemUsage}}' z100-build-211
z100-build-211 MEM=2.972GiB / 8GiB          # 构建峰值约 2.9 GiB，未触发 --memory=8g 上限
$ docker exec z100-build-211 bash -lc 'cat /sys/fs/cgroup/memory.events'
low 0
high 0
max 0        # 未触及 cgroup 上限
oom 0
oom_kill 0
```

> 容器 cgroup 在新容器里读的是 `memory.events`；`max 0 / oom 0 / oom_kill 0`。
> 宿主 `free -g` 全程 `available >= 11 GB`，未出现整体内存压力。

### 3.10 自检：基线容器 / 生产容器 / 仓库都未被破坏

```console
# 1) z100-build（2.10 基线）原样保留，两份 .so 未被碰
$ docker exec z100-build bash -lc 'ls -l /mnt/game/port/gfx906-vllm/vllm/*.so; md5sum /mnt/game/port/gfx906-vllm/vllm/*.so'
-rw-r--r-- 1 root root 78034896 2026-10-08 01:57 /mnt/game/port/gfx906-vllm/vllm/_C.abi3.so
-rw-r--r-- 1 root root 35130920 2026-10-08 01:58 /mnt/game/port/gfx906-vllm/vllm/_rocm_C.abi3.so
4a721d65e712e15d7018e04ecaa99f44  _C.abi3.so
3616084a0c2cfd52ca667af6ea91620c  _rocm_C.abi3.so
$ docker exec z100-build bash -lc 'python -c "import importlib.metadata as md; print(md.version(\"torch\"))"'
2.10.0+das.opt1.dtk2604.20260325.g6b060a        # 仍是 2.10 基线
$ docker ps -a --format '{{.Names}}\t{{.Status}}'
z100-build      Up 6 hours      # 未删、未升级
z100-build-211  Up ...          # 本步新建
qwen38-vllm     Up 7 hours      # 生产容器未受影响

# 2) 仓库无产物、无本步源码改动
$ cd /home/lyh/z100-vllm && git status --short
 M CMakeLists.txt                    <- 全部是 S1–S5 的既有改动
 M cmake/hipify.py
 M csrc/fused_qknorm_rope_kernel.cu
 M csrc/libtorch_stable/quantization/gptq/compat.cuh
 M csrc/quantization/gptq/compat.cuh
 M setup.py
 M tests/kernels/mamba/test_fused_sigmoid_gating.py
 M vllm/_custom_ops.py
 M vllm/envs.py
 M vllm/model_executor/kernels/linear/scaled_mm/rocm.py
 M vllm/model_executor/layers/utils.py
 M vllm/platforms/__init__.py
 M vllm/platforms/rocm.py
?? csrc/dtk_rocprim_compat.h ... ?? tools/dcu/ ...  (untracked，均为 S1–S5 交付)
$ ls -d build *.egg-info vllm/*.so
ls: cannot access 'build': No such file or directory
ls: cannot access '*.egg-info': No such file or directory
ls: cannot access 'vllm/*.so': No such file or directory
```

- 本步**没有**修改 `requirements/dcu*.txt`（S4 固定的两枚直链原样可用）。
- 本步新建的 `z100-build-211` 里没有任何残留编译进程（`ps | grep -E "ninja|clang"` 为空）；
  `z100-build` 里有 **4 个 S4/S5 遗留的 defunct(`<defunct>`) ninja**（PPID=1，容器以 `sleep infinity` 为 1 号进程，
  不会回收），它们是**僵尸**、不占 CPU/内存，不是本步产生，也不影响任何结论——仅记录以免后续误判。
- `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md` 未改；
  DTK 的 RCCL/rocBLAS/hipBLASLt/MIOpen/triton 后端目录、`/opt/hyhal`、`_ON_GFX9`/`on_gfx9()` 未动。
- **未执行任何自编译 HIP 可执行程序**：只用了 `hipcc`/`clang++ -c`、`strings`、`python` 侧 import/torch 探针/triton JIT 派发。

---

## 4. 门禁判定

| # | 门禁（原文） | 判定 | 证据指向 |
|---|---|---|---|
| 1 | `torch.__version__` = `2.11.0+das.opt1.dtk2604` | ✅ **通过（口径修正）** | §3.2：`importlib.metadata.version('torch') = 2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7`、`torch.version.__hcu_version__ = 2.11.0+das.opt1.dtk2604`；`torch.__version__` 只报基础版 `2.11.0`，**2.10 基线同样如此**（DTK 打包行为，非回归） |
| 2 | `triton.__version__` = `3.6.0…` | ✅ **通过（同一口径）** | §3.2：`importlib.metadata.version('triton') = 3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1`；`triton.__version__` 报 `3.5.0`（轮子内置基础版号） |
| 3 | `python -c "import torch, vllm, vllm._C, vllm._rocm_C; print('IMPORT_OK')"` | ✅ **通过** | §3.5 第 2 段：`IMPORT_OK .../vllm/_C.abi3.so`，rc=0（torch 先 import） |
| 4 | 新 `.so` 只含 gfx906（`strings` 口径） | ✅ **通过** | §3.5 第 3 段：`_C.abi3.so → gfx906`、`_rocm_C.abi3.so → gfx906`；对照 vendor 轮子 `gfx906 gfx926 gfx928 gfx936`（口径有区分度） |
| 5 | `ROCM_SKINNY_GEMM_ENABLED` 可导入且为 `False` | ✅ **通过** | §3.5 第 4 段：`ROCM_SKINNY_GEMM_ENABLED = False`、`hasattr LLMM1 = False` |
| 6 | 回归 `test_gguf_fixed_cols_gfx906.py` 13/13 | ✅ **通过** | §3.6：`13 passed, 3 warnings in 29.42s` |
| 7 | 回归 `test_fused_sigmoid_gating.py` 5/5 | ✅ **通过** | §3.6：`5 passed, 1 warning in 14.74s` |
| 8 | 顺带判定：新 triton 上 `unsupported target: 'gfx906'` 是否消失 | ⛔ **未消失（结论明确）** | §3.8 探针 A'/B：amd 与 hcu 两条路径都在 LLVM lowering 报 `error: unsupported target: 'gfx906'` ⇒ S6 仍须 VEGA 补丁重编 |
| 9 | 先只编 `_C _rocm_C`，不裸跑全量 ninja | ✅ **遵守** | §3.4：`--target=_C --target=_rocm_C`；日志 29/29、`error: 0` |
| 10 | 只装两枚目标轮子（不装 dcu-build.txt 依赖） | ✅ **遵守** | §3.1：`pip install --no-deps <torch> <triton>`，`Successfully installed` 只有这两枚；本步未装任何其它 pip 包 |
| 11 | `_C_stable_libtorch` 的 `cuda_view/layout()` 结论 | ✅ **已解决**（升级动因达成） | §3.7.1：`hip_view.hip`（= `cuda_view.cu` 的 hip 化）编译通过，S5 的 `no member named 'layout'` 不再出现 |
| 11b | `_C_stable_libtorch` / `_moe_C_stable_libtorch` 能否编成 | ✅ **编成**（需 2 项构建树本地修复，仓库侧待落地） | §3.7.4：`40688664 B` / `29748928 B`，`error: 0`；两者 `strings → gfx906`；`IMPORT_OK_ALL4`。修复清单见 §5.1 |
| 11c | 四个扩展齐备后重跑门禁 | ✅ **通过** | §3.7.6：`_C_stable_libtorch` 缺失告警消失，版本/import/arch/skinny 门禁全过 |
| 12 | 全量 rsync、排除项锚定 | ✅ **通过** | §3.3：`requirements/build/dcu.txt` 在新副本里存在 |
| 13 | 产物只落 `/mnt/game`；仓库无 `build/`/`.so`/`egg-info` | ✅ **通过** | §3.10 |
| 14 | 不碰生产容器、不删/不升级 `z100-build`、不动 DTK 库与 `_ON_GFX9` | ✅ **通过** | §3.10：`qwen38-vllm` Up、`z100-build` 仍 2.10 且两份 `.so` md5 未变 |
| 15 | 禁止执行自编译 HIP 可执行程序 | ✅ **遵守** | 只用 `hipcc/clang++ -c`、`strings`、python 侧 import / triton JIT 派发 |
| 16 | 不能有挂死/异常放大 | ✅ **遵守** | triton 探针各有明确退出码（rc=1），无 hang、无重试 |

**S4b 总体：✅ 门禁全过。** 升级本身成功且值得保留：
`_C`/`_rocm_C` 在新栈上从零重编且只含 gfx906，S5 的 2 个 gfx906 内核回归无退化，
`cuda_view` 阻塞按期消失。没有触发设计文档 §失败处置 里的"回退 `z100-build`"条件。

---

## 5. 未完成项 / 阻塞与原因

### 5.1 `_C_stable_libtorch` / `_moe_C_stable_libtorch`：**已能编成，但依赖两项尚未进仓库的修复**

- 状态：**已编成**（§3.7.4，两个 `.so` 都只含 gfx906、都能 import）。
- **前提**：本次是在**构建副本/构建树内**打的两个临时补丁，仓库 `/home/lyh/z100-vllm` **零改动**；
  重新 `rsync` 或换构建目录后必须重新落地，否则会退回失败态。**这是本步唯一需要 Lead 决策的落地项。**
- 需要落进仓库的两项修复（都很小，均为 S5 风格的工具链缺陷修复）：

| # | 文件 | 修复 | 归属 | 证据 |
|---|---|---|---|---|
| ① | `cmake/hipify.py`（调用 `hipify()` 之前） | 把 DTK torch 2.11 的裸字符串值补成元组：<br>`from torch.utils.hipify import cuda_to_hip_mappings as _m`<br>`for _k, _v in list(_m.CUDA_MACROS_MAP.items()):`<br>`    if isinstance(_v, str): _m.CUDA_MACROS_MAP[_k] = (_v,)` | **新 DTK 工具链缺陷 #5**（仅 torch 2.11 出现；2.10 的值本来就是元组，`isinstance` 守卫对两版本都安全） | §3.7.2 根因 + §3.7.3 验证 |
| ② | `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu:43` | 与 S5 在 `csrc/fused_qknorm_rope_kernel.cu:45-46` 的修复完全一致：`#if defined(HIP_VERSION) && HIP_VERSION < 70000000 && \`<br>`    !defined(HIP_ENABLE_WARP_SYNC_BUILTINS)` | **S5 缺陷 #2 的漏修孪生文件**（与 torch 版本、与 gfx906 都无关） | §3.7.5 第二行；对照 §3.7.4 |

- 影响面（若①②不落地）：`csrc/libtorch_stable/**` 里所有使用 `CUDA_VERSION` 的 `.cu` 全部无法编译
  （`activation_kernels.cu`、`cache_kernels.cu`、`custom_all_gather_reduce_scatter.cu`、
  `moe/permute_unpermute_kernels/*`、`quantization/w8a8/cutlass/scaled_mm_entry.cu` 等）；
  `_C`/`_rocm_C` **不受影响**（逐处核对：受影响的两处 `_C` header 守卫都必然短路，§3.7.3）。

### 5.2 升级带来的第二个 DTK 侧问题：triton 3.6 轮子 Python 与 `libtriton.so` ABI 不一致

`amd.passes.ttgpuir.add_convert_to_buffer_ops`（Python 5 参 / 原生 4 参）。
绕法：`AMDGCN_USE_BUFFER_OPS=0`。这是 S6 处理 triton 时必须先知道的前置（§3.8）。

### 5.3 明确"未执行"的项

- `triton_kernels`（CMake 目标）的构建：不在本步范围（`TRITON_KERNELS_SRC_DIR` 已备好）。
- `tests/conftest.py` 相关依赖补齐（`tblib`/`mistral_common`/`openai`/`xgrammar`）：**未做也不需要**，
  两个目标测试用 `--noconftest` 跑通（§3.6）。
- 生产容器 `qwen38-vllm` 的任何操作（只做过 `docker ps`/`docker inspect` 只读观察）。
- 任何自编译 HIP 可执行程序的运行（硬约束）。
- 真机 e2e 服务冒烟（S9 范围）。

### 5.4 是否回退

**不回退。** 设计文档 §失败处置 的回退条件是"升级后 `import torch` 崩 / HIP 版本不匹配"，
本次 4 卡枚举正常、HIP 同系列、2 个 gfx906 内核回归全过，`_C`/`_rocm_C` 从零重编成功，
因此**保留 2.11 新栈**，`z100-build`（2.10）继续原样保留作回滚基线。

---

## 6. 对后续步骤的影响

### 6.1 新栈下已经能跑什么（可直接复用）

| 资源 | 路径 | 状态 |
|---|---|---|
| 构建容器 | `z100-build-211`（未 `--rm`，`--memory=8g`）；工作区只读挂 `/src` | 运行中，torch 2.11 + triton 3.6 |
| 构建副本 | `/mnt/game/port/gfx906-vllm-211` | 四个扩展均已安装且**只含 gfx906**：`_C`（78,102,728 B, md5 `6fba0596…`）、`_rocm_C`（35,156,368 B, md5 `5d1e50e1…`）、`_C_stable_libtorch`（40,688,664 B, md5 `773f6882…`）、`_moe_C_stable_libtorch`（29,748,928 B, md5 `996b282c…`）。**注意**：后两个 `.so` 依赖构建副本内的两处本地补丁（§5.1），重新 rsync 前需先落地仓库修复 |
| 持久构建目录 | `/mnt/game/port/build-211/temp` | 增量；`_C`/`_rocm_C` 最新 |
| 脚本 | `/mnt/game/port/{setenv-211.sh,configure-211.sh,build-211.sh,install-211.sh,gate-211.sh}` | 直接可用 |
| 回滚基线 | `z100-build` + `/mnt/game/port/gfx906-vllm`（2.10） | 未被触碰 |
| 已验证能力 | GGUF Q4_K/Q5_K/Q6_K 固定列内核（13 测）、fused sigmoid gating gfx906 内核（5 测）、`_rocm_C::paged_attention` 注册面、`ROCM_SKINNY_GEMM_ENABLED=False` 短路 | 全过 |

### 6.2 S6 从哪里开始（本步给出的入口判断）

1. **不要再试"用 DAS 3.6.0 轮子直接跑 gfx906"**：§3.8 探针 A'/B 已证伪（amd 与 hcu 两条路径都在
   LLVM lowering 报 `unsupported target: 'gfx906'`）。
2. S6 第一步应把 `/home/lyh/server/gfx906_port/artifacts/triton-gfx906.patch` 的 3 处 VEGA 改动
   移植到 **3.6.0**（或按设计文档 §4 退到已验证的 3.5 基座）后**自行重编 triton**。
3. 编译期必须知道的两个坑：(a) 轮子默认把 gfx906 分流给 stock `amd` 后端，而该后端 Python 与
   `libtriton.so` 的 `add_convert_to_buffer_ops` 参数个数不一致 → 需 `AMDGCN_USE_BUFFER_OPS=0`
   或修 `make_backend` 分流表；(b) hcu 后端在 DTK 上会找 `/opt/dtk/aillvm`（本镜像没有，非致命告警）。
4. 判定脚本可直接用 `/mnt/game/port/probe-triton-211-add.py`（设计文档 §1 那段探针的落地版）。
5. S6 的 2 个 unified-attention 测试仍需 `--noconftest` + 先 `import torch`（todolist 硬约束 8）。

### 6.3 对 S7/S8/S9 的影响

- **S7/S8/S9 建议换到 211 容器/副本上做**（新栈已是"目标栈"，且四个扩展齐备）。
  若某一步必须回到 2.10 基线，`z100-build` 与 `build-gfx906` 仍然完整可用。
- `cmake/hipify.py` 的 `CUDA_MACROS_MAP` 修复（DTK 工具链缺陷 #5）建议**优先落地**：它很便宜（3 行）、
  对 2.10 也安全（`isinstance` 守卫），落地后 `_C_stable_libtorch` 只剩 S5 漏修的 `__syncwarp` 孪生文件一处；
  两项合并即可让"全量 `ninja`（默认所有目标）"在新栈上首次走通（S5 §6.6 的"不要裸跑全量 ninja"限制随之解除）。
- 建议把 §5.1 的两项与 S5 的 4 类缺陷合并成 **"DTK 工具链缺陷清单"**（第 5 条 + `__syncwarp` 漏修项），
  在 S9 固化/回滚章节统一说明；回滚到 2.10 时①不需要、②仍然需要（②与 torch 版本无关）。
- 门禁口径补充：**版本判据应使用 `importlib.metadata.version()` 或 `torch.version.__hcu_version__`**，
  不要用 `torch.__version__`（DTK 轮子只报基础版本号）。

---

## 7. 复现方式

> 前提：宿主 docker 可用；本地已有
> `harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`；
> `z100-build` 与 `/mnt/game/port/{wheels,triton-kernels,hf-cache}` 仍在。
> **宿主对 `/mnt/game` 是只读的，所有 `/mnt/game` 下的写入都要经 `docker exec`。**

```bash
# 0) 新建容器（不碰 z100-build / qwen38-vllm）
docker run -d --name z100-build-211 \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --group-add video --security-opt label=disable --ipc host \
  --memory=8g --memory-swap=8g \
  -v /opt/hyhal:/opt/hyhal:ro -v /home/lyh/z100-vllm:/src:ro -v /mnt/game:/mnt/game \
  -w /mnt/game/port \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 \
  bash -c 'sleep infinity'
docker exec z100-build-211 bash -lc 'git config --global --add safe.directory "*"'

# 1) 只装两枚目标轮子（torch 落 /mnt/game，triton 复用既有副本）
docker exec z100-build-211 bash -lc 'mkdir -p /mnt/game/port/wheels-211 && cd /mnt/game/port/wheels-211 && \
  curl -sL -o torch-2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7-cp310-cp310-manylinux_2_28_x86_64.whl \
   "https://download.sourcefind.cn:65024/directlink/4/pytorch/DAS1.8.5/torch-2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7-cp310-cp310-manylinux_2_28_x86_64.whl" && \
  cp -a /mnt/game/port/wheels/triton-3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl .'
docker exec z100-build-211 bash -lc 'cd /mnt/game/port/wheels-211 && pip install --no-deps ./*.whl'
docker exec z100-build-211 bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; python -c "
import torch, triton, importlib.metadata as m
print(m.version(\"torch\"), m.version(\"triton\"))
print(torch.version.hip, torch.version.__hcu_version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"'

# 2) 全量同步工作区（排除项必须锚定）
docker exec z100-build-211 bash -lc '
  rsync -a --delete --exclude=/build --exclude=/.deps --exclude=/.tmp-bf16 \
        --exclude="/*.egg-info" --exclude=/.venv /src/ /mnt/game/port/gfx906-vllm-211/'

# 3) 放脚本（内容见 §2.2 / 本档 §3.3；setenv/configure/build/install/gate 五个文件）
#    —— 与 S4 的 /mnt/game/port/*.sh 同构，只把 REPO_DIR/BUILD_DIR/LOG 换成 211 版本。

# 4) configure（参数与 setup.py:313-379 一致）
docker exec z100-build-211 bash /mnt/game/port/configure-211.sh
#    期望含：Building PyTorch for GPU arch: gfx906
#            -- DTK rocPRIM placement-new workaround enabled: -include .../csrc/dtk_rocprim_compat.h
#            -- Only gfx1250/gfx906, skipping skinny_gemms

# 5) 只编 _C _rocm_C（显式目标，不裸跑 ninja）
docker exec -d z100-build-211 bash -lc 'MAX_JOBS=3 bash /mnt/game/port/build-211.sh _C _rocm_C'
docker exec z100-build-211 bash -lc 'grep -E "^\[2[0-9]/29\]|^FAILED" /mnt/game/port/build-211.log; grep -c "error:" /mnt/game/port/build-211.log'
docker exec z100-build-211 bash /mnt/game/port/install-211.sh _C _rocm_C

# 6) S4b 门禁
docker exec z100-build-211 bash /mnt/game/port/gate-211.sh
docker exec z100-build-211 bash -lc 'source /mnt/game/port/setenv-211.sh >/dev/null 2>&1; cd "$REPO_DIR"; \
  python -c "import torch, vllm, vllm._C, vllm._rocm_C; print(\"IMPORT_OK\")"'
docker exec z100-build-211 bash -lc 'for so in /mnt/game/port/gfx906-vllm-211/vllm/_C.abi3.so /mnt/game/port/gfx906-vllm-211/vllm/_rocm_C.abi3.so; do \
  echo -n "$(basename $so): "; strings -a $so | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sed "s/.*--//" | sort -u | tr "\n" " "; echo; done'
docker exec z100-build-211 bash -lc 'source /mnt/game/port/setenv-211.sh >/dev/null 2>&1; cd "$REPO_DIR"; \
  python -c "import torch, vllm._rocm_C; from vllm._custom_ops import ROCM_SKINNY_GEMM_ENABLED; print(ROCM_SKINNY_GEMM_ENABLED)"'

# 7) 回归（先 import torch，--noconftest）
docker exec z100-build-211 bash -lc 'source /mnt/game/port/setenv-211.sh >/dev/null 2>&1; cd "$REPO_DIR"
  for f in tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py tests/kernels/mamba/test_fused_sigmoid_gating.py; do
    echo "########## $f"
    python -c "import torch, pytest, sys; sys.exit(pytest.main([\"-v\",\"--no-header\",\"--noconftest\",\"-p\",\"no:cacheprovider\",\"$f\"]))"
    echo "##### rc=$? $f"
  done'
# 期望：13 passed / 5 passed

# 8) triton 3.6 的 gfx906 支持判定（S6 入口）
docker exec z100-build-211 bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; \
  python /mnt/game/port/probe-triton-211-add.py'                          # 期望 TypeError(add_convert_to_buffer_ops)
docker exec z100-build-211 bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; \
  AMDGCN_USE_BUFFER_OPS=0 python /mnt/game/port/probe-triton-211-add.py'  # 期望 error: unsupported target: 'gfx906'
docker exec z100-build-211 bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; \
  python /mnt/game/port/probe-triton-211-hcu.py'                          # 期望同样 unsupported target

# 9) stable 目标：现状（未打补丁）= 复现 §5.1 的报错；打两项补丁后 = 全部编成
docker exec -d z100-build-211 bash -lc 'MAX_JOBS=3 LOG=/mnt/game/port/build-211-stable.log \
  bash /mnt/game/port/build-211.sh _C_stable_libtorch _moe_C_stable_libtorch'
docker exec z100-build-211 bash -lc 'grep -E "^FAILED" /mnt/game/port/build-211-stable.log; \
  grep -n "error:" /mnt/game/port/build-211-stable.log | head -3'
# 期望：FAILED .../activation_kernels.hip.o，error: use of undeclared identifier 'D'

# 10) 应用两项构建树本地补丁（内容见本档「附」；**不改仓库**）
docker exec z100-build-211 bash -lc 'cd /mnt/game/port/build-211/temp && cp -a build.ninja build.ninja.s4b-orig && \
  sed -i "/^  COMMAND = cd \/mnt\/game\/port\/build-211\/temp && \/usr\/bin\/python \/mnt\/game\/port\/gfx906-vllm-211\/cmake\/hipify.py/s#/usr/bin/python /mnt/game/port/gfx906-vllm-211/cmake/hipify.py#/usr/bin/python /mnt/game/port/hipify-fix-211.py /mnt/game/port/gfx906-vllm-211/cmake/hipify.py#" build.ninja'
docker exec z100-build-211 bash -lc 'F=/mnt/game/port/gfx906-vllm-211/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu; cp -a $F $F.s4b-orig; \
  perl -0pi -e "s/#if defined\(HIP_VERSION\) && HIP_VERSION < 70000000\n/#if defined(HIP_VERSION) \&\& HIP_VERSION < 70000000 \&\& \\\\\n      !defined(HIP_ENABLE_WARP_SYNC_BUILTINS)\n/" $F; sed -n "40,48p" $F'
docker exec -d z100-build-211 bash -lc 'MAX_JOBS=3 LOG=/mnt/game/port/build-211-stable-fixed3.log \
  bash /mnt/game/port/build-211.sh _C_stable_libtorch _moe_C_stable_libtorch'
docker exec z100-build-211 bash -lc 'grep -E "^\[3[0-9]/37\]|^FAILED" /mnt/game/port/build-211-stable-fixed3.log; \
  grep -c "error:" /mnt/game/port/build-211-stable-fixed3.log'
# 期望：[37/37] Linking HIP shared module _moe_C_stable_libtorch.abi3.so，error 计数 0
docker exec z100-build-211 bash /mnt/game/port/install-211.sh _C_stable_libtorch _moe_C_stable_libtorch
```

---

## 8. 参考

- 本步设计文档：[docs/design/z100-04b-stack-upgrade.md](../design/z100-04b-stack-upgrade.md)
- 前序归档：
  [z100-04-build-system-dtk-result.md](z100-04-build-system-dtk-result.md)（§3.4 构建环境、§3.5 `build_ext --inplace` 理由、§3.7 rsync 锚定、§3.10 S4 门禁、§7 复现配方）、
  [z100-05-csrc-kernels-result.md](z100-05-csrc-kernels-result.md)（§3.5 `_C` 编成、§3.9 5 个测试基线、§5.1 `layout()` 阻塞、§6 可复用资源）
- 后续步骤：[docs/design/z100-06-triton-gfx906.md](../design/z100-06-triton-gfx906.md)（本档 §3.8/§6.2 是它的入口）
- Todolist 与硬约束（8/9/10 条）：[docs/z100-todolist.md](../z100-todolist.md)
- 归档模板：[docs/result/README.md](README.md)
- 关键源码/镜像位置：
  - `setup.py:313-379`（cmake 参数真相源）、`setup.py:96-111`（`_DTK_PREFIX`/`PYTORCH_ROCM_ARCH`）
  - `cmake/hipify.py:64-104`（S5 的 `dtk_macros.h` 后处理；DTK 工具链缺陷 #5 的候选修复点）
  - `csrc/libtorch_stable/activation_kernels.cu:244-297`（`LAUNCH_ACTIVATION_GATE_KERNEL`，用 `CUDA_VERSION`）
  - torch 侧：`/usr/local/.../torch/utils/hipify/cuda_to_hip_mappings.py`（2.10 有 `CUDA_MACROS_MAP` 元组版 vs 2.11 裸字符串版）、
    `/usr/local/.../torch/utils/hipify/hipify_python.py:831-834`
  - triton 侧：`/usr/local/.../triton/compiler/compiler.py:363`（`make_backend` 分流，`_HCU_ARCHS` 不含 gfx906）、
    `/usr/local/.../triton/backends/amd/compiler.py:265`、`/usr/local/.../triton/backends/hcu/compiler.py:354-370`（`aillvm`）
  - 轮子直链：`requirements/dcu.txt:28-29`

---

## 附：两项构建树本地补丁的确切内容（供落地仓库时照抄）

```text
# ① 构建树里对 hipify 的封装（等价于未来落在 cmake/hipify.py 里的 3 行）
#    /mnt/game/port/hipify-fix-211.py 全文：
import runpy
import sys
from torch.utils.hipify import cuda_to_hip_mappings as m
print("BEFORE:", dict(m.CUDA_MACROS_MAP))
for _k, _v in list(m.CUDA_MACROS_MAP.items()):
    if isinstance(_v, str):
        m.CUDA_MACROS_MAP[_k] = (_v,)          # <- 仓库侧应加的就是这三行
print("AFTER :", dict(m.CUDA_MACROS_MAP))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
#    生效方式：build-211/temp/build.ninja 的 hipify_all COMMAND 前插入
#    `/usr/bin/python /mnt/game/port/hipify-fix-211.py`（原 build.ninja 备份为 build.ninja.s4b-orig）

# ② 构建副本里的 stable 孪生守卫（仓库侧应改 csrc/libtorch_stable/fused_qknorm_rope_kernel.cu:43）
-  #if defined(HIP_VERSION) && HIP_VERSION < 70000000
+  #if defined(HIP_VERSION) && HIP_VERSION < 70000000 && \
+      !defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
#    备份：/mnt/game/port/gfx906-vllm-211/csrc/libtorch_stable/fused_qknorm_rope_kernel.cu.s4b-orig
```

> 这两项**都只在 `/mnt/game` 的构建树/构建副本里**，`/home/lyh/z100-vllm` 一行未改。
> 因此 §3.4/§3.5/§3.6 的证据来自**未打 ② 的 `_C`/`_rocm_C`**（其受影响的守卫必然短路，语义等价，§3.7.3），
> 而 §3.7.4/§3.7.6 的 `_C_stable_libtorch`/`_moe_C_stable_libtorch` 证据来自**打了 ①②的构建树**。
