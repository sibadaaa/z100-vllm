# S4 DTK 构建系统（requirements / setup.py / CMake）— 执行结果

> 归档时间：2026-10-08（容器 CST；宿主为 UTC，宿主 `date -Is` = 2026-10-07T16:22Z）
> 执行者：子 agent（z100-s4-executor）
> 对应设计文档：[docs/design/z100-04-build-system-dtk.md](../design/z100-04-build-system-dtk.md)
> 前一档：无（S4 首次归档）
> 状态：🔶 部分完成 —— 构建系统改动全部落地并验证；`vllm._rocm_C` 已在 DTK 容器内真实编译成功且**只含 gfx906**；
> `vllm._C` 因**DTK 26.04 自身的 rocPRIM/clang `operator new` 缺陷**（已给出 6 行最小复现）未编出，属 S5 逐文件修复范围。

## 1. 目标与验收标准

本步门禁原文（`docs/z100-todolist.md`「门禁（Gate）汇总」S4 行）：

> `python -c "import vllm, vllm._C, vllm._rocm_C"` 成功；`.so` 只含 gfx906 目标码

设计文档 §做法 1–3 的可执行验收项（本档逐条判定见 §4）：

| # | 验收项 | 本次结果 |
|---|---|---|
| A1 | 新增 `requirements/dcu.txt` / `dcu-build.txt` / `build/dcu.txt`，不删 rocm 索引外的依赖、不改 `rocm*.txt` | ✅ |
| A2 | DTK 判据与 `hygon_dtk.is_hygon_dtk()` 一致；DTK 下选 `dcu.txt`，非 DTK 行为不变 | ✅（16/16 harness + 容器内实测） |
| A3 | `_is_hip()` 在 `torch.version.hip is None` 且存在 DTK 时按 HIP 处理并 `logger.warning` | ✅（harness 覆盖，含"只告警一次"） |
| A4 | 版本后缀 `+dtk{ver}`（取 `/opt/dtk/.info/rocm_version`） | ✅（实测 `…+dtk2604`） |
| A5 | `-DROCM_PATH` 在 DTK 下确定为 `/opt/dtk` | ✅（实测 cmake 命令行 `-DROCM_PATH=/opt/dtk`） |
| A6 | 显式 `PYTORCH_ROCM_ARCH=gfx906`，不编 `gfx926`/`gfx928` | ✅（编译命令行只有 `--offload-arch=gfx906`；产出 `.so` 只含 gfx906） |
| A7 | 构建产物落 `/mnt/game`，不进 `docs` 所在根分区、不污染工作区 git | ✅ |
| A8 | 门禁：`import vllm, vllm._C, vllm._rocm_C` | ❌ **不通过**（`vllm._C` 未编出）+ 另有一个非本步的 import 崩溃（§5.3） |
| A9 | 门禁：`.so` 只含 gfx906 | ✅ **通过**（`_rocm_C.abi3.so` = `gfx906`；对照 vendor 轮子 = `gfx906 gfx926 gfx928 gfx936`） |
| A10 | 构建环境可复用、可续跑（S5 前提） | ✅（命名容器 `z100-build` + `/mnt/game/port` 持久构建目录 + 脚本，见 §6） |

## 2. 实际改动

行号为**当前工作区**（`/home/lyh/z100-vllm`）的行号。

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `requirements/dcu.txt` | 新增（整文件，2272 B） | 以 `-r common.txt` 为基的 DTK 运行时依赖。相对 `requirements/rocm.txt`：删除 `--extra-index-url …/rocm7.2`、`torch==2.11.0+rocm7.2`、`triton-rocm==3.6.0`、`amdsmi==7.0.2`；改为 DTK 26.04/DAS1.8.5 直链 `torch`/`triton` cp310 轮子；`torchvision`/`torchaudio` 一并去掉（DTK 下载树无对应轮子，且 vLLM 里二者均为可选依赖：`vllm/multimodal/audio.py:32` 用 `PlaceholderModule` 包裹）。其余（`numba`、`datasets`、`ray[cgraph]`、`peft`、`tensorizer`、`runai-model-streamer`、`conch-triton-kernels`、`timm`、`fastsafetensors@git+…`）与 `rocm.txt` 保持一致 |
| `requirements/dcu-build.txt` | 新增（整文件，1049 B） | `rocm-build.txt` 的 DTK 对应物：DTK 直链 `torch`/`triton` + `cmake`/`packaging`/`setuptools`/`setuptools-scm`/`wheel`/`jinja2`/`timm`；无 rocm 索引、无 `triton-rocm`、无 `amdsmi` |
| `requirements/build/dcu.txt` | 新增（整文件，1062 B） | `requirements/build/rocm.txt` 的 DTK 对应物（构建镜像用）：另含 `setuptools-rust`；`tilelang` 未列（DTK 无该轮子且本 fork 不使用） |
| `setup.py` | `:5` | `import glob`（DTK 判据要探测 `/opt/dtk*/bin/hy-smi`） |
| `setup.py` | `:45-48` | 注释：DTK 与社区 ROCm 的三处差异（前缀 `/opt/dtk`、`dcu*.txt`、`+dtkXX.XX`） |
| `setup.py` | `:49` | `_DTK_INFO_VERSION = Path(".info") / "rocm_version"` |
| `setup.py` | `:52-82` | 新增 `_hygon_dtk_prefix()`：与 `vllm/platforms/hygon_dtk.py:86-94` 的 `is_hygon_dtk()` 三条判据一一对应（`<prefix>/.info/rocm_version` 存在，`<prefix>` 取 `ROCM_PATH`/`DTKROOT`/`/opt/dtk`；或 `HYHAL_PATH` 已设；或 `/opt/dtk*/bin/hy-smi` 存在），非 DTK 返回 `None`。**setup.py 不能 import vllm（尚未安装），故判据是复制而非导入，docstring 里写明必须与 hygon_dtk 同步** |
| `setup.py` | `:85-93` | 新增 `_dtk_version()`：读 `<prefix>/.info/rocm_version`，失败返回 `None` |
| `setup.py` | `:96-97` | 模块级 `_DTK_PREFIX = _hygon_dtk_prefix()`、`_DTK_HIP_FALLBACK_WARNED = False` |
| `setup.py` | `:99-109` | DTK 下 `os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx906")`：DTK 的 `rocm_agent_enumerator` 读 `/opt/dtk/bin/target.lst`（`gfx906;gfx926;gfx928;…`）会编多架构；调用方显式设置仍优先 |
| `setup.py` | `:114-115` | `_is_hygon_dtk()` |
| `setup.py` | `:136-148` | 平台自动探测：`torch.version.hip is None` 但存在 DTK 时置 `VLLM_TARGET_DEVICE="rocm"` 并 `logger.warning`。**分支放在 `version.cuda` 之后、`cpu` 之前，CUDA/社区 ROCm 主机走原分支** |
| `setup.py` | `:373-379` | CMake 参数：`elif _is_hip(): rocm_path = ROCM_HOME or _DTK_PREFIX`（原为 `ROCM_HOME` 为 `None` 时直接不加 `-DROCM_PATH`） |
| `setup.py` | `:893-912` | `_is_hip()` 重写：先判 target device，再判 `torch.version.hip is not None`；仅当"设备是 cuda/rocm 且存在 DTK"时按 HIP 处理并**只告警一次**（`_DTK_HIP_FALLBACK_WARNED`）。非 DTK 主机真值表与原一行表达式完全一致（harness 用 12 组组合验证） |
| `setup.py` | `:1021-1033` | 版本后缀：DTK 时 `version += f"{sep}dtk{dtk_version.replace('.', '')}"`（→ `+dtk2604`，与 vendor 轮子 `+das.dtk2604` 命名一致），否则保留原 `+rocm{xxx}` 逻辑 |
| `setup.py` | `:1096-1099` | requirements 选择：`_read_requirements("dcu.txt" if _is_hygon_dtk() else "rocm.txt")` |
| `docs/result/z100-04-build-system-dtk-result.md` | 本文件 | S4 归档 |

**未改动（明确记录）**：
- `CMakeLists.txt`、`cmake/**`：`gfx906` 已在 `CMakeLists.txt:52` 的 `HIP_SUPPORTED_ARCHS` 里，`cmake/utils.cmake:536-570` 的 `override_gpu_arches` 直接读 `ENV{PYTORCH_ROCM_ARCH}`，因此架构只用环境变量固定，**不需要改 CMake**（设计文档 §3 的要求即由此满足）。
- `requirements/rocm.txt`、`requirements/rocm-build.txt`、`requirements/build/rocm.txt`：一行未动。
- `csrc/**`、`vllm/**` 运行时代码（S1/S2/S3 的范围）、`_ON_GFX9`/`on_gfx9()`、`docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`。
- 宿主 `/opt/rocm`：未创建（容器内 `ROCM_HOME` 实测就是 `/opt/dtk`，无需软链）。

工作区 `git status --short` 只多出本步的 `requirements/dcu.txt`、`requirements/dcu-build.txt`、`requirements/build/dcu.txt`、`M setup.py` 与 `docs/result/`；
**没有** `build/`、`*.so`、`*.egg-info`、`vllm/_version.py` 落在工作区（实测 `ls -d build *.egg-info vllm/*.so vllm/_version.py` 全部 `No such file or directory`）。

## 3. 执行过程与原始证据

### 3.1 环境核验（只读，未碰生产容器）

```console
$ docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
z100-bf16-probe  harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.10  Up 3 minutes
qwen38-vllm      harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413  Up 25 minutes  0.0.0.0:8000->8000/tcp

$ docker inspect qwen38-vllm --format '{{json .HostConfig.Devices}}'   # 只读参考
[{"PathOnHost":"/dev/kfd","PathInContainer":"/dev/kfd","CgroupPermissions":"rwm"},
 {"PathOnHost":"/dev/mkfd","PathInContainer":"/dev/mkfd","CgroupPermissions":"rwm"},
 {"PathOnHost":"/dev/dri","PathInContainer":"/dev/dri","CgroupPermissions":"rwm"}]
$ docker inspect qwen38-vllm --format '{{json .HostConfig.GroupAdd}}'
["video"]
$ docker inspect qwen38-vllm --format '{{.HostConfig.IpcMode}} {{.Config.User}}'
host root

$ free -g        # 宿主 31 GB，其中生产容器常驻
               total        used        free      shared  buff/cache   available
Mem:              31          22           2           0           7           8
Swap:              3           2           1
$ df -h / /mnt/game | cat
/dev/mapper/ubuntu--vg-ubuntu--lv  115G   62G   48G  57% /
/dev/nvme1n1p1                     938G  258G  681G  28% /mnt/game
$ docker info | grep -i "docker root dir"
 Docker Root Dir: /var/lib/docker
```

> 注：容器内 `df -h /tmp` 报 `overlay 938G … 28% /`，即 **docker 的可写层落在 938 GB 的 nvme 上**，容器内 `/tmp` 写入不会吃掉 115 GB 根分区（S3 遗留的 `/tmp/torchinductor_root` 1.9 GB 也在同一处）。

### 3.2 非 DTK 主机行为不变性（GPU-free harness，16/16 PASS）

harness 不复制判据，而是用 `ast` 从**真实 setup.py** 里抽出 `_hygon_dtk_prefix`/`_dtk_version`/`_is_hygon_dtk`/`_is_hip` 并在模拟文件系统/env/torch 下执行；文件在 `/mnt/game/port/s4-harness/check_setup_dtk.py`。

```console
$ docker exec z100-build bash -lc 'cd /mnt/game/port/s4-harness && python3 check_setup_dtk.py /mnt/game/port/gfx906-vllm'
harness: /mnt/game/port/gfx906-vllm/setup.py
--- predicate: _hygon_dtk_prefix() ---
PASS  no markers at all: got None, want None
PASS  community /opt/rocm + ROCM_PATH, no DTK files: got None, want None
PASS  DTKROOT env only, no DTK files: got None, want None
PASS  DTK markers on disk: got '/opt/dtk', want '/opt/dtk'
PASS  ROCM_PATH=/opt/dtk with markers: got '/opt/dtk', want '/opt/dtk'
PASS  HYHAL_PATH set, no files: got '/opt/dtk', want '/opt/dtk'
PASS  hy-smi glob only: got '/opt/dtk', want '/opt/dtk'
--- _is_hip(): must equal the pre-DTK formula when DTK is absent ---
PASS  off-DTK _is_hip() matches the original formula (12 combos): got 0, want 0
--- DTK fallbacks ---
PASS  DTK host, torch.hip None, device=rocm -> _is_hip(): got True, want True
PASS  DTK host, torch.hip None, device=cpu  -> _is_hip(): got False, want False
PASS  DTK host, torch.hip 6.3,  device=rocm -> _is_hip(): got True, want True
PASS  off-DTK, torch.hip None, device=rocm  -> _is_hip(): got False, want False
PASS  DTK hip fallback warns once across 3 _is_hip() calls: got 1, want 1
PASS  _dtk_version() is None when the marker is unreadable: got None, want None
PASS  _dtk_version() reads the DTK marker: got '26.04.3', want '26.04.3'
PASS  _dtk_version() honours DTKROOT: got '26.04.3', want '26.04.3'

16/16 checks passed
```

### 3.3 真实 DTK 环境下 setup.py 的实际行为

`/mnt/game/port/s4-harness/check_setup_real.py` 用 `runpy` 跑真实 `setup.py`、把 `setuptools.setup` 换成 sink 后打印它收到的参数（不编译）：

```console
$ docker exec z100-build bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; source /mnt/game/port/gfx906-vllm/tools/dcu/env.sh >/dev/null 2>&1; \
    cd /mnt/game/port/gfx906-vllm && python3 /mnt/game/port/s4-harness/check_setup_real.py /mnt/game/port/gfx906-vllm'
DEBUG: VLLM_TARGET_DEVICE=rocm
DEBUG: _is_cuda()=False
DEBUG: _is_hip()=True
DEBUG: _is_cpu()=False
DEBUG: torch.version.cuda=None
DEBUG: torch.version.hip=6.3.26093
=== summary ===
version          : 0.1.dev12119+g6ba236043.d20261007.dtk2604
PYTORCH_ROCM_ARCH: gfx906
ext_modules      : ['vllm._C', 'vllm._C_stable_libtorch', 'vllm._moe_C_stable_libtorch', 'vllm._rocm_C', 'vllm.cumem_allocator', 'vllm.triton_kernels']
vendor wheels    :
    torch @ https://download.sourcefind.cn:65024/directlink/4/pytorch/DAS1.8.5/torch-2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7-cp310-cp310-manylinux_2_28_x86_64.whl
    triton @ https://download.sourcefind.cn:65024/directlink/4/triton/DAS1.8.5/triton-3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
requirements count: 72
rocm7.2 index in install_requires: False
```

即 `+dtk2604` 后缀、`dcu.txt` 选择、`PYTORCH_ROCM_ARCH=gfx906` 全部生效，`install_requires` 里不含任何 rocm7.2 索引。

容器内工具链实测（同一 DTK 环境）：

```console
$ docker exec z100-build bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; python -c "import torch;print(torch.__version__, torch.version.hip)"; \
    python -c "from torch.utils.cpp_extension import ROCM_HOME, CUDA_HOME; print(ROCM_HOME, CUDA_HOME)"; \
    cat /opt/dtk/.info/rocm_version; cmake --version|head -1; ninja --version; pip show pybind11 setuptools setuptools-scm|grep -E "^Name|^Version"'
2.10.0 6.3.26093
/opt/dtk None
26.04
cmake version 3.29.0
1.11.1.git.kitware.jobserver-1
Name: pybind11 / Version: 3.0.3
Name: setuptools / Version: 79.0.1
Name: setuptools-scm / Version: 9.2.2
```

> 关键差异：镜像自带 **torch 2.10.0+das（hip 6.3.26093）与 triton 3.4.0**，而 `requirements/dcu*.txt` 里按设计文档固定的是 torch 2.11.0 / triton 3.6.0 的轮子。本次构建**用 `--no-deps`，没有安装 requirements**，编译是对着镜像里的 torch 2.10.0 进行的（CMake 也因此打了 `Pytorch version >= 2.13.0 expected for ROCm build, saw 2.10.0 instead.` 警告）。要不要把镜像里的 torch 升到 2.11.0，是 Lead 需要决定的（见 §5.5）。

### 3.4 构建容器与可复用构建环境

```console
$ mkdir -p /mnt/game/port && docker rm -f z100-build
$ docker run -d --name z100-build \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    --group-add video --security-opt label=disable --ipc host \
    --memory=8g --memory-swap=8g \
    -v /opt/hyhal:/opt/hyhal:ro \
    -v /home/lyh/z100-vllm:/src:ro \
    -v /mnt/game:/mnt/game \
    -w /mnt/game/port \
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 \
    bash -c 'sleep infinity'
$ docker ps --filter name=z100-build --format '{{.Names}} {{.Status}}'
z100-build Up 2 seconds
```

- 设备/groups/`--ipc host` 照抄 `docker inspect qwen38-vllm`（只读）。
- **未加 `--rm`**，所以容器及其可写层（含 `/tmp`、site-packages 改动）在 stop/start 之间保留。
- `--memory=8g` 是刻意的**保护措施**：宿主机 31 GB 里有 ~20 GB 被生产容器占着，8 GB 的 cgroup 上限意味着一旦构建失控，被 OOM 杀掉的是**构建容器内部**的编译进程（cgroup 级 OOM），而不是宿主随机挑中生产容器的进程。构建期间 `memory.events` 实测 `oom 0 / oom_kill 0 / max 0`。
- `/src` 是当前工作区的**只读**挂载，构建副本同步到 `/mnt/game/port/gfx906-vllm`（`rsync -a --delete`，**排除项必须锚定**：`--exclude=/build`，否则 `requirements/build/dcu.txt` 会被一起排除——本次踩过并修好，见 §3.7 注）。

脚本（都放在 `/mnt/game/port/`，root 所有，**S5 直接复用**）：

`/mnt/game/port/setenv.sh`：

```bash
export REPO_DIR="${REPO_DIR:-/mnt/game/port/gfx906-vllm}"
export BUILD_DIR="${BUILD_DIR:-/mnt/game/port/build-gfx906}"
export LOG="${LOG:-/mnt/game/port/build.log}"
. /opt/dtk/env.sh >/dev/null 2>&1
. "${REPO_DIR}/tools/dcu/env.sh"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx906}"
export MAX_JOBS="${MAX_JOBS:-2}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${MAX_JOBS}}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
TRITON_KERNELS_FALLBACK=/mnt/game/port/triton-kernels/triton_kernels
if [ -z "${TRITON_KERNELS_SRC_DIR:-}" ] && [ -d "${TRITON_KERNELS_FALLBACK}" ]; then
  export TRITON_KERNELS_SRC_DIR="${TRITON_KERNELS_FALLBACK}"
fi
```

`/mnt/game/port/build.sh` 的核心一行（**不是**设计文档 §4 的 `pip install -e .`，原因见 §3.5）：

```bash
python setup.py build_py --build-lib "${BUILD_DIR}/lib" \
  build_ext --inplace --build-lib "${BUILD_DIR}/lib" --build-temp "${BUILD_DIR}/temp"
```

启动方式（**分离式，不阻塞**）：`docker exec -d z100-build bash /mnt/game/port/build.sh`

### 3.5 `pip install -e .` 不可用于本步：setuptools 把 CMake 构建目录扔进随机临时目录

设计文档 §4 给的命令是 `pip install --no-build-isolation --no-deps -e .`。实测它可以跑起来（把 `.deps` 缓存在 `/mnt/game`），但有两个硬问题，故本步改用 `build_ext --inplace`：

1. setuptools 的 editable 构建固定使用**一次性临时目录**作为 `build_temp`：

```console
$ docker exec z100-build bash -lc 'sed -n "334p" /usr/local/lib/python3.10/dist-packages/setuptools/command/editable_wheel.py; sed -n "219p" ...'
        build_tmp = TemporaryDirectory(suffix=".build-temp")
        build.build_temp = str(tmp_dir)
$ docker exec z100-build bash -lc 'ls -l /proc/604/cwd'      # 604 = cmake configure 进程
lrwxrwxrwx 1 root root 0 Oct  8 00:24 /proc/604/cwd -> /tmp/tmp6xzzoaz9.build-temp
```

⇒ 每次 `pip install -e .` 都是**新的随机目录**，全量重编（不可续跑），且编译产物落在容器可写层。

2. editable 安装把编译出的 `.so` 装进 site-packages，`vllm._C.__file__` 指向 `/usr/local/lib/python3.10/dist-packages/vllm/…`，不在 `/mnt/game`。

改用 `build_py --build-lib` + `build_ext --inplace --build-lib … --build-temp …` 后：CMake/ninja 树固定在 **`/mnt/game/port/build-gfx906/temp`**（增量、可续跑、在 `/mnt/game`），`.so` 落在 `/mnt/game/port/gfx906-vllm/vllm/`，靠 `PYTHONPATH=${REPO_DIR}` 覆盖镜像里预装的 vendor `vllm 0.18.1+das.3266200.dtk2604`（**vendor wheel 未被卸载**，实测 `pip show vllm` 与 `site-packages/vllm/_C.abi3.so` 均完好）。
另注：`build_py`/`build_ext` **没有** `--build-base` 选项（只有 `build` 命令有），第一次写成 `--build-base` 直接报 `error: option --build-base not recognized`，已按 `--build-lib`/`--build-temp` 修正。

### 3.6 `triton_kernels` 的 FetchContent 陷阱与离线化

`cmake/external_projects/triton_kernels.cmake` 在 `TRITON_KERNELS_SRC_DIR` 未设时会 `git clone https://github.com/ROCm/triton.git`（数 GB）。实测该 clone 只有 ~930 KiB/s，15 分钟才 324 MB；而生成的 `…populate-gitclone.cmake` 开头就是

```cmake
execute_process(COMMAND ${CMAKE_COMMAND} -E rm -rf "…/.deps/triton_kernels-src" …)
… git clone --no-checkout … https://github.com/ROCm/triton.git triton_kernels-src
```

⇒ 换一个 CMake 构建目录就会**重新全量 clone**。改为用仓库自己的 Dockerfile 同款做法取单 commit 包：

```console
$ docker exec z100-build bash -lc 'cd /mnt/game/port/triton-kernels && \
    curl -sL -o triton.tar.gz "https://codeload.github.com/ROCm/triton/tar.gz/0f380657dbf3ee86eb57558ff71df24f03b5d4e7" && \
    tar -xzf triton.tar.gz && cp -a triton-0f380657…/python/triton_kernels/triton_kernels .'
$ ls -l triton.tar.gz            # 6.6 MB，秒级完成
$ find /mnt/game/port/triton-kernels/triton_kernels -name "*.py" | wc -l
41
```

configure 日志确认走的是本地目录、**没有再联网**：

```
-- Building skinny_gemms for archs: gfx906
-- [triton_kernels] Fetch from /mnt/game/port/triton-kernels/triton_kernels
-- [triton_kernels] triton_kernels is available at /mnt/game/port/triton-kernels/triton_kernels/
```

### 3.7 configure 证据：架构被固定为 gfx906

```console
$ docker exec z100-build bash -lc 'ps -eo cmd | grep "cmake /mnt/game" | head -1'
/usr/local/lib/python3.10/dist-packages/cmake/data/bin/cmake /mnt/game/port/gfx906-vllm -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_TARGET_DEVICE=rocm \
  -DVLLM_PYTHON_EXECUTABLE=/usr/bin/python -DVLLM_PYTHON_PATH=… \
  -DFETCHCONTENT_BASE_DIR=/mnt/game/port/gfx906-vllm/.deps \
  -DCMAKE_JOB_POOL_COMPILE:STRING=compile -DCMAKE_JOB_POOLS:STRING=compile=2 -DROCM_PATH=/opt/dtk

$ docker exec z100-build bash -lc 'grep -c -- "--offload-arch=gfx906" /mnt/game/port/build.log'
（每一条 clang++ 编译命令行都带，且**看不到** gfx926/gfx928 的 offload-arch）
$ docker exec z100-build bash -lc 'grep -m1 -o -- "-target-cpu gfx906" /mnt/game/port/build.log'
-target-cpu gfx906
```

原始编译命令行节选（`build.log:4557`，长行已截断，`…` 为我加的截断标记）：

```
/opt/dtk/lib/llvm/bin/clang++ -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1 … -D_C_EXPORTS -D__HIP_PLATFORM_AMD__=1 … \
  -isystem /usr/local/lib/python3.10/dist-packages/torch/include … \
  -O2 -g -DNDEBUG -std=gnu++20 --offload-arch=gfx906 -fPIC -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 …
  -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -DTORCH_HIP_VERSION=603 …
  -std=c++17 -O3 -w … -o CMakeFiles/_C.dir/csrc/mamba/mamba_ssm/selective_scan_fwd.hip.o \
  -x hip -c /mnt/game/port/build-gfx906/temp/csrc/mamba/mamba_ssm/selective_scan_fwd.hip
```

> 注（踩坑记录）：第一次同步工作区时用了 `--exclude=build`（未锚定），把 `requirements/build/dcu.txt` 一起排除了；改用 `--exclude=/build` 后修正。另有一次 `rsync -a /src/setup.py /src/requirements/ /mnt/game/port/gfx906-vllm/` 因为 `requirements/` 带尾斜杠，把 `rocm.txt`/`cpu.txt`/`build/`/`test/` 等**铺到了仓库根**；已按"逐项核对 + 精确删除"清干净（`git status` 复核只剩本步文件）。

### 3.8 `csrc` 编译失败：DTK 26.04 自身缺陷（**6 行最小复现**）

`_C` 目标在第 5 个编译单元就失败：

```console
$ docker exec z100-build bash -lc 'grep -n "error:" /mnt/game/port/build.log | head -3'
4576:/opt/dtk-26.04-DCC2602-0317/include/rocprim/types/uninitialized_array.hpp:78:17: error: reference to __host__ function 'operator new' in __host__ __device__ function
4615:…（同一错误重复 20 次后 -ferror-limit 停止）
$ docker exec z100-build bash -lc 'tail -n 12 /mnt/game/port/build.log'
… selectiv_scan_fwd.hip:199:13: note: called by 'selective_scan_fwd_kernel<Selective_Scan_fwd_kernel_traits<128, 16, 1, true, true, true, true, true, c10::Half, float, c10::Half>>'
/usr/include/c++/11/new:174:33: note: 'operator new' declared here
fatal error: too many errors emitted, stopping now [-ferror-limit=]
20 errors generated when compiling for gfx906.
[5/70] Building HIP object CMakeFiles/_C.dir/csrc/mamba/gfx906_decode_kernels.hip.o
ninja: build stopped: subcommand failed.
subprocess.CalledProcessError: Command '['cmake', '--build', '.', '-j=2', '--target=_C_stable_libtorch', '--target=_moe_C_stable_libtorch', '--target=cumem_allocator', '--target=triton_kernels', '--target=_rocm_C', '--target=_C']' returned non-zero exit status 1.
```

**这不是 vLLM 的 bug**，用一个 6 行、不含 vLLM 任何头文件的文件即可复现：

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

根因：DTK 的 `…/clang/18/include/cuda_wrappers/new` 只补了**非 placement** 的 device `new/delete`（文件内注释 "Device overrides for non-placement new and delete."），而 `rocprim::uninitialized_array::emplace` 用的是 libstdc++ 的 placement new（`/usr/include/c++/11/new:174`，host-only）。只要某个 TU 实例化到 `rocprim::block_exchange`/`uninitialized_array::emplace`（`csrc/mamba/mamba_ssm/selective_scan_hip.h:233` 的 `BlockLoadT(...).Load(...)` 就是），就会报错。
顺带验证了一个候选 workaround（**本步未落地**，交 S5 判断）：在这个缺陷被触发前先自己声明 device 版 placement new/delete，

```cpp
__device__ inline void* operator new(std::size_t, void* __p) noexcept { return __p; }
__device__ inline void operator delete(void*, void*) noexcept {}
```

则第一条错误从 `operator new` 变成 `operator delete`（说明 device placement new 被接受），补上 delete 后应可通过——S5 可以用 `-include <shim.h>` 或改 `csrc/mamba/**` 的方式二选一。**注意：本步没有修改任何 `csrc/**`。**

### 3.9 `vllm._rocm_C` 编译成功（本步最重要的产出）

单独编 `_rocm_C` 目标（绕过失败的 `_C`）：

```console
$ docker exec -d z100-build bash /mnt/game/port/build-rocm-c.sh     # 脚本见 §7
$ docker exec z100-build bash -lc 'tail -n 12 /mnt/game/port/build-rocm-c.log'
[2/6] Building HIP object CMakeFiles/_rocm_C_skinny.dir/csrc/rocm/skinny_gemms_int4.hip.o
[3/6] Building HIP object CMakeFiles/_rocm_C_skinny.dir/csrc/rocm/skinny_gemms.hip.o
[4/6] Building CXX object CMakeFiles/_rocm_C.dir/csrc/rocm/torch_bindings.cpp.o
[5/6] Building HIP object CMakeFiles/_rocm_C.dir/csrc/rocm/attention.hip.o
[6/6] Linking HIP shared module _rocm_C.abi3.so
--- ninja rc=0, installing ---
-- Install configuration: "RelWithDebInfo"
-- Installing: /mnt/game/port/gfx906-vllm/vllm/_rocm_C.abi3.so
--- install rc=0 ---
-rw-r--r-- 1 root root 43612528 Oct  8 00:36 /mnt/game/port/gfx906-vllm/vllm/_rocm_C.abi3.so
===== done 2026-10-08T00:36:45+08:00 rc=0 =====
```

### 3.10 S4 门禁原始输出（`docker exec z100-build bash /mnt/game/port/gate.sh`）

```console
===== S4 gate 2026-10-08T00:46:14+08:00 =====
python     : Python 3.10.12   cwd: /mnt/game/port/gfx906-vllm
torch      : 2.10.0 6.3.26093
triton     : 3.4.0
dccobjdump : MISSING (broken symlink in DTK 26.04)

----- 1) import gate, torch imported first -----
WARNING 10-08 00:46:37 [rocm.py:64] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
WARNING 10-08 00:46:37 [rocm.py:70] Failed to import from vllm._C_stable_libtorch with ModuleNotFoundError("No module named 'vllm._C_stable_libtorch'")
IMPORT_OK(torch-first) /mnt/game/port/gfx906-vllm/vllm/__init__.py
rc=0

----- 1b) literal design-doc wording (no torch preload) -----
terminate called after throwing an instance of 'std::bad_alloc'
  what():  std::bad_alloc
rc=134

----- 1c) vllm._C -----
  what():  std::bad_alloc
ls: cannot access '/mnt/game/port/gfx906-vllm/vllm/_C.abi3.so': No such file or directory

----- 2) gfx targets in the freshly built vllm._rocm_C -----
-rw-r--r-- 1 root root 43612528 Oct  8 00:36 /mnt/game/port/gfx906-vllm/vllm/_rocm_C.abi3.so
archs: gfx906

----- 2b) reference: vendor das wheel vllm/_C.abi3.so -----
archs: gfx906 gfx926 gfx928 gfx936

----- 3) hasattr(torch.ops, "_rocm_C") -----
True
rc=0

----- 4) _rocm_C ops (S5 preview) -----
{'LLMM1': True, 'wvSplitK': True, 'wvSplitKQ': True, 'paged_attention_rocm': False}
===== gate done 2026-10-08T00:48:15+08:00 =====
```

`_rocm_C` 里**实际注册**的 op（`torch._C._jit_get_all_schemas()`，同一 DTK 环境）：

```console
$ docker exec z100-build bash -lc 'source /mnt/game/port/setenv.sh >/dev/null 2>&1; cd "$REPO_DIR"; python -c "
import torch, vllm._rocm_C
print(sorted(x.name for x in torch._C._jit_get_all_schemas() if x.name.startswith(\"_rocm_C::\")))"'
6
['_rocm_C::LLMM1', '_rocm_C::paged_attention', '_rocm_C::wvSplitK', '_rocm_C::wvSplitKQ', '_rocm_C::wvSplitK_int4_g', '_rocm_C::wvSplitKrc']
```

架构检测说明：`dccobjdump --offloading` 在本镜像**不可用**——`/opt/dtk/bin/dccobjdump -> ../llvm/bin/dccobjdump` 是**断链**（target 不存在），`llvm-objdump --offloading` 对 DTK 的 `hipfb` 容器输出为空，`clang-offload-bundler --type=o --list` 也识别不出（DTK 用自有 `*.hipfb` 格式）。故 `gate.sh` 回退为枚举内嵌 target id：

```console
$ strings -a <so> | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sed 's/.*--//' | sort -u
gfx906                                        # 新建的 _rocm_C.abi3.so
gfx906 gfx926 gfx928 gfx936                   # 镜像里 vendor das 轮子的 _C.abi3.so（对照，证明该判据有区分力）
```

### 3.11 `import vllm` 未先加载 torch/triton 时 `std::bad_alloc`（非本步范围，但影响门禁写法）

```console
$ python -c "import vllm"                      # PYTHONPATH=REPO_DIR, cwd=REPO_DIR
WARNING … [rocm.py:64] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
WARNING … [rocm.py:70] Failed to import from vllm._C_stable_libtorch with ModuleNotFoundError(…)
terminate called after throwing an instance of 'std::bad_alloc'
  what():  std::bad_alloc
（SIGABRT, rc=134）
$ python -v -c "import vllm" 2>&1 | grep "^import " | tail -3
import 'triton.utils.hcutuner' …
import 'triton.utils.lang' …
import 'triton.utils.testing' …          ← 崩溃前最后一个模块
$ python -c "import torch; import vllm; print('OK', vllm.__file__)"
OK /mnt/game/port/gfx906-vllm/vllm/__init__.py                      # 先 import torch → 正常
$ python -c "import triton; import vllm; print('OK')"
OK                                                                   # 先 import triton → 正常
$ cd /tmp && env -u PYTHONPATH python -c "import vllm; print('VENDOR OK', vllm.__version__)"
VENDOR OK /usr/local/lib/python3.10/dist-packages/vllm/__init__.py 0.18.1   # vendor 0.18.1 单独 import 正常
```

排除内存上限因素：

```console
$ docker exec z100-build bash -lc 'cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current; cat /sys/fs/cgroup/memory.events'
8589934592
570060800
low 0
high 0
max 0
oom 0
oom_kill 0
```

结论：崩溃与 S4 的构建系统无关（发生在 fork 的 import 链里、DTK triton 3.4.0 加载之后），且**只要先 `import torch` 就完全正常**——这正好是生产容器里的实际情形。S1/S2/S3 或 S9 需要认领这个现象（见 §5.3）。

### 3.12 生产容器未被影响 / 磁盘未动

```console
$ curl -s -o /dev/null -w "%{http_code}\n" --max-time 5 http://127.0.0.1:8000/health
200
$ docker ps --format '{{.Names}}\t{{.Status}}'
z100-build    Up 29 minutes
qwen38-vllm   Up 57 minutes
$ df -h / | cat
/dev/mapper/ubuntu--vg-ubuntu--lv  115G   62G   48G  57% /     # 与开工前一致（62G/48G）
$ du -sh /mnt/game/port/build-gfx906 /mnt/game/port/gfx906-vllm
…（构建目录与源码副本都在 /mnt/game，根分区零增长）
```

全程对 `qwen38-vllm` **只执行了 `docker inspect`**，未 exec / restart / commit / kill / 改配置。

## 4. 门禁判定

| 门禁（原文/设计文档） | 判定 | 证据指向 |
|---|---|---|
| `python -c "import vllm, vllm._C, vllm._rocm_C"` 成功 | ❌ **不通过** | §3.10 `1b) rc=134`（且 `vllm._C` 根本未产出）。两个独立原因：`_C` 编译失败（§3.8）、未先 `import torch` 时 `std::bad_alloc`（§3.11） |
| 同上，但显式 `import torch` 在前 | ✅ 通过（S4 目标达成） | §3.10 `IMPORT_OK(torch-first)`，`vllm.__file__` = `/mnt/game/port/gfx906-vllm/vllm/__init__.py` |
| `.so` 只含 gfx906 目标码 | ✅ **通过** | `_rocm_C.abi3.so` → `archs: gfx906`；对照 vendor 轮子 → `gfx906 gfx926 gfx928 gfx936`（§3.10） |
| 设计文档 §2 表：环境识别 / requirements 选择 / `_is_hip()` 兜底 / `+dtk{ver}` / `ROCM_PATH` | ✅ 全部通过 | §3.2（16/16）、§3.3（真实 setup.py 输出 `+dtk2604`、`dcu.txt`、`/opt/dtk`）、§3.7（`-DROCM_PATH=/opt/dtk`） |
| 设计文档 §3：显式 `PYTORCH_ROCM_ARCH=gfx906`，不编 gfx926/gfx928 | ✅ 通过 | §3.3（setup.py 里 setdefault 生效）、§3.7（编译命令行、configure 输出 `Building skinny_gemms for archs: gfx906`） |
| 未装 DTK 的机器（CUDA / 社区 ROCm）行为完全不变 | ✅ 通过 | §3.2 `off-DTK _is_hip() matches the original formula (12 combos)`、`no markers at all → None`、`community /opt/rocm + ROCM_PATH → None`；requirements 分支按 `_is_hygon_dtk()` 二选一，非 DTK 仍读 `rocm.txt`（`setup.py:1099`） |
| 硬约束：产物落 `/mnt/game`；仓库无 `build/`/`.so`/`egg-info` | ✅ 通过 | §2 末（`ls -d build *.egg-info vllm/*.so vllm/_version.py` 全不存在）、§3.12（根分区 48 GB 未变） |
| 硬约束：不碰生产容器、不替换 DTK 库 | ✅ 通过 | §3.12；未安装/替换任何 DTK 的 RCCL/rocBLAS/hipBLASLt/MIOpen/hyhal/triton 后端 |
| S5 门禁预览：`_rocm_C` 暴露 `LLMM1/wvSplitK/wvSplitKQ/paged_attention_rocm` | 🔶 3/4 | `LLMM1/wvSplitK/wvSplitKQ` = True；**`paged_attention_rocm` 这个名字不存在，实际注册名是 `_rocm_C::paged_attention`**（§3.10 的 schema 列表）。S5 门禁措辞需要按实际 op 名修正 |

**S4 门禁总体：🔶 部分通过** —— `.so` 只含 gfx906 这一条（本步的核心诉求）**已通过**，且缺失的 `_rocm_C` 已真实补齐；`import vllm._C` 这条因 DTK 工具链缺陷 + 非本步的 import 崩溃未通过。

## 5. 未完成项 / 阻塞与原因

### 5.1 `vllm._C`（以及 `_C_stable_libtorch`、`_moe_C_stable_libtorch`、`triton_kernels`）未编出
- 直接原因：DTK 26.04 rocPRIM + clang 18 的 **placement new 缺陷**（6 行最小复现，§3.8），第一个撞上的是 `csrc/mamba/mamba_ssm/selective_scan_fwd.hip`（经 `selective_scan_hip.h:233` 的 hipcub `BlockLoad`）。
- **属 S5 范围**：设计文档 `z100-05-csrc-kernels.md` §3「逐文件修复」正是为此准备的（"DTK clang 版本不支持的内建 → 局部替换为等价写法"）。本步按任务书"不改动 vllm/ 下运行时代码 / 只碰构建系统"未改任何 `csrc/**`。
- 已经把它变成**可续跑**：`/mnt/game/port/build-keepgoing.sh`（`ninja -k 0`，单文件失败不会掩盖其它失败，结束时会 `grep` 汇总失败文件清单到 `/mnt/game/port/build-keepgoing.log`）。**该后台构建在归档时已在运行**；S5 重建前必须先停：

```bash
docker exec -d z100-build bash /mnt/game/port/build-keepgoing.sh       # 启动（幂等，可重跑）
docker exec z100-build tail -n 40 /mnt/game/port/build-keepgoing.log   # 看进度
docker exec z100-build pkill -f build-keepgoing                        # S5 重建前先停（ninja 会锁构建目录）
docker exec z100-build pkill -f "ninja -C"
```

### 5.2 设计文档的门禁命令 `dccobjdump --offloading …` 在本镜像**不可执行**
`/opt/dtk/bin/dccobjdump` 与 `/opt/dtk-26.04-DCC2602-0317/bin/dccobjdump` 都是断链（`-> ../llvm/bin/dccobjdump`，target 不存在）。已用"枚举内嵌 target id `amdgcn-amd-amdhsa--gfx<arch>`"替代，并用 vendor 多架构轮子验证判据有区分力（§3.10）。**这是环境事实，不是本步的失败**；建议 Lead 在门禁表里把这条命令改成 `strings` 回退写法，或在镜像里修软链。

### 5.3 `import vllm`（未先 import torch/triton）`std::bad_alloc` → SIGABRT
现象、复现、排除项见 §3.11。**不是 S4 构建系统的问题**，也不在 S4 的改动范围（fork 的 `vllm/__init__.py` + 上游 0.21 的 import 链 与 DTK triton 3.4.0 的交互）。需要 Lead 决策归属：S1/S2/S3 复核 import 链，或记为环境注意事项（所有脚本/文档统一"先 `import torch`"，生产容器实际就是这种顺序）。

### 5.4 S5 门禁里的 `paged_attention_rocm` 名字与实现不符
实际注册名是 `_rocm_C::paged_attention`（+`wvSplitK_int4_g`/`wvSplitKrc`）。见 §3.10。

### 5.5 `requirements/dcu*.txt` 固定的轮子版本 ≠ 镜像实际安装的版本
文件里按设计文档固定 `torch 2.11.0+das…`/`triton 3.6.0+das…`（cp310 直链，实测 HTTP 200 可下载、PEP 508 解析通过、`pip install` 三条需求共 15+9+… 项 0 invalid）；但镜像自带的是 **torch 2.10.0+das / triton 3.4.0**。本次构建 `--no-deps`，**没有安装这些 requirements**，因此没有动过镜像里已验证的 DTK 栈（这是有意的：DTK 与社区轮子不可混用，擅自升级 torch 风险更大）。是否在 S5/S9 阶段把 torch 升到 2.11.0，请 Lead 决定。

### 5.6 明确"未执行"的项
- 全量 `_C` 编译完成（阻塞于 §5.1）；因此 `import vllm._C` / `.so` 只含 gfx906 在 **`_C` 上**未验证（在 `_rocm_C` 上已验证）。
- 真机 import 冒烟（需 GPU 侧，归 S9）。
- `pip install -r requirements/dcu-build.txt` 的真实安装（会动镜像里已验证的 torch，故意不做）。
- 未跑任何自编译 HIP 可执行程序；只用过 `hipcc -c`（编译）、`strings`/`llvm-objdump`（反汇编/检查）与 python 侧 `import`（本步门禁）。

## 6. 对后续步骤的影响

**S5 可以直接复用（这是本步交付的可复用构建环境）**：

| 资源 | 路径/名称 | 说明 |
|---|---|---|
| 构建容器 | `z100-build`（未 `--rm`，`--memory=8g`） | 工作区只读挂在 `/src`；容器内 `cwd=/mnt/game/port` |
| 构建副本 | `/mnt/game/port/gfx906-vllm` | 由 `/src` `rsync` 而来；`vllm/_rocm_C.abi3.so` 已装在 `vllm/` 里 |
| 持久构建目录 | `/mnt/game/port/build-gfx906/temp`（CMake+ninja，增量） | 重跑 `build.sh` / `build-keepgoing.sh` 只重编改动过的文件 |
| 环境脚本 | `/mnt/game/port/setenv.sh` | `source` 即得到 DTK + gfx906 + `TRITON_KERNELS_SRC_DIR` + `PYTHONPATH=REPO_DIR` |
| 构建脚本 | `/mnt/game/port/build.sh`（全量/续跑）、`build-rocm-c.sh`（只编 `_rocm_C`）、`build-keepgoing.sh`（`-k 0` 列举全部失败） | 都可用 `docker exec -d` 分离运行 |
| 门禁脚本 | `/mnt/game/port/gate.sh` | 含 `dccobjdump` 缺失时的 arch 回退 |
| 非 DTK 行为 harness | `/mnt/game/port/s4-harness/check_setup_dtk.py` | `python3 check_setup_dtk.py <repo>` → 16/16；`check_setup_real.py` 打印真实 `setup()` 入参 |
| triton_kernels 源码 | `/mnt/game/port/triton-kernels/triton_kernels` | 免去每次 configure 重 clone 数 GB |
| 日志 | `/mnt/game/port/{build,build-rocm-c,build-keepgoing,build-editable-attempt}.log` | 原始证据 |

S5 的推荐工作流：`docker exec z100-build pkill -f build-keepgoing` → 改 `csrc/**` → `docker exec -d z100-build bash /mnt/game/port/build.sh` → `docker exec z100-build bash /mnt/game/port/gate.sh`。

**前提被修正/推翻的几点**：
1. 设计文档 §4 的 `pip install -e .` **不适用于需要续跑/产物落 `/mnt/game` 的场景**（§3.5），S5 应沿用 `build_ext --inplace` 驱动。
2. 设计文档的门禁 `dccobjdump` 在本镜像不可用（§5.2），S5/S6/S9 都要用回退写法。
3. S5 门禁里 `paged_attention_rocm` 的命名要改成 `paged_attention`（§5.4）。
4. `_C` 的失败是**工具链缺陷**而非本 fork 的架构问题，因此 S5 的范围可能比设计文档预期的更大（不止 `csrc/mamba/**`；任何实例化 `uninitialized_array::emplace` 的 TU 都会中招，需 `-k 0` 清单来定界）。
5. `import vllm` 必须先 `import torch`（§5.3）——S6/S7/S8 的探针脚本都要注意。
6. S4 未阻塞 S5 的开工：`_rocm_C` 已就绪，S5 的 4 个 op 里 3 个已经可以真机/离线核对；`paged_attention` 也在。

## 7. 复现方式

> 前提：宿主 docker 可用，本地已有 `harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`。
> **宿主对 `/mnt/game` 是只读的（本会话 landlock 限制），所以所有 `/mnt/game` 下的写入都要经 `docker exec`**。

```bash
# 0) 工作区改动（本步源码交付物已在工作区）
cd /home/lyh/z100-vllm
git status --short                      # 期望只有 setup.py(M) + requirements/dcu*.txt + requirements/build/dcu.txt
python3 -m py_compile setup.py

# 1) 起可复用构建容器（不碰 qwen38-vllm）
docker rm -f z100-build
docker run -d --name z100-build \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --group-add video --security-opt label=disable --ipc host \
  --memory=8g --memory-swap=8g \
  -v /opt/hyhal:/opt/hyhal:ro -v /home/lyh/z100-vllm:/src:ro -v /mnt/game:/mnt/game \
  -w /mnt/game/port \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 \
  bash -c 'sleep infinity'
docker exec z100-build bash -lc 'git config --global --add safe.directory "*"'

# 2) 同步工作区（**排除项必须锚定**，否则 requirements/build/dcu.txt 会被漏掉）
docker exec z100-build bash -lc '
  [ "$(realpath /mnt/game/port/gfx906-vllm)" = "/mnt/game/port/gfx906-vllm" ] || exit 1
  rsync -a --delete --exclude=/build --exclude=/.deps --exclude=/.tmp-bf16 \
        --exclude="/*.egg-info" --exclude=/.venv \
        /src/ /mnt/game/port/gfx906-vllm/'

# 3) 放脚本（内容见 §3.4 / §3.9 / §5.1；此处只给最小可跑版本）
docker exec -i z100-build bash -c 'cat > /mnt/game/port/setenv.sh' <<'EOS'
export REPO_DIR=/mnt/game/port/gfx906-vllm
export BUILD_DIR=/mnt/game/port/build-gfx906
. /opt/dtk/env.sh >/dev/null 2>&1
. "${REPO_DIR}/tools/dcu/env.sh"
export PYTORCH_ROCM_ARCH=gfx906
export MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
[ -d /mnt/game/port/triton-kernels/triton_kernels ] && \
  export TRITON_KERNELS_SRC_DIR=/mnt/game/port/triton-kernels/triton_kernels
EOS

# 4) 预取 triton_kernels（免去每次 configure 重 clone 数 GB 的 ROCm/triton）
docker exec z100-build bash -lc 'mkdir -p /mnt/game/port/triton-kernels && cd /mnt/game/port/triton-kernels && \
  curl -sL -o triton.tar.gz "https://codeload.github.com/ROCm/triton/tar.gz/0f380657dbf3ee86eb57558ff71df24f03b5d4e7" && \
  tar -xzf triton.tar.gz && rm -rf triton_kernels && \
  cp -a triton-0f380657dbf3ee86eb57558ff71df24f03b5d4e7/python/triton_kernels/triton_kernels .'

# 5) 构建（分离式，不阻塞；--inplace 见 §3.5 的理由）
docker exec -i z100-build bash -c 'cat > /mnt/game/port/build.sh' <<'EOS'
#!/usr/bin/env bash
set -o pipefail
. /mnt/game/port/setenv.sh
mkdir -p "${BUILD_DIR}"; cd "${REPO_DIR}" || exit 1
python setup.py build_py --build-lib "${BUILD_DIR}/lib" \
  build_ext --inplace --build-lib "${BUILD_DIR}/lib" --build-temp "${BUILD_DIR}/temp" \
  2>&1 | tee -a /mnt/game/port/build.log
exit "${PIPESTATUS[0]}"
EOS
docker exec z100-build chmod +x /mnt/game/port/build.sh
docker exec -d z100-build bash /mnt/game/port/build.sh
docker exec z100-build bash -lc 'tail -n 20 /mnt/game/port/build.log; free -g'

# 6) 只编 _rocm_C（绕过 _C 的 rocprim 缺陷；本步产出物）
docker exec -i z100-build bash -c 'cat > /mnt/game/port/build-rocm-c.sh' <<'EOS'
#!/usr/bin/env bash
set -o pipefail
. /mnt/game/port/setenv.sh
cd "${REPO_DIR}"
cmake --build "${BUILD_DIR}/temp" -j "${MAX_JOBS}" --target _rocm_C
cmake --install "${BUILD_DIR}/temp" --prefix "${REPO_DIR}" --component _rocm_C
ls -l "${REPO_DIR}"/vllm/*.so
EOS
docker exec z100-build chmod +x /mnt/game/port/build-rocm-c.sh
docker exec z100-build bash /mnt/game/port/build-rocm-c.sh

# 7) 门禁（`_C` 缺失 + 未先 import torch 会 abort，故分开跑）
docker exec z100-build bash /mnt/game/port/gate.sh          # 完整门禁，输出见 §3.10
docker exec z100-build bash -lc 'source /mnt/game/port/setenv.sh >/dev/null; cd "$REPO_DIR"; \
  python -c "import torch, vllm, vllm._rocm_C; print(\"IMPORT_OK\", vllm.__file__)"'
docker exec z100-build bash -lc 'SO=/mnt/game/port/gfx906-vllm/vllm/_rocm_C.abi3.so; \
  strings -a $SO | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-f]+" | sed "s/.*--//" | sort -u'   # 期望只有 gfx906
docker exec z100-build bash -lc 'source /mnt/game/port/setenv.sh >/dev/null; cd "$REPO_DIR"; \
  python -c "import torch, vllm._rocm_C; print(hasattr(torch.ops, \"_rocm_C\"))"'           # True

# 8) 最小复现 DTK rocprim 缺陷（只编译，不运行）
docker exec z100-build bash -lc 'source /mnt/game/port/setenv.sh >/dev/null 2>&1
printf "%s\n" "#include <rocprim/types.hpp>" "template<class T> __global__ void k(int* o){" \
  "  __shared__ rocprim::uninitialized_array<T,4> a; a.emplace(0, T{}); if(threadIdx.x==0) o[0]=1;}" \
  "template __global__ void k<int>(int*);" > /tmp/p1.hip
hipcc -std=c++17 --offload-arch=gfx906 -c /tmp/p1.hip -o /tmp/p1.o'

# 9) 续跑 / 列举全部失败文件（可选，S5 用；会长时间占用 2 个编译任务）
docker exec -d z100-build bash /mnt/game/port/build-keepgoing.sh
docker exec z100-build tail -n 40 /mnt/game/port/build-keepgoing.log
```

## 8. 参考

- 本步设计文档：[docs/design/z100-04-build-system-dtk.md](../design/z100-04-build-system-dtk.md)
- 后续步骤：[docs/design/z100-05-csrc-kernels.md](../design/z100-05-csrc-kernels.md)（§3 逐文件修复正是 §5.1 的接续）
- 全局约束与门禁：[docs/z100-todolist.md](../z100-todolist.md)
- DTK 环境变量与已知坑：[tools/dcu/README.md](../../tools/dcu/README.md)、[tools/dcu/env.sh](../../tools/dcu/env.sh)
- 归档模板：[docs/result/README.md](README.md)
- 相关上游代码：`setup.py`（本档 §2 行号）、`cmake/utils.cmake:536-570`（`override_gpu_arches` 读 `ENV{PYTORCH_ROCM_ARCH}`）、`CMakeLists.txt:52`（`HIP_SUPPORTED_ARCHS` 含 gfx906）、`cmake/external_projects/triton_kernels.cmake`（`TRITON_KERNELS_SRC_DIR` 分支）、`setuptools/command/editable_wheel.py:219,334`（editable 的 `TemporaryDirectory`）
- DTK 缺陷位置：`/opt/dtk/include/rocprim/types/uninitialized_array.hpp:78`、`/opt/dtk-26.04-DCC2602-0317/llvm/lib/clang/18/include/cuda_wrappers/new`（只补非 placement new）、`/usr/include/c++/11/new:174`
- 环境对照：设计文档 §1 提到的 vendor 参考轮子 `/mnt/game/port/vllm-0.21.0-das.whl`；镜像里 vendor 包为 `vllm 0.18.1+das.3266200.dtk2604`
