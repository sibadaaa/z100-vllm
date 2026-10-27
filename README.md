[English](./README_EN.md)

# vLLM for Hygon DCU Z100

[![vLLM](https://img.shields.io/badge/vLLM-Hygon%20DCU-red)](https://github.com/vllm-project/vllm)
[![DTK](https://img.shields.io/badge/DTK-26.04-purple)](https://www.hygon.cn/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/vllm-project/vllm/blob/main/LICENSE)

**专为海光 DCU Z100/Z100L（gfx906）优化的 LLM 推理引擎。**

> **当前主线**：DTK 26.04 + torch 2.11.0+das + 自编 Triton 3.6 + Qwen3.8-27B-FP8 TP=4。
> 2026-10-27 生产实测：7.5k 单流 **51.6 t/s**（MTP k=2，引擎生成段），128k **31.4 t/s**，
> 30 min decode 长跑 0 失败。
> 性能现状 / 优化路线 / 有效优化总结：
> [STATUS](docs/perf/STATUS-20261027.md) ·
> [ROADMAP](docs/perf/ROADMAP-20261027.md) ·
> [OPT-EFFECTIVE](docs/perf/OPT-EFFECTIVE-20261027.md)。

## 版权声明

本项目基于 [vLLM](https://github.com/vllm-project/vllm) 开发，感谢原项目所有贡献者。

- **原始项目**：[vLLM](https://github.com/vllm-project/vllm) by UC Berkeley Sky Computing Lab
- **许可证**：Apache License 2.0
- **本项目**：包含针对海光 DCU Z100/Z100L 的优化和修复
- **衍生与参考**：本分支在 Apache-2.0 下吸收并参考了公开的 gfx906 适配资料；
  归属见 [NOTICE](NOTICE)

## 项目状态

⚠️ **社区维护分支，生产可用但强依赖具体栈版本。** 本分支不是上游 vLLM 的通用 GPU 构建：
它针对 Z100 的 ISA 缺陷、DTK 运行时差异、无 MFMA/无原生 fp8 的硬件事实做了
大量专用路径。**换 DTK、换 torch、换 Triton 或混装非 DTK 轮子，都可能直接崩或静默算错。**

当前验证过的组合：

| 项 | 值 |
|---|---|
| GPU | 4× Hygon Z100L（gfx906，60 CU，32 GiB，PCIe 4.0 x16） |
| 宿主 | Ubuntu 22.04.5，内核 5.15，12 vCPU，31 GiB RAM + 51 GiB swap |
| DTK | **26.04**（HIP 6.3.26113） |
| Python | **3.10**（DTK 轮子只有 cp310） |
| torch | 2.11.0+das.opt1.dtk2604 |
| Triton | **自编** 3.6.0+git7c56a5e4（gfx906 补丁 + SFU hazard 修复） |
| vLLM 扩展 | `_C` / `_C_stable_libtorch` / `_moe_C_stable_libtorch` / `_rocm_C`，只含 gfx906 |
| 生产模型 | Qwen3.8-27B-FP8，TP=4，MTP k=2，piecewise graph |

> **支持范围**：本 README 只覆盖 **DTK 26.04** 栈。DTK 26.10/其他厂商栈未验证；
> `/opt/dtk` 或 `DTK_ROOT` 必须指向 DTK 26.04。

## 当前性能（2026-10-27）

Qwen3.8-27B-FP8，TP=4，单流，7469-token 真实文档，MTP `num_speculative_tokens=2`。
`t/s` 为**引擎生成段**口径（`tok/step ÷ step_p50`，不含 TTFT）。

| 上下文 | step p50 | tok/步 | t/s |
|---:|---:|---:|---:|
| 7.5k | 46.5 ms | 2.40 | **51.6** |
| 32k | 52.6 ms | 2.27 | 43.0 |
| 60k | 60.6 ms | 2.23 | 36.7 |
| 128k | 62.2 ms | 1.95 | 31.4 |

- 无投机基线 36.9 t/s → MTP k=2 51.6 t/s（+40%，贪心）；采样口径 34.7 → 40.6 t/s（+17%）。
- MTP 的 warm TTFT 从 0.72 s 升到 2.40 s；输出少于约 **220–230 token** 时，端到端可能比无投机更慢。
- 30 min decode 连发 139 探测 / 0 失败；生产 60k 均值约 32.8 t/s。
- c=4 合批聚合 59–70 t/s（decode 相 66–70）；4 路只有单流的 1.2–1.4×，合批 verify 仍是已知低效项。

## 1. 硬件与软件栈

### 1.1 最低要求

| 项 | 要求 |
|---|---|
| GPU | 海光 DCU Z100/Z100L（gfx906，60 CU，32 GiB） |
| 驱动 | 海光 DCU 内核驱动 + `/dev/kfd`、`/dev/mkfd`、`/dev/dri` 权限 |
| OS | Linux x86_64；已在 Ubuntu 22.04 验证 |
| DTK | **26.04**；DTK 26.10/24.04 未验证，不要混用 |
| Python | 3.10（cp310） |
| 磁盘 | 源码 + 构建 + 模型 + 缓存建议 ≥150 GB；本机 `/mnt/game` 余 ~205 GB |
| Docker | 用于构建/实验（本机 29.8.1）；生产也可以裸机直跑 |
| `rocker` | 可选；本机未安装。见 §4.3 |

### 1.2 本机验证过的版本

```text
DTK_ROOT=/mnt/game/host-runtime/dtk2604      # 26.04
python 3.10.12
torch 2.11.0+das.opt1.dtk2604
torch.version.hip 6.3.26113
triton 3.6.0+git7c56a5e4（/mnt/game/triton-gfx906/install）
cmake 3.29（容器内；宿主 3.22 太旧，不要用宿主 cmake）
```

> 注：DTK 的构建变量/产物仍沿用 `PYTORCH_ROCM_ARCH`、`_rocm_C`、
> `.info/rocm_version` 等历史名称；它们是 DTK 的 ABI 标识，不是第二套厂商栈。

### 1.3 三个必须知道的硬件事实

> 海光 DCU Z100/Z100L 的 ISA 标识是 gfx906；下文用 gfx906 指代该架构。

1. **gfx906 没有 MFMA/WMMA**：大 GEMM 走 VALU/`v_dot2`，小 M 走专用 GEMV；
   DTK 的 `hipBLASLt` 在该架构上没有可用 kernel，不要依赖。
2. **gfx906 没有原生 fp8/bf16 指令**：fp8 走 LUT + `v_dot2` 仿真；
   bf16 是软件模拟，实测比 fp16 慢（4096³ 仅 0.36×），默认仍用 fp16。
3. **gfx906 的 Triton 有 SFU hazard**：`tl.exp2`/`tl.sqrt` 可能读回旧值，
   必须使用本仓库补丁后的 Triton 3.6；镜像自带 triton 在 gfx906 上会
   `unsupported target` 或静默算错。

## 2. 目录速览

| 路径 | 内容 |
|---|---|
| `vllm/` | Python 运行时；`vllm/*.abi3.so` 是四个扩展的软链 |
| `csrc/` | HIP/C++ 内核：GDN、fp8 emu、W8A16、custom AR、GGUF、MoE |
| `tools/dcu/` | DTK 环境、构建、微基准、e2e、指纹、探针 |
| `requirements/dcu*.txt` | DTK 26.04 的 torch/triton/依赖固定版本 |
| `docs/perf/` | 性能现状、路线、有效优化总结、SOL/GEMV/ATTN 报告 |
| `docs/design/` | 各轮设计/施工图（S29–S35） |
| `docs/result/` | 各轮实测结果与门禁记录 |
| `docker/` | 上游 vLLM Dockerfile，**不是本 fork 的 DTK 构建入口** |

## 3. 获取 DTK、torch/triton 与依赖

| 组件 | 获取方式 | 本机状态 |
|---|---|---|
| DTK 26.04 | 海光容器镜像 §3.1，或海光安装包 §3.2 | ✅ 容器 `/opt/dtk`；宿主 `/mnt/game/host-runtime/dtk2604` |
| torch 2.11 / triton 3.6 | `requirements/dcu*.txt` 的 sourcefind 直链 §3.3 | ✅ `.venv` 内已安装 |
| Triton gfx906 | 本仓库脚本自编 §3.4 | ✅ `/mnt/game/triton-gfx906/install` |
| vLLM 四扩展 | 本仓库源码编译 §5 | ✅ `vllm/*.abi3.so` |
| RCCL | DTK 自带 | ✅ DTK RCCL + push AR |

### 3.1 DTK 26.04（推荐：容器镜像）

DTK 不在本仓库内。最省事的方式是使用海光发布的 DTK 26.04 容器镜像：

```bash
# 需要海光 Harbor 账号；没有权限时向集群管理员申请
docker login harbor.sourcefind.cn:5443

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413
```

镜像约 **88.5 GB**，内含：

- `/opt/dtk`：DTK 26.04（HIP 6.3.26113），`/opt/dtk/env.sh`
- rocBLAS / RCCL / MIOpen / hipBLAS / hipFFT 等运行时
- Python 3.10、`hy-smi`
- 不包含本项目；源码通过 `-v` 挂载

> 构建/实验容器可直接基于该镜像创建；不要复用生产容器做编译或探针。

### 3.2 宿主机 DTK 树（生产用）

本机生产**不在容器里**，而是裸机 + 宿主机 DTK 26.04：

```bash
export DTK_ROOT=/mnt/game/host-runtime/dtk2604
test -f "$DTK_ROOT/env.sh"
cat "$DTK_ROOT/.info/rocm_version"    # 期望 26.04
source tools/dcu/env.sh               # 会 source $DTK_ROOT/env.sh 并补齐库/头文件
```

`tools/dcu/env.sh` 会做这些事（缺一不可）：

- 过滤出真正导出 `rsmi_init` 的 `librocm_smi64`；
- 补齐 gcc 11 的 C++ 头文件路径（否则 hipcc 报 `'cmath' file not found`）；
- 从 KFD 拓扑设置 `VLLM_DCU_GCN_ARCH=gfx906`；
- 设置 `TRITON_HIP_LLD_PATH`、`TRITON_CACHE_DIR`；
- 设置 `HSA_FORCE_FINE_GRAIN_PCIE=1`（RCCL P2P 必需）；
- **不导出** `ROCBLAS_TENSILE_LIBPATH`（该变量在 DTK 26.04 上会让 rocBLAS
  `Tensile host` 初始化失败并 SIGABRT，见 §10）。

> **重装 DTK**：从海光获取 DTK 26.04 安装包（通常为离线包，无公开 CDN），
> 解包到 `/opt/dtk-26.04` 或 `/mnt/game/host-runtime/dtk2604`，保持 `env.sh` 与
> `.info/rocm_version` 布局。本仓库不附带 DTK 二进制。
>
> **注意**：本机宿主 `/opt/dtk` 当前指向 **26.10**；不要用它编译/运行 26.04 产物。
> 容器内 `/opt/dtk` 才是 26.04。

### 3.3 torch / triton 轮子

`requirements/dcu.txt` 与 `requirements/dcu-build.txt` 固定了 DTK 26.04 配套版本
（cp310 / manylinux_2_28）：

```text
torch @ https://download.sourcefind.cn:65024/directlink/4/pytorch/DAS1.8.5/torch-2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7-cp310-cp310-manylinux_2_28_x86_64.whl
triton @ https://download.sourcefind.cn:65024/directlink/4/triton/DAS1.8.5/triton-3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

安装（宿主机）：

```bash
# 如果还没有 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

cd /path/to/z100-vllm
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements/dcu-build.txt
```

> - `download.sourcefind.cn:65024` 是普通文件树，不是 PEP 503 索引；因此
>   requirements 用直链固定 wheel。
> - **不要**混装非 DTK 的 torch/triton/amdsmi 轮子；DTK 与非 DTK 栈的 ABI 不兼容。
> - DTK 不发 `torchvision`/`torchaudio` 轮子；vLLM 运行时可缺省。
> - 本机 `.venv` 为 Python 3.10.12；未 `source tools/dcu/env.sh` 时
>   `import torch` 会报 `libgalaxyhip.so.5`，这是环境未加载，不是坏了。

### 3.4 自编 Triton gfx906（必需）

DTK 自带的 Triton 在 gfx906 上不可用（`unsupported target`），必须自编：

```bash
bash tools/dcu/build_triton_gfx906.sh
```

脚本会：

1. 创建/复用 `z100-triton` 容器（DTK 26.04 镜像）；
2. clone `triton-lang/triton` v3.6.0 到 `/mnt/game/triton-gfx906/src`；
3. 应用 `tools/dcu/triton-gfx906-3.6.0.patch`（gfx906 ISA family、DTK
   `hipDeviceProp_t`、SFU hazard `s_nop 2`）；
4. `pip install . --target /mnt/game/triton-gfx906/install`，不动 DTK 原装 triton；
5. 打印探针命令。

使用：

```bash
export PYTHONPATH=/mnt/game/triton-gfx906/install:$PYTHONPATH
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
```

验证：

```bash
.venv/bin/python tools/dcu/probe_triton.py
# 期望 TRITON_OK / REDUCE_OK / INT8DOT_OK
```

> 状态：已过 74 项 gfx906 attention 测试；SFU hazard 修复已进二进制。
> 改过 Triton 后端 `.py`/include 后必须清 `TRITON_CACHE_DIR`，否则会命中旧 `.so`。

### 3.5 RCCL

- DTK 自带 `/opt/dtk/lib/librccl.so.1.0`，是 **本栈唯一支持的通信库**；
  TP=2 / TP=4 都直接使用它，不需要任何替换库。
- 生产快路径是 **gfx906 push one-shot custom AR**（见 §6.3），RCCL 是 fallback。
- TP=1 不经过 RCCL。

## 4. 启动容器（Docker ）

### 4.1 构建 / 实验容器（已验证）

```bash
IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

docker run -d --name z100-dev \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --group-add video --security-opt label=disable --ipc host \
  --shm-size=16g --memory=8g --memory-swap=8g \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$PWD":/src \
  -v /mnt/game:/mnt/game \
  -w /src \
  "$IMAGE" bash -c 'sleep infinity'
```

说明：

- `/dev/kfd`、`/dev/mkfd`、`/dev/dri` 三个都要透传；只给 `/dev/kfd` 常见于
  抄 NVIDIA 配方，会看不到 DCU 或 P2P 异常。
- `--group-add video` 让非 root 用户也能访问设备；容器内默认 root 时可省，
  但建议保留。
- `--ipc host`、`--shm-size=16g`：vLLM/多进程/RCCL 需要。
- `--memory=8g` 是**构建容器的保护**：防止编译失控把宿主 31 GiB 内存打爆。
- **服务容器不要设 cgroup 内存上限**（`--memory=0`）；27B 加载期需要
  staging/页缓存，有限制会在启动阶段 OOM。
- 不加 `--rm`，容器可 stop/start 复用；构建产物在 `/mnt/game`，不丢。
- 生产容器/服务**不要 exec、restart、commit**；实验一律新建一次性容器。

### 4.2 服务容器（如果要容器化部署）

在上面命令基础上改成：

```bash
--memory=0 --memory-swap=0 --shm-size=16g --ipc host
```

并把 `-p 8000:8000`、模型目录、API key 挂进去。当前生产是**宿主机裸机**，
容器只用于构建/实验，避免两层环境差异。

### 4.3 进容器后的环境自检

```bash
docker exec -it z100-dev bash

source /opt/dtk/env.sh
python3 -c "import torch; print(torch.__version__, torch.version.hip)"
# 期望 2.11.0 6.3.26113（镜像基线可能是 2.10；按 §3.3 装 2.11 轮子）

hy-smi --showproductname
# 期望 4× Z100L

python3 -c "import torch; print(torch.cuda.device_count())"
# 期望 4
```

宿主机裸机：

```bash
export DTK_ROOT=/mnt/game/host-runtime/dtk2604
source tools/dcu/env.sh
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.hip)"
```

## 5. 编译

### 5.1 Python 环境

- 必须 **Python 3.10**（DTK 轮子只有 cp310）。
- 推荐 `uv`；容器内没有 `uv` 时先执行
  `curl -LsSf https://astral.sh/uv/install.sh | sh`。
- 不要把构建目录放在 `/`（根分区只有 ~42 GB）；统一放 `/mnt/game`。

```bash
cd /src   # 或 /path/to/z100-vllm
uv venv --python 3.10 /mnt/game/venvs/z100
source /mnt/game/venvs/z100/bin/activate
uv pip install -r requirements/dcu-build.txt
```

### 5.2 编译四个 vLLM 扩展

```bash
cd /src
source /opt/dtk/env.sh          # 容器内；宿主机用 source tools/dcu/env.sh
source tools/dcu/env.sh

export PYTORCH_ROCM_ARCH=gfx906
export MAX_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4

python setup.py build_ext --inplace
```

说明：

- **必须显式 `PYTORCH_ROCM_ARCH=gfx906`**。DTK 的 `rocm_agent_enumerator` 会读
  `/opt/dtk/bin/target.lst`（`gfx906;gfx926;gfx928;...`），不限制会把 926/928
  一起编进来：体积 ×3，且可能生成 Z100 不能执行的 ISA。
- 首次全量编译约 **1.5–3 h**（无 ccache）；`MAX_JOBS` 从 4 起，观察内存再调。
  本机 31 GiB RAM，6 路并行有 OOM 风险。
- 本 fork 在 DTK 上**推荐 `build_ext --inplace`**；`pip install -e .` 在
  setuptools 临时构建目录上有失败先例。若坚持 editable install，确认构建目录
  在 `/mnt/game` 且失败后清理 `build/`。
- 编译失败先看 `dmesg | grep -i oom`，不要盲目加 `-j`。

构建完成后，让 Python 找到本仓库（二选一）：

```bash
# A) 直接用仓库目录（生产臂就是这样跑的）
export PYTHONPATH="$PWD:$PYTHONPATH"

# B) 只注册 Python 包，不解析依赖（避免拉 torchvision/flash_attn）
uv pip install --no-build-isolation --no-deps -e .
```

> `requirements/dcu-build.txt` 里的 `triton` 是 DTK 原装轮子；**gfx906 必须用
> §3.4 自编的 Triton 覆盖 `PYTHONPATH`**。否则会踩 `unsupported target: 'gfx906'`。

产物（`vllm/` 下，或 `build/lib.../vllm/`）：

| 扩展 | 主要内容 |
|---|---|
| `_C.abi3.so` | GDN/Mamba、GGUF、fork 的 gfx906 内核 |
| `_C_stable_libtorch.abi3.so` | stable ABI 的 `torch.ops._C`；push AR 内核也在这里 |
| `_moe_C_stable_libtorch.abi3.so` | MoE 内核 |
| `_rocm_C.abi3.so` | fp8 emu、W8A8 sdot4、W8A16、bf16 emu |

> **运行期 `torch.ops._C` 实际由 `_C_stable_libtorch` 提供**（stable ABI 路径）。
> 改 `csrc/libtorch_stable/**` 时要重编/换装的是 `_C_stable_libtorch`，
> 只改 `_C` 可能看不到行为变化。

### 5.3 增量构建与换装

增量构建建议使用独立的构建目录（放在数据盘，不要放 `/` 或源码树内），
并在换装前备份现有扩展：

```bash
export BUILD_DIR=/path/to/vllm-build   # 独立构建目录

# 1) 备份当前扩展
cp vllm/_rocm_C.abi3.so "vllm/_rocm_C.abi3.so.bak-$(date +%Y%m%d-%H%M)"

# 2) 在已配置的 CMake 构建目录里只编目标（示例：_rocm_C）
cmake --build "$BUILD_DIR" -j "${MAX_JOBS:-4}" --target _rocm_C

# 3) 换装：复制到 vllm/（或用软链；产物路径以你的构建目录为准）
cp "$BUILD_DIR/_rocm_C.abi3.so" vllm/
```

> 如果没有独立 CMake 构建目录，也可以重新执行
> `python setup.py build_ext --inplace`；首次全量编译较慢，之后增量会明显加快。

目标选择：

| 改动区域 | 目标 |
|---|---|
| `csrc/rocm/fp8_emu_gemm.cu`、W8A16、sdot4 | `_rocm_C` |
| `csrc/custom_all_reduce.cuh`（push AR） | `_C_stable_libtorch` |
| `csrc/mamba/gfx906_decode_kernels.cu`（GDN） | `_C` |
| `csrc/quantization/gguf/moe*` | `_C_stable_libtorch`（或 `_moe_C_stable_libtorch`，按注册表） |

换装后：

```bash
# 确认四个扩展的版本/时间戳/校验和一致
ls -l vllm/*.abi3.so
```

> 换 `.so` 后必须重启服务并核对策略指纹/校验和。若构建日志显示 hipify
> 未产出且产物未更新，重跑构建；不要手改生成物。

### 5.4 编译门禁

```bash
# 1) 四个扩展可导入
python -c "import vllm, vllm._C, vllm._rocm_C; print('IMPORT_OK')"

# 2) 只含 gfx906（dccobjdump 在 DTK 镜像里是断链，用 strings 回退）
strings vllm/_C.abi3.so | grep -o 'gfx[0-9]*' | sort -u
strings vllm/_rocm_C.abi3.so | grep -o 'gfx[0-9]*' | sort -u

# 3) 扩展清单与软链
bash tools/dcu/use_stack.sh status

# 4) 关键单测（venv 已激活，且需要 GPU）
python -m pytest tests/kernels/quantization/test_gfx906_fp8_emu.py -q
```

### 5.5 其他组件状态

| 组件 | 状态 | 说明 |
|---|---|---|
| DTK 26.04 | ✅ 已验证 | 容器 `/opt/dtk`；宿主 `/mnt/game/host-runtime/dtk2604` |
| torch 2.11.0+das | ✅ 已验证 | 只用 `requirements/dcu*.txt` 的 cp310 轮子 |
| Triton 3.6 自编 | ✅ 必需 | `/mnt/game/triton-gfx906/install`；镜像自带不可用 |
| RCCL | ✅ DTK 自带 | TP=2/TP=4 可用；生产快路径是 push custom AR |
| rocBLAS | ✅ 可用 | gfx906 Tensile 覆盖良好；**不要**设 `ROCBLAS_TENSILE_LIBPATH` |
| hipBLASLt | ❌ 无 gfx906 | DTK 的 hipBLASLt 无可用 kernel；prefill 走 rocBLAS/自研 tile |
| amdsmi | ⚠️ 不使用 | DTK 无 python amdsmi；用 `hy-smi` + KFD sysfs |
| torchvision/audio | ⚠️ 无轮子 | 可选依赖；vLLM 运行时允许缺失 |
| fastsafetensors | ✅ requirements | 固定 git commit |
| compressed-tensors | ✅ requirements | 固定 0.15.0.1（0.14 会缺 `pack_quantized`） |
| flash-attn | ❌ 不使用 | gfx906 attention 走自研 Triton |

## 6. 运行

### 6.1 下载模型

```bash
export VLLM_USE_MODELSCOPE=true
# 方式 A：ModelScope（若未安装：uv pip install modelscope）
modelscope download Qwen/Qwen3.8-27B-FP8 --local_dir /mnt/game/models/Qwen3.8-27B-FP8

# 方式 B：Hugging Face
huggingface-cli download Qwen/Qwen3.8-27B-FP8 --local-dir /mnt/game/models/Qwen3.8-27B-FP8
```

确保目标目录有 `config.json`、safetensors 权重和 tokenizer；FP8 checkpoint 的
`quantization_config` 会被 vLLM 自动识别。

### 6.2 最小启动（不带生产调优）

```bash
export DTK_ROOT=/mnt/game/host-runtime/dtk2604
source tools/dcu/env.sh

vllm serve /mnt/game/models/Qwen3.8-27B-FP8 \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --compilation-config '{"cudagraph_mode":"piecewise","cudagraph_capture_sizes":[1,2,4]}' \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 8000
# 如需鉴权，追加：--api-key "$VLLM_API_KEY"
```

> 若没有 `vllm` 命令，用
> `python -c 'from vllm.entrypoints.cli.main import main; main()' serve ...`，
> 或先按 §5.2 做 editable 安装。
>
> 这条命令只覆盖“能跑”；**生产性能还依赖一组 `VLLM_DCU_*` / `VLLM_GFX906_*`
> 环境开关**（push AR、GEMV stack、AR size 门、W8A16、GDN M1、BM16 等）。
> 生产配方请参考 [STATUS §1.3](docs/perf/STATUS-20261027.md)。

### 6.3 生产配方与运维

生产部署建议由进程管理器（systemd / supervisor / 容器编排）托管，并只暴露必要端口。
以 systemd 为例：

```bash
systemctl status <vllm-service>
systemctl start  <vllm-service>
journalctl -u <vllm-service> -f
```

- 监听地址/端口：按部署需要设置（示例 `<host>:<port>`）。
- API key：通过环境变量或密钥管理服务注入，不要写进仓库或启动脚本。
- 启动/回滚：使用你自己的部署脚本/服务单元；回滚 = 切回上一稳定配置
  （去掉上一轮的 AR/GEMV/`MAX_KIB` 环境开关，或恢复上一版扩展）。

生产 env 摘要：

| 组 | 关键开关 |
|---|---|
| 通信 | `VLLM_DCU_CUSTOM_AR=1`、`VLLM_DCU_CUSTOM_AR_ALLOC=finegrained`、`VLLM_CUSTOM_ALLREDUCE_ALGO=oneshot`、`VLLM_DCU_CUSTOM_AR_MAX_KIB=128` |
| 线性 | `VLLM_DCU_FP8_W8A16=1`、`VLLM_DCU_FP16_PREFILL=auto`、`VLLM_DCU_FP8_GEMV_STREAM_ROWS=4`、`VLLM_DCU_FP8_GEMV_M1=1`、`VLLM_DCU_FP8_DECODE_CFG=35` |
| GDN | `VLLM_GFX906_GDN_DEC_M1=1`、`VLLM_QWEN35_MTP_GDN_TRITON=1` |
| Attention | `VLLM_GFX906_ATTN_DECODE_BLOCK_M=16`、`VLLM_GFX906_ATTN_VERIFY_3D=1` |
| 数值 | `ROCBLAS_WORKSPACE_CONFIG=:4096:8`、`HIPBLAS_WORKSPACE_CONFIG=:4096:8` |

> **运维注意**：若已有手工 `vllm serve` 占用端口，先确认无在飞请求并停止，
> 再启动托管单元；不要直接 `systemctl restart`。

### 6.4 API 冒烟

```bash
curl -s http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "你好！请介绍一下你自己"}],
    "max_tokens": 100
  }'
```

### 6.5 多模态 GGUF

Qwen3.5/Qwen3.6 的 GGUF 主模型与 `mmproj` 放在同一目录，启动时用
`--limit-mm-per-prompt` 限制 profiling 尺寸：

```bash
vllm serve /path/to/Qwen3.x-xxB-GGUF \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":{"count":1,"width":512,"height":512},"video":0}' \
  --reasoning-parser qwen3_5
```

> GGUF 路径仍建议 `--enforce-eager`；只有 FP8 safetensors 的 Qwen3.8-27B
> 生产臂验证过 piecewise graph + MTP。两者不要混用配方。

## 7. 模型支持

**已测试并确认可用：**

- ✅ Qwen/Qwen3.5-0.8B（原始 F16 权重）
- ✅ Qwen/Qwen3.5-2B（原始 F16 权重）
- ✅ Qwen/Qwen3.5-4B（原始 F16 权重）
- ✅ Qwen/Qwen3.5-9B（原始 F16 权重）
- ✅ unsloth/Qwen3.5-27B-GGUF
- ✅ unsloth/Qwen3.6-27B-GGUF
- ✅ unsloth/Qwen3.8-27B-GGUF
- ✅ **Qwen/Qwen3.8-27B-FP8**（4×gfx906，TP=4，MTP k=2 生产验证）
- ✅ unsloth/Qwen3.6-35B-A3B-GGUF

**已知无法运行：**

- ❌ **Qwen/Qwen3.5-35B-A3B-GPTQ-Int4**：MoE + GPTQ Int4 组合在 gfx906 上
  无法正常启动或推理。
- ❌ **cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit**：返回整串 `!` 或初始化/推理崩溃。

**未经测试（同步自上游，可能不工作）：**

- 🔄 其他 MoE（exaone_moe、glm4_moe_lite 等）
- 🔄 视觉模型（colqwen3、molmo2 等）
- 🔄 音频模型（whisper、funasr 等）
- 🔄 嵌入模型（colbert、voyage 等）

> 性能数据只对 Qwen3.8-27B-FP8 + DTK 26.04 + 4×Z100L 这一组负责；
> 其他模型可能“能跑”，但没有经过本仓库的 kernel/带宽调优。

## 8. 量化支持

| 量化 | 状态 | 说明 |
|---|---|---|
| FP8（W8A8 block） | ✅ 生产 | gfx906 无原生 fp8；走 LUT + `v_dot2` 仿真，权重字节减半 |
| W8A16 | ✅ 生产 | 激活不量化，权重 1 B/参数；decode 1.2–2.1×，prefill M≥512 走 fp16 物化 |
| W8A8 INT8 sdot4 | ⚠️ 候选 | `v_dot4_i32_i8` 原生 3.9× 潜力；需质量 eval + 用户裁决，未默认 |
| W4A16 | ⚠️ 实验 | 只建议量化 MLP；`v_dot8` 已实测 No-Go，需走 LUT 路径并 eval |
| GPTQ | ⚠️ 部分 | kernel/形状/MoE 组合仍有兼容性问题 |
| AWQ | ⚠️ 部分 | 默认 Triton；部分模型输出异常或启动失败 |
| MoE 量化 | ⚠️ 不推荐 | MoE + GPTQ/AWQ 失败组合；见 S35 文档 |

## 9. 已知限制

1. **首次推理较慢**：Triton 内核首次编译，等待 1–2 分钟属正常。
2. **内存占用大**：KV cache 预分配会占大量显存；27B 建议 `util 0.85`。
3. **实验性质**：本分支针对特定栈；升级 DTK/torch/Triton 前先读
   [AGENTS.md](AGENTS.md) 与性能文档。
4. **MoE + GPTQ 兼容性**：见 §7；不要在生产直接试。
5. **MTP k=2 的 TTFT 代价**：7.5k warm TTFT 0.72→2.4 s；短回复
   （约 <220–230 输出 token）端到端可能更慢。
6. **长上下文冷启动**：60k/128k 冷 prefill 约 178 s/520 s；prefix cache
   命中率决定 TTFT，长会话尽量复用前缀。
7. **`/v1/completions` + `logprobs≥2`**：本 fork 存在可致 API server 退出的
   历史缺陷；生产建议用 chat completions 或 `logprobs=1`，使用前先验证。
8. **MoE e2e 未在本机当前栈验证**：S35 是独立轨，见
   [MoE 设计](docs/design/z100/s35/z100-s35-moe-design.md)。
9. **hipBLASLt 无 gfx906**：不要期待 `torch.mm` 走 hipBLASLt；实际走 rocBLAS。
10. **`dccobjdump` 断链**：DTK 镜像里不能用来查 offloading；用 `strings`/`clang`。

## 10. 故障排除

| 现象 | 原因 / 处置 |
|---|---|
| `ImportError: libgalaxyhip.so.5` | 没 `source tools/dcu/env.sh`；或 `DTK_ROOT` 指错 |
| `undefined symbol: rsmi_init` | `librocm_smi64` 选到了 stub；用 `tools/dcu/env.sh`（会过滤） |
| `fatal error: 'cmath' file not found` | 没加载 DTK 的 C++ 头路径；`source tools/dcu/env.sh` |
| `rocBLAS error: Could not initialize Tensile host` + SIGABRT | **取消** `ROCBLAS_TENSILE_LIBPATH`；DTK 只有 lazy 库 |
| `std::bad_alloc` / SIGABRT on import | 先 `import torch` 再 `import vllm`；检查 swap；不要给容器设内存上限 |
| `unsupported target: 'gfx906'`（Triton） | 用了镜像自带 Triton；改用 §3.4 自编版本 |
| `Data-dependent branching`（full graph） | 用 `cudagraph_mode=piecewise`，不要 FULL |
| 启动/加载 OOM | 降 `--max-model-len`/`--max-num-batched-tokens`/`util`；确认 51 GiB swap；容器 `--memory=0` |
| `Address already in use` / 8000 被占 | 有手工臂在跑；先停手工进程，再 `systemctl start` |
| GPU 有占用但无进程 | 孤儿 worker：`pgrep -af "VLLM[:] :Worker"` 找 pid，kill -9 后 `hy-smi` 归零 |
| `hy-smi` 打不开 `mkfd` | 设备没透传或驱动异常；检查 `/dev/mkfd` 与容器 `--device` |
| 视觉 profiling OOM | 加 `--skip-mm-profiling` 定位；或用 `--limit-mm-per-prompt` 降低尺寸 |
| 首次请求挂起 | Triton 编译；等 1–2 分钟，不要重启 |

## 11. 开发与文档

- **贡献前必读**：[AGENTS.md](AGENTS.md)（重复工作检查、测试、PR 纪律）
- **性能**：[STATUS](docs/perf/STATUS-20261027.md)、
  [ROADMAP](docs/perf/ROADMAP-20261027.md)、
  [OPT-EFFECTIVE](docs/perf/OPT-EFFECTIVE-20261027.md)
- **设计/施工**：[Z100 文档索引](docs/design/z100/README.md)
- **实测结果**：`docs/result/z100-*.md`
- **工具**：`tools/dcu/`（环境、构建、微基准、e2e、指纹、探针）
- **生产配置**：见 [STATUS](docs/perf/STATUS-20261027.md) §1.3

常用检查：

```bash
bash tools/dcu/use_stack.sh status
.venv/bin/python tools/dcu/dcu_policy_fingerprint.py --prefix "[policy] "
hy-smi --showproductname
```

## 12. 对 1Cat-vLLM 的借鉴

本仓库在**控制面与测量方法**上参考了 [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)
（一个同样没有原生 fp8 的 vLLM 分支）。**没有移植它的数据面 kernel**：两边的矩阵指令、
内存层级和并行粒度不同，其 attention/GEMM/量化内核在 gfx906 上不可用；可迁移的是
“每项优化都带准入谓词、回滚开关和数值门”的做法。

### 12.1 已落地

| 借鉴点 | 1Cat 的做法 | Z100 的落地 | 证据 |
|---|---|---|---|
| W8A16 契约 | 权重 fp8 常驻 HBM，激活 fp16 直入，`apply_input_quant=False`，去掉激活量化 | `VLLM_DCU_FP8_W8A16` + fp16 prefill 物化；decode 1.2–2.1×，prefill M≥512 1.4–2× | [`w8a16`](docs/result/z100-w8a16-fp16-prefill-result.md) |
| 五相位 profiling | `draft` / `draft→target` / `target` / `target→draft` / `round` | [`s27_five_phase_profile.py`](tools/dcu/s27_five_phase_profile.py) 直接采用该切分；五段和与 step 差 1.95% | [`decode-prefill`](docs/result/z100-decode-prefill-final-summary.md) |
| 门禁形式（+ Z100 自建指纹） | admission 谓词、可回滚开关、bitwise/质量门；1K/2K/4K/8K 不退化 | 同窗口 A/B、N≥2 重启、贪心 parity、30 min 长跑；另加自建 policy fingerprint | [`s29-pin`](docs/result/z100-s29-accept-pin.md) |
| “只搬控制面”的判断 | 数据面 kernel 约 7 万行、控制面约 150–250 行；前者绑定专用矩阵核 | 把精力放在 GDN/GEMV/AR/attention 的调度与并行度，而不是复刻矩阵核 | [`1cat-v2`](docs/design/z100/research/z100-1cat-transfer-research-v2.md) |

### 12.2 调研但未落地

- **balanced prefill chunks / mixed-prefill budget**：1Cat 有实测收益，但 Z100 的 scheduler
  更新，且 c=1 下收益不确定；只保留在调研文档，未默认启用。
- **DFlash2 `draft_block` / lookup 增广**：参考了 1Cat 的 `speculator.py` / `lookup.py`，
  但 Z100 已有更并行的 ngram scan，q16 复杂度高；未落地。
- **1Cat 的数据面 kernel**：attention / GEMM / 量化内核与 Z100 的指令集和内存层级不匹配，
  明确不做。

### 12.3 未采纳的负结果

- 1Cat 的 mixed-batch decode 有静默损坏先例；Z100 的 c=1 场景也没有收益。
- 1Cat 的持久化 tile runtime 实测变慢；Z100 不做。
- 不照抄 1Cat 的 676 个开关名；Z100 只保留有测量背书的开关。

### 12.4 调研文档

- [`z100-1cat-transfer-research-v2.md`](docs/design/z100/research/z100-1cat-transfer-research-v2.md)
- [`z100-w8a16-prefill-fp16-construction.md`](docs/design/z100/w8a16/z100-w8a16-prefill-fp16-construction.md)
- [`z100-decode-prefill-optimization-design.md`](docs/design/z100/decode-prefill/z100-decode-prefill-optimization-design.md)
- [Z100 文档索引](docs/design/z100/README.md)（含历史归档与已移除清单）

## 13. 致谢 / 许可 / 引用 / 联系

### 致谢

- **原始 vLLM**：[UC Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu)
- **海光 DTK**：DCU 运行时、编译器与驱动
- **1Cat-vLLM**：[控制面与测量方法参考](https://github.com/1CatAI/1Cat-vLLM)；
  未移植其数据面 kernel

### 许可证

Apache License 2.0，详见 [LICENSE](LICENSE)。

### 引用

如果你在研究中使用了本分支，请同时引用原始 vLLM 论文并说明 gfx906 适配：

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

### 联系方式

- **仓库**：[sibadaaa/z100-vllm](https://github.com/sibadaaa/z100-vllm)
- **问题反馈**：[Issues](https://github.com/sibadaaa/z100-vllm/issues)
- **代码提交**：[Pull Requests](https://github.com/sibadaaa/z100-vllm/pulls)
