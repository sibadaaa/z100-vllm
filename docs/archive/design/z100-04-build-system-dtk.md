# S4 DTK 构建系统（requirements / setup.py / CMake）

> 上层文档：[适配方案](../海光Z100-DTK26.04-gfx906-vLLM适配方案.md) · [Todolist](../z100-todolist.md)
> 前置：[S1](z100-01-arch-normalization.md) · 状态：⬜ 待做

## 目的

让本 fork 在 DTK 容器里从源码构建出**只含 gfx906 目标码**、且带 `_rocm_C` 的 vLLM。das 预编译轮子剥掉了 `_rocm_C`（`LLMM1`/`wvSplitK`/`wvSplitKQ`/`paged_attention_rocm`），自编译是当前唯一能补齐它的途径，也是"支持差"的物理来源之一。

## 前置条件

- S0/S1 完成；`source tools/dcu/env.sh` 可用（已修 `rsmi_init` 与 `'cmath' file not found`）。
- **Python 3.10**：所有 DCU 轮子只有 cp310（`torch-2.11.0+das.opt1.dtk2604-cp310`、`triton-3.6.0+das185...-cp310`）。
- 构建目录必须在 `/mnt/game`（根分区约剩 46 GB；fork 的 csrc 全量构建数 GB）。

## 做法

### 1. 新增 DCU requirements（不要修改 rocm*.txt）

新建 `requirements/dcu-build.txt`、`requirements/dcu.txt`、`requirements/build/dcu.txt`，以 `-r common.txt` 为基，然后：

- 删除 `--extra-index-url https://download.pytorch.org/whl/rocm7.2`、`torch==2.11.0+rocm7.2`、`triton-rocm==3.6.0`、`amdsmi==7.0.2`（DTK 自带 SMI，不需要 python amdsmi）；
- 用 DTK 直链固定轮子（cp310）：
  - `torch-2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7-cp310-cp310-manylinux_2_28_x86_64.whl`
  - `triton-3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1-cp310-...whl`
  - 来源目录：`https://download.sourcefind.cn:65024/directlink/4/pytorch/DAS1.8.5/`、`.../triton/DAS1.8.5/`
- 保留 `requirements/common.txt` 的其余依赖。

> 参考基线（不直接使用）：官方 `vllm-0.21.0+das.dtk2604.torch2100` 轮子在 `.../vllm/DAS1.8.5/`。

### 2. `setup.py` 增加 DTK 分支

| 改动点 | 要求 |
|---|---|
| 环境识别 | DTK 判据与 `hygon_dtk.is_hygon_dtk()` 保持一致：`ROCM_PATH` 指向 dtk / `/opt/dtk/.info/rocm_version` 存在 / `DTKROOT`、`HYHAL_PATH` 已设 |
| requirements 选择 | DTK 时读 `dcu.txt` / `dcu-build.txt`，否则保持 `rocm.txt` |
| `_is_hip()` 兜底 | 若 `torch.version.hip is None` 但 DTK 存在，按 HIP 处理并 `logger.warning`（DTK 的部分 torch 构建不填 `hip` 字段） |
| 版本后缀 | `+rocmXX` → `+dtk{ver}`（`ver` 取 `/opt/dtk/.info/rocm_version`） |
| `ROCM_PATH` 传递 | 已有 `-DROCM_PATH={ROCM_HOME}`；确认 DTK 下 `ROCM_HOME=/opt/dtk`（torch 可能报 `/opt/dtk`，否则显式导出 `ROCM_PATH=/opt/dtk`） |

### 3. CMake

- `HIP_SUPPORTED_ARCHS` 已含 `gfx906`，无需改。
- **必须显式**设置 `PYTORCH_ROCM_ARCH=gfx906`：否则 CMake 会从 `rocm_agent_enumerator` 取到 `/opt/dtk/bin/target.lst` 的 `gfx906;gfx926;gfx928`，编出多余架构（体积 ×3、还可能踩 gfx928 指令）。
- 若 CMake 找不到 ROCm：容器内 `ln -sfn /opt/dtk /opt/rocm`（宿主不要动）。
- 若 hipcc 仍报 C++ 头找不到：`-DCMAKE_HIP_FLAGS="-I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11"`（`env.sh` 已通过 `CPLUS_INCLUDE_PATH` 处理，这是兜底）。

### 4. 构建命令

```bash
# 在 DTK 容器内
source /opt/dtk/env.sh && source /home/lyh/z100-vllm/tools/dcu/env.sh
export BUILD_DIR=/mnt/game/port/build-gfx906
export MAX_JOBS=6 CMAKE_BUILD_PARALLEL_LEVEL=6
export PYTORCH_ROCM_ARCH=gfx906
cd /mnt/game/port/gfx906-vllm
pip install --no-build-isolation --no-deps -e . 2>&1 | tee /mnt/game/port/build.log
dmesg | grep -i oom          # 编译 OOM 兜底检查
```

首次全量编译预计 1.5–3 h（无 ccache）；`-j` 从 4 起，观察内存再调。

## 验收（门禁）

```bash
python -c "import vllm, vllm._C, vllm._rocm_C; print('IMPORT_OK')"
dccobjdump --offloading $(python -c "import vllm._C,os;print(vllm._C.__file__)") | grep -o "gfx[0-9]*" | sort -u
# 期望：只有 gfx906
python -c "import torch;print(hasattr(torch.ops,'_rocm_C')) "
```

## 失败处置

| 现象 | 处置 |
|---|---|
| `fatal error: 'cmath' file not found` | 未 `source tools/dcu/env.sh`；或按上面 `CMAKE_HIP_FLAGS` 兜底 |
| `none of the detected ROCm architectures ... supported` | 显式给 `PYTORCH_ROCM_ARCH=gfx906` |
| 编译 OOM / 磁盘满 | 降 `MAX_JOBS`；清理 `/mnt/game/port/build-*`；确认没把 build 落在 `/` |
| 混装了社区 rocm 轮子的 torch | 全部卸掉重装 DTK 轮子；**DTK 与社区 ROCm 不可混用** |

## 参考

- 前序计划 §5.4 构建配方（含 `/opt/rocm` 链接、MAX_JOBS、磁盘告警）
- [tools/dcu/README.md](../../tools/dcu/README.md)
