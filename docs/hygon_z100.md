# 海光 Z100（DTK / gfx906）适配记录

> 相关：[适配方案](海光Z100-DTK26.04-gfx906-vLLM适配方案.md) · [适配 Todolist](z100-todolist.md) · [分步设计文档](design/)

本文件记录 `gfx906/main` 分支适配海光 DCU Z100 的实测依据、设计取舍与验证方式。

## 0. 结论摘要

Z100 在 ISA 层就是 **gfx906**（Vega20 / GCN5.1）。分支内已有的 382 处 `gfx906` 门控与专用内核在 Z100 上语义仍然成立，**不需要重写**；真正需要适配的是 **DTK 运行时栈与社区 ROCm 的差异**：路径、探测、内存统计、缺失组件。

因此实现采用两条正交轴：

| 轴 | 判据 | 作用 |
|---|---|---|
| ISA 轴（既有） | `on_gfx906()` / `on_cdna()` / `on_gfx9()` | 算力特性门控，语义不动 |
| 运行时轴（新增） | `vllm.platforms.hygon_dtk.is_hygon_dtk()` | DTK 栈差异，全部为附加分支 |

新增代码集中在 `vllm/platforms/hygon_dtk.py`，其余改动都是在已存在的失败点补 fallback，未安装 DTK 的机器行为不变。

bf16 方面（[真机判定归档](result/z100-03-bf16-verification-result.md)）：gfx906 无原生 bf16 ISA，但真机探针判定**数值可用**（bf16 GEMM 相对误差 0.229%）；实现上保留 bf16 为**可选** dtype（`VLLM_DCU_BF16=auto` 默认、fp16 仍排第一），实测 bf16 明显更慢（4096³ 仅 0.36×），故**不设为默认**。

架构归一化方面（[S1 归档](result/z100-01-arch-normalization-result.md)）：海光 SMI 会把 Z100 报成 `gfx916`，必须**归一化**为 `gfx906` 而不是 allow-list（否则 `capability` 停在 `(9, 1)`，静默废掉所有 `minor == 0` 门控）。

构建与运行时基线（详见 §7）：目标栈已升到 **torch 2.11.0+das.opt1.dtk2604 + 自编 triton 3.6.0+git7c56a5e4**，四个扩展全部只含 gfx906；`_rocm_C` 的 paged-attention / skinny-GEMM 在 gfx906 上是 MFMA-only 桩（已短路），**attention 走 Triton（已打通，39/12 测试全过）**；另有重大发现：**gfx906 的 SFU hazard 会让 Triton 静默算错，既有 3.4 生产轮子同样中招**，自编补丁已修。

## 1. 本机环境核验记录（2026-10-07）

| 检查项 | 实测结果 |
|---|---|
| DTK | `/opt/dtk` → `/opt/dtk-24.04.3`，`/opt/dtk/.info/rocm_version` = `24.04.3`，`hipconfig` = HIP 5.7.24405，clang 15.0.0 |
| 社区 ROCm | `/opt/rocm` **不存在**；`rocminfo`/`hipcc` 仅在 `/opt/dtk/bin`、`/opt/dtk/llvm/bin` |
| DCU 数量/容量 | 4 张，各 32 GiB（`/sys/class/drm/cardN/device/mem_info_vram_total` = 34342961152） |
| 每卡架构 | `/sys/class/kfd/kfd/topology/nodes/{1..4}/properties` → `gfx_target_version 90106`、`simd_count 240`；PCI DID `0x55B7`×2 / `0x54B7`×2；`hy-smi` 报 `ZIFANG`×2 / `Z100`×2 → **4 张卡 ISA 同为 gfx906**，仅板卡 SKU 不同 |
| 干扰设备 | `/sys/class/drm/card0` 是 NVIDIA GTX 1060（`0x10DE:0x1C03`, nouveau）。DRM 序号与 HIP 序号**不一致**，必须按 vendor + PCI BDF 映射（见 `_dcu_drm_cards()`） |
| DTK 目标列表 | `/opt/dtk/bin/target.lst` = `gfx906;gfx926;gfx928`；无 KFD 权限时 `rocm_agent_enumerator` 输出的**就是该列表**，不能当作设备信息 |
| rocBLAS | `lib/rocblas/library/` 含 `Kernels.so-000-gfx906.hsaco` / `TensileLibrary_gfx906.co` → gfx906 覆盖良好 |
| hipBLASLt | `lib/hipblaslt/library/` 只有 `gfx928` → **gfx906 不可用**，必须避开该路径 |
| amdsmi | 本机 **无** python `amdsmi` 模块；DTK 26.04 上可导入但 `target_graphics_version` 返回 `gfx18446744073709551615` |
| `torch.version.hip` 死锁点 | `torch.cuda.get_device_properties("cuda")` 在 DTK 上会死锁，必须用整数索引 `get_device_properties(0)` |
| GPU 访问 | agent 会话内 `/dev/kfd` 打开被拒（`EACCES`，沙箱限制），宿主侧无法直接跑 GPU；**真机验证改为在一次性 Docker 容器内进行**（`docker inspect` 只读照抄生产容器的设备参数，绝不触碰生产容器） |
| Docker | 可用（用户在 `docker` 组）。生产容器 `qwen38-vllm`（:8000）禁止 exec/重启/commit；影子容器用完 `docker rm -f` |
| 宿主机资源 | 内存 31 GB（可用常仅 ~7 GB，无 swap）、12 vCPU；根分区剩 ~44 GB，构建产物一律落 `/mnt/game`（剩 ~680 GB） |

### 1.1 真机容器要点与高危坑（[S3 归档](result/z100-03-bf16-verification-result.md)）

| 项 | 实测 |
|---|---|
| 容器基线 | 生产镜像 `custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`；容器内 torch `2.10.0`、HIP `6.3.26093`、`gcnArchName=gfx906:sramecc+:xnack-`、`get_device_capability()=(9,0)` |
| 版本口径 | 生产镜像内是 torch 2.10.0，而适配目标栈（`requirements/dcu*.txt`）固定 torch 2.11.0+das.opt1.dtk2604 / triton 3.6.0+das185；构建时必须记录实际用哪套 |
| **`ROCBLAS_TENSILE_LIBPATH` 是毒药** | 该目录只有 lazy 库 `TensileLibrary_lazy_gfx906.dat`；显式导出该变量会让 rocBLAS 去找不存在的 `TensileLibrary.dat` → `rocBLAS error: Could not initialize Tensile host: std::bad_alloc` + SIGABRT/SIGSEGV，**任何** GEMM（含 64×64）都崩。生产容器不设该变量；`tools/dcu/env.sh` 已**删除**该导出（A/B/A 确证因果） |
| 探针导入顺序 | 手写一行式探针必须**先 `import torch`** 再 `from vllm.platforms import rocm`，否则在 DTK 上稳定 `std::bad_alloc` abort（`check_bf16.py` 与 vLLM 正常启动不受影响，它们先导 torch） |
| 显存现状 | 生产容器占 4 卡各 ~28.3 GB（总 32 GB），真机实验只剩 ~4.4 GB/卡 |

## 2. bf16 专项结论

**结论：gfx906 / Z100 没有原生 bf16 指令，"是否可用"必须由设备实测决定，不能硬编码。因此该条已从"恒返回 fp16/fp32"改为"探针 + 显式开关"。**

### 2.1 ISA 证据（DTK 自带 LLVM 15 汇编器，本机可复现）

```bash
CLANG=/opt/dtk/llvm/bin/clang
printf '\tv_cvt_f32_bf16 v0, v1\n\ts_endpgm\n' > /tmp/t.s
$CLANG -target amdgcn-amd-amdhsa -mcpu=gfx906 -c /tmp/t.s -o /tmp/t.o
```

| 指令 | gfx906 | 汇编器输出 |
|---|---|---|
| `v_pk_fma_f16` | ✅ 支持 | OK |
| `v_dot2_f32_f16` | ✅ 支持 | OK |
| `v_cvt_f32_bf16` | ❌ 无此指令 | `error: invalid instruction, did you mean: v_cvt_f32_f16?` |
| `v_dot2_f32_bf16` | ❌ 该 GPU 不支持 | `error: instruction not supported on this GPU` |

即：gfx906 连 bf16 的**转换指令**都没有，bf16 张量的数学只能是"位运算转换 + fp32 计算"的软件模拟。模拟可以数值正确，但通常比 fp16 慢（fp16 有原生 packed FMA 与 dot）。

### 2.2 两个常见误判来源

- `torch.cuda.is_bf16_supported()`：在 ROCm 构建上**恒返回 True**（见 [ROCm/pytorch commit 363320f](https://github.com/ROCm/pytorch/commit/363320f6b31645bf26f890ebc64e6ce11993a8cd)），不反映 gfx906 缺 bf16 ISA，**不作为判据**；`tools/dcu/check_bf16.py` 只把它打印出来做对照。
- `torch.set_default_dtype(torch.bfloat16)`：它只改变**新建 float 张量的默认 dtype**，不是能力开关，也不会让 vLLM 走 bf16（vLLM 的 dtype 由 `--dtype` / 模型配置 + `supported_dtypes` 门控决定）。正确的显式开关是 `--dtype bfloat16` 配合下面的环境变量。真机实测（[S3 归档 §3.5](result/z100-03-bf16-verification-result.md)）：它只改变新建浮点张量（含 `nn.Linear` 权重）的默认 dtype，整型默认值、后端标志、`get_device_capability()` 全部不变；`is_bf16_supported()` 即便带 `including_emulation=False` 仍返回 `True`，在无 bf16 ISA 的 gfx906 上同样不可信。

### 2.3 已落地的实现

| 位置 | 行为 |
|---|---|
| `VLLM_DCU_BF16=auto`（默认） | 运行探针 `dcu_bf16_probe()`：bf16 存储往返、bf16 GEMM 对比 fp32 参考、bf16 softmax；全部通过才认为可用 |
| `VLLM_DCU_BF16=1` | 强制开启（用于实验），即使探针失败也让 bf16 进入 `supported_dtypes` |
| `VLLM_DCU_BF16=0` | 强制关闭 |
| `RocmPlatform.supported_dtypes` | gfx906 上 fp16 仍排第一（`--dtype auto` 行为不变）；探针/开关通过时才在中间插入 `bfloat16` |

### 2.4 已知约束

- gfx906 的 MTP / GDN 专用内核通过 `VLLM_DISPATCH_HALF_TYPES`（仅 fp16）派发（如 `csrc/mamba/gfx906_decode_kernels.cu` 的 `causal_conv1d_gfx906_mtp_update`、`fused_sigmoid_gating_delta_rule_gfx906_mtp_*`）。因此 **bf16 + MTP 会以明确的 "not implemented for BFloat16" 报错失败，不会静默算错**；阶段 B 若要支持，需把这些内核扩展到 bf16 派发。
- bf16 的 KV cache / 量化路径未在本次改动范围内验证。

### 2.5 在真实 DCU 上的验证与判定（2026-10-07 完成）

```bash
# 一次性容器内（照抄生产容器设备参数；生产容器本身不动）
source tools/dcu/env.sh
python3 tools/dcu/check_bf16.py ; echo "exit=$?"
```

**判定：`exit=0`（四项全 PASS），bf16 保留在 `supported_dtypes`，`VLLM_DCU_BF16` 默认维持 `auto`，`rocm.py` / `hygon_dtk.py` 无需改动。**

| 项目 | 实测（gfx906 / Z100，DTK 26.04 容器，两次全新容器重跑逐字一致） |
|---|---|
| bf16 存储往返 | `max_abs_err=0.011368` ✅ |
| bf16 GEMM vs fp32 参考 | `max_abs_err=0.249939`，**rel=0.229%** ✅ |
| `linear(bf16 × bf16)` | `rel=0.173%` ✅ |
| `softmax(bf16)` | `max_abs_err=0.000025` ✅ |
| 4096³ 矩阵乘 | fp16 **7.84 ms** vs bf16 **21.54 ms** → bf16 仅 **0.36×** |
| `F.linear` NT 各 M | M=1/M=8 相当，M=128 慢 1.5×，M=2048 慢 2.3× |
| `_rocm_supported_dtypes()` | `auto` → `[fp16, bf16, fp32]`；`0` → `[fp16, fp32]`；`1` → `[fp16, bf16, fp32]` |

- 因为 fp16 恒排第一，**`--dtype auto` 的行为不因本判定改变**；变化只在于显式 `--dtype bfloat16` / bf16-only 权重在 gfx906 上开箱可用。
- 不建议把默认改为 `1`（绕过探针且更慢），也不建议改为 `0`（会关掉 bf16-only 模型的可用性）；`VLLM_DCU_BF16=0` 保留为保守逃生门。
- 退出码约定仍有效：`0` = 可用、`1` = 不可靠（保持 `--dtype float16`）、`2` = 无 GPU/不可判定。
- `torch.set_default_dtype(torch.bfloat16)` **不是**能力开关（§2.2），不要用它来"启用 bf16"。
- 遗留：与生产基线的完整 A/B 对拍（同模型/同 prompt）放到 S9；设计文档早期引用的"生产 bf16 GEMM ≈ 8.2 ms"与本次 4096³ 实测不符，口径待 S9 用 profiler 复核。

## 3. 本轮代码改动

| 文件 | 改动 |
|---|---|
| `vllm/platforms/hygon_dtk.py`（新增） | DTK 探测、`VLLM_DCU_GCN_ARCH` 覆盖、GCN arch 校验（拒绝 `gfx18446744073709551615`）、KFD sysfs 解析（跳过 CPU 节点）、按 vendor+BDF 映射的 DCU sysfs 显存/UUID、bf16 探针与策略、`rocminfo` 候选路径 |
| `vllm/platforms/rocm.py` | arch 查询改为可校验多级（DTK 走 helper；社区 ROCm 保持 amdsmi 优先，但校验返回值）；`get_device_properties("cuda")` → `(0)`；`with_amdsmi_context` 容忍 amdsmi 缺失；`get_device_name/uuid/total_memory` 增加 sysfs fallback；`is_fully_connected` 在无法判定时返回 False（保留标准 RCCL allreduce）；gfx906 的 bf16 门控改为探针驱动 |
| `vllm/platforms/__init__.py` | `rocm_platform_plugin()` 在 DTK 上直接选择 ROCm 平台（否则 DTK 上 amdsmi 缺失 + 无 `/opt/rocm/bin/rocminfo` 会导致**无平台可用**）；rocminfo fallback 支持 DTK 路径 |
| `vllm/envs.py` | 新增 `VLLM_DCU_GCN_ARCH`、`VLLM_DCU_BF16` |
| `tools/dcu/env.sh`（新增） | DTK 环境引导：过滤出真正导出 `rsmi_init` 的 `librocm_smi64`；补齐 gcc C++ 头文件搜索路径；按 KFD 拓扑导出 `VLLM_DCU_GCN_ARCH`；Triton 缓存/lld；gfx906 关闭 skinny GEMM；DCU 可见性；**不导出 `ROCBLAS_TENSILE_LIBPATH`**（真机实测会崩，见 §1.1） |
| `tools/dcu/check_bf16.py`（新增） | bf16 数值与性能实测脚本 |
| `tests/platforms/test_hygon_dtk.py`（新增） | GPU-free 单测：arch 校验、KFD 解析（含 CPU 节点）、DRM 过滤与 BDF 排序、bf16 策略、dtype 顺序 |

## 4. 已完成的验证

| 验证 | 结果 |
|---|---|
| GPU-free 单测 | 基线 **29 passed / 1 skipped**；S1 落地后 **49 passed / 0 failed / 3 skipped**（stub harness 实跑；3 skip = 本机无 torch/regex） |
| S1 架构归一化（真机等价路径） | stub 加载**真实 `rocm.py`**：设备报 `gfx916` → `_GCN_ARCH=gfx906`、capability `(9,0)`、`on_gfx906=True on_cdna=True on_gfx9=False`；`gfx942`→`(9,4)`、`gfx90a` 行为不变 |
| bf16 真机判定 | 一次性容器内 `check_bf16.py` **exit=0**；GEMM rel 0.229%、linear 0.173%、softmax 2.5e-05；bf16 比 fp16 慢（4096³ 0.36×） |
| sysfs 逻辑真机核对 | 4 张卡全部 `vram=34342961152`、`arch=gfx906`，UUID 可读；`card0`（NVIDIA）被正确排除 |
| hipcc 头文件问题 | 未修复前 `fatal error: 'cmath' file not found`；`source tools/dcu/env.sh` 后同一 `.hip` 文件编译成功（0 error，产出 `.o`） |
| `llvm-mc` 缺失 | DTK 未提供 `llvm-mc`，改用 `clang -target amdgcn-amd-amdhsa -mcpu=gfx906 -c` 验证 ISA |
| `ROCBLAS_TENSILE_LIBPATH` | A/B/A 确证：导出即崩（任意 GEMM）；`env.sh` 已删除该导出 |

## 5. 待办（阶段 A 剩余）

| 项 | 内容 |
|---|---|
| 已归档步骤 | **S1 架构归一化 ✅**（[归档](result/z100-01-arch-normalization-result.md)）· **S3 bf16 真机判定 ✅**（[归档](result/z100-03-bf16-verification-result.md)）· S4 构建系统（执行中） |
| A2 | `setup.py` DTK 探测与 requirements 选择、版本后缀、`_is_hip()` 兜底；`CMakeLists.txt` 把 gfx906 加入 skinny GEMM 排除集；`scaled_mm/rocm.py` 的 gfx1250 判据扩展到 gfx906 |
| A1 | `requirements/dcu*.txt`；`tools/dcu/README.md` |
| A4 | DTK 26.04 下 csrc 全量试编；DCU triton 3.6 的 gfx906 支持验证（失败则按 VEGA ISAFamily 补丁自编）；CUDA graph 实测 |
| A5 | `MemorySnapshot` 在 DTK 上的峰值钳制；`gpu_worker.py` 的 KV cache 兜底扩展到 DTK；TP/RCCL 冒烟 |
| A6 | 冒烟脚本、`diagnostic_runner.py` / `collect_env.py` 扩展、README 章节 |
| 真机验收 | A1–A8 判据（本文档 2.5 与方案中的验收表）现已可在**一次性容器**内执行（见 §1.1）；生产容器 `qwen38-vllm` 保持不动 |

## 6. 真机验证步骤（在有 DTK 26.04 与 `/dev/kfd` 权限的环境）

```bash
# 1. 环境
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -r requirements/dcu-build.txt          # A1 落地后
source tools/dcu/env.sh

# 2. 前提核验
python3 -c "import torch;print(torch.__version__, torch.version.hip)"
python3 -c "import torch;[print(i, torch.cuda.get_device_properties(i).gcnArchName) for i in range(torch.cuda.device_count())]"
python3 tools/dcu/check_bf16.py

# 3. 构建（显式限定 gfx906，避免 DTK target.lst 把 926/928 一起编进来）
PYTORCH_ROCM_ARCH=gfx906 MAX_JOBS=$(nproc) uv pip install --no-build-isolation -e .

# 4. 平台与 bf16 门控
python3 -c "from vllm.platforms.rocm import _GCN_ARCH, on_gfx906, _rocm_supported_dtypes; print(_GCN_ARCH, on_gfx906(), _rocm_supported_dtypes())"

# 5. 服务冒烟（fp16 为主，bf16 对照）
vllm serve Qwen/Qwen3.5-0.8B --dtype float16 --max-model-len 4096
vllm serve Qwen/Qwen3.5-0.8B --dtype bfloat16 --max-model-len 4096   # 需 probe 通过或 VLLM_DCU_BF16=1
```

## 7. 目标栈基线与自编 gfx906 Triton（S4b/S4c/S4d/S6）

### 7.1 目标栈（2026-10-07 决策：完全按方案升级）

| 项 | 值 |
|---|---|
| torch | `2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7`（看 `torch.version.__hcu_version__`） |
| triton | **自编** `3.6.0+git7c56a5e4`（上游 `triton-lang/triton` v3.6.0 / commit `7c56a5e` + 本项目补丁） |
| HIP | 6.3.26113；4×Z100L gfx906 |
| 构建容器 | `z100-build-211`（`--memory=8g`）；副本 `/mnt/game/port/gfx906-vllm-211`；构建树 `/mnt/game/port/build-211` |
| 回滚基线 | 容器 `z100-build`（torch 2.10.0+das）与 `/mnt/game/port/gfx906-vllm`，**不要动** |

四个扩展全部编成且只含 gfx906：`_C` 78,102,728 B、`_rocm_C` 35,156,368 B、`_C_stable_libtorch` 40,688,664 B、`_moe_C_stable_libtorch` 29,748,928 B。升级动因之一：`_C_stable_libtorch` 原先卡在 `csrc/libtorch_stable/cuda_view.cu` 的 `torch::stable::Tensor::layout()`（镜像 torch 2.10 无此 API），torch 2.11 下消失。

**运行时 `torch.ops._C` 实际由 `_C_stable_libtorch` 提供**（`TORCH_TARGET_VERSION` 稳定 ABI 路径）：S7c 实测只有重编 `_C_stable_libtorch` 才能改掉 `_C.silu_and_mul` 的行为，而 `vllm/_C.abi3.so` 逐字节未变。改 `csrc/libtorch_stable/**` 时要重编/安装的是 `_C_stable_libtorch`。

**口径修正**：DTK 轮子的 `torch.__version__`/`triton.__version__` 只报基础版号（`2.11.0`/`3.5.0`）→ 判定版本必须用 `importlib.metadata`。

### 7.2 自编 Triton（S6）

- 源码/构建树/安装前缀/缓存全在 `/mnt/game/triton-gfx906/`（约 12 GB），`pip install --target`，**DTK 原装 triton 未触碰**。用法：
  `export PYTHONPATH=/mnt/game/triton-gfx906/install:$PYTHONPATH`、`export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache`（别写 `/`）。
- 补丁 [`tools/dcu/triton-gfx906-3.6.0.patch`](../tools/dcu/triton-gfx906-3.6.0.patch)（5 文件 48 行）：VEGA ISAFamily 3 文件（`isCDNA()` 保持 false、`supportsVDot()` 含 VEGA）+ `driver.py`（DTK 的 `hipDeviceProp_t` 是 792 B / `gcnArchName@396`，vendored 头是 1472 B/1160，必须用 DTK 头编译，否则设备属性乱码）+ `compiler.py`（SFU hazard，见 7.3）。
- 门禁：探针 `TRITON_OK`/`REDUCE_OK`/`INT8DOT_OK`/32 条 `v_dot`；`test_triton_unified_attention_gfx906.py` **39 passed**、`test_triton_unified_attn_gfx906_v1.py` **12 passed**；S5 遗留的 14 个 `unsupported target: 'gfx906'` **14/14 消除**，五文件同口径 **74 passed / 0 failed**。
- `AMDGCN_USE_BUFFER_OPS=0` 与 hcu 后端（缺 `/opt/dtk/aillvm`）都不再需要——自编译后 Python 与 `libtriton.so` 同源。

### 7.3 SFU hazard（高优先级：既有生产轮子同样中招）

只打 VEGA 补丁时，10 项 attention 测试仍数值错（含非确定）：**gfx906 的 SFU 与紧邻的前一条 VALU 指令存在 hazard**，`tl.exp2` / `tl.sqrt` 在每个 SIMD16 的第 8–11 道会读回 VALU 写入前的旧值。6 行最小复现（`repro9.py`），4 张卡一致，**且生产验证过的 triton 3.4.0+gitc817b9b6 轮子同样复现 ⇒ 不是本次引入，而是 VEGA 路线一直带着的地雷**。修复：gfx906 时在 SFU 指令前插 `s_nop 2`（`insert_sfu_waits`，已验证该指令进入二进制、修复后全矩阵 OK）。

⇒ **gfx906 上跑 Triton 必须带这条修复；现有 3.4 产线/镜像的数值正确性需重新评估（S9）。**

踩坑：triton 的 `compile_module_from_src` 与 JIT 磁盘缓存**不含 `include_dirs` 与后端 `.py` 的哈希** → 改这些之后必须清 `TRITON_CACHE_DIR`，否则命中旧 `.so`/`.hsaco`。

### 7.4 构建可增量性（S4d）

`hipify_all` 原先从不创建 stamp（COMMAND 里没有 `cmake -E touch`），导致每次 `ninja` 都重跑 hipify 并全量重编（~16 min）。现在 [`cmake/utils.cmake`](../cmake/utils.cmake) 用 `add_custom_command(OUTPUT .../hipify_all.stamp)` + staging 树 + `cmake -E copy_directory_if_different`（按内容同步、mtime 不动）+ 每个 `.hip` 加 `OBJECT_DEPENDS`。实测：无改动 `ninja: no work to do` **0.085 s**、只改 `hipify.py` 6 s、单 TU 40–43 s、全量 963 s；四个**构建树原始** `.so` 与基线逐字节一致。注意 ninja 仍按 mtime 判 clean（见 todolist 硬约束 11）。

### 7.5 已在仓库修掉的 DTK 工具链缺陷（S5/S4c）

| # | 现象 | 修复位置 |
|---|---|---|
| 1 | DTK rocPRIM `uninitialized_array::emplace` 的 placement `new` 在 `__host__ __device__` 函数里被 clang 18 拒绝 | 新增 [`csrc/dtk_rocprim_compat.h`](../csrc/dtk_rocprim_compat.h)，仅在 `ROCM_PATH MATCHES "dtk"` 时 `-include` 注入 |
| 2 | `__syncwarp` 重定义（DTK HIP 6.3 已自带该内建） | [`csrc/fused_qknorm_rope_kernel.cu`](../csrc/fused_qknorm_rope_kernel.cu) 与 `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu` 加 `!defined(HIP_ENABLE_WARP_SYNC_BUILTINS)` |
| 3 | `atomicAdd(half/half2*)` 与 DTK 原生重载二义 | 两份 `quantization/gptq/compat.cuh` 用 `__DTK_ARCH__` 跳过软件模拟 |
| 4 | DTK 的 hipify 补丁注入 `#include <ATen/dtk_macros.h>`，在 `TORCH_TARGET_VERSION` 下 `#error` | [`cmake/hipify.py`](../cmake/hipify.py) 后处理就地展开 `__DTK_ARCH__` 并去掉该 include |
| 5 | torch 2.11 下 hipify 的 `CUDA_MACROS_MAP` 值是裸字符串，`value[0]` 把 `CUDA_VERSION` 变成 `D` | `cmake/hipify.py` 在导入 `hipify_python` 前元组化（对 2.10 的元组形式 no-op） |

### 7.6 真机地雷（已摘除）

`_rocm_C::paged_attention` 与 `wvSplitK`/`wvSplitKQ` 在 gfx906 上都是 MFMA-only 的 `UNREACHABLE_CODE` 桩（真机 device assert + VMFault 复现）。前者原先被 `use_rocm_custom_paged_attention()` 通过 `on_cdna()` 的 `"gfx9" in "gfx906"` 子串误选 → 已短路；后者已从 gfx906 构建里排除，并加了 `ROCM_SKINNY_GEMM_ENABLED` 兜底。**attention 因此只能走 Triton（S6 已打通）。**

### 7.7 遗留

- `tests/model_executor/test_qwen3_5_gdn.py` 在 **采集期**就报 `operator torchvision::nms does not exist`：容器内 torchvision 0.25.0 是为 torch 2.10 构建的，与 torch 2.11 不匹配（与 S6 无关，需装匹配 2.11 的 torchvision 才能复评 S5 的那 2 个替身失败）。
- `python -c "import vllm"`（未先 `import torch`）仍 `std::bad_alloc` → SIGABRT(134)：已知、非回归，所有脚本/门禁一律先 `import torch`。
- SFU 修复的性能代价（每条 SFU 多 2 周期）未做吞吐对比。

### 7.8 单卡内存 / KV cache / graph 与 `--enforce-eager` 的 NaN（S7 + Lead 复核）

Lead 用 [`tools/dcu/s7_gate.sh`](../tools/dcu/s7_gate.sh) 在 `z100-serve` 容器里**独立复跑**（单卡 0、`Qwen3-0.6B`、fp16、`max-model-len 2048`、`gpu_memory_utilization=0.9`），与 S7 的结论逐项一致：

| 模式 | `Available KV cache memory` | `GPU KV cache size` | 输出 | decode |
|---|---|---|---|---|
| 默认（`VLLM_COMPILE` + `FULL_AND_PIECEWISE` graph） | 26.75 GiB | 250,384 tokens（2048 时并发 122.26×） | `' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'` | **12.02 ms/token** |
| `--enforce-eager`（`CompilationMode.NONE`） | 27.18 GiB | 254,464 tokens（124.25×） | token id 全是 `1023`（即 `"lock"`），复读 | 36.87 ms/token |

- graph 捕获成功（PIECEWISE 51/51 + FULL 35/35，8–10 s、0.36 GiB）；KV cache 由引擎自动算出，**不需要** `--kv-cache-memory-bytes`，也未触发 gfx906 的 0.98 兜底；`vllm/utils/mem_utils.py`、`vllm/v1/worker/gpu_worker.py` **零改动**。
- **Lead 复核时抓到的关键线索**：两种模式的 `kernel_config` 不同 —— compile 模式是 `rms_norm=['native']` / `fused_add_rms_norm=['native']`，而 eager 模式是 `rms_norm=['vllm_c','native']` / `fused_add_rms_norm=['vllm_c','native']`。即 `CompilationMode.NONE` 会让 `ir_op_priority` 选中 **`vllm_c`（`_C` 里的自定义 RMSNorm 内核）**，这是 S7b 要验证的第一嫌疑。
- 结论：**gfx906 上禁用 `--enforce-eager`**（todolist 硬约束 13）；S8/S9 一律用默认 compile 模式。serve 冒烟（:8001，容器 bridge）`/health` 200、chat/completions 200、8 路并发 8/8 成功。

### 7.9 NaN 根因：`_C.silu_and_mul` 的 fp16 packed 分支被编坏（S7b）

| 项 | 结论 |
|---|---|
| 首个异常位置 | `model.layers.0.mlp.act_fn`（`SiluAndMul`）：6/15360 元素非有限、absmax 5320（输入 absmax 5.922），随后经 `down_proj` 扩散成全层 NaN |
| 根因算子 | `_C.silu_and_mul` 与 `_C.silu_and_mul_with_clamp`（**不是** attention / KV cache / rms_norm / 编译模式） |
| 具体位置 | `csrc/libtorch_stable/activation_kernels.cu:110-159` 的 **fp16 `__half2` 向量化分支**（`packed_compute<..., act_first=true>` → `packed_silu_kernel`, `:171`；profiler 反解出的 kernel 名 `act_and_mul_kernel<half,__half2,silu_kernel,packed_silu_kernel,true,true,false,false>`）在 DTK 26.04 + gfx906 上被编坏 |
| 边界（10 次/项实测） | **坏**：fp16 向量化分支（inf/NaN，最多 3708/196608；同输入 20 次坏集不一致 ⇒ **非确定**）。**好**：fp16 标量分支（`d % 8 != 0`）、`mul_and_silu`（`act_first=false`）、`gelu_and_mul`、bf16、fp32 |
| 与 S6 的关系 | 与 gfx906 的 SFU/VALU hazard 同类（都是 DTK 代码生成在 gfx906 上的非确定性错误），但 ISA 机理未证明 |
| 当前修复 | [`vllm/model_executor/layers/activation.py`](../vllm/model_executor/layers/activation.py)：新增 `_broken_gfx906_fp16_packed_silu()`，`SiluAndMul.forward_hip` / `SiluAndMulWithClamp.forward_hip` 在 gfx906+fp16 时回退 `forward_native` |
| 验证 | eager ids == `VLLM_COMPILE` == HF transformers **逐 token 一致**；逐层扫描非有限 385/396 → **0**；6 个 prompt 全正常；eager 34.6→37.4 ms/token（+8%），编译路径不变（32.7） |
| 为什么只有 eager 中招 | 两种模式的 `ir_op_priority` 不同：compile 用 `native`，`CompilationMode.NONE` 用 `vllm_c`（见 7.8） |
| 覆盖缺口 | MoE / GGUF 的**直调点**（`fused_moe/activation.py:356/311/387`、`fused_moe/utils.py:590`、`gguf.py:1720`、minimax `swiglu_oai`）绕过上述 Python 修复：实测 `apply_moe_activation(SILU)` 仍 1917/98304 非有限 ⇒ 需 csrc 根修（S7c）：在 `activation_kernels.cu` 对 **DTK + gfx906 的 fp16 关闭 `use_vec`** |
| 纪律 | gfx906 上**不要**用 `--enforce-eager`（慢 3.2× 且此 bug 只在它上面暴露）；默认 `VLLM_COMPILE` 路径不受影响 |

### 7.10 fp16 packed SiLU 的 csrc 根修（S7c）

| 项 | 结论 |
|---|---|
| 改动 | [`csrc/libtorch_stable/activation_kernels.cu`](../csrc/libtorch_stable/activation_kernels.cu)：新增 host 侧守卫 `dtk_gfx906_fp16_no_packed(dtype)`（`:240-274`），3 处 `use_vec`（`:296-297`、`:1003-1004`、`:1407-1408`）全部加守卫。`csrc/activation_kernels.cu` **无需改**：它没有打包/向量化路径，`_C::silu_and_mul` 只在 `libtorch_stable/torch_bindings.cpp:605/876` 注册 |
| 守卫为什么不能照抄 `__gfx906__` | **host pass 里 `__gfx906__` 未定义**（`hipcc --cuda-host-only -S` 实测），而 `use_vec` 是 host 侧决策 → 用 `#if defined(__DTK_ARCH__)` + 运行时 `strncmp(get_device_prop()->gcnArchName, "gfx906", 6) == 0` + `dtype == Half`（不含 bf16） |
| 负向控制 | 关掉 `__DTK_ARCH__` 时 `use_vec` 的 host 汇编与改动前**逐指令相同**（`diff` IDENTICAL）；真实构建里确有 `#define __DTK_ARCH__ 520` 与 `--offload-arch=gfx906`；静态复证：新 `.so` 的 10 个激活入口各含 1 个 `strncmp@plt`，旧 `.so` 为 0 |
| 非有限计数（改前 → 改后） | `_C.silu_and_mul` fp16（d=3072，连打 20 次）424–455/次 → **0**；`silu_and_mul_with_clamp` 107/24576 → **0**；`apply_moe_activation(SILU)` **1907/98304 → 0**；bf16/fp32/`mul_and_silu` 对照组逐位不变 |
| 端到端 | **关掉 Python 兜底**的树上 eager == graph == HF transformers 逐 token 一致（改前同配置是 `lock`×16）；逐层扫描 396 模块 **0** 非有限 |
| 回归 | 39/12/13/5/5 全过，`gate-211.sh` 与 S4c 基线逐字一致；新增 [`tests/kernels/test_gfx906_packed_silu_fp16.py`](../tests/kernels/test_gfx906_packed_silu_fp16.py)（3 passed；**旧 `.so` 上 3 failed** ⇒ 能锁回归。注意该文件被 `.gitignore:301` 的 `test_*.py` 忽略，固化时需 `git add -f`，见硬约束 14） |
| 产物 | 被修的是 `_C_stable_libtorch`：`cc2aca69…` → **`93bf7b4a391ac97e2f67cc7141633cb1`**（40,691,216 B）；`vllm/_C.abi3.so` 仍 `6fba0596…`。回滚件 `/mnt/game/port/s7c/_C_stable_libtorch.abi3.so.installed-before` |
| 性质 | 与 S6 的 SFU hazard 一样是**等价规避**（走标量分支），ISA 级机理仍未定性；Python 侧兜底保留为 belt-and-braces，注释已说明"csrc 已根修" |

## 8. 阶段 A 验收结论（S9 / S9b）

### 8.1 状态

A1–A6 / A8 / A9 **通过**；A7 只做了 GPU-free（本机没有社区 ROCm 环境）；三项受宿主资源限制未完成（见 8.4）。归档：[S9](result/z100-09-acceptance-and-bench-result.md)、[S9b](result/z100-09b-full-cudagraph-result.md)。

### 8.2 真机跑通的场景

| 场景 | 结果 |
|---|---|
| 0.6B fp16 单卡 | KV **13.98 GiB** 自动算出（无手工 flag）、8K 日志 `TRITON_ATTN`、7295-token 请求 200 / 4.20 s |
| 0.6B TP=2 / TP=4 | 各一次完整请求 |
| **27B W8A8 INT8 TP=2** | 端到端**文本正确**；KV 11.59 GiB / 161,792 tok / 39.5×；默认 `FULL_AND_PIECEWISE` 可捕获（PIECEWISE 51/51 + FULL 35/35，`--cudagraph-metrics` 证明真在 replay） |
| **W4A16**（tinyllama group128） | 走 `Gfx906GPTQWNA16LinearKernel`，端到端文本正确 |
| **GDN**（Qwen3.5-4B） | 端到端跑通（729 s ready，FULL 35/35，HTTP 200） |
| 测试 | 7 个文件 **146 passed / 0 failed**；新增的 gfx906 SiLU 回归测试 3 passed |
| 诊断 | `tools/dcu/s9_diag.py` rc=0、`vllm collect-env` rc=0、`s8b_gate.sh` ALL GREEN 15/15 |

### 8.3 性能归因（**纠正过一次，重要**）

| 指标 | 本树（gfx906 + DTK 26.04） | 生产基线 | 倍率 | 主因 |
|---|---|---|---|---|
| decode（27B INT8，TP=2） | 378.8–380.6 ms/token | 122.47 | **3.09×** | **不是图模式**：FULL 相对 PIECEWISE 只快 0.5%。差异来自内核与并行度——生产是 **TP4 + eager + 自研 `int8_sdot4` + GDN torch 替换**，本树是 **TP2 + Triton INT8 + 原生 FLA/GDN** |
| prefill | 3.508 ms/prefill-token（TP=2，in=8000） | 2.789 | 1.26× | 缺现网那套 GDN prefill 补丁（历史记录 +27% 同源） |
| conc8 吞吐 | 14.78 tok/s | 无对照数据 | — | — |

⇒ **功能正确性已闭环；性能欠账明确落在"内核 + 并行度（TP=4）"上**，不在适配层，属于后续优化工作。

### 8.4 未完成项与原因（都是宿主资源限制，不是实现缺陷）

| 项 | 原因 |
|---|---|
| A7 非 DTK 环境对照 | 本机没有社区 ROCm/CUDA 环境，只能 GPU-free（52 passed） |
| 严格 TTFT@8192 | **TP=4 被宿主内存卡死**：宿主仅 30.5 GiB，容器 32 GiB → 启动 4 min 即 `oom_kill 1`（生产容器 `MemLimit=0` 才能跑 4 卡）。已给 TP=2 `in=8000` 的 per-token 口径替代 |
| 27B 的独立数值参照（HF fp16） | 反量化 27B INT8 到 fp16 需要 >54 GB 内存，本机不可行；小模型（0.6B）上 FULL == PIECEWISE **逐 token 一致** |
| 27B FULL vs PIECEWISE 逐 token 一致 | 3 条 greedy 里 2 条不同（近 tie，margin 0.66 nats）；两模式**各自跨进程完全确定**，差异源自 `cudagraph_mode` 会改变编译产物。**缺独立参照时，生产建议继续 PIECEWISE**（FULL 只快 0.5%，不值得冒险） |

### 8.5 上线与回滚

固化清单与可直接执行的回滚步骤见 [S9 归档](result/z100-09-acceptance-and-bench-result.md) 附录 A/B 与 [S9b 归档](result/z100-09b-full-cudagraph-result.md) 附录 A/B：四扩展/triton 补丁/requirements/门禁脚本的 md5、环境变量、容器清单，以及"退回 2.10 基线 / 撤掉自编 triton / 撤掉 SiLU 修复 / 撤掉 S9b 修复"四条回滚路径。生产容器 `qwen38-vllm` 由用户自行启停，全程只做过只读 `docker inspect`。

## 9. PCIe fabric 与跨卡 P2P：ACS 重定向（2026-10-09，用户提供脚本 + Lead 实测）

### 9.1 现象

驱动层的 P2P **本来就是开的**（`torch.cuda.can_device_access_peer` 12/12 有向组合全 YES、`pcie_p2p=Y`/PeerDirect），**瓶颈在 PCIe fabric 的 ACS 重定向**：4 张 DCU 父链上 **12 个桥的 ACS Control 全是 `0x001d`**（`SrcValid|ReqRedir|CmpltRedir|UpstreamFwd`）。`ReqRedir`/`CmpltRedir` 把 GPU 之间的 peer TLP 重定向到 CPU 根复合体，而本机根端口 `0000:00:03.1` 只有 **PCIe 3.0 x8（≈7.9 GB/s 上限）**，交换机侧却是 Gen4 x16。

### 9.2 实测（同一脚本、同一容器、128 MiB f16 copy 协议）

> **口径更正（2026-10-09，S12 复测）**：Lead 最初记录的"改前 3.12 GB/s / 8–9×"是在 **S11m 的 27B serve 正占用卡 0/1 时**测的（卡位污染，见硬约束 25）；干净复测的改前值是 **6.95 GB/s**，正确倍率是 **4.1×**。下表用 S12 的干净数据。

| 项 | ACS=`0x001d`（改前） | ACS=`0x0000`（改后） | 倍率 |
|---|---|---|---|
| 跨卡 `copy_`（12 个方向） | **6.95 GB/s** | **28.3–28.4 GB/s** | **4.1×** |
| 本卡 D2D（对照） | — | 145–282 GB/s | 本卡基线 |
| `can_device_access_peer` | 12/12 YES | 12/12 YES | **前后都一样**（它不反映真实带宽） |
| **NCCL/RCCL all-reduce** | — | — | **两态完全一致**（见下） |

**NCCL 为什么没变**：`NCCL_DEBUG=INFO` 抓取显示 **RCCL 在这台机上只用 `SHM/direct/direct`——56/56 条 channel 全是 SHM、0 条 P2P**，根本不产生 peer TLP；连 `NCCL_P2P_DISABLE=1` 都测不出差别。所以"关 ACS 让某套 NCCL 算法/协议吃到收益"**不成立**。

**对我们的工作负载**（27B W8A8 INT8 TP=2、`=1`、同会话 A/B）：ACS 关 decode **84.68** / prefill in3800 **8686.90** / conc8 **33.45**；ACS 开 **84.59 / 8684.11 / 33.23** ⇒ **全部 ≤0.7%，无可测差异** ⇒ **历史 TP≥2 基线不需要因为 ACS 重测**。

### 9.3 操作、持久化与回滚

```bash
sudo python3 /home/lyh/z100-vllm/tools/dcu/dcu_p2p_acs.py show    # 只读
sudo python3 /home/lyh/z100-vllm/tools/dcu/dcu_p2p_acs.py apply   # 快照 + 清零 12 个桥
sudo python3 /home/lyh/z100-vllm/tools/dcu/dcu_p2p_acs.py revert  # 从快照还原
docker exec z100-s9 bash -lc 'source /opt/dtk/env.sh; python /src/tools/dcu/p2p_verify.py'   # 带宽验证
```

脚本只改**桥**（class `0x0604xx`），**绝不碰 DCU 端点本身**；`apply --keep-sv` 可只清 `RR|CR|UF`、保留 `SrcValid`；快照在 `/var/lib/dcu-p2p-acs.snapshot`。

**开机自启（已落地）**：[`tools/dcu/dcu-p2p-acs.service`](../tools/dcu/dcu-p2p-acs.service) → `/etc/systemd/system/dcu-p2p-acs.service`，`enable` 且 `active (exited)`：

```bash
systemctl status dcu-p2p-acs   # oneshot：ExecStart=apply、ExecStop=revert
systemctl stop  dcu-p2p-acs    # 还原 ACS（= revert）
systemctl start dcu-p2p-acs    # 再次清零（幂等）
```

**踩坑与修复（回滚完整性）**：`apply` 原本每次都重写快照，而 systemd 每次开机都会调用它 —— 同一次开机里重复 `apply` 会把 `old_ctl` 记成 `0x0000`，**`revert` 就变成了空操作**（实测踩到）。已修：`apply` 默认**保留已存在的快照**（新增 `--force-snapshot` 才重写），并把快照恢复成原始的 12×`0x001d`。验证：再次 `apply` → `快照已存在,保留原值` + `已修改 0/12 个桥`，12 个桥仍为 `0x0000`。

**佐证**：`dmesg` 里有 `pcieport 0000:00:03.1: hydcu: change to gen4 is not supported because the link has a bridge or switch!` —— 这几张 DCU 本来就挂在 PCIe 交换机后面、根端口上不去 Gen4，所以 peer 流量**必须**走交换机直连；ACS 恰恰把它重定向回了根复合体（那侧只有 3.0 x8）。

- **重启即失效**（改的是活跃 PCIe 配置）→ **已用 systemd unit 处理**（见上）；
- **ACS 是 DMA 隔离特性**：单租户推理机无影响，**多租户/VFIO 场景不要关**；
- 脚本已入仓：[`tools/dcu/dcu_p2p_acs.py`](../tools/dcu/dcu_p2p_acs.py)、[`tools/dcu/p2p_verify.py`](../tools/dcu/p2p_verify.py)（后者修正了旧 `p2p.py`"只同步 cuda:0 导致 1324 GB/s 假带宽"的坑）。

### 9.4 对本项目的影响（S12 量化 + S13 复核定稿）

- **NCCL 两态完全一致**（RCCL 全走 SHM，0 条 P2P channel）⇒ 关 ACS **对 NCCL 通信零收益**；
- **27B TP=2 工作负载两态差异 ≤0.7%**（decode 84.68 vs 84.59）⇒ 现有 TP=2 配方不受影响，历史基线无需重测；
- **TP=4：两次更正后的正确结论（S14 实测）**。① 我们最初判"这台机跑不了 TP=4"是错的（用户在当前机器上用 W8A8 INT8 跑通过）；② 但"真因 = 影子容器 `z100-s9` 的 cgroup 32 GiB 上限、只要换成 `MemLimit=0` 就行"**同样错误，S14 已用内核原文推翻**：三次 OOM（S9b / S12 / S14）全部是 `oom-kill:constraint=CONSTRAINT_NONE` + **`global_oom`** + `anon-rss ≈6.9–7.25 GB` + **`file-rss:0kB`**，S14 用的容器 `z100-s14` 已确认 `HostConfig.Memory=0`、cgroup `memory.max=max`、`memory.current` 峰值 **29.76 GB < 32 GiB** ⇒ **没有任何一次是 cgroup 上限触发**。真因是 **graph 模式 TP=4 的宿主内存峰值 ≈34–35 GB ≥ 本机 32.0 GB RAM + 4.1 GB swap**（`file-rss:0` **不代表**权重在匿名内存——见下方 §3.7 实测更正）。
  生产容器能跑 TP=4 的关键是 [`serve_27b_tp4.sh`](/home/lyh/server/serve_27b_tp4.sh) 默认 **`ENFORCE_EAGER=1`**（eager 无图捕获 ⇒ 峰值更小），S13 的 8 次 vendor TP=4 运行也全是 eager ⇒ **"能跑 TP=4"对 eager 成立、对 graph 默认配置不成立（本机）**。
  **S14 定稿结果**：**eager TP=4 可行**（190 s 就绪、`oom_kill` 零增量、decode **126.83** ms/token，与生产基线 121.43 差 4.4%）；**graph TP=4 需挂 16 GiB swap** 才起得来（350 s 就绪，decode **49.66** ms/token）。**已排除的两个方向**：图池 1.67→0.22 GiB 仍 OOM（**不是图池**）、关掉 `=1` 改走 `TritonInt8ScaledMMLinearKernel` 同样 OOM（**不是 sdot4 权重重排**）⇒ 多出来的宿主内存在**编译/切分执行路径**。**内存构成实测（§3.7）**：权重在**显存**（`Model loading took 7.27 GiB`×4），每 worker `RssAnon` 4.3–4.5 GiB（eager TP=4）/ 6.6–6.9 GiB（graph TP=2）、`RssFile` 0.19/1.27 GiB，EngineCore 另约 1.67 GiB；OOM 时的 `file-rss:0` 只是**页缓存优先被回收**（**不是**权重在匿名内存——这点更正在前一版 §9.4/硬约束 26 里写过，已改）；swap 桥接的是**加载期 staging/冷页**（启动期换出 7.78 GB，引擎一关回落 0.53 GB），但**启动余量一度只剩 507 MB ⇒ 生产不宜照搬**（→ S14c）。
  **★★ 2×2 矩阵（`maxlen 262144` / `util 0.85` / `MBT 2048` / `VLLM_DCU_USE_INT8_SDOT4=1`，decode 中位 TPOT，ms/token）**：

  | decode 中位 TPOT | TP=2 | TP=4 | 结论 |
  |---|---|---|---|
  | **eager** | **113.51**（n=5） | **126.83**（n=5） | TP=4 **慢 1.12×** |
  | **graph** | **84.22**（n=5） | **49.66**（n=10） | TP=4 **快 1.70×** |

  ⇒ **"TP=4 更慢"是 eager↔graph 的口径错配，不是并行度的锅；后续任何"并行度值不值"的比较必须固定编译模式。** 对生产基线 121.43：**TP=4 graph = 0.409×（快 2.44×）**、TP=2 graph = 0.694×（快 1.44×）、TP=4 eager = 1.044×。ttft8k：TP=2 20838 → TP=4 12962 ms（1.61×）；conc8：33.34 → 35.55（eager）→ **41.13 tok/s（graph TP=4）**。**本机最优口径 = TP=4 + graph + 4 桶捕获 + 16 GiB swap = 49.66 ms/token**，但**保留意见**：它的图配置与 TP=2 默认配置不同、默认配置在本机起不来 ⇒ "TP=4 更强"里混入的图配置贡献无法在本机剥离。
  **新未决问题（→ S14b，已完成定性与纠正）**：同 TP 下 graph↔eager **实质改变 token 流**，但**方向与本文件上一版写反了**——**退化成英文重复循环的是 graph 的 P2 臂**（`'The'` 4.75 nat），**eager 侧才是连贯中文**（`'用户'`）；噪声底 0.1–0.6 nat ⇒ 分歧超噪声底、属系统性。**S14b 判定**（[`z100-14b-graph-vs-eager-result.md`](result/z100-14b-graph-vs-eager-result.md)）：**以 B（可解释数值）为主** —— 真值锚（未量化 4B TP=1 + CPU HF，首 token 4/4 一致）显示 **eager 差真值 0.0025 nat、graph 0.0023 nat、graph↔eager 仅 0.054 nat**（编译模式本身只值 0.05 nat），27B 的 1–4 nat 是**动态 per-token INT8 激活量化 × 64 层 GDN 递推**对这点差异的放大（离散取整 + 尾部），**不是 S7b/S7c 那类算子被编坏**（负向控制：packed decode 开/关在 eager 上逐位相同、**cudagraph 与数值无关**、无 NaN/Inf）。**但另有一个 A 形状缺陷**：**graph 运行间不可复现** —— P2 那次用逐字同配置/同 cache/同 AOT hash/冷首请求重跑得到 `'用户'`/gap 1.047 ⇒ P2 是**一次坏 draw**；唯一相关因子是"AOT 现场编译 vs 磁盘重载"（H-AOT 只部分成立：重载 3/4 prompt 逐位一致、第 4 个从第 59 位起差到 9.5 nat）→ **S14d**。
  ⇒ **质量/精度参照 = eager**（逐位可复现：1024 token、8/8 flip、同引擎重复 Δ=0.000）；**graph 的 token 流不得用于质量/精度评测**；**eager 可安全用于生产**（生产默认就是 eager）。**性能结论不受影响。**
  另外 **4B 的 TP 曲线（1.46× → 1.07×）不能外推到 27B**：27B decode 是**权重带宽主导**，TP=4 把权重切到 4 张卡后每卡流量减半，预期收益应显著大于 4B 那个 1.07×；
- **公平基线（忠实复现生产栈）**：在无内存上限容器里装回 vendor vLLM `0.18.1+das` + phaseA + `int8_sdot4` + `gdn_decode` + **生产容器里那份自建 triton `3.4.0+gitc817b9b6`**（镜像自带 triton 编不出 gfx906），日志与生产逐条对上（`TorchSdot4Int8ScaledMMLinearKernel`、KV 18.42 GiB、并发 4.51×）。decode TPOT：ACS 开 **120.64/128.56**（把权威 122.47 夹住）vs ACS 关 **130.50/132.28** ⇒ 当时记为"ACS 关反而让**生产栈**慢 ~5.5%"，**但那是提示性 n=2、态内抖动 6.6% —— S13 已用 n=4/态证伪该差异**：ACS 关 **121.43** [117.57,124.44] vs 开 **122.19** [118.99,124.64]，**区间完全重叠、均值差 0.6%** ⇒ **两态的生产栈公平基线视为无差异（≈121–122 ms/token TP=4）**；
- **同 fabric 真实倍率：我们 TP=2 比生产栈 TP=4 快 1.43×–1.56×**（跨口径对 122.47 是 1.446×，与 S11l 的 1.458× 吻合）；
- **唯一可能吃到 ACS 收益的机制：已证伪（S13）**。原假设是"去掉 `--disable-custom-all-reduce`，让 vLLM 自研 AR 走直接 peer 写（`copy_` 那条 4.1× 路径）"。实测：① **实际选中的 AR 后端是 NCCL**（两臂日志逐字 `Using ['PYNCCL'] all-reduce backends`）；② custom AR 在 gfx906 上被**双重结构性阻断** —— 平台门禁 [`rocm.py:1239-1241`](../vllm/platforms/rocm.py) 只认 gfx94/gfx95 ⇒ [`parallel.py:1104-1109`](../vllm/config/parallel.py) **强制** `disable_custom_all_reduce=True`（CLI 带不带这个 flag 是同一套配置，配置回显始终 True）；即使强闯，[`csrc/custom_all_reduce.cu:156-159`](../csrc/custom_all_reduce.cu) 走 `hipDeviceMallocUncached`，探针在该 flag 上返回 `hipErrorOutOfMemory`（同尺寸普通 `hipMalloc` 成功 ⇒ 非内存不足）；③ **A/B 收益 = 0**：不带 flag 84.12 / 8667.97(in=3854) / conc8 33.39 vs 带 flag 83.98 / 8650.34 / 33.51，**全部 ≤0.4%（噪声）**；④ 正确性过关且 flag **不进 AOT key**（同 vintage 双加载臂 **0/505 逐位一致**、Δlogprob 0.000000；跨 vintage 的 319–431/512 全是缓存 vintage 效应）。
  机理层面为什么不奇怪：`s13_pw_emul` 显示 10 KB 消息的 peer 写是**延迟受限**（59.8 µs ACS 关 vs 62.2 µs ACS 开，仅差 4%；RCCL 同尺寸 88–100 µs/call），**ACS 的 4.1× 要到 ≥1 MiB 才出现**（2 MiB 3.2×、128 MiB 4.07×）——而 TP=2 的 AR 消息只有 KB 量级 ⇒ 原方案里"5.7 ms/step、6–7%"的预期**高估了一个数量级**。**结论：生产配方不动（保留该 flag，既无害也无用），不为它改平台门禁/csrc。**
- **补齐项（S13）**：**8K prefill = 2.578 ms/prefill-token**（in=8154、Median TTFT 21019.66 ms；S11l 口径 2.571、生产 2.789 ⇒ 优于生产 **1.082×**）。

### 9.5 TP=4 graph 的生产化：内存关已关闭（S14c，2026-10-10）

**问题重述**：TP=4 + graph 是本机最快口径（**49.66 ms/token = 生产栈 121.43 的 2.44×**），但**默认图配置在本机起不来**（3 次宿主级 `global_oom`）。S14c 用 **17 个 arm** 把"多出来的几 GB 是谁、能不能拿掉"做成了定量结论：

**① 归因（推翻"启动尖峰"假设）**
- **峰值是"稳态装不下"，不是"启动尖峰装不下"**：`peak − steady` = **4 ~ 1469 MiB**（TP=4 graph / eager），C2/C3 甚至为 **0**；最危险的时刻是**就绪后的稳态/bench 期间**（`min MemAvailable` 往往落在 P5_steady）⇒ **削启动尖峰的手段（限捕获桶、清编译缓存、AOT 重载）注定无效**，与实测一致。
- **TP 对齐后的净缺口 ≈4 GiB**（每 worker graph 比 eager 多 **≈1.4 GiB ×4**，被 EngineCore 反向的 ≈1.4 GiB 抵消）——**S14 首版估的 8.8 GB 是"TP 与模式双重变化"的高估**。
- **默认 51+2 桶配置不可用于生产**：ready **1645 s**、余量仅 **66 MB**，只换 4.3% decode。

**② 两个真正的杠杆（其余全部否证）**
| 杠杆 | 实测效果 |
|---|---|
| **AOT 磁盘重载**（先预热填缓存再以重载方式服务） | **现场编译在 4 GiB swap 下必 OOM；重载后同一 swap 就能活**。时间上：`init engine` **183 → 87 s**、ready **351 → 230 s（−34%）**；**内存上 +1.8%（无益）** ⇒ 它是"能不能活"和"启动快"的关键，不是省内存的手段 |
| **`vm.swappiness` 默认 60 → 200** | 同样 10 GiB swap 下余量 **0.29 → 2.71 GiB**，**decode 零代价**（49.83 ms）。机制：把加载期 staging/冷页主动换出，给匿名热页腾 RAM |
| ~~`MALLOC_ARENA_MAX=1`~~ / ~~`MBT=512`~~ / ~~`max-num-seqs 64`~~ / ~~`combo_kernels` 关~~ / ~~`expandable_segments=False`~~ / ~~图池压缩 1.67→0.22 GiB~~ / ~~`VLLM_DCU_USE_INT8_SDOT4=0`~~ | **全部无收益**（MBT=512 反而最紧：minAvail 0.33 GiB） |

**③ 达标配方（G3 口径② 达成）**
> 生产 flags（maxlen 262144 / util 0.85 / MBT 2048 / prefix cache / mamba align / `--disable-custom-all-reduce`，**无** `--enforce-eager`）
> \+ `-cc.cudagraph_mode=PIECEWISE --cudagraph-capture-sizes "1 2 4 8"`
> \+ **AOT 重载预热**
> \+ **新增 4 GiB swapfile**（与宿主原有 4 GiB `/swap.img` 合计 8 GiB）
> \+ **`vm.swappiness=200`**
> ⇒ **ready 230 s、decode 中位 49.65 ms、`min MemAvailable` 2.35 GiB**（arm C14）

- **口径说明**：验收线里的"≤4 GiB swap"指**新增的** swapfile；若按"总 swap ≤4 GiB"（仅 `/swap.img`，arm C15）则余量只有 **1.78 GiB** 且 swap 打满。
- **保留意见（S14c §5，最大一条）**：推荐配方**只测了 decode（n=5）**，**conc8 / 8K prefill / 长稳态未复测** —— 并发会改变页缓存压力与换出行为，**上生产前必须补测**。
- **未钉死的一条**：每 worker ~265 个 ~4 MiB 匿名 mmap 的**子系统归属**（只到 VMA 形状级）；S14c 建议用 `malloc_info()`/`gdb -p` 只读取证，**不要**用 `LD_PRELOAD`（与硬约束 2 的精神相悖）。
- **单 writer 归档**：[`z100-14c-tp4-graph-memory-result.md`](result/z100-14c-tp4-graph-memory-result.md)（61 KB）+ Lead 独立复核 [`z100-14c-lead-crosscheck.md`](result/z100-14c-lead-crosscheck.md)；**生产配方已写成 [`海光Z100-生产部署配方.md`](海光Z100-生产部署配方.md)**。
- **第二关（可复现性）：❌ 硬失败（S14d，2026-10-10）—— graph 不能用于任何依赖 token 流的用途**。
  * **E4**：生产候选配方下，**预热那次（现场编译，ready 352 s、`aot_load=0`）与服务那次（同 cache 重载，ready 230 s、`aot_load=4`）对 12 prompt 的 token 流 12/12 不逐位一致**；同上下文（字节相同 context）`prompt_logprobs` **mean|Δ| 1.04–1.76 nat、最坏 12.67 nat**、argmax flips 9–28、差异覆盖 **91.6%** 位置、首个分歧中位 ≈9 token。
  * **根因**：同一 cache 的第 2/3 次重载与第 1 次**同样 12/12 不一致** ⇒ **H-AOT（现场编译 ≠ 磁盘重载）被否证**，真因子是 **graph 路径运行间不可复现**。**7 臂 / 21 对 / 252 个 prompt-pair 仅 6 个（2.4%）逐位一致**，21 对的 `mean|Δ|` 全落在 **1.24–1.45 nat** 窄带且与配置正交；关掉 `combo_kernels`/`benchmark_combo_kernel` **无收益（1/36）**。
  * **与 TP 正交** ⇒ **TP=2 graph 在 token 流上同样危险**（S14b 的 P2 `'The'` vs A1r `'用户'` 就是 TP=2 上的答案级翻转）。
  * **S14e G1（04:37Z）："用默认图配置换回可复现"这条路也被排除**。在 S11m 口径（TP=2 / maxlen 4096 / **默认 `FULL_AND_PIECEWISE`，51 capture sizes**）下，两次**真 AOT 重载**（同一 artifact，md5 `91176924…`；`aot_load=2`、`ind_hit=1`）仍 **403/512 位置不同**（同上下文 `mean\|Δ\|` 1.03–1.15、`max` 7.92 nat、9/12 prompt 流不同），而**启动路径完全同构**（ready **1542 / 1542 s**、两轮 capture 684+696 / 683+695 s）⇒ 不是"某跑多编了一次"。
  * **精确化（S11m 冲突的解释）**：仪器 A 上 5 次独立重载（S2/S3/S4/G1a/G1b）**落在 2 个变体上**，10 对里 **6 对一致（60%）**；`S2sw1hit vs G1b = 0/512` ⇒ G1b 就是 S11m 那个变体。**S11m 的"加载 AOT 产物是确定的"只在"同一容器 + 同一时间窗"内成立**，跨时间窗会出现**离散变体**。
  * **S14e 收口（05:31Z，三次重载）**：**G1b ≡ S11m 的 S2/S3/S4（0/512）**，**G1a/G1c 与它们差 403/512**；6 跑逐位一致率 **7/15 ≈ 47%**，**同变体内仍有 1e-2~6e-2 nat 抖动**（足以在 128 步翻 1 token）⇒ **S11m 连抽 3 次同变体（≈12.5%）是运气**。**G2（只换图配置）**：3 跑 **3 个变体**、一致率 **8/36（22%）** ⇒ **省内存配置不是"会不会分叉"的原因，只是抬高频率**（默认 50% → 省内存 22% → S14d 2.4%）。
  * **生产结论**：**不能选默认图配置**（慢 **9×**：1542 s vs 170–180 s；S14c C7 在 TP=4/262144 是 1645 s vs 351 s、余量仅 66 MiB、decode 只快 4.3%），**却照样不可复现**；要 token 流可复现**只能 eager**，否则用省内存图配置并接受"每次重启换输出"（mean≈1.1 / max≈8.9 nat）。**⚠️ S11m 的"冷启动预热 → 重启即可复现"部署配方据此被否证**（预热只省时间）。
  * **S14g 收口（05:xx）：变体钉不住 ⇒ 不存在"可复现的 graph 配方"**。① **ASLR 否证**：容器默认 seccomp 拦 `personality(2)`（`setarch -R` EPERM，**不是没装 setarch**），另建 `z100-s14g`（`seccomp=unconfined`）后跑通——**PIE 基址被钉死**（`555555554000`，两个 worker 继承）但 **heap/mmap 每跑仍不同**，两跑 **0/8** 逐位一致（`max\|Δlp\|` 0.401 nat），且落到 14 跑未见过的**新变体**；ASLR-on 对照落回最常见吸引子（与 S11m S2/S3/S4、S14e G1b/G2c 同签名）⇒ **布局参与决定但不是可钉的开关**。② **autotune 排除**：7 个 `@triton.autotune` kernel 两次 run **config 逐字相同、全缓存命中**、8 次健康 run 未写新 autotune JSON。③ **工程发现**：`PYTHONPATH`/sitecustomize 注入**到不了 `VLLM::Worker_TP*`**，唯一有效通道是 site-packages `.pth`；worker `/proc/*/environ` **不可信**。④ **残余**：allocator 残余地址布局 → Triton specialization/cache key 变化（需编译期取证）。
  * **生产定论（无第三条路）**：**要 token 流可复现 ⇒ eager**（131.78–133.83 ms，且要固定批组合）；**要 2.44× ⇒ 省内存 graph 配方 + SLA 写明"重启后输出可变"**（12 题中 8 题 token 流变化、同上下文 `max\|Δ\| ≈0.3–0.4 nat`）。
  * **两处口径纠正**：① **"eager 需要内存余量"不成立**（挂 16 GiB swapfile 133.83 vs 不挂 131.78 ms = 1.6%，且那 16 GiB **一页未用**）⇒ 之前怀疑的"C2 142.59 vs P4e 126.83 是没挂 swap"被否证，是**抖动/vintage**；② **新洞：eager 的"批组合效应"**（conc=2 与串行在第 6 token 分叉、mean 1.82/max 15.02 nat，而同一批组合跨引擎**逐位一致**）⇒ **本项目此前所有 conc=1 与 conc=8 混用的数值对比都可能混入了批组合**（→ 立 S14f，评测必须固定并记录 `--max-num-seqs`/批组合）。
  * **可执行口径（S14d §4.3）**：**质量/精度/回归/审计一律用 eager 且固定批组合；graph 只作延迟/吞吐口径，绝不进入任何 token 流相关的门禁/SLA**。若要用 graph 服务，必须在 SLA 写明"同一请求跨重启/跨批组合输出不保证一致"。
  * 归档 [`z100-14d-determinism-result.md`](result/z100-14d-determinism-result.md)（705 行、md5 `2a9cabca…`）。
