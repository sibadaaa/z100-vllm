# vLLM 切换新栈（torch 2.16.dev / ROCm 10.1 / Triton 3.6 Z100）——适配实施计划

> 日期：2026-10-18 · 状态：**计划（未执行）** · 适用对象：执行本计划的 agent / 工程师
> 前置工件（必须一起读）：
> - PyTorch 构建与验收：`/mnt/game/torch-rocm10/README.md`（尤其 §0/§4/§6.4）
> - Triton 构建与门禁：`/home/lyh/triton-z100/docs/rocm101-README.md`
> - 上一轮 ROCm 10.1 编译尝试：`docs/z100-rocm10-build.md`
> - 当前树与 DTK 基线：`docs/z100-sync-20261016-redo.md`
> - 性能基线配方：`docs/perf/PERF1-recipe.md` / `PERF1-results.md`
> - 三份调研证据：`/tmp/z100-adapt-plan/{stack-auditor,vllm-compat,verify-plan}.md`

---

## 0. 目标与非目标

### 0.1 目标

在 **Z100（4× gfx906，`sramecc+:xnack-`）** 上，把 `/home/lyh/z100-vllm`（分支
`sync/upstream-20261016-v2`，上游 main `64175d38`）从 **DTK 26.04 / torch 2.11+rocm7.2 /
triton 3.6.0(das)** 切换到 **本地自编 ROCm 10.1 栈**：

| 组件 | 目标版本/路径 |
|---|---|
| PyTorch | `2.16.0.dev20261016+rocm10.1`（自编 gfx906 + SFU 补丁版，`/mnt/game/torch-rocm10/wheel-sfu/`，sha256 `46a5f447…4e7e49`） |
| HIP / ROCm | HIP `7.16.26385` / ROCm `10.1.0`（镜像 `z100-rocm10.1:24.04-r642`，`ROCM_PATH=/opt/rocm/core-10.1`） |
| 编译器 | 自建 clang 24 + SFU 勘误补丁：`/mnt/game/torch-rocm10/llvm-build/bin` |
| Triton | `3.6.0`（gfx906/ROCm10.1 构建，`/mnt/game/triton-z100-rocm101/install`；wheel sha256 `de3e4c75…504876`） |
| vLLM | 当前树重新编译的 4 个扩展（`_C / _C_stable_libtorch / _moe_C_stable_libtorch / _rocm_C`） |

**完成定义（DoD）**：新栈上通过 §7 的 **L0–L5 全部门禁**，且：
1. `Qwen3-0.6B`（dense）、`Qwen3.5-4B`（GDN 混合）离线推理 token 与 DTK 冻结基线逐 token 一致；
2. G8（FLA/GDN 对拍）`max|diff| ≤ 1e-4` 且 NaN=0；
3. `Qwen3.8-27B-FP8` TP=4 生产配方可启动并完成性能 anchor：decode ≥0.95×（同树基线）；
4. 交付物齐备：可一键启动的新栈运行脚本、环境契约、验收报告、**DTK 回滚通道演练通过**。

### 0.2 非目标

- **不追求** torch.compile/inductor 与 triton 3.9 的版本对齐（生产/质量口径一律 `--enforce-eager`；
  torch.compile 仅作可选实验臂，见 D4）。
- **不升级** vLLM 上游、不做新性能优化（本计划只设"不退化"门槛）。
- **不动** DTK 生产容器/镜像/已部署服务；两栈并行，DTK 始终可回滚。
- 多模态 / embedding / GGUF 模型 / GSM8K 评测不纳入首轮门禁（本机无权重/数据集，见 §9-Q6）。

---

## 1. 结论摘要（先看这个）

1. **"能编译"已经不是问题，剩下的是"数值正确 + 能跑起来 + 多卡通"**：
   - 新 torch wheel 已通过 19 项数值门禁（softmax 8.4e-9 / tanh 7.2e-8 / erf 7.3e-8 / layer_norm 4.0e-7…），
     gfx906 code object 真实存在（220 个 offload bundle，全部 gfx906）。
   - 新 Triton 已通过 G0–G7（8/8）+ SFU 正/负向变异门禁；gfx906 路由与 `s_nop` 规避在 ROCm 10.1 上有效。
   - vLLM 4 个扩展**上一轮已在 ROCm 10.1 下编译成功**（`docs/z100-rocm10-build.md`），但用的是
     官方 torch wheel + stock clang：ABI 可参考、**产物不可复用**（数值不可信）。
2. **两个会影响基线的发现（调研期实测）**：
   - **F1**：redo merge 把 S7c 修过的 `csrc/libtorch_stable/activation_kernels.cu`（DTK+gfx906 fp16 标量守卫）
     换成了上游向量化版本 ⇒ **DTK 栈 packed-SiLU fp16 回归（3/3 fail，输出 inf）**。对新栈不阻塞
     （测试在新栈 skip），但**DTK 作为回滚目标时模型级正确性存疑** → 建议切换前修（§9-Q8）；
   - **F2**：`docs/z100-sync-20261016-redo.md` §6 的"已知失败"已过时（当前树 DTK 上 attention 39/39、
     attn_v1 18/18、recurrent 5/5 全绿）⇒ **L-1 必须以实测冻结 expected-fail 集，不要照抄文档**。
3. **真正的新工作只有 5 件**：
   a. 用 **SFU 补丁版 clang 24** 重编 vLLM 4 个 HIP 扩展（vLLM 自有 kernel 里 expf/rsqrtf 密集，
      实测同一 kernel：stock clang `128/512` lane 错 → 补丁 clang `0/512`）；
   b. 搭一个干净的 ROCm 10.1 运行时容器 + 依赖清单（Python 3.12、无 pip/pydev/libopenblas 的镜像坑）；
   c. 少量代码适配（平台探测/bf16 策略/requirements，源码级"必须改"仅 3 处 gate + 1 个 merge 残留 bug）；
   d. **RCCL 重编——当前最大阻塞**：ROCm 10.1 自带 RCCL 2.30.7 **没有 gfx906 device code**，
      4 卡 all_reduce 报 `invalid device function`；必须从源码按 `-DAMDGPU_TARGETS=gfx906` 重编
      （upstream RCCL 默认 arch 列表含 gfx906）。**RCCL 修好前 TP>1 不可行**；
   e. 冻结同树 DTK 基线后再做 L2–L5 验证（跨树 PERF1 数字只能当量级参考）。
4. **工期估计**：P0 环境+编译 ~1 天；P1 重编+加载门 ~0.5 天；P2 代码适配 ~1 天；P3 正确性 ~1–2 天；
   P4 精度/性能/多卡 ~1–2 天；P5 固化 ~0.5 天。**RCCL 与 P0/P1 并行**，总关键路径 ≈ 5–6 个工作日。
5. **回滚成本极低**：只需把 `vllm/*.so` relink 回 DTK 构建的 4 个 .so 并重启服务（不改 git、不改镜像）。

---

## 2. 现状盘点（自包含事实基座）

> 本节事实均已实测/有日志，执行时**不要重复调研**；引用位置见括号。

### 2.1 硬件 / 镜像 / 容器

| 项 | 值 |
|---|---|
| GPU | 4× 海光 DCU：DCU0/1 = Z100L @1.60 GHz，DCU2/3 = Z100 @1.32 GHz，各 60 CU，PCIe 互联；`gfx906:sramecc+:xnack-`，32 GiB/卡 |
| 架构坑 | amdsmi 报 `gfx916`；HIP/torch 报 `gfx906`（fork 的 `normalize_gcn_arch` 会做别名映射） |
| ROCm 10.1 镜像 | `z100-rocm10.1:24.04-r642`（TheRock 布局），`ROCM_PATH=/opt/rocm/core-10.1`；镜像**无 pip / python3-dev / zlib1g-dev / libopenblas.so.0** |
| ROCm SDK wheels | `/mnt/game/rocm10-wheels/rocm_sdk_{core,libraries,devel}-10.1.0-*.whl`（core+libraries 运行必需；devel 编译/amdsmi/librccl 路径需要） |
| DTK 基线容器 | `z100-regress` / `z100-build-211` / `z100-triton`（DTK 26.04；**保持不动**，作为基线与回滚） |
| ROCm10 已有容器 | `z100-torch-build`（torch 构建，已装新 torch 2.16 + rocm-sdk-devel，可用作只读参考）、`z100-triton-rocm101`（triton 构建）、`z100-rocm10-build`（上一轮 vLLM 编译，现装 torch 2.9.1 旧试验）、`z100-vllm-rt-probe`（本轮调研建的干净运行验证容器） |
| 磁盘 | `/mnt/game` ~368 GB 空闲；新栈 runtime ≈4 GB + build ≈5 GB，够用 |

### 2.2 新栈三件套验收状态（已有证据）

| 资产 | 状态 | 证据 |
|---|---|---|
| torch wheel-sfu | ✅ `cp312-cp312-linux_x86_64`，167,990,747 B，sha256 `46a5f447…4e7e49`；`get_arch_list()==['gfx906']`；code object V6；`is_nccl_available=True` | `/mnt/game/torch-rocm10/logs/final-verification.log`（`ALL_PASS` + `NUMERIC_GATE_PASS`） |
| SFU 补丁 clang 24 | ✅ `/mnt/game/torch-rocm10/llvm-build/bin`（`clang++ --version` = 24.0.0git；补丁在 `SIPreEmitPeephole`，gfx906 的 SFU 指令前插 `S_NOP 2`）；`compiler-rt` 已补齐 | torch README §6.4；`patches/0002-llvm24-amdgpu-sfu-workaround.patch` |
| Triton 3.6.0 (rocm101) | ✅ G0–G7 **8/8 PASS** + SFU 变异门禁 PASS（负向 262048/1048576 复现，100% 落 lane 8–11）；G8/G9 未做（就是本计划 P3） | `/mnt/game/triton-z100-rocm101/gates/*.log` |
| torch 2.16 ↔ triton 3.6 互操作 | ✅ 源码硬性最低要求 = triton 3.6（`torch/_native/triton_utils.py`）；`torch.compile`(inductor) 在 gfx906 上实测可编译运行（pointwise / fp16 GEMM） | 调研附录 `/tmp/z100-adapt-plan/stack-auditor.md` §3 |
| vLLM 4 扩展（ROCm10.1, stock clang, 官方 wheel） | ⚠️ 编译成功、可加载注册；与本轮自编 wheel 同 torch 版本（ABI 大体一致），但**数值不可信（SFU）**，本计划要求重编、不复用产物 | `docs/z100-rocm10-build.md`、`/mnt/game/port/build-rocm10/temp/*.so` |
| **RCCL** | ❌ **rockm 10.1 的 RCCL 2.30.7 无 gfx906 device code**：4 卡/1 卡 all_reduce 均 `[FATAL ERROR]: HIP failure: 'invalid device function'`；7 组 env 矩阵（P2P/PROTO/ALGO/JIT/CUMEM/SDMA）无效 | `/tmp/z100-adapt-plan/stack-auditor.md` §4.4 |

### 2.3 vLLM 树现状与已完成的 ROCm10.1 编译修复

- 树：`/home/lyh/z100-vllm`，分支 `sync/upstream-20261016-v2`，HEAD `6067e19a6`。
- 当前 pin：`pyproject.toml` build-system `torch == 2.11.0+rocm7.2`；`requirements/rocm.txt`
  `torch/torchvision/torchaudio==…+rocm7.2`；`requirements/rocm-build.txt` `triton-rocm==3.6.0`。
  上游 `requirements/cuda.txt` 实际已指向 `torch==2.14.0`（fork 的 rocm pin 是本地加的）。
- 已提交的 ROCm 10.1 编译适配（`5700a50c7`，3 处，均已有）：
  `cmake/hipify.py`（torch 2.16 删除 `CUDA_MACROS_MAP`，加 getattr 守卫）、
  `csrc/quantization/gptq/compat.cuh`（HIP≥7 原生 `atomicAdd(half*)`，改按 `HIP_VERSION` 关模拟）、
  `CMakeLists.txt`（删除重复 `fs_io_C` target）。
- 上一轮 vLLM 构建脚本（可复用，需改 4 处）：`/mnt/game/port/{setenv,configure,build}-rocm10.sh`
  + 构建目录 `/mnt/game/port/build-rocm10/`。
- 生产/质量口径关键事实：**vLLM 一律 `--enforce-eager`**（graph 路径 token 不可复现，S14d），
  而 `--enforce-eager` 在本树会同时关闭 torch.compile 与 CUDAGraphs（`vllm/config/vllm.py:1861`），
  因此"torch 2.16 期望 triton 3.9、我们只有 3.6"**不阻塞主路径**（仅影响可选 compile 实验臂）。
- 生产模型：`Qwen3.8-27B-FP8`（qwen3_5，64 层 = 48 GDN + 16 full attn），TP=4，MTP k=2，
  max-model-len 262144，MBT 2048，util 0.85，`VLLM_DCU_USE_FP8_EMU=1`、`VLLM_DCU_USE_BF16_GEMV=1`、
  `VLLM_DCU_FP8_GEMV_MAX_M=4`。

### 2.4 现网 DTK 基线（新栈要对齐的参照；详见 `verify-plan.md` §1/§2.3）

| 项 | DTK 26.04 基线 |
|---|---|
| 当前树 4 个 .so md5 | `_C d44117ae…` / `_C_stable 565f8c26…` / `_moe_C_stable 50af36b9…` / `_rocm_C f205be8f…`（`/mnt/game/port/build-sync/temp/`） |
| 性能基线（PERF1，**另一棵树** `gfx906-vllm-211`） | 新配方：step p50 249→165 ms、decode 11.04→18.19 t/s、prefill 449→540 t/s、TTFT 3755→3114 ms（均值）；生产 .so md5 `8e98bd60…` |
| 正确性基线（**2026-10-18 实测，文档已过时**） | `test_qwen3.py` 84/84、`test_hygon_dtk.py` 52/52、`test_gdn_metadata_builder.py` 62/62（必须单文件跑）、attention `39/39` + v1 `18/18`、recurrent `5/5`、`sigmoid_gating 5/5`、`gguf 13/13`、`w4a16 3/3`；**旧的 "attention PassManager failed" 已消失**（`docs/z100-sync-20261016-redo.md` §6 不再作准） |
| 已知失败（expected-fail，L-1 真题为准） | `test_qwen3_5_gdn.py` 44/3 failed（fork 测试桩过时，与栈无关）；`test_bf16_gemv.py` 15/1、`test_gfx906_packed_silu_fp16.py` 0/3 —— **后者是 merge 回归**（见 F1）；`test_causal_conv1d.py` 161/66（既有，须在 L-1 重测冻结） |
| **F1（merge 回归，影响 DTK 回滚基线）** | redo merge 把 `csrc/libtorch_stable/activation_kernels.cu` 的 S7c DTK+gfx906 fp16 标量守卫换成了上游向量化版本 ⇒ **DTK 栈 packed-SiLU fp16 误编译（输出 inf）**，`test_gfx906_packed_silu_fp16.py` 3/3 fail。对**新栈不阻塞**（该测试在新栈会 skip；ROCm10.1 不命中原误编译），但若 DTK 作为回滚目标，**其模型级正确性存疑**（S7b NaN 根因），L-1 跑 G9 时必须查 NaN |
| **F2（文档过时）** | `docs/z100-sync-20261016-redo.md` §6 的已知失败清单在当前树+当前 DTK 构建上大半已消失 ⇒ **L-1 必须以实测冻结 expected 集合，不要照抄文档** |
| G8/G9 历史 | DTK 修复前 GDN 内核 `max|diff|≈1e26`、8192 NaN；修复后 DTK 上应 MATCH/0 NaN（G8 正式判据 `max|diff|≤1e-4` 且 NaN=0）；G9 `The capital of France is` → ` Paris.` |
| PERF1 严格数字（跨树参考） | anchor p512/g64：step **135.0 ms**、decode **21.30 t/s**、TTFT **1658 ms**、prefill **272.2 t/s**、tok/step **3.048**；全长度 decode 20.39 / 22.25 / 18.34 / 11.79 t/s（真实 119/451/1806/7443 tok）。**同树 L-1 重跑前只作量级参考** |

### 2.5 缺口清单（按阻塞级别）

| 级别 | 缺口 | 影响 |
|---|---|---|
| **阻塞（多卡）** | RCCL 无 gfx906 device code | TP>1/all_reduce 不可用 → 27B TP=4 不可跑 |
| **阻塞（数值）** | vLLM 4 扩展未用 SFU 补丁 clang 重编 | attention/activation/layernorm/GDN C++ kernel 数值不可信 |
| 阻塞（运行） | 专用运行时容器 + 依赖未固化；`USE_LIBUV=0`、`RCCL_USE_AMD_SMI_LIB=1` 为必带开关 | `import torch`/`init_process_group` 失败 |
| 高 | 平台探测/amdsmi/rocminfo 在 TheRock 布局下的回落（`/opt/rocm/bin` 不存在） | 无可见性 env 时 "No platform detected" |
| 高 | bf16 策略：`is_hygon_dtk()=False` → gfx906 bf16 被禁用（Qwen3.5-4B bf16 基线不一致） | L3 G9 token parity 无法对齐 |
| 中 | requirements/pyproject pin 仍指 DTK 2.11/rocm7.2；`setup.py:1268-1269` merge 残留 bug | pip/uv 路径不可用；仓库卫生 |
| 中 | AOT compile 默认开（torch≥2.10 `VLLM_USE_AOT_COMPILE=1`） | 非 eager 实验臂可能踩未验证路径 |
| 低 | `_opaque_base` deprecated、`TORCH_SUPPORTED_VERSION_ROCM` 仍是 2.14 | warning/日志噪声 |
| 待确认 | `HSA_FORCE_FINE_GRAIN_PCIE=1` 在 ROCm 10.1 是否仍需要 | RCCL P2P 性能/正确性 |

---

## 3. 关键决策（D1–D7）

| # | 决策 | 理由 / 依据 |
|---|---|---|
| **D1** | 新栈走**独立容器 + 独立可写工作副本**，不覆盖 DTK；DTK 栈与 `z100-regress` 保持可用 | 生产可回滚；沿用 S4b 纪律 |
| **D2** | **必须**用 SFU 补丁 clang 重编 vLLM 4 个 HIP 扩展；code object 版本**先与 torch 对齐（不传 `-mcode-object-version`，即 clang24 默认 V6）**，若加载失败回 V4 | torch 自身 V6 已验证可加载；V4 是上一轮的口径，两者 driver 都吃；实测 stock clang 数值错误（128/512） |
| **D3** | Triton **固定用 3.6.0(rocm101) 本地构建**（PYTHONPATH 前置），不装任何 `triton/triton-rocm/pytorch-triton-rocm` 顶层包 | 只有这份 triton 带 gfx906 路由 + SFU overlay；torch 2.16 硬性最低要求 = 3.6 |
| **D4** | 质量/验收口径一律 `--enforce-eager`；`VLLM_USE_AOT_COMPILE=0` 作为 bring-up 默认；torch.compile 只作 P4 的可选实验臂 | 生产就是这样跑的（S14d）；eager 会同时关 compile，绕开 3.6 vs 3.9 的未验证路径 |
| **D5** | RCCL 作为 **P0 并行关键路径**：从源码按 `-DAMDGPU_TARGETS=gfx906` 重编 `librccl.so.1`，装在 `/mnt/game/rccl-gfx906/lib`，LD_LIBRARY_PATH 前置（**不改** rocm-sdk 安装） | upstream RCCL `CMakeLists.txt` 默认 arch 含 gfx906，TheRock 构建覆盖掉了；DTK 的 RCCL 依赖 `libgalaxyhip` 不能混用 |
| **D6** | bf16 策略：把 `_rocm_supported_dtypes()` 的 gfx906 分支从 `is_hygon_dtk() and probe` **泛化为 gfx906 探针**（`gfx906_bf16_enabled()`），保持 `VLLM_DCU_BF16` 三态（auto/on/off） | torch 2.16 报 `is_bf16_supported=True`、bf16 matmul 实测 PASS；DTK 基线的 Qwen3.5-4B 是 bf16，不放开无法做 token parity。**这是二选一的语义决策，见 §9-Q1** |
| **D7** | 新栈 pin 放**新文件** `requirements/rocm101.txt` / `requirements/rocm101-build.txt` + 新环境脚本，DTK 的 `requirements/rocm.txt` 等保持不动（仅在 `setup.py` 加分支） | 两栈并行期避免互相污染；DTK 回滚不需要 git revert |

---

## 4. 目标架构

### 4.1 容器与目录

```
┌─ 宿主机 /mnt/game（368G 空闲） ──────────────────────────────────────────────┐
│ torch-rocm10/wheel-sfu/*.whl         ← 新 torch（只读，勿覆盖）               │
│ torch-rocm10/llvm-build/bin          ← SFU 补丁 clang（只读，勿覆盖）        │
│ triton-z100-rocm101/install          ← 新 triton（只读）                     │
│ rocm10-wheels/rocm_sdk_*             ← ROCm 10.1 SDK wheels                  │
│ rccl-gfx906/lib                      ← 【P0 新建】gfx906 版 librccl.so.1     │
│ port/gfx906-vllm-rocm101/            ← 【新建】vLLM 可写工作副本（构建产物落这里）│
│ port/build-rocm101/                  ← 【新建】CMake 构建目录                 │
│ z100-baseline/                       ← 【新建】L-1 冻结基线（pytest xml/G9 json/perf）│
│ models/                              ← Qwen3-0.6B / Qwen3.5-4B / Qwen3.8-27B-FP8 … │
└──────────────────────────────────────────────────────────────────────────────┘
        ▲ /mnt/game rw                ▲ /src:ro (/home/lyh/z100-vllm)
        │                             │
┌───────┴─────────────────────────────┴──────────────┐
│ 容器 A：z100-vllm-rocm101-rt（新栈运行容器，新建） │  ← 验收/服务都在这里跑
│ 镜像 z100-rocm10.1:24.04-r642；--device kfd/mkfd/dri；shm 16g │
│ 已装：rocm-sdk core+libraries(+devel) / torch wheel-sfu / triton(PYTHONPATH) │
│         vLLM 运行依赖 + 测试依赖                                            │
├────────────────────────────────────────────────────┤
│ 容器 B：z100-rocm10-build（改造为编译容器）        │
│ 直接复用（已有 rocm-sdk-devel + python3-dev）；      │
│ `pip --no-deps --force-reinstall` 新 torch wheel     │
│ 构建时：HIP_CLANG_PATH/CMAKE_HIP_COMPILER=补丁 clang │
├────────────────────────────────────────────────────┤
│ 容器 C/D/E：z100-regress / z100-build-211 / z100-triton（DTK，**不动**）│
└────────────────────────────────────────────────────┘
```

### 4.2 源码与产物策略

- **唯一代码真相**：`/home/lyh/z100-vllm`（新分支 `rocm101-torch216`，从当前 HEAD 拉出）。
- 编译跑在**可写副本** `/mnt/game/port/gfx906-vllm-rocm101`（rsync 同步，排除 `.git` 大对象/缓存），
  产物用 `cmake --install ... --prefix <副本>` 落到 `<副本>/vllm/*.so`（沿用 `install-211.sh` 方式）。
- 运行容器 `PYTHONPATH=/mnt/game/port/gfx906-vllm-rocm101:/mnt/game/triton-z100-rocm101/install`，
  `/src:ro` 只用于对照/测试代码。
- 每阶段结束把副本里的源码差异（`.py/.cu/.h/.txt`）回写 repo 并提交；`.so` 为 gitignored 产物。
- 记录纪律：每次 GPU 作业前记录 **git HEAD + 4 个 .so md5 + triton `__file__`**，日志落 `/mnt/game`。

---

## 5. 任务板与关键路径

| 阶段 | 任务 | 依赖 | 预计 | 验收门 | 主要产出 |
|---|---|---|---|---|---|
| **P0** | 环境/栈冻结 + 运行时容器 | 无 | 0.5–1 d | L0 全绿 | `z100-vllm-rocm101-rt`、`tools/dcu/env-rocm101.sh`、依赖已装 |
| **P0-R** | **RCCL gfx906 重编**（并行） | 无 | 0.5 d | `llvm-objdump` 有 gfx906 + 4 卡 all_reduce PASS | `/mnt/game/rccl-gfx906/lib`、rccl 构建脚本 |
| **P1** | 用新 torch+补丁 clang 重编 4 扩展 | P0 | 0.5 d | L1 全绿 | 4 个 .so（新 md5 登记） |
| **P2** | 运行期/代码适配（requirements/平台/bf16/env/setup.py） | P0,P1 | 1 d | `import vllm` + `vllm serve` 0.6B 启动 | 分支提交、`requirements/rocm101*.txt`、env 脚本 |
| **P3** | 正确性：L2 kernel 数值 + L3 G8/G9 | P1,P2 | 1–2 d | L2/L3 门禁 | G8/G9 报告、pytest xml |
| **P4** | 精度/perf/多卡：L4 + L5 | P3,P0-R | 1–2 d | L4/L5 门禁 | perf 对比报告、TP=4 服务 |
| **P5** | 固化交付 + 回滚演练 + 文档 | 全部 | 0.5 d | 一键启动/回滚脚本 + 验收报告 | 交付文档、分支合并/标签 |

**关键路径**：P0 → P1 → P2 → P3 → L4 → L5（RCCL 若晚于 L4，L4 可先用 TP=1 完成）。
**并行机会**：P0-R（RCCL，独立容器/独立目录）；L-1 基线冻结可在 P0 期间由另一人同时做（DTK 容器）。

---

## 6. 逐阶段 Runbook

> 通用约定：
> - 容器内命令均以 `bash -lc` 执行；**先 `import torch` 再 `import vllm`**（DTK 时代硬规则；新栈 L0.7 复核）。
> - 所有 GPU 作业前 `pgrep -af vllm` + `hy-smi`；作业期间持 `/tmp/perf1_build.lock`；显式 `HIP_VISIBLE_DEVICES`。
> - 质量口径 `--enforce-eager`；不并行跑多个 GPU 作业；每步保存原始日志路径。
> - `RT` = 新栈运行容器 `z100-vllm-rocm101-rt`；`BUILD` = `z100-rocm10-build`；`REG` = `z100-regress`。

### 6.1 P0 —— 环境与栈冻结

#### P0.1 新建运行容器（约 10 min）

```bash
docker run -d --name z100-vllm-rocm101-rt \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --shm-size=16g \
  -v /mnt/game:/mnt/game -v /home/lyh/z100-vllm:/src:ro \
  z100-rocm10.1:24.04-r642 sleep infinity

docker exec z100-vllm-rocm101-rt bash -lc '
  sed -i "s|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g; \
          s|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g" /etc/apt/sources.list.d/ubuntu.sources
  DEBIAN_FRONTEND=noninteractive apt-get update -qq &&
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      python3-pip python3-dev python3-venv zlib1g-dev \
      libopenblas0 liblapack3'
```

> 镜像自带 gcc 13.3 / cmake 3.28 / ninja / git / Python 3.12.3；**缺** pip、python3-dev、zlib1g-dev、
> libopenblas.so.0（缺后 `import torch` 直接失败）。apt 走 aliyun 镜像实测可用。

#### P0.2 Python 栈安装（约 2 min，全部本地 wheel）

```bash
docker exec z100-vllm-rocm101-rt bash -lc '
  python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  python3 -m pip config set global.break-system-packages true
  python3 -m pip install -q --no-deps --break-system-packages \
    /mnt/game/rocm10-wheels/rocm_sdk_core-10.1.0-py3-none-linux_x86_64.whl \
    /mnt/game/rocm10-wheels/rocm_sdk_libraries-10.1.0-py3-none-linux_x86_64.whl \
    /mnt/game/rocm10-wheels/rocm_sdk_devel-10.1.0-py3-none-linux_x86_64.whl   # torch RUNPATH + amdsmi 需要
  # amdsmi：用 SDK 自带 27.1.0（不用 PyPI 7.0.2）。该目录不是 pip 包（无 setup.py），
  # 用 PYTHONPATH 提供（见 P0.3 env）；也可 cp -a .../share/amd_smi/amdsmi $SP/ 固化。
  python3 -m pip install -q --no-deps --break-system-packages --force-reinstall \
    /mnt/game/torch-rocm10/wheel-sfu/torch-2.16.0.dev20261016+rocm10.1-cp312-cp312-linux_x86_64.whl
  python3 -m pip install -q --break-system-packages --upgrade \
    filelock fsspec jinja2 networkx sympy typing-extensions "setuptools>=77"
'
```

- **triton 不 pip 安装**，用 `PYTHONPATH=/mnt/game/triton-z100-rocm101/install`（可回滚、零污染）；
  若必须安装，用 `dist/triton-3.6.0-cp312-cp312-linux_x86_64.whl --no-deps`。
- **roc-sdk-devel 也要装进 RT**（约 +3 GB）：torch wheel 的 RUNPATH 里写死了
  `_rocm_sdk_devel/lib`（librccl 的 NEEDED 解析路径），且 **amdsmi 27.1.0 的 python 包在
  `_rocm_sdk_devel/share/amd_smi/`**（直接加进 PYTHONPATH 即可；该目录不是 pip 包）。
  **不要**装 PyPI `amdsmi==7.0.2`（版本/ABI 都不对）。见 P0.3 env 脚本。
- **torchvision/torchaudio 不装**：纯文本路径不需要（processor 是懒加载；多模态阶段再议）。
- vLLM 运行依赖（common.txt 子集）在 P2.1 安装；测试依赖（pytest/tblib 等）P3 前安装。

#### P0.3 环境契约脚本 `tools/dcu/env-rocm101.sh`（新文件，入 repo）

从 `tools/dcu/env.sh`（DTK 版）派生，去掉 `DTKROOT` 依赖，核心内容：

```bash
# ROCm 10.1 / TheRock
export ROCM_PATH=/opt/rocm/core-10.1
export HIP_PATH=$ROCM_PATH
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
SP=/usr/local/lib/python3.12/dist-packages
export LD_LIBRARY_PATH=/mnt/game/rccl-gfx906/lib:$SP/_rocm_sdk_libraries/lib:$SP/_rocm_sdk_core/lib:$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export USE_LIBUV=0                      # torch 编译时未开 libuv（必需）
export RCCL_USE_AMD_SMI_LIB=1           # Hygon vendor 0x1d94 的 ARSMI 映射（必需）
export HSA_FORCE_FINE_GRAIN_PCIE=1      # RCCL P2P；ROCm10.1 上待 A/B（Q4）
export VLLM_DCU_GCN_ARCH=gfx906         # 固定 arch 解析，避免 import 期初始化 HIP
export VLLM_DCU_BF16=auto               # 与 D6 一起定
export VLLM_ROCM_USE_SKINNY_GEMM=0      # gfx906 无 MFMA
export VLLM_USE_AOT_COMPILE=0           # bring-up 期（D4）
export PYTHONPATH=/mnt/game/port/gfx906-vllm-rocm101:/mnt/game/triton-z100-rocm101/install:$SP/_rocm_sdk_devel/share/amd_smi:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/mnt/game/triton-z100-rocm101/cache/jit
export VLLM_CACHE_ROOT=/mnt/game/vllm-cache-rocm10
# amdsmi 27.1.0 由 PYTHONPATH 里的 _rocm_sdk_devel/share/amd_smi 提供（P0.2 已装 devel wheel）
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # 沿用 DTK 时代稳妥选择
export TOKENIZERS_PARALLELISM=false
# 不要设置 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True（见 §9.2-F3）：
# torch 2.16 + ROCm 10.1 gfx906 上 expandable segments 的 VMM 路径会让首次
# GPU 分配永久挂死在 drm_syncobj_timeline_wait_timeout；unset 即正常。
unset PYTORCH_HIP_ALLOC_CONF 2>/dev/null || true
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
# 注意：不要 unset 镜像自带的 ROCBLAS_TENSILE_LIBPATH —— ROCm 10.1 的 rocBLAS 有完整
# gfx906 Tensile 目录（/opt/rocm/core-10.1/lib/rocblas/library/gfx906，210 个文件），镜像 ENV 已设对。
# DTK 时代"不要设"的结论不适用于 ROCm 10.1。
```

> `USE_LIBUV=0`：torch 未编 libuv（`USE_TENSORPIPE=0`），不设会 `DistStoreError: use_libuv was
> requested but PyTorch was built without libuv support`。
> `RCCL_USE_AMD_SMI_LIB=1`：不设会 `ARSMI_lib: <BDF> device index not found`。

#### P0.4 栈自检（L0，约 5–8 min）

```bash
docker exec z100-vllm-rocm101-rt bash -lc '
  . /src/tools/dcu/env-rocm101.sh
  python3 /mnt/game/torch-rocm10/verify-gpu.py            # ALL_PASS
  python3 /mnt/game/torch-rocm10/verify-numerics.py       # NUMERIC_GATE_PASS
  python3 -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.get_arch_list(), torch.cuda.device_count())"
  python3 -c "import triton;from triton.runtime import driver;print(triton.__version__, triton.__file__);print(driver.active.get_current_target())"
'
```

期望：torch `2.16.0.dev20261016+rocm10.1` / hip `7.16.26385` / `['gfx906']` / 4；
triton `3.6.0` + `GPUTarget(backend=\'hip\', arch=\'gfx906\', warp_size=64)`。
**失败处置**：softmax 误差 0.05 或 tanh=inf ⇒ 装错 wheel（非 wheel-sfu）；`triton.__file__` 不在
`/mnt/game/triton-z100-rocm101/install` ⇒ PYTHONPATH 错；4 卡不可见 ⇒ `/dev/kfd` 挂载/env。

#### P0.5 RCCL gfx906 重编（P0-R，与 P0 并行，独立容器）

> **这是 TP>1 的唯一通路。** 若失败，按 §8-R1 降级为"单卡交付"，并升级给用户决策。

1) 取 RCCL 源码。安装的 RCCL 版本字符串为 `2.30.7-HEAD:7da4302`（commit 前缀 `7da4302`）：

```bash
# 宿主机（网络可达；GitHub 实测可用）
SHA=$(curl -s https://api.github.com/repos/ROCm/rccl/commits/7da4302 | python3 -c 'import json,sys;print(json.load(sys.stdin)["sha"])')
echo "rccl commit=$SHA"
curl -L --retry 3 -o /mnt/game/rccl-$SHA.tar.gz https://codeload.github.com/ROCm/rccl/tar.gz/$SHA
mkdir -p /mnt/game/rccl-src && tar xzf /mnt/game/rccl-$SHA.tar.gz -C /mnt/game/rccl-src
# 若 commit 在 upstream 不存在（TheRock 的 fork）：改用 ROCm/TheRock rock-6.4.2 清单里的 rccl pin（§9-Q5）
```

2) 在 `BUILD` 容器里编译（ROCm 10.1 工具链；allreduce kernel 不用超越函数，**不需要**SFU 补丁 clang，
   用 stock clang 24 即可；若后续发现数值问题再换补丁 clang 重编）：

```bash
docker exec z100-rocm10-build bash -lc '
  . /mnt/game/port/setenv-rocm10.sh
  SRC=/mnt/game/rccl-src/rccl-<sha>
  cmake -S "$SRC" -B /mnt/game/rccl-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/mnt/game/rccl-gfx906 \
    -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/llvm/bin/clang++ \
    -DCMAKE_HIP_ARCHITECTURES=gfx906 \
    -DAMDGPU_TARGETS=gfx906 \
    -DROCM_PATH=/opt/rocm/core-10.1 \
    -DBUILD_TESTS=OFF
  ninja -C /mnt/game/rccl-build -j8 && ninja -C /mnt/game/rccl-build install'
```

3) 验收（三条都要过）：

```bash
# a) device code 真的含 gfx906
/opt/rocm/core-10.1/llvm/bin/llvm-objdump --offloading /mnt/game/rccl-gfx906/lib/librccl.so.1 \
  | grep -o 'gfx906' | sort | uniq -c
# b) 新 .so 被进程采用（不是 SDK 里那份）
LD_LIBRARY_PATH=/mnt/game/rccl-gfx906/lib:... python3 -c "import torch;print(torch.cuda.nccl.version())"
# c) 4 卡 all_reduce（5120 elem + 1 MiB，两档都要），要求无 NaN/hang
docker exec z100-vllm-rocm101-rt bash -lc '
  . /src/tools/dcu/env-rocm101.sh
  torchrun --standalone --nproc_per_node=4 /path/to/allreduce_smoke.py'
```

**最小 allreduce 冒烟脚本**（P0 新建，落 `tools/dcu/`）：`torch.distributed.init_process_group("nccl")` →
每 rank `x = torch.ones(5120, device="cuda")` → `all_reduce(x, op=SUM)` → 断言 `x == 4` + 1 MiB 张量重复一次。

**失败处置**：
- 源码 commit 取不到 → 用 upstream `develop` 里版本最接近的 2.30.x 提交（ABI 靠 `nm -D` 对齐 NCCl 符号），
  标记"未与 SDK 完全同源"；
- 编译报错 → 先看 RCCL 支持的 `AMDGPU_TARGETS` 语法（`gfx906;gfx908` 或 `gfx906`）；
- 编译过但 all_reduce 仍 `invalid device function` → `llvm-objdump` 复查、`NCCL_DEBUG=INFO` 看
  "using device code"；必要时用补丁 clang 重编 device 部分；
- 仍不行 → 记录证据，启用 §8-R1 降级路径（先交付 TP=1 功能，多卡另行立项）。

#### P0 产出/门禁

- [ ] `z100-vllm-rocm101-rt` 容器可用，`/src/tools/dcu/env-rocm101.sh` 提交；
- [ ] L0 表全绿（torch 身份/verify-gpu/verify-numerics/triton G0-G7/SFU 变异/import 顺序/platform 属性）；
- [ ] RCCL：`llvm-objdump` 含 gfx906 + 4 卡 all_reduce PASS（或登记为阻塞并进入降级路径）。

---

### 6.2 P1 —— 用新 torch + SFU 补丁 clang 重编 vLLM 4 扩展

> 目标：产出 **ABI 对应新 torch、device code 由补丁 clang 生成**的 4 个 .so，并通过 L1 加载/注册门。
> 预计 **30–60 min**（上一轮 4 扩展 ~7 min，但本轮要清空重建、另加首编译缓存）。

#### P1.1 准备可写副本与构建目录

```bash
# 宿主机
BR=rocm101-torch216
git -C /home/lyh/z100-vllm switch -c $BR        # 从当前 HEAD 拉新分支（唯一代码真相）
rsync -a --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='*.so' --exclude='.venv' \
  /home/lyh/z100-vllm/ /mnt/game/port/gfx906-vllm-rocm101/
mkdir -p /mnt/game/port/build-rocm101 /mnt/game/port/gfx906-vllm-rocm101/{vllm,temp}
```

#### P1.2 编译容器装新 torch（BUILD 容器改造）

```bash
docker exec z100-rocm10-build bash -lc '
  . /mnt/game/port/setenv-rocm10.sh
  python3 -m pip install -q --no-deps --break-system-packages --force-reinstall \
    /mnt/game/torch-rocm10/wheel-sfu/torch-2.16.0.dev20261016+rocm10.1-cp312-cp312-linux_x86_64.whl
  python3 -c "import torch;print(torch.__version__, torch.version.hip)"'
```

#### P1.3 构建（对 `/mnt/game/port/{configure,build}-rocm10.sh` 改 4 处）

1. `REPO_DIR=/mnt/game/port/gfx906-vllm-rocm101`、`BUILD_DIR=/mnt/game/port/build-rocm101`；
2. 加 `HIP_CLANG_PATH=/mnt/game/torch-rocm10/llvm-build/bin`，`PATH=$HIP_CLANG_PATH:$PATH`；
3. CMake 追加：
   `-DCMAKE_HIP_COMPILER=$HIP_CLANG_PATH/clang++ -DCMAKE_HIP_ARCHITECTURES=gfx906`；
4. **不再传** `-DCMAKE_HIP_FLAGS=-mcode-object-version=4`（与 torch 的 V6 对齐；若 L1 加载失败回退 V4）；
   显式给 `-DCMAKE_HIP_ARCHITECTURES=gfx906`（同时保留 `PYTORCH_ROCM_ARCH=gfx906`，两者不冲突）。

```bash
docker exec -u 1000:1000 z100-rocm10-build bash -lc '
  export REPO_DIR=/mnt/game/port/gfx906-vllm-rocm101
  export BUILD_DIR=/mnt/game/port/build-rocm101
  . /mnt/game/port/setenv-rocm10.sh
  export HIP_CLANG_PATH=/mnt/game/torch-rocm10/llvm-build/bin
  export PATH=$HIP_CLANG_PATH:$PATH
  mkdir -p /mnt/game/port/build-rocm101/temp && cd /mnt/game/port/build-rocm101/temp
  PY_STR="${REPO_DIR}:$(python3 -c "import sys;print(\":\".join(sys.path))")"
  cmake "${REPO_DIR}" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DVLLM_TARGET_DEVICE=rocm -DVLLM_PYTHON_EXECUTABLE="$(which python3)" \
    -DVLLM_PYTHON_PATH="$PY_STR" \
    -DCMAKE_HIP_COMPILER=$HIP_CLANG_PATH/clang++ \
    -DCMAKE_HIP_ARCHITECTURES=gfx906 \
    -DROCM_PATH=/opt/rocm/core-10.1 \
    -DFETCHCONTENT_BASE_DIR=/mnt/game/port/rocm10-deps \
    -DCMAKE_JOB_POOLS:STRING=compile=4 2>&1 | tee /mnt/game/port/configure-rocm101.log
  rm -f CMakeFiles/hipify_all.stamp
  ninja hipify_all > /tmp/hipify-rocm101.log 2>&1
  MAX_JOBS=4 cmake --build . -j4 \
    --target=_C --target=_C_stable_libtorch \
    --target=_moe_C_stable_libtorch --target=_rocm_C 2>&1 | tee /mnt/game/port/build-rocm101.log
  cmake --install . --prefix "${REPO_DIR}" --component _C
  cmake --install . --prefix "${REPO_DIR}" --component _C_stable_libtorch
  cmake --install . --prefix "${REPO_DIR}" --component _moe_C_stable_libtorch
  cmake --install . --prefix "${REPO_DIR}" --component _rocm_C
  ls -l "${REPO_DIR}"/vllm/*.so'
```

**关键校验（编译期）**：
- [ ] 日志中 HIP 编译使用 `/mnt/game/torch-rocm10/llvm-build/bin/clang++`（可 `grep -c` 编译命令）；
- [ ] 每个 .so 的 device code 只含 gfx906：
  `strings -a <so> | grep -oE 'amdgcn-amd-amdhsa--gfx[0-9a-f]+' | sed 's/.*--//' | sort -u` → 只有 `gfx906`；
- [ ] **SFU 补丁生效抽查**：反汇编一个已知 SFU kernel（如 gfx906 layernorm kernel 的 `.hsaco`/`.o`），
  确认 SFU 指令前有 `s_nop`（用 `/mnt/game/torch-rocm10/llvm-build/bin/llvm-objdump -d`）；
- [ ] 4 个 .so 的 md5 记录进 `docs/` 或基线目录（新栈识别码）。

#### P1.4 L1 加载/注册门（在 RT 容器）

```bash
docker exec z100-vllm-rocm101-rt bash -lc '
  . /src/tools/dcu/env-rocm101.sh
  cd /mnt/game/port/gfx906-vllm-rocm101
  python3 -c "import torch, vllm, vllm._C, vllm._C_stable_libtorch, vllm._rocm_C; print(\"IMPORT_OK\", vllm.__file__)"
  python3 -c "import torch, vllm._rocm_C; print(sorted(s.name for s in torch._C._jit_get_all_schemas() if s.name.startswith(\"_rocm_C::\")))"
  python3 -c "import torch, vllm._C; print(hasattr(torch.ops._C, \"rms_norm_gated_gfx906\"), hasattr(torch.ops._C, \"ggml_dequantize\"))"'
```

期望：`IMPORT_OK`；`_rocm_C` 含 `w8a8_sdot4_gemv`/`fp8_emu_*`/`bf16_emu_*`；`_C` 含
`rms_norm_gated_gfx906`/`ggml_dequantize`；`ROCM_SKINNY_GEMM_ENABLED == False`。
**失败处置**：`undefined symbol`/ABI 报错 ⇒ 编译时找错 torch（BUILD 容器里可能仍是 2.9.1）；
加载 `hipErrorInvalidImage` ⇒ code object 版本不是 driver 支持的 → 回退 `-mcode-object-version=4` 重编。

#### P1 产出/门禁

- [ ] 4 个新 .so（md5 登记），device code 仅 gfx906；
- [ ] L1 全绿（import/注册/arch/版本/env）。

---

### 6.3 P2 —— 运行期与代码适配

> 目标：在新栈上 `import vllm` 成功、平台探测正确、`vllm serve` 能起 `Qwen3-0.6B`。
> 源码级"必须改"只有 3 处 gate + 1 个 merge 残留 bug；其余是环境/依赖/策略。

#### P2.1 vLLM 依赖安装（RT 容器）

```bash
docker exec z100-vllm-rocm101-rt bash -lc '
  . /src/tools/dcu/env-rocm101.sh
  # 运行依赖（common.txt 全量；torch/triton 已单独装，故 --no-deps 安装这些包）
  python3 -m pip install --break-system-packages -q \
    regex cachetools psutil sentencepiece numpy requests tqdm blake3 py-cpuinfo \
    "transformers>=5.16.1,<5.20.0" "huggingface_hub>=1.31.0" "tokenizers>=0.23.2" \
    "safetensors>=0.6.2" "protobuf>=5.29.6" "fastapi[standard]>=0.133.0,<0.137.0" \
    "starlette>=1.0.1" "aiohttp>=3.14.3" "openai>=2.25.0" "pydantic>=2.12.0" \
    prometheus_client "pillow>=12.3.0" prometheus-fastapi-instrumentator "tiktoken>=0.6.0" \
    lm-format-enforcer llguidance outlines_core lark xgrammar typing_extensions filelock \
    partial-json-parser "jsonschema>=4.23.0" pyzmq msgspec "mistral_common[image]>=1.11.5" \
    opencv-python-headless pyyaml six einops compressed-tensors depyf cloudpickle watchfiles \
    python-json-logger ninja pybase64 cbor2 ijson setproctitle oss-harmony anthropic mcp \
    opentelemetry-sdk opentelemetry-api opentelemetry-exporter-otlp opentelemetry-semantic-conventions-ai \
    model-hosting-container-standards
  # 测试依赖
  python3 -m pip install --break-system-packages -q pytest pytest-asyncio tblib
'
```

**冲突清单（对照 `requirements/rocm.txt`/`rocm-build.txt`，新栈一律不要装）**：
`--extra-index-url rocm7.2`、`torch/torchvision/torchaudio==…+rocm7.2`、`triton-rocm`、
`amdsmi==7.0.2`、`numba==0.61.2`（可用 0.65+ 替换，仅 N-gram spec decode 需要）、`tilelang`、
`apache-tvm-ffi`、`amd-quark`、`conch-triton-kernels`、`mooncake-transfer-engine-rocm`、`ray`、
`tensorizer`、`fastsafetensors`、`timm`、`datasets`/`peft`（按需）。`packaging/setuptools/setuptools-scm` 保留。

#### P2.2 代码/配置改动（新分支提交）

| # | 文件 | 改动 | 类型 |
|---|---|---|---|
| 1 | `vllm/platforms/rocm.py:281-300` | gfx906 bf16 分支：`is_hygon_dtk() and gfx906_bf16_enabled()` → `gfx906_bf16_enabled()`（探针/`VLLM_DCU_BF16` 三态保持不变） | **必须（D6，二选一，见 §9-Q1）** |
| 2 | `vllm/platforms/hygon_dtk.py:251-260` | `rocminfo_candidates()` 增加 `$ROCM_PATH/bin/rocminfo`（TheRock 布局没有 `/opt/rocm/bin`） | 必须（无 amdsmi 时平台探测） |
| 3 | `setup.py:1268-1269` | merge 残留：`dcu.txt if is_hygon_dtk() else rocm.txt` 被下一行无条件覆盖 → 修正并加 `rocm101` 分支 | 必须（仓库卫生/安装路径） |
| 4 | `requirements/rocm101.txt` / `requirements/rocm101-build.txt` | 新文件：torch 指向本地 wheel 说明、triton 指向本地 wheel、amdsmi 说明、去掉 rocm7.2 index | 必须（D7） |
| 5 | `CMakeLists.txt:71-72` | `TORCH_SUPPORTED_VERSION_ROCM` 2.14.0 → 2.16.0（仅 warning） | 建议 |
| 6 | `vllm/utils/torch_utils.py:999,1007` | `torch._opaque_base` 在 2.16 deprecated；加 `_custom_class_base` 优先、`_opaque_base` 兜底 | 建议（warning） |
| 7 | `vllm/envs.py` / 新 env 脚本 | `VLLM_USE_AOT_COMPILE=0` bring-up 默认（代码已有 2.10+ 默认开） | 必须（D4，环境层） |
| 8 | `tools/dcu/env-rocm101.sh` + `tools/dcu/use_stack.sh` | 新环境契约；`use_stack.sh dtk\|rocm101` relink 4 个 .so + 打印 md5 | 必须（回滚/纪律） |

> 平台探测兜底：即使不装 amdsmi、也不改 rocminfo，只要运行脚本显式给 `HIP_VISIBLE_DEVICES`
> （fork 的 `rocm_platform_plugin` 会直接选 ROCm 平台）即可。本计划仍建议装 amdsmi（版本/温度/拓扑）。

#### P2.3 平台/导入自检（L1.5 + 0.6B 冒烟门）

```bash
docker exec z100-vllm-rocm101-rt bash -lc '
  . /src/tools/dcu/env-rocm101.sh
  python3 -c "import torch, vllm; from vllm.platforms import current_platform as p; \
    print(p.device_name, p.is_rocm(), p.get_device_capability(), p.get_device_name(0), p.supported_dtypes)"
  python3 -c "import torch; from vllm.entrypoints.cli.main import main; main()" serve \
    /mnt/game/models/Qwen3-0.6B --dtype bfloat16 --served-model-name qwen3-0.6b \
    --tensor-parallel-size 1 --max-model-len 2048 --gpu-memory-utilization 0.30 \
    --enforce-eager --host 127.0.0.1 --port 8020 --log-file /tmp/smoke06.log &
  for i in $(seq 60); do curl -sf http://127.0.0.1:8020/v1/models >/dev/null && break; sleep 2; done
  curl -s http://127.0.0.1:8020/v1/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"qwen3-0.6b\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0}"'
```

期望：`device_name=rocm`、`(9, 0)`、dtype 列表含 fp16/bf16（按 D6 决策）、服务 ready、输出 ` Paris`。
**失败处置**：`ImportError: amdsmi` 仅 warning，不致命；`No platform detected` → 设 `HIP_VISIBLE_DEVICES`
或装 amdsmi/改 rocminfo；`Dtype bfloat16 is not supported` → D6 未落地（L1.5 门）；GDN/attention 无关注。

#### P2 产出/门禁

- [ ] 新分支提交（上表改动 + env 脚本 + use_stack 脚本）；
- [ ] `import vllm`（torch-first）成功、平台属性正确、`vllm serve Qwen3-0.6B` 冒烟通过；
- [ ] 依赖清单固化（可直接重建容器）。

---

### 6.4 P3 —— 正确性验证（L2 kernel 数值 + L3 G8/G9）

> 前置：**L-1 同树 DTK 基线已冻结**（见 §7-L-1），否则 L3 token 对比没有基准。
> 执行顺序：L2（快 5 个文件）→ L2 慢文件 → G8 → G9 → GDN pytest 交叉。预计 0.5–1 天（含失败定位）。

1. **L2 单文件单进程**跑 13 个 gfx906 测试文件（命令模板见 §7-L2）。
   判定：`新栈 failures ⊆ DTK failures`；DTK 通过而新栈失败 = 门禁失败。
   已知 expected-fail 与"新增长失败"见 §7-L2 / `verify-plan.md` §2.3。
2. **G8**：用 `/mnt/game/hygon-qwen-scripts/probe_gdn_compare2.py`（先打印 `triton.__file__` 与 target），
   判据：6 配置 **NaN=0 且 `max|diff| ≤ 1e-4`**。
3. **G9**：`Qwen3.5-4B` TP=1、`--dtype bfloat16`、eager、3 prompt greedy 出 ` Paris.`/` 2`/中文正确，
   并与 DTK 冻结 JSON **逐 token 对比**（首个分歧 top1-top2 margin < 0.01 nat 视为近 tie 可放行）。
   同时跑 `Qwen3-0.6B`（dense 对照）二分故障域。
4. **GDN pytest 交叉**：`tests/model_executor/test_qwen3_5_gdn.py`（47 items，≥44 passed）、
   `tests/v1/attention/test_gdn_metadata_builder.py`（62，单文件跑）、`tests/parser/engine/test_qwen3.py`（84）。

**失败定位树**（来自 `verify-plan.md` §3.2）：
G8 fail & G9 fail ⇒ GDN Triton 内核（回 L0.4/L0.5 复核 triton/SFU 门禁）；
G8 pass & G9 fail & 0.6B pass ⇒ GDN/模型层路径（比对 DTK 的 FLA 调用参数）；
G8 pass & 0.6B fail ⇒ attention/量化/通用路径；全 NaN/爆炸 ⇒ 先查 triton build。

#### P3 产出/门禁：L2、L3 全绿（或已登记且 Lead 签字放行的差异），报告落 `docs/result/`。

---

### 6.5 P4 —— 精度 / 性能 / 多卡（L4 + L5）

> 前置：L3 通过；RCCL（P0-R）通过（否则 L4 只做 TP=1，L5 挂起）。

1. **L4.1 跨栈 greedy parity（最强正确性口径）**：27B-FP8 TP=4 或 4B TP=1，12 prompt × 64 token，
   top-5 logprobs，两栈各一份，`tools/dcu/s11l_flip_compare.py` 对比（规则见 §7-L4）。
2. **L4.2/4.3 质量电池**：`probe_quality.py`（Paris/1+1/中文首都 3/3）、
   `qfix_quality_battery.py --label rocm10`（tool_call/JSON/reasoning）。
3. **L4.5–4.7 性能**：PERF1 anchor（p512/g64×3/conc=1/temp=0）+ 全长度扫描（119/451/1806/7443）
   + `vllm bench serve`（对外口径）。阈值：同树基线 decode ≥0.95×、step p50 ≤1.05×、TTFT ≤1.10×；
   跨树基线放宽到 0.90×/1.10×/1.15×。
4. **L5 多卡/稳定性**：4B TP 曲线（TP1/2/4）+ 27B-FP8 TP=4 生产配方启动（ready 时间/KV cache 记录）
   + 并发混合负载 + 30–60 min 稳态（VRAM 漂移 <200 MiB、无 NaN/500/worker 重启）+ 回滚演练。
5. **（可选）compile 实验臂**：`VLLM_USE_AOT_COMPILE=1` + `-O2`（非 eager）只作为附加数据，
   **不进 DoD**；若失败不阻塞交付（记录原因：triton 3.6 vs torch 2.16 期望 3.9）。

**失败定位**：perf 掉但 parity 过 ⇒ 派发/内核选择（日志搜 `Selected .*LinearKernel`/attention backend/
`VLLM_DCU_*` 是否生效）；parity 掉但 perf 过 ⇒ 数值顺序变化（回看 L3）。TP 相关失败 ⇒ RCCL/P2P
（`NCCL_DEBUG=INFO`）+ `HSA_FORCE_FINE_GRAIN_PCIE` A/B。

#### P4 产出/门禁：L4、L5 全绿；perf 对比表落 `docs/perf/`（新栈基线）。

---

### 6.6 P5 —— 固化与交付

- [ ] 分支 `rocm101-torch216` 整理（commit 信息含"为什么/验证命令/日志路径"），合入 `sync/upstream-20261016-v2` 或保留分支并打标签；
- [ ] 交付脚本：`tools/dcu/env-rocm101.sh`、`use_stack.sh dtk|rocm101`、`serve_rocm101.sh`（生产 27B-FP8 配方 ROCm10.1 版）、`build-rocm101.sh`、`rccl-gfx906-build.sh`；
- [ ] 容器固化：`z100-vllm-rocm101-rt` 的完整构建命令写入 `docs/`（可重建）；是否提交为镜像由用户决定；
- [ ] 验收报告：`docs/result/z100-rocm101-acceptance.md`（L0–L5 表 + .so md5 + git HEAD + 原始日志路径 + 已知差异）；
- [ ] 回滚演练记录（relink DTK .so → 重启服务 → G9 仍过）；
- [ ] 更新 `docs/z100-rocm10-build.md` 顶部指针，避免后人拿旧结论（"hipErrorInvalidImage"）误导。

---

## 7. 验证矩阵（L-1 → L5）

> 以下矩阵来自 `/tmp/z100-adapt-plan/verify-plan.md`（已含命令/阈值/耗时/失败定位），执行时按表逐项勾选。

### L-1 —— 冻结同树 DTK 基线（**P0 期间并行做，必须先于切换**，约 1–1.5 h）

在 `REG` 容器、当前树 DTK 产物（`/mnt/game/port/build-sync/temp/` 4 个 .so）上抓：
1. 13 个 gfx906 测试文件（`--junitxml`）→ `/mnt/game/z100-baseline/dtk-pytest.xml`；
2. G9 token ids：Qwen3.5-4B + Qwen3-0.6B 各 3 prompt × 12 token（JSON）；
3. G8：`probe_gdn_compare2.py` 6 行输出；
4. perf：PERF1 anchor（p512/g64×3）→ 同树 `B0_anchor`；
5. 记录 4 个 .so md5 + `git rev-parse HEAD`。

### L0 —— 环境/栈自检（阻塞；≈5–8 min）

| # | 检查 | 判定 |
|---|---|---|
| L0.1 | torch 身份 | `2.16.0.dev20261016+rocm10.1` / `7.16.26385` / `['gfx906']` / 4 卡 |
| L0.2 | `verify-gpu.py` | `ALL_PASS` |
| L0.3 | `verify-numerics.py` | `NUMERIC_GATE_PASS`（softmax≤1e-6、tanh/erf≤1e-6、layer_norm≤1e-6、decoder_block≤2e-3） |
| L0.4 | triton G0–G7 | 8/8 PASS（G2 `s_nop=1`、G5 `v_mfma=0`） |
| L0.5 | SFU 变异门禁 | 正向 0 mismatch；负向复现且 100% lane 8–11 |
| L0.6 | 4 卡 + RCCL | 4 rank all_reduce 5120 elem + 1 MiB 无 NaN/hang |
| L0.7 | import 顺序 | `import vllm` 先 vs `import torch` 先均成功（记录约束） |
| L0.8 | 平台属性 | ROCm；`on_gfx906()` 真；capability `(9,0)`；bf16 按 D6 |
| L0.8b | **`is_hygon_dtk()` 必须为 False** | 新镜像无 `/opt/dtk`、`/opt/hyhal`、`DTKROOT`；若为 True 会误开全部 DTK 分支（高危） |

### L1 —— 扩展加载/算子注册（阻塞；≈2–3 min）

4 个 .so 存在 + md5；device code 仅 gfx906；`import vllm`/`vllm._C`/`vllm._rocm_C` OK；
`_rocm_C` 算子表与 `_C` 算子（`rms_norm_gated_gfx906`/`ggml_dequantize`）注册 OK；
`ROCM_SKINNY_GEMM_ENABLED=False`；`tests/platforms/test_hygon_dtk.py` 52 passed。

### L2 —— kernel 数值门禁（核心；≈10–20 min）

13 个 gfx906 测试文件**逐文件单进程**（`--junitxml` 留档）：
`test_hygon_dtk.py`（52）、`test_gfx906_fp8_emu.py`、`test_gfx906_int8_sdot4.py`、
`test_gguf_fixed_cols_gfx906.py`、`test_gfx906_bf16_gemv.py`、`test_gfx906_packed_silu_fp16.py`、
`test_fused_sigmoid_gating.py`（5）、`test_fused_recurrent_packed_decode.py`、
`test_triton_unified_attention_gfx906*.py`、`test_causal_conv1d.py`（227）、
`tests/kernels/core/test_layernorm.py`/`test_activation.py`/`test_fused_rms_norm_gated.py`。

**判定**：新栈 failures ⊆ DTK failures（expected-fail 清单见 `verify-plan.md` §2.3）；
**任何 DTK 通过而新栈失败 = FAIL**。`PassManager::run failed` 若消失 = 加分；若仍出现需确认是
triton 3.6(rocm101) 还是 kernel 本身。

### L3 —— G8 + G9（决定性门；≈10–20 min）

- G8：6 配置 **NaN=0 且 max|diff| ≤ 1e-4**（T6 正式判据）；
- G9：Qwen3.5-4B bf16 TP=1 eager 3 prompt 正确输出 + 与 DTK 冻结 token 流一致（近 tie 规则）；
  Qwen3-0.6B 对照一致；**同时检查 DTK 冻结基线本身的 NaN/有限性**（F1：DTK 当前有 packed-SiLU 回归，
  若 DTK 基线输出有 inf/NaN，则 L-1 基线无效，需先修 F1 或用 fp16 之外口径）；
- GDN pytest 交叉：47 items（≥44 passed）/ 62 passed / 84 passed。

### L4 —— 模型级精度 + perf（≈1–1.5 h）

跨栈 greedy parity（27B-FP8 或 4B）；质量电池 3/3；PERF1 anchor + 全长度扫描 + `vllm bench serve`；
阈值见 §6.5。GSM8K **不纳入**（本机无数据集，用跨栈 parity 替代）。

### L5 —— 多卡 TP=4 + 稳定性（≈2–3 h）

4B TP 曲线（TP1/2/4，decode ≥0.90× DTK，TTFT ≤1.15×，TP 单调性）；
27B-FP8 TP=4 生产配方启动（ready ≤1.5× DTK，记录 KV cache）；并发混合负载；
30–60 min 稳态（VRAM 漂移 <200 MiB、无 NaN/500/worker 重启）；回滚演练。

**总耗时预算**：L-1 ≈1–1.5 h；切换后 L0–L5 ≈4.5–6.5 h（不含失败重跑/修复）。

---

## 8. 风险登记表

| # | 风险 | 概率/影响 | 触发信号 | 对策 |
|---|---|---|---|---|
| **R1** | **RCCL 重编失败/源码不可得**（多卡硬阻塞） | 中/高 | 编译报错 / all_reduce 仍 `invalid device function` | 先交付 TP=1 功能与全部 L0–L4（单卡）门禁；把 RCCL 单独立项，升级用户决策；DTK 继续承担 TP=4 生产 |
| R2 | GDN/FLA 在新 triton 上数值不达标（G8 一票否决） | 中/高 | G8 max|diff|>1e-4 或 NaN | 复核 triton build/SFU 门禁与 `triton.__file__`；用 triton-z100 的 gfx906 路由/SFU 修补流程定位；必要时回退 DTK |
| R3 | vLLM C++ kernel 仍有 SFU 类数值错误（漏编/漏路径） | 低/高 | L2 中 layernorm/activation/attention 与 fp64 偏差大 | 确认全部 4 扩展由补丁 clang 编译；抽查 SFU kernel 反汇编含 `s_nop`；必要时整树 clean 重编 |
| R4 | bf16 策略选错（D6）导致与 DTK 基线不可比/数值漂移 | 中/中 | G9 token 分歧、L2 bf16 用例异常 | 以 L-1 DTK 基线为准做 A/B（bf16 vs fp16 两臂）；默认按 Q1 结论 |
| R5 | 新 torch allocator/graph 行为差异（显存、hang） | 中/中 | L5 VRAM 漂移、`kfd_wait_on_events`、OOM | eager 口径；**不要开 `expandable_segments`（本栈会挂死，见 §9.2-F3）**；对比 DTK `Available KV cache` |
| R6 | 基线非同树导致 perf 判定争议 | 高/低 | 争论"退化"来源 | 先做 L-1 同树基线（本计划已排）；否则阈值放宽 0.90× 并注明量级比较 |
| R7 | 镜像缺依赖/网络受限导致环境重建失败 | 低/中 | apt/pip 不可达 | 全部大文件本地缓存（rocm wheels/torch/triton/models）；apt 走 aliyun、pip 走清华 |
| R8 | torch `USE_LIBUV=0`/RCCL env 漏配导致启动失败 | 中/低 | `DistStoreError` / `ARSMI_lib` | env 脚本固化（P0.3）；L0 门禁覆盖 |
| R9 | fork 特有特性（GGUF/INT8/DFlash2）在新栈未覆盖 | 中/低 | 相关测试/模型失败 | 首轮 DoD 不含它们；按需单独立项（模型权重/数据集缺） |
| R10 | 误用旧产物/错容器（stock clang 或 2.9.1 torch） | 中/中 | 数值异常、ABI 报错 | 每次作业先打印 `torch.__version__`/`triton.__file__`/4 个 .so md5（纪律固化） |
| R11 | **DTK 回滚基线本身有 merge 回归（F1）**：`activation_kernels.cu` 的 fp16 守卫被上游版本覆盖，packed-SiLU fp16 输出 inf | 中/高 | `test_gfx906_packed_silu_fp16.py` 3/3 fail、回滚后 G9 出现 NaN/inf | 切换前（L-1 阶段）先恢复 S7c 守卫并重编 DTK 的 `_C`；或至少冻结基线时记录该已知异常，回滚后不把 NaN 归因于新栈 |

---

## 9. 待确认 / 需拍板

| # | 问题 | 建议默认 |
|---|---|---|
| Q1 | gfx906 bf16 是否在新栈开放（D6）？生产 `--dtype float16` 不依赖，但 Qwen3.5-4B 的 G9 基线是 bf16 | **开放**（探针 auto + `VLLM_DCU_BF16` 可关），否则 G9 需把 DTK 基线也改成 fp16 重冻 |
| Q2 | 是否接受 RCCL 重编作为 P0 并行任务（需联网取源码） | **接受**；这是 TP=4 唯一通路 |
| Q3 | 新栈容器命名与是否要提交镜像 | 先命名 `z100-vllm-rocm101-rt`；镜像化放 P5 由用户决定 |
| Q4 | `HSA_FORCE_FINE_GRAIN_PCIE=1` 在 ROCm 10.1 是否仍需要 | L0.6 做 A/B；默认开（沿用 DTK 结论） |
| Q5 | RCCL 源码 commit `7da4302` 在 upstream 是否可解析（可能在 TheRock fork） | P0-R 第一步验证；失败按 §8-R1 降级 |
| Q6 | GSM8K / GGUF / DFlash2 / 多模态是否纳入验收 | 首轮**不纳入**（无权重/数据集）；用跨栈 parity 替代 |
| Q7 | 新栈是否允许跑 `torch.compile` 实验臂（triton 3.6 vs 期望 3.9） | 只作附加数据，不进 DoD；eager 为唯一质量口径 |
| Q8 | **F1（DTK packed-SiLU merge 回归）是否在切换前修** | **建议修**：DTK 是回滚目标，回滚后必须模型级正确；修法=恢复 `csrc/libtorch_stable/activation_kernels.cu` 的 S7c fp16 标量守卫并重编 DTK `_C`（约 0.5 h），在 L-1 前完成 |

### 9.1 已拍板（2026-10-18，用户确认）

| # | 决定 | 执行要求 |
|---|---|---|
| Q1 | **gfx906 bf16 开放** | 按 D6 把 `_rocm_supported_dtypes()` 的 gfx906 分支泛化为 `gfx906_bf16_enabled()` 探针；保留 `VLLM_DCU_BF16` 三态；G9 用 bf16 与 DTK 基线可直接对比 |
| Q2 | **接受 RCCL 重编为 P0 并行任务** | 按 §6.1-P0.5 取源码、`-DAMDGPU_TARGETS=gfx906` 重编、objdump + 4 卡 all_reduce 验收；失败按 R1 降级并升级 |
| Q8 | **切换前先修 F1** | 先恢复 S7c fp16 标量守卫（**注意：merge 已把文件移到 `csrc/activation_kernels.cu`**，守卫源在 `/mnt/game/port/gfx906-vllm-211/csrc/libtorch_stable/activation_kernels.cu`）并重编 DTK `_C_stable_libtorch`，再冻结 L-1 基线 |

### 9.2 执行期发现（2026-10-18，执行中新增/更正的事实）

| # | 发现 | 对计划的影响 |
|---|---|---|
| **F3** | **`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` 在本栈会让首次 GPU 分配永久挂死**：进程停在 `drm_syncobj_timeline_wait_timeout`（无限超时，等待 timeline point 1），CPU/GPU 0%、无 dmesg 报错，且会连锁让后续进程一起挂。A/B 实测（同容器同 wheel）：设置 → `torch.ones(4,device='cuda')` 挂死；**不设置 → alloc/sync/randn 全 OK，`verify-gpu.py` 全 PASS**。 | §6.1-P0.3 环境契约**必须 `unset PYTORCH_HIP_ALLOC_CONF`**（已改）；§10.2 同步。DTK 栈不适用（该值源于 DTK 时代）。这是 torch 2.16 + ROCm 10.1 gfx906 的 VMM 路径问题，已并入 R5 对策。 |
| **F4** | 镜像里的 `rocm_sdk_devel` wheel **只带 `$SP/rocm_sdk_devel/_devel.tar`**（3.1 GB），必须 `tar xf $SP/rocm_sdk_devel/_devel.tar -C $SP` 解包，torch RUNPATH 指向的 `_rocm_sdk_devel/lib` 与 amdsmi 的 `share/amd_smi` 才会存在；只 pip install 会导致 `import torch` 直接失败（RUNPATH 解析不到库）。 | §6.1-P0.2 补该步骤。 |
| **F5** | RCCL 源码实际在 **`ROCm/rocm-systems` monorepo** 的 `projects/rccl`，pin = `7da43026e4f2ef7600433c38c820732f7a205d22`（`release/therock-10.1` tip），与 SDK 完全同源；且 SDK `librccl.so.1` 的 `.hip_fatbin` 是 **NOBITS（完全无 device code）**，不只是缺 gfx906——这解释了 world_size=1 也报 `invalid device function`。 | §6.1-P0.5 源码获取路径更正；R1 的"重编是唯一通路"依据加强。 |
| **F6** | clang 24（SFU 补丁版）拒绝在函数级 `static constexpr` 初始化里调用 `__device__ constexpr` 的 `Utils::get_warp_size()`（`WARP_SIZE` 宏），报 `reference to __device__ function in global initializer`；改为普通 `constexpr` 局部即可（`csrc/activation_kernels.cu` 3 处，已修，commit `29d605c44`）。 | §6.2-P1 编译期已知修复项，已入库。 |
| **F7** | DTK `_C_stable_libtorch` 的 F1 重编最终身份 = **HEAD `29d605c44` + md5 `e303b0b2…`**（`590dc52e…` 是前一 HEAD `8b18b93cc` 的构建）；L-1 基线与新栈验收必须绑定同一 HEAD。 | §7-L-1 身份记录口径：以"实际 HEAD + 4 个 .so md5"为准。 |

---

## 10. 附录

### 10.1 关键路径速查

```
torch wheel     /mnt/game/torch-rocm10/wheel-sfu/torch-2.16.0.dev20261016+rocm10.1-cp312-cp312-linux_x86_64.whl
SFU clang       /mnt/game/torch-rocm10/llvm-build/bin/{clang++,clang-24,ld.lld}
triton          /mnt/game/triton-z100-rocm101/install（wheel: dist/triton-3.6.0-cp312-cp312-linux_x86_64.whl）
ROCm SDK        /mnt/game/rocm10-wheels/rocm_sdk_{core,libraries,devel}-10.1.0-*.whl
旧构建脚本      /mnt/game/port/{setenv,configure,build}-rocm10.sh
旧编译产物      /mnt/game/port/build-rocm10/temp/*.so（**不可复用，仅参考**）
新构建副本      /mnt/game/port/gfx906-vllm-rocm101/  +  /mnt/game/port/build-rocm101/
RCCL 新产物     /mnt/game/rccl-gfx906/lib/librccl.so.1
验收脚本        /mnt/game/torch-rocm10/{verify-gpu.py,verify-numerics.py}
G8/G9 脚本      /mnt/game/hygon-qwen-scripts/{probe_gdn_compare2.py,probe_quality.py,probe_transformers.py}
模型            /mnt/game/models/{Qwen3-0.6B,Qwen3.5-4B,Qwen3.8-27B-FP8}
DTK 基线 .so    /mnt/game/port/build-sync/temp/*.so
```

### 10.2 环境变量（新栈最小集）

```
ROCM_PATH=/opt/rocm/core-10.1  HIP_PATH=$ROCM_PATH
LD_LIBRARY_PATH=/mnt/game/rccl-gfx906/lib:$SP/_rocm_sdk_libraries/lib:$SP/_rocm_sdk_core/lib:$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib
HSA_OVERRIDE_GFX_VERSION=9.0.6   USE_LIBUV=0   RCCL_USE_AMD_SMI_LIB=1
HSA_FORCE_FINE_GRAIN_PCIE=1     VLLM_DCU_GCN_ARCH=gfx906   VLLM_DCU_BF16=auto
VLLM_ROCM_USE_SKINNY_GEMM=0     VLLM_USE_AOT_COMPILE=0
PYTHONPATH=/mnt/game/port/gfx906-vllm-rocm101:/mnt/game/triton-z100-rocm101/install
TRITON_CACHE_DIR=/mnt/game/triton-z100-rocm101/cache/jit
VLLM_CACHE_ROOT=/mnt/game/vllm-cache-rocm10
TOKENIZERS_PARALLELISM=false    # PYTORCH_HIP_ALLOC_CONF 保持未设置（F3）
VLLM_WORKER_MULTIPROC_METHOD=spawn   HIP_VISIBLE_DEVICES=0,1,2,3
```

### 10.3 回滚手册（无需 git 操作）

```bash
# 1) 把 4 个 .so relink 回 DTK 构建产物
REPO=/mnt/game/port/gfx906-vllm-rocm101   # 当前运行副本
DTK=/mnt/game/port/build-sync/temp
for so in _C _C_stable_libtorch _moe_C_stable_libtorch _rocm_C; do
  ln -sf $DTK/$so.abi3.so $REPO/vllm/$so.abi3.so
done
# 2) 用 DTK 环境重启服务（z100-regress / serve_fp8_8000.sh）
# 3) 跑 G9 冒烟确认回滚通道可用
```

### 10.4 与既有工件的关系

| 工件 | 本计划中的角色 |
|---|---|
| `/mnt/game/torch-rocm10/README.md` | 新 torch 的事实基座；**不要**再重编 torch（除非要带 libuv 根治 USE_LIBUV） |
| `/home/lyh/triton-z100/docs/rocm101-README.md` | triton 运行契约与 G0–G7 证据；G8/G9 由本计划 P3 补上 |
| `docs/z100-rocm10-build.md` | 上一轮（官方 wheel + stock clang）编译记录；结论已部分过时（`hipErrorInvalidImage` 根因已由自编 wheel 解释），P5 更新指针 |
| `docs/perf/PERF1-*` | 跨树性能基线；L-1 补同树基线后以同树为准 |
| `docs/archive/design/z100-04b-stack-upgrade.md` | DTK 内部升级（2.10→2.11）的同类流程模板，可借鉴容器/门禁纪律 |
| `/tmp/z100-adapt-plan/*.md` | 三份调研原始证据（本计划 §2/§6/§7 的来源） |

---

### 附：本计划使用的硬性纪律（执行 agent 必读）

1. **不覆盖** `/mnt/game/torch-rocm10/` 与 `/mnt/game/triton-z100-rocm101/` 下任何已有产物（只读使用）。
2. **不改** DTK 容器/镜像/生产服务（`z100-regress`/`z100-build-211`/`z100-triton`）。
3. **不用** stock clang 编 vLLM HIP 扩展；**不用**镜像/其它 triton；**不装** `triton-rocm`/官方 torch。
4. 所有 GPU 作业显式 `HIP_VISIBLE_DEVICES`，作业前 `pgrep -af vllm` + `hy-smi`，作业期间持锁。
5. 质量/token 一致性口径一律 `--enforce-eager`；graph/compile 只作独立实验臂。
6. 每次作业前打印并记录：git HEAD、4 个 .so md5、`triton.__file__`、`torch.__version__`。
7. 每阶段门禁不过**不得**进入下一阶段；失败先回滚再定位，不要带病前进。
