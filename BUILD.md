# z100 编译与部署（BUILD）

> 本文把「从源码编译 → 生产 `_rocm_C` relink → 编译门禁 → 换装/回滚」串成一条
> 可执行流程。三条边界：
>
> - **部署运维**（启动链、已采纳配置、禁令）→ [`DEPLOY.md`](DEPLOY.md)
> - **性能实验流程**（每轮 design/construction/result 规范）→
>   [`docs/design/z100/README.md`](docs/design/z100/README.md)
> - **通用参数说明**（DTK/torch/triton 获取、容器、各扩展职责）→
>   [`README.md`](README.md) 第 3–6 章
>
> 适用环境：`/home/lyh/z100-vllm`，DTK 26.04，4× Z100（gfx906），Python 3.10。

## 0. 先选路径

| 场景 | 路径 | 耗时 | 产物 |
|---|---|---|---|
| 改通用 C++/HIP 内核、日常迭代 | **A**：`setup.py build_ext` / `cmake --build` | 首次 1.5–3 h，之后分钟级 | 四个扩展 |
| 只换 `_rocm_C` 里的某个内核（生产热路径） | **B**：宿主 relink | 3–8 min | 一个 `_rocm_C.abi3.so` 变体 |

当前生产只走 **B** 换 `_rocm_C`；`_C` / `_C_stable_libtorch` /
`_moe_C_stable_libtorch` 三个扩展是软链到 `/mnt/game/port/build-sync/temp/`
的已验证产物，不动。

```bash
# 一页速查（详细步骤见 §3–§6）
cd /home/lyh/z100-vllm && source tools/dcu/env.sh
export OUT=/mnt/game/arms/build_x
OUT="$OUT" TB_SRC=$PWD/csrc/rocm/torch_bindings.cpp \
  bash tools/dcu/attn_stage/build.sh              # 编（3–8 min）
md5sum "$OUT/_rocm_C.abi3.so"                     # 记
ln -sfn "$OUT/_rocm_C.abi3.so" vllm/_rocm_C.abi3.so
bash start.sh restart && bash start.sh status     # 装 + 验
```

## 1. 环境（两条路径共用）

```bash
cd /home/lyh/z100-vllm
# 注意：/opt/dtk 是 26.10（libhipblas.so.3），生产/编译都用 26.04 宿主树；
# 不先定 DTK_ROOT，env.sh 会把 26.10 的头文件塞进 CPLUS_INCLUDE_PATH。
export DTK_ROOT=/mnt/game/host-runtime/dtk2604
source tools/dcu/env.sh          # DTK/HIP/编译缓存/GCN arch，别用系统 python3
.venv/bin/python -V              # 必须 3.10（DTK 轮子只有 cp310）

export PYTORCH_ROCM_ARCH=gfx906  # 必须显式；否则 gfx926/928 一起编进来
export MAX_JOBS=4                # 31 GiB RAM，6 路并行有 OOM 前例
export PYTHONPATH="$PWD:/mnt/game/triton-gfx906/install"
export VLLM_CACHE_ROOT=/mnt/game/vllm-cache-g2
```

依赖来源：`requirements/dcu-build.txt`（DTK DAS 1.8.5 的 torch 2.11 与 triton 3.6
cp310 轮子）。**gfx906 必须用自编 Triton 覆盖 `PYTHONPATH`**（见
[README §3.4](README.md#L258)），否则 `unsupported target: 'gfx906'`。

自检：

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
.venv/bin/python tools/dcu/probe_triton.py     # 期望 TRITON_OK / REDUCE_OK / INT8DOT_OK
hy-smi | head                                  # 期望 4 张 Z100L
```

构建目录统一放 `/mnt/game`（根分区只有 ~42 GB）。

## 2. 路径 A：全量 / 增量编译

### A1 全量（四个扩展）

```bash
cd /home/lyh/z100-vllm && source tools/dcu/env.sh
export PYTORCH_ROCM_ARCH=gfx906 MAX_JOBS=4
.venv/bin/python setup.py build_ext --inplace
```

> 本 fork 在 DTK 上推荐 `build_ext --inplace`；`pip install -e .` 在 setuptools
> 临时构建目录上有失败先例。编译失败先看 `dmesg | grep -i oom`，不要盲加 `-j`。

### A2 增量（已有 CMake 构建目录）

```bash
cmake --build /mnt/game/port/build-sync/temp -j 4 --target _rocm_C
```

| 改动区域 | 目标 |
|---|---|
| `csrc/rocm/fp8_emu_gemm.cu`、W8A16、sdot4 | `_rocm_C` |
| `csrc/custom_all_reduce.cuh`（push AR） | `_C_stable_libtorch` |
| `csrc/mamba/gfx906_decode_kernels.cu`（GDN） | `_C` |
| `csrc/quantization/gguf/moe*` | `_C_stable_libtorch` / `_moe_C_stable_libtorch` |

> 运行期 `torch.ops._C` 实际由 `_C_stable_libtorch`（stable ABI 路径）提供；
> 改 `csrc/libtorch_stable/**` 时要换装的是它，只换 `_C` 可能看不到行为变化。

## 3. 路径 B：生产 `_rocm_C` 宿主 relink（重点）

### 3.1 为什么要有这条路

生产 `.so` 不是用当前 CMake 构建树直接产出的，原因有两个历史事故：

1. **构建树的 `torch_bindings.cpp.o` 曾早于 S81**，relink 时静默丢掉
   `gfx906_attnd_decode_inner_kvint8` 的算子注册（S89/S91 事故）。
   → bindings 一律从 `csrc/rocm/torch_bindings.cpp` **重编**（`TB_SRC`，脚本强制）。
2. **`w8a8_sdot4.hip.o` 有同类前科**（S91：旧对象静默缺符号）。
   → 动这个文件时同样从源码重编（S99 wrapper 就是这么做的）。

所以路径 B 的做法是：**源码重编少数 TU + 复用其余已验证对象 + 用 CMake 原始
链接参数重链**，产物小步可回滚。

### 3.2 参数从哪来

编译与链接参数不手写，取自 `.s56/rocm_c_rebuild/stmt.txt`（CMake 的
`file(GENERATE)` 语句），再由脚本把路径 rebase 到本机：

| 原始路径 | rebase 后 |
|---|---|
| `/opt/dtk-26.04-DCC2602-0317`、`/opt/dtk/` | `/mnt/game/host-runtime/dtk2604` |
| `/usr/local/lib/python3.10/dist-packages/torch` | `.venv/lib/python3.10/site-packages/torch` |
| `/src/csrc/dtk_rocprim_compat.h` | `$PWD/csrc/dtk_rocprim_compat.h` |

工具链：`$DTK/llvm/bin/clang++` 编译，`$DTK/hip/bin/hipcc_cmake_linker_helper`
链接。实现见 [`tools/dcu/attn_stage/build.sh`](tools/dcu/attn_stage/build.sh)。

### 3.3 通用 relink（换 attention 内核 / 加 staging 内核）

```bash
cd /home/lyh/z100-vllm
export DTK_ROOT=/mnt/game/host-runtime/dtk2604
source tools/dcu/env.sh
export OUT=/mnt/game/arms/build_x
OUT="$OUT" \
TB_SRC=$PWD/csrc/rocm/torch_bindings.cpp \
ATTN_SRC=$PWD/.s97/arms/phsr.hip \
bash tools/dcu/attn_stage/build.sh
# 产物：$OUT/_rocm_C.abi3.so
```

| 环境变量 | 必填 | 作用 |
|---|---|---|
| `OUT` | ✅ | 输出目录（绝对路径，放在 `/mnt/game` 或 `.sNN/arms/<arm>`） |
| `TB_SRC` | ✅ | `csrc/rocm/torch_bindings.cpp`，**必须是源码**（旧对象禁用） |
| `ATTN_SRC` | – | 替换 attention inner-loop 源（如当轮变体 `.hip`） |
| `STAGE_HIP` + `STAGE_OPS` | – | 编入 staging 计时内核（两个必须成对） |

> `LD_DIR` 只写在脚本头注释里，当前实现未使用，不要依赖。

### 3.4 按轮 wrapper（当前生产配方）

当前生产 `.so`（md5 `3ccebe80`）由 `.s99/build.sh` 产出，与 §3.3 的差异：

| 项 | 通用 `build.sh` | S99 生产 |
|---|---|---|
| `torch_bindings.cpp` | `TB_SRC` 重编 | 重编 |
| `w8a8_sdot4.hip` | 用构建树 `.o` | **从源码重编** |
| `sdot4_pc.hip` | 无 | **编入**（per-channel int8 新算子） |
| attention inner | `ATTN_SRC` 或 `.s89/arms/t1int8.hip.o` | `.s97/arms/phsr.hip`（与生产同源） |
| 符号门 | 无 | `strings \| grep -c int8_pc` ≥ 1 |

```bash
cd /home/lyh/z100-vllm
OUT=.s99/arms/build_pc bash .s99/build.sh
# 期望末尾：strings int8_pc count = 28 / md5 3ccebe80...
```

`.s9x/build.sh` 属于当轮台账；若该目录被清理，按上表在
`tools/dcu/attn_stage/build.sh` 基础上追加「重编 `w8a8_sdot4.hip` + 编入
`csrc/rocm/sdot4_pc.hip`」即可复现。

### 3.5 依赖的预置对象（清理时不要删）

| 路径 | 作用 |
|---|---|
| `.s56/rocm_c_rebuild/stmt.txt`、`sysroot/` | 编译/链接参数与头文件源 |
| `/mnt/game/port/build-sync/temp/CMakeFiles/_rocm_C.dir/csrc/rocm/{attention,bf16_emu_gemv,attnd_inner}.hip.o` | 构建树既有对象 |
| `.s89/arms/fp8_emu.hip.o` | FP8 emu GEMM（S89 定版） |
| `.s89/arms/t1int8.hip.o` | attention inner 默认变体 |
| `.s97/arms/phsr.hip` | 当前生产 attention 源 |

## 4. 编译门禁（换装前必须全过）

```bash
cd /home/lyh/z100-vllm

# 1) 四个扩展可导入（用 venv，不要系统 python3）
.venv/bin/python -c "import vllm, vllm._C, vllm._rocm_C; print('IMPORT_OK')"

# 2) offload 目标只含 gfx906（别用 `grep -o 'gfx[0-9]*'`：gfx11/gfx12 是库里的字符串噪声）
strings "$OUT/_rocm_C.abi3.so" \
  | grep -oE '(hipv[0-9]+-)?amdgcn-amd-amdhsa--gfx[0-9a-z]+' | sort -u

# 3) 本轮新增符号在（按轮替换成自己的门）
strings "$OUT/_rocm_C.abi3.so" | grep -c int8_pc      # S99 门：>= 1

# 4) 扩展清单与当前指向（含 md5，便于对比）
bash tools/dcu/use_stack.sh status

# 5) 关键单测（需要 GPU）
.venv/bin/python -m pytest tests/kernels/quantization/test_gfx906_fp8_emu.py -q
```

## 5. 换装与验收

```bash
cd /home/lyh/z100-vllm
OUT=/mnt/game/arms/build_x                          # 与 §3.3 一致
md5sum "$OUT/_rocm_C.abi3.so"                       # 记进当轮台账
ln -sfn "$OUT/_rocm_C.abi3.so" vllm/_rocm_C.abi3.so
bash start.sh restart                               # 走 systemd，别手工 vllm serve
bash start.sh status
```

验收清单：

| 项 | 期望 |
|---|---|
| `start.sh restart` | health 200（带编译缓存冷启 215–260 s） |
| `status` 的 `rocm .so` / `md5` | 指向新产物，md5 与 §4 记录一致 |
| `status` 的 `policy: FP` | 与变更前一致（只换内核不动 env 时应不变） |
| `spec=` | `num_speculative_tokens: 2` |
| API 冒烟 | 中文连贯（见 [README §6.4](README.md#L593)） |
| decode 逐位 | 涉及输出路径的改动必须比对 md5/贪心流 |

## 6. 回滚

三级，越靠前越轻：

```bash
# 1) env 级：改 serve_deploy.sh 的开关行 + restart（不换 .so）
# 2) 内核级：切回上一稳定 .so + restart
ln -sfn /home/lyh/z100-vllm/.s97/arms/build_phsr/_rocm_C.abi3.so vllm/_rocm_C.abi3.so
bash start.sh restart
# 3) 全量：恢复四个扩展的既有指向
bash tools/dcu/use_stack.sh dtk
```

可用回滚点（详见 [`DEPLOY.md`](DEPLOY.md) §四）：

| 路径 | md5 | 角色 |
|---|---|---|
| `.s99/arms/build_pc/_rocm_C.abi3.so` | `3ccebe80` | 当前生产（S99 PC） |
| `.s97/arms/build_phsr/_rocm_C.abi3.so` | `adba0acb` | 回滚 1 |
| `.s93/arms/_rocm_C_dot2.abi3.so` | `7d2ed2a2` | 回滚 2 |
| `.s91/arms/_rocm_C_kvint8fix.abi3.so` | `5af8b848` | 回滚 3 |

## 7. 常见故障

| 症状 | 原因 | 处理 |
|---|---|---|
| relink 后新算子 “不存在” | 链了构建树的旧 `torch_bindings.cpp.o` / `w8a8_sdot4.hip.o` | 必须走 `TB_SRC` / 源码重编（§3.1） |
| `chunk_gated_delta_rule() got an unexpected keyword argument 'output_checkpoints'` | 改动落在 `third_party/flash_linear_attention/`，运行时用的却是 `model_executor/layers/fla/ops/chunk.py` | 改对模块或回退；补丁见 `archive/cleanup-20261115/b1-gdn-checkpoint-wip.patch` |
| `Invalid JSON: trailing characters` + `--compilation-config` | bash 里把 JSON 默认值内联进 `${X:-...}`，`}` 被当成展开结束符 | 默认值放独立变量再引用（见 `launch_mtp2.sh`） |
| decode 掉到 ~2.5 s/token | `VLLM_CACHE_ROOT` 串用旧图（S89 事件） | 换新 cache root 或清 `torch_compile_cache`，再启动 |
| 启动即报 `VLLM_DCU_GCN_ARCH is not gfx906` | KFD topology 未就绪（驱动还在探测） | 等 hymgr 完成，或显式 `VLLM_DCU_GCN_ARCH=gfx906` |
| 编译 OOM | `MAX_JOBS` 过高 | 降到 4 或更低，先 `dmesg \| grep -i oom` |
| 换了源文件但产物没变 | hipify/增量未触发 | 删构建目录重配，或直接走路径 B |
| 服务启动后 0 tok/s 假死 | `/stop_profile` 后立刻发请求 | 重启服务；不要在生产跑 profiler 后立即压测 |

## 8. 参考

- [`README.md`](README.md) §3–6：依赖获取、容器、通用编译、运行
- [`DEPLOY.md`](DEPLOY.md)：启动链、已采纳配置、禁令、回滚点
- [`tools/dcu/env.sh`](tools/dcu/env.sh)：DTK/HIP/编译缓存环境
- [`tools/dcu/attn_stage/build.sh`](tools/dcu/attn_stage/build.sh)：生产 relink 实现
- [`docs/contributing/incremental_build.md`](docs/contributing/incremental_build.md)：上游 CMake preset 方式
- [`docs/z100-rocm10-build.md`](docs/z100-rocm10-build.md)：ROCm 10.1 镜像编译（历史，运行期受阻未采用）
