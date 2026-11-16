# Z100 栈升级：DTK 26.04 → DTK 26.10（2026-11-15）

**状态：构建侧打通，运行期受阻，生产已回滚到 26.04。**

## 结论

| 阶段 | 结果 |
|---|---|
| DTK 26.10 运行时 | ✅ `/opt/dtk`（HIP 7.2.26365）可用，4 卡可枚举 |
| torch 2.13 / triton 3.7（DAS 2.0, cp312） | ✅ 能 import；gfx906 上 matmul/bf16 正常 |
| 四个 vLLM 扩展重编 | ✅ 全部编译通过，只含 gfx906，导入正常 |
| 厂商 Triton 3.7 的 gfx906 支持 | ✅ `probe_triton.py` 三项全过，不需要自编 |
| 模型加载 + torch.compile | ✅ 权重加载 7.58 GiB/rank，compile 与图捕获可完成 |
| 首个前向 / 服务可用 | ❌ `hipErrorInvalidValue` + pinned allocator 崩溃 |
| 生产回滚 | ✅ 26.04 服务已恢复（health 200，`FP 71c3af949b564865`，spec k=2） |

26.10 的**工具链本身能编译、能加载、能编译图**；卡在运行期的是
DTK 26.10 + torch 2.13 的 **HIP 运行时/分配器**，不是本仓库内核代码。

## 26.10 构建产物（已保留）

| 路径 | 内容 |
|---|---|
| `.venv-dtk2610` | Python 3.12.16 + torch 2.13.0+dtk2610 + triton 3.7.0+dtk2610 + 全量依赖 |
| `.dtk2610/built-26.10/` | 四个扩展 + `cumem_allocator` / `spinloop`（26.10 构建产物） |
| `/mnt/game/port/build-dtk2610/temp` | 完整 CMake 构建树（可增量） |
| `/mnt/game/port/build-dtk2610/triton-full` | 给 CMake 用的 triton 源码（triton_kernels 拷贝） |
| `/mnt/game/arms/dtk2610-wheels` | torch / triton 轮子缓存 |
| `/mnt/game/vllm-cache-g2-dtk2610` | 26.10 的 torch.compile 图缓存 |

## 为什么必须换 Python / torch

| 组件 | 26.04 | 26.10 |
|---|---|---|
| HIP | 6.3.26113（galaxyhip 5.2.26105） | 7.2.26365（galaxyhip 5.2.26365） |
| `libhipblas.so` | `.2` | `.3` |
| `librocblas.so` | `.4` | `.5` |
| 厂商栈 | DAS 1.8.5 | DAS 2.0 |
| torch | `2.11.0+das.opt1.dtk2604`（cp310） | `2.13.0+dtk2610`（cp312） |
| triton | `3.6.0+das185.dtk2604`（cp310） | `3.7.0+dtk2610`（cp312） |
| torchvision | `0.26.0+rocm7.2` | `0.28.0+dtk2610`（cp312） |

DAS 2.0 **只发布 cp312 轮子**，没有 cp310 版本；旧 torch 在 26.10 上连
`import torch` 都过不了（缺 `libhipblas.so.2` / `librocblas.so.4`）。

## 宿主机额外依赖（已安装，保留）

| 项 | 原因 | 落地 |
|---|---|---|
| ROCm-aware OpenMPI 4.0.3 | 26.10 的 torch 引用 `MPIX_Query_rocm_support`，DTK 不带该 MPI | 从 26.04 torch wheel 的 `torch.libs/` 取出，`patchelf --set-soname libmpi.so.40` 后放 `/usr/local/lib` + `ldconfig` |
| `libgflags-dev` / `libgoogle-glog-dev` | Caffe2 CMake 配置硬性要求 | apt |
| `librt.so` 符号链接 | 链接 `_C_stable_libtorch` 时 ninja 报缺失 | `ln -s librt.so.1 /usr/lib/x86_64-linux-gnu/librt.so` |
| cmake 3.31 + ninja | vLLM 0.18 要求 `cmake<4`，宿主 3.22 太旧 | 装进 `.venv-dtk2610`，构建时 `PATH` 前置 |

## 代码适配（已在 tree 内，26.04 下不生效）

1. `cmake/hipify.py`：torch 2.13 的 hipify 把 `__CUDA_ARCH__` 改名为
   `__FPT_DTK_ARCH__`，生成代码还测 `FPT_DTK_VERSION`；shim 现在同时定义
   `__DTK_ARCH__` / `__FPT_DTK_ARCH__`、`DTK_VERSION` / `FPT_DTK_VERSION`。
2. `tools/dcu/env.sh`：补 `LIBRARY_PATH` 指向带 `libstdc++.so` 的 gcc 目录，
   否则 CMake 的 HIP 编译器探测报 `ld.lld: error: unable to find library -lstdc++`。
3. `vllm/platforms/rocm.py`：新增 `VLLM_DCU_NO_PIN_MEMORY`（默认关）。
   置 1 时 `is_pin_memory_available()` 返回 False，vLLM 走 `NonUvaBuffer`。

## 运行期阻塞（按出现顺序）

### 1. 权重常驻后 `hipHostMalloc` 全面失败

- 每 rank 加载 7.58 GiB 后，`pin_memory=True` 一律报
  `hipErrorOutOfMemory`；设备仍有 22.8 GiB 空闲，宿主 `MemAvailable` 12.6 GiB。
- 1 / 4 / 8 / 12 MiB 全失败；直接 ctypes 调 `hipHostMalloc(1 MiB)` 也返回 2。
- 与 `VLLM_DCU_CUSTOM_AR`、`HSA_FORCE_FINE_GRAIN_PCIE` 无关（分别关闭仍复现）。
- 独立进程「7.58 GiB 设备分配 + 8 GiB pinned」不复现，只在 vLLM worker 的
  加载序列后出现。
- 影响 `vllm/v1/worker/gpu/buffer_utils.py` 的 `UvaBuffer`。
- **绕行**：`VLLM_DCU_NO_PIN_MEMORY=1` → `NonUvaBuffer`，可以继续到下一个阶段。

### 2. 首个 fused all-reduce+rmsnorm 前向 `hipErrorInvalidValue`

- 位置：`vllm/distributed/parallel_state.py:256`，
  `s = x.to(torch.float32) + residual.to(torch.float32)`。
- 张量元数据完全正常（fp16、contiguous、`cuda:2`、8192×5120）。
- 已排除：custom AR（关掉仍失败）、pynccl（standalone 测试 4 rank
  all-reduce 正常）、stream/当前设备（两者都正确）、内存不足（22.8 GiB 空闲）。
- 失败后在同一进程里再执行同样的 cast/clone/add 又能成功（sticky error 被清）。
- 在同一表达式上还会触发
  `!handles_.at(i) INTERNAL ASSERT FAILED at c10/hip/HIPCachingAllocator.cpp:469`
  ——torch 的 HIP caching allocator 内部记账被破坏。
- `HIP_LAUNCH_BLOCKING=1` / `AMD_LOG_LEVEL=2` 都没能把错误钉到更早的内核，
  说明错误由分配器/运行时状态触发，而不是某个自定义 kernel 的 launch 参数。

### 3. 禁用 pynccl 后 worker 静默死亡

- `VLLM_DISABLE_PYNCCL=1` + `VLLM_DCU_CUSTOM_AR=0`：worker 过完模型加载与
  page-size 设置后无 Python traceback 直接退出（`exit code: None`），
  dmesg 无 OOM 记录。

## 复现 26.10 栈

```bash
cd /home/lyh/z100-vllm
export DTK_ROOT=/opt/dtk
export PATH=$PWD/.venv-dtk2610/bin:$PATH
source tools/dcu/env.sh
export VLLM_CACHE_ROOT=/mnt/game/vllm-cache-g2-dtk2610
export VLLM_DCU_NO_PIN_MEMORY=1
ln -sfn $PWD/.dtk2610/built-26.10/_C.abi3.so vllm/_C.abi3.so
# ... 其余三个扩展同理
.venv-dtk2610/bin/python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' \
  serve /mnt/game/models/Qwen3.8-27B-FP8 --tensor-parallel-size 4 --enforce-eager ...
```

构建：

```bash
export PATH=$PWD/.venv/bin:$PATH DTK_ROOT=/opt/dtk
source tools/dcu/env.sh
export PYTORCH_ROCM_ARCH=gfx906 MAX_JOBS=4
export FETCHCONTENT_BASE_DIR=$PWD/.deps
export TRITON_KERNELS_SRC_DIR=/mnt/game/port/build-dtk2610/triton-full/python/triton_kernels/triton_kernels
.venv/bin/python setup.py build_ext --inplace \
    --build-temp /mnt/game/port/build-dtk2610/temp -j 4
```

（`triton_kernels` 用 codeload tarball 本地解包，因为本机 git clone
`github.com/ROCm/triton.git` 被限速。）

## 回滚记录（已执行）

```bash
rm .venv && mv .venv-dtk2604 .venv        # 26.04 venv
git checkout -- requirements/dcu*.txt requirements/build/dcu.txt
ln -sfn /mnt/game/port/build-sync/temp/_C.abi3.so vllm/_C.abi3.so
ln -sfn /mnt/game/port/build-sync/temp/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
ln -sfn /mnt/game/port/build-sync/temp/_moe_C_stable_libtorch.abi3.so vllm/_moe_C_stable_libtorch.abi3.so
ln -sfn .s99/arms/build_pc/_rocm_C.abi3.so vllm/_rocm_C.abi3.so
# systemd drop-in g2-ms16.conf 的 VLLM_CACHE_ROOT 改回 /mnt/game/vllm-cache-g2
bash start.sh restart
```

26.04 的 `.venv-dtk2604` 已改回 `.venv`；`launch_mtp2.sh` /
`serve_deploy.sh` / `requirements/dcu*.txt` / README / BUILD 都已恢复 26.04 口径。

## 后续定位（2026-11-16）

### 根因方向：厂商 Triton 3.7 没有 gfx906 的 SFU hazard 修复

26.04 生产栈用的是**自编/打过补丁**的 Triton 3.6
（`tools/dcu/triton-gfx906-3.6.0.patch`），补丁里有一条硬件 workaround：

> gfx906 的 SFU 指令（`v_exp*` / `v_log*` / `v_sqrt*` / `v_rcp*` / `v_rsq*` …）
> 如果消费的是紧邻前一条 VALU 写出的 VGPR，SIMD16 的 lane 8–11 会读到旧值。
> 补丁在每条 SFU 前插一个 `s_nop 2`。

DAS 2.0 的 `triton-3.7.0+dtk2610` 虽然能报 `gfx906` 目标、add/reduce/int8-dot
探针也过，但**没有这条 workaround**。证据：

1. `VLLM_GFX906_GDN_TRITON_PREFILL=0` + `VLLM_QWEN35_MTP_GDN_TRITON=0` 时，
   TP=4 eager 能起服务（health 200，KV 18.59 GiB），但输出是乱码
   （`1+1` 返回 `oki_quit_ACL...`）——说明还有别的 Triton kernel 在静默算错。
2. 之前 TP=4 在 fused all-reduce+rmsnorm 处报 `hipErrorInvalidValue`，
   同一条路径在 TP=1（world_size=1，跳过 AR）能走完——错误是前一个
   Triton kernel 把 HIP 错误队列污染后，在下一个算子处冒出来。

### 已落地的补丁

`tools/dcu/patch_triton_gfx906_sfu.py`：把 3.6 patch 里的 SFU `s_nop` 后处理
直接注入已安装的 Triton 3.7 的 `backends/amd/compiler.py`
（`make_amdgcn` 返回前对 gfx906 的 asm 插入 `s_nop 2`）。幂等、纯 Python、
不需要重编 Triton。

```bash
.venv-dtk2610/bin/python tools/dcu/patch_triton_gfx906_sfu.py
# 换新的 TRITON_CACHE_DIR，旧 cache 里的 kernel 不会重新生成
export TRITON_CACHE_DIR=/mnt/game/triton-cache-dtk2610-sfu
```

打补丁后（`TRITON_CACHE_DIR=/mnt/game/triton-cache-dtk2610-sfu`，冷 cache）：

- TP=4 + 默认 GDN Triton + eager：**启动成功**，`Available KV cache
  memory: 18.22 GiB`，health 200，此前 `hipErrorInvalidValue` 不再出现
  （engine init 607 s，冷 Triton 编译；热 cache 会快很多）；
- **但输出仍是乱码**（`1+1` 返回 `arзно...`），说明还有第二个数值问题，
  与 SFU hazard 无关；候选：
  1. 用 26.10 工具链重编的自定义 HIP kernel（FP8 emu / W8A16 / sdot4 / GDN），
  2. Triton 3.7 在 gfx906 上的其他 codegen 差异，
  3. int8 KV / attention 路径。
- 下一步：在 26.10 栈上跑仓库自带的内核单测（如
  `tests/kernels/quantization/test_gfx906_fp8_emu.py`）逐项定位，而不是继续
  用整模型试错。

### 内核单测矩阵（2026-11-16）

在 26.10 栈上直接跑仓库内核单测（py3.12 + 26.10 扩展 + 已打 SFU 补丁的 triton）：

| 测试 | 26.10 结果 | 26.04 基线 | 说明 |
|---|---|---|---|
| `tests/kernels/quantization/test_gfx906_fp8_emu.py` | **239 passed** | — | FP8 emu GEMM 数值正确 |
| `tests/kernels/attention/test_triton_unified_attention_gfx906.py` | **39 passed** | — | unified attention 正常 |
| `tests/kernels/mamba/test_fused_recurrent_packed_decode.py` + `test_causal_conv1d.py` | 74 failed / 158 passed | **同样 74 failed / 158 passed** | 既有失败，不是 26.10 回归，不能当判据 |
| `tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py` | 13 failed / 28 passed | 未跑 | 27B 模型不用 GGUF |

其它判据：

- 27B 在 26.10 上换成默认 KV dtype（fp16）仍是乱码 → **不是 int8 KV 的锅**；
- Qwen3-0.6B 在 **26.04 上同样乱码** → 小模型不能当对照组（本 fork 对 Qwen3 dense 的支持本身就不完整）；
- 因此乱码来自 FP8 emu / unified attention 之外的路径，下一步应做
  **逐层 hidden-state/logits 对比**（同一 prompt、同一权重，26.04 vs 26.10），
  定位第一个发散层（优先查 GDN/linear-attn、fused rmsnorm/quant、embedding）。

### 生产配置下的二分（2026-11-16 晚）

先修正一个判据：用 ad-hoc 参数（`--enforce-eager --max-model-len 4096`、
无 MTP、`VLLM_DCU_CUSTOM_AR=0`）跑 27B 时，**26.04 也输出乱码**，所以
ad-hoc 乱码不能归因于 26.10；有效对比必须用 `serve_deploy.sh` 的生产参数。

26.10 上按生产参数（graph + MTP k=2 + prefix cache + S99 旋钮）逐项关闭：

| 变体 | 结果 |
|---|---|
| 生产默认（custom AR=1） | `hipErrorInvalidValue`，位置仍是 `_fused_add_rmsnorm_gemma` |
| `VLLM_DCU_CUSTOM_AR=0` | 同样 `hipErrorInvalidValue` |
| `EAGER=1`（关图） | 同样 `hipErrorInvalidValue` |
| `SPEC_OFF=1`（关 MTP） | worker **SIGSEGV**（exit -11），死在初始化/首次前向 |

结论：不是 custom AR、不是 cudagraph、也不是单独 MTP 的问题；26.10 的
HIP 运行时在这个 fork 的前向路径上会以 `hipErrorInvalidValue` / SIGSEGV
的方式崩，且 ad-hoc 能启动的配置输出也不可信。**没有找到任何能稳定、正确
服务的 26.10 配置。**

## 建议的下一步

1. 拿这三个运行期现象找海光确认：26.10 的 `hipHostMalloc` 回归、
   torch 2.13 wheel 的 `HIPCachingAllocator` 兼容性、RCCL/pynccl 行为变化。
2. 试 DAS 2.0 的后续 torch 构建（`torch-2.13.0+dtk2610.*` 新时间戳）或 26.10
   的补丁版 HIP。
3. 若短期无法解决，维持 26.04 生产栈；26.10 的构建树与 venv 保留，等新轮子
   后只需替换 `.venv-dtk2610` 里的 torch/triton 并重跑上面的构建命令。
