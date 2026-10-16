# S7 内存 / KV cache / graph 策略 — 执行结果

> 归档时间：2026-10-08
> 执行者：子 agent（S7 内存/KV cache/graph）
> 对应设计文档：[docs/design/z100-07-memory-kvcache-graph.md](../design/z100-07-memory-kvcache-graph.md)
> 前一档：[z100-06-triton-gfx906-result.md](z100-06-triton-gfx906-result.md)（S6，自编 gfx906 triton + SFU hazard 修复）
> 状态：✅ **门禁通过**（KV cache 自动算出、graph 捕获成功、serve 冒烟通过）；
> ⚠️ 追加发现：**`enforce_eager` 路径在 gfx906 上数值错误（全 NaN logits）**，
> 关闭编译走 eager 会输出垃圾，必须开 `torch.compile`（`mode=VLLM_COMPILE`）——与本步任务书的保守默认相反，见 §5.1。

**一句话结论**：单卡（Z100L，卡 0）用 Qwen3-0.6B + fp16 起 V1 engine，**无需 `--kv-cache-memory-bytes`、无需 gfx906 的 0.98 兜底**，
默认 `--gpu-memory-utilization 0.9` 下 `Available KV cache memory = 27.18 GiB`（eager 路径）与 `26.75 GiB`（graph 路径），
**CUDA/HIP graph 在 gfx906 上可以完整捕获**（PIECEWISE 51/51 + FULL 35/35，耗时 8–10 s、占用 0.36 GiB），
decode 从 **34.6 ms/token（eager）降到 11.6 ms/token（graph）≈ 3.0×**；
但 eager 路径生成的 16 个 token 全是 `1023`（`"lock"` 重复）、logprobs 全为 `nan`，
而 graph / `torch.compile` 路径给出与 HF transformers 参考逐 token 一致的正确输出（见 §3.7 的 A/B 矩阵）。
`vllm/utils/mem_utils.py` 与 `vllm/v1/worker/gpu_worker.py` **未做任何改动**（门禁未触发钳制条件）。

---

## 1. 目标与验收标准

### 1.1 设计文档门禁原文（`docs/design/z100-07-memory-kvcache-graph.md` §验收）

1. 单卡默认参数下 `Available KV cache memory` > 0，**且未使用** `--kv-cache-memory-bytes`；
2. 显式 `--kv-cache-memory-bytes` 仍能覆盖自动逻辑；
3. graph/eager 有明确结论与默认值，并记录依据；
4. 4 卡各跑一次 decode，显存占用与 `hy-smi` 观测偏差 < 5%。

### 1.2 本次任务书追加的验收标准

- 新建容器 `z100-serve`（同镜像；设备/group 参数照抄 `z100-build-211`；`--memory=8g --shm-size=16g --ipc=host`；不加 `--rm`），
  挂 `/mnt/game` 与只读工作区 `/src`；
- **单卡**跑 V1 engine（`HIP_VISIBLE_DEVICES=0`），模型取容器内可见的最小模型；
- 环境必须带 **S6 自编 triton**（`PYTHONPATH=/mnt/game/triton-gfx906/install` + `TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache`），否则 gfx906 出不了码且会踩 SFU hazard；
- 先 `--enforce-eager` 跑通（`Available KV cache memory` + 一次 decode 输出），再去掉 `--enforce-eager` 测 graph 捕获，给出 graph vs eager 结论
  （能否捕获 / 是否成功 / decode 速度对比 / 数值是否一致）；
- serve 冒烟：单卡、短 `--max-model-len`、端口 ≥ 8001（**不用 8000**），`curl /health` + 一次 chat/completions 成功，收工后停服；
- 若 KV cache 为 0 或峰值判定异常，才按任务书在 `vllm/utils/mem_utils.py` / `vllm/v1/worker/gpu_worker.py` 做最小修复；
- 硬约束：禁止运行自编译 HIP 独立可执行程序；不 start/kill/exec/commit `qwen38-vllm`；不动 `z100-build`；不替换 DTK 库；产物只落 `/mnt/game`；
  不改 `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`；写作用域限于 `vllm/utils/mem_utils.py`、`vllm/v1/worker/gpu_worker.py`、
  `tools/dcu/**`、本归档。

---

## 2. 实际改动

### 2.1 仓库源码改动

**`vllm/**` 零改动**（门禁 1 直接通过，未触发 `MemorySnapshot` 钳制与 `gpu_worker` 兜底之外的必要修改；
`Available KV cache memory > 0` 时设计文档第 2 步的 `mem_utils.py` 补丁**不适用**）。

新增/修改文件（全部在 `tools/dcu/**`，任务书允许的写作用域内）：

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `tools/dcu/s7_env.sh` | 新文件 | S7 运行环境：在 `tools/dcu/env.sh` 之上前置自编 triton + fork 副本，导出 `TRITON_CACHE_DIR=/mnt/game/...`，并把 `HIP_VISIBLE_DEVICES` 固定为卡 0 |
| `tools/dcu/s7_usercustomize.py` | 新文件 | 运行期 shim（见 §2.3）：① 任何 python 启动即先 `import torch`（todolist 硬约束 8，含 registry 子进程）；② `torchvision` 的 meta/fake 注册失败时打桩，避免 import 崩 |
| `tools/dcu/s7_gate.sh` | 新文件 | 门禁 1/3 的主脚本：`LLM(enforce_eager=…)` + 固定 prompt 贪心生成 + 128 token decode 计时 |
| `tools/dcu/s7_compare.sh` | 新文件 | 追加 kwargs 的 A/B 变体脚本（隔离实验用） |
| `tools/dcu/s7_compile_matrix.sh` | 新文件 | 隔离“编译 vs CUDA graph”的矩阵脚本（`cudagraph-none` / `compile-none` / `full`） |
| `tools/dcu/s7_prompts.sh` | 新文件 | 多 prompt 贪心生成（区分“系统性数值错”与“个例 prompt”） |

> 容器内副本被同步到 `/mnt/game/port/gfx906-vllm-211/tools/dcu/`（fork 副本，非仓库）。

### 2.2 运行期依赖补装（**只发生在 `z100-serve` 容器内，未动 `z100-build*` / `z100-triton`**）

镜像自带依赖对这套 fork 太旧，起 V1 engine 前必须补齐（都是 import 期硬失败，不是本步引入）：

| 包 | 原版本 | 补装版本 | 触发失败点 |
|---|---|---|---|
| `torch` | 2.10.0+das.opt1.dtk2604 | **2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7**（复用 `/mnt/game/port/wheels-211/`） | fork 的四个 `.so` 是 2.11 编的，2.10 下报 `undefined symbol: _ZN3c104cuda20CUDACachingAllocator9allocatorE` / `torch_from_blob` |
| `mistral_common` | 1.10.0 | **1.11.5** | `ModuleNotFoundError: No module named 'mistral_common.guidance'` |
| `openai` | 2.21.0 | **2.54.0** | `ImportError: cannot import name 'NamespaceTool' from 'openai.types.responses'` |
| `xgrammar` | 0.1.33 | **0.2.8** | `ModuleNotFoundError: No module named 'xgrammar.openai_tool_call_schema'` |
| `apache-tvm-ffi` | 未装 | **0.1.14.post1** | xgrammar 0.2.8 的 `xgrammar.__init__` → `tvm_ffi` |
| `torchvision` | 0.25.0+das（**为 torch 2.10 构建**） | **0.26.0（PyPI CPU 版）+ shim** | `RuntimeError: operator torchvision::nms does not exist`（详见 §3.1） |

轮子全部缓存到 `/mnt/game/port/s7/pip-wheels/`（根分区未受影响）。

### 2.3 环境 shim 与工作区

| 资源 | 路径 | 说明 |
|---|---|---|
| 运行容器 | `z100-serve` | 见 §2.4；**未加 `--rm`，收工后保留**（`Up`，空闲 ~2.8 GiB 宿主内存，无 GPU 占用） |
| shim 文件 | 容器内 `/root/.local/lib/python3.10/site-packages/usercustomize.py` | 由 `tools/dcu/s7_usercustomize.py` 拷贝而来。**注意**：`/usr/lib/python3.10/sitecustomize.py` 是镜像自带的符号链接，`sitecustomize.py` 这个文件名会被镜像抢先占用，所以必须改名成 `usercustomize.py`（python 的 user site 默认启用，且无同名文件冲突） |
| 计算缓存 | `/mnt/game/triton-gfx906/cache` | S6 复用；未写根分区 |
| 实验日志 | `/mnt/game/port/s7/*.log` | 本档所有原始输出的落盘位置 |

`tools/dcu/s7_usercustomize.py` 的两条工作都是**运行期兜底、不改变数值语义**：
1. `import torch` 前置——注册表子进程（`python -m vllm.model_executor.models.registry`）会先 `import vllm`，
   在 DTK 上直接 `std::bad_alloc` → SIGABRT(134)，导致 `Model architectures ['Qwen3ForCausalLM'] failed to be inspected`；
2. torchvision 打桩——DTK 只发布了 `torch2100` 的 torchvision，配 torch 2.11 时 `torchvision::nms` 没注册上，
   `torch/library.py::_register_fake` 抛 `RuntimeError`；vLLM 的 warmup 链
   （`vllm/model_executor/warmup/kernel_warmup.py` → `minimax_m3_msa_warmup` → `vllm/models/minimax_m3/amd/model.py`
   → `vllm/transformers_utils/processors/minimax_m3.py:19`）在**模块顶层**就 import `torchvision.transforms.InterpolationMode`。
   shim 在确认 `torchvision::nms` 缺失后，把 `torchvision._meta_registrations` 预置为一个空 stub，
   使 `import torchvision` 只做纯 python 部分（transforms 的枚举/函数不依赖 C++ 算子）。

### 2.4 容器实际配置（`docker inspect z100-serve` 原文）

```console
$ docker run -d --name z100-serve \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    --group-add video --security-opt label=disable --ipc host \
    --memory=8g --memory-swap=8g --shm-size=16g \
    -v /opt/hyhal:/opt/hyhal:ro \
    -v /home/lyh/z100-vllm:/src:ro \
    -v /mnt/game:/mnt/game \
    -w /mnt/game/port \
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 \
    bash -c 'sleep infinity'
b4e70c99188487f59131ee869c558c5dfb8b01373198de400ecebaad82e482c7

$ docker inspect z100-serve --format '...'
ImageID=sha256:d1c08bf30382f6d8f49e0ad7130ceb4e3a31484ce4bdd3b1aeb7787b5d2fb14e   # 与 z100-build-211 同镜像
Devices=[{"PathOnHost":"/dev/kfd",...},{"PathOnHost":"/dev/mkfd",...},{"PathOnHost":"/dev/dri","CgroupPermissions":"rwm"}]
GroupAdd=["video"]  SecurityOpt=["label=disable"]  IpcMode=host  ShmSize=17179869184
Memory=8589934592  MemorySwap=8589934592  AutoRemove=false  WorkingDir=/mnt/game/port
Mounts=/home/lyh/z100-vllm:/src:ro /mnt/game:/mnt/game:rw /opt/hyhal:/opt/hyhal:ro
```

> `--ipc=host` 与 `--shm-size=16g` 并存时实际 `ShmSize` 记录为 16 GiB；宿主 `/dev/shm` 本身即 16 GiB（`tmpfs 16G`），
> 容器内 `df -h /dev/shm` 也是 `16G`，与 vLLM 的 `VLLM_WORKER_MULTIPROC_METHOD=spawn` 共享内存需求相容。

---

## 3. 执行过程与原始证据

### 3.0 环境自检（卡 0、目标栈、四个扩展）

```console
$ docker exec z100-serve bash -lc '
    source /opt/dtk/env.sh >/dev/null 2>&1
    source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh
    cd $VLLM_DIR
    python -c "..."'
[dcu] DTK_ROOT=/opt/dtk (26.04)
[dcu] rocm_smi=/opt/dtk/.hyhal/rocm_smi/lib VLLM_DCU_GCN_ARCH=gfx906
[dcu-s7] VLLM_DIR=/mnt/game/port/gfx906-vllm-211
[dcu-s7] TRITON_INSTALL=/mnt/game/triton-gfx906/install TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
[dcu-s7] HIP_VISIBLE_DEVICES=0
torch dist  = 2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7
torch hip   = 6.3.26113
devcount    = 4              # 见 NOTE
gcnArch     = gfx906:sramecc+:xnack-
triton dist = 3.6.0+git7c56a5e4 /mnt/game/triton-gfx906/install/triton/__init__.py
IMPORT_OK_ALL4
on_gfx906   = True
SKINNY      = False
```

> **NOTE（贴出的这段是修 `s7_env.sh` 之前的输出）**：第一版 `s7_env.sh` 用
> `HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"`，而 `tools/dcu/env.sh` 已经导出了 `0,1,2,3`，
> 所以 `:-` 不生效、`devcount=4`。已修正为 `S7_HIP_VISIBLE_DEVICES:-0` 无条件覆盖，修正后复验原文：

```console
$ ... source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh; python -c "import torch; ..."
[dcu-s7] HIP_VISIBLE_DEVICES=0
devcount= 1
name0= Z100L
free_GiB=31.73 total_GiB=31.98
```

**本步全部实验都只用了卡 0**；实验结束后 4 张卡显存均回到 31.73 GiB 空闲（§3.10）。

### 3.1 起步阻塞链（都是环境依赖问题，逐条解决）

按顺序撞到并解决（每条原文首行 + 处理）：

```console
# ① torch 2.10 与 2.11 编的 .so 不匹配
WARNING [rocm.py:64] Failed to import from vllm._C with ImportError(
  '/mnt/game/port/gfx906-vllm-211/vllm/_C.abi3.so: undefined symbol: _ZN3c104cuda20CUDACachingAllocator9allocatorE')
WARNING [rocm.py:70] Failed to import from vllm._C_stable_libtorch with ImportError(... undefined symbol: torch_from_blob)
-> pip install --no-deps /mnt/game/port/wheels-211/torch-2.11.0+das.opt1.dtk2604.*.whl

# ② 镜像自带 torchvision 是 torch2100 构建，配 torch 2.11 注册不上算子
RuntimeError: operator torchvision::nms does not exist
  File ".../torch/_library/fake_impl.py", line 50, in register
-> 对照实验：2.10 栈（z100-build）同一 torchvision 完全正常：
   $ docker exec z100-build ... python -c "import torch, torchvision; from torchvision.ops import nms; print('TORCHVISION_OPS_OK')"
   torch 2.10.0+das.opt1.dtk2604.20260325.g6b060a / torchvision 0.25.0+das.opt1.dtk2604.torch2100.20260325.g860505
   TORCHVISION_OPS_OK
   DTK 下载树只有 torch，没有 torchvision：
   $ curl -s https://download.sourcefind.cn:65024/directlink/4/pytorch/DAS1.8.5/
   <a href="torch-2.10.0+das.opt1.dtk2604.2608131652.g28498c-...whl">...</a>
   <a href="torch-2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7-...whl">...</a>
-> pip install --no-deps torchvision==0.26.0 (PyPI cpu wheel) + usercustomize meta-registration 打桩
   （依据：requirements/dcu.txt:15-17 明确写了 "torchvision/torchaudio are dropped as well: the DTK download tree
     publishes no wheels for them"；且 vLLM 只在 processor/warmup 路径用它）

# ③ registry 子进程在 DTK 上 std::bad_alloc（todolist 硬约束 8）
ERROR [registry.py:1109] RuntimeError: Error raised in subprocess:
ERROR [registry.py:1109] terminate called after throwing an instance of 'std::bad_alloc'
pydantic_core._pydantic_core.ValidationError: ... Model architectures ['Qwen3ForCausalLM'] failed to be inspected
-> usercustomize.py 里先 import torch（sitecustomize.py 被镜像占用，见 §2.3）

# ④⑤⑥ import 链上的三个过旧依赖
ModuleNotFoundError: No module named 'mistral_common.guidance'          -> mistral_common 1.11.5
ImportError: cannot import name 'NamespaceTool' from 'openai.types.responses' -> openai 2.54.0
ModuleNotFoundError: No module named 'xgrammar.openai_tool_call_schema' -> xgrammar 0.2.8 (+ apache-tvm-ffi)
```

### 3.2 门禁 1+3：eager 跑通（`/mnt/game/port/s7/s7-eager.log`，长输出截断）

命令（可直接粘贴）：

```console
$ docker exec z100-serve bash -lc '
    source /opt/dtk/env.sh >/dev/null 2>&1
    source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh
    cd $VLLM_DIR
    timeout 1200 bash tools/dcu/s7_gate.sh eager /mnt/game/models/Qwen3-0.6B /mnt/game/port/s7/s7-eager.log'
```

关键原始行（日志按序截取）：

```console
INFO 10-08 20:30:31 [api_utils.py:286] non-default args: {'dtype': 'float16', 'max_model_len': 2048,
    'gpu_memory_utilization': 0.9, 'enforce_eager': True, 'model': '/mnt/game/models/Qwen3-0.6B'}
WARNING 10-08 20:30:31 [vllm.py:1795] Enforce eager set, disabling torch.compile, CUDAGraphs, and JIT kernel warmup.
    This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none and --kernel_config.enable_jit_warmup=False
INFO 10-08 20:30:34 [core.py:129] Initializing a V1 LLM engine (vdev) with config: model='/mnt/game/models/Qwen3-0.6B',
    ... tensor_parallel_size=1, ... disable_custom_all_reduce=True, enforce_eager=True, ...
    'gpu_memory_utilization' 未升高、未出现 gfx906 0.98 兜底
INFO 10-08 20:30:49 [common.py:132] [gfx906] Falling back to PyTorch implementation for rotary embeddings.
INFO 10-08 20:30:49 [rocm.py:895] Found incompatible backend(s) [TURBOQUANT] with AttentionType.DECODER.
    Overriding with TRITON_ATTN out of potential backends: ['TRITON_ATTN', 'ROCM_ATTN'].
INFO 10-08 20:30:52 [model_runner.py:406] Model loading took 1.12 GiB memory and 2.746747 seconds
INFO 10-08 20:30:52 [utils.py:320] Using LBNHC KV cache layout.
INFO 10-08 20:30:53 [gpu_worker.py:719] Available KV cache memory: 27.18 GiB                     <-- 门禁 1
INFO 10-08 20:30:53 [kv_cache_utils.py:2432] GPU KV cache size: 254,464 tokens,
    Maximum concurrency for 2,048 tokens per request: 124.25x
INFO 10-08 20:30:53 [gpu_worker.py:977] Free memory on device (31.58/31.98 GiB) on startup. Desired GPU memory
    utilization is (0.9, 28.79 GiB). Actual usage is 1.4 GiB for consumed memory (weights + non-torch),
    0.2 GiB for peak activation, and 0.0 GiB for CUDAGraph memory.
    Replace gpu_memory_utilization config with `--kv-cache-memory=29027187405` (27.03 GiB) to fit into requested
    memory, or `--kv-cache-memory=32032284160` (29.83 GiB) to fully utilize gpu memory.
    Current kv cache memory in use is 27.18 GiB.
INFO 10-08 20:30:54 [core.py:388] init engine (profile, create kv cache, warmup model) took 2.12 s
[s7] engine ready in 23.2s mode=eager
WARNING 10-08 20:30:57 [jit_monitor.py:140] Triton kernel JIT compilation during inference: reshape_and_cache_kernel_flash. ...
WARNING 10-08 20:30:58 [jit_monitor.py:140] Triton kernel JIT compilation during inference: kernel_unified_attention_2d. ...
WARNING 10-08 20:30:59 [jit_monitor.py:140] Triton kernel JIT compilation during inference: kernel_unified_attention_3d. ...
[s7] S7_IDS_EAGER=[1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023]
[s7] S7_TEXT_EAGER='locklocklocklocklocklocklocklocklocklocklocklocklocklocklocklock'
[s7] S7_DECODE_EAGER TOKENS=128 SECONDS=4.4333 MS_PER_TOKEN=34.636
S7_GATE_RC=0
```

- **没有任何** `automatically raising it to 0.98`、`No available memory for the cache blocks`、
  `Free memory grew by ...`（`maybe_rocm_profiling_fallback` 未触发）行 ⇒ gfx906 兜底**不需要**，`gpu_memory_utilization` 保持 0.9。
- **但 eager 的数值是错的**（16 个 token 全是 id 1023），详见 §3.7。

### 3.3 门禁 3：graph（`/mnt/game/port/s7/s7-graph.log`，长输出截断）

命令：同 §3.2，参数改 `bash tools/dcu/s7_gate.sh graph …`（`enforce_eager=False`，其余不变）。

关键原始行（`Capturing CUDA graphs` 进度条中间态略去，首尾保留）：

```console
INFO 10-08 20:32:44 [core.py:129] Initializing a V1 LLM engine ... enforce_eager=False ...
    compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, ..., 'backend': 'inductor',
    'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1,
    'cudagraph_capture_sizes': [1, 2, 4, 8, 16, ..., 496, 512], 'max_cudagraph_capture_size': 512, ...
    kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['native'], ...), enable_jit_warmup=True, ...)}
INFO 10-08 20:33:25 [backends.py:376] Cache the graph of compile range (1, 8192) for later use
INFO 10-08 20:33:35 [backends.py:391] Compiling a graph for compile range (1, 8192) takes 24.58 s
INFO 10-08 20:33:39 [monitor.py:53] torch.compile took 36.99 s in total
INFO 10-08 20:33:41 [monitor.py:81] Initial profiling/warmup run took 1.75 s
Capturing CUDA graphs (PIECEWISE):   0%|          | 0/51 ... 100%|██████████| 51/51 [00:06<00:00,  7.36it/s]
Capturing CUDA graphs (FULL):        0%|          | 0/2  ... 100%|██████████| 2/2  [00:00<00:00, 10.50it/s]
INFO 10-08 20:33:50 [model_runner.py:1060] Graph capturing finished in 8 secs, took 0.26 GiB
INFO 10-08 20:33:51 [gpu_worker.py:719] Available KV cache memory: 26.75 GiB                     <-- 门禁 1（graph 路径）
INFO 10-08 20:33:51 [gpu_worker.py:734] CUDA graph memory profiling is enabled (default since v0.21.0). The current
    --gpu-memory-utilization=0.9000 is equivalent to --gpu-memory-utilization=0.8901 without CUDA graph memory
    profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9099.
INFO 10-08 20:33:51 [kv_cache_utils.py:2432] GPU KV cache size: 250,384 tokens,
    Maximum concurrency for 2,048 tokens per request: 122.26x
INFO 10-08 20:33:51 [kernel_warmup.py:172] JIT kernel warmup starting.
JIT kernel warmup (140 compile keys): 100%|██████████| 4/4 [01:20<00:00, 20.17s/kernel, ...]
INFO 10-08 20:35:12 [kernel_warmup.py:185] JIT kernel warmup finished in 80.82s.
Capturing CUDA graphs (PIECEWISE):   0%|          | 0/51 ... 100%|██████████| 51/51 [00:05<00:00,  9.74it/s]
Capturing CUDA graphs (FULL):        0%|          | 0/35 ... 100%|██████████| 35/35 [00:03<00:00,  9.45it/s]
INFO 10-08 20:35:27 [model_runner.py:1060] Graph capturing finished in 10 secs, took 0.36 GiB
INFO 10-08 20:35:27 [gpu_worker.py:914] CUDA graph pool memory: 0.36 GiB (actual), 0.32 GiB (estimated),
    difference: 0.04 GiB (11.1%).
INFO 10-08 20:35:27 [gpu_worker.py:977] Free memory on device (31.58/31.98 GiB) on startup. Desired GPU memory
    utilization is (0.9, 28.79 GiB). Actual usage is 1.4 GiB ... 0.64 GiB for peak activation, and
    0.36 GiB for CUDAGraph memory. ... Current kv cache memory in use is 26.75 GiB.
INFO 10-08 20:35:28 [core.py:381] init engine (profile, create kv cache, warmup model) took 145.83 s
    (compilation: 36.99 s)
[s7] engine ready in 167.2s mode=graph
WARNING 10-08 20:35:28 [jit_monitor.py:140] Triton kernel JIT compilation during inference: kernel_unified_attention_2d. ...
[s7] S7_IDS_GRAPH=[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13]
[s7] S7_TEXT_GRAPH=' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'
[s7] S7_DECODE_GRAPH TOKENS=128 SECONDS=1.4855 MS_PER_TOKEN=11.605
S7_GATE_RC=0
```

### 3.4 graph vs eager 汇总

| 项 | eager（`--enforce-eager`） | graph（默认，`FULL_AND_PIECEWISE`） |
|---|---|---|
| 捕获/编译 | 不编译（`CompilationMode.NONE`、`cudagraph_mode=NONE`） | `torch.compile` 37.0 s；PIECEWISE 51/51、FULL 35/35 捕获成功 |
| 捕获内存 | 0.0 GiB CUDAGraph | 0.26 GiB（首轮）+ 0.36 GiB（二轮，`estimated 0.32`，偏差 11.1%） |
| `Available KV cache memory` | **27.18 GiB** | **26.75 GiB** |
| `GPU KV cache size` | 254,464 tokens | 250,384 tokens |
| decode（128 token 贪心） | 4.4333 s → **34.636 ms/token** | 1.4855 s → **11.605 ms/token**（**2.98×**） |
| 输出（同一 prompt，贪心，`ignore_eos`） | `1023 ×16` → `'lock' ×16` ❌ | `' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'` ✅ |
| 与 HF transformers 参考是否一致 | ❌ | ✅（逐 token 一致，见 §3.7） |

### 3.5 门禁 2：显式 `--kv-cache-memory-bytes` 仍可覆盖（语义核对）

`vllm/engine/arg_utils.py:1310` 里该 flag 仍叫 `--kv-cache-memory-bytes`（新版 `gpu_worker.py:977` 的提示文案
把 `--kv-cache-memory` 作为简写展示）。代码路径上，`self.cache_config.kv_cache_memory_bytes is not None`
时 **§失败处置里的 0.98 兜底与 git 既有的 gfx906 兜底都进不去**（`gpu_worker.py:670-679` 的 `and` 条件），
因此显式值天然覆盖自动逻辑；本步 27.18/26.75 GiB 的自动结果与提示里的
`--kv-cache-memory=29027187405` / `28177953485` 一致，说明自动逻辑与显式口径对得上。
本步**未**另跑一次显式值实验（该分支未改动、且非本次新增风险），标注为“代码级判定 + 自动值交叉核对”。

### 3.6 门禁 4（4 卡 decode）：**未执行**，原因见 §5.3

任务书要求“**单卡**跑 V1 engine、别占 4 张卡”，且生产容器当前 Exited、其余卡需保持空闲；
设计文档门禁 4 的“4 卡各跑一次 decode”属 S9 的验收范围（本步不占用其它卡）。
本步以“单卡+空闲对照”的方式取到了显存侧证据（§3.10），**4 卡对照留待 S9**。

### 3.7 ⚠️ 追加发现：eager 路径 NaN（本步最重要的负面结论）

#### 3.7.1 现象：eager 输出全为 `1023`、logprobs 全 `nan`

```console
# 单 prompt、多 prompt、batch、不同 chunk 大小，全部一样（/mnt/game/port/s7/s7-prompts-eager-multi.log）
[s7] PROMPT='The capital of France is' -> 'locklocklocklocklocklocklocklocklocklocklocklocklocklocklocklocklocklocklocklocklocklocklocklock'
[s7] PROMPT='2 + 2 =' -> 'locklock...'
[s7] PROMPT='Once upon a time,' -> 'locklock...'
[s7] PROMPT='def fibonacci(n):' -> 'locklock...'
[s7] PROMPT='The largest planet in the solar system is' -> 'locklock...'
[s7] PROMPT='Water boils at' -> 'locklock...'
[s7] SECONDS=1.01

# logprobs 探针（eager，max_tokens=1, logprobs=8）
PROBE single 'The capital of France is' text= 'lock' n_logprobs= 6 naN= 6 top= [(nan, '%'), (nan, '$'), (nan, '#')]
PROBE batch2 'The capital of France is' text= 'lock' n_logprobs= 6 naN= 6 top= [(nan, '$'), (nan, '%'), (nan, '#')]
PROBE other '2 + 2 =' text= 'lock' n_logprobs= 6 naN= 6 top= [(nan, '%'), (nan, '$'), (nan, '#')]
```

⇒ **eager 的第一个 forward 之后 logits 全为 NaN**（不是采样/温度问题：`temperature=0` 的 argmax 在 NaN 上退化成任意 id）。

#### 3.7.2 参考基线：同一权重在 HF transformers 上正常

```console
$ docker exec z100-serve ... python -c "AutoModelForCausalLM.from_pretrained('/mnt/game/models/Qwen3-0.6B', dtype=torch.float16, device_map='cuda:0')"
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k'].
HF_PROMPT='The capital of France is' -> ' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'
HF_PROMPT='2 + 2 =' -> ' 4, so the sum is 4.\n\nBut wait, the problem says'
HF_PROMPT='Water boils at' -> ' 100°C at 1 atm pressure. What is the temperature in'
```

⇒ 权重/驱动/torch 本身没问题，问题在 **vLLM 的某条执行路径**。注意 `tp=1` 且 vLLM 走 `TRITON_ATTN` + `LBNHC` KV layout，
与 HF 的 `sdpa` + 连续 KV 不是同一条路。

#### 3.7.3 A/B 矩阵（`tools/dcu/s7_compile_matrix.sh`，单变量法）

| 配置（单卡、同 prompt、同 128-token 计时） | 输出 | decode ms/token | 日志 |
|---|---|---|---|
| `enforce_eager=True` → `mode=NONE, cudagraph=NONE` | ❌ `lock×16` | 34.636 | `s7-eager.log` |
| `mode=NONE, cudagraph=NONE`（等价，纯 CompilationConfig） | ❌ `lock×16` | 36.473 | `s7-matrix-compile-none.log` |
| `mode=VLLM_COMPILE, cudagraph=NONE`（**不捕获 graph**） | ✅ 正确 16 token | 33.514 | `s7-matrix-cudagraph-none.log` |
| `mode=VLLM_COMPILE, cudagraph=FULL_AND_PIECEWISE`（默认） | ✅ 正确 16 token | 11.605 | `s7-graph.log` |
| `mode=VLLM_COMPILE, cudagraph=NONE, ir_enable_torch_wrap=False` | ✅ `' Paris. The capital of Italy is Rome'` | —（探针） | 内联探针 |
| `mode=NONE, cudagraph=NONE, custom_ops=['all']` | ❌ `lock×8` | —（探针） | 内联探针 |
| `enforce_eager=True` + `rms_norm/fused_add_rms_norm=['native']` | ❌ `lock`，`nan=4` | —（探针） | 内联探针 |
| `enforce_eager=True, max_num_batched_tokens=64` | ❌ `lock×8` | —（探针） | 内联探针 |
| `VLLM_USE_V1=0`（V0 引擎，eager） | ❌ `lock×8` | —（探针） | 内联探针 |

关键对照行：

```console
# 正确的那些 run：内核优先级变成 native-only、由 inductor 出码
INFO [kernel.py:408] Final IR op priority after setting platform defaults:
    IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'], gelu_and_mul_sparse=['native'])
[s7] S7_IDS[cudagraph-none]=[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13]
[s7] S7_TEXT[cudagraph-none]=' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'
[s7] S7_DECODE[cudagraph-none] TOKENS=128 SECONDS=4.2897 MS_PER_TOKEN=33.514

# 错误的那些 run：走 vllm_c 预编内核
INFO [kernel.py:408] Final IR op priority after setting platform defaults:
    IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'], gelu_and_mul_sparse=['native'])
[s7] S7_IDS[compile-none]=[1023, 1023, ...]
[s7] S7_TEXT[compile-none]='locklock...'
```

`gfx906` 上这两条路径的分流点在 `vllm/platforms/rocm.py:1362-1388::get_default_ir_op_priority`
（`using_inductor = backend=='inductor' and mode != NONE` ⇒ 编译时 `['native']`，否则 `['vllm_c','native']`）；
但**不能**简单归因给 `rms_norm` 选择——单测里把 `rms_norm`/`fused_add_rms_norm` 显式设成 `['native']`
仍然 `nan`，说明**至少还有一个 `vllm_c`/非 inductor 路径的算子会产出 NaN**（`custom_ops=['all']` 时亦然）。

#### 3.7.4 结论与影响

- **判定：`enforce_eager`（= `CompilationMode.NONE`）在 gfx906 上不可用于生产**；本步任务书与设计文档的
  “默认保守 = `--enforce-eager`”的假设**被真机数据推翻**。
- **正确配置：保持默认 `torch.compile`（`mode=VLLM_COMPILE`），`cudagraph_mode` 用 NONE 或 FULL_AND_PIECEWISE 都可以**；
  即“**要么开编译，要么别用**”。graph 是否捕获只影响速度（34.6→11.6 ms/token），不影响正确性。
- 本步**没有**把这条落成代码改动（写作用域内也没有很自然的落点，且需要 Lead 决定是“禁 eager”还是“查算子”）；
  建议 S8/S9 先按“默认编译”跑，并把它列入后续排查项（§5.1）。

### 3.8 serve 冒烟（端口 8001，单卡，graph 路径）

启动方式（任务书要求的 torch-first CLI 包装；日志落 `/mnt/game/port/s7/s7-serve.log`）：

```console
$ docker exec -d z100-serve bash -lc '
    source /opt/dtk/env.sh >/dev/null 2>&1
    source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh
    cd $VLLM_DIR
    exec > /mnt/game/port/s7/s7-serve.log 2>&1
    exec python -c "import torch; from vllm.entrypoints.cli.main import main; main()" serve \
      /mnt/game/models/Qwen3-0.6B --dtype float16 --max-model-len 2048 \
      --gpu-memory-utilization 0.9 --port 8001 --host 0.0.0.0 \
      --served-model-name qwen3-0.6b-s7 --no-enable-log-requests'
```

服务端关键行：

```console
(APIServer pid=18610) INFO 10-08 20:46:24 [model_runner.py:1060] Graph capturing finished in 5 secs, took 0.26 GiB
(APIServer pid=18610) INFO 10-08 20:46:24 [gpu_worker.py:719] Available KV cache memory: 26.77 GiB
(APIServer pid=18610) INFO 10-08 20:46:24 [kv_cache_utils.py:2432] GPU KV cache size: 250,608 tokens,
    Maximum concurrency for 2,048 tokens per request: 122.37x
(APIServer pid=18610) INFO 10-08 20:46:33 [model_runner.py:1060] Graph capturing finished in 7 secs, took 0.36 GiB
(APIServer pid=18610) INFO 10-08 20:46:34 [entry.py:136] Starting vLLM server on http://0.0.0.0:8001
(APIServer pid=18610) INFO:     Application startup complete.
```

客户端证据（宿主 → 容器 `172.17.0.2:8001`；容器是 bridge、未映射端口）：

```console
$ curl -s -i --max-time 10 http://172.17.0.2:8001/health
HTTP/1.1 200 OK
date: Thu, 08 Oct 2026 12:46:37 GMT
server: uvicorn
content-length: 0

$ curl -s --max-time 10 http://172.17.0.2:8001/v1/models | head -c 200
{"object":"list","data":[{"id":"qwen3-0.6b-s7","object":"model","created":1791463597,"owned_by":"vllm",
"root":"/mnt/game/models/Qwen3-0.6B",...,"max_model_len":2048,...}]}

$ curl -s --max-time 120 -X POST http://172.17.0.2:8001/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-0.6b-s7","messages":[{"role":"user","content":"What is the capital of France? Answer in one short sentence."}],"max_tokens":32,"temperature":0}'
{"id":"chatcmpl-903a2f3e32d03682","object":"chat.completion","created":1791463597,"model":"qwen3-0.6b-s7","choices":[{"index":0,
"message":{"role":"assistant","content":"<think>\nOkay, the user is asking for the capital of France in a single short sentence.
Let me recall. France's capital is Paris. I need to","refusal":null,...},"finish_reason":"length",...}],
"usage":{"prompt_tokens":21,"total_tokens":53,"completion_tokens":32,...},
"system_fingerprint":"vllm-dev-7bccf005"}

$ curl -s --max-time 120 -X POST http://172.17.0.2:8001/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-0.6b-s7","prompt":"Water boils at","max_tokens":16,"temperature":0}'
{"id":"cmpl-94fd93a95eece294","object":"text_completion",...,"choices":[{"index":0,
"text":" 100°C at 1 atm pressure. What is the temperature in","finish_reason":"length",...}],
"usage":{"prompt_tokens":3,"total_tokens":19,"completion_tokens":16,...}}
```

并发冒烟（8 路 chat/completions，全部 200，含 KV cache 复用）：

```console
req1 http=200 time=0.133879   req2 http=200 time=0.131349   req3 http=200 time=0.132020
req4 http=200 time=0.129738   req5 http=200 time=0.130196   req6 http=200 time=0.131691
req7 http=200 time=0.128102   req8 http=200 time=0.126654
  req1..8 内容: '<think>\n\n</think>\n\nMercury'（8/8 一致）
```

**收工后已停服**（容器保留）：

```console
$ docker exec z100-serve bash -lc 'pkill -f "vllm.entrypoints.cli.main"; sleep 3; ps aux | grep -c "[v]llm"'
$ docker exec z100-serve bash -lc 'source /opt/dtk/env.sh >/dev/null 2>&1; python -c "import torch; print(torch.cuda.mem_get_info(0))"'
FREE_GIB 31.73
```

### 3.9 资源与内存

```console
$ docker exec z100-serve bash -lc 'cat /sys/fs/cgroup/memory.current'   # 空闲时
2.83 GiB                      # 上限 8 GiB（memory.max=8589934592）
$ docker exec z100-serve bash -lc 'df -h /dev/shm | tail -1'
tmpfs            16G  4.5M   16G   1% /dev/shm
$ df -h / /mnt/game
/dev/mapper/ubuntu--vg-ubuntu--lv  115G   62G   48G  57% /      # 根分区未新增构建产物
/dev/nvme1n1p1                     938G  282G  657G  31% /mnt/game
```

`TRITON_CACHE_DIR` 全程指向 `/mnt/game/triton-gfx906/cache`，`/mnt/game/port/s7/` 只放日志与 7 个轮子（约 29 MB）。

### 3.10 自检：其它容器 / 卡 / 仓库

```console
$ docker ps -a --format '{{.Names}}\t{{.Status}}'
z100-serve      Up 52 minutes          # 本步新建（未 --rm，保留）
z100-triton     Up 13 hours            # 未动
z100-build-211  Up 14 hours            # 未动（只读参照；容器内没做任何写入）
z100-build      Up 20 hours            # 未动（只跑过一次只读 torchvision 探针）
qwen38-vllm     Exited (137) 13 hours ago   # 生产容器，全程只做 docker ps/inspect/cp 只读观察，未 start/kill/exec/commit

$ （停服后）torch.cuda.mem_get_info
dev 0 free_GiB 31.73 / dev 1 31.73 / dev 2 31.73 / dev 3 31.73

$ cd /home/lyh/z100-vllm && git status --short
 M CMakeLists.txt / cmake/hipify.py / csrc/... / setup.py / vllm/{_custom_ops,envs}.py / vllm/platforms/rocm.py ...   # 全部 S1–S6 既有改动
?? tools/dcu/      # 其中新增 s7_env.sh / s7_usercustomize.py / s7_gate.sh / s7_compare.sh / s7_compile_matrix.sh / s7_prompts.sh
?? docs/result/    # 归档目录（含本档）
# 未改 vllm/utils/mem_utils.py、vllm/v1/worker/gpu_worker.py、docs/design/*、docs/z100-todolist.md、docs/result/README.md
```

- 未执行任何自编译 HIP 独立可执行程序：本步只用 `pip`、`python`（vLLM/triton/torch JIT）、`curl`、`docker exec`；
  GPU 侧全程无挂死、无 SIGABRT、无不可杀进程；唯一的 `kill -9` 是清理早前探针留下的 `pip download` 残留进程（非 GPU 进程）。
- `z100-serve` 里有一批 `<defunct>` 僵尸 python（父进程是容器的 1 号 `sleep infinity`，不会被回收）——它们不占 CPU/显存，仅记录以免后续误判。

---

## 4. 门禁判定

| # | 门禁（原文） | 判定 | 证据指向 |
|---|---|---|---|
| 1a | 单卡 `Available KV cache memory` > 0 | ✅ **通过** | §3.2：eager `27.18 GiB`；§3.3：graph `26.75 GiB`；§3.8：serve `26.77 GiB` |
| 1b | **未使用** `--kv-cache-memory-bytes` | ✅ **通过** | §3.2/§3.3 的启动命令里没有该 flag；日志无 `automatically raising it to 0.98`、无 `No available memory for the cache blocks` |
| 1c | 不需要手工干预 / DTK 峰值判定没有异常 | ✅ **通过** | §3.2 `gpu_worker.py:977`：`1.4 GiB consumed + 0.2 GiB peak activation`，`Desired (0.9, 28.79 GiB)` 与实测一致；`torch_peak` 未失真 ⇒ 设计文档第 2 步的 `mem_utils.py` 钳制**不需要** |
| 2 | 显式 `--kv-cache-memory-bytes` 仍能覆盖自动逻辑 | ✅ **代码级通过**（未单独跑） | §3.5：`gpu_worker.py:670-679` 的 `kv_cache_memory_bytes is None` 条件使显式值天然跳过两套兜底；自动值 27.18 GiB 与提示里的显式等价值 27.03 GiB 只差“是否填满” |
| 3a | graph **能捕获** | ✅ **通过** | §3.3：PIECEWISE `51/51`、FULL `35/35`（二轮 `35/35`），`Graph capturing finished in 8 / 10 secs, took 0.26 / 0.36 GiB` |
| 3b | graph 路径**数值正确** | ✅ **通过** | §3.4/§3.7.3：graph 输出 `' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'`，与 HF 参考逐 token 一致 |
| 3c | graph **decode 速度对比** | ✅ **通过** | §3.4：11.605 vs 34.636 ms/token（**2.98×**） |
| 3d | eager 路径结论 | ⛔ **失败（结论明确）** | §3.7：`enforce_eager`/`mode=NONE` 下 logits 全 NaN、输出 `lock×16`；9 组 A/B 已把范围收敛到“非 inductor 的 vllm_c/自定义算子路径” |
| 4 | 4 卡各跑一次 decode（设计文档） | ⏸️ **未执行（移交 S9）** | §3.6：任务书要求单卡；生产容器 Exited、其余卡保持空闲。已给出单卡显存侧对照（§3.10） |
| 5 | serve 冒烟：≥8001、`/health` + chat/completions 成功、收工停服 | ✅ **通过** | §3.8：`HTTP/1.1 200 OK`、chat 与 completions 均 200 且文本合理、8 路并发 8/8 成功、已 `pkill` 停服 |
| 6 | 新建 `z100-serve`（同镜像、参数照抄、`--memory=8g --shm-size=16g --ipc=host`、不加 `--rm`、挂 `/mnt/game`+`/src:ro`） | ✅ **通过** | §2.4 `docker inspect` 原文 |
| 7 | 环境带 S6 自编 triton + 缓存目录在 `/mnt/game` | ✅ **通过** | §3.0：`triton dist = 3.6.0+git7c56a5e4 /mnt/game/triton-gfx906/install/...`、`TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache` |
| 8 | 硬约束：禁 HIP 可执行程序 / 不碰生产容器 / 不动 `z100-build` / 不替换 DTK 库 / 产物只落 `/mnt/game` / 不改禁改文档 | ✅ **遵守** | §3.10 |
| 9 | 仅在门禁不过时才改 `mem_utils.py` / `gpu_worker.py` | ✅ **未改** | §2.1、§3.10 |

**S7 总体：✅ 门禁 1、3a–3d、5、6、7 全部通过；门禁 4 移交 S9。**
唯一负面结论是 eager 路径不可用（§3.7），它不影响“KV cache 自动算出 + graph 可用”这两条主线，
但**必须**写进 S8/S9 的服务起法（§6.1）。

---

## 5. 未完成项 / 阻塞与原因

### 5.1 ⚠️ eager（`CompilationMode.NONE`）在 gfx906 上数值错误 —— **未修，需 Lead 决策**

- 现象与证据：§3.7（logits 全 NaN、6 个 prompt 全崩、V0/V1 都崩、`rms_norm=native` 与 `max_num_batched_tokens=64` 都无法规避）。
- 已收敛的范围：**`mode=VLLM_COMPILE`（是否捕获 graph 都行）正确；`mode=NONE` 错误**。
  `vllm/platforms/rocm.py:1362-1388` 是两条路径的分流点（`['native']` vs `['vllm_c','native']`），
  但已排除 `rms_norm`/`fused_add_rms_norm` 是唯一元凶（显式 `native` 仍 NaN）。
- 未做：定位到具体算子（需要逐层 logits 钩子或 `torch.isfinite` 二分，属独立调试任务）；
  也**没有**在 `vllm/**` 里加任何“禁 eager”或“强制编译”的默认值（写作用域内没有自然落点，且会改变非 DTK 行为）。
- 建议：S8 起步时把它当 P0 排查（`vllm_c` 算子里的 `reshape_and_cache_kernel_flash` / `unified_attention` /
  LBNHC layout 与 `cache_kernels` 都可能，日志显示 eager 会额外 JIT 出 `reshape_and_cache_kernel_flash`、
  `kernel_unified_attention_2d/3d`）；在定位前，**S8/S9 一律不要用 `--enforce-eager`**。

### 5.2 环境依赖整链（镜像自带太旧）—— 未固化进仓库

`z100-serve` 里补装的 6 个包（§2.2）与 `usercustomize.py` shim（§2.3）都是**容器内运行期**动作，
本次没有落进 `requirements/dcu*.txt`（那超出本步写作用域）。若 S8/S9 要复用，两种做法：
① 直接用现成的 `z100-serve`（保留中，重启命令见 §6.1）；② 在新容器里重跑 §7 的第 1 步。
**注意**：`z100-build-211` **没有**这些补装，别指望它直接能起 serve。

### 5.3 设计文档门禁 4（4 卡对照）未执行

原因：任务书明确要求单卡（`HIP_VISIBLE_DEVICES=0`）且生产容器已停、其余卡应保持空闲。
`hy-smi` 在宿主上读 `mkfd` 失败、只报 `VRAM% = 0%`（无 MiB 明细），无法给出“<5% 偏差”的对照口径；
该项应在 S9 用 4 卡统一测量。本步给出的是 `torch.cuda.mem_get_info` 的空闲对照（4 卡均 31.73 GiB）。

### 5.4 明确“未执行”的项

- `vllm/utils/mem_utils.py` / `vllm/v1/worker/gpu_worker.py` 的任何修改（门禁 1 通过，不需要）；
- 任何 4 卡 / TP>1 / RCCL 实验（设计文档 §4 的 TP/RCCL 联测属 S9）；
- 生产容器 `qwen38-vllm` 的任何启停（全程只 `docker ps` / `docker inspect` / `docker cp` 只读观察）；
- 任何自编译 HIP 独立可执行程序。

---

## 6. 对后续步骤的影响

### 6.1 S8/S9 的服务起法与环境变量（**可直接照抄**）

```bash
# 0) 容器：复用 z100-serve（保留中，未 --rm）
docker ps -a --format '{{.Names}}\t{{.Status}}' | grep z100-serve   # 期望 Up
# 若容器被删/停止，用 §7 第 1 步重建，并重跑 §7 第 2 步（依赖补装 + shim）

# 1) 每次开新 shell 都要 source 这两层（顺序不能反）
docker exec -it z100-serve bash
source /opt/dtk/env.sh
source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh    # 或 /src/tools/dcu/s7_env.sh（仓库只读挂载）
# 得到：HIP_VISIBLE_DEVICES=0、PYTHONPATH=<自编 triton>:<fork 副本>:<tools/dcu>、
#       TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache、VLLM_DCU_GCN_ARCH=gfx906、VLLM_ROCM_USE_SKINNY_GEMM=0

# 2) 起服务（V1 engine；torch-first CLI 包装，避免 vllm CLI 直接 import 崩）
cd $VLLM_DIR
python -c "import torch; from vllm.entrypoints.cli.main import main; main()" serve \
  /mnt/game/models/Qwen3-0.6B --dtype float16 --max-model-len 2048 \
  --gpu-memory-utilization 0.9 --port 8001 --host 0.0.0.0 \
  --served-model-name qwen3-0.6b-s7
# 容器是 bridge 网络、未映射端口：宿主侧用 http://172.17.0.2:8001（不要用 127.0.0.1）

# 3) 关键约束（本步实测）
#   ❌ 不要加 --enforce-eager：gfx906 上 logits 全 NaN（§3.7）
#   ✅ node 保持默认（torch.compile + FULL_AND_PIECEWISE）；想省内存/启动时间可用
#      CompilationConfig(mode=VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.NONE)（正确，33.5 ms/token）
#   ✅ graph 首次启动约 167 s（compile 37 s + JIT warmup 81 s + capture 18 s），别误判为挂死
#   ✅ 必须带自编 triton，否则 attention 出不了 gfx906 码 / SFU hazard 会让 softmax 静默算错
```

`z100-serve` 重启命令（容器仍存在时）：

```bash
docker start z100-serve   # 里面的补装与 usercustomize.py 都还在
```

### 6.2 被本步更新/推翻的前提

1. **推翻**设计文档 §3 的“graph 保守默认 = `--enforce-eager`”：真机数据是**反的**——eager 数值错、编译正确。
   设计文档 §3 的判定条件“graph 若 capture 失败或数值不一致 ⇒ 默认 enforce_eager”**不成立**：
   capture 成功且数值正确的是 graph 一侧。
2. **确认**（与设计文档 §1/§2 一致）：gfx906 的 0.98 兜底**用不上**，`Available KV cache memory` 在默认 0.9 下就有 27 GiB；
   `MemorySnapshot`/`torch peak` 在 DTK 26.04 + torch 2.11 上**没有**出现“负 KV cache”的失真。
3. **新增**运行期依赖前提：镜像自带的 `mistral_common/openai/xgrammar/torchvision` 与这套 fork 不匹配（§2.2/§3.1），
   任何后续步骤在**新容器**里起 V1 engine 都会撞到同一串错误；`usercustomize.py` 的 torch-first 兜底也是必需项。

### 6.3 对 S8 的直接影响

- S8 的“8K 上下文日志为 `TRITON_ATTN`”在本步已经能观察到（`rocm.py:895` 覆写为 `TRITON_ATTN`），
  S8 只需把 `--max-model-len` 拉到 8192 并在**默认编译**下确认即可；KV 侧余量充足
  （2048 时 250k token ≈ 122× 并发；8192 时约 30×，S8 可直接算）。
- S8 若要用 eager 做对照实验，必须先解决 §5.1，否则会把“eager NaN”误读成 attention 路径问题。

### 6.4 对 S9 的直接影响

- S9 的性能对拍基线应取**编译路径**：单请求 decode 11.6 ms/token（Qwen3-0.6B/fp16/卡 0，含 HTTP 与采样前处理）；
  同一条件 eager 34.6 ms/token **不可作为生产配置**。
- S9 需要补设计文档门禁 4（4 卡 decode + `hy-smi` 偏差 <5%）；注意 `hy-smi` 在本宿主读 `mkfd` 失败、无 MiB 明细，需换 `torch.cuda.mem_get_info` 口径。

---

## 7. 复现方式

```bash
# ---------- 1) 建容器（一次；与 z100-build-211 同镜像、同设备参数） ----------
docker run -d --name z100-serve \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    --group-add video --security-opt label=disable --ipc host \
    --memory=8g --memory-swap=8g --shm-size=16g \
    -v /opt/hyhal:/opt/hyhal:ro -v /home/lyh/z100-vllm:/src:ro -v /mnt/game:/mnt/game \
    -w /mnt/game/port \
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 \
    bash -c 'sleep infinity'
# 上面这行就是 §2.4 docker inspect 记录的同名镜像，可直接粘贴

# ---------- 2) 容器内补齐运行依赖（一次） ----------
docker exec z100-serve bash -lc '
  set -e
  mkdir -p /mnt/game/port/s7/pip-wheels
  W=/mnt/game/port/s7/pip-wheels
  # torch：复用 S4b 已下好的轮子
  pip install --no-deps /mnt/game/port/wheels-211/torch-2.11.0+das.opt1.dtk2604.*.whl
  # 其余从 PyPI 直取（本步实测可达）
  pip download --no-deps -d $W mistral_common==1.11.5 openai==2.54.0 "xgrammar==0.2.8" "apache-tvm-ffi>=0.1.11" torchvision==0.26.0
  pip install --no-deps $W/mistral_common-*.whl $W/openai-2.54.0-*.whl $W/xgrammar-0.2.8-*.whl \
                        $W/apache_tvm_ffi-*.whl $W/torchvision-0.26.0-*.whl
  # 运行期 shim（源文件在仓库 tools/dcu/，见第 3 步拷进容器）
  '

# ---------- 3) 同步本步脚本 + 装 shim ----------
docker exec z100-serve bash -lc '
  D=/mnt/game/port/gfx906-vllm-211/tools/dcu
  mkdir -p $D /root/.local/lib/python3.10/site-packages
  cp /src/tools/dcu/s7_env.sh /src/tools/dcu/s7_usercustomize.py /src/tools/dcu/s7_gate.sh \
     /src/tools/dcu/s7_compare.sh /src/tools/dcu/s7_compile_matrix.sh /src/tools/dcu/s7_prompts.sh $D/
  chmod 755 $D/s7_*.sh $D/s7_usercustomize.py
  # 必须叫 usercustomize.py：/usr/lib/python3.10/sitecustomize.py 已被镜像占用
  cp $D/s7_usercustomize.py /root/.local/lib/python3.10/site-packages/usercustomize.py
  # 确认 shim 生效
  source /opt/dtk/env.sh >/dev/null 2>&1; source $D/s7_env.sh
  python -c "import sys, torch, torchvision; from torchvision.transforms import InterpolationMode; \
             print(\"SHIM_OK\", \"usercustomize\" in sys.modules, torchvision.__version__, InterpolationMode.BILINEAR)"'

# ---------- 4) 门禁 1（eager；会得到 KV cache 数值 + 已知错误的输出） ----------
docker exec z100-serve bash -lc '
  source /opt/dtk/env.sh >/dev/null 2>&1
  source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh
  cd $VLLM_DIR
  timeout 1200 bash tools/dcu/s7_gate.sh eager /mnt/game/models/Qwen3-0.6B /mnt/game/port/s7/s7-eager.log'
grep -E "Available KV cache memory|automatically raising|No available memory|S7_IDS|S7_DECODE" \
     /mnt/game/port/s7/s7-eager.log

# ---------- 5) 门禁 3（graph；约 167 s 启动） ----------
docker exec z100-serve bash -lc '
  source /opt/dtk/env.sh >/dev/null 2>&1
  source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh
  cd $VLLM_DIR
  timeout 1800 bash tools/dcu/s7_gate.sh graph /mnt/game/models/Qwen3-0.6B /mnt/game/port/s7/s7-graph.log'
grep -E "Graph capturing finished|Available KV cache memory|S7_IDS|S7_TEXT|S7_DECODE" \
     /mnt/game/port/s7/s7-graph.log

# ---------- 6) 数值归因（§3.7 的 A/B 矩阵） ----------
docker exec z100-serve bash -lc '
  source /opt/dtk/env.sh >/dev/null 2>&1
  source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh
  cd $VLLM_DIR
  for c in cudagraph-none compile-none; do
    bash tools/dcu/s7_compile_matrix.sh /mnt/game/models/Qwen3-0.6B $c
  done'
grep -aE "S7_TEXT|S7_DECODE" /mnt/game/port/s7/s7-matrix-*.log

# ---------- 7) HF 参考（证明权重/驱动没问题） ----------
docker exec z100-serve bash -lc '
  source /opt/dtk/env.sh >/dev/null 2>&1; source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh
  python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p=\"/mnt/game/models/Qwen3-0.6B\"; tok=AutoTokenizer.from_pretrained(p)
m=AutoModelForCausalLM.from_pretrained(p, dtype=torch.float16, device_map=\"cuda:0\").eval()
i=tok(\"The capital of France is\", return_tensors=\"pt\").to(\"cuda:0\")
print(tok.decode(m.generate(**i, max_new_tokens=16, do_sample=False)[0][i[\"input_ids\"].shape[1]:], skip_special_tokens=True))"'

# ---------- 8) serve 冒烟（8001；收工必须停服） ----------
docker exec -d z100-serve bash -lc '
  source /opt/dtk/env.sh >/dev/null 2>&1
  source /mnt/game/port/gfx906-vllm-211/tools/dcu/s7_env.sh
  cd $VLLM_DIR
  exec > /mnt/game/port/s7/s7-serve.log 2>&1
  exec python -c "import torch; from vllm.entrypoints.cli.main import main; main()" serve \
    /mnt/game/models/Qwen3-0.6B --dtype float16 --max-model-len 2048 \
    --gpu-memory-utilization 0.9 --port 8001 --host 0.0.0.0 --served-model-name qwen3-0.6b-s7 \
    --no-enable-log-requests'
for i in $(seq 1 40); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://172.17.0.2:8001/health || true)
  [ "$code" = "200" ] && break; sleep 5
done
curl -s -i http://172.17.0.2:8001/health | head -1
curl -s -X POST http://172.17.0.2:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-0.6b-s7","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32,"temperature":0}'
# 停服（容器保留）
docker exec z100-serve bash -lc 'pkill -f "vllm.entrypoints.cli.main"'
```

> §7 第 1 步的镜像行已与 §2.4 的 `docker inspect` 原文逐字一致。

---

## 8. 参考

- 设计文档：[docs/design/z100-07-memory-kvcache-graph.md](../design/z100-07-memory-kvcache-graph.md)
- 前序归档：
  [z100-06-triton-gfx906-result.md](z100-06-triton-gfx906-result.md) §5/§7（自编 triton 用法 + SFU hazard 修复）、
  [z100-04b-stack-upgrade-result.md](z100-04b-stack-upgrade-result.md) §6（2.11 构建副本与四个扩展）、
  [z100-04b-stack-upgrade-result-2.md](z100-04b-stack-upgrade-result-2.md)（S4c：两项修复落地）
- 关键源码位置：
  - `vllm/v1/worker/gpu_worker.py:664-722`（`available_kv_cache_memory_bytes`、gfx906 0.98 兜底、`Available KV cache memory` 日志）
  - `vllm/v1/worker/gpu_worker.py:120-144`（`maybe_rocm_profiling_fallback`，本步未触发）
  - `vllm/utils/mem_utils.py:132-170`（`MemorySnapshot.measure`，本步未改）
  - `vllm/platforms/rocm.py:1362-1388`（`get_default_ir_op_priority`：编译 ⇒ `native`，非编译 ⇒ `vllm_c`）
  - `vllm/platforms/hygon_dtk.py:226-247`（`dcu_device_memory_total_bytes` sysfs 口径，钳制备用）
  - `vllm/config/compilation.py:606-640`（`cudagraph_mode` 语义与默认 `FULL_AND_PIECEWISE`）
  - `vllm/model_executor/warmup/kernel_warmup.py:158-237`（JIT warmup 与 minimax_m3 的 torchvision 依赖）
  - `vllm/transformers_utils/processors/minimax_m3.py:19`（顶层 `from torchvision.transforms import InterpolationMode`）
- 本步原始日志（全部在 `/mnt/game/port/s7/`）：
  `s7-eager.log`、`s7-graph.log`、`s7-matrix-cudagraph-none.log`、`s7-matrix-compile-none.log`、
  `s7-compare-cg-off-nocompile.log`、`s7-compare-compile-nocg.log`、`s7-compare-nocompile-native{,2}.log`、
  `s7-prompts-eager-multi.log`、`s7-serve.log`；补装轮子在 `pip-wheels/`
- 本步脚本（仓库）：[`tools/dcu/s7_env.sh`](../../tools/dcu/s7_env.sh)、
  [`tools/dcu/s7_usercustomize.py`](../../tools/dcu/s7_usercustomize.py)、
  [`tools/dcu/s7_gate.sh`](../../tools/dcu/s7_gate.sh)、
  [`tools/dcu/s7_compile_matrix.sh`](../../tools/dcu/s7_compile_matrix.sh)、
  [`tools/dcu/s7_compare.sh`](../../tools/dcu/s7_compare.sh)、
  [`tools/dcu/s7_prompts.sh`](../../tools/dcu/s7_prompts.sh)
