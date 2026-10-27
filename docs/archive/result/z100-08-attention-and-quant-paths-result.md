# S8 路径与后端核对（attention / 量化 / 禁用清单）— 执行结果

> 归档时间：2026-10-08（容器 CST）
> 执行者：子 agent（S8 路径与后端核对）
> 对应设计文档：[docs/design/z100-08-attention-and-quant-paths.md](../design/z100-08-attention-and-quant-paths.md)
> 前一档：[z100-07-memory-kvcache-graph-result.md](z100-07-memory-kvcache-graph-result.md)（S7：服务起法、环境补装、`usercustomize.py` shim、**禁 `--enforce-eager`**）
> 状态：✅ **门禁 1/2/4 通过，门禁 3 以"自研 sdot4 数值跑通"判定通过**；
> ⚠️ 追加发现（并已最小修复）：gfx906 上**所有 W4A16/W8A16（compressed-tensors / GPTQ / AWQ-marlin / quark）模型在 kernel 选择期即 AttributeError 崩溃**，
> 以及**任何带 `quantization_config` 的量化模型都会被 `MoeWNA16Config.override_quantization_method()` 的过期签名打死**（见 §3.7/§5.1）；
> ⏸️ 生产 27B W8A8 INT8 单卡端到端未通过（权重 28.51 GiB / 卡 31.98 GiB，编译器阶段停摆，见 §3.9）。

**一句话结论**：单卡（卡 1）8K 上下文 `vllm serve` 日志为 **`TRITON_ATTN`**（`rocm.py:895` 原文），
aiter / hipBLASLt / skinny GEMM / flash-attn **四条路径全部确认未被进入**（各有运行期原文：`_rocm_C` 只剩
`paged_attention`、`rocblaslt warning: No paths matched ... *gfx906*co`、`is_aiter_found_and_supported()=False`、
`flash-attn is not available on gfx906 ROCm`）；`on_gfx906()/on_cdna()/on_gfx9()` 真机为 **`True True False`**；
W8A8 INT8 走 **`TritonInt8ScaledMMLinearKernel` → triton_scaled_mm`，6 组形状 rel_err ≤ 4.1e-04 + `probe_triton.py` `INT8DOT_OK` / 32 条 `v_dot`。
本步在 `vllm/**` 做了 **2 处最小修复**（3 个文件的面：`MPLinearKernel.py` + 5 个 kernel 调用点、`moe_wna16.py`），
两条修复都有"修复前 / 修复后"原始日志对比。

---

## 1. 目标与验收标准

### 1.1 设计文档门禁原文（`docs/design/z100-08-attention-and-quant-paths.md` §验收）

1. 8K 上下文请求成功且日志为 `TRITON_ATTN`；
2. 日志中不出现 aiter / hipBLASLt / skinny GEMM / flash-attn 被选中的痕迹；
3. W8A8 INT8（若属验收范围）用自研 sdot4 跑通一次，并与生产基线对比记录；
4. `on_gfx906()/on_cdna()/on_gfx9()` 真机输出为 `True True False`。

### 1.2 本步任务书追加的验收标准

- 新建容器 `z100-s8`（同镜像；设备/group 参数照抄 `z100-build-211`；`--memory=10g --shm-size=16g --ipc=host`；不加 `--rm`），
  挂 `/mnt/game` 与只读工作区 `/src`；按 S7 归档 §7 清单补装纯 Python 依赖与 `usercustomize.py`；
- **固定单卡 `HIP_VISIBLE_DEVICES=1`**（Lead 用卡 0、另一子 agent 用卡 2）；
- **不用 `--enforce-eager`**（todolist 硬约束 13）；如需对照只允许 `VLLM_COMPILE`；
- 量化路径逐项给出"gfx906 上走哪条、是否可用、若不可用是否明确报错"；能实测的实测，不能实测的写"未执行 + 原因" + 静态 `file:line` 依据；
- 产出**禁用清单**（哪些后端/路径必须显式关闭或绕开）并给出代码位置；
- 若发现误选/误判路径，做**最小修复**并给出修复前后日志对比。

### 1.3 硬约束（本步遵守情况见 §4 门禁 8）

禁止执行自编译 HIP 独立可执行程序；不 start/kill/exec 生产容器 `qwen38-vllm`；不动 `z100-build`/`z100-build-211` 构建树；
不替换 DTK 库；产物与缓存只落 `/mnt/game`；不改 `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`；
写作用域 `vllm/**`（仅最小修复）、`tools/dcu/**`、本归档。

---

## 2. 实际改动

### 2.1 `vllm/**` 代码改动（2 处最小修复）

| # | 文件 | 行 | 改动摘要 |
|---|---|---|---|
| A | `vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py` | +7（dataclass）、+1（`__init__` 参数）、+1（`self.w_gidx_name`）、+11（新方法） | ① `MPLinearLayerConfig` 补 `has_g_idx: bool = False`；② `__init__` 补 `w_gidx_param_name`／`self.w_gidx_name`；③ 新增 `_get_gidx_param()`（**不改变 `_get_weight_params` 的 3 元组契约**） |
| A′ | `.../mixed_precision/gfx906_gptq_wna16.py:122-123` | ±2 | `w_q, w_s, w_zp = self._get_weight_params(layer)` + `w_g_idx = self._get_gidx_param(layer)` |
| A″ | `.../mixed_precision/{exllama.py:157-158,triton_w4a16.py:278,conch.py:130,torch_wna16.py:51}` | ±1 各 | 同上；其中 `triton_w4a16`/`conch`/`torch_wna16` 丢弃第 4 个返回值，直接改回 3 元组解包 |
| B | `vllm/model_executor/layers/quantization/moe_wna16.py:149-151` | ±3 | `override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None)`：补 `hf_config` 形参 |

**A 的论证（为什么是"误选路径"而不是新功能）**：`gfx906_gptq_wna16.py:37`、`triton_w4a16.py:214`、`torch_wna16.py:34`、
`exllama.py:39/111` 这 5 个 fork 自带的 kernel 都在读 `c.has_g_idx`，但 `MPLinearKernel.py` 的 dataclass 从来没定义过它
（与 `/mnt/game/port/upstream-vllm` 的同名文件逐行 diff：上游第 22 行有 `has_g_idx: bool`）。
`Gfx906GPTQWNA16LinearKernel` 是 `_POSSIBLE_KERNELS[PlatformEnum.ROCM]` 的**第 1 个**且其 `can_implement` 在
`on_gfx906()` 判定通过后立刻读 `c.has_g_idx` ⇒ gfx906 上**每次** kernel 选择都抛 `AttributeError`。
同理，`_get_weight_params` 在 fork 里返回 **3** 元组，而上述 5 个 kernel 按 **4** 元组解包 ⇒ 即使跳过选择也会在首个 forward 抛
`ValueError: not enough values to unpack`。两条都是"fork 的 kernel 已按新 API 写、而 base 类落后"的错配，修复方向唯一。

**B 的论证**：`vllm/config/model.py:1570-1572` 调用
`method.override_quantization_method(quant_cfg, self.quantization, hf_config=self.hf_config)`；
`QuantizationConfig.override_quantization_method`（`base_config.py:154-165`）与其余 6 个实现
（`auto_gptq.py:214`、`auto_awq.py:263`、`mxfp4.py:123`、`humming.py:179`、`inc/inc.py:361`、`modelopt.py:440/756/1160/1568`）
**全部**已接受 `hf_config`，只有 `moe_wna16.py` 没有；而 `_verify_quantization` 的探测循环（`model.py:1567-1572`）
对**任何**带 `quantization_config` 的模型都会遍历到 `moe_wna16` ⇒ 该模型在量化校验期必然 `TypeError` 崩溃。

**非 DTK / 非 gfx906 行为**：`has_g_idx` 有默认值，A′/A″ 全部是"把已经写错的解包改对"，B 只是补一个被忽略的形参；
三者都不改变任何分支条件，也不影响未装 DTK 的机器（无 `is_hygon_dtk()` 相关分支，无 arch 判定改动）。

### 2.2 `tools/dcu/**` 新增（本步产物；容器内副本同步到 `/mnt/game/port/gfx906-vllm-211/tools/dcu/`）

| 文件 | 说明 |
|---|---|
| `tools/dcu/s8_env.sh` | 在 `s7_env.sh` 之上把 `HIP_VISIBLE_DEVICES` 固定为卡 1，并把 `VLLM_CACHE_ROOT` 指到 `/mnt/game/port/s8/vllm-cache` |
| `tools/dcu/s8_path_probe.py` | 运行期路径探针（arch 谓词 / aiter / skinny / `_rocm_C` op 面 / attention 优先级 / ViT backend / gfx906 专用 op） |
| `tools/dcu/s8_quant_probe.py` | 量化路径探针（INT8 / W4A16 / FP8 的 kernel 选择，`has_g_idx` 语义，GGUF 门控，MoE WNA16 门控） |
| `tools/dcu/s8_int8_gfx906_check.py` | `triton_scaled_mm` 6 组形状的 INT8 数值自检（自研 sdot4 实测） |

> `probe_triton.py`、`s7_*.sh` 为复用 S6/S7 资产，未改动。
> **未改动**：`docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`、`tests/**`、`csrc/**`、`tools/dcu/env.sh`、`tools/dcu/s7_*`。

### 2.3 容器 `z100-s8` 实际配置（`docker inspect` 原文）

```console
$ docker run -d --name z100-s8 \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    --group-add video --security-opt label=disable --ipc host \
    --memory=10g --memory-swap=10g --shm-size=16g \
    -v /opt/hyhal:/opt/hyhal:ro -v /home/lyh/z100-vllm:/src:ro -v /mnt/game:/mnt/game \
    -w /mnt/game/port \
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 \
    bash -c 'sleep infinity'
e9e7399bdf540b458499887625f0a344db66133358098612a34a464275ee52f3

$ docker inspect z100-s8 --format '...'
ImageID=sha256:d1c08bf30382f6d8f49e0ad7130ceb4e3a31484ce4bdd3b1aeb7787b5d2fb14e   # 与 z100-build-211 同镜像
Devices=[{/dev/kfd /dev/kfd rwm} {/dev/mkfd /dev/mkfd rwm} {/dev/dri /dev/dri rwm}]
GroupAdd=[video] SecurityOpt=[label=disable] Ipc=host
ShmSize=17179869184  Memory=10737418240  MemorySwap=10737418240  AutoRemove=false
WorkingDir=/mnt/game/port  Binds=[/opt/hyhal:/opt/hyhal:ro /home/lyh/z100-vllm:/src:ro /mnt/game:/mnt/game]
```

### 2.4 运行期依赖补装（**在 S7 §2.2 的 6 个包之上追加第 7 个**）

S7 §7 第 2 步的 6 个包（torch 2.11.0+das / mistral_common 1.11.5 / openai 2.54.0 / xgrammar 0.2.8 /
apache-tvm-ffi 0.1.14.post1 / torchvision 0.26.0）＋ `usercustomize.py` shim 全部照抄；**本步新发现 1 个必需包**：

| 包 | 镜像自带 | 需要 | 触发失败点 |
|---|---|---|---|
| `compressed-tensors` | **0.14.0.1** | **0.15.0.1**（`requirements/common.txt:40`） | `ModuleNotFoundError: No module named 'compressed_tensors.compressors.pack_quantized'`（`schemes/__init__.py:15` → `compressed_tensors_wNa8o8.py:14`） |

```console
$ docker exec z100-s8 ... pip install --no-deps /mnt/game/port/s8/pip-wheels/compressed_tensors-0.15.0.1-py3-none-any.whl
CT_VERSION 0.15.0.1
PACK_QUANTIZED_OK
CT_META 0.15.0.1
```

> **生产容器 `qwen38-vllm` 里也是 0.14.0.1**（`docker cp` 只读取出其 `dist-info/METADATA`：`Version: 0.14.0.1`，
> 且 `compressed_tensors/compressors/` 同样没有 `pack_quantized/`）——说明"跑通生产模型"的那个 vLLM 构建与本 fork
> 不是同一份代码，**本 fork + 本镜像**需要这个升级。
> 网络：`pip`（PyPI）在容器内可用；HF 也可达（`hf=200`）但下载很不稳定（见 §3.8）。

### 2.5 在 `/mnt/game` 下新增的产物

```
/mnt/game/port/s8/                         本步全部日志（见 §8）
/mnt/game/port/s8/pip-wheels/              compressed_tensors-0.15.0.1-py3-none-any.whl
/mnt/game/port/gfx906-vllm-211/tools/dcu/  s8_env.sh / s8_path_probe.py / s8_quant_probe.py / s8_int8_gfx906_check.py
/mnt/game/models/tinyllama-w4a16-group128/ 下载未完成的 W4A16 测试模型（见 §5.3）；
                                           截断权重已改名 model.safetensors.INCOMPLETE-426459208
                                           （426,459,208 B；正确长度 761,968,600 B），避免后续步骤误当可用权重
```

`TRITON_CACHE_DIR` 全程 `/mnt/game/triton-gfx906/cache`。
**例外（如实记录）**：`torch.compile` 的编译缓存默认落 `$HOME/.cache/vllm`，本次跑出 **2.1 GB** 在容器可写层；
`tools/dcu/s8_env.sh` 已加 `VLLM_CACHE_ROOT=/mnt/game/port/s8/vllm-cache` 修正，但**本步已产生的 2.1 GB 在容器层**。

---

## 3. 执行过程与原始证据

### 3.0 环境自检（卡 1、目标栈、自编 triton）

```console
$ docker exec z100-s8 bash -lc '
    source /opt/dtk/env.sh >/dev/null 2>&1
    source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh
    cd $VLLM_DIR
    python /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_path_probe.py'
[dcu-s8] HIP_VISIBLE_DEVICES=1
torch dist   = 2.11.0
torch meta   = 2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7
torch hip    = 6.3.26113
devcount     = 1
name0        = Z100L arch=gfx906:sramecc+:xnack-
triton dist  = 3.6.0+git7c56a5e4 / /mnt/game/triton-gfx906/install/triton/__init__.py
```

### 3.1 门禁 4：arch 谓词真机输出（原文，`/mnt/game/port/s8/s8-path-probe.log`）

```console
--- arch predicates ---
on_gfx906    = True
on_cdna      = True
on_gfx9      = False
on_gfx90a    = False
on_gfx942    = False
on_gfx950    = False
on_gfx1250   = False
on_gfx1x     = False
on_rdna4     = False
cdna_version = 0
_GCN_ARCH    = gfx906
_ON_CDNA     = True  _ON_GFX9 = False
```

⇒ 与设计文档"语义澄清"一致：**不要把 gfx906 加进 `_ON_GFX9`**；`get_cdna_version()` 在 gfx906 返回 **0**（只映射 gfx90a/942/950/1250），
这是 §3.2 全部 aiter 门控的数值基础。

### 3.2 门禁 2（a）aiter：装了也进不去（原文）

```console
--- aiter ---
INFO [rocm.py:958] Using Flash Attention (Triton backend) for ViT model on CDNA.
WARNING [_aiter_ops.py:174] Could not locate PyTorch's HIP runtime; AITER JIT extensions may fail to resolve HIP symbols.
IS_AITER_FOUND                        = True
is_aiter_found_and_supported()        = False
is_aiter_found_and_supported_on_rdna4() = False
rocm_aiter_ops.is_enabled                         = None
rocm_aiter_ops.is_linear_enabled                  = None
rocm_aiter_ops.is_mha_enabled                     = None
rocm_aiter_ops.is_mla_enabled                     = None
rocm_aiter_ops.is_fused_moe_enabled               = None
rocm_aiter_ops.is_rdna_aiter_enabled              = False
rocm_aiter_ops.is_triton_unified_attn_enabled     = None
VLLM_ROCM_USE_AITER                  = False
```

三重证据，互相独立：

1. **`import aiter` 本身就失败**（不是 arch 门控，是依赖链）：
   ```console
   $ python -c "import torch; import aiter"
   AITER_IMPORT_FAIL ImportError /opt/dtk/lib/libamdhip64.so.6: undefined symbol: hsa_ext_log_finish_alloc, version ROCR_1
   ```
   ⇒ `AiterInt8ScaledMMLinearKernel.is_supported()` 给出的理由就是 `requires \`aiter\` to be installed.`（见 §3.6）。
2. **arch 门控**：`is_aiter_found_and_supported()` 实现是 `get_cdna_version() > 2`（`vllm/_aiter_ops.py:202-218`），
   gfx906 上 `get_cdna_version()==0` ⇒ 恒 `False`（**与 `VLLM_ROCM_USE_AITER` 环境变量无关**，看/不看 env 都不行）。
3. **所有 aiter 子开关被装饰器短路**：`rocm_aiter_ops` 的 `is_enabled/is_linear_enabled/is_mha_enabled/is_mla_enabled/`
   `is_fused_moe_enabled/is_triton_unified_attn_enabled` 全部带 `@if_aiter_supported`（`vllm/_aiter_ops.py:278-290`），
   装饰器在 `is_aiter_found_and_supported()` 为假时**直接返回 `None`** ⇒ 上面 6 行 `= None` 就是"即使把
   `VLLM_ROCM_USE_AITER=1`、`VLLM_ROCM_USE_AITER_MHA=1`… 全部打开，attention/linear/MoE 的 aiter 分支也拿不到真值"的运行期证明。
   `is_rdna_aiter_enabled()` 没有装饰器，但 `on_rdna4()==False`。
   **唯一的例外是 `is_aiter_found_and_supported()` 被直接使用的地方**：`rocm.py:592-593` 的
   `ROCM_AITER_UNIFIED_ATTN`，它同样吃到 `False`。

### 3.3 门禁 1：8K 上下文 `vllm serve`（端口 8002，**无 `--enforce-eager`**）

命令（可直接粘贴）：

```console
$ docker exec -d z100-s8 bash -lc '
    source /opt/dtk/env.sh >/dev/null 2>&1
    source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh
    cd $VLLM_DIR
    exec > /mnt/game/port/s8/s8-serve-8k.log 2>&1
    export VLLM_LOGGING_LEVEL=DEBUG
    exec python -c "import torch; from vllm.entrypoints.cli.main import main; main()" serve \
      /mnt/game/models/Qwen3-0.6B --dtype float16 --max-model-len 8192 \
      --gpu-memory-utilization 0.9 --port 8002 --host 0.0.0.0 \
      --served-model-name qwen3-0.6b-s8 --no-enable-log-requests'
```

关键原始行（长行截断）：

```console
DEBUG [platforms/rocm.py:857] Some attention backends are not valid for rocm with
  AttentionSelectorConfig(head_size=128, dtype=torch.float16, kv_cache_dtype=auto, block_size=None,
  use_mla=False, has_sink=False, use_sparse=False, ..., attn_type=decoder, has_sliding_window=False,
  use_non_causal=False, ..., use_kv_connector=False, ...). Reasons: {TURBOQUANT: [kv_cache_dtype not supported]}.
INFO  [platforms/rocm.py:895] Found incompatible backend(s) [TURBOQUANT] with AttentionType.DECODER.
  Overriding with TRITON_ATTN out of potential backends: ['TRITON_ATTN', 'ROCM_ATTN'].      <-- 门禁 1
rocblaslt warning: No paths matched /opt/dtk/lib/hipblaslt/library/*gfx906*co.               <-- 门禁 2(b)
  Make sure that HIPBLASLT_TENSILE_LIBPATH is set correctly.
INFO  [v1/attention/backends/utils.py:320] Using LBNHC KV cache layout.
INFO  [v1/worker/gpu/model_runner.py:406] Model loading took 1.12 GiB memory and 2.825363 seconds
INFO  [v1/worker/gpu_worker.py:719] Available KV cache memory: 26.77 GiB
INFO  [v1/core/kv_cache_utils.py:2432] GPU KV cache size: 250,608 tokens,
  Maximum concurrency for 8,192 tokens per request: 30.59x
INFO  [v1/worker/gpu/model_runner.py:1060] Graph capturing finished in 8 secs, took 0.36 GiB
INFO  [entrypoints/.../api_server/entry.py:136] Starting vLLM server on http://0.0.0.0:8002
INFO:     Application startup complete.
```

启动配置里 `enforce_eager=False`、`compilation_config.mode=VLLM_COMPILE`、`cudagraph_mode=FULL_AND_PIECEWISE`
（与硬约束 13 一致：**没有**用 `--enforce-eager`）。

**请求原文**（容器内 `127.0.0.1:8002`，避免依赖 bridge IP）：

```console
$ python - <<'PY'   # 容器内
PROMPT_TOKENS= 7801 CHARS= 24700
ELAPSED_S= 3.43
FINISH= length
USAGE= {'prompt_tokens': 7801, 'total_tokens': 7833, 'completion_tokens': 32, ...}
TEXT= ' The Z100 accelerator is a Hygon DCU built on the Vega20 gfx906 ISA, which has no matrix cores. The Z'
CHAT= "<think>\nOkay, the user is asking for the capital of France in a single short sentence. Let me recall. France's capital is Paris. I need to make sure it's a one-sentence answer without any extra information. Let me"
```

⇒ **8K 上下文（7801 prompt token）请求成功、数值合理**（不是 `lock×N` 那种 NaN 退化），且 attention 后端是 `TRITON_ATTN`。

**后端真的跑到 Triton 内核**（`jit_monitor` 原文，证明运行时确实 JIT 出并执行了 unified attention）：

```console
WARNING [utils/jit_monitor.py:140] Triton kernel JIT compilation during inference:
  kernel_unified_attention_2d. This causes a latency spike; consider extending warmup to cover this shape/config.
```

**没有** `ROCM_ATTN` 被选中的任何痕迹：全日志 `grep -aiE "aiter|skinny|wvsplit|LLMM1|flash|hipblas|rocblas"` 的命中只有
`_aiter_ops.py:174` 的 HIP runtime 警告、环境变量 dump（`VLLM_ROCM_USE_AITER': False`、`VLLM_ROCM_USE_SKINNY_GEMM': False`）、
`FlashInfer unavailable since package was not found` 和上面那行 `rocblaslt warning`。

### 3.4 门禁 2（b/c/d）hipBLASLt / skinny GEMM / flash-attn

**(b) hipBLASLt：镜像里只有 gfx928，gfx906 上必然落空（原文 + 静态）**

```console
$ ls /opt/dtk/lib/hipblaslt/library/
extop_gfx928.co  hipblasltExtOpLibrary.dat  hipblasltTransform.hsaco
Kernels.so-000-gfx928.hsaco  TensileLibrary_gfx928.co  TensileLibrary_gfx928.dat  TensileManifest.txt
```

```console
# 运行期（本步两次独立进程各出现一次，均在第一次 GEMM/handle 初始化时）
rocblaslt warning: No paths matched /opt/dtk/lib/hipblaslt/library/*gfx906*co.
  Make sure that HIPBLASLT_TENSILE_LIBPATH is set correctly.
```

⇒ DTK **没有** gfx906 的 hipBLASLt 内核（设计文档 §必须关 表格第 4 行核实）。
**结论：不需要额外开关**——该 .so 目录里没有 gfx906 目标码，rocBLASLt 自己会 warn 并退回 rocBLAS Tensile（`env.sh` 已记录的
"不要导出 `ROCBLAS_TENSILE_LIBPATH`" 是关键，见 `tools/dcu/env.sh:100-112`）。
唯一会主动选 hipBLASLt 的是 aiter 的 `AiterHipbMMPerTokenFp8ScaledMMLinearKernel`（`is_supported` 要求
`VLLM_ROCM_USE_AITER=1 + _LINEAR=1 + _LINEAR_HIPBMM=1`，默认 `_LINEAR_HIPBMM=False`，见
`vllm/envs.py:1270-1280`），而 aiter 在 §3.2 已三重关闭 ⇒ 该路径进不去。

**(c) skinny GEMM：算子根本没编进 `_rocm_C`（原文）**

```console
--- skinny GEMM / _rocm_C ---
ROCM_SKINNY_GEMM_ENABLED = False
VLLM_ROCM_USE_SKINNY_GEMM (env read) = False
_rocm_C ops = ['_rocm_C::paged_attention']
```

⇒ `_rocm_C` 的注册面**只剩 `paged_attention`**，`LLMM1/wvSplitK/wvSplitKQ/wvSplitKrc/wvSplitK_int4_g` 一个都不在
（与 S5 的处置一致：`csrc/rocm/skinny_gemms*.cu` 已按 arch 排除）。
判据位置：`vllm/_custom_ops.py:20-27` 的 `ROCM_SKINNY_GEMM_ENABLED = hasattr(torch.ops._rocm_C, "LLMM1")`（运行期探针，非 arch 分支）；
`vllm/model_executor/kernels/linear/scaled_mm/rocm.py:96-97` 的 `ROCmFP8ScaledMMLinearKernel.is_supported()` 明确
`return False, "wvSplitKQ (skinny GEMM) is not built on gfx906 (no MFMA)"`；`tools/dcu/env.sh:113-115` 兜底导出
`VLLM_ROCM_USE_SKINNY_GEMM=0`。
**注意**：`vllm/model_executor/kernels/linear/scaled_mm/rocm.py:33` 的 `rocm_per_tensor_float_w8a8_scaled_mm_impl`
里**没有** gfx906 早退，M ≤ 4 时会直接调 `ops.wvSplitKQ`；它之所以安全，是因为调用方（`ROCmFP8ScaledMMLinearKernel`）
已在选择期被否决——**属"靠上游门控、自身不设防"，列为残余风险**（§5.4）。

**(d) flash-attn：`fa_utils.py` 的 gfx906 早退生效（原文）**

```console
WARNING [fa_utils.py:25] flash-attn is not available on gfx906 ROCm. Using alternative attention backend.
flash_attn_triton_available() = False
is_flash_attn_varlen_func_available() = False
v1 is_flash_attn_varlen_func_available() = False
```

判据位置：`vllm/attention/utils/fa_utils.py:22-27`（`if on_gfx906(): logger.warning(...); return False`）。

### 3.5 必须开：attention 后端优先级与 ViT backend（原文）

```console
_get_backend_priorities(use_mla=False) = ['TRITON_ATTN', 'ROCM_ATTN', 'TURBOQUANT']
_get_backend_priorities(use_mla=True) = ['TRITON_MLA']
use_rocm_custom_paged_attention(fp16,h64,blk16,gqa4,8k)  = False
use_rocm_custom_paged_attention(fp16,h128,blk16,gqa4,8k) = False
vit_attn_backend(h64,torch.float16)  = TRITON_ATTN
vit_attn_backend(h128,torch.bfloat16) = TRITON_ATTN
vit_attn_backend(h256,torch.float16) = TRITON_ATTN
INFO [rocm.py:958] Using Flash Attention (Triton backend) for ViT model on CDNA.
```

结论链：
- `rocm.py:602-604` 的 `if on_gfx906(): backends.remove(TRITON_ATTN); backends.insert(0, TRITON_ATTN)` **生效**
  （返回顺序里 `TRITON_ATTN` 在 `ROCM_ATTN` 之前）；
- `rocm.py:500-501` 的 `if on_gfx906(): return False` 让 `use_rocm_custom_paged_attention()` 恒假
  （两个 head_size 都实测 `False`）⇒ 即使落到 `ROCM_ATTN` 也不会进 MFMA-only 的 `_rocm_C::paged_attention`；
- ViT（多模态）attention 走 `rocm.py:950-961` 的 Triton 分支，实测 `TRITON_ATTN`。

### 3.6 量化路径核对

#### 3.6.1 W8A8 INT8（生产在用）：走 `TritonInt8ScaledMMLinearKernel`，sdot4 实测可用

选择面（`s8_quant_probe.py` 原文）：

```console
=== 1. W8A8 INT8 (production scheme) ===
_POSSIBLE_INT8_KERNELS[ROCM] = ['AiterInt8ScaledMMLinearKernel', 'TritonInt8ScaledMMLinearKernel']
  is_supported(AiterInt8ScaledMMLinearKernel) -> (False, 'requires `aiter` to be installed.')
  is_supported(TritonInt8ScaledMMLinearKernel) -> (True, None)
INFO [__init__.py:798] Selected TritonInt8ScaledMMLinearKernel for probe.pt
init_int8_linear_kernel[per-tensor sym (channelwise=False, static=True)] -> TritonInt8ScaledMMLinearKernel
init_int8_linear_kernel[per-channel sym (channelwise=True, static=True)] -> TritonInt8ScaledMMLinearKernel
init_int8_linear_kernel[per-tensor sym, dynamic act] -> TritonInt8ScaledMMLinearKernel
init_int8_linear_kernel[per-tensor asym input] -> TritonInt8ScaledMMLinearKernel
```

**真机端到端选择证据**（生产 27B INT8 模型的真实加载日志，见 §3.9）：

```console
INFO [__init__.py:798] Selected TritonInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
```

数值实测（`tools/dcu/s8_int8_gfx906_check.py`，自研 sdot4 路径）：

```console
torch dist  = 2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7 hip=6.3.26113
triton dist = 3.6.0+git7c56a5e4 / /mnt/game/triton-gfx906/install/triton/__init__.py
device      = gfx906:sramecc+:xnack-
OK   M=  16 K=  512 N=  256 per_token=True per_channel=True rel_err=2.746e-04
OK   M= 128 K= 1024 N=  512 per_token=True per_channel=True rel_err=4.070e-04
OK   M= 256 K= 2048 N= 1024 per_token=True per_channel=True rel_err=3.062e-04
OK   M=   1 K= 1024 N=  512 per_token=True per_channel=True rel_err=3.131e-04
OK   M=  64 K=  512 N=  256 per_token=False per_channel=True rel_err=3.614e-04
OK   M=  64 K=  512 N=  256 per_token=True per_channel=False rel_err=3.091e-04
INT8_SCALED_MM_OK
```

S6 自编 triton 的 `supportsVDot()/v_dot4` 复核（`tools/dcu/probe_triton.py`，本步复跑）：

```console
check 3: int8 dot (supportsVDot / v_dot4 path)
max_abs_err = 0
INT8DOT_OK
v_dot occurrences in amdgcn asm = 32
VDOT_IN_ASM = True
PROBE_RC = 0
```

**与生产 13–14 ms/token 的对比：未执行**。原因：生产基线是 27B 多卡 W8A8（见 §3.9，单卡装不下），
模型/栈/并行度都不同，本步单卡约束下无法构造可比口径 ⇒ 属 S9 的性能对拍范围。

#### 3.6.2 W4A16（AWQ / GPTQ / compressed-tensors wNa16）：**修复前必崩，修复后落到 `Gfx906GPTQWNA16LinearKernel`**

**修复前**（`/mnt/game/port/s8/s8-quant-probe.log`，节选；完整见 §3.7）：

```console
_POSSIBLE_KERNELS[ROCM] = ['Gfx906GPTQWNA16LinearKernel', 'RDNA3W4A16LinearKernel',
  'RDNAHybridW4A16LinearKernel', 'TritonW4A16LinearKernel', 'ConchLinearKernel', 'ExllamaLinearKernel']
  can_implement(Gfx906GPTQWNA16LinearKernel, uint4b8 g128) -> RAISED AttributeError:
    'MPLinearLayerConfig' object has no attribute 'has_g_idx'
    | vllm/model_executor/kernels/linear/mixed_precision/gfx906_gptq_wna16.py:37, in can_implement
    |     if c.has_g_idx:
  can_implement(RDNA3W4A16LinearKernel, uint4b8 g128) -> (False, 'RDNA3 W4A16 kernel requires gfx1100')
  can_implement(RDNAHybridW4A16LinearKernel, uint4b8 g128) -> (False, 'RDNAHybridW4A16LinearKernel only targets gfx11/gfx12')
  can_implement(TritonW4A16LinearKernel, uint4b8 g128) -> (False, 'Triton W4A16 does not compile for ROCm gfx906')
  can_implement(ConchLinearKernel, uint4b8 g128) -> (False, 'Conch Triton kernels do not compile for ROCm gfx906')
  can_implement(ExllamaLinearKernel, uint4b8 g128) -> (False, 'Exllama is disabled on ROCm gfx906')
choose_mp_linear_kernel(uint4b8, g128, zp=False) [AWQ/GPTQ common] -> RAISED AttributeError: ...
choose_mp_linear_kernel(uint4b8, g128, zp=True)  [AWQ with zero points] -> RAISED AttributeError: ...
choose_mp_linear_kernel(uint8b128, g128)         [W8A16]                  -> RAISED AttributeError: ...
MPLinearLayerConfig fields = ['act_type', 'full_weight_shape', 'group_size', 'out_type',
                              'partition_weight_shape', 'weight_type', 'zero_points']      <-- 没有 has_g_idx
```

**修复后**（`/mnt/game/port/s8/s8-quant-probe-AFTER.log` 原文）：

```console
  can_implement(Gfx906GPTQWNA16LinearKernel, uint4b8 g128) -> (True, None)
  can_implement(TritonW4A16LinearKernel, uint4b8 g128) -> (False, 'Triton W4A16 does not compile for ROCm gfx906')
  can_implement(ConchLinearKernel, uint4b8 g128) -> (False, 'Conch Triton kernels do not compile for ROCm gfx906')
  can_implement(ExllamaLinearKernel, uint4b8 g128) -> (False, 'Exllama is disabled on ROCm gfx906')
choose_mp_linear_kernel(uint4b8, g128, zp=False)  [AWQ/GPTQ common] -> Gfx906GPTQWNA16LinearKernel
choose_mp_linear_kernel(uint4b8, g128, zp=True)   [AWQ with zero points] -> Gfx906GPTQWNA16LinearKernel
choose_mp_linear_kernel(uint8b128, g128)          [W8A16] -> Gfx906GPTQWNA16LinearKernel
MPLinearLayerConfig fields = ['act_type', 'full_weight_shape', 'group_size', 'has_g_idx', 'out_type',
                              'partition_weight_shape', 'weight_type', 'zero_points']
extra attrs passed by callers: has_g_idx=True
choose_mp_linear_kernel(uint4b8, g128, has_g_idx=True) [desc_act/actorder]
  -> RAISED ValueError: Failed to find a kernel that can implement the WNA16 linear layer. Reasons:
```

（最后一条是**期望行为**：`desc_act`/`actorder=group` 的权重在 gfx906 上没有支持的 kernel，
应当是"明确报错"而不是静默算错；报错原因串里含 `Activation reordering is not supported`。）

g_idx 存取通路自检（修复后，`=== 2b. kernel plumbing ===` 原文）：

```console
Gfx906GPTQWNA16LinearKernel.w_gidx_name before packing = None
_get_weight_params(fake) shapes = (128, 8), (8, 256), (8, 32)
_get_gidx_param(fake) after w_gidx_name='g_idx' = True
_get_gidx_param(fake) with w_gidx_name=None = None
```

**修复的独立复现（仓库自带测试，`tests/kernels/quantization/test_w4a16_kernel_selection.py`）**

```console
# 修复前：/mnt/game/port/s8/s8-pytest-w4a16-sel-BEFORE.log
>       if c.has_g_idx:
E       AttributeError: 'MPLinearLayerConfig' object has no attribute 'has_g_idx'
vllm/model_executor/kernels/linear/mixed_precision/gfx906_gptq_wna16.py:37: AttributeError
FAILED tests/kernels/quantization/test_w4a16_kernel_selection.py::test_choose_mp_linear_kernel_uint4b8
FAILED tests/kernels/quantization/test_w4a16_kernel_selection.py::test_choose_mp_linear_kernel_uint4_asymmetric
======================== 2 failed, 17 warnings in 6.74s =========================
BEFORE_RC=1

# 修复后：/mnt/game/port/s8/s8-pytest-w4a16-sel-AFTER.log
E       AssertionError: assert 'Gfx906GPTQWNA16LinearKernel' == 'TritonW4A16LinearKernel'
E       AssertionError: assert 'Gfx906GPTQWNA16LinearKernel' == 'TritonW4A16LinearKernel'
FAILED tests/kernels/quantization/test_w4a16_kernel_selection.py::test_choose_mp_linear_kernel_uint4b8
FAILED tests/kernels/quantization/test_w4a16_kernel_selection.py::test_choose_mp_linear_kernel_uint4_asymmetric
======================== 2 failed, 17 warnings in 6.79s =========================
AFTER_RC=1
```

⇒ 崩溃（AttributeError）已消除；剩下的是**该测试自身的期望值过期**：它的 `_expected_rocm_kernel()`
（`test_w4a16_kernel_selection.py:32-40`）只认 gfx1100/gfx11x，gfx906 上期望 `TritonW4A16LinearKernel`
——而 `TritonW4A16LinearKernel.can_implement` 自己就拒绝 gfx906（`triton_w4a16.py:206-207`），
所以这个测试在 gfx906 上**无论 vLLM 怎么写都不可能过**。测试文件不在本步写作用域内，**未改动**，列为遗留项（§5.2）。

底层算子可用性（`gptq_shuffle` / `gptq_gemm` 是 `Gfx906GPTQWNA16LinearKernel.apply_weights` 唯一依赖的两个 C++ op）：

```console
$ ... pytest -v --no-header --noconftest tests/kernels/quantization/test_gptq.py
tests/kernels/quantization/test_gptq.py::test_gptq_shuffle_opcheck PASSED [ 50%]
tests/kernels/quantization/test_gptq.py::test_gptq_gemm_opcheck PASSED   [100%]
======================== 2 passed, 17 warnings in 20.84s ========================
GPTQ_OPS_RC=0
```

**W4A16 端到端（真模型推理）：未执行**，原因见 §5.3（主机上没有本地 W4A16 权重，HF 下载反复中断）。
本步给出的是"kernel 选择 + g_idx 通路 + 底层 op"三级证据，**没有**跑出真实 W4A16 的文本输出。

#### 3.6.3 FP8：`is_supported` 全部为假，且是"明确报错"

```console
=== 3. FP8 (documented as unsupported on gfx906) ===
_POSSIBLE_FP8*[per-tensor/per-channel][ROCM] =
  is_supported(AiterHipbMMPerTokenFp8ScaledMMLinearKernel) -> (False, 'requires setting `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_LINEAR=1`, and `VLLM_ROCM_USE_AITER_LINEAR_HIPBMM=1`.')
  is_supported(AiterPreshuffledPerTokenFp8ScaledMMLinearKernel) -> (False, 'requires setting `VLLM_ROCM_USE_AITER=1` and `VLLM_ROCM_USE_AITER_LINEAR=1`. ...')
  is_supported(AiterPerTokenFp8ScaledMMLinearKernel) -> (False, 'requires setting `VLLM_ROCM_USE_AITER=1` and `VLLM_ROCM_USE_AITER_LINEAR=1`. ...')
  is_supported(ROCmFP8ScaledMMLinearKernel) -> (False, 'wvSplitKQ (skinny GEMM) is not built on gfx906 (no MFMA)')
  is_supported(PerTensorTorchFP8ScaledMMLinearKernel) -> (False, 'requires a platform with torch FP8 scaled-MM support.')
  is_supported(RowWiseTorchFP8ScaledMMLinearKernel) -> (False, 'requires platform with torch FP8 scaled-MM support.')
  is_supported(ChannelWiseTorchFP8ScaledMMLinearKernel) -> (False, 'requires platform with torch FP8 scaled-MM support.')
_POSSIBLE_FP8*[block-scaled][ROCM] =
  is_supported(AiterFp8BlockScaledMMKernel) -> (False, 'Only supported on ROCm platform with aiter package installed.')
  is_supported(TritonFp8BlockScaledMMKernel) -> (True, None)
```

⇒ per-tensor/per-channel FP8 在 gfx906 上**全体否决**（torch `_scaled_mm` 的 FP8 在 DTK/gfx906 上不支持）。
**但 block-scaled FP8 有一个漏网之鱼**：`TritonFp8BlockScaledMMKernel.is_supported()` 返回 `True`
（纯 triton 实现，不查 arch）⇒ 如果用户拿 block-FP8 权重来，vLLM 会选它并在 gfx906 出码运行。
gfx906 无 FP8 单元，这条路径**没有 arch 门控**，属**必须显式避开的路径**（禁用清单第 6 行 + §5.4）。

#### 3.6.4 GGUF（阶段 B）：门控存在，本步只做静态核对

```console
=== 5. GGUF (phase B, static check only) ===
_gguf_gfx906_small_batch_mmvq_enabled() = True
_gguf_gfx906_mmvq_max_batch()  = 8
_gguf_gfx906_q4_mmvq_max_batch() = 16
_gguf_gfx906_q6_mmvq_max_batch() = 16
```

判据位置：`vllm/model_executor/layers/quantization/gguf.py:115-143`（`ENABLE_GGUF_GFX906_SMALL_BATCH_MMVQ` +
`on_gfx906()`）、`:959-965`（按量化类型取 MMVQ 上限）、`:2077/:2185`（Qwen3.5 linear-attn gfx906 回退）；
`vllm/config/vllm.py:256-358/1726/1986/2029`（gfx906 GGUF 的 async-scheduling 与 PIECEWISE graph 策略）。
**未执行**实机 GGUF 推理（设计文档明确划到 S10 阶段 B）。

#### 3.6.5 MoE + GPTQ-Int4 / AWQ-4bit（设计文档标注"已知不可用"）：核对到具体门控

```console
=== 6. MoE + GPTQ/AWQ (MoE WNA16) ===
moe_wna16.on_gfx906 = True
MoE WNA16 Torch fallback trigger: on_gfx906() and weight_bits not in (4,8) or group_size not in (32,64,128)
  -> vllm/model_executor/layers/quantization/moe_wna16.py:506-521
```

- `moe_wna16.py:415-427` 的 `_apply_gfx906_torch_fallback()`：gfx906 上 Triton GPTQ/AWQ 融合 MoE 内核编不出来时，
  回落 Torch 版（会打 `[gfx906] Falling back to Torch MoE WNA16 path ...` 警告），**并且该回退不支持
  `apply_router_weight_on_input`**（`:428-432` raise `NotImplementedError`）。
- `moe_wna16.py:506-521` 只在 `weight_bits ∈ {4,8}` 且 `group_size ∈ {32,64,128}` 时**不**回退，否则走 Torch 回退。
- **新发现（阻塞级）**：MoE WNA16 路径在**任何**带 `quantization_config` 的模型上都会先在
  `_verify_quantization` 里因 `override_quantization_method()` 签名过期而 `TypeError`（§3.7.2）——即在讨论
  "MoE + GPTQ-Int4 能不能用"之前，模型根本过不了量化校验。

#### 3.6.6 量化路径结论表

| 路线 | gfx906 上走哪条（实测/静态） | 是否可用 | 不可用时是否明确报错 | 依据 |
|---|---|---|---|---|
| W8A8 INT8（compressed-tensors，生产格式） | `TritonInt8ScaledMMLinearKernel` → `triton_scaled_mm`（`v_dot4`） | ✅ **可用**（6 组形状 rel_err ≤ 4.1e-04；27B 真机选择日志同名） | — | §3.6.1 |
| W8A16 / W4A16（compressed-tensors wNa16 / wNa8 / wNa4、quark w4a16、awq_marlin） | `Gfx906GPTQWNA16LinearKernel`（`gptq_shuffle`+`gptq_gemm`）| ✅ 修复后**可选且可加载**；端到端未实测 | `desc_act`/`actorder=group` ⇒ `ValueError … Activation reordering is not supported` | §3.6.2 |
| AWQ（`AWQLinearMethod`，非 marlin） | 走 **`awq_triton`**（`_use_gfx906_awq_triton()==True`），不进 MPLinearKernel | ⚠️ 未实测 | `awq_gemm`/`awq_dequantize` 两个 C++ op **不存在**于 `_C`，但该分支不用它们 | §3.6.7 |
| GPTQ（`GPTQConfig`→`GPTQLinearMethod`） | exllama/GPTQ 分支（`gptq.py`），**不**走 `choose_mp_linear_kernel` | ⚠️ 未实测 | gfx906 上 exllama 明确禁用（`mixed_precision/exllama.py:36-37`） | §3.6.7 |
| MoE + GPTQ-Int4 / AWQ-4bit | MoE WNA16 → Torch 回退（`moe_wna16.py:415`）| ⛔ **不支持**（设计文档一致）；且受 §3.7.2 阻塞 | Torch 回退对 `apply_router_weight_on_input` 抛 `NotImplementedError` | §3.6.5 |
| FP8 per-tensor / per-channel | 无 | ⛔ 不可用 | ✅ 全部 `is_supported=False`（`ROCmFP8ScaledMMLinearKernel` 给出 gfx906 明确理由） | §3.6.3 |
| **FP8 block-scaled** | `TritonFp8BlockScaledMMKernel` | ⛔ **必须显式避开**（无 arch 门控） | ❌ `is_supported()` 返回 `True`，会静默选中 | §3.6.3 |
| GGUF（Q4_K/Q6_K/IQ4_XS 等） | gfx906 专用列布局 + MMVQ 小 batch 门控 | ⚠️ 阶段 B（S10）；门控已在 | — | §3.6.4 |

#### 3.6.7 AWQ / GPTQ 具体分叉（静态，`file:line`）

- **AWQ**：`vllm/model_executor/layers/quantization/awq.py:36-49` 定义 `_use_gfx906_awq_triton()` /
  `_use_gfx906_mobydick_awq()`；本机 `_use_gfx906_awq_triton() = True`、`_use_gfx906_mobydick_awq() = False`
  （`VLLM_ROCM_USE_GFX906_MOBYDICK_AWQ=False`，`vllm/envs.py:1388-1390`）。`awq.py:125-130` 打
  `[gfx906] Falling back to Triton AWQ kernels instead of exllama/GPTQ-compatible AWQ path.`；
  `awq.py:311-312`（`process_weights_after_loading` 在 gfx906 直接 `return`，**不做** `awq.py:318-319` 的 qzeros/qweight shuffle），
  `apply` 走 `awq_gemm_triton`（`awq.py:330-352`）。⇒ AWQ 在 gfx906 上是**独立 triton 路线**，与 `awq_gemm` C++ op 无关
  （实测 `hasattr(torch.ops._C,'awq_gemm')=False`，与设计一致）。
- **GPTQ**：`gptq.py:174-191` 的 `get_quant_method()` 返回 `GPTQLinearMethod`（exllama 风格），
  **不经过** `choose_mp_linear_kernel`；`exllama` 在 gfx906 被 `mixed_precision/exllama.py:36-37` 明确禁用，
  所以 GPTQ 在本 fork 上实际落到 `gptq.py:228+` 的 GPTQ/exllama 自实现——**本步未实测**（无 GPTQ 权重，见 §5.3）。
- `AutoAWQMarlinLinearMethod`（`auto_awq.py:391-452`）与 `AutoGPTQLinearMethod`（`auto_gptq.py:301-431`）
  都会调 `choose_mp_linear_kernel`，因此**同吃 §3.7.1 的 AttributeError**（修复后落到 `Gfx906GPTQWNA16LinearKernel`）。
  注意 `AutoGPTQLinearMethod` 在当前 fork 里**没有被任何 `get_quant_method()` 接线**（只有 `openpangu_vl.py:891`
  用 `AutoGPTQConfig` 做类型判断），属 dead path ⇒ 其 `has_g_idx` 保持默认 `False`（本步未改 `auto_gptq.py`）。

### 3.7 修复前后的日志对比

#### 3.7.1 修复 A（`has_g_idx` / g_idx 通路）

见 §3.6.2 的"修复前 / 修复后"两段原文，以及 §3.6.2 末尾的 pytest BEFORE/AFTER。
补充量化口径：**修复前 = 2 failed（AttributeError）**，**修复后 = 2 failed（期望值过期，不再是崩溃）**；
`choose_mp_linear_kernel()` 的返回值从"抛异常"变成 `Gfx906GPTQWNA16LinearKernel`。

#### 3.7.2 修复 B（`moe_wna16` 签名）：生产 27B INT8 模型的真实日志对比

```console
# 修复前（/mnt/game/port/s8/s8-serve-int8-27b.log 第一次运行）
File "/mnt/game/port/gfx906-vllm-211/vllm/config/model.py", line 1570, in _verify_quantization
    quantization_override = method.override_quantization_method(
TypeError: MoeWNA16Config.override_quantization_method() got an unexpected keyword argument 'hf_config'

# 修复后（同一命令重跑，长行截断）
INFO [__init__.py:798] Selected TritonInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
INFO [v1/worker/gpu/model_runner.py:406] Model loading took 28.51 GiB memory and 38.250413 seconds
```

⇒ 修复 B 之前，**任何**带 `quantization_config` 的模型（不止 MoE）都过不了 `_verify_quantization`；
修复后压缩张量 W8A8 模型能完成量化校验、kernel 选择与权重加载（后续卡在编译期，见 §3.9）。

### 3.8 W4A16 端到端尝试与放弃（如实记录）

主机上没有本地 W4A16 权重；唯一可用的是仓库自带 e2e 测试用的
`nm-testing/tinyllama-oneshot-w4a16-group128-v2`（compressed-tensors W4A16，group_size=128，symmetric int4，
`recipe.yaml` 与 `config.json.quantization_config` 均已核对，正好命中 `CompressedTensorsWNA16`）。

```console
# 容器内 HF 可达
$ curl -sS -o /dev/null -w "hf=%{http_code}\n" https://huggingface.co/api/models/nm-testing/tinyllama-oneshot-w4a16-group128-v2
hf=200
$ curl -sS ".../api/models/...?blobs=true" | grep size
model.safetensors 761968600          # 727 MiB

# 但下载反复中断（三次尝试，全部失败）
Fetching 8 files:  12%|█▎ | 1/8 [00:06<00:42,  6.04s/it]      # snapshot_download 停在 66 MB 不再前进（>9 min）
$ curl -L --retry 3 -C - -o model.safetensors <resolve/main URL>
curl: (92) HTTP/2 stream 0 was not closed cleanly: CANCEL (err 8)   # 8m34s，仅 319 MB
$ curl -L --http1.1 --retry 5 --retry-all-errors -C - ...            # 第 3 次：CDN 报 Content-Length=407 MiB，
                                                                     # 与实际 727 MiB 不符，产物必然截断
```

⇒ 判定为"网络不可靠"，**主动放弃**，未把两处修复"端到端证明"到位。已在 §5.3 明确列为未完成项。

### 3.9 生产 27B W8A8 INT8 单卡尝试（未通过，含挂死记录）

模型：`/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8`（`Qwen3_5ForConditionalGeneration`，
`compression_config.quant_method=compressed-tensors`，`language_model_only: true`，18 个 shard，safetensors 合计 **28.76 GiB**）。

命令：端口 8003、`--dtype auto --max-model-len 4096 --gpu-memory-utilization 0.97`（单卡）。

进展（原始行）：

```console
INFO [core.py:129] Initializing a V1 LLM engine (vdev) with config: model='/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8',
  ..., dtype=torch.bfloat16, max_seq_len=4096, tensor_parallel_size=1, ..., quantization=compressed-tensors,
  enforce_eager=False, ...
INFO [__init__.py:798] Selected TritonInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
INFO [v1/worker/gpu/model_runner.py:406] Model loading took 28.51 GiB memory and 38.250413 seconds
INFO [compilation/backends.py:1151] Dynamo bytecode transform time: 27.77 s
<日志到此为止，时间戳 2026-10-08 21:25:41>
```

**挂死判定与处置**（硬约束"出现挂死立即停止并如实记录"）：

```console
$ ps -o pid,time,pcpu,rss -p 6453          # 21:40:01
    PID     TIME %CPU   RSS
   6453 00:02:03 12.4 5654520
$ sleep 30; ps -o pid,time,pcpu,rss -p 6453   # 30 秒后
    PID     TIME %CPU   RSS
   6453 00:02:04 12.1 5654508
$ stat -c "%y" /mnt/game/port/s8/s8-serve-int8-27b.log
2026-10-08 21:25:41.820871295 +0800          # 距判定时刻已 ~15 分钟无任何日志
$ cat /proc/loadavg
1.29 1.08 1.10 6/629 8080                   # 系统 load 正常
$ python -c "import torch; print(round(torch.cuda.mem_get_info(1)[0]/2**30,2))"
dev1_free_GiB 0.0                            # 卡 1 被占满（31.98 GiB）
$ kill -9 6453; sleep 5; ... ; dev1_free_GiB 31.73   # 干净退出，无不可杀进程、无 load 飙升
```

⇒ **判定为 `torch.compile`（inductor）阶段停摆**：CPU 30 s 只涨 1 s、日志 15 分钟无进展、GPU 显存占满。
根因不是 gfx906 后端选错（量化校验/内核选择/权重加载都过了），而是
**28.51 GiB 权重 / 31.98 GiB 单卡 + `--memory=10g` 容器**下编译资源不足（容器 cgroup `memory.events max=199612`、
`oom_kill=0`）。**未观察到 GPU 驱动死锁**；进程 `kill -9` 后立即释放显存。
单卡 27B W8A8 端到端 ⇒ 交 S9（生产为多卡）。

### 3.10 复跑 S6 探针 + 门禁 3 数值（原文）

```console
$ python tools/dcu/probe_triton.py
check 1: elementwise add (lowering accepts gfx906)   max_abs_err = 0.0   TRITON_OK
check 2: warp-level reduce (TargetInfo::warpReduce / DPP path)  max_abs_err = 0.0  REDUCE_OK
check 3: int8 dot (supportsVDot / v_dot4 path)       max_abs_err = 0     INT8DOT_OK
v_dot occurrences in amdgcn asm = 32
VDOT_IN_ASM = True
PROBE_RC = 0
```

### 3.11 `gfx906` 专用 op 面（"必须开的路径"第 3 行，原文）

```console
hasattr(torch.ops._C, 'rms_norm_gated_gfx906') = True
hasattr(torch.ops._C, 'gemma_rms_norm_gfx906') = True
hasattr(torch.ops._C, 'gemma_fused_add_rms_norm_gfx906') = True
hasattr(torch.ops._C, 'causal_conv1d_gfx906_decode_update') = True
hasattr(torch.ops._C, 'causal_conv1d_gfx906_mtp_update') = True
hasattr(torch.ops._C, 'fused_sigmoid_gating_delta_rule_gfx906_decode') = True
hasattr(torch.ops._C, 'fused_sigmoid_gating_delta_rule_gfx906_prefill') = True
```

⇒ GDN / causal_conv1d 的 gfx906 内核（`csrc/mamba/gfx906_decode_kernels.cu`）已注册在 `_C` 里。
**未执行**实机 GDN 推理（`Qwen3.5-4B` 的 `linear_attention` 层）：时间预算用在了 §3.9 的 27B 尝试上，
列为遗留项（§5.3）。

### 3.12 ⚠️ 顺带发现：gfx906 的 bf16 设备探针**不稳定**（两次独立运行结果相反）

```console
# 21:00 那次（s8_path_probe.py 首次运行，本次会话）
INFO [hygon_dtk.py:398] bf16 probe passed on this device (round-trip 0.00780, gemm 0.12494, softmax 0.00024).
gfx906_bf16_enabled() = True

# 21:36 那次（同一脚本、同一卡、同一容器，日志落盘版本）
WARNING [hygon_dtk.py:345] bf16 probe failed on this device
  (RuntimeError: CUDA error: HIPBLAS_STATUS_ALLOC_FAILED when calling `hipblasCreate(handle)`);
  treating bf16 as unsupported. Set VLLM_DCU_BF16=1 to force it on.
gfx906_bf16_enabled() = False
```

⇒ `_rocm_supported_dtypes()` 会把 bf16 插进 `[fp16, fp32]`（`vllm/platforms/rocm.py:286-291`），
所以 **`--dtype auto` 在不同次运行里可能解析成 fp16 或 bf16**；
`hipblasCreate` 的 `ALLOC_FAILED` 不是 arch 问题，但会让 dtype 选择非确定。S9 必须显式传 `--dtype`。

### 3.13 资源与自检

```console
$ （收工后）4 卡空闲
['dev0=31.73', 'dev1=31.73', 'dev2=31.73', 'dev3=31.73']
$ docker exec z100-s8 cat /sys/fs/cgroup/memory.current ; cat /sys/fs/cgroup/memory.events
3684585472            # 3.43 GiB / 上限 10 GiB
low 0 / high 0 / max 199612 / oom 0 / oom_kill 0
$ df -h /dev/shm | tail -1
tmpfs            16G  4.5M   16G   1% /dev/shm
$ df -h / /mnt/game
overlay                          938G  293G  646G  32% /          # 容器层（含 2.1 GB vllm 编译缓存）
/dev/nvme1n1p1                   938G  293G  646G  32% /mnt/game
$ docker ps -a --format '{{.Names}}\t{{.Status}}'
z100-eager      Up 33 minutes                # 另一子 agent（卡 2），未动
z100-s8         Up 34 minutes                # 本步新建（未 --rm，保留）
z100-serve      Up 2 hours                   # S7 容器，未动
z100-triton     Up 14 hours                  # 未动
z100-build-211  Up 15 hours                  # 未动（只读参照）
z100-build      Up 21 hours                  # 未动
qwen38-vllm     Exited (137) 14 hours ago    # 生产容器，全程只 docker ps/inspect/cp 只读观察
```

- **未执行任何自编译 HIP 独立可执行程序**：本步只用 `pip` / `python`（vLLM+triton+torch JIT）/ `pytest` / `curl` / `docker exec`。
- 全程无 GPU 驱动死锁、无 loadavg 飙升；唯一的 `kill -9` 是 §3.9 的停摆进程（见该节证据），杀后显存立即释放。
- 生产容器 `qwen38-vllm` 全程只做 `docker ps` / `docker inspect` / `docker cp`（只读取出 `compressed_tensors-0.14.0.1.dist-info/METADATA`），
  **未 start / kill / exec / commit**。
- `z100-build` / `z100-build-211` / `z100-triton` / `z100-serve` 内的构建树未做任何写入。

---

## 4. 门禁判定

| # | 门禁（原文） | 判定 | 证据指向 |
|---|---|---|---|
| 1 | 8K 上下文请求成功且日志为 `TRITON_ATTN` | ✅ **通过** | §3.3：`rocm.py:895 Overriding with TRITON_ATTN out of potential backends: ['TRITON_ATTN','ROCM_ATTN']`；`--max-model-len 8192`、7801-token 请求成功、`GPU KV cache size: 250,608 tokens / 30.59x`、`kernel_unified_attention_2d` JIT 于推理期 |
| 2a | 日志中不出现 **aiter** 被选中 | ✅ **通过** | §3.2：`import aiter` 直接 `ImportError: ... undefined symbol: hsa_ext_log_finish_alloc`；`is_aiter_found_and_supported()=False`（`get_cdna_version()==0`）；6 个 `is_*_enabled()` 被 `@if_aiter_supported` 短路成 `None`；8K 日志只有 `_aiter_ops.py:174` 警告与 `VLLM_ROCM_USE_AITER: False` 的 env dump |
| 2b | 不出现 **hipBLASLt** 被选中 | ✅ **通过** | §3.4(b)：`ls /opt/dtk/lib/hipblaslt/library/` 只有 `*gfx928*`；运行期 `rocblaslt warning: No paths matched /opt/dtk/lib/hipblaslt/library/*gfx906*co`；唯一主动入口 `AiterHipbMM...` 被 2a + `_LINEAR_HIPBMM=False` 关闭 |
| 2c | 不出现 **skinny GEMM** 被选中 | ✅ **通过** | §3.4(c)：`ROCM_SKINNY_GEMM_ENABLED=False`、`_rocm_C ops = ['_rocm_C::paged_attention']`（`LLMM1/wvSplitK*` 全无）、8K 日志只有 `VLLM_ROCM_USE_SKINNY_GEMM: False` |
| 2d | 不出现 **flash-attn** 被选中 | ✅ **通过** | §3.4(d)：`fa_utils.py:25 flash-attn is not available on gfx906 ROCm`；`is_flash_attn_varlen_func_available()=False`（v0/v1 两条入口都 False）；ViT 也走 `TRITON_ATTN` |
| 3 | W8A8 INT8 用自研 sdot4 跑通一次，并记录与生产基线差距 | ✅ **通过（口径：数值跑通）**；基线对比 ⏸️ 移交 S9 | §3.6.1：`init_int8_linear_kernel → TritonInt8ScaledMMLinearKernel`（4 种配置）、6 组形状 `rel_err ≤ 4.070e-04`、`INT8DOT_OK` + `v_dot=32`、27B 真机 `Selected TritonInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8`。**未做**生产 13–14 ms/token 对比（单卡跑不了 27B，§3.9/§5.3） |
| 4 | `on_gfx906()/on_cdna()/on_gfx9()` 真机为 `True True False` | ✅ **通过** | §3.1：`on_gfx906 = True  on_cdna = True  on_gfx9 = False` |
| 5 | 量化路径逐项结论 + 不可用时明确报错 | ✅ **通过** | §3.6.6 结论表；FP8 per-tensor/per-channel 与 desc_act/W4A16 均"明确报错"，**block-FP8 例外**（无门控，已入禁用清单） |
| 6 | 产出禁用清单并给出代码位置 | ✅ **通过** | §6.1 |
| 7 | 新建 `z100-s8`（同镜像、参数照抄、`--memory=10g --shm-size=16g --ipc=host`、不加 `--rm`、挂 `/mnt/game` + `/src:ro`）＋ 按 S7 清单补装环境 | ✅ **通过** | §2.3 `docker inspect` 原文；§2.4 依赖补装（S7 的 6 个包 + 本步第 7 个 `compressed-tensors==0.15.0.1`）＋ `usercustomize.py` shim |
| 8 | 固定单卡、**不用 `--enforce-eager`** | ✅ **通过** | §2.2 `s8_env.sh` 无条件设 `HIP_VISIBLE_DEVICES=1`；§3.3 起的 8K 服务命令与启动配置里 `enforce_eager=False`、`mode=VLLM_COMPILE`；全部实验只用卡 1 |
| 9 | 硬约束：禁 HIP 可执行程序 / 不碰生产容器 / 不动构建树 / 不替换 DTK 库 / 缓存落 `/mnt/game` / 不改禁改文档 | ⚠️ **基本遵守，1 处偏差** | §3.13；**偏差**：`torch.compile` 默认把 2.1 GB 编译缓存写进了容器可写层（`$HOME/.cache/vllm`），已在 `s8_env.sh` 加 `VLLM_CACHE_ROOT` 修正；偏差原因=该变量默认值在 vLLM 内部，`s8_env.sh` 首版未覆盖 |
| 10 | 发现误选/误判路径时做最小修复 + 给出 `file:line` + 前后对比 | ✅ **通过** | §2.1（2 处修复、逐处论证）、§3.7（两组前后对比）、§3.6.2（pytest BEFORE/AFTER） |
| 11 | 门禁 1–13 之外：`docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md` 未改 | ✅ **通过** | `git status --short` 只含 `vllm/**`（2 处修复）+ `tools/dcu/s8_*` + 本归档 |
| 12 | 出现挂死立即停止并如实记录 | ✅ **通过** | §3.9：判定依据（15 min 无日志 / 30 s 只涨 1 s CPU / loadavg 1.29 / 显存占满）+ `kill -9` 后显存立即回到 31.73 GiB |

**S8 总体：✅ 门禁 1、2(a–d)、3(数值口径)、4、5、6、7、8、10、11、12 通过；
⏸️ 门禁 3 的"与生产基线对比"与 27B / W4A16 两个端到端实测移交 S9（原因见 §5）。
⚠️ 门禁 9 有 1 处缓存落盘偏差，已修正并如实记录。**

---

## 5. 未完成项 / 阻塞与原因

### 5.1 本步修掉的两个阻塞级缺陷（已修，但需要 Lead 复核）

1. **A：`MPLinearLayerConfig` 缺 `has_g_idx` + g_idx 存取通路**（`MPLinearKernel.py` 与 5 个 kernel 调用点）。
   影响：gfx906 上**所有** W4A16/W8A16（compressed-tensors wNa16/wNa8/wNa4、quark w4a16、awq_marlin、
   AutoAWQ/GPTQ marlin 方法）在 **kernel 选择期**就 `AttributeError`；即使绕过，首个 forward 也会
   `ValueError: not enough values to unpack`。**同一缺陷对非 gfx906 的 ROCm 也有影响**
   （`exllama`/`conch`/`triton_w4a16` 在 gfx942 等 arch 上会被选中，然后同样 4↔3 元组错配）。
2. **B：`MoeWNA16Config.override_quantization_method()` 少 `hf_config` 形参**（`moe_wna16.py:149`）。
   影响：**任何**带 `quantization_config` 的模型（不论是否 MoE）在 `ModelConfig._verify_quantization()`
   里必然 `TypeError` ⇒ 本 fork 在当前代码下**加载不了任何量化模型**（含生产的 compressed-tensors W8A8）。
   与 `compressed-tensors` 升级（§2.4）**必须同时具备**才能把量化模型带起来。

> 两处都在 `vllm/**` 写作用域内、都做了"最小 + 前后对比"；但它们**超出了"路径核对"的字面范围**，
> 属"核对过程中发现的误选/崩溃路径"。Lead 若认为风险不可接受，回滚点分别是
> `git checkout -- vllm/model_executor/kernels/linear/mixed_precision/` 与 `git checkout -- vllm/model_executor/layers/quantization/moe_wna16.py`。

### 5.2 `tests/kernels/quantization/test_w4a16_kernel_selection.py` 的期望值过期（**未修，不在写作用域**）

该测试的 `_expected_rocm_kernel()`（第 32-40 行）不知道 gfx906 的 `Gfx906GPTQWNA16LinearKernel` 优先级，
在 gfx906 上期望 `TritonW4A16LinearKernel`——而后者自己就拒绝 gfx906。
⇒ 该文件在 gfx906 上**结构性不可过**。建议 Lead 让 S9 或一个专门的小步骤把
`_expected_rocm_kernel()` 补上 `if on_gfx906(): return "Gfx906GPTQWNA16LinearKernel"`（含 `uint4` 非对称分支的
`has_g_idx`/group_size 语义），并补一个 `has_g_idx=True ⇒ ValueError` 的用例。

### 5.3 明确"未执行"的项

| 项 | 原因 |
|---|---|
| **W4A16 端到端真模型推理** | 主机无本地 W4A16 权重；唯一的仓库测试模型 `nm-testing/tinyllama-oneshot-w4a16-group128-v2`（727 MiB）从 HF 下载 3 次全部中断（`HTTP/2 stream ... CANCEL`、CDN `Content-Length` 与实际不符），见 §3.8。已给出 kernel 选择 / g_idx 通路 / `gptq_gemm`+`gptq_shuffle` op 三级证据替代 |
| **生产 27B W8A8 INT8 端到端**（含文本输出、ms/token、与 13–14 ms/token 基线对比） | 单卡约束（任务书指定卡 1）下权重 28.51 GiB / 卡 31.98 GiB，`torch.compile` 阶段停摆（§3.9），已按硬约束停止。生产为多卡，交 S9 |
| **GDN / causal_conv1d gfx906 内核实机推理** | 仅完成注册面核对（§3.11，7 个 op 全在）；`Qwen3.5-4B` 的 `linear_attention` 层实机跑没做，时间预算用在了 27B 尝试。交 S9 |
| **AWQ / GPTQ 真模型** | 同 W4A16（无本地权重）；仅完成静态分叉核对（§3.6.7） |
| **GGUF 实机** | 设计文档划到 S10 阶段 B；仅静态核对门控（§3.6.4） |
| **`hy-smi` 口径的显存对比** | 与 S8 门禁无关；S7 §5.3 已记录 `hy-smi` 在本宿主读 `mkfd` 失败 |
| **`--enforce-eager` 任何对照** | 硬约束 13（S7 已证 gfx906 上 logits 全 NaN）；本步无一处使用 |

### 5.4 残余风险（本步发现但未处置）

1. **block-scaled FP8 无 arch 门控**：`TritonFp8BlockScaledMMKernel.is_supported()` 在 gfx906 返回 `True`
   ⇒ 用户拿 block-FP8 权重能静默进入。要求：**禁用清单显式禁止 `--quantization fp8*` 的 block 变体**。
2. **`rocm_per_tensor_float_w8a8_scaled_mm_impl` 自身不设防**（`scaled_mm/rocm.py:27-41`）：
   M ≤ 4 时无条件调 `ops.wvSplitKQ`，靠调用方（`ROCmFP8ScaledMMLinearKernel`）被否决来兜底。
   若将来有人把该 impl 直接注册成自定义 op 并绕过 kernel 选择，会在 gfx906 上踩 S5 已证实的 device assert。
3. **bf16 设备探针不稳定**（§3.12）：`--dtype auto` 可能在 fp16/bf16 间摇摆。
4. **`compressed-tensors` 版本硬依赖未固化进仓库**：`requirements/dcu*.txt` 没有这个包，
   镜像自带 0.14.0.1 不够（§2.4）。建议 S9 把它写进 DTK 依赖清单。
5. **torch.compile 缓存落点**：`VLLM_CACHE_ROOT` 若不显式设置会写容器层（本步 2.1 GB）。

---

## 6. 对后续步骤的影响

### 6.1 禁用清单（gfx906 + DTK 26.04 上必须显式关闭 / 绕开）

| # | 后端 / 路径 | 要求 | 判据在代码里的位置（`file:line`） | 本步运行期证据 |
|---|---|---|---|---|
| 1 | **attention：`ROCM_ATTN`（含 `_rocm_C::paged_attention`）** | 不显式指定 `--attention-backend`；让自动选择落到 `TRITON_ATTN` | `vllm/platforms/rocm.py:602-604`（`on_gfx906()` 把 `TRITON_ATTN` 提到首位）；`rocm.py:500-501`（`use_rocm_custom_paged_attention()` 在 gfx906 恒 `False`）；`csrc/rocm/attention.cu:3293-3307`（MFMA-only `UNREACHABLE_CODE`） | §3.3 `Overriding with TRITON_ATTN`；§3.5 两个 head_size 均 `False` |
| 2 | **`VLLM_ROCM_USE_SKINNY_GEMM`** | 保持 `0`（`tools/dcu/env.sh:113-115` 已兜底导出） | `vllm/_custom_ops.py:20-27`（`ROCM_SKINNY_GEMM_ENABLED` 运行期探测）；`scaled_mm/rocm.py:96-97`；`scaled_mm/rocm.py:103-104` | §3.4(c) `ROCM_SKINNY_GEMM_ENABLED=False`、`_rocm_C ops=['_rocm_C::paged_attention']` |
| 3 | **`VLLM_ROCM_USE_AITER*`（全部子开关）** | 保持默认（全 `0`/不设）；**不要**为了"提速"打开 | `vllm/_aiter_ops.py:202-218`（`is_aiter_found_and_supported()` 要求 `get_cdna_version()>2`）；`_aiter_ops.py:278-290`（`@if_aiter_supported` 装饰器）；`envs.py:1258-1280` | §3.2（`import aiter` 失败 + 6 个 `is_*_enabled()==None`） |
| 4 | **FlashAttention（`flash_attn_2_cuda.so` / `FLASH_ATTENTION_TRITON_AMD_ENABLE`）** | 不用；不要设 `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` | `vllm/attention/utils/fa_utils.py:22-27`（gfx906 早退）；`rocm.py:529-560`（`flash_attn_triton_available()` 要求 `on_gfx1x()`） | §3.4(d) |
| 5 | **hipBLASLt / MIOpen 的 gfx906 路径** | 无需开关（DTK 根本没有 gfx906 kernel）；**不要**导出 `ROCBLAS_TENSILE_LIBPATH` | `tools/dcu/env.sh:100-112`（导出即任意 GEMM 崩的实测结论）；镜像 `/opt/dtk/lib/hipblaslt/library/` 只有 `*gfx928*` | §3.4(b) `rocblaslt warning: No paths matched ... *gfx906*co` |
| 6 | **FP8（`--quantization fp8*` / `VLLM_ROCM_FP8_*`），尤其 block-scaled** | **显式禁止**；per-tensor/per-channel 会被 arch 否决，但 **block-scaled 会静默进 `TritonFp8BlockScaledMMKernel`** | `scaled_mm/rocm.py:96-97`（否决 ROCm 路径）；torch FP8 kernel 的 `is_supported()` 要求平台支持；**`TritonFp8BlockScaledMMKernel.is_supported()` 无 arch 门控** | §3.6.3（per-tensor 全 False；block-scaled `-> (True, None)`） |
| 7 | **`--enforce-eager`（`CompilationMode.NONE`）** | **禁止**（硬约束 13）；用默认 `VLLM_COMPILE`（graph 是否捕获都行） | `vllm/platforms/rocm.py:1362-1388`（`get_default_ir_op_priority` 两条路径分流点） | S7 §3.7 的 9 组 A/B；本步全部实验 `enforce_eager=False` |
| 8 | **`--dtype auto`** | **不推荐**；显式 `--dtype float16` | `vllm/platforms/rocm.py:286-291`（bf16 依赖设备探针）；`hygon_dtk.py:329-345` | §3.12 同容器两次探针结果相反 |
| 9 | **vLLM 编译缓存落点** | 设 `VLLM_CACHE_ROOT=/mnt/game/...` | 本步偏差记录 §2.5 | §3.13 容器层 2.1 GB |
| 10 | **`compressed-tensors` < 0.15.0.1** | 必须先升到 `0.15.0.1`（`requirements/common.txt:40`） | `compressed_tensors_wNa8o8.py:14`（`from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32`） | §2.4 `ModuleNotFoundError` |
| 11 | **`tests/kernels/quantization/test_w4a16_kernel_selection.py` 在 gfx906 上** | 不要把它当门禁（期望值过期），见 §5.2 | 测试自身 `:32-40` | §3.6.2 AFTER 日志 |

### 6.2 S9 该用什么参数

```bash
# 容器：新建（或复用）时务必把 --memory 提到 >= 24g（27B 的 torch.compile 装不进 10g）
docker run -d --name z100-s9 \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    --group-add video --security-opt label=disable --ipc host \
    --memory=24g --memory-swap=24g --shm-size=16g \
    -v /opt/hyhal:/opt/hyhal:ro -v /home/lyh/z100-vllm:/src:ro -v /mnt/game:/mnt/game \
    -w /mnt/game/port \
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 \
    bash -c 'sleep infinity'

# 依赖：S7 的 6 个包 + 本步第 7 个
pip install --no-deps /mnt/game/port/wheels-211/torch-2.11.0+das.opt1.dtk2604.*.whl
pip install --no-deps /mnt/game/port/s7/pip-wheels/{mistral_common-1.11.5-py3-none-any.whl,openai-2.54.0-py3-none-any.whl,xgrammar-0.2.8-*.whl,apache_tvm_ffi-0.1.14.post1-*.whl,torchvision-0.26.0-*.whl}
pip install --no-deps /mnt/game/port/s8/pip-wheels/compressed_tensors-0.15.0.1-py3-none-any.whl   # 本步新增
cp /src/tools/dcu/s7_usercustomize.py /root/.local/lib/python3.10/site-packages/usercustomize.py

# 起服务（S8 已验证可用的形态；端口不要用 8000）
source /opt/dtk/env.sh
source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh     # 自带 VLLM_CACHE_ROOT 与卡 1
cd $VLLM_DIR
python -c "import torch; from vllm.entrypoints.cli.main import main; main()" serve \
  /mnt/game/models/Qwen3-0.6B --dtype float16 --max-model-len 8192 \
  --gpu-memory-utilization 0.9 --port 8002 --host 0.0.0.0 \
  --served-model-name qwen3-0.6b-s9 --no-enable-log-requests
#  ❌ 不要加 --enforce-eager          （硬约束 13）
#  ✅ --dtype 显式给出，不要用 auto   （§3.12）
#  ✅ 不要设 --attention-backend，让它自动选 TRITON_ATTN（§3.3）
#  ✅ 27B W8A8 类模型请用 TP>=2（生产口径）；单卡 27B 在 31.98 GiB 卡上不可行（§3.9）
```

需要 S9 补的门禁/实验：
1. 生产 27B W8A8 的多卡（TP≥2）端到端 + decode ms/token，与 13–14 ms/token 基线对比；
2. W4A16 端到端（网络恢复后跑 `tests/kernels/quantization/test_rocm_compressed_tensors_w4a16.py`，
   或先修好 §5.2 的选择测试）；
3. GDN / causal_conv1d gfx906 内核实机（`Qwen3.5-4B`）；
4. 设计文档门禁 4（4 卡 decode + 显存偏差 <5%，`torch.cuda.mem_get_info` 口径，S7 §5.3 已记录 `hy-smi` 不可用）。

### 6.3 被本步更新 / 推翻的前提

1. **推翻**"量化路径已经有 fork 的 gfx906 专用实现、抽查一下就行"：真机上
   **W4A16 在 kernel 选择期就崩**（`has_g_idx`）、**任何量化模型都过不了量化校验**（`hf_config` 签名）、
   **compressed-tensors 包版本不够**——三者叠加，"生产 W8A8 在单卡上跑通"在本步之前不可达。
2. **确认**设计文档 §必须关 表格的 6 行里，第 1/2/3/4/5 行（skinny / aiter / flash-attn / hipBLASLt / FP8-per-tensor）
   判据都成立；**第 6 行（CUDA graph 默认 `--enforce-eager`）已被 S7 推翻**，本步继续按"默认编译"执行且 8K 数值正确。
3. **新增**：FP8 **block-scaled** 是唯一没有 arch 门控的漏网路径（§5.4-1）。
4. **新增**：`--dtype auto` 在 gfx906 上不具确定性（§3.12）。
5. **新增**：镜像自带 `compressed-tensors 0.14.0.1` 不够（生产容器同样是 0.14.0.1，见 §2.4）。
6. **确认**设计文档"语义澄清"三条：`on_gfx9()` 在 gfx906 为 `False`（未把 gfx906 加进 `_ON_GFX9`）、
   `on_cdna()` 为 `True`（子串匹配，被 ViT 选择等场景预期使用）、`on_gfx906()` 与 `on_gfx9()` 组合使用。

---

## 7. 复现方式

```bash
# ---------- 1) 建容器（一次；与 z100-build-211 同镜像、同设备参数） ----------
docker run -d --name z100-s8 \
    --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
    --group-add video --security-opt label=disable --ipc host \
    --memory=10g --memory-swap=10g --shm-size=16g \
    -v /opt/hyhal:/opt/hyhal:ro -v /home/lyh/z100-vllm:/src:ro -v /mnt/game:/mnt/game \
    -w /mnt/game/port \
    harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 \
    bash -c 'sleep infinity'

# ---------- 2) 补装依赖（S7 的 6 个 + 本步第 7 个） + usercustomize shim ----------
docker exec z100-s8 bash -lc '
  set -e
  mkdir -p /mnt/game/port/s8/pip-wheels /mnt/game/port/s8
  W7=/mnt/game/port/s7/pip-wheels ; W8=/mnt/game/port/s8/pip-wheels
  pip install --no-deps /mnt/game/port/wheels-211/torch-2.11.0+das.opt1.dtk2604.*.whl
  pip install --no-deps $W7/mistral_common-1.11.5-py3-none-any.whl $W7/openai-2.54.0-py3-none-any.whl \
      $W7/xgrammar-0.2.8-*.whl $W7/apache_tvm_ffi-0.1.14.post1-*.whl $W7/torchvision-0.26.0-*.whl
  pip download --no-deps -d $W8 "compressed-tensors==0.15.0.1"
  pip install --no-deps $W8/compressed_tensors-0.15.0.1-py3-none-any.whl
  cp /src/tools/dcu/s7_usercustomize.py /root/.local/lib/python3.10/site-packages/usercustomize.py
  cp /src/tools/dcu/s8_env.sh /src/tools/dcu/s8_path_probe.py /src/tools/dcu/s8_quant_probe.py \
     /src/tools/dcu/s8_int8_gfx906_check.py /src/tools/dcu/probe_triton.py \
     /mnt/game/port/gfx906-vllm-211/tools/dcu/
  # 修复 A/B 同步进 fork 副本（若不打算用仓库改动，可跳过）
  cp /src/vllm/model_executor/kernels/linear/mixed_precision/{MPLinearKernel,gfx906_gptq_wna16,triton_w4a16,torch_wna16,exllama,conch}.py \
     /mnt/game/port/gfx906-vllm-211/vllm/model_executor/kernels/linear/mixed_precision/
  cp /src/vllm/model_executor/layers/quantization/moe_wna16.py \
     /mnt/game/port/gfx906-vllm-211/vllm/model_executor/layers/quantization/
  '

# ---------- 3) 通用环境（每次新 shell） ----------
docker exec -it z100-s8 bash
source /opt/dtk/env.sh
source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh    # → HIP_VISIBLE_DEVICES=1

# ---------- 4) 门禁 4 + 门禁 2：路径探针 ----------
cd $VLLM_DIR
python /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_path_probe.py | tee /mnt/game/port/s8/s8-path-probe.log
#   期望：on_gfx906=True on_cdna=True on_gfx9=False ; cdna_version=0
#         IS_AITER_FOUND=True 但 is_aiter_found_and_supported()=False；6 个 is_*_enabled()=None
#         ROCM_SKINNY_GEMM_ENABLED=False ; _rocm_C ops=['_rocm_C::paged_attention']
#         _get_backend_priorities(use_mla=False)=['TRITON_ATTN','ROCM_ATTN','TURBOQUANT']
#         use_rocm_custom_paged_attention(...)=False （两个 head_size）
#         flash-attn 不可用；vit_attn_backend=TRITON_ATTN；7 个 gfx906 op 全 True

# ---------- 5) 门禁 1：8K 服务（端口 8002，无 --enforce-eager） ----------
docker exec -d z100-s8 bash -lc '
  source /opt/dtk/env.sh >/dev/null 2>&1
  source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh
  cd $VLLM_DIR
  exec > /mnt/game/port/s8/s8-serve-8k.log 2>&1
  export VLLM_LOGGING_LEVEL=DEBUG
  exec python -c "import torch; from vllm.entrypoints.cli.main import main; main()" serve \
    /mnt/game/models/Qwen3-0.6B --dtype float16 --max-model-len 8192 \
    --gpu-memory-utilization 0.9 --port 8002 --host 0.0.0.0 \
    --served-model-name qwen3-0.6b-s8 --no-enable-log-requests'
for i in $(seq 1 60); do
  [ "$(docker exec z100-s8 bash -lc 'curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:8002/health')" = "200" ] && break
  sleep 10
done
docker exec z100-s8 bash -lc 'grep -aE "rocm.py:895|Available KV cache memory|GPU KV cache size|Starting vLLM server" \
    /mnt/game/port/s8/s8-serve-8k.log'
#   期望：Overriding with TRITON_ATTN out of potential backends: [\x27TRITON_ATTN\x27, \x27ROCM_ATTN\x27]
#         Available KV cache memory: 26.77 GiB ; GPU KV cache size: 250,608 tokens / 30.59x

# 8K 请求（7801 prompt token）
docker exec z100-s8 bash -lc 'python - <<"PY"
import json, urllib.request
def post(p,o):
    r=urllib.request.Request("http://127.0.0.1:8002"+p, data=json.dumps(o).encode(),
        headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read().decode())
prompt=("The Z100 accelerator is a Hygon DCU built on the Vega20 gfx906 ISA, which has no matrix cores. ")*260
print("TOKENS", len(post("/tokenize", {"model":"qwen3-0.6b-s8","prompt":prompt})["tokens"]))
out=post("/v1/completions", {"model":"qwen3-0.6b-s8","prompt":prompt,"max_tokens":32,"temperature":0})
print("USAGE", out["usage"]); print("TEXT", repr(out["choices"][0]["text"])[:200])
PY'
docker exec z100-s8 bash -lc 'grep -aE "jit_monitor.py:140" /mnt/game/port/s8/s8-serve-8k.log'
#   期望：Triton kernel JIT compilation during inference: kernel_unified_attention_2d
docker exec z100-s8 bash -lc 'pkill -f "entrypoints.cli.main"'   # 停服（容器保留）

# ---------- 6) 门禁 2 负向核对（在 8K 日志上） ----------
docker exec z100-s8 bash -lc '
  echo "--- aiter ---";     grep -aiE "aiter"      /mnt/game/port/s8/s8-serve-8k.log
  echo "--- skinny ---";    grep -aiE "skinny|wvsplit|LLMM1" /mnt/game/port/s8/s8-serve-8k.log
  echo "--- hipblaslt ---"; grep -aiE "hipblaslt|rocblaslt"   /mnt/game/port/s8/s8-serve-8k.log
  echo "--- flash ---";     grep -aiE "flash"     /mnt/game/port/s8/s8-serve-8k.log'
#   期望：aiter 只有 _aiter_ops.py:174 警告 + env dump（VLLM_ROCM_USE_AITER: False）
#         skinny 只有 env dump（VLLM_ROCM_USE_SKINNY_GEMM: False）
#         hipblaslt 只有 rocblaslt warning: No paths matched ...*gfx906*co
#         flash 只有 FlashInfer unavailable / env dump

# ---------- 7) 量化路径探针（修复前/后对比） ----------
docker exec z100-s8 bash -lc '
  source /opt/dtk/env.sh >/dev/null 2>&1; source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh
  cd $VLLM_DIR
  python /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_quant_probe.py'
#   期望：INT8 -> TritonInt8ScaledMMLinearKernel（4 种配置）
#         W4A16 -> Gfx906GPTQWNA16LinearKernel；has_g_idx=True -> ValueError
#         FP8 per-tensor 全 False；block-scaled TritonFp8BlockScaledMMKernel -> (True, None)

# ---------- 8) 修复 A 的回归（前后各跑一次即可看出差别） ----------
docker exec z100-s8 bash -lc '
  source /opt/dtk/env.sh >/dev/null 2>&1; source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh
  cd $VLLM_DIR
  python -c "import torch, pytest, sys; sys.exit(pytest.main([\"-v\",\"--no-header\",\"--noconftest\",\
\"-p\",\"no:cacheprovider\",\"tests/kernels/quantization/test_w4a16_kernel_selection.py\",\
\"tests/kernels/quantization/test_gptq.py\"]))"'
#   修复前：w4a16_kernel_selection 2 项 AttributeError
#   修复后：w4a16_kernel_selection 2 项因测试期望值过期而失败（§5.2）；test_gptq.py 2 passed

# ---------- 9) 门禁 3：W8A8 INT8 sdot4 数值 ----------
docker exec z100-s8 bash -lc '
  source /opt/dtk/env.sh >/dev/null 2>&1; source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh
  cd $VLLM_DIR
  python tools/dcu/probe_triton.py            # 期望 INT8DOT_OK + v_dot=32
  python tools/dcu/s8_int8_gfx906_check.py'   # 期望 INT8_SCALED_MM_OK（6 组形状）
#   实测：rel_err 2.746e-04 … 4.070e-04

# ---------- 10) （可选）生产 27B W8A8 单卡尝试；--memory=10g 会在编译期停摆（§3.9） ----------
docker exec -d z100-s8 bash -lc '
  source /opt/dtk/env.sh >/dev/null 2>&1; source /mnt/game/port/gfx906-vllm-211/tools/dcu/s8_env.sh
  cd $VLLM_DIR
  exec > /mnt/game/port/s8/s8-serve-int8-27b.log 2>&1
  exec python -c "import torch; from vllm.entrypoints.cli.main import main; main()" serve \
    /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 --dtype auto --max-model-len 4096 \
    --gpu-memory-utilization 0.97 --port 8003 --host 0.0.0.0 \
    --served-model-name qwen38-int8-s8 --no-enable-log-requests'
#   期望（--memory>=24g 时才会走完）：Selected TritonInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
#   --memory=10g 实测：Model loading took 28.51 GiB 后 torch.compile 停摆 ⇒ kill -9
```

---

## 8. 参考

- 设计文档：[docs/design/z100-08-attention-and-quant-paths.md](../design/z100-08-attention-and-quant-paths.md)
- 前序归档：
  [z100-07-memory-kvcache-graph-result.md](z100-07-memory-kvcache-graph-result.md)（服务起法 / 环境补装 / `usercustomize.py` / 禁 `--enforce-eager`）、
  [z100-06-triton-gfx906-result.md](z100-06-triton-gfx906-result.md) §5/§7（自编 triton 用法 + SFU hazard）、
  [z100-05-csrc-kernels-result.md](z100-05-csrc-kernels-result.md) §3.7/§4/§6（`_rocm_C` 注册面、`wvSplitK*`/`paged_attention` 的 gfx906 桩）、
  [z100-03-bf16-verification-result.md](z100-03-bf16-verification-result.md)（bf16 判定口径）
- 关键源码位置：
  - `vllm/platforms/rocm.py:255-272`（`_ON_CDNA`/`_ON_GFX9`/`_ON_GFX906` 与 `on_gfx906()`）、`:286-291`（`_rocm_supported_dtypes`）
  - `vllm/platforms/rocm.py:436-445`（`get_cdna_version()`）、`:478-526`（`use_rocm_custom_paged_attention` 的 gfx906 早退）
  - `vllm/platforms/rocm.py:563-606`（`_get_backend_priorities`，`:602-604` 是 TRITON_ATTN 提权点）、`:841-910`（后端选择与日志）
  - `vllm/platforms/rocm.py:921-961`（`get_vit_attn_backend`）
  - `vllm/_aiter_ops.py:112-122`（`IS_AITER_FOUND`）、`:202-218`（`is_aiter_found_and_supported`）、`:278-290`（`@if_aiter_supported`）、`:1973-2010/2090-2130`（各 `is_*_enabled`）
  - `vllm/_custom_ops.py:20-27`（`ROCM_SKINNY_GEMM_ENABLED`）、`:348-365/413-450/2791-2850`（gfx906 专用 op 包装）
  - `vllm/attention/utils/fa_utils.py:22-27`（flash-attn gfx906 早退）
  - `vllm/model_executor/kernels/linear/__init__.py:411-421`（`_POSSIBLE_INT8_KERNELS[ROCM]`）、`:503-530`（`_POSSIBLE_KERNELS[ROCM]`）、`:635-706`（`choose_scaled_mm_linear_kernel`）、`:780-806`（`init_int8_linear_kernel`）、`:817-890`（`choose_mp_linear_kernel`）
  - `vllm/model_executor/kernels/linear/scaled_mm/rocm.py:19-52`（`rocm_per_tensor_float_w8a8_scaled_mm_impl` 用 `wvSplitKQ`）、`:74-106`（`ROCmFP8ScaledMMLinearKernel.is_supported` 的 gfx906 否决）
  - `vllm/model_executor/kernels/linear/scaled_mm/aiter.py:37-58`（`AiterInt8ScaledMMLinearKernel.is_supported`）、`scaled_mm/triton.py:27-40`（`TritonInt8ScaledMMLinearKernel.is_supported`）
  - **本步修复**：`vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py:22-28/49-58/98-107`、
    `.../mixed_precision/{gfx906_gptq_wna16.py:122-123,exllama.py:157-158,triton_w4a16.py:278,conch.py:130,torch_wna16.py:51}`、
    `vllm/model_executor/layers/quantization/moe_wna16.py:148-155`
  - `vllm/model_executor/layers/quantization/compressed_tensors/schemes/__init__.py:15` + `compressed_tensors_wNa8o8.py:14`（`pack_quantized` 硬依赖）
  - `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:330-337`（actorder 明确拒绝 ⇒ `has_g_idx=False` 的语义依据）
  - `vllm/model_executor/layers/quantization/{awq.py:36-49/125-141/305-355, gptq.py:174-191, moe_wna16.py:415-432/506-521}`
  - `vllm/config/model.py:1560-1575`（`_verify_quantization` 的 override 探测循环，调用点）
  - `vllm/platforms/hygon_dtk.py:329-345`（bf16 设备探针）、`:408-415`（`gfx906_bf16_enabled`）
- 本步脚本（仓库）：[`tools/dcu/s8_env.sh`](../../tools/dcu/s8_env.sh)、
  [`tools/dcu/s8_path_probe.py`](../../tools/dcu/s8_path_probe.py)、
  [`tools/dcu/s8_quant_probe.py`](../../tools/dcu/s8_quant_probe.py)、
  [`tools/dcu/s8_int8_gfx906_check.py`](../../tools/dcu/s8_int8_gfx906_check.py)
- 本步原始日志（全部在 `/mnt/game/port/s8/`）：
  `s8-path-probe.log`、`s8-quant-probe.log`（修复前）、`s8-quant-probe-AFTER.log`（修复后）、
  `s8-serve-8k.log`（8K 门禁，1055+ 行，含 DEBUG）、`s8-serve-int8-27b.log`（27B 尝试 + 停摆）、
  `s8-pytest-w4a16-sel-BEFORE.log`、`s8-pytest-w4a16-sel-AFTER.log`、`s8-pytest-gptq-ops.log`、
  `s8-hf-download.log`、`s8-hf-curl.log`、`s8-hf-curl2.log`、`install-deps.log`、`pip-wheels/`
