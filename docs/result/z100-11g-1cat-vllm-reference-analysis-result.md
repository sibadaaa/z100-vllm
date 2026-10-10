# S11g 参考实现分析：1CatAI/1Cat-vLLM（V100/SM70）在无 FP8/FP4 硬件上"模拟新量化格式"的做法 — 执行结果

> 归档时间：2026-10-08（宿主 CST）
> 执行者：子 agent（S11g 参考实现分析）
> 对应设计文档：[docs/海光Z100-性能优化方案.md](../海光Z100-性能优化方案.md) §4C（7 条目标清单）
> 前一档：无（本条为"只读参考分析"，不改 `csrc/**`、`vllm/**`）
> 状态：✅ **7/7 条逐条给出 `file:line` + 原文摘录**；可迁移技术表 + P2/P5/P6 明确建议已给出
> 取源方式：**克隆成功**（Gitee/GitHub 直连），非 API 定向读取

**一句话结论**：1Cat-vLLM 把"老硬件跑新量化格式"当成**算子设计问题**而非 loader 问题，其核心配方只有三条：
① **存储保持压缩格式（FP8 / FP4 / GGUF 原始块），反量化在内核寄存器里"边读边转"**，从不整体物化 FP16 权重副本；
② **累加一律 FP32**（`mma.sync…f32.f16.f16.f32`、或 `__dp4a` 输出再乘 scale 之前保持 int32/FP32），**精度契约用"precision revision"版本号硬绑定**；
③ **按精确 shape（含 M）选 kernel + split-K**，而不是写一个通用 GEMM；并有 `capability → 选择 → fallback 原因` 的全链路诊断。
对本项目最直接的价值：**P6（FP16 模拟 FP8）应照抄"边读边转 + packed FP16 内积 + FP32 累加"的形态**，
**P2（W8A8 INT8 `sdot4`）应放弃"通用 GEMM"念头、改为 exact-shape 变小 N/split-K 改编排**，
**P5（冷启动）应照抄 vLLM 自带 torch.compile AOT 缓存的四个哈希因子与缓存目录约定**。

---

## 1. 目标与验收标准

### 1.1 任务书原文（§4C 的 7 条 = 本步门禁）

| # | 它的做法 | 对我们的映射 | 预期价值 |
|---|---|---|---|
| 1 | **FP8 E4M3/E5M2 只用于 KV 存储**，QK/PV 用 FP32 计算（明确写了一版 E4M3 修复需要 precision revision 4） | gfx906 上 `--kv-cache-dtype fp8`：**KV 流量减半**，attention 里反量化到 fp16/fp32 再算 | 长上下文 decode 直接受益（KV 带宽是 decode 大头） |
| 2 | **NVFP4 / MXFP4 权重在无 FP4 硬件上跑**（TurboMind SM70 / marlin_v100 路线），W4A16 反量化 + FP16/FP32 内积 | 与 P6（FP8→FP16）同一类问题：**边读边转 + packed FP16 内积 + FP32 累加** | 直接给 P6 的 kernel 形态做参考 |
| 3 | **GGUF 原始块算子**：能力探测 + fallback；IQ3 用 **FP32 内积 + split-K**，`M8/N4352/K5120` 有专用 shape | 我们 S10 的 GGUF + S11e 的 INT8 `sdot4`；`csrc/quantization/gguf/**` 已有 `__dp4a` 可借鉴 | 少走弯路 |
| 4 | **exact-shape decode GEMV**（按精确 shape 选 GEMV 而不是通用 GEMM） | 我们的 decode 是 M=1..8 —— 这正是 decode 的主要形态 | **可能比"重写 GEMM"更划算** |
| 5 | **编译产物复用**：首次启动 271 s → 同配置重启 95 s（编译子图复用） | 我们 27B 冷启动 ~29 min | 生产可重启形态（P5） |
| 6 | **内核选择诊断**：`vllm.sm70_profiles show`、`/v1/sm70/acceleration` 同时报告"配置的能力 / 实际加载的内核 / fallback 原因" | 我们的 `s8b_gate.sh` + `ROCm_SKINNY_GEMM_ENABLED` 可升级成 Z100 profile 报告 | 避免"配置了但没走到"的假象 |
| 7 | **精度/验收契约**：A/B/A token id + 文本 + SHA256 一致、PPL 偏差、`docs/design/sm70_*` 的 precision contract | 我们 S9/S11 的验收口径 | 让性能数字可被信任 |

附加要求：每条必须有 `file:line` 或 PR/文档引用 + 原始摘录；产出可迁移技术表 + P2/P5/P6 建议；
**只读参考，不把它的代码拷进本仓库**（Apache-2.0 允许适配，但必须保留出处/许可声明）。

### 1.2 本步验收标准

- 7 条逐条回答，每条 ≥1 个**真实存在的 `file:line` + 原文摘录**；引用行号经 grep 复核；
- 明确区分"代码里有的"与"只有文档说的"，以及"未找到/不确定"的项；
- 产出 §6 的 P2/P5/P6 + 诊断的**明确建议**；
- 不改 `csrc/**`、`vllm/**`、`docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`；
- 不 start/kill/exec 生产容器 `qwen38-vllm`；不动 `z100-build`；不运行自编译 HIP 可执行程序；
- 克隆物只落 `/mnt/game`。

### 1.3 硬约束遵守情况（逐条）

| 硬约束 | 遵守 | 证据 |
|---|---|---|
| 1 禁自编译 HIP 独立可执行程序 | ✅ | 本步**零编译、零 GPU 运行**；只用 `git`/`grep`/`sed`/`read` |
| 2 不 start/kill/exec 生产容器 `qwen38-vllm` | ✅ | 只在 `z100-a7`（非生产）+ 宿主只读操作；`qwen38-vllm` 未被触碰 |
| 3 克隆物只落 `/mnt/game`（根分区仅剩 ~42 GB） | ✅ | `/mnt/game/1cat-vllm`，190 MB（`--depth 1`）；根分区无新增 |
| 4 不替换 DTK 库 | ✅ | 未触碰任何 DTK 目录 |
| 5 不把 gfx906 加进 `_ON_GFX9` | ✅ | 本仓库 `vllm/**` 零改动 |
| 6 新增分支可被 `is_hygon_dtk()` 包裹 | N/A | 本步无代码改动 |
| 12 需自编 triton 才有 gfx906 码 | N/A | 无 Triton 运行 |
| 13 gfx906 禁 `--enforce-eager` | ✅ | 无 `serve` 命令 |
| 21 授权新增 HIP 算子但禁跑独立可执行程序 | ✅ | 未新增、未编译 |
| 只读参考、不拷代码进仓库 | ✅ | 本仓库除本归档外零改动（§2 已核） |
| 未改禁改文档 | ✅ | `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md` 零改动 |

---

## 2. 实际改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `docs/result/z100-11g-1cat-vllm-reference-analysis-result.md` | 新增（本档） | 唯一交付物；本仓库 `csrc/**`、`vllm/**` 零改动 |

**未落盘物（刻意）**：`/mnt/game/1cat-vllm`（190 MB，`--depth 1`，只读参考；属 `/mnt/game` 的取源副本，
按任务书"只读参考、不进仓库工作区"处理，未 `git add`、未拷入本仓库任何路径）。

`git status` 核对（本仓库）：

```text
$ git status --short docs/ csrc/ vllm/
?? docs/result/z100-11g-1cat-vllm-reference-analysis-result.md
```
（`csrc/`、`vllm/` 无输出 ⇒ 零改动。）

---

## 3. 执行过程与原始证据

### 3.0 取源与身份核对（克隆成功）

宿主对 `/mnt/game` **无写权限**（workspace-write 沙箱），改用容器 `z100-a7` 写 `/mnt/game`（该容器内可写）：

```bash
$ docker exec z100-a7 bash -lc 'cd /mnt/game && git clone --depth 1 \
    https://github.com/1CatAI/1Cat-vLLM.git 1cat-vllm'
Cloning into '1cat-vllm'...
CLONE_RC=0
190M    /mnt/game/1cat-vllm

$ docker exec z100-a7 bash -lc 'cd /mnt/game/1cat-vllm && git log -1 --format="%H %ad %s" \
    && git remote -v && git branch --show-current && git rev-parse HEAD'
c4f6245f841466782752a8c3283e4727565cf17a Thu Oct 8 08:50:20 2026 +0800 \
  [Kernel] Reuse M8 gate operands and TP4 norm packets (#1041)
origin  https://github.com/1CatAI/1Cat-vLLM.git (fetch)
origin  https://github.com/1CatAI/1Cat-vLLM.git (push)
main
c4f6245f841466782752a8c3283e4727565cf17a
```

**许可核对**：`/mnt/game/1cat-vllm/LICENSE:1` = `Apache License / Version 2.0, January 2004`。
源码文件头普遍带 SPDX：

```text
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```
（例：`vllm/sm70_profiles/acceleration.py:1-2`、`vllm/entrypoints/serve/sm70/api_router.py:1-2`）

**混杂许可（重要）**：其 GGUF 路径**内嵌了 llama.cpp（MIT）** 与 TensorRT-LLM 风格的 Skinny 内核：

```text
# csrc/sm70_turbomind/ops/gguf_dp4a.cuh:3-5
// Integer-dot formulas and Q8_1 layout follow ggml-org/llama.cpp
// ggml-cuda/vecdotq.cuh and quantize.cu (MIT; see packaged llama.cpp LICENSE).
```
```text
# docs/design/gguf_native_sm70.md:13-17
- llama.cpp is pinned to `002a12ad25503a93501b2e188c360029830a241a` (MIT).
- The Torch bridge starts from vllm-gguf-plugin PR 141 at
  `aa09d6522f29325d64d999d7d7c794f79836de07` (Apache-2.0).
- Unmodified upstream files, original bridge hashes, and local changes are
  recorded in `csrc/quantization/gguf_upstream/source_manifest.json`.
  Both licenses and binary provenance ship inside the wheel.
```
⇒ **许可注意（回填 §6）**：我们若适配其 **GGUF/dp4a** 相关实现，许可链条是 **Apache-2.0(1Cat 改动) + MIT(llama.cpp 原始公式)** 双层，
必须同时保留 `1CatAI/1Cat-vLLM` 出处与 llama.cpp MIT 声明；纯 SM70 QPN/NVFP4/TurboMind 部分只需 Apache-2.0 声明与出处。

---

### 3.1 【目标 1】FP8 KV cache：怎么存、怎么反量化、QK/PV 用什么精度、精度契约、实测收益

#### (a) 怎么存：KV cache 字节以 **uint8 张量** 承载 E4M3/E5M2，dtype 与"模型权重量化"解耦

`vllm/v1/attention/backends/flash_attn_v100.py:795-800`（CUDA graph workspace 规划按 KV dtype 分支）：

```text
    if kv_cache_dtype in ("fp8", "fp8_e4m3") and key_cache.dtype == torch.uint8:
        # Plan a p64 workspace envelope. The native G6 path keeps this captured
        # shape while selecting p64/p256 and long wave partitions from device
        # sequence lengths.
        return 64
    if kv_cache_dtype == "fp8_e5m2" and key_cache.dtype == torch.uint8:
        # Plan the largest p256 workspace once. ...
        return 256
```

dtype 解析 `flash_attn_v100.py:2868-2874`：

```text
def _fp8_dtype_from_cache_dtype(kv_cache_dtype: str) -> torch.dtype:
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    if kv_cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    raise ValueError(f"Unsupported FLASH_ATTN_V100 fp8 dtype: {kv_cache_dtype}")
```

**契约日志（明确写了 KV dtype 与权重量化无关）** `flash_attn_v100.py:813-832`：

```text
    if kv_cache_dtype == "fp8":
        logger.warning(
            "SM70 Flash-V100 received an unresolved `fp8` KV-cache dtype and "
            "will interpret it as upstream E4M3. Normal EngineArgs processing "
            "rewrites the SM70 `fp8` shorthand to `fp8_e4m3`; this warning "
            "usually means the backend was constructed directly. KV-cache "
            "dtype is independent of model weight quantization."
        )
    elif kv_cache_dtype == "fp8_e4m3":
        logger.info(
            "SM70 Flash-V100 is using explicitly requested E4M3 KV cache. "
            ...
```
并且 SM70 的 `fp8` 别名**默认解析为 E4M3**（`docs/design/sm70_dflash2_fp32_defaults.md:3-6`）：

```text
The SM70 Flash-V100 `fp8` KV alias now resolves to `fp8_e4m3`. Explicit
`fp8_e5m2` remains available for reproducing older deployments. Explicit
formats, checkpoint-resolved formats and non-SM70 backends are preserved.
This changes the FP8 alias, not the general `auto` cache policy or model weights.
```

#### (b) 怎么反量化：**一次向量化 gather+expand 到共享 FP16 页工作区**，然后交给**原本的 FP16 Tensor Core 内核**

prefill 侧（桥接算子）`docs/design/sm70_flash_v100_fp8_kv_long_context.md:82-96`：

```text
`fp8_e5m2_paged_kv_to_fp16` performs one vectorized gather/expansion pass:

- reads two E5M2 values at a time and expands them exactly into `half2`;
- resolves non-contiguous physical pages through the live block table;
- supports partial tail pages and non-unit K/V scales;
- writes into preallocated FP16 pages of 784 tokens;
- performs no allocation or host synchronization inside the CUDA op.
```
decode 侧（XQA 面板加载器内联反量化）`…fp8_kv_long_context.md:127-135`：

```text
- one aligned 8-byte global load reads eight E5M2 values;
- register bit expansion produces eight FP16 values for shared-memory staging;
- the existing HMMA/XQA schedule then serves six query heads from one KV read;
- `k_scale` is applied with the score scale and `v_scale` with partition output,
  matching the scalar path's scale placement.
```

**CUDA 实现（E5M2 位扩展确实在寄存器里做）** `flash-attention-v100/kernel/fp8_kv_bridge.cu:18-30`：

```text
template <int KV_DTYPE>
__device__ __forceinline__ __half2 load_bridge_half2(const void* cache,
                                                     int64_t offset) {
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2) {
    return flash_v100::load_fp8_e5m2_half2_unscaled(cache, offset);
  } else {
    const uint16_t pair = reinterpret_cast<const uint16_t*>(cache)[offset >> 1];
    return __float22half2_rn(make_float2(
        flash_v100::fp8_e4m3fn_to_float(static_cast<uint8_t>(pair)),
        flash_v100::fp8_e4m3fn_to_float(static_cast<uint8_t>(pair >> 8))));
  }
}
```
op 注册链：`flash-attention-v100/kernel/fused_mha_api.cpp:77-80` → `flash-attention-v100/kernel/fp8_kv_bridge.cu:231,241`
→ python `flash-attention-v100/flash_attn_v100/flash_attn_interface.py:1707,1739`（`fp8_e4m3_paged_kv_to_fp16` / `fp8_e5m2_paged_kv_to_fp16`）
→ `flash_attn_v100.py:2174-2185`（`_get_fp8_e5m2_paged_kv_bridge_op()`，`ImportError` 时返回 `None` ⇒ **缺 op 时是"降级/不可用"而非静默错误**）。

#### (c) QK/PV 精度：**明确 FP32 累加/部分和，FP16 只用于张量核操作数与最终输出**

`docs/design/sm70_dflash2_fp32_defaults.md:14-18`（最直接的一段）：

```text
The repaired computation retains compensated QK accumulation, compensated
probabilities, tile-local FP32 PV accumulation, FP32 unnormalized partition
numerators, separate FP32 max/sum, and FP32 combination. KV storage remains
E4M3; Tensor Core operands and the final activation remain FP16. This is not a
full-FP32 model and does not remove FP8 quantization error.
```
E4M3 XQA 路线 `…fp8_kv_long_context.md:452-456`：

```text
- E4M3 XQA keeps softmax probabilities, PV accumulation state, partition
  outputs, and the final partition reduction in FP32. The public output remains
  FP16, matching the model interface.
```
代码侧注释 `flash_attn_v100.py:5091-5094`：

```text
        # V100 FA2 kernels consume fp16 Q. FP8 KV cache support is implemented
        # as storage compression only, with K/V dequantized inside FA2 kernels.
        self.supports_quant_query_input = False
```
python 侧无前缀反量化辅助（供非原生路径用）`flash_attn_v100.py:2878+` `_dequantize_fp8_contiguous_kv(...)`。

#### (d) 精度契约文档：`docs/design/sm70_dflash2_fp32_defaults.md` + **precision revision 4**

`sm70_dflash2_fp32_defaults.md:20-25`：

```text
Precision capability revision **4** adds page sizes **1728/3456** to the existing
800/848/1616/1648/3296 runtime-stride implementation. The arithmetic is unchanged
from repaired revision 3. It also adds FP32 scalar E4M3 partial storage and
reduction. The wrapper requires revision 4 so a new page or FP32 workspace
cannot reach an incompatible native binary. Rebuild the extension and restart
workers together with this Python update.
```
**revision 不匹配 = 硬报错，不静默降级**：`sm70_dflash2_fp32_defaults.md:30-32`

```text
lacks revision 4, E4M3 scalar calls raise a rebuild error instead of silently
storing half partials.
```
代码侧同一思路（多处 `raise RuntimeError("Rebuild Flash-V100 for …")`）：
`flash-attention-v100/flash_attn_v100/flash_attn_interface.py:1186-1197`：

```text
    if not flash_attn_grouped_e4m3_fp32_available():
        raise RuntimeError(
            "Rebuild Flash-V100 for E4M3 grouped FP32 precision revision 4"
        )
    if (
        q.shape[1] != 6
        or k_cache.shape[1] not in (800, 848, 1616, 1648, 1728, 3296, 3456)
    ) and int(flash_attn_v100_cuda.grouped_e4m3_fp32_precision_version()) < 5:
        raise RuntimeError("Rebuild Flash-V100 for multi-head E4M3 revision 5")
    ...
        raise RuntimeError("Rebuild Flash-V100 for request-major E4M3 revision 6")
```
以及 `flash_attn_v100.py:5086-5091`（python 侧 warning_once）。文档契约 `sm70_dflash2_fp32_defaults.md:43-49`：

```text
## Observability and reproduction

A compatible worker reports `E4M3 grouped FP32 route selected` with its page,
query rows and FP32 state representation. Route summaries include
`prefill_smallq_e4m3_grouped_fp32` and `fp8_kv_decode_grouped_fp32`.
Check these alongside the resolved KV dtype and FP32-logit preparation; an image
name or requested flag is insufficient evidence of the numerical route.
```

#### (e) 实测收益（全模型，有对照契约）

**E5M2 原始问题（FP8 反而更慢）** `…fp8_kv_long_context.md:33-40`：

```text
| Context | FP16 prefill | Old FP8 prefill | FP16 TPOT | Old FP8 TPOT |
|---:|---:|---:|---:|---:|
| 16K | 10.098 s | 14.680 s | 19.532 ms | 21.762 ms |
| 64K | 60.328 s | 127.842 s | 26.014 ms | 32.768 ms |
| 128K | 169.206 s | 450.290 s | 35.218 ms | 47.119 ms |
```
**修好后的路线收益** `…fp8_kv_long_context.md:165-171`：

```text
| Context | Old FP8 prefill | New prefill | Prefill delta | Old FP8 TPOT | New TPOT | Decode tok/s delta |
|---:|---:|---:|---:|---:|---:|---:|
| 16K | 14.680 s | 10.317 s | -29.72% | 21.762 ms | 19.715 ms | +10.39% |
| 64K | 127.842 s | 62.964 s | -50.75% | 32.768 ms | 26.910 ms | +21.77% |
```
**注：他们最终把 FP8 KV 判定为"主要是 prefill 桥接收益，decode 侧 FP8 相对 FP16 仍有 +0.49%~+2.73% 的残差开销"**
（TP4 矩阵 `…fp8_kv_long_context.md:250-260`，例如 `27B AWQ 64K: FP16 TPOT 17.283 ms / FP8 17.458 ms = +1.02%`）。
⇒ **对我们（gfx906）的映射要诚实：KV 存储减半是真的，但"attention 里多一次反量化"的开销也是真的，收益期望应压在 prefill/长上下文带宽而非 decode 计算。**

**E4M3 decode 的算子上限收益（256K，CUDA graph）** `…fp8_kv_long_context.md:610-620`：

```text
| B1/Hq6/Hkv1/D256 | 800 | 256K | 3.2975 ms | 0.4834 ms | 6.82x | 0 |
| B1/Hq6/Hkv1/D256 | 1568 | 256K | 3.4051 ms | 0.4922 ms | 6.92x | `2.38e-7` |
| B32/Hq6/Hkv1/D256 | 800 | 256K | 96.4159 ms | 14.6154 ms | 6.60x | `4.77e-7` |
```
**注意他们的结论：这个 6.8× 是"XQA 相对 scalar"的收益，不是 FP8 相对 FP16 的收益**（`…:623-625` 明确"route counters"与"not a speedup"口径）。

**关键工程教训（可直接复用到我们的 KV 桥接设计）** `…fp8_kv_long_context.md:44-56`：

```text
NCU showed that this was not HBM saturation. The generic FP8 prefill kernel
used 96.64 KiB dynamic shared memory, admitted one CTA per SM, reached only
25% occupancy, and had about 4% tensor activity. It repeatedly converted the
same KV values for the many query CTAs.
```
以及 workspace 复用 `…:100-105`：

```text
The backend workspace is shared across attention layers and keyed by GPU,
CUDA stream, `Hkv`, and head dimension. At 256K and `Hkv=2,D=256`, one K/V
workspace costs about 512 MiB per rank. It is reused serially by all full
attention layers, rather than allocating one copy per layer. FP8 still saves
substantially more memory across the model than this single workspace costs.
```

---

### 3.2 【目标 2】NVFP4 / MXFP4 在无 FP4 硬件上：哪些 kernel、怎么组织、有没有"边读边转"

#### (a) 两条路线都存在，**生产路线是 TurboMind SM70 QPN，Marlin 是备选（不记为收益）**

- TurboMind 生产路线：`vllm/model_executor/kernels/linear/nvfp4/sm70.py`（`TurboMindNvFp4LinearKernel` / `Qpn2NvFp4LinearKernel`）
- Marlin 备选：`vllm/model_executor/kernels/linear/nvfp4/marlin.py`（`MarlinNvFp4LinearKernel`，显式 warning）
- 原生算子：`csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu`、`…/nvfp4_qpn4_sm70.cu`、`…/mxfp4_qpn_m1_sm70.cu`
- Marlin SM70 稠密/MoE：`csrc/quantization/marlin/sm70_marlin_nvfp4_gemm.cu`、`…/sm70_marlin_mxfp4_gemm.cu`、
  `csrc/moe/marlin_moe_wna16/sm70_marlin_nvfp4_gemm.cu`、`…/sm70_marlin_mxfp4_gemm.cu`

Marlin 的门槛与"没有 FP4 硬件"的自我说明 `vllm/model_executor/kernels/linear/nvfp4/marlin.py:26-40, 42-47`：

```text
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if is_fp4_marlin_supported():
            return True, None
        return False, "Marlin FP4 not available"
    ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logger.warning_once(
            "Your GPU does not have native support for FP4 computation but "
            "FP4 quantization is being used. Weight-only FP4 compression "
            "will be used leveraging the Marlin kernel. This may degrade "
            "performance for compute-heavy workloads."
        )
```
`is_fp4_marlin_supported()` 的 SM70 分支 `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:26-31`：

```text
def is_fp4_marlin_supported():
    if not current_platform.is_cuda():
        return False
    if current_platform.has_device_capability(75):
        return True
    return current_platform.is_device_capability((7, 0)) and ops.sm70_marlin_available()
```
TurboMind 的门槛（**只要 SM70/SM72 就支持 NVFP4**）`nvfp4/sm70.py:138-145`：

```text
    @classmethod
    def is_supported(cls, compute_capability=None):
        if compute_capability is None:
            capability = current_platform.get_device_capability()
            compute_capability = capability.to_int() if capability is not None else None
        if compute_capability not in (70, 72):
            return False, "requires SM70 or SM72 (Volta)"
        return True, None
```
**明确"生产不记 Marlin 功劳"** `docs/design/sm70_qwen38_nvfp4_decode.md:8-11`：

```text
- NVFP4 projections use TurboMind W4A16. No accepted result in
this document credits a Marlin fallback.
```
`docs/design/sm70_deepseek_v4_mxfp4_grouped_decode.md:7-8`：

```text
- Quantization backend: TurboMind only; Marlin is out of scope
```
MXFP4 在图上也是 TurboMind 原生（`csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu`），
E8M0 scale 的解码在 `csrc/quantization/marlin/sm70_marlin_mxfp4_gemm.cu:49-67`（`e8m0x2_to_half2_fast`：把 E8M0 的 8 位指数直接 **左移 23 位当作 FP32 位模式**）：

```text
__half2 e8m0x2_to_half2_fast(uint16_t e8m0_x2) {
  ...
  uint32_t const bits0 = static_cast<uint32_t>(e8m0_x2 & 0xFF) << 23;
  uint32_t const bits1 = static_cast<uint32_t>(e8m0_x2 >> 8) << 23;
  ...
void dequant_e8m0_scales_to_half2(int q, half2* scale_cache) {
  ...
  __half2 res0 = e8m0x2_to_half2_fast(lo_16);
  __half2 res1 = e8m0x2_to_half2_fast(hi_16);
```

#### (b) 反量化 + 内积怎么组织：**4-bit 码在寄存器里位扩展成 FP16 → `mma.sync…f32.f16.f16.f32`（FP16 输入 / FP32 累加）+ split-K + 多行复用**

**E2M1×8 位扩展成 FP16（纯位运算，无 LUT 查表访存）** `csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu:143-163`：

```text
__device__ __forceinline__ void dequant_e2m1x8(unsigned packed, half2 scale,
                                               half2 output[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x0e000e00u;
  unsigned values[4];
  values[0] = ((packed << 12) & kSign) | ((packed << 9) & kExponentMantissa);
  values[1] = ((packed << 8) & kSign) | ((packed << 5) & kExponentMantissa);
  values[2] = ((packed << 4) & kSign) | ((packed << 1) & kExponentMantissa);
  values[3] = (packed & kSign) | ((packed >> 3) & kExponentMantissa);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    // Undo the FP4 exponent bias on the code, not on its scale. Scaling
    // the scale by 2^14 first loses subnormals and can overflow even when the
    // final dequantized weights are finite.
    const half2 code = __hmul2(*reinterpret_cast<half2*>(&values[index]),
                               __float2half2_rn(16384.0f));
    output[index] = __hmul2(code, scale);
  }
}
```
**累积是 FP32（MMA 指令自带 f32 累加器）** `nvfp4_qpn2_sm70.cu:188-193`：

```text
#define VLLM_SM70_QPN2_MMA(C, A0, A1, B0, B1)                       \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))
```
**accum 数组是 float（每线程 `[RowTiles][NAcc][8]`）** `nvfp4_qpn2_sm70.cu:588` 附近：

```text
  float accum[2][8] = {};
```
**split-K 选择是显式常量表（按 (K,N,gated) 定 split 与 accumulator 链数）** `nvfp4/sm70.py:60-77`：

```text
_SM70_NVFP4_QPN2_CONFIGS = {
    # (K, N, fused gated-SiLU): (split-K, independent accumulator chains)
    (1536, 5120, False): (8, 2),
    (5120, 3584, False): (16, 2),
    # Qwen3.8 GDN qkvzba is logically N=4120 on TP4.  QPN2 consumes the
    # zero-padded physical N=4128 layout and the caller crops the result.
    (5120, 4128, False): (16, 2),
    (5120, 8704, False): (8, 2),
    (5120, 8704, True): (8, 2),
    (4352, 5120, False): (16, 2),
}
```
**scale 布局（NVFP4 group=16）** `nvfp4_qpn2_sm70.cu:113-128`：

```text
__device__ __forceinline__ half2 nvfp4_effective_scale(uint8_t value,
                                                       float global_scale) {
  // Match TurboMind's W4A16 weights: multiply the exact E4M3 group scale
  // by the FP32 global scale before rounding once to FP16. Rounding the
  // global factor first changes the model's weights on every decode step.
  const half raw = __low2half(fp8e4m3_to_half2(value));
  const half scaled =
      __float2half2_rn(__fmul_rn(__half2float(raw), global_scale));
  return __halves2half2(scaled, scaled);
}
```
E4M3→half2 也是位运算 `nvfp4_qpn2_sm70.cu:104-111`：

```text
__device__ __forceinline__ half2 fp8e4m3_to_half2(uint8_t value) {
  const unsigned short bits =
      ((static_cast<unsigned short>(value) & 0x80u) << 8) |
      ((static_cast<unsigned short>(value) & 0x7fu) << 7);
  const half converted =
      __hmul(__ushort_as_half(bits), __ushort_as_half(0x5c00));
  return __halves2half2(converted, converted);
}
```
**打包布局：一个分配里同时放 256 B FP4 码 + 32 B E4M3 scale（"bundled scale"）**
`docs/design/sm70_nvfp4_bundled_scale.md:3-6`：

```text
The qualified native QPN2 path stores one allocation containing 256 FP4 code
bytes and 32 E4M3 scale bytes per N32/K16 group. Code and scale tensors are
strided views of that allocation. Decode uses the existing reduction and
FP16 rounding order with adjacent group addresses.
```
代码侧模板参数与 stride 印证 `nvfp4_qpn2_sm70.cu:199-215`（`BundledScales` 决定 `w + 256` 与 `288` 步长）：

```text
      : codes(w, tile, groups, lane),
        scales((BundledScales ? w + 256 : static_cast<const uint8_t*>(s)) +
               static_cast<size_t>(tile) * groups * (BundledScales ? 288 : 32) +
               lane),
```

#### (c) **"边读边转"的正面证据（强）**：权重常驻 FP4，FP16 只在寄存器/共享内存中出现

`nvfp4_qpn2_sm70.cu:255-268`（主循环：读 `uint2` 码 → 现算 scale → 现反量化 → 立刻 MMA，**没有 FP16 权重数组**）：

```text
    for (int group = begin; group < begin + groups_per_warp; ++group) {
      const uint2 code = FullRows ? next_codes : reader.load(group);
      const half2 scale = nvfp4_effective_scale(
          FullRows ? next_scale
                   : __ldg(scale_ptr + static_cast<size_t>(group) *
                                           (BundledScales ? 288 : 32)),
          global_scale);
      ...
      half2 weights[8];
      dequant_e2m1x8(code.x, scale, weights);
      dequant_e2m1x8(code.y, scale, weights + 4);
      const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
      for (int tile_row = 0; tile_row < RowTiles; ++tile_row) {
        uint4 a01 = make_uint4(0, 0, 0, 0);
        ...
        VLLM_SM70_QPN2_MMA(accum[tile_row][0], a0[0], a0[1], b[0], b[1]);
        ...
```
（`weights[8]` 是**寄存器内的临时反量化结果**，紧接着就进 MMA；整个循环没有 FP16 权重数组、没有共享内存权重缓冲。）
更直接的是**加载后把 checkpoint 参数释放成空张量**，`nvfp4/sm70.py:154-166`：

```text
    def process_weights_after_loading(self, layer):
        sm70_tm.prepare_nvfp4_linear(
            layer, interleave_gated_silu=self.config.dense_gated_silu
        )
        self._release_checkpoint_parameters(layer)

    @staticmethod
    def _release_checkpoint_parameters(layer):
        layer.weight = Parameter(
            torch.empty(0, dtype=torch.uint8, device=layer.weight.device),
            requires_grad=False,
        )
        layer.weight_scale = Parameter(
            torch.empty(0, dtype=torch.float8_e4m3fn, device=layer.weight_scale.device),
            requires_grad=False,
        )
```
以及**Marlin 路线也只把 int4 重排、不物化 FP16 权重** `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:230-244`：

```text
    # WEIGHT
    # Repack weights to marlin format
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = layer.weight.view(torch.int32).T.contiguous()

    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1
    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=4,
        is_a_8bit=is_a_8bit,
    )
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)
```
（`layer.weight` 仍是 **4-bit packed int32**；**只有 scale 被 `.to(param_dtype)` 物化为 FP16**，
见同文件 `:248-266`。⇒ **"权重不展开、只把 scale 升到 fp16"** 正是 P6 需要的形态。）

**反面对照（确实存在会物化 fp16 的路径，但不能算"权重副本"）**：
`docs/design/sm70_nvfp4_bundled_scale.md:12-20` 的 prefill 路线**为了 prefill 把码+scale 还原进一块有界共享 scratch**：

```text
For prefill, a single kernel restores compact codes and rounded scale
operands into shared scratch. All serialized layers and captured graphs use
one bounded code buffer (21.25 MiB for these shapes), together with the
existing FP16 and scale workspaces. No layer retains a second weight layout.
If the code buffer cannot be reserved during preparation, the existing
native layout remains available. Worker shutdown releases these buffers.
```
⇒ **关键区别：scratch 是"每层共用、有界、可释放"，不是"每层一份 FP16 副本"**。这正是我们 §4B 要遵守的约束的更精细版本。

#### (d) 实测数字

**主结果（Qwen3.8-27B-NVFP4，TP4 V100，decode 逐步恢复）** `docs/design/sm70_qwen38_nvfp4_decode.md:60-70`：

```text
| Milestone | Steady decode | TPOT |
| Original mixed checkpoint route | 29.35 tok/s | about 34.07 ms |
| Channel-FP8 TurboMind | 57.285 tok/s | 17.457 ms |
| E4M3 XQA p256 | 62.615 tok/s | 15.971 ms |
| Exact channel-FP8 QPN8 | 65.522 tok/s | 15.262 ms |
| E4M3 XQA p128 | 67.880 tok/s | 14.732 ms |
| NVFP4 N32 selectors | 69.991 tok/s | 14.288 ms |
| Clean final binaries, native sampler, E4M3 p128 | 69.904 tok/s | 14.305 ms |
| Clean pre-merge binaries, native sampler, E4M3 p64 | 71.732 tok/s | 13.941 ms |
| Merged-source confirmation, native sampler, E4M3 p64 | **71.342 tok/s** | **14.017 ms** |
```
**算子级（bundled scale 的收益，M8 有效、M32 无效——被诚实记录）** `docs/design/sm70_nvfp4_bundled_scale.md:40-46`：

```text
Matched CUDA 12.8, Torch 2.10/cu128, V100-SXM2-32GB cold MLP graphs measure
73.144 to 69.842 microseconds at M8 (paired saving 3.302, 95% interval
3.072–3.548), and 133.268 to 133.914 at M32 (saving -0.645, interval
-0.865–-0.389). Both layouts have two compute kernels. The M32 result requires
an explicit same-service C4 check before promotion; ...
```
**MXFP4 grouped decode（合并 6 个单行专家的 launch）** `docs/design/sm70_deepseek_v4_mxfp4_grouped_decode.md:36-46`：

```text
| Stage | Six independent launches | One compact launch | Result |
|---|---:|---:|---|
| W13, K4096/N512 | 0.1156 ms | 0.0286 ms | Bitwise equal |
| W2, K256/N4096 | 0.0433 ms | 0.0117 ms | Bitwise equal |
...
The latest graph-node trace confirms that MXFP4 service falls from 7.542 to
1.957 ms/token and launches fall from 516 to 86.
```
**Marlin MoE 的 exact-shape 参数自动化（CTA 几何/split-K 环境变量）** `csrc/moe/marlin_moe_wna16/sm70_marlin_gemm.cuh:48-55`：

```text
inline constexpr char const* kSupportedSm70MarlinMoeCtaGeometries =
    "{CTA_M}x{CTA_N}x{CTA_K}x{Warps}x{WarpM}x{WarpN}x{WarpK} "
    "with CTA_M in {32,64}, CTA_N in {64,128,256}, "
    "CTA_K in {16,32,64,128}, Warps in {4,8}, WarpM in {32,64}, "
    "WarpN in {32,64}, WarpK in {16,32}, a valid CTA/warp "
    "decomposition, current CUTLASS Volta thread-map support, and "
    "phase-aware SM70 Marlin warp-K offset support. CTA_M=128/256 "
    "geometries are dense-only.";
```
（对应环境开关 `SM70_MARLIN_MOE_CTA_GEOMETRY` / `SM70_MARLIN_MOE_SPLIT_K` / `SM70_MARLIN_MOE_METADATA_CACHE`，
见 `sm70_marlin_gemm.cuh:57-60`。）

#### (e) 精度契约

见 §3.1(d)（revision 4/5/6）与 §3.7（KL 阈值）；NVFP4 特定的"FP32 先乘后一次 RNE 到 FP16"规则已引于 (b)。
另有 `docs/design/sm70_nvfp4_bundled_scale.md:8-11`：`Automatic selection is restricted to exact SM70, the existing qualified
shared native layout policy, and the TP4 MLP dimensions K5120/N8704 and K4352/N5120. Other shapes retain their existing layout.`

---

### 3.3 【目标 3】GGUF 原始块算子：`__dp4a`/INT8 内积、FP32 内积 + split-K、能力探测与 fallback

#### (a) `__dp4a` 内积：**"只让整数码进 dp4a，原始权重/激活 scale 在点积之后施加"**

`csrc/sm70_turbomind/ops/gguf_dp4a.cuh:33-35`（注释即设计契约）：

```text
// Shared by dense and routed kernels. Only integer codebook values reach
// dp4a; original weight and activation scales are applied after the dot.
```
核心解码 + 内积 `gguf_dp4a.cuh:56-115`（节选关键段）：

```text
  __device__ static float dot(const uint8_t* row, int group, const Q8_1& x,
                              const uint32_t* book, const uint32_t* masks) {
    const uint8_t* b = row + (group / 8) * kBlockBytes;
    const int sub = group % 8;
    const uint32_t low = load_u32_2(b + 2 + sub * (Type == 22 ? 4 : 8));
    ...
      const uint32_t s0 = masks[sign & 15], s1 = masks[sign >> 4];
      const int w0 = __vsub4(book[first] ^ s0, s0);
      const int w1 = __vsub4(book[second] ^ s1, s1);
      if constexpr (Type == 22) {
        ...
      } else {
        sum0 = __dp4a(w0, activation[2 * octet], sum0);
        sum0 = __dp4a(w1, activation[2 * octet + 1], sum0);
      }
    }
    const float d =
        __half2float(*reinterpret_cast<const half*>(b)) * __low2float(x.ds);
    ...
      return d *
             float(sum0 * (1 + 2 * (scale & 15)) +
                   sum1 * (1 + 2 * (scale >> 4))) *
             .125f;
```
**激活量化（Q8_1）是 warp 内一次 shuffle 归约** `gguf_dp4a.cuh:18-29`：

```text
__device__ __forceinline__ void quantize_q8_1_warp(Q8_1* out, float value) {
  const int lane = threadIdx.x % 32;
  float maximum = fabsf(value), sum = value;
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }
  const float d = maximum / 127.f;
  out->qs[lane] = maximum == 0.f ? 0 : int8_t(roundf(value / d));
  if (!lane) out->ds = __floats2half2_rn(d, sum);
}
```
**Q4_K / IQ4_XS 的 canonical 整数点积** `gguf_dp4a.cuh:130-170`（含 `__byte_perm` 重排 + `__dp4a`）：

```text
      const int w0 = __byte_perm(even, odd, 0x5140);
      const int w1 = __byte_perm(even, odd, 0x7362);
      sum = __dp4a(w0, activation[fragment * 2], sum);
      sum = __dp4a(w1, activation[fragment * 2 + 1], sum);
    }
    ...
    return float(sum) * (__half2float(scale) * __low2float(x.ds));
```

#### (b) **FP32 内积 + split-K 的实现文件 + `M8/N4352/K5120` 专用 shape**

- **原始块 FP32 向量点积（"native control"）**：`csrc/sm70_turbomind/ops/gguf_dmvq_sm70.cu`（原始记录 DMVQ reader）、
  `gguf_lattice_decode_sm70.cu` / `gguf_lattice_raw_sm70.cu` / `gguf_native_linear_sm70.cu`（lattice 原始块）
- **共享激活的 M8 投影面 + split-K**：`gguf_linear_shared_a_sm70.cuh`、`gguf_pair_shared_a_sm70.cuh`
- **专用 shape**：`docs/design/gguf_dmv_projections.md:41-46`（`N5120,K4352,M8,KW8,split1` 与 `split7` 并存）

`docs/design/gguf_dp4a_small_m.md:1-13`（**"FP32 累加 + FP16 投影边界不变"的契约**）：

```text
For M at most20, quantize each activation row once to standard Q8_1 blocks:
32 signed int8 values, FP16 scale and FP16 sum. Integer-codebook readers feed
SM70 dp4a directly, with original weight and activation scales applied after
the integer dot. FP32 accumulation and the final FP16 projection boundary
remain unchanged. Activations incur the explicitly authorized int8 rounding.
...
The common header under TurboMind is intended for both dense and routed GGUF
projections. The first reader is IQ3_S.
```
**同一文档给出的实测（M5 未达标也照实记录）** `gguf_dp4a_small_m.md:36-44`：

```text
| M | Native gate/up | dp4a fused | Q8_1 plus fused | Unique-source bandwidth |
|---|---:|---:|---:|---:|
|5|73.82us|42.85us|44.06us|403.94GB/s|
|20|210.37us|120.90us|121.99us|490.90GB/s|
...
The M5 result does **not** meet the450GB/s expansion gate.
```
**split-K 在共享激活面里的契约** `docs/design/gguf_dmv_projections.md:5-12`：

```text
Weights are reconstructed with FP16 group coefficients and accumulated in
FP32. Admission requires numerical checks, same-wheel model distribution
comparisons, and matched graph benchmarks. Other batch sizes retain the
existing projection route. Split-K workspaces and counters belong to each
layer and must remain stable across graph replay.
```
**他们明确"FP32 内积 + split-K"对 M=8 是有效的**（DMV 路线，`docs/design/gguf_dmv_projections.md:41-46`）：

```text
The temporary original-record DMVQ reader is admitted only for TP4 IQ2_XS
and IQ2_S down matrices (N5120,K4352,M8,KW8,split1). Same-process cold-L2
ABBA measured 23.81 versus 32.32 µs for IQ2_XS and 25.12 versus 33.91 µs
for IQ2_S against canonical GEMM. Relative L2 errors were 2.9e-4.
Q2_K down (33.70 versus 28.68 µs) and IQ1_M gate (43.46 versus 32.72 µs)
retain their prior routes.
```
**`M8/N4352/K5120` 的专用 shape 收益** `docs/design/gguf_projection_clock_diagnostics.md:24`：

```text
| IQ3_S gate/up, N=4352 K=5120 | 38.652 us | 37.807 us | identical FP16 bits |
```

#### (c) 能力探测与 fallback 的选择逻辑 —— **这一条对我们 P2/P6 价值最高**

**C++ 侧返回 bitmask 能力位**（`vllm/model_executor/layers/quantization/gguf_native.py:99-118`）：

```text
    native = torch.ops._C_gguf
    capabilities = native.ggml_dense_upstream_capabilities(
        weight, x, weight_type, weight.shape[0]
    )
    # Volta has FP16 tensor cores and dp4a, but no INT8 tensor cores. Admit
    # cuBLAS before MMQ for prefill and concurrent batches.
    device_index = x.device.index
    ...
    volta = current_platform.is_device_capability((7, 0), device_id=device_index)
    if (capabilities & 16) and volta and x.shape[0] >= prefill_min_m:
        return native.ggml_dense_blas(weight, x, weight_type, weight.shape[0])
    if capabilities & 4:
        return native.ggml_dense_mmvq(weight, x, weight_type, weight.shape[0])
    if capabilities & 16:
        return native.ggml_dense_blas(weight, x, weight_type, weight.shape[0])
    if (capabilities & 8) and not volta:
        return native.ggml_dense_mmq(weight, x, weight_type, weight.shape[0])
    return None
```
**能力位语义**（`gguf_native.py:137-146`，启动期不跑 GEMM 只探测）：

```text
        result["capabilities"][f"m{m}"] = {
            "mmvq": bool(bits & 4),
            "mmq": bool(bits & 8),
            "dequant_blas": bool(bits & 16),
        }
        if not bits:
            result["reason"] = "native_shape_or_storage_not_admitted"
```
**结构化 fallback 原因字符串 + 逐 M 带的校准（"测过更慢就不选"）** `vllm/model_executor/kernels/gguf.py:369-425`：

```text
def dp4a_expert_capabilities(...):
    """Measured M bands for Q8_1 gate/up plus integer down and route reduction."""
    canonical = source_type in (20, 23) and down_type == 20
    reason = None
    if not enabled:
        reason = "disabled_by_kernel_config"
    elif not is_sm70:
        reason = "requires_sm70"
    elif dtype != torch.float16:
        reason = "requires_fp16_activations"
    elif not canonical and (
        source_type not in (18, 21, 22) or down_type not in (20, 42)
    ):
        reason = "dp4a_expert_source_formats_unavailable"
    elif (k, n, num_experts) != (2560, 160, 512):
        reason = "dp4a_expert_shape_has_no_calibration"
    ...
    else:
        for operator in (
            "gguf_quantize_q8_1_sm70_out",
            "gguf_dp4a_lut4_gate_up_sm70_out"
            if canonical
            else "gguf_dp4a_gate_up_sm70_out",
            "gguf_dp4a_down_unroute_sm70_out",
        ):
            if not hasattr(torch.ops._C, operator):
                reason = f"operator_missing:{operator}"
                break
```
同文件 `:360-361` 把"M20 实测比 canonical grouped GEMM 慢"直接写成 reason：

```text
                "measured_slower_than_canonical_grouped_gemm"
                if source_type == 22 and m == 20
                else None
```
**GGUF 原生线性方法侧的 reason** `vllm/model_executor/layers/quantization/gguf_native.py:123-144`：

```text
        result["reason"] = "unsupported_native_format"
        ...
        result["reason"] = "native_extension_or_cuda_unavailable"
        ...
            result["reason"] = "native_shape_or_storage_not_admitted"
```
**能力/选择/回退是"声明式"的** `docs/design/gguf_native_sm70.md:22-36`：

```text
`KernelConfig.sm70_gguf` defaults to enabled. Each prepared projection reports
its operator capabilities, local shape, format, and rejected-route reason.
Existing formats retain their previous production operators. Formats absent
from those operators use the new fallback. Its explicit reference selector
uses dequantization followed by cuBLAS for M >= `prefill_min_m` (default 8),
and admitted MMVQ for smaller batches, otherwise BLAS. This threshold is not
a TurboMind default or evidence that the fallback wins at those shapes.
FP32 activations retain FP32 BLAS compute; they are not narrowed through FP16.
```

**对我们 `csrc/quantization/gguf/**`（已有 `__builtin_amdgcn_sdot4`，见
`csrc/quantization/gguf/ggml-common.h:1129`）的映射**：
① 把"能力探测"从 `__has_builtin` 编译期守卫升级为**运行期返回 bitmask 的 op**（这样诊断报告能显示实际走了哪条）；
② 把"哪种块格式 × 哪个 M 带 → 哪个 kernel"做成**带 reason 的候选列表**，并把"实测更慢"也写进 reason。

---

### 3.4 【目标 4】exact-shape decode GEMV：按 shape 选 kernel 的设计 + M=1 实测差

#### (a) 设计：`ExactMnkKernelImpl<..., M, N, K>` 精确模板特化 + 选择器，而不是通用 GEMM

`docs/design/sm70_awq_small_n_hmma_operator.md:440-448`：

```text
## 2026-07-12 TP4 M=1 Gate/Up CTA Fill

TP4 turns the merged gate/up descriptor into `M=1,N=8704,K=5120`. The old
`8x256x64`, split-2 tactic created only 68 CTAs, below the 72 V100 SMs. This is
a different scaling failure from the accepted M=5 verifier path above.

The accepted TP4 tactic is `8x64x64`, split 2, swizzle 4, with the existing
conflict-free A layout and lookahead-two mainloop. It is registered through
`ExactMnkKernelImpl<..., 1, 8704, 5120>` and selected only for the exact AWQ
descriptor. It cannot become feasible for prefill, M=5 verification, or the
TP4 row-projection shapes.
```
**M=1 相对通用 GEMM 的实测差（算子级 + 端到端）** 同文件 `:450-464`：

```text
| metric | TP4 old | TP4 N64 candidate |
|---|---:|---:|
| CTA count | 68 | 288 |
| registers/thread | 159 | 120 |
| dynamic shared memory | 20.51 KiB | 6.69 KiB |
| achieved occupancy | 6.24% | 11.51% |
| DRAM peak throughput | 37.35% | 48.67% |
| SM throughput | 27.23% | 37.04% |
| rank-0 236-call real-weight aggregate | 7.1955 ms | 6.4429 ms |

Rank 0 and rank 1 each pass all 236 real calls and 1,385,984 output elements
with zero difference from their fixed route. A full same-binary no-MTP TP4
CUDA-graph A/B also preserves every generated token ID and text over three
512-to-256 greedy repeats while improving `70.7158 -> 74.2807 tok/s` and
`14.1411 -> 13.4624 ms` TPOT (`+5.04%`, `-4.80%`).
```
**第二个 M=1 shape（`M=1,N=5120,K=1536`）——"换 tile 形状"而非"换算法"** 同文件 `:472-486`：

```text
## 2026-07-12 TP4 M=1 Row Projection K64

The TP4 row descriptor `M=1,N=5120,K=1536` has enough baseline CTAs but
spends most issue gaps at CTA barriers. Its accepted tactic is therefore
different from the gate/up N64 fill: `8x256x64`, `MMA=1x4x1`, split 12,
swizzle 4. ...
All 236 real calls on both ranks are exact. Paired
all-real rank-0 timing is `6.4509 -> 6.1731 ms` (`-4.31%`) on top of the
accepted gate/up route. The no-MTP TP4 graph result becomes
`74.2807 -> 75.9375 tok/s` and `13.4624 -> 13.1687 ms` TPOT, with identical
three-repeat token IDs/text.
```
**同样重要的"别乱改"的反例（负数结果，防止我们重犯）** 同文件 `:637-646`（`TG=1x8x1` 反而 +17.5%）：

```text
An explicit `TG=1x8x1` candidate also preserved CTA `8x256x64`, K64, split 7,
and all output bits. It reduced resources from the baseline 159 registers to
115 with zero local memory, but changes each warp from an N64 tile to N32. The
real-layer CUDA-event result is `24.1195 -> 28.3348 us` (`+17.5%`). The extra
per-warp HMMA/shared/control work overwhelms the expected residency gain; the
registration was removed. Do not retry bare `TG_N=8`.
```
以及 `:599-620`（split-K 延迟归约被 CUDA-event 否决）：

```text
... a correct synthetic split-7 reducer improves
`10.4850 -> 5.5552 us` (`-47.0%`). That result is not a valid GEMM predictor.
...
production is `24.375 us` per call (`1.5356 ms` for 63
down-proj calls) and stage plus reducer is `38.7448 us` (`2.4409 ms`, `+58.9%`).
```

#### (b) 另一条 exact-shape 路线：**把一个 launch 合并 6 个单行专家**（比"逐个 GEMV"更省）

见 §3.2(d)：`sm70_deepseek_v4_mxfp4_grouped_decode.md:36-46`（W13 0.1156→0.0286 ms，W2 0.0433→0.0117 ms，bitwise equal）、
`…:64-70`（direct top-6 路由：每层 0.101144 → 0.051226 ms）。

#### (c) GEMV 的实测对照表（另一处，逐 shape 带宽）

`docs/design/gguf_fp16_projection.md:88-95`：

```text
| Projection | Original mm, A arms (µs) | Row GEMV, B arms (µs) | Source bytes | Source-byte bandwidth (GB/s) |
```
（该文档 `:17-18` 说明 kernel 在 `vllm/model_executor/kernels/linear/fp16_gemv_silu.py`，
调 Triton `_fp16_gemv_silu_ranges_kernel`。）

#### (d) M=1 vs M=5 的"shape 决定战术"总结（对我们的直接价值）

`docs/design/sm70_awq_small_n_hmma_operator.md:439-475` 已给：
`M=1,N=8704,K=5120` → `8x64x64/split2/swizzle4`；`M=1,N=5120,K=1536` → `8x256x64/MMA=1x4x1/split12`；
`M=1,N=5120,K=4352` → **保持** `8x256x64/MMA=1x4x1/split7/swizzle0`（`…:601-602`）。
⇒ **结论：M=1 的赢法不是"新算法"，而是"每个 (M,N,K) 各自一组 split/CTA_N/swizzle 参数"，
并且必须用真实权重 + CUDA-event + A/B/A 逐 shape 验证；不要写一个通用 GEMM 指望它覆盖所有 decode shape。**

---

### 3.5 【目标 5】编译产物复用 / 启动优化：首次启动 271 s → 重启 95 s 的机制

> ⚠️ **重要更正**：任务书给出的"**271 s → 95 s**"这两个数字**在仓库中未找到**。
> 检索范围与结果（原始命令与输出）：
> ```bash
> $ grep -rn --include="*.md" --include="*.json" "271" docs/design docs/deployment benchmarks
> （命中的均为无关数值：ms、SHA256 片段、token/s 等；无数值与"启动/冷启动"共现）
> $ grep -rn --include="*.md" "startup\|cold start" docs/design/ | grep -v startup_cache
> （命中的均为算子预热/路由禁用等语境，无启动秒数对表）
> ```
> **因此本条的"271 s → 95 s"标记为 `未找到（未在仓库中定位到该数字对）`**；
> 下面给出的是**仓库中真实存在的机制与约定**（这才是可迁移的部分）。

#### (a) 缓存的**是什么**：`torch.compile` 的 **Inductor AOT 产物**（FX 子图级），**不是**权重/图的缓存

`docs/deployment/startup-cache.md:1-21`（测量口径，明确"启动 ≠ 只测权重加载"）：

```text
# Measuring model startup and compilation-cache reuse

Startup includes Python and worker initialization, weight loading, model
compilation, memory profiling, CUDA graph capture, and warmup. The weight-loader
timer alone does not measure the wait before a user can send a request.

`benchmarks/benchmark_startup_cache.py` measures fresh server processes from
process launch until the loopback health endpoint succeeds. ...

1. `uncached`: `VLLM_DISABLE_COMPILE_CACHE=1`.
2. `cache-fill`: `VLLM_DISABLE_COMPILE_CACHE=0`, allowing artifacts to be saved.
3. `cache-hit`: another process using the same caches, command, and requests.

The last phase must log an AOT artifact load.
```
代码侧：`benchmarks/benchmark_startup_cache.py:89-96`（三相位驱动）、`:134-142`（**判定 AOT 命中的字符串**）：

```text
                    row["aot_load_messages"] = log.count("Directly load AOT")
                    ...
                            "Warm launch did not fully reuse the AOT cache"
```

#### (b) **放在哪**：`VLLM_CACHE_ROOT/torch_compile_cache/<10 位哈希>/rank_<r>_<dp>/<prefix>/`

`vllm/compilation/backends.py:1056-1074`：

```text
        if not self.compilation_config.cache_dir:
            # no provided cache dir, generate one based on the known factors
            # that affects the compilation. if none of the factors change,
            # the cache dir will be the same so that we can reuse the compiled
            # graph.
            factors = [env_hash, config_hash, code_hash, compiler_hash]
            # Use SHA-256 for cache key hashing to be consistent across
            # compute_hash functions. Truncate for a short cache dir name.
            hash_key = hashlib.sha256(str(factors).encode()).hexdigest()[:10]
            cache_dir = os.path.join(
                envs.VLLM_CACHE_ROOT, "torch_compile_cache", hash_key
            )
            self.compilation_config.cache_dir = cache_dir
        ...
        local_cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}", self.prefix)
```
**四个哈希因子**（同一处 `:1043-1055`）：

```text
        env_factors = envs.compile_factors()
        env_hash = hash_factors(env_factors)
        # Compute config/compiler/code hashes once and reuse
        config_hash = vllm_config.compute_hash()
        compiler_hash = self.compiler_manager.compute_hash(vllm_config)
        forward_code_files = list(sorted(self.compilation_config.traced_files))
        ...
        code_hash = _compute_backend_code_hash(forward_code_files)
```
子图落盘/读回 `vllm/compilation/compiler_interface.py:389-391`（save）与 `:427-433`（load）：

```text
        path = os.path.join(self.cache_dir, key)
        ...
        inductor_compiled_graph = torch._inductor.CompiledArtifact.load(
            path=path, format=self.save_format
        )
        compilation_counter.num_compiled_artifacts_loaded += 1
```

#### (c) 怎么失效：**四因子哈希 + 显式开关**；并有 `cache_key_factors.json` 供诊断

`vllm/compilation/compiler_interface.py:193-207`：

```text
def is_compile_cache_enabled(
    vllm_additional_inductor_config: dict[str, Any],
) -> bool:
    vllm_inductor_config_disable_cache = vllm_additional_inductor_config.get(
        "force_disable_caches", False
    )
    ...
    return (
        not envs.VLLM_DISABLE_COMPILE_CACHE
        and not torch._inductor.config.force_disable_caches
        and not vllm_inductor_config_disable_cache
    )
```
`docs/deployment/startup-cache.md:65-90`（失效与诊断的**实战清单**，含他们自己踩过的坑）：

```text
- A log saying `vLLM's torch.compile cache is disabled` means that persistent
  Inductor/Triton directories alone do not enable vLLM AOT artifact reload.
- The SM70 compile-graph policy historically disables that reload because of
  output drift. [PR #536](https://github.com/1CatAI/1Cat-vLLM/pull/536) adds
  unregistered `VLLM_*` kernel switches to the cache key.
  [PR #621](https://github.com/1CatAI/1Cat-vLLM/pull/621) proposes restoring the
  default and carries a Torch 2.10 serialization backport.
- Test the very first reload after an empty-cache compile. Set
  `VLLM_FORCE_AOT_LOAD=1` on that reload to expose the original traceback instead
  of a silent fallback. This flag does not change the cache key.
```
`…:77-81`：

```text
- The non-mega vLLM AOT format saves the Triton kernels and constant arguments
  referenced by its FX graph, then remaps their indices into the new process's
  table. Torch's bundled-AOT serialization backport alone does not cover this
  vLLM serializer. Older artifacts containing Triton nodes without the table
  must be regenerated once; ...
```
`…:96-97`（**重要：缓存不覆盖 CUDA graph 捕获**）：

```text
- CUDA graph capture must still run in the new CUDA process; a compilation-cache
  hit does not eliminate that phase.
```
`docs/serving/sm70_release_profile.md:59-68`（1.5.1 起**默认开启**，位置与语义）：

```text
The 1.5.1 release requires compilation caching to be enabled by default, without
user environment variables. Removing the two existing SM70 cache opt-outs is
tracked in [#621](https://github.com/1CatAI/1Cat-vLLM/pull/621), ...
...
The standard cache location is `~/.cache/vllm`, or `$XDG_CACHE_HOME/vllm` when
configured. Compilation artifacts are reused for matching configuration,
compiler, environment and source hashes. Each process still loads model weights
and captures its CUDA graphs; enabling the cache does not eliminate all startup
work. An explicit `VLLM_DISABLE_COMPILE_CACHE=1` remains a troubleshooting opt-out.
```
`vllm/sm70_profiles/acceleration.py:794-821`（**缓存是否启用进入诊断报告**）：

```text
    cache_reason = None
    if envs.VLLM_DISABLE_COMPILE_CACHE:
        cache_reason = "compile_cache_disabled"
    elif cfg.model_config.enforce_eager or compilation.mode in (
        None,
        CompilationMode.NONE,
    ):
        cache_reason = "compilation_disabled"
    elif not is_compile_cache_enabled(cache_config):
        cache_reason = "inductor_cache_disabled"
    paths["compile_cache"] = _row(
        cache_reason,
        mode=getattr(compilation.mode, "name", None),
        switches={
            "VLLM_DISABLE_COMPILE_CACHE": envs.VLLM_DISABLE_COMPILE_CACHE,
            "VLLM_USE_AOT_COMPILE": envs.VLLM_USE_AOT_COMPILE,
            "force_disable_caches": cache_config.get("force_disable_caches", False),
            "torch_force_disable_caches": inductor_config.force_disable_caches,
        },
    )
```

#### (d) 有无"预热/预热一次"的部署配方

有：`docs/deployment/startup-cache.md:11-14` 的 `cache-fill` 相位就是**显式的预热步骤**；
`:23-28` 要求"用未用的 loopback 端口、新输出目录、不改模型/运行时/GPU/KV dtype/graph 设置"：

```text
## Run a controlled comparison

Reserve the GPUs before running. Use an unused loopback port and a new output
directory. Keep the model files, runtime, GPU set, TP size, context length,
batch limits, KV dtype, speculative configuration, and graph settings unchanged.
Do not compare a reduced-context or eager launch with a full production launch.
```
`:44-56` 给了完整可粘贴命令（含 `VLLM_CACHE_ROOT` 之外的 3 相位驱动方式）。

**⇒ 与我们的 P5（27B 冷启动 ~29 min）直接对应**：我们硬约束 17 已要求 `VLLM_CACHE_ROOT` 落 `/mnt/game`，
但**缺少"cache-fill → cache-hit 三相位测量 + `Directly load AOT` 命中判定 + `VLLM_FORCE_AOT_LOAD=1` 暴露 traceback"**这三件套。

---

### 3.6 【目标 6】内核选择诊断：`vllm.sm70_profiles show`、`/v1/sm70/acceleration`

#### (a) CLI：`python -m vllm.sm70_profiles {show,argv} [name] [--json] [--argv-lines] [--draft PATH]`

`vllm/sm70_profiles/profile.py:44-51`：

```text
def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m vllm.sm70_profiles")
    parser.add_argument("action", choices=("show", "argv"))
    parser.add_argument("name", nargs="?", default=PROFILE_NAME)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--argv-lines", action="store_true")
    parser.add_argument("--draft", type=Path, help="Use a local draft checkpoint")
```
profile 名与文件 `profile.py:10-17`：

```text
PROFILE_NAME = "qwen38_27b_nvfp4_dflash2"
_PROFILE_PATH = Path(__file__).with_name(f"{PROFILE_NAME}.json")
```
`vllm/sm70_profiles/__main__.py:1-6` 与 `__init__.py:1-6` 提供 `-m` 入口。
文档用法 `docs/serving/sm70_release_profile.md:4-8`：

```text
```bash
python -m pip install ./1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl
serve_qwen38_27b_nvfp4_v100.sh /path/to/target
serve_qwen38_27b_nvfp4_v100.sh /path/to/target --draft /path/to/downloaded-draft
python -m vllm.sm70_profiles show qwen38_27b_nvfp4_dflash2 --json
```
```

#### (b) HTTP 端点 `/v1/sm70/acceleration`

`vllm/entrypoints/serve/sm70/api_router.py:1-17`（**全文只有 17 行**）：

```text
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/v1/sm70/acceleration")
async def acceleration(request: Request):
    config = getattr(request.app.state, "vllm_config", None)
    report = getattr(config, "sm70_acceleration_report", {})
    return JSONResponse(report)


def attach_router(app):
    app.include_router(router)
```
配置字段 `vllm/config/vllm.py:988`（`sm70_acceleration_report: dict[str, Any] = Field(...)`）；
文档 `docs/serving/sm70_release_profile.md:37-39`：

```text
At configuration completion, before graph capture, the engine logs configured
SM70 route capabilities and final switch values. Disabled expected paths emit
warnings with a reason. `GET /v1/sm70/acceleration` returns the same report and
uses the server's existing API-key authentication.
```

#### (c) **报告了哪些字段**（"配置能力" vs "实际加载" 的分层是设计核心）

**报告顶层**（`vllm/sm70_profiles/acceleration.py:483-505`）：

```text
    report = {
        "profile": "qwen38_27b_nvfp4_dflash2" if release_profile else None,
        "sm70": sm70,
        "scope": "configured_capabilities",
        "expected_acceleration": load_profile()["expected_acceleration"]
        if release_profile
        else [],
        "paths": paths,
    }
    report["linear_kernel_policies"] = linear_policy_report(cfg.kernel_config)
    report["linear_kernel_selections"] = cfg.kernel_config.linear_kernel_selections
    report["moe_kernel_selections"] = cfg.kernel_config.moe_kernel_selections
    report["collective_kernel_selections"] = (...)
    report["ple_disk_cascade"] = {...}
    report["qsa_auto_e4m3"] = {...}
    report["sparse_kernel_policy"] = {
        "scope": "indexed_sparse_attention",
        "status": "runtime_guarded" if sparse_policy.reason is None else "fallback",
        "reason": sparse_policy.reason,
        ...
```
**每一行的统一 schema（`enabled`/`reason` + 任意附加值）** `acceleration.py:250-252`：

```text
def _row(reason: str | None = None, **values: Any) -> dict[str, Any]:
    return {"enabled": reason is None, "reason": reason, **values}
```
**"配置能力" vs "实际加载"的**明确**分离与自我警示**：
模块 docstring `acceleration.py:3`：

```text
"""Configured route capabilities, not a claim that a request hit a kernel."""
```
`docs/design/sm70_loaded_route_reporting.md:1-16`：

```text
# Loaded SM70 route reporting

The common linear selectors record their actual decision and rejection reasons
on the engine's `KernelConfig`. No second kernel registry or parallel capability
predicate is introduced. NVFP4, AWQ and FP8 use the same reporting lifecycle as
other linear kernels. Candidates below the chosen kernel are marked
`lower_priority`; reporting does not probe them or claim they are unsupported.

The diagnostic field is excluded from `KernelConfig.compute_hash`. Recording a
choice cannot invalidate a graph cache, alter priority or mutate environment.
Separate engines own separate dictionaries.

After initialization, the API process fetches worker reports once. Startup
logging and `/v1/sm70/acceleration` then read the same cached table. The report
contains resolved policies discovered from the existing SM70 configuration
fields, actual selector decisions, and the final kernel instances in loaded
layers. Final instances distinguish a retained kernel from an earlier candidate
whose QPN4 workspace preparation fell back. Local shapes and rejection reasons
remain available per rank.
```
末段（**这条正是我们要防的"配置了但没走到"假象**）`sm70_loaded_route_reporting.md:23-26`：

```text
These are loaded choices and runtime capability/preparation guards, not request
hit counters. Attention/verifier/collective/prefill category migrations still
need to replace their retained legacy policy rows with their own ordinary
selector/backend decisions.
```
**worker 侧"实际加载"报告的字段** `vllm/v1/worker/gpu_worker.py:956-1005`：

```text
    def get_sm70_acceleration_report(self) -> dict:
        """Read local selector decisions without rerunning capability probes."""
        ...
        report = {
            "rank": self.rank,
            "scope": "loaded_layer_selection",
            "compilation_mode": self.compilation_config.mode.name,
            "cudagraph_mode": mode.name,
            "decode_cudagraph_mode": mode.decode_mode().name,
            "mixed_cudagraph_mode": mode.mixed_mode().name,
            "linear_kernel_selections": selections,
            "collective_kernel_selections": (...),
            "ple_result_transports": transports,
            "ple_disk_row_readers": row_readers,
            "ple_pinned_decoders": ...,
            "ple_pinned_tables": [...],
            "prepared_linear_kernels": loaded_linear_kernels(self.model_runner.model),
            "prepared_gguf_layers": loaded_gguf_layers(self.model_runner.model),
            "model_input_preparation": {...},
```
`acceleration.py:230-248`（API 进程**一次性**拉取 worker 报告并缓存）：

```text
async def collect_worker_reports(engine_client, cfg) -> None:
    ...
        report["worker_routes"] = await rpc("get_sm70_acceleration_report", timeout=30)
    ...
    logger.info("SM70 loaded acceleration routes: %s", report["worker_routes"])
```
**原生能力探测（"哪些 native op 真的在 DSO 里"）** `acceleration.py:371-408`：

```text
    return {
        "fp16_grouped": hasattr(torch.ops._vllm_fa2_C, "sm70_grouped_fp16_fwd"),
        "fp16_short_splits": hasattr(
            torch.ops._vllm_fa2_C, "sm70_grouped_fp16_short_split_revision"
        ),
        "grouped_fp32": bool(flash_attn_grouped_e4m3_fp32_available()),
        "long_operator": builtin_long_attention() is not None,
        "long_enabled": long_attention_enabled(),
        "page_supported": long_attention_page_supported(page_size, BUILTIN_MANIFEST),
        "scalar": scalar_tail_attention_available(),
        "q8000": _get_sm70_d256_gqa_architecture_q8192_op() is not None,
        "bm32_aligned_pages": (
            getattr(flash_attn_v100_cuda, "paged_prefill_bm32_page_alignment", 0) == 16
        ),
    }
```
**强制门禁（可选）：`VLLM_SM70_REQUIRE_PROFILE_ACCELERATION=1` 时任一 expected path 未启用即拒绝启动**
`acceleration.py:823-846`：

```text
    failures = []
    if report["sm70"] or envs.VLLM_SM70_REQUIRE_PROFILE_ACCELERATION:
        for name in report["expected_acceleration"]:
            row = report["paths"][name]
            if not row["enabled"]:
                failures.append(f"{name}: {row['reason']}")
                logger.warning(
                    "SM70 profile acceleration is disabled: %s (%s). "
                    "Compare final CLI overrides with the release profile; "
                    "for operator_missing, reinstall the complete SM70 wheel.",
                    name, row["reason"],
                )
        logger.info("SM70 acceleration status: %s", report)
    report["expected_failures"] = failures
    if envs.VLLM_SM70_REQUIRE_PROFILE_ACCELERATION and failures:
        raise ValueError(
            "SM70 profile acceleration requirement failed: " + "; ".join(failures)
        )
```
**"配置能力 ≠ 请求命中"的补丁：worker 路由计数器 + 优雅关闭时 flush**
`docs/serving/sm70_release_profile.md:39-45`：

```text
uses the server's existing API-key authentication. A configured capability is
not proof that a particular request executed a kernel; retain worker route
logs and the graceful-shutdown route summary for performance qualification.
Worker shutdown explicitly flushes route counters, because multiprocessing
workers can exit without invoking Python's `atexit` hooks. Counters cover Python
dispatch and graph capture; CUDA graph replay does not increment them.
```
**fallback 原因字符串清单**（结构化、可 grep，散在代码中，非单一 enum）：
- 线性层策略：`candidate_not_qualified` / `lower_priority`（`sm70_loaded_route_reporting.md:5-7`）、
  `configuration_not_resolved`（`acceleration.py:551-552`）、`requires SM70 or SM72 (Volta)`（`nvfp4/sm70.py:143`）、
  `requires weight-only compressed-tensors NVFP4 layout`（`nvfp4/sm70.py:150`）、
  `packed weight must be rank two` / `requires K > 0, K % 128 == 0 and N > 0` /
  `missing native QPN2 operators: [...]`（`nvfp4/sm70.py:171-177`）
- GGUF：`unsupported_native_format` / `native_extension_or_cuda_unavailable` /
  `native_shape_or_storage_not_admitted`（`gguf_native.py:123,126,144`）、
  `dp4a_expert_source_formats_unavailable` / `dp4a_expert_shape_has_no_calibration` /
  `operator_missing:<name>` / `requires_fp16_activations` / `disabled_by_kernel_config`（`kernels/gguf.py:390-410`）、
  `measured_slower_than_canonical_grouped_gemm`（`kernels/gguf.py:360-362`）
- 编译缓存：`compile_cache_disabled` / `compilation_disabled` / `inductor_cache_disabled`（`acceleration.py:798-803`）
- attention：`operator_missing:sm70_d256_gqa_architecture_q8192_fwd`（`acceleration.py` q8000 分支）、
  `E4M3 grouped FP32 requires Flash-V100 precision revision 4; ...`（`flash_attn_v100.py:5087-5091`）

---

### 3.7 【目标 7】精度与验收契约：A/B/A、PPL delta、precision revision

#### (a) **A/B/A（ABBA）配对测量**是默认统计口径（不是"跑一次看看"）

- 算子级：`docs/design/gguf_fp16_projection.md:95`「Each ABBA arm contains 84 timing samples, with a 16 MiB cache flush before」
- 同进程 ABBA：`docs/design/gguf_dmv_projections.md:42`「Same-process cold-L2 ABBA measured 23.81 versus 32.32 µs …」
- 每层图 ABBA + 配对区间：`docs/design/sm70_nvfp4_bundled_scale.md:40-49`
  「paired saving 3.302, 95% interval 3.072–3.548」/「the per-replay critical rank (paired saving 5.871, 95% interval 4.943–6.813)」
- 算子文档内**强制"每个字节/带宽/图节点覆盖"都要给**：`docs/design/gguf_dflash2_iq3s.md:47-56`：

```text
Operator probes use actual checkpoint weights and TP4 shapes, M=8 with
M=1–16 coverage, cold L2 and repeated calls inside CUDA graph replay. Report
physical bytes, GB/s, latency delta and error against official FP32 GGUF
dequantization. Layer graphs cover full GDN, attention, draft and head tails.
```
- `docs/design/sm70_dflash2_fp32_defaults.md:64-69`（**算子基准的固定协议**）：

```text
Point `PYTHONPATH` at the task's Python package and rebuilt extension. The
operator benchmark holds E4M3 bytes, query, causal visibility, KV scales and
native library fixed. It reports 20 warmups, 25 ABBA blocks (50 samples per
variant), latency percentiles, PyTorch FP32/FP64 arithmetic error and the FP16
output-rounding floor.
```

#### (b) **token id + 文本 + SHA256 一致性**（"A/B/A"在**端到端**层的写法）

`docs/design/sm70_flash_v100_fp8_kv_long_context.md:178-181`：

```text
Both final 256-token hashes exactly match the old FP8 baseline:

- 16K: `bbf3175c6946c021376d51d26ecc7c21ae8d3e22baed664b475508534cb8b587`
- 64K: `100b97590c707aecd4398eba34e12e2087f5b793ef8e0073492393c08ba04892`
```
`…:298-303`（**同一段里同时给出"哈希不同时不硬当成回归"的判据纪律**）：

```text
The 27B AWQ and NVFP4 FP16/FP8 pairs have identical 16K and 64K output hashes.
The 35B sampled hashes can differ between FP16 and FP8 KV because the cache
dtype itself changes attention arithmetic. The relevant implementation gate is
stronger and more local: page 1056 and page 1088 bridge outputs are bitwise
equal to the previous accurate FP8 path for the tested full and partial query
chunks.
```
`docs/design/sm70_qwen38_nvfp4_decode.md:246-247`（任务通过率 A/B，同样含"零新失败"）：

```text
| Frozen pre-optimization baseline | 226/250 | 90.4% | 0 |
| Hand-sidecar candidate | 227/250 | 90.8% | 0 |
```
**"相同二进制 + 三轮 greedy 的 token id 与文本完全一致"** `docs/design/sm70_awq_small_n_hmma_operator.md:462-464`：

```text
... preserves every generated token ID and text over three
512-to-256 greedy repeats while improving `70.7158 -> 74.2807 tok/s` and
`14.1411 -> 13.4624 ms` TPOT (`+5.04%`, `-4.80%`).
```
**SHA256 用于"检查点/二进制/源码身份"**（不是输出比对）：`docs/design/sm70_nvfp4_bundled_scale.md:54-57`
（`config SHA256 1b3c71868d1299e52df6fc907deb202d5132b1ef0f72aae0ef6d15185dd53a5c`）、
`docs/design/sm70_dflash2_fp32_defaults.md:106-107`（native library SHA256）、
`docs/design/sm70_qwen38_distribution_acceptance.md:448`（"All 16 native hashes match"）。

#### (c) **PPL delta / 分布阈值**：PPL 用于回归门禁，**精度降低**改用 KL 阈值

PPL 出现在质量门清单里 `docs/design/sm70_dflash2_nvfp4_17ms.md:164`：

```text
- the existing GSM8K, MATH-500, HumanEval, corrected MBPP, and WikiText PPL
```
**精度降低类改动的判据（这是最可照搬的一套）** `docs/design/sm70_qwen38_distribution_acceptance.md:44-58`：

```text
For precision-reducing changes such as online QPN8, admission limits are shared
by no-MTP, target verification and draft probes. Apply them separately to the
pooled aligned positions for each role; retain per-prompt and language/task
stratum summaries for diagnosis:

| Metric | Initial limit |
| --- | ---: |
| Mean forward KL, nats | <= 0.001 |
| p99 forward KL, nats | <= 0.01 |
| Maximum forward KL, nats | <= 0.05 |
| Top-1 agreement | >= 99% |
| Maximum absolute raw-logit error | record only |
| Nonfinite logits | zero |
```
门槛的**计算协议**（可照抄的方法论）`…:37-42`：

```text
Use natural-log KL, temperature 1, the complete valid vocabulary, and logits
before sampling processors, top-k, top-p or temperature scaling. For each
fixed teacher-forcing prefix compute `KL(P_default || P_candidate)` in FP64
from stable log-softmax. Report reverse KL as a diagnostic. Never compare
free-running continuations with different prefixes. Ignore padded vocabulary
entries using the tokenizer/model vocabulary contract, not probability cutoff.
```
**测量噪声先标定，噪声大就"阻塞"而不是"放宽阈值"** `…:66-70`：

```text
These are initial engineering limits, not empirically calibrated guarantees.
Before first admission, repeat the default arm three times on identical prefixes
to establish measurement/replay noise. Noise exceeding a limit blocks
interpretation rather than automatically widening that limit.
```
**冻结输入（token id / tokenizer revision / probe 位置 / 哈希）** `…:75-90`：

```text
Freeze prompt token IDs, tokenizer revision, reference continuation IDs,
probe positions and hashes before evaluating a candidate. Start with the
existing 36-case quality manifest (12 MBPP, 12 GSM8K, eight Chinese QA and four
needle contexts), add four English prose prompts, and probe at least 16
teacher-forced decode positions per case. ...
Run the distribution gate at C1 only.
```
**"算子 L2 低 ≠ 模型质量好"的显式免责** `docs/design/sm70_dflash2_fp32_defaults.md:71-75`：

```text
Model acceptance must be checked with request-level draft/accepted-token
counter deltas. A rolling logger's mean acceptance length does not provide a
matched request comparison. FP32 is the chosen arithmetic contract, but lower
operator L2 alone does not establish model-quality improvement or guarantee
identical sampled tokens.
```
**"性能来自哪条路线要用 route counter 证明，别拿镜像名/flag 当证据"** `docs/design/sm70_dflash2_fp32_defaults.md:43-49`（已引）
+ `docs/design/sm70_flash_v100_fp8_kv_long_context.md:186-192`（用量化计数证明）：

```text
For the 256K request, every TP rank records 480 Q8192 FP32-accumulated
long-prefill calls and 496 E4M3 bridge calls. The final route summary also
records 48 `decode_xqa_e4m3_dynamic_page800` calls per rank. The run contains
no prefix-cache hit, non-finite value, worker failure, or CUDA error.
```
**多流/重叠场景的额外教训（防止用算子收益冒充端到端收益）**
`docs/design/sm70_deepseek_v4_mxfp4_grouped_decode.md:104-107`：

```text
The FP8 split result is the key scheduling lesson: service-time improvements
on an auxiliary stream are invalid unless the full overlap timeline also
shortens.
```

---

## 3.8 **可迁移技术表**（核心产出）

> `可行性` 列：✅ 可平移（只需接口/指令替换）｜🔶 需改造（思路可移植，指令/后端不同）｜⛔ 不适用。
> `预期收益` 是**量级判断**，非本步实测；引用数字均来自 1Cat-vLLM 自己文档（标注来源）。

| # | 技术 | 他们的证据（`file:line` / PR / 文档） | 在 gfx906 上是否可行 | 预期收益 | 移植成本 | 许可注意 |
|---|---|---|---|---|---|---|
| T1 | **KV 只做存储压缩**：KV 存 FP8 字节（uint8 张量），attention 内反量化到 FP16、softmax/PV/partial 全 FP32 | `flash_attn_v100.py:795-800`（uint8 KV）、`:2868-2874`（dtype 解析）；`sm70_dflash2_fp32_defaults.md:14-18`；`docs/design/sm70_flash_v100_fp8_kv_long_context.md:452-456` | ✅ 思路完全适用；gfx906 无 FP8 张量核**不是障碍**（他们也没有，全靠位扩展） | KV 显存/带宽减半；他们 prefill −29.7%~−50.8%、decode +10.4%~+21.8%（相对**他们自己的旧 FP8**，`…:165-171`）；**对 FP16 基线的 decode 收益仅 +0.49%~+2.73%**（`…:250-260`）⇒ 我们应把收益期望押在**长上下文 prefill/KV 容量**而非 decode 计算 | 中：需在 Triton `TRITON_ATTN`（S8 已选后端）里加"KV 反量化到 fp16 再做 QK/PV"，并改 KV 分配 dtype 与 scale 施加点 | Apache-2.0；仅借鉴设计，不必拷代码 |
| T2 | **prefill 用一次性 gather+expand 桥接到共享 FP16 页工作区**，而不是"每个 query CTA 各自转换" | `flash-attention-v100/kernel/fp8_kv_bridge.cu:18-30`；`kernel/fused_mha_api.cpp:77-80`；`…fp8_kv_long_context.md:82-96`、`:44-56`（NCU 根因：96.64 KiB SMEM/1 CTA每SM/25% occupancy）、`:100-105`（workspace 按 GPU×stream×Hkv×D 复用，512 MiB/rank） | ✅ 高度适用：gfx906 同样是带宽受限 + 我们 Triton attention 无共享 KV 工作区 | 他们 prefill 桥接 4.52×~4.88×（vs 旧 FP8）；我们的对应收益是"避免重复反量化" | 中：需一块**跨层复用、有界**的 fp16 KV 工作区 + 与 CUDA/HIP graph 捕获兼容（他们注释强调"图不能中途换 kernel"） | Apache-2.0 |
| T3 | **权重保持 4-bit 打包，反量化在寄存器内做位扩展，累加 FP32**；不物化 FP16 权重副本 | `csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu:143-163`（E2M1×8 位扩展）、`:188-193`（`mma.sync…f32.f16.f16.f32`）、`:255-264`（边读边转主循环）；`vllm/model_executor/kernels/linear/nvfp4/sm70.py:154-166`（**把 checkpoint 权重释放成空张量**） | ✅ **这就是 §4B 要的形态**；gfx906 对应指令是 `v_dot2_f32_f16`/`v_pk_fma_f16`（§4A 已确认可用） | 显存不增（我们 32 GB 宿主硬约束）；P6 目标 ≥10×（4096³ 304 ms → ≤30 ms） | 中高：需自写 HIP kernel（硬约束 21 已授权）；反量化位扩展逻辑可照抄公式 | Apache-2.0（QPN/TurboMind 部分）；**无 MIT 牵连** |
| T4 | **scale 处理顺序**：FP8/FP4 码先与"精确 E4M3 group scale × FP32 global scale"合成 FP16，再乘；**先在 FP32 里乘、只 round 一次** | `nvfp4_qpn2_sm70.cu:113-128`（注释明确"先 round global factor 会逐步改模型权重"）、`:104-111`（E4M3→half2 位运算） | ✅ 直接适用（我们 S8b 的 `scale_a*scale_b` epilogue 口径可对齐） | 精度：避免每步权重漂移（他们的 E4M3 修复即为此） | 低：口径规则，非代码量 | Apache-2.0 |
| T5 | **打包 scale 与码同分配**（"bundled scale"：256 B 码 + 32 B E4M3 scale / N32K16），减少一次访存 | `docs/design/sm70_nvfp4_bundled_scale.md:3-6`；`nvfp4_qpn2_sm70.cu:199-215`（`w + 256`、步长 288） | ✅ 适用：我们 KV/权重 scale 目前分开存，合并可降访存 | 他们 M8 冷 MLP 图 73.144→69.842 µs（−4.5%，95% CI 3.072–3.548）；**M32 反而 −(−0.645) 即变慢**⇒ 只在 M≤32 采用 | 中：改 layout + prepare kernel；注意他们的"限制到 exact SM70 + 特定 K/N"纪律 | Apache-2.0 |
| T6 | **exact-shape (`ExactMnkKernelImpl<...,M,N,K>`) 选战术**，而不是写通用 GEMM：按 (M,N,K) 定 CTA_N / split-K / swizzle / lookahead | `docs/design/sm70_awq_small_n_hmma_operator.md:440-448`（`M=1,N=8704,K=5120`→`8x64x64/split2/swizzle4`，注册为 `ExactMnkKernelImpl<...,1,8704,5120>`）、`:450-464`（CTA 68→288、occupancy 6.24%→11.51%、236 调用 7.1955→6.4429 ms、端到端 **70.7158→74.2807 tok/s，+5.04%**）、`:472-486`（另一个 M=1 shape −4.31%） | ✅ **对 P2 是最重要的一条**：gfx906 的 `sdot4` 也一样是"CTA 填充率/占用率"问题 | 他们同一 shape 端到端 +5.04%（decode TPOT −4.80%） | 低中：不需要新算法，只需参数/模板特化 + 逐 shape A/B；**但需要一套逐 shape 的 ABBA 基准设施** | Apache-2.0 |
| T7 | **`split-K` 要算"全部代价"**：联合/延迟归约若引入"部分平面写回+再读"，会反噬 | `…small_n_hmma_operator.md:599-620`（合成基准 −47.0%，真实层却 +58.9%，被 CUDA-event 否决）、`:637-646`（`TG=1x8x1` 寄存器 159→115 却 +17.5%，"Do not retry bare `TG_N=8`"） | ✅ **直接适用**：避免我们重蹈"以为降低寄存器就更快" | 避免负收益（他们两次都是 −0% ~ +50% 的反向教训） | 低：纪律，不是代码 | Apache-2.0 |
| T8 | **GGUF 原始块算子**：整数码进 `__dp4a`，**权重/激活 scale 在点积之后施加**，FP32 累加 | `csrc/sm70_turbomind/ops/gguf_dp4a.cuh:33-35`（设计契约注释）、`:56-115`（`__dp4a` 主循环）、`:18-29`（Q8_1 warp 量化）、`:130-170`（Q4_K/IQ4_XS `__byte_perm`+`__dp4a`） | ✅ **我们已有对应物**：`csrc/quantization/gguf/ggml-common.h:1129` 的 `__builtin_amdgcn_sdot4`（仅 MMVQ，W8A8 GEMM 未用） | 他们 M20 210.37→120.90 µs（2× 量级，`gguf_dp4a_small_m.md:36-44`）；**M5 未达 450 GB/s 门禁，照实记录** | 低：我们指令已在用；缺的是"把它接进 GEMM 路径 + 选择逻辑" | **Apache-2.0 + MIT（llama.cpp 公式）双许可**，必须同时署名 |
| T9 | **能力探测 = 运行期 op 返回 bitmask**，选择器按 (能力, arch, M) 决策，缺 op 时显式 reason | `vllm/model_executor/layers/quantization/gguf_native.py:99-118`（`capabilities & 4/8/16` + `volta` + `x.shape[0] >= prefill_min_m`）、`:137-146`（`mmvq`/`mmq`/`dequant_blas` 三键）、`:123,126,144`（reason 字符串） | ✅ 直接适用；比我们现在**编译期 `__has_builtin` 守卫**更强（编译期无法进入诊断报告） | 避免"配置了但没走到"；让禁用清单可执行化（我们已有 `tools/dcu/s8b_gate.sh` 的雏形） | 低：把 `__has_builtin` 分支提升为一个 `int` 返回的探测函数 | Apache-2.0 |
| T10 | **校准带（measured M bands）+ "实测更慢"也写成 reason** | `vllm/model_executor/kernels/gguf.py:369-425`（按 M∈{5,20} 逐带给 capability）、`:360-361`（`measured_slower_than_canonical_grouped_gemm`）、`:394,396`（`dp4a_expert_source_formats_unavailable` / `dp4a_expert_shape_has_no_calibration`） | ✅ 直接适用（我们的 W8A8/FP8 也是"某些 M 才划算"） | 把"性能上避免"从文档结论变成**机器可读的选择依据**（对齐我们 S8b 的"不硬禁用、给一次性告警"） | 低 | Apache-2.0 |
| T11 | **`M8/N4352/K5120` 级别专用 shape 化**：把 head/N/K 对齐到 kernel 友好的物理布局（含零 padding + 调用侧裁剪） | `docs/design/gguf_projection_clock_diagnostics.md:24`（IQ3_S gate/up N=4352 K=5120：38.652→37.807 µs）；`vllm/model_executor/kernels/linear/nvfp4/sm70.py:83-95`（`_pad_qpn2_output_rows`：逻辑 N=4120→物理 N=4128，QPN2 的 32 列契约，调用侧裁剪） | ✅ 适用：我们的 INT8/FP8 kernel 同样可对齐 K 到 16/32 的倍数 | 小幅但确定（他们 2.2%）；主要价值是**去掉尾部处理分支** | 低中：需要 prepare 阶段 padding + 调用侧裁剪 | Apache-2.0 |
| T12 | **编译产物复用**：torch.compile/Inductor AOT 子图缓存，key = `sha256([env_hash, config_hash, code_hash, compiler_hash])[:10]`，落 `<VLLM_CACHE_ROOT>/torch_compile_cache/<key>/rank_r_dp/<prefix>/` | `vllm/compilation/backends.py:1043-1074`；`vllm/compilation/compiler_interface.py:389-391, 427-433, 193-207` | ✅ **vLLM 自带，我们无需移植代码，只需正确使用 + 测量**（我们 S9b 已用 `VLLM_CACHE_ROOT`） | 他们 1.5.1 起默认开启（`docs/serving/sm70_release_profile.md:59-68`）；**CUDA graph 捕获仍必须重跑**（`docs/deployment/startup-cache.md:96-97`）⇒ 对 27B 冷启动 29 min，收益上限是"编译部分"，不是全部 | 低：改配置/加测量脚本 | Apache-2.0（且这本来就是上游 vLLM 机制） |
| T13 | **三相位启动测量 + 命中判定**：`uncached`(`VLLM_DISABLE_COMPILE_CACHE=1`) / `cache-fill` / `cache-hit`，判定字符串 `Directly load AOT`，失败即报错 | `benchmarks/benchmark_startup_cache.py:89-96`（三相位）、`:134-142`（命中判定 + `Warm launch did not fully reuse the AOT cache`）；`docs/deployment/startup-cache.md:1-28, 44-56` | ✅ 直接照搬（我们 P5 正缺这个） | 把"重启 95 s"这类说法变成**可复现门禁**；`VLLM_FORCE_AOT_LOAD=1` 可暴露静默回退 | 低：一个 ~150 行脚本 + 文档 | Apache-2.0 |
| T14 | **诊断报告：`scope` 分层 + `_row(reason)` 统一 schema + `expected_acceleration` 硬门禁** | `vllm/sm70_profiles/acceleration.py:3`（docstring 自我限定）、`:250-252`（`_row`）、`:483-505`（顶层 schema）、`:823-846`（`VLLM_SM70_REQUIRE_PROFILE_ACCELERATION` 拒绝启动）；`vllm/entrypoints/serve/sm70/api_router.py:9-13`（17 行 GET） | ✅ 高度适用：我们已有 `tools/dcu/s8b_gate.sh`（14 项禁用清单），可升级为"Z100 profile 报告 + 硬门禁" | 消除"配置了但没走到"；把一次性告警变成 CI/启动期可执行检查 | 中：要写一个 profile JSON + 报告 builder（可照 `acceleration.py` 结构）；我们无 FastAPI 端点也可以先只做 CLI | Apache-2.0 |
| T15 | **"配置能力 ≠ 请求命中"的诚实标注 + worker 路由计数器（关闭时 flush）** | `docs/design/sm70_loaded_route_reporting.md:1-26`；`docs/serving/sm70_release_profile.md:39-45`（**"A configured capability is not proof that a particular request executed a kernel"**；"CUDA graph replay does not increment them"） | ✅ 直接适用（我们 S8 已经吃过"aiter/hipBLASLt/skinny 均未进入"的核对苦） | 让性能归因可信；避免把"可配置"当成"已生效" | 低中：计数器 + 关闭 flush（注意 multiprocessing 不走 `atexit`） | Apache-2.0 |
| T16 | **precision revision 版本号硬绑定 native 库**：不匹配就 `raise` 重建，绝不静默降级 | `flash-attention-v100/flash_attn_v100/flash_attn_interface.py:1186-1197`（revision 4/5/6 三处 `raise RuntimeError`）；`docs/design/sm70_dflash2_fp32_defaults.md:20-32`（"cannot reach an incompatible native binary"、"raise a rebuild error instead of silently storing half partials"） | ✅ 直接适用（我们 S8b/S9b 已两次遇到"改了实现但旧 `.so` 还在"） | 消除最高危的一类静默错误（用旧二进制算新精度） | 低：一个 `int` 版本常量 + 启动期校验 | Apache-2.0 |
| T17 | **ABBA 配对 + 95% 区间 + 逐 shape 真实权重全量调用**（不用代表层外推） | `…small_n_hmma_operator.md:33-36`（"must use all 236 real checkpoint weights"）；`docs/design/gguf_fp16_projection.md:95`（84 samples/arm + 16 MiB flush）；`docs/design/sm70_nvfp4_bundled_scale.md:40-46`（paired saving + 95% interval） | ✅ 直接照搬（我们目前多为单次测量） | 让 `±1%` 级别的结论可信；避免 `sm70_awq_small_n_hmma_operator.md:637-646` 那类"短样本看走眼" | 低：测量纪律 + 小脚本 | Apache-2.0 |
| T18 | **端到端判据分层**：FP16 无损变换用"任务通过率 + 三轮 greedy token id/文本一致"；**精度降低类**才用 KL 阈值 | `docs/design/sm70_qwen38_distribution_acceptance.md:31-42`（FP16 融合三条件）、`:44-58`（KL 阈值表）、`:66-70`（噪声先标定，噪声大即阻塞）；`docs/design/sm70_qwen38_nvfp4_decode.md:246-247`（226/250 vs 227/250） | ✅ **直接照搬**：明确区分"重排/融合（应零损失）"与"降精度（需 KL 门禁）"——这正是我们 P6 需要的口径 | 让 P6 的"数值不劣于 S8b"有正式的、分层级的判据（而不是只有一个 `mean_rel_err`） | 低：文档/脚本层面；`KL` 采集脚本需教学式前缀 + FP64 log-softmax | Apache-2.0 |
| T19 | **"算子 L2 低 ≠ 模型质量好" + route counter 证明路线生效** | `docs/design/sm70_dflash2_fp32_defaults.md:71-75`；`docs/design/sm70_flash_v100_fp8_kv_long_context.md:186-192` | ✅ 直接照搬 | 防止用算子上限收益冒充端到端收益 | 低 | Apache-2.0 |
| T20 | **多流/重叠场景：服务时间改善必须落到完整重叠时间线** | `docs/design/sm70_deepseek_v4_mxfp4_grouped_decode.md:104-107`（"service-time improvements on an auxiliary stream are invalid unless the full overlap timeline also shortens"） | ✅ 直接适用（我们有 TP=4 + graph 的重叠） | 避免"单流更快、端到端更慢" | 低（纪律） | Apache-2.0 |
| T21 | **Marlin 作为"无 FP4 硬件"的 W4A16 回退路线** | `vllm/model_executor/kernels/linear/nvfp4/marlin.py:26-47`（warning 明说"no native FP4 support … weight-only FP4 compression"）；`vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:26-31`（SM70 走 `ops.sm70_marlin_available()`）、`:230-244`（只重排 int4、**不物化 fp16 权重**，仅 scale `.to(param_dtype)`） | 🔶 我们已有 `Gfx906GPTQWNA16LinearKernel`（W4A16 已跑通，见 §4 的 A/B/C 决策）⇒ **借鉴价值在"只升 scale 不升权重"这一条**，不在于换 kernel | 明确"权重不展开"的实现方式（scale 可升 fp16，权重保持 packed） | 低（口径） | Apache-2.0 |
| T22 | **小 N / exact-M 的 Tensor-Core 补充路线**（不是放弃 TC，而是把 N 调到 64 让 CTA 填满） | `docs/design/sm70_awq_small_n_hmma_operator.md:1-36`（236 调用 11.3419→9.0439 ms，−20.26%，1.254×，**全部 bitwise equal**） | 🔶 gfx906 有 FP16 张量核（S11d 已确认 `v_dot2_f32_f16` 可用）⇒ 我们的 INT8/FP8 也可考虑"用 fp16 张量核算模拟路径" | 他们 1.254× 且零精度损失 | 中高：需自编 HIP kernel | Apache-2.0 |

---

## 4. 门禁判定

| 门禁 | 结果 | 证据指向 |
|---|---|---|
| 目标 1：FP8 KV（存/反量化/QK-PV 精度/契约文档/实测收益） | ✅ 通过 | §3.1(a)-(e)；核心：`flash_attn_v100.py:795-800,2868-2874`、`fp8_kv_bridge.cu:18-30`、`sm70_dflash2_fp32_defaults.md:14-18,20-25`、`…fp8_kv_long_context.md:165-171,250-260` |
| 目标 2：NVFP4/MXFP4 无 FP4 硬件（kernel/组织/边读边转） | ✅ 通过 | §3.2(a)-(e)；核心：`nvfp4_qpn2_sm70.cu:143-163,188-193,255-264`、`nvfp4/sm70.py:60-77,154-166`、`marlin_utils_fp4.py:230-244`、`sm70_nvfp4_bundled_scale.md:3-6` |
| 目标 3：GGUF 原始块（dp4a / FP32+split-K / 能力探测与 fallback） | ✅ 通过 | §3.3(a)-(c)；核心：`gguf_dp4a.cuh:33-35,56-115`、`gguf_native.py:99-118,137-146`、`kernels/gguf.py:369-425`、`gguf_dp4a_small_m.md:1-13,36-44` |
| 目标 4：exact-shape decode GEMV（设计与 M=1 实测差） | ✅ 通过 | §3.4；核心：`…small_n_hmma_operator.md:440-464,472-486,599-620,637-646`、`sm70_deepseek_v4_mxfp4_grouped_decode.md:36-46` |
| 目标 5：编译产物复用（271 s → 95 s） | 🔶 **部分**：机制全部查清并给出 `file:line`；但**"271 s → 95 s"两个数字未在仓库中找到**（已列检索命令与范围） | §3.5；核心：`backends.py:1043-1074`、`compiler_interface.py:193-207,389-391,427-433`、`startup-cache.md:1-28,65-97`、`benchmark_startup_cache.py:89-96,134-142` |
| 目标 6：内核选择诊断（CLI / 端点 / 字段 / 实现文件） | ✅ 通过 | §3.6；核心：`profile.py:44-51`、`api_router.py:9-13`、`acceleration.py:3,250-252,371-408,483-505,823-846`、`gpu_worker.py:956-1005`、`sm70_loaded_route_reporting.md:1-26` |
| 目标 7：精度与验收契约（A/B/A、PPL、revision） | ✅ 通过 | §3.7；核心：`sm70_qwen38_distribution_acceptance.md:37-58,66-90`、`…fp8_kv_long_context.md:178-181,298-303`、`flash_attn_interface.py:1186-1197`、`sm70_dflash2_fp32_defaults.md:71-75` |
| 可迁移技术表 + P2/P5/P6 建议 | ✅ 通过 | §3.8（22 行）+ §6.1/§6.2/§6.3 |
| 只读参考、不拷代码进仓库 | ✅ 通过 | §2；`git status` 仅本档 |
| 未改禁改文档 | ✅ 通过 | §2 |

**门禁结论：7 条中 6 条 ✅ 完全通过，1 条（目标 5）🔶 部分通过**——
"271 s → 95 s"的两数字**未在仓库中定位到**（可能是发布说明/PR 描述中的数字，未随代码入库）；
但该条的**机制、缓存内容、位置、失效方式、测量协议**均已逐条查清并给 `file:line`，可迁移性不受影响。

---

## 5. 未完成项 / 阻塞与原因

| 项 | 状态 | 原因 |
|---|---|---|
| "首次启动 271 s → 重启 95 s"的原始出处 | **未找到** | 已按 `docs/design`、`docs/deployment`、`benchmarks/` 全量 grep `271`/`95`（命令与范围见 §3.5 开头）；命中的均为无关数值。可能是 release note / PR 描述中的口径，未随源码入库。**不编造**，标记为未找到 |
| 用 GitHub API 定向读取 | **未使用**（不需要） | 克隆成功，无需退化到 API/raw |
| 在他们的真机上验证任何数字 | **未执行** | 本步为纯代码/文档分析，且无 V100 硬件；所有数字均为**引用他们文档的原文**，非我们实测 |
| 把任何 1Cat 代码拷入本仓库 | **刻意未做** | 任务书 §"只读参考"要求；仅做映射与建议 |
| `/mnt/game` 直写 | **被沙箱拒绝，已绕行** | workspace-write 只允许写会话工作区；宿主 `touch /mnt/game/...` → `[sandbox: file access denied under workspace-write mode]`。改用容器 `z100-a7` 内 `git clone`（容器内可写 `/mnt/game`），**未申请提权**（本会话审批已禁用） |
| 逐条数字的"可复现性" | **部分** | 他们文档给的是结论表；原始 artifact 多数在私有路径（如 `bench_results/...`、`/home/ymzx/...`）不在本仓。只能引用其结论表原文 |

---

## 6. 对后续步骤的影响（**明确建议**）

### 6.1 P2（gfx906 W8A8 INT8 `sdot4`）：**优先 exact-shape 改编排，而不是写通用 GEMM**

**建议：先做 T6 + T7 + T9 + T17，把"通用 GEMM"降级为兜底。**

1. **放弃"一个通用 W8A8 GEMM 覆盖所有 shape"的目标**。照 T6：我们的 decode 是 `M=1..8`，
   而 1Cat 在 M=1 上的收益**全部来自"per-(M,N,K) 的 CTA_N/split-K/swizzle 重选"**，不是新算法：
   - `M=1,N=8704,K=5120`：`8x64x64/split2/swizzle4` ⇒ CTA 68→288、occupancy 6.24%→11.51%、端到端 **+5.04%**
     （`sm70_awq_small_n_hmma_operator.md:440-464`）；
   - `M=1,N=5120,K=1536`：`8x256x64/MMA=1x4x1/split12` ⇒ 再 **−4.31%**（`:472-486`）；
   - `M=1,N=5120,K=4352`：**保持** `8x256x64/MMA=1x4x1/split7/swizzle0`（`:599-602`）——同一个 M，不同 N/K 就是不同战术。
2. **对每个真实 decode shape 建一份"战术表"**（`(M,N,K) → CTA_N, split-K, 线程映射`），
   并用 T17 的 ABBA + **全部真实权重**逐 shape A/B（他们的门槛是"no shape regresses"，`:33-36`）。
   这比"写通用 GEMM 再调参"便宜得多，且与我们已跑通的 `Gfx906GPTQWNA16LinearKernel`/GGUF 复用面一致。
3. **把 `split-K` 的完整代价算进去**（T7）：不要引入"部分平面写回 + 第二 kernel 归约"，
   除非它能在**真实层权重 + CUDA-event** 上净赢 ≥2%（他们两次都因此否决，`:599-620`、`:637-646`）。
4. **DP4A 路线（T8）优先级高于"自写 fp16 张量核 INT8"**：我们已有 `__builtin_amdgcn_sdot4`
   （`csrc/quantization/gguf/ggml-common.h:1129`），缺的正是 1Cat 的"能力探测 → 选择 → reason"链路（T9/T10）。
   把编译期 `__has_builtin` 升格为**运行期返回 bitmask 的 op**，diagnostic 才能显示实际走了哪条路。
5. **许可提示**：若复用其 `dp4a` 公式，注意 **Apache-2.0 + MIT（llama.cpp）** 双署名（§3.0）。

### 6.2 P6（FP16 模拟 FP8）：**照抄"边读边转 + 单次 RNE scale + FP32 累加"，并在 M≤32 才用打包 scale**

**建议：T3 为主，T4/T5 为精度与访存细节，T18 为验收口径。**

1. **权重保持 FP8 常驻，反量化只在寄存器/共享内存里发生**（T3 的实现模板）：
   - 位扩展反量化（照 `nvfp4_qpn2_sm70.cu:143-163` 的"纯位运算 + 每步只 round 一次"风格），
     gfx906 上对应 `v_cvt_f32_fp8`（若 DTK 认）或 `v_cvt_pkrtz_f16_f32` 路径 + `v_dot2_f32_f16`/`v_pk_fma_f16`；
   - **累加必须 FP32**（§4B 已要求，1Cat 用 `mma.sync…f32.f16.f16.f32` 印证）；
   - **绝不整体物化 FP16 权重**：1Cat 的强证据是 `process_weights_after_loading` **把 checkpoint 权重释放为空张量**
     （`nvfp4/sm70.py:154-166`），Marlin 路线也只升 **scale** 到 fp16、权重保持 packed int32
     （`marlin_utils_fp4.py:230-244, 248-266`）。
2. **scale 施加顺序（T4）**：把 `E4M3 group scale × FP32 global scale` **在 FP32 里乘、只 round 一次到 fp16**；
   先 round global factor 会逐步改变权重（`nvfp4_qpn2_sm70.cu:113-128` 注释明说）。
   这与我们 S8b 的 `scale_a*scale_b` epilogue 口径可以直接对齐。
3. **打包 scale（T5）只在 M≤32 用**：他们 M8 冷 MLP 图 −4.5%（95% CI 3.07–3.55 µs/72.8 µs），
   **M32 反而变慢**（`sm70_nvfp4_bundled_scale.md:40-46`）⇒ 不要无条件套用。
4. **prefill 侧学 T2**：如果 FP8 只做 KV 存储压缩，**先做一次 gather+expand 到跨层复用的有界 fp16 工作区**，
   而不是让每个 query tile 各自反量化（1Cat 的 NCU 根因：96.64 KiB SMEM、1 CTA/SM、25% occupancy、4% tensor activity）。
   他们图捕获的约束也要抄：**"a graph cannot switch kernels later as context grows"** ⇒ 用 `seq_lens` 设备端选 p 尺寸。
5. **验收口径（T18）分层**：
   - P6 属"精度降低类"⇒ 用 KL 门禁（mean ≤0.001 / p99 ≤0.01 / max ≤0.05 / top-1 ≥99% / 非有限=0），
     并**先跑 3 次默认臂标定噪声**，噪声超限就"阻塞"而非放宽；
   - 同时保留我们 S8b 的 operator 级判据（`mean_rel_err` 不劣于 1.465e-07 量级、全 finite），
     并按 T19 补一条"4096³ 从 304 ms 降到 ≤30 ms **且** profiler 里出现 packed FP16/dot 指令（不能再是清一色 `v_fmac_f32`）"。
6. **不要硬禁用 FP8 block-scaled**（与 S8b 结论一致）：改成 T10 的"校准带 + reason"
   （`measured_slower_than_canonical_grouped_gemm` 这种写法），让"性能上避免"变成机器可读的选择依据。

### 6.3 P5（冷启动）与"内核选择诊断"：**直接照搬 T12 + T13 + T14 + T15 + T16（我们几乎不用写新 kernel）**

**P5（冷启动/编译产物复用）**：
1. **照搬三相位测量（T13）**：`uncached`(`VLLM_DISABLE_COMPILE_CACHE=1`) → `cache-fill` → `cache-hit`，
   命中判定用日志里的 **`Directly load AOT`** 计数，未命中即报错；诊断首次 reload 用 **`VLLM_FORCE_AOT_LOAD=1`** 暴露 traceback
   （脚本模板：`benchmarks/benchmark_startup_cache.py:89-96,134-142`）。
2. **明确缓存边界（T12）**：缓存的是 **torch.compile/Inductor 的 AOT 子图**，
   key = `sha256([env_hash, config_hash, code_hash, compiler_hash])[:10]`，落 `<VLLM_CACHE_ROOT>/torch_compile_cache/<key>/rank_r_dp/<prefix>/`
   （`backends.py:1043-1074`）。**权重加载与 CUDA graph 捕获必须重跑**（`startup-cache.md:96-97`）
   ⇒ 对 27B 冷启动 ~29 min，收益上限只覆盖"编译"那一段；**预热量必须包含 gdraph 捕获**，别把 95 s 当成端到端目标。
3. **别把"persistent Inductor/Triton 目录"当成"vLLM AOT 复用已开"**（`startup-cache.md:65-66`），
   这正是我们硬约束 17 只设了 `VLLM_CACHE_ROOT` 却可能仍全量重编的隐患。
4. **失效因子要含未注册的 `VLLM_*` 开关**：他们踩过的坑是"改了 kernel 开关但没进 cache key ⇒ 静默复用旧图"
   （`startup-cache.md:67-69`，PR #536）；我们新增 Z100 专用开关时必须同样处理。

**内核选择诊断（T14 + T15 + T16）**：
1. **做 `z100_profiles`（CLI 优先，HTTP 可选）**，schema 照 `_row(reason)` 统一为 `{"enabled": bool, "reason": str|None, ...}`（`acceleration.py:250-252`）。
2. **必须分两层**：`scope: "configured_capabilities"`（配置能力）与 `scope: "loaded_layer_selection"`（实际加载的 kernel 实例），
   并在模块/接口 docstring 里明写 **"Configured route capabilities, not a claim that a request hit a kernel"**（`acceleration.py:3`）。
   我们的对应物就是 `s8b_gate.sh` 里那 14 项禁用清单 + `ROCM_SKINNY_GEMM_ENABLED` 一类的"配置了但没走到"（S8 已实际遇到）。
3. **加"请求级命中"计数器**（T15）：worker 侧计数 + **优雅关闭时显式 flush**（multiprocessing 不走 `atexit`），
   并注明 **CUDA/HIP graph replay 不会自增计数器**（`sm70_release_profile.md:39-45`）。
4. **加 `VLLM_SM70_REQUIRE_PROFILE_ACCELERATION` 式的硬门禁**（`acceleration.py:823-846`）：
   我们自己叫 `Z100_REQUIRE_PROFILE_ACCELERATION`，`expected_acceleration` 里的每项未启用即拒绝启动，并打印 `reason`。
5. **加 precision revision 版本号（T16）**：native 库暴露一个 `int` revision，python 侧不匹配即 `raise`，
   明确文案照抄"Rebuild the extension and restart workers together with this Python update"。
   我们 S8b/S9b 已两次遇到"实现改了但旧 `.so` 还在"，这条能直接消灭该类静默错误。

### 6.4 对 S11 其余步骤的间接影响

- **S11b（P0 kernel 剖析）**：建议直接采用 T17（ABBA + 配对区间 + 全部真实权重）与 T19（route counter 证明路线生效）作为剖析报告的口径。
- **S11d（算力审计）**：T22 提示"小 N 也可能用张量核"——审计结论若显示 `v_dot2_f32_f16` 可用，
  应把"fp16 张量核的 exact-shape 变体"列为 P2/P6 的候选，而不是只走 SIMT。
- **S11e/S11f**：见 §6.1/§6.2；两份实现都应**先写"能力探测 + reason"骨架**（T9/T10），再写 kernel。
- **S11c（P1 内存绕法）**：T1/T2 的"KV 存储压缩 + 有界跨层工作区"是 32 GB 宿主下更省内存的一条正路（我们 §4B 的硬约束正是内存）。

---

## 7. 复现方式

```bash
# 0) 前提：本步零 GPU、零编译。宿主对 /mnt/game 无写权限，克隆在容器内做。
docker ps --format '{{.Names}}\t{{.Status}}'          # 确认 z100-a7 在跑；不要用 qwen38-vllm

# 1) 取源（只落 /mnt/game；--depth 1，约 190 MB）
docker exec z100-a7 bash -lc 'cd /mnt/game && rm -rf 1cat-vllm && \
  git clone --depth 1 https://github.com/1CatAI/1Cat-vLLM.git 1cat-vllm'
docker exec z100-a7 bash -lc 'cd /mnt/game/1cat-vllm && git log -1 --format="%H %ad %s"'
# 期望：c4f6245f841466782752a8c3283e4727565cf17a ... [Kernel] Reuse M8 gate operands and TP4 norm packets (#1041)

# 2) 宿主可直接只读访问该副本（无需 docker exec）
R=/mnt/game/1cat-vllm

# 3) 逐条复核（每条都应打出对应 file:line 与原文）
# T1/T2 FP8 KV
sed -n '795,800p;2868,2874p;5086,5094p'   $R/vllm/v1/attention/backends/flash_attn_v100.py
sed -n '18,30p'                           $R/flash-attention-v100/kernel/fp8_kv_bridge.cu
sed -n '14,32p'                           $R/docs/design/sm70_dflash2_fp32_defaults.md
sed -n '160,175p;245,262p'                $R/docs/design/sm70_flash_v100_fp8_kv_long_context.md
# T3/T4/T5 NVFP4 边读边转 + scale + bundled
sed -n '113,128p;143,163p;188,193p;255,264p' $R/csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu
sed -n '60,77p;138,166p'                  $R/vllm/model_executor/kernels/linear/nvfp4/sm70.py
sed -n '230,266p'                         $R/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py
sed -n '3,6p;40,46p'                      $R/docs/design/sm70_nvfp4_bundled_scale.md
# T8/T9/T10 GGUF dp4a + 能力探测
sed -n '17,35p;56,115p;130,168p'          $R/csrc/sm70_turbomind/ops/gguf_dp4a.cuh
sed -n '99,118p;123,146p'                 $R/vllm/model_executor/layers/quantization/gguf_native.py
sed -n '355,425p'                         $R/vllm/model_executor/kernels/gguf.py
sed -n '1,13p;36,44p'                     $R/docs/design/gguf_dp4a_small_m.md
# T6/T7 exact-shape M=1
sed -n '1,36p;440,464p;472,486p;599,646p' $R/docs/design/sm70_awq_small_n_hmma_operator.md
# T12/T13 编译缓存
sed -n '1043,1074p'                       $R/vllm/compilation/backends.py
sed -n '193,207p;389,391p;427,433p'       $R/vllm/compilation/compiler_interface.py
sed -n '1,28p;65,97p'                     $R/docs/deployment/startup-cache.md
sed -n '89,96p;134,142p'                  $R/benchmarks/benchmark_startup_cache.py
# T14/T15/T16 诊断与 revision
sed -n '3p;250,252p;371,408p;483,505p;823,846p' $R/vllm/sm70_profiles/acceleration.py
sed -n '44,51p'                           $R/vllm/sm70_profiles/profile.py
sed -n '9,13p'                            $R/vllm/entrypoints/serve/sm70/api_router.py
sed -n '956,1005p'                        $R/vllm/v1/worker/gpu_worker.py
sed -n '1,26p'                            $R/docs/design/sm70_loaded_route_reporting.md
sed -n '1186,1197p'                       $R/flash-attention-v100/flash_attn_v100/flash_attn_interface.py
# T18 验收契约
sed -n '31,90p'                           $R/docs/design/sm70_qwen38_distribution_acceptance.md
# 许可
head -3 $R/LICENSE
sed -n '3,5p' $R/csrc/sm70_turbomind/ops/gguf_dp4a.cuh
sed -n '13,17p' $R/docs/design/gguf_native_sm70.md

# 4) 复核"271 s → 95 s"确实不在仓库（应只命中无关数值）
grep -rn --include="*.md" --include="*.json" "271" $R/docs/design $R/docs/deployment $R/benchmarks | head

# 5) 本仓库零改动的核对
git -C /home/lyh/z100-vllm status --short docs/ csrc/ vllm/
# 期望：只有 docs/result/z100-11g-1cat-vllm-reference-analysis-result.md
```

---

## 8. 参考

**被分析仓库（只读参考，未拷入本仓库）**
- `1CatAI/1Cat-vLLM`，默认分支 `main`，commit `c4f6245f841466782752a8c3283e4727565cf17a`（2026-10-08），
  取源副本 `/mnt/game/1cat-vllm`（`--depth 1`，190 MB），许可 **Apache-2.0**（`LICENSE:1`），
  其中 GGUF 子路径含 **llama.cpp（MIT）** 上游代码与归属（`gguf_dp4a.cuh:3-5`、`docs/design/gguf_native_sm70.md:13-17`、
  `csrc/quantization/gguf_upstream/source_manifest.json`）。
- 相关 PR（其文档内引用）：[#536](https://github.com/1CatAI/1Cat-vLLM/pull/536)（未注册 `VLLM_*` 开关进 cache key）、
  [#621](https://github.com/1CatAI/1Cat-vLLM/pull/621)（恢复编译缓存默认 + Torch 2.10 序列化 backport）、
  [#494](https://github.com/1CatAI/1Cat-vLLM/pull/494)（page4 planner 修复）、
  [#507](https://github.com/1CatAI/1Cat-vLLM/pull/507)、[#288](https://github.com/1CatAI/1Cat-vLLM/pull/288)、
  [#160](https://github.com/1CatAI/1Cat-vLLM/pull/160)、[#163](https://github.com/1CatAI/1Cat-vLLM/pull/163)、
  [#504](https://github.com/1CatAI/1Cat-vLLM/pull/504)、[#506](https://github.com/1CatAI/1Cat-vLLM/pull/506)。

**本项目内参考**
- [docs/海光Z100-性能优化方案.md](../海光Z100-性能优化方案.md) §4A（Z100 算力清单）、§4B（P6：FP16 模拟 FP8）、§4C（本步 7 条目标）、§4（P2 W8A8）、§7（P5 冷启动）
- [docs/result/z100-08b-gating-and-requirements-result.md](z100-08b-gating-and-requirements-result.md)（FP8 block-scaled 现状：纯 `v_fmac_f32` 模拟、36× 慢、数值干净）
- [docs/result/z100-08-attention-and-quant-paths-result.md](z100-08-attention-and-quant-paths-result.md)（`TRITON_ATTN` 选型、aiter/hipBLASLt/skinny/flash-attn 均未进入）
- [docs/result/README.md](README.md)（归档模板与索引）
- [docs/z100-todolist.md](../z100-todolist.md) 全局硬约束 1–21（尤其 1、2、3、17、18、21）
- 本项目已有 INT8 复用点：`csrc/quantization/gguf/ggml-common.h:1129`（`__builtin_amdgcn_sdot4`，仅 GGUF MMVQ）
