# S8 路径与后端核对（attention / 量化 / 禁用清单）

> 上层文档：[适配方案](../海光Z100-DTK26.04-gfx906-vLLM适配方案.md) · [Todolist](../z100-todolist.md)
> 前置：[S5](z100-05-csrc-kernels.md) · 状态：⬜ 待做

## 目的

内核能编出来 ≠ 路径选对了。本步骤核对"gfx906 上哪些路径必须开、哪些必须关"，并把结果固化成运行期默认与文档，避免后续误改。

## 必须开的路径

| 路径 | 位置 | 期望 |
|---|---|---|
| TRITON_ATTN 优先 | `vllm/platforms/rocm.py` 的 backend 列表末尾：`if on_gfx906(): backends.remove(TRITON_ATTN); backends.insert(0, TRITON_ATTN)` | 8K 上下文日志为 `TRITON_ATTN`（gfx906 无 MFMA，ROCM_ATTN 的 2D paged kernel 在长上下文慢约 24×） |
| unified attention gfx906 预填配置 | `vllm/attention/ops/triton_unified_attention.py`、`v1/attention/ops/triton_reshape_and_cache_flash.py` | 走 gfx906 专用 config / eager 回退分支 |
| GDN / causal_conv1d gfx906 内核 | `vllm/model_executor/layers/mamba/{ops/causal_conv1d.py,gdn_linear_attn.py}` | 命中 `csrc/mamba/gfx906_decode_kernels.cu` 的 op |
| `on_gfx906()` 门控整体生效 | 382 处 | 见 [S1](z100-01-arch-normalization.md)，前提是架构已归一化 |

## 必须关的路径

| 开关/路径 | 值 | 理由 |
|---|---|---|
| `VLLM_ROCM_USE_SKINNY_GEMM` | `0` | gfx906 无 MFMA；`wvSplitK` 不可用（见 [S5](z100-05-csrc-kernels.md)） |
| `VLLM_ROCM_USE_AITER*` | 全 `0` | aiter 面向 MI3xx，DTK 不提供 |
| FlashAttention（`flash_attn_2_cuda.so`） | 不使用 | DTK 自带的该 `.so` 为 CDNA/gfx928 构建，gfx906 上 `invalid device function`；fork 已有 `fa_utils.py` 的 gfx906 早退 |
| hipBLASLt / MIOpen 的 gfx906 路径 | 不进入 | DTK 的 `hipblaslt/library` 只有 gfx928 |
| FP8（`--quantization fp8*`、`VLLM_ROCM_FP8_*`） | 不用 | gfx906 无 FP8 单元 |
| CUDA graph | 默认 `--enforce-eager` | 见 [S7](z100-07-memory-kvcache-graph.md) |

## 语义澄清（防止被"善意修正"改坏）

- **不要把 `gfx906` 加进 `_ON_GFX9`**。本 fork 的 `on_gfx9()` 有意排除 gfx906/gfx908（`fused_moe/experts/gpt_oss_triton_kernels_moe.py:53` 有注释）。
- `on_cdna()` 在 gfx906 上为 **True**（实现是 `"gfx9" in _GCN_ARCH` 子串匹配），这被 fork 用于 ViT attention 选择等场景，属预期。
- `on_gfx906()` 与 `on_gfx9()` 同时使用的地方（例如 `layers/utils.py:451` 的 `n <= 16 and not on_gfx9()`）是**有意组合**，不要合并两者语义。

## 量化路径核对

| 路线 | 现状 | 本步骤要做 |
|---|---|---|
| W8A8 INT8（生产在用） | 依赖自研 `int8_sdot4`（`V_DOT4_I32_I8`），Triton 需 `supportsVDot()` 对 VEGA 为真 | 确认 [S6](z100-06-triton-gfx906.md) 完成后 sdot4 仍可用；记录与生产 13–14 ms/token 的差距 |
| AWQ / GPTQ W4A16 | fork 有 `mixed_precision/{gfx906_gptq_wna16,torch_wna16,triton_w4a16,exllama,conch}.py` | 抽查 `kernels/linear/__init__.py` 的选择逻辑在 gfx906 上落到可用实现 |
| GGUF | fork 有 gfx906 专用列布局与 mmvq 分支 | 阶段 B（[S10](z100-10-phase-b-features.md)） |
| MoE + GPTQ-Int4 / AWQ-4bit | 已知不可用 | 立即在文档标注"不支持"，不要浪费验收时间 |

## 做法

1. 起服务并抓路径证据（每项留一行日志/一处 `hasattr` 结果）：
   ```bash
   source tools/dcu/env.sh
   VLLM_LOGGING_LEVEL=DEBUG vllm serve <model> --dtype float16 --max-model-len 8192 --enforce-eager \
     2>&1 | tee /tmp/serve_paths.log
   grep -E "Using backend|TRITON_ATTN|ROCM_ATTN|aiter|skinny|wvSplit|flash" /tmp/serve_paths.log
   ```
2. 对每个"必须关"的路径，确认日志中**没有**它被选中；对"必须开"的路径，确认被选中。
3. 把结论（开/关 + 证据行）写入 `docs/hygon_z100.md`，必要时把小配置固化到 `tools/dcu/env.sh`。

## 验收（门禁）

1. 8K 上下文请求成功且日志为 `TRITON_ATTN`；
2. 日志中不出现 aiter / hipBLASLt / skinny GEMM / flash-attn 被选中的痕迹；
3. W8A8 INT8（若属验收范围）用自研 sdot4 跑通一次，并与生产基线对比记录；
4. `on_gfx906()/on_cdna()/on_gfx9()` 真机输出为 `True True False`。

## 失败处置

| 现象 | 处置 |
|---|---|
| 日志显示 `ROCM_ATTN` | 检查 backend 排序分支是否被改动；确认 gfx906 gate 生效 |
| `_rocm_C has no wvSplitK` | skinny GEMM 未排除干净 → 回 [S5](z100-05-csrc-kernels.md) 对齐 CMake 与运行期判据 |
| GDN 相关报 `not implemented` | 确认该 op 的 gfx906 内核已注册（`csrc/torch_bindings.cpp`）且未被 `VLLM_DISPATCH_HALF_TYPES` 之外的类型调用 |

## 参考

- [hygon_z100.md](../hygon_z100.md)（架构/bf16/环境）
- 前序计划 §5.3「可以开/必须关的 gfx9 路径」
