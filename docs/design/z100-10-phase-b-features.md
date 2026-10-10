# S10 阶段 B：GGUF / MoE / MTP / 多模态对齐

> 上层文档：[适配方案](../海光Z100-DTK26.04-gfx906-vLLM适配方案.md) · [Todolist](../z100-todolist.md)
> 前置：[S9](z100-09-acceptance-and-bench.md)（阶段 A 验收通过） · 状态：⬜ 待做

## 目的

把 fork README 里已验证过的能力矩阵在 Z100+DTK 上重新跑通，并明确哪些项**确定不支持**（不要浪费验证预算）。

## 目标矩阵

| 能力 | 相关代码 | 备注 |
|---|---|---|
| GGUF dense（27B 级） | `csrc/quantization/gguf/{gguf_kernel.cu,mmvq.cuh,vecdotq.cuh}`、`layers/quantization/gguf.py`、`config/vllm.py` 的 `_is_gfx906_gguf` 系列 | 需要 DTK clang 支持 gfx906 的 dot2/dot4 系内建 |
| Qwen3.5 MoE GGUF | `_is_gfx906_qwen35_gguf`、`fused_moe` 相关、`csrc/libtorch_stable/moe/*` | MoE+GGUF 组合历史上坑最多 |
| MTP 投机解码 | `csrc/mamba/gfx906_decode_kernels.cu` 的 MTP update；`config/vllm.py` 的 async scheduling 开关 | 见下方硬约束 |
| 多模态 mmproj | 视觉配置 / 权重名映射 / Conv3d patch embed 合并 / mm 输入链路 | 显存 32 GiB 下可能必须 `--skip-mm-profiling` |
| 已知不支持 | MoE + GPTQ-Int4、MoE + AWQ-4bit | 直接标注不支持，不排验证 |

## 硬约束（先读）

1. **MTP / GDN 内核是 fp16-only**：`csrc/mamba/gfx906_decode_kernels.cu` 中 `causal_conv1d_gfx906_mtp_update`、`fused_sigmoid_gating_delta_rule_gfx906_mtp_v2/mtp_update` 使用 `VLLM_DISPATCH_HALF_TYPES`（仅 Half）。
   ⇒ 该路径上 bf16 会以明确错误失败（不会静默算错）。要支持 bf16 必须**主动扩展派发**，这属于本步骤的决策项，不要顺手改。
2. **GGUF 与 async scheduling**：`vllm/config/vllm.py` 的 `_auto_enable_async_scheduling_for_gfx906_gguf` 与 PIECEWISE graph 安全护栏是 fork 专门加的，别绕过。
3. **`is_hygon_dtk()` 的既有约束**：所有 DTK 兼容分支必须保留可回退性（见方案 §7）。
4. **不要动 `on_gfx9()`/`_ON_GFX9`**（fork 有意排除 gfx906）。

## 做法（逐项闭环，每项都要有门禁）

### 通用流程

```bash
source tools/dcu/env.sh
# 每项都用同一套脚本跑：起服务 → 功能请求 → 记录 decode ms/token 与输出质量
python3 tools/dcu/smoke.sh <model> <extra-args>
```

1. **GGUF dense**：先 `gguf_kernel.cu` 单文件试编（**只编译不运行**）；通过后跑 `tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py`；再起服务做端到端。
2. **Qwen3.5 MoE GGUF**：确认专家权重加载路径与 `_is_gfx906_qwen35_gguf` 命中；先单卡小 `--max-model-len`，再放大。
3. **MTP**：先用 fp16 验证投机解码出词与接受率；如需 bf16，先决定是否扩展 `VLLM_DISPATCH_HALF_TYPES`（若扩展，同步补数值对齐测试）。
4. **多模态**：先按纯文本确认主链路，再启用 mmproj；显存不足时先试 `--skip-mm-profiling`，并在文档写明该 flag 的语义与副作用。
5. **已知不支持项**：在 `docs/hygon_z100.md` 与 README 明确列出，避免后续重复投入。

## 验收（门禁）

1. 上表"目标矩阵"中**每一项**都有结论：通过（附命令与日志）或不支持（附失败证据）；
2. 每项通过后回归 [S9](z100-09-acceptance-and-bench.md) 的 A3–A5（单卡/长上下文/TP）不劣化；
3. README（zh/en）已更新 Z100 支持矩阵，明确标注未测项与不支持项；
4. 新增/修改的 csrc 与 Python 分支均满足"无 DTK 时行为不变"。

## 失败处置

| 现象 | 处置 |
|---|---|
| GGUF 内核报内建不支持 | 记录具体内建（如 dot2/dot4）与文件行号；该 GGUF 变体标记为不支持，走非 GGUF 路线 |
| MTP bf16 报 `not implemented for BFloat16` | 预期；用 fp16 或显式做派发扩展（单独提交 + 数值测试） |
| 多模态 profiling OOM | `--skip-mm-profiling`；仍不行则只交付纯文本能力 |
| MoE 启动即崩 | 先确认是否命中已知不支持组合（GPTQ-Int4 / AWQ-4bit） |

## 参考

- fork README 的"已测试/已知无法运行/未测试"三张清单
- 前序计划 §4 Phase 3（量化路线决策）与 §5.6（量化决策树）
