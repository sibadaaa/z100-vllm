# S3 bf16 门控与真机判定

> 上层文档：[适配方案](../海光Z100-DTK26.04-gfx906-vLLM适配方案.md) · [Todolist](../z100-todolist.md) · [实测证据](../hygon_z100.md)
> 前置：[S2](z100-02-platform-compat-layer.md) · 状态：✅ 已完成（真机判定 PASS，见 [S3 归档](../result/z100-03-bf16-verification-result.md)）

## 目的

回答一个**必须用数据裁决**的问题：Z100 上是否保留 bf16？

- 生产容器现在就跑 bf16，其 profiler 显示 bf16 rocBLAS GEMM ≈ 8.2 ms/次；
  **（2026-10-07 真机复核：该数字与 4096³ 实测不符——实测 fp16 7.84 ms / bf16 21.54 ms，口径待 S9 用 profiler 复核，见 [S3 归档 §3.7](../result/z100-03-bf16-verification-result.md)）**；
- 但 ISA 层 gfx906 **没有 bf16 指令**（无 bf16 转换、无 bf16 dot），bf16 只能软件模拟；
- 归一化到 `gfx906` 后，fork 的 `_rocm_supported_dtypes()` 会剔除 bf16 —— 这是一次**行为变更**，不能预设结论。

## 事实（已钉死，别重复验证）

| 指令 | gfx906 | 汇编器输出 |
|---|---|---|
| `v_pk_fma_f16` / `v_dot2_f32_f16` | ✅ | OK |
| `v_cvt_f32_bf16` | ❌ | `error: invalid instruction, did you mean: v_cvt_f32_f16?` |
| `v_dot2_f32_bf16` | ❌ | `error: instruction not supported on this GPU` |

复现（**只编译，不要运行产物**）：

```bash
CLANG=/opt/dtk/llvm/bin/clang
printf '\tv_cvt_f32_bf16 v0, v1\n\ts_endpgm\n' > /tmp/t.s
$CLANG -target amdgcn-amd-amdhsa -mcpu=gfx906 -c /tmp/t.s -o /tmp/t.o   # 期望报 invalid instruction
```

两个误判来源：`torch.cuda.is_bf16_supported()` 在 ROCm 上**恒为 True**；`torch.set_default_dtype(torch.bfloat16)` **不是能力开关**（只改新建张量默认 dtype）。

## 已落地的实现（评审对象）

| 位置 | 行为 |
|---|---|
| `hygon_dtk.dcu_bf16_probe()` | 存储往返 / bf16 GEMM vs fp32 参考 / bf16 softmax，全通过才算可用（`lru_cache`） |
| `hygon_dtk.gfx906_bf16_enabled()` | `VLLM_DCU_BF16=auto`(默认,探测) / `1`(强制开) / `0`(强制关) |
| `rocm._rocm_supported_dtypes()` | gfx906：`[fp16, fp32]`，探针/开关通过时在中间插入 `bfloat16`（**fp16 仍排第一**，`--dtype auto` 行为不变） |
| `tools/dcu/check_bf16.py` | 真机数值 + 性能实测，退出码 0/1/2 |

## 做法（真机判定）

```bash
source tools/dcu/env.sh
python3 tools/dcu/check_bf16.py ; echo "exit=$?"
python3 -c "from vllm.platforms.rocm import _rocm_supported_dtypes; print(_rocm_supported_dtypes())"
```

然后与生产基线做 **A/B 对拍**（同一模型、同一 prompt、同一并发）：

| 组 | 参数 | 记录 |
|---|---|---|
| A（fp16） | `--dtype float16` | decode ms/token、TTFT、并发吞吐、输出质量抽查 |
| B（bf16） | `--dtype bfloat16`（必要时 `VLLM_DCU_BF16=1`） | 同上 |

判定规则：
- `check_bf16.py` 退出码 1（数值不可靠）→ **移除 bf16**，在 `docs/hygon_z100.md` 记录依据，并把 `VLLM_DCU_BF16` 默认改为 `0`。
- 退出码 0 且 B 组吞吐不低于 A 组（或模型本身要求 bf16）→ 保留 `auto`。
- 退出码 0 但 B 组明显更慢 → 保留可选（`auto`），并在文档给出"默认仍用 fp16"的建议。

## 验收（门禁）

1. `check_bf16.py` 给出明确 PASS/FAIL 与数值；
2. `_rocm_supported_dtypes()` 的真机输出与判定结论一致；
3. 结论写入 `docs/hygon_z100.md` 与 [适配方案](../海光Z100-DTK26.04-gfx906-vLLM适配方案.md) §2.3/§5（A9）；
4. 现有 GPU-free 单测（`test_gfx906_bf16_enabled`、`test_gfx906_supported_dtypes_keep_fp16_first`）保持通过。

## 失败处置

| 现象 | 处置 |
|---|---|
| 探针抛 `RuntimeError: not implemented for BFloat16` | 说明该 DTK/torch 未提供 bf16 GEMM 内核 → 判定为不可用（退出码 1） |
| bf16 + MTP/GDN 报 `not implemented for BFloat16` | **预期行为**：gfx906 的 MTP 内核走 `VLLM_DISPATCH_HALF_TYPES`（仅 fp16）；阶段 B 再决定是否扩展派发 |
| 数值 PASS 但输出乱码 | 记入 A/B 结果，按"质量抽查"一票否决 |

## 参考

- [hygon_z100.md](../hygon_z100.md) §2（bf16 专项）
- 前序计划 §5.2 提醒 3（bf16 A/B 项）
