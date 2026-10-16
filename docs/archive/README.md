# docs/archive/ — 历史文档归档

本目录保存 **2026-10-13 之前** 的 Z100 (gfx906) 适配/优化叙述性文档。它们**不再是事实来源**：
其中的结论有的已被 2026-10-13 的三层归因测试报告推翻或修正（例如"INT8 更低精度可提速"、
"decode 步内 GPU 空闲" 等中间结论已在报告 §4.1/§9 作废）。

新的入口是 [`docs/perf/`](../perf/README.md)；证据基线是
`/home/lyh/vllm-test/results/`（PERF_REPORT.md、step_breakdown.md、pmc_findings.md）。

## 归档内容

| 路径 | 原位置 | 说明 |
|---|---|---|
| `hygon_z100.md` | `docs/hygon_z100.md` | Z100 适配主文档（S1–S15 叙述） |
| `z100-todolist.md` | `docs/z100-todolist.md` | 分步任务清单与状态列 |
| `海光Z100-*.md`（4 份） | `docs/` | 性能优化方案、生产部署配方、DTK 适配方案、优化与算子开发全记录 |
| `design/z100-*.md`（17 份） | `docs/design/` | 各步骤设计文档 |
| `result/*.md`（46 份） | `docs/result/` | 各步骤执行归档 + 原 README |

## 归档规则

1. 只读：不再修改，新结论写进 `docs/perf/`。
2. 需要引用其中某个实测数字时，必须先在当次实验里复现，或在引用处标注"历史结论，未复现"。
3. 上游 vLLM 的 `docs/design/*.md`（非 `z100-*`）**不属于**本归档，仍属主文档树。
