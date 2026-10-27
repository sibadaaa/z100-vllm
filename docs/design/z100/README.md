# Z100 设计与施工文档索引

> 适用范围：`docs/design/z100/`。这里只收 **Z100（gfx906 / DTK 26.04 / ROCm 10.1）** 相关的设计与施工文档；
> 上游 vLLM 的设计文档仍平铺在 `docs/design/` 根目录，未做改动。
> 更早的 S1–S18 设计与适配记录在 `docs/archive/design/`、`docs/archive/hygon_z100.md`；实测结果在 `docs/result/`，本索引不重复收录。
>
> 最近整理：2026-10-27。整理动作：按主题分组、路径脱敏、修正内部链接、删除已作废/已合并/被取代的 9 篇文档。
>
> 状态图例：**现行** = 当前轮次仍在用；**参考** = 仍有效的契约/数据依据；**历史** = 已结束轮次，仅存档；**已移除** = 见文末。

## 脱敏约定

文档中的本机绝对路径、用户名、内部服务名和密钥路径已替换为占位符。阅读或复制命令前，请按本机实际情况替换：

| 占位符 | 含义 | 典型替换 |
|---|---|---|
| `$WORKDIR` | 本仓库工作目录 | `export WORKDIR=.../z100-vllm` |
| `$HOME` | 本机用户主目录 | 通常无需显式设置 |
| `$DATA_ROOT` | 模型、缓存、构建产物和数据根目录 | `export DATA_ROOT=/data` |
| `$SECRETS` | API key 等密钥所在目录 | 只读挂载，勿提交 |
| `$SERVICE` | 推理服务的 systemd unit 名 | `systemctl --user status $SERVICE` |
| `$JOB_ID` | 后台任务/日志 id | 按实际任务替换 |

保留未脱敏的技术事实：`gfx906`、`Z100`/`Z100L`、`Hygon DCU`、DTK/ROCm 版本、模型名（`Qwen3.8-27B-FP8` 等）、
开关名、内核符号、md5 指纹。`127.0.0.1:8000` 是 vLLM 默认回环地址，未替换。

## 文档地图

```text
docs/design/z100/
├── platform/          # 平台/硬件事实
├── research/          # 外部项目调研
├── w8a16/             # W8A16 + prefill fp16 物化
├── decode-prefill/    # 2026-10-24 decode/prefill 设计三件套
├── s34/               # 当前轮次：生产 MTP 臂收尾
├── s35/               # 当前轮次：MoE 适配
└── archive/           # 已结束轮次（S24–S33、ROCm 10.1、MMA、P0）
```

## 现行与参考文档

### S34 · 生产 MTP 臂收尾（2026-10-27，现行）

| 文档 | 状态 | 内容 |
|---|---|---|
| [`z100-s34-design.md`](s34/z100-s34-design.md) | 现行 | 生产 MTP k=2 臂的 AR/带宽/长上下文收尾：现状账本、WP 划分、风险表。 |
| [`z100-s34-construction.md`](s34/z100-s34-construction.md) | 现行 | S34 实施任务书（WP1/WP2/WP3/WP6），含基线快照、探针协议与结果归档规范。 |
| [`z100-s34-wp4-ar-fusion.md`](s34/z100-s34-wp4-ar-fusion.md) | 设计待裁决 | 把 add-rmsnorm / fp8 act-quant 融合进 push allreduce 核。 |
| [`z100-s34-wp5-splitk-gemv.md`](s34/z100-s34-wp5-splitk-gemv.md) | 设计待裁决 | fp8 GEMV split-K + 固定段序归约。 |
| [`z100-s34-wp7-int8-ab.md`](s34/z100-s34-wp7-int8-ab.md) | 约束外预案 | INT8 sdot4 W8A8 只做速度 A/B 与 diff 板，不采纳。 |

### S35 · MoE 架构适配（2026-10-27，现行）

| 文档 | 状态 | 内容 |
|---|---|---|
| [`z100-s35-moe-design.md`](s35/z100-s35-moe-design.md) | 现行 | MoE 适配设计：Tier A/B、数值门禁、VGPR 审计、GGUF 内核优化目标。 |
| [`z100-s35-moe-construction.md`](s35/z100-s35-moe-construction.md) | 现行 | S35 实施任务书（WP0–WP9），含环境坑、门禁与产物规范。 |

### 基础与专题（参考）

| 文档 | 状态 | 内容 |
|---|---|---|
| [`z100-bf16-to-fp16-entry-coercion.md`](platform/z100-bf16-to-fp16-entry-coercion.md) | 参考 | gfx906 无原生 BF16 的硬件验证；入口层 bf16→fp16 改造、W8A16 路由与回滚开关。 |
| [`z100-1cat-transfer-research-v2.md`](research/z100-1cat-transfer-research-v2.md) | 参考 | 1Cat-vLLM 代码级调研：569 个提交增量、可移植控制面补丁、对 v1 三处推断的更正。 |
| [`z100-w8a16-prefill-fp16-construction.md`](w8a16/z100-w8a16-prefill-fp16-construction.md) | 参考（合并主文档） | W8A16 + prefill fp16 按层物化的统一施工图：契约、内核改动、门禁、回滚。 |
| [`z100-w8a16-rocm101-port-plan.md`](w8a16/z100-w8a16-rocm101-port-plan.md) | 历史（已执行） | W8A16 三件套移植到 ROCm 10.1 树的施工图与移植边界。 |
| [`z100-dual-prefill-reconciliation-projection.md`](decode-prefill/z100-dual-prefill-reconciliation-projection.md) | 参考（数据依据） | 拆解 DTK/ROCm 两条 prefill 臂的 step/accept 账，给出 decode/prefill 投影与三个决定性 A/B。 |
| [`z100-decode-prefill-optimization-design.md`](decode-prefill/z100-decode-prefill-optimization-design.md) | 参考（设计图） | 锁定配方下按证据强度排序 7 个漏点，定义阶段门与契约。 |
| [`z100-decode-prefill-optimization-construction.md`](decode-prefill/z100-decode-prefill-optimization-construction.md) | 参考（施工图） | 把设计拆成 P0–P5 工作包，逐包给属主、独占文件、命令、阈值、停止线与回滚。 |

## 历史归档（`archive/`）

| 文档 | 轮次 | 状态 | 内容 |
|---|---|---|---|
| [`z100-mma-synthesis-plan.md`](archive/z100-mma-synthesis-plan.md) | MMA | 历史方案 | 合并 MMA 方案/路线图/1Cat 分析；含 `v_dot8_i32_i4` 实测。decode 投影已被后续实测取代。 |
| [`z100-p0-retest-plan.md`](archive/z100-p0-retest-plan.md) | P0 | 一次性方案 | prefill fp16 物化 A/B + 并发恢复复测。 |
| [`z100-rocm101-perf-construction.md`](archive/z100-rocm101-perf-construction.md) | ROCm 10.1 | 已结束 | ROCm 10.1 性能调优施工图（压缩版）。 |
| [`z100-rocm101-perf-p3-construction.md`](archive/z100-rocm101-perf-p3-construction.md) | ROCm 10.1 P3 | 已结束 | P3 任务卡版施工图。 |
| [`z100-rocm101-perf-p3-optimization-plan.md`](archive/z100-rocm101-perf-p3-optimization-plan.md) | ROCm 10.1 P3 | 已结束 | 基于 kernel 档报告的判读与施工图。 |
| [`z100-s24-l3-dynamic-k-design.md`](archive/z100-s24-l3-dynamic-k-design.md) | S24 | 设计稿，未实现 | 长上下文动态降 k（speculative verify 截断）。 |
| [`z100-s24-l4-kv-quant-design.md`](archive/z100-s24-l4-kv-quant-design.md) | S24 | 设计稿，未实现 | KV cache fp8 量化与读路径 dequant。 |
| [`z100-s29-construction-accept-lottery-smallm-bm16-w4.md`](archive/z100-s29-construction-accept-lottery-smallm-bm16-w4.md) | S29 | 已完成 | accept 彩票根因 → SMALLM 复核 → BM16 → W4 探针。 |
| [`z100-s30-decode-optimization-plan.md`](archive/z100-s30-decode-optimization-plan.md) | S30 | 历史计划 | decode step 89.8 ms 的剩余路线。 |
| [`z100-s30b-pure-decode-plan.md`](archive/z100-s30b-pure-decode-plan.md) | S30b | 历史计划 | 纯 decode（无 DFlash）基线与优化计划。 |
| [`z100-s30b-construction-gdn-m1-triton-gemv-hostgap.md`](archive/z100-s30b-construction-gdn-m1-triton-gemv-hostgap.md) | S30b | 被 S30c 取代 | GDN M=1 Triton 并行化 → fp8 GEMV 调参 → host gap。 |
| [`z100-s30c-pure-native-design.md`](archive/z100-s30c-pure-native-design.md) | S30c | S34 基线 | 纯 decode 原生路线（csrc 解禁）+ 显存最小化设计。 |
| [`z100-s30c-pure-native-construction.md`](archive/z100-s30c-pure-native-construction.md) | S30c | S34 基线 | 上述设计的实施任务书。 |
| [`z100-s31-ceiling-gap-plan.md`](archive/z100-s31-ceiling-gap-plan.md) | S31 | 历史计划 | 对齐天花板。 |
| [`z100-s32-fp8-scope-plan.md`](archive/z100-s32-fp8-scope-plan.md) | S32 | 历史计划 | fp8 约束下的剩余优化空间。 |
| [`z100-s32-unconstrained-plan.md`](archive/z100-s32-unconstrained-plan.md) | S32 | 历史储备 | 全开放优化方案（nvfp4 / spec 路线储备）。 |
| [`z100-s33-fp8-scope-design.md`](archive/z100-s33-fp8-scope-design.md) | S33 | 被 S34 继承 | fp8 约束内对齐天花板设计。 |
| [`z100-s33-fp8-scope-construction.md`](archive/z100-s33-fp8-scope-construction.md) | S33 | 被 S34 继承 | 上述设计的实施任务书。 |

## 已移除文档

以下 9 篇已从工作区删除。删除依据：文首自述“已作废/已合并”，或已被后续轮次的文档取代。它们曾由 git 跟踪，可按需恢复。

| 原路径 | 删除原因 | 替代文档 |
|---|---|---|
| `docs/design/z100-1cat-transfer-research.md` | v2 明确更正了 v1 的三处推断，v1 仅作历史 | [`z100-1cat-transfer-research-v2.md`](research/z100-1cat-transfer-research-v2.md) |
| `docs/design/z100-mma-plan-projection.md` | 文首标注 **已作废**：decode 估算用错基线 | 见 `archive/z100-mma-synthesis-plan.md` 与后续 S29–S34 实测 |
| `docs/design/z100-projection-correction.md` | 对已作废投影的纠偏便签，结论已由后续实测覆盖 | 同上 |
| `docs/design/z100-fp8-emu-fp16-conversion-plan.md` | 文首标注 **已合并** | [`z100-w8a16-prefill-fp16-construction.md`](w8a16/z100-w8a16-prefill-fp16-construction.md) |
| `docs/design/z100-fp8-w8a16-implementation.md` | 文首标注 **已合并** | 同上 |
| `docs/design/z100-fp16-prefill-workspace-evaluation.md` | 文首标注 **已合并** | 同上 |
| `docs/design/z100-fp8-emu-optimization-plan.md` | S23 执行计划，已被 S24 及之后的 S29–S34 轮次取代 | `archive/` 中的 S29–S34 文档 |
| `docs/design/z100-longctx-vram-optimization-plan.md` | S24 执行计划，已被 S29–S34 轮次取代 | `archive/` 中的 S24 设计与 S29–S34 文档 |
| `docs/design/z100-opt-round3-construction.md` | 第 3 轮施工方案，已被后续轮次取代 | `archive/` 中的 S29–S34 文档 |

恢复方式（删除尚未提交时）：

```bash
git show HEAD:docs/design/<原文件名>.md > /tmp/<原文件名>.md
```

删除已提交后，可从历史中找回：

```bash
git log --all --diff-filter=D -- docs/design/<原文件名>.md
git show <删除提交>^:docs/design/<原文件名>.md > /tmp/<原文件名>.md
```

## 维护约定

- 新文档按主题放进对应子目录；跨轮次的一次性计划放 `archive/`，不要继续堆在 `docs/design/` 根目录。
- 文档内引用结果文件时使用相对路径指向 `docs/result/`（本目录下是 `../../../result/`）。
- 新增/移动文档后，更新本索引的表格；删除文档时在“已移除文档”表登记原因和替代项。
- 功能入口脚本（`.s34/wpe_probes.sh`、`.s34/probe_c4.py`、`runs/specbench/build_prompt_ids.py`）已同步到新路径；
  历史脚本与源码注释中的旧路径未逐一改写，以本索引为准。
- 本目录文档已脱敏；提交前用下面的命令自检：

```bash
grep -rnE '(/home|/mnt)/[a-z]|z100-vllm-hos[t]|ly[h]' docs/design/z100/ || echo clean
```
