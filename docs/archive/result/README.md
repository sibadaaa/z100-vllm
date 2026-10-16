# 步骤执行结果归档（docs/result/）

本目录存放 **Todolist 每一步由子 agent 执行后的结果归档**。
Todolist 见 [`docs/z100-todolist.md`](../z100-todolist.md)，分步做法见 [`docs/design/`](../design/)。

## 归档规则

1. **一步一档**：每个 Todolist 步骤（S0–S10）由**一个子 agent** 独立执行；该子 agent 结束时
   **必须**写出 `docs/result/z100-SN-<slug>-result.md`，未写归档视为该步未完成。
2. **命名**：`z100-<步骤号>-<与设计文档同名的 slug>-result.md`。
   例：`z100-01-arch-normalization-result.md`（对应 `docs/design/z100-01-arch-normalization.md`）。
3. **真实证据优先**：归档中的命令、输出、计数必须是子 agent 实际执行得到的原文（可截断长输出，
   但不得改写、不得凭记忆编造）。无法执行的项要写明"未执行 + 原因"。
4. **一步可以有多档**：若某步因阻塞条件被拆成多次执行（例如 S3 先做离线判定、后做真机判定），
   后续轮次写 `...-result-<序号或日期>.md`，并在文件头部写清与前一档的关系。
5. **只增不改**：归档文件写出后不再修改结论；有新结论时新增一份并在其中引用旧档。
6. **主 agent 只读结论**：Lead 依据归档 + 门禁复核更新 `docs/z100-todolist.md` 的状态列与
   「执行记录」小节，不把子 agent 的中间过程带回主上下文。
7. **写作用域**：子 agent 只写自己的归档文件与其任务书声明的源码/测试文件；
   **不修改** `docs/design/*`、`docs/z100-todolist.md`、本文件。

## 归档模板（子 agent 必须按此结构书写）

```markdown
# <步骤号> <步骤名> — 执行结果

> 归档时间：YYYY-MM-DD
> 执行者：子 agent（<name>）
> 对应设计文档：docs/design/z100-<NN>-<slug>.md
> 前一档：无 / docs/result/...
> 状态：✅ 完成 / 🔶 部分完成 / ⛔ 阻塞

## 1. 目标与验收标准
（本步门禁原文 + 本次执行的验收标准）

## 2. 实际改动
| 文件 | 位置 | 改动摘要 |
（含行号；无改动的步骤写"无代码改动"）

## 3. 执行过程与原始证据
（命令 + 关键原始输出；长输出截断处标注）

## 4. 门禁判定
（逐条门禁 → 通过/不通过 + 证据指向）

## 5. 未完成项 / 阻塞与原因
（明确"未执行 + 原因"，含环境限制）

## 6. 对后续步骤的影响
（哪些步骤可以开始、哪些前提被推翻）

## 7. 复现方式
（一条条可直接粘贴的命令）

## 8. 参考
```

## 归档索引

| 步骤 | 归档文件 | 状态 | 归档时间 |
|---|---|---|---|
| S0 | 待补（前期工作已记入 todolist 执行记录） | — | — |
| S1 | [z100-01-arch-normalization-result.md](z100-01-arch-normalization-result.md) | ✅ 门禁通过（GPU-free；真机复核移交 S9） | 2026-10-07 |
| S2 | 待补（前期工作已记入 todolist 执行记录） | — | — |
| S3 | [z100-03-bf16-verification-result.md](z100-03-bf16-verification-result.md) | ✅ 门禁 PASS（真机容器判定：bf16 保留、默认 `auto`） | 2026-10-07 |
| S4 | [z100-04-build-system-dtk-result.md](z100-04-build-system-dtk-result.md) | 🔶 部分通过（`_rocm_C` ✅ 只含 gfx906；`_C` 被 DTK rocPRIM 缺陷阻塞 → S5） | 2026-10-07 |
| S4b | [z100-04b-stack-upgrade-result.md](z100-04b-stack-upgrade-result.md) | ✅ 门禁全过（4 扩展只含 gfx906；S5 阻塞消失；triton 3.6 轮子仍不支持 gfx906 → S6） | 2026-10-07 |
| S4c | [z100-04b-stack-upgrade-result-2.md](z100-04b-stack-upgrade-result-2.md) | ✅ S4b 第二档：两项修复落地 + 全量 ninja rc=0、两次构建逐字节一致 | 2026-10-07 |
| S4d | [z100-04d-build-incrementality-result.md](z100-04d-build-incrementality-result.md) | ✅ 增量重建修好（no-op 0.085 s、单 TU 40 s、产物与基线逐字节一致） | 2026-10-07 |
| S5 | [z100-05-csrc-kernels-result.md](z100-05-csrc-kernels-result.md) | 🔶 编译器侧全达成（`_C` 78 MB / `_rocm_C` 35 MB，均只含 gfx906）；测试 66/16 → 14 项由 S6 解除 | 2026-10-07 |
| S6 | [z100-06-triton-gfx906-result.md](z100-06-triton-gfx906-result.md) | ✅ 门禁全过（探针 + 39/12 attention 测试；14/14 `unsupported target` 消除；含 SFU hazard 修复） | 2026-10-07 |
| S7 | [z100-07-memory-kvcache-graph-result.md](z100-07-memory-kvcache-graph-result.md) | ✅ 门禁过（KV cache 27.18/26.75 GiB；graph 51/51+35/35；decode 2.98×）；**发现 `--enforce-eager` NaN → S7b** | 2026-10-07 |
| S7b | [z100-07b-eager-nan-result.md](z100-07b-eager-nan-result.md) | ✅ 根因定位到 `_C.silu_and_mul` 的 fp16 packed 分支；Python 侧修复后 eager == compile == HF | 2026-10-07 |
| S7c | [z100-07c-packed-silu-csrc-fix-result.md](z100-07c-packed-silu-csrc-fix-result.md) | ✅ csrc 根修（MoE 直调点 1907→0 非有限；新单测锁回归；修的是 `_C_stable_libtorch`） | 2026-10-07 |
| S8 | [z100-08-attention-and-quant-paths-result.md](z100-08-attention-and-quant-paths-result.md) | ✅ 8K → `TRITON_ATTN`；aiter/hipBLASLt/skinny/flash-attn 均未进入；量化路径逐项核对；**另修 2 个 fork 真 bug** | 2026-10-07 |
| S8b | [z100-08b-gating-and-requirements-result.md](z100-08b-gating-and-requirements-result.md) | ✅ 门禁 16/16 + `s8b_gate.sh` ALL GREEN；FP8 block-scaled 数值干净（慢 36×，不硬禁用）；W4A16 选择测试 2 failed → 3 passed | 2026-10-07 |
| S9 | [z100-09-acceptance-and-bench-result.md](z100-09-acceptance-and-bench-result.md) | ✅ A1–A6/A8/A9 通过、A7 仅 GPU-free；27B INT8 + W4A16 端到端通过（三项受宿主限制，见归档） | 2026-10-07 |
| S9b | [z100-09b-full-cudagraph-result.md](z100-09b-full-cudagraph-result.md) | ✅ 修 `pad_for_cudagraph` 漂移、FULL 图恢复、GDN 4B 跑通；纠正性能归因（3.09× 在 int8/GDN 内核与 TP=4） | 2026-10-07 |
| S10 | 待补 | — | — |
| S11 | [../海光Z100-性能优化方案.md](../海光Z100-性能优化方案.md)（方案本身） | 🔶 方案已出，P0–P5 逐步执行 | 2026-10-08 |
| S11a | [z100-11a-a7-non-dtk-result.md](z100-11a-a7-non-dtk-result.md) | ✅ A7 六步全做：两种环境 pytest 52 passed×2、plugin/setup.py 一致；**发现 F1（已由 Lead 修）** 与 F2（保留并记录） | 2026-10-09 |
| S11b | [z100-11b-perf-profiling-result.md](z100-11b-perf-profiling-result.md) | ✅ 零重跑（抢救 29 MB×2 trace + 独立重算）：**GEMM 97.3% / GDN 0.4% / CPU 3.3%** ⇒ 只做 P2；并改正 P2 理由（64× M padding，非缺 dot4） | 2026-10-09 |
| S11c | 待补（P1：TP 缩放曲线 + 内存绕法） | — | — |
| S11d | [z100-11d-z100-isa-capability-audit-result.md](z100-11d-z100-isa-capability-audit-result.md) | ✅ 算力四列表 + Triton codegen 实测 + 机器码复用点；**挖出前序 `int8_sdot4` 资产**（Lead 复跑 21/21 MATCH） | 2026-10-08 |
| S11e | [z100-11e-w8a8-sdot4-result.md](z100-11e-w8a8-sdot4-result.md) | 🔶 内核/数值/性能全过（63/63 rel_err ≤3e-04、与 Triton **逐位相同**、decode 形状 ×4.9–22.6、no shape regresses）；**端到端被 `torch.compile` 的 `ConstraintViolationError` 否决 → S11k** | 2026-10-09 |
| S11k | [z100-11k-sdot4-compile-integration-result.md](z100-11k-sdot4-compile-integration-result.md) | ✅ **P2 落地**：`custom_op`+`register_fake` 修好 compile；**decode 3.9×（84.07/82.74 vs 329.05/320.30）**，已越过生产基线 122.47；文本在并列点翻转（数值性） | 2026-10-09 |
| S11l | [z100-11l-sdot4-reacceptance-result.md](z100-11l-sdot4-reacceptance-result.md) | ✅ 重验收：7 文件 129/0（=0 一致）；**隔离实验洗清内核**（扰动来自 custom_op 的图变化）；同口径 **84.00 vs 321.09（3.82×）、快生产基线 1.458×**；但发现 `=1` 跨重启不可复现 → S11m | 2026-10-09 |
| S11m | [z100-11m-aot-restart-reproducibility-result.md](z100-11m-aot-restart-reproducibility-result.md) | ✅ AOT 重载真因（缺尾换行口径不一致）+ 修复（54.15→19.5 s）；`=1` 热启动 3/3 逐位一致；**修正：token 变体由 vintage 决定而非开关（同会话 `=0`/`=1` 仅差 2/512）**；146 passed | 2026-10-09 |
| S12 | [z100-12-p2p-acs-impact-result.md](z100-12-p2p-acs-impact-result.md) | ✅ **NCCL 两态完全一致**（RCCL 全走 SHM、0 条 P2P）；27B TP=2 差异 ≤0.7%；TP=4 不可行（4B 曲线 TP2→4 仅 1.07×）；公平基线：我们 TP=2 快生产栈 1.43–1.56×；顺带修掉 `revert` 的真 bug | 2026-10-09 |
| S13 | 待补（去掉 `--disable-custom-all-reduce` 的收益测评 + 8K prefill / vendor 重复） | — | — |
| S11f | [z100-11f-fp8-bf16-emu-kernels-result.md](z100-11f-fp8-bf16-emu-kernels-result.md) | 🔶 内核级完成：FP8 W8A8（LDS LUT + dot2）与 bf16/fp16 decode GEMV；`v_fmac_f32`=0、FP8 sweep 384/384、28 passed；FP8 decode 1.6–5.5×、dense decode 1.3–5.7×（≤0.39 ulp）；**端到端 / block-scaled 未做** | 2026-10-11 |
| S11e2 | [z100-11e2-int8-family-integration-result.md](z100-11e2-int8-family-integration-result.md) | ✅ INT8 `sdot4` 并入内核族：per-tensor/per-channel × 动态/静态 scale 契约、族总开关 `VLLM_DCU_USE_GFX906_EMU`、新单测 28 项；56 passed、`--mode int` 全 bit_equal、`--mode check` 2.988e-04 与 S11e 一致 | 2026-10-11 |
| S11g | [z100-11g-1cat-vllm-reference-analysis-result.md](z100-11g-1cat-vllm-reference-analysis-result.md) | ✅ 7/7 条 `file:line` 证据；三条结论已折进方案 §4C.1（Lead 修正其一：271→95 s 出处是 `README.md:1303-1308`） | 2026-10-08 |
