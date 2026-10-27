# Z100 ROCm 10.1 性能调优——P0–P2 判读（VERDICT）

> 施工依据：[`docs/design/z100-rocm101-perf-construction.md`](../design/z100-rocm101-perf-construction.md) §6.4/§6.5
> 日期：2026-10-24；run 目录：`/mnt/game/port/perf-runs/20261024-*-E{1,2,3,4}`
> 信任基线：[`z100-rocm101-perf-baseline-freeze.md`](z100-rocm101-perf-baseline-freeze.md)（B1–B4 未重跑）

## 1. 一页结论

| 臂 | 树 | 开关 | 512c1 TTFT p50 | 512c1 step p50 | 512c1 TPOT | 接受长度/率 |
|---|---|---|---|---|---|---|
| B1（冻结） | rocm101 docker | 无 w8a16 | 4937 ms(709) | 192.2 ms | 37.1 ms | 5.29 / 57.5% |
| B3（冻结） | rocm101 host | W8A16=1 | 2393.7 ms（手动） | 238–265 ms | — | 不可比 |
| **E1** | T2 | W8A16=**0** | 5536 ms | 269.0 ms | 60.0 | 4.65 / 52.1% |
| **E2** | merged | W8A16=**1** | **2722 ms** | **240.4 ms** | 45.7 | 5.36 / 62.3% |
| **E3** | merged | W8A16=**0** | 5599 ms | 269.1 ms | 59.6 | 4.65 / 52.1% |
| **E4** | merged | =E2 + O2 + BLOCK_M=16 | 2718 ms | 235.6 ms | 46.7 | 4.73 / 53.2% |

- **E1 decode 269 ms 未回到 192 ms 底线** → "W8A16 拖慢 decode"假设不成立（判读树走 E2 分支）。
- **merge 有真实改善**：decode 269→240 ms（−11%）、TTFT 5536→2722 ms（−51%），但 decode 仍 >210 ms。
- **E4（O2 + attention BLOCK_M=16）仅 −2.0%**（<5% 阈值）→ 按施工图停止组合调参，不拆更多臂。
- **709c1**：E2 TTFT 3743 ms（B1 同口径 4937、DTK 1595）；8k1/512c4 无塌陷（step 242/385 ms）。

## 2. decode 主因 / prefill 主因

**decode 主因**：不是 W8A16（E1/E3 关掉没改善）、不是 S23 后调优缺失（merge 已带入，E2 改善 11%）、
不是 O2/attention BLOCK_M（E4 只 −2%）。剩余主因指向 **host rocm101 栈本身**：
B1 同代码跑在 **docker** 栈上是 192.2 ms，host 栈上（E1–E4 全部）236–270 ms。
下一步（P3 深挖，kernel 工程师）：rocprofv3/torch-profiler 抓 decode step 的 kernel 级 trace，
对比 docker B1 镜像与 host 栈的同一 kernel（`fp8_emu_gemm_*`、attention、launch gap），
并复测时钟/驱动路径差异；如需再对比 B1 镜像内直接起服（docker RUNTIME）一次。

**prefill 主因**：物化路径生效后 512-token prefill ≈ 2710 ms（E2），仍 >2.2 s：
rocBLAS fp16 GEMM 在 ROCm 10.1 的形状选择/库路径（P3.2：`VLLM_DCU_FP16_PREFILL_MIN_M`
单旋钮微调 + `ROCBLAS_TENSILE_LIBPATH`/hipBLASLt 检查）仍是待办；Must ≤2.5 s 差 9%。

## 3. 生产候选

**E2：merged（`rocm101-perf` @ `03343efb1`）+ `VLLM_DCU_FP8_W8A16=1` + `VLLM_DCU_FP16_PREFILL=1` + `VLLM_DCU_BF16=1`**。

理由：decode/TTFT 双项最优（E4 增益 <5% 且多两个旋钮、512c4 TTFT 反而恶化 10120 vs 8579 ms）；
G0–G4 全绿（stack_identity PASS、op 注册含 w8a16、`W8A16 engaged=True`、
`first materialization route engaged`、`' Paris.'`）。

**Plan B（W8A16=0）按施工图预期不成立，需用户决策**：施工图 §1.2/§7.1 假设
"W8A16=0 + FP16_PREFILL=1 保留 TTFT 收益"，但代码层 `gfx906_fp16_prefill.try_prefill`
要求 W8A16 路由接通（`apply_input_quant = not _W8A16_ENGAGED`，gfx906_fp8.py:867），
`W8A16=0` 时物化必然关闭——E1/E3 实测 TTFT 回到 ~5.5 s（0 次命中）。
若要复活 Plan B，需要一个小型解耦改动（materialization 不依赖 W8A16 路由）+ 重编 `_rocm_C`
无关（纯 Python）+ 一次复测；是否立项请用户拍板。

## 4. Must 达标情况（512/128 c1，E2/E4）

| 指标 | Must | E2 | E4 | 判定 |
|---|---|---|---|---|
| TTFT p50 | ≤2.5 s | 2722 ms | 2718 ms | ✗（差 ~9%） |
| decode step p50 | ≤210 ms | 240.4 | 235.6 | ✗（差 ~12%） |
| TPOT p50 | ≤45 ms | 45.7 | 46.7 | ✗（差 ~4%） |
| 接受长度 | ≥4.0 | 5.36 | 4.73 | ✓ |
| 接受率 | ≥45% | 62.3% | 53.2% | ✓ |
| 709 TTFT | ≤2.5 s | 3743 ms | 3738 ms | ✗ |
| 8k/c4 无塌陷 | — | 242/385 ms | 242/445 ms | ✓（step 量级正常） |

长上下文（8k1）接受长度 2.0、接受率 14.3%——投机解码在长 ctx 下塌陷，单独记录（P3.4 候选）。

## 5. 出口与下一步

- E1/E2/E3（+触发的 E4）各 4 组 summary 完整：`probe-*/http_summary.json`、`case_metrics.json`。
- 本文档即 §6.5 要求的一页 verdict。
- 下一步选项（按施工图 P3/P4）：
  1. **P4 最终验收**（生产候选 E2 + prefix caching 复验 + G5–G8 + 稳定性 + 回滚）——Must 未全绿，
     按施工图 P4 门禁 G5/G6 会卡 TTFT/step 两项；建议先做下面的深挖再验收；
  2. **P3 深挖 decode**（kernel trace，docker vs host 栈对比）；
  3. **P3.2 prefill**（rocBLAS 定界，单旋钮）；
  4. **Plan B 解耦**（小改动，需用户批准）。
