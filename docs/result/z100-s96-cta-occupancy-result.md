# S96 结果：CTA 占用率证伪与 issue 预算（2026-11-14）

设计：[`../design/z100/s96-cta-occupancy/z100-s96-cta-occupancy-design.md`](../design/z100/s96-cta-occupancy/z100-s96-cta-occupancy-design.md)
施工：[`...-construction.md`](../design/z100/s96-cta-occupancy/z100-s96-cta-occupancy-construction.md)
台账：[`.s96/README.md`](../../.s96/README.md)

## 结论

1. **出口 C（按图）**：相位局部 dot4 发射槽占比 30.8%（≤40%），
   WP-C（kTile8/c62）免立项；生产仍 `dot2`（`7d2ed2a2`）。
2. **70% 缺口的归因是指令形态，不是 wave 数**：全核 issue 占用 95.2%，
   QK 相位 69% 的发射槽不是 dot4；PV 的真实 dot2 达峰是 44%（S94 的
   22% 用了 2× 错的分母）。
3. **pad2k 证伪了"3→2 CTA 不敏感"**：131k +20.23%（882.44 → 1060.98 ms），
   税集中在 load（+51.7%）。这不推翻出口 C：3 CTA 已贴发射上限，
   4 CTA 只剩 ≤5.3%，且不改变 dot4 的 31% 槽占比。

## 臂表（131k / m=8192 / seg=4 / 9 iters，A/B ×3）

| 臂 | vgpr/spill/LDS/ctas | 131k (ms) | Δ | 数值 |
|---|---|---:|---:|---|
| dot2（基底） | 70/0/20608/3 | 882.442 | — | 基准 |
| pad2k（+2 KB 未用 LDS） | 71/0/22656/2 | 1060.980 | **+20.23%** | 逐位同 |

数值门：prefill 8k `b1fa6e88`、decode 65k M=1 `b2beb5a0` / M=8 `ba0e1cce`。

## issue 预算（`.s96/counters/issue-budget.md`）

- 常量：60 CU × 1.6 GHz = 96.0e9 instr/s；dot4 19.26 TMAC/s = 75.23e9
  instr/s；dot2 19.61 TFLOP/s = 9.805 TMAC/s = 76.60e9 instr/s。
- 字面整环体 ceil = 13.2%（8.4–16.2%），低于实测 → 分母口径错，重做。
- 相位局部：QK dot4 30.8%（实测 30.8%）、PV dot2 44.3%（实测 44.3%）；
  全核 80.68e9 条 / 0.882 s = 95.2% 发射占用；VALU 中 dot 占 24.7%，
  其余 VALU 48.6%、LDS 10.7%。

## prefill 分段（stage op，seg=4，prod dot2）

| ctx | wall | load (GB/s) | qk (TMAC/s) | sm | pv (TMAC/s) | wait |
|---|---:|---:|---:|---:|---:|---:|
| 8k | 105.75 | 22.93 (1141) | 33.96 (4.55) | 0.05 | 46.44 (3.33) | 2.37 |
| 32k | 262.88 | 56.85 (1151) | 84.46 (5.49) | 0.12 | 115.54 (4.01) | 5.91 |
| 131k | 889.78 | 191.07 (1164) | 286.36 (5.94) | 0.40 | 391.90 (4.34) | 20.05 |

占比：load 21.5% / qk 32.2% / pv 44.0% / wait 2.25% / sm 0.05%。
QK 达峰 23.6% → 28.5% → 30.8%（随 ctx 上升，staging 摊薄）；
PV 34.0% → 40.9% → 44.3%（分母是 9.805 TMAC/s，非 19.26）。

pad2k 131k：load 289.83（+51.7%）、qk 323.87（+13.1%）、
pv 443.88（+13.3%）、wait 19.51（−2.7%）⇒ 少 CTA 主要伤全局 KV 读的
延迟隐藏。load 已 1164 GB/s，4 CTA 也换不来带宽。

## 下一步

- WP-C 不立项；生产签名/链接/服务 argv 未动。
- 减指令优先：S92b 判决 4（8→4 bpermute，预估 wall −5~12 ms）；
  再加一项按 disasm 审计"其他 VALU 48.6%"（地址/cvt/softmax）的构成。

## 偏差

- 用户令停服（图写不停服）；微测与 S95 同为停服态，服务收工保持 inactive。
- pad2k vgpr 70→71（保活 +1，2 CTA 下不影响占用）；首版谓词保活被
  LLVM 折叠，改 volatile 写 + opaque 读后 census 达标。
- 未做 counters 二审，用 pad2k stage 分段替代定位。
