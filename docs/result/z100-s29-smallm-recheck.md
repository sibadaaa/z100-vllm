# S29 WP-B：SMALLM=rocblas 复核（accept 彩票钉死后重测）

> 施工图：`docs/design/z100-s29-construction-accept-lottery-smallm-bm16-w4.md` WP-B。
> 背景：S22 建 `VLLM_DCU_SMALLM_GEMM=rocblas` 派发；s28 实测 step 101.1→96.46（−4.65 ms）
> 但 accept 1.969→1.771 被否决。s28d §2 证明 accept 是重启彩票后，该否决证据不足
> （1.771 落在午后对照吸引子集合内）。本 WP 在 **WP-A 钉死 accept 世界**
> （`docs/result/z100-s29-accept-pin.md`：`ROCBLAS_WORKSPACE_CONFIG/:4096:8` +
> `HIPBLAS_WORKSPACE_CONFIG/:4096:8`）的前提下重测。

## 1. 臂与协议

- 控制 = 当前采纳臂（SMALLM=off）+ workspace pin：`s29-pin1/2/3`、`s29-adopted2`
  共 4 次重启 × 3 reps（超出协议要求的 2 次）。
- arm = `VLLM_DCU_SMALLM_GEMM=rocblas`（launcher 不变，env 注入）+ 同一 pin：
  `s29-smlm-arm-r1/r2` 共 2 次重启 × 3 reps。
- 探针口径：7469 真实文档 / 96 max-tokens / 3 reps（§0.3 模板）。
- parity：`perf1_e2e_bench.py --mode parity` 8 个 prompt 长度
  （256/439/512/1024/2048/4096/6144/7168 配置值）× 64 tok × 2 reps，
  `perf1_e2e_parity.py` 对拍。

## 2. 服务级结果（全部在 pin 下，世界选择可复现）

| 臂 | 重启 | accept（3 reps） | tok/step | step p50 (ms) |
|---|---|---|---:|---:|
| control（SMALLM=off） | 4 | 1.969 ×12 | 2.909 | 92.95 / 93.08 / 93.18 / 93.03 |
| arm（SMALLM=rocblas） | 2 | 1.771 ×6 | 2.667 | 90.14 / 89.31 |

- **step p50：−3.4 ms**（中位 89.7 vs 93.06，−3.6%）——方向确认、幅度小于
  s28 的 −4.65 ms（当日基线已含 A1，收益部分重叠）。
- **accept：arm 两个值均低于 control 最小值**（1.771 < 1.969），且臂内
  逐位稳定 —— 在 pin 之下这是**可复现的配置差异**，不是彩票：
  SMALLM=rocblas 给 draft 塔引入的 rocBLAS 状态把世界选择带回了低吸引子。
- 端到端 fwd tok/s：control 2.909/0.09306 = **31.26**，arm 2.667/0.08972 =
  **29.72**（**−4.9%**）。step 收益吃不掉 accept 损失。

## 3. Greedy parity（8 prompt × 64 tok × 2 reps）

`perf1_e2e_parity.py` 总判定 **refuted**（`.s29/parity-smlm-ctl-vs-arm.json`）：

- 7/8 长度 token 流逐位相同（top-1 logprob 漂移 ≤0.137 nat，多为 ≤0.03）；
- **p439 翻位**：首分歧位置 18，`"3 for latency regressions."` →
  `"4 for latency regressions."`，流内 32/64 位置不同，
  max |Δlogprob(top1)| = 1.766 —— 远超近平局（0.05 nat）协议；
- 两臂臂内确定性全部成立（8 长度 × 2 reps 逐位一致），对拍有效。

## 4. 判定

**keep rejected（维持否决）**——按 §2 判据第 2 条原文："若 arm 一致性走低 ⇒
keep rejected，存档"。与 s28 不同的是，本结论不再依赖单点彩票：控制 4 重启
全部 1.969，arm 2 重启全部 1.771，且 **parity refuted（p439 翻位 32 处、
max Δlogprob 1.77，远超 0.05 nat 近平局线）本身就是独立否决依据**
（数值上改变 greedy 输出 = 正确性回归，除非走近平局例外 + 显式拍板）。

**注意（WP-A 后期复核）**：workspace pin 后来被证明非完全钉死
（见 `z100-s29-accept-pin.md` §5 残留），arm 的 {1.771, 1.771} 含有
世界选择的贡献，不能 100% 归因 SMALLM 数值本身。但 parity 翻位证据
与"step −3.4 ms 换不来端到端收益"（即使 accept 取最好世界 1.969，
SMALLM 的 −3.4 ms 也需要 accept 损失 ≤ 0.17 才打平）共同支撑否决。
`VLLM_DCU_SMALLM_GEMM` 保持默认 off；不写 launcher。

微测依据（`docs/perf/data/draft-m7-sweep.json`，除 lm_head 外全形状
rocBLAS 快 1.4–5.6×）依然成立——kernel 更快，但服务级被 accept 抵消。

原始数据：`.s29/s20_s29-smlm-arm-r{1,2}_*.json`、`.s29/s29-smlm-{ctl,arm}_p512_g64_*.json`、
`.s29/parity-smlm-ctl-vs-arm.json`、`.s29/serve-s29-smlm-*.log`。
