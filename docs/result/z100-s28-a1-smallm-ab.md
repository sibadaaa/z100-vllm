# S28 A/B：A1（`in_proj_ba` GEMV）与 SMALLM 路由的服务级裁决（2026-10-25）

> **窗口**：15:04–15:40（+08），三臂同窗、同探针、同文档。
> **口径**：`tools/dcu/s20_single_probe.py`，真实 7469-token 文档
> （`docs/design/z100-dual-prefill-reconciliation-projection.md` truncate 7469）、
> max_tokens 96、repeat 5、graph-A 配方（`s27_graph_launch.sh`，MBT=8192、MAX_SEQS=4、TP=4）。
> **代码**：A1 = `_gfx906_gemv_shape_ok` 加 `n_out<=64 → M<=8`（带
> `VLLM_DCU_GEMV_NARROW_MAX_M` 旋钮，kernel 上限 kMsMax=8）；
> 微测依据 `docs/perf/data/a1-ba-gemv-probe.json`（M=8：emu 40.9 µs vs triton 188.6 µs）。
> **服务指纹**：`VLLM_DCU_GEMV_NARROW_MAX_M@raw=4|8`、`VLLM_DCU_SMALLM_GEMM=off|rocblas`
> 均已在 serve log 核实；`MTP_GDN_TRITON=1`、MBT=8192 三臂一致。

## 1. 结果（srv_step p50，5 reps，rep 间波动 <0.3 ms）

| 臂 | 配置 | step p50 | tok/step | accept | 服务端 fwd t/s | Δ vs 臂1 |
|---|---|---:|---:|---:|---:|---|
| 1 control | NARROW_MAX_M=4（改动前） | **108.85** | 2.909 | 1.969 | 26.72 | — |
| 2 **A1** | 默认（=8） | **101.11** | 2.909 | 1.969 | 28.77 | **−7.74 ms（−7.1%）** |
| 3 A1+SMALLM | + `VLLM_DCU_SMALLM_GEMM=rocblas` | 96.46 | **2.667** | **1.771** | 27.65 | step 再 −4.65，但 **accept −8.3%** |

客户端 out t/s：臂1 16.91 → 臂2 17.75（+5.0%）→ 臂3 17.26。

**裁决**：
- **A1 采纳**（默认生效）：step −7.74 ms 落在微测预测区间（−7.1~−8.5）正中；
  tok/step 与 accept 逐位不变；parity 通过（见 §2）。
- **SMALLM=rocblas 拒绝**：draft tower 的 matmul 走 rocBLAS 改变了 draft 数值分布，
  7469 文档上 accept 2.909→2.667（−8.3%），损失 > step 收益。短 prompt（439 tok）
  上 accept 与 token 流均不变——**accept 损伤是长上下文特异**，只有真实文档口径能抓到
  （`draft_m7_sweep` 的"−6.0 ms 最优路由"投影在 e2e 上不成立，e2e 实得 −4.65 ms step 但净亏）。

## 2. Greedy parity（`perf1_e2e_parity.py`，439 prompt × 64 token × 2 reps）

| 对比 | token 流 | top-1 logprob 最大漂移 | 备注 |
|---|---|---:|---|
| 臂1 vs 臂2（A1） | **64/64 相同（2 reps confirmed）** | 0.0236 nat | 2/64 超 0.01 tol，量级远低于 verify-3D 时代的 0.80 近平局类；无翻位 |
| 臂1 vs 臂3（SMALLM） | 64/64 相同 | 0.0317 nat | 短 prompt 无差异；accept 损伤只在长文档 |

臂内确定性：两臂 reps 逐位相同。

## 3. 与基线的关系

昨日同口径基线 srv_step 114.17（`s27/p1/s20_graphA-7469-real_20261024-144727`，
tok/step 2.909 与今日一致）。今日 control 108.85（−4.7%，在记载的 ±5–6% 单 run 噪声带内，
不单独归因）。**A1 的 −7.74 ms 是同窗单变量差值，不受该漂移影响。**

## 4. 状态与遗留

- 最终服务停在 **A1 臂（默认配置）**：`.s28/serve-final-a1.log`，:8000，策略指纹见 log。
- 遗留：`VLLM_DCU_GEMV_NARROW_MAX_M` 加入 `tools/dcu/dcu_policy_fingerprint.py` 的 KNOBS
  （本次以 @raw 记录）；存量测试失败 `test_dense_gemm_dispatch_is_off_by_default`
  （位比较判别器失效）单独修。
- 下一步（按 SOL-GAP §5 路线）：A2（FA BLOCK_M=16，需 parity 板）→ draft attention
  tile 调优（5×1.23 ms，KV 仅 37 MB）→ W4 探针（`docs/perf/W4-RESEARCH-20261025.md`）。

原始 JSON：`.s28/s20_s28-arm{1,2,3}_*.json`、`.s28/e2e-arm{1,2,3}.json/`、
`.s28/parity-arm1-vs-arm{2,3}.json`、serve log `.s28/serve-arm{1,2,3}*.log`。
