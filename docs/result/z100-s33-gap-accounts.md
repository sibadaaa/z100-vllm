# S33 WP0：核间空隙分账 —— 结果（2026-10-26）

> 施工依据：`docs/design/z100-s33-fp8-scope-construction.md` §WP0（设计 D1，
> 只测量，不改任何代码）。现役臂 s30c ctrim（`LAUNCH_VARIANT=ctrim`），
> 同臂热身 e2e：**step p50 = 29.331 ms**、热 TTFT 720–724 ms、27.2 t/s
> （`.s33/s20_s33-gap_*.json`，落在基线 29.24–30.05 带内）。
> 产物：`.s33/gap_accounts.json`、本档。分账脚本：`.s33/gap_accounts.py`。

## 1. 口径

- **窗口**：torch profiler Chrome trace 的 CPU `execute_context` decode
  annotation（piecewise graph 臂下跨整 round），20 rounds，rank0，
  `s33gap_rank0.1793011319106075196.pt.trace.json.gz`（4 rank 全落盘）。
- **busy**：窗口内全部 GPU kernel/memcpy/memset 时长的**区间并集**
  （多流重叠去重、跨界裁剪）。kdiff 全表按 (name,grid) 求和会把跨窗核重复
  计入（s30cm1b 表求和 30.74 ms > wall 29.19），并集口径才是真占用。
- **空隙**：窗口内并集补集 = head（窗起→首核）+ 核间隙（1023 处）+ tail（≈0）。

## 2. 分账（插桩 trace，s33gap）

| 项 | ms/round | 占 round |
|---|---|---|
| wall（插桩窗口 p50） | 30.489 | 100% |
| GPU busy（并集） | **24.863** | 81.5% |
| head gap | 0.535 | 1.8% |
| 核间空隙（1023 处 × 均值 5.1 µs） | **5.096** | 16.7% |
| tail gap | 0.000 | 0% |
| **分账覆盖**（head+inter+tail vs wall−busy） | | **100%** ✓ |

nccl 行（本窗）：p50 4.948、min 4.696、stdev 0.249 ms → busy 内 skew wait
≤0.25 ms（本窗低 skew）。边界跨界核 spillover ≈0.03 ms（可忽略）。

## 3. 插桩工件剥离（ITER_DETAILS 佐证）

`ITER_DETAILS=1` 起服 1 rep（未插桩）：引擎侧 decode iteration elapsed
**p50 = 28.63 ms**（mean 28.79，min 28.33，max 31.59，n=87；
`.s33/wp0_iterdetails_summary.txt`）。host 侧 step p50 29.33 → host 步间开销
≈0.7 ms（与 s30c host-gap 结论一致：GPU-bound）。

| 层 | round wall | 差值来源 |
|---|---|---|
| 插桩 trace wall | 30.49 | − |
| 未插桩引擎 round（ITER_DETAILS） | 28.63 | **−1.86 = profiler 工件**（1023 gap × ~1.8 µs，CUPTI 每 launch 开销，全部落在核间空隙里） |
| 未插桩 host step p50 | 29.33 | +0.70 = host 步间开销（引擎外） |

→ **真实核间空隙 ≈ 3.8 ms/round（≈13% round）**；busy 24.86 为硬件真值
（CUPTI 不改 kernel 时长本身）。

## 4. nccl 行跨 trace 方差（勘误①的窗口工件实证）

| trace | nccl ms/round p50 | min | stdev |
|---|---|---|---|
| s30cm1b（03:30，native 臂） | 8.558 | 4.435 | 1.634 |
| s33gap（本次，ctrim 臂） | 4.948 | 4.696 | 0.249 |

同臂类、同 20 步 collect，nccl 均值差 3.6 ms —— 窗口 skew 是 collect 间
显著变量（设计图"nccl 行跨 trace 方差"工件成立）。**busy 里的 nccl 部分
必须用本窗自带 min 去偏后才能跨 trace 比较**；WP2 仲裁窗口选低 skew 窗。

## 5. 与设计图预期对照 & WP3 优先级

- 设计图勘误④："wall 29.48 − GPU busy 27.4 ≈ 2 ms 未分账"。本分账给出更
  完整的口径：busy（并集）24.86 + 真实空隙 3.8 + host 步间 0.7 = 29.3 全闭合；
  设计图的 27.4/2 ms 拆分与本次并存留档（其 busy 来自 kdiff 求和口径，含
  跨窗重复计入，见 §1）。
- **空隙大（3.8~5.6 ms/round、~1000 处 × 5 µs）→ 融合（WP3）升值成立**：
  核数下降直接兑现空隙；AR 核内 norm（消 48+16 c/r 独立 norm 核）+
  gateup epilogue（消 64 c/r silu-mul）是正确的两个落点（勘误③红线不变）。
- WP4 attention 分段的空隙收益次之（16 c/r attention + 16 c/r reduce）。

## 6. 服务生命周期记录（须知 4 实证）

collect 后 /metrics 超时→拒连（12/20 步，引擎退出，已知危害复现）；已按
须知立即重启（s33gap-post）。traces 4 rank 完整落盘后服务才被重启。
后续 WP 起服时 ITER_DETAILS 恢复默认 0。
