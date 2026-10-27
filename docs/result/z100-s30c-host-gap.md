# S30c WP3 结果：host gap 削减 —— 出表停手（2026-10-26）

> 施工依据：`docs/design/z100-s30c-pure-native-construction.md` §WP3。
> 判据：合计 step p50 **−1 ms 以上** + 贪心流零翻转；单项 <0.3 ms 不采纳；
> **落空则如实出表停手，不得为凑数引入数值风险**。

## 1. 前提复核（WP3 的 3 ms 前提在 WP1-N 之后不再成立）

host gap = wall step − GPU busy。两份 trace 的全 kernel 聚合
（`kdiff-s29ns-s30cm1b.csv`，rank0，20 rounds）：

| 臂 | wall step p50 | GPU busy/round（trace） | host gap |
|---|---|---|---|
| s29-nospec 基线 | 39.24 ms | 36.24 ms | **3.00 ms**（WP3 立项依据） |
| s30c m1（WP1-N 后） | 29.44-29.76 ms | 30.74 ms（含 nccl 窗口工件 ~1 ms；去工件 ~29.3-29.7） | **≈ 0（已隐藏）** |

ITER_DETAILS=1 实测（`.s30c/serve-m1iter.log`，100 个 decode step，
`iteration elapsed time` = engine core 每步 schedule+execute+sync 墙钟）：

| 统计 | 值 |
|---|---|
| p50 | 28.67 ms |
| p90 | 29.44 ms |
| min / max | 28.54 / 33.57 ms |

engine 每步循环耗时 p50 28.67 ms **低于** step 间隔 29.6 ms —— host 侧
（调度 + Python + 提交）已完全隐藏在 GPU 执行之下，流水线转为 **GPU-bound**：
wall step = GPU busy。此时任何无数值影响的 Python 侧削减（重复张量构造、
可缓存查询、多余 device 同步、纯 Python 分支）都不可能压低 wall step，
因为 wall 不再由 host 段决定。

## 2. 处置

按判据（−1 ms 以上）与"落空则如实出表停手"：**WP3 零改动收口**。
- 不做任何 Python 侧修改（改了也只能降 engine elapsed 28.67→更低，wall 不变）。
- WP1-N 已把 host gap 从 3.0 ms 压到 ≈0（GPU 时间 −9.7 ms > host gap 3.0 ms，
  削减过程中 host 段被逐步埋入 GPU 阴影）。
- host 段若未来重新露出（如 GPU 时间 < host 时间的新配置），本表可复测；
  工具：`ITER_DETAILS=1` + `grep "iteration elapsed"` + kdiff 全表求和。

## 3. 与设计图的对照

设计图 D3 预期 host 段 −1 ms。实际路径：WP1-N 的 GPU 削减（−9.7 ms）使
wall step 从 39.24 → 29.6，其中 3.0 ms host gap 全部被吸收（GPU busy 30.7 ≈
wall 29.6）。**"host gap 被框架开销吸收"在此反向成立：GPU 削减吃掉了 host gap
的存在前提**，D3 的 −1 ms 并入 WP1-N 的实测 −9.5~−9.8 ms 中兑现，无独立增量。
