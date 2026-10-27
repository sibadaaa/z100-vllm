# S34 WP5 设计 + 施工图：fp8 GEMV split-K（2026-10-27）

> 状态：**立项（设计 + 施工图），实现与验收留裁决**。
> 触发条件（S33-WP1 停点）已成立：whole-A 两变体（LDS / L2 直读）在
> 位同序约束下卡 4-way bank 冲突结构性下限，qkv 402.5 / o 361.3 / gdn_out
> 363.6 GB/s < 450 门；折算 e2e 仅 −0.9~−1.1 ms。换算法类（split-K）才能
> 把带宽抬过门，但会改累加序 → 必须过零翻转硬门 + 裁决。

## 1. 现状与靶子

- 现役 fp8 emu GEMV（`_rocm_C` 内，`stream_block`/`stream_m1` 两变体）：
  256 次/step，~48.8 µs/op，~12.5 ms/step，有效带宽 ~490-590 GB/s（形状而异）。
- S30c-WP2-N 的 rows=4 流式变体（env 可开）已拿走 −1.0~−1.4 ms；本 WP 是
  在其之上的算法类更换：**K 维切分（split-K）+ 多 block 分片累加**，
  目标 qkv/o/gdn_* 形状 ≥450 GB/s（S33-WP1 判据 A），e2e −1.5 ms 采纳门。

## 2. 算法与位同策略

- GEMV：y[n] = Σ_k W[n,k]·x[k]。split-K 把 K=5120（qkv/o）或 17408×2
  （gateup）切成 S 段，S 个 block 并行，各自算 partial y_s[n]（fp32），
  再由第二个 kernel（或 atomic/last-block-wins）按 **s = 0..S-1 固定序**
  归约成 y[n]，最后乘 per-channel scale（epilogue 序与现路径一致）。
- 位同关键：partial 归约**必须固定段序**（禁止 atomics 无序累加）。
  推荐 two-kernel 方案（partial → reduce），牺牲一次 5 µs 级 launch 换
  确定性；`last-block-wins` + 段序循环兜底也可，但要防 block 调度乱序。
- L2/bank：分片后每 block 扫 K/S 列，A 读的 4-way bank 冲突模式随
  分片起点错开，预期冲突率下降（这正是 whole-A 变体做不到的）。

## 3. 形状与工作量

| 形状 | N×K | 现带宽 | 门 |
|---|---|---|---|
| qkv | 6144×5120（attn 侧） | 402.5 | 450 |
| o | 5120×5120 | 361.3 | 400（S33 判据） |
| gateup | 2×17408×5120 | 586.9 | 600 |
| gdn_in/out | 8192×5120 等 | 440.8 / 363.6 | 450 |

M=1（MTP draft 单 token）与 M=3（verify）两档都要过；
prefill M=8192 不动（GEMM 路径）。

## 4. 验收门与回滚

1. 微测：`tools/dcu/kernel_microbench_p13.py` 风格，六形状 × M∈{1,3}，
   门 = 上表；数值对拍现 fp8 emu 核（T=1 fp16 位同不可得时，按
   "max|Δ| ≤ 1e-3 相对 + top5 logprob 偏移板"走裁决口径）。
2. e2e：7469 doc / 96 tok / 3 reps，step p50 −1.5 ms 采纳门。
3. 流：nospec 臂 `cmp_capture` 零翻转硬门；MTP 臂 in-session 位同。
4. 开关：新 env `VLLM_DCU_FP8_GEMV_SPLITK=1`（默认关，A/B 与回滚一体）；
   csrc 改动走 build-sync 纪律（备份 + md5 + 竞态重跑）。
5. 风险：two-kernel 的第二次 launch 在 ~1000 处核间空隙敏感区，
   若 e2e 不升反降，先并 partial-reduce 进 push AR 空闲波（与 WP4 协同）。

## 5. 与其他 WP 的关系

- 与 WP3（rows=4/M1 env）正交：先测 WP3（纯 env），split-K 落地后在
  其之上再 A/B（rows 参数会被 split-K 内部覆盖时需在施工图里写明优先级）。
- 与 INT8 sdot4（约束外）关系：若 INT8 checkpoint 路线被裁决采纳，
  fp8 split-K 的价值重估（INT8 原生 v_dot4 3.9× 于 fp8 emu）。
