# z100 投机解码 TPS 探索（2026-10-27）

> 问题：s33 优化后，不开投机解码还剩多少空间？开 MTP / DFlash2 能到多少 t/s？
> 方法：三臂同口径 A/B。每臂独立重启服务，policy 指纹强制一致
> （`FP e9f6b1ff41e682d6` = 现役 s33 ctrim + WP2 push-AR 臂）。
> 口径：单流 batch=1，7469-token probe prompt（同 s33 部署协议），prefix cache 热。
> W1 = 96 tok 贪心 ×3（s33 部署口径）；W2 = 512 tok temp0.7+rp1.05 ×2（agent 采样口径）。
> t/s = 输出 token 数 / (末 token 时刻 − 首 token 时刻)，不含 TTFT。

## 结果

| 配置 | W1 贪心 t/s | W2 采样 t/s | warm TTFT | 接收长度 |
|---|---|---|---|---|
| **A. 无投机（现役 s33）** | **36.9**（36.65/36.90/37.16） | **34.7**（34.4/34.9） | **0.72 s** | 1.0 |
| **B. MTP k=2**（目标自带 mtp.* 权重） | **51.6**（47.9/51.6/52.0）**+40%** | **40.6**（40.3/40.9）**+17%** | 2.40 s | 贪心 ~2.1–2.6，采样 ~1.9–2.3 |
| **C. DFlash2 k=7**（-ms draft，生产原配方） | **22.7**（22.6/22.7/24.2）**−38%** | **20.4**（20.3/20.4）**−41%** | 2.11 s | ~2.0–2.5（k=7 下严重浪费） |

现役基线复算：36.9 t/s ⇔ 27.1 ms/step，与 DEPLOY.md 的 step p50 26.7–27.3 ms 吻合；
DSH GUI 显示的 32 tok/s 位于 W1/W2 之间（含工具解析等宿主开销），口径自洽。

## 判读

1. **无投机已到屋顶**：27 ms/step 是 s30c M1 GDN kernel + FP8 W8A16 GEMV(cfg35)
   + piecewise graph [1,2,4] + push-AR oneshot 的合力，全部按 M=1 decode 特化。
2. **MTP k=2 是当前唯一正收益的投机路径**：1 层 MTP draft 便宜，verify M=3
   仍近似 M=1 成本（step ≈ 46.5 ms 换 ~2.4 token）。k=3 无收益
   （per-position 接收率 0.83/0.73 衰减 + step 增大，采样口径反而更差）。
3. **DFlash2 k=7 在 s33 栈上是净亏损**：verify M=8 + 5 层 bf16 draft tower + 独立
   draft KV 完全绕开 M=1 特化路径（step ≈ 101 ms），而该 doc 上接收率在
   position 3 后崩塌（0.52/0.31/0.12/0.04/0.03/0/0，draft 接收率仅 14–21%）。
   k=7 是旧栈（130–190 ms/step 时代）的遗产配方，对 27 ms/step 的新栈不适用。
4. **TTFT 代价**：两个投机臂 warm TTFT 都 +1.4–1.7 s（0.72 → 2.1–2.4 s）。
   对 DSH 长 prompt agent 负载（现 TTFT 6.8 s）意味着 ~8.3–8.5 s。
5. **贪心无损性**：投机 verify 本身无损，但 batch 形状不同 ⇒ 浮点归约顺序不同 ⇒
   近平票 token 翻转后各自成篇（MTP 首 diff @46，DFlash2 首 diff @2；
   两侧文本均连贯，且前缀逐位一致证明 verify 正确）。

## 折算到 DSH agent 负载（现 32 tok/s）

- MTP k=2：32 × (40.6/34.7) ≈ **37.5 tok/s**；一轮 ~2分44s 的生成段 ≈ 2分15s（−18%），
  但每轮 TTFT +1.7 s（输出越长越划算；DSH 单轮 ~2000 tok，净收益 ~15%）。
- DFlash2 k=7：32 × (20.4/34.7) ≈ **18.8 tok/s**，生成段翻倍，不可用。

## 建议

- 若接受 TTFT +1.7 s：现役配置加
  `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` 一项即可拿 +17%。
- DFlash2 路线想翻盘需要两件事：把 M=1 特化 kernel 覆盖到 verify 形状（M=5–8），
  以及 k 降到 ≤4（接收率崩塌点之前）。在完成前，DFlash2 不应回到现役栈。

## 产物

- `e2e-A-nospec.json` / `e2e-B-mtp2.json` / `e2e-C-dflash7.json` — 三臂明细
- `parity-*.json` — 96-token 贪心流（跨臂比对）
- `serve-*.log` / `window-*.log` — 服务日志与窗口编排日志
- `launch_ctrim_spec.sh` — 生产 launcher 的 SPEC 透传副本（未改动生产文件）
- `window_spec.sh` — 窗口编排（停现役 → 起 SPEC 臂 → bench → 恢复现役 + 指纹校验）
- `metrics-*.txt` — /metrics 抓取（该栈 spec 计数走 API server 日志，见 serve log）

## 追加（2026-10-27 下午）：DFlash2 k=4 尝试与 M=5–8 施工评估

**k=4 无法直接测**：窗口 D-dflash4 的 spec 服务器在启动显存画像 dummy run 时
**GPU VM fault**（`_selector_walk_kernel`，即 DFlash2 候选选择器走格 kernel，经
`triton_poi_fused_index_1` 融合；另有 `triton_poi_fused_cat_expand_index_mul_slice_unsqueeze_0`），
worker 被杀，启动失败。k=7 同路径正常 → 疑似 num_speculative_tokens=k 与
draft `block_size=8` / 缓冲形状之间的索引越界。生产服务已恢复（health 200，
指纹一致），故障窗口未留任何残留。`serve-D-dflash4.log` 保留 VMFault 分析转储。

外推成本曲线（实测三点）：M=1 → 27.1 ms/step（36.9 t/s）；M=3 → 46.5 ms（51.6 t/s，
MTP k=2）；M=8 → 101 ms（22.7 t/s，DFlash2 k=7）。k=4 的 M=5 内插 ≈ 70–80 ms ×
acc≈2.05（k=7 per-position 接收率 0.52/0.31/0.12/0.04 截断）≈ **26–29 t/s，
仍低于无投机基线** —— k≤4 单独不翻正，必须配合 verify 形状的 kernel 覆盖。
