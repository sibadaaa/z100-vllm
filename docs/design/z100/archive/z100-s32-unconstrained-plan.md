# S32 全开放优化方案（"全部都可以动"，2026-10-26）

> 前提：解除 S31 计划的全部保留项（nccl 结构性、state/KV/lm_head 精度、
> TTFT-保持、spec 退役等），以"每 token 墙钟"为唯一主目标，显存/TTFT/数值
> 稳定性转为**记账的代价项**而不是禁区。基线 = s30c-full 现役臂
> **29.48 ms/tok**（GPU busy 27.4 ms，host≈0）。
> 纪律不变（AGENTS.md）：真实 shape 微测 → 数值对照 → 服务 smoke → 同口径
> e2e；影响模型输出的改动必须带 eval（tests/evals / `vllm bench serve`）+
> 贪心流 diff 报表 + 用户裁决。

## 0. 物理地板先算清楚（7.5k ctx，单请求，TP4）

decode 每 token 的不可约工作是"把权重读一遍 + 通信 + KV"：

| 项 | fp8 现状 | w4a16 | 说明 |
|---|---|---|---|
| 权重字节/round（线性 256 calls） | 5.58 GB | **2.79 GB** | 精度是唯一能改的量 |
| @650 GB/s | 8.6 ms | 4.3 ms | |
| 通信 | 7.44 ms（nccl） | 同 | 128 AR + 1 AG，10-30 KiB 载荷，延迟主导 |
| attention KV | 121 MB | 61 MB（fp8 KV） | 7.5k ctx；262k ctx 时按比例放大 |
| 小核/GDN/杂项 | ~4.5 ms | 融合后 ~2 ms | |
| **纯 decode 地板** | **~21 ms**（S31 结论） | **~11–13 ms** | 各块同时到顶不现实，现实 ~13–15 |

即：不动精度的极限是 ~21 ms（S31 已给路径）；动精度 + 通信，纯 decode 的
现实终点是 **~13–15 ms/tok**；要再往下（<10 ms）只能靠"一步出多个 token"
（speculative，见 §3）。

## 1. Tier A：无损层（确定性/结构改动，不依赖评测裁决）→ 目标 ~20.5–21.5 ms

S31-WP1~WP4 全量采纳（split-K GEMV −1.5~2.5、split-KV attention −0.6~0.9、
epilogue 融合 −0.6~1.2、GDN CFG −0.2~0.4），再加一项 S31 因"结构性"排除、
本层解锁的：

### S32-A1：gfx906 专用 one-shot all-reduce（−4~5 ms）

- **现状**：nccl 128 AR/round，10 KB 载荷跑 54–80 µs——LL 协议 + channel
  机器开销占大头（载荷本身 PCIe P2P ~2 µs 级）。comm report 关闭的是
  **vLLM 自带** custom AR（gfx94/95 门 + 未缓存分配失败），不是"写一个"。
- **方案**：基于已验证的 fine-grain PCIe（S17：P2P 8.9×、`HSA_FORCE_FINE_GRAIN_PCIE=1`
  已开）写 gfx906 one-shot AR：每 rank 把自己的 10 KB 块直写 3 个 peer 的
  预留 fine-grain 缓冲 + flag 自旋 + 固定 rank 序本地归约。128 处调用点
  （`vllm/distributed` 派发层换 custom op，走 S26 opaque-auto-op 同款接线）。
- **关键性质**：固定 rank 序归约 = **逐 run 确定**（同 seed 同世界），消除
  nccl 路径的 run 间 skew 源之一；ULP 级 logprob 偏移与 BM16 同量级，按
  parity 板流程交付。
- **判据**：微测（10 KB one-shot vs nccl，10/30/64 KiB 三档）；e2e −3 ms 以上；
  贪心流 diff 报表（预期流仍落基线重启世界内）；长跑 0 崩溃/0 挂起（自旋核
  的 watchdog 路径要写）。
- **风险**：自旋死锁面（peer 失速）；显存 +~1.5 MB/AR 缓冲（可复用）。

## 2. Tier B：精度层（全部需要评测 + 用户裁决）→ 目标 ~14.5–16 ms

### S32-B1：线性层 w4a16（最大单项，−4~6.3 ms）

- 全量 int4：2.79 GB @400–450（LUT dot2 与 fp8 同款解压）= 6.2–7.0 ms vs
  现状 12.49。树上 GGUF/MMVQ 服务路径现成（`VLLM_QWEN35_GROUPED_GGUF_MMVQ`
  家族 + `gguf_kernel.cu`），先测 existing kernel 的 gfx906 实际带宽再决定
  复用或新写。
- **低风险变体（推荐先做）**：只量化 MLP（gateup+down = 4.28 GB/5.58），
  attention/GDN 投影保持 fp8：−3.8~4.3 ms，质量敏感面最小。
- **门**：困惑度/任务 eval（对齐 `docs/contributing/model/tests.md` 精神）+
  8-prompt 贪心 diff 板 + top5 logprob 偏移报表；验收线由用户定
  （建议：eval 退化 ≤0.5%、贪心流 diff 率与 top1 偏移阈值入裁决单）。

### S32-B2：lm_head fp8/W8A16（−0.45 ms）

636→318 MB。直接改 top-1 logprob 分布 → 必须单独裁决；与 B1 一起评。

### S32-B3：KV fp8 + 核内 LUT 反量化（−0.15~0.25 @7.5k，长 ctx 线性放大）

前测 ITL +41% 的根因假设 = 反量化未融合进 attention kernel；LUT 方案与
fp8 GEMV 同款。先微测证伪/证实，再进服务级。

### S32-B4：SSM state fp16（−0.15~0.2 ms）

量小、漂移风险真实（D5.4）。放最后，除非 B 层其他项全部落地后仍差最后一点。

## 3. Tier C：一步多 token（算法层）→ 目标 ~9–14 ms/tok

### S32-C1：speculative 重开（条件：B1/B2 先落地，否则不划算）

- 现状账：dflash(BM16, fp8) 89.9 ms/step ÷ 2.97 tok ≈ 30.3 ms/tok > 纯 decode
  29.48——**纯 decode 现在更快**，spec 不能原样重开。
- 但 spec 的 verify 一次读全部权重（M=8 与 M=1 的线性成本相同）：B1 之后
  verify 的线性块 12.5→~6.5 ms、draft 塔 int4 后 ~18 ms → step ~55 ms，
  accept 2.97 → **~18.5 ms/tok**。
- 再往上只能提高 accept：ngram_gpu（本 workload 是重复技术文档，匹配面大，
  树上已有该 arm）或训 EAGLE 式 drafter（离线训练成本单列）。
  accept 4.0 → step ~60 ms ÷ 5 ≈ **12 ms/tok**。
- 代价记账：draft 塔回驻（−1.46 GiB KV 容量回吐）、TTFT 升（draft prefill，
  预估 ~1.5–2×）、accept 彩票管理（workspace pin 已固化）。TTFT 目标建议
  放宽为"≤2× 现值"入裁决单。

### S32-C2（研究向）：persistent megakernel

把"小核 + GEMV + AR 等待"并成整层常驻核，消灭核边界延迟（triton 小核
3.26 ms 与 one-shot AR 的剩余延迟都受益）。工程量大、gfx906 无 MFMA 反而
简化数据流；作为 B 层落地后的探索项，不设日期。

## 4. 汇总路线（我实际会做的顺序）

| 步 | 内容 | 预期 step | 依赖/门 |
|---|---|---|---|
| 1 | S31 Tier A（WP1–4） | 29.48 → 26.5–27 | 零翻转硬门 |
| 2 | S32-A1 one-shot AR | → ~21–22 | parity 板 + 长跑稳定性 |
| 3 | S32-B1 低风险变体（MLP int4）+ eval | → ~18–19 | eval + 裁决 |
| 4 | S32-B2/B3（lm_head、KV fp8） | → ~17–18 | eval + 裁决 |
| 5 | S32-C1 spec 重开（drafter 改进） | → ~12–14 | TTFT/显存代价入裁决单 |
| — | 物理地板（纯 decode，全量化+全优化） | ~11–13 | — |

**不变的建议**：无论多开放，每一步仍然一格一验收、一格一回滚；
B/C 层每格都带 eval 与贪心 diff 报表——速度可以激进，账必须诚实。
