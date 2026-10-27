# Qwen3.8-27B-FP8 有效优化总结（截至 2026-10-27）

> **范围**：只收录“已进入生产或已在服务级同口径 A/B 中过门”的项；
> 被证伪/拒绝的另列 §5，避免重开。
> **口径**：除特别注明外，7469-token 真实文档 / 96 tok / 3 reps / TP=4；
> `t/s` 为引擎生成段口径（`tok/step ÷ step_p50`，不含 TTFT），
> 与历史文档里的 `out_tok_per_s`（含 TTFT 的墙钟口径）不可直接比。
> **一句话**：真正有效的不是“把带宽堆上去”，而是
> **拆掉串行、压低每 op 延迟、按真实形状派发、用投机摊薄权重读取**；
> 反过来，凡是“只改字节/只改 precision/只降 launch”的项，基本都被实测挡回。

---

## 0. 总表（16 项：12 项主优化 + 4 项配置/方法论）

| # | 优化 | 层 | 实测收益 | 状态 | 证据 |
|---|---|---|---|---|---|
| 1 | piecewise graph + 捕获 [1,2,4] | 图/调度 | step **−11.67%**（129.26→114.17 ms），输出 +10% | 采纳 | [`routea`](../result/z100-routea-deploy.md) |
| 2 | FP8 emu `DECODE_CFG=35`（BK64/BN32） | 内核 | 内核 **−18.7%**，e2e **−7.34%**，逐 bit 同 | 采纳 | [`fp8-emu`](../result/z100-fp8-emu-optimization-result.md) §6 |
| 3 | W8A16 + fp16 prefill 物化 | 内核/路由 | decode **1.2–2.1×**；prefill M≥512 **1.4–2×** | 已启用 | [`w8a16`](../result/z100-w8a16-fp16-prefill-result.md) §4.2 |
| 4 | A1：`in_proj_ba` GEMV | 内核/派发 | step **−7.74 ms（−7.1%）** | 采纳 | [`s28-a1`](../result/z100-s28-a1-smallm-ab.md) |
| 5 | draft 塔 3D split-KV 路由 | 内核/派发 | step **−8.2 ms（−8.1%）** | 采纳 | [`s28d`](../result/z100-s28d-draft3d-accept-lottery.md) |
| 6 | FA verify `BLOCK_M=16` | 内核 | step **−3.1 ms（−3.3%）** | 采纳 | [`s29-bm16`](../result/z100-s29-bm16-ab.md) |
| 7 | GDN 原生 M=1 kernel | 内核 | 216→12.5 µs/call（**17×**），e2e **−9.5~−9.8 ms（−25%）** | 采纳 | [`s30c-gdn`](../result/z100-s30c-gdn-m1-native.md) |
| 8 | gfx906 push one-shot AR | 通信 | no-spec **−2.0~−2.6 ms**；MTP 臂 AR-off 代价 **+10.07 ms** | 采纳 | [`s33-ar-fix`](../result/z100-s33-oneshot-ar-fix.md) |
| 9 | AR size 门 64→128 KiB | 通信 | c4 **−4.6 ms/步**（聚合 +3.4%）；c1 无回归 | 采纳 | [`STATUS`](STATUS-20261027.md) §2.3 |
| 10 | GEMV stack（rows=4 + M1） | 内核 | MTP **−2.03 ms（−4.1%）**；no-spec −1.0~−1.4 ms | 采纳 | [`STATUS`](STATUS-20261027.md) §2.3 |
| 11 | MTP k=2 | 投机 | 贪心 **36.9→51.6 t/s（+40%）**；采样 34.7→40.6（+17%） | 采纳 | [`specbench`](../../runs/specbench/REPORT.md) |
| 12 | 退役 DFlash2 k=7 | 投机 | 22.7→36.9 t/s（**移除 −38% 回归**） | 采纳 | 同上 |
| 13 | O1/O2 host fastpath | 宿主 | c≥4 **+6.6% / +3.55%**；c1 持平 | 采纳（条件收益） | [`fp8-emu`](../result/z100-fp8-emu-optimization-result.md) §8 |
| 14 | prefix caching + `mamba align` | 配置 | 生产命中率 **91.5%**；60k 冷 prefill 178 s / 128k 520 s | 采纳 | [`STATUS`](STATUS-20261027.md) §2.4 |
| 15 | accept workspace pin | 数值/方差 | 1.969 世界 ~80% vs 未钉 ~33% | 采纳（非确定性） | [`s29-pin`](../result/z100-s29-accept-pin.md) |
| 16 | 度量纪律（同窗/N≥2/指纹） | 方法论 | 避免 ±6–10% 假收益 | 强制 | [`s29-pin`](../result/z100-s29-accept-pin.md) §4 |

---

## 1. 里程碑链（同一 7469-doc 探针，口径逐行标注）

| 日期 | 配置 | step p50 | tok/step | ms/tok | 引擎 t/s | 口径 |
|---|---|---:|---:|---:|---:|---|
| 10-24 | graph-A，DFlash2 k=7 | 114.17 | 2.909 | 39.2 | 25.5 | 投机 round |
| 10-25 | +A1 | 101.11 | 2.909 | 34.8 | 28.8 | 投机 round |
| 10-25 | +draft-3D | 93.14 | 2.667 | 34.9 | 28.6 | 投机 round（accept 世界低） |
| 10-25 | +BM16 | 89.88 | 2.909 | 30.9 | 32.4 | 投机 round |
| 10-26 | S30c native，**无投机** | 29.48 | 1.0 | 29.5 | 33.9 | 纯 decode（墙钟 26.8–27.3） |
| 10-27 | +push AR，无投机 | 26.70 | 1.0 | 26.7 | 37.5 | 纯 decode（墙钟 28.3–28.8） |
| 10-27 | +MTP k=2 | 46.50 | 2.40 | **19.4** | **51.6** | 投机 round |

> 读法：**不要跨“有无投机”直接比 step**。同一投机口径内，
> 114.17→89.88 是 A1/draft-3D/BM16 的合力；无投机口径内，
> 39.24→29.48→26.70 是 GDN M1 + push AR；MTP 把每 token 从 26.7 压到 19.4，
> 代价是 step 变成 46.5（多出 draft + verify）与 TTFT +1.7 s。

---

## 2. 分项：为什么有效

### 2.1 图与调度（#1、#14）

**piecewise graph + 捕获 [1,2,4]**：把 decode 的整段 host 发射收进图，
消掉 per-op launch 与一部分核间空隙；C2 四档上下文全部更快（−9.8%~−20.1%），
G1 8/8 逐 token 一致。代价是冷启动 150→255–369 s、KV −6.2%。
**注意**：收益来自“整图重放”，不是单纯“降 launch 开销”；纯 launch 项的
天花板在 S23 已被证明很小。

**prefix caching + `mamba-cache-mode align`**：长 prompt 的 TTFT 由命中率决定，
不是由 kernel 决定。生产 91.5% 命中时 60k 均值可接受；未命中时 60k 冷
prefill 178 s、128k 520 s。任何长上下文优化都必须先区分冷/热口径。

### 2.2 内核与派发（#2–#7、#10）

**cfg35（BK=64/BN=32）**：旧 cfg16 每行每 chunk 只读 32 B，每次都要付
一整笔 DRAM 延迟；BK=64 把每轮开销摊到 2 倍元素上。6 个 decode 形状
−15.3%~−23.2%，e2e −7.34%，且 `torch.equal` 逐 bit 相同——这是“改几何
不改数学”的典范。

**A1 `in_proj_ba`**：48 次 triton_matmul，每 rank 只读 123 KB 却要 217 µs
（0.57 GB/s），是整条 step 效率最差的 kernel。把 `n_out<=64` 的形状
放行到 bf16 GEMV（M≤8），M=8 微测 40.9 vs 188.6 µs（4.6×），e2e −7.74 ms。
**关键不是“换 GEMM”，而是“按真实形状派发”**。

**draft 3D 路由**：draft 塔的 7-token block forward 原本走 unsplit 2D、
grid (2,2,1)，4 个 CTA 串行扫 2048 窗口，1234 µs/次 × 5 层。
映射到 verify-3D 同款 small-q 路由后 −8.2 ms。**同一个病，只是在 draft 形状上。**

**BM16**：FA verify 原来 8 个 q-block × 128 段，每个 CTA 只有 1 个 token 的
6 行有效，KV 被重读 8 遍（逻辑带宽 11.6–15.7 GB/s）。BLOCK_M=16 把 KV
重读减半，−3.1 ms；16/16 case 零翻位。

**GDN M1 native**：串行核 grid (6,1,1)、216 µs/call，state 布局导致每 warp
跨 32 条 cache line；m1 native 改成 lane 持 2 个 k、warp 持 VW 个 v、
wave64 butterfly，384 blocks，12.5–12.9 µs/call（17×），e2e −9.5~−9.8 ms。
**这是“并行度”而不是“带宽”的胜利**。

**GEMV stack（rows=4 + M1）**：M=1 专用 stream kernel 剥掉 M 维谓词/分组，
rows=4 提高 wave；per-op −6.1%~−12.1%，MTP 臂 −2.03 ms。
值得记录的是：**S30c 当时按“边际”不采纳，S34 在 MTP 臂上重测后采纳**——
栈变了，同一个 kernel 的边际价值会变，必须重 A/B。

### 2.3 通信（#8、#9）

**push one-shot AR**：gfx906 的 PCIe **出站读**看不到对端 GPU L2 的脏行，
所以 upstream 的 pull/1stage 核读到陈旧数据；**入站写**对目标 L2 一致，
所以 push + volatile mid 屏障可行。10 KiB 128.8→69.1 µs（+46%），
no-spec e2e −2.0~−2.6 ms；MTP 臂上 AR-off 的代价是 **+10.07 ms/步**
（128+ 个同步点放大 RCCL 的 lockstep skew-wait）。**这条是“延迟”不是“带宽”**。

**size 门 128 KiB**：c4 verify 载荷 120 KiB 原本被 64 KiB 门打回 RCCL；
120 KiB 微测 push +15.4%，服务级 c4 −4.6 ms/步（聚合 +3.4%），c1 无回归
（30 KiB 在门内，路径不变）。256 KiB 处 push 无增益（−0.7%），所以 128 是甜点。

### 2.4 投机（#11、#12）

**MTP k=2**：目标模型自带 1 层 `mtp.*`，draft 便宜；verify M=3 的线性层
成本与 M=1 近似（M=1..8 对 GEMM 平坦），所以“一步出 2.4 token”几乎是白拿。
贪心 36.9→51.6（+40%），采样 34.7→40.6（+17%）；代价是 warm TTFT
0.72→2.40 s（draft 预填充），以及 accept 彩票。

**退役 DFlash2 k=7**：在 M=1 特化的新栈上，DFlash2 的 5 层 bf16 draft 塔
+ M=8 verify 绕开了所有优化，step ≈101 ms、22.7 t/s，比无投机低 38%。
k=4 尝试在启动时 `_selector_walk_kernel` GPU VM fault；外推 k≤4 仍 26–29 t/s。
**“删掉一个负收益的旧配方”本身就是一项优化**，还顺带释放 1.46 GiB
权重、+407k KV token（S30c V1）。

### 2.5 精度与宿主（#3、#13）

**W8A16 + fp16 prefill 物化**：W8A16 让激活不量化、权重仍 1 B/参数；
decode 的 quant+widen+gemv 三 launch 合成一 launch，全形状 1.2–2.1×；
prefill M≥512 走 dequant + rocBLAS，0.50–0.70×（1.4–2× 快），rocBLAS
达 16.5–17.7 TF/s vs tile 6.3–6.5。**代价**：W8A16 与 W8A8 数值不逐 bit
相同；启用时 e2e parity/soak 未单独跑（运维决定），后续 s30c–s34 的
生产稳定性是间接背书。

**O1/O2 host fastpath**：去掉 custom_op 包装税（−57.9 µs/次，−13.9 ms CPU/步）
与量化 host 包装（−12.8 µs/次，−3.3 ms CPU/步）。c=1 完全被 GPU 吸收
（两次反序窗口互相抵消），c=4 +6.6%、c=16 +3.55%。**这是一条“只在
并发/ prefill 场景兑现”的优化**，不能拿 c=1 去否定它。

### 2.6 配置与数值方差（#14–#16）

- `--max-num-batched-tokens 8192`：7469 prompt 单 chunk，避免 2048 的
  3 个 chunk 边界（历史 ~337 ms/边界）；但 chunk 越大，AR 消息越大，
  要和服务级 TTFT 一起算。
- `--max-model-len 262144` + `--max-num-seqs 4`：容量与并发的折中；
  KV 池约 1.17 M token。
- accept workspace pin：`ROCBLAS_WORKSPACE_CONFIG` / `HIPBLAS_WORKSPACE_CONFIG`
  把 1.969 世界概率从 ~33% 提到 ~80%，但**不是完全钉死**；所以 accept 类
  A/B 仍然 N≥2 取分布，不能取单点。
- 指纹工具：每个 `VLLM_*` knob 进 KNOBS，启动前可打印 `policy: FP ...`；
  没有指纹的“提速”一律不接受。（本臂日志未捕获该行，需单独跑一次。）

---

## 3. 六个可复用的判据

1. **先分墙**：带宽 / 算力 / 并行度 / 每 op 延迟 / launch。
   GDN 是并行度，AR 是延迟，GEMV 是每字节 LDS 流量，prefill 是 MFU。
2. **微测解释机制，e2e 决定去留**：draft matmul 微测 −6.0 ms，但换 rocBLAS
   改了 draft 分布，长文档 accept −8.3%，净亏；A1 微测与 e2e 同向，才采纳。
3. **改累加序必须过 parity**：位同（kernel）→ 16/16 零翻位（BM16）→
   近平局披露（S29-BM16 0.0213 nat）是三个不同强度的门，不能混用。
4. **accept 是彩票，不是常量**：跨重启 ±6–10%；任何 accept/tok-per-step
   类结论必须 N≥2；step p50 才是稳定量。
5. **栈变了要重测边际**：GEMV stack 在 S30c 是“边际不采纳”，在 S34 MTP
   臂是 −2 ms；AR 在 no-spec 是 −2.5 ms，在 MTP 臂是 −10 ms。
6. **一个 WP 一条回滚**：env 不设 / 二进制 cp 回 / 删除新文件；
   md5、policy 指纹、原始 JSON 三件套齐全才算交付。

---

## 4. 数值与验收协议（现行）

| 场景 | 门 |
|---|---|
| 内核改几何/不改累加序 | `torch.equal` 逐 bit + 真实形状微测 |
| 改累加序（BM16/A1/draft-3D） | 16/16 或 64/64 token 流；零翻位；top-1 logprob 漂移报表 |
| 近平局翻转 | 翻位处 top-2 gap <0.05 nat；双方互有；逐条披露 |
| 服务级 | 同窗口 A/B、3–5 reps、step p50、accept、TTFT、0 crash |
| 稳定性 | 30 min decode 连发 0 fail；dmesg 0；长上下文另测 |
| 精度（INT8/W4A16/lm_head/KV） | eval + 贪心 diff 板 + 用户裁决；accept 不能显著掉 |

---

## 5. 被证伪 / 拒绝的尝试（不要再开）

| 尝试 | 结果 | 原因 / 证据 |
|---|---|---|
| W4 `v_dot8_i32_i4` GEMV | **No-Go** | 实测 54 GB/s(fp8-eq)，距 450 门 8×；ISA 是 i4×i4，不是 int8×int4（[`W4`](W4-RESEARCH-20261025.md) §6） |
| DFlash2 k=7 | 退役 | 22.7 t/s vs 无投机 36.9；draft 塔 + M=8 verify 绕开 M=1 特化（[`specbench`](../../runs/specbench/REPORT.md)） |
| DFlash2 k=4 | 不可用 | 启动 `_selector_walk_kernel` VM fault；外推 26–29 t/s 仍低（同上） |
| SMALLM=rocblas | 拒绝 | 当次按净亏拒绝：step −4.65 ms，但长文档 accept 2.909→2.667（−8.3%）。S28d 后来指出 accept 证据受彩票污染；即便如此，step 收益也不足以单独立项（[`s28-a1`](../result/z100-s28-a1-smallm-ab.md)） |
| whole-A M=1 GEMV | 停 | 两变体 +5–17%，qkv 387–402 / o 361 <450 门；后由 rows=4+M1 拿 −2 ms（[`s33-wholea`](../result/z100-s33-wholea-gemv.md)） |
| `FP8_GEMV_MAX_M=1` | 不采纳 | 真实 7469 step +3.9%、36k −2.3%，方向相反且 <5%（[`decode-prefill`](../result/z100-decode-prefill-final-summary.md) §1） |
| custom AR pull/2stage | 停 | 出站读看不到对端 L2 脏行；2stage 在本栈不可用（[`s33-ar-fix`](../result/z100-s33-oneshot-ar-fix.md)） |
| RCCL ALGO/PROTO/CHANNELS | 停 | 7 变体 × 5 尺寸无 separability（[`p2p-acs`](../archive/result/z100-12-p2p-acs-impact-result.md) §3.2） |
| KV fp8 默认开 | 暂缓 | 反量化未融合时历史 ITL +41%；需先做核内 LUT（[`s32`](../design/z100/archive/z100-s32-unconstrained-plan.md) B3） |
| O4 stream 路由 | 不发布 | `5120:1536@M=8` 与两种 tile 数值不同；parity 8 条里 5 条分歧（[`fp8-emu`](../result/z100-fp8-emu-optimization-result.md) §7） |
| O5 量化融合 | 不发布 | kernel 逐 bit 同，但几何固定 BK128/BN64，e2e **+54%** 回归（同上 §14） |
| O1/O2 想救 c=1 | 无效 | c=1 GPU-bound；两次反序窗口互相抵消；只有 c≥4 兑现（同上 §8） |
| CUDAGraph FULL | 不可行 | 编译路径 `Data-dependent branching` graph break；piecewise 已采纳（同上 §13） |
| chunk 边界物化 | 排除 | 关掉 fp16 物化后每 chunk 额外成本仍在（0.3–0.42 s），残余在 chunked-prefill 公共路径（[`chunk`](../result/z100-chunk-boundary-rootcause.md)） |
| 纯 n-gram（q16） | 跳过 | 上限 acc 1.315 tok/step，边缘收益（[`ngram`](../result/z100-ngram-upper-bound.md)） |
| prefill 940–1030 t/s | 目标下调 | 逐形状 GEMM 18.4–19.9 TF/s ≈ 屋顶；hipBLASLt 无 gfx906 kernel（[`prefill-audit`](../result/z100-prefill-rocblas-audit.md)） |
| MoE + GPTQ/AWQ、GPT-OSS MXFP4、aiter、EP | 不做 | 已知失败组合 / 无数据通路 / 无 all2all（[`s35 design`](../design/z100/s35/z100-s35-moe-design.md) D11） |

---

## 6. 相关文档

- [现状](STATUS-20261027.md)
- [后续路线](ROADMAP-20261027.md)
- [S33 终局](../result/z100-s33-final.md)
- [S30c 终局](../result/z100-s30c-final.md)
- [S29 BM16](../result/z100-s29-bm16-ab.md)
- [S28 A1](../result/z100-s28-a1-smallm-ab.md)
- [S23 fp8 emu](../result/z100-fp8-emu-optimization-result.md)
- [W8A16/prefill](../result/z100-w8a16-fp16-prefill-result.md)
- [投机 TPS](../../runs/specbench/REPORT.md)
