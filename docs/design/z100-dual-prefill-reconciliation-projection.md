# 两条 prefill 路径的换算 + 最新优化：Z100 上 decode/prefill 能到多少

> 输入：用户 2026-10-24 两组真机数据（DTK prefill 臂 / rocm prefill 臂），
> 模型 `Qwen3.8-27B-FP8` + DFlash2 k=7、c=1、`temperature=0`、`--enforce-eager`、TP=4。
> 本仓库实测历史：[z100-cudagraph-route-a-20261021.md](../result/z100-cudagraph-route-a-20261021.md)、
> [z100-rocm101-perf-p3-optimization-plan.md](z100-rocm101-perf-p3-optimization-plan.md)、
> [z100-rocm101-perf-verdict.md](../result/z100-rocm101-perf-verdict.md)。
> **下游**：[`z100-decode-prefill-optimization-design.md`](z100-decode-prefill-optimization-design.md)（设计图）→
> [`z100-decode-prefill-optimization-construction.md`](z100-decode-prefill-optimization-construction.md)（施工图）。
> 本文是它们的**数据依据**；本文内部 §10 已被 §11 修正，冲突以 §11 为准。
> 本文性质：**换算 + 投影，不含代码改动**。标注沿用仓库惯例：
> **【实测】**=在本仓历史或用户数据里可直接复算；**【推断】**=由实测推导；**【待测】**=本文无法判定。

---

## 0. 一页结论

1. **两组数据的 decode 差异不是 prefill 路径造成的，是接受长度造成的。**
   把吞吐拆成 `step_ms / acc`：DTK 臂 `129 / 2.84`，rocm 臂 `167.4 / 5.82`。
   1.58× 的差距里 **2.05× 来自接受长度、÷1.30 来自 step**。接受长度是 prompt 的属性，
   不是 `VLLM_DCU_FP16_PREFILL` 的属性。**【推断，§1/§2】**

2. **`VLLM_DCU_FP16_PREFILL` 在代码上物理不可能改变 decode step。**
   `_try_prefill` 里 `if m < _min_m(role, None) or m < _FORCE_MIN_M: return None`，
   而 `_FORCE_MIN_M = 16`、单请求 verify 的 `M = k+1 = 8`；源码注释原文写着
   "decode (M <= 8) never routes"。**【实测（读码）】**

3. **prefill 两臂不同口径。** rocm 臂两点的 TTFT 拟合出 `684 ms 固定 + 1.809 ms/token`，
   所以它那个"319 t/s"里 **43% 是固定开销**。折到同一长度（7469）：
   DTK 8.894 s vs rocm 14.199 s，**DTK 真快 1.60×**。**【推断】**

4. **"中和"不需要取舍**：DTK 已经把 prefill 拿走了，decode 那边要修的是独立的一件事
   （要么接受长度，要么那个 167→129 的 30% step 回归）。

5. **最该做的单项是开图（路线 A）**，已提交、已实测、零新代码：
   step **127.5 → 118.4 ms（−7.1%）**、输出 36.5 → 38.0 tok/s、TTFT 1.38 → 1.35 s 持平、
   c=4 聚合 48.4 → 51.6 tok/s、8/8 逐 token 一致。你现在挂了 `--enforce-eager`，这一项在桌上没吃。

6. **投影（7469 上下文，你的实测接受长度 2.84）**：

   | 阶段 | step ms | tok/s |
   |---|---:|---:|
   | 现状 | 129.0 | **22.0** |
   | + 图路线 A | 119.8 | **23.7** |
   | + allreduce / 草稿塔 / lm_head / GEMV 带宽回收 | 100.6 | **28.2** |
   | 以上 + 长上下文接受长度修到 4.0 | 100.6 | **39.8** |

   短上下文（512，接受长度 5.82）同口径是 **34.7 → 37.4 → 44.5**。

---

## 1. 先把两张表拆成 `step` 和 `acc`

decode 的唯一正确口径：**每个输出 token 的成本 = step / 接受长度**。
两张表给的 `ITL` 是**每 step**（一次前向）的时延，`输出速度` 是按**被接受的 token** 计的吞吐——
只有这样才能自洽（`输出速度 × ITL = 每 step 接受的 token 数`）。

| 臂 | step ms | 接受长度 | token/step 率(k=7→8 行) | ms/token | tok/s |
|---|---:|---:|---:|---:|---:|
| DTK，7469 tok，MBT 8192 | **129** | **2.84** | 35.5% | 45.45 | **22.0** |
| DTK，7469 tok，MBT 2048 | 131 | 2.95 | 36.8% | 44.44 | **22.5** |
| rocm，512 tok | **167.4** | **5.82** | 72.7% | 28.78 | **34.7** |
| rocm，640 tok | 170.8 | 5.81 | 72.7% | 29.39 | **34.0** |
| 【参照】E0 eager，512 tok，10-21 | 127.5 | 4.65 | 58.2% | 27.4 | 36.5 |

拆解结果：

- `34.74 / 22.0 = 1.579`
- 接受长度贡献 `5.816 / 2.838 = 2.049`
- step 贡献的倒数 `167.41 / 129 = 1.298`
- `2.049 / 1.298 = 1.579` ✓

**结论：接受长度是 2.05× 的大头，step 是 1.30× 的小头，方向相反。**
你现在把它读成"rocm prefill 让 decode 更快 58%"，其实是"短 prompt 让接受长度翻倍，同时那个 run 的 step 慢了 30%"。

---

## 2. `VLLM_DCU_FP16_PREFILL` 到底影响什么（代码级）

三个事实，都能在树上直接读出来：

**① 闸门是 M 阈值，decode 永远进不去。**
`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py:437-442`：

```python
m = x.numel() // K
if m < _min_m(role, None) or m < _FORCE_MIN_M:
    return None
```

`_FORCE_MIN_M = 16`（:132），`_MODE == "1"`（也就是你显式设的 `=1`）时
`_min_m()` 直接返回 16，**绕过整张按角色的表**（:139-145）。
单请求 + k=7 的 verify 是 `M = k + 1 = 8 < 16` ⇒ 必然回落 W8A16。
源码注释原文：`# Cheap gates first: while VLLM_DCU_FP16_PREFILL is on this runs for every linear layer of every step, and decode (M <= 8) never routes.`（:434-435）

**② 它唯一付给 decode 的代价，是每层每步一次 Python 调用。**
`gfx906_fp8.py:1070-1073`：`if _PREFILL_ROUTE: routed = gfx906_fp16_prefill.try_prefill(...)`，
`_PREFILL_ROUTE` 对 `1` 和 `auto` 都为真（:269）。decline 路径要付
`x.dtype` / `is_contiguous()` / `_role(layer)`（一次 `rsplit` 字符串操作）/ dict 查找。
按 64 层 × 4 个线性 ≈ 256 次/步、3–8 µs/次估：**0.8–2.0 ms/step，占 129 ms 的 0.6–1.5%**。**【推断】**
所以 `=0` 理论上比 `=1` 的 step **略快**（1% 量级），但 prefill 会掉一大截——**不是 30%。**

**③ 它顺带改的是容量，不是速度。**
物化要一份 fp16 权重副本（13.5 GB/TP4 = 3.4 GB/卡），10-21 实测 KV 806,753 → 742,334 tok（**−8%**）。
这只影响能塞多长，不影响每步耗时。**【实测】**

⇒ **167.4 vs 129 这 30% 必须另有解释。** 两个候选：
(a) 那个 rocm run 是旧树（早于 `f5083195e` / `0e1475163` / `0e1928ca4`）；
(b) 当前树上真有 30% 的 step 回归，而它的成因和 prefill 开关无关。
两者都能用一次 A/B 分辨（§7 A1）。**如果是 (b)，那它是全表最大的单项：−8 tok/s。**
同一批树上，10-21 的 E0 eager 是 **127.5 ms**、你 DTK 臂是 **129 ms**，两个独立点都指向 ~128，
所以 (a) 更可能。**【推断】**

---

## 3. prefill：两臂不是同一个口径

rocm 臂给了两个长度，可以做线性拟合（固定开销 + 每 token 成本）：

```
TTFT(512) = 1610.38 ms
TTFT(640) = 1841.97 ms
→ 斜率 = (1841.97 − 1610.38) / 128 = 1.8093 ms/token   → 边际 552.6 t/s
→ 截距 = 1610.38 − 512 × 1.8093 = 684.0 ms
```

于是：

| 量 | rocm 臂 | DTK 臂 MBT 8192 | DTK 臂 MBT 2048 |
|---|---:|---:|---:|
| 固定开销 | 684.0 ms | （假设同 684） | （假设同 684） |
| 每 token | 1.8093 ms | **1.0992 ms** | 1.2347 ms |
| 边际吞吐 | 552.6 t/s | **909.8 t/s** | 810.0 t/s |
| @7469 折算 TTFT | 14.20 s | **8.894 s** | 9.906 s |
| @7469 折算吞吐 | 526 t/s | **839.8 t/s（实测）** | 754.0 t/s（实测） |

**三条判读：**

1. **"319 t/s" 不是 prefill 能力，是短 prompt 的固定开销占比。** 512 token 时 684 ms 的截距
   占 TTFT 的 **42.5%**；到了 7469 token 它只占 4.8%。所以 319 vs 840 的对比里，
   一半以上是口径差，不是路径差。
2. **同一长度上 DTK 真快 1.60×**（8.894 s vs 14.199 s）。这个差距是硬的：
   边际 1.099 vs 1.809 ms/token，**差 1.65×**，且不依赖截距假设的绝对值。
3. **MBT 8192 比 2048 好 10.2%（TTFT）/ 11.4%（吞吐）。** 7469 < 8192 ⇒ 单 chunk；
   2048 ⇒ 4 个 chunk，多出 3 个 chunk 边界，代价 **1012 ms / 3 = 337 ms 每个边界**。
   ⇒ chunk 边界在这台机器上很贵，**保持 MBT ≥ 最大 prompt 长度**是零成本的。

**prefill 离屋顶还有多远：** 27B / TP4 ⇒ 每 rank 每 token 13.5 GFLOP；
fp16 屋顶 19.61 TF/s ⇒ **0.6885 ms/token = 1452 t/s**。
DTK 的 839.8 t/s = **SOL 的 57.8%**，边际 909.8 = **62.7%**。
余量 ≤1.73×；其中 chunk 边界已在上表拿掉，GDN chunked 已在生产脚本里
（`serve_host_dflash2.sh:112`），所以剩下的主要是 rocBLAS fp16 的 M 形状效率。
**现实可达：950–1050 t/s（+13~25%）。**（10-24 E2C 实测 709 token 时 fp16 GEMM 只有 5.06 TFLOP/s/rank，
是屋顶的 26–28%，但那是最差形状；7469 已经跑到 11.3 TFLOP/s/rank。）**【实测+推断】**

---

## 4. "中和一下" 的三层含义，逐个判定

| # | 你可能想做的事 | 判定 |
|---|---|---|
| 1 | 用 DTK 的 prefill + rocm 的 decode | **伪命题**。decode 不受该开关控制（§2）。把 rocm 臂的 decode 数字搬过来，等于搬一个**短 prompt 的接受长度**。 |
| 2 | 两个都保留，按 prompt 长度切换 | **不可行**。切换发生在进程启动时（`_MODE` 在 import 期读一次，:50），不是请求级。且 `auto` 已经做按角色/按 M 的路由，两者在 7469 上等价。 |
| 3 | 吃下 DTK prefill，另外修 decode 的两件事（step 回归 + 长上下文接受长度） | **正解**，见 §6。 |

补充：`=1` 与 `auto` 在 7469 token 上**等价**（`M = 7469` 对每个角色都过阈值），
唯一差别是 `=1` 把未知角色的默认阈值从 4096 拉到 16、并多打告警。
生产脚本默认就是 `auto`（`serve_host_dflash2.sh:86`），**建议回 `auto`，`=1` 只在排障时用**。

---

## 5. "最新优化"清单：哪些已经在手、哪些还没吃

| # | 项 | 状态 | 增量 | 依据 |
|---|---|---|---|---|
| A | **图模式路线 A**（去 `--enforce-eager` + `cudagraph_mode=piecewise`，物化路由 compile-safe） | 已提交（`f5083195e`），**未部署** | step **−7.1%**、输出 +4%、TTFT 持平、c=4 +6.6% | 【实测】10-21 路线 A |
| B | gfx906 custom all-reduce（`VLLM_DCU_CUSTOM_AR=1`） | 代码在树（`envs.py:173`），生产仍传 `--disable-custom-all-reduce` | allreduce 占 step 20–23%，屋顶 ~13 ms ⇒ 可回收 19–24 ms | 【推断】无本栈 A/B |
| C | rocBLAS 非量化 GEMM（草稿塔 + lm_head） | 未动 | E2C 实测 36.9 ms / 步，屋顶 ~2 ms | 【实测 240.9 口径，需重测】 |
| D | W8A16 decode GEMV 带宽 | 未动 | 实测 10.268 GB / 46.1 ms = **175 GB/s**，同 run D2D 721.5 GiB/s（774 GB/s）⇒ **距屋顶 4.4×** | 【实测】microbench_windows.json |
| E | GDN 线性注意力 decode（曾占 97.9 ms） | **已收**（`VLLM_QWEN35_MTP_GDN_TRITON=1` 等） | 240.9 → 129 ms 的主要来源 | 【推断：由 −111.9 ms 反推】 |
| F' | **（v2 删除）** grouped verify / split-K | 撤项 | 在 Z100 上 ≤1%（§10.2②） | 【推断，KV 算术】 |
| F | GDN chunked prefill | 已在生产脚本 | 单层 14.11 → 0.758 ms（14.7×） | 【实测】 |
| G | MBT ≥ 最大 prompt | 你已在用 8192 | +11.4% prefill | 【实测】 |
| H | 长上下文接受长度（2.84 @ 7.5k vs 5.82 @ 512） | 未动 | **最多 2.05×**，是最大杠杆也是最难的 | 【实测】 |
| I | dot8 / int4（实测 2× dot4、9.11× fma） | 探针完成，未接 | 只作用于 GEMV 那 4.3%→ 的算子内层，整步 ≤2% | 【实测】微基准 |
| J | WMMA/MMA 层 | 不可用（gfx906 无 MFMA/WMMA 指令） | 0 | 【实测】汇编探针 |

**注意 A 的硬代价**（10-21 实测）：冷启动 99 s → **7.4 min**，宿主 RAM+swap 21 → 34 GiB，
KV −8%（806k → 742k tok）。这是"要不要上"的真正决策点，不是性能。

---

## 6. 投影

乘数都作用在 step 上，接受长度单独一列（两者正交：verify 的行数是 `k+1 = 8`，与接受长度无关）。

### 6.1 7469 上下文（你的实测：step 129 ms，acc 2.84）

| 阶段 | 变化 | step ms | ms/token | **tok/s** |
|---|---|---:|---:|---:|
| 现状 | — | 129.0 | 45.45 | **22.0** |
| +A 图路线 A | ×0.929 | 119.8 | 42.21 | **23.7** |
| +B/C/D 回收（中档） | ×0.78 | 100.6 | 35.45 | **28.2** |
| 上者 + 接受长度 4.0 | acc 4.0 | 100.6 | 25.16 | **39.8** |
| +B/C/D 回收（乐观） | ×0.62 | 80.0 | 28.18 | **35.5** |
| 乐观 + 接受长度 4.0 | acc 4.0 | 80.0 | 20.00 | **50.0** |

### 6.2 512 上下文（你的实测：step 167.4 ms，acc 5.82）

| 阶段 | step ms | ms/token | **tok/s** |
|---|---:|---:|---:|
| 现状 | 167.4 | 28.78 | **34.7** |
| +A 图路线 A | 155.5 | 26.73 | **37.4** |
| +B/C/D 中档 | 130.6 | 22.45 | **44.5** |
| +B/C/D 乐观 | 103.8 | 17.84 | **56.0** |
| *（若 167 是回归，修正到 129）* | *129.0* | *22.17* | ***45.1*** |

**6.2 最后一行是整个报告最值钱的一格**：如果 167.4 是回归而不是旧树，
那修掉它一项就等于做完 A+B+C+D 的全部（45.1 vs 44.5）。**先查它，再谈优化。**

### 6.3 prefill

| 项目 | @7469 TTFT | t/s |
|---|---:|---:|
| 现状 MBT 8192 | 8.894 s | 839.8 |
| MBT 2048（对照） | 9.906 s | 754.0 |
| 图路线 A | 不变（prefill 走 `NONE` 分支，实测持平） | 839.8 |
| rocBLAS fp16 形状审计（现实） | 7.2–7.9 s | **940–1030** |
| SOL 屋顶 | 5.14 s | 1452 |

**prefill 这一栏没有 2× 可拿。** 57.8% 的 SOL 已经是这台机器相当好的水平；
1.73× 是理论余量，现实 1.13–1.25×。

### 6.4 TTFT 随上下文长度（用两个同配方实测点拟合）

拿两个**都在当前配方上**的实测点做线性拟合，避免用 rocm 臂的软截距：

```
点1  TTFT(709 tok)  = 1350 ms   10-21 路线 A 实测（"512" 用例的真实前向是 709 token）
点2  TTFT(7469 tok) = 8894 ms   你 DTK 臂 839.8 t/s 折算
→ 斜率 = (8894 − 1350) / (7469 − 709) = 1.1160 ms/token   → 边际 896 t/s
→ 截距 = 1350 − 709 × 1.1160 = 559 ms
```

| prompt | 现状 TTFT | 现状 prefill t/s | 最终配方 TTFT | 最终 prefill t/s |
|---:|---:|---:|---:|---:|
| 512 | 1130 ms | 453 | **1016–1065 ms** | 481–504 |
| 709 | 1350 ms | 525 | 1192–1259 ms | 563–595 |
| 2048 | 2845 ms | 720 | 2388–2582 ms | 793–858 |
| 4096 | 5130 ms | 798 | 4216–4604 ms | 890–972 |
| 7469 | 8894 ms | 840 | **7227–7936 ms** | **941–1033** |
| 8192 | 9701 ms | 844 | 7873–8650 ms | 947–1040 |

> 截距 559 ms 是**软项**：它含调度、first-step、采样、HTTP 往返。
> 两次独立拟合给出 559（本式）与 684（rocm 臂）两个值，差 125 ms；
> 这只影响 ≤512 token 的短 prompt，7469 上两种截距只差 1.4%。

### 6.5 最终配方一页表（TTFT × decode）

| | 保守：只吃已实测项（图路线 A） | 中档：+ allreduce / 草稿塔 / lm_head / GEMV 回收 | 激进：+ 长上下文接受长度 4.0 |
|---|---:|---:|---:|
| **TTFT @512** | 1.10 s（709 用例实测 1.38→1.35 s） | **1.02–1.07 s** | 1.02–1.07 s |
| **TTFT @7469** | 8.89 s（长 prefill 走 `NONE`，图基本无影响） | **7.23–7.94 s** | 7.23–7.94 s |
| **decode @512** | 37.4 tok/s | **44.5 tok/s** | 44.5 tok/s |
| **decode @7469** | 23.7 tok/s | **28.2 tok/s** | **29–38 tok/s**（LABD/q16，§10.3②） |
| 把握度 | 实测（step 127.5→118.4 ms、TTFT 1.38→1.35 s） | 区间；前提是 §10.3① 的五相位 profile | 依赖 GEMV 的 M 敏感性（§10.3③），先测再立项 |

> 6.4 的"最终配方"列**已包含**图路线 A 的 −2.7% TTFT 修正（折进固定项，在 512 上约 −30 ms，在 7469 上可忽略）。

---

## 7. 三个决定性 A/B（把区间收窄到点估计）

| # | 做什么 | 唯一变量 | 判读 |
|---|---|---|---|
| **A1** | 同一条 7469 prompt，`VLLM_DCU_FP16_PREFILL=0` 跑一次 | 只翻 prefill 开关 | step 若仍是 **~131 ms**（比 129 高 1–2% 的 hook 成本）⇒ §2 成立，167 是旧树/回归；若掉到 ~167 ⇒ 我的读码结论错了，立刻改文 |
| **A2** | 同一条 7469 prompt 截到 512 token，同服务 | 只改 prompt 长度 | acc 若跳到 ~5.8 ⇒ H（接受长度）是长上下文塌陷；若仍 ~2.8 ⇒ 是这条 prompt 的内容，得换 prompt 集再测 |
| **A3** | 去 `--enforce-eager` + `cudagraph_mode=piecewise`，同 7469 prompt | 只翻图 | 预期 step 129 → ~120、TTFT ±2%；顺手验 `first materialization route engaged` 有没有 4 条 |

A1 和 A2 各 5 分钟，A3 要付 7.4 min 冷启动。**A1 先做**——它决定后面所有数字的基准。

---

## 8. 停止线 / 不建议投的

| 项 | 为什么不做 |
|---|---|
| `k > 7` 提接受长度 | ~~k=7 是 verify 组边界，别动 k~~ **已由 §10.2① 修正**：神经 k 不能加（block-8 是训练出来的），但 **lookup k 可以加**（1Cat 的 LABD 把第 8–15 位用 n-gram 填满，零训练）。 |
| WMMA / MMA 移植 | gfx906 无 `v_mfma_*` / `v_wmma_*`（汇编探针实测 "instruction not supported"）。唯一能碰的是 GEMV 内层 4.3% 那一块，整步 ≤2%。 |
| 把 prefill 再往 1Cat 的 align-chunking 上靠 | Z100 的 per-step 固定成本只占 0.24%，1Cat 的 −57% 不可迁移；真正的 chunk 收益已经由 MBT 8192 拿到（+11.4%）。 |
| 学 1Cat 的 mixed-batch decode | 有静默损坏先例（20/44 → 44/44 是坏的那一侧），c=1 场景零收益。 |

---

## 9. 一句话回答"能到多少"

**7469 上下文：现在 22.0 → 开图 23.7 → 回收 allreduce/草稿塔/GEMV 到 28 左右；
把长上下文接受长度从 2.84 修回 4.0，就是 40 左右。**
**512 上下文：34.7 → 37.4 → 44.5（若 167 ms 的 step 回归被修掉，起步就是 45）。**
**TTFT：7469 token 8.89 s → 7.23–7.94 s；512 token 1.13 s → 1.02–1.07 s。prefill 940–1030 t/s，没有 2×。**

而这三条里唯一"已经做完、只差部署"的是图路线 A；唯一可能白捡 8 tok/s 的是那个 167→129 的问号。

---

## 10. 结合 1Cat SM70 经验的修正（v2）

> 依据：`.refs/1cat-vllm` @ `c4f6245`（origin/main `6a14319`）的
> `sm70_dflash2_lookup_augmented.md`、`sm70_dflash2_long_verify_decay.md`、
> `sm70_dflash2_acceptance_rd.md`、`sm70_quasar_dflash2_*`。
> 本节**修正** §6.5、§7、§8 的若干净结论，不推翻 §1–§5 的换算。
> **⚠ 本节是读文档得出的；§11 是读代码得出的，已改掉本节三条判断**
> （不需要写扫描核 / q16 有三个行号级阻塞 / P0-b 应换成 `GEMV_MAX_M=1` 的 A/B）。
> 冲突时以 §11 为准。

### 10.1 先拿 1Cat 的 V100 数字做一次校准

| | 1Cat（4×V100，27B） | Z100（4×Z100L，27B） | 倍数 |
|---|---:|---:|---:|
| decode 单位 | complete round **18.6 ms**（短上下文；k=7/q8） | step **129 ms** | 6.9× |
| 权重精度 | NVFP4/FP8 混合 ⇒ **3.38 GB/卡** | FP8 ⇒ **6.75 GB/卡** | 2.0× |
| 归一化 | **5.51 ms/GB** | **19.11 ms/GB** | **3.47×** |

**判读：把量化差异扣掉，栈本身还差 3.47×。**
⇒ §6.5 里"中档 ×0.78"（把 129 压到 100）大概率**偏保守**——1Cat 证明这个量级的 step
在同类模型上是可达的，只是需要把栈铺完（FULL graph、融合 verifier、5 相位核算）。
⇒ 但 §6.5 的"激进"那一格（39.8 tok/s）此前被我标成"无本栈证据"，**现在有机制了**，见 10.3。

1Cat 的 3.749（pooled，score 工作负载）/ 4.896（LABD mixed-run）/ 7.647（128K 固定诊断请求）
三个接受长度同时存在，说明**他们的接受长度同样随工作负载在 3.7–7.6 之间摆动**，
"1Cat 在长上下文不掉接受长度"这句话**不成立**，不能拿来当目标。

### 10.2 三条必须改的判定

**① §8 的"`k > 7` 提接受长度 → 不做"要改。**
原文说"要么换 drafter，要么别动 k"，这只对**神经** draft 成立。1Cat 的做法是：

- DFlash2 神经 checkpoint 仍只产 **7 个 draft 位置**（block-8，训练出来的，改不了）；
- 多出来的**第 8–15 位由 prompt lookup（n-gram 后缀匹配）填**，不需要任何训练；
- `num_speculative_tokens=15` → **q16** 验证；控制器在"连续两次强复制信号"后才进 q16，
  普通/低置信流量留在 q8（`VLLM_DFLASH2_LOOKUP_ADAPTIVE=0` 可强制固定 q16）。
- 实测：q16 轮里 **第 8–15 位在 15 轮中被接受 14 次**（≈93%），mixed-run 接受长度 **4.896**，
  steady decode **316.27 tok/s / TPOT 3.162 ms**。

⇒ 改写为：**神经 k 不能加；lookup k 可以加。** 见 10.3。

**② §5 表里 v1 报告的"B3 grouped verify"要撤（有定量理由）。**
1Cat 分组验证的收益：32K round 36.31 → 26.36 ms（−27%）、128K 69.86 → 40.78 ms（−42%），
split 40→80 再给 128K −10%。**但那是 Hkv=1、D=256、128K 上下文下 KV 读主导的场景。**

Z100 的算术（TP4、GQA、head_dim 128、16 个全注意力层、上下文 7469）：

| KV 头/rank | 字节 | 每 step KV 读量 | @775 GB/s |
|---:|---:|---:|---:|
| 1 | 1 (fp8) | 29.2 MB | 0.038 ms |
| 1 | 2 (fp16) | 58.4 MB | 0.075 ms |
| 2 | 2 (fp16) | 116.7 MB | 0.151 ms |

即使 verify **完全不做** KV tile 共享（读 8 遍），上限也只是 0.3–1.2 ms，
占 129 ms 的 **0.2–0.9%**。
⇒ **grouped verify / split-K 在 Z100 上值 ≤1%，从计划里删掉。**
（KV 在 Z100 上的价值是**容量**不是带宽，二者别混。）

**③ §3 的"337 ms/chunk 边界"从口径提醒升级为 bug 项。**
1Cat 的生产门禁就是 `max_num_batched_tokens=4096` + 平衡切分 + Mamba align，
他们没有可见的边界惩罚。我们 7469 token 从 1 块切成 4 块要多付 **1012 ms（3×337 ms）**。
最可能的解释是 **GDN prefill 每个 chunk 重算前缀状态**（48 层 × 4 chunk）。
判定实验：固定 4096 token，MBT 取 1024/2048/4096，斜率即每边界成本。
若确认，它在**并发/chunked 服务**下付同样的钱，不止影响单请求 TTFT。

### 10.3 三条要加进计划的项

**① `complete round` 五相位核算（1Cat 的门禁单位）**
1Cat 每一份优化报告都按 `draft / draft→target / target / target→draft / complete round`
五个数报（见 `sm70_dflash2_long_verify_decay.md` §"Closed trace attribution"）。
我们现在的 129 ms 是一个黑箱，所以 §6.5 的中档只能是猜。
**这五项一出来，"中档 ×0.78"就能变成点估计。** 零代码，一次 profiler。

**② LABD / `ngram_assist`（这是 39.8 那格唯一的机制）**
`q16` + adaptive LABD 在本栈的预期：

| q16 对 step 的代价 | 接受长度 4.9 | 接受长度 5.5 |
|---:|---:|---:|
| ×1.0（GEMV 对 M 不敏感） | **38.0 tok/s** | 42.6 tok/s |
| ×1.3 | 29.2 tok/s | 32.8 tok/s |

⇒ §6.5 "激进"那一格从裸的 39.8 改成 **29–38 tok/s**，并把成败押在下面这个微基准上。

**③ GEMV 的 M 扫描（决定 q16 是否免费）—— 本次最便宜的决胜实验**
E2C 微基准：M=8 时搬 10.268 GB 用 58.5 ms = **175 GB/s**，是同 run D2D 的 4.4 分之一。
**如果它真是权重读主导，M 从 8 加到 16 的增量接近零**，那 q16 就是白拿的；
如果它其实是占用率/延迟受限，M=16 会近似翻倍，LABD 就划不来。
用 `.s24/probes/w8a16_perf_check.py` 把 M 列表改成 `[8, 12, 16, 24]` 即可，分钟级。

与它配对的是 CPU 侧的 LABD 可行性判定——已写成
[`tools/dcu/labd_lookup_probe.py`](../../tools/dcu/labd_lookup_probe.py)：
输入 `{"prompt_ids": [...], "output_ids": [...]}`，按 1Cat 的"最长后缀匹配 + 近因打破平局"
贪心模拟 lookup 轮次，输出 `lookup_coverage` 与 `mean_extra_tokens_per_round`。
判定线：覆盖率 ≥40% 立项、15–40% 只在强复制信号下启用、<15% 不立项。

**④ `ExecutionPolicy` 策略块 + 策略指纹**
1Cat 用 X-macro 表 + `ExecutionPolicy` 把散落的 `getenv` 收成一份可打印、可快照的策略。
我们有 ~20 个 `VLLM_DCU_*` / `VLLM_GFX906_*` 开关，本次会话已经在"哪套 env"上翻车两次
（§2 的 167 vs 129 就是这一类）。做法：启动打印一行完整策略 + 一个策略哈希，
所有 A/B 报告引用哈希。**收益不是性能，是让后面每个数字可追溯。**

### 10.4 修订后的执行顺序

| 序 | 动作 | 成本 | 值 | 依赖 |
|---|---|---|---|---|
| **P0-a** | 同 7469 prompt 跑 `FP16_PREFILL=0`（§7 A1） | 5 min | 定基准（167 vs 129） | — |
| **P0-b** | GEMV M=`[8,12,16,24]` 微基准（10.3③） | ~30 min | 决定 q16/LABD 立项与否 | — |
| **P0-c** | 离线 n-gram 命中率：[`tools/dcu/labd_lookup_probe.py`](../../tools/dcu/labd_lookup_probe.py) | 秒 | LABD 对该 prompt 是否可能有效 | 需 prompt/output token ids |
| **P1-a** | 图路线 A 上生产（已实测 −7.1%） | 半天 | 23.7 / 37.4 tok/s | — |
| **P1-b** | complete round 五相位 profile（10.3①） | 半天 | 把"中档"从猜变测 | P1-a |
| **P1-c** | 接受长度扫 1K/2K/4K/8K（1Cat 门禁式） | 半天 | 判定 2.84 是长度还是内容 | — |
| **P2** | 按 P1-b 的实测分配做回收（allreduce / 草稿塔 / GEMV） | 3–5 天 | 28 左右 | P1-b |
| **P2'** | LABD/q16（仅当 P0-b 与 P0-c 都为正） | 5–10 天 | 29–38 | P0-b, P0-c |
| **P3** | 337 ms/chunk 边界的根因（GDN 前缀重算？） | 2–3 天 | TTFT + 并发 | — |
| **P4** | rocBLAS fp16 形状审计 | 2–3 天 | prefill 940–1030 t/s | — |

**优先级变化（相对 §6.5）**：LABD/q16 从"最不确定、别立项"升到 **P2'，且有两个零成本前置判定**；
grouped verify 从 P1 降到**删除**；五相位 profile 从"建议"升到 **P1-b 强制门禁**。

### 10.5 从 1Cat 继承的四条硬规矩（不是建议）

1. **任一优化必须在 1K/2K/4K/8K 四个上下文上都不退化**（1Cat 是 1K/32K/128K/256K）。
2. **接受长度与 latency 分开报**，禁止用吞吐一个数代替。
3. **微基准测出方向，服务级才算账**——1Cat 的 `sm70_dflash2_*` 里每一份都同时给
   operator 微基准和 full-model round，两者差 >5% 就回去查。
4. **诊断内核不得进被测量路径**（1Cat 明确要求 diagnostic 默认关、零分配）。

---

## 11. 读码后的修正（v3）

> §10 是**读文档**得出的。本节是**读代码**得出的，改掉了 §10 的三条判断，并发现一件
> §1–§10 全程漏掉的、零成本的 A/B。
> 对照：`.refs/1cat-vllm` @ `c4f6245` vs 本树 `dtk` @ HEAD。

### 11.1 LABD 的真实实现（代码级，不是文档级）

`vllm/v1/worker/gpu/spec_decode/dflash2/lookup.py`（10.5 KB，2 个 Triton 核）：

- `_suffix_lookup_kernel`（:20）——**每请求一个 program**，以 `BLOCK=1024`、`num_warps=4`
  扫整段历史（`search_max` 默认 `1<<30`，即全扫），在 `NMIN..NMAX` 上取最长后缀。
  打分 `match_len * 2^32 + candidate_end`，一次 `max` 同时选"最长"和"最近"。
  注释明确**允许候选匹配与当前后缀重叠**（"preserves periodic patterns such as
  indentation, list markers, and code fences"）。
- `_fuse_draft_kernel`（:115）——准入规则，原文三行：

```python
take_head = (match_len >= nstrong) | ((match_len >= nmin) & (agreement >= agree_min))   # :161
take_tail = (match_len >= nmin_tail) & (take_head | (agreement >= draft_block))          # :165
has_tail  = (match_len >= long_min) & (valid > draft_block) & take_tail                  # :184
```

  `agreement` = 神经 draft 与 lookup 在前 `draft_block` 位上**首次分歧的位置**（全一致则为 `draft_block`）。
  尾行（`offsets >= draft_block`）只有在**头被采用**或**神经 draft 全一致**时才能填。
  `has_tail` 喂给 q8/q16 控制器。

- 默认常量（`vllm/envs.py`）：`NSTRONG=6`(:10330)、**`AGREE=0`**(:10343)、`NMIN_TAIL=4`(:10356)、
  `LONG_MIN=6`(:10369)、`SEARCH=1<<30`(:10382)、`ENTRY_STREAK=2`(:10395)、`STICKY=3`(:10408)。
  **`AGREE=0` 意味着 `agreement >= 0` 恒真 ⇒ 头那一路退化成"`match_len >= NMIN` 就换"。**
  文档写的"连续两次强复制信号"门禁只控制 **q16 宽度**，不控制头替换——这是文档看不出来、代码看得出来的差别。
- 控制器（`:306`、`:322`）：`wants_long = take & (emitted >= full_emitted)`，
  再过 `ENTRY_STREAK` 次连击才 latch；**`sticky_steps` 仅在 `num_reqs == 1` 时生效**，
  并发下直接退回 q8。这就是那个"多请求 miss 不能靠 q16 蒙"的安全阀。

### 11.2 Z100 已经有扫描核，而且比 1Cat 的更并行

本树 `vllm/v1/worker/gpu/spec_decode/ngram/speculator.py` 已有
`_ngram_scan_kernel`(:21) + `_ngram_finalize_kernel`(:91) + `NgramGPUSpeculator`(:151)，
打分方式与 1Cat **同构**（`cand = n_iter * Lp1 + pos + 1` vs 1Cat 的
`match_len * 2^32 + candidate_end`），同样是"最长 n、最右 pos"。

差别在并行度：1Cat 是**单 program 循环整段历史**；Z100 是**二维 grid `(num_reqs, n_blocks)`**
按块并行 + 一个 finalize 归约，且块整块越界就提前返回
（`blk * BLOCK_L <= seq_len - MIN_N - 1`，7469 token 时 1024 块里只有约 30 块真跑）。

**⇒ 不需要写扫描核。** §10.3② 里"需要一个 Triton lookup 核"是错的。

而且 `method=ngram_gpu` 在本树**已经可选**（`vllm/config/speculative.py:67,1184,1206,1962`）。
⇒ P0-c 不必跑我写的离线脚本：**直接在真机上用 `ngram_gpu` 量这条 7469 prompt 的 n-gram 接受长度**
（真 tokenizer、真 sampler、真拒绝采样）。离线脚本降级为"没卡时的替代"。

缺的只有三样：`fuse_draft` 的融合准入、`draft_block` 与验证宽度的解耦、q8/q16 控制器。

### 11.3 q16 在本树上的三个硬阻塞（都有行号）

**阻塞 1 —— 注意力把 q16 判成 prefill。**
`vllm/v1/attention/ops/triton_unified_attention.py:67`：`GFX906_VERIFY_MAX_QUERY_LEN = 8`（硬编码）。
`gfx906_small_q_decode`(:243) 要求 `max_seqlen_q <= GFX906_VERIFY_MAX_QUERY_LEN`(:262)。
q16 的 `max_seqlen_q = 16` 直接掉出 split-KV decode 路径，退回 2D 单 CTA——正是
`.s22/S22_LONGCTX_DECODE.md:150` 记录过的"verify 被误判成 prefill"那个 bug。
（`:1478-1483` 还有第二处同源判断。）

**阻塞 2 —— M=16 正好踩在 fp16 物化路由的阈值上。**
`gfx906_fp16_prefill.py:132` `_FORCE_MIN_M = 16`，`:441` 的闸门是 `m < _min_m(...) or m < _FORCE_MIN_M`。
q16 的 decode `M = 1 + 15 = 16` ⇒ **两个比较都是 False ⇒ 路由生效**，
每层权重当步 dequant 成 fp16 再走 rocBLAS。`auto` 模式下更糟：
`_ROLE_MIN_M`(:73) 里 `down_proj`/`o_proj`/`out_proj`/`in_proj_qkvz`/`in_proj_z` 的门槛都是 **16**，
q16 会有 4–5 个角色被静默切走。源码注释"decode (M <= 8) never routes"只在 `k <= 7` 下成立。

**阻塞 3 —— 验证宽度和 draft 宽度是同一个变量。**
1Cat 的 `dflash/speculator.py`：`self.draft_block = get_dflash_model_draft_tokens(...)`(:58)，
`self.num_query_per_req = 1 + self.draft_block`(:59)，并在 `:61-70` 把 `max_num_tokens`
扩到 `max_num_reqs * (1 + draft_block)`，免得 draft 的 query 批次吃掉 target 的 prefill 预算。
本树 `dflash/speculator.py:71` 是 `self.num_query_per_req = 1 + self.num_speculative_steps`
——**本树没有 `draft_block` 这个符号**，验证宽度 ≡ 模型宽度。
调用点倒是现成的：1Cat 在 `propose()` 末尾（`:1312-1315`）做
`zero_()` 尾巴 → `_cache_draft_logits` → `_apply_lookup`；本树 `propose()` 在
`dflash/speculator.py:441`，同一个位置。

### 11.4 被漏掉的一项：`VLLM_DCU_FP8_GEMV_MAX_M=1`（零代码，算子级 7.38×，端到端从未跑过）

`gfx906_fp8.py:73-83` 的原文注释：

> the GEMV cost grows ~linearly with M (M=2 ~2.1x, M=4 ~3.9x) while the tiled GEMM is M-flat
> (M=1..4 all ~0.94x of the M=1 GEMV) -- so speculative-decoding verify (M = 1 + num_spec_tokens)
> **wants the GEMM path**. Default keeps the old policy; set `VLLM_DCU_FP8_GEMV_MAX_M=1` …

`docs/archive/result/z100-18c-gemm-dispatch-m-flat-result.md` 的实测：

- **GEMM 路径在 `M=1..8` 完全平坦**（0.5884 → 0.5900 ms，波动 0.3%）；
- **GEMV 在 M=8 涨到 7.38×**；
- `M=1` 时 GEMV 反而快 30% ⇒ 最优策略是 `M=1 → GEMV、M≥2 → GEMM`，即 `_GEMV_MAX_M=1`；
- **`:80` 明写"端到端 A/B 未跑（卡被电池占用）"。**

而生产脚本 `serve_host_dflash2.sh:80` 用的还是 **8**，也就是让**验证形状 M=8**（正好是最坏点）
走 GEMV。这解释了 p3 计划里"GEMV 46.1 ms、距屋顶 2.2–4.4×"的一大半——
**它不是需要重写内核，而是需要换一条已经存在的平坦路径。**

两条限定：`_GEMV_MS_MAX = 8`(:82) 是**编译期上限**（`kMsMax`），所以 `=16` 会被静默截到 8，
要用 M=16 必须重编 `fp8_emu_gemm.cu`；且 S18 是在 W8A8 fp8-emu 路径上测的，
生产跑的 W8A16 共用同一阈值但**没有单独复测**。

### 11.5 对 §10 的三处修订

| §10 的说法 | 修订 |
|---|---|
| "需要一个 Triton lookup 核"（隐含） | **不用写**：本树已有更并行的 `_ngram_scan_kernel`。缺的是融合与宽度解耦。 |
| "LABD/q16 5–10 天" | 拆开：`draft_block` 解耦 + `propose()` 挂钩 ≈ 1 天；`lookup.py` 移植 ≈ 1–2 天；**三个阻塞各 1–2 天**；控制器 ≈ 1 天。合计 **4–7 天**，但必须先过 §11.3 的三个阻塞。 |
| P0-b "GEMV 的 M=8/16/24 微基准" | **降级**：M 依赖已经被 S18 测过了（GEMV 线性、GEMM 平坦）。改成零成本的 **`GEMV_MAX_M=8` vs `=1` 端到端 A/B**。 |

### 11.6 修订后的 P0（全部零代码或环境变量）

| 序 | 动作 | 成本 | 赌注 |
|---|---|---|---|
| **P0-a** | `FP16_PREFILL=0` vs `=1`，同 7469 prompt | 5 min | §2 的基准（167 vs 129） |
| **P0-b′** | `GEMV_MAX_M=1`（替换 §10 的 P0-b） | 1 次 A/B | 算子级已测 **7.38×**，e2e 未知 ⇒ 可能是全表最大单项 |
| **P0-c′** | `--speculative-config '{"method":"ngram_gpu","prompt_lookup_min":…,"prompt_lookup_max":…}'`，同 7469 prompt | 1 次 | 这条 prompt 的纯 n-gram 接受长度（LABD 的物理上限） |
| **P0-d** | 确认 P0-b′ 与 P0-c′ 不互斥（`ngram_gpu` 会取代 DFlash2，只能二选一测） | — | — |

**P0-b′ 的判读**：若 step 掉 15% 以上 → 立刻进生产配方，并把 §11.3 阻塞 2 的修法从
"抬阈值"改成"确保 M≥2 走 GEMM"；若几乎不动 → 说明 W8A16 路径的 GEMV 已经不是瓶颈，
p3 计划的 R2 项（GEMV 46.1 ms）要整体重估。

**P0-c′ 的判读**：若纯 n-gram 在这条 prompt 上的接受长度 ≥ 2，LABD 才值得投 4–7 天；
若 ≈ 0，`lookup_coverage` 也救不了，直接砍掉 §10.3②。
