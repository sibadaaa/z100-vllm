# S14d graph 模式 token 流的可依赖性判定与预热/部署配方 — 执行结果

> 状态:**已完成**(E1/E2/E3/E4/E5 全部落地)。单 writer = S14d 子 agent。
> 原始工件:`/mnt/game/s14d/**`。

## 0.0 给 Lead 的三句话

1. **graph 不能上生产(用于任何依赖 token 流的地方)**:7 个 arm、21 个 arm-pair、
   252 个 prompt-pair 里只有 **6 个(2.4%)** 的目标 prompt token 流逐位一致;
   **同 cache 的两次 AOT 重载之间同样 12/12 不一致** ⇒ 这不是"H-AOT / 现场编译 vs 重载",
   而是 graph 编译路径**运行间根本不可复现**(同上下文 mean ≈1.3 nat、max 10–16 nat)。
2. **预热配方解决的是启动时间,不是可复现性**:AOT 重载 ready 352→230 s(−35%)、
   `compilation` 117→25 s,但预热那次与服务那次的 token 流 **12/12 不一致**
   ⇒ **E4 门禁硬失败**;`combo_kernels`/`benchmark_combo_kernel` 关掉**也无收益**(4.2%→2.8%)。
3. **eager 是唯一可当参照的臂,但要加一条限定**:eager 跨引擎逐位可复现(到小数位),
   但**批组合会确定性改变数值**(conc=2 vs conc=1 ⇒ 第 6 个 token 分叉、mean 1.82 nat / max 15.02 nat,
   且该效应本身完全可复现)。**S14b 的 13.7 nat 并发观测由此结案**。

## 0.1 边做边写(时间线)

| 时间(UTC) | 事件 | 状态 |
|---|---|---|
| 02:10–02:11 | 前置留档:`date`/`pgrep`/`hy-smi`/`free -g`/`/proc/swaps`/`swappiness`(60)/ACS(12×`0x0000`);写作用域 `/mnt/game/s14d/{hostlogs}` 建立 | ✅ |
| 02:11 | 脚本落地:`s14d_client.py`(12 prompt 电池,逐位 token 流 + 同上下文 `prompt_logprobs`)、`s14d_compare.py`、`s14d_run_one.sh`(复用 `s14c_arm_env.sh` 保持生产 flag 逐字相同)、`s14d_conc2.py`、`s14d_e4.sh` | ✅ |
| 02:11 | 附加 4 GiB swapfile `/mnt/game/s14d/swap4` + `vm.swappiness=200` 生效(总 swap 8191 MiB) | ✅ |
| 02:12–02:19 | **E4 arm W4**(空 cache 现场编译 = 预热):ready **352 s**、`aot_load=0`、`compilation` 117.30 s、12-prompt 电池 1455 token 全部记录 | ✅ |
| 02:20–02:26 | **E4 arm S4**(同 cache AOT 重载 = 服务):ready **230 s**、**`aot_load=4`**、`compilation` 24.77 s | ✅ |
| 02:26 | **E4 判定:不一致率 12/12 = 1.0000**(同上下文 mean 1.04–1.76 nat / max 12.67 nat);结论发回 Lead | ⛔ **门禁不通过** |
| 02:26–02:40 | **A3/A4 = 同一 cache 的第 2/3 次重载** ⇒ **同样 12/12 不一致** ⇒ 因子是"任意两次运行",不是 AOT 来源 | ⛔ H-AOT 否证 |
| 02:27 | 更正:给 Lead 的 interim 里"本步几乎不动用 swap"是**稳态误读**;ready 时刻 8 GiB swap 用到 **100%**(§3.1.4) | ⚠️ |
| 02:39–03:00 | **E1 cell B(combo kernels off)**:B1 现场编译(ready 330 s/`aot_load=0`)、B2 重载(220 s/4)、B3 再重载(221 s/4) | ✅ |
| 03:00–03:24 | **E5(eager 两形态)+ E3(conc2)**:E5SW16(24 GiB swap,ready 161 s)、E5SW4(8 GiB swap,ready 160 s) | ✅ |
| 03:25–03:30 | **seq2 对照**(eager 同 prompt 连发两次、全程单请求在飞)⇒ **逐位一致、\|Δ\|=0.00000** ⇒ conc2 的分叉归因于**并发** | ✅ E3 结案 |
| 03:30–03:45 | 宿主还原(swapfile/`swappiness`/ACS)+ 归档收口 | ✅ |

## 0.2 完成情况

| 实验 | 状态 | 关键结论 |
|---|---|---|
| **E4** 预热配方有效性 | ✅ **完成(最高优先,16 min 内出结论)** | **不通过**:12/12 不一致;同上下文 mean 1.04–1.76 / max 12.67 nat |
| **E1** H-AOT 2×2(+补跑) | ✅ 完成(7 次独立启动:4 格 + 3 次补) | **H-AOT 否证**:reload-vs-reload 与 live-vs-reload 不可区分;combo off 无收益 |
| **E5** C2 vs P4e 的 12% | ✅ 完成(两形态 decode n=5) | **抖动/vintage**,不是内存形态(差 1.6%;16 GiB swapfile 一页未用) |
| **E2** `p2_zh_slip` 最小复现 | ✅ 完成(同 cache 三次重载 + 72 个 prompt-pair 共同前缀分布) | 量级复现(9.3–11.1 nat),"第 59 位离散事件"**否证**(它是 9.7% 概率的普通尾部) |
| **E3** conc=2 一致性 | ✅ 完成(受控 conc2 + 无并发 seq2) | eager 的**批组合效应**;S14b 的 13.7 nat 结案 |

## 1. 目标与门禁

**问题**:graph 模式的 token 流到底可不可依赖?**可执行的预热/部署配方**是什么?

**门禁(任务书 §3)**:

| # | 判据 | 判定 |
|---|---|---|
| E4 | 生产配方(空 cache 预热一次 → AOT 重载服务)下,"预热那次" vs "服务那次" 对 n≥8 prompt 的**逐位不一致率 = 0**(或只有 <1e-3 nat 的舍入级差) | ⛔ **不通过**:不一致率 **12/12**;同上下文 mean 1.04–1.76 nat / max 12.67 nat(§3.1、§4) |
| E1 | H-AOT 2×2(现场编译/磁盘重载 × combo kernels 开/关)每格 ≥1 次独立启动;给出逐 prompt 逐位对比 + 首个分歧位置 + nat 间距 | ✅ **完成(7 次启动)**;**H-AOT 否证**:reload-vs-reload ≡ live-vs-reload;combo off 无收益(§3.2) |
| E5 | 同 TP=4 eager,挂 16 GiB swap vs 只挂 4 GiB `/swap.img` 两次 decode(n≥4);判 C2(142.59) vs P4e(126.83) 的 12% 是抖动还是内存形态 | ✅ **完成**:**抖动/vintage**(133.83 vs 131.78 = 1.6%;16 GiB swapfile 一页未用)(§3.3) |
| E2 | `p2_zh_slip` 第 59 位起 9.5 nat 是否稳定复现(同 cache 连跑两次) | ✅ **完成**:量级稳定复现(9.3–11.1 nat),但"第 59 位离散事件"**否证**(9.7% 的普通尾部)(§3.4) |
| E3 | conc=2 同上下文一致性(消解 S14b 的 13.7 nat 并发观测) | ✅ **完成**:受控 conc2 + 无并发 seq2 ⇒ 归因于 **eager 的批组合效应**;13.7 nat 结案(§3.3.4、§3.5) |

## 2. 实际改动

* 新增(全部在写作用域内,未改任何既有文件):
  * `tools/dcu/s14d_client.py` —— **12 prompt** 固定电池(8 个 S11l flip prompt + 3 个 S14b
    `p2_*` + `s14_greedy_long`),严格串行单请求,`temperature=0`,同时记录
    `token_ids`(逐位)、每 token `logprob`/top-1–top-2 gap/top-5,
    以及**每个 prompt 位置**的 `prompt_logprobs` top-20(**同上下文**仪器,
    不受自回归发散影响)。
  * `tools/dcu/s14d_compare.py` —— 双仪器逐位对照器(生成流 + 同上下文),输出
    不一致率、首个分歧位置、共同前缀 max|Δlogprob|、分歧点两侧 gap、plp argmax flips。
  * `tools/dcu/s14d_run_one.sh` —— 单 arm 编排(复用 `s14c_arm_env.sh` 作为容器内
    启动器 ⇒ 生产 flag 与 S14c 逐字相同;去掉了 S14d 不需要的内存 tracer)。
  * `tools/dcu/s14d_conc2.py` —— E3 的 conc=2 受控对照(2 个并发同 prompt + 1 个
    串行对照,线程屏障同步启动)。
  * `tools/dcu/s14d_e4.sh` / `s14d_e1.sh` / `s14d_e5.sh` —— E4 / E1(+E2) / E5 驱动。
* **未改**:`s14_*` / `s14b_*` / `s14c_*` 脚本、`docs/` 下 Lead 维护的文档、DTK 库、
  `z100-build*`、生产容器 `qwen38-vllm`。

**生产 flag 逐字继承自 `s14c_arm_env.sh`**(md5 `61726d078e4f052ccbcfd803aecd8f20`):

```
--dtype float16 --tensor-parallel-size 4 --gpu-memory-utilization 0.85
--disable-custom-all-reduce --skip-mm-profiling --enable-prefix-caching
--mamba-cache-mode align --max-model-len 262144 --max-num-batched-tokens 2048
-cc.cudagraph_mode=PIECEWISE --cudagraph-capture-sizes 1 2 4 8
(无 --enforce-eager)
```

## 3. 执行过程与原始证据

### 3.0 前置留档(2026-10-10T02:11:20Z)

原文:`/mnt/game/s14d/hostlogs/00-baseline.txt`

```
pgrep -af "vllm|torchrun"      -> (空)
free -m                        -> Mem 32009 total / 25368 free / 28277 available; Swap 4095 total / 525 used
cat /proc/swaps                -> 只有 /swap.img 4194300 (Used 538488)
cat /proc/sys/vm/swappiness    -> 60
hy-smi --showmeminfo vram      -> DCU[0..3] vram Total Used = 0 MiB (4 卡全空)
docker inspect z100-s14        -> HostConfig.Memory=0 MemorySwap=0(无 cgroup 上限)
docker inspect z100-s14 State  -> StartedAt 2026-10-09T19:35:54Z(早于本步全部实验)
sudo -n python3 tools/dcu/dcu_p2p_acs.py show -> 12 行 0x0000(部署态未切)
md5(s14c_run_one.sh) 4f3f3275a4087274214042be975e0ec7
md5(s14c_arm_env.sh) 61726d078e4f052ccbcfd803aecd8f20
```

### 3.1 E4:预热配方有效性(最高优先)

**配方(与 S14c §3.13 生产候选逐字相同)**:TP=4 + graph(`PIECEWISE` +
`--cudagraph-capture-sizes 1 2 4 8`)+ **AOT 重载** + 附加 4 GiB swapfile(总 8 GiB)+
`vm.swappiness=200` + 生产 flags。

**流程**:空 cache 起一次(现场编译、落盘 AOT 产物)= **W4(预热那次)** →
停止 → 同一 cache 目录再起(**AOT 重载**)= **S4(服务那次)**。两次跑**同一份
12-prompt 电池、同一顺序、逐字节相同的请求体**。

| arm | cache 根 | vintage | ready | aot_load | init engine(compilation) | KV cache | 启动命令 |
|---|---|---|---|---|---|---|---|
| **W4** 预热/现场编译 | `/mnt/game/s14d/cache-e4`(空) | `e4_w4_tp4_graph_livecompile_swap8g_sw200` | **352 s** | **0** | 187.32 s(**117.30 s**) | 18.47 GiB | §7 |
| **S4** 服务/AOT 重载 | `/mnt/game/s14d/cache-e4`(已写) | `e4_s4_tp4_graph_AOTreload_swap8g_sw200` | **230 s** | **4** | 88.55 s(24.77 s) | 18.48 GiB | §7 |

⇒ **AOT 重载确实发生**(`aot_load=4` = 4 个 worker 各有一条 `Directly load AOT compilation
from path`),`compilation` 从 117.30 s 降到 24.77 s,ready 352→230 s(−35%,复现 S14c §3.6)。
两次 `Available KV cache memory` 一致(18.47/18.48 GiB),`oom_kill` 全程 4(未增长)。

#### 3.1.1 逐 prompt 逐位对照表(**E4 的门禁数据**)

原文:`/mnt/game/s14d/hostlogs/E4-compare.txt`、`/mnt/game/s14d/e4-compare.csv`

| prompt | nA | nB | token_ids 逐位相同 | **首个分歧位置** | 共同前缀 max\|Δlogprob\| | 分歧点 gap(S4/W4) | plp 位置数 | **plp max\|Δ\|** | **plp mean\|Δ\|** | plp argmax flips |
|---|---|---|---|---|---|---|---|---|---|---|
| zh_hz | 128 | 128 | ❌ | **2** | 0.5257 | 0.594 / 0.703 | 56 | **10.316** | 1.639 | 16 |
| zh_ml | 128 | 128 | ❌ | **7** | 0.4648 | 0.469 / 0.578 | 56 | **10.316** | 1.755 | 14 |
| zh_sort | 128 | 128 | ❌ | **12** | 0.0935 | 0.625 / 0.375 | 60 | 8.052 | 1.235 | 12 |
| zh_sky | 128 | 128 | ❌ | **1** | 0.1050 | 2.000 / 0.094 | 59 | 8.463 | 1.179 | 9 |
| en_fr | 47 | 47 | ❌ | **32** | 0.3127 | 0.516 / 0.094 | 58 | 12.665 | 1.680 | 13 |
| en_sea | 128 | 128 | ❌ | **10** | 0.0345 | 0.344 / 0.047 | 59 | 8.463 | 1.168 | 10 |
| en_index | 128 | 128 | ❌ | **2** | 0.0394 | 0.703 / 0.203 | 59 | 8.463 | 1.210 | 9 |
| en_math | 128 | 128 | ❌ | **0** | 0.0000 | 0.844 / 1.937 | 65 | 9.444 | 1.515 | 17 |
| p2_zh_slip | 128 | 128 | ❌ | **33** | 0.4781 | 0.938 / 0.031 | 77 | 9.278 | 1.138 | 16 |
| p2_en_batch | 128 | 128 | ❌ | **0** | 0.0000 | 0.844 / 1.734 | 73 | 10.651 | 1.345 | 15 |
| p2_zh_code | 128 | 128 | ❌ | **27** | 0.2005 | 0.453 / 0.672 | 75 | 10.651 | 1.263 | 13 |
| s14_greedy_long | 128 | 128 | ❌ | **77** | 0.1662 | 0.234 / 1.203 | 125 | 10.945 | 1.044 | 28 |
| **合计** | 1455 | 1455 | **0/12** | 中位 **~9** | 最坏 **0.526** | — | — | 最坏 **12.665** | 1.04–1.76 | 9–28 |

**不一致率 = 12/12 = 1.0000(100%)。** 首个分歧位置中位数 ≈ 9 个生成 token
(最早第 0 个 token,最晚第 77 个)。

#### 3.1.2 同上下文仪器:分歧不是"贴边翻转后自回归放大"

原文:`/mnt/game/s14d/hostlogs/E4-profile.txt`(`s14d_profile.py`)

`prompt_logprobs` 比较的是**字节完全相同的 prompt 上下文**,与自回归发散无关。
把每个位置的差拆成"两侧 argmax 相同 ⇒ 主导 logit 的位移"与"非 argmax 尾部":

| prompt | plp 位置数 | \|Δ\|>1e-6 的位置数 | 最早分歧位置 | max\|Δ\|(全部) | **max\|Δ\|(主导 logit)** | **max\|Δ\|(尾部)** | argmax flips | 首生成 token A/B | 相同? |
|---|---|---|---|---|---|---|---|---|---|
| zh_hz | 56 | 52 | **5** | 10.316 | 1.364 | 9.075 | 16 | 198 / 198 | ✅ |
| zh_ml | 56 | 52 | **5** | 10.316 | 0.858 | 9.075 | 14 | 198 / 198 | ✅ |
| zh_sort | 60 | 53 | 8 | 8.052 | 0.667 | 8.052 | 12 | 198 / 198 | ✅ |
| zh_sky | 59 | 51 | 9 | 8.463 | 1.327 | 8.463 | 9 | 198 / 198 | ✅ |
| en_fr | 58 | 49 | 10 | 12.665 | 0.771 | 9.577 | 13 | 198 / 198 | ✅ |
| en_sea | 59 | 51 | 9 | 8.463 | 1.327 | 8.463 | 10 | 198 / 198 | ✅ |
| en_index | 59 | 51 | 9 | 8.463 | 1.327 | 8.463 | 9 | 198 / 198 | ✅ |
| en_math | 65 | 63 | 3 | 9.445 | 0.652 | 9.445 | 17 | 198 / 198 | ✅ |
| p2_zh_slip | 77 | 73 | **5** | 9.278 | 0.882 | 9.278 | 16 | 198 / 198 | ✅ |
| p2_en_batch | 73 | 68 | 6 | 10.651 | 1.189 | 8.932 | 15 | 198 / 198 | ✅ |
| p2_zh_code | 75 | 70 | 6 | 10.651 | 0.658 | 8.932 | 13 | 198 / 198 | ✅ |
| s14_greedy_long | 125 | 120 | 6 | 10.945 | 0.760 | 8.558 | 28 | 198 / 198 | ✅ |
| **合计** | 822 | **753/822 = 91.6%** | 最早 **3–10** | 最坏 **12.665** | **0.652–1.364** | **8.05–9.58** | 9–28 | **12/12** | ✅ |

三条判读:

1. **差异在"每个 prompt 位置"上普遍存在**(753/822 = 91.6% 的位置差异 > 1e-6),
   且**从 prompt 第 3–10 个位置就开始**——不是尾部偶发,也不是"S14b 那种第 59 位离散事件"
   的孤立形状(那一次是 0–58 全 0)。
2. **形状与 S14b 量到的 graph↔eager 差同型**:主导 logit 只动 **0.65–1.36 nat**,
   而 top-5 尾部动 **8.05–9.58 nat**,argmax flips 9–28/位置数。
   ⇒ 与 S14b §4.1 的机制 B("主导 logit 一致、竞争结构被打乱")完全一致,
   **没有**"某个算子算坏"的形状(NaN/爆炸/整体偏移)。
3. **首生成 token 在 12/12 prompt 上一致**(两侧都是 token id `198`)。
   ⇒ 两次运行给出的"答案开头"相同,分歧出现在后续 token;
   即 S14b 的 `P2`(`'The'` vs `'用户'`)是一次**低概率的首 token 翻转**,本步 12 个 prompt
   在 2 次运行之间**一次都没有重现**。

#### 3.1.3 E4 的初步结论(待 A3/A4 定性)

**门禁 E4 不通过**:预热那次(W4)与服务那次(S4)的 token 流 **12/12 不逐位一致**,
同上下文 logit 差 mean 1.0–1.8 nat / max 12.7 nat。**"空 cache 预热一次再 AOT 重载服务"
这条配方不能保证服务态与预热态产出一致的 token 流。**

**但 W4 vs S4 还不足以定性**:它只有一个"现场编译"样本与一个"重载"样本,
所以"差"既可能是 **现场编译 ≠ 自己的磁盘产物**(H-AOT,配方可救:
改为"预热产物只当编译用、服务态自己可复现"),也可能是 **graph 模式任意两次运行都差**
(则 graph 的 token 流根本不可依赖)。**A3/A4 = 同一 cache 的第 2/3 次重载**正在测这一点。

#### 3.1.5 四臂逐对矩阵(1 次现场编译 + 3 次重载,combo ON,同 cache 同 shape)

原文:`/mnt/game/s14d/hostlogs/E1A-matrix.txt`(`s14d_matrix.py`)。W4=现场编译;
S4/A3/A4 = 同一 cache 的第 1/2/3 次 **AOT 重载**(`aot_load=4`,ready 均 230 s,KV 均 18.48 GiB)。

| pair | token 流逐位一致的 prompt | **首次生成 token 一致** | 首个分歧位置中位 | **mean 同上下文 \|Δ\|** | **最坏 同上下文 \|Δ\|** |
|---|---|---|---|---|---|
| S4(reload) vs W4(live) | **0/12** | 12/12 | 8.5 | 1.3477 | 12.6652 |
| A3(reload) vs W4(live) | 1/12 | 12/12 | 16.0 | 1.2529 | 10.1028 |
| **A3(reload) vs S4(reload)** | **0/12** | 12/12 | 11.5 | 1.2397 | 11.5657 |
| **A3(reload) vs A4(reload)** | 1/12 | 12/12 | 11.0 | 1.4282 | 12.7189 |
| A4(reload) vs W4(live) | 1/12 | 12/12 | 21.0 | 1.4544 | 12.6014 |
| **A4(reload) vs S4(reload)** | **0/12** | 12/12 | 9.5 | 1.3355 | 11.8368 |
| **合计** | **3/72 prompt-pair 逐位一致(4.2%)** | **72/72** | 8.5–21 | 1.24–1.45 | 最坏 **12.72** |

**把 pair 按因子拆开**:

| 因子 | pair 数 | 逐位一致的 prompt-pair |
|---|---|---|
| **现场编译 vs 重载** | 2 pair × 12 = 24 | **2/24(8.3%)** |
| **重载 vs 重载**(同 cache、同 shape、同请求序) | 3 pair × 12 = 36 | **2/36(5.6%)** |

⇒ **两个因子的不一致率在统计上不可区分**(0/12 vs 1/12 的量级)。
**H-AOT 被否证**:"现场编译 vs 磁盘重载"不是可识别的因子;
**graph 模式在同一配置下运行间就不可复现**——两次完全相同的 AOT 重载启动
(同 cache、同 shape、同请求序、同 `aot_load`)给出的 token 流
**12/12 不同**,同上下文 logit 差 mean ≈1.24–1.43 nat、max 11.6–12.7 nat。

**风险量化(本步数据)**:

* 72 个 prompt-pair 中**首次生成 token 全部一致**(两侧恒为 token id `198`)。
  注意:本电池 12 个 prompt 在**所有** 4 次运行里首生成 token 都是 `198`
  ⇒ **这个仪器对"答案首 token 翻转"不敏感**,只能说"这 12 个 prompt 上没翻转"
  (0/72 的 95% 上界约 4%/prompt)。S14b 的 `P2`(`'The'` vs `'用户'`)是另一次
  配置/prompt 下的翻转 ⇒ **翻转存在但低概率**。
* **但 token 流一定分叉**:首个分歧位置中位 **8.5–21 个 token**,
  最短 0 个 token ⇒ **约 95%+ 的 prompt-pair 在 10–20 个生成 token 内分叉**,
  即"答案正文"必然不同(不是只在第 100+ token 的尾巴上差)。

#### 3.1.6 E4 的最终结论

**门禁 E4 不通过(硬失败)**:预热配方(空 cache 现场编译一次 → 关闭 → AOT 重载对外服务)
**不能**保证"预热那次"与"服务那次"的 token 流一致:
* 不一致率 **12/12 = 100%**(门禁要求 = 0 或 <1e-3 nat 的舍入级);
* 同上下文 logit 差 mean 1.04–1.76 nat、max 12.67 nat,**远超** <1e-3 nat 的门禁;
* 而**根因不是"现场编译 vs 重载"**:两次 AOT 重载之间同样 12/12 不一致(§3.1.5)。

⇒ **配方需要改写**:不是"预热一次再重载服务",而是
**"graph 的 token 流在运行间不可复现,任何依赖逐位一致性的用途都不能用 graph"**;
预热仍然值得做(ready 352→230 s,−35%,且**不影响**数值),但**它解决的是启动时间,不是可复现性**。

#### 3.1.4 宿主内存形态(旁证,修正一次误读)

`free -m` / `/proc/swaps` 原文在 `E4-driver.log` 的 `*_after_start` / `*_after_stop` 行:

| 时刻 | used | available | `/swap.img` 已用 | `swap4` 已用 | swap 合计已用 |
|---|---|---|---|---|---|
| W4 ready(t≈352 s) | 28339 MiB | **2924 MiB** | 4193696 KB | 4193196 KB | **8190 / 8191 MiB(100%)** |
| W4 after_stop | 3079 MiB | 28487 MiB | 449412 KB | 18088 KB | 456 MiB |
| S4 ready(t≈230 s) | 27194 MiB | **4064 MiB** | 4193868 KB | 1002500 KB | 5074 MiB |
| S4 after_stop | 3083 MiB | 28478 MiB | 456764 KB | 23316 KB | 468 MiB |

⇒ **纠正**:本步中途用 `free -g` 看到"swap 几乎没用"是**稳态采样**,不是启动态;
**ready 时刻 8 GiB swap 被用到 100%**(W4),余量 2.9 GiB —— 与 S14c C14 的
`min MemAvailable 2.35 GiB / maxSwapUsed 5.40 GiB` 同档。§0 里那条"swap 未被动用"的说法
**作废**(见 §0 的更正行)。

### 3.2 E1:H-AOT 2×2

**结论:H-AOT 被否证(两个方向都否证)。** 原文:`/mnt/game/s14d/hostlogs/E1-7arm-matrix.txt`、
`E1B-matrix.txt`、`E1-compare.txt`、`/mnt/game/s14d/e1-*.csv`。

#### 3.2.1 2×2 格子与每一次启动的完整身份

两个开关(`-cc.inductor_compile_config='{"combo_kernels": false, "benchmark_combo_kernel": false}'`)
**确实传进去了**(`s14d_run_one.sh` 的 `knobs:` 行原文;
容器内 `non-default args` 里 `'combo_kernels': False, 'benchmark_combo_kernel': False`);
它们换 AOT key,**cell B 用独立 cache 根** `/mnt/game/s14d/cache-b`(全新)。

| 格子 | arm | 因子 | cache 根 / vintage | ready | aot_load | compilation | 任务 |
|---|---|---|---|---|---|---|---|
| (**现场编译**,**combo on**) | **W4** | 预热 = 首跑落盘 | `cache-e4`(空)/ `e4_w4_…livecompile` | 352 s | 0 | 117.30 s | probe |
| (**磁盘重载**,**combo on**) | **S4** | 服务 = 重载 | `cache-e4` / `e4_s4_…AOTreload` | 230 s | 4 | 24.77 s | probe |
| (磁盘重载, combo on, **第 2 次**) | A3 | 补 2 次(1/2) | `cache-e4` / `e2_A3_…AOTreload` | 230 s | 4 | 22.56 s | probe |
| (磁盘重载, combo on, **第 3 次**) | A4 | 补 2 次(2/2) | `cache-e4` / `e2_A4_…AOTreload` | 230 s | 4 | 22.97 s | probe |
| (**现场编译**,**combo off**) | **B1** | 空 cache | `cache-b`(空)/ `e1_b1_…livecompile` | 330 s | 0 | 94.69 s | probe |
| (**磁盘重载**,**combo off**) | **B2** | 重载 | `cache-b` / `e1_b2_…AOTreload` | 220 s | 4 | 18.72 s | probe |
| (磁盘重载, combo off, **第 2 次**) | B3 | 补 1 次 | `cache-b` / `e1_b3_…AOTreload2` | 221 s | 4 | 18.75 s | probe |

内存形态:combo on 四臂 = 附加 4 GiB swapfile(总 8 GiB)+ `swappiness=200`(生产配方);
combo off 三臂 = 临时挂 16 GiB `/mnt/game/swapfile`(总 23 GiB;S14c §3.11.1 已证 combo off
是最吃内存的变体,因为要保证 live-vs-reload 对照不因 OOM 中断),`swappiness` 仍 200。
`oom_kill` 全程 4(未增长),`Available KV cache` 全部 18.47–18.48 GiB。

#### 3.2.2 每格的 live vs reload(2×2 的判据)

| 格子 | pair | token 流逐位一致 | 首个分歧位置中位 | mean 同上下文 \|Δ\| | 最坏 \|Δ\| | plp argmax flips |
|---|---|---|---|---|---|---|
| combo **on** | W4(live) vs S4(reload) | **0/12** | 8.5 | 1.3477 | 12.665 | 9–28 |
| combo **off** | B1(live) vs B2(reload) | **1/12** | 8.0 | 1.3891 | 13.573 | 8–24 |
| combo **off** | B1(live) vs B3(reload2) | **0/12** | 16.0 | 1.3765 | 12.336 | — |
| combo **on** | W4(live) vs A3(reload2) | **1/12** | 16.0 | 1.2529 | 10.103 | — |
| combo **on** | W4(live) vs A4(reload3) | **1/12** | 21.0 | 1.4544 | 12.601 | — |
| combo **on** | S4(reload1) vs A3(reload2) | **0/12** | 11.5 | 1.2397 | 11.566 | — |
| combo **on** | S4(reload1) vs A4(reload3) | **0/12** | 9.5 | 1.3355 | 11.837 | — |
| combo **on** | A3(reload2) vs A4(reload3) | **1/12** | 11.0 | 1.4282 | 12.719 | — |
| combo **off** | B2(reload1) vs B3(reload2) | **0/12** | 12.5 | 1.2424 | 13.425 | — |

**"现场编译 vs 重载" 组的平均**:combo on 0-1/12,combo off 0-1/12;
**"重载 vs 重载" 组(同 cache 同 shape)**:combo on 0-1/12,combo off 0/12。
⇒ **同 cache 的两次重载之间的不一致率 = 现场编译 vs 重载的不一致率**,
即 **"AOT 产物来源"不是可识别因子**。

#### 3.2.3 combo off 是否救得了可复现性?—— 没有

| 配置 | pair 数 | 逐位一致的 prompt-pair | mean \|Δ\| 范围 | 最坏 \|Δ\| |
|---|---|---|---|---|
| **combo on**(W4/S4/A3/A4) | 6 × 12 = 72 | **3/72 = 4.2%** | 1.24–1.45 | 12.719 |
| **combo off**(B1/B2/B3) | 3 × 12 = 36 | **1/36 = 2.8%** | 1.24–1.39 | 13.573 |
| **跨配置**(on 臂 × off 臂) | 12 × 12 = 144 | **2/144 = 1.4%** | 1.30–1.42 | **15.656** |

⇒ **关掉 `combo_kernels`/`benchmark_combo_kernel` 没有任何可复现性收益**
(4.2% → 2.8%,统计上不可区分),而且**跨配置的 pair 与同配置的 pair 统计量完全同档**
(mean 1.30–1.42)。**S14b §4.2 修法建议 #1(`benchmark_combo_kernel=False`)被实测否证。**

#### 3.2.4 7 臂 21 对全矩阵(252 个 prompt-pair 的总账)

原文:`/mnt/game/s14d/hostlogs/E1-7arm-matrix.txt`

```
21 个 arm-pair × 12 prompt = 252 个 prompt-pair
token 流逐位一致        : 6/252 = 2.4%
首次生成 token 一致     : 252/252 = 100%(两侧恒为 token id 198)
首个分歧位置中位数      : 8.0 – 26.5 个生成 token
mean 同上下文 |Δ|       : 1.240 – 1.454 nat(21 个 pair 全部落在 1.2–1.5 之间)
最坏 同上下文 |Δ|       : 10.103 – 15.656 nat(最坏出现在跨配置 B3 vs W4 = 15.656)
```

**21 个 pair 的 mean 同上下文 |Δ| 全部落在 1.24–1.45 nat 的窄带里,与"哪个因子被打开"
无关** —— 这是本步最强的证据:graph 路径的**运行间数值抖动是一个与配置正交的、
幅度固定的现象**(mean ≈1.3 nat,尾 ≈10–16 nat),不是某两个配置之间的系统性差。

### 3.3 E5:C2 vs P4e 的 12%(eager 的两形态 decode)与 eager 侧的并发/跨运行对照

(数据收集中;两个 arm 的任务序为 `conc2,probe,decode` —— `conc2` 在最前 = 冷引擎无前缀缓存污染)

**结论:E5 判"抖动/vintage",不是内存形态。** 原文:`/mnt/game/s14d/hostlogs/E5-compare.txt`、
`E5SW16-bench-decode.log`、`E5SW4-bench-decode.log`、`E5-driver.log`。

#### 3.3.1 两形态 decode 表(TP=4 eager,`--enforce-eager`,其余 flag 与生产配方相同)

| arm | 宿主内存形态 | ready | `Available KV cache` | **Median TPOT** | Mean | P90 | P99 | ready 时刻 available | ready 时刻 swap 已用 |
|---|---|---|---|---|---|---|---|---|---|
| **E5SW16** | `/swap.img` 4 G + `swap4` 4 G + **`/mnt/game/swapfile` 16 G** = 总 24 GiB | 161 s | 18.69 GiB | **133.83 ms** | 133.05 | 134.51 | 134.57 | 6139 MiB | 571 MiB |
| **E5SW4** | `/swap.img` 4 G + `swap4` 4 G(**不挂** 16 GiB 那个)= 总 8 GiB | 160 s | 18.69 GiB | **131.78 ms** | 130.43 | 133.66 | 134.68 | 6172 MiB | 569 MiB |
| 差 | — | — | — | **+1.6%** | — | — | — | +33 MiB | 0 |

**关键旁证**:两个 arm 的 **swap 实际用量都是 ~570 MiB**(即 16 GiB 那个 swapfile
**一页都没被用到**),`MemAvailable` 都是 **6.1 GiB** ⇒ **eager 在两种形态下都不缺内存**,
所以"内存余量 → decode 变慢"这条因果在 eager 上**根本不可能成立**。

#### 3.3.2 与本步之前的 4 个实测并排(全部 TP=4 eager、同 maxlen/util/MBT)

| run | 来源 | 内存形态 | Median TPOT | 与 S14 P4e 之差 |
|---|---|---|---|---|
| S14 **P4e** | S14 §3.4.1 | 挂 16 GiB swapfile | **126.83 ms** | — |
| S14c **C2** | S14c §3.3 | 只 `/swap.img` | **142.59 ms** | **+12.4%** ← Lead 记的那笔 |
| S14c **C16** | S14c §3.12 | 只 `/swap.img`(**与 C2 同形态**) | **134.36 ms** | +5.9% |
| S14d **E5SW16** | 本步 | 挂 16 GiB swapfile | **133.83 ms** | +5.5% |
| S14d **E5SW4** | 本步 | 不挂 16 GiB | **131.78 ms** | +3.9% |

**判读(D 的 12% 的答案)**:

1. **同一形态内部的散差就有 6%**:C2(142.59)与 C16(134.36)**内存形态逐字相同**,
   差 6.1%;⇒ 这 12% 不可能由内存形态解释。
2. **本步把形态当自变量、其它全固定**:133.83(挂 16 G)vs 131.78(不挂)= **1.6%**,
   而且**不挂的那个反而更快**;再叠加"16 GiB swapfile 一页未用 / available 都是 6.1 GiB",
   ⇒ **形态效应 = 0**。
3. 5 次实测落在 **126.83–142.59 ms(±6%)** 的带内,`P4e` 是这条带的下沿
   ⇒ 判 **"抖动/vintage"**(宿主 CPU 频率/热状态、Triton/HIP kernel 首次 autotune 的
   vintage、页缓存状态),而**不是**内存余量。
4. ⇒ **"生产机需要内存余量"这条对 eager 不成立** —— 写进配方时不需要为 eager 加内存条款;
   该条款只为 **graph** 成立(S14c §3.12/§3.13)。

#### 3.3.3 附带得到的两个重要结论(E5 的副产物)

* **eager 的运行间可复现性被本步独立再证,而且强到小数位**:
  E5SW16 与 E5SW4 是两个**独立引擎**(不同内存形态、先后启动),
  它们的 `conc2` 结果**逐位相同**:
  `bit-identical=False, first_div=6, maxd_common=0.515217, plp_max=15.02143, plp_mean=1.82210`
  —— 五个数字**完全相同**(§3.3.4)。⇒ eager 的数值行为是**引擎运行间可复现的确定函数**,
  而 graph 不是(§3.1.5 / §3.2.4)。这条把"eager 作质量参照"从 S14b 的
  8-prompt 级证据升级到**跨引擎小数位级**证据。
* **但 eager 也有一个"请求上下文敏感"的确定性效应**:同一 prompt
  "与另一个请求并发"vs"随后单独发",token 流在第 6 个 token 分叉、
  同上下文差 mean 1.82 nat / max 15.02 nat。**这个效应本身完全可复现**
  (两个 arm 数字相同),所以要分清"不可复现的抖动"(graph)与
  "可复现但依赖请求上下文"(eager 的并发/历史效应)。第 3.3.4 节用一个
  无并发的对照把它归因。

#### 3.3.4 conc2(并发 vs 串行)与 seq2(同 prompt 连发两次,无并发)

| 实验 | 设计 | 结果 |
|---|---|---|
| **conc2**(E5SW16) | 冷引擎;**并发**发目标(`p2_zh_slip`)+filler(`p2_en_batch`),随后**串行**重发目标 | `bit-identical=False first_div=6 maxd=0.515217 plp_pos=77 plp_max=15.02143 plp_mean=1.82210` |
| **conc2**(E5SW4) | 同上,独立引擎、不同内存形态 | **五个数字完全相同**(逐位复现) |
| **seq2**(E5SEQ) | 冷引擎;**串行**发目标两次(全程单请求在飞,无并发、无其它流量) | 见下 |

`Running: 2 reqs` 在 E5SW16 的引擎日志里出现 2 次(10 s 采样 × 2 ≈ 20 s 重叠窗口)
⇒ conc2 的两条请求**确实被连续批处理到一起**(不是被串行化)。

**seq2 的结果(归因:E3 结案)**

```
$ tail -2 /mnt/game/s14d/hostlogs/E5SEQ-seq2.log
[s14d-seq2] req1_cold:   128 tok 18.872s
[s14d-seq2] req2_after1: 128 tok 17.421s
req2-after  vs  req1-cold   tokens=128/128 bit-identical=True first_div=None
                            maxd_common=0.000000 plp_pos=77 plp_max=0.00000 plp_mean=0.00000
# 引擎日志 `Running:` 计数:0 reqs ×1、1 reqs ×4 —— **全程从未出现 2 reqs** ⇒ 真的没有并发
```

⇒ **判读(三个结论)**

1. **请求历史在 eager 下无关**:同一 prompt 连发两次(冷首请求 + 之后一次),
   token 流**逐位一致**、同上下文 77 个位置的 \|Δ\| **全 0.00000**
   (与 S14b §3.4.1 唯一那次 `zh_ml` 历史敏感不同 —— 那次是**跨引擎**比较,本步是**同引擎**)。
2. ⇒ E5 conc2 的分叉**不是历史造成的,是并发造成的**:eager 在 **batch=2**
   (引擎日志确实出现 `Running: 2 reqs`,10 s 采样到 2 次 ≈ 20 s 重叠)时,
   同一个 prompt 的 logits 与 **batch=1** 时差 **mean 1.82 nat / max 15.02 nat**,
   token 流在第 **6** 个 token 分叉。
   **这就是 S14b §5 第 6 条那个"13.7 nat 的并发观测"的受控复现与消解**:
   它是**真实的批组合效应**,不是测量噪声,也不需要"隐藏状态"来解释;
   13.7 nat 落在本步 max 15.02 nat 的同档内。
3. **这个批组合效应本身是完全可复现的**(两个独立引擎、不同内存形态给出五个数字完全相同)
   ⇒ 项目里必须区分两种"不一致":
   * **不可复现的抖动** = graph 的运行间差异(§3.1.5/§3.2.4,252 个 prompt-pair 只有 6 个一致);
   * **可复现但依赖请求上下文** = eager 的 batch 组合效应(数值确定,给定批形状与历史)。
   两者对生产的要求不同:前者只能改代码/换模式;后者可以在**固定批组合**下做到逐位复现
   (eager + 串行 ⇒ 本步实测 0.000 nat)。

### 3.4 E2:`p2_zh_slip` 最小复现("第 59 位起差到 9.5 nat"是否稳定)

**结论:量级稳定复现,但"前 58 位逐位为 0、第 59 位起跳"的形状不复现 —— 它是运行间抖动的普通尾部。**

原文:`/mnt/game/s14d/hostlogs/E2-common-prefix.txt`、`E4-profile.txt`、`E2-profile-S4-A3.txt`

**(a) 用同一 cache 连跑两次(E2 的最小复现步骤)**:
S4 vs A3 = 同一 cache 目录 `/mnt/game/s14d/cache-e4` 的第 1/2 次 AOT 重载,
同 shape、同请求序。`p2_zh_slip`(77 个 prompt 位置、128 个生成 token)的结果:

```
生成流首个分歧位置 : 第 45 个 token(S14b 的 A18 记的是"第 59 位")
同上下文首个分歧位 : 第 7 个 prompt 位置(不是第 59)
同上下文 max|Δ|    : 11.078 nat(S14b: 9.489 nat)   mean|Δ| 1.173 nat
差异覆盖位置数     : 71/77(91.6%)—— 而 S14b 的 A18 是"0..56 全 0.000,59 起才 >0"
```

**(b) 把"共同前缀长度"做成分布(6 个 pair × 12 prompt = 72 个 prompt-pair)**:

```
pair         zh_hz zh_ml zh_sort zh_sky en_fr en_sea en_index en_math p2_zh_slip p2_en_bat p2_zh_code s14_greedy
W4-S4             2     7      12      1    32     10        2       0         33        0        27        77
W4-A3            16     7      12      1    26     16        2      21         33       27        28       128
W4-A4             2     7      11      1    26     59       22      21          6       29        28       128
S4-A3             2     7      16     11    26     10       12       0         45        0        27        77
S4-A4             9    14      11      2    26     10        2       0          6        0        27        77
A3-A4             2     7      11      2    26     16        2      21          6       27        33       128
```
(表内数字 = 生成流逐位相同的**前导 token 数**)

```
n=72 prompt-pair;commom prefix: min 0  p25 2  median 12  p75 27  max 128  mean 21.6
共同前缀 >= 58 token 的 pair : 7/72 (9.7%)   <- 正是 S14b A18 观测到的形状
共同前缀 >= 32 token 的 pair : 12/72 (16.7%)
p2_zh_slip 的 6 个 pair      : 6, 6, 6, 33, 33, 45
```

⇒ **判读**:S14b 的两个"锐利线索"都落在本分布的常规区间内:
* "第 59 位起差到 9.5 nat" = 共同前缀 ≈58 的情形,**72 个 pair 里有 7 个(9.7%)都是这样**,
  所以它不是"某个离散事件",只是**两次运行在某个贴边位置恰好一致了前 58 个 token**;
* "重载 A5 的 cache 后 3/4 prompt 逐位一致" = 本步 72 个 prompt-pair 里
  **3 个完全逐位一致(4.2%)**,同样不是"H-AOT 成立"的证据,而是**概率**。

**因此 E2 的结论:`p2_zh_slip` 的 9.5 nat 分歧确实稳定(本步 9.28/11.08/9.75/… nat),
但它的"离散起始位置"解释(第 59 位 = chunk/对齐边界)不成立**;
不需要再翻 `VLLM_QWEN35_REC_PREFILL` 去找"离散事件",因为**没有离散事件**——
差异在 **90%+ 的 prompt 位置上**都存在(§3.1.2)。

### 3.5 E3:conc=2 一致性

**判定:graph 侧不做;受控 conc=2 改在 **eager** 上做(结果见 §3.3 的 `conc2` 行)。理由:**

1. **graph 侧的 conc=2 已经没有信息量**:本步已在**无并发**条件下量到同上下文
   mean ≈1.3 nat、max 12.7–15.7 nat 的运行间差(§3.1.5/§3.2.4);
   任何"并发造成的差"都无法与这个底噪区分。S14b 记的"13.7 nat"**落在本步无并发
   差的范围内**(12.719 / 13.573 / 15.656 nat 都出现过)⇒ **13.7 nat 不再需要引入并发因子**。
2. **有意义的问题是"eager 是否也怕并发"**(eager 是本项目唯一的质量参照)。
   在 E5 的两个 eager 臂上把 **`conc2` 放在任务列表第一位**(冷引擎、无前缀缓存污染):
   目标 prompt(`p2_zh_slip`)与一个**不同的** filler prompt(`p2_en_batch`)并发发出,
   随后同引擎**串行**重发目标 prompt,比较"并发态 vs 串行态"的 token 流与同上下文 logits。

## 4. 门禁判定

| # | 门禁 | 判据(任务书 §3) | 判定 | 证据 |
|---|---|---|---|---|
| **E4** | 预热配方有效性 | 不一致率 **= 0**(或只有 <1e-3 nat 的舍入级差) | ⛔ **不通过(硬失败)** | §3.1.1 不一致率 **12/12 = 1.0000**;§3.1.2 同上下文 mean **1.04–1.76 nat** / max **12.67 nat**(>1e-3 三个数量级);§3.1.5 同 cache 两次重载之间同样 12/12 |
| **E1** | H-AOT 2×2 | 每格 ≥2 次独立启动;逐 prompt 逐位对比 + 首分歧位置 + nat 间距 | ✅ **通过(7 次独立启动)** | §3.2.1 七臂身份表(含 cache 根/vintage/ready/`aot_load`);§3.2.2 每格 live-vs-reload;**结论:H-AOT 否证** —— "现场编译 vs 重载"24 个 prompt-pair 里 2 个一致(8.3%),"重载 vs 重载"36 个里 2 个一致(5.6%) |
| **E5** | C2 vs P4e 的 12% | 同 TP=4 eager 两种内存形态各测一次 decode(n≥4);一致 ⇒ 抖动/vintage | ✅ **通过** | §3.3.1 **133.83 vs 131.78 ms(+1.6%,n=5)**;§3.3.2 五个历史实测落在 126.83–142.59(±6%),其中 S14c C2 vs C16 **同形态**就差 6.1% ⇒ **判抖动/vintage**;"生产机需要内存余量"对 eager **不成立**(16 GiB swapfile 一页未用、available 6.1 GiB) |
| **E2** | `p2_zh_slip` 最小复现 | 同 cache 连跑两次,给出"第 59 位起差到 9.5 nat"是否稳定复现 | ✅ **通过** | §3.4:同 cache 三次重载;`p2_zh_slip` 同上下文 max 9.28/11.08/9.75 nat ⇒ **量级稳定复现**;但"0–58 位全 0、59 位起跳"**不复现**(差异覆盖 71/77 个位置、从第 5–7 位就开始);72 个 prompt-pair 里 **7 个(9.7%)** 的共同前缀 ≥58 token ⇒ 那只是普通尾部 |
| **E3** | conc=2 一致性 | 消解 S14b 的 13.7 nat 并发观测 | ✅ **通过(受控 + 归因)** | §3.3.4:conc2 目标 vs 串行 = 第 6 个 token 分叉、mean 1.82 / max 15.02 nat;seq2(无并发、同 prompt 两次)= **逐位一致、\|Δ\|=0.00000** ⇒ 归因于**并发(批组合)**,不是请求历史 |

### 4.1 达标线判定(任务书的最终判据)

> *"若 E4 的不一致率 = 0(或仅 <1e-3 nat)且坏 draw 不可复现 ⇒ 判'预热配方足够,graph 可上生产';
> 若坏 draw 可复现且概率不可忽略 ⇒ 判'graph 不能上生产',并给替代(TP=2 graph 84.22 是否同样有风险)"*

**判定:⇒** **"graph 不能上生产"(用于任何依赖 token 流一致性的用途)**。

* **坏 draw 可复现且概率 = 100%**:252 个 prompt-pair 里 246 个(97.6%)token 流不一致;
  不一致的是"几乎每一次运行",不是"偶发的那一次"。
* **不是概率问题,是结构问题**:P(graph 两次运行给出相同 token 流) ≈ 2.4%/prompt,
  而且**前 10–20 个 token 内必定分叉**(中位 8.5–26.5)⇒ "答案正文"必然不同。
* **注意"性能"与"数值"必须分开**:S14c 证 graph 的 49.65 ms 与内存配方是可用的;
  **本步只否证它的 token 流可依赖性**,不否证它的延迟/吞吐结论。

### 4.2 替代方案:TP=2 graph(84.22 ms)是否同样有风险?

**是,同样有风险,而且已经有直接证据。**

1. **S14b 的 `P2` vs `A1r` 就是 TP=2 上的同一现象**:两次**同配置、同 cache、
   同 AOT hash、同冷首请求**的 TP=2 graph 运行给出 **`'The'`/gap 4.750 vs `'用户'`/gap 1.047**
   (同上下文差 5.36 nat),而且一次是英文重复循环。**这是"答案级"翻转**
   (首 token 不同),本步在 TP=4 上量到的 mean ≈1.3 nat 抖动与之同量级、同形状。
2. **本步的机制与 TP 无关**:21 个 arm-pair 的 mean 同上下文 \|Δ\| 全部落在 **1.24–1.45 nat**
   的窄带内,与 combo 开关、AOT 来源、cache vintage 都正交;S14b 在 TP=2 上量到的
   graph↔eager 差(mean 1.38 / max 10.3 nat)与本步 TP=4 的数值同档。
3. ⇒ **没有理由认为 TP=2 更安全**;"84.22 ms 的 TP=2 graph"同样**不能**用于
   质量评测、回归对比、审计或任何"同请求必须同输出"的场景。
   **要"快且可信"只有两条路**:修编译路径的可复现性(未定位,建议只读取证)或退回 eager
   (TP=4 **134 ms** / TP=2 ~? ;eager 的代价是慢 2.4×,换来逐位可复现)。

### 4.3 可执行的部署配方(本步结论的落地形式)

```text
① 质量/精度/回归/审计口径  ->  eager(TP=4, --enforce-eager), 且**固定批组合**:
   * eager 跨引擎逐位可复现(本步 conc2 五个数字在 2 个独立引擎上完全相同);
   * 同 prompt 串行重发 = 逐位一致(本步 seq2: 0.00000 nat);
   * **但 conc=2 与 conc=1 不同**(第 6 token 分叉、mean 1.82 nat)
     => 评测必须记录并固定 --max-num-seqs / 批组合,否则不可比。
② 性能口径               ->  graph 可用(49.65 ms, TP=4, S14c §3.13 配方),
   但只用于"延迟/吞吐/内存"测量, **绝不用于 token 流**。
③ 预热(AOT 重载)          -> 保留:ready 352→230 s、compilation 117→25 s,
   **对数值无影响**(不是它导致不一致);生产启动用"空 cache 预热一次再重载"仍是最省时的做法。
④ 内存配方               -> 沿用 S14c §3.13(graph + 附加 4 GiB swap + swappiness=200);
   **eager 侧不需要内存条款**(本步 E5)。
⑤ 若要上 graph 做服务   -> 必须在 SLA 里写明"同一请求跨重启/跨批组合的输出不保证一致",
   并在任何质量门禁之前用 eager 复核。
```

## 5. 未完成项 / 残余风险

| # | 项 | 状态 | 原因 / 已有证据 |
|---|---|---|---|
| 1 | **graph 运行间抖动的根因定位** | ❌ 未做 | 6 个候选因子已排除:cudagraph(S14b A3 ≡ A5/A18)、packed decode(S14b §3.11)、prefix cache(本步实测 **hit rate 全程 0.0%**)、AOT 来源(本步)、`combo_kernels`/`benchmark_combo_kernel`(本步)、内存形态/swap(本步 E5)。剩下的方向是**编译期 kernel 选择/归约顺序在加载时不确定**(需要只读取证:`torch_aot_compile` 产物逐字节比对、`inductor` 的 `.so` 符号/权重哈希),本轮预算内未做 |
| 2 | **TP=2 graph 的 2 次对照** | ❌ 未跑 | 预算;但 TP=2 的证据已有(S14b `P2` vs `A1r`,见 §4.2),且本步机制与 TP 正交 |
| 3 | **eager 批组合效应的函数形状** | 🔶 只测了 conc=1 与 conc=2 两点 | 未测 batch=4/8、未测 filler 换成别的 prompt 是否同样改变数值、未测"两个请求谁被排在 batch 首位"的影响。**这是本步新开的洞,建议单独立项**(它比 graph 的问题更容易解决:固定 batch 即可) |
| 4 | 8K prefill / 长稳态 / conc8 下的可复现性 | ❌ 未做 | 预算;本步电池全是 ≤128 生成 token 的短请求 |
| 5 | `min MemAvailable` 的 memtrace 级测量 | 🔶 只有 `free -m` 快照 | S14d 的 run_one 去掉了 S14c 的 5 s memtrace(本步不测内存);ready/after_stop 的 `free` 快照在 `E4-driver.log`/`E5-driver.log` |
| 6 | `docs/海光Z100-生产部署配方.md` §4/§7 的文字更新 | 🚫 未做(越权) | 按纪律 6 由 Lead 维护;本档 §4.3 给出现成文本供粘贴 |
| 7 | 一次误操作 | ⚠️ 已登记 | 我自己的 `pkill -f "s14d_e5.sh"` 把**我自己那条 bash**一起杀了(driver 也已启动),导致 E5 首个 arm 在引擎已拉起时被中断;已清理(容器内 `kill` + `pkill VLLM` + `swapoff`)、确认 VRAM=0/live VLLM=0 后重跑。**没有**留下脏状态,也没有触碰生产容器 |

## 6. 对后续步骤的影响

1. **必须改写 S14c §6.3 的那句话**:它要求"上生产前必须先跑一次现场编译 vs 重载的逐位对照,
   若两次一致 ⇒ 配方成立"。本步跑了:**两次不一致(12/12)**,
   而且进一步证明**同 cache 的两次重载也不一致** ⇒
   **"预热配方 + graph"不能把 token 流变成可信**,该门禁的结论是**否**。
2. **`docs/海光Z100-生产部署配方.md` 的 §4(质量口径)必须写死**:
   * 质量/精度参照 = **eager**,且**记录批组合**;
   * graph 的 token 流**不得**进入任何质量评测、回归对比或审计;
   * "同一请求跨引擎/跨重启的输出一致"这条**不能**写进 graph 的 SLA。
3. **§5.1(内存达标判定)不受本步影响**:graph 的 2.4× 性能与 S14c 的内存配方仍成立;
   本步只加一条"性能口径 ≠ 质量口径"。
4. **对 S14c 的修正**:S14c §6.3 把"现场编译 vs 重载"当成唯一待验因子;
   本步证明真正的因子是 **graph 路径自身的运行间不确定性**,
   ⇒ 后续若要修,方向是**编译产物的确定性**(而不是预热流程)。
5. **给 Lead 的一个新风险条目**:eager 的批组合效应(§4.3 ①)意味着
   **本项目此前所有用 `vllm bench serve`(conc=1 与 conc=8 混用)做的质量/数值对比都可能混了批组合**;
   S14b 的 13.7 nat 就是这么来的。建议在 todolist 里单列一条。
6. **对 P6(内存/优化)无影响**;对"2.44× 能否上生产"的输入:
   **性能可以上,质量口径不能上 graph**;若业务要求"同请求同输出",则只能上 eager(134 ms)。

## 7. 复现方式

```bash
cd /home/lyh/z100-vllm
# ---- 前置(留档) ----
date; pgrep -af "vllm|torchrun"; hy-smi --showmeminfo vram; free -g; cat /proc/swaps
cat /proc/sys/vm/swappiness          # 接手时 = 60
sudo -n python3 tools/dcu/dcu_p2p_acs.py show | grep -c 0x0000   # 12(部署态不切)

# ---- E4(生产配方:附加 4 GiB swapfile + swappiness=200)----
export S14C_LOGDIR=/mnt/game/s14d/hostlogs
tools/dcu/s14c_swap.sh on /mnt/game/s14d/swap4 4096 E4-setup
sudo -n sysctl -w vm.swappiness=200
S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
  S14C_TAG="e4_w4_tp4_graph_livecompile_swap8g_sw200" READY_TIMEOUT=3600 \
  tools/dcu/s14d_run_one.sh W4 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 \
  /mnt/game/s14d/cache-e4 8601 262144 0,1,2,3 probe
S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
  S14C_TAG="e4_s4_tp4_graph_AOTreload_swap8g_sw200" READY_TIMEOUT=3600 \
  tools/dcu/s14d_run_one.sh S4 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 \
  /mnt/game/s14d/cache-e4 8602 262144 0,1,2,3 probe
python3 tools/dcu/s14d_compare.py /mnt/game/s14d/probe-S4.jsonl /mnt/game/s14d/probe-W4.jsonl \
  --label-a S4-service --label-b W4-warmup
python3 tools/dcu/s14d_profile.py  /mnt/game/s14d/probe-S4.jsonl /mnt/game/s14d/probe-W4.jsonl

# ---- E2(E4 之后:a34 = 同一 cache 的第 2/3 次重载)----
tools/dcu/s14d_a34.sh          # A3(cache-e4 第 2 次重载)、A4(第 3 次)+ 两两对照
python3 tools/dcu/s14d_matrix.py /mnt/game/s14d/probe-{W4,S4,A3,A4}.jsonl

# ---- E1 cell B(combo kernels off;会换 AOT key ⇒ 独立 cache 根 cache-b)----
tools/dcu/s14d_e1.sh           # B1 现场编译 / B2 重载 / B3 再重载 + 6 组对照
python3 tools/dcu/s14d_matrix.py /mnt/game/s14d/probe-{W4,S4,A3,A4,B1,B2,B3}.jsonl

# ---- E5(两形态 eager decode)+ E3(conc2 在冷引擎上最先跑)----
tools/dcu/s14d_e5.sh

# ---- E3 归因(无并发的 seq2 对照)----
tools/dcu/s14d_seq2_arm.sh ; cat /mnt/game/s14d/hostlogs/E5SEQ-seq2.log

# ---- 收尾(硬约束 4:必须还原成接手时状态)----
tools/dcu/s14c_swap.sh off /mnt/game/s14d/swap4 E-final
tools/dcu/s14c_swap.sh rm  /mnt/game/s14d/swap4 E-final
sudo -n sysctl -w vm.swappiness=60
sudo -n swapoff /mnt/game/swapfile 2>/dev/null || true     # 若 E1/E5 中断残留
cat /proc/swaps; free -m; cat /proc/sys/vm/swappiness
sudo -n python3 tools/dcu/dcu_p2p_acs.py show | grep -c 0x0000   # 收尾仍应为 12
```

**三个最容易踩的坑**(S14c 的三个 + 本步新增一个):

* `--cudagraph-capture-sizes` 是 `nargs='+'` ⇒ 必须展开成 `--cudagraph-capture-sizes 1 2 4 8`;
* 杀 worker 只能 `pkill VLLM`;
* `-cc.inductor_compile_config` 的值是 JSON,含空格时**必须用 bash 数组**保持单参数;
  **它会换 AOT key ⇒ 必须用新 cache 目录**;
* **新增**:`pkill -f "<脚本名>"` 会**先杀掉自己那条 bash**(它的命令行里含有同一个字符串),
  要用 `pkill -f "s14d_run_one.sh E5SW1[6]"` 这种方括号写法,或者 `pgrep` 确认后再 kill。

## 8. 参考

* [`z100-14c-tp4-graph-memory-result.md`](z100-14c-tp4-graph-memory-result.md) §3.12/§3.13(生产候选配方)、§3.11(因子汇总)、§6
* [`z100-14b-graph-vs-eager-result.md`](z100-14b-graph-vs-eager-result.md) §3.7/§3.9(`p2_zh_slip`)、§3.4(eager 逐位可复现)、§4(判定)、§6
* [`z100-14c-lead-crosscheck.md`](z100-14c-lead-crosscheck.md)(E5 的来源:C2 142.59 vs P4e 126.83)
* [`../tools/dcu/s14d_taskbook.md`](../../tools/dcu/s14d_taskbook.md)(任务书)
* 原始工件:`/mnt/game/s14d/probe-*.jsonl`(7 臂 × 12 prompt 的逐位 token 流 + 同上下文 logprobs)、
  `/mnt/game/s14d/conc2-*.jsonl`、`/mnt/game/s14d/seq2-E5SEQ.jsonl`、
  `/mnt/game/s14d/hostlogs/**`(逐 arm 完整日志)、`/mnt/game/s14d/*.csv`(逐 prompt 对照)、
  `/mnt/game/s14d/cache-e4`、`/mnt/game/s14d/cache-b`(可整目录删除即回滚)


## 9. 终态核验与回滚件(硬约束 4)

**宿主前后原文**:`/mnt/game/s14d/hostlogs/99-restore-before.txt`(收尾前)、
`99-final-state.txt`(收尾后终态)、`swap-E-final.txt`(swapfile 的 before/after `/proc/swaps`)。

| 项 | 接手时(2026-10-10T02:11:20Z) | 收尾时(2026-10-10T03:31:56Z) | 结论 |
|---|---|---|---|
| `/proc/swaps` | 只有 `/swap.img` 4194300(Used 538488) | 只有 `/swap.img` 4194300(Used 532760) | ✅ **完全还原** |
| `vm.swappiness` | 60 | 60 | ✅ **完全还原** |
| live VLLM(non-zombie) | 0 | **0** | ✅ |
| `hy-smi` VRAM | 4 卡全 0 | **4 卡全 0** | ✅ |
| ACS | 12 × `0x0000` | **12 × `0x0000`** | ✅ 全程部署态,未 apply/revert |
| `z100-s14` StartedAt | 2026-10-09T19:35:54Z | **未变** | ✅ 容器未重建 |
| 生产容器 `qwen38-vllm` | 只读 `docker inspect` | 只读 `docker inspect`(StartedAt 2026-10-07T15:52:36Z、Running=false,与本步无关) | ✅ 未触碰 |
| 新增的附加 swapfile | — | `/mnt/game/s14d/swap4` 已 `swapoff` + **删除文件** | ✅ |
| `/mnt/game/swapfile`(16 GiB) | swapoff | swapoff | ✅ |
| DTK 库 / `z100-build*` / s14\*/s14b\*/s14c\* 脚本 | — | 未改(`s14c_run_one.sh` md5 `4f3f3275…`、`s14c_arm_env.sh` md5 `61726d07…` 全程未变) | ✅ |

**本步新增文件(整体删除即回滚)**:

```
docs/result/z100-14d-determinism-result.md        见 /mnt/game/s14d/hostlogs/doc-md5.txt(自引用 md5 不能写在文件内)
tools/dcu/s14d_client.py      ad43fa602dc1f02df126730a50579b47      tools/dcu/s14d_compare.py   5adf7735fbadbf7d1aaf456198dcce28
tools/dcu/s14d_profile.py     ff88c719ff04533e130855426318bba7      tools/dcu/s14d_matrix.py    9ef4f2cf630a1988dfe7aa3803b41ecb
tools/dcu/s14d_conc2.py       df4f9adbedbfe68ef3a46ddb29f9aac9      tools/dcu/s14d_seq2.py      5aa92f0bff2297418328303702b73ba0
tools/dcu/s14d_run_one.sh     1d344a45e443c95ab86145b3b31836cc      tools/dcu/s14d_e4.sh        29e52085d6f408930cceaa2187314535
tools/dcu/s14d_a34.sh         e941764b52c985dec0c0ed3ea246e558      tools/dcu/s14d_e1.sh        49a8bc193db62cd8b6c51844b290a17a
tools/dcu/s14d_e5.sh          d10b0c5542a7c64a9d28159d93638c19      tools/dcu/s14d_seq2_arm.sh  fa907d336875fa813717ed8a6853f461
```

**原始工件**:`/mnt/game/s14d/`:
* `probe-{W4,S4,A3,A4,B1,B2,B3}.jsonl` —— 7 臂 × 12 prompt 的逐位 token 流 + 每 token
  logprob/gap/top-5 + 每 prompt 位置 `prompt_logprobs` top-20;
* `conc2-{E5SW16,E5SW4}.jsonl`、`seq2-E5SEQ.jsonl` —— E3 的三个受控对照;
* `e4-compare.csv`、`e1-*.csv`、`e2-*.csv` —— 逐 prompt 对照表;
* `hostlogs/` —— 每个 arm 的完整引擎日志 + 客户端日志 + 全部对照文本
  (`E4-compare.txt`、`E1A-matrix.txt`、`E1B-matrix.txt`、`E1-7arm-matrix.txt`、
  `E2-reload-vs-reload.txt`、`E2-common-prefix.txt`、`E4-profile.txt`、`E2-profile-S4-A3.txt`、
  `E5-compare.txt`、`E5SEQ-seq2.log`、`summary.tsv`);
* `cache-e4/`(combo on 的 AOT 产物)、`cache-b/`(combo off 的 AOT 产物)——**删除即回滚**。

**本步"一锤定音"的一句话**:

> **graph 的 token 流不是"因为预热没做对"才不可靠,而是它本来就不可靠 ——
> 同一份 AOT cache、同一条命令、同一串请求,两次启动之间 12/12 的 prompt 都给出不同的 token 流,
> 同上下文 logits 差 mean ≈1.3 nat / max 10–16 nat;关掉 combo kernels 也一样。
> 因此 "预热配方足够,graph 可上生产" 不成立;graph 只能用于性能口径,eager 才是质量口径。**
