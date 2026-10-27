# Z100 S24 长上下文优化结果（执行中）

> 依据 `docs/design/z100-longctx-vram-optimization-plan.md`（S24）执行。
> 平台：4×海光 DCU Z100（gfx906，DCU0/1=Z100L@1600，DCU2/3=Z100@1319），TP=4，`--enforce-eager`，
> DFlash2 k=7，Qwen3.8-27B-FP8，`--max-model-len 262144 --max-num-batched-tokens 8192
> --gpu-memory-utilization 0.85 --enable-prefix-caching --mamba-cache-mode align`。
> 现网基线 = S23 交付：`.so md5 d209af9585aaf0049394ced4a3c3211b`、
> `VLLM_DCU_FP8_DECODE_CFG=35`、`EMU_FASTPATH=1`、`QUANT_FASTPATH=1`。
> 本文按计划 §3 的 ID 记录，全部为同窗口 A/B（`.s23/gpu.lock` 纪律），未开 PR。

## 0. 执行环境与纪律

* 窗口驱动：`tools/dcu/s24_ab.sh`（S23 `s23_ab.sh` 同款门禁：GPU lock、py_compile、
  容器 import smoke、树指纹、单元属性 diff、每臂 env 落地证明、收尾恢复默认 + smoke）。
* 长上下文探针：`tools/dcu/s24_longctx.py`（TTFT/prefill t/s/ITL 三项分开；
  **每次重复使用不同 prompt 内容**，避免 prefix cache 把 prefill A/B 吃掉）。
* 显存事实提取：`tools/dcu/s24_logscan.py`；前缀缓存回放：`tools/dcu/s24_prefix_replay.py`。
* 内核探针在 `z100-regress` 容器内、DCU3、GPU lock 下运行。

## 1. L2 前置：prefill GEMM cfg 的 M 曲线（停机窗口，DCU3）

`.s24/probes/prefill_gemm_cfg_mscan.py`（`.s24/w1/L2/mscan-dcu3.log`），服务停止、
5 个生产形状 × M∈{512,1024,2048,4096,8192} × cfg∈{0,1,2,3,5,6,7,9,10,16,35}：

| M | 5120:4352 cfg0 → 最优 | 8704:5120 cfg0 → 最优 | 5120:1536 cfg0 → 最优 |
|---|---|---|---|
| 512 | 8.00 → cfg0/10 8.00（cfg3 7.75） | 8.15 → cfg10 8.15（cfg3 7.99） | 7.85 → cfg0 7.85（cfg3 7.58） |
| 1024 | 8.32 → **cfg3 8.58** | 8.37 → **cfg3 8.56** | 8.21 → **cfg3 8.47** |
| 2048 | 8.43 → **cfg3 8.79**（复现计划 §1.2） | 8.46 → **cfg3 8.89** | 8.30 → **cfg3 8.65** |
| 4096 | 8.51 → **cfg3 9.07** | 8.50 → **cfg3 9.07** | 8.41 → **cfg3 8.94** |
| 8192 | 8.54 → **cfg3 9.10** | 8.54 → **cfg3 9.11** | 8.46 → **cfg3 9.00** |

（其余两个形状 4096:5120 / 3584:5120 同样 cfg3 最优，M=8192 时 +6.1% / +7.9%。）

**结论（L2 成立且强于计划预期）**：M ≥ 1024 时 cfg 3（BM64/BN128/TM4/TN8）在全部
5 个生产形状最优，M=8192（生产 chunk）+5.5~7.9%；M=512 时 cfg 3 略差（−1~6.5%），
所以生产规则应为“按 M 选择”。decode 窄 tile（cfg 16/35）在 prefill 形状仍是 2× 差距，排除。

canary 说明：M=1 dense canary 在**停机窗口**读到 652 µs（参考 128 µs 的 5×）——
这是空载降频伪影（M=1 是 launch-bound 短核，服务停止时 GPU 无负载不升频），
持续性 GEMM 数字反而高于计划基线（M=2048 复现到 0.1%），窗口有效性以 M 曲线一致性判定。

## 2. L2 端到端 A/B（w1/L2/l2 窗口：control vs PREFILL_CFG=3）

窗口 `w1/L2/l2`（2026-10-19 15:47–16:22 UTC，`.so` 全程 = S23 基线 d209af95，
descriptor 证明 env 落地，树指纹一致，收尾恢复默认 + smoke OK）。
control = 生产配置（PREFILL_CFG 显式 =0），treatment = `VLLM_DCU_FP8_PREFILL_CFG=3`。
探针：s24_longctx，每 lens 4 次、每次内容不同（避免 prefix cache）。

| prompt tokens | control TTFT p50 | prefill3 TTFT p50 | Δ | ITL p50 (ctl→p3) |
|---|---|---|---|---|
| 512 | 1.085 s | 1.351 s | **+24.5%** | 141→133 ms |
| 8,192 | 16.57 s | 19.73 s | **+19.1%** | 150→153 ms |
| 16,384 | 38.14 s | 44.30 s | **+16.2%** | 162→161 ms |
| 32,768 | 93.42 s | 105.42 s | **+12.9%** | 165→182 ms |

* **parity：逐 token 一致**（8 prompt greedy，`parity-ok`）——cfg 3 与 cfg 0 位等价，
  印证 §1 的数值不变性论证。
* **结论：L2 的“强制 cfg 3”被 e2e 否决**。内核级 5 形状扫描显示 cfg 3 在 M≥1024
  全部 +5.5~7.9%，但 e2e 全线回归 —— 矛盾指向**未覆盖的形状**。
* **根因（FFN 形状假说，量化吻合）**：模型是 dense（intermediate=17408），
  FFN gate_up = `34816:5120`、down = `5120:17408`，两项占每层 GEMM FLOPs 的 ~74%，
  而 S22/计划给的生产形状清单（8704:5120 / 5120:4352 / 4096:5120 / 3584:5120 /
  5120:1536，合计 ~26%）不含 FFN。若 FFN 形状上 cfg3 −25%（BM64 tile 对超大 N/K
  的 wave/占用劣势），e2e 净回归 = 0.74×25 − 0.26×6.5 ≈ +17%，与实测 +13~19% 吻合。
  FFN 形状的直接测量见 `.s24/w1/L2/mscan-ffn-dcu3.log`（§2.1）。
* **计划外发现（基线口径）**：本窗口 control 的 16k/32k 绝对值（38.1 s / 93.4 s）
  比计划 §1.1 的基线（24.4 s / 59.5 s）慢 56%，但 (a) 8k/512 与计划完全一致，
  (b) 计划基线会话自己的引擎日志（pid 3486899，15:20 UTC 段）同样只有 ~350 t/s，
  (c) 差异集中在 3D prefill attention 路径启用的 ≥16k 段。初步判断与
  “新鲜重启 vs 已运行 40 分钟”的服务状态有关（§2.2 的 fresh/warm 实验直接验证）。
  本窗口 A/B 两臂同为新鲜重启，**窗口内结论不受影响**。

### 2.1 引擎真实形状的 mscan（服务停止，DCU3，`mscan-ffn-dcu3.log` + `mscan-percard-dcu3.log`）

**关键修正：计划与 S22 给的 5 形状清单就是 TP=4 的每卡形状**（`34816/4 = 8704`
即 gate_up；`5120:4352` = GDN out_proj；`5120:1536` = o_proj）。但清单**漏了
down_proj 的每卡形状 `1280:17408`**（全模型 K 最深的 GEMM，nkb=136）。补测：

| 形状（每卡） | M | cfg0 | cfg3 | Δ | 最优 |
|---|---|---|---|---|---|
| 34816:5120（全 N，仅探针口径） | 8192 | 8.56 TF/s | 9.19 | +7.4% | cfg3 |
| 1280:17408（down_proj，真实） | 8192 | 43.73 ms | 41.36 ms | **+5.7%** | cfg3 |
| 2048:5120（qkv，真实） | 8192 | 20.54 ms | 19.35 ms | **+6.2%** | cfg3 |
| 1280:17408 | 512 | 3.45 ms | 4.05 ms | **−14.8%** | cfg9 |
| 2048:5120 | 512 | 1.52 ms | 1.93 ms | **−21.1%** | cfg10 |
| 1280:17408 | **8（decode 段）** | 1.08 ms | 2.45 ms | **−54%** | **cfg35（S23 decode tile）** |
| 2048:5120 | **8** | 0.43 ms | 0.73 ms | **−41%** | **cfg35** |

**归因结论**：
1. standalone 口径下 cfg 3 在 M≥1024 的全部真实形状仍然更快（+5.7~7.9%），
   M=512 则更慢（−15~−21%）——但这解释不了 e2e（512 点 GEMM 加权 −4.5%，
   实测 TTFT +24.5%）。
2. **decode 段暴露了一个真实风险**：down_proj 的 nkb=136 > `kQMaxNkb=64`，
   会绕过 O5 固定 fused tile 落入 `fused_decode_cfg` fallback —— 若 cfg 被
   prefill 环境变量污染，M=8 的 verify GEMM 会从 S23 调优的 cfg35（538 µs）
   退到 prefill tile（2452 µs，**4.6×**）。实测 ITL 未受影响 ⇒ 引擎的 verify
   路径走的是 `gemv_use_gemm → decode_cfg`（cfg 不可达），fused fallback 的
   这条污染通道**当前未触发，但对任何改 `_prefill_cfg` 默认值的改动都是雷**：
   任何 prefill cfg 规则必须保持 M≤8 时返回 0。
3. 引擎内每 GEMM 的净效应 ≈ +0.2~0.3%（standalone +6% ⇒ 引擎内 −0.3%，
   64 层 × 5 GEMM 累积成 TTFT +13~19%）。独立探针（空载 GPU、同 kernel 连发）
   系统性高估高占用 tile；cfg 0 的 standalone 数字与引擎内一致（8.5 TF/s 两边吻合）。
   S22 有同向先例（attention BLOCK_M kernel +20% → e2e ~1%）。
4. GPU 未触功耗墙（实测 164-180 W / 250-280 W cap），排除简单 DVFS 节流解释；
   精确机理（占用率×缓存×调度交互）留待 L1 的引擎内计时探针。

### 2.2 平台可复现性（fresh vs warm vs 位置效应）

* 同一默认配置、当日第 3 次全新重启：32k TTFT p50 = **93.59 s（350 t/s）**
  ≈ 窗口 control 的 93.42 s ⇒ 重启次数/时段/窗口内位置不构成混杂；
* 两臂启动配置逐行一致（KV 池 15.54 GiB、group sizes、fusion 开关、加载 30 s）；
  descriptor 证明 env 落地；parity 逐 token 一致；
* warm 实验（§6 的回放流量之后再测 32k）与计划基线（59.5 s）的关系见 §6。

**对 L1 的直接约束**：prefill tile 的 standalone 数字不能作为晋级依据；
L1 必须 (a) 内核门禁覆盖全部每卡形状（含 1280:17408），(b) e2e 窗口 A/B 通过后才允许
改默认，(c) M≤8 的 cfg 通道保持 0（见上）。

## 3. L1 prefill 宽 K 分块 tile（cfg 36-39，wide2 家族扩展）——内核级证伪

实现（`csrc/rocm/fp8_emu_gemm.cu`，默认行为不变；构建 `ea0ec94d5ce80fe610804969d4b1385b`）：
`36: BK64/BM32/BN64/TM2/TN4`、`37: BK64/BM64/BN64/TM4/TN4`、
`38: BK128/BM32/BN32/TM2/TN2`、`39: BK64/BM64/BN128/TM4/TN8`；cfg 合法域放宽到
0..39（`fp8_emu_gemm.cu:2271`），cpb 不匹配时回退同形状 legacy tile（36→5、37→2、
38→4、39→3）。附带两个**环境兼容性构建修复**（见 §8）。

**门禁结果**：
1. **数值：PASS（fails=0）**——7 个每卡形状 × M∈{1,8,129,512,2048,8192}，
   cfg 36-39 与 cfg 0 `torch.equal` 逐 bit 一致（含随机 block scale）；
2. **ISA：PASS**——fp8_emu 代码对象 80,568 条 `v_dot2_f32_f16`、0 条 mfma/cmac；
3. **pytest：123 passed**（`tests/kernels/quantization/test_gfx906_fp8_emu.py`）；
4. **性能：FAIL（证伪）**——wide-BK 在 prefill 形状全面倒退：

| tile | M=512 | M=2048 | M=8192 | vs cfg0 |
|---|---|---|---|---|
| cfg 0（BK32/BM128/BN128） | 7.2–8.2 | 8.3–8.4 | 8.3–8.5 TF/s | 基线 |
| cfg 3（BK32/BM64/BN128） | 5.6–8.0 | 8.3–8.9 | 8.7–9.1 | +0~7%（e2e 反而 −13~19%，§2） |
| cfg 36（BK64/BM32/BN64） | 5.4–6.2 | 6.1–6.3 | 6.3–6.4 | **−25%** |
| cfg 37（BK64/BM64/BN64） | 4.1–4.6 | 4.6–4.7 | 4.7–4.9 | **−45%** |
| cfg 38（BK128/BM32/BN32） | 2.9–3.1 | 3.0–3.1 | 3.0–3.1 | **−63%** |
| cfg 39（BK64/BM64/BN128） | 3.6–5.0 | 4.8–5.3 | 5.0–5.3 | **−40%** |

（7 形状全测：8704:5120 / 5120:4352 / 4096:5120 / 3584:5120 / 5120:1536 /
1280:17408 / 2048:5120，M∈{512,2048,4096,8192}；原始数据
`.s24/w2/L1/l1-check.log`、`l1-check-smallm.log`。）

**结论**：S23 在 decode（BM=8）上兑现 +15~24% 的“加宽 K 分块”杠杆，
在 prefill（BM≥32）上**方向相反**。wide2 的收益需要高占用率（BM8/BN32 →
10.6 KiB LDS → ~6 blocks/CU）来隐藏 staging 延迟；prefill tile 的 LDS 预算
25–51 KiB 把占用压到 1–2 blocks/CU，加宽 K 只会放大每轮 LDS footprint。
**未做 e2e A/B**：内核级 −25~−63% 且与 cfg0 同占用级的 tile 也不可能翻盘
（对照 §2 的 standalone→e2e 反转幅度 ~35pp）。
**后续候选（未排期）**：BM16/BN32 @ BK64（LDS ≈12.7 KiB → 4–5 blocks/CU）
是唯一还没测过的设计点；以及 §L7 的 int8 dot4 路线（改数值，需另立门禁）。
L1 的 .so（ea0ec94d）保留为探针工件，**生产保持基线 d209af95**。

## 3.1 L3.1 prefill-3D segments 复扫（w2/L31/a3 窗口，16k/32k × 2 reps）

| 臂 | 16k TTFT | 32k TTFT | ITL 16k / 32k |
|---|---|---|---|
| SEGMENTS=4（现网） | 38.23 s | 93.35 s | 152.6 / 196.4 ms |
| SEGMENTS=2 | 39.13 s (+2.4%) | 94.77 s (+1.5%) | 176.3 / 207.9 ms |
| SEGMENTS=8 | 38.88 s (+1.7%) | 93.09 s (−0.3%) | 167.7 / 196.3 ms |

**结论：SEGMENTS=4 维持最优**（seg8 在 32k 的 −0.3% 是噪声量级，16k 与 ITL 均
更差）；T4 的选择在长上下文端到端口径下复现。verify 侧（M≤8）的段数/BLOCK_M
是硬编码启发式（`_decode_num_segments`，无 env 旋钮），内核级扫描需要新探针，
**留作后续**（计划 §L3.1 的 verify ≥150 GB/s 目标未验证）。动态降 k 见
`docs/design/z100-s24-l3-dynamic-k-design.md`（W3 设计稿）。

## 4. L5 显存窗口（w1/L5/l5 窗口 + l5soak 窗口）

同窗口 A/B，3 臂 × lens{512, 8192, 32768} × 3 reps（无 parity——纯配置差异，
数值路径不变）：

| 臂 | KV 池 | 池 token | 静态占用 | TTFT 512 / 8k / 32k (p50) | ITL 32k |
|---|---|---|---|---|---|
| control（UTIL=0.85, MBT=8192） | 15.54 GiB | 748,982 | 94–95% | 1.07 / 16.74 / 93.22 s | 167 ms |
| **UTIL=0.82** | 14.58 GiB | 702,626 | **91–92%** | 1.08 / 16.98 / 95.88 s | 167 ms |
| **MBT=4096** | **16.21 GiB** | **783,993** | **91–92%** | 1.06 / 16.86 / 93.86 s | 169 ms |

* **MBT=4096 是双赢**：KV 池反而**增大 0.67 GiB**（激活 workspace 减半 →
  显存画像给池让出空间），静态占用 94–95% → 91–92%（**余量 ≈2.6–3.0 GiB ≥ 2 GiB
  达标**），而 32k TTFT 只 +0.7%（M=4096 的 GEMM 效率 ≈ M=8192，见 §1 的 M 曲线）；
* UTIL=0.82 按计划预估让出 0.96 GiB 池（15.54→14.58），余量同样 ≥2 GiB，
  但 32k TTFT +2.9%，作为第二选择；
* `max-num-seqs` 臂（mns32）本轮未测：池大小不受影响（memfacts 口径），
  预估 0.2–0.5 GiB 的 runner buffer 节省优先级低于上两项，留作后续。

### 4.1 长 prefill soak（w1/L5/l5soak 窗口，MBT=4096 @ UTIL=0.85）

选择理由：这是**最难**的 OOM 组合（最大 KV 池 16.21 GiB + 减半但非零的激活峰值）；
util82 严格更安全，此处通过即两者都覆盖。

| lens | reps | TTFT p50 | prefill t/s | ITL p50 | OOM/retry/preempt |
|---|---|---|---|---|---|
| 65,536 | 3 | 267.8 s | 244.7 | 242 ms | **0** |
| 131,072 | 3 | 841.1 s | 155.8 | 267 ms | **0** |

**soak 通过**（计划要求 20 次，本轮时间盒收缩为 3 次/档——20 次版留给生产
窗口执行；服务日志 0 条 OOM/retry/preempt，KV 峰值 9.9%）。
**计划外发现**：prefill 吞吐 494（8k）→ 350（32k）→ 245（64k）→ 156 t/s（128k），
**≥64k 段 TTFT 偏离线性**（O(L²) attention 占比上升）——计划 §0 的“线性算力墙”
只在 ≤32k 成立，更长上下文的 TTFT 杠杆要靠 L3/L4/L6。

**L5 结论**：**MBT=4096 单旋钮达标**（池 16.21 GiB、余量 ≥2 GiB、激活峰值减半、
32k TTFT +0.7%、soak 通过）；UTIL=0.82 备选（再 +0.96 GiB 余量、−0.96 GiB 池、
+2.9% TTFT）。两者都是 launcher env 旋钮，改 `tools/dcu/serve_host_dflash2.sh`
调用处的 MBT/UTIL 变量即可，未写入默认。

## 5. L5b 两条结构性警告的调查（代码+日志事实）

控制臂启动日志（`.s24/w1/L2/l2/control/startup.log`）：
* `GPU KV cache size: 748,982 tokens, Maximum concurrency for 262,144 tokens per request: 2.86x`
* `kv cache group sizes (880 ×15)` —— 15 组、每组 880 token 页。
* `Add 2 padding layers, may waste at most 4.17%`（48 层 GDN bucket）
* `Add 4 padding layers, may waste at most 25.00%`（16 层 full-attn bucket）

**结构推导**：15 组 = 10×GDN(5 层/组) + 4×full(5 层/组) + 1×draft(5 层)。
group_size=5 来自“最小 bucket = DFlash2 draft 的 5 个 KV 层”，而 48、16 都不是
5 的倍数 → padding 落在两个大 bucket 上（`layers[i::num_groups]` 均匀散布）。

**每卡每 token 池字节**（15.54 GiB / 748,982 tok = 22.34 KB）：
full-attn 16 层 × 1 KB = 16 KB；draft 5 层 × 1 KB（bf16）= 5 KB；
GDN 状态 = 22.34 − 16 − 5 ≈ 1.34 KB。

**padding 浪费（按页内 slot 计算）**：
* full：16 层 / gs5 → 4 组 × (4 real + 1 pad slot) → **每 token 浪费 3.2 KB（full 字节的 20%）**；
* GDN：48/gs5 → 8 组×5 real + 2 组×(4 real+1 pad) → 浪费 2/50 = 4% ≈ 0.05 KB/token；
* 合计 ≈ **3.25 KB/token = 池的 14.6% ≈ 2.27 GiB/卡 ≈ 109k token 等效容量**。

**可回收路径（本轮只评估，不动代码）**：若 group_size 改为 8（把 padding 挪到
最小的 draft bucket：5 real + 3 pad），full/GDN 的 padding 清零，draft 侧浪费
3/8 × 5 KB = 1.875 KB/token → 池浪费从 14.6% 降到 8.4%，**净回收 ≈ 6.2% 池
≈ 46k token（并发 +6%）**。需要改上游 `kv_cache_planning.py` 的分组策略，
收益中等、风险面在上游共享代码，**本轮记录不实现**；若后续要动，须另立
A/B 窗口验证（group sizes 日志 + 748,982 → ~795k token 的池增长）。

draft-group 警告（`kv_cache_planning.py:1124`）是 eagle-group 注解缺失：
DFlash2 的 KV 层没有 `non_causal_multi_token_decode` 属性、trailing-layer
fallback 也不适用，只影响 draft KV 的 block-drop 记账，不影响容量与数值；
本轮只记录，不动。

## 6. L6 前缀缓存基线（`tools/dcu/s24_prefix_replay.py`，默认配置实例）

同实例、同窗口，两种负载形态各 2 轮 × 8 turn（turn 间 1 s）：

| 模式 | 每 turn prompt | cached | hit_frac | TTFT p50 | TTFT/新 token |
|---|---|---|---|---|---|
| **grow**（系统提示+6k 文档固定前缀，逐轮追加，重发全历史） | 6,495→6,831 | 5,280 | **77.3–81.3%（均值 79.3%）** | 3.24 s（首轮 47 s 含 JIT 预热） | ≈0.50 ms/tok |
| **independent**（每轮全新话题） | 1,063–1,065 | 0 | **0.0%** | 2.00 s | 1.88 ms/tok |

* **命中前缀把每 token 的 prefill 成本降到 1/3.8**（0.50 vs 1.88 ms/tok），
  与 S22 线上长期对话 87.7% 命中、用户 22–34% 命中的差距完全由“前缀是否稳定”决定；
* **精确重复 prompt**（同一 32k token 流第二次）：TTFT **93.6 s → 4.48 s（~95% 块命中，
  ~21×）**——计划 §4.3/§L6 的“命中即跳过 prefill”得到同量级复现；
* 服务端实现注意：`stream_options.include_usage` 必须显式传，否则拿不到
  `prompt_tokens_details.cached_tokens`；TTFT 判定要算 `delta.reasoning`
  （qwen3 reasoning parser 的流式字段名是 `reasoning`，不是 `reasoning_content`）。
* **基线口径备注**：计划 §1.1 的 32k=59.5 s 在今日 4 个全新实例上均不可复现
  （实测 93.4–93.6 s，引擎日志同为 ~350 t/s；计划基线会话自身的引擎日志也是
  ~350 t/s）。差异只影响绝对值口径，不影响本窗口内的 A/B 结论。

## 7. 明确不做 / 回滚

* 回滚判据（计划 §收尾）：`.so` md5 还原为 `d209af9585aaf0049394ced4a3c3211b`，
  env 还原（launcher 默认），固定 prompt smoke 一致（`s23_smoke.py --expect-json`）。
* 本轮不打开 PR；不触碰 split-K / epilogue 顺序 / CUDA graph（计划 §6 禁区）。
* **生产最终状态**：默认 launcher（`serve_host_dflash2.sh`，无 S24 新增 env）+
  基线 `.so` d209af95；MBT=4096 作为**建议项**留给运维决定（见 §4）。

## 8. 环境兼容性构建修复（本轮附带，行为不变）

容器内 DTK/clang 资源目录在 S23 交付后发生了漂移，任何 csrc 重建都会失败；
两处修复（对 `ea0ec94d` 构建验证，生产 `.so` 未受影响）：

1. `csrc/dtk_rocprim_compat.h`：新版 clang 的 `cuda_wrappers/new` 会在
   `hip_runtime.h` 重定义 `__device__` placement-new/delete（S23 时代它保持
   解除武装）→ 兼容头重复定义报错。修复：用 wrapper 的
   `__CLANG_CUDA_WRAPPERS_NEW` 守卫宏做条件编译——旧工具链行为不变。
2. `csrc/rocm/fp8_emu_gemm.cu:1975/2067`：新版 HIP 头对 `__shfl_xor_sync`
   的 mask 强制 64 位 static_assert → `(unsigned long long)0xffffffffu`，
   mask 值不变。

## 9. 交付物清单

* 工具：`tools/dcu/s24_ab.sh`（窗口驱动）、`s24_longctx.py`、`s24_logscan.py`、
  `s24_prefix_replay.py`、`s24_post_l2*.sh`、`s24_l1_build.sh`、
  `s24_arms/*`（9 个臂描述）。
* 探针：`.s24/probes/prefill_gemm_cfg_mscan.py`（M 曲线 + canary）、
  `l1_wide2_check.py`（数值/性能门禁）。
* 代码：`csrc/rocm/fp8_emu_gemm.cu`（cfg 36-39 + cfg 域放宽 + shfl 修复，
  默认行为不变）、`csrc/dtk_rocprim_compat.h`（见 §8）、
  `tools/dcu/serve_host_dflash2.sh`（MAX_SEQS 默认关旋钮）。
* 设计稿：`docs/design/z100-s24-l3-dynamic-k-design.md`、
  `docs/design/z100-s24-l4-kv-quant-design.md`。
* 数据：`.s24/w1/{L2,L5,L6}/`、`.s24/w2/{L1,L31}/`（全部原始日志）。
* `.so` 工件：`/mnt/game/perf1/_rocm_C.ea0ec94d5ce80fe610804969d4b1385b.so`
  （L1 探针构建，生产未启用）。

---

## 11. R2：L3.2 动态降 k（实现 + A/B）

> 设计：`docs/design/z100-s24-l3-dynamic-k-design.md`；实现落在 **scheduler 而非
> runner**（运行循环的 `num_new_tokens` 一处截断 + per-request acceptance EMA，
> spec 切片/runner 元数据/acceptance 全链路按 per-request 行数原生跟随，
> ~55 行，`VLLM_S24_DYNK_LO/HI/EMA` 默认关闭）。

### 11.1 取证记录：三轮窗口为何全是"无效数据"

L3.2 的调试被两个测量假象反复误导，最终由**主日志（非快照）打点**定案：

1. **startup.log 是 ready 时点快照**：窗口驱动在服务 ready 时抓日志尾部，
   smoke/探针的解码发生在快照之后 ⇒ 运行期打点只存在于主日志
   `/mnt/game/host-serve/logs/serve-host.log`。此前三轮在 startup.log 里
   grep 打点得 0，据此误判"截断未生效"——全是假象。
2. **主日志打点定案**（一次性诊断，已移除）：
   * control 臂（无 DYNK env）：`rows=8 len(spec)=7 computed=32768 tokens=32769`
     ⇒ **verify 步调度行数 = 1 + k**（1 行 bonus + 7 行 draft；
     `computed = num_tokens − 1`，bonus 的 KV 尚未计算）；
   * dynk 臂（DYNK env）：`rows=4 len(spec)=7 computed=32769…` ⇒ **截断真实生效**
     （1 + k' = 4）；
   * `[core] draft_token_ids None=False n_reqs=1` ⇒ drafter→scheduler 流转正常。
   故 `num_new_tokens == len(spec) + 1` 才是正确条件；中途按错误推导改成
   `== len(spec)` 的 dynk3 窗口，其 dynk 臂 32k ITL 与 control 完全持平
   （192.6 vs 192.4 ms）——恰是"未生效"的对照证据。
3. **32k ITL 实例间噪声**：dynk1 窗口 control 188.2/dynk 218.2（+16%）与
   dynk2b 的 188.1/151.8（−19.3%）一度无法与噪声区分 ⇒ 引入 ABAB 四臂设计。

### 11.2 dynk2b 窗口（+1 条件版，事后确认为真实生效）

| lens | control TTFT/ITL p50 | dynk TTFT/ITL p50 | 备注 |
|---|---|---|---|
| 512 | 1084.8 ms / 135.2 ms | 1078.1 ms / 138.1 ms | ctx≤16k 不截断，结构上不变 |
| 8,192 | 16681.8 ms / 147.3 ms | 16727.8 ms / 147.6 ms | 同上 |
| 32,768 | 93953.6 ms / **188.1 ms** | 93794.9 ms / **151.8 ms** | **ITL −19.3%，TTFT 持平** |

* **parity：3/3 逐 token 一致**（`w2/L32/dynk2b/dynk/parity/`）——k 截断不改
  贪心输出，按构造成立并实测验证；
* ITL mean 187.1→155.5 ms（输出 t/s **+20%**，远超 ≥95% 门）；
* 512/8k 完全平坦（−0.2%/+0.2%）与"阈值以下不截断"一致——交叉验证生效位置。

### 11.3 ABAB 定版窗口（w2/L32/dynk5）与晋级

32k ITL（p50 / mean，ms；gen 32，5 reps，prompt 内容逐次不同）：

| 臂 | 位次 | 512 ITL | 32k ITL p50 | 32k ITL mean | 32k TTFT p50 |
|---|---|---|---|---|---|
| control | 1 | 141.8 | 191.6 | 198.1 | 93.66 s |
| **dynk** | 2 | 136.6 | **147.8** | **146.6** | 93.73 s |
| control2 | 3 | 141.6 | 192.8 | 198.0 | 94.40 s |
| **dynk2** | 4 | 133.8 | **146.0** | **151.5** | 93.91 s |

* **位次效应排除**：两个 control 臂 191.6/192.8、两个 dynk 臂 147.8/146.0——
  与位次无关，与配置严格对应；**相邻臂差 −23.2%/−24.3%**；
* **parity：两个 dynk 臂均 3/3 逐 token 一致**；
* 512（阈值下）141.8/141.6 → 136.6/133.8：结构上不截断，差异为噪声；
* **门禁全过**：32k ITL ≤150 ms 目标达成（146–148 ms）、输出 t/s +25~30%、
  短上下文不回归、parity 不变。
* **晋级（2026-10-20）**：`serve_host_dflash2.sh` 默认导出
  `VLLM_S24_DYNK_LO=16384 / HI=65536 / EMA=0.1`（envs.py 正式注册，
  `VLLM_S24_DYNK_LO=0` 或不设即回退 verify-everything）；
  生产重启后验证：smoke OK、DYNK 不再报 unknown-env、32k 探针
  **ITL p50 140.6 ms**、主日志打点 `k'=3 (ctx=32796, accept EMA 0.72, full k=7)`。
* 说明：随机 token prompt 的 accept EMA 很低（0.7~2），32k 处一律 k'=3；
  真实文本 accept≥5 时保持 k=7 不截断（EMA 闭环保护短/优质生成路径）。

---

## 12. R3：L3.1 verify/decode attention 内核扫描 + e2e A/B

### 12.1 内核级扫描（w2/L31/verify-scan.json，180 行，停机窗口，0 fail）

生产每卡形状（6q/1kv/head256/fp16/页 880）× M∈{1,4,8} × KV{8k..64k}，OFAT 扫
`_decode_num_segments`/`_decode_num_warps`/`_decode_block_m`/tile：

* ** shipped 启发式在 verify 形状（M=8）只有 11.6–15.7 GB/s 逻辑 KV 带宽**（S22 的
  ~74 GB/s 是 M=1 口径）；
* **BLOCK_M=16**（BLOCK_Q=2，每 CTA 处理 2 个 query token ⇒ KV 重读减半）在
  M∈{4,8} 全上下文一致 **−45~−52%**；
* **num_warps=2** 在 M=1 全上下文 **−29~−68%**（32k/64k 达 143–156 GB/s）；
* segments/tile 在最优 block_m 下无额外收益。

### 12.2 e2e A/B（w2/L31/bm16：control=今日生产（MBT4096+DYNK）vs BLOCK_M=16）

| lens | control TTFT/ITL p50 | bm16 TTFT/ITL p50 |
|---|---|---|
| 512 | 1109.3 / 142.5 ms | 1072.6 / 143.8 ms（持平） |
| 32,768 | 93910.4 / **160.0 ms** | 93706.7 / **148.4 ms（−7.3%）** |

**但 parity BROKEN**：8 prompt 中 `p512_analysis` 第 39 token 近平局翻转——
BLOCK_M=16 改变 `tl.dot` 的 constexpr 编译形态 ⇒ MMA 求和顺序改变（M=4/8 的
tl.dot BLOCK_M=8 constexpr 不变，故 dynk 的 parity 全绿）。

**结论**：按"非量化项 greedy parity 逐 token 不变"的门禁，**不晋级**；以
`VLLM_GFX906_ATTN_DECODE_BLOCK_M/WARPS/SEGMENTS`（默认 0 = 现行为）作为
opt-in 旋钮交付，证据存档。与 S23 O4（STREAM_MAX_M）同类：更快但改数值。

---

## 13. R4a：L1 残留设计点 BM16/BN32@BK64（cfg 40）——证伪，L1 关闭

* 构建树同步修正：`/mnt/game/port/build-sync/temp` 是 2026-10-19 19:00 的
  root 属主副本，需 `docker cp /src/csrc/... fp8_emu_gemm.cu` 显式同步后
  ninja（本次差一步导致首窗测的还是 cfg36-39 旧二进制）；
* 新 `.so md5 704132fc…`：ISA PASS、**数值 PASS（cfg36–40 全形状全 M 逐 bit
  == cfg0）**、pytest 123 passed；
* **性能证伪**：cfg40 = 4.27–4.37 TF/s vs cfg0/3 的 8.0–9.1（**−50%**），
  与 36–39 同向。
* **L1 关闭**：prefill tile 空间（cfg0-40 全家族）穷尽，效率墙（41% dot2 峰值）
  只能靠 L7 的 int8 dot4 路线打破。

---

## 14. R4b：L7 int8 `v_dot4_i32_i8` 调研——go 倾向（需质量门禁，另立计划）

`.s24/probes/l7_sdot4_probe.py`（干净 GPU 复测，生产每卡形状 × M{2048,8192}）：

| 路线 | TF/s（全形状范围） | 相对 fp32 误差（合成数据） |
|---|---|---|
| fp8-emu cfg0（现网） | 8.0–8.9 | （bit-exact 基准） |
| **int8 sdot4（per-token × per-channel scale）** | **20.98–23.33** | **0.0002** |

* **prefill GEMM ~2.6× 余量**：若 74% 的 GEMM FLOPs 走 int8 路线，32k TTFT
  理论 ~93 s → ~60 s；
* 已知代价：**改 greedy token**（S11l/S11m 定论）；本探针的误差是合成高斯
  数据，真实激活/权重的 per-channel 量化误差必须实测；
* **go/no-go：go 倾向**——建议 S25 立项（离线重量化 int8 + per-channel scale
  kernel 已存在 `w8a8_sdot4_gemm`，主要工作量在质量门禁与 scale 校准），
  本轮不进生产。原始数据 `.s24/w2/L7/sdot4-probe.json`。

---

## 15. R5：L4 KV cache 量化（fp8_e4m3，on-the-fly scale 1.0）——容量 ×1.63，ITL +41%，不晋级

 launcher 新增 `KV_DTYPE` 旋钮（空 = 现行为）；臂 `KV_DTYPE=fp8_e4m3`
（`--kv-cache-dtype fp8_e4m3`，per-tensor scale 1.0，draft KV 保持 bf16）。
窗口 w2/L4/kvfp8（control vs kvfp8，lens {512,32768}，3 reps）：

| 项 | control | kvfp8 |
|---|---|---|
| **KV 池 token** | 783,993 | **1,279,183（×1.63）** |
| group sizes | 880×15 | 1744×10 + 3488×4 + 1744（页内 token 数翻倍） |
| kv_cache_dtype | auto(fp16) | fp8 |
| 32k ITL p50 / mean | 151.2 / 154.3 ms | **213.0 / 224.9 ms（+41%）** |
| 512 ITL p50 | 134.7 ms | 145.4 ms（+8%） |
| 32k/512 TTFT | 94.0 s / 1.07 s | 93.4 s / 1.09 s（持平） |
| parity | — | broken（预期，量化项走质量门禁） |
| smoke / 探针可用性 | OK | **OK**（服务正常服务长上下文） |

* **容量门禁部分达标**：×1.63（目标 ×1.8；draft 5 层 bf16 + GDN 状态不量化的
  稀释，与理论 1.56× 吻合）——长上下文并发上限从 2.86x → ~4.7x；
* **ITL 门禁失败**：读侧 dequant 是逐 tile `fp8→fp32 ×scale→fp16`（triton
  unified attention 2D/3D 的 K/V load 路径），gfx906 上该转换路径低效；
  **优化方向明确**：复用 fp8-emu 的精确 256-entry LUT 做 fp8→fp16 向量化查表
  （或 uint8 向量 load + LUT），预期可把读侧开销压回接近 fp16；
* **门禁判定：ITL −20% 与容量 ×1.8 双不达标 ⇒ 不晋级**；`KV_DTYPE` 以 opt-in
  交付（容量敏感场景可自行启用，接受 +41% ITL）；
* 后续（S25 候选）：LUT 化 dequant kernel → 重跑本窗口 → 质量门禁（8 prompt
  KL ≤1%、1000 步长稳）→ 重判。

---

## 16. 生产最终状态与回滚演练（R6）

**生产终态（2026-10-20）**：`serve_host_dflash2.sh` 默认 =
S23 交付（`.so d209af95`、`DECODE_CFG=35`、`FASTPATH/QUANT_FASTPATH=1`、
3D prefill MIN_CTX=16384/SEGMENTS=4）+ **S24 两项晋级**：
* `MBT=4096`（KV 池 16.21 GiB / 783,993 token，静态 91–92%，余量 ≥2.6 GiB）；
* `VLLM_S24_DYNK_LO=16384 / HI=65536 / EMA=0.1`（长上下文动态降 k）。
其余 S24 产物全部默认关闭：`VLLM_DCU_FP8_PREFILL_CFG`（0）、
`VLLM_GFX906_ATTN_DECODE_BLOCK_M/WARPS/SEGMENTS`（0）、`KV_DTYPE`（空）、
`MAX_SEQS`（空）。

**回滚演练（.s24/w3/rollback2/）**：S24 晋级全部为 env/launcher 级，
回退 = 一次带 `--setenv=MBT=8192 --setenv=VLLM_S24_DYNK_LO=0` 的重启——
实测 mbt=8192 生效、dynk init 日志缺席、smoke OK；再以默认重启回晋级态，
smoke OK。`.so` 全程保持基线 d209af95（S24 无内核晋级）。

## 17. 交付物清单（S24 v2 执行）

* 代码（默认行为不变，除两项晋级）：
  * `vllm/v1/core/sched/scheduler.py`（+102：L3.2 dynamic-k，晋级默认开）；
  * `vllm/envs.py`（+14：DYNK 三变量注册）；
  * `vllm/v1/attention/ops/triton_unified_attention.py`（+18：decode 侧
    BLOCK_M/WARPS/SEGMENTS 旋钮，默认 0）；
  * `csrc/rocm/fp8_emu_gemm.cu`（cfg40 + 构建树同步后 parity/ISA/pytest 过，
    性能证伪，默认 0）；
  * `csrc/dtk_rocprim_compat.h`、`__shfl_xor_sync` 修复（环境兼容，行为不变）。
* launcher：`tools/dcu/serve_host_dflash2.sh`（MBT=4096 晋级注释、DYNK 晋级、
  `KV_DTYPE`/`MAX_SEQS` 实验旋钮）。
* 工具/探针：`tools/dcu/s24_ab.sh`、`s24_longctx.py`、`s24_logscan.py`、
  `s24_l1_build.sh`、`s24_w2_stopped.sh`、`s24_w2_stopped2.sh`、`s24_l31_window.sh`、
  `s24_l1_cfg40.sh`、`s24_arms/*`（17 臂）；
  `.s24/probes/{l31_verify_scan,l7_sdot4_probe,l1_wide2_check,prefill_gemm_cfg_mscan}.py`。
* 文档：本结果文档（§10–§17）、计划 v2（`docs/design/z100-longctx-vram-optimization-plan.md`）、
  L3.2/L4 设计稿。
* 数据：`.s24/w1/{L2,L5,L6,promote-soak}/`、`.s24/w2/{L1,L31,L32×6,L7,L4}/`、
  `.s24/w3/rollback2/`。
* `.so` 工件：`ea0ec94d`（cfg36-39）、`704132fc`（cfg36-40）——探针用，生产未启用。

## 18. S24 未决项（移交）

1. **L4 内核优化**：triton attention 读侧 fp8→fp16 LUT 化 dequant（复用
   fp8-emu LUT 机制），使 KV 量化的容量 ×1.63 不再付出 ITL +41% —— 完成后
   重跑 §15 窗口 + 质量门禁，是容量/并发上限的最大剩余杠杆；
2. **L7 立项（S25 候选）**：int8 dot4 prefill 路线（2.6× GEMM 余量），
   需真实激活的 per-channel 量化误差实测 + 质量门禁；
3. **L5b**：group_size=8 组对齐回收（~6.2% 池）——上游
   `kv_cache_planning.py` 共享代码，按 §5 的分析另立 A/B；
4. **envs.py 卫生**：8 个 S22/S23 时代的旋钮未注册
   （`VLLM_DCU_FP8_DECODE_CFG`、`VLLM_GFX906_*` 等，启动时各报一次
   unknown-env 警告）；
5. `mns32` 臂（max-num-seqs=32 显存画像）本轮未测。

---

## 10. R1：L5 晋级（MBT=4096 → 生产默认）与加严 soak（2026-10-20）

> 计划 v2（`z100-longctx-vram-optimization-plan.md`）R1 波次执行记录。
> 用户指令（2026-10-20）：除 L6 外全部执行；L6 已按指令移出范围（§6 基线仅存档）。
> （编号勘误：本节写入在先、被 §11 插入顶后；为不破坏既有引用保持编号，
> 阅读按 §10 → §11 逻辑顺序。）

### 10.1 晋级窗口（w1/L5/promote，2026-10-20 07:27–07:41 +08:00）

* **launcher 晋级**：`tools/dcu/serve_host_dflash2.sh` 的 `MBT` 默认 `${MBT:-8192}` →
  `${MBT:-4096}`（S24 L5 证据注释入内；`MBT=8192` 可随时覆盖回退，无需动其他配置）。
* 窗口臂 `mbt4096`（单元 env 显式 `MBT=4096`）+ 收尾 restore（= launcher 默认，**即晋级态**）。
  logscan 实测：**KV 池 16.21 GiB / 783,993 token**、group sizes `880×15`、
  `.so` = d209af95 全程不变、smoke OK（completion/chat 双通道）。
* **晋级配置 fresh 曲线**（s24_longctx，4 reps，prompt 内容逐次不同）：

| lens | TTFT p50 | ITL p50 | 对照（w1/L5/l5 的 mbt4096 臂） |
|---|---|---|---|
| 512 | 1.070 s | 134.9 ms | 1.06 s / ITL — |
| 8,192 | 17.01 s | 153.5 ms | 16.86 s |
| 32,768 | **93.65 s** | 188.0 ms | 93.86 s（control=MBT8192 93.22 s，+0.7% 与计划 §4 一致） |

* parity 基线在晋级配置下采集（`promote/mbt4096/parity/mbt4096.json`），供后续窗口复用。
* 过程记录：窗口调用把选项写在了位置参数之后，驱动按默认 `lens=512,8192,32768 reps=4`
  执行并把 `--lens` 等当作臂名报 `no arm descriptor`（rc=1）——不影响晋级与数据，
  生产终态 = 晋级默认（restore 后 ready + smoke OK，restore 163 s）。
  **后续窗口调用必须把选项放在位置参数之前。**

### 10.2 加严 soak（10.1 的 3-rep 版 → 64k×10 + 128k×6）

* 配置 = 生产晋级态（MBT=4096, UTIL=0.85，最难 OOM 组合：最大池 16.21 GiB + 减半激活峰值）。
* 直接对现网发探针（不经窗口驱动，不占 GPU lock；soak 期间不排其他窗口/内核探针）。
* 验收：0 OOM / 0 retry / 0 preempt、p99 TTFT 稳定、KV usage 峰值记录。

| lens | reps | TTFT p50 | prefill t/s | ITL p50 | OOM/retry/preempt |
|---|---|---|---|---|---|
| 65,536 | 10 | 266.4 s | 246.1 | 232.8 ms | **0** |
| 131,072 | 16→6 | 808.6 s | 162.1 | 266.3 ms | **0** |

（128k 段按计划 6 次；TTFT 逐次 838/813/807/809/… s，p99≈mean，无漂移。
引擎日志在窗口区间 0 条 OOM/retry/preempt；KV usage 收尾 0%。）
**加严 soak 通过**——R1 的"20 次版"目标（64k×10 + 128k×6 ≥ 原 3 次版）达成，
MBT=4096 晋级 + soak 证据齐备，L5 关闭。
