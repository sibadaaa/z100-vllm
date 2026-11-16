# 立项:prefill 长上下文速度衰退(注意力线)(2026-11-06)

> **立项拍板**(用户,2026-11-06):验收 = **冷 131k / 262k 极限尾段**;范围 = **先全链路
> 归因再收窄**(立项只锁方法学,施工边界由 P1 归因数据决定);P0 基线立即在运行中的
> 服务上抓取。
> **与 ROADMAP 的关系**:本专项即 [ROADMAP-20261027](ROADMAP-20261027.md) R12
> (prefill/TTFT,"需要单独立账")的升格;decode 线现状见
> SOL-GAP-20261025 §5(93.1 ms / 31.3 t/s,已到诚实判读 1.32×)。
> R12 原注"不要排 2× 预算"针对**常数项 GEMM**(已 18.4–19.9 TF/s ≈ 屋顶,
> [prefill-rocblas-audit](../result/z100-prefill-rocblas-audit.md));本专项打的是
> **随长度增长的项**,两者不冲突。

---

## 0. 问题定义

**现象**:单请求 prefill 效率随累积 KV 增长持续衰退(非固定形状的常数项):

| prompt | TTFT | 折算 | 口径 |
|---|---:|---:|---|
| 8k | 16.6 s | ~494 t/s | S24 control(w8a16 前,含 3D 路径) |
| 16k | 38.1 s | ~429 t/s | S24 control;**≥16k 段集中在 3D prefill 路径启用后** |
| 32k | 93.4 s | ~351 t/s | S24 control(fresh 重启第 3 次复现 93.59 s) |
| 冷 131k | ~1370 s | 7.0 ms/tok | P0 复测 B 臂(FP16 物化);尾段 chunk **注意力主导(200k KV)** |

**问题一(长度项)**:尾段 chunk(M=8192,KV≈200k)的注意力成本是否已贴内核屋顶?
若未贴,空间在哪(并行度/tile/路由)?
**问题二(绝对值回归,遗留 1)**:[z100-p0-prefill-materialization-result](../result/z100-p0-prefill-materialization-result.md)
记录 upstream 同步(bb7e8dae1)后 prefill 效率 5647 → 9685 ms/1k(~1.7×)待 bisect。
本专项只**量化它在当前树的残留**,bisect 本身单独立账,不阻塞主线。

## 1. 栈身份(P0 口径;原服务 pid 954,2026-11-05 23:38 起,2026-11-06 01:38 UTC
因 §8 事故二重启为 pid 66030,env 与原栈逐项核对一致)

`Qwen3.8-27B-FP8`,TP=4(4/4 Z100L),`max_model_len 262144`,MBT 8192,max_num_seqs 4,
MTP k=2,cudagraph FULL_DECODE_ONLY,torch profiler 挂 `/mnt/game/s30c-prof`。
关键 env(实测 `/proc/954/environ`):

| env | 值 | 与本专项的关系 |
|---|---|---|
| `VLLM_DCU_FP16_PREFILL` | auto | 物化路线 ON(P0 复测 B 臂同款) |
| `VLLM_DCU_FP8_W8A16` | 1 | W8A16 整线 ON |
| `VLLM_GFX906_TRITON_ATTN_PREFILL_3D` | 1 | prefill split-KV(3D)总开关 |
| `VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX` | **16384** | **3D 路径门槛:S24 观察的衰退起点** |
| `VLLM_GFX906_ATTN_PREFILL_SEGMENTS` | **4** | **split-KV 段数:尾段并行度旋钮** |
| `VLLM_GFX906_ATTN_PREFILL_BLOCK_M/TILE/WARPS/STAGES` | 默认 | 4k–16k 调优值"32/32/4/1"(s22),长形状未复扫 |
| `VLLM_GFX906_ATTN_DECODE_BLOCK_M` | 16 | BM16 已采纳(decode 线) |
| `VLLM_DCU_BF16` | 1 | bf16 数据路径 opt-in ON |

模型注意力构成:64 层 = **48 GDN + 16 full-attn**(interval 4),full-attn GQA 24/4,
**head_dim 256**,RoPE 262144。

## 2. 注意力路径事实(gfx906)

* custom paged attention(csrc MFMA)在 gfx906 判死
  ([rocm.py:543](../../vllm/platforms/rocm.py#L543));full-attn 走本 fork 的
  [triton_unified_attention.py](../../vllm/v1/attention/ops/triton_unified_attention.py):
  prefill 在 `max_seqlen_k ≥ 3D_MIN_CTX` 时路由到 split-KV(3D)内核,否则 2D。
* 内核屋顶(同机实测):best GEMM 10.64 TF/s(attention prefill GEMM-only),
  bf16 大 GEMM 3.4 TF/s,D2D copy 777.5 GB/s
  (KERNEL_REPORT E2C §0)。
* E2C 短上下文(ctx≤709)口径:full-attn 仅占设备时间 1.06–1.42%,GDN ~25.8% ——
  **衰退项在短上下文不可见,只在长上下文显现**,这正是历史归因没抓住它的原因。
* 粗算术(立项依据,TP=4 按每卡分摊):每卡每 full-attn 层每 chunk FLOPs ≈
  4·(24/4)·q·KV·256 = 6144·q·KV。尾段 chunk(q=8192,KV≈200k)≈ 1.0e13 FLOP/层,
  × 16 层 = **1.6e14 FLOP/卡**;在 10.64 TF/s 屋顶 = 15.1 s,在 3 TF/s = 53.7 s。
  对照常数项(见 §7 初步拟合,≈22–25 s/chunk):**注意力只有在内核低于 ~7 TF/s 时
  才主导尾段**——内核效率是可证伪的第一假设,常数项是第二假设。

## 3. 问题分解

| 项 | 归属 | 随长度? | 已知状态 |
|---|---|---|---|
| **长度项** | full-attn 16 层,q×KV FLOPs + KV 读放大 | ∝ | 本专项主目标 |
| 常数项 | GDN prefill(与 KV 无关,~14 ms/call@M=512,M 线性)、W8A16 GEMM(≈屋顶)、AR | ✗ | 不在本专项施工范围,归因时需剥离 |
| 状态项 | fresh/warm 服务状态(S24 §2.2 未关闭)、遗留 1 回归残留 | ? | P0/P1 先量化 |

## 4. 假设清单(全部可证伪)

| # | 假设 | 证伪/证实判据 |
|---|---|---|
| H1 | 尾段 full-attn 3D 内核在 KV≥100k 形状远离屋顶(TF/s 或 GB/s 口径) | P2 微测:KV∈{32k..262k}×q=8192 逐形状实测 vs 10.6 TF/s / KV 字节带宽 |
| H2 | `SEGMENTS=4` 使尾段 CTA 数不足(60 CU wave 量化),并行度不随 KV 扩展 | P2:segments∈{4,8,16,32} × KV 扫描;占用率清单(PLAN-20261029 §0 流程) |
| H3 | `3D_MIN_CTX=16384` 门槛附近路由次优(16k–32k 段 2D/3D 选择不当) | P1:门槛前后 chunk 时间跳变;P2:同形状 2D vs 3D 对拍 |
| H4 | 遗留 1(1.7×)在当前树仍存在 | P0:当前树 ms/1k vs 033606 基线 5647 / 物化 B 臂 7723 |
| H5 | 非注意力项随 chunk 序号累积(host gap / 调度 / prefix-cache hash) | P1:chunk 时间戳线性回归,残差归因 |
| H6 | 262k 段存在新病(KV 池压力 / 调度 / 数值) | P4 冒烟:262k 单请求完成、0 崩溃、贪心一致 |

## 5. 阶段计划与门禁

| 阶段 | 内容 | 产物 / 门禁 |
|---|---|---|
| **P0**(✅ 2026-11-06) | 服务级 TTFT–长度曲线 4k→131k(本服务口径);对齐历史三点(5647/7723/S24 曲线) | `runs/prefill-lctx/p0/`;§7 曲线 + H4 判定(证伪) |
| **P1**(✅ 2026-11-06) | chunk 级归因:profiler trace 四 rank;选点 131k(全程 19 step) | §8:注意力 1.47 TF/s(14% 屋顶)、占 85%;H1 证实、H5 证伪 |
| **P2**(✅ 2026-11-06) | 内核级:ofat 38 配置 + block_size/locality/stride/负载/时钟 五轮对拍 | §8 P2 结论:旋钮收干(3d:4 最优 2.39 TF);**in-vivo 1.63× 惩罚坐实且归因未解** |
| **P3** | **已交接(2026-11-06,用户裁决"设计/施工图交其他 agent")**:设计图 + 施工图见 [docs/design/z100/prefill-lctx/](../design/z100/prefill-lctx/z100-prefill-lctx-design.md)——P3-a(in-vivo 1.63× 根因,rocprof 对拍,潜在 -37%)+ P3-b(段往返税 −12~19%;KV 扫描重构 2.4→4-6 TF),每项过六步 | 施工图含 WP0-WP4、精确命令、预期值与雷区清单 |
| **P4** | 验收:冷 131k 尾段 chunk 时延 + 全曲线;decode TPOT ≤±3%;262k−32 冒烟(H6,注意 400 教训) | result 文档 + 采纳/否决裁决 |

## 6. 纪律(承 AGENTS.md 与项目先例)

1. 单臂服务;A/B 同窗口、同代码树、env 逐项钉死;**不跨窗口比较**。
2. 每次重复换 prompt 内容(prefix cache 会把 prefill A/B 吃成 cache A/B,S24 教训)。
3. 长点(≥64k)测量成本高(reps=1 起步);正式 A/B 时 64k reps≥2、131k 至少 2 次取 p50,
   并区分 fresh/warm(S24 §2.2)。`s24_longctx.py` 必须 `gen≥2`(gen=1 会使其
   summary 崩溃,2026-11-06 首轮教训)。
4. 注意力改动会改 MMA 归约顺序 → 按近平局披露协议
   ([ATTN-OPT §3](ATTN-OPT-20261025.md)):per-op fp64 ≤1e-3 + 翻转位置 top-2 差披露。
5. 不动 decode 旋钮(BM16/MTP/AR 栈已采纳);P3 任何 prefill cfg 改动保持 M≤8 通道为 0
   (S24 §2.1 的 4.6× 雷)。
6. **长点判死必须先过模型预算**(2026-11-06 事故):131k 外推要按"平均 ms/tok 随
   长度继续上升"折算,262k 全程预算 ≈40 min(平均 ≈9 ms/tok);>预算 1.5× 且
   KV 占用无进展才判死。
7. 服务只允许一个 arm;重启必须复刻原 cmdline+env——项目的 launcher 脚本
   (s27_graph_launch.sh,MBT 4096+piecewise)与生产栈(MBT 8192+MTP k=2+
   FULL_DECODE_ONLY)已分叉,不能直接用;取证模板见
   `runs/prefill-lctx/incident/`(environ/cmdline 快照 + relaunch.py)。

## 7. P0 首批数据(2026-11-06,生产栈口径见 §1)

运行记录:`runs/prefill-lctx/p0/longctx.jsonl`(口径 `s24_longctx.py`,服务空载,
每请求前 running/waiting=0;内容逐次换 seed 防 prefix cache)。

| prompt tokens | TTFT | prefill t/s | ms/tok | chunks(MBT 8192) | reps |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 2,055 ms | 1,993 | 0.50 | 1 | 2 |
| 8,192 | 9,720 ms | 843 | 1.19 | 1 | 2 |
| 16,384 | 24,309 ms | 674 | 1.48 | 2 | 2 |
| 32,768 | 67,061 ms | 489 | 2.05 | 4 | 2 |
| 65,536 | 208,675 ms | 314 | 3.19 | 8 | 1 |
| 131,072 | **722,863 ms** | **181** | **5.52** | 16 | 1 |
| 262,000 | **2,667,788 ms** | **98** | **10.18** | ~34+尾 | 1 |
| 262,144 | HTTP 400:262144 prompt + 16 out > max_model_len;上限冒烟用 262000 | — | — | — | 0 |

(4k 首轮 gen=1 跑出 4.63 s,是服务自 07:38 起的首批请求,含一次性 warmup;
gen=16 复测 2.05 s 为干净值。64k 点来自首轮,当时服务已暖,数字有效。)

**形状解读(初步,P1 待证)**:

1. **每 token 成本 4k→64k 劣化 6.4×**(0.50→3.19 ms/tok);对比线性扫描基线
   (若每 chunk 成本恒定),64k 处超额 6.4 倍——衰退陡于"多扫一倍 KV"的直觉。
2. **边际 chunk 成本随 KV 陡增**:8k→16k 增量 14.6 s(1 个新 chunk,KV≈12k)、
   16k→32k 增量 42.8 s(2 个)、32k→64k 增量 141.6 s(4 个)。与"注意力 ∝ q·KV +
   每 chunk 常数项(附 GDN+GEMM+AR)"模型同向;常数项与注意力项的占比正是 P1
   profiler 归因要给出的数。
3. **两参数模型拟合(TTFT = n·C + Σ_attn,用块内因果修正 K̄=p+4096)**:
   四个 TTFT 点(8k/16k/32k/64k)联立解出 **C ≈ 7.3 s/chunk(常数项)、
   注意力有效吞吐 A ≈ 1.36 TF/s**;对 32k 边际(预测 43.8 vs 实测 42.75 s)与
   64k 边际(预测 145.9 vs 141.6 s)外推误差均 <3%。**131k 点独立验证:预测 740 s
   vs 实测 722.9 s,误差 2.4%** ——五点全部自洽,模型在 P1 被证伪之前可用:
   * 131k 段注意力占 TTFT ≈ **86%**(623/723 s)——"尾段注意力主导"首次有了数;
   * **full-attn prefill 内核只跑到屋顶的 ~13%(1.36 vs 10.64 TF/s),8× 空间**;
     若内核到屋顶、常数项不动,131k TTFT 723 → ~196 s(**3.7×**),这是本专项
     的头条目标值;
   * 常数项 ≈7.3 s/chunk × 16 chunk ≈ 117 s(131k 占 16%),第二优先;
   * **注意**:这是把全部非注意力项压缩成一个常数的粗归因,若非注意力项本身
     随 KV 缓增(量化/调度/AR 大消息 84MB/次),A 会被低估——P1 profiler 必须复核,
     P2 内核微测(脱离服务、真实形状)是最终裁决。
4. **ITL 同步劣化**:decode ITL p50 41.8→65.5 ms(4k→32k)→ 130.1 ms(131k),
   长上下文 decode 的 KV 读放大已在;专项门禁"decode TPOT ≤±3%"需以同长度对照,
   不跨长度。
5. **H4 判定(遗留 1 回归)**:当前树 131k = 5.52 ms/tok,**快于**物化 B 臂
   (2026-10-21)7.0 ms/tok(1.27×)。遗留 1 的 ~1.7× 回归在当前树 131k 全长口径
   **未复现**;两代数字的口径差(冷 vs 暖)不足以翻转结论。正式闭环留一次
   fresh-restart 冷 131k 对照(P1 窗口顺带);bisect 不立项。
6. **与历史对比**:8k/32k 比 S24 control 快 1.7×/1.39×(W8A16+物化采纳后),
   S24 时代绝对值不再作基线。

## 8. P1 归因结果(2026-11-06,profiler trace 四 rank,job `bash-4ce3964e7744444b`)

**方法**:`/start_profile`(prefix `p1lctx131k`)→ 131k 请求(第二次独立 run,
TTFT 724.8 s,与 P0 的 722.9 s 差 0.3%——profiler 开销可忽略、数字可复现)→
`/stop_profile`;解析器 `tools/dcu/p1_chunk_attribution.py`(本次新增),
按 `gpu_user_annotation execute_context_*` 切 step 窗口,窗内 kernel 按
name 分类聚合。产物:`runs/prefill-lctx/p1/attr_rank0.json` +
`/mnt/game/s30c-prof/p1lctx131k_rank{0..3}.*.pt.trace.json.gz`。

**事实表**(四 rank 一致,<0.5% 差异;稳态主 chunk = 7488 tokens,共 17 个 + 尾部
2496 + 1280 两个):

| 指标 | 值 | 说明 |
|---|---|---|
| chunk 切分 | **7488 tok/chunk**(非 8192) | MBT 8192 被 MTP 预算挤占;P2 形状须用 q=7488 |
| 注意力内核 | `kernel_unified_attention_3d` **1.44–1.47 TF/s** | 屋顶 10.64 的 **~14%**,7.2× 空间 |
| 注意力占墙钟 | p50 **85.1–85.3%**(尾段 90.8%) | 两参数模型预测 86% ✓ |
| 每 chunk 常数项 | **≈6.2 s** = GEMM 4.59(Cijk fp16 物化,256 次)+ AR 0.68 + GDN 0.25 + other 0.66 | 各 chunk 恒定 |
| GDN(48 层) | **仅 0.25 s/chunk**(`chunk_gated_delta_rule_fwd_*` 族) | chunked 路径高效;M 线性外推(10.8 s)证伪 |
| GPU busy | **99.9%**(kernel 68048 / wall 68110 ms) | H5(非注意力随长度累积/host gap)尾段证伪 |

**逐 chunk 一行式**(last 主 chunk,KV≈124k):attn 61.9 s(16 层 × 3.87 s/层)+
GEMM 4.56 + AR 0.68 + 杂 1.0 = 68.1 s。注意力若到屋顶(8.6 s)该 chunk → ~15 s。

**假设裁决**:
* **H1 证实(带数字)**:full-attn prefill 3D 内核在真实尾段形状 q=7488/KV≈124k
  只跑 1.44–1.47 TF/s;
* **H5 证伪(尾段)**:常数项逐 chunk 恒定,GPU busy 99.9%;
* **H4 证伪**:当前树 131k 快于 B 臂(§7.5);
* H2/H3(segments 并行度、16k 路由门槛)移交 P2 内核微测裁决。

**262k 首测事故与教训(2026-11-06)**:首个 262k 请求在 41.2 分钟时被误判卡死
掐断(客户端断开 → 引擎 ~2 min 内 GPU 干净回闲置)。事后按两参数模型重算:
262k 理论时长 ≈ **39.6 min**(34 chunk × (6.2 s + 0.5005·K̄ s),K̄ 总和 4.33M)
+ 未建模的 draft 262k prefill 尾巴——**掐断时可能只剩 1-2 个 chunk**,属误杀。
静态检查确认 prefill 3D 的 segment workspace 按 chunk(7488)分配、decode 段
缓冲按 token 行分配,**ctx=262k 无缓冲上限**。

**事故二(H6 真发现)**:客户端 abort 后 **API 服务整体挂起**——`/v1/completions`
与 `/v1/models` 全部超时、GPU 已空闲、SIGTERM 被无视(需 SIGKILL 清树)。
即:**262k 级 mid-prefill 请求被 abort 后,引擎虽然停了算力,但 API/核心循环
没有恢复服务能力**。根因未定位(py-spy 需 sudo,栈未抓到);处置:重启服务
(用户侧重启,env 与被杀栈逐项一致,2026-11-06 01:38 UTC 健康);取证快照
(environ/cmdline/relaunch.py)存 `runs/prefill-lctx/incident/`。
**此条升格为 H6 的正式子项:任何 262k 冒烟协议必须带"abort 后服务可用性"检查,
P4 验收前需向引擎侧单独报修。**重跑已落地,见下方事故一结局。**

**事故一结局(2026-11-06 02:21 UTC)**:262k 重跑落地 **TTFT = 2,667.8 s(44.5 min,
10.18 ms/tok)**——首测 41.2 min 掐断时确实还差 ~3 min,误杀实锤;真实时长比线性
模型(39.6 min)多 ~12%(疑似 draft 262k prefill + 尾段 cache 效应,未归因)。
**H6 就绪性:262k 单请求完整跑通,0 OOM/0 崩溃,decode ITL p50 288.6 ms**;
每 token 成本 4k→262k 全程 **20.4×**(0.50→10.18 ms/tok)。
**P2 结论(2026-11-06,ofat 38 配置 + 4 轮专项对拍,`runs/prefill-lctx/p2/`)**:

1. **形状天花板**:现设计内核在 131k/262k 尾档的旋钮空间最优 = 32/32/4/1/3d:4
   (2447 ms @k̄=127k,2.39 TF useful,79.9 GB/s;262k 档 4960 ms 完美线性)。
   segments 2/4/8/16 持平、2D +9.8%,其余 30 变体全差;6 配置 LDS>64KB/OOM 硬墙。
   **C1/C2 无 env-only 收益;block_size 16 vs 800 无差;shuffled vs identity 无差。**
2. **in-vivo 惩罚(未解,升 P3-a)**:serving 逐 chunk 稳态 **31.3 µs/k-token** vs
   探针 19.2 µs/k-token = **均匀 1.63×**(chunk 5–17 线性拟合 R²≈1,无固定项;
   chunk 1–3 有 ~13% 一次性 JIT/工作区暖机)。已证伪:配置/grid/strides/池布局/
   block_size/block locality/负载竞争(+3%)/时钟(sclk 1600 双方一致)/profiler
   膨胀(TTFT 差仅 +1.9s)。**这是当前最大的单杆:解掉 = 注意力 -37%,先于内核重写。**
   取证需 rocprofv3 附着 serving(SCLK_UTIL/MemUnitBusy/带宽%),要一次重启窗口。
3. 数值一致性:baseline 带 ref-check 通过(2D);3d:4 的 ref-check 待 P3 前补。

**P2 OFAT 裁决(2026-11-06,38 配置 @ctx=131072/q=7488,`runs/prefill-lctx/p2/ofat131k.json`)**:

1. **serving 现配 32/32/4/1/3d:4 = 2446.8 ms(2.39 TF useful,79.9 GB/s)已是
   现有旋钮空间最优**:3d:8/16 持平(±0.2%),3d:2 +1.6%,2D +9.8%;
   其余 30 变体全部更差(BM 8/16/64/128、tile 16/64/128、warps 2/8、stages 2/3、
   以及 23 个双旋钮组合)。6 个大 tile/多 stage 配置 LDS 超 64KB 或段 workspace
   OOM(硬墙证据)。**C1(segments)与 C2(tile)空间收干,无 env-only 级别收益**。
2. **现有内核设计的形状天花板 ≈2.4 TF/s = 屋顶 22%**——旋钮无解,剩余空间在
   C3(内核本身:stages=1 无预取、KV scan 结构、LDS 复用)。
3. **主线索转为 serving-vs-probe 1.58× 差**(3.87 vs 2.45 s/层,同配置同形状):
   嫌疑排序 = 分页 KV block 分布(探针 identity vs serving 分页散布)>
   时钟状态 > launch 细节。验证批次(job `bash-c7897b11504049d5`):
   `--block-map shuffle` 直接测 locality 敏感性 + 262k 尾档复测。
   若 locality 实锤,P3 新增长上下文 KV 分配顺序/压缩候选(框架层)。

**施工清单 v2(OFAT 后按证据重排,替代原 C1/C2/C3)**:
1. **P3-a locality/路由(框架层,若 shuffle 实锤)**:长上下文 prefill 的 KV
   block 分配顺序修正(或 >阈值段改回 2D 路由),内核上限收益 ~1.4×;
2. **P3-b 内核改造(C3)**:3D 内核 prefetch/双缓冲、KV scan 结构——2.4 TF 起,
   目标 4–6 TF( kernels 门禁:微测→数值对照→服务 smoke→同口径 e2e);
3. **第二优先(不变)**:常数项 6.2 s/chunk(GEMM 4.6 + AR 0.68 + GDN 0.25 +
   other 0.66);验收折算:注意力到 3 TF/s 即 131k 723→~370 s;到屋顶 ~200 s。

### 8.1 P2 runbook(2026-11-06 修订:**不需停机窗口,只需服务静默**——本会话已实测
服务同款 env 可直接初始化 DCU(cuda_ok,旧 SOL-GAP §5.1 沙箱警告对新沙箱不成立);
探针在 DCU3 跑,内存 ~1–2 GB << 每卡余量 ~5 GB;s22 探针免改造——常量即每卡
6/1/256,`--ctx/--m-list` 直接传真实尾档。执行时用 flock 串行化,并在服务无
在飞请求时跑(P0 探针的 quota 记录可佐证静默):

```bash
ENV=runs/prefill-lctx/incident/with_serve_env.py
# 0) baseline 复现(P1 数字对齐:目标 ≈1.47 TF/s @ KV=124k, q=7488)
.venv/bin/python $ENV .venv/bin/python tools/dcu/s22_attn_prefill_probe.py \
    --ctx 124000 --m-list 7488 --preset baseline --tag p2base
# 1) C1: segments OFAT(env 等价旋钮,VLLM_GFX906_ATTN_PREFILL_SEGMENTS)
.venv/bin/python $ENV .venv/bin/python tools/dcu/s22_attn_prefill_probe.py \
    --ctx 124000 --m-list 7488 --segments-list 2,4,8,16,32 \
    --block-m-list 32 --tile-list 32 --warps-list 4 --stages-list 1 --tag p2c1
# 2) C2: 长形状 tile 复扫
.venv/bin/python $ENV .venv/bin/python tools/dcu/s22_attn_prefill_probe.py \
    --ctx 124000 --m-list 7488 --block-m-list 32,64,128 --tile-list 32,64,128 \
    --warps-list 4,8 --stages-list 1,2,3 --segments-list 8 --tag p2c2
# 3) 尾档: 同上两组在 --ctx 64000 与 248000 各跑一遍(262k 验收上限的服务形状)
# 4) H3: 门槛形状 2D vs 3D 对拍(--ctx 16384/32768,含 path 配置的 preset)
```

判据:C1/C2 任一配置在三个 KV 档同时 ≥1.5× 且数值对拍(fp32 参考)通过 →
进服务级单旋钮 A/B(env-only,可避开停机);否则进 C3 内核改造
(rocprof 字节量/L2 命中定位 KV 重读)。262k 服务形状验收点已由 P0 补测
(prompt 262000,job `bash-6c701830ec5d4ebb`)。

**测量纪律教训(记入 §6 第 3 条)**:`s24_longctx.py` 的 summary 段在 `gen=1` 时崩溃
(`itl_mean_ms=None` 进 `statistics.median`),首轮 8k/16k/32k 因此漏跑;
该探针按设计要求 `gen≥2`,后续一律用默认 gen=16。64k 点(gen=1)本身有效。

**P1 通道已确认**:服务带 `--profiler-config {"profiler":"torch", dir=/mnt/game/s30c-prof}`,
`POST /start_profile|/stop_profile` 端点可用
([profile/api_router.py](../../vllm/entrypoints/serve/profile/api_router.py));
服务 stdout 为 socket(非文件),chunk 级时间戳归因以 profiler trace 为准。
P2 探针 [s22_attn_prefill_probe.py](../../tools/dcu/s22_attn_prefill_probe.py)
已支持 `--ctx 49152` 级、segments/block-m/tile/warps/stages 扫描与 fp32 参考对拍,
仅需扩长 KV 档(128k/200k/262k)——注意其内核微测需停机窗口或 DCU3 空闲纪律。
