# S37 设计图:W3 收口 + MTP 重启裁决 + W4 AR 融合 + split-K 评估

> 日期:2026-10-29。输入:S36 终局(`docs/result/z100-s36-final.md`)、
> 当日分解/扫描原始数据(`docs/perf/data/s36-*.txt`)、B1 根因
> (`docs/perf/data/s36-b1-dynk.txt`)、WP5 存档
> (`docs/design/z100/s34/z100-s34-wp5-splitk-gemv.md`)、wp4 融合图
> (`docs/design/z100/s34/z100-s34-wp4-ar-fusion.md`)。
> **与 S36 的关系**:S36 完成 W2' 分解(两条修法均位同 PASS 但回归,已回滚)、
> W3 段数扫描(256 档正向)、B1 根因(dynk 阈值错配);本设计承接其三个
> 未收口项并补最大单项(W4)。

---

## 0. 结构与排序

```
Track A(内核/服务轨,全部 e2e 在无 MTP 臂、同口径 A/B)
  A1  W3 收口    SEG_MAX 128↔256 ABAB 确认 + executor 自杀治理   → 默认值裁决
  A2  W4         push AR 出口融合(采纳 s34-wp4 §2-3)            −0.8~−1.4 ms/round
  A3  split-K    可行性微测(0.5d,kill-fast)→ [视结果最小实现]   o ≤18µs(packed)
Track B(MTP 轨,独立部署与测量,不与 A 混测)
  B1  B2 裁决    MTP 臂 dynk-off 重启 A/B + durable 修法          draft/drafts≈2.0
```

排序理由:确定性优先——W3 收口只差一次统计确认;MTP 修复是零代码先导 +
十行 durable 修法,且长 ctx 收益上限最大(步长 45→~28ms 的折算);W4 是
1-1.5 天 csrc 工程,排 MTP 决议之后(步构成变化后收益口径重测,wp4 原判据);
split-K 按 S36 证据收益上界最小(≈0.3-0.4 ms/步),放最后 kill-fast。

- **测量隔离**(不变):A 轨永远在无 MTP 臂、基线 `.so` 口径 A/B;B 轨
  引入 MTP 口径后单独成档,两轨数据不交叉归因。
- **基线快照**(2026-10-29,S36 收尾态):服务 base2 在役(默认 env +
  基线 `.so` md5 `86470c5d`);无 MTP ITL 28.0/29.7/35.0/45.1 ms
  @16k/32k/64k/128k(15 token 点,`s36-w3-sweep-base2.txt`)。

---

## 1. A1/W3 收口:SEG_MAX 默认值裁决 + executor 自杀治理

### 1.1 现状(S36 遗留)

- 段数策略已 env 化:`VLLM_GFX906_ATTN_DECODE_SEG_MAX`
  (triton_unified_attention.py:56-64,默认 128 = 行为零变更;缓冲随
  capacity 自动放大,256 档 +~17MB/卡)。
- 单次交叉对比(跨实例):256 档 16k −4.4%(双基 28.1/28.0 高度可复现)、
  128k −5.3%(双侧干净 15-token 采样)、32k 不可判定(基线自身离散 ~5%)。
- 缺最后一口:**同协议下确认默认值改动**。跨实例 A/A 噪声(32k 点 ~5%)
  大于预期收益,必须 ABAB 同日交错消除实例次序混杂。

### 1.2 设计:ABAB 交错 + 加大采样

- **协议**:A=SEG_MAX=256,B=默认(128),次序 A→B→A→B 共 4 实例;
  每实例测 {16384, 32768, 65536, 131072} × 48 token × 2 rep(把
  ctx_sweep 的 max_tokens 参数化,16→48,p50 更稳;128k 每实例 1 rep,
  因 warm 345s)。
- **判据**:每个 ctx 取 A 两实例中位 vs B 两实例中位;128k 差 ≥3% 且方向
  为负 → **默认值改 256**;16k 同向 ≥3% 为佐证;若 32k 仍不可判定则披露
  (不再追)。
- **统计口径**:全部数据落 `docs/perf/data/s37-w3-abab-*.txt`;结论表
  必须同时给出同实例 A/A 重复的 drift 作为噪声带。

### 1.3 executor 自杀治理

- S36 事故:连跑 5h 的实例在 128k 首测时 executor RPC 超时自杀
  (`multiproc_executor` 等待 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`(默认
  300s)超时 → EngineGenerateError → 引擎自杀;此前 shm_broadcast 60s×3
  警告)。新实例未再现,疑与某长 ctx 变体首次 Triton 特化编译 >300s 有关。
- 治理三件:
  1. **预热线**:所有臂的启动器在 health 200 后、探针前,先跑一条 128k
     warm(2 token)并等待完成(把首次编译成本从测量窗口挪走;同时该
     请求本身就复现/排除自杀路径);
  2. **超时旋钮**:探针/长 ctx 臂 env `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200`
     (仅臂级,不改生产默认);
  3. **留证**:若仍自杀,在重启用 `py-spy dump --pid <worker>`(间隔 30s×5)
     抓卡点栈,落 `.s37/exec-stall-*.txt`,定位到具体编译/同步点后另行立项。

### 1.4 回滚

knob 默认值改动是一行 Python(env 缺省 128→256);回滚即 revert 该行。
缓冲容量随 env 走,无需回滚二进制。

---

## 2. A2/W4:push AR 出口融合(采纳 wp4 §2-§3)

- 设计与施工**照抄** `docs/design/z100/s34/z100-s34-wp4-ar-fusion.md` §2-§3:
  归约出口就地 add-rmsnorm(MLP 侧再加 fp8 quant),host 签名
  `all_reduce(..., residual=, norm_weight=, quant_out=)`,
  env `VLLM_DCU_CUSTOM_AR_FUSION=1` 默认关。
- S36 新增口径:AR(push 核)28.7µs × 125/步 ≈ 3.6ms/步(7.7% 设备时,
  32k profile;S36 设计图引的 69.4µs×128=8.9ms 是 10-29 晨另一 profile,
  两条口径都在,裁决时以 W4 开工当日的 profile 重测为准)。
- **位同门**:wp4 §4 的 T=1/T=4 双档 + `torch.equal`(融合核不改归约序;
  归约序变则按 max|Δ|≤1e-3 + top5 logprob 板裁决,与 wp5 同口径)。
- **触点**:`csrc/custom_all_reduce.cuh`(push 核第 4 步归并出口)、
  `vllm/distributed/device_communicators/custom_all_reduce.py`(签名+env,
  oneshot 门区)、模型层 communicator 封装;影响 `.so`:
  `_C_stable_libtorch.abi3.so`(备份 md5 见 STATUS §1.4 惯例)。
- **时点**:排在 B 轨 MTP 决议之后(MTP 改变步构成 → 融合收益重测),
  但 W4 自身的微测(per-op)与 e2e A/B 都在无 MTP 臂,两轨不互斥。

---

## 3. A3:split-K GEMV 复活评估(WP5 存档的修订重启)

### 3.1 与 WP5 存档的对账

- **保留**:two-kernel 方案(partial → 固定段序 reduce)、env 门
  `VLLM_DCU_FP8_GEMV_SPLITK`(默认 0)、M∈{1,3} 两档、数值裁决口径。
- **前提替换**:WP5 的立项前提是"S33-WP1 whole-A 变体 4-way bank 冲突
  结构性下限";S36 分解证明现役 stream_m1 的病灶不是 bank 冲突而是
  **低 block 数下的 DRAM 并行度**(320 块 ≈ 378-535 GB/s,2176 块 ≈ 640-660
  GB/s;barrier/restage 链合计仅 ~0.9µs),且 H1a/H1b(加深/加大)双双
  回归——**改变并行度是唯一没试过的杠杆**,这正是 split-K 的机制。
- **形状表替换**(WP5 表基于旧口径,以 S36 实测为准,in-graph packed 口径):
  | 形状 | K×N | 现状(µs) | bytes 下限 | 可回收 |
  |---|---|---|---|---|
  | o | 6144×1280 | 20.8 | ~12.3 | ~5 |
  | qkv | 5120×2048 | 22.6 | ~16.0 | ~3 |
  | gateup | 5120×8704 | ~69.7 | ~62(660GB/s) | ~3(边际) |
  | down | 17408×1280 | 41.6 | ~34(535GB/s) | ~2(边际) |
- **收益上限(诚实账)**:o+qkv 各 62 launch/步(S36 32k profile:248/步
  = 62 层 × qkv/o/gateup/down),o+qkv 回收 ~8µs → **e2e ≈ −0.3~−0.5
  ms/步(−1.2~−1.8%)**。这是"值得做但不是大鱼"的量级,故 kill-fast:
  可行性微测不过门即停,不进实现。

### 3.2 可行性微测(第一日,零生产风险)

问题:把 o/qkv 的 block 数翻倍/三倍,K 段分片后,单 block 带宽是否如
N-sweep 外推那样上升。**不改生产核**,用等价实验回答:
- 实验 1(并行度响应曲线):现役内核,固定 K,人为把 N 拼大(等价于
  block 数×2/×3 的负载),测 GB/s-vs-blocks 曲线,拟合 o 在 640/960 块
  处的可达带宽;
- 实验 2(K 分片无损性):partial-then-reduce 的数值代价预演——用两个
  半 K 的现役内核调用 + torch 求和,对拍全 K 单次调用,量 max|Δ| 分布
  (预期 1e-3 相对量级,给裁决口径定标尺);
- **门**:外推 o@640 块 ≥450 GB/s(即 17.5µs 级)才进实现;否则关闭
  WP5 存档并记录"并行度假设不成立"的负结果。

### 3.3 最小实现(仅当 3.2 过门)

- `fp8_emu_gemm.cu`:m1 核增加 splitk 模板参数(S=2 起步):每 block 扫
  K/S 段(slice 1024 几何不变——S36 已证它是局部最优),partial 写 fp32
  workspace;新增 reduce 核(固定 s=0..S-1 段序 + bias/out 写出)。
- dispatch:`VLLM_DCU_FP8_GEMV_SPLITK` env 门(默认 0),仅 M=1、
  N ≤ 2048、K ≥ 4096 生效(保护 gateup/down 不进此路径);
  workspace 在 op 内按 (S,N) 缓存(decode 单流,设计允许;多流风险记录)。
- **数值门(与 W2' 不同,必须写清)**:split-K 改累加序,**不可得
  torch.equal**;走"max|Δ| ≤ 1e-3 相对 + top5 logprob 偏移板"裁决
  (WP5 §4 原口径),并按 S29 惯例披露。
- **微测门(packed 口径)**:o ≤18µs、qkv ≤20µs;gateup/down 零回归
  (env 门隔离,理应不触)。
- **e2e 门**:无 MTP 臂 step p50 −0.3 ms 以上;否则 env 默认保持 0 存档。

---

## 4. B 轨:B2 —— MTP 重启裁决(dynk 修复)

### 4.1 先导 A/B(零代码)

- 臂 MTP+dynk0:启动器 `.s34/launch_mtp2.sh` 基础上 env
  `VLLM_S24_DYNK_LO=0`(dynk 整体关闭;其余与 A 轨基线一致,含
  `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`);
  对照臂 = 当前 MTP 部署口径(dynk on,draft/drafts≈1.02 的旧态)。
- 探针:7.5k(7469 doc/96tok/3rep)+ 60k + 128k(带 S36 的预热线);
  metrics 三条:`spec_decode_num_drafts_total`、`num_draft_tokens_total`、
  `accepted_tokens_per_pos`。
- **判据**:dynk0 臂 draft_tokens/drafts ≈ 2.0 且 pos1 验收 ≥0.6
  (S34-WP6 dynkoff 参考值 0.72-0.85);step p50 vs 无 MTP 基线在 60k/128k
  口径 −10% 以上 → MTP 重启"值得";否则 MTP 继续冻结,B 轨关档。

### 4.2 durable 修法(十行,独立于 4.1 的裁决)

- `vllm/v1/core/sched/scheduler.py:302-305`:
  `dynk_ema_good = min(5.0, 0.75 * num_spec_tokens)`(k=7 → 5.25→min 5.0
  行为不变;k=2 → 1.5);`dynk_mid_k = min(dynk_mid_k, num_spec_tokens)`
  (k=2 时 mid=2,消除 mid>k 的死分支)。
- 自测:scheduler 单测(dynk 表对 k∈{2,7} × ctx∈{16k,65k,128k} 的 k' 矩阵),
  证明 k=7 路径逐值不变。
- 该修法默认生效不改部署语义(k=7 生产行为不变),与 4.1 的 env 方案
  二选一或叠加(env 关闭优先)。

### 4.3 风险

- MTP verify 走 3D attn 段数路径:若 A1 把 SEG_MAX 默认改 256,MTP verify
  的段数同步变——B 轨探针须在 A1 裁决后的默认上跑,避免口径漂移;
- MTP verify 的 m1 GEMV(M=3)在 split-K 落地时需补 verify 档微测
  (S36 设计图 §6 风险项的延续)。

## 5. 验收门汇总

| WP | 微测门 | e2e 门(无 MTP 臂) | 数值门 |
|---|---|---|---|
| A1/W3 | — | 128k p50:ABAB 双侧 ≥3% 同向 | 零翻位(策略类,无数值面) |
| A2/W4 | 融合核 per-op −20%(wp4 原判据) | e2e −0.8 ms/round | T=1/T=4 位同,失败走 logprob 板 |
| A3/split-K | o ≤18µs、qkv ≤20µs(packed);gateup/down 零回归 | step p50 −0.3 ms | max|Δ|≤1e-3 相对 + top5 logprob 板(S29 披露) |
| B1/MTP | — | 60k/128k step p50 −10%(vs 无 MTP) | draft/drafts≈2.0、pos1≥0.6 |

## 6. 风险

- W3 的 32k 点继续不可判定(基线自身离散)→ 只按 16k/128k 裁决,32k 披露;
- executor 自杀若复现且 py-spy 无法定位 → 长 ctx 探针全部带 1200s 超时 +
  预热线兜底,不阻塞其它 WP;
- W4 融合核改归约序 → 位同门失败即 env 回退(wp4 设计默认关,风险可控);
- split-K 数值面全新(无位同)→ 若 logprob 板越界,直接关档,不上 e2e;
- MTP 重启与 A1/A2 的时序耦合 → B 轨探针一律在 A1 裁决后的默认配置上重跑
  一次口径校准(15 分钟级)。
