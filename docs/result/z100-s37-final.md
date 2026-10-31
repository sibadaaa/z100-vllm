# S37 终局报告:W3 收口 + W4 AR 融合(部分)+ split-K 评估 + B2 MTP 裁决 —— 2026-10-30

> 输入:`docs/design/z100/s37/z100-s37-design.md` 与施工图。执行环境:生产
> 无 MTP 服务(4×TP,Qwen3.8-27B-FP8,gfx906)。原始数据 `docs/perf/data/s37-*.txt`,
> 工作目录快照 `.s37/`(探针/重启工具、diff 台账)。

## 0. 结果一览

| WP | 判定 | 一句话 |
|---|---|---|
| A1/W3 收口 | **done(默认值改 256)** | ABAB 四实例:128k −6.8%(双侧 −7.1/−6.5%),无 ctx 回归;冒烟与 A 臂一致 |
| A1/executor 治理 | done | 预热线 + 臂级 1200s 超时全程零自杀;128k 首编译 488-518s 全部移出测量窗 |
| B1 dynk durable 修法 | **done(已落树)** | `dynk_ema_good=min(5.0,0.75k)`、`mid_k=min(3,k)`;单测 4/4(k=7 矩阵逐值不变) |
| A3/split-K | **关档(负结果,微测门未过)** | E1 并行度响应真实(640 块 483-488 GB/s);但两核结构(partial+reduce)吃掉收益:o 23.0µs(门 18)、qkv 28.2µs(门 20) |
| A2/W4 | **部分交付** | 融合内核+host+绑定+python+微测完成:residual 位同(d=0),per-op −12~−18%;模型层穿透与 e2e 未做(见 §4) |
| B2/MTP 重启裁决 | **GO(10-30 修正:重启获推荐)** | 修正 chunk≠token 口径后:真 token 吞吐 7.5k 53.7 / 60k 40.0 / 128k 30.8 tok/s,比无 MTP 基线 +29~42%;draft/drafts=2.0、验收 0.91。首测的"第二层病灶"不成立(见 §5 修正) |

生产状态:无 MTP 臂在役,`VLLM_GFX906_ATTN_DECODE_SEG_MAX` 默认 256(源码级,
无 env);臂级带 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200`。
`.so` 状态:`_rocm_C.abi3.so` md5 `ca05f746`(含 splitk 代码,env 默认 0 惰性,
S=0 路径与基线 `86470c5d` bit 同);`_C_stable_libtorch.abi3.so` md5 `7dcce4f2`
(S37 前基线,W4 融合核未部署)。备份:`.r3/so-backup/_rocm_C.s37-before.so`、
`_C_stable_libtorch.s37-before.so`。

## 1. A1/W3:SEG_MAX 默认值裁决(GO)

ABAB 协议(A=256,B=128;次序 A→B→A→B;每实例 重启→128k 预热→4 ctx×2 rep,
48 token 点;全部原始输出在 `s37-w3-abab-*.txt`):

| ctx | A 中位 | B 中位 | Δ |
|---|---|---|---|
| 16384 | 26.5 | 27.2 | −2.4% |
| 32768 | 29.0 | 29.8 | −2.7% |
| 65536 | 33.7 | 34.6 | −2.6% |
| 131072 | **41.8** | **44.9** | **−6.8%** |

- 128k 门(双侧 ≥3% 同向)达成:A1 41.8 vs B1 45.0(−7.1%)、A2 41.8 vs B2
  44.7(−6.5%);A 两实例 128k 读数完全一致(41.8/41.8)。
- 16k 佐证 −2.4%(同向,未到 3% 线——披露);32k 按设计不再追(基线离散);
  **无任何 ctx 回归**。
- 默认值改动:`triton_unified_attention.py` 常量缺省 `"128"`→`"256"`(一行+
  注释,diff `.s37/s37-a1-segmax-default.diff`);生产臂无 env 冒烟:16k 26.7 /
  128k 41.8 ms,与 A 臂一致(`s37-w3-default-smoke.txt`)。
- executor 自杀治理:预热线(health 200 后 128k warm 2-token)把首次 128k
  编译(488-518s)全部移出测量窗;臂级 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200`
  全程零自杀、零 60s×3 shm 警告复发。S36 事故路径未再现。

## 2. B1:dynk durable 修法(已落树)

- `vllm/v1/core/sched/scheduler.py`(~302):`dynk_ema_good=min(5.0, 0.75*
  num_spec_tokens)`、`dynk_mid_k=min(3, num_spec_tokens)`。k=7 逐值不变
  (5.0/3),k=2 得 1.5/2,消除 mid>k 死分支。diff `.s37/s37-b1-scheduler.diff`。
- 改前基线矩阵存档 `s37-b1-dynk-baseline.txt`(k=2 的 weak→3 死分支清晰可见);
  单测 `tests/v1/core/test_dynk_spec_len.py` 4/4 通过(k∈{2,7}×ctx×EMA 矩阵,
  k=7 行 == 改前基线)。
- **但 B2 裁决(§5)显示该修法对 MTP 吞吐无影响**——病灶不在 dynk 层(见下)。
  修法本身保留:它修正了阈值的量纲错误,是任何后续 MTP 工作的前提。

## 3. A3:split-K GEMV 复活评估(关档)

- **可行性微测过门**(`s37-splitk-feasibility.txt`):现役内核 N-sweep
  K=6144:N=2560(640 块)483-488 GB/s ≥ 450 门,外推 o@640 块 ≈16.1µs;
  分片数值预演 max|Δ|≈6e-05 ≤ 3e-4。并行度假设成立 → 进最小实现。
- **最小实现**(~200 行,diff `.s37/s37-a3-splitk.diff`):m1 内核加段界
  (`seg_len` + grid.y 驱动,`k0/k1/Kend` 段内扫描,slice/scale 绝对索引不变;
  plain 路径 S=1 与改前 bit 同——实测 torch.equal 一致);新增固定段序
  reduce 核;dispatch 门 `VLLM_DCU_FP8_GEMV_SPLITK`(默认 0;N≤2048、
  4096≤K≤8192、K%S==0);fp32 workspace 静态缓存。
- **微测门未过**(`s37-splitk-microbench.txt`,packed 口径):

  | 形状 | S=0(现役) | S=2 | 门 |
  |---|---|---|---|
  | o | 20.50 µs | **23.03 µs** | ≤18 未过 |
  | qkv | 22.71 µs | **28.16 µs** | ≤20 未过 |
  | gateup/down | 70.9/41.7 | 70.9/41.7(零回归,env 隔离)✓ | — |

- **负结果的机理**(对后续设计有用):E1 的并行度响应是在**每块全 K 深度**
  下测的;split-K 把每块 K 砍半后,两片预取流水线的跑道变短,per-block 带宽
  跌回 320 块水平(K=3072@320 块 ≈344 GB/s,S36 K-sweep 已见同一现象);
  再叠加 reduce 核 ~4µs 发射开销(o 的理论收益恰好 ~4.4µs,被全额抵消;
  qkv 还倒贴)。**单核内的并行度收益要用不增加核数的结构去取**(例如
  block 内跨 slice 拆 lane、或 L2 常驻 A 复用),split-K 两核结构在该内核上
  判死。env 默认 0 = 行为回现役路径,代码留档。

## 4. A2/W4:push AR 出口融合(部分交付,余下路径明确)

已交付(全部微测级验证):
- 内核 `cross_device_reduce_push_fused`(`csrc/custom_all_reduce.cuh`):
  push 五步不变、归约序不变(rank 0..3 fp32、对角直读),出口就位
  add-rmsnorm;**AR 和位同**——fp32 和先 round 回 fp16(与未融合路径的
  AR 结果写存再读的量化点一致)再加 residual。网格 = 每隐藏行一块
  (1024 线程,hidden 5120 → 640 packed 单元),行内 Σs² block 归约树。
- host `CustomAllreduce::allreduce_fused` + 绑定
  `_C_custom_ar.all_reduce_fused`(schema `Tensor! out, Tensor! residual`);
  python 链路:`custom_all_reduce.py::custom_all_reduce_fused`(env
  `VLLM_DCU_CUSTOM_AR_FUSION` 默认关 + 静态形状门,trace 稳定)→
  `cuda_communicator.all_reduce_fused` → `parallel_state` 注册 custom op
  `vllm.all_reduce_fused_rmsnorm`(mutates residual,带回退:
  融合不可用时 = 现役 AR + `fused_add_rms_norm`,语义等价)。
- 微测(`.s37/car_fused_bench.py`,4 rank,`s37-w4-fused-microbench.json`):

  | rows | 未融合对(AR+norm) | 融合 | ratio | max\|Δout\| | Δresidual |
  |---|---|---|---|---|---|
  | 1 | 50.5-51.1 µs | 41.4 | **0.81-0.82** | 2.4-4.9e-04 | **0(位同)** |
  | 4 | 45.9-46.1 | 39.1 | 0.85 | 4.9e-04 | 0 |
  | 8 | 44.4 | 38.6 | 0.87 | 4.9e-04 | 0 |
  | 12(MTP verify 档) | 44.6 | 39.1 | 0.88 | 4.9e-04 | 0 |

  per-op 绝对节省 5.5-10.1µs;×124 op/步 ≈ **0.7-1.2 ms/步** 的上限口径,
  与 wp4 预期带(−0.8~−1.4 ms/round)一致。严格 −20% 门仅 T=1 达到
  (−18~−19%),其余 −12~−15%——**per-op 门判"未全达",按施工图交 e2e 裁决**;
  e2e 才是验收门(−0.8 ms/round)。
- **未做**:模型层 `ar_epilogue` 穿透(Qwen3NextDecoderLayer → attn/GDN/
  mlp → RowParallelLinear → `all_reduce_fused_rmsnorm`,两条融合落点:
  attn 侧 post_attention_layernorm、MLP 侧下层 input_layernorm)与 e2e
  A/B。**部署状态**:融合核未进在役 `.so`(`_C_stable_libtorch` 保持基线
  md5 `7dcce4f2`;重建即得,`docker exec -e MAX_JOBS=8 z100-regress bash -lc
  'bash /mnt/game/port/build-sync2.sh _C_stable_libtorch'`——注意必须走
  build-sync2,它先刷新 hipify 暂存;裸 ninja 会编到陈旧副本,S37 已踩坑)。
  全部 diff 在 `.s37/s37-w4-{csrc,python}.diff`。

## 5. B2:MTP 重启裁决(10-30 修正:GO,重启获推荐)

> **修正(2026-10-30 晨)**:首测的探针把流式 chunk 数当 token 数。MTP 的
> 流式响应是**每 verify 步一个 chunk、每 chunk 携带 1+accept 个 token**
> (实测 96 tok 请求只到 32 个 chunk),因此 M 臂的"body_tps 21"实为
> ~21 chunk/s ≈ 59 token/s,per-chunk ITL 46-57ms 是**每 verify 步**的
> 耗时而非每 token——"第二层病灶"不成立,系测量口径错误。以下为修正后
> 的裁决(真 token 口径,`usage.completion_tokens`,数据
> `s37-b2-true-tps-MTP.txt`):

| 口径(tok/s,排除 TTFT) | 7.5k | 60k | 128k |
|---|---|---|---|
| 无 MTP 基线(base256,S37 实测) | 37.8 | 30.30 | 23.97 |
| MTP(现役 systemd 臂,10-30 实测) | **53.7** | **40.0** | **30.8** |
| Δ | **+42%** | **+32%** | **+29%** |

- 判据:draft/drafts = 220/110 = **2.0**(∈[1.9,2.1] ✓);验收
  accepted/drafted = 200/220 = **0.91**(S34-WP6 dynkoff 参考 0.72-0.85 ✓);
  真 token 吞吐 **+29~42%**(远超 +10% 门 ✓)→ **MTP 重启获推荐**。
- 原"step p50 −10%"判据是 k=7 时代口径(k≈accept,步与步可比);k=2 MTP
  每步成本高(~46-57ms vs 无 MTP 步 25-33ms)但每步产出 ~2.8 token,
  步级比较无意义,吞吐才是验收面——已在判据表中注明。
- dynk 两臂(M=on,D=env-off)在验收率高时行为一致(均全量 verify k=2),
  与预期相符;durable 修法保留(修正阈值量纲,k=7 逐值不变)。
- 运维注:主机 10-29 23:40 重启,systemd `z100-vllm-host.service` 自启即
  现役 MTP 臂(带 `--speculative-config mtp k=2`)——当前在役的就是修好
  后的正确配置,无需回切。10-29 深夜的 NO-GO 结论作废。

## 6. 台账

- diff:`.s37/s37-b1-scheduler.diff`(24 行)、`s37-a1-segmax-default.diff`
  (36 行)、`s37-a3-splitk.diff`(198 行)、`s37-w4-csrc.diff`(310 行)、
  `s37-w4-python.diff`(251 行)。
- `.so`:备份/回滚对齐 `.r3/so-backup/` 惯例(`_rocm_C.s37-before.so`、
  `_C_stable_libtorch.s37-before.so` + md5 文件)。
- 测试:`pytest tests/v1/core/test_dynk_spec_len.py` 4/4;W3 冒烟
  `s37-w3-default-smoke.txt`;A3 S=0 与改前 bit 同(torch.equal)。
- 工具沉淀:`.s37/`(ctx_sweep NTOK 参数化、prewarm128k、restart_arm /
  restart_mtp_arm、mtp_probe、abab_adjudicate、car_fused_bench、
  splitk_feasibility / splitk_microbench、w4_profile_table)。

## 7. 下轮建议

1. **W4 收尾**(半天):模型层 `ar_epilogue` 穿透 + e2e A/B
   (门 −0.8 ms/round;融合 `.so` 重建后单 env 臂即可)。
2. **MTP 重启落地**(已获推荐,人工拍板):现役 systemd 臂即为 MTP 配置,
   吞吐 +29~42%;如需把裁决口径固化,补跑一次 7469-doc 口径的真 token
   三 rep 探针即可(工具 `.s37/mtp_probe.py` 的 chunk 口径已废弃,以
   `usage.completion_tokens` 为准)。
3. W4 的 per-op −20% 门在 T≥4 未达:若 e2e 过门即采纳;若 e2e 也不足,
   融合的余量在把 norm 的 Σs² 与 AR 读 overlap(现实现是串行 pass)。
