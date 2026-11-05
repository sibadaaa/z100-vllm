# S49 最终报告:图模式投产收口(P0→P3)+ decode 各项速度(2026-11-03)

> 执行依据:`docs/design/z100/s49/z100-s49-design.md`(设计图)与
> `z100-s49-construction.md`(施工图)。事实基线:
> `runs/fulldec/graph_recalibration.md`、`runs/fulldec/acceptance.md` §6-8。
> 判定文件:`.s49/{a-adopt,b-close,c-close,d-close}.md`、
> `.s49/c2-inventory.md`;原始数据 `docs/perf/data/s49-*`。

## 0. 三十秒看懂

- **P0-A 全绿**:down_smul 回退(S49-A1)+ 图臂 A3 门 + conc4 + 10-boot 全过
  ——图臂 down 41.92 µs/531.5 GB/s(plain 水平),GEMV 桶 12.81 → **10.91
  ms/步**,TPOT(B)24.8-25.5 → **22.57 ms**,10/10 boot 0 fault。
- **P0-B 结案(出口 i)**:11-02 的"128k 回退是图模式固有"**被证伪**——
  无 MTP 图臂 128k 无塌方(25.6 tok/s 真口径);根因归因 = MTP verify 放大
  (AR 4.6×、M 行 GEMV +4.6 ms/步);GEMV/时钟/热三路线全部排除;
  attention @128k 17.4 ms/步 > 5 ms → **attention 立项**。
- **P1 关档(明确)**:大停顿 3.17 ms/步的宿主主体 = engine 采样/输出同步
  (py-spy:sample_tokens+get_output_async = 95.8%);C3a/b 量化上限
  <门限 15% 不立项,C3c 留立项材料。
- **P2 采纳**:AR 全融合在 FULL 图域**三门全过**(连贯/逐字节 parity/conc4
  0 fault)——S44 禁令限定为 piecewise 域;图臂配方可开 `_MLP=1`。
- **P3 交付**:E2 KNOBS 补 15 项 + selftest 42/42 全绿;E1/E3 README 段;
  decode_stages 支持 .gz。
- **生产未动**:全程结束恢复在役 A0 核臂(MTP k=2 + PIECEWISE),恢复后
  96-token 贪心与 S49 前参照**逐字节一致**。

## 1. 各轨结果

| 轨 | 门 | 结果 | 判定 |
|---|---|---|---|
| P0-A1 回退 | greedy parity 逐字节同 | 01:33 已验(`.s49/parity_*.txt`) | PASS |
| P0-A3 图臂门 | down ≤43 µs;桶 ≤11.5;其余五形状 ±3%;silu ≈64×5 µs;TPOT ≤23.6 | 41.92 µs/531 GB/s;10.908;±0.2%;63 发≈0.31 ms;B 22.57/A 23.28 | **PASS** |
| A3-bis | conc4 0 fault;10/10 boot | 3 轮全完成 0 fault;10/10 GREEN,TPOT A 22.99-23.40/B 22.45-22.68 | **PASS** |
| P0-B | 判据树叶子 + 出口三选一 | B-1 GEMV 排除→attention 立项;B-2 时钟排除;B-3 不触发;B-4 触发;出口 (i) | **结案** |
| P1 C1/C2 | 归因固化 + 清单 + 宿主定位 | 交付(`.s49/c2-inventory.md`) | 完成 |
| P1 C3 | 大停顿 ≤1.5 ms/步 | 关档(C3a/b 上限 <15%;C3c 需评审) | **关档(明确)** |
| P2 AR | 连贯 + parity + conc4 | 三门全过 | **采纳(图臂)** |
| P3-E | selftest 全绿 + README | 42/42;三段落档 | **交付** |
| DEP | 投产复验 | 留操作员决策(§4) | 未触发 |

臂指纹:图臂 FP `0050c4a7bc9a8f07`(SPEC_OFF=1 + FULL_DECODE_ONLY [1,2,4]);
生产臂 FP 见 `.s34/serve_deploy.sh` 原状(SILU_MUL=0,FUSION=1/_MLP=0,
.so md5 `6dee7fad`/`54cc2e2` 开工核对未变)。

## 2. decode 各项速度(本轮实测,两臂同工具同窗口型)

### 2.1 图臂(无 MTP,FULL_DECODE_ONLY [1,2,4];A3 trace,256 步,~110 tok ctx)

分段(decode_stages,profiler 口径,busy 17.65 ms/步):

| 分段 | ms/步 | %busy | 发/步 | 11-02 对照 |
|---|---:|---:|---:|---|
| **gemv wholea_m1(桶)** | **10.992** | 62.0% | 256 | 12.801(−1.81) |
| inductor_norm_act | 2.388 | 13.5% | 452 | 1.985 |
| ar(push_fused 1.290+push 0.995) | 2.286 | 12.9% | 129 | 3.166 |
| misc(含 GDN recurrent 0.546+conv 0.280) | 1.438 | 8.1% | 152 | 1.445 |
| attention_verify(3D 0.249+seg 0.262+cache 0.079) | 0.589 | 3.3% | 48 | 0.59 |
| aten_copy_fill | 0.030 | 0.2% | 5 | 0.030 |

wholea 逐形状(classify,中位 µs / GB/s):

| 形状 | µs | GB/s | vs 11-02 |
|---|---:|---:|---|
| attn_qkv 3584×5120 | 36.64 | 500.8 | 同值 |
| attn_o 5120×1536 | 23.84 | 329.9 | 同值 |
| gdn_in 2560×5120 | 36.48 | 359.3 | 同值 |
| gdn_out 5120×1536 | 24.64 | 319.2 | 同值 |
| gate_up 8704×5120 | 68.00 | 655.4 | 顶格保持 |
| **down 5120×4352(plain)** | **41.92** | **531.5** | smul 69.76/319.4 → **−40%,回 plain** |

host/idle(gap_attribution):步间 gap 0;大停顿 2.05 发/步 = **3.17 ms**
(prev=`postprocess_mamba_fused_kernel` → next=`FillFunctor<bool>`,
与修前逐项同);小间隙 16.5 发/步 = 2.78 ms。步首 prep GPU 仅 ~8 发/~40 µs。

TPOT(single_speed,3 连,真实 token 口径):**A(2419 tok)23.28 ms /
43.0 tok/s;B(99 tok)22.57 ms / 44.3 tok/s**(11-02:24.8-25.5 → −9~12%)。

### 2.2 在役生产臂(MTP k=2 + PIECEWISE;s49prod trace,111 步,~110 tok ctx)

| 分段 | ms/步 | %busy | 发/步 |
|---|---:|---:|---:|
| gemv M 行 stream 桶(verify 行) | 22.794 | 55.0% | 256 |
| ar(push_fused 9.415+push 2.933) | 12.348 | 29.8% | 129 |
| inductor_norm_act | 2.156 | 5.2% | 451 |
| misc | 1.313 | 3.2% | 57 |
| gdn_state_update(MTP) | 1.058 | 2.6% | 96 |
| aten_copy_fill | 1.013 | 2.4% | 152 |
| attention_verify(短 ctx) | 0.727 | 1.8% | 48 |

TPOT(single_speed):A **19.39 ms**(51.6 tok/s)、B **20.88 ms**(47.9
tok/s)——MTP ~2.3 token/chunk 口径已按 usage 归一(single_speed 分母用
completion_tokens)。

### 2.3 128k 稳态(90k 探针,ABAB 双臂 ×2 复现;详见 `.s49/b-close.md`)

| ctx | G 图臂 ITL ms(=真每 token) | P 生产 ITL ms(chunk,MTP ~1.9-2.3 tok/chunk) |
|---|---:|---:|
| 16k | 24.5 / 25.3 | 45.0 / 45.2 |
| 32k | 27.5 / 26.9 | 48.4 / 47.8 |
| 64k | 32.0 / 31.6 | 55.6 / 55.6 |
| 128k | **39.1 / 39.9** | **48.5 / 48.6** |

- G 臂随 ctx 平滑增长(+60%,attention 主导),**无塌方**;
- P 臂(MTP)真实 token 吞吐仍为两臂第一(≈39-47 tok/s @128k 估计);
- 128k 分段(G 臂):attention 3D **17.0 ms/步**(16 发,唯一 ctx 增长项)、
  wholea 11.06(短 ctx 10.99 → ctx 无关 ✓)、AR 2.31、GDN+conv 0.83、
  大停顿 3.53 ms;
- 时钟:两臂稳态 sclk **level 7(1600 MHz)**×4,64-70°C,~160-169 W
  (hy-smi)——热/功耗路线排除。

## 3. 运维事件(1 起,已恢复)

生产臂恢复后采 s49prod profile 窗,随后 single_speed 触发 worker 挂起
(60 s ×4 次 "No available shm broadcast block")→ engine RPC 超时自杀
(05:33:08)。与 S36/S37 "executor-RPC suicide" 同现象,两次事件均与
torch profiler 窗强相关;restart 5 min 恢复,parity 逐字节复验通过,
此后无复发。整改建议见 `docs/perf/data/s49-incident-prod-hang.txt`
(生产加 EXECUTE_MODEL_TIMEOUT=1200;profiler 窗后制度化 restart)。
**S49 全部测量数据在事件前完成,不受影响。**

## 4. 后续立项(不在 S49 内执行)

1. **attention @128k 立项**(P0-B B-1 叶子):unified_3d 17.0 ms/步 @90k,
   目标把 90-128k decode 的 attention 压回 SEG_MAX=256 调优/分块策略;
   附带核对 SEG_MAX 生效值(已核,=256)。
2. **MTP verify @128k 专项**(P0-B 出口 (i)):AR 行数门控 + M 行 GEMV
   平台复核,目标 MTP 图臂 128k 回到 PIECEWISE-MTP 同级。
3. **C3c 重叠立项**(P1):材料 = `.s49/c-close.md` + py-spy 两份
   speedscope + 两窗 gap json;先过 design 补充评审。
4. **S50 L2-smul 内核**:维持低优先(桶内一阶空间关账;桶 62% busy 的
   "≥28%" 触发线虽满足,但 §4 关账结论与新内核风险面不支持本轮立项)。
5. **图臂投产(DEP)**:无 MTP 图臂全门绿;若启用,图臂配方需同步
   `VLLM_DCU_CUSTOM_AR_FUSION_MLP=1`(P2 采纳);是否以短 ctx MTP 收益
   换长 ctx FULL 图一致性 = 操作员决策。

## 5. 产物清单

| 类 | 文件 |
|---|---|
| 判定 | `.s49/a-adopt.md`(A1+A3+A3-bis)、`b-close.md`、`c-close.md`、`d-close.md`、`c2-inventory.md` |
| 数据 | `docs/perf/data/s49-a3-{boot,tpot,trace}.txt`、`s49-b1-{G1,P1,G2,P2}.txt`、`s49-b2-{G,P}.txt`、`s49-d-run.txt`、`s49-e-hardening.diff`、`s49-e2-selftest.txt`、`s49-incident-prod-hang.txt` |
| JSON | `runs/fulldec/recal_s49a_{shapes,gaps}.json`、`recal_s49b{G,P}_{shapes,gaps}.json`、`recal_s49prod_shapes.json`、`s49a_stages.*`、`s49b{G,P}_stages.*`、`s49prod_stages.*`、`s49a3_profile_meta.json` |
| 稳定性 | `runs/fulldec/s49a_validation.tsv`(10 boot)、`s49a_conc4_*.log`、`s49a_*.log` |
| 工具 | `.s49/a3_profile.py`(前缀参数化)、`b2_profile.py`、`s49b_run.sh`、`s49d_run.sh`、`vm_a3_boot.sh`、`vm_a3_conc4.sh` |
| trace | `/mnt/game/s30c-prof/s49a3_rank0.*`、`s49b2G_rank0.*`、`s49b2P_rank0.*`、`s49prod_rank0.*`(.gz) |
| 观测硬化 | `tools/dcu/dcu_policy_fingerprint.py`(KNOBS +15,selftest 42/42)、`tools/dcu/README.md`(E1/E3 段)、`runs/tritonq/decode_stages.py`(.gz) |
| 现场还原 | `/etc/systemd/system/z100-vllm-mtp2.service.d/` 无 S49 drop-in;`.s34/serve_deploy.sh` md5 `75c2aab8` = S49-A1 回退后在役状态(A1 前为 `1551fff3`,备份 `.s49/serve_deploy.sh.s49-before`;P2 临时翻转已还原) |

## 6. 风险与注记

1. 本轮全部内核 `.so` 零改动(md5 开工/收工核对一致);代码变更仅观测
   工具三处(diff 存档)与 `.s49/`、`docs/` 新增。
2. ITL/chunk 口径:MTP 臂 ~2.3 token/chunk,ctx_sweep 的 ITL 表是 chunk
   口径;跨臂绝对吞吐比较必须按 usage 归一(single_speed 口径)。
3. profiler 窗与生产臂并存有 worker 挂起前科(§3)——下次生产采 profile
   按 incident 文件的两条整改执行。
4. b2-P 的流式读侧被中止(40 步 profiled 窗完整落盘后手工重分类),
   P 臂 128k stage 表来自该 40 步窗,样本量小于 G 臂(同为 40 步)。
