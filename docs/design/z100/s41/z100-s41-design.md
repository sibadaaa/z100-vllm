# S41 设计图:GEMV 第二轮(wholea 采纳 + 多流发射 + o/out 固定开销)

> 日期:2026-10-31。输入:S40 终局(`docs/result/z100-s40-final.md`,
> 1 采纳 2 关档)、S40 当日补充测量(`docs/perf/data/s40-{wholea-window,
> rows8-window,ceiling2-probe}.txt`)、S40 三连证伪(pf[4] 环深 −11~−20%、
> A-slice 双缓冲 −53%、宽 lane 连续 611)。**本轮问题:S40 把"怎么把
> GEMV 推向墙"的三条错误路径堵死之后,清单式审计(访存合并/向量化/
> TILE_V)指向的剩余正向空间还剩多少、怎么拿。**

---

## 0. 基线快照(2026-10-31,S40 采纳态,一切判据的对照面)

| 项 | 值 |
|---|---|
| 在役臂 | 无 MTP(`SPEC_OFF=1`,systemd `z100-vllm-mtp2`)+ S40 ADOPTED 段(`VLLM_DCU_FP16_GEMV_STREAM=1`);FP 工具 `270fbac5c3c1f953`(S40 开关未入 FP key 表,缺口见 s40-final §4) |
| `.so` | `_rocm_C.abi3.so` md5 `f72bb39d`(改前 `ca05f746`,备份 `.r3/so-backup/`) |
| 服务级(当日带) | W1 贪心 36.0–37.4 t/s;TTFT 0.73–0.83 s |
| decode 步构成(S39 账本 + S40 修正) | GEMV 12.30、**lm_head ~0.86(S40 −0.09)**、push-AR 3.76、attn 1.42、norm/碎核 ~2.6、GDN 1.00、bf16 小核 0.55、AG 0.19;GPU busy ~24 ms/step |
| GEMV 窗口基线(m1 rows4,`s40-a2-pf2.txt`) | gate_up **510**、attn_qkv 381、gdn_in_qkv 381、gdn_in_z 289、attn_o 271、gdn_out 270、down 417 |
| wholea 窗口(休眠门实测,`s40-wholea-window.txt`) | gate_up **569(+11.6%)**、qkv 424(+11.4%)、gdn_qkv 413、gdn_z 325、o/out 271/268(**0%**)、down 443(+6.4%) |
| rows=8 窗口(`s40-rows8-window.txt`) | 全线 +4~5%(K=5120 类),o/out 0% —— 被 wholea 支配,不单独立案 |
| 天花板阶梯(`s40-ceiling2-probe.txt`) | pin ~1024;triton 宽load 905;顺序 deep4 828;**K-E 轮转 deep4 783(GEMV 裸形最强)**;K-F +LUT 723;**K-J 宽lane连续×4 611(证伪)**;torch.sum 793 |
| S40 已证伪(不再重试) | pf[4] 环深(复用距离≠并发);A-slice 双缓冲(×2 竞态,×3 环 −53%);宽 lane 连续(905 模式不可移植);LDS 消除/位技巧(S39);o/out 的 wholea/rows 几何 |
| 测量纪律 | 窗口口径(30rep 单事件对)+ K-A 锚点前后复核(±3%)+ 服务 trace 双口径;非位同改动走 `max|Δ|≤3e-4` + S29 板(0.2255 nat)+ 贪心流记录 |
| 测量工具(全部现成) | `.s40/gemv_bw_bench.py`(窗口+lm_head)、`.s40/{a2_parity,b2_delta}.py`、`.s40/widen_tax.cu`(mode 0–11)、`.s40/parse_trace_s40.py`、`runs/specbench/e2e_ts.py`、`.s38/parity_probe.py` |

**排序与结构**

```
Track A(wholea 采纳,零施工,今天已过微测)
  A0  位同复核 dump(WA=1 vs M1=1,torch.equal)             0.5h
  A1  部署 ABAB(WA=1 ↔ 0)+ S29 板 + trace                  0.5d
Track B(fp8 多流发射,探针先行 → 内核化;位同构造优先)
  B0  探针:widen_tax 加 mode 12/13(group-of-4 背靠背 vs      0.5d
      wholea+kPf=4),对照 K-E 783;≥+6% 才内核化
  B1  内核化(赢者):m1 组发射 或 wholea kPf=4                0.5-1d
  B2  门:位同(torch.equal)+ 窗口 gate_up/qkv ≥+6%           0.5h
  B3  MS 内核跟随(视 B2,RW≤2,S40 A3 先例)                 视 B2
Track C(o/out 固定开销,time-box 0.5d,旋钮先行)
  C1  零施工:STREAM_ROWS=16 离线微测 o_out(≥+10% 才立案)   0.5h
  C2  过 → K 条件化几何规则(K<2048 → rows16/nt512)          0.5d
不施工(记录):K-J 宽连续(611 证伪)、pf 环深、barrier 减半、
合并权重分配(S39 已否,收益上限 828×0.923≈764 不改判 880 不可达)。
```

排序理由:**确定性优先**——A 是零施工 + 今天窗口已过 +6% 门,只欠服务级
ABAB 与位同 dump;B 是 S40 结论(多流≠环深)的内核化,位同构造优先
(组发射不改变 lane→chunk 映射与 per-lane k 序),探针不过即关;C 有
S39 A5 前科,旋钮先行、time-box 最紧。

---

## 1. Track A:wholea 采纳(目标:GEMV 桶 −0.9~1.1 ms/step)

### 1.1 现状与机理(全部实测)

- wholea(S33 WP1 建成,休眠至今):整行 A(K×2 B)+ 折算 scale 一次进
  LDS,**K 循环零 `__syncthreads()`**,W 环 kPf=3。位同于 m1 按构造成立
  (S33 判据 B:A 值同 bits、slice/chunk 遍历序、折算乘加序逐行一致)。
- S33 当日 per-rep 口径只测到 +2~4%(噪声带内)未采纳;S40 窗口口径重测
  **+8~13%**(K=5120 类)重新立案——窗口/trace 互证纪律(S39/S40)是
  这次能看清的原因。
- 机理:省掉每 slice 的 A 重暂存(2 KB LDS 写 + 2 barrier);K=5120 时
  5 slice × (stage+2 barrier) 的结构税就是 wholea 与 m1 的差。
- **o/out(K=1536)不受益(0%)**:预填 2 slice 已覆盖 1.5 slice,
  m1 在该 K 下本就近乎 barrier-free;其瓶颈是每 block 固定开销
  (Track C)。
- 路由事实(`fp8_emu_gemm.cu:3392`):`wholea_enabled() &&
  wholea_shape_ok(K, nkb)` **优先抢占** m1;7 形状全过 12 KB LDS 门
  (K=5120 → 10.4 KB)。SPLITK(休眠)被 WA 抢占,记录即可。

### 1.2 A0:位同复核(0.5h,kill-fast)

`a2_parity.py` 复用:同 seed 七形状,`VLLM_DCU_FP8_GEMV_WA=1`(抢占比
`M1=1`)vs `WA=0,M1=1` 两进程 dump,`torch.equal` 全绿才进 A1。失败 =
构造判据被打脸,先修再谈。

### 1.3 A1:部署 ABAB(0.5d)

- B 臂:`serve_deploy.sh` 增 `export VLLM_DCU_FP8_GEMV_WA=1`(S41 标注);
  A 臂 = 现役(WA 不设)。
- 服务 smoke + S29 板(应 0 错位——位同)+ trace(`parse_trace_s40.py`,
  GEMV 桶速率,内核名应切换为 `fp8_emu_gemv_wholea_m1_kernel`)+ e2e ABAB。
- **门**:trace GEMV 桶分类速率 **≥ +6%**(微测已 +8~13%,服务级折损
  预留);e2e 无回归(当日带内)。
- 回滚:WA 不设 = 现役逐位(位同已证)。

---

## 2. Track B:fp8 多流发射(目标:向 K-F 裸形 723 逼近,内核 ≥+6%)

### 2.1 机理与 S40 约束

- S40 A-close 的结论:可移植方向是**每 lane 多条独立 load 同时在飞**
  (deep4 探针 783),不是环深(pf[4] −11~−20%)、不是宽连续(K-J 611)。
- **位同构造(B-MULTI)**:组发射——以 4 slice 为组,组首背靠背发射本组
  4 个 W chunk(`c = lane + 64d`,d=0..3,与 K-E 探针同构),随后逐 slice
  stage A + barrier + MAC 消费第 d 寄存器。lane→chunk 映射与 per-lane k 序
  **完全不变** → 按构造位同(与 S40-A 的 pf[4] 同级,但发射模式忠实于
  探针——S40 的教训:内核必须复刻探针的发射时序,不是复刻它的环深)。
- 寄存器:4 uint4 = 16(lane)/行,与 pf[4] 同量;A0 预检沿用
  (编译无 spill + 微测不降速即过,MS=8 实例明确不开)。
- **B-WIDE(备选,非位同)**:lane 每 slice 2 uint4(32 B)——改变归约
  划分 → 数值门(max|Δ|≤3e-4 + S29 板,Track B/S40 先例 2.74e-4)。
  仅当 MULTI 探针不过而 WIDE 探针过才立案。

### 2.2 B0:探针(0.5d,kill-fast)

`widen_tax` 加两模式(复制 `.s40/widen_tax.cu` → `.s41/`,不覆盖):

- **mode 12 "K-M group4"**:K-E 同款 strided-deep4,但组间插入"A-stage
  模拟 barrier 对"(每 slice 2 KB LDS 写 + 2 `__syncthreads()`),检验
  barrier 是否吃掉组发射收益 —— 这是内核化的真实形态。
- **mode 13 "K-N wholea4"**:wholea 骨架(整行 A 一次进 LDS、零 barrier)
  + W 环 kPf=4(现 kPf=3)。

**门**:窗口 ≥ **830**(= K-E 783 × 1.06;低于此不内核化,组发射连探针
都过不了 +6% 就没有施工价值)。对照面:K-E 783(无 barrier 上限)、
K-F 723(含 LUT 下限)。

### 2.3 B1:内核化(0.5-1d)

- **K-M 赢**:`fp8_emu_gemv_stream_m1_kernel` 组发射改造(loop 改组结构、
  W 载荷 4×uint4、A/slice/barrier 原样),env `VLLM_DCU_FP8_GEMV_M1_G4=1`
  (默认 0)。
- **K-N 赢**:wholea `kPf` 3→4(env `VLLM_DCU_FP8_GEMV_WA4=1`,默认 0;
  寄存器 +2 uint4/行,A0 预检同 S40)。
- 两门可叠加 WA(Track A 先部署)——矩阵:A ∈ {WA=0/1} × B ∈ {off/G4/WA4},
  合并批先测总量、过门后逐项单开归因(S39/S40 惯例)。

### 2.4 B2/B3:门(0.5h + 视况)

- 位同:`torch.equal` 全绿(MULTI/G4 按构造;WA4 同)。
- 窗口:gate_up、attn_qkv ≥ **+6%**(对照 wholea 臂),其余形状 ≤2% 回归。
- MS 内核跟随仅当 B2 过且寄存器预检过(RW≤2;S40 A3 先例,RW≥4 永不开)。
- 不过 → 关档留证(env 归零)。

---

## 3. Track C:o/out 固定开销(time-box 0.5d,旋钮先行)

- 现状:271 vs 裸形 641(2.4×);wholea 0%、rows8 0% —— 已排除 barrier
  与几何,剩**每 block 固定开销**(LUT 512 B + scale + A stage + epilogue
  摊 6 KB 流量)与短 burst DRAM。
- C1(零施工,0.5h):`VLLM_DCU_FP8_GEMV_STREAM_ROWS=16` 离线窗口微测
  o/o_out(RW=2,nt=512,grid 320)——**≥+10% 才立案**,否则关。
- C2(过门才做,0.5d):`stream_block_select` 加 K 条件化规则
  (K<2048 → rows16/nt512),env `VLLM_DCU_FP8_GEMV_SMALLK16=1`(默认 0;
  S39 A5 的 SMALLK 教训 = 只动 K<2048 类,不碰 K=5120 类)。
- 门:窗口 o/out ≥+10% 且 K=5120 类 ≤2% 回归;位同(几何变化不改变
  lane 映射内的 k 序?**否——rows16 改变 warp→行归属,归约划分变,
  非位同**,走数值门,同 B-WIDE)。
- 不过 → 关档留证;o/out 维持现状(271,桶内占 ~0.6 ms,风险收益比差)。

---

## 4. Track D:测量基建(0.5h,先行)

- `widen_tax` 加 mode 12/13(§2.2);`a2_parity.py` 支持 WA 臂(env 由
  调用方设,脚本已无碍);其余工具全部沿用 S40。
- 一切判据以窗口口径为准,锚点前后复核,服务 trace 第二口径。

## 5. 预期终态与验收总表

| WP | 微测门(窗口) | trace 门 | e2e 门 | 数值门 |
|---|---|---|---|---|
| A wholea | 七形状已过(o/out 0% 记录) | GEMV 桶 ≥+6% | 无回归 | `torch.equal`(构造位同) |
| B 多流 | 探针 ≥830;内核 gate_up/qkv ≥+6% | ≥+6%(叠加 A) | 并入合并门 | `torch.equal`(构造位同) |
| C rows16 | o/out ≥+10%(探针) | — | 并入合并门 | max|Δ|≤3e-4 + S29 板 |
| D 工具 | (无门) | — | — | — |

**合计现实预期**:A −0.9~1.1 ms/step;B 再 −0.4~0.8(向 687 逼近);
C −0.2~0.4(存疑)。GEMV 桶 12.3 → **10.4–11.0 ms**(无 B/C 时 11.2–
11.4);W1(ABAB 配对口径)+4~6%。全部 WP env 门独立回滚;任一门不过 →
关档留证。完成判据 = 逐行落档 `docs/result/z100-s41-final.md`。

## 6. 风险

- **A 的服务级折损**:微测 +8~13% 是单核无干扰口径;服务上 GEMV 与
  push-AR/attention 交错,trace 门 +6% 留了折损余量;ABAB 当日带
  ±2 t/s,e2e 只做无回归判据,采纳以 trace 为主。
- **B 的 S40 幽灵**:pf[4] 证明"环深≠并发",本设计 B-MULTI 的组发射
  必须在探针里连 barrier 一起模拟(mode 12),内核化严格复刻探针时序;
  若 mode 12 <830 而 mode 13 ≥830,直接走 wholea 路线,不硬改 m1。
- **C 的 A5 前科**:rows 几何在 K=5120 类输过;C2 严格限定 K<2048,
  且先过数值门(归约划分变化)。
- **窗口口径漂移**:每轮微测前后 K-A 锚点 ±3% 守门(S40 纪律照抄)。
- profile 规程(S38 事故教训,强制):一发请求 + 立即 `/stop_profile`;
  异常楔死 → `sudo systemctl restart z100-vllm-mtp2`(~270 s,今日实测)。
