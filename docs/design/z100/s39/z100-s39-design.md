# S39 设计图:GEMV LDS 消除 + 集合通信归位 + 4 并发口径(W4 capture 探索)

> 日期:2026-10-31。输入:S38 第一轮收档(`.s38/a1-close.md`,W4 全融合
> capture 根因隔离、attn-only 位同 −0.63 ms 未采纳)、**LDS 冲突走树结论**
> (csrc 代码事实 + 带宽对账,见 §1.1)、带宽校准
> (`docs/perf/data/s38-bw-ceiling.txt`,**纯读 905 GB/s**,REPORT §8 口径修正)、
> S34 REPORT §3(decode 分环节)与 §8、S38 设计图(A2/A3 承接)。
> **与 S38 的关系**:S38 A1(W4 收尾)已收档未采纳;其两条实证结论决定了
> 本设计的取舍——(1) AR 出口融合的可压缩量只有 ~10 µs/op(自旋主导),
> 两出口全融合升级为 capture 根因探索(time-box);(2) GEMV 的 62–68% 天花板
> 占比缺口经走树定位为 **LDS LUT bank conflict**,是当前最大、机理最明确的
> 单项(44.8% 桶内 −1.5~3 ms)。

---

## 0. 基线快照(2026-10-31,全部判据的对照面)

| 项 | 值 |
|---|---|
| 在役臂 | 无 MTP(`SPEC_OFF=1`,systemd `z100-vllm-mtp2`),FP `2f3793deb97e138f` |
| 服务级(当日带) | W1 贪心 38.2–39.0 t/s,step ≈ 25.6–26.2 ms;W2 采样 35.9–37.6;TTFT 0.71–0.78 s |
| step 构成(GPU busy 24.2/26.7) | **GEMV 10.84(44.8%)**、AR 复合 ≈10.1 sum(push 8.16 自旋主导 + nccl 1.59 + AllGather 0.37)、norm/elem 2.86(含 dequant 0.65)、attn 1.55、GDN 0.89、bf16 0.52、host ≈2.2 |
| `.so` | `_rocm_C.abi3.so` md5 `ca05f746`(splitk 在库 env0);`_C_stable_libtorch.abi3.so` md5 `54ccf2e2`(W4 惰性在库,env 关=基线位同已验证);备份 `.r3/so-backup/` |
| **带宽口径(修正)** | **纯读 905 / D2D copy 829.4 / reduce 792.5 GB/s**(`s38-bw-ceiling.txt`);GEMV 分类:gate_up 615 / down 575 / qkv 524 / packed 477 / **o,out 309**;加权 561 = 纯读的 **62%** |
| W4 状态 | 全融合 capture 乱码(根因=编译/捕获编排,eager 干净,op 运行期验证正确);attn-only 位同 −0.63 ms 未采纳;bisect 开关在树(`FUSION_MLP/SKIP/DEBUG/EAGER`) |
| 测量工具 | `runs/s34nospec/{profile_window,parse_trace,gemv_bw_bench}.py`、`runs/specbench/e2e_ts.py`、`.s38/{car_fused_bench,bw_ceiling,parity_probe,fused_capture_check}.py`、位同语料 `.s36/parity/` |

**排序与结构**

```
Track A(GEMV 内核轨,_rocm_C;机理明确、最大桶 44.8%)
  A1  widen 税微基准(证伪 LDS 冲突假说)     0.5h kill-fast → 决定 A2/A3
  A2  LUT dword 化(swizzle)                位同,−0.3~0.6 ms
  A3  位技巧 widen(消 LUT)                 位同,累计 −1~2 ms
  A4  dequant 归并定位(fp8_dequant_to_fp16) −0.3~0.65 ms
  A5  K=1536 跑道(RW=2 → slice512)         o/out −0.7 ms
  A6  部署 + trace/e2e 复测(A 组合并门)
Track B(集合通信/宿主轨,_C_stable_libtorch)
  B1  nccl ring 回退归位(定位先行)          −1.0~1.6 ms
  B2  logits AllGather 定位/优化            −0.3 ms
Track C(服务级,零代码)
  C1  4 并发验证与口径固化                  聚合 ~×3.5,单流保护 ≤2%
Track D(探索,time-box 0.5d)
  D1  W4 全融合 capture 根因(影子校验/splitting_ops)
不施工(记录):AR 出口融合两出口版(待 D1)、MTP 重启(操作者拍板,与 4 并发
口径交互需单独测)、norm 深融合(二阶,待 A/B 落地后重测)、split-K(S37 判死)。
```

排序理由:**确定性优先**——A1 半小时证伪实验决定 A2/A3 的 1.5d 投入;GEMV
是最大桶且本轮已完成代码级定位(LDS 16-way 冲突 + 带宽对账自洽),三案全部
位同可验;B1 收益第二大(真实 kernel 时间,非自旋)且与 A 零耦合;C1 零代码、
先出容量口径;D1 time-box 防止再陷 capture 泥潭。A 内部顺序:A1→A2→A3→A4→A5
(A5 的 slice 几何与 A2/A3 正交,叠加后一次部署)。

---

## 1. Track A:GEMV LDS 消除(目标:加权 561 → 650–750 GB/s,GEMV −1.5~3 ms)

### 1.1 现状(走树结论,全部代码事实)

- 全局访问已优:`uint4` 全合并(1 KiB/warp 请求)+ slice 双缓冲预取
  (`fp8_emu_gemm.cu:603-604, 721-734`)——树的 F 节点无事可做。
- **LDS LUT**:`sl[256]` __half(512 B),`fp8_lut_u16` = 每权重字节一次 2B 读
  (`gfx906_emu_common.cuh:64-67`);W 路径每 (row, slice, lane) 16 次查表。
- **Bank conflict(可算)**:lane j 持字节 `[16j,16j+16)`,dword 索引 `8j+k`,
  bank `(8j+k)%32` → 64 lane 只碰 4 bank ≈ **16-way/指令**;带宽对账:gate_up
  2.68 GB/步 ÷ 60 CU ÷ 64 lane = 160M wave-inst/s/CU,C ≈ 10.6 —— 与静态
  分析同量级,**内核是 LDS-conflict-bound 而非 DRAM-bound**,精确解释
  62–68% 天花板。
- 生产 a16 路径与 `fp8_dequant_to_fp16`(98.9 µs × 256/步 = 0.65 ms)的输入
  归属 **未定位**(A4 第一步)。

### 1.2 A1:widen 税微基准(0.5h,kill-fast)

三个内核同形状同字节量(冷读轮转,复用 `gemv_bw_bench.py` 方法):
(a) 纯读 uint4(参照,应 ≈905);(b) 纯读 + `fp8x4_to_f16x2x2`×4 + 校验和丢弃
(= 现役 W 路径);(c) 纯读 + 位技巧 widen(1.4 的公式)+ 丢弃。
**门**:(b) 比 (a) 慢 ≥25% 且 (c) 比 (b) 快 ≥15% → LDS 假说证实,进 A2/A3;
否则假说证伪,A2/A3 关档留证(A5 不受影响,继续)。
产物:`docs/perf/data/s39-a1-widentax.{txt,json}`。

### 1.3 A2:LUT dword 化(0.5d,位同)

- `fp8_lut_u16` 表项改 **uint32-per-entry**(1 KiB 表):lane 读 4B,bank =
  `byte%32`,冲突 16-way → ~2-3 way。host 建表(`gfx906_fp8.py`)同步改。
- 影响面:`fp8x2_to_f16x2 / fp8x4_to_f16x2x2` 的全部调用点(stream/block/tiled
  GEMM/dequant),LDS +512 B/block(MS=8 时 16 KiB slice + 1 KiB 表,44 KiB 门内)。
- **门**:harness 中位 ≥2 形状 +8%;`torch.equal` 位同(值不变,仅读取方式变)。
- 不过门 → 该改法留档,直接跳 A3(dword 化只是 A3 的退路)。

### 1.4 A3:位技巧 widen(1d,位同,核心)

- 数学(e4m3fn,lossless 注释见 `gfx906_emu_common.cuh:55-63`):
  `h = ((b & 0x80) << 8) | ((b & 0x7f) << 7) | 0x2000`(normal 码精确);
  **subnormal(e=0,m≠0,14 码)**:真值 (1.mmm)×2⁻⁹,fp16 exp=6 → 修正
  `h −= 0x1000`(branchless:v_cmp + cndmask,+2-3 ALU/2B);**零码**:置 0;
  **NaN 码(0x7f/0xff)**:权重静态且量化后必有限 → host 加载端 clamp(留痕),
  内核不含 NaN 分支。
- e5m2 同法单独推导(其 subnormal 落在 fp16 subnormal,naive 公式不同,实现前
  先写 1 页推导 + 对拍 256 码全表(逐码 `torch.equal` 对照现 LUT)——**全表
  256 码位同是硬门**)。
- 集成:`fp8x4_to_f16x2x2` 换位技巧实现,env `VLLM_DCU_FP8_WIDEN_BIT=1` 门
  (默认 0 = LUT 现役);`sl[256]` 与 `fp8_lut_stage` 在该路径下不再使用。
- **门**:256 码全表 `torch.equal`;harness:gate_up/down ≥700 GB/s、加权
  ≥650;失败 = 实现错误(不是口径问题),修到位同为止。

### 1.5 A4:dequant 归并定位(0.5h 定位 + 视结果 0.5d)

- 定位:`fp8_dequant_to_fp16`(fp8_dequant.cuh:75)的调用点/输入形状/必要性;
  98.9 µs 对 10 KB 级 M=1 张量明显异常(疑似 grid 过小或 scale 同步)。
- 裁决:若为线性层前激活宽展 → 并入 stream kernel prologue(a16=0 路径,A 侧
  查表已按块摊销)或直接 a16=1 喂 fp16 → 消 kernel;若为同步伪影 → 单列。
- **门**:dequant kernel trace 计数 → 0(或 µs 降至 <20);并入 A 组 e2e 门。

### 1.6 A5:K=1536 跑道(0.5h + 1d)

- 承 S38 §4(B1-a RW=2 双行 → B1-b slice512 → B1-c),**与 A2/A3 叠加**
  (o/out 同时吃跑道修正与 LDS 消除两份收益;先 A 后 A5,归因分开)。
- **门**:o/out harness −20%(对照 rows4 基线);trace o/out ≥450 GB/s;
  e2e −0.3 ms;数值:RW=2 `torch.equal`、slice512 `max|Δ| ≤3e-4`。

### 1.7 A6:部署与验收

- 采纳序:A2/A3 位同过 → 合并部署一臂 → e2e ABAB(B→A→B→A,3rep,
  `e2e_ts.py --tag s39-a*`)+ trace 复测(分类速率:gate_up/down ≥700、加权
  ≥650、o/out ≥450)→ **A 组合并门:step p50 −0.8 ms**;过门逐项单开复核
  (A2、A3、A5 各一臂)归因。
- 回滚:env 门归零 = 现役路径;`.so` 恢复备份 + md5(见施工图 §0)。

---

## 2. Track B:集合通信归位(−1.3~1.9 ms)

### 2.1 B1:nccl ring 回退归位(1d,定位先行)——承 S38 §2,证据更新

- 现状不变:3.3 次/step 走 nccl(481 µs,共 1.59 ms;GDN core 65 + Compiled
  FxGraph 48 + fp8_mm 8 + python 7,窗口总量);**这是真实 kernel 时间,非
  自旋**——AR 复合桶内确定性最高的一项。
- 定位(0.5d,kill-fast):csrc `custom_all_reduce.cuh/cpp` nccl 回退分支加
  一次性日志(env `VLLM_DCU_CUSTOM_AR_FALLBACK_LOG=1`,rank0,每 (op,bytes,
  capture) 首见一行)→ 1 个 profile 窗口裁决 H1(图捕获回退)/H2(尺寸门)。
- 修法按位点:H1 → 把 AR 移出编译区(核对 `vllm.all_reduce` 为何未分裂)或
  push-AR capture-safe 化(仅当 finegrained flag 合法);H2 → 单独评估 push/拆分。
- **门**:trace `ncclDevKernel_AllReduce_Ring` decode 计数 = 0;e2e −0.5 ms;
  贪心 parity ≤S29 界(0.2255 nat);冷启动全捕获形状 [1,2,4] smoke 零 fail。
- 回滚:行为改动 env `VLLM_DCU_CUSTOM_AR_NO_NCCL=1`(默认现役)。
- **注意**:本 WP 改 `custom_all_reduce.cuh` → 重编 `_C_stable_libtorch`
  (build-sync2,§0);与 Track A 的 `_rocm_C` 互不影响。

### 2.2 B2:logits AllGather 定位(0.5d)——承 S38 §3

- 365 µs × 1/step 的 AllGather(疑似 logits 交换)+ 未见 lm_head 大核的矛盾;
  读 `compute_logits`(M=1 分支)+ shard 头 → `record_shapes` 单窗口定尺寸。
- 三分支:已有快路径 → 闭账不改码;小消息(≤256 KB)→ custom one-shot
  allgather(≤50 µs 可达);124 MB 级 → 字节账修正 + reduce-scatter 方案单列。
- **门**:AllGather ≤100 µs;e2e −0.2 ms(或不改码仅闭账)。

---

## 3. Track C:4 并发验证与口径固化(0.5h,零代码)

- 操作者口径:产流量允许 4 并发,单流速度不得明显受损。
- 协议:(a) solo M=1 基线(现役,已有多组);(b) 4 路 7469-probe 并发
  (96 tok,`e2e_ts.py` 改并发或 4 进程),记聚合 tps / 每流 ITL / per-token
  两桶成本;(c) solo 回归复测。
- **门**:聚合 ≥×3(≈115 t/s);每流 ITL ≤ +15%(vs solo);solo 回归 ≤2%。
- 产物:`runs/s38/../s39-c1-concurrency.json` + 容量口径写入 DEPLOY 备注
  (M=1 永远是现役路径,4 并发只是上限,continuous batching 天然保护单流)。

---

## 4. Track D:W4 全融合 capture 根因(0.5d,time-box)

- 已知:op/接线/数学全对(eager 连贯、op 运行期对照 d_out~1e-4、公式对拍 Δ=0);
  乱码仅出现在 piecewise 编译+捕获的两出口编排。
- 三个探针(按序,任一命中即停):
  (a) `enforce_eager` 全量 W1 parity(定量 eager 惩罚,决定"eager 采纳"是否
  可行——若 eager 惩罚 <0.5 ms,eager 采纳即解锁全融合收益);
  (b) `vllm.all_reduce_fused_rmsnorm` 加入 `splitting_ops` → 两出口 op 出图
  复测(规避捕获,直接验证图编排假说);
  (c) 影子校验:capture 重放路径加一次性旁路对照(`FUSION_DEBUG` 思路移植到
  replay)。
- **门**:位同(piecewise 下)→ 补 ABAB(门 −0.5 ms);time-box 到 0.5d,
  未破 → 维持 attn-only 存档,记录探针证据。

---

## 5. 预期终态与验收总表

| WP | 微测门 | trace 门 | e2e 门 | 数值门 |
|---|---|---|---|---|
| A1 | (b) vs (a) ≥25% 慢、(c) vs (b) ≥15% 快 | — | — | — |
| A2 dword 表 | ≥2 形状 +8% | — | 并入 A 门 | `torch.equal` |
| A3 位技巧 | gate_up/down ≥700、加权 ≥650 | 分类速率全面上升 | A 组合并 **−0.8 ms** | 256 码全表 `torch.equal` |
| A4 dequant | — | dequant 计数→0 或 <20 µs | 并入 A 门 | 位同 |
| A5 K=1536 | o/out −20% | o/out ≥450 GB/s | −0.3 ms | 位同(a)/3e-4(b) |
| B1 nccl | car harness 对拍 | `AllReduce_Ring`=0 | −0.5 ms | parity ≤0.2255 nat |
| B2 AG | — | AllGather ≤100 µs | −0.2 或闭账 | — |
| C1 并发 | — | — | 聚合 ≥×3;solo ≤2% | — |
| D1 capture | — | — | 位同后 −0.5 ms | parity S29 |
| **合计** | 单流 step 26.7 → **~23–24.5 ms**(GEMV −1.5~3 + B −1.3~1.9);GEMV 加权 561→**650–750**;4 并发聚合口径 ~×3.5,单流保护 ≤2% |

全部 WP 独立回滚(env 或单文件 diff);任何一门不过 → 该 WP 关档留证。
完成判据 = 逐行落档 `docs/result/z100-s39-final.md`。

## 6. 风险

- A3 位技巧是数值面改动:**位同按构造成立**(trick+修正与 LWT 给出同一实数),
  若 `torch.equal` 失败一律按实现 bug 处理,不启用 3e-4 退路;subnormal/NaN
  推导先出 1 页纸 + 256 码全表对拍再进内核。
- A2/A3 只解 LDS 税;o/out 的跑道问题(K=1536 半 slice)仍需 A5——两者叠加
  才能达到 o/out ≥450,归因必须分开做。
- B1 的 H1(图捕获回退)修法涉及编译区边界 → 冷启动/全形状 smoke 强制;
  push-AR capture-safe 化若卷入主机同步则放弃(维持 nccl,记录豁免)。
- C1 的 ITL +10~15% 是带宽受限 decode 的物理代价,写入容量口径而非缺陷。
- D1 time-box:本轮 W4 已花两轮,capture 泥潭不再续期;探针证据留档即可。
- profile 规程(S38 事故教训,强制):一发请求 + 立即 `/stop_profile`;异常
  楔死 → `sudo systemctl restart z100-vllm-mtp2`(~60 s,编译缓存热)。
