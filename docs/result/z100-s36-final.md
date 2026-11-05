# S36 终局报告:A 轨(W2'/W3)+ B 轨(MTP 调查)—— 2026-10-29

> 输入:`docs/design/z100/s36/z100-s36-design.md` 与施工图。执行环境:生产
> 无 MTP 服务(4×TP,Qwen3.8-27B-FP8,gfx906),`.so` 基线 md5 `86470c5d`
> (备份 `.r3/so-backup/_rocm_C.s36-before.so`)。所有原始数据在
> `docs/perf/data/s36-*.txt`,工作目录快照 `.s36/`。

## 0. 结果一览

| WP | 判定 | 一句话 |
|---|---|---|
| W2'-分解 | **done(高价值)** | F≈3.5µs 截距 + 每 slice 超额;in-graph 无间隙;H2 拒绝;复活 split-K 条件 |
| W2'-H1a(pf[2]→pf[3]) | **回归,已回滚** | 位同 PASS;o in-graph +14%,gateup 破 570 门 |
| W2'-H1b(m1 slice 2048) | **回归,已回滚** | 位同 PASS;全线 +3~12%,gateup 515 GB/s |
| W3-零步 | done | decode=3D 核实;段数=128(顶格);reduce 仅 0.9% 步时 |
| W3-扫描 | **done(正向)** | S=32/64 在 64k 回归 +10%;**SEG_MAX=256:16k −4.6%/32k −6.1%/128k 42.7ms( −10% vs 破损参考)** |
| W3-策略 | done(交付 knob) | `VLLM_GFX906_ATTN_DECODE_SEG_MAX`(默认 128 不变,行为零变更) |
| W4(AR 融合) | **延后** | 见 §5 |
| B1(MTP 调查) | **根因定位** | dynk 阈值按 k=7 标定,MTP k=2 必然"weak"→ctx>64k 时 k'=1 |

生产状态:服务已恢复基线配置在役(无 MTP、默认段数策略、基线 .so);
源码树仅保留 W3 的 env-knob 补丁(默认行为零变更)。

## 1. W2':F 分解与两条修法(均被门拦截)

**分解结论(详见 `docs/perf/data/s36-f-decomp.txt`)**:o 形状 in-graph
20.8µs/kernel = bytes 下限 ~10.5µs(745GB/s 口径)+ 截距 ~3.5µs + 每 slice
超额 0.7-1.1µs×6。三个关键事实:
1. **packed 图内间隙 ≈ 0**(profiler p50 20.80µs ≈ event 20.76µs)——生产
   piecewise cudagraph 不支付单 kernel/graph 口径里的 ~7µs graph 间隙;
   **W2' 微测门今后必须按 packed 口径评**(fill_model 旧口径含间隙,核内改进
   永远到不了 "o≤20µs")。
2. **H2 拒绝**:rows/nt A/B(rows=8,grid 减半)o 仅 −0.5µs,qkv/gateup 反向。
3. **m1 vs wholea**:barrier+restage 链合计仅 ~0.9µs(5 slice)——每 slice
   超额的主项不是同步结构,是低 block 数(320-512)下的 DRAM 并行度。

**H1a(施工图 §1.2,pf[2]→pf[3])**:位同 120/120 PASS、VGPR 44/零 spill,
但 o in-graph 20.8→23.7µs(+14%)、down +19%、gateup 537GB/s(<570 门)→回滚。
**H1b(§1.3 变体,m1 核 kM1Slice=2048,保序映射)**:位同 120/120 PASS,
但 o +6.6%、qkv +10%、gateup 515GB/s →回滚(patch 存 `.s36/w2p-h1b-m1.patch`)。

**裁决**:1024-slice/depth-2 是该访问模式的局部最优;剩余 ~5µs/kernel 超额
属结构项,**满足设计图 §2.3 的 split-K 复活条件**("若 W2'-分解发现 F 实为
可摊薄结构,重新评估")——建议下轮 WP 用新证据重启 S34-WP5 的 split-K GEMV
评估(仅 o/qkv 类小 N 形状;目标把 o 压向 bytes 下限 ~12-13µs)。

## 2. W3:段数扫描与策略 knob

- 零步(路由):decode 走 `kernel_unified_attention_3d`,8k/32k 段数=128(顶格),
  `reduce_segments` 仅 0.58-0.9% 步时——**归并不是瓶颈,并行度才是**
  (batch-1 decode grid=(1,1,SEGS):段数=块数,128 块 ≈ 2 块/CU)。
- 扫描(段数 env,16 token/点,同口径 A/A;base 两实例做热态配对):
  - S=32:16k +0.7% / 32k +1.9% / **64k +11.0%**
  - S=64:16k +2.1% / 32k −0.3% / **64k +10.4%** → <128 方向在 64k 明确回归;
  - **SEG_MAX=256(adaptive 段数升到 256)** vs 128 段(base-aa/base2 双基):
    16k 26.8 vs 28.1/28.0(**−4.4%**,双基高度可复现)/ 32k 29.3 vs 31.2/29.7
    (基线自身离散 ~5%,不可判定)/ 64k 34.5 vs 34.7/35.0(持平)/
    128k **42.7 vs 45.1(−5.3%,双侧均 15/15 token 干净采样)**。
- 交付:`VLLM_GFX906_ATTN_DECODE_SEG_MAX`(triton_unified_attention.py,
  默认 128 = 行为零变更;缓冲随 capacity 自动放大,256 档显存增量 ≈ +17MB/卡)。
- **注意**:跨实例 A/A 漂移在 32k 点高达 ~5%(16k/64k ≤0.4%);16k −4.4% 与
  128k −5.3% 均为双侧干净采样,方向一致可信;**W3 门(128k −10%)未达,但
  256 档无任何 ctx 回退且 128k −5.3%、16k −4.4%**。建议下轮同实例内做
  128↔256 交错切换(48 token/点)后把 `VLLM_GFX906_ATTN_DECODE_SEG_MAX`
  默认值改 256。
- 事故记录:base 实例(连跑 5h)在 128k 首测时 executor RPC 超时自杀
  (shm 60s×3 → TimeoutError);新实例复测未再现。疑与长_ctx 首次 Triton
  特化编译 >3min 有关,**建议下轮给 128k 加预热步骤并调查 executor 超时**。

## 3. B1:MTP draft 计数异常(只读,已定位)

**根因是 S24 dynamic-k 与 MTP k=2 的阈值错配,不是 proposer 起草分支**:
`scheduler.py:302` `dynk_ema_good=5.0` 按当年 k=7 草稿塔标定(注释自证);
k=2 的 acceptance EMA 上限 2.0,**永远判为 weak** → ctx>65536 时 verify 截到
k'=1 → draft_tokens/drafts≈1.0、pos1 验收≈0。S34-WP6 dynkoff 臂同口径数据
(mean acceptance 2.56,pos1 0.72-0.85)完整对账。详见
`docs/perf/data/s36-b1-dynk.txt`。**影响**:在役 MTP 臂在长 ctx(>64k,生产
主力)实际按 k=1 跑,S34 长 ctx MTP 收益口径失真。
**修复建议**:a) MTP 臂 env `VLLM_S24_DYNK_LO=0`(零代码,立即生效);
b) durable:`dynk_ema_good` 随 num_spec_tokens 缩放(如 0.7×k)。B2 重启裁决
建议先 a 后 b,并重测 60k/128k 的 MTP 收益。

## 4. 基线与口径备注

- 微测基线(基线 .so):o 25.8-27.8 / qkv 27.6-29.0 / gateup 76.6-77.2
  (581GB/s)/ down 46.7-48.4 / syn4352 44.4 µs;M=4 全形状见
  `s36-w2p-baseline-gemvbench.txt`;位同语料 120 张量(`.s36/parity/`)。
- 本次全程遵守:改码前 .so 备份+md5、位同硬门、编译后 VGPR/spill 复核、
  服务重启前 in-flight 检查、每臂 env 指纹核验、失败回滚并复验 md5。

## 5. W4(push AR 出口融合)延后说明

AR 现状:push 核 28.7µs×125/步(7.7% 设备时,32k profile),wp4 设计已交付。
延后理由:① W2' 两条修法均回归,步构成与设计图预设不同,融合落点收益需在
"split-K 决议之后"重测(设计图 §3 自己的要求);② 本轮窗口被 128k 服务
事故与 A/A 漂移复核占满;③ W4 是 1-2 天量级的 csrc 工程,不宜在窗口尾部
开建。下轮入口:`docs/design/z100/s34/z100-s34-wp4-ar-fusion.md` §2-§3。

## 6. 下轮建议(按优先级)

1. W3 收口:同实例 128↔256 切换复核(48 token/点)+ 干净 128k 基线 →
   决定 `VLLM_GFX906_ATTN_DECODE_SEG_MAX` 默认值(数据倾向 256);
   排查 executor 60s×3 自杀路径,给 128k 加预热。
2. split-K GEMV 复活评估(§1 裁决):先做 o/qkv 形状的可行性微测(无需改
   生产核),对齐 S34-WP5 存档设计。
3. B2:MTP 重启裁决(先 `VLLM_S24_DYNK_LO=0`),重测 60k/128k MTP 收益,
   顺带落 b) 修法。
4. W4:按 wp4 施工图执行,收益口径按新步构成重测。
