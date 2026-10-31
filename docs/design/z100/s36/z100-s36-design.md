# S36 设计图:内核优化轨(A 轨,无 MTP 基线)+ MTP 独立轨(B 轨)

> 日期:2026-10-29。输入:S34 终局、当日三测试(profile + 微测 + ctx 扫描)、
> 占用率实证(VGPR 40/6 波/无 spill)与 F 模型(T = F + bytes/727GB/s,5/5 形状验证)。
> **与 S34-WP5(split-K GEMV 设计)的关系**:F 模型将其收益上界修正为
> ~0.2-0.4 ms/step(见 §2.3),本设计以 W2'(F 削减)替代其优先级,
> WP5 设计保留存档,若 W2'-分解推翻本模型再复活。

---

## 0. 双轨结构

```
Track A(内核轨,无 MTP 基线,本设计主体)
  A1  W2'  m1 核 F 分解与削减        −2~−4 ms/step   (csrc 重编译)
  A2  W3   decode 长上下文分段调参    128k −10% step  (零/极少代码)
  A3  W4   push AR 出口融合 norm      −0.8~−1.4 ms    (采纳 s34-wp4 设计)
Track B(MTP 轨,独立推进,不与 A 混测)
  B1  draft 计数核查(draft_tokens/drafts≈1.02 异常)
  B2  MTP 重启决策(依赖 B1,独立裁决)
```

- **测量隔离**:A 轨全部在无 MTP 臂上同口径 A/B(基线:7469-doc 探针
  26.7-28.6 ms/步);B 轨启用后才引入 MTP 口径。两轨不得交叉归因。
- **基线快照**(2026-10-29):无 MTP ITL 28.3/28.3/31.1 ms @2k/16k/32k;
  fp8 GEMV 11.5 ms/步(43%),push AR 8.9 ms/步(26%),GDN 1.9,attention ~1。

---

## 1. A1/W2':m1 核 F 分解与削减

### 1.1 问题定义

生产 decode 的 fp8 GEMV 走 `fp8_emu_gemv_stream_m1_kernel<1>`(VGPR 40、
零 spill、理论 6/8 waves/SIMD)。实测五形状符合:

```
T = F + slice_bytes / 745 GB/s      F ≈ 15 µs/launch(仅末 cohort 支付)
```

稳态速率 ≈ copy 屋顶 93% —— **主循环无大病灶**;每次启动的固定开销 F
在 o/qkv 形状占 55%,全步 256 次 × F ≈ 3.8 ms(14% 步时)。

### 1.2 F 分解(第一步,决定后续路线)

四段假设与判别方法(rocprofv2 时间线 + 单 kernel trace):

| # | 假设 | 判别特征 | 对应修法 |
|---|---|---|---|
| H1 | 首两 slice 预取延迟暴露(~2×600ns×系数) | 时间线前 2 slice 段空泡 | `kStreamSlice` 1024→2048 |
| H2 | 启动 ramp(320-2176 block 波次铺开) | 前段 SM 占用爬坡 | `nt=512` + `__launch_bounds__` 钳制 |
| H3 | LUT/scale 预置串行段 | 核头部固定段 | LUT 移 `__constant__`;scale 并入首 slice |
| H4 | 冷 TLB/L2(HBM 首触) | 与缓冲旋转强相关 | 预热不可行;改持久权重布局(远期) |

### 1.3 修法设计(H1 主线,其余按分解结果)

- **H1 修法**:`kStreamSlice` 1024→2048(编译期常量,fp8_emu_gemm.cu:625)。
  效应:slice 数减半(o 6→3)、每 warp 在途 2KB、barrier 减半;
  代价:LDS 2.6→5KB(仍 <8% 闸)、尾部 slice 粒度变粗(K=6144 → 3 整片)。
  m1 与 stream_block 两变体共享常量,MS>1 路径需同步验证(LDS 门 44KiB 内)。
- **H2 修法**:launch_gemv_stream_m1 的 nt 由 rows 推导改为显式 512 +
  `__launch_bounds__(512, 3)`(3 block/CU × 8 warp = 24 warp/CU,恰为
  6 波/SIMD 上限);rows 语义不变(warp 持行映射不变)。
- **H3 修法**:`fp8_lut_stage` 改读 `__constant__` 256×half(启动时一次性
  hipMemcpyToSymbol);scale fold 保留(每 block 1.3KB,廉价)。
- **不动**:行-波映射(rows=4/nt=256 的 warp 容量不变量)、累加序
  (位同硬门)、w8a16 tiled 路径(cfg 体系不动)。

### 1.4 位同策略

F 削减不改任何累加序(slice 内 dot2 链、cross-lane 归约、scale 折算点全部
不动)→ 目标 **bit 级位同**;`torch.equal` 硬门,失败即回滚。

---

## 2. A2/W3:decode 长上下文分段调参

### 2.1 现状(代码事实)

- 分段管线**已在**:3D 核 + `reduce_segments` + 后端缓冲
  (triton_attn.py:161 按 `gfx906_decode_segments_capacity()` 分配);
- 容量策略已按 ctx 自适应(triton_unified_attention.py:285-306):
  short(<2048)=16 段;长 ctx 目标 grid 1024→4096,段数 = ceil(target/base_grid),
  上限 128;
- 生产短 ctx profile 显示 decode 已走 `kernel_unified_attention_3d` ✓;
- R1 墙:128k +15.7ms/步,KV 带宽只解释 ~3ms → 疑点在**段数×归并开销**
  与 TILE/段数配比,而非缺并行机制。

### 2.2 设计:三步测量驱动调参(零/极少代码)

1. **路由确认**:ctx ∈ {8k,32k,128k} 各采一份 profile,核对 decode 实际
   走 3D 还是 2D、段数是多少(内核 grid 第三维可从 trace 读出);
2. **段数扫描**:`VLLM_GFX906_ATTN_DECODE_SEGMENTS` ∈ {32,64,128,256}
   × ctx ∈ {16k,32k,64k,128k}(前缀缓存预热口径,`/tmp/ctx_sweep.py`);
3. **策略修订**:按扫描结果改 `gfx906_decode_segments_capacity()` 的
   target_grid_size/上限(若 batch-1 长 ctx 需 >128 段或不同 TILE 配比)。

### 2.3 与 S34-WP5(split-K GEMV)的对账

WP5 的前提"S33-WP1 4-way bank 冲突结构性下限"针对 whole-A 旧变体;
现役 stream_m1 稳态 745 GB/s(96% 屋顶)无此病灶。F 模型下 split-K
对 F 主导形状(o/qkv)收益上界 ≈ F 摊薄差,全步 <0.4 ms —— **不立项**;
若 W2'-分解发现 F 实为可摊薄结构(如 ramp),重新评估。

---

## 3. A3/W4:push AR 出口融合(采纳 s34-wp4 设计)

- 设计与施工图**已交付**(docs/design/z100/s34/z100-s34-wp4-ar-fusion.md):
  归约出口就地 add-rmsnorm(MLP 侧再加 fp8 quant),归约序不变,
  host 签名 `all_reduce(..., residual=, norm_weight=, quant_out=)`,
  env `VLLM_DCU_CUSTOM_AR_FUSION=1` 默认关。
- 本设计新增证据:AR 现 8.9 ms/步(26% 设备时间,69.4µs×128),
  融合收益口径从"空隙 3.8ms"上调为"空隙 + norm 小核 5-22µs × 128"。
- A3 排在 A1/A2 之后:W2'/W3 改变步构成后,融合落点的收益需重测
  (S34-WP3 教训:栈变了边际价值会变)。

---

## 4. Track B:MTP(独立轨,冻结现状)

- 现状:服务以无 MTP 配置在役(2026-10-29 起);MTP 配置存档于
  `/tmp/launch_nospec.sh` 的反向(加回 `--speculative-config`)。
- B1:生产 metrics 复核 draft_tokens/drafts≈1.02、pos1 验收 1.7% 的成因
  (step3p5 proposer `parallel_drafting` 分支 vs 顺序起草);只读调查,
  不动 A 轨。
- B2:重启用裁决——A 轨每落一项,B2 重测 MTP 臂(step 构成变了,
  MTP 的 verify/draft 收益重估);**B 轨不设时间表,独立裁决**。

## 5. 验收门汇总

| WP | 微测门 | e2e 门(无 MTP 臂) | 数值门 |
|---|---|---|---|
| W2' | o ≤20µs、qkv ≤22µs;gateup ≥570 GB/s 零回归 | step p50 −5% 以上 | `torch.equal` bit 级 |
| W3 | —(策略调参) | 128k p50 −10%;7.5k/32k 回退 ≤2% | 零翻位或 S29 口径披露 |
| W4 | 融合核 per-op −20%(wp4 原判据) | e2e −0.8 ms/round | T=1/T=4 位同微测 |

## 6. 风险

- W2'-H1 的 slice 加大对 MS>1(MTP verify)路径的 LDS 压力 → A 轨无 MTP
  期间 MS>1 仅微测覆盖;B2 重启前补 verify 档微测;
- W3 段数加大 → 归并核成为新瓶颈(reduce_segments 占比入 profile 门);
- W4 融合改变 norm 的归约序 → 位同微测双档,失败回退 env。
