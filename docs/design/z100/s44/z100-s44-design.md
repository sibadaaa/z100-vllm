# S44 设计图:push-AR 出口融合收尾(W4')+ 碎核账本猎杀(Track N)

> 日期:2026-11-01。输入:S37 W4 部分交付(内核+微测)、S38 A1 树上模型
> 穿透、S39 decode 步账本、S43 P0 关档(GEMV 一阶空间关账)、`.s44/
> ledger_classes.py` 首次 decode 窗类目账。**用户裁决:MTP 复役后置,先把
> 内核侧非 GEMV 的可融合成本收干净。**本轮问题:GPU busy 20.7 ms/step 里,
> AR 出口 norm(~1.5)+ 碎核/GDN 链(~2.2)两块,能不靠 GEMV 内核收掉
> 多少?

---

## 0. 基线快照(2026-11-01,在役 S41 态)

| 项 | 值 |
|---|---|
| 在役臂 | 无 MTP(`SPEC_OFF=1`)+ S40 段 + S41 段;`_rocm_C.abi3.so` md5 `f72bb39d`,**`_C_stable_libtorch.abi3.so` md5 `54ccf2e2`(custom AR 库,含 fused 核)** |
| 服务级 | W1 贪心 38.7-40.0 t/s;TTFT 0.706-0.75 s;step p50 25.6-26.2 ms |
| decode 步桶(rank3,S41 B 臂 trace 重切,楔死窗) | fp8_gemm 10.97、**allreduce 4.01(楔窗污染,净参考 = S39 账本 3.76)**、**norm_elemwise 3.31**、attn 1.35、gdn 0.95、bf16 0.58、other 0.36;GPU busy 20.73 ms/step;服务 step − busy ≈ **1.5-2 ms 纯空隙** |
| Track N 范围类目账(decode 窗,39 步,`.s44/ledger_classes.py`) | norm 五类合计 **≈1.50**(`fused_add_rms_norm_5` 1.05 + `_3` 0.26 + 全注意力层 `_3` 0.10 + `_1` 0.09);GDN 链 `fused_recurrent` 0.61 + `bf16_emu_gemv_dot2` 0.58 + `triton_per_fused_1` 0.36 + `causal_conv1d` 0.34 + `mul_silu_slice_4` 0.24;**fp8 量化链 0.24**;`triton_poi_fused_0` 0.29、`_6` 0.23、`reduce_segments` 0.32(attn 归属);合计 scope 5.19 ms/step |
| W4 存量资产(S37 交付,树内) | 内核 `cross_device_reduce_push_fused`(`csrc/custom_all_reduce.cuh:118`,push 五步/归约序不变,出口 add-rmsnorm,AR 和位同——fp32 和 round 回 fp16 的量化点与未融合路径一致);host `all_reduce_fused`(`csrc/custom_all_reduce.cu:320`)→ `torch.ops._C_custom_ar.all_reduce_fused`(`vllm/_custom_ops.py:4466`);python 全链 + 回退(`custom_all_reduce.py:custom_all_reduce_fused`,env `VLLM_DCU_CUSTOM_AR_FUSION` 默认关,形状门 rows 1-36/hidden%16);**模型层穿透已在树**(`qwen3_next.py` patch 逻辑,`Qwen3_5DecoderLayer` 继承生效:两出口 `reduce_results=False` + next-norm 权重管线 + `skip_input_norm`);DEBUG 自检(`parallel_state.py`,前 16 调用对拍 d_out/d_res) |
| S37 微测(4 rank) | per-op 融合/未融合 = **0.81-0.88×**(rows 1-12 全档),residual Δ=0,max\|Δout\| 2.4-4.9e-4;×~124 op/step ≈ **0.7-1.2 ms/step 上限口径** |
| 已证伪/不重试 | GEMV 结构+位宽空间(S33-S43 全谱);AR 自旋部分 = rank 到达差吸收,**通信核不可消**(S39);eager 化(+55 ms/step,S39) |
| 判读纪律 | 楔死窗桶中位数不可用 → 单发中位 + 类目账口径(本轮 ledger 即此法);S29 板对本轮适用(同权重,序差类变更) |

**排序与结构**

```
Track AR(W4' 收尾,0.5-1d,零重编)
  A0 功能探针(DEBUG 自检 16 调用对拍)     0.5h
  A1 数值门(d_out ≤5e-4 + S29 板留档)     0.5h
  A2 微核窗复测(当日锚点)                 0.5h
  A3 e2e ABAB(env 翻转,同 .so)+ 桶门      0.5d
  A4 采纳(env 入 serve_deploy + DEPLOY)   0.5h
Track N(碎核猎杀,2-3d,N1 需重编 _C_stable_libtorch)
  N0 账本归档(已完成,本设计即据)          0h
  N1 fp8 act-quant 折进 AR 出口(csrc)      0.5-1d
  N2 GDN 链融合(silu/conv1d 入邻核)        1d
  N3 triton 碎核定性与处置(per_fused_1 等) 0.5d
  N4 reduce_segments→attn 出口(可选)       0.5d
不施工:GEMV(S43 关档)、MTP(用户后置)、INT8(WP7 预案另裁,预期修正
为 −1~1.7 ms/step,见 S43 引擎账)。
```

排序理由:Track AR 全部资产就绪、零重编、预期收益最确定(≥0.8 ms/step),
先拿;Track N 的 N1 依赖 AR 融合形态定案(quant 折进 AR 出口),排后;
N2/N3 与 N1 无依赖可并行。

---

## 1. Track AR:W4' 收尾(0.5-1d,零重编)

### 1.1 机理与数据流

- TP4 每层 2 次 AR(attn o_proj/out_proj 出口、MLP down_proj 出口),
  64 层 = 128 次/step,每次出口紧跟 add-rmsnorm(Gemma (1+w) 约定)。
  融合 = AR 内核第 4 步(本地归约)写出前就地做 add+norm:
  `s = Σ_rank fp32(写回 fp16) + residual;out = s·rrms·(1+w)`,
  归约序不变(rank 0..3 fp32)→ AR 和与 residual **位同**;新序只有行内
  Σs² 归约树 → **数值面归 S29 板管辖**(先例:S34 push-AR 采纳同型)。
- 两个出口:attn 侧配本层 post_attention_layernorm;MLP 侧配**下一层**
  input_layernorm(权重由模型循环管线注入,下一层 skip 自己的 input
  norm)。MLP 侧可独立关(`VLLM_DCU_CUSTOM_AR_FUSION_MLP=0` = attn-only
  混合臂,二分定位用)。
- 收益构成:①norm kernel 本体(~1.5 ms/step,128 发);②AR→norm 之间的
  核间空隙(S33-WP0 口径 ~5 µs × 128 ≈ 0.6 ms);③norm 的读写往返
  (hidden 5120 fp16 ×2 × 128 次 ≈ 1.3 MB/step,带宽占比小)。上限口径
  = S37 微测 0.7-1.2 ms/step + 空隙回收,e2e 验收门按 **−0.8 ms/step
  GPU busy**。

### 1.2 门(全链,env 翻转式 A/B,同 `.so`)

| 门 | 判据 |
|---|---|
| A0 功能 | `VLLM_DCU_CUSTOM_AR_FUSION_DEBUG=1` 前 16 调用对拍:**d_res = 0(硬)**,d_out ≤ 5e-4(S37 微测同档);形状门命中(rows≤36/hidden%16) |
| A1 数值 | S29 板(贪心流 top5 logprobs ×3 rep vs 基线):留档首分歧位置/偏移分布;**无乱码/无复读**(序差类变更先例 = S34 push-AR 采纳) |
| A2 微核窗 | `car_fused_bench.py` 当日复测 per-op ≤ 未融合 ×0.90(锚点漂移 ±3% 内) |
| A3 e2e | ABAB(W1 96×3 + W2 512×2 配对):**GPU busy −0.8 ms/step 门**(桶账:norm_elemwise 3.31 → ≤1.9,allreduce 桶 ±5% 内);W1 ≥ +2%;TTFT 不回归 |
| A4 采纳 | env 写入 `serve_deploy.sh` + `.s34/DEPLOY.md` + 回滚行;FP 指纹更新 |

### 1.3 风险

- **graph 捕获指针烘焙**:融合核与现役 push 核同机制(S33 教训已吸收:
  `_capture_registered` 路径复用),但 norm_weight/next-norm 指针是新烘焙
  面——A0 必须在 piecewise graph 形态下验(与部署形态一致,不能只跑
  eager)。
- **边界层**:最后一张 MLP 出口的 next-norm 是 final norm——S38 代码的
  skip 管线对尾层的行为需 A0 里显式核验(`qwen3next_fused_next_norm`
  对最后层传什么)。
- 回滚:env 默认 0,翻转即回;`.so` 不动。

---

## 2. Track N:碎核账本猎杀(2-3d)

### 2.1 N0:账本(已完成)

`.s44/ledger_classes.py` 对 S41 B 臂 trace 的 decode 窗类目账即本设计
§0 表;归档 `docs/perf/data/s44-n0-ledger.txt`。**后续任何融合提案必须
引用账本行**(总时长 × 每步发数 × 中位 µs),不接受拍脑袋候选。

### 2.2 N1:fp8 act-quant 折进 AR 出口(0.5-1d,csrc)

- 现状:MLP 出口融合后产 normed fp16,下一 GEMV 前还有激活量化
  (`fp8_emu_mm_block_w8a16` 链,0.24 ms/step + 每层一处核间空隙)。
- 改动:融合核 epilogue 增 `quant_out` 变体(`out_fp8 = e4m3(s·rrms·(1+w))`,
  per-128 scale 另出)——S37 W4 设计 §2 已留草图;python 侧
  `custom_all_reduce_fused` 增可选 `quant_out` 参数,`_MLP` 臂启用。
- 门:A0 型对拍(d_out ≤5e-4 **且 quant 路径 vs 现役 quant 核位同**——
  量化是逐元素映射,序无关,可硬门 torch.equal)→ 窗口 → e2e。
- 重建:**必须 build-sync2**(`docker exec -e MAX_JOBS=8 z100-regress
  bash -lc 'bash /mnt/game/port/build-sync2.sh _C_stable_libtorch'`),
  裸 ninja 编到陈旧副本(S37 已踩坑);先验符号无新增 UND。

### 2.3 N2:GDN 链融合(1d)

GDN 48 层 decode 链现状(每层 5 核,~48 µs 链内 + 空隙):
`bf16_emu_gemv_dot2`(in_proj_a/b,0.58)→ `causal_conv1d_update`(0.34)
→ `mul_silu_slice`(gating,0.24)→ `fused_recurrent`(0.61)→ norm(已由
AR 融合收)。

- N2a:silu·mul 折进 `bf16_emu_gemv_dot2` epilogue(gating 的两个输入
  正是 in_proj_a/b 的输出)——消 0.24 ms/step + 48 处空隙。
- N2b:`causal_conv1d_update` 折进 `fused_recurrent` prologue 或保持
  独立(卷积窗状态更新与 delta rule 共享状态布局,融合收益在空隙不在
  字节;若 prologue 融合使寄存器/状态布局复杂化,允许关档)。
- 门:每项独立四步(微测 → 位同硬门——GDN 链逐元素/小归约,torch.equal
  可达 → 窗口 → e2e);N2 合计现实预期 **−0.3~−0.5 ms/step**。

### 2.4 N3:triton 碎核定性与处置(0.5d)

`triton_per_fused_1`(0.36)、`triton_poi_fused_0`(0.29,63 发/step)、
`_6`(0.23)——inductor 自动融合产物,内容需归因:torch profiler 对齐
源算子(`TORCH_COMPILE_DEBUG` 或 stack 计数)。处置三分:能并进邻居核
→ 融合;本身已是多算子融合 → 声明最小形态关档;发起方是 python 侧冗余
调度 → 上游修。**逐项独立,微-窗-板-四步**。

### 2.5 N4:reduce_segments → attention 出口(0.5d,可选)

`reduce_segments` 15.6 发/step × 19.8 µs = 0.32 ms/step(16 全注意力层
的 segment 归约,S37 SEG_MAX 语义)。折进 attention 内核 epilogue 的
收益 0.3 ms 级,但动 attn 内核须过 attn 位同门(S37 A1 同门);列为
可选,视 N1-N3 后的余量决定。

### 2.6 Track N 总门

- 桶账:norm_elemwise+gdn+bf16+other 合计 **5.19 → ≤3.7 ms/step**
  (AR 融合吃 1.5 + N1-N4 吃 0.5-1.0,留 ±0.3 折损);
- 每项独立落档,任何一项 kill 不牵连其他;
- e2e 汇总门:相对 S44 起点 W1 ≥ +5%(0.8+0.5 ms busy 折算下限)。

---

## 3. 预期终态与验收总表

| WP | 门 | 判据 |
|---|---|---|
| AR 收尾 | busy −0.8 ms;norm 桶 3.31→≤1.9;d_res=0 硬门 | env 翻转 ABAB,零重编 |
| N1 quant fold | quant 路径 torch.equal 位同 + 窗口不降 | csrc 重编(build-sync2) |
| N2 GDN 链 | 逐项位同 + 微测 −µs | 每项独立四步 |
| N3/N4 | 账本引用 + 归因,逐项关/立 | 单发中位口径 |
| Track N 总 | scope 桶 5.19→≤3.7 | e2e 汇总 W1 ≥+5% |

**合计现实预期**:GPU busy 20.7 → **18.0-18.7 ms/step**;W1(无 MTP)
38.7-40 → **41.5-43.5 t/s(+8-11%)**;TTFT 持平。完成后内核侧(结构/
位宽/融合/碎核)全谱收账,**GEMV 与融合之外的服务级成本只剩 AR 自旋
(~1.9 ms,rank 到达差)与空隙残余(~1-1.5 ms,调度面)**——MTP 复役
(S45)在该基线上叠加 S37 GO 口径 +29-42%。

## 4. 风险

- **Σs² 序差非位同**:AR 和/residual 位同(d=0 硬门)但 norm 输出非
  位同——贪心流必然可见分歧。门 = S29 板留档 + 语义双查 + e2e 配对,
  不设"0 错位"这类对序差变更不适用的门(S34 先例)。
- **MLP 侧管线边界**:最后层 next-norm / 第一层 skip 的边界行为——A0
  显式核验;有问题退 `_MLP=0` 混合臂(收益 ×0.6,仍正)。
- **N1 重建引入回归**:fused 核变更重编 `_C_stable_libtorch`——重建后
  先跑 A0 型对拍 + 现役 AR 桶复测(±3%)再上 ABAB;`.r3/so-backup/`
  双 `.so` 备份。
- **楔死窗口径**:trace 桶在楔死时以单发中位/类目账判读(本轮方法已
  固化到 `.s44/ledger_classes.py`);e2e 门不依赖 trace 桶。
