# S39 终局报告:GEMV LDS 消除 + 集合通信归位 + 4 并发口径(W4 capture 探索)

> 2026-10-31。输入:`docs/design/z100/s39/z100-s39-design.md` /
> `z100-s39-construction.md`;基线:无 MTP 臂(systemd `z100-vllm-mtp2`,
> `SPEC_OFF=1`,FP `2f3793deb97e138f`,`.so` md5 `ca05f746`/`54ccf2e2`)。
> 收档模板对齐 s37/s38。**全部判定台账:`.s39/{a1,a4,a5,b1,b2,c1,d1}-close.md`。**

## 0. 结果一览(全部 WP,逐门验收)

| WP | 门 | 实测 | 判定 |
|---|---|---|---|
| A1 widen 税微基准 | (B/A ≤0.75) 且 (C/B ≥1.15) | gate_up **B/A=0.948**(税 5.2%)、**C/B=0.474**(2.1× 慢);o_out 同向 | **LDS 假说证伪** |
| A2 LUT dword 化 | 依赖 A1 证实 | A1 证伪 | **关档留证** |
| A3 位技巧 widen | 256 码全表位同 + 微测 | 全表位同已证(254 码,公式修正);但 VALU 预算判死(≈50 ALU/dword vs 4 issue/cycle/CU) | **关档留证** |
| A4 dequant 归并 | decode 计数 →0 | **decode 相本来就是 0**(129 次全在 prefill,12–772 ms;"0.65 ms/步"是窗口平均伪影) | **闭账不改码** |
| A5 K=1536 跑道 | o/out −20% | RW=2:+3~5%(更差);wholea 代测:−3.6~−7%(噪声带内,±3–5%);B1-c 不施工 | **FAIL 关档** |
| A6 部署 + e2e | A 组合并 −0.8 ms | 无采纳项 | **撤销** |
| B1 nccl 归位 | decode AR 计数=0;e2e −0.5 ms | **decode 相 AR=0(两窗一致)**;"3.3/step"是 prefill AR 的窗口摊销伪影 | **前提证伪,闭账** |
| B2 AllGather | AG ≤100 µs;−0.2 ms 或闭账 | 24 B 采样器打包 AG,skew 吸收(最后到达者 ~55 µs);**发现 lm_head 947 µs/step 大核**(417 GB/s) | **闭账 + 新 WP 候选** |
| C1 4 并发 | 聚合 ≥×3;ITL ≤+15%;solo ≤2% | **×1.95**(76.0 t/s);ITL +67~145%;solo 回归 −5.7%(归因未决) | **FAIL,口径入 DEPLOY** |
| D1 W4 capture | eager 惩罚 <0.5 ms | **惩罚 ≈ +55 ms/step(3× 慢)**,探针 a 命中即停 | **eager 采纳否决,关档** |
| **净效果** | step 26.7 → 23–24.5 ms | **0 收益,0 部署变更**;账本修正 + 容量口径 + 下轮靶点 | 见 §2/§3 |

**生产位(终态,逐位核验)**:FP `2f3793deb97e138f`(原值)、`.so`
`ca05f746`/`54ccf2e2`(原值)、env 零变化(探针 env 已全部移出
`serve_deploy.sh`)、W1 贪心流与 `runs/s38/parity-s38-prod-restore.json`
**96/96 逐位一致**。S39 无任何采纳项——这本身是本轮最重要的产出:
**三条主打优化路径(LDS 消除、nccl 归位、AG 优化)被测量证伪,
避免了三处无效的服务风险**。

## 1. 本轮证伪了什么(证据强度)

1. **GEMV 的瓶颈不是 LDS LUT bank conflict**。同结构同字节量微测:现役
   LUT widen 只值 5.2%(gate_up)/12.7%(o_out)的税,而位技巧 widen 慢
   2.1×。机理:e4m3fn subnormal 需重归一化(clz + 变移位,≈50 ALU/dword),
   gfx906 CU 每周期只有 4 个 VALU issue——**位技巧在架构上不可能赢**;
   LDS 是独立流水,冲突后的吞吐上限仍高于工作点。静态 16-way 冲突分析
   ("C≈10.6")与直接测量不符。
2. **905 GB/s 纯读上限不是 GEMV 形状可达的**。s38 的 905 是 triton read1
   特定 codegen 的产物;本微测下 GEMV 形状(16B/lane)纯读 ~700,
   64B/lane 连续 594,torch reduce 784/D2D 825(与 s38 复现 ±2%)。
   生产 GEMV(615 trace 口径)对形状诚实上限的效率 ≈ 88%,不是 62%。
3. **"3.3 次/step nccl AR"是窗口摊销伪影**。129 个 AR 全部属于窗口内的
   prefill chunk(尺寸门设计行为);decode 相 AR = 0(两窗、四 rank 一致)。
   A4 的"dequant 0.65 ms/步"同型(decode 相 0 次)。
4. **tile GEMM 在 decode M≤4 上不可用**(cfg-35 tile M=4 慢 12×);S18
   "tile M-flat" 的引用与今日实测矛盾。
5. **eager 不是 capture 问题的出路**(+55 ms/step)。

## 2. 账本修正后的 decode 步构成(新窗,rank0,38 步,GPU busy 24.15 ms/步)

| 桶 | ms/step | 说明 |
|---|---|---|
| fp8 stream GEMV(m1) | 12.30 | 263 发/step,p50 43 µs(含 fp8 lm_head? 否——见下行) |
| **fp16 lm_head GEMV** | **0.95** | bf16_emu_gemv_dot2,947 µs×1 发/step,**417 GB/s**,本轮新发现 |
| push-AR(custom) | 3.76 | 132 发/step,28 µs,吸收 rank 松弛 |
| attention 3d | 1.42 | |
| norm/elementwise(triton) | ~2.6 | fused_add_rms_norm 0.98+0.27 + 碎核 |
| GDN core + conv1d | 1.00 | |
| bf16 GEMV(GDN 内部小核) | ~0.55 | |
| nccl AG(采样器打包) | 0.19 | 24 B,skew 吸收 |
| nccl AR | **0** | |

与 W1 服务级 step 25.6–26.2 ms(含间隙)自洽。设计 §0 的"AR 复合 10.1 ms"
账本修正为:push 3.76(自旋)+ nccl 0 + AG 0.19 ≈ **4 ms 级**,且其中
自旋部分吸收的是 rank 间真实到达差,不可由通信内核消除。

## 3. 模型现状(操作者口径)

- **在役**:systemd `z100-vllm-mtp2`,Qwen3.8-27B-FP8,TP4,piecewise
  cudagraph [1,2,4],无 MTP(`SPEC_OFF=1`),FP `2f3793deb97e138f`,
  端口 8000。S39 零变更(代码树留有 A5 休眠门与非采纳 diff,均未部署)。
- **单流(M=1,现役路径)**:W1 贪心 96 tok 当日带 **35.4–39.1 t/s**
  (早晨 38.2–39.1,午后 36.0–38.8,收盘 35.4;同 FP 同码,判环境漂移,
  非代码回归;TTFT 稳定 0.71–0.78 s)。贪心流逐位一致(重启边界核验)。
- **4 并发容量口径(新,已写入 `.s34/DEPLOY.md`)**:聚合 **~76 t/s
  (×1.95)**,单流 16–23 t/s(ITL ~2×),TTFT 0.8–2.7 s。这是 batch M=4
  下 GEMV 1.45–1.75× 缩放的物理上限(无 M-flat 内核可用);4 并发是吞吐
  口径,时延敏感场景保持 M=1 独占。
- **回滚面**:S39 全部 diff 未部署;`git checkout` 各文件即回到 S38 树。

## 4. 下轮候选(按本轮修正后的账本排序)

1. **fp16 lm_head GEMV stream 化**(B2 顺带发现):947 µs @417 GB/s →
   若达 fp8 stream 的 ~600 GB/s 级,潜在 **−0.3 ms/step**;需把
   slice/prefetch 几何移植到 fp16 dot2 路径(新内核,~1d)。
2. **GEMV 访问粒度重构**(A1 顺带发现):每 lane 16B 的形状天花板 ~700 GB/s
   (64B/lane 形状也仅 594,重构需连 dot2 一起改)——高成本低确定性,
   建议先做 1 页微设计再决定。
3. **M-flat decode GEMV**(C1 顺带发现):4 并发聚合 ×1.95 → 若存在
   M≈flat 的 decode 内核则可 ×3+;今日实测 tile cfg-35 慢 12×,需要新内核。
4. **W4 全融合 capture 修复**:仅剩 splitting_ops/影子校验两条路;收益预算
   需按修正后账本重估(attn-only −0.63 ms 的解读部分依赖已被证伪的
   nccl 账本)。
5. **不建议再投**:A2/A3(LDS/位技巧)、B1(nccl 归位)、AG one-shot、
   eager 采纳——均已被本轮测量否决。

## 5. 台账(施工即台账)

- 收档:`.s39/{a1,a4,a5,b1,b2,c1,d1}-close.md`
- diff:`.s39/s39-a5-csrc.diff`(fp8_emu_gemm.cu 工作树累计,未部署)、
  `.s39/s39-b1-probe.diff`(探针,已移除 env)
- 数据:`docs/perf/data/s39-a1-widentax.txt`、
  `s39-a5-harness-{off,on,wholea}.txt`、`runs/s39/e2e-s39-*.json`(c1×6、
  b1-post、d1-eager、final-check)、`.s39/gemv_bw_rows4.{smallk0,smallk1,wholea}.json`、
  `.s39/lut_e4m3.bin`、trace `/mnt/game/s30c-prof/s34nospec_rank*.1793405*.gz`
- 工具:`.s39/widen_tax.cu`(独立 HIP 微测,6 模式)、`.s39/gemv_bw_bench.py`(副本)
- `.so` 前后 md5:`ca05f746`→`fd856950`→回滚 `ca05f746`(备份
  `.r3/so-backup/_rocm_C.s39-before.so`);`54ccf2e2` 未动
- 容量口径:`.s34/DEPLOY.md` §S39-C1
