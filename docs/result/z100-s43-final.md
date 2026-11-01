# S43 终局报告:INT4 w4a16 GEMV(P0 探针双变体关档,零采纳)

> 2026-11-01。输入:`docs/design/z100/s43/{z100-s43-design,
> z100-s43-construction}.md`;基线:S41 终局(无 MTP 臂 + S40 段 +
> S41 段,`.so` md5 `f72bb39d`)。**本轮 `.so` 零改动**(md5 全程
> 不变),服务零改动,唯一施工面为独立探针。判定台账:
> `.s43/p-close.md` + `docs/perf/data/s43-{anchor,p0-probe}.txt`。

## 0. 结果一览

| WP | 门 | 实测 | 判定 |
|---|---|---|---|
| §0 基建 | 锚点漂移 ≤3% | `.s43/` 目录、`.so` md5 备份 `.r3/so-backup/s43-before.md5`;锚点 read 792.6(S41 792.9,0.04%)、gate_up 现役窗 567.9 | **完成** |
| P0 几何探针 | gate_up 物理 ≥550 GB/s(0.5 B/元素) | **v1 381.7 / v2 403.9**(= 763/808 G-elem/s);同轮 mode15 = 643.3 与历史复现;数值自检 v1 1.6e-7 / v2 1.5e-6 | **双变体 FAIL → P/K/DEP 整轨关档**(`.s43/p-close.md`) |
| K0/K1/K2/K3 | (视 P0) | 不施工 | 关档 |
| DEP ABAB | (视 P0) | 不施工 | 关档 |
| Q 量化/PPL | (kill-fast 前置) | RTN 管线 + bin 已留档(`.s43/gen_int4_probe_bins.py`、`int4_{v1,v2}.bin`);完整 GPTQ+PPL 需 ~5 h 停服窗口,轨死不投入 | **产物保留,不追加** |

**净效果**:1 探针、1 关档、0 采纳、0 代码/服务改动。在役路径
(S41 态)原样。

## 1. 本轮结论(证据强度:同轮零漂移 + 数值自检 + 双重投影否决)

1. **INT4 w4a16 在 wholea 骨架上不付**(P0,强证据):字节减半只买到
   元素吞吐 ×1.256(808 vs 643 G-elem/s),门要求 ×1.65(≥1060)。
   GEMV 引擎(A-LDS 读 + dot2 MAC + 宽展 + 折算)在 ~800 G-elem/s
   饱和,与载荷位宽无关;即使宽展成本归零,上界 ~900-950 仍不够
   (S42 已证 643→782 的 gap 在寄存器预算内不可回收)。
2. **独立否决线**:即使强推,投影 GEMV 桶 11.15 → 9.07 ms/step
   (−18.6%),W1 约 +8-9%,不足 DEP 门 +15% 的一半。
3. **K=5120 尾半 tile 代价量化**:tile-2048(83% util)比 tile-1024
   (100% util)慢 **5.5%**——v2 布局在两个形状上一致占优。
4. **探针方法学勘误(留档)**:wholea 系探针 `out[row]=(uint32_t)v`
   存 int 截断值,kernel-vs-kernel EXACT 比较成立;int4 内核改存
   float 位模式以支持绝对参考对照。DTK `__int2half_rn` runtime
   miscompile 疑云撤案(测试侧解码错误,gfx906 cvt 路径无 bug)。

## 2. 对后续轮的意义

- S33-S42 证伪的是**内核结构空间**,S43 P0 证伪的是**载荷位宽杠杆**:
  2 FLOP/B 的 GEMV 在此引擎上,`物理 GB/s × 元素/字节` 的乘积上界
  ~810 G-elem/s,fp8(643)已在其 79%。字节侧与结构侧均已穷尽,
  **GEMV 桶的一阶空间关账**;剩余空间在服务级(push-AR ~3-4 ms、
  norm/碎核 ~2.6 ms)或算法侧(spec/MTP),不再在单内核形态里找。
- INT4 权重质量评估若未来单独立项(蒸馏/压缩基线),Q0/Q1 脚本从
  `.s43/` 直接可起,不需要本轮任何遗留状态。

## 3. 产物

- 探针源:`.s43/widen_tax.cu`(mode 20/21 + int4 bin loader + fp64 参考)
- bin:`.s43/int4_v1.bin` / `int4_v2.bin`(g128 对称 RTN,V1/V2 布局
  位同交叉验证 0 diff)+ `int4_probe_meta.json`
- 脚本:`.s43/gen_int4_probe_bins.py`(RTN 管线,可复用)、
  `debug_int4.cu`、`t_cvt.cu` / `t_cvt2.cu`(cvt 路径证据)
- 判定:`.s43/p-close.md`
- 数据:`docs/perf/data/s43-anchor.txt`、`docs/perf/data/s43-p0-probe.txt`
- 分支:`s43`(自 gfx906/main@9a66d66 切出)
