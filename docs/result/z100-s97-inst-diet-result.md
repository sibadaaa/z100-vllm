# S97 结果：指令瘦身第一轮（ph / sr / phsr，2026-11-14）

设计：[`../design/z100/s97-inst-diet/z100-s97-inst-diet-design.md`](../design/z100/s97-inst-diet/z100-s97-inst-diet-design.md)
施工：[`...-construction.md`](../design/z100/s97-inst-diet/z100-s97-inst-diet-construction.md)
台账：[`.s97/README.md`](../../.s97/README.md)

## 结论

1. **出口 A**：`phsr`（p 保持 half + alpha==1 跳 rescale）131k 中位
   **872.104 ms，−1.174%**（−10.36 ms），优于单臂最优 `sr −0.754%`；
   登记为**采纳候选**；**2026-11-14 用户裁决采纳进生产**（生产链接 →
   `.s97/arms/build_phsr/_rocm_C.abi3.so`，md5 `adba0acb`，131k 冷读
   TTFT 230.36 s vs dot2 231.62 s）。见 [`deploy-adopt`](../../.s97/deploy-adopt.md)。
   短上下文同口径：8k −1.59%、32k −1.14%，与 131k 同量级。
2. 三臂四形状**逐位同**，静态红线全绿（vgpr 70 / spill 0 / LDS 20608 /
   ctas 3）；`ph` 单独 −0.438% 未过 −0.5% 门，记平；`sr` 过门。
3. 收益来源可证：`ph` 静态 −21 条（cvt 34→18、pack 16→8）；`sr` 每半 tile
   省 16 条 mul（>99% 迭代 alpha==1），静态 +8 条守卫。传导折扣 ~50%
   （预测 −2~3%）。

## 臂表（131k / m=8192 / seg=4 / prod op / 9 iters；A/B ×3）

| 臂 | vgpr/spill/LDS/inst | 中位 (ms) | Δ | 数值 |
|---|---|---:|---:|---|
| dot2（基底） | 70/0/20608/1578 | 882.466 | — | 基准 |
| ph | 70/0/20608/1557 | 878.605 | −0.438% | 逐位同 |
| sr | 70/0/20608/1586 | 875.853 | −0.754% | 逐位同 |
| **phsr** | 70/0/20608/1565 | **872.104** | **−1.174%** | 逐位同 |

基底复测 882.490 ms（门 878.2–886.6 ✓）；A/B 三轮同号，波动 <0.03%。

## 门

- 静态：三臂 vgpr 70、spill 0、LDS 20608、bucket 20992、ctas 3；
  `total_inst` ph 1557 / sr 1586 / phsr 1565（红线 <1578 对 ph/phsr 达标，
  sr 为动态收益，静态 +8 在图允许的 1578±10 内）。
- 数值：prefill 8k `b1fa6e88`；32k `f1324f93`；decode 65k seg256
  M=1 `b2beb5a0`、M=8 `ba0e1cce`；decode seg4 `cc39ed05`（均与 dot2 同）。
- HB1：`sr` 守卫编译为 `v_cmp_neq_f32 + v_cndmask + v_cmp_ne_u32 +
  s_cbranch_vccz`，wave 级统一分支成立（形态与图的 `s_or + execz` 不同）。

## prefill 分段（stage op，seg=4，9 iters，ms）

| ctx | 臂 | wall | load | qk | sm | pv | wait |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8k | dot2 | 105.65 | 22.94 | 33.92 | 0.05 | 46.38 | 2.37 |
| 8k | phsr | 105.08 | 22.65 | 33.69 | 7.02 | 39.31 | 2.43 |
| 32k | dot2 | 262.85 | 56.91 | 84.42 | 0.12 | 115.48 | 5.91 |
| 32k | phsr | 260.90 | 55.91 | 83.77 | 17.43 | 97.77 | 6.03 |
| 131k | dot2 | 889.64 | 191.09 | 286.29 | 0.40 | 391.81 | 20.04 |
| 131k | phsr | 882.49 | 187.47 | 284.12 | 58.99 | 331.52 | 20.40 |

三形状 `stage_bitexact=true`。dot2 的 `sm≈0` 是编译器把 rescale mul 沉入
pv 窗口的记账假象；`sr` 守卫把 rescale 钉回 sm（131k 下 pv −60.3、sm +58.6
互补），故"收益全落 pv"不成立，属窗口搬移。净 wall 以 prod 为准 −10.4 ms。
dot2 达峰：load 1141/1150/1164 GB/s，qk 23.0 TMAC/s，pv 16.8 TMAC/s。

## 服务与现场

- S97 收工时按用户令停服（`inactive`），`vllm/_rocm_C.abi3.so →
  .s93/arms/_rocm_C_dot2.abi3.so`（`7d2ed2a2`）；当日深夜用户裁决进生产，
  链接改指 `.s97/arms/build_phsr/_rocm_C.abi3.so`（`adba0acb`），服务
  `active`、health 200，见 [`deploy-adopt`](../../.s97/deploy-adopt.md)。
- 锚点：[`wp0-anchor-pre.md`](../../.s97/wp0-anchor-pre.md) /
  [`post`](../../.s97/wp0-anchor-post.md)。

## 偏差

- 图的 decode 门 md5 与 `--segments` 口径不一致（`b2beb5a0` 为 seg 256）；
  已按两口径各测，均逐位同。
- `sr` 守卫形态非预期 `s_or + execz`，但语义与逐位等价成立。
- 图预估 wall −2~3%，实测 −1.174%；减指令仍有正收益，但传导被
  barrier/发射重叠吃掉约一半。
