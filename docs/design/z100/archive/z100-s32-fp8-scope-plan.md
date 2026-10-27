# S32-fp8：fp8 约束下的剩余优化空间（2026-10-26）

> 约束：权重保持 fp8（nvfp4 列为技术储备，见 §6）；基线 = s30c-full
> **29.48 ms/tok**。本文是 S31/S32 在"不动权重精度"前提下的细化路线。
> 口径与门禁沿用：7469 doc / 96 tok / 3 reps、双口径、N≥2、贪心流零翻转、
> C7 构建（GDN 在 `_C`、fp8 在 `_rocm_C`）。

## 1. fp8 约束下的地板重算

线性层 5.58 GB/round 是 fp8 的硬字节数（M=1 每权重字节恰读一次，L2 无复用
可挖）→ @650 GB/s = **8.6 ms 硬墙**。现状 12.49 ms（447 GB/s）。全块到顶后：

| 块 | 现状 | fp8 现实目标 | 依据 |
|---|---|---|---|
| fp8 线性 | 12.49 | **9.0–10.0** | whole-A + split-K + 冲突审计（§2） |
| 通信 | 7.44 | **3.0–4.5** | one-shot AR（**−2.5~4**，skew 等待不消失）+ 分块重叠（−0.5~1）（§3） |
| triton 小核 | 3.26 | ~2.0 | 融合（§5；norm 须融进 AR 核，见勘误③） |
| attention | 1.61 | ~0.6 | split-KV + KV fp8 LUT（§4） |
| lm_head | 0.91 | ~0.46 | fp8 化（§5，eval 裁决） |
| GDN 递推+conv | 0.97 | ~0.65 | CFG 扩展 + conv 合并 |
| 核间空隙/发射 | ~1–2（待实测分账） | ~0.5–1 | 勘误④：wall 29.48 − GPU busy 27.4 未分账部分 |
| **合计（wall）** | **29.48** | **~16.5–18.5**（确定性项落地 ~21–23.5） | |

结论：**fp8 内不靠算法，现实终点 ~17–18.5 ms；确定性内核/结构项先兑现到 ~21–23.5**。

### 勘误（2026-10-26 自审计）

1. one-shot AR 收益从 −4~5 修正为 **−2.5~4**：nccl 时长里含"等最慢的卡"的
   skew 成分，one-shot 只省协议/发射开销，等人的时间省不掉。
2. whole-A 对 down（K=17408 → 34 KB LDS）会把每 CU 压到 1 block（占用过低，
   延迟反而暴露）——down 必须配合 K 切分（4×8.7 KB）才成立；qkv/o/gateup
   （10 KB）无此问题。
3. **rms_norm 不能融进行并行 GEMV 的 epilogue**：该 GEMV 各 rank 只持有部分和
   （AR 之前），归一化需要 AR 之后的完整行 → norm+quant 的正确融合点是
   one-shot AR 核本身（归约完顺手做 norm）；silu-mul 融 gateup epilogue
   仍成立（逐元素、无跨卡依赖）。
4. wall 29.48 与 GPU busy 27.4 之间 ~2 ms 此前被"host≈0"一笔带过：应归为
   核间空隙/发射/同步尾巴，须先用 ITER_DETAILS+trace 分账实测，再决定用
   减核数（融合）还是核内吸收来兑现。

## 2. fp8 GEMV 深挖（12.49 → 9.0–10.0，主战场）

### WP-f1：m1 whole-A 变体（无 barrier K 循环）——本文新增的核心项

- **机制**：M=1 时整行激活仅 K×2B（qkv/o/gateup 10 KB，down 34 KB），一次
  stage 进 LDS（含 nkb 折算 scale）后，K 循环内**只剩 W 预取 + MAC，零
  `__syncthreads`**。现 m1 内核每 slice 2 次 barrier × 5–17 slice，是 552→650
  之间的主要摩擦。LDS 预算：10 KB（qkv/o/gateup）→ 4–5 block/CU 正常；
  **down 34 KB → 每 CU 仅 1 block，占用塌缩（勘误②），必须配 K 切分
  （4 份 × 8.7 KB + 部分和归约）才成立**。
- **与 stream2 的区别**（s2 实测更慢的教训）：s2 固定 rows=64/512 线程，小 N
  网格塌缩（o 只有 24 blocks）；whole-A 沿用 m1 的 rows=4/8 几何 + PF[3–4]
  深预取（barrier 消失后寄存器余量允许可加深）。
- **数值性质**：slice 序、每 lane chunk 序、acc 累加序与现 m1 完全一致 →
  **与生产位同**（微测仍出 max|Δ|=0 报表确认），零翻转门免检 pass。
- **判据**：gateup ≥600 GB/s、down ≥560、qkv/o ≥450（微测）；e2e −1.5 ms 以上。

### WP-f2：split-K 小 N（S31-WP1 细化）

o(1536)/GDN-out(1536)/qkv(2048) 网格不足的问题与 barrier 无关时仍存在 →
K 切 2–4 份 + 第二步归约。**改变累加序** → 零翻转硬门 + max|Δ| 报表 +
0-margin 平局按 WP-C 裁决。放在 WP-f1 之后做（f1 可能已吃掉大半 gap，
按 f1 后的实测决定是否需要）。

### WP-f3：LDS bank 冲突审计

lane 读 `sA[lane*16]` 32 B 跨步 → 8-bank 组 4-way 冲突嫌疑；对生成代码
（hip 汇编/isa_probe 流程）做 ds_read 冲突审计，必要时 A slice 行距 pad 1×
uint4。预期 +5–10% 带宽，数值位同（只改地址不改序）。

## 3. 通信（7.44 → 2.0–3.0）

- **S32-A1 one-shot AR**（全文开放方案已列，fp8 无关）：−4~5 ms，固定 rank 序。
- **WP-f4 分块重叠 AR**：row-parallel GEMV 的 N 维切 2 份，N[0:·] 算完即发
  one-shot AR，与 N[·:] 计算流水——把剩余 AR 延迟藏进 GEMV。在 one-shot
  落地后再做（依赖其缓冲/flag 基建），目标再 −0.5~1 ms。

## 4. attention（1.61 → ~0.6）

- split-KV decode 分段（S31-WP2）：−0.6~0.9。
- **KV fp8 + 核内 LUT 反量化**（fp8 主题内允许，前测 ITL +41% 的根因假设 =
  反量化未融合）：KV 字节减半 121→61 MB，@7.5k ctx −0.15~0.25，262k ctx
  按比例放大。先微测证伪"未融合"假设再进服务级。

## 5. 其余

- 融合（S31-WP3 修正版，见勘误③）：**rms_norm/残差/quant 融进 one-shot AR 核**
  （归约完成即得完整行，顺手做 norm；S31 原方案的"并入行并行 GEMV epilogue"
  不成立——该处只有 AR 前的部分和）；**silu-mul 融进 gateup epilogue**（逐元素，
  成立）；add_clone 类冗余拷贝随融合消灭 —— 合计 −0.6~1.2，并顺带减核数
  （对勘误④的核间空隙同样有收益）。
- **lm_head fp8**：907→~460 µs（−0.45）。fp8 模型内把最后一个 bf16 大块也收进
  fp8，属"fp8 量化优化"题内之义；但直接改 top-1 logprob 分布 → 单独 eval +
  裁决（贪心 diff 板 + top5 偏移报表）。
- GDN：CFG 表扩展（每 lane 2 v-行档等，纯查表数值位同）+ conv 并入 combined 核。
- fp8 量化方案本身（scale 粒度/校准）对**速度中性**（字节量不变），其价值是
  释放质量预算去换 lm_head fp8 / KV fp8 这类"再省字节"的票——按 eval 决定
  花在哪里。

## 6. nvfp4 技术储备注记

现内核家族以 **LUT 指针 + 反量化函数**为精度插拔点（fp8x4→f16x2 LUT）；
nvfp4（E2M1 + fp8 块 scale + tensor scale 的两级缩放）只需替换反量化函数与
scale 折算逻辑，网格/流水/barrier 结构全部复用；微测/parity/e2e 管线
（`s30c_gdn_m1_bench`、`p0_gemv_bench`、capture/kdiff）同样直接复用。
建议把 §2 的 whole-A/split-K 内核按"精度无关"接口实现，为 nvfp4 预留。

## 7. 顺序与预期

| 步 | 内容 | 预期 step | 数值性质 |
|---|---|---|---|
| 0 | 核间空隙分账实测（勘误④） | 只测量，不改 | — |
| 1 | WP-f1 whole-A（qkv/o/gateup）+ f3 冲突审计；down 配 K 切分 | 29.48 → ~27 | **位同** |
| 2 | 融合（AR 核内 norm + gateup epilogue）+ GDN CFG | → ~25–26 | 报表门 |
| 3 | one-shot AR（+f4 重叠） | → ~22–24 | 报表门 + 长跑 |
| 4 | split-K 小 N（按需）+ split-KV | → ~21–22.5 | 零翻转硬门 |
| 5 | lm_head fp8 + KV fp8（eval 后） | → ~20–21 | eval + 裁决 |

fp8 内不引入 spec/量化的现实终点 **~20–21 ms**；其中确定性项（步 1–3）
不依赖任何评测裁决，可先行。
