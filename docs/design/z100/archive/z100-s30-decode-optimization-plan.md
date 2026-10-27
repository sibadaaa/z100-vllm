# S30 计划：decode step 89.8 → 诚实地板的剩余路线（2026-10-25 晚）

> 基于 `docs/perf/SOL-GAP-20261025.md` §1.1/§1.2 更新后的桥表。性质：**计划/调研**，
> 开工前按惯例另立施工图（WP 分包 + 验收判据 + 原始数据纪律）。
> 当前基线：**89.77 ms step p50 / 32.4 t/s（1.969 世界）**，qwen3.8-27B-FP8 TP4
> + DFlash2 k=7，graph-A + A1 + draft-3D + BM16 + workspace pin。
> 纪律（不变）：accept 对比一律 **N≥2 重启取分布**；任何触数值的改动过
> parity 板（翻位仅当 Δlogprob(top1,top2) < 0.05 nat 且逐条披露）；
> 每改动：真实形状微测 → 数值对照 → 服务 smoke → 同口径 e2e → 文档。

## 0. 剩余可收的地图（来自桥表 1.1/1.2）

| # | 杠杆 | 可收 ms | 风险 | 证据状态 |
|---|---|---:|---|---|
| 1 | fp8 GEMM 派发 + A 暂存 → 163 GB/s | **~10–13** | 中（纯调度，数值不变） | 机制已证（P0 §4），服务级未做 |
| 2 | draft lm_head → triton 路径 | **~3.2** | 中高（改提案数值） | 微测已在案（sweep 2348 vs 5512 µs） |
| 3 | 采样尾 + host gap | ~2–3.4 | 低（host 侧） | 五相位 trace 已定位尾部 5.1–7.0 ms |
| 4 | FA BM48 实验 | ~1–2 | 中高（tl.dot constexpr 再变） | BM16 零翻位先例；S24 "再减半（实验）" |
| 5 | GDN/norm/quant 杂 | ~2 | 低 | 桥表 |
| | **合计** | **~19–24** | | 89.8 → **~66–71 ms → 41–44 t/s** |

## WP1：fp8 GEMM 形状派发 + A 暂存（最大单杠杆，~10–13 ms）

**问题**：fp8 GEMM 组 38.1 ms（192 次），诚实地板 24.7（4.03 GB @163 GB/s）。
M=8 LDS 读墙的解法（A 暂存 + 形状派发）在部分形状/调用点未到位。

1. **审计**：从 S29 trace 按 (kernel, grid, 形状) 聚合 192 次 GEMV/tile 的
   GB/s，列出低于 163 的形状清单与差距（预期大头集中在少数形状）。
2. **微测**：对差距形状做 A 暂存/派发变体微测（复用 P0 探针框架），
   记录 GB/s 与数值对照（bit-exact 要求：W8A16 路径累加顺序不得变——
   O4 的教训：`FP8_STREAM_MAX_M` 因改累加顺序被否）。
3. **服务 smoke → e2e**：同口径探针 + N≥2 重启 accept 分布 + parity 板。
4. 产出：`docs/result/z100-s30-fp8-gemm-dispatch.md`。
5. 判据：step p50 −5 ms 以上才值得采纳；parity 零翻位（纯调度应天然满足）。

## WP2：draft lm_head → triton 路径派发（−3.2 ms，证据最全）

**问题**：`draft.lm_head`（62080×5120×M7，1 次/round）生产走 rocBLAS
Cijk 5512 µs（115 GB/s）；sweep 实测 triton 2348 µs（271 GB/s）。

1. 定位生产调用点（draft tower lm_head 的 matmul 分派），
   确认 triton 路径在服务内可达（`triton_matmul_kernel` 已在 draft qkv 等形状使用）。
2. 加派发旋钮（默认 off），微测 → 服务内 per-op 对照（trace 计时）。
3. **门禁（强制）**：parity 板 8 长度 × 2 reps + 96-token capture 对拍
   （改提案数值 → 翻位风险在 draft 侧，target 侧 logprob 披露照常）；
   accept N≥2 重启分布（对照 BM16 的 {1.969×2, 1.771} 教训）。
4. 产出：`docs/result/z100-s30-lmhead-dispatch.md`。
5. 判据：parity 零翻位或全部落在 0.05 nat 线内 + accept 分布不降
   （对照臂内分布）+ step −2.5 ms 以上 ⇒ 可提请采纳（用户拍板）。

## WP3：采样尾 + host gap（~2–3.4 ms，host 侧）

**问题**：target→draft 尾部（verify/sampling/prepare）5.1–7.0 ms（profiler
窗口）+ 0.66 ms gap；五相位已定位。

1. 用 ITER_DETAILS=1 + 五相位 parse 细分尾部：sampling/penalties/prepare/
   draft-input-build 各占多少。
2. 只做无数值影响的 host 优化（内核合并、冗余 H2D 消除、同步点削减），
   逐项 smoke。
3. 产出：`docs/result/z100-s30-tail-host.md`；判据 step −1.5 ms 以上。

## WP4：FA BM48 实验（~1–2 ms，实验性质）

BM16 之后 KV 仍被 ~4 个 q-block 重读；BM48 再减半（S24 "实验"标注）。
1. 微测 BM48 vs BM16（`VLLM_GFX906_ATTN_DECODE_BLOCK_M` 旋钮扩展或模板值），
   寄存器压力/occupancy 核对。
2. parity 板强制（tl.dot constexpr 变了）；accept N≥2。
3. 判据：step −1 ms 以上且 parity 零翻位才提请；否则存档。

## WP5：GDN/norm/quant 杂项（~2 ms）

零散：norm/quant 真实计算为主，收干空间小；只在 WP1-3 完成后按残余 trace
顺手收（不做"顺手优化"之外的任何改动——每项仍走完整纪律）。

## 顺序与依赖

```
WP2（证据最全、风险中高）→ WP1（最大杠杆、纯调度）→ WP3（host）→ WP4（实验）→ WP5
（WP2 与 WP1 相互独立可并行；都动 launcher，采纳时一次提交）
```

## 禁止事项

- 不动 `csrc/`（除非另立增量构建流程）。
- accept 单点结论一律不作（N≥2 重启分布）；pin 保留但视作偏置非保证。
- 任何数值变化必须过 parity 板；"更快但改数值"类需用户显式拍板（SMALLM 先例）。
- 窗口结束留一个可用服务在 :8000（现挂 BM16 采纳臂）。

## 预期落点

全部采纳：**89.8 → ~66–71 ms（−21~−24 ms），32.4 → ~41–44 t/s**；
对 SOL 天花板（29.3 ms 假设）的比值从 3.06× 收到 ~2.3×，剩余为
基线乐观成分（TP 结构性 + KV 物理带宽），按 §2 的口径即"工程欠账基本清零"。
