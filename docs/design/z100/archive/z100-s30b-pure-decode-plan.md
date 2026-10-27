# S30b 计划：纯 decode（无 DFlash）路线 —— 实测基线 + 优化计划（2026-10-25 深夜）

> 应"先去掉 dflash 再做优化计划"而生：`.s29/graph_profile_launch_nospec.sh`
> （= 采纳臂去掉 `--speculative-config` 与 `--per-request-spec-decode-metrics`，
> 其余不变：A1/draft-3D 旋钮失效、BM16、workspace pin）。实测后已恢复 dflash
> 基线服务（`:8000`）。
> 配套工具修正：`tools/dcu/s20_single_probe.py` 的 `metrics()` 对缺失的
> spec-decode 计数器默认 0（纯 decode 服务无 `vllm:spec_decode_*` 指标，
> 原探针在此配置下 KeyError）。

## 0. 实测基线（7469 真实文档 / 96 max-tokens / 3 reps，s29-nospec）

| 口径 | 纯 decode（无 dflash） | dflash（采纳臂，对照） |
|---|---:|---:|
| step（=每 token） | **39.2 ms** | 89.77 ms ÷ 2.909 tok = **30.9 ms/tok** |
| 单流 t/s | **21.3** | **32.4**（1.969 世界；跨世界 28.6–29.7） |
| TTFT（热） | **0.74 s** | 2.1 s（含 draft tower 的 7469-token prefill） |
| 每 step 生成 | 1 tok | 2.909 tok |

**dflash 每 token 仍快 31%**——但纯 decode 的 TTFT 好 3 倍、无 accept 彩票、
无提案 parity 风险面、部署/调试面小一个数量级。

## 1. 块表（trace 普查：s29ns_rank0，20 decode rounds，GPU busy 36.24 ms + host ~3 = 39.2）

| 块 | ms/round | 占比 | 说明 |
|---|---:|---:|---|
| fp8 GEMV M=1（253.5 次 × 48.2 µs） | 12.22 | 34% | **330 GB/s**（M=1 的 LDS 压力比 M=8 verify 的 163 GB/s 好一倍）；纯流地板 ~10 @ 400 GB/s |
| **GDN M=1 递推（47.5 次 × 216 µs）** | **10.28** | **28%** | `fused_recurrent_gated_delta_rule_gfx906_packed_decode_kernel`，**grid [6,1,1] = 6 个 block（=KV head 数）严重欠并行**；dflash 侧同功能 `_gdn_mtp_update_kernel` 25.7 µs/次 |
| AR nccl（126.8 次） | 7.44 | 20% | TP4 每层 2 次，结构性；64.7 µs/op 已低于 90 µs 独立地板 |
| inductor fused（norm/quant/copy） | 2.46 | 6.8% | |
| attention M=1（15.8 次 × 77 µs） | 1.22 | 3.4% | BM16 已生效 |
| bf16 GEMV（47.5 小 + lm_head 1 大） | 1.40 | 3.9% | lm_head [62080,1,1] 单次 ~0.7 ms（**~900 GB/s，已高效**） |
| misc | 0.14 | 0.4% | |
| host gap | ~3.0 | | wall 39.2 − GPU 36.24 |

## WP1：GDN M=1 递推并行化（−8~9 ms，本路线最大杠杆）

- **现状**：47.5 层 × 216 µs，grid [6,1,1] —— 每 KV head 一个 block，state
  更新在 block 内串行，延迟型 kernel（dflash 侧同类更新 25.7 µs，差 8×）。
- **方案**：按 (head, state-tile) 二维展开并行；或把
  `_gdn_mtp_update_kernel` 的并行骨架参数化到 M=1。纯代码重构（csrc 外，
  triton/gfx906 kernel 文件内）。
- **数值**：每 head 内的递推顺序不变 ⇒ 应 bit-exact；仍走全门禁
  （per-op 对照 → parity 板 → N≥2 重启 accept——纯 decode 无 accept，
  以 per-token 流 + TTFT/e2e 分布代替）。
- **判据**：per-op ≤ 40 µs；e2e step p50 −5 ms 以上。
- **产出**：`docs/result/z100-s30b-gdn-m1-parallel.md` + 微测 JSON。

## WP2：fp8 GEMV M=1 → 400 GB/s（−2 ms）

330 GB/s 已优于 M=8 路线；按 P0/gemv_probe 结论调 UNROLL/ROWS/NWARP
（M=1 的 A 只有 1 行，LDS 压力小，理论可到纯读流的 ~50%）。同 kernel
配置域内调参，数值不变；判据 per-op −15%、e2e −1.5 ms 以上。

## WP3：host gap ~3 → ~1.5 ms

`--enable-logging-iteration-details` 细分 host 段（采样/prepare/同步点）；
只做无数值影响的削减。判据 e2e −1 ms 以上。

## WP4：AR 7.44 ms —— 结构性，记录不动

每层 2 次 × TP4 是架构决定；64.7 µs/op 已低于独立测试地板。除非改
通信拓扑（出图外），不动。

## 落点与决策含义

- 全部采纳：**39.2 → ~27–29 ms/tok → 34–37 t/s**，TTFT 保持 0.74 s。
- **战略对照**：dflash 现状 30.9 ms/tok（32.4 t/s，跨世界 28.6–29.7）——
  **WP1 达成后纯 decode 即反超 dflash**，且没有 accept 彩票、没有提案
  parity 风险面、TTFT 3×、显存省出 draft 权重与 KV。
- 两条线并行推进（S30 主计划吃 dflash 路线，本计划吃纯 decode 路线），
  以同口径 e2e（7469 doc / 96 tok / N 重启分布）做最终取舍；
  交互/长上下文场景 TTFT 权重更高时纯 decode 额外加分。

## 禁止事项

- 不动 `csrc/`；不做"顺手优化"；每项改动完整走
  微测 → 数值对照 → smoke → e2e → 文档。
- 纯 decode 的对比结论必须标注 TTFT 与 t/s 两个口径（单流交互场景
  TTFT 常常比吞吐更决定体感）。

原始数据：`.s29/s20_s29-nospec_*.json`、`.s29/trace/s27_five_phase_collect_s29-nospec_*.json`、
`$DATA_ROOT/s29-prof/s29ns_rank0.*.pt.trace.json.gz`、`.s29/serve-s29-nospec.log`。
