# S87 结果：decode 单流冲刺（P2）—— 出口 B，零内核改动

日期：2026-11-12。设计：[`s87-decode-single-stream-design.md`](../design/z100/s87-decode-single-stream/z100-s87-decode-single-stream-design.md)；
施工：[`s87-decode-single-stream-construction.md`](../design/z100/s87-decode-single-stream/z100-s87-decode-single-stream-construction.md)；
台账：[`.s87/README.md`](../../.s87/README.md)、[`.s87/wp0_baseline.md`](../../.s87/wp0_baseline.md)。

## 一句话

**WP0 一开工就否掉了两条主刀的前提：verify 腿的 GEMV 早就是"权重读一遍、M 行共享"，
M=3 实测 361–381 GB/s（设计基线 297）；lm_head 服务侧早就是 1.85 ms/343 GB/s
（设计基线 2.82 ms/115 GB/s，那是绕过路由的 rocBLAS 口径）。两刀的剩余空间
分别 <2% 和 −1.6%，都不值得一次 .so 采纳，按施工图出口 B 结案。**

## 1. 判决

| 刀 | 设计预期 | WP0 实测 | 判决 |
|---|---|---|---|
| P2-a verify GEMV | 297 → 425 GB/s，−6.8 ms | 现役已 **361 / 375 GB/s**；几何扫描最好 +1.6% | **出口 B** |
| P2-b lm_head | 2.82 ms/115 GB/s → ≤1.7 ms | 服务路由 M=3 已 **1.85 ms/343 GB/s**；现成多行 GEMV 打平 | **出口 B** |

服务级 A/B（WP4）未启动：没有 ≥3% 的预期收益，不做无意义的 .so 与配对测量。
`csrc/`、`vllm/`、服务 argv/env 全部零改动（§5）。

## 2. WP0 实测（生产 `.so` `188b6e3a`，rows=4）

M 梯度（11 个真实形状，权重字节/耗时，split / fused 加权）：

| M | split GB/s | fused GB/s |
|---:|---:|---:|
| 1（wholea） | 614.8 | 648.4 |
| **3（verify）** | **361.0** | **375.5** |
| 4 | 317.3 | 328.2 |
| 8 | 154.9 | 157.8 |
| 12（tile） | 66.3 | 71.0 |

几何/开关扫描（M=3，同一 `.so`，无重编）：rows=16 最好（364.6/381.5，+1.0/1.6%），
rows=8 劣（335.7），rows=32/64 崩（188.8/104.4），既有 `M3FAST` 臂反劣（355.1），
`STREAM=2` 崩（63.6）。**配置层已经到顶。**

ISA 证据（`census.py`，fp16、RW=1，只改 MS）：MS=1/4/8 的
`v_dot2_f32_f16` 计数 32/128/256、`ds_read_b128` 8/32/64，严格正比于 MS；
源码
`csrc/rocm/fp8_emu_gemm.hip:808-890` 是"每 (row, slice) 读 1 个 W uint4 → 对全部
M 行各做 8 条 dot2"。**随 M 线性增长的是 dot2 与 A 行 LDS 读，不是 W 的重复读。**
设计图 §1.1 的"每个 M 行重读一份 W"在现役代码里不存在，§1.2 的主刀没有落点。
`M3FAST` 的负结果与 S58 的"`v_pk_fma_f16` 与 dot2 同 MAC 速率"一致，换指令无收益。

lm_head `[62080, 5120]` fp16 = 636 MB/rank：

| 路径 | M=1 | M=3 | M=12 |
|---|---:|---:|---:|
| `F.linear`（绕过路由，rocBLAS） | 4.54 ms | 4.55 | 4.55 |
| **服务路由** `rocm_unquantized_gemm_impl` | 0.81 ms / 783 GB/s | **1.85 ms / 343 GB/s** | 2.57 ms / 247 GB/s |
| `bf16_emu_gemv`（现成多行 GEMV） | 0.91 / 700 | 1.84 / 346 | — |

设计图的 2.82 ms/115 GB/s 与 `F.linear` 的 rocBLAS 口径一致（S57 的旧账），不是
服务实际路径。M=3 现成多行 GEMV 与服务路由打平（1.84 vs 1.85），M≥4 更慢；
即使新内核做到 500 GB/s，也只省 0.58 ms = 步时的 1.6%，低于 −3% 门。

## 3. 为什么不继续 WP1/WP2

1. **P2-a 无可改对象**：主刀就是把现役 stream 内核的 A 侧 LDS 驻留 + M 路 dot2 写一遍，
   而现役代码已经是这个形态；重写只会复现同一份指令序列。
2. **bit-exact 封死了剩余路径**：M=3 的缺口是每条 W 的 M 路 dot2 发射，不是带宽。
   gfx906 上唯一的 2× 指令是 int8 `v_dot4`（4 MAC/inst），要重新量化权重/激活，
   属于有损路线；split-K/多链能提 ILP 但改变 K 求和序，破坏 out_md5 逐位门。
   两者都超出 S87 的零风险框架。
3. **P2-b 的收益在预算内**：即使内核达到设计目标，服务级也只有 −1.6%，而 S81 的
   c=2 +5.4% 悬案还在，风险收益倒挂。

## 4. 收工测量：prefill 131k 各阶段（生产配置）

工具 `.s82b/prefill_stages.py` + `.s87/prefill_stages_rollup.py`；服务按生产 argv/env
重启（`t1int8` `.so`、`--kv-cache-dtype int8_per_token_head`、TP4、MBT 8192、
`NOCACHE=1`），仅用于本轮测量。131072-token prompt，gen=4。

- **首字时间 235.95 s**（与 S85 采纳态 235.78 s 一致，profiler 开销 ≈0.1%）；
- kernel 合计 235.06 s = **99.6% 时间在 GPU kernel 内**；
- 整条 prompt 吞吐 **555 tok/s**。

| 阶段 | 代表内核 | 耗时 (s) | 占比 | 阶段吞吐 |
|---|---|---:|---:|---:|
| **注意力（prefill 内环）** | `attnd_inner_kvsdot4_kernel` | **125.80** | **53.5%** | **1042 tok/s** |
| **GEMM** | fp8-emu / Cijk / Tensile | **82.20** | **35.0%** | **1595 tok/s** |
| allreduce | nccl / custom AR | 10.33 | 4.4% | — |
| dequant / copy | fp8 dequant、convert、CatArrayBatchedCopy | 8.91 | 3.8% | — |
| GDN | chunk gated delta / conv | 5.44 | 2.3% | — |
| other | 其余 | 1.57 | 0.7% | — |
| norm | rms / rope / silu | 0.81 | 0.3% | — |

旁证：同一脚本在"漏传 `--kv-cache-dtype int8_per_token_head`"的错误启动上测得
TTFT 272.13 s、注意力 159.65 s（+27%），说明本轮的 235.95 s 确实是生产 int8 KV
路径，而不是 fp16 回退。测完服务已停（`:8000` 无监听，pid kill + `ss` 双确认）。

## 5. 零改动核对

| 项 | 期望 | 收工 |
|---|---|---|
| `csrc/rocm/fp8_emu_gemm.hip` | `16018d1b284b` | `16018d1b284b` |
| `csrc/rocm/w8a8_sdot4.hip` | `bf7a678b9b45` | `bf7a678b9b45` |
| `vllm/_rocm_C.abi3.so` readlink | `.s85/arms/build_t1int8` | 同 |
| 生产 `.so` md5 | `188b6e3aae53` | `188b6e3aae53` |

## 6. 排除表 / 留给后续

| 项 | 结论 | 依据 |
|---|---|---|
| M=3 GEMV 继续挖 bit-exact | 无空间：现役已是 W 单读 + M 路 dot2，配置层 +1.6% 封顶 | §2 |
| 换 `v_pk_fma_f16` | 否：与 dot2 同 MAC 速率，实测 −1.6% | M3FAST 臂、S58 |
| lm_head 多行内核（M≤8） | 否：M=3 打平服务路由，M≥4 更慢 | 表 3 |
| int8 dot4 GEMV（有损） | 未评估，超出本专项；需新的数值契约与重新标定 | — |
| split-K / 多链 GEMV | 破坏逐位门，需重新定义验收 | — |

## 7. 产物

- `.s87/wp0_baseline.md`：WP0 全量台账；
- `.s87/logs/` + `runs/prefill-lctx/s87/`：探针输出与 JSON；
- `runs/prefill-lctx/s87/prefill_stages_131k_int8.json` +
  `prefill_stages_rollup.json`：§4 原始数据；
- `docs/design/z100/README.md`：S87 索引行改为"已结案（出口 B）"。
