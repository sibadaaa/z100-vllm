# S93 dot2-PV 复活 —— 结果（出口 A）

日期：2026-11-14。状态：**结案，出口 A（P5b + dot2 采纳）**。
设计：[z100-s93-pvdot2-revival-plan.md](../design/z100/s93-dot2-pv/z100-s93-pvdot2-revival-plan.md)；
施工：[z100-s93-pvdot2-revival-construction.md](../design/z100/s93-dot2-pv/z100-s93-pvdot2-revival-construction.md)。
原始数据：`.s93/`、`runs/pvdot2-revival/`。

## 0. 一句话

**P5b 8→4 bpermute 逐位同，131k 939.47→922.65 ms（−1.79%）；dot2-PV
（p fp16 同源）再降到 882.41 ms（−4.36%），对生产 921.43 ms 净赢
−4.23%；G-B1/G-B2 双绿，服务级 131k TTFT −2.18%，accept 不降。**

## 1. 判决

| id | 假设 | 实测 | 判决 |
|---|---|---|---|
| HA1 | 选择交换把 LDS insts 增量砍半，load 窗口 ≤+25 ms | `SQ_INSTS_LDS` −427.8 M（+856→+428 M）；load +40.1→**+31.8 ms** | 绿（counters）；load 未到 +25 |
| HA2 | P5b 131k 928–936，逐位同 | **922.65**（A/B 中位）；对 P5 逐位同 | 绿 |
| HB1 | 变体 B rel ≤1e-3 / ≤3e-3 | 预估 2.13e-4；真内核 m=1/3 ~4e-4，m=16 1.66e-3（生产基线 1.69e-3） | 绿 |
| HB2 | ≤P5b×0.97 且 ≤生产×0.97 | **882.49** vs 门 894.9 / 893.5 | 绿 |
| HB3 | 服务 accept 不降，TTFT ≥−2% | 32k/131k accept 与 tok/step 持平；131k TTFT **−2.18%** | 绿 |
| HC1 | kBlockM 32 值得立项 | LDS ~40.25 KB ⇒ 1 CTA/CU（现 3），vgpr 70+32 超 90 | 红（纸面闭案） |

## 2. 锚点

开工：`5926429c` / `b93db390`，生产 `.so` `5af8b848`；服务按用户令停止。
收工：源与生产 `.so` 未改；生产符号链接指向采纳臂
`.s93/arms/_rocm_C_dot2.abi3.so`（md5 `7d2ed2a2`），服务 active、health 200。

## 3. prefill wall（prod op，9 iters 中位）

| 臂 | 8k | 32k | 131k | Δ vs 生产 |
|---|---:|---:|---:|---:|
| 生产 `5af8b848` | 109.94 | 272.32 | 921.43 | — |
| P5 重建 | 111.78 | 277.64 | 939.52 | +1.96% |
| **P5b** | 109.57 | 272.65 | **922.72** | **+0.14%** |
| **dot2** | **105.31** | **260.80** | **882.49** | **−4.21 / −4.23 / −4.23%** |

A/B 交替 ×3（131k，3 轮中位的中位）：P5 `939.473`、P5b `922.648`、
dot2 `882.412`。P5b→dot2 = **−40.24 ms（−4.36%）**。

decode（ctx 65k，seg=256）：

| 臂 | M=1 | M=3 | M=8 |
|---|---:|---:|---:|
| 生产 | 0.2355 | 0.3920 | 0.5643 |
| P5b | 0.2347 | 0.3914 | 0.5646 |
| **dot2** | **0.2200** | **0.3688** | **0.5362** |

## 4. prefill 分段测速（stage op，seg=4，9 iters 中位，ms）

### 131k

| 臂 | wall | load | qk | sm | pv | wait | load GB/s | pv TF/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 生产 | 929.44 | 161.44 | 287.35 | 0.40 | 446.96 | 33.29 | 1378 | 14.76 |
| P5b | 931.81 | 193.23 | 280.79 | 0.40 | 434.34 | 23.05 | 1151 | 15.19 |
| **dot2** | **889.61** | **191.11** | **286.28** | **0.40** | **391.79** | **20.03** | **1164** | **16.84** |

dot2 份额：load 21.5% / qk 32.2% / pv 44.0% / wait 2.3%。
PV 窗口较生产 **−55.2 ms（−12.3%）**，较 P5b −42.6 ms；PV TF/s 14.76→16.84。
load 仍比生产多 29.7 ms（P5b 减税后的残税）。

### 8k / 32k（load / qk / sm / pv / wait）

| ctx | 臂 | wall | load | qk | pv | wait | pv TF/s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8k | 生产 | 110.69 | 19.53 | 34.12 | 53.03 | 3.95 | 7.77 |
| 8k | P5b | 110.55 | 23.13 | 33.27 | 51.37 | 2.73 | 8.03 |
| 8k | **dot2** | **105.70** | 22.95 | 33.93 | **46.40** | 2.37 | **8.89** |
| 32k | 生产 | 274.64 | 48.13 | 84.74 | 131.83 | 9.82 | 12.51 |
| 32k | P5b | 275.40 | 57.68 | 82.78 | 128.03 | 6.79 | 12.88 |
| 32k | **dot2** | **262.88** | 56.90 | 84.45 | **115.51** | 5.90 | **14.28** |

## 5. 静态与 counters

| 项 | P5 | P5b | dot2 |
|---|---:|---:|---:|
| vgpr / spill / LDS | 70 / 0 / 20608 | 70 / 0 / 20608 | 70 / 0 / 20608 |
| dot2 / fma_mix / bpermute | 0 / 288 / 56 | 0 / 288 / 44 | **128 / 32 / 44** |
| 总静态指令 | 1717 | 1707 | 1611 |
| `SQ_INSTS_LDS` | 9.466 B | **9.038 B** | — |
| `SQ_WAVE_CYCLES` | 265.27 B | 260.52 B（−1.79%） | — |
| `WAIT_INST_LDS` | 16.38 B | 12.49 B（−23.8%） | — |
| LDS CMD/DATA FIFO full | 5.885 / 1.159 B | 4.622 / 0.913 B | — |
| `SQ_VALU_STARVE` | 28.91 B | 26.88 B（−7.0%） | — |
| bank conflict | 7.7007 B | 7.7007 B（逐位相等） | — |

## 6. 数值

- 变体 B 纯 PyTorch 预估（48 例）：最差 rel `2.13e-4`；
- 真内核 numgate 48 例：最差 `1.663e-3`（同形状生产基线 `1.694e-3`；
  m=16 的 1.5e-3 是工具基线误差，本臂新增 ≈2e-4）；
- KL 最大 `6.73e-5`（门 1e-2）；wrapper1616 最大 `5.0e-4`（门 1.5e-3）；
- **out_md5 必然漂移**（S84 lossy 惯例，开工声明）：8k `a98cb833`→`b1fa6e88`；
  这是 p fp16 舍入的预期结果，不是缺陷。

## 7. WP-C kBlockM 16→32（只判决）

kBlockM 32 需 acc[16]→acc[32]、q_lds 4→8 KB、K/V 缓冲翻倍、ks 256 B，
LDS 20608 → **~40.25 KB**。64 KB/CU 下 3 CTA 必失守（仅 1 CTA），
vgpr 70+32 也越 90 红线；8.8% KV 重读收益无法补偿 3× 占用损失。
**纸面红，不做微测，kBlockM 32 落排除表，不立 S94。**

## 8. 服务级配对 A/B（停服窗）

同参数（token-id prompt，gen=64，各长度 1 次有效测量）：

| ctx | TTFT 生产 | TTFT dot2 | Δ | accept/draft 生产→dot2 | tok/step 生产→dot2 |
|---:|---:|---:|---:|---:|---:|
| 8k | 8258.5 | 8211.5 | −0.57% | 1.250→1.172 | 2.207→2.133 |
| 32k | 37351.9 | 37001.7 | −0.94% | 1.520→1.520 | 2.462→2.462 |
| 131k | 236789.4 | 231621.5 | **−2.18%** | 1.032→1.032 | 2.000→2.000 |

8k 的 accept 在复测中同配置自身波动 1.17–1.74（抽样噪声/前缀命中），
32k/131k 计数逐位持平；ITL 131k 49.60→48.96 ms。TTFT 门达标。

## 9. 出口

- **出口 A**：P5b 与 dot2 全过，采纳。生产符号链接 → dot2 臂，服务 active；
- PV 议题闭案于 dot2 形态；attention 剩余战场 = load 残税（+29.7 ms）与
  GEMM/AR；kBlockM 32 排除；
- 回滚：`ln -sfn .s91/arms/_rocm_C_kvint8fix.abi3.so vllm/_rocm_C.abi3.so
  && sudo systemctl restart z100-vllm-mtp2`。

> **S94 加注（2026-11-14）**：dot2 臂 vgpr 70/90 仍有余量 ~14–20；
> PV 环 drain 探针（d3 = 3 深预发 / b64 载入）已立项为
> [S94](../design/z100/s94-vgpr-headroom/z100-s94-vgpr-headroom-design.md)
> （待裁决开工，disasm 证据见其证据附录）。本文其余结论不变。

## 10. 复现

```bash
OUT=.s93/arms/build_p5b TB_SRC=csrc/rocm/torch_bindings.cpp \
  ATTN_SRC=.s93/arms/p5b.hip bash tools/dcu/attn_stage/build.sh
OUT=.s93/arms/build_dot2 TB_SRC=csrc/rocm/torch_bindings.cpp \
  ATTN_SRC=.s93/arms/dot2.hip bash tools/dcu/attn_stage/build.sh
.venv/bin/python tools/dcu/attn_stage_timing.py --mode prod --kind prefill \
  --ms 8192 --ctxs 8192,32768,131072 --segments 4 --iters 9 --out ...
```
