# S95 QK 读-算流水结果：预发只把时间从 qk 搬到 pv

日期：2026-11-14。状态：**收工（出口 C：QK 读-算流水不采纳）**。
设计/施工：[设计图](../design/z100/s95-qk-pipeline/z100-s95-qk-pipeline-design.md)、
[施工图](../design/z100/s95-qk-pipeline/z100-s95-qk-pipeline-construction.md)、
[证据](../design/z100/s95-qk-pipeline/z100-s95-qk-pipeline-evidence.md)。
台账 `.s95/`，原始数据 `.s95/*.json`、`.s95/ab/*.json`。

## 一句话

**k4h（QK 环 1 深读预发，+8 vgpr）逐位同、census 绿；131k A/B ×3
882.408 → 875.727 ms（−0.76%），未过 HA2 门（≤850.0）。stage 显示
qk 286.32 → 242.24 ms（−15.4%），但 pv 391.84 → 432.23 ms（+10.3%），
wall 基本不动：3 CTA 已把 LDS 读延迟掩蔽，预发只是把停顿从 qk 挪到 pv。**

## 0 环境与锚点

- 服务按用户令**停**（`z100-vllm-mtp2` inactive，卡 0 独占微测）；收工
  `systemctl start` 恢复 active。
- 源零改动：`attnd_inner_kvint8.hip` `5926429c`、`attnd_inner.hip`
  `b93db390`；生产 `.so` `7d2ed2a2`。
- dot2 同日复测：8k 105.21 / 32k 260.78 / **131k 882.52**（±0.5% 锚定成立）。
- 裁决口径：prod op，A/B 进程外交替 ×3，取中位的中位。

## 1 臂表（131k，prod op）

| 臂 | 改动 | vgpr/spill | 131k (ms) | Δ vs dot2 | 数值 |
|---|---|---|---|---|---|
| dot2（基底/生产） | — | 70/0 | 882.408 | — | 基准 |
| **k4**（4 深预发） | 8 uint4 FIFO | 84/**12** | 未计时 | 静态红 | — |
| **k4h**（1 深 +8） | 消费 c 时预发 c+1 | 80/0 | **875.727** | **−0.76%** | 逐位同 |
| k4h4（尾预发守卫） | k4h + `c<14` | 76/0 | 875.770 | −0.76% | 逐位同 |
| k4r1（单行 +4） | 只预发 kv0 | 73/0 | 883.673 | +1.15% | 逐位同 |
| k128k（K 布局单独） | `(u&1)<<3`→`(u&3)<<2` | 70/0 | 884.599 | +0.24% | 逐位同 |

- k4 的 +32 vgpr 撞 84 红线（spill 12），按 R1 降档到 k4h（+8）。
- k4h 两条反证：k4r1（少一半在飞）直接变慢；k4h4（去掉 2 条尾部死预发）
  与 k4h 同速 → 收益/损失都不在“在飞条数”。
- k4q 未做：k4h 875.7 不在设计图规定的 850–870 触发带，按图跳过。
- k128 全臂未施工：Q 读 `q_lds + pair*256 + t*16` 按字面在 `t+c≥16`
  时跨入相邻 pair 的 256 B 行（`t+c≤30`），不是同一 Q 行；K 布局单独成
  k128k 复证“冲突是否压制吞吐”，结果中性。

## 2 数值门

| 形状 | dot2 | k4h |
|---|---|---|
| prefill 8192@8192 | `b1fa6e88` | `b1fa6e88` |
| decode M=1 @65k | `b2beb5a0` | `b2beb5a0` |
| decode M=8 @65k | `ba0e1cce` | `ba0e1cce` |

decode 65k 9 iters：k4h 0.2152 / 0.3672 / 0.5306 ms vs dot2
0.2195 / 0.3693 / 0.5368（−2.0% / −0.6% / −1.2%），记录门通过。

## 3 静态与机制（HA1）

| 臂 | vgpr | spill | LDS | CTA/CU | b128 | waitcnt | lgkmcnt(0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dot2 | 70 | 0 | 20608 | 3 | 44 | 95 | 49 |
| k4h | 80 | 0 | 20608 | 3 | 46 | 94 | 45 |

disasm（`ILb1ELi4E`）实证：QK 环由 `waitcnt(0)` 每 4 条 dot4 排空，变为
`waitcnt(2)` 稳态 3 条在读；HA1 绿。k4h 的 2 条额外 b128 是尾部死预发。

## 4 prefill 各阶段测速（stage op，seg=4）

### 131k（ms）

| 臂 | wall | load | qk | sm | pv | wait |
|---|---:|---:|---:|---:|---:|---:|
| dot2 | 889.70 | 191.12 | 286.32 | 0.40 | 391.84 | 20.02 |
| k4h | 888.38 | 190.73 | **242.24** | 0.41 | **432.23** | 22.77 |
| Δ | −1.32 | −0.39 | **−44.08 (−15.4%)** | +0.01 | **+40.39 (+10.3%)** | +2.75 |

有效速率：qk 5.76 → **6.81 TMAC/s**（+18.2%）；pv 4.21 → **3.82 TMAC/s**
（−9.3%）；load 1164 → 1166 GB/s（不变）。

### 8k / 32k（ms）

| ctx | 臂 | wall | load | qk | sm | pv | wait |
|---|---|---|---:|---:|---:|---:|---:|
| 8k | dot2 | 105.75 | 22.95 | 33.95 | 0.05 | 46.43 | 2.37 |
| 8k | k4h | 105.02 | 22.75 | 28.59 | 0.05 | 50.95 | 2.68 |
| 32k | dot2 | 262.86 | 56.90 | 84.44 | 0.12 | 115.50 | 5.90 |
| 32k | k4h | 262.50 | 56.92 | 71.41 | 0.12 | 127.34 | 6.71 |

k4h 的 qk 降幅在三个 ctx 一致（−16%~−18%），pv 涨幅一致（+10%），
wall 基本抵消。

## 5 判决

- qk 5.76/19.26 = 30% 达峰**不是读延迟**：3 CTA/CU 已把 LDS 读延迟
  掩蔽。预发减少的 qk 停顿被 pv 等价收回，prod wall 仅 −0.76%，
  HA2（≤850.0）红。
- k128 的 K 布局单独实测中性（+0.24%），不支持“bank 冲突压制预发收益”；
  Q 逐 lane 偏移按字面跨行，设计图 §1.3 的 Q 读法不成立。
- 出口 C：QK 读-算流水“结构性饱和”落排除表；attention 内核层
  QK/PV/load 三段收口。生产符号链接仍指 `.s93/arms/_rocm_C_dot2.abi3.so`。
- 后续若要再动 QK，只能动 LDS 流量/指令形态（如减读、减冲突发射），
  不再加深预发。

## 6 复现

```bash
OUT=.s95/arms/build_k4h TB_SRC=csrc/rocm/torch_bindings.cpp \
  ATTN_SRC=.s95/arms/k4h.hip bash tools/dcu/attn_stage/build.sh
ln -sfn /home/lyh/z100-vllm/.s95/arms/build_k4h/_rocm_C.abi3.so vllm/_rocm_C.abi3.so
.venv/bin/python tools/dcu/attn_stage_timing.py --mode prod --kind prefill \
  --ms 8192 --ctxs 131072 --segments 4 --iters 9 --check 0 --out .s95/ab/k4h-131k-1.json
ln -sfn /home/lyh/z100-vllm/.s93/arms/_rocm_C_dot2.abi3.so vllm/_rocm_C.abi3.so
```
