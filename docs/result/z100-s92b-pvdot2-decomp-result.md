# S92b PV dot2 分解探针 —— 结果（单一文档）

日期：2026-11-14。状态：**结案（出口 A，边界 +1.98%）**。
设计：[z100-s92b-pvdot2-decomp-design.md](../design/z100/s92b-pvdot2-decomp/z100-s92b-pvdot2-decomp-design.md)；
施工：[z100-s92b-pvdot2-decomp-construction.md](../design/z100/s92b-pvdot2-decomp/z100-s92b-pvdot2-decomp-construction.md)。
本文件汇总 WP0–WP4 全部结果、分段测速、ISA 与 counters；`.s92b/` 只保留原始数据，
结论以本文为准。

---

## 0. 一句话

**S92 的 +9.0% 是捆绑数字：PV drain 调度（V3+V4）占 5.77 pt，staging 形态及其
调度伴生占 1.26 pt；纯 pair 布局残税只有 +1.98%（P5 131k 939.398 ms；同日
+2% 门 939.545 ms，施工图固定门 939.7 ms）。出口 A 成立（边界，余量
0.147 ms / 0.016%；按固定门 0.30 ms）——S92 出口 B 撤销，改判“工程形态税”，
dot2-PV（S92 WP1 算术臂）复活。**

---

## 1. 假设判决

| id | 假设 | 实测（131k，prod op） | 判决 |
|---|---|---|---|
| HB1 | V4 主因：drain 软件流水把 pv 窗口 +89.3 ms 收回 ≤+15 ms | P3 pv 536.37→**448.35**；P5 **438.57**；目标 ≤460 | **绿** |
| HB2 | V2 次因：staging 回生产形态把 load 窗口 +36.7 ms 收回 ≤+5 ms | P4a load 198.27→**197.22**（目标 ≤165）；GB/s 1122→1128（目标 ~1378） | **红** |
| HB3 | 剩余真布局税（V1）≤ +2% | P5 **939.398** vs 生产 **921.123** = **+1.98%**；门 939.545（固定 939.7） | **绿（边界）** |
| HB4 | 合成臂数值逐位同 | 四形状 md5 全等；stage op 逐位同 | **绿** |

P4b（int4 载入）**未执行**：KV token stride = 520 B，`520 % 16 = 8`，
奇数 key 的基址只有 8 B 对齐，int4 会错位；P4a 已覆盖“回生产全局形态”变量。

---

## 2. 零改动锚点（开工/收工一致）

| 项 | 值 |
|---|---|
| `csrc/rocm/attnd_inner_kvint8.hip` | md5 `5926429c7f541212488d934086b4203f` |
| `csrc/rocm/attnd_inner.hip` | md5 `b93db390a0fe50e83f19642605bfc729` |
| 生产 `.so` | `.s91/arms/_rocm_C_kvint8fix.abi3.so` md5 `5af8b848f08e` |
| 策略指纹 | `FP 8bf8b075a6469af0` |
| 服务 | 开工前 active；收工 `z100-vllm-mtp2` active，health 200（加载 220 s） |

生产 `csrc/`、`vllm/`、服务 argv/env 全程只读。`.s92/` 只读引用。

---

## 3. 方法

- 环境：`.venv/bin/python`；`source .s34/base_env.sh`；计时 `torch.cuda.Event`，
  `torch.cuda.set_device(0)`。
- 构建：`tools/dcu/attn_stage/build.sh`（`TB_SRC=csrc/rocm/torch_bindings.cpp`
  强制；`ATTN_SRC=` 换臂；stage 变体另加 `STAGE_HIP=`/`STAGE_OPS=`）。
- 计时口径：prefill 一律 **seg=4**（S92/S92b 口径），decode **seg=256**；
  9 iters、warmup 3、取中位；判决只用 **prod-op wall**，stage 窗口只作归因。
- 数值门：四形状 `segm md5` 对生产逐位相等 + `stage_bitexact=true`。
- A/B：P5 与生产进程外交替 ×3（每轮 prod→P5，各 9 iters），取 3 轮中位的
  中位；P2/P3/P4a 为同窗口单轮 9 iters。
- counters：`rocprof` v1 RPL，单 kernel、1 次 dispatch；`TCC_*` 在本栈
  `Context Create failed`，用 `SQ_INSTS_VMEM_RD` 作全局读请求代理；
  已用 8k 形状校验计数器随 tile 数线性缩放（131k/8k=8.5×）。

---

## 4. 全臂 wall（prod op，9 iters 中位）

### 4.1 prefill（m=8192，seg=4）

| 臂 | 8k ms | 32k ms | 131k ms | Δ vs 生产（8k/32k/131k） |
|---|---:|---:|---:|---|
| 生产 `5af8b848` | 109.83 | 272.24 | 921.12 | — |
| P2（S92 pair2 重建） | 119.21 | 296.41 | 1004.13 | +8.54% / +8.88% / **+9.01%** |
| P3（drain 流水） | 113.21 | 281.11 | 950.98 | +3.08% / +3.26% / +3.24% |
| P4a（staging 形态） | 114.61 | 284.95 | 964.48 | +4.35% / +4.67% / +4.71% |
| **P5（P3+P4a）** | **111.73** | **277.55** | **939.398** | **+1.73% / +1.95% / +1.98%** |

131k 门线：同日基线×1.02 = 939.545 ms（施工图固定门 939.7 ms）；
P5 余量 **0.147 ms**（按固定门 0.30 ms）。

### 4.2 decode（ctx=65k，seg=256）

| 臂 | M=1 | M=3 | M=8 |
|---|---:|---:|---:|
| 生产 | 0.2328 | 0.3920 | 0.5643 |
| P2 | 0.2546 (+9.34%) | 0.4235 (+8.04%) | 0.6102 (+8.13%) |
| P3 | 0.2402 (+3.16%) | 0.4026 (+2.69%) | 0.5805 (+2.86%) |
| P4a | 0.2429 (+4.33%) | 0.4075 (+3.96%) | 0.5883 (+4.26%) |
| P5 | 0.2387 (+2.54%) | 0.3973 (+1.35%) | 0.5731 (+1.56%) |

决策门只对 prefill 131k；decode 仅记录。

---

## 5. prefill 分段测速（同日 stage op，9 iters 中位）

### 5.1 131k 全臂（ms；括号为占 wall 份额）

| 臂 | wall | load | qk | sm | pv | wait | load GB/s | pv TF/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 生产 | 929.29 | 161.49 (17.4%) | 287.29 (30.9%) | 0.40 | 446.85 (48.1%) | 33.25 (3.6%) | 1378 | 14.76 |
| P2 | 1009.53 | 198.27 (19.6%) | 254.80 (25.2%) | 0.40 | 536.37 (53.1%) | 19.69 (2.0%) | 1122 | 12.30 |
| P3 | 957.01 | 202.49 (21.2%) | 282.86 (29.6%) | 0.40 | 448.35 (46.8%) | 22.91 (2.4%) | 1099 | 14.71 |
| P4a | 972.88 | 197.22 (20.3%) | 280.54 (28.8%) | 0.40 | 462.63 (47.6%) | 32.08 (3.3%) | 1128 | 14.26 |
| **P5** | **948.27** | **201.56 (21.3%)** | **284.15 (30.0%)** | **0.40** | **438.57 (46.2%)** | **23.58 (2.5%)** | **1104** | **15.04** |

### 5.2 8k / 32k（生产 vs P5）

| ctx | 臂 | wall | load | qk | sm | pv | wait | load GB/s | pv TF/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8k | 生产 | 110.73 | 19.55 (17.7%) | 34.13 (30.8%) | 0.05 | 53.05 (47.9%) | 3.95 | 1339 | 7.77 |
| 8k | P5 | 112.67 | 24.25 (21.5%) | 33.70 (29.9%) | 0.05 | 51.88 (46.1%) | 2.79 | 1079 | 7.95 |
| 32k | 生产 | 274.70 | 48.12 (17.5%) | 84.78 (30.9%) | 0.12 | 131.87 (48.0%) | 9.82 | 1360 | 12.51 |
| 32k | P5 | 280.28 | 60.19 (21.5%) | 83.80 (29.9%) | 0.12 | 129.23 (46.1%) | 6.94 | 1087 | 12.76 |

### 5.3 chunk 扫描（ctx=32768，seg=4）

| m | 生产 wall / load GB/s | P5 wall / load GB/s | P5 份额 load/qk/pv/wait |
|---:|---:|---:|---|
| 1024 | 29.32 / 1309 | 29.91 / 1053 | 21.4 / 29.9 / 46.2 / 2.5 |
| 2048 | 59.79 / 1322 | 61.00 / 1059 | 21.5 / 29.9 / 46.1 / 2.5 |
| 4096 | 124.70 / 1344 | 127.25 / 1077 | 21.5 / 29.9 / 46.1 / 2.5 |
| 8192 | 274.56 / 1360 | 280.33 / 1088 | 21.4 / 29.9 / 46.1 / 2.5 |

份额在 chunk 1k–8k、ctx 8k–131k 恒定。

**分段结论**：P5 的 PV 窗口已低于生产（438.57 vs 446.85，+1.9% TF/s）；
残差集中在 load 窗口（+40.1 ms，GB/s 1378→1104）。qk/wait 受窗口漂移影响，
只作定性（P2 的 qk −32 ms 为已知漂移）。

---

## 6. S92 +9.0% 的分解账（131k，同日 prod op）

| 项 | Δ ms | Δ pt | 证据 |
|---|---:|---:|---|
| S92 捆绑判决（P2） | +83.0 | +9.01 | 1004.13 vs 921.12 |
| V3+V4：drain 软件流水 | −53.1 | −5.77 | P3 950.98；pv 窗口 −88.0 ms |
| staging/调度伴生 | −11.6 | −1.26 | P5 939.40 |
| **残税（P5 vs 生产）** | **+18.3** | **+1.98** | load +40.1 被 pv −8.3 / wait −9.7 / qk −3.1 抵消 |

关于 V2：P4a 把全局读形态换回生产（16 线程/行 × 4 dword），load 窗口仍
197.22 ms / 1128 GB/s（P2 198.27/1122）——**单换全局形态不足以收回 load
窗口**；P4a 单臂的 −39.6 ms 来自 pv 窗口 −73.7 ms 与 qk/wait 漂移的伴生调度。
load 税的主体是 pair staging 的交换/打包（见 §8）。

---

## 7. 静态红线 / ISA（ILb1ELi4E 实例）

| 臂 | vgpr | spill | LDS | fma_mix(静) | dot4 | read_b128 | write_b128 | bpermute | waitcnt(静) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 生产 | 82 | 0 | 20608 | 160 | 32 | 28 | 11 | 24 | 61 |
| P2 | 76 | 0 | 20608 | 160 | 32 | 28 | 11 | 24 | 62 |
| P3 | 76 | 0 | 20608 | 288* | 32 | 44 | 11 | 32 | 76 |
| P4a | 81 | 0 | 20608 | 160 | 32 | 28 | 11 | 48 | 86 |
| P5 | 70 | 0 | 20608 | 288* | 32 | 44 | 11 | 56 | 99 |

\* 静态数随 unroll 4 放大；每 tile/lane 动态仍是 256 条 fma_mix。
vgpr ≤90、spill 0、LDS 20608（3 CTA/CU）全绿；P3/P5 的 drain SASS 已确认
`ds_read_b128` 与 fma 群交错、waitcnt 落 `lgkmcnt(2)/(1)`（≥2 条在飞）。

---

## 8. counters 定因（2026-11-14）

工具：`rocprof` v1（v2 PMC 在本栈会挂起）；单 kernel、131k、1 次 dispatch；
`TCC_*` 全部 `Context Create failed`，无法取 DRAM 字节。计数器已用 8k 校验
随 tile 数线性缩放（131k/8k = 8.5×），非陈旧值。

### 8.1 SQ（wave 级，全 kernel）

| counter | 生产 | P2 | P5 | P2/生产 | P5/生产 |
|---|---:|---:|---:|---:|---:|
| SQ_WAVES | 49,152 | 49,152 | 49,152 | 1.000 | 1.000 |
| SQ_WAVE_CYCLES | 260.08e9 | 283.51e9 | 265.27e9 | 1.090 | **1.020** |
| SQ_INSTS | 80.68e9 | 81.91e9 | 79.77e9 | 1.015 | 0.989 |
| SQ_INSTS_LDS | 8.610e9 | 8.610e9 | 9.466e9 | 1.000 | **1.099** |
| SQ_LDS_BANK_CONFLICT | 7.701e9 | 9.412e9 | 7.701e9 | **1.222** | **1.000** |
| SQ_LDS_IDX_ACTIVE | 67.17e9 | 68.88e9 | 70.59e9 | 1.025 | 1.051 |
| SQ_INSTS_VALU | 59.16e9 | 57.99e9 | 56.49e9 | 0.980 | 0.955 |
| SQ_INSTS_VMEM_RD | 427.9e6 | 641.8e6 | 427.9e6 | **1.500** | **1.000** |
| SQ_ACTIVE_INST_LDS | 12.45e9 | 12.59e9 | 14.27e9 | 1.011 | **1.146** |
| SQ_ACTIVE_INST_ANY | 82.45e9 | 83.44e9 | 80.95e9 | 1.012 | 0.982 |
| SQ_BUSY_CYCLES | 5.811e9 | 6.335e9 | 5.926e9 | 1.090 | 1.020 |

### 8.2 Stall（wave 级）

| counter | 生产 | P2 | P5 | P2/生产 | P5/生产 |
|---|---:|---:|---:|---:|---:|
| SQ_WAIT_CNT_LGKM | 75.41e9 | 118.45e9 | 95.90e9 | **1.571** | **1.272** |
| SQ_WAIT_ANY | 114.68e9 | 154.04e9 | 132.82e9 | 1.343 | 1.158 |
| SQ_WAIT_BARRIER | 10.04e9 | 5.656e9 | 8.110e9 | 0.563 | 0.808 |
| SQ_WAIT_INST_LDS | 19.46e9 | 11.08e9 | 16.38e9 | 0.569 | 0.842 |
| SQ_WAIT_INST_VALU | 42.72e9 | 34.11e9 | 34.40e9 | 0.798 | 0.805 |
| SQ_LDS_DATA_FIFO_FULL | 0.851e9 | 1.319e9 | 1.159e9 | 1.550 | **1.361** |
| SQ_LDS_CMD_FIFO_FULL | 8.327e9 | 1.891e9 | 5.885e9 | 0.227 | 0.707 |
| SQ_VALU_STARVE | 24.53e9 | 32.56e9 | 28.91e9 | 1.327 | **1.179** |
| SQ_WAVE_READY | 145.39e9 | 129.51e9 | 132.46e9 | 0.891 | 0.911 |

### 8.3 判决

1. **P5 的 load 税不是 KV 带宽**：全局读指令数与 bank conflict 均与生产
   **逐位相等**；多出来的是每 tile 8 条 even/odd `ds_bpermute`（动态
   LDS insts +856M = +9.9%），LDS active +14.6%，并转成 LGKM 等待 +27.2%、
   LDS data FIFO +36.1%、VALU starve +17.9%。load 窗口的 1378→1104 GB/s
   是“窗口里多了 LDS 工作”，不是读得慢。
2. **P2 是另一种税**：0 bpermute（LDS insts = 生产），但 8 B 双行载入使
   VMEM_RD +50%、bank conflict +22.2%、LGKM 等待 +57.1%。
   P4a/P5 的“生产全局形态 + 寄存器交换”方向正确，P2 形态不可取。
3. **bank swizzle 无收益**：P5 的 conflict 已与生产完全相同。
4. 直接修复：**8→4 bpermute**（每 lane 只发伙伴真正需要的 8 维，4 个 dword），
   动态 LDS 增量 +856M→+428M；预测 wall −5~12 ms（P5 939.4→~928-934，
   +0.5~1.2%），逐位同。**尚未实施/计时**。
5. load 的下一大杠杆仍是 **kBlockM 16→32 减半 KV 重读**（生产侧 load 已近
   带宽顶）；但 P5 当前 load 是 LDS 限，应先做第 4 条再评估。

> **S93 更新（2026-11-14）**：第 4 条已实施为 P5b——动态 LDS insts
> +856M→+428M，load +40.1→+31.8 ms，逐位同，131k A/B 922.65 ms。
> 第 5 条 kBlockM 32 纸面红（LDS ~40.25 KB ⇒ 1 CTA/CU），落排除表。

---

## 9. 机制结论

- **drain 调度是 S92 +89 ms pv 窗口的主因**：P2 每个 `ds_read_b128` 后紧跟
  `lgkmcnt(0)`，LDS 延迟串行暴露；P3 用 unroll 4 + 2 深预发后，pv 窗口
  536.37→448.35（−88.0 ms），每 tile 动态 256 条 fma 不变。
- **pair 布局的 LDS 冲突无劣化**：vslot 消费端与生产同为 2-way，counters 的
  bank conflict 逐位相等。
- **staging 全局形态不是 load 税的全部**：P4a 换成生产形态后 load 窗口不动；
  load 税的主体是 pair staging 的 even/odd 交换 + 打包。P4a 单臂的收益来自
  pv 窗口的伴生调度，不是结构性 staging 收益。
- **P5 的 pv 已优于生产**：438.57 ms / 15.04 TF/s vs 446.85 / 14.76。

---

## 10. 出口与下一步

- **出口 A（边界）**：P5 131k +1.98%（939.398 ≤ 939.545，同日 +2% 门；
  固定门 939.7），布局税 ≤+2%。S92 出口 B 撤销，排除表 “dot2-PV = 布局税否证（+9.0%）” 改判为
  “+9.0% 为工程形态税，布局内在税 +1.98%”；dot2-PV 复活。
- 余量 0.147 ms（0.016%；按施工图固定门 939.7 为 0.30 ms），按图成立但属边界。
- 推荐顺序：
  1. **P5b：8→4 bpermute**（低风险、逐位同，预测 +0.5~1.2%）；
  2. **S93 dot2 算术臂**（PV 46.2%，设计预期内核 −3~8%；w fp16 数值门先行）；
  3. **kBlockM 16→32**（KV 重读减半，理论 −8.8%；需先做占用/vgpr 探针）。

---

## 11. 复现 / 重建 / 回滚

```bash
# 环境
source .s34/base_env.sh && unset HIP_VISIBLE_DEVICES

# 重建任一臂（P2/P3/P4a/P5）
OUT=.s92b/arms/build_p5 TB_SRC=csrc/rocm/torch_bindings.cpp \
  ATTN_SRC=.s92b/arms/p5.hip bash tools/dcu/attn_stage/build.sh

# stage 变体：P3/P5 用 pipeline 生成器；P2/P4a 用 .s92 原生成器
.venv/bin/python .s92b/make_stage_pipeline.py --src .s92b/arms/p5.hip \
  --out .s92b/arms/p5_stage.hip

# prod-op 计时（prefill 口径 seg=4）
.venv/bin/python tools/dcu/attn_stage_timing.py --mode prod --kind prefill \
  --ms 8192 --ctxs 8192,32768,131072 --segments 4 --iters 9 --check 0 --out x.json

# counters（单 kernel，仅 SQ/stall；TCC 在本栈不可用）
rocprof -i .s92b/counters/pmc_sq.txt -o out.csv \
  .venv/bin/python tools/dcu/attn_stage_timing.py --mode prod --kind prefill \
  --ms 8192 --ctxs 131072 --segments 4 --iters 1 --warmup 0 --check 0 --out y.json
```

生产未改：`.so` 指回 `.s91/arms/_rocm_C_kvint8fix.abi3.so`。误换臂回滚：

```bash
ln -sfn .s91/arms/_rocm_C_kvint8fix.abi3.so vllm/_rocm_C.abi3.so
systemctl restart z100-vllm-mtp2
```

---

## 12. 数据与文件

| 类别 | 路径 |
|---|---|
| 源码 | `.s92b/arms/p2.hip`、`p3.hip`、`p4a.hip`、`p5.hip`；`*_stage.hip` |
| 构建 | `.s92b/arms/build_{p2,p3,p4a,p5}{,_stage}` |
| 锚点 | `.s92b/wp0-anchor-pre.md`、`.s92b/wp0-anchor-post.md` |
| wall | `.s92b/wp0-prod-{pre,dec}.json`、`wp0-p2-*`、`wp1-p3-*`、`wp2-p4a-*`、`.s92b/ab/{prod,p5}-*-{1,2,3}.json` |
| 分段 | `.s92b/wp0-prod-stage-pre.json`、`wp0-p2-stage-pre.json`、`wp1-p3-stage-pre.json`、`wp2-p4a-stage-pre.json`、`wp3-p5-stage-pre.json`、`wp0-prod-chunks.json`、`wp3-p5-chunks.json` |
| 数值门 | `.s92b/chk-*.json`（md5 见下） |
| counters | `.s92b/counters/{prod,p2,p5}_131k_{sq,stall}.csv`、`prod_8k_sq.csv`、`pmc_*.txt` |
| 工具 | `.s92b/isa_census.py`、`.s92b/make_stage_pipeline.py`、`.s92b/counters/README.md` |
| 台账 | `.s92b/README.md` |

数值门四形状 md5：prefill 32k `cac9a869`、prefill 131k `ea243f9b`、
decode M=1 `7a53eece`、decode M=8 `608489bf`；P2/P3/P4a/P5 全部相等，
stage op `stage_bitexact=true`。

---

## 13. S92 勘误摘要

已追加到 [z100-s92-pv-dot2-result.md](z100-s92-pv-dot2-result.md) §8：

- Δ 口径统一为 prod-op：`1004.38 / 921.29 = +9.02%`（原文 §2 混用 stage 口径）。
- “布局税 +9.0%”改判为“工程形态税 ~7.0% + 布局内在税 +1.98%”；
  S92 出口 B 撤销，dot2-PV 复活。
- 排除表行 “dot2-PV = 布局税否证（+9.0%）” 加注：该数字为捆绑值，
  不再作为 dot2-PV 的否证依据。
- 生产零改动、服务收工态与原文一致。
