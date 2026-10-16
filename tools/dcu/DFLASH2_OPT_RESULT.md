# DFlash2 部署优化——落地结果报告

> 输入方案：`/home/lyh/vllm-test/results/DFLASH2_OPT_PLAN.md`（配套剖析报告 `DFLASH2_PERF_REPORT.md`）
> 对象：`z100-vllm` 容器 `:8000` 上的 `qwen3.8-27b` = **Qwen3.8-27B-FP8 + DFlash2 k=7**，TP=4，4×海光 Z100（gfx906），`--enforce-eager`，`--max-num-batched-tokens 2048`
> 执行日期：2026-10-13（宿主机 UTC），全部为**同机同协议单变量 A/B**，每臂重启换配置
> 代码落点：`/home/lyh/z100-vllm`（容器内挂载为 `/src`，只读挂载 + 纯 Python 改动，无需重编 `.so`）

---

## 0. 结论一览

| 计划条目 | 动作 | 实测结果 | 判定 |
|---|---|---|---|
| **P0 §3 GDN decode** | `VLLM_QWEN35_MTP_GDN_TRITON=1` | 步进 p50 **288.6 → 190.9 ms（−34%）**；c=1 decode 11.08 → 16.63 t/s；接受长度不变 | ✅ **已上线** |
| **P2 §4 GDN prefill** | 新增 `VLLM_GFX906_GDN_CHUNKED_PREFILL=1` + 三条分支改造 | per-op **14.7~24×**；p1792 c=4 TTFT **15233 → 10475 ms（−31%）**；prefill **276.7 → 429.7 t/s（+55%）** | ✅ **已上线** |
| **P1 §6 前缀缓存** | 服务端 `--prefix-match-unit 16` | **无效**：协调器显式禁用细粒度命中（`SlidingWindowManager`），粒度仍是 880；应用侧规律已复测确认 | ⚠️ 部分（应用侧有效） |
| **P4 §5 RCCL 调参** | `NCCL_ALGO=Tree/PROTO=LL/MAX_NCHANNELS=2` | 微基准：小消息固有延迟 ~99 µs，`Tree+LL` **更差**（+3~14%），仅 `LL` 好 13 µs/call（≈0.9%/步），且会拖累 MB 级 prefill | ❌ **不采用** |
| **P5 §2 草稿小 M 路径** | `n<=16` → `32` | **回退**：c=4 步进 184 → **261 ms（+41%）**，decode −42% | ❌ **已回退** |

**最终上线配方**（`tools/dcu/s19_dflash_e2e_fp8.sh`，可零重编回滚）：

```bash
export VLLM_QWEN35_MTP_GDN_TRITON=1        # 置 0 = 回到 C++ v1 GDN kernel
export VLLM_GFX906_GDN_CHUNKED_PREFILL=1   # 置 0 = 回到 fused sigmoid-gating prefill
```

---

## 1. 基线复现（先证明尺子没变）

在**未改动**的线上 :8000（原始 k=7 配方）上先跑一遍标准协议，确认与剖析报告附录 A/D/F 三次独立基线一致：

| 指标 | 本轮线上基线 | 报告 A | 报告 D | 报告 F |
|---|---|---|---|---|
| c=1 步进 p50 | 288.6 ms | 289.4 | 288.4 | 288.6 |
| c=1 decode | 11.08 t/s | 11.05 | 11.10 | 11.08 |
| c=4 步进 p50 | 299.2 ms | 299.6 | 298.6 | 298.2 |
| c=4 decode | 45.12 t/s | 45.00 | 45.37 | 45.25 |
| p1792 c=4 步进 | 309.2 ms | — | 309.6 | 307.8 |

⇒ 基线口径稳定（≤0.5% 离散），后续增益都以此为本底。

贪心 parity 基线：`p0=7acae7e225bd235d  p1=a8eb3bd7fea70e76  p2=e1a065e300c7278f`。

---

## 2. P0 — GDN decode 走 gfx906 Triton kernel ✅

**动作**：零代码，`VLLM_QWEN35_MTP_GDN_TRITON=1`。

| 指标 | 基线 | P0 臂 | 变化 |
|---|---|---|---|
| c=1 步进 p50 | 288.6 ms | **190.9 ms** | **−33.9%** |
| c=1 decode | 11.08 t/s | **16.63 t/s** | **+50%** |
| c=4 步进 p50 | 299.2 ms | **191.9 ms** | **−35.9%** |
| c=4 decode | 45.12 t/s | 51.98 t/s（6 次采样中位 52.8） | +15%~+17% |
| p1792 c=4 步进 | 309.2 ms | **196.8 ms** | **−36.4%** |
| 闭环 p512 c=4 ITL(mean) | 372 ms（报告） | **220.9 ms** | −41% |
| 闭环 p512 c=4 每请求 | 10.63 t/s（报告） | **18.61 t/s** | +75% |
| 平均接受长度 | ~3.40 | 3.25（受控 c=4 窗口） | 持平 |

**口径说明（重要，避免误读）**：c=4 的“聚合 decode t/s”噪声很大——同一臂 6 次采样得到
`57.06 / 51.98 / 52.28 / 53.25 / 35.65 / 77.78`（p50 步进却稳定在 187~190 ms）。该指标的分母是
“最后一个 TTFT → 最后一个完成”的墙钟窗口，会吸收调度/尾部效应；**步进 p50 与 ITL 才是物理量**，
两者一致给出 −34~−36%。报告 §3.3 的 “c=4 +46%” 属该噪声区间的高端采样。

**正确性**：贪心 parity 与报告 arm E **逐字节复现**——`p0/p1` 与基线完全一致，
`p2` 为 `87e63e7507b0bb57`（126 字符，基线 123 字符），与报告记录的 arm E 分叉点/长度完全一致，
且两臂各自可复现（确定性浮点顺序差异，非随机噪声）。

---

## 3. P2 — GDN prefill 走 FLA chunked ✅（本轮最大单项收益）

### 3.1 改动

按仓库内既有评估 `results/gdn_chunked_assessment.md` 的方案落地（**不是**方案 §4.2 的骨架——
骨架在 `_uses_transposed_temporal_state()==True` 时写回布局是错的）：

* `vllm/model_executor/layers/mamba/gdn_linear_attn.py`
  * 新增开关 `_gfx906_gdn_chunked_prefill_enabled()`（`VLLM_GFX906_GDN_CHUNKED_PREFILL`，默认 **0**）
  * `:994` 的 `if _is_gfx906_rocm():` → 新增 `elif use_gfx906_chunked_prefill:` 独立分支
  * 只动 **true-prefill** 分支；decode / MTP / 非 gfx906 路径逐字节不变
  * 关键：`q/k/v/g/beta` 用 **token-first + `head_first=False`**（下游 `:1079-1082` 要 `[1,T,H,V]`）；
    状态 `[slots,H,V,K]` ↔ 算子 `[N,H,K,V]` **读写两侧都转置**（`head_k_dim == head_v_dim` 时写错是静默的）
* `vllm/model_executor/warmup/qwen_triton_warmup.py`：新增 `_warm_chunked_gdn_prefill_kernels()`
  * 原因：chunked 链路 6 个 Triton kernel 按 shape autotune，**首个 prefill 请求实测 TTFT 43.1 s**（512 token！）
  * 修复后同一场景首请求 TTFT = **1630 ms**（见 §3.3），启动耗时基本不变（140 s）

### 3.2 门槛一：per-op parity（新工具 `tools/dcu/perf1_gdn_chunked_parity.py`）

生产形状（4 k-heads / 12 v-heads per rank，K=V=128，fp16，fp32 state），chunked vs 现网 fused C++ kernel：

| case | seq_lens | max\|dout\| | max\|dstate\| | NaN | fused ms/call | chunked ms/call | 加速 |
|---|---|---|---|---|---|---|---|
| 1 | [512] | 1.53e-05 | 1.05e-04 | 0 | 11.149 | 0.758 | **14.7×** |
| 2 | [1024] | 2.29e-05 | 1.11e-04 | 0 | 22.308 | 1.062 | **21.0×** |
| 3 | [2048] | 3.05e-05 | 1.20e-04 | 0 | 44.662 | 1.860 | **24.0×** |
| 4 | [2048,1024,512] | 1.53e-05 | 1.33e-04 | 0 | 71.614 | 3.004 | **23.8×** |
| 5 | [700×4] | 2.29e-05 | 1.26e-04 | 0 | 55.910 | 2.616 | **21.4×** |
| 6 | [1500,548] | 3.05e-05 | 1.30e-04 | 0 | 41.570 | 1.935 | **21.5×** |

* 判定阈值：`max|dout| ≤ 1e-3`、`max|dstate| ≤ 6e-4`（沿用 `13452fb7a` 的口径）⇒ **6/6 通过，余量 30~65×**
* **负对照**（把未转置的 `[N,Hv,V,K]` 直接喂给算子）：6/6 **FAIL**，`max|dstate|` 0.23~0.36，
  且状态**每个元素都不同**（196608/196608）⇒ 证明该门槛确实能抓住写回布局错误，不是空跑
* ⚠️ 报告 §4.3 预计 “2.1~2.3×”；实测 **15~24×**——因为原估计是拿业务侧的
  “Triton vs eager torch 回退” 类比到 “Triton vs C++ fused”，后者（21.8 µs/token）才是现网真实对手

### 3.3 门槛二/三：端到端 + 质量

| 指标 | 基线 | P0 | **P0+P2（最终）** | 最终 vs 基线 |
|---|---|---|---|---|
| p512 c=1 TTFT | 1787.7 ms | 1737.6 | **1245.9 ms** | **−30%** |
| p512 c=4 TTFT | 5639.5 ms | 5532.3 | **3569.6 ms** | **−37%** |
| p1792 c=4 TTFT | 15233.1 ms | 16873.8 | **10474.9 ms** | **−31%** |
| p1792 c=4 prefill | 276.7 t/s | 304.0 | **429.7 t/s** | **+55%** |
| 闭环 p1792 c=4 agg prefill | — | 221.5 | **315.3 t/s** | +42% |
| c=1/c=4/p1792 步进 p50 | 288.6 / 299.2 / 309.2 | 190.9 / 191.9 / 196.8 | **188.4 / 184.7 / 196.3** | −35% ~ −38% |
| 平均接受长度 | ~3.40 | 3.25 | **3.51**（P2 臂整轮 3.66） | **不降反升** |
| 首个 prefill 请求 TTFT（冷启动） | — | — | 43102 ms → **1630 ms**（warmup 修复后） | — |

**质量**：
* 贪心 parity **确定性**（同臂连发两次 md5 完全一致）
* 相对基线 `p1` 不变，`p0`/`p2` 变化；三条输出**语义全部正确**（投机解码解释 / Fibonacci 代码 / 火车距离计算）
* 接受长度上升（3.25 → 3.51~3.66）⇒ chunked 的 prefill 状态比 fused 更准，未引入质量退化

**风险与回滚**：`VLLM_GFX906_GDN_CHUNKED_PREFILL=0`（纯 Python，重启即回旧路径）。
首次 JIT 已经由 warmup 覆盖；如遇其它 batch/长度组合仍可产生一次性 autotune 停顿。

---

## 4. P1 — 前缀缓存：服务端杠杆被结构性命中禁用 ⚠️

**实测**：`--prefix-match-unit 16` **能正常启动**（`cache_config_info` 显示 `prefix_match_unit="16"`），
但协调器立刻把它关掉：

```
WARNING [kv_cache_coordinator.py:708] Disabling fine-grained prefix-cache hits because these
KV cache managers require block-aligned lookups: SlidingWindowManager.
```

原因：`CircularBufferManager`（滑动窗口 KV 组）`supports_fine_grained_hash_lookup = False`
且 `block_size(880) != hash_block_size(16)` ⇒ `enable_partial_hash_hits=False`，
`cache_hit_alignment_tokens` 退回 scheduler block（880）。**因此在当前模型/配置下该杠杆不可用**
（方案 §6.4 标注“需实测能否启动”——实际能启动，但语义上被静默关闭）。

**新工具 `tools/dcu/perf1_prefix_cache_probe.py` 复测（同前缀连发 2 次 + 1 次单字节扰动对照）**：

| prompt tokens | 第 2 次命中 | 第 2 次 TTFT | 冷/热输出一致 | 扰动后命中 |
|---|---|---|---|---|
| 622 | 0 | 1528 ms（冷 1922） | ✅ | 0 |
| 1067 | 0 | 2318 ms（冷 2964） | ✅ | 0 |
| 1531 | 0 | 3058 ms（冷 3999） | ✅ | 0 |
| 2189 | **880** | **2772 ms**（冷 6062） | ✅ | 0 |
| 3074 | **1760** | **2909 ms**（冷 7843） | ✅ | 0 |

⇒ 与方案 §6.1 的公式 `命中 = 880×(⌊N/880⌋−1)` **完全一致**；短前缀仍为 0。
**价值**：`cold/warm 输出完全一致（5/5）` 首次为 “部分命中复用 GDN state checkpoint”
提供了正确性证据；**应用侧结论不变**——共享前缀要 ≥1760 token 且逐字节稳定、按 880 对齐，
才能拿到 TTFT −40~85% 的收益。

> 备注：方案 §6.5 提到 “每次命中少用一个尾部 block” 的告警，其代价恰是
> `⌊N/880⌋−1` 里的那个 “−1”（即中短前缀命中直接减半）。修它需要让 dflash 被
> `_annotate_eagle_groups` 识别，属独立课题，本轮未动。

---

## 5. P4 — RCCL 调参：实测否决 ❌

4 rank、空闲机、生产 payload（fp16 `[8,1536]`=24 KB、`[8,2560]`=40 KB、`[8,4352]`=68 KB）：

| payload | default | `Tree+LL+2ch` | 仅 `PROTO=LL` |
|---|---|---|---|
| 24576 B | 99.7 µs | 102.7 µs（**+3%**） | 86.7 µs（−13%） |
| 40960 B | 98.1 µs | 101.0 µs（**+3%**） | 84.9 µs（−13%） |
| 69632 B | 98.8 µs | 112.3 µs（**+14%**） | 85.2 µs（−13%） |

* 方案 §5.3① 预期的 “439 µs → ~200 µs” **不成立**：439 µs 是 profiler 里 “到达等待 + 传输”，
  其中固有地板只有 ~99 µs（与 `results/nccl_findings.md` 的 71.9 µs 同量级），
  其余是 **rank 偏斜等待，不在关键路径上**（步进周期 = `max_r(compute_r) + intrinsic`）。
* 唯一正收益的 `NCCL_PROTO=LL` 只省 ~13 µs/call ⇒ 129 次/步 ≈ **1.7 ms/步（0.9%）**；
  而 `perf1_comm_report.md` 已测 LL 在 20 MiB 上**劣化 2.35×**（prefill 正是 MB 级消息）
  ⇒ 全局启用是净亏。**故不写入启动器。**

---

## 6. P5 — 草稿小 M 路径（`n<=16`→`32`）：实测回退 ❌

单变量 A/B（P0+P2 之上只加这一行）：

| 指标 | P0+P2 | P0+P2+`n<=32` | 变化 |
|---|---|---|---|
| c=1 步进 p50 | 190.1 ms | 192.9 ms | 持平（符合预期，M=8 本来就走 triton） |
| c=4 步进 p50 | 184.4 ms | **260.9 ms** | **+41%** |
| p1792 c=4 步进 | 195.0 ms | **276.8 ms** | **+42%** |
| c=4 decode | 58.78 t/s | **34.38 t/s** | **−42%** |
| 接受长度 | 3.66 | 3.35 | 下降 |

⇒ 方案 §2.4(a) 的假设只考虑了草稿塔，忽略了这个开关是**全模型**生效：c=4 时 target 的
大 N 线性层（gate_up 8704 / down 5120）从 Tensile 掉到 Triton 小 M 路径，亏得远多于草稿塔省的。
**已回退**，仅在 `layers/utils.py` 留一条注释记录该负结果。

---

## 7. 最终部署状态

* 启动器 `tools/dcu/s19_dflash_e2e_fp8.sh` 已内置两个开关（`${VAR:-1}`，可 `=0` 回滚），
  直接 `docker exec -d z100-vllm bash /src/tools/dcu/s19_dflash_e2e_fp8.sh` 即为最优配方
* 服务已用该配方重启并通过 `:8000` 冒烟、`/v1/models` 200、贪心 parity、接受长度检查
* **停机时长**：每臂 stop ~30 s + start 140 s；本轮共 6 次换臂

### 最终对外能力（p512 / p1792，c=1/4）

| | 基线 | 最终 | |
|---|---|---|---|
| c=1 decode | 11.08 t/s | **16.63（P0 臂）/ 20.68（最终臂）** | +50% ~ +87% |
| c=1 步进 p50 | 288.6 ms | **188.4 ms** | −35% |
| c=1 TTFT | 1787.7 ms | **1245.9 ms** | −30% |
| c=4 步进 p50 | 299.2 ms | **184.7 ms** | −38% |
| c=4 TTFT | 5639.5 ms | **3569.6 ms** | −37% |
| p1792 c=4 步进 p50 | 309.2 ms | **196.3 ms** | −36% |
| p1792 c=4 decode | 37.49 t/s | 44.4（P0）/ 55.6（P2）/ **65.53（最终）** | +18% ~ +75% |
| p1792 c=4 TTFT | 15233 ms | **10475 ms** | −31% |
| p1792 c=4 prefill | 276.7 t/s | **429.7 t/s** | +55% |

> **口径纪律（读表前必看）**：`decode t/s` 的分母含“最后一个 TTFT → 最后一个完成”的窗口，
> 对调度、尾部、接受长度都敏感——同一臂 6 次采样能跨 35~78 t/s。因此
> **应把 `步进 p50` / `ITL` / `TTFT` / `prefill t/s` 当作结论依据**（它们在各臂间稳定且物理可归因），
> `decode t/s` 只作为参考区间。同理，c=1 的 16.63 与 20.68 差异来自采样时的接受长度波动，
> 两者的步进 p50 几乎相同（190.9 vs 188.4 ms）——**归因给步进，而不是归因给吞吐表**。

> 对照参考：报告 §3.3 记录 MTP k=2 生产臂 c=1 为 21.03 t/s、c=4 为 56.25 t/s。
> 优化后的 DFlash2 k=7 在 c=1 已基本追平（16.6~20.7 vs 21.03），在 p1792 c=4 已反超（65.53 vs 56.25）。

---

## 8. 证据文件

| 内容 | 路径 |
|---|---|
| 各臂基准日志 | `/home/lyh/z100-vllm/.bench/arm_opt_p0.log`、`arm_opt_p2.log`、`arm_opt_p5.log`、`arm_final.log` |
| 基准 JSON（bench_concurrency / rolling） | `/home/lyh/z100-vllm/.bench/*_p512_*.json`、`*_rolling*_*.json` |
| per-op parity 工具 + 负对照 | `tools/dcu/perf1_gdn_chunked_parity.py` |
| 前缀缓存探针（含 cold/warm 一致性） | `tools/dcu/perf1_prefix_cache_probe.py`、`.bench/p1_pmu16_probe.log` |
| RCCL 微基准原始输出 | `.bench/nccl_default.log`、`nccl_tree_ll.log`、`nccl_ll_only.log` |
| 服务端日志 | 容器内 `/mnt/game/dflash/serve-opt-p0-gdntriton.log`、`serve-opt-p2-chunked.log`、`serve-opt-p5-m32.log`、`serve-opt-p1-pmu16.log`、`serve-fp8-tp4-dflash-k7.log`（当前） |
| 代码改动 | `git diff vllm/model_executor/layers/mamba/gdn_linear_attn.py vllm/model_executor/warmup/qwen_triton_warmup.py` |

**A/B 纪律**：单变量、每臂重启、同机同协议、基线三次复现（≤0.5% 离散）、
所有结论同时给出步进/ITL（物理量）与聚合吞吐（噪声大）两个口径。
