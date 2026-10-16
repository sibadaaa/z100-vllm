# S18c GEMM 派发消除 `M` 线性代价（S18-4 关键路径突破）— 执行结果

> 归档时间：2026-10-12
> 执行者：Lead（微基准 + op 级分派验证）
> 对应设计文档：docs/design/z100-18-verify-gemm-efficiency.md、docs/design/z100-18-mtp-optimization.md §0.2
> 前一档：[z100-18b](z100-18b-mtp-first-baselines-and-cost-scaling-result.md)
> 状态：🔶 内核层已闭环；**端到端 A/B 待补**（卡被 S18-1 的 TP=2/INT8 电池占用）

## 1. 目标

S18b 定位到 MTP 净负的根因是 `M=1+k` 的 verify 代价随 M 近似线性（FP8 block-scaled GEMV：M=2 ≈ 2.1×、M=4 ≈ 3.9×）。本步要回答：**这是内核本身不可免的，还是分派策略选错了？**

## 2. 改动

| 文件 | 位置 | 改动 |
|---|---|---|
| `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py` | `:41`（`import os`）、`:60` | `_GEMV_MAX_M = int(os.getenv("VLLM_DCU_FP8_GEMV_MAX_M", "8"))` —— **默认值不变**（仍是旧策略），可用环境变量把 `M≥2` 全部路由到 tiled GEMM |
| `vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py` | `:78` | `_GEMV_MAX_M = int(os.getenv("VLLM_DCU_INT8_GEMV_MAX_M", "8"))`（同上，INT8 侧） |
| `tools/dcu/s18_fp8_block_mprobe.py` | 重写 | 增加 `--modes gemv,gemm`、`--threads-list`、`--gemm-cfg`；行一致性检查改为"同 shape 只生成一次数据、按 M 切片"（修掉 S18b §3.4 登记的空检查） |
| `tools/dcu/s18_dispatch_check.py` | 新增 | 在 op 级（`torch.ops.vllm.fp8_emu_mm_block`）验证环境变量确实改变了分派与耗时 |

`.so` **未改动**，纯 Python 分派层。

## 3. 原始证据

### 3.1 微基准：GEMM vs GEMV（`8704×5120`，卡 2，iters=30，synthetic block-scaled fp8）

```
   8704x5120    gemv  128  1   0.5997     74     1.00
   8704x5120    gemv  128  2   1.1421     39     1.90
   8704x5120    gemv  128  4   2.2783     20     3.80
   8704x5120    gemv  128  6   3.2591     14     5.43
   8704x5120    gemv  128  8   4.4278     10     7.38
   8704x5120    gemm    0  1   0.5884     76     0.98
   8704x5120    gemm    0  2   0.5882     76     0.98
   8704x5120    gemm    0  4   0.5883     76     0.98
   8704x5120    gemm    0  6   0.5899     76     0.98
   8704x5120    gemm    0  8   0.5900     76     0.98
```

- **GEMM 路径在 `M=1..8` 完全平坦**（0.5884 → 0.5900 ms，波动 0.3%）；GEMV 在 M=8 涨到 **7.38×**；
- 另一个形状 `5120×1536`：GEMV（nt=128）M=1 = **0.1326 ms**，GEMM = 0.1733 ms ⇒ **M=1 时 GEMV 反而快 30%** ⇒ 最优策略是 **`M=1` 用 GEMV、`M≥2` 用 GEMM**（即 `_GEMV_MAX_M=1`），而不是"全用 GEMM"；
- 首轮（nt=64/128/256 扫描，`8704×5120` + `5120×1536`，M=1..4）结论一致：**nt 对 M 标度无实质影响**（nt=128 略优），把 M 从 GEMV 挪到 GEMM 才是决定性因素。

### 3.2 op 级验证（`s18_dispatch_check.py`，卡 3，`VLLM_DCU_FP8_GEMV_MAX_M=1`）

```
[check] VLLM_DCU_FP8_GEMV_MAX_M=1 -> module _GEMV_MAX_M=1
[check] M=1 op-level 0.6202 ms  weight_gbs=72
[check] M=2 op-level 0.5839 ms  weight_gbs=76
[check] M=3 op-level 0.5831 ms  weight_gbs=76
[check] M=4 op-level 0.5828 ms  weight_gbs=76
```

⇒ 生产实际调用的 `vllm::fp8_emu_mm_block`（而非裸 kernel）在 `M≥2` 时已走 GEMM，**且比 M=1 的 GEMV 还快 6%**。

### 3.3 正确性

- `rel_err = 0.00e+00`（GEMM 路径，vs fp32 参考，M=1..8 全为 0）；GEMV 路径在 `5120×1536` 上为 `5.96e-05`（fp16 累加顺序差异，量级符合预期）；
- 行一致性检查（修好后）：`M=2/3/4 vs M=1: OK`（同一份数据切片）⇒ 没有模板 padding 污染。

## 4. 门禁判定（对照设计文档 z100-18-verify-gemm-efficiency.md）

| 门禁 | 判定 | 证据 |
|---|---|---|
| G-K1 绝对吞吐（M=1 ≥400 GB/s） | ❌ 未达（仍 72–79 GB/s） | §3.1；绝对提速仍需新内核（S18-4c），但**不再是 MTP 的前置** |
| G-K2 M 摊销（M=2 ≤1.2×、M=4 ≤1.6×） | ✅ **通过**（0.98× / 0.98×） | §3.1 |
| G-K3 数值（rel_err ≤1e-2） | ✅ `0.00e+00` | §3.3 |
| G-K4 无形状回退 | ✅ M≥2 全形状 GEMM 更快或持平；M=1 保持 GEMV | §3.1 两个形状 |
| 端到端（≥1.3×） | ⏳ **待测** | 需在卡空出后跑 `A1'(k=1, GEMM 派发)` vs `A0=123.32 ms` |

## 5. 影响与预判

- **A0（非投机）不受影响**：其所有调用都是 `M=1`，新旧策略下都走 GEMV ⇒ 123.32 ms/token 直接作为对照；
- 预判端到端：若 verify 的 GEMM 成本与 M=1 持平，则 `ms/step(k=1) ≈ 123 + (draft 头 + 元数据 + 采样)`，取 `MAL=1.75` ⇒ **TPOT ≈ 70–80 ms（1.5–1.75×）**，k=2（MAL 2.29）更好；
- 这条改动对 **INT8 侧同样适用**（`VLLM_DCU_INT8_GEMV_MAX_M`），但 INT8 的绝对带宽（172 GB/s @TP=2）高于 FP8（~77 GB/s）⇒ INT8 + GEMM 派发可能是**更好的组合**，待电池与后续 A/B 定夺。

## 6. 未完成项

1. **端到端 A/B 未跑**（卡被电池占用）：`fp8 tp4 k1` 与 `fp8 tp4 k2`，均带 `VLLM_DCU_FP8_GEMV_MAX_M=1`；
2. INT8 侧的 GEMM 派发 A/B 未跑（`VLLM_DCU_INT8_GEMV_MAX_M=1`）；
3. 该开关尚未写进 `vllm/envs.py`（当前用 `os.getenv`，与 `VLLM_QWEN35_*` 的既有做法一致；上线前需登记并发一次性告警）；
4. 绝对吞吐（G-K1）未解决 ⇒ 新内核（S18-4c）仍保留为**独立收益项**，但已从"MTP 前置"降级。

## 7. 复现方式

```bash
# 微基准：GEMM vs GEMV 的 M 标度（单卡）
docker exec z100-s18 bash -lc 'cd /src && . /opt/dtk/env.sh >/dev/null 2>&1 && . /src/tools/dcu/env.sh >/dev/null 2>&1 && \
  export PYTHONPATH=/src:/mnt/game/triton-gfx906/install HIP_VISIBLE_DEVICES=2 && \
  python /src/tools/dcu/s18_fp8_block_mprobe.py --shapes 8704:5120 --ms 1,2,4,6,8 --modes gemv,gemm --threads-list 128'
# op 级分派验证
docker exec z100-s18 bash -lc 'cd /src/tools/dcu && . /opt/dtk/env.sh >/dev/null 2>&1 && . /src/tools/dcu/env.sh >/dev/null 2>&1 && \
  export PYTHONPATH=/src/tools/dcu:/src:/mnt/game/triton-gfx906/install HIP_VISIBLE_DEVICES=3 && \
  python /src/tools/dcu/s18_dispatch_check.py --max-m 1'
# 端到端（待跑）
docker exec -d z100-s18 bash -lc 'export VLLM_DCU_FP8_GEMV_MAX_M=1; export S18_PROBE_MODES="greedy prod"; \
  export S18_BENCH_MODES=decode; bash /src/tools/dcu/s18_one_arm.sh fp8 4 1 8192 gemmdisp'
```

## 8. 参考

- `docs/design/z100-18-verify-gemm-efficiency.md`（本发现把该任务书的 G-K2 提前达成、G-K1 降为独立项）
- `docs/result/z100-18b-mtp-first-baselines-and-cost-scaling-result.md`（M 线性现象与本步之前的分派策略来源）
- 内核源码 `csrc/rocm/fp8_emu_gemm.cu`（GEMV `:442-529`、GEMM `:623-723`、host 入口 `:1084-1130`）
