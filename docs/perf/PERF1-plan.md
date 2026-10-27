# PERF1 计划：报告驱动的分层优化

**输入**：`/home/lyh/vllm-test/results/{PERF_REPORT.md, step_breakdown.md, pmc_findings.md}`
**基线**：decode 稳态 step p50 = 219.6 ms；145.8 prefill t/s @119 tok；10.28–14.14 decode t/s（MTP k=2）
**目标**：在不改变数值结果的前提下，把 decode step 压到报告证据支持的水平。

---

## 1. 分层目标与依据

| 层 | 目标 | 依据（报告中的实测） | 理论下限 |
|---|---|---|---|
| **L3 内核** | FP8 模拟 GEMM decode 路径逼近访存 roofline | 每 rank 每步只把权重分片读一遍 = 27 GB/4 = **6.75 GB**；实测 776 GB/s 上限 → **8.7–10 ms**；探针的诚实参考实现（512 线程、合并访问）达 621 GB/s → **11–15 ms** 现实目标。当前 **97.5 ms**（63 GB/s = 峰值 9.4 %） | ≈ 10 ms |
| **L2 通信** | 降低每步 140 次 all-reduce 的 49 ms | rank0/1（Z100L）在 AllReduce 内累计 2475 ms vs rank2/3（Z100）1522 ms ⇒ 快卡在**空等慢卡**；这不是链路慢，是 2×Z100L+2×Z100 混插造成的同步成本 | 不可完全消除；同型配对可削 |
| **L2 GDN** | 48 次/步 × 716.6 µs，grid 仅 (6,1,1) | 60 CU 上只起 6 个 block；Z100 上 860 µs（+20 %） | 待测 |
| **L2 Triton 回退** | 53 次/步 × 279 µs = 14.8 ms | 需先定位模块 | 待测 |
| **L1 配置** | MBT / warmup / 前缀缓存 | 启动告警 `max_num_scheduled_tokens is set to 2048`（实为 CLI 原值，非硬 clamp，见 comm-opt 复核）；首轮 prefill 279→1039 t/s 属 JIT 冷启动 | — |

## 2. 假设与证伪标准

| # | 假设 | 支持证据 | 证伪标准 |
|---|---|---|---|
| H1 | 该 kernel 是**并行度/延迟**受限，而非带宽或 ALU 受限 | 删掉 100 % global load 不加速（341.6 vs 320.4 µs）；`L1MemoryBusy` 12.4 %；216 VGPR ⇒ 2 blocks/CU | 若提高 occupancy/线程数后耗时不变，则假设被证伪 |
| H2 | 同 tile 几何下 M 是"免费"的（M=1 与 M=32 同价，+0.4 %） | `tile_mimic_M32` = 321.8 µs vs M1 320.4 µs | 若新 kernel 里 M 增大显著变慢，需重新评估投机解码收益 |
| H3 | decode 的 GEMV/GEMM 选路应按 (M,N,K) 分形状，而非全局阈值 | 实测：N=1280 K=5120 时 GEMV(64 thr) M2/M3 = 158/230 µs，而 GEMM = 344 µs（生产用 `VLLM_DCU_FP8_GEMV_MAX_M=1` 全走 GEMM，慢 1.5–2.2×）；N=4352 时相反（GEMV M3 = 723 vs GEMM 355 µs） | 若分形状后 e2e 无变化，则说明 GEMM 调用已被其他瓶颈掩盖 |
| H4 | NCCL 的 332–370 µs/次主要是**等慢卡**，不是链路 | rank 间 AllReduce 时间差 953 ms ≈ 计算时间差；4 rank 步进周期完全一致 | 若把 GDN/GEMM 的卡间差异抹平后 AllReduce 仍 ~330 µs，则假设被证伪 |
| H5 | 混插卡（2×1.60 GHz + 2×1.32 GHz）的同步成本可通过**同型配对**（TP=2 × DP=2，DCU0/1 与 DCU2/3 各成一组）消除 | 同 kernel 耗时比 1.211 = 精确时钟比 1.60/1.32 | TP=2 使每卡权重流量翻倍；若新 kernel 已接近 roofline，则 TP=2 会净亏 —— **必须在内核修好之后再测** |
| H6 | `--enforce-eager` 不应为了 launch 开销改掉 | 空档仅 16 ms/step（7 %），CPU 有 2.6× 余量；且 S14b 证明 graph 路径的 token 流不可复现，eager 是唯一可作质量基准的臂 | 若 graph 模式下贪心 token 逐位一致，可重开该臂 |

## 3. 验收纪律

1. **数值**：任何 kernel/派发改动必须给出贪心 token 逐位一致性（`perf1_e2e` parity）或 fp32 参考的 rel_err；不一致即回归。
2. **证据**：每个结论配原始日志路径（`/mnt/game/perf1/`）+ 复现命令。
3. **不做重复工作**：新增实验前先检索 `docs/archive/`（S1–S22 已覆盖大量方向），命中即引用或复现，不重跑长周期实验。
4. **服务纪律**：同时刻只允许一个 e2e arm；任何时刻 `:8000` 必须有一个可用服务。
5. **GPU 独占纪律**：`/tmp/perf1_build.lock` 是**全局 GPU 独占锁**（构建 + 微基准 + e2e arm 一律先取锁），
   任何 GPU 作业都必须整段持锁：
   `docker exec z100-vllm bash -c 'flock -w 1800 /tmp/perf1_build.lock bash -c "<gpu cmd> > /mnt/game/perf1/x.log 2>&1"'`。
   2026-10-13 11:00 已发生一次并发事故（4 卡 torchrun 与 e2e control arm 同时跑，control 数据作废）。
6. **口径绑定**：每个 arm 必须记录 `md5sum /src/vllm/_rocm_C.abi3.so`；.so 变了的 arm 一律作废。
7. **Python 改动即刻生效**：容器以 `PYTHONPATH=/src` 直接导入工作区源码，任何 Python 改动**立刻**影响
   正在跑的服务。因此：改完必须在服务上做一次冒烟请求（`max_tokens=1`，断言 HTTP 200）再交给验收；
   e2e arm 开跑前也要先做同样的探活，避免整轮 arm 报废。
   （2026-10-13 事故：`fused_sigmoid_gating.py` 的 debug 分支读写模块级全局却未声明 `global`，
   导致每个请求 500、worker 崩溃，verifier 的 stage-1 第 1 个 arm 因此作废。）

## 4. 明确不做的方向（报告已证伪）

见 [`README.md`](README.md) §3 表格：降位宽、追带宽、修 `aten::item`、为 launch 开销上 CUDAGraph、
打开 custom all-reduce、RCCL 环境变量矩阵。
