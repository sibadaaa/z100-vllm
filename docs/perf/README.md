# Z100 (gfx906) 推理性能优化 —— 报告驱动

**日期**：2026-10-13 起
**证据基线**：`/home/lyh/vllm-test/results/`
（`PERF_REPORT.md` 三层归因 · `step_breakdown.md` 逐 step 分解 · `pmc_findings.md` PMC + roofline 探针）
**测试对象**：`Qwen3.8-27B-FP8`（Qwen3.5 混合架构，64 层 = 48 GDN 线性注意力 + 16 全注意力），
TP=4，`--enforce-eager`，MTP `num_speculative_tokens=2`，`--max-num-batched-tokens 2048`。
**硬件**：4× Hygon DCU（DCU0/1 = Z100L @1.60 GHz，DCU2/3 = Z100 @1.32 GHz，各 60 CU），PCIe 互联。

> 2026-10-13 之前的 Z100 文档已全部归档到 [`docs/archive/`](../archive/README.md)，不再作为事实来源。

---

## 1. 报告给出的基线（decode 稳态，p50 step = 219.6 ms）

逐 kernel 复算（`tools/dcu/perf1_trace_kernels.py`，50 个稳态 step，rank0）：

| 类别 | kernel | 调用/step | µs/次 | ms/step | 占比 |
|---|---|---:|---:|---:|---:|
| **FP8 模拟 GEMM** | `fp8_emu_gemm_block_kernel<32,64>` | 260 | 375.9 | **97.5** | **44.4 %** |
| **TP 通信** | `ncclDevKernel_AllReduce_Ring_{Simple,LL}_Sum_f16` | 140 | 332–370 | **49.0** | **22.3 %** |
| **GDN 线性注意力** | `fused_sigmoid_gating_delta_rule_gfx906_mtp` | 48 | 716.6 | **34.4** | **15.7 %** |
| Triton matmul 回退 | `triton_matmul_kernel` | 53 | 279.2 | 14.8 | 6.7 % |
| 全注意力 | `kernel_unified_attention_2d` | 18 | 467.9 | 8.3 | 3.8 % |
| 其余（量化/逐元素/rmsnorm/启动空档等） | — | ~1500 | — | ~15 | ~7 % |

三层归因结论（报告 §0）：

1. **不是** DRAM 带宽受限：删掉该 kernel 100 % 的 global load 后**不加速**（341.6 vs 320.4 µs），
   `L1MemoryBusy` 仅 12.4 %。
2. **是**片上流水线/延迟受限：216 VGPR/thread → ≤2 blocks/CU，68 blocks 对 60 CU，occupancy ≈ 11 %。
3. 同样时间里本可算 32 行（`tile_mimic_M32` = 321.8 µs，仅 +0.4 %）——**并行度**才是杠杆。
4. 混插 2×Z100L + 2×Z100 造成 1.21× 时钟不对称，集合通信把 4 个 rank 锁成同步，
   快卡在 AllReduce 里空等（rank0/1 的 AllReduce 累计 2475 ms vs rank2/3 的 1522 ms）。

---

## 2. 优化分层的任务板

| 层 | 内容 | 任务 | 状态 |
|---|---|---|---|
| L3 内核/指令 | FP8 模拟 GEMM decode 路径：occupancy、split-K、按形状派发 | task-1 | 进行中 |
| L2 框架/算子 | 集合通信配置；GDN MTP kernel；Triton matmul 回退 | task-2 / task-3 | 进行中 |
| L1 服务/配置 | `max-num-batched-tokens`、warmup、前缀缓存等 | task-2 | 进行中 |
| L0 度量/验收 | e2e 基线 + 贪心 token 一致性 + 独立复核 | task-4 / task-5 | 进行中 |

排期与验收标准见 [`PERF1-plan.md`](PERF1-plan.md)，实测结果见 [`PERF1-results.md`](PERF1-results.md)。

---

## 3. 已被证据证伪、不要再做的方向

| 方向 | 证伪证据 |
|---|---|
| 降权重位宽（INT8/更低精度）换速度 | 删掉 100 % global load 后不加速（341.6 vs 320.4 µs），访存不是瓶颈（报告 §7.1） |
| 追求更高 DRAM 带宽 / 换更快的卡 | 硬件上限实测 776 GB/s，kernel 只用到 8.8 % |
| 优化 `aten::item` / 同步开销 | decode 中 21 次/step、0.82 ms CPU（<0.5 % 步时） |
| 靠 CUDAGraph 单纯降 launch 开销 | 空档合计 16 ms/step，且 CPU 有 2.6× 余量 |
| 打开 custom all-reduce | 平台上 `CustomAllreduce` 从未被实例化（`['PYNCCL']`），实测收益 0（`docs/archive/result/z100-13-custom-allreduce-result.md`） |
| RCCL 环境变量调优（ALGO/PROTO/CHANNELS） | 7 变体 × 5 尺寸无 separability（`docs/archive/result/z100-12-p2p-acs-impact-result.md` §3.2） |

---

## 4. 度量纪律

* 微基准：`tools/dcu/s18_fp8_block_mprobe.py`（形状 × M × 路径，自带 fp32 参考校验）。
* e2e：`tools/dcu/perf1_e2e_*.py`，口径与 `vllm-test/scripts/bench_baseline.py` 一致（JSON 可直接对比）。
* 任何"提速"必须同时给出：**逐位/贪心 token 一致性**（`perf1_e2e` parity 模式）+ **原始日志路径**。
* 服务只允许一个 arm 在跑；任何时刻必须留一个可用服务在 `:8000`。
