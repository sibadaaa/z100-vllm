# S11b P0：27B W8A8 INT8（TP=2）decode 380 ms/token 的 kernel 级剖析 — 执行结果

> 归档时间：2026-10-09
> 执行者：子 agent（S11b 续跑；上一任在同一任务上静默失败，无遗言）
> 对应设计文档：`docs/海光Z100-性能优化方案.md` §1（归因假设表）与 §2（P0 做法与阈值）——本步无独立 `docs/design/` 文档
> 前一档：无（`docs/result/z100-11b-perf-profiling-result.md` 在本轮接手时不存在）
> 状态：🔶 **部分完成** —— **主门禁已达成（分类聚合表 + top-20 + CPU/GPU 拆分齐全且经独立复算）**；`cudagraph_mode=NONE` 对照口径**两次独立复现同一确定性崩溃**，判定为本树 vLLM 缺陷（§3.9 / §5）。
> ⚠️ **时效性提示**：剖析完成后共享树已被 S11e 改动，W8A8 内核已从 `TritonInt8ScaledMMLinearKernel` 切为 `Gfx906Sdot4Int8ScaledMMLinearKernel` ⇒ **本档案是"P2 之前"的对照基线**，必读 §3.10 与 §6.0。
> 修订历史：本文件按任务要求**边做边写**：① 接手第 1 步创建骨架并登记日志抢救清单；② 追加独立复算与门禁判定。所有结论均在最终修订后保持不变。

---

## 1. 目标与验收标准

**问题**：27B W8A8 INT8（TP=2）decode 的 **380 ms/token 花在哪？**

**门禁**（`docs/海光Z100-性能优化方案.md` §2 原文）：

| 类别 | 目标产出 | 判据 |
|---|---|---|
| `gemm/int8` | ms + 占比 | **> 45% → 先做 P2** |
| `gdn+mamba` | ms + 占比 | **> 35% → 先做 P3** |
| `attention` | ms + 占比 | — |
| `norm+activation` | ms + 占比 | — |
| `sampling` | ms + 占比 | — |
| `其它` | ms + 占比 | — |
| **CPU 侧** | ms + 占比 | **> 30% → 先做 P5** |

外加：top-20 kernel 名（原始 profiler 表进归档）；分类占比需闭合到 ~100%；默认（图开启）与 `-cc.cudagraph_mode=NONE`（关图）两种口径都测并说明差异；每 step 的 CPU wall vs GPU 时间、`--cudagraph-metrics` 的 replay 命中、`hy-smi` 采样。

**本次执行的验收标准**：① 上述表存在且占比闭合；② 数据可追溯到原始 trace/日志；③ 门禁判定可直接驱动「下一步 P2 还是 P3」。

---

## 2. 实际改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `docs/result/z100-11b-perf-profiling-result.md` | 本文件 | 新建（本步交付物 A） |
| `tools/dcu/s11b_analyze2.py` | 新增 | **独立复算脚本**（stdlib-only，直接解析 chrome trace JSON；不依赖上一任遗留的 `s11b_analyze.py`，该文件 0600 root 不可读） |
| `vllm/**` | — | **无代码改动（本步只观测）** |
| `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md` | — | 未触碰（硬约束） |

生产容器 `qwen38-vllm`（Exited）**全程未 start/kill/exec/commit**。未运行任何独立的自编译 HIP 可执行程序；全部测算经 python / vllm / triton 派发。

---

## 3. 执行过程与原始证据

### 3.0 日志抢救（未重跑引擎即拿到主结果）

接手时 `/mnt/game/port/s11b/` 的**真实**内容比交接说明所列更多。逐项抢救结论：

| 文件 | 大小 | 抢救结论 |
|---|---|---|
| `run-default-z23b.log` | 444 KB | ✅ **主证据**。默认（图开启）32-step profiled 运行全日志 |
| `run-default-z23b/result.json` | 5.5 KB | ✅ driver 侧逐 step wall/cpu 原始数组（32 step 全量） |
| `run-default-z23b/*.trace.json` | 29.4 MB ×2 | ✅ **rank0/rank1 chrome trace 原始文件**（122,433 / 122,369 events） |
| `run-default-z23b/hysmi.txt` | 5.3 KB | ✅ profiled 窗口内 7 次 hy-smi 采样 |
| `run-default-z23b/shapes_triton_matmul_pid*.tsv` | 20 KB ×2 | ✅ 仅记录了 `triton_matmul`（fp16 lm_head）的 M/K/N |
| `analyze-default-z23b.log` | 25.8 KB | ✅ 上一任分析脚本的**完整输出**（分类表 / top-20 / 逐 step 窗口）→ 本档案 §3.5–3.6 据此 + 我的独立复算 |
| `run-default.log`、`run-default-z23.log`、`run-default/`、`analyze-attempt0.log` | 435/416 KB | ⚪ 同口径的更早轮次，未采用（结论一致，见 `analysis-summary.txt` 24,188 B） |
| `run-none-z23.log` | 3.4 MB | ❌ **NONE 模式唯一一次尝试，崩溃于引擎初始化 `profile_run`，未产出任何 trace**（`run-none-z23/` 为空目录）→ §3.9 |
| `probe-*.log` | 14.7–411 KB | ⚪ 早期 probe（spawn/fork/离线 TP=2），仅用于确认启动配方（`VLLM_WORKER_MULTIPROC_METHOD=spawn` 必需、fork 会在 `init_device()` 挂） |
| `s11b_run.sh` / `s11b_profile_27b.py` | 0600/0711 root | 以 `docker exec -u root z100-s9 cat` 读出（本任务自身产物）；确认了完整口径 |

**结论：主结果无需重跑引擎即可完整重建**（避免了一次 ~29 min 冷编译）。

### 3.1 运行口径（以日志原文为准）

来源：`run-default-z23b.log:18–19,2541` 与 `result.json`。

```
[s11b 07:54:52] mode=default out=/mnt/game/port/s11b/run-default-z23b steps=32 torch=2.11.0 hip=6.3.26113 mp=spawn
resolved={"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2],
 "max_cudagraph_capture_size": 2, "max_model_len": 4096, "dtype": "torch.float16", "tp": 2,
 "enforce_eager": false, "compile_mode": "CompilationMode.VLLM_COMPILE",
 "kv_cache_dtype": "auto", "worker_multiproc_method": "spawn"}
```

| 项 | 值 |
|---|---|
| 模型 | `/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8`（`Qwen3_5ForConditionalGeneration`，`model_type=qwen3_5`，29 GB / 18 shards） |
| 结构 | 64 层 = **48 `linear_attention`（GDN）+ 16 `full_attention`**（`full_attention_interval=4`）；hidden 5120、intermediate 17408、heads 24×256、KV heads 4、vocab 248,320 |
| 量化 | compressed-tensors `int-quantized`：权重 int8 per-channel symmetric，激活 int8 **dynamic per-token** symmetric；`ignore` 覆盖 norm/embed |
| 并行 | TP=2、`--dtype float16`、**未加 `--enforce-eager`**（硬约束 13）、`max_model_len 4096`、`max_num_batched_tokens 2048`、`gpu_memory_utilization 0.9`、`enable_prefix_caching`、`disable_custom_all_reduce`、`mamba_cache_mode=align`、`cudagraph_metrics=True`、`disable_log_stats=True` |
| 环境 | `VLLM_CACHE_ROOT=/mnt/game/vllm-cache-27b`（复用，**无冷编译**）、`PYTHONPATH=/mnt/game/triton-gfx906/install:/mnt/game/port/gfx906-vllm-211`、`TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache`、`VLLM_DCU_GCN_ARCH=gfx906`、`VLLM_WORKER_MULTIPROC_METHOD=spawn` |
| 设备 | Radeon Instinct MI50 级 gfx906：**60 CU / 32752 MiB / warp 64 / LDS 65536 B**（trace `deviceProperties`） |

> ⚠️ **口径偏差（据实登记）**：默认运行的 worker **实际跑在物理 2、3 号卡上，而不是任务书指定的 0、1 号卡**。三条独立证据：① trace `deviceProperties` 只有 2 个条目（进程只见 2 卡），而 kernel 事件 `args.device` = **2**（rank0）/ **3**（rank1）——roctracer 报物理号；② `hysmi.txt` 在整个 profiled 窗口内显示**物理 2/3 为 90% VRAM、27–100% 利用率**，而 0/1 为 0% VRAM、41 W 空载；③ `s11b_run.sh` 原文 `export S8_HIP_VISIBLE_DEVICES="${S11B_DEVICES:-0,1}"`，说明上一任是用 `S11B_DEVICES=2,3` 覆盖的。**但实测 383.2 ms/token 与 S9b 归档基线 378.8–380.6 ms/token 一致（偏差 <0.5%），故数据本身有效、无争用**；仅"卡号"与任务书不符。

### 3.2 decode 实测（driver 侧，`result.json` 原文）

| 阶段 | 值 |
|---|---|
| 冷/热启动 | `[s11b 07:54:52]` 进程起 → `[s11b 08:00:55] LLM ready` = **≈6 min 3 s**（复用 compile 缓存，非 29 min 冷编译） |
| warmup（16 step） | `wall_ms=6362.94` → **397.68 ms/step** |
| **sync run**（每 step `torch.cuda.synchronize()`） | prefill **568.20 ms**；decode **mean_wall 383.24 ms/step**、mean_cpu 383.18 ms/step（32 step，min 381.79 / max 392.78） |
| **profiled run**（无逐 step sync） | prefill 591.82 ms；32 step；driver cpu span **mean 386.57 ms/step**（max 400.07）、driver wall **386.58 ms/step**、含末次 sync 386.58 ms/step |
| 输出健康 | `finish_reason=length`、`n_tokens=33`、`distinct_tokens=19`、文本连贯（非 NaN/非乱码） |

**对比 S9b 基线 378.8–380.6 ms/token：一致（profiled 期间 +1.5% 的 profiler 开销）。**

### 3.3 cudagraph 口径与 replay 命中

- `--cudagraph-metrics` **已启用**（`ObservabilityConfig(cudagraph_metrics=True)`），但 **`disable_log_stats=True` 抑制了 vLLM 的周期性 stats 输出**，因此**没有拿到数值化的 replay 命中率**——这是本步的已知缺口，不编造。
- **替代证据（强）**：step 内 **1448 个 kernel 由 `hipGraphLaunch` × 1 次提交**（CPU 侧 5.985 ms，rank0；见 §3.7），另有 38 次 `hipLaunchKernel` + 15 次 `hipModuleLaunchKernel`（图外 eager）。配合 `cudagraph_capture_sizes=[1,2]`、decode batch=1、`## Call CompiledFxGraph ##` 每 step 恰好 32 次（=32 step），可判定 **decode（batch=1）确实命中 FULL 图重放**；`splitting_ops` 中的 `vllm::unified_attention_with_output` 走 PIECEWISE（图外 eager），与 trace 中 `kernel_unified_attention_3d` 出现在图外一致。
- 口径差异（图开 vs 关）：**本步只能用默认口径**（NONE 崩溃，§3.9）。但默认口径下 profiler 仍给出**真实 kernel 名**（图重放内的 kernel 对 CUPTI/roctracer 可见），因此"NONE 才能看见 kernel 名"这一口径需求**在本树并不成立**——这是 NONE 缺失不阻塞主结论的关键理由。

### 3.4 hy-smi 采样（`run-default-z23b/hysmi.txt`）

profiled 窗口内 7 次采样（2026-10-09 08:01:14 → 08:01:32，即 `profiler started` → `profiler stopped` 的全程），原文摘录：

```
HCU  Temp   AvgPwr  PwrCap  VRAM%  HCU%
0    47.0C  41.0W   280.0W  0%     0%
1    49.0C  39.0W   280.0W  0%     0%
2    52.0C  65.0W   250.0W  90%    10%
3    55.0C  88.0W   250.0W  90%    67%
（后续采样 2/3 卡：40/40%、27/27%、40/27%、100/67%、0/0%、0/0%；功率 38–115 W）
```

- ✅ 有效信息：**2/3 卡 90% VRAM + 有功率/利用率 ⇒ 引擎确实驻留在 2/3 卡**（与 §3.1 口径偏差结论互证）。
- ⚠️ 采样质量差：08:01:29 与 08:01:32 两次读到 0% / 41 W，而 profiled 循环此时仍在跑（`result.json` 记到 08:01:32）⇒ `HCU%` 是瞬时且不可靠的，**本步不把它用于定量**。功率长期 38–115 W ≪ 250 W 上限，定性上支持"GEMM 既非算力饱和也非带宽饱和"。
- 注：0/1 卡 PwrCap 280 W 而 2/3 卡 250 W，**两对卡的功耗上限不同**，这也是后续若做 0/1 vs 2/3 对照时必须登记的混杂因素。

### 3.5 分类聚合表（**门禁主证据**）

**口径**：GPU kernel self time（`cat=kernel` 的 `dur` 之和）按 32 个 `execute_context_0(0)_generation_1(1)` 注解切成 31 个稳态窗口（丢弃含 prefill/编译残留的第 0 窗）取平均；`% gpu` 以 GPU busy 总和为分母。

#### rank0（物理 2 卡）

| 类别 | ms/step | % GPU busy | calls/step |
|---|---|---|---|
| **gemm/int8** | **370.37** | **97.3%** | 913.0 |
| 其它 | 7.80 | 2.1% | 349.0 |
| gdn+mamba | 1.48 | 0.4% | 99.0 |
| attention | 0.42 | 0.1% | 35.0 |
| norm+activation | 0.38 | 0.1% | 48.0 |
| sampling | 0.05 | 0.0% | 4.0 |
| **TOTAL (GPU busy)** | **380.50** | **100.0%** | 1448.0 |

#### rank1（物理 3 卡）

| 类别 | ms/step | % GPU busy | calls/step |
|---|---|---|---|
| **gemm/int8** | **369.35** | **97.2%** | 913.0 |
| 其它 | 8.47 | 2.2% | 349.0 |
| gdn+mamba | 1.48 | 0.4% | 99.0 |
| attention | 0.42 | 0.1% | 35.0 |
| norm+activation | 0.38 | 0.1% | 48.0 |
| sampling | 0.05 | 0.0% | 4.0 |
| **TOTAL (GPU busy)** | **380.13** | **100.0%** | 1448.0 |

**占比闭合**：分类之和 = GPU busy 总和，**100.0%**（无未归类残留；`其它` 已单列）。

**各类 top kernel**（rank0，原文）：

```
gemm/int8      : scaled_mm_kernel (346.981 ms, 304/step), triton_matmul_kernel (19.592 ms, 97/step),
                 vllm::dynamic_scaled_int8_quant_kernel<c10::Half,float> (2.530 ms, 304/step),
                 triton_poi_fused__to_copy_add_dynamic_scaled_int8_quant_... (0.298 ms, 48/step),
                 triton_red_fused__to_copy_add_dynamic_scaled_int8_quant_... (0.294 ms, 48/step),
                 triton_poi_fused_dynamic_scaled_int8_quant_mul_silu_slice_3 (0.267 ms, 48/step)
其它           : ncclDevKernel_Generic_4 (5.671 ms, 128/step), ncclDevKernel_AllReduce_Ring_LL_Sum_f16 (0.376 ms, 1/step),
                 reduce_segments (0.298 ms, 16/step), triton_per_fused__to_copy_mean_pow_view_0 (0.294 ms, 48/step),
                 ncclDevKernel_AllGather_Ring_Simple_Sum_i8 (0.202 ms, 1/step), triton_poi_fused_zeros_5 (0.186 ms, 32/step)
gdn+mamba      : fused_sigmoid_gating_delta_rule_gfx906_mtp_v2_kernel (1.029 ms, 48/step),
                 causal_conv1d_gfx906_decode_update_kernel (0.425 ms, 48/step),
                 precopy/postprocess/preprocess_mamba_align_fused_kernel (~0.03 ms, 1/step each)
attention      : kernel_unified_attention_3d (0.292 ms, 16/step), reshape_and_cache_kernel_flash (0.091 ms, 16/step),
                 _gather_block_tables_kernel (0.012 ms, 1/step), _prepare_rope_positions_kernel (0.012 ms, 1/step),
                 _compute_slot_mappings_kernel (0.011 ms, 1/step)
norm+activation: triton_red_fused__to_copy_add_copy__fused_add_rms_norm_4 (0.248 ms, 32/step),
                 triton_red_fused__to_copy_add_copy__fused_add_rms_norm_3 (0.127 ms, 16/step)
sampling       : at::native::reduce_kernel<512,1,...> (0.017), _gumbel_sample_kernel (0.015),
                 _combine_sampled_and_draft_tokens_kernel (0.010), _get_num_sampled_and_rejected_kernel (0.005)
```

**calls/step 与模型结构的交叉验证**：`fused_sigmoid_gating` 48/step = 48 个 GDN 层；`kernel_unified_attention_3d` 16/step = 16 个 full-attention 层；`scaled_mm` **304/step = 48×3 + 16×2 + 64×2**（GDN 的 in_proj_qkvz/in_proj_ba/out_proj + full-attn 的 qkv/o + 全 64 层的 gate_up/down）——**逐项精确吻合**，证明分类无遗漏、无重复计数。

### 3.6 独立复算（防止单点依赖上一任脚本）

用 `tools/dcu/s11b_analyze2.py`（stdlib-only，**我自写的分类正则与窗口切法**）直接解析两个 29 MB trace：

```
$ python3 tools/dcu/s11b_analyze2.py .../s11b-default_rank0.1791504087748760345.pt.trace.json
events=122433 kernels=46336 (60 unique) cpu_ops=22017 step_annotations=32
steady-state per-step (n=31): gpu_span=384.76 ms  gpu_busy=380.50 ms  cpu_union=383.97 ms  kernels/step=1448.0
category           ms/step   % gpu  calls/step
gemm/int8           370.37   97.3%       913.0
其它                    7.80    2.1%       349.0
gdn+mamba             1.48    0.4%        99.0
attention             0.42    0.1%        35.0
norm+activation       0.38    0.1%        48.0
sampling              0.05    0.0%         4.0
TOTAL(gpu busy)     380.50  100.0%      1448.0
```

与上一任 `analyze-default-z23b.log` 的 **370.4 / 7.8 / 1.5 / 0.4 / 0.4 / 0.0 / 380.5** 逐项一致（唯一差异：我把 `_gather_block_tables_kernel`、`_compute_slot_mappings_kernel` 归入 `attention`，导致 `其它` 7.8→2.1% vs 2.0%，属分类边界选择，不影响任何门禁）。**结论经两条独立实现互证。**

### 3.7 CPU 侧 vs GPU 侧（门禁 CPU>30% 的判据）

**rank0 原始 CPU self-time top（稳态窗口，per step）**：

| cat | op | ms/step | calls/step |
|---|---|---|---|
| cuda_runtime | **hipEventSynchronize** | **371.193** | 1.0 |
| user_annotation | execute_context_0(0)_generation_1(1) | 11.419 | 1.0 |
| cuda_runtime | **hipGraphLaunch** | **5.985** | 1.0 |
| cpu_op | aten::copy_ | 0.594 | 36.0 |
| cpu_op | aten::to | 0.567 | 28.0 |
| cpu_op | aten::_to_copy | 0.529 | 18.0 |
| cpu_op | vllm::all_gather | 0.335 | 1.0 |
| cuda_runtime | hipLaunchKernel | 0.331 | 38.0 |
| cpu_op | aten::slice | 0.326 | 71.0 |
| cpu_op | vllm::rocm_unquantized_gemm | 0.311 | 1.0 |
| cpu_op | Torch-Compiled Region: 1/1 | 0.297 | 1.0 |
| cpu_op | c10d::_allgather_base_ | 0.207 | 1.0 |
| cuda_runtime | hipModuleLaunchKernel | 0.193 | 15.0 |
| cuda_runtime | hipMemcpyAsync | 0.173 | 12.0 |

**关键拆分**：

| 口径 | ms/step | 占 step（386.6 ms） |
|---|---|---|
| step 墙钟（profiled，driver） | 386.58 | 100% |
| GPU kernel busy（rank0 / rank1） | 380.50 / 380.13 | **98.4% / 98.3%** |
| GPU span（首 kernel 起 → 末 kernel 止） | 384.76 / 384.63 | 99.5% |
| CPU 区间并集（含阻塞式同步） | 383.97 / 13.12 | — |
| ├ **`hipEventSynchronize`（纯等 GPU，非 CPU 工作）** | **371.19** / — | **96.0%** |
| └ **CPU 真实活跃（并集 − 同步）** | **12.78 / 13.12** | **3.3% / 3.4%** |

- **CPU 侧 = 约 12.8 ms/step（3.3%）**；**96% 的 step 时间 CPU 阻塞在等 GPU**。
- 图重放开销已很小：整个 step 的 1448 个 kernel 只需 **1 次 `hipGraphLaunch`（5.99 ms）**。
- rank1 的并集本就是 13.12 ms（rank1 trace 无 `hipEventSynchronize`），与 rank0 相减得到的 12.78 ms 互证。

### 3.8 GEMM 解剖：347 ms 的 `scaled_mm_kernel` 是什么、慢在哪

**(a) kernel 身份**（源码定位）

```
/mnt/game/port/gfx906-vllm-211/vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py:22
    def scaled_mm_kernel(...)
调用点：/mnt/game/port/gfx906-vllm-211/vllm/model_executor/kernels/linear/scaled_mm/triton.py:137
    out = triton_scaled_mm(x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias)
```

`triton_scaled_mm(...)` 用默认 `use_heuristic=True`、未传 `block_size_*`（已 grep 确认无覆盖）⇒ tile 由启发式决定：

```python
next_power_of_2_M = max(32, triton.next_power_of_2(M))            # M=1 -> 32
if next_power_of_2_M <= 32: tile_shape = (64, 64, 256) if is_small_N else (64, 128, 256)
...
accumulator_dtype = tl.float32 if input.is_floating_point() else tl.int32
accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)   # int8 x int8 -> int32
```

⇒ **decode（M=1）一律使用 `BLOCK_SIZE_M = 64`**：A tile 是 64×256 int8，`masks_am = offsets_am < M` 使第 1..63 行全被 mask 掉，**63/64 = 98.4% 的 tile 计算是 padding 冗余**。

**(b) 实测吞吐 vs 上界（roofline）**

| 量 | 值 | 推导 |
|---|---|---|
| `scaled_mm` 调用数/step | **304** | = 48×3 + 16×2 + 64×2（§3.5 已验证） |
| 有用 int8 GEMM 计算量 | **24.35 GFLOP/step/GPU** | gate_up 11.41 + down 5.70 + qkv 1.17 + o 0.50 + gdn_qkvz 4.03 + gdn_ba 0.02 + gdn_out 1.51 |
| 实测 `scaled_mm` 时间 | 346.98 ms | rank0 |
| **有用算力达成率** | **≈70 GFLOP/s** | 24.35 / 0.34698 |
| int8 权重/GPU | **13.45 GB** | 26.89 B 参数 ÷ 2（int8 = 1 B/参数） |
| **有效权重带宽** | **≈38.8 GB/s** | 13.45 GB / 0.34698 s |
| 访存下界（1.02 TB/s，MI50 级） | **13.13 ms/token** | 13.45 GB ÷ 1.02 TB/s |
| **实测 GEMM 相对访存下界** | **26.4× 慢** | 346.98 / 13.13 |
| 计入 64× padding 的机器实际吞吐 | ≈4.49 TFLOP/s | 64 × 24.35 / 0.34698 |

⇒ **GEMM 既不是算力饱和、也不是带宽饱和：它只吃到约 3.8% 的 HBM 带宽**，是把权重流"卡住"的实现问题。

**(c) 机器码实测：本树**已经**在用 `v_dot4_i32_i8`（**修正方案 §1 假设 #2 的前提**）

对 `TRITON_CACHE_DIR` 内所有 `scaled_mm_kernel` 变体做 ISA 普查（`grep -c` 汇编）：

| 变体 ttir 的 `tt.dot` | `shared` | warps/stages | `v_dot4_i32_i8` | `v_dot2_f32_f16` | `v_mfma`/`v_wmma`/`v_fma_f32` |
|---|---|---|---|---|---|
| `tensor<64x256xi8> * tensor<256x64xi8> -> tensor<64x64xi32>` | 32768 | 4 / 2 | **2048** | 0 | 0 / 0 / 0 |
| `tensor<64x128xi8> * tensor<128x128xi8> -> tensor<64x128xi32>` | 24576 | 4 / 2 | **2048** | 0 | 0 / 0 / 0 |
| `tensor<128x128xi8> * tensor<128x128xi8> -> tensor<128x128xi32>` | 32768–65536 | 4 / 2–3 | **4096–6144** | 0 | 0 / 0 / 0 |

（`target = {'backend': 'hip', 'arch': 'gfx906', 'warp_size': 64}`；缓存见 `N5YVJLTHWPU3…` / `5DAJLLEBI3I3…` / `QI44KD3WY2AW…` 等目录。）

**⇒ 本树自编 triton 3.6 已把 int8 `tl.dot` 正确降到 `v_dot4_i32_i8`（2048–6144 条/内核），且 0 条 fp16/fp32 dot、0 条 MFMA/WMMA。** 方案 §1 假设 #2 里"Triton 走 int8→fp16 反量化 + fp16 GEMM、没吃到 `v_dot4_i32_i8`"的描述**与本树实际不符**；"缺指令"不是 380 ms 的原因。

**(d) 两个可量的浪费来源**

1. **M=1 的 64× tile 冗余（硬证据）**：见 (a)，`BLOCK_SIZE_M=64` 对 M=1 是 63/64 空转。
2. **K 循环疑似未双缓冲（推断，有配置算术支撑）**：decode 用的 (64,64,256) 变体 `shared=32768 B`，恰好等于**单份** A+B 缓冲（64×256 + 256×64 = 32 KB）；双缓冲需 64 KB，正好顶到 gfx906 的 64 KB LDS 上限（`sharedMemPerBlock=65536`）⇒ 流水被 LDS 容量压掉、权重流延迟暴露，与"只有 3.8% HBM 带宽"吻合。**此项为推断，需 P2 用微基准证伪/证实。**

### 3.9 `cudagraph_mode=NONE` 对照口径 —— 两次均失败（附我本次的重试）

**上一任的尝试**：`run-none-z23.log`（3.4 MB）在 `08:03:07` 起，`08:05:40` 崩溃，`run-none-z23/` 为空 ⇒ 无 trace。根因（日志原文）：

```
File ".../vllm/v1/worker/gpu_worker.py", line 615, in determine_available_memory
    self.model_runner.profile_run()
File ".../vllm/v1/worker/gpu/model_runner.py", line 949, in profile_run
    hidden_states, sample_hidden_states = self._dummy_run(
File ".../vllm/v1/worker/gpu/model_runner.py", line 1992, in execute_model
    model_output = self.model(**model_inputs)
File ".../vllm/compilation/decorators.py", line 672, in __call__
    self.aot_compiled_fn = self.aot_compile(*args, **kwargs)
torch.fx.experimental.symbolic_shapes.ConstraintViolationError: Constraints violated
  (L['inputs_embeds'].size()[0], L['positions'].size()[1])!
  - You marked L['inputs_embeds'].size()[0] as dynamic but your code specialized it to be a
    constant (2048). If you're using mark_dynamic, either remove it or use maybe_mark_dynamic.
```

⇒ 发生在引擎初始化的 `profile_run → _dummy_run`（`max_num_batched_tokens=2048` 被特化为常量，与 `mark_dynamic` 的 `inputs_embeds` dim0 冲突），**早于任何 profiler 窗口**，与 GPU 争用无关。

**我本次的重试（接手后）**：确认 4 张卡全部空载（`hy-smi` 0% VRAM / ~40 W）、复用 `VLLM_CACHE_ROOT=/mnt/game/vllm-cache-27b`，在**任务书指定的 0、1 号卡**上（`[dcu-s8] HIP_VISIBLE_DEVICES=0,1`）按同一 `s11b_run.sh none` 重跑，日志 tee 到 `/mnt/game/port/s11b/run-none-v2.log`，外层 `timeout 2400`。

**结果：逐字复现同一崩溃，`RUN_EXIT=1`，`run-none-v2/` 为空目录（无 trace）：**

```
[s11b 08:16:02] mode=none out=/mnt/game/port/s11b/run-none-v2 steps=32 ... mp=spawn
[dcu-s8] HIP_VISIBLE_DEVICES=0,1
(Worker_TP0 pid=81598) INFO 08:16:47 Model loading took 14.53 GiB memory and 28.95 seconds
(Worker_TP0 pid=81598) [rank0]:E1009 08:17:39.235000 torch/_guards.py:368] raise ConstraintViolationError(
(Worker_TP1 pid=81599) [rank1]:E1009 08:17:40.758000 torch/_guards.py:368] raise ConstraintViolationError(
torch.fx.experimental.symbolic_shapes.ConstraintViolationError: Constraints violated
  (L['inputs_embeds'].size()[0], L['positions'].size()[1])!
  - You marked L['inputs_embeds'].size()[0] as dynamic but your code specialized it to be a
    constant (2048). ...
  - You marked L['positions'].size()[1] as dynamic but your code specialized it to be a
    constant (2048). ...
(Worker_TP1) ... vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py:173, in apply_weights
    out = torch.ops._rocm_C.w8a8_sdot4_gemm(x_q, w, x_s, w_s, bias, x.dtype, 0)
(Worker_TP1) ... vllm/v1/worker/gpu/model_runner.py:949 profile_run -> :836 _dummy_run -> :1992 execute_model
WARNING 08:17:44 [multiproc_executor.py:484] [shutdown] Executor: workers still running after grace period; sending SIGTERM count=2
RUN_EXIT=1
```

**结论（两次独立复现，跨 2/3 卡与 0/1 卡两组、卡全空载、同一 `max_model_batched_tokens=2048` 口径）：`cudagraph_mode=NONE` 在本树初始化阶段确定性失败，与资源/争用无关，属 vLLM fork 的 `_dummy_run` 动态形状（`mark_dynamic` vs 特化为常量 2048）缺陷。** 修复它属改 `vllm/**` 源码，超出本步（只观测）范围。

### 3.10 ⚠️ 剖析之后共享树已被 S11e 改动（对本步结论的时效性至关重要）

两次运行的**内核选择日志**不同，可精确对齐时间线：

```
run-default-z23b.log:107  (容器 07:55:42 = 2026-10-08 23:55:42Z)
  INFO [__init__.py:805] Selected TritonInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
run-none-v2.log:107       (容器 08:16:47 = 2026-10-09 00:16:47Z)
  INFO [__init__.py:805] Selected Gfx906Sdot4Int8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
```

| 时间（UTC） | 事件 | 证据 |
|---|---|---|
| 2026-10-08 23:50:35 | **S11e 落地** `vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py`，并改 `scaled_mm/__init__.py` 注册顺序 | `stat` mtime |
| 2026-10-08 23:54:52 → 23:55:42 | S11b 默认剖析运行启动；内核选择**回退到 `TritonInt8ScaledMMLinearKernel`**（说明彼时 `_rocm_C.w8a8_sdot4_gemm` 尚不可用，`is_supported()` 为 False） | 上表日志行 |
| 2026-10-09 00:01:32 | S11b 默认剖析运行结束 → **本档案的 380.5 ms 分解即 Triton 路径的基线** | `result.json` mtime |
| 2026-10-09 00:16:47 | 我的 NONE 复跑：内核已切换为 **`Gfx906Sdot4Int8ScaledMMLinearKernel`** | 上表日志行 |

**含义**：P2 的自研 W8A8 内核**已经集成进共享树并成为 CompressedTensorsW8A8Int8 的默认选择**（`vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py:173` → `torch.ops._rocm_C.w8a8_sdot4_gemm`，并出现在 `torch/_guards.py` 的编译栈里）。因此：

1. 本档案的分类表是**"P2 之前"的对照基线**（`TritonInt8ScaledMMLinearKernel` + `scaled_mm_kernel` 346.98 ms/step），这正是 S11b 该交付的东西；S11e 的收益必须**与本档案同口径对比**（§6.1 的 380.5 ms GPU busy / 346.98 ms GEMM）。
2. **我停止继续重跑引擎**：共享树正被 S11e) 实时修改（本轮已亲历两次不同内核选择），此时再测会得到不可归因的混杂结果。
3. NONE 口径的确定性崩溃**同样会挡住 S11e 的 NONE 对照**；但 S11e 不需要 NONE —— 新内核是自定义 op，在图开启口径下也会以 `w8a8_sdot4_gemm` 出现在 trace 里（可归因）。

---

## 4. 门禁判定

| 门禁 | 阈值 | 实测（rank0 / rank1） | 判定 |
|---|---|---|---|
| **GEMM 占比** | > 45% → 先做 P2 | **97.3% / 97.2%**（370.37 / 369.35 ms per 380.5 ms） | ✅ **触发 P2（且远超阈值，是压倒性主项）** |
| **GDN+mamba 占比** | > 35% → 先做 P3 | **0.4% / 0.4%**（1.48 ms） | ✅ 不触发 P3 |
| **CPU 侧占比** | > 30% → 先做 P5 | **3.3% / 3.4%**（12.78 / 13.12 ms） | ✅ 不触发 P5 |
| 分类占比闭合到 ~100% | ~100% | **100.0%**（6 类之和 = GPU busy 380.50 ms） | ✅ 闭合 |
| top-20 kernel 名 | 需提供 | 见 §3.5/§3.6（原始表已进归档） | ✅ |
| 每 step CPU wall vs GPU | 需记录 | 386.58 ms wall / 380.50 ms GPU busy / 12.78 ms CPU 活跃 | ✅ |
| `--cudagraph-metrics` replay 命中 | 需记录 | ⚠️ 无数值（`disable_log_stats=True` 抑制）；以 `hipGraphLaunch`×1/step 作替代证据 | 🔶 部分 |
| `hy-smi` 采样 | 需记录 | 🔶 7 次采样已记录，但瞬时 `HCU%` 不可靠；有效信息 = 引擎位于 2/3 卡 | 🔶 部分 |
| 默认 + NONE 双口径 | 需都测 | ❌ NONE **两次独立复现**同一 `ConstraintViolationError`（2/3 卡 + 0/1 卡，均空载） | ❌ 缺口（**确定性树缺陷**，非资源问题；且默认口径已给真 kernel 名，不阻塞主结论） |

**综合：主门禁通过 ⇒ 归因结论为 "decode 380 ms 中约 347 ms（≈90% 墙钟）花在 W8A8 int8 GEMM，且该 GEMM 只吃到 3.8% 的 HBM 带宽"；下一步应做 P2，不做 P3、不做 P5。**

---

## 5. 未完成项 / 阻塞与原因

| # | 未完成项 | 原因 | 影响 |
|---|---|---|---|
| 1 | `cudagraph_mode=NONE` 对照口径 | 引擎初始化 `profile_run → _dummy_run` 抛 `ConstraintViolationError`（`inputs_embeds`/`positions` 被特化为常量 2048 vs `mark_dynamic`）。上一任一次（`run-none-z23`，2/3 卡），**我在 0/1 卡复用缓存再试一次并逐字复现**（`run-none-v2`，`RUN_EXIT=1`，无 trace） | **判定为确定性树缺陷，不阻塞主结论**：默认口径的 trace **已含真实 kernel 名**（图重放内 kernel 对 roctracer 可见），NONE 的唯一额外价值是"无图时的 CPU 启动开销"，而生产形态是图开启、CPU 仅 3.3%。修复需改 `vllm/**` → 应单独立项 |
| 6 | **基线时效性** | 共享树在剖析后被 S11e 改动（`Gfx906Sdot4Int8ScaledMMLinearKernel` 已自动接管 W8A8），见 §3.10 | 本档案是**"P2 之前"的对照基线**；任何"新内核更快"的结论**必须与本档案同口径对比**，且必须在树不再变动时测 |
| 2 | `--cudagraph-metrics` replay 命中率数值 | `disable_log_stats=True` 抑制了 vLLM 周期性 stats 输出 | 用 `hipGraphLaunch`=1/step、1448 kernel/step 作替代证据 |
| 3 | hy-smi 定量利用率 | `HCU%` 瞬时采样不可靠（同窗口内 100% 与 0% 交替）；且 0/1 与 2/3 卡 `PwrCap` 不同（280 vs 250 W） | 仅用于"引擎在哪张卡"的定性判定 |
| 4 | GEMM 逐 shape 耗时（哪个 N/K 最贵） | roctracer 对 `scaled_mm_kernel` **未提供 `grid`/`block`** 参数（9728 个事件全部 `args.grid` 为空），无法从 trace 反推 shape；上一任的 `shapes_*` TSV 只记录了 fp16 `triton_matmul` | 已用解析法给出 FLOP 预算（§3.8b），但"最大单 GEMM 是哪一层"需 P2 自己打点 |
| 5 | 卡号与任务书不符 | 默认运行在物理 2/3 卡（`S11B_DEVICES=2,3`），非指定的 0/1 | 实测值与 S9b 基线一致，数据有效；但**若后续要做 0/1 vs 2/3 对照，须登记功耗上限差异** |

---

## 6. 对后续步骤的影响

### 6.0 最重要的影响：**P2 已经在飞行中，本档案转为它的"改动前基线"**

§3.10 已证明：S11e 已把 `Gfx906Sdot4Int8ScaledMMLinearKernel` 集成进共享树并**成为 `CompressedTensorsW8A8Int8` 的默认选择**。因此 Lead 的下一步**不是"开始 P2"，而是**：

1. **用新内核复测 27B decode（默认图开启口径）**，与**本档案同口径**对比：`GPU busy 380.50 ms/step`、`scaled_mm GEMM 346.98 ms/step`、`triton_matmul 19.59 ms/step`、`1448 kernels/step`、`cpu 12.78 ms/step`。
2. **归因很干净**：本步已证明除 GEMM 外的所有类别合计只占 **2.7%**（10.2 ms/step）。若新内核把 GEMM 从 347 ms 降到 (347/k) ms，则 **decode 预期 ≈ 33.5 + 347/k ms**。参照方案 §4B 的"22 TFLOPS vs Triton 2.2"（≈10×），落点约 **35–70 ms/token**；即使只拿到 2×，也有 **≈207 ms/token**。⇒ **P2 的收益上限远高于方案原先"端到端 1.3–1.5×"的估计。**
3. **验收门槛建议**：新内核在 profiler 里必须以 `w8a8_sdot4_gemm`（而非 `scaled_mm_kernel`）出现，且 §3.8 列出的 **M=1、N∈{17408, 8192, 7168, 5120, 48}** 各档 decode shape 都要覆盖、且 **no shape regresses**（与方案 §4A2/§4B 一致）。
4. **测量纪律**：复测必须在共享树**不再变动**时进行（我本轮已亲历 30 min 内两次不同的内核选择），否则无法归因。

### 6.1 结论：**下一步先做 P2（W8A8 INT8 GEMM），不做 P3、不做 P5**

| 优先级 | 依据 |
|---|---|
| **P2（做）** | GEMM 97.3% ≫ 45% 阈值；347 ms/step 是 380 ms 的绝对主体。P3 的 GDN 只有 1.48 ms（0.4%），即使优化 100% 也仅省 0.4%；P5 的 CPU 只有 12.8 ms（3.3%） |
| P3（缓） | GDN+mamba 1.48 ms/step（48 层 → 0.031 ms/层）。**方案 §1 假设 #3「GDN/线性注意力路径占大头」被本步实测否定**（decode 侧）。prefill 的 +27% 同源问题仍可能独立存在，但不在 decode 380 ms 的构成里 |
| P5（缓） | CPU 12.8 ms/step，且 371 ms 是纯 `hipEventSynchronize` 等 GPU。**方案 §1 假设 #6「CPU 侧开销」被否定** |

### 6.2 对 P2 方案本身的修正（重要）

本步的 ISA 普查**推翻了 P2 立项时的一个前提**，会改变 P2 内的打法排序：

1. **不要再以"本树没吃到 `v_dot4_i32_i8`"为立项理由** —— 实测本树 triton 已发 2048–6144 条 `v_dot4_i32_i8`、0 条 fp16 dot。因此方案 §4B「`int8_sdot4` 的主要收益来自用上 dot4 指令」这一收益机制**不成立**（假设 #2 的"1.3–2×"来源需重估）。**但 S11e 的新内核仍然值得做**，只是理由要改成下面第 2 条——这也解释了为什么"直接用上 dot4"不足以解释 26× 的差距。
2. **真正的大杠杆是 M=1 的 tile 形状**：`BLOCK_SIZE_M=64` 对 M=1 是 **64× 冗余**（硬证据：缓存内核的 `tt.dot` = `64x256 * 256x64 -> 64x64`）。方案 §4B 里 `int8_sdot4` 的 **M≤8 分支（"每输出列一个 block、64 线程切 K/4 个 int32、int32 精确累加 + LDS 归约"）恰好是没有 M padding 的 GEMV 形态** ⇒ **P2 应以"M≤8 的 GEMV/细长 GEMM"为主攻方向**，即方案 §4A2「exact-shape 战术表 + 逐 shape ABBA」正是对的先后次序，且应把"消除 64× M padding"作为首要验收指标。
3. **次要怀疑点**：K 循环双缓冲被 64 KB LDS 上限压掉（§3.8d）。P2 应顺带做 `BLOCK_SIZE_K`/`num_stages` 的扫描（(64,64,128)/(64,64,64) + 2–3 stages），门槛仍是 **no shape regresses**。
4. **收益量级重估**：若把 GEMM 从 347 ms 压到访存下界 13 ms（26× 理论空间），decode 有 380 → 数十 ms 的空间；方案原先"GEMM 占 50% ⇒ 端到端 1.3–1.5×"的估计**低估了上限**。建议 P2 的验收改为**两级**：先拿"GEMM 时间 ÷ 2"（≈ decode 190 ms），再冲"GEMM 接近访存下界"。
5. **给 S11e 的衔接（已确认落地，见 §3.10）**：`gfx906_sdot4.py` + `scaled_mm/__init__.py`（2026-10-08 23:50:35）已进树并被自动选中；`/mnt/game/triton-gfx906/cache/s11e/` 下另有 >40 个 `scaled_mm_kernel` 变体（逐 shape 调参痕迹）。⇒ 本步 §3.8 的 347 ms/Roofline 数字应直接作为 S11e 的 **before 基线**，其 ABBA 必须包含 **M=1、N∈{17408, 8192, 7168, 5120, 48}** 这几档 decode shape。

### 6.3 不建议再投入的方向（本步新增证据）

- 不再试图用 `cudagraph_mode=NONE` 换取"看见 kernel 名"：默认口径的 trace 已有真名；且 NONE 在本树初始化即崩（§3.9）。**若确需 NONE 口径，应先修 `_dummy_run` 的 `mark_dynamic` 冲突（属 vllm 源码问题，需单独立项）**。
- 不再把 decode 归因于 GDN / attention / 采样 / CPU（合计 2.7%）。
- **不要在共享树正在被改动时做性能复测**：本轮 30 min 内同一棵树先后选了 `TritonInt8ScaledMMLinearKernel` 与 `Gfx906Sdot4Int8ScaledMMLinearKernel`（§3.10），此时测得的时间无法归因。复测前先冻结树（记录 `scaled_mm/__init__.py` 与 `_rocm_C.abi3.so` 的 md5）。

---

## 7. 复现方式

```bash
# 0) 环境（宿主机 lyh；4 卡须空闲）
cd /home/lyh/z100-vllm

# 1) 复核 decode 实测与 driver 侧逐 step 时间（纯读，不占 GPU）
python3 -c "import json;d=json.load(open('/mnt/game/port/s11b/run-default-z23b/result.json'));\
print(d['sync_run']['mean_wall_ms_per_step'], d['profiled_run']['driver_cpu_span_mean_ms'])"

# 2) 从原始 trace 独立复算分类表 / top-20 / CPU-GPU 拆分（stdlib-only，无需容器/GPU）
python3 tools/dcu/s11b_analyze2.py \
  /mnt/game/port/s11b/run-default-z23b/s11b-default_rank0.1791504087748760345.pt.trace.json
python3 tools/dcu/s11b_analyze2.py \
  /mnt/game/port/s11b/run-default-z23b/s11b-default_rank1.1791504087799751312.pt.trace.json

# 3) 上一任分析脚本的完整输出（分类表 + top-20 + 逐 step 窗口，原文）
sed -n '1,140p' /mnt/game/port/s11b/analyze-default-z23b.log

# 4) hy-smi 采样原文
cat /mnt/game/port/s11b/run-default-z23b/hysmi.txt

# 5) 若需重跑引擎（复用 compile 缓存；tee 必须在容器内做，宿主 shell 无法写 /mnt/game）
DOCKER="docker exec -u root z100-s9"
timeout 2400 $DOCKER bash -lc 'cd /mnt/game/port/s11b && S11B_DEVICES=0,1 \
  bash s11b_run.sh none /mnt/game/port/s11b/run-none-v2 2>&1 | tee /mnt/game/port/s11b/run-none-v2.log'
#    默认口径：把 none 换成 default；约 6 min 到 "LLM ready"（复用 /mnt/game/vllm-cache-27b）

# 6) 内核身份与 ISA 普查
sed -n '1,60p;180,240p' \
  /mnt/game/port/gfx906-vllm-211/vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py
grep -c 'v_dot4_i32_i8' \
  /mnt/game/triton-gfx906/cache/N5YVJLTHWPU3ZHKZOTG76NRU6MTEIFJ5Z5IPJYGE7J4MM7KQO2QQ/scaled_mm_kernel.amdgcn
```

原始证据位置：`/mnt/game/port/s11b/run-default-z23b/`（trace ×2、result.json、hysmi.txt、shapes TSV、上一任 analysis-summary.txt）、`/mnt/game/port/s11b/run-default-z23b.log`、`/mnt/game/port/s11b/run-none-z23.log`、`/mnt/game/port/s11b/run-none-v2.log`。

---

## 8. 参考

- [`docs/海光Z100-性能优化方案.md`](../海光Z100-性能优化方案.md) §1（归因假设表）、§2（P0 做法与判据）、§4/§4A/§4B（P2 与 `int8_sdot4`）、§4A.2
- [`docs/result/z100-09-acceptance-and-bench-result.md`](z100-09-acceptance-and-bench-result.md)（27B 启动配方与 decode 基线）
- [`docs/result/z100-09b-full-cudagraph-result.md`](z100-09b-full-cudagraph-result.md)（FULL 图；decode 378.8–380.6 ms/token）
- [`docs/result/z100-11d-z100-isa-capability-audit-result.md`](z100-11d-z100-isa-capability-audit-result.md)（gfx906 ISA 清单；`v_dot4_i32_i8` 可用、无 MFMA/WMMA）
- [`docs/z100-todolist.md`](../z100-todolist.md)（硬约束 1–21）、[`docs/result/README.md`](README.md)（归档模板）
- 源码：`vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py:22,190–240`；`vllm/model_executor/kernels/linear/scaled_mm/triton.py:137`
- 本步新增工具：[`tools/dcu/s11b_analyze2.py`](../../tools/dcu/s11b_analyze2.py)
