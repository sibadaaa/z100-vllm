# S7 内存 / KV cache / graph 策略

> 上层文档：[适配方案](../海光Z100-DTK26.04-gfx906-vLLM适配方案.md) · [Todolist](../z100-todolist.md)
> 前置：[S4](z100-04-build-system-dtk.md) · 状态：🔶 现有兜底在 Z100 上已自动生效

## 目的

1. 让 KV cache 自动算得出来（不依赖人工 `--kv-cache-memory-bytes`）；
2. 明确 CUDA/HIP Graph 在 Z100 上的取舍（历史结论是维持 `--enforce-eager`）；
3. 处理 DTK 上 `torch.cuda.memory_stats()` 峰值统计异常（曾导致"可用 KV cache 为负"）。

## 关键事实（已核对代码）

- `vllm/v1/worker/gpu_worker.py` 已有 gfx906 兜底：当 `available_kv_cache_memory_bytes <= 0`、未显式给 `--kv-cache-memory-bytes`、`gpu_memory_utilization` 恰为默认 0.9 时，自动抬到 `_GFX906_FALLBACK_GPU_MEMORY_UTILIZATION = 0.98`。判据是 `on_gfx906()`，**Z100 上本就为真** ⇒ 该兜底自动适用于 Z100（这是 das 预编译版缺的那段逻辑）。
- `vllm/utils/mem_utils.py::MemorySnapshot.measure()` 依赖 `torch.accelerator.memory_stats()`（峰值）与 `torch.accelerator.get_memory_info()`（free/total）。
- 免上下文的 VRAM 总量来源：`hygon_dtk.dcu_device_memory_total_bytes()`（`/sys/class/drm/cardN/device/mem_info_vram_total`）。

## 做法

### 1. 先验证现有兜底是否足够（不要先改代码）

```bash
source tools/dcu/env.sh
vllm serve <model> --dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.9 2>&1 | tee /tmp/serve.log
grep -E "Available KV cache memory|automatically raising it to 0.98|No available memory" /tmp/serve.log
```

- 出现 `Available KV cache memory: X GiB`（X > 0）⇒ 无需改动，进入第 3 步。
- 出现 `No available memory for the cache blocks` ⇒ 进入第 2 步。

### 2. 只在此情形下加钳制（改 `mem_utils.py`，DTK 分支）

在 `MemorySnapshot.measure()` 中，DTK 且 sysfs 总量可用时：

```python
from vllm.platforms.hygon_dtk import dcu_device_memory_total_bytes, is_hygon_dtk

        if is_hygon_dtk():
            sysfs_total = dcu_device_memory_total_bytes(device.index or 0)
            if sysfs_total:
                # DTK 的 peak 统计可能失真；以 sysfs 总量为准，并在异常时退回 allocated。
                self.total_memory = sysfs_total
                if not 0 < self.torch_peak <= self.total_memory:
                    logger.warning_once(
                        "DTK reported an implausible torch peak (%s); using "
                        "allocated bytes instead.",
                        self.torch_peak,
                    )
                    self.torch_peak = self.torch_allocated
```

规则：**只在 DTK 上生效**；`self.cuda_memory`/`non_torch_memory` 的推导随之保持一致；不改 ROCm 路径。

### 3. Graph / eager 策略（实测后定，默认保守）

| 步骤 | 命令 | 记录 |
|---|---|---|
| eager（默认建议） | `vllm serve <model> --enforce-eager` | 出词正确性、decode ms/token |
| graph 开启 | 去掉 `--enforce-eager`，必要时 `--cuda-graph-mode PIECEWISE` | 是否 capture 成功、数值是否一致、是否崩 |

判定：graph 若 capture 失败或数值不一致 ⇒ 在 DTK 上**默认 `enforce_eager`**（仅加 DTK-gated 默认值 + env 覆盖），并把结论写入 `docs/hygon_z100.md`。

### 4. TP / RCCL（与 [S9](z100-09-acceptance-and-bench.md) 联测）

- 若 TP>2 卡死：先加 `--disable-custom-all-reduce`（`is_fully_connected()` 在无 amdsmi 时已返回 False，world_size>2 会自动禁用 custom allreduce）。
- 确认 `TORCH_NCCL_*` 系列环境变量在 DTK 的 RCCL 上确实生效；不生效就不要依赖它们。

## 验收（门禁）

1. 单卡默认参数下 `Available KV cache memory` > 0，**且未使用** `--kv-cache-memory-bytes`；
2. 显式 `--kv-cache-memory-bytes` 仍能覆盖自动逻辑；
3. graph/eager 有明确结论与默认值，并记录依据；
4. 4 卡各跑一次 decode，显存占用与 `hy-smi` 观测偏差 < 5%。

## 失败处置

| 现象 | 处置 |
|---|---|
| 抬到 0.98 仍为负 | 用 sysfs 总量为 `total_memory` 钳制（第 2 步）；仍不行则文档化要求手工 `--kv-cache-memory-bytes` |
| `Free memory less than desired` | 先 `pkill -9 -f vllm` 清理残留进程再起服务 |
| graph 阶段 OOM/崩 | 回 `--enforce-eager`；怀疑 graph 内存估算不准时降低 `--gpu-memory-utilization` |

## 参考

- 前序计划 §5.3「HIP/CUDA Graph 维持 --enforce-eager」
- `vllm/v1/worker/gpu_worker.py` 现存的 gfx906 兜底分支
