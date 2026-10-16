# S2 DTK 平台/运行时兼容层

> 上层文档：[适配方案](../海光Z100-DTK26.04-gfx906-vLLM适配方案.md) · [Todolist](../z100-todolist.md)
> 前置：[S1](z100-01-arch-normalization.md) · 状态：✅ 已落地，待评审

## 目的

DTK 缺 `/opt/rocm`、缺 python `amdsmi`，而本 fork 的 `builtin_platform_plugins` **只保留了 rocm 插件**（cuda/xpu/cpu/tpu 都被注释掉）。因此 `rocm_platform_plugin()` 一旦返回 `None`，vLLM 就**没有任何可用平台**，服务直接起不来。本步骤把所有"DTK 上必然失败"的调用点补上 fallback，且**不初始化 HIP 上下文**。

## 已落地的实现（评审对象）

| 位置 | 行为 |
|---|---|
| `vllm/platforms/hygon_dtk.py` | `is_hygon_dtk()`（环境变量 / `/opt/dtk/.info/rocm_version` / `/opt/dtk*/bin/hy-smi`，torch-free、GPU-free）；`dtk_version()`；`is_valid_gcn_arch()`；`normalize_gcn_arch()`；`_read_kfd_topology()`；`_dcu_drm_cards()`；`dcu_device_memory_total_bytes()` / `dcu_device_unique_id()`；`rocminfo_candidates()`；bf16 相关（见 [S3](z100-03-bf16-verification.md)） |
| `vllm/platforms/__init__.py` | `rocm_platform_plugin()` 首步 `if is_hygon_dtk(): return "vllm.platforms.rocm.RocmPlatform"`；rocminfo fallback 遍历 `rocminfo_candidates()` |
| `vllm/platforms/rocm.py` | `_HAS_AMDSMI` 标记；`with_amdsmi_context` 在无 amdsmi 时直接调用被包装函数；`get_device_name/uuid/total_memory` 的 sysfs fallback；`is_fully_connected` 无法判定返回 `False`；`_get_gcn_arch` 多级校验 |

### 三条必须遵守的实现规则

1. **优先级：显式 > 免上下文 > 需要上下文**
   `VLLM_DCU_GCN_ARCH` → torch `gcnArchName`（整数索引）→ KFD sysfs。amdsmi **不参与** DTK 路径。设置 `VLLM_DCU_GCN_ARCH` 可以让 vLLM 在导入期完全不初始化 HIP 上下文（Ray worker 更稳）。
2. **sysfs 索引必须按 vendor + PCI BDF 映射**
   本机 `/sys/class/drm/card0` 是 NVIDIA 卡，DCU 是 `card1..card4`；`card0-DP-1` 这类 connector 目录也会匹配 `card[0-9]*`。`_dcu_drm_cards()` 过滤 `^card\d+$`、保留 vendor `0x1d94`、按 PCI bus/device/function 排序，索引即 HIP ordinal。
   逻辑设备→物理设备：`get_device_total_memory` 复用 `device_id_to_physical_device_id()`；`get_device_uuid` 与既有 amdsmi 分支保持同一约定（用逻辑 id）。
3. **无法判定时取保守值**
   `is_fully_connected()` 在无 amdsmi 时返回 `False` ⇒ vLLM 保留标准 RCCL allreduce（world_size > 2 时自动禁用 custom allreduce），并 `debug_once` 提示可用 `--disable-custom-all-reduce`。

## 验收（门禁）

```bash
# 1. 无 amdsmi / 无 /opt/rocm 时能选中平台
python -c "from vllm.platforms import rocm_platform_plugin; print(rocm_platform_plugin())"
# 期望: vllm.platforms.rocm.RocmPlatform

# 2. 设备 API 不抛异常
python -c "from vllm.platforms.rocm import RocmPlatform as P; \
           print(P.get_device_name(0), P.get_device_uuid(0), P.get_device_total_memory(0))"

# 3. 平台识别与寄存器信息
python -c "from vllm.platforms import current_platform; print(current_platform.device_name, current_platform.device_count())"

# 4. GPU-free 单测
python -m pytest tests/platforms/test_hygon_dtk.py -v
```

**回归要求**：在没有 DTK 的机器（或 `rm -rf` 掉 `/opt/dtk` 的容器）上，`rocm_platform_plugin()` 的原有分支顺序与返回必须不变。

## 失败处置

| 现象 | 处置 |
|---|---|
| 平台仍是 `None` | 确认 `is_hygon_dtk()` 为 True（容器里可能未挂载 `/opt/dtk/.info`），必要时显式 `export VLLM_DCU_GCN_ARCH=gfx906` 或设置任一可见性变量（`HIP_VISIBLE_DEVICES`）走短路分支 |
| `get_device_uuid` 多卡重复/为空 | 检查 `_dcu_drm_cards()` 顺序与 `unique_id` 可读性；UUID 参与分布式识别，必须逐卡唯一 |
| TP>2 卡死 | 加 `--disable-custom-all-reduce` 复测；若仍卡死转 [S9](z100-09-acceptance-and-bench.md) 排查 RCCL |

## 参考

- 生产版补丁对照：`/home/lyh/server/gfx906_port/artifacts/rocm.py.production`
- 实证记录：[hygon_z100.md](../hygon_z100.md) §1
