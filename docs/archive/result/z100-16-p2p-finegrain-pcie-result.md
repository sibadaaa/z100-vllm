# Z100 RCCL P2P 真因复核：`HSA_FORCE_FINE_GRAIN_PCIE=1`（S16）

> 触发：外部诊断建议按 “错误码 → BAR/Resizable BAR → IOMMU/ACS → RCCL 日志 → uncached 开关 → 容器权限”
> 顺序排查 `hipExtMallocWithFlags(Uncached)` 失败与 `via P2P=0 / via SHM=56`。
> 本文件按该清单逐条实测，并给出**此前结论被推翻的部分**与新发现的可用开关。
>
> 前置归档：[S12 ACS](z100-12-p2p-acs-impact-result.md)、[S13 custom-AR](z100-13-custom-allreduce-result.md)、
> [S14 TP=4](z100-14-27b-tp4-result.md)；文档入口 [`docs/hygon_z100.md`](../hygon_z100.md) §5。

## 0. 结论（一句话）

**RCCL 走 SHM 不是硬件/拓扑/容器/BAR/IOMMU 问题，而是缺 `HSA_FORCE_FINE_GRAIN_PCIE=1`：**
该变量决定 `hipDeviceAttributeFineGrainSupport`（实测 `0 → 1`），RCCL 的
`p2pCanConnect()` 在 AMD 平台上**硬门禁** `if (!info1->hasFineGrain || !info2->hasFineGrain) return 0;`，
且其 shareable buffer 用的 `hipDeviceMallocFinegrained`（**不是** `Uncached`）在没有该变量时返回
`hipErrorOutOfMemory`。导出该变量后：**4 rank `via P2P/IPC=56 / via SHM=0`**，
1 MiB all-reduce **1143 µs → 143 µs（8×）**，4 MiB **4388 µs → 318 µs（13.8×）**，
≤64 KiB 小消息**持平**（噪声内），归约结果与 SHM 臂**逐位一致**。

## 1. 逐条回答外部清单

| # | 清单项 | 实测 | 判定 |
|---|---|---|---|
| 1 | 错误码 + 设备属性 | `hipErrorOutOfMemory`(rc=2)，**4 KiB 与 1 MiB 同样失败**；同卡 `Default`(0x0) 成功 | 确认非容量 OOM |
| 1 | 设备属性 | `gcnArchName=gfx906:sramecc+:xnack-`、`integrated=0`、`managedMemory=1`、`canMapHostMemory=1`、`pageableMemoryAccess=0`、`isLargeBar=1`、32 GiB/卡；`cap=(9,0)` | 消费级限制假设不成立（Z100 是 Instinct 型 dGPU，Large BAR 已开） |
| 2 | BAR / Resizable BAR / Above 4G | `Region 0: 32G prefetchable`（4 卡皆然，`0x4000000000` 以上）；驱动 `isLargeBar=1` | **BAR 假设排除** |
| 3 | IOMMU / ACS / 拓扑 | `iommu=pt` 已生效（S11f 已重启验证）；12 个父桥 ACS 全 `0x0000` + 内核 `pci=disable_acs_redir=…`；`hy-smi --showtopo`：4 卡 PCIE、2 hops、weight 40；KFD `p2p_links type=2 flags=3 max_bandwidth=32000` | **IOMMU/ACS 假设排除**（且与 P2P=0 无因果关系） |
| 4 | RCCL 为何禁用 P2P | `RCCL_DEBUG=INFO`：`Channel XX : 0[2e000] -> 1[33000] via SHM/direct/direct`，**P2P=0、SHM=56**；日志同时出现 `NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment …` 与 `Disabled GDRCopy equivalent memory allocation on gfx906`；`NCCL_P2P_DISABLE` 未设 | 根因在 RCCL 门禁，**日志已经点名** |
| 5 | RCCL 内 uncached 开关 | `nm -D librccl.so` 有 `hipExtMallocWithFlags@HIP`；但反汇编 `ncclP2pAllocateShareableBuffer` 显示**正常 P2P 路径传 `0x1 = hipDeviceMallocFinegrained`**（`useMemcpy` 路径才传 `0x1000`）；`strings` 无 “uncached” 开关，只有 `RCCL_GDR_WRITE_USE_XDP_ENABLE` | “uncached 是根因”**是误判**（见 §3） |
| 6 | 容器权限 | `Privileged=false`、`IpcMode=host`、`ShmSize=16 GiB`、`--device=/dev/kfd,/dev/mkfd,/dev/dri`、`--group-add video`；实测 `hipMalloc`/`hipIpcGetMemHandle`/`hipMemcpyPeer` 均可用 | **容器假设排除**（不需要 `--privileged`/`IPC_LOCK`） |
| 7 | 最小验证 | 普通 `hipMalloc` + `hipIpcGetMemHandle` `rc=0`；`hipDeviceCanAccessPeer(0,1)=1`、`hipDeviceEnablePeerAccess=0`、真实 `hipMemcpyPeer` **19.9–20.5 GB/s**、跨卡 `copy_` **28.0 GB/s** | 硬件 P2P **本来就通**，只差 RCCL 的软件门禁 |

## 2. 关键实验

工具（新增，均 ctypes/纯 Python，无自编译 HIP 可执行程序）：

| 文件 | 作用 |
|---|---|
| [`tools/dcu/p2p_capability_probe.py`](../../tools/dcu/p2p_capability_probe.py) | flag×尺寸的错误码、设备属性、`CanAccessPeer`、真实 `hipMemcpyPeer`、IPC |
| [`tools/dcu/p2p_finegrain_sweep.py`](../../tools/dcu/p2p_finegrain_sweep.py) | 4 rank all-reduce 尺寸扫描 + 正确性指纹，含 `hipDeviceAttributeFineGrainSupport` |
| [`tools/dcu/finegrain_side_effects.py`](../../tools/dcu/finegrain_side_effects.py) | 开关的副作用面：pinned H2D/D2H、pageable H2D、D2D、peer copy、fp16 GEMM 数值 |

### 2.1 开关直接决定 flag 可用性与设备属性

`HSA_FORCE_FINE_GRAIN_PCIE` 未设时：

```
4 KiB / 1 MiB  Default 0x0 rc=0 | Finegrained 0x1 rc=2 out of memory
               Uncached 0x3 rc=2 out of memory | UncachedXdp 0x1000 rc=2 out of memory
               SignalMemory 0x2 / Contiguous 0x4 rc=1 invalid argument
hipDeviceAttributeFineGrainSupport(20004) = 0
```

导出 `HSA_FORCE_FINE_GRAIN_PCIE=1` 后：

```
Finegrained 0x1 rc=0 hipSuccess        ← RCCL shareable buffer 用的就是这个
Uncached 0x3 rc=2 / 0x1000 rc=2（仍失败，但 RCCL 正常路径不用）
hipDeviceAttributeFineGrainSupport(20004) = 1
```

即：**同一个 API、同一个 4 KiB 请求，只由这个环境变量决定成败** → 不是“显存不够”，
也不是“平台没有 uncached 内存池”这么绝对（`Finegrained` 是被环境变量门控的）。

### 2.2 A/B（4 rank，与生产 TP=4 同 world size）

`NCCL_DEBUG=INFO` 通道统计与延迟（两次独立运行，取范围）：

| 尺寸 | 默认（SHM） | `HSA_FORCE_FINE_GRAIN_PCIE=1`（P2P/IPC） | 倍率 |
|---|---|---|---|
| 5 KiB | 80.9 – 84.3 µs | 79.1 – 141.0 µs | ≈1（噪声） |
| 10 KiB | 89.7 – 102.5 µs | 82.4 – 93.8 µs | ≈1 |
| 20 KiB | 117.1 – 126.2 µs | 97.7 – 103.7 µs | 1.2× |
| 64 KiB | 148.9 – 165.7 µs | 95.4 – 114.0 µs | 1.5× |
| 256 KiB | 363.0 – 367.1 µs | 102.4 – 102.5 µs | **3.6×** |
| 1 MiB | 1142.0 – 1144.7 µs | 140.5 – 146.6 µs | **8.0×** |
| 4 MiB | 4384.7 – 4395.1 µs | 308.0 – 328.1 µs | **13.8×** |

通道：默认 `via P2P=0 / via SHM=56`（56 = 4 rank × 14 通道对数）；
开变量 `via P2P=56 / via SHM=0`，日志逐通道 `via P2P/IPC`，
并出现 `Allocated shareable buffer … size 2097152/2822144`。

正确性：两臂 `all_reduce` 指纹完全一致 ——
`sum=2709504.0000000000 sha256=f7a09216326e43ba21b62ded89b5ae36`（4 次运行同值）。

### 2.3 副作用面（开关是否拖慢别的东西）

| 项 | 默认 | `=1` |
|---|---|---|
| pinned H2D | 7.04 GB/s | 7.03 GB/s |
| pinned D2H | 7.06 GB/s | 7.06 GB/s |
| pageable H2D | 6.59 GB/s | 6.60 GB/s |
| 本地 D2D | 7608 GB/s | 7617 GB/s |
| 跨卡 `copy_` | 27.98 GB/s | 28.03 GB/s |
| fp16 GEMM 相对 fp64 误差 | 4.708e-04 | 4.708e-04 |

⇒ 该开关对宿主拷贝、空载带宽、GEMM 数值**无可测影响**，收益集中在 ≥256 KiB 的集合通信。

## 3. 对既有结论的更正（重要）

| 旧结论（S11f §10 / S13 / todolist #24） | 更正 |
|---|---|
| “`hipDeviceMallocUncached` 不可用 ⇒ RCCL P2P 与 custom-AR 的共同根因 ⇒ 只能向厂商提需求” | **RCCL 正常 P2P 路径不用 `Uncached`**：反汇编 `ncclP2pAllocateShareableBuffer` 传 `0x1(hipDeviceMallocFinegrained)`（源码同：`HIP_UNCACHED_MEMORY` 未定义时用 `hipDeviceMallocFinegrained`）。真正的门禁是 `hipDeviceAttributeFineGrainSupport==0`，而它由 `HSA_FORCE_FINE_GRAIN_PCIE` 控制 |
| “与 IOMMU 无关（对），但要推进只能找厂商（错）” | IOMMU/BAR/ACS/容器确实无关（本轮复核确认），但**有用户态开关可开**：`HSA_FORCE_FINE_GRAIN_PCIE=1` |
| “NCCL_P2P_LEVEL=SYS / IGNORE_DISABLED_P2P 都无效 ⇒ P2P 结构上不可能” | 那些开关作用于 `ncclTopoCheckP2p`，而 `p2pCanConnect()` 的 `hasFineGrain` 门禁在它**之前**；所以它们不可能生效。变量一开，通道立刻全 P2P |
| custom all-reduce “即使强闯也死” | custom-AR 的 `csrc/custom_all_reduce.cu` 用的是 `hipDeviceMallocUncached`(0x3)，**开变量后仍 rc=2**（§2.1）⇒ 平台门禁（`rocm.py` 只认 gfx94/gfx95）与 csrc 的 uncached 结论**不受本次发现影响**，仍维持“不实施” |

## 4. 建议

1. **在 Z100 上默认导出 `HSA_FORCE_FINE_GRAIN_PCIE=1`**（容器/宿主进程环境，如 `tools/dcu/env.sh`）。
   RCCL 自己在启动时以 `NCCL WARN` 索要该变量（“否则低性能/不稳定/hang”），导出后
   `FineGrainSupport=1`、P2P 通道建立、<64 KiB 不退化、≥256 KiB 8–14×。
2. **仍需端到端验证**：本轮只测到集合通信层（RCCL + torch.distributed，world=4）。
   需要在真实 `vllm serve`（27B TP=4 / 生产配方 `tools/dcu/serve_z100_tp4.sh`）上做两臂 A/B：
   `HF` 变量关/开，观察 TTFT（prefill，尺寸大 ⇒ 预期明显收益）与解码 tok/s（10 KB ⇒ 预期持平）。
   注意启动含两轮 graph capture（~23–24 min/次），A/B 需按 S11m 的三相位口径报数。
3. 不要在 decode 小消息上承诺收益：10 KB 量级两态持平（S13 的“10 KB 延迟受限”结论依然成立）。

## 5. 现场与复现命令

```bash
# 能力探针（tty 内，容器 z100-s14 影子容器）
sudo docker exec z100-s14 bash -lc 'source /src/tools/dcu/env.sh && \
  HIP_VISIBLE_DEVICES=0,1 python /src/tools/dcu/p2p_capability_probe.py'

# 4 rank A/B（两臂各跑一次）
sudo docker exec z100-s14 bash -lc 'source /src/tools/dcu/env.sh && \
  HIP_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT \
  torchrun --standalone --nproc_per_node=4 /src/tools/dcu/p2p_finegrain_sweep.py shm'
sudo docker exec z100-s14 bash -lc 'source /src/tools/dcu/env.sh && \
  HSA_FORCE_FINE_GRAIN_PCIE=1 HIP_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT \
  torchrun --standalone --nproc_per_node=4 /src/tools/dcu/p2p_finegrain_sweep.py finegrain'

# 副作用面
sudo docker exec z100-s14 bash -lc 'source /src/tools/dcu/env.sh && \
  HSA_FORCE_FINE_GRAIN_PCIE=1 HIP_VISIBLE_DEVICES=0,1 python /src/tools/dcu/finegrain_side_effects.py'
```

测量时段：2026-10-11 21:59–22:2x（宿主 `up 1:31`，内核 5.15.0-191，`iommu=pt`）；
前两轮 GPU 显存占用 88%（生产容器在跑），后两轮 0%（生产已卸载）——SHM 臂两轮数值一致
（±0.3%），说明结论不依赖占卡状态。未触碰生产容器 `z100-vllm` 的服务进程、未改 DTK、未动 GRUB。
