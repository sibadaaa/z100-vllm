# S0 基线与前提核验

> 上层文档：[适配方案](../海光Z100-DTK26.04-gfx906-vLLM适配方案.md) · [Todolist](../z100-todolist.md) · [实测证据](../hygon_z100.md)
> 前置：无 · 状态：🔶 部分完成

## 目的

在没有可回滚基线和可信环境事实之前，不要动任何代码。本步骤产出三样东西：**可回滚的生产镜像 tag**、**写进文档的环境事实**、**可复现的性能基线**。

## 硬约束

**禁止执行任何自编译 HIP 二进制。** 2026-10-07 曾因运行 `/tmp/t`（hipcc 编译产物）触发 hydcu 驱动 `kfd_wait_on_events` 死锁：进程进入 D 状态不可杀、持 mmap_lock 导致 `ps` 全部阻塞、loadavg 涨到 878、`hy-smi` 超时，最终只能重启。允许 `hipcc -c/-S`（只编译不运行）、`dccobjdump`/`llvm-objdump`。

## 做法

### 1. 固化生产容器（最高优先，先做）

```bash
docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'      # 确认 qwen38-vllm 状态
docker commit qwen38-vllm gfx906-vllm:prod-0.18.1-das-$(date +%Y%m%d)
docker images | grep gfx906-vllm
```

同时归档注入层（容器可写层一删即失）：

```bash
cd /home/lyh/server/gfx906_port/artifacts
for f in attn_torch.py gdn_torch.py gdn_torch_patch.py int8_sdot4_patch.py phase3_prof.py; do
  docker cp qwen38-vllm:/usr/local/lib/python3.10/dist-packages/$f prod_$f; done
for f in zz_gdn_patch.pth zz_int8_sdot4.pth zz_phase3_prof.pth; do
  docker cp qwen38-vllm:/usr/local/lib/python3.10/dist-packages/$f .; done
```

**根分区只剩约 46 GB**：只做 `docker commit`，不要 `docker save`。

### 2. 采集环境事实

在容器内执行（**不要**在宿主用系统 python）：

```bash
source /opt/dtk/env.sh
cat /opt/dtk/.info/rocm_version
python3 -c "import torch;print(torch.__version__, torch.version.hip, torch.version.cuda)"
python3 -c "import torch;[print(i, torch.cuda.get_device_properties(i).gcnArchName) for i in range(torch.cuda.device_count())]"
python3 -c "import triton;print(triton.__version__, triton.__file__)"
python3 -c "import torch;import os;print(torch.ops._rocm_C if hasattr(torch.ops,'_rocm_C') else 'no _rocm_C')"
hy-smi --showproductname
```

宿主侧免 GPU 的信息（本方案代码会用到）：

```bash
grep -H gfx_target_version /sys/class/kfd/kfd/topology/nodes/*/properties
ls /opt/dtk/lib/rocblas/library/            # 期望含 gfx906 的 hsaco/co
ls /opt/dtk/lib/hipblaslt/library/          # 期望只有 gfx928（gfx906 不可用）
ls -l /sys/class/drm/card*/device/vendor    # 识别非 DCU 卡（如 NVIDIA）
```

把结果填入 [hygon_z100.md](../hygon_z100.md) 的「本机环境核验记录」表。

### 3. 采集性能基线

用现成工具对生产容器（:8000）跑，结果存档到 `/home/lyh/server/gfx906_port/bench/`：

```bash
/home/lyh/vllmbench <按该工具用法>     # decode ms/token、冷 prefill、TTFT、并发 8 吞吐
bash /home/lyh/server/gfx906_port/bench/round.sh
```

## 验收（门禁）

1. `docker images` 里存在 `gfx906-vllm:prod-0.18.1-das-<date>` tag；能从该 tag 起出与现网等价的服务。
2. 4 张卡 `gcnArchName` 全部为 `gfx906`（若出现 `gfx916`，立即转 [S1](z100-01-arch-normalization.md)，这是已知坑）。
3. 基线三项指标（decode ms/token、TTFT、并发吞吐）复现误差 < 5%。
4. 环境事实表已写入 `docs/hygon_z100.md`。

## 失败处置

- 容器已丢失：用镜像名重建（`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`），注入层从 `artifacts/` 恢复。
- 发现残留卡死进程（`ps` 输出 D 状态、loadavg 异常高）：`bash /home/lyh/server/gfx906_port/fix/install_reaper.sh`，必要时重启。
- 机器刚重启过：先读 `/home/lyh/server/gfx906_port/重启后必读.md`。

## 参考

- 前序计划 `§ 4 Phase 0`：`/home/lyh/server/gfx906_port/适配计划-gfx906-vLLM-DTK.md`
- 故障根因记录：`/home/lyh/server/gfx906_port/重启后必读.md`
