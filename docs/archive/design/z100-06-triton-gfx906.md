# S6 Triton gfx906 支持固化

> 上层文档：[适配方案](../海光Z100-DTK26.04-gfx906-vLLM适配方案.md) · [Todolist](../z100-todolist.md)
> 前置：[S4](z100-04-build-system-dtk.md) · 状态：⬜ 待做

## 目的

Triton 必须能为 gfx906 出码，否则 attention（TRITON_ATTN/unified attention）、GDN、以及 **INT8 sdot4 资产**全部失效。das 镜像里的 triton 曾为 gfx928 构建，会在 `make_llir` 阶段 `PassManager::run failed`（Hygon 自研后端 `triton/backends/amd/compiler_hcu.py`）。

## 关键事实

- VEGA ISAFamily 补丁（3 文件、57 行）已归档并经生产验证：`/home/lyh/server/gfx906_port/artifacts/triton-gfx906.patch`（验证于 triton `c817b9b6` / 3.4.0）。
- 补丁包含三处：`TargetUtils.h` 增加 `VEGA`；`TargetUtils.cpp` 的 `deduceISAFamily()` 加 `GK_GFX906 → VEGA`、`supportsVDot()` 加 `VEGA`；`TargetInfo.cpp` 的 `getWarpSize()` 与 `warpReduce()` 走 VEGA(64)。
- **`supportsVDot()` 是 INT8 sdot4 的根基**（Triton 只有在 VDOT 可用时才生成 `v_dot` 指令）。
- 本仓库依赖 `triton-rocm==3.6.0` / DAS `triton-3.6.0+das185`；生产验证过的补丁是 3.4.0 版本，**移植需确认 3.6.0 的对应位置**。

## 做法

### 1. 先跑最小探针（30 秒定性，不改任何东西）

```python
# tools/dcu/probe_triton.py（建议入库）
import torch, triton, triton.language as tl

@triton.jit
def addk(x_ptr, y_ptr, o_ptr, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    tl.store(o_ptr + i, tl.load(x_ptr + i, mask=m) + tl.load(y_ptr + i, mask=m), mask=m)

x = torch.randn(4096, device="cuda")
y = torch.randn(4096, device="cuda")
o = torch.empty(4096, device="cuda")
addk[(4,)](x, y, o, 4096, BLOCK=1024)
torch.cuda.synchronize()
print("TRITON_OK", (o - (x + y)).abs().max().item())
```

判定：打印 `TRITON_OK` 即当前 triton 可用；报 `PassManager::run failed` / 提到 `compiler_hcu.py` ⇒ 必须走第 2 步。

### 2. 优先用镜像内 triton，其次 DAS 轮子，最后源码重编译

```bash
python -c "import triton;print(triton.__version__, triton.__file__)"
python tools/dcu/probe_triton.py                       # 镜像自带（可能是 Hygon 3.4.0）
pip install --no-deps /mnt/game/port/wheels/triton-3.6.0+das185.dtk2604.torch2100*.whl
python tools/dcu/probe_triton.py                       # DAS 3.6.0
```

两者都失败才源码重编译（第 3 步）。

> 另有一个现成的 gfx906 维护版可供**对照/补丁来源**：`nlzy/triton-gfx906`（最高 3.5.0，只作参考、不作基座）——理由与逐点对照见本文档 §4。

### 3. 源码重编译（把归档补丁移植到目标版本）

```bash
cd /mnt/game/port && git clone -b v3.6.0 --depth 1 https://github.com/triton-lang/triton triton-3.6.0-src
cd triton-3.6.0-src
# 将 artifacts/triton-gfx906.patch 的三处改动按 3.6.0 的代码结构重新落位：
#   third_party/amd/include/TritonAMDGPUToLLVM/TargetUtils.h   → ISAFamily 增加 VEGA
#   third_party/amd/lib/TritonAMDGPUToLLVM/TargetUtils.cpp     → GK_GFX906→VEGA；supportsVDot 加 VEGA
#   third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp      → getWarpSize / warpReduce 走 VEGA(64)
source /opt/dtk/env.sh
export TRITON_HIP_LLD_PATH=/opt/dtk/llvm/bin/ld.lld
export TRITON_CACHE_DIR=/mnt/game/triton-cache
export MAX_JOBS=6
pip install . --no-build-isolation --no-deps 2>&1 | tee /mnt/game/port/triton_build_360.log
```

构建放 `/mnt/game`（历史量级：3.4.0 用 `MAX_JOBS=6` 约 14 分钟、产物 277 MB；3.6.0 预计更长）。

### 4. 参考实现：`nlzy/triton-gfx906`（**只作参考/补丁来源，不作构建基座**）

仓库：<https://github.com/nlzy/triton-gfx906>（描述：triton for AMD gfx906 GPUs, e.g. Radeon VII / MI50 / MI60；上游凭据：`Said-Akbar/triton-gcn5`）

已核对事实（2026-10-07，经 GitHub API / raw 文件读取）：

| 项 | 实测 |
|---|---|
| 分支 | `gfx906/main`（`python/triton/__init__.py` = `3.4.0`）、`gfx906/v3.3.x`、`gfx906/v3.4.x`、`gfx906/v3.5.x`（= `3.5.0`） |
| Tag | `v3.4.0+gfx906`、`v3.4.0+gfx906.update1`、`v3.5.0+gfx906` |
| 最高版本 | **3.5.0** —— 本仓库/本栈需要 **3.6.0**，直接当基座会版本不符 |
| 安装方式 | **无预编译轮子**，只能源码构建：`pip3 install -r python/requirements.txt && pip3 install --no-build-isolation .` |
| 面向栈 | 社区 ROCm + gfx906（README 未涉及 DTK/hyhal，也不认 `gfx916` 命名） |

其 gfx906 改动集中在 `third_party/amd/` 的三个文件，与我们的补丁**目标一致、结构不同**：

| 点 | `nlzy/triton-gfx906` | 我们的 `artifacts/triton-gfx906.patch` |
|---|---|---|
| `ISAFamily` | 新增 `VEGA20` | 新增 `VEGA` |
| `deduceISAFamily` | `GK_GFX906 → VEGA20` | `GK_GFX906 → VEGA` |
| `supportsVDot()` | 含 `VEGA20` ✅ | 含 `VEGA` ✅（**sdot4 依赖此项**） |
| warp 64 | 通过把 `VEGA20` 放进 **`isCDNA()`** 间接得到 | 在 `getWarpSize()` / `warpReduce()` **显式**判定 VEGA |
| `warpReduce` DPP | 走 CDNA 路径（仅 `isCDNA() && ISAFamily::CDNA1` 早退） | 显式让 VEGA 走 DPP 广播路径 |
| `supportsDirectToLdsLoadBitWidth` | `VEGA20` 返回 32 位可用（**我们补丁未含此 hunk**，可对照补齐） | 走 default（`false`） |
| `isCDNA()` 对 gfx906 | **True** | **False** |

**结论与用法**：

1. ✅ **可作为参考实现与补丁来源**：两边最关键的成败点（warp 64 + `supportsVDot`）一致，佐证"gfx906 必须开 VDOT"是共识；`supportsDirectToLdsLoadBitWidth` 等我们未覆盖的 hunk 可从这里对照补。
2. ❌ **不要用它替换 DTK 构建基座**：版本只到 3.5.0（我们需要 3.6.0），且面向社区 ROCm；DTK 的 triton 必须用 DTK 的 LLVM / `ld.lld` 重编（见第 3 步），换基座会同时踩版本与运行时两个坑。
3. ❌ **不要采用它"把 `VEGA20` 折进 `isCDNA()`"的结构**：那会让所有 CDNA 假设（含 MFMA 相关优化）在 gfx906 上生效，正是 blog 与本仓库都警告过的风险点。保持"VEGA 独立、`isCDNA()` 为 false"的保守结构，且我们这份补丁已被生产的 INT8 sdot4 验证。
4. ⚠️ 若从该仓库**移植任何 hunk**：必须 (a) 落到 3.6.0 的对应位置；(b) 重跑本步门禁；(c) 补一次 INT8/sdot4 数值回归；(d) 最终 diff 归档为 `triton-gfx906-3.6.0.patch`。

**枚举其完整 gfx906 改动集（用于对照，只在 `/mnt/game` 下操作）**：

```bash
git clone --depth 1 -b gfx906/v3.5.x https://github.com/nlzy/triton-gfx906 /mnt/game/port/triton-nlzy-3.5
git clone --depth 1 -b v3.5.0 https://github.com/triton-lang/triton /mnt/game/port/triton-upstream-3.5
diff -ru /mnt/game/port/triton-upstream-3.5 /mnt/game/port/triton-nlzy-3.5 --exclude=.git > /mnt/game/port/nlzy-gfx906.diff
wc -l /mnt/game/port/nlzy-gfx906.diff
```

### 5. 必须遵守的约束

- ❌ 不换成 PyPI 的 `triton-rocm`（无 VEGA 补丁，会同时废掉 sdot4）；
- ❌ 不替换 DTK 的 triton 后端目录（`compiler_hcu.py` 等），只能在其基础上重编译；
- ✅ 补丁落地后把最终 diff 存回 `artifacts/`（命名 `triton-gfx906-3.6.0.patch`），便于回滚与复用。

## 验收（门禁）

1. `python tools/dcu/probe_triton.py` → `TRITON_OK`（误差 < 1e-5）；
2. fork 的两个 gfx906 Triton attention 测试通过：
   `tests/kernels/attention/test_triton_unified_attention_gfx906.py`、`tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py`；
3. 8K 上下文服务日志出现 `TRITON_ATTN`（与 [S8](z100-08-attention-and-quant-paths.md) 联测）；
4. 若使用自编译 triton，记录 `triton.__version__`/源码 commit 与构建日志路径。

## 失败处置

| 现象 | 处置 |
|---|---|
| 探针报 `PassManager::run failed` | 该 triton 无 gfx906 后端 → 换 DAS 轮子或自编译 |
| 自编译后探针通过但 sdot4 变慢/失效 | 检查 `supportsVDot()` 是否包含 VEGA（这是唯一开关） |
| `TRITON_HIP_LLD_PATH` 拼错 | 确认 `/opt/dtk/llvm/bin/ld.lld` 存在；`env.sh` 已设默认 |
| 首次 JIT 编译极慢（1000+ 缓存文件） | 正常；确认 `TRITON_CACHE_DIR` 指向 `/mnt/game` 且可写 |

## 参考

- 补丁：`/home/lyh/server/gfx906_port/artifacts/triton-gfx906.patch`
- 构建日志：`/home/lyh/server/gfx906_port/artifacts/triton_build.log`
- 前序计划 §5.1
- 参考实现（另一套 gfx906 补丁写法，最高 3.5.0）：<https://github.com/nlzy/triton-gfx906>（分支 `gfx906/v3.5.x`、tag `v3.5.0+gfx906`；上游凭据 `Said-Akbar/triton-gcn5`）
