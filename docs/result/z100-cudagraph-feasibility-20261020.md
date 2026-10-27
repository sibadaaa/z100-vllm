# Z100（gfx906）能否开启 CUDA graph 加速：只读调研报告

> 日期：2026-10-20（Asia/Shanghai）
> 范围：本机 4×海光 Z100/Z100L（gfx906）+ `/home/lyh/z100-vllm`（vLLM 0.18.1 + DTK 26.04 栈）
> 方法：**只读**——静态阅读源码/配置/历史归档与运行日志，只读枚举卡信息（`hy-smi` 查询、sysfs、`lspci`、`nm`）。
> 调研期间**未跑任何 GPU 负载**（无 serve/bench/pytest/微基准/roofline/profiler），**未重启或改动生产服务**，**未修改任何卡配置**。现有生产服务全程在跑。
> 结论标注：**【实测】**=本机日志/归档或只读命令可直接证实；**【推断】**=由实测事实推导、尚未在本机直接验证。
> 相关归档：[DTK/gfx906 适配记录](../archive/hygon_z100.md) · [生产部署配方](../archive/海光Z100-生产部署配方.md) · [S14 TP=4/graph](../archive/result/z100-14-27b-tp4-result.md) · [S14c 内存归因](../archive/result/z100-14c-tp4-graph-memory-result.md) · [S14d 可复现性](../archive/result/z100-14d-determinism-result.md) · [S23 O7 复核](z100-fp8-emu-optimization-result.md) · [S20 固定开销](../../.s20/S20_FIXED_COST.md)
> 行号约定：代码行号以 2026-10-20 当前工作树为准（工作树含未提交改动，后续行号可能漂移；引用同时给函数/文件名）。

---

## 0. 结论摘要（先看这 7 条）

1. **卡、驱动、DTK 运行时这一层：能开。** 4 张 Z100/Z100L 在 KFD 里都是 `gfx_target_version=90106`（gfx906），驱动 `hycu 6.4.2-V2.1.0` 的 HIP 运行时导出了完整的 `hipGraph*` API（生产用的 DTK 26.04 用户态里 `libamdhip64.so` 有 95 个 `hipGraph` 符号）；本机历史上**已经实测过** HIP graph 的 capture+replay 成功（0.6B 单卡 PIECEWISE 51/51 + FULL 35/35；27B TP=2/TP=4 也都起过图）。详见 §3。**【实测】**
2. **当前生产配方（Qwen3.8-27B-FP8 + DFlash k=7 + GDN hybrid + TP=4 + maxlen 262144 + DTK 26.04/torch 2.11）现在开不起来。** 唯一已定位的硬阻塞不是图能力，而是 fork 的 FP8-emu 代码里有一处 Dynamo 无法追踪的 **data-dependent branch**（`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py:163` 的 `if inf_codes.any():`）：去掉 `--enforce-eager` 后，4 个 worker 在 `torch.compile` 初始化阶段全部报 `torch._dynamo.exc.Unsupported: Data-dependent branching`，进程在捕获图之前就退出。S23 O7 于 2026-10-19 14:00 实测复现（4/4 worker），见 §5。**【实测】**
3. **这个阻塞是可修的，且改动极小。** 把该 tensor 条件改为 Python 级静态分支（e4m3fn 本来就没有 inf 编码，`if fp8_dtype == torch.float8_e5m2:` 即可，逐位等价），就能让编译路径继续往下走；但**修完之后是否还有第二个 Dynamo graph break 未知**（O7 是 time-box 到第一处失败就关闭的）。**【推断】**
4. **"开 cudagraph"在本仓库里等于"开 torch.compile + PIECEWISE graph"这一整包，不是只加一层图。** `--enforce-eager` 同时关掉编译与图（`vllm/config/vllm.py:1861-1877`）；历史同机 2.44× 的收益（S14，TP=4，49.66 vs 126.83 ms/token）是"编译+图"相对"纯 eager"的整包收益；S9b 已证明在 PIECEWISE 之上再叠 FULL 图只多 0.5%。所以**真正值得测的对照是三臂：eager / 仅编译 / 编译+PIECEWISE**，不要把它当成"图本身能提速 2.44×"。**【实测 + 推断】**
5. **收益在当前栈上不能外推。** 当前生产是 DFlash 投机解码（accept 7.0、tok/step 7.53）且 S23 已把 host 路径优化到 c=1 step p50 ≈ 138.7 ms（warm p50 137.8 ms）；S20 在同机 profile 到 launch/dispatch 气泡只占 **20.4%**（GPU idle 54.9/269.5 ms）。即使图/编译把这些气泡全部吃掉，c=1 的理论上限也是两成左右，c≥4 时更小；历史 2.44× 不能作为本次预期。**【推断，依据 §7】**
6. **资源上当前不满足"就地开"。** 生产 4 卡在跑（VRAM 91–92%），宿主 31 GiB 已用 24 GiB（available ≈6.8 GiB）、swap 4 GiB 已用 2.1 GiB；而 TP=4 graph 的历史结论是**每 worker 匿名常驻 +1.44 GiB、净 +3.5–4.3 GiB，需要额外 swapfile**（S14c）。**必须先开维护窗口 + 挂 swapfile + 裁捕获桶**，不能在现网直接切。**【实测】**
7. **可复现性有硬限制。** S14d/S14e 证明 graph（含默认图配置）的输出**跨重启不可复现**（逐位一致率 2.4%，同上下文 mean ≈1.1–1.45 nat）；只有 eager 是逐位可复现口径。**graph 只能用于延迟/吞吐口径**，质量评测/回归/需要"同请求跨重启一致"的对外 SLA 必须留在 eager。当前生产 `temperature=0.7` 采样本身不确定，但上 graph 前仍需做质量对拍。**【实测 + 推断】**

一句话：**"能不能开"——卡和栈能；"现在能不能开"——当前生产配方被一行 FP8-emu 代码卡住，修完还要过宿主内存、可复现性、投机解码/长上下文三关；建议按 §9 的分阶段方案在维护窗口做影子验证，不要动现网。**

---

## 1. 分层结论

| 层级 | 结论 | 关键证据 |
|---|---|---|
| ISA / 硬件 | ✅ 支持 | 4×gfx906（`gfx_target_version=90106`）、240 SIMD、每卡 31.98 GiB；`§2.1` |
| 内核驱动 | ✅ 支持 | `hycu 6.4.2-V2.1.0`（2026-09-11），KFD/CWSR 正常；`§2.2` |
| HIP 运行时（生产 DTK 26.04 用户态） | ✅ 支持 | `hipGraph*` 95 个导出符号；`§3.1` |
| vLLM 代码路径 | ✅ 支持（有门禁） | `CUDAGraphMode` 全模式可用；attention backend 报 `ALWAYS`；DFlash 有自己的 `DFlashCudaGraphManager`；`§4` |
| 当前生产配方（FP8-emu + DFlash + GDN + TP=4 + 262144） | ❌ 暂不可用 | Dynamo `Data-dependent branching` @ `gfx906_fp8.py:163`；S23 O7 实测 4/4 worker 失败；`§5` |
| 宿主内存/swap | ❌ 现网不满足，需窗口 | available ≈6.8 GiB、swap 4 GiB（用 2.1）；TP=4 graph 历史需净 +3.5–4.3 GiB + 额外 swap；`§6.1` |
| 收益 | ⚠️ 未知，需实测 | 历史 2.44× 是"编译+图 vs eager"且是另一模型/无投机；当前栈上限 ~20% 量级；`§7` |
| 可复现性 | ⚠️ 只可用于延迟/吞吐 | S14d 2.4% 逐位一致；S14e 默认图配置同样不可复现；`§6.2` |

---

## 2. 本机配置现状（只读枚举）

### 2.1 卡与拓扑

| 项 | 实测值 | 来源 |
|---|---|---|
| DCU 数量 | 4（另有 1 张 NVIDIA GTX 1060 为干扰设备，`nouveau`） | `lspci`、`/sys/class/drm` |
| SKU | 2× ZIFANG（=Z100L，PCI DID `0x55B7`）+ 2× Z100（PCI DID `0x54B7`） | `/sys/class/kfd/kfd/topology/nodes/{1..4}/properties`、`lspci -d 1d94:` |
| ISA | 全部 `gfx906`（Vega20/GCN5.1）；`gfx_target_version=90106`、`simd_count=240`、`vendor_id=7572(0x1d94)` | 同上 |
| 显存 | 每卡 `34342961152 B = 31.98 GiB` | `/sys/class/drm/card{1..4}/device/mem_info_vram_total` |
| 当前占用 | VRAM 91–92%（已用 ≈31.4 GB/卡），温度 72–81 ℃，功耗 78–254 W | `hy-smi`（查询模式） |
| 拓扑 | 4 卡经 PCIe switch（Broadcom PEX88048）；`hive_id=0`；P2P 依赖 `HSA_FORCE_FINE_GRAIN_PCIE=1` | sysfs、`tools/dcu/env.sh:121-127` |
| DP4X 缺陷 | 卡 0/1=Z100L 安全；**卡 2/3=Z100 有 DP4X 静默错算缺陷**（仅 Triton 内核有运行时 guard） | `docs/result/z100-moe-feasibility-investigation.md:25-33`、`docs/result/F24-investigation-20261018.md` |

### 2.2 驱动与运行时栈（注意：系统 `/opt/dtk` ≠ 生产用的 DTK）

| 项 | 实测值 | 来源 |
|---|---|---|
| 内核 | `5.15.0-191-generic`（Ubuntu 22.04.5） | `uname -r` |
| DCU 驱动 | `hycu 6.4.2-V2.1.0`（dkms 构建 2026-09-11），另有 `hycu_sched/hykcl/hyhal`；`cwsr_enable`、`user_sched`、`dp4x_enable` 等模块参数存在 | `/sys/module/hycu/version`、`/opt/hyhal/.info/*`、`/sys/module/hycu/parameters/` |
| **生产用户态栈** | `/mnt/game/host-runtime/dtk2604`（`DTK-26.04-DCC2602-0317`，2026-03-17）；`hipconfig` 报 HIP 6.2.0-0 | `tools/dcu/serve_host_dflash2.sh:1-10`、`cat …/.dtk_version`、`…/bin/hipconfig` |
| 系统默认 `/opt/dtk` | → `/opt/dtk-26.10`（`DTK-26.10-release`，HIP 7.2.26365）。**生产服务不用它**：26.10 只有 `libhipblas.so.3`，而 torch/vLLM 轮子按 26.04 的 `libhipblas.so.2` 构建 | `readlink -f /opt/dtk`、`serve_host_dflash2.sh:4-6` 注释 |
| torch / vLLM | `torch 2.11.0+das.opt1.dtk2604`（`torch.version.hip=6.3.26113`）、`vllm 0.18.1+das.3266200.dtk2604`；triton 由 `PYTHONPATH=/mnt/game/triton-gfx906/install` 提供自编 gfx906 版本 | `.venv/lib/python*/site-packages/*dist-info`、`serve_host_dflash2.sh` |
| graph API | `libamdhip64.so` 导出 **95 个 `hipGraph*` 符号**（含 `hipStreamBeginCapture`、`hipGraphInstantiate`、`hipGraphLaunch`、`hipGraphExecUpdate` 等） | `nm -D /mnt/game/host-runtime/dtk2604/lib/libamdhip64.so` |

> 关键点：本机同时存在 26.04 与 26.10 两套 DTK 用户态，**只有 26.04 与当前 torch/vLLM 轮子匹配**；任何"开 graph"的验证必须在 26.04 栈（即生产 launcher 的 `DTK_ROOT`）上做，不能误用 `/opt/dtk`。

### 2.3 当前生产服务与资源占用（调研时正在跑）

服务是 transient systemd unit `z100-vllm-host.service` → `tools/dcu/serve_host_dflash2.sh`（只读 `systemctl cat` 确认）：

| 项 | 值 | 来源 |
|---|---|---|
| 模型 | `/mnt/game/models/Qwen3.8-27B-FP8`（FP8，GDN hybrid，Qwen3.5/3-Next 系） | 启动日志 |
| 投机解码 | DFlash2 k=7（`Qwen3.8-27B-DFlash2-ms`，dtype bf16，kv bf16） | 同上 |
| 并行 | TP=4（`--disable-custom-all-reduce`），DCP=1 | 同上 |
| 关键参数 | `--dtype float16 --max-model-len 262144 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.85 --enable-prefix-caching --mamba-cache-mode align --enforce-eager` | `ps`、启动日志 |
| 编译/图状态 | `enforce_eager=True`；`CompilationMode.NONE`；`CUDAGraphMode.NONE`；`splitting_ops=[]`；`cudagraph_capture_sizes=[]` | 启动日志 `[vllm.py:1873]`、`[vllm.py:2133]`、config dump |
| 启动耗时 | 23:28:23 起，23:29:55 KV 就绪、23:30:18 对外服务（≈115 s）；`model loading` 9.04 GiB/≈30 s/worker，`init engine` 20.91 s | `/mnt/game/host-serve/logs/serve-host.log:78505-78568` |
| 显存账 | 每卡启动时 free 31.24/31.98 GiB，util 0.85→27.19 GiB；consumed 9.87 GiB（weights+non-torch）、peak activation 1.11 GiB、**CUDAGraph 0.0 GiB**；KV cache **16.21 GiB / 783,993 tokens**，262144 上下文并发 2.99× | `serve-host.log:78555-78558`、`:78520-78524` |
| 宿主 | 31 GiB RAM（used 24、available ≈6.8）、swap `/swap.img` 4 GiB（used 2.1）；`vm.swappiness=60`；memlock 4 GiB；THP 3 GiB；无 hugetlb | `free -h`、`/proc/swaps`、`/proc/sys/vm/*` |

### 2.4 调研约束

- 4 卡全部被上述生产服务占用（hy-smi VRAM 91–92%），因此**本次不做任何真机实验**；所有"能不能"的结论来自：源码静态分析 + 历史归档（同为 DTK 26.04/torch 2.11 栈）+ 只读栈能力枚举 + 生产日志。
- 任何真机验证（哪怕单卡）都需要先停生产服务 → **必须维护窗口**，见 §9。

---

## 3. 能力取证：hipGraph 与栈

### 3.1 API/ABI 面：完整

- `nm -D` 静态确认生产 DTK 26.04 用户态的 `libamdhip64.so` 导出 95 个 `hipGraph*` 符号，覆盖 capture/instantiate/launch/exec-update；`/opt/dtk-26.10` 同样有 95 个。**【实测】**
- 头文件 `hip_runtime_api.h` 声明了 `hipGraphInstantiate*`、`hipStreamBeginCaptureToGraph` 等；没有发现 gfx906 专属的图 API 裁剪。**【实测】**
- 驱动侧 `cwsr_enable`（Compute Wave Save/Restore）存在且默认开启；graph replay 依赖的 ring/queue 机制与 DP4X 缺陷无关。**【实测 + 推断】**

### 3.2 本机历史实测：图确实在这台机器上跑起来过

| 归档 | 配置 | graph 结果 | 数字 |
|---|---|---|---|
| S7 `z100-07-memory-kvcache-graph-result.md:13-14,286-287` | Qwen3-0.6B，fp16，单卡（Z100L） | **PIECEWISE 51/51 + FULL 35/35 捕获成功**，8–10 s，pool 0.36 GiB | decode 34.6 → **11.6 ms/token（≈3.0×）**；当轮 eager 有 NaN bug（后修） |
| S9 `z100-09-acceptance-and-bench-result.md:10-11` | 27B W8A8 INT8，TP=2 | 修掉 `pad_for_cudagraph` 后 **PIECEWISE 可用**（默认 FULL 会在捕获期崩） | 380.70 ms/token（该轮 TP=2/maxlen 4096 口径） |
| S9b `z100-09b-full-cudagraph-result.md:7,229-243` | 27B W8A8 INT8，TP=2 | FULL 捕获 35/35，replay 确定性 | FULL 378.78 vs PIECEWISE 380.62 ms/token ⇒ **FULL 只快 0.5%** |
| S14 `z100-14-27b-tp4-result.md:12-14` | 27B W8A8 INT8，**TP=4**，maxlen 262144，无投机解码 | 默认图配置 3 次宿主 OOM；**16 GiB swap + 捕获桶 [1,2,4,8] 后成功** | **graph 49.66 vs eager 126.83 ms/token = 2.44×**（相对当时基线 121.43） |
| S14c `z100-14c-tp4-graph-memory-result.md:12,22-26` | 同上 | 内存归因 | 每 worker 匿名常驻 **+1.44 GiB**（×4≈+5.7），净 **+3.5–4.3 GiB**；推荐配方 ready **230 s**、decode **49.65 ms**、最低 MemAvailable 2.35 GiB（额外 4 GiB swapfile + `swappiness=200`） |

⇒ **"驱动/DTK 不支持 graph"这个假设可以排除**：本机在 26.04 栈上多次 capture/replay 成功，包括 TP=4 与 RCCL all-reduce 被包进图里的场景。**【实测】**

### 3.3 当前栈（torch 2.11 + FP8-emu）的单算子级证据

- `.s23/w1/O1/graph.log`：`torch=2.11.0`，FP8-emu 自定义算子 **capture+replay 逐 bit 相同**（`RESULT {"captured": true, "replay_bitwise_equal": true, ...}`）。⇒ 新的 FP8-emu 内核本身是图安全的。**【实测】**
- 但这是**算子级 harness**，不是整模型 e2e；整模型 e2e graph 在当前栈上**没有一次成功记录**（最近一次 e2e 尝试即 §5 的 S23 O7，失败在编译期）。**【实测】**

### 3.4 机器级约束（会挡 graph 的，不是驱动能力）

| 约束 | 现状 | 对 graph 的影响 |
|---|---|---|
| 宿主 RAM | 31 GiB；生产在跑时 available ≈6.8 GiB | TP=4 graph 历史峰值需要更多宿主常驻 + swap；**最大现实约束** |
| swap | 4 GiB（已用 2.1 GiB） | S14c 配方需要 8 GiB 总量 + `swappiness=200` |
| 显存余量 | 生产占 91–92%，KV 16.21 GiB | graph pool 会挤 KV；需按 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`（默认 1）重估，或用 `cudagraph_capture_sizes` 限桶 |
| DP4X 卡 2/3 | 仅 Triton 内核有 VGPR guard；RCCL allreduce VGPR=248 不在 guard 覆盖内 | 不影响"能否 capture"，但图上跑的内核仍受此约束；**禁止对 DP4X worker 挂 profiler** |
| 长上下文 262144 | KV 16.21 GiB / 783,993 tokens，2.99× 并发 | graph 池与捕获桶会压缩 KV；长上下文 OOM 是首要失败模式 |
| 启动时间 | 现网 ready ≈115 s（eager） | 开编译后历史 ready 230 s（省内存图配置）到 351 s+；需窗口余量 |

---

## 4. 代码路径与开关（本仓库实测代码位置）

### 4.1 三个开关决定一切

| 开关 | 作用 | 代码位置 |
|---|---|---|
| `--optimization-level`（默认 **O2**） | O2 ⇒ `mode=VLLM_COMPILE` + `cudagraph_mode=FULL_AND_PIECEWISE` | `vllm/config/vllm.py:546`、`:422` |
| `--enforce-eager` | 同时把 `mode=NONE`、`cudagraph_mode=NONE`、关 JIT warmup | `vllm/config/vllm.py:1861-1877` |
| `-cc/--compilation-config` 或 `--cudagraph-capture-sizes` / `--max-cudagraph-capture-size` | 精细控制图模式与捕获桶 | `vllm/engine/arg_utils.py:1808-1811`、`vllm/config/compilation.py:651-706` |

`CUDAGraphMode` 取值：`NONE / PIECEWISE / FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE`（`vllm/config/compilation.py:53-90`）。默认捕获桶生成规则：`[1,2,4] + range(8,256,8) + range(256,513,16)`（`compilation.py:698-706`）⇒ 默认 51 个桶，这也是历史日志里 "PIECEWISE 51/51" 的来源；投机解码会再 append 一组 uniform decode 桶。

### 4.2 PIECEWISE 的捕获边界：attention 与 GDN core 是"断点"，不进图

- PIECEWISE 要求 `mode=VLLM_COMPILE` 且 `splitting_ops` 非空（`vllm/config/vllm.py:2303` 断言）。
- 默认 `splitting_ops` 就是 attention 类算子列表，包含 `vllm::unified_attention_with_output`、`vllm::mamba_mixer*`、`vllm::short_conv`、**`vllm::gdn_attention_core`** 等（`vllm/config/compilation.py:768-793`，赋值见 `:1189`）。
- 含义：**图只包住线性层/MLP/量化等"安全段"，attention 与 48 个 GDN core 在段间 eager 执行**（S20 `S20_FINAL.md:101` 也记录了这一点）。所以即使 graph 打开，GDN/attention 的 host 开销仍在；这也解释了为什么 S9b 里 FULL 相对 PIECEWISE 只有 0.5%——瓶颈不在图启动。

### 4.3 投机解码 / GDN hybrid 的图支持

- attention backend 在本 fork 的 gfx906 上走 `TRITON_ATTN`（`vllm/platforms/rocm.py:617-624`），报告 `AttentionCGSupport.ALWAYS`（`vllm/v1/attention/backends/triton_attn.py:104`；`ROCM_ATTN` 同样是 ALWAYS，`rocm_attn.py:69`）⇒ attention 层面不挡 FULL/PIECEWISE。
- DFlash 有**自己的** graph manager：`vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py:75`（`DFlashCudaGraphManager`），由 `speculator.py:149-185` 按 `cudagraph_mode` 初始化并 capture；drafter 还专门把 query buffer 地址固定以适配图（`dflash.py:43-60`）。
- EAGLE 系 drafter 只支持 PIECEWISE（`vllm/v1/worker/gpu_model_runner.py:6235-6246`）；DFlash 走的是 v2 runner 的专用 manager。
- Mamba/GDN 的 FULL 图有额外约束（decode 每条序列占一个 mamba cache block，`compilation.py:1515-1535`）；PIECEWISE 不受此限。

⇒ **代码层面支持"TP=4 + DFlash + GDN + PIECEWISE"这条组合**；但这条组合在当前栈上从未真正跑通过（S19/S20/O7 都在更早的阶段失败），属于**未被验证的组合**，不是"已知可行"。**【实测代码 + 推断】**

### 4.4 另一条路：breakable cudagraph（不依赖 torch.compile）

本树还有 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`（`vllm/envs.py:843`）：

- 原理：不做 FX 切分/不跑 torch.compile（`vllm/config/vllm.py:931-960` 会把 `compilation_config.mode` 置 `NONE`），改为整段 stream capture，在标了 `@eager_break_during_capture` 的 attention/GDN/mamba 算子处断点、eager 执行再继续 capture（`vllm/compilation/breakable_cudagraph.py:1-20`；断点见 `attention.py:765`、`qwen_gdn_linear_attn.py:2191/2239`、`mamba_mixer.py:469` 等，仓库内共 78 处）。
- 在当前模型上，这正好绕开 §5 的 Dynamo 分支。
- **但**：代码注释明确"ROCm 上 breakable 对能用 torch.compile piecewise 的模型会回退性能，默认不启用"，只有 `DeepseekV41ForCausalLM` 默认开（`vllm/config/vllm.py:109-125`）；且 gfx906/TP=4/RCCL 捕获、DFlash 组合均未验证。适合作为"绕过编译阻塞"的实验臂，**不建议作为第一选择**。

---

## 5. 当前配方为什么开不起来：一个确定的 Dynamo 阻塞点

### 5.1 证据链

1. **S19 时代（S20 记录）**：直接去掉 `--enforce-eager` 起服务，**7 分钟没到 `/v1/models`**，被记为 "MEASURED DEAD END"（`.s20/S20_FINAL.md:101,137`；`.s20/S20_FIXED_COST.md:279`）。当时只知道"起不来"，不知道卡在哪。
2. **S23 O7（2026-10-19 14:00）**：用与生产 launcher 逐字相同的配置、只把 `--enforce-eager` 换成 `--compilation-config '{"cudagraph_mode":"piecewise"}'` 做单窗口 arm（`tools/dcu/s23_launch_piecewise.sh`、`tools/dcu/s23_arms/r_piecewise.env`）。**4 个 worker 全部在 `torch.compile` 初始化阶段失败**：
   - `torch._dynamo.exc.Unsupported: Data-dependent branching`
   - `RuntimeError: Worker failed with error 'Data-dependent branching`（`grep -c` 在 `serve-host.log` 命中 5 行，时刻 14:00:49）
   - 栈顶 user code：`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py` 的 `fp8_to_fp16_lut` → `if inf_codes.any():`（当时日志行号 154，当前工作树行号 **163**），调用链 `apply_block_scaled_mm` → `_device_lut` → `fp8_to_fp16_lut`，触发点在 Qwen3.5/3-Next 的 `in_proj_qkvz`（FP8 线性层）。
   - 归档结论：`docs/result/z100-fp8-emu-optimization-result.md:283-293`——"O7 从 cudagraph 不可启动细化为编译路径还有一处 data-dependent branch；一旦该 break 消掉，piecewise 才有复核价值"。
3. **与 O1 改动无关**：同一份 `torch.compile` 探针在 O1 开关关闭时同样报该错（`.s23/w1/O1/compile2.log` 的 `compile_control`），所以这是**树里既有的 FP8-emu 代码问题**，不是 S23 优化引入的。**【实测】**

### 5.2 为什么是它：Dynamo 不能追踪 tensor 条件

`if inf_codes.any():`（当前 `gfx906_fp8.py:163`）在 trace 时需要一个设备同步才能取 bool，Dynamo 直接判 "data-dependent branching" 并放弃 `fullgraph`。而这个分支**只在构造 256 项 LUT 时用一次**，且对 `float8_e4m3fn` 而言 `inf_codes` 恒为全零（e4m3fn 没有 inf 编码，`man==0x7 & exp==0xF` 是 NaN）⇒ 条件完全可以静态化。

### 5.3 最小修复（草案，未执行）

```diff
--- a/vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py
+++ b/vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py
@@
-    if inf_codes.any():
+    # e4m3fn has no inf encodings; only e5m2 needs the inf rewrite.
+    # Keep this a *Python* branch so Dynamo can trace it (S23 O7 blocker).
+    if fp8_dtype == torch.float8_e5m2:
         out = torch.where(inf_codes, sign * float("inf"), out)
```

- 对 e4m3fn：行为逐位不变（原分支对全零条件不做任何改写）；对 e5m2：`inf_codes` 为非零 mask，条件等价。
- 验证方式（窗口内）：`tests/kernels/test_gfx906_fp8_emu.py`（S23 记录 123 passed）+ 4 种开关组合；再跑一次 `torch.compile` 探针确认不再报 Data-dependent branching。
- **残余不确定性**：Dynamo 一次只报第一处 break；修完可能暴露第二处（模型/GDN/DFlash 路径在本机从未编译过），所以 §9 的第一关必须是"能 trace 到底"而不是"能起服务"。**【推断】**

---

## 6. 开启方法与配方（草案，**均未执行**）

### 6.1 路线 A（推荐）：修 LUT 分支 + torch.compile PIECEWISE

与生产 launcher 的 diff（其余 env/参数全部不动）：

```diff
-  --per-request-spec-decode-metrics summary --enforce-eager \
+  --per-request-spec-decode-metrics summary \
+  --compilation-config '{"cudagraph_mode":"piecewise","cudagraph_capture_sizes":[1,2,4,8,16]}' \
+  --cudagraph-metrics \
```

- **必须保留** `--disable-custom-all-reduce`（F13/F24 结论：gfx906 上 custom AR 会崩，必须走 RCCL；S14 的 graph TP=4 也是在 RCCL 下跑通的）。
- 建议先显式给 `cudagraph_capture_sizes`（如 `[1,2,4,8,16]`），把图池压到 S14c 实测的 0.22 GiB 量级；等验证通过后再试默认 51 桶。
- 固定 `VLLM_CACHE_ROOT`（现网 `/mnt/game/vllm-cache-host`），保证 AOT 重载；首次换配置会现场编译。
- 宿主侧按 S14c 配方：额外 4 GiB swapfile + `vm.swappiness=200`（窗口内临时，结束后回 60）。
- 观测点：`Graph capturing finished … took X GiB`、`CUDA graph pool memory`、`--cudagraph-metrics` 的 `cudagraph_mode=PIECEWISE` 行、ready 时间、`/proc/meminfo` 的 MemAvailable 最低值、`hy-smi` VRAM。

### 6.2 路线 B（备选）：breakable cudagraph，绕过 torch.compile

```bash
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
# launcher 仍去掉 --enforce-eager，并给：
--compilation-config '{"cudagraph_mode":"piecewise"}'
```

- 由于 `_maybe_enable_breakable_cudagraph()` 会把 `mode` 置 NONE，**§5 的 Dynamo 阻塞不再适用**，理论上可绕开。
- 代价/风险：ROCm 上该路径默认被认为比 compile piecewise 慢（`vllm/config/vllm.py:113-117` 注释）；gfx906 + TP=4 + RCCL 捕获 + DFlash 组合**零验证**；没有 inductor pass 融合。适合作为"快速验证图能不能跑"的旁路，不适合直接上生产。**【推断】**

### 6.3 三臂对照（真正该测的东西）

| 臂 | 配置 | 目的 |
|---|---|---|
| E0（现网） | `--enforce-eager` | 基线：138 ms 级 step、质量/回归参照 |
| E1（仅编译） | `-cc.mode=vllm_compile -cc.cudagraph_mode=none`（同一 LUT 修复后） | 分离"编译"与"图"的贡献——历史上从未在 Z100 上单独测过 |
| E2（编译+PIECEWISE） | §6.1 | 目标形态；与 E0/E1 同窗口对比 |

> 说明：`--enforce-eager` 关的是"编译+图"一整包；只测 E0 vs E2 会把编译收益错误归给 cudagraph（S9b 已证 FULL 只值 0.5%，提示大头可能在编译/pass 融合上）。

---

## 7. 预期收益：不能拿历史 2.44× 直接外推

### 7.1 历史数字（同机，但栈/模型/口径不同，禁止跨栈混算）

| 来源 | 日期 | 栈 | 模型/量化 | 投机解码 | TP | 对照 | 结果 |
|---|---|---|---|---|---|---|---|
| S7 | 2026-10-08 | DTK 26.04 + torch 2.11 | Qwen3-0.6B fp16 | 无 | 1 | eager(有 NaN bug) vs 编译+图 | 34.6 → 11.6 ms/token |
| S9b | 2026-10-09 | 同上 | 27B W8A8 INT8 | 无 | 2 | PIECEWISE vs FULL | 380.70 vs 378.78（FULL +0.5%） |
| S14 | 2026-10-09/10 | 同上 | 27B W8A8 INT8 | 无 | 4 | eager vs 编译+PIECEWISE | **126.83 → 49.66 ms/token（2.44×）** |
| S23 O7 | 2026-10-19 | 同上（当前树） | **27B FP8-emu** | **DFlash k=7** | 4 | 尝试编译+PIECEWISE | ❌ 编译期失败（§5） |

### 7.2 当前栈的锚点

- 现网 eager：S23 O6 同窗口实测 c=1 step **p50 138.7–140.3 ms（warm p50 137.8–142.3 ms）**、输出 24.95 t/s（warm 29.09）、accept 7.00、tok/step 7.53（`docs/result/s23-o6-o2-recommended.md:20-24`）。
- S20 对旧生产臂的 profile：每 step **2509 个 GPU event、4379 次 HIP API、13 669 次 ATen op**，GPU idle **54.9/269.5 ms = 20.4%**，engine 线程 CPU busy 70.8%（`.s20/S20_FIXED_COST.md:20,249`）。
- S23 O1/O2 已把 host 调用税削掉约 13.9 + 3.3 ms/step（`z100-fp8-emu-optimization-result.md:18,76-80,168-170`）。

### 7.3 判断

- **c=1**：如果图+编译能吃掉大部分 launch/dispatch 气泡，上限约为 step 的 **15–20%**；实际大概率更低（部分 idle 是真依赖/同步，且 GDN/attention 按 §4.2 仍在图外 eager）。**【推断】**
- **c≥4**：S23 实测 c=16 forward 1052 t/s、GPU-bound，图的增量会更小。**【推断】**
- **TTFT/长上下文**：prefill 的 piecewise 图能减少 launch 开销，但 262144 长上下文主要成本在 attention/GEMM，图不改变算力；S14c 只测了 decode。**【推断】**

⇒ **本次上 graph 的合理期望是"个位数到 ~15% 的 c=1 改善"，不是历史 2.44×**；是否值得做，取决于窗口实测。

---

## 8. 代价与风险

### 8.1 宿主内存/swap（第一道硬门）

- S14c 实测：TP=4 graph 每 worker 匿名常驻 **+1.44 GiB**（×4 ≈ +5.7 GiB），EngineCore 反向抵消后**净 +3.5–4.3 GiB**；分配需求比 eager 多 **7–13 GiB**；峰值在"就绪后的稳态"，不在启动尖峰。
- 现网：31 GiB RAM（available ≈6.8 GiB，生产已吃 24 GiB）、swap 4 GiB（用 2.1 GiB）。**不满足**。
- 缓解：窗口内挂 4 GiB swapfile + `swappiness=200`（S14c 配方 ⇒ 余量 2.35 GiB、ready 230 s、decode 49.65 ms）；同时用捕获桶 `[1,2,4,8]` 把图池压到 0.22 GiB。
- 注意 `swappiness=200` 是**宿主级**参数，窗口结束必须回 60；swapfile 窗口结束要 `swapoff`。

### 8.2 可复现性（第二道硬门）

- S14d：7 arm / 21 pair / 252 prompt-pair，**仅 6 个（2.4%）逐位一致**；同 cache 两次 AOT 重载也 12/12 不一致；同上下文 mean ≈1.3 nat、max 10–16 nat（`z100-14d-determinism-result.md:9-11`）。
- S14e：**默认图配置（FULL_AND_PIECEWISE）同样不可复现**（三跑落两个离散变体、A vs B 差 403/512 位置）；"省内存桶"不是原因（`z100-14e-reproducibility-boundary-result.md:11-21`）。
- 结论：**graph 只用于延迟/吞吐口径；质量/精度/回归/需要跨重启一致的 SLA 必须用 eager**。当前生产 `temperature=0.7` 本身不确定，但若对外承诺"同请求可复现"或做质量门禁，不能上 graph。
- 额外：S14b 显示编译模式会实质改变 logits（27B 上 mean 1.35–1.38 nat），但 4B 真值锚显示编译+图本身只值 0.05–0.11 nat，放大来自 INT8 动态量化的 64 层递推；**上 graph 前要做固定批组合的 logprob/质量对拍**。

### 8.3 DP4X / RCCL / 长上下文

- 卡 2/3 是 DP4X 缺陷卡：仅 Triton 内核有 `num_warps` guard（`vllm/platforms/hygon_dp4x.py`），C++ HIP 内核与预编译库（RCCL/MIOpen/rocBLAS）**不在覆盖内**，理论上仍有 VGPR 取整到 256 的灰区。需要强调：**F24 的 RCCL VGPR=248 故障是在自编 ROCm 10.1（DP4X-unaware 用户态）上测到的**；当前生产用 DTK 26.04（DP4X-aware 用户态）+ guard，TP=4 RCCL 已稳定运行（现网服务即是），所以它不是"当前已知故障"，而是"不该放松的底层约束"。graph 不改变内核 VGPR，但会延长单次 launch 的生命周期；**禁止对 DP4X worker 挂 profiler/debugger**（CWSR 会触发 page fault）。
- 长上下文 262144：graph 池 + 编译 peak 会挤 KV（现网 16.21 GiB）；`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 默认 1 会把图内存计入 profiling，但配置桶过大会直接 OOM（S14 默认桶 3 次 host OOM 即此类）。

### 8.4 启动时间 / 维护窗口

- 现场编译 + 捕获会让 ready 从 115 s 涨到 **230–351 s+**（S14c/S9b），换参数还会触发新 AOT key；窗口要给足 ≥1 h，且失败要能一键回滚。

### 8.5 路线 B 的额外风险

- ROCm 上 breakable 默认被认为比 compile piecewise 慢（代码注释），且本机零验证；RCCL 捕获、DFlash drafter 的 breakable wrapper、TP>1 的 stream 语义均未验证。只能当"可行性探测"，不能直接当生产方案。

---

## 9. 建议方案（分阶段，均需维护窗口）

### 阶段 0（已完成，本次）：只读可行性调研

- 产出本报告；不占卡、不动服务、不改配置。

### 阶段 1（维护窗口 ~2–3 h）：影子验证"能不能跑起来"

1. 停生产 unit（`systemctl --user stop z100-vllm-host.service` 或按 S24 驱动流程），确认 4 卡 VRAM 归零、宿主 available 恢复。
2. 挂 4 GiB swapfile + `swappiness=200`（记录基线 `free`/`MemAvailable`）。
3. 在**工作树内本地**应用 §5.3 的 1 行修复（不提交、不进生产 unit；先跑 `tests/kernels/test_gfx906_fp8_emu.py` 与 `python -m compileall` 级检查）。
4. 用 `systemd-run --user` 起影子 unit（端口 8001，4 卡，`-cc.cudagraph_mode=piecewise --cudagraph-capture-sizes "1 2 4 8 16" --cudagraph-metrics`），观察：
   - 是否越过 Dynamo（出现 `Graph capturing` 而不是 `Data-dependent branching`）；
   - 是否出现第二处 graph break；
   - 捕获桶数/pool 大小、ready 时间、宿主 MemAvailable 最低值；
   - `hy-smi` VRAM 与 KV cache 大小。
5. 冒烟：固定 8 条 greedy 请求 + 1 条 8K/32K 长上下文；与 eager 臂做 logprob/文本对照（按 S14b 方法）。
6. **门禁**：4 卡 capture 成功、无 OOM、KV>0、质量差异在噪声底内、ready ≤15 min。任何一条不过 ⇒ 停止，回滚，记录原因。

### 阶段 2（维护窗口 ~2–4 h）：三臂性能 A/B（E0/E1/E2）

- 同窗口、同 `.so`、同缓存、同 prompt 集；测 c=1 / c=4 / c=16 的 step p50、TTFT、ITL、输出 t/s；每臂 ≥3 次重复；同时记宿主 `MemAvailable` 最低值。
- **采纳门槛（建议）**：E2 相对 E0 的 c=1 step p50 改善 ≥10%、c≥4 无回归、内存余量 ≥2 GiB、质量对拍通过；否则收益不足以承担复杂度。
- 若 E1（仅编译）已拿到大部分收益，则**不必上 graph**，只保留编译（但这需要同样的 LUT 修复，见 §5）。

### 阶段 3（可选）：生产切换

- 仅在阶段 2 达标时考虑；SLA 写明"重启/不同批组合输出不保证一致"；质量/回归流水线固定用 eager。
- 保留一键回滚。

### 回滚（任何阶段）

```bash
systemctl --user stop z100-vllm-host.service
# 恢复 unit 的 ExecStart（serve_host_dflash2.sh，带 --enforce-eager）
systemctl --user start z100-vllm-host.service
swapoff <window-swapfile>; sysctl vm.swappiness=60
```

> 现网 unit 是 transient unit；S24 驱动每次窗口会重建它，回滚即"回到 `tools/dcu/serve_host_dflash2.sh` 默认参数"，不涉及代码回退。

---

## 10. 待验证清单（窗口内逐条打勾）

| # | 项 | 判据 |
|---|---|---|
| V1 | LUT 静态分支修复后 Dynamo 可 trace | 不再出现 `Data-dependent branching` |
| V2 | 修完后是否有第二处 graph break | 能进入 `Graph capturing` |
| V3 | PIECEWISE 捕获桶与 pool 大小 | 桶数符合预期、pool ≤0.3 GiB（限桶） |
| V4 | TP=4 下 RCCL allreduce 图内正确性 | 4 卡输出与 eager 对拍（S14 方法） |
| V5 | 卡 2/3（DP4X）在该图下的表现 | 无 vmfault/XID、数值与卡 0/1 一致 |
| V6 | DFlash drafter 的图路径 | accept 率、tok/step 与 eager 一致；无 assert |
| V7 | 宿主内存 | 峰值 MemAvailable ≥2 GiB；无 OOM；swap 用量可控 |
| V8 | 长上下文 262144 | 8K/32K 冒烟 + KV 自动算出 >0 |
| V9 | 收益 | E0 vs E1 vs E2 的 c=1/c=4/c=16 数据 |
| V10 | 可复现性边界 | 同配置两次重启的 token 流差异（预期：不可复现，仅记录） |

---

## 11. 引用索引

**本机实测/只读命令**
- `/sys/class/kfd/kfd/topology/nodes/{1..4}/properties`（gfx_target_version=90106、simd=240、DID 0x55B7/0x54B7）
- `/sys/class/drm/card{1..4}/device/mem_info_vram_{total,used}`（34,342,961,152 B；91–92%）
- `/sys/module/hycu/version`、`/opt/hyhal/.info/*`（hycu 6.4.2-V2.1.0）
- `/mnt/game/host-runtime/dtk2604/.dtk_version`（DTK-26.04-DCC2602-0317）、`…/bin/hipconfig`
- `nm -D /mnt/game/host-runtime/dtk2604/lib/libamdhip64.so`（95 个 hipGraph 符号）
- `systemctl cat z100-vllm-host.service`（MBT=4096、PREFILL_CFG=0、ExecStart=serve_host_dflash2.sh）
- `/mnt/game/host-serve/logs/serve-host.log:78505-78568`（本次启动）、`:78520-78558`（KV/显存）、`:77224-77228`（eager 关闭编译/图）、`:46781-47100`（S23 O7 4 worker Data-dependent branching）
- `free -h`、`/proc/swaps`、`/proc/sys/vm/swappiness`、`ulimit -l`

**代码位置（当前工作树）**
- `vllm/config/compilation.py:53-90`（CUDAGraphMode）、`:651-706`（capture sizes/max）、`:768-793`（splitting ops）、`:1189`、`:1515-1535`（Mamba FULL 约束）
- `vllm/config/vllm.py:109-125`（breakable 的 ROCm 默认）、`:422`（O2=FULL_AND_PIECEWISE）、`:546`（默认 O2）、`:931-960`（breakable 启用）、`:1861-1877`（enforce-eager）、`:2303`（PIECEWISE 断言）
- `vllm/platforms/rocm.py:617-624`（gfx906 选 TRITON_ATTN）、`:1261-1270`（CP>1 时 FULL→PIECEWISE）
- `vllm/v1/attention/backends/triton_attn.py:104`、`rocm_attn.py:69`（AttentionCGSupport.ALWAYS）
- `vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py:75`、`speculator.py:149-185`、`vllm/v1/spec_decode/dflash.py:43-60`
- `vllm/v1/worker/gpu_model_runner.py:6235-6246`（EAGLE 仅 PIECEWISE）、`:5556-5575`（breakable wrapper）
- `vllm/compilation/breakable_cudagraph.py:1-20` 及 `eager_break_during_capture` 断点（attention/GDN/mamba 共 78 处）
- `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py:133`（`fp8_to_fp16_lut`）、`:163`（`if inf_codes.any()`，阻塞点）、`:555/:712`（调用点）
- `vllm/envs.py:843`（`VLLM_USE_BREAKABLE_CUDAGRAPH`）、`:2362`（`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`）

**历史归档**
- `docs/archive/result/z100-07-memory-kvcache-graph-result.md:13-14,286-287,306`（S7 捕获成功/3.0×）
- `docs/archive/result/z100-09-acceptance-and-bench-result.md:10-11,368`（S9 PIECEWISE）
- `docs/archive/result/z100-09b-full-cudagraph-result.md:7,229-243`（FULL +0.5%）
- `docs/archive/result/z100-14-27b-tp4-result.md:12-14`（2.44×；16 GiB swap）
- `docs/archive/result/z100-14c-tp4-graph-memory-result.md:12,22-26`（+1.44 GiB/worker；230 s 配方）
- `docs/archive/result/z100-14d-determinism-result.md:9-11`（2.4% 可复现）
- `docs/archive/result/z100-14e-reproducibility-boundary-result.md`（默认图配置同样不可复现）
- `docs/result/z100-fp8-emu-optimization-result.md:283-293`（S23 O7）
- `.s23/w1/O6/wR/r_piecewise/startup.log`（Data-dependent branching 全栈）
- `.s23/w1/O1/graph.log`（FP8-emu 算子 graph capture+replay 逐 bit 相等）
- `.s20/S20_FIXED_COST.md:20,249,279`、`.s20/S20_FINAL.md:101,137`（S19/S20 死路 + 20.4% 气泡）
- `docs/result/s23-o6-o2-recommended.md:20-24`（当前 c=1 基线）
- `docs/result/F24-investigation-20261018.md`（DP4X/RCCL 约束）、`docs/result/z100-moe-feasibility-investigation.md:25-33`（卡 2/3 DP4X）
- `tools/dcu/serve_host_dflash2.sh`（生产 launcher 与 DTK_ROOT 说明）、`tools/dcu/s23_launch_piecewise.sh`（O7 arm 的逐字 diff）
