# Z100 ROCm 10.1 调优性能未落地：原因调查

> 调查时间：2026-10-24
> 调查对象：`/mnt/game/port/gfx906-vllm-rocm101`（ROCm 10.1 + torch 2.16）
> 对照对象：`/home/lyh/z100-vllm`（DTK 26.04 + torch 2.11，分支 `dtk`）
> 结论一句话：**不是开关没开，而是只移植了 w8a16 三件套，S23 之后的调优没有合并；
> 而用来证明“已集成”的 1595 ms 数据，实际跑在 DTK 树上。**

---

## 0. 先看结论

| # | 结论 | 证据等级 |
|---|---|---|
| 1 | `runs/20261023-150120-serve` 的 TTFT 1595 ms / TPOT 39.9 ms 是 **DTK 栈**，不是 ROCm 10.1。它的 `serve_launch.sh` 里 `cd /home/lyh/z100-vllm`，解释器是 `.venv/bin/python`（torch 2.11.0），`PYTHONPATH` 也指向 DTK 树 | 已证实 |
| 2 | rocm101 树不是 DTK 树的分支或 worktree，而是一个**手工拼装的快照**：在 base 中存在的 5756 个源文件里，**5747 个与 S23 提交 `f7547bac0` 逐字节相同**，只有 9 个被改过；另有 2 个新文件 | 已证实（逐文件 hash） |
| 3 | 被改的 9 个文件里，w8a16 相关的是“新代码”，但 `BlockScaledMMLinearKernel.py`、`fp8_utils.py`、`triton_unified_attention.py` **比 S23 还旧**；说明目标树是多版本文件拼出来的，不是“S23 + 一个干净的 w8a16 patch” | 已证实 |
| 4 | 真正跑在 rocm101 上的 16:41 / 16:47 两次运行，w8a16 与 FP16 prefill **都已生效**（`W8A16 engaged=True`、`first materialization route engaged`），但 512-token prefill 用了 **2393.7 ms**，decode 每步 **238–265 ms**；DTK 同口径是 709-token prefill **1594 ms**、decode **200.5 ms** | 已证实（日志） |
| 5 | 因此问题是“**功能移植了，性能没有移植**”，而不是“功能没生效”。性能差距的主因分三层：**集成方式（拷贝而非 merge）**、**ROCm 10.1 项目的验收范围（只要求不退化）**、**ROCm 10.1 工具链/rocBLAS 的代码生成与 GEMM 选择**。第三层目前只有间接证据，需要用同口径 A/B 和 kernel trace 定界 | 前两层已证实；第三层高置信、待验证 |

---

## 1. 三个“树/栈”必须先分清

这次调查中所有“为什么没有”的困惑，首先来自把三个对象混在一起：

| 代号 | 路径 | 运行栈 | 版本管理 | 角色 |
|---|---|---|---|---|
| **T1 / DTK 树** | `/home/lyh/z100-vllm` | DTK 26.04、torch 2.11.0、triton gfx906（`/mnt/game/triton-gfx906/install`） | git，当前 `dtk` 分支 `7078b5fe1` | 之前所有调优（S16–S26、w8a16、P0 复测）的**源树**，也是 1595 ms 数据的实际来源 |
| **T2 / rocm101 树** | `/mnt/game/port/gfx906-vllm-rocm101` | ROCm 10.1、torch 2.16.0、triton 3.6.0 rocm101 | **没有 `.git`**，手工快照 | 本次 ROCm 10.1 移植的**目标树**，16:41 / 16:47 实际运行的树 |
| **T3 / rocm101 git 分支** | 本仓库 `rocm101-torch216` / `rocm101-w8a16` | 设计上也是 ROCm 10.1 | git | 记录 ROCm 10.1 适配与 w8a16 port 的分支，但**不是** T2 的实际状态 |

关键事实：

- T2 **不是** T3 的 checkout。T2 没有 `.git`，只能通过文件 hash 反推它对应哪次提交。
- T1 的 `dtk` HEAD 比 `rocm101-torch216` 多 7 个提交（`0e1475163`…`7078b5fe1`），但 T2 连 `rocm101-torch216` 的完整内容都不是。
- `docs/design/z100-w8a16-rocm101-port-plan.md` 明确把 T1 称为“源、功能超集”，T2 称为“目标、待移植”，并规定“只拷贝 w8a16 三件套，不移植 s24 动态 k、调度器改动、DTK 树的 env.sh、platforms”。这份施工图本身没有错，但它也解释了为什么 T2 只会有 w8a16，而不会有完整调优。

---

## 2. 证据链 A：1595 ms 那次不是 ROCm 10.1

`runs/20261023-150120-serve` 是“移植后性能达标”这一判断的直接依据（TTFT p50 1594.9 ms、TPOT 39.9 ms、接受率 50.6%）。但它的启动脚本和元数据都指向 DTK：

`runs/20261023-150120-serve/serve_launch.sh`：

```bash
cd /home/lyh/z100-vllm
export DTK_ROOT=/mnt/game/host-runtime/dtk2604
...
export PYTHONPATH=/home/lyh/z100-vllm:/home/lyh/vllm-test/llm-engine-probe:/mnt/game/triton-gfx906/install
exec /home/lyh/z100-vllm/.venv/bin/python -c 'import torch; ...' serve ...
```

`runs/20261023-150120-serve/run_meta.json`：

```json
"host": {
  "python": "/home/lyh/z100-vllm/.venv/bin/python",
  "python_version": "3.10.12",
  "torch_version": "2.11.0"
}
```

而真正的 rocm101 运行 `runs/20261023-164704-serve/run_meta.json` 是：

```json
"host": {
  "python": "/home/lyh/rocm101-host/venv/bin/python",
  "python_version": "3.12.14",
  "torch_version": "2.16.0.dev20261016+rocm10.1"
}
```

`runs/20261023-150120-serve` 生成于 15:01:40；当前探测仓库 `.env` 的 mtime 是 16:46:46，晚于这次运行。
也就是说，**15:01 生成的启动脚本仍是 DTK 形态**（`HOST_PYTHON`/`SRC_DIR` 还没有切到 rocm101）；
后来 16:37 起才是真正的 rocm101 启动。

因此：

- `z100-w8a16-rocm101-port-plan.md` 的 G5“TTFT ≤ 2.5 s”**没有在 rocm101 上被验证过**；被验证的是 DTK 栈。
- 15:01 的 1595 ms 不能作为“w8a16 已成功移植到 rocm101”的证据。
- 这也是“我明明看到性能变好了，为什么说没集成”的直接来源。

---

## 3. 证据链 B：rocm101 树到底是什么

### 3.1 逐文件指纹

用 T2 中每个源文件的 `git hash-object` 与 S23 提交 `f7547bac0` 的 tree 逐项比较：

- T2 中存在、且 `f7547bac0` 中也存在的文件：**5756 个**
- 逐字节相同：**5747 个**
- 不同：**9 个**

反向说：**T2 的 99.84% 源文件停在 S23**。它不是“S24 + w8a16”，也不是“HEAD 的子集”，而是“S23 的底子 + 9 个手工替换文件”。

### 3.2 被改的 9 个文件 + 新增的 2 个文件

| 文件 | T2 中的状态 | 与 S23 / HEAD 的关系 | 对性能的影响 |
|---|---|---|---|
| `csrc/rocm/fp8_emu_gemm.cu` | 已含 w8a16 + S24 prefill tile 36–40 | 等于 `4638418a3`（`rocm101-w8a16` 的 -Werror 修复），比 `dtk` HEAD 少 K4/w4 | W8A16 / prefill 物化的 kernel 本体在 |
| `csrc/rocm/fp8_emu_gemm.h` | w8a16 声明 | HEAD 同源 | 在 |
| `csrc/rocm/torch_bindings.cpp` | w8a16 算子注册 | HEAD 同源 | 在 |
| `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py` | w8a16 路由、`_PREFILL_ROUTE`、`_GEMV_MAX_M` 等 | HEAD 同源 | W8A16 路由在 |
| `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py` | prefill fp16 物化（新文件） | HEAD 同源 | 物化路由在 |
| `csrc/rocm/fp8_dequant.cuh` | dequant kernel（新文件） | HEAD 同源 | 在 |
| `vllm/envs.py` | 在 S23 基础上只加了 4 个 w8a16/prefill 旋钮 | 基于 S23，非 HEAD | 4 个开关在；S24 DYNK、S23 O2 等没有 |
| `vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py` | **没有 S23 O2/O5 的 host fast path** | **比 S23 旧** | 每步多付量化 host 包装；S23 实测约 +3.3 ms CPU/步 |
| `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | **没有 S23 O2 的 fast quant** | **比 S23 旧** | 同上 |
| `vllm/v1/attention/ops/triton_unified_attention.py` | prefill 3D 仍 `segments=2`、`MIN_CTX=0`，没有 S24 decode overrides | **比 S23 旧** | 当前 `.env` 未开 3D prefill 时不影响；S24 decode 调优默认也是 0 |
| `vllm/v1/engine/core_client.py` | rocm101 的 `InprocClient` 修补 | rocm101 专属 | 启动路径，不是性能项 |

> 注：“比 S23 旧”不是推测，是实际 diff：T2 的 `BlockScaledMMLinearKernel.py` / `fp8_utils.py` 缺少 S23 提交 `f7547bac0` 就已经含有的 O2/O5 代码；T2 的 `triton_unified_attention.py` 也缺少该提交的 T4 默认值/verify 排除。
>
> 另一个混合来源的直接证据：T2 的 `csrc/rocm/fp8_emu_gemm.cu` 等于 `4638418a3`（`rocm101-w8a16` 分支），
> 而 `vllm/v1/core/sched/scheduler.py`、`vllm/platforms/rocm.py` 等于 `f7547bac0`（S23）；
> 同一棵树里同时存在 S23 的调度器和 `rocm101-w8a16` 的 kernel。这正是“按文件拷贝”而不是“按提交合并”的产物。

### 3.3 S23 之后、但没有进入 T2 的东西

以 `f7547bac0`（S23，2026-10-17）为基线，T1 后续提交中与性能/行为相关的至少包括：

| 提交 | 内容 | T2 是否具备 |
|---|---|---|
| `2bbc67b20` | S24 prefill tile 36–40、**attention decode overrides**、**scheduler dynamic-k** | kernel/prefill 部分随文件拷贝进入；**attention override 和 scheduler 没有** |
| `8b8993b21`…`6d6af0d44` | W8A16 P0/P1、极性修复、C++ hardening、gate | kernel/路由在；test/platform 配套没有 |
| `0e1475163` | bf16→fp16 入口 coercion、gfx906 custom AR opt-in | **没有**（`config/model.py`、`platforms/rocm.py` 仍是 S23） |
| `0e1928ca4` | 允许 fp16 目标 + bf16 草稿 KV（DFlash2） | **没有**（`config/vllm.py` 仍是 S23） |
| `f46f75dbb` / `c53840d03` | P0 prefill 物化启用、路由日志、复测 | 代码文件被拷贝；验证记录没有 |
| `f5083195e` | compile-safe W8A16 routing | `gfx906_fp8.py` 已含 |
| `7078b5fe1` | 本文档所依据的移植施工图 | docs，不影响运行 |

这解释了为什么 T2 能“跑起来 w8a16”，但并不等于“带着 T1 的性能”。

---

## 4. 证据链 C：rocm101 上真实跑出来是多少

### 4.1 可比运行

| 运行 | 栈 | 关键配置 | TTFT / prefill | decode / step | 接受率 |
|---|---|---|---|---|---|
| `20261023-135729-serve` | **rocm101，无 w8a16**（施工图基线） | 709 prompt / 128 out / TP4 / c1 | TTFT p50 **4937.0 ms**；709-token prefill **4930 ms** | step p50 **192.2 ms**；TPOT **37.1 ms** | 57.5% |
| `20261023-150120-serve` | **DTK，有 w8a16 + FP16 prefill** | 709 prompt / 128 out / TP4 / c1 | TTFT p50 **1594.9 ms**；709-token prefill **1594 ms** | step p50 **200.5 ms**；TPOT **39.9 ms** | 50.6% |
| `20261023-164058-serve` | **rocm101，有 w8a16 + FP16 prefill** | 手动请求，非 probe | 193-token prefill **2032.8 ms** | **237–257 ms** | 35% 左右（波动） |
| `20261023-164704-serve` | **rocm101，有 w8a16 + FP16 prefill** | 手动 + tool parser，非 probe | 512-token prefill **2393.7 ms** | **238–265 ms** | 15–43%（波动） |

16:47 的路由证据（不是“开关没开”）：

```text
gfx906 fp8-emu route: VLLM_DCU_FP8_W8A16=1, W8A16 engaged=True (built op True), GEMV_MAX_M=8.
gfx906 fp16 prefill materialization: VLLM_DCU_FP16_PREFILL=1, role min_m {...}
gfx906 fp16 prefill: first materialization route engaged (role=in_proj_qkvz, M=8192, N=4096, K=5120)
```

### 4.2 能读出什么

1. **功能确实生效**：W8A16 与 FP16 物化都命中，不是“没编译”“没注册”“开关没传”。
2. **prefill 比 DTK 慢约 2×**：按 token 折算，DTK 709 token 用 1594 ms ≈ 2.25 ms/token；
   rocm101 512 token 用 2394 ms ≈ 4.68 ms/token。FP16 物化把 rocm101 从 4937 ms 拉到 2394 ms（约 2×），
   但仍只到 DTK 的一半左右。
3. **decode 比 DTK 慢 18%–31%**：DTK 200.5 ms/step，rocm101 238–265 ms/step。
4. **接受率数字不能直接比**：16:47 是带 tool parser 的手动请求，prompt/采样参数与 probe 不同；
   16:41 也不是标准 probe。接受率 15–43% 只能说明“需要一次干净复测”，不能单独定罪。
5. **rocm101 基线本身 decode 不差**：13:57 无 w8a16 时 step p50 192.2 ms，比 DTK 200.5 ms 还快。
   这提示：**w8a16 路线在 rocm101 上可能反而拖慢了 decode**，或者 16:41/16:47 的负载/时钟与 13:57 不可比。
   这是下一步最值得做的 A/B。

### 4.3 缺失的验证

到目前为止，rocm101 上没有一次完整的 `probe_all.sh`（steps 档）跑分：

- 16:37 第一次启动失败：缺 `torchvision`，随后 PyNCCL 报 `invalid device function`；
- 16:41 启动成功，但只做了手动 smoke；
- 16:47 启动成功，但只做了手动请求，且加了 tool parser。

因此，**“rocm101 的 G5/G6 是否通过”目前没有合法证据**。

---

## 5. 根因分层

### 5.1 流程根因：集成方式是“拷文件”，不是 merge（已证实）

施工图 S2/S3 的做法是：

```bash
cp -a $A/csrc/rocm/fp8_emu_gemm.{cu,h} $B/csrc/rocm/
cp -a $A/vllm/.../gfx906_fp8.py $B/...
cp -a $A/vllm/.../gfx906_fp16_prefill.py $B/...
```

只拷贝“看起来相关”的文件，没有把 T1 的提交历史带进 T2。结果是：

- 依赖文件不一定跟着更新（`BlockScaledMMLinearKernel.py`、`fp8_utils.py` 就比 S23 还旧）；
- 文件之间的版本不一致（kernel 是 HEAD，host 基类却是 pre-S23）；
- 无法用 `git log` 回答“T2 现在到底是哪个版本”；
- 无法用 `git bisect` 定位性能回归；
- 下次再拷一次，同样的问题会重演。

这不是“某一行写错了”，而是**集成模型选错了**。

### 5.2 范围根因：ROCm 10.1 项目本来只要求“不退化”（已证实）

`docs/z100-torch216-rocm101-adapt-plan.md` 的目标写得很清楚：

> 目标：把 DTK 栈切换到本地自编 ROCm 10.1 栈。
> 非目标：**不升级 vLLM 上游、不做新性能优化（本计划只设“不退化”门槛）**。
> DoD：27B-FP8 TP=4 可启动并完成性能 anchor：**decode ≥ 0.95×（同树基线）**。

而 `docs/result/z100-rocm101-acceptance.md` 的结论也是“打通到可生产验证”，并把
“27B-FP8 TP2 性能调优”列为**下一步**，不是已完成项。

所以从项目范围上，ROCm 10.1 树从来没有承诺“把 DTK 的 S23/S24 调优全部带过来”。
w8a16 是后来单独追加的移植（10-23），并不改变这个范围。

### 5.3 代码根因：S23 之后的调优没有合并（已证实，影响可量化）

| 缺失项 | 证据 | 预期影响 | 对本次 16:47 的实际影响 |
|---|---|---|---|
| S23 O2 量化 host fast path | T2 的 `BlockScaledMMLinearKernel.py` / `fp8_utils.py` 比 S23 旧 | S23 实测：每次 `apply_weights` 112.94→100.15 µs，约 **−3.3 ms CPU/步** | 不是 40 ms 的主因，但说明快照不完整 |
| S23 O3 decode tile（cfg35） | T2 的 `gfx906_fp8.py` 是 HEAD，`.cu` 是 `4638418a3`；O3 实际在 | DTK 同窗口 e2e **−7.34%** | **已在**，不是缺失项 |
| S24 decode attention overrides | T2 的 `triton_unified_attention.py` 无 `GFX906_DECODE_*` | 默认 0，只有显式设置才生效 | 当前 `.env` 未设，**不是本次主因** |
| S24 scheduler dynamic-k | T2 的 `scheduler.py` 与 `f7547bac0` 完全一致 | 默认 `LO=0` 关闭 | 当前未启用，**不是本次主因** |
| S26 bf16→fp16 入口 coercion | T2 的 `config/model.py` 无 `gfx906_coerce_model_dtype` | 影响 bf16 入口语义 | 本次显式 `VLLM_DCU_BF16=1`，不触发 coercion；对草稿模型的影响待测 |
| fp16 目标 + bf16 草稿 KV 校验 | T2 的 `config/vllm.py` 无该修补 | 施工图 §1.6 已确认 T2 没有这条强校验 | 当前能启动；不是性能项，但两栈的 KV 校验语义不同 |
| custom AR opt-in | T2 的 `platforms/rocm.py` 无 | 本次 `--disable-custom-all-reduce`，不涉及 | 无 |

**结论**：代码缺失可以解释“为什么不是完整调优”，但**不能单独解释 200→240 ms 的 decode 差距**；
S23 文档明确说 O2 在 c=1 不兑现 e2e，S24 的两个大项默认又是关的。真正的大头更可能在运行栈。

### 5.4 运行栈根因：同一份源码，ROCm 10.1 编译/GEMM 选择不同（高置信，待验证）

这是目前最值得怀疑、但还没有被 kernel trace 钉死的一层：

1. **编译器不同**：T1 用 DTK 26.04 的 clang，T2 用 ROCm 10.1 的 clang 24。
   同样的 `fp8_emu_gemm.cu`，机器码和 occupancy 可能不同。T2 的构建日志里已经有：

   ```text
   ld.lld: warning: ... failed to meet occupancy target given by 'amdgpu-waves-per-eu'
     in '...fp8_emu_gemm_block_kernel...': desired occupancy was 4/6, final occupancy is 1/2
   ```

   这正是 decode 最敏感的 kernel 家族。需要把 T1/T2 的同一 kernel 的 VGPR/occupancy/指令数拉出来对比。

2. **rocBLAS/Tensile 不同**：FP16 prefill 物化最终走 `torch.mm`，即 rocBLAS。
   T1 的日志明确写“本机 hipBLASLt 无 gfx906 tensile 库，`torch.mm` 实际走 rocBLAS”；
   T2 的 ROCm 10.1 是自编/TheRock 树，Tensile library 与 DTK 不同。
   施工图 §5 风险 6 已经预判：“若 G5 的 TTFT 显著劣于 2.5 s，优先排查 fp16 物化路由是否生效（G3）
   与 **rocBLAS 在 rocm10.1 的 fp16 GEMM 形状选择**”。现在正是这个现象。

3. **torch 2.16 vs 2.11 的 host/launch 开销**：decode 每步 256 次 GEMM 调用，
   Python/自定义 op 的 launch 成本会随 torch 版本变化。S23 实测 O1/O2 的 host 优化在 c=1
   被 GPU 吸收，但在 rocm101 上 GPU 变慢后，host 项可能重新暴露。

4. **窗口/时钟不可比**：S23 文档自己记录过同一树不同窗口有 25–30% 的 step 差异。
   13:57（rocm101 基线）与 16:47（rocm101 + w8a16）之间没有记录 GPU 时钟、功耗、
   是否热态；不能排除“窗口差异”贡献了一部分 192→240 ms。

> 这一层的结论目前只能是：**“同一份 kernel 在 ROCm 10.1 上可能编译/调度得更差，且 prefill 的
> rocBLAS 路径与 DTK 不同”**。要把它从“高置信”变成“已证实”，必须做 kernel 级 A/B。

### 5.5 度量根因：没有“栈身份”硬门禁（已证实）

探测框架其实记录了 `run_meta.json` 的 python/torch，但启动脚本没有在起飞前校验：

- `SRC_DIR` 是否真的是目标树；
- `HOST_PYTHON` 的 `vllm.__file__` 是否在 `SRC_DIR` 下；
- `torch.__version__` 是否匹配目标栈；
- 当前 git commit / 文件 hash 是否匹配。

于是 15:01 的 DTK 运行被当成了 rocm101 的成功，直到 16:37 才暴露出“启动的是另一棵树”。
这是流程问题，不是代码问题，但代价最大：它让错误结论多活了一个多小时。

---

## 6. 影响评估

| 维度 | 现状 | 判断 |
|---|---|---|
| W8A16 功能 | `engaged=True (built op True)` | ✅ 已生效 |
| FP16 prefill 物化 | 路由日志与 `role min_m` 表都在 | ✅ 已生效 |
| TTFT / prefill | rocm101 512 token ≈ 2394 ms；DTK 709 token ≈ 1594 ms | ⚠️ 比 DTK 慢约 2×（按 token 折算），接近施工图 2.5 s 上限 |
| decode | rocm101 238–265 ms/step；DTK 200.5 ms | ⚠️ 慢 18%–31% |
| 接受率 | 16:47 15–43% 波动；非 probe 口径 | ❓ 不可比，需干净复测 |
| 正确性 | 16:41 有 `' Paris.\nThe capital of Germany is'` smoke；16:47 有手动请求 | ⚠️ 没有在 rocm101 上跑 G4 数值门 |
| 可观测性 | 没有 rocm101 的完整 `probe_all.sh` | ❌ 缺失 |
| 可回滚性 | DTK 树/`.so` 未动，`.env.bak-dtk-20261023` 在 | ✅ 可回滚 |

---

## 7. 建议

### P0：先把“到底跑的哪棵树”钉死（半天）

1. **把 T2 变成 git 可管理对象**
   在 T2 执行 `git init` 并把当前状态提交为 `rocm101-current-snapshot`（或把它做成 T1 的
   `git worktree`）。至少让下一次改动有 diff、有 commit、有 bisect 的可能。

2. **给启动脚本加栈身份硬门禁**
   起飞前断言：
   ```text
   vllm.__file__          以 SRC_DIR 开头
   torch.__version__      匹配目标栈（DTK 2.11 / ROCm 2.16）
   triton.__version__     匹配目标 triton
   SRC_DIR 的 git commit  或文件 hash 指纹匹配预期
   ```
   任何一项不匹配就拒绝启动，而不是在 `run_meta.json` 里“记录一下”。

3. **在 rocm101 上跑一次干净 probe**
   与 15:01 完全相同的 `.env`（不加 tool parser、不加 `--enable-auto-tool-choice`），
   `PROBE_LEVEL=steps`，`CONCURRENCY=1`，512/128，repeat=3。产出一份真正的 rocm101 REPORT。
   同时记录 `hy-smi` 时钟/功耗，避免窗口差异。

### P1：用 merge/cherry-pick 取代拷贝（1–2 天）

1. 以 `f7547bac0` 为 base，建 `rocm101-perf` 分支；用
   `git rev-list --reverse f7547bac0..dtk` 的顺序 cherry-pick（docs 提交可跳过），至少覆盖：
   - `2bbc67b20`（S24 kernel + attention + scheduler）
   - `8b8993b21`…`6d6af0d44`（W8A16 P0/P1/hardening/极性修复）
   - `0e1475163`（bf16 coercion + custom AR）
   - `f46f75dbb`…`c53840d03`（P0 prefill 物化启用与复测）
   - `f5083195e`（compile-safe W8A16）
   - `0e1928ca4`（bf16 草稿 KV）
   冲突解决时保留 rocm101 专属的 `core_client.py`、`tools/dcu/env-rocm101.sh`、`platforms/` 差异。

2. 至少补齐这些文件（不要只拷 w8a16 三件套）：
   ```text
   vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py
   vllm/model_executor/layers/quantization/utils/fp8_utils.py
   vllm/v1/attention/ops/triton_unified_attention.py
   vllm/v1/core/sched/scheduler.py
   vllm/config/model.py
   vllm/config/vllm.py
   vllm/platforms/rocm.py
   vllm/platforms/hygon_dtk.py
   vllm/envs.py
   ```

3. 重新编译**四个**扩展（`_C`、`_C_stable_libtorch`、`_moe_C_stable_libtorch`、`_rocm_C`），
   不要只重编 `_rocm_C`；记录每个 `.so` 的 md5。

4. 跑施工图 S6 六门 + S24 的两个开关（先关）：
   G1 op 注册、G2 路由、G3 物化、G4 数值、G5 性能、G6 投机；
   另外加一项：`VLLM_S24_DYNK_LO=0` 与 `VLLM_GFX906_ATTN_DECODE_*` 默认值下的回归。

### P2：性能定界（1–3 天）

按“先隔离，再优化”的顺序：

1. **W8A16 开关 A/B**（同树、同窗口、同 probe）：
   `VLLM_DCU_FP8_W8A16=0` vs `=1`，`VLLM_DCU_FP16_PREFILL=1` 固定。
   如果 `=0` 的 decode 回到 192 ms，说明 W8A16 路线在 rocm101 上就是慢的，
   应立即把 rocm101 的默认值改为 `0`（或 `auto` 但排除 decode），只保留 FP16 prefill。

2. **FP16 prefill A/B**：
   `VLLM_DCU_FP16_PREFILL=0` vs `=1`，验证 2394 ms 是否已由物化贡献到极限；
   如果没有，检查 `gfx906_fp16_prefill.py` 的 `_ROLE_MIN_M` 是否与 DTK 一致。

3. **kernel 级 trace**：
   用 `rocprofv3` / torch profiler 抓一个 decode step，列出：
   - `fp8_emu_gemm_block_w8a16_kernel`（gemv/stream/tile）耗时；
   - attention decode kernel 耗时；
   - 每步 kernel 总数与 launch gap。
   与 DTK 同 shape 的 trace 并排；如果 W8A16 kernel 单项就慢 20%+，问题在编译/occupancy，
   优先调 `__launch_bounds__` / `amdgpu-waves-per-eu` / block size，而不是继续改 Python。

4. **prefill rocBLAS 定界**：
   在 T1/T2 各跑一个 `torch.mm` fp16 微基准（如 4096×5120×4096），
   对比 TFLOP/s；如果 T2 明显低，问题在 ROCm 10.1 的 rocBLAS/Tensile，
   可尝试 `TORCH_BLAS_PREFER_HIPBLASLT`、Tensile library 路径，或把 prefill 物化
   暂时回退到 W8A16 tile（用 `_ROLE_MIN_M` 控制）。

5. **接受率**：
   只有在完成 1–4 后才做。用标准 probe（不带 tool parser）对比 DFlash2 接受长度；
   如果 rocm101 仍显著低，再查 bf16 草稿 KV / `VLLM_DCU_BF16` / `gfx906_coerce_model_dtype` 的差异。

---

## 8. 复现与验收命令

### 8.1 复核“T2 是 S23 + 9 文件”

```bash
cd /home/lyh/z100-vllm
python3 - <<'PY'
import subprocess, os
B = '/mnt/game/port/gfx906-vllm-rocm101'
base = 'f7547bac0'
base_map = {}
for line in subprocess.check_output(['git', 'ls-tree', '-r', base], text=True).splitlines():
    meta, path = line.split('\t', 1)
    base_map[path] = meta.split()[2]
same = diff = 0
for root, dirs, files in os.walk(B):
    dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__')]
    for fn in files:
        if not fn.endswith(('.py', '.cu', '.cuh', '.h', '.cpp', '.hip')):
            continue
        p = os.path.join(root, fn)
        rel = os.path.relpath(p, B)
        if rel not in base_map:
            continue
        h = subprocess.check_output(['git', 'hash-object', p], text=True).strip()
        if h == base_map[rel]:
            same += 1
        else:
            diff += 1
            print('DIFF', rel)
print('same', same, 'diff', diff)
PY
```

预期：`same 5747 diff 9`，并打印 §3.2 的 9 个路径。

### 8.2 复核“15:01 是 DTK”

```bash
grep -E 'cd /home/lyh/z100-vllm|PYTHONPATH|exec ' \
  /home/lyh/vllm-test/llm-engine-probe/runs/20261023-150120-serve/serve_launch.sh
python3 -c "import json;m=json.load(open('/home/lyh/vllm-test/llm-engine-probe/runs/20261023-150120-serve/run_meta.json'));print(m['host'])"
```

### 8.3 复核 rocm101 路由与 decode

```bash
R=/home/lyh/vllm-test/llm-engine-probe/runs/20261023-164704-serve
grep -E 'W8A16 engaged|first materialization route|Iteration\(' "$R/serve.log" | head -20
```

### 8.4 建议的验收矩阵

| 门 | rocm101 目标 | 证据 |
|---|---|---|
| G1 op | `check_rocm_c_ops.py` exit 0，含 w8a16 | 容器输出 |
| G2 路由 | `W8A16 engaged=True` | serve.log |
| G3 物化 | `first materialization route engaged` | serve.log |
| G4 数值 | `' Paris.'` 语义一致 | curl 输出 |
| G5 性能 | TTFT ≤ 2.5 s；decode ≤ 45 ms TPOT | http_summary.json |
| G6 投机 | 接受长度 ≥ 4，接受率 ≥ 45% | steps_summary.json |
| G7 栈身份 | python/torch/vllm 路径与目标一致 | run_meta.json + 启动门禁 |

---

## 9. 一句话给决策者

**w8a16 的代码已经进了 rocm101 树，但“调优后的性能”没有：rocm101 树 99.8% 的源文件还是
S23，只有 9 个文件被手工替换；而 1595 ms 那次跑的是 DTK。** 下一步不是继续调 Python，
而是先把 T2 变成可 merge 的 git 树，补上 S23 之后的提交，然后用同口径 A/B 把
“W8A16 kernel 慢”与“ROCm 10.1 rocBLAS 慢”分开。

> 具体执行步骤、门禁与 A/B 矩阵见
> [`docs/design/z100-rocm101-perf-construction.md`](../design/z100-rocm101-perf-construction.md)。

---

## 附录 A：调查中使用的关键路径

| 用途 | 路径 |
|---|---|
| DTK 源树 | `/home/lyh/z100-vllm` |
| rocm101 目标树 | `/mnt/game/port/gfx906-vllm-rocm101` |
| w8a16 移植施工图 | `docs/design/z100-w8a16-rocm101-port-plan.md` |
| ROCm 10.1 适配计划 | `docs/z100-torch216-rocm101-adapt-plan.md` |
| ROCm 10.1 验收报告 | `docs/result/z100-rocm101-acceptance.md` |
| W8A16 P0 记录 | `docs/result/z100-w8a16-fp16-prefill-result.md` |
| P0 prefill 物化 A/B | `docs/result/z100-p0-prefill-materialization-result.md` |
| S23 优化结果 | `docs/result/z100-fp8-emu-optimization-result.md` |
| 15:01 DTK 运行 | `/home/lyh/vllm-test/llm-engine-probe/runs/20261023-150120-serve` |
| 13:57 rocm101 基线 | `/home/lyh/vllm-test/llm-engine-probe/runs/20261023-135729-serve` |
| 16:41 rocm101 运行 | `/home/lyh/vllm-test/llm-engine-probe/runs/20261023-164058-serve` |
| 16:47 rocm101 运行 | `/home/lyh/vllm-test/llm-engine-probe/runs/20261023-164704-serve` |

## 附录 B：本次没有做的事

- 没有在 rocm101 上跑完整 `probe_all.sh`（需要写探测仓库 `runs/`，当前会话只允许写本工作区）；
- 没有做 W8A16 0/1 的同窗口 A/B（需要重启服务并独占 4 卡）；
- 没有抓 rocprofv3 kernel trace（需要服务 + profiler 同时运行）；
- 没有修改任何运行时代码或 `.so`。

这些正是 P0/P2 要补的实验。
