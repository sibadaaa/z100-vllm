# 方案 B 第二批执行结果：B3 否决、A4 完成（trace 到手）、崩溃机制闭环

> 归档时间：2026-10-13 00:40（CST）
> 执行者：Lead（真机 A/B + torch profiler 轻量配置取证；**未改任何 vllm 源码**）
> 前置：[`z100-19`](z100-19-longthinking-collapse-result.md) · [`z100-20 方案`](../design/z100-20-shorten-silence-options.md) · [`z100-21`](z100-21-b1-closed-int8-prefill-ab-result.md)
> 状态：B1 ⛔ 已否决（z100-21）· **B3 ⛔ 否决（两条独立证据）** · **A4 ✅ 完成**（trace + 逐步拆解）· 机制 ✅ 闭环

## 1. 结论速览

| 项 | 结论 | 证据 |
|---|---|---|
| **B3** spinloop 扩展 | ⛔ **不可构建**（CMake 门禁 `Python_VERSION >= 3.11` + `USE_SABI 3.11`；运行时 Python **3.10.12**） | §2.1 |
| **B3'** 等价替代（调小读端停靠上限） | ⛔ **无效**：`recheck 5000→500` 对孤儿态探针 TTFT **无影响**（5.05 s vs 5.01 s） | §2.2 |
| **A4** torch profiler | ✅ 轻量配置（`with_stack=false`）**可用**，抓到两窗口 trace；默认（带 stack）会卡死引擎 | §3.1 |
| 每步构成（健康 vs 孤儿） | 步长 **187 ms**，其中 GPU **160–169 ms**、host 18–30 ms、步间 52 ms —— **孤儿不会拖慢每步** | §3.2 |
| 崩溃机制 | ✅ **闭环**：步长随**在跑请求的上下文长度**放大（短→225 ms，**32k→763 ms**，96k→1.6 s/token），新请求 TTFT 被抬到 15 s+，整机吞吐掉到 ~2 t/s | §4 |

## 2. B3：两条独立证据否决

### 2.1 扩展根本无法构建（Python ABI 门禁）

`CMakeLists.txt:151-165`（`/mnt/game/port/gfx906-vllm-211` 与仓库同源）：

```cmake
# This extension requires SABI 3.11 since it relies on Py_buffer support.
# Skip the target entirely on Python < 3.11 so the build doesn't break.
if(Python_VERSION VERSION_GREATER_EQUAL "3.11")
  define_extension_target(spinloop ... USE_SABI 3.11 WITH_SOABI)
endif()
```

容器内运行时：

```
$ python -V
Python 3.10.12
$ VLLM_USE_SPINLOOP_EXT=1 python -c "import vllm.distributed.device_communicators.shm_broadcast as s; print(s.SPINLOOP_EXT_ENABLED)"
WARNING [shm_broadcast.py:55] spinloop extension could not be loaded, disabling VLLM_USE_SPINLOOP_EXT!
SPINLOOP_EXT_ENABLED = False
vllm.spinloop findable: False
```

DTK 侧轮子只有 cp310（`torch-2.11.0+das.opt1.dtk2604-cp310`），所以**不是"没编"，而是这套栈上该目标被设计性跳过**。
要落地只能把扩展从 SABI 3.11 降到 3.10（`Py_buffer` → 旧 buffer 协议），属独立工程。

### 2.2 它要治的症状根本不是读端停靠（A/B）

新增可配项 `VLLM_SHM_READER_RECHECK_INTERVAL_MS`（默认 5000，保持上游行为）后做 A/B，
用可稳定复现的"孤儿 + 探针"口径：

| 臂 | 健康探针 TTFT | 孤儿态探针 TTFT（5 次） |
|---|---|---|
| `recheck=5000`（默认） | 0.76 s | 4.91 / 5.07 / 5.12 / 4.91 / 5.18 → **均值 5.05 s** |
| `recheck=500` | 0.74 s | 4.94 / 5.09 / 5.12 / 4.94 / 4.96 → **均值 5.01 s** |

⇒ 停靠上限与实测停顿无关。**据此把这次改动整体回退**（`shm_broadcast.py` 已回到零 diff；`envs.py` 只保留用户原有 S18 改动），
同时**撤回**我此前"`SHM_READER_RECHECK_INTERVAL_MS=5000` 解释了旧日志 4.9–5.9 s 停顿"的说法。

## 3. A4：torch profiler（轻量）完成取证

### 3.1 配置与产物

- 默认（`torch_profiler_with_stack=True`）：四个 worker 卡在 `torch.profiler.stop()`，引擎 10 min+ 无响应、trace 60 MB 未收尾 ⇒ 不可用。
- 改用 `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/mnt/game/prof","torch_profiler_with_stack":false,"active_iterations":5}'`：
  两窗口均成功落盘、gzip 完整、引擎自行恢复（`/health` 200）。
- 产物（保留在 `/mnt/game/prof/`，共 84 MB）：
  - 健康窗：`rank{0..3}.1791822348*.pt.trace.json.gz`（≈4.9 MB/rank）
  - 孤儿窗：`rank{0..3}.1791822397*.pt.trace.json.gz`（≈16.9 MB/rank）
  - API server 侧 `*.async_llm.*.gz` 为空（1 KB），驱动循环不可见 —— 遗留项。

### 3.2 逐步拆解（`execute_context` 用户标注，rank0）

| | 健康窗 | 孤儿窗 |
|---|---|---|
| 窗口 | 5.62 s | 10.28 s |
| step 数 | 13 | 43 |
| **步长 mean / p50** | **189.8 / 170.4 ms** | **187.3 / 172.5 ms** |
| 步内 GPU | 160.0 ms（84%） | 168.8 ms（90%） |
| 步内 host | 29.8 ms | 18.5 ms |
| 步间间隔 | 52.1 ms | 51.6 ms |
| **GPU 占空比** | **45.7%** | **90.9%** |

GPU 时间去向（孤儿窗）：`fp8_emu_gemm_block_kernel<32,64>` **4110 ms（44%）**、
`fused_sigmoid_gating_delta_rule_gfx906_mtp_update_kernel` 1482 ms、
`ncclDevKernel_AllReduce_*` 1083+706=**1789 ms（19%）**、`triton_matmul_kernel` 625 ms、`unified_attention_2d` 553 ms。
CPU 侧热点：`vllm::gdn_attention_core` 1371 ms、`aten::item`（B1 议题，占比 ~3%）。

**两条重要修正**：
1. 孤儿态里**每步并未变慢**（187 ms，与健康一致）⇒ "孤儿拖慢引擎每步"的说法**不成立**；孤儿造成的是**新请求排队/投递延迟**（TTFT 0.9 s → 5.2 s）。
2. 孤儿态 GPU **占空比 91%**，而当时 `hy-smi` 读数是 2% —— 与项目既有记录一致（"[HCU% 瞬时采样不可靠](../result/z100-11b-perf-profiling-result.md)"）：**"DCU 使用率极低"很可能是采样假象，不能作为 GPU 闲着的证据**。

## 4. 崩溃机制闭环：步长随在跑请求的上下文长度放大

新增实验（`/tmp/z100diag/multi_orphan.py`、`long_orphan.py`）：

| 在跑请求 | 引擎步长 | 新请求 TTFT |
|---|---|---|
| 无 | — | 0.59–0.90 s |
| 1 个短上下文孤儿 | 4.43 steps/s = **225 ms** | **5.22 s** |
| 2 个短上下文孤儿 | 4.27 steps/s = 234 ms | 5.40 s |
| 4 个短上下文孤儿 | 4.41 steps/s = 227 ms | 5.19 s |
| **1 个 32k 上下文孤儿** | 1.31 steps/s = **763 ms** | **15.11 s** |
| 96k 上下文（bench 实测 1.6 s/token × ~2.9 token/step） | **~2–4 s** | 分钟级 |

⇒ 完整因果链（与用户在 DSH 里看到的现象一一对应）：

```
一轮长思考（上下文 20k–100k）
  ├─ 客户端 5 分钟流空闲 watchdog 判超时 → 中止请求（连接常未干净关闭）
  │    └─ vLLM 侦测不到 → 孤儿继续生成到 maxTokens
  ├─ 孤儿上下文很长 ⇒ 每步 0.76 s（32k）→ 2–4 s（96k）而不是 0.2 s
  │    └─ 其它请求被排在同一 step 里：TTFT 0.9 s → 5.2/15.1 s，吞吐掉到 ~2 t/s
  └─ hy-smi 瞬时 DCU% 误读为 2% ⇒ 看起来像"算力突然下降、GPU 空转"
```

**静默的构成**据此定量：冷 prefill（FP8：32k=157 s、96k=600 s；INT8：32k=75 s、96k=486 s）+ 长上下文每步放大 + 新请求投递延迟。

## 5. 对方案选择的更新

- **B3 关闭**：不可构建 + 无效，不要再排期。
- **B1 关闭**（z100-21）：≈1% 收益，不改。
- **A4 关闭**：trace 已到手，机制已闭环；唯一遗留是 API server 侧 trace 为空（驱动循环看不到），如需可另开。
- **剩下真正有效的一条是 C1**（z100-21 实测）：换 INT8 `sdot4` 把 8k/32k 冷 prefill 提 2–3×，直接压短静默；代价是需过质量门禁。
- 次优（本轮量化后优先级明确）：**长上下文步长**才是吞吐天花板（32k→763 ms/步），要动它属于 attention/MoE 内核工作，不是 conv/sync 这类小修；**NCCL allreduce 占 GPU 时间 19%** 也值得单独评估（TP=4 over PCIe）。

## 6. 复现命令

```bash
# B3 证据
sed -n '146,152p' /mnt/game/port/gfx906-vllm-211/CMakeLists.txt
docker exec -e VLLM_USE_SPINLOOP_EXT=1 -e PYTHONPATH=/src z100-vllm bash -lc '. /opt/dtk/env.sh; . /src/tools/dcu/env.sh; python /mnt/game/port/diag/check_env.py'
# B3' A/B
ARM=r5000 MTP=2 bash /tmp/z100diag/serve_arm.sh && python3 /tmp/z100diag/orphan_ab.py r5000
ARM=r500  MTP=2 RECHECK_MS=500 bash /tmp/z100diag/serve_arm.sh && python3 /tmp/z100diag/orphan_ab.py r500
# A4
ARM=proflight MTP=2 PROFILER=1 PROF_LIGHT=1 bash /tmp/z100diag/serve_arm.sh
python3 /tmp/z100diag/trace_steps.py /mnt/game/prof/rank0.<ts>.pt.trace.json.gz
# 机制
python3 /tmp/z100diag/multi_orphan.py     # 1/2/4 孤儿：步长不变
python3 /tmp/z100diag/long_orphan.py 32768  # 32k 孤儿：步长 763 ms
```
