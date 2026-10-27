# Z100 五相位 round profile（WP1.2，G5）

> 由 `tools/dcu/s27_five_phase_profile.py --phase parse` 生成；相位定义与归因方法见该工具 docstring 与 §附注。

## 1. 一句话结论

| 指标 | 值 |
|---|---:|
| complete round | 116.40 ms/step |
| 五段之和 | 116.40 ms/step |
| srv_step_ms | 114.17 ms（WP1.1 graph arm, .s27/p1/s20_graphA-7469-real_20261024-144727.json, same arm/env/doc, adjacent unprofiled window）|
| 差 | 1.949 % （判据 <5% ⇒ PASS）|
| gap/未知 | 0.663 ms |

**G5 一句话判读**：五段之和 116.397 ms vs 独立 `srv_step_ms` 114.172 ms，差 1.949%（判据 <5% ⇒ PASS）；其中 trace 自洽 round 周期 116.394 ms 与非 profile srv_step_ms 的差 1.946% 归因为 profiler 开销（上界），不是相位归因缺口。

## 2. 五相位分配表

| 相位 | ms/step | 占比 | GPU busy ms | 调用/步 | top-5 算子 |
|---|---:|---:|---:|---:|---|
| `draft` | 33.829 | 29.1% | 30.714 | 154.4 | `triton_matmul_kernel` 13.336·30.72; `kernel_unified_attention_2d` 6.098·4.94; `Cijk_Alik_Bljk_BBH_MT64x64x16_SE_APM1_AF0EM1_AF1EM1_AMAS3_ASAE01_ASCE01_ASEM1_BL` 5.773·1.94; `elementwise_kernel_manual_unroll` 3.587·10.67; `ncclDevKernel_AllReduce_Ring_Simple_Sum_f16` 1.196·10.89 |
| `draft->target` | 0.000 | 0.0% | 0.000 | 0.0 | （无 kernel，纯 gap/sync） |
| `target` | 77.810 | 66.8% | 76.934 | 1242.0 | `fp8_emu_gemm_block_w8a16_kernel` 38.073·192.00; `triton_matmul_kernel` 10.406·48.00; `ncclDevKernel_Generic_4` 8.283·128.00; `kernel_unified_attention_3d` 8.219·16.00; `fp8_emu_gemv_stream_block_kernel` 4.846·64.00 |
| `target->draft` | 4.758 | 4.1% | 5.397 | 23.0 | `triton_matmul_kernel` 4.922·2.00; `ncclDevKernel_AllGather_Ring_LL_Sum_i8` 0.191·1.00; `_prepare_dflash_inputs_kernel` 0.073·1.00; `_penalties_kernel` 0.049·1.00; `elementwise_kernel_manual_unroll` 0.038·2.00 |
| **complete round** | 116.397 | 100% | | | |

### 逐 rank（ms/step）

| rank | draft | draft->target | target | target->draft | complete round | rounds kept | split methods |
|---|---:|---:|---:|---:|---:|---:|---|
| `s27five3_rank0` | 33.829 | 0.000 | 77.810 | 4.718 | 116.357 | 18/19 | target_annotation×18 |
| `s27five3_rank1` | 33.828 | 0.000 | 77.806 | 4.751 | 116.385 | 18/19 | target_annotation×18 |
| `s27five3_rank2` | 33.829 | 0.000 | 77.810 | 4.766 | 116.404 | 18/19 | target_annotation×18 |
| `s27five3_rank3` | 33.825 | 0.000 | 77.811 | 4.799 | 116.436 | 18/19 | target_annotation×18 |

## 3. env 全文 + 策略指纹 + 树 SHA

```json
{
 "request": {
  "chunks": 39,
  "completion_tokens": 39,
  "error": null,
  "prompt_tokens": 0,
  "ttft_ms": 8844.3,
  "wall_s": 332.328
 },
 "tag": "graphA-fivephase",
 "traces": [
  {
   "mtime": 1792854504.0092866,
   "path": "/mnt/game/s27-prof/s27five3_rank0.1792854499912000168.pt.trace.json.gz",
   "sha256_16": "405a960e354bef01",
   "size": 3599200
  },
  {
   "mtime": 1792854504.041287,
   "path": "/mnt/game/s27-prof/s27five3_rank1.1792854499930735473.pt.trace.json.gz",
   "sha256_16": "b87c9a22d8f2aff9",
   "size": 3584491
  },
  {
   "mtime": 1792854504.161288,
   "path": "/mnt/game/s27-prof/s27five3_rank2.1792854499973611684.pt.trace.json.gz",
   "sha256_16": "dc90ff7b5cb3fde8",
   "size": 3602752
  },
  {
   "mtime": 1792854504.1172876,
   "path": "/mnt/game/s27-prof/s27five3_rank3.1792854499929710094.pt.trace.json.gz",
   "sha256_16": "5d81f19eb83c659b",
   "size": 3568564
  }
 ],
 "window": {
  "accept_len": null,
  "generation_tokens": 0.0,
  "itl_ms": null,
  "profiled_steps": 20,
  "step_ms": null,
  "steps": 0.0
 }
}
```

- 树 SHA：`7078b5fe137feeab3750ee243daeaf172dc439d8`（dirty=True）
- 策略指纹：`7e70fc2c130b9191`

采集臂 env 快照（`graphA-mbt8192-7469.env.txt`，逐字）：
```text
AMDGPU_TARGETS=gfx906;gfx926;gfx928;gfx936;gfx938
C_INCLUDE_PATH=/mnt/game/host-runtime/dtk2604/dcc/gcvm/include:/mnt/game/host-runtime/dtk2604/include:/opt/hyhal/include:/mnt/game/host-runtime/dtk2604/llvm/include:/mnt/game/host-runtime/dtk2604/opencl/include
CMAKE_PREFIX_PATH=/opt/hyhal:
CPLUS_INCLUDE_PATH=/mnt/game/host-runtime/dtk2604/dcc/gcvm/include:/mnt/game/host-runtime/dtk2604/include:/opt/hyhal/include:/mnt/game/host-runtime/dtk2604/llvm/include:/mnt/game/host-runtime/dtk2604/dushmem/include:/mnt/game/host-runtime/dtk2604/opencl/include:/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11
DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
DSH_HOME=/home/lyh/.dsh
DSH_PROFILE_DIR=/home/lyh/.dsh/profiles/web
DSH_PROFILE=web
DSH_SESSION_ID=session-16c9e466-f374-4e59-9f05-6ca0f72c4d0e
DSH_SHELL=1
DSH_WEB_URL=http://127.0.0.1:3090
DTK_ROOT=/mnt/game/host-runtime/dtk2604
DTKROOT=/mnt/game/host-runtime/dtk2604
GIT_PAGER=cat
HIP_PATH=/mnt/game/host-runtime/dtk2604/hip
HIP_VISIBLE_DEVICES=0,1,2,3
HOME=/home/lyh
HSA_FORCE_FINE_GRAIN_PCIE=1
HYHAL_PATH=/opt/hyhal
INVOCATION_ID=62ff92450bbe44a8b99ad30d64feda6d
JOURNAL_STREAM=8:30465
LANG=en_US.UTF-8
LD_LIBRARY_PATH=/mnt/game/host-runtime/dtk2604/.hyhal/rocm_smi/lib:/mnt/game/host-runtime/dtk2604/dcc/gcvm/lib:/mnt/game/host-runtime/dtk2604/hip/lib:/mnt/game/host-runtime/dtk2604/llvm/lib:/mnt/game/host-runtime/dtk2604/lib:/mnt/game/host-runtime/dtk2604/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/mnt/game/host-runtime/dtk2604/dushmem/lib:/mnt/game/host-runtime/dtk2604/opencl/lib:
LIBRARY_PATH=/opt/hyhal/lib:/mnt/game/host-runtime/dtk2604/opencl/lib:/mnt/game/host-runtime/dtk2604/dushmem/lib
LOGNAME=lyh
MANAGERPID=1038
MAX_SEQS=4
MBT=8192
MIOPEN_FIND_MODE=3
NO_COLOR=1
OLDPWD=/home/lyh/z100-vllm
PAGER=cat
PATH=/mnt/game/host-runtime/dtk2604/bin:/mnt/game/host-runtime/dtk2604/llvm/bin:/mnt/game/host-runtime/dtk2604/hip/bin:/mnt/game/host-runtime/dtk2604/hip/bin/hipify:/opt/hyhal/bin:/mnt/game/host-runtime/dtk2604/opencl/bin:/home/lyh/.nvm/versions/node/v24.21.0/bin:/usr/local/bin:/usr/bin:/bin
PORT=8000
PWD=/home/lyh/z100-vllm
PYTHONPATH=/home/lyh/z100-vllm:/mnt/game/triton-gfx906/install
PYTHONUNBUFFERED=1
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
QT_ACCESSIBILITY=1
ROCM_PATH=/mnt/game/host-runtime/dtk2604
SHELL=/bin/bash
SHLVL=1
SYSTEMD_EXEC_PID=1116
TERM=dumb
TOKENIZERS_PARALLELISM=false
TRITON_CACHE_DIR=/mnt/game/host-serve/triton-cache
TRITON_HIP_LLD_PATH=/mnt/game/host-runtime/dtk2604/llvm/bin/ld.lld
USER=lyh
VLLM_CACHE_ROOT=/mnt/game/vllm-cache-host
VLLM_DCU_BF16=1
VLLM_DCU_FP16_PREFILL=auto
VLLM_DCU_FP8_DECODE_CFG=35
VLLM_DCU_FP8_EMU_FASTPATH=1
VLLM_DCU_FP8_GEMV_MAX_M=8
VLLM_DCU_FP8_QUANT_FASTPATH=1
VLLM_DCU_FP8_W8A16=1
VLLM_DCU_GCN_ARCH=gfx906
VLLM_DCU_USE_BF16_GEMV=1
VLLM_DCU_USE_FP8_EMU=1
VLLM_GFX906_ATTN_FORCE_CAUSAL=0
VLLM_GFX906_ATTN_PREFILL_3D_MIN_CTX=16384
VLLM_GFX906_ATTN_PREFILL_SEGMENTS=4
VLLM_GFX906_ATTN_VERIFY_3D=1
VLLM_GFX906_GDN_CHUNKED_PREFILL=1
VLLM_GFX906_TRITON_ATTN_PREFILL_3D=1
VLLM_LOGGING_LEVEL=INFO
VLLM_QWEN35_MTP_GDN_TRITON=1
VLLM_ROCM_USE_SKINNY_GEMM=0
VLLM_S24_DYNK_EMA=0.1
VLLM_S24_DYNK_HI=65536
VLLM_S24_DYNK_LO=16384
VLLM_TORCH_PROFILER_DIR=/mnt/game/s27-prof
VLLM_WORKER_MULTIPROC_METHOD=spawn
XDG_DATA_DIRS=/usr/local/share/:/usr/share/:/var/lib/snapd/desktop
XDG_RUNTIME_DIR=/run/user/1000
```
- trace 文件（sha256[:16]）：
  - `s27five3_rank0.1792854499912000168.pt.trace.json.gz` (3.4 MiB, 405a960e354bef01)
  - `s27five3_rank1.1792854499930735473.pt.trace.json.gz` (3.4 MiB, b87c9a22d8f2aff9)
  - `s27five3_rank2.1792854499973611684.pt.trace.json.gz` (3.4 MiB, dc90ff7b5cb3fde8)
  - `s27five3_rank3.1792854499929710094.pt.trace.json.gz` (3.4 MiB, 5d81f19eb83c659b)

## 4. 判读

- 五段之和 116.397 ms vs srv_step_ms 114.172 ms：差 1.949%，**PASS**（<5%）。
- gap/未知：0.663 ms/step（host 侧 step 入口到首个 kernel；应 <2% step）。
- 参与 rank：['s27five3_rank0', 's27five3_rank1', 's27five3_rank2', 's27five3_rank3']；跳过：[]。
- 独立 `srv_step_ms` 出处：WP1.1 graph arm, .s27/p1/s20_graphA-7469-real_20261024-144727.json, same arm/env/doc, adjacent unprofiled window
- 采集口径：profiling 期间 `/metrics` 的 step 已含 profiler 开销，比较基准必须是同臂/同 env/同负载的 `srv_step_ms`。
- 若独立基线与 profiling 窗口**不是同一 wall-clock 窗口**（本例：`/stop_profile` 落盘时服务 segfault，无法取同窗口 metrics），则五段和与基线的差拆成「profiling 开销上界 + 相位归因缺口」：profiling 开销上界 = 1.946%（trace 自洽 round vs 非 profile 基线），相位归因缺口 ≈ 1.949% - 1.946%。

## 5. 反证条件

- 若 `phase_sum` 与 `srv_step_ms` 差 >5% ⇒ 先修采集（稳态、窗口、rank、trace 是否含 prefill），**不出结论**（BUILD WP1.2 停止线）。
- 若 `draft` 段的 nccl/fp8_emu/quant 调用数接近 target ⇒ 归因错误，检查 `--cluster-gap-us` 与目标签名，并用 annotation cross-check 行与单步时间轴人工复核。
- 若某 rank 的 round 周期与其它 rank 差 >5% ⇒ 该 rank trace 抓到 prefill/异常步，需重抓。
- 若 `aten::mm` 出现在 draft/target 之外（gap/未知）⇒ 该步有未归因GEMM，检查是否 prefill 混入或窗口切错。

## 6. 回滚方式

本报告为只读证据；回滚 profile 采集只需停服务。图臂回滚：删除 `tools/dcu/s27_graph_launch.sh`，回到 `tools/dcu/serve_host_dflash2.sh`。

## 7. 采集风险与事故记录（2026-10-24，C5 口径）

本节记录本次 G5 采集过程中在该栈上真实发生的事故与处置，**不是性能结论**；
按 C5（诊断隔离），profiling 窗口数据只用于相位归因，不与非 profiling 的
生产测量混算，两者仅通过显式标注的独立基线对照。

### 7.1 profiler dump 触发宿主 OOM（15:00:42）

- 现象：一次 `torch_profiler_with_stack=true` 的 20 步采集在 `/stop_profile`
  dump 期间触发全局 OOM；内核日志 `dmesg`：
  `Out of memory: Killed process 1018366 (VLLM::Worker_TP) anon-rss 6.8GB`，
  随后 `Worker proc VllmWorker-3 died unexpectedly (exit code: -9)`，engine
  core 死亡，后续请求报 `AssertionError: collective_rpc should not be called
  on follower node`。
- 量化：`with_stack=true` 时每 rank trace **53–58 MB**（4 rank 合计 >200 MB），
  dump 时每 worker 需在内存里构造完整 JSON，4 worker 并发 → 宿主
  **31 GiB RAM + 19 GiB swap** 被击穿。
- 处置（已生效）：swap 增加到 **~51 GiB**（+32 GiB `/mnt/game/swap-32g`），
  且 profiler config 改为 `torch_profiler_with_stack=false`，
  每 rank trace 降到 **3.4–3.6 MB（约 1/15）**；本报告所用 trace 即该配置产物，
  采集全程无 OOM。
- 结论：**任何后续 profile 采集都必须带 `with_stack=false`**；需要 stack 的
  诊断另开窗口，不与生产/性能臂混跑。

### 7.2 `/stop_profile` 在本栈 segfault（已记录风险）

- 现象：trace 落盘后 API server 在 stop 阶段崩溃（日志尾
  `hsa_stop undefined symbol` → Segfault），随后 4 个 worker 被
  SIGTERM/SIGKILL。服务因此停止，**同窗口 `/metrics` 未能采到**
  （`collect JSON` 的 `window.step_ms/accept_len` 为 null 即为该原因）。
- 影响与处置：G5 的「五段和 vs srv_step_ms」改用**同臂、同 env、同 7469
  文档、紧邻的非 profiling 窗口**作为独立基线（WP1.1 graph 臂探测
  `.s27/p1/s20_graphA-7469-real_20261024-144727.json`，`srv_step_ms =
  114.172 ms`），并把差值拆成 profiling 开销上界 + 相位归因缺口（见 §4）。
  该做法不冒充"同一 wall-clock 窗口"。
- 反证条件：若有人能证明 profiling 自身开销 >5%（而非本报告测得的 ≈2%），
  则 G5 判**不可判定**，需换采集方式（例如只在 worker 侧用 rocprofv3
  或 `max_iterations` 单窗口 auto-stop）重抓。

### 7.3 采集/解析工具侧修复（供复现）

- `/stop_profile` 必须 **POST**（GET 返回 405）；trace dump 期间 `/metrics`
  会被阻塞，client 需重试并等待 dump 完成（本报告 wait 上限 300 s）。
- `VLLM_TORCH_PROFILER_DIR` 在本树**无消费者**，profiler 只能用 CLI
  `--profiler-config` 注册端点（否则 `/start_profile` 404）。
- 鉴权：`/v1/models` 等路由需要 Bearer key；`urllib` 必须用 `Request`
  对象才能被 `s23_common.install_auth` 加头（裸 URL 会 401）。
- `with_stack=false` 的 trace 无 `python_function`/`args.stack` 事件；
  解析器不依赖这些字段（只读 `cat/name/ph/ts/dur` + `process_labels`）。

## 附注：归因方法与可复现性

- 步骤窗口：vLLM `execute_context_*_generation_*(N)` 注释，只取 `N>0` 的 decode 步；`ProfilerStep#N` 为 fallback。
- draft/target 切分（`--split-mode auto`，两种 trace 结构通用）：**piecewise CUDA-graph 臂**（本报告）：CPU `execute_context` 注解覆盖整个 round，而 GPU 侧同名注解只覆盖 target verify（~78/116 ms）；auto 据 GPU 注解/round 周期 <0.85 自动选用 target-annotation 切分 —— target = GPU 注解内 kernels，draft = round 内其前全部 kernels（草稿塔+selector），target->draft = verify/sampling 尾部。**eager 臂**（GPU 注解覆盖整个 step）：按 target 签名 kernel（fp8_emu / per_token_group_quant / attention / GDN / allreduce）的最大时间簇切 target，无签名时用最大 inter-kernel gap + 签名计数。同一张表只保留一种切分方法（`--consistent-split`，默认开），profile 窗口边缘不完整 round 剔除并计入 `rounds_trimmed`。
- `draft->target` / `target->draft` 是相邻 burst 的 gap（含 KV 写、同步、小 kernel）；`gap/未知` 只留 step 入口到首个 kernel 的 host 时间。
- 解析器只读 `cat/name/ph/ts/dur` 与 `process_labels` metadata，不依赖 `args.stack`/`python_function`（`with_stack=false` 的 trace 完全兼容；`with_stack=true` 只多出可忽略的 python_function 事件）。
- 逐 rank 结果见 JSON `--out`；本表取跨 rank 中位数。

