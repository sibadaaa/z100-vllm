# 方案 B 第一批执行结果：B1 被实测否决、C1（INT8 vs FP8 冷 prefill）2–3×

> 归档时间：2026-10-12（CST 24:00 前后）
> 执行者：Lead（真机 A/B；**未改任何 vllm 源码**）
> 前置：[`z100-19 长思考崩溃定位`](z100-19-longthinking-collapse-result.md) · [`z100-20 缩短静默方案`](../design/z100-20-shorten-silence-options.md)
> 状态：B1 ⛔ 实测无收益（不建议实施）；C1 ✅ 有大收益（2–3×，待用户决策）；B3 🔶 未构建；A4 ⛔ 该栈上 torch profiler 会卡死引擎

## 1. 结论速览

| 项 | 结论 | 关键数字 |
|---|---|---|
| **B1** conv fallback 去 `.item()` | ⛔ **否决**：conv 路径总成本 ≈1%，去掉同步后收益在噪声内 | 见 §2 |
| **C1** INT8 `sdot4` vs FP8 仿真 | ✅ **冷 prefill 2–3×**（8k/32k），96k 只 1.23×（注意力主导） | 见 §3 |
| **A4** torch profiler 取证 | ⛔ 默认配置（`with_stack=True`）**把引擎卡死 10 min+**、trace 60 MB 未写完 | 见 §4 |
| **B3** spinloop 扩展 | 🔶 未做：健康态通知正常（空闲后首 chunk 0.57 s），停靠主要出现在退化态，优先级低于 C1/占用问题 | 见 §5 |

## 2. B1：微基准 + 端到端 A/B（否决依据）

真实形状（Qwen3.8-27B GDN：`linear_conv_kernel_dim=4`、key 16×128、value 48×128 ⇒ conv 通道 **10240**；chunk **2048**），健康机上单层计时：

```
F.conv1d(groups=dim)      :    0.394 ms
F.silu(out)               :    0.163 ms
cat(prev,x)               :    0.183 ms
weight.unsqueeze().contiguous(): 0.004 ms
torch.zeros((dim,3))      :    0.022 ms
int(idx[0].item())        :    0.021 ms      <- 同步只要 21 µs（不是先前估的 ~2 ms）
bool(hstate[0].item())    :    0.020 ms
idx.cpu().tolist()        :    0.034 ms
_causal_conv1d_gfx906_fallback（整层, batch=1）: 1.014 ms
```

⇒ 64 层 ≈ **65 ms / 个 2048-token chunk**；而该 chunk 的端到端 GPU 时间是 **~8 s** ⇒ conv 路径占 **≈1%**。

端到端验证（直接*完全绕开* fallback，用异步 Triton conv 路径，即 B1 的收益上限）：

| prompt | fallback（`VLLM_GFX906_TRITON_CAUSAL_CONV1D=0`） | Triton（`=1`） |
|---|---|---|
| 8192 | 32.79 s / 33.57 s | **33.27 s** |
| 32768 | 157.55 / 162.09 / 162.53 s | **161.43 s** |

差异在跑间噪声内 ⇒ **B1 不值得改**（与 S11f 记录 "prefill 394 vs 424 t/s"、即 fallback 慢 ~7% 同量级；那 7% 是整条 FLA prefill kernel 的差异，不是 conv 同步）。

复现：`/mnt/game/port/diag/conv_bench.py`；端到端用 `ARM=tritonconv TRITON_CONV=1 bash /tmp/z100diag/serve_arm.sh`。

## 3. C1：INT8 `sdot4` vs FP8 仿真（**本轮最大发现**）

同容器、同仓、同 flags（TP=4 / `--enforce-eager` / MTP k=2 / MBT=2048 / maxlen 262144 / 前缀缓存开），只换检查点与内核族：

| prompt | FP8 仿真（当前线上） | **INT8 `sdot4`** | 提升 |
|---|---|---|---|
| 8192 | 32.79–33.57 s（≈250 t/s） | **11.02 s（743 t/s）** | **3.0×** |
| 32768 | 157.55–162.53 s（≈202 t/s） | **74.99 s（437 t/s）** | **2.2×** |
| 98304 | 599.93 s（164 t/s） | **486.32 s（202 t/s）** | 1.23× |
| decode ITL @8k | 317–320 ms | **221 ms** | 1.4× |
| decode ITL @32k | 703–729 ms | **603 ms** | 1.2× |

引擎日志佐证内核确实换了：
- FP8 臂：`Selected Gfx906Fp8BlockScaledMMLinearKernel for Fp8LinearMethod`
- INT8 臂：`Selected Gfx906Sdot4Int8ScaledMMLinearKernel for CompressedTensorsW8A8Int8`

**判读**：gfx906 没有 FP8 硬件，FP8 走的是仿真内核；INT8 走 `sdot4`（硬件 int8 点积）。
8k–32k 段线性层是瓶颈 ⇒ 2–3×；96k 段 **prefill 注意力（O(n²)）成为瓶颈** ⇒ 只剩 1.23×。

**对"静默"的直接意义**（客户端 watchdog 300 s）：
- 32k 冷 prefill：**160 s → 75 s**，稳进阈值内；
- 按 32k 的 437 t/s 外推，**≈60k token 以内冷 prefill 可控制在 ~150 s**；
- 96k 冷 prefill 仍 486 s > 300 s ⇒ 超长上下文仍会触发超时（需配合 DSH 侧上下文上限，用户已明确不动）。

## 4. A4：torch profiler 在这套栈上不可用（阻塞性发现）

用 `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/mnt/game/prof"}'` 起实例：
- `/start_profile` 正常；一次 32-token 请求后 `/stop_profile`；
- 随后**四个 worker 全部卡在 `torch.profiler.stop()`**（`profiler.py:882 → _transit_action → stop_trace`），
  `shm_broadcast.py:812 "No available shared memory broadcast block found in 60 seconds"`，
  `/health` 与 `/metrics` 全部无响应；trace 写到 60 MB 仍未收尾（`ROCTracer produced duplicate flow start`）。
- 处理：`pkill -9` 后重启恢复；已删除半成品 trace。

⇒ 结论：**在这套 DTK/torch 上，torch profiler（带 stack）会卡死引擎**，不适合在线取证。
后续可选用轻量配置（`torch_profiler_with_stack=False`）或 rocprof，但要单独排期。

## 5. B3 现状（未做，附判断）

- 扩展在 `csrc/spinloop.cpp`，CMake 目标 `vllm.spinloop`（`CMakeLists.txt:151-164`、`setup.py:1122`），开关 `VLLM_USE_SPINLOOP_EXT`（默认 0）。
- 需要一次 CMake 构建并把 `.so` 落进 `vllm/` 包目录（容器内 `/src` 只读，须从宿主侧写入仓库树）。
- 收益判断：健康态通知链正常（空闲数分钟后首 chunk 仍 0.57 s，说明 notify 没丢）；`SHM_READER_RECHECK_INTERVAL_MS=5000` 的停靠主要在**退化态（引擎被占住）**暴露，而那是占用问题不是读出问题。⇒ 排在 C1 与占用问题之后。

## 6. 本轮口径与遗留

- 全部为真机实测；全部实验跑在同一容器 `z100-vllm`（生产容器 `qwen38-vllm` 全程未动）。
- C1 是"检查点+内核族"联合 A/B：差异既可能来自量化格式，也可能来自内核实现（gfx906 无 FP8 硬件这一事实使 INT8 占优符合预期）。
- **未做质量门禁**：换 INT8 需要按 S15/S18 口径做 token 对拍与质量评测（用户侧此前生产用的就是 INT8 w8a8，工具有现成的）。
- 服务已在实验后恢复为用户原配置（FP8 + MTP k=2 + eager + MBT=2048）。

## 7. 复现命令

```bash
# B1 微基准
docker exec -e HIP_VISIBLE_DEVICES=0 ... z100-s18 bash -lc '. /opt/dtk/env.sh; export PYTHONPATH=/src; python /mnt/game/port/diag/conv_bench.py'
# B1 上限 A/B（Triton conv 路径）
ARM=tritonconv MTP=2 TRITON_CONV=1 bash /tmp/z100diag/serve_arm.sh && python3 /tmp/z100diag/bench.py tritonconv 32 8192,32768
# C1（INT8 sdot4）
ARM=int8 MTP=2 INT8=1 bash /tmp/z100diag/serve_arm.sh && python3 /tmp/z100diag/bench.py int8 32 8192,32768,98304
# 恢复原配置
ARM=baseline MTP=2 FP8_EMU=1 GEMM_MAX_M=1 EAGER=1 MBT=2048 bash /tmp/z100diag/serve_arm.sh
```
