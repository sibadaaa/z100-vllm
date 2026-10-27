# S23 — Z100 fp16-模拟-fp8 优化结果（O1–O7）

> 执行依据：[`docs/design/z100-fp8-emu-optimization-plan.md`](../design/z100-fp8-emu-optimization-plan.md)
> 调研依据：[`docs/result/z100-fp8-emu-fixed-time-investigation.md`](z100-fp8-emu-fixed-time-investigation.md)
> 平台：4×海光 DCU Z100（gfx906；DCU0/1 = Z100L@1600 MHz，DCU2/3 = Z100@1319 MHz），TP=4，
> `--enforce-eager`，DFlash2 k=7，模型 Qwen3.8-27B-FP8
> 分支：`/home/lyh/z100-vllm`，基线 commit `ab2a89b1a`，基线 `.so` md5 `f205be8f43d3ffb1cfb3c6f093bfab4b`
> 执行协议与探针口径：[`.s23/S23_EXEC_PROTOCOL.md`](../../.s23/S23_EXEC_PROTOCOL.md)

## 0. 一句话结论

本轮把计划里的 7 个工作项全部执行到「有数据、可回退、口径可复算」的程度：

* **内核侧兑现了**：O3 用「K 分块加宽 + 收窄 N tile」（`VLLM_DCU_FP8_DECODE_CFG=35`）把生产 5 个
  decode 形状的 fp8 GEMM 全部改善 −15~−24%，每步 fp8 阶段 52.5 → 42.7 ms（**−18.7%**），
  且与改动前**逐 bit 相同**；O4 把 `5120:1536` 从 streaming 改判回 tile，再省 1.84 ms/步。
* **Host 侧实现了但 c=1 未兑现**：O1 去掉 `custom_op` 包装税（−57.9 µs/次，每步 −13.9 ms CPU），
  O2 去掉量化 host 包装（−12.8 µs/次，每步 −3.3 ms CPU），但 c=1 实测 period 无变化
  （control 138.00 ms vs o1 139.24 ms p50），因为 **c=1 的步进是 GPU-bound，host 时间被 GPU 完全吸收**。
  这直接推翻了计划 §2 的前提（「c=1 由 idle/派发主导」），是本次最重要的负面结论，也解释了
  S20 的「内核 −33 ms/步 → e2e −1~7%」和 S23 的 `corr(period, idle)=+0.739` 为何不能外推到 host 侧。
* **程序总目标（c=1 19.2 → ≥21.1 t/s）本轮未达成**，原因不是实现失败，而是目标建立在错误的
  瓶颈归因上；按实测，c=1 的可控上限来自 GPU 侧 fp8 阶段（38% → 31% of period）。
  真实可交付的收益是 c≥4（host 成为关键路径）与 fp8 阶段本身，见 §7。

## 1. 基线（全部实测，非引用）

| 指标 | 值 | 窗口 / 口径 |
|---|---|---|
| c=1 step p50（control 臂，本轮代码，开关全关） | **138.00 ms** | `.s23/w1/O6/w1a2/control` |
| 输出 t/s / accept / tok·step⁻¹ / TTFT | 26.95 / 7.00 / 7.53 / 2506 ms | 同上 |
| fp8 阶段（256 次/步，M=8，DCU3 实测 ×1/1.21 → DCU0 等效） | 52.5 ms/步（stream 生效）/ 51.6（全 tile） | `.s23/w2/O3/sweep/` |
| decode tile 拦截项（N=768:K=5120,M=8） | 211.0 µs（DCU3）= 174 µs（DCU0 等效） | 同上 |
| K 分块斜率 | **t = 5.2 µs + 1.28 µs/chunk**（r²≈1，DCU3） | `.s23/w2/O3/sweep/kscale_dev3.log` |
| 计划文档里的历史基线 193.8 ms / 19.2 t/s | **不同窗口，不可比**（本次同日实测 138 ms / 26.95 t/s） | 见 §2 口径纪律 |

> 计划 §1 的 193.8 ms 与本次同日 138.00 ms 相差 25–30%。原因未查（不同 capture / 不同宿主机负荷），
> 因此本文件所有收益一律用**同窗口 A/B 比值**表达，绝不跨窗口做差。

## 2. 口径与可复算性

1. 同窗口背靠背；每臂 4 次 probe；报 p50 + 均值 + 输出 t/s + accept。
2. host 计时：4 轮交错（A/B/A/B），取中位数。
3. canary：`8704:5120` M=1 dense 参考 0.10625 ms（DCU0）；>110% 判该窗口作废。DCU3 的
   参考值按 1.21× 折算为 ≈0.128 ms，实测 0.098–0.116 ms。
4. 数值门禁：内核改动用 `torch.equal`（rel_err = 0）；量化用 `torch.equal` on `(x_q, x_s)`。
5. 机器码门禁：`.s23/probes/isa_gate.py`（遍历 `.hip_fatbin` 里**每一个** clang offload bundle；
   注意 ROCm 每个 TU 一个 bundle，`clang-offload-bundler --unbundle` 只给第一个，因此朴素检查会
   误报「没有 fp8 kernel」）。
6. 所有 GPU 探针经 `docker exec z100-regress` + `flock .s23/gpu.lock` 串行；生产池常驻 ~32.6 GiB，
   单次分配上限 64 MiB。

## 3. 交付的开关（全部默认关闭）

| 开关 | 作用 | 归属 |
|---|---|---|
| `VLLM_DCU_FP8_EMU_FASTPATH=1` | eager 下绕过 `torch.ops.vllm.fp8_emu_mm*`，直调 `_rocm_C` 裸 op | O1 |
| `VLLM_DCU_FP8_QUANT_FASTPATH=1` | 激活量化跳过 `QuantFP8.forward_hip/forward_cuda` 与逐次分支 | O2 |
| `VLLM_DCU_FP8_DECODE_CFG=35` | 选定 O3 的 BK=64/BN=32 tile kernel（`16` = 改动前） | O3 |
| `VLLM_DCU_FP8_STREAM_MAX_M=8` | M≥8 时不用 streaming（`unset` = 改动前行为） | O4 |
| `VLLM_DCU_FP8_FUSE_QUANT=1` | 量化并入 emu GEMM（O5，见 §6） | O5 |

## 4. O1 — 去掉 `torch.library` custom_op 包装税

**实现**（`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py`）：custom_op 的 body 抽成
`_fp8_emu_mm_impl` / `_fp8_emu_mm_block_impl` 纯函数，custom_op 与 eager fast path 共用；fast path 条件
`_EMU_FASTPATH and not torch.compiler.is_compiling()`；`_EMU_FASTPATH` 在 import 时读一次。
参数与 custom_op 完全一致（`nthreads=_GEMV_THREADS`、`_prefill_cfg(M, N)`、per-tensor 分支 cfg=0）。

| 口 | 数字 | 日志 |
|---|---|---|
| host，调用点（真实权重、K+256 stride，4 轮交错中位） | 90.98 → 33.42 µs（8704:5120 M=8）；−52.6…−62.0 µs 逐形状 | `.s23/w1/O1/host2.log` |
| host，派发层（同一组预处理实参） | 81.56 → 23.66 µs；裸 op 同窗口 20.84 µs ⇒ fast path = 裸 op 地板 + ~3 µs | `.s23/w1/O1/prep.log` |
| kernel census | kernel 名/次数/平均时长与基线一致（1.0134 / 0.9927），无新增 kernel | `.s23/w1/O1/census3.log` |
| 编译路径 | `torch.compile` 下 op census 仍是 `vllm::fp8_emu_mm_block`，输出与 eager 逐 bit 相同 | `.s23/w1/O1/compile2.log` |
| CUDA graph | capture+replay 逐 bit 相同 | `.s23/w1/O1/graph.log` |
| 单测 | `test_gfx906_fp8_emu.py` **123 passed**（关/开/QUANT 单独/全开 四种状态） | `.s23/w1/O6/w1c/unit-gate/` |
| e2e c=1 | control 138.00 ms vs o1 139.24 ms（warm +1.8%，均值 +0.2%），parity 8/8 逐 token 一致 | `.s23/w1/O6/w1a2/` |

**判定**：host 门禁通过（−57.9 µs/次，每步 −13.9 ms CPU，优于计划预估的 −8…−14）；编译/图捕获/单测/parity
全绿；端到端 c=1 **未兑现**（见 §7）。

## 5. O2 — 量化阶段 host 包装下沉

**前置问题（计划要求先回答）**：生产的 256 次/步激活量化**没有被 `fuse_act_quant` 融合**。证据三条：
`tools/dcu/serve_host_dflash2.sh` 用 `--enforce-eager`（编译期 pass 不运行）；调用链
`BlockScaledMMLinearKernel.apply_weights → QuantFP8.__call__ → forward_hip → forward_cuda →
fp8_utils.per_token_group_quant_fp8`，每层一次、无 memoisation；步骤 census 仍是 256 次
`_C::per_token_group_fp8_quant`（若已融合应为 0）。因此本项前提成立。

**实现**：`fp8_utils.py` 新增 `per_token_group_quant_fp8_fast_ok(...)`（唯一的判据来源）与模块级
缓存的 fp8 dtype/min/max，在 `per_token_group_quant_fp8` 入口早退；`BlockScaledMMLinearKernel.__init__`
一次性算出 `_quant_fastpath`（含 `type(self.quant_fp8) is QuantFP8` 与 `_quant_fastpath_src` 身份守卫，
Cutlass/torch 子类自动走回原路径），`apply_weights` 直调 `per_token_group_quant_fp8`。**每次调用新建
`x_q/x_s`**，不缓存、不跨层复用。

| 口 | 数字 | 日志 |
|---|---|---|
| 逐层分解（M=8，K=5120，4 轮交错中位） | 调用点 42.4 µs → `forward_hip` 38.7 → `fp8_utils` 35.9 → 裸 op+alloc 30.8 | `.s23/w1/O2/layers.log` |
| 生产线调用点 `apply_weights` | 112.94 → 100.15 µs（**−12.79 µs/次 = −3.27 ms/步 CPU**） | `.s23/w1/O2/session2.log` |
| 数值 parity | **30/30**（5 形状 × 3 seed × M∈{8,48}，`torch.equal` on `(x_q,x_s)` 与完整输出） | 同上 |
| 单测 | 4 个文件 × 4 种开关状态 = **382 passed**（每次） | `.s23/w1/O6/wP/unit-gate/` |
| c=1 e2e | 预期持平（host 隐藏），见 §7 | — |

**地板与取舍（写清为什么不能再省）**：单次地板 ≈ 8.1 µs（裸 kernel，预分配）+ ~9 µs（两次
`new_empty`）+ ~2 µs（帧/条件）≈ 20 µs；生产负荷窗口里实测 27–30 µs，剩余差距是 CPU 争用不是代码。
一次大 buffer + 3 个 view 的「单次分配」变体实测 49.0 µs（分配本身 23–25 µs），**已放弃**并记录在案。
不缓存 out-buffer（计划明令禁止，缺 stream 级证据）。

## 6. O3 — decode tile 拦截项（本轮唯一兑现的设备侧收益）

**先测机制再改代码**（这是本轮最有价值的方法论结论）：

* K 扫描（N=768，M=8，cfg16）：K=640/1280/2560/3840/5120 → 30.86/56.43/106.98/158.88/210.32 µs，
  拟合 **t = 5.2 µs + 1.28 µs/chunk（r²≈1）**。⇒ 每 chunk 成本真实存在，且不是 launch 开销。
* H1（每 chunk 暴露 DRAM 延迟）**被证伪**：BK=32 下加深寄存器流水（RST=3/4/6/8）得到
  196.4/196.5/199.0/198.7 µs 的平地，且大形状变差；四者的 `scratch = 0`、vgpr 58/61/62/68，
  **没有 local memory**，所以「fragment 溢写」这一假设（执行者事前提出的）也被排除。
* H2（occupancy）用资源回答：cfg16 是 `vgpr 166 × 256 线程` ⇒ **1 block/CU**；把 occupancy 提到
  3 blocks/CU 反而让大 N 慢 3%，说明大形状是 issue/带宽受限。
* 真正起作用的是**K 分块宽度**：BK=32 时每行每 chunk 只读 32 B，每次都要付一整笔 DRAM 延迟；
  BK=64（`wide2` tile，BN 收窄到 32 以保持 LDS 与 block 数）= 每轮开销 0.43 µs 摊到 2 倍元素上。

| 形状（M=8，DCU3） | cfg16（改动前） | **cfg35** | Δ |
|---|---|---|---|
| 768:5120（拦截项） | 211.0 µs | 162.1 | −23.2% |
| 8704:5120 | 292.9 | 248.2 | −15.3% |
| 5120:4352 | 202.6 | 165.5 | −19.0% |
| 4096:5120 | 238.9 | 195.5 | −18.9% |
| 3584:5120 | 215.4 | 172.6 | −20.7% |
| 5120:1536 | 77.1 | 63.6 | −17.9% |

每步（64/64/48/16/64 配比，M=8）：fp8 阶段 51.6（全 tile）/52.5（现状，含 stream）→
**42.7 ms/步（−18.7%）**；占 c=1 period 38% → 31%。

**端到端（window P，同窗口 A/B，`.so 7f87e185`，tree fp 6498e118）**

| 臂 | c=1 warm p50 | 输出 t/s（warm） | accept | tok/step | TTFT | parity |
|---|---|---|---|---|---|---|
| o3_control（`.so f205be8f`, cfg16） | 148.58 ms | 27.91 | 7.00 | 7.53 | 2509 ms | 参考 |
| **cand35_qt_off（cfg35）** | **137.68 ms（−7.34%）** | **29.06（+4.13%）** | 7.00 | 7.53 | 2498 ms | **8/8 一致** |

c=4/c=16 扫（4 probe/臂）：control 899.8 / 1050.2，cand35 907.7 / 1029.7 → **无回归**（该扫为
prefill 主导，fwd/step≈840 @c=16，对 O3 只是回归检查）。

**判定**：兑现。实现 −18.7% 内核收益，端到端 −7.34%（预测上限 −7.1%，实现在上限内），
数值逐 bit 不变，大形状无回归，机器码门禁通过。**未达成**计划写死的「拦截项 ≤120 µs」
（实测 159.8 µs@DCU3 = 132 µs@DCU0 等效）与「mix ≤30 ms/步」：剩余成本是 0.0265 µs/元素
的逐字节工作量在 12–24 block 的小 grid 上，而余下机制要么被禁（split-K），要么需要另一种
kernel 结构（W 常驻寄存器、A 从 LDS 广播，约 1/TM+1/TN 次 LDS 读/MAC 而非 1.5），已按 time-box 停手。

## 7. O4 — streaming 路径（实现了，**未发布：数值门禁未过**）

零代码 A/B（`5120:1536`，M=8）：stream `fp8_emu_gemv_stream_block_kernel<8,3,Half>` 92.1 µs
（86 GB/s）vs cfg16 tile 77.1 vs cfg35 tile 63.6（124 GB/s）⇒ 改选择规则可再省 1.84 ms/步（mix −22.2%）。

**实现**：`stream_max_m()`（`VLLM_DCU_FP8_STREAM_MAX_M`，未设/≤0 = 今天的 dispatch 不变；V = M≥V 跳过
stream），插在 `stream_block_select` 的 `want/M` 守卫之后。

**端到端**：`cand35_qt_on`（cfg35 + gate=8）c=1 warm p50 134.91 ms（−9.20%）、t/s 29.42（+5.39%），
但**greedy parity 8 条里 5 条逐 token 分歧**（重试仍分歧）⇒ 按计划 §1① / §6.2 **不发布**。

> 归因待定：kernel 级探针显示 `5120:1536` 上 cfg16 tile 与 stream 逐 bit 相同、cfg35 也与 cfg16
> 逐 bit 相同，因此「换路由就改求和序」这一解释与探针数据冲突；window Q 用
> `cand16_qt_on`（cfg16+gate）判别「路由切换」与「cfg35 kernel」谁是真因，并复核 wP 的分歧
> prompt 在 `cand35_qt_off` 下是否也分歧。**在判别结果出来前，本项状态是「已实现、已测量、未发布」。**

## 8. O1/O2 的端到端横评（四窗口，全部同窗口 A/B）

host 侧两个改动在 CPU 上是确定的（O1 −13.9 ms/步、O2 −3.3 ms/步），但**能否转成 step 时间**必须按并发档位分开说：

| 并发 | 臂对比 | 指标 | 结果 | 判定 |
|---|---|---|---|---|
| c=1 | control vs O1（w1a2，control 先） | warm p50 | 137.83 → 140.35 ms（+1.83%） | 持平（反向） |
| c=1 | O1 vs control（w1c，O1 先） | warm p50 | 139.05 → 137.81 | 持平（反向） |
| **c=4** | control vs O1（w1c） | fwd t/s 中位 | 845.6 → 901.1（**+6.6%**） | **正向信号**（control 有一个 1008.5 的 19% 离群 rep；剔除后两侧紧致区间 843–848 vs 900–911 不重叠） |
| **c=16** | control vs O1（w1c） | fwd t/s 中位 | 1001.7 → 1037.3（**+3.55%**） | **成立**（warm 区间 999–1009 vs 1035–1055 不重叠） |
| c=1 | cfg35 vs cfg35+O2+O1（Q2） | p50 / warm p50 | 136.77/140.08 → 136.12/135.97 | 持平 |
| c=16 | cfg35 vs cfg35+O2 / +O1+O2（Q、Q2） | fwd t/s 中位 | 1037.2 vs 1051.4（+1.4%） | **不可判定**（区间重叠） |

**口径说明**：c≥4 用 `fwd t/s` 而不是 `step_ms`，因为 step_ms 的 warm 区间在当前负荷下边际重叠，
而 fwd t/s 的非重叠更干净；两者方向一致（c=16 step_ms 1421.0 → 1373.4，−3.35%）。
**c=4/c=16 的 sweep 是 prefill 主导的**（c=16 时 fwd/step≈840），所以它对 O1 是「host 在 prefill 段
也生效」的证据；对 O3/O4 只是回归检查（见 §6）。

**结论（按 vllm-py 建议的措辞，已核对原始 JSON）**：host delta 真实且可复现；其转成 step 时间
在 c=1 **未出现**（两个窗口、两种臂序互相抵消），在 c=4/c=16 **对 O1 出现**（+6.6% / +3.55%），
但 **O1+O2 组合在 cfg35 构建上 c=16 不可判定**，组合的 c=4 未测（缺口，非失败）。

## 9. O5 — 量化与 GEMM 融合（kernel 门禁绿；e2e 回归，**不发布**，见 §14）

**做法**：新增两个 C++ op（`fp8_emu_gemm_block_quant` 生产用，只返回 `out`；`..._quant_x` 只给
逐 bit 门禁用，返回 `(out, x_q, x_s)`），tile 路径单次启动 `fp8_emu_gemm_block_q_kernel`，
per-128-group 的 `y_s` 留在 LDS（1.28 KiB）不落地；stream / M>8 的情况在 C++ 内部回退为
「两次新分配 buffer 的 quant + 原 kernel」，语义与未融合路径一致。Python 侧按**层配置**分支
（不以 M 过滤），开关 `VLLM_DCU_FP8_FUSE_QUANT`（默认 False），未融合路径逐字节不变。

**逐 bit 契约**（实现严格照抄 `per_token_group_quant.cu:189-300`）：absmax 以 `eps` 起、
`y_s = amax / 448.0f`（真除法）、`inv_y = 1.0f / y_s`（第二次真除法）、`scaled = v * inv_y`、
`clamp(±448)`、`c10::Float8_e4m3fn(scaled)`。

| 门禁 | 结果 | 日志 |
|---|---|---|
| 逐 bit（6 形状，M=8，route-of-record） | **`x_q=eq x_s=eq out=eq` 6/6** | `.s23/w2/O5/gate_final.log` |
| 启动次数/次调用 | **2 → 1** | 同上（`CENSUS launches/call`） |
| NaN/Inf 输入断言 | nan=0 inf=0，全部形状 | 同上 |
| 机器码 | v_dot2 62136，无 v_mfma/v_cmac/v_dot4 | 同上 + `.s23/probes/isa_gate.py` |

**过程中修掉的真实缺陷（记录在案，因为它们是这道门禁的价值）**：① fused kernel 里 `blk` 未清零，
而 `fp8_gemm_compute_wide` 是累加语义 ⇒ 首个 group 吃到寄存器垃圾（表现为 768:5120 的 `out=inf` 且
不可复现）；② `x_q` witness 返回扁平 `(M*K,)` 而非 `[M,K]`；③ `build_fp8.sh` 漏了 host bindings
object，导致新 op 不注册。三处都已修并复验。

## 10. O6 — 验收工具、parity、回滚

**工具**（全部落在 `tools/dcu/`，可复跑）：`s23_ab.sh`（窗口驱动，整窗持锁、臂描述符、换 .so + 重启 +
ready/健康检查）、`s23_{common,probe,sweep,parity,smoke,preflight,descriptor,report}.py`、
`s23_kernel_census.py`、`s23_logscan.py`、`s23_launch_piecewise.sh`。

**发现的三个「假信号」陷阱（都已做成硬门禁，值得后人复用）**：
1. **臂序漂移**：w1a2 显示 O1 +1.8%，w1c 反向臂序显示 −0.3% ⇒ 差异是臂序/窗口漂移。之后所有
   「大于 2%」的结论都要求 **warm 区间不重叠**，不接受中位数单点。
2. **窗口污染**：wQ 的推荐臂 c=16 采到单调下降序列（991.6→904.1→923.9），而参考臂平稳 ⇒ 判
   「不可解释、拒绝判定」；wQ2 反序重测（1051.4 平稳）证明那是污染而非配置效应。**污染窗口的
   数字一律不进结论**。
3. **假 parity 断言的两种错法**：(a) 把 `arm 不自洽`（非确定性）当成数值回归 —— 现在分开记录
   `retry_matches_arm_first_run` 与 `reproduced_after_retry`；(b) 请求体不同却比较输出 —— 现在每个
   case 记 `body_sha` 并要求两侧相等，否则标「不是数值主张」。

**「gate 是否真的生效」的证伪器**：每个臂都断言内核名集合。例：`STREAM_MAX_M=8` 生效时 M=8 必须
出现 `wide2<64,8,32,1,2,Half>` 且**不得**出现 `fp8_emu_gemv_stream_block_kernel<8,3,Half>`；
`cand16_qt_on` 的 census（`wide2=False stream=False`，20×cfg16 tile）证明开关确实落地，
因此它的 parity 失败不是「无效果」假阴性。

## 11. 推荐生产配置与回滚

推荐（**只用同窗口实测过的组合**；O4 的 gate 故意不设）：

```bash
# .so md5 7f87e185c628bd3c54c53ff9d5e0d78a（本轮全部 e2e 数字所属的构建）
# 最终交付构建 d209af9585aaf0049394ced4a3c3211b（= 同一 cfg35 kernel + O5 ops + 变体裁剪，
#   ISA/numeric 门禁已复跑；e2e 实测见 §14：该构建 + 本配置 c=1 step p50 137.56 ms，
#   parity 8/8，与 7f87e185 上的 137.68 ms 一致）
export VLLM_DCU_FP8_DECODE_CFG=35      # O3: BK=64/BN=32 tile（改动前 16）
export VLLM_DCU_FP8_EMU_FASTPATH=1     # O1
export VLLM_DCU_FP8_QUANT_FASTPATH=1   # O2
export VLLM_DCU_FP8_FUSE_QUANT=0       # O5 默认关（e2e 回归，见 §14）
# VLLM_DCU_FP8_STREAM_MAX_M 不设（O4 因数值门禁不发布）
```

实测收益（全部同窗口 A/B）：

| 项 | 并发 | 指标 | 结果 | 状态 |
|---|---|---|---|---|
| O3 | c=1 | warm step p50 / 输出 t/s | 148.58 → 137.68 ms（**−7.34%**）/ 27.91 → 29.06（**+4.13%**） | 兑现，parity 8/8 |
| O3 | c=4/c=16 | fwd t/s | 无回归 | 成立 |
| O1 | c=4 / c=16 | fwd t/s 中位 | **+6.6%** / **+3.55%** | 正向（c=16 区间不重叠） |
| O1 | c=1 | warm p50 | 两次反序窗口互相抵消 | 持平 |
| O2 | c=1 / c=16 | p50 / fwd t/s | 持平 / 组合不可判定 | 见 §8 |
| O4 | c=1 | warm p50 | −9.20%（含 O3） | **不发布（parity）** |
| O5 | — | 启动数/步 | 256 → 128（次调用 2→1；**e2e 回归 −54%，不发布**） | 见 §14 |

**回滚**（一句命令 + 判据）：

```bash
cp /mnt/game/perf1/_rocm_C.f205be8f43d3ffb1cfb3c6f093bfab4b.so \
   /mnt/game/port/build-sync/temp/_rocm_C.abi3.so   # 然后 restart，判据：
md5sum /mnt/game/port/build-sync/temp/_rocm_C.abi3.so   # 必须读回 f205be8f43d3ffb1cfb3c6f093bfab4b
# + 固定 prompt smoke 与基线逐字一致
```

## 12. 明确未达成 / 未覆盖

| 项 | 状态 | 原因 |
|---|---|---|
| 程序总目标 c=1 19.2 → ≥21.1 t/s（+10%） | **未达成** | 目标建立在「c=1 由 host/idle 主导」上；实测 GPU-bound。O3 的 −7.34% 是真实上限内的兑现 |
| O3 拦截项 ≤120 µs / mix ≤30 ms 步 | 未达成 | 实测 159.8 µs@DCU3（132 µs@DCU0 等效）/ 40.9 ms；剩余是 0.0265 µs/元素 的逐字节成本在 12–24 block 小 grid 上 |
| O4 发布 | 否 | parity：stream 路由在 `5120:1536@M=8` 上与两种 tile kernel 数值不同（判别臂证明） |
| O7 | 不可行 | 见 §13；阻塞点是**编译期**的既有 Dynamo graph break，不是 cudagraph 配置 |
| O2 的 c=4、O1+O2 组合的 c=4 | 未测 | sweep 在 c≥4 是 prefill 主导，对 host 项只剩「prefill 也生效」的含义（已由 O1 证明） |
| `FUSE_QUANT=1` 的 e2e | **已测，回归 −54%，不发布** | 见 §14：融合 kernel 的几何固定为 BK=128/BN=64，单调用比 O3 的 BK=64/BN=32 慢 1.5–3.9× |

## 13. O7 — CUDA graph（piecewise）复核：**不可行，但阻塞点已改变**

* 配置侧**不再是**阻塞点：`cudagraph_mode` 确实解析为 `PIECEWISE`，`max_cudagraph_capture_size=512`，
  capture sizes 1..512，`fuse_attn_quant=False`（没有发生「静默升级为 FULL」）。
* 真正失败点在 **worker 初始化 / Dynamo fullgraph capture**：
  1. `RuntimeError: Worker failed with error 'Data-dependent branching`（`r_piecewise/startup.log:1826`）；
  2. 调用栈 `torch/_dynamo/convert_frame.py:1251 fullgraph_capture` → `:1318 _fullgraph_capture_frame`
     （同文件 L1502-1504）；
  3. 该 graph break **与本次改动无关**：同一份 `torch.compile` 探针在 O1 开关关闭时同样报
     `Data-dependent branching`（`.s23/w1/O1/compile2.log` 的 `compile_control`）。
* 结论：O7 从「cudagraph 在此配方下不可启动」细化为「**编译路径在此模型上还有一处 data-dependent
  branch**」；一旦该 break 被消掉，piecewise 才有复核价值。本项按 time-box 关闭。

## 14. 最终窗口（wFinal）与 O5 的最终判定

**窗口内容**（`tools/dcu/s23_ab.sh --unit-gate --repeat 4 --ref-arm final_ctrl wFinal final_ctrl final_fused`）：
同一份交付构建 `d209af9585aaf0049394ced4a3c3211b`（= cfg35 wide2 + O5 ops + 变体裁剪；ISA/numeric
门禁已在其上复跑），tree fp `5d73e54d38da8973`，git `ab2a89b1a`。

| 臂 | switches (EMU/QNT/FUSE) | step p50 | out t/s | accept | parity |
|---|---|---|---|---|---|
| `final_ctrl`（推荐配置） | 1/1/0 | **137.56 ms** | 27.29 | 7.00 | 参考 |
| `final_fused`（+O5） | 1/1/1 | **211.84 ms（+54.0%）** | 18.27 | 7.00 | **8/8 一致** |

单测门禁（引擎停机、4 种开关状态）：**382 passed × 4**，含 `FUSE_QUANT=1`
（`.s23/w1/O6/wFinal/unit-gate/`）。

**O5 判定：kernel 级成立，引擎级显著回归，不发布。**

* 逐 bit 与次数：`x_q/x_s/out` 6/6 形状 `torch.equal`；启动数 2 → 1
  （`.s23/w2/O5/gate_final.log`）。Python 层 `apply_weights` 实测：M=8 113.98 → 37.67 µs
  （**−76.3 µs/次**，每步 −19.5 ms CPU），输出逐 bit 相同；M=48 219.13 → 80.09 µs。
* **回归原因（已量化，不是猜测）**：融合 kernel 必须让「一个 K chunk = 一个 128 元素量化组」，
  因此几何固定为 **BK=128 / BN=64**，而 O3 的赢家是 **BK=64 / BN=32**。同一 .so、同一 cfg、同 M=8
  的单调用实测（`.s23/probes/o5_slow_probe.py`）：
  `8704:5120` unfused 256.2 µs vs **fused 990.6 µs（3.87×）**；`3584:5120` 217.4 vs 335.5 µs（1.54×）。
  逐形状放大后即 +74 ms/步，与窗口实测一致。
* 结论：**`VLLM_DCU_FP8_FUSE_QUANT` 保持默认关**；要兑现 O5，需要把量化融合到 O3 的
  BK=64/BN=32 几何上（一个 128 组跨两个 chunk，amax 需跨 chunk 累加），那是另一个迭代，
  本轮按 time-box 停手。这是计划 §6.2 明确定义的「已实现但未兑现」，且**不是**测量噪声。

**回滚演练（正式，同窗口内完成）**：`d209af95 → f205be8f`，实测
`so_swap_ms=829`、`restart_to_ready_ms=218665`（引擎重载模型为主）、`ready_to_smoke_ms=2736`、
`total_rollback_s=222`；判据：md5 读回 `f205be8f43d3ffb1cfb3c6f093bfab4b` + 固定 prompt
` Paris.\nThe capital of Germany is Berlin.\nThe` 逐字一致 + `/v1/models` 200。
（`.s23/w1/O6/wFinal/restore/`、`window-summary.md`）

**最终生产状态**：默认配置（`.so f205be8f`、无 `VLLM_DCU_*` 覆盖、unit `Environment=HOST=0.0.0.0`），
按计划 §6.3「不动生产默认 env」保持原样；启用推荐配置见 §11。

## 15. 变更集合（供人工评审）

| 文件 | 增/删 | 内容 |
|---|---|---|
| `vllm/envs.py` | +27 | O1/O2/O5 三个默认关闭的开关 + 注释 |
| `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py` | +~250 | O1 impl 抽取与 fast path；O5 两个 op 的 custom_op/fake + impl + `apply_block_scaled_mm_quantized` + `fuses_input_quant` |
| `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | +~90 | O2 早退快路 + 判据拆分为 `shape_ok`（switch-free）与 `fast_ok` |
| `vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py` | +~75 | O2 一次性判据 + 直调；O5 `fuses_input_quant` 分支与默认 `NotImplementedError` |
| `csrc/rocm/fp8_emu_gemm.cu` | +829 | O3 `wide2` tile（cfg 32..35）；O4 `VLLM_DCU_FP8_STREAM_MAX_M`；O5 `fp8_emu_gemm_block_q_kernel` + 两个 host 入口 |
| `csrc/rocm/fp8_emu_gemm.h` | +31 | 上述 host 入口声明 |
| `csrc/rocm/torch_bindings.cpp` | +20 | O5 两个 op 的 schema + Meta impl |
| `tools/dcu/s23_*`、`.s23/w*/` | 新增 | 窗口驱动/parity/分段 census/logscan/探针与全部原始日志 |

> 注意：`vllm/v1/attention/ops/triton_unified_attention.py` 的本地改动**不是本轮产物**（执行前即存在），
> 评审时应单独处理。
>
> lint：`ruff check` 在本轮改动的 4 个 python 文件上仍只有基线的 6 个既有告警
> （5×ISC004 + 1×I001，另 `envs.py` 3 个既有 F601/B033），**无新增**；新增行均在 88 列内；
> 单测 382 passed × 4 种开关状态；`py_compile` 全绿。

## 16. 生产启用记录（用户授权，2026-10-19）

用户确认后，推荐配置已写入 `tools/dcu/serve_host_dflash2.sh` 并重启生产（用户 unit，14:39 UTC）：

* `.so` 由 `f205be8f` 换为 **`d209af9585aaf0049394ced4a3c3211b`**（旧件仍在
  `/mnt/game/perf1/_rocm_C.f205be8f43d3ffb1cfb3c6f093bfab4b.so`，一条 `cp` 可回滚）；
* 脚本新增/修改（全部保留 `${VAR:-默认}` 形式，A/B 臂可覆盖）：
  `VLLM_DCU_FP8_DECODE_CFG` 默认 16 → **35**、`VLLM_DCU_FP8_EMU_FASTPATH` 默认 **1**、
  `VLLM_DCU_FP8_QUANT_FASTPATH` 默认 **1**；`VLLM_DCU_FP8_STREAM_MAX_M` 与
  `VLLM_DCU_FP8_FUSE_QUANT` **保持不设**（O4 因 parity、O5 因 e2e 回归，均不启用）。
* 验证：`/v1/models` 200（带 api-key）、固定 prompt smoke 与基线逐字一致
  （`巴黎.` 语义文本 `' Paris.\nThe capital of Germany is Berlin.\nThe'`）、
  c=1 4 次 probe（重启后新窗口）step p50 **146.6 ms**、输出 25.6–27.9 t/s、accept 7.00、
  tok/step 7.53；该绝对值和窗口内的 137.56 ms 不直接可比（跨窗口纪律见 §2），
  回滚判据与命令见 §11。
