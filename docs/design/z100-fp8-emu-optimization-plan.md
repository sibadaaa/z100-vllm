# Z100 fp16-模拟-fp8 优化方案（S23，供执行 agent 使用）

> 依据：[`docs/result/z100-fp8-emu-fixed-time-investigation.md`](../result/z100-fp8-emu-fixed-time-investigation.md)（S23 调研）
> 平台：4×海光 DCU Z100（gfx906，2×Z100L@1600 + 2×Z100@1319），TP=4，`--enforce-eager`，DFlash2 k=7
> 生产：systemd `z100-vllm-host.service`（PID 2622752），live `.so` md5 `f205be8f43d3ffb1cfb3c6f093bfab4b`
> 分支工作区：`/home/lyh/z100-vllm`；文档纪律见 `AGENTS.md`（Python 一律走 `uv`/`.venv`，行长 88，Google docstring）

本方案把调研出的**每个固定成本项**变成一条可执行、可验收、可回滚的工作项。执行 agent 必须按
「§2 总表 → §3 单项规格 → §5 测量协议 → §6 验收」做，不许跳过验收门禁。

---

## 1. 基线（不可争议的起点，全部实测）

| 指标 | 基线值 | 来源 |
|---|---|---|
| c=1 步进 period | **193.8 ms**（窗口 A）/ 217.3 ms（窗口 B）；输出 **19.2 / 20.0 t/s**；accept 7.00 | `.s20/S20_KERNEL_RESULT.md` §3 |
| fp8 GEMM 阶段（256 次/步，M=8） | **42.8–44.9 ms/步**（cfg16 tile 192 次 + streaming 64 次） | S23 §4.1（Lead census + T2） |
| ├ 其中与 M 无关的固定部分 | **30.9–31.1 ms/步（69–72%）** | S23 §4.1 |
| └ 与 M 成正比（dot2） | 13.7 ms/步 | 同上 |
| decode tile 拦截项（每次调用） | **t = 171.6 µs + W/779 GB/s**（r²=0.943；T2 独立：169.6 µs + W/643） | S23 §3.3 |
| streaming（`5120:1536`）每次调用 | 75.7–85 µs @ **116 GB/s**（M=1 时 488 GB/s） | S23 §3.3 |
| activation 量化 | GPU **8.4 µs/次**（M 无关）+ host **43–51 µs/次** | S23 §3.1 |
| emu custom_op host | **43–83 µs/次**（裸 `_rocm_C` op 仅 12–23 µs） | S23 §3.2 |
| GPU 可见 launch 气泡 | 12.86 ms/步（legacy）/ **4.29 ms/步**（当前 cfg16+stream） | S23 §4.1（trace） |
| 每步调用次数 | **512 kernel = 256 quant + 256 emu**，c=1/4/16 与 5 档并发扫描**完全不变** | S23 §0.1（trace） |
| 步进节奏归因 | `corr(period, idle)=+0.739`、`corr(period, gpu_ms)=+0.071`（c=1） | S23 §4.3 |

**程序总目标**：c=1 输出 t/s **19.2 → ≥ 21.1（+10%）**，且
① greedy 8 条 prompt 逐 token 与基线**完全一致**；
② prefill TTFT（14K/32K/36K）与 c=4/c=16 吞吐**无回归**；
③ 每项均可单开关回退。

---

## 2. 优化项总表（按 ROI × 风险排序）

| ID | 工作项 | 性质 | 预期收益（每步） | 风险 | 依赖 | 负责人角色 | 任务板 |
|---|---|---|---|---|---|---|---|
| **O1** | 去掉 `torch.library` custom_op 包装税 | host-only | **−8~14 ms CPU**；e2e +2~6% | 低 | — | vllm-py | `task-5` |
| **O2** | 量化阶段 host 包装下沉 | host-only | **−8.5 ms CPU**；e2e +2~5% | 低-中（别名/融合交互） | — | vllm-py | `task-6` |
| **O3** | 攻 decode tile 的 171.6 µs 拦截项 | kernel（.cu） | **−14 ms GPU**（mix 44.9→30）；c=16 e2e −10~15% | 中（需重测数值） | — | kernel-dev | `task-7` |
| **O4** | streaming 路径带宽修复（116 GB/s） | kernel（.cu） | −2.3 ms（该形状 75.7→≤45 µs） | 中 | O3（同文件） | kernel-dev | `task-9` |
| **O5** | 量化与 GEMM 融合（每步少 256 次 launch） | 结构（.cu+bindings+py） | −2.6~4 ms + 256 launch | 中-高 | O3,O4,O1 | kernel-dev + vllm-py | `task-10` |
| **O6** | 端到端验收、parity、回滚演练（横切） | 工具/流程 | 保护性（无直接收益） | 低 | 每波都要用 | perf-verifier | `task-8` |
| **O7** | CUDA graph（piecewise）可行性复核（time-box） | 配置探索 | 若成立 −20~31 ms（消全部派发税） | 高（S20 记录不可启动） | — | perf-verifier | `task-11` |

**收益上限说明（必须写进每份结果文档）**：c=1 步进由 idle/派发主导（`corr(period,idle)=+0.739`），
kernel 侧收益只有 30–70% 能落到端到端（S20 已实测：内核 −33 ms/步 → e2e −1~7%）。
因此 **O1/O2 这类 host 侧削减在 c=1 上的转化率高于 O3**；O3 的主战场是 c≥4（c=16 时 fp8 占 period 51%）。

---

## 3. 单项规格

### O1 — 去掉 `torch.library` custom_op 包装税（P0，先做）

**现状与根因**
`Gfx906Fp8BlockScaledMMLinearKernel.apply_block_scaled_mm` 调 `torch.ops.vllm.fp8_emu_mm_block`
（`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py:426-436`）。该 custom_op 的存在理由只有一个：
compile 模式下原始 `TORCH_LIBRARY` op 缺少 shape 契约会让 Dynamo 给 token 维加 `Eq(dim, hint)` guard 并
`ConstraintViolationError`（S11k，见 `gfx906_fp8.py:169-210` 的 docstring）。
生产是 `--enforce-eager`，**这条理由不成立**，但每次调用仍付 31–55 µs 的 schema/dispatcher/Python 帧开销
（实测：custom_op 43–83 µs vs 裸 op 12–23 µs）。

**改动点**
1. `vllm/envs.py`：新增 `VLLM_DCU_FP8_EMU_FASTPATH`（bool，**默认 False**，与既有 `VLLM_DCU_*` 同段落风格）。
2. `gfx906_fp8.py`：
   - 把 custom_op body 里的分支抽成一个**纯函数**（如 `_fp8_emu_mm_block_impl(...)` / `_fp8_emu_mm_impl(...)`），
     custom_op 与 fast path **共用同一实现**，避免两份分支逻辑漂移；
   - 在三个 kernel 类（per-tensor/per-channel `apply_scaled_mm`、block `apply_block_scaled_mm`）里加：
     `if envs.VLLM_DCU_FP8_EMU_FASTPATH and not torch.compiler.is_compiling(): return _impl(...)`
     否则走原 custom_op。
3. 不要动 `apply_weights` 的上层调用约定与返回 shape。

**必须验证的三个坑**
- **compile 路径不能退化**：`torch.compiler.is_compiling()` 必须真的为真（用一个 `torch.compile` 的
  小 linear 单测断言走的是 custom_op 分支，不是裸 op）；否则等于把 S11k 的坑再打开一遍。
- **CUDAGraph capture 兼容**：capture 期间同样不要引入 python 分支依赖 M（分支只在 eager 上按运行时 M 判断，
  capture 时 M 固定），capture 一个 2 层小模型不报错。
- **参数完全等价**：fast path 必须传与 custom_op 完全相同的 `nthreads=_GEMV_THREADS` 与
  `_prefill_cfg(a.shape[0], w.shape[0])`；用 census 断言 GPU kernel 名与数量与基线一致。

**验收门禁（全过才算完成）**
1. `pytest tests/kernels/quantization/test_gfx906_fp8_emu.py -q` 全绿（在 `docker exec z100-regress` 里跑）。
2. host 直测：`torch.ops.vllm.fp8_emu_mm_block` vs fast path，同窗口 4 轮交错，fast path **≤ 30 µs/call**
   （目标 12–16 µs），判定用中位数（见 §5 第 6 条）。
3. kernel census：fast path 下 kernel 名/次数/平均时长与基线逐项一致（±2%）。
4. e2e 同期 A/B（`O6` 的工具）：`FASTPATH=0` vs `=1` 各 4 次 probe，**step p50 改善 ≥ 2%**，输出 t/s ≥ +2%，
   c=4 与 c=16 无回归（≤1%）。
5. parity：greedy 8 prompt 逐 token 一致。

**预期**：host 11.0–19.7 → 3.1–5.9 ms/步；e2e −2~6%（受 idle 吸收影响）。

---

### O2 — 量化阶段 host 包装下沉（P0）

**现状与根因**
block-scaled 线性层的激活量化走 `QuantFP8.forward_hip → forward_cuda →
fp8_utils.per_token_group_quant_fp8(x, 128)`（`vllm/model_executor/layers/quantization/input_quant_fp8.py:136-162`
→ `vllm/model_executor/layers/quantization/utils/fp8_utils.py:672-719`），host **43–51 µs/次**，
而裸 `torch.ops._C.per_token_group_fp8_quant` 只要 8 µs。差值来自 `get_fp8_min_max()`、两次 `torch.empty`、
参数分支与 python 帧。

**改动点**
1. `vllm/envs.py`：`VLLM_DCU_FP8_QUANT_FASTPATH`（bool，默认 False）。
2. 在 quant 的 ROCm 分支加快速路径：条件 `fastpath and x.is_contiguous() and group_size == 128 and not is_compiling()`
   → 直接发 `torch.ops._C.per_token_group_fp8_quant(x, x_q, x_s, group_size, eps, fp8_min, fp8_max, False, False, False)`
   （`fp8_min/max` 由 dtype 决定，模块级缓存一次即可）。
3. **先测再决定 buffer 策略**：第一版必须**逐调用新建** `x_q/x_s`（严禁跨层复用同一 buffer，
   否则 CUDA graph/piecewise 捕获与并发请求会踩别名）。只有确认「新建 buffer 版」仍 ≥ 20 µs 时，
   才评估按 `(M, K, dtype)` 缓存 out-buffer，并必须证明：同一 buffer 不会被两个 kernel 同时在飞（需 stream 级证据）。

**必须先回答的一个前置问题**：生产的这一层 quant 是否已被 `fuse_act_quant` pass（`compilation_config.pass_config.fuse_act_quant=True`）
融合到上游 op？若已融合，则本项收益为 0，**先量再改**：用 census/host 计时确认每步 256 次 quant 是独立 python 调用
（trace 侧已知是 256 次 `_C::per_token_group_fp8_quant`，但确认一下 python 侧调用点）。

**验收门禁**
1. 数值 parity：同输入下 `(x_q, x_s)` 与基线**逐 bit 相同**（`torch.equal`）。
2. host：快速路径 **≤ 20 µs/call**（4 轮交错中位）。
3. e2e：step p50 ≥ +2%，c=4/c=16 无回归；**1000 步长稳**（无 NaN、无 scales 漂移：抽 3 个 step 校验 scale 与基线一致）。
4. `pytest tests/kernels/quantization/test_gfx906_fp8_emu.py` + 相关 fp8 单测全绿。

**预期**：−33 µs/次 → **−8.5 ms/步 CPU**；e2e −2~5%。

---

### O3 — 攻 decode tile 的 171.6 µs 拦截项（P1，收益最大但需内核工作）

**实测事实（必须当成前提，不要重测推翻）**
- `t = 171.6 µs + W/779 GB/s`：N=768（12 个 block，60 CU 大半空转）单次就要 **179 µs**，N=8704 也只要 233 µs。
- tile 路径 M=1..8 完全平坦（T2 强制 tile 测 8704:5120：239.7/239.8/240.6/245.8 µs，±2.5%）⇒ 不是 MAC、不是 A 加宽。
  （注意：M=1..4 生产默认走 streaming 只有 105–234 µs，不要拿默认 dispatch 的 M 曲线当 tile 曲线。）
- 结构：`csrc/rocm/fp8_emu_gemm.cu:205-330`，`BK=32`、2 级 LDS 双缓冲、每 chunk 两次 barrier；
  K=5120 ⇒ **160 个 chunk**；cfg16 = `BM=8,BN=64,TM=1,TN=2` ⇒ MG=8/NG=32 ⇒ **NT=256 线程（8 warp/block）**；实测只有 **9.07 warps/CU**（≈1 个 block/CU 在飞）。
  即 171.6 µs / 160 chunk ≈ **1.07 µs/chunk**，是**每 chunk 串行延迟**，不是带宽。

**假设与验证顺序（每条都要给 kernel 级数据，不能只给端到端）**
- **H1 每 chunk 的 barrier + 单层预取不隐藏 global 延迟**：把 `BK` 提到 64（或 3 级 LDS pipeline）后，
  若 N=768 的拦截项近似对折 ⇒ 确认。代价：LDS 占用翻倍，注意 occupancy。
- **H2 片上并行不足（每 CU 只 1 个 block 在飞）**：cfg16 = 8 warp/block 但实测 **9.07 warps/CU**，即 1 个 block/CU，没有第二个 block 去隐藏第一个的 global 延迟。方向：降 smem/寄存器把 2–3 个 block 挤进同一 CU，或 `BN=128`（NT=512）让单 block 自带更多 warp；同时观察 VGPR 溢出（`--save-temps` 看 `.vgpr_count`）与 LDS 上限。
- **H3 LDS bank conflict**：`kFKp=17` 的 pad 对 fp16 dword 是否够（BK=32 时 A/W 行各 16 dword + 1 pad）；
  用 `rocprofv3 --pmc` 或 kernel 内 split 计时量化 conflict 数。
- **H4 wave quantization**：N=8704 ⇒ 136 block ÷ 60 CU = 2.27 波，最后 0.27 波浪费；
  试 N-major persistent block（1 block 连做多个 N-tile，权重在 L2 复用），把 136 block 压到 ≤ 60。
- **H5（已排除，不要重复）**：A staging 不是主因（M 平坦）；streaming 路径不适用本项。

**禁区（违反即打回）**
- **不做 split-K**：会引入非确定求和序，违反本仓库「可复现优先」纪律；确需做必须另立设计文档并说明确定性方案。
- 不改 epilogue 的 `(acc*a_s)*w_s` 顺序、不改 fp16 LUT 语义、不改 fp8→fp16 的无损加宽。
- 不为了达标去改 `VLLM_DCU_FP8_GEMV_MAX_M` 这类影响 dispatch 的默认值（只允许在 A/B 臂里显式设）。

**验收门禁**
1. kernel 级（`tools/dcu/perf1_kernel_mprobe.py` 或自写 probe，`--iters 40`，min-of-3×20，canary 见 §5）：
   - N=768/K=5120/M=8 拦截点 **179 → ≤ 120 µs**（首选 ≤100）；
   - 生产 mix **44.9 → ≤ 30 ms/步**（用 64/64/48/16/64 配比 × 逐形状时间）；
   - 大形状不回归（`8704:5120` ≥ 基线 95%）。
2. 数值：`pytest tests/kernels/quantization/test_gfx906_fp8_emu.py` 全绿（cfgs 11..19 参数化）；
   mprobe `rel_err = 0`（S20 判据）；greedy parity 逐 token 一致。
3. 机器码门禁（沿用 S11f/S20 口径）：`.hip_fatbin` 拆包 + `llvm-objdump`，`v_dot2_f32_f16` 必须存在、
   `v_mfma*` 必须为 0、`v_cmac*`/标量模拟必须为 0。
4. e2e：c=4 与 c=16 **≥ +8%**；c=1 ≥ +2%（可能被 idle 吸收，如实报告）。

**构建方式（实测可用）**
```bash
docker exec z100-regress bash -c 'cd /src && ninja -C /mnt/game/port/build-sync/temp _rocm_C'
```
（6 步、只重编 4 个 HIP object + 链接。产物 `_rocm_C.abi3.so` 就是 live 服务加载的那个路径，
**换装前必须按 §6.3 备份 + 记 md5**。）

---

### O4 — streaming 路径带宽修复（P1，依赖 O3 的文件）

**实测事实**：`5120:1536`（K=1536）在 M=8 时走 `fp8_emu_gemv_stream_block_kernel<8,3,Half>`，
**116 GB/s**（M=1 时 488 GB/s），单次 75.7–85 µs，每步 5.45 ms。原因是 A 加宽后 LDS 读放大 16:1
（每块重读 A）已解决，但 M=8 时 W 的流式读取未被充分利用。

**步骤（先零代码、后改代码）**
1. **零代码 A/B**：`VLLM_DCU_FP8_GEMV_STREAM=0` 让该形状回到 cfg16 tile，测 3 组时间。
   若 tile 更快（很可能，因为 tile 在 M=8 上与 M 无关），**最优解可能只是改选择规则**（一行）。
2. 若改规则：`stream_block_select`（`fp8_emu_gemm.cu:777-811`）加一条 M 相关门槛（如 `M>=8` 时不用 stream），
   开关 `VLLM_DCU_FP8_GEMV_STREAM` 已有语义要保留（0=关、1/2=变体），新增的是「自动」判据。
3. 若要求保留 stream 路线：修 A-LDS 读放大（A 仅 8×BK，可寄存器常驻 + 单次广播到 LDS），目标 ≥300 GB/s。

**验收**：该形状 **≤45 µs/call**；mix −2 ms/步以上；数值与 parity 门禁同 O3。

---

### O5 — 量化与 GEMM 融合（P2，最后做）

**动机**：每步 512 次 kernel launch 是固定项；把 activation 量化并进 emu kernel 可去 256 次 launch
（−2.15 ms GPU + 256×launch）。A 只有 8×K（M=8），在内核里量化成本极低。

**做法**：新增 `fp8_emu_quant_mm_block`（`csrc/rocm/fp8_emu_gemm.cu` + `csrc/rocm/torch_bindings.cpp` +
`gfx906_fp8.py` 注册 custom_op），或在现有 op 里加「未量化输入」分支。block-scaled 语义要求
每 128 个 K 一组 scale，内核需两遍 A（一遍求 amax→scale，一遍量化），A 只有 80 KB，可全放 LDS。
**必须**：提供 `VLLM_DCU_FP8_FUSE_QUANT`（默认 False）与回退到 O2 路径的能力。

**验收**：`(x_q,x_s)` 与基线逐 bit 等价（同一 group×token 的 amax 归约顺序必须固定）；
每步 kernel 数 512 → 256（census）；step p50 改善 ≥ 1.5%；parity 全绿。

---

### O6 — 端到端验收、parity、回滚（横切，每个波次都用）

1. **工具**：由 `tools/dcu/s20_ab.sh`（换 .so + 重启 + `s20_single_probe.py`）改造出
   `tools/dcu/s23_ab.sh`，增加「python-only 臂」（只 restart，不换 .so）与「arm 描述文件」（记录
   git rev / .so md5 / 全部 `VLLM_DCU_*` / profile 名）。
2. **协议**（缺一不可）：
   - 同窗口背靠背；每臂 **4 次 probe**；报告 **p50 + 均值 + 输出 t/s + accept 长度**；
   - **禁止跨 capture/跨窗口比差**：S23 实测同一配置两次 capture 的 period 差 25%；
   - c=1 用 `s20_single_probe.py`（引擎 step 口径），c=16 用 `s20_conc_sweep.py`；
   - 每次重启后确认 ready + 一次真实 chat smoke（记录 usage）。
3. **回滚**（照 `docs/result/rollback-rehearsal.md` 纪律）：
   - 换装前备份 `TGT=/mnt/game/port/build-sync/temp/_rocm_C.abi3.so` 到
     `/mnt/game/perf1/_rocm_C.<md5>.so`，记录新旧 md5；
   - 回滚判据：md5 与基线一致 + smoke 输出与基线一致（` Paris. The capital of Italy is Rome` 类固定 prompt）；
   - 每个波次结束做一次**真实回滚演练**并记录。
4. **不回归门禁**：prefill TTFT 14K/32K/36K（`tools/dcu/perf1_prefix_cache_probe.py` 或 `s22` 的口径）
   与 c=4/c=16 吞吐；长上下文 decode（`.s22/S22_LONGCTX_DECODE.md` 的探针）至少 1 点。

---

### O7 — CUDA graph（piecewise）可行性复核（P3，time-box 1 天）

S20 记录「CUDA graphs 在此配方下不可启动」（`.s20/S20_FIXED_COST.md` §6-1），但构建与
`compilation_config` 已变（`cudagraph_decoder_replay=True`、`trim_threshold 768`）。

**只允许**：用 systemd unit 的 env 覆盖或新 arm 脚本，在**维护窗口**里试 `--cudagraph-mode piecewise`
（或 vLLM 等价开关），不得改默认配置。

**判据**：能启动（无 OOM，KV 262144 + graph 池）→ parity → step p50。若 1 天内不成立，
写 3 条失败原因（每条带日志行）并关闭本项，不留给后续波次。

**若成立**：可消掉几乎全部派发税（20–31 ms/步），优先级立即升到 P0，但仍需 parity + 长稳。

---

## 4. 执行编排

| 波次 | 并行项 | 说明 |
|---|---|---|
| **W1** | O1、O2、O6（工具就绪）、O7（探索） | O1/O2 都是 python-only，**必须分别做 A/B 臂**（不要同臂叠加后再归因） |
| W1 结束 | O6 跑一次「O1 单独」「O2 单独」「O1+O2」三臂 + 回滚演练 | 若 O1+O2 非线性叠加，记录原因 |
| **W2** | O3（kernel） → O4（同文件，串行） | 每个 cfg/tile 改动都要 mprobe 全 shape A/B，禁止只测一个形状 |
| W2 结束 | O6 跑 c=1/c=4/c=16 三档 | O3 的主战场是 c≥4 |
| **W3** | O5 | 需要 `.cu` + bindings + python，写域最宽，最后做 |
| W3 结束 | O6 终验 + 全项叠加 + 回滚演练 + 结果文档 | 交 `docs/result/z100-fp8-emu-optimization-result.md` |

**写域（互斥，冲突必须串行）**

| 文件 | 归属 |
|---|---|
| `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py`、`vllm/envs.py` | **O1**（O5 也要改 `gfx906_fp8.py` → 必须排在 O1 之后） |
| `vllm/model_executor/layers/quantization/input_quant_fp8.py`、`.../utils/fp8_utils.py`、`.../scaled_mm/BlockScaledMMLinearKernel.py` | **O2** |
| `csrc/rocm/fp8_emu_gemm.cu`、`csrc/rocm/gfx906_emu_common.cuh` | **O3 → O4 → O5**（同一文件，严格串行） |
| `csrc/rocm/torch_bindings.cpp` | O5 |
| `tools/dcu/s23_ab*.sh`、`tools/dcu/s23_*.py`、`docs/result/*` | **O6** |
| `.s23/w1..w3/`（日志与探针原始输出） | 各 agent 自己的子目录，不互相覆盖 |

> AGENTS.md 纪律：**不要为这些小改动单独开 PR**，全部改动进入同一个 S23 变更集合，
> 由人工提交者 review 每一行；PR 描述必须含「为什么不是重复 PR / 跑了哪些测试 / 端到端数据 / AI 辅助声明」。

---

## 5. 统一测量协议（七个坑，实测踩过）

1. **HIP event 对短 kernel 有 +20~35 µs 正偏**（空 event 对 20.3 µs）⇒ 测 quant/launch 地板/小 N
   **必须用 profiler census 或 HIP graph replay**，per-call event 只作参考。
2. **裸 `import vllm` 不注册 op**（惰性导入）：探针必须 `import vllm._rocm_C` 与 `import vllm._C`。
3. **生产池 hipBLAS 不可用**（`HIPBLAS_STATUS_ALLOC_FAILED`）⇒ 参考实现禁用 `@`/`einsum`，
   用 broadcast+sum 手算。
4. **显存**：`mem_get_info()` 报 0 B free，≤64 MiB 可分配，256 MiB 即 OOM ⇒ 一次一个形状，
   测完 `del` + `empty_cache()`；fp8 权重用 `torch.empty_strided(...).random_(0,127).view(float8_e4m3fn)` 生成。
5. **GPU 只能经 `docker exec z100-regress`**（宿主 DSH 沙箱禁止 `open("/dev/kfd")`）；
   一次只跑一个 GPU 探针，`HIP_VISIBLE_DEVICES` 指定单卡。
6. **host 计时受生产 CPU 争用**：同窗口内 4 轮**交错**测（A/B/A/B…），取**中位数**；
   单轮值可能差 1.7×（实测 12 µs vs 23 µs）。
7. **canary 必打**：每个计时臂都跑同一 canary —— `8704:5120` M=1 dense，参考 **0.10625 ms**；
   偏离 >110% 说明被抢占，该窗口数据作废。

---

## 6. 验收与交付

### 6.1 每项必须交付
1. 代码（默认关闭的 env 开关 + 等价实现 + 注释里写清「为什么」而不只是「做了什么」）；
2. `.s23/w<N>/<ID>/` 下的原始日志（probe stdout/json、census、mprobe）；
3. `docs/result/` 下的一份结果文档：基线 vs 该臂的 **kernel 级 + 端到端**双表，含 canary、窗口、
   `.so` md5、`VLLM_DCU_*` 全量 env、未覆盖项；
4. 回滚方法（一句命令 + 判据）。

### 6.2 判定口径
- kernel 级改善 ≥ 目标值时算「内核收益成立」；
- 端到端改善 ≥ 2%（c=1）/ 8%（c=4、c=16）时算「可发布」；
- 只有 kernel 收益、端到端在噪声内 ⇒ **如实写「已实现但未兑现」**（这不是失败，S20 已有先例，
  但不得用「测量噪声」掩盖）。

### 6.3 变更安全
- 换 `.so`：先备份 + 记录 md5；换装后 `ready` + smoke；不合格立即回滚。
- 不动生产默认 env；所有新开关默认 False；A/B 臂显式设置并记录。
- 不在生产进程运行时重编译进 live `.so` 路径（先编译到 build tree，再按 §6.3 换装）。

---

## 7. 明确不做的事

| 不做 | 原因 |
|---|---|
| 把 fp8 权重解量化成 fp16 常驻 | 27B FP8 ≈ 27 GiB，fp16 副本 54 GiB 装不下（S11f 非目标） |
| split-K（无确定性设计） | 破坏可复现求和序，违反仓库纪律 |
| 改 KV-cache fp8 / 改量化语义 | 不在本次范围，收益与风险不匹配 |
| 为「减少 kernel 数」去关掉 activation 量化 | 语义变化，会改数值 |
| 用跨 capture/跨窗口的 period 差当收益 | S23 实测同配置两次 capture 差 25% |
| 基于 `S20` 的 9.21 µs 量化值做决策 | 该 kernel 已被上游 sync 替换（当前是 `..._register_kernel`） |

---

## 8. 参考

- S23 调研：[`docs/result/z100-fp8-emu-fixed-time-investigation.md`](../result/z100-fp8-emu-fixed-time-investigation.md)
- 阶段清单：`.s23/fp8_emu_stage_inventory.md`；微基准：`.s23/fp8_emu_kernel_fixed_times.md`；trace 归因：`.s23/fp8_emu_trace_attribution.md`
- 环境与探针入口：`.s23/PROBE_ENV.md`
- 历史：`.s20/S20_FIXED_COST.md`、`.s20/S20_KERNEL_RESULT.md`、`tools/dcu/s20_kernel_NOTES.md`
- 构建：`/mnt/game/port/build-sync.sh`、`/mnt/game/port/setenv-sync.sh`、`docs/z100-rocm10-build.md`
- 回滚：`docs/result/rollback-rehearsal.md`、`tools/dcu/use_stack.sh`
