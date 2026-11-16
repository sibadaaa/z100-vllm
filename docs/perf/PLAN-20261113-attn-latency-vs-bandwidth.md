# 测试计划:「注意力上不去 = 取数延迟」假设的判定(2026-11-13)

> **待判定假设**(2026-11-13 讨论):注意力慢是不是"算完了,但是要去缓存取数据导致的延迟"。
> **初步拆分**(基于 [SOL-DECODE-PREFILL-20261107](SOL-DECODE-PREFILL-20261107.md)):
> - **H1(decode)**:KV 读取有效带宽 64.5–387 GB/s(屋顶 784 的 8–49%)→ 更像**并发不足/重读**,不是纯延迟等待;
> - **H2(prefill)**:attention 1.56 TF/s(屋顶 2%),已列四因:无 MFMA、**单条 fp32 累加链**、2 CTA/CU 占用、每 tile 重读 Q。
> 本计划用可执行的实验把 H1/H2 各自**证实或证伪**,并区分"取数延迟 / 带宽瓶颈 / 算力未发出"三种机制。
>
> **纪律**:遵循 [AGENTS 诊断流程](../../AGENTS.md) 六步与 §5 方法学(重启彩票 N≥2、同口径、微测→数值→服务三关)。
> kernel 性能探针一律放 `tools/dcu/`(不在 `tests/` 放一次性微基准),正确性进 pytest。

---

## 0. 判定矩阵(先锁判据)

| 实验结果形态 | 结论 | 后续动作 |
|---|---|---|
| 拆段计时:KV-load 段占比 >50%,且有效带宽 ≥600 GB/s(>75% 屋顶) | **真·取数瓶颈**(带宽,不是延迟) | 减 KV 字节:int8 KV / 减重读(BLOCK_M↑),D4 线 |
| KV-load 段占比高,但带宽 <300 GB/s,提 segments/warps/CTA 后带宽升、时间降 | **延迟未隐藏(并发不足)** ← 倾向假设 | 提高并发:segments/occupancy/双缓冲;S58 A2 |
| KV-load 占比低,QK/PV 占比高,VALU busy 高但 FLOPs 低 | **算力未发出(依赖链/ILP)** ← prefill 倾向假设 | 多链累加、拆 dot2 链;S58 A2 |
| KV-load、QK/PV 都低,尾部 softmax/reduce 高 | 拆段归因有漏,重新切分 | 重切段再测 |
| 单 CT(Cold-L2)与热 CT 差异 >15%,且加 prefetch/双缓冲后明显收敛 | **缓存命中延迟敏感** | 预取/双缓冲立项 |

**总判据**:若 T2/T3 的任一"提高并发"臂(见 §2/§3)能把 decode KV 段或 prefill attention 拉到
**≥300 GB/s 或 ≥6 TF/s**,则"算完等取数"的说法**被证伪**——瓶颈在结构不在延迟。

---

## T0. 服务级基线(口径先钉,半天,零改动)

| # | 项 | 命令/工具 | 判定 |
|---|---|---|---|
| T0.1 | 钉 KV dtype 与重扫口径 | `grep kv_cache_dtype` 于现役 launcher + `dcu_policy_fingerprint.py`;确认生产 int8_per_token_head 的字节/token/卡 | §3.1 的 64.5 vs 129 GB/s 口径二选一,**不钉死不准往下解释** |
| T0.2 | 同口径 ITL 斜率复测 | `.venv/bin/python tools/dcu/s24_longctx.py --lens 512,8192,32768,131072 --reps 3 --gen 24 --out runs/attn-diag-<date>` | 斜率应复现 121.3 µs/1k ±10%,否则先查环境 |
| T0.3 | 拆 decode/prefill 两相 | `.venv/bin/python tools/dcu/s27_five_phase_profile.py --phase parse` | 得到 target/draft 各自 kernel 普查,作为 T1 的对照 |

## T1. kernel 内部拆段:QK / PV / softmax / KV-load(1–2 天,核心实验)

SOL 报告 §5.3 A1 的"必做第一步",decode 与 prefill 两个内核都要做
(`kernel_unified_attention_2d` @prefill、`kernel_unified_attention_3d` @decode/verify)。

- **做法 A(改动最小)**:对 [triton_unified_attention.py](../../vllm/v1/attention/ops/triton_unified_attention.py)
  出探针分支(带 kill-switch env,默认关闭),在内核内用 `tl.debug_barrier()` + device-side 计时,
  或拆成 4 个临时内核分别计时(数值等价性见 T4)。
- **做法 B(零改动)**:`rocprofv2 --metrics <list>` 抓单内核,按 roofline 分相归因;
  若 rocprof 在 gfx906 上不支持分相,退回做法 A。
- **真实形状**(照抄 l31 探针):6 q head / 1 kv head / head 256 / block 880;
  decode M∈{1,4,8} × ctx∈{8192,16384,32768,65536};prefill T∈{4096,8192} chunked。
- **必须同时输出**:每段时间 + 每段字节数(KV bytes = ctx × 8KB/int8 或 16KB/bf16 × 重读系数)
  ⇒ 每段有效 GB/s。**只有分段的 GB/s 才能区分"延迟"与"带宽"。**
- 交付:`tools/dcu/attn_stage_timing.py`(探针)+ `runs/attn-diag-<date>/stage-*.json`。

## T2. decode KV 读:并发/重读扫描(判 H1,依赖 T1)

恢复并扩展 l31 探针(git 对象 `2bd80d4327:.s24/probes/l31_verify_scan.py`,已确认可恢复):
`git show 2bd80d4327:.s24/probes/l31_verify_scan.py > tools/dcu/s24_l31_verify_scan.py`

| # | 扫描臂 | 旋钮 | 判定 |
|---|---|---|---|
| T2.1 | segments 因子扫描 | `_decode_num_segments`(l31 已支持) | 若带宽随 segments 上升 → 并发不足(延迟未隐藏) |
| T2.2 | warps / CTA 占用 | `_decode_num_warps`、`num_warps` 上扫到 8 | 配合 rocprof `CTA occupancy` 确认 2 CTA/CU 是否解除 |
| T2.3 | BLOCK_M=16(KV 重读减半) | `VLLM_GFX906_ATTN_DECODE_BLOCK_M=16`(旋钮已存在) | 预期 514→~260 µs/层;**数值门禁见 T4** |
| T2.4 | Cold-L2 vs 热缓存 | HIP graph + 旋转缓冲 >L2(skill 的 ROCm 模式) | 差异 >15% ⇒ 缓存延迟敏感 ⇒ 支持原假设;否则证伪 |
| T2.5 | int8 KV 读路径 | `_kvint8_inner_available` 分支单独计时 | 钉死 int8 口径的字节量与带宽,服务 T0.1 对齐 |

**通过判据**:单层 KV scan 带宽 ≥150 GB/s(S24 计划门,l31 头部已写)→ 追查到 ≥300 GB/s(SOL §5.1 D4)。
**数值门**:每臂对 chunked fp32 参考 `max|Δ| ≤ 1e-3`(l31 自带,不得跳过)。

## T3. prefill attention:ILP/占用归因(判 H2,依赖 T1)

对 `kernel_unified_attention_2d`(8k prefill,82.9 ms/次)按 S58 的四因逐条排除:

| # | 实验 | 工具 | 判定 H2 成立/证伪 |
|---|---|---|---|
| T3.1 | 单链 vs 多链 fp32 累加 | 微基准复用 `.s58` 结论:单链 6.05 / 8 链 22.7 TF/s | 生产内核落在哪一档?靠近单链 ⇒ 依赖链是主因 |
| T3.2 | 占用扫描 | rocprof `Occupancy`/waves;`peak.hip` 对照 2/8/32 CTA/CU | 若内核 2 CTA/CU 且时间随 occupancy 改善 ⇒ 占用是主因 |
| T3.3 | Q 重读消除 | S58 已落地的去重复行改动(1.32×)复测 | 若复测复现 ⇒ 该因仍在,继续追剩余 3× 口径差 |
| T3.4 | dot2 vs dot8 | `v_dot2`/`v_dot8` 微基准(GEMV-OPT 路线) | 无 MFMA 的算力天花板钉死,作为 SOL 修正输入 |
| T3.5 | MemUnitBusy / VALU 计数器 | `rocprofv2 --metrics MemUnitBusy,VALUUtilization,FetchSize` | MemUnitBusy 高 + VALU 低 ⇒ 访存侧;反之为算力侧。**这是 H1/H2 的最终裁决计数器** |

## T4. 数值门禁(每改一臂必过)

```bash
.venv/bin/python -m pytest tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py -q
.venv/bin/python -m pytest tests/kernels/attention/test_triton_unified_attention_gfx906.py -q
.venv/bin/python .s24/probes/l31_verify_scan.py 的内置 fp32 参考(探针内)
```

- per-op `max|Δ| ≤ 1e-3`(对 fp64/ chunked fp32 参考);
- BLOCK_M 改 M 求和顺序 ⇒ 近平局翻转按 ATTN-OPT §3 的**可执行 parity 协议**披露
  (top-2 logprob 差 <0.05 nat、双向、逐条记录),不默认拒。

## T5. 服务级验证(只对 T2/T3 中胜出的 1–2 臂做)

```bash
VLLM_GFX906_ATTN_DECODE_BLOCK_M=16 MAX_SEQS=4 MBT=8192 PORT=8000 bash tools/dcu/s27_graph_launch.sh
.venv/bin/python tools/dcu/s20_single_probe.py --reps 5 --tokens 128
.venv/bin/python tools/dcu/perf1_e2e_parity.py --arm <arm> --prompts 8
.venv/bin/python tools/dcu/s27_five_phase_profile.py --phase parse   # 确认收益落在 target 段
```

- 门:ITL p50 改善 ≥3%(32k 口径,同窗口 5 reps)、accept 不降、439-prompt parity 无单向劣化、
  0 crash;重启 N≥2(`ROCBLAS/HIPBLAS_WORKSPACE_CONFIG=:4096:8`)。
- **收益必须能从服务级 TPOT 里看到**,否则按 AGENTS 第 6 步记为"被框架开销抵消",不采纳。

---

## 排期与依赖

| 顺序 | 内容 | 依赖 | 产出 |
|---|---|---|---|
| T0(0.5 天) | dtype/斜率/五相位基线 | 无 | 口径钉死 + 对照数据 |
| T1(1–2 天) | 拆段计时探针 | T0 | stage-*.json,初步归因 |
| T2/T3(并行,2–3 天) | decode 扫描 / prefill 归因 | T1 | 判定矩阵逐行填表 |
| T4(随做) | 数值门禁 | — | pytest + parity 记录 |
| T5(0.5 天/臂) | 服务级验证 | T4 | 采纳/退役决定 |

**反证条款**:若 T1 显示 KV-load 段带宽已 ≥600 GB/s(屋顶 75%+)且 QK/PV 是大头,
则本轮"取数延迟"假设**不成立**,主线回到 S58 的算力/ILP 路线,不做 T2 的并发扫描。

---

## 执行状态（2026-11-13 收工）

**已完成 T0–T5**，结果见
[`docs/result/z100-s91-attn-latency-vs-bandwidth.md`](../result/z100-s91-attn-latency-vs-bandwidth.md)，
台账 `.s91/README.md`，数据 `runs/attn-diag-20261113/`。

| 项 | 状态 | 关键结论 |
|---|---|---|
| T0.1 口径 | ✅ | int8 KV，block 1616，**520 B/token/卡**；16 层 full attention |
| T0.2 斜率 | ✅（先修栈） | 旧 `.so` 406.9 µs/1k → 修复后 **118 µs/1k**（文档 121.3，差 2.7%） |
| T0.3 五相位 | 部分 | 改为 32k prefill 内核普查：`attnd_inner_kvsdot4` 24.7%、GEMM 55%、AR 6.8% |
| T1 拆段（内核内 `clock64()`） | ✅ | decode load 20–25% / qk 29–30% / pv 44–46%；prefill load 17.5% / qk 31% / pv 48% |
| T2 decode 并发 | ✅ | segments 16→256 把 load 段带宽 228→705 GB/s，256 后饱和；BLOCK_M/warps 已是编译期常量 |
| T3 prefill ILP/占用 | ✅ | segments 1→8 时间不变（127.2→124.5 ms）⇒ 不是并发，是 PV 指令形态 |
| T4 数值门 | ⚠️ 部分 | stage op 逐位同、dispatch vs fp32 ≤6.4e-4 过；pytest 9 failed 为**既存**（两臂同红） |
| T5 服务级 | ✅ 采纳 | 131k TTFT 991.7→237.6 s、e2e 65.3→72.7 t/s、accept 不降 |

**判定**：H1（decode「算完等取数」）**证伪**；H2（prefill 算力未发出）**成立**。
反证条款触发 ⇒ 主线回到算力/ILP，不在 decode 立项预取/双缓冲。

**栈级事故（先于判定）**：S89 relink 用了构建树里 S81 之前的 `torch_bindings.cpp.o`，
`gfx906_attnd_decode_inner_kvint8` 注册丢失，decode/prefill 退回 Triton int8 反量化；
修复 `.so` = `.s91/arms/_rocm_C_kvint8fix.abi3.so`（md5 `5af8b848f08e`），已上线。
