# S34 无 MTP（SPEC_OFF）decode 分环节实测 — 2026-10-30

> **后续**：本报告结论已折成施工设计 `docs/design/z100/s38/z100-s38-design.md`
> 与 `docs/design/z100/s38/z100-s38-construction.md`（A1 W4 融合 AR 部署 /
> A2 nccl 回退归位 / A3 logits 定位 / B FP8 GEMV 小 K 特化）。
> 目的：按操作要求关闭本地 27B-FP8 服务的 MTP，重测 decode 各环节用时，找出剩余优化点。
> 结果：MTP 已从 systemd 生产臂持久关闭（改动仅 2 处脚本），服务恢复并复测通过；
> decode step ≈ **26.7 ms**（37.3–38.3 greedy t/s），GPU 占空比 ≈ 92%，
> 最大两个可优化面：**AllReduce 复合路径 ≈38%** 与 **FP8 GEMV 加权 561 GB/s
>（读上限 784.8 的 71%，短板 K=1536 的 o/out 仅 309，详见 §6）**。

## 1. 配置变更（MTP 关闭）

| 文件 | 改动 |
|---|---|
| `.s34/launch_mtp2.sh` | 实现 DEPLOY.md 里写了但从未实现的 `SPEC_OFF=1` 契约：置 SPEC_OFF=1 且未给 SPEC_SET 时不传 `--speculative-config`，其余（kernel/env 栈）零改动 |
| `.s34/serve_deploy.sh` | `export SPEC_OFF=1`（含回滚注释），systemd 链生效 |

- 启动日志确认：`SPEC_OFF=1 -> no --speculative-config`，`spec=`（空）。
- 指纹 `FP 2f3793deb97e138f` 与上一臂 `937d69861f19c41d` 的 FP_FULL 全量 diff 只有一行：
  `VLLM_DCU_GCN_ARCH=auto → gfx906`（env.sh 把空值解析成 gfx906 后指纹记录了显式值；
  envs.py 默认空 = auto-detect）。**功能配置无差异**，指纹漂移是 S36/S37 阶段的记录口径变化。
- **回滚**：注释 `.s34/serve_deploy.sh` 里的 `export SPEC_OFF=1` 一行，
  `sudo systemctl restart z100-vllm-mtp2`（MTP k=2：greedy 51.6 t/s，warm TTFT 2.4 s）。
- 注意：`.s34/start.sh` 的"spec 行应含 num_speculative_tokens: 2"检查在本配置下会打印
  `NO SPEC LINE`——属预期，不算失败。

## 2. 服务级复测（同口径：7469-token probe，batch=1，prefix 热）

| 配置 | W1 贪心 t/s | W2 采样 t/s（0.7/rp1.05） | warm TTFT | step 折算 |
|---|---|---|---|---|
| **本臂（S34 栈 − MTP）** | **37.45**（37.03/37.45/38.08）重启后复测 37.84 | **37.07**（37.06/37.08） | **0.72–0.76 s** | **26.7–27.0 ms** |
| S34 MTP k=2（关闭前，REPORT.md） | 51.6 | 40.6 | 2.40 s | ~19.4 ms/tok（验收 ~2.1–2.6） |
| 旧 s33 A 臂（REPORT.md，WP3/WP2 之前） | 36.9 | 34.7 | 0.72 s | 27.1 ms |

→ 无 MTP + WP3 GEMV/AR128 栈比旧 A 臔 greedy +1.5%、采样 +6.8%；TTFT 相对 MTP 臂 3.3× 改善。
W1 三条流 in-arm 逐 token 一致（确定性正常）。产物：`e2e-s34nospec.json`、`e2e-s34nospec-smoke.json`、`parity-*.json`。

## 3. decode 分环节用时（torch profiler，rank0，39 个 decode step，batch=1，ctx≈7.5k）

窗口：`/start_profile(max_iterations=40)` → 96-token 贪心流。trace：
`/mnt/game/s30c-prof/s34nospec_rank0..3.*.pt.trace.json.gz`。分析脚本：`parse_trace.py`。

**每步中位数：CPU wall 29.09 ms｜GPU busy(union) 24.50 ms｜GPU span 28.72 ms｜步间 CPU gap 4.36 ms**
（profiler 自身放大约 6–7 ms host；真实 step 26.7 ms = GPU busy 24.5 + host 空隙 ≈2.2 → **GPU 占空比 ~92%，已基本 GPU-bound**）

| 环节 | ms/step | 占 GPU busy | 明细 |
|---|---:|---:|---|
| **FP8 W8A16 GEMV（线性层）** | **10.84** | **44.8%** | `fp8_emu_gemv_stream_m1` 45.2 µs × 256/step（4/层）；有效带宽 **623 GB/s** vs D2D 上限 777.5 |
| **AllReduce 复合路径** | **≈10.1** | **≈38%** | ① push-AR `cross_device_reduce_push` 63.3 µs × 129/step（2/层+1）≈ 8.16 ms（union 7.45）② **nccl ring 回退** 481 µs × 3.3/step ≈ 1.59 ms（GDN core 65 次 + compiled graph 内 48 次）③ lm_head logits **AllGather** 365 µs × 1/step |
| norm/elementwise | 2.86 | 11.8% | fused_add_rms_norm 15.2 µs × 49 + rope/qk-norm 等 ~150 个小 kernel/step；另有 `fp8_dequant_to_fp16` 98.9 µs × 256/step（0.65 ms） |
| attention core（16 层 full-attn） | 1.23 | 5.1% | `unified_attention_3d` 83.3 µs × 16/step；KV 读 ≈123 MB/step（4 KV 头 TP4，head_dim 256）→ 有效 ~100 GB/s |
| GDN（48 层 linear-attn） | 0.89 | 3.7% | m1_native delta-rule 12.6 µs + conv1d 8.4 µs × 48/step |
| bf16 GEMV（门控/嵌入侧小线性） | 0.52 | 2.1% | dot2 29.9 µs × 49/step |
| kv cache 写 + memcpy + other | 0.45 | 1.9% | — |
| **GPU busy 合计** | **24.2** | 100% | 单流（stream 29），无跨流重叠 |
| host 空隙（真实） | **≈2.2** | — | profiled 内部 gap 3.57（python 层）+ 步间 4.36，CUPTI 放大后仍与 26.7 ms 对账吻合 |

**对账**：10.84+7.45+1.59+0.37+2.86+1.23+0.89+0.52+0.45 ≈ 25.2（sum 口径）≈ union 24.2 + host 2.2 ≈ 真实 26.7 ms → 分解自洽，AR 的 spin-wait 在生产里也是真实占用（非纯 profiler 伪影）。

**`.item()` 澄清**：聚合表 267 次/345 ms 看似惊人，按 step 切开后 decode 只有 **3 次/step 且 self≈0**；
1.3 ms 级的同步几乎全在 prefill（~150 次）。decode 主循环**不是** `.item()` 受害者。

**prefill（顺带）**：269-token 缓存未命中段 763 ms：attention_2d 178 ms（16×11.15 ms）+ rocBLAS fp16 路径 GEMM ≈283 ms（P0 物化路径生效）。

## 4. 优化点排序（按 26.7 ms step 的预计收益）

1. **AllReduce 复合路径（≈38%，预计 −2~4 ms，−8~15%）**
   a. *nccl ring 回退*：3.3 次/step 走了 481 µs 的 nccl 而不是 63 µs 的 push-AR（位置：GDN core 内 65 次 + 3 个 compiled graph 内 48 次 + fp8_mm 内 8 次）。让这些 AR 也走 custom push（graph 捕获边界/条件排查）→ **−1.0~1.6 ms**。
   b. *lm_head AllGather 365 µs*：M=1 只 gather 一个 token 的分片，365 µs 明显偏高（对齐 push-AR 量级应 ≤50 µs）→ **−0.3 ms**。
   c. *结构性*：2 AR/层 → 1（attn-out 与 mlp-out 合并，或 sequence-parallel）；或把 AR 与无依赖 GEMV 重叠（WP3 stream 思路延伸）→ 理论 **−3~4 ms**。W4 融合 AR 核（S37 已建好未部署）即属此类。
2. **FP8 GEMV（44.8%，详见 §6 专题：预计 −1~2 ms）**：加权 561 GB/s = 读上限的 71%；短板在 K=1536 的 o/out 两类（309 GB/s）。
3. **norm/elementwise（−0.5~1 ms）**：~150 个 5–15 µs 小 kernel/step，继续融合（rms-norm 尾部、残差 cast、dequant 融进 GEMV prologue）。
4. **attention decode kernel（≤−0.6 ms）**：~100 GB/s 是占用率/延迟受限（batch=1、BLOCK_M=16），绝对值小，优先级低。
5. **host 空隙 ≈2.2 ms**：scheduler/sampler/属主侧，收益需框架级改动；decode 已 92% GPU-bound，单独压缩 host 帮助有限（与 AGENTS.md 诊断流程的判断一致）。

## 5. 过程事故与处置（重要，影响后续 profile 方法）

- `max_iterations=40` 只截断**本轮 trace**，profiler 保持激活；本栈 push-AR 是自旋等，
  CUPTI 使 4 rank 失步 → 自旋被无限放大，窗口后 95 token 用了 319 s（~3.3 s/step）；
  结束时 `sample_tokens` RPC 超时、`/stop_profile` 也超时，引擎楔死。
- 处置：`sudo systemctl restart z100-vllm-mtp2`（~60 s 恢复，编译缓存热），复测 37.84 t/s 正常。
- **结论/规程**：本栈上 profile 窗口必须"一发请求 + 立即 `/stop_profile`"，不要依赖 max_iterations 兜底；
  或 profile 前临时 `VLLM_CUSTOM_ALLREDUCE_ALGO` 换非自旋路径。已按此拿到干净的 39-step 窗口。

## 6. FP8 GEMV 专题（算术强度 + 分级带宽，2026-10-30 追加）

> 方法论（操作者框架）：M=1 GEMV 先看算术强度定 roofline 位置；GEMM 类 kernel 以**带宽**
> 为主指标（延迟由海量线程 + 预取流水线隐藏，带宽是物理硬限）；优化是否有效看它
> 是否降低某级存储的带宽需求。

### 6.1 算术强度与 roofline

- M=1：AI = 2·N·K / (N·K·1B 权重 + 2B 激活 + 2B 输出) ≈ **2 FLOP/B**（权重主导，与形状无关）。
- 机器平衡点（ridge）：实测最好 GEMM 10.64 TF/s ÷ 读上限 784.8 GB/s ≈ **13.6 FLOP/B**。
- **AI 2 ≪ 13.6 → 深度 memory-bound**；即使带宽打满，算力利用率也只有 ~15%。
  → 带宽是唯一指标，"提升算数强度"在此无意义（M=1 无复用可挖），唯一杠杆是把
  各级存储的实际搬运速率逼近上限。

### 6.2 本机带宽上限（本轮实测，同卡同进程）

| 上限 | 值 | 说明 |
|---|---:|---|
| 只读 reduce（1 GiB fp16 sum） | **784.8 GB/s** | GEMV 的诚实目标（纯读） |
| D2D copy（512 MiB，读+写） | 809.6 GB/s | 与 KERNEL_REPORT 的 777.5 锚点一致 |
| L2 | 8 MiB | 60 CU（Z100L）；单次 GEMV 权重 7.9–44.6 MB，**单用无复用，L2 帮不上** |

### 6.3 trace 纯 GPU 时长的分类速率（decode 步内，真值）

每步 fp8 GEMV 239.67 次 / 10.84 ms，权重字节 6.08 GB/步 → **加权 561 GB/s = 读上限 71%**：

| 形状类 | [N,K] per-rank | 次/步 | µs/次 | GB/s | vs 读上限 |
|---|---|---:|---:|---:|---:|
| gate_up | 8704×5120 | ~60 | 72.5 | **615** | 78% |
| down | 5120×4352 | ~44–64 | 38.8 | **575** | 73% |
| attn_qkv | 3584×5120 | ~14–16 | 35.1 | **524** | 67% |
| gdn packed-in（in_qkv+z） | 4096×5120 | ~58 | 44.0 | **477** | 61% |
| **attn_o / gdn_out** | 5120×1536 | ~59–64 | 25.6 | **309** | **39%** |

### 6.4 冷读微测（`gemv_bw_bench.py`，rows∈{4,8}，43–102 份轮转权重 defeat L2）

- **cold == hot**（逐形状差 <2%）：确认这些流式读在 8 MiB L2 下本就无复用可言，
  微测不是 L2 虚高——但微测绝对值含 ~20–26 µs/次的 launch+event+sync 测量开销
  （对比 trace 同形状 25–72 µs 可证），**trace 分类速率才是地面真值**。
- **rows=8 全面慢于 rows=4**（qkv 57.4 vs 55.8 µs、gate_up 100.0 vs 96.8、down 67.5 vs 65.3）
  → 生产 rows=4（WP3）在 M=1 已是该轴最优，不必回调。
- split-K（S37 A3）已判死：两核结构 + 每块 K 减半后预取跑道变短，收益被全额吃掉。

### 6.5 短板诊断与方向

- **瓶颈不是流水线结构**：kernel 已是"LDG 先行"——2 个 slice 的双缓冲预取
  （`pf[2][RW]`，计算 slice s 时 s+2 的 W 已在飞），A/LUT 走 LDS，累加器全程寄存器。
- **K 归约已在 V6 终点**：寄存器累加跨 slice 累计 + 结尾一次 `__shfl_down` 波内归约
  （无 shared 归约、无额外 syncthreads）——V1→V6 阶梯在此 kernel 无剩余收益；
  阶梯的意义在评估其它归约（如 Triton 生成的 norm 归约，编译器已内建 shuffle 树）。
- **真正的短板：K=1536 的 o/out 两类（309 GB/s，64 次/步共 1.64 ms）**。K=1536 只有
  1.5 个 slice（kStreamSlice=1024），双缓冲预取刚起飞就收尾，跑道不足 + 半 slice 尾巴。
  方向（遵守 S37 结论"不增加核数"）：
  a. 小 K 形状用更小的 kStreamSlice（如 512）→ 3 个整 slice，预取充分展开；
  b. 尾半 slice 在 block 内跨 lane 拆分（lane 级 K 拆分），避免半空转；
  c. RW=2 双行交错（每 warp 2 个独立累加链，同 slice 的 LDG 双倍在飞）。
  若 o/out 提到 ~500 GB/s：**−0.6 ms/step（−2.4%）**。
- gate_up/down（73–78%）剩 22–27%：LDS-LUT 查表指令压力与扇区利用率，属二阶收益。
- **GEMV 总可挖空间：−1~2 ms/step（−4~8%）**；与 AR 路径（W4 融合核已建好待部署，
  −0.7~1.2 ms）相比，**先落地 W4 更便宜**。

### 6.6 待查（不阻塞）

- decode 步内**未见独立 lm_head logits 大核**（无 >150 µs 调用、无 Cijk/aten::mm），
  但每步有 1 次 365 µs AllGather（疑似 logits 交换）。logits 计算路径需单独追
  （Qwen3_5 compute_logits 的 M=1 分支），若它也是 GEMV 且隐藏在某个分类里，
  字节账还有 ~0.6 GB/step 的变数。

## 7. 产物

- `runs/s34nospec/e2e-s34nospec.json` / `e2e-s34nospec-smoke.json` / `parity-s34nospec*.json`
- `runs/s34nospec/steps-s34nospec.json`（39 步逐帧分类/流/gap 归因）
- `runs/s34nospec/{profile_window.py,parse_trace.py}`（可复用）
- `/mnt/game/s30c-prof/s34nospec_rank0..3.pt.trace.json.gz` + `profiler_out.txt`
- 服务现状：**MTP 关闭运行中**（port 8000，greedy 37.8 t/s，TTFT 0.72 s）
- `runs/s34nospec/gemv_bw_rows{4,8}.json` + `gemv_bw_bench.py` / `run_gemv_bench.sh`
  （冷/热 GEMV 带宽微测 + 只读/D2D 上限；§7 数据源）

## 8. 口径修正(2026-10-31)

§6.2 的"只读 reduce 784.8 GB/s"是 torch.sum 归约口径。s38 用 triton 独立宽
load 纯读内核重标(1 GiB,击穿 L2):**纯读流式上限 ≈ 905 GB/s**(D2D copy
829.4、reduce 792.5;数据 `docs/perf/data/s38-bw-ceiling.txt`,工具
`.s38/bw_ceiling.py`)。Z100L(HBM2,pin ~1 TB/s)的纯读效率 88%,归约口径
低 ~13%。因此 §6.3/§6.5 中各形状"vs 读上限"的百分比应按 905 重算:
GEMV 加权 561 → 62%,o/out 309 → 34%,gate_up 615 → 68%。GEMV 桶的诚实
地板(6.08 GB/步 ÷ 905)= 6.72 ms(现 10.84)。
