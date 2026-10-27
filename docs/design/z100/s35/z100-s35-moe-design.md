# S35 设计图：MoE 架构适配（2026-10-27）

> **决策输入**：用户确认范围 = **正确性 + 基线 + 内核性能优化**；GPU 卡分配**执行时定**
> （WP0 门禁，生产服务为红线，永不 kill）。
> **依据**：`docs/result/z100-moe-feasibility-investigation.md`（2026-10-19，下称
> **MoE 调查**，其风险编号 R1–R11 沿用）；施工步骤见配套施工图
> `docs/design/z100/s35/z100-s35-moe-construction.md`。
> **定位**：本图是 MoE 调查 §7"最小验证矩阵"的决策定案版 + 性能阶段扩展。
> 本机（Z100 / DTK 26.04 栈）**从未跑过任何 MoE e2e**；fork 时代（2026-09，
> gfx906 lineage）的 GGUF MoE 实测结论未在当前栈复验。

## 1. 目标与口径

| 维度 | 口径 | 现值 | 目标 |
|---|---|---|---|
| 覆盖 | 本机 MoE 模型 serving-ready 数 | 0 | **≥2**（Tier A + Tier B） |
| 数值 | MoE 内核数值门禁 | 无（R4） | `tests/kernels/test_gfx906_moe.py` 全绿，不依赖 aiter |
| DP4X | C++ MoE 内核 VGPR 审计 | 未审计（R1） | 审计完成，卡 2/3 放行或锁定（二选一定案） |
| 性能·基线 | MoE decode/TTFT 本地数据 | 无（R7，仅有上游 Issue #29 传闻） | 定案基线（N≥2 重启） |
| 性能·优化 | GGUF MoE decode 内核 | 未调优 | 过门项落地：per-kernel ≥1.5× 且 e2e decode **+20% tok/s** |
| 文档 | README MoE 状态 | fork 时代口口相传 | 实测 + 机制化说明（走哪条内核 / guard 覆盖 / SKU 差异） |

强制口径（不变）：**数值先于性能**；e2e 结论 N≥2 重启；每 WP 独立可回滚；
改模型输出的项走**零翻转硬门**；每 WP 归档 `docs/result/z100-s35-*.md`
（结构：目标/改动/原始证据/门禁判定/未完成项/对后续影响/复现方式）。

## 2. 墙位（已核实的代码/环境事实）

### 2.1 MoE 内核选型（gfx906 实际路径，MoE 调查 §2.1 复核有效）

| 权重类型 | 实际内核 | 语言 | DP4X guard 覆盖 |
|---|---|---|---|
| 非量化 fp16/bf16 | `fused_experts` → `TritonExperts` | Triton | ✅（`hygon_dp4x.py` 包 `JITFunction.run`，`_DP4X_MAX_VGPR=253`） |
| GGUF（Q*_K） | `_fused_moe_gguf` → `ops.ggml_moe_a8`（prefill, MMQ）/ `ggml_moe_a8_vec`（decode, moe_vec） | **C++ HIP** | ❌ |
| WNA16（GPTQ/AWQ, bits∈{4,8}, gs∈{32,64,128}） | `should_moe_wna16_use_cuda`→True → `ops.moe_wna16_gemm` | **C++ HIP** | ❌（README 记载失败组合） |
| aiter MoE 后端 | `get_cdna_version()==0` 硬性关闭 | — | 排除 |
| FP4/MXFP4 | gfx906 无数据通路 | — | 排除 |

- guard 实现：`vllm/platforms/hygon_dp4x.py`（仅包装 Triton；C++ 内核裸奔）。
- **栈相关**：F24 故障在自编 ROCm 10.1 用户态复现；当前生产栈 **DTK 26.04
  用户态带 DP4X 支持**（`docs/result/z100-minimax-h3-quant-feasibility.md` §0.9）
  ⇒ C++ 内核风险在当前栈上低于 MoE 调查估计，但仍需审计定案（WP4）。
- 调优配置：`vllm/model_executor/layers/fused_moe/configs/` 仅 7 个
  `AMD_GFX906` JSON（`E=128,N=768/int8_w8a16` 14 档、`E=128,N=192/int8_w8a16`
  18 档、`E=256,N={128,256,512,1024}/int4_w4a16` 各 4 档、`E=20,N=1536` 4 档）。
  **fp16 MoE（dtype=None）无任何配置，必回退通用 tile**（`fused_moe.py:1094+`
  `get_moe_configs`；用户目录覆盖 env：`VLLM_TUNED_CONFIG_FOLDER`）。
  **GGUF 路径不消费这些 JSON**（独立 C++ 内核）⇒ 配置调优只对 Triton 路径有效。

### 2.2 候选模型（MoE 调查 §3 矩阵，复核有效）

| 档 | 模型 | 形状（E/topk/N/hidden） | 显存 | 备注 |
|---|---|---|---|---|
| A | `allenai/OLMoE-1B-7B-0924`（fp16 原生） | 64/8/1024/2048 | 12.9 GiB 单卡 | 流水线验证车；非生产候选（模型较老） |
| B | `unsloth/Qwen3.6-35B-A3B-GGUF` | 256/8/512/2048，40 层，**GDN 混合架构**，n_kv_heads=2 | ~20 GiB 单卡（int4） | 唯一有 fork 时代实测记录的 MoE；**性能优化主目标** |
| C | Qwen3-30B-A3B W8A16（条件档） | 128/8/768/2048 | 14.2 GiB/卡 ×TP=2 | 唯一有深度调优配置（14 档）；W8A16 MoE 走 C++ `moe_wna16_gemm`（README 失败组合，需 WP4 先行） |

排除：MoE+GPTQ-Int4/AWQ-4bit（AutoGPTQ，已实测失败）、GPT-OSS-20B
（MXFP4 无通路）、Qwen3-235B（4×32 GiB 任何格式不可行）、Mixtral fp16。

### 2.3 硬件与环境

- 4 卡：torch 0/1 = **Z100L**（DP2X，`dp4x_status=0`），torch 2/3 = **Z100**
  （DP4X，`dp4x_status=1`）；均 32 GiB，gfx906，无 MFMA（GEMM 纯
  VALU/`v_dot*`），有 `v_dot2_f32_f16` / `v_dot4_i32_i8`，无 FP8/FP4/bf16 原生指令。
- 主机 RAM ~31 GiB（TP≥3 加载期需 swap，历史配方见 `docs/result/z100-14c-*.md`）。
- 生产服务：`Qwen3.8-27B-FP8 TP=4` 占满 4 卡（:8000，VRAM ~91%）⇒ **卡分配
  执行时定**（WP0 门禁）；MoE 服务端口用 8101/8102，绝不碰 8000。
- 生效栈：DTK 26.04（`tools/dcu/env.sh` + `DTK_ROOT=$DATA_ROOT/host-runtime/dtk2604`
  + `.venv`）；开工先 `tools/dcu/use_stack.sh status` 确认。
- 磁盘：`$DATA_ROOT` 余 ~231 GB（权重下载无压力）。
- 本机无任何 MoE 权重 ⇒ WP0 含下载（`tools/dcu/s9_par_dl.py`，带 sha256 校验；
  HF `resolve` 端点 Content-Length 有短报先例，S8b 教训）。

## 3. 设计决策

### D1（WP0）：资源门禁与卡分配
- 空闲卡判据：`hy-smi` VRAM <10% 且无 vllm 进程绑定。
- 决策树：**无空闲卡 → 停，报告用户**（不 kill 生产）；仅卡 0 → 单卡路径
  （WP1/2/3），WP4 顺延；卡 0+1 → 同上 + WP9（TP=2）可行；卡 2/3 空闲 →
  WP4 可在卡 2 执行（审计本身需要 Z100 卡）。
- 权重下载（OLMoE ~14 GB + GGUF ~20 GB）与 GPU 等待**并行**，WP0 内启动。

### D2（WP1）：MoE 数值门禁（关闭 R4）
- 新文件 `tests/kernels/test_gfx906_moe.py`：fp16 **Triton 路径**
  （`fused_experts`/`TritonExperts`）vs 手写 torch expert-loop 参考
  （fp32 累加）；E=8, topk=2, K=512, N=256；M∈{1,8,64,256}（decode/prefill
  两域）；silu_mul。
- 判据：max(|Δ|/(|ref|+1e-3)) < 2e-2 且 cosine > 0.9995（对 fp32 参考）+
  两次运行位同 + 无 aiter import。
- GGUF/WNA16 的合成权重内核级对照列为**可选延伸**（不阻塞门禁）；其数值验证
  由 WP3 的 e2e 连贯性 + 卡间对照承担。
- 纯新增文件，删除即回滚。

### D3（WP2）：Tier A e2e + 首个本地 MoE 基线
- `OLMoE-1B-7B-0924` fp16 TP=1 卡 0，端口 8101；cudagraph 默认
  （FULL_AND_PIECEWISE），捕获失败逐级降级 PIECEWISE → enforce-eager
  （S7c 后 eager 已安全），**基线必须记录所用图模式**。
- 判据：ready + ≥3 条中英文请求文本连贯 + 无 NaN + dmesg 0 故障 +
  decode tok/s / TTFT / conc8 基线（N≥2 重启，`.s35/probe.sh` 口径）。
- 意义：本机第一个 MoE 数据点，替换上游 Issue #29 传闻（R7）。

### D4（WP3）：Tier B GGUF 复验（生产候选）
- `unsloth/Qwen3.6-35B-A3B-GGUF` 单卡（卡 0，端口 8102）。
- 判据：ready + 中英文长生成连贯 + dmesg 0 故障 + decode 基线（N≥2）。
- **卡 2 对照（若卡 2 空闲）**：同权重卡 2 vs 卡 0 token 流对照
  （S14b 方法论：先测同卡噪声底再判分歧）⇒ DP4X 在当前 DTK 栈上的
  **行为级**定案（与 WP4 的 VGPR 数据互补）。
- 附加风险面（R10）：GDN 混合架构的 FLA Triton 内核历史 VGPR=256（F24 §1.4），
  在 guard 覆盖内，但 dmesg 监控必开。

### D5（WP4）：C++ MoE 内核 VGPR 审计（关闭 R1）
- 方法：hip shim（`$DATA_ROOT/f24-investigation/hipshim/libhiplog.so`，
  `LD_PRELOAD` + `HIPLOG_VGPR_WARN=0` 出全表；shim 在 ROCm 10.1 栈验证过，
  DTK 栈上先跑其 README 冒烟）。
- 对象：`ggml_moe_a8`、`ggml_moe_a8_vec*`、`topk_softmax`、
  `moe_align_block_size`、`moe_sum`（跑 WP3 的 GGUF 模型于卡 2 采集；
  `moe_wna16_gemm` 仅当 WP9 启动时补）。
- 判据：全部 `numRegs ≤ 203` → R1 关闭，卡 2/3 放行 MoE；任一 ≥248 →
  MoE 锁定卡 0/1（写入 README）。**DTK 栈用户态带 DP4X 支持** ⇒ ≥248 也不
  必然故障，最终以 D4 的卡 2 行为级对照为准；两者结论冲突时按更保守者定案。

### D6（WP5）：MoE 性能剖析与墙位鉴定（只测量，不改码）
- 对象：Tier B 模型（性能优化主目标）；OLMoE 作对照（Triton 路径）。
- 工具：`tools/dcu/s27_five_phase_profile.py --phase collect` +
  `.s29/kdiff.py`（collect 会 wedge 服务，collect 后立即重启，S33 须知 4）。
- 产出：MoE 相关内核（专家 GEMM / topk / align / permute / gate / 共享专家）
  的 us_per_round 分账 + **有效带宽**（权重字节/耗时，对照稠密侧实测天花板
  GEMV ~650–700 GB/s、d2d ~771 GB/s）+ decode/prefill 双域墙位鉴定
  （带宽 / 算力 / launch 三选一为主）。
- 判据：分账覆盖 ≥90%；产出 `.s35/wp5_moe_wall.json` + 墙位结论。
  **本 WP 结果决定 WP7/WP8 的方向与优先级**（AGENTS.md 诊断纪律：
  先确认撞哪堵墙，再写受保护的优化路径）。

### D7（WP6）：Triton 路径调优配置（低成本，不改内核）
- 对象：OLMoE fp16（`E=64,N=1024,device_name=AMD_GFX906.json`，dtype=None）；
  若 WP9 启动则加 30B int8（已有 14 档配置，只做复核）。
- 工具：新写 `tools/dcu/s35_moe_tune.py`（扫 BLOCK_SIZE_M/N/K、GROUP_SIZE_M、
  SPLIT_K × M 档，写 JSON 到 `.s35/tuned/`，经 `VLLM_TUNED_CONFIG_FOLDER` 注入）。
- 判据：e2e decode 改善 ≥5% 且其它 M 档不回退；否则弃用配置（文件删除即回滚）。
- **GGUF 路径不在此 WP 范围**（不消费配置 JSON）。

### D8（WP7）：GGUF MoE decode 内核优化（主力内核项，分三阶段）
- **阶段 1（微基准）**：真实 shape 微测现 `ggml_moe_a8_vec`
  （decode 行宽 = topk 8 × N=512、Q2_K/Q4_K 反量化），测 GB/s 与天花板比；
  读 `csrc/quantization/gguf/moe_vec.cuh` 现有旋钮（vec 宽度 / split-K / 线程布局）。
- **阶段 2A（旋钮）**：若现有旋钮可及天花板 ≥80% → 只调参，不改核。
- **阶段 2B（新核，条件启动）**：若 2A 后仍 <50% 天花板 → 写 exact-shape
  MoE expert GEMV（参照稠密侧 `fp8_emu_gemv_wholea_m1_kernel` 经验 + 1Cat
  exact-shape 战术，`docs/result/z100-11g-1cat-vllm-reference-analysis-result.md`）；
  新文件 + 独立 env `VLLM_DCU_MOE_GEMV`（默认 0）+ 指纹 KNOBS 同步 +
  build-sync 换装（C7 纪律，目标 `_C_stable_libtorch`）。
- **硬门**：per-kernel ≥1.5×；e2e decode **+20% tok/s**（N≥2）；
  **零翻转**（改累加序项走 S14b/BM16 流程）；长跑 30 min 0 故障；
  dmesg 0。
- 回滚：env 一行 / 二进制 cp 回 bak。

### D9（WP8，条件）：GGUF prefill MMQ 优化
- 启动条件：WP5 显示 prefill 阶段 MoE 专家 GEMM 占 prefill 时间 >40%。
- 对象：`ggml_moe_a8`（MMQ）；同 D8 三阶段结构（微测 → 旋钮 → 新核）。
- 硬门：prefill tok/s +15% 且 TTFT 不回退；零翻转。

### D10（WP9，条件）：Tier C Qwen3-30B-A3B W8A16 TP=2 投资判定
- 前置：WP4 审计通过（`moe_wna16_gemm` numRegs ≤203 或卡 2 行为级对照通过）
  + 用户在 WP5 墙位报告后拍板。
- 形态：W8A16 权重 TP=2 卡 0,1（命中 `E=128,N=768,int8_w8a16` 14 档配置）；
  判据：ready + 文本 + decode 与稠密 27B 基线对比（A3B 激活参数优势是否
  被路由/内核开销吃掉）。
- 注意：这是 README 记载失败组合的**内核路径**（C++ `moe_wna16_gemm`），
  失败即止损，不恋战。

### D11：明确不做（防重开）
MoE+GPTQ-Int4/AWQ-4bit（AutoGPTQ）；GPT-OSS-20B（MXFP4）；Qwen3-235B；
aiter 后端；**EP（专家并行）**（gfx906 无可用 all2all 后端，单机 4×32 GiB
无 EP 场景）；MoE 上叠 MTP（R6：K2 token-parity 未决）；MoE 多模态 mmproj；
bf16 MoE 权重（无原生指令，优先 fp16/int4）。

## 4. 预期收益汇总

| WP | 机制 | 预期 |
|---|---|---|
| WP1 | 数值门禁 | 关闭 R4；后续所有 MoE 改动有回归网 |
| WP2 | Tier A e2e | 首个本地 MoE 基线；替换 R7 传闻 |
| WP3 | Tier B 复验 | 生产候选定案；DP4X 行为级证据 |
| WP4 | VGPR 审计 | 关闭 R1；卡 2/3 放行/锁定 |
| WP5 | 墙位鉴定 | 优化方向定案（只测量） |
| WP6 | 调优配置 | OLMoE/Triton 路径 decode +5% 级 |
| WP7 | GGUF decode 内核 | **e2e decode +20% tok/s（主收益项）** |
| WP8 | GGUF prefill 内核 | TTFT 改善（条件项） |
| WP9 | Tier C 判定 | 30B W8A16 投产决策数据（条件项） |

## 5. 风险登记

| 风险 | 严重度 | 缓解 |
|---|---|---|
| **N1 生产窗口竞争**：4 卡被生产占满，无空闲卡则全线停摆 | 高 | WP0 门禁：无卡即停并报告；下载与 GPU 等待并行 |
| **R1 C++ MoE 内核 VGPR 未审计**（MoE 调查） | 高 | WP4 + WP3 卡 2 对照双证据；保守定案 |
| **R3 MoE+GPTQ-Int4/AWQ-4bit 已知失败** | 高 | D11 排除，不进本图 |
| **N2 HF 下载不稳**（S8 三次中断先例） | 中 | `s9_par_dl.py`（range 并行 + sha256）；断点续传 |
| **N4 35B GGUF = GDN 混合架构**（R10：FLA 内核历史 VGPR=256） | 中高 | dmesg 监控必开；guard 覆盖 Triton；异常即停 |
| **N3 OLMoE 模型较老** | 低 | 仅作流水线验证车，不作生产候选 |
| **N5 MoE + cudagraph 捕获未验证**（S9b 曾遇 GDN FULL 捕获崩） | 中 | 三级降级（FULL_AND_PIECEWISE → PIECEWISE → eager）；基线记录图模式 |
| **N6 WP7 csrc 改动**：构建竞态 / 换装失误 | 中 | build-sync 纪律（S33 须知 3）：备份 md5、竞态原样重跑、禁手改产物 |
| **N7 WP7 改累加序致平局翻转** | 中 | 零翻转硬门（S14b/BM16 流程）；env 默认关 |
| **R8 主机 RAM ~31 GiB**（TP≥3 加载 swap） | 中 | WP9 仅 TP=2；如需 TP=4 沿用 14c 配方 |
| **R6 MTP 正确性未决** | 低 | D11 排除 MoE+MTP |

## 6. 与既有文档的关系

- 执行化：`docs/result/z100-moe-feasibility-investigation.md` §7（最小验证矩阵）
  → 本图 WP0–WP4；§2.5（未覆盖内核清单）→ WP4 审计对象。
- 方法论引用：F24（DP4X 判据/hip shim）、S14b（噪声底/世界判定）、
  S33（build-sync/collect-wedge/贪心流纪律）、11g（exact-shape 战术）。
- 不取代任何现役文档；S30c–S34 稠密侧工作与本图正交（MoE 服务独立端口）。
- 完工后：README"已测试并确认可用 / 已知限制"按实测改写（终局 WP）。
