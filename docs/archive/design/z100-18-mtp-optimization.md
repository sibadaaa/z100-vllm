# S18：MTP（多 token 预测 / 投机解码）在 Z100 上的适配与优化方案

> 上层文档：[Todolist](../z100-todolist.md) · [性能优化方案](../海光Z100-性能优化方案.md) §5（P3）· [适配记录](../hygon_z100.md) §9
> 前置：[S17](../result/z100-17-p2p-finegrain-e2e-result.md)（RCCL P2P 已开）· [S11k/S11l](../result/z100-11k-sdot4-compile-integration-result.md)（int8 sdot4 3.9×）· [S14d/e](../result/z100-14e-reproducibility-boundary-result.md)（graph 不可复现裁定）
> 状态：⬜ **方案已定稿，用户决定"先不动手"**（本文件只定义"怎么做/怎么判"，不含任何已执行结果）
>
> **用户决策（2026-10-11）**：① 目标槽 = **FP8 block-scaled（当前生产 :8000）优先**；② **暂不动手**，只保留方案（真机基线 S18-1 需要卡窗口，待另行批准）；③ **不引入 GGUF 正对照**，判别实验只用现有 INT8/FP8 两份权重。
>
> **用户决策（2026-10-11 晚，覆盖 ②）**：**停掉生产、全力配合适配** ⇒ 生产已停（4 卡 VRAM 0，回滚配方见 [S18a 归档](result/z100-18a-mtp-tooling-and-baseline-result.md)），实验容器 `z100-s18`（:8001）。

## 0.0 首轮真机中期结论（2026-10-12，**改变关键路径**）

| 项 | A0 对照（K=0） | A1（K=1） | 判读 |
|---|---|---|---|
| decode TPOT（S9 协议） | **123.32 ms** | **136.71 ms** | MTP **慢 10.9%** |
| 真实文本吞吐 | 7.73 tok/s | 6.94 tok/s | 同上 |
| 启动 | 143 s | 194 s | 可接受 |
| `Mean acceptance length`（服务日志，4 rank） | — | **1.73–1.87** | **接受率不是问题**（≈73–87%，远高于旧的 15.5%） |
| gfx906 专用 op | — | `QWEN35_MTP_GDN_CUSTOM_HIT` 命中 | 自定义 op 生效 |
| gfx906 v2 内核 | — | `GFX906_MTP_V2 MISS`（`key_dim=128 value_dim=128 kv_heads=12 heads=4 maxq=2`） | v2 优化内核**没吃到**，退到 v1 |
| 每投机步耗时（=TPOT × 接受长度） | 123 ms | **≈246 ms** | **`M=1+k` 的 verify 代价 ≈ 2×**，无权重复用红利 |

**结论（方案随之改向）**：

1. **旧结论"接受率只有 15.5%"在当前树上不成立** —— 那是 2026-10-07 的旧栈（vendor 0.18.1 + phaseA + bf16）产物。S18-2 的"接受率根因"由此**降级**：接受率已可测且健康，转为常规监控项；
2. **新的阻塞点 = `M=1+k` 的边际代价**：多验证 1 个 token 就要多花**整整一个 decode step** 的钱 ⇒ 当前实现下 MTP 在任何 k 都是净负。这与"权重只读一遍、verify 近乎免费"的预期相反；
3. 因此 **S18-4.6（block-scaled 的 `M=1..8` 调优）与"为什么 M 线性"的诊断升级为关键路径**，S18-3 的开销分解优先级也相应前移。读内核源码（`csrc/rocm/fp8_emu_gemm.cu:442-529`）后，三个候选机制按可能性排序：
   - **H-M1（激活重读放大，最可能）**：`fp8_emu_gemv_block_kernel` 是"**一 block 一输出列**"（`grid(N)`，`:456,542`），权重 `wp[i]` 在 `m` 循环外只读一次（`:486`），但**激活 `ap[m*K4+i]` 在每个输出列 block 里都要按 m 重读**（`:498`）⇒ 激活 HBM/LDS 流量 ∝ M，且总量与权重同量级。等于说"权重摊销了、激活没有"。用微基准 `tools/dcu/s18_fp8_block_mprobe.py`（`M=1..8` × 真实 N/K + 行一致性）可**直接量化**；
   - **H-M2（spec 分支关掉了 gfx906 快路，同样可能）**：`gdn_linear_attn.py:763-769` 一旦出现 spec 掩码就**跳过 packed/combined decode 快路**（那是为本平台专门做的），且本次实测 `GFX906_MTP_V2 MISS` ⇒ 48 层 GDN 在每个 verify 步都退到较慢路径。**即便 GEMM 侧完全摊销，这一项也会吃掉收益**；
   - **H-M3（draft 头与元数据）**：每步多一次 MTP 头前向（1 层权重 ≈ 0.4 GiB/卡）+ 调度/元数据/采样开销。
   注意：**"桶式分派（`MS=1/4/8`）造成额外计算"已被源码否定** —— `:490-502` 的 `if (m < M)` 会跳过超出 M 的行，`MS` 只影响寄存器/共享内存占用。原 H-M1 的表述据此更正。
4. **另两条可立即回收的收益**（不影响上述结论）：v2 内核 MISS 的 shape 修复（fork 实测 55.5→42.8 µs/层）、以及 verify 期的 kernel launch/JIT（首请求出现 `_combine_sampled_and_draft_tokens_kernel` 的 JIT 尖峰）。

## 0.1 背景：原先的收益上限推算（2026-10-11；**其"带宽受限"前提已被本轮实测推翻**）

> 保留原文以记录判断的演变。实测结论见 §0.0 与 §0.2：本平台的 FP8 block-scaled 仿真**不是带宽受限**（M=1 仅 74–89 GB/s），所以"MTP 近乎免费"的推算是错的。

- 当前生产两条槽都**没开**投机解码：`:8000` = Qwen3.8-27B-**FP8**、TP=4、`--enforce-eager`、maxlen 262144、`temperature=0.7 / repetition_penalty=1.05`；回滚槽 = Qwen3.8-27B-**INT8** W8A8。归档基线：INT8 TP=4 eager **121.43 ms/token**（[S13](../result/z100-13-custom-allreduce-result.md)）、FP8 eager 稳态 123–125 ms/token（[S17b](../result/z100-17-p2p-finegrain-e2e-result.md) §8）。
- decode 是"**每步把权重读一遍**"的带宽受限模式：[S11j](../result/z100-11l-sdot4-reacceptance-result.md) 实测 sdot4 之后 GPU busy **82.5 ms/step**、其中 `gemm/int8` **72.5 ms（88%）**。MTP 一次前向验证 `1+k` 个 token，**权重只读一遍** ⇒ 理论上限 ≈ `1 + Σp_i`（k=2、p≈0.8 时 ≈2.6 token/step）。
- 机制在 gfx906 上**已被 fork 证明**（commit 声明，非本机独立复测）：`Qwen3.8-27B-UD-Q4_K_XL GGUF` MTP=2 接受率 **82–90%**、decode 22.66 → **30.58 tok/s**（`1f79673d9`→`fe8a140b2`）；`qwen35moe GGUF` MTP=2 接受长度 ~2.5、**50.5 vs 37.6 tok/s（+35%）**（`8cfc349ac`）。
- **本项目唯一一次本地 MTP 实测是失败的**（2026-10-07，旧 `gfx906_port` + phaseA/vendor 栈 + bf16）：要么引擎崩（`gdn_torch.py:845` 形状不匹配），要么走原生 FLA 时 **接受率仅 15.5%**、147.83 ms/token vs 基线 122.47（净负收益）。⇒ 本方案的第一目的不是"再调参数"，而是**重新基线 + 定位 15.5% 的根因**。

## 0.2 为什么 FP8 槽上 MTP 可能"结构上赢不了"，而 INT8 可能赢（2026-10-12 分析）

把每步成本写成两个可测量：**W = 读一遍权重的 HBM 时间**、**D = 每行激活的 dot 工作量**。则

```
ms/step(M) ≈ max(W, M × D)     每 token 成本 = max(W, M×D) / (1 + Σp_i)
```

- **实测反推 FP8 block-scaled**：M=1 时权重 44.6 MB / 0.51 ms ⇒ **W ≈ 0.09 ms**（按微基准的有效带宽 87 GB/s 折算；若内核能跑到 ~700 GB/s，W ≈ 0.064 ms）；而 M=2 的成本是 2.12×M=1 ⇒ `M×D` 已经是主导项，即 **D ≈ W**。两者同量级不是巧合：fp8 权重 1 B/元素，而 gfx906 的 `v_dot2_f32_f16` 每指令只算 2 个 MAC，**dot2 峰值吞吐折算下来恰好≈ 读 fp8 权重的带宽时间**。
  ⇒ 要 MTP 获胜需 `M×D/1.75 < D`（k=1 取 MAL=1.75），即 `M < 1.75` —— **M≥2 时不可能**。这正解释了实测的 −10.7%（模型预测 −14%/−10.7% 吻合）。
  ⇒ **结论：不改 emulation 的每元素指令数，FP8 槽上的 MTP 在结构上无法回本**；把内核做快只会让 W 与 D 同步变小，比值不变。
- **INT8 sdot4 的差别**：权重同为 1 B/元素，但 `__builtin_amdgcn_sdot4` 一条指令算 **4 个 MAC**（是 dot2 的两倍），且没有 256 项 LDS 查表
  ⇒ `D_int8 ≈ D_fp8 / 2`，于是 `W_int8 > D_int8` 有机会成立 ⇒ **M=2 的额外一行可以藏在权重读取后面**（`max(W,2D)=W`）。
  这正是正在跑的判别臂要验证的：若 INT8 的 `M=2` 比值落到 **≈1.2–1.3×**，MTP 就应该迁到 **INT8(+sdot4)**，FP8 槽留给非 MTP 场景。
- 若 INT8 也不达标（比值仍 ≥1.7×），则 MTP 在本平台的可行路径只剩"换更省指令的数值格式"（如把 block-scaled fp8 在**块内**转 int8 后用 sdot4，但会引入额外量化，须先做精度评估），届时应把 MTP 判为**当前不可行**并把预算转投 baseline 内核提速（那本身有 5–9× 空间，且与 MTP 正交）。

### 0.2.2 **收口（2026-10-12）：K4 通过、K2 不通过 ⇒ MTP 性能可用、正确性不可用**

| 门禁 | 结果 | 归档 |
|---|---|---|
| K1 功能（起/出词/MTP op 命中） | ✅ | [S18b](../result/z100-18b-mtp-first-baselines-and-cost-scaling-result.md) |
| K3 接受率（MAL ≥2.0） | ✅ k=2 MAL **2.246**、k=1 1.781 | S18d |
| **K4 性能（≥1.3×）** | ✅ **k=2 83.92 ms/token（1.47×、10.94 tok/s）**、k=1 99.34 ms（1.24×），同 flags 非投机 123.30 ms | [S18d](../result/z100-18d-mtp-gemm-dispatch-e2e-result.md) |
| **K2 逐 token 一致** | ⛔ **失败**：k=1 与 k=2 均出现 **8.4–12.1 nat** 的自信翻转；两条 MTP 臂彼此只差 near-tie（≤0.44 nat）⇒ **与 S18c 的派发改动无关，是 verify 路径固有** | [S18e](../result/z100-18e-mtp-parity-failure-and-regression-result.md) |

⇒ **纪律判定**：MTP **不得**用于任何依赖 token 流的用途（质量评测、生产输出），与 S14d/e 对 graph 的裁定同类；性能结论（延迟/吞吐）不受影响。**S18-5 上线暂停**，直到 K2 通过；定位方向见 S18e §6（mamba align × spec / 前缀缓存 × spec / V2 speculator 的 logits 行映射 / position）。

**K2 定位进展（2026-10-12 追加，证据见 [S18e](result/z100-18e-mtp-parity-failure-and-regression-result.md) 附录 A–C）**：

1. **不是位置错位**（分叉点与相邻位置的 top-5 Jaccard 峰值都在**同 index**，`zh_ml`/`zh_sky` 的 top-5 集合完全相同、仅排序变化）；
2. **不是数值噪声**：`en_fr` 首次分叉两侧为 12.11 nat vs 0.328（控制臂近乎确定性），匹配位置 `mean|Δlogprob|` 仅 0.054 nat；
3. **前缀缓存与 `mamba-cache-mode` 均被排除**：默认 / `--no-enable-prefix-caching` / `--mamba-cache-mode none --no-enable-prefix-caching` 三对臂给出**逐位相同**的数字（flips 280/469、同样的分叉表）⇒ 分叉是 spec 路径的**确定性**性质，可一键验证任何候选修复；
4. **V1 runner 判别不可用**（`VLLM_USE_V2_MODEL_RUNNER=0` 起不来，HTTP 500）；
5. 正在做**代码级二分**：`VLLM_QWEN35_MTP_GDN_CUSTOM=0` / `VLLM_QWEN35_MTP_CONV_CUSTOM=0`（默认不变）逐个关掉 gfx906 的 spec 专用 op，与既有 `ctl` 对照。
6. **op 级二分结果（2026-10-12）**：**conv MTP op 是两个"自信翻转"的直接来源**（关掉它后 `en_fr` 变 64 token 逐位一致、`en_sea` 分叉从第 2 后移到第 55），GDN MTP op 为次要来源（改变其它 prompt 的分叉点）；两个都关后残余降到 **7/8 prompt 首次分叉 ≤0.48 nat（符合噪声底）+ 1 个 3.4 nat（`zh_sky`，对应 1.79 nat 分布位移）**。**但"关 op"被代价否决**：两 op 都关时吞吐从 ~6–7 掉到 **~1 tok/s（≈6× 慢）**，且 fallback 与 op 数值不等价 ⇒ 必须**修正 op**（而非绕过）。
7. 下一步（bounded）：**按层二分**首分叉步——对同一 prompt 分别跑 k=0/k=1，hook 逐层 hidden state 与 verify 批次的 `positions`/`logits_indices`，找出第一处实质差异所在的层/算子（本项目 S7b 有同类 hook 先例），再决定是修 attention 多 token query、position，还是其余按 M 分支的算子。

### 0.2.1 **已解决（2026-10-12，见 [S18c 归档](../result/z100-18c-gemm-dispatch-m-flat-result.md)）：把 `M≥2` 从 GEMV 改派到 GEMM**

实测（`tools/dcu/s18_fp8_block_mprobe.py` + `s18_dispatch_check.py`）：

| `8704×5120` | M=1 | M=2 | M=4 | M=6 | M=8 |
|---|---|---|---|---|---|
| GEMV（生产当前策略） | 0.5997 ms | 1.1421 | 2.2783 | 3.2591 | 4.4278（**7.38×**） |
| **GEMM（tiled）** | 0.5884 | 0.5882 | 0.5883 | 0.5899 | 0.5900（**0.98×，全程平坦**） |

- op 级（`vllm::fp8_emu_mm_block`，`VLLM_DCU_FP8_GEMV_MAX_M=1`）：M=1 0.6202 / M=2 0.5839 / M=3 0.5831 / M=4 0.5828 ms；
- `rel_err = 0.00e+00`、行一致性 OK；
- 但 **M=1 时 GEMV 在小 N 形状更快**（`5120×1536`：0.1326 vs 0.1733）⇒ 最优策略是 **`M=1` 用 GEMV、`M≥2` 用 GEMM**，即 `_GEMV_MAX_M=1`（已通过 `VLLM_DCU_FP8_GEMV_MAX_M` / `VLLM_DCU_INT8_GEMV_MAX_M` 暴露，默认值仍是旧的 8）；
- **对 A0（非投机）零影响**（其调用全是 M=1）⇒ 123.32 ms/token 仍是有效对照；⇒ **修复前的"结构上赢不了"结论随之撤销**，转为"端到端 A/B 待测（预判 1.5–1.75×）"。绝对吞吐（72–79 GB/s，峰值 6–10%）仍未解决 ⇒ S18-4c 的新内核保留为**独立收益项**，但不再是 MTP 的前置。

## 1. 现状盘点（已核实；含两条既有文档更正）

### 1.1 代码路径

| 环节 | 落点 |
|---|---|
| 配置/校验 | `vllm/config/speculative.py:39-66`（`MTPModelTypes`）、`:916-934`（Qwen3.5 家族 → `qwen3_5_mtp`/`Qwen3_5MoeMTP`，`n_predict=mtp_num_hidden_layers`）、`:1470-1498`、`:1811-1821`（`num_speculative_tokens` 默认/整除校验）、`:1703-1736`（draft TP=target TP） |
| draft 模型 | `vllm/model_executor/models/qwen3_5_mtp.py:82-190`（`fc` 拼接 embedding+目标 hidden；每层一个 `full_attention` decoder layer）、`:400-404`（`mamba_cache_mode="all"` 直接 `NotImplementedError`，必须 `align`）、`:487-502`（`mtp.`→`model.` 重映射）；MoE 版 `:505-508` |
| 权重共享 | `vllm/v1/spec_decode/llm_base_proposer.py:1416-1505`（embedding 共享）、`:1508-1609`（lm_head / `topk_indices_buffer` / `index_share_for_mtp_iteration`）；V2 版 `vllm/v1/worker/gpu/spec_decode/eagle/utils.py:15-107` |
| **默认执行路径（更正 2）** | ROCm 上 Qwen3.5/Qwen3-Next 的 **V2 runner** 是默认：`vllm/v1/worker/gpu/spec_decode/mtp/speculator.py` + `.../autoregressive/speculator.py:197-387`；`vllm/v1/spec_decode/eagle.py` 只是 `llm_base_proposer.py` 的 `pass_hidden_states_to_model=True` 子类，V1 路径是本树回退（`vllm/config/vllm.py:3164-3181`） |
| verify / 接受 | `vllm/v1/worker/gpu_model_runner.py:2510-2557`、`:3090-3160`（`_calc_spec_decode_metadata`）、`:1743`（`num_accepted_tokens`）；rejection `vllm/v1/sample/rejection_sampler.py:410-520`（greedy `:726-781`、随机 `:829-901`） |
| 指标 | `vllm/v1/spec_decode/metrics.py:107-129`：`mean_acceptance_length = 1 + accepted/drafts`，另有 per-position 接受率 |

### 1.2 gfx906 专用钩子（MTP 路上真正会生效的那些）

- conv：`vllm/model_executor/layers/mamba/ops/causal_conv1d.py:1411-1415`（`use_gfx906_fallback`；`VLLM_ROCM_USE_GFX906_TRITON_CAUSAL_CONV1D_UPDATE` 默认 **False**，`vllm/envs.py:159,1424-1428`）、`:1430-1452`（`mtp_custom_eligible` → `ops.causal_conv1d_gfx906_mtp_update`）、`:1454-1498`（`VLLM_QWEN35_MTP_CONV_DEBUG` 打 `CUSTOM_HIT`/`FALLBACK`/`INELIGIBLE`）。
- GDN：`vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py:601-632`（`mtp_gfx906_path` → `ops.fused_sigmoid_gating_delta_rule_gfx906_mtp_update`）、`:633-686`（`VLLM_QWEN35_MTP_GDN_DEBUG`）；C++ 侧 `VLLM_GFX906_GDN_MTP_V2` 默认**开**（`csrc/mamba/gfx906_decode_kernels.cu:2602-2607`）。
- GDN 层：`vllm/model_executor/layers/mamba/gdn_linear_attn.py:763-769`、`:777-794`、`:817-831`、`:934-945`、`:998-1042`。
- 上游层（每 token 都会走的 gfx906 分支）：`layers/activation.py:114-145`（fp16 packed SiLU 兜底）、`layers/layernorm.py:172-211`、`layers/rotary_embedding/{common,mrope}.py`、`layers/utils.py:429-461`、`layers/fused_moe/fused_moe.py:1104-1113`、`v1/attention/ops/triton_unified_attention.py:1025-1416`。
- cudagraph：`vllm/config/compilation.py:1541-1592` 会把捕获桶向上取整到 `num_speculative_tokens+1`；Qwen3.5 GGUF 在 gfx906 上被强制 eager/PIECEWISE（`vllm/config/vllm.py:1983-2012`）。

### 1.3 **更正 1**：bf16 + MTP 的"必报错"结论不成立

`docs/hygon_z100.md:91` 与 `docs/design/z100-10-phase-b-features.md:22` 写的是"MTP/GDN 内核走 `VLLM_DISPATCH_HALF_TYPES`（仅 fp16），bf16 会明确报 `not implemented for BFloat16`"。**实际代码**：

- `csrc/dispatch_utils.h:22-27`：`VLLM_DISPATCH_HALF_TYPES` = `Half` **+** `BFloat16`；
- `csrc/mamba/gfx906_decode_kernels.cu:1893-1901`、`:2527-2539` 的 `TORCH_CHECK` 都接受 bf16，且该文件**无任何 `__half`/`half2` 专用内建**（数学全在 `float`）；
- `tests/kernels/mamba/test_causal_conv1d.py:142-196` 已把 bf16 作为通过用例参数化。

真正的 bf16 门在 Python 侧：`vllm/platforms/rocm.py:1300-1305` + `:275-296`（gfx906 默认 `[fp16,fp32]`，仅 `is_hygon_dtk() and VLLM_DCU_BF16` 才插入 bf16）。⇒ 该条按"**待否证/待落地为测试**"处理（S18-4.2），不作为 MTP 的阻塞项。

### 1.4 **更正 2**：S10 把 MTP 理解为"`gfx906_decode_kernels.cu` 的 MTP update + async scheduling 开关"

实际是一个完整链路（§1.1），`async scheduling` 只与其一（`speculative.py:647-655`）相关。S10 的"先用 fp16 验证出词与接受率"因此**低估了工作面**，本方案把它独立成 S18。

### 1.5 已存在的、与 MTP 直接冲突/缺失的缺口（"适配"的实际工作量）

| # | 缺口 | 证据 | 后果 |
|---|---|---|---|
| G-a | 只要出现 spec 掩码，gfx906 **packed decode 快路被跳过** | `gdn_linear_attn.py:763-769` | 开 MTP 就丢掉已落地的 decode 快路 |
| G-b | `VLLM_QWEN35_SHARED_GDN_CORE` 默认关，注释明写"shared core lacks MTP verification on gfx906" | `qwen3_5.py:316-318` | 只能用 specialized core；shared core 路线被 MTP 挡住 |
| G-c | `fused_sigmoid_gating_delta_rule_gfx906_mtp_update` **没有直接单测** | `tests/kernels/mamba/test_fused_sigmoid_gating.py` 只覆盖非 MTP；`tests/model_executor/test_qwen3_5_gdn.py:28` 只断言"被选中" | 数值错误只能靠端到端发现 |
| G-d | 算子存在性探针漏掉 GDN MTP op | `tools/dcu/s8_path_probe.py:180-190` | 门禁漏检 |
| G-e | `docs/result/z100-11c-gdn-mtp-result.md` 计划了但**不存在** | `docs/result/` 无 11c | P3 从未闭环 |

### 1.6 量化契约（决定 MTP 的成本结构）

两份权重都带 **22 个 `mtp.*` 张量**（`mtp.fc` 无 scale ⇒ 非量化）：

| 槽 | 权重 | 激活 | 对 MTP 的含义 |
|---|---|---|---|
| INT8 `Qwen3.8-27B-Channel-INT8-w8a8` | int8 **per-channel 静态**（`compression_config.config_groups`） | int8 **动态 per-token** | 落 sdot4 GEMV（M≤8 覆盖 `1+k`）⇒ **带宽受限，MTP ROI 大** |
| FP8 `Qwen3.8-27B-FP8` | **block-scaled 128×128**（`weight_block_size:[128,128]`） | e4m3 dynamic | **已核实：block-scaled 仿真内核在树里**（S11f `dac6861d5` 的 `Gfx906Fp8BlockScaledMMLinearKernel` + `fp8_emu_gemv_block/gemm_block`，`csrc/rocm/fp8_emu_gemm.cu:68` 的 `kMsMax=8`：**GEMV 覆盖 M≤8**，M>8 走 GEMM）⇒ MTP 的 `M=1+k` verify **已在覆盖范围内**（"block-scaled 走 36× Triton"是 S8b 的旧状态，已被 S11f 取代）；ROI 从"预判差"改为**"可用，待实测"** |
| GGUF（fork 验证过的那条） | K-quant + MMVQ | — | fork 已在此拿到 82–90% |

两份 checkpoint 的 ignore 列表都包含 `mtp.fc`、`mtp.pre_fc_norm*`、全部 norm、`lm_head`、`embed_tokens`；**不含** `mtp.layers.0.{self_attn,mlp}.*` ⇒ MTP 层的 attention/MLP 是量化的（draft 头被量化是与接受率直接相关的嫌疑项）。

> **决策影响（2026-10-11 二次修正）**：S11f 已落地 block-scaled FP8 仿真（原审计漏读，见上表修正），因此 S18-4.6 **不是"新写内核"**，而是"**验证并按 `M=1+k` 调优已有的 block-scaled GEMV**"：当前分派是桶式（`M≤1→MS=1`、`M≤4→MS=4`、否则 `MS=8`），`k=1,2,3,5` 会分别落进 `MS=4/4/4/8` 的模板，**M=5,6 有 (8−M)/8 的 padding 浪费**。这一步因此变成低风险的调优项，而不是 FP8 槽 MTP 的阻塞前置。

### 1.7 重复劳动核查（按 AGENTS.md §1）

`git log --all --grep=mtp -i` 的全部提交**都是 HEAD 的祖先**（逐 hash 用 `merge-base --is-ancestor` 验证），`gfx906/main..HEAD` 无遗漏 ⇒ fork 的 MTP 修复/优化（`da5f8ce75` Qwen3.8 MTP 正确性、`1e23b7af1` gfx906 多 token attention、`8cfc349ac` draft 加载 + GDN spec-decode 路径、`a87749152`/`fe8a140b2` GDN MTP 状态更新 334→42.8 µs/层、`9e1c1a17a` 多向量 MMVQ）**已经在树里**，不应重写。**但**：这些结论全部在 **GGUF** 上取得，且**没有任何当前启动脚本开 MTP**（`serve_27b_tp4.sh:126`、`tools/dcu/s11f_e2e_fp8.sh` 均无 spec 参数）⇒ 本方案的增量是"**在新栈 + 我们的量化 + 我们的采样参数下重新基线、定位接受率、补 gfx906 快路**"，不是重复 fork 的工作。

## 2. 目标与验收（KPI）

| 编号 | 判据 |
|---|---|
| K1 功能 | `--speculative-config '{"model":<ckpt>,"method":"mtp","num_speculative_tokens":k}'` 在目标槽能起、能出连贯文本；`VLLM_QWEN35_MTP_{CONV,GDN}_DEBUG` 显示 **CUSTOM_HIT 命中、FALLBACK/INELIGIBLE 为 0** |
| K2 正确性 | eager、固定批组合（[S14f](../result/z100-14d-determinism-result.md) 的批组合效应）下，greedy 与同 flags 非投机臂**逐 token 一致**；若不一致，差异必须 ≤ 本项目噪声底（0.1–0.6 nat）且给出归因 |
| K3 接受率 | k=2 时 **position-0 ≥ 60%、平均接受长度 ≥ 2.0**（参照 fork GGUF 的 72–90% / ~2.5） |
| K4 性能 | decode TPOT 相对同 flags 非投机臂 **≥1.3×**；TTFT / conc8 劣化 ≤5%；`oom_kill` 零增量 |
| K5 可回滚 | 不传 spec 参数即回退；**不改任何默认值**；涉及 `.so` 时记新旧 md5 与回滚件 |

## 3. 分步执行（每步：任务书 → 门禁 → `docs/result/` 归档 → Lead 复核）

### S18-0 前置与口径固化（0.5 d，GPU-免）

- 交付 `tools/dcu/s18_mtp_arm.sh`（唯一入口，结构照 `s11f_e2e_fp8.sh`：显式 `HIP_VISIBLE_DEVICES`、`HSA_FORCE_FINE_GRAIN_PCIE=1`、`VLLM_CACHE_ROOT`、`--dtype float16`、`--enforce-eager`、`--mamba-cache-mode align`、`SPEC=mtp:k`、三个 MTP DEBUG 全开）与 `tools/dcu/s18_mtp_probe.py`（读 `/metrics` 的 `spec_decode_num_{drafts,draft_tokens,accepted_tokens}{,_per_pos}_total`，按 prompt 打表，口径对齐旧归档 `gfx906_port/bench/probe_real.py:89-94`）。
- GPU-免单测：`tests/kernels/mamba/test_causal_conv1d.py -k gfx906_mtp`、`tests/model_executor/test_qwen3_5_gdn.py -k mtp`、`tests/v1/spec_decode/test_mtp.py`；**新增** `fused_sigmoid_gating_delta_rule_gfx906_mtp_update` 的直接单测（vs 非 MTP 参考的数值对齐 + bf16 派发），并把该 op 加进 `tools/dcu/s8_path_probe.py`。
- **门禁 G0**：单测集合全绿（含新增）；探针能复现旧归档的指标口径。

### S18-1 重新基线（1–2 d，**需要卡窗口**）

- 目标槽 = **FP8**：臂 `A0` 非投机（= 现在的 `:8000` 配方：Qwen3.8-27B-FP8 / TP=4 / fp16 / `VLLM_DCU_USE_FP8_EMU=1` / eager / `mamba align` / 前缀缓存 / `MBT=2048` / maxlen 先 8192 再放大）→ `A1` MTP k=1 → `A2` MTP k=2。**基线口径取 FP8 槽自己的数**（decode 稳态 123–125 ms/token、8K TTFT ~31.9 s @P2P，[S17b](../result/z100-17-p2p-finegrain-e2e-result.md) §8），不要用 INT8 的 121.43。
- **替代正对照（因已决定不用 GGUF）**：用**同架构、仅量化不同的 INT8 W8A8 权重**（`/mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8`，本机已有）作为**诊断臂 `D1`**，只回答一个问题——"低接受率是栈问题还是量化问题"。它比 GGUF 便宜（权重现成），且同样能给出 A/B 判别；若 `D1` 与 FP8 同样低、且栈侧证据（S18-2 的 E1/E2）也指向运行时，则 GGUF 是否补测不影响结论。
- 栈内自证（不依赖任何第二份权重）：`E2` draft 头 teacher-forcing 命中率、`E4` 采样参数曲线。
- **门禁 G1**：K1 达成（含 DEBUG 证据）；K2 达成；K3 有数据（低也记录）。不满足则**停在 S18-2**，不进 S18-3。

### S18-2 接受率根因定位（2–4 d，成败点）

一次只动一个变量，全部给原始数字与反证据：

| 实验 | 做法 | 判读 |
|---|---|---|
| E1 位置依赖 | 接受率 vs 序列位置（前 32 / 中段 / 长程） | 一开始就低 ⇒ draft 头本身；运行中崩 ⇒ 状态更新（`num_accepted_tokens` 偏移、`transposed state`，参见 `7026983a8`） |
| E2 draft 头孤立对拍 | teacher forcing：喂目标 hidden + 真值 token，测 draft top-1 命中目标 top-1 的比例 | 分离"头/权重"与"运行时/状态" |
| E3 量化敏感性 | INT8 / FP8 / GGUF /（若可行）draft 层去量化成 fp16；逐个 tensor 抽查 `mtp.*` 权重与 scale 是否按 checkpoint 正确装载（per-channel 轴、`weight_scale` vs `weight_scale_inv`、lm_head 共享口径 `llm_base_proposer.py:1546-1605`） | 量化是否为主因 |
| E4 采样参数 | greedy vs 生产 `temp 0.7 / rep 1.05` vs `rep 1.2`（fork `5749599db`：84% → 5%） | 决定生产能否开 MTP |
| E5 k 与元数据 | k=1/k=2 的 per-pos 曲线；`index_share_for_mtp_iteration`；chunked-prefill 的 `-1` 掩码与 `spec_sequence_masks` 推导（`gpu_model_runner.py:2528-2531`、`v1/attention/backends/gdn_attn.py:172-192`） | 调度/元数据缺陷 |

- **门禁 G2**：给出**单一主因 + 反证据**。接受长度 ≥2.0 ⇒ 进 S18-3；若 <1.3 **且** P0（GGUF）达标 ⇒ 判"该量化 checkpoint 不适合 MTP"，写**否证归档**并停止（止损）。

### S18-3 开销分解与 k/TP/批 矩阵（1–2 d）

- 成本拆解（profiler，方法复用 `tools/dcu/*kprof*` 与 [S11b](../result/z100-11b-perf-profiling-result.md)）：draft 头 / verify GEMV(GEMM) / GDN+conv MTP op / 采样与 rejection / TP 通信；输出 `ms/step(k)` 与 `tokens/step(k)` 表求最优 k。
- 矩阵：`k∈{1,2,3,5} × TP∈{2,4}`，eager 为主；graph 只作"**延迟天花板**"参考臂，**不得**用于质量口径（[S14d/e](../result/z100-14e-reproducibility-boundary-result.md)）。
- 重点验证一个假设：**MTP 会改变 TP=2/TP=4 的优劣** —— 每步通信 payload ×(1+k)，而 [S16/S17](../result/z100-17-p2p-finegrain-e2e-result.md) 已证 ≥256 KiB 时 P2P 有 8–14×，MTP 恰好把 decode 的小消息推过阈值。
- **门禁 G3**：`k*`、`TP*` 有数据支撑；K4 达成或给出"未达成 + 原因"。

### S18-4 gfx906 专项适配（3–6 d，"优化适配"的主体）

1. **MTP 下的 GDN/conv 快路恢复**（对应 G-a/G-b）：让 spec-token 布局也吃到 packed/fused decode —— 扩展 `gdn_linear_attn.py:763-769` 的抑制条件与 `qwen3_5.py:710-732`（`_packed_decode_writes_full_core_output` 要求 spec 掩码为 None），并逐条验证 `VLLM_QWEN35_SHARED_GDN_CORE=1` 在 MTP 下的正确性（fork 在 `causal_conv1d`/`fused_sigmoid_gating` 做过同类工作，但没有覆盖 packed 核心）。做不动就给出"为何不能"的否证。
2. **bf16 派发**（对应 §1.3 更正）：以单测钉死三个 MTP op 的 bf16 行为；在 `VLLM_DCU_BF16` 门控下跑一条 bf16 MTP 臂做对照（不改默认值）。
3. **verify 侧 GEMM/GEMV**：确认 `M=1+k ≤ 8` 落 sdot4 GEMV（`kernels/linear/scaled_mm/gfx906_sdot4.py:175 can_implement`），避免 `M=1→64 padding`；FP8 block-scaled 槽则量化 M 增长对 36× 仿真路径的代价，据此决定"是否只在 INT8 槽开 MTP"。
4. **draft 头小算子**：`mtp.fc`（未量化，gather_output）与 MTP 层量化线性层在 M≤6 的开销；必要时走 gfx906 dense GEMV（`layers/utils.py:429-461`）。
5. **调度交互**：`mamba align` + 前缀缓存 + chunked prefill 与 MTP 的 token 预算/block 对齐（生产 `MBT=2048` 是 TTFT 调优值，MTP 会改变调度算术）。
6. **block-scaled FP8 的 `M=1+k` 调优（目标槽权重最高，但已不是阻塞项）**：block-scaled 仿真**已在树里**（`Gfx906Fp8BlockScaledMMLinearKernel`、`fp8_emu_gemv_block/gemm_block`，GEMV 覆盖 `M≤8`）。要做的是：① 微基准扫 `M=1..8 × 真实 N/K`，量出 `M=1+k` 相对 `M=1` 的**实际放大倍数**（若接近 1.0 ⇒ MTP 在 FP8 槽天然划算）；② 现有桶式分派（`MS=1/4/8`）在 `M=2,3` 与 `M=5,6` 有 padding，必要时加 `MS=2/6` 模板或改循环；③ 顺带确认 MTP 层（`mtp.layers.0`）与 `mtp.fc` 走的是哪条 kernel（`mtp.fc` 在 `modules_to_not_convert` 里 ⇒ 非量化）。
- **门禁 G4**：每项"生效 / 未生效 / 否证"逐条表 + 端到端 A/B（n≥4、给区间、同 flags、逐字登记编译模式）；第 6 项须给 `M=1..8` 的 per-shape 数值（rel_err ≤1e-2 量级）与带宽/耗时对照。

### S18-5 上线与固化（1 d）

- 生产封装：`S18_MTP`/`SPEC` 开关进 `serve` 脚本（**默认关**）、AOT 预热纪律（[S11m](../result/z100-11m-aot-restart-reproducibility-result.md)）、`/metrics` 接受率告警、回滚演练。
- 文档：更新 `docs/hygon_z100.md`（含 §1.3 更正）、README 支持矩阵、[生产部署配方](../海光Z100-生产部署配方.md)、`z100-todolist.md`（补 S18 行、关闭 11c 缺口）。
- **门禁 G5**：生产配方同构跑 ≥30 min 稳定、`oom_kill` 零增量、回滚一次成功、归档齐。

## 4. 资源与纪律（本机现实约束）

1. **生产槽 `:8000` 占满 4 张卡**（FP8/TP=4，四卡各 ~28.6 GiB）+ 宿主 **RAM 25/31 GB 已用**、swap 25 GB。⇒ 真机 MTP 测量**必须有窗口**：经用户批准的窗口内以影子容器替换 `:8000`，或只做"同构替换 + 立即回滚"。**禁止**在生产占用时抢卡（硬约束 25）。
2. 实验容器 `:8001+`，显式 `HIP_VISIBLE_DEVICES`，同一时刻只让一个 agent 占卡；构建产物落 `/mnt/game`；不动生产容器 / DTK 库 / `z100-build*`；**禁止**执行自编译 HIP 可执行文件（硬约束 1）。
3. dtype：生产一律 fp16；bf16 只在 S18-4.2 做受控对照。
4. 不改默认值、不碰 `_ON_GFX9`、所有 DTK 分支保持 `is_hygon_dtk()` 可回退（硬约束 6/22）。
5. 归档强制 `docs/result/z100-18-*.md`；每步先过门禁再进下一步。

## 5. 风险与止损点

| 风险 | 判据 | 处置 |
|---|---|---|
| 接受率确实只有 ~15%（头/量化固有） | 诊断臂 D1（INT8）达标而 FP8 <1.3；或 E2 显示 draft 头本身命中率就低 | 判"该量化不适合 MTP"；收工或只保留非投机路线 |
| FP8 槽的 `M=1+k` 放大倍数过高（桶式 padding / 计算受限） | S18-4.6 的 `M=1..8` 微基准；`tokens/step` vs `ms/step` 曲线 | 加 `MS=2/6` 模板或改循环；仍不行就把 MTP 结论限定为"某 k 值以下可用" |
| graph 不可复现与 MTP 冲突 | 只能 eager 时收益 <1.3× | 只用 graph 报"延迟天花板"，不上生产 |
| 宿主 OOM（TP=4 + MTP 的状态/KV） | `oom_kill` Δ≠0 | 降 k / 降 maxlen / 退 TP=2 |
| 生产采样 `rep=1.05` 压低接受率 | E4 曲线 | 生产保持不开 MTP，或先调 penalty |
| 与 fork 工作重复 | 每步先做 ancestry/脚本 grep 核查 | 复用 fork 已落地提交，只做增量 |

## 6. 工作量与待确认

**工作量**：8–14 人日（S18-0 0.5 / S18-1 1–2 / S18-2 2–4 / S18-3 1–2 / S18-4 3–6 / S18-5 1）；**止损点在 G2**。

**已定决策（2026-10-11）**：目标槽 = FP8 block-scaled（`:8000` 同配方）；**先不动手**；不引入 GGUF 正对照（判别改用 INT8 诊断臂 D1 + 栈内 E2/E4）。

**动工前仍需确认**：

1. **卡窗口**：S18-1 起的一切真机测量都需要替换/停用 `:8000`（它占满 4 卡）。在此之前只做 S18-0（脚本 + 单测）与 S18-4.2（bf16 派发单测）这两块 GPU-免工作——**是否现在就做这两块？**
2. **诊断臂 D1 是否保留**：INT8 权重现成、约多 0.5–1 天；不保留则低接受率时的"栈 vs 量化"判别只能靠 E2/E4 的间接证据。
3. **S18-4.6 不再是前置**（2026-10-11 二次修正）：block-scaled 仿真已在树里，该项降级为可选调优。⇒ 现在的顺序是"先测 MTP ROI，再决定是否调 `MS` 模板"。

## 7. 参考

- [性能优化方案](../海光Z100-性能优化方案.md) §5（P3 GDN/MTP）、§8（阶段表）
- [阶段 B 设计文档](z100-10-phase-b-features.md)（本方案细化并**更正**其 bf16 与"工作范围"两处结论）
- [结果归档索引](../result/README.md)
- 旧实测：`/home/lyh/server/gfx906_port/对比测试报告-v1.md`（2026-10-07，MTP 15.5% / 崩溃）
