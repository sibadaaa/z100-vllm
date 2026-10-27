# S18b MTP 首轮真机基线与代价标定（S18-1 / S18-3 前半）— 执行结果

> 归档时间：2026-10-12
> 执行者：Lead（真机臂 + 内核微基准）；臂电池的 TP=2/INT8 扩测另派子 agent（见 §5）
> 对应设计文档：docs/design/z100-18-mtp-optimization.md
> 前一档：[z100-18a-mtp-tooling-and-baseline-result.md](z100-18a-mtp-tooling-and-baseline-result.md)（工具链 + 单测基线 + 停生产）
> 状态：🔶 部分完成（功能与接受率达标；**性能门禁 K4 不通过**，且已定位到根因层级）

## 1. 目标与验收标准

- **K1 功能**：MTP 能起、出词、gfx906 专用 op 命中；
- **K2 正确性**：greedy 下与同 flags 非投机臂逐 token 一致（或差异 ≤ 噪声底）；
- **K3 接受率**：k=2 时 position-0 ≥ 60%、平均接受长度 ≥ 2.0；
- **K4 性能**：decode TPOT 相对同 flags 非投机臂 ≥ 1.3×；
- 附带回答 S18-3 的第一个问题：**`ms/step` 随 verify 宽度 M 的增长形态**。

## 2. 实际改动

| 文件 | 改动摘要 |
|---|---|
| `tools/dcu/s18_fp8_block_mprobe.py` | 新增（158 行）：block-scaled FP8 GEMV 的 `M=1..8` 微基准（真实 N/K、权重字节带宽、M 比值、行一致性、vs fp32 参考的 rel_err） |
| `tools/dcu/s18_one_arm.sh` | 修 2 个真 bug：① 起臂后**立即**用 `pgrep` 判"进程已退出"→ 把刚起来的引擎误判失败并杀掉（首次 A0 因此空转 8 min），加 60 s "进程出现"宽限期；② 探针改为按 `S18_PROBE_MODES` 跑多模式（greedy + prod） |
| `tools/dcu/s18_mtp_probe.py` | 修 2 个 bug：① 指标名匹配漏 `_total` 后缀 → 恒报 `drafts=0`；② `accept_rate=None` 时格式化崩溃（首轮 A1 因此丢掉了精确计数，只有服务日志的窗口值） |

**vllm / csrc 源码零改动**；生产容器 `z100-vllm` 未动（仍停着，回滚配方见 S18a）。

## 3. 执行过程与原始证据

### 3.1 三条臂（FP8 = `Qwen3.8-27B-FP8`，TP=4，eager，`mamba align`，前缀缓存，MBT 2048，maxlen 8192，`HSA_FORCE_FINE_GRAIN_PCIE=1`）

| 臂 | K | 启动 | greedy MAL | greedy accept_rate | prod MAL（temp .7/rep 1.05） | bench Median TPOT | 推出 ms/step（=TPOT×MAL） | 相对 A0 |
|---|---|---|---|---|---|---|---|---|
| A0 | 0 | 143 s | — | — | — | **123.32 ms** | 123.3 | 1.00× |
| A1 | 1 | 173 s | **1.751** | 0.7514（drafts 547） | 1.672 | **136.53 ms** | **239.1** | **1.94×** |
| A2 | 2 | 183 s | **2.294** | 0.6471（drafts 418） | 2.101 | **145.46 ms** | **333.7** | **2.71×** |

- 真实文本吞吐：A0 7.73 tok/s、A1 6.57/6.25 tok/s、A2 6.24/5.84 tok/s（greedy/prod）⇒ **MTP 净负**：k=1 −10.7%、k=2 −15.9%。
- 每步成本拟合：**`ms/step ≈ 123 + ~105×(M−1)`**，即**每多验证一个 token ≈ 再付一个完整 decode step 的钱**（M=2 时 +116 ms，M=3 时 +94 ms）。
- 接受率（服务日志窗口，4 rank 一致）：k=1 `Mean acceptance length` **1.88–1.94**、per-position **0.919–0.944**；k=2 最好样本 **2.77**、per-position **0.933 / 0.833**（早期样本 0.63–0.78）。⇒ **接受率不是问题**，旧的 15.5% 结论不成立。
- prod 采样（`temperature 0.7 / repetition_penalty 1.05`）只把接受率从 0.75 压到 0.67（k=1）、0.65 压到 0.55（k=2）⇒ 与 fork 记录的 rep=1.2 崩溃式下跌不同，生产采样**不是**障碍。

### 3.2 gfx906 MTP 算子命中情况（`serve-fp8-tp4-k1-m8192.log`）

```
(Worker_TP0/1/2/3) QWEN35_MTP_GDN_CUSTOM_HIT q=(1, 2, 4, 128)/torch.float16
                    state=(1380, 12, 128, 128)/torch.float32 accepted=torch.int32
GFX906_MTP_V2 MISS: key_dim=128 value_dim=128 maxq=2 kv_heads=12 heads=4
                    q_s3=1 k_s3=1 v_s3=1 state_s3=1 state_s2=128 a_s1=1 b_s1=1
```

- GDN 的 **v1 MTP 自定义 op 命中**（4 rank 全中）；
- **v2 优化内核 MISS**（fork 记录 v2 把 55.5 → 42.8 µs/层），说明这个 shape 配置没吃到 v2 路径；
- 另有一次 `Triton kernel JIT compilation during inference: _combine_sampled_and_draft_tokens_kernel`（首请求 JIT 尖峰）。

### 3.3 内核层标定：block-scaled FP8 GEMV 的 M 标度（`tools/dcu/s18_fp8_block_mprobe.py`，单卡合成权重，TP=4 分片形状，iters=30）

| shape (N×K) | M=1 | M=2 | M=3 | M=4 | M=6 | M=8 |
|---|---|---|---|---|---|---|
| 8704×5120 | 0.5105 ms / **87 GB/s** | 2.12× | 3.03× | 3.84× | 5.21× | 7.09× |
| 5120×4352 | 0.2513 ms / **89 GB/s** | 2.15× | 3.19× | 4.13× | 5.30× | 7.32× |
| 2048×5120 | 0.1422 ms / **74 GB/s** | 2.10× | 3.11× | 3.95× | 4.18× | 6.04× |
| 5120×1536 | 0.0982 ms / **80 GB/s** | 1.68× | 2.12× | 2.61× | 3.94× | 5.32× |

（`rel_err = 0.00e+00` vs fp32 参考 —— 内核数值精确，逐 M 均如此。）

**两条硬结论**：

1. **耗时随 M 近似线性**（M=2 ≈ 1.7–2.15×，M=4 ≈ 2.6–4.1×）⇒ 与 §3.1 的端到端标度完全一致：**根本没有"权重只读一遍"的红利**；
2. **M=1 时的有效权重带宽只有 74–89 GB/s**（约为峰值带宽的 6–10%）⇒ 该内核**不是带宽受限**，而是每元素转换（256 项 LDS 查表）+ 指令/占用率受限。这也解释了为什么"多验证一个 token"就要多付一份钱：**每行的逐元素工作都是实打实要做的**。

机制与源码对得上（`csrc/rocm/fp8_emu_gemm.cu`）：

- kernel 是 **一 block 一输出列**（`grid(N)`，`:456`/`:542`）——权重 `wp[i]` 在 `m` 循环外只读一次（`:486`），但**激活 `ap[m*K4+i]` 在每个输出列里按 m 重读**（`:498`）；
- `M ≥ 2` 时 host 会先把激活**整体预转成 fp16**（`fp8_emu_widen_a_kernel`，`:980-988`）再进 PRECONV 分支（`:493-497`）⇒ 激活侧流量再翻倍（fp8→fp16 后按 M 行重读）；
- `MS` 桶只影响寄存器/共享内存，**不会多算**（`:490-502` 的 `if (m < M)`）——原设计文档里的"桶式 padding 假设"据此**否定**。

### 3.4 一处需要澄清的探针缺陷（诚实登记）

首轮微基准的"行一致性检查"（row-consistency M=k vs M=1）**无效**：脚本在每个 M 上都重新生成了一组随机权重/激活，所以必然 DIFFERS。已修（同一 shape 只生成一次、按 M 切片），但**修正后的复跑被 GPU 占用（臂电池）推迟**，本档不引用该检查的任何结论。

## 4. 门禁判定

| 门禁 | 判定 | 证据 |
|---|---|---|
| K1 功能（能起/出词/op 命中） | ✅ | 三条臂 ready、greedy/prod 共 960×2 token 正常输出；`QWEN35_MTP_GDN_CUSTOM_HIT` ×4 rank |
| K2 正确性（greedy 逐 token 一致） | ⏳ **未做** | 本轮未做逐 token 对照；须补（S18b2） |
| K3 接受率（p0 ≥60%、MAL ≥2.0） | ✅ | k=1 MAL 1.75–1.94、p0 0.75–0.94；k=2 MAL 2.29、p0/p1 最高 0.93/0.83；prod 模式 MAL 2.10 也达标 |
| K4 性能（≥1.3×） | ❌ **不通过** | k=1 **−10.7%**、k=2 **−15.9%**；ms/step 1.94×/2.71× |
| S18-3 问题（ms/step vs M） | ✅ 已回答 | 端到端 + 内核微基准双证据：**近似线性** |

**因果链已闭合**：`M=1+k` 的 decode GEMV 是"每元素逐行做工作"的指令受限内核（74–89 GB/s）⇒ 每个 verify token 追加约一个完整 step 的成本 ⇒ 接受率再高也无法覆盖。

## 5. 未完成项 / 阻塞与原因

1. **K2 逐 token 对照**未做（需同 engine 内跑一次非投机请求或对比 A0/A1 的 prompt_logprobs；S18b2 补）；
2. **TP=2 / INT8 的判别臂**（4 条）正在由子 agent 执行：FP8 TP=2 k=1/k=0、INT8 TP=2 k=1/k=0 ——目的是回答"换基底能否让 verify 变便宜"（INT8 sdot4 GEMV 每元素只有 1 条 `sdot4`，M 的边际成本应显著低于 FP8 的查表路径）；
3. v2 内核 MISS 的 shape 根因未查（`VLLM_GFX906_GDN_MTP_V2` 的 dispatch 条件）；
4. `QWEN35_MTP_CONV_CUSTOM`（conv MTP op）的命中计数未提取（本轮只核了 GDN 侧）；
5. 微基准的行一致性检查修正后未复跑（GPU 被占）。

## 6. 对后续步骤的影响（关键路径改向）

1. **MTP 的价值不再由接受率决定，而由 `M=1+k` 的代价决定**。当前实现下 **任何 k 都是净负**（k=1 −10.7%、k=2 −15.9%）；
2. 因此 **S18-4.6 升级为项目的关键路径**，目标可量化：
   - **首要目标**：让 `M=2` 的 GEMM 成本 ≤ **1.2×** M=1（当前 1.7–2.15×）⇒ 配合 k=1 的 0.75 接受率才有 ~1.3–1.6× 的 decode 收益；
   - 方向（按可行性）：① 用**列 tile + 激活 LDS 常驻**减少按列重读；② 提高 M=1 时的占用率/ILP（当前只有 74–89 GB/s，说明离吞吐上限很远）；③ 让 `M≥2` 复用**同一份已转换权重**并避免 fp8→fp16 激活重读翻倍；
3. **另立一个可能更大的收益项（与 MTP 正交）**：内核在 M=1 就到不了带宽上限（74–89 GB/s vs 峰值 ~900）⇒ **baseline decode 本身有 5–9× 的理论空间**。这不是 MTP 专属工程，但它是 MTP 收益的前置（MTP 只是把这一步的代价乘以 M）；
4. 建议的决策点（等 §5 的 TP=2/INT8 电池回来）：若 INT8+sdot4 的 M 标度显著优于 FP8 block（例如 M=2 ≤1.3×），则**把 MTP 目标槽改为 INT8/TP=2**，FP8 槽留给不需要 MTP 的场景 —— 这会推翻 2026-10-11 的"目标槽 = FP8"决策，但依据是实测。

## 7. 复现方式

```bash
# 对照臂（K=0）与 MTP 臂（K=1/2），每条臂自动完成 起→探针→基准→杀
docker exec -d z100-s18 bash -lc 'export S18_PROBE_MODES="greedy prod"; export S18_BENCH_MODES=decode; \
  bash /src/tools/dcu/s18_one_arm.sh fp8 4 0 8192'
docker exec -d z100-s18 bash -lc 'export S18_PROBE_MODES="greedy prod"; export S18_BENCH_MODES=decode; \
  bash /src/tools/dcu/s18_one_arm.sh fp8 4 1 8192'
docker exec -d z100-s18 bash -lc 'export S18_PROBE_MODES="greedy prod"; export S18_BENCH_MODES=decode; \
  bash /src/tools/dcu/s18_one_arm.sh fp8 4 2 8192'
# 结果：/mnt/game/s18/summary-fp8-tp4-k{0,1,2}-m8192.md + probe-*.json + bench-*.log

# 内核 M 标度（单卡）
docker exec z100-s18 bash -lc 'cd /src && . /opt/dtk/env.sh >/dev/null 2>&1 && . /src/tools/dcu/env.sh >/dev/null 2>&1 && \
  export PYTHONPATH=/src:/mnt/game/triton-gfx906/install HIP_VISIBLE_DEVICES=0 && \
  python /src/tools/dcu/s18_fp8_block_mprobe.py --shapes 8704:5120,5120:4352,2048:5120,5120:1536 \
    --ms 1,2,3,4,6,8 --iters 30 --check'
```

## 8. 参考

- 设计文档 `docs/design/z100-18-mtp-optimization.md`（§0.1 中期结论已按本档回写）
- 前档 `docs/result/z100-18a-mtp-tooling-and-baseline-result.md`
- 内核源码 `csrc/rocm/fp8_emu_gemm.cu:442-529`（GEMV 主循环）、`:978-989`（M≥2 的激活预宽化）、`csrc/rocm/w8a8_sdot4.cu:62-123`（INT8 对照：同结构，但每元素仅 1 条 `sdot4`）
- 原始产物 `/mnt/game/s18/`（summary/probe/bench/serve 日志、`s18_fp8_block_mprobe.json`）
