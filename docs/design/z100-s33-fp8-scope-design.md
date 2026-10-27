# S33 设计图：fp8 约束内对齐天花板（2026-10-26）

> **决策输入**：用户确认权重保持 fp8（nvfp4 列为技术储备），在 fp8 内把
> 每 token 墙钟做下去。本图是 `docs/design/z100-s32-fp8-scope-plan.md`
> （含 2026-10-26 自审计勘误①–④）的决策定案版；施工步骤见配套施工图
> `docs/design/z100-s33-fp8-scope-construction.md`。
> **基线**：s30c-full 现役臂 **step p50 = 29.48 ms**（N=3：29.24/29.48/30.05），
> 热 TTFT 701–746 ms，热 26.8–27.3 t/s（`docs/result/z100-s30c-final.md`）。

## 1. 目标与口径

| 维度 | 口径 | 现值 | 目标 |
|---|---|---|---|
| 速度·每 token | step p50（=ms/tok，1 tok/step） | 29.48 | **确定性项 ~21–23.5；全部落地 ~20–21** |
| 速度·单流 | out_tok_per_s（热） | 26.8–27.3 | ~30–33 |
| 交互 | 热 TTFT（7469 doc） | 701–746 ms | ±3% |
| 数值 | 贪心流 vs 基线重启世界 | 落世界内 | 保持落世界内；改累加序项过零翻转硬门 |
| 精度 | fp8 权重 | 5.58 GB/round | 不变（nvfp4 仅储备） |

强制口径不变：7469 doc / 96 tok / 3 reps、双口径、N≥2 重启、每 WP 独立回滚、
改模型输出的项必须 eval + 用户裁决。

## 2. 墙位（已核实的代码/测量事实）

- **fp8 线性**：256 calls/round（16 qkv + 16 o + 48 GDN-in + 48 GDN-out +
  64 gateup + 64 down），5.58 GB/round，12.49 ms = **447 GB/s**。分形状
  （微测，m1 rows=4 口径）：gateup 552 / down 464 / qkv 386 / o 329 GB/s。
  M=1 每权重字节恰读一次，L2 无复用 → @650 GB/s 规划天花板 = 8.6 ms 硬墙。
- **核结构**：`fp8_emu_gemv_stream_block_kernel`（v1）+ `..._m1_kernel`
  （WP2-N，默认关）；kStreamSlice=1024 / kStreamChunk=16 / 每 slice 2 次
  barrier；派发在 `run_block_w8a16` 宏（fp8_emu_gemm.cu）。
- **通信**：nccl 128 AR + 1 AG + AR_LL ≈ 6.9–7.4 ms/round（窗口去偏后），
  载荷 10 KB（hidden 5120 fp16），延迟/协议主导。**one-shot 基建已存在**：
  `csrc/custom_all_reduce.cuh:276` 接受 `VLLM_CUSTOM_ALLREDUCE_ALGO=oneshot`；
  `custom_all_reduce.py:270-300` 有 gfx906 opt-in（`VLLM_DCU_CUSTOM_AR=1`），
  fp16 有核（bf16 无核走 RCCL）。历史阻塞点（comm report）：细粒度分配失败
  （uncached alloc）+ S13 实测中性 → 本图按"复活+修复+实测"立项，不是新写。
- **triton 小核**：25 个 3.26 ms/round，全部延迟受限；`fused_add_rms_norm`
  系 1.63 ms（48+16 c/r）、silu-mul 0.31、add_clone-quant 0.24。
- **attention**：`kernel_unified_attention_3d` grid [1,1,128]，16 c/r ×
  79.7 µs；KV 1 KB/token/层/rank → 7.5k ctx 124 MB → **97 GB/s**（网格欠填）。
  `VLLM_GFX906_ATTN_DECODE_SEGMENTS` 钩子已存在（triton_unified_attention.py:129，
  默认 0）；`reduce_segments` 核可复用。
- **lm_head**：bf16 gemv [62080,1,1]，636 MB，907 µs = **700 GB/s，已在天花板**
  （d2d 实测 771，读主导探针 621）。fp8 内唯一空间是改精度（−0.45 ms，裁决项）。
- **GDN**：m1_native 48×12.9 µs（state 75.5 MB r+w @122 GB/s）+ conv 345 µs。
- **空隙**：wall 29.48 − GPU busy 27.4 ≈ **2 ms 未分账**（核间空隙/发射尾巴，
  勘误④），先测后治。

## 3. 设计决策

### D1（WP0）：空隙分账——只测量
ITER_DETAILS + trace 全表和 vs wall，把 ~2 ms 拆成"核间空隙 / 发射尾巴 /
其他"，产出 `.s33/gap_accounts.json`。不改任何代码；结果决定 §WP3 融合的
优先级权重。

### D2（WP1）：whole-A M=1 GEMV（数值位同，主力项）
- 新核 `fp8_emu_gemv_wholea_m1_kernel<RW,T>`：整行激活 + 折算 scale 一次
  stage 进 LDS，K 循环**零 barrier**，PF 深预取加深；rows 几何沿用
  `stream_block_select`（ROWS override 有效）。
- 适用形状：qkv/o/GDN-in/GDN-out/gateup（A 行 10 KB → 4–5 block/CU）。
  **down（K=17408 → 34 KB → 每 CU 1 block，占用塌缩）排除在本核外**：留在
  现路径 + 冲突审计，或按需配 K 切分（方案 B，见 D6）。
- **数值性质**：slice 序 / lane chunk 序 / acc 累加序与现产核一致 → 与生产
  **位同**（微测 max|Δ|=0 报表确认），零翻转门免检。
- env：`VLLM_DCU_FP8_GEMV_WA`（默认 0）。
- WP-f3 bank 冲突审计（ds_read 4-way 嫌疑 + pad）并入本 WP 交付。

### D3（WP2）：gfx906 one-shot AR——复活+修复+实测（不是新写）
- 打开 `VLLM_DCU_CUSTOM_AR=1` + `VLLM_CUSTOM_ALLREDUCE_ALGO=oneshot`，
  根治历史阻塞点（细粒度分配失败），验证：fp16 10 KB 路径、piecewise
  graph capture 兼容、size 门（≤64 KiB one-shot，prefill 大载荷保持 RCCL，
  TTFT 不回退）。
- **收益修正（勘误①）**：skew 等待不消失 → 目标 −2.5~4 ms，验收门 −2 ms。
- bf16 一律保持 RCCL（无核），行为沿用现有 gate。
- 逐 run 确定（固定 rank 序）＝消除一个 run 间 skew 源；logprob ULP 偏移
  用贪心 diff 板交付，不落基线世界则按 WP-C 裁决。

### D4（WP3）：融合层（依赖 WP2 的 AR 核）
- **AR 核内 norm+residual+quant**（勘误③的正确融合点）：one-shot AR 归约
  完成即得完整行，顺手做 norm——替代 48+16 c/r 的独立 norm 核；数学逐条
  对齐现 triton 核（fp32 域 max|Δ| 报表）。
- **gateup silu-mul epilogue**：行配对 (i, i+4352) 逐元素，成立。
- **GDN CFG 扩展**（可选）：m1_native CFG 表加档，纯查表位同。
- 核数下降同时兑现勘误④的部分空隙。

### D5（WP4）：attention decode 分段（+按需 split-K GEMV）
- `VLLM_GFX906_ATTN_DECODE_SEGMENTS`（钩子已在）按 seq 自适应分段，
  段间归约复用 `reduce_segments`；7.5k ctx 目标 97→300+ GB/s。
- **改累加序 → 贪心流零翻转硬门**；32k/128k 档加测（长 ctx 收益放大）。
- split-K 小 N GEMV 仅当 WP1 后 qkv/o 仍 <450 GB/s 时启动（同门）。

### D6（WP5）：fp8 内的精度扩展（eval + 用户裁决，不默认）
- **lm_head fp8**（W8A16 路线现成）：−0.45 ms；改 top-1 logprob → 贪心
  diff 板 + top5 偏移报表 + eval，单独裁决。
- **KV fp8 + 核内 LUT 反量化**：前测 ITL +41% 的根因假设 = 反量化未融合；
  先微测证伪/证实。短文 −0.15~0.25，长文放大。

### D7：明确不做（防重开）
L2 tiling/亲和（M=1 无复用）；fp8 scale 粒度/校准类（速度中性，价值只在
换质量预算）；SSM state fp16（量小风险真，除非最后差一步且用户点头）；
spec 重开（dflash 30.3 > 纯 29.48，verify 已用 v2 快核，fp8 内不划算）。

## 4. 预期收益汇总

| WP | 机制 | 预期 |
|---|---|---|
| WP1 | whole-A（5 形状）+ 冲突审计 | 线性 12.49→9.5–10.5，e2e **−1.5~−2.5** |
| WP2 | one-shot AR | e2e **−2.5~−4**（门 −2） |
| WP3 | AR 核内 norm + gateup epilogue（+CFG） | e2e **−0.8~−1.4** |
| WP4 | split-KV（+按需 split-K） | e2e **−0.3~−0.9** |
| WP5 | lm_head/KV fp8（裁决） | e2e **−0.5~−0.7** |
| 合计 | | **29.48 → 20–23.5**（按裁决项与否） |

## 5. 风险登记

| 风险 | 缓解 |
|---|---|
| whole-A 占用/寄存器误调 | down 已排除；qkv/o/gateup 先微测定档再上服务 |
| one-shot AR 挂起（peer 失速/自旋） | size 门 + watchdog + 长跑 30 min；env 一行回 RCCL |
| AR 数值序变化致平局翻转 | 贪心 diff 板交付，落世界外则 WP-C 裁决 |
| norm 融合数学偏差 | fp32 域 max|Δ| 报表逐条对齐现 triton 核 |
| split-K/split-KV 改累加序 | 零翻转硬门（O4/BM16 流程） |
| 构建竞态（hipify/ninja） | build-sync 重跑即恢复，禁手改产物 |
| lm_head/KV fp8 改输出分布 | eval + 用户裁决，不默认 |

## 6. 与既有文档的关系

- 取代：`z100-s32-fp8-scope-plan.md` 的执行版（含勘误①–④）；S31 的
  WP1–4 并入本图 WP1/WP4，S32 的 A1 并入 WP2、B2/B3 并入 WP5。
- 留档不变：`z100-s31-ceiling-gap-plan.md`（天花板账本）、
  `z100-s32-unconstrained-plan.md`（nvfp4/spec 路线储备）。
