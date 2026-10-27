# S31 优化计划：对齐天花板（2026-10-26）

> 基线 = **s30c-full 现役臂**（m1 GDN + capture[1,2,4]）：**step p50 = 29.48 ms**
> （N=3：29.24/29.48/30.05），热 TTFT 701–746 ms，热 26.8–27.3 t/s。
> 证据：`docs/result/z100-s30c-final.md`、`.s30c/kdiff-s29ns-s30cm1b.csv`
> （rank0，20 rounds，s30cm1b trace）、d2d 天花板今日实测。
> 口径沿用：7469 doc / 96 tok / 3 reps、双口径（step + TTFT）、N≥2 重启、
> 贪心流零翻转硬门、每 WP 独立回滚（env > 脚本 > `.so`）。

## 1. 与天花板的对比（每 decode round，按 kdiff 分类聚合）

设备天花板（本机今日实测）：d2d copy 512 MB → **771 GB/s**（读写各半）；
读主导的 GEMV 流（P0 探针几何）实测 621 GB/s；lm_head 现状已在 700 GB/s。
计划取 **650 GB/s** 作为读主导 kernel 的保守规划天花板，620 GB/s 作可达档。

| 块 | 现状 ms/round | 字节量/round | 有效带宽 | floor @天花板 | gap | 判读 |
|---|---|---|---|---|---|---|
| fp8 线性（256 calls） | **12.49** | 5.58 GB（qkv 168M + o 126M + GDN in 629M + GDN out 378M + gateup 2854M + down 1426M） | **447 GB/s** | 8.6 @650 | **+3.9** | 最大杠杆；gateup 已 552（85% 天花板），qkv/o/GDN-out 拖后 |
| nccl（TP4，2/层） | 7.44* | — | — | 结构性 | 0 | comm report 全杠杆已关闭（D4/WP4 留档）；*取 s29ns 同期窗口值去偏 |
| triton 逐元素/归约（25 kernels） | **3.26** | 每核 <100 KB | 延迟受限 | ~2.0（并核/入 epilogue） | **+1.3** | 22 µs 级小核，发射/延迟主导，fused_add_rms_norm×48 为大头 |
| attention（16 层，7.5k ctx） | **1.61** | KV 121 MB（1 KB/tok/层） | ~95 GB/s | 0.35（split-KV） | **+1.2** | grid [1,1,128] 填不满机器；长上下文按比例放大 |
| bf16 gemv | 1.46 | lm_head 636 MB；GDN 项 ~4.6 MB×48 | lm_head **700 = 已达天花板**；GDN 项待定位 | 0.91 + 融合 | +0.5 | lm_head 无空间（除非 fp8 化，见拒绝项） |
| GDN 递推 | 0.62 | state 75.5 MB（读+写） | 132 GB/s | 0.15 @650（延迟结构 ~0.3） | +0.3 | 小而稳 |
| GDN conv | 0.35 | conv state 小 | 延迟受限 | ~0.2 | +0.15 | 低优先 |
| memcpy/采样/杂项 | 0.20 | — | — | — | — | — |
| **GPU busy 合计** | **27.4**（去偏） | | | **~20.6** | **+6.8** | |
| **wall step** | **29.48**（host≈0，GPU-bound） | | | **~21**（结构性下限） | | 现实目标 **25.5–27** |

已达天花板、不再列 WP：lm_head（700 GB/s）、gateup GEMV（552 GB/s）、
m1_native（延迟结构，state 字节量小）、nccl（结构性）。

## 2. S31 工作包（按 gap×可行性排序）

### S31-WP1：fp8 GEMV 小 N split-K + 全形状带宽对齐（目标 −1.5~−2.5 ms）

- **现状**：qkv/o/GDN-out 三个小 N 形状（N=2048/1536/1536，80 calls，671 MB）
  有效带宽 289–386 GB/s；gateup（N=8704）已 552。小 N 的 1-行-每-warp 流
  网格小、每 slice 两次 barrier 摊不薄（WP2-N 已证 rows 扫描到顶）。
- **方案**：`fp8_emu_gemv_stream_m1` 家族加 split-K 变体（K 按 1024 切 S 份，
  S 路部分和 → 第二步跨块归约或 atomic；N≤2048 时启用），env 门
  `VLLM_DCU_FP8_GEMV_SPLITK`（默认 0）；P0 探针 621 GB/s 几何为参照。
- **判据**：热形状 per-op −15% 以上；**贪心流零翻转**（split-K 改变累加序，
  O4 先例：必须硬门 + 0-margin 平局按 WP-C 裁决流程）；服务 A/B e2e −1 ms 以上。
- **风险**：累加序变化 → parity；两步归约的 out 二次写。回滚：env 一行。

### S31-WP2：attention decode split-KV（目标 −0.5~−0.9 ms，长上下文更大）

- **现状**：`kernel_unified_attention_3d` grid [1,1,128]，7.5 MB/层的 KV 读在
  ~95 GB/s——网格填不满；prefill 已有 3D 分段（T4 先例），decode 无。
- **方案**：decode 侧 seq 分段（flash-decoding 式：段内部分和 + 段间归约核），
  段数按 seq_len 自适应（≤8k: 4 段；262k: 更多），env 门
  `VLLM_GFX906_ATTN_DECODE_SEGMENTS`（已有旋钮位，现值 0=off）。
- **判据**：per-op −30% 以上（7.5k ctx）；贪心流零翻转（BM16 先例流程，
  归约序变化需报 max|Δ top-1 logprob|）；e2e −0.3 ms 以上（短 ctx 收益小、
  长ctx 大，按 32k/128k 各补一档微测）。

### S31-WP3：小核并核/入 epilogue（目标 −0.6~−1.2 ms）

- **现状**：25 个 triton 小核 3.26 ms/round；`fused_add_rms_norm` 22 µs×48、
  `mul_silu_slice` 4.9 µs×48、quant 类 4.9 µs×48——全部延迟受限。
- **方案**：(a) rms_norm+残差+quant 作为 fp8 gemv 输出 epilogue（csrc，48+16 层
  命中）；(b) silu-mul 并入 gateup gemv epilogue（64 层）。不动调度/顺序，
  只动输出侧逐元素数学，逐算子数值表对齐（fp32 域 max|Δ| 报表）。
- **判据**：核数减少 ≥40；贪心流零翻转；e2e −0.4 ms 以上；单项 <0.3 ms 不采纳。
- **风险**：epilogue 内做 norm 需要 per-row 归约（跨 warp），N 维分块需整行落
  block——只对 N≥2048 的形状可行，先做 GDN-out/gateup 两形状。

### S31-WP4：GDN 递推与 conv 微调（目标 −0.2~−0.4 ms）

- **现状**：m1_native 132 GB/s（state 75.5 MB）；conv 345 µs。
- **方案**：m1 CFG 表加每-lane 2 v-行档（VW=2×2 布局）与 128 线程档组合扫描
  （已有 `VLLM_GFX906_GDN_DEC_M1_CFG` 机制，纯查表扩展）；conv 与 m1 同 stream
  已相邻，评估 conv 并入 combined 核（低优先）。
- **判据**：微测 per-op −20% 以上；零翻转（纯查表/布局不改数学则数值位同）。

### S31-WP5（可选，用户裁决）：lm_head fp8 化（−0.45 ms）

lm_head 已在 700 GB/s 天花板，唯一空间是减字节：bf16(636 MB)→W8A16 fp8(318 MB)
≈ 907→460 µs。**直接改动 top-1 logprob 分布**（全词表矩阵），必须先出 96-token
贪心流 + 8-prompt parity 板 + top5 logprob 偏移报表，按 WP-C 用户裁决，不默认做。

## 3. 预期收益汇总

| 档 | 采纳项 | step p50 | 单流 t/s |
|---|---|---|---|
| 保守 | WP4 + WP1 部分（qkv/o split-K） | ~27.5–28 | ~29–30 |
| 基准 | WP1 全 + WP2 | **~26–27** | **30–32** |
| 进取 | + WP3 | ~25–25.5 | ~32–33 |
| 结构下限 | 全部物理 gap 收干 | ~21（含 nccl 7.44） | — |

结构性下限说明：nccl 7.44 + fp8 权重 5.58 GB@650 = 8.6 + KV/激活流 ~1.5 +
其余 ~2 ≈ 20–21 ms——TP4 不变时 25 ms 以下每一步都要动结构性前提。

## 4. 拒绝项 / 边界（防重开）

- **nccl 调优**：comm report 已全杠杆关闭（协议/算法/custom AR/P2P），留档。
- **SSM state fp16**：递推精度锚（D5.4 前测留档）。
- **KV fp8_e4m3**：ITL +41% 前测不晋级（z100-longctx §15）。
- **GEMV rows/变体类无 parity 门的调参**：O4 先例已证破坏位级 parity，
  一律带零翻转硬门重做。
- **decode 池布局迁移**：m1_native 已适配原生池，永久无必要。

## 5. 施工纪律（沿用）

每 WP：真实 shape 微测 → 数值对照（fp32 域 max|Δ| 报表 + 96-token 贪心流零
翻转）→ 服务 smoke（environ + 指纹核验）→ 同口径 e2e（3 reps × N≥2 重启）→
文档 `docs/result/z100-s31-*.md`。csrc 改动走 C7（备份 + md5 + build-sync.sh，
GDN 在 `_C`、fp8 在 `_rocm_C`）。度量基建复用 S30c：`.s30c/probe.sh`、
`capture.py`、`cmp_capture.py`、`kdiff.py`、`s30c_gdn_m1_bench.py`、
`p0_gemv_bench.py`（+ 为 split-K 扩展 run 模式）。
