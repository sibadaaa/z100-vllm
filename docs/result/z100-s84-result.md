# S84 结果：dot4-PV 内核 −9.6%、服务级 −5.5%，但数值门与服务级 accept_len 双红，闭案
<!-- 追加诊断（关 MTP 机制判定）见 §9：accept_len 红门 = 量化漂移 × Greedy tie 翻转 × MTP 接受计数放大，无实现 bug。 -->

日期：2026-11-12。上游：设计图
[`z100-s84-pv-int-dot-design.md`](../design/z100/s84-pv-int-dot/z100-s84-pv-int-dot-design.md)、
施工图 [`z100-s84-pv-int-dot-construction.md`](../design/z100/s84-pv-int-dot/z100-s84-pv-int-dot-construction.md)。
台账 `.s84/`，原始数据 `runs/prefill-lctx/s84/`、`.s84/logs/`。

---

## 0. 三句话

1. **8-bit dot4-PV 内核快 9.6%，服务级 131k TTFT −5.5%**：131k
   882.4 → 797.5 ms、8k 55.3 → 50.3 ms、decode M=1 0.4376 → 0.4003 ms、
   verify M=12 1.6946 → 1.5366 ms；服务级 131k TTFT 215.13 → 203.33 s。
2. **两道硬门红了**：p 的 8-bit 权重量化固有误差使 numgate rel =
   7.1e-3~1.2e-2（门 3e-3），且服务级 accept_len 不再逐位一致（c=1
   1.415→1.153、c=2 1.153→1.277）。**不采纳。**
3. **fix2（两级 15-bit 权重）数值过门但比生产慢 6.5%**（131k 938.8 ms）。
   两条路都失败 → **出口 B**：PV 保留 fma_mix，attention 发射形态专项
   收官；下一战场为 GEMM（33.0%）与 allreduce（4.2%）。

---

## 1. WP0/WP1 静态账（census，生产实例 `ILb1ELi4E`）

| 臂 | vgpr | spill | LDS | bucket | CTA/CU | 静态指令 | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| 生产 X4（本会话重建） | 69 | 0 | 20608 | 20992 | 3 | 972 | V fp16 单缓冲 |
| `pvdot8`（8-bit 主刀） | 78 | 0 | 21248 | 21504 | 3 | 1022 | V int8 双缓冲 4352×2 B |
| `pvdot_hi`（fix2 两级） | 77 | 0 | 21248 | 21504 | 3 | 1073 | 同上 + 第二套 dot4 |

- 8-bit 臂通过静态红线（vgpr ≤85 / spill 0 / 3 CTA 保住）；LDS 比设计
  的 20992 上限多 256 B，但 bucket 仍是 21504、仍是 3 CTA，不构成倒退。
- **4-CTA 顺带核（WP3）未触发**：vgpr 78 > 64 且 LDS 21248 > 16384，
  设计 §0.3 的准入条件不成立。
- 8-bit 臂的 V 转置 staging 用字节写落在 LDS：SASS 里是
  `ds_write_b8`（静态 24 条）+ `v_dot4_i32_i8` 32 条（16 条 QK +
  16 条 PV 循环体）；`v_fma_mix_f32` 从 96 条降到 32 条，剩下的
  是 tile fold / alpha 缩放，PV 内循环已无 fma_mix。

转置布局 `p(d) = d + (d>>4)`：每 16 个 dim 插一个 slot，PV 侧 lane 的
16 个 dim 起点错开 4 个 bank，16 lane 的 `ds_read_b128` 起点覆盖
8 组 bank（2-way，等于 256 B/warp 的数据量下限）；staging 的 16 条
字节写按 (dim, row) 分散，实测没有把 9.6% 的收益吃掉。

## 2. 内核计时（9 iters 中位，四形状，本会话同口径）

| 形状 | 生产 X4 | 8-bit | Δ | fix2 两级 | Δ |
|---|---:|---:|---:|---:|---:|
| 131k prefill | 882.40 ms | **797.50** | **−9.62%** | 938.82 | +6.39% |
| 8k prefill | 55.34 | **50.29** | **−9.11%** | 59.12 | +6.85% |
| decode M=1 | 0.4376 | **0.4003** | **−8.52%** | 0.4669 | +6.69% |
| verify M=12 | 1.6946 | **1.5366** | **−9.32%** | 1.8097 | +6.80% |

- 生产臂重建后 `out_md5` 与生产逐位同（`278905b1` / `7dd8029d` /
  `f244f342` / `e715a0bb`），基线可信。
- 8-bit 臂 `out_md5` 漂移（`dbd947aa` / `cf5754f9` / `7506ea54` /
  `83f6daf6`），符合 p 有损的预期。
- **两级量化慢的原因**：dot4 从 64 条/lane/tile 翻到 128 条，PV 又多了
  32 条 fold，PV 从"省 256 fma"退化成"与 fma 同量级"，把 staging 的
  收益全部吃回并倒亏。dot4 的指令速率不足以支撑翻倍。

## 3. 数值门

### 3.1 8-bit 主刀：rel 红、KL 绿

| 门 | 结果 |
|---|---|
| numgate 48 例（本轮 18 例，ctx 240/4096/32768 × m 1/3/16 × seed 21/73） | **0/18 过门，rel 7.07e-3~1.21e-2**（门 3e-3） |
| wrapper1616 公共入口 | 5/6 OK（rel 4.3e-4~5.0e-4），m=7488 **8.54e-3 BAD** |
| KL 门（one-hot，ctx 128~240，m 3~16） | **10/10 OK，KL 3.27e-4~1.86e-3**（门 1e-2） |

**根因不是实现 bug**：host 侧复算同一套"per-tile max 定标 + 8-bit
round"得到 rel 7.87e-3，与 kernel 的 8e-3 同带。误差来自 8-bit
权重本身的量化步长（≤1/(2·127)≈0.4%），在长上下文 diffuse
attention 的加权和上放大到 ~0.8%。把定标粒度从 16 token 缩到 4 token
（host 复算）也只能到 **3.79e-3**，仍过不了 3e-3；要过门必须上两级
量化（≈15-bit），也就是 fix2。

### 3.2 fix2 两级量化：数值全绿

| 门 | 结果 |
|---|---|
| numgate 6 例（ctx 4096/20000 × m 1/3/16，seed 21） | **6/6 OK，rel 2.22e-4~1.57e-3** |
| 与生产同带的 wrapper/KL | 无需复跑（rel 已回到 1e-3 内） |

## 4. 服务级配对（同一会话，同一语料/协议，只换 `.so`）

启动口径：`NOCACHE=1 KV_DTYPE=int8_per_token_head MIN_CTX=4096`，
日志确认 `gfx906 int8-KV decode inner loop enabled`。

| 指标 | 生产 X4 | 8-bit pvdot | Δ |
|---|---:|---:|---:|
| TTFT 131k | 215.13 s | **203.33 s** | **−5.49%** |
| TTFT 32k | 31.84 s | 31.27 s | −1.79% |
| TTFT 8k | 5.087 s | 5.088 s | 0.00% |
| TPOT c=1 | 36.56 ms | 36.69 ms | +0.4% |
| TPOT c=2 | 86.30 ms | 88.28 ms | +2.3% |
| TPOT c=4 | 171.14 ms | 171.42 ms | +0.2% |
| **accept_len c=1** | **1.415** | **1.153** | **不一致（硬门红）** |
| **accept_len c=2** | **1.153** | **1.277** | **不一致（硬门红）** |
| accept_len c=4 | 1.057 | 1.057 | 一致 |

- 生产侧 c=1/c=2 的 1.415/1.153 与 S83 记录逐位相同，协议可复现；
  8-bit 臂的 c=1 直接掉到 1.153，不是噪声。
- 短请求探针（8k/32k/131k，max_tokens=8）两侧 accept_len 同为
  2.0/1.667/1.667；差异只在 128-token 生成上暴露 —— 说明 0.8% 的
  attention 输出扰动会改变长生成中的 draft 接受路径。
- 结论：**服务级一票否决，S83 segments=180 的纪律照此执行。**

## 5. 出口与后续

- **出口 B**：dot4-PV 闭案于"8-bit 数值门 + 服务级 accept_len"。PV 保留
  fma_mix 形态；attention 内核的发射/占用/精度三条线全部收口。
- 10 Waves / 24 vgpr 双墙维持判死（设计 §1.4 账不变）。
- 4-CTA 未触发（静态门不满足），无新增前科数据。
- 下一战场：GEMM 33.0% + allreduce 4.2%，按新口径独立立项。

## 6. prefill 各阶段速度（生产 X4，131k，torch profiler）

口径：synthetic 131072 token prompt、`max_tokens=4`、TP=4、MBT=8192；
带 profiler TTFT **249.29 s**，同会话不带 profiler 215.13 s，按
**0.863** 折算。kernel 合计 248.46 s ≈ 请求窗口，GPU 无空转。

| 阶段（kernel 类） | 实测 | 占比 | 折算生产 |
|---|---:|---:|---:|
| **attention**（`attnd_inner_kvsdot4_kernel`） | **138.98 s** | **55.9%** | **119.9 s** |
| GEMM（fp8-emu/ Tensile/ Cijk） | 82.09 s | 33.0% | 70.8 s |
| allreduce | 10.39 s | 4.2% | 9.0 s |
| dequant/copy | 6.62 s | 2.7% | 5.7 s |
| GDN | 5.45 s | 2.2% | 4.7 s |
| other | 3.80 s | 1.5% | 3.3 s |
| norm | 0.92 s | 0.4% | 0.8 s |
| attn_reduce | 0.19 s | 0.1% | 0.2 s |
| attn_triton | 0.02 s | ≈0% | 0.02 s |

逐 chunk（17 个 engine step；窗口内 kernel 合计 238.3 s，窗口外
prologue/epilogue ≈10 s）：

| step | ctx_tok | wall | attn | gemm | allreduce | attn 占比 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8080 | 7.45 s | 0.88 s | 4.83 s | 0.61 s | 12% |
| 3 | 8080 | 10.32 | 3.69 | 4.93 | 0.61 | 36% |
| 7 | 8080 | 14.04 | 7.43 | 4.92 | 0.61 | 53% |
| 11 | 8080 | 17.80 | 11.17 | 4.92 | 0.62 | 63% |
| 15 | 8080 | 21.51 | 14.89 | 4.92 | 0.61 | 69% |
| 16 | 1792 | 5.93 | 4.13 | 1.26 | 0.16 | 71% |

- GEMM 是每 8080 token 的常数（4.92 s），allreduce 也是常数（0.61 s）；
  attention 每多 8080 token 上下文 **+0.93 s**，尾 chunk 已占 71%。
- 与 S82-b 的 X4 归因（attention 139.84 s / GEMM 82.20 s）互证，生产
  仍是 X4 形态，本轮未改动生产。
- 服务级 131k 的 −5.5% 与 attention 占比 55.9% × 内核 −9.6% 的
  Amdahl 预期（−5.4%）吻合。

## 7. 零改动核对（收工）

| 项 | 值 |
|---|---|
| `csrc/rocm/attnd_inner_kvint8.hip` | `5a16cb37`（未改） |
| `csrc/rocm/attnd_inner.hip` | `b93db390`（未改） |
| 生产 `.so` | `.s83/rocm_c_release_v2/_rocm_C.abi3.so`，`d789bfce`（软链已还原） |
| 服务 | 已停，`:8000` 释放，4 卡 VRAM 0% |
| 变体/构建 | 仅 `.s84/arms/{pvdot.hip,pvdot_hi.hip,build_base,build_pvdot3,build_pvdot_hi}` |

## 8. 文件

- `.s84/README.md`：台账索引
- `.s84/arms/pvdot.hip`：8-bit 主刀源码（V int8 dim-major + 双缓冲单屏障）
- `.s84/arms/pvdot_hi.hip`：fix2 两级量化源码
- `runs/prefill-lctx/s84/timing_*.log`：四形状计时原始输出
- `runs/prefill-lctx/s84/numgate_pvdot3.txt`：18 例数值门
- `runs/prefill-lctx/s84/prefill_stages_131k_prod.json` + `_chunks.json`：
  prefill 阶段归因
- `.s84/logs/`：服务级配对日志与 s20 探针 JSON

## 9. 追加诊断（2026-11-12）：关 MTP 后的机制判定

任务指令："先关闭 mtp，然后确定为何不一致"。诊断台账
[`.s84/diag/DIAG-NOMTP.md`](../../.s84/diag/DIAG-NOMTP.md)。

- **内核侧**（service 精确形状 ctx 4100/m=3/1616，同输入 A/B）：prod
  vs pvdot8 pairwise relF = 7.2~7.4e-3（seed 73/21 双验证），与 host
  复算的 8-bit 定标地板 7.87e-3 同带，**无实现 bug**；
- **服务侧**（关 MTP，Greedy 128 token ×4 条）：2 条逐 token 全同、
  1 条只在 **gap=0.000 nat 的精确并列位**翻转 1 位、1 条在 gap=0.047
  nat 近并列位翻转后 91/128 位重排（蝴蝶效应）。prod 自身 Greedy
  margin 就有 0~0.05 nat 的近并列带；
- **机制结论**：accept_len 红门 = 8-bit 量化漂移（7e-3 量级）×
  Greedy 近并列 tie 的离散翻转 × MTP 接受计数放大（个位数接受差异
  把 1.415 推档到 1.153）。不是实现错误，也不是质量退化——是
  "接受计数对 tie 翻转的离散敏感性"；
- 若未来要复活 pvdot8：数值门需从 "accept_len 逐位" 改为
  "tie 翻转容忍" 定义（KL + 质量门 + 长生成抽检），或 tie 高风险段
  混合 fp16；
- 诊断中的两起启动事故（悬空 arm 软链第三次复刻 S82b §5、
  `pkill -f "main() serve"` 匹配不到服务进程）已记录在诊断台账，
  停服纪律新增 **按 pid kill + `ss -tlnp` 双确认**；
- 生产态已复原：MTP 开、`.so` = `d789bfce`、c=1 accept_len 复现
  **1.415**（53 步）。
