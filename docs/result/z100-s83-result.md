# S83 结果：X4 采纳上生产；segments=180 内核赢 17%、服务级中毒闭案

日期：2026-11-12。上游：S82-b（施工图
[`s82b-v-lds-shrink-construction.md`](../design/z100/s82b-v-lds-shrink/z100-s82b-v-lds-shrink-construction.md)、
结果 [`z100-s82b-result.md`](z100-s82b-result.md)）。
台账 `.s83/`，原始数据 `runs/prefill-lctx/s83/`、`.s83/logs/`。

---

## 0. 三句话

1. **X4 采纳完成**：`csrc/rocm/attnd_inner_kvint8.hip` = V 单缓冲 + X-REG +
   `launch_bounds(256,3)`（md5 `bd53a07c`），生产 `.so` =
   `.s83/rocm_c_release_v2/_rocm_C.abi3.so`（md5 `d789bfce`，内核与
   `.s82b/arms/build_x4` 逐位一致）。内核复验四形状全对齐 S82-b（±0.1%），
   服务级 131k TTFT **215.1 s**（基线 232.7 → −7.6%），数值门全过。
2. **decode 波量化（segments 256→180）内核级 −17.2%**（M=1 0.4371→0.3618 ms，
   M=12 −1.1%，输出与 256 逐位同），但**服务级输出中毒**（temperature=0 下
   乱码、accept_len 0.8/0.4）——内核级无法复现（内核+reduce 在服务精确
   形状上 180==256 逐位同），嫌疑锁定在 graph 捕获路径的交互。**不采纳**，
   留作独立专项。
3. 本轮顺带修好一个真 bug 面：`reduce_segments` 的 segment 维不再要求
   pow2/与缓冲同宽（`SEG_STRIDE`/`SEG_ARANGE` 参数化），为后续任何
   非整波段的 segments 调优铺路。

---

## 1. A 线：X4 采纳（全过）

### 1.1 内核级复验（9 iters 中位，与 S82-b 对照）

| 形状 | S82-b X4 | 本轮复验 | out_md5 |
|---|---:|---:|---|
| 131k | 881.3 | 881.4 | `278905b1`（=基线） |
| 8k | 55.5 | 55.4 | `7dd8029d`（=基线） |
| decode M=1 | 0.4339 | 0.4290 | `f244f342`（=基线） |
| verify M=12 | 1.6899 | 1.6931 | `e715a0bb`（=基线） |

ISA：`ILb1ELi4E` vgpr 69 / spill 0 / LDS 20608（bucket 20992 → **3 CTA/CU**）。

### 1.2 数值门

| 门 | 结果 |
|---|---|
| wrapper1616（公共入口，生产页形） | 6/6 OK，rel ≤5.0e-4 |
| numgate 48 例 | 18/18 OK（本轮 2 seed 子集）+ 全量复跑 18/18 |
| corner 96 例（MIN_CTX 4k 带） | **96/96** |
| KL 门 | （S82-b 已全绿，内核未变） |

### 1.3 服务级复验（NOCACHE 口径，同语料）

| 指标 | 基线 (5fd07f21) | X4 采纳 | Δ |
|---|---:|---:|---:|
| TTFT 131k | 232.7 s | **215.1 s** | **−7.6%** |
| TTFT 32k | 33.08 s | 31.98 s | −3.3% |
| TTFT 8k | 8.13–8.37 s | 9.94 s | 噪声带外**待复测**（首请求冷启动效应，32k/131k 干净） |
| TPOT c=1 | 36.98 ms | 36.61 ms | −1.0% |
| TPOT c=4 | 174.05 ms | 175.70 ms | +0.9%（噪声带内） |
| accept_len | 1.415 / 1.153 | 1.415 / 1.153 | 逐位一致 |

与 S82-b WP5（216.3 s / −7.1%）互证。**采纳成立。**

### 1.4 固化

- `.s83/rocm_c_release_v2/_rocm_C.abi3.so`（`d789bfce`）；
- `serve_deploy.sh` 注释：md5 链 `.s83(d789bfce) → .s82(5fd07f21) →
  .s81(b032e0cb) → .s77(e90b0fdb)`；
- 回滚：一行 symlink + 重启；
- 服务健康：int8 内环日志、block 1616、`vllm._rocm_C` 导入失败 0、
  `fp8_utils:1070` 警告 0、冒烟正常。

---

## 2. B 线：decode 波量化（segments 180）—— 内核赢、服务中毒

### 2.1 动机与内核账

3 CTA/CU = 180 驻留槽。decode M=1 网格 = segments 个 CTA：
256 → 1.42 波（尾波只装 76/180）；**180 → 恰好 1.0 整波**。

内核级（生产精确形态：cap-256 缓冲 + stride=256 + 服务 hint）：

| 形状 | segs=256 | segs=180 | Δ |
|---|---:|---:|---:|
| M=1 @131k | 0.4371 | **0.3618** | **−17.2%** |
| M=3 verify | 0.7238 | 0.6952 | −4.0% |
| M=12 verify | 1.7002 | 1.6818 | −1.1% |
| 输出 vs 256 | — | max diff ≤3.8e-6（近逐位） | ✅ |

非 180 的扫描（128/160/200/224/256）全部劣于 180，整波拟合是唯一甜点。

### 2.2 服务级：中毒

`VLLM_GFX906_ATTN_DECODE_SEGMENTS=180` 重启后，temperature=0 单请求输出
**统计性乱码**，TPOT sweep accept_len 0.812/0.727/0.405（正常 1.415/1.153）。

### 2.3 排除记录（都验过）

| 嫌疑 | 实验 | 结果 |
|---|---|---|
| reduce 非整波掩码 | segstride_test2/3（exact 180 与 cap256+stride180） | 180==256 逐位同 → 排除 |
| 服务 hint（92=ceil(262144/2880) 非桶） | dec180_service_shape（hint 92 vs 64） | 180==256 逐位同 → 排除 |
| ctx=4097 M=3 服务形 | dec180_ctx4k | 180==256 逐位同 → 排除 |
| M=3/M=12 内核 | dec180_m3 | 逐位同 → 排除 |
| **graph 捕获交互** | 未测（服务 decode 全程 FULL_DECODE_ONLY 图内） | **未排除，头号嫌疑** |
| 控制组（同 .so 同 py 改动，segs=256） | serve-seg256ctl | 输出正常 → 损伤确系 180 触发 |

### 2.4 处置

**不采纳，回退 segments=256。** 内核级收益是真的（−17% @M=1），但服务级
正确性没过门就是没过门——按第 6 步纪律，服务级有一票否决权。
后续专项需要：CUPTI/graph 回放下的逐 kernel 对照 + 捕获期与回放期的
shape/stride 差异审计（`is_current_stream_capturing` 在 decode 图内
恒真，门行为与 eager 不同）。这也是"非整数 segments 只在图外安全"的
候选结论，待证。

---

## 3. 附带修复：reduce_segments 的段维参数化

`reduce_segments` 原来硬约束 `tl.arange(0, NUM_SEGMENTS_PER_SEQ)`（pow2 且
等于缓冲段宽）。新增 `SEG_STRIDE`（缓冲段宽，寻址用）与 `SEG_ARANGE`
（next_pow2，arange 用）两个 constexpr，全部 6 个调用点已带参；
老口径（SEG_STRIDE == NUM_SEGMENTS_PER_SEQ == pow2）数学不变。
本轮验证：prefill/decode 全门 + 96 角点全绿。

## 4. 启动纪律教训（本轮两次险情）

1. **arm 名必须是已存在的 `build_*` 目录**：`.s82b/run_serve_env.sh
   adopt-x4` 造出悬空软链 → `import vllm._rocm_C` 失败 → Triton FP8 回退
   （S82-b §5 事故复刻），并新写污染 key `500b8911c0`（已隔离
   `_quarantine/500b8911c0-bad`）。鉴别特征：好 key `fp8_emu_mm_block`
   832 次，坏 key 0 次 + `scaled_mm` 2304 次。
2. **改 env 文件必须确认落地**：`VLLM_GFX906_ATTN_DECODE_SEGMENTS` 的
   append 在被 pkill 中断的命令块里丢失过一次，服务带着旧值起、日志报的
   segments 与预期不符才发现。启动后必须核对日志内的
   `segments=%d` 实际值。
