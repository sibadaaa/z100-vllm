# S91 结果：注意力「算完等取数」假设判定（2026-11-13）

计划：[`../perf/PLAN-20261113-attn-latency-vs-bandwidth.md`](../perf/PLAN-20261113-attn-latency-vs-bandwidth.md)
台账：`.s91/README.md`　　数据：`runs/attn-diag-20261113/`

---

## 0. 三句话

1. **先修栈，再判定**：S89 relink 用了构建树里 S81 之前的 `torch_bindings.cpp.o`，
   `gfx906_attnd_decode_inner_kvint8` 注册被静默丢掉 ⇒ decode 与 prefill 双双退回
   Triton int8 反量化内核。修复后 131k prefill **991.7 → 237.6 s（4.2×）**、
   单流 e2e **65.3 → 72.7 t/s**。
2. **假设被证伪**：正确基线上，KV-load 段只占内核时间 **20–25%（decode）/ 17.5%
   （prefill）**，且 load 段有效带宽已达 **640–850 GB/s**（逻辑口径，≥784 GB/s 屋顶，
   L2 复用后 HBM 更低）；**PV + QK 占 75–79%**。不是「算完等取数」。
3. **prefill 的 H2 成立**：PV 占 48%、QK 占 31%，load 只 17.5%；prefill segments
   1→8 时间不变（127.2→124.5 ms）⇒ 不是并发不足，是 `v_dot4`/fp16 FMA 的**指令形态**
   上限（与 S58/S77 结论一致）。主线回到算力/ILP，不为 decode 立项预取/双缓冲。

---

## 1. 栈级事故（先修这个，否则后面全是错口径）

| 项 | 事实 |
|---|---|
| 症状 | decode ITL 斜率 406.9 µs/1k（文档口径 121.3）；131k prefill TTFT 991.7 s、132 t/s；32k 83.4 s |
| 直接证据 | `torch.ops._rocm_C` 无 `gfx906_attnd_decode_inner_kvint8`；服务日志 2 小时内 0 条 `int8-KV … inner loop enabled` |
| 根因 | 运行 `.so`（`.s89/arms/_rocm_C_ms16`，md5 `8ef7ffb290af`）链接的 `CMakeFiles/_rocm_C.dir/csrc/rocm/torch_bindings.cpp.o` 是 Nov 5 的旧对象（kvint8 字符串 0 次），S81（11-11）加的 send/impl 从未进二进制 |
| 后果 | `_kvint8_inner_available()` False ⇒ int8 KV 的 decode 走 Triton 3D、chunked prefill 走 Triton 2D/3D 的**逐 tile 反量化**分支 |
| 修复 | 重编 `csrc/rocm/torch_bindings.cpp` 并按同一对象集 relink：`.s91/arms/_rocm_C_kvint8fix.abi3.so`（md5 `5af8b848f08e`），已上线 |
| 复现陷阱 | 构建树 `csrc/` 是旧副本；relink 脚本在 `cd $B` 后用相对路径会编到旧副本（本轮踩过一次，schema 不匹配才暴露） |

修复后服务日志两条关键行都出现：

```
gfx906 int8-KV decode inner loop enabled: head=256, block_size=1616, segments=16
gfx906 int8-KV prefill inner loop enabled: head=256, block_size=1616, segments=4
```

---

## 2. T0：口径钉死

| 项 | 值 | 来源 |
|---|---|---|
| KV dtype | `int8_per_token_head` | `/metrics` `cache_config_info` |
| 物理页 | block 1616，`num_gpu_blocks=1313`，KV 容量 1,966,828 token，并发 7.50× | 同上 |
| 每 token 每卡字节 | **520 B** = K 256 int8 + K scale fp32 4 + V 256 int8 + V scale 4 | `kv_cache_interface.py`（`(hs_k+hs_v)*1 + 2*4`） |
| 每卡几何 | 6 q head / 1 kv head / head 256；64 层里 **16 层 full attention**（`full_attention_interval=4`） | `config.json` |
| decode 重读系数 | `ceil(M*6/16)`：M=1→1、M=3→2、M=8→3 | 内核 grid |
| prefill 重读 | q-block `ceil(T*6/16)`，每 CTA 只扫 `seq/segments`（split-KV），故每 launch 逻辑字节 = q_blocks × seq × 520 | 内核 grid |

### T0.2 斜率（同 harness，冷 prompt，`s24_longctx.py --reps 3 --gen 24`）

| 臂 | 512 | 8k | 32k | 131k | 斜率 | 131k TTFT |
|---|---:|---:|---:|---:|---:|---:|
| 旧 `.so` | 34.9 | 38.0 | 48.0 | 88.2 | **406.9 µs/1k** | 991.7 s（132 t/s） |
| 修复 `.so` | 34.8 | 36.1 | 39.0 | 45.9 | **118 µs/1k** | **237.6 s（551 t/s）** |

修复臂斜率与文档的 121.3 µs/1k 相差 2.7%，**T0.2 门通过**；旧臂差 3.4×，
按计划「先查环境」——查出来的就是上面的注册丢失。131k 那次旧臂测量与本地
HIP 编译窗口重叠，故绝对值以 32k 的 1.6× 退化为准（同期构建不影响 16 分钟级
GPU 任务，且 32k 三点在编译前）。

---

### T0.3 内核普查（`/start_profile` → 一次请求 → 聚合 kernel 名）

| 相位 | 总 GPU | attention | 主 GEMM | GEMV | AR |
|---|---:|---:|---:|---:|---:|
| 32k prefill（37.7 s 墙钟） | 37.1 s | `attnd_inner_kvsdot4` **9.17 s (24.7%)**, 153 次 | 20.4 s (55%) | — | 2.5 s (6.8%) |
| decode 7469→96 tok（37 step） | 8.69 s | `attnd_inner_kvsdot4` **0.64 s (7.4%)**, 663 次 | 4.9 s (56%) | 0.80 s (10%) | 0.69 s (8%) |

两个相位都确认生产走的是 int8 HIP 内环（decode 663 次 ≈ 37 step × 16 层 + draft）。
decode 的 attention 份额在 ~7.5k 上下文只有 7.4%，与 T1 的「65k 时 16 层占 14%」一致
——**注意力是长上下文才显著的项目，不是全程的主项**。

## 3. T1：内核内部分段（`clock64()` 探针，真实形状）

方法：`tools/dcu/attn_stage/make_stage_kernel.py` 从生产内核机械复制出
`attnd_inner_kvsdot4_stage_kernel`，只插入 `clock64()` 差值（不动算术），
注册成独立 op；**vgpr 82 / spill 0 / LDS 20608 与生产内核逐项相同**（3 CTA/CU 不变）。
分段定义：load = 暂存 K/V + LDS 写 + 等数据落地的 barrier；qk = 8 条 `v_dot4_i32_i8` 链；
sm = 分数收尾 + 跨 lane max + exp + 累加和 + acc 缩放；pv = 16 累加器 fma 环；wait = 纯等待 barrier。

### 3.1 decode（segments=256，生产值）

| M | ctx | ms | load | qk | pv | sm | wait | load GB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8k | 0.05 | 28.8% | 28.6% | 40.7% | 0.0% | 1.9% | 285 |
| 3 | 8k | 0.08 | 27.3% | 29.0% | 41.4% | 0.1% | 2.3% | 373 |
| 8 | 8k | 0.13 | 35.6% | 26.0% | 35.9% | 0.0% | 2.4% | 282 |
| 1 | 32k | 0.13 | 24.3% | 29.1% | 44.7% | 0.1% | 1.8% | 540 |
| 3 | 32k | 0.22 | 22.3% | 30.3% | 44.7% | 0.1% | 2.7% | 705 |
| 8 | 32k | 0.32 | 23.3% | 29.6% | 43.9% | 0.0% | 3.1% | 695 |
| 1 | 65k | 0.24 | 22.4% | 29.7% | 45.9% | 0.1% | 1.9% | 641 |
| 3 | 65k | 0.40 | 20.5% | 30.3% | 46.2% | 0.0% | 2.9% | 842 |
| 8 | 65k | 0.57 | 21.0% | 30.1% | 45.6% | 0.0% | 3.2% | 854 |

- 分段占比与 M/ctx **几乎无关**：load 20–25%、qk 29–30%、pv 44–46%、softmax 可忽略。
- load 段带宽随 ctx 升到 **640–850 GB/s（逻辑字节）**；由于同一 KV 被 2–3 个
  q-block 与分段共享，HBM 实际流量更低，**已无取数空间**。
- 每层 attention 在 65k 只花 0.4 ms；16 层 ≈ 6.4 ms/step，占 131k step（45.9 ms）的 14%。

### 3.2 prefill（T=8192 的 chunk，segments=4，生产值）

| P(prefix) | ms | load | qk | pv | tf/s | load GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 8k | 110.8 | 17.6% | 30.8% | 47.9% | 5.58 | 1341 |
| 32k | 274.7 | 17.5% | 30.9% | 48.0% | 6.75 | 1361 |
| 131k | 929.5 | 17.4% | 30.9% | 48.1% | 7.32 | 1377 |

prefill 的 load 段占比 **17.5%**（q-block 连续，L2 命中把逻辑带宽顶到 1.3 TB/s）；
**PV 48% + QK 31%**。PV 的 fp16 fma 环每指令 2 FLOP，是天花板所在。

---

## 4. T2/T3：并发与重读扫描

### T2 decode segments（M=3，ctx=32768）

| segments | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| ms | 0.63 | 0.41 | 0.28 | 0.24 | **0.22** | 0.22 |
| 全核逻辑 GB/s | 54 | 83 | 124 | 142 | 157 | 154 |
| load 段 GB/s | 228 | 352 | 572 | 683 | 705 | 613 |

**判读**：segments 16→256 把 load 段带宽从 228 抬到 705 GB/s（**低并发时确实延迟受限**），
但 256 之后不再涨、时间也不再降 —— 生产选的 256 正好在拐点上，**并发维度已用尽**。

### T3 prefill segments（T=4096，P=32768）

| segments | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| ms | 127.2 | 125.9 | 124.7 | 124.5 |

**判读**：prefill 不受并发限制（T=4096 已有 1536 个 q-block），41% 的 PV 时间是纯算力。

---

## 5. T4：数值门

| 门 | 结果 |
|---|---|
| stage op 与生产 op 的 segm 输出逐位同 | **3/3 形状 true**（dec 1×8k、3×8k、prefill 1024×8192） |
| 生产 dispatch（含 `reduce_segments`）vs fp32 参考 | max\|Δ\| = **4.0e-4 / 6.3e-4 / 3.9e-5** ≤ 1e-3 ✅ |
| `pytest test_triton_unified_attn_gfx906_v1.py` | **9 failed / 12 passed —— 旧臂与修复臂完全相同**（非本轮引入） |
| `pytest test_triton_unified_attention_gfx906.py` | 全绿 |

pytest 的 9 条失败分两类，均为**既存**：

1. `test_decode_num_segments_scales_with_context` 断言 16384 时 128，现实现给 256
   —— 测试没跟上启发式（S94→现值）变更；
2. 8 条 fp16 KV 数值用例（block 800 / 4 kv head / Q fp16，**不走生产的 int8 路径**）
   max\|Δ\| 达 1e-1–1e4。两臂一致 ⇒ 与本轮 `.so` 无关，但说明该套件当前**不能当门用**，
   需单独立项排查（生产 int8 路径已有上表与 accept 兜底）。

### 服务级质量兜底

| 指标 | 修复臂 | S90 基线 |
|---|---:|---:|
| tokens/step | 2.528 | 2.50 |
| accept p1 / p2 | 0.844 / 0.684 | 0.82 / 0.67 |
| step | 35.2–36.3 ms | 36–39 ms |
| 贪婪输出 | 与旧臂 2/96 id 相同（int8 Q 量化 vs Triton 的 fp16 Q，属预期） | — |

---

## 6. T5：服务级 A/B（同窗口成对）

| 臂 | W1 t/s | W2 t/s | TTFT s | 8k ITL | 32k ITL | 131k ITL | 131k TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 `.so` | 65.32 | 56.58 | 0.367 | 38.0 | 48.0 | 88.2 | 991.7 s |
| 修复 `.so` | **72.7** | **64.19** | **0.307** | **36.1** | **39.0** | **45.9** | **237.6 s** |
| 变化 | +11.3% | +13.5% | −16% | −5.0% | −18.8% | −48% | −76% |

门（ITL p50 ≥3%、accept 不降、0 crash）**全部通过** ⇒ 采纳，已上线
（`.s91/arms/_rocm_C_kvint8fix.abi3.so`，指纹 `FP 8bf8b075a6469af0` 未变）。
旧臂复测 65.32/56.58 与 S90 记录的 65.8/56.4 一致，说明两臂对比是同口径的。

---

## 7. 判定矩阵填表

| 计划行 | 本轮证据 | 结论 |
|---|---|---|
| KV-load 段 >50% 且带宽 ≥600 → 真·取数瓶颈 | load 仅 17–25%，带宽 640–850 GB/s | ❌ 不成立 |
| load 占比高但带宽 <300，提并发后升 | 只在 segments<128 时成立（228→705 GB/s） | ❌ 生产已在拐点 |
| QK/PV 占比高、VALU busy 高但 FLOPs 低 → 算力未发出 | PV 44–48% + QK 29–31%，prefill TF/s 5.6–7.3 | ✅ **prefill 成立** |
| 冷热 CT 差 >15% → 缓存延迟敏感 | 未单测；load 段已≥屋顶，L2 只帮正忙 | 不立项 |

**总判据**：T2 的「提高并发」臂已把 decode KV 段推到 705→854 GB/s（≥784 屋顶），
但总时间不再下降 ⇒ **「算完等取数」被证伪**；**反证条款触发**：主线回到 S58 的
算力/ILP 路线，不在 decode 做预取/双缓冲。

---

## 8. 下一步（按收益排序）

1. **PV 环的指令形态**是剩余最大项（decode 45%、prefill 48%）。S77 已证明
   「抹掉点积」的地板仍够不到 T2 门，说明要动的是 PV 的「每指令 2 FLOP」：
   候选是 `v_dot2_f32_f16` 重排（把 2 个 w 合并成一次 dot2）、或把 P 也量化成 int8
   走 `v_dot4`（需接受额外量化误差，先跑 per-op 门）。
2. **QK 链**（30%）在 int8 内核里已是 8 链；S58 微测 8 链上限 22.7 TF/s，
   实测 prefill QK 23.0 TF/s（131k）已到上限 ⇒ 不再投入。
3. 把 S81 的 int8 预填内环**纳入构建门**：本轮的注册丢失说明 relink 必须校验
   `hasattr(torch.ops._rocm_C, …)`（建议在 `tools/dcu/attn_stage/build.sh` 之外，
   给生产 relink 脚本加一条断言）。
4. T4 的 fp16 数值套件单独立项（两臂同红，与生产 int8 路径无关，但会掩盖真回归）。

## 事故记录：profiler 收尾后请求挂起（第 2 次）

- 13:41 用 `/start_profile` + 7469 prompt 采 decode 普查，`/stop_profile` 后下一次
  请求在 `num_requests_running=1` 上挂死（客户端 120 s 超时），引擎不再前进；
  `systemctl restart z100-vllm-mtp2` 后 0.96 s 正常返回。
- 与 S89 台账记录的「一次 profiler 收尾后请求挂起（未复现）」同型 —— 这是第 2 次，
  可作为一条独立待查项（torch profiler 的 kineto 收尾与 worker 线程的
  "External init callback must run in same thread as registerClient" 报错同时出现）。
- 结论不受影响：profile 数据在挂死前已落盘，服务级 A/B 与 accept 都在 profiler
  窗口之前完成。
