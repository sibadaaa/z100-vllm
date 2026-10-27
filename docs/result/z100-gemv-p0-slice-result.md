# WP-GEMV P0/P1/P2 施工结果：stream GEMV A-slice 化（2026-10-25）

> 施工依据：[`docs/perf/GEMV-OPT-20261025.md`](../perf/GEMV-OPT-20261025.md) §3 P0/P1/P2。
> 树：`dtk` @ `7078b5fe1` + 本次改动。栈：4×Z100L（gfx906）、DTK 26.04、torch 2.11、TP=4、
> DFlash2 k=7、`--enforce-eager`、单请求。
> 产物：`_rocm_C.abi3.so` md5 `f42d0064…`
> （`/mnt/game/perf1/_rocm_C.gemv-p0-final-f42d0064.so`；基线 `.before.so` md5 `ff04f1b9…` 未动）。

---

## 1. 一句话结论

**P0 门限问题已修死：M=8 重新走 stream 路径、K=17408 的 M≥2 解锁，单 kernel 全面持平或大胜旧路径
（M=4/M=8 +17~31%，down 投影 M≥2 提速 3~10 倍）；但文档 §5 的"M=8 生产 fp8 ≥350 GB/s"没有达到
（实测 100–163 GB/s）——fp8 数据通路的 A 按(fp16 对)读放大 2 倍 + LUT/scale，M=8 撞的是
LDS 读流量墙，fp16 probe 的 451 GB/s 不可能迁移到生产 fp8。服务级 step p50 在 WP0.2 同口径下
无变化（123.6 → 129.2 ms，±5–6% 噪声内），与 WP0.2 已有结论一致：该工作点的 decode step 不受
GEMV 派发影响。**

---

## 2. 改动清单（全部落在 `csrc/rocm/fp8_emu_gemm.cu`，另两处配套）

| 项 | 内容 |
|---|---|
| **P0 内核** | `fp8_emu_gemv_stream_block_kernel` 整块 A-LDS（M·K·2 B）→ **按 K 切片**：每片 `kStreamSlice=1024` 个 fp8 元素/行（64 个 uint4 = 每 lane 每(片,行)恰一个 W chunk），LDS = `MS·1024·2 B`（MS=8 时 16 KB）+ 折叠 scale。循环结构：片外层 → 暂存 A 片 → barrier → 各 warp 消费 → barrier。W 侧加**深度 2 的寄存器预取流水**（片循环按 2 展开，buffer 下标为编译期字面量——运行期下标会溢出到 scratch，实测比无流水还慢）。门限 `stream_block_select` 改为按 slice 计：`MS·kStreamSlice·2 + M·nkb·4`，want==2（variant 2）仍按整块 A 计。 |
| **P1** | m 维按 4+4 两组复用同一份加宽后的 W 寄存器（`p[0..7]` 每片每 chunk 只算一次）。**注**：§3-P1 的"LDS 读减半"机制不成立——每个 A 字节恰与一个 W 字节点积（k 对齐 1:1），8m×32 B/16 B-W 是下界；实测 4+4 对 GB/s 中性（±3% 噪声内），保留因为 VGPR 无恶化。 |
| **P2** | rows 派发改为 **rows=8（每 warp 一行）**，N/8 < 120 blocks 时降 rows=4（nt=256）；`VLLM_DCU_FP8_GEMV_STREAM_ROWS` 新旋钮（0=auto，4/8/16/32/64 覆盖）供服务级 A/B；`VLLM_DCU_FP8_GEMV_PREF` 随旧逐行流水退役（slice 设计无逐行 K 循环），`dcu_policy_fingerprint.py` 同步换名。variant 2（stream2，M≤4 A 整块）未动。 |
| 配套 | `tools/dcu/dcu_policy_fingerprint.py`（旋钮换名）；新探针 `tools/dcu/p0_gemv_bench.py`（生产 op 微测+对拍）、`p0_gemv_probe.py` / `p0_ref_check.py`（小形状对拍）。 |

rows A/B（gateup 5120×8704 w8a16，GB/s）：rows=4: 307/169/108（M=1/4/8）；**rows=8: 494/250/163**；
rows=16: 458/265/164；rows=32: 524→（预取前）336/162（预取后退步）；rows=64: 459/212/140。
默认取 rows=8。

## 3. 门禁

1. **ISA**：`.s23/probes/isa_gate.py` PASS（`v_dot2_f32_f16`=92382，无 mfma/cmac）。
2. **数值对拍**（`VLLM_DCU_FP8_GEMV_STREAM=0` tiled 路径为基准，fp16 输出口径）：
   40 个 (形状×路由×M) 组合 **max|Δ| ≤ 2.44e-4**，多数 ≤1e-4，门限 1e-3 ✓。
   边界形状：K=1152（尾片）、K=512（单片）、a16/fp8 两暂存路径、variant 2 均 ≤1.2e-4 ✓。
3. **pytest**：`tests/kernels/quantization/test_gfx906_fp8_emu.py` **188 passed / 3 failed**，
   3 个失败在基线 `.so` 上同样失败（`prefill_route_traces_under_compile`、
   `lut_op_matches_device_lut`、`apply_weights_w8a16_full_path`，环境性，与本次无关）。
4. **§5 复跑**：`gemv_probe.cpp` 数字与文档证据一致（lds_m8 451 / lds_m4 595 / lds_m1 725，
   dot1 751–770），硬件状态未漂移。

## 4. 生产 kernel 微测（hip event，30 iters，空闲 GPU3；GB/s 只计 W 的 fp8 字节）

### W8A16（生产路由）

| 形状 (K×N) | M | 基线 .so（旧派发） | 新 slice（rows=8） | Δ |
|---|---:|---:|---:|---|
| gateup 5120×8704 | 1 | 421 | **496** | +18% |
| | 2 | 336 | 336 | = |
| | 3 | 273 | **287** | +5% |
| | 4 | 191 | **250** | +31% |
| | 8 | 139（tiled，被旧门拒） | **163** | +17% |
| down 17408×1280 | 1 | 310 | **434** | +40% |
| | 2 | 34（旧 GEMV） | **297** | 8.7× |
| | 3 | 27 | **254** | 9.4× |
| | 4 | 22 | **224** | 10.2× |
| | 8 | 47（tiled） | **144** | 3.1× |
| qkv 5120×2048 | 1..4 | 316/262/213/142 | 339/255/220/195 | M4 +37% |
| | 8 | 67（tiled） | **124** | +85% |
| o 5120×1536 | 1..4 | 293/246/204/138 | 285/238/206/182 | M4 +32% |
| | 8 | 50（tiled） | **118** | +136% |

W8A8 路由同趋势（gateup M=8：102 → 139 GB/s）。原始 JSON：`.s27/p0-ref/bench_*.json`。

### 为什么 M=8 达不到文档的 350–450（判据表"及格/优秀"）

fp16 probe 每 16 B 的 W（1×uint4=8 个 fp16）每 m 读 **16 B** A；生产 fp8 每 16 B 的 W（16 个 fp8）
加宽后每 m 要读 **32 B**（2×uint4 fp16 对），再加每 chunk 8×u16 的 LUT 读与 8×u32 的折叠 scale——
合计 ≈ **21 B LDS 读 / 1 B W（M=8）**，是 fp16 probe（8.5 B/B）的 2.5 倍。probe 的 451 GB/s
建立在 8.5 B/B 上；2.5× 流量下同墙 ≈ 180 GB/s，与实测 163 吻合。A 的 LDS 读字节是 k 对齐 1:1
的硬下界，**在该 datapath 内无编译/调度手段可再降**（P1 的"减半"机制不存在，见 §2）。要再往下压
M=8 的权重项，只剩文档 P3 的 W4/`v_dot8_i32_i4` 路线（把每字节的点积开销减半），属独立立项。

## 5. 服务级同口径 A/B（step p50，真实 7469-token 文档，DFlash2 k=7，单请求，`s20_single_probe`）

| 臂 | `.so`（map_files 核实） | step p50 | 备注 |
|---|---|---:|---|
| 基线 | `_rocm_C.before.so`（ff04f1b9） | **123.60 ms**（3 reps） | M=8 走 tiled |
| 新 | `f42d0064`（f42d…，map 核实 28 段映射） | **129.24 ms**（5 reps：129.2/123.3/133.6/136.0/128.8） | M=8 走 slice stream |

Δ = −0.3~+4.5 ms，在文档记载的 ±5–6% 单 run 噪声内，**判"无 step 级可测差异"**。
与 WP0.2（gemv8 vs gemv1：129.26 → 134.27，判"GEMV 已不是瓶颈"）一致：单请求 + accept 1.54 的
decode step 由 draft 循环/主机开销等主导，M=8 GEMV 权重项（估算 30–50 ms/步）的红利在该工作点
不浮现。按文档 P4 的判断，红利应在 **4–5 路并发 / 更高 accept**（权重按步摊薄后）才可测——那是
下一个 A/B 窗口的活，本次只做单请求口径。

**服务 smoke**：新 `.so` 启动正常，`W8A16 engaged=True (built op True), GEMV_MAX_M=8`，
真实补全/对话返回正常（`.s27/p0-smoke.json`）。测试后服务已停、GPU 已释放（恢复进场状态）。

## 5.1 服务内逐层 profile（回答"是否走到了正确的路径"）

用 fork 自带打点（`VLLM_QWEN35_LAYER_PROFILE`，event 计时，M=8/tokens=8）对两个臂各跑一次
verify step（新臂 LIMIT=132 覆盖两个 step 取稳态；原始日志 `/tmp/serve-prof-{new,base,new2}.log`）：

| 段（每层，TP0 rank） | 基线 .so | 新 .so | 与微测预测 |
|---|---:|---:|---|
| layer0 (GDN) mlp | 0.539 ms | 0.498 ms | **−41 µs ≈ gateup 微测差 −47 µs ✓** |
| layer3 (FA) mlp | 0.73 ms | 0.68–0.91 ms | 同向 |
| layer3 (FA) attn 首次 | 16.4 ms | 18.2 ms | Triton verify-3D 首调 JIT，非稳态 |

**结论：新内核确实在服务里执行**——mlp 段的臂间差值与内核级预测吻合，不是"没走到"。

**稳态 step 预算（新 .so，单个 verify step，M=8）**：

| 组成 | 每层 | 小计 | 占比 |
|---|---:|---:|---:|
| GDN attn（48 层，线性注意力 + 其投影） | 1.20–1.23 ms | ≈ 58 ms | ≈ 47% |
| FA attn（16 层，verify-3D） | 1.74–1.80 ms | ≈ 28 ms | ≈ 22% |
| GDN mlp | 0.70–0.73 ms | ≈ 34 ms | ≈ 27% |
| FA mlp | 0.50–0.56 ms | ≈ 8.5 ms | ≈ 7% |

即 **attention ≈ 86 ms（~66%）**，fp8 MLP ≈ 42 ms（~32%）；本次改动只触到 mlp 里的 gateup 与
FA 层的 qkv/o，合计 **≈ −3~−6 ms/步（−2.5~4.5%）**——低于 §5 A/B 的 ±5–6% 噪声，与 step p50
"无可测差异"自洽。**§5 之前"权重项 55.7→29.7 ms（−26 ms）"的推算作废**：它按孤立 op 微测相加，
而服务内基线 mlp（0.54–0.81 ms/层）本来就低于孤立和（0.79 ms），服务里 down 投影等部分的实际
代价低于孤立项测值；每层 −41 µs 的实测差值才是生产口径的真实收益。

顺带：GDN 层的 attn（1.2 ms×48 = 58 ms）是 step 里最大的单项——下一个 step 级优化对象应是它，
不是 GEMV。

## 6. 回滚

`vllm/_rocm_C.abi3.so` 是指向 build 目录的符号链接；回滚 =
`ln -sfn <repo>/.r3/so-backup/_rocm_C.before.so vllm/_rocm_C.abi3.so`（md5 校验 `ff04f1b9…`）。
基线与新 `.so` 均另存 `/mnt/game/perf1/`（`_rocm_C.gemv-p0-final-f42d0064.so` /
interim `b3ba6e8b`）。无 Python 契约变化（op 签名未动），无需要回滚的 Python 侧改动。

## 7. 遗留 / 后续

- **P3（W4 / `v_dot8_i32_i4`）**：M=8 的下一级杠杆（每字节点积开销减半），独立立项。
- **并发口径 A/B**：4–5 路 MBT 窗口里验证 P0 红利是否浮现为 step/吞吐（文档 P4 的论断）。
- GDN 层的线性投影不走 fp8-emu GEMV（bf16/GDN 专用核），本次 per-step 权重项估算未含其影响；
  精确的 step 分解建议跑一次 `s27_five_phase_profile.py`。
- pytest 的 3 个环境性失败建议单独修（与本 WP 无关）。
