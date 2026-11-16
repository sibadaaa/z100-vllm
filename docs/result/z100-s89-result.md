# S89 合批 GEMV 行数经济 —— 结果

日期：2026-11-12。状态：**结案，出口 B（零内核/零生产文件改动）**。
设计/施工：[设计图](../design/z100/s89-batch-gemv-economy/z100-s89-batch-gemv-economy-design.md)
/ [施工图](../design/z100/s89-batch-gemv-economy/z100-s89-batch-gemv-economy-construction.md)。
台账：[`.s89/README.md`](../../.s89/README.md)；原始数据 `runs/prefill-lctx/s89/`。

## 一句话

**三刀全部落空，但把真正的攻击面量出来了。** int8 dot4 GEMV 现成内核在
M=3/6/8 只有 W8A16 的 **0.82×**（不是设计预期的 1.5×）⇒ H2 出口 B；
rows=16 的微测收益（M=6/8 **+23%**）在服务上**完全不兑现**（c=1/2/4 全在
±1% 内），因为引擎实际喂给 GEMV 的主形状是 **M≤2**，不是设计假设的
M=B×(1+k)；H3 要拆的“other 46–49%”在 decode-only 窗口里其实是
**B=4 的 tile GEMM（69%）**，真正的 elementwise/norm/quant 合计 ≤5%。

## WP 判决

| WP | 内容 | 门 | 判决 |
|---|---|---|---|
| WP0 | M 梯度 / rows 曲线 / other 清单 / AR 行域 | 账面闭合 | ✅ 完成 |
| WP1 | H1 rows 自适应 | M=6/8 +14%、M=3 不劣 | ⛔ 微测成立、**服务零收益**；不采纳 |
| WP2 | H2 int8 dot4 GEMV 到 M=12 | G-2a~d | ⛔ **出口 B**（WP2.1 判决点即死） |
| WP3 | H3 other 合批 | other 单价 −30%、步时 −8% | ⛔ 闭案（无 40% 可拆对象） |
| WP4 | 服务 A/B | G-2d | ✅ 完成（rows 4 vs 16：null） |
| WP5 | 收账 | — | ✅ 本文 + 台账 |

## 1. WP0：崩塌是路由台阶，不是 M 曲线

生产 rows=4，单卡 11 真实形状（split 加权）：

| M | 路由 | GB/s | ms/步（单卡，fused） |
|---:|---|---:|---:|
| 1 | GEMV | 609.3 | 30.8 |
| 3 | GEMV | 359.5 | 53.2 |
| 6 | GEMV | 190.0 | 102.2 |
| 8 | GEMV | 154.3 | 126.0 |
| 9 | tile | 66.2 | 279.3 |
| 12 | tile | 66.2 | 279.3 |
| 16 | tile | 65.8 | 281.1 |

M=9..16 是 tile，时间**几乎不随 M 变**；M=8→9 是 `kMsMax=8` 的**路由台阶**。
M=12 的 279.3/4 = 69.8 ms/卡，与设计图 66.9 一致（差 4%）。
设计图“M≥4 随 M 线性崩塌”应改为“M≤8 内随 M 线性，M≥9 切 tile 后平坦”。

rows 曲线（独立进程直切 env；bit-exact 全同）：

| M | rows=4 | rows=8 | rows=16 |
|---:|---:|---:|---:|
| 1 | **629.1** | — | 588.8 (−6.4%) |
| 2 | **457.5** | — | 441.6 (−3.5%) |
| 3 | 359.5 | 333.3 (−7.3%) | **365.6 (+1.7%)** |
| 6 | 190.0 | 225.0 (+18.4%) | **233.1 (+22.7%)** |
| 8 | 154.3 | 186.7 (+21.0%) | **190.0 (+23.1%)** |
| 12 | 66.2 | 66.2 | 66.2（tile，无关） |

## 2. WP1：H1 微测成立，服务不兑现

服务（t1int8、int8 KV、NOCACHE=1，512 prompt / 128 gen）：

| 臂 | c=1 ITL | c=1 聚合 | c=2 ITL | c=2 聚合 | c=4 ITL | c=4 聚合 |
|---|---:|---:|---:|---:|---:|---:|
| rows=4 | 35.93 ms | 52.8 t/s | 55.04 ms | 60.1 t/s | 125.28 ms | 52.5 t/s |
| rows=16 | 36.03 ms | 52.9 t/s | 55.54 ms | 59.6 t/s | 125.75 ms | 52.9 t/s |

三档全在噪声内（TTFT 同样持平）。decode-only profiler（rows=16）给出原因：

- **c=2**：主 GEMV 模板是 `fp8_emu_gemv_stream_block_kernel<2,8,...>`（MS=2，
  M≤2），占设备时间 61%；M=6 的 +23% 杠杆根本没有被喂到；
- **c=4**：`fp8_emu_gemm_block_w8a16_kernel`（tile，M=12）占 69%，rows 无关。

**不采纳**：env-only 的全局 rows=16 只会让 M=1/2 付 −3.5~−6.4%；
M 自适应要改 .so 宿主角选择并重编，服务侧无收益 ⇒ 不立项。工作树/生产 argv 未动。

## 3. WP2：int8 dot4 出口 B（判决性数据）

现成 `w8a8_sdot4_gemv`（M≤8）同 11 形状、同口径对比：

| M | int8 GB/s | W8A16 GB/s | 比 |
|---:|---:|---:|---:|
| 3 | 293.5 | 359.5 | **0.82×** |
| 6 | 159.8 | 190.0 | **0.84×** |
| 8 | 126.0 | 154.3 | **0.82×** |

M=9..12 被拒（`M must be in [1,8]`）。线程 64/128/256/512 扫描无 1.5× 头寸
（M=8 最好 134 GB/s）。**设计图 R1 的止损条件（int8 < W8A16 ×1.2）直接命中**：
现成内核是“一 block 一行 W、64 线程”的老结构，换 sdot4 只省 MAC 发射，
省不掉访存/归约/launch；S86 的 prefill dot4 2.00× 不能平移到 GEMV。
H2 的 kMsMax 重编与数值全套不启动。

AR 行域：生产 `CUSTOM_AR_MAX_KIB=128`，M=12 的 120 KiB 在门内；
M=16 的 160 KiB 会摔回 RCCL。H2 不出门，该交互不触发。

## 4. WP3：other 清单（decode-only，B=4）

| 家族 | 占比 | 单价 |
|---|---:|---:|
| W8A16 tile（M=12） | **69.2%** | 235–752 µs/次 |
| triton_matmul（lm_head/draft） | 14.5% | 376 µs/次 |
| allreduce | 4.9% | 41 µs/次 |
| GEMV stream（M≤2，draft） | 3.4% | 80–122 µs/次 |
| triton 小算子（norm/quant/act 融合） | 2.2% | 5.5 µs/次 |
| elementwise/copy | 1.2% | 6.2 µs/次 |
| GDN | 1.5% | — |
| attention | 0.5% | — |

**判决**：SOL §5.2 的“other 46–49%”是**把 tile GEMM 当成了 other**。
真正的 elementwise/norm/silu/quant 类合计 ≤5%，前三名不可能占 `other` 40%，
按设计图 R4 闭案：没有 −8% 步时的合批对象。B=4 的钱在 tile GEMM（69%），
不在小算子。

## 5. 服务侧事故与修复（对后续所有专项有效）

按生产 argv/env 起服务后 decode 实测 **2.5 s/token**，profiler 显示线性层
走 `w8a8_triton_block_scaled_mm_func`（16 ms/次），一条 `fp8_emu` kernel 都没有，
而启动日志明确 `Selected Gfx906Fp8BlockScaledMMLinearKernel` + `W8A16 engaged=True`。
根因：`VLLM_CACHE_ROOT=/mnt/game/vllm-cache-host` 里的旧 torch.compile 图
被复用。**加 `VLLM_DISABLE_COMPILE_CACHE=1`（NOCACHE=1）重启后恢复**
（ITL 35.9 ms @c=1）。后续服务测量必须带 NOCACHE，否则会拿到 Triton
降级路径的假数字。

## 6. S90 建议

1. **主刀改为“W8A16 stream 到 M=12”**（不是 int8）：tile 66 GB/s vs M=8 的
   154 GB/s，MS=16 按 1/MS 外推 ≈103 GB/s，即 tile 段 1.5×、B=4 步时约 −25%。
   收益大且**逐位不变**；难点是 LDS 归约深度和 S57 旧 ms16 试验的否定结果，
   必须在现役（S85 后）内核上重编重测，不能沿用 S57 数字。
2. int8 dot4 GEMV 线**关闭**，除非先证明存在“W 单读 + M 路 sdot4”的新内核形态，
   而不是给现成老内核加 kMsMax。
3. B=4 的 `triton_matmul`（14.5%）单列一项归因（lm_head 还是 MTP draft），
   它比所有小算子都值钱。
4. 服务 A/B 一律 NOCACHE=1；行选择类实验先 profile 出真实 MS 形状再立项。

## 7. 零改动核对

| 项 | 期望 | 收工 |
|---|---|---|
| `csrc/rocm/fp8_emu_gemm.hip` | `16018d1b284b` | `16018d1b284b` |
| `csrc/rocm/w8a8_sdot4.hip` | `bf7a678b9b45` | `bf7a678b9b45` |
| `vllm/_rocm_C.abi3.so` readlink | `.s85/arms/build_t1int8` | 同 |
| 生产 `.so` md5 | `188b6e3aae53` | `188b6e3aae53` |
| 生产 argv/env | 未改 | 未改（`_GEMV_STREAM_ROWS=4`） |

## 8. 排除表

| 项 | 结论 | 依据 |
|---|---|---|
| int8 dot4 GEMV（现成内核，M≤8） | 0.82–0.84× W8A16，无 thread 头寸 | WP2.1 |
| H1 rows=16（env-only） | 微测 +23%，服务 0；M=1/2 −6% | WP4 |
| H3 elementwise 合批 | 可拆对象 ≤5%，无 −8% 空间 | decode-only profile |
| SOL §5.2“other 46–49%” | 口径错误：是 tile GEMM | 同上 |

---

## 9. 增补（同日）：G2 本体跑通——kMsMax=16 + exact-M=12

S89 原结论否掉了 int8 捷径，但把 G 轨的 G2 留成了空白。本节是随后补做的
G2 本体：**在现役 W8A16 stream 内核上把 M 从 8 扩到 16，并为生产 B=4 的
M=12 加一个 compile-time 定宽臂**。变体源
[`.s89/arms/fp8_emu_gemm_ms16.hip`](../../.s89/arms/fp8_emu_gemm_ms16.hip)
（相对现役只多 3 处：`kMsMax=16`；M=5..8 仍走原 MS=8、M=9..16 走 MS=16；
M==12 走 MS=12 的 EXACT 臂；M>8 时 rows=8/nt=512）。

**构建可对照**：用同一套对象（S85 t1int8 + 构建树 attention/w8a8/bf16/
bindings）先链一个“原源码”控制臂，其 M=3/8 × 11 形状 out_md5 与生产
`.so` **逐位全同**；变体与控制臂只差上述源码补丁。

### 9.1 微测（11 形状，split 加权 GB/s / fused ms 每步）

| M | tile（现役） | MS16 rows=16 | MS16 rows=8（EXACT@12） |
|---:|---:|---:|---:|
| 9 | 66.2 / 279.3 | 85.9 / 224.7 | **104.2 / 187.8** |
| 10 | 66.2 / 279.2 | 79.4 / 243.1 | **98.1 / 199.2** |
| 11 | 66.2 / 279.3 | 73.5 / 262.8 | **91.2 / 214.2** |
| 12 | 66.2 / 279.3 | 134.6 / 141.9 | **137.8 / 140.7（2.08×）** |
| 16 | 65.8 / 281.1 | 56.0 / 345.5 | 70.0 / 279.1 |

M=12 的 2× 主要来自 EXACT 定宽：去掉 4 个死行槽后，守卫分支消失，MAC
发射从 16 槽降到 12 槽；非 EXACT 的 M=9..11 只有 1.4–1.6×，M=16 打平。
rows=8 优于 16（更多 block、更好占用），rows=4/32/64 都因 A 重铺或寄存器
压力更差。

### 9.2 服务 A/B（t1int8 + int8 KV + NOCACHE，512 prompt / 128 gen）

| 臂 | c=1 ITL / 聚合 | c=2 ITL / 聚合 | c=4 ITL / 聚合 |
|---|---:|---:|---:|
| 现役 rows=4 | 35.93 ms / 52.8 t/s | 55.04 / 60.1 | 125.28 / 52.5 |
| **ms16** | 36.05–36.27 / 52.5–52.8 | 55.08–55.27 / 60.1–60.3 | **82.65–82.80 / 70.1–72.4** |

- **c=4：ITL −33.9%，聚合 +33~38%**（两轮重复一致）；
- c=1/c=2 在噪声内（+0.3~0.9%），c=2 远低于 +2% veto 门；
- decode-only profiler：设备时间 6093 → 4189 ms（−31%），tile 臂归零，
  GEMV stream 占 60%（M=12 单次 193 µs）。

### 9.3 数值与正确性

- **M≤8 不变量**：11 形状 × M=3/6/8 的 out_md5 与生产 `.so` 全同（逐位）；
- **M=9..16**：rel 2.2e-4–4.3e-4，与 tile/生产同带；M=12 从 tile 换 GEMV
  后不是逐位（归约序不同），这是本臂唯一的数值口径变化；
- **greedy parity**：c=4、4 prompt × 32 token、温度 0，4/4 输出文本完全一致。

### 9.4 事故与边界

- **profiler 收尾后请求会卡死（已复现 2 次）**：`/start_profile` →
  `/stop_profile` 之后的第一个/第一批请求会让引擎停在 running（GPU 空转、
  `shm_broadcast` 反复超时），只能 kill -9 重启；干净启动、不碰 profiler
  时同一工作负载正常。上线实例**不要开 profiler**；需要 profile 时先用
  独立实例，测完整体重启。
- 变体 `.so` 由手工重链产出，**生产 `.so` 未改**；正式采纳需在 S90 里
  重建生产扩展、跑完整 G1–G5/长跑/显存门。

### 9.5 判决

**G2 成立（采纳候选）**：B=4 的 tile 段被现役同族内核以 2× 替换，端到端
ITL −34%、聚合 +35%，B≤2 逐位不变。这是 S89 原表“S90 建议 1”的实测兑现；
int8 线保持关闭。

### 9.6 上线记录（2026-11-12 15:26 UTC）

- 生产软链：`vllm/_rocm_C.abi3.so → .s89/arms/_rocm_C_ms16.abi3.so`；
- 启动：`.s89/start_live.sh`（生产 argv/env + `VLLM_DCU_FP8_GEMV_MAX_M=16`
  + 新编译缓存根 `/mnt/game/vllm-cache-g2`），`--host 0.0.0.0:8000`；
- 实测：c1 35.93 ms / 53.4 t/s，c2 54.9 / 59.4，c4 **83.6 ms / 70.9 t/s**；
- 回滚：`ln -sfn .s85/arms/build_t1int8/_rocm_C.abi3.so vllm/_rocm_C.abi3.so`
  并重启（旧 `.so` 未动，md5 `188b6e3aae53`）。

### 9.7 回归与修复（2026-11-13）：G2 只在手工启动器里，unit 重启会丢

**现象**：11-13 02:40 线上实例做完 torch profiler 后请求挂起（§9.4 已知
bug），运维 `systemctl restart z100-vllm-mtp2` 重启后 c=4 ITL 从 82.8 ms
回落到 **98–125 ms**、聚合 67→52 t/s，c=1/c=2 与基线一致（35.8 / 54.9 ms）。

**根因**：G2 的 `GEMV_MAX_M=16` + `VLLM_CACHE_ROOT=/mnt/game/vllm-cache-g2`
只写在 `.s89/start_live.sh`（手工启动器）里，systemd unit 从来没有。而
`vllm/_rocm_C.abi3.so` 软链是**全局持久**的，于是重启后出现半 G2 状态：
ms16 内核在、env 却回到 `GEMV_MAX_M=8`，M=12 掉回 tile GEMM（66 GB/s）。
这是 S89 上线记录只写了“启动器”没写“unit”的资产遗漏，不是内核回归。

**修复**：新增
`/etc/systemd/system/z100-vllm-mtp2.service.d/g2-ms16.conf`，把两个 env
固化进 unit（注释含回滚法），`daemon-reload` 后 `systemctl show` 确认生效。

**验收**（重启后，同一 `svc_bench.py` 口径）：

| 窗口 | 降级时 | 修复后 | G2 原始 |
|---|---:|---:|---:|
| c=1 ITL | 35.78 ms | 35.77 ms | 35.93 ms |
| c=2 ITL | 54.94 ms | 54.75 ms | 55.04 ms |
| c=4 ITL | 97.99–100.63 ms | **82.49 ms** | 82.80 ms |
| c=4 聚合 | 67.6 t/s | **68.3 t/s** | 70.9–72.4 t/s |

**运维注意（两条）**：

1. 路由日志 `gfx906 fp8-emu route: ... GEMV_MAX_M=8` 在 G2 下**照旧打印 8**，
   因为 python 侧 `gfx906_fp8.py` 的 `_GEMV_MS_MAX` 硬钳在 8（对旧 `.so`
   的安全网，`env>kMsMax` 会把 M=12 递进 `fp8_emu_gemv_block` 的
   `M must be in [1,8]` TORCH_CHECK）。**不能用这行判断 G2 是否生效**，
   要看 `/proc/<pid>/maps` 里的 `.so` + `/proc/<pid>/environ` 的
   `GEMV_MAX_M`，或直接跑 c=4 ITL（82 ms = G2 在，98+ ms = 掉了）。
2. 线上实例**不要开 profiler**（§9.4）；要 profile 另起实例，测完重启。
