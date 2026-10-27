# PERF1 推荐生产配方

**基线**：`/mnt/game/port/serve_fp8_8000.sh`（k=2、`VLLM_DCU_FP8_GEMV_MAX_M=1`、`S11F_BF16_GEMV=0`）
**目标 .so**：`_rocm_C.abi3.so` md5 `8e98bd60b94ebeb6f69ed8e39d754647`（本文件所有数字的二进制）

---

## 1. 配方差异（4 处，均可回滚）

| # | 项目 | 旧 | 新 | 依据 |
|---|---|---|---|---|
| 1 | FP8 decode 派发 | `VLLM_DCU_FP8_GEMV_MAX_M=1`（M≥2 全走 tiled GEMM） | `S11F_MTP_GEMM=0` + `VLLM_DCU_FP8_GEMV_MAX_M=4` | M=3 单次调用 355 → 84 µs（N=5120） |
| 2 | 稠密 fp16/bf16 decode GEMV | `S11F_BF16_GEMV=0`（整条路径关闭） | `S11F_BF16_GEMV=1` | 单变量 arm：step 190.9→179.3 ms、decode 15.93→16.90 t/s |
| 3 | GEMM tile 自动规则 | cfg0 → (M≤32?1:2) | cfg0 → cfg5(M≤32) / cfg10(N≥6144) | prefill M=16..129 **1.09–2.00×**，TTFT −17.6 % |
| 4 | MTP 深度 | k=2 | **k=2（不变，实测 k=3 更差）** | 全长度扫描：k=2 均值 18.19 vs k=3 17.36 t/s |
| 5 | 启动预热 | 无 | `/health` 就绪后跑 `tools/dcu/perf1_warmup.py --lens 8192,2048,512,128` + 丢弃一次 decode | 冷启动 4.18× 惩罚 |

配方落地开关（`tools/dcu/s11f_e2e_fp8.sh` 已实现，**环境变量显式值优先**）：

```bash
export S11F_MTP_GEMM=0            # 1 = 回滚到 perf1 之前的 M≥2 → tiled GEMM
export S11F_BF16_GEMV=1           # 0 = 回滚稠密 GEMV
export VLLM_DCU_FP8_GEMV_MAX_M=4  # 1 = 精确复现旧生产行为；缺省 4
export S11F_MTP_TOKENS=2          # 保持 k=2
```

`/mnt/game/port/serve_fp8_8000.sh` 需要改的一行（它目前硬编码 `S11F_BF16_GEMV=0`）：

```diff
-export S11F_BF16_GEMV=0
+export S11F_BF16_GEMV=1
```

## 2. 实测（锚点协议 p451/g64×3，temp=0，并发 1，`8e98bd60`）

### 2.1 各层归因（单变量 arm）

| arm | step p50 | decode t/s | TTFT ms | prefill t/s |
|---|---:|---:|---:|---:|
| 旧配方（`fbfd45ed`，MAX_M=1） | 214.9 | 14.18 | 2046 | 220.4 |
| 新 .so + 旧派发（含 cfg 自动规则） | 190.9 | 15.93 | 1685 | 267.6 |
| 新 .so + M≥2 走流式 GEMV（MAX_M=4） | 135.0 | 21.30 | 1658 | 272.2 |
| 对照：MAX_M=4 但**旧** GEMV | 359.3 | 8.48 | 1852 | 243.5 |
| 稠密 GEMV 路由（`BF16_GEMV=1`，MAX_M=1 隔离） | 179.3 | 16.90 | 1671 | 270.0 |

⇒ 收益来自**内核**而非阈值（只抬阈值不用流式内核会慢 88 %）。

### 2.2 最终配方全长度扫描（`combined_k2`：新 .so + MAX_M=4 + BF16 路由 + k=2）

| 真实 prompt | TTFT 基线 → 最终 | prefill t/s 基线 → 最终 | decode t/s 基线 → 最终 | step p50 基线 → 最终 |
|---|---|---|---|---|
| 119 | 873 → **680** | 137.8 → **174.9** | 10.60 → **20.39（+92 %）** | 209.5 → **138.8** |
| 451 | 2046 → **1676** | 220.5 → **269.9** | 13.93 → **22.25（+60 %）** | 215.3 → **133.6** |
| 1806 | 5010 → **4124** | 388.2 → **471.9** | 11.85 → **18.34（+55 %）** | 238.1 → **147.8** |
| 7443 | 7089 → **5977** | 1049.9 → **1245.2** | 7.78 → **11.79（+52 %）** | 333.3 → **239.2** |
| 均值 | 3755 → **3114（−17 %）** | 449 → **540（+20 %）** | 11.04 → **18.19（+65 %）** | 249 → **165（−34 %）** |

## 3. 已知悬崖与约束

* **投机深度悬崖**：M=1..4 走流式 GEMV（N=5120：61/76/84/126 µs），**M≥5 回退 tiled GEMM**
  （321 µs）。因此 `num_speculative_tokens=4`（M=5）会突然变贵；k≤3 安全。
* **显存/长度**：配方沿用 `maxlen 262144`、`util 0.85`。
* **冷启动**：新引擎首请求 p8192 TTFT **29.7 s vs 稳态 7.1 s（4.18×）**；`--enforce-eager`
  会硬关 JIT warmup（`vllm/config/vllm.py:1793-1803`），所以必须在 `/health` 就绪后
  显式跑一次预热（`tools/dcu/perf1_warmup.py`），且按 prompt 长度桶各预热一次
  （首个新长度的请求仍慢 22–62 %）。
* **首次部署**：换 .so 后必须重启服务；`build-211.sh` 只链接，**必须再跑 `install-211.sh _rocm_C`**。

## 4. 未采纳（已实测证伪或收益不足）

| 方向 | 证据 | 处置 |
|---|---|---|
| MTP k=3 | 全长度均值 k=2 更优 4.8 %（18.19 vs 17.36 t/s） | **保持 k=2** |
| MBT 8192 | NULL（step 190.72 vs 190.94 ms） | 保持 2048 |
| GDN 线性注意力 Triton 内核 | 内核 705→19.5 µs（36×），e2e A-B-A +10.8 ms/step（CPU 启动受限） | **不采纳**，开关默认关 |
| `NCCL_PROTO=LL` | 干净复测：解码尺寸无收益（85.7 vs 84.6 µs），20 MiB 慢 2.35× | 保持 protocol auto |
| custom all-reduce | 平台结构性未启用，S13 实测 ±0.4 % | 关闭 |
| `lm_head` 走模拟 GEMV | M=3/4 反而更慢（1876 vs 1856） | 保持 Triton（M≤2 已走 GEMV） |
| `lm_head` 量化 FP8 | 需改 checkpoint/quant config | 独立课题（636 MB/rank/次，384 GB/s 已近 roofline） |
| CUDAGraph 降 launch 开销 | 空档 16 ms/step、CPU 有 2.6× 余量；且 graph 路径 token 不可复现 | 保持 `--enforce-eager` |
| 混插卡（2×Z100L+2×Z100）同步成本 | 7–18.9 ms/step，无软件解 | 记录为硬件约束 |
| 降权重位宽换带宽 | 删掉 100 % global load 不加速 | 不做 |

## 5. 下一步（瓶颈已转移）

decode step 219.6 → ~135–148 ms 后，逐层 host 时间（48 × 2.83 ms ≈ 136 ms）已与 step 同量级：
**启动开销现在是第一瓶颈**。下一步应做 graph capture / 减少每层 launch 数，
届时 GDN 内核（已就绪、env 开关）可兑现 −33 ms/step，`kStreamMinBlocks=100` 也可一并落地。
