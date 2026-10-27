# S23 — 用 fp16 模拟 fp8（gfx906 FP8 emu）各流程阶段的固定用时调研

> 平台：4×海光 DCU Z100（gfx906），2×Z100L@1600 MHz + 2×Z100@1319 MHz，TP=4
> 仓库：`/home/lyh/z100-vllm`（vLLM fork）
> 被测构建：`vllm/_rocm_C.abi3.so` = md5 **`f205be8f43d3ffb1cfb3c6f093bfab4b`**（= 生产 live 构建）
> 生产服务：systemd `z100-vllm-host.service`（PID 2622752，**全程未重启/未改配置**）
> 模型：`/mnt/game/models/Qwen3.8-27B-FP8`（fp8 e4m3，`weight_block_size [128,128]`，dynamic 激活）
> 观测时间：2026-10-19 07:20–08:20 UTC
> 配套材料：`.s23/fp8_emu_stage_inventory.md`（T1 静态）、`.s23/fp8_emu_kernel_fixed_times.md`（T2 微基准）、
> `.s23/fp8_emu_trace_attribution.md`（T3 trace）、`.s23/PROBE_ENV.md`（环境事实）

---

## 0. 一句话结论 + 固定用时总表

**Z100 上没有 fp8 硬件，FP8 权重靠「内核内 fp16 LUT 加宽 + `v_dot2_f32_f16` fp32 累加」模拟。
这条流程真正的固定成本不在 MAC，而在「每次调用的派发/启动 + 权重流式读取」：
生产 decode 步（M=8，256 次调用）里 fp8 GEMM 阶段耗时 ≈ **43–45 ms/步**，其中
**≈ 31 ms/步（69–72%）与 M 无关**（N=768 时单次调用就已经要 179 µs，N=8704 也只要 241 µs，
N 涨 11 倍时间只涨 35%）；再加上 activation 量化 2.2 ms/步（完全固定）、
GPU 可见的 launch 气泡 4.3–12.9 ms/步、以及 host 侧 22–31 ms/步的派发税，
整个 fp8 仿真流程的固定部分 ≈ **设备侧 38–46 ms/步 + host CPU 22–31 ms/步（二者部分重叠）**，
合计占 194–270 ms 步进的 **17–25%**。
固定项可以压的只有「派发税」（每次 ~30–55 µs 的 `torch.library` 包装）与「每个 block 的 LUT staging/ramp」，
权重流式读取是物理成本，只能被更好的 tile 摊薄。**

### 0.1 每阶段固定用时总表（生产 decode，c=1，M = 1+k = 8）

| 阶段 | 每次调用的固定用时 | 其中「与 M 无关」的部分 | 每步（256 次） | 证据强度 |
|---|---|---|---|---|
| **S1 activation 量化**（`per_token_group_quant_8bit_register_kernel`，K=5120，group=128） | GPU **8.4 µs**（8.0–10.2，M=1..8 全平）；host **43 µs** | 全部（M 无关） | GPU **2.15 ms**；host **11.0 ms** | 实测（census+graph+host） |
| **S2 host dispatch**（`vllm::fp8_emu_mm_block` → `_rocm_C::fp8_emu_gemv_block`） | custom_op **43–83 µs**（窗口内中位 ~77）；裸 `_rocm_C` op **12–23 µs** | 除入参一维检查外全部 | **11.0–19.7 ms**（裸 op 3.1–5.9） | 实测（4 轮交错） |
| **S3 A 加宽 / staging**（无独立 kernel！） | tile 路径：包含在 S4 内，**不是固定项主体**（M=1..8 耗时 ±2.5%）；streaming 路径：体现为**带宽塌缩 488→116 GB/s**，固定项仅 +3 µs | — | 计入 S4 | 实测 + 机制推断 |
| **S4 权重流式读取 + LUT 加宽**（形状项） | tile cfg16：`t = 171.6 µs + W/779 GB/s`；streaming K=1536：`t = 9.75 µs + W/116 GB/s` | **拦截项 171.6 µs 与 M 无关**（∝N 但 M 平坦） | **31.1 ms**（固定部分） | 实测（N 扫描拟合 r²=0.94–0.99） |
| **S5 dot2 MAC** | ∝ M ⇒ **不是固定项** | 0 | — | 实测（M 斜率） |
| **S6 epilogue**（折 scale / cast / bias） | `bias=None`（生产）⇒ **0**；out fp16→fp32 **+0.15–0.32 µs**；带 bias **+2.2–2.4 µs** | 全部 | ≈ 0 | 推断（≤±2.4 µs 量级） |
| **S7 输出视图**（`.to()` 同 dtype + `.view()`） | **0**（no-op） | — | 0 | 静态 + trace（无 cast/copy kernel） |
| **S0 LUT 构建 + 上传**（256×fp16 = 512 B） | **一次性**，`lru_cache(8)/(16)` 命中后 0 | — | 0 | 静态 |
| **合计（设备侧固定）** | — | — | **GEMM 31.1 + quant 2.15 = 33.3 ms/步** | — |
| **合计（host 侧固定）** | 55–106 µs/次 | — | **22–31 ms/步** | — |
| **GPU 可见 launch 气泡**（trace 实测） | 50.2 µs/次（legacy 构建）/ 16.8 µs/次（cfg16 构建） | 次数固定、时间随 host 争用变化 | **12.86 / 4.29 ms/步** | trace 实测 |

### 0.2 三条必须记住的口径

1. **「固定」有两层含义**，本文分开报：**(a) 与 M 无关**（同一次调用内，batch/verify 行数不影响的部分）；
   **(b) 与并发无关**（每步调用次数固定 256 次，与 c=1/4/16 无关，trace 实测一个不差）。
2. **权重流式读取 ∝ N×K，与 M 无关** ⇒ 它属于 (a) 意义上的固定项，但**不可能消除**（fp8 权重必须从 HBM 读）。
   真正可优化的是「拦截项 171.6 µs」与「host 派发税」。
3. **生产 M=8 的 256 次调用分两条路径**：192 次 cfg16 精确-M tile（`fp8_emu_gemm_block_kernel<8,64,1,2,Half,1>`）
   + 64 次 streaming GEMV（`fp8_emu_gemv_stream_block_kernel<8,3,Half>`，仅 `5120:1536` 命中）。
   入口 op 是 `_rocm_C::fp8_emu_gemv_block`（不是 `..._gemm_block`）。

---

## 1. 流程定义

### 1.1 「用 fp16 模拟 fp8」到底模拟了什么

gfx906 没有任何 fp8 指令（`v_cvt_f32_fp8`/`v_cvt_pk_fp8_f32`/fp8 MFMA 全部 ❌，见
`docs/archive/result/z100-11d-z100-isa-capability-audit-result.md`），也没有 bf16 运算指令。
本项目的做法是（`csrc/rocm/fp8_emu_gemm.cu` + `csrc/rocm/gfx906_emu_common.cuh`）：

* **权重常驻 fp8，永不解量化**（27B FP8 ≈ 27 GiB，fp16 副本 54 GiB 装不下）；
* 内核里用 **256 项 fp16 LUT（512 B，常驻 LDS，`ds_read_u16`）** 把 fp8 码点无损加宽成 fp16
  （e4m3 峰值 448、e5m2 峰值 57344，全部有限码在 fp16 里精确可表示 ⇒ 加宽零误差）；
* MAC 用 **`v_dot2_f32_f16`**（1 条指令 2 个 fp16 MAC、fp32 累加），正是 fp8 想要的累加口径；
* scale 在 epilogue 里只乘一次（block-scaled 变体按 K 方向每 128 折一次 `a_s·w_s`）。

### 1.2 端到端阶段图（dense linear，block-scaled 变体 = 生产实际路径）

```
Fp8LinearMethod.apply                                        vllm/model_executor/layers/quantization/fp8.py:418-461
└ Gfx906Fp8BlockScaledMMLinearKernel.apply_block_scaled_mm   vllm/.../scaled_mm/gfx906_fp8.py:400-437
  ├ S1 量化   QuantFP8.forward_hip → per_token_group_quant_fp8
  │           vllm/.../quantization/input_quant_fp8.py:136-162 → fp8_utils.py:708-719
  │           → _C::per_token_group_fp8_quant
  │             → per_token_group_quant_8bit_register_kernel（register 路径，2026-10-17 上游 sync 引入）
  │             csrc/quantization/w8a8/fp8/per_token_group_quant.cu:189,309-322,471-479
  ├ S0 LUT  _device_lut(A.dtype, A.device)                   gfx906_fp8.py:158-166（lru_cache，512 B）
  ├ S2 派发 torch.ops.vllm.fp8_emu_mm_block(...)              gfx906_fp8.py:222-243（custom_op，opaque body）
  │         → _rocm_C::fp8_emu_gemv_block(...)                csrc/rocm/fp8_emu_gemm.cu:1686-1697
  │           ~20 条 TORCH_CHECK / 每调用一次 getenv+atoi      fp8_emu_gemm.cu:1216-1242, 1493-1502
  ├ 分支   M<=8 && stream_block_select 命中 → streaming; 否则 → cfg16 tile
  │                                                          fp8_emu_gemm.cu:777-811, 1529-1550, 1559
  ├ S3 A 加宽（在 kernel 内，无独立 kernel launch）
  │         tile: LDS staging 直接 LUT 加宽                    fp8_emu_gemm.cu:273-302, 1046-1078
  │         stream: 每 block A→LDS 一次加宽                     fp8_emu_gemm.cu:644-654
  ├ S4 权重流式读取 + LUT 加宽（global→reg→LDS 双缓冲）        fp8_emu_gemm.cu:211-330, 589-593
  ├ S5 dot2 MAC（fp32 累加）                                   v_dot2_f32_f16
  ├ S6 epilogue：acc*a_s*w_s → cast(out_dtype) → bias          fp8_emu_gemm.cu:1082-1100, 1181-1194
  └ S7 返回 out.view(*output_shape)（no-op）                   gfx906_fp8.py:437
```

### 1.3 两条 dispatch（同一入口，C++ 内分支）

| 条件 | 实际 kernel | 生产命中 | 依据 |
|---|---|---|---|
| `M ≤ VLLM_DCU_FP8_GEMV_MAX_M(=8)` 且 streaming 门限满足（A 加宽后 LDS ≤ 44 KiB） | `fp8_emu_gemv_stream_block_kernel<M,3,Half>` | **64 次/步**（仅 `5120:1536`，K=1536，LDS 24960 B） | `fp8_emu_gemm.cu:777-811,1529-1535`；trace 实测 |
| `M ≤ 8` 且 streaming 不满足 | `fp8_emu_gemm_block_kernel<8,64,1,2,Half,1>`（cfg 16） | **192 次/步**（其余 4 个形状，K=5120/4352 需 83200/70720 B） | 同上；census 实测 |
| `M > 8`（prefill） | `fp8_emu_gemm_block_kernel<32,64,2,4,Half,{1,4}>`（cfg 5 / cfg 10，`_prefill_cfg=0` auto） | prefill 步 | 实测（§3.4） |

---

## 2. 方法（可复现、口径、误差）

### 2.1 GPU 探针入口

DSH 沙箱拒绝 `open("/dev/kfd")`（宿主 `.venv` 的 torch 看到 0 卡），唯一可用入口是 docker：

```bash
docker exec z100-regress bash -c 'cd /src && DTK_ROOT=/mnt/game/host-runtime/dtk2604 bash -c "
set +u; source tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install HIP_VISIBLE_DEVICES=0
export VLLM_DCU_USE_FP8_EMU=1 VLLM_DCU_FP8_GEMV_MAX_M=8 VLLM_DCU_FP8_DECODE_CFG=16
.venv/bin/python .s23/logs/kernel/lead_confirm.py
"'
```

* `import torch` 之后必须 **`import vllm._rocm_C` 与 `import vllm._C`**：裸 `import vllm` 不会注册
  `torch.ops._rocm_C.fp8_emu_*` / `torch.ops._C.per_token_group_fp8_quant`（惰性导入）。
* 生产池下 **hipBLAS 不可用**（`HIPBLAS_STATUS_ALLOC_FAILED`），参考实现禁止用 `@`/`einsum`。
* 显存：`mem_get_info()` 报 0 B free，但 ≤64 MiB 分配可用；256 MiB 即 OOM ⇒ 一次一个形状，测完 free。

### 2.2 三种计时口径（这是本次能给出可信数字的关键）

| 口径 | 做法 | 适用 | 已知偏差 |
|---|---|---|---|
| **A per-call HIP event** | 每次调用一对 event | ≥100 µs 的 kernel | **对 <30 µs 的 kernel 有 +20~35 µs 正偏**（空 event 对本身 20.3 µs） |
| **B amortized** | N 次背靠背只用一对 event | 稳态吞吐 | 长窗口会被生产抢占抬高（作**上限**用） |
| **G HIP graph replay** | 捕获 N 次调用后重放 | 纯设备时间（去掉 host 发射） | 无 |
| **C profiler census** | `torch.profiler` CUDA activity 单 kernel 均值 | 任何长度 | profiler 略有膨胀（+3~10%） |
| **HOST** | 无 sync 的入队 wall time | host 派发 | 受宿主机 CPU 争用影响（见 §3.2） |

本次定案的设备侧数字以 **G + C** 为准，A/B 只作对照。

### 2.3 canary（证明没被生产抢占）

`8704:5120`、M=1、dense stride，参考值 **0.10625 ms**（`tools/dcu/s20_kernel_NOTES.md` §2）：

| 测量 | canary（graph） | 占参考 |
|---|---|---|
| T2（17 次，全程） | 0.0975–0.1090 ms | 92–103%（part5 中段一次抢占窗口已剔除） |
| Lead 复核（首/中/末） | 0.10522 / 0.10519 / 0.10816 ms | 99.0 / 99.0 / 101.8% |

⇒ 两轮独立测量都在参考值 ±3% 内，**没有系统性抢占**。

### 2.4 第三张独立复核：Lead 自己重写了一份 trace 统计脚本

为避免「同一份脚本自证」，Lead 用一份独立实现（`.s23/logs/trace/lead_spotcheck.py`，
只按 `ProfilerStep#` 区间 + kernel 名做 phase 归类）重跑 trace A，与 T3 的逐次调用值对照：

| 阶段 | Lead spot-check µs/次 | T3 µs/次 | 差 |
|---|---|---|---|
| `fp8_emu_gemm_block_kernel` | 296.07 | 296.35 | −0.09% |
| `per_token_group_quant_8bit_kernel` | 9.20 | 9.21 | −0.11% |

（每步 ms 未对齐：我的独立脚本步区间切法更粗，93 步/220.8 ms vs T3 的 48 步/269.5 ms，
因此 n/step 只有 246.7 —— 这是**窗口切法**差异，不影响逐次调用的对照结论。）

---

## 3. 逐阶段固定用时明细

### 3.1 S1 activation 量化：**每次 8.4 µs，与 M 完全无关**

| 项 | 实测 | 来源 |
|---|---|---|
| kernel 名（live 构建） | `per_token_group_quant_8bit_register_kernel<c10::Half, c10::Float8_e4m3fn, false, false, 128>` | census 实测 |
| GPU 时间（graph，M=1/4/8） | 7.54 / 8.50 / 8.09 µs | Lead `C5_quant` |
| GPU 时间（graph，M=1/8） | 8.48 / 7.51 µs | T2 `p6_quant_raw_M1/M8` |
| GPU 时间（census，M=1/4/8） | 8.99 / 10.18 / 9.84 µs | Lead `C5_quant` |
| 裸 op host（M=1/4/8） | 8.07 / 7.80 / 7.57 µs | Lead `C5_quant` |
| **生产 python 包装** `per_token_group_quant_fp8(x,128)` host | **43.5 / 51.2 / 40.5 µs**（T2：46.3 / 45.4） | Lead + T2 |
| **每步 ×256** | GPU **2.15 ms**、host **11.0 ms** | [估算] |

* **M 平坦性证据**：M 从 1 到 8（8×），GPU 时间 8.0→8.1 µs ⇒ 这一阶段是**纯固定开销**；
  trace 里跨 c=1/4/16 与 5 档并发扫描（M=8…48，6× 变化）也在 2.14–2.36 ms/步之间**无趋势**（T3 §3.2）。
* **S20 的 9.21 µs/call 已过时**：那是 2026-10-14 构建的 `..._8bit_kernel`（smem 路径）；
  当前树的 register 路径是 `bb7e8dae1`（2026-10-17 上游 sync）引入的，名字与耗时都变了（此处 GPU 时间相近，
  但 host 侧整条量化阶段是 **43 µs**，S20 只报了 kernel 的 9.21 µs）。

### 3.2 S2 host dispatch：**每次 55–106 µs（其中纯 Python 包装 31–55 µs）**

| 项 | 实测（4 轮交错，同窗口） | T2 实测 | 说明 |
|---|---|---|---|
| 裸 `_rocm_C::fp8_emu_gemv_block`（M=1） | 11.6 / 20.3 / 22.8 / 20.4 µs | 13.67 µs | 纯 ATen+HIP 入队 |
| 裸 `_rocm_C::fp8_emu_gemv_block`（M=8） | 11.6 / 23.1 / 23.0 / 21.4 µs | 13.76 µs | 与 M 无关 ⇒ M=1/M=8 相同 |
| `torch.ops.vllm.fp8_emu_mm_block`（M=1） | 43.2 / 78.1 / 77.7 / 78.0 µs | 48.0 µs | **custom_op 包装** |
| `torch.ops.vllm.fp8_emu_mm_block`（M=8） | 54.0 / 76.7 / 79.3 / 76.5 µs | 48.25 µs | 同 |
| **每步 ×256** | **11.0–19.7 ms**（裸 op 只占 3.1–5.9 ms） | 12.3 ms | [估算] |

* **纯包装开销 = custom_op − 裸 op ≈ 31–55 µs/次**，来自 `torch.library` 的 schema 校验、dispatcher、
  opaque body 的 Python 帧；**内核怎么优化都消不掉**。按 256 次/步 ⇒ **8–14 ms/步**。
* 测量噪声说明：同一窗口的 4 轮里第 1 轮明显更快（host CPU 尚空闲），之后稳定在 ~77 µs；
  **两轮独立测量给出 43–83 µs 的区间**，生产引擎线程占满 CPU 是主因。取中位 **~77 µs（Lead 稳定窗口）/
  ~48 µs（T2 短窗口）**，报告一律给区间。
* 这一点与 trace 的**独立**证据吻合：GPU 侧可见的 fp8 相邻空隙 **50.2 µs/次**（legacy 构建，T3 §4.3），
  与 host 侧 43–83 µs 同一量级 ⇒ 这 50 µs 是 host 派发，不是 GPU 延迟。

### 3.3 S3/S4 emu GEMM（decode M=8）：**首拦截 171.6 µs + 权重/779 GB/s**

**N 扫描拟合**（K=5120、M=8、cfg16 tile，profiler census，5 点）：

| W (MB) | 3.93 | 7.86 | 15.73 | 26.21 | 44.56 |
|---|---|---|---|---|---|
| t (µs) | 179.4 | 186.0 | 185.4 | 200.5 | 232.8 |

拟合 **`t = 171.6 µs + W / 779 GB/s`，r² = 0.943**（T2 独立拟合：**`169.6 µs + W / 643 GB/s`，r²=0.948**）。
两个独立实现的拦截项相差 **1.2%**，取 **t_fixed ≈ 170 µs**。

> **这条拟合是本调研最重要的发现**：N 从 768 涨到 8704（11.3×、权重 3.9 MB→44.6 MB），
> 单次调用只从 179 µs 涨到 233 µs。也就是说 decode 路径的 emu GEMM **几乎是纯固定成本**——
> 它在等自己的 ramp（每个 block 的 LUT staging + 双缓冲填充 + 尾部归约），不是在等 MAC 算力。
> 与 `.s20/S20_KERNEL_RESULT.md` §1 的 occupancy/延迟受限结论一致。

**M 平坦性**（同一 shape，M=1/2/4/8）：

| shape | M=1 | M=2 | M=4 | M=8 | 说明 |
|---|---|---|---|---|---|
| 8704:5120 | 105.8（stream） | 133.5（stream） | 234.0（stream） | **232.8（cfg16 tile）** | M=4 走 stream 反而比 M=8 的 tile 慢 |
| 5120:1536 | 26.6（stream） | 36.8 | 52.7 | 105.7 / 85.2（两次窗口） | 唯一走 stream 的生产形状 |

⇒ cfg16 tile 在 M=1..8 上**完全平坦**（T2：239.7/239.8/240.6/245.8 µs，±2.5%），
**A 加宽（S3）不是固定项主体**；streaming 路径的 fixed 只有 +3 µs，它的代价是带宽塌缩。

**备注**：`fp8_emu_widen_a_kernel` 在**生产两条路径上都不发射**（发射条件
`gemv && !use_stream && !gemv_use_gemm && M>=2` 恒假，`fp8_emu_gemm.cu:1550`）；
7 份 trace 中该 kernel 出现 **0 次**，census 里也只看到一个 kernel。A 加宽实际发生在
tile 的 LDS staging 与 streaming 的 per-block staging 内。

### 3.4 prefill（M>8）的固定用时：**221–308 µs，比 decode 更高**

cfg=0 auto（python 默认），`fp8_emu_gemm_block` 入口，census：

| shape | M=16 | M=64 | M=256 | M=512 | 拟合 |
|---|---|---|---|---|---|
| 8704:5120（cfg10 `<32,64,2,4,Half,4>`） | 481.3 | 821.0 | — | 4726.6 | **t = 308.1 µs + 8.62 µs/行**，r²=0.9998 |
| 5120:4352（cfg5/10） | 327.4 | 444.1 | 1299.3 | 2371.2 | **t = 221.3 µs + 4.19 µs/行**，r²=0.9986 |

* prefill 的拦截项（221–308 µs）≈ 「权重串行读一遍」的时间（22.3 MB/169 µs ≈ 132 GB/s、
  44.6 MB/308 µs ≈ 145 GB/s），比 decode 的 170 µs 更大（cfg10 MINB=4、流水填充更慢）。
* prefill 的每行斜率 4.19/8.62 µs 才是真正∝M 的部分；按 `--max-num-batched-tokens 8192`，
  一个 prefill 步每层的固定成本会被 ~8192 行摊薄，因此 **prefill 侧固定成本不是问题**，
  decode（M=8，无处摊销）才是。

### 3.5 S6/S7 epilogue 与输出

* `bias=None`（生产，`gfx906_fp8.py:432` 传 `None`）⇒ 实测 bias 分支的 +2.2–2.4 µs **不发生**。
* `out_dtype` = fp16 = 输入 dtype ⇒ `.to()` 是 no-op；`.view()` 是 no-op。
* trace 侧独立确认：`vllm::fp8_emu_mm_block` 子树内**没有**任何 cast/copy kernel（T3 §2.3）。

---

## 4. 每步汇总（256 次调用）与 S20 历史数据对齐

### 4.1 生产 decode 步（c=1，M=8）的 fp8 流程预算

| 项 | 每步 | 占 269.5 ms 步进（legacy trace）/ 194 ms（live p50） |
|---|---|---|
| S1 量化 GPU | 2.15 ms | 0.8% / 1.1% |
| S3+S4+S5 emu GEMM（live，cfg16+stream） | **44.9 ms**（Lead）/ 42.8 ms（T2） | 16.7% / 23.1% |
| ├ 其中 **与 M 无关的固定部分** | **31.1 ms（69.4%）** | 11.5% / 16.0% |
| └ 其中 ∝M 的 dot2 部分 | 13.7 ms | 5.1% / 7.1% |
| S2 host 派发（CPU） | 22–31 ms | 8–12% / 11–16% |
| GPU 可见 launch 气泡 | 4.3 ms（cfg16 构建）/ 12.9 ms（legacy 构建） | 1.6% / 2.2% |
| **fp8 流程固定合计** | **设备侧 33.3 + 可见气泡 4.3–12.9 = 38–46 ms**，host CPU 另 22–31 ms（部分重叠） | **17–25%** |

### 4.2 与 S20 的历史数字对齐（逐项解释差异，不是矛盾）

| 量 | S20/S21 记录 | 本次（同构建口径） | 差异原因 |
|---|---|---|---|
| fp8 GEMM ms/步 | 75.9（legacy cfg5/10，Oct-14） | **44.9 / 42.8**（cfg16+stream，live） | 就是 S20 那次 tile 修复的收益（−33 ms/步），**已兑现** |
| 量化 µs/次 | 9.21（`..._8bit_kernel`） | 8.4（`..._8bit_register_kernel`） | 2026-10-17 上游 sync 换了 kernel |
| host custom_op µs/次 | 58.6（trace 推算，含外层） | 43–83（直测，含/不含外层口径差） | 一致区间内 |
| GPU idle/步 | 54.9 | 55.74（trace 复现，逐项对上） | 同 trace 同窗口 ⇒ 复现成功 |
| 每步事件数 | 2509 | 2509.4 | 同上 |

**每步 ms 的三个历史口径**（74.6 / 77.8 / 75.87）差异来源已定位：74.6 = per-call **均值** 0.2915 ms × 256；
77.8 = per-形状 **median** 加权；75.87 = 本次 trace 48 步窗口直接实测（均值口径）。
三者是「均值 vs 中位 vs 窗口」的区别，不是测量冲突。

### 4.3 为什么「内核 −33 ms/步」端到端只有 −1…−7%（量化解释）

trace 实测（T3 §5，48 步窗口）：`period 269.5 ms = GPU busy 214.6 + idle 55.7`；
busy 里 fp8 78.2（36.5%）+ NCCL 73.6（34.3%）+ 其它 62.7（29.2%）。

* fp8 GEMM 只占 period 的 28.2% ⇒ 把它压到修复版 48.1 ms，**理论上界 −10.3%**；
* 同一 capture 内 48 步的相关性：`corr(period, idle) = +0.739`、`corr(period, gpu_ms) = +0.071`
  ⇒ **c=1 的步进方差由 idle/派发决定，不由 GPU 工作量决定**（c=16 时反过来：`corr(period, gpu_ms)=+0.799`）；
* **派发税不随 kernel 变快而减少**：仍是 512 次 launch + 256 次 custom_op / 步，
  其在步进中的占比反而从 16.4% 升到 21.3%。

---

## 5. 哪些固定成本不可消除 / 哪些可优化

| # | 固定成本 | 量级 | 可否消除 | 建议 |
|---|---|---|---|---|
| 1 | **权重流式读取** | 6.08 GB/步 ÷ 779 GB/s ≈ 7.8 ms（HBM 理论），实测占 GEMM 的 24–31% | ❌ 物理成本（fp8 必须从 HBM 读） | 只能靠更大 tile / 更好 occupancy 逼近带宽上限；已在 S20 的 cfg16 做过一轮 |
| 2 | **拦截项 171.6 µs/次**（每 block 的 LUT staging + 双缓冲 ramp + 尾部归约） | **31.1 ms/步 中的主体** | ⚠️ 部分可压 | 这是 **decode 的第一优化目标**：N=768 的空栅格也要 179 µs，说明每 block 的启动/填充固定开销没被摊掉。方向：合并 K 循环更深的单 block 覆盖多个 N-tile、LUT staging 与首个 W chunk 预取重叠 |
| 3 | **custom_op 包装 31–55 µs/次** | **8–14 ms/步** | ✅ 可消除（有限定条件） | 生产是 `--enforce-eager`，此时可绕开 `torch.library` 包装直接调 `_rocm_C` op；**compile 模式不行**——自定义 op 的 shape 契约是为了避免 Dynamo 给 token 维加 `Eq(dim, hint)` guard（S11k 教训，`gfx906_fp8.py:169-210`）。落地前必须验证 graph/eager 两种模式都不回退，预计回收 8–14 ms/步 |
| 4 | **量化阶段 host 43 µs/次** | **11.0 ms/步** | ⚠️ 部分 | `per_token_group_quant_fp8` 的 python 包装（含 3 次 `torch.empty`）可下沉到与 GEMM 同一个 custom_op，或换成 out-buffer 复用 |
| 5 | **量化 kernel 8.4 µs/次** | 2.15 ms/步 | ⚠️ 可融合 | `fuse_act_quant` 已开，但 fp8-emu 路径内部仍单独发一次；可尝试与上游 elementwise 融合（风险：tile 内嵌 quant 会改 delay） |
| 6 | **rdist launch 地板 1.8–2.3 µs/次**（空 kernel graph） | 0.5 ms/步 | ❌ | 只能靠减少调用次数（不是减少内核时间） |
| 7 | **NCCL 屏障 73.6 ms/步**（非本流程，但决定端到端上限） | 34.3% of busy，其中 83–88% 是到达自旋 | ❌ 硬件时钟差分（2×Z100L 1600 vs 2×Z100 1319） | 见 `.s20/S20_RANK_SKEW.md` |

**结论**：把「派发税 + 量化 host」合并下沉（#3+#4）可回收 **≈ 19–25 ms/步（7–13%）**，
且**风险低、不改数值**——这是当前最划算的一步；#2 需要内核工作，收益上界与权重带宽相关。

---

## 6. 未覆盖与后续建议

1. **live 构建上没有 trace**：生产服务不允许 re-profile，因此 §4.1 的 live 每步值是
   `T2/Lead per-call × 256 配比` 的 [估算]；trace 侧数字（§4.3）来自 Oct-14/16 的旧构建。
   若要 live trace，需要在维护窗口内以 `--enable-logging-iteration-details` 或 profiler 重启一次（有代价）。
2. **K=4352 的 tile 拦截项是残差推断**（133.4 µs，T2 §7）：没有做 `t_fixed(K)` 的 K 扫描。
3. **epilogue 的分离是推断**（±2.4 µs 量级），因为生产 `bias=None` 且 `out_dtype==输入 dtype`，本身为 0。
4. **`VLLM_DCU_FP8_GEMV_STREAM=2` / `GEMV_PREF` 变体未测**；`5120:1536` 的 streaming 在 M=8 时
   只有 116 GB/s（M=1 时 488 GB/s），A-LDS 16:1 的读放大是已知原因（S20 §1）。
5. **prefill M=1760/7920 未测**（`8704:5120` M=1760 需要 84 MB 驻留，超过生产池下的可用余量），
   用 M≤512 拟合外推。
6. **跨 rank 时序未展开**（只分析 rank0）；rank 间偏斜是屏障项的解释，已在 S20 单独结案。

---

## 7. 证据索引

| 结论 | 原始证据 |
|---|---|
| canary、N 扫描、M 扫描、量化、host 派发（Lead 复核） | `.s23/logs/kernel/lead_confirm.py` → `lead_confirm.json` |
| host 派发 4 轮交错 | `.s23/logs/kernel/lead_dispatch2.py` → `lead_dispatch2.json` |
| prefill 固定/每行斜率 | `.s23/logs/kernel/lead_prefill.py` → `lead_prefill.json` |
| 独立同口径全量微基准 | `.s23/fp8_emu_kernel_fixed_times.md` + `.s23/logs/kernel/t2_probe{,2..6}.json`、`t2_final_analysis.py` |
| 阶段图 / file:line / 调用配比 / 缓存边界 | `.s23/fp8_emu_stage_inventory.md`（T1） |
| 每步次数、idle 归属、gap、相关性（trace） | `.s23/fp8_emu_trace_attribution.md` + `.s23/logs/trace/attr_*.json`、`analyze_summary.txt` |
| Lead 独立重写的 trace 复核（逐次调用对照） | `.s23/logs/trace/lead_spotcheck.py` |
| 环境与探针入口 | `.s23/PROBE_ENV.md` |
| 历史对照 | `.s20/S20_FIXED_COST.md`、`.s20/S20_KERNEL_RESULT.md`、`tools/dcu/s20_kernel_NOTES.md` |
| ISA 依据（无 fp8/bf16 指令） | `docs/archive/result/z100-11d-z100-isa-capability-audit-result.md` |
| 内核实现 | `csrc/rocm/fp8_emu_gemm.cu`、`csrc/rocm/gfx906_emu_common.cuh`、`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py` |

### 复现（只读，不触碰生产）

```bash
# Lead 复核全套（约 12 分钟）
docker exec z100-regress bash -c 'cd /src && DTK_ROOT=/mnt/game/host-runtime/dtk2604 bash -c "
set +u; source tools/dcu/env.sh >/dev/null 2>&1
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install HIP_VISIBLE_DEVICES=0
export VLLM_DCU_USE_FP8_EMU=1 VLLM_DCU_FP8_GEMV_MAX_M=8 VLLM_DCU_FP8_DECODE_CFG=16
.venv/bin/python .s23/logs/kernel/lead_confirm.py"'

# trace 侧（纯 CPU，约 15 分钟）
cd .s23/logs/trace && bash rerun_all.sh
```
