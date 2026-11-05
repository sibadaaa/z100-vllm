# 优化计划 v3(2026-10-29,占用率清单 + F 模型修订)

> **修订原因**:按占用率流程图完成实证后,建立了 m1 核的解析耗时模型
> **T = F + bytes/727GB/s**(F≈15us 每次启动的固定开销,5/5 形状验证,±2us)。
> 该模型**证伪 split-K**(对 F 主导的形状收益≈0),替换为 F 分解与削减。
> 基线与纪律同 v2。

---

## 0. 占用率清单判定(流程图逐分支,实测)

热点内核 m1(`fp8_emu_gemv_stream_m1_kernel<1>`,生产 M=1,43% 步时):

| 流程图节点 | 判定 | 证据 |
|---|---|---|
| Occupancy 是瓶颈? | **部分是**——但限制因素不在清单三项内 | 理论 6/8 waves/SIMD;小 N 形状单 cohort,F 占 55% |
| 寄存器限制? | **否** | VGPR=40(上限 256/40→6 波),**spill=0**;机器驻留 1440 warp |
| 共享内存限制? | **否** | 静态 512B(LUT)+ 动态 ~2.6KB,64KB 的 4% |
| Block 大小? | **否** | 256 线程=4 波匹配 4 SIMD;rows ∈{2,4,8}×3 组 A/B 全部持平(warp 总数=N 是不变量) |
| spill 风险 | **m1<8> spill=381B**、m1<4> spill=16B | 生产 RW=1 安全;**任何把 rows 提到 16/32 的配置会落到 <4>/<8> 档,禁止** |

稳态速率 **727 GB/s = copy 屋顶的 93%** → 内核主循环近乎完美;
gateup"只有 75%"是 F 摊薄问题,不是 bank conflict(N-A 降级为可选)。

## 0.5 F 模型(本计划的理论基础)

```
T(启动) = F + Σ_slice bytes / 727 GB/s        F ≈ 15 us
cohort 数 = ceil(warps / 1440);只有最后一个 cohort 支付 F
```

| 形状 | N | slice 数 | cohort | 实测 us | 模型 us | F 占比 |
|---|---:|---:|---:|---:|---:|---:|
| o | 1280 | 6 | 0.89 | 27.4 | 26.9 | **55%** |
| qkv | 2048 | 5 | 1.42 | 27.5 | 29.4 | 54% |
| down | 1280 | 17 | 0.89 | 47.8 | 48.7 | 31% |
| gateup | 8704 | 5 | 6.04 | 76.9 | 76.3 | 20% |
| syn4352(合成) | 4352 | 5 | 3.02 | 44.5 | 45.7 | 34% |

**推论**:每步 256 次 m1 启动 × F ≈ **3.8ms/步(14% 步时)是纯启动开销**;
split-K 无法消除它(o 切 3 段 → 25.9us vs 现 27.4,收益≈0)。

---

## 1. P0(本周)

### W1. 恢复 MTP + draft 计数核查(不变,前置)

### W2'(替换原 split-K):F 分解与削减(最大单项,−2~−4ms/步)

- **第一步(半天)**:分解 F 的 15us。rocprofv2 单 kernel 时间线
  (`--hip-api --kernel-trace` 或 rocpd),切四段:LUT/scale 预置、
  首两 slice 预取暴露、wave 启动 ramp、HBM/TLB 冷启动(旋转缓冲下)。
- **第二步(按分解结果选)**:
  - 若首 slice 延迟暴露为主 → `kStreamSlice` 1024→2048(slice 数减半,
    预取在途翻倍,barrier 减半;LDS 增至 ~5KB,仍远低于闸);
  - 若启动 ramp 为主 → 试 nt=512(rows=8 等价 warp 数,启动批更大)+
    `__launch_bounds__(512, 3)` 钉 occupancy;
  - 若 LUT/scale 预置为主 → LUT 移 `__constant__`/全局常量,免每 block 预置。
- **门**:o/qkv 的 F 可测分量减半;微测 o ≤ 20us、qkv ≤ 22us;
  gateup ≥ 570 GB/s 零回归;数值 bit 级同。
- **不做**:split-K(模型证伪)、rows 旋钮(已证伪)、rocBLAS。

### W3. decode 长上下文 split-KV(不变)

零代码 segments ∈ {2,4,8} × ctx 扫描 → 不足再自适应分段;
KV b128 审计;`tl.exp`→exp2 跟车。门:128k p50 −10%,7.5k/32k 回退 ≤2%。

---

## 2. P1(下周,不变)

- **W4** push AR 出口融合 add-rmsnorm(R3):AR 8.9ms/步(26%),−0.8~−1.4ms。
- **W5** c=4 verify 效率(R2):W1-W3 改变步构成后重 profile 再动手。

## 3. P2(备选)

- **W6 INT8**(R5):`v_dot4_i32_i8` 在位;W2' 后 GEMV 若仍 F 主导,INT8
  同时减字节与 slice 数(17→9),双重收益 −9ms/步;需量化评估。
- **W7** `VLLM_S24_DYNK_HI/LO` 阈值复核。**W8** prefill/TTFT(R12)。

---

## 4. 预期汇总(引擎口径)

| 口径 | 现值 | W2' 后 | +W3 后 | +W4 后 |
|---|---:|---:|---:|---:|
| 7.5k MTP t/s | 51.6 | 55-58 | 57-61 | 59-63 |
| 128k MTP t/s | 31.4 | 32-34 | 35-38 | 36-39 |
| 理论锚点 | fp8 GEMV 11.5ms/步 | −2~−4ms | — | — |

## 5. 执行顺序

```
W1(0.5h)→ W2'-分解(0.5d)→ W2'-削减(1-2d)→ W3(1-2d)→ W4 → W5
```

W3 与 W2' 独立可并行。所有改动过:微测 → 数值对照 → 服务 smoke → 同口径 e2e。

## 6. 工具与产物

- 占用率元数据:`/tmp/devmeta.cu`(设备码抽取 TU)+ llvm-readobj 流程
  (hip_fatbin → 长度前缀 bundle → NT_AMDGPU_METADATA msgpack/YAML)
- 微测:`/tmp/fill_model.py`(F 模型验证)、`/tmp/fp8_gemv_bench.py`(cfg/M A/B)
- 长上下文:`/tmp/ctx_sweep.py`;profile:`/start_profile`→请求→`/stop_profile`
