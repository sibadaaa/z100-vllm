# S30c WP1-N 结果：GDN M=1 原生并行化（m1_native kernel）（2026-10-26）

> 施工依据：`docs/design/z100-s30c-pure-native-construction.md` §WP1-N；
> 设计依据：`docs/design/z100-s30c-pure-native-design.md` D1。
> 状态：**微测/构建/服务验证完成，e2e 判据 D 达成；贪心流门存在基线固有平局翻转，
> 见 §5 判读**。基线：s29-nospec 臂 step p50 = 39.24 ms / TTFT 739 ms / 21.33 t/s。

## 1. 实现摘要

- 新增 device kernel `fused_recurrent_gated_delta_rule_gfx906_m1_native_kernel`
  （`csrc/mamba/gfx906_decode_kernels.cu`，`_mtp_v2_kernel` 之后）：
  v2 骨架的"转置"——lane 持 2 个 k 元素（float2，512B 整行合并），warp 持 VW 个
  v 行，v2 的 cross-warp smem 交换换成 wave64 `__shfl_xor_sync` butterfly（6 步）。
  grid = `(v_tiles, kv_heads, tokens)`，batched decode（tokens>1）由 z 维直接覆盖
  （同时修复 DEC_V2 `tokens==1` 门把 c>1 全留给串行版的缺陷）。
- host 门在 `fused_recurrent_gated_delta_rule_gfx906_packed_decode` 内、DEC_V2
  判断**之前**：`state_stride_v==key_dim && state_stride_k==1`（原生池，与 v2 门
  互补）+ K=V=128 + tokens∈[1,65535] + 既有 stride 清单 + float2 8B 对齐防御检查。
  plain 与 combined 两入口同时生效（combined 内部调同一 host 函数，已核实
  gfx906_decode_kernels.cu:3078 一带）；op 签名/bindings/Python 零 diff。
- 可观测/回滚：`VLLM_GFX906_GDN_DEC_M1`（默认 1，=0 关）、`_CFG`（选档，默认 1）、
  `_DEBUG`（一次性 HIT/MISS fprintf）、`_TIME`（128 事件环 / 64 次聚合打点）。
  串行 kernel 保留为兜底与 parity 锚。指纹工具
  `tools/dcu/dcu_policy_fingerprint.py` 已补录四个 knob。
- 臂脚本：`.s30c/launch_native.sh`（= s29 nospec 臂 + M1 环境块；profiler 目录
  /mnt/game/s30c-prof）；`.s30c/restart.sh` 服务生命周期；`.s30c/probe.sh` 探针。

## 2. 构建（C7 纪律）

| 项 | 值 |
|---|---|
| 目标 | `_C`（**注意**：GDN decode kernel 在 `_C.abi3.so`；施工图 §0.3 的 `_rocm_C` 目标仅覆盖 WP2-N 的 fp8 文件） |
| 改前 `_C.abi3.so` md5 | `d44117aeac3705bafe95cddfb19768db`（备份 `.bak-s30c`） |
| 改后 `_C.abi3.so` md5 | `34d8b074775fa547a6098864f40eff23` |
| `_rocm_C.abi3.so` md5 | `f42d006481efa8e12504a1b73bbabdb5`（本 WP 未动，备份同在） |
| 构建命令 | `docker exec -e MAX_JOBS=5 z100-regress bash -lc 'bash /mnt/game/port/build-sync.sh _C'` |
| 构建 log | `.s30c/wp1_build3.log`（首次两跑因 hipify 拷贝竞态未触发编译，重跑即正常，见 wp1_build1/2.log） |

## 3. 微测（判据 A/B）

工具：`tools/dcu/s30c_gdn_m1_bench.py`（run/compare 两模式；env 进程级先置）。
形状：T∈{1,4}、H=4、HV=12、K=V=128、fp16 qkv/out、fp32 state 原生布局、种子 30。
微测命令（示例）：

```bash
export DTK_ROOT=/mnt/game/host-runtime/dtk2604; source tools/dcu/env.sh
HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD .venv/bin/python \
  tools/dcu/s30c_gdn_m1_bench.py run --impl m1 --cfg 1 --tag cfg1
HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD .venv/bin/python \
  tools/dcu/s30c_gdn_m1_bench.py compare .s30c/gdn_m1_serial1_T1_ctl.npz \
  .s30c/gdn_m1_m11_T1_cfg1.npz
```

### 3.1 cfg 扫描（T=1，event us/call，200 iters）

| impl | kernel | us/call | vs 串行 |
|---|---|---|---|
| serial（生产现状） | `packed_decode_kernel<Half,float>` grid[6,1,1] | 210.9 | 1.0× |
| m1 CFG1（64×VW4，384 blocks） | `m1_native_kernel<Half,float,4>` | **11.9** | **17.7×** |
| m1 CFG2（64×VW8，192 blocks） | 同上 VW=8 | 16.6 | 12.7× |
| m1 CFG3（128×VW4，192 blocks） | 同上 block=128 | 12.3 | 17.1× |
| m1 CFG4（256×VW4，96 blocks） | 同上 block=256 | 12.0 | 17.6× |

- **判据 A（≤40 µs）达成**，最优 CFG1=11.9 µs（设计预期 15-30 µs，参照地平线
  25.7 µs）。有效 state 带宽 ≈ 1.57 MB / 11.9 µs ≈ 132 GB/s（串行 ~7 GB/s）。
- batched decode（T=4）：串行 252.5 µs → m1 CFG1 22.1 µs（11.4×）。
- enqueue 开销 ~8 µs/call（与 WP3 相关，本 WP 不动）。
- **定档：CFG1**（默认值，launch_native.sh 已固化）。

### 3.2 parity 表（判据 B：max|Δ| ≤ 1e-5）

| 对比 | max\|Δout\|（fp16 域） | fp16-out 不同元素 | max\|Δstate\|（fp32 域） |
|---|---|---|---|
| T=1 serial vs m1-CFG1 | **0** | 0/1536 | 2.384e-07 |
| T=1 serial vs m1-CFG2/3/4 | 0 | 0/1536 | 2.384e-07 |
| T=4 serial vs m1-CFG1 | 1.526e-05 | **2/6144** | 2.384e-07 |
| T=1 slot=-1 serial vs m1 | 0（精确） | 0/1536 | 0（精确，无效 slot 不碰 state） |

T=4 的 2 个差异元素逐一核验：均为**恰好 1 个 fp16 ULP**（−0.012955/−0.012962 与
0.023727/0.023743；该量级 fp16 ULP 步距 1.27e-05 / 2.32e-05）——树形归约与串行
k 循环的 fp32 累加序差异所致，与设计图 D1"不空许 bit-exact"声明一致；dot/new_state
的乘加顺序已逐条对齐串行版（:1397-1444），排除实现 bug。e2e 探针为单请求
（T=1），该路径 fp16 out 与串行**位相同**。

## 4. 服务级验证

- 臂脚本核验：`/proc/<pid>/environ` 含 `VLLM_GFX906_GDN_DEC_M1=1 CFG=1`；
  指纹行 `policy: FP ...`（M1 四 knob 已入 KNOBS）。
- DEBUG：CUDA graph capture 期一次性打印
  `GFX906_GDN_DEC_M1 HIT: tokens=4 heads=4 kv_heads=12 cfg=1 sv=128 sk=1
  tiled_map=0 transposed=0`（`.s30c/serve-m1dbg.log:1368`）——生产池形状命中，
  batched decode 同路径生效。
- TIME（`VLLM_GFX906_GDN_DEC_M1_TIME=1`）：见 §6（待本轮补）。
- e2e（7469 doc / 96 tok / 3 reps，`bash .s30c/probe.sh s30c-m1-r1`）：

| restart | run | ttft_ms | step_ms | out t/s |
|---|---|---|---|---|
| r1 | 1（冷） | 8450 | 29.4 | 8.5 |
| r1 | 2 | 733 | 29.8 | 26.8 |
| r1 | 3 | 722 | 29.4 | 27.1 |
| r1 | p50 | — | **29.44** | 20.8（含冷启动） |
| r2b（TIME=0 干净） | 1（冷） | — | — | — |
| r2b | 2 | 770 | 30.0 | 26.3 |
| r2b | 3 | 732 | 29.4 | 27.0 |
| r2b | p50 | — | **29.76** | 20.6（含冷启动） |

  **判据 D（step p50 −5 ms 以上）达成，N=2 重启分布 29.44 / 29.76 ms**
  vs 基线 39.24 ms = **−9.5~−9.8 ms**（预期 −8~9 ms；47.5 calls/round ×
  (216−12.5) µs ≈ −9.7 ms/round，kdiff 核账闭合）。
  热启动 TTFT 722-770 ms vs 基线 739 ms（−1%~+4%，门内）。
  `srv_tok_per_step=1.0`。热 out t/s 26.3-27.1（基线 21.33；设计目标 34-37 t/s
  属 S30c 全家合计，WP2-N/WP3 待做）。
  （r2 TIME=1 仪器化一跑 31.06 ms 已废弃不用，证据在 probe-m1-r2.log。）

## 5. 贪心流门判读（96-token，temperature=0，top5 logprobs）

`.s30c/capture-s30c-m1.json` vs 基线 `.s29/capture-s29-bm16-r{1,2}.json`
（基线两份 = 同臂两次独立重启）：

1. **基线自身跨重启在 pos 2 翻转**：r1 top1=96540（margin 0.0000）与
   r2 top1=115508（margin 0.0156）——0-margin 精确平局（两 token top1 logprob
   同为 −1.6009），基线 r1/r2 首分歧即 pos 2（s28d"accept lottery"同源）。
2. **m1 流 = 基线 r2 世界**：`cmp_capture(s29-bm16-r2, s30c-m1)` →
   **STREAMS IDENTICAL**（96/96 token 位同）。m1 未产生基线分布之外的新流。
3. 共同前缀 top1 logprob 偏移：m1 vs r2 全 96 token max 0.0116 nat；
   基线 r1 vs r2（前 2 token）0.0022 nat。0.0116 < S29 BM16 采纳先例的
   0.0213 nat（docs/result/z100-s29-bm16-ab.md）。
4. 会话内确定性：m1 3 reps 逐位相同（`in_session_identical: True`）。

**判读**：严格字面的"m1 vs 基线 r1 零翻转"不可满足——因为基线跨重启在
同一平局处自己翻转；m1 的流落在基线自身重启分布内，且扰动量级有 BM16 采纳
先例背书。按设计图风险表，此处属"S29 WP-C 流程交用户裁决"事项，不构成
实现缺陷证据；第 2 次 m1 重启流验证见 §6（若仍落在基线两世界内则闭环）。

## 6. 第二次重启（TIME + 第二份贪心流 + kdiff）

### 6.1 服务内 TIME（`VLLM_GFX906_GDN_DEC_M1_TIME=1`，restart r2）

- capture 期首批聚合：`GFX906_GDN_DEC_M1_TIME avg 28.9 us over 64 calls`
  （capture batch 为 tokens=4 混合档）。
- 稳态 decode（piecewise graph 段间 eager 发射，14400 calls/聚合）：
  **avg 20.5-20.9 µs**。高于微测 11.9 µs（+73%，超出 ±20% 对齐窗）——事件对
  夹住的是"图段间排队等待 + kernel 执行"，微测是空载背靠背；下述 kdiff 的
  per-op 时延（12.5 µs/call，纯 GPU 执行时间）与微测一致，e2e 收益
  （−9.8 ms）与微测外推（−9.7 ms/round）闭合，判定为度量口径差而非实现回退。
- ⚠️ TIME=1 的 `cudaEventSynchronize`（每 64 calls）会周期性阻塞 rank0 提交流，
  使对端 skew 变大 → 本轮 kdiff 中 nccl/rms_norm 行为污染值（见 6.3），
  e2e 亦受仪器化影响（r2=31.06 ms 含此开销）。

### 6.2 第二份贪心流（第 2 次 m1 重启）

`capture-s30c-m1-r2.json`：**与 s29-bm16-r1 流 96/96 位同（STREAMS IDENTICAL）**，
与 s29-bm16-r2 在 pos 2 分歧（同一 0-margin 平局）。两次 m1 重启分别落在基线
自身的两个重启世界（r1/r2），**m1 未产生基线分布之外的流** ——§5 判读闭环。
会话内 3 reps 位同。

### 6.3 kdiff（s29ns 基线 trace vs s30cm1 trace，rank0，20 rounds）

`PYTHONPATH=tools/dcu .venv/bin/python .s29/kdiff.py
/mnt/game/s29-prof/s29ns_rank0.1792940639539999916.pt.trace.json.gz
/mnt/game/s30c-prof/s30cm1_rank0.*.pt.trace.json.gz --csv .s30c/kdiff-s29ns-s30cm1.csv`

| kernel | c/r A→B | us/c A→B | us/r A→B |
|---|---|---|---|
| `packed_decode_kernel`（串行）[6,1,1] | 47.55 → 0 | 216.3 → 0 | 10283 → 0 |
| `m1_native_kernel` [32,12,1] | 0 → 48 | 0 → **12.5** | 0 → 601.4 |
| **GDN 递推合计** | | | **−9681.6 µs/round** |
| `causal_conv1d_gfx906_decode_update_kernel` | 47.55 → 48 | 7.9 → 7.2 | −32.7 |
| `fp8_emu_gemv_stream_block_kernel` | 253.5 → 256 | 48.2 → 48.1 | +99.8 |
| `kernel_unified_attention_3d` | 15.85 → 16 | 77.0 → 75.9 | −7.0 |
| `bf16_emu_gemv_dot2_kernel` | 47.55 → 48 | 11.6 → 11.5 | +4.2 |

- **m1_native per-op 12.5 µs/call**（纯 GPU 执行时间，与微测 11.9 µs 对齐 ±5%）；
  每轮 GDN 净收益 −9.68 ms ≈ e2e 实测 −9.8 ms，账闭合。
- **conv kernel 持平（7.9→7.2 µs/call，偏差 <2 µs 门内）**——本方案不动 conv 的
  预期成立；fp8 GEMV / attention / bf16 GEMV 均持平（WP2-N 目标未受扰动）。
- ⚠️ nccl 行（`Generic_4` 54.2→99.0 µs/c）与个别 triton_red 行的 A→B 膨胀最初
  疑为 TIME=1 事件同步污染；随后 **TIME=0 干净 trace 复核（§6.4）显示该膨胀
  仍在**（`Generic_4` 54.2→80.6 µs/c、rms_norm_5 10.7→22.0 µs/c），判定为
  **profiling 窗口内 rank 间 skew 差异**（nccl 自旋 kernel 的时长=对端到达抖动；
  仅比较 rank0 窗口）。反证：无 profiler 的 e2e step 差（−9.5~−9.8 ms）与 GDN
  kernel 差（−9.66 ms）闭合——若 nccl 真 +3.4 ms/round，e2e 只能改善 ~6 ms；
  且 compute 型 kernel（fp8 GEMV/attention/conv/bf16 GEMV）全部 ≤±0.3 µs/c 持平。

### 6.4 干净 trace 复核（TIME=0，s30cm1b）

`kdiff(s29ns_rank0, s30cm1b_rank0)` → `.s30c/kdiff-s29ns-s30cm1b.csv`：
serial 47.55×216.3 → 0；`m1_native_kernel` 48 c/r × **12.9 µs/c** = 618.9 µs/round
（两次采样 12.5/12.9 µs，与微测 11.9 µs 同域）；conv 7.9→7.2、fp8 GEMV
48.2→48.8、attention 77.0→75.9 —— 全部持平。nccl 膨胀如上属窗口 skew 工件。


## 7. 回滚

- env 一行：`VLLM_GFX906_GDN_DEC_M1=0`（或臂脚本不 export）→ 串行路径，立即生效需重启。
- 二进制：`cp /mnt/game/port/build-sync/temp/_C.abi3.so.bak-s30c
  /mnt/game/port/build-sync/temp/_C.abi3.so` 后重启（md5 回 `d44117ae…`）。
- 代码默认值固化纪律：env 默认开（=1）已按判据 D 达成前置；如用户裁决否决，
  改 launcher 不 export + 代码默认回 0。
