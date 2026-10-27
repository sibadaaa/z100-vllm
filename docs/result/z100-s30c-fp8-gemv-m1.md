# S30c WP2-N 结果：fp8 GEMV M=1 两级推进（2026-10-26）

> 施工依据：`docs/design/z100-s30c-pure-native-construction.md` §WP2-N；
> 设计依据 D2。基线 = WP1-N 采纳后的 m1 臂（`VLLM_GFX906_GDN_DEC_M1=1`，
> ROWS 默认 auto=8、GEMV_M1 默认 0）。

## 1. 形状清单

生产 decode（W8A16 route，`fp8_emu_gemv_stream_block_kernel`）4 个 (N,K) 形状
（Qwen3.8-27B TP4，`p0_gemv_bench.py` SHAPES 表；trace 无 per-instance grid，
按模型投影取全集）：qkv 5120×2048、o 5120×1536、gateup 5120×8704、
down 17408×1280；基线块表 253.5 c/r × 48.2 µs = 12.22 ms/round。
`VLLM_DCU_FP8_STREAM_MAX_M` 保持 unset（M 门未动）。

## 2. 第一级：env 扫描（零代码）

`tools/dcu/p0_gemv_bench.py --mode bench`（每组合独立进程，M=1，w8a16 route）：

| 组合 | qkv | o | gateup | down |
|---|---|---|---|---|
| s1 rows=auto(8)（基线） | 30.9 µs / 339G | 27.2 / 289G | 89.7 / 497G | 51.2 / 435G |
| s1 rows=4 | 28.9 / 363G (−6.5%) | 25.6 / 308G (−5.9%) | 86.3 / 516G (−3.8%) | 50.0 / 446G (−2.3%) |
| s1 rows=8 | 持平基线 | 持平 | 持平 | 持平 |
| s1 rows=16/32/64 | 全面变慢（+19%~+464%） | | | |
| s2（variant 2） | +57.8%~+136.7% 全面变慢 | | | |

- **第一级判据（热形状 per-op −15%）不达标**（最优 rows=4 仅 −2.3%~−6.5%）；
  qkv/o 仍在 400 GB/s 以下 → 按施工图进入第二级。
- rows=4 独自的 e2e 外推 ≈ −0.5 ms/round，单独不构成采纳项。

## 3. 第二级：MS=1 专用 stream kernel（`fp8_emu_gemv_stream_m1_kernel`）

- 实现：`csrc/rocm/fp8_emu_gemm.cu`（v1 stream kernel 之后）——M 维剥离：
  A 单行 fp16 slice 拷贝、折算 scale 表 [nkb]、acc[RW]（无 [RW][MS] 组）、
  内积循环无 `m<M` 谓词/分组；2-slice 预取管线与 v1 相同。
- 门：`VLLM_DCU_FP8_GEMV_M1=1`（**默认 0**），W8A16 分派处 `M==1` 优先命中；
  rows 仍走 `stream_block_select`（ROWS override 继续有效）。指纹工具已补 knob。
- 构建（C7）：`build-sync.sh _rocm_C`；改前 md5 `f42d006481efa8e12504a1b73bbabdb5`
  （`.bak-s30c`），改后 `f5d279c5b5a47dcfaa443d0bc3eea4a0`；log `.s30c/wp2_build1.log`。
- 微测（M=1，vs 第一级基线）：

| 组合 | qkv | o | gateup | down |
|---|---|---|---|---|
| m1 rows=auto(8) | 29.6 / 354G (−4.2%) | 25.6 / 307G (−5.9%) | 83.9 / 531G (−6.5%) | 49.9 / 447G (−2.5%) |
| **m1 rows=4** | **27.2 / 386G (−12.0%)** | **23.9 / 329G (−12.1%)** | **80.7 / 552G (−10.0%)** | **48.1 / 464G (−6.1%)** |
| m1 rows=8 | −4.9% | −6.2% | −6.5% | −3.5% |

- **数值**：`p0_gemv_probe.py`（M=1，4 形状，fp32 反量化参照）——m1 rows=4 与
  基线 stream kernel 的 max|delta| **逐形状完全相同**
  （6.0e-5 / 6.0e-5 / 9.1e-5 / 1.2e-4）：M 维剥离不改变每行 K 累加序，数值上
  与生产 kernel 位同。贪心流验证：m1gemv 臂 96-token 流 = 基线 r2 世界
  （STREAMS IDENTICAL，见 `.s30c/capture-s30c-m1gemv*.json`）。
- 服务 A/B（m1 臂 + ROWS=4 + GEMV_M1=1）：

| restart | step p50 | TTFT(热) | 备注 |
|---|---|---|---|
| m1gemv（第 1 次） | (33.56) | — | **污染样本弃用**：本臂随后在 profiler collect 中
  `sample_tokens` RPC 超时 wedged 并退出（已知 profiling 危害，s29 同款），该样本不可信 |
| m1gemv2 | **28.32** | 736-749 ms | |
| m1gemv3 | **28.76** | 721-781 ms | |

  vs m1 臂基线 29.44 / 29.76 → **Δ ≈ −1.0~−1.4 ms**。kdiff（s30cm1b vs
  s30cm1gemv）证实 GEMV per-op 44.5 vs 48.8 µs/c（−1.16 ms/round，与微测吻合）；
  同表 nccl 行 +2.3 ms/round 膨胀与 §WP1-N §6.3 同源（trace 窗口 rank-skew
  工件；若为真实增量，则 m1gemv2/3 的 e2e 无法成立），以 wall-clock e2e 为准。

## 4. 判据与处置

| 判据 | 门 | 实测 | 结论 |
|---|---|---|---|
| 热形状 per-op（两级合计） | −15% 以上 | −6.1%~−12.1% | **未达标** |
| 服务 e2e（两级合计） | −1.5 ms 以上 | −1.0~−1.4 ms | **未达标（边际）** |
| 贪心流零翻转 | 硬门 | 流=基线世界，位同 | 通过（但前置门未过） |

**处置：按施工图判据兜底不采纳**——`VLLM_DCU_FP8_GEMV_M1` 默认保持 0、
`ROWS` 保持 auto，launch_native.sh 不 export（默认臂行为 = 第一级前基线）。
两级成果（kernel + 扫描表）保留在 `.so` 与本文档，属零数值风险、用户可裁量的
~1.0-1.4 ms 备选臂：启用 = 重启时加
`VLLM_DCU_FP8_GEMV_STREAM_ROWS=4 VLLM_DCU_FP8_GEMV_M1=1`；停用 = 去掉两个 env。
若用户按"速度优先"目标函数裁定采纳，仅影响启动 env，不需改代码。

## 5. 回滚

- env 不设即回滚（kernel 门默认 0，路径不被命中）。
- 二进制：`cp /mnt/game/port/build-sync/temp/_rocm_C.abi3.so.bak-s30c
  /mnt/game/port/build-sync/temp/_rocm_C.abi3.so` 后重启。
