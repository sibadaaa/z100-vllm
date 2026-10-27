# S33 WP1：fp8 GEMV whole-A 内核 —— 结果与停点报告（2026-10-26）

> 施工依据：`docs/design/z100-s33-fp8-scope-construction.md` §WP1（设计 D2）。
> **状态：判据 B（位同）PASS；判据 A（带宽门）未达标 → 按施工图纪律停，
> 不上服务 A/B，内核保持默认关（`VLLM_DCU_FP8_GEMV_WA=0`）。**

## 1. 交付物

- 新核 `fp8_emu_gemv_wholea_m1_kernel<RW, T>`（`csrc/rocm/fp8_emu_gemm.cu`，
  m1 kernel 之后）：整行 A（K×2B）+ 折算 scale 一次 stage 进动态 LDS，stage 后
  **一次** `__syncthreads()`，K 循环**零 barrier**；W 走 uint4 预取环 `pf[3]`
  （peel-by-3 字面量下标，m1 核 peel-by-2 同法）；slice/lane/chunk 遍历序、
  dot2 链、kb 折算、acc 累加序与 `fp8_emu_gemv_stream_m1_kernel` 逐行一致。
- 派发：`run_block_w8a16` 宏 `M == 1 && (stream_m1_enabled() || wholea_enabled())`
  分支内，`wholea_enabled() && wholea_shape_ok(K, nkb)` 选 whole-A，否则原
  slice-m1；`W8A8` 宏未动。`wholea_shape_ok`：`2K + 4·nkb ≤ 12 KiB` 且
  ≤ `stream_smem_limit()` —— **down（K=17408 → 34 KB）被门排除**，留现路径。
- env：`VLLM_DCU_FP8_GEMV_WA`（默认 0，单 env 语义：WA=1 独立生效，不需 M1=1；
  M1=1 & WA=0 行为与 s30c 现役完全一致）。指纹 KNOBS 已加
  `VLLM_DCU_FP8_GEMV_WA`（`tools/dcu/dcu_policy_fingerprint.py`）。
- 微测工具：`tools/dcu/p0_gemv_bench.py` SHAPES 增加 `gdn_in (5120,2560)`、
  `gdn_out (5120,1536)`（施工图 §1.1 形状清单）。

## 2. 构建与 md5 台账（C7）

| 件 | md5 |
|---|---|
| `_rocm_C.abi3.so.bak-s33`（改前，= s30c 现役） | `f5d279c5b5a47dcfaa443d0bc3eea4a0` |
| `_rocm_C.abi3.so`（WP1 后） | `a9325bffadb7b487d53b2641e6babf25` |
| 符号核实 | `strings \| grep wholea` = 174 命中（RW 1/2/4/8 × Half/BF16/float） |

build-sync `_rocm_C` 两次（首版+修 bug 版，均 ninja rc=0，无 hipify 竞态）。
台账：`.s33/wp1_so_md5.log`。**回滚**：cp bak-s33 回 temp 后重启，或 env 不设
WA（默认 0 即关）。

## 3. 位同对照（判据 B）—— PASS

ref = tiled 路径（`VLLM_DCU_FP8_GEMV_STREAM=0`，`/tmp/p0ref`，DTK 卡 3）。
M=1、routes=w8a16、每形状独立进程：

| 形状 (K×N) | 基线 delta（vs ref） | whole-A delta（vs ref） | 位同 |
|---|---|---|---|
| qkv 5120×2048 | 4.768e-07 | 4.768e-07 | **✓** |
| o 5120×1536 | 6.104e-05 | 6.104e-05 | **✓** |
| gateup 5120×8704 | 1.526e-05 | 1.526e-05 | **✓** |
| down 17408×1280 | 1.526e-05 | 1.526e-05 | **✓**（走现路径，不变） |
| gdn_in 5120×2560 | 3.052e-05 | 3.052e-05 | **✓** |
| gdn_out 5120×1536 | 6.104e-05 | 6.104e-05 | **✓** |
| M=8 全形状 | 与基线逐值相同 | 同 | ✓（M>1 路径不受影响） |

开发注记：首版实现曾把整行 LDS 读地址写成 slice 相对（漏 `+s0_` 偏移），
微测 `max|Δ|≈0.3` 立即暴露，修为绝对寻址后位同逐值通过 —— 微测口径有效。

## 4. 带宽微测（判据 A）—— 未达标，停

环境说明：4 卡 HBM 曾被在役 qwen-image-2.1 服务部分占用（卡 0/1），微测在
**空闲卡 3**（`HIP_VISIBLE_DEVICES=3`）进行；同卡同条件跑基线对照。容器停用、
全机空闲后复测 rows=4 两臂：基线 368.0/310.9/517.8/447.2/398.3/310.6，
WA 402.3/360.5/585.8/461.1/438.4/359.4 —— 与负载时数值一致（±0.5%），排除
邻卡负载干扰，结论不变。

GB/s（fp8 权重字节口径，M=1，50 iters；数据 `.s33/wp1_bench_*.json`）：

| 形状 | 基线 rows=auto(8) | 基线 rows=4 | WA rows=4（最优档） | 判据 A 门 | 结果 |
|---|---|---|---|---|---|
| gateup | 496.0 | 518.1 | **586.9** | ≥600 | ✗（−2.2%） |
| qkv | 343.2 | 368.3 | **402.5** | ≥450 | ✗（−10.6%） |
| o | 289.5 | 311.4 | **361.3** | ≥400 | ✗（−9.7%） |
| gdn_in | 374.7 | 399.9 | **440.8** | ≥450 | ✗（−2.0%） |
| gdn_out | 290.0 | 310.1 | **363.6** | ≥450 | ✗（−19.2%） |
| down | 436.7 | 446.3 | 451.4（现路径） | —（排除） | — |

rows 扫档（1/2/4/16/32）：rows=4 全形状最优；rows≥16 塌（wave 内行少）；
rows=1/2 ≈ rows=8。与设计图风险表"微测定档"一致，定档 rows=4。

相对增益（同卡 rows=4）：gateup +13.3%、o +16.0%、gdn_out +17.3%、
gdn_in +10.2%、qkv +9.3%。按生产 calls/round（16 qkv + 16 o + 48 GDN-in +
48 GDN-out + 64 gateup）折算每 round ≈ **−0.9~−1.1 ms**，低于 WP1 的 e2e
判据（−1.5 ms 以上），与判据 A 未过互相印证。

## 5. WP-f3 bank 冲突审计（并入 WP1 交付）

- 静态算术确认：A chunk 读 = `ds_read_b128`，lane 步距 32 B（8 words）→
  32 lanes 覆盖 8 个 bank-相位、每相位 4 lanes = **4-way 冲突**（施工图
  "4-way 嫌疑"成立）。
- pad 1×uint4（48 B 步距 = 12 words）：`12L mod 32` 周期 8 → 仍 **4-way**，
  对 M=1 whole-A 无效（16B 对齐约束下 b128 步距必为 4-word 倍数，冲突因子
  ≥4 是结构性的）；无冲突需奇数 word 步距（36 B），破坏 b128 16B 对齐。
- 生产 m1/stream 核的同款 A 读同样带 4-way —— 非 whole-A 回归；位同序约束下
  （lane 内 chunk 遍历序不可重排）无合规消除手段。
- RDC 构建下 `.hip.o` 不含 fatbin 段，`extractkernel` 仅出首个 bundle
  （w8a8_sdot4），wholea 的反汇编审计未走通；结论以静态算术 + 微测带宽为准。
  无 spill 迹象（pf[3] 在 RW=1 仅 12 寄存器；带宽无"pre&1 spill 级"塌陷）。

## 6. 结论与下一步（交用户裁决）

1. **按施工图判据 A 停**：不上服务 A/B，`VLLM_DCU_FP8_GEMV_WA` 保持默认 0；
   二进制保留（默认关，无行为差异）。
2. 可选（图外裁量）：若用户接受"位同 + rows=4 + −0.9~−1.1 ms/round 预期"
   的折衷，可在服务空闲时以 `VLLM_DCU_FP8_GEMV_STREAM_ROWS=4
   VLLM_DCU_FP8_GEMV_WA=1` 跑一次 N≥2 A/B 实测 e2e，按实测数决定采纳与否
   （不满足图内 −1.5 ms 门，属带条件采纳）。
3. 判据 A 未过的候选因素：卡 0/1 被 qwen-image-2.1 占用导致只能用卡 3 微测
   （环境较设计图基线慢 ~5%）；gdn_out（N=1536，最小 N）离门最远，grid 波尾
   占比高，是后续真正想拿到 o/gdn_out 收益时（NVFP4 储备路线或占用消除后）
   的第一优化点。

## 7. 修复尝试（用户裁决后追加，2026-10-27）：L2 直读变体

对 §5 的结构性 4-way bank 冲突做第二条路线：`fp8_emu_gemv_wholea_l2_m1_kernel`
（整行 A 不进 LDS，各 warp 直接从 GLOBAL/L2 读，64-lane wave 2 KB 连续
coalesced，LDS 仅剩折算 scale）。同值同序 = 位同（delta 表与基线逐值相同）。

| 形状 | 基线 rows4 | LDS 变体 rows4 | L2 变体 rows4 | 门 |
|---|---|---|---|---|
| gateup | 517.6 | 586.9 | 564.0 | 600 ✗ |
| qkv | 368.2 | 402.5 | 386.8 | 450 ✗ |
| o | 311.4 | 361.3 | 337.9 | 400 ✗ |
| gdn_in | 399.9 | 440.8 | 404.5 | 450 ✗ |
| gdn_out | 310.6 | 363.6 | 339.0 | 450 ✗ |

结论：两条独立实现（LDS 共享摊销 vs L2 直读免冲突）在同一水平撞墙——
**判据门需要换算法类（split-K GEMV，改累加序 → 零翻转硬门 + 用户裁决）**；
施工图 WP4 的"WP1 后 qkv/o 仍 <450 则启动 split-K"条件已触发，留待裁决。
env：`VLLM_DCU_FP8_GEMV_WA_L2`（默认 0；子模式）。位同证据同 §3。
