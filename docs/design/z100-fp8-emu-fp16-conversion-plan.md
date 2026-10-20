# Z100 fp8 算子「fp16 化」方案 v2

> 参考实现：`/mnt/game/1cat-vllm`（vLLM fork，针对 **V100 / sm_70**；同样**没有 fp8 硬件**，
> 靠「权重 fp8 + 激活 fp16（W8A16）」把 fp8 checkpoint 跑起来）。
> 落地文档：决策 1 → [`z100-fp8-w8a16-implementation.md`](z100-fp8-w8a16-implementation.md)；
> 决策 2 → [`z100-fp16-prefill-workspace-evaluation.md`](z100-fp16-prefill-workspace-evaluation.md)。
> 本文只出方案：**不跑卡、不改生产代码**。
> 口径（按用户指示）：**不把此前的 benchmark 数字当判据**——旧窗口的每调用固定项、TF/s、
> 每步 ms 一律不作为前提；需要数字的地方都写成「阶段 0 待测」。判据只来自
> ①ISA 事实 ②两边代码事实 ③参考实现已成型的工程结构。

---

## 0. 结论（三句话）

1. **「把 fp32 累加改成 fp16 累加」不解决问题**：`v_dot2_f32_f16` 已经是
   「2 个 fp16 MAC + 1 个 fp32 加」一条指令，与 `v_pk_fma_f16` 同发射口、同峰值；
   换累加精度只改寄存器占用与数值契约，不改指令数。
2. **参考实现给出的正解不是换累加，而是换契约**：**W8A16**——权重继续 fp8 常驻 HBM，
   **激活不量化、直接 fp16 进 kernel**。这一步同时删掉「激活 fp8 加宽」和「每层一次激活量化
   launch」，decode 每步少掉一半 launch，且权重带宽不变。
3. **fp16 真正该出现的两处**：①激活/输出用 fp16（W8A16 契约）②大 M（prefill）把权重
   **一次性解量化成 fp16 再交给 rocBLAS**。小 M（decode）保持 fp8 权重，另用
   **组对齐 split-K + 固定顺序归约**去啃每次调用的串行 K 链。

---

## 1. 判据（ISA 与代码事实，不含旧测量）

| # | 事实 | 出处 |
|---|---|---|
| F1 | 无任何 fp8 指令、无 MFMA/WMMA；只有 VOP3P 打包 fp16 | `csrc/rocm/gfx906_emu_common.cuh:1-26` |
| F2 | `v_dot2_f32_f16` = 2 fp16 MAC + fp32 累加 / 指令；`v_pk_fma_f16` = 2 fp16 MAC + fp16 累加 / 指令 | `.s24/z100-constraints.md:31-37` |
| F3 | `v_dot2c_f32_f16`（带累加的形式）**不存在**，会退化成 dot2+单独加 | `.s24/z100-constraints.md:35` |
| F4 | 我们当前是 **W8A8**：A、W 都 fp8，各带 scale（`a_s` fp32、`w_s` fp32）；内核用 256 项 LDS LUT 把 fp8 加宽成 fp16 | `csrc/rocm/fp8_emu_gemm.cu:1-44`、`:1470-1500` |
| F5 | 生产是 block-scaled：`w_s` 为 `[cdiv(N,block_n), cdiv(K,block_k)]`，`block_k=128`；epilogue 顺序 `(acc*a_s)*w_s` | `fp8_emu_gemm.cu:426-447`、`:395-408` |
| F6 | 参考实现是 **W8A16**：`kFp8Static128BlockSym`（e4m3 权重、static、128×128 block、fp32 scale）、**A/C 都是 fp16**，`apply_input_quant = False`，`apply_scaled_mm` 只接受未量化激活 | `1cat-vllm: vllm/model_executor/kernels/linear/scaled_mm/sm70_fp8.py:441-466`、`qpn8_blk.py:112-116,150-158`、`quant_utils.py:151-152` |
| F7 | 参考实现的大 M 路径 = `dequant(weights×scale)→fp16 workspace` 后一次 `at::mm_out`（cuBLAS），workspace 约 `max(K·N)·2 B`（85 MiB 级），按 role 给 `min_m` | `sm70_fp8.py:364-413`；`awq_sm70_gemm.cu:3055-3145` |
| F8 | 参考实现的小 M 路径 = 32 列 CTA + split-K（warp 级）+ SMEM fp32 归约 + 单次 fp32→fp16 写回；split 边界必须对齐 scale group | `fp8_qpn8_sm70.cu:294,363-366,400-425`；`sm70_marlin_splitk.cuh:27-97` |
| F9 | 参考实现的 TileRT 式「持久化 tile runtime」**实测变慢**（TPOT −9.25%/−32.8%），已弃用 | `1cat-vllm: docs/design/sm70_tile_runtime_exploration.md:441-485` |

---

## 2. 参考实现拆解 → 迁移判定

| # | 机制 | 位置（1cat-vllm） | 迁到 gfx906 的形态 | 迁移 |
|---|---|---|---|---|
| M1 | W8A16 契约（激活不量化） | `sm70_fp8.py:441-466` | 内核直接吃 fp16 A；删激活量化与 A 侧 LUT | **首选** |
| M2 | 权重 fp8→fp16 加宽与 group scale 在寄存器里就地折叠 | `sm70_marlin_fp8_gemm.cu:173-207,287-289` | 我们已有（LUT + `sscale` 折叠），保持 | 已具备 |
| M3 | 位运算加宽（无 LUT、无 LDS）：结果 = 真值/256，把 ×256 折进 fp16 scale | `fp8_qpn8_sm70.cu:38-66,95-108` | 需先解决 e4m3 **次正规码**（我们是 per-block scale，权重里次正规码常见）→ 见 §4.1 决策 | **条件迁移** |
| M4 | 权重离线 prepack（每线程向量化连续读 + macro-N 交织） | `sm70_marlin_iterator_utils.cuh:89-116`、`sm70_awq_marlin_repack.cu:122-160` | wave64/LDS bank 与 sm70 不同，**几何要重推**（8K×64N 或 16K×32N） | 部分 |
| M5 | 大 M：解量化一次 + 厂商 GEMM（含 workspace/`min_m` 表/按 role 区分） | `sm70_fp8.py:364-413`、`awq_sm70_gemm.cu:3055-3145` | `fp8→fp16` kernel + rocBLAS Tensile fp16；workspace 按层前缀绑定 | **首选（大 M）** |
| M6 | 小 M：32 列 CTA + split-K + SMEM 固定顺序归约（非 atomic） | `fp8_qpn8_sm70.cu:294-425`、`sm70_marlin_splitk.cuh:349-359` | 我们的 tile kernel 加 split-K；归约走 workspace + 一段串行求和，保证确定性 | **首选（小 M）** |
| M7 | 组对齐 split：分区边界取整到 `block_k/CTA_K` 的整数倍 | `sm70_marlin_splitk.cuh:27-97` | 直接照抄（我们 `block_k=128`，`CTA_K` 取 16/32/64） | 逐字 |
| M8 | 合批：同 A 的 2–4 个 GEMM 一次 launch（by-value 描述符 + `tile0` 扁平栅格） | `gguf_dense_segments_sm70.cuh:297-310,470` | **我们的 dense GQA 每层 4 个 GEMM 各用不同 A ⇒ 不可直接合批**；只搬「by-value 描述符 + 段内确定性 split-K」 | **需重新定界**（见 §3-R2） |
| M9 | 未装 box 的 raw op + 调用方持有 `out=` + 位置参数 | `vllm/_sm70_ops.py:129-135`、`csrc/torch_bindings.cpp:311-315` | 与我们 S23-O1 同向，再进一步：`out=` 复用、去掉 fake impl（eager） | 是 |
| M10 | 每形状调参表（(K,N,是否带激活)→(split_k, 累加链数, 预取)） | `sm70_fp8.py:51-59`、`sm70_marlin_common.cuh:487-580` | 我们的 cfg 表可加 `split_k` 维度；**只按 op 形状键，不按键名/模型身份** | 方法可用，表不可抄 |
| M11 | workspace 按**层前缀**绑定，不用 `data_ptr()` 当常量 | `sm70_layer_workspaces.py:1-30` | 直接照抄（graph/compile 下指针会变） | 逐字 |
| M12 | 路由快照 golden（(role,(K,N))×M 桶 → 选中的 route 哈希） | `tools/sm70_fp8_route_snapshot.py:29-38` | 给我们的 env/cfg 矩阵加回归快照 | 是 |
| M13 | 持久化 tile runtime / PDL / device-set 条件节点 | `sm70_tile_runtime_exploration.md`、`dsv3_fused_a_gemm.cu:43-55`、`sm70_sampler_graph_branch.cu:223-289` | HIP 无等价物；且参考实现自己实测变慢 | **不做** |

---

## 3. 路线总表

排序依据 = 机制收益 ÷ 数值风险 ÷ 工作量。**不含旧测量数字**，收益只给机理上界。

| ID | 路线 | 主要收益机理 | 数值影响 | 工作量 | 依赖 |
|---|---|---|---|---|---|
| **R1** | **W8A16 契约**（激活 fp16，权重 fp8 常驻） | 每层少 1 次量化 launch（每步少一半 launch）+ 删掉 A 侧 LUT 加宽与 A-LDS 读写 | 变（更准：无 A 量化误差） | 2–3 天 | — |
| **R2** | 段内 **组对齐 split-K + 固定顺序归约**（窄 N/小 M 填满 60 CU） | 把「每 block 串行 K 链」摊到多个 CTA；确定性归约不引入 atomic 抖动 | 不变（同加法顺序可保逐 bit） | 3–5 天 | R1（同文件） |
| **R3** | 大 M：**解量化一次 + rocBLAS fp16** | 权重加宽从热循环消失；GEMM 交给 Tensile | 变（w_s 烘进 fp16 的一次舍入） | 2–3 天 | 阶段 0 交叉点 |
| **R4** | host 税：raw op + `out=` + 去 per-call `getenv` | 每调用 host 时间与 `torch.library` 开销 | 不变 | 1–2 天 | — |
| **R5** | 位运算加宽替代 LDS LUT（M3） | 每元素少一次 LDS 读；但需处理 e4m3 次正规码 | 不变（若实现精确） | 3–5 天 | 基准见 §4.5 |
| **R6** | fp16 **累加**（`v_pk_fma_f16`） | 只省寄存器 → 更大 tile/occupancy | 变（溢出/精度） | 3–5 天 | 仅在阶段 0 证伪 F2 时 |
| **R7** | int8 `v_dot4_i32_i8` 全量化 | 4 MAC/指令 | 大改（greedy token 变） | 5–8 天 | 独立立项 |

> **关于 R2 的合批**：参考实现的 segments 前提是「多个投影共享同一个 A」。我们的 dense GQA
> 每层 4 个投影（qkv / o / gate_up / down）**各用不同的 A**，且互相有依赖，不能合批。
> 所以只搬「by-value 描述符 + 段内确定性 split-K + 扁平栅格」，不搬「多投影合批」。
> 若将来上 MLA/MoE（如 `fused_wqa_wkv` 那种共享 A 的形态），合批才直接可用。

---

## 4. 实现规格

### 4.1 R1 — W8A16 契约（最高优先）

**接口层**（不新增 op，改调用契约）：
`FP8ScaledMMLinearKernel.apply_weights` 现在固定走 `self.quant_fp8(x_2d, …)` 再
`apply_scaled_mm`（`ScaledMMLinearKernel.py:135-171`）。W8A16 需要：

1. `Gfx906Fp8BlockScaledMMLinearKernel` 覆写 `apply_weights`：不量化 A，直接
   `apply_scaled_mm_w8a16(A=x_fp16, B=w, Bs=w_s, …)`（参考实现的做法：
   `apply_scaled_mm` 对量化激活直接抛错，只走 `apply_weights`，`sm70_fp8.py:463-466`）。
2. 开关：`VLLM_DCU_FP8_W8A16=0|1`（默认 0 = 现值稳定），以及按 dtype 的准入
   （`input_dtype==fp16`、block-scaled、`block_k` 已知）。
3. M 分支留在**opaque op 内部**（参考实现的教训：Python 侧 `if M<=8` 会被 trace 住）。

**kernel 层**：新增 `fp8_emu_gemm_block_w8a16_kernel`（与现有 block kernel 同骨架）：
- staging：A 从 gmem 直读 fp16 → LDS（**无 LUT**）；W 仍 fp8 → LUT → LDS；
- `sscale = a_s * w_s` 的折叠变成只折 `w_s`（A 无 scale）；
- MAC / epilogue 不变（`dot2` + fp32 acc）；
- 对齐约束：`K % block_k == 0`、`N % BN == 0`（我们的 block_k=128，生产形状均满足）。

**收益机理**：每层由 `quant + GEMM` 两次 launch 变一次；A 侧每元素少 1 次 LDS 读 + 打包 ALU；
A 的 HBM 流量从 `M·K·1 B` 变 `M·K·2 B`——相对权重 `N·K` 可忽略（M≤8 时 A 只有几十 KB；
M=8192 时 A≈84 MB vs 权重 44.6 MB，但此时是计算受限，见 §4.3）。

**数值**：结果不再是当前 W8A8 的逐 bit 值（更接近真值）。门禁：单层 `rel_err` vs CPU 参考、
greedy 逐 token parity、logprobs finite。**默认 OFF**。

### 4.2 R2 — 组对齐 split-K + 确定性归约

- 分区：`active_split = min(requested, K / CTA_K)`；每区起点向上取整到
  `block_k / CTA_K` 的整数倍（照抄 `sm70_marlin_splitk.cuh:27-97`），保证 **128 元素的
  scale 组永不被切开**；`block_k % CTA_K == 0` 用 `static_assert` 锁死。
- 归约：workspace 里放 `[split][BM][BN]` 的 fp32 分片，由**同一 launch 内的最后一个 CTA**
  按 index 升序求和（参考实现 `gguf_dense_segments_sm70.cuh:470` 的写法），**不用 atomic**
  ⇒ 加法顺序固定、可逐 bit 复现。
- 每次调用必须清零 workspace：清零要么无条件发（不要用 host `if` 包，见陷阱 T1），
  要么由 kernel 内 lane0 负责。
- 适用：窄 N（本模型 o/out 投影 N=1536、qkv N=3584）+ 小 M 的 tiled 路径；
  streaming 路径不动。

### 4.3 R3 — 大 M：解量化一次 + rocBLAS fp16

```
w16 = fp16_scratch[0:N*K]                      # 进程级，按 (device, dtype, elements) 缓存
fp8_dequant_out(w16, Wq, w_s, group=128)       # 一个新 kernel：K-major [K,N] 输出
out = torch.mm(x_fp16, w16_view[K,N])          # rocBLAS Tensile fp16, fp32 累加
```

- 参考实现的关键细节直接搬：**输出 K-major `[K,N]`**（供列主序 GEMM）、每线程 8 个 fp8
  （`uint2`）、32 列 tile 摊薄 scale 读取、对齐 `K%8==0 / N%32==0 / group==128`
  （`awq_sm70_gemm.cu:3055-3145`）。
- workspace 尺寸 = `max_layer(K·N)·2 B`（本模型约 89 MB）。
- **指针绑定**：不要用 `data_ptr()` 做参数（参考实现踩过：编译产物会把编译期地址写死），
  要按层前缀在 runtime 查表（`sm70_layer_workspaces.py:1-30`）。
- `min_m`：参考实现给 3920（V100/24 GB 的产物），我们**必须自测**，起点取 `4096`，
  且按 role 分档（gate_up / down / out / qkv 各测一次）。
- 失败回退：workspace OOM → 打 warning 并回 fp8 路径（不要抛）。

### 4.4 R4 — host 税

- 模块级缓存 `torch.ops._rocm_C.<name>` 句柄、全位置参数、调用方持有 `out=`
  （参考 `vllm/_sm70_ops.py:129-135`）；eager 下不需要 `register_fake`。
- 把每调用一次的 `getenv + atoi`（我们的 `decode_gemm_cfg()`、`_prefill_cfg()`）改成
  `std::call_once` + 函数内 static，或 python 侧 import 时读一次。
- 量化 host 包装（3 次 `torch.empty`）在 R1 后自然消失。

### 4.5 R5 — 位运算加宽（可选项，先做基准）

参考实现的 e4m3→fp16 是 2–3 条 ALU（结果 = 真值/256，×256 折进 fp16 scale），不需要 LDS 表。
我们没直接用的原因是 **e4m3 次正规码**（指数 0）在位运算路径要额外修正，而 per-block scaled
的权重大量落在小值区。可行方案：给 LUT 路线与位运算路线各做一版，用真实权重的
`torch.equal` 门禁对比，只有位运算版也能精确时才换。

### 4.6 R6 — fp16 累加（判否，留翻案条件）

由 F2：指令数与峰值不变，收益仅剩「accumulator 减半 → 更大 tile」。
翻案条件：阶段 0 的 A1 实测 `v_pk_fma_f16 : v_dot2` ≥1.9（说明 dot2 是半速）。
若翻案，必须同时上：`L=8` 分层累加 + 2 的幂指数配平（`2^-s` 烘进 LUT，边界精确还原）+ 独立门禁。

### 4.7 R7 — int8 dot4（独立立项）

`v_dot4_i32_i8` 是 4 MAC/指令，历史记录里 decode 有 3.9× 量级，但会改 greedy token。
若要「decode 也大幅提升」，这是唯一量级够的路线，与 fp16 无关。

---

## 5. 陷阱清单

| # | 陷阱 | 来源 | 我们的动作 |
|---|---|---|---|
| T1 | **capture 期间被冻结的 host `if`**：`if (flag) cudaMemsetAsync(...)` 在捕获时求值一次，之后永远不重置 workspace | `1cat-vllm: csrc/libtorch_stable/topk.cu:162-172` | 清零无条件发，或搬进 kernel |
| T2 | **workspace 指针被当常量**（`data_ptr()` 进图/编译产物） | `sm70_layer_workspaces.py:1-30` | 按层前缀查表 |
| T3 | **workspace 跨层别名**：同一块 scratch 被两层同时用会互相踩 | `sm70_fp8.py:74-78` | 只在「单层串行」的调用点开 R3；并发 prefill 禁止 |
| T4 | **split 跨越 scale 组** ⇒ 边界用错 scale | `sm70_marlin_splitk.cuh:27-40` | 分区对齐 `block_k/CTA_K` |
| T5 | fp16 atomic 归约：丢精度 + 每元素 S 次转换 | `sm70_marlin_splitk.cuh:349-352` | 用 fp32 workspace + 固定顺序求和 |
| T6 | **÷256 指数偏置陷阱**：位运算加宽的真值是 `/256`，scale 必须先乘回 256，否则整体差 256× | `fp8_qpn8_sm70.cu:38-52 vs 95-108` | R5 实施时用「LUT 版 vs 位运算版」互校 |
| T7 | 转换/打包的物理字节序（`[0,2,1,3]`） | `awq_sm70_gemm.cu:3087-3089` | 我们 LUT 路线无此问题；R3 的 dequant kernel 要显式测序 |
| T8 | per-call `getenv` | `fp8_qpn8_sm70.cu:2023-2031`（他们自己也踩了） | `call_once` + static |
| T9 | 捕获期禁止 autotune（计时 kernel 进图） | `awq_sm70_gemm.cu:1330-1434` | 路由决策全部前置/静态 |
| T10 | 持久内核对 occupancy 是硬前提，且**实测可能更慢** | `sm70_ring_collectives.cu:152-155`、`sm70_tile_runtime_exploration.md:441-485` | 不部署持久化 tile runtime |
| T11 | 缺 op 要降级、显式开关要报错（两种行为分开） | `sm70_fp8.py:697-712` | env 显式 `=1` 但 op 缺失 → 抛错；auto 缺失 → 降级 |
| T12 | workspace 分配 OOM 必须回退 | `sm70_fp8.py:321-334` | R3 里 try/except → 回 fp8 |
| T13 | 按 role 名准入 vs 按 (K,N) 调参混用会串味 | `sm70_fp8.py:179-186` | 统一用 (K,N,dtype,block) 键 |

---

## 6. 阶段 0（卡空后 60–90 min，纯探针，不改生产）

| ID | 探针 | 判据 |
|---|---|---|
| A1 | `v_dot2_f32_f16` / `v_pk_fma_f16` / `v_fma_f32` 三条纯寄存器循环的每 lane-cycle MAC 数 | ≥1.9 ⇒ R6 翻案；否则 R6 关闭 |
| A2 | W8A16 kernel 变体 vs 现 W8A8：4 个生产形状 × M∈{1,8} 的每次调用时间与 kernel 名 census | 只用于定 R1 是否默认开（不看旧数字） |
| A3 | split-K 扫描：N∈{1536,3584,768}，split∈{1,2,4,8}，看栅格占用与时间 | 选 split 表并确认确定性归约 |
| A4 | rocBLAS fp16 在 `[M,K]×[K,N]` 生产形状的 TF/s，M∈{512,1760,4096,8192} | 定 `min_m` 与是否上 R3 |
| A5 | dequant kernel 带宽（读 fp8 + 写 fp16）与 `torch.equal`-级正确性（LUT 版对照） | ≥600 GB/s 且精确 |
| A6 | W8A16 数值门禁：单层 rel_err、greedy 8 prompt parity、logprobs finite | 决定能否 opt-in |
| A7 | 静态核对：本模型每层 4 个投影的输入是否真不同（决定 R2/合批是否永远不适用） | 5 分钟代码核对，不需要卡 |

**产出**：一张决策表（R1 默认值、R2 的 split 表、R3 的 `min_m` 表、R6 是否翻案），
写进 `docs/result/`。之后按 R1 → R2 → R3 顺序实施，R5/R6 作为插队项。

---

## 7. 验收与回退

1. 每项一个 env 开关，默认 OFF；`.so` md5 入档。
2. 机器码门禁：`.s23/probes/isa_gate.py` 口径，遍历每个 offload bundle，
   校验新增 kernel 的 `v_dot2`/`v_fmac`/`v_perm` 计数。
3. 数值门禁分层：kernel 级（vs CPU/参考实现）、单层级、e2e（greedy parity + logprobs）。
   R1/R3 必然改数值 ⇒ **不允许**用 `torch.equal` 卡；R2 目标是**保持**逐 bit。
4. 路由快照 golden（参考 M12）：(role,(K,N))×M 桶 → route 哈希，防回归。
5. 同窗口背靠背 A/B，只报比值，不跨窗口做差。

---

## 8. 不做的事 / 待决策

**不做**
- 不做持久化 tile runtime / PDL / device-set 条件节点（HIP 无等价物，且参考实现实测更慢）；
- 不做「多投影合批」（我们的 4 个投影 A 不同、且有依赖）——除非将来上共享 A 的 MLA/MoE；
- 不把 fp16 累加做默认（F2）；
- 不在 decode 用 fp16 权重副本（权重流量翻倍，收益机理不存在）；
- **不做 R7 int8 `v_dot4_i32_i8`**（决策 3：已否决，路线表里保留仅为留档）；
- **不做全量常驻 fp16 权重副本**（决策 2：与长上下文 KV 预算冲突，见评估文档）。

**已定的三个决策**（2026-10-20，用户）

| # | 决策 | 落地文件 |
|---|---|---|
| 1 | **做**：W8A16 作为生产路径（激活 fp16、不量化） | [`z100-fp8-w8a16-implementation.md`](z100-fp8-w8a16-implementation.md)（给执行 agent 的施工图） |
| 2 | **重新评估**：prefill 的 fp16 权重是常驻还是按层物化 → 结论「按层 scratch，否决常驻」，且评估只用静态分析、不占卡 | [`z100-fp16-prefill-workspace-evaluation.md`](z100-fp16-prefill-workspace-evaluation.md) |
| 3 | **不做**：int8 `v_dot4_i32_i8` 路线（R7 关闭，不再评估） | 见下「不做」 |
