# S30c 设计图：纯 decode 原生路线（csrc 解禁）+ 显存最小化（2026-10-26）

> **决策输入**：用户目标函数改为**只追求速度与尽量少用显存**，构建成本不考虑，
> csrc 解禁。本图取代 `docs/design/z100-s30b-construction-gdn-m1-triton-gemv-hostgap.md`
> 中 WP1/WP2 的 triton 执行路线（该文降级为 Plan-B 存档，其 §0 环境须知仍适用）。
> 上游实测基线：`docs/design/z100-s30b-pure-decode-plan.md`（39.2 ms/tok 块表，
> 不重推导）。施工步骤见配套施工图
> `docs/design/z100-s30c-pure-native-construction.md`。

## 1. 目标与口径

| 维度 | 口径 | 现值（s29-nospec） | 目标 |
|---|---|---|---|
| 速度·每 token | `srv_step_ms`（=ms/tok，1 tok/step） | 39.24 ms | **27–29 ms** |
| 速度·单流 | out_tok_per_s | 21.33 t/s | **34–37 t/s** |
| 速度·交互 | TTFT（热，7469 doc） | 0.74 s | **保持**（±3%） |
| 显存 | 切换时 serve log + hy-smi 快照差 | dflash 在役占用 | **省出 draft 塔 + spec 缓冲**；graph 池裁剪；可选 UTIL/MBT 臂 |

对齐口径强制：e2e 一律 7469 真实文档（`--truncate-tokens 7469`）/ 96 max-tokens /
3 reps / N≥2 重启分布；速度结论必须同时报 step 与 TTFT 双口径（计划禁止事项）。

## 2. 路线定案

**纯 decode 单路线，csrc 原生实现。** 依据：

1. WP1（GDN M=1，−8~9 ms）达成后纯 decode 即反超 dflash（30.9 ms/tok），
   且 TTFT 3×、无 accept 彩票、无提案 parity 风险面；
2. 用户目标含显存 → 纯 decode 天然省出 draft 塔权重 + draft KV + spec 调度缓冲
   （draft = 5 层 bf16 塔 + embedding，具体以切换时快照差为准，不预估入账）；
3. dflash 对照不再维护双臂，退役为留档数字；终局验收 = 本路线自身判据 +
   显存快照差。

## 3. 墙位（已核实的代码事实）

GDN M=1 递推 47.5 次/round × 216 µs（占 28%）是最大杠杆。机制：

- 生产调用链（纯 decode，无 spec）：`GdnLinearAttn.forward`（gdn_linear_attn.py:1200
  combined 默认开）→ ratio2 门 `kv_heads==2*heads` 12≠8 **MISS**（fused_recurrent.py:787）
  → `_C` combined op（内部 conv + GDN 两连发）→ host 函数
  `fused_recurrent_gated_delta_rule_gfx906_packed_decode`（gfx906_decode_kernels.cu:2786）。
- **DEC_V2 warp-cooperative 快路 MISS**：门要求 state v 连续（stride_v==1，:2871），
  生产 decode 池为原生 `[slot][head][v][k]` k 连续（`use_transposed_state_for_packed_decode
  = False`，gdn_linear_attn.py:446）→ 落到串行 kernel（:1234）：grid [6,1,1]、256 线程、
  每线程一个 (head,v) 列、k=128 串行。total = 1×12×128 = 1536 线程对 60 CU——纯延迟暴露。
- 有效带宽核算：每调用 state 字节 = 12 head × 128×128×4B = 786 KB（读+写 1.57 MB），
  216 µs ⇒ **~7 GB/s，离流墙 50 倍**。这不是带宽问题，是 block 数问题 ⇒ 修法 = 网格
  展开 + 布局友好的合并读写，与 v2 的 register-resident 骨架无关地成立。

## 4. 设计决策

### D1（WP1-N，主杠杆）：`m1_native` kernel —— 把 mtp_v2 骨架"转置"到原生布局

现有 `_mtp_v2_kernel`（:731，dflash 侧同功能 25.7 µs 级）的分解是 lane 持 2 个 **v**
元素、warp 分 k 行，要求 v 连续池。设计上做对称翻转，适配原生 k 连续池：

- **网格**：`grid = (v_tiles, kv_heads, tokens)`，block = wave64（64 线程），
  每 warp 持 VW 个 v 行（默认 VW=4 → v_tiles=32 → 384×T blocks；扫描
  VW ∈ {2,4,8} × block ∈ {64,128,256}，经 `VLLM_GFX906_GDN_DEC_M1_CFG` 选档，
  微测定默认）。batched decode（c>1）由 z 维直接扩展——现有 DEC_V2 门
  `tokens==1` 把 c>1 全留给串行版的缺陷一并修复。
- **每 v 行流水**：`float2`/lane 载入 state[v, 2k, 2k+1]（64 lane × 8B = 512B 整行，
  完全合并）→ decay 乘 → `dot = butterfly(__shfl_xor, 6 步)` 全波归约（顶替 v2 的
  cross-warp smem 交换）→ `v' = (v − dot)·β` → `h += v'⊗k` → `out[v] = butterfly(Σ h·q)`
  → `float2` 写回。每 lane 寄存器 2×VW=8 个 state 浮点（v2 是 64），占用可忽略。
- **每 block 一次**的头内公共量：q/k l2norm（`rsqrtf(Σ+1e-6)`，scale 只乘 q）、
  门控（softplus 阈值 20、`decay=exp(−exp(A_log)·sp)`、`β=sigmoid(b)`）——
  语义逐条对齐串行版 :1273-1275（q/k/v 偏移）、:1531-1544（门控）、:457-460/:1326-1336
  （norm）。
- **无效 slot 语义**：`state_indices[t] < 0` ⇒ out 该 (t,hv,·) 段写 0、不读不写
  state（:1357-1361）。
- **数值声明（诚实）**：butterfly 归约 ≠ 串行 k 循环，与生产 kernel 存在 fp32 ULP 级
  差异。不空许 bit-exact；验收 = per-op max|Δ| 报表 + **96-token 贪心流零翻转**
  （S29 BM16 先例：零翻转 + max|Δ top-1 logprob| 0.0213 nat 采纳）。生产 attention
  路径本就含 wave 级归约，风险量级有先例背书。若出现翻转：退回 S30b triton
  路线（Plan-B，已留档）或按 S29 WP-C 流程交用户裁决。
- **集成（零 Python diff）**：host 函数 :2862 的 `dec_v2_shape_ok` 旁加第三变体门——
  `state_stride_v==key_dim && state_stride_k==1`（与 v2 门互补）+ K=V=128 +
  既有 stride 检查（照抄 :2868-2872 清单）+ `tokens>=1`。op 签名、bindings、
  `_custom_ops.py`、`fused_recurrent.py` **全部不动**；plain 与 combined 两入口
  （combined 内部调同一 host 函数）同时生效，**无需** `VLLM_QWEN35_COMBINED_PACKED_DECODE=0`
  （triton 路线被迫拆路径的代价在这里不存在）。
- **可观测**：仿 v2 的 DEBUG/TIME 模式（:2873-2891、:2900-2989）加
  `VLLM_GFX906_GDN_DEC_M1_DEBUG`（HIT/MISS 一次性打印）与
  `VLLM_GFX906_GDN_DEC_M1_TIME`（128 事件环，64 次聚合打点）。
- **kill-switch**：`VLLM_GFX906_GDN_DEC_M1=0` 整体关闭，落回串行 kernel；
  串行 kernel 永不删除（兜底 + parity 锚）。
- **预期**：15–30 µs/次（判据 ≤40 µs，参照地平线 25.7 µs）；e2e −8~9 ms。

### D2（WP2-N，第二杠杆）：fp8 GEMV M=1 两级推进

现状：W8A16 路线（`run_block_w8a16`，:2509）M=1 已走 stream 变体 1（默认 mode=1，
rows 自动 8/4），块表 253.5 次 × 48.2 µs，330 GB/s。

- **第一级（零代码）**：env 扫描 `VLLM_DCU_FP8_GEMV_STREAM{=1,2}` ×
  `_ROWS{4,8,16,32,64}`（全部 per-process 读一次，免重建）。⚠️ rows/变体会改块内
  累加序（O4 先例曾破坏位级 parity）——贪心流零翻转是硬门，不豁免。
  `VLLM_DCU_FP8_STREAM_MAX_M` 保持 unset（M 门不碰）。
- **第二级（csrc，第一级 <400 GB/s 时启动）**：MS=1 专用 stream kernel
  （`fp8_emu_gemv_stream_m1_kernel`，env `VLLM_DCU_FP8_GEMV_M1=1` 门）：
  M=1 的 A 仅 1 行，删 A-restage 与 M 维索引开销、加深 K unroll；P0 探针独立
  实验打到过 451–621 GB/s（fp8_emu_gemm.cu:591-599 证据），现实空间 +30~80 GB/s。
  判据：热形状 per-op −15% 以上、e2e −1.5 ms 以上（两级合计）。
- **判据兜底**：per-op 达标但 e2e 不达标 ⇒ 如实记录"被框架开销吸收"，不采纳。

### D3（WP3）：host gap ~3 → ~1.5 ms（Python 侧，维持 S30b 方案不变）

`ITER_DETAILS=1` 先出 host 段分布表，再挑无数值影响的削减；合计 −1 ms 以上，
单项 <0.3 ms 不采纳。csrc 解禁不改变本 WP（host 大头在调度/Python）。

### D4（WP4）：AR 7.44 ms —— 记录不动

每层 2 次 × TP4 结构性，64.7 µs/op 已低于 90 µs 独立地板。与 csrc 无关。

### D5（显存路线，新增）

按"速度不回退前提下尽量少用显存"排序：

1. **V1 纯 decode cutover（主红利）**：draft 塔权重（5 层 bf16 + embedding）+
   draft KV 池 + spec 调度缓冲全部释放。固定 `UTIL=0.85` 下释放量自动转为 KV
   容量（tokens↑，速度中性）；切换前后各抓一次 serve log 启动统计
   （weight/KV GiB + #tokens）+ `hy-smi` 四卡 used，**差值入账，不预估**。
2. **V2 cudagraph capture 裁剪**：`cudagraph_capture_sizes [1,2,4,8,16] → [1,2,4]`。
   依据：MAX_SEQS=4 且无 spec ⇒ decode batch ≤4，capture 8/16 永不命中，其
   graph 池是纯浪费。门禁：启动日志确认 capture 清单、e2e step ±2%、显存差入账。
   ⚠️ 注记：若未来调高 MAX_SEQS 必须 re-add。
3. **可选臂（按场景选用，不默认）**：
   - `UTIL` 下调（如 0.85→0.75）：总 HBM 占用 ↓，KV tokens 相应 ↓；前提 =
     单流 262k 上下文容量仍满足（783k→~640k tokens），e2e 不变才收。
   - `MBT 4096→2048`：activation workspace 再减半（S24 L5 先例：M=4096 效率
     ≈ M=8192）；门禁 = TTFT ±3%（prefill 敏感则弃）。
4. **明确拒绝（留档，防止实施 agent 重开）**：
   - **KV fp8_e4m3**：前测容量 ×1.63 但 **ITL +41%，双不达标不晋级**
     （z100-longctx-vram-optimization-result.md §15）——与速度目标直接冲突；
   - **SSM state fp32→fp16**：GDN 递推的精度锚，长上下文漂移风险，省 ~72-144
     MB/秩不抵风险；
   - **decode 池布局迁移**（迁就 DEC_V2 的 v 连续门）：横跨 MTP/prefill 布局契约，
     D1 的"kernel 适配池"方案使此迁移永久无必要。

### D6（门禁与回滚纪律，全程强制）

- 每 WP：真实 shape 微测 → 数值对照 → 服务 smoke → 同口径 e2e → 文档；
  贪心流零翻转 = 所有数值类改动的硬门；e2e 报 N≥2 重启分布。
- 回滚优先级：env kill-switch（一行）> 臂脚本（不换脚本即回滚）> `.so` 换回
  （备份 md5 对齐 C7 纪律）。
- csrc 改动纪律（C7，z100-opt-round3-construction.md）：改前备份旧
  `_rocm_C.abi3.so` + md5 → `build-sync.sh` 重建 → 记录新 md5 → 禁止手改
  build-sync/temp 产物。
- 服务收尾：**纯 decode 原生臂留在 :8000 作为现役**（dflash 退役）。

## 5. 预期收益汇总

| 项 | 机制 | 预期 |
|---|---|---|
| WP1-N | GDN M=1 384-block 并行化 | 216→15-30 µs/次，e2e **−8~9 ms** |
| WP2-N | env 调参（+可选 MS=1 kernel） | e2e **−1.5~2 ms** |
| WP3 | host 段无损削减 | e2e **−1 ms** |
| 合计 | | **39.2 → 27–29 ms/tok（34–37 t/s）**，TTFT 0.74 s 保持 |
| V1 | draft 塔 + spec 缓冲释放 | 切换快照差入账 → KV 容量（速度中性） |
| V2 | capture [1,2,4] | graph 池差值入账 |
| V3 可选 | UTIL/MBT | 按场景，e2e 不变才收 |

## 6. 风险登记

| 风险 | 缓解 |
|---|---|
| butterfly 归约引发贪心流翻转 | 零翻转硬门；Plan-B = S30b triton 路线（留档）；或按 S29 WP-C 流程交用户裁决 |
| m1_native 与 combined 入口交互 | combined 内部就是同一 host 函数（已核实），门加在其内即双入口生效；DEBUG 打印确认 HIT |
| conv 路径意外变化 | 本方案不动 conv（与 triton 路线不同），kdiff 复核 conv kernel 时间持平即可 |
| build-sync 全量重 hipify | 只用 `build-sync.sh`，禁 `build-sync2.sh`（opt-round3 §C7 教训） |
| capture 裁剪遇 c>4 场景 | 注记 MAX_SEQS 联动；裁剪前 grep 启动日志 capture 清单 |
| KV fp8 被误启用 | D5.4 留档 ITL +41% 前测；KV_DTYPE 保持空 |
