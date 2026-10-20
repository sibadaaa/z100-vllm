# Z100 长上下文（>30k）变慢 + 显存占满 优化方案（S24）— v2 执行计划

> 平台：4×海光 DCU Z100（gfx906，DCU0/1 = Z100L@1600，DCU2/3 = Z100@1319），TP=4，
> `--enforce-eager`，DFlash2 k=7，Qwen3.8-27B-FP8（64 层 = 48 GDN 线性注意力 + 16 全注意力，
> hidden 5120 / 24 heads / 4 kv heads / head_dim 256，`full_attention_interval=4`），
> `--max-model-len 262144 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 --enable-prefix-caching --mamba-cache-mode align`。
> 现网基线 = S23 交付：`.so md5 d209af9585aaf0049394ced4a3c3211b`、
> `VLLM_DCU_FP8_DECODE_CFG=35`、`EMU_FASTPATH=1`、`QUANT_FASTPATH=1`。
> 关联：`docs/result/z100-longctx-vram-optimization-result.md`（S24 执行结果，持续追加）、
> `docs/design/z100-s24-l3-dynamic-k-design.md`、`docs/design/z100-s24-l4-kv-quant-design.md`、
> `.s23/S23_EXEC_PROTOCOL.md`、`.s22/S22_LONGCTX_DECODE.md`、
> `docs/result/z100-fp8-emu-optimization-result.md`（S23 交付）。

> **v2 修订（2026-10-20）**：
> ① 用户指令：**除 L6 外全部执行**，L6 移出范围（其基线已顺带测得，仅存档）；
> ② 并入 2026-10-19 W1/W2 执行结论（L2 端到端证伪、L1 wide-BK 内核级证伪、
> L3.1 SEGMENTS=4 确认、L5 MBT=4096 胜出未晋级、L5b 量化为 14.6% 页浪费）；
> ③ 基线口径修正：v1 §1.1 的 32k TTFT = 59.5 s 在 4 个全新实例上不可复现，
> **fresh 口径 = 93.4–93.6 s（~350 t/s）**（结果文档 §2.2/§6），本版目标全部改用 fresh 口径；
> ④ §3 加"状态"列，§4 每项加"执行状态/剩余工作"，§5 改为剩余波次 R1–R6。
> §0–§2/§6 的编号与内容保持向后兼容（结果文档引用的编号不变）。

---

## 0. 一句话结论（先纠正两个直觉）

1. **30k+ 变慢不是 decode、也不是显存不够，而是 prefill 的 FP8 模拟 GEMM 效率**。
   实测（探针见 §1.2）：fp8→fp16 LUT + `v_dot2_f32_f16` 模拟跑 prefill，每卡只有
   **8.0–8.8 TFLOP/s**，而 gfx906 的 dot2 理论上限是 **20.3 TF/s（DCU3）/ 24.6 TF/s（DCU0）**，
   即**只跑到 41% 峰值**。TTFT 可以精确预测：`TTFT ≈ 2 × 27e9 × tokens / (4 × 8.4e12)`，
   32k prompt 预测 52.7 s、fresh 实测 **93.5 s**（GEMM 占比 ~56%，其余为 attention/调度；
   v1 的 59.5 s 是 warm 实例口径）。30k+ 的"慢"是**近似线性于 token 数的算力墙**，
   靠调度/并发/显存改不动，只能提高内核效率或减少要算的 token。
   **执行修正（W1/W2）**：§1.2 的 41% 缺口已被三条路攻击过——prefill cfg 重选（L2）、
   宽 BK tile（L1）——**双双证伪**（§3/§4），standalone 内核数字不能预测 e2e（§1.4-3）。
   剩余可信杠杆：**verify/decode 侧（L3.1/L3.2）**、**KV 带宽（L4）**、**int8 dot4（L7）**。
2. **"VRAM 快被占满"是设计使然，不是长上下文导致的**：启动时 vLLM 按 `util=0.85`
   预分配 KV 池 **748,982 token（15.54 GiB/卡）**，加权重 9.04 GiB/卡，静态 ~94.5%
   （余量 1.8 GiB）。**单个 32k 请求的 KV 只占池子 4.3–6.7%**（`preemptions=0`）。
   L5 已量化并给出解：**MBT=4096 让 KV 池反而增大到 16.21 GiB、静态占用降到 91–92%
   （余量 ≈2.6–3.0 GiB），32k TTFT 仅 +0.7%** —— R1 晋级。

---

## 1. 基线（实测事实）

### 1.1 长上下文曲线（v1 口径，2026-10-19 23:19–23:35 +08:00，现网配置，单请求）

| prompt tokens | TTFT | prefill t/s | ITL/ token | 备注 |
|---|---|---|---|---|
| 512 | 1.23 s | 417 | 134 ms | 与用户 512 prompt / TTFT 1979 ms 同量级 |
| 2,048 | 4.18 s | 489 | 152 ms | |
| 4,096 | 6.27 s | 654 | 149 ms | |
| 8,192 | 16.58 s | 494 | 197 ms | 首次（冷） |
| 16,384 | 24.37 s | 672 | 177 ms | |
| 32,768 | 59.47 s | 551 | 202 ms | **warm 实例口径，fresh 不可复现** |
| 8,192（第二次） | 2.74 s | 2,994 | 141 ms | 同 prompt 命中前缀缓存 |

**口径修正（结果文档 §2.2/§6）**：32k 在 4 个全新实例上 = **93.4–93.6 s（~350 t/s）**，
引擎日志同 ~350 t/s；8k/512 与 v1 完全一致；差异集中在 3D prefill attention 路径
启用的 ≥16k 段。**后续所有 A/B 以 fresh 口径 93.5 s 为 32k 基线**，同窗口内自比。

### 1.2 prefill GEMM 的效率墙（`.s24/probes/prefill_gemm_cfg_probe.py`，DCU3）

cfg 显式扫描（5120:4352）：M=512 cfg0 8.00 TF/s（cfg3 7.76）；M=2048 cfg0 8.43
（cfg3 8.80，+4.4%）；decode 窄 tile（cfg16/35）在 prefill 形状 4.1 TF/s（2× 差距）。
**效率墙在模拟内核本身（41% 峰值），不在 tile 选择**。

### 1.3 显存账（启动日志实测）

| 项 | 每卡 | 说明 |
|---|---|---|
| 模型加载（含 DFlash2 draft） | **9.04 GiB** | `Model loading took 9.04 GiB` |
| KV 池（MBT=8192） | **15.54 GiB** | 748,982 token；每 token 每卡 22.34 KB |
| 实测占用 | **30.9 GiB（94.5%）** | 余量 1.8 GiB |
| 32k 单请求 KV | ≈0.5 GiB（4.3%） | `KV usage 3.6–6.7%`，`preemptions=0` |
| 长上下文并发上限 | 262,144 token × 2.86 | 30k 档 ≈24 条、120k 档 ≈6 条 |

结构约束：`--mamba-cache-mode align` 把 attention block 抬到 880 token；
15 组 = 10×GDN(5) + 4×full(5) + 1×draft(5)；**padding 浪费 ≈3.25 KB/token = 池的
14.6% ≈ 2.27 GiB/卡**（结果文档 §5 量化），group_size=8 可净回收 ≈6.2%（L5b，R6 可选）。

### 1.4 W1/W2 执行后新增事实（2026-10-19，结果文档 §1–§3）

1. **M 曲线（停机窗口，5 生产形状 × M∈{512..8192}）**：cfg3 在 M≥1024 全形状最优
   （M=8192 +5.5~7.9%），M=512 略差（−1~6.5%）。
2. **真实形状清单修正**：每卡 GEMM 含 `1280:17408`（down_proj，nkb=136）与
   `2048:5120`（qkv）——任何 prefill tile 门禁必须覆盖全部 7 形状，且 **M≤8 的
   cfg 通道必须保持 0**（down_proj 的 nkb=136 会走 fused fallback，cfg 污染即 4.6× 退化）。
3. **standalone → e2e 反转**：cfg3 standalone +6% ⇒ e2e TTFT **+13~19%**（全线回归）。
   空载 GPU 独立探针系统性高估高占用 tile；**晋级判据必须是 e2e 同窗口 A/B**。
4. **wide-BK（cfg36–39）内核级证伪**：prefill 形状 −25~−63%；S23 wide2 的收益
   需要 decode 的占用率（BM8），prefill 的 LDS 预算（25–51 KiB）压 occupancy 到
   1–2 blocks/CU，加宽 K 反而放大每轮 LDS footprint。未做 e2e（内核级已判死）。
5. **长 prefill 吞吐衰减**：494（8k）→ 350（32k）→ 245（64k）→ 156 t/s（128k），
   **≥64k 段 O(L²) attention 占比上升、TTFT 偏离线性** —— 64k+ 的杠杆在 L3/L4。
6. **64k/128k soak（MBT=4096，3 reps）通过**：0 OOM/retry/preempt，KV 峰值 9.9%。

---

## 2. 目标与验收（v2，fresh 口径）

**目标（可度量）**：
* TTFT：32k **93.5 → ≤65 s（−30%）**（杠杆 = L3.2 + L4 + L3.1，GEMM 侧已被证伪）；
  64k ≤ 240 s；512/8k 不回归（≤2%）。
* ITL（真实文本 accept≈7）：32k 上下文 **≤150 ms**（现 165–200 ms）。
* 显存：**MBT=4096 晋级为 launcher 默认**（池 16.21 GiB、静态 91–92%、余量 ≥2.5 GiB），
  长 prefill soak **64k×10 + 128k×6 无 OOM/retry/preempt**（结果文档 §4.1 的 3-rep 版加严）。
* 正确性：非量化项 greedy parity 逐 token 不变；L4 是唯一允许改数值的项，走独立质量门禁。

**统一口径**（沿用 S23/S24 纪律）：同窗口 A/B（`tools/dcu/s24_ab.sh`，GPU lock +
树指纹 + 单元属性 diff + 每臂 env 落地证明 + 收尾恢复默认 + smoke）、每臂 ≥4 probe、
p50 + 均值、禁止跨窗口比差；长上下文探针 `tools/dcu/s24_longctx.py`
（**每次重复使用不同 prompt 内容**，避免 prefix cache 吃掉 A/B）；每次窗口带 canary。

---

## 3. 工作项总表（v2 状态版）

| ID | 工作项 | 性质 | 状态（2026-10-20） | 剩余动作 | 波次 |
|---|---|---|---|---|---|
| L1 | prefill 宽 K tile（cfg36–40） | kernel | **证伪关闭（含残留点 BM16@BK64 = −50%）**；cfg36-40 代码保留（默认 0，parity/ISA/pytest 全过） | 无；tile 空间穷尽，效率墙只剩 L7 路线 | — |
| L2 | prefill cfg 微调（PREFILL_CFG=3） | 配置 | **证伪关闭**（standalone +6% ⇒ e2e +13~19%） | 无（env 保留，默认 0；M≤8 通道保持 0） | — |
| L3.1 | verify/decode attention 调优 | kernel | **内核级 −45~−52% 实测**（BLOCK_M=16，M∈{4,8} 全上下文）；e2e −7.3% 32k ITL；**parity 破（tl.dot 形态变）⇒ 不晋级** | 以 `VLLM_GFX906_ATTN_DECODE_BLOCK_M/WARPS/SEGMENTS`（默认 0）opt-in 交付 | — |
| L3.2 | 按上下文长度动态降 k | 调度 | **设计稿就绪**（`z100-s24-l3-dynamic-k-design.md`），未实现 | 实现（~80 行，runner:5238/5372 的 per-request 截断）+ 门禁 + A/B | R2 |
| L4 | KV cache 量化（fp8 e4m3 + per-token-per-head scale） | 结构 | **设计稿就绪**（`z100-s24-l4-kv-quant-design.md`），未实现 | 实现（~300 行，Triton 三路径 dequant + 独立 quant kernel）+ 质量门禁 + A/B | R5 |
| L5 | 显存余量：UTIL/MBT 扫描 + soak | 配置 | **MBT=4096 胜出**（池 +0.67 GiB、余量 ≥2.6 GiB、32k TTFT +0.7%、3-rep soak 过）；**未晋级** | 晋级 launcher 默认 + 加严 soak（64k×10 + 128k×6）+ mns32 臂（可选） | R1 |
| L5b | 880 页组对齐 padding 回收（group_size=8） | 上游代码 | **已量化**（14.6% 浪费，可回收 ≈6.2% 池 ≈46k token），未动代码 | 可选实现 + A/B（风险在上游 `kv_cache_planning.py` 共享代码） | R6（可选） |
| L6 | 前缀缓存命中率 | 工具/流程 | **用户指令排除**（基线已顺带测得：grow 79.3% 命中、TTFT 1/3.8 per-token，结果文档 §6） | 无 | — |
| L7 | int8 `v_dot4_i32_i8` 权重路线 | kernel+精度 | **调研完成：21–23.3 TF/s（~2.6× 余量），合成数据 rel_err 0.02%，go 倾向** | 建议 S25 立项（离线重量化 + 质量门禁）；本轮不进生产 | R4 完成 |

**收益上限说明**：TTFT 的 GEMM 项转化率接近 100% 但 GEMM 侧已被证伪封顶（8.4 TF/s）；
剩余 TTFT 杠杆在 **verify/decode 的 KV 扫描成本（L3）与 KV 带宽（L4）**——32k decode
单步 KV 读 ≈0.5 GB/卡（full 16 层 × 32k × 1 KB），128k ≈1.9 GB/卡，
ITL 随上下文的恶化（134→202 ms）主因在此；≥64k 还有 O(L²) attention 项（§1.4-5）。

---

## 4. 单项规格（v2：状态 + 剩余工作详规）

### L1 — prefill GEMM 宽 K 分块（已证伪关闭；残留一枪）

**执行状态**：cfg36–39 已实现（`csrc/rocm/fp8_emu_gemm.cu`，默认 0，cpb 不匹配回退
legacy tile），parity 7 形状×6 个 M 全 PASS、ISA PASS（80,568 v_dot2 / 0 mfma）、
pytest 123 passed；性能证伪（§1.4-4）。构建工件 `/mnt/game/perf1/_rocm_C.ea0ec94d….so`
保留为探针工件，**生产保持 d209af95**。

**剩余工作（R4，一枪）**：唯一未测的设计点 **BM16/BN32 @ BK64**（LDS ≈12.7 KiB →
4–5 blocks/CU，介于 decode wide2 与已证伪 tile 之间）。若它也 <cfg0 −10%，
L1 彻底关闭，prefill GEMM 效率墙的 tile 空间穷尽，剩余路径只有 L7。
**门禁**：l1_wide2_check 口径（7 形状 × M∈{1,8,512,2048,8192}，torch.equal + TF/s 表）；
M≤8 通道回归 0；**不做 e2e**（沿用 §1.4-3 判据：内核级 <+10% 不进窗口）。
**写域**：`csrc/rocm/fp8_emu_gemm.cu`（+1 个 case）；回滚 = 不改默认，无需回滚。

### L2 — prefill cfg 微调（已证伪关闭）

**执行状态**：M 曲线 + FFN/下投影补测完成；端到端 A/B（w1/L2/l2 窗口）**全线回归
+12.9~24.5%**（TTFT），parity 逐 token 一致（数值不变性成立但性能证伪）。
**保留资产**：`VLLM_DCU_FP8_PREFILL_CFG` env（默认 0）；
**硬约束**（写进任何后续 prefill cfg 改动）：M≤8 必须返回 0（fused fallback 污染通道，
结果文档 §2.1-2）。

### L3.1 — 长上下文 attention：prefill 侧已确认，verify 侧待扫（R3）

**执行状态**：w2/L31/a3 窗口：SEGMENTS∈{2,4,8} × 16k/32k × 2 reps —— **SEGMENTS=4
维持最优**（seg8 在 32k 的 −0.3% 是噪声量级）；verify 侧未动。

**剩余工作（R3）**：
1. 新内核探针 `.s24/probes/l31_verify_scan.py`：直接驱动
   `triton_unified_attention` 的 decode/verify 路径（M∈{1,3,8} × KV∈{8k,16k,32k,64k}），
   扫 `_decode_num_segments` 输出值 ±1 档、`_decode_block_m` 候选、num_warps ±1；
   基线指标 = S22 实测的单层 KV 扫描带宽 ~74 GB/s，目标 ≥150 GB/s（v1 §L3.1 未验证项）。
   探针在停机窗口跑（需要真实 KV 池形状）。
2. 若某参数 ≥+15%：把该参数提成 env（默认 = 现行为），e2e 同窗口 A/B
   （lens {512, 8k, 32k} × 4 reps + ITL），晋级判据 = 32k ITL −10% 且 512/8k 不回归。
3. 若无 win：记录关闭，ITL 杠杆只剩 L3.2/L4。

**写域**：`vllm/v1/attention/ops/triton_unified_attention.py`（+env 旋钮，默认不变）、
`.s24/probes/l31_verify_scan.py`；回滚 = env 默认不变。

### L3.2 — 动态降 k（R2，ITL 主杠杆）

**设计**：见 `docs/design/z100-s24-l3-dynamic-k-design.md`（机制：截断 k 不改贪心输出，
verify 只算前 1+k' 个位置；策略：L≤16k→7；16k<L≤64k→accept_ema≥5?7:3；
L>64k→accept_ema≥5?3:1；ema=0.9/0.1；三个阈值与 ema 系数走 env
`VLLM_S24_DYNK_LO/HI/EMA`，未设=关闭）。

**实现面（已核实锚点）**：`vllm/v1/worker/gpu_model_runner.py:5238/5372` 的
`num_spec_tokens_to_schedule` 处按请求截断（drafter 照常出 7 个，只截 verify 长度）；
accept_ema 状态挂 request 对象；DFlash2 drafter（`vllm/v1/spec_decode/dflash.py`）只读。
估计 ~80 行，默认关闭。

**门禁**：① greedy parity 8/8 prompt 逐 token 一致（按构造应全绿——k 截断不改输出）；
② 32k 真实文本 ITL ≤150 ms（目标）或 ≥−15%；③ 输出 t/s ≥ 对照臂 95%（步数代价）；
④ 短上下文（≤4k）不回归；⑤ 同窗口 A/B、4 probe、canary。
**风险**：per-request k' 使 verify 的 M 变化（M=3/4 等）→ GEMV/GEMM 路径边界复查
（M≤8 全走 GEMV/stream，S24 §2.1 已确认 cfg 通道不可达）。
**写域**：`vllm/v1/worker/gpu_model_runner.py`、（若需）`vllm/v1/spec_decode/dflash.py`（只读）、
`tools/dcu/s24_arms/dynk*.env`；回滚 = env 不设即关。

### L4 — KV cache 量化（R5，容量+带宽，唯一改数值项）

**设计**：见 `docs/design/z100-s24-l4-kv-quant-design.md`（KV 以 e4m3 存储 +
per-token-per-head fp32 scale（amax/448），Triton unified attention 读入后用现有
精确 fp8→fp16 LUT + 一次乘法 dequant；decode(3D)/verify(M≤8 3D)/prefill(2D/3D)
三路径；GDN 状态不量化；写路径独立 quant kernel；draft KV 分开关）。

**实现步骤**：
1. config/validation：`--kv-cache-dtype fp8_e4m3` 在 gfx906 后端放行（校验 + 告警）；
2. KV 写路径 quant kernel（K/V 投影输出 → fp8+scale）；
3. Triton 三路径读侧 dequant 分支（K 与 V 各一次乘法，在 QK^T 之前）；
4. `--mamba-cache-mode align` 交互复查（页字节数减半 → attention 组页大小检查，
   必要时回退非对齐模式并量化 880 对齐损失）。

**质量门禁（硬）**：① 8 条固定 prompt：与 fp16-KV 的输出 perplexity / logit KL 偏差
≤1%（含 prefill 精度回归口径）；② 长稳 1000 步 decode 无 NaN（含 64k 驻留）；
③ decode ITL −20% **或** 池容量 ×1.8 至少一项 e2e 成立；④ 同窗口 A/B。
**不达标即不启用**（默认关闭，证据存档）。
**写域**：`vllm/v1/attention/ops/triton_unified_attention.py`、
`vllm/v1/attention/backends/`（dtype 校验）、`vllm/v1/worker/gpu_model_runner.py`（quant kernel 调用）、
`tools/dcu/s24_arms/kvfp8*.env`；回滚 = env/flag 不开即关。

### L5 — 显存余量（R1：晋级 + 加严 soak）

**执行状态**：w1/L5/l5 窗口 3 臂 A/B —— **MBT=4096 胜出**：KV 池 15.54→**16.21 GiB**
（783,993 token）、静态 94–95%→**91–92%**（余量 ≈2.6–3.0 GiB）、32k TTFT +0.7%、
ITL 不变；UTIL=0.82 备选（再 +0.96 GiB 余量、−0.96 GiB 池、+2.9% TTFT）；
3-rep soak（64k/128k）0 OOM/retry/preempt、KV 峰值 9.9%。

**剩余工作（R1）**：
1. **晋级**：`tools/dcu/serve_host_dflash2.sh` 的 `MBT` 默认 `${MBT:-8192}` → `${MBT:-4096}`
   （一行，env 仍可覆盖回 8192），生产重启 + `s24_logscan` 验证（池 16.21 GiB、
   group sizes、KV 并发上限 783,993 口径）+ 固定 prompt smoke。
2. **加严 soak**（生产配置 = MBT=4096，后台长窗口）：64k×10 + 128k×6，
   验收 = 0 OOM/0 retry/0 preempt、p99 TTFT 稳定、KV usage 峰值记录。
3. （可选）mns32 臂：`max-num-seqs=32` 单臂 + 显存画像 diff，若池无变化且
   TTFT 无回归才考虑记录为第三旋钮。
**回滚**：`MBT=8192` 重启即回 v1 口径（KV 池 15.54 GiB）。

### L5b — 组对齐 padding 回收（R6 可选）

**已量化**：group_size 5 → 8 可把池浪费 14.6% → 8.4%（净回收 ≈6.2% ≈ 46k token，
并发 +6%）。**实现面**：上游 `vllm/v1/core/kv_cache_planning.py`（或本 fork 同名文件）
的分组策略 —— 风险在共享代码、收益中等。**执行条件**：R1–R5 全部落地且余量充足时
才做；门禁 = group sizes 日志（15 组 → group_size=8 布局）、池 token 748,982→~795k
（MBT=4096 口径 ~830k）、parity、A/B 窗口。不满足条件则存档关闭。

### L6 — 前缀缓存命中率（用户指令排除）

v1 的服务端/客户端工程项**全部移出范围**。基线数据已存档（结果文档 §6）：
grow 模式命中 79.3%、TTFT 0.50 ms/新 token vs independent 1.88 ms/tok；
精确重复 32k prompt TTFT 93.6→4.48 s。供运维侧自行参考，不再投入。

### L7 — int8 dot4 路线研究（R4）

**依据**：gfx906 `v_dot4_i32_i8` = 4 MAC/instr（dot2 的 2×）；`csrc/rocm/w8a8_sdot4.cu`
（S11l/S11m：decode 3.9×，**改 greedy token**）。若 FP8 权重离线重量化 int8
（per-128-block scale）+ 激活 per-token-group int8，prefill 理论 ~16 TF/s（vs 现 8.4）。

**步骤（仅调研，不进生产）**：
1. ISA 确认：`objdump` w8a8_sdot4 代码对象统计 `v_dot4_i32_i8`（复用 isa_gate 遍历逻辑）；
2. standalone TF/s：复用/改造 w8a8_sdot4 的 GEMM 探针，测 7 个每卡形状 × M∈{2048,8192}，
   与 fp8-emu cfg0 同表对比（目标 ≥12 TF/s 才有工程意义）；
3. 精度评估：离线把 3 个代表性层的 fp8 权重 dequant→requant int8（per-128-block），
   激活 per-token-group-128 int8，比较 int8 路线 vs fp8-emu vs bf16 的相对误差；
4. 产出 go/no-go 报告（含：改数值的代价、质量门禁设计草案、工作量估计），
   **本轮不实现生产路径**；若 go，另立计划（S25 候选）。
**写域**：`.s24/probes/l7_*.py`、`docs/result/`（报告）；不动生产与共享代码。

---

## 5. 执行编排（v2：剩余波次）

| 波次 | 内容 | 窗口/重启 | 依赖 |
|---|---|---|---|
| **R1（0.5 天）** | L5 晋级（launcher MBT 默认 4096 + 重启 + logscan + smoke）→ 随即 **soak 长窗口**（64k×10 + 128k×6，后台 ~2.5 h）；期间并行写 L3.2 代码（不占 GPU） | 1 次重启 | — |
| **R2（1 天）** | L3.2 动态降 k：实现 + 静态检查 + A/B 窗口（control vs DYNK，lens {512,8k,32k}×4 + 真实文本 ITL + parity） | 1 窗口（2 次重启） | R1 的 soak 不占用窗口即可 |
| **R3（0.5–1 天）** | L3.1 verify 内核扫描（**停机窗口**，新探针）→ 有 win 则 env 旋钮 + e2e 窗口；无 win 关闭 | 1 停机窗口（+可选 1 窗口） | — |
| **R4（0.5 天）** | L1 残留 BM16 一枪（并入 R3 停机窗口）+ L7 调研（探针 + 精度 + 报告，与 R3 同窗口或停机窗口） | 0–1 窗口 | L1 需重建 .so（s24_l1_build.sh 口径） |
| **R5（2–3 天）** | L4 KV 量化：实现（4 步）+ 质量门禁 + A/B 窗口（含 64k 驻留长稳） | 2–3 窗口 | 建议在 L3.2 落地后（同内核文件串行） |
| **R6（0.5 天）** | L5b（可选，条件满足才做）+ 全项叠加复测（32k/64k TTFT、ITL、显存画像）+ 回滚演练 + 结果文档定稿 | 1–2 窗口 | R1–R5 |

**窗口纪律**：每窗口 = `tools/dcu/s24_ab.sh`（GPU lock、py_compile、容器 import smoke、
树指纹、单元属性 diff、每臂 env 落地证明、收尾恢复默认 + smoke）；内核探针在
`z100-regress` 容器内、单卡、`flock /src/.s23/gpu.lock`；停机窗口才能跑大显存探针
（mscan/l1-check 口径），生产在跑时限 64 MiB。
**写域总表**：`csrc/rocm/fp8_emu_gemm.cu`（L1 残留）；`vllm/v1/worker/gpu_model_runner.py`
（L3.2/L4）；`vllm/v1/attention/ops/triton_unified_attention.py`（L3.1/L4）；
`vllm/v1/attention/backends/`（L4 校验）；`tools/dcu/serve_host_dflash2.sh`（L5 一行）；
`tools/dcu/s24_arms/*`、`.s24/probes/*`（新探针）；文档 `docs/design|result/`。

---

## 6. 明确不做 / 已排除（v2 追加证伪项）

| 不做 | 原因 |
|---|---|
| **L6 前缀缓存命中率工程** | **用户指令排除**（2026-10-20）；基线已存档供运维参考 |
| 靠"换 prefill cfg"拿大收益 | **已证伪**：cfg3 standalone +6% ⇒ e2e +13~19%（§1.4-3） |
| prefill 宽 BK tile（cfg36–39 已实现者） | **已证伪**：内核级 −25~−63%，wide-BK 在 prefill 占用级下放大 LDS footprint（§1.4-4） |
| 靠"降 UTIL 腾显存"提速 | 32k 单请求 KV 只占 4.3–6.7%，速度与 KV 占用无关（§1.3）；UTIL=0.82 降级为备选 |
| split-K（无确定性设计）/ 改 epilogue 顺序 | 违反仓库可复现纪律（S23 同款禁区） |
| 打开 CUDA graph 消派发税 | S23 O7 已证：piecewise 的阻塞点是编译期既有 Dynamo graph break |
| 动 `--max-model-len` 当省显存手段 | KV 池按 util 预分配，max-model-len 不改变池大小 |
| 任何 prefill cfg 规则覆盖 M≤8 通道 | down_proj nkb=136 的 fused fallback 污染通道（结果文档 §2.1-2）：M≤8 必须返回 0 |

---

## 7. 参考

* 执行结果（持续追加）：`docs/result/z100-longctx-vram-optimization-result.md`
* 设计稿：`docs/design/z100-s24-l3-dynamic-k-design.md`、`docs/design/z100-s24-l4-kv-quant-design.md`
* 本轮实测探针：`.s24/probes/longctx_probe.py`、`prefill_gemm_cfg_probe.py`、
  `prefill_gemm_cfg_mscan.py`、`l1_wide2_check.py`（R3/R4 新增 `l31_verify_scan.py`、`l7_*`）
* 机理与历史数据：`.s22/S22_LONGCTX_DECODE.md`、`tools/dcu/perf1_attn_verify.md`、
  `.s23/fp8_emu_kernel_fixed_times.md`
* 工具与纪律：`.s23/S23_EXEC_PROTOCOL.md`、`tools/dcu/s24_ab.sh`（窗口 A/B + parity + 回滚）、
  `tools/dcu/s24_longctx.py`、`tools/dcu/s24_logscan.py`、`tools/dcu/s24_prefix_replay.py`
* 现网配置与回滚：`tools/dcu/serve_host_dflash2.sh`、`docs/result/z100-fp8-emu-optimization-result.md` §11/§16
