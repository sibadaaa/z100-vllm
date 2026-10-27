# S30b 施工图：纯 decode 路线 —— GDN M=1 triton 并行化 → fp8 GEMV 调参 → host gap（2026-10-26）

> **交给实施 agent 的完整任务书**。上游计划：`docs/design/z100/archive/z100-s30b-pure-decode-plan.md`
> （基线实测与块表以它为准，本文不重推导）。按 WP 顺序执行；每个 WP 独立可验收、
> 独立回滚；任何 WP 失败不阻塞其他 WP。
>
> **当前基线**（勿重推导，s29-nospec 口径 = `.s29/s20_s29-nospec_20261025-*.json`）：
> 服务 = `.s29/graph_profile_launch_nospec.sh`（piecewise cudagraph capture [1,2,4,8,16]、
> workspace pin `ROCBLAS_WORKSPACE_CONFIG=:4096:8` + `HIPBLAS_WORKSPACE_CONFIG=:4096:8`、
> BM16 `VLLM_GFX906_ATTN_DECODE_BLOCK_M=16`、无 spec 参数），
> 7469 真实文档（`docs/design/z100/decode-prefill/z100-dual-prefill-reconciliation-projection.md`
> `--truncate-tokens 7469`）/ 96 max-tokens / 3 reps：
> **step p50 = 39.24 ms，TTFT = 739 ms，单流 21.33 t/s**。
> 纯 decode 路线全部完成后目标 **27–29 ms/tok（34–37 t/s）**；WP1 单项达成即反超
> dflash 现状 30.9 ms/tok（对照 `docs/result/` 的 dflash 臂数据）。
>
> **本施工图已核实的代码事实**（2026-10-26 实查，实施时如对不上以代码为准并停下报告）：
>
> - 模型 Qwen3.8-27B-FP8：64 层、`full_attention_interval=4` → **48 层 GDN**（块表
>   47.5 次/round 与之吻合）、hidden 5120、`linear_num_value_heads=48`（TP4 → **12/秩**）、
>   `linear_num_key_heads=16`（→ **4/秩**）、K=V=128。
> - 纯 decode M=1 的 GDN 实际调用链（`vllm/model_executor/layers/mamba/gdn_linear_attn.py:1200`）：
>   `enable_combined_packed_decode`（gfx906 默认开）→
>   `causal_conv1d_recurrent_gated_delta_rule_packed_decode`（`fla/ops/fused_recurrent.py:746`）
>   → ratio2 门 `kv_heads == 2*heads` = 12 ≠ 8 **MISS**（fused_recurrent.py:787）
>   → `_C::causal_conv1d_recurrent_gated_delta_rule_gfx906_packed_decode`（combined，C++ 内部
>   conv kernel + GDN 两连发）→ `fused_recurrent_gated_delta_rule_gfx906_packed_decode`
>   host 函数（`csrc/mamba/gfx906_decode_kernels.cu:2786`）。
> - **DEC_V2 warp-cooperative 快路在生产 MISS**：门要求
>   `state_stride_v==1 && state_stride_k==value_dim`（gfx906_decode_kernels.cu:2871），
>   而 decode 路径的 SSM 池是原生 `[slot][head][v][k]` k 连续布局
>   （`use_transposed_state_for_packed_decode=False`，gdn_linear_attn.py:446，
>   `_uses_transposed_temporal_state()` 原样返回，:567）→ stride_v=128 → 落到**串行
>   kernel** `fused_recurrent_gated_delta_rule_gfx906_packed_decode_kernel`
>   （:1234）：total = 1×12×128 = 1536，threads=256（`VLLM_GDN_PACKED_DECODE_THREADS`
>   默认 256）→ **grid [6,1,1]，每线程持一个 (head, v) 列、k 维 128 步串行**——
>   即计划块表里 216 µs 的欠并行 kernel。V2 快路本身（`_mtp_v2_kernel`，dflash 侧
>   同骨架 25.7 µs）要求 v 连续池，原生布局下不可用。
> - csrc 侧已有 `VLLM_GFX906_GDN_DEC_V2` 开关与 host 端门，但**改它 = 动 csrc，禁止**
>   （见计划禁止事项）。WP1 因此走 triton 重写、绕开 csrc，与计划一致。

---

## 0. 环境须知（先读，全是踩过的坑）

1. **GPU 实测必须在 DSH 沙箱外**。沙箱挡 HSA 的 /sys、/dev 访问（`hsa_init=4104`、
   "No CUDA GPUs" 是假象）。正路：`docker exec fp8-w8a16-gpu bash -c '...'`，或
   沙箱外宿主 shell（**宿主直跑是唯一能跑 pytest 的路径**）。
2. **服务生命周期**（每次重启 ~4.5 min；任何时刻只有一个服务；收尾把 dflash
   生产臂留在 :8000）：
   ```bash
   cd $WORKDIR
   # 起 nospec 臂（后台）：
   MAX_SEQS=4 bash .s29/graph_profile_launch_nospec.sh > .s30b/serve-<arm>.log 2>&1 &
   # 等 ready（最多 ~5 min）：
   for i in $(seq 1 24); do sleep 15; code=$(curl -s -o /dev/null -w "%{http_code}" \
     --max-time 3 http://127.0.0.1:8000/health); [ "$code" = 200 ] && break; done
   # 停（正则转义，防 pkill 自杀）：
   pkill -f "entrypoints[.]cli"; sleep 12; ps aux | grep "[e]ntrypoints" | wc -l   # 应为 0
   ```
   换臂 = 停 → 改 env → 起。**改 env 后必须重启进程**：WP2 的
   `VLLM_DCU_FP8_GEMV_STREAM*` 与 WP1 的所有新开关都是 per-process 读一次
   （`static const` / `functools.lru_cache`），热改无效。
3. **探针鉴权**：服务带 API key（`$SECRETS/api-key`），探针套壳见
   S29 施工图 §0.3（`tools/dcu/s23_common.py install_auth()` + exec
   `s20_single_probe.py`），tag 用 `s30b-<arm>`、out 用 `.s30b/`。
4. **环境核验**：每次起服务后核对 serve log 的 `[host-serve] policy: FP_FULL ...`
   指纹行 + `/proc/<pid>/environ` 中本 WP 的开关实际值；结果文档引用指纹。
5. **Python 侧改动重启即生效**；csrc 改动需要构建树同步——**本项目禁止动 csrc**，
   出现"必须动 csrc 才能继续"的结论时停下写文档，不顺手开洞。
6. **度量纪律**（AGENTS.md，逐 WP 强制）：真实 shape 微测 → 数值对照 → 服务
   smoke → 同口径 e2e → 文档。原始 JSON 落 `.s30b/`；结果文档
   `docs/result/z100-s30b-<wp>.md`；微测可复现命令必须写进文档。
7. **纯 decode 对比结论必须同时报 TTFT 与 t/s 两个口径**（计划禁止事项；
   交互场景 TTFT 常常比吞吐更决定体感）。

**共用门禁命令**（每个 WP 的 e2e / parity 板都跑这两条，口径与 s29-nospec 完全一致）：

```bash
# E2E：7469 真实文档 / 96 tok / 3 reps（srv_step_ms、ttft_ms、out_tok_per_s 三个数为准）
.venv/bin/python tools/dcu/s20_single_probe.py \
  --prompt-file docs/design/z100/decode-prefill/z100-dual-prefill-reconciliation-projection.md \
  --truncate-tokens 7469 --max-tokens 96 --repeat 3 \
  --tag s30b-<arm> --out .s30b

# 贪心数值板：temperature=0 流式抓 96 token 的 token_ids + top5 logprobs
# （复用 .s29/capture.py，3 reps；对照组 = 改动前同命令的 JSON）
```

验收通行标准（除 WP 各自判据外都必须满足）：
- 96-token 贪心流 token 序列与改动前**零翻转**；
- `srv_tok_per_step == 1.0` 保持（无 accept、无丢 token）；
- TTFT 不劣化超过 +3%（prefill 路径不得被本 WP 波及）；
- N≥2 次重启的 e2e 分布（step p50 报 min/p50/max），单次重启不作数。

---

## WP1：GDN M=1 递推并行化（−8~9 ms，本路线最大杠杆，最先做）

### 1.0 现状与手术点（已核实，勿重查）

- 216 µs 的 `fused_recurrent_gated_delta_rule_gfx906_packed_decode_kernel`：grid [6,1,1]、
  256 线程、每线程一个 (head∈12, v∈128) 列、k=128 串行——60 CU 的卡上只有 1536 个线程，
  纯延迟型。
- **手术点在 Python 层**：`fla/ops/fused_recurrent.py:497`
  `fused_recurrent_gated_delta_rule_packed_decode()` 的 `_is_gfx906_rocm()` 分支。
  到达该函数的前提 = 层走**非 combined** 路径（gdn_linear_attn.py:1239-1272 的
  else 分支：`causal_conv1d_update` + 本函数），由 env
  `VLLM_QWEN35_COMBINED_PACKED_DECODE=0` 切换（读点 gdn_linear_attn.py:432-443）。
- 可借用的并行骨架：`fla/ops/gfx906_gdn_mtp_triton.py` 的 `_gdn_mtp_update_kernel`
  （grid `(VD/BV, kv_heads, requests)`、行主序状态读写、B 行 = 512B 连续）。它是
  MTP 契约（cu_seqlens / num_accepted 链），不能直接复用，但**布局处理与 launch 纪律
  照抄**；该文件同时证明了 triton kernel 在 piecewise cudagraph + 本生产启动器下
  capture 安全（`VLLM_QWEN35_MTP_GDN_TRITON=1` 已在生产 dflash 臂运行）。

### 1.1 改动清单（仅两个文件 + 一个新文件，全部 csrc 外）

1. **新文件 `vllm/model_executor/layers/fla/ops/gfx906_gdn_decode_triton.py`**，
   模块纪律照抄 `gfx906_gdn_mtp_triton.py`（enabled() env 函数、launch cfg lru_cache、
   docstring 写明契约与验证方式）。内容 = M=1 契约的精简版：

   - 入参签名对齐 `_C::fused_recurrent_gated_delta_rule_gfx906_packed_decode` 的
     Python 视角（fused_recurrent.py:499-512 的实参表）：`mixed_qkv [T, 2560]`、
     `a [T,12]`、`b [T,12]`、`A_log [12]`、`dt_bias [12]`、`state [slots,12,128,128]`
     （**原生布局，stride(2)=128、stride(3)=1，stride 全部传参不假设**）、`out
     [T,1,12,128]`、`state_indices [T] int32`、`scale`、`use_qk_l2norm_in_kernel`、
     `use_tiled_qk_head_mapping`、`use_transposed_state`。
   - 语义契约（与 :1234 串行 kernel 逐条对齐，每条都要在 parity 里覆盖）：
     - 头映射：`q_head = tiled ? hv % H : hv // (HV/H)`；q 偏移 `q_head*K`，k 偏移
       `H*K + q_head*K`，v 偏移 `2*H*K + hv*V`（packed 布局，:1273-1275）。
     - q/k l2norm：`rsqrt(sum(x²)+1e-6)`，scale 只乘 q（:457-460 / :1326-1336）。
     - 门控：`x = a[t,hv] + dt_bias[hv]`；softplus 阈值 **20.0**
       （x≤20 → `log1p(exp(x))`，否则 x）；`decay = exp(-exp(A_log)*softplus)`；
       `beta = sigmoid(b[t,hv])`（:1531-1544）。
     - 状态更新（fp32 累加）：`h *= decay`；`v' = (v − Σ_k h·k) * beta`；
       `h += v' ⊗ k`；`out = Σ_k h · (q*scale_normed)`（:471-475）。
     - **slot<0**：out 该 (t,hv,·) 段写 0、**不读不写 state**（:1357-1361 的
       out=0 提前返回语义；T=1 时整 token 无效 = 全 0 输出）。
     - dtype：out 写 `q.dtype`（fp16），state 保持 fp32（本模型池为 fp32；
       kernel 对 state dtype 用 `tl.load(...).to(tl.float32)` / 存回
       `state_ptr.dtype.element_ty`，兼容 fp16/fp32 池）。
   - 网格：`grid = (cdiv(VD, BV), HV, T)`，T=1 ⇒ `(VD/BV, 12, 1)`。每个 program
     持 BV 个 v 行 × 128 k 的状态块（行主序 512B 连续读写）。扫描矩阵
     `(BV, num_warps) ∈ {(4,1), (8,1), (8,2), (16,2), (16,4)}`（mtp 侧默认 BV=4/W=1，
     在 384 program 处；BV 加大换每 program 寄存器压力，微测定取舍），
     经 `VLLM_GFX906_GDN_M1_TRITON_BV / _WARPS` 可配，默认先 `(8,1)`。
   - k 点积的归约：直接 `tl.sum`（树形）。**与 C++ 串行 k 循环存在 fp32 ULP 级
     差异**——这不是"数值不变"，按下文 parity 板验收（S29 BM16 先例：零翻转 +
     max|Δ top-1 logprob| 0.0213 nat 即采纳）。若 parity 板出现 token 翻转，再降级
     为 k 串行 unroll 循环版本对照（这是决策点，不是默认路径）。
2. **改 `fla/ops/fused_recurrent.py`**（:497 分支处，一处）：在调用 `_C` op 之前加
   env 门（`VLLM_GFX906_GDN_M1_TRITON`，默认关）：
   - 开且 `mixed_qkv.shape[0]==1` 且布局检查通过（K==V==128 不**硬编码**进门条件，
     门只看 `state.stride(-1)==1 or transposed` 等通用可支持条件；不满足条件**静默
     落回 `_C` 原路径**，绝不 raise）→ 调新 kernel；
   - 关/异常 → 原行为一字不改。默认关闭 ⇒ merge 后无行为差异，回滚 = 不设 env。
3. **臂脚本 `.s30b/launch_m1_triton.sh`**：复制
   `.s29/graph_profile_launch_nospec.sh`，只加两行 export：
   `VLLM_QWEN35_COMBINED_PACKED_DECODE=0`、`VLLM_GFX906_GDN_M1_TRITON=1`。
   其余（graph、pin、BM16、fingerprint 行）逐字保留。

### 1.2 施工顺序与命令

1. **微测（新工具 `tools/dcu/s30b_gdn_m1_bench.py`，仿 `perf1_gdn_bench.py` 结构）**：
   - case 构造对齐生产 shape：T=1、H=4、HV=12、K=V=128、fp32 state、slots≥真实池、
     含 slot=-1 用例与 slot 有效用例；state 用**原生 k 连续布局**构造。
   - 三个实现同输入对拍：`_C` 串行 op（`ops.fused_recurrent_gated_delta_rule_gfx906_packed_decode`
     直接调，路径=生产）、新 triton（各 BV/W 档）、eager torch 参照。
   - 输出：per-档 kernel µs（torch profiler，match 名单加 `gdn_m1`）+ event µs。
   - **判据 A（计划）**：最优档 per-op **≤ 40 µs**（现 216 µs；dflash 侧同功能
     25.7 µs 为参照地平线）。
2. **数值对照（parity）**：
   - CPU 级：`TRITON_INTERPRET=1` 下对照 torch 逐 token 参照（仿
     `perf1_gdn_interp.py`），覆盖 slot=-1。
   - GPU 级：仿 `perf1_gdn_parity.py` 出表——`max|Δout|`（fp16 域）、
     `max|Δstate|`（fp32 域）、fp16 输出不同元素数。**判据 B**：跨 20 个随机
     case，`max|Δstate| ≤ 1e-5`、`max|Δout| ≤ 1e-5`（树形归约的合理域；若超，
     查的是实现 bug 而非归约序）。
   - **判据 C（硬门）**：服务级 96-token 贪心流零 token 翻转（§0 共用命令）。
3. **服务 smoke + 逐层确认**：
   ```bash
   pkill -f "entrypoints[.]cli"; sleep 12
   MAX_SEQS=4 bash .s30b/launch_m1_triton.sh > .s30b/serve-m1triton.log 2>&1 &
   # ready 后：
   tr '\0' '\n' < /proc/$(pgrep -f "entrypoints[.]cli" | head -1)/environ \
     | grep -E "GDN_M1_TRITON|COMBINED_PACKED"        # 两行都在
   # smoke：一条 7469 prompt 的 8-token 请求 200 且有输出
   ```
4. **per-op 落点复核**：重启一次带 profiler 的短采集，`.s29/kdiff.py` 同口径
   聚合 decode 窗口——确认 `packed_decode` 类 kernel 时间从 216 µs 级落到判据 A，
   且 **conv kernel 时间与切路前持平**（切到非 combined 后 conv 走独立
   `causal_conv1d_update`，若 conv 变慢超过 +2 µs/层要在结果文档里单列并计入
   e2e 结论）。
5. **e2e**：§0 共用探针 ×3 reps ×N≥2 重启。**判据 D（计划）**：step p50
   **−5 ms 以上**（即 ≤ 34.2 ms），TTFT 保持 0.74 s 口径 ±3%。
6. **文档**：`docs/result/z100-s30b-gdn-m1-parallel.md`（计划指定落点）+ 微测
   JSON 落 `.s30b/`。内容必须含：BV/W 扫描表、parity 表、kdiff 前后块表、
   e2e 双口径、fingerprint、回滚方式（两行 env）。

**回滚**：`.s30b/launch_m1_triton.sh` 不用（回到 nospec 原臂）即全部回滚；
代码默认关，合入主线亦无行为差异。

---

## WP2：fp8 GEMV M=1 → 400 GB/s（−2 ms，纯 env 调参，零代码改动）

### 2.0 现状与可调域（已核实）

- 生产路线：`VLLM_DCU_FP8_W8A16=1` → `run_block_w8a16`
  （`csrc/rocm/fp8_emu_gemm.cu:2509`）→ `stream_block_select`（:936）→
  `dispatch_gemv_stream(2)`。**M=1 今天已在 stream 变体 1**（`VLLM_DCU_FP8_GEMV_STREAM`
  默认 1，:987-995），rows 自动规则：rows=8（512 线程），块数 <120 时降 rows=4。
- 全部旋钮 per-process 读一次，**A/B 只需重启微测/服务进程，无需重建 .so**：
  - `VLLM_DCU_FP8_GEMV_STREAM` = 0（旧 GEMV）/ 1（流式变体 1）/ 2（变体 2，要求
    M≤4 且 block_n%64==0，rows 固定 64、nt=512）；
  - `VLLM_DCU_FP8_GEMV_STREAM_ROWS` ∈ {4,8,16,32,64}（须整除 block_n，否则被忽略
    回自动）；
  - `VLLM_DCU_FP8_STREAM_MAX_M`（M≥V 时跳过流式路径）——**保持 unset**：O4 先例
    表明改 M 门会改变 5120:1536 累加序并破坏位级 parity（launcher 注释），本 WP
    不碰 M 门。
- stream 路径参与条件（自查用）：K≥64 且 K%16==0、w_stride%16==0、block_n%8==0
  （kStreamChunk=16，:943-946）。

### 2.1 施工顺序与命令

1. **形状清单提取**（一次性）：从 `$DATA_ROOT/s29-prof/s29ns_rank0.*.pt.trace.json.gz`
   用 `.s29/kdiff.py` 的 decode 窗口逻辑导出 M=1 fp8 gemv 的
   distinct (N, K) × calls/round × µs/call 表（计划块表：253.5 次 × 48.2 µs 均值）。
   存 `.s30b/s30b_gemv_shapes.json`。**只调 top 覆盖 ≥90% 时间的形状**。
2. **微测扫描**：`tools/dcu/p0_gemv_bench.py`（`build_case`/`run_op`）对每个入选
   形状 × 组合 {mode 1 × ROWS 4/8/16/32/64} ∪ {mode 2}，W8A16 route，记录
   µs 与 GB/s。每组合独立进程（env 进程级生效）。
   - **判据 A（计划）**：热形状 per-op **−15% 以上**。
   - GB/s 算法入文档（bytes = K + N·K + scales，w8a16 权重 fp8 1B/elt）。
3. **数值对照（不可省）**：`tools/dcu/p0_gemv_probe.py`（STREAM env 记录进输出）
   对每入选形状对照 fp32 反量化参照——**rows/变体会改块内累加序**（O4 先例），
   参照容差沿用 P0 结论；随后**服务级 96-token 贪心流零翻转**（硬门）。
4. **服务 A/B**：选定组合后，nospec 原臂 + 仅加该 env 重启（改
   `.s30b/launch_gemv_<cfg>.sh`，复制 nospec 臂加一行 export），e2e ×3 reps ×N≥2 重启。
   **判据 B（计划）**：step p50 **−1.5 ms 以上**，TTFT 口径不变。
   - 若 per-op 达标但 e2e 不达标 → 如实记录"被框架开销吸收"，组合保留结论但不
     采纳（对照计划 WP 序言的墙位判断）。
5. **文档**：`docs/result/z100-s30b-fp8-gemv-m1.md`：形状 × 组合全表、采纳组合、
   parity、e2e 双口径、回滚（删一行 export）。

**回滚**：去掉 export 行。零代码 diff。

---

## WP3：host gap ~3 → ~1.5 ms（无数值影响的削减）

1. **测量**：`ITER_DETAILS=1 bash .s30b/launch_<base>.sh` 起服务（即
   `--enable-logging-iteration-details`，arg_utils.py:1705；逐 iteration 明细由
   `vllm/v1/utils.py::compute_iteration_details` + scheduler
   `capture_iteration_details` 产生），跑共用探针 1 rep，把每 step 的 host 段
   （调度/采样/prepare/同步点）与 GPU busy 39.2−36.2≈3 ms 的分布对齐。
   字段语义以 `vllm/v1/utils.py` 与 `vllm/v1/core/sched/scheduler.py:1631` 附近
   实际输出为准，先出"host 3 ms 花在哪"的表，**再挑削减点**。
2. **允许的削减**：仅无数值影响的（重复张量构造、可缓存的 shape/stride 查询、
   多余的 device 同步、纯 Python 热路径分支）。**禁止**改变任何 kernel 参数、
   顺序、dtype；禁止顺手改调度策略。
3. **门禁**：每个削减点单独 e2e ×3 reps；**判据（计划）**：合计 step p50
   **−1 ms 以上**且贪心流零翻转。单项 <0.3 ms 的不采纳（避免碎片风险）。
4. **文档**：`docs/result/z100-s30b-host-gap.md`：host 段分布表 + 各削减点
   A/B。若测量结论是"host 3 ms 无法无损压到 1.5 ms"，如实出表并停——计划
   允许该 WP 落空，不得为凑数引入有数值风险的改动。

---

## WP4：AR 7.44 ms —— 记录，不动（结构性）

- 每层 2 次 × TP4 是架构决定；64.7 µs/op 已低于 90 µs 独立地板
  （`tools/dcu/perf1_comm_report.md`）。本 WP 产出 = 在
  `docs/result/z100-s30b-final.md` 里引用既有结论一段话 + 数字，不做任何实验。

---

## 收尾与最终取舍（全部 WP 结束后）

1. **同口径 e2e 终局对比**：`s30b-full` 臂（采纳的全部 env 叠加）×N≥3 重启
   vs dflash 采纳臂（`tools/dcu/serve_host_dflash2.sh`，:8000 现役配置）。
   双口径报告：step/tok 与 TTFT；**注意 dflash 是 2.9 tok/step，比的是
   ms/token**（s20 输出里用 `srv_step_ms / srv_tok_per_step`）。
2. **战略判定**（写进 final 文档）：WP1 判据 D 达成 ⇒ 纯 decode 27–29 ms/tok
   反超 dflash 30.9，且 TTFT 3×、无 accept 彩票、无提案 parity 风险面、显存省出
   draft 权重与 KV；交由用户做路线取舍（两条线并行推进的既定格局不变）。
3. **恢复现场**：dflash 生产臂留在 :8000（现网状态）；`.s30b/` 全部 JSON、
   各 result 文档、本施工图的 WP 状态表（含"未采纳/部分达成"如实标注）。
4. **禁止事项重申**（计划原文）：不动 `csrc/`；不做"顺手优化"；每项改动完整走
   微测 → 数值对照 → smoke → e2e → 文档；对比结论标注 TTFT 与 t/s 双口径。

---

## WP 状态表（实施时逐项更新）

| WP | 判据 | 状态 | 证据 |
|---|---|---|---|
| WP1 GDN M=1 triton | per-op ≤40 µs；step p50 −5 ms；流零翻转 | 未开始 | |
| WP2 fp8 GEMV 调参 | per-op −15%；step p50 −1.5 ms；流零翻转 | 未开始 | |
| WP3 host gap | step p50 −1 ms；流零翻转 | 未开始 | |
| WP4 AR | 记录即可 | 未开始 | |
| 终局对比 | 双口径 vs dflash 采纳臂 | 未开始 | |
