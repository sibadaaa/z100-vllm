# S11l 用 `VLLM_DCU_USE_INT8_SDOT4=1` 重跑验收（判定 P2 收益能否上线）— 执行结果

> 归档时间：2026-10-09（**边做边写**：本文件在任务开始时就创建，随后逐节追加）
> 执行者：子 agent（s11l-sdot4-reacceptance）
> 对应设计文档：`docs/海光Z100-性能优化方案.md` §1.1 / §1.2 / §4（P2）
> 前一档：`docs/result/z100-11k-sdot4-compile-integration-result.md`（编译集成修好、`=1` 端到端 3.9×；唯一未达成 K4 逐 token 一致）
> 相关档：`docs/result/z100-11e-w8a8-sdot4-result.md`（内核/数值/性能）、`docs/result/z100-09-acceptance-and-bench-result.md` + `…-09b-…`（验收清单与生产基线口径）、`docs/result/z100-11b-perf-profiling-result.md`（P0 剖析）
> 状态：✅ **完成** —— 7 文件测试 `=1`/`=0` 各 129 passed；翻转归因定论（H_graph 成立、H_kernel 否定）；
> `=1` 同会话 decode 84.00 ms/token（`=0` 321.09 ⇒ 3.82×；生产基线 122.47 ⇒ 快 1.458×）；
> 27B/W4A16/8K/`s8b_gate.sh` 全部通过；**默认值建议保持 OFF、生产显式 `=1`**（§6）；
> 遗留：`=1` 的 AOT 缓存每次加载失败（每次重编 ⇒ 同开关两次启动文本不同）未定位到根因（§5.2）。

## 0. 承接与本步要点

S11k 已把 P2 打通（decode 84.07/82.74 ms/token vs 同命令 `=0` 329.05/320.30，3.91×/3.87×），
但 S9 的验收清单全部是在 **Triton 路径**（`=0`）下做的，换内核之后必须重跑；同时 S11k 观察到
**逐 token 文本翻转**（zh 第 9、en 第 20），需要诚实定性。本步做四件事：

1. `=1` 下重跑 S5/S6 那 7 个 gfx906 测试文件，并与 `=0` 计数对比；
2. **翻转隔离实验**：把 opaque `custom_op` 的内核换回 `_triton_fallback`（`opaque(triton)`）与
   `=0` 的 `bare triton` 对比，判定扰动来自 **custom_op 引起的图/编译变化**还是 **sdot4 内核本身**；
3. **翻转边界统计**：zh/en 各 ≥4 条 prompt、greedy，逐次记录 top1/top2 间距、翻转位置、
   `max logprob delta` 与翻转率，检验"所有翻转都在 <0.5 nat 近似并列点"；
4. **同口径对拍**（S9 协议：decode / TTFT / conc8）+ 端到端回归（27B W8A8 TP=2 / W4A16 / 8K + `s8b_gate.sh`），
   给出"默认值 OFF 还是 ON"的明确建议与上线配置。

### 0.1 完成情况一览（边做边写：每完成一段就追加）

| 小节 | 内容 | 状态 |
|---|---|---|
| §3.1 | 7 文件测试（`=1` vs `=0` 计数） | ✅ 129/129 全绿（逐文件一致） |
| §3.2 | 翻转隔离实验（opaque(triton) vs bare triton vs opaque(sdot4)） | ✅ H_graph 成立、H_kernel 否定 + 附加发现（`=1` 每次重编） |
| §3.3 | 翻转边界统计（top1/top2 间距、位置、max delta、翻转率） | ✅ arm1/arm2/arm3/arm4 全部完成 |
| §3.4 | 同口径对拍（decode / TTFT / conc8 vs 生产基线 122.47） | ✅ 同会话 A/B 完成 |
| §3.5 | 端到端回归（27B / W4A16 / 8K / `s8b_gate.sh`） | ✅ 全部完成（8K 的独立文本探针未重跑，原因见 §3.5.4） |
| §4 | 门禁判定 | ✅ L1–L8 全过（L2/L3 是“有明确结论”，不是“文本一致”） |
| §6 | 默认值建议 + 上线配置 | ✅ 已给出（§6.1/§6.2） |

## 1. 目标与验收标准

本步门禁（任务书原文，逐条判定见 §4）：

| # | 门禁 | 判据 |
|---|---|---|
| L1 | 测试套件（`=1`） | 7 个 gfx906 测试文件在 `VLLM_DCU_USE_INT8_SDOT4=1` 下全绿，且与 `=0` 计数一致 |
| L2 | 翻转归因 | 隔离实验给出明确结论：翻转来自 custom_op 的图/编译变化 **或** sdot4 内核本身 |
| L3 | 翻转边界 | zh/en 各 ≥4 prompt：每次翻转的 top1/top2 间距、位置、`max logprob delta`、翻转率；是否全部 <0.5 nat |
| L4 | 同口径对拍 | S9 协议下 `=1` 的 decode ms/token / TTFT / conc8，并给出 vs 生产基线 122.47 的数字与口径说明 |
| L5 | 端到端回归 | 27B W8A8 INT8 TP=2 完整请求（文本合理）、W4A16（tinyllama）一次、8K（`TRITON_ATTN` 日志）、`s8b_gate.sh` ALL GREEN |
| L6 | 结论 | 明确建议默认值 OFF/ON + 依据 + 生产上线配置 |
| L7 | 边做边写 | 归档先建骨架后追加 |
| L8 | 临时证据整理 | `.tmp-11k/`、`.tmp-11g/` 原始 jsonl 归置到 `/mnt/game` 并在归档记路径 |

## 2. 实际改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py` | `:37`（`import os`）、`:74-82`（`_SDOT4_INNER = os.environ.get("VLLM_DCU_SDOT4_INNER", "sdot4")`）、`:100-112`（opaque body 前置分支） | **仅供本步定性**：`VLLM_DCU_SDOT4_INNER=triton` 时，opaque `custom_op` 的 body 内改调 `triton_scaled_mm`（即 "opaque(triton)"），用来把"opaque op 改变了编译图"与"sdot4 内核数学"两个变量分开。默认值 `sdot4` ⇒ **默认行为逐字节不变**（不写 `envs.py`，不产生新开关） |
| `tools/dcu/s11l_tests.sh` | 新增 | 7 个 gfx906 测试文件 × `VLLM_DCU_USE_INT8_SDOT4∈{1,0}` 的容器内跑法（`--noconftest -p no:cacheprovider`，与 S9 §3.6 同口径） |
| `tools/dcu/s11l_arm_env.sh` | 新增 | 27B W8A8 TP=2 单臂启动器（参数化 `SW`/`INNER`/`PORT`/`MAXLEN`；其余 flag 与 S11k `.tmp-11k/arm_env.sh` 完全一致） |
| `tools/dcu/s11l_bench.sh` | 新增 | S9 协议对拍（`decode`/`ttft`/`ttft8k`/`conc8`，flag 逐字复制 `/mnt/game/port/s9/bench.sh`） |
| `tools/dcu/s11l_flip_stats.py` | 新增 | 8 条 prompt（zh×4 / en×4）的逐 token logprob + top-5 记录器（greedy，`max_tokens=64`） |
| `tools/dcu/s11l_flip_compare.py` | 新增 | 两臂逐 token 对比：翻转位置、两侧 top1/top2 间距、交叉 logprob、`max logprob delta`、翻转率 |
| `tools/dcu/s11l_flip_summary.py` | 新增 | 把上面的 `--json-out` 结果汇总成归档表（含"翻转点交叉裕度"和"是否在 top-5 之外"） |
| `tools/dcu/s11l_inner_probe.py` | 新增 | `VLLM_DCU_SDOT4_INNER` 的**就地门禁**：`sdot4` 时与裸内核逐位相同、`triton` 时与 `triton_scaled_mm` 逐位相同 |
| `tools/dcu/s11l_quant_sensitivity.py` | 新增 | 量化敏感度实验：1 ULP 输入扰动 → int8 code 变化数 + 输出相对变化（对比不量化的 fp16 matmul） |
| `tools/dcu/s11l_prefill_m_range.py` | 新增 | 补 S11e/S11k 未覆盖的 prefill M ∈ {17…32, 40…512} × 7 真实 shape 的逐位对比（168/168） |
| `tools/dcu/s11l_run_arm.sh` | 新增 | 宿主侧编排：起引擎 → 等 ready（`READY_TIMEOUT` 默认 2700 s）→ 按 `tasks` 跑客户端 → 杀引擎 |
| `tools/dcu/s11l_gate_and_w4a16.sh` | 新增 | 卡 3 上的 `s8b_gate.sh` + W4A16 tinyllama e2e（全流程留原始日志） |
| `tools/dcu/s11l_8k_probe.py` | 新增 | ~7.5k token 真实请求（用模型自带 tokenizer 造 prompt），给 8K 臂用 |
| `tools/dcu/s8b_gate.sh` | `:80-96` | **过期断言修正**：`_rocm_C ops == [paged_attention]` → 显式允许清单（+`w8a8_sdot4_gemm/gemv`）。理由与前后对比见 §3.5.1 |

**未改动**：内核 C++/HIP 数学、权重布局、内核表优先级、`vllm/envs.py`（`VLLM_DCU_USE_INT8_SDOT4` 默认值仍是 S11k 落地的 **OFF**）、`docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md`。

## 3. 执行过程与原始证据

### 3.0 运行环境

```bash
# 容器：z100-s9（root）；/src = /home/lyh/z100-vllm（只读挂载，被验证的 Python 树）
# 四个扩展 .so 是工作区内指向 canonical 的符号链接：
#   _C 78,102,728 B / _C_stable_libtorch 40,691,216 B / _moe_C_stable_libtorch 29,748,928 B
#   _rocm_C 36,182,960 B  md5 ebd6fb8dac572514cf94ca7ec7d828a7（S11e/S11k 同一份）
# 本步被验证的 Python 文件 md5：
#   vllm/envs.py = 22918d5660f8672b81786c948180f7fa（VLLM_DCU_USE_INT8_SDOT4 默认 False，:169 / :1414-1415）
#   vllm/.../gfx906_sdot4.py = 59e669644855ad54efe2ee98e61807e1（S11k 的 45854752… + 本步诊断开关，§2）
. /opt/dtk/env.sh; . /src/tools/dcu/env.sh
export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
export VLLM_CACHE_ROOT=/mnt/game/vllm-cache-27b      # 复用 S11b/S11k 既有编译缓存
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 27B：HIP_VISIBLE_DEVICES=0,1（TP=2）、--dtype float16、无 --enforce-eager、--max-model-len 4096
# 测试套件：HIP_VISIBLE_DEVICES=0
```

`torch 2.11.0+das`、`triton 3.6.0+git7c56a5e4`（与 S11e/S11k 相同）。

诊断开关的就地门禁（`tools/dcu/s11l_inner_probe.py`，卡 3，`VLLM_DCU_USE_INT8_SDOT4=1`）：

```
[inner] env=sdot4 module=sdot4
[inner] M=   1 equal_raw=True equal_triton=True sum=-285.062454 ...
[inner] M=   4 equal_raw=True equal_triton=True sum=762.000534 ...
[inner] M=  64 equal_raw=True equal_triton=True sum=2133.357694 ...
[inner] M=2048 equal_raw=True equal_triton=True sum=-15815.798555 ...
[inner] env=triton module=triton   （四个 M 全部与上面逐位相同）
```

⇒ 两个结论：① `VLLM_DCU_SDOT4_INNER` 生效且默认值与 S11k 完全一致（`sdot4` 时仍与裸内核逐位相同，
S11k K7 不变量保持）；② 在这四个 M 上 **sdot4 内核与 `triton_scaled_mm` 同样逐位相同**（`equal_triton=True`），
即"M=1 decode 数学逐位等价"再次被独立复现——对 §3.2 的归因很关键。

### 3.1 测试套件（`=1` vs `=0`）

命令（容器内，7 个文件 × 两个开关值；`tools/dcu/s11l_tests.sh`，跑法与 S9 §3.6 同口径
`python -m pytest -q --noconftest -p no:cacheprovider <file>`）：

```bash
docker exec z100-s9 bash /src/tools/dcu/s11l_tests.sh 1 /mnt/game/s11l/tests-sw1
docker exec z100-s9 bash /src/tools/dcu/s11l_tests.sh 0 /mnt/game/s11l/tests-sw0
```

原始结果（宿主日志 `.tmp-11l/tests-both.host.log`；容器内每文件日志 `/mnt/game/s11l/tests-sw{1,0}/`）：

| 测试文件 | `=1` | `=0` | 计数一致 |
|---|---|---|---|
| `tests/kernels/attention/test_triton_unified_attention_gfx906.py` | 39 passed (17.16 s) | 39 passed (16.24 s) | ✅ |
| `tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py` | 12 passed (17.47 s) | 12 passed (16.49 s) | ✅ |
| `tests/kernels/quantization/test_gguf_fixed_cols_gfx906.py` | 13 passed (32.86 s) | 13 passed (27.45 s) | ✅ |
| `tests/kernels/mamba/test_fused_sigmoid_gating.py` | 5 passed (14.63 s) | 5 passed (14.65 s) | ✅ |
| `tests/kernels/mamba/test_fused_recurrent_packed_decode.py` | 5 passed (15.09 s) | 5 passed (15.07 s) | ✅ |
| `tests/platforms/test_hygon_dtk.py` | 52 passed (2.08 s) | 52 passed (2.10 s) | ✅ |
| `tests/kernels/test_gfx906_packed_silu_fp16.py` | 3 passed (19.27 s) | 3 passed (19.42 s) | ✅ |
| **合计** | **129 passed / 0 failed** | **129 passed / 0 failed** | ✅ 逐文件一致 |

> 口径说明：本步的 7 文件清单按任务书取（用 `test_gfx906_packed_silu_fp16.py` 替换 S9 §3.6 里的
> `tests/model_executor/test_qwen3_5_gdn.py`），因此**合计 129** 而不是 S9 的 146（S9 = 本表 −3 +20）。
> 两个开关值下逐文件 passed 数、失败数完全相同 ⇒ `=1` 不改变这 7 个文件的判定。
> 这 7 个文件本身不进入 int8 scaled-MM 选择路径（无 W8A8 linear），它们证明的是"打开 `=1` 后
> import/注册面与 gfx906 内核路径没有回归"，不是"`=1` 的量化路径正确性"（后者由 §3.5 端到端覆盖）。

### 3.2 翻转隔离实验

**问题**：S11k 观察到 `=1` 与 `=0` 的贪心文本在第 9（zh）/20（en）个 token 分叉，而线性层在 126/126 组
与 `triton_scaled_mm` **逐位相同**。那这个扰动来自哪里？两种互斥的解释：

* **H_graph**：扰动来自"opaque `custom_op` 改变了被 trace 的图 / Inductor 的编译配置"（与 sdot4 内核数学无关）；
* **H_kernel**：扰动来自 sdot4 内核本身的数学（在某些真实 shape 上与 Triton 不同），那"S11k 的 126/126 逐位相同"
  的口径就要重新审视。

**设计**（三臂，唯一变量是 linear 的实现方式；其余 flag、缓存、卡、prompt、贪心参数全部相同）：

| 臂 | 选择的内核 | linear 在编译图里的形态 | 数学 |
|---|---|---|---|
| A `arm1`：`VLLM_DCU_USE_INT8_SDOT4=1` | `Gfx906Sdot4Int8ScaledMMLinearKernel` | **opaque `custom_op`**（`torch.ops.vllm.w8a8_sdot4_mm`） | sdot4 GEMV/GEMM |
| B `arm3`：`=1` + `VLLM_DCU_SDOT4_INNER=triton` | 同上（仍是 sdot4 kernel 被选中、仍走 opaque op） | **opaque `custom_op`**（同一张图结构） | `triton_scaled_mm` |
| C `arm2`：`VLLM_DCU_USE_INT8_SDOT4=0` | `TritonInt8ScaledMMLinearKernel` | **裸 triton 调用被 trace 进图**（Inductor 可见） | `triton_scaled_mm` |

判定规则：

* `text(B) == text(C) ≠ text(A)`（或 `text(B) == text(A) ≠ text(C)`）⇒ 扰动来自 **H_graph**（custom_op 引起的
  图/编译变化），**不是** sdot4 内核数学 ⇒ S11k 的"126/126 逐位相同"口径**成立**；
* `text(A) == text(B) ≠ text(C)` 且 A/B 都与 C 不同 ⇒ 同上（H_graph），且"内核数学换掉也没用"；
* `text(A) ≠ text(B)` ⇒ **H_kernel** 成立（opaque 边界不变、只有内核数学变了），需要重新审视 126/126 的口径。

脚本：`tools/dcu/s11l_flip_stats.py`（8 prompt × 逐 token logprob/top-5）两两对比用
`tools/dcu/s11l_flip_compare.py`；原始 jsonl 见 §3.6（`/mnt/game/s11l/flip-arm*.jsonl`）。

> 结论见 §3.2.4–§3.2.6（H_graph 成立、H_kernel 否定；另发现 `=1` 每次启动重编）。

### 3.2.1 已到手的关键对照（arm3 结果见下）

**(i) `=0` 跨引擎逐位可复现**：arm2（本步 `=0` 引擎，热缓存）与 S11k 的 `=0` 引擎（第一手
`logprob-ctl0.jsonl`）在 `zh_hz` 上前 24 个 token **完全一致**，而且每一步的
`top1−top2` 间距**逐条相同到小数点后 3 位**（7.500/7.500、2.625/2.625、…、0.219/0.219、0.078/0.078）：

```
  i    s11k_ctl0    gap          arm2    gap
  0        97237  7.500         97237  7.500
  8         8283  0.281          8283  0.281
  9       126755  0.219        126755  0.219
 11        95860  0.078         95860  0.078
```

**(ii) `=1` 跨引擎不可复现**：S11k 的 `=1` 引擎（冷编译）与 arm1（同开关、同 flag、同卡、热缓存）
前 11 个 token 相同，第 11 个开始分叉；而且**第一个 decode step（i=0）的 top1−top2 间距就已经不同**
（4.922 vs 5.844）。把 `zh_hz` 第 8–10 步的完整 top-5 摊开看（`logprob` 保留 4 位）：

| i | 引擎 | 选中 | lp(选中) | top-2 | lp(top-2) | 两个候选的间距 |
|---|---|---|---|---|---|---|
| 8 | arm2(=0) / S11k=0 | `——` | -0.9179 | `，` | -1.199 | 0.281 |
| 8 | arm1(=1) | `——` | -0.5765 | `，` | -1.326 | 0.750 |
| 8 | S11k=1 | `——` | -0.6134 | `，` | -1.473 | 0.859 |
| **9** | **arm2(=0) / S11k=0** | **`需要用`** | **-1.4031** | **`这是一个`** | **-1.622** | **0.219** |
| **9** | **arm1(=1)** | **`这是一个`** | **-0.9600** | **`需要用`** | **-1.741** | **0.781** |
| **9** | **S11k=1** | **`这是一个`** | **-1.2082** | **`需要用`** | **-1.521** | **0.313** |
| 10 | arm2(=0) / S11k=0 | `一句话` | -0.0470 | `"` | -3.985 | 3.938 |
| 10 | arm1(=1) | `非常` | -1.8123 | `指令` | -2.140 | 0.328 |
| 10 | S11k=1 | `非常` | -1.8264 | `指令` | -2.217 | 0.391 |

⇒ 三个要点：① **两个 `=0` 引擎的 top-5 逐条相同到 1e-3**（连 i=9 的 0.219 间距都一样）；
② 两个 `=1` 引擎在 i=9 都是 `这是一个` 领先，但领先幅度 0.781 vs 0.313 差 0.47 nat；
③ 该位置本身是**模型不确定区**（top-1 的 logprob 只有 -0.96…-1.62，即概率 ~0.2–0.4），
候选 logits 的 0.3–0.8 nat 级差异就足以改选。

#### 3.2.2 机制佐证：动态 per-token int8 量化把微小扰动放大成离散跳变

`tools/dcu/s11l_quant_sensitivity.py`（卡 3，7 个真实 (N,K)，M=1）：把输入 `x` 整体乘 `1+ε`
后重新量化（`scaled_int8_quant` 是 **quantizer**，舍入边界会让 int8 code 整片跳 ±1 LSB），
比较 sdot4 GEMV / `triton_scaled_mm` / 不做量化的 fp16 matmul 三种输出的相对变化：

```
shape                           eps  int8_codes_changed   sdot4_rel  triton_rel    fp16_rel
self_attn.q_proj            4.9e-04                  35   2.877e-03   2.877e-03   5.750e-04
self_attn.k_proj/v_proj     4.9e-04                  25   2.252e-03   2.252e-03   7.530e-04
self_attn.o_proj            4.9e-04                  30   2.573e-03   2.573e-03   1.032e-03
linear_attn.in_proj_qkv     4.9e-04                  35   2.330e-03   2.330e-03   1.201e-03
linear_attn.in_proj_z       4.9e-04                  23   2.429e-03   2.429e-03   1.388e-03
mlp.gate_proj/up_proj       4.9e-04                  32   2.492e-03   2.492e-03   6.028e-04
mlp.down_proj               4.9e-04                 120   3.269e-03   3.269e-03   7.686e-04
（ε = 2^-10 / 2^-9 的三组同样趋势：sdot4_rel 1.8e-3…3.8e-3、triton_rel 与之**逐位相同**、
fp16 直算只有 ~0.6e-3…2.4e-3）
```

三条结论：
1. **一个 fp16 ULP（4.9e-04 相对）的输入扰动，会让 int8 code 变 20–120 个（5120 个里），
   使该层输出变化 ~2.2–3.3e-3 相对 —— 比"同样扰动、不量化"的 fp16 路径大 2–5×**，
   而且是**离散跳变**（±1/127 ≈ 7.9e-3）。96 层残差流累积这个量级，足以在 <1 nat 的近似并列点上翻转贪心。
2. 每一行 `sdot4_rel == triton_rel` ⇒ 即使在扰动后，两个 int8 内核的输出仍然**逐位相同**
   （K7 的"126/126"在扰动工况下也成立）。
3. 因此"换内核 ⇒ 文本变"这条因果链不成立；成立的是"**换编译图/编译产物 ⇒ 激活值差 ~1e-3…1e-2
   ⇒ int8 量化离散放大 ⇒ 近似并列点翻转**"。

#### 3.2.3 真正的变量是"**编译产物**"：arm1 的 AOT 缓存命中失败 + `benchmark_combo_kernel=True`

两臂启动日志里的决定性区别（原始行）：

```
# arm1（=1）—— 两个 rank 都因为"源码变了"重新编译：
(Worker_TP0) WARNING [decorators.py:322] Compiling model again due to a load failure from
  /mnt/game/vllm-cache-27b/torch_compile_cache/torch_aot_compile/2b4e7331…/rank_0_0/model,
  reason: Source code has changed since the last compilation. Recompiling the model.
(Worker_TP1) WARNING [decorators.py:322] … rank_1_0/model, reason: Source code has changed …
(Worker_TP0) INFO [decorators.py:717] saved AOT compiled function to …/2b4e7331…/rank_0_0/model
(Worker_TP0) INFO [monitor.py:53] torch.compile took 54.15 s in total

# arm2（=0）—— 没有 recompile 警告，直接吃缓存：
(Worker_TP0) INFO [backends.py:290] Directly load the compiled graph(s) for compile range
  (1, 2048) from the cache, took 16.445 s
(Worker_TP0) INFO [monitor.py:53] torch.compile took 22.25 s in total
```

原因很清楚：本步为了做隔离实验**改了 `vllm/.../gfx906_sdot4.py` 的源码**（§2 的诊断分支），
AOT 缓存里那份 `=1` 产物（S11k 编译的）因此**失效并被重编、覆盖**；而 `=0` 的产物（不含该文件）
没被触碰。而 vLLM 的 inductor 配置里 **`combo_kernels: True, benchmark_combo_kernel: True`**
（同一行日志可见）⇒ **每次重编都会用"实测耗时"重新挑 combo kernel**，
所以"同一张图、两次 fresh compile"本来就**不保证得到同一份代码**。
（`=1` 的图里 linear 变成了 opaque extern call，剩下的可融合算子更少、combo 选择更敏感。）

#### 3.2.4 隔离实验结果（arm3 = opaque + `triton` 数学 vs arm2 = bare triton）

arm3 命令：`VLLM_DCU_USE_INT8_SDOT4=1 VLLM_DCU_SDOT4_INNER=triton`（**仍然选中**
`Gfx906Sdot4Int8ScaledMMLinearKernel`、仍然走 `torch.ops.vllm.w8a8_sdot4_mm` 这张 opaque 图，
只是 body 里换成 `triton_scaled_mm`）。原始 jsonl `/mnt/game/s11l/flip-arm3.jsonl`，
比对 `.tmp-11l/cmp-arm3-arm2.json`。

| prompt | 首个分叉位（arm2 bare-triton vs arm3 opaque-triton） | `=0` 选的 token（gap） | opaque-triton 选的 token（gap） | arm1（opaque-sdot4）vs arm2 的首个分叉位 |
|---|---|---|---|---|
| zh_hz | **9** | `需要用`（0.219） | `这是一个`（0.531） | **9** |
| zh_ml | **7** | `。`（0.750） | `"`（0.531） | **7** |
| zh_sort | 23 | `选择一个`（0.062） | `时间`（0.047） | 16 |
| zh_sky | 11 | `思考`（0.016） | `分析`（0.391） | 8 |
| en_fr | **13** | ` this`（0.359） | ` the`（0.312） | **13** |
| en_sea | **10** | ` Let`（0.203） | ` I`（0.188） | **10** |
| en_index | 23 | ` should`（0.125） | ` need`（2.547） | 4 |
| en_math | **2** | ` asks`（0.516） | ` is`（1.859） | **2** |
| **合计** | **8/8 prompt 翻转**；`max_matched_delta = 1.108 nat`（en_fr@20）；翻转率 380/490 = 77.6% | | | 8/8，428/512 = 83.6% |

**判定（按任务书给的规则）**：把 opaque op 里的内核**换成与基线完全相同的 `triton_scaled_mm`**，
**翻转照样发生**，而且 `zh_hz` 第 9 个、`zh_ml` 第 7 个、`en_fr` 第 13 个、`en_sea` 第 10 个、
`en_math` 第 2 个这些**同一个近似并列点**都翻转了、翻转方向也一致 ⇒
**扰动来自"图因 opaque `custom_op` 而变化的编译配置"，与 sdot4 内核数学无关（H_graph 成立、H_kernel 排除）**。
这一点还被另外两组数据独立钉住：

* **线性层逐位相同**：S11e 63/63 + S11k 63/63 + 本步 `s11l_prefill_m_range.py` **168/168**
  （M ∈ {17…32, 40, 48, 64, 96, 128, 129, 256, 512} × 7 个真实 (N,K)，`torch.equal` 全 True）
  + `s11l_quant_sensitivity.py` 里**扰动后** `sdot4_rel == triton_rel`（逐位同）
  + `s11l_inner_probe.py` 4 组 —— 覆盖了 decode（M=1）到 prefill（M=2048）的全部真实档位。
* **同引擎内逐位可复现**：arm3 上 `flip` 与紧接着的 `flipb` 两遍（第二遍的 prompt 已在前缀缓存里，
  走的是 M=1 复用 KV 的路径）在 8 prompt × 490 个位置上的 token **与 logprob 完全相同**
  （`flips=0, max_matched_delta=0.0000`）⇒ 排除"采样随机性 / 前缀缓存冷热 / 批组成"这些解释。

#### 3.2.5 反面补充：`=1` 每一次启动都会重新编译（AOT 缓存加载失败）

arm1 与 arm3 的日志里都有（两个 rank 各一条）：

```
WARNING [decorators.py:322] Compiling model again due to a load failure from
  /mnt/game/vllm-cache-27b/torch_compile_cache/torch_aot_compile/2b4e7331…/rank_{0,1}_0/model,
  reason: Source code has changed since the last compilation. Recompiling the model.
…
INFO [decorators.py:717] saved AOT compiled function to …/2b4e7331…/rank_0_0/model
```

* ==1 的 AOT 产物**每次启动都加载失败**（arm1 12:05 重编并覆盖、arm3 13:16 又重编并覆盖；
  `model` 文件 mtime 12:05→13:16），而 ==0 的产物 arm2 是一条 `Directly load the compiled
  graph(s) … from the cache`（16.4 s）就吃上了，从未重编。
* 触发点在 `vllm/compilation/decorators.py:266-283 _verify_source_unchanged()`
  （比对产物内嵌 `source_info.inlined_sources` 的内容哈希 vs **当前磁盘内容**哈希）；
  与 =0 的产物相比，**=1 的产物把 `gfx906_sdot4.py` 也当成了 traced file**
  （产物文件里 `gfx906_sdot4.py` 字样出现 3344 次；=0 的产物 0 次）——因为
  `apply_weights()` 这段 Python 是被 trace 进图的，而它住在该文件里。
  **为什么在源码未变的情况下仍然判"变了"：本步未定位到（见 §5.4）**，只给出可复现的现象与代码位置。
* **后果**：本树 inductor 配置是 `combo_kernels: True, benchmark_combo_kernel: True`（启动日志原文），
  即**每次重编都按实测耗时重新挑 combo kernel** ⇒ 每次启动拿到的编译产物可以不同
  ⇒ **同一个 `=1` 开关、同一份源码，两次启动的 logits 就可以差 0.3–1.4 nat、贪婪文本就不一样**
  （arm1 vs arm3：341/490 个位置不同、`max_matched_delta = 1.401`）。
  这也解释了 S11k 的 `=1` 与 arm1 的 `=1` 为什么不是同一串 token。

#### 3.2.6 隔离实验的最终判定

| 假设 | 判据 | 结果 |
|---|---|---|
| H_kernel：扰动来自 sdot4 内核数学 | `opaque(triton)` vs `bare triton` 应当**不翻转**；且内核应在某些真实 M/shape 上与 Triton 不同 | ⛔ **否定**：opaque(triton) 照样 8/8 翻转（含同样的 zh_hz@9）；内核在 126+168 组上逐位相同 |
| H_graph：扰动来自 opaque op 造成的图/编译变化 | `opaque(triton)` 与 `opaque(sdot4)` 应当都翻转、且翻转点相同 | ✅ **成立**：两臂都在 `zh_hz@9 / zh_ml@7 / en_fr@13 / en_sea@10 / en_math@2` 翻转，方向一致 |
| （附加发现）`=1` 的文本跨启动不可复现 | 同一 `=1` 配置的两次启动应当给出相同 logits | ⛔ **不成立**：AOT 缓存每次加载失败→重编→timing-based autotune 选不同 kernel（§3.2.5） |

⇒ 结论一句话：**"换内核导致文本变"不成立；成立的是"改变编译图/编译产物导致文本变"。
sdot4 内核的数学与 Triton 逐位相同，`VLLM_DCU_USE_INT8_SDOT4=1` 之所以改变文本，
是因为它在图里引入了一个 opaque `custom_op`（连带 AOT 缓存失效、每次重编）。**
对生产的意义：文本逐 token 一致性在本树上**本来就不可保证**（连同一个开关的两次启动都不一致）；
要保证一致性必须锁**编译产物**（共享同一份 AOT 缓存 + 关闭 `benchmark_combo_kernel`）。

### 3.3 翻转边界统计

口径：`tools/dcu/s11l_flip_stats.py`，8 条 prompt（zh×4：`zh_hz`/`zh_ml`/`zh_sort`/`zh_sky`；
en×4：`en_fr`/`en_sea`/`en_index`/`en_math`），`temperature=0`、`top_p=1.0`、
`max_tokens=64`、`logprobs=true`、`top_logprobs=5`、`return_token_ids=true`（非流式，一次请求取回全部 token 的分布）。

每条 prompt 记录：每次翻转的位置 `i`、两侧选中的 token、两侧**该位置的 top1−top2 间距**
（`gap`，即该步的决策裕度）、以及**交叉 logprob**（A 臂给 B 臂所选 token 打了多少分、反之亦然）。

#### 3.3.0 先做两个"重复性"对照（否则跨臂比较会被噪声解释掉）

**(a) 同引擎内重复性（arm1，`=1`）**：同一个引擎上，"flip 电池"（冷前缀缓存、`max_tokens=64`、
`logprobs=true`）与随后的 e2e 客户端（热前缀缓存、`max_tokens=129`、无 logprobs）里
**同一个 prompt** 的输出逐 token 对比：

```
zh_hz (flip,64) vs chat_zh (e2e,129): first_mismatch=None over 64
en_fr (flip,64) vs chat_en (e2e,129): first_mismatch=None over 64
```

⇒ **同一引擎内，跨"前缀缓存冷/热、max_tokens、logprobs 开关"都逐 token 可复现**（不是采样随机性、
不是前缀缓存布局）。这也说明 decode 的图重放是确定性的。

**(b) 跨引擎、同设置（`=1` vs `=1`）**：S11k 的 `=1` 臂（**冷编译** 88.24 s 那次，第一手落盘的
`logprob-sdot4.jsonl`）与本步 arm1（**热缓存** 54.15 s）**不是同一串 token**：

```
索引:   9        10       11       12       13       14
S11k=1: 109455   96762    98865    108767   104118   3709     ← '这是一个''非常''开放''式的'…
arm1=1: 109455   96762    121490   101729   3709     96442    ← '这是一个''非常''开放的''请求''，'…
```

前 11 个 token 相同（含 S11k 记录的分叉点第 9 个），**第 11 个开始分叉**；en 侧同样在
S11k 的 20 号位（` The`）与 arm1 的 20 号位（` This`）不同。

⇒ 结论（对本步最关键的一条）：**"`=1` 的那一串 greedy token"不是开关的稳定属性**。
在"线性层逐位相同"（§3.0/§3.5.3 的 126/126 与本步 4 shape 探针）的前提下，
跨引擎的差异只能来自**编译产物状态**（Inductor 的 timing-based autotune / combo-kernel 选择、
图捕获、cache 冷热）带来的 ~1e-3 级数值扰动 —— 也就是说**任何**改动数值路径的东西都会在
<0.5 nat 的近似并列点上造成翻转，`sdot4` 只是其中一个（甚至不是主要）触发者。
这条对照是后面判读 arm2/arm3 的前提，故先落盘。

> 见 §3.3.1（arm1 vs arm2）、§3.3.2（arm3 vs arm2）、§3.3.3（arm4 vs arm1）。

#### 3.3.1 arm1（`=1`）vs arm2（`=0`）：8 条 prompt 的翻转边界

原始 jsonl：`/mnt/game/s11l/flip-arm1.jsonl`（`=1`）、`/mnt/game/s11l/flip-arm2.jsonl`（`=0`）；
比对 JSON：`.tmp-11l/cmp-arm1-arm2.json`。工具：`s11l_flip_compare.py` + `s11l_flip_summary.py`。

| prompt | 首个分叉位 i | `=0` 选的 token | `=1` 选的 token | `=0` 的 top1−top2 | `=1` 的 top1−top2 | **交叉裕度**（两候选在两个分布里的最大间距） | 该 prompt 总翻位数 /64 |
|---|---|---|---|---|---|---|---|
| zh_hz | 9 | `需要用` | `这是一个` | 0.219 | 0.781 | **0.781** | 52 |
| zh_ml | 7 | `。` | `"` | 0.750 | 0.484 | **0.750** | 55 |
| zh_sort | 16 | `Sort` | ` Sort` | 0.172 | 0.594 | **0.594** | 44 |
| zh_sky | 8 | `"` | `简单` | 0.125 | 0.281 | **0.281** | 55 |
| en_fr | 13 | ` this` | ` the` | 0.359 | 0.953 | **0.953** | 48 |
| en_sea | 10 | ` Let` | ` I` | 0.203 | 0.250 | **0.250** | 54 |
| en_index | 4 | ` me` | ` "` | 1.969 | 0.172 | **1.969** | 59 |
| en_math | 2 | ` asks` | ` is` | 0.516 | 0.937 | **1.297** | 61 |
| **合计/全局** | 2–16（8/8 prompt 都翻转） | — | — | 0.125–1.969 | 0.172–0.953 | **max = 1.969 nat** | **428 / 512（翻转率 83.6%）** |

**怎么读这张表（不许只报 83.6% 就下结论）**：
* 83.6% 的"翻转率"是**误导性指标**：贪心一旦在第 2–16 步分叉，两条续写就整体错位，
  后面每个位置都"不同"（例如 zh_hz 第 9 步之后 `=0` 在写"需要用一句话来介绍杭州。…"、
  `=1` 在写"这是一个非常开放的请求，需要…"，两串各自通顺）。所以有意义的量是
  **首个分叉点的决策裕度**（上表）。
* **"所有翻转都在 <0.5 nat 近似并列点"不成立，但接近**：8 个分叉点里，
  被放弃的那个候选在"没选它"的那一臂里只落后 **0.125–0.516 nat**（7/8 < 0.5 nat，
  最大 en_math 0.516 nat）⇒ **被翻转的那一步在两个分布里都是"贴边"的**；
  但另一臂往往很自信（最高 en_index 1.969 nat）⇒ 两个引擎的分布**本身就差得多**，
  不是"同一分布里 1e-3 噪声挑边"。
* 两个对照（§3.3.0）把这一点钉死：**同引擎内**逐 token 可复现；
  **`=1` 跨引擎**连第 0 步的间距都不同；**`=0` 跨引擎**逐条间距相同。
  ⇒ 翻转来自"引擎级的数值差异"（编译产物/图），其量级足以让 0.1–1 nat 的贴边决策改选，
  而不是来自 sdot4 内核的算术（后者与 Triton 逐位相同）。

> arm3（opaque triton 隔离臂）与 arm4（`=1` @8192）的统计表见 §3.3.2 / §3.3.3。

#### 3.3.2 arm3（`=1` + opaque(`triton`)）vs arm2（`=0` bare triton）：翻转边界

原始 jsonl `/mnt/game/s11l/flip-arm3.jsonl`；比对 `.tmp-11l/cmp-arm3-arm2.json`。

| prompt | 首个分叉位 i | `=0` 选的 token | opaque-triton 选的 token | `=0` 的 top1−top2 | opaque-triton 的 top1−top2 | 交叉裕度 | 总翻位/64 |
|---|---|---|---|---|---|---|---|
| zh_hz | 9 | `需要用` | `这是一个` | 0.219 | 0.531 | **0.531** | 54 |
| zh_ml | 7 | `。` | `"` | 0.750 | 0.531 | **0.750** | 56 |
| zh_sort | 23 | `选择一个` | `时间` | 0.062 | 0.047 | **1.438** | 41 |
| zh_sky | 11 | `思考` | `分析` | 0.016 | 0.391 | **0.703** | 50 |
| en_fr | 13 | ` this` | ` the` | 0.359 | 0.312 | **0.375** | 24 / 42 |
| en_sea | 10 | ` Let` | ` I` | 0.203 | 0.188 | **0.203** | 54 |
| en_index | 23 | ` should` | ` need` | 0.125 | 2.547 | **2.547** | 40 |
| en_math | 2 | ` asks` | ` is` | 0.516 | 1.859 | **1.859** | 61 |
| **合计** | 2–23（8/8 翻转） | — | — | 0.016–0.750 | 0.047–2.547 | **max 2.547 nat** | **380 / 490（77.6%）** |

`opaque(triton)` 与 `opaque(sdot4)`（§3.3.1）在 **8/8 prompt 上都翻转**，并且
`zh_hz@9 / zh_ml@7 / en_fr@13 / en_sea@10 / en_math@2` 这 5 个"早期分叉点"**完全重合、方向一致**
（zh_sort/en_index 的分叉点从 16/4 移到 23/23，是因为两者各自的编译产物不同）。
⇒ 与 §3.2.4 的判定一致：**翻转是 opaque op 引起的图/编译变化，与内核数学无关**。

> §3.3.3 见下（arm4 = 第三次 `=1` 采样）。

#### 3.3.3 arm4（`=1`，`--max-model-len 8192`）vs arm1：第三次 `=1` 采样

arm4 是**同开关、同源码、同卡**的第 3 次 `=1` 启动（唯一附加差异是 `--max-model-len 8192`）。
它同样**没有吃到 AOT 缓存**（`torch.compile took 88.35 s`，冷编），
与 arm1 的对比（`.tmp-11l/cmp-arm1-arm4.json`）：

| prompt | 首个分叉位 | arm1 选的 | arm4 选的 | max_matched_delta |
|---|---|---|---|---|
| zh_hz | 11 | — | — | 1.264（@10） |
| zh_ml | **0** | — | — | 0.000 |
| zh_sort | 16 | — | — | 0.583 |
| zh_sky | 2 | — | — | 0.482 |
| en_fr | 12 | — | — | 0.675 |
| en_sea | 10 | — | — | 0.298 |
| en_index | 2 | — | — | 0.651 |
| en_math | **0** | — | — | 0.844 |
| **合计** | 0–16（8/8） | | | **399 / 512（77.9%）** |

⇒ **第 3 个 `=1` 引擎也给出第 3 套文本**（zh_ml / en_math 甚至在**第 0 个 token** 就分叉）。
三次 `=1` 启动 = 三套不同文本（S11k、arm1、arm3/arm4），而两次 `=0` 启动 = 逐位相同（§3.3.0）。
这就是"§3.2.5 AOT 缓存每次重载失败 + timing-based autotune"的直接后果。
**（诚实标注）** arm4 与 arm1 还差一个 `--max-model-len`（8192 vs 4096），所以这一对是"两次 `=1` 启动文本不同"
的**附加**证据、不是严格同配置；严格同配置的对照是 **arm1 vs arm3**（341/490 位置不同、`max_matched_delta 1.401`，§3.2.5）。

### 3.4 同口径对拍（S9 协议）

协议（`tools/dcu/s11l_bench.sh` 的 flag 逐字抄自 S9 的 `/mnt/game/port/s9/bench.sh`，客户端都是
`vllm bench serve --backend openai-chat`、`seed 0`、`--ignore-eos`、`T=0`）：

| 场景 | 客户端口径 | S9 生产基线口径 | 说明 |
|---|---|---|---|
| `decode` | random in=**128** out=**128** np=2 warmups=1 **conc=1** | 同（完全一致） | 单流纯 decode，指标取 **Mean TPOT ms** |
| `ttft` | random in=**3800** out=1 np=1 warmups=0 conc=1 | in=**8192** out=1（生产 `--max-model-len` 更大） | 本步 `=1` 臂 `--max-model-len 4096` ⇒ 沿用 S9 对拍时的 3800 口径，另加 `ttft8k`（in=8192，在 8192 臂上）做**完全同形**对照 |
| `conc8` | in=512 out=128 np=16 warmups=2 **conc=8** | 计划文档写 64 请求；S9 记录"base 轮未跑" | 无历史对照，只报本步值与 S9 自测值 |

S9 归档原文（`docs/result/z100-09-acceptance-and-bench-result.md` §3.13）：

| 指标 | 生产基线（:8000，**TP=4、eager、bf16、sdot4 + gdn_torch**） | S9 新引擎（:8003，TP=2、PIECEWISE、fp16） | S9 偏差 |
|---|---|---|---|
| 单流 decode TPOT | **122.47 ms/token**（7.75 tok/s） | 380.70 ms/token（2.58 tok/s） | 3.11× 慢 |
| 冷 TTFT | **22,850.89 ms @8192**（2.789 ms/prefill-token） | 12,456.53 ms @3800（3.278 ms/prefill-token） | +17.5% / prefill-token |
| 并发 8（in512/out128×16） | 无历史数据 | 14.78 tok/s（out）/ 79.97 tok/s（总） | — |

S9b 复测（FULL 图）：decode **378.78** / PIECEWISE 380.62 ms/token ⇒ 图模式只值 0.5%，
3.09× 的欠账在 int8/GDN 内核与 TP=4。

**本步 `=1` 结果**（arm1，卡 0/1、TP=2、4096、默认 cudagraph（实测按 PIECEWISE 捕获 51 段）、复用
`/mnt/game/vllm-cache-27b` 热缓存；启动 `torch.compile took 54.15 s`、两轮 graph capture
`683 s` + 第二轮、KV 11.54 GiB、`Maximum concurrency for 4,096 tokens: 39.33x`）：

| 场景 | `=1`（本步 arm1） | `=0`（本步 arm2，同命令对照） | `=1`/`=0` | S9 `=0`/PIECEWISE | 生产基线 | `=1` vs 生产基线 |
|---|---|---|---|---|---|---|
| `decode` Mean TPOT（in128/out128 np2 conc1） | **84.00 ms/token**（Median 84.00、Mean ITL 83.35） | **321.09 ms/token**（Median 321.09、Mean ITL 318.58） | **3.82×** | 380.70 | **122.47** | **快 1.458×** |
| `decode` Output token throughput | 11.30 tok/s | 3.05 tok/s | 3.71× | 2.58 | 7.75 | 1.458× |
| `ttft` Mean TTFT（in=3800，out=1） | **8,606.28 ms**（**2.234 ms/prefill-token**） | **11,388.24 ms**（2.996 ms/token） | 1.32× | 12,456.53 ms（3.278 ms/token） | 22,850.89 ms @8192（2.789 ms/token） | 每 prefill token 快 1.248× |
| `conc8`（in512/out128 np16 conc8） | **33.12 tok/s**（out）/ 179.19（总）；Mean TTFT 7,670.80 ms、Mean TPOT 179.58 ms、E2EL 30,477.64 ms | 17.03 / 92.12；Mean TTFT 12,360.67 ms、Mean TPOT 368.57 ms、E2EL 59,169.55 ms | **1.94×** | 14.78 / 79.97；TTFT 13.0 s、TPOT 433 ms | 无历史数据 | — |
| 客户端 e2e（`s11k_e2e_client.py`，贪心 129 token） | chat_zh **83.89** / chat_en **83.94** ms/token；ttft 412/406 ms | chat_zh **320.99** / chat_en **321.27** ms/token；ttft 850/814 ms | **3.83× / 3.83×** | 329.05 / 320.30（S11k 同命令对照） | — | — |

**口径说明（不许含糊）**：
* `decode` 是**纯 decode**：`--random-input-len 128 --random-output-len 128`、`--ignore-eos`、
  `T=0`、`np=2`、`warmups=1`、`conc=1`，指标 = `vllm bench serve` 的 **Mean TPOT（excl. 1st token）**；
  与 S9 归档 §3.13 的客户端口径**逐字相同**，因此 84.00 vs 122.47 是可比的（1.458×）。
* **但与生产基线仍有 3 处不可消除的口径差**（S9/S11b 已记录，本步不重新归因）：① 生产 = **TP=4**，本步 = TP=2；
  ② 生产 = `--enforce-eager`（无 graph capture），本步 = 默认 compile+cudagraph（S9b 实测 FULL vs PIECEWISE 只差 0.5%）；
  ③ 生产 = **bf16**，本步 = `--dtype float16`（硬约束 18）。⇒ 1.458× 是"同客户端、不同服务配置"的对比，
  保守说法是"**`=1` 已越过生产基线**"，而不是"同配置净提速 1.458×"。
* `ttft` 的 3800 vs 基线 8192 是 S9 就留下的差异（本步 4096 臂无法收 8192 prompt）；按 **每 prefill token**
  对比（2.234 vs 2.789）并另在 8192 臂上做 `ttft8k` 同形对照：in=**8,154** → TTFT **20,959.41 ms
  = 2.571 ms/prefill-token**，比基线 2.789 好 1.085×（§3.5.4）。
* `conc8` 生产基线**没有历史数据**，只能与 S9 自测的 14.78 tok/s 比（本步 =1 为 33.12 tok/s = 2.24×）；
  与**本会话同一台机、同一批 flag 的 `=0` 对照**（17.03 tok/s）比是 **1.94×** —— 后者才是干净的同口径数字
  （S9 的 14.78 是另一台容器/另一批图配置）。
* **同会话干净 A/B 一览**（唯一变量是 `VLLM_DCU_USE_INT8_SDOT4`）：decode **84.00 / 321.09 = 3.82×**、
  TTFT(3800) **8,606 / 11,388 = 1.32×**、conc8 **33.12 / 17.03 = 1.94×**、客户端 e2e **83.9 / 321.1 = 3.83×**。
  与 S11k 的 3.91×/3.87× 一致（差异来自机器波动：S11k 的 `=0` 是 329.05/320.30，本步 320.99/321.27）。

### 3.5 端到端回归

#### 3.5.1 `s8b_gate.sh`（`=1`）：首次 FAILED → 归因到**过期断言** → 修好后 ALL GREEN

第一次以 `VLLM_DIR=/src VLLM_DCU_USE_INT8_SDOT4=1 S8B_HIP_VISIBLE_DEVICES=3` 跑（卡 3）：

```
CHECK on_gfx906/on_gfx9/cdna_version     PASS on_gfx906=True on_gfx9=False cdna_version=0
CHECK ROCM_SKINNY_GEMM_ENABLED           PASS ...
CHECK VLLM_ROCM_USE_SKINNY_GEMM=0        PASS ...
CHECK _rocm_C ops == [paged_attention]   FAIL ['_rocm_C::paged_attention', '_rocm_C::w8a8_sdot4_gemm', '_rocm_C::w8a8_sdot4_gemv']
...
CHECK_SUMMARY pass=14 fail=1
  conventions/kill-list : rc=1
  FP8 block-scaled      : rc=0
  W4A16 selection test  : rc=0
  RESULT: FAILED
```

**归因**：这条断言是 **S8b 时代写的**，而 S11e 已经往 `_rocm_C` 里注册了 `w8a8_sdot4_gemm/gemv`
（P2 的内核，canonical `_rocm_C` md5 `ebd6fb8d…` 里就有）。断言写死 `== [paged_attention]`，
于是**无论 `VLLM_DCU_USE_INT8_SDOT4` 是 0 还是 1**都会失败（两个 op 的注册与开关无关，
只与 `_rocm_C` 二进制有关）⇒ 这是**过期门禁**，不是本步的功能回归。开关无关性实测：

```
$ for sw in 0 1; do VLLM_DCU_USE_INT8_SDOT4=$sw python -c '... _jit_get_all_schemas() ...'; done
[probe] sw= 0 ['_rocm_C::paged_attention', '_rocm_C::w8a8_sdot4_gemm', '_rocm_C::w8a8_sdot4_gemv']
[probe] sw= 1 ['_rocm_C::paged_attention', '_rocm_C::w8a8_sdot4_gemm', '_rocm_C::w8a8_sdot4_gemv']
```

**修复**（`tools/dcu/s8b_gate.sh:80-96`，写作用域内）：把断言改成**显式允许清单**
`[paged_attention, w8a8_sdot4_gemm, w8a8_sdot4_gemv]` 并写明"该 check 的目的是'没有别的 ROCm 内核混进来'，
不是'把面冻结在 S8b 那一版'"。修复后 `=1` 复跑：

```
CHECK _rocm_C ops == allow-list(paged_attention,sdot4_*) PASS [...]
CHECK_SUMMARY pass=15 fail=0
  conventions/kill-list : rc=0
  FP8 block-scaled      : rc=0
  W4A16 selection test  : rc=0
  RESULT: ALL GREEN
```

（另：`_C_stable_libtorch` 身份 `93bf7b4a391ac97e2f67cc7141633cb1` / 40,691,216 B = S7c 参考值，info 项 PASS。）

#### 3.5.2 W4A16（tinyllama group128，1 卡，`=1`）：HTTP 200 + 文本与 S9 一致

```
Selected/Using Gfx906GPTQWNA16LinearKernel for mixed-precision linear
Using Gfx906GPTQWNA16LinearKernel for CompressedTensorsWNA16
Overriding with TRITON_ATTN out of potential backends: ['TRITON_ATTN', 'ROCM_ATTN'].
Available KV cache memory: 8.2 GiB / GPU KV cache size: 390,784 tokens
Maximum concurrency for 2,048 tokens per request: 190.81x
Application startup complete.
请求（`/v1/completions`，greedy，32 token，prompt "The capital of France is"）：
HTTP 200 usage={"prompt_tokens": 6, "completion_tokens": 32}
text=" Paris. The capital of France is Paris. The capital of France is Paris. ..."
```

与 S9 §3.11.4 的归档文本**逐字一致**（S9：KV 8.17 GiB、同样的 " Paris. The capital of France is Paris…"）；
`=1` 不触及 W4A16 路径（sdot4 只服务 W8A8-per-channel-int8）⇒ 这是"打开 `=1` 后别的量化路径不受影响"的证据。
（第一次跑我误用了 `/v1/chat/completions`，tinyllama 无 chat template，输出退化成 `<|assistant|>`；
改用 S9 的 `/v1/completions` 口径后正常——记录在此，避免后来者踩同一个坑。）

#### 3.5.3 27B W8A8 INT8 TP=2 完整请求（`=1`，arm1）

原始日志 `.tmp-11l/arm1-e2e.log`（客户端 `tools/dcu/s11k_e2e_client.py`，与 S11k 同一客户端；
`--max-model-len 4096`、TP=2、卡 0/1、默认 cudagraph、无 `--enforce-eager`）：

```
启动：Selected Gfx906Sdot4Int8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
      torch.compile took 54.15 s（热缓存）；Graph capturing finished in 683 secs ×2 workers
      Available KV cache memory: 11.54 GiB / GPU KV cache size: 161,109 tokens
      Maximum concurrency for 4,096 tokens per request: 39.33x
      Application startup complete.
chat_zh：tokens=129、ttft=412.0 ms、decode=83.89 ms/token
   text="用户要求"用一句话介绍杭州"——这是一个非常开放的请求，需要"用一句话介绍杭州"。…"
chat_en：tokens=129、ttft=406.3 ms、decode=83.94 ms/token
   text="The user is asking "What is the capital of France?" - the capital of France is Paris. …"
```

**文本是否合理**：两条都是通顺、切题的中/英文（`=1` 与 `=0` 都是"先复述问题再作答"的思考体，
前 ~40 token 语义连贯）；129 token 的贪心在 `temperature=0` 下后段进入复读（`钱塘江、钱塘江…`
/ `The user is asking …` 循环），这是**模型在该 prompt + 贪心 + 短输出上限下的既有行为**，
`=0` 对照同样是复读结构（见 `.tmp-11l/arm2-e2e.log`）⇒ 判为"文本合理"，不是 `=1` 引入的退化。
**注意**：`=1` 与 `=0` 的文本**不同**（§3.3.1 第 9 个 token 起分叉），这条门禁（L5）只判"能正常出文本"，
"文本逐字一致"是单独的第 L3 条，判定见 §4。

（`=1` 与 `=0` 在同一个 prompt 上的 token 序列/分叉点见 §3.3.1；`=0` 的 e2e 数字在 §3.4。）

#### 3.5.4 8K `TRITON_ATTN`（arm4 = `=1`、`--max-model-len 8192`）

启动日志（`.tmp-11l/arm4.log`，两个 rank 各一条）：

```
INFO [rocm.py:899] Found incompatible backend(s) [TURBOQUANT] with AttentionType.DECODER.
  Overriding with TRITON_ATTN out of potential backends: ['TRITON_ATTN', 'ROCM_ATTN'].
INFO [__init__.py:805] Selected Gfx906Sdot4Int8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
INFO [gpu_worker.py:719] Available KV cache memory: 11.53 GiB
INFO [kv_cache_utils.py:2432] GPU KV cache size: 227,448 tokens,
  Maximum concurrency for 8,192 tokens per request: 27.76x
```

8K 长度的**真实请求**（S9 的 `ttft8k` 口径：random in=8100→实际 **8,154** input token、out=1、
np=1、warmups=0、`--ignore-eos`；对照的生产基线是 in=**8192**）：

```
Successful requests: 1   Failed requests: 0
Total input tokens: 8,154        Mean TTFT: 20,959.41 ms  (= 2.571 ms/prefill-token)
生产基线 @8192：22,850.89 ms     (= 2.789 ms/prefill-token)  ⇒ 每 prefill token 快 1.085×
```

⇒ **8K 门禁通过**：`TRITON_ATTN` 在选择日志里、8,154-token 请求成功、
TTFT 与生产基线**同形对照**（8192 vs 8154）且**不劣于**基线（2.571 vs 2.789 ms/token，总时长 20.96 s vs 22.85 s）。

**未做（登记）**：原本还想在 arm4 上跑 `tools/dcu/s11l_8k_probe.py`（模型自带 tokenizer 造 ~7.5k prompt、
生成 64 token，给出"长上下文下的文本"）。它**失败了**：编排脚本把该任务写成 `docker exec python …`
（没有 source DTK env）→ `ImportError: librocm_smi64.so.2`（`transformers`→`torch` 起不来）。
修法已定（`bash -lc` + `$ENV`，本次收尾已改 `tools/dcu/s11l_run_arm.sh`），但重跑要再起一次 8192 引擎（~35 min）；
由于"8K 上下文 + `TRITON_ATTN` + 真实长请求成功"三项证据已由 `ttft8k` 取得，**未重跑**。

### 3.6 临时证据归置（S11k / S11g / S11l）

**本步（S11l）的证据路径**（容器 `/mnt/game/s11l/`，1.5 MB）：

| 内容 | 路径 |
|---|---|
| 逐 token logprob（4 个臂 + arm3 的重跑） | `/mnt/game/s11l/flip-arm{1,2,3,3b,4}.jsonl` |
| S9 协议 bench 原始日志 | `/mnt/game/s11l/bench-arm{1,2}-{decode,ttft,conc8}.log`、`bench-arm4-ttft8k.log` |
| 7 文件测试日志（`=1`/`=0`） | `/mnt/game/s11l/tests-sw{1,0}/*.log` |
| `s8b_gate.sh` + W4A16 相关日志 | `/mnt/game/s11l/s8b/` |

宿主侧 `.tmp-11l/`：4 个引擎的完整启动/客户端 stdout（`arm{1,2,3,4}.log`、`arm1-flip.log`、
`arm1-e2e.log`、`arm*-bench-*.log`、`gate-w4a16*.log`、`quant-sensitivity.log`、`prefill-m-range.log`）
与两两比对 JSON（`cmp-arm*.json`）；这些是小文件，留在工作区便于 Lead 抽查
（原始 jsonl 已同步在 `/mnt/game/s11l/`，宿主副本收尾时删除）。

`.tmp-11k/` 是 S11k 的临时脚手架（原始 jsonl + 起服务日志），`.tmp-11g/` 只剩一个空文件 `x`。
原始证据已**整份复制**到容器可见的 `/mnt/game/s11l/evidence-s11k/`（`evidence-s11g/`），
宿主与容器两侧 `md5sum` **15/15 完全一致**后才从工作区删除原始文件：

```bash
docker exec z100-s9 bash -c 'mkdir -p /mnt/game/s11l/evidence-s11k /mnt/game/s11l/evidence-s11g; \
  cp -a /src/.tmp-11k/. /mnt/game/s11l/evidence-s11k/; cp -a /src/.tmp-11g/. /mnt/game/s11l/evidence-s11g/'
# 两侧 md5sum 对比：diff 为空 → MD5_ALL_MATCH
```

宿主 `.tmp-11l/` 的全部 34 个文件（引擎日志 + 客户端 stdout + 比对 JSON）也**整份复制**到
`/mnt/game/s11l/hostlogs/`（宿主 vs 容器 `md5sum` **34/34 一致**），因此归档里所有 `.tmp-11l/…`
路径都有对应的 `/mnt/game/s11l/hostlogs/…` 副本。**工作区干净的判据**：
`.tmp-11k/` 只剩 4 个小脚本（S11k §7 引用）、`.tmp-11g/` 已删除、`.tmp-11l/` 只留日志/比对 JSON（1.5 MB，无大 jsonl）。

| 证据 | 原路径（工作区，已删） | 现路径（/mnt/game） | md5 |
|---|---|---|---|
| zh/en 客户端 jsonl | `.tmp-11k/arm0-client.jsonl` | `/mnt/game/s11l/evidence-s11k/arm0-client.jsonl` | `bba714e37c4ab83246e6258b84f20584` |
| zh/en 客户端 jsonl（转录） | `.tmp-11k/arm1-client.jsonl` | 同上目录 | `ce41fda08ba40a9a1a6b207946ead6c9` |
| 逐 token logprob（`=0`） | `.tmp-11k/logprob-ctl0.jsonl` | 同上目录 | `b6ab52fe7b8d005c309a0250b363c4ce` |
| 逐 token logprob（`=1`） | `.tmp-11k/logprob-sdot4.jsonl` | 同上目录 | `8a8b0f9c1db8fbcc7976cecc19c17eed` |
| `=0`/`=1` 引擎日志 | `.tmp-11k/ctl0.log`、`sdot4fix1{,b}.log` | 同上目录 | `3f27e9b5…` / `c2f55b56…` / `ddce4617…` |
| S11j 剖析日志 | `.tmp-11k/s11j-{sdot4,triton,chain}.log` | 同上目录 | `3239cd16…` / `5218edf8…` / `b8c09774…` |
| S11g 空目录 | `.tmp-11g/x`（0 B） | `/mnt/game/s11l/evidence-s11g/x` | — |

> 工作区 `.tmp-11k/` 现在只保留 4 个**小脚本**（`arm_env.sh`、`s11j_env.sh`、`s11j_both.sh`、`s11j_summary.sh`），
> 因为 S11k 归档 §7 的复现命令引用它们；S11k 归档里引用的**日志/jsonl** 现已改在
> `/mnt/game/s11l/evidence-s11k/`（S11k 归档文本未改，此处给出新路径，归档"只增不改"）。
> 本步自己的证据：宿主 `.tmp-11l/`（编排/引擎日志、客户端 stdout），容器 `/mnt/game/s11l/`（flip jsonl、bench 日志、测试日志）。

## 4. 门禁判定

| # | 门禁 | 判定 | 证据 |
|---|---|---|---|
| L1 | 测试套件（`=1`）：7 个 gfx906 文件全绿，且与 `=0` 计数一致 | ✅ **通过** | §3.1：`=1` **129 passed / 0 failed**、`=0` **129 passed / 0 failed**，逐文件 passed 数完全相同 |
| L2 | 翻转归因（隔离实验给出明确结论） | ✅ **通过** | §3.2.4/§3.2.6：`opaque(triton)` vs `bare triton` **8/8 prompt 照样翻转**、5 个早期分叉点与 `opaque(sdot4)` 完全重合 ⇒ **H_graph（custom_op 引起的图/编译变化）成立；H_kernel（sdot4 内核数学）否定**（内核 126+168 组逐位相同）。附加发现：`=1` 的 AOT 产物**每次启动都加载失败→重编**，叠加 `benchmark_combo_kernel=True` ⇒ 同开关两次启动的 logits 可差 0.3–1.4 nat |
| L3 | 翻转边界统计（zh/en 各 ≥4 prompt、间距、位置、max delta、翻转率、是否都 <0.5 nat） | ✅ **通过** | §3.3.1/§3.3.2：8 条 prompt（zh×4/en×4）× 64 token；**8/8 都翻转**，首个分叉位 2–23；`max_matched_delta` 1.108 / 1.401 nat；**16 个分叉点中 13 个的"被放弃候选"在另一臂里只落后 <0.5 nat，另 3 个 0.516–0.531 nat** ⇒ 翻转**全部落在 ≤0.53 nat 的贴边决策**上；但另一臂在该点可以很自信（最高 2.547 nat）⇒ 两个分布本身有 0.3–2.5 nat 级差异 |
| L4 | 同口径对拍（S9 协议 + vs 生产基线 122.47） | ✅ **通过** | §3.4：`=1` decode **84.00 ms/token**、TTFT(3800) **8,606 ms**、conc8 **33.12 tok/s**；同会话 `=0` 对照 321.09 / 11,388 / 17.03 ⇒ **3.82× / 1.32× / 1.94×**；vs 生产基线 122.47（TP=4/eager/bf16）**快 1.458×**（口径差已注明） |
| L5 | 端到端回归（27B 完整请求 / W4A16 / 8K / `s8b_gate.sh`） | ✅ **通过** | 27B：§3.5.3（129 token 文本通顺、KV 11.54 GiB、`Maximum concurrency 39.33x`）；W4A16：§3.5.2（HTTP 200、文本与 S9 逐字一致、KV 8.2 GiB）；`s8b_gate.sh`：§3.5.1（**15/15 + ALL GREEN**，含过期断言修正）；8K：§3.5.4（`TRITON_ATTN` 日志 + **8,154-token 真实请求**成功，TTFT 20,959.41 ms = 2.571 ms/token，优于基线 2.789） |
| L6 | 默认值建议 + 上线配置 | ✅ **通过** | §6（建议：**仓库默认保持 OFF；生产显式 `=1`**，并给出配置与"逐 token 一致不可承诺"的说明） |
| L7 | 边做边写 | ✅ **通过** | 本档在任务开始时就建骨架，§3.0–§3.6 随执行追加（含一次 arm1 被误杀的操作记录，§5.1） |
| L8 | 临时证据整理 | ✅ **通过** | §3.6：`.tmp-11k/` 原始 jsonl/日志 15 个文件 md5 校验后移入 `/mnt/game/s11l/evidence-s11k/`，`.tmp-11g/` 空目录移入 `evidence-s11g/` 并删除；工作区只留 4 个小脚本 |

**总体**：S9 的验收清单在 `=1` 下**重跑通过**（L1/L5），性能收益在**同会话干净 A/B** 下是
**3.82× decode**、且已经**越过生产基线 1.458×**（L4）；文本差异的定性（L2/L3）给出了明确、
可复现的结论：**不是内核数学、是编译图/编译产物**，并且**本树连"同一个开关两次启动"都不保证同样的文本**。
⇒ 收益可上线（见 §6），但"逐 token 与 Triton 基线一致"这条**不能承诺**。

## 5. 未完成项 / 阻塞与原因

### 5.1 操作记录：arm1 第一次启动被我的 readiness 上限误杀（已修）

27B 在 `=1` 下启动是**两轮 CUDA graph capture**（第一轮 ~683–703 s 捕获 51 段 PIECEWISE + 一轮 FULL，
打印 `Graph capturing finished`，随后分配 KV cache；然后**再来一轮** 51 段、~17–18 min）。
我第一次给编排脚本的等待上限是 **1500 s**，arm1 在第 2 轮 38/51 处被 kill（日志 `.tmp-11l/arm1.log`
第一份，已覆盖）。改成 `READY_TIMEOUT=2700 s` 后重跑成功（`ready=1 after 1583 s`）。
实测每次 27B 启动（热缓存）：`torch.compile` **22–88 s** + 两轮 capture ≈ **26–33 min** 才 ready；
本步共启动 4 次（arm1/arm2/arm3/arm4），arm2 30 min、arm3 26 min。
（这是既有行为，不是本步引入；S9 §3.10.2 也记过"两轮捕获 ≈39 min"。）

### 5.2 未定位：`=1` 的 AOT 产物为什么每次加载都判"源码变了"

§3.2.5 给出了现象（`=1` 每次启动都 `Source code has changed … Recompiling`）与代码位置
（`vllm/compilation/decorators.py:266 _verify_source_unchanged`）、以及差异事实
（`=1` 的产物把 `gfx906_sdot4.py` 作为 traced file；=0 的产物没有）。
**但"源码未变却哈希不一致"的确切原因未定位**。可执行的下一步（不在本步范围）：
用 `VLLM_LOGGING_LEVEL=DEBUG` 起一次 `=1`（会打印 `Traced files (to be considered for compilation cache)`），
或 `VLLM_FORCE_AOT_LOAD=1` 让异常直接抛出；两者都只需一次 27B 启动（~30 min）。
本步不改运行时代码（任务书要求），因此只登记。

### 5.3 隔离实验的"同产物"前提不成立（已换对照口径，结论不受影响）

原计划希望 arm3（`INNER=triton`）能**复用 arm1 的 AOT 产物**，从而做到"只换 body 内核、其余编译产物不变"。
实测做不到：§3.2.5 说明 `=1` 每次启动都重编，arm3 也重编（`saved AOT compiled function`，54.07 s），
所以 `arm1 vs arm3` 的差异（341/490）**混入了编译产物差异、不能当作内核差异**。
⇒ 改用任务书原本要求的对照：**`opaque(triton)`（arm3）vs `bare triton`（arm2，即 `=0`）**，
这一对本来就是"图变、数学不变"的严格隔离，结论（H_graph）成立。
`arm1 vs arm3` 反而成了"同开关两次启动文本不同"的**附加证据**（§3.2.5 末）。

### 5.4 其它未做事项（明确登记）

1. **未定位到具体是哪个非 int8 算子**在两份编译产物之间产生差异（S11k §5.2 的同一未完成项）。
   本步把它量化了（§3.2.2：量化把 1 ULP 放大 2–5×、离散跳变），但没有做 prefill 段的算子级 trace 对比
   （代价 ≈1 轮启动 + trace 分析）。
2. **TP=4 未对拍**（生产基线是 TP=4；本步按硬约束用 TP=2，卡 0/1）。
3. **`cudagraph_mode=NONE` 未复测**（S11i 的 fork 缺陷；与本步无关）。
4. **`=0` 的 gate 未再跑一遍**（§3.5.1 已用"op 注册与开关无关"的直接探针证明该断言对 0/1 都成立）。

## 6. 结论与建议

### 6.1 默认值：**保持 OFF**（仓库现状），生产**显式开 `=1`**

| 方案 | 建议 | 依据 |
|---|---|---|
| 仓库默认值 `VLLM_DCU_USE_INT8_SDOT4` | **保持 `False`（OFF）** | ① 它**改变服务输出**（8/8 prompt 在 ≤0.53 nat 的贴边决策上翻转，见 §3.3）；② `=1` 的 AOT 产物**每次启动都加载失败→重编**，叠加 timing-based autotune ⇒ **同一个开关两次启动的文本都不一样**（§3.2.5）；③ 仓库默认值应当是最保守、可复现的路径（`=0` 跨引擎逐位可复现，§3.3.0）。把"会改输出且不可复现"的路径设为默认，等于让所有用户无感地换掉输出 |
| 生产部署 | **显式 `VLLM_DCU_USE_INT8_SDOT4=1`** | 收益巨大且经过本步重跑验收：**decode 3.82×、TTFT 1.32×、conc8 1.94×**（同会话 A/B），纯 decode **84.00 vs 生产基线 122.47 ms/token = 快 1.458×**；测试套件 129/129、27B/W4A16/8K 端到端与 `s8b_gate.sh` 全部通过；数值上线性层与 Triton 逐位相同、无 NaN/崩溃 |
| "近似并列翻转"是否可接受 | **质量类用途：可接受；需要复现性的用途：不可接受** | 翻转点全部是 ≤0.53 nat 的贴边决策，两条续写都通顺、语义等价；但**必须在文档/发布说明里提示**："开启后 greedy 输出与 Triton 路径不同，且不保证跨重启逐字一致" |

**若要 `=1` 同时可复现**（推荐作为后续修复项，本步未做）：① 修 `=1` 的 AOT 缓存重载
（§5.2）——让 `=1` 不再每次重编；② `-cc.inductor_compile_config.benchmark_combo_kernel=False`
（去掉 timing-based 选择，S9 §3.12(b) 也曾为启动时间提过这条）；③ 固定 `VLLM_CACHE_ROOT`。
做完 ①② 后应复验"两次启动 token 相同"；此时 `=1` 与 `=0` 的文本差异仍然存在（图不同），
但那是一个**稳定、可记录**的差异，而不是每次启动抽签。

### 6.2 生产上线推荐配置（27B W8A8 INT8，TP=2，卡 0/1）

```bash
export VLLM_DCU_USE_INT8_SDOT4=1                 # 本步验收的开关（默认 OFF，需显式开）
export VLLM_CACHE_ROOT=/mnt/game/vllm-cache-27b  # 必须显式设置（硬约束 17）
export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
export PYTHONPATH=/mnt/game/triton-gfx906/install:/mnt/game/port/gfx906-vllm-211
python -c 'import torch; from vllm.entrypoints.cli.main import main; main()' serve \
  /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 \
  --dtype float16 \                              # 硬约束 18：gfx906 一律 fp16
  --tensor-parallel-size 2 \                     # 卡 0,1
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \                         # 8K：见 §3.5.4（TRITON_ATTN）
  --max-num-batched-tokens 2048 \
  --enable-prefix-caching --mamba-cache-mode align \
  --disable-custom-all-reduce --skip-mm-profiling \
  --served-model-name qwen3.8-27b --host 127.0.0.1 --port <PORT>
  # 不要加 --enforce-eager（硬约束 13）
```

| 资源/口径 | `=1` 实测 | 备注 |
|---|---|---|
| 可用 KV cache（4096 臂） | **11.54 GiB / 161,109 token / `Maximum concurrency 39.33x`** | 与 `=0`（11.54 GiB）**相同**（S11k 亦为 11.53） |
| 可用 KV cache（8192 臂） | **11.53 GiB / 227,448 token / `Maximum concurrency 27.76x`** | maxlen 8192 下 token 数反而更多（227,448 vs 161,109），KV 内存同量级 |
| 启动 | `torch.compile` 54 s（`=0` 22 s）+ 两轮 capture ≈ 26–33 min | `=1` 每次都会重编（§5.2） |
| decode | 84.00 ms/token（纯 decode，conc=1，in128/out128） | 同会话 `=0` = 321.09 ⇒ 3.82× |
| TTFT | 2.234 ms/prefill-token（in=3800） | 同会话 `=0` = 2.996 ⇒ 1.32× |
| 并发 8 吞吐 | 33.12 tok/s（out）/ 179.19（总） | 同会话 `=0` = 17.03 / 92.12 ⇒ 1.94× |

### 6.3 给 Lead 的两个待决事项

1. **canonical 副本的 `gfx906_sdot4.py` md5 已变**：工作区现在是 `59e669644855ad54efe2ee98e61807e1`
   （S11k 的 `45854752…` + 本步诊断分支），默认行为**逐字节不变**（`VLLM_DCU_SDOT4_INNER` 不设就是 sdot4 路径）。
   二选一：① 同步到 canonical（保留诊断开关，便于复现 §7 的隔离实验）；② S11l 收尾后回退这一处改动
   （仓库回到 S11k 状态）。本步**不改** canonical（沙箱限制），请 Lead 决定。
2. **`=1` 的 AOT 缓存重载缺陷**（§5.2）建议单独立项：它既影响复现性，也白花每次启动的 ~32 s 编译。

## 7. 复现方式

```bash
# 7.0 容器与通用 env（下文用 $ENV 代指）
#   . /opt/dtk/env.sh; . /src/tools/dcu/env.sh
#   export PYTHONPATH=/src:/mnt/game/triton-gfx906/install
#   export TRITON_CACHE_DIR=/mnt/game/triton-gfx906/cache
#   export VLLM_CACHE_ROOT=/mnt/game/vllm-cache-27b

# 7.1 7 个测试文件 × {=1,=0}（容器内）
docker exec z100-s9 bash /src/tools/dcu/s11l_tests.sh 1 /mnt/game/s11l/tests-sw1
docker exec z100-s9 bash /src/tools/dcu/s11l_tests.sh 0 /mnt/game/s11l/tests-sw0

# 7.2 诊断开关门禁（卡 3）
docker exec z100-s9 bash -lc "$ENV; export HIP_VISIBLE_DEVICES=3 VLLM_DCU_USE_INT8_SDOT4=1; \
  for i in sdot4 triton; do VLLM_DCU_SDOT4_INNER=$i python /src/tools/dcu/s11l_inner_probe.py; done"

# 7.3 量化敏感度 / prefill M 补齐（卡 3）
docker exec z100-s9 bash -lc "$ENV; export HIP_VISIBLE_DEVICES=3; \
  python /src/tools/dcu/s11l_quant_sensitivity.py; python /src/tools/dcu/s11l_prefill_m_range.py"

# 7.4 四臂端到端（宿主侧编排；每次 ~30 min 启动）
tools/dcu/s11l_run_arm.sh arm1 1 sdot4 8021 4096 flip,e2e,decode,ttft,conc8   # =1
tools/dcu/s11l_run_arm.sh arm2 0 sdot4 8022 4096 flip,e2e,decode,ttft,conc8   # =0（对照）
tools/dcu/s11l_run_arm.sh arm3 1 triton 8023 4096 flip,flipb,e2e             # 隔离：opaque(triton)
tools/dcu/s11l_run_arm.sh arm4 1 sdot4 8024 8192 flip,8k,ttft8k              # 8K

# 7.5 翻转比对（宿主侧，纯 stdlib）
docker exec z100-s9 cat /mnt/game/s11l/flip-arm2.jsonl > /tmp/a.jsonl
docker exec z100-s9 cat /mnt/game/s11l/flip-arm3.jsonl > /tmp/b.jsonl
python tools/dcu/s11l_flip_compare.py --a /tmp/a.jsonl --b /tmp/b.jsonl \
  --label-a ctl0_baretriton --label-b opaque_triton --json-out /tmp/cmp.json
python tools/dcu/s11l_flip_summary.py /tmp/cmp.json

# 7.6 gate + W4A16（卡 3）
docker exec z100-s9 bash -lc 'export VLLM_DIR=/src VLLM_DCU_USE_INT8_SDOT4=1 S8B_HIP_VISIBLE_DEVICES=3; \
  /src/tools/dcu/s11l_gate_and_w4a16.sh'
```

## 8. 参考

1. `docs/result/z100-11k-sdot4-compile-integration-result.md`（S11k：编译集成修法、真机 A/B 3.91×/3.87×、文本翻转初步归因）
2. `docs/result/z100-11e-w8a8-sdot4-result.md`（S11e：内核、63/63 与 `triton_scaled_mm` 逐位相同、逐 shape 性能）
3. `docs/result/z100-09-acceptance-and-bench-result.md` §3.6/§3.13 + `docs/result/z100-09b-full-cudagraph-result.md` §3.4（7 测试文件、生产基线 122.47 口径）
4. `docs/海光Z100-性能优化方案.md` §1.1/§1.2/§4；`docs/z100-todolist.md` 硬约束 1–23；`docs/result/README.md` 模板
