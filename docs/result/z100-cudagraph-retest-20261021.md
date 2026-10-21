# Z100（gfx906）开启图模式复测：三臂真机结果（2026-10-21）

> 平台：4×海光 DCU Z100/Z100L（gfx906）、DTK 26.04 + torch 2.11、`/home/lyh/z100-vllm`（HEAD `c53840d03`）
> 模型：`Qwen3.8-27B-FP8` + DFlash2 投机解码 k=7、TP=4、maxlen 262144、MBT 4096、`max_num_seqs=4`、util 0.85
> 窗口：2026-10-21 21:35–22:00（Asia/Shanghai），生产 unit 已停、4 卡空闲、宿主 swap 已扩到 16 GiB
> 前一档只读调研：[z100-cudagraph-feasibility-20261020.md](z100-cudagraph-feasibility-20261020.md)（当时结论：卡/栈支持，被一行 FP8-emu Dynamo 分支卡死，未真机验证）
> 结论标注：**【实测】**=本窗口运行日志/探针产物可直接证实；**【推断】**=由实测推导
> **更新（同日第二窗口）**：本文的"不建议上"结论已被 [z100-cudagraph-route-a-20261021.md](z100-cudagraph-route-a-20261021.md) 部分推翻——
> 把 prefill 物化路由收进 `custom_op` 并让 LUT 离开计算图后，编译+PIECEWISE 臂的 TTFT 从 3.35 s 回到 1.35 s，
> decode step 仍为 118.4 ms（-7~-9%），长上下文/质量对拍持平。本文保留为"未修复时的基线"。

---

## 0. 结论摘要

1. **图模式不再是死路：三臂全部起得来。** 2026-10-20 的唯一已定位硬阻塞（`gfx906_fp8.py` 的 `if inf_codes.any()`）已由 `0e1475163` 改成静态 dtype 分支；本窗口 `--compilation-config '{"cudagraph_mode":"piecewise","cudagraph_capture_sizes":[1,2,4,8,16]}'` 在**生产配方逐字配置**下编译 182.6 s、捕获 5/5 桶、DFlash drafter 也捕获成功、服务 ready，**一次都没有报 Dynamo/graph break**。**【实测】**

2. **但当前配方下"开图"是净亏，不建议上。** 同窗口 A/B：decode step p50 **130.4 → 118.4 ms（-9.2%）**，而 512-token 请求 TTFT **1.35 → 3.35 s（+148%）**，c=1 输出 **36.6 → 23.9 tok/s（-35%）**、c=4 聚合吞吐 **48–56 → 29.7 tok/s（-40%）**。**【实测】**

3. **TTFT 崩掉的根因不是图，是 compile 模式把 P0 的 prefill fp16 物化路由静默关掉了。** `gfx906_fp16_prefill.py:236` 的 `if torch.compiler.is_compiling(): return None` 让物化路由在编译臂完全不生效；证据：eager 臂日志有 4 条 `first materialization route engaged`（每 worker 1 条），两个 compile 臂 **0 条**。**【实测】**

4. **decode 的 9.2% 里，编译占 4.3%、图占 ~5%。** 仅编译（`cudagraph_mode=none`）臂 step p50 **124.8 ms**（-4.3% vs eager），叠加 PIECEWISE 后 **118.4 ms**（-9.2%）。这与历史 S9b「FULL 相对 PIECEWISE 只多 0.5%」不矛盾——那次比较的是两种图，这次比较的是"有图/无图 + 编译"。**【实测】**

5. **代价不小：宿主内存 +~13 GiB、KV 少 8%、冷启动 99 s → 7.4 min。** 图池只有 0.11 GiB，真正贵的是编译态（AOT 二进制 + 4 个 worker）；编译臂把 16 GiB 新增 swap 全部吃满。**【实测】**

6. **质量没有翻车。** 8 条固定 prompt 的 32-token greedy 输出**逐 token 完全一致**（8/8，token id 级对比），接受长度 7.53 tok/step、accept 7.0 三臂相同。S14d 的"图模式不可跨重启复现"仍是需要尊重的边界，但本次窗口内跨臂/跨启动没有观察到翻转。**【实测】**

一句话：**能开、也测出了数字，但"编译 + 图"这整包在当前配方里换来的 decode 9% 被 prefill 2.5× 的倒退吃掉还倒亏；先把 prefill 物化路由做成 compile-safe，或改试 breakable cudagraph（绕开 compile、保留 eager 快路径），才值得再谈上图。**

---

## 1. 三臂定义（同一棵树、同一窗口、同一套客户端）

| 臂 | 编译 | 图 | `--enforce-eager` | 关键 flag | 端口 | 编译缓存 |
|---|---|---|---|---|---|---|
| **E0** 现网 | 关 | 关 | ✅ | 生产 launcher 原样 | 8000 | `/mnt/game/vllm-cache-host` |
| **E1** 仅编译 | 开（VLLM_COMPILE） | `none` | 去掉 | `--compilation-config '{"cudagraph_mode":"none"}'` | 8002 | `/mnt/game/vllm-cache-graph` |
| **E2** 编译+图 | 开 | `piecewise` | 去掉 | `--compilation-config '{"cudagraph_mode":"piecewise","cudagraph_capture_sizes":[1,2,4,8,16]}' --cudagraph-metrics` | 8001 | `/mnt/game/vllm-cache-graph` |

- 三个 launcher 都由生产脚本派生，只改表中列出的行：
  [s26_graph_launch.sh](../../tools/dcu/s26_graph_launch.sh)（E2）、[s26_e1_compileonly_launch.sh](../../tools/dcu/s26_e1_compileonly_launch.sh)（E1）；E0 直接用 [serve_host_dflash2.sh](../../tools/dcu/serve_host_dflash2.sh)。
- 其余 env 逐字保留：`VLLM_DCU_FP8_W8A16=1`、`VLLM_DCU_FP16_PREFILL=auto`、`VLLM_DCU_FP8_EMU_FASTPATH=1`、`VLLM_DCU_FP8_DECODE_CFG=35`、`VLLM_GFX906_TRITON_ATTN_PREFILL_3D=1`、`VLLM_S24_DYNK_*`、`--disable-custom-all-reduce`。
- 客户端：c=1 用 [s23_probe.py](../../tools/dcu/s23_probe.py)（S20 单请求探针，服务端 `iteration_tokens_total` 算 step）；并发用本窗口新写的 [bench_conc.py](../../.s26/graph/bench_conc.py)（同一 `stream_once`/prompt builder，512-token prompt + 128 output，`temperature=0`）。

---

## 2. 启动：编译、捕获、内存【实测】

| 项 | E0 eager | E1 仅编译 | E2 编译+PIECEWISE |
|---|---|---|---|
| 起服务→ready | **99 s** | 407 s | **444 s** |
| `init engine` | 11.29 s | 281.06 s（编译 **182.28 s**） | 276.84 s（编译 **182.61 s**） |
| Dynamo/graph break | — | **0** | **0** |
| Graph 捕获 | 无 | 跳过 | **5/5 桶，3–4 s**；DFlash speculator 各自捕获 |
| 图池 | 0.0 GiB | 0.0 GiB | **0.11 GiB（actual）/ 0.13 GiB（estimated）** |
| KV cache | **806,753 tok**（3.08×@262144） | 748,634 tok（2.86×） | **742,334 tok（2.83×）** |
| VRAM | 92% | 87% | 87% |
| 宿主（RAM+swap 合计） | 11+10 = **21 GiB** | 15+19 = **34 GiB** | 15+19 = **34 GiB** |

- E2 捕获统计（`cuda_graph.py:123`）显示 decode 批 8 token 命中 `PIECEWISE`（count 24），prefill 批 1064/3208 走 `NONE` —— **图确实在 replay，而且只包 decode 的安全段**（attention/GDN 按 splitting_ops 留在图外，与 §4.2 预期一致）。
- 编译时间比历史（S9b 51 桶约 39 min）低一个量级，是因为只捕获 5 个桶且 AOT 一次编译：`torch.compile took 159.7–161.2 s / worker`，产物存到 `torch_aot_compile/7e8b31dd…`。
- **宿主内存是第一硬约束**：编译臂把 16 GiB 新增 swapfile 吃满（E0 只用 10 GiB），E0 的 84 s vs 编译臂 7.4 min 里也有换页成分。原 4 GiB swap 的旧环境大概率起不来。
- 冷启动缓存：`/mnt/game/vllm-cache-graph` 已落 AOT 产物；同 arm 二次启动应显著更快（未在本窗口验证，记为待验证项）。

---

## 3. 性能 A/B【实测】

### 3.1 c=1（512-token prompt，128 output，greedy）

| 指标 | E0 eager | E1 仅编译 | E2 编译+图 | E2 vs E0 |
|---|---|---|---|---|
| **decode step p50**（服务端） | **130.4 ms** | 124.8 ms | **118.4 ms** | **-9.2%** |
| ITL p50（客户端） | 128.2 / 128.9 / 137.7 | 130.1 | 123.0 / 123.4 | -4~-9% |
| tokens/step | 7.53 | 7.53 | 7.53 | 0 |
| accept len | 7.00 | 7.00 | 7.00 | 0 |
| **TTFT（warm）** | **1.35 s** | 3.37–3.42 s | **3.34–3.36 s** | **+148%** |
| 输出吞吐（含 TTFT 的 wall） | **36.6 tok/s** | 23.3 tok/s | **23.9 tok/s** | **-35%** |

- step p50 的离散度：E0 四跑 136.2/130.4/130.4/123.4（首跑冷）；E2 四跑 **118.3–118.5**（几乎无抖动）。图 replay 顺带把 step 抖动压平了。
- E1 与 E2 的 TTFT 几乎相同（3.37 vs 3.35 s）⇒ **TTFT 的倒退来自编译模式，不来自图捕获**。

### 3.2 c=4 并发（4×512 prompt，128 output）

| 指标 | E0 eager | E1 仅编译 | E2 编译+图 |
|---|---|---|---|
| TTFT p50 | **3.3 / 4.1 s** | 9.1 / 11.2 s | 14.1 / 14.1 s |
| ITL p50 | 197 / 214 ms | 192 / 226 ms | 179.7 ms |
| 聚合输出 | **48.2 / 56.5 tok/s** | 27.5 / 28.9 tok/s | 29.7 / 29.9 tok/s |

c=4 是 prefill 主导：每条 512-token 请求在编译臂要 ~3.35 s，4 条串起来就是 ~14 s TTFT；eager 因为物化路由只要 ~1 s/条。

### 3.3 质量对拍（greedy token id）

- 8 条固定 prompt（代码/翻译/算术/JSON 各型）× 32 token，`temperature=0`、`return_token_ids=true`：
  **E0 与 E2 逐 token 完全一致，8/8**（[parity_eager.json](../../.s26/graph/parity_eager.json) vs [parity_graph.json](../../.s26/graph/parity_graph.json)）。
- 接受长度/每步 token 三臂一致 ⇒ DFlash drafter 的图路径（`DFlashCudaGraphManager`）没有改变投机语义。
- 边界仍然成立：S14d/S14e 证明图模式**跨重启不可复现**，本次只证明"同窗口、跨臂"一致。质量门禁/SLA 仍应走 eager。

---

## 4. 归因：为什么 TTFT 会倒退 2.5×

| 事实 | 证据 |
|---|---|
| compile 模式关闭了 prefill fp16 物化路由 | `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py:236` `if torch.compiler.is_compiling(): return None` |
| eager 臂确实走了物化 | E0 日志 4 条（每 worker 1 条）`first materialization route engaged (role=in_proj_qkvz, M=4096, N=4096, K=5120)` |
| 两个 compile 臂一条都没有 | `grep -c` E1/E2 均为 **0**；启动横幅只打印 role min_m 配置，不打印实际路由 |
| 后果：prefill GEMM 回到 W8A16 tile（~6.5 TF/s），而不是 dequant+rocBLAS（16.5–17.7 TF/s） | P0 复测 [z100-p0-prefill-materialization-result.md](z100-p0-prefill-materialization-result.md) §2：eager 两臂 512-token TTFT 2.34 s（tile）vs 1.08 s（物化） |

`3.35 s`（compile+tile，本窗口）对比 P0 的 `2.34 s`（eager+tile，221021 早窗口）还多约 1 s，这部分残差未隔离（候选：inductor 产出对 prefill 形状不友好、编译态 W8A16 custom_op 的 tile 配置与 eager 不同、或本次换页干扰）。**【推断】**

---

## 5. 静态审计：图路径的残余风险（只读，未改代码）

来自本窗口的只读代码审计（证据行号见各文件）：

1. **`_device_lut` / `fp8_to_fp16_lut` 的 `lru_cache` 在 traced 区被内联**（`gfx906_fp8.py:208`、`:924`、`:1016`；torch `_dynamo/variables/functions.py:2192` 的警告本次运行确实打了 8 条）。256 项 LUT 的构造 + `.to(device)` 会进图、每层每步重算，理论上还会在捕获期引入 pageable H2D。实测没有崩（捕获成功），但**这是本次 decode 只拿到 9% 而不是更多的候选原因之一**；建议把 LUT 变成 buffer 或包成 `custom_op`。
2. `_apply_block_scaled_mm_w8a16` 里 `cfg = ... if A.shape[0] > _GEMV_MAX_M else 0`（`:928`）是符号 M 的 python 分支；vLLM 丢 guard，首次 trace 的臂会被烘焙。当前语义无害（op body 内按真实 M 再分支），但违反本文件自述的"图内无符号 M 分支"不变量。
3. `gfx906_fp16_prefill.py` 的 compile 守卫是**功能缺失**而非风险：它让 P0 的收益在 compile 臂整体消失（§4）。
4. GDN：`gdn_attention_core`、`short_conv` 是 splitting op ⇒ 48 层 GDN core 与 attention 都在图外 eager，这与"图只值 5%"吻合；`layernorm.py:489` 的 `is_compiling()` 还会让 GDN 的 fused RMSNormGated 退化为 fp32 native（潜在额外损耗）。
5. fastpath（`VLLM_DCU_FP8_EMU_FASTPATH=1`）在 compile 下自动失效（`:736/:935/:1021` 的 `not torch.compiler.is_compiling()`），所以 S23 的 host 优化在编译臂是拿不到的——**这也是 E1/E2 对比 E0 时"编译收益只有 4%"的一个混淆项**。

---

## 6. 建议

### 6.1 现在

**不要**把生产切到 compile+PIECEWISE：decode 只 -9%，prefill +148%，c≥4 吞吐 -40%，KV -8%，宿主 +13 GiB，冷启动 7.4 min。现网 `--enforce-eager` 保持。

### 6.2 值得做的三个前置件（按性价比排序）

| # | 动作 | 预期 | 代价 |
|---|---|---|---|
| 1 | **让 prefill fp16 物化在 compile 下重新生效**：把 `gfx906_fp16_prefill` 的 dequant+rocBLAS 收进一个 `torch.library.custom_op`（`mutates_args=("out",)` 或返回新张量），compile 下不再 `return None` | 把 TTFT 从 3.35 s 拉回 ~1.3 s，此时 E2 相对 E0 变成 **decode -9% + prefill 持平**，才是净收益 | 需处理每层复用的 fp16 scratch（显存 +~1 GiB/层级别的 workspace 要复用，不能每层新申请） |
| 2 | **LUT 图常量**：`_device_lut` 改成注册 buffer / custom_op，别让 256 项表每层每步重算 | 可能把 decode 从 -9% 再往前推；去掉捕获期 H2D 隐患 | 小，改 1 个函数 |
| 3 | **试 breakable cudagraph**（`VLLM_USE_BREAKABLE_CUDAGRAPH=1`，不跑 inductor） | 绕开 compile 模式的全部功能缺失（物化/fastpath 保留），只拿图 replay | 未验证路径（RCCL 捕获、DFlash、TP=4），ROCm 上被认为比 compile piecewise 慢；适合先做可行性臂 |

### 6.3 复测口径（下一窗口直接用）

```bash
# E2（本窗口已验证可用）
bash tools/dcu/s26_graph_launch.sh            # 端口 8001，缓存 /mnt/game/vllm-cache-graph
# 判据：ready ≤15 min、5/5 捕获、KV>700k、8 条 greedy 与 eager 逐 token 一致
# 性能：c=1 step p50、c=4 聚合吞吐、512/4k/16k 三档 TTFT
```

宿主侧先确认 `free` 与 `/proc/swaps`：编译臂需要 ≥16 GiB swap，否则启动期换页会把 ready 拖到不可预测。

---

## 7. 产物索引

| 内容 | 路径 |
|---|---|
| E2 启动日志 | `.s26/graph/startup-1021-1334.log` |
| E1 启动日志 | `.s26/graph/e1-startup-1021-1351.log` |
| E0 启动日志 | `.s26/graph/eager-startup-1021-1346.log` |
| c=1 探针（三臂） | `.s26/graph/probe-c1/s20_{graph,e1,eager}_20261021-*.json` |
| 并发探针（三臂） | `.s26/graph/bench/{graph,e1,eager}_c{1,4}_*.json` |
| greedy 对拍 | `.s26/graph/parity_{graph,eager}.json` |
| 客户端脚本 | `.s26/graph/bench_conc.py`、`.s26/graph/parity.py` |
| arm launcher | `tools/dcu/s26_graph_launch.sh`、`tools/dcu/s26_e1_compileonly_launch.sh` |
| 内存快照 | `.s26/graph/mem-eager.txt`、`.s26/graph/mem-e1.txt`、`.s26/graph/pre-state.txt` |

> 窗口结束状态：三个 arm 均已 SIGTERM 优雅退出，4 卡 VRAM 归 0%，RAM/swap 回到窗口前基线（宿主 3.5 GiB + swap 1.9 GiB）；生产 unit 维持窗口开始时的"已停"状态，未做任何驱动/unit 改动。
