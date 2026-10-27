# 开启图模式路线 A 复测（①② 修复后）：编译+PIECEWISE 变成净收益，breakable 路线判死

> 日期：2026-10-21（Asia/Shanghai）第二窗口
> 平台/栈/模型与上一轮相同：4×海光 Z100/Z100L（gfx906）、DTK 26.04 + torch 2.11、HEAD `c53840d03` + 本次改动；
> `Qwen3.8-27B-FP8` + DFlash2 k=7、TP=4、maxlen 262144、MBT 4096、`max_num_seqs=4`、util 0.85。
> 上一轮（未修）：[z100-cudagraph-retest-20261021.md](z100-cudagraph-retest-20261021.md)、只读调研：[z100-cudagraph-feasibility-20261020.md](z100-cudagraph-feasibility-20261020.md)
> 标注：**【实测】**=本窗口日志/探针产物直接证实；**【推断】**=由实测推导。

---

## 0. 三条路线的结论

| # | 路线 | 结果 | 关键数字【实测】 |
|---|---|---|---|
| ① | 把 fp16 物化路由收进 `custom_op`，让 compile 下重新生效 | ✅ **成功**，且是这次质变的来源 | compile+图臂 TTFT（512 tok）**3.35 s → 1.35 s**，与 eager 的 1.38–1.40 s 持平；物化路由在编译臂真的打了 4 条 `first materialization route engaged (role=auto, M=4096…)` |
| ② | 让 fp8→fp16 LUT 不再每层每步进图重算 | ✅ **成功**（作为①的一部分 + 独立 op） | compile 期 `Dynamo detected a call to functools.lru_cache` 警告 **8 → 0**；`torch.compile` 时间 **160–182 s → 77–81 s/worker**；AOT 重编译提示消失 |
| ③ | `VLLM_USE_BREAKABLE_CUDAGRAPH=1` 旁路 | ❌ **判死**（不是慢，是数值坏 + GPU VM fault） | c=1 `tok/step=1.00、accept=0.00`、输出 7.4 tok/s（eager 36.5）；`logprobs` 返回 **NaN/Inf**（HTTP 400）；greedy 文本退化成 "duct Register Register…"；c=4 时 **KERNEL VMFault**，引擎退出，并发请求 500 |

**净结论：路线 A（去 `--enforce-eager` + `cudagraph_mode=piecewise`）在①②修复后已经是"decode 快 7–9%、prefill 持平、c=4 不回归"的形态，值得进入生产前验证；路线 B（breakable）在当前模型/DFlash/GDN 组合上不可用。**

---

## 1. 代码改动（本次，未提交）

### ① 物化路由 compile-safe

问题：`torch.compile` 下 token 数是符号量，路由里 `m = x.numel() // K` 的阈值分支不能 trace；原先的做法是 `is_compiling()` 时直接 `return None`，于是编译臂静默退回 6.5 TF/s 的 W8A16 tile。

做法：把"按真实 M 决定走物化还是 tile"整段塞进一个 opaque `custom_op`，Dynamo 只看到一次调用，运行时 body 在真实 M 上按原策略分支。

- `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py`
  - 新增 `vllm::fp8_emu_mm_block_w8a16_auto`（`_register_fp8_emu_mm_block_w8a16_auto`）+ fake impl；body 转调 `gfx906_fp16_prefill.materialize_or_tile`。
  - `_apply_block_scaled_mm_w8a16`：`is_compiling()` 分支改为调该 op，并传 `_FORCE_TILE_MIN_M`（该 fallback 只在路由已经拒绝时到达，必须走 tile）；eager 路径保持原 fastpath/custom_op 不变。
- `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py`
  - 抽出 `_layout_ok`（trace 安全的 dtype/stride/block 常量检查，eager 与 traced 共用）、`_materialize`（dequant+rocBLAS+bias，原 `_try_prefill` 主体）、`materialize_or_tile`（op body：真实 M 决策）、`_traced_route`（trace 期只做常量检查+调 op）。
  - 模块 docstring 的 contract 从 "Eager-only" 改写为 "Compile-safe (S26)"。

### ② LUT 不再进图

- 新增 `vllm::fp8_emu_lut(ref, dtype_code)`：`ref` 只是 trace anchor（**无 tensor 入参的 custom_op 会被 AOT 常量折叠成 Meta tensor**，本次先踩到再修）；body 命中原来的 `lru_cache`。
- `_lut_for(ref, fp8_dtype)`：trace 时走该 op、eager 仍走 `_device_lut`。
- W8A16 主路径已由①整体收进 auto op（LUT 在 body 里取），W8A8 路径改用 `_lut_for`。

### ③ breakable

只加了一个 arm launcher：[tools/dcu/s26_breakable_launch.sh](../../tools/dcu/s26_breakable_launch.sh)（生产 launcher + `VLLM_USE_BREAKABLE_CUDAGRAPH=1`，port 8003，无代码改动）。

### 测试（本次新增，全部通过）

`tests/kernels/quantization/test_gfx906_fp8_emu.py`：

- `test_prefill_route_auto_op_matches_eager`：op body 在 M=8 走 tile、M=64 走物化，与 eager 路由**逐 bit 相等**（`torch.equal`）。
- `test_prefill_route_traces_under_compile`：`torch.compile(fullgraph=True, dynamic=True)` 下两种 M 都能编译并数值一致。
- `test_lut_op_matches_device_lut`：op 表与 `_device_lut` 逐元素相等，且 compile 后返回的是**真实 CUDA tensor**（防 Meta 折叠回归）。

```
190 passed in 51.17s
```

---

## 2. 启动与数值门禁【实测】

| 项 | E0 eager（现网配方，含本次改动） | E2 编译+PIECEWISE（①②修复后） | E2 旧（未修） |
|---|---|---|---|
| 起服务→ready | **99 s** | **306 s**（init engine 170.3 s，编译 85.2 s） | 444 s（编译 160–182 s） |
| 图捕获 | 无 | **5/5 桶，3 s/worker，pool 0.11 GiB** | 5/5，0.11 GiB |
| Dynamo `lru_cache` 警告 | — | **0**（旧臂 8） | 8 |
| 物化路由 | 4 条 `route engaged` | **4 条 `route engaged (role=auto, M=4096…)`** | **0 条**（路由被 `is_compiling` 关掉） |
| KV cache | 806,753 tok（3.08×） | **786,432 tok（3.00×，-2.5%）** | 742,334（-8.0%） |
| VRAM | 92% | 87% | 87% |
| 宿主 RAM+swap | 23 GiB（12+11） | **26 GiB（11+15）** | 34 GiB（15+19） |
| greedy 8 prompt × 32 tok 对拍 | 基准 | **8/8 逐 token 一致** | 8/8 一致 |

---

## 3. 性能 A/B（同窗口、同客户端、同 prompt 集）【实测】

### 3.1 c=1（512-token prompt / 128 output，greedy）

| 指标 | E0 eager | E2 旧（未修） | **E2 修复后** | 修复后 vs E0 |
|---|---|---|---|---|
| decode step p50 | 127.5 ms（本窗口；上一窗口 130.4） | 118.4 ms | **118.4 ms** | **-7.1%（对上一窗口 -9.2%）** |
| ITL p50 | 130.8 ms | 123.2 ms | **123.1 ms** | -5.9% |
| TTFT（warm） | 1.38–1.40 s | 3.34–3.36 s | **1.35–1.41 s** | **持平** |
| 输出吞吐 | 36.5 tok/s | 23.9 tok/s | **37.9–38.1 tok/s** | **+4%** |
| accept / tok-per-step | 7.00 / 7.53 | 7.00 / 7.53 | 7.00 / 7.53 | 一致 |

### 3.2 c=4 并发（4×512-token prompt / 128 output，4 次重复）

| 指标 | E0 eager | E2 旧（未修） | **E2 修复后** |
|---|---|---|---|
| 聚合输出 tok/s（4 次） | 46.4 / 49.1 / 48.0 / 48.8（中位 **48.4**） | 29.7 / 29.9 | 50.8 / 55.5 / 42.3 / 52.5（中位 **51.6**） |
| TTFT p50 | 4.0–6.6 s | 14.1 s | 3.3–6.3 s |
| ITL p50 | 192–214 ms | 179.7 ms | 186–214 ms |

### 3.3 长上下文（同一 prompt，87,194 token，冷）

| 臂 | TTFT |
|---|---|
| E0 eager | **342.8 s** |
| E2 修复后 | **348.1 s（+1.5%，在噪声内）** |

> 说明：上一轮"16k"实测其实是 87k token（prompt 词数≈token 数×5.3），两臂用同一 prompt 逐字对比，所以这个比较是有效的。
> 连续第二个长上下文请求（前缀缓存命中）TTFT 7.2 s —— 与 eager 的前缀缓存行为一致。

---

## 4. 路线 B（breakable cudagraph）为什么判死【实测】

启动本身很快（170 s，`CompilationMode.NONE`、无 inductor，init engine 21.8 s、5/5 捕获、宿主只 +2 GiB），但一上请求就坏：

| 现象 | 证据 |
|---|---|
| 投机解码完全失效 | `tok/step=1.00`、`accept=0.00`，输出 7.4 tok/s（eager 36.5）；`spec_decode_num_drafts_total` 在涨、accepted 为 0 |
| logits 出现 NaN/Inf | `return_token_ids`/`logprobs=0` 请求 → HTTP 400 `Out of range float values are not JSON compliant` |
| greedy 文本退化 | 同一 prompt 输出 `'duct Register Register Register…'`（eager：`' Paris.\nThe capital of Germany is Berlin…'`） |
| GPU 页错误、引擎死亡 | c=4（batch 32 > max capture 16，走未捕获路径）触发 **`KERNEL VMFault`**，14:17:24 引擎退出；4 路并发请求全 500，服务自动关闭 |

判读【推断】：breakable 的 eager-break 标注只覆盖了 attention/GDN 的一部分；本 fork 的 DFlash speculator + GDN core + mamba 状态在分段捕获里没有正确的断点/状态回写，导致验证步读到过期状态（accept=0）、logits NaN，最终越界访问。要救它需要逐算子补 `@eager_break_during_capture` 并重做 drafter 的图契约，工作量远大于路线 A，且路线 A 已达标，**不建议继续投入**。

---

## 5. 残余风险（上生产前必须补的门禁）

1. **收益幅度**：decode -7~-9%、c=4 +7%、prefill 持平。相对复杂度，收益不算大；是否切生产取决于这 7–9% 值不值 5 min 冷启动与 +3–5 GiB 宿主。
2. **可复现性**：S14d/S14e 的结论未变——图模式跨重启不可复现，质量/SLA 口径必须留在 eager。本窗口只证明了**同窗口跨臂 8/8 一致**，样本小。
3. **质量对拍**：仅 8 条 prompt×32 token，需要扩到 S14b 式的 logprob 分布对比（27B 上编译会实质改变 logits）。
4. **首请求成本**：编译臂启动后第一个 512-token 请求 TTFT 2.6 s（后续 1.38 s），首个长上下文请求还要吃 JIT/捕获剩余成本；生产要预热。
5. **冷启动**：306 s（换配置必编译）。同配置二次启动应命中 AOT 缓存（编译 85 s 那部分应变成秒级）——**未验证**，下次窗口第一件事。
6. **262144 上限与并发**：KV 786k token（3.00×@262144）比 eager 少 2.5%；`max_num_seqs=4` 的 pinned 预算问题与本改动无关。
7. **`max_num_seqs=4` 下的 c=4 数据噪声大**（42.3–55.5），需要 ≥3 个窗口取中位。

---

## 6. 复现

```bash
# E0 现网配方（eager）
bash tools/dcu/serve_host_dflash2.sh                      # :8000

# E2 编译+PIECEWISE（①② 已在树内）
bash tools/dcu/s26_graph_launch.sh                        # :8001，缓存 /mnt/game/vllm-cache-graph

# 客户端
.venv/bin/python tools/dcu/s23_probe.py --base http://127.0.0.1:8001 --repeat 4 --tag e2 --out .s26/graph4/probe-c1
.venv/bin/python .s26/graph/bench_conc.py --base http://127.0.0.1:8001 --concurrency 4 --repeat 4 --tag e2
.venv/bin/python .s26/graph/parity.py --base http://127.0.0.1:8001 --tag e2 --out .s26/graph4

# 单测
.venv/bin/python -m pytest tests/kernels/quantization/test_gfx906_fp8_emu.py -q   # 190 passed

# 回滚 E2：去掉 --compilation-config/--cudagraph-metrics、加回 --enforce-eager 即可（代码改动 eager 路径逐 bit 不变）
```

## 7. 产物索引

| 内容 | 路径 |
|---|---|
| E2 修复后启动/运行日志 | `.s26/graph4/e2-patched-*.log` |
| E0 修复后启动日志 | `.s26/graph4/e0-patched-*.log`、`.s26/graph4/e0b-*.log` |
| breakable 日志（含 VMFault dump） | `.s26/graph3/breakable-*.log` |
| c=1 / c=4 / 长上下文探针 | `.s26/graph4/{probe-c1,bench}/` |
| greedy 对拍 | `.s26/graph4/parity_{e0patched,e2patched}.json` |
| 内存快照 | `.s26/graph4/mem-{e0b,e2patched}.txt` |
| arm launcher | `tools/dcu/s26_graph_launch.sh`、`tools/dcu/s26_e1_compileonly_launch.sh`、`tools/dcu/s26_breakable_launch.sh` |

> 窗口结束状态：E0/E2 均 SIGTERM 优雅退出，4 卡 VRAM 0%，宿主回到 3.7 GiB + swap 2.6 GiB；生产 unit 维持"已停"。
> 代码改动（3 个源文件 + 1 个测试文件）**尚未提交**，等待评审。
