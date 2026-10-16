# 缩短"静默"（根治方向）—— 方案与选项

> 归档时间：2026-10-12
> 执行者：Lead（本轮新增真机测量；未改任何 vllm 源码）
> 背景：见 [`docs/result/z100-19-longthinking-collapse-result.md`](../result/z100-19-longthinking-collapse-result.md)
> 状态：📋 **待用户选择**（每条都已标注"预期收益 / 依据 / 成本 / 风险 / 门禁 / 回滚"）

## 0. 先把问题写成可计算的式子

DSH 的失败判据是 **连续 300 s（`streamIdleTimeoutMs = 3e5`）没有任何"被解析出来的流事件"**（注释型心跳无效，见 §2）。
所以真正要压的是这个量：

```
静默 = 命中缓存后仍需 prefill 的 token 数 / prefill 吞吐 + 退化态附加延迟
```

本轮实测（`:8000`，Qwen3.8-27B-FP8 / TP=4 / eager / MTP k=2 / maxlen 262144）：

| 场景 | TTFT | 吞吐 | 依据 |
|---|---|---|---|
| 32k prompt，**冷**（无缓存） | **157.55 s** | 208 t/s | `bench_cachetestA.json` |
| 32k prompt，**同一前缀第二次** | **8.66 s** | — | `bench_cachetestB.json`（**18×**） |
| 96k prompt，冷 | 599.93 s | 164 t/s | `bench_long96k.json` |
| 41 token prompt，**崩溃态** | **44.03 s** | 1 t/s | 重启前实测 |
| 128 token prompt，正常 | 0.58–1.02 s | — | `bench_probe*.json` |

两条结论：

1. **前缀缓存命中时，静默根本不是问题**（32k → 8.7 s）。冷 prefill 才是。
2. **配置允许的最坏静默与客户端容忍度严重不匹配**：`max-model-len 262144` + DSH `contextWindow 262144` 允许 20 万 token 的 prompt，冷 prefill 约 **20 分钟**，而客户端只容忍 **5 分钟**。这是"突然掉速/超时"的量级来源。

> 换算参考：这台机器冷 prefill ≈ 160–210 t/s ⇒ **prompt ≲ 50k token 时冷 prefill 才稳在 300 s 以内**（≈50k/180）。

## 1. 已排除/已证伪的选项（不要再做）

> 2026-10-12 更新：B1 与 C1 已实测，见 [`../result/z100-21-b1-closed-int8-prefill-ab-result.md`](../result/z100-21-b1-closed-int8-prefill-ab-result.md)

| 选项 | 实测结果 | 结论 |
|---|---|---|
| **B1** conv fallback 去 `.item()` | 真实形状微基准：整层 fallback **1.014 ms**（含 21 µs 的同步）⇒ 64 层 ≈ 65 ms / 8 s chunk ≈ **1%**；端到端用 Triton 路径完全绕开 fallback：8192 33.27 s（vs 32.79–33.57）、32768 161.43 s（vs 157.55–162.53） | **否决，无收益** |
| **C1** INT8 `sdot4` vs FP8 仿真 | 8192: 33.6→**11.0 s（3.0×）**；32768: 160→**75.0 s（2.2×）**；98304: 600→**486 s（1.23×）**；decode@8k 318→221 ms | ✅ **有效，待决策换检查点**（需质量门禁） |
| **A4** torch profiler 取证 | 默认 `with_stack=True` 下四个 worker 卡在 `torch.profiler.stop()`，引擎 10 min+ 无响应、trace 60 MB 未收尾；只能 `pkill -9` | ⛔ 该栈不可用；改轻量配置或 rocprof 需另排期 |
| **B3** spinloop 扩展 | CMake 门禁 `Python_VERSION >= 3.11`（`USE_SABI 3.11`），运行时 Python **3.10.12** ⇒ 强制开启只打印 warning；等价替代（`recheck 5000→500`）A/B 实测**无影响**（5.05 → 5.01 s） | ⛔ 关闭：不可构建 + 无效（见 [z100-22](../result/z100-22-b3-closed-a4-trace-mechanism-result.md)） |
| **A4** torch profiler 取证 | 轻量配置（`with_stack=false`）**可用**：健康/孤儿两窗口 trace 到手；逐步拆解：步长 187 ms（GPU 160–169 ms）、孤儿不改步长、GPU 占空比 46%→91%，GPU 时间 44% 在 `fp8_emu_gemm_block_kernel`、19% 在 NCCL | ✅ 完成；连带修正"孤儿拖慢每步"与"DCU% 极低"两处误读 |


| 选项 | 实测结果 | 结论 |
|---|---|---|
| `--max-num-batched-tokens` 2048 → 8192 | 2048-token prompt：8.14 → 8.31 s；32768-token：162.53 → **162.09 s** | **无收益，撤回**（我上一轮的建议③作废） |
| vLLM `--sse-keep-alive-interval`（注释心跳）| 内容为 `": keep-alive\n\n"`，SSE 合规客户端忽略 | 对 DSH **无效**（只对反代/隧道有效） |
| "空 delta 数据心跳" | pi-ai SDK 要求 `delta.content.length > 0` 才产生 `text_delta` | **无效**；要产生事件就必须注入可见文本，否决 |

## 2. 事实核对（本轮新增，供方案选择用）

1. **DSH 侧**：`streamIdleTimeoutMs = source.streamIdleTimeoutMs ?? 3e5`，可在 `~/.dsh/profiles/web/cordis.patch.yml` 的 `providers.local-vllm` 下覆盖；重试策略 `maxRetries=5`、可重试码 `[EMPTY_RESPONSE,RATE_LIMIT,SERVER,TIMEOUT,TRANSPORT]`。
2. **触发记录**（session-14cd2ccf）：step 4 第一次尝试 300 s 内 **0 个事件** → TIMEOUT；step 6 第一次尝试流了 1782 个 reasoning chunk / 3702 s（2.08 s/chunk）后出现 **≥5 min 无事件** → TIMEOUT。最后 `turn/end = aborted(user)`。
3. **心跳无效**的原因：`toStreamChunks` 对 `text_delta`/`thinking_delta` 无条件 yield，但上游 SDK 对空串不产生这两个事件（见 §1）。
4. **conv fallback 的同步是真的**：`InputBatch.query_start_loc` 建在 `device=device`（`vllm/v1/worker/gpu/input_batch.py:28`），而 `_causal_conv1d_gfx906_fallback` 里是 `int(query_start_loc[seq_idx].item())` ⇒ 每序列每层一次 GPU→CPU 同步。
5. **前缀缓存可用且强**：同前缀第二次 TTFT 157.55 → 8.66 s（§0 表）。

## 3. 方案清单（供选择）

### 方案 A —— 让"最坏静默"落进客户端容忍度（配置与运维，半天内可完成）

| 项 | 改动 | 预期 | 风险 |
|---|---|---|---|
| A1 | DSH `providers.local-vllm.streamIdleTimeoutMs`: `3e5` → `1.2e6`（20 min） | 冷 prefill ≤ ~20 min 都不会被误判；不再"超时→重试→孤儿" | 极低；代价是孤儿最长可活 20 min（用 A3 兜） |
| A2 | DSH `contextWindow`: `262144` → `65536`（或 `98304`）；`maxTokens`: `32768` → `8192` | 冷 prefill 上限从 ~20 min 压到 **~5 min（65k）**；孤儿寿命 ÷4 | 低；代价是长上下文能力下降（DSH 会更早压缩历史） |
| A3 | 服务端孤儿看门狗（host 脚本）：`num_requests_running>0` 且无对应活跃连接 → 告警/重启 | 不掉速窗口从"最长 1 小时"降到分钟级 | 低（只读 `/metrics` + `ss`） |
| A4 | 崩溃态取证开关：测试实例加 `--profiler-config`，复现后 `/start_profile` 抓一条 trace | 把退化态那 2.13 s/step 的构成钉死（目前仍未完全归因） | 低（只在测试实例） |

**验收口径**：真机跑一次 ≥6 轮的 DSH 会话，要求 `llm/retry` 事件 **= 0**；每轮 TTFT 记录（冷/热分开）；`sessionStats.ms-per-token` 回到 130–150。

### 方案 B —— 压缩冷 prefill 与退化态（服务端代码，需门禁，1–2 天）

| 项 | 改动 | 预期 | 门禁 |
|---|---|---|---|
| B1 | `causal_conv1d.py` gfx906 fallback 去 `.item()`：把 `query_start_loc`/`cache_indices`/`has_initial_state` 换成 CPU 元数据（`query_start_loc_np` 已存在）或每次调用只同步一次 | 冷 prefill 提升待测（估计 5–20%）；GPU 不再被逐层排空 | `tests/kernels/mamba/test_causal_conv1d.py` + token 逐位对拍 + 32k/96k TTFT |
| B2 | 打开 `VLLM_GFX906_TRITON_CAUSAL_CONV1D=1`（保留 `..._SYNC=1`）做 A/B | S15 记录：@2048 token prefill 394 vs 424 t/s（fallback 慢 ~7%） | S15 质量对拍（该路径曾改 token 流） |
| B3 | 启用 spinloop 扩展（`VLLM_USE_SPINLOOP_EXT=1` 并编 `vllm.spinloop`） | 消掉 5 s 级停靠（旧日志 max ITL 4.9–5.9 s） | 长 prefill 期间 max inter-chunk gap 分布 |
| B4 | 空闲请求看门狗（服务端进程内）：消费者 N 秒未取走输出 → abort 该请求 | 从根上消灭孤儿（A3 是进程级兜底，这是请求级） | 用 `abort_test.py` 门禁：中断后 `num_requests_running` 应在一个 step 内归零 |

### 方案 C —— 换更快的 prefill 检查点（A/B 后决定，1 天 + 质量门禁）

本机 INT8（`Qwen3.8-27B-Channel-INT8-w8a8` + `sdot4` 自编 GEMM）是你们优化最充分的路径（S11e/S11e2），
而现在线上跑的是 **FP8 仿真内核**（`Selected Gfx906Fp8BlockScaledMMLinearKernel`）。

| 项 | 改动 | 预期 | 门禁 |
|---|---|---|---|
| C1 | 同口径 A/B：FP8 vs INT8 的 8k/32k/96k 冷 prefill t/s（只换模型目录，其余 flags 不变） | 若 INT8 快 1.5–2×，则静默同比缩短；否则维持 FP8 | S15/S18 质量口径对拍 + 逐 token 对比 |

## 4. 我推荐的组合（按批次）

- **批次 1（今晚，低风险，立竿见影）**：A1 + A2 + A3。
  理由：直接消灭"静默 > 客户端容忍"这个数量级不匹配，不需要改 vllm 源码；A3 把最坏窗口兜住。
- **批次 2（诊断，1–2 天）**：A4 取证 + 前缀缓存命中率诊断（真实会话里按步记录 `usage.prompt_tokens_details.cached_tokens`，找出哪些步骤没命中、为什么）。
  理由：已证明命中缓存时 32k 只需 8.7 s ⇒ 这是**上限最高**的一条；先诊断再动手。
- **批次 3（服务端根治，需门禁）**：B1 → B3 → B4；B2/C1 作为 A/B 备选。
  理由：B1 顺带解释并消除"GPU 空转（2% DCU / 55 W）"；B4 让孤儿问题从"运维兜底"变成"引擎自愈"。

## 5. 需要你决策的点

1. 走 **A 批（配置+运维）**，还是直接上 **B 批（改服务端代码）**？
2. 是否接受 **A2**（DSH 上下文窗口 262144 → 65536）这个质量/速度折中？
3. **C1**（INT8 vs FP8 prefill A/B）要不要现在做？它只花一次重启 + 20 分钟测量，可能直接砍掉一半静默。
4. 是否允许我在测试实例上开 `--profiler-config` 做 A4 取证（需要一次重启 + 一次复现）。

## 6. 复现/测量口径（方案落地后共用）

```bash
# 冷/热 prefill 对比（热 = 同前缀第二次）
python3 /tmp/z100diag/bench.py cold 16 32768
python3 /tmp/z100diag/bench.py warm 16 32768
# 静默口径：每个请求的 TTFT + max inter-chunk gap（bench.py 里 ITL max 即近似）
# 引擎侧：/metrics 的 time_to_first_token / inter_token_latency / prefix_cache_hits
# 客户端侧：DSH 会话 jsonl 里的 llm/retry 事件数 + sessionStats.ms-per-token
```
