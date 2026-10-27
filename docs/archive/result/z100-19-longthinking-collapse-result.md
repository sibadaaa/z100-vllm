# 长思考对话吞吐骤降（GPU 空转、~2 t/s）—— 定位结果

> 归档时间：2026-10-12（CST 22:30）
> 执行者：Lead（真机复现 + 外部探针；未改 vllm 源码）
> 复现对象：`z100-vllm` 容器内 `:8000`（Qwen3.8-27B-FP8 / TP=4 / eager / MTP k=2 / MBT=2048 / maxlen=262144）
> 状态：✅ **根因已复现并闭环**（客户端中断 → 孤儿请求 → 全线降速 → 孤儿结束 → 立即恢复）；修复建议见 §5，**尚未落地代码改动**

## 1. 现象

长时间"思考"对话中，`avgpwr` 与 DCU 使用率突然降到极低，平均吞吐掉到 ~2 t/s，且不会自己恢复；
`:8000` 的 `/metrics` 开始超时，引擎日志停更。

## 2. 结论（TL;DR）

1. **不是显卡/驱动故障**：全新进程的 torch 微基准满速（§3.1）。
2. **不是配置本身**：同一份配置冷启动后快 10–35×（§3.2）⇒ 是**运行期状态退化**，重启即恢复。
3. **主因（已闭环复现）：被中断/放弃的流式请求不会在引擎侧 abort**。客户端断开后引擎仍按 `max_tokens` 继续生成
   （实测：无任何客户端流量时 `num_requests_running=1`、`generation_tokens_total` 以 **8.3 tok/s** 增长，持续 9 min+），
   期间**其它所有请求** TTFT 从 0.6–1.0 s 涨到 **5.5–6.8 s**、ITL 从 139–181 ms 涨到 **~2000 ms**；
   孤儿请求跑完的瞬间，探针立即回到 0.57 s / 139 ms（§3.3）。
   DSH 的 local-vllm provider `maxTokens=32768`：一个孤儿最长可占住引擎约 1 小时。
4. **放大器（同一现象的"GPU 空转"侧证据）**：长对话每轮都要 prefill 全历史，而 gfx906 的 GDN conv1d
   prefill 走 `_causal_conv1d_gfx906_fallback`（逐序列 Python 循环 + `.item()` 逐层同步），四个 worker 主线程
   全部卡在该函数（§3.4）⇒ **GPU 2% / 50–60 W 空转，引擎在 host 上爬**；同时 MBT=2048 使一次 96k prefill
   独占引擎 **600 s**，期间其它请求完全拿不到服务（§3.4）。
5. 另有可观测的 **~5 s 级停顿**：`SHM_READER_RECHECK_INTERVAL_MS = 5000`
   （`vllm/distributed/device_communicators/shm_broadcast.py:67`）——唤醒通知丢失时读端最多停 5 s。
   旧测速日志里的 `max ITL 4938/5322/5864 ms` 与该常量吻合。

## 3. 原始证据

### 3.1 硬件排除（z100-s18 容器，host 侧 4 卡空闲时）

```
matmul 512^3 fp16 x50 : 0.002s (0.04 ms/iter)   # dev0；dev1/2/3 = 0.05/0.07/0.09 ms
empty add_ + sync x50 : 0.107s (2.15 ms/iter)
P2P: can_device_access_peer 12/12 YES；跨卡 copy 28.16–28.24 GB/s；本卡 D2D 396–401 GB/s
```

### 3.2 崩溃态 vs 同配置冷启动

| 指标 | 崩溃态（重启前，实例 C） | 同配置冷启动后（本次） |
|---|---|---|
| 41 token prompt TTFT（非流式，`max_tokens=1`） | **44.03 s** | 0.57–1.05 s |
| 128 token prompt TTFT / decode | — | 1.02 s / 6.33 t/s，ITL mean 180.6 ms（min 1.15 ms） |
| 逐 token ITL | **2124/2129/2127/2127/2126/2127/2131/2128/2129 ms（±0.2%）** | 139–181 ms |
| `hy-smi` | 4 卡 **DCU% 2%**、**50–60 W**，20 Hz 采样只有零星 50 ms 满占用 | 有请求时 100%/(65–177 W) |
| `/metrics` | **超时（12 s 无响应）** | 200，1.8 s，59 KB |
| 探针（host → :8000） | `client.py` 32 token 用 70.6 s | 见上 |

### 3.3 主因闭环复现（本次实例，未改配置）

```bash
# 1) 中断：发长请求，读到 20 个 chunk 后直接丢弃连接（max_tokens=4096）
python3 /tmp/z100diag/abort_test.py
# 2) 之后每 10 s 探一个 max_tokens=4 的小请求
```
```
dropped after 20 chunks / 13.1s
[probe t+0s]  ttft=5.63s tokens=3 total=6.13s
[probe t+10s] ttft=5.60s tokens=3 total=6.11s
[probe t+20s] ttft=5.51s tokens=3 total=6.02s
[probe t+30s] ttft=5.65s tokens=3 total=6.16s
[probe t+40s] ttft=5.79s tokens=3 total=6.31s
[probe t+50s] ttft=5.69s tokens=3 total=6.21s
```
引擎侧（无任何客户端流量，仅 `curl /metrics` 计数）：
```
generation_tokens_total: 3461 -> 3646  (+185 in 22.2s = 8.3 tok/s)   num_requests_running = 1
Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 7.5-8.6 tokens/s,
            Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.0-1.3%
```
孤儿请求跑完（22:28:30 起 `Running: 0`）后，同一探针：
```
recovered probe: ttft=0.58s tokens=4 ITL mean=139ms min=1.30ms max=209ms
recovered probe: ttft=0.57s tokens=4 ITL mean=139ms min=1.24ms max=209ms
```
对照（同实例、孤儿存在前）：`ttft=1.02s ITL mean=180.6ms min=1.15ms decode=6.33 t/s`。
日志中**没有**任何 abort/disconnect 记录（唯一命中是路由名 `/v1/responses/{response_id}/cancel`）。

### 3.4 "GPU 空转"侧的原始证据（长 prefill + conv1d fallback）

96k prompt（压缩前随机文本，命中 23.5% 前缀缓存）：
```
len= 98304 ttft=599.93s out=7 decode=0.72 t/s ITL mean=1615.8ms min=2.23ms max=2213.7ms
len= 32768 ttft=162.53s out=24 decode=1.43 t/s ITL mean=729.2ms min=1.50ms max=769.4ms
```
该 prefill 进行中（14:12 前后）四个 worker 的 py-spy 栈**完全一致**：
```
_causal_conv1d_gfx906_fallback (vllm/model_executor/layers/mamba/ops/causal_conv1d.py:151)
  causal_conv1d_fn (…/causal_conv1d.py:830)
  _forward_core (vllm/model_executor/layers/mamba/gdn_linear_attn.py:834)
  gdn_attention_core (vllm/model_executor/models/qwen3_next.py:1968)
```
即 `start = int(query_start_loc[seq_idx].item())` 这类**逐序列 GPU→CPU 同步**在热路径上（同文件模块头注释亦自述
"prefill 走 fallback、无 per-layer `.item()` 的 Triton 路径默认关闭"）：`VLLM_GFX906_TRITON_CAUSAL_CONV1D` 默认 0。
期间 `hy-smi`：`0:100% 1:100% 2:0% 3:100%`、116–177 W（prefill 段）；而崩溃态实例空闲时是 2%/50–60 W。
同一 prefill 期间发的 `max_tokens=1` 小请求 **240 s 未完成**（`timeout 240` 被杀）⇒ 长 prefill 独占引擎。

### 3.5 用户侧会话统计（DSH `/home/lyh/.dsh/storages/session_projcache/sessions/`）

| 会话（UTC） | steps | 输出 token | decode ms | ms/token | ttft 合计 |
|---|---|---|---|---|---|
| 10-12 13:55（崩溃期） | 6 | 10,195 | 6,185,597 | **607** | 956,257 ms（≈159 s/step） |
| 10-12 02:47 | 44 | 17,441 | 2,546,014 | 146 | 353,051 ms |
| 10-12 01:20 | 24 | 1,977 | 262,846 | 133 | 67,458 ms |
| 10-11 15:27 | 3 | 437 | 60,310 | 138 | 74,958 ms |

正常期 local-vllm 约 **133–146 ms/token**；崩溃期 **607 ms/token ≈ 1.6 t/s**（用户口述"2 t/s"）。
该会话 token 统计：`uncachedInput 69,828 / output 10,195 / cacheRead 193,024`，上下文 56.5k（窗口 262,144）。

### 3.6 宿主环境（放大器）

- CPU 仅 **6 核 12 线程（Ryzen 5 2600X）**，RAM **31 GiB**；崩溃态实测 4 worker + API server 合计 **~3.5 核纯忙等**
  （`sched_yield`/SHM 自旋、`AMDKFD_IOC_WAIT_EVENTS`），`/proc/loadavg` ≈ 4.2–5.0。
- `dmesg`：10-11 有两次 `Out of memory: Killed process (VLLM::Worker_TP)`（当时 swap 4 GiB 已满）；
  实测崩溃态 vLLM 进程组 `VmSwap` 合计 ≈ 2.8 GiB，`free` 仅 ~0.3–0.5 GiB；换页计数在请求期间**没有增长**（已排除"换页抖动"作为直接原因）。
- 容器内累积 **284 个僵尸 worker**（历次 OOM/重启残留）。

## 4. 机制

```
长时间思考轮（DSH maxTokens 最长 32768）
  ├─ 每轮 prefill 全历史（20k–100k token）→ MBT=2048 分块，单步最长 600 s，期间引擎不服务其它请求
  │    └─ 分块 prefill 的 GDN conv1d 走 host-bound fallback（逐层 .item()）→ GPU 2%/55 W 空转
  ├─ 客户端超时/停止/断线（浏览器、agent 客户端读超时）
  │    └─ vLLM 不在引擎侧 abort → 孤儿请求继续生成到 max_tokens（32768 token ≈ 1 h）
  │         └─ 之后每个新请求都被孤儿拖慢：TTFT 0.6→5.7 s，ITL 0.14→2.0 s
  └─ SHM 读端 recheck=5 s：引擎节奏被打乱后偶发丢唤醒 → 4.9–5.9 s 停顿
⇒ 表现为"算力突然下降、avgpwr/DCU 极低、吞吐暴降到 ~2 t/s"，且重启前不会自愈
```

## 5. 建议（按性价比排序，均未落地）

**P0 立刻可做**
1. **别让轮次变成孤儿**：DSH local-vllm provider 的 `maxTokens` 从 32768 降到 8192，并把客户端读超时放宽到 ≥ 单轮最长耗时的 2–3 倍（当前是"长思考 → 超时断线 → 孤儿 → 更慢 → 更容易超时"的正反馈）。
2. 出事后**重启引擎**即恢复（已验证）；若不想重启，等孤儿自然结束（按 `max_tokens`/实测 8 tok/s 估算）。
3. 运维看门狗：轮询 `/metrics`，若 `num_requests_running > 0` 且 host 侧 `ss` 已无对应 HTTP 连接，判为孤儿并重启/告警。

**P1 引擎侧（需要改代码）**
4. **补 abort-on-disconnect**：流式 chat 的路由在 `StreamingResponse` 侦测到客户端断开时，除 `CancelledError` 外
   还应调用 `engine_client.abort(request_id)`（本仓 `vllm/entrypoints/openai/chat_completion/serving.py:935`
   只 `return self.create_error_response("Client disconnected")`，没有 abort 引擎请求）。
   可用 §3.3 的 `abort_test.py` 做门禁：中断后 `num_requests_running` 应在 1 个 step 内归零。
5. **缩短/消除 host-bound conv1d fallback**：把 `causal_conv1d.py` 的 gfx906 fallback 去 `.item()` 化
   （`query_start_loc`/`has_initial_state`/`cache_indices` 常驻 CPU 或整段向量化），或在质量门禁允许时打开
   `VLLM_GFX906_TRITON_CAUSAL_CONV1D=1` + 保持 `..._SYNC=1`（S15 记录过 token 流副作用，需重跑质量门禁）。
6. **长 prefill 不再饿死其它请求**：`--max-num-batched-tokens` 2048 → 8192（本仓文档称其为 prefill 吞吐的关键旋钮），
   并确认 chunked prefill 下 decode 优先不被整段 prefill 挤掉。
7. **SHM 唤醒**：启用 `VLLM_USE_SPINLOOP_EXT=1` 并编出 `vllm.spinloop` 扩展（当前 `SPINLOOP_EXT_ENABLED=False`），
   或把 `SHM_READER_RECHECK_INTERVAL_MS` 调小，消掉 ~5 s 级停顿。

## 6. 复现方式

```bash
# 0) 引擎（本次复现用；等价于原 :8000 配置，日志落到容器 /tmp/arm_baseline.log）
ARM=baseline MTP=2 FP8_EMU=1 GEMM_MAX_M=1 EAGER=1 bash /tmp/z100diag/serve_arm.sh
# 1) 健康基线
python3 /tmp/z100diag/bench.py probe1 16 128          # ttft≈1.0s, ITL≈181ms
# 2) 中断 → 孤儿 → 全体变慢
python3 /tmp/z100diag/abort_test.py                   # ttft 5.5–6.8s, ITL≈2s
# 3) 观测孤儿
curl -s http://127.0.0.1:8000/metrics | grep -E "num_requests_running|generation_tokens_total"
docker exec z100-vllm tail -f /tmp/arm_baseline.log   # Running: 1 reqs, 8 tok/s, 客户端已消失
# 4) 长 prefill 的 host-bound 取证
python3 /tmp/z100diag/bench.py long96k 16 98304       # ttft≈600s
#    期间在 host 上：sudo py-spy dump --pid <VLLM::Worker_TP pid>   → _causal_conv1d_gfx906_fallback:151
```

## 7. 未做 / 待验证

- 未落任何代码修复；§5 的 P1 项都只有定位证据，没有改后复测数据。
- 未取内核级 trace（本次未以 `--profiler-config`/rocprof 起服务）；崩溃态只有 py-spy 墙钟分布 + hy-smi。
- 未验证 `with_cancellation`（`vllm/entrypoints/serve/utils/api_utils.py:51`）在 chat 路由上是否真的挂上、
  以及断线是否被 starlette 侦测到（本机直接丢弃连接后引擎未 abort 是实测事实）。
- 284 个僵尸进程与宿主内存压力是否参与"崩溃不可自愈"尚未单独消融。

## 8. 参考

- 复现脚本：`/tmp/z100diag/{serve_arm.sh,bench.py,client.py,conc.py,abort_test.py,sample3.sh,threads.py}`
  （容器内启动脚本 `/mnt/game/port/diag/arm_*.sh`，GPU 探针 `/mnt/game/port/diag/gpu_probe.py`）
- `vllm/distributed/device_communicators/shm_broadcast.py:48,58,67`（spinloop / recheck 常量）
- `vllm/model_executor/layers/mamba/ops/causal_conv1d.py:36-56,151,1053`（fallback 与 launch-safety sync）
- `vllm/entrypoints/openai/chat_completion/serving.py:935`（客户端断线只回 error，不 abort 引擎）
- 用户侧：`/home/lyh/.dsh/storages/session_projcache/sessions/*.json`（sessionStats）、
  `/home/lyh/.dsh/profiles/web/cordis.patch.yml`（local-vllm provider，maxTokens 32768）
