# S9b 修 `VllmConfig.pad_for_cudagraph` 缺失、启用 FULL cudagraph 并固化收尾 — 执行结果

> 归档时间：2026-10-09（容器 CST）
> 执行者：子 agent（S9b FULL cudagraph 修复与固化）
> 对应设计文档：[docs/design/z100-09-acceptance-and-bench.md](../design/z100-09-acceptance-and-bench.md)
> 前一档：[z100-09-acceptance-and-bench-result.md](z100-09-acceptance-and-bench-result.md)（S9：A1–A6/A8/A9 过，27B 端到端跑通但只能用 PIECEWISE，decode 380.70 ms/token）
> 状态：🔶 **FULL 已能捕获/起服务/确定性地 replay（27B TP=2 默认 `FULL_AND_PIECEWISE`：PIECEWISE 51/51 + FULL 35/35，≈29–33 min ready，dispatch 统计 `FULL×20`，请求成功；）但 27B 的 greedy 输出与 PIECEWISE **不是逐 token 一致**（3 条里 2 条不同，且两模式各自跨进程完全确定），且 **FULL 对 decode 性能几乎无收益（378.78 / 378.43 vs 380.62 ms/token，≈0.5%）** ⇒ S9 把 3.11× 归因于"缺 FULL 图"是**错误归因**，已用同口径实测证伪；FULL 数值正确性缺独立参照，**生产不建议为此打开 FULL**（§3.9/§4/§6）。GDN(4B) 端到端**首次跑通**（12 min ready、FULL 35/35、请求 200）**

**一句话结论**：`pad_for_cudagraph` 的语义可由上游代码与本 fork 历史**确证**——它是 `CompilationConfig.bs_to_padded_graph_size[batch_size]` 的旧入口，
上游 #28579（本树祖先 `56539cdda`）把这张"batch→图捕获桶"映射搬进 `CudagraphDispatcher` 并在 **build attention metadata 之前**完成 padding，
于是 `CommonAttentionMetadata.num_actual_tokens` 在本树里**已经是捕获桶大小**（`gpu_model_runner.py:2720` ← `:4343` ← `cudagraph_dispatcher.py:139`），旧调用点变成冗余；
本步按"取正确取值来源"路线，把 `gdn_attn.py:348/402` 两处旧调用换成 `m.num_actual_tokens`（**逐位等价**，因为该映射把捕获桶映射到自身），
**没有**往 `VllmConfig` 塞回一个上游已删除的方法。修复后 27B 默认 `FULL_AND_PIECEWISE` 完整跑通（S9 的 `AttributeError` 崩溃消失）。

---

## 1. 目标与验收标准

### 1.1 任务书验收标准（原文）

| # | 判据 |
|---|---|
| 1 | 诊断并修 `VllmConfig` 缺 `pad_for_cudagraph` 导致 FULL 捕获崩溃；语义要能从上游代码或本 fork 历史**明确论证**；修复二选一（补字段并透传 / 改取正确取值来源）；**无法确证语义时不硬猜**，改成显式降级 + 日志 + 结论 |
| 2a | 27B W8A8 INT8、TP=2、**默认 cudagraph 模式（不传 `-cc.cudagraph_mode`）**：FULL + PIECEWISE 捕获日志、服务起来、请求成功、**输出与 PIECEWISE 基线逐 token 一致** |
| 2b | decode ms/token：修复后（FULL 生效）vs S9 的 PIECEWISE 380.70 vs 基线 122.47，给出新数字与"是否回到基线量级"的结论 |
| 2c | 若可行：TP=4 一次 27B 请求 + TTFT@8192（说明口径） |
| 2d | 回归：`s8b_gate.sh` 仍 ALL GREEN；S7 的 0.6B 图捕获仍正常（PIECEWISE 51/51 + FULL 35/35） |
| 3 | GDN(4B) 端到端再试一次，**30 min 硬超时**；不 ready 就写"编译时间问题 + 需要什么" |
| 4a | `.gitignore` 白名单：至少 `tools/dcu/**` 与 `tests/kernels/test_gfx906_packed_silu_fp16.py`；贴 `git status --short` 证据 |
| 4b | 补齐固化与回滚（关键路径 / md5 / 容器清单 / 环境变量 / 退 2.10 / 撤自编 triton / SiLU 回滚件），可直接执行；**不动生产容器 `qwen38-vllm`** |

### 1.2 硬约束遵守情况

| 约束 | 遵守 |
|---|---|
| 禁止执行自编译 HIP 独立可执行程序 | ✅ 本步只跑 python/pytest/`vllm serve`/`vllm bench serve` |
| 不 start/kill/exec/commit 生产容器 `qwen38-vllm` | ✅ 本步**全程未触碰**该容器（它保持 S9 时的 `Exited(137)`；未 start/exec/kill/commit） |
| 4 卡空闲可用、容器 `--memory>=32g` | ✅ 复用 S9 的 `z100-s9`（32g/32g、shm 16g）；TP=2 用卡 2,3，单卡实验用卡 0 |
| 显式 `--dtype float16`、不用 `--enforce-eager`、`VLLM_CACHE_ROOT` 指 `/mnt/game` | ✅ 全部 serve（`VLLM_CACHE_ROOT=/mnt/game/vllm-cache-27b` / `-gdn` / `-zerob`）；**无任何 `--enforce-eager`** |
| 宿主对 `/mnt/game` 只读，写操作经 `docker exec` | ✅ 所有脚本/日志/JSON 都由 `docker exec` 写入 `/mnt/game/port/s9b/` |
| 出现停摆立即 kill 并如实记录 | ✅ 本步无挂死；`/proc/loadavg` 全程 < 3；GDN 用 30 min 硬超时（见 §3.8） |
| 写作用域：`vllm/**`、`.gitignore`、`tests/**`、`tools/dcu/**`、本归档 | ✅ 代码只改 `vllm/v1/attention/backends/gdn_attn.py`；另改 `.gitignore`；本归档；**未改** `docs/design/*`、`docs/z100-todolist.md`、`docs/result/README.md` |
| GDN 30 min 硬超时 | ✅ §3.8 |
| 生产容器由用户自管；回滚演练只对影子容器 | ✅ 回滚演练在 `z100-s9` 的 canonical 树副本上做（§3.10） |

---

## 2. 实际改动

### 2.1 变更总表

| 文件 | 位置 | 改动摘要 | 类别 |
|---|---|---|---|
| `vllm/v1/attention/backends/gdn_attn.py` | `:348-357`（spec-decode FULL 分支）、`:401-406`（非 spec FULL 分支） | 删除 `num_actual_tokens = self.vllm_config.pad_for_cudagraph(m.num_actual_tokens)`，改为 `num_actual_tokens = m.num_actual_tokens`（值不变，见 §2.2 的逐位等价论证）；补注释指向真正的取值来源；`batch_size = m.num_reqs` 保持原样 | **本步核心修复** |
| `.gitignore` | 末尾新增 5 行 | 白名单 `!tools/dcu/**`（并随后重新忽略 `tools/dcu/__pycache__/`）、`!tests/kernels/test_gfx906_packed_silu_fp16.py`、`!tests/platforms/test_hygon_dtk.py` | 固化（硬约束 14/19） |
| `docs/result/z100-09b-full-cudagraph-result.md` | 新增 | 本归档 | 归档 |

`vllm/**` 只有 1 个文件、1 处语义改动（两行代码），md5：

```
# 修复前（= HEAD，canonical 树 S9 状态）
1de9e9d91a03968cd079153b33d41f09  vllm/v1/attention/backends/gdn_attn.py
# 修复后（= /mnt/game/port/gfx906-vllm-211 运行面，两者 md5 一致）
b17a394e2d17d5caeae2ddce595bbae7  vllm/v1/attention/backends/gdn_attn.py
```

### 2.2 `pad_for_cudagraph` 的语义论证（为什么这样修，而不是往 config 里补字段）

**(a) 它原本是什么。** 修复前的 fork 历史里定义如下（`git show e8ae2d129^:vllm/config/vllm.py`，`:425-430`）：

```python
    def pad_for_cudagraph(self, batch_size: int) -> int:
        # if batch_size > self.compilation_config.max_cudagraph_capture_size,
        # it should raise an IndexError.
        # the caller should make sure the batch_size is within the range,
        # i.e., batch_size <= self.compilation_config.max_cudagraph_capture_size
        return self.compilation_config.bs_to_padded_graph_size[batch_size]
```

即"把 token 数向上对齐到最近的 CUDA graph 捕获桶"。它**不是** GDN 专用语义，与 GDN 后端无关。

**(b) 为什么本树里没有了。** 上游 #28579（本树祖先 `56539cdda`，*"[Core] Refactor padding logic and pad for CUDA graphs before attention metadata building"*）
把 `bs_to_padded_graph_size` 从 `CompilationConfig` 搬进 `CudagraphDispatcher`，并把 padding 提前到**构造 attention metadata 之前**；
上游 main（本地检出 `/mnt/game/port/upstream-vllm`，`a26e8dc`）里该符号**已经完全消失**：

```
$ grep -rn "pad_for_cudagraph" /mnt/game/port/upstream-vllm/vllm/   # 无输出（rc=1）
$ sed -n 325,326p /mnt/game/port/upstream-vllm/vllm/v1/attention/backends/gdn_attn.py
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens
$ sed -n 139p /mnt/game/port/upstream-vllm/vllm/v1/cudagraph_dispatcher.py
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]
```

**(c) 本树今天的真实取值来源（逐条可查）。**

| 环节 | 位置 | 内容 |
|---|---|---|
| dispatch 时按捕获桶 padding | `vllm/v1/cudagraph_dispatcher.py:71-90`、`:130-154`、`:139` | `_compute_bs_to_padded_graph_size()` / `_create_padded_batch_descriptor()`：`num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]` |
| 模型 runner 采用 padding 后的值 | `vllm/v1/worker/gpu_model_runner.py:4343` | `num_tokens_padded = batch_descriptor.num_tokens` |
| 该值直接进入 `CommonAttentionMetadata` | `vllm/v1/worker/gpu_model_runner.py:2714-2720` | `query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1]` … `num_actual_tokens=num_tokens_padded` |
| attention metadata builder 拿到的是已 padding 的值 | 上游注释 + 本树 `gdn_attn.py` | 上游明写 `m.num_actual_tokens is already padded by the model runner for CUDAGraph` |
| 旧方法在 sync 里被删 | `git show e8ae2d129 -- vllm/config/vllm.py` | 删除 `def pad_for_cudagraph`（`-` 行），同时把 `vllm/config/vllm.py` 换成上游 b44895cf 版本；`gdn_attn.py` 却保留了旧调用点（fork 侧 GDN 实现） |

**(d) 为什么"改成 `m.num_actual_tokens`"是逐位等价、不是改语义。** 只有 FULL 模式会进这两个分支（`use_full_cuda_graph = cudagraph_mode.has_full_cudagraphs()`，`compilation.py:90-91`；PIECEWISE 下恒 False），
而 FULL 只会在 `batch_descriptor`（其 `num_tokens` 已由 `_bs_to_padded_graph_size` 映射得到）命中 FULL key 时被 dispatch；
FULL key 由 `cudagraph_capture_sizes` 构造（`cudagraph_dispatcher.py:216-229`），而该映射把每个捕获桶映射到自身（`:82-90`，`bs == start → start`）。
⇒ **凡是能走到这两行的调用，`pad_for_cudagraph(m.num_actual_tokens) == m.num_actual_tokens`**；旧方法若还在，返回值与我的新写法完全一致。
同一论证也覆盖 capture 路径：capture 的 `num_tokens` 取自 `get_capture_descs()`（`gpu_model_runner.py:7014`/`7222`），即捕获桶本身。

**(e) 为什么没有选"把字段补回 `VllmConfig`"。** 那会让本树与上游 main 的 API 继续分叉，且必须把 `VllmConfig` 反向依赖到 `CudagraphDispatcher` 的内部状态
（`bs_to_padded_graph_size` 现在是 dispatcher 的**私有**属性、`CompilationConfig` 里已无此字段，`grep` 全树只有 `cudagraph_dispatcher.py` 命中）；
此外该调用点在语义上**本就冗余**（padding 已提前完成）。按任务书"取正确取值来源"路线处理。

**(f) 回归锁。** 本改动不需要新测试：仓库已有 `tests/v1/attention/test_gdn_metadata_builder.py` 里的两个 FULL 用例
（`test_full_cudagraph_spec_metadata_uses_request_count`、`test_cudagraph_capture_batch_stays_decode_only`）正好走这两行；
修复前它们在 `pad_for_cudagraph` 处抛 `AttributeError`，修复后通过（§3.10 回滚演练给出前后对照）。

---

## 3. 执行过程与原始证据

### 3.0 环境与前置门禁

修复先落到工作区 `/home/lyh/z100-vllm`，再 `docker exec` 同步进 canonical 运行树 `/mnt/game/port/gfx906-vllm-211`（`/src` 只读挂载）：

```
$ docker exec z100-s9 bash -c 'cp /src/vllm/v1/attention/backends/gdn_attn.py $V/vllm/v1/attention/backends/gdn_attn.py'
$ grep -n "pad_for_cudagraph" $V/vllm/v1/attention/backends/gdn_attn.py || echo NO_PAD_REF_LEFT
NO_PAD_REF_LEFT
b17a394e2d17d5caeae2ddce595bbae7  /mnt/game/port/gfx906-vllm-211/vllm/v1/attention/backends/gdn_attn.py
b17a394e2d17d5caeae2ddce595bbae7  /src/vllm/v1/attention/backends/gdn_attn.py
```

`tools/dcu/s8b_gate.sh`（S8b 门禁，15 项）在修复后的树上复跑：

```
$ docker exec z100-s9 bash -lc 'source /opt/dtk/env.sh; export VLLM_CACHE_ROOT=/mnt/game/vllm-cache; cd /mnt/game/port/gfx906-vllm-211
    python -c "import torch; import vllm; print(\"import ok\", vllm.__file__)"
    bash tools/dcu/s8b_gate.sh > /mnt/game/port/s9b/s9b-gate.log 2>&1; echo "GATE_RC=$?"'
import ok /mnt/game/port/gfx906-vllm-211/vllm/__init__.py
GATE_RC=0
$ grep -E "CHECK_SUMMARY|ALL GREEN" /mnt/game/port/s9b/s9b-gate.log
CHECK_SUMMARY pass=15 fail=0
  RESULT: ALL GREEN
$ grep -E "md5|reference" /mnt/game/port/s9b/s9b-gate.log | tail -2
  md5 93bf7b4a391ac97e2f67cc7141633cb1  size 40691216
  S7c-fixed reference: 93bf7b4a391ac97e2f67cc714163cb1 40691216
```

### 3.1 27B W8A8 INT8 / TP=2 / **默认 cudagraph 模式**：FULL 捕获成功（S9 的崩溃消失）

命令（`/mnt/game/port/s9b/run_27b_tp2_full.sh`，卡 2,3，:8003，**没有 `-cc.cudagraph_mode`**，`--max-model-len 8192`）：

```
python -c "import torch;from vllm.entrypoints.cli.main import main;main()" serve \
  /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 \
  --dtype float16 --tensor-parallel-size 2 --max-model-len 8192 \
  --max-num-batched-tokens 2048 --gpu-memory-utilization 0.9 \
  --disable-custom-all-reduce --skip-mm-profiling \
  --enable-prefix-caching --mamba-cache-mode align \
  --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8003
```

原始日志（`/mnt/game/port/s9b/27b-tp2-full.log`；进度条带 `\r`，此处按行拆开）：

```
torch.compile took 155.57 s in total
Capturing CUDA graphs (PIECEWISE): 100%|██████████| 51/51 [12:01<00:00, 14.15s/it]   <- 显存 profile 轮
Capturing CUDA graphs (FULL):     100%|██████████|  2/2 [00:01<00:00,  1.07it/s]     <- 显存 profile 轮
Graph capturing finished in 725 secs, took 1.66 GiB
Available KV cache memory: 11.53 GiB
GPU KV cache size: 227,448 tokens, Maximum concurrency for 8,192 tokens per request: 27.76x
Capturing CUDA graphs (PIECEWISE): 100%|██████████| 51/51 [12:00<00:00,  1.59s/it]   <- 正式捕获轮
Capturing CUDA graphs (FULL):     100%|██████████| 35/35 [00:26<00:00,  1.11it/s]    <- **FULL 35/35 成功**
Graph capturing finished in 749 secs, took 1.47 GiB
INFO: Application startup complete.
```

时间线：`00:11:34` 起服务 → `00:40:50` FULL/PIECEWISE 全部捕获完（**≈29 min**，S9 的 PIECEWISE-only 是 ≈39 min（两轮 51），
本步 PIECEWISE 51+51 之外多了 FULL 35；注意本步 `--max-model-len` 8192 较 S9 的 4096 更大，KV 池 227,448 vs 161,792 tokens）。

崩溃对照（S9 §3.10.1，同一命令去掉 `-cc.cudagraph_mode=PIECEWISE`）：

```
Capturing CUDA graphs (FULL):   0%|          | 0/2 [00:00<?, ?it/s]
Worker_TP1 ERROR AttributeError: 'VllmConfig' object has no attribute 'pad_for_cudagraph'
  File ".../vllm/v1/attention/backends/gdn_attn.py", line 400, in build
```
⇒ 同一配置在本步 **0 次 `AttributeError`**（全日志 `grep -c "AttributeError|Worker failed"` = 0；134 条
`TypeError: Object of type function is not JSON serializable` 是 S9 附录 D-7 记录的 dynamo metrics 无害噪声）。

### 3.2 27B 请求成功（FULL 生效）

```
$ python s9b_req_compare.py 8003 /mnt/game/port/s9b/req-full.json
[chat_hangzhou] prompt=56 completion=64 ids=64 finish=length
    text='用户要求"用一句话介绍杭州"——这是一个非常开放式的提问，需要我给出一个简洁、准确、信息量大的单句。\n\n用户要求...'
[cap_france] prompt=5 completion=24 ids=24 finish=length
    text=' Paris, the largest city in the country. The capital of France is Paris, the largest city in the country. The'
[longer] prompt=17 completion=48 ids=48 finish=length
    text=' a large language model developed by Alibaba Cloud. Its architecture is based on a large language model develo...'
```
文本正常、非 NaN、非 `"lock"` 退化；`return_token_ids=True` 拿到了精确 token id（下游逐 token 对比用）。

### 3.3 运行时确实在 replay FULL 图（不是只"捕获成功"）

`--cudagraph-metrics` 打开后，engine 周期性打印 dispatch 统计。0.6B 对照组（同一 flag 下 FULL / PIECEWISE 各跑一遍，见 §3.6）：

```
# FULL_AND_PIECEWISE（默认）
| 1  | 1  | 0 | FULL      | 155 |     <- 单请求 decode 全部走 FULL 图 replay
| 17 | 24 | 7 | PIECEWISE | 2   |     <- prefill（非 uniform）走 PIECEWISE
# PIECEWISE
| 1  | 1  | 0 | PIECEWISE | 155 |
```
两组 unpadded/padded token 计数逐行相同 ⇒ 两组请求负载等价，唯一差别是 decode 的 runtime mode。
27B 的直接取证：29 号那轮 27B TP=2 FULL **没开** `--cudagraph-metrics`；为不留"只捕获不 replay"的疑问，
本步又补跑了一轮 TP=2 默认模式 + `--cudagraph-metrics`（`/mnt/game/port/s9b/27b-tp2-full-metrics.log`），
结果见 §3.12——`| 1 | 1 | 0 | FULL | … |` 行直接证明 27B 的单请求 decode 也在 replay FULL 图。
（TP=4 那轮因宿主内存 OOM 未跑成，见 §3.7。）

### 3.4 性能对拍：FULL 对 decode **几乎零收益**（S9 的 3.11× 归因被证伪）

口径与 S9/生产基线一致（`vllm bench serve`，`--backend openai-chat`，dataset random，T=0，ignore_eos，seed 0）：

| 场景 | 命令参数 | 生产基线（:8000 TP=4 eager bf16 + 自研 int8_sdot4 + GDN torch 补丁） | S9 PIECEWISE（TP=2, maxlen 4096） | **S9b FULL（TP=2, maxlen 8192）** | **S9b PIECEWISE（TP=2, maxlen 8192）** |
|---|---|---|---|---|---|
| 单流 decode TPOT | in=128 out=128 np=2 warmups=1 conc=1 | **122.47 ms/token** | 380.70 ms/token | **378.78**（第二轮复测 **378.43**；ITL 375.82, TTFT 1284.70 / 1308.05） | **380.62**（ITL 377.65, TTFT 1305.63） |
| 冷 TTFT | in=8000/8180 out=1 chat（S9 用 3800；基线 8192） | 22,850.89 ms @8192 = 2.789 ms/prefill-token | 12,456.53 ms @3800 = 3.278 | **28,066.83 ms @8000 = 3.508** | **28,191.63 ms @8000 = 3.524** |
| 并发 8 | in=512 out=128 np=16 warmups=2 conc=8 | 无历史数据 | 14.78 tok/s(out) / Mean TPOT 433 | **14.92 tok/s(out) / 80.71 total / Mean TTFT 13.51 s / Mean TPOT 428.05** | 14.94 / 80.80 / 14.23 s / 423.81 |

原始摘要（各自日志：`/mnt/game/port/s9b/27b-tp2-full.log`、`27b-tp2-pw.log`）：

```
$ bash bench_s9b.sh 8003 full decode | grep Mean TPOT
Mean TPOT (ms):                          378.78
$ bash bench_s9b.sh 8003 pw decode | grep Mean TPOT
Mean TPOT (ms):                          380.62
```

**结论（重要，与 S9 §3.13-1 相反）**：
1. 打开 FULL 图后 decode = **378.78 ms/token**，相对 S9 的 PIECEWISE 380.70 只快 **0.5%**，相对生产基线 122.47 仍慢 **3.09×**。
   ⇒ "decode 3.11× ≈ 缺 FULL decode cudagraph 的代价"这一归因**不成立**。S9 引用的是 S7 在 0.6B 上的
   `FULL_AND_PIECEWISE`(11.6) vs **`--enforce-eager`**(34.6) —— 那比较的是"编译+图"与"完全不编译"，不是 FULL 与 PIECEWISE。
   本步的 PIECEWISE 已经包含 torch.compile + 分段图，所以再叠一层 FULL 几乎无增量（decode 每步的瓶颈在算子本身，不在图启动开销）。
   同一结论在 conc8（14.92 vs 14.94 tok/s）与 TTFT（3.508 vs 3.524 ms/token）上一致，全部落在噪声内。
2. 因此 3.09× 的欠账要另找原因：生产基线是 **TP=4 + eager + 用户自研 int8_sdot4 GEMV/GEMM + GDN 纯 torch 替换**（`/home/lyh/server/serve_27b_tp4.sh` 的
   `INT8_SDOT4_*` / `GDN_TORCH_DISABLE` / `PHASEA_GDN_MODE` 开关），本树走 **TP=2 + Triton `TritonInt8ScaledMMLinearKernel` + 原生 FLA/GDN**。
   这是 kernel/并行度层面的差异，**不是图模式差异**（本步已用同栈 A/B 钉死）。
3. 27B 的 TTFT@8192 精确基线口径在 TP=4 那轮补（§3.7）；TP=2 这轮受 `--max-model-len 8192` 限制，
   chat 后端会先做 `prompt+max_tokens > max_model_len` 预检（in=8180 都被 400 拒），故降到 in=8000 并给 per-token 口径。

### 3.5 输出一致性：27B **不满足**"与 PIECEWISE 逐 token 一致"；0.6B 满足

PIECEWISE 重跑（同一构建、同一 flag，仅加 `-cc.cudagraph_mode=PIECEWISE` 与 `--cudagraph-metrics`；两轮 PIECEWISE 51/51、
`Graph capturing finished in 721 / 719 secs`、KV 11.59 GiB / 228,412 tokens、`01:27:25` ready）：

```
$ python s9b_cmp.py req-pw-1.json req-pw-2.json req-pw-3.json req-full.json
REQ chat_hangzhou   ids=64
  pw-1 vs pw-2: IDENTICAL (64 ids)     <- PIECEWISE 三次完全确定
  pw-1 vs pw-3: IDENTICAL (64 ids)
  pw-1 vs full: DIFF first at 13 (101729 vs 104118), total diff positions 47
REQ cap_france      ids=24
  pw-1 vs pw-2 / pw-3: IDENTICAL ; pw-1 vs full: IDENTICAL (24 ids)
REQ longer          ids=48
  pw-1 vs pw-2 / pw-3: IDENTICAL ; pw-1 vs full: DIFF first at 0 (3349 vs 264), total diff positions 48
```

分歧位置的 top-k logprobs（PIECEWISE 实测，`logprobs-pw.json`）与 token 名：

```
REQ longer   pos 0: PW 选 ' large'(3349) logprob=-1.3125 ; 次选 '.'=-1.3516（margin 0.039 nats）
                    ' the'=-2.2500, ' a'=-2.4844 ; FULL 选的是 ' a'(264)
REQ chat     pos 13: PW 选 '请求'(101729) -0.5720 ; 次选 '提问'(104118) -1.2280（margin 0.656 nats）
                     FULL 选的是 '提问'
（token 名用模型自带 tokenizer 解码确认：264=' a', 3349=' large', 104118='提问', 101729='请求'）
```

0.6B 对照组（**同栈、同 flag、只差 cudagraph_mode**，5 条 prompt × 32 token，卡 0）：

```
$ python s9b_cmp.py zerob-ids-full.json zerob-ids-pw.json
REQ cap_france / qwen_intro / code / long_en / zh : IDENTICAL (32 ids) ×5     <- FULL == PIECEWISE 逐 token
$ grep -a "Available KV cache memory" zerob-*.log     # 两轮 KV 池也略有差别（14.2 vs 14.04 GiB）
```

**怎么解释这个"27B 不一致、0.6B 一致"**（不掩盖，逐条给证据）：

1. **FULL 图 replay 本身没有错**：0.6B 在 FULL（155 次 FULL replay）与 PIECEWISE（155 次 PIECEWISE replay）下输出**逐 token 相同**，
   而 S7/S7b 已经把 0.6B 的 `VLLM_COMPILE`（含 FULL 图）与 HF transformers 对到逐 token 一致（硬约束 13 的来源）。
2. **27B 的两轮不是"同一份编译产物"**：`cudagraph_mode` 是 `CompilationConfig` 的 hash 因子（`vllm/config/compilation.py:194` 文档，
   `compute_hash()` 把 compilation_config 的 hash 拼进 `VllmConfig.compute_hash()`），实测四条运行落在**四个不同的 torch.compile 缓存目录**：
   `S9 PW@4096 → 018de25520`、`S9 FULL_AND_PIECEWISE@4096(崩) → fda6c2438a`、`S9b FULL@8192 → 0bb72b1d53`、`S9b PW@8192 → 92f2345be5`。
    inductor 的融合/kernel 选择（num_warps、归约切分）随之不同 ⇒ 同一 prompt 的 logits 并非逐位相同。
3. **PIECEWISE 自己就不是"配置不变量"**：S9 归档 §3.10.2 记录的 PIECEWISE@maxlen4096（缓存 `018de25520`）对
   `The capital of France is` 的输出是 `"...country.\nThe capital..."`（prompt_tokens=5、greedy、24 token，与本步同一请求），
   而本步 PIECEWISE@maxlen8192（`92f2345be5`）是 `"...country. The capital..."`。
   两次运行的**代码路径相同**（本修复在 PIECEWISE 下是死分支），实质差异只有 `--max-model-len`（KV 池 161,792 → 228,412 tokens）
   与随之不同的编译缓存目录 ⇒ greedy 续写对"编译产物/缓存几何"敏感，不能拿某一次 PIECEWISE 当逐位基准。
   （S9 侧文本取自归档 §3.10.2 的客户端记录；服务日志本身不落响应文本。本步两次 PIECEWISE 则在同目录留下了
   `req-pw-{1,2,3}.json` 三份逐 token 可复核的记录。）
4. 分歧点的 margin 也说明问题：chat 的第 13 个 token 是 `请求` vs `提问`（margin 仅 0.66 nats）；`longer` 的第 0 个 token
   分布本身极其平坦（-1.31 / -1.35 / -2.25 / -2.48，模型在该续写上不确定）。

**判定**：本步**不满足**"27B 输出与 PIECEWISE 逐 token 一致"这条验收；但现有证据**不足以判定 FULL replay 算错**——
反而支持"不同 cudagraph_mode ⇒ 不同编译产物 ⇒ 近 tie 的 greedy 选择翻转"。缺少的是**独立参照**（27B 的 HF/fp16 逐 token 对照），
因此按任务书"静默算错比慢更不可接受"的要求，**给出保守建议而非宣称正确**（§4/§6）。

### 3.6 回归：0.6B 图捕获仍正常（PIECEWISE 51/51 + FULL 35/35）

```
# FULL_AND_PIECEWISE（默认）
ready_after=52s
Capturing CUDA graphs (PIECEWISE): 100%|██████████| 51/51 [00:04<00:00, 16.23it/s]
Capturing CUDA graphs (FULL):     100%|██████████| 35/35 [00:02<00:00, 16.51it/s]
Available KV cache memory: 14.2 GiB ; Graph capturing finished in 7 secs, took 0.36 GiB ; Application startup complete.
# PIECEWISE
ready_after=64s
Capturing CUDA graphs (PIECEWISE): 100%|██████████| 51/51 [00:03<00:00, 15.37it/s]
Available KV cache memory: 14.04 GiB ; Graph capturing finished in 5 secs, took 0.26 GiB
```
⇒ 与 S7 §3.3（51/51 + 35/35）一致，**FULL/PIECEWISE 捕获与 replay 均正常**；两组 5 条 prompt 输出逐 token 相同（§3.5）。

### 3.7 TP=4 + TTFT@8192（27B，默认 cudagraph 模式）→ **宿主内存受限，未跑成（如实记录）**

命令 `/mnt/game/port/s9b/run_27b_tp4_full.sh`：TP=4、`--max-model-len 16384`（让基线口径 chat `in=8192 out=1` 能过服务端预检）、
默认 `FULL_AND_PIECEWISE`、`--cudagraph-metrics`、卡 0–3、:8005。

原始结果（`/mnt/game/port/s9b/27b-tp4-full.log`）：

```
INFO 10-09 02:07:30   <- 启动
(Worker_TP0 pid=55553) INFO 10-09 02:09:18 [backends.py:1151] Dynamo bytecode transform time: 32.07 s
(APIServer pid=55474) ERROR 10-09 02:11:35 [multiproc_executor.py:314] Worker proc VllmWorker-0 died unexpectedly (exit code: None), shutting down executor.
(APIServer pid=55474)     raise RuntimeError("cancelled")
(APIServer pid=55474) RuntimeError: cancelled
```

根因 = **cgroup OOM**（非代码问题）：

```
$ docker exec z100-s9 bash -c 'grep oom_kill /sys/fs/cgroup/memory.events; cat /sys/fs/cgroup/memory.max'
oom_kill 1
34359738368              # z100-s9 = --memory=32g（S9 建容器时按"容器 --memory>=32g"设置）
$ grep MemTotal /proc/meminfo
MemTotal:       32777216 kB      # 宿主总共只有 30.5 GiB 可用 RAM
$ docker inspect qwen38-vllm --format 'MemLimit={{.HostConfig.Memory}}'   # 只读，未启停
MemLimit=0                       # 生产容器**没有** cgroup 内存上限 —— baseline 的 TP=4 27B 就是这么跑的
```

⇒ 27B TP=4 在"30.5 GiB 物理内存 + 32 GiB cgroup 上限"下**不可行**：4 个 rank 各持一份权重分片与 torch/inductor 运行时，
峰值超过 32 GiB（启动后 4 min 05 s、在 Dynamo 之后、进图捕获之前被杀）。**未重试**：把 `--memory` 放开会让**宿主整体** OOM，
可能连累其它 agent 的容器（`z100-build`/`z100-triton` 等都在同一台机器上）——该风险不可接受。

**TTFT@8192 的口径与可得数字**：

| 口径 | 环境 | 数字 |
|---|---|---|
| 基线（生产） | chat `in=8192 out=1`、np=1、warmups=0、conc=1；容器无内存上限；TP=4、maxlen=262144 | **22,850.89 ms** = 2.789 ms/prefill-token |
| 本步 FULL | TP=2、maxlen=8192；`in=8000`（`in=8180` 已被预检 400 拒：`This model's maximum context length is 8192 tokens … your prompt contains at least 8192 input tokens`） | 28,066.83 ms = **3.508 ms/prefill-token** |
| 本步 PIECEWISE | 同上 | 28,191.63 ms = **3.524 ms/prefill-token** |

⇒ 按每 prefill token 比，本步比基线慢 **+25.8%（FULL）/+26.3%（PIECEWISE）**（prefill 在两模式下都走 PIECEWISE，两组差 0.5% 是噪声）；
与 S9 的"+17.5%"同源（缺生产侧 GDN prefill 加速 + TP 数不同）。
**要拿到严格的 TTFT@8192**：需在 `--memory=0`（或 ≥48 GiB RAM）的环境里用 `--max-model-len ≥ 8193` 重跑，
脚本已备好（`run_27b_tp4_full.sh` + `bench_s9b.sh <port> <label> ttft8192`）。S9 已在 Qwen3-0.6B 上验证过 27B-无关的 TP=4 可用性（S9 §3.5）。

### 3.8 GDN（Qwen3.5-4B）端到端：30 min 硬超时重试 → **首次跑通**

命令（`/mnt/game/port/s9b/run_gdn_full.sh`，卡 0，:8004，**默认 cudagraph 模式**，`--max-model-len 4096 --gpu-memory-utilization 0.6`，
脚本内 `for i in $(seq 1 1800)` 轮询 ready，超时即 `kill -9` 并打印卡点）：

```
$ docker exec -d z100-s9 bash -c 'bash /mnt/game/port/s9b/run_gdn_full.sh > /mnt/game/port/s9b/gdn-full-run.log 2>&1'
$ cat /mnt/game/port/s9b/gdn-full-run.log
GDN_READY=1 ELAPSED=729s                      <- 12 min 09 s 就绪（S9 三次都没到过 ready）
...(APIServer) Found incompatible backend(s) [TURBOQUANT] ... Overriding with TRITON_ATTN out of potential backends: ['TRITON_ATTN', 'ROCM_ATTN'].
(Worker_TP0) Dynamo bytecode transform time: 7.44 s
Capturing CUDA graphs (PIECEWISE): 100%|██████████| 51/51 [02:29<00:00,  2.93s/it]   <- profile 轮
Capturing CUDA graphs (FULL):     100%|██████████|  2/2 [00:01<00:00,  1.57it/s]     <- profile 轮
Graph capturing finished in 151 secs, took 0.82 GiB
Available KV cache memory: 6.97 GiB
GPU KV cache size: 122,880 tokens, Maximum concurrency for 4,096 tokens per request: 30.00x
Capturing CUDA graphs (PIECEWISE): 100%|██████████| 51/51 [02:24<00:00,  2.84s/it]   <- 正式捕获轮
Capturing CUDA graphs (FULL):     100%|██████████| 35/35 [00:14<00:00,  2.35it/s]    <- **GDN 的 FULL 35/35（本修复生效点）**
Graph capturing finished in 161 secs, took 0.90 GiB
INFO: Application startup complete.
HTTP 200 elapsed=2.04s
usage = {"prompt_tokens": 5, "total_tokens": 29, "completion_tokens": 24, ...}
text  = " Paris.\nA. True\nB. False\nAnswer:\nA\n\nWhich of the following is NOT a characteristic"
--- killing serve pid=48265 ---
GDN_FULL_DONE
```

⇒ **GDN(4B) 端到端首次跑通**：`TRITON_ATTN`、PIECEWISE 51/51 + **FULL 35/35**（正式捕获轮）、
KV 6.97 GiB / 122,880 tokens、HTTP 200、24 token 文本正常（非 NaN、非 `lock`）。

与 S9 的对照（S9 §3.12）：

| 轮次 | 图模式 | 等待上限 | 结果 |
|---|---|---|---|
| S9-1 | `PIECEWISE` | 420 s | 未就绪（被 kill） |
| S9-2 | `PIECEWISE` | 1500 s | 25 min 仍未就绪 |
| S9-3 | `cudagraph_mode=NONE` | 900 s | 15 min 仍未就绪（卡在 inductor 编译） |
| **S9b** | **默认 `FULL_AND_PIECEWISE`** | **1800 s（未触发）** | **729 s 就绪 + FULL 35/35 + 请求成功** |

**为什么这次快（如实归因，不夸大本修复的作用）**：
1. 本修复**只影响 FULL 路径**（`use_full_cuda_graph` 为 False 时两个分支是死代码，`compilation.py:90-91`）；S9 的三次都跑 PIECEWISE 或 NONE，
   所以"S9 卡在 inductor 编译"与本修复无关。唯一的直接增益是：**GDN 现在能捕获 FULL 35/35**（S9 只要一开 FULL 就崩）。
2. 本次的 `torch.compile` 缓存 `183d04da1e` 是**本次运行新建的**（`ls -la` mtime = `10-09 01:54`；S9 的两个目录是 `ec26fce339`@`10-08 23:47` 与
   `ff168afc98`@`10-08 23:04`），所以这次**不是**靠热缓存省的：冷编译 + 两轮捕获合计 12 min。
   最可能的原因是宿主机/容器当时的 CPU 争用与缓存状态（S9 是在 27B 两轮长跑之后紧接着跑 GDN，其日志显示 100% CPU、RSS 6.6 GiB 却停在同一行），
   而不是任何一行代码差异。
3. **给 S10/生产的可直接执行建议**（S9 §3.12 的方向已不必要）：
   - 不需要 `-cc.inductor_compile_config.benchmark_combo_kernel=False`，也不需要降图捕获规模；
   - 需要的是"**空闲宿主机 + `VLLM_CACHE_ROOT` 持久化 + ≥15 min 启动预算**"（本轮 12 min；建议留 30 min 上限），
     并继续**禁止** `--enforce-eager`（硬约束 13）。

### 3.9 性能/一致性归因小结

| 问题 | 本步实测答案 | 关键数字 |
|---|---|---|
| 默认 `FULL_AND_PIECEWISE` 还会崩吗？ | **不会**（27B TP=2、0.6B、GDN-4B 全部捕获成功并服务；27B TP=4 因宿主内存 OOM，未到捕获阶段） | PIECEWISE 51/51 + FULL 35/35；0 次 `AttributeError`；`oom_kill 1`（TP=4，§3.7） |
| FULL 图在 decode 时真的在 replay 吗？ | **是**（runtime dispatch 统计） | 0.6B：`FULL ×155`；PIECEWISE 对照：`PIECEWISE ×155`（负载逐行相同） |
| FULL 能把 decode 拉回基线量级吗？ | **不能**（≈0.5% 收益） | 378.78 vs 380.62 ms/token；基线 122.47 ⇒ 仍 **3.09×** |
| 3.11× 的欠账在哪？ | **kernel/并行度**，不是图模式 | 生产 = TP4 + eager + 自研 `int8_sdot4` + GDN torch 替换；本树 = TP2 + Triton INT8 + 原生 FLA/GDN |
| FULL 数值可信吗？ | **0.6B 可信（逐 token == PIECEWISE）；27B 未证实** | 0.6B 5×32 token 全同；27B 2/3 条不同（近 tie 翻转 + 编译产物不同） |
| 为什么 27B 两模式不可逐位比？ | 换 `cudagraph_mode` 就换编译产物 | 4 个缓存目录 `018de25520` / `fda6c2438a` / `0bb72b1d53` / `92f2345be5`；PIECEWISE 自身换 `--max-model-len` 也翻 token |

**结论链**：
`pad_for_cudagraph` 缺失 ⟶ 已修且逐位等价（§2.2）⟶ FULL 捕获/服务/请求全通（§3.1/3.2/3.6/3.8）⟶
但 **FULL 不是 3.11× 的答案**（§3.4）⟶ 且 **FULL 与 PIECEWISE 在 27B 上不能互证**（§3.5）⟶
所以：修复按"缺陷已消除"归档；**性能欠账重新归因到 int8 GDN/GEMV 内核与 TP=4**（S10 的新方向，§6.1-2）；
**FULL 的数值正确性留一个显式未完成项**（需要 27B 的独立参照），生产继续用 PIECEWISE（收益本来就≈0）。

### 3.10 固化：`.gitignore` 白名单生效

改动（`.gitignore` 末尾，md5 `799806351bd0488f52cdb30c2405e732` → `db23f88aad38acf79d026e1da22e4403`）：

```gitignore
# --- Hygon Z100 / gfx906 deliverables (hard rules 14 & 19) -------------------
# The generic "*.sh" rule above hides tools/dcu/*.sh (s8b_gate.sh, s7_env.sh,
# ...) and "test_*.py" hides tests/kernels/test_gfx906_packed_silu_fp16.py and
# tests/platforms/test_hygon_dtk.py, so the Z100 gate scripts and regression
# tests never showed up in `git status`. Whitelist them explicitly.
!tools/dcu/**
# ...but keep bytecode caches inside the whitelisted directory ignored.
tools/dcu/__pycache__/
!tests/kernels/test_gfx906_packed_silu_fp16.py
!tests/platforms/test_hygon_dtk.py
```

生效前后（`git status --short`，只列交付物相关行）：

```
BEFORE
!! tests/kernels/test_gfx906_packed_silu_fp16.py      <- .gitignore:301 test_*.py
!! tests/platforms/test_hygon_dtk.py                  <- 同上（S2 交付物，一并白名单）
!! tools/dcu/build_triton_gfx906.sh                   <- .gitignore:257 *.sh
!! tools/dcu/env.sh
!! tools/dcu/s5_gate.sh ... tools/dcu/s8b_gate.sh
（`git status --short` 里只有 `?? tools/dcu/` 一个折叠目录）
AFTER
$ git status --short
?? tests/kernels/test_gfx906_packed_silu_fp16.py
?? tests/platforms/test_hygon_dtk.py
?? tools/dcu/
$ git status --short -uall | grep tools/dcu
?? tools/dcu/README.md
?? tools/dcu/build_triton_gfx906.sh
?? tools/dcu/check_bf16.py
...
?? tools/dcu/s8b_gate.sh
?? tools/dcu/s9_diag.py
?? tools/dcu/s9_par_dl.py
?? tools/dcu/triton-gfx906-3.6.0.patch
（无 __pycache__ 泄漏：`git status --short -uall | grep -c __pycache__` = 0；`git check-ignore -v tools/dcu/__pycache__/x.pyc`
 → `.gitignore:326:tools/dcu/__pycache__/`）
```
⇒ 硬约束 14/19 的"交付物在 `git status` 里可见"已落地，且不需要 `git add -f`。

### 3.11 回滚演练（**只对影子容器/工作区**，生产容器未触碰）

脚本 `/mnt/game/port/s9b/s9b_rollback.sh`（日志 `rollback-rehearsal.log`）：把 canonical 树
`vllm/v1/attention/backends/gdn_attn.py` 换成修复前的版本（`/tmp` 侧 `git show HEAD:…` 经 `docker cp` 入容器，md5 断言），
跑测试，再换回修复版并复测；全程只动 `z100-s9` 的影子树。

```
$ docker exec z100-s9 ... bash /mnt/game/port/s9b/s9b_rollback.sh
===== 0) snapshot the fixed file =====
fixed md5 = b17a394e2d17d5caeae2ddce595bbae7  (expect b17a394e2d17d5caeae2ddce595bbae7)
===== 1) pre-fix file (from git HEAD, staged by the host) =====
before md5 = 1de9e9d91a03968cd079153b33d41f09  (expect 1de9e9d91a03968cd079153b33d41f09)
351:            num_actual_tokens = self.vllm_config.pad_for_cudagraph(m.num_actual_tokens)
400:            num_actual_tokens = self.vllm_config.pad_for_cudagraph(m.num_actual_tokens)
===== 2) roll the canonical tree back to the pre-fix file =====
tree md5 = 1de9e9d91a03968cd079153b33d41f09
pad_for_cudagraph refs = 2
hasattr(VllmConfig, pad_for_cudagraph) = False          <- 修复前状态复现
===== 3) GDN metadata builder tests on the ROLLED-BACK tree =====
FAILED ...::test_decode_does_not_transfer_context_lens
FAILED ...::test_full_cudagraph_spec_metadata_uses_request_count          <- FULL 用例失败
FAILED ...::test_one_token_chunk_classification[first-chunk|resumed-chunk|padding]
FAILED ...::test_one_token_first_chunk_excludes_padding
FAILED ...::test_cudagraph_capture_batch_stays_decode_only                <- FULL 用例失败
7 failed, 17 warnings in 20.52s
--- failure detail (the two FULL-cudagraph cases) ---
vllm/v1/attention/backends/gdn_attn.py:473: in build_for_cudagraph_capture
E           AttributeError: 'VllmConfig' object has no attribute 'pad_for_cudagraph'
vllm/v1/attention/backends/gdn_attn.py:400: AttributeError
E           AttributeError: 'VllmConfig' object has no attribute 'pad_for_cudagraph'
vllm/v1/attention/backends/gdn_attn.py:351: AttributeError
===== 4) restore the fix, re-run the two FULL-cudagraph tests =====
tree md5 = b17a394e2d17d5caeae2ddce595bbae7  (expect b17a394e2d17d5caeae2ddce595bbae7)
2 passed, 17 warnings in 20.38s
ROLLBACK_REHEARSAL_DONE
```

结论：
1. 撤掉修复 ⇒ **FULL 路径以 `AttributeError` 失败**（与 S9 §3.10.1 的真机崩溃同一根因），且失败点正是 §2.2 改的两行；
2. 换回修复 ⇒ 两个 FULL 用例 `2 passed`；
3. 另外 5 个失败在**修复前后都存在**（同一份旧 stub / 旧 API），与本修复无关（§5）；
4. canonical 树最终 md5 = `b17a394e2d17d5caeae2ddce595bbae7` = 修复版（已复核，未留在回滚状态）。
   生产容器 `qwen38-vllm` 未被任何一条命令触碰；镜像层的 2.10 回退路径见附录 B-1/B-2/B-3。

### 3.12 27B 的 FULL replay 直接取证 + FULL/PIECEWISE 各自的跨进程确定性

第一轮 27B FULL（§3.1）没开 metrics，为排除"只捕获、不 replay"的疑问，补跑一轮同配置 + `--cudagraph-metrics`
（`/mnt/game/port/s9b/27b-tp2-full-metrics.log`；编译缓存命中，`torch.compile took 19.73 s`；
`02:19` 启动 → PIECEWISE 51/51（profile 轮 716 s）+ PIECEWISE 51/51 + `FULL 35/35` + `Graph capturing finished in 737 secs`
→ `02:50:42 Application startup complete`，KV 11.54 GiB）：

```
(APIServer) **CUDAGraph Config Settings:**
(APIServer) - Mode: FULL_AND_PIECEWISE
(APIServer) - Capture sizes: [1, 2, 4, 8, 16, ..., 248, 256, 272, ..., 512]
(APIServer) **CUDAGraph Stats:**
(APIServer) | Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Count |
(APIServer) |-----------------|---------------|--------------|--------------|-------|
(APIServer) | 1               | 1             | 0            | FULL         | 20    |   <- 27B 单请求 decode 确实在 replay FULL 图
```

同一份请求集在两轮**互相独立的 FULL 进程**上逐 token 相同（说明 FULL 模式本身是确定性的，不是抖动）：

```
$ python s9b_cmp.py req-full.json req-full-metrics.json
REQ chat_hangzhou : IDENTICAL (64 ids)
REQ cap_france    : IDENTICAL (24 ids)
REQ longer        : IDENTICAL (48 ids)
```

补测的 decode（与 §3.4 第一轮一致）：

```
$ bash bench_s9b.sh 8003 fullmetrics decode | grep -E "Mean TPOT|Mean TTFT"
Mean TTFT (ms):                          1308.05
Mean TPOT (ms):                          378.43
```

⇒ 汇总：**FULL 确定且可复现**（20 次 FULL replay、两轮进程输出逐 token 相同），
**PIECEWISE 也确定**（3 轮相同），**但两者之间不一致**（§3.5）——差异来自 `cudagraph_mode` 改变后必然不同的 torch.compile 产物，
而不是随机性。

### 3.13 固化留档位置

| 文件 | 内容 |
|---|---|
| `/mnt/game/port/s9b/27b-tp2-full.log` | 27B TP=2 **默认 FULL_AND_PIECEWISE** 全日志（PIECEWISE 51/51 + FULL 35/35、KV、startup） |
| `/mnt/game/port/s9b/27b-tp2-pw.log` | 27B TP=2 PIECEWISE 重跑日志（含 `--cudagraph-metrics` 的 dispatch 统计） |
| `/mnt/game/port/s9b/27b-tp2-full-metrics.log` | 27B TP=2 默认模式的**补跑**（`--cudagraph-metrics`）：`FULL×20 @ bs=1` 的 replay 取证 + 二轮 decode 378.43 |
| `/mnt/game/port/s9b/27b-tp4-full.log` | 27B TP=4 尝试（cgroup OOM 现场：`Worker proc VllmWorker-0 died` / `RuntimeError: cancelled`） |
| `/mnt/game/port/s9b/req-full.json` / `req-full-metrics.json` / `req-pw-{1,2,3}.json` | FULL（两轮）/ PIECEWISE（三轮）的精确 token id（`return_token_ids`） |
| `/mnt/game/port/s9b/rollback-rehearsal.log` | 回滚演练前后对照（§3.11） |
| `/mnt/game/port/s9b/logprobs-pw.json` | 分歧位置 top-k logprobs |
| `/mnt/game/port/s9b/zerob-full.log` / `zerob-pw.log` / `zerob-ids-*.json` / `zerob-cmp.log` | 0.6B 回归 + FULL/PIECEWISE 逐 token 对照 + dispatch 统计 |
| `/mnt/game/port/s9b/gdn-full.log` / `gdn-full-run.log` | GDN 30 min 硬超时重试 |
| `/mnt/game/port/s9b/s9b-gate.log` | 修复后 `s8b_gate.sh` 15/15 ALL GREEN |
| `/mnt/game/port/s9b/*.sh` / `s9b_req_compare.py` / `s9b_cmp.py` / `s9b_logprobs.py` / `s9b_req_generic.py` | 本步全部可复现脚本 |

---

## 4. 门禁判定

| # | 判据 | 判定 | 证据 |
|---|---|---|---|
| 1 | `pad_for_cudagraph` 语义论证 + 修复（不硬猜） | ✅ 通过 | §2.2（上游 main 已无该符号；`CudagraphDispatcher` 承接映射并在 build metadata 前 padding；`git show e8ae2d129^` 给出旧实现；两调用点改为 `m.num_actual_tokens`，**逐位等价**证明 §2.2-d） |
| 2a-捕获 | 27B TP=2 默认模式：FULL + PIECEWISE 捕获 | ✅ 通过 | §3.1（`PIECEWISE 51/51` + `FULL 35/35`，`Graph capturing finished in 749 secs`，0 次 `AttributeError`） |
| 2a-服务/请求 | 服务起来 + 请求成功 | ✅ 通过 | §3.1/§3.2（`Application startup complete`；3 条 greedy 请求 200，文本正常） |
| 2a-逐 token | **输出与 PIECEWISE 逐 token 一致** | ❌ **不通过** | §3.5（cap_france 一致；chat 第 13 token、longer 第 0 token 不同；PIECEWISE 自身对 max-model-len 也不不变；0.6B 对照则完全一致） |
| 2b | decode ms/token vs 380.70 / 122.47 | ✅ 通过（结论为"FULL 无收益"） | §3.4（FULL 378.78、PW 380.62、基线 122.47 ⇒ 3.09× 仍存在，S9 归因被证伪） |
| 2c | TP=4 一次 27B 请求 + TTFT@8192 | ⚠️ 部分（环境阻塞） | §3.7：TP=4 27B 被宿主内存卡死（cgroup `oom_kill 1`，宿主 30.5 GiB，生产容器 `MemLimit=0`）；TTFT 以 TP=2 `in=8000` 的 **per-prefill-token** 口径给出（3.508 / 3.524 vs 基线 2.789） |
| 2d | `s8b_gate.sh` ALL GREEN + 0.6B 图捕获正常 | ✅ 通过 | §3.0（pass=15 fail=0 ALL GREEN）、§3.6（51/51 + 35/35） |
| 3 | GDN(4B) 端到端 30 min 硬超时重试 | ✅ 通过（首次跑通） | §3.8（`GDN_READY=1 ELAPSED=729s`、PIECEWISE 51/51 + FULL 35/35、KV 6.97 GiB、HTTP 200 24 token） |
| 4a | `.gitignore` 白名单 + `git status` 证据 | ✅ 通过 | §3.10 |
| 4b | 固化与回滚一节（可直接执行） | ✅ 通过 | §3.11 + 附录 A/B |
| 硬约束 | 不碰生产容器 / 无自编译 HIP 可执行 / 无 `--enforce-eager` / 显式 fp16 / 缓存落 `/mnt/game` | ✅ 通过 | §1.2；`grep -c enforce-eager /mnt/game/port/s9b/*.sh` = 0 |

---

## 5. 未完成项 / 阻塞与原因

| 项 | 状态 | 原因 / 需要什么 |
|---|---|---|
| **27B 与 PIECEWISE 逐 token 一致** | ❌ 未达成 | 见 §3.5：两模式必然落到不同 torch.compile 产物（`cudagraph_mode` 是 hash 因子），且 PIECEWISE 自身对 `--max-model-len` 已经翻转 token。**要下结论必须补独立参照**：用 HF transformers（fp16，device_map 铺 4 卡）对同 3 条 prompt 做逐 token 对照，或对 vLLM 侧固定同一份编译产物（当前框架做不到） |
| FULL 是否可上生产 | ⏸️ 建议**暂不** | 收益 0.5%（§3.4），风险是未验证的数值差异（§3.5）；PIECEWISE 已是 S9/S10 的既有安全区 |
| `tests/v1/attention/test_gdn_metadata_builder.py` 5 个 CPU 用例 | ❌ 既有失败（非本步引入） | 该文件的 `_create_stateless_builder()` 仍用旧 API（`common.num_computed_tokens_cpu`，已在 sync 中移除）与旧 stub（缺 `kv_cache_spec`/`vllm_config`），`AttributeError: 'CommonAttentionMetadata' object has no attribute 'num_computed_tokens_cpu'`。属"过期测试替身"，与 S9 修的 `test_qwen3_5_gdn.py` 同类；本步只做 pad 修复，未扩到该文件（其两个 FULL 用例通过，见 §3.11） |
| 是否按"无法确证语义"分支加显式降级/告警 | ✅ 不需要（已确证语义） | 任务书要求"无法确证语义时不要硬猜、改成安全回退并打日志"。本步**确证了语义**：旧方法 = `bs_to_padded_graph_size[bs]`，而该映射对捕获桶是恒等映射，且 padding 已由 model runner 提前完成 ⇒ 新写法与旧返回值**逐位相同**，不存在"猜一个近似实现"的问题。因此**不**在 gfx906/GDN 路径硬编码降级（那会与已验证的等价性矛盾、也是额外风险）；"GDN 生产继续用 PIECEWISE"作为**部署建议**写入 §6.1，而非代码强制 |
| 27B 的"独立数值参照"（HF fp16 逐 token） | ⏸️ 未做 | 27B 反量化后 fp16 需 ≈54 GB，要 4 卡 device_map + HF 侧 Qwen3.5 实现；不在本步时间盒内（任务书也未要求）。这是"FULL 数值是否可信"的**唯一缺口**，建议 S10 或专项补 |
| 27B TP=4（含严格 TTFT@8192 口径） | ⛔ 环境阻塞 | 宿主 30.5 GiB RAM + `z100-s9` 32 GiB cgroup ⇒ `oom_kill 1`（§3.7）。需要 `--memory=0` 的容器或 ≥48 GiB RAM 的机器；生产容器本身就是 `MemLimit=0` |

---

## 6. 对后续步骤的影响

### 6.1 S10 与生产上线

1. **`pad_for_cudagraph` 崩溃已消除**：任何 GDN/mamba 混合模型 + `FULL_AND_PIECEWISE`（默认）**不再在捕获期崩**。
   `docs/z100-todolist.md` 硬约束 20 与 S9 §6.1-1/9 的"禁止 FULL 图"前提**已失效**，可改为"FULL 可用但暂不推荐用于 GDN 模型"。
2. **性能欠账的真实位置**：decode 378.78 vs 基线 122.47（3.09×）**不是图模式**。S10/生产优化应聚焦：
   (a) W8A8 INT8 的 GEMV/GEMM —— 生产用自研 `int8_sdot4`（`INT8_SDOT4_*` 开关），本树用 Triton `TritonInt8ScaledMMLinearKernel`；
   (b) GDN prefill —— 生产用"方案 A 纯 torch GDN 替换"（`GDN_TORCH_DISABLE=0` / `PHASEA_GDN_MODE=fastinv`），
       本树用原生 FLA/Triton（S9 实测 prefill +17.5%/token 与此一致）；
   (c) TP=4 vs TP=2（生产 TP=4）。
   **不要再把 decode 差距归因到 cudagraph 模式**（本步同栈 A/B：378.78 vs 380.62）。
3. **数值一致性是新门禁项**：同一模型在"换 cudagraph_mode / 换 `--max-model-len`"后 greedy 输出可能在近 tie 处翻转
   （§3.5）。任何"性能优化"验收都应带**逐 token 对照**，且基准必须钉住 KV 几何与编译缓存，否则对照本身无意义。
4. **启动预算**：27B/TP=2 默认 `FULL_AND_PIECEWISE`（PIECEWISE 51+51 与 FULL 35）实测 ≈29 min ready（S9 的 PIECEWISE-only ≈39 min，
   差别主要来自 `--max-model-len` 与机器状态）。生产若要开图，仍要按 §6.1-8 给足窗口。
5. **固化与回滚**（附录 A/B）可直接用于 S10 的环境搭建与生产回退；`tools/dcu/*.sh` 现在在 `git status` 里可见，入库可直接 `git add`。
6. **回归锁**：`tests/v1/attention/test_gdn_metadata_builder.py` 的两个 FULL 用例即是本修复的回归测试（§3.11 给出修复前失败、修复后通过）。
7. **环境规格（S10 必读）**：27B 的 **TP=4** 在本机不可行 —— 宿主只有 **30.5 GiB RAM**，而影子容器 `z100-s9` 有 **32 GiB cgroup 上限**
   （S9 建容器时按"`--memory>=32g`"设的），27B/TP=4 启动 4 min 即 `oom_kill 1`（§3.7）。
   生产容器 `qwen38-vllm` 的 `MemLimit=0`（无限制），所以它的 TP=4 能跑。
   ⇒ S10 若要复现 TP=4/严格 TTFT@8192，必须用 `--memory=0` 的容器或 ≥48 GiB RAM 的机器；**不要**在共享宿主上放开 `--memory`（会连累其它容器）。

---

## 7. 复现方式

```bash
# ---------- 0) 前置：z100-s9（32g/32g, shm 16g, /src:ro, /mnt/game rw）；canonical 树 /mnt/game/port/gfx906-vllm-211
# ---------- 1) 落修复 + 同步 ----------
cd /home/lyh/z100-vllm && git diff --stat vllm/v1/attention/backends/gdn_attn.py   # 期望 1 file, +8 -4 左右
docker exec z100-s9 bash -c 'V=/mnt/game/port/gfx906-vllm-211
  cp /src/vllm/v1/attention/backends/gdn_attn.py $V/vllm/v1/attention/backends/gdn_attn.py
  grep -c pad_for_cudagraph $V/vllm/v1/attention/backends/gdn_attn.py'   # 期望 0

# ---------- 2) 门禁（必须 ALL GREEN） ----------
docker exec z100-s9 bash -lc 'source /opt/dtk/env.sh; export VLLM_CACHE_ROOT=/mnt/game/vllm-cache
  cd /mnt/game/port/gfx906-vllm-211 && bash tools/dcu/s8b_gate.sh 2>&1 | tee /mnt/game/port/s9b/s9b-gate.log | tail -8'
# 期望：CHECK_SUMMARY pass=15 fail=0 / RESULT: ALL GREEN

# ---------- 3) 27B TP=2 默认 cudagraph（FULL 35/35 + PIECEWISE 51/51） ----------
docker exec -d z100-s9 bash -c 'bash /mnt/game/port/s9b/run_27b_tp2_full.sh > /mnt/game/port/s9b/27b-tp2-full.log 2>&1'
# 就绪判定：日志出现 Application startup complete（实测 ≈29-40 min）
docker exec z100-s9 bash -c 'grep -aoE "Capturing CUDA graphs \((PIECEWISE|FULL)\): 100%[^\r]*" /mnt/game/port/s9b/27b-tp2-full.log | tail -2'
# 请求 + 精确 token id
docker exec z100-s9 bash -c 'python /mnt/game/port/s9b/s9b_req_compare.py 8003 /mnt/game/port/s9b/req-full.json'
# decode / conc8 / TTFT
docker exec z100-s9 bash -c 'bash /mnt/game/port/s9b/bench_s9b.sh 8003 full decode'
docker exec z100-s9 bash -c 'bash /mnt/game/port/s9b/bench_s9b.sh 8003 full conc8'
docker exec z100-s9 bash -c 'bash /mnt/game/port/s9b/bench_s9b.sh 8003 full ttft8000'

# ---------- 4) PIECEWISE 对照（同一构建 + --cudagraph-metrics） ----------
# 先杀旧服务（comm 口径；勿用 pkill -f 匹配到自己的命令行）
docker exec z100-s9 bash -c 'pkill -9 VLLM; sleep 5'
docker exec -d z100-s9 bash -c 'bash /mnt/game/port/s9b/run_27b_tp2_pw.sh > /mnt/game/port/s9b/27b-tp2-pw.log 2>&1'
docker exec z100-s9 bash -c 'for i in 1 2 3; do python /mnt/game/port/s9b/s9b_req_compare.py 8003 /mnt/game/port/s9b/req-pw-$i.json; done
  python /mnt/game/port/s9b/s9b_cmp.py /mnt/game/port/s9b/req-pw-1.json /mnt/game/port/s9b/req-pw-2.json /mnt/game/port/s9b/req-pw-3.json /mnt/game/port/s9b/req-full.json'

# ---------- 5) 0.6B 回归 + FULL/PIECEWISE 逐 token 对照 ----------
docker exec -d z100-s9 bash -c 'bash /mnt/game/port/s9b/s9b_zerob_cmp.sh > /mnt/game/port/s9b/zerob-cmp.log 2>&1'
docker exec z100-s9 bash -c 'cat /mnt/game/port/s9b/zerob-cmp.log'

# ---------- 6) GDN(4B) 30 min 硬超时 ----------
docker exec -d z100-s9 bash -c 'bash /mnt/game/port/s9b/run_gdn_full.sh > /mnt/game/port/s9b/gdn-full-run.log 2>&1'
docker exec z100-s9 bash -c 'cat /mnt/game/port/s9b/gdn-full-run.log'
# 期望：GDN_READY=1 ELAPSED≈729s；PIECEWISE 51/51 + FULL 35/35；HTTP 200

# ---------- 6b) 27B 默认模式 + dispatch 取证（可选复跑，编译缓存命中后 ≈33 min 到 ready） ----------
docker exec -d z100-s9 bash -c 'bash /mnt/game/port/s9b/run_27b_tp2_full_metrics.sh > /mnt/game/port/s9b/27b-tp2-full-metrics.log 2>&1'
docker exec z100-s9 bash -c 'bash /mnt/game/port/s9b/bench_s9b.sh 8003 fullmetrics decode'
# 期望日志：| 1 | 1 | 0 | FULL | <n> |

# ---------- 6c) TP=4（本机会 OOM，仅作现场留档；勿在共享宿主放开 --memory） ----------
# docker exec -d z100-s9 bash -c 'bash /mnt/game/port/s9b/run_27b_tp4_full.sh > /mnt/game/port/s9b/27b-tp4-full.log 2>&1'
# docker exec z100-s9 bash -c 'grep oom_kill /sys/fs/cgroup/memory.events'   # 期望 oom_kill 1（§3.7）

# ---------- 7) .gitignore 白名单 ----------
cd /home/lyh/z100-vllm && git status --short | grep -E "tools/dcu|test_gfx906|test_hygon"
git status --short -uall | grep tools/dcu
```

---

## 8. 参考

- 前一档：[S9 验收与对拍](z100-09-acceptance-and-bench-result.md)（§3.10.1 崩溃现场、§5.2-1 缺陷、§6.1-9 性能欠账）
- [S7 内存/KV/graph](z100-07-memory-kvcache-graph-result.md)（0.6B：PIECEWISE 51/51 + FULL 35/35；decode 11.6 vs eager 34.6）／[S7b](z100-07b-eager-nan-result.md)／[S7c](z100-07c-packed-silu-csrc-fix-result.md)（硬约束 13/15）
- 上游代码（只读）：`/mnt/game/port/upstream-vllm/vllm/v1/cudagraph_dispatcher.py:71-154`、`vllm/v1/attention/backends/gdn_attn.py:325-326`；`/mnt/game/port/upstream-vllm-031/`
- 本 fork 历史：`56539cdda`（上游 #28579，"pad for CUDA graphs before attention metadata building"，本树祖先）、
  `e8ae2d129`（squash sync：删 `VllmConfig.pad_for_cudagraph`、留旧调用点）、`31d39f0fc`（fork dev mirror：`batch_size = m.num_reqs`）
- 运行面：`/mnt/game/port/gfx906-vllm-211`、`/mnt/game/triton-gfx906/{install,cache}`、`/mnt/game/vllm-cache-27b|-gdn|-zerob`
- 生产基线数据（只读）：`/home/lyh/server/gfx906_port/bench/results/base/{bench_decode.log,bench_ttft.log}`、`/home/lyh/vllmbench/config.env`、`/home/lyh/server/serve_27b_tp4.sh`

---

# 附录 A 固化清单（S9b 更新，S9 附录 A 的增量）

## A.1 容器（本步只新建/复用了 `z100-s9`；生产容器未触碰）

| 容器 | 状态 | 用途 |
|---|---|---|
| **`z100-s9`** | Up（S9 新建，本步复用） | 本步全部运行面：torch 2.11.0+das.opt1.dtk2604 + 自编 triton 3.6.0+git7c56a5e4 + canonical 树 4 扩展；`--memory=32g --memory-swap=32g --shm-size=16g`，`-v /home/lyh/z100-vllm:/src:ro -v /mnt/game:/mnt/game` |
| `z100-triton` / `z100-build` | Up | **2.10 基线栈**回滚面（`torch 2.10.0+das…20260325` + DTK triton 3.4.0 + vendor vllm 0.18.1 轮子）；本步**只读**，未编译 |
| `z100-s8` / `z100-serve` / `z100-eager` / `z100-build-211` | Up | 历史实验面，未动 |
| `qwen38-vllm` | **Exited(137)**（用户自管） | **本步未 start/kill/exec/commit**；镜像与容器未被本步任何命令写入 |

## A.2 关键路径与 md5（本步实采）

```bash
# 4 个扩展（canonical 安装面，运行用）
6fba05966e153b3d04771d6960c782bb  78102728  vllm/_C.abi3.so
5d1e50e1a77de74f2d308519f8120745  35156368  vllm/_rocm_C.abi3.so
93bf7b4a391ac97e2f67cc7141633cb1  40691216  vllm/_C_stable_libtorch.abi3.so   <- 硬约束 15 的身份，门禁复核一致
996b282c4c6a3d3b46b5e8ba4922af39  29748928  vllm/_moe_C_stable_libtorch.abi3.so
# 自编 triton / 补丁 / requirements / 门禁脚本
59f976927f7800850a4ad79a9050a59a  /mnt/game/triton-gfx906/triton-gfx906-3.6.0.patch
1629165197554e87d38eabe5bd3a78f6  requirements/dcu.txt
376e3749c1354e89876f0706249a77d6  requirements/dcu-build.txt
c03f1c34d46e35fb8bcba02399ef7211  tools/dcu/s8b_gate.sh
# 本步改动的两个文件
1de9e9d91a03968cd079153b33d41f09 -> b17a394e2d17d5caeae2ddce595bbae7  vllm/v1/attention/backends/gdn_attn.py
799806351bd0488f52cdb30c2405e732 -> db23f88aad38acf79d026e1da22e4403  .gitignore
```

## A.3 环境变量（与 S9 附录 A.6 相同；本步实采）

| 变量 | 值 |
|---|---|
| `VLLM_DCU_GCN_ARCH` | `gfx906` |
| `VLLM_DCU_BF16` | `auto`（启动一律显式 `--dtype float16`） |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `0` |
| `VLLM_CACHE_ROOT` | `/mnt/game/vllm-cache-27b`（27B）/ `-gdn`（4B）/ `-zerob`（0.6B） |
| `PYTHONPATH` | `${TRITON_INSTALL}:${VLLM_DIR}:${tools/dcu}`（`tools/dcu/s7_env.sh` 自动） |
| `TRITON_CACHE_DIR` | `/mnt/game/triton-gfx906/cache` |
| `TRITON_HIP_LLD_PATH` | `/opt/dtk/llvm/bin/ld.lld` |
| `HIP_VISIBLE_DEVICES` | 27B TP=2 → `2,3`；单卡实验 → `0`；0.6B TP2/TP4 历史 → `0,1`/`0-3` |
| `ROCBLAS_TENSILE_LIBPATH` | **必须 unset**（导出即崩） |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True`（27B/4B） |
| `--dtype` | `float16`（全部 serve） |
| `-cc.cudagraph_mode` | 本步默认（`FULL_AND_PIECEWISE`）；对照轮 `PIECEWISE` |

## A.4 本步新增/复用的脚本（`/mnt/game/port/s9b/`）

| 脚本 | 用途 |
|---|---|
| `run_27b_tp2_full.sh` | 27B TP=2 **默认 cudagraph**（不传 `-cc.cudagraph_mode`），:8003，maxlen 8192 |
| `run_27b_tp2_pw.sh` | 同上 + `-cc.cudagraph_mode=PIECEWISE --cudagraph-metrics`（对照） |
| `bench_s9b.sh` | `decode` / `ttft8000` / `ttft8192` / `conc8` 四种口径的 `vllm bench serve` |
| `s9b_req_compare.py` | 27B 固定 greedy 请求集 + `return_token_ids`（精确 id） |
| `s9b_req_generic.py` / `s9b_cmp.py` / `s9b_logprobs.py` | 任意模型 id 采集 / 序列比对 / top-k logprobs |
| `run_zerob_mode.sh` / `s9b_zerob_cmp.sh` | 0.6B FULL↔PIECEWISE 逐 token 对照（含 dispatch 统计） |
| `run_gdn_full.sh` | GDN(4B) 默认 cudagraph + **1800 s 硬超时**（超时 kill -9 并记录卡点） |

---

# 附录 B 回滚程序（可直接执行；**只对影子容器/工作区**）

**B-1 退回 2.10 基线（= 生产那一套 vendor 栈）**
```bash
docker exec -it z100-triton bash          # 或 z100-build；两者都是 2.10 基线栈
source /opt/dtk/env.sh                    # 不要 source tools/dcu/s7_env.sh（那会前置自编 triton + 211 树）
python -c "import importlib.metadata as m; import torch; print(m.version('torch'), m.version('vllm'), m.version('triton'))"
# 期望：2.10.0+das.opt1.dtk2604.20260325.g6b060a / 0.18.1+das.3266200.dtk2604 / 3.4.0+git1ef59765
vllm serve <model> --dtype float16 ...    # 走 site-packages 的 vendor 轮子
# 生产容器 qwen38-vllm 的启停由用户执行（本步与 S9 都未触碰）
```

**B-2 只撤自编 triton（保留 211 树）**：不导出 `PYTHONPATH=/mnt/game/triton-gfx906/install:…`、不 `source tools/dcu/s7_env.sh`。
DTK 原装 triton 目录从未被改（硬约束 4）⇒ 无需恢复。

**B-3 撤 S7c 的 fp16 packed-SiLU 修复（`_C_stable_libtorch`）**
```bash
docker exec z100-s9 bash -c '
  cp /mnt/game/port/s7c/pre-fix/vllm/_C_stable_libtorch.abi3.so \
     /mnt/game/port/gfx906-vllm-211/vllm/_C_stable_libtorch.abi3.so
  md5sum /mnt/game/port/gfx906-vllm-211/vllm/_C_stable_libtorch.abi3.so'
# 期望 cc2aca69ca9a9afcb67b7428ede181b1 40688656（回退后 fp16 packed SiLU 的非有限值会复现，仅用于复现 S7b/S7c）
```

**B-4 撤本步的 `pad_for_cudagraph` 修复（回到 S9 的 FULL 崩溃状态）**
```bash
# 工作区
cd /home/lyh/z100-vllm && git checkout -- vllm/v1/attention/backends/gdn_attn.py
# 或从 HEAD 取旧文件推进 canonical 树（在 z100-s9 内）
docker exec z100-s9 bash -c 'git -C /src show HEAD:vllm/v1/attention/backends/gdn_attn.py \
  > /mnt/game/port/s9b/gdn_attn.py.before
  grep -c pad_for_cudagraph /mnt/game/port/s9b/gdn_attn.py.before        # 期望 2
  md5sum /mnt/game/port/s9b/gdn_attn.py.before'                           # 期望 1de9e9d91a03968cd079153b33d41f09
```
（本步的真演练与前后对照见 §3.11；撤掉后 FULL 模式会重新抛
`AttributeError: 'VllmConfig' object has no attribute 'pad_for_cudagraph'`，PIECEWISE 不受影响。）

**B-5 撤 `.gitignore` 白名单**
```bash
cd /home/lyh/z100-vllm && git checkout -- .gitignore
git status --short --ignored | grep -E "tools/dcu|test_gfx906|test_hygon"   # 会重新变回 !! 列表
```

**B-6 生产容器的"回滚"口径**：`qwen38-vllm` 当前 `Exited(137)`，镜像 tag 未变、容器未被本步任何命令写入
⇒ "新引擎失败不影响生产"在**镜像与容器层面成立**；其启停由用户执行（§1.2）。

---

# 附录 C 本步已测矩阵（增量）

| 模型 | 量化 | 卡数 | cudagraph_mode | 栈 | 结果 |
|---|---|---|---|---|---|
| Qwen3.8-27B-Channel-INT8-w8a8 | W8A8-int8 | 2（卡2,3） | **默认 `FULL_AND_PIECEWISE`** | 211 + 自编 triton | ✅ FULL 35/35 + PIECEWISE 51/51、ready ≈29 min、KV 11.53 GiB/227,448 tok、请求 200、decode 378.78 ms/tok；⚠️ greedy 输出与 PIECEWISE 2/3 条不同（§3.5） |
| Qwen3.8-27B-Channel-INT8-w8a8 | W8A8-int8 | 2（卡2,3） | **默认 `FULL_AND_PIECEWISE` + `--cudagraph-metrics`（复跑）** | 211 + 自编 triton | ✅ 51/51 + 35/35、编译缓存命中（`torch.compile took 19.73 s`）、ready 737 s 捕获、KV 11.54 GiB；dispatch = **FULL×20 @ bs=1**；decode 378.43 ms/tok；两轮 FULL 进程输出逐 token 相同 |
| Qwen3.8-27B-Channel-INT8-w8a8 | W8A8-int8 | 2 | `PIECEWISE` | 同上 | ✅ 51/51、KV 11.59 GiB/228,412 tok、decode 380.62 ms/tok（= S9 的 380.70，复现性良好）；3 次请求 id 完全确定 |
| Qwen3-0.6B | fp16 | 1（卡0） | 默认 `FULL_AND_PIECEWISE` | 同上 | ✅ 51/51 + 35/35、ready 52 s、decode dispatch = **FULL×155** |
| Qwen3-0.6B | fp16 | 1 | `PIECEWISE` | 同上 | ✅ 51/51、decode dispatch = PIECEWISE×155、**与 FULL 组 5×32 token 逐 token 相同** |
| Qwen3.5-4B（GDN 混合） | fp16 | 1 | 默认 `FULL_AND_PIECEWISE` | 同上 | ✅ **首次端到端跑通**（GDN_READY=729 s）：PIECEWISE 51/51 + **FULL 35/35**、KV 6.97 GiB/122,880 tok、HTTP 200/2.04 s/24 token |
| Qwen3.8-27B-Channel-INT8-w8a8 | W8A8-int8 | 4（卡0–3） | 默认 `FULL_AND_PIECEWISE` | 同上 | ⛔ 未跑成：cgroup OOM（`oom_kill 1`，启动 02:07:30 → 02:11:35 被杀；宿主仅 30.5 GiB RAM） |

---

# 附录 D 本步发现/处置的缺陷

| # | 缺陷 | 性质 | 处置 |
|---|---|---|---|
| D-1 | `VllmConfig.pad_for_cudagraph` 缺失 ⇒ GDN/混合模型 FULL 捕获必崩（S9 附录 D-1） | sync 只带了一半的上游重构 | ✅ **已修**：两调用点改用真正的取值来源 `m.num_actual_tokens`（§2.2），27B 默认模式 FULL 35/35 通过 |
| D-2 | S9 把 decode 3.11× 归因于"缺 FULL 图" | 归因错误（比较对象是 eager，不是 PIECEWISE） | ✅ **已证伪**：同栈 A/B 378.78（FULL）vs 380.62（PIECEWISE）⇒ 图模式几乎无收益（§3.4） |
| D-3 | 换 `cudagraph_mode`/`--max-model-len` ⇒ 不同 torch.compile 产物 ⇒ 近 tie 的 greedy token 翻转 | 框架行为（`cudagraph_mode` 是编译 hash 因子）+ 量化模型 logits 平坦 | ⚠️ 记录 + 门禁建议（§3.5/§6.1-3）；27B 逐 token 一致性**未达成**，需独立参照 |
| D-4 | `tests/v1/attention/test_gdn_metadata_builder.py` 5 个 CPU 用例用旧 API | 过期测试替身（非本步引入） | ⚠️ 记录（§5）；同文件两个 FULL 用例即本修复的回归锁 |
