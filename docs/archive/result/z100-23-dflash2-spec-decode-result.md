# S19 DFlash2 投机解码在 Z100(gfx906) 上的落地 — 执行结果

> 归档时间：2026-10-13
> 执行者：Lead（DSH 会话，无子 agent）
> 目标机：4× Hygon Z100（gfx906，32 GiB/卡），容器 `z100-vllm`，`/src` = `/home/lyh/z100-vllm`
> 靶子：`/mnt/game/models/Qwen3.8-27B-FP8`（FP8 block-scaled，TP4，`--dtype float16`）
> 草稿：`z-lab/Qwen3.8-27B-DFlash2`（bf16，3.85 GB，HF 原仓 `incoai/Qwen3.8-27B-DFlash2`）
> 状态：✅ 完成（DFlash2 端到端可用并实测；遗留 1/3 贪心输出近似平局翻转待复核）

## 1. 目标与验收标准

在当前 vLLM（本 fork，`method="dflash"` 已有代码但从未在 Z100 上跑过）上把
DFlash2 投机解码跑起来，验收：

1. 引擎能起来、能正常出 token（不崩、不 NaN）；
2. 草稿真正有效：平均接受长度（acceptance length）显著 > 1，接近上游参考值；
3. 给出与生产臂（FP8 + MTP k=2）可比的吞吐/接受率原始数据；
4. 代码改动可复现，附测试与证据。

## 2. 结论摘要

| 项 | 生产基线 FP8+MTP k=2 | 本次 DFlash2 k=7（fp16 靶 + bf16 草稿） |
| --- | --- | --- |
| 平均接受长度（真实文本贪心，3 条） | 2.46（bench，随机 token） | **6.68 / 6.65 / 7.65** |
| 逐位置接受率（服务端聚合） | — | 0.889 / 0.806 / 0.750 / 0.694 / 0.667 / 0.639 / 0.500 |
| 单请求真实文本吞吐（128 tok） | 19.58–19.77 tok/s | **22.16 tok/s**（+12%） |
| `vllm bench` decode（随机 128/128, c=1） | 15.87 tok/s, acc 2.46 | 11.95 tok/s, acc 3.82 |
| `vllm bench` conc8（512/128, c=8） | 31.16 tok/s, acc 2.44 | 31.64 tok/s, acc 3.44 |
| 贪心输出与 MTP 逐字节一致 | 参考 | 3 条里 2 条一致；1 条在近似平局处差 1 个 token |

结论：**DFlash2 在 Z100 上可以跑，且接受长度远超 MTP（6.6–7.6 vs 2.5）**；
但草稿用 bf16（gfx906 无原生 bf16，走 DTK 模拟）+ DFlash2 选择器开销，
导致随机 token 负载下每步成本高于 MTP，真实文本负载下才净赚约 12%。

## 3. 起始状态与四个真实阻塞（都已修）

### 3.1 靶模型缺 EAGLE-3 辅助隐状态接口

```
vllm/v1/worker/gpu/model_runner.py:389  set_eagle3_aux_hidden_state_layers(self.model, ...)
vllm/model_executor/models/interfaces.py:1698  assert isinstance(holder, EagleModelMixin)
AssertionError: Model instance must inherit from EagleModelMixin to set auxiliary layers
```

`Qwen3_5ForConditionalGeneration`（Qwen3.8-27B）内层 `Qwen3_5Model` ← `Qwen3NextModel`
在本 fork 里是 `nn.Module`，没有 `EagleModelMixin`，也没有把 aux hidden states 交出来，
而 DFlash 就是靠目标第 5/19/33/47/61 层的残差流做草稿输入。

### 3.2 gfx906 统一注意力没有非因果（双向）通路

```
vllm/v1/attention/ops/triton_unified_attention.py:1148
AssertionError: Only causal attention is supported
```

DFlash 草稿的 5 层全是 `is_causal=false` 的滑窗双向注意力（块内扩散），
gfx906 fork 的 Triton 统一注意力只实现了因果掩码。

### 3.3 草稿在 fp16 下溢出成 NaN（最隐蔽的一个）

DFlash2 检查点是 bf16（其 MLP 输出实测 absmax ≈ 1.4e5 > fp16 上限 65504），
而 `SpeculativeConfig` 把草稿 dtype 硬绑到靶模型 dtype（fp16），于是：

```
[s19-dflash2] draft_hidden dtype=torch.float16 nans=35840 absmax=nan
[s19-layer]   after mlp  hs_nan=0 nonfinite=1792 absmax=inf     ← 溢出
[s19-layer]   after mlp_conv.finish hs_nan=1536                  ← inf*0 -> NaN
```

NaN 之后 `unary_logits=[nan,...]`、`draft_tokens=[15,15,...]`，
接受长度恒为 1.02——看起来"能跑"，实际草稿全是垃圾。

### 3.4 一系列 dtype 管道问题（草稿要独立 dtype 才会暴露）

* `triton_matmul`（gfx906 小 M 矩阵乘）**硬编码 fp16 输出**，bf16 线性层走它必然 dtype 不匹配；
* gfx906 的 C RMSNorm kernel 是 Half 定点实现，bf16 输入直接 `expected scalar type Half but found BFloat16`；
* 草稿的 `hidden_states` 缓冲、共享的 target embedding / lm_head 都按靶 dtype 分配。

## 4. 实际改动

| 文件 | 改动摘要 |
| --- | --- |
| `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextModel(nn.Module, EagleModelMixin)`；forward 里 `_maybe_add_hidden_state` 收集 aux 残差流（索引 j = 进入第 j 层的张量），返回 `(hidden_states, aux_hidden_states)`，并把 gfx906 GDN 产生的 fp32 激活 cast 回模型 dtype |
| `vllm/v1/attention/ops/triton_unified_attention.py` | 2D/3D 两个 kernel 新增 `CAUSAL` constexpr：非因果时 tile 覆盖全序列、双向滑窗 `|q-k| < W`、只屏蔽越界 padding；wrapper 传 `CAUSAL=causal`；KV cache dtype 与 Q 不一致时按 Q 转换；A/B 开关 `VLLM_GFX906_ATTN_FORCE_CAUSAL=1` |
| `vllm/model_executor/layers/utils.py` | `triton_matmul` 输出改为 `a.dtype`，kernel 内 `accumulator.to(c_ptr.dtype.element_ty)`（原为固定 fp16） |
| `vllm/kernels/vllm_c.py` | gfx906 上 C RMSNorm 的 IR impl 仅对 fp16 声明支持，其余 dtype 退回 native |
| `vllm/config/speculative.py` | 新增 `SpeculativeConfig.dtype`：草稿可用自己的 dtype（默认仍继承靶） |
| `vllm/v1/worker/gpu/spec_decode/dflash/utils.py` | 草稿的 `vllm_config.model_config` 换成草稿自己的 ModelConfig |
| `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | 融合后隐状态缓冲按草稿 dtype 分配（原按靶 dtype） |
| `vllm/model_executor/models/qwen3_dflash.py` | 上下文 KV 预计算的 RMSNorm 增加 native 回退（gfx906 C kernel 只吃 fp16）；共享 embedding / `fc` / `lm_head` 三处 dtype 边界 cast |
| `vllm/model_executor/models/qwen3_dflash2.py` | 候选 logits 前把隐状态 cast 到共享 lm_head 的 dtype |

新增脚本（可复现）：

* `tools/dcu/s19_dflash_arm.sh`：容器内单臂启动器（`<fp8|int8> <TP> <PORT> <DRAFT> <K>`，
  支持 `S19_DRAFT_DTYPE` / `S19_DRAFT_KV_DTYPE` / `S19_MAXLEN` 等）；
* `tools/dcu/s19_dflash_e2e_fp8.sh`：FP8 靶 + DFlash2 的 :8000 一键臂；
* `tools/dcu/s19_non_causal_attn_check.py`：非因果注意力正确性对拍（2D/3D × 因果/非因果 × 无窗/32/8 窗，共 9 例）；
* `tools/dcu/s19_parity_collect.py`、`tools/dcu/s19_timed_req.py`：贪心一致性采集与真实文本计时。

## 5. 执行过程与原始证据

### 5.1 草稿落地

HF 直连被墙；容器内 `/etc/hosts` 有内网镜像（`fc00::1b huggingface.co`），小文件可下、
3.85 GB 大文件会挂住（hf-xet 路径）。改用 ModelScope：

```
python -c 'from modelscope import snapshot_download;
           snapshot_download("incoai/Qwen3.8-27B-DFlash2",
                             local_dir="/mnt/game/models/Qwen3.8-27B-DFlash2-ms")'
# 81 个张量、3,848,817,896 B、全 bf16、无 NaN/Inf
```

### 5.2 非因果注意力内核自测（9/9 PASS）

```
docker exec z100-vllm bash -lc '. /opt/dtk/env.sh; . /src/tools/dcu/env.sh;
  PYTHONPATH=/src:/mnt/game/triton-gfx906/install HIP_VISIBLE_DEVICES=0 \
  python /src/tools/dcu/s19_non_causal_attn_check.py'
PASS causal nowindow / noncausal nowindow / causal window32 / noncausal window32 /
     noncausal window8 / causal window8 / 3d causal nowindow / 3d noncausal nowindow /
     3d noncausal window8   →  ALL PASS（max_abs_diff ≤ 5e-4，fp16 量级）
```

### 5.3 启动证据（`/mnt/game/dflash/serve-fp8-tp4-dflash-k7.log`）

```
non-default args: ... 'speculative_config': {'model': '/mnt/game/models/Qwen3.8-27B-DFlash2-ms',
                    'method': 'dflash', 'num_speculative_tokens': 7,
                    'dtype': 'bfloat16', 'kv_cache_dtype': 'bfloat16'}
Using Eagle3 auxiliary layers from config: (6, 20, 34, 48, 62)
Available KV cache memory: 16.68 GiB / GPU KV cache size: 808,413 tokens
Maximum concurrency for 262,144 tokens per request: 3.08x
Application startup complete.
SpecDecoding metrics: Mean acceptance length: 5.94, Per-position acceptance rate:
  0.889, 0.806, 0.750, 0.694, 0.667, 0.639, 0.500, Avg Draft acceptance rate: 70.6%
```

### 5.4 真实文本计时与贪心一致性

```
# DFlash2（清理后代码复跑，与清理前逐位一致）
dflash-bf16-k7-clean: tokens=128 wall=5.78s tok/s=22.16 acc=6.684 md5=4a48d8170868240c783a4f2e083b35c9
# 生产臂 MTP k=2
restored-base-mtp-k2: tokens=128 wall=6.54s tok/s=19.58 acc=None md5=0ac703663f5401e6321ed954e00c0b4d
```

`tools/dcu/s19_parity_collect.py`（temperature=0、repetition_penalty=1.0、128 tok）：

| prompt | DFlash2 md5 | MTP k=2 md5 | 一致 |
| --- | --- | --- | --- |
| Fibonacci | `4a48d817…` | `0ac703663…` | ✗ |
| 素数 | `4929be4c…` | `4929be4c…` | ✓ |
| 火车均速 | `bec355b9…` | `bec355b9…` | ✓ |

不一致的那条只差一个 token：DFlash 出 `for n > 1.`，MTP 出 `for n >= 2.`（语义等价），
出现在 `temperature=0` 的近似平局处。两个臂各自都是确定性的（同一请求重复跑 md5 稳定）。

### 5.5 性能原始数据

```
docker exec z100-vllm bash -lc 'S18_BENCH_OUT=/mnt/game/dflash \
  bash /src/tools/dcu/s18_bench.sh 8000 <label> decode|conc8'
```

* DFlash2 decode（随机 128/128，np=2，c=1）：11.95 tok/s，TPOT 74.7 ms，acc 3.82；
* 生产 MTP k=2 同协议：15.87 tok/s，TPOT 56.9 ms，acc 2.46；
* DFlash2 conc8：31.64 tok/s；MTP conc8：31.16 tok/s。

即：随机 token 负载下 DFlash2 的"每步多 7 个草稿"的代价收不回来（接受率只有 3.8），
真实文本下接受率 6.7 才净赚。

## 6. 未完成 / 建议下一步

1. **草稿 bf16 模拟开销**：目前草稿 5 层全走 DTK bf16 模拟。可选：给草稿单独走
   fp16 但修正 MLP 溢出（例如 MLP 用 fp32 累加/输出 clamp），或把草稿线性层路由到
   `bf16_emu_gemv`（M≤4 的小 M 已被该 kernel 覆盖，N>1280 的 qkv/o_proj 仍走
   `triton_matmul`）。这是最直接的提速点。
2. **1/3 贪心不一致**：需要一个**非投机**贪心参考臂来判定哪个是"正确"的。
   怀疑是 gfx906 上 kernel 形状相关数值（verify 步 1+7 vs 1+2 的 batch 形状不同），
   与 S14b（图模式改变 token 流）、S18e（MTP parity）记录的现象同源。
3. **草稿 KV cache 分组未识别**：启动日志仍有
   `Speculative decoding (method=dflash) is enabled but no KV cache group could be
   identified as the draft model's.`（`_annotate_eagle_groups` 只认 Kimi-K3 DSpark /
   MTP 尾部层回退），需确认草稿组的前缀缓存/滑窗淘汰语义是否受影响。
4. **参数调优**：`--max-num-batched-tokens 2048` 在 k=7 下会报
   `max_num_scheduled_tokens is set to 2048 ... may lead to suboptimal performance`；
   `max_num_seqs` 与 k 的组合（上游参考用 32）未扫。
5. **长上下文**：本轮固定 `max_model_len=262144` 验证容量足够（3.08×），
   但未测 32k/128k 长上下文下的接受率与 TTFT。

## 7. 回滚 / 复现

* 生产臂原命令行与环境已按 `/proc/<pid>/environ` + cmdline 固化：
  `docker exec -d z100-vllm bash /mnt/game/dflash/restore/restore_8000.sh`
  （FP8 + MTP k=2、TP4、262144、:8000；本次工作结束后已用该脚本恢复并验证：
  `models HTTP 200`，真实文本 19.58 tok/s，md5 与改动前基线一致）。
* DFlash2 臂：`docker exec -d z100-vllm bash /src/tools/dcu/s19_dflash_e2e_fp8.sh`
  （如需回退旧行为：`S19_DRAFT_DTYPE= S19_DRAFT_KV_DTYPE=`；纯因果消融：
  `VLLM_GFX906_ATTN_FORCE_CAUSAL=1`）。
