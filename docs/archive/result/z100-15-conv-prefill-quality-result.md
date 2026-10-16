# S15: 27B W8A8 INT8「提速后智能下降」根因定位与修复 — 执行结果

> 归档时间：2026-10-10
> 执行者：opencode/DSH 会话（用户指令：允许编译 HIP；使用 llm_speedtest 测试；不恢复旧服务）
> 状态：✅ 主修复落地并验证；残留风险见 §5

## 1. 目标与验收标准

用户报告：本树（gfx906 / DTK 26.04）让 Qwen3.8-27B W8A8 INT8 prefill/decode 变快后，
**eager 与 graph 两种模式下智能均严重下降**（重复循环、中英混杂、空回复/早停、工具调用失败），
对照基线是旧生产容器 `qwen38-vllm`（vendor 0.18.1+das + torch GDN/attention 补丁 + sdot4 + gdn_decode，bf16）。

验收：定位系统性根因 → 修复 → 同一质量电池下恢复到与生成正确性自洽的水平，速度损失可接受。

## 2. 结论摘要（一句话）

**根因不是任何内核算错**，而是 gfx906 的 prefill 快路径会**确定性地把模型推向一个灾难性退化吸引子**：

- `causal-conv1d` 的 gfx906 Triton prefill 内核与 torch fallback **逐位一致**（12/12 服务内实测对拍），
  但只要它处于启用状态，27B 引擎在 **4/4 次启动**中进入"思考循环、永不输出正文"的退化模式；
  关闭它后 **0/5 次**出现该模式（Q2/Q3/Q3b/Q3d/Q5）。
- 因此把 `VLLM_GFX906_TRITON_CAUSAL_CONV1D` 默认改为 **0**（走 fallback），质量电池恢复
  （工具 6/6，推理 4/6，中文短文正常），prefill 395.8 t/s（vs 内核路径 424 t/s，仅 −6.6%）。

## 3. 实际改动

| 文件 | 位置 | 改动 |
|---|---|---|
| `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` | `ENABLE_GFX906_TRITON_CAUSAL_CONV1D`（约 :26-38） | 默认值 `"1"` → `"0"`，附根因与代价注释；同步到 canonical 副本 `/mnt/game/port/gfx906-vllm-211/...` |
| `tools/dcu/qfix_arm_env.sh` | 新增 | 实验启动器（= `s14_arm_env.sh` + `S14_EXTRA_ENVS`/`S14_DTYPE` 注入） |
| `tools/dcu/qfix_*`（在 `/mnt/game/qfix/`） | 新增 | 质量电池、llm_speedtest 驱动、conv 落盘 shim、离线对拍脚本 |

## 4. 执行过程与原始证据

### 4.1 复现与方法

- 影子容器 `z100-s14`（`HostConfig.Memory=0`），TP=4，`--dtype float16`，`VLLM_DCU_USE_INT8_SDOT4=1`，
  `--enable-prefix-caching --mamba-cache-mode align`，agent 三件套 + `reasoning_effort=low`
  （= 用户 `serve_z100_tp4.sh` 口径），统一 A/B 只改一个变量。
- 质量电池（greedy，固定批组合）：6 条工具调用（`tool_choice:auto` 解析）+ 6 条可验证推理
  + 中文短文（检查 content 是否输出、是否陷入思考循环）。
- 速度：`/home/lyh/server/llm_speedtest`（同 2048/128 c=1 口径）。

### 4.2 实验矩阵（raw：`/mnt/game/qfix/runs/`、`hostlogs/`）

| 臂 | 关键 env | 引擎是否起来 | 短文 probe（关键判据） | 电池 |
|---|---|---|---|---|
| Q0/Q0b/Q0g | 默认（conv Triton=1） | ✅ | **思考循环 571 字符、content 空**（三次逐字一致） | tool 6/6，reason 2/6，lang 0.00 |
| Q1 | `--dtype bfloat16` | ✅（455 s，bf16 Triton 全量编译） | **乱码**（"用户\n-1 Kron..."） | 中止，见 §5 |
| Q2 | `VLLM_GFX906_GDN_TRITON_PREFILL=0` + conv=0 | ✅ | 正常收尾（78 token） | reason 4/6，lang 0.78 |
| Q3/Q3b/Q3d | 仅 conv=0 | ✅ | 输出正文，偶见"重写草稿"（非恶性） | tool 6/6，reason 4/6，lang 0.74（两次一致） |
| Q4 | 仅 GDN=0（conv=1） | ✅ | **思考循环** | — |
| Q5 | conv 默认=0（本修复） | ✅ | 输出正文 | tool 6/6（greedy）·4/4（temp 0.7 采样）·reason 4/6 |

- 速度（llm_speedtest 2048/128 c=1）：Q0 **prefill 424.7 / decode 6.71 t/s**；
  Q5（修复）**prefill 395.8 / decode 7.44 t/s**；Q3d 复测 394.0（证伪了 Q3b 单次 158 的污染值）。

### 4.3 内核级对拍（证明 conv Triton 数学正确）

- 独立探针（8 组：两种布局 × has_init × 多处 seqlen/batch）：Triton vs fallback
  `out max ≤ 9.8e-4`（fp16 1 ulp），**state max = 0.0**。
- 服务内 dump（`zz_qfix_convshim`，真实引擎张量落盘 → 离线回放）：
  **12/12 call 回放逐位一致（replay=0.00e+00），fallback vs dump ≤ 1.5e-4，state 0.0**。
  ⇒ 退化不是 conv 数值造成，而是该路径的间接副作用（布局/内核调度 → 后续 Triton 特化变体）。

### 4.4 附带发现（未修）

1. **eager 也存在"启动变体"**：同配置 conv=0 的 Q2 与 Q3/Q3b/Q3d 文本不同（S14g 在 graph 上
   观察到的 allocator→特化变体现象，在 eager 也存在），只是循环吸引子与 conv Triton 强相关。
2. **bf16 在 27B 上输出乱码**（Q1）：与"gfx906 packed-decode 内核按 fp16 元素尺寸读取
   A_log/dt_bias"的隐患一致（`gdn_linear_attn.py:367-374` 注释），属独立 bug。
3. `causal_conv1d` Triton 的 GPU 监控环 wedge 历史（模块注释）未复现，但禁用后该类风险一并消失。

## 5. 门禁判定与残留风险

- ✅ 主症状（思考循环/空正文、语言退化）在 eager 下消除（4/4 → 0/5）。
- ✅ 工具调用在 greedy 与采样下均正常；中文输出正常。
- ✅ 速度损失小：prefill −6.6%，decode 噪声内（Q0 6.71 vs Q5 7.44 t/s，受 prompt/抖动影响）。
- ⚠️ **创作类 prompt 在 temp 0.7 采样下仍有约 2/4 的"草稿重写"循环**（rp=1.0 亦如此）；
  旧栈同 prompt 无数据（用户要求不恢复旧服务），需后续用 DSH 真实 prompt 对拍判定是否模型本身属性。
- ⚠️ **graph 仍不可复现**（S14d/S14e/S14g 结论不变）：质量/agent 口径建议 eager；
  graph 只用于延迟/吞吐测量。
- ⚠️ 吸引子选择的底层机理未定性（allocator → Triton specialization，S14g 遗留方向）。

## 6. 复现方式

```bash
# 引擎（修复后默认）：容器 z100-s14，日志 /mnt/game/qfix/hostlogs/Q5-default-launch.log
/mnt/game/qfix/launch_arm.sh Q5 4 1 8535 262144 1 /mnt/game/qfix/cache-Q5 \
  /mnt/game/qfix/hostlogs/Q5-default-launch.log --host 0.0.0.0 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --override-generation-config '{"temperature":0.7,"repetition_penalty":1.05}' \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}'

# 质量电池 / 速度 / conv 对拍
python3 /mnt/game/qfix/quality_battery.py --label X --api-url http://172.17.0.2:8535/v1/chat/completions --out /mnt/game/qfix/runs/X.json
PYTHONPATH=/home/lyh/server/llm_speedtest/.pydeps python3 /mnt/game/qfix/run_speedtest.py --label X --api-url .../chat/completions --concurrency 1 --length 2048 --output-length 128 --out ...
# dump 模式：VLLM_QFIX_CONVSHIM_MODE=dump，服务内 /tmp/replay_conv.py 对拍
```

## 7. 参考

- `docs/hygon_z100.md` §9.4、`docs/result/z100-14d-determinism-result.md`（graph 不可复现）、
  `docs/result/z100-14g-variant-pinning-result.md`（变体钉不住）、
  `docs/result/z100-14b-graph-vs-eager-result.md`（per-token INT8 × GDN 递推放大机制）
- 本步证据：`/mnt/game/qfix/{runs,dump,hostlogs}/`、`tools/dcu/qfix_arm_env.sh`
