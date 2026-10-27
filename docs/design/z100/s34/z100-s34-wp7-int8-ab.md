# S34 WP7（约束外）：INT8 sdot4 W8A8 速度 A/B —— 预案（2026-10-27）

> 状态：**只测速度与产出 diff 板，不采纳**（改动权重精度 → 需 eval +
> 用户裁决，用户口径"约束外后续跟进"）。
> 依据：`Qwen3.8-27B-Channel-INT8-w8a8` checkpoint 已在机器上（29G，
> compression_config：weights per-channel symmetric int8 + activations
> dynamic per-token symmetric int8，ignore 覆盖 norm/embed/mtp.pre_fc_norm，
> MTP 层已量化）——与 `gfx906_sdot4` 内核契约完全匹配
> （`vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py`）。
> envs.py 注释先例："production opts in with '1'（3.9× decode）"；
> S11l/S11m：启用会改贪心 token（数值性，非调度性）。

## 1. 动机与预期

- 现役 fp8 emu GEMV：256 次/step × 48.8 µs ≈ 12.5 ms/step（40.6% 占比）。
- INT8 原生 `v_dot4_i32_i8` 3.9× → GEMV ≈ 3.2 ms → **step −9 ms 级**，
  MTP 臂 7.5k 口径 step 49.4 → ~40 ms（ms/tok 20.6 → ~16.7，+23%）。
- 权重显存：int8 与 fp8 同为 1 B/参数 → KV 池不变。

## 2. A/B 步骤（一次重启窗口内完成）

```bash
# 臂 = 生产 MTP 臂 + 模型换 INT8 checkpoint + sdot4 开
bash .s34/restart_arm.sh s34-wp7-int8 \
  VLLM_DCU_USE_INT8_SDOT4=1
# 注意：launch_arm.py 的 cmdline 固化了 FP8 模型路径；WP7 需要扩展
# launch_arm.py 支持 --model 覆盖（或手工启动一次），见 §4 待办。
# 探针：7.5k 协议（step p50 / accept_len）+ 60k（fwd t/s）
# diff 板：.s33/capture.py 贪心流 top5 logprobs × 3 rep → 与 FP8 臂对照：
#   - 首分歧位置、96 token 内分歧 token 数
#   - top1 logprob 偏移分布（对照 S29-BM16 先例门 0.0213 nat）
```

## 3. 裁决材料（交付给用户）

| 项 | FP8 现役 | INT8 |
|---|---|---|
| step p50 @7.5k / @60k | 49.4 / 62.4 ms | （待测） |
| fwd t/s | 48.6 / 35.7 | （待测） |
| 贪心流 vs FP8 | — | （待测：首分歧位置/偏移 nat） |
| 质量评测 | — | （未做；采纳前置） |

## 4. 待办（实施前）

1. `launch_arm.py` 增加模型路径覆盖参数（cmdline 里 `$DATA_ROOT/models/Qwen3.8-27B-FP8`
   需替换为 INT8 路径；served-model-name 保持 `qwen3.8-27b` 免改探针）。
2. 确认本 fork 的 config 读取兼容 `compression_config`（transformers v5 风格，
   FP8 checkpoint 用的是 `quantization_config`）——起服日志会立刻给出答案。
3. MTP draft 权重同样走 INT8（ignore 只排除 pre_fc_norm）→ accept-lottery
   必然漂移，diff 板口径按"分布内"判读（S30c §2 先例）。
