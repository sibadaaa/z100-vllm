# S24 L3.2 设计：长上下文动态降 k（speculative verify 截断）

> 状态：设计稿（W3）。实现窗口未排期；本文给出机制、实现面、门禁与回滚。
> 关联：`docs/design/z100-longctx-vram-optimization-plan.md` §L3、`.s22/S22_LONGCTX_DECODE.md`。

## 1. 机制与依据

* DFlash2 k=7：每个 decode step 的 verify 把 M=1+k=8 个 query 送去扫 KV。
  S22 实测 verify 的单步 KV 扫描是裸 decode 的 2–3×；而长上下文下接受长度
  随机文本 1.45–2.31、真实文本从 7.0 掉到 2.3–4.5 ⇒ **每“有用 token”的 KV
  成本被放大 (1+k)/accept 倍**。k 是净放大项。
* **关键事实：截断 k 不改贪心输出**。verify 的产出 = target 模型自己的贪心
  token 前缀；draft 只决定一次 step 验证多少个位置。把每步的 k' ≤ k（draft
  仍然提出 7 个，verify 只算前 1+k' 个位置），发射的 token 序列不变，只是
  步数可能变多 ⇒ **greedy parity 按构造保持**（与 O4 的数值路径改动不同类）。
* 收益口径：verify 的 KV 扫描 ∝ (1+k')；k'=2 时 M=8→3，长上下文每步 KV 读
  约 −60%，而接受损失只影响步数（真实文本 accept≈7 时不应触发降 k）。

## 2. 策略（闭环用 per-request acceptance）

```
k'(L, accept_ema):
  L ≤ 16k            → 7（现网默认，短上下文不动）
  16k < L ≤ 64k      → accept_ema ≥ 5 ? 7 : 3
  L > 64k            → accept_ema ≥ 5 ? 3 : 1
accept_ema ← 0.9·accept_ema + 0.1·accept_len(本步)
```

阈值用 `--per-request-spec-decode-metrics summary` 已有的 acceptance 长度闭环；
三个阈值与 ema 系数走 env（`VLLM_S24_DYNK_LO=16384`,
`VLLM_S24_DYNK_HI=65536`, `VLLM_S24_DYNK_EMA=0.1`），未设 = 关闭（现网行为）。

## 3. 实现面（估计 ~80 行）

1. `vllm/v1/worker/gpu_model_runner.py`：verify 调度处按请求计算 k'，把
   spec decode metadata 的 `num_speculative_tokens`（per-step, per-request）
   截断到 k'——DFlash2 drafter 照常出 7 个 token，只截 verify 长度。
2. 接受长度已在 per-request metrics 里暴露，ema 状态挂在 request 对象。
3. 写域：`vllm/v1/worker/gpu_model_runner.py` + DFlash2 drafter 的 metadata
   构造（只读）。默认关闭（env 缺省 = k'=k）。

## 4. 门禁

1. **parity**：固定 8 prompt greedy 逐 token 一致（k 截断不改输出，应全绿）。
2. **ITL**：32k 真实文本 ITL ≤150 ms（计划目标）；短上下文（≤4k）不回归。
3. **步数代价**：输出 t/s 不低于对照臂的 95%（降 k 增加步数）。
4. 同窗口 A/B、≥4 probe、canary、树指纹（S24 纪律）。

## 5. 风险

* per-step k' 使 verify 的 M 变化 → 触发 GEMV/GEMM 路径切换的形状增加
  （M=3/4 等），需要一次 `VLLM_DCU_FP8_GEMV_MAX_M` 边界复查（M≤8 全走
  GEMV/stream 路径，不受影响——S24 §2.1 已确认 M≤8 的 cfg 通道不可达）。
* 与 prefix caching 的交互：无（KV 扫描长度只由已确认上下文决定）。
