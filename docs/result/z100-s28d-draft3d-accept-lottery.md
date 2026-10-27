# S28d：draft 形状接入 3D split-KV + accept 重启彩票的发现（2026-10-25 下午）

> **改动**：`vllm/v1/attention/ops/triton_unified_attention.py` —— `_decode_block_m`
> 增加 draft 形状映射 `(head 128, GQA4) → BLOCK_M 16`（=4 token×4 行，无 padding），
> 使 DFlash2 draft 塔的 7-token block forward 走 verify-3D 同款 small-q→3D split-KV 路由；
> kill switch `VLLM_GFX906_ATTN_VERIFY_3D_DRAFT=0`（默认 1）。
> **背景**：graph-A trace 里 draft attention 是 unsplit 2D、grid (2,2,1)——4 个 CTA
> 串行扫 2048 窗口，1234 µs/次 × 5 层（与 verify-3D 修掉的病同源，只是发生在 draft 形状上）。

## 1. 服务级裁决（7469 真实文档、3 reps、 patched 树）

| 臂 | 重启 | step p50 | tok/step | accept |
|---|---|---:|---:|---:|
| control（DRAFT=0） | 2 | 101.35 / 101.37 | 2.667 | 1.771 |
| **draft-ON** | 2 | **93.14 / 93.14** | 2.667 | 1.771 |

**−8.2 ms（−8.1%），accept 同吸引子，采纳。** step 跨重启方差 <0.1 ms；
（det1 前序点 101.23 为 reverted 树，行为等价。）

**parity**（439 prompt × 64 tok × 2 reps，vs 早晨 control）：token 流 64/64 逐位相同，
top-1 logprob 最大漂移 0.0759 nat（A1 是 0.0236；均远低于 verify-3D 披露过的 0.80 近平局类），
无翻位。短 prompt accept_rate 0.7571 与所有臂一致。

## 2. 重大发现：accept 存在"重启彩票"（±6~10% t/s）

同树（仅差 metadata/测试）多次重启的 accept/tok-per-step：

| 服务重启 | 树状态 | accept | tok/step |
|---|---|---:|---:|
| 早晨 arm1（NARROW=4） | A1 前 | 1.96875 | 2.909 |
| 早晨 arm2（A1） | A1 后 | 1.96875 | 2.909 |
| arm3（+SMALLM） | | 1.7714 | 2.667 |
| 午后 s28b-control | +draft 补丁（DRAFT=0） | 1.7714 | 2.667 |
| 午后 revert-check | **attention 还原** | **1.8235** | **2.743** |
| 午后 det1（确定性 workspace） | attention 还原 | 1.7714 | 2.667 |

- **revert-check 与早晨 arm2 有效代码等价，accept 却差 −7.4%**（1.8235 vs 1.96875）；
  det1 的 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 没有锁住。
- 推论：**accept 不是重启稳定量**，跨重启的 tok/s 对比被 ±6~10% 的彩票污染。
  受影响的历史结论需要复核——包括本日 arm3 的"accept −8.3%"否决
  （1.7714 落在午后对照吸引子集合 {1.7714, 1.8235} 内，**SMALLM 的 accept 惩罚证据不足**，
  其 step −4.65 ms 仍真实）。S24 dynk 等 accept 类 A/B 同样受影响。
- **step p50 跨重启稳定**（本日全部 101.2–101.4 / 93.14±0.01），step 类结论不受影响。
- 根因候选（未定）：per-restart 的 Triton/Inductor autotune 时序抽奖、piecewise graph
  捕获次序、rocBLAS algo 选择。**下一个方法论工作包**：定位并钉死（如固定 autotune、
  缓存调优结果到盘），否则一切 accept 类对比必须 N≥2 重启取分布。

## 3. 今日累计（step p50，同口径）

| 里程碑 | step p50 | Δ | 服务端 fwd t/s（当日 accept 口径） |
|---|---:|---:|---:|
| 早晨 control | 108.85 | — | 26.7 |
| +A1 ba GEMV | 101.11 | −7.74 | 28.8（accept 2.909 吸引子） |
| +draft 3D 路由 | **93.14** | −8.20 | 28.6（accept 1.771 吸引子） |

**注意**：t/s 的跨臂比较被 accept 彩票污染——A1 臂落在 2.909 吸引子、draft 臂落在 2.667。
若 accept 吸引子可钉在 2.909，draft 臂的 t/s 应为 **31.2**（=2.909/0.0931）——
钉住 attractor 本身是比继续抠 kernel 更大的一项。

## 4. 状态与遗留

- 最终服务停在 **draft-ON 臂**（:8000，`serve-s28d-draft2.log`，默认配置）。
- 测试：`tests/kernels/attention/test_triton_unified_attn_gfx906_v1.py` 20 passed
  （新增 draft 路由 3 项：3D+滑窗精确性、kill switch、block 映射）；
  1 个存量失败 `test_v1_unified_attention_prefill_split_kv_matches_reference`
  （改动前即失败，`_prefill_segm_state` 断言过期，单独修）。
- `dcu_policy_fingerprint.py`：新增 `VLLM_DCU_GEMV_NARROW_MAX_M`、
  `VLLM_GFX906_ATTN_VERIFY_3D_DRAFT`，selftest 42/42。
- 遗留队列：**accept 彩票根因（升为最高优先）** → SMALLM 复核（2 重启/臂）→
  A2 BM16 → W4 探针。

原始数据：`.s28/s20_s28d-*.json`、`.s28/serve-s28*.log`、
`.s28/e2e-draft-on.json/`、`.s28/parity-draft-vs-arm1.json`。
