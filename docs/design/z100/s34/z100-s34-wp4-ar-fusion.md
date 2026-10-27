# S34 WP4 设计 + 施工图：rms_norm / quant 融合进 push AR 核（2026-10-27）

> 状态：**设计 + 施工图交付，实施留裁决**（需 csrc 重编译，走 build-sync.sh）。
> 上游依据：S33-WP0 分账（真实核间空隙 ≈3.8 ms/round，~1000 处 × 5 µs）；
> S33-WP3 停点（两个融合落点分别随 WP2/WP1 停，现 WP2 push AR 已采纳，
> 本 WP 的 AR 侧落点解锁）。

## 1. 靶子与预期

- TP4 每层 2 次 allreduce（attn-out、MLP-out），64 层 = 128 次/step；
  每次 AR 出口紧跟 fused_add_rms_norm（attn 侧）或 add_rms_norm + fp8 quant
  （MLP 侧，量化在下一 GEMV 前）。这两个 norm/quant 小核每次 5~22 µs，
  加上 AR→norm 之间的核间空隙（~5 µs × 128 ≈ 0.6~1.2 ms/round 的纯空隙）。
- 预期：把 add-rmsnorm（以及 MLP 侧的 fp8 act-quant）并进 push AR 核的
  出口路径，per-op −20% 判据口径（S33-WP3 原判据），e2e 门 −0.8 ms/round。

## 2. 内核草图（`csrc/custom_all_reduce.cuh` / `cross_device_reduce_push`）

现 push 核五步：`barrier_at_start` → push 写 peer region →
`barrier_mid_volatile` → 本地归约（rank 0..3 序 fp32 累加，对角直读 src）→
`barrier_at_end<true>`。融合 = 在第 4 步归约写出前就地做 epilogue：

```
// 每线程负责 stride 循环的元素 i
float s = (float)src[i]                       // 自己的贡献（对角）
        + (float)peer_buf[0..3 excl rank][i]; // 其余 rank region（本地读）
s += (float)residual[i];                      // 融合 add（可选）
out[i] = __float2half(s * rrms[i]);           // 融合 rms-norm（行内 RS 除再乘）
// MLP 侧变体：out_fp8[i] = quant_e4m3(s * rrms[i])，scale 另出
```

- 归约序**不变**（仍 rank 0..3 fp32 累加）→ 数值与现 push AR 位同的前提保留；
  norm 的 rsqrt(mean) 需行内先归约 Σx²（新增一次 block 级 reduce，fp32），
  这会引入一次行内广播——位同性需微测 T=1/T=4 双档确认。
- 行长度 = hidden 5120（fp16）= 10 KiB，正好在 push 核单 block 覆盖范围内；
  residual/weight 指针由 host 侧 `all_reduce(..., residual=, norm_weight=)`
  新签名传入（graph 捕获下指针烘焙，与现 push 核同机制）。

## 3. 集成点（host 侧）

- `vllm/distributed/device_communicators/custom_all_reduce.py`：`all_reduce`
  增加可选 `residual`/`norm_weight`/`quant_out` 参数 → python 门
  （新 env `VLLM_DCU_CUSTOM_AR_FUSION=1`，默认关）。
- 模型侧调用点：`vllm/model_executor/layers/` 的 communicator 调用封装，
  把 `fused_add_rms_norm` 从 AR 后的独立 kernel 换成 AR 融合出口；
  仅 decode（M=1..4）走融合，prefill 保持现路径（payload 大、走 RCCL）。
- 调度安全：融合核仍在一个 piecewise graph 段内；capture 路径指针烘焙
  与 push 核现行为一致（`_capture_registered=False` 路径复用）。

## 4. 位同风险与门

| 风险 | 缓解 |
|---|---|
| rsqrt 行内归约引入新浮点序 | 微测 T=1/T=4 对拍现路径（max|Δ| ≤ 1 ULP 目标；不可位同则走"数值门+eval"裁决） |
| graph 捕获下新增指针参数 | 复用 push 核捕获机制，`cmp_capture` 贪心流零翻转门 |
| 与 size 门叠加的载荷边界 | 融合只在 ≤64 KiB（现役默认门）内激活 |

验收门：微测 per-op −20%；e2e（7469 doc / 96 tok / 3 reps）step p50
−0.8 ms；流零翻转（nospec 口径）或 in-session 位同（MTP 口径）；30 min 长跑。

## 5. 实施顺序与回滚

1. build：`docker exec -e MAX_JOBS=5 z100-regress bash -lc 'bash $DATA_ROOT/port/build-sync.sh _C_stable_libtorch'`（注意 push 核在 `_C_stable_libtorch.abi3.so`）。
2. 单臂 A/B：`VLLM_DCU_CUSTOM_AR_FUSION=1` vs 不设，env 一键回滚；
   二进制 cp bak 后再回滚。
3. 若位同不可达：转"数值板 + top5 logprob 偏移报表 + eval"裁决口径
   （先例：S29-BM16 0.0213 nat 门）。
