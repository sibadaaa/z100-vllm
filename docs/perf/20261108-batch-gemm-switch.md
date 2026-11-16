# 合批为什么几乎不涨：GEMV→GEMM 切换 + GEMV 随 M 线性

> **结论先行**：合批吞吐上不去不是“并发不够”，是**两条 kernel 路径在小 M 下
> 都不省时间**。用 profiler 在在役服务上直接看到了切换点。
>
> **实测时间**：2026-11-08 09:20–09:45 (+08)，生产在役栈（PID 955），
> 4×Z100L，无其他负载。
> **方法**：`/start_profile`、`/stop_profile`（torch profiler，已在 argv 里配好
> `--profiler-config torch,torch_profiler_dir=/mnt/game/s30c-prof`），
> 分别跑 B=1/2/4 各一轮 80 token 的 decode，读 rank0 的 kernel 名与 `dur`。
> **原始 trace**：`/mnt/game/s30c-prof/rank0.1794100{694935314361,712779433904,
> 765346668765}.pt.trace.json.gz`（B=1/2/4）。
> **上层结论**：[SOL-DECODE-PREFILL](SOL-DECODE-PREFILL-20261107.md) §2.3 / §5.2。

---

## 0. 三句话

1. **B=1 和 B=2 走 GEMV**（`fp8_emu_gemv_stream_block_kernel`），
   **B=4 走 GEMM**（`fp8_emu_gemm_block_w8a16_kernel<32,32,64>`）。
   切换点是 `M = B × (1+k) > VLLM_DCU_FP8_GEMV_MAX_M = 8`，
   即 **B=4 时 M=12 越过 8**。
2. **GEMV 的成本随 M 线性涨**：每次调用 78.6 µs(M=3) → 139.1 µs(M=6)，
   **+77% 换 2× 的 batch**。这就是 B=1→2 步时涨 1.66× 的原因。
3. **GEMM 每次都贵 4.6×**：切换后每次调用 **364.0 µs**，虽然它对 M 平坦，
   但**起点太高**，B=4 时 49.9% 的 kernel 时间在它身上。

**所以合批的两条路都堵**：GEMV 路随 M 线性，GEMM 路单价高。
真正的修法不是调 B，是**让 GEMV 在 M=3..12 时能跨请求复用一份权重读**。

---

## 1. 直接证据：kernel 名在 B=4 时变了

同一份服务、同一份代码、同一个 ctx=512（KV 可忽略），只改并发：

| B | `gemv_stream` 调用 | `gemm_w8a16` 调用 | 走哪条 |
|---:|---:|---:|---|
| 1 | **9546** | 0 | **GEMV** |
| 2 | 9774 | 0 | **GEMV** |
| 4 | 662 | **8320** | **GEMM** |

> B=4 时残留的 662 次 `gemv_stream` 是 MTP draft 那一路（M=3，独立于主模型
> B×3=12），不是主路径。

**代码里的阈值**（`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py:82`）：

```python
_GEMV_MS_MAX = 8
_GEMV_MAX_M = min(max(envs.VLLM_DCU_FP8_GEMV_MAX_M, 0), _GEMV_MS_MAX)
...
if a.shape[0] <= _GEMV_MAX_M:      # line 314 / 349
    return torch.ops._rocm_C.fp8_emu_gemv(...)
return torch.ops._rocm_C.fp8_emu_gemm_block(...)
```

生产 env：`VLLM_DCU_FP8_GEMV_MAX_M=8`，MTP `num_speculative_tokens=2`
⇒ 每个请求的 M = 1+2 = 3：

```
B=1 → M=3  ≤ 8  → GEMV
B=2 → M=6  ≤ 8  → GEMV
B=4 → M=12  > 8  → GEMM   ← 切换点
```

---

## 2. 两个路径的单次成本（profile 实测）

### 2.1 B=1（GEMV，M=3）

| 类别 | 调用 | kernel 时间 | 占比 | µs/次 |
|---|---:|---:|---:|---:|
| other（elementwise/triton/norm…） | 33277 | 896.0 ms | 48.9% | 26.9 |
| **gemv_stream** | **9546** | **750.4 ms** | **40.9%** | **78.6** |
| allreduce（push + push_fused） | 4863 | 96.0 ms | 5.2% | 19.7 |
| gemv_dot2_bf16 | 1838 | 90.7 ms | 4.9% | 49.3 |

### 2.2 B=2（GEMV，M=6）

| 类别 | 调用 | kernel 时间 | 占比 | µs/次 |
|---|---:|---:|---:|---:|
| other | 34737 | 1436.0 ms | 46.6% | 41.3 |
| **gemv_stream** | **9774** | **1359.5 ms** | **44.1%** | **139.1** |
| gemv_dot2_bf16 | 1891 | 159.5 ms | 5.2% | 84.4 |
| allreduce | 4998 | 126.4 ms | 4.1% | 25.3 |

**GEMV 调用次数几乎没变（9546 → 9774，+2.4%），但每次贵了 77%**
⇒ **2× 的 batch 换来的不是 2× 的并行，而是 1.77× 的单价**。
净效果：kernel 时间 750 → 1360 ms（1.81×），与实测步时 1.66× 吻合。

> 这与代码注释里 S18 的结论一致：
> “GEMV cost grows ~linearly with M (M=2 ~2.1x, M=4 ~3.9x)”。

### 2.3 B=4（GEMM，M=12）

| 类别 | 调用 | kernel 时间 | 占比 | µs/次 |
|---|---:|---:|---:|---:|
| **gemm_w8a16** | **8320** | **3028.4 ms** | **49.9%** | **364.0** |
| other | 34953 | 2815.4 ms | 46.4% | 80.5 |
| allreduce | 4593 | 162.8 ms | 2.7% | 35.4 |
| gemv_stream（draft） | 662 | 52.9 ms | 0.9% | 79.9 |

**GEMM 对 M 平坦（这正是注释里选它的理由），但单价 364 µs 是 GEMV@M=3 的 4.6×。**
B=4 的总 kernel 时间是 B=1 的 **3.31×**（6070 vs 1833 ms），batch 涨 4×
⇒ kernel 时间**基本是“按 batch 线性放大”**，不是共享。

**按每 token 归一化后收益几乎为零**：

| | kernel ms / 生成 token |
|---|---:|
| B=1（38 tok） | 48.2 |
| B=4（132 tok） | **46.0** |
| 改善 | **4.6%** |

⇒ **合批在 kernel 层面只买到 4.6% 的效率**，这解释了为什么聚合吞吐
几乎不随 B 动（65–80 t/s）。

> **`other` 那 46–49% 不能忽略**：它在 B=1 时 896 ms、B=4 时 2815 ms，
> 也是随 batch 涨的（26.9 → 80.5 µs/次）。这部分是 elementwise / triton /
> rmsnorm / silu 等，**说明小算子也在按 batch 重跑**，不是只有 GEMM 有问题。
> 下一轮该拆 `other`。

---

## 3. 为什么 `VLLM_DCU_FP8_GEMV_MAX_M=8` 这个默认值对生产不利

代码注释说得很清楚，这个阈值的**原始论证对象是 MTP verify 的 M=3**：

> “speculative-decoding verify (M = 1 + num_spec_tokens) wants the GEMM path.
>  Default keeps the old policy; set `VLLM_DCU_FP8_GEMV_MAX_M=1` to route every
>  M >= 2 to the GEMM.”

也就是说：**作者认为 M≥2 就该走 GEMM**（因为 GEMM 对 M 平坦）。
但那个结论是在**单请求**（B=1，M=3）下得出的，而**合批会把 M 推到 12**——
这正是“M 平坦”最占优势的区间，可 GEMM 的绝对单价（364 µs）又太高。

**所以这里有一个没被验证的中间地带**：M = 9–12 时，
GEMV（按 M 线性外推 ≈ 78.6 × 4 = 314 µs）和 GEMM（364 µs）**其实很接近**。
默认阈值 8 把 B=4 推到 GEMM，但如果 GEMV 在 M=12 只涨到 ~300 µs，
**保持在 GEMV 可能更快**。这是一个**一次 AB 就能测清**的问题。

---

## 4. 三条可执行的修法（按性价比）

### W1（最低风险，先做）：`VLLM_DCU_FP8_GEMV_MAX_M` 扫描

不改代码，只改 env。在 M=6/9/12（B=2/3/4）上各跑一次 AB：

```bash
# 同一窗口、同一 prompt、同一 ctx，只改阈值
for m in 8 12 16; do
  # 重启服务（带 VLLM_DCU_FP8_GEMV_MAX_M=$m），跑 B=4
  .venv/bin/python tools/dcu/s24_longctx.py --lens 512 --reps 3 --gen 96 --tag gemvmax$m
done
# 同时抓 profile 确认走的是 gemv 还是 gemm
```

**判据**：看 ITL。如果 M=12 走 GEMV 比走 GEMM 快，就把阈值提到 16
（注意 `_GEMV_MS_MAX=8` 是**编译期**上限，env 会被 `min(...,8)` 夹住，
所以**提阈值需要重编译 `fp8_emu_gemm.cu` 并把 `kMsMax` 提到 16**）。

> ⚠️ **这是我读代码发现的硬约束**：`_GEMV_MAX_M = min(env, _GEMV_MS_MAX=8)`，
> 所以**光设 env 无效**，M=12 永远走 GEMM。要测这个中间地带必须重编译。
> 先做 W1 的 profile 部分（零成本确认现状），再决定是否值得一次编译。

### W2（真正的解法）：让 GEMV 在 M=3..12 跨请求复用权重

现在 GEMV 的 `M` 是**整个 batch 拼起来的**（B×3=12），kernel 内部对每个 m
各读一遍权重 ⇒ 单价随 M 线性。**改成“一次读权重、寄存器里对 M 个 m 同时累加”**
就能把 M=12 的单价压到接近 M=3。

这正是 `GEMV-OPT-20261025.md` 里 `lds_*` 系列做过的事：
`SLICE=512` 时 M=8 的 318 GB/s vs `SLICE=1024` 的 425 GB/s，
以及 M=4 的 630 / M=1 的 730 —— **A 的复用方式决定单价**。
把这条经验从“单请求内 M”推广到“跨请求 M”是 W2 的核心。

### W3（并行做）：拆 `other` 那 46–49%

`other` 在 B=1 已占 48.9%，B=4 占 46.4%，**且单价从 26.9 涨到 80.5 µs/次**。
它包含 elementwise / triton poi / rmsnorm / silu / quant。
**这些算子的 K 都等于 hidden 5120，天然可以按 B 合批**，
单价不该随 B 线性涨。**W3 可能比 W1/W2 更值钱，因为它的基数更大。**

---

## 5. 这一轮的净结论

| 问题 | 11-07 的猜测 | 实测 | 证据 |
|---|---|---|---|
| 合批为什么不涨 | “撞算力墙需 24 路” | **B=4 就 3.5× 损失** | ITL 35.61→127.08 ms @ctx512 |
| 是否共享权重 | 假设共享 | **没有**：边际 30.5 ms/路 | `step(B)=5.1+30.5B` |
| 合批收益 | 预期 ×B | **每 token 只改善 4.6%** | 48.2 → 46.0 ms/token |
| 根因 | 未查 | **GEMV 随 M 线性 + M>8 切到高价 GEMM** | kernel 名/计数/单价三组数据 |
| 小算子 | 未查 | **`other` 占 46–49% 且随 B 涨** | 同 trace |

**一句话**：合批在当前实现下基本是“把同样多的活摊给更多路”，
**聚合吞吐 65–80 t/s 几乎不随 B 变**。要让它变，必须先让**权重读**和
**小算子**在 batch 内共享 —— 这两件事都不需要新硬件。
