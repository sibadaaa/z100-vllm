# Z100 FP8 emu 改造施工图：**W8A16（决策 1）+ prefill fp16 物化（决策 2）**

> **读者**：接下来执行本改造的 agent（可能 2–4 个并行）。本文件是唯一施工依据，冲突时以本文件为准。
> **配套**（背景与参考实现分析，不必重复阅读）：
> [`z100-fp8-emu-fp16-conversion-plan.md`](z100-fp8-emu-fp16-conversion-plan.md)（总方案）、
> [`z100-fp16-prefill-workspace-evaluation.md`](z100-fp16-prefill-workspace-evaluation.md)（决策 2 的论证）。
> 参考实现：`/mnt/game/1cat-vllm`（V100/sm70，同为无 fp8 硬件的 W8A16 实现）。
> **约束**：GPU 正被其他工程占用 —— **P0 阶段全部可做且不碰卡；P1 阶段的任何 kernel 执行都要等卡空。**

---

## 0. 三十秒看懂

做完之后，一个线性层的调用形状是：

```
                ┌─ M ≥ min_m[role] 且 fp16 物化可用 ─→ dequant(W)→fp16 scratch ─→ torch.mm(A_fp16, W16ᵀ)
apply_weights ──┤
                ├─ 否则（decode / 小 M）────────────→ W8A16 emu kernel（A fp16 直入，W fp8 常驻）
                └─ W8A16 开关关掉 / 前置不满足 ─────→ 现状 W8A8（逐 bit 不变）
```

- **W8A16**：删掉「激活 fp8 量化 + A 侧 LUT 加宽」，每层少一次 launch；权重仍 fp8 常驻（decode 带宽不增）。
- **prefill fp16 物化**：只对大 M 生效，用一块 **89 MB 级、按层复用的 scratch**，把当前层权重解量化成
  fp16 后交给 rocBLAS；**不做常驻副本**（会吃掉 12.6 GiB，等于 59 万 token 的 KV）。
- 两条改动**共享同一个前置**：激活已经是 fp16。所以 **W8A16 必须先落地**，prefill 路线才能干净实现。

---

## 1. 契约与不变量（所有 agent 都要遵守）

### 1.1 数学

```
W8A16：      out[m,n] = Σ_kb ( Σ_{k∈kb} a[m,k] · w[n,k] ) · w_s[n/block_n, kb]
prefill：    W16[n,k] = fp16( w_q[n,k] · w_s[n/block_n, k/block_k] )，out = A_fp16 · W16ᵀ
```

### 1.2 dtype / 布局

| 对象 | W8A16 路径 | prefill 路径 |
|---|---|---|
| A | `float16 [M,K]`，K 连续 | 同左 |
| W | `uint8` fp8 `[N,K]`，**`stride(1)==1`，`stride(0)` 可能是 `K+256`**（`_maybe_pad_fp8_weight` 的 padding） | 左列的 fp8 |
| W16 | — | `float16 [N,K]` 连续（**保持 `[N,K]` 行序，不做转置**） |
| w_s | `float32 [cdiv(N,block_n), cdiv(K,block_k)]` | 同左 |
| a_s | **不存在** | 不存在 |
| out | `out_dtype`（生产 = fp16），bias 在 `out_dtype` 加 | 同左 |

### 1.3 硬不变量

1. **W 永远不常驻 fp16**；prefill 的 fp16 只有一块按层复用的 scratch。
2. **W8A8 路径逐 bit 不变**：开关关闭时，所有输出必须与改造前完全一致（有专门单测）。
3. A 只接受 `float16`；`bfloat16` 直接报错（gfx906 无 bf16 指令）。
4. 形状约束：`block_k=128 | K`、`K % 8 == 0`、`K % 2 == 0`、`w.stride(0) % 8 == 0`、`N % BN == 0` 由既有 cfg 保证。
5. **每调用不得读 env**：所有开关在 import / `std::call_once` 时读一次。
6. **构建串行**：同一时刻只允许一个 agent 触发 `.so` 重建（见 §7）。

---

## 2. 任务板（并发安全的分工）

| ID | 任务 | write scope（只改这些） | 依赖 | 并行性 |
|---|---|---|---|---|
| **T1** | C++：W8A16 decode（gemv + stream） | `csrc/rocm/fp8_emu_gemm.cu`（**仅 gemv/stream 区块**） | — | 与 T2 同文件 → **串行** |
| **T2** | C++：W8A16 tile / wide2 | `csrc/rocm/fp8_emu_gemm.cu`（**仅 tile 区块**） | T1 合并后 | 串行（接 T1） |
| **T3** | C++：header + bindings | `csrc/rocm/fp8_emu_gemm.h`、`csrc/rocm/torch_bindings.cpp` | T1/T2 的函数名定稿 | 可与 T4 并行（不同文件） |
| **T4** | Python：W8A16 接线 | `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py` | T3 | 串行（文件唯一属主） |
| **T5** | C++：dequant kernel（新文件） | `csrc/rocm/fp8_dequant.cuh` + `csrc/rocm/fp8_dequant.cu`（新建） | — | **完全并行** |
| **T6** | Python：prefill 路由 + workspace（新模块） | `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py`（新建） | T4 的 hook（预留在 T4 里） | 与 T5 并行 |
| **T7** | env 开关（**唯一属主**） | `vllm/envs.py` | 需各任务给参数名 | 串行（一个 agent 独占） |
| **T8** | 单测 | `tests/kernels/quantization/test_gfx906_fp8_emu.py`（追加） | T1–T6 | 可与实现并行（先写测试） |
| **T9** | 集成 + 结果文档 | `docs/result/z100-w8a16-fp16-prefill-result.md`（新建） | 全部 | 最后 |

> **文件唯一属主规则**：`fp8_emu_gemm.cu`、`gfx906_fp8.py`、`envs.py`、bindings 各只有一个属主，
> 其他 agent 只能提「需求清单」，不得直接编辑。新功能一律落在新文件里（T5/T6 就是为此拆分）。

---

## 3. 最终路由矩阵

| 条件 | 路径 | 判断点 |
|---|---|---|
| `VLLM_DCU_USE_FP8_EMU` 关闭 | 不选本 kernel（现状） | `is_supported` |
| W8A16 关闭（`VLLM_DCU_FP8_W8A16=0` 或 op 缺失） | **现状 W8A8**（逐 bit） | `apply_input_quant=False` 不生效 |
| W8A16 开 + `M ≤ VLLM_DCU_FP8_GEMV_MAX_M` | W8A16 gemv（含 stream 变体） | op 内部 M 分支 |
| W8A16 开 + `M > GEMV_MAX_M` + `M < min_m[role]` | W8A16 tile GEMM | python 侧 M 判断 |
| W8A16 开 + `M ≥ min_m[role]` + 物化可用 | dequant → `torch.mm` | `gfx906_fp16_prefill.maybe_run` |
| 上一条任一前置失败（OOM / dtype / stride / 开关） | 回退 W8A16 tile GEMM（**打一次 warning**） | 同上 |

---

## 4. 详细规格

### 4.1 T1 — decode：gemv + streaming

**关键发现：不用写新 kernel。** 现有 `fp8_emu_gemv_block_kernel<MS, T, PRECONV>` 的
`PRECONV=true` 分支本来就是「A 从 fp16 指针成对读」（`fp8_emu_gemm.cu:503-510`）：

```cpp
const uint32_t* __restrict__ arow = a_h2 + (int64_t)m * (K4 * 2);   // K4*2 == K/2 dwords
x0 = arow[2 * i];  x1 = arow[2 * i + 1];
```

改动清单：

1. 模板加 `bool HAS_A_S`（默认 `true`，W8A16 用 `false`）：
   ```cpp
   template <int MS, typename T, bool PRECONV, bool HAS_A_S>
   __global__ void fp8_emu_gemv_block_kernel(...)
   ...
   for (int kb = t; kb < nkb; kb += nt) {
     const float wsv = wsr[kb];
     float s[MS];
     for (int m = 0; m < MS; ++m)
       s[m] = (m < M) ? (HAS_A_S ? a_s[m * nkb + kb] * wsv : wsv) : 0.0f;
   ```
2. W8A16 host 分支：`a_h2 = reinterpret_cast<const uint32_t*>(a.data_ptr())`，
   **不再** 分配 `a_scratch`、**不再** 发射 `fp8_emu_widen_a_kernel`（`fp8_emu_gemm.cu:1858-1867`）。
3. `fp8_emu_gemv_stream_block_kernel`（:601）与 `..._stream2...`（:1037）：
   - A staging（:644-654）从「LUT 加宽」改成 fp16 直拷（`uint4` 16 B/次，`sA` 本来就是 `__half*`）；
   - scale 折叠（:659-666 / :948）改成 `sscale[i] = wsr[kb]`；
   - 具体量：`sA[m*K + k]`，每 warp 处理 `kStreamChunk` 元素，直接 `reinterpret_cast<const uint4*>` 拷贝。
4. dispatch：`dispatch_gemv_block` / `dispatch_gemv_stream*` 各加一个 `w8a16` 版本，或加一个
   `bool a16` 运行期参数（decode 内层热循环**必须**模板化，不要 runtime 分支）。

**验收**：§6 的 `test_block_gemv_w8a16_*` 全绿。

### 4.2 T2 — tile / wide2（prefill 与 M>8 的 decode）

三处改动，全部在 K 维 staging 与 block 边界上，**不碰 MAC 内层**。

**(a) gmem→寄存器**（`fp8_gemm_load_block:1077`）
```cpp
// W8A16: A 每 dword 2 个元素
constexpr int kA16LogKw = 4;        // log2(kFKw) = log2(16)
const uint32_t* ap16 = reinterpret_cast<const uint32_t*>(A) + (int64_t)m0 * (K >> 1) + (k0 >> 1);
for (int u = 0; u < A4_16; ++u) {
  const int idx = tid + u * NT;
  const int r  = idx >> kA16LogKw;
  const int jd = idx & (kFKw - 1);
  uint32_t v = 0;
  if (idx < BM * kFKw && m0 + r < M && (kfull || (k0 + (jd + 1) * 2) <= K))
    v = ap16[(int64_t)r * (K >> 1) + jd];
  dstA[u] = v;
}
// A4_16 = (BM * kFKw + NT - 1) / NT      ← 注意是 kFKw(=16)，不是 kFKw4(=8)
```
**(b) 寄存器→LDS**（`fp8_gemm_store:302`）
```cpp
As[r * kFKp + jd] = src[u];      // 直接写；不要再 2*jd 展开、不要再调 fp8x4_to_f16x2x2
```
**(c) block scale 折叠**（`fp8_gemm_fold_block_scale:1110`）
```cpp
acc[i][j] += blk[i][j] * (HAS_A_S ? sa * w_si : w_si);   // sa 只在 HAS_A_S 时读
```
`wide2` 家族（`KConf` / `fp8_gemm_load_wide:1266`）把 `Kw4 = BK/4` 换成 `Kw = BK/2`，其余同理。

> **编译预算**：**不要**给 cfg 1..40 全量实例化 W8A16 版本。A 侧 staging 在 K 循环外层，可以用
> `const bool a16` 运行期分支（uniform，代价可忽略）；只对实际会用的 cfg（10/16/35/36–40）允许模板化。
> 由 T2 属主决定，并在 PR 描述里写清实例数变化。

### 4.3 T3 — header + bindings

`fp8_emu_gemm.h` 追加：
```cpp
// W8A16: A is fp16 [M,K] (unquantized), W is fp8 [N,K] with fp32 block scales.
// Same kernels as the W8A8 block-scaled family; the A-side LUT widening and
// the per-token activation scale are absent.
torch::Tensor fp8_emu_gemm_block_w8a16(
    torch::Tensor a, torch::Tensor w, torch::Tensor w_s, torch::Tensor lut,
    std::optional<torch::Tensor> bias, at::ScalarType out_dtype, int64_t nthreads,
    int64_t cfg, int64_t block_n, int64_t block_k);
torch::Tensor fp8_emu_gemm_block_w8a16_meta(...);   // 同上签名
```

`torch_bindings.cpp`（`TORCH_LIBRARY_IMPL_EXPAND` 的 CUDA 区块内，:139 附近追加）：
```cpp
rocm_ops.def(
    "fp8_emu_gemm_block_w8a16(Tensor a, Tensor w, Tensor w_s, Tensor lut, "
    "Tensor? bias, ScalarType out_dtype, int nthreads, int cfg, "
    "int block_n, int block_k) -> Tensor");
rocm_ops.impl("fp8_emu_gemm_block_w8a16", torch::kCUDA, &fp8_emu_gemm_block_w8a16);
// Meta 区块：
rocm_meta_ops.impl("fp8_emu_gemm_block_w8a16", &fp8_emu_gemm_block_w8a16_meta);
```

T5 的 dequant op 也在这里注册：
```cpp
rocm_ops.def(
    "fp8_dequant_to_fp16_out(Tensor(a!) out, Tensor wq, Tensor ws, Tensor lut, "
    "int block_n, int block_k) -> ()");
rocm_ops.impl("fp8_dequant_to_fp16_out", torch::kCUDA, &fp8_dequant_to_fp16_out);
```

### 4.4 T4 — Python 接线（`gfx906_fp8.py`，唯一属主）

**4.4.1 开关与能力检查**
```python
_W8A16_MODE = os.environ.get("VLLM_DCU_FP8_W8A16", "0").strip().lower()  # 0|1|auto
_W8A16 = _W8A16_MODE in ("1", "auto")

def ops_available_block_w8a16() -> bool:
    return (hasattr(torch.ops._rocm_C, "fp8_emu_gemm_block_w8a16")
            and hasattr(torch.ops.vllm, "fp8_emu_mm_block_w8a16"))
```

**4.4.2 类属性**
```python
class Gfx906Fp8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    # W8A16: activations arrive fp16 and are NOT quantized (base-class hook,
    # same mechanism as FlashInfer/CPU block kernels).  Base then passes a
    # 1-element placeholder as `As` — this kernel must never read it.
    apply_input_quant = _W8A16 and ops_available_block_w8a16()
```
`is_supported()` 追加：`_W8A16_MODE == "1"` 且 op 缺失 → 返回 `(False, "…w8a16 op missing; rebuild.")`
（显式开启必须报错，`auto` 才允许静默降级）。

**4.4.3 分流**（`apply_block_scaled_mm`，保持现有 W8A8 分支逐字不动）
```python
if A.dtype == torch.float16 and not self.apply_input_quant:
    return _fp8_emu_mm_block_w8a16_impl(
        A, w, Bs.contiguous(), _device_lut(w.dtype, A.device), None,
        self.config.out_dtype, _GEMV_THREADS,
        _prefill_cfg(A.shape[0], w.shape[0]),
        int(self.weight_group_shape.row), int(self.weight_group_shape.col),
    )
if A.dtype == torch.bfloat16:
    raise RuntimeError("gfx906 W8A16 requires fp16 activations (no bf16 ISA).")
# ↓ 原 W8A8 代码，逐字保留
```

**4.4.4 `_fp8_emu_mm_block_w8a16_impl` + custom_op**
```python
def _fp8_emu_mm_block_w8a16_impl(a, w, w_s, lut, bias, out_dtype, nthreads, cfg,
                                 block_n, block_k):
    M = a.shape[0]
    if M <= _GEMV_MAX_M:
        return torch.ops._rocm_C.fp8_emu_gemm_block_w8a16(
            a, w, w_s, lut, bias, out_dtype, nthreads, 0, block_n, block_k)
    return torch.ops._rocm_C.fp8_emu_gemm_block_w8a16(
        a, w, w_s, lut, bias, out_dtype, nthreads, cfg, block_n, block_k)
```
（M 分支留在 opaque 体内，理由同 S11k：Dynamo 不得看到符号 M 分支。）
custom_op 名字 `vllm::fp8_emu_mm_block_w8a16`，`register_fake` 返回正确 shape/dtype。

**4.4.5 给 T6 预留的 hook**（本任务里就写好，T6 只填实现）
```python
def apply_weights(self, layer, x, bias=None, **kwargs):
    from . import gfx906_fp16_prefill  # T6 提供
    routed = gfx906_fp16_prefill.try_prefill(self, layer, x, bias)
    if routed is not None:
        return routed
    return super().apply_weights(layer, x, bias, **kwargs)
```
> 这个 override 只在 `apply_input_quant=False` 时才有意义；若 W8A16 关闭，`try_prefill` 必须返回 `None`。

### 4.5 T5 — dequant kernel（新文件，与 T1/T2 零冲突）

`csrc/rocm/fp8_dequant.cuh` + `csrc/rocm/fp8_dequant.cu`。

```cpp
// out[N,K] fp16 <- wq[N,K] fp8 * ws[n/block_n, k/block_k]
// 布局：保持 [N,K] 行序（不转置）；调用方用 x @ out.T()，hipBLAS 直接吃。
__global__ void fp8_dequant_to_fp16_kernel(
    __half* __restrict__ out, const uint8_t* __restrict__ wq,
    const float* __restrict__ ws, const __half* __restrict__ lut,
    int N, int K, int w_stride, int block_n, int block_k, int nkb) {
  extern __shared__ __half sl[];                 // 256 * 2 B
  for (int i = threadIdx.x; i < 256; i += blockDim.x) sl[i] = lut[i];
  __syncthreads();

  const int words_per_row = K >> 3;              // 8 fp8 / word
  const int64_t n_words = (int64_t)N * words_per_row;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t wid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; wid < n_words;
       wid += stride) {
    const int n = (int)(wid / words_per_row);
    const int k0 = (int)(wid % words_per_row) << 3;
    const uint2 v = *reinterpret_cast<const uint2*>(wq + (int64_t)n * w_stride + k0);
    const float s = ws[(int64_t)(n / block_n) * nkb + (k0 / block_k)];
    __half h[8];
    const uint32_t w[2] = {v.x, v.y};
#pragma unroll
    for (int u = 0; u < 2; ++u) {
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const __half raw = sl[(w[u] >> (8 * b)) & 0xff];
        h[u * 4 + b] = __float2half(__half2float(raw) * s);   // fp32 乘后舍入
      }
    }
    uint4 pack;
    pack.x = *reinterpret_cast<uint32_t*>(&h[0]);
    pack.y = *reinterpret_cast<uint32_t*>(&h[2]);
    pack.z = *reinterpret_cast<uint32_t*>(&h[4]);
    pack.w = *reinterpret_cast<uint32_t*>(&h[6]);
    *reinterpret_cast<uint4*>(out + (int64_t)n * K + k0) = pack;   // 16 B 对齐
  }
}
```
host 侧校验（缺一不可）：`out` fp16 连续、`wq` uint8 `stride(1)==1`、`ws` fp32、
`K % 8 == 0`、`K % block_k == 0`、`w_stride % 8 == 0`、`out.size(1) == K`、
`block_k == 128`（生产）。
grid：`blocks = min<int>((n_words + 255) / 256, 4096)`，`threads = 256`，smem = 512 B。

> 为什么不是「写 K-major [K,N]」：上游是给 TurboMind 的 packed 布局准备的；我们的 W 是 `[N,K]`
> 行序，转置会多一次 LDS 转置和一半的随机写。**保持 [N,K]，让 GEMM 吃 `W16.T()`**。

### 4.6 T6 — prefill 路由 + workspace（新模块）

`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py`

```python
_ROLE_MIN_M = {          # 占位；P1 实测后按 role 填，未知 role 用 _DEFAULT_MIN_M
    "gate_up_proj": None, "down_proj": None, "qkv_proj": None,
    "o_proj": None, "out_proj": None,
}
_DEFAULT_MIN_M = 4096

_WS: dict[tuple, torch.Tensor] = {}      # (device_index, torch.float16) -> buffer

def _enabled() -> bool:
    return _MODE != "0" and hasattr(torch.ops._rocm_C, "fp8_dequant_to_fp16_out")

def _role(layer) -> str:
    return (getattr(layer, "prefix", "") or "").rsplit(".", 1)[-1]

def _min_m(role: str, w: torch.Tensor) -> int:
    v = _ROLE_MIN_M.get(role)
    return int(v) if v is not None else _DEFAULT_MIN_M

def _workspace(w: torch.Tensor) -> torch.Tensor | None:
    key = (w.device.index, torch.float16)
    ws = _WS.get(key)
    need = w.numel()
    if ws is not None and ws.numel() >= need:
        return ws
    try:
        ws = torch.empty(need, dtype=torch.float16, device=w.device)
    except torch.OutOfMemoryError:                      # 必须回退，不许抛
        logger.warning_once("gfx906 fp16 prefill: workspace OOM, using W8A16 kernel")
        return None
    _WS[key] = ws
    return ws

def try_prefill(kernel, layer, x, bias):
    if not _enabled() or getattr(kernel, "apply_input_quant", True):
        return None
    from .gfx906_fp8 import _device_lut
    from .BlockScaledMMLinearKernel import FP8BlockParams

    params = FP8BlockParams.from_layer(layer)
    w = params.weight
    w_s = params.block_scale            # Fp8LinearMethod→weight_scale_inv；compressed-tensors→weight_scale
    if w is None or w.element_size() != 1 or w_s.dtype != torch.float32:
        return None
    if x.dtype != torch.float16 or w.stride(1) != 1:
        return None
    K = x.shape[-1]
    if w.shape[1] != K or K % 8 or w.stride(0) % 8:
        return None
    M = x.numel() // K
    if M < _min_m(_role(layer), w) or M < 16:            # 小 M 交回 emu kernel
        return None
    ws = _workspace(w)
    if ws is None:
        return None
    w16 = ws[:w.numel()].view(w.shape)                   # [N,K]
    torch.ops._rocm_C.fp8_dequant_to_fp16_out(
        w16, w, w_s, _device_lut(w.dtype, w.device),
        int(kernel.weight_group_shape.row), int(kernel.weight_group_shape.col))
    out = torch.mm(x.view(M, K), w16.t())
    if bias is not None:
        out = out + bias
    return out.to(kernel.config.out_dtype).view(*x.shape[:-1], out.shape[-1])
```

要点（每条都是踩过的坑）：

| # | 要点 |
|---|---|
| 1 | `weight_scale_inv` 的属性名按实际 layer 参数（`FP8BlockParams`）取，不要硬编码；先读 `_get_layer_params(layer).block_scale` |
| 2 | workspace **只在变小时复用、变大时替换**；图捕获前必须定型（捕获期分配 = 报错） |
| 3 | 不要 `data_ptr()` 当参数跨进程用；本实现直接传 tensor，天然安全 |
| 4 | 同一流内下一层的 dequant 覆盖上一层 buffer 是安全的（kernel 顺序执行）；**禁止跨 stream 并发** |
| 5 | `torch.mm` 走 hipBLAS：**确认 compute type 是 fp32**；若精度不足，评估关闭 `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction`（会变慢，需门禁） |
| 6 | 失败一律返回 `None` 回退 W8A16，**只 warn 一次**，不要每层刷屏 |

### 4.7 T7 — env 开关（唯一属主）

| env | 默认 | 含义 |
|---|---|---|
| `VLLM_DCU_FP8_W8A16` | `0` | `0` 现状；`1` 强制（op 缺失报错）；`auto` 有 op 就用 |
| `VLLM_DCU_FP16_PREFILL` | `0` | `0` 不启用 prefill 物化；`1` 强制；`auto` 按 `min_m` 表 |
| `VLLM_DCU_FP16_PREFILL_MIN_M` | 未设 | 覆盖 `_DEFAULT_MIN_M`（调试用） |
| `VLLM_DCU_FP16_WS_BYTES` | 未设 | 强制 workspace 字节数（图捕获前预热用） |

写法沿用 `vllm/envs.py` 现有 lambda 风格（:1462 附近），并在同名注释里写清「默认关、A/B 与回退开关」。

### 4.8 T8 — 测试（追加到 `tests/kernels/quantization/test_gfx906_fp8_emu.py`）

```python
# W8A16：参考实现 = 同一 fp8 权重 + a_s 全 1
def _reference_block_w8a16(a, w_q, w_s):        # a: fp16
    return _reference_block(a, None, w_q, w_s)  # 内部把 a_s 当 1

@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
def test_block_gemv_w8a16(fp8_ops, m, out_dtype): ...

@pytest.mark.parametrize("m", [16, 64, 129, 512])
@pytest.mark.parametrize("cfg", [0, 5, 10, 16, 35, 36, 39, 40])
def test_block_gemm_w8a16(fp8_ops, m, cfg): ...

def test_block_w8a16_bf16_and_fp8_a_rejected(fp8_ops): ...
def test_w8a16_off_is_bitwise_identical_to_w8a8(fp8_ops, monkeypatch): ...
def test_dequant_kernel_matches_reference(fp8_ops): ...          # 与 w_q*w_s 的 fp32 参考比
def test_dequant_rejects_bad_alignment(fp8_ops): ...
def test_prefill_route_matches_w8a16_within_tolerance(fp8_ops): ...  # 单层，cos>=0.99999
def test_prefill_route_falls_back_on_oom(fp8_ops, monkeypatch): ...
def test_route_snapshot(): ...                                   # (role,(K,N))×M 桶 → 路径哈希
```

单元测试**可以现在就写**（不需要卡），执行放在 P1。

---

## 5. 施工顺序（建议）

```
   T5（新文件，独立）─┐
                      ├─→ 编译（串行，§7）→ T8 跑单测（P1）
   T1 → T2 → T3 ──→ T4 ──→ T6 ──→ T7 ──→ T9
```
- **T5 与 T1 可以同一个 agent 先做**（互不冲突，且 T5 能独立验证）。
- T4 必须等 T3 的 op 名字定稿；T6 必须等 T4 的 hook 合入（否则 import 失败）。
- 每一步都要过 §6 的静态门禁，**不要求**卡。

---

## 6. 门禁

### 6.1 静态（无卡可过，每步都要）

| 门禁 | 命令/方法 |
|---|---|
| Python 风格 | `uv run ruff check vllm tests`（行宽 88，Google docstring） |
| Python 导入 | `.venv/bin/python -c "import vllm.model_executor.kernels.linear.scaled_mm.gfx906_fp8"` |
| C++ 编译 | 串行构建（§7），必须 0 warning 新增 |
| 机器码 | `.s23/probes/isa_gate.py`：新 TU 必须出现 `v_dot2_f32_f16`，不得出现 `v_mfma*`；A 侧 `ds_read_u16` 计数应低于 W8A8 版本 |
| 差异检视 | `git diff` 里 **W8A8 分支不得有语义改动**（只允许新增分支） |

### 6.2 数值（P1，卡空后）

| 层 | 判据 |
|---|---|
| dequant kernel | 与 `w_q.float() * w_s` 的 fp16 结果 `torch.equal`（同一舍入路径）或 ≤1 ulp |
| W8A16 kernel | vs fp32 CPU 参考：`rel_err ≤ 1e-3`；fp16 输出 `atol=1e-3` |
| prefill 路线 | vs W8A16 kernel 同输入：`cos ≥ 0.99999`，`max_abs_err` 入档 |
| 单层真实权重 | `blocks.0.attn.qkv_proj`：`cos ≥ 0.99999` |
| e2e | greedy 逐 token parity（W8A16 vs W8A8 记录的差异方向与幅度）、logprobs finite、DFlash2 k=7 accept 不降 |
| 长稳 | 32k 单请求 + 64k soak：0 OOM/0 retry/0 preempt |

> W8A16 **必然**不逐 bit 等于 W8A8（激活更准）。要记录的是差异，不是相同。
> 「开关关闭时逐 bit 等于改造前」是**另一条**门禁，必须绿。

### 6.3 性能（P1）

每形状 × `M ∈ {1,8,16,512,1024,1760,4096,8192}`；同窗口背靠背 A/B；报比值不跨窗口做差。
判定线：**decode 不回归**、每层 launch 数由 2 降为 1、prefill 在 `M ≥ min_m` 上严格更快。

---

## 7. 构建与协作纪律

1. **构建串行**：`.so` 重建同一时刻只允许一个 agent；构建前 `git status` 必须干净（只含预期改动），
   构建后记录 md5 到 `docs/result/`。
2. **不要**在构建期间启动生产服务或探针（GPU 是别人的）。
3. 任何跨文件的接口变动（op 名、参数表）先在任务板写清再动手；**先合 T3 的声明，再写调用方**。
4. 每个任务结束把「改了什么 / 怎么验的 / 还剩什么」写进任务板，不要只写 commit message。

---

## 8. 坑清单（统一版）

| # | 坑 | 规避 | 出处 |
|---|---|---|---|
| P1 | 每调用读 `getenv` | import / `call_once` 读一次 | 我们也这么干过；上游同款 |
| P2 | `As` 占位符（`new_empty(1)`）被读 | W8A16 内核签名不含 `a_s`，或传 `nullptr` 并断言 | `BlockScaledMMLinearKernel.py:198-205` |
| P3 | A 的 dword 索引沿用 `kFKw4`（4 元素/字）→ 只读一半，结果错但"不崩" | `kFKw = kFBk/2`、`kFLogKw = 4` | 本文件 §4.2 |
| P4 | LDS store 仍 `2*jd` 展开 | `As[r*kFKp + jd] = v` | 同上 |
| P5 | 忘记 W 的 padding stride（`K+256`） | 一律用 `w.stride(0)`；dequant 也用它 | `fp8_utils.py:_maybe_pad_fp8_weight` |
| P6 | `torch.compile` 把 M 分支/开关 trace 死 | 开关是 import 常量；M 分支在 custom_op 内 | S11k |
| P7 | 图捕获期分配 workspace | 捕获前预热；`VLLM_DCU_FP16_WS_BYTES` 固定尺寸 | 上游 stale-pointer 教训 |
| P8 | workspace 跨层被并发用 | 单 batch 前向内层串行即可；禁止跨 stream | `1cat-vllm: sm70_fp8.py:74-78` |
| P9 | `w_s` 烘进 fp16 时在 fp16 域乘（多一次舍入） | 我们**在 fp32 域乘再舍入**，并在结果文档记录与上游的差异 | `awq_sm70_gemm.cu:3087` |
| P10 | 用 `torch.equal` 卡 W8A16 / prefill | 分层门禁（§6.2） | — |
| P11 | 为 W8A16 全量实例化 cfg 1..40，编译时间爆炸 | 只模板化 decode + 实际使用的 prefill cfg | 本文件 §4.2 |
| P12 | dequant 输出写成 `[K,N]` | 保持 `[N,K]`，GEMM 吃 `w16.t()` | 本文件 §4.5 |

---

## 9. 交付物

1. 代码：T1–T7（开关默认 `0`/`0`，**不改变现状行为**）。
2. 单测：T8（先写；P1 执行）。
3. `docs/result/z100-w8a16-fp16-prefill-result.md`：契约、门禁原始日志、每形状 A/B、
   `min_m` 表、e2e parity、回退演练、`.so` md5。
4. 路由快照（`(role,(K,N))×M 桶 → 路径` 的哈希）作为回归护栏。
5. **晋级提交**（单独一个 commit，只改默认值）：`W8A16=auto`、`FP16_PREFILL=auto`，
   前提是 §6.2/§6.3 全绿。
