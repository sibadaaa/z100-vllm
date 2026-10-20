# Z100 FP8 emu → **W8A16** 实施方案（供执行 agent 使用）

> **已合并**：决策 1 与决策 2 的统一施工图见
> [`z100-w8a16-prefill-fp16-construction.md`](z100-w8a16-prefill-fp16-construction.md)，
> **施工以该文件为准**；本文件保留为 W8A16 单点的背景说明。
>
> 决策：**做**（用户已定）。目标是把 `Gfx906Fp8BlockScaledMMLinearKernel` 的激活侧从
> 「fp8 量化 + LUT 加宽」改成「**fp16 直入、不量化**」，权重仍 fp8 常驻。
> 上游同类实现（V100/sm70，同样无 fp8 硬件）见 `/mnt/game/1cat-vllm`：
> `vllm/model_executor/kernels/linear/scaled_mm/sm70_fp8.py:463-466`、
> `qpn8_blk.py:112-116`。
> 本文件是**施工图**：不含 benchmark 数字，所有性能验收都在实现完成后按 §8 跑。

---

## 1. 契约定义（先对齐，再写码）

| 项 | W8A8（现状） | **W8A16（目标）** |
|---|---|---|
| A | fp8 e4m3 `[M,K]` + `a_s` `[M,nkb]` fp32 | **fp16 `[M,K]`，无 `a_s`** |
| W | fp8 e4m3 `[N,K]`（K-contiguous，可带 padding）+ `w_s` `[cdiv(N,block_n), nkb]` fp32 | 同左 |
| 累加 | fp32（`v_dot2_f32_f16`） | 同左 |
| 数学 | `out[m,n] = Σ_kb (Σ_{k∈kb} a·w) · a_s[m,kb] · w_s[nb,kb]` | `out[m,n] = Σ_kb (Σ_{k∈kb} a·w) · w_s[nb,kb]` |
| 输出 | `out_dtype`（生产 = fp16），bias 在 `out_dtype` 加 | 同左 |

**不变量**
1. 权重**永不解量化、永不复制**到显存（decode 带宽是生命线）。
2. W8A8 路径保持逐 bit 不变；W8A16 是**新增分支**，默认关闭，开关打开才生效。
3. A 只接受 `float16`。bf16 在 gfx906 无指令 → 直接报错，不做兼容。
4. `block_k` 仍必须整除 `K`；`N % BN` 的既有约束不变；`K` 必须为偶数（fp16 成对读）。

---

## 2. 改动总表（文件 → 符号 → 动作）

| # | 文件 | 符号 | 动作 |
|---|---|---|---|
| C1 | `csrc/rocm/fp8_emu_gemm.h` | 新增 `fp8_emu_gemm_block_w8a16(...)` + `..._meta(...)` | 声明，注释写清 A 是 fp16 |
| C2 | `csrc/rocm/fp8_emu_gemm.cu` | `run_block`（:1814）/ `check_block_common`（:1596） | 新增 W8A16 版校验 + 派发（**不改原函数**，新增 `run_block_w8a16`） |
| C3 | 同上 | `fp8_emu_gemv_block_kernel<MS,T,PRECONV>`（:449） | W8A16 = 现成的 `PRECONV=true` 路径 + `a_h2 = A`；新增模板参数 `bool HAS_A_S`（false 时 `s[m]=wsv`） |
| C4 | 同上 | `fp8_emu_gemv_stream_block_kernel`（:601）/ `..._stream2...` | A staging 改成 fp16 直拷；`sscale[i] = wsr[kb]` |
| C5 | 同上 | `fp8_gemm_load_block`（:1077）/ `fp8_gemm_load_wide`（:1266）/ `fp8_gemm_store`（:302） | A 的 gmem→寄存器→LDS 三段改成 fp16 直通（见 §4.3） |
| C6 | 同上 | `fp8_gemm_fold_block_scale`（:1110） | 加 `HAS_A_S`：false 时 `acc += blk * w_s[...]` |
| C7 | 同上 | 新 host 函数 `fp8_emu_gemm_block_w8a16` | 组装：M 分支 / stream 选择 / cfg 选择，与 `run_block` 同构 |
| C8 | `csrc/rocm/torch_bindings.cpp` | `rocm_ops.impl` / `rocm_meta_ops.impl`（:139-179 区域） | 注册新 op + meta |
| C9 | `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py` | `_fp8_emu_mm_block_w8a16_impl`、`torch.ops.vllm.fp8_emu_mm_block_w8a16`（custom_op + fake）、`ops_available_block_w8a16()` | 新增 |
| C10 | 同上 | `Gfx906Fp8BlockScaledMMLinearKernel` | 加 `apply_input_quant = False`（受开关控制）、`apply_block_scaled_mm` 里按 `A.dtype` 分流、`is_supported` 加能力检查 |
| C11 | `vllm/envs.py` | `VLLM_DCU_FP8_W8A16` | 新增（`0`/`1`/`auto`） |
| C12 | `tests/kernels/quantization/test_gfx906_fp8_emu.py` | 新用例 | 见 §7 |

**不要动**：`Gfx906Fp8ScaledMMLinearKernel`（per-tensor/per-channel 家族）、
`fp8_emu_gemm_block_quant*`（S23-O5 融合量化，W8A16 生效后该路径对本 kernel 变为不可达，
但代码保留，别的入口仍可能在用）、任何 cfg 编号与 tile 形状。

---

## 3. Python 侧接线（先做，可独立自测）

### 3.1 关键事实

`Fp8BlockScaledMMLinearKernel.apply_weights`（`BlockScaledMMLinearKernel.py:141-217`）
已经内置了「不量化激活」的钩子：

```python
apply_input_quant: ClassVar[bool] = True        # :65
...
if self.apply_input_quant:                      # :161  走量化（现状）
    ...
else:                                           # :198
    q_input = input_2d
    input_scale = input_2d.new_empty(1)         # 占位符，子类不得使用
output = self.apply_block_scaled_mm(A=q_input, B=weight, As=input_scale, Bs=weight_scale)
```

`sm70_fp8.py` 用的是同一套机制（`apply_input_quant = False`）。所以 Python 侧改动很小：

1. `Gfx906Fp8BlockScaledMMLinearKernel` 增加类属性：
   ```python
   apply_input_quant = _W8A16 and ops_available_block_w8a16()
   ```
   `_W8A16` 在 import 时读一次 env（禁止每调用读 env）。
2. `apply_block_scaled_mm(A, B, As, Bs)` 内分流：
   - `A.dtype == torch.float16` → 调 `_fp8_emu_mm_block_w8a16_impl(A, w, Bs, lut, bias, out_dtype, ...)`；
   - `A.element_size() == 1` → 现有 `_fp8_emu_mm_block_impl`（原样不动）。
   - `A.dtype == bfloat16` → `raise RuntimeError("gfx906 W8A16 requires fp16 activations")`。
3. `is_supported()` 增加：`ops_available_block_w8a16()` 存在；`can_implement()` 增加：`c.weight_quant_key` 是 block-scaled（已有）。
4. `fuses_input_quant` / `_fuse_quant` / `_quant_fastpath` 在 `apply_input_quant=False` 时不会被求值，
   **不要去删它们**（其他配置仍可能用到；删了会把 S23 的 A/B 口径弄丢）。

### 3.2 新 custom_op

```python
@torch.library.custom_op("vllm::fp8_emu_mm_block_w8a16", mutates_args=())
def fp8_emu_mm_block_w8a16(a, w, w_s, lut, bias, out_dtype, nthreads, cfg,
                           block_n, block_k) -> torch.Tensor
```
- 参数与 `fp8_emu_mm_block` 对齐但不含 `a_s`；`M <= GEMV_MAX_M` 的 M 分支**留在 op 内部**
  （复用 `_fp8_emu_mm_block_impl` 的写法，见 `gfx906_fp8.py:228-263`）。
- fake impl 只需返回正确 shape/dtype。

---

## 4. 内核实现（核心，逐函数）

### 4.1 decode GEMV（M ≤ 8）——最省事的一条

现有 `fp8_emu_gemv_block_kernel<MS, T, PRECONV>` 的 **`PRECONV=true`** 分支已经是
「A 从 fp16 指针读」：

```cpp
const uint32_t* __restrict__ arow = a_h2 + (int64_t)m * (K4 * 2);
x0 = arow[2 * i]; x1 = arow[2 * i + 1];
```
（`fp8_emu_gemm.cu:503-510`；`K4*2 == K/2` 正好是 fp16 行的 dword 步长）

因此：
- W8A16 直接实例化 `PRECONV=true`，把 `a_h2` 指向 **A 的原始显存**（不再分配 scratch、
  不再发射 `fp8_emu_widen_a_kernel`）；
- 新增模板参数 `bool HAS_A_S`：`false` 时 `s[m] = wsv`（删掉 `a_s[m*nkb+kb] *`）；
- `a_s` 参数保留在签名里但传 `nullptr`（或删参，二选一，建议保留以减小 diff）。

### 4.2 streaming GEMV（窄 N 的 decode 快路）

`fp8_emu_gemv_stream_block_kernel`（:601）把 A 一次性加宽进 LDS（`sA` 是 `__half*`，
:634）。W8A16：
- A staging：`sA[m*K + k] = A_half[m*K + k]`（向量化拷贝，`uint4`/`uint2`，无需 LUT）；
- scale 折叠（:659-666）：`sscale[i] = wsr[kb]`（去掉 `a_s[...]*`）；
- 其余（W 的 LUT 加宽、dot2、归约、epilogue）不动。

### 4.3 tiled GEMM（M > 8，prefill）

三处，全部在**热循环外**或**K 维 staging**，不影响 MAC 端：

1. **gmem→寄存器**（`fp8_gemm_load_block:1077`）：A 现在按 fp16 读，每 dword 2 个元素。
   ```
   A_DWORDS_PER_ROW = kFKw = kFBk/2 = 16        // 现为 kFKw4 = 8
   idx = tid + u*NT;  r = idx >> kFLogKw;  jd = idx & (kFKw-1);
   v = A32[(int64_t)(m0+r) * (K>>1) + (k0>>1) + jd];
   A4_16 = (BM * kFKw + NT - 1) / NT            // 注意是 kFKw，不是 kFKw4
   ```
   N 越界/尾部 K 不足时补 0 的守卫要保持（`k0 + (jd+1)*2 <= K`）。
2. **寄存器→LDS**（`fp8_gemm_store:302`）：A 那一半不再调 `fp8x4_to_f16x2x2`，直接
   `As[r*kFKp + jd] = src[u]`（**不再 `2*jd` 展开**，因为已经是 fp16 对）。B 那一半不动。
3. **block scale 折叠**（`fp8_gemm_fold_block_scale:1110`）：`HAS_A_S=false` 时不读 `a_s`。

`wide2` 家族（`KConf`，`fp8_gemm_load_wide:1266`）同理，把 `Kw4 = BK/4` 换成 `Kw = BK/2`。

> **实现方式建议**：A 侧是否模板化取决于编译预算。gemv 三条路径**必须**模板化（内层热循环）；
> tile 的 staging 在 K 维循环外层，可用 runtime uniform 分支（`const bool a16`）避免
> cfg 1..40 的实例翻倍。**不要**为了 W8A16 复制整张 cfg 表。

### 4.4 host 侧（`fp8_emu_gemm_block_w8a16`）

与 `run_block`（:1814）同构，差异只在：
- 校验：`a.scalar_type()==at::kHalf`、`a.size(1)%2==0`、`w.element_size()==1`、
  `w.stride(1)==1`、`w_s` fp32、`block_k % 32 == 0`、`nkb = cdiv(K, block_k)`；
  **不再要求** `a_s`。
- 指针：`const uint32_t* A16 = reinterpret_cast<const uint32_t*>(a.data_ptr())`。
- **不发射** `fp8_emu_widen_a_kernel`。
- 其余（stream 选择、`gemv_use_gemm`、cfg 选择、`cpb`）完全复用。

---

## 5. 路由与开关

| env | 值 | 行为 |
|---|---|---|
| `VLLM_DCU_FP8_W8A16` | `0`（默认，落地阶段） | 现状 W8A8，逐 bit 不变 |
| | `1` | 强制 W8A16（op 缺失则**抛错**） |
| | `auto` | op 存在且 `A.dtype==fp16` 时用 W8A16；否则 W8A8 |
| `VLLM_DCU_USE_FP8_EMU` | 不变 | 总闸 |

**晋级路径**：代码合入时默认 `0` → §7 门禁全绿 → 单独一个 commit 把默认改成 `auto`
（不改代码路径，只改默认值）→ 观察一个维护窗口 → 再评估是否删除 W8A8 分支（本方案不删）。

---

## 6. 分阶段施工（每阶段可独立回退）

| 阶段 | 内容 | 完成判据 |
|---|---|---|
| **K0** | C1/C2/C8：C++ op + binding + meta；python 侧 `_impl` + custom_op（先只支持 GEMV，tile 走「不支持→回 W8A8」） | `torch.ops._rocm_C.fp8_emu_gemm_block_w8a16` 可调用；形状 1/2/4/8 正确 |
| **K1** | C3/C4：gemv + stream 两条 decode 路径 | 与 `_reference_block`（a_s 全 1 的 fp8 参考）逐 bit 一致 |
| **K2** | C5/C6/C7：tile + wide2 | cfg 10/16/35/36-40 × M∈{16,129,512} 与参考一致 |
| **K3** | C10/C11：python 路由 + env | `apply_input_quant=False` 生效；`As` 占位符不被读取 |
| **K4** | 门禁（§7）+ e2e A/B | 见 §7 |
| **K5** | 默认值晋级（`auto`）+ 结果文档 `docs/result/z100-w8a16-result.md` | — |

---

## 7. 测试与门禁

### 7.1 单测（加到 `tests/kernels/quantization/test_gfx906_fp8_emu.py`）

1. `test_block_gemv_w8a16_matches_reference[out_dtype × M∈{1,2,3,4,5,6,8}]`：
   参考实现 = `_reference_block(a.astype(fp16), ones_as, w_q, w_s)`（把 A 直接当 fp16 用，
   `a_s=1` 等价）。
2. `test_block_gemm_w8a16_matches_reference[M∈{16,64,129} × cfg∈{0,5,10,16,35,36..40}]`。
3. `test_block_w8a16_stream_path`（窄 N + 小 K，强制 stream 命中）。
4. `test_block_w8a16_padded_weight_stride`（沿用 :376 的 padded stride 用例）。
5. `test_block_w8a16_rejects_bf16` / `rejects_fp8_a`（错误路径）。
6. `test_w8a16_switch_off_matches_w8a8_bits`：开关关掉时，同一输入输出与改动前**逐 bit 相同**
   （这条是仓库默认位稳定原则的护栏）。
7. `test_w8a16_requires_block_scaled_config`（per-tensor 配置被拒）。

### 7.2 数值门禁（必须全绿才允许晋级默认）

| 层 | 判据 |
|---|---|
| kernel | vs fp32 CPU 参考：`rel_err ≤ 1e-3`（fp16 输出再加 `atol=1e-3`） |
| 单层 | 真实权重（`blocks.0.attn.qkv_proj` 级）+ 真实激活：`cos ≥ 0.99999`，`max_abs_err` 记录入档 |
| e2e | 8 条固定 prompt：greedy **逐 token parity**（W8A16 vs W8A8）、logprobs 全 finite、DFlash2 k=7 accept 不降 |
| 长上下文 | 32k 单请求 + 64k soak：0 OOM / 0 retry / 0 preempt |

> W8A16 必然**不**逐 bit 等于 W8A8（激活不再量化，更准）。所以要记录的是「差异方向与幅度」，
> 不是「相同」。若 greedy parity 不过，先查是不是 A 的 dtype/stride 被 `.view()` 破坏。

### 7.3 机器码门禁

`.s23/probes/isa_gate.py` 口径（遍历每个 offload bundle）：新 TU 必须出现 `v_dot2_f32_f16`，
且**不得**出现 `v_mfma*`；A 侧 LUT 相关的 `ds_read_u16` 计数应相对 W8A8 版本下降。

---

## 8. 性能验证（实现完成后跑，不预设数字）

1. 每形状 × M∈{1,8,16,512,1760,4096} 的**每调用时间**（同窗口背靠背，A/B 两臂）。
2. kernel census：确认每层由 2 次 launch（quant + GEMM）变 1 次。
3. host 时间：`apply_weights` 端到端 vs 现状。
4. e2e：c=1 / c=16 两档，报比值，不跨窗口做差。
5. 判定线：**decode 不回归**（每调用时间不超过现状），且 host+launch 项必须下降；
   否则回到 K3 检查是否落到了 `apply_input_quant=False` 之外的分支。

---

## 9. 已踩过的坑（不要重复）

| # | 坑 | 规避 |
|---|---|---|
| P1 | 每调用读 `getenv`（我们就这么干过：`decode_gemm_cfg()`） | import/`call_once` 读一次 |
| P2 | `As` 占位符 `new_empty(1)` 被读 → 垃圾 scale | W8A16 内核签名不含 `a_s`，或传 `nullptr` 并断言 |
| P3 | A 的 dword 索引沿用 `kFKw4`（4 元素/字），导致只读一半 | 明确 `kFKw = kFBk/2`、`kFLogKw = 4` |
| P4 | LDS store 仍做 `2*jd` 展开（那是加宽路径的写法） | `As[r*kFKp + jd] = v` |
| P5 | 忘了 `K` 必须偶数 / `w` 的 padding stride | 校验 + 用 `w.stride(0)`（`fp8_gemm_load_block` 已处理） |
| P6 | 在 `torch.compile` 下让 `apply_input_quant` 变成符号分支 | 类属性 + import 时常量；M 分支留在 custom_op 内 |
| P7 | 顺手把 `_fuse_quant` 删了 | 保留；W8A16 生效时它自然不可达 |
| P8 | 用 `torch.equal` 卡 W8A16 | 用 §7.2 的分层门禁 |

---

## 10. 交付物

1. 代码（C1–C12）+ `.so` md5。
2. `docs/result/z100-w8a16-result.md`：契约、门禁原始日志、每形状 A/B、e2e、回退演练。
3. 路由快照：`(role,(K,N)) × M 桶 → route(W8A8/W8A16)` 的哈希（防回归）。
