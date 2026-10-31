# Z100 W8A16（E5M2 / E4M3）内核审查修复计划

> **读者**：执行本计划的 agent（可 2–3 个并行）。本文件是唯一施工依据，冲突时以本文件为准。
> **来源**：对 `csrc` 中 W8A16 E5M2/E4M3 内核实现的代码审查。核心数值路径（LUT 无损展宽 +
> `v_dot2_f32_f16` fp32 累加）经逐行核对**未发现会导致错误结果的 bug**；本计划闭合审查发现的
> **测试覆盖缺口**与**一处防御性校验**，外加两条维护提示。
> **配套背景**（不必重复阅读）：`z100-w8a16-prefill-fp16-construction.md`（施工总方案）。

---

## 0. 三十秒看懂

审查结论：**内核正确，问题在覆盖**。做完之后：

```
T1  W8A16 GEMM/GEMV 单测  ──►  e4m3 + e5m2 双覆盖（闭合唯一的"中"级问题）
T2  prefill 路由 _layout_ok ──►  补 block_k % 8 == 0（防御性，一行）
T3  wholea/m1 特化 parity  ──►  子进程开启 env 的 parity 用例（可选，较重）
T4  两条维护提示          ──►  折叠顺序差异 + 孪生内核（纯文档，可选）
```

- **T1 + T2 是最小组合**，收益/风险比最高，闭合审查里唯一的"中"级问题（e5m2 端到端覆盖）和一个防御点。
- **T3 / T4 看团队要求**再定（A/B 路径是否强制自动化、是否要文档化维护陷阱）。
- **全部任务都不碰内核数值路径**（`csrc/rocm/fp8_emu_gemm.cu` 只读参考），因此低风险、可并行。

---

## 1. 契约与不变量（所有 agent 都要遵守）

1. **不改内核数值**：本计划的所有改动落在测试 / Python 路由 / 注释，`csrc/rocm/fp8_emu_gemm.cu`
   与 `csrc/rocm/fp8_dequant.cuh` 的算术、tiling、fold 节拍一律不动。
2. **文件唯一属主**：`test_gfx906_fp8_emu.py`、`gfx906_fp16_prefill.py` 各只有一个属主；
   T1 与 T3 同属测试文件，**若要一起做必须串行（先 T1 后 T3）**。
3. **环境**：Python 一律走 `.venv/bin/python`（AGENTS.md 强制，禁用系统 python3 / 裸 pip）。
4. **硬件门槛**：这些测试带 `pytestmark = skipif(not _gfx906_stack())`，**只在 gfx906 盒子上真正
   执行**，其余平台 skip。验收必须在 gfx906 机器上跑，不能以"本地 skip 通过"当作通过。
5. **容差纪律**：T1 参数化后**不得放宽容差**来掩盖 e5m2 的偶发超差；若真出现超差，先记录原始
   `max|err|/max|ref|` 再判断（见 T1 验收）。

---

## 2. 涉及文件

| 角色 | 路径 | 本计划是否改 |
|---|---|---|
| W8A16 内核（tile GEMM / streaming GEMV / m1 / wholea） | `csrc/rocm/fp8_emu_gemm.cu` | 否（只读参考） |
| dequant 内核 | `csrc/rocm/fp8_dequant.cuh` | 否（只读参考） |
| 展宽原语 | `csrc/rocm/gfx906_emu_common.cuh` | 否 |
| W8A16 单测 | `tests/kernels/quantization/test_gfx906_fp8_emu.py` | **是（T1、T3）** |
| prefill fp16 路由 | `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py` | **是（T2）** |
| 维护提示落点 | 内核注释 或 `docs/design/z100/w8a16/` 结果文档 | **是（T4）** |

---

## 3. 任务板

### T1【高价值 / 必做】W8A16 GEMM/GEMV 单测补 e5m2 覆盖

**问题**：4 个 W8A16 端到端用例硬编码 `torch.float8_e4m3fn`，e5m2 权重的 GEMM/GEMV 数据通路
从未端到端跑过（目前只有 dequant 内核测试与 LUT 无损测试覆盖 e5m2）。

**Write scope**：仅 `tests/kernels/quantization/test_gfx906_fp8_emu.py`

**改动**：给下面 4 个函数各加一个参数化装饰器，函数签名加上 `fp8_dtype` 形参，并删除函数体内
那行 `fp8_dtype = torch.float8_e4m3fn`：

```python
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
```

| 函数 | 现位置 | 删除的行 |
|---|---|---|
| `test_block_gemv_w8a16` | `:630` | `:633` `fp8_dtype = torch.float8_e4m3fn` |
| `test_block_gemm_w8a16` | `:648` | `:651` `fp8_dtype = torch.float8_e4m3fn` |
| `test_block_w8a16_bias_and_bf16_out` | `:663` | `:666` `fp8_dtype = torch.float8_e4m3fn` |
| `test_block_w8a16_accepts_padded_weight_row_stride` | `:686` | `:689` `fp8_dtype = torch.float8_e4m3fn` |

细节：
- `test_block_gemm_w8a16` 已有 `@pytest.mark.parametrize("cfg", W8A16_CFGS)`，新装饰器叠在其上即可
  （两层参数化，pytest 会自动做笛卡尔积）。
- `test_block_w8a16_accepts_padded_weight_row_stride` 内部用 `w8a16_ops(fp8_dtype)` 取 LUT，
  参数化后 LUT 自然跟着 dtype 变。
- 其余函数体内凡引用 `fp8_dtype` 的地方（`_quantize_block(w, fp8_dtype, ...)`、
  `w8a16_ops(fp8_dtype)`）保持不变，直接用形参。

**为什么容差不用改**：参考 `_reference_block_w8a16` 与内核吃**同一份** `w_q`
（参考侧 `_dequantized_weight` 用 `w_q.to(torch.float32)`），唯一差异是 fp32 累加顺序 / 末位
cast，与 e4m3/e5m2 无关。`_quantize_block` 用 `torch.finfo(fp8_dtype)`，dtype 无关。
保留现有容差：fp32 `1e-5`、fp16 `1e-3`、bf16 `1e-2`。

**验收**（必须在 gfx906 盒子）：
```bash
.venv/bin/python -m pytest tests/kernels/quantization/test_gfx906_fp8_emu.py -k "w8a16" -v
```
e4m3 与 e5m2 两套全绿。若 e5m2 出现偶发超差，**先记录原始 `max|err|/max|ref|` 与触发形状**，
再判断是真回归还是容差边界，不要盲目放宽容差。

---

### T2【一行 / 防御性】`_layout_ok` 补 `block_k % 8 == 0`

**问题**：Python 侧 gate `_layout_ok` 只查 `block_k > 0` 与 `K % block_k`，没查 `block_k % 8`，
而 C++ dequant 明确要求 `block_k % 8 == 0`（`csrc/rocm/fp8_dequant.cuh:159`）。异常 `block_k`
会触发 C++ `TORCH_CHECK`，虽被 `try_prefill` 的 `try/except`（`gfx906_fp16_prefill.py:420`）
兜底回退到 W8A16 内核（**非正确性问题**），但白跑一次 + 打一条 warning。

**Write scope**：仅 `vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp16_prefill.py`

**改动**：`_layout_ok`（`:233`）
```python
# 旧
if block_n <= 0 or block_k <= 0 or K % block_k:
    return False
# 新
if block_n <= 0 or block_k <= 0 or block_k % 8 or K % block_k:
    return False
```

**验收**（必须在 gfx906 盒子）：
```bash
.venv/bin/python -m pytest tests/kernels/quantization/test_gfx906_fp8_emu.py -k "prefill" -v
```
现有 prefill 路由用例全绿（合法 `block_k = 128` 行为不变）。

---

### T3【可选 / 较重】wholea / m1 特化 parity 测试

**问题**：M==1 的 wholea / m1 内核由 `VLLM_DCU_FP8_GEMV_WA` / `VLLM_DCU_FP8_GEMV_M1` 门控
（`csrc/rocm/fp8_emu_gemm.cu:1104` / `:1067`），默认关，默认套件只走到通用 streaming 内核。
内核注释声称它们与生产路径 bit-identical，但无自动化背书。

**关键约束**：这两个 env 在 .cu 里是 `static const` **进程内只读一次**（`wholea_enabled` /
`stream_m1_enabled`），**无法在进程内 monkeypatch 翻转**，必须子进程设 env 才能生效。

**Write scope**：仅 `tests/kernels/quantization/test_gfx906_fp8_emu.py`（与 T1 同文件 → 串行）

**推荐做法（子进程 parity）**：新增一个用例，用 `subprocess` 起一个设了 `VLLM_DCU_FP8_GEMV_WA=1`
（以及可选 `VLLM_DCU_FP8_GEMV_M1=1`）的子进程；子进程内构造 M=1 的 W8A16 输入、跑
`torch.ops._rocm_C.fp8_emu_gemm_block_w8a16`（此时路由进 wholea/m1），对照 fp32 参考断言 parity，
把结果码 / 误差回传父进程断言。这样验证 wholea/m1 数值正确。

**轻量替代（默认建议）**：鉴于 wholea/m1 是默认关的 A/B 实验，可在测试文件加一条 TODO 注释，
说明这两条路径由 env 门控、需子进程开启、暂以带外 A/B 验证，避免为低收益路径引入脆弱的子进程
测试。**二选一，默认走轻量替代**，除非团队要求强制自动化。

**验收**：若做子进程版，`-k "wholea or m1"` 在 gfx906 盒子上通过；若走轻量替代，确认 TODO 注释
已落、不影响既有用例。

---

### T4【可选 / 纯文档】两条维护提示

不改行为，仅在注释 / 设计文档补两句，防止后续踩坑：

1. **streaming vs 非 streaming 的标量折叠顺序不同**：streaming 系列是 `Σ(chunk_sum·s)`
   （`csrc/rocm/fp8_emu_gemm.cu:818`，每个 16 元素 chunk 折一次），非 streaming GEMV 是
   `(Σblock)·s`（整块求和后再乘 scale）。两者浮点舍入顺序不同，**不能用 `torch.equal` 互卡**
   （W8A8 既有特性，W8A16 沿用）。
2. **孪生内核**：`fp8_emu_gemm_block_w8a16_kernel`（`csrc/rocm/fp8_emu_gemm.cu:2358`）与
   `fp8_emu_gemm_block_wide2_kernel` 除 A 侧（fp16 直读 vs fp8 LUT 展宽）外逐行一致，
   `A16Conf` 也复制了 `KConf` 的 `Kw/Kp/MG/NG/NT`。改 wide2 的 bug 需**手动同步**到 W8A16。

**Write scope**：内核注释（`csrc/rocm/fp8_emu_gemm.cu` 相应位置）或 `docs/design/z100/w8a16/`
下的结果文档。二选一，倾向落在结果文档，避免为加注释去动内核 TU 触发重编。

**验收**：提示已写入、表述准确；若改内核注释，确认增量编译通过（见 AGENTS.md 增量构建流程）。

---

## 4. 并行与归属

| 任务 | 文件属主 | 依赖 | 可并行 |
|---|---|---|---|
| T1 | `test_gfx906_fp8_emu.py` | — | ✅ 与 T2 / T4 |
| T2 | `gfx906_fp16_prefill.py` | — | ✅ 与 T1 / T4 |
| T3 | `test_gfx906_fp8_emu.py` | 与 T1 同文件 → **串行（接 T1）** | — |
| T4 | 内核注释 / docs | — | ✅ |

> T1 与 T3 同属一个测试文件，若要一起做需串行（先 T1 后 T3），否则按文件唯一属主规则拆到不同 agent。

**最小组合**：只做 **T1 + T2** 就闭合了审查里唯一的"中"级问题（e5m2 覆盖）和一个防御点。
T3 / T4 看团队对 A/B 路径自动化和文档的要求再定。

---

## 5. 交付物

1. 代码：T1（测试参数化）、T2（一行防御）。
2. 可选：T3（子进程 parity 用例或 TODO 注释）、T4（维护提示）。
3. 验收日志：T1/T2 的 `-k "w8a16"` / `-k "prefill"` 原始输出（gfx906 盒子），附 e5m2 的
   `max|err|/max|ref|` 数值，作为回归护栏。
