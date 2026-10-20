# Z100 W8A16 + prefill fp16 物化 — P0 施工记录

> 施工依据：[`docs/design/z100-w8a16-prefill-fp16-construction.md`](../design/z100-w8a16-prefill-fp16-construction.md)。
> 本文档记录 **P0（不碰卡）** 阶段的交付与门禁原始日志；§6.2/§6.3 的数值与性能门禁
> 留待 P1（卡空后）填写。最后更新：2026-10-20。

## 1. 交付状态

| 任务 | 状态 | 落点 |
|---|---|---|
| T1 decode gemv + stream（HAS_A_S / a16 直拷） | ✅ | `csrc/rocm/fp8_emu_gemm.cu` |
| T2 tile（A16Conf 家族，5 cfg + W8A8 cfg 重映射） | ✅ | 同上 |
| T3 header + bindings + dequant op | ✅ | `fp8_emu_gemm.h`、`torch_bindings.cpp` |
| T4 python W8A16 接线（custom_op + 分流 + hook） | ✅ | `gfx906_fp8.py` |
| T5 dequant kernel（新文件，header-only） | ✅ | `csrc/rocm/fp8_dequant.cuh` |
| T6 prefill 路由 + workspace（新模块） | ✅ | `gfx906_fp16_prefill.py` |
| T7 env 开关（默认全关） | ✅ | `vllm/envs.py` |
| T8 单测（已写，待 P1 执行） | ✅ 代码 | `tests/kernels/quantization/test_gfx906_fp8_emu.py` |
| T9 数值/性能/e2e | ⏳ P1 | 本文档 §4 |

Checkpoint commit：`8b8993b21`（T1–T8 初稿）；P0 复审修正见 §3（随其后的 commit）。

## 2. 构建记录（§7 纪律：串行、不碰生产）

生产服务保持运行（P0 约束），因此**没有**就地重链被服务映射的 `.so`：

| 项 | 值 |
|---|---|
| 基线（生产在用，未动） | `/mnt/game/port/build-sync/temp/_rocm_C.abi3.so` md5 `d209af9585aaf0049394ced4a3c3211b` |
| P0 暂存产物 | `/mnt/game/port/build-w8a16-p0/_rocm_C.abi3.so` md5 `475d7eba0468fddab576597a084099d2` |
| 构建方式 | `cp -a` 构建树 → `/mnt/game/port/build-w8a16-p0`，`ninja _rocm_C`（仅重编 `fp8_emu_gemm.hip.o` + `torch_bindings.cpp.o` 并重链） |
| 新增告警 | **0**（build log：`.s23/logs/w8a16-p0-build.log`） |
| 导出符号 | `fp8_emu_gemm_block_w8a16`、`fp8_emu_gemm_block_w8a16_meta`、`fp8_dequant_to_fp16_out` 等 5 个 |

P1 卡空窗口的晋级步骤：停服 → `cp` 暂存 `.so` 覆盖 `build-sync/temp/_rocm_C.abi3.so`
（先备份基线）→ 起 `VLLM_DCU_FP8_W8A16=1`（或 `auto`）跑 §4 → 结果回填本文档。

## 3. P0 复审修正（8b8993b21 之后）

对初稿逐条对照施工图 §3/§4/§8 的复审结论与修正：

1. **gemv 阈值**：初稿硬编码 `M <= kMsMax(8)`；施工图 §3 路由矩阵要求
   `M ≤ VLLM_DCU_FP8_GEMV_MAX_M`。已改为 op 内 `w8a16_gemv_max_m()`（进程内读一次，
   clamp 到 [1,8]，与 python 侧 `_GEMV_MAX_M` 同源）。
2. **decode 臂 cfg**：初稿把 `_prefill_cfg(M,N)` 传进所有调用；§4.4.4 要求 decode 臂
   固定传 0（避免 prefill tile 泄漏进 decode）。已在 `_fp8_emu_mm_block_w8a16_impl`
   内按 M 分流。
3. **A 对齐护栏**（P0 新发现）：W8A8 的 A 永远是 fresh 量化缓冲，W8A16 的 A 是调用方的
   原始张量——所有 W8A16 kernel 以 `uint32` 读 A（stream 变体 `uint4`）。已加：
   入口 `TORCH_CHECK(a % 4 == 0)`（fail loud 而非 IMA）；stream 变体要求 16B 对齐，
   不满足则静默回退非 stream gemv（仍正确）。
4. **host 开销**：`apply_weights` override 初稿无条件调路由器；默认双开关关闭时每层每步
   白付一次调用。已加模块常量 `_PREFILL_ROUTE` 预过滤（开关关闭 = 一次全局读 + 分支）。
5. **bias 次序**：prefill 路由初稿在 matmul dtype 加 bias 后再 cast；§1.2 契约是
   out_dtype 域加（与 emu kernel epilogue 一致）。已调序（仅 out_dtype=fp32 时有差异）。
6. **`a`/`w` 必须是 2-D**：入口补 dim 检查（3-D 张量会被误读 K）。

## 4. 门禁记录

### 4.1 静态门禁（§6.1，全部可做，2026-10-20）

| 门禁 | 结果 | 原始输出 |
|---|---|---|
| `ruff check`（改动 4 文件） | ✅ 与 HEAD 基线同为 9 条既有告警，**0 新增** | 本节下方 |
| Python 导入（容器内，生产 env） | ✅ `W8A16_MODE=0 PREFILL=0 ROUTE=False GEMV_MAX_M=8` | docker exec |
| C++ 构建 | ✅ 0 告警（见 §2 md5 `475d7eba…`） | `.s23/logs/w8a16-p0-build.log` |
| `.s23/probes/isa_gate.py` 暂存 `.so` | ✅ **RESULT PASS**（所有 fp8_emu code object：`v_dot2_f32_f16` 在、`v_mfma`/`v_cmac` 无） | 见下 |
| `ds_read_u16`（同几何模板对比） | ✅ A 侧 LUT 读显著下降 | 见下表 |
| `git diff` W8A8 分支 | ✅ 只有新增分支；W8A8 调用点仅新增 `false`/`true` 常量实参，语义不变 | commit diff |

同几何 tile 的静态 `ds_read_u16` / `v_dot2_f32_f16` 计数（llvm-objdump，按模板实参配对）：

| geometry | W8A8 wide2 | W8A16 | Δ u16 |
|---|---|---|---|
| BK128/BM32/BN32/TM2 | 288 / 2304 | 144 / 2304 | **−50%** |
| BK64/BM32/BN64/TM2 | 216 / 2304 | 144 / 2304 | **−33%** |
| BK64/BM64/BN64/TM4 | 288 / 4608 | 144 / 4608 | **−50%** |

（`v_dot2` 计数逐几何完全一致 ⇒ MAC 流/累加顺序未动；`ds_read_u16` 的减少即
A 侧 LUT 查找消失，W 侧加宽查找保留。）

### 4.2 数值与性能（§6.2/§6.3 — P1，卡空后执行）

- [ ] `pytest tests/kernels/quantization/test_gfx906_fp8_emu.py`（T8 全绿）
- [ ] W8A16 kernel vs fp32 参考：`rel_err ≤ 1e-3`
- [ ] dequant kernel vs `w_q*w_s` fp16 参考：`torch.equal`
- [ ] prefill 路由 vs W8A16 kernel：`cos ≥ 0.99999`；单层真实权重同判据
- [ ] 开关关闭 = 改造前逐 bit（默认臂 pytest 即覆盖）
- [ ] 每形状 × M∈{1,8,16,512,1024,1760,4096,8192} 同窗口 A/B；decode 不回归；
      每层 launch 2→1；prefill `M ≥ min_m` 严格更快
- [ ] e2e：greedy parity、logprobs finite、DFlash2 k=7 accept 不降；32k + 64k soak
- [ ] `min_m` 表实测回填 `gfx906_fp16_prefill._ROLE_MIN_M`
- [ ] 晋级 commit（只改默认值 `auto`/`auto`）

## 5. 已知限制 / P1 备注

- `_ROLE_MIN_M` 仍为占位（`_DEFAULT_MIN_M=4096`，`VLLM_DCU_FP16_PREFILL_MIN_M` 可覆盖）。
- `torch.mm`（hipBLAS）fp16 计算精度：P1 首跑 prefill 路由数值门禁时确认
  `allow_fp16_reduced_precision_reduction` 是否需要关（施工图 §4.6 要点 5）。
- W8A16 tile A 侧 staging 目前按 `uint32`（2 fp16/线程·次）；若 P1 实测 staging 成为
  瓶颈，可评估 `uint4` 化（需 16B 对齐门，stream 变体已有同款护栏）——先测再改。
- 生产服务从未重启、基线 `.so` 未被触碰；P1 晋级前请按 §2 步骤备份。
