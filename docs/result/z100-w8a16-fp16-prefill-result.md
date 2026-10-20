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

P1 窗口（2026-10-20）已将暂存 `.so` 晋级为 `build-sync/temp/_rocm_C.abi3.so`
（md5 `475d7eba…`），基线备份于 `/mnt/game/perf1/_rocm_C.d209af9585….so`；
服务已带新 `.so` 重启（开关默认关 ⇒ 行为不变）。

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

### 4.2 数值与性能（§6.2/§6.3 — 2026-10-20 P1 窗口，HIP_VISIBLE_DEVICES=3）

pytest：**180 passed / 0 failed**（全文件，含 W8A8 既有回归 = 开关关闭逐 bit 门禁）。
初跑的 2 个失败均为测试自身 bug（bf16 臂漏 bias、`w_s[:1]` 空操作），已修，内核无改动。

| 门禁 | 结果 | 说明 |
|---|---|---|
| W8A16 kernel vs fp32 参考（7 生产形状 × M∈{1,8,129,512,4096}） | ✅ 全部 rel_err < 1e-3 | g1，零失败 |
| dequant kernel vs `w_q*w_s` fp16 | ✅ 按判据「torch.equal 或 ≤1 ulp」 | 见下方差异归档 |
| prefill 路由 vs W8A16 kernel（7 形状 @ M=4096） | ✅ cos ≥ 0.99999995 | g3 |
| 真实权重 layers.0 `in_proj_qkv`（10240:5120，checkpoint bf16 scale→fp32） | ✅ cos 0.9999999999 @ M=8；0.99999997 @ M=4096 | g4 |

**dequant 差异归档（P9）**：失配仅出现在精确 fp16 平分点上（约 0.01% 元素），`__float2half`
在本 DTK 上被降低为 magic-multiply 转换（双重舍入），tie 方向随尾数位变化；torch 侧为
RN-EVEN。两侧各距精确值 0.5 ulp、彼此 ≤1 ulp——都是合法最近值，幅度 ≈0.004%，比 fp8
量化噪声低三个量级。g1/g3/g4 全绿证明对下游无影响。

**性能（同窗口背靠背，HIP events，best-of-3；完整行数据
`.s23/logs/w8a16-p1-perf.log` / `-min_m_sweep.log`）**：

| 判定线 | 结果 |
|---|---|
| decode 不回归 | ✅ **全形状提速 1.2–2.1×**（M=1 ratio 0.49–0.79，M=8 ratio 0.48–0.82；quant+widen+gemv 三launch → 单 launch） |
| 每层 launch 数 | ✅ decode 2→1（stream）/ 3→1（tile）；prefill 2→1；prefill 物化 = dequant + rocBLAS 共 2 |
| prefill M ≥ min_m 严格更快 | ✅ M ≥ 512：fp16 物化 = 现状 W8A8 的 0.50–0.70×（如 8704:5120 M=8192：42.6 ms vs 86.0 ms）；rocBLAS 达 16.5–17.7 TF/s vs tile 6.3–6.5 |

**`_ROLE_MIN_M` 实测回填**（`w8a16_min_m_sweep.py`，判据 = fp16 物化严格快于现状 W8A8）：
`gate_up_proj=128`、`down_proj=16`、`o_proj=16`、`out_proj=16`、`qkv_proj=128`（保守覆盖
未实测的 2560:5120 几何），未知 role 维持 `_DEFAULT_MIN_M=4096`。W8A16 tile 相对现状的
已知窄回归带：M∈[16, min_m) 的宽 N 形状（tile 慢 ≤1.31×）。

备注：本机 hipBLASLt 无 gfx906 tensile 库（警告），`torch.mm` 实际走 rocBLAS。

## 5. 已知限制 / 后续

- ✅ `_ROLE_MIN_M` 已按 P1 实测回填（§4.2）；`VLLM_DCU_FP16_PREFILL_MIN_M` 仍可覆盖。
- `torch.mm`（实际 rocBLAS，本机无 hipBLASLt gfx906 库）fp16 精度满足 g3/g4 门禁；
  `allow_fp16_reduced_precision_reduction` 无需关闭。
- W8A16 tile A 侧 staging 目前按 `uint32`（2 fp16/线程·次）；tile 相对现状 W8A8 在大 M
  慢 1.3×，但生产上 tile 只服务 M∈[16, min_m) 窄带——若后续要收窄该带，优先调
  `_ROLE_MIN_M`，其次评估 staging `uint4` 化。
- **e2e（greedy parity / DFlash2 accept / 32k+64k soak）未跑**，需独立服务窗口
  （W8A16=1 臂 + W8A8 臂各起一次）；晋级 commit（默认值 → `auto`）以 e2e 全绿为前提。
- 生产服务恢复：已带新 `.so`（md5 `475d7eba…`）重启，开关默认关 ⇒ 行为与基线逐 bit
  一致；回滚 = `cp /mnt/game/perf1/_rocm_C.d209af9585aaf0049394ced4a3c3211b.so` 回
  `/mnt/game/port/build-sync/temp/`。
