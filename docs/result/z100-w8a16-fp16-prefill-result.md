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

- **P1 后生产重启发现的缺陷（已修，commit 756096310）**：施工图 §4.4.2 示例的
  `apply_input_quant = _W8A16 and ops_available(...)` 极性反了——开关关闭时该钩子应为
  True（base class 量化 → W8A8），False 会让 base 把原始 fp16 交给 W8A16 分支，baseline
  `.so` 下首次 forward 即 AttributeError 崩溃循环。单测没抓住的原因：pytest 从不覆盖
  「switch off + apply_weights 全链路」。修复后新增
  `.s24/probes/w8a16_apply_polarity.py` 集成探针：off 臂逐位复现 W8A8（cos=1.00000000）、
  `W8A16=1` 臂逐位复现 W8A16（cos=1.00000000）；route 测试断言已改为正确极性。
- **服务恢复状态**：生产已带新树 + 新 `.so`（`475d7eba…`）以 transient unit 重建并健康
  （health 200，固定 prompt smoke 输出正常、logprobs finite）；serve 脚本不设 W8A16/
  FP16_PREFILL ⇒ 默认关，行为与基线逐 bit 一致。回滚：`cp
  /mnt/game/perf1/_rocm_C.d209af9585….so /mnt/game/port/build-sync/temp/_rocm_C.abi3.so`
  + 重启 unit。
- `_ROLE_MIN_M` 已按 P1 实测回填（§4.2）；`VLLM_DCU_FP16_PREFILL_MIN_M` 仍可覆盖。
- `torch.mm`（实际 rocBLAS，本机无 hipBLASLt gfx906 库）fp16 精度满足 g3/g4 门禁；
  `allow_fp16_reduced_precision_reduction` 无需关闭。
- W8A16 tile A 侧 staging 目前按 `uint32`（2 fp16/线程·次）；tile 相对现状 W8A8 在大 M
  慢 1.3×，但生产上 tile 只服务 M∈[16, min_m) 窄带——若后续要收窄该带，优先调
  `_ROLE_MIN_M`，其次评估 staging `uint4` 化。
- **e2e（greedy parity / DFlash2 accept / 32k+64k soak）未跑**——但按运维决定（2026-10-20
  18:2x），生产已在 e2e 之前**直接启用**新路由：manager env `VLLM_DCU_FP8_W8A16=auto` +
  `VLLM_DCU_FP16_PREFILL=auto` 注入，transient unit 重建，engine 已带新树 + `.so`
  （`475d7eba…`）运行。启用证据链：manager env → unit 继承 → 集成探针（auto ⇒
  `apply_input_quant=False` ⇒ W8A16 / prefill 按 min_m 表路由）→ live 日志无回退告警 →
  多请求 smoke 正常。**注意：W8A16 与 W8A8 数值上必然不逐 bit 相同（激活不量化，更准）**，
  greedy 输出可能与启用前有轻微差异；观察指标 = 异常输出/长稳 OOM/ITL 回归。
  **回滚**：`systemctl --user unset-environment VLLM_DCU_FP8_W8A16 VLLM_DCU_FP16_PREFILL`
  + 重启 unit（或换回基线 `.so`）。e2e parity/soak 仍建议尽快补做。

## 6. 2026-10-20 代码评审修复（Python 侧，无需重编 `.so`）

评审结论：算子本体（kernel / dequant / bindings）未发现正确性缺陷；问题集中在
路由与调优层，且**其中一个已在生产 auto 下生效**。本节记录修复内容与依据，
重启 unit 后生效（生产 `PYTHONPATH` 指向本仓库）。

### 6.1 生产退化：`_ROLE_MIN_M` 漏掉运行时真实层名 `in_proj_qkvz`

Qwen3.5 家族在 fp8 下把 checkpoint 的 `in_proj_qkv` + `in_proj_z` 融合成
`...linear_attn.in_proj_qkvz`（`GatedDeltaNetAttention(create_in_proj_qkvz=True)`，
权重 `[16384,5120]`，TP4 每 rank `4096:5120`，64 层里 48 层）。该 role 不在表里
⇒ 落 `_DEFAULT_MIN_M=4096` ⇒ M∈[16,4096) 走 W8A16 tile。按本仓 P1 日志
`.s23/logs/w8a16-p1-perf.log`（`4096:5120`）：

| M | tile/W8A8 | route/tile | route/W8A8 |
|---|---|---|---|
| 16 | 0.960 | 0.685 | 0.66 |
| 512 | 1.259 | 0.656 | 0.83 |
| 1024 | 1.278 | 0.429 | 0.55 |
| 4096 | 1.290 | 0.394 | 0.51 |

即 48/64 层在 prefill M≥512 上比改造前 W8A8 慢 26–29%，M≥4096 才切到 route。
修复：`_ROLE_MIN_M` 增加 `in_proj_qkvz=16`（同族 split 名 `in_proj_qkv=128`、
`in_proj_z=16` 一并覆盖，待实测替换）；未知 role 首次进路由器打一次 warning，
避免同类表漏再次静默。

### 6.2 其余修复

- **显式 `VLLM_DCU_FP8_W8A16=1` 改为启动期抛错**：`is_supported()==False` 只会落到
  下一个 kernel（gfx906 上即 ~36× 慢的 Triton 软件模拟），并不"报错"；现在 raise。
- **prefill 路由器 compile/eager 边界**：`torch.compiler.is_compiling()` 时直接返回
  `None`（dequant 是无 Meta 的裸 op、M 分支是符号的），被 trace 的路径回到 W8A16
  custom_op；模块 docstring 明确路由器是 eager-only（CUDA graph 下因 workspace
  按 stream 归属也必然拒绝）。
- **宿主侧热点**：`apply_weights` 的 router 前导代码缩短——先算 M/role 并做 min_m
  早退，再构造 `FP8BlockParams`；`_enabled()` 的 op 查询缓存；`_prefill_cfg` 改为
  import 时快照（不再每次调用 `os.environ.get`，decode 臂根本不算 cfg）。
- **bf16 降级**：`_W8A16_ENGAGED` + 逐实例 `apply_input_quant`——auto 下 bf16 层保持
  W8A8（不再首次 forward 崩）；`=1` + bf16 在 `__init__` 抛错。
- **`_GEMV_MAX_M` clamp 到 [0, kMsMax=8]**：env>8 时 python 曾会把 M=9..N 交给
  `fp8_emu_gemv_block`，其入口 `TORCH_CHECK(M<=8)` 会直接失败；同时使 W8A16 的
  "decode 臂恒 cfg 0" 与 kernel 内部阈值一致。
- **显式 `import vllm._rocm_C`**：`torch.ops._rocm_C` 是惰性命名空间，import 期若扩展
  尚未加载，`_W8A16_ENGAGED` 会静默为 False（auto 下静默退回 W8A8）；现在模块 import
  时先注册扩展（失败则保持原行为）。
- **测试**：新增纯 Python 用例（role 名解析、表覆盖、未知 role warning、compile 守卫、
  `=1` 缺 op 抛错、bf16 逐实例降级）与真实 `apply_weights` 全链路用例（on/off 两臂）；
  W8A16/dequant 用例改走 `w8a16_ops` fixture，baseline/回滚 `.so` 下 skip 而不是 ERROR。

### 6.3 仍待处理

1. **e2e 门禁未跑**（§5）：greedy parity / DFlash2 accept / 32k+64k soak。
2. `_ROLE_MIN_M` 的 128/16 交点主要在 11B/TP2 形状上扫得；建议在 27B TP4 的
   `gate_up_proj=4352:5120`、`qkv_proj=2048:5120` 上补扫，并按"vs W8A16 tile"（而非
   vs W8A8）重定 16 阶梯（`out_proj/o_proj` 在 M=16 时 route/tile=1.13）。
3. ~~C++ 硬化项~~：已在 §6.4 的构建窗口完成并晋级（新 `.so` md5 `ff04f1b9…`）。

### 6.4 C++ 硬化批次（2026-10-20 卡空窗口，已构建/晋级/验证）

**改动**（`csrc/rocm/fp8_emu_gemm.cu`、`csrc/rocm/fp8_dequant.cuh`；均无行为变化）：

1. 动态 LDS 显式 `extern __shared__ __align__(16) uint32_t dyn[]`（stream / stream2
   两个 kernel）：`sA` 的 `uint4` 读写此前靠 static LUT 512 B 的布局巧合对齐。
2. W8A16 臂不再传 `a_s=nullptr`，改传设备有效的 `w_s` 指针；`a16`/`HAS_A_S` 关闭时
   kernel 仍不读它，但不再依赖编译器不对 null 做投机加载。
3. dequant helper 把 `CUDAGuard` 移到 `getCurrentCUDAStream()` 之前；kernel 放进匿名
   namespace（header 单 TU 守卫）；host 侧增加 `out != wq` 别名检查。
4. W8A16 入口先校验 `a/w` 为 2-D 再取 `size()`；`check_block_w8a16` 增加跨张量设备
   一致性、`w.data_ptr()%4`、`stride(0)>=K` 检查。
5. `KConf`/`A16Conf` 增加 `BK ∈ {32,64,128}` `static_assert`（否则 LogKw 掩码会静默错）。
6. `w8a16_cfg`：cfg 20..31 显式归 auto；W8A8 cfg 11..19 仅在 `M<=kMsMax` 映射到
   exact-M tile，prefill 调用映射到 cfg 2/1（不再出现 BM=8 的 prefill 悬崖）。

**构建**（§7 串行、staging、不动运行中的服务）：

| 项 | 值 |
|---|---|
| 构建树 | `cp -a` → `/mnt/game/port/build-w8a16-p0`，`ninja _rocm_C`（仅重编 `fp8_emu_gemm.hip.o` + 重链） |
| 新增告警 | **0**（`.s23/logs/w8a16-p1-build.log`） |
| 新产物 | `/mnt/game/port/build-w8a16-p0/_rocm_C.abi3.so` md5 **`ff04f1b9540765367eb4a25184cac0de`** |
| 晋级 | 覆盖 `/mnt/game/port/build-sync/temp/_rocm_C.abi3.so`（现役）；旧版 `475d7eba…` 备份于 `/mnt/game/perf1/_rocm_C.475d7eba0468fddab576597a084099d2.so` |
| 服务 | 未重启、未改动；构建/测试期间保持 active，health 200（新 `.so` 与 Python 修复在下次重启生效） |

**门禁**：

| 门禁 | 结果 |
|---|---|
| `pytest tests/kernels/quantization/test_gfx906_fp8_emu.py`（容器内，`HIP_VISIBLE_DEVICES=3`，GPU flock） | **187 passed / 0 failed**（`.s23/logs/w8a16-p1b-pytest.log`） |
| 机器码 | `.s23/probes/isa_gate.py` 新 `.so`：**RESULT PASS**；`v_dot2_f32_f16` 92742（与旧版逐计数相同）、`v_mfma`/`v_cmac`=0；code object 仅 +48 B |
| 逐 bit A/B（新 `ff04f1b9` vs 旧 `475d7eba`） | `.s23/probes/w8a16_bitwise_ab.py`：W8A8 stream GEMV(m=1/4/8)、W8A8 tile(m=512)、W8A16 stream/tile(m=1/4/8/512)、dequant **9/9 `torch.equal`**（`.s23/w8a16_p1/ab_new.pt` vs `ab_old.pt`） |
| 回滚 | `cp /mnt/game/perf1/_rocm_C.475d7eba….so /mnt/game/port/build-sync/temp/_rocm_C.abi3.so` + 重启 unit |

注：期间新增的 `test_apply_weights_w8a16_full_path` 首跑暴露测试自身的参照错误——W8A8
臂用了测试内 `_quantize_block`（除法求 scale）而 base 走 `per_token_group_quant_fp8`
（倒数乘法），16/8192 个激活码翻转导致 0.69% 假失败；改用库量化器做参照后通过。
内核/宿主代码无问题。
