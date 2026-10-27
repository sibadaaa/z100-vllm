# P5.1 prefill rocBLAS fp16 形状审计

> 施工依据：BUILD §9 WP5.1；设计依据：DESIGN §1/L7、§4/D5、§7
> 属主：A（Lead）。日期：2026-10-24。树 SHA `7078b5fe137feeab3750ee243daeaf172dc439d8`（分支 `dtk`）。
> 栈：4×Z100（gfx906）、DTK 26.04（`DTK-26.04-DCC2602-0317`）、torch 2.11.0、HIP 6.3.26113、TP=4、eager、7469-token prefill。

---

## 1. 一句话结论

**prefill 的 GEMM 几何本身已经贴着 fp16 屋顶；`940–1030 t/s` 目标在本栈没有可执行路径，按 BUILD §9 判读下调为"当前 ~832–840 t/s 为实际可达"。**
逐形状实测 `torch.mm`（M=7469）走 **rocBLAS Tensile `Cijk_Alik_Bljk_HBH_MT128x128x16_…_ISA906`**，单核吞吐 **18.4–19.9 TF/s = fp16 屋顶（19.61 TF/s）的 94–101%**；
**hipBLASLt 在 gfx906 上没有任何 kernel**（DTK 的 hipblaslt/library 只有 gfx928 等），不存在"换库路径"这一手。
单旋钮 `VLLM_DCU_FP16_PREFILL_MIN_M` 在本模型上是**死旋钮**（只改未知角色的默认阈值；本模型 8 个角色全在
`_ROLE_MIN_M` 表里）：MIN_M=1024 臂的 7469 cold TTFT 8906 ms vs 基线 8971 ms（−0.7%，噪声内），路由行为不变。
⇒ 端到端 8.971 s 里约 **5.3 s 是纯 GEMM（≈屋顶）**，剩余 **≈3.7 s（41%）是非 GEMM**（GDN chunked prefill、attention、
量化/物化 dequant、调度/launch），那不是 P5 的范围（DESIGN D5/§9 停止线：不动 attention/GDN、不换库）。

---

## 2. A/B / 审计表

### 2.1 后端与库（`DTK_ROOT=/mnt/game/host-runtime/dtk2604`）

| 检查 | 结果 |
|---|---|
| `torch.version.hip` | `6.3.26113`（DTK 26.04 用户态） |
| `ROCBLAS_TENSILE_LIBPATH` | **未设置**（正确：`tools/dcu/env.sh` §5 明确要求不设，默认位置已编译进 rocBLAS） |
| rocBLAS `lib/rocblas/library` 的 gfx906 条目 | **178 个**（含 `Kernels.so-000-gfx906.hsaco`、全套 `TensileLibrary_*_gfx906.{co,dat}`，含 lazy 库） |
| hipBLASLt `lib/hipblaslt/library` 的 gfx906 条目 | **0 个**（只有 gfx928 等其它 arch；启动日志也报 `rocblaslt warning: No paths matched …/*gfx906*co`） |
| 实际 GEMM 后端 | **rocBLAS Tensile（Cijk … ISA906）**，非 hipBLASLt、非 Tensile 失败回退 |

### 2.2 逐形状 `torch.mm`（fp16，M=7469，单次调用 = 该角色一层；3 次取均值）

| 角色 | 形状 M×N×K | kernel | ms/iter | 有效 TFLOP/s | 对 19.61 TF/s 屋顶 |
|---|---|---:|---:|---:|---:|
| `in_proj_qkvz` | 7469×4096×5120 | `Cijk_Alik_Bljk_HBH_MT128x128x16_…_ISA906` | 16.144 | **19.39** | 98.9% |
| `down_proj` | 7469×5120×4352 | 同上 | 16.778 | **19.85** | 101.2% |
| `o_proj` | 7469×5120×4096 | 同上 | 16.986 | **18.43** | 94.0% |
| `gate_up` | 7469×3584×5120 | 同上 | 13.877 | **19.75** | 100.7% |
| `lm_head` | 7469×1536×5120 | 同上 | 6.040 | **19.44** | 99.1% |

（TFLOP/s = 2·M·N·K / (ms/iter)；屋顶 19.61 TF/s 来自 PROJ §3 的 27B/TP4 fp16 屋顶点。个别 >100% 在计时噪声内。）
⇒ **rocBLAS 形状不是瓶颈**：无论 M=4096 还是 7469、无论 N/K 角色，单 GEMM 都已到屋顶；BUILD §9 的
"现实可达 940–1030 t/s（+13~25%）"所假设的"rocBLAS 形状效率损失"不存在。

### 2.3 `VLLM_DCU_FP16_PREFILL_MIN_M` 单旋钮臂

| 臂 | 启动日志 | 7469 cold TTFT (ms) | step p50 (ms) |
|---|---|---:|---:|
| 基线（MIN_M 未设，=0） | `role min_m {gate_up_proj:128, down_proj:16, …, o_proj:16, out_proj:16}, default 4096` | 8971 | 129.26 |
| `VLLM_DCU_FP16_PREFILL_MIN_M=1024` | 同一张 role 表，`default 1024` | **8906（−0.7%）** | 123.2（噪声） |

代码证据：`gfx906_fp16_prefill.py:86-89`：`_ENV_MIN_M` 只改 `_DEFAULT_MIN_M`（未知角色的兜底），
`_min_m()` 对已知角色返回 `_ROLE_MIN_M[role]`（:139-145）。本模型 8 个角色全在表中 ⇒ 该 env 对路由**零影响**。
（与 WP0.4 发现的 `VLLM_DCU_FP8_CVT` 同为"死旋钮"类，已由策略指纹工具在启动 WARN 中可见。）

---

## 3. env 全文 + 策略指纹 + 树 SHA

- 树 SHA `7078b5fe137feeab3750ee243daeaf172dc439d8`；`vllm/` 无改动。
- 基线臂 env：`.s27/p0/b0-mbt8192-7469.env.txt`；MIN_M 臂 env/日志：`.s27/p5/minm1024.serve.log`。
- 策略指纹：基线 `FP 7e70fc2c130b9191`；MIN_M 臂指纹按启动日志（`VLLM_DCU_FP16_PREFILL_MIN_M` 入指纹）。
- 审计脚本（本地诊断，非交付文件）：`.s27/rocblas_audit.py`；输出 `.s27/p5/`（本次会话 stdout 已记录）。
- 采集命令：

```bash
export DTK_ROOT=/mnt/game/host-runtime/dtk2604
source tools/dcu/env.sh
.venv/bin/python .s27/rocblas_audit.py                     # 后端 + 逐形状
VLLM_DCU_FP16_PREFILL_MIN_M=1024 MBT=8192 MAX_SEQS=4 PORT=8000 bash tools/dcu/serve_host_dflash2.sh &
.venv/bin/python tools/dcu/s23_probe.py --prompt-file docs/design/z100-dual-prefill-reconciliation-projection.md \
  --truncate-tokens 7469 --max-tokens 96 --repeat 2 --tag p5-minm1024 --out .s27/p5
```

---

## 4. 判读（对照 BUILD §9 WP5.1）

| 判据 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| prefill 提升 | ≥10% ⇒ 采纳 | 0%（GEMM 已到屋顶；MIN_M 死旋钮） | 不采纳 |
| prefill 提升 | <5% ⇒ 记为"已到 rocBLAS 形状极限"，目标下调为 840 | 0% | **按此落** |
| 停止线 | 不改 rocBLAS 库/换 DTK 版本 | 未改 | 遵守 |
| 单旋钮 | 只调 `VLLM_DCU_FP16_PREFILL_MIN_M` | 已做，无效果（死旋钮） | 完成 |

**目标修订**：prefill `839.8 → 940–1030 t/s` 的 Target/Stretch 作废；**B0 的 832–840 t/s 就是当前可达值**。
若未来要动 prefill，方向应从"rocBLAS 形状"改成"非 GEMM 的 41%"（GDN chunked prefill、attention、量化/dequant、launch gap），
但那与 DESIGN D5 的"prefill 只动 rocBLAS、不动 attention/GDN"冲突，需 DESIGN 复评后再立项。

---

## 5. 反证条件

- 若存在一个 **hipBLASLt gfx906 库包**（DTK 26.10 或后续版本）能替换 rocBLAS 路径且实测 GEMM >19.61 TF/s，则本结论需重验；
  但 fp16 屋顶由硬件决定，>100% 不可能，故只可能改善**非 GEMM** 部分。
- 若真实服务里的 GEMM 形状与审计脚本不同（例如融合 GEMM、不同 TP 切分、bias 融合），单形状结论不能直接外推；
  但五相位 profile（WP1.2）显示 target 段 256 次/步 GEMM = 38.1 ms 的极大占比已在 decode，prefill 侧的对应证据就是本审计。
- 若 MIN_M 臂的 −0.7% 被证明不是噪声（需要 >4 次重复 + 区间不重叠），则死旋钮结论仍需保留（代码级证据更强）。

## 6. 回滚方式

```bash
# env 臂，无需回滚代码；不设 VLLM_DCU_FP16_PREFILL_MIN_M 即恢复默认
MBT=8192 MAX_SEQS=4 PORT=8000 bash tools/dcu/serve_host_dflash2.sh
```
