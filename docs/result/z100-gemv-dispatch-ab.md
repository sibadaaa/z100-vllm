# WP0.2 GEMV 派发 A/B（`VLLM_DCU_FP8_GEMV_MAX_M` 8 vs 1）

> 施工依据：BUILD [`z100-decode-prefill-optimization-construction.md`](../design/z100-decode-prefill-optimization-construction.md) §4 WP0.2
> 设计依据：DESIGN §1/L2、§4/D2、§8（"`GEMV_MAX_M=16` 被编译期上限静默截到 8"）
> 属主：A（Lead）。日期：2026-10-24。树 SHA `7078b5fe137feeab3750ee243daeaf172dc439d8`（分支 `dtk`）。
> 栈：4×Z100（gfx906）、DTK 26.04、torch 2.11.0、TP=4、DFlash2 k=7、c=1、greedy、MBT 8192、`max_num_seqs=4`、util 0.85、`--enforce-eager`。

---

## 1. 一句话结论

**两臂无 ≥5% 差异，按 BUILD 停止线判"W8A16 路径的 GEMV 已不是瓶颈"：不采纳 `GEMV_MAX_M=1`，P2 权重移到 L3（allreduce）/L4（rocBLAS 草稿塔 + lm_head）。**
真实 7469 文档上 step p50 **129.26 → 134.27 ms（+3.9%，变慢）**；36k 合成 prompt 上 **142.60 → 139.28 ms（−2.3%，变快）**；
两个 prompt 源方向相反且都 <5% ⇒ 不可判定、无生产收益。G1 对拍 **8/8 逐 token 一致**（派发策略变化不改数值）。

---

## 2. A/B 表

| 臂 | 唯一变量 | 策略指纹 | 启动日志证据 | 真实 7469 step p50 (ms) | 合成 36k step p50 (ms) | G1 对拍 |
|---|---|---|---:|---:|---:|---|
| `gemv8`（对照 = B0） | `VLLM_DCU_FP8_GEMV_MAX_M=8` | `7e70fc2c130b9191` | `W8A16 engaged=True (built op True), GEMV_MAX_M=8.` | 129.26（135.2 / 123.3） | 142.60（142.9 / 142.3） | 基准 8/8 |
| `gemv1`（处理） | `VLLM_DCU_FP8_GEMV_MAX_M=1` | `e472fc80881fd4fe` | `W8A16 engaged=True (built op True), GEMV_MAX_M=1.` | 134.27（130.3 / 138.3） | 139.28（139.3 / 139.2） | **8/8 与基准逐 token 相同** |
| **Δ** | — | 指纹按预期改变 | 派发旋钮被真实消费 | **+3.87%** | **−2.33%** | 一致 |

- 真实 7469 prompt：`docs/design/z100-dual-prefill-reconciliation-projection.md` 按 token 截断到 7469（WP0.5 探针）。
- 合成 36k：`s23_probe.py --prompt-tokens 7469`（实际 36,238 token，见 B0 报告 §2.2；保留此源是为了与历史 S18 口径对照）。
- 每臂 2 次重复；单 run 波动 ±5–6%（见括号内两值），小于该噪声的效应不能当收益。
- 原始 JSON：`.s27/p0/{s20_b0-mbt8192-7469,fileprobe_real7469-proj,s20_gemv1-synth7469,...}.json`；diff/algo 证据：`docs/result/z100-p0-fingerprint-probe.md` §WP0.4-2。

### 2.1 旋钮确实生效（不是静默无效）

1. 启动日志逐臂打印 `GEMV_MAX_M=8` / `GEMV_MAX_M=1`（`vllm/model_executor/kernels/linear/scaled_mm/gfx906_fp8.py:737`）。
2. 策略指纹按预期改变（`7e70fc2c…` → `e472fc80…`），且 WP0.4 工具对 W8A16 派生键 `FP8_GEMV_MAX_M_W8A16` 单列。
3. 代码路径：Python 侧 `_GEMV_MAX_M` 只决定 prefill tile（`gfx906_fp8.py:1032`），**W8A16 的 GEMV/GEMM 阈值在 C++ kernel 内部**
   以同一 env 解析（`env==null → 8；否则 max(1, min(v, 8))`，=0 也会被抬到 1），所以本 A/B 覆盖的正是 verify 形状 M=8 的派发。

---

## 3. env 全文 + 策略指纹 + 树 SHA

- 树 SHA：`7078b5fe137feeab3750ee243daeaf172dc439d8`；`vllm/` 无改动。
- 控制臂 env：`.s27/p0/b0-mbt8192-7469.env.txt`（FP `7e70fc2c130b9191`）
- 处理臂 env：`.s27/p0/gemv1-mbt8192-7469.env.txt`（FP `e472fc80881fd4fe`）
- 两臂仅一行不同：`VLLM_DCU_FP8_GEMV_MAX_M=8` → `=1`（其余 73 行逐字相同）。
- 采集命令：

```bash
# 臂 1（对照）
MBT=8192 MAX_SEQS=4 PORT=8000 bash tools/dcu/serve_host_dflash2.sh > .s27/p0/gemv8.serve.log 2>&1 &
# 臂 2（处理）
VLLM_DCU_FP8_GEMV_MAX_M=1 MBT=8192 MAX_SEQS=4 PORT=8000 \
  bash tools/dcu/serve_host_dflash2.sh > .s27/p0/gemv1-mbt8192-7469.serve.log 2>&1 &
# 每臂：真实文档 7469 + 合成 36k 各 repeat 2；再跑 G1 对拍
.venv/bin/python tools/dcu/s23_probe.py --prompt-file docs/design/z100-dual-prefill-reconciliation-projection.md \
  --truncate-tokens 7469 --max-tokens 96 --repeat 2 --tag gemv8 --out .s27/p0
.venv/bin/python .s26/graph/parity.py --base http://127.0.0.1:8000 --tag gemv1 --out .s27/p0/parity
```

---

## 4. 判读（对照 BUILD §4 WP0.2 阈值）

| 判据 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| 采纳（进 P1 生产候选） | step 降 ≥15% | 真实 +3.9% / 合成 −2.3% | **不采纳** |
| 部分成立 | step 降 5–15% | 无源达到 5% | 不适用 |
| 停止线 | step 降 <5% ⇒ 停，GEMV 不是瓶颈，P2 权重移 L3/L4 | 两源 \|Δ\|<5% 且方向相反 | **停** |
| G1 数值 | 8 条 greedy 逐 token 一致 | `ALL_IDENTICAL`（`.s27/p0/parity/parity_{gemv8,gemv1}.json`） | **过** |
| C3 单变量 | 只改 `VLLM_DCU_FP8_GEMV_MAX_M` | `git diff` 为空（env 臂）；env 快照仅 1 行不同 | **过** |

**判读结论**：S18 的"GEMV 在 M=8 是平坦 GEMM 的 7.38×"是在 **W8A8 fp8-emu** 路径上测的；在生产 W8A16 路径上该收益
**不传导到端到端**（DESIGN §7 风险表第 2 行命中的正是这一情形）。因此：
1. P2 不再考虑重写 GEMV 内核；
2. P2 的权重按 DESIGN 候选顺序移到 **L3 allreduce / L4 rocBLAS 草稿塔+lm_head**，具体分配等 WP1.2 五相位 profile；
3. `GEMV_MAX_M` 在生产保持默认 **8**（回滚即默认值，无需动作）。

---

## 5. 反证条件

- 若在**同一台机、同一窗口**用 ≥4 次重复测得真实 7469 的 step 降 ≥5%（当前 2 次重复的 run 间噪声 ±5–6% 足以掩盖 3–4% 的效应），
  则本结论推翻，应重开 WP0.2。
- 若有人证明生产 W8A16 路径实际上没有消费该 env（本报告的 3 条间接证据不够），则本 A/B 是空实验，需改用 kernel 级 trace 确认派发。
- 若把 `_GEMV_MS_MAX` 从 8 重编到 16 后出现新效应，那是另一个工作包（DESIGN §8 明确 `=16` 不能靠 env 绕）。

## 6. 回滚方式

一条命令（env 臂，停服务即回）：

```bash
# 直接不设该变量（默认 8）重启即可；若已设为 1：
VLLM_DCU_FP8_GEMV_MAX_M=8 MBT=8192 MAX_SEQS=4 PORT=8000 bash tools/dcu/serve_host_dflash2.sh
```
