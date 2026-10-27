# WP0.3 纯 n-gram 接受长度上限（LABD / P3 的判死或立项）

> 施工依据：BUILD §4 WP0.3；设计依据：DESIGN §4/D4、§1/L5
> 属主：A（Lead）。日期：2026-10-24。树 SHA `7078b5fe137feeab3750ee243daeaf172dc439d8`（分支 `dtk`）。
> 栈：4×Z100（gfx906）、DTK 26.04、torch 2.11.0、TP=4、c=1、greedy、MBT 8192、`max_num_seqs=4`、util 0.85、`--enforce-eager`。

---

## 1. 一句话结论

**纯 n-gram（`method=ngram_gpu`, `prompt_lookup_min=3`, `prompt_lookup_max=12`, `num_speculative_tokens=7`）在真实 7469-token 文档上
acc = 1.315 tok/step（draft 接受率 0.319 / 4.4–4.6%），落在 BUILD 的 1.0–2.0「边缘」区**：LABD **不单独立项**，
P3（q16 原子包 + lookup 移植）**整段跳过**（BUILD §0：只有 WP0.3 判"立项"才执行 P3）。
附带实测：去掉草稿塔（纯 n-gram）后 step 129.26 → 109.85 ms（−15.0%），这是 wp1.2 profile 之前对"草稿塔 ≈ 19 ms/step"的**直接口径证据**，仅作线索、不作结论。

---

## 2. A/B 表

唯一变量：`--speculative-config`（`SPEC` env 覆盖；WP0.3 前置小改动，见 §3）。同 prompt 源 = 同一份真实文档按 token 截断到 7469。

| 臂 | spec | 策略指纹 | cold TTFT (ms) | step p50 (ms) | acc = tok/step | draft 接受率 | out tok/s |
|---|---|---|---:|---:|---:|---:|---:|
| 对照 = B0（DFlash2 k=7） | `{"method":"dflash","num_speculative_tokens":7,...}` | `7e70fc2c130b9191` | 8971 | 129.26 | **2.667** | 0.664※ | 10.81 |
| 处理（纯 n-gram） | `{"method":"ngram_gpu","prompt_lookup_min":3,"prompt_lookup_max":12,"num_speculative_tokens":7}` | `7e70fc2c130b9191`（SPEC 是 CLI 参数，不进 env 指纹；启动日志单列 `spec=...`） | 10742† | **109.85** | **1.315** | **0.319** | 7.72 |

- 处理臂 2 次重复：step 106.5 / 113.2 ms；tok/step 1.315 / 1.315；接受率 0.32 / 0.32（稳定）。
- ※ 对照臂的"接受率"= probe 的 `accept_len`（accepted/drafted，从引擎 spec 指标反算）；两侧定义一致。
- † ngram 臂 cold TTFT 含首请求内核 JIT 等一次性成本（run2 1.108 s），**不用于判读**；WP0.3 只看 acc。
- 原始 JSON：`.s27/p0/s20_wp03-ngram-7469-real_*.json`；serve 日志 `.s27/p0/ngram-mbt8192.serve.log`；env `.s27/p0/ngram-mbt8192.env.txt`。

引擎侧旁证（`ngram` 臂日志，`metrics.py:120` 两个 10 s 窗口）：

```text
SpecDecoding metrics: Mean acceptance length: 1.33 / 1.31; Avg Draft acceptance rate: 4.6% / 4.4%
```

⇒ 与探针口径一致（tokens/step ≈ 1.31–1.33），不是探针 artifact。

---

## 3. env 全文 + 策略指纹 + 树 SHA

**前置小改动（BUILD WP0.3 要求，已落盘 `tools/dcu/serve_host_dflash2.sh`）**：原 `SPEC` 硬编码改为 env 可覆盖，
默认值与改前逐字节一致（rollback = 不设 `SPEC`）：

```bash
_DEFAULT_SPEC="{\"model\":\"${DRAFT}\",\"method\":\"dflash\",\"num_speculative_tokens\":7,\"dtype\":\"bfloat16\",\"kv_cache_dtype\":\"bfloat16\"}"
SPEC="${SPEC:-${_DEFAULT_SPEC}}"
...
echo "[host-serve] spec=${SPEC}"
```

启动日志（证明覆盖生效）：

```text
[host-serve] spec={"method":"ngram_gpu","prompt_lookup_min":3,"prompt_lookup_max":12,"num_speculative_tokens":7}
[host-serve] policy: FP 7e70fc2c130b9191
```

- 树 SHA：`7078b5fe137feeab3750ee243daeaf172dc439d8`；`vllm/` 无改动。
- `VLLM_TORCH_PROFILER_DIR` 未设；`ngram` 不需要 `model`（`vllm/config/speculative.py` 自动填），日志确认 `SpeculativeConfig(method='ngram_gpu', model='.../Qwen3.8-27B-FP8')`。
- 采集命令：

```bash
SPEC='{"method":"ngram_gpu","prompt_lookup_min":3,"prompt_lookup_max":12,"num_speculative_tokens":7}' \
  MBT=8192 MAX_SEQS=4 PORT=8000 bash tools/dcu/serve_host_dflash2.sh > .s27/p0/ngram-mbt8192.serve.log 2>&1 &
.venv/bin/python tools/dcu/s23_probe.py --prompt-file docs/design/z100-dual-prefill-reconciliation-projection.md \
  --truncate-tokens 7469 --max-tokens 96 --repeat 2 --tag wp03-ngram-7469-real --out .s27/p0
```

---

## 4. 判读（对照 BUILD §4 WP0.3 阈值）

| 判据 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| 立项 LABD/P3 | acc ≥ 2.0 | 1.315 | 不立项 |
| 边缘（只在 q16+三阻塞修复后试，不单独立项） | acc 1.0–2.0 | **1.315 ∈ 区间** | **边缘** |
| 砍掉 P3 整段 | acc ≤ 1.0 | — | 不触发 |
| 停止线：ngram_gpu 起不来且与 gfx906 无关 ⇒ 记录、跳过、按"未判定" | — | 服务正常起、探测正常 | 不触发 |

**执行动作**：按 BUILD §0「P3 只在 WP0.3 结论为立项时执行，否则整段跳过，写清原因」——
**P3（WP3.1/3.2/3.3/3.4）整段跳过**，原因：纯 n-gram 的物理上限在本工作负载上只有 1.315 tok/step，
即便 LABD 把第 8–15 位全填满、三个阻塞全修完，收益上限相对 q8 基线（2.667）也仅 ~1.6×，且需要 4–7 天 + 三项内核/调度改动（DESIGN §4/D3、PROJ §11.3），
不满足"单独立项"门槛。接受长度这条杠杆按 BUILD 降级为"仅能通过换 drafter 解决，不在本计划范围内"。

**附带线索（不作结论）**：同一 prompt、同一 MBT，纯 n-gram 臂 step 109.85 ms vs DFlash2 臂 129.26 ms，
差 **19.4 ms/step（−15.0%）**。纯 n-gram 没有草稿塔前向，所以这 19.4 ms 是"DFlash2 drafter 塔 + verify 形状差异"的上界估计，
与 PROJ §5"草稿塔+lm_head 36.9 ms（240.9 ms 旧 step 口径）"同向。**该数字进 WP1.2 五相位 profile 后从头核对。**

---

## 5. 反证条件

- 若用户的真实 7469 文档是**高重复性文档**（代码/表格/模板），纯 n-gram acc 可能显著高于本报告用的技术文档；此时 WP0.3 需用用户文档重跑。
  这是本工作包最大的外部效度风险（PROJ §1 的教训：acc 是 prompt 属性）。
- 若 `ngram_gpu` 的 `prompt_lookup_min=3/max=12` 不是最优参数（本报告按 BUILD 给定值，未扫参），最优参数下的上限可能更高；
  但 BUILD 的判据是"纯 n-gram 上限"，不存在"扫参到 ≥2 才立项"的许可。
- q16（`num_speculative_tokens=15`）下接受率可能不同；BUILD 明确边缘情况只在三阻塞修完后才试，本次未执行。

## 6. 回滚方式

```bash
# 不设 SPEC 重启即回到 DFlash2 k=7；代码层保留 WP0.3 的 SPEC env 覆盖（默认值逐字节等于改前）
MBT=8192 MAX_SEQS=4 PORT=8000 bash tools/dcu/serve_host_dflash2.sh
```
