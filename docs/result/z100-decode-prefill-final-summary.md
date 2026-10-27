# Z100 decode / prefill 优化施工总结（v1，2026-10-24）

> 设计图：[`z100-decode-prefill-optimization-design.md`](../design/z100-decode-prefill-optimization-design.md)（DESIGN）
> 施工图：[`z100-decode-prefill-optimization-construction.md`](../design/z100-decode-prefill-optimization-construction.md)（BUILD）
> 数据依据：[`z100-dual-prefill-reconciliation-projection.md`](../design/z100-dual-prefill-reconciliation-projection.md)（PROJ）
> 执行日期：2026-10-24（Asia/Shanghai 22:09 起）；树：`/home/lyh/z100-vllm`，分支 `dtk`，
> **HEAD `7078b5fe137feeab3750ee243daeaf172dc439d8`**（本会话改动的文件见 §9）。
> 平台：4×Z100/Z100L（gfx906）、DTK **26.04**、torch 2.11.0、Python 3.10、`Qwen3.8-27B-FP8` + DFlash2 k=7、TP=4、MBT 8192、`max_num_seqs=4`、util 0.85。
> 运行方式：全部**物理机裸机**（容器内直接起服务，无 Docker），`DTK_ROOT=/mnt/game/host-runtime/dtk2604`。

---

## 1. 一页结论

1. **B0 基线复现成功**：真实 7469-token 文档、MBT 8192 下 step p50 **129.26 ms**（参考 129，129±6 内）、
   cold TTFT **8.971 s / prefill 832.6 t/s**（参考 8.894 s / 839.8，差 **+0.9% / −0.9%**）。
   ⚠ 发现 BUILD §3.3 的口径错误：`s23_probe.py --prompt-tokens 7469` 实际造的是 **36,238 token** 的合成 prompt
   （每个 `itemN` 词被切成 ~4.85 个 token），**必须用 `--prompt-file + --truncate-tokens` 才能量"7469"**。
2. **唯一被采纳的性能改动 = 图模式路线 A（WP1.1）**：step **129.26 → 114.17 ms（−11.67%）**、cold TTFT
   **+0.77%**、输出 **+10.0%**、G1 8/8 逐 token 一致、C2 四档上下文全部更快（−9.8%~−20.1%）。
   代价：冷启动 150 → 255–369 s、KV −6.2%、宿主内存压力（需 ≥16 GiB swap，实配 51 GiB）。
3. **三项"看起来最大"的杠杆被实测否掉**：
   - **L2 `VLLM_DCU_FP8_GEMV_MAX_M=1`**：真实 7469 上 step +3.9%、36k 合成上 −2.3%，**均 <5% 且方向相反 ⇒ 不采纳**，
     W8A16 路径的 GEMV 不是瓶颈（S18 是 W8A8 测的）。
   - **LABD/q16（P3）**：纯 n-gram 上限 acc = **1.315 tok/step**（BUILD 判据 1.0–2.0 属"边缘、不单独立项"）⇒ **P3 整段跳过**。
   - **prefill 940–1030 t/s（P5）**：逐形状 GEMM 已达 **18.4–19.9 TF/s = fp16 屋顶的 94–101%**，hipBLASLt 在 gfx906
     **零 kernel**，`FP16_PREFILL_MIN_M` 是死旋钮 ⇒ **目标下调为当前 ~832–840 t/s**。
4. **五相位 profile（G5）通过**：`draft 33.83 / draft→target 0 / target 77.81 / target→draft 4.76 / round 116.40 ms`，
   与同臂独立 `srv_step_ms 114.17 ms` 差 **1.95% < 5%**（差额归因 profiler 开销 ≈2.0%）。target 段 256 次/步 GEMM（38.07 ms）
   是最大单项，但 GEMV A/B 已证明该路径没有便宜的回收口。
5. **最终 Must 未全绿**：decode tok/s **✓**、G1 **✓**、C2 **✓**；**TTFT ✗（9.04 s vs ≤8.65 s）**、
   **prefill t/s ✗（826–833 vs ≥866）**——两者同源（prefill），而 P5 已证明本栈没有合规的 prefill 提升路径。
   G8（15 min soak）未做满。⇒ 交付形态 = **"decode 配方已采纳（图路线 A），TTFT/prefill Must 按证据下调"**，非"全绿"。

---

## 2. 施工进度（对照 BUILD §0 任务板）

| 阶段 | 工作包 | 状态 | 产物/结论 |
|---|---|---|---|
| **P0** | WP0.1 基线冻结 | ✅ | [`z100-b0-freeze.md`](z100-b0-freeze.md)：6 点全有；主点 129.26 ms / 8.971 s / 832.6 t/s |
| | WP0.2 GEMV 派发 A/B | ✅ 判停 | [`z100-gemv-dispatch-ab.md`](z100-gemv-dispatch-ab.md)：\|Δ\|<5%，不采纳，P2 权重移 L3/L4 |
| | WP0.3 纯 n-gram 上限 | ✅ 判"边缘" | [`z100-ngram-upper-bound.md`](z100-ngram-upper-bound.md)：acc 1.315 ⇒ P3 跳过 |
| | WP0.4 策略指纹 | ✅ | [`z100-p0-fingerprint-probe.md`](z100-p0-fingerprint-probe.md)：`FP 7e70fc2c130b9191`，109 旋钮实际生效值 + clamp WARN |
| | WP0.5 探针 `--prompt-file` | ✅ | 同上；真实文档 token 计数 + `--truncate-tokens` + `--dry-run`，live 校验通过 |
| **P1** | WP1.1 图路线 A | ✅ **采纳** | [`z100-routea-deploy.md`](z100-routea-deploy.md)：step −11.67%、TTFT +0.77%、G1 8/8、C2 四档全绿 |
| | WP1.2 五相位 profile | ✅ **G5 PASS** | [`z100-five-phase-profile.md`](z100-five-phase-profile.md)：五段和 vs step 差 1.95% |
| | WP1.3 接受长度扫描 | ✅ 判"内容效应" | [`z100-acceptance-vs-context.md`](z100-acceptance-vs-context.md)：4.80/2.29/3.00/2.67，非单调 |
| **P2** | 按 profile 回收 | ⏸ **未执行** | 见 §5：profile 下无"≥5 ms/步且有屋顶"的合规项（L3 allreduce 9.7 ms < 20 ms 门槛；L4 草稿塔需另立 dtype/形状审计） |
| **P3** | q16 原子包 | ⏭ **整段跳过** | WP0.3 acc 1.315 ∈ [1.0,2.0] 边缘，不单独立项（BUILD §0） |
| **P4** | chunk 边界根因 | ✅ 出报告、不修 | [`z100-chunk-boundary-rootcause.md`](z100-chunk-boundary-rootcause.md)：>200 ms/边界；排除物化，残余在 chunked-prefill 公共路径 |
| **P5** | rocBLAS fp16 形状审计 | ✅ 目标下调 | [`z100-prefill-rocblas-audit.md`](z100-prefill-rocblas-audit.md)：GEMM 到屋顶、无库路径可换 |

---

## 3. 关键数字总表（同一真实 7469-token 文档、c=1、greedy、MBT 8192）

| 指标 | B0（eager） | 采纳配方（graph-A） | Δ | Must/Target |
|---|---:|---:|---:|---|
| decode step p50 (ms) | 129.26 | **114.17** | **−11.67%** | — |
| acc（tok/step） | 2.667 | 2.909 | +9.1% | — |
| decode tok/s（1000·acc/step） | 20.63 | **25.48** | +23.5% | **Must ≥23.7 ✓**，Target 28.2 ✗ |
| cold TTFT @7469 (ms) | 8971 | 9040 | +0.77% | **Must ≤8650 ✗（+4.5%）** |
| prefill t/s @7469 | 832.6 | 826.2 | −0.8% | **Must ≥866 ✗（−4.6%）** |
| KV cache (tok) | 759,509 | 712,748 | −6.2% | 容量代价 |
| 冷启动 (s) | 150 | 255–369 | +70~146% | 运维代价 |
| G1 8×32 greedy | 基准 | **8/8 逐 token 一致** | — | ✓ |
| C2 1K/2K/4K/8K step | 128.6/124.9/125.2/128.5 | **102.7/108.9/111.0/115.9** | −20.1/−12.8/−11.3/−9.8% | ✓ |

> 短上下文（WP0.1，eager）：512 tok TTFT 770 ms；1024 tok 1147 ms；2048 tok 2450 ms；prefill 依次 665/893/836 t/s（固定开销主导）。
> `@512` 的现状与 DESIGN 参考 1.13 s 不同，因为本报告用真实文档 + cold prefill；跨 prompt 源不可直比。
> **跨 prompt 源注意**：acc 是本报告文档的属性（2.667），用户自己的 7469 文档是 2.84；decode tok/s 的 Must 比较需按同源 acc 折算
> （用 2.84 折算：B0 22.0 → graph **24.87 tok/s**，仍 ≥23.7）。

---

## 4. 门禁状态（G0–G9）

| 门 | 状态 | 证据 |
|---|---|---|
| G0 栈身份/4 卡空闲/swap≥16 GiB | ✅ | 4×Z100 0% 显存空闲；DTK 26.04；swap 51 GiB（4+16+32 GiB） |
| G1 数值（8×32 greedy） | ✅ | WP0.2/WP1.1 两次 A/B 均 8/8 逐 token 一致 |
| G2 静态检查 | ✅（等价口径） | `.venv` 无 pre-commit；用 ruff check/format 对改动 Python 全绿，shell 用 `bash -n` |
| G3 策略指纹 | ✅ | 每个 serve 日志首行 `policy: FP …`；报告引用同一 FP |
| G4 P0 五包 | ✅ | §2 表 |
| G5 五相位 | ✅ | 1.95% < 5% |
| G6 q16 原子包 | N/A | P3 跳过 |
| G7 C2 四档上下文 | ✅（decode）/N/A（prefill 无改动） | WP1.1 §2.1 |
| G8 2 重启 + 15 min c1/c4 soak | ❌ 未做满 | graph 臂起停 3 次无 crash；15 min soak 与 c=4 未跑 |
| G9 回滚演练 | ⚠ 实质完成 | 生产脚本本会话起停 ≥6 次，均回到 eager B0；未写独立回滚脚本 |

---

## 5. P2 分配（按五相位 profile 的实测）

BUILD §6 的准入：可回收 ≥5 ms/步 **且** 有对应【实测】屋顶。五相位表：

| 候选 | profile 实测 | BUILD 门槛 | 结论 |
|---|---:|---:|---|
| L3 TP allreduce | target `nccl_Generic_4` 8.28 ms + draft `allreduce_f16` 1.20 + `allgather_i8` 0.19 ≈ **9.7 ms/步** | ≥20 ms/步 | **不立项** |
| L4 rocBLAS 草稿塔 + lm_head | draft 段 33.83 ms（其中 rocBLAS Cijk 5.77、triton_matmul 13.34）+ target→draft 4.76 ms ≈ **38.6 ms/步** | ≥25 ms/步 | **有资格**，但动作是"先查 dtype/形状"，属新工作包（草稿塔 bf16 vs target fp16），需另立并过 G1/C2；本窗口未执行 |
| GEMV 残余 | — | 仅当 WP0.2 部分成立 | 不成立，不做 |

⇒ P2 **未执行**（无合规的"改一行就回收 ≥5 ms"项）；L4 是唯一值得下一轮立项的方向，建议：draft 塔 bf16→fp16 / 形状审计 +
G1/C2 验证（预期上界 ~38.6 ms/步里的一部分，需先量化 bf16 GEMM 与 fp16 的差）。

---

## 6. 本会话对 BUILD 口径的修正（重要，后续引用必须带）

| # | BUILD/DESIGN 原假设 | 本会话实测/读码 | 影响 |
|---|---|---|---|
| 1 | `s23_probe --prompt-tokens N` 是 N token | 实际 **≈4.85N token**（7469→36238） | 历史 "512-token prompt" 实为 ~1940 token；B0/P1 一律改用 `--prompt-file` |
| 2 | L2 `GEMV_MAX_M=1` 可能是全表最大单项 | 真实 7469 +3.9%、36k 合成 −2.3%（均 <5%） | L2 判停，GEMV 重写不做 |
| 3 | P3 接受长度可修到 4.0 | 纯 n-gram 上限 1.315；同文档 acc 随截断点 4.80/2.29/3.00/2.67 非单调 | P3 跳过；"接受长度长度效应"不成立 |
| 4 | prefill 余量在 rocBLAS fp16 M 形状 | 逐形状 GEMM 18.4–19.9 TF/s ≈ 屋顶；hipBLASLt gfx906 无 kernel | prefill 目标下调；剩余 41% 在非 GEMM |
| 5 | 337 ms/chunk 根因首选 GDN 前缀重算 | 关掉 fp16 物化后每 chunk 额外成本仍在（0.3–0.42 s/chunk）⇒ 物化被排除；残余在 chunked-prefill 公共路径 | 不改 GDN；MBT≥prompt 兜底 |
| 6 | `VLLM_TORCH_PROFILER_DIR` 可开 profiler | 本树为死变量，须 CLI `--profiler-config '{"profiler":"torch",…}'` | WP1.2 采集方式 |
| 7 | `VLLM_DCU_FP16_PREFILL_MIN_M` 可调路由阈值 | 只改未知角色默认值，本模型 8 角色全在表 ⇒ 死旋钮 | P5 单旋钮无效 |

---

## 7. 事故与风险（C5 口径）

1. **profiler OOM**：`with_stack=true` 时 20 步 × 4 rank 的 trace 达 53–58 MB/rank，dump 时宿主 31 GiB RAM + 19 GiB swap 被击穿，
   内核 OOM kill 一个 `VLLM::Worker_TP`（dmesg 15:00:38）。**处置**：加 32 GiB swap（总 51 GiB）+ 改 `torch_profiler_with_stack=false`
   （trace 降到 3.4 MB/rank，全程无 OOM）。
2. **`/stop_profile` segfault**：本栈 profiler 关闭路径 `hsa_stop undefined symbol` → API server 挂掉，因此**同窗口 /metrics 拿不到**；
   G5 用"同臂相邻非 profile 基线 114.17 ms + trace 自洽周期 116.40 ms"，差 1.95%，并在 WP1.2 报告写明反证条件。诊断数据不混入生产（C5）。
3. **graph 臂数值**：G1 8/8 一致，但接受长度从 2.667 升到 2.909（draft 数值路径变化）；greedy 输出不变。跨重启可复现性未验证（SLA 场景仍建议 eager）。
4. **未完成门**：G8（15 min c1/c4 soak）、P2 L4、P5 非 GEMM 深挖、c=4 混批。

---

## 8. 交付物索引

| 内容 | 路径 |
|---|---|
| B0 基线冻结 | [`docs/result/z100-b0-freeze.md`](z100-b0-freeze.md) |
| GEMV 派发 A/B | [`docs/result/z100-gemv-dispatch-ab.md`](z100-gemv-dispatch-ab.md) |
| n-gram 上限 | [`docs/result/z100-ngram-upper-bound.md`](z100-ngram-upper-bound.md) |
| 指纹 + 文档探针 | [`docs/result/z100-p0-fingerprint-probe.md`](z100-p0-fingerprint-probe.md) |
| 图路线 A 部署 | [`docs/result/z100-routea-deploy.md`](z100-routea-deploy.md) |
| 五相位 profile（G5） | [`docs/result/z100-five-phase-profile.md`](z100-five-phase-profile.md) |
| 接受长度 vs 上下文 | [`docs/result/z100-acceptance-vs-context.md`](z100-acceptance-vs-context.md) |
| chunk 边界根因 | [`docs/result/z100-chunk-boundary-rootcause.md`](z100-chunk-boundary-rootcause.md) |
| prefill rocBLAS 审计 | [`docs/result/z100-prefill-rocblas-audit.md`](z100-prefill-rocblas-audit.md) |
| 本文（总总结） | [`docs/result/z100-decode-prefill-final-summary.md`](z100-decode-prefill-final-summary.md) |
| 图臂 launcher | `tools/dcu/s27_graph_launch.sh`（派生自 `serve_host_dflash2.sh`，两处 diff） |
| 五相位采集/解析 | `tools/dcu/s27_five_phase_profile.py`（`collect`/`parse`/`--selftest` 27/27） |
| 策略指纹 | `tools/dcu/dcu_policy_fingerprint.py`（`--selftest` 41/41） |
| 真实文档探针 | `tools/dcu/s20_single_probe.py`（新增 `--prompt-file/--truncate-tokens/--dry-run`） |
| 生产启动脚本改动 | `tools/dcu/serve_host_dflash2.sh`（WP0.3 `SPEC` env 覆盖 + WP0.4 指纹行；默认行为不变） |
| 本地产物（不入库） | `.s27/`（p0/p1/p4/p5 日志与 JSON、b-fingerprint、profile trace） |
| **物理机启停脚本（用户交付）** | `~/server/scripts/local-llm-start.sh` / `local-llm-stop.sh`（见 §13） |

---

## 9. 源码/脚本改动清单（树内）

```text
 M .gitignore                              # 删除误伤的整目录 tools/dcu 规则（恢复既有白名单意图）
 M tools/dcu/serve_host_dflash2.sh         # + WP0.3 SPEC env 覆盖（默认值逐字不变）+ WP0.4 指纹 echo 行 + spec 日志行
 M tools/dcu/s20_single_probe.py           # WP0.5 加法式：--prompt-file / --truncate-tokens / --dry-run / max_model_len guard
 M tools/dcu/perf1_gdn_parity.py           # 会话前已存在的改动，未触碰
?? tools/dcu/dcu_policy_fingerprint.py     # WP0.4 新增
?? tools/dcu/s27_graph_launch.sh           # WP1.1 新增
?? tools/dcu/s27_five_phase_profile.py     # WP1.2 新增
?? docs/result/z100-*.md                   # 本总结 §8 的九份报告
```

`vllm/` 下**零改动**；没有 git commit（按会话约定不提交）。

---

## 10. 复现（最短路径）

```bash
cd /home/lyh/z100-vllm
export DTK_ROOT=/mnt/game/host-runtime/dtk2604

# (a) 生产 / 回滚配方（eager）
MBT=8192 MAX_SEQS=4 PORT=8000 bash tools/dcu/serve_host_dflash2.sh &

# (b) 采纳的图路线 A（候选，需补 G8 soak）
MBT=8192 MAX_SEQS=4 PORT=8000 bash tools/dcu/s27_graph_launch.sh &

# 真实 7469-token 探测（同源可比）
.venv/bin/python tools/dcu/s23_probe.py \
  --prompt-file docs/design/z100-dual-prefill-reconciliation-projection.md \
  --truncate-tokens 7469 --max-tokens 96 --repeat 2 --tag check --out .s27/check

# 策略指纹（应打印 FP 7e70fc2c130b9191）
.venv/bin/python tools/dcu/dcu_policy_fingerprint.py --prefix "[policy] "
```

---

## 11. 下一步建议（按收益/风险排序）

1. **补 G8**：graph 臂 2 次重启 + 15 min c1/c4 soak（0 crash/OOM/NaN），过了再把 `s27_graph_launch.sh` 升为生产默认；否则保留 eager。
2. **P2-L4 立项**：草稿塔 bf16（profile 33.8 ms/步，其中 rocBLAS 5.8 / triton_matmul 13.3）+ lm_head 4.8 ms/步；先做 dtype/形状审计，
   目标回收 ≥5 ms/步，按 C3/G1/C2 流程走。**这是当前 profile 下唯一 ≥5 ms/步的候选。**
3. **prefill 非 GEMM 的 41%**（GDN chunked prefill/attention/量化）——需 DESIGN 复评是否打破"D5 不动 attention/GDN"。
4. **MBT 策略**：生产默认 4096，但 7469 长 prompt 用 8192 单 chunk 可省 ~812 ms；建议按最大 prompt 自适应（需 scheduler 侧改动，另立项）。
5. **用户 7469 文档**：用 WP0.5 探针在用户的真实 prompt 上复测 step/acc/tok-s，确认 2.84 的 acc 可复现（本报告用技术文档 acc=2.667）。

---

## 12. 最终验收表（Must）

| 指标 | Must | 实测（graph-A，同源真实 7469） | 判定 |
|---|---:|---:|---|
| decode tok/s @7469 | ≥23.7 | **25.48**（按用户 acc 2.84 折算 24.87） | **✓** |
| TTFT @7469 | ≤8.65 s | **9.04 s**（eager 8.97 s） | **✗**（+4.5%，prefill 无合规提升路径） |
| prefill t/s @7469 | ≥866 | **826–833** | **✗**（−4.6%；P5 证明 GEMM 到屋顶） |
| C1 8 条 greedy | 全一致 | 8/8（graph vs eager） | **✓** |
| C2 四档上下文 | 无退化 | 1K/2K/4K/8K 全更快 | **✓** |
| G7–G9 | 全通过 | G7 ✓；G8 未做满；G9 实质演练 | **✗（部分）** |

**结论：施工按图完成，decode 侧拿到并采纳了唯一被实测支持的 −11.7% step；TTFT/prefill 两项 Must 被实测证明在本栈不可达（P5），
按 BUILD §9 的判读规则下调为"当前 832–840 t/s 为实际可达"，因此不属于施工失败，而是目标修订。**

---

## 13. 物理机本地大模型启停脚本（`~/server/scripts/`）

本轮新增并**已实测**：

| 脚本 | 功能 | 实测 |
|---|---|---|
| [`~/server/scripts/local-llm-start.sh`](../../../server/scripts/local-llm-start.sh) | 物理机裸机启动 Qwen3.8-27B-FP8 + DFlash2 k=7；默认 `ARM=graph`（图路线 A，采纳配方），`ARM=eager` 回退生产配方；默认 `PORT=8100`、`MBT=8192`、`MAX_SEQS=4`；预检端口/残留进程/显存/swap；等待 `/health`（最多 20 min）；打印 endpoint/策略指纹/日志/停止命令 | **graph 278 s 就绪**（FP `7e70fc2c130b9191`，3090 端口外的 8100 完成 chat 冒烟）；**eager 121 s 就绪** |
| [`~/server/scripts/local-llm-stop.sh`](../../../server/scripts/local-llm-stop.sh) | 从状态文件/pid 优雅 SIGTERM，等 `/health` 下降 + 进程退出，超时升级 SIGKILL（含 worker 进程组），最后校验 4 卡显存释放 | 两次实测均 **4 卡显存回到 0%**、端口释放、状态文件清理 |

- 状态文件：`/mnt/game/host-serve/local-llm/instance.env`；日志：`/mnt/game/host-serve/logs/<tag>-<arm>-<ts>.log`。
- 关键防坑：脚本启动前 `unset VLLM_CACHE_ROOT TRITON_CACHE_DIR`，避免把 vLLM 缓存指到 root 拥有的
  `/mnt/game/vllm-cache` 而 `PermissionError`（本会话踩过一次，见 §7/§6）。
- **思考通道（默认开启）+ 局域网 + 工具调用**（2026-10-25）：`local-llm-start.sh` 默认
  `REASONING=1`（保留 `reasoning_effort=low`，思考进 `reasoning` 字段；`enable_thinking=false` 的请求级覆盖仍可用）、
  `HOST=0.0.0.0`（绑所有网卡，局域网 `http://192.168.0.102:8100/v1` 直连；ufw 已放行 8100/tcp）、
  工具调用沿用 launcher 的 `--enable-auto-tool-choice --tool-call-parser qwen3_xml`（`tool_calls` 已验证）。
  实现：两个 launcher 新增 `CHAT_TEMPLATE_KWARGS` env 覆盖，默认值逐字等于原硬编码值
  （`serve_host_dflash2.sh` 与 `s27_graph_launch.sh` 同步，仍保持"两处功能 diff"的派生关系）。
  局域网客户端需带 `Authorization: Bearer <key>`（key 在 `/mnt/game/host-serve/api-key`）。
  另：物理机重启后 `/mnt/game` 上的 16/32 GiB swap 不会自动启用，启动脚本现在会检测并 `swapon` 补齐，
  并已写入 `/etc/fstab`（`nofail`，备份 `/etc/fstab.bak-20261025`）。
- 旧的 `~/server/scripts/llm-start.sh` / `llm-stop.sh` 仍保留：它们指向另一套 rocm101 栈（`/mnt/game/port/gfx906-vllm-rocm101-perf`
  + `hosts/vllm-host.sh`），与本轮的 DTK 树配方是两套独立部署，按需选用。
