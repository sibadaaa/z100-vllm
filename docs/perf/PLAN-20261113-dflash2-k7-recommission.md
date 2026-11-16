# 立项:DFlash2 k=7 重新上线 + 接收率拉向 50%(2026-11-13)

> **立项拍板**(用户,2026-11-13):目标 = 在现役 27B 服务上**加回 DFlash2 k=7**,
> 并把接收率拉向 ~50%;允许改 kernel/代码;先出方案,不开维护窗口。
> **现役(不中断)**:pid 201811,:8000,MTP k=2 + FULL_DECODE_ONLY 图 + KV int8
> (s30c 栈,cmdline 实测 2026-11-13)。
> **一句话警钟**:specbench 实测 DFlash2 k=7 在新栈上是 **22.7 t/s vs 无投机 36.9 vs
> MTP k=2 51.6**([specbench/REPORT](../../runs/specbench/REPORT.md))。
> **50% 接收率翻不了盘**(§1 的账);本专项同时押注"接收率"与"step 时间"两条腿,
> 缺一条都是净亏损。

---

## 0. 目标与判据(先锁判据再施工)

| 目标 | 判据 | 口径 |
|---|---|---|
| G1 加回 k=7 且可用 | 服务 ready,parity 看门狗无 accept=0/NaN;3 次重启 accept 落同一吸引子 ±5% | 7469-token probe,greedy,W1×3 |
| G2 接收率 →50% | per-position 接收率 p1≥0.7 且 p1..p4 均 ≥0.5(等价 draft 接收率 ≥50%) | 同上;≥2 重启取分布(重启彩票,§5) |
| G3 不亏于现役 | DFlash2 臂 t/s ≥ 51.6(MTP k=2)× 0.95 ≈ **49 t/s** | 同口径 A/B,胜过 36.9 无投机 baseline |
| G4 现役可回滚 | SPEC 一行切回 MTP;指纹校验一致;窗口外服务不中断 | 窗口编排照抄 window_spec.sh |

**若 G3 不可达**:k=7 按结论留档退役,不做默认配方;可以只作为可选臂保留。

---

## 1. 已知账(为什么 50% 接收率本身不够)

### 1.1 specbench 三臂(2026-10-27,单流 batch=1,7469 probe,greedy)

| 臂 | t/s | step | 接收 |
|---|---:|---:|---|
| 无投机 | 36.9 | 27.1 ms | — |
| MTP k=2 | 51.6 | 46.5 ms | acc ~2.4 |
| DFlash2 k=7 | **22.7** | **101 ms** | acc ~2.0–2.5 |

### 1.2 per-position 接收率崩塌

`0.52 / 0.31 / 0.12 / 0.04 / 0.03 / 0 / 0` ⇒ 平均 draft 接收率 **14–21%**,
离 50% 差 2.5–3×。且接收率**由内容决定、非长度单调**
(1024/2048/4096/7469 截断 acc = 4.80/2.29/3.00/2.67,
[result/z100-acceptance-vs-context](../result/z100-acceptance-vs-context.md))。

### 1.3 50% 接收率的收支表(核心)

- DFlash2 step ≈ 101 ms(M=8 verify + 5 层 bf16 draft 塔,全部绕开 M=1 特化)。
- 若 per-position 接收 = 0.5 递减到 0,acc ≈ Σ0.5^i ≈ 2 tok/step
  ⇒ **≈20 t/s,净亏 vs 36.9**。
- **追平 MTP 51.6 需 acc ≥ 51.6×0.101 ≈ 5.2 tok/step** ⇒ per-position ≥ 0.85 量级。
- 两条腿必须同时走:**A 线拉接收率**(§3),**B 线压 step 时间**(§4)。
  粗略敏感性:acc=2 时需要 step ≤ 40 ms 才追平 MTP;acc=3 时 step ≤ 58 ms;
  acc=4 时 step ≤ 77 ms。

---

## 2. 现状与资产盘点

| 资产 | 状态 | 出处 |
|---|---|---|
| 现役服务 | MTP k=2,pid 201811,:8000,s30c 栈 | cmdline 实测 |
| DFlash2 权重 | `/mnt/game/models/Qwen3.8-27B-DFlash2-ms`(k=7 用)与 `-DFlash2` 两份 | ls 实测 |
| k=7 臂 launcher | specbench `launch_ctrim_spec.sh`(SPEC 透传,未动生产文件) | runs/specbench |
| SPEC env 覆盖 | `serve_host_dflash2.sh:180` 与 `s27_graph_launch.sh:204` 的 `_DEFAULT_SPEC` | 已就位(WP0.3) |
| k=4 VM fault | `_selector_walk_kernel` 启动即 fault,未修 | serve-D-dflash4.log |
| dynk 自适应 verify | `VLLM_S24_DYNK_*` 三键,已在生产 launcher | serve_host_dflash2.sh:100-102 |
| 重启彩票 pin | `ROCBLAS/HIPBLAS_WORKSPACE_CONFIG=:4096:8` | 已采纳(s29) |

**注意**:现役 s30c 栈的 KV 是 `int8_per_token_head` + `prefix-match-unit 16` +
`FULL_DECODE_ONLY [2,3,6,12]`,与 DFlash2 时代的 launcher(bf16 KV + piecewise
[1,2,4,8,16])**不是一套**。DFlash2 臂要显式声明自己的 KV/图配置(§6 臂定义),
不能假设继承。

---

## 3. A 线:接收率 →50%

按性价比排序;A1 是唯一根治,其余是增益或减损。

| # | 动作 | 预期 | 成本 | 判定 |
|---|---|---|---|---|
| **A1** | **换/重训 drafter**:现权重就是 14–21% 的水平;针对真实负载(agent 代码/长文档)蒸馏重训 draft 塔,或找社区同几何 (Qwen3 形状、causal=False dflash_config、候选头) 的更强权重 | 接收率上限由权重决定;14→50% 只能靠这 | 高(训练数据+算力) | 训练前先用 439-prompt parity 板测候选权重的 per-position 曲线 |
| A2 | `draft_sample_method: "probabilistic"`(采样负载);greedy 负载保持 greedy | 采样口径 +几个 pt | 一行 SPEC | W2 口径 A/B,≥2 重启 |
| A3 | dynk(`VLLM_S24_DYNK_LO/HI/EMA`):接收崩塌尾部不 verify,把 101 ms 里白烧的 p4..p7 时间省掉 | 不改接收率,但把"50% 只在前 3 位"变成可接受形态;S24 实测 ITL −23%/t/s +25-30% | 已就位,零改动 | step p50 + t/s |
| A4 | **排除 k=4 阻塞**:修 `_selector_walk_kernel` 越界(k 与 draft block_size=8 的缓冲形状),解锁 k≤4 臂 | k=4 外推 26–29 t/s,单独不翻正,但配合 B 线是低成本备选档 | 中(1 个索引 bug) | k=4 臂 ready + per-position 曲线 |

**A 线的现实预期**:A2+A3 在现权重大约把有效 acc 从 ~2 提到 ~2.3–2.6;
**50% 必须靠 A1**。A1 立项前置 = 先用探针量化"真实负载 prompt 分布"上的
per-position 曲线(specbench 只测过 1 篇文档),否则训练目标无从设。

---

## 4. B 线:step 时间 101 ms → ≤58–77 ms

101 ms 的构成(specbench + SOL 报告):M=8 verify 绕开 M=1 GEMV/GDN 特化 +
5 层 bf16 draft 塔前向 + draft 独立 KV 读写。

| # | 动作 | 预期 | 依据 |
|---|---|---|---|
| B1 | **M=5–8 verify 形状的 kernel 特化**:把 s33 的 M=1 GEMV/W8A16 特化覆盖到 verify 批形状(M=8 目标 + draft M=8) | step 大头;目标 M=8 verify ≤ 60 ms | specbench 建议②;M=3(MTP)46.5 ms 说明 M=5–8 有压缩空间 |
| B2 | draft 塔与 verify 的 kernel 复用/融合(s28d 已把 draft attention 接入 3D split-KV,−8.2 ms;继续把 draft 的 GEMV 走同一特化) | 再 −5–10 ms | s28d 实测 |
| B3 | 图捕获:DFlash 臂用 s30c 同款 `FULL_DECODE_ONLY`(而非旧 piecewise [1,2,4,8,16]),capture 桶按 M=8 verify 形状重选 | 与现役对齐,免双栈维护 | 现役 cmdline |
| B4 | KV dtype:DFlash2 臂试 `int8_per_token_head`(现役同款)替代 bf16 draft KV | 显存 +54% 容量先例(s30c);带宽减半 | s30c 实测 |

---

## 5. 方法学纪律(不遵守则结论作废)

1. **重启彩票**:accept 跨重启 ±6–10%;一切 accept 类对比 **N≥2 重启取分布**,
   `ROCBLAS/HIPBLAS_WORKSPACE_CONFIG=:4096:8` 保留(s28d/s29)。
2. **同口径**:7469-token 真实 probe + 439-prompt parity 板;W1 greedy 与 W2
   temp0.7+rp1.05 双口径;指纹 `dcu_policy_fingerprint.py` 必打。
3. **微测→数值→服务三关**(AGENTS 流程):每个 kernel 改动先真实 shape 微测,
   再 parity 数值对照,再同口径 e2e;缺一不采纳。
4. **窗口编排**:照抄 `runs/specbench/window_spec.sh`(停现役→起臂→bench→
   恢复现役→指纹校验);k=4 fault 教训 = 启动失败也要恢复现役(discard 残留)。

---

## 6. 施工排期(P0→P3)

| 阶段 | 内容 | 判据 |
|---|---|---|
| **P0**(窗口 0.5h) | 起 DFlash2 k=7 臂(specbench launcher + 现役 env 快照对齐),抓 per-position 基线 + 439-prompt 接收分布;**同时抓现役 MTP 臂同口径基线** | G1 ready;得到 50% 差距的真实分布(不是单文档) |
| **P1**(零窗口) | A2/A3 落地(旋钮级);A4 定位 k=4 索引越界(读 `_selector_walk_kernel` launch 参数与缓冲 shape);B3/B4 臂配置 | A2/A3 有 A/B 数;k=4 根因定位报告 |
| **P2**(窗口 2–3 次) | B1 M=5–8 verify 特化(微测→parity→e2e 三关);B2 draft 塔复用 | step ≤77 ms(acc=4 时)或 ≤58 ms(acc=3 时) |
| **P3** | A1 训练立项(依赖 P0 的 prompt 分布数据);终判 G1–G4 | G3 达标→晋级讨论;不达标→留档退役,保持 MTP 现役 |

**依赖关系**:B1 是 G3 的必要条件(acc≤2.6 时无论接收多高都追不上 MTP);
A1 是"接收率 50%"字面目标的唯一路径。P0 的数据决定 A1 是否值得立项。

---

## 7. 回滚

- 现役回滚 = `SPEC` 换回 `{"method":"mtp","num_speculative_tokens":2}`(一行,
  WP0.3 机制),或直接用 s30c launcher 重启;DFlash2 臂全部走 specbench 副本,
  生产 `serve_host_dflash2.sh`/`s27_graph_launch.sh` 不改。
- kernel 改动(B1/B2)各自带 kill switch env,默认关闭,采纳才晋级。
- 反证条件:P0 发现真实负载(非单文档)per-position p1 已 <0.4 ⇒ A1 训练也难到
  50%,整专项降级为"低成本旋钮臂(A2/A3/A4)+ B 线单独立账"。
