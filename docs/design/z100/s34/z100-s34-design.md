# S34 设计图：生产 MTP 臂上的 AR/带宽/长上下文收尾（2026-10-27）

> 前置事实纠偏：16:05–16:54 的会话内评估曾判定"05:44 生产 MTP 臂未启用 push AR"。
> **该判断错误**：worker 的 `/proc/<pid>/environ` 在 `VLLM_WORKER_MULTIPROC_METHOD=spawn`
> 下被运行时重写，不反映父进程启动 env。主进程 environ 实证三 env 齐全
> （`VLLM_DCU_CUSTOM_AR=1` / `VLLM_CUSTOM_ALLREDUCE_ALGO=oneshot` /
> `VLLM_DCU_CUSTOM_AR_ALLOC=finegrained`），二进制 md5 `7dcce4f2…` = WP2 修复版。
> **结论：push AR 已在生产 MTP 臂现役。** 本图以此为准。

## 0. 现状账本（2026-10-27 16:50，全部实测）

| 口径 | 配置 | step p50 | tok/step | ms/tok | 备注 |
|---|---|---|---|---|---|
| S30c 基线（nospec，AR off） | ctrim 臂 | 29.48 ms | 1.0 | 29.48 | `z100-s30c-final.md` |
| S33-WP2 push AR（nospec，AR on） | ctrim 臂 | 26.70 ms | 1.0 | 26.70 | `z100-s33-oneshot-ar-fix.md`，长跑 0 fail |
| **生产 MTP 臂（AR on）7.5k 探针** | 05:44 臂，MTP k=2 | **49.44 ms** | **2.40** | **20.60** | `.s34/s20_s34-wp1-base-r1_*.json`，accept_len 1.487 |
| 生产 MTP 臂，生产流量统计 | 48 请求，avg prompt 60.3k | ~63 ms（推算） | 2.06 | 30.5 | `/metrics` TPOT sum/count → 32.8 t/s |
| 短上下文（5 tok prompt） | 同臂 | ~35 ms | ~2.06 | ~17 | 会话内实测 ~57 t/s |

要点：

1. **MTP 切换相对 S33 nospec 臂是净收益**：20.6 vs 26.7 ms/tok（−23%）。
2. **长上下文回落依旧**：fwd t/s 从 7.5k 的 48.6 掉到生产均值 32.8（60k 均值），
   128k 更差——WP6 要把这个曲线测出来并定位增量项。
3. **异常待查**：MTP 臂 warm TTFT ≈ 2.4 s（同 7469 doc，S33 臂 ≈ 0.72 s），
   三 rep 一致 → 非冷启动；WP6 一并定位。
4. 已知结构项：batch 4 时 MTP verify 载荷 12 tok × 5120 × 2 B = 120 KiB > 64 KiB
   size 门 → verify 步回落 RCCL（WP2 的靶子）。

## 1. 基线快照与回滚（.s34/）

- `base_cmdline.txt` / `base_environ.txt`：05:44 臂的逐字快照（32 个 VLLM_ knob）。
- `so_md5_baseline.txt`：`_C_stable=7dcce4f2…`（push AR 修复版）、`_C=34d8b074…`
  （GDN m1）、`_rocm_C=86470c5d…`（含默认关的 whole-A 变体）。
- 臂重启器：`.s34/launch_arm.py`（快照复现 + `KEY=VALUE`/`-KEY` 增删 env）+
  `.s34/restart_arm.sh`（孤儿 worker 处置 → hy-smi 归零检查 → 起服 → health →
  env/policy 验证；有在飞请求则拒止）。
- 回滚 = `bash .s34/restart_arm.sh prod-restore`（无 override = 逐字恢复 05:44 臂）。

## 2. 工作包

### WP1：MTP 臂上的 AR A/B（本图把"带三 env 重启"改为"AR-off 腿补测"）

- 动机：S33 的 AR ±2.6 ms 是在 nospec 臂上验收的；MTP 臂上尚未量化。
- 腿：A = 现役（AR on，已测）→ B = 三 env 移除（RCCL）→ 恢复。
- 判据：7.5k 探针 step p50 差 ≥1.5 ms 视为与 S33 一致；60k 生产口径同向。
- 流门说明：MTP 臂存在 accept-lottery（s28d），跨臂流位同**不作为门**，
  以 in-session 3 rep 位同 + 0 错误 + accept_len 稳定为门。
- 产物：`.s34/s20_s34-wp1-{base,aroff}-*.json`、`docs/result/z100-s34-wp1-ar-ab.md`。

### WP2：push AR size 门 64→256 KiB（env 化）

- 动机：c=4 verify 载荷 120 KiB 被 64 KiB 门打回 RCCL，AR 增益在并发下丢失；
  bench 已证 64 KiB 处 push 仍 +22%。
- 改法：新增 env `VLLM_DCU_CUSTOM_AR_MAX_KIB`（默认 64 = 行为零变更），
  `custom_all_reduce.py` 的 `inp_size > 64*1024` 判据改读该 knob。纯 Python
  改动，**不需要 csrc 重编译**（kernel 侧 region chunk = 2 MiB ≥ 256 KiB）。
- 门：car_bench 128/256 KiB 载荷 vs nccl vs fp64（NaN/误差表）→ 服务级
  c=4 探针（120 KiB verify 实走 push）→ e2e 门 c=4 step p50 −0.5 ms →
  30 min 长跑（复用 S33 长跑法）。
- 产物：`.s34/wp2_car_bench_*.json`、`docs/result/z100-s34-wp2-size-gate.md`。

### WP3：GEMV stack（`VLLM_DCU_FP8_GEMV_STREAM_ROWS=4` + `VLLM_DCU_FP8_GEMV_M1=1`）

- s30c-WP2-N 二级已在 nospec 臂验证"数值位同、流=基线世界、e2e −1.0~−1.4 ms"，
  当时因"边际"未采纳；现在与 push AR 叠加（不同资源，fix 报告 §6.1 亦建议）。
- 门：MTP 臂 7.5k step p50 −1.0 ms 级；流应与 WP1-A 臂位同（GEMV 位同 +
  draft 塔 bf16 路径不受影响 → accept 序列不变）。
- 产物：`.s34/s20_s34-wp3-*.json`、capture 对照、`docs/result/z100-s34-wp3-gemv-stack.md`。

### WP4：rms_norm/quant 融合进 push AR 核（设计 + 施工图，实施视进度裁决）

- WP0 分账的真实核间空隙 ≈3.8 ms/round（~1000 处 × 5 µs）是直接靶子；
  push 核归约完成即得完整行 → 顺手做 add-rmsnorm epilogue。
- 本轮交付：`docs/design/z100/s34/z100-s34-wp4-ar-fusion.md`（内核草图、集成点、
  位同风险——归约序不变 + norm 的 fp32 中间量、门、回滚）；实施需
  csrc 重编译（build-sync.sh），按判据单独过门。

### WP5：split-K GEMV 立项（设计 + 施工图）

- 触发条件已成立（WP1 whole-A 停点：qkv 402.5 / o 361.3 < 450 门）。
- 本轮交付：`docs/design/z100/s34/z100-s34-wp5-splitk-gemv.md`（K 维分块 + 分片序
  的位同策略、零翻转硬门方案、预期 −1 ms 级）；实现与验收留裁决。

### WP6：长上下文口径（32k / 64k / 128k × c{1,4}，在最终胜出臂上）

- 动机：S30–S33 全部门在 7469-doc（~7.5k）口径过门；生产是 60k 均值/128k 长尾
  （用户最初的问题）。KV-fp8、lm_head-fp8 的价值也只在长 ctx 口径才放大。
- 协议：`.s34/longctx_corpus.md`（2.04 M token 真实语料，md5 入台账），
  `--truncate-tokens {32768,65536,131072} --max-tokens 256 --repeat 3`，
  c=1 用 s20 探针；c=4 用 `.s34/probe_c4.py`（4 并发流 + 引擎步周期交叉核对）。
- 产出：步周期随 ctx 的分解曲线（attention/块表 host/AR 载荷门命中）、
  TTFT 2.4 s 异常定位、KV-fp8 与 lm_head-fp8 在长 ctx 的预期收益重估表。
- 产物：`.s34/s20_s34-wp6-*.json`、`docs/result/z100-s34-wp6-longctx.md`。

## 3. 约束外路线（只立项，后续跟进，全部需 eval + 用户裁决）

沿用 `z100-s32-unconstrained-plan.md` 的账本，不再重复推导：
w4a16 MLP（−4~6.3 ms，最大单项）→ lm_head fp8（−0.45 ms，改 top-1 分布）→
KV fp8 LUT（长 ctx 放大，WP6 后重估）→ SSM state fp16（−0.15~0.2 ms）→
drafter 改进重开 spec（S32-C1）。地板 12–14 ms/tok。

## 4. 风险表

| 风险 | 缓解 |
|---|---|
| 重启打断生产流量 | restart_arm.sh 在飞请求拒止；每次重启 ~5 min 窗口 |
| GPU0 僵尸上下文污染（S33 教训） | 逐 pid kill + hy-smi 归零后才起服 |
| MTP accept-lottery 干扰流门 | 跨臂不设流位同门（WP1）；WP3 除外（位同路径） |
| size 门放大暴露 push 核长载荷缺陷 | 先微测 128/256 KiB vs fp64，NaN 即停 |
| 长跑稳定性 | WP2 采纳前复跑 30 min 长跑 |
