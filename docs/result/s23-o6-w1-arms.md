# S23 O6 — W1 端到端 A/B 结果（c=1）

> 本文件只放 **O6 的实测原始数据与判读**（perf-verifier）。工具与协议见
> [`s23-o6-tooling.md`](s23-o6-tooling.md)；最终汇总由 lead 写进
> `docs/result/z100-fp8-emu-optimization-result.md`。
> 纪律：**只比同一个窗口内的臂**。plan §1 的 193.8 ms / 19.2 t/s 是历史窗口，
> 与本文任何数字都不可直接比较（今天同机默认配置已比它快 25–30%）。

## 0. 一句话

截至目前 **O1（custom_op 绕过）在 c=1 上没有兑现**：同一窗口内 step p50 138.00 ms（control）
vs 139.24 ms（o1），输出 26.95 vs 26.90 t/s，accept 与 tok/step 完全不变（7.00 / 7.53），
parity 8/8 逐 token 一致。差异在 ±2% 噪声内，且方向对 O1 不利。
这与 plan §2 的警告一致：c=1 步进由 idle/派发主导，host 侧省下的 ~10 ms/步 被 idle 吸收。

## 1. 窗口 w1a（作废，仅为门禁与流程证据）

- 目的：gate 1（pytest）一次性占用「引擎停机 + 4 卡空闲」的窗口；同时首次跑臂。
- **gate 1 通过**：服务停止、`VRAM sum 0%` 后
  `pytest tests/kernels/quantization/test_gfx906_fp8_emu.py -q`
  跑两遍——三个开关全 0：`123 passed, 15 warnings in 30.44s`；
  `VLLM_DCU_FP8_EMU_FASTPATH=1`：`123 passed, 15 warnings in 30.34s`。
  无任何 harness 侧失败。
  日志：`.s23/w1/O6/w1a/unit-gate/{pytest-switches-off.log,pytest-emu-fastpath-on.log,summary.txt}`。
- **`.so` 锁不变量**：拿锁时 live `.so` 是 `3dac38f6…`（kernel-dev 的重编产物），不是基线；
  按 lead 的新不变量先恢复基线 `f205be8f43d3ffb1cfb3c6f093bfab4b` 再开臂，
  记录在 `.s23/w1/O6/w1a/so-at-window-start.txt`。
- **两个臂作废**：首版身份 canary 从 `/proc/<engine pid>/environ` 读开关值，而该系统托管进程的
  `environ` 在本沙箱是 `Permission denied`（实测），于是两个臂都被误判 `env-mismatch`，**无任何
  probe/parity 数据产出**。这不是 O1 的问题，也不回收为 control。
- 修复：臂开关改为 `systemd-run --setenv=K=V` 注入（落在单元自身的 `Environment=` 属性里，
  可被 `systemctl --user show` 读回），`descriptor.py` 在 `/proc` 不可读时以
  `env_source=unit-props` 记录生效值；restore 增加硬判据：默认单元 `Environment` 必须恰为
  `HOST=0.0.0.0`。

## 2. 窗口 w1a2：control vs o1（有效）

窗口信息（两次臂之间无任何变化）：

| 项 | 值 |
|---|---|
| git rev | `ab2a89b1af1b79e529ecc842620eabef4cf98979`（短 `ab2a89b1a`） |
| `.so` md5 | `f205be8f43d3ffb1cfb3c6f093bfab4b`（两臂相同） |
| python 树指纹 | `4adf3a0db0fdaff6`（两臂相同，只含 `vllm/`、`csrc/` 下 `.py`） |
| 每臂 probe | 4 次请求（512 prompt / 128 输出），引擎 `iteration_tokens_total` 步进口径 |
| 窗口时长 | 10:36:59–10:48:25 UTC（含 3 次 unit 重建） |

| 臂 | 开关 EMU/QNT/FUSE | step p50 | warm p50 | step 均值 | out t/s | out t/s(warm) | accept | tok/step | TTFT |
|---|---|---|---|---|---|---|---|---|---|
| control | 0/0/0 | **138.00 ms** | 137.83 ms | 138.26 ms | **26.95** | 29.08 | 7.00 | 7.53 | 2506 ms |
| o1 | 1/0/0 | **139.24 ms** | 140.35 ms | 138.52 ms | **26.90** | 28.77 | 7.00 | 7.53 | 2493 ms |

逐 run 步进（ms，按时间顺序，第 1 个 run 含冷 TTFT）：

```
control: 139.6  138.2  137.8  137.4
o1     : 135.2  140.3  140.4  138.1
```

| 判据（plan §3 O1 门禁 4/5） | 结果 |
|---|---|
| step p50 改善 ≥ +2% | ❌ 实测 +1.8%（更慢） |
| 输出 t/s ≥ +2% | ❌ 实测 −0.2% |
| accept 不变 | ✅ 7.00 / 7.00 |
| parity greedy 8 prompt 逐 token | ✅ 8/8 完全一致（8/54/412/465/1678/1643/3254/16 prompt tokens，各 64 生成 token） |
| 每臂真实 chat smoke + usage | ✅ 两臂 `completion_tokens=22`，完成串 ` Paris.\nThe capital of Germany is Berlin.\nThe` |

**判读**：c=1 上 O1 属于「已实现、未兑现」（plan §6.2 的第二种情形，不是失败）。理由链：
S23 trace 已证 `corr(period, idle)=+0.739`、`corr(period, gpu_ms)=+0.071`，且每步 256 次
custom_op 的 host 时间与 GPU 工作重叠；等 O1 在 c=4/c=16（GPU busy 主导）上再判一次。
w1a2 里 o1 排在 control 之后，+1.8% 也可能是臂序/窗口内漂移，w1c（o1 在前）用来分离。

**为什么"没变"是可判读的测量、而不是"没测出来"**：本判读只在**设备侧工作量逐项相同**这一前提下成立，
而该前提不是假设——vllm-py 的 O1 门禁 3 在同一构建上做了 kernel census，`EMU_FASTPATH=0/1` 两侧
kernel 名称、次数、平均 device time 逐项一致（`±2%` 内，`extra_kernels_fastpath` 为空）。
也就是说 O1 只可能改变 host 时间（实测 45–53 µs → 5.9 µs/次，256 次/步 ≈ −10 ms/步），
不可能改变 GPU 时间。于是「period 不变 + host 时间确实减少」⇒ 结论只能是 host 时间被 idle 吞掉，
而不是某个未知的 GPU 加速把收益抵消了。

| 臂 | 判决 | 什么会证伪这个判决 |
|---|---|---|
| control | 窗口内参考（138.00 ms / 26.95 t/s） | 若同窗口重测 step p50 漂移 >2%，则该窗口的所有差值都不可用（这正是本窗不跨窗口比较的原因） |
| o1（c=1） | 已实现、未兑现 | ① 若 o1 臂的 kernel census 与 control 不一致（kernel 名字/次数变化）→ 判决作废，因为那已不是"纯 host 改动"；② 若 o1 的 host 直测没有下降（仍 ~45–53 µs/次）→ 说明开关没生效，判决作废；③ 若 w1c 反序窗口（o1 在前）给出 step p50 改善 ≥2% → 本窗的 +1.8% 是臂序/漂移伪影，判决改为"c=1 收益在噪声内、不可判定" |

## 2b. 窗口 w1c：control vs o1，**反序** + c=4/c=16（决定性数据）

目的：① 用臂序反转检验 w1a2 里 o1 的 +1.8% 是不是臂序/窗口漂移；② 拿 c=4/c=16 —— lead 重定优先级后，
O1/O2 这类 host 侧削减的判据在 c≥4（c=1 已证 host 被 idle 吞掉），而不是 c=1。

窗口：2026-10-19 11:14:30–11:40:26 UTC；git `ab2a89b1a`；`.so` `f205be8f`（两臂同）；
树指纹 `75f4d76dd0c17304`（两臂同）。臂序 **o1 先、control 后**（parity 基线由 o1 capture，control 对它逐 token 校验）。

| 臂 | step p50 | warm p50 | step 均值 | 输出 t/s | warm 输出 t/s | accept | tok/step | TTFT |
|---|---|---|---|---|---|---|---|---|
| o1 | 139.05 ms | 137.38 ms | 140.17 ms | 26.76 | 29.10 | 7.00 | 7.53 | 2504 ms |
| control | 139.49 ms | 137.81 ms | 140.27 ms | 26.78 | 29.06 | 7.00 | 7.53 | 2509 ms |

c=1：o1 相对 control **−0.3%（warm p50）**；w1a2（control 先）是 **+1.8%**。两次窗口、两种臂序、符号相反、
幅度都在 ±2% 内 ⇒ **w1a2 的 +1.8% 是臂序/窗口内漂移，不是 O1 变慢**。O1 在 c=1 上是平的，这一点现在有两次独立测量背书。

| 并发 | 臂 | fwd t/s（warm 中位） | fwd t/s（全轮中位） | gen t/s | step ms | accept | 每轮 fwd t/s（第 1 轮冷） |
|---|---|---|---|---|---|---|---|
| 4 | o1 | 901.9 | 901.1 | 37.4 | 291.2 | 8.00 | 505.0 901.9 900.3 911.3 |
| 4 | control | 847.7 | 845.6 | 35.1 | 294.3 | 8.00 | 460.2 1008.5 843.5 847.7 |
| 16 | o1 | **1039.4** | 1037.3 | 40.5 | 1373.4 | 8.00 | 592.5 1035.2 1039.4 1055.4 |
| 16 | control | **1004.7** | 1001.7 | 39.2 | 1421.0 | 8.00 | 587.4 998.8 1004.7 1008.7 |

判读（保守版）：
- **c=16 是本窗口唯一干净的信号**：o1 的三个热轮 1035.2 / 1039.4 / 1055.4 全部高于 control 的 998.8 / 1004.7 / 1008.7，
  **区间不重叠**，中位 **+3.45%**（best +4.6%）。这是 O1 在整个 S23 里第一次「兑现」——它落在
  「host 是临界路径」的并发区间，而不是 c=1。
- **c=4 判为不可判定**，不是 +6.6%：中位差 +6.4%，但 control 有一个 1008.5 的高离群轮，热轮区间
  （control 843.5–1008.5 vs o1 900.3–911.3）**重叠**。按 w1c 的非重叠判据，只有 c=16 成立。
- c=1 的 4 次 probe 里第 1 次是冷 TTFT（3.8 s vs 2.07 s 热），把 out t/s 均值拉低；故表里同时给 warm 列。

门禁（同一停机槽）：fp8-emu 套件在**四个开关状态**下全绿（每次 `123 passed`，~29.7 s）：
`.s23/w1/O6/w1c/unit-gate/{pytest-switches-off,pytest-emu-fastpath-on,pytest-quant-fastpath-on,pytest-all-switches-on}.log`。

复原：default unit 重建（`Environment` 恰为 `HOST=0.0.0.0`）、`.so` `f205be8f`、`/v1/models` 200、
restore smoke 与 control 臂 `smoke-identical`。

**什么会证伪 c=16 的 +3.45%**：若下一窗口把 `--sweep-reps` 提到 ≥6 后 o1 与 control 的热轮区间出现重叠，
则本判决降为「不可判定」；若 control 热轮上界超过 o1 热轮下界 1035.2，同样降级。仅有中位数差、无区间分离不算证据。

## 3. 复原证据（窗口结束态）

```
unit Environment = HOST=0.0.0.0            （无任何开关残留）
.so md5          = f205be8f43d3ffb1cfb3c6f093bfab4b
/v1/models       = 200
restore smoke    = smoke-identical（与 control 臂的固定 prompt 完成流逐 token 相同）
```

## 4. 原始文件索引

| 内容 | 路径 |
|---|---|
| 窗口汇总表（机器生成） | `.s23/w1/O6/w1a2/window-summary.md` / `.json` |
| control 臂身份 | `.s23/w1/O6/w1a2/control/descriptor.json` |
| o1 臂身份 | `.s23/w1/O6/w1a2/o1/descriptor.json` |
| probe 原始 JSON + stdout | `.s23/w1/O6/w1a2/{control,o1}/s20_w1a2_*.json`、`probe.log` |
| parity 基线 / 校验 | `.s23/w1/O6/w1a2/control/parity/control.json`、`o1/parity/o1-check.json` |
| smoke（含 usage） | `.s23/w1/O6/w1a2/{control,o1,restore}/smoke.json` |
| 单元属性前/后快照 + diff | `.s23/w1/O6/w1a2/*/unit-props-{before,after}.txt`、`unit-props-diff.txt` |
| gate 1（pytest） | `.s23/w1/O6/w1a/unit-gate/` |
