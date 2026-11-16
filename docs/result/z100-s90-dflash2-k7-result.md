# S90：DFlash2 k=7 重新上线专项 —— 结果（P0/P1 执行 + 两处栈级修复）

日期：2026-11-13。立项：[`docs/perf/PLAN-20261113-dflash2-k7-recommission.md`](../perf/PLAN-20261113-dflash2-k7-recommission.md)。
台账/产物：[`.s90/README.md`](../../.s90/README.md)，原始数据 `.s90/bench/`、`.s90/serve-*.log`。

## 一句话

**k=7 的 G1 与 G2 判定都没过，G3 也不可能过；但施工过程中挖出并修好了两个
栈级正确性缺陷（Triton hip_utils 头文件不匹配、gfx906 SFU 等待槽丢失），
这两个缺陷会让任何投机臂（含 MTP）静默产出乱码并且 accept 恒为 0。**
修好后现役 MTP k=2 恢复到 65.1–65.8 t/s（W1）/ 55.5–56.4 t/s（W2）、
per-position 0.82/0.67，与 2026-10-27 specbench 记录（51.6/40.6）同向且更高；
k=7 只有 29.6/23.6，k=4 只有 33.3/27.7 —— **DFlash2 仍应留档退役**。

## 0. 判据对照（§0 G1–G3）

| 目标 | 判据 | 实测 | 判定 |
|---|---|---|---|
| G1 加回 k=7 且可用 | ready、无 accept=0/NaN、3 次重启 accept 落同一吸引子 ±5% | ready 三次（D1b / K7-clean / K7-prof），per-position 曲线三次一致（p1 0.697/0.708/0.703）；**但 accept 并非恒非零前提**——见 §2 的头文件缺陷，修复前 k=7 的 accept 是 0 | **条件通过**（修复后才成立） |
| G2 接收率 →50% | p1≥0.7 且 p1..p4 均 ≥0.5 | p1 0.708 ✓、p2 0.544 ✓、p3 0.405 ✗、p4 0.299 ✗；draft 接收率 35.3% | **不通过** |
| G3 不亏于现役 | ≥ 51.6×0.95 ≈ 49 t/s | k=7 W1 29.6 / W2 23.6；k=4 W1 33.3 / W2 27.7 | **不通过**（差 1.5–2.1×） |
| G4 现役可回滚 | SPEC 一行切回、指纹一致、窗口外不中断 | 指纹 `8bf8b075a6469af0` 逐次一致；贪婪流与当日晨基线 96/96 逐位相同 | **通过** |

## 1. 同口径总表（7469-token probe，单流，prefix 热）

`e2e_ts.py` 口径：W1 = 96 tok 贪心 ×3，W2 = 512 tok temp0.7+rp1.05 ×2–3。
`acc/step` 与 `step_ms` 取自 `/metrics` 的 spec_decode 计数与 ITL 均值（同一窗口）。

| 臂 | W1 t/s | W2 t/s | TTFT s | acc/step | step ms | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MTP k=2（现役，修复后）** | **65.8** | **56.4** | 0.37 | **2.50** | **36–39** | 0.82 | 0.67 | — | — | — | — | — |
| DFlash2 k=7（生产 env） | 29.6 | 23.6 | 3.67 | 3.48 | 93.1 | 0.708 | 0.544 | 0.405 | 0.299 | 0.221 | 0.170 | 0.126 |
| DFlash2 k=7 + dynk(4096/32768) | 23.9 | 22.8 | 3.67 | 3.35 | 94.9 | 0.706 | 0.546 | 0.405 | 0.239 | 0.180 | 0.147 | 0.114 |
| DFlash2 k=4 | 33.3 | 27.7 | 3.83 | 2.97 | 77.7 | 0.704 | 0.533 | 0.420 | 0.310 | — | — | — |
| DFlash2 k=4 + dynk | 25.0 | 22.1 | 3.81 | 2.90 | 91.4 | 0.700 | 0.527 | 0.416 | 0.258 | — | — | — |
| DFlash2 k=4 + probabilistic | 27.4 | 26.5 | 3.79 | 2.90 | 86.1 | 0.703 | 0.530 | 0.414 | 0.255 | — | — | — |
| DFlash2 k=4 + int8 draft KV | 32.4 | 27.7 | 3.79 | 2.97 | 78.4 | 0.704 | 0.539 | 0.417 | 0.307 | — | — | — |

- 每个臂独立重启；`ROCBLAS/HIPBLAS_WORKSPACE_CONFIG=:4096:8` 全程保留。
- dynk 臂的 `step_ms` 升高是**空转口径**：probe prompt 在第 3 位后接收率塌到 0，
  dynk 把 verify 截到 1–3 行，省下的是 p4..p7 的 verify 时间，但 acc/step 也从
  3.48 掉到 3.35（截断位置之后不再有被接受的机会），净 e2e 是 −19%。
  **A3 在 k=7 上不成立**，与 S24 时代（k=7 但有 3 位以上真实接收）的结论不同。
- A2（`draft_sample_method: probabilistic`）只在采样口径有意义；k=4 采样臂
  27.4/26.5 vs 贪心臂 33.3/27.7，**没有正收益**，不采纳。
- B4（draft 用 int8 KV）：能启动且数值曲线不变，但它只把 **target 侧 KV 容量**
  从 710k 抬到 1.377M tokens（draft 侧 KV 组未被识别，见 §3.3），**step 时间不变**
  （77.7 → 78.4 ms），对 G3 无贡献，不采纳。

## 2. 两个栈级缺陷（本次最重要的产出）

### 2.1 Triton `hip_utils` 用错了 HIP 头 → 设备属性乱码

`triton/backends/amd/driver.py` 把 `driver.c` 编译成 `hip_utils` 时只带 vendored
头。祖传补丁 `tools/dcu/triton-gfx906-3.6.0.patch` 要求改成 DTK 头，因为 DTK 的
`hipDeviceProp_t` 是 792 B（`gcnArchName@396`），vendored 是 1472 B。triton 升到
3.8.0 后该补丁**没有重新落位**，且 launcher 的
`PYTHONPATH=…:/mnt/game/triton-gfx906/install` 指向的目录**在本机已不存在**，
于是走到 vendored 头路径：

- `get_device_properties(0)['arch']` 返回 `''`（解码 `gcnArchName` 直接
  `UnicodeDecodeError: 0x84`），进程内**任何 Triton kernel launch 都会失败**；
- 首次触发点是 `rope.py` 的 `_prepare_rope_positions_kernel`（dummy run），
  所以 worker 在 profiling 阶段就死；偶尔看得到“能起服务”的假象，是因为
  shim 命中了 10-18 留下的**旧 ABI 缓存**（另一个 key 下 22 KB 的 `.so`），
  那个 shim 属性正确但缺 `unload_module`/`build_signature_metadata`。

修复（已落到本机 venv，diff 存 `.s90/triton-patch/driver.patch`）：按
`ROCM_PATH`/`HIP_PATH` 找 DTK 头优先、vendored 兜底，并用
`-DHIP_LAUNCH_CONFIG=hipDrvLaunchConfig_t` 补上 DTK 的等价结构体名。
验证：`get_current_target() → GPUTarget(hip, gfx906, 64)`、
`multiprocessor_count=60`、冷缓存下 Triton 冒烟 kernel 正确。

### 2.2 gfx906 SFU 等待槽丢失 → 所有 transcendental 结果错

同一个祖传补丁的第二段（`compiler.py` 的 `make_hsaco` 在 `v_exp/v_log/v_sqrt/…`
前插 `s_nop 2`）同样没有落位。冷缓存 A/B（4096 元素 `tl.exp2()+tl.sqrt()`）：

| 编译器 | 不匹配元素 | 最大误差 |
|---|---:|---:|
| 裸 triton 3.8.0 | **1024 / 4096（25%）** | 1.05 |
| 加 `insert_sfu_waits` | **0 / 4096** | 4.8e-7 |

服务侧的后果正好对上今天所有反常现场：triton 融出的
`triton_poi_fused_*`（softmax/GDN/attention epilogue 里的 exp）在部分 SIMD16
lane 上读到**上一个 VALU 的旧值**，于是：

- 投机臂 draft 与 target 的分布对不上 ⇒ **accept 恒为 0**，`Mean acceptance
  length: 1.00`，但每步仍然 propose（`drafts` 正常增长）；
- 关闭投机后现象更清楚：同一服务 W1 从 65 t/s 掉到 26 t/s，输出变成
  ` ar\tset\tset\tset\tset…` 这类**可打印但无意义**的串；
- 修好后 MTP k=2 立刻回到 65.8/56.4 t/s、accept 0.82/0.67，贪婪流与当日晨
  baseline（修复前用旧缓存跑出来的那条）**96/96 逐位一致**。

修复同样已落到 venv，diff 存 `.s90/triton-patch/compiler.patch`。
**注意**：`triton_key()` 会把 `triton/backends/**` 的源码哈希进 kernel 缓存键，
所以补丁一改，旧 kernel 缓存自动失效（不会静默命中错误二进制）；反过来，
**这两段补丁一旦丢失，旧缓存里的 `.so` 会继续“看起来能跑”**，这正是本次
舒适当中最危险的地方。

### 2.3 由此产生的口径事故（必须记录）

`/mnt/game/vllm-cache-g2/torch_compile_cache` 在 07:00–09:20 之间被污染：
期间所有臂（含 `P0-mtp2-freshcache`、`E0b-k7`）都复用了错数值的图，
表现为 **10.7 t/s、accept 0.0**。该目录已整体移出为
`torch_compile_cache.poisoned-20261113`（44 GB，留作证据），生产已用新缓存
重建。**任何“当天的慢”都不能拿污染窗口内的数当基线**；本报告的基线一律取
09:37 之后的测量。

## 3. 分项判决

### 3.1 A 线（接收率）

| 动作 | 结果 |
|---|---|
| A1 换/重训 drafter | 未做（依赖训练资源）；本次给出了真实负载板的 per-position 曲线，p1 已经 0.70–0.71，但 p3 只有 0.40、p4 只有 0.30，**离 50% 仍差 1.3–1.7×** |
| A2 `probabilistic` | k=4 采样臂 W2 26.5 vs 贪心 27.7 ⇒ 无正收益，**不采纳** |
| A3 dynk | k=7 acc/step 3.483→3.346、W1 29.6→23.9，**负收益，不采纳**（k≤4 同理） |
| A4 k=4 阻塞 | **根因不是 `_selector_walk_kernel` 越界**：k=4 起不来是 `prefix_cache_retention_interval` 必须是 scheduler block size 的倍数，而块大小随 k 变（k=2 → 1616、k=4 → 1664、k=7 → 1712）。改正后 k=4 正常 ready，历史 VM fault **未复现**。A4 的“解锁 k≤4”目标达成，但 k=4 本身仍不翻正 |
| 接收率板的代表性 | 真实负载形状差异极大：7469 文档 0.708/0.544/0.405/0.299；代码板 0.688/0.488/0.363/0.263；散文板 0.718/0.539/0.385/0.256；**表格板 0.976/0.857/0.786/0.738**（复述类内容几乎全收）。⇒ 50% 在“可预测”内容上已经达到，在自由生成上不可达，A1 的训练目标应按内容类型分层 |

### 3.2 B 线（step 时间）

k=7 单步 93–94 ms 的构成（rank0 Kineto，56 个 decode 窗，`.s90/bench/trace-K7-prof.txt`）：

| 类别 | ms/step | 说明 |
|---|---:|---|
| `fp8_emu_gemv_stream_block`（target GEMV） | **45.2** | 14336 次/56 步 = **256 次/步**，176.7 µs/次；M=8 的 verify 全部落在 M≤8 的 GEMV 路线，没有吃 M=1 特化 |
| `triton_matmul` | 18.3 | 33.6 次/步，552 µs/次 |
| attention（`kernel_unified_attention_3d`） | 7.3 | 21 次/步，1.53 ms/次（M=8 verify） |
| rocBLAS（`Cijk_*`） | 5.9 | draft 塔的 5 层 bf16 |
| elementwise / AR / GDN | 5.2 / 4.9 / 1.1 | |
| draft bf16 GEMV | 2.1 | |
| **合计** | **93.9** | 与 ITL 93.1 ms 一致 |

对照 MTP k=2 的 36–39 ms/步：**target GEMV 一项就 20 ms vs 45.2 ms**，
且 k=7 的调用数是 256 次/步 vs MTP 的 ~112 次/步 —— 说明 verify 的 M=8 批次
既没有走 GEMV 特化，也没能合批，B1（M=5–8 verify 形状特化）是唯一能动
step 的杠杆；即使把 GEMV 压到 MTP 水平，93.9 → ~69 ms，acc 3.48 仍只有
~50 t/s 量级，**刚好在 G3 门槛附近，但远不足以覆盖 A 线的接收率缺口**。

### 3.3 B3/B4

- B3（`FULL_DECODE_ONLY`）：k=7 用 capture `[8,16,24,32]`（k+1=8 的倍数），
  与现役 `[2,3,6,12]` 同一策略；**注意** `adjust_cudagraph_sizes_for_spec_decode`
  会把非 8 倍数的 size 向上取整到 8 的倍数，直接用 `[2,3,6,12]` 会塌成 `[8]`。
- B4（draft int8 KV）：target KV 容量 710k → 1.377M tokens（+94%），
  step 不变；draft 侧 KV 组仍报
  `no KV cache group could be identified as the draft model's`，**draft 的 KV
  并没有被单独量化**，所以“带宽减半”这条预期不成立。

## 4. 结论与建议

1. **G3 不可达**：k=7 最好 29.6 t/s，k=4 最好 33.3 t/s，都低于无投机 36.9 的
   历史基线，更低于现役 MTP 的 65.8。按计划 §0“若 G3 不可达：k=7 留档退役，
   不做默认配方”，**本次判定 DFlash2 退役**；生产保持 MTP k=2。
2. **真正的收益不是投机，而是两个栈级修复**：现役 MTP k=2 从“看起来 26 t/s、
   accept 0、输出乱码”回到 65.8/56.4 t/s、accept 端正。这两个补丁必须
   **长期驻留**，并补进部署清单（否则下次升级 triton 会重演）。
3. **A1 若要立项**，训练目标应按内容分层，且只对“可预测内容”期望 50%；
   自由生成的 p3/p4 只有 0.40/0.30，靠调参不可能到 0.5。
4. **A4 的历史结论需更正**：k=4 的阻塞是 prefix-cache retention 的块大小
   约束，不是 `_selector_walk_kernel` 越界。这条要回写进
   `runs/specbench/REPORT.md` 的追加段。

## 5. 回滚与现状

- 现役：`z100-vllm-mtp2.service`，pid 见 `systemctl status`，
  `FP 8bf8b075a6469af0`，`:8000` 健康 200，enabled。
- 回滚到“修复前”**不建议**：那等于恢复乱码输出。若要回滚投机配置，
  只需把 drop-in 里的 `SPEC_*` 改回 `{"method":"mtp","num_speculative_tokens":2}`。
- Triton 补丁回滚：`.s90/triton-patch/{driver,compiler}.py.orig` 覆盖回去，
  并清空 `TRITON_CACHE_DIR` 下由补丁版本产生的条目。
- 遗留物：`/mnt/game/vllm-cache-g2/torch_compile_cache.poisoned-20261113`（44 GB，
  可删）；`/mnt/game/triton-gfx906/install` 在 launcher 的 `PYTHONPATH` 里但
  **不存在**，建议从 `launch_mtp2.sh`/`.s90/launch_dflash2.sh` 移除或重建。

## 6. 产物

- `.s90/README.md` —— 台账（每一步命令/口径/指纹）
- `.s90/bench/e2e-*.json`、`parity-*.json`、`accept-*.json`、`metrics-*-{before,after}.txt`
- `.s90/serve-*.log`、`.s90/window-*.log` —— 每个臂的启动与窗口日志
- `.s90/triton-patch/` —— 两个 Triton 补丁的 orig/new/diff
- `.s90/{serve_dflash2.sh,launch_dflash2.sh,window_multi.sh,accept_probe.py,one_stream.py}`
  —— 臂启动器、窗口编排、per-position 探针
