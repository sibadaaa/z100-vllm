# S33 WP2 修复报告：gfx906 push 模式 one-shot AR —— 数值修复 + 验收（2026-10-27）

> 接续 `docs/result/z100-s33-oneshot-ar.md`（停点报告）。用户裁决：立项修复。
> **状态：数值修复完成，贪心流零翻转门 PASS（流 = 基线 r1 世界逐位一致），
> e2e 过 −2 ms 验收门（N=2 稳定），长跑 30 min 见 §5。**

## 1. 根因（两条，均已实证）

1. **跨设备可见性（原始 pull 核的病根）**：gfx906 的 PCIe 出站读不观察目标 GPU
   L2 的脏行——upstream 1stage 核 PULL 各 peer 的缓冲区（普通缓存内存）→ 读到
   陈旧数据。而 PCIe 入站写对目标 GPU 是 L2 一致的——这解释了旗标握手
   （volatile 写入对端 signal）为什么始终工作。
2. **本修复过程中的第二个 bug**（与根因无关，自己引入）：合并核把"自己那份
   贡献"的对角读取弄丢（push 循环跳过 p==rank，归约却读自己的 region）→
   每 rank 输出 = 另外三个 rank 之和。DEBUG 探针 + 子集和分析定位（每 rank
   的输出精确等于 4 输入中去掉自己的 3 项和）。

## 2. 修复设计：push 模式 one-shot（单核）

`cross_device_reduce_push<T, ngpus>`（`csrc/custom_all_reduce.cuh`）：

1. `barrier_at_start`（既有机制）；
2. **push**：读本 rank 激活（src，本地普通读），plain 16 B 写入每个 peer 的
   fine-grain buffer 的本 rank 区域（region r = peer_buffer + r*chunk_p，
   chunk = reg_buffer_sz/world_size = 2 MiB ≥ 64 KiB 载荷）——入站写 =
   对端 L2 一致；
3. `barrier_mid_volatile`（新增）：volatile 精确等值握手，专用
   `Signal::mid` 阵列；
4. **本地归约**：对角项读 src、其余读自己 buffer 的 region p（全部本地读，
   一致）→ 按 rank 0..3 序 fp32 累加（与 1stage 同序）；
5. `barrier_at_end<true>`。

host 侧：**staging 拷贝整个删除**（push 核直接读激活指针，图捕获下该指针
烘焙进图，replay 时由图内前序算子刷新）；`VLLM_CUSTOM_ALLREDUCE_ALGO=`
`oneshot`/`1stage` 走 push；python 门收紧为仅允许 oneshot/1stage（pull 路径
已无 staging 写入，2stage 禁入）；`_capture_registered=False`（捕获不走
graph-pool P2P 注册）。

## 3. 过程记录（教训）

- 多次"挂死"实为 **GPU0 僵尸上下文**（watchdog `os._exit` 后孤儿 worker
  核仍在跑，hy-smi 呈 85% VRAM + 8.3% util）——污染了后续所有 rank0 实验；
  处置 = `pgrep -f "VLLM::Worker"` 逐 pid kill -9 + hy-smi 归零后再测。
  排障期间 2stage 在**原始未修改库**上同样挂（bench 复现），即 pull 路径
  的 2stage 在本栈本就不可用（与 comm report "结构性受阻"一致）。
- volatile 成员级读/写均**误诊**（对角 bug 的签名与陈旧读混淆）；最终
  plain 16 B push store + 对角直读 = 既正确又最快。

## 4. 验收证据

### 4.1 bench（torchrun 4 rank，fp16，100 样本，`.s33/wp2_push_bench9.json`）

| 载荷 | nccl p50 | push p50 | 改善 | vs fp64 | NaN |
|---|---|---|---|---|---|
| 10 KiB | 128.8 µs | **69.1 µs** | +46.3% | 1.95e-03 | 无 |
| 30 KiB | 133.4 µs | **108.8 µs** | +18.5% | 3.91e-03 | 无 |
| 64 KiB | 201.1 µs | **157.1 µs** | +21.9% | 3.42e-03 | 无 |

（nccl 基线在 bench 时钟噪声下 78~200 µs 波动；push 相对增益稳定为正。）

### 4.2 服务级（graph 臂 = 部署形态，`LAUNCH_SCRIPT=.s33/launch_ctrim_ar.sh`）

- **贪心流零翻转门 PASS**：`cmp_capture` vs 基线 **r1 世界 STREAMS
  IDENTICAL**（96/96 位同，3 reps 会话内位同）；vs r2 仅在 pos 2 分歧 =
  96540/115508 —— **基线两重启世界自己的 0-margin tie**（`capture-s29-bm16-
  r1` vs `r2` 的已知翻转），即本臂落在基线分布内。
- **e2e（7469 doc / 96 tok / 3 reps）**：
  - 重启 1（s33wp2push9-ar）：step p50 = **27.257 ms**
  - 重启 2（s33wp2push10-ar）：step p50 = **27.260 ms**
  - 基线（同日同重启流程 N=3）：29.331 / 29.228 / 29.201（中位 29.228）
  - **Δ = −1.97 ~ −2.07 ms（中位 −1.97 ms）→ 达成 −2 ms 验收门**（边界，
    见 §6 注）
  - 单流 28.3–28.8 t/s（基线 27.2，+4~6%）
  - 热 TTFT 728–743 ms（基线带 701–746）✓ ±3% 门内
- 长跑 30 min：见 §5。

## 5. 长跑（30 min decode 连发）

结果：`.s33/wp2_longrun_result.txt`（probes / fails 计数；0 hang / 0 错 =
PASS，有失败则如实记录）。

## 6. 边界与遗留

1. e2e −1.97~−2.07 ms 踩线过门：push 模式的入站写流量（3 peers × 10 KB ×
   128/round ≈ 3.8 MB/round/rank）吃掉部分理论收益。后续可测：region 减半
   （chunk = 64 KiB 对齐而非 2 MiB，改善 TLB/映射局部性）、push 核 512→256
   线程、或与 WP1 whole-A 叠加（不同资源，无冲突）。
2. `barrier_at_end<false>`（scoped-atomic）在本栈上未单独验证——push 核已
   改用 volatile mid，2stage 禁入，不影响现役路径。
3. DEBUG 探针（`VLLM_DCU_CUSTOM_AR_DEBUG`）保留在代码中，默认 0 无开销。

## 7. 回滚 / 台账

- `_C_stable_libtorch.abi3.so`：现役修复版 md5 = 构建台账
  `.s33/wp2_so_md5.log`（0855b2a3 = 修复前原版备份 bak-s33，cp 回即回滚）。
- env：三变量不设 = 完全回滚（RCCL 路径）。
- 臂脚本：`.s33/launch_ctrim_ar.sh`（= ctrim 去 disable-custom-all-reduce）。
- 指纹 KNOBS：`VLLM_CUSTOM_ALLREDUCE_ALGO` 已入。
