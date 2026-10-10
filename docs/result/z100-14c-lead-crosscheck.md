# S14c 原始内存 trace 的 Lead 独立复核(2026-10-10T00:28Z)

> 本文由 **Lead** 写,不是子 agent 的步骤归档。目的:在 S14c 收尾前,用**它自己落盘的原始 trace**独立重算一遍关键量,确认 §3.7 那张表不是抄错/口径混淆。S14c 的正式归档仍是 [`z100-14c-tp4-graph-memory-result.md`](z100-14c-tp4-graph-memory-result.md)(**单 writer = 子 agent**)。

## 复核方法
* 数据:`/mnt/game/s14c/hostlogs/C{1..6}-memtrace.log`(5 s 采样;`S` 行 = 系统级,`P` 行 = 逐进程)。
* 独立重算:每个时间戳取 **所有 worker(`role` 以 `W` 开头)的 `rssanon` 之和**;`minAvail` 取 `S` 行 `mem_avail_kb` 最小值;`maxSwapUsed = swap_total - swap_free`;`oomk` 取 `S` 行最大值。
* 与归档口径的**已知差异**:归档的 `ΣRssAnon` **把 EngineCore/APIServer(E)也加进去**,且"稳态"用**相位窗口**(P1..P5)定义;我用的是"最后 20 个采样的中位数"。所以**绝对峰值会小一些、稳态可能被尾部空转拉低**,只有 **`peak−steady` 的量级结论**与 **`minAvail`/`maxSwapUsed`** 可直接对拍。

## 重算结果(单位 MiB,除标注外)
| arm | 配置 | 我算的 peak ΣworkerAnon | peak−steady | minAvail | maxSwapUsed | oomk |
|---|---|---|---|---|---|---|
| C1 | TP=4 graph + 16 GiB swap(现场编译) | 25888 | (被尾部空转拉低,不可比) | **0.86 GiB** @23:48:52 | **8.56 GiB** | 3(未增) |
| C2 | TP=4 **eager**,仅 `/swap.img` 4 GiB | 18832 | **0** | **4.69 GiB** | 0.45 GiB | 3(未增) |
| C3 | TP=2 graph,仅 `/swap.img` 4 GiB | 14210(2 worker) | **0** | **9.88 GiB** | 0.98 GiB | 3(未增) |
| C4 | C1 + `MALLOC_ARENA_MAX=1`,仅 4 GiB | 26631 | 924 | 0.96 GiB | **4.00 GiB(吃光)** | **3→4** |
| C5 | C1 但 **AOT 磁盘重载** | 25044 | 932 | **0.60 GiB(更差)** | 3.33 GiB | 4(未增) |
| C6 | C1 + **`MBT=512`** | 27122 | 996 | **0.33 GiB(最差)** | 4.46 GiB | 4(未增) |

## 三项独立确认
1. **"峰值是稳态、不是启动尖峰"成立**:C2/C3(不挂 16 GiB swap)的 **peak−steady = 0**;C4/C5/C6 也只有 **0.9–1.0 GiB**。⇒ **限捕获桶/清编译缓存/AOT 重载这类"降尖峰"手段不可能解决问题**(与 S14 的实测一致)。
2. **`MALLOC_ARENA_MAX=1` 无效**(C4:吃光 4 GiB swap 后 `oom_kill` 3→4);**`MBT=512` 无效**(C6:`minAvail` 0.33 GiB、`maxSwapUsed` 4.46 GiB,是六条里**最紧**的——注意它当时用的是现场编译口径,与 C1 可比)。
3. **AOT 重载对内存无益、对时间有益**:C5 的 `minAvail` 0.60 GiB **比 C1 的 0.86 GiB 更差**,与其 §3.6 的 "+1.8% 峰值、ready 351→230 s" 一致。

## 两条留给我自己(最终报告必须处理)
1. **C2 的 decode = 142.59 ms vs S14 P4e 的 126.83 ms(同为 TP=4 eager、同 maxlen/util/MBT)**——差 **12%**。**已排除的**:配置不同(P4e 的 `non-default args` 与 C2 逐项一致:`enforce_eager=True`、`max_model_len=262144`、`gpu_memory_utilization=0.85`、`max_num_batched_tokens=2048`、`Available KV cache memory` 都是 **18.69 GiB**)、bench 方法不同(都是 `vllm bench serve`、`random_input_len=128`/`output_len=128`/`num_prompts=5`/`seed=0`)、**陈旧引擎污染**(2026-10-10T00:28 复核:`ps` 里活着的只有 C6 那一套 1 个 EngineCore + 4 个 worker;`pgrep` 数到的 ~95 个绝大多数是 **zombie**,不占 CPU/内存)。**未定性的剩余候选**:① 挂不挂 16 GiB swapfile(P4e 挂了、C2 只有 `/swap.img` 4 GiB)——但 C2 期间 `minAvail` 还有 **4.69 GiB**、`maxSwapUsed` 只有 0.45 GiB,**当时并不缺内存**,所以这条不成立的可能性较大;② cache vintage / 首次 Triton JIT autotune;③ 宿主 CPU 频率/热状态。⇒ 已把"**同配置两种内存形态各测一次 TP=4 eager decode**"作为 **E5** 写进 [`tools/dcu/s14d_taskbook.md`](../../tools/dcu/s14d_taskbook.md),由 S14d 一并收口;在结论出来前,**C2 的 142.59 不得被当成"eager TP=4 的真实性能"**。
2. **C6 的 `MBT=512` 口径**:它是在**现场编译**下测的(与 C1 可比),不要与 C5(AOT 重载)混着比"MBT 的效果"。
