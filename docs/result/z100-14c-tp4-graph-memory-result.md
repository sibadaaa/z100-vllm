# S14c 27B TP=4 graph 模式的宿主内存归因与压缩 — 执行结果

> 归档时间：2026-10-09（边做边写；本文件在实验进行中持续追加）
> 执行者：子 agent（S14c）
> 对应设计文档：无独立 design 文档；任务来源 = Lead 下达的 S14c 任务书
> 前一档：[`z100-14-27b-tp4-result.md`](z100-14-27b-tp4-result.md)（md5 `7e39492ebf079f66c00c87c1481fd721`）、
> [`z100-14b-graph-vs-eager-result.md`](z100-14b-graph-vs-eager-result.md)（md5 `abf25e197e0053721cf8a706f9943235`）
> 状态：进行中 → 收尾时改为最终状态

## 0. 执行摘要（三句话）

1. **多出来的几 GB 是谁**：TP=4 graph 的宿主匿名**常驻**比 TP=4 eager 多 **≈1.44 GiB/worker（×4 ≈ +5.7 GiB）**，
   被 EngineCore 反向小掉的 1.4 GiB 部分抵消 ⇒ **净常驻 ≈ +3.5~4.3 GiB**；
   但它的**分配需求**比 eager 多 **≈7~13 GiB**（eager 全程 swap≈0，graph 必须换出 3~9 GiB）。
   **峰值不在启动尖峰上，而在"就绪后的稳态"**（`peak − steady` 只有 4~1469 MiB），
   所以"降启动尖峰"的所有做法（限捕获桶 / 清编译缓存 / AOT）都不解决问题 —— 与实测一致。
   桶归属（`[heap]` vs unnamed mmap）**不稳定**（同一配置可在两者间搬家 ≈1 GiB），**只有 `ΣRssAnon` 可比**；
   把 `MALLOC_ARENA_MAX=1` 打上去只会搬家、不减总量 ⇒ **那几 GB 是真实活跃数据，不是 glibc arena 浪费**。
2. **能否不挂 swap 跑 TP=4 graph**：**现场编译路径不能**（4 GiB `/swap.img` 下 3 次 OOM，`anon-rss 6.8 GiB`、死亡点逐字相同）；
   **AOT 重载路径能**，而且余量随 `vm.swappiness` 上升：`swappiness=60` → 余量 **0.72 GiB**、
   `swappiness=200` → **1.78 GiB**（此时 4 GiB `/swap.img` 被 100% 用满，是硬上限）。
   **再挂一个 4 GiB swapfile（总 8 GiB）+ `swappiness=200` ⇒ 余量 2.35 GiB，跨过 ≥2 GB 门禁。**
3. **推荐生产配方（实测达标）**：`TP=4 + graph(PIECEWISE, 捕获桶 [1,2,4,8]) + AOT 重载 + 4 GiB 附加 swapfile + vm.swappiness=200`
   ⇒ ready **230 s**、decode 中位 **49.65 ms/token**（= 基线 49.66 的 1.00×，远优于 55 ms 门禁）、
   **最低 MemAvailable 2.35 GiB**、swap 实际用 5.40 GiB、`oom_kill` 零增量。
   风险：余量只占 RAM 的 7.5%、swap 用掉 7.5 GiB 中的 5.4 GiB、`swappiness=200` 是**宿主级**内核参数；
   **未做**：该配方下的 conc8 / 8K prefill / 长时间稳态复测（见 §5）。

## 1. 目标与验收标准

任务书门禁原文：

* **G1 相位归因（本步核心）**：给出"宿主内存 × 时间"的**分相位**曲线（权重加载 / `torch.compile` /
  graph capture / warmup / 稳态），至少覆盖 **TP=4 graph（挂 swap）**、**TP=2 graph（不挂 swap）**、
  **TP=4 eager**；标出**峰值出现在哪个相位、比稳态高多少、由哪些进程构成**
  （`smaps_rollup` 的 `RssAnon`/`RssFile` + `[heap]`/`[anon]` 聚合）。**结论要能回答"多出来的几 GB 是谁"**。
* **G2 压缩试验（逐条 ABBA，一次只改一个因子）**，至少覆盖：glibc 堆（`MALLOC_ARENA_MAX=1`）；
  编译路径（`-cc.inductor_compile_config` 的 `combo_kernels`/`benchmark_combo_kernel`，**并区分现场编译 vs 磁盘重载 AOT 的峰值**）；
  `--max-num-batched-tokens`（2048→1024/512）与 `--max-num-seqs`；`-cc.cudagraph_capture_sizes`（默认 vs `[1,2,4,8]`）；
  `PYTORCH_HIP_ALLOC_CONF` 变体；`VLLM_DCU_USE_INT8_SDOT4` 仅作对照（已知无帮助，可跳过）。
  每条给"**峰值变化 + decode 是否劣化**"两个数字。
* **G3 结论与配方**：给出**可执行启动配方**，优先级 ① 完全不挂 swap 且峰值余量 ≥2 GB；
  ② 挂的小 swap ≤4 GiB 且余量 ≥2 GB；③ 都做不到则给"最小可行 swap + 实测余量 + 风险"及原因
  （具体到"哪几 GB 拿不掉"）。**decode 不得劣于 55 ms/token**（49.66 的 +10% 以内）。
* **G4 口径纪律**：每行标 并行度/模式/maxlen/util/n/是否挂 swap **和 swap 大小**/cache vintage；ACS 全程部署态不切；不得混口径比较。
* **G5 归档**：本文件 8 小节齐全，含原始命令、原始日志路径、md5、回滚件、生产建议。

**目标口径（不变）**：our tree + `VLLM_DCU_USE_INT8_SDOT4=1`、TP=4/2、`--dtype float16`、
`--max-model-len 262144`、`--max-num-batched-tokens 2048`、`--gpu-memory-utilization 0.85`、
`--enable-prefix-caching --mamba-cache-mode align --disable-custom-all-reduce --skip-mm-profiling`，
**不用** `--enforce-eager`。基准 = **49.66 ms/token**（TP=4 graph + 16 GiB swap + 4 桶捕获）。

## 2. 实际改动

| 文件 | 位置 | 改动摘要 |
|---|---|---|
| `tools/dcu/s14c_memtrace.sh` | 新增 | 5 s 采样的宿主内存 × 时间 tracer：`S`（host/cgroup/vram）、`P`（每进程 `RssAnon/RssFile/VmSwap` + `[heap]/unnamed/...` smaps 分桶）、`M`（引擎日志相位标记） |
| `tools/dcu/s14c_smaps_agg.awk` | 新增 | 把 `/proc/<pid>/smaps` 按 VMA 种类聚合：`[heap]`(=glibc 主 arena/brk) vs **unnamed mmap**(=glibc 次级 arena 等) vs `[stack]` vs file-backed |
| `tools/dcu/s14c_phase_report.py` | 新增 | 把 memtrace 按相位（P1 权重加载 → P2 model loaded → P3 compile → P4 capture → P5 kv/warmup → P6 稳态）汇总成表 + 全局峰值构成 + 降采样曲线 |
| `tools/dcu/s14c_arm_env.sh` | 新增 | `s14_arm_env.sh` 的副本 + S14c A/B 开关（`MALLOC_ARENA_MAX`/`MALLOC_TRIM_THRESHOLD`/`S14C_HIP_ALLOC_CONF`/`S14C_INDUCTOR_CFG`/`S14C_MAX_NUM_SEQS`/`S14C_EXTRA_ARGS`） |
| `tools/dcu/s14c_run_one.sh` | 新增 | `s14_run_one.sh` 的副本 + 自动起 memtrace + `S14C_*` 透传 + 产物落 `/mnt/game/s14c` |
| `tools/dcu/s14c_wait.sh` | 新增 | 等一个 arm 跑完，然后自动打印"G2 数字 + G1 极值 + 相位表 + 峰值构成"（避免逐次手工抄） |
| `tools/dcu/s14c_anon_top.sh` | 新增 | 每个 worker 的**最大匿名 VMA** 列表 + 匿名 Rss 的 **VMA 大小分桶**（§3.13 的形状证据） |
| `tools/dcu/s14c_swap.sh` | 新增 | 附加 swapfile 的 `on/off/rm/show` 记账（每次调用都把 `/proc/swaps`+`free -m` 前后原文落 `/mnt/game/s14c/hostlogs/swap-*.txt`） |
| 宿主内核参数 | **运行时改 + 已还原** | `sysctl -w vm.swappiness=150/200`（G3 的关键杠杆）；收尾还原为 `60`，前后原文见 `hostlogs/sysctl-{before,after-restore}.txt`。**未改任何 sysctl 配置文件** |

**未改动**（纪律 6）：`s14_*`/`s14b_*`/更早脚本、`docs/z100-todolist.md`、`docs/hygon_z100.md`、
`docs/海光Z100-性能优化方案.md`、DTK 库、`z100-build*`、`qwen38-vllm` 容器。

基线脚本 md5（冻结证据）：

```
e09d46bb44ce015667ae0cb301276da5  tools/dcu/s14_run_one.sh
a529ec55ed41c612370184337b373ed8  tools/dcu/s14_arm_env.sh
eced9511dca100c5a5eb6d22bc97d2c6  tools/dcu/s14_bench.sh
5a2944896b498bb8df4396fed299d419  tools/dcu/s14_monitor.sh
6ce50903a2334012b285ca82afe4c344  tools/dcu/s14_stop_arm.sh
d632e37c8038565772c1eeabafe94542  tools/dcu/s14_smaps.sh
```

## 3. 执行过程与原始证据

### 3.0 前置检查与宿主留档（2026-10-09T23:41Z）

```
$ date; pgrep -af "vllm|torchrun"
Fri Oct  9 11:41:04 PM UTC 2026
（无 live 引擎；`pgrep VLLM` 88 个历史 zombie，state=Z，VmRSS 空 ⇒ 不占内存）

$ free -g
               total        used        free      shared  buff/cache   available
Mem:              31           3          16           0          11          27
Swap:              3           0           3

$ cat /proc/swaps
Filename                                Type            Size            Used    Priority
/swap.img                               file            4194300         488656  -2
（⇒ `/mnt/game/swapfile`（16 GiB）当前 **swapoff**，与任务书 §2 描述一致）

$ hy-smi --showmeminfo vram   → DCU[0..3] Used = 0 / 32752 MiB（4 卡全空）

$ docker inspect z100-s14 --format 'Memory={{.HostConfig.Memory}} MemorySwap={{.HostConfig.MemorySwap}}'
Memory=0 MemorySwap=0
$ docker inspect qwen38-vllm --format 'Status={{.State.Status}} ExitCode={{.State.ExitCode}} Memory={{.HostConfig.Memory}}'
Status=exited ExitCode=137 Memory=0          ← 生产容器未被触碰（只读 inspect）
```

`oom_kill` 基线（容器 `z100-s14`，继承 S14 的 3 次）：本步全部 arm 都用**增量**判定。

### 3.1 工具自检：`[heap]` vs `unnamed` 分桶的正确性（本步第一个坑）

第一版 `s14c_smaps_agg.awk` 用 `/^\[/{m=$1}` 取映射名 —— **错**：`/proc/<pid>/smaps` 的映射头是
`<addr>-<addr> perms offset dev inode <name>`，名字是**第 6 个字段**（且默认 FS 会把尾随空格吃掉，
匿名映射的第 6 字段根本不存在）。修正后按 `NF<6 ⇒ unnamed`、`$6=="[heap]" ⇒ heap` 分桶，
用 `Anonymous:` 字段求和（而不是 `Rss:`），与 `/proc/<pid>/status` 的 `RssAnon` 对得上：

```
$ sudo -n awk -f tools/dcu/s14c_smaps_agg.awk /proc/<dockerd>/smaps
# cols: heap unnamed named stack other file anon_total
0 34 0 0 0 19 34          # dockerd: RssAnon=36064 kB=35.2 MiB ⇒ anon_total 34 MiB ✔
$ sudo -n awk -f tools/dcu/s14c_smaps_agg.awk /proc/<EngineCore>/smaps
1657 620 0 0 0 1286 2278  # RssAnon=2355096 kB=2299 MiB ⇒ anon_total 2278 MiB ✔
```

> **为什么这个分桶是本步的关键**：glibc 的**主** arena 是 `[heap]`（brk），但**每一个次级 arena
> 都是无名 `rw-p` mmap**，落在 `unnamed` 桶里 —— 所以 `MALLOC_ARENA_MAX` 的效果必须在
> `unnamed` 列上看，而不是 `[heap]` 列。S14 §3.7 只给了 `[heap]` 1.85–1.97 GiB/worker，
> 没有把次级 arena 拆出来，这正是本步要补的。

### 3.2 G1 arm C1：TP=4 graph + 16 GiB swap（现场编译，相位归因主臂）

**目的**：复现 S14 的 P4g2/P4g3（49.66 ms/token 基线），并第一次把它拆到相位粒度。

**swap 前后原文**（任务书纪律 4）：

```
$ cat /proc/swaps                      # BEFORE
Filename                                Type            Size            Used    Priority
/swap.img                               file            4194300         483280  -2

$ sudo -n swapon /mnt/game/swapfile    # rc=0   （未用 -p，保持 S14 的默认优先级 -3，保证可比）
$ cat /proc/swaps                      # AFTER
Filename                                Type            Size            Used    Priority
/swap.img                               file            4194300         483280  -2
/mnt/game/swapfile                      file            16777212        0       -3
$ free -m
               total        used        free      shared  buff/cache   available
Mem:           32009        3184       17075           0       11749       28361
Swap:          20479         471       20008
```

**启动命令**：

```
$ cd /home/lyh/z100-vllm
$ S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
    S14C_TAG="c1_tp4_graph_swap16_livecompile" S14C_MEMTRACE_IV=5 S14C_MEMTRACE_DUR=1800 S14C_HOLD_S=60 \
    READY_TIMEOUT=3600 S14_ENGINE_TIMEOUT=5400 \
    nohup tools/dcu/s14c_run_one.sh C1 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 \
    /mnt/game/s14c/cache-27b-c1 8430 262144 0,1,2,3 decode \
    > /mnt/game/s14c/hostlogs/orch-C1.log 2>&1 &
```

容器内生效行（`hostlogs/C1.log`）：

```
[s14c] name=C1 ... tp=4 sw=1 port=8430 maxlen=262144 car=1 util=0.85 mbt=2048 eager=0 cg_mode=PIECEWISE cg_sizes=1 2 4 8 extra=
[s14c] vintage_tag=c1_tp4_graph_swap16_livecompile
[s14c] knobs: MALLOC_ARENA_MAX=unset MALLOC_TRIM_THRESHOLD=unset PYTORCH_HIP_ALLOC_CONF=expandable_segments:True inductor_cfg=unset max_num_seqs=unset extra_args=unset
[s14c] cache=/mnt/game/s14c/cache-27b-c1 devices=0,1,2,3 served=qwen3.8-27b
```

**结果一览（与 S14 P4g2/P4g3 逐项对齐）**：

```
[orch14c] ready=1 after 351s  2026-10-09T23:49:02+00:00
[orch14c] summary: ready=351s aot_load=0 aot_recompile=0 oom_kill=3 |
   init engine (profile, create kv cache, warmup model) took 183.28 s (compilation: 112.83 s) |
   Available KV cache memory: 18.47 GiB
```

| 指标 | C1（本次） | S14 P4g2/P4g3 | 判读 |
|---|---|---|---|
| ready | **351 s** | 350 / 351 s | 逐秒复现 |
| `init engine` | 183.28 s（compilation 112.83 s） | 184.27 / 182.72 s（105.23 / 104.18 s） | 同档 |
| `Available KV cache` | 18.47 GiB | 18.47 GiB | 相同 |
| decode 中位 TPOT（n=5） | **50.14 ms**（Mean 50.13 / P90 50.32 / P99 50.41） | 49.46 / 49.86（合并 49.66） | **+1.0%**，在噪声内 |
| `oom_kill` 增量 | **0**（保持 3） | 0 | 相同 |

**memtrace 相位表**（`hostlogs/C1-memtrace.log`，5 s 采样，98 样本；`peakAnon`/`=heap`/`unnamed`/`file`
都是**全部引擎进程（4 worker + EngineCore）之和**的 MiB）：

```
phase              t0    t1   dur   n minAvail maxSwap peakAnon  =heap  unnamed  file maxCgGiB oomk
---------------------------------------------------------------------------------------------------
P0_bootstrap        0    96    96  18     9615     471        0      0        0     0    27.59    3
P1_weight_load    103   109     6   2    12793    2804    14718   8923     5702  3989    27.71    3
P2_compile        118   224   106  17     2810    5069    24929  10747    14169   252    27.60    3
P3_capture_kv     230   296    66  11     1616    6601    26001  10263    15719   435    27.61    3
P4_warmup         303   335    32   6     1147    7404    26395  10155    16218   409    27.60    3
P5_steady         341   484   143  22      885    8769    25775   9648    16082   830    27.45    3
```

（`t0/t1/dur` 单位 s，以 tracer 首个样本为 0；`P0_bootstrap` 的 96 s 是容器启动 + 引擎进程起飞前的窗口，
`n` 为样本数。相位边界原文见 memtrace 文件头。）

**全局峰值样本（原文，t=+335 s = 23:48:46Z，正好是 P4_warmup）**：

```
   MemAvailable=1147 MiB  MemFree=270 MiB Cached=1391 MiB  swap_used=7404 MiB
   cg_current=27.51 GiB oom_kill=3 vram=114344 MiB
   role      pid  RssAnon  [heap]  unnamed  RssFile  VmSwap  VmSize
   W     2973028     6279    2321     3953       97    1096   60105
   W     2973025     6270    2303     3962       98    1102   60187
   W     2973021     6091    2222     3865       99    1184   60107
   W     2973024     6044    2206     3833       98    1229   60119
   E     2969981     1711    1103      605       17    1544   29582
   TOT              26395   10155    16218      409    6155
```

**最低 MemAvailable 样本（t=+341 s，进入稳态的瞬间）**：

```
   MemAvailable=885 MiB  MemFree=245 MiB Cached=1151 MiB  swap_used=8769 MiB
   W 6131/5957/5693/5577 (heap 2171/2073/1927/1883, unnamed 3955/3880/3761/3689)
   E 1550 (heap 1120, unnamed 427)
   TOT  24908 MiB
```

**稳态中位数（P5，n=22，+341…+484 s）**：

```
   MemAvailable=2528 MiB  swap_used=8737 MiB
   SumRssAnon=24926 MiB (heap=9191 unnamed=15714)  SumRssFile=432 MiB  SumVmSwap=7464 MiB
   peak SumRssAnon (whole run) = 26395 MiB ; steady median = 24926 MiB ; peak - steady = 1469 MiB
```

#### 3.2.1 C1 的 G1 结论（回答"多出来的几 GB 是谁"）

1. **峰值相位 = P4_warmup（KV cache 分配后的 warmup，+303…+335 s）**，`ΣRssAnon = 26395 MiB (25.8 GiB)`。
   但**它只比稳态高 1469 MiB（5.9%）** —— **graph TP=4 的问题不是"启动尖峰"，而是稳态本身就太高**：
   稳态 `ΣRssAnon = 24926 MiB`，已是 31.26 GiB RAM 的 **78%**。
   ⇒ 这正是"启动期 `available` 只剩 0.86–1.1 GB"的原因，而不是某个瞬时分配没被回收。
2. **进程构成（峰值，MiB）**：

   | 进程 | RssAnon | `[heap]` | unnamed mmap | RssFile | VmSwap |
   |---|---|---|---|---|---|
   | Worker ×4（每个） | **6044–6279** | **2206–2321** | **3833–3962** | 97–99 | 1096–1229 |
   | EngineCore/APIServer | **1711** | 1103 | 605 | 17 | 1544 |
   | **合计** | **26395** | **10155** | **16218** | 409 | 6155 |

3. **"多出来的几 GB" 的口径**：与**本次自己的 TP=4 eager（C2，同一会话同一天）**逐项对比，
   而不是与 S14 那次的 P4e（跨会话）对比：

   | 口径 | 每 worker `RssAnon` | 每 worker `[heap]` | 每 worker `unnamed` | 每 worker `RssFile` |
   |---|---|---|---|---|
   | **C1 = TP=4 graph** | **6044–6279** | **2206–2321** | **3833–3962** | 97–99 |
   | C2 = TP=4 eager | 4607–4808 | 2019–2104 | 2567–2683 | 113–114 |
   | **Δ（graph − eager，TP 对齐）** | **+1437 ~ +1471** | +137 ~ +217 | **+1266 ~ +1279** | −15 |
   | （参照）S14 P4e = TP=4 eager | 4300–4520 | 1850–1970 | 2542–2670 | 190–200 |

   ×4 worker ⇒ 常驻 **+5.7~5.9 GiB**；EngineCore 反向 −1.4 GiB ⇒ **净常驻 ≈ +3.5~4.3 GiB**。
   **⚠️ 桶归属不可靠**：C1 与 C5（同配置、只差 AOT）的每 worker 总量几乎相同，但 `[heap]`/`unnamed`
   互换了 ≈1 GiB；C4（`MALLOC_ARENA_MAX=1`）也是同一现象 ⇒ **"多出来的在 unnamed 还是 `[heap]`"
   在本步的测量里不是稳定指纹，只有 `ΣRssAnon` 可比**（详见 §3.6 末尾与 §3.7 第 1 条）。
   上表那一行"Δ"只是**这一次采样**的分布，不能当作机制结论。
4. **`[heap]` 与"arena 浪费"的关系（本步被实测否证）**：C1 的 `Σ[heap] = 10155 MiB`
   （4 worker 9191 + EngineCore 1103）。`[heap]` 是 glibc **主 arena（brk）**，
   而 **glibc 的次级 arena 是无名 mmap**。按这个推理 `MALLOC_ARENA_MAX=1` 应该能砍掉一大块 `unnamed`——
   **实测（§3.5 的 C4）它只把 ≈2.4–3.1 GiB 从 `unnamed` 搬到 `[heap]`，总量不降反升，而且照样 OOM**
   ⇒ **`unnamed` 桶里绝大部分不是 arena 空闲/碎片，而是真实活跃的分配。**
   另外 **`[heap]` 在权重加载期就已经是 1.9–2.1 GiB/worker**（P1 相位：4 worker heap 合计 8923 MiB，
   当时还没开始 `torch.compile`）⇒ 这一块与编译无关，是**加载路径**（safetensors 读取 / 反量化 staging）。
5. **顺序读的旁证**：`P1_weight_load` 相位 `RssFile = 3989 MiB`（18 个 shard 的 page cache 计入 RSS），
   加载完立刻掉到 252 MiB ⇒ 与 S14 §3.7.2 判读一致（权重本体在**显存**，宿主只有 staging 页）。
6. **`maxCgGiB ≈ 27.5–27.7 GiB` 全程平坦**：cgroup `memory.current` 峰值 27.71 GiB，
   而**宿主 31.26 GiB RAM 被 25.8 GiB 引擎匿名 + 无法回收的页缓存挤爆**——这是 `global_oom` 的直接来源；
   容器无上限（`memory.max=max`）这一点被再次确认。
7. **swap 的定量角色（本次）**：启动期换出 **7.4–8.8 GB**，而引擎一关 swap 立刻回落到 ~0.4 GB
   （`free_after_stop`：`Swap: 20479 508 19971`）⇒ 被换出的确实**不是每 token 都碰的热页**；
   但 **decode 仍只有 50.14 ms/token**，说明代价没有被吃进延迟里。
   **然而最低 `MemAvailable = 0.86 GiB`** ⇒ **不满足 G3 的"余量 ≥2 GB"**。

### 3.3 G1 arm C2：TP=4 eager（**不挂** 16 GiB swapfile，只有 4 GiB `/swap.img`）

**swap 前后原文**（先 `swapoff`）：

```
$ cat /proc/swaps                      # BEFORE
/swap.img                               file            4194300         462104  -2
/mnt/game/swapfile                      file            16777212        31868   -3
$ sudo -n swapoff /mnt/game/swapfile   # rc=0
$ cat /proc/swaps                      # AFTER  ← 本 arm 全程只有 4 GiB
Filename                                Type            Size            Used    Priority
/swap.img                               file            4194300         460836  -2
```

**命令**：

```
$ S14C_CT=z100-s14 S14_EAGER=1 S14C_TAG="c2_tp4_eager_swap_swapimg4g_only" \
    S14C_MEMTRACE_IV=5 S14C_MEMTRACE_DUR=1200 S14C_HOLD_S=45 READY_TIMEOUT=1800 S14_ENGINE_TIMEOUT=3600 \
    nohup tools/dcu/s14c_run_one.sh C2 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 \
    /mnt/game/s14c/cache-27b-c2 8431 262144 0,1,2,3 decode > /mnt/game/s14c/hostlogs/orch-C2.log 2>&1 &
```

**结果**：

```
[orch14c] ready=1 after 160s  2026-10-09T23:56:15+00:00
[orch14c] summary: ready=160s aot_load=0 aot_recompile=0 oom_kill=3 | NA | Available KV cache memory: 18.69 GiB
decode: Successful 5  Median TPOT 142.59 ms (Mean 142.00, P90 143.79, P99 144.20)  Median TTFT 735.98
min MemAvailable = 4.69 GiB   max swap_used = 0.45 GiB   max cgroup.current = 27.65 GiB   oom_kill = 3（未增长）
```

```
phase              t0    t1   dur   n minAvail maxSwap peakAnon  =heap  unnamed  file maxCgGiB oomk
P1_weight_load      0   108   108  20    11768     459    16021   9636     6323  6803    27.65    3
P2_compile        115   121     6   2     7601     462    20214   9670    10441   692    27.56    3
P4_warmup         127   152    25   5     5947     463    21708  10528    11073   807    27.49    3
P5_steady         158   345   187  30     4798     464    22797  11139    11528   921    27.54    3
（无 P3：eager 不做 graph capture）

## GLOBAL PEAK (max Sum RssAnon)  t=+281s
   MemAvailable=4800 MiB  swap_used=464 MiB  oom_kill=3
   W 4808/4808/4609/4607 MiB  (heap 2104/2104/2019/2019, unnamed 2683/2683/2568/2567, RssFile 114)
   E(APIServer/EngineCore) 3116 MiB (heap 2436, unnamed 659)
   E(第二进程)               849 MiB (heap 457, unnamed 368, RssFile 402)
   TOT 22797 MiB (heap 11139, unnamed 11528, file 921)
## STEADY (P5) medians over n=30
   MemAvailable=4826 MiB  swap_used=464 MiB  SumRssAnon=22793 MiB (heap=11135 unnamed=11527)
   peak SumRssAnon = 22797 MiB ; steady 22793 ; peak - steady = 4 MiB
```

**判读**：

1. **eager TP=4 完全不需要那 16 GiB swapfile**：swap 峰值只有 **0.45 GiB**（= 启动前 `/swap.img` 的既有占用），
   `min MemAvailable = 4.69–4.80 GiB` ⇒ **满足 G3 的"余量 ≥2 GB"且不挂额外 swap**。
2. **但它的 decode = 142.59 ms/token**（本次实测），**远高于 55 ms/token 的门禁** ⇒ 不能作为生产配方。
   ⚠️ 与 S14 §3.4.1 的 P4e（126.83 ms）差 **+12.4%**：两次都是 eager/TP=4/同 maxlen/util/MBT。
   差异原因未定位（本机此刻 loadavg 5–6，另有多容器在跑；本步不重复验证），
   **故 C2 的 decode 只作"eager 数量级"用，不与 S14 的 126.83 混口径比较**。
3. **峰值 = 稳态（peak − steady = 4 MiB）**：eager 没有任何启动尖峰，22.8 GiB 匿名就是它的稳态。
4. **EngineCore 在 eager 下反而更大**（3116 MiB，`[heap]` 2436 MiB），而 graph 下只有 1711 MiB ⇒ 见 §3.2.1 第 3 条。

### 3.4 G1 arm C3：TP=2 graph（**不挂** 16 GiB swapfile）

**命令**（与 C1 逐字相同，只改 TP 与卡位；**同样只是 4 GiB `/swap.img`**）：

```
$ S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
    S14C_TAG="c3_tp2_graph_swap_swapimg4g_only" S14C_MEMTRACE_IV=5 S14C_MEMTRACE_DUR=1200 S14C_HOLD_S=45 \
    READY_TIMEOUT=2400 S14_ENGINE_TIMEOUT=3600 \
    nohup tools/dcu/s14c_run_one.sh C3 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 2 1 \
    /mnt/game/s14c/cache-27b-c3 8432 262144 0,1 decode > /mnt/game/s14c/hostlogs/orch-C3.log 2>&1 &
```

**结果**：

```
[orch14c] ready=1 after 281s  2026-10-10T00:05:04+00:00
[orch14c] summary: ready=281s aot_load=0 aot_recompile=0 oom_kill=3 |
   init engine (profile, create kv cache, warmup model) took 136.63 s (compilation: 92.64 s) |
   Available KV cache memory: 11.53 GiB
decode: Successful 5  Median TPOT 86.31 ms (Mean 86.32, P90 86.38, P99 86.41)  Median TTFT 642.09
min MemAvailable = 9.88 GiB   max swap_used = 1.00 GiB   oom_kill = 3（未增长）
```

```
phase              t0    t1   dur   n minAvail maxSwap peakAnon  =heap  unnamed  file maxCgGiB oomk
P1_weight_load      0   107   107  20    17699     485    10430   5528     4970  5162    27.67    3
P2_compile        113   201    88  16    12706     496    15328   7795     7471  3398    27.61    3
P3_capture_kv     207   244    37   7    11897     866    16077   7972     8056  3466    27.61    3
P4_warmup         250   274    24   5    10854     937    17139   8660     8433  3467    27.61    3
P5_steady         280   427   147  25    10121    1002    17777   9141     8566  3888    27.16    3

## GLOBAL PEAK (max Sum RssAnon)  t=+323s
   MemAvailable=10125 MiB  swap_used=1000 MiB  cgroup=27.16 GiB  oom_kill=3  vram=56970 MiB
   W 7106/7104 MiB  (heap 3251/3251, unnamed 3834/3832, RssFile 1176/1175)   ← 只有 2 个 worker
   E(APIServer/EngineCore) 2717 MiB (heap 2182, unnamed 531, RssFile 1132)
   E(第二进程)               850 MiB
   TOT 17777 MiB (heap 9141, unnamed 8566, file 3888)
## STEADY (P5) medians over n=25
   MemAvailable=10291 MiB  swap_used=1000 MiB  SumRssAnon=17562 MiB (heap=8936 unnamed=8549)
```

**判读**：

1. **TP=2 graph 极其宽裕**：峰值 17.4 GiB 匿名、`min MemAvailable = 9.88 GiB`、swap 峰值 1.0 GiB（既有占用），
   `oom_kill` 未增长 ⇒ **"graph 模式必须挂 swap"只对 TP=4 成立，对 TP=2 不成立**。
2. **decode 86.31 ms** 与 S14 的 P2（默认 51+2 桶 `FULL_AND_PIECEWISE`）84.22 同档（+2.5%）——
   即 4 桶 `PIECEWISE` 与默认图配置在本机对 TP=2 的 decode 几乎等价；
   但**本步 C3 的图配置与 S14 P2 不同（vintage 不同）**，不混口径。
3. **关键新观测（喂给 §3.2.1 的归因）**：TP=2 graph 每 worker 的 **`unnamed` = 3832–3834 MiB，
   与 TP=4 graph 的 3833–3962 MiB 几乎相同**（权重分片差 2×，这个桶几乎不动）
   ⇒ **`unnamed` 桶是"每进程固定开销"，与分片大小无关**；
   而 `[heap]` 从 TP=4 的 2206–2321 涨到 TP=2 的 3251 MiB，`RssFile` 从 97 涨到 1176 MiB
   ⇒ **与分片规模相关的是 `[heap]` + `RssFile`（加载/反量化 staging），不是 `unnamed`。**

### 3.5 G2 因子 1：`MALLOC_ARENA_MAX=1`（arm C4）—— ❌ 无效，且**否证了"glibc arena 浪费"假设**

**动机**：4 个 worker 的 `[heap]` 合计 ≈10 GiB；本机 12 核 ⇒ glibc 默认
`MALLOC_ARENA_MAX = 8 × 12 = 96`，而每个 worker 有 43–44 线程 ⇒ **次级 arena 可能很多**；
次级 arena 是**无名 mmap**，正好落在 C1 的 `unnamed` 桶（15.7 GiB 稳态）里。
这是最便宜的一刀，必须实测。

**命令**（唯一改动 = 环境变量 `MALLOC_ARENA_MAX=1`，**swap 只有 4 GiB `/swap.img`**）：

```
$ S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" S14C_MALLOC_ARENA_MAX=1 \
    S14C_TAG="c4_tp4_graph_arena1_swap_swapimg4g_only" S14C_MEMTRACE_IV=5 S14C_MEMTRACE_DUR=1500 S14C_HOLD_S=45 \
    READY_TIMEOUT=1800 S14_ENGINE_TIMEOUT=3600 \
    nohup tools/dcu/s14c_run_one.sh C4 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 \
    /mnt/game/s14c/cache-27b-c4 8433 262144 0,1,2,3 decode > /mnt/game/s14c/hostlogs/orch-C4.log 2>&1 &
```

**开关确实生效**（不是"没传进去"）：

```
$ sudo -n cat /proc/<APIServer 3094903>/environ | tr '\0' '\n' | grep -E '^(MALLOC_ARENA_MAX|PYTORCH_HIP_ALLOC_CONF)='
MALLOC_ARENA_MAX=1
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

**结果：❌ 仍然 OOM（第 6 次）**

```
[orch14c] ready=0 after 303s
[orch14c] summary: ready=303s aot_load=0 aot_recompile=0 oom_kill=4 | NA | Available KV cache memory: 18.47 GiB
(APIServer pid=79932) ERROR 10-10 08:13:17 [multiproc_executor.py:314] Worker proc VllmWorker-2 died unexpectedly (exit code: -9), shutting down executor.

$ sudo -n dmesg -T          # 6th global OOM 原文
[Sat Oct 10 00:13:11 2026] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=user.slice,mems_allowed=0,
    global_oom,task_memcg=/system.slice/docker-5292b9242a3899f70c1139274c9ec43145b8096ccda8938544ce15b67314d34b.scope,
    task=VLLM::Worker_TP,pid=3098140,uid=0
[Sat Oct 10 00:13:11 2026] Out of memory: Killed process 3098140 (VLLM::Worker_TP)
    total-vm:61293096kB, anon-rss:7109336kB, file-rss:0kB, shmem-rss:177240kB, UID:0 pgtables:21420kB
```

```
phase              t0    t1   dur   n minAvail maxSwap peakAnon  =heap  unnamed  file maxCgGiB oomk
P1_weight_load      0   111   111  20    11770    2889    17589  11159     6481  8111    27.83    3
P2_compile        119   225   106  17     2082    4095    25920  13164    12739   276    27.79    3
P3_capture_kv     231   277    46   8     1177    4095    26636  13319    13299   179    27.72    3   ← 死在这里
min MemAvailable = 0.96 GiB   max swap_used = 4.00 GiB（**4 GiB `/swap.img` 被吃光**）
```

**对照（唯一因子 = `MALLOC_ARENA_MAX`）**：

| 相位 | C1（未设，16 GiB swap） | C4（`=1`，4 GiB swap） | 判读 |
|---|---|---|---|
| P2_compile | peakAnon 24929（heap 10747 / unnamed 14169） | **25920**（heap **13164** / unnamed 12739） | 总量 +4.0%，**桶间搬家 ≈2.4 GiB** |
| P3_capture_kv | peakAnon 26001（heap 10263 / unnamed 15719） | **26636**（heap **13319** / unnamed 13299） | 总量 +2.4%，**桶间搬家 ≈3.1 GiB** |

> **结论（G2 因子 1）：`MALLOC_ARENA_MAX=1` 不是解药，而且它给出了一个更重要的诊断。**
> 它把 **≈2.4–3.1 GiB 从 `unnamed` 桶搬到了 `[heap]` 桶，但匿名总量没有下降（反而略升）**
> ⇒ **C1 的 `unnamed` 桶里绝大部分不是"glibc 次级 arena 的空闲/碎片"，而是真实活跃的分配**
> （`MALLOC_ARENA_MAX=1` 只是把同样的字节改从 brk 主 arena 出）。
> 同时**死亡点逐字相同**（`Available KV cache memory: 18.47 GiB` 之后 ~9 s、第 2 遍 capture 起步），
> 被 OOM 进程 `anon-rss = 6.78 GiB`，与 S14 的 6.76/7.09/7.21/7.28 GB 完全同档。

### 3.6 G2 因子 2：**现场编译 vs 磁盘重载 AOT**（arm C5）

**动机（S14b H-AOT 的直接推论）**：若 graph 模式的宿主峰值主要是 `torch.compile` 的**编译产物/现场编译器状态**，
那么"先用空 cache 预热填缓存、再以 AOT 重载方式服务"就能把峰值降下来 —— 这正是任务书点名要区分的两条路径。

**命令**（与 C1 **逐字相同**，唯一差别 = `S14_CACHE` 指向 C1 已经写满的 AOT cache）：

```
$ cat /proc/swaps | tail -2      # 16 GiB swapfile 已挂（同 C1）
/swap.img                               file            4194300         613952  -2
/mnt/game/swapfile                      file            16777212        0       -3
$ S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
    S14C_TAG="c5_tp4_graph_swap16_AOTreload" S14C_MEMTRACE_IV=5 S14C_MEMTRACE_DUR=1200 S14C_HOLD_S=60 \
    READY_TIMEOUT=1800 S14_ENGINE_TIMEOUT=3600 \
    nohup tools/dcu/s14c_run_one.sh C5 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 \
    /mnt/game/s14c/cache-27b-c1 8434 262144 0,1,2,3 decode > /mnt/game/s14c/hostlogs/orch-C5.log 2>&1 &
```

**AOT 重载确实成立**（4/4 rank 从磁盘直接载入，没有重新编译）：

```
$ grep -c "Directly load AOT compilation from path" hostlogs/C5.log
4
[orch14c] summary: ready=230s aot_load=4 aot_recompile=0 oom_kill=4（未增长）|
   init engine (profile, create kv cache, warmup model) took 87.42 s (compilation: 24.98 s) |
   Available KV cache memory: 18.48 GiB
decode: Successful 5  Median TPOT 49.96 ms (Mean 51.21, P90 53.59, P99 53.73)
```

**结果：AOT 重载把"时间"减半，但把"内存"完全没有降下来（反而略高）**：

| 指标 | C1 现场编译 | **C5 AOT 重载** | Δ |
|---|---|---|---|
| `init engine` | 183.28 s（compilation 112.83 s） | **87.42 s（compilation 24.98 s）** | **−96 s（−52%）** |
| ready | 351 s | **230 s** | **−121 s（−34%）** |
| decode 中位 TPOT | 50.14 ms | 49.96 ms | 同档 |
| **峰值 ΣRssAnon** | 26395 MiB | **26878 MiB** | **+483 MiB（+1.8%）** |
| 稳态 ΣRssAnon | 24926 MiB | 26590 MiB | +1664 MiB |
| **min MemAvailable** | 0.885 GiB | **0.60 GiB** | **更差** |
| max swap used | 8769 MiB | 3410 MiB | 更少 |
| `oom_kill` | 3（未增长） | 4（未增长） | 都不涨 |

```
phase              t0    t1   dur   n minAvail maxSwap peakAnon  =heap  unnamed  file maxCgGiB
P1_weight_load      0    82    82  15    11766     599    15973   9726     6141  8192    27.57
P2_compile         88   114    26   5     3029     525    24884  13551    11249   439    27.54
P3_capture_kv     120   172    52   9     1700    1942    25910  13783    12039   667    27.58
P4_warmup         179   210    31   6     1218    2783    26878  13957    12894   537    27.56   ← 峰值
P5_steady         218   358   140  22      611    3410    26590  14084    12462   755    27.60

## GLOBAL PEAK t=+210s
   W 6250/6241/6094/6080 MiB  (heap 3215/3206/3201/3188, unnamed 3024/3022/2882/2880)
```

> **结论（G2 因子 2）**：**H-AOT 对内存不成立、对时间成立。**
> ① 内存：峰值 **26395 → 26878 MiB（+1.8%）**，`min MemAvailable` **0.885 → 0.60 GiB（更差）**
> ⇒ **graph 模式多出来的几 GB 不是现场编译的 scratch/编译产物**（重载后同样大）；
> ② 时间：`init engine` **183 → 87 s**、ready **351 → 230 s（−34%）**，decode 不变（49.96 ms）
> ⇒ **"先预热填缓存、再以重载方式服务" 作为"生产配方"在时间上是净收益，在内存上不是解药**；
> ③ 可作为生产配方的**加速部件**（启动快 121 s），但不解决 swap 需求。

**⚠️ 一个重要的方法学发现（必须登记，会修正 §3.2.1 的表述）**：
C1 与 C5 的**每 worker 总量几乎一样（6044–6279 vs 6080–6250 MiB），但 `[heap]` / `unnamed` 的
分配明显不同**（C1：heap 2206–2321 / unnamed 3833–3962；C5：heap **3188–3215** / unnamed **2880–3024**）。
C4（`MALLOC_ARENA_MAX=1`）也出现同样的"heap 变大、unnamed 变小"。
⇒ **`[heap]` vs `unnamed` 的桶间分配不是一个稳定指纹（受分配顺序/碎片影响，可在 1 GiB 量级上来回搬家），
只有 `ΣRssAnon` 总量是稳定的比较量。** §3.2.1 第 3/4 条关于"主要在 unnamed 而不是 `[heap]`"的说法
**据此降级为"桶间分配不稳定，只有总量可比"**；结论改为：
**graph 比 eager 多 ≈1.4 GiB/worker 的匿名总量（TP 对齐后），其桶归属不稳定。**

### 3.7 G1 收口：三口径对照（全部为本次实测，口径逐行标注）

| # | arm | TP | 模式 | 图配置 | swap 挂载 | maxlen | util | MBT | n | ready | decode 中位 TPOT | 峰值 ΣRssAnon | 稳态 ΣRssAnon | peak−steady | min MemAvailable | max swap used | oom_kill |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | graph | 4 | VLLM_COMPILE | PIECEWISE, [1,2,4,8] | `/swap.img` 4 GiB + **swapfile 16 GiB** | 262144 | 0.85 | 2048 | 5 | 351 s | **50.14 ms** | **26395 MiB** | 24926 MiB | **+1469** | **0.885 GiB** | 8.77 GiB | 3→3 |
| C5 | graph | 4 | VLLM_COMPILE | 同上 | 同上（AOT 重载） | 262144 | 0.85 | 2048 | 5 | **230 s** | **49.96 ms** | 26878 MiB | 26590 MiB | +288 | 0.60 GiB | 3.41 GiB | 4→4 |
| C4 | graph | 4 | VLLM_COMPILE | 同上 + `MALLOC_ARENA_MAX=1` | **仅 `/swap.img` 4 GiB** | 262144 | 0.85 | 2048 | — | ❌ OOM | — | 26636 MiB（死前） | — | — | 0.96 GiB | 4.00 GiB（吃光） | 3→**4** |
| C2 | eager | 4 | NONE | — | **仅 `/swap.img` 4 GiB** | 262144 | 0.85 | 2048 | 5 | 160 s | 142.59 ms | 22797 MiB | 22793 MiB | **+4** | **4.80 GiB** | 0.46 GiB | 3→3 |
| C3 | graph | 2 | VLLM_COMPILE | PIECEWISE, [1,2,4,8] | **仅 `/swap.img` 4 GiB** | 262144 | 0.85 | 2048 | 5 | 281 s | 86.31 ms | 17777 MiB | 17562 MiB | +215 | **10.13 GiB** | 1.00 GiB | 3→3 |

**对照 S14（不混口径，只作参照）**：P4g2/P4g3（=C1 的同一配置）49.46/49.86 ms、ready 350/351 s、swap 峰值 7.78 GB；
P4e（=C2 的同一配置）126.83 ms；P2（默认 51+2 桶）84.22 ms。

**G1 的三条判读（回答"多出来的几 GB 是谁"）**：

1. **"多出来的几 GB" 必须分成两个量说：常驻（resident）增量 与 分配需求（demand）增量。**
   * **常驻增量（TP=4 对齐，同口径）**：graph 每 worker **6044–6279 MiB** vs eager **4607–4808 MiB**
     ⇒ **+1437 ~ +1471 MiB/worker**，×4 = **+5.7–5.9 GiB**；
     EngineCore/APIServer 反向 **−1405 MiB**（graph 1711 MiB vs eager 3116 MiB）
     ⇒ **净常驻增量 ≈ +3.5 ~ +4.3 GiB**（C1 峰值 26395 / C5 26878 / C6 27237 vs C2 22797）。
   * **分配需求增量**：eager 全程 swap≈0（0.46 GiB 是既有占用），
     而 graph 必须换出 **3.41–8.77 GiB** 才能装下 ⇒ **需求增量 ≈ +7 ~ +13 GiB**。
   * ⇒ **S14 §3.8.1 的"≈8.8 GB 缺口"按"分配需求"口径是站得住的**；
     但它**不能用"graph TP=2 的 6.6 GiB/worker vs eager TP=4 的 4.4 GiB/worker"来推**
     （那是 TP 与模式双重变化）。**TP 对齐后的常驻增量只有 ≈4 GiB，其余约 3–9 GiB 是被换出的部分。**
     本步 §3.6 已经用 `ΣRssAnon + ΣVmSwap` 复核过这个量级，但它的运行间离散度较大（见 §3.9 噪声底），
     所以正式表述用"常驻 +5.7 GiB / 需求 +7–13 GiB"这种区间。
   * **桶归属不稳定**：C1 与 C5 每 worker 总量几乎相同（6044–6279 vs 6080–6250 MiB），
     但 `[heap]`（2206–2321 vs 3188–3215）与 `unnamed`（3833–3962 vs 2880–3024）**互换了 ≈1 GiB**；
     C4（`MALLOC_ARENA_MAX=1`）也是同一现象。⇒ **只有 `ΣRssAnon` 总量可比，
     "多出来的在 `[heap]` 还是 unnamed" 在本步的测量里不是一个稳定指纹**（见 §3.6 末尾）。
2. **峰值不在"启动尖峰"上，而在稳态本身**（C1/C2/C3 三条 `peak − steady` 分别只有
   **1469 / 4 / 215 MiB**）⇒ **这是"稳态装不下"，不是"瞬时尖峰装不下"**。
   因此"降启动尖峰"的做法（限捕获桶、AOT 重载、清编译缓存）都不解决问题，与实测一致。
3. **峰值相位 = P4_warmup（KV cache 分配后的 warmup）**：三条臂的 `peakAnon` 都落在
   P4_warmup 或 P5_steady，而 `min MemAvailable` 往往落在 **P5_steady**（C1：0.885 GiB @ +341 s，
   C5：0.611 GiB @ +358 s）⇒ **最危险的时刻是"就绪后的稳态/bench 期间"，不是就绪前**。
   这与 S14 §3.8.2 观察到的"bench 期间 swap 用到 7.78 GB、available 只剩 1.0–1.9 GB"是同一现象。

### 3.8 G2 因子 3：`--max-num-batched-tokens` 2048 → 512（arm C6）

**命令**（唯一改动 = `S14_MBT=512`；现场编译，新 cache；16 GiB swapfile 已挂）：

```
$ S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" S14_MBT=512 \
    S14C_TAG="c6_tp4_graph_swap16_mbt512" S14C_MEMTRACE_IV=5 S14C_MEMTRACE_DUR=1200 S14C_HOLD_S=30 \
    READY_TIMEOUT=1800 S14_ENGINE_TIMEOUT=3600 \
    nohup tools/dcu/s14c_run_one.sh C6 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 \
    /mnt/game/s14c/cache-27b-c6 8435 262144 0,1,2,3 decode > /mnt/game/s14c/hostlogs/orch-C6.log 2>&1 &
```

**结果：❌ 没有帮助，反而略差**

```
[orch14c] ready=1 after 340s | init engine took 156.79 s (compilation: 99.05 s) |
   Available KV cache memory: 18.41 GiB | oom_kill=4（未增长）
decode: Successful 5  Median TPOT 50.01 ms (Mean 50.06, P90 50.37, P99 50.48)   ← decode 不变（好）
min MemAvailable = 0.24 GiB   max swap_used = 6.74 GiB   max cgroup.current = 27.70 GiB
```

```
phase              t0    t1   dur   n minAvail maxSwap peakAnon  =heap  unnamed  file maxCgGiB
P1_weight_load      0   110   110  20    10525     530    17652   9621     8007  8251    27.70
P2_compile        117   212    95  16     1343    1843    26514  12720    13714   411    27.63
P3_capture_kv     218   267    49   8      532    3670    27237  12126    15056   683    27.60   ← 峰值
```

**对照（唯一因子 = MBT）**：

| 指标 | C1（MBT 2048） | **C6（MBT 512）** | Δ |
|---|---|---|---|
| decode 中位 TPOT | 50.14 ms | **50.01 ms** | **无劣化（还快 0.3%）** |
| 峰值 ΣRssAnon | 26395 MiB | **27237 MiB** | **+842 MiB（+3.2%）** |
| min MemAvailable | 0.885 GiB | **0.24 GiB** | **−0.645 GiB** |
| ready | 351 s | 340 s | 同档 |
| Available KV cache | 18.47 GiB | 18.41 GiB | 同档 |

⇒ **MBT 512 是"免费"的（decode 不劣化），但在内存上是零收益/负收益**：
它把 profile/warmup 的 token 预算砍到 1/4，峰值反而 +842 MiB、最低余量 −0.645 GiB。
**不建议作为省内存手段。**

### 3.9 噪声底：本次运行的因子效应分辨率（**必读，决定了上面几条的置信度**）

C1 / C5 / C6 三条臂在**"什么都不改"或"改了不影响分配"**的情况下，`min MemAvailable` 分别是
**0.885 / 0.600 / 0.240 GiB**，峰值 ΣRssAnon 分别是 **26395 / 26878 / 27237 MiB**。

| 量 | C1 | C5 | C6 | 极差 |
|---|---|---|---|---|
| 峰值 ΣRssAnon | 26395 | 26878 | 27237 | **842 MiB** |
| 稳态 ΣRssAnon | 24926 | 26590 | — | ≥1664 MiB |
| min MemAvailable | 0.885 | 0.600 | 0.240 | **0.645 GiB** |

> **⇒ 本步的单次运行噪声底约 ±0.4 GiB（峰值）与 ±0.35 GiB（余量）。**
> 因此：
> * §3.5（`MALLOC_ARENA_MAX=1`，峰值 +241 MiB）与 §3.6（AOT 重载，峰值 +483 MiB）的**峰值差异在噪声内**，
>   它们的**真正结论来自"是否 OOM / 是否 ready / 时间"，而不是那几百 MiB**；
> * §3.8（MBT 512，峰值 +842 MiB、余量 −0.645 GiB）**刚刚超出噪声底**，可判为"轻微负收益"；
> * **任何 <0.7 GiB 的"改善"在本步的测量分辨率下都不能算数**——这也是为什么下面几条
>   "没帮上忙"的结论必须写成"未见改善"而不是"更差"。
> * 之所以噪声这么大：宿主 12 核在跑多个其它容器（loadavg 5–6），且 `min MemAvailable` 是
>   一个受页缓存回收时机影响的极值统计量。
> * **旁证（本步实测）**：同一时刻宿主上除本引擎外**没有任何别的匿名内存大户** ——
>   `docker stats` 里 `z100-s12`/`z100-a7*`/`z100-s9`/`z100-serve`/`z100-build*` 全部是 **KB 级**；
>   全宿主按 `RssAnon` 排序第 5 名是 `dockerd`（30 MiB）。
>   ⇒ **"换台更干净的机器"或"停掉别的容器"在这台机上救不了这 4 GiB。**

### 3.10 G2 因子 4：默认图配置（`FULL_AND_PIECEWISE` + 51+2 桶）vs `PIECEWISE` + `[1,2,4,8]`（arm C7）

**动机**：S14 §3.8.4 明确登记了本步最大的保留意见 —— **49.66 ms 那个 arm 用的是 4 桶 `PIECEWISE`，
而 P2 的 84.22 用的是默认 51+2 桶**，所以"TP=4 快 1.70×"里混着"图配置"的贡献，
而且**默认图配置在本机从没跑起来过**（P4 无 swap → OOM；P4g1 限了桶 → OOM）。
本 arm 就是把默认图配置**挂 16 GiB swap 跑一次**，把这条保留意见变成实测数字。

**命令**（不设 `S14_CG_MODE`/`S14_CG_SIZES`，即用 vLLM 默认；其余与 C1 逐字相同）：

```
$ S14C_CT=z100-s14 S14C_TAG="c7_tp4_graph_swap16_DEFAULT_cudagraph" \
    S14C_MEMTRACE_IV=10 S14C_MEMTRACE_DUR=2400 S14C_HOLD_S=30 READY_TIMEOUT=3000 S14_ENGINE_TIMEOUT=5400 \
    nohup tools/dcu/s14c_run_one.sh C7 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 \
    /mnt/game/s14c/cache-27b-c7 8436 262144 0,1,2,3 decode > /mnt/game/s14c/hostlogs/orch-C7.log 2>&1 &
```

**结果：✅ 默认图配置在 16 GiB swap 下**跑起来了**，而且是 decode 最快的一臂**

```
(Worker_TP0..3) INFO [model_runner.py:1060] Graph capturing finished in 646 secs, took 1.64 GiB
   ← 4 桶配置是 "6~8 secs, took 0.10~0.22 GiB"；图池 1.64 GiB/worker（与 S14 P4 的 1.67 GiB 一致）
(Worker_TP0) INFO [gpu_worker.py:719] Available KV cache memory: 16.99 GiB     ← 4 桶是 18.47，图池吃掉 1.48 GiB
[orch14c] ready=1 after 1645s  2026-10-10T00:58:26+00:00
[orch14c] summary: ready=1645s aot_load=0 aot_recompile=0 oom_kill=4（未增长）|
   init engine (profile, create kv cache, warmup model) took 1482.41 s (compilation: 103.13 s)
decode: Successful 5  Median TPOT 47.97 ms (Mean 48.02, P90 48.19, P99 48.28)   ← 目前最快
min MemAvailable = 0.06 GiB (66 MiB!)   max swap_used = 8.80 GiB
```

```
phase              t0    t1   dur   n minAvail maxSwap peakAnon  =heap  unnamed  file maxCgGiB oomk
P1_weight_load     85   107    22   3    12072     527    15898   9649     6141  6804    27.65    4
P2_compile        119   209    90   9      560    2502    27355  12722    14552   118    27.79    4
P3_capture_kv     221  1585  1364  99       66    6002    27387  12332    15041   230    27.73    4   ← 两遍捕获，1364 s
P4_warmup        1596  1631    35   4      463    7581    27122  11224    15878   638    27.46    4
P5_steady        1642  1746   104  10     1130    9008    26250  10365    15836   968    27.35    4

## GLOBAL PEAK t=+923s（第 1 遍捕获结束、KV cache 分配后）
   MemAvailable=357 MiB  MemFree=242 MiB Cached=677 MiB  swap_used=4089 MiB  cg=27.43 GiB  vram=36270 MiB
   W 6881/6858/6774/6759 MiB  (heap 3113/3092/3064/3056, unnamed 3766/3762/3707/3698, RssFile 54–58)
   E 115 MiB (heap 7, unnamed 108, **VmSwap 2176 MiB** ← APIServer 被换出 2.1 GiB)
   TOT 27387 MiB (heap 12332, unnamed 15041, file 230)
```

**对照（唯一因子 = 图配置）**：

| 指标 | C1（`PIECEWISE` 4 桶） | **C7（默认 `FULL_AND_PIECEWISE` 51+2 桶）** | Δ |
|---|---|---|---|
| ready | 351 s | **1645 s** | **+1294 s（4.7×）** |
| 捕获耗时 / 图池 | 6–8 s / 0.10–0.22 GiB | **646 s×2 遍 / 1.64 GiB** | 图池 7–16× |
| `Available KV cache` | 18.47 GiB | 16.99 GiB | −1.48 GiB（显存） |
| decode 中位 TPOT | 50.14 ms | **47.97 ms** | **−4.3%（更快）** |
| 峰值 ΣRssAnon | 26395 MiB | **27387 MiB** | **+992 MiB** |
| min MemAvailable | 0.885 GiB | **0.066 GiB** | **−0.82 GiB（几乎 OOM）** |
| max swap used | 8.77 GiB | 8.80 GiB | 相同 |
| `oom_kill` | 3 | 4 | 都不增长 |

> **结论（G2 因子 4）**：
> ① **默认图配置并不是"跑不起来"，只是"没有 16 GiB swap 就跑不起来"** —— 挂上之后它
>    `Graph capturing finished in 646 secs ×2 遍`，`oom_kill` 未增长，且 decode **47.97 ms 比 4 桶还快 4.3%**；
> ② 代价是 **ready 从 351 s 变成 1645 s（+21.6 min）**、`min MemAvailable` 从 0.885 GiB 掉到 **66 MiB**；
> ③ ⇒ **S14 §3.8.4 的"图配置不同"保留意见可以部分关闭**：
>    49.66 与 84.22 之间的 1.70× **不主要是图配置造成的**（默认配置的 TP=4 decode 甚至更快）；
>    **4 桶 `PIECEWISE` 的真正价值是"启动时间 + 余量"，不是速度**。
> ④ 生产上**不建议**用默认 51+2 桶：**多花 21.6 min 启动、把余量压到 66 MB，只换 4.3% 的 decode**。

### 3.11 G2 因子汇总表 + 因子 5/6/7

#### 3.11.0 七个因子的总表（每条都给"峰值变化 + decode 是否劣化"）

全部以 **C1（TP=4 / graph / `PIECEWISE` + `[1,2,4,8]` / maxlen 262144 / util 0.85 / MBT 2048 / 16 GiB swapfile / 现场编译）**
为对照；"峰值变化"指峰值 `ΣRssAnon`（全部引擎进程之和）；噪声底见 §3.9。

| # | 因子 | arm | 唯一改动 | 峰值 ΣRssAnon | min MemAvailable | decode 中位 TPOT | 判定 |
|---|---|---|---|---|---|---|---|
| 1 | **glibc 堆** | C4 | `MALLOC_ARENA_MAX=1`（**swap 只有 4 GiB**） | 死前 26636（+241） | — | — | ❌ **仍 OOM（第 6 次）**；死亡点逐字相同；桶间搬家 ≈2.4–3.1 GiB 但总量不减 ⇒ **否证"arena 浪费"假设** |
| 2 | **编译路径（时间）** | C5 | AOT 磁盘重载（同一 cache） | 26878（+483） | 0.885 → **0.600** | 50.14 → 49.96 | ⏱ **时间 −121 s（−34%）**；内存无收益 |
| 3 | `--max-num-batched-tokens` | C6 | 2048 → **512** | 27237（+842） | 0.885 → **0.240** | 50.14 → 50.01 | ❌ decode 不劣化，但**内存零/负收益** |
| 4 | `--cudagraph-capture-sizes` | C7 | 默认 `FULL_AND_PIECEWISE` 51+2 桶 | 27387（+992） | 0.885 → **0.066** | 50.14 → **47.97（−4.3%）** | ⚠️ **可行**（挂 16 GiB swap 后 oom_kill 不涨）但 ready **+1294 s**、余量仅 **66 MB** ⇒ 不值 |
| 5 | **编译路径（inductor）** | C8 | `combo_kernels=false, benchmark_combo_kernel=false`（换 AOT key，新 cache） | 27430（+1035） | 0.885 → **0.200** | 50.14 → **50.87（+1.5%，P99 57.1）** | ❌ 无收益、尾延迟变差 |
| 6 | `--max-num-seqs` | C9 | 默认 → **64** | 27034（+639） | 0.885 → **0.690** | 50.14 → **52.27（+4.2%）** | ❌ 无收益、decode 变差（KV 池反而 +0.36 GiB 显存） |
| 7 | `PYTORCH_HIP_ALLOC_CONF` | C17 | `expandable_segments:False`（在 **C14 达标配方**上改） | 25150（vs C14 25473，−323） | 2.35 → **2.75** | 49.65 → 50.09 | ➖ **中性**（设备侧分配器；差异全在噪声内） |
| ✚ | **（本步新发现）AOT 重载 + 4 GiB swap** | C11 | 在 C4 的"4 GiB swap"上把现场编译换成 AOT 重载 | — | 0.96（OOM）→ **0.72** | 50.04 | ✅ **把 OOM 变成可用** |
| ✚ | **（本步新发现）`vm.swappiness`** | C15/C13 | `60 → 200` | 25626/24837（−769/−1558） | 0.72 → **1.78**；0.29 → **2.71** | 50.04 → 50.30 / 49.60 → 49.83 | ✅ **余量 +1.06 ~ +2.65 GiB，decode 零代价** |

**总判读**：**七个"压缩宿主内存"的因子全部失败（含 4 个是负收益）**；
真正的两个杠杆是 **AOT 重载（把换出需求从 8.8 GiB 降到 ~3 GiB）** 与 **`vm.swappiness`（把冷匿名页换出去）**——
**两者都不削需求，而是改变"需求如何在 RAM 与 swap 之间分配"**。

#### 3.11.1 因子 5：`-cc.inductor_compile_config='{"combo_kernels": false, "benchmark_combo_kernel": false}'`（arm C8）

**为什么它是"编译路径"里最该试的一条**：vLLM 在 torch ≥ 2.9.0.dev 上**默认打开**
`combo_kernels`/`benchmark_combo_kernel`（`vllm/config/compilation.py:987-995`），
`benchmark_combo_kernel` 在编译期会**基准测试候选融合核**（产生额外的中间对象/子进程）。
它**会改变 AOT key**，所以必须用**新 cache**（vintage = `c8_tp4_graph_swap16_nocombokernel`）。

**开关确实生效**（不是"没传进去"，`non-default args` 原文节选）：

```
'inductor_compile_config': {'combo_kernels': False, 'benchmark_combo_kernel': False,
                            'enable_auto_functionalized_v2': False, 'size_asserts': False,
                            'alignment_asserts': False, 'scalar_asserts': False}
```

**结果**：

```
[orch14c] ready=1 after 342s | init engine 170.40 s (compilation: 101.06 s) | Available KV 18.47 GiB | oom_kill=4
decode: Successful 5  Median TPOT 50.87 ms (Mean 53.08, P90 56.81, P99 57.14)
min MemAvailable = 0.20 GiB   max swap_used = 4.94 GiB
phase P2_compile peakAnon 26331 | P3_capture_kv peakAnon 26793 | 全局峰值 27430 MiB
```

⇒ **无内存收益（峰值 27430 vs C1 26395，+1035 MiB，略超噪声底），decode 反而差 1.5%、
尾延迟明显变差（P90 56.81 / P99 57.14 vs C1 的 50.32 / 50.41）** ⇒ **不建议关闭**。

#### 3.11.2 因子 6：`--max-num-seqs 64`（arm C9）

**动机**：`max_model_len=262144` 下每个序列的 block table 是 `262144/16 = 16384` 项；
若某个宿主/显存缓冲按 `max_num_seqs × 块数` 分配，把它从默认（vLLM 默认 256）降到 64 会有量级收益。

```
$ S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" S14C_MAX_NUM_SEQS=64 \
    nohup tools/dcu/s14c_run_one.sh C9 ... 4 1 /mnt/game/s14c/cache-27b-c1 8438 262144 0,1,2,3 decode &
（虽然复用了 cache-27b-c1，但 max_num_seqs 进了编译配置 ⇒ **aot_load=0，仍然现场编译 102.99 s**）
[orch14c] ready=1 after 310s | Available KV cache memory: 18.83 GiB | oom_kill=4
decode: Successful 5  Median TPOT 52.27 ms (Mean 52.32, P90 54.66, P99 55.38)
min MemAvailable = 0.69 GiB   max swap_used = 3.57 GiB
phase P3_capture_kv peakAnon 26713 | P4_warmup 27034
```

⇒ **无内存收益（峰值 27034 vs 26395，+639 MiB，噪声内），decode 差 4.2%** ⇒ **不建议降 `max-num-seqs`**。
（旁证：`Available KV cache` 反而从 18.47 涨到 18.83 GiB ⇒ 降 `max_num_seqs` 主要影响**显存侧的 KV 池**，不是宿主匿名。）

#### 3.11.3 因子 7：`PYTORCH_HIP_ALLOC_CONF` 变体（arm C17）

**先验**：`PYTORCH_HIP_ALLOC_CONF` 是 **设备侧** caching allocator 的开关，**理论上对宿主匿名内存零影响**。
本 arm 就是把这个"理论上"变成实测：在**达标配方**（C14）上只改这一个变量。

```
$ S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
    S14C_HIP_ALLOC_CONF="expandable_segments:False" \
    nohup tools/dcu/s14c_run_one.sh C17 ... 4 1 /mnt/game/s14c/cache-27b-c1 8446 262144 0,1,2,3 decode &
（容器内回声：PYTORCH_HIP_ALLOC_CONF=expandable_segments:False）
[orch14c] ready=1 after 241s | init engine 91.30 s (compilation: 23.85 s) | oom_kill=4
decode: Successful 5  Median TPOT 50.09 ms (Mean 50.05, P90 50.23, P99 50.23)
min MemAvailable = 2.75 GiB   max swap_used = 6.38 GiB
```

**对照（唯一因子 = HIP alloc conf；两者 swappiness=200、总 swap 8 GiB、AOT 重载）**：

| 指标 | C14 (`expandable_segments:True`) | **C17 (`:False`)** | Δ |
|---|---|---|---|
| min MemAvailable | 2.35 GiB | 2.75 GiB | +0.40 GiB（**噪声内**） |
| 峰值 ΣRssAnon | 25473 MiB | 25150 MiB | −323 MiB（**噪声内**） |
| decode 中位 TPOT | 49.65 ms | 50.09 ms | +0.9%（噪声内） |
| ready | 230 s | 241 s | 同档 |

⇒ **中性**（符合先验）。**建议保持 vLLM/本仓的默认 `expandable_segments:True`** ——
`False` 在长期服务里有显存碎片风险，而本步没有量到任何收益。

### 3.12 G3：把"能跑"变成"生产可用"（arm C11~C15）

G2 的七个因子没有一个能削掉常驻需求。于是转入 **G3 的第三个口径**：
**在不削需求的前提下，能不能靠"少换一点/换得聪明一点"把余量做出来？**
关键洞察来自 C5/C11 的对照：**AOT 重载路径的 swap 用量（3.0~3.4 GiB）远小于现场编译路径（8.8 GiB）**，
尽管两者的常驻峰值差不多。顺着这条线做了 5 个 arm：

```
$ tools/dcu/s14c_swap.sh off /mnt/game/swapfile C10               # 先还原 16 GiB
$ tools/dcu/s14c_swap.sh on  /mnt/game/s14c/swap6 6144 C10        # 附加 6 GiB，总 10 GiB
$ sudo -n sysctl -w vm.swappiness=150  /  200                     # C12 / C13
$ tools/dcu/s14c_swap.sh off /mnt/game/s14c/swap6 C14
$ tools/dcu/s14c_swap.sh on  /mnt/game/s14c/swap4 4096 C14        # 附加 4 GiB，总 8 GiB
所有 C10~C15 都用同一份 AOT cache（`/mnt/game/s14c/cache-27b-c1`，`aot_load=4`）
```

| arm | 附加 swap | 总 swap | `vm.swappiness` | 其它 | ready | decode 中位 TPOT | **min MemAvailable** | swap 峰值 | `oom_kill` 增量 | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| C4 | 0 | 4 GiB | 60 | 现场编译 + `MALLOC_ARENA_MAX=1` | ❌ | — | 0.96 GiB（死前） | **4.00 GiB（用满）** | **+1（第 6 次 OOM）** | ❌ |
| C11 | 0 | 4 GiB | 60 | AOT 重载 | 221 s | 50.04 ms | 0.72 GiB | 3.02 GiB | 0 | ⚠️ 活但余量小 |
| **C15** | **0** | **4 GiB** | **200** | AOT 重载 | 241 s | 50.30 ms | **1.78 GiB** | **4.00 GiB（用满）** | 0 | ⚠️ 口径 ① 差 0.22 GiB |
| **C14** | **+4 GiB** | **8 GiB** | **200** | AOT 重载 | **230 s** | **49.65 ms** | **2.35 GiB** | 5.40 GiB | 0 | ✅ **口径 ② 达标** |
| C12 | +6 GiB | 10 GiB | 150 | AOT 重载 | 230 s | 49.94 ms | 1.95 GiB | 4.69 GiB | 0 | ✅（差一点 2 GB） |
| C13 | +6 GiB | 10 GiB | 200 | AOT 重载 | 240 s | 49.83 ms | **2.71 GiB** | 5.93 GiB | 0 | ✅（余量最好） |
| C5 | +16 GiB | 20.5 GiB | 60 | AOT 重载 | 230 s | 49.96 ms | 0.60 GiB | 3.41 GiB | 0 | ⚠️ 大 swap 也没用 |
| C10 | +6 GiB | 10 GiB | 60 | AOT 重载 | 230 s | 49.60 ms | 0.29 GiB | 3.35 GiB | 0 | ⚠️ 同上 |
| C1 | +16 GiB | 20.5 GiB | 60 | **现场编译** | 351 s | 50.14 ms | 0.885 GiB | 8.77 GiB | 0 | ⚠️ 基线 |
| C2/C16 | 0 | 4 GiB | 60 | **eager** | 160/130 s | 142.59 / 134.36 ms | 4.69 / 4.74 GiB | 0.45/0.55 GiB | 0 | ✅ 内存好、**decode 不合格** |

**C14 的完整相位表（达标配方的原始证据，`hostlogs/C14-memtrace.log`）**：

```
# samples=53/56 (truncated at arm stop +344s) span=344s
# boundaries: +0 P1_weight_load | +89 P2_compile | +115 P3_capture_kv | +174 P4_warmup | +222 P5_steady
phase              t0    t1   dur   n minAvail maxSwap peakAnon  =heap  unnamed  file maxCgGiB oomk
P1_weight_load     56    82    26   5    11140    1044    17911   9646     7584  8481    27.74    4
P2_compile         89   109    20   4     5155    1887    23065  12159    10855  6468    27.45    4
P3_capture_kv     115   167    52   9     2462    2658    25240  13168    11986  6586    27.63    4
P4_warmup         174   215    41   7     2403    4316    25473  13292    12185  6589    27.66    4   ← 峰值
P5_steady         222   326   104  17     3330    5527    24373  11835    12491  7003    27.00    4

## GLOBAL PEAK = LOWEST MemAvailable  t=+200s (P4_warmup 内)  2026-10-10T01:46:58+00:00
   MemAvailable=2403 MiB  MemFree=208 MiB Cached=2708 MiB  swap_used=3495 MiB  cg=27.66 GiB  oom_kill=4
   role      pid  RssAnon  [heap]  unnamed  RssFile  VmSwap  VmSize
   W     3382402     6084    3126     2961     1329     253   59145
   W     3382403     6066    3122     2940     1329     271   59058
   W     3382400     6011    3146     2861     1332     174   59009
   W     3382401     5952    3115     2848     1331     232   59023
   E     3380655     1360     783      575     1268    1544   29231
   TOT              25473   13292    12185     6589    2474

## STEADY (P5) medians over n=17  [+222s..+326s]
   MemAvailable=3716 MiB  swap_used=5509 MiB  SumRssAnon=23830 MiB (heap=11512 unnamed=12282)
   SumRssFile=6894 MiB  SumVmSwap=4636 MiB
   峰值 ΣRssAnon = 25473 MiB ; 稳态 = 23830 MiB ; peak − steady = 1643 MiB
# 注：整个 arm 的 min MemAvailable = 2403 MiB = 2.35 GiB，正好落在 P4_warmup（与 G1 的"峰值在 warmup"一致）
```

> 与 C1（现场编译、同样 4 桶）对照：**峰值 25473 vs 26395 MiB、稳态 23830 vs 24926 MiB** ——
> `swappiness=200` 把约 **1.3–1.6 GiB 从"常驻"搬到了"已换出"**（`SumVmSwap` 从 C1 的 7464 变成 4636？——
> 注意 C1 的 `SumVmSwap` 是**在 swappiness=60 下**由内核自行决定的换出量，两者的"换出"口径不可比；
> 可比的是 **`min MemAvailable`：0.885 → 2.35 GiB** 与 **swap 实际占用：8.77 → 5.40 GiB**）。

#### 3.12.1 G3 的三条结论

1. **口径 ② 达成（本步推荐配方）**：
   **`TP=4 + graph(PIECEWISE + `--cudagraph-capture-sizes 1 2 4 8`) + AOT 重载 + 附加 4 GiB swapfile（总 8 GiB）+ `vm.swappiness=200`**
   * ready **230 s**（比现场编译的 351 s 快 121 s）
   * decode 中位 **49.65 ms/token**（n=5，Mean 49.81 / P90 50.32 / P99 50.56）= 基线 49.66 的 **1.00×**，远优于 55 ms 门禁
   * **最低 MemAvailable 2.35 GiB（≥2 GB）**；swap 实际用 **5.40 GiB**（7.5 GiB 可用中的 72%）
   * `oom_kill` 零增量；`Available KV cache memory: 18.48 GiB`
2. **口径 ① 未达成**：完全不挂附加 swap（只用系统既有的 4 GiB `/swap.img`）+ `swappiness=200` 的最好成绩是
   **余量 1.78 GiB**（C15），而且此时 **swap 已经被 100% 用满**（4.00/4.00 GiB）——这是硬上限，
   **不可能再高**（没有可换出的空间了）。
3. **"哪几 GB 拿不掉"（具体化）**：把 C14 稳态的 23.8 GiB 常驻匿名拆开看（4 worker + EngineCore）：
   * **每 worker ≈5.7 GiB 常驻 + 约 1.2 GiB 已被换出**（swap 峰值 5.40 GiB / 4 worker ≈ 1.35 GiB/worker）
   * 其中 **`[heap]` ≈2.2–3.2 GiB/worker**（glibc 主 arena，与权重加载/反量化 staging 相关；
     **在 `torch.compile` 开始之前就已经是 ≈2.0 GiB/worker**，见 C1 的 P1 相位）
   * **unnamed mmap ≈2.9–4.0 GiB/worker**（§3.14 的形状），**这是唯一还没定位到分配者的那块**
   * **EngineCore/APIServer ≈1.5–3.1 GiB**
   * 已实测**拿不掉**的部分（都试过了）：图池（S14 §3.8.1）、sdot4 权重重排（S14 §3.8.5）、
     glibc arena（§3.5）、编译产物（§3.6）、MBT<2048（§3.8）、`max-num-seqs<256`（§3.11.2）、
     `combo_kernels=false`（§3.11.1）、HIP 分配器（§3.11.3）。
   ⇒ **要真正达到"不挂 swap 也留 2 GB"，唯一剩下的方向是把每 worker 的 unnamed mmap 削掉 ≈1.3 GiB**
     （定位它需要只读的 malloc 栈取证，见 §5 第 2 条）。

### 3.13 VMA 级形状（回答"unnamed 桶长什么样"，arm C13 稳态快照）

工具 `tools/dcu/s14c_anon_top.sh`（宿主只读，`sudo -n` 读 `/proc/<pid>/smaps`）。

```
=== worker pid=3358502 （graph / swappiness=200 / 稳态）===
VmSize: 60573204 kB   RssAnon: 5501476 kB   RssFile: 1363692 kB
-- 最大的匿名 VMA（addr range, size MiB, rss MiB, flags）--
1048576 1048576 7f5622800000-7f5662800000 rw-p        ← 1 个整 GiB（4 个 worker 都有，尺寸完全一致）
 262144  262144 7f587fe00000-7f588fe00000 rw-p        ← 256 MiB
  59748   59748 7f55d90b3000-7f55dcb0c000 rw-p        ← 58 MiB
  49536   49924 7f58c0000000-7f58c30c1000 rw-p  ┐
  49424   49904 7f58bc000000-7f58bf0bc000 rw-p  │ 地址间隔 **恰好 0x4000000 = 64 MiB**
  48868   49064 7f58b8000000-7f58bafea000 rw-p  │ （glibc `HEAP_MAX_SIZE` 对齐特征，
  ...                                            │  与 C13 另一 worker 的 7fef04000000/7fef08000000 同型）
-- 匿名 Rss 按 VMA 大小分桶（MiB / VMA 数）--
   1M-64M        1686 MiB  (267 VMAs)     ← **主桶**
   64M-1G         277 MiB  (3 VMAs)
   ge1G          1024 MiB  (1 VMAs)
   lt1M            12 MiB  (430 VMAs)
```

**判读**：

1. **4 个 worker 的形状几乎逐项相同**（1 GiB ×1、256 MiB ×1、~10 个 42–46 MiB、~265 个 ~4 MiB），
   ⇒ 这是**每进程固定的运行时结构**，不是"某个 rank 的权重分片"（TP=2 时的 `unnamed` 也是 3.83 GiB，§3.4 第 3 条）。
2. **那 ~10 个 42–46 MiB、地址间隔恰好 64 MiB 的区域是 glibc 次级 arena 的形状**；
   但 **`MALLOC_ARENA_MAX=1`（C4）并没有消掉它们**（C4 的 `1M-64M` 桶仍有 261 个 VMA / 852 MiB），
   而且总量不变 ⇒ **不能把它们简单归因成"arena 浪费"**。这需要下一步用只读栈取证确认。
3. `RssFile` ≈1.36 GiB/worker（graph 稳态）—— 与 S14 观察到的"TP=2 graph 时 1.27 GiB"同档；
   这些是 mmap 的 safetensors / 编译产物页，**OOM 时刻会被首批回收**（S14 §3.3.1 内核记录 `file-rss:0kB`）。

### 3.14 收尾与终态核验（2026-10-10T02:12Z）

**宿主还原（硬约束 4，前后原文在 `/mnt/game/s14c/hostlogs/swap-*.txt` 与 `sysctl-*.txt`）**：

```
$ sudo -n sysctl -w vm.swappiness=60
$ tools/dcu/s14c_swap.sh off /mnt/game/s14c/swap4 C18-restore   # 附件的 swap4 先 swapoff
$ tools/dcu/s14c_swap.sh rm  /mnt/game/s14c/swap4 C18-restore   # 再删除文件（写作用域内）
$ rm -f /mnt/game/s14c/swap6
$ sudo -n swapoff /mnt/game/swapfile                            # C1/C5~C9 用的 16 GiB 还原为 swapoff

$ cat /proc/swaps                       # 终态：与 2026-10-09T23:41Z 的开局逐字相同
Filename                                Type            Size            Used    Priority
/swap.img                               file            4194300         562808  -2

$ ls -la /mnt/game/swapfile             # 文件本体未被动过
-rw------- 1 root root 17179869184 Oct  4 15:19 /mnt/game/swapfile

$ for k in vm.swappiness vm.vfs_cache_pressure vm.min_free_kbytes; do ...; done   # 终态原文
vm.swappiness            = 60
vm.vfs_cache_pressure    = 100
vm.min_free_kbytes       = 67584
```

**引擎清理与终态**：

```
$ pgrep -af "vllm|torchrun"          → 只有本步自己的 bash 包装；无引擎
$ live VLLM = 0 ; zombies = 95       （开局 88，本步新增 7 个僵尸；state=Z，不占内存）
$ hy-smi --showmeminfo vram          → DCU[0..3] Used = 0
$ free -m                            → Mem 32009/3209/25523  Swap 4095/549/3546
$ docker exec z100-s14 bash -c 'cat /sys/fs/cgroup/memory.events'
low 0 / high 0 / max 0 / oom 0 / oom_kill 4      （基线 3 → 终态 4：本步新增 1 次，arm C4）
$ docker inspect z100-s14 --format 'Memory={{.HostConfig.Memory}} MemorySwap={{.HostConfig.MemorySwap}}'
Memory=0 MemorySwap=0                             （口径未变）
```

**ACS 全程部署态（硬约束：不 apply / 不 revert）**：

```
$ sudo -n python3 tools/dcu/dcu_p2p_acs.py show | tail -3
  0000:3a:00.0 0x2a0    0x0000    (none)
$ systemctl is-active dcu-p2p-acs
active
（unit active since 2026-10-09 18:29:46 UTC，早于本步全部实验 23:41Z 起 ⇒ 全程未切）
```

**未触碰的资产**：

```
$ docker inspect qwen38-vllm --format 'Status={{.State.Status}} ExitCode={{.State.ExitCode}} Memory={{.HostConfig.Memory}}'
Status=exited ExitCode=137 Memory=0        ← 与 S14 收尾时逐字相同
```

`VLLM_DCU_USE_INT8_SDOT4` 的仓库默认值仍为 OFF（全程 `docker exec -e` 注入，未改 `vllm/envs.py`）；
未改 DTK 库、未改 `z100-build*`、未启动生产容器、未执行任何自行编译的 HIP 可执行文件（硬约束 2）。

### 3.15 机读汇总与原始工件索引

`/mnt/game/s14c/hostlogs/S14c-summary.txt`（由 `/tmp/s14c_summarize.sh` 生成，列：
`ARM TP MODE READY DECODE minAvail peakAnon steadyAnon swapMax oomk tag`）：

```
ARM   TP   MODE   READY DECODE minAvail peakAnon steadyAnon swapMax   oomk  tag
C1    4    graph  1     50.14  0.86   26395    24926    8.56      3     c1_tp4_graph_swap16_livecompile
C2    4    eager  1     142.59 4.69   22797    22793    0.45      3     c2_tp4_eager_swap_swapimg4g_only
C3    2    graph  1     86.31  9.88   17777    17562    0.98      3     c3_tp2_graph_swap_swapimg4g_only
C4    4    graph  0            0.96   26824             4.00      4     c4_tp4_graph_arena1_swap_swapimg4g_only
C5    4    graph  1     49.96  0.60   26878    25678    3.33      4     c5_tp4_graph_swap16_AOTreload
C6    4    graph  1     50.01  0.24   27237    25882    6.74      4     c6_tp4_graph_swap16_mbt512
C7    4    graph  1     47.97  0.06   27387    25625    8.80      4     c7_tp4_graph_swap16_DEFAULT_cudagraph
C8    4    graph  1     50.87  0.20   27430    26200    4.94      4     c8_tp4_graph_swap16_nocombokernel
C9    4    graph  1     52.27  0.69   27034    26191    3.57      4     c9_tp4_graph_swap16_maxseqs64_warmcache
C10   4    graph  1     49.60  0.29   27362    25762    3.35      4     c10_tp4_graph_swap10_total_AOTreload
C11   4    graph  1     50.04  0.72   27297    26261    3.02      4     c11_tp4_graph_swap4only_AOTreload
C12   4    graph  1     49.94  1.95   25916    24835    4.69      4     c12_tp4_graph_swap10_AOT_swappiness150
C13   4    graph  1     49.83  2.71   24837    23016    5.93      4     c13_tp4_graph_swap10_AOT_swappiness200
C14   4    graph  1     49.65  2.35   25473    23830    5.40      4     c14_tp4_graph_swap8_AOT_swappiness200   ← ★ 推荐
C15   4    graph  1     50.30  1.78   25626    25149    4.00      4     c15_tp4_graph_swap4only_AOT_swappiness200
C16   4    eager  1     134.36 4.74   22803    22800    0.55      4     c16_tp4_eager_swap4only_swappiness60
C17   4    graph  1     50.09  2.75   25150    23321    6.38      4     c17_tp4_graph_swap8_AOT_sw200_noexpandable
```

`/mnt/game/s14c/hostlogs/summary.tsv`（`s14c_run_one.sh` 每臂追加一行，含
`NAME MODEL TP SW maxlen devices ready ready_s aot_load aot_recompile oom_kill timestamp tag`）。

原始工件（`/mnt/game/s14c/`）：

| 类别 | 文件 |
|---|---|
| 引擎日志 | `hostlogs/{C1..C17}.log` |
| 编排日志 | `hostlogs/orch-{C1..C17}.log` |
| **内存时间序列（G1 的核心证据）** | `hostlogs/{C1..C17}-memtrace.log`（5 s 采样，`S`/`P`/`M` 三类行；列定义见文件头） |
| 相位归因报告 | 由 `tools/dcu/s14c_phase_report.py` 现算（不落盘，命令见 §7） |
| VMA 级快照 | `hostlogs/anon-top-C5.txt`、`anon-top-C5b.txt`、`anon-top-C13.txt` |
| bench 原始 | `hostlogs/{C1..C17}-bench-decode.log` + `/mnt/game/s14c/bench-*-decode.log` |
| swap/sysctl 前后原文 | `hostlogs/swap-C1.txt`、`swap-C10.txt`、`swap-C11.txt`、`swap-C12.txt`、`swap-C14.txt`、`swap-C15.txt`、`swap-C18-restore.txt`、`sysctl-before.txt`、`sysctl-after-restore.txt` |
| cgroup | `cgroup-events-<ARM>-{before,after}.txt` |
| 机读汇总 | `hostlogs/S14c-summary.txt`、`hostlogs/summary.tsv` |
| 编译缓存（vintage） | `/mnt/game/s14c/cache-27b-{c1,c2,c3,c4,c6,c7,c8}` |
| 宿主开局留档 | `hostlogs/00-baseline-C1.txt` |
| 归档自身 | `docs/result/z100-14c-tp4-graph-memory-result.md` |

**回滚件**：`tools/dcu/s14c_*.sh` + `tools/dcu/s14c_smaps_agg.awk` + `tools/dcu/s14c_phase_report.py`
可整体删除（不影响任何运行）；`/mnt/game/s14c/**` 为新增产出，按纪律保留作原始证据。
**宿主核验**：`/proc/swaps` 只有 `/swap.img`、`vm.swappiness=60`、
附加 swapfile（`swap4`/`swap6`）已删除、`/mnt/game/swapfile` 仍为 swapoff —— 见 §3.14。

## 4. 门禁判定

| # | 门禁 | 判据 | 判定 | 证据 |
|---|---|---|---|---|
| **G1** | 相位归因 | 给出"宿主内存 × 时间"的**分相位**曲线（权重加载 / `torch.compile` / graph capture / warmup / 稳态），覆盖 **TP=4 graph（挂 swap）/ TP=2 graph（不挂 swap）/ TP=4 eager**；标出**峰值相位、比稳态高多少、由哪些进程构成**（`smaps_rollup` 的 `RssAnon`/`RssFile` + `[heap]`/`[anon]` 聚合）；回答"多出来的几 GB 是谁" | ✅ **通过** | §3.2 C1（TP=4 graph + 16 GiB swap，相位表 + 全局峰值构成 + 稳态中位数）、§3.3 C2（TP=4 eager）、§3.4 C3（TP=2 graph）、§3.7 三口径对照表与三条判读；`[heap]`/`unnamed` 的分桶正确性自检 §3.1；**结论**：峰值相位 = P4_warmup/P5_steady，`peak−steady` = 4~1469 MiB，构成 = 4 worker + EngineCore（逐进程 `RssAnon/[heap]/unnamed/RssFile/VmSwap` 都在表里） |
| **G2** | 压缩试验（逐条 ABBA） | 至少覆盖 `MALLOC_ARENA_MAX`；编译路径（`combo_kernels`/`benchmark_combo_kernel` **+ 现场编译 vs AOT 重载**）；`--max-num-batched-tokens` 与 `--max-num-seqs`；`--cudagraph-capture-sizes`；`PYTORCH_HIP_ALLOC_CONF` 变体；每条给"峰值变化 + decode 是否劣化" | ✅ **通过（7/7 因子实测，1 项按 S14 结论跳过）** | §3.11 因子汇总表：① `MALLOC_ARENA_MAX=1` §3.5；② AOT vs 现场编译 §3.6；③ MBT 512 §3.8；④ 默认图配置 §3.10；⑤ `inductor_compile_config` combo off §3.11；⑥ `max-num-seqs 64` §3.11；⑦ `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False` §3.11。`VLLM_DCU_USE_INT8_SDOT4=0` 按任务书"S14 已知无帮助、可跳过"**未重复**（§5 第 6 条）。**额外**发现并实测了两个真正的杠杆：**AOT 重载**（把 4 GiB swap 下的 OOM 变成可用）与 **`vm.swappiness`**（余量 +1.06~2.65 GiB，decode 零代价） |
| **G3** | 结论与配方 | 可执行启动配方；① 不挂 swap 且余量 ≥2 GB；② 小 swap ≤4 GiB 且余量 ≥2 GB；③ 否则给"最小可行 swap + 实测余量 + 风险"及"哪几 GB 拿不掉"；decode 不得劣于 55 ms/token | ✅ **通过（达成口径 ②）** | §3.13：**②达成** = 附加 4 GiB swapfile（总 8 GiB）+ `vm.swappiness=200` + AOT 重载 ⇒ **余量 2.35 GiB**、decode **49.65 ms**（C14）；①**未达成**（不挂附加 swap 时最好 1.78 GiB，C15，且 swap 100% 用满）。"哪几 GB 拿不掉"见 §3.13 第 3 条 |
| **G4** | 口径纪律 | 每行标 并行度/模式/maxlen/util/**n**/是否挂 swap **和 swap 大小**/cache vintage；ACS 全程部署态不切；不得混口径比较 | ✅ **通过** | §3.7 三口径表与 §3.11 因子表每行都带 TP/模式/图配置/swap 大小/vintage/n；bench 客户端与 S14 **逐字节相同**（复用 `tools/dcu/s14_bench.sh`，md5 `eced9511…`）；ACS §3.15（12×`0x0000`，unit `active since 18:29:46Z` 早于本步全部实验）；不混口径：S14 的 126.83/84.22/49.66 只作**参照列**并显式标注 |
| **G5** | 归档 | 本文件 8 小节齐全，含原始命令、原始日志路径、md5、回滚件、生产建议 | ✅ **通过** | 本文件；原始工件 `/mnt/game/s14c/**`（§3.16 索引）；md5 §3.16；回滚件 §3.16；生产建议 §3.13 与 §0 |

## 5. 未完成项 / 阻塞与原因

| # | 项 | 状态 | 原因 / 已有证据 |
|---|---|---|---|
| 1 | **推荐配方下的 conc8 / 8K prefill / 长稳态复测** | ❌ 未做 | 时间预算（≤4 h）用在了 17 个 arm 上；`C14` 只测了 `decode`（n=5）。峰值与 KV 分配都在 **warmup/稳态** 阶段（§3.2.1），判据指标是 `MemAvailable` 极值，理论上与并发无关，但**并发会改变页缓存压力与换出行为，必须复测**。这是本步最大的保留意见 |
| 2 | **unnamed mmap 桶的分配者身份** | 🔶 只到"形状"级 | VMA 级形状已测（§3.14：1 GiB + 256 MiB + ~10×42–46 MiB@64 MiB 步长 + ~265×~4 MiB/worker），但**没有做到"哪个子系统/哪个分配器"**。要钉死需要 `LD_PRELOAD` 钩子或 gdb 抓 `malloc` 调用栈 —— 属于"编译并运行自有原生代码"，与硬约束 2 的精神相悖，且时间不够。**建议下一步用 `malloc_info()` / `gdb -p` 只读抓栈，不要编译 LD_PRELOAD** |
| 3 | **`/mnt/game/s14c/swap4` 之外的更小附加 swap 边界** | 🔶 只测了两点 | 附加 0 GiB → 1.78 GiB 余量（swap 用满）；附加 4 GiB → 2.35 GiB；附加 6 GiB → 2.71 GiB。**2~3 GiB 附加是否够没测**（推测不够：附加 4 GiB 时已用 5.40 GiB，其中约 3.2 GiB 是系统 `/swap.img`，附加项只被用了约 2.2 GiB ⇒ 附加 2 GiB 可能刚好卡在边界）。要精确边界需再跑 2 个 arm |
| 4 | **`--max-num-seqs` 与 MBT 的交互** | ❌ 未做 | 只做了单因子（MBT 512 / max-num-seqs 64），都无收益（§3.8/§3.11）。交叉项未测，但两个单因子都是负收益，交叉项预期也为零 |
| 5 | **`ZONE_RECLAIM_MODE` / `min_free_kbytes` / cgroup 侧 `memory.high` 预压** | ❌ 未做 | `vm.swappiness` 已经吃到了"把冷匿名页换出去"的全部收益（余量 0.29→2.71 GiB）。`memory.high` 需要改容器配置（超写作用域），`min_free_kbytes` 是"人为抬 `MemAvailable`"的会计手段，不做 |
| 6 | `VLLM_DCU_USE_INT8_SDOT4=0` 对照 | 🚫 **按任务书跳过** | 任务书明确"已知无帮助，可跳过"；S14 §3.8.5 已用第 5 次 OOM（`anon-rss 7.28 GB` 同档）否证该假设 |
| 7 | **7 次 `global_oom` 之后宿主是否已处于降级状态** | 🔶 有旁证、无定论 | 本步 C1 的峰值（26395 MiB）与 S14 P4g2 的口径一致、decode 50.14 vs 49.46~49.86，**没有看到"越跑越差"**；但 `swaps_after_stop` 里 `/swap.img` 的既有占用从 450 MiB 涨到 562 MiB，属正常波动。建议 Lead 在长期跑之前重启一次宿主以清干净（本步**不能**做，会打断 `z100-*` 其它容器） |
| 8 | `docs/z100-todolist.md` / `docs/hygon_z100.md` / 性能方案 的文字更新 | 未做（**越权**） | 按纪律 6，这三份由 Lead 维护；本档提供证据供 Lead 更新 |

## 6. 对后续步骤的影响

1. **"TP=4 graph 需要 swap" 这句要改写**：准确的表述是
   **"TP=4 graph 的宿主匿名需求 ≈ 均衡态 25–26 GiB 常驻 + 3–9 GiB 换出，稳居 31.26 GiB RAM 之上；
   现场编译路径必须 ≥9 GiB swap 才稳，AOT 重载路径 4 GiB 就够活，但要 ≥2 GB 余量还得再加 4 GiB 并调 `vm.swappiness`。"**
2. **给生产方的可执行结论**（`serve_27b_tp4.sh` 的候选演进）：
   * 若继续用 eager（现状）：**内存**安全（余量 4.7 GiB、swap 0.5 GiB），但 decode **134–143 ms**；
   * 若要拿 2.4× 的 decode 收益：用 §3.13 的配方（graph + AOT 重载 + 4 GiB swapfile + `swappiness=200`）⇒ **49.65 ms，余量 2.35 GiB**；
   * **不要**用默认 51+2 桶（启动 1645 s、余量 66 MB，只换 4.3% decode）；
   * **不要**指望 `MALLOC_ARENA_MAX` / MBT / `max-num-seqs` / `combo_kernels` 省内存（§3.11）。
3. **对 S14b（graph 质量）的影响**：本步没有改变 S14b 的结论 ——
   **graph 的 token 流仍不得用于质量评测，质量参照仍是 eager**。本步只是把 graph 的**内存**做成生产可用。
   特别注意：本步的推荐配方含 **AOT 重载**，而 S14b 的 H-AOT 线索正是"现场编译 vs 重载"可能给出不同 token 流
   （§4.2 第 1/2 条）。⇒ **上生产前必须先用 S14b 的最小复现跑一次"现场编译 vs 重载"逐位对照**。
4. **给 Lead 的新结论（需要更新 todolist）**：
   * **`vm.swappiness` 是本步发现的唯一有效宿主级杠杆**（余量 +1.06~2.65 GiB，decode 零代价）——
     这条在任何既有文档里都没有；
   * **AOT 重载对内存无效、对时间有效**（ready 351→230 s，−34%）；
   * **`peak − steady` ≈ 0~1.5 GiB ⇒ 这类问题不能靠"削启动尖峰"解决**，只能削稳态需求或动用宿主策略。
5. **对 P6（内存/优化）的输入**：可省的是"分配需求"，方向 = **减少每 worker 的常驻匿名**；
   已排除：图池大小（S14）、sdot4 路径（S14）、glibc arena（本步）、编译产物（本步 AOT）、
   批大小/并发上限（本步 MBT、max-num-seqs）、inductor 组合核（本步）、HIP 分配器（本步）。
   **剩下唯一没被排除的方向 = §3.14 里那 ~265 个 ~4 MiB 的匿名 mmap 的归属**（需要只读的 malloc 栈取证）。

## 7. 复现方式

前置（每个 arm 开跑前留档；原文 `/mnt/game/s14c/hostlogs/00-baseline-C1.txt`）：

```bash
cd /home/lyh/z100-vllm
docker inspect z100-s14 --format '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}'   # 0 0
pgrep -af "vllm|torchrun"      # 应为空（VLLM::Worker_TP 的 comm 被改过，pgrep -a 看不到）
hy-smi --showmeminfo vram      # 4 卡全 0
free -g; cat /proc/swaps       # 只应有 4 GiB /swap.img
sudo -n python3 tools/dcu/dcu_p2p_acs.py show | tail -3   # 12×0x0000
```

```bash
# ---- G1 主臂 C1：TP=4 graph + 16 GiB swap（现场编译）----
sudo -n swapon /mnt/game/swapfile            # 收尾必须 swapoff
S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
  S14C_TAG="c1_tp4_graph_swap16_livecompile" S14C_MEMTRACE_IV=5 S14C_MEMTRACE_DUR=1800 S14C_HOLD_S=60 \
  READY_TIMEOUT=3600 S14_ENGINE_TIMEOUT=5400 \
  nohup tools/dcu/s14c_run_one.sh C1 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 \
  /mnt/game/s14c/cache-27b-c1 8430 262144 0,1,2,3 decode > /mnt/game/s14c/hostlogs/orch-C1.log 2>&1 &
tools/dcu/s14c_wait.sh C1 3600            # 等完 + 自动打印相位表/峰值构成/曲线

# ---- G1 C2：TP=4 eager（只有 4 GiB /swap.img）----
sudo -n swapoff /mnt/game/swapfile
S14C_CT=z100-s14 S14_EAGER=1 S14C_TAG="c2_tp4_eager_swap_swapimg4g_only" \
  S14C_MEMTRACE_IV=5 S14C_MEMTRACE_DUR=1200 S14C_HOLD_S=45 READY_TIMEOUT=1800 \
  nohup tools/dcu/s14c_run_one.sh C2 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 \
  /mnt/game/s14c/cache-27b-c2 8431 262144 0,1,2,3 decode > /mnt/game/s14c/hostlogs/orch-C2.log 2>&1 &

# ---- G1 C3：TP=2 graph（只有 4 GiB /swap.img）----
S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
  S14C_TAG="c3_tp2_graph_swap_swapimg4g_only" S14C_MEMTRACE_IV=5 S14C_MEMTRACE_DUR=1200 S14C_HOLD_S=45 \
  nohup tools/dcu/s14c_run_one.sh C3 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 2 1 \
  /mnt/game/s14c/cache-27b-c3 8432 262144 0,1 decode > /mnt/game/s14c/hostlogs/orch-C3.log 2>&1 &

# ---- G2 因子 1：MALLOC_ARENA_MAX=1（4 GiB swap，预期 OOM）----
S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" S14C_MALLOC_ARENA_MAX=1 \
  S14C_TAG="c4_tp4_graph_arena1_swap_swapimg4g_only" READY_TIMEOUT=1800 \
  nohup tools/dcu/s14c_run_one.sh C4 /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 4 1 \
  /mnt/game/s14c/cache-27b-c4 8433 262144 0,1,2,3 decode > /mnt/game/s14c/hostlogs/orch-C4.log 2>&1 &

# ---- G2 因子 2：AOT 重载（复用 C1 的 cache）----
S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
  S14C_TAG="c5_tp4_graph_swap16_AOTreload" \
  nohup tools/dcu/s14c_run_one.sh C5 ... 4 1 /mnt/game/s14c/cache-27b-c1 8434 262144 0,1,2,3 decode &

# ---- G2 因子 3：MBT 512 ----
S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" S14_MBT=512 \
  nohup tools/dcu/s14c_run_one.sh C6 ... 4 1 /mnt/game/s14c/cache-27b-c6 8435 262144 0,1,2,3 decode &

# ---- G2 因子 4：默认图配置（不传 S14_CG_*）----
S14C_CT=z100-s14 S14C_MEMTRACE_IV=10 S14C_MEMTRACE_DUR=2400 READY_TIMEOUT=3000 \
  nohup tools/dcu/s14c_run_one.sh C7 ... 4 1 /mnt/game/s14c/cache-27b-c7 8436 262144 0,1,2,3 decode &

# ---- G2 因子 5：inductor combo kernels off（会换 AOT key，必须新 cache）----
S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
  S14C_INDUCTOR_CFG='{"combo_kernels": false, "benchmark_combo_kernel": false}' \
  nohup tools/dcu/s14c_run_one.sh C8 ... 4 1 /mnt/game/s14c/cache-27b-c8 8437 262144 0,1,2,3 decode &

# ---- G2 因子 6：max-num-seqs 64 ----
S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" S14C_MAX_NUM_SEQS=64 \
  nohup tools/dcu/s14c_run_one.sh C9 ... 4 1 /mnt/game/s14c/cache-27b-c1 8438 262144 0,1,2,3 decode &

# ---- G3 swap 边界（附加 swapfile 用 tools/dcu/s14c_swap.sh 管理，写作用域 /mnt/game/s14c）----
tools/dcu/s14c_swap.sh off /mnt/game/swapfile C10
tools/dcu/s14c_swap.sh on  /mnt/game/s14c/swap6 6144 C10          # 总 10 GiB
S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" nohup tools/dcu/s14c_run_one.sh C10 ... \
  /mnt/game/s14c/cache-27b-c1 8439 262144 0,1,2,3 decode &                      # AOT 重载 + 10 GiB
tools/dcu/s14c_swap.sh off /mnt/game/s14c/swap6 C11
S14C_CT=z100-s14 S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" nohup tools/dcu/s14c_run_one.sh C11 ... \
  /mnt/game/s14c/cache-27b-c1 8440 262144 0,1,2,3 decode &                      # 只有 4 GiB，仍活
sudo sysctl -w vm.swappiness=150      # C12（10 GiB）
sudo sysctl -w vm.swappiness=200      # C13（10 GiB）
tools/dcu/s14c_swap.sh off /mnt/game/s14c/swap6 C14; tools/dcu/s14c_swap.sh on /mnt/game/s14c/swap4 4096 C14
S14C_CT=z100-s14 ... S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" nohup tools/dcu/s14c_run_one.sh C14 ... \
  /mnt/game/s14c/cache-27b-c1 8443 262144 0,1,2,3 decode &                      # ★ 达标配方（余量 2.35 GiB）
tools/dcu/s14c_swap.sh off /mnt/game/s14c/swap4 C15 && tools/dcu/s14c_swap.sh rm /mnt/game/s14c/swap4 C15
S14C_CT=z100-s14 ... nohup tools/dcu/s14c_run_one.sh C15 ... 4 1 /mnt/game/s14c/cache-27b-c1 8444 262144 0,1,2,3 decode &
# C16 eager 复测；C17 = 因子 7（PYTORCH_HIP_ALLOC_CONF=expandable_segments:False）
S14C_CT=z100-s14 S14C_HIP_ALLOC_CONF="expandable_segments:False" S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
  nohup tools/dcu/s14c_run_one.sh C17 ... 4 1 /mnt/game/s14c/cache-27b-c1 8446 262144 0,1,2,3 decode &

# ---- 分析（任意 arm）----
python3 tools/dcu/s14c_phase_report.py /mnt/game/s14c/hostlogs/C1-memtrace.log --curve-every 12
tools/dcu/s14c_anon_top.sh C13 12        # VMA 级形状 + 大小分桶

# ---- 收尾（硬约束 4：必须还原）----
sudo -n sysctl -w vm.swappiness=60
tools/dcu/s14c_swap.sh off /mnt/game/s14c/swap4 C18-restore; tools/dcu/s14c_swap.sh rm /mnt/game/s14c/swap4 C18-restore
sudo -n swapoff /mnt/game/swapfile
cat /proc/swaps; free -m; hy-smi --showmeminfo vram
```

**两个最容易踩的坑**（与 S14 相同，另有本步新增的一个）：

* `--cudagraph-capture-sizes` 是 `nargs='+'` ⇒ 必须展开成 `--cudagraph-capture-sizes 1 2 4 8`（脚本里用**数组**，不要用字符串拼接）；
* 杀 worker 只能 `pkill VLLM`；
* **新增**：`-cc.inductor_compile_config` 的值是 **JSON**，含空格时**必须用 bash 数组保持单参数**
  （`s14c_arm_env.sh` 已改用数组；用字符串拼接会被 word-splitting 拆成多个参数）。
  另外它会**改变 AOT key** ⇒ 三种编译期开关（`combo_kernels`/`benchmark_combo_kernel`、MBT、`max_num_seqs`）
  **都必须用新的 cache 目录**，vintage 才不会串。

## 8. 参考

* [`z100-14-27b-tp4-result.md`](z100-14-27b-tp4-result.md) §3.3（P4 OOM 时间线）/§3.4.1（P4e）/§3.7（内存构成）/§3.8（P4g1~P4g4）/§6（对后续步的影响）
* [`z100-14b-graph-vs-eager-result.md`](z100-14b-graph-vs-eager-result.md) §4（A/B 判定）/§4.2（**H-AOT** 与最小复现）/§6（质量口径 = eager）
* 内核原文：`sudo -n dmesg -T | grep -A3 "oom-kill:constraint"`（本步新增第 6 次 `global_oom`，§3.5）
* 原始工件：`/mnt/game/s14c/hostlogs/**`（§3.16 索引）
* 基线脚本（未改动，md5 见 §2）：`tools/dcu/s14_{arm_env,run_one,bench,monitor,stop_arm,smaps}.sh`
* ACS 工具：[`tools/dcu/dcu_p2p_acs.py`](../tools/dcu/dcu_p2p_acs.py)（本步只用 `show`）
