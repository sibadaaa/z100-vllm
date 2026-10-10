# S14e 可复现性边界：是 graph 通病，还是 S14c 省内存图配置特有？ — 执行结果

> 归档时间：2026-10-10（边做边写）
> 执行者：子 agent（S14e）
> 对应任务书：[`tools/dcu/s14e_taskbook.md`](../../tools/dcu/s14e_taskbook.md)
> 状态：🔄 进行中
> 原始数据：`/mnt/game/s14e/**`；日志 `/mnt/game/s14e/hostlogs/**`

## 0. 三句话

1. **答任务书的问题：不是省内存配置特有的。** 用**默认图配置**（`FULL_AND_PIECEWISE`、51+2 桶、
   不省内存）在 TP=2/maxlen 4096 上做**三次独立 AOT 重载**，三跑也没能全一致：
   第 1、3 跑落在变体 A，第 2 跑落在变体 B，**A vs B 差 403/512 个位置**；
   同上下文 `prompt_logprobs` **mean 1.0–1.15 nat / max 7.9 nat**——与 S14d 在省内存配置下测到的
   1.24–1.45 / 7.83–11.57 nat **同一量级**。⇒ **"graph 运行间不可复现"是 graph 的普遍性质。**
2. **但它不是一个"必然每次都不一样"的连续漂移，而是"两个离散变体、每次启动抽一个"。**
   默认配置下 6 次独立重载（本步 3 次 + S11m 3 次）只落在 **2 个变体**上，
   两两逐位一致率 **7/15 ≈ 47%**（本容器内 1/3 对）；同变体内部只剩 **1e-2 ~ 6e-2 nat** 抖动，
   而这点抖动在 128 步贪心解码里**仍会翻 token**（G1a vs G1c：11/12 prompt 相同、1 个在第 70 位分叉）。
   S11m 的"3/3 一致"是**真实但不稳健**的——本步的 G1b 与它逐位相同（0/512），G1a/G1c 则不是。
3. **生产结论**：**默认图配置不是"可复现性的解药"**——它比省内存配置慢 **9×**
   （ready **1542 s vs 170–180 s**；S14c 在 TP=4/262144 上是 **1645 s vs 351 s**、余量 66 MiB），
   却**照样不可复现**。要 token 流可复现只能 **eager**（S14d：12/12 逐位一致、Δ=0.00000 nat、
   131.78–133.83 ms），且要注意 eager 的**并发批组合**仍会确定性地改变数值；
   否则用省内存图配置（ready 170–180 s）并**接受每次重启换输出**（mean ≈1.1 nat / max ≈8–9 nat）。

## 1. 环境留档（开跑前）

宿主 `date`：`Sat Oct 10 03:33:31 AM UTC 2026`

```
$ pgrep -af "vllm|torchrun"
(无引擎在跑；唯一匹配是查询命令自身)
$ hy-smi --showmeminfo vram
DCU[0..3] vram Total 32752 MiB / Used 0 MiB   （四卡全空）
$ free -g
               total        used        free      shared  buff/cache   available
Mem:              31           3          22           0           5          27
Swap:              3           0           3
$ cat /proc/swaps
Filename                                Type            Size            Used            Priority
/swap.img                               file            4194300         519704          -2
$ cat /proc/sys/vm/swappiness      # `sysctl` 二进制不在宿主 PATH 里，读 procfs 等价
60
```

* 容器 `z100-s14`：`HostConfig.Memory=0`（无 cgroup 上限）；挂载 `/home/lyh/z100-vllm -> /src (ro)`、
  `/mnt/game`（读写共享）、`/opt/hyhal -> /opt/hyhal (ro)`。
* **本步不需要动 swap / swappiness**：G1/G2/G5 都是 TP=2 + maxlen 4096，S11m 当年即在该内存形态下完成；
  G3/G4（maxlen 262144）未执行（见 §7）。收尾仍复核 `/proc/swaps` 与 `vm.swappiness` 原文（§6）。
* 代码 vintage：`vllm/compilation/decorators.py` mtime `2026-10-09 06:23:01`、
  `vllm/compilation/caching.py` `2026-10-09 06:27:43`；被 `git status` 标为修改的全部源文件 mtime **均早于**
  该时刻 ⇒ 本步代码 vintage 与 S11m 产出 AOT 产物时**一致**（这一点被实测证实：见 §3.1 的 `Directly load AOT`）。
* **ACS（部署态，未切换）**：`sudo python3 tools/dcu/dcu_p2p_acs.py`（只读）——
  父链 12 个带 ACS 的桥 `ACS_Ctl` **全部 `0x0000`**（ReqRedir/CmpltRedir 均未置位）。
  本步全程不改 ACS，收尾复核同值（§6）。
* **环境漂移：无（一次自查记录，留给后人）**。G1a 前 8 个桶的 tqdm 速率显示
  `1/51 [00:21, 21.47s/it]` … `8/51 [02:49, 21.30s/it]`，一度让人以为比 S11m 的 ≈13.6 s/桶慢 1.6×；
  但 **轮次完成行的实测是 `Graph capturing finished in 684 secs, took 1.67 GiB`（TP0/TP1 各一条）**，
  与 S11m 当年的 **693 s / 700 s 基本一致** ⇒ 那 21 s/it 是 tqdm 被**前几个桶的冷启动+首次 kernel
  load 拖高的早期均值**，不是环境变慢。S11m 的结论所依赖的环境条件在本步仍然成立。

## 2. 设计、脚本与启动命令

### 2.1 复用的资产（只复制、不改）

| 本步新脚本 | 来源 | 改动 |
|---|---|---|
| `tools/dcu/s14e_client.py` | `s14d_client.py` | 无（12 prompt 逐位 + `prompt_logprobs` 电池） |
| `tools/dcu/s14e_compare.py` | `s14d_compare.py` | 无 |
| `tools/dcu/s14e_matrix.py` | `s14d_matrix.py` | 无 |
| `tools/dcu/s14e_run_one.sh` | `s14_run_one.sh` | 产物根改 `/mnt/game/s14e`；新增 `e14` 任务调 `s14e_client.py`；额外记两轮 graph capture 墙钟 |

容器内启动器用**现成的** `tools/dcu/s14_arm_env.sh`（未改），通过 `S14_CG_MODE` / `S14_CG_SIZES` 控制图配置。

### 2.2 关键设计决定：G1 用 2 次真重载而不是 3 次启动

S11m §3.4 实测：**默认图配置（`FULL_AND_PIECEWISE`，51 个 capture size）每次启动都要重做两轮
graph capture，`=1` 冷启动 1663 s / 热启动 1553–1573 s，其中 1383–1395 s（≈23 min）是 capture**
⇒ 每一次默认图配置的启动都是 **≈26 min**，与 cache 命中与否几乎无关。

因此"空 cache 冷起 + 2 次热启动"的朴素三跑协议要 **≈78 min**，单是 G1 就吃掉 65% 预算。
本步改用**等价的 2 跑协议**：把 S11m 那次已预热好的 cache（`/mnt/game/vllm-cache-11m/one`，
代码 vintage 与本步一致）**复制**到 `/mnt/game/s14e/g1` 作为"已预热的 cache"，
于是 **第 1 跑与第 2 跑都是真正的 AOT 重载**，两者逐位对照就是任务书要的
"两次独立 AOT 重载"（G1 = 2 跑 ≈ 53 min，省下 26 min）。
为可核查，启动日志里的 `Directly load AOT compilation from path` 行已留档（§3.1）。

> 口径诚实标注：G1 用的 AOT 产物 vintage = S11m（2026-10-09）写下、代码 vintage 相同的那一份；
> 这**正是**"同 cache 的两次独立重载"，与 S11m 的 S2/S3/S4 同源。

### 2.2b 关键设计决定 2：G1 同时跑**两套仪器**

G1 要同时回答两件事：
1. "两次独立 AOT 重载是否逐位一致"（本步的普遍问题）——用 S14d 的 12-prompt 仪器即可；
2. "**是否复现 S11m 的 3/3**"——S11m 当年用的是 `tools/dcu/s11l_flip_stats.py`
   （8 prompt × **64** token、**不发** `prompt_logprobs`）。

之前实测确认：`s11l_flip_stats.py` 的 `PROMPTS` 与 `s14e_client.py` 的 `FLIP` 元组**8 条逐字相同**，
且两者的 JSONL schema 都被 `tools/dcu/s11m_flip_matrix.py` 读得懂（`label` / `prompt_id` / `token_ids` / `rows[].logprob`），
因此本步在同一个引擎里**依序**跑两套仪器（`S14E_TASKS=flip,e14`）：

* `flip` **先跑** ⇒ 与 S11m 的 S2/S3/S4 条件一致（全新引擎、第一个电池、64 token），
  可直接用 `s11m_flip_matrix.py` 把 **G1a-flip / G1b-flip / S11m-S2 / S11m-S3 / S11m-S4 五份产物放一起做簇判定**；
* `e14` 后跑（12 prompt × 128 token + `prompt_logprobs`）⇒ 与 S14d 同口径，用于跨配置比较。

两跑的 task 顺序**完全相同**，所以 G1a vs G1b 的任何差异都不可能是"请求顺序/前缀缓存"造成的。

### 2.3 逐格启动命令（宿主侧）

统一前缀：`S14_UTIL` / `S14_CG_MODE` / `S14_CG_SIZES` / `S14_DEVICES` / `S14_CACHE` 显式给出；
`SW`（`VLLM_DCU_USE_INT8_SDOT4`）为第 4 个位置参数。

```bash
# G1 (TP=2 / maxlen 4096 / 默认图配置) —— 两跑，两次独立 AOT 重载
S14_UTIL=0.9 S14E_TASKS="flip,e14" tools/dcu/s14e_run_one.sh G1a \
    /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 2 1 /mnt/game/s14e/g1 8621 4096 0,1
S14_UTIL=0.9 S14E_TASKS="flip,e14" tools/dcu/s14e_run_one.sh G1b \
    /mnt/game/models/Qwen3.8-27B-Channel-INT8-w8a8 2 1 /mnt/game/s14e/g1 8621 4096 0,1
```

（`S14_UTIL=0.9` 是**故意**对齐 S11m 的 `--gpu-memory-utilization 0.9`，让 G1 成为 S11m 的严格复现；
图配置不传任何 `-cc` ⇒ 生效值由日志实证为
`cudagraph_mode=FULL_AND_PIECEWISE`、`cudagraph_capture_sizes=[1,2,4,8,16,…,512]`（51 个）——
与 S11m 的 "51+2 桶" 一致。）

```bash
# G2 (TP=2 / maxlen 4096 / PIECEWISE + capture-sizes 1 2 4 8) —— 与 G1 只差图配置
# 3 跑：第 1 跑空 cache（冷编译+填 cache），第 2/3 跑是同 cache 的两次独立 AOT 重载
S14_UTIL=0.9 S14E_TASKS="flip,e14" S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
  tools/dcu/s14e_drive.sh G2 2 1 /mnt/game/s14e/g2 8622 4096 0,1 3

# G5 = G2 但 SW=0（关掉我们自己的 sdot4 kernel）
S14_UTIL=0.9 S14E_TASKS="flip,e14" S14_CG_MODE=PIECEWISE S14_CG_SIZES="1 2 4 8" \
  tools/dcu/s14e_drive.sh G5 2 0 /mnt/game/s14e/g5 8623 4096 0,1 3

# G3 (TP=2 / 262144 / 默认) 与 G4 (TP=4 / 262144 / 默认) —— 见 §7（预算不足，未执行）
```

`s14e_drive.sh` 会把"两次热启动"之外的**所有** pairwise 组合都算出来（G2a/G2b/G2c 三跑 ⇒ 3 对），
其中 **G2b-vs-G2c 才是与 G1a-vs-G1b 同口径的"两次独立 AOT 重载"**；
G2a-vs-G2b/G2c 是"现场编译 vs 重载"（H-AOT 口径），一并给出以备交叉核对。

## 3. 逐格结果

### 3.1 G1 — TP=2 / maxlen 4096 / **默认图配置**（复现 S11m 的 3/3？）

#### 3.1.1 启动事实（证明"这两次都是 AOT 重载"）

| run | 启动(UTC) | ready 墙钟 | `Directly load AOT` | `ind_hit` | `torch.compile took` | 两轮 PIECEWISE capture | `init engine` | KV cache | `oom_kill` Δ |
|---|---|---|---|---|---|---|---|---|---|
| **G1a** | 03:39:07 | **1542 s**（≈25.7 min） | **2**（rank 0/1，artifact `2b4e7331…`） | 1 | 19.55 s | **684 s**（轮 1）+ **696 s**（轮 2），另 FULL 只 2 桶 | **1433.33 s**（compilation 20.19 s） | 11.54 GiB | **0**（4 → 4） |
| **G1b** | 04:08:17 | **1542 s**（≈25.7 min） | **2**（同一 artifact） | 1 | 19.22 s | **683 s** + **695 s** | **1430.86 s**（compilation 20.41 s） | 11.54 GiB | **0**（4 → 4） |

关键日志原文（G1a，`/mnt/game/s14e/hostlogs/G1a.log`）：

```
(Worker_TP0) INFO 10-10 11:40:45 [decorators.py:321] Directly load AOT compilation from path
   /mnt/game/s14e/g1/torch_compile_cache/torch_aot_compile/2b4e7331b5917485697b9af412969a813712449805d4e9a957ccf454ca6bfc15/rank_0_0/model
(Worker_TP0) INFO 10-10 11:40:45 [monitor.py:53] torch.compile took 19.55 s in total
(Worker_TP1) INFO 10-10 11:40:45 [decorators.py:321] Directly load AOT compilation from path .../rank_1_0/model
(Worker_TP0/TP1) INFO 10-10 11:52:33 [model_runner.py:1060] Graph capturing finished in 684 secs, took 1.67 GiB
(Worker_TP0/TP1) INFO 10-10 12:04:15 [model_runner.py:1060] Graph capturing finished in 696 secs, took 1.48 GiB
(APIServer) INFO 10-10 12:04:17 [core.py:381] init engine (profile, create kv cache, warmup model) took 1433.33 s (compilation: 20.19 s)
(APIServer) INFO:     Application startup complete.
```

**两跑确实是"同 cache 的两次独立 AOT 重载"**：两次都在同一个 artifact 目录上打印
`Directly load AOT compilation from path`（每跑 2 条 = 两个 rank），`ind_hit=1`、无 `Compiling a graph`。
产物未被重写：`md5sum` = `911769244fccaac536a3cf0ff699d736`（rank_0_0）/
`71f6a3c2940ea847e31a5c2fbf5edff7`（rank_1_0），且 `rank_0_0/model` 的 md5 与 S11m 归档里记的
`911769244fccaac536a3cf0ff699d736` **逐字符相同** ⇒ 我复制的就是 S11m 那一份产物（见
`/mnt/game/s14e/artifact-md5-after-G1a.txt`）。

#### 3.1.2 仪器 A：`s11l_flip_stats.py`（= S11m 当年的仪器，8 prompt × 64 token）

**（a）先做 S11m 基线核对**（只放 S11m 自己的三份产物）：

```
$ python3 tools/dcu/s11m_flip_matrix.py /mnt/game/s11m/flip-S2sw1hit.jsonl \
      /mnt/game/s11m/flip-S3sw1hit.jsonl /mnt/game/s11m/flip-S4sw1hit.jsonl
[PAIR] S2sw1hit vs S3sw1hit: 0/512 positions differ, max_matched_delta=0.000048
[PAIR] S2sw1hit vs S4sw1hit: 0/512 positions differ, max_matched_delta=0.000000
# ALL_IDENTICAL=True
```

⇒ S11m 的 3/3 在本步的工具链上**逐位复现**（0/512、max Δ 4.8e-5）——工具口径无问题。

**（b）关键发现：G1a 重载了 S11m 的同一份产物，却与 S2/S3/S4 不是同一串 token**

```
$ python3 tools/dcu/s11m_flip_matrix.py <S2> <S3> <S4> /mnt/game/s14/flip-P2.jsonl \
      /mnt/game/s14/flip-P2e.jsonl /mnt/game/s14/flip-P4e.jsonl /mnt/game/s14e/flip-G1a.jsonl
[PAIR] S2sw1hit vs S3sw1hit:  0/512  max Δ 0.000048
[PAIR] S2sw1hit vs S4sw1hit:  0/512  max Δ 0.000000
[PAIR] S2sw1hit vs P2:      410/512  max Δ 0.911751     # S14 的 TP=2/262144/默认图
[PAIR] S2sw1hit vs P2e:     393/504  max Δ 0.972388     # S14 的 TP=2/262144/eager
[PAIR] S2sw1hit vs P4e:     448/496  max Δ 0.688486
[PAIR] S2sw1hit vs G1a:     403/512  max Δ 0.957769     # §本步 G1a：同一 artifact、同 flags
[PAIR] G1a vs P2:           382/512  max Δ 1.103523
[PAIR] G1a vs P2e:          313/504  max Δ 1.005680
```

**必须在归档里写清的判读**：G1a 与 S2/S3/S4 **加载的是同一个 artifact 目录、同一份产物字节**
（§3.1.1 的 md5），flags 也逐字相同（TP=2 / maxlen 4096 / util 0.9 / MBT 2048 / prefix cache / mamba align /
`VLLM_DCU_USE_INT8_SDOT4=1` / `VLLM_DCU_SDOT4_INNER` 默认值就是 `sdot4`——S11m 显式设为 `sdot4`，我未设，
两者等价，已查 `vllm/model_executor/kernels/linear/scaled_mm/gfx906_sdot4.py:85`
`os.environ.get("VLLM_DCU_SDOT4_INNER", "sdot4")`），
**却给出了 403/512 不同的 token 串**（差异从第 0 个生成 token 就开始）。

也就是说：**S11m 的"加载 AOT 产物这条路是确定的"这条结论有一个它当年没暴露的前置条件——
必须在同一个容器/同一个时间窗内**。跨容器（z100-s9 → z100-s14）、跨天（10-09 → 10-10）、
跨 `TRITON_CACHE_DIR` 的累积状态之后，**同一份产物重载出来的变体变了**。
这不直接回答本步的问题（本步问的是"同一环境内两次重载是否一致"），但它是任务书
"若不一致先排查环境/vintage 漂移"那一条的直接答案：**漂移是真实存在的，且足以换掉整个 token 变体**。

> 附注（同量级对照，说明"变体差异"的量级本来就很大）：S11m 自己记录的 10 次启动里就有
> ≥6 个变体，且**同开关跨引擎**的差异（arm2 vs Z1，都是 `=0`：421/512）与
> **跨开关**的差异（arm1 vs arm2：428/512）一样大。所以 403/512 这个数**不能**单独用来
> 判定"可复现/不可复现"，只能用来判定"属于不同变体"。

#### 3.1.3 仪器 B：`s14e_client.py`（= S14d 的仪器，12 prompt × 128 token + 同上下文 `prompt_logprobs`）

```
$ python3 tools/dcu/s14e_compare.py /mnt/game/s14e/probe-G1a.jsonl /mnt/game/s14e/probe-G1b.jsonl \
      --label-a G1a --label-b G1b --csv /mnt/game/s14e/cmp-G1a-vs-G1b.csv
prompt               nA   nB  ids=  first      maxd     gapA     gapB | plp_n  plp_max  plp_mean  flips
zh_hz               128  128 False     11  0.259400    0.641    0.062 |    56  7.92478   1.08707     11
zh_ml               128  128 False      7  0.100191    0.141    0.125 |    56  7.92478   1.11849     12
zh_sort             128  128 False      0  0.000000    1.375    0.125 |    60  7.92478   1.05289     13
zh_sky              128  128 False     13  0.111415    0.203    0.062 |    59  7.92478   1.03017     12
en_fr               128  128 False     13  0.152181    0.016    0.344 |    58  7.92478   1.15447     12
en_sea              128  128 False     10  0.120653    0.516    0.391 |    59  7.92478   1.05886     14
en_index            128  128 False     37  0.208091    0.234    0.172 |    59  7.92478   1.10909     14
en_math             128  128 False      5  0.083814    0.406    1.109 |    65  7.92478   1.10353     13
p2_zh_slip          128  128  True   None  0.000000        -        - |    77  0.00000   0.00000      0
p2_en_batch         128  128  True   None  0.000000        -        - |    73  0.00001   0.00000      0
p2_zh_code          128  128 False     70  0.356153    1.297    0.438 |    75  0.00001   0.00000      0
s14_greedy_long     128  128  True   None  0.129301        -        - |   125  0.00781   0.00000      0
-------------------------------------------------------------------------------------------------------
# prompts compared            : 12
# prompts token-identical     : 3/12   (inconsistency rate = 9/12 = 0.7500)
# worst |dlogprob| at first divergence: 0.356153 nat  (p2_zh_code)
```

#### 3.1.4 与 S14d 的对照（同仪器 B，不同格）

| 格 | TP | maxlen | 图配置 | 同 cache 两次 AOT 重载的逐位一致率 | 同上下文 `prompt_logprobs` mean / max \|Δ\| |
|---|---|---|---|---|---|
| S14d A3 vs A4 | 4 | 262144 | PIECEWISE + 1 2 4 8 | **0/12** | 1.24–1.45 / 7.83–11.57 nat |
| **G1a vs G1b** | 2 | 4096 | **默认**（FULL_AND_PIECEWISE 51+2） | **3/12**（9/12 不一致） | **1.03–1.15 / 7.92 nat** |

> 口径差异（必须标注，否则会被误读）：S14d 的 e14 是**空 prefix cache**下第一个任务；
> 本步 G1 为了让 `flip` 与 S11m 严格同条件，`flip` 先跑，于是 G1 的 e14 有 8 个 prompt 命中 prefix cache。
> 因此 **G1 的 `mean|Δ|` 不能直接与 S14d 的数比大小**；可以比的是"**同一格内两次重载的一致率**"。

#### 3.1.5 G1 小结（**本步最关键的一段**）

1. **启动阶段是完全可复现的**：两跑 ready 都是 **1542 s**，`init engine` **1433.33 / 1430.86 s**，
   两轮 capture **684+696 / 683+695 s**，KV cache 都是 11.54 GiB，`oom_kill` Δ 都是 0
   ⇒ 两跑的**编译/加载/捕获路径完全同构**，不存在"A 跑多编了一次"这类解释。
2. **但推理结果不可复现**：
   * 仪器 A（S11m 口径）：**G1a vs G1b = 403/512 位置不同**；
   * 仪器 B（S14d 口径）：**12 个 prompt 里 9 个 token 流不同**，同上下文 `mean|Δ|` **1.03–1.15 nat**、
     `max|Δ|` **7.92 nat**、argmax flips 11–14 —— **与 S14d 在 TP=4/262144/省内存配置下测到的量级同档**
     （S14d：0/12、1.24–1.45 nat、7.83–11.57 nat）。
3. **S11m 的"3/3"被复现了，但它只覆盖了两个变体中的一个**：
   `S2sw1hit vs G1b = 0/512（max Δ 8.2e-3）` ⇒ **G1b 与 S11m 的 S2/S3/S4 属于同一变体**；
   而 **G1a 与它们差 403/512**。也就是说，
   **同一份 artifact、同一组 flags、同一个容器，两次独立 AOT 重载也会抽到不同变体**。
   S11m 的"加载 AOT 产物这条路是确定的"这句话，实际成立范围是
   "**同一个容器 + 同一个时间窗内的连续几次启动**"，不是一条普遍规律。
4. **对任务书那句话的直接回答**：G1 用**默认图配置**（不省内存、51+2 桶），**也没有**拿到两次重载逐位一致
   ⇒ **"graph 运行间不可复现"不是 S14c 省内存配置特有的性质**。
   （定量上仍有一个需要 G2 才能说清的层次：**"会不会分叉"是 graph 普遍的，"分叉概率"是否随配置变化**。）

### 3.2 G2 — TP=2 / maxlen 4096 / `PIECEWISE` + `1 2 4 8`（与 G1 只差图配置）

#### 3.2.1 生效配置与启动事实

日志实证（`/mnt/game/s14e/hostlogs/G2a.log`）：
`'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>, 'cudagraph_capture_sizes': [1, 2, 4, 8]`
——与 S14c/S14d 的省内存配置**逐字相同**，是本步唯一与 G1 不同的因子。

| run | 相位 | 启动(UTC) | ready | `aot_load` | `ind_hit` | `torch.compile took` | `init engine` |
|---|---|---|---|---|---|---|---|
| G2a | 空 cache（现场编译 + 填 cache） | 04:37:48 | **270 s** | 0 | 0 | 91.36 s | 133.80 s |
| G2b | 同 cache **AOT 重载 #1** | 04:45:42 | **170 s** | **2** | 1 | 19.50 s | 57.78 s |
| G2c | 同 cache **AOT 重载 #2** | 04:52:03 | **180 s** | **2** | 1 | 19.92 s | 58.58 s |

**同一格的启动成本对比（顺带回答"省内存配置值多少"）**：默认图配置 **1542 s** vs 省内存配置
**170–180 s** ⇒ 省内存配置每次启动**省 ≈22.8 min**。这与 S14c 在 TP=4/262144 上测到的
1645 s vs 351 s 的结论一致。

#### 3.2.2 逐对对照（三跑 ⇒ 三对；**G2b vs G2c 才是与 G1a vs G1b 同口径的那一对**）

仪器 A（`s11l_flip_stats.py`，8 prompt × 64 token）：

```
[PAIR] G2a vs G2b: 341/490 positions differ, max_matched_delta=1.401050   # 现场编译 vs 重载
[PAIR] G2a vs G2c: 316/490 positions differ, max_matched_delta=1.400876   # 现场编译 vs 重载
[PAIR] G2b vs G2c: 363/512 positions differ, max_matched_delta=1.543604   # ★ 重载 vs 重载
```

仪器 B（`s14e_client.py`，12 prompt × 128 token + 同上下文 `prompt_logprobs`）：

| 对 | 类型 | token 流逐位一致 | 同上下文 mean \|Δ\| (nat) | max \|Δ\| (nat) | argmax flips |
|---|---|---|---|---|---|
| G2a vs G2b | 现场编译 vs 重载 | **2/12** | 0–1.098 | 9.489 | 0–14 |
| G2a vs G2c | 现场编译 vs 重载 | **2/12** | 0–1.231 | 9.489 | 0–13 |
| **G2b vs G2c** | ★ **重载 vs 重载** | **4/12** | **0–1.186**（有差异的 prompt：**1.00–1.19**） | **8.742** | 0–18 |

★ 对的逐 prompt 表（`/mnt/game/s14e/cmp-G2b-vs-G2c.txt`）：

```
zh_hz     False  11  | plp 56  8.74219  1.18596  13
zh_ml     False  17  | plp 56  8.70182  1.09235  14
zh_sort   False   0  | plp 60  8.70182  1.00420  15
zh_sky    False  13  | plp 59  8.70182  1.07789  15
en_fr     False  38  | plp 58  8.70182  1.08678  15
en_sea    False  40  | plp 59  8.70182  1.01665  14
en_index  False   4  | plp 59  8.70182  1.05131  16
en_math   False   5  | plp 65  8.70182  1.11847  18
p2_zh_slip  True  None | 0.00000
p2_en_batch True  None | 0.00000
p2_zh_code  True  None | 0.00000
s14_greedy_long True None | 0.00000（生成流 token 相同，但生成 token 的 logprob 差 0.187 nat）
```

#### 3.2.3 G2 小结

* 省内存图配置下，**同 cache 的两次独立 AOT 重载同样不逐位一致**（flip 363/512 不同；
  e14 4/12 相同、同上下文 mean 1.00–1.19 nat / max 8.70 nat）——
  与 S14d 在 TP=4/262144 上测到的 0/12、1.24–1.45 nat **同一量级**。
* "现场编译 vs 重载"（G2a vs G2b/G2c：2/12）比"重载 vs 重载"（4/12）**更差**，
  再次与 S14d 的 H-AOT 否证一致：**差异的主因不是"AOT 来源"，而是"任意两次运行"**。
* 因此 G2 与 G1 在"是否可复现"上**没有区别**（都不行），
  在"分叉幅度"上也**没有区别**（同上下文 mean 都在 1.0–1.2 nat 窄带内）。

### 3.3 G1c — 默认图配置的**第三次**独立重载（任务书要求：G1 不一致则加到 3 次）

G1c = 与 G1a/G1b **逐字相同**的第三次启动（同容器、同 cache、同 artifact、同 flags、同 task 顺序），
启动于 `04:59:40Z`，ready **1553 s**、`aot_load=2`、`ind_hit=1`、`init engine` **1435.33 s**（compilation 19.78 s）、
两轮 capture **685 s + 697 s**、`oom_kill` Δ = 0 —— 与 G1a/G1b 完全同构（三跑 ready：1542 / 1542 / 1553 s）。

#### 3.3.1 **关键结果：默认图配置下只有两个离散变体，每次启动随机抽一个**

| 对 | 仪器 A（flip，512 位置） | 仪器 B（e14，12 prompt） | 同上下文 plp_max |
|---|---|---|---|
| **G1a vs G1c** | **0/512 不同**（max Δ **0.0603**） | **11/12 逐位相同** | **0.008 nat** |
| G1a vs G1b | 403/512 不同（max Δ 0.9578） | 3/12 相同 | 7.925 nat |
| G1b vs G1c | 403/512 不同（max Δ 0.9578） | 4/12 相同 | 7.925 nat |
| S2sw1hit vs G1b | **0/512 不同**（max Δ 0.0082） | （S11m 未跑 e14） | — |
| S2sw1hit vs G1c | 403/512 不同 | — | — |

⇒ **本步 3 次独立重载落在 2 个变体上**：`{G1a, G1c}` 与 `{G1b, S2, S3, S4}`。
两个变体之间的差异是**粗粒度**的（403/512 位置、≈1 nat）；**同一变体内部**只剩
**1e-2 ~ 6e-2 nat 的浮点抖动**，而这点抖动在 128 步贪心解码里**偶尔仍会翻一个 token**
（G1a vs G1c 的 `p2_zh_code` 在第 70 位分叉，尽管它们的 `prompt_logprobs` 最大只差 1e-5）。

#### 3.3.2 由 3 跑得到的率估计

* **本容器内**（3 跑 ⇒ 3 对）：flip 口径 **1/3 对**逐位相同；e14 口径 **18/36 个 prompt-pair** 相同（**50%**）。
* **与 S11m 的 3 跑合并**（6 跑 ⇒ 15 对，变体分布 A=2、B=4）：**7/15 = 47%** 的对逐位相同。
* 变体是**离散两态**而不是连续漂移 ⇒ "两次重启是否一致"近似是**一次 50% 的抛硬币**。
  S11m 当年连续 3 次都抽到 B（概率 ≈12.5%）属于**可能的运气**，不能当作"重载路径确定"的证据。

### 3.4 G5 — G2 + `VLLM_DCU_USE_INT8_SDOT4=0`

（未执行，原因见 §7）

### 3.5 G3 / G4 — 未执行

（见 §7）

## 4. 判定

### 4.1 主判定（任务书 §2 的第二条分支被命中）

> **G1（TP=2 / maxlen 4096 / **默认图配置**，不省内存）的两次独立 AOT 重载不逐位一致**
> ⇒ 结论 = **"graph 全配置不可复现"**；
> **S14c 的省内存图配置（`PIECEWISE` + `1 2 4 8`）不是原因**。

一句话回答任务书的唯一问题：**"graph 运行间不可复现"是 graph 的普遍性质，不是省内存配置特有的。**
默认 `FULL_AND_PIECEWISE` 51+2 桶把 23 分钟花在 capture 上、把内存余量压到 66 MiB，
**换不来可复现性**。

### 4.2 按因子逐条结论（不是只堆数字）

| 因子 | 隔离方式 | 结论 | 证据 |
|---|---|---|---|
| **图配置（省内存 vs 默认）** | G1 vs G2：TP、maxlen、util、MBT、开关、容器、日期**全同**，只差 `cudagraph_mode` + capture sizes | **不是"会不会分叉"的原因**：两者都不可复现。**但它影响"分叉频率"**：默认配置 3 跑只落 **2 个变体**（e14 prompt-pair 一致率 **50%**、flip 1/3 对）、省内存配置 3 跑落 **3 个变体**（**22%**、flip 0/3 对）⇒ 默认图配置把胜率从 ~20% 抬到 ~50%，**没有让它变成 1.0**。 | §3.1.3、§3.2.2、§3.3 |
| **maxlen（4096 vs 262144）** | G3 未做（§7）。可用的弱证据：S14d 在 262144 上、G1/G2 在 4096 上，**分歧幅度落在同一条 1.0–1.5 nat 窄带** | **无证据表明 maxlen 起作用**；但**没有做隔离实验**，这一点必须诚实标注为"未判定" | §3.1.4、§3.2.2 |
| **TP（2 vs 4）** | G4 未做（§7）。同上：S14d(TP=4/262144) 与 G1/G2(TP=2/4096) 同档 | **无证据表明 TP 起作用**；同样**未判定** | 同上 |
| **我们自己的 sdot4 kernel** | `VLLM_DCU_USE_INT8_SDOT4=1` 在 **G1 与 G2 里都开着**，两格是"只差图配置"的严格对照 | **逻辑上被排除**：同一开关值下可复现性可以不同（或相同）到任何程度，sdot4 都无法解释 G1 与 G2 的差异。另有：任务书静态检查（无原子累加、`cfg` 固定整数）+ S14d 关 `combo_kernels` 无收益 | §3.1/§3.2、§7 |
| **H-AOT（现场编译 vs 磁盘重载）** | G2 内三跑：a=现场编译、b/c=重载 | **再次否证**："现场编译 vs 重载"（2/12）**不比**"重载 vs 重载"（4/12）更差，甚至更差 ⇒ 分歧的主因是"**任意两次运行**"，不是 AOT 来源。与 S14d 同结论 | §3.2.2 |
| **vintage / 环境（新发现，重要）** | G1a/G1b 加载的是 **S11m 那一份 artifact（md5 逐字符相同）**；再与 S11m 的 S2/S3/S4 比 | **"同一份产物 ⇒ 同一串 token"在跨容器/跨天时不成立**：`S2 vs G1b = 0/512`（G1b 落在 S11m 变体里）但 `S2 vs G1a = 403/512`。S11m 的"重载路径是确定的"只在**同一容器 + 同一时间窗**内成立 | §3.1.2 |

### 4.3 定量对比表（生产选型的依据）

| 格 | 图配置 | TP / maxlen | "两次独立 AOT 重载"逐位一致率 | 同上下文 `prompt_logprobs` | 单次启动 ready | 来源 |
|---|---|---|---|---|---|---|
| **G1** | **默认** `FULL_AND_PIECEWISE` 51+2 | 2 / 4096 | **3 跑 2 变体**：flip **1/3 对** 相同；e14 **18/36 prompt-pair（50%）**；与 S11m 合并 6 跑 ⇒ **7/15 对（47%）** | mean **1.03–1.15** / max **7.92** nat | **1542 / 1542 / 1553 s** | 本步 §3.1、§3.3 |
| **G2** | `PIECEWISE` + `[1,2,4,8]` | 2 / 4096 | **3 跑 3 变体**：flip **0/3 对** 相同；e14 **8/36 prompt-pair（22%）** | mean **1.00–1.19** / max **8.74** nat | **170–180 s** | 本步 §3.2 |
| S14d A3 vs A4 | `PIECEWISE` + `[1,2,4,8]` | 4 / 262144 | **0/12 prompt**（7 臂 21 对 252 prompt-pair 仅 2.4% 相同） | mean 1.24–1.45 / max 11.57 nat | 230 s | S14d |
| **eager（对照）** | NONE | 4 / 262144 | **12/12 逐位一致，Δ = 0.00000** | 0.0 | 160 s | S14d E5 |

**用同一把尺子看（e14 口径的 prompt-pair 一致率）**：
默认图配置 **50%** > 省内存图配置(TP=2/4096) **22%** > 省内存图配置(TP=4/262144) **2.4%**。
⇒ **"会不会分叉"是 graph 普遍的（任务书的唯一问题，答案是"普遍"）；
但"分叉的频率"确实随配置/规模变化**，默认图配置只是把抛硬币的胜率抬高，并没有变确定。
这一层次是任务书没预设的，也是本步额外给出的：**"默认图配置可复现"这个选项并不存在**。

**默认图配置的代价（任务书要求量化）**：按 S14c **C7** 的实测（TP=4/262144，与 C1 省内存配置同口径对照）——
ready **351 s → 1645 s（+1294 s = +21.6 min）**、`min MemAvailable` **0.885 GiB → 66 MiB**（@16 GiB swap），
换来的只是 decode **50.14 → 47.97 ms（−4.3%）**。本步在 TP=2/4096 上独立复现了同一结论：
**1542 s vs 170–180 s（≈9×）**。

⇒ **"用默认图配置换可复现性"这条路不存在**：多花的 21.6 min 启动与几乎归零的内存余量
**买不到任何可复现性**（§3.1 已实测不一致）。

### 4.4 因此生产该选什么

1. **要 token 流/数值可复现 ⇒ 只能 `--enforce-eager`**。S14d 实测 eager 跨引擎 **12/12 逐位一致、
   Δ = 0.00000 nat**（131.78–133.83 ms decode）。
   **注意限定**：eager 的**批组合**会确定性改变数值（S14d E3：conc=2 vs conc=1 在第 6 个 token 分叉、
   mean 1.82 / max 15.02 nat）⇒ eager 保证的是"**同请求序列**可复现"，并发调度变了数值仍会变。
2. **不选默认图配置**：它不可复现（本步实测）且启动代价 9×、内存余量 66 MiB。
3. **若接受不可复现**：可用省内存图配置（ready 170–180 s，比默认快 9×），
   但要接受**每次重启后同 prompt 的输出会变**：本步实测同上下文 `prompt_logprobs`
   **mean ≈1.0–1.2 nat、max ≈8–9 nat**、每个 prompt 有 **10–18 个位置 argmax 翻转**。
   这个幅度**不是舍入级**（1 nat 量级已与常见 top-1/top-2 间距同量级），会直接改变贪心 token 流。

### 4.5 对 S11m 那条部署配方的影响（必须让 Lead 知道）

S11m 当年给出的部署流程是"**冷启动一次预热（写 AOT 产物）→ 之后重启对外服务**，
并断言此后每次重启逐 token 可复现"。本步证明这条断言**不成立**：
G1a/G1b 正是"同 cache 的两次重启"，加载的是同一份产物（md5 与 S11m 记录逐字符相同），
**仍然差 403/512**。⇒ 该配方只能保证"**启动更快**"，**不能保证"重启不换输出"**。

## 5. 每格原始数据索引（全部在 `/mnt/game/s14e/`）

| 文件 | 内容 |
|---|---|
| `MANIFEST.md` | 逐 run 的单元格/TP/开关/maxlen/卡/cache/图配置/util/tasks/启动时刻 |
| `env-baseline.txt` | 开跑前：`date`/`pgrep`/`hy-smi`/`free -g`/`/proc/swaps`/`vm.swappiness`/ACS 12 桥原文 |
| `hostlogs/summary.tsv` | 机器可读汇总：run、TP、sw、maxlen、卡、cg_mode、cg_sizes、ready、秒数、`aot_load`、`ind_hit`、`oom_kill`、时间 |
| `hostlogs/G1a.log` `G1b.log` `G1c.log` | 三跑的引擎全量日志（含 `Directly load AOT`、两轮 capture、`init engine`） |
| `hostlogs/G1a.orch.log` … `G2c.orch.log` | 编排器日志（ready 秒数、task 起止、summary 行） |
| `hostlogs/G2.drive.log` | G2 三跑的驱动日志 + 三对对照的完整输出 |
| `flip-G1a/G1b/G2a/G2b/G2c.jsonl` | 仪器 A 原始记录（`s11l_flip_stats.py`，8 prompt × 64 token） |
| `probe-G1a/G1b/G2a/G2b/G2c.jsonl` | 仪器 B 原始记录（`s14e_client.py`，12 prompt × 128 token + 逐 prompt 位置 `prompt_logprobs` top-20） |
| `cmp-*.csv` / `cmp-*.txt` | 逐 prompt 逐位对照（`s14e_compare.py` 输出 + csv） |
| `g1/` `g2/` | 本步的 `VLLM_CACHE_ROOT`（`g1` 是 S11m `one` 的完整复制；`g2` 是 G2 现场编译产物） |
| `artifact-md5-after-G1a.txt` | 证明产物未被重写，且等于 S11m 归档里的 md5 |
| `s11m-baseline-cluster.txt` | S11m S2/S3/S4 三份产物的基线簇核对（0/512、ALL_IDENTICAL=True） |
| `cgroup-events-*-before/after.txt` | 每次启动前后的 cgroup `memory.events`（`oom_kill` 未增长） |
| `/mnt/game/s11m/flip-S2/S3/S4sw1hit.jsonl` | **S11m 的原始产物（只读引用，未改动）** |
| `/mnt/game/s14/flip-P2.jsonl`、`flip-P2e.jsonl`、`flip-P4e.jsonl` | S14 的同类 flip 电池（只读引用，用于跨格变体对照） |
| `/mnt/game/s14d/probe-*.jsonl` | S14d 的 e14 电池（只读引用，用于量级对照） |

脚本（本步新增，均在 `tools/dcu/s14e_*`，未改任何既有脚本）：

| 脚本 | 作用 |
|---|---|
| `s14e_client.py` | 12 prompt 逐位 + 同上下文 `prompt_logprobs` 记录器（= `s14d_client.py` 的复制） |
| `s14e_compare.py` | 双仪器逐位对照器（= `s14d_compare.py` 的复制） |
| `s14e_matrix.py` | N 臂两两汇总（= `s14d_matrix.py` 的复制） |
| `s14e_run_one.sh` | 单 run 编排：启动→等 ready→按 `S14E_TASKS` 跑 `flip`/`e14`→停引擎→汇总 |
| `s14e_drive.sh` | 单格 N 跑串行驱动 + 自动两两对照 |

## 6. 收尾与还原

| 项 | 开跑前 | 收尾时 | 是否还原 |
|---|---|---|---|
| `/proc/swaps` | `/swap.img file 4194300 已用 519704`（唯一 swap） | `/swap.img file 4194300 已用 530972`（**无新增 swapfile**） | ✅ 本步未加过任何 swapfile |
| `/proc/sys/vm/swappiness` | `60` | `60` | ✅ 未改过 |
| ACS（父链 12 个桥 `ACS_Ctl`） | 12 × `0x0000` | **12 × `0x0000`**（`/mnt/game/s14e/acs-final.txt`） | ✅ 保持部署态，未切过 |
| 生产容器 `qwen38-vllm` | 未 start/kill/exec/commit/inspect | 同上 | ✅ |
| DTK 库 / `z100-build*` | 未改 | 同上 | ✅ |
| 自编译 HIP 可执行文件 | 未执行任何自编译独立可执行文件 | — | ✅ |
| 卡位 | `HIP_VISIBLE_DEVICES` 全程显式（`0,1`），无 `unset`；同一时刻单一引擎（G1→G2→G1c 严格串行） | 同上 | ✅ |
| 残留进程 | — | **存活 VLLM 进程 0**（`ps -eo stat,comm \| grep VLLM \| grep -v Z \| wc -l` = 0）；四卡 VRAM **全 0** | ✅ |

> 诚实标注（容器既有现象，非本步引入）：`z100-s14` 内累积了 **89 个 zombie（Z 态）进程**，
> 其中 30 个的 comm 是 `VLLM`⇒ 容器 PID 1 不回收孤儿子进程。它们**不占 VRAM、不占内存**（`hy-smi` 四卡 0 MiB、
> `free -g` available 27 GiB），对本步的数值测量无影响；但后续若继续在 `z100-s14` 里跑几十次，
> 需要留意 PID 上限。

## 7. 未做项与原因

| 格 | 定义 | 状态 | 原因（可核查） | 已有同口径证据 |
|---|---|---|---|---|
| **G3** | TP=2 / maxlen 262144 / 默认图配置 | ❌ **未做** | 默认图配置**每次启动实测 1542 s（≈25.7 min）**（§3.1.1），3 跑 ≈80 min；本步 120 min 预算里 G1 的两跑就占 52 min。 | S14 的 **P2** 臂（TP=2/maxlen 262144/默认图配置）跑过同一个 `s11l_flip_stats.py` 电池，但**只有 1 次启动**，无法做重载对照。可用的跨格事实：`S2sw1hit vs P2 = 410/512`、`G1a vs P2 = 382/512`（§3.1.2）⇒ P2 也不在 S11m 的变体里，但这不是复现性证据。 |
| **G4** | TP=4 / maxlen 262144 / 默认图配置 | ❌ **未做** | 同 G3（S14c **C7** 实测 ready **1645 s**、`min MemAvailable` 只剩 **66 MiB**，还要挂 16 GiB swap）⇒ 单格 ≈85 min + 内存风险。 | S14c C7 单次启动的 ready/内存/decode 数据（§4 用于量化"默认图配置的代价"）。 |
| G5 | TP=2 / 4096 / PIECEWISE+1 2 4 8 + `VLLM_DCU_USE_INT8_SDOT4=0` | ❌ **未做** | 预算上被 **G1c** 挤掉：任务书要求"G1 不一致则加到 3 次"（G1c 实测 1553 s 启动，05:25Z 才 ready），而 120 min 预算在 G1(2 跑)+G2(3 跑)+G1c 之后已用尽（实际总墙钟 **≈122 min**）。 | 逻辑上已不需要：`VLLM_DCU_USE_INT8_SDOT4=1` 在 G1 与 G2 里**都开着**，两格只差图配置 ⇒ 差异不可能由 sdot4 解释。另有任务书静态检查（无原子累加、`cfg` 固定整数）与 S14d 的 `combo_kernels` 无收益（1/36）。 |

**为什么 G5 的逻辑优先级低于 G2/G1c**：`VLLM_DCU_USE_INT8_SDOT4=1` 在 **G1 与 G2 里都开着**，
两格**只差图配置**。因此只要 G1 可复现而 G2 不可复现（或反之），
差异就**不可能**由 sdot4 解释。G5 只有在"G1 与 G2 表现相同"时才是必要的排除项。
任务书本身也把 G5 写成括号项（`G1 → G2 → G3 → (G5) → (G4)`）。
另有两条独立佐证：任务书的静态检查（sdot4 **无原子累加**、`cfg` 是固定整数参数）与
S14d 的实测（关掉 `combo_kernels`/`benchmark_combo_kernel` 无收益，1/36）。
